//! What happens when Echo comes back after a crash, a power cut, or a forced
//! quit in the middle of a meeting.
//!
//! The rule (DESIGN §3 Crash recovery, mantra 3): **committed audio is never
//! lost.** The meeting row was written before capture started and every chunk
//! was journalled as it became durable, so on relaunch there is always enough on
//! disk to finish the job.
//!
//! At launch [`scan`] looks for meetings that never ended:
//!
//! * committed audio on disk → mark it interrupted, put capture into
//!   `Recovering`, and offer the person a choice (finish, or throw it away).
//! * nothing on disk → there is nothing to offer, so the empty row is put out of
//!   the way instead of asking a question with no useful answer.
//!
//! Finishing queues exactly the same jobs a normal stop does, which is what makes
//! "transcription resumes from the last committed audio offset" true for free.

use std::path::PathBuf;
use std::sync::Arc;

use crate::db::repo;
use crate::events::{NoticeLevel, NoticePayload, RecoveryAvailablePayload};
use crate::session::ports::{EventSink, UiEvent};
use crate::session::{CaptureEvent, Inner, SessionError};
use crate::types::{Id, JobKind, Meeting, MeetingStatus, RecoveryAction};

/// Look for unfinished meetings and offer them. Idempotent: running it twice
/// reports the same list and changes nothing further.
pub(crate) async fn scan(inner: &Arc<Inner>) -> Result<Vec<Id>, SessionError> {
    let candidates = repo::list_interrupted_meetings(&inner.db).await?;
    let mut recoverable: Vec<Id> = Vec::new();

    for meeting in candidates {
        // Never touch the meeting that is being recorded right now.
        let live_now = {
            let state = inner.live.lock().expect("capture state lock");
            state.meeting_id.as_deref() == Some(meeting.id.as_str()) && super::is_live(state.state)
        };
        if live_now {
            continue;
        }

        // The files on disk come first. A crash between fsyncing a chunk and
        // writing its row — or a database write that failed while the disk was
        // full — leaves complete audio that the journal knows nothing about, and
        // deciding "there was no audio" from the journal alone would bin a real
        // meeting (mantra 3: the audio is the truth, the journal is bookkeeping).
        let rejournalled = rejournal_from_disk(inner, &meeting).await;
        if rejournalled > 0 {
            tracing::info!(
                meeting = %meeting.id,
                pieces = rejournalled,
                "found saved audio the interrupted meeting had not recorded yet"
            );
        }

        let committed = super::jobs::committed_end_ms(&inner.db, &meeting.id).await?;
        if committed <= 0 {
            // A row with no audio behind it: nothing to recover, nothing worth
            // asking about.
            tracing::info!(meeting = %meeting.id, "an empty interrupted meeting was tidied away");
            let _ = repo::set_meeting_status(&inner.db, &meeting.id, MeetingStatus::Failed).await;
            let _ = repo::soft_delete_meeting(&inner.db, &meeting.id).await;
            continue;
        }

        if !matches!(meeting.status, MeetingStatus::Interrupted) {
            repo::set_meeting_status(&inner.db, &meeting.id, MeetingStatus::Interrupted).await?;
        }
        if meeting.duration_ms < committed {
            let _ = repo::set_meeting_duration(&inner.db, &meeting.id, committed).await;
        }
        recoverable.push(meeting.id);
    }

    if recoverable.is_empty() {
        return Ok(recoverable);
    }

    {
        let mut state = inner.live.lock().expect("capture state lock");
        state.recovering = recoverable.clone();
    }
    inner.transition(&CaptureEvent::InterruptedFound);
    inner.emit_state();
    inner
        .ports
        .events
        .emit(UiEvent::RecoveryAvailable(RecoveryAvailablePayload {
            meeting_ids: recoverable.clone(),
        }));
    inner.notice(NoticePayload {
        level: NoticeLevel::Info,
        message: "Echo found a meeting it didn't get to finish. You can pick up where it left off."
            .into(),
        persistent: true,
        meeting_id: recoverable.first().cloned(),
        tag: Some("recoveryAvailable".into()),
    });

    tracing::info!(count = recoverable.len(), "unfinished meetings found");
    Ok(recoverable)
}

/// Where this meeting's audio was written. The folder on the row wins, but only
/// when it names this meeting — a row that recorded the storage root by accident
/// must not send a scan at the whole root.
fn audio_dir_of(inner: &Arc<Inner>, meeting: &Meeting) -> PathBuf {
    let recorded = PathBuf::from(&meeting.audio_dir);
    let names_this_meeting = recorded
        .file_name()
        .map(|name| name == std::ffi::OsStr::new(meeting.id.as_str()))
        .unwrap_or(false);
    if names_this_meeting {
        recorded
    } else {
        inner.paths.meeting_dir(&meeting.id)
    }
}

/// Give every chunk file on disk a journal row, so recovery can see it.
///
/// Returns how many rows were added. Chunks already journalled are left exactly
/// as they are: their recorded offsets are better than anything that can be
/// worked out from file lengths. A file that cannot be read is not trusted —
/// [`crate::audio::recover_chunks`] stops at the first one — because a
/// half-written chunk with a plausible length would corrupt the timeline.
async fn rejournal_from_disk(inner: &Arc<Inner>, meeting: &Meeting) -> u32 {
    let dir = audio_dir_of(inner, meeting);
    let found = match crate::audio::recover_chunks(&dir).await {
        Ok(found) => found,
        Err(error) => {
            tracing::warn!(%error, "could not look at the recording folder");
            return 0;
        }
    };
    if found.is_empty() {
        return 0;
    }
    let known = match repo::list_chunks(&inner.db, &meeting.id, None).await {
        Ok(known) => known,
        Err(error) => {
            tracing::warn!(%error, "could not read what this recording already recorded");
            return 0;
        }
    };

    let mut added = 0;
    for channel in [crate::types::Channel::Mic, crate::types::Channel::System] {
        // Walk the channel in order, keeping a running position: a chunk whose
        // row exists sets the clock, and one without a row is placed where the
        // audio before it ends.
        let mut cursor = 0i64;
        for chunk in found.iter().filter(|c| c.channel == channel) {
            if let Some(row) = known
                .iter()
                .find(|row| row.channel == channel && row.seq == chunk.seq as i64)
            {
                cursor = cursor.max(row.t_end_ms);
                continue;
            }
            let t_start = cursor;
            let t_end = cursor + chunk.duration_ms();
            let path = chunk.path.to_string_lossy().into_owned();
            match repo::insert_chunk(
                &inner.db,
                &meeting.id,
                channel,
                chunk.seq as i64,
                &path,
                t_start,
                t_end,
            )
            .await
            {
                Ok(id) => {
                    if let Err(error) = repo::commit_chunk(&inner.db, &id, t_end).await {
                        tracing::warn!(%error, "could not mark recovered audio as saved");
                        continue;
                    }
                    added += 1;
                    cursor = t_end;
                }
                Err(error) => {
                    tracing::warn!(%error, "could not record a piece of recovered audio");
                }
            }
        }
    }
    added
}

/// Act on the person's choice. Idempotent: a second answer for the same meeting
/// is accepted and does nothing.
pub(crate) async fn resolve(
    inner: &Arc<Inner>,
    meeting_id: &str,
    action: RecoveryAction,
) -> Result<(), SessionError> {
    let Some(meeting) = repo::get_meeting(&inner.db, meeting_id).await? else {
        return Err(SessionError::Db(crate::db::DbError::NotFound(format!(
            "meeting {meeting_id}"
        ))));
    };

    match action {
        RecoveryAction::Finish => {
            if !matches!(meeting.status, MeetingStatus::Processing) {
                repo::set_meeting_status(&inner.db, meeting_id, MeetingStatus::Processing).await?;
            }
            queue_finalization(inner, meeting_id).await?;
            tracing::info!(meeting = %meeting_id, "finishing an interrupted meeting");
        }
        RecoveryAction::Discard => {
            // Out of the person's way, but the audio itself stays on disk until
            // they delete the meeting for real (mantra 3).
            let _ = repo::set_meeting_status(&inner.db, meeting_id, MeetingStatus::Failed).await;
            repo::soft_delete_meeting(&inner.db, meeting_id).await?;
            tracing::info!(meeting = %meeting_id, "an interrupted meeting was set aside");
        }
    }

    inner.ports.events.emit(UiEvent::MeetingUpdated(
        crate::events::MeetingUpdatedPayload {
            meeting_id: meeting_id.to_string(),
            status: match action {
                RecoveryAction::Finish => MeetingStatus::Processing,
                RecoveryAction::Discard => MeetingStatus::Failed,
            },
            title: Some(meeting.title),
            duration_ms: meeting.duration_ms,
            deleted: matches!(action, RecoveryAction::Discard),
        },
    ));

    let settled = {
        let mut state = inner.live.lock().expect("capture state lock");
        state.recovering.retain(|id| id != meeting_id);
        state.recovering.is_empty()
    };
    if settled {
        inner.transition(&CaptureEvent::RecoveryResolved);
        inner.emit_state();
    }
    Ok(())
}

/// The jobs that turn committed audio into a finished meeting. The same list
/// runs after a normal stop and after a recovery, in this order.
pub(crate) async fn queue_finalization(
    inner: &Arc<Inner>,
    meeting_id: &str,
) -> Result<(), SessionError> {
    let auto_summarize = crate::settings::load(&inner.db)
        .await
        .map(|s| s.auto_summarize)
        .unwrap_or(false);

    let mut wanted = vec![
        JobKind::TranscribeCatchup,
        JobKind::Diarize,
        JobKind::Mixdown,
    ];
    if auto_summarize {
        wanted.push(JobKind::Summarize);
    }
    for kind in wanted {
        inner.jobs.queue(Some(meeting_id), kind).await?;
    }
    Ok(())
}
