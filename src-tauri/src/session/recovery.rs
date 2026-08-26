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
//!
//! The other half of this module is the same question asked at the end of every
//! recording: is there anything here worth keeping? [`meeting_has_content`] is
//! the one answer both the stop path and the finish-after-recovery path use, and
//! a meeting that fails it is deleted outright rather than kept as a husk.

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

    let mut discarded_as_empty = false;
    match action {
        RecoveryAction::Finish => {
            if !matches!(meeting.status, MeetingStatus::Processing) {
                repo::set_meeting_status(&inner.db, meeting_id, MeetingStatus::Processing).await?;
            }
            // Same rule as a normal stop: there is nothing to finish when there
            // is nothing worth keeping, and "finish" must not conjure a husk.
            match queue_finalization(inner, meeting_id).await? {
                Finalized::Queued => {
                    tracing::info!(meeting = %meeting_id, "finishing an interrupted meeting")
                }
                Finalized::Discarded => discarded_as_empty = true,
            }
        }
        RecoveryAction::Discard => {
            // Out of the person's way, but the audio itself stays on disk until
            // they delete the meeting for real (mantra 3).
            let _ = repo::set_meeting_status(&inner.db, meeting_id, MeetingStatus::Failed).await;
            repo::soft_delete_meeting(&inner.db, meeting_id).await?;
            tracing::info!(meeting = %meeting_id, "an interrupted meeting was set aside");
        }
    }

    // An empty meeting has already announced itself as gone.
    if !discarded_as_empty {
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
    }

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

/// What happened to a meeting that just ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Finalized {
    /// The finishing work is queued.
    Queued,
    /// There was nothing worth keeping; the meeting and its files are gone.
    Discarded,
}

/// Is there anything in this meeting worth keeping?
///
/// The predicate, deliberately generous, because throwing away someone's meeting
/// is unforgivable and keeping an empty row is merely untidy. A meeting is kept
/// when **either**:
///
/// * it has at least one finished line of transcript — words are the product; or
/// * at least [`super::MIN_KEPT_MS`] of audio is committed on disk. Audio counts
///   as content on its own (mantra 3): a meeting whose transcription simply has
///   not run yet is *not* empty, and the catch-up job will turn that audio into
///   words later. Only committed chunks count, because an uncommitted one may be
///   a half-written file; but if the journal comes up empty the folder itself
///   gets one last look, so a row that was never written cannot cost someone
///   audio that is right there on disk.
///
/// What is left is a husk: no words, and under three seconds of sound nobody
/// will ever ask for — the shape of pressing Start and Stop while looking for
/// the right button.
pub(crate) async fn meeting_has_content(
    inner: &Arc<Inner>,
    meeting_id: &str,
) -> Result<bool, SessionError> {
    if repo::count_final_segments(&inner.db, meeting_id).await? > 0 {
        return Ok(true);
    }
    let committed = super::jobs::committed_end_ms(&inner.db, meeting_id).await?;
    if committed >= super::MIN_KEPT_MS {
        return Ok(true);
    }

    // Last look before anything is deleted: the journal is bookkeeping, the files
    // are the truth (mantra 3). A pipeline that ran out of time writing its rows,
    // or a database write that failed while the disk was full, must never cost
    // somebody a meeting that is sitting there on disk.
    let dir = match repo::get_meeting(&inner.db, meeting_id).await? {
        Some(meeting) => audio_dir_of(inner, &meeting),
        None => inner.paths.meeting_dir(meeting_id),
    };
    let on_disk: i64 = match crate::audio::recover_chunks(&dir).await {
        Ok(found) => found.iter().map(|chunk| chunk.duration_ms()).sum(),
        Err(error) => {
            // Cannot tell: assume there is something rather than delete blind.
            tracing::warn!(%error, "could not look at the recording folder before tidying up");
            return Ok(true);
        }
    };
    if on_disk >= super::MIN_KEPT_MS {
        tracing::info!(
            meeting = %meeting_id,
            on_disk,
            "the journal looked empty but there is audio on disk; keeping it"
        );
        return Ok(true);
    }
    Ok(false)
}

/// Delete a meeting nobody will ever want, with its files, and say so once.
///
/// Nothing is queued for it and anything already queued is cancelled, so no
/// catch-up, speaker pass or recap runs against a row that is on its way out.
async fn discard_empty_meeting(inner: &Arc<Inner>, meeting_id: &str) {
    discard_empty_meeting_inner(inner, meeting_id, true).await;
}

/// One pass over the finished meetings already in the library, discarding any
/// husk the same test at stop-time would have discarded. Exists for records
/// created before that test did, so an old empty row cannot sit in the list
/// looking like a bug. Runs once at launch; quiet on purpose — a toast per
/// stale husk would greet the person with noise about meetings they never had.
pub(crate) async fn sweep_husks(inner: &Arc<Inner>) -> u64 {
    let candidates = match repo::finished_meeting_ids(&inner.db).await {
        Ok(ids) => ids,
        Err(error) => {
            tracing::warn!(%error, "could not look for empty meetings to tidy up");
            return 0;
        }
    };
    let mut swept = 0;
    for meeting_id in candidates {
        match meeting_has_content(inner, &meeting_id).await {
            Ok(false) => {
                discard_empty_meeting_inner(inner, &meeting_id, false).await;
                swept += 1;
            }
            Ok(true) => {}
            Err(error) => {
                // Cannot tell: keep it. Deleting blind is the one wrong answer.
                tracing::warn!(%error, meeting = %meeting_id, "left a meeting alone: could not check it");
            }
        }
    }
    if swept > 0 {
        tracing::info!(swept, "tidied up empty meetings from before");
    }
    swept
}

async fn discard_empty_meeting_inner(inner: &Arc<Inner>, meeting_id: &str, announce: bool) {
    // Stop the work first: a job holding this id must not carry on reading files
    // that are about to disappear.
    inner.ports.asr.forget_meeting(meeting_id);
    inner.cancel_jobs_for(meeting_id).await;

    let (dir, ran_for_ms) = match repo::get_meeting(&inner.db, meeting_id).await {
        Ok(Some(meeting)) => (audio_dir_of(inner, &meeting), meeting.duration_ms),
        _ => (inner.paths.meeting_dir(meeting_id), 0),
    };
    match repo::delete_meeting(&inner.db, meeting_id).await {
        Ok(files) => {
            for file in files {
                let _ = std::fs::remove_file(file);
            }
        }
        Err(error) => {
            // Could not remove it: leave it out of the way rather than in the
            // list, and never fail a stop over tidying up.
            tracing::warn!(%error, "could not remove an empty meeting");
            let _ = repo::soft_delete_meeting(&inner.db, meeting_id).await;
        }
    }
    // Only ever a folder named after this meeting, never the folder the person
    // chose to keep recordings in.
    if dir
        .file_name()
        .map(|name| name == std::ffi::OsStr::new(meeting_id))
        .unwrap_or(false)
    {
        let _ = std::fs::remove_dir_all(&dir);
    }

    tracing::info!(meeting = %meeting_id, ran_for_ms, "nothing was recorded, so nothing was kept");
    inner.ports.events.emit(UiEvent::MeetingUpdated(
        crate::events::MeetingUpdatedPayload {
            meeting_id: meeting_id.to_string(),
            status: MeetingStatus::Failed,
            title: None,
            duration_ms: 0,
            deleted: true,
        },
    ));
    if announce {
        inner.notice(ending_notice(ran_for_ms));
    }
}

/// How long a recording has to have run before "too short to keep" is a lie
/// about it.
///
/// Half a minute, and the two numbers either side of it are what pick it.
/// [`super::MIN_KEPT_MS`] is three seconds — the shape of pressing Start and
/// Stop while looking for the right button, and the only shape the old sentence
/// was ever written for. `SILENT_CHANNEL_GRACE` in `crate::audio` is ten seconds —
/// how long a source that opened may deliver nothing before the watchdog says
/// so on screen. Past both, a person has watched a timer run, watched a banner
/// they may already have been shown, and waited: they know how long that
/// recording was, and being told it was too short reads as Echo not knowing
/// what it just did. Under half a minute, the quieter sentence is still true
/// enough and is the kinder one — nobody wants a warning about the half-second
/// they mis-clicked.
const HEARD_NOTHING_AFTER_MS: i64 = 30_000;

/// The last thing a person hears about a recording that is being deleted.
///
/// Two endings, because there are two things that happen and only one of them
/// used to be said. Deleting the meeting is right in both — see
/// [`discard_empty_meeting`] — but a recording that ran for three quarters of an
/// hour and saved nothing is not "too short", and saying so was the closing lie
/// of the incident of 2026-08-21: a silent failure, then a false explanation,
/// then the folder gone.
///
/// The long ending is a `Warning` and persistent on purpose. Nothing was kept,
/// so there is no meeting left to open and look at afterwards — this sentence is
/// the entire record of what happened, and a toast that fades in four seconds is
/// only a quieter kind of silence (the rule this branch exists for).
fn ending_notice(ran_for_ms: i64) -> NoticePayload {
    if ran_for_ms >= HEARD_NOTHING_AFTER_MS {
        NoticePayload {
            level: NoticeLevel::Warning,
            message: "Echo couldn't hear anything for that whole recording, so there was nothing to keep. Check that Echo is allowed to use your microphone and to record this computer's audio.".into(),
            persistent: true,
            meeting_id: None,
            tag: Some("heardNothing".into()),
        }
    } else {
        NoticePayload {
            level: NoticeLevel::Info,
            message: "That one was too short to keep, so Echo let it go.".into(),
            persistent: false,
            meeting_id: None,
            tag: Some("nothingToKeep".into()),
        }
    }
}

/// The jobs that turn committed audio into a finished meeting. The same list
/// runs after a normal stop and after a recovery, in this order.
///
/// A meeting with nothing worth keeping never gets a queue: it is deleted
/// instead ([`meeting_has_content`]).
pub(crate) async fn queue_finalization(
    inner: &Arc<Inner>,
    meeting_id: &str,
) -> Result<Finalized, SessionError> {
    if !meeting_has_content(inner, meeting_id).await? {
        discard_empty_meeting(inner, meeting_id).await;
        return Ok(Finalized::Discarded);
    }

    // On by default (mantra 4: the happy path is Start → the recap appears).
    // Someone who turned it off has `false` stored, and that wins.
    let auto_summarize = crate::settings::load(&inner.db)
        .await
        .map(|s| s.auto_summarize)
        .unwrap_or(crate::settings::DEFAULT_AUTO_SUMMARIZE);

    let mut wanted = vec![
        JobKind::TranscribeCatchup,
        JobKind::Diarize,
        JobKind::Mixdown,
    ];
    if auto_summarize {
        // Queued now, run last: `repo::next_queued_job` hands work out
        // catch-up → speakers → playback → recap, so the recap is written from
        // the finished transcript with its speakers already sorted out.
        wanted.push(JobKind::Summarize);
    }
    for kind in wanted {
        inner.jobs.queue(Some(meeting_id), kind).await?;
    }
    Ok(Finalized::Queued)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_mis_click_and_a_lost_meeting_do_not_get_the_same_ending() {
        let mis_click = ending_notice(1_500);
        assert_eq!(mis_click.tag.as_deref(), Some("nothingToKeep"));
        assert!(!mis_click.persistent, "a mis-click nagged");
        assert_eq!(mis_click.level, NoticeLevel::Info);

        let lost = ending_notice(45 * 60_000);
        assert_eq!(lost.tag.as_deref(), Some("heardNothing"));
        assert!(lost.persistent, "the only record of it faded on its own");
        assert_eq!(lost.level, NoticeLevel::Warning);
        assert!(
            !lost.message.contains("too short"),
            "{} calls three quarters of an hour short",
            lost.message
        );
        assert!(
            lost.message.contains("couldn't hear anything"),
            "{} does not say what actually happened",
            lost.message
        );
    }

    /// The boundary is a decision, so it is asserted rather than left to drift.
    #[test]
    fn the_honest_ending_starts_at_half_a_minute() {
        assert_eq!(
            ending_notice(HEARD_NOTHING_AFTER_MS - 1).tag.as_deref(),
            Some("nothingToKeep")
        );
        assert_eq!(
            ending_notice(HEARD_NOTHING_AFTER_MS).tag.as_deref(),
            Some("heardNothing")
        );
        const {
            assert!(
                HEARD_NOTHING_AFTER_MS > super::super::MIN_KEPT_MS,
                "the honest ending starts before the meeting is even worth keeping"
            )
        };
    }

    /// Zero jargon, and no shouting at somebody who has just lost a recording.
    #[test]
    fn both_endings_are_said_in_the_app_s_own_voice() {
        for ran_for_ms in [0, 45 * 60_000] {
            let message = ending_notice(ran_for_ms).message;
            assert!(!message.contains('!'), "{message} shouts");
            for jargon in ["buffer", "stream", "chunk", "OSStatus", "segment"] {
                assert!(
                    !message.contains(jargon),
                    "{message} leaks {jargon} to the person"
                );
            }
        }
    }
}
