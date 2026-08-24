//! "There were four of us": the person correcting how many people Echo heard.
//!
//! The speaker pass counts the voices itself and is right often enough to be the
//! default, wrong often enough to need a way out. The way out is not a threshold
//! — nobody outside a lab could reason about one (mantra 2) — it is a number, and
//! a number a human states about a conversation they were in is better evidence
//! than any distance measured on somebody else's corpus. So this stores the
//! number and queues the pass again, cut to that many voices — and, when the
//! recording holds fewer voices than that which can be told apart, says so
//! plainly instead of handing back speaker rows with no words on them
//! ([`crate::diarize::cluster::ForcedOutcome`]).
//!
//! What it deliberately does *not* touch:
//!
//! * **The recording.** Never (mantra 3).
//! * **The transcript text.** The words are the words; only who they are
//!   attributed to is the pass's output, so only the speaker pass is re-queued —
//!   no catch-up, no mixdown, and no recap, which is the person's to keep.
//! * **Anything, while a recording of this meeting is live.** Recording has
//!   absolute resource priority (DESIGN §3).
//! * **Anything, while the pass or a listen-again is already working on this
//!   meeting.** Changing the target from under a running pass would leave the
//!   rows it is halfway through writing keyed to the old count.
//!
//! The number stored is the total, counting whoever was at this computer — see
//! [`crate::diarize::remote_target`], which owns the arithmetic that turns it
//! into a number of remote voices, and the module docs of [`crate::diarize`] for
//! what happens to renames when the count changes.

use std::sync::Arc;

use crate::db::repo;
use crate::events::SpeakersUpdatedPayload;
use crate::session::ports::{EventSink, UiEvent};
use crate::session::{Inner, SessionError};
use crate::types::{Id, JobKind, JobQuery};

/// What a request to set the count did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpeakerCountSet {
    /// Stored, and the speaker pass is queued again. Carries the job, which is
    /// what progress arrives against.
    Requeued(Id),
    /// The meeting already had this answer and the pass is already working on
    /// it, so this changed nothing. The second click is the first one (DESIGN §3:
    /// every entry point is idempotent).
    Unchanged,
}

/// Is the speaker pass — or a listen again, which ends in one — already working
/// on this meeting?
///
/// Both kinds write speaker rows for the meeting, so both are reasons to say
/// "not yet" rather than to re-target a pass mid-flight.
async fn work_in_flight(inner: &Arc<Inner>, meeting_id: &str) -> bool {
    for kind in [JobKind::Diarize, JobKind::TranscribeCatchup] {
        let active = repo::list_jobs(
            &inner.db,
            &JobQuery {
                meeting_id: Some(meeting_id.to_string()),
                kind: Some(kind),
                active_only: Some(true),
                ..Default::default()
            },
        )
        .await
        .unwrap_or_default();
        if !active.is_empty() {
            return true;
        }
    }
    false
}

/// Is this meeting the one being recorded right now?
fn is_being_recorded(inner: &Arc<Inner>, meeting_id: &str) -> bool {
    let live = inner.live.lock().expect("capture state lock");
    let recording =
        super::is_live(live.state) || matches!(live.state, crate::types::CaptureState::Starting);
    recording && live.meeting_id.as_deref() == Some(meeting_id)
}

/// Store the count and queue the pass. `None` clears back to automatic.
///
/// Callers hold the session command lock, so this cannot interleave with a start
/// or a stop.
pub(crate) async fn run(
    inner: &Arc<Inner>,
    meeting_id: &str,
    count: Option<u32>,
) -> Result<SpeakerCountSet, SessionError> {
    if repo::get_meeting(&inner.db, meeting_id).await?.is_none() {
        return Err(SessionError::Db(crate::db::DbError::NotFound(format!(
            "meeting {meeting_id}"
        ))));
    }
    // A guard rail, not the validation: the command turned an out-of-range
    // request down with a sentence before it got here. This is what keeps a
    // number from some other caller out of the clustering loop.
    let count = count.map(crate::diarize::clamp_people);

    // A meeting being recorded owns the machine, and re-cutting the voices of a
    // meeting that has not finished happening is not a thing to queue behind it.
    if is_being_recorded(inner, meeting_id) {
        return Err(SessionError::AlreadyRecording);
    }

    let current = repo::speaker_count_override(&inner.db, meeting_id).await?;
    if work_in_flight(inner, meeting_id).await {
        // Same answer, already being worked on: this is the double-click, and the
        // right response to it is silence, not an error.
        if current == count {
            tracing::debug!(
                meeting = %meeting_id,
                "people count unchanged and already being worked out"
            );
            return Ok(SpeakerCountSet::Unchanged);
        }
        return Err(SessionError::MeetingBusy);
    }

    repo::set_speaker_count_override(&inner.db, meeting_id, count).await?;
    tracing::info!(
        meeting = %meeting_id,
        people = ?count,
        "working out who said what again, for this many people"
    );

    // Say the new number out loud straight away, on the same event the pass will
    // use when it finishes. The chips are still the old ones for now, and that is
    // the honest state of things: the person has said how many people were here
    // and Echo is working out which of them said what. Without this the number on
    // screen would stay the old one for as long as the pass takes.
    let (people_count, people_count_is_override) =
        repo::people_count(&inner.db, meeting_id).await?;
    inner
        .ports
        .events
        .emit(UiEvent::SpeakersUpdated(SpeakersUpdatedPayload {
            meeting_id: meeting_id.to_string(),
            speakers: repo::list_speakers(&inner.db, meeting_id).await?,
            people_count,
            people_count_is_override,
            // The person has just said how many people were here and the pass
            // has not run yet. Anything Echo could say about how many voices it
            // can hear belongs to the previous cut, so it says nothing.
            voices_found: None,
            alternative_count: None,
        }));

    // Only the speaker pass. The words are untouched, so nothing needs
    // transcribing again; the audio is untouched, so the playback file is still
    // right; and the recap is the person's, to rewrite when they choose.
    // `queue` is deduplicated per meeting and kind, so this is safe to reach
    // twice.
    let job = inner.jobs.queue(Some(meeting_id), JobKind::Diarize).await?;
    Ok(SpeakerCountSet::Requeued(job.id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::repo;
    use crate::diarize::{MAX_PEOPLE, SELF_CLUSTER_KEY};
    use crate::session::mock::Harness;
    use crate::types::{Channel, JobStatus, MeetingStatus, SegmentDraft};

    /// A finished meeting with a transcript and the speakers a pass left behind.
    async fn a_finished_meeting(h: &Harness) -> Id {
        let meeting = repo::create_meeting(&h.db, "Weekly sync", "/tmp/echo-people", None)
            .await
            .unwrap();
        let chunk = repo::insert_chunk(
            &h.db,
            &meeting.id,
            Channel::System,
            0,
            "/tmp/echo-people/system-0.flac",
            0,
            60_000,
        )
        .await
        .unwrap();
        repo::commit_chunk(&h.db, &chunk, 60_000).await.unwrap();
        repo::finish_meeting(
            &h.db,
            &meeting.id,
            60_000,
            Some("en"),
            None,
            MeetingStatus::Complete,
        )
        .await
        .unwrap();
        repo::insert_segments(
            &h.db,
            &[SegmentDraft {
                meeting_id: meeting.id.clone(),
                t_start_ms: 0,
                t_end_ms: 4_000,
                channel: Channel::System,
                text: "we ship on Friday".into(),
                revision: 1,
                is_final: true,
                ..Default::default()
            }],
        )
        .await
        .unwrap();
        repo::upsert_speaker(&h.db, &meeting.id, SELF_CLUSTER_KEY, "You", true)
            .await
            .unwrap();
        repo::upsert_speaker(&h.db, &meeting.id, "speaker-01", "Speaker 1", false)
            .await
            .unwrap();
        meeting.id
    }

    #[tokio::test]
    async fn a_correction_is_stored_and_only_the_speaker_pass_is_queued() {
        let h = Harness::new().await;
        let meeting_id = a_finished_meeting(&h).await;

        let outcome = h
            .session
            .set_speaker_count(&meeting_id, Some(4))
            .await
            .unwrap();
        assert!(
            matches!(outcome, SpeakerCountSet::Requeued(_)),
            "{outcome:?}"
        );

        assert_eq!(
            repo::speaker_count_override(&h.db, &meeting_id)
                .await
                .unwrap(),
            Some(4)
        );
        assert_eq!(
            repo::people_count(&h.db, &meeting_id).await.unwrap(),
            (4, true),
            "the number the person typed is the number the UI shows"
        );
        assert_eq!(
            h.queued_kinds(&meeting_id).await,
            vec![JobKind::Diarize],
            "the words are untouched, so nothing else needs doing"
        );

        // Said out loud straight away, on the event the pass will use again when
        // it finishes — so the number on screen is the person's from the moment
        // they set it, not once the pass is done.
        let announced = h.events.speaker_updates();
        assert_eq!(announced.len(), 1);
        assert_eq!(announced[0].meeting_id, meeting_id);
        assert_eq!(announced[0].people_count, 4);
        assert!(announced[0].people_count_is_override);
        assert_eq!(
            announced[0].speakers.len(),
            2,
            "the chips are still the ones the last pass left; the pass will fix them"
        );
    }

    #[tokio::test]
    async fn clearing_it_goes_back_to_echos_own_count() {
        let h = Harness::new().await;
        let meeting_id = a_finished_meeting(&h).await;

        h.session
            .set_speaker_count(&meeting_id, Some(6))
            .await
            .unwrap();
        // The pass has to have finished, or the second request is refused as busy.
        for job in repo::list_jobs(
            &h.db,
            &JobQuery {
                meeting_id: Some(meeting_id.clone()),
                ..Default::default()
            },
        )
        .await
        .unwrap()
        {
            repo::set_job_status(&h.db, &job.id, JobStatus::Done, None)
                .await
                .unwrap();
        }

        h.session
            .set_speaker_count(&meeting_id, None)
            .await
            .unwrap();
        assert_eq!(
            repo::speaker_count_override(&h.db, &meeting_id)
                .await
                .unwrap(),
            None
        );
        assert_eq!(
            repo::people_count(&h.db, &meeting_id).await.unwrap(),
            (2, false),
            "You and the one voice the pass found"
        );
    }

    /// A meeting happening now outranks tidying up how many people were in one
    /// that already ended — and the refusal must not store anything.
    #[tokio::test]
    async fn the_meeting_being_recorded_is_turned_down() {
        let h = Harness::new().await;
        let meeting_id = h.session.start(Default::default()).await.unwrap();

        let error = h
            .session
            .set_speaker_count(&meeting_id, Some(3))
            .await
            .unwrap_err();
        assert!(
            matches!(error, SessionError::AlreadyRecording),
            "expected a polite refusal, got {error:?}"
        );
        assert_eq!(
            repo::speaker_count_override(&h.db, &meeting_id)
                .await
                .unwrap(),
            None,
            "a refusal must not leave the number behind"
        );
        assert!(!h
            .queued_kinds(&meeting_id)
            .await
            .contains(&JobKind::Diarize));
    }

    /// A different meeting being recorded is not this meeting's problem: the
    /// work queues and the runtime parks it until the recording is over.
    #[tokio::test]
    async fn a_recording_of_something_else_does_not_block_it() {
        let h = Harness::new().await;
        let finished = a_finished_meeting(&h).await;
        let live = h.session.start(Default::default()).await.unwrap();
        assert_ne!(live, finished);

        h.session
            .set_speaker_count(&finished, Some(3))
            .await
            .unwrap();
        assert_eq!(
            repo::speaker_count_override(&h.db, &finished)
                .await
                .unwrap(),
            Some(3)
        );
    }

    #[tokio::test]
    async fn asking_again_while_the_pass_runs_changes_nothing() {
        let h = Harness::new().await;
        let meeting_id = a_finished_meeting(&h).await;

        let first = h
            .session
            .set_speaker_count(&meeting_id, Some(3))
            .await
            .unwrap();
        let job_id = match first {
            SpeakerCountSet::Requeued(id) => id,
            other => panic!("{other:?}"),
        };
        repo::set_job_status(&h.db, &job_id, JobStatus::Running, None)
            .await
            .unwrap();

        // The same number again: nothing to do, and no second pile of work.
        let again = h
            .session
            .set_speaker_count(&meeting_id, Some(3))
            .await
            .unwrap();
        assert_eq!(again, SpeakerCountSet::Unchanged);
        assert_eq!(h.queued_kinds(&meeting_id).await, vec![JobKind::Diarize]);
        assert_eq!(
            repo::get_job(&h.db, &job_id).await.unwrap().unwrap().status,
            JobStatus::Running,
            "the running pass is left alone"
        );
    }

    /// A *different* number while the pass runs is a real conflict: the pass is
    /// already writing rows for the old one.
    #[tokio::test]
    async fn changing_the_number_while_the_pass_runs_is_refused() {
        let h = Harness::new().await;
        let meeting_id = a_finished_meeting(&h).await;

        let first = h
            .session
            .set_speaker_count(&meeting_id, Some(3))
            .await
            .unwrap();
        let job_id = match first {
            SpeakerCountSet::Requeued(id) => id,
            other => panic!("{other:?}"),
        };
        repo::set_job_status(&h.db, &job_id, JobStatus::Running, None)
            .await
            .unwrap();

        let error = h
            .session
            .set_speaker_count(&meeting_id, Some(5))
            .await
            .unwrap_err();
        assert!(matches!(error, SessionError::MeetingBusy), "{error:?}");
        assert_eq!(
            repo::speaker_count_override(&h.db, &meeting_id)
                .await
                .unwrap(),
            Some(3),
            "the refused number was not stored"
        );
    }

    /// A listen again ends in a speaker pass of its own, so it counts as busy.
    #[tokio::test]
    async fn a_listen_again_in_flight_counts_as_busy() {
        let h = Harness::new().await;
        let meeting_id = a_finished_meeting(&h).await;
        h.session
            .queue_job(Some(&meeting_id), JobKind::TranscribeCatchup)
            .await
            .unwrap();

        let error = h
            .session
            .set_speaker_count(&meeting_id, Some(2))
            .await
            .unwrap_err();
        assert!(matches!(error, SessionError::MeetingBusy), "{error:?}");
    }

    #[tokio::test]
    async fn an_absurd_number_is_pulled_back_into_range_rather_than_stored() {
        let h = Harness::new().await;
        let meeting_id = a_finished_meeting(&h).await;
        h.session
            .set_speaker_count(&meeting_id, Some(500))
            .await
            .unwrap();
        assert_eq!(
            repo::speaker_count_override(&h.db, &meeting_id)
                .await
                .unwrap(),
            Some(MAX_PEOPLE)
        );
    }

    #[tokio::test]
    async fn a_meeting_that_does_not_exist_is_not_found() {
        let h = Harness::new().await;
        let error = h
            .session
            .set_speaker_count(&repo::new_id(), Some(2))
            .await
            .unwrap_err();
        assert!(
            matches!(error, SessionError::Db(crate::db::DbError::NotFound(_))),
            "{error:?}"
        );
    }
}
