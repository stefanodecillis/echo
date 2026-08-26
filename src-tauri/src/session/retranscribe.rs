//! "Listen again": read a finished meeting's audio back and write the
//! transcript from scratch.
//!
//! Why it exists: some meetings were recorded while the transcription pipeline
//! was broken. Their audio is intact on disk, which is the whole point of
//! mantra 3 — the recording is the truth and everything else is derived — so the
//! words are still in there, waiting for a pipeline that works.
//!
//! What it does is deliberately the *stop* path with one thing missing:
//!
//! 1. refuse while a recording is live (a meeting happening now outranks tidying
//!    up one that already ended, DESIGN §3) and do nothing at all when a listen
//!    again is already under way for this meeting;
//! 2. cancel whatever is queued or running for the meeting, so no older pass is
//!    still writing rows when the wipe lands;
//! 3. delete the transcript and the speakers ([`repo::clear_transcript`]) and
//!    tell the screens the revision moved on;
//! 4. queue catch-up and then the speaker pass, exactly the chain a normal stop
//!    queues — minus the mixdown, because the audio did not change and the
//!    playback file is still correct, and minus the recap, because the recap is
//!    the person's to keep. They can ask for a new one when the words are back.
//!
//! Nothing here knows how to transcribe anything. Catch-up already works out,
//! per channel, which stretches of committed audio have no text against them and
//! reads exactly those ([`crate::asr::catchup`]); with the transcript deleted
//! that coverage is empty, so "the holes" and "the whole meeting, both channels,
//! `[0, duration]`" are the same list. That is why this is a delete followed by
//! the ordinary job, and not a second transcription path to keep in step.

use std::sync::Arc;

use crate::db::repo;
use crate::events::{SpeakersUpdatedPayload, TranscriptRevisedPayload};
use crate::session::ports::{EventSink, UiEvent};
use crate::session::{jobs, Inner, SessionError};
use crate::types::{Id, JobKind, JobQuery, MeetingStatus};

/// What a "listen again" request did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Retranscribed {
    /// The transcript is gone and the work is queued. Carries the catch-up job,
    /// which is what progress arrives against.
    Queued(Id),
    /// One was already under way, so this changed nothing. A second click must
    /// not wipe the rows the first pass has already written (DESIGN §3: every
    /// entry point is idempotent).
    AlreadyRunning(Id),
}

/// Is a listen again already in flight for this meeting?
///
/// An active catch-up job *is* the answer: it is the only kind that writes
/// transcript rows from disk, whether it was queued by a stop, a recovery or by
/// this module, and in every one of those cases the right response to "listen
/// again" is "it already is".
async fn catch_up_in_flight(inner: &Arc<Inner>, meeting_id: &str) -> Option<Id> {
    repo::list_jobs(
        &inner.db,
        &JobQuery {
            meeting_id: Some(meeting_id.to_string()),
            kind: Some(JobKind::TranscribeCatchup),
            active_only: Some(true),
            ..Default::default()
        },
    )
    .await
    .unwrap_or_default()
    .into_iter()
    .next()
    .map(|job| job.id)
}

/// Wipe this meeting's transcript and queue the work that writes it again.
///
/// Callers hold the session command lock, so this cannot interleave with a start
/// or a stop.
pub(crate) async fn run(
    inner: &Arc<Inner>,
    meeting_id: &str,
) -> Result<Retranscribed, SessionError> {
    // A recording owns the machine, and it owns it whichever meeting this is
    // about: the weights, the disk and the person's attention are all pointed
    // somewhere else.
    if super::is_live(inner.state()) || matches!(inner.state(), super::CaptureState::Starting) {
        return Err(SessionError::AlreadyRecording);
    }

    let Some(meeting) = repo::get_meeting(&inner.db, meeting_id).await? else {
        return Err(SessionError::Db(crate::db::DbError::NotFound(format!(
            "meeting {meeting_id}"
        ))));
    };

    // Idempotent: the second click is the first one.
    if let Some(job_id) = catch_up_in_flight(inner, meeting_id).await {
        tracing::debug!(meeting = %meeting_id, "listen again ignored, one is already running");
        return Ok(Retranscribed::AlreadyRunning(job_id));
    }

    // Without audio there is nothing to listen to again — the person deleted the
    // recording and kept the words, which is a choice Echo offers.
    let committed = jobs::committed_end_ms(&inner.db, meeting_id).await?;
    if committed <= 0 {
        return Err(SessionError::NoRecordedAudio);
    }

    // Stop everything else for this meeting first. A speaker pass or a recap
    // reading rows that are about to disappear is wasted work at best, and a
    // recap written from a half-deleted transcript at worst.
    inner.cancel_jobs_for(meeting_id).await;

    let cleared = repo::clear_transcript(&inner.db, meeting_id).await?;
    // The row is only half of where the meeting's language lives: the engine
    // keeps its own answer in memory for the length of a run, and catch-up asks
    // it whenever the row has nothing (`asr::catchup`'s `prior`). Clearing one
    // and not the other would hand the next pass the same wrong language it was
    // asked to get rid of, out of a place nobody can see.
    inner.ports.asr.forget_meeting(meeting_id);
    tracing::info!(
        meeting = %meeting_id,
        segments = cleared.segments_deleted,
        speakers = cleared.speakers_deleted,
        // Named, because "the transcript came back in the wrong language" is
        // the reason this button gets pressed, and this line is the evidence
        // that the next pass starts without that answer.
        was_language = meeting.language.as_deref().unwrap_or("none"),
        language_cleared = cleared.language_cleared,
        // Stretches a previous pass had decided were the computer's own audio
        // coming back. They are judged again from the recording, because that
        // is what "listen again" means.
        spans_unmarked = cleared.spans_unmarked,
        duration_ms = meeting.duration_ms.max(committed),
        "listening to this meeting again from the recording"
    );

    // Everything on screen for this meeting is now wrong. The revision only ever
    // moves forward, and the span is the whole meeting, so anything holding
    // transcript rows refetches and finds none until the pass writes the first.
    inner
        .ports
        .events
        .emit(UiEvent::TranscriptRevised(TranscriptRevisedPayload {
            meeting_id: meeting_id.to_string(),
            revision: cleared.revision,
            from_ms: 0,
            to_ms: i64::MAX,
            segment_ids: Vec::new(),
        }));
    // And the speaker chips, whose rows are gone. Said out loud rather than left
    // for the speaker pass to correct at the end, or the transcript would spend
    // the whole pass pointing at people who no longer exist.
    // The person's own count of how many people were here is theirs and survives
    // the wipe — it is a fact about the meeting, not something derived from the
    // transcript — so it is read back rather than zeroed. Without one there is
    // nobody left to count until the pass runs again.
    let (people_count, people_count_is_override) =
        repo::people_count(&inner.db, meeting_id).await?;
    inner
        .ports
        .events
        .emit(UiEvent::SpeakersUpdated(SpeakersUpdatedPayload {
            meeting_id: meeting_id.to_string(),
            speakers: Vec::new(),
            people_count,
            people_count_is_override,
            // The rows are gone and the pass has not run yet, so there is
            // nothing true to say about how many voices are in there.
            voices_found: None,
            alternative_count: None,
        }));

    // Back to Processing while the work runs. This is also what lets the meeting
    // settle: `JobRuntime::settle_meeting` moves a meeting to Complete when its
    // last job finishes *and* it is Processing, so without this the meeting would
    // sit on Complete throughout and never announce that it had finished again.
    if !matches!(meeting.status, MeetingStatus::Processing) {
        repo::set_meeting_status(&inner.db, meeting_id, MeetingStatus::Processing).await?;
    }
    inner.ports.events.emit(UiEvent::MeetingUpdated(
        crate::events::MeetingUpdatedPayload {
            meeting_id: meeting_id.to_string(),
            status: MeetingStatus::Processing,
            title: Some(meeting.title.clone()),
            duration_ms: meeting.duration_ms,
            deleted: false,
        },
    ));

    // Catch-up, then who said what. `next_queued_job` sorts by kind, so the order
    // holds however they are queued. Queueing is also what claims the speech
    // engine: `JobRuntime::queue` recomputes residency from "is a capture live"
    // and "does any meeting have work outstanding", and this just made the second
    // one true (see `session::engine_stays_resident`).
    let job = inner
        .jobs
        .queue(Some(meeting_id), JobKind::TranscribeCatchup)
        .await?;
    inner.jobs.queue(Some(meeting_id), JobKind::Diarize).await?;
    // No mixdown: the audio did not change, so the playback file is still the
    // right one. No recap either — the existing one and its tasks stay, and
    // rewriting it is a separate button the person presses when they have read
    // the new transcript.
    Ok(Retranscribed::Queued(job.id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::mock::Harness;
    use crate::types::{Channel, JobStatus, SearchQuery, SegmentDraft, TranscriptQuery};

    /// A finished meeting with audio on disk, a transcript and two speakers —
    /// the shape of a real meeting recorded while the pipeline was broken.
    async fn a_recorded_meeting(h: &Harness) -> Id {
        let meeting = repo::create_meeting(&h.db, "Weekly sync", "/tmp/echo-listen-again", None)
            .await
            .unwrap();
        let chunk = repo::insert_chunk(
            &h.db,
            &meeting.id,
            Channel::Mic,
            0,
            "/tmp/echo-listen-again/mic-0.flac",
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
            Some("/tmp/echo-listen-again/mixed.flac"),
            MeetingStatus::Complete,
        )
        .await
        .unwrap();

        let speaker = repo::upsert_speaker(&h.db, &meeting.id, "mic", "You", true)
            .await
            .unwrap();
        let alias = repo::upsert_speaker(&h.db, &meeting.id, "cluster-2", "Speaker 2", false)
            .await
            .unwrap();
        repo::merge_speakers(&h.db, &alias.id, &speaker.id)
            .await
            .unwrap();
        repo::insert_segments(
            &h.db,
            &[SegmentDraft {
                meeting_id: meeting.id.clone(),
                t_start_ms: 0,
                t_end_ms: 4_000,
                channel: Channel::Mic,
                speaker_id: Some(speaker.id.clone()),
                text: "aubergine gibberish".into(),
                language: Some("en".into()),
                avg_confidence: Some(0.2),
                // The person tidied a line up by hand, so the transcript is
                // past its first revision.
                revision: 3,
                is_final: true,
                model_name: None,
                model_revision: None,
                corrections: Vec::new(),
            }],
        )
        .await
        .unwrap();
        meeting.id
    }

    async fn segments(h: &Harness, meeting_id: &str) -> Vec<crate::types::Segment> {
        repo::get_segments(
            &h.db,
            &TranscriptQuery {
                meeting_id: meeting_id.to_string(),
                ..Default::default()
            },
        )
        .await
        .unwrap()
    }

    /// A meeting happening now outranks one that already ended, so the answer is
    /// no — and, crucially, nothing is deleted on the way to saying so.
    #[tokio::test]
    async fn a_live_recording_means_no() {
        let h = Harness::new().await;
        let meeting_id = a_recorded_meeting(&h).await;
        h.session.start(Default::default()).await.unwrap();

        let error = h.session.retranscribe(&meeting_id).await.unwrap_err();
        assert!(
            matches!(error, SessionError::AlreadyRecording),
            "expected a polite refusal, got {error:?}"
        );
        assert_eq!(
            segments(&h, &meeting_id).await.len(),
            1,
            "a refusal must not touch the transcript"
        );
        assert_eq!(
            repo::list_speakers(&h.db, &meeting_id).await.unwrap().len(),
            2
        );
        assert!(h.queued_kinds(&meeting_id).await.is_empty());
    }

    /// The 2026-08-24 meeting: 75 minutes of Italian written down as Danish
    /// because one short utterance was read that way and the answer stuck.
    /// "Listen again" is the whole repair path, and it only works if the wrong
    /// language goes with the words — in both places it is kept, the row and
    /// the engine's own memory of the meeting. Otherwise the next pass reads
    /// "da" as its prior and spends an hour writing the same transcript again.
    #[tokio::test]
    async fn a_wrong_language_does_not_survive_listening_again() {
        let h = Harness::new().await;
        let meeting_id = a_recorded_meeting(&h).await;
        repo::set_meeting_language(&h.db, &meeting_id, "da")
            .await
            .unwrap();

        h.session.retranscribe(&meeting_id).await.unwrap();

        let meeting = repo::get_meeting(&h.db, &meeting_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            meeting.language, None,
            "the next pass has to be free to reach a different answer"
        );
        assert_eq!(
            h.asr.forgotten(),
            vec![meeting_id.clone()],
            "the engine's own copy of the language goes too"
        );
    }

    /// The whole sequence: the transcript and the speakers go, the search index
    /// goes with them, the revision moves forward, and exactly catch-up and the
    /// speaker pass are queued — no mixdown, no recap.
    #[tokio::test]
    async fn the_transcript_is_wiped_and_the_work_requeued() {
        let h = Harness::new().await;
        let meeting_id = a_recorded_meeting(&h).await;
        let before = repo::transcript_revision(&h.db, &meeting_id).await.unwrap();
        assert_eq!(before, 3);
        // Something else was queued for this meeting and has to be got out of
        // the way first.
        h.session
            .queue_job(Some(&meeting_id), JobKind::Summarize)
            .await
            .unwrap();

        let outcome = h.session.retranscribe(&meeting_id).await.unwrap();
        let job_id = match outcome {
            Retranscribed::Queued(id) => id,
            other => panic!("expected the work to be queued, got {other:?}"),
        };

        // Gone: the rows, the speakers (the alias row included) and the words in
        // the search index, which is kept in step by a trigger on `segments`.
        assert!(segments(&h, &meeting_id).await.is_empty());
        assert!(repo::list_speakers(&h.db, &meeting_id)
            .await
            .unwrap()
            .is_empty());
        assert!(repo::search_segments(
            &h.db,
            &SearchQuery {
                text: "aubergine".into(),
                ..Default::default()
            }
        )
        .await
        .unwrap()
        .is_empty());

        // The audio and the recap-shaped things it did not ask about are intact.
        let meeting = repo::get_meeting(&h.db, &meeting_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(meeting.status, MeetingStatus::Processing);
        assert!(
            meeting.mixed_path.is_some(),
            "the playback file is still the right one"
        );
        assert_eq!(
            repo::last_committed_offset_ms(&h.db, &meeting_id, Channel::Mic)
                .await
                .unwrap(),
            60_000,
            "the recording itself is never touched"
        );

        // The screens were told the revision moved on, by a number that is
        // higher than the one they were holding.
        let revised = h.events.transcript_revisions();
        assert_eq!(revised.len(), 1);
        assert!(
            revised[0].revision > before,
            "revision went {before} -> {}",
            revised[0].revision
        );
        assert_eq!(revised[0].from_ms, 0);
        assert_eq!(revised[0].to_ms, i64::MAX);

        // Catch-up first, then the speaker pass, and nothing else. The recap
        // that was queued before is cancelled rather than left to run against a
        // transcript that no longer exists.
        let active: Vec<JobKind> = repo::list_jobs(
            &h.db,
            &JobQuery {
                meeting_id: Some(meeting_id.clone()),
                active_only: Some(true),
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .into_iter()
        .map(|job| job.kind)
        .collect();
        assert!(active.contains(&JobKind::TranscribeCatchup), "{active:?}");
        assert!(active.contains(&JobKind::Diarize), "{active:?}");
        assert!(!active.contains(&JobKind::Mixdown), "{active:?}");
        assert!(!active.contains(&JobKind::Summarize), "{active:?}");
        assert_eq!(
            repo::next_queued_job(&h.db).await.unwrap().map(|j| j.kind),
            Some(JobKind::TranscribeCatchup),
            "catch-up runs before the speaker pass"
        );
        // Queueing work for a meeting is what holds the speech engine
        // (`session::engine_stays_resident`). Nothing here computes that a
        // second time; this is the check that going through `JobRuntime::queue`
        // really does claim it.
        assert!(
            h.asr.is_resident(),
            "a meeting with work outstanding holds the engine"
        );
        assert_eq!(
            repo::get_job(&h.db, &job_id).await.unwrap().unwrap().kind,
            JobKind::TranscribeCatchup,
            "progress is reported against the catch-up job"
        );
    }

    /// The double-click, and the impatient second press minutes later: neither
    /// may throw away the rows the running pass has already written.
    #[tokio::test]
    async fn asking_again_while_it_runs_changes_nothing() {
        let h = Harness::new().await;
        let meeting_id = a_recorded_meeting(&h).await;
        let first = h.session.retranscribe(&meeting_id).await.unwrap();
        let job_id = match first {
            Retranscribed::Queued(id) => id,
            other => panic!("{other:?}"),
        };

        // The pass has started and has written its first line back.
        repo::set_job_status(&h.db, &job_id, JobStatus::Running, None)
            .await
            .unwrap();
        repo::insert_segments(
            &h.db,
            &[SegmentDraft {
                meeting_id: meeting_id.clone(),
                t_start_ms: 0,
                t_end_ms: 4_000,
                channel: Channel::Mic,
                text: "we ship on Friday".into(),
                revision: 1,
                is_final: true,
                ..Default::default()
            }],
        )
        .await
        .unwrap();
        let jobs_before = repo::list_jobs(
            &h.db,
            &JobQuery {
                meeting_id: Some(meeting_id.clone()),
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .len();

        let again = h.session.retranscribe(&meeting_id).await.unwrap();
        assert_eq!(again, Retranscribed::AlreadyRunning(job_id.clone()));
        assert_eq!(
            segments(&h, &meeting_id).await.len(),
            1,
            "the running pass keeps what it has written"
        );
        assert_eq!(
            repo::get_job(&h.db, &job_id).await.unwrap().unwrap().status,
            JobStatus::Running,
            "the job that is running is left alone"
        );
        assert_eq!(
            repo::list_jobs(
                &h.db,
                &JobQuery {
                    meeting_id: Some(meeting_id.clone()),
                    ..Default::default()
                }
            )
            .await
            .unwrap()
            .len(),
            jobs_before,
            "no second pile of work"
        );
    }

    /// Putting the meeting back to Processing is what lets it finish again:
    /// `JobRuntime::settle_meeting` only moves a meeting to Complete from there,
    /// so a listen again that left the status alone would leave the meeting
    /// looking finished all the way through and never announce that it was.
    #[tokio::test]
    async fn the_meeting_finishes_again_when_the_work_does() {
        let h = Harness::new().await;
        let meeting_id = a_recorded_meeting(&h).await;
        h.session.retranscribe(&meeting_id).await.unwrap();

        h.session.start_job_runner().await.unwrap();
        h.executor.wait_for_kinds(2).await;

        let mut settled = None;
        for _ in 0..200 {
            let meeting = repo::get_meeting(&h.db, &meeting_id)
                .await
                .unwrap()
                .unwrap();
            if matches!(meeting.status, MeetingStatus::Complete) {
                settled = Some(meeting);
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(
            settled.is_some(),
            "the meeting never went back to finished after the work ran"
        );
        assert_eq!(
            h.executor.finished(),
            vec![JobKind::TranscribeCatchup, JobKind::Diarize],
            "catch-up, then who said what — and nothing else"
        );
        // And with nothing left outstanding, the engine's grace period can start.
        h.wait_until("the engine to be let go", || !h.asr.is_resident())
            .await;
    }

    /// Nothing to listen to: the words were kept and the recording was deleted.
    #[tokio::test]
    async fn a_meeting_with_no_audio_is_turned_down() {
        let h = Harness::new().await;
        let meeting = repo::create_meeting(&h.db, "Notes only", "/tmp/echo-listen-again", None)
            .await
            .unwrap();
        let error = h.session.retranscribe(&meeting.id).await.unwrap_err();
        assert!(
            matches!(error, SessionError::NoRecordedAudio),
            "got {error:?}"
        );
        assert!(h.queued_kinds(&meeting.id).await.is_empty());
    }
}
