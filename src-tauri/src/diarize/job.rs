//! The speaker pass as a row in the `jobs` table.
//!
//! Jobs are persisted so work survives a restart, and this one is cancellable,
//! reports progress, and steps aside for a recording (DESIGN §3 "State model":
//! *recording has absolute resource priority; background jobs pause*).
//!
//! Stepping aside means `paused`, not `failed`: `repo::resume_paused_jobs` will
//! queue it again when the recording ends, and the pass reads only finished
//! audio off disk, so nothing is lost either way. What it does *not* do any more
//! is start over: the expensive half is written down on the way out and picked
//! up at the window it stopped on ([`super::scan_cache`]), because finishing a
//! meeting costs about a sixth of its length and a day of back-to-back meetings
//! never leaves that much of a gap.

use std::sync::Arc;

use crate::asr::models;
use crate::db::{repo, Db};
use crate::types::{AssetKind, JobStatus};

use super::pipeline::{self, DiarizeControl};
use super::{DiarizationResult, DiarizeError};

/// Where the two ONNX assets live, resolved through the download catalog.
///
/// Neither is optional: without the segmenter there are no turns, and without
/// the fingerprints every turn is the same person. A missing asset is
/// [`DiarizeError::NotInstalled`], which the UI renders as an offer to fetch it
/// — never as a failure the person cannot act on.
pub async fn model_paths(
    db: &Db,
) -> Result<(std::path::PathBuf, std::path::PathBuf), DiarizeError> {
    let segmenter = models::installed_path(db, AssetKind::SpeakerSegmenter)
        .await
        .map_err(|e| DiarizeError::Load(e.to_string()))?
        .ok_or(DiarizeError::NotInstalled)?;
    let embedder = models::installed_path(db, AssetKind::SpeakerEmbedder)
        .await
        .map_err(|e| DiarizeError::Load(e.to_string()))?
        .ok_or(DiarizeError::NotInstalled)?;
    Ok((segmenter, embedder))
}

/// Run one `diarize` job to completion, keeping its row honest throughout.
///
/// `should_yield` is how the session layer says "a recording just started".
/// Returns the pass's own result so the caller can emit
/// [`crate::events::SPEAKERS_UPDATED`] without a second query.
pub async fn run(
    db: &Db,
    job_id: &str,
    meeting_id: &str,
    control: DiarizeControl,
) -> Result<DiarizationResult, DiarizeError> {
    repo::set_job_status(db, job_id, JobStatus::Running, None)
        .await
        .map_err(|e| DiarizeError::Failed(e.to_string()))?;

    let outcome = attempt(db, job_id, meeting_id, control).await;

    let (status, error) = match &outcome {
        Ok(_) => (JobStatus::Done, None),
        Err(DiarizeError::Cancelled) => (JobStatus::Cancelled, None),
        // Not a failure: the recording won, and this pass costs nothing to redo.
        Err(DiarizeError::Yielded) => (JobStatus::Paused, None),
        Err(other) => (JobStatus::Failed, Some(other.to_string())),
    };
    repo::set_job_status(db, job_id, status, error.as_deref())
        .await
        .map_err(|e| DiarizeError::Failed(e.to_string()))?;

    outcome
}

async fn attempt(
    db: &Db,
    job_id: &str,
    meeting_id: &str,
    control: DiarizeControl,
) -> Result<DiarizationResult, DiarizeError> {
    // Cancellation first: a job the person already stopped must not go looking
    // for models, let alone load them.
    control.checkpoint()?;
    let (segmenter_path, embedder_path) = model_paths(db).await?;

    // The pass reports progress from a synchronous callback, which cannot write
    // to the database. Funnel it through a channel and let one task do the
    // writing, coalescing anything finer than a percent so a long meeting does
    // not turn into thousands of UPDATEs.
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<f32>();
    let writer_db = db.clone();
    let writer_job = job_id.to_string();
    let writer = tokio::spawn(async move {
        let mut last = -1.0f32;
        while let Some(p) = rx.recv().await {
            if p >= 1.0 || p - last >= 0.01 {
                last = p;
                let _ = repo::set_job_progress(&writer_db, &writer_job, p).await;
            }
        }
    });

    let control = control.on_progress(Arc::new(move |p| {
        let _ = tx.send(p);
    }));

    let result = pipeline::refine(db, meeting_id, &segmenter_path, &embedder_path, &control).await;

    // Dropping the control closes the channel, which ends the writer task.
    drop(control);
    let _ = writer.await;
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{JobKind, MeetingStatus};

    async fn db() -> Db {
        let db = crate::db::connect_in_memory().await.expect("in-memory db");
        crate::db::migrate(&db).await.expect("migrations");
        db
    }

    #[tokio::test]
    async fn without_the_assets_the_job_asks_for_a_download_rather_than_failing_quietly() {
        let db = db().await;
        let err = model_paths(&db).await.unwrap_err();
        assert!(matches!(err, DiarizeError::NotInstalled), "{err:?}");
    }

    #[tokio::test]
    async fn a_job_that_cannot_find_the_assets_ends_up_failed_with_a_reason() {
        let db = db().await;
        let meeting = repo::create_meeting(&db, "Standup", "/tmp/echo-test", None)
            .await
            .unwrap();
        repo::set_meeting_status(&db, &meeting.id, MeetingStatus::Processing)
            .await
            .unwrap();
        let job = repo::create_job(&db, Some(&meeting.id), JobKind::Diarize)
            .await
            .unwrap();

        let err = run(&db, &job.id, &meeting.id, DiarizeControl::new())
            .await
            .unwrap_err();
        assert!(matches!(err, DiarizeError::NotInstalled), "{err:?}");

        let row = repo::get_job(&db, &job.id).await.unwrap().expect("job row");
        assert_eq!(row.status, JobStatus::Failed);
        assert!(row.error.is_some());
    }

    #[tokio::test]
    async fn cancelling_before_the_job_starts_leaves_it_cancelled_not_failed() {
        let db = db().await;
        let meeting = repo::create_meeting(&db, "Retro", "/tmp/echo-test", None)
            .await
            .unwrap();
        let job = repo::create_job(&db, Some(&meeting.id), JobKind::Diarize)
            .await
            .unwrap();

        let control = DiarizeControl::new();
        control.cancel();
        // Cancellation is checked before the assets are, so this is Cancelled
        // even on a machine that has never downloaded them.
        let err = run(&db, &job.id, &meeting.id, control).await.unwrap_err();
        assert!(matches!(err, DiarizeError::Cancelled), "{err:?}");
        let row = repo::get_job(&db, &job.id).await.unwrap().expect("job row");
        assert_eq!(row.status, JobStatus::Cancelled);
        assert!(row.error.is_none(), "cancelling is not an error");
    }

    #[tokio::test]
    async fn a_recording_starting_parks_the_job_so_it_can_be_picked_up_again() {
        let db = db().await;
        let meeting = repo::create_meeting(&db, "Client call", "/tmp/echo-test", None)
            .await
            .unwrap();
        let job = repo::create_job(&db, Some(&meeting.id), JobKind::Diarize)
            .await
            .unwrap();

        let control = DiarizeControl::new().yield_when(Arc::new(|| true));
        let err = run(&db, &job.id, &meeting.id, control).await.unwrap_err();
        assert!(matches!(err, DiarizeError::Yielded), "{err:?}");

        let row = repo::get_job(&db, &job.id).await.unwrap().expect("job row");
        assert_eq!(row.status, JobStatus::Paused);
        assert!(row.error.is_none());
        // And the queue picks it back up once the recording is over.
        repo::resume_paused_jobs(&db).await.unwrap();
        let row = repo::get_job(&db, &job.id).await.unwrap().expect("job row");
        assert_eq!(row.status, JobStatus::Queued);
    }
}
