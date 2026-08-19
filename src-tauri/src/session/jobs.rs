//! The jobs runtime: one loop, one job at a time, over the `jobs` table.
//!
//! Rules that come straight out of DESIGN §3:
//! * **The table is the truth.** Queue, progress and failure live in SQLite, so
//!   closing Echo mid-recap loses nothing.
//! * **Recording has absolute priority.** Starting a capture parks whatever is
//!   running ([`JobRuntime::preempt`]); the job goes back to `paused` with its
//!   progress intact and picks up where it stopped when the recording ends.
//! * **Everything is cancellable**, and cancelling something already finished is
//!   success, not an error.
//! * **Order matters**: catch-up before speakers before the mixdown before the
//!   recap, which is what `repo::next_queued_job` sorts by.
//!
//! What a job actually *does* lives behind [`JobExecutor`], so the ordering and
//! preemption logic can be tested without a speech engine.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::future::BoxFuture;
use tokio::sync::Notify;

use crate::db::{repo, Db, DbError};
use crate::events::JobProgressPayload;
use crate::paths::AppPaths;
use crate::session::ports::{AsrPort, EventBus, EventSink, Ports, UiEvent};
use crate::types::{
    Channel, Id, Job, JobKind, JobQuery, JobStatus, MeetingStatus, SummaryReq,
};

/// How long the loop waits before looking at the table again when it has
/// nothing to do.
///
/// Long on purpose. Every path that puts work in the table calls
/// [`JobRuntime::queue`] or [`JobRuntime::release`], both of which wake the loop
/// immediately, so this is only a safety net for a row that appeared some other
/// way. An idle Echo should be a detection poll and nothing else (mantra 1), and
/// a query every couple of seconds is not nothing.
const IDLE_POLL: Duration = Duration::from_secs(30);
/// How often the loop re-checks whether the recording finished.
const BLOCKED_POLL: Duration = Duration::from_secs(5);
/// Progress updates are cheap but not free; cap them.
const PROGRESS_INTERVAL: Duration = Duration::from_millis(250);

// ---------------------------------------------------------------------------
// Cancellation
// ---------------------------------------------------------------------------

const RUNNING: u8 = 0;
const CANCELLED: u8 = 1;
const PREEMPTED: u8 = 2;

/// Cooperative cancel with two flavours: the person cancelled (terminal), or a
/// recording started (park it, resume later).
#[derive(Debug, Clone, Default)]
pub struct Cancel {
    state: Arc<AtomicU8>,
    /// Mirror for [`crate::summarize`], which has its own flag type.
    summarize: crate::summarize::CancelFlag,
}

impl Cancel {
    pub fn new() -> Self {
        Self::default()
    }

    /// The person asked for this to stop.
    pub fn cancel(&self) {
        self.state.store(CANCELLED, Ordering::SeqCst);
        self.summarize.cancel();
    }

    /// A recording needs the machine. The job must stop at its next checkpoint
    /// and stay resumable.
    pub fn preempt(&self) {
        let _ = self
            .state
            .compare_exchange(RUNNING, PREEMPTED, Ordering::SeqCst, Ordering::SeqCst);
        self.summarize.cancel();
    }

    pub fn is_cancelled(&self) -> bool {
        self.state.load(Ordering::SeqCst) != RUNNING
    }

    pub fn is_preempted(&self) -> bool {
        self.state.load(Ordering::SeqCst) == PREEMPTED
    }

    /// Flag to hand to [`crate::summarize`].
    pub fn as_flag(&self) -> crate::summarize::CancelFlag {
        self.summarize.clone()
    }

    /// `Err(..)` when the job should stop now.
    pub fn check(&self) -> Result<(), JobFailure> {
        match self.state.load(Ordering::SeqCst) {
            RUNNING => Ok(()),
            PREEMPTED => Err(JobFailure::Preempted),
            _ => Err(JobFailure::Cancelled),
        }
    }
}

/// Why a job did not finish.
#[derive(Debug)]
pub enum JobFailure {
    /// The person cancelled it. Terminal.
    Cancelled,
    /// A recording took the machine. Goes back to `paused`, keeps its progress.
    Preempted,
    /// Went wrong. The string is shown to the person, so it says what they can
    /// do, never what broke internally (mantra 2).
    Failed(String),
}

impl JobFailure {
    fn failed(message: impl Into<String>) -> Self {
        JobFailure::Failed(message.into())
    }
}

impl From<DbError> for JobFailure {
    fn from(err: DbError) -> Self {
        tracing::warn!(error = %err, "a job could not reach the database");
        JobFailure::failed("Echo couldn't finish this. Nothing was lost, try again in a moment.")
    }
}

// ---------------------------------------------------------------------------
// Copy
// ---------------------------------------------------------------------------

/// The sentence the UI shows while a job runs. Zero jargon (mantra 2).
pub fn label_for(kind: JobKind) -> &'static str {
    match kind {
        JobKind::TranscribeCatchup => "Catching up on the last few minutes…",
        JobKind::Diarize => "Working out who said what…",
        JobKind::Summarize => "Writing your recap…",
        JobKind::Export => "Saving your file…",
        JobKind::Download => "Downloading what Echo needs to understand speech…",
        JobKind::Mixdown => "Getting the recording ready to play back…",
    }
}

// ---------------------------------------------------------------------------
// Progress
// ---------------------------------------------------------------------------

/// Writes progress to the table and out to the UI, rate-capped.
pub struct Progress {
    db: Db,
    job: Job,
    events: Arc<EventBus>,
    last: std::sync::Mutex<Option<Instant>>,
    /// Where the job already was. Resumed work never appears to go backwards.
    floor: f32,
}

impl Progress {
    fn new(db: Db, job: Job, events: Arc<EventBus>) -> Self {
        let floor = job.progress.unwrap_or(0.0).clamp(0.0, 1.0);
        Self {
            db,
            job,
            events,
            last: std::sync::Mutex::new(None),
            floor,
        }
    }

    /// 0.0..=1.0. The ends always go through; the middle is capped.
    pub async fn set(&self, value: f32) {
        let value = value.clamp(0.0, 1.0).max(self.floor);
        let forced = value <= 0.0 || value >= 1.0;
        if !forced && !self.due() {
            return;
        }
        if let Err(error) = repo::set_job_progress(&self.db, &self.job.id, value).await {
            tracing::debug!(%error, "could not store job progress");
        }
        let mut job = self.job.clone();
        job.status = JobStatus::Running;
        job.progress = Some(value);
        self.events.emit(UiEvent::JobProgress(JobProgressPayload {
            label: Some(label_for(job.kind).to_string()),
            job,
        }));
    }

    fn due(&self) -> bool {
        let Ok(mut last) = self.last.lock() else {
            return false;
        };
        let now = Instant::now();
        match *last {
            Some(previous) if now.duration_since(previous) < PROGRESS_INTERVAL => false,
            _ => {
                *last = Some(now);
                true
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

/// Everything one job needs.
pub struct JobContext {
    pub job: Job,
    pub db: Db,
    pub paths: AppPaths,
    pub asr: Arc<dyn AsrPort>,
    pub events: Arc<EventBus>,
    pub cancel: Cancel,
    /// Shared so a handler can hand it to a task that reports progress from
    /// inside a long-running pass.
    pub progress: Arc<Progress>,
}

impl JobContext {
    /// Jobs that work on a meeting need one.
    pub fn meeting_id(&self) -> Result<&str, JobFailure> {
        self.job.meeting_id.as_deref().ok_or_else(|| {
            JobFailure::failed("Echo lost track of which meeting this was for. Try again.")
        })
    }
}

/// What a job kind actually does. Swapped out in tests.
pub trait JobExecutor: Send + Sync + 'static {
    fn run<'a>(&'a self, ctx: &'a JobContext) -> BoxFuture<'a, Result<(), JobFailure>>;
}

/// The real handlers.
pub struct DefaultJobExecutor;

impl JobExecutor for DefaultJobExecutor {
    fn run<'a>(&'a self, ctx: &'a JobContext) -> BoxFuture<'a, Result<(), JobFailure>> {
        Box::pin(async move {
            match ctx.job.kind {
                JobKind::TranscribeCatchup => catch_up(ctx).await,
                JobKind::Diarize => diarize(ctx).await,
                JobKind::Summarize => summarize(ctx).await,
                JobKind::Mixdown => mixdown(ctx).await,
                JobKind::Download => download(ctx).await,
                // Exports run straight from the command that asked for one;
                // nothing queues them.
                JobKind::Export => {
                    tracing::warn!("an export reached the queue; exports run inline");
                    Ok(())
                }
            }
        })
    }
}

/// How far the committed audio reaches, across channels.
pub async fn committed_end_ms(db: &Db, meeting_id: &str) -> Result<i64, DbError> {
    let mic = repo::last_committed_offset_ms(db, meeting_id, Channel::Mic).await?;
    let system = repo::last_committed_offset_ms(db, meeting_id, Channel::System).await?;
    Ok(mic.max(system))
}

async fn catch_up(ctx: &JobContext) -> Result<(), JobFailure> {
    let meeting_id = ctx.meeting_id()?.to_string();
    ctx.progress.set(0.0).await;
    ctx.cancel.check()?;

    let committed = committed_end_ms(&ctx.db, &meeting_id).await?;
    if committed <= 0 {
        // Nothing on disk: there is nothing to catch up on, and that is fine.
        ctx.progress.set(1.0).await;
        return Ok(());
    }

    if let Err(error) = ctx.asr.prewarm().await {
        return Err(asr_failure(&ctx.cancel, error));
    }
    ctx.cancel.check()?;

    tracing::info!(
        meeting = %meeting_id,
        committed,
        "filling in whatever the live text missed, from the audio on disk"
    );
    ctx.progress.set(0.05).await;

    // The pass reports progress from its own task with a synchronous callback,
    // so it goes through a channel rather than blocking on a database write.
    // Dropping the pass's future drops the sender, which is what ends the drain.
    let (progress_tx, mut progress_rx) = tokio::sync::mpsc::unbounded_channel::<f32>();
    let drain = {
        let progress = ctx.progress.clone();
        tokio::spawn(async move {
            while let Some(fraction) = progress_rx.recv().await {
                progress.set(0.05 + fraction * 0.9).await;
            }
        })
    };
    let control = {
        let cancel = ctx.cancel.clone();
        crate::session::ports::CatchUpControl {
            cancel: Some(Arc::new(move || cancel.is_cancelled())),
            on_progress: Some(Arc::new(move |fraction| {
                let _ = progress_tx.send(fraction);
            })),
        }
    };

    // No offset: the pass works out per channel which stretches of audio have no
    // text against them and reads exactly those. An utterance the live queue
    // dropped mid-meeting is *behind* the end of the transcript, and the two
    // channels reach different lengths, so "carry on from where the text ends"
    // would lose words that are sitting on disk (mantra 3).
    let written = ctx
        .asr
        .catch_up(&ctx.db, &meeting_id, None, control)
        .await
        .map_err(|e| asr_failure(&ctx.cancel, e));
    let _ = drain.await;
    let written = written?;

    // Live partials are superseded now.
    if let Err(error) = repo::delete_partial_segments(&ctx.db, &meeting_id).await {
        tracing::debug!(%error, "could not clear live text");
    }
    if let Ok(hist) = repo::language_histogram(&ctx.db, &meeting_id).await {
        if let Some((language, _)) = hist.first() {
            let _ = repo::set_meeting_language(&ctx.db, &meeting_id, language).await;
        }
    }

    tracing::info!(meeting = %meeting_id, written, "catch-up finished");
    ctx.progress.set(1.0).await;
    Ok(())
}

async fn diarize(ctx: &JobContext) -> Result<(), JobFailure> {
    let meeting_id = ctx.meeting_id()?.to_string();
    ctx.progress.set(0.0).await;
    ctx.cancel.check()?;

    let (segmenter, embedder) =
        crate::diarize::job::model_paths(&ctx.db)
            .await
            .map_err(|error| match error {
                crate::diarize::DiarizeError::NotInstalled => JobFailure::failed(
                    "Echo needs a one-time download before it can tell voices apart.",
                ),
                other => diarize_failure(&ctx.cancel, other),
            })?;

    ctx.progress.set(0.1).await;

    // Same shape as catch-up: the pass reports through a synchronous callback,
    // so a channel carries it to the one task allowed to write.
    let (progress_tx, mut progress_rx) = tokio::sync::mpsc::unbounded_channel::<f32>();
    let drain = {
        let progress = ctx.progress.clone();
        tokio::spawn(async move {
            while let Some(fraction) = progress_rx.recv().await {
                progress.set(0.1 + fraction * 0.85).await;
            }
        })
    };
    let cancel = ctx.cancel.clone();
    let control = crate::diarize::DiarizeControl::new()
        .on_progress(Arc::new(move |fraction| {
            let _ = progress_tx.send(fraction);
        }))
        // Cancel and preempt reach the pass through the same flag; the runner
        // decides afterwards which of the two it was.
        .yield_when(Arc::new(move || cancel.is_cancelled()));

    let outcome =
        crate::diarize::refine_speakers_with(&ctx.db, &meeting_id, &segmenter, &embedder, &control)
            .await;
    drop(control);
    let _ = drain.await;
    let result = outcome.map_err(|error| diarize_failure(&ctx.cancel, error))?;

    let _ = repo::recompute_speaking_time(&ctx.db, &meeting_id).await;
    ctx.events.emit(UiEvent::SpeakersUpdated(
        crate::events::SpeakersUpdatedPayload {
            meeting_id: meeting_id.clone(),
            speakers: result.speakers.clone(),
        },
    ));
    // The pass re-points segments at new speakers, so anything showing the
    // transcript has to refetch.
    if let Ok(revision) = repo::transcript_revision(&ctx.db, &meeting_id).await {
        ctx.events.emit(UiEvent::TranscriptRevised(
            crate::events::TranscriptRevisedPayload {
                meeting_id: meeting_id.clone(),
                revision,
                from_ms: 0,
                to_ms: i64::MAX,
                segment_ids: Vec::new(),
            },
        ));
    }
    ctx.progress.set(1.0).await;
    Ok(())
}

fn diarize_failure(cancel: &Cancel, error: crate::diarize::DiarizeError) -> JobFailure {
    use crate::diarize::DiarizeError as D;
    match error {
        // "Yielded" is the pass noticing a recording; both arrive through the
        // same flag, so the runner's own state decides which it was.
        D::Cancelled | D::Yielded => {
            if cancel.is_preempted() {
                JobFailure::Preempted
            } else {
                JobFailure::Cancelled
            }
        }
        D::NotInstalled => {
            JobFailure::failed("Echo needs a one-time download before it can tell voices apart.")
        }
        other => {
            tracing::warn!(error = %other, "the speaker pass stopped");
            JobFailure::failed(
                "Echo couldn't tell the voices apart this time. The transcript is fine.",
            )
        }
    }
}

async fn summarize(ctx: &JobContext) -> Result<(), JobFailure> {
    let meeting_id = ctx.meeting_id()?.to_string();
    ctx.progress.set(0.0).await;
    ctx.cancel.check()?;

    // What the person actually asked for, when they asked for something
    // specific ("write this again, with that style"). Falling back to settings
    // covers the automatic recap that runs when a meeting ends.
    let settings = crate::settings::load(&ctx.db).await?;
    let requested: Option<SummaryReq> = repo::get_job_payload(&ctx.db, &ctx.job.id)
        .await
        .ok()
        .flatten()
        .and_then(|raw| match serde_json::from_str::<SummaryReq>(&raw) {
            Ok(req) => Some(req),
            Err(error) => {
                tracing::debug!(%error, "could not read what this recap was asked for");
                None
            }
        });
    let req = match requested {
        Some(mut req) => {
            // The meeting the job is attached to always wins over the payload.
            req.meeting_id = meeting_id.clone();
            if req.template_id.is_none() {
                req.template_id = settings.summary_template_id.clone();
            }
            if req.provider.is_none() {
                req.provider = Some(settings.summary_provider);
            }
            if req.language.is_none() {
                req.language = Some(settings.summary_language.clone());
            }
            req
        }
        None => SummaryReq {
            meeting_id: meeting_id.clone(),
            template_id: settings.summary_template_id.clone(),
            provider: Some(settings.summary_provider),
            model: None,
            language: Some(settings.summary_language.clone()),
            force: None,
        },
    };

    ctx.progress.set(0.1).await;
    let summary = crate::summarize::summarize_meeting(&ctx.db, &req, ctx.cancel.as_flag())
        .await
        .map_err(|error| summarize_failure(&ctx.cancel, error))?;

    ctx.progress.set(0.8).await;
    match crate::summarize::extract_action_items(
        &ctx.db,
        &meeting_id,
        &summary.id,
        ctx.cancel.as_flag(),
    )
    .await
    {
        Ok(items) => {
            match repo::replace_action_items(&ctx.db, &meeting_id, Some(&summary.id), &items).await
            {
                Ok(stored) => ctx.events.emit(UiEvent::ActionItemsUpdated(
                    crate::events::ActionItemsUpdatedPayload {
                        meeting_id: meeting_id.clone(),
                        items: stored,
                    },
                )),
                Err(error) => tracing::warn!(%error, "could not store the task list"),
            }
        }
        Err(error) => {
            // A recap without a task list is still a good recap.
            tracing::warn!(error = %error, "no task list came back");
        }
    }

    ctx.events
        .emit(UiEvent::SummaryReady(crate::events::SummaryReadyPayload {
            meeting_id,
            summary_id: summary.id,
        }));
    ctx.progress.set(1.0).await;
    Ok(())
}

async fn mixdown(ctx: &JobContext) -> Result<(), JobFailure> {
    let meeting_id = ctx.meeting_id()?.to_string();
    ctx.progress.set(0.0).await;
    ctx.cancel.check()?;

    let chunks = repo::list_chunks(&ctx.db, &meeting_id, None).await?;
    // Each chunk carries where it belongs, so a gap or an unreadable piece
    // becomes silence in the right place instead of pulling the rest of the
    // meeting earlier (click-to-play has to land on the words it shows).
    let sources: Vec<crate::audio::ChunkRef> = chunks
        .iter()
        .filter(|c| c.committed && c.channel != Channel::Mixed)
        .map(crate::audio::ChunkRef::from_journal)
        .collect();
    if sources.is_empty() {
        // Audio-only-deleted, or nothing was captured. Not a failure.
        ctx.progress.set(1.0).await;
        return Ok(());
    }

    let destination = ctx.paths.mixed_path(&meeting_id);
    ctx.progress.set(0.2).await;
    crate::audio::build_mixdown(&sources, &destination)
        .await
        .map_err(|error| {
            tracing::warn!(error = %error, "could not build the playback file");
            JobFailure::failed(
                "Echo couldn't get this recording ready to play back. The recording itself is safe.",
            )
        })?;
    // Record where playback should look. `get_playback_path` also checks the
    // file on disk, so this is provenance rather than a hard dependency.
    if let Err(error) =
        repo::set_meeting_mixed_path(&ctx.db, &meeting_id, &destination.to_string_lossy()).await
    {
        tracing::debug!(%error, "could not store the playback path");
    }
    ctx.progress.set(1.0).await;
    Ok(())
}

async fn download(ctx: &JobContext) -> Result<(), JobFailure> {
    ctx.progress.set(0.0).await;
    ctx.cancel.check()?;
    // Which preset was asked for travels with the job (migration 0002), so
    // queueing a download never overwrites the preset the person is recording
    // with. Older rows have no payload; those meant "the selected one".
    let requested: Option<String> = repo::get_job_payload(&ctx.db, &ctx.job.id)
        .await
        .ok()
        .flatten()
        .and_then(|raw| match serde_json::from_str::<String>(&raw) {
            Ok(level) => Some(level),
            Err(error) => {
                tracing::debug!(%error, "could not read which download this was");
                None
            }
        })
        .map(|level| level.trim().to_string())
        .filter(|level| !level.is_empty());
    let level_id = match requested {
        Some(level) => level,
        None => crate::settings::load(&ctx.db).await?.accuracy_level_id,
    };

    // The download callback is synchronous and must stay cheap, so it only
    // emits and stores a number; a ticker turns that into job progress.
    let permille = Arc::new(AtomicU32::new(0));
    let events = ctx.events.clone();
    let level_for_progress = level_id.clone();
    let shared = permille.clone();
    let on_progress: crate::asr::models::ProgressFn = Box::new(move |p| {
        let mut eta_seconds = None;
        if p.total_bytes > 0 {
            let value = ((p.received_bytes as f64 / p.total_bytes as f64) * 1000.0) as u32;
            shared.store(value.min(1000), Ordering::Relaxed);
            if p.bytes_per_second > 1.0 {
                let left = (p.total_bytes - p.received_bytes).max(0) as f64;
                eta_seconds = Some((left / p.bytes_per_second) as u64);
            }
        }
        events.emit(UiEvent::DownloadProgress(
            crate::events::DownloadProgressPayload {
                asset_id: p.asset_id.clone(),
                level_id: p
                    .level_id
                    .clone()
                    .or_else(|| Some(level_for_progress.clone())),
                received_bytes: p.received_bytes,
                total_bytes: p.total_bytes,
                bytes_per_second: Some(p.bytes_per_second),
                eta_seconds,
                done: p.done,
                error: None,
            },
        ));
    });

    let result =
        crate::asr::models::download_level(&ctx.db, &ctx.paths, &level_id, on_progress).await;

    ctx.progress
        .set(permille.load(Ordering::Relaxed) as f32 / 1000.0)
        .await;
    result.map_err(|error| asr_failure(&ctx.cancel, error))?;
    ctx.progress.set(1.0).await;
    Ok(())
}

fn asr_failure(cancel: &Cancel, error: crate::asr::AsrError) -> JobFailure {
    use crate::asr::AsrError as A;
    match error {
        A::Cancelled => {
            if cancel.is_preempted() {
                JobFailure::Preempted
            } else {
                JobFailure::Cancelled
            }
        }
        A::NotInstalled => JobFailure::failed(
            "Echo still needs a one-time download before it can write down speech.",
        ),
        A::NotEnoughSpace { .. } => JobFailure::failed(
            "There isn't enough room on this computer to finish. Free some up and try again.",
        ),
        other => {
            tracing::warn!(error = %other, "speech work stopped");
            JobFailure::failed("Echo couldn't finish writing this one down. You can try again.")
        }
    }
}

fn summarize_failure(cancel: &Cancel, error: crate::summarize::SummarizeError) -> JobFailure {
    use crate::summarize::SummarizeError as S;
    match error {
        S::Cancelled => {
            if cancel.is_preempted() {
                JobFailure::Preempted
            } else {
                JobFailure::Cancelled
            }
        }
        S::NoTranscript => JobFailure::failed("There's nothing written down for this meeting yet."),
        S::MissingCredential => JobFailure::failed(
            "Echo needs a key for that service before it can write recaps with it.",
        ),
        S::Unreachable(_) => JobFailure::failed(
            "Echo couldn't reach the place that writes your recaps. Check it's running.",
        ),
        S::Timeout => JobFailure::failed("That took too long. You can try writing it again."),
        other => {
            tracing::warn!(error = %other, "the recap stopped");
            JobFailure::failed("Echo couldn't write the recap this time. You can try again.")
        }
    }
}

// ---------------------------------------------------------------------------
// The runtime
// ---------------------------------------------------------------------------

/// Drains the `jobs` table, one job at a time, and yields to recordings.
pub struct JobRuntime {
    db: Db,
    paths: AppPaths,
    ports: Ports,
    /// Id and cancel handle of whatever is running right now.
    running: std::sync::Mutex<Option<(Id, Cancel)>>,
    /// A recording owns the machine.
    blocked: AtomicBool,
    stopped: AtomicBool,
    started: AtomicBool,
    wake: Notify,
}

impl JobRuntime {
    pub fn new(db: Db, paths: AppPaths, ports: Ports) -> Arc<Self> {
        Arc::new(Self {
            db,
            paths,
            ports,
            running: std::sync::Mutex::new(None),
            blocked: AtomicBool::new(false),
            stopped: AtomicBool::new(false),
            started: AtomicBool::new(false),
            wake: Notify::new(),
        })
    }

    /// Start the loop. Calling it twice is a no-op, so a second launch path
    /// cannot double-run everything.
    pub fn start(self: &Arc<Self>) {
        if self
            .started
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return;
        }
        let runtime = self.clone();
        tokio::spawn(async move { runtime.run_loop().await });
    }

    /// Queue work, deduplicated per meeting and kind.
    pub async fn queue(&self, meeting_id: Option<&str>, kind: JobKind) -> Result<Job, DbError> {
        self.queue_with_payload(meeting_id, kind, None).await
    }

    /// [`JobRuntime::queue`], remembering the request that asked for it.
    pub async fn queue_with_payload(
        &self,
        meeting_id: Option<&str>,
        kind: JobKind,
        payload: Option<&str>,
    ) -> Result<Job, DbError> {
        let job = repo::ensure_job_with_payload(&self.db, meeting_id, kind, payload).await?;
        self.announce(&job);
        self.wake.notify_one();
        Ok(job)
    }

    /// Cancel a job. Already finished is success.
    pub async fn cancel(&self, job_id: &str) -> Result<(), DbError> {
        let Some(job) = repo::get_job(&self.db, job_id).await? else {
            return Err(DbError::NotFound(format!("job {job_id}")));
        };
        if job.status.is_terminal() {
            return Ok(());
        }
        if let Ok(running) = self.running.lock() {
            if let Some((id, cancel)) = running.as_ref() {
                if id == job_id {
                    cancel.cancel();
                }
            }
        }
        repo::set_job_status(&self.db, job_id, JobStatus::Cancelled, None).await?;
        if let Some(updated) = repo::get_job(&self.db, job_id).await? {
            self.announce(&updated);
        }
        Ok(())
    }

    /// Put a failed, cancelled or parked job back in the queue. It resumes from
    /// whatever it already committed.
    pub async fn retry(&self, job_id: &str) -> Result<(), DbError> {
        let Some(job) = repo::get_job(&self.db, job_id).await? else {
            return Err(DbError::NotFound(format!("job {job_id}")));
        };
        if matches!(job.status, JobStatus::Running) {
            return Ok(());
        }
        repo::set_job_status(&self.db, job_id, JobStatus::Queued, None).await?;
        if let Some(updated) = repo::get_job(&self.db, job_id).await? {
            self.announce(&updated);
        }
        self.wake.notify_one();
        Ok(())
    }

    /// A recording is starting. Park everything, keep it resumable.
    pub async fn preempt(&self) -> Result<(), DbError> {
        self.blocked.store(true, Ordering::SeqCst);
        if let Ok(running) = self.running.lock() {
            if let Some((id, cancel)) = running.as_ref() {
                tracing::info!(job = %id, "parking background work for a recording");
                cancel.preempt();
            }
        }
        let parked = repo::pause_active_jobs(&self.db).await?;
        if parked > 0 {
            tracing::info!(parked, "background work parked");
        }
        Ok(())
    }

    /// The recording finished. Let the parked work continue.
    pub async fn release(&self) -> Result<(), DbError> {
        self.blocked.store(false, Ordering::SeqCst);
        let resumed = repo::resume_paused_jobs(&self.db).await?;
        if resumed > 0 {
            tracing::info!(resumed, "background work picked back up");
        }
        self.wake.notify_one();
        Ok(())
    }

    /// True while a recording holds the machine.
    pub fn is_blocked(&self) -> bool {
        self.blocked.load(Ordering::SeqCst)
    }

    /// Id of the job running right now, if any.
    pub fn running_job(&self) -> Option<Id> {
        self.running
            .lock()
            .ok()
            .and_then(|r| r.as_ref().map(|(id, _)| id.clone()))
    }

    /// Stop the loop; used on the way out.
    pub fn shutdown(&self) {
        self.stopped.store(true, Ordering::SeqCst);
        if let Ok(running) = self.running.lock() {
            if let Some((_, cancel)) = running.as_ref() {
                cancel.preempt();
            }
        }
        self.wake.notify_one();
    }

    fn announce(&self, job: &Job) {
        self.ports
            .events
            .emit(UiEvent::JobProgress(JobProgressPayload {
                job: job.clone(),
                label: Some(label_for(job.kind).to_string()),
            }));
    }

    async fn run_loop(self: Arc<Self>) {
        loop {
            if self.stopped.load(Ordering::SeqCst) {
                return;
            }
            if self.blocked.load(Ordering::SeqCst) {
                self.wait(BLOCKED_POLL).await;
                continue;
            }
            match repo::next_queued_job(&self.db).await {
                Ok(Some(job)) => {
                    self.execute(job).await;
                }
                Ok(None) => self.wait(IDLE_POLL).await,
                Err(error) => {
                    tracing::warn!(%error, "could not read the work queue");
                    self.wait(IDLE_POLL).await;
                }
            }
        }
    }

    async fn wait(&self, longest: Duration) {
        tokio::select! {
            _ = self.wake.notified() => {}
            _ = tokio::time::sleep(longest) => {}
        }
    }

    /// Run one job to completion (or to its interruption) and record what
    /// happened. Public so a recovery path can force one through.
    pub async fn execute(&self, job: Job) {
        let cancel = Cancel::new();
        if let Ok(mut running) = self.running.lock() {
            *running = Some((job.id.clone(), cancel.clone()));
        }
        // A recording may have started in the gap between picking this job off
        // the queue and registering it here — [`JobRuntime::preempt`] would have
        // seen `running == None`, so nothing told this job to stop. Check now
        // that the cancel handle is reachable, and leave the row parked: the
        // recording has absolute priority (DESIGN §3), and `release()` puts it
        // back in the queue with its progress intact.
        if self.blocked.load(Ordering::SeqCst) {
            if let Ok(mut running) = self.running.lock() {
                *running = None;
            }
            if let Err(error) =
                repo::set_job_status(&self.db, &job.id, JobStatus::Paused, None).await
            {
                tracing::warn!(%error, "could not park work for a recording");
            }
            if let Ok(Some(parked)) = repo::get_job(&self.db, &job.id).await {
                self.announce(&parked);
            }
            tracing::info!(job = %job.id, "a recording claimed the machine before this could start");
            return;
        }
        if let Err(error) = repo::set_job_status(&self.db, &job.id, JobStatus::Running, None).await
        {
            tracing::warn!(%error, "could not mark work as running");
        }

        let mut started = job.clone();
        started.status = JobStatus::Running;
        self.announce(&started);

        let ctx = JobContext {
            job: job.clone(),
            db: self.db.clone(),
            paths: self.paths.clone(),
            asr: self.ports.asr.clone(),
            events: self.ports.events.clone(),
            cancel: cancel.clone(),
            progress: Arc::new(Progress::new(
                self.db.clone(),
                job.clone(),
                self.ports.events.clone(),
            )),
        };

        let outcome = self.ports.executor.run(&ctx).await;
        // A preempt that landed while the handler was mid-flight wins: the job
        // has to stay resumable.
        let outcome = match outcome {
            Ok(()) => Ok(()),
            Err(JobFailure::Cancelled) if cancel.is_preempted() => Err(JobFailure::Preempted),
            other => other,
        };

        let (status, error) = match outcome {
            Ok(()) => (JobStatus::Done, None),
            Err(JobFailure::Preempted) => (JobStatus::Paused, None),
            Err(JobFailure::Cancelled) => (JobStatus::Cancelled, None),
            Err(JobFailure::Failed(message)) => (JobStatus::Failed, Some(message)),
        };

        if let Err(error) = repo::set_job_status(&self.db, &job.id, status, error.as_deref()).await
        {
            tracing::warn!(%error, "could not record how the work ended");
        }
        if let Ok(mut running) = self.running.lock() {
            *running = None;
        }

        if let Ok(Some(finished)) = repo::get_job(&self.db, &job.id).await {
            self.announce(&finished);
        }
        if matches!(status, JobStatus::Done) {
            if let Some(meeting_id) = job.meeting_id.as_deref() {
                self.settle_meeting(meeting_id).await;
            }
        }
    }

    /// A meeting is complete once nothing is queued for it any more.
    async fn settle_meeting(&self, meeting_id: &str) {
        let active = repo::list_jobs(
            &self.db,
            &JobQuery {
                meeting_id: Some(meeting_id.to_string()),
                active_only: Some(true),
                ..Default::default()
            },
        )
        .await
        .unwrap_or_default();
        if !active.is_empty() {
            return;
        }
        let Ok(Some(meeting)) = repo::get_meeting(&self.db, meeting_id).await else {
            return;
        };
        if !matches!(meeting.status, MeetingStatus::Processing) {
            return;
        }
        if let Err(error) =
            repo::set_meeting_status(&self.db, meeting_id, MeetingStatus::Complete).await
        {
            tracing::warn!(%error, "could not mark the meeting finished");
            return;
        }
        self.ports.events.emit(UiEvent::MeetingUpdated(
            crate::events::MeetingUpdatedPayload {
                meeting_id: meeting_id.to_string(),
                status: MeetingStatus::Complete,
                title: Some(meeting.title),
                duration_ms: meeting.duration_ms,
                deleted: false,
            },
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_preempt_cannot_be_downgraded_and_a_cancel_wins() {
        let c = Cancel::new();
        assert!(!c.is_cancelled());
        c.preempt();
        assert!(c.is_cancelled());
        assert!(c.is_preempted());
        assert!(matches!(c.check(), Err(JobFailure::Preempted)));
        // A person cancelling something already parked is still a cancel.
        c.cancel();
        assert!(!c.is_preempted());
        assert!(matches!(c.check(), Err(JobFailure::Cancelled)));
        // Preempt no longer takes over a cancelled job.
        c.preempt();
        assert!(!c.is_preempted());
    }

    #[test]
    fn the_summarize_flag_follows_the_job() {
        let c = Cancel::new();
        let flag = c.as_flag();
        assert!(!flag.is_cancelled());
        c.preempt();
        assert!(flag.is_cancelled());
    }

    #[test]
    fn every_label_is_a_plain_sentence() {
        let banned = [
            "whisper",
            "onnx",
            "vad",
            "diariz",
            "connector",
            "token",
            "model",
            "asr",
            "flac",
        ];
        for kind in [
            JobKind::TranscribeCatchup,
            JobKind::Diarize,
            JobKind::Summarize,
            JobKind::Export,
            JobKind::Download,
            JobKind::Mixdown,
        ] {
            let label = label_for(kind).to_lowercase();
            for word in banned {
                assert!(!label.contains(word), "{label:?} leaks {word:?}");
            }
            assert!(label_for(kind).ends_with('…'));
        }
    }
}
