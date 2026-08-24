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
use crate::events::{JobProgressPayload, NoticeLevel, NoticePayload};
use crate::paths::AppPaths;
use crate::session::ports::{AsrPort, EventBus, EventSink, Ports, UiEvent};
use crate::types::{Channel, Id, Job, JobKind, JobQuery, JobStatus, MeetingStatus, SummaryReq};

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
/// How often the loop re-derives the speech engine's lifecycle from the table,
/// whichever edges did or did not fire.
///
/// The backstop for [`JobRuntime::reconcile`]: every path that changes either
/// fact calls [`JobRuntime::refresh_engine_residency`] itself, and this is what
/// makes that a belt rather than the only thing holding the trousers up.
const RECONCILE_INTERVAL: Duration = Duration::from_secs(30);
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

/// The sentence the UI shows for a stage inside a job. Zero jargon (mantra 2).
///
/// "Setting up" would be true here too, and it is what the whole job is called —
/// which is exactly the problem: a person watching an unmoving bar under a
/// sentence that has not changed in ten minutes has no way to tell working from
/// stuck. This one says the download part is over and something else is
/// finishing, without naming a single thing inside the machine.
pub fn phase_label_for(phase: crate::events::JobPhase) -> &'static str {
    match phase {
        crate::events::JobPhase::PreparingEngine => "Finishing one-time setup…",
    }
}

/// The sentence the UI shows while a job runs. Zero jargon (mantra 2).
pub fn label_for(kind: JobKind) -> &'static str {
    match kind {
        JobKind::TranscribeCatchup => "Catching up on the last few minutes…",
        JobKind::Diarize => "Working out who said what…",
        JobKind::Summarize => "Writing your recap…",
        JobKind::Export => "Saving your file…",
        // Two phases, one job: the bytes arrive, then they are made ready to
        // use on this particular machine (which on Apple silicon is not
        // instant — see `download`). "Setting up" is true of both; "downloading"
        // would stop being true halfway through.
        JobKind::Download => "Setting up what Echo needs to understand speech…",
        JobKind::Mixdown => "Getting the recording ready to play back…",
        // Deliberately not "loading" and not "compiling". What is happening is
        // that a new set of weights is being made ready for this particular
        // machine, once; what a person needs to know is that Echo is not able
        // to write anything down until it finishes.
        JobKind::PrepareEngine => "Getting Echo ready to understand speech…",
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
            phase: None,
        }));
    }

    /// This job has moved on to a stage of its own, with no fraction to report.
    ///
    /// Progress in the table is left exactly where it was — the stage is a fact
    /// about now, not a rewind — but the event carries no number, because there
    /// is no honest one to carry and a bar frozen at 100% for a quarter of an
    /// hour reads as broken.
    pub fn phase(&self, phase: crate::events::JobPhase) {
        let mut job = self.job.clone();
        job.status = JobStatus::Running;
        job.progress = None;
        self.events.emit(UiEvent::JobProgress(JobProgressPayload {
            label: Some(phase_label_for(phase).to_string()),
            job,
            phase: Some(phase),
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
                JobKind::PrepareEngine => prepare_engine(ctx).await,
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
/// How much work is outstanding for meetings: queued, running or parked.
///
/// The other half of the engine's lifecycle rule (see
/// [`super::engine_stays_resident`]). Only work that belongs to a meeting counts:
/// a download has nothing to do with the speech engine, and holding 1.6 GB while
/// one runs would be exactly the waste mantra 1 is about.
///
/// Errors count as "there is something": being wrong the other way would unload
/// the engine in the middle of a meeting's work because one query failed.
pub async fn outstanding_meeting_jobs(db: &Db) -> usize {
    match repo::list_jobs(
        db,
        &JobQuery {
            active_only: Some(true),
            ..Default::default()
        },
    )
    .await
    {
        Ok(jobs) => jobs.iter().filter(|job| job.meeting_id.is_some()).count(),
        Err(error) => {
            tracing::debug!(%error, "could not read the work queue; assuming there is work");
            1
        }
    }
}

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

    // The engine may not be up yet, and on Apple silicon the first load after a
    // model changes compiles the encoder for this machine — minutes, with no
    // fraction to report. Named as its own stage, because "Catching up on the
    // transcript" over a still bar is what a person read for eighteen minutes
    // while this was what was happening (field report of 2026-08-21).
    //
    // Only when it really is not up, though. After an ordinary meeting the
    // weights are still in memory and this returns in microseconds, and
    // announcing a stage for that puts a pill on screen that blinks once at the
    // end of every meeting for no reason anybody could name.
    if !ctx.asr.is_loaded() {
        ctx.progress.phase(crate::events::JobPhase::PreparingEngine);
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
    ctx.events.emit(UiEvent::SpeakersUpdated(speakers_event(
        &meeting_id,
        &result,
    )));
    if let Some(notice) = voices_short_notice(&meeting_id, &result) {
        tracing::info!(
            meeting = %meeting_id,
            asked = ?result.voices_asked,
            found = result.voices_found,
            "the meeting holds fewer separable voices than the count asks for"
        );
        ctx.events.emit(UiEvent::Notice(notice));
    }
    // The pass is the one thing that changes the remembered voices without
    // anybody clicking: it links a voice (so somebody was "last heard" just
    // now), and it re-embeds a profile the network moved on from (so the quiet
    // "Echo is refreshing" note has stopped being true). Settings > People is
    // event-driven and would otherwise sit on a stale answer until something
    // else happened to it. Silent when nobody is enrolled, which is every
    // machine until somebody says "remember this voice".
    match crate::diarize::people::list_people(&ctx.db).await {
        Ok(people) if !people.is_empty() => {
            ctx.events.emit(UiEvent::PeopleUpdated(
                crate::events::PeopleUpdatedPayload { people },
            ));
        }
        Ok(_) => {}
        Err(error) => tracing::debug!(%error, "could not read back the remembered voices"),
    }

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

/// The speaker rows the pass just wrote, as the event every open view reads.
///
/// Straight off the pass rather than out of a second query, so the chips and
/// the number beside them cannot disagree.
fn speakers_event(
    meeting_id: &str,
    result: &crate::diarize::DiarizationResult,
) -> crate::events::SpeakersUpdatedPayload {
    crate::events::SpeakersUpdatedPayload {
        meeting_id: meeting_id.to_string(),
        speakers: result.speakers.clone(),
        people_count: result.people_count,
        people_count_is_override: result.people_count_is_override,
        voices_found: Some(result.voices_found),
        alternative_count: result
            .choice
            .as_ref()
            .and_then(|choice| choice.runner_up.map(|(count, _)| count as u32)),
    }
}

/// Say it out loud when the recording does not hold as many separable voices as
/// the person asked for — and say nothing at all otherwise.
///
/// Echo used to answer a shortfall by making the difference up in empty speaker
/// rows, which the person then met in the naming dialog as voices with nothing
/// behind them (2026-08-24). The number they typed is left exactly as it is:
/// they were in the meeting and Echo was not, so this states what Echo can hear
/// and asks for nothing.
fn voices_short_notice(
    meeting_id: &str,
    result: &crate::diarize::DiarizationResult,
) -> Option<NoticePayload> {
    let asked = result.voices_asked?;
    if result.voices_found >= asked {
        return None;
    }
    Some(NoticePayload {
        level: NoticeLevel::Info,
        message: voices_short_message(result.voices_found),
        persistent: false,
        meeting_id: Some(meeting_id.to_string()),
        // One meeting, one such message: a re-run replaces it rather than
        // stacking a second copy of the same sentence.
        tag: Some("voicesShort".into()),
    })
}

/// What Echo says when it could not find as many voices as it was asked for.
///
/// "Distinct" is doing the work — it says the voices are in there and Echo
/// cannot tell them apart, which is the true shape of the problem on a
/// recording of a room. No jargon, no exclamation mark, nothing to press.
fn voices_short_message(found: u32) -> String {
    match found {
        0 => "Echo can't tell any voices apart in this recording.".into(),
        1 => "Echo can only hear one voice clearly in this recording.".into(),
        n => format!("Echo can only hear {n} distinct voices in this recording."),
    }
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

/// Whether the one gentle "recaps need setting up" notice has already gone out
/// this run. Automatic recaps happen after every meeting, and someone who
/// skipped that step in onboarding must not be nagged after every one of them.
static RECAP_SETUP_NOTICE_SENT: AtomicBool = AtomicBool::new(false);

/// Is a backend actually set up to write recaps with?
///
/// Cheap and local: no network probe and no key read beyond "is one stored". The
/// automatic recap uses this to decide between doing the work and stepping
/// quietly aside; a recap the person explicitly asked for still runs and still
/// reports what it needs.
async fn recap_backend_ready(db: &Db, provider: crate::types::Provider) -> bool {
    use crate::types::Provider;
    match provider {
        Provider::OnThisComputer => repo::get_setting(db, crate::settings::keys::OLLAMA_MODEL)
            .await
            .ok()
            .flatten()
            .is_some_and(|model| !model.trim().is_empty()),
        Provider::Gemini => crate::secrets::has(crate::secrets::accounts::GEMINI_API_KEY)
            .await
            .unwrap_or(false),
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
    // No payload means nobody pressed anything: this is the recap that follows a
    // meeting on its own.
    let asked_for_by_hand = requested.is_some();
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

    // Nothing is set up to write recaps with (onboarding skipped, most likely).
    // An automatic recap steps aside quietly: recording and transcripts work
    // perfectly well without a recap backend (DESIGN §0.4), so this is not a
    // failure and must not turn into a red mark after every meeting.
    let provider = req.provider.unwrap_or(settings.summary_provider);
    if !asked_for_by_hand && !recap_backend_ready(&ctx.db, provider).await {
        tracing::info!(
            meeting = %meeting_id,
            "skipping the automatic recap: nothing is set up to write one"
        );
        if !RECAP_SETUP_NOTICE_SENT.swap(true, Ordering::SeqCst) {
            ctx.events
                .emit(UiEvent::Notice(crate::events::NoticePayload {
                    level: NoticeLevel::Info,
                    message:
                        "Echo can write recaps of your meetings once you pick who writes them, \
                          in Settings."
                            .into(),
                    persistent: false,
                    // Deliberately no meeting: the UI turns a notice that names one
                    // into a link to it, and what this one asks for is in Settings.
                    meeting_id: None,
                    tag: Some("recapNeedsSetup".into()),
                }));
        }
        ctx.progress.set(1.0).await;
        return Ok(());
    }

    ctx.progress.set(0.1).await;
    // The recap and its task list are written together, in one place
    // (`summarize_meeting_with_actions`). This job used to ask for the recap and
    // then run the extraction pass again itself, which meant two model calls —
    // four when both took their repair retry — and a second, worse task list
    // overwriting the first.
    let outcome =
        match crate::summarize::summarize_meeting_with_actions(&ctx.db, &req, ctx.cancel.as_flag())
            .await
        {
            Ok(outcome) => outcome,
            // A meeting with nothing written down cannot have a recap. When
            // nobody asked for one, that is simply the end of it.
            Err(crate::summarize::SummarizeError::NoTranscript) if !asked_for_by_hand => {
                tracing::info!(meeting = %meeting_id, "no words to write a recap from");
                ctx.progress.set(1.0).await;
                return Ok(());
            }
            Err(error) => return Err(summarize_failure(&ctx.cancel, error)),
        };
    let summary = outcome.summary;

    ctx.progress.set(0.9).await;
    // Read the list back rather than forwarding the one in hand. An extraction
    // that fell over hands back an empty list while the previous revision's
    // items are still in the table, and telling the screen "no tasks" when there
    // are some is worse than telling it nothing. The recap itself is never
    // failed over its task list — `summarize_meeting_with_actions` logs why one
    // is missing.
    match repo::list_action_items(&ctx.db, &meeting_id).await {
        Ok(items) => ctx.events.emit(UiEvent::ActionItemsUpdated(
            crate::events::ActionItemsUpdatedPayload {
                meeting_id: meeting_id.clone(),
                items,
            },
        )),
        Err(error) => tracing::warn!(%error, "could not read the task list back"),
    }

    ctx.events
        .emit(UiEvent::SummaryReady(crate::events::SummaryReadyPayload {
            meeting_id: meeting_id.clone(),
            summary_id: summary.id,
        }));
    // The recap arrives minutes after the meeting ended, when the person has
    // moved on to something else, so it says so out loud once.
    ctx.events
        .emit(UiEvent::Notice(crate::events::NoticePayload {
            level: NoticeLevel::Info,
            message: "Your recap is ready.".into(),
            persistent: false,
            meeting_id: Some(meeting_id),
            tag: Some("recapReady".into()),
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

    // The download that just finished may have been the last file an upgrade
    // was waiting for. Reconciling here is what makes the switch happen at the
    // moment it becomes safe, rather than on the next launch or the next time
    // the UI happens to ask (see `crate::asr::reconcile`).
    //
    // Deliberately after the download succeeded and before progress reaches 1:
    // the old weights are gone by the time anything reads "done".
    //
    // No "is a recording running" guard here, unlike
    // `SessionManager::ensure_speech_current`: a recording parks this job
    // (recording has absolute priority — see `JobRuntime::preempt`), so
    // reaching this line at all means nothing is being listened to.
    match crate::asr::models::reconcile(&ctx.db, &ctx.paths).await {
        Ok(plan) if plan.switching() => {
            tracing::info!(
                serving = plan.serving.unwrap_or("?"),
                "Echo switched to the speech model it wants"
            );
            ctx.events
                .emit(UiEvent::Notice(crate::events::NoticePayload {
                    level: crate::events::NoticeLevel::Info,
                    message: "Echo upgraded how it understands speech.".into(),
                    persistent: false,
                    meeting_id: None,
                    tag: Some("speechUpgraded".into()),
                }));
        }
        Ok(_) => {}
        // Not a download failure: the bytes arrived and are installed. The
        // tidying up is retried at the next launch or readiness check.
        Err(error) => tracing::warn!(%error, "could not tidy up older speech weights"),
    }

    // Load the weights once, here, while the person is already waiting on a
    // download that says it happens once.
    //
    // This is not a warm cache for its own sake. On Apple silicon the encoder
    // companion is compiled *for this machine* the first time it is loaded, and
    // for the full model that took eighteen minutes on the machine this was
    // measured on, against three seconds once the OS had cached the result — a
    // genuine one-off, and a big one. Paid here it is part
    // of "Downloading what Echo needs to understand speech". Paid on first use
    // it is a meeting whose captions do not appear until it is over. The audio
    // would still be on disk and the transcript would still arrive from the
    // catch-up pass (mantra 3), but "no captions for the first meeting after an
    // update" is not something to leave in.
    //
    // Failing here is not a failed download: the bytes are installed and
    // verified, and the next load will try again.
    //
    // Announced as its own stage first. Everything above this line has a
    // fraction; nothing below it does, and the eighteen minutes are all below
    // it (field report of 2026-08-21).
    ctx.progress.phase(crate::events::JobPhase::PreparingEngine);
    if let Err(error) = ctx.asr.prewarm().await {
        tracing::warn!(%error, "the new weights did not load on the first attempt");
    }

    ctx.progress.set(1.0).await;
    Ok(())
}

/// What a setup row that was interrupted by the process going down is left
/// saying. Not jargon and not a diagnosis: nobody can be told what killed the
/// app from inside the app that was killed.
pub(crate) const SETUP_INTERRUPTED: &str =
    "Echo closed while it was getting ready to understand speech.";

/// Whether the record of an attempt at the one-time setup should be taken back,
/// given how the load ended and how long it lasted.
///
/// Pure, because this is the rule the whole one-attempt policy turns on and it
/// is worth being able to read on its own.
///
/// * a load that ground away and *then* failed is what the marker exists for.
///   Something on this machine cannot finish that compile, and starting it again
///   at every launch would cost a quarter of an hour a time and get no further.
/// * a load that failed inside the time an ordinary load takes never reached the
///   compile — there were no weights where they should be, the disk was full for
///   a moment, the engine would not initialise. Nothing was protected by
///   remembering it, and remembering it would leave this machine unwarmed for
///   good on the strength of one bad second.
/// * a recording taking the machine is not a failure at all: the row is parked
///   and comes back to finish what it started.
///
/// The half-minute is [`crate::asr::engine::LIKELY_COMPILED_AT`], the same line
/// the engine draws between a compile and a slow disk.
fn attempt_is_worth_forgetting(outcome: &JobFailure, spent: std::time::Duration) -> bool {
    matches!(outcome, JobFailure::Failed(_)) && spent < crate::asr::engine::LIKELY_COMPILED_AT
}

/// The one-time setup a set of weights needs on this machine, paid on purpose
/// and in the open instead of by the next meeting that happens to start.
///
/// Queued by [`crate::session::SessionManager::ensure_speech_current`] when the
/// weights that are serving have never been through a load here — which is how
/// a model that arrived without a download job (a restore, a copy between
/// machines, an out-of-band unpack) got as far as a real meeting on 2026-08-24
/// and spent its first sixteen minutes compiling.
async fn prepare_engine(ctx: &JobContext) -> Result<(), JobFailure> {
    // Which weights this row is for travels with it. A row that has outlived a
    // catalog change asks the disk again rather than recording an attempt
    // against a name nothing will ever load.
    let requested: Option<String> = repo::get_job_payload(&ctx.db, &ctx.job.id)
        .await
        .ok()
        .flatten()
        .and_then(|raw| serde_json::from_str::<String>(&raw).ok())
        .map(|name| name.trim().to_string())
        .filter(|name| !name.is_empty());
    let file_name = match requested {
        Some(name) => Some(name),
        None => crate::asr::models::installed_speech_file_name(&ctx.db)
            .await
            .unwrap_or(None),
    };
    let Some(file_name) = file_name else {
        // Nothing installed to get ready. Not a failure: the download job is
        // what this install is waiting for, and it warms the weights itself.
        tracing::info!("nothing to get ready yet: there are no speech weights installed");
        ctx.progress.set(1.0).await;
        return Ok(());
    };

    // Recorded *before* the load, never after. The compile inside it is minutes
    // of heavy work, and anything that takes the process down in the middle of
    // it would otherwise be repeated at every launch for as long as it keeps
    // happening — see `models::mark_warm_attempted`. What the attempt turns out
    // to have cost decides whether it is kept; that is settled below, once the
    // load has come back one way or the other.
    if let Err(error) = crate::asr::models::mark_warm_attempted(&ctx.db, &file_name).await {
        // Costs a repeat of this job at the next launch, nothing more.
        tracing::debug!(%error, "could not remember that the one-time setup was started");
    }

    // Everything from here has no fraction to report, so it says what it is
    // instead of leaving a bar somewhere it will sit for a quarter of an hour.
    ctx.progress.phase(crate::events::JobPhase::PreparingEngine);
    tracing::info!(model = %file_name, "getting the speech engine ready for this machine");

    // No cancel check around the load, and none after it, on purpose.
    //
    // The only thing that interrupts this job is a recording starting, and a
    // recording needs exactly the load that is already in flight: the engine
    // runs one load on one thread, so a meeting that preempted this would queue
    // behind the very compile it just abandoned the row for, and arrive no
    // sooner. Parking the row would only mean doing the bookkeeping twice, and
    // leaving it half-marked. Sixteen minutes is sixteen minutes whoever is
    // waiting for it (incident of 2026-08-24); the difference this job makes is
    // that nobody is in a meeting while they pass.
    let started = std::time::Instant::now();
    if let Err(error) = ctx.asr.prewarm().await {
        let failure = asr_failure(&ctx.cancel, error);
        let spent = started.elapsed();
        let spent_ms = spent.as_millis() as u64;
        // What the attempt cost decides whether it stands — see
        // `attempt_is_worth_forgetting`, which is that rule and nothing else.
        if attempt_is_worth_forgetting(&failure, spent) {
            if let Err(error) = crate::asr::models::forget_warm_attempt(&ctx.db, &file_name).await {
                tracing::debug!(%error, "could not take back the record of the attempt");
            }
            tracing::warn!(
                model = %file_name,
                spent_ms,
                "getting the speech engine ready failed before it had started; the next launch \
                 will try again"
            );
        } else if matches!(failure, JobFailure::Failed(_)) {
            tracing::warn!(
                model = %file_name,
                spent_ms,
                "getting the speech engine ready did not finish; it will not be started again on \
                 its own"
            );
        } else {
            // Stopped rather than failed: the row is parked or cancelled, and
            // the attempt it recorded stands for whenever it picks back up.
            tracing::info!(
                model = %file_name,
                spent_ms,
                "getting the speech engine ready stopped short"
            );
        }
        return Err(failure);
    }

    tracing::info!(model = %file_name, "the speech engine is ready for this machine");
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
    // One line for every recap that did not finish, before the taxonomy below
    // turns it into a sentence for the screen. The screen copy is deliberately
    // free of the service's own words, so this is the only place the internal
    // reason is written down — and every arm needs it, not just the catch-all.
    if !matches!(error, S::Cancelled) {
        tracing::warn!(error = %error, "the recap stopped");
    }
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
        // Same taxonomy as the command path (`UiError::from<SummarizeError>`), in
        // job-sized sentences. Every one of these used to land on "Echo couldn't
        // write the recap this time", which is true of all of them and useful
        // about none of them.
        S::Rejected(_) => JobFailure::failed(
            "That service wouldn't accept the key Echo has saved for it. Open Settings and paste \
             it again.",
        ),
        // The key was accepted and the request was not. Saying "paste the key
        // again" here is a wild goose chase (review of 2026-08-20, finding 5).
        S::RequestRefused { .. } => JobFailure::failed(
            "That service wouldn't write a recap from this request. Your recording and transcript \
             are safe, and you can pick something else to write recaps in Settings.",
        ),
        S::QuotaExhausted => JobFailure::failed(
            "That service won't take any more requests just now. Your recording and transcript \
             are safe — try the recap again later.",
        ),
        S::Blocked { .. } => JobFailure::failed(
            "That service wouldn't write a recap from this meeting. Your transcript is safe, and \
             you can pick something else to write recaps in Settings.",
        ),
        S::EmptyReply => {
            JobFailure::failed("That service sent back an empty recap. You can try again.")
        }
        S::MalformedReply => JobFailure::failed(
            "Echo couldn't make sense of what that service sent back. You can try again.",
        ),
        S::Timeout => JobFailure::failed("That took too long. You can try writing it again."),
        // Already a finished sentence naming the model and what to do about it,
        // and "try again" would be a lie: the same setting would pick the same
        // retired model. It goes on the job as written.
        ref not_found @ S::ModelNotFound { .. } => JobFailure::failed(not_found.to_string()),
        _ => JobFailure::failed("Echo couldn't write the recap this time. You can try again."),
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
    /// Asked "is a capture live right now?" once the session spine has wired
    /// itself up.
    ///
    /// [`JobRuntime::blocked`] is close but not the same thing: it is a flag
    /// somebody has to remember to clear, and the engine's lifecycle is not
    /// something to hang on a flag (review of 2026-08-20, finding 2). This asks
    /// the capture state machine, which is the fact itself.
    capturing: std::sync::Mutex<Option<Arc<dyn Fn() -> bool + Send + Sync>>>,
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
            capturing: std::sync::Mutex::new(None),
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
        // Work for a meeting is a reason to keep the speech engine, whether or
        // not this loop gets to it in the next second.
        self.refresh_engine_residency().await;
        self.wake.notify_one();
        Ok(job)
    }

    /// Point the residency rule at the capture state machine. Called once, when
    /// the session spine builds itself.
    pub(crate) fn watch_capture(&self, is_live: Arc<dyn Fn() -> bool + Send + Sync>) {
        *self.capturing.lock().expect("capture hook poisoned") = Some(is_live);
    }

    /// Is a capture running right now?
    fn capture_is_live(&self) -> bool {
        let hook = self
            .capturing
            .lock()
            .expect("capture hook poisoned")
            .clone();
        match hook {
            Some(is_live) => is_live(),
            // Before the spine wires itself up — and in tests that drive this
            // runtime on its own — the parking flag is the best fact there is.
            None => self.blocked.load(Ordering::SeqCst),
        }
    }

    /// Hold the speech engine while a recording or a meeting's work is
    /// outstanding, and let the grace period start once neither is true.
    ///
    /// **The** recompute: every caller — queue, cancel, retry, a job finishing, a
    /// recording starting or stopping, the periodic reconcile — asks this one
    /// question and gets the answer from the same two facts, both read fresh.
    /// Nothing accumulates, so no edge can be "the one that was missed": the
    /// worst a missed call costs is being right one tick late
    /// (review of 2026-08-20, finding 2).
    pub(crate) async fn refresh_engine_residency(&self) {
        let capturing = self.capture_is_live();
        let outstanding = outstanding_meeting_jobs(&self.db).await;
        self.ports
            .asr
            .hold_resident(super::engine_stays_resident(capturing, outstanding));
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
        // Cancelling the last thing a meeting was waiting for is exactly as much
        // of an edge as queueing the first one, and it used to be the one edge
        // nobody reported: the engine stayed held with nothing left to do and no
        // later edge that could ever clear it.
        self.refresh_engine_residency().await;
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
        // And the mirror image: a meeting with work in the queue again needs the
        // engine again, whether or not the loop reaches the row this second.
        self.refresh_engine_residency().await;
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
        // Also the launch path, where work left over from a crash is un-parked:
        // whatever is outstanding now decides whether the engine is held.
        self.refresh_engine_residency().await;
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
                phase: None,
            }));
    }

    async fn run_loop(self: Arc<Self>) {
        let mut last_reconcile = Instant::now() - RECONCILE_INTERVAL;
        loop {
            if self.stopped.load(Ordering::SeqCst) {
                return;
            }
            if last_reconcile.elapsed() >= RECONCILE_INTERVAL {
                last_reconcile = Instant::now();
                self.reconcile().await;
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

    /// The lifecycle backstop, run on the loop's own tick.
    ///
    /// Two things, both derived from the table rather than from a flag: a row
    /// claiming to be running that nothing is running is put back in the queue,
    /// and the engine's residency is recomputed from what is left. Together they
    /// are the answer to "what if an edge is missed": nothing here accumulates,
    /// so a wrong answer lasts one tick rather than until Echo is restarted
    /// (review of 2026-08-20, finding 2).
    pub(crate) async fn reconcile(&self) {
        self.requeue_abandoned_rows().await;
        self.refresh_engine_residency().await;
    }

    /// A row marked `running` that this loop is not running belongs to nobody.
    ///
    /// It can only happen if a status write failed or something died between
    /// marking the row and running it — and while it sits there it counts as
    /// outstanding work forever, which is 1.6 GB of weights held for a job that
    /// will never finish. Queued again, it either runs or fails, and either way
    /// it reaches a terminal state.
    ///
    /// Safe because this loop is the only thing that runs jobs and it never gets
    /// here while one is in flight: [`JobRuntime::execute`] is awaited from the
    /// same task, and the id it registers is skipped anyway.
    async fn requeue_abandoned_rows(&self) {
        if self.blocked.load(Ordering::SeqCst) {
            // A recording parks rows on purpose. Nothing here is abandoned.
            return;
        }
        let mine = self.running_job();
        let rows = repo::list_jobs(
            &self.db,
            &JobQuery {
                status: Some(JobStatus::Running),
                ..Default::default()
            },
        )
        .await
        .unwrap_or_default();
        for job in rows {
            if mine.as_deref() == Some(job.id.as_str()) {
                continue;
            }
            tracing::warn!(
                job = %job.id,
                kind = ?job.kind,
                "work was marked as running with nothing running it; queueing it again"
            );
            if let Err(error) =
                repo::set_job_status(&self.db, &job.id, JobStatus::Queued, None).await
            {
                tracing::warn!(%error, "could not put abandoned work back in the queue");
                continue;
            }
            if let Ok(Some(requeued)) = repo::get_job(&self.db, &job.id).await {
                self.announce(&requeued);
            }
            self.wake.notify_one();
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
        // This may have been the meeting's last job. If it was, and nothing is
        // being recorded, this is the moment the engine's grace period starts
        // (mantra 1's amendment of 2026-08-20).
        self.refresh_engine_residency().await;
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

    /// The automatic recap has to be able to tell "nothing is set up" from "this
    /// went wrong", without a network call. Only the on-this-computer branch is
    /// exercised here: the other one asks the OS keychain, which a test must not.
    #[tokio::test]
    async fn a_local_backend_counts_as_ready_only_once_something_is_chosen() {
        let db = crate::db::connect_in_memory().await.unwrap();
        assert!(
            !recap_backend_ready(&db, crate::types::Provider::OnThisComputer).await,
            "a skipped setup is not ready"
        );

        repo::set_setting(&db, crate::settings::keys::OLLAMA_MODEL, "   ")
            .await
            .unwrap();
        assert!(
            !recap_backend_ready(&db, crate::types::Provider::OnThisComputer).await,
            "an empty choice is no choice"
        );

        repo::set_setting(&db, crate::settings::keys::OLLAMA_MODEL, "something-local")
            .await
            .unwrap();
        assert!(recap_backend_ready(&db, crate::types::Provider::OnThisComputer).await);
    }

    /// Build a `JobContext` around a real database and a stub speech engine, so
    /// a handler can be run on its own without the whole runtime.
    async fn context_for(
        db: &Db,
        job: Job,
    ) -> (JobContext, Arc<super::super::mock::CollectingEvents>) {
        let events = crate::session::ports::EventBus::new();
        let seen = Arc::new(super::super::mock::CollectingEvents::default());
        events.set(seen.clone());
        let progress = Arc::new(Progress::new(db.clone(), job.clone(), events.clone()));
        (
            JobContext {
                job,
                db: db.clone(),
                paths: crate::paths::AppPaths::rooted_at(
                    std::env::temp_dir().join("echo-jobs"),
                    None,
                ),
                asr: Arc::new(super::super::mock::MockAsr::new()),
                events,
                cancel: Cancel::new(),
                progress,
            },
            seen,
        )
    }

    /// The regression this batch was for: the summarize job asked for the recap,
    /// which writes the task list, and then ran the task-list pass *again*
    /// itself. Two model calls became four (each pass has a repair retry), and
    /// the second, worse list overwrote the first. One recap call, one task-list
    /// call, and that is all.
    #[tokio::test]
    async fn the_summarize_job_asks_for_the_task_list_exactly_once() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        // One reply that reads as a recap and as the task-list JSON, so the
        // count is the only thing under test.
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                "{\"message\":{\"content\":\"{\\\"items\\\":[{\\\"description\\\":\\\"Send the \
                 invoice\\\"}]}\"},\"done\":true}\n",
                "application/x-ndjson",
            ))
            .mount(&server)
            .await;

        let db = crate::db::connect_in_memory().await.unwrap();
        repo::set_setting(&db, crate::settings::keys::OLLAMA_BASE_URL, &server.uri())
            .await
            .unwrap();
        repo::set_setting(&db, crate::settings::keys::OLLAMA_MODEL, "test-model")
            .await
            .unwrap();

        let meeting = repo::create_meeting(&db, "Weekly sync", "/tmp/echo-test", None)
            .await
            .unwrap();
        repo::insert_segments(
            &db,
            &[crate::types::SegmentDraft {
                meeting_id: meeting.id.clone(),
                t_start_ms: 0,
                t_end_ms: 4_000,
                channel: crate::types::Channel::Mic,
                speaker_id: None,
                text: "We ship on Friday.".into(),
                language: Some("en".into()),
                avg_confidence: Some(0.9),
                revision: 1,
                is_final: true,
                model_name: None,
                model_revision: None,
            }],
        )
        .await
        .unwrap();

        let job = repo::ensure_job(&db, Some(&meeting.id), JobKind::Summarize)
            .await
            .unwrap();
        let (ctx, seen) = context_for(&db, job).await;
        summarize(&ctx).await.expect("the recap should be written");

        let calls = server.received_requests().await.unwrap();
        assert_eq!(
            calls.len(),
            2,
            "one recap call and one task-list call, not two of each"
        );

        // Stored once, and the screen was told about it once.
        let stored = repo::list_action_items(&db, &meeting.id).await.unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].description, "Send the invoice");
        assert_eq!(seen.count(crate::events::ACTION_ITEMS_UPDATED), 1);
        assert_eq!(seen.count(crate::events::SUMMARY_READY), 1);
    }

    /// Every reason a recap can fail says something different, and none of them
    /// says "something went wrong". Each of these used to collapse into the
    /// catch-all sentence, which told the person nothing they could act on.
    #[test]
    fn each_way_a_recap_can_fail_says_its_own_thing() {
        use crate::summarize::SummarizeError as S;

        let cancel = Cancel::new();
        let sentences: Vec<String> = [
            S::Rejected("401".into()),
            S::QuotaExhausted,
            S::Blocked {
                reason: "SAFETY".into(),
            },
            S::EmptyReply,
            S::MalformedReply,
            S::Timeout,
            S::Unreachable("connection refused".into()),
            S::MissingCredential,
            S::NoTranscript,
        ]
        .into_iter()
        .map(|error| match summarize_failure(&cancel, error) {
            JobFailure::Failed(message) => message,
            other => panic!("expected a failure sentence, got {other:?}"),
        })
        .collect();

        for sentence in &sentences {
            assert!(
                !sentence.contains("Echo couldn't write the recap this time"),
                "still falling through to the catch-all: {sentence}"
            );
        }
        let mut unique = sentences.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(
            unique.len(),
            sentences.len(),
            "two different failures are telling the person the same thing"
        );

        // The service's own word for a block never reaches the screen (mantra 2).
        let blocked = match summarize_failure(
            &cancel,
            S::Blocked {
                reason: "RECITATION".into(),
            },
        ) {
            JobFailure::Failed(message) => message,
            other => panic!("{other:?}"),
        };
        assert!(!blocked.contains("RECITATION"), "{blocked}");
    }

    /// A result shaped like the end of a speaker pass, so the two pure
    /// functions above can be driven without models or audio.
    fn pass_result(asked: Option<u32>, found: u32) -> crate::diarize::DiarizationResult {
        crate::diarize::DiarizationResult {
            people_count: asked.unwrap_or(found),
            people_count_is_override: asked.is_some(),
            voices_asked: asked,
            voices_found: found,
            ..Default::default()
        }
    }

    /// The 2026-08-24 shape: four people asked for, two voices in the
    /// recording. Echo says so instead of inventing the other two, and the
    /// number the person typed is left exactly as they typed it.
    #[test]
    fn a_count_the_recording_cannot_deliver_is_said_out_loud() {
        let notice = voices_short_notice("meeting-1", &pass_result(Some(4), 2))
            .expect("a shortfall the person can see in the chips has to be said");
        assert_eq!(
            notice.message,
            "Echo can only hear 2 distinct voices in this recording."
        );
        assert_eq!(notice.meeting_id.as_deref(), Some("meeting-1"));
        assert!(!notice.persistent, "one sentence, not a banner to dismiss");

        let payload = speakers_event("meeting-1", &pass_result(Some(4), 2));
        assert_eq!(payload.people_count, 4, "the person's number is theirs");
        assert!(payload.people_count_is_override);
        assert_eq!(
            payload.voices_found,
            Some(2),
            "the dialog needs the true number to annotate the count with"
        );
    }

    #[test]
    fn a_count_that_was_delivered_says_nothing() {
        assert!(voices_short_notice("m", &pass_result(Some(3), 3)).is_none());
        assert!(
            voices_short_notice("m", &pass_result(Some(2), 3)).is_none(),
            "more voices than asked for is not a shortfall"
        );
        assert!(
            voices_short_notice("m", &pass_result(None, 1)).is_none(),
            "nobody asked for a number, so there is nothing to fall short of"
        );
    }

    #[test]
    fn the_shortfall_sentence_reads_as_one_and_carries_no_jargon() {
        let banned = [
            "cluster",
            "diariz",
            "silhouette",
            "threshold",
            "override",
            "embed",
            "voiceprint",
        ];
        for found in [0u32, 1, 2, 7] {
            let message = voices_short_message(found);
            assert!(
                message.ends_with('.'),
                "{message:?} should read as a sentence"
            );
            assert!(
                !message.contains('!'),
                "{message:?} is shouting at somebody"
            );
            let lower = message.to_lowercase();
            for word in banned {
                assert!(!lower.contains(word), "{message:?} leaks {word:?}");
            }
        }
        // A number a person reads has to agree with itself.
        assert!(voices_short_message(1).contains("one voice"));
        assert!(voices_short_message(2).contains("2 distinct voices"));
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
            JobKind::PrepareEngine,
        ] {
            let label = label_for(kind).to_lowercase();
            for word in banned {
                assert!(!label.contains(word), "{label:?} leaks {word:?}");
            }
            assert!(label_for(kind).ends_with('…'));
        }
        for phase in [crate::events::JobPhase::PreparingEngine] {
            let label = phase_label_for(phase).to_lowercase();
            for word in banned {
                assert!(!label.contains(word), "{label:?} leaks {word:?}");
            }
            assert!(phase_label_for(phase).ends_with('…'));
            // Against every job, not just the download it started life inside:
            // a stage is only worth announcing if it reads differently from
            // whatever sentence is already on screen.
            for kind in [
                JobKind::TranscribeCatchup,
                JobKind::Diarize,
                JobKind::Summarize,
                JobKind::Export,
                JobKind::Download,
                JobKind::Mixdown,
                JobKind::PrepareEngine,
            ] {
                assert_ne!(
                    phase_label_for(phase),
                    label_for(kind),
                    "a stage worth naming has to read differently from the job"
                );
            }
        }
    }

    /// The setup job, end to end: it says which stage it is on, it reports no
    /// fraction while the machine does the one-time work, and the engine comes
    /// up. This is the sixteen minutes of 2026-08-24, paid before a meeting.
    #[tokio::test]
    async fn the_setup_job_names_its_stage_and_reports_no_fraction() {
        let db = crate::db::connect_in_memory().await.unwrap();
        let job = repo::ensure_job_with_payload(
            &db,
            None,
            JobKind::PrepareEngine,
            Some(&serde_json::to_string("some-weights.bin").unwrap()),
        )
        .await
        .unwrap();
        let (ctx, seen) = context_for(&db, job).await;

        prepare_engine(&ctx).await.expect("the setup should finish");

        assert!(
            ctx.asr.is_loaded(),
            "the point of the job is a loaded engine"
        );
        let announced = seen.job_progress();
        let stage = announced
            .iter()
            .find(|p| p.phase == Some(crate::events::JobPhase::PreparingEngine))
            .expect("the stage was announced");
        assert!(
            stage.job.progress.is_none(),
            "there is no honest fraction for a compile, and a still bar reads as broken"
        );
        assert_eq!(
            stage.label.as_deref(),
            Some(phase_label_for(crate::events::JobPhase::PreparingEngine))
        );
        assert_eq!(
            announced.last().map(|p| p.job.progress),
            Some(Some(1.0)),
            "and it finishes"
        );

        // The one attempt is on the record, against these weights by name.
        assert_eq!(
            repo::get_setting(&db, crate::settings::keys::SPEECH_WARM_ATTEMPTED)
                .await
                .unwrap()
                .as_deref(),
            Some("some-weights.bin")
        );
    }

    /// A setup row whose weights, name and all, is what the failing job is for.
    async fn a_setup_job_for(db: &Db, weights: &str) -> Job {
        repo::ensure_job_with_payload(
            db,
            None,
            JobKind::PrepareEngine,
            Some(&serde_json::to_string(weights).unwrap()),
        )
        .await
        .unwrap()
    }

    /// A failure that came back before the compile could have started cost
    /// nothing, so the record of the attempt goes with it: keeping it would
    /// retire this machine's only automatic setup on the strength of one bad
    /// second, and hand the sixteen minutes back to the next real meeting.
    #[tokio::test]
    async fn a_setup_that_failed_before_it_started_is_not_the_one_attempt() {
        let db = crate::db::connect_in_memory().await.unwrap();
        let job = a_setup_job_for(&db, "some-weights.bin").await;
        let (mut ctx, _seen) = context_for(&db, job).await;
        let asr = Arc::new(super::super::mock::MockAsr::new());
        asr.fail_prewarm();
        ctx.asr = asr;

        let outcome = prepare_engine(&ctx).await;
        assert!(
            matches!(outcome, Err(JobFailure::Failed(_))),
            "a failed setup is a row that failed, not a silent nothing: {outcome:?}"
        );
        assert_eq!(
            repo::get_setting(&db, crate::settings::keys::SPEECH_WARM_ATTEMPTED)
                .await
                .unwrap(),
            None,
            "nothing expensive happened, so the next launch is free to try again"
        );
    }

    /// A recording taking the machine is not a failure and settles nothing: the
    /// row is parked, and it keeps the attempt it has already recorded.
    #[tokio::test]
    async fn a_setup_a_recording_took_the_machine_from_is_not_a_failed_attempt() {
        let db = crate::db::connect_in_memory().await.unwrap();
        let job = a_setup_job_for(&db, "some-weights.bin").await;
        let (mut ctx, _seen) = context_for(&db, job).await;
        let asr = Arc::new(super::super::mock::MockAsr::new());
        asr.cancel_prewarm();
        ctx.asr = asr;
        ctx.cancel.preempt();

        let outcome = prepare_engine(&ctx).await;
        assert!(matches!(outcome, Err(JobFailure::Preempted)), "{outcome:?}");
        assert_eq!(
            repo::get_setting(&db, crate::settings::keys::SPEECH_WARM_ATTEMPTED)
                .await
                .unwrap()
                .as_deref(),
            Some("some-weights.bin"),
            "the row comes back for the rest of it; it does not start over"
        );
    }

    /// The rule the one-attempt policy turns on, read on its own: only a load
    /// that lasted long enough to have been the compile is worth remembering,
    /// and only a real failure settles anything at all.
    #[test]
    fn only_a_failure_that_lasted_counts_as_the_one_attempt() {
        use std::time::Duration;
        let failed = JobFailure::failed("nope");
        let long = crate::asr::engine::LIKELY_COMPILED_AT;

        assert!(attempt_is_worth_forgetting(
            &failed,
            Duration::from_millis(40)
        ));
        assert!(attempt_is_worth_forgetting(
            &failed,
            long - Duration::from_millis(1)
        ));
        assert!(!attempt_is_worth_forgetting(&failed, long));
        assert!(!attempt_is_worth_forgetting(
            &failed,
            Duration::from_secs(16 * 60)
        ));

        for kept in [JobFailure::Preempted, JobFailure::Cancelled] {
            assert!(
                !attempt_is_worth_forgetting(&kept, Duration::from_millis(40)),
                "{kept:?}: the load was stopped, not tried and found wanting"
            );
        }
    }

    /// Every meeting ends with a catch-up pass, and after an ordinary one the
    /// weights are still in memory. Announcing a stage for a load that returns
    /// at once puts a pill on screen that blinks for no reason.
    #[tokio::test]
    async fn a_catch_up_with_the_engine_already_up_announces_no_stage() {
        let db = crate::db::connect_in_memory().await.unwrap();
        let meeting = repo::create_meeting(&db, "Weekly sync", "/tmp/echo-test", None)
            .await
            .unwrap();
        let chunk = repo::insert_chunk(
            &db,
            &meeting.id,
            crate::types::Channel::Mic,
            0,
            "/tmp/echo-test/mic-000000.flac",
            0,
            60_000,
        )
        .await
        .unwrap();
        repo::commit_chunk(&db, &chunk, 60_000).await.unwrap();

        let job = repo::ensure_job(&db, Some(&meeting.id), JobKind::TranscribeCatchup)
            .await
            .unwrap();
        let (ctx, seen) = context_for(&db, job).await;
        // The engine this meeting was recorded with is still holding its weights.
        ctx.asr.prewarm().await.unwrap();

        catch_up(&ctx).await.expect("the catch-up should finish");

        assert!(
            seen.job_progress().iter().all(|p| p.phase.is_none()),
            "nothing to announce: the engine was already up"
        );
    }

    /// A stage with no fraction says so, rather than leaving a bar parked at the
    /// last number it had.
    #[tokio::test]
    async fn a_stage_of_a_job_reports_no_fraction_and_names_itself() {
        let db = crate::db::connect_in_memory().await.unwrap();
        let job = Job {
            id: "j1".into(),
            meeting_id: None,
            kind: JobKind::Download,
            status: JobStatus::Running,
            progress: Some(0.4),
            error: None,
            created_at: String::new(),
            updated_at: String::new(),
        };
        let (ctx, seen) = context_for(&db, job).await;
        ctx.progress.set(1.0).await;
        ctx.progress.phase(crate::events::JobPhase::PreparingEngine);

        let announced = seen.job_progress();
        assert_eq!(announced.len(), 2);
        assert_eq!(announced[0].job.progress, Some(1.0));
        assert!(announced[0].phase.is_none());
        assert_eq!(
            announced[1].phase,
            Some(crate::events::JobPhase::PreparingEngine)
        );
        assert!(
            announced[1].job.progress.is_none(),
            "a bar at 100% for the next quarter of an hour reads as broken"
        );
        assert_eq!(
            announced[1].label.as_deref(),
            Some(phase_label_for(crate::events::JobPhase::PreparingEngine))
        );
        assert_eq!(announced[1].job.status, JobStatus::Running);
    }
}
