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
    /// The stage written on the row right now, so the first fraction after one
    /// knows there is something to take back off it.
    stage: tokio::sync::Mutex<Option<crate::events::JobPhase>>,
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
            stage: tokio::sync::Mutex::new(None),
        }
    }

    /// 0.0..=1.0. The ends always go through; the middle is capped.
    pub async fn set(&self, value: f32) {
        let value = value.clamp(0.0, 1.0).max(self.floor);
        // A number to report means the stage is over — the catch-up job waits
        // for the engine and then gets on with its own work. Taken off the row
        // before the rate cap can turn this call into a no-op, because a
        // sentence about a one-time setup left standing over a moving bar is a
        // lie the same size as the missing one the column was added to fix.
        self.leave_stage().await;
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
        job.phase = None;
        self.events.emit(UiEvent::JobProgress(JobProgressPayload {
            label: Some(label_for(job.kind).to_string()),
            job,
        }));
    }

    /// This job has moved on to a stage of its own, with no fraction to report.
    ///
    /// Progress in the table is left exactly where it was — the stage is a fact
    /// about now, not a rewind — but nothing carries a number, because there is
    /// no honest one to carry and a bar frozen at 100% for a quarter of an hour
    /// reads as broken.
    ///
    /// Written to the row as well as announced. The announcement is one moment;
    /// the stage it describes can last a quarter of an hour, and the screen that
    /// has to show it may not exist yet when the moment passes (the setup job is
    /// queued at launch, before the window has finished loading). The row is
    /// what any screen mounting later reads.
    pub async fn phase(&self, phase: crate::events::JobPhase) {
        if let Err(error) = repo::set_job_phase(&self.db, &self.job.id, Some(phase)).await {
            // Back to what it was before: the announcement below still reaches
            // whoever is listening now, and only a screen opened later loses the
            // sentence.
            tracing::warn!(%error, job = %self.job.id, "could not store what this work is doing");
        }
        *self.stage.lock().await = Some(phase);
        let mut job = self.job.clone();
        job.status = JobStatus::Running;
        job.progress = None;
        job.phase = Some(phase);
        self.events.emit(UiEvent::JobProgress(JobProgressPayload {
            label: Some(phase_label_for(phase).to_string()),
            job,
        }));
    }

    /// Take the stage back off the row, if it is wearing one.
    async fn leave_stage(&self) {
        let mut stage = self.stage.lock().await;
        if stage.is_none() {
            return;
        }
        if let Err(error) = repo::set_job_phase(&self.db, &self.job.id, None).await {
            tracing::warn!(%error, job = %self.job.id, "could not clear what this work was doing");
            return;
        }
        *stage = None;
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
        ctx.progress
            .phase(crate::events::JobPhase::PreparingEngine)
            .await;
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
    let pass = ctx
        .asr
        .catch_up(&ctx.db, &meeting_id, None, control)
        .await
        .map_err(|e| asr_failure(&ctx.cancel, e));
    let _ = drain.await;
    // Nothing below this line may run for a pass that did not finish, and the
    // `?` is what enforces it. On 2026-08-24 the pass came back `Ok` after a
    // recording cut it short: the code below cleared the live text for stretches
    // that were never read back, set progress to 1.0, and the row went down as
    // done. The meeting was left 69.4% transcribed and nothing said so. A
    // cancellation is an error now, and a preempted job is parked instead —
    // `release()` puts it back in the queue and the next pass reads the holes
    // this one did not get to.
    let pass = pass?;

    // Live partials are superseded now — except over stretches this pass could
    // not read, where nothing superseded them (see
    // [`repo::delete_partial_segments_except`]).
    //
    // Worth being exact about what this can spare, because a sentence below
    // used to promise more than it can: a live guess reaches the screen as an
    // announcement and reaches the database only once its words have settled,
    // so nothing this app writes today leaves an unfinished row behind. The
    // sweep is a guard against rows an older build could have left, never a
    // source of text for a stretch that would not read — those seconds had no
    // text of any kind, which is exactly why the pass was sent at them.
    let unread: Vec<(Channel, i64, i64)> = pass.unread_merged();
    if let Err(error) = repo::delete_partial_segments_except(&ctx.db, &meeting_id, &unread).await {
        tracing::debug!(%error, "could not clear live text");
    }
    if let Ok(hist) = repo::language_histogram(&ctx.db, &meeting_id).await {
        // Most speaking time wins, but a language holding a sliver of the
        // transcript never does — one sentence misread as Chinese is not what
        // the meeting was in (see [`crate::asr::language::spoken_in`]).
        if let Some(language) = crate::asr::language::spoken_in(&hist).dominant {
            let _ = repo::set_meeting_language(&ctx.db, &meeting_id, &language).await;
        }
    }

    let unread_ms = pass.unread_ms();
    tracing::info!(
        meeting = %meeting_id,
        written = pass.segments_written,
        unread = pass.unread.len(),
        unread_ms,
        "catch-up finished"
    );
    // A pass that finished with holes in it is still a pass that finished: the
    // job goes down as done at full progress, because it did everything it can
    // do. What must not happen is that being the whole of the story — the
    // stretch that would not read is the one the app promised to fill in when
    // the meeting ended, and a promise that failed is said out loud.
    if let Some(notice) = unread_stretch_notice(&meeting_id, unread_ms) {
        tracing::warn!(
            meeting = %meeting_id,
            unread = pass.unread.len(),
            unread_ms,
            "part of this recording would not read back and has no words against it"
        );
        ctx.events.emit(UiEvent::Notice(notice));
    }
    ctx.progress.set(1.0).await;
    Ok(())
}

/// Say it out loud when a finished catch-up pass left part of the recording
/// unread — and say nothing at all otherwise.
///
/// The pass is allowed to give up on a window: it asks the engine twice and
/// carries on rather than abandoning the rest of the meeting. What it is not
/// allowed to do is give up quietly. Those seconds are exactly the ones the app
/// promised to come back for — "Echo will fill in the rest when the meeting
/// ends" — so when it cannot, the person hears it rather than finding a gap
/// months later.
///
/// It used to end "The live text from then is still here", and that was not
/// true. The pass is sent at the stretches of the recording that have **no**
/// text against them — that is how its work is planned — so a window it could
/// not read is a window over seconds nothing had ever written down. Saying the
/// rough text survived, on top of a transcript that reads there exactly like
/// one where nobody spoke, is a false explanation for a real hole: the closing
/// move of the 2026-08-21 incident, in a different place.
///
/// Persistent, for the reason the ending of a lost recording is (see
/// `session::recovery::ending_notice`): nothing else on any screen marks those
/// seconds. The transcript is silent there, the left-out card on the same tab
/// is about a different decision entirely, and a pass finishes minutes after
/// somebody has walked away from the meeting that queued it. A toast that
/// fades in four seconds is a quieter kind of silence.
fn unread_stretch_notice(meeting_id: &str, unread_ms: i64) -> Option<NoticePayload> {
    if unread_ms <= 0 {
        return None;
    }
    Some(NoticePayload {
        level: NoticeLevel::Warning,
        message: unread_stretch_message(unread_ms),
        persistent: true,
        meeting_id: Some(meeting_id.to_string()),
        // One meeting, one such message: reading it again replaces the sentence
        // rather than stacking a second copy of it.
        tag: Some("someOfItUnread".into()),
    })
}

/// What Echo says when it could not read part of a recording back.
///
/// Rounded up to the nearest minute, and never below "less than a minute": the
/// exact figure is a sum of window spans, which is precise about the wrong
/// thing — a person wants to know whether to go back and listen, and "about 3
/// minutes" answers that while "2 minutes 47 seconds" pretends to an accuracy
/// this number does not have. Up rather than down, because rounding a shortfall
/// down is the direction that flatters Echo.
///
/// The second sentence is the one useful thing left to say: the words are gone
/// but the audio is not, so the meeting can be listened to, or read again from
/// its own screen. It does not name the button that does the reading — that
/// name lives with the button, and a copy of it here is a copy that can drift.
fn unread_stretch_message(unread_ms: i64) -> String {
    // `div_ceil` on a signed integer is not settled in the compiler this builds
    // on, and rounding up is one line of arithmetic.
    let how_much = if unread_ms < 60_000 {
        "less than a minute".to_string()
    } else {
        match (unread_ms + 59_999) / 60_000 {
            1 => "about a minute".to_string(),
            minutes => format!("about {minutes} minutes"),
        }
    };
    format!(
        "Echo couldn't read {how_much} of this recording back, so that part of the \
         transcript has no words in it. The recording itself is still here."
    )
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
    ctx.progress
        .phase(crate::events::JobPhase::PreparingEngine)
        .await;
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
    ctx.progress
        .phase(crate::events::JobPhase::PreparingEngine)
        .await;
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
    /// Held while who owns the machine changes, and while the row of a job a
    /// recording stopped is written.
    ///
    /// [`JobRuntime::blocked`] is what decides between `paused` and `queued` for
    /// such a job, and under this lock the read of the flag and the row that
    /// follows from it cannot be split by a `release()` landing in between. That
    /// split is the whole of the 2026-08-26 stall: a job already back in the
    /// queue was written straight back to `paused`, where nothing un-parks it
    /// (see [`JobRuntime::park_for_a_recording`]).
    ///
    /// Nothing slow happens under it — a flag and one statement — and no other
    /// lock is taken while it is held.
    parking: tokio::sync::Mutex<()>,
    /// Asked "is a capture live right now?" once the session spine has wired
    /// itself up.
    ///
    /// [`JobRuntime::blocked`] is close but not the same thing: it is a flag
    /// somebody has to remember to clear, and the engine's lifecycle is not
    /// something to hang on a flag (review of 2026-08-20, finding 2). This asks
    /// the capture state machine, which is the fact itself.
    capturing: std::sync::Mutex<Option<Arc<dyn Fn() -> bool + Send + Sync>>>,
    /// Asked "does Echo think a meeting is happening right now?" — the third
    /// fact the tray rule needs, and the only one that lives outside this
    /// runtime and the capture state machine. Asked, never remembered, for the
    /// same reason as [`JobRuntime::capturing`].
    detecting: std::sync::Mutex<Option<Arc<dyn Fn() -> bool + Send + Sync>>>,
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
            parking: tokio::sync::Mutex::new(()),
            capturing: std::sync::Mutex::new(None),
            detecting: std::sync::Mutex::new(None),
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
        // not this loop gets to it in the next second — and a reason for the
        // menu bar to say so. This is also the edge that starts the working
        // icon for every path that finishes a meeting: a normal stop, an
        // interrupted meeting being picked back up, a transcript being redone.
        self.refresh_engine_residency().await;
        self.refresh_tray_state().await;
        self.wake.notify_one();
        Ok(job)
    }

    /// Point the residency rule at the capture state machine. Called once, when
    /// the session spine builds itself.
    pub(crate) fn watch_capture(&self, is_live: Arc<dyn Fn() -> bool + Send + Sync>) {
        *self.capturing.lock().expect("capture hook poisoned") = Some(is_live);
    }

    /// Point the tray rule at the meeting watcher. Called once, at launch,
    /// because the watcher lives with the app handle rather than with the
    /// session spine. Never wired in tests, where nothing is detected.
    pub(crate) fn watch_detection(&self, is_detected: Arc<dyn Fn() -> bool + Send + Sync>) {
        *self.detecting.lock().expect("detection hook poisoned") = Some(is_detected);
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

    /// Does Echo think a meeting is happening right now?
    fn meeting_is_detected(&self) -> bool {
        let hook = self
            .detecting
            .lock()
            .expect("detection hook poisoned")
            .clone();
        // No watcher to ask means nothing has been detected — which is the truth
        // on a machine where detection is turned off, and the safe answer
        // everywhere else: the worst it costs is a badge the watcher's own poll
        // puts back within [`crate::detect::POLL_INTERVAL_SECS`].
        hook.is_some_and(|is_detected| is_detected())
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

    /// Say what the menu bar should be showing, recomputed from scratch.
    ///
    /// The tray's half of [`JobRuntime::refresh_engine_residency`], and
    /// deliberately built the same way: every caller — queue, cancel, retry, a
    /// job ending however it ended, a recording being released — asks the same
    /// question of the same three fresh facts. Nothing here accumulates, so
    /// there is no edge that could be "the one that was missed" and leave the
    /// menu bar claiming Echo is busy with a meeting it finished an hour ago.
    ///
    /// Cancelled and failed work stops counting for free: `active_only` means
    /// queued, running or parked, so the row that ends any of those three ways
    /// is the row that lets the icon settle.
    pub(crate) async fn refresh_tray_state(&self) {
        let capturing = self.capture_is_live();
        let detected = self.meeting_is_detected();
        let outstanding = outstanding_meeting_jobs(&self.db).await;
        self.ports
            .events
            .emit(UiEvent::TrayState(super::tray_state_for(
                capturing,
                detected,
                outstanding,
            )));
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
        self.refresh_tray_state().await;
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
        self.refresh_tray_state().await;
        self.wake.notify_one();
        Ok(())
    }

    /// A recording is starting. Park everything, keep it resumable.
    pub async fn preempt(&self) -> Result<(), DbError> {
        let parked = {
            // Under the lock with the write, so a job on its way out cannot read
            // this flag on one side of a `release()` and write its row on the
            // other.
            let _ordering = self.parking.lock().await;
            self.blocked.store(true, Ordering::SeqCst);
            if let Ok(running) = self.running.lock() {
                if let Some((id, cancel)) = running.as_ref() {
                    tracing::info!(job = %id, "parking background work for a recording");
                    cancel.preempt();
                }
            }
            repo::pause_active_jobs(&self.db).await?
        };
        if parked > 0 {
            tracing::info!(parked, "background work parked");
            // And say so on the screens, not only in the menu bar. A meeting
            // whose work is parked used to keep whatever it was last told —
            // "Working out who said what…" over a bar that had stopped moving —
            // for as long as the recording lasted, which on a day of
            // back-to-back meetings is most of the day. The rows have already
            // changed in the table; this is what carries the change to anything
            // looking at them.
            self.announce_all_active().await;
        }
        Ok(())
    }

    /// Re-announce every unfinished job, so screens holding a list of them
    /// redraw from what the table says now.
    async fn announce_all_active(&self) {
        let query = JobQuery {
            active_only: Some(true),
            ..Default::default()
        };
        match repo::list_jobs(&self.db, &query).await {
            Ok(jobs) => {
                for job in jobs {
                    self.announce(&job);
                }
            }
            Err(error) => tracing::warn!(%error, "could not say which work is waiting"),
        }
    }

    /// The recording finished. Let the parked work continue.
    pub async fn release(&self) -> Result<(), DbError> {
        let resumed = {
            // The other half of the pair, and the reason for the lock: this used
            // to put a job back in the queue only for that job's own last words
            // to write `paused` over it.
            let _ordering = self.parking.lock().await;
            self.blocked.store(false, Ordering::SeqCst);
            repo::resume_paused_jobs(&self.db).await?
        };
        if resumed > 0 {
            tracing::info!(resumed, "background work picked back up");
            // The other edge of the same sentence. Parking says "paused until
            // the recording ends"; without this, that line stays on screen
            // after the recording has ended, on every row except the one that
            // happens to start running next — which is the same stale sentence
            // this pair of announcements exists to stop.
            self.announce_all_active().await;
        }
        // Also the launch path, where work left over from a crash is un-parked:
        // whatever is outstanding now decides whether the engine is held, and
        // what the menu bar says. A launch that finds a meeting's work still in
        // the table is a launch that should say "still working on it" before
        // anyone asks.
        self.refresh_engine_residency().await;
        self.refresh_tray_state().await;
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

    /// Write down that a recording stopped this job: parked while the recording
    /// still holds the machine, straight back in the queue when it does not.
    ///
    /// It used to be `paused`, always — and `release()` is the only thing that
    /// un-parks a row. A job that reported itself stopped **after** `release()`
    /// had already put it back in the queue wrote `paused` over that, and then
    /// sat there: `active_only` counts parked work, so the menu bar stayed on
    /// Processing, 1.6 GB of speech weights stayed loaded, the meeting never
    /// left "Processing" in the library, and the screen said "paused until the
    /// recording ends" while nothing was recording and nothing would ever end.
    /// The next recording to finish does clear it, so it is a stall rather than
    /// a stop — but for somebody who does not record again it never ends, and
    /// that sentence is false for the whole of it.
    ///
    /// [`JobRuntime::parking`] is what makes the answer trustworthy: the flag is
    /// read and the row is written with no `release()` able to fit between them.
    ///
    /// Shutting down comes through here too, and comes out `queued`. That is
    /// what it should be: nothing is recording, and a row waiting for a
    /// recording to end would be waiting for something that already happened.
    async fn park_for_a_recording(&self, job_id: &str) -> JobStatus {
        let _ordering = self.parking.lock().await;
        let status = if self.blocked.load(Ordering::SeqCst) {
            JobStatus::Paused
        } else {
            JobStatus::Queued
        };
        if let Err(error) = repo::set_job_status(&self.db, job_id, status, None).await {
            tracing::warn!(%error, job = %job_id, "could not record work a recording stopped");
        }
        status
    }

    /// Write down how a job that reached its own end ended.
    async fn record_end(
        &self,
        job_id: &str,
        status: JobStatus,
        error: Option<String>,
    ) -> JobStatus {
        if let Err(failed) = repo::set_job_status(&self.db, job_id, status, error.as_deref()).await
        {
            tracing::warn!(error = %failed, "could not record how the work ended");
        }
        status
    }

    /// Say where a job stands, from the row as it reads now — stage included,
    /// because the stage is the truer sentence when there is one.
    fn announce(&self, job: &Job) {
        let label = match job.phase {
            Some(phase) => phase_label_for(phase).to_string(),
            None => label_for(job.kind).to_string(),
        };
        self.ports
            .events
            .emit(UiEvent::JobProgress(JobProgressPayload {
                job: job.clone(),
                label: Some(label),
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
            let status = self.park_for_a_recording(&job.id).await;
            if let Ok(Some(parked)) = repo::get_job(&self.db, &job.id).await {
                self.announce(&parked);
            }
            tracing::info!(
                job = %job.id,
                status = status.as_str(),
                "a recording claimed the machine before this could start"
            );
            // The recording ended in the moment this took, so the row is queued
            // rather than parked and the loop has work to come back to.
            if matches!(status, JobStatus::Queued) {
                self.wake.notify_one();
            }
            return;
        }
        if let Err(error) = repo::set_job_status(&self.db, &job.id, JobStatus::Running, None).await
        {
            tracing::warn!(%error, "could not mark work as running");
        }

        let mut started = job.clone();
        started.status = JobStatus::Running;
        // The status write above cleared any stage the row had; this is a job
        // beginning, and it is not in one yet.
        started.phase = None;
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

        let status = match outcome {
            Ok(()) => self.record_end(&job.id, JobStatus::Done, None).await,
            Err(JobFailure::Cancelled) => {
                self.record_end(&job.id, JobStatus::Cancelled, None).await
            }
            Err(JobFailure::Failed(message)) => {
                self.record_end(&job.id, JobStatus::Failed, Some(message))
                    .await
            }
            // Not a status this can decide on its own: parking is only right
            // while a recording still holds the machine
            // ([`JobRuntime::park_for_a_recording`]).
            Err(JobFailure::Preempted) => self.park_for_a_recording(&job.id).await,
        };
        if let Ok(mut running) = self.running.lock() {
            *running = None;
        }
        // A recording that ended while this job was on its way out leaves it
        // queued, not parked. Nothing else will notice that on its own when
        // `execute` was driven from outside the loop.
        if matches!(status, JobStatus::Queued) {
            self.wake.notify_one();
        }

        if let Ok(Some(finished)) = repo::get_job(&self.db, &job.id).await {
            self.announce(&finished);
        }
        // This may have been the meeting's last job. If it was, and nothing is
        // being recorded, this is the moment the engine's grace period starts
        // (mantra 1's amendment of 2026-08-20) and the moment the menu bar stops
        // saying Echo is working on something.
        //
        // Here rather than in `settle_meeting`, deliberately: that one only runs
        // for work that finished, and a meeting whose last job was cancelled or
        // failed is just as finished as far as the icon is concerned. The
        // meeting row stays Processing in that case — which is true, something
        // did not get done — but nothing is being worked on, and an icon that
        // spun forever after a cancel would be a lie nobody could clear.
        self.refresh_engine_residency().await;
        self.refresh_tray_state().await;
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

    /// The icon must settle when the meeting's work is *over*, not when it
    /// happened to succeed. A cancelled or failed row ends the work just as
    /// truly as a finished one — `active_only` is what makes that free, and a
    /// regression here would leave the menu bar claiming Echo is still busy
    /// with a meeting nobody is working on any more.
    #[tokio::test]
    async fn work_stops_counting_however_it_ended() {
        let db = crate::db::connect_in_memory().await.unwrap();
        let meeting = repo::create_meeting(&db, "", "/tmp", None).await.unwrap();
        assert_eq!(outstanding_meeting_jobs(&db).await, 0);

        let catch_up = repo::create_job(&db, Some(&meeting.id), JobKind::TranscribeCatchup)
            .await
            .unwrap();
        let diarize = repo::create_job(&db, Some(&meeting.id), JobKind::Diarize)
            .await
            .unwrap();
        let recap = repo::create_job(&db, Some(&meeting.id), JobKind::Summarize)
            .await
            .unwrap();
        // Work that belongs to no meeting never held the icon in the first
        // place: a model download is not "still finishing your meeting".
        repo::create_job(&db, None, JobKind::Download)
            .await
            .unwrap();
        assert_eq!(outstanding_meeting_jobs(&db).await, 3);

        repo::set_job_status(&db, &catch_up.id, JobStatus::Done, None)
            .await
            .unwrap();
        assert_eq!(outstanding_meeting_jobs(&db).await, 2);
        repo::set_job_status(&db, &diarize.id, JobStatus::Cancelled, None)
            .await
            .unwrap();
        assert_eq!(outstanding_meeting_jobs(&db).await, 1);

        // The last one, and it failed. The meeting is still over.
        repo::set_job_status(&db, &recap.id, JobStatus::Failed, Some("no"))
            .await
            .unwrap();
        assert_eq!(outstanding_meeting_jobs(&db).await, 0);
        assert_eq!(
            super::super::tray_state_for(false, false, 0),
            crate::types::TrayState::Idle
        );
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
        context_with_asr(db, job, Arc::new(super::super::mock::MockAsr::new())).await
    }

    /// [`context_for`] for a test that has to set the engine up first.
    async fn context_with_asr(
        db: &Db,
        job: Job,
        asr: Arc<super::super::mock::MockAsr>,
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
                asr,
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
                corrections: Vec::new(),
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
            .find(|p| p.job.phase == Some(crate::events::JobPhase::PreparingEngine))
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
        //
        // Contract change of 2026-08-26: the marker names the build of Echo that
        // made the attempt as well as the weights, because the compiled encoder
        // is cached against the binary and a rebuilt app has to pay again. The
        // weights are still the front of it; what follows the `@` is whichever
        // build is running the test.
        assert_eq!(
            attempt_marker(&db).await.as_deref().map(name_in_marker),
            Some("some-weights.bin")
        );
    }

    /// What the settings table is holding as the one attempt, if anything.
    async fn attempt_marker(db: &Db) -> Option<String> {
        repo::get_setting(db, crate::settings::keys::SPEECH_WARM_ATTEMPTED)
            .await
            .expect("the settings table")
    }

    /// The weights half of a marker, which is all these tests are about; the
    /// build half belongs to `asr::models`, which is where it is tested.
    fn name_in_marker(marker: &str) -> &str {
        marker.split('@').next().expect("a marker names weights")
    }

    /// The order the whole one-attempt policy rests on: the attempt is on the
    /// record *before* the load begins, not after it comes back.
    ///
    /// The window this test looks into is the one a crash falls into. Anything
    /// that takes the process down inside the compile — an out-of-memory, a
    /// driver fault, somebody force-quitting an app that looks frozen — leaves
    /// the marker standing, and that is what stops the next launch from starting
    /// the same quarter of an hour again.
    #[tokio::test]
    async fn the_attempt_is_on_the_record_before_the_load_starts() {
        let db = crate::db::connect_in_memory().await.unwrap();
        let job = a_setup_job_for(&db, "some-weights.bin").await;
        let (mut ctx, _seen) = context_for(&db, job).await;
        let asr = Arc::new(super::super::mock::MockAsr::new());
        asr.prewarm_takes(Duration::from_millis(300));
        ctx.asr = asr;

        let while_it_loads = async {
            tokio::time::sleep(Duration::from_millis(80)).await;
            attempt_marker(&db).await
        };
        let (outcome, midway) = tokio::join!(prepare_engine(&ctx), while_it_loads);
        outcome.expect("the setup finishes");
        assert_eq!(
            midway.as_deref().map(name_in_marker),
            Some("some-weights.bin"),
            "the load had not come back yet, and the attempt was already written"
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
            attempt_marker(&db).await,
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
            attempt_marker(&db).await.as_deref().map(name_in_marker),
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
            seen.job_progress().iter().all(|p| p.job.phase.is_none()),
            "nothing to announce: the engine was already up"
        );
    }

    /// A meeting with a minute of committed audio and one live guess sitting
    /// against the second half of it, plus its catch-up job.
    async fn a_meeting_mid_catch_up(db: &Db) -> (String, Job) {
        let meeting = repo::create_meeting(db, "Weekly sync", "/tmp/echo-test", None)
            .await
            .unwrap();
        let chunk = repo::insert_chunk(
            db,
            &meeting.id,
            crate::types::Channel::Mic,
            0,
            "/tmp/echo-test/mic-000000.flac",
            0,
            60_000,
        )
        .await
        .unwrap();
        repo::commit_chunk(db, &chunk, 60_000).await.unwrap();
        repo::insert_segment(
            db,
            &crate::types::SegmentDraft {
                meeting_id: meeting.id.clone(),
                t_start_ms: 30_000,
                t_end_ms: 40_000,
                channel: crate::types::Channel::Mic,
                text: "a live guess nobody has replaced yet".into(),
                is_final: false,
                revision: 1,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let job = repo::ensure_job(db, Some(&meeting.id), JobKind::TranscribeCatchup)
            .await
            .unwrap();
        (meeting.id, job)
    }

    async fn live_guesses(db: &Db, meeting_id: &str) -> usize {
        repo::get_segments(
            db,
            &crate::types::TranscriptQuery {
                meeting_id: meeting_id.to_string(),
                include_partial: Some(true),
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .iter()
        .filter(|s| !s.is_final)
        .count()
    }

    /// The 2026-08-24 data loss, at the handler: a recording took the machine
    /// part-way through the catch-up pass, and the pass came back saying it was
    /// finished. The job was recorded done at full progress, the live text was
    /// cleared for stretches that had never been read back, and the meeting was
    /// left 69.4% transcribed with nothing anywhere saying so.
    ///
    /// A pass cut short is a failure the runner understands: parked, not done.
    /// And nothing after it in the handler may run — least of all the clearing
    /// of the live guesses, which are the only text those stretches have until
    /// the pass that finishes actually reads them.
    ///
    /// This is the handler's half of that guarantee, and only that half: the
    /// pass itself keeps the same promise window by window, and
    /// `asr::catchup::tests::a_pass_cut_short_keeps_the_live_guesses_over_what_it_never_read`
    /// is where the pass is held to it.
    #[tokio::test]
    async fn a_catch_up_a_recording_stopped_is_parked_and_keeps_the_live_text() {
        let db = crate::db::connect_in_memory().await.unwrap();
        let (meeting_id, job) = a_meeting_mid_catch_up(&db).await;
        let (ctx, _seen) = context_for(&db, job).await;
        assert_eq!(live_guesses(&db, &meeting_id).await, 1);

        // A recording starting is a preempt, and it reaches the pass through the
        // same flag a cancel does.
        ctx.cancel.preempt();
        let outcome = catch_up(&ctx).await;

        assert!(
            matches!(outcome, Err(JobFailure::Preempted)),
            "a pass a recording interrupted is parked, not finished: {outcome:?}"
        );
        assert_eq!(
            live_guesses(&db, &meeting_id).await,
            1,
            "the live guess is all those seconds have until the pass reads them back"
        );
        assert!(
            repo::get_job(&db, &ctx.job.id)
                .await
                .unwrap()
                .and_then(|j| j.progress)
                .is_none_or(|p| p < 1.0),
            "an interrupted pass does not leave a full bar behind"
        );
    }

    /// A pass that finished with a hole in it does not finish quietly.
    ///
    /// The live text told the person "Echo will fill in the rest when the
    /// meeting ends". When a window would not read, that stretch is exactly the
    /// rest — and the runner used to mark the job done at 100% with nothing
    /// anywhere saying otherwise.
    ///
    /// The sentence is held to what it may claim as much as to being said at
    /// all. It used to end "The live text from then is still here", which is
    /// not true of a stretch the pass was sent at *because* nothing had written
    /// it down: those seconds have no words, and the transcript reads there
    /// exactly like one where nobody spoke.
    ///
    /// The guess the fixture writes is the shape an older build could leave
    /// behind, and the sweep still spares it: text that exists is never thrown
    /// away over seconds nothing replaced.
    #[tokio::test]
    async fn a_stretch_that_would_not_read_is_said_out_loud_without_a_false_comfort() {
        let db = crate::db::connect_in_memory().await.unwrap();
        let (meeting_id, job) = a_meeting_mid_catch_up(&db).await;
        let asr = Arc::new(super::super::mock::MockAsr::new());
        // The fixture's guess sits at 30 s–40 s on the microphone, and this is
        // the window the engine would not read either time it was asked.
        asr.catch_up_leaves_unread(vec![crate::asr::catchup::UnreadSpan {
            channel: crate::types::Channel::Mic,
            from_ms: 30_000,
            to_ms: 60_000,
        }]);
        let (ctx, seen) = context_with_asr(&db, job.clone(), asr).await;

        catch_up(&ctx).await.expect("the pass itself finished");

        assert_eq!(
            live_guesses(&db, &meeting_id).await,
            1,
            "nothing replaced those seconds, so the guess is all the text they have"
        );
        let notices = seen.notices();
        let said = notices
            .iter()
            .find(|n| n.tag.as_deref() == Some("someOfItUnread"))
            .expect("a hole in somebody's transcript is their business, not only the log's");
        assert_eq!(
            said.message,
            "Echo couldn't read less than a minute of this recording back, so that part \
             of the transcript has no words in it. The recording itself is still here."
        );
        assert!(
            said.persistent,
            "the only mark those seconds get anywhere faded on its own"
        );
        assert_eq!(said.meeting_id.as_deref(), Some(meeting_id.as_str()));
        assert_eq!(said.level, NoticeLevel::Warning);
        // The pass did everything it can do, so the row is done — the sentence
        // is what carries the shortfall, not a job stuck at 90%.
        assert_eq!(
            repo::get_job(&db, &job.id).await.unwrap().unwrap().progress,
            Some(1.0)
        );
    }

    /// And a pass with nothing to declare declares nothing: every guess goes,
    /// because the real reading replaced all of them.
    #[tokio::test]
    async fn a_pass_that_read_the_whole_meeting_clears_the_live_text_and_says_nothing() {
        let db = crate::db::connect_in_memory().await.unwrap();
        let (meeting_id, job) = a_meeting_mid_catch_up(&db).await;
        let (ctx, seen) = context_for(&db, job).await;

        catch_up(&ctx).await.unwrap();

        assert_eq!(live_guesses(&db, &meeting_id).await, 0);
        assert!(
            !seen.notice_tagged("someOfItUnread"),
            "nothing was missing, so there is nothing to interrupt anybody about"
        );
    }

    /// The sentence a person actually reads, at each of the three sizes it comes
    /// in, and with none of the words the mechanism thinks in.
    #[test]
    fn the_unread_sentence_reads_as_one_and_carries_no_jargon() {
        assert!(
            unread_stretch_notice("m", 0).is_none(),
            "nothing unread is nothing to say"
        );
        assert!(unread_stretch_notice("m", -1).is_none());

        assert!(unread_stretch_message(20_000).contains("less than a minute"));
        assert!(unread_stretch_message(60_000).contains("about a minute"));
        // Rounded up: rounding a shortfall down is the direction that flatters
        // Echo, and a person deciding whether to go back and listen is owed the
        // larger number.
        assert!(unread_stretch_message(61_000).contains("about 2 minutes"));
        assert!(unread_stretch_message(167_000).contains("about 3 minutes"));

        let banned = [
            "window", "decode", "buffer", "stream", "vad", "segment", "span", "whisper", "engine",
            "asr", "retry",
        ];
        for unread_ms in [1_i64, 20_000, 60_000, 61_000, 3_600_000] {
            let message = unread_stretch_message(unread_ms);
            assert!(
                message.ends_with('.'),
                "{message:?} should read as sentences"
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
    }

    /// A runner with one job in it and an engine a test can hold on to.
    fn a_runtime_around(db: &Db, asr: Arc<super::super::mock::MockAsr>) -> Arc<JobRuntime> {
        let events = crate::session::ports::EventBus::new();
        events.set(Arc::new(super::super::mock::CollectingEvents::default()));
        JobRuntime::new(
            db.clone(),
            crate::paths::AppPaths::rooted_at(std::env::temp_dir().join("echo-jobs"), None),
            crate::session::ports::Ports {
                capture: Arc::new(super::super::mock::MockCapture::new()),
                asr,
                executor: Arc::new(DefaultJobExecutor),
                events,
            },
        )
    }

    /// The stall of 2026-08-26: a job that reported itself stopped **after** the
    /// recording had already ended and put it back in the queue was written
    /// straight back to `paused`, and nothing un-parks a row outside
    /// `release()`.
    ///
    /// What that cost, for as long as it lasted: the menu bar pinned on
    /// Processing, 1.6 GB of speech weights held, the meeting stuck on
    /// "Processing" in the library, and a screen reading "paused until the
    /// recording ends" with nothing recording and nothing left to end. The next
    /// recording to finish cleared it — so for anybody who recorded again it was
    /// a stall, and for anybody who did not it was the end of that meeting.
    #[tokio::test]
    async fn a_job_already_back_in_the_queue_is_not_parked_behind_a_recording_that_ended() {
        let db = crate::db::connect_in_memory().await.unwrap();
        let (_meeting_id, job) = a_meeting_mid_catch_up(&db).await;

        let asr = Arc::new(super::super::mock::MockAsr::new());
        // Still reading when the recording starts, and still on its way out
        // when it ends: the whole of the race, held open.
        asr.catch_up_waits_to_be_stopped();
        asr.catch_up_holds_on_its_way_out();
        let runtime = a_runtime_around(&db, asr.clone());

        let running = {
            let runtime = runtime.clone();
            let job = job.clone();
            tokio::spawn(async move { runtime.execute(job).await })
        };
        while !asr.catch_up_started() {
            tokio::task::yield_now().await;
        }

        runtime.preempt().await.unwrap();
        assert_eq!(
            repo::get_job(&db, &job.id).await.unwrap().unwrap().status,
            JobStatus::Paused,
            "the recording has the machine, so the work waits for it"
        );
        // And now the recording ends, while the pass is still on its way out.
        runtime.release().await.unwrap();
        assert_eq!(
            repo::get_job(&db, &job.id).await.unwrap().unwrap().status,
            JobStatus::Queued
        );

        asr.let_the_catch_up_report();
        running.await.unwrap();

        assert_eq!(
            repo::get_job(&db, &job.id).await.unwrap().unwrap().status,
            JobStatus::Queued,
            "nothing is recording, so there is nothing for this work to wait behind"
        );
        assert_eq!(
            repo::next_queued_job(&db)
                .await
                .unwrap()
                .map(|queued| queued.id),
            Some(job.id),
            "and the loop can reach it, which is what makes it not stuck"
        );
    }

    /// The other side of the same ordering, which the fix must not trade away:
    /// while the recording really does still have the machine, work parks.
    #[tokio::test]
    async fn work_a_recording_is_still_holding_parks_and_waits_for_it() {
        let db = crate::db::connect_in_memory().await.unwrap();
        let (_meeting_id, job) = a_meeting_mid_catch_up(&db).await;

        let asr = Arc::new(super::super::mock::MockAsr::new());
        asr.catch_up_waits_to_be_stopped();
        asr.catch_up_holds_on_its_way_out();
        let runtime = a_runtime_around(&db, asr.clone());

        let running = {
            let runtime = runtime.clone();
            let job = job.clone();
            tokio::spawn(async move { runtime.execute(job).await })
        };
        while !asr.catch_up_started() {
            tokio::task::yield_now().await;
        }

        runtime.preempt().await.unwrap();
        // No release: the meeting is still being recorded when the pass reports.
        asr.let_the_catch_up_report();
        running.await.unwrap();

        assert_eq!(
            repo::get_job(&db, &job.id).await.unwrap().unwrap().status,
            JobStatus::Paused,
            "a recording has absolute priority, and this work still has to happen"
        );
        assert!(
            repo::next_queued_job(&db).await.unwrap().is_none(),
            "and nothing may pick it up while the recording is running"
        );

        // And it comes back the moment the recording is over.
        runtime.release().await.unwrap();
        assert_eq!(
            repo::get_job(&db, &job.id).await.unwrap().unwrap().status,
            JobStatus::Queued
        );
    }

    /// The same thing one level up, where the damage was actually recorded: the
    /// row the runner writes when a recording claims the machine mid-pass. It
    /// said `done` on 2026-08-24. It has to say paused, so `release()` queues it
    /// again and the next pass reads the holes this one never reached.
    #[tokio::test]
    async fn the_row_a_recording_interrupted_says_paused_and_not_done() {
        let db = crate::db::connect_in_memory().await.unwrap();
        let (meeting_id, job) = a_meeting_mid_catch_up(&db).await;

        let asr = Arc::new(super::super::mock::MockAsr::new());
        // Still reading when the recording starts, which is the whole case.
        asr.catch_up_waits_to_be_stopped();
        let events = crate::session::ports::EventBus::new();
        events.set(Arc::new(super::super::mock::CollectingEvents::default()));
        let runtime = JobRuntime::new(
            db.clone(),
            crate::paths::AppPaths::rooted_at(std::env::temp_dir().join("echo-jobs"), None),
            crate::session::ports::Ports {
                capture: Arc::new(super::super::mock::MockCapture::new()),
                asr: asr.clone(),
                executor: Arc::new(DefaultJobExecutor),
                events,
            },
        );

        let running = {
            let runtime = runtime.clone();
            let job = job.clone();
            tokio::spawn(async move { runtime.execute(job).await })
        };
        while !asr.catch_up_started() {
            tokio::task::yield_now().await;
        }
        runtime.preempt().await.unwrap();
        running.await.unwrap();

        let row = repo::get_job(&db, &job.id).await.unwrap().unwrap();
        assert_eq!(
            row.status,
            JobStatus::Paused,
            "the recording has priority, and this work still has to happen"
        );
        assert_eq!(
            live_guesses(&db, &meeting_id).await,
            1,
            "nothing after the pass ran, so nothing cleared the live text"
        );

        // And the runner picks it back up when the recording is over, which is
        // the half that makes parking it safe.
        runtime.release().await.unwrap();
        assert_eq!(
            repo::get_job(&db, &job.id).await.unwrap().unwrap().status,
            JobStatus::Queued
        );
    }

    /// Parking work is not a silent state change.
    ///
    /// The rows go to `paused` the moment a recording starts, but until the
    /// screens are told, a meeting keeps whatever it last said — "Working out
    /// who said what…" over a bar that has stopped moving — for as long as the
    /// recording lasts. On a day of back-to-back meetings that is most of the
    /// day, and it reads as Echo being stuck rather than as Echo waiting.
    #[tokio::test]
    async fn work_parked_for_a_recording_is_announced_as_parked() {
        let db = crate::db::connect_in_memory().await.unwrap();
        let meeting = repo::create_meeting(&db, "Weekly sync", "/tmp/echo-test", None)
            .await
            .unwrap();
        let job = repo::create_job(&db, Some(&meeting.id), JobKind::Diarize)
            .await
            .unwrap();

        let collecting = Arc::new(super::super::mock::CollectingEvents::default());
        let events = crate::session::ports::EventBus::new();
        events.set(collecting.clone());
        let runtime = JobRuntime::new(
            db.clone(),
            crate::paths::AppPaths::rooted_at(std::env::temp_dir().join("echo-jobs"), None),
            crate::session::ports::Ports {
                capture: Arc::new(super::super::mock::MockCapture::new()),
                asr: Arc::new(super::super::mock::MockAsr::new()),
                executor: Arc::new(DefaultJobExecutor),
                events,
            },
        );

        runtime.preempt().await.unwrap();

        let said = collecting.job_progress();
        let parked = said
            .iter()
            .find(|p| p.job.id == job.id)
            .expect("the parked job was announced");
        assert_eq!(parked.job.status, JobStatus::Paused);
    }

    /// And neither is un-parking it.
    ///
    /// Only one row is announced when work actually starts, so every other row
    /// would go on saying "paused until the recording ends" with nothing
    /// recording — the same stale sentence, on the other edge. A meeting with
    /// two pieces of work outstanding is the ordinary case, so this test has
    /// two.
    #[tokio::test]
    async fn work_let_go_when_the_recording_ends_is_announced_as_waiting() {
        let db = crate::db::connect_in_memory().await.unwrap();
        let meeting = repo::create_meeting(&db, "Weekly sync", "/tmp/echo-test", None)
            .await
            .unwrap();
        let diarize = repo::create_job(&db, Some(&meeting.id), JobKind::Diarize)
            .await
            .unwrap();
        let summarize = repo::create_job(&db, Some(&meeting.id), JobKind::Summarize)
            .await
            .unwrap();

        let collecting = Arc::new(super::super::mock::CollectingEvents::default());
        let events = crate::session::ports::EventBus::new();
        events.set(collecting.clone());
        let runtime = JobRuntime::new(
            db.clone(),
            crate::paths::AppPaths::rooted_at(std::env::temp_dir().join("echo-jobs"), None),
            crate::session::ports::Ports {
                capture: Arc::new(super::super::mock::MockCapture::new()),
                asr: Arc::new(super::super::mock::MockAsr::new()),
                executor: Arc::new(DefaultJobExecutor),
                events,
            },
        );

        runtime.preempt().await.unwrap();
        runtime.release().await.unwrap();

        // The last thing said about each row is what a screen is left showing.
        let said = collecting.job_progress();
        for id in [&diarize.id, &summarize.id] {
            let last = said
                .iter()
                .rfind(|p| &p.job.id == id)
                .expect("the job was announced");
            assert_eq!(
                last.job.status,
                JobStatus::Queued,
                "a row left saying it is paused with nothing recording"
            );
        }
    }

    /// The stage goes on the row, not only into an announcement nobody may be
    /// there to hear — and the first honest fraction takes it back off.
    ///
    /// The launch case is the whole reason: the setup job is queued while the
    /// window is still loading, so the announcement lands before any screen is
    /// listening, and the row is all the corner pill has to go on when it
    /// finally mounts. And the catch-up case is why it has to come off again:
    /// that job waits for the engine, then gets on with its own work, and
    /// "Finishing one-time setup…" over a moving bar would be the same lie
    /// pointing the other way.
    #[tokio::test]
    async fn a_stage_is_stored_while_it_lasts_and_only_while_it_lasts() {
        let db = crate::db::connect_in_memory().await.unwrap();
        let job = repo::ensure_job(&db, None, JobKind::PrepareEngine)
            .await
            .unwrap();
        let (ctx, _seen) = context_for(&db, job.clone()).await;

        ctx.progress
            .phase(crate::events::JobPhase::PreparingEngine)
            .await;
        assert_eq!(
            repo::get_job(&db, &job.id).await.unwrap().unwrap().phase,
            Some(crate::events::JobPhase::PreparingEngine),
            "a screen mounting now has to be able to read what is happening"
        );

        // Rate-capped or not, a fraction ends the stage: `set` was just called
        // with 0.05, which the cap would ordinarily swallow.
        ctx.progress.set(0.05).await;
        assert!(
            repo::get_job(&db, &job.id)
                .await
                .unwrap()
                .unwrap()
                .phase
                .is_none(),
            "the stage is over the moment there are numbers again"
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
            phase: None,
        };
        let (ctx, seen) = context_for(&db, job).await;
        ctx.progress.set(1.0).await;
        ctx.progress
            .phase(crate::events::JobPhase::PreparingEngine)
            .await;

        let announced = seen.job_progress();
        assert_eq!(announced.len(), 2);
        assert_eq!(announced[0].job.progress, Some(1.0));
        assert!(announced[0].job.phase.is_none());
        assert_eq!(
            announced[1].job.phase,
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
