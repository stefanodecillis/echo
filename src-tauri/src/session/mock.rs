//! Test doubles for the seams in [`super::ports`], plus the harness that wires
//! them to a real in-memory database.
//!
//! The point of these is that the whole spine — start, degrade, stop, journal,
//! batch, recover, preempt — can be exercised with no microphone, no weights and
//! no window.

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::future::BoxFuture;
use tokio::sync::mpsc;

use crate::asr::catchup::{CatchUpReport, UnreadSpan};
use crate::asr::engine::PartialFn;
use crate::asr::{AsrError, TranscribeJob, Transcription};
use crate::audio::{AudioError, CaptureConfig, CaptureStarted};
use crate::db::Db;
use crate::paths::AppPaths;
use crate::session::jobs::{JobContext, JobExecutor, JobFailure};
use crate::session::ports::{
    AsrPort, CaptureHandle, CapturePort, CaptureSender, CaptureSignal, CaptureStart, EventBus,
    EventSink, Ports, UiEvent,
};
use crate::session::SessionManager;
use crate::types::{Channel, JobKind};

/// How long a `wait_*` helper keeps trying before it calls the test failed.
const PATIENCE: Duration = Duration::from_secs(5);

// ---------------------------------------------------------------------------
// Capture
// ---------------------------------------------------------------------------

/// Shared between the port and the handle it produced.
#[derive(Default)]
struct CaptureShared {
    elapsed_ms: AtomicI64,
    paused: AtomicBool,
    /// Dropped when the capture stops, which closes the feed.
    sender: std::sync::Mutex<Option<CaptureSender>>,
    channels: std::sync::Mutex<Vec<Channel>>,
}

/// A capture that does whatever the test tells it to.
pub(crate) struct MockCapture {
    shared: Arc<CaptureShared>,
    fail: std::sync::Mutex<Option<AudioError>>,
    system_audio_error: std::sync::Mutex<Option<String>>,
    opens: AtomicU32,
    /// The speech engine, so opening a device can record what the engine's
    /// lifecycle looked like at that exact moment.
    asr: std::sync::Mutex<Option<Arc<MockAsr>>>,
    /// Was the engine already claimed when the first device was opened?
    resident_at_open: std::sync::Mutex<Option<bool>>,
}

impl MockCapture {
    pub(crate) fn new() -> Self {
        let shared = Arc::new(CaptureShared::default());
        *shared.channels.lock().unwrap() = vec![Channel::Mic, Channel::System];
        Self {
            shared,
            fail: std::sync::Mutex::new(None),
            system_audio_error: std::sync::Mutex::new(None),
            opens: AtomicU32::new(0),
            asr: std::sync::Mutex::new(None),
            resident_at_open: std::sync::Mutex::new(None),
        }
    }

    /// Watch the engine, so the order of "claim the engine" and "open the
    /// devices" is something a test can assert on.
    pub(crate) fn watch_engine(&self, asr: Arc<MockAsr>) {
        *self.asr.lock().unwrap() = Some(asr);
    }

    /// Whether a meeting was already holding the engine when the first device
    /// was opened. `None` until something opens one.
    pub(crate) fn engine_held_when_opened(&self) -> Option<bool> {
        *self.resident_at_open.lock().unwrap()
    }

    /// The next `open` fails with this.
    pub(crate) fn fail_with(&self, error: AudioError) {
        *self.fail.lock().unwrap() = Some(error);
    }

    /// Microphone only, with a reason, the way a missing permission looks.
    pub(crate) fn without_system_audio(&self, detail: &str) {
        *self.system_audio_error.lock().unwrap() = Some(detail.to_string());
        *self.shared.channels.lock().unwrap() = vec![Channel::Mic];
    }

    pub(crate) fn open_count(&self) -> u32 {
        self.opens.load(Ordering::SeqCst)
    }

    pub(crate) fn set_elapsed(&self, t_ms: i64) {
        self.shared.elapsed_ms.store(t_ms, Ordering::SeqCst);
    }

    /// Push something up the feed, as the real capture threads would.
    pub(crate) fn send(&self, signal: CaptureSignal) {
        let sender = self.shared.sender.lock().unwrap().clone();
        match sender {
            Some(sender) => sender.send(signal).expect("the feed is listening"),
            None => panic!("nothing is capturing"),
        }
    }
}

impl CapturePort for MockCapture {
    fn open<'a>(&'a self, _cfg: CaptureConfig) -> BoxFuture<'a, Result<CaptureStart, AudioError>> {
        Box::pin(async move {
            let first = self.opens.fetch_add(1, Ordering::SeqCst) == 0;
            if first {
                let held = self
                    .asr
                    .lock()
                    .unwrap()
                    .as_ref()
                    .map(|asr| asr.is_resident());
                *self.resident_at_open.lock().unwrap() = held;
            }
            if let Some(error) = self.fail.lock().unwrap().take() {
                return Err(error);
            }
            let (tx, feed) = mpsc::unbounded_channel();
            *self.shared.sender.lock().unwrap() = Some(tx);
            self.shared.paused.store(false, Ordering::SeqCst);
            let started = CaptureStarted {
                channels: self.shared.channels.lock().unwrap().clone(),
                system_audio_error: self.system_audio_error.lock().unwrap().clone(),
                microphone_error: None,
                speech_detection_degraded: false,
            };
            Ok(CaptureStart {
                handle: Box::new(MockHandle {
                    shared: self.shared.clone(),
                }),
                started,
                feed,
            })
        })
    }
}

struct MockHandle {
    shared: Arc<CaptureShared>,
}

impl CaptureHandle for MockHandle {
    fn elapsed_ms(&self) -> i64 {
        self.shared.elapsed_ms.load(Ordering::SeqCst)
    }

    fn levels(&self) -> (f32, f32) {
        (0.1, 0.1)
    }

    fn active_channels(&self) -> Vec<Channel> {
        self.shared.channels.lock().unwrap().clone()
    }

    fn pending_utterances(&self) -> u32 {
        0
    }

    fn pause(&self) -> BoxFuture<'_, Result<(), AudioError>> {
        self.shared.paused.store(true, Ordering::SeqCst);
        Box::pin(async { Ok(()) })
    }

    fn resume(&self) -> BoxFuture<'_, Result<(), AudioError>> {
        self.shared.paused.store(false, Ordering::SeqCst);
        Box::pin(async { Ok(()) })
    }

    fn stop(self: Box<Self>) -> BoxFuture<'static, Result<i64, AudioError>> {
        // Dropping the sender is what tells the pipeline the recording is over.
        let sender = self.shared.sender.lock().unwrap().take();
        drop(sender);
        let elapsed = self.shared.elapsed_ms.load(Ordering::SeqCst);
        Box::pin(async move { Ok(elapsed) })
    }
}

// ---------------------------------------------------------------------------
// Speech
// ---------------------------------------------------------------------------

/// A speech engine that always says the same thing, and that tracks whether it
/// is "loaded" so the lazy-lifecycle rule can be asserted.
pub(crate) struct MockAsr {
    loaded: AtomicBool,
    /// What the session layer last said about a meeting needing the engine.
    resident: AtomicBool,
    calls: AtomicU32,
    catch_up_segments: AtomicU32,
    /// Hold inside the catch-up pass until something cancels it: how a real
    /// pass behaves when the next meeting starts while it is still reading.
    catch_up_waits_to_be_stopped: AtomicBool,
    /// Set the moment the catch-up pass begins, so a test can wait for the job
    /// to really be in flight before it takes the machine away.
    catch_up_started: AtomicBool,
    fail_prewarm: AtomicBool,
    /// The load gives up because something else wanted the machine — what a
    /// recording starting mid-load looks like from here.
    cancel_prewarm: AtomicBool,
    /// How long loading takes, the way a first-ever launch takes minutes.
    prewarm_takes: std::sync::Mutex<Option<Duration>>,
    /// How long one live decode takes, so a test can still have one in flight
    /// when something else happens.
    transcribe_takes: std::sync::Mutex<Option<Duration>>,
    text: std::sync::Mutex<String>,
    /// Rows the next live backlog pass writes: (t_start_ms, t_end_ms, text).
    backlog_writes: std::sync::Mutex<Vec<(i64, i64, String)>>,
    /// The `to_ms` each live backlog pass was asked for.
    backlog_calls: std::sync::Mutex<Vec<i64>>,
    /// What was put in front of the audio on each live decode, and what kind of
    /// decode it was — so a test can check which lanes are told the words Echo
    /// should know (see [`crate::asr::glossary`]).
    live_plans: std::sync::Mutex<Vec<(crate::asr::engine::JobKind, Option<String>)>>,
    /// Meetings the session layer asked the engine to forget, in order. The
    /// real engine holds a meeting's settled language in memory as well as on
    /// the row, so "was this one forgotten" is the only way to tell that a
    /// wrong language is really gone.
    forgotten: std::sync::Mutex<Vec<String>>,
    /// Stretches the next catch-up pass reports it could not read back at all.
    /// Empty unless a test says otherwise, because that is the ordinary case.
    catch_up_unread: std::sync::Mutex<Vec<UnreadSpan>>,
    /// What the engine has settled on for each meeting. Empty until a test says
    /// otherwise: the real engine settles on evidence, not on the first line it
    /// managed to write (see [`crate::asr::language`]).
    settled: std::sync::Mutex<std::collections::HashMap<String, String>>,
}

impl MockAsr {
    pub(crate) fn new() -> Self {
        Self {
            loaded: AtomicBool::new(false),
            resident: AtomicBool::new(false),
            calls: AtomicU32::new(0),
            catch_up_segments: AtomicU32::new(0),
            catch_up_waits_to_be_stopped: AtomicBool::new(false),
            catch_up_started: AtomicBool::new(false),
            fail_prewarm: AtomicBool::new(false),
            cancel_prewarm: AtomicBool::new(false),
            prewarm_takes: std::sync::Mutex::new(None),
            transcribe_takes: std::sync::Mutex::new(None),
            text: std::sync::Mutex::new("hello there".to_string()),
            backlog_writes: std::sync::Mutex::new(Vec::new()),
            backlog_calls: std::sync::Mutex::new(Vec::new()),
            live_plans: std::sync::Mutex::new(Vec::new()),
            catch_up_unread: std::sync::Mutex::new(Vec::new()),
            forgotten: std::sync::Mutex::new(Vec::new()),
            settled: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// The engine has worked out what language this meeting is in — what a real
    /// one does once it has heard enough speech agreeing.
    #[allow(dead_code)]
    pub(crate) fn settles_on(&self, meeting_id: &str, language: &str) {
        self.settled
            .lock()
            .unwrap()
            .insert(meeting_id.to_string(), language.to_string());
    }

    /// Which meetings the engine was told to forget, oldest first.
    #[allow(dead_code)]
    pub(crate) fn forgotten(&self) -> Vec<String> {
        self.forgotten.lock().unwrap().clone()
    }

    /// The prompts handed to the live lanes so far, newest last.
    #[allow(dead_code)]
    pub(crate) fn live_plans(&self) -> Vec<(crate::asr::engine::JobKind, Option<String>)> {
        self.live_plans.lock().unwrap().clone()
    }

    #[allow(dead_code)]
    pub(crate) fn calls(&self) -> u32 {
        self.calls.load(Ordering::SeqCst)
    }

    /// Make the catch-up pass sit there until it is told to stop, the way a real
    /// one does while it still has audio to read.
    #[allow(dead_code)]
    pub(crate) fn catch_up_waits_to_be_stopped(&self) {
        self.catch_up_waits_to_be_stopped
            .store(true, Ordering::SeqCst);
    }

    /// The next catch-up pass finishes, but with these stretches unread — an
    /// engine that refused the same window twice.
    pub(crate) fn catch_up_leaves_unread(&self, spans: Vec<UnreadSpan>) {
        *self.catch_up_unread.lock().unwrap() = spans;
    }

    /// Has a catch-up pass begun?
    #[allow(dead_code)]
    pub(crate) fn catch_up_started(&self) -> bool {
        self.catch_up_started.load(Ordering::SeqCst)
    }

    #[allow(dead_code)]
    pub(crate) fn fail_prewarm(&self) {
        self.fail_prewarm.store(true, Ordering::SeqCst);
    }

    /// The load stops short because something else claimed the machine, rather
    /// than because anything went wrong with it.
    #[allow(dead_code)]
    pub(crate) fn cancel_prewarm(&self) {
        self.cancel_prewarm.store(true, Ordering::SeqCst);
    }

    /// Is a meeting holding the engine right now?
    pub(crate) fn is_resident(&self) -> bool {
        self.resident.load(Ordering::SeqCst)
    }

    /// The weights take this long to come up, so a test can be the person who
    /// pressed Start while Echo was still getting ready.
    pub(crate) fn prewarm_takes(&self, how_long: Duration) {
        *self.prewarm_takes.lock().unwrap() = Some(how_long);
    }

    /// Every decode takes this long, so a test can be the meeting where a live
    /// utterance is still in the engine when the backlog pass claims its stretch.
    ///
    /// Deliberately *not* cancellable: this stands in for the engine whose abort
    /// landed a moment too late and answered anyway, which is the only way a live
    /// final can still turn up below the floor.
    pub(crate) fn transcribe_takes(&self, how_long: Duration) {
        *self.transcribe_takes.lock().unwrap() = Some(how_long);
    }

    /// What every decode from now on comes back with. Lets a test be the
    /// meeting where the engine answered a pause with "Grazie." (2026-08-24).
    pub(crate) fn says(&self, text: &str) {
        *self.text.lock().unwrap() = text.to_string();
    }

    /// What the next live backlog pass finds on disk and writes down.
    pub(crate) fn backlog_writes(&self, rows: &[(i64, i64, &str)]) {
        *self.backlog_writes.lock().unwrap() = rows
            .iter()
            .map(|(from, to, text)| (*from, *to, (*text).to_string()))
            .collect();
    }

    /// How far each live backlog pass was asked to read.
    pub(crate) fn backlog_calls(&self) -> Vec<i64> {
        self.backlog_calls.lock().unwrap().clone()
    }
}

impl AsrPort for MockAsr {
    fn prewarm<'a>(&'a self) -> BoxFuture<'a, Result<(), AsrError>> {
        Box::pin(async move {
            let takes = *self.prewarm_takes.lock().unwrap();
            if let Some(takes) = takes {
                tokio::time::sleep(takes).await;
            }
            if self.cancel_prewarm.load(Ordering::SeqCst) {
                return Err(AsrError::Cancelled);
            }
            if self.fail_prewarm.load(Ordering::SeqCst) {
                return Err(AsrError::NotInstalled);
            }
            self.loaded.store(true, Ordering::SeqCst);
            Ok(())
        })
    }

    fn hold_resident(&self, resident: bool) {
        self.resident.store(resident, Ordering::SeqCst);
    }

    fn settled_language(&self, meeting_id: &str) -> Option<String> {
        self.settled.lock().unwrap().get(meeting_id).cloned()
    }

    fn transcribe_live<'a>(
        &'a self,
        job: TranscribeJob,
        plan: crate::asr::engine::DecodePlan,
        on_partial: Option<PartialFn>,
    ) -> BoxFuture<'a, Result<Transcription, AsrError>> {
        self.live_plans
            .lock()
            .unwrap()
            .push((plan.kind, plan.prompt.clone()));
        self.transcribe(job, on_partial)
    }

    fn transcribe<'a>(
        &'a self,
        job: TranscribeJob,
        on_partial: Option<PartialFn>,
    ) -> BoxFuture<'a, Result<Transcription, AsrError>> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.loaded.store(true, Ordering::SeqCst);
            let takes = *self.transcribe_takes.lock().unwrap();
            if let Some(takes) = takes {
                tokio::time::sleep(takes).await;
            }
            let text = self.text.lock().unwrap().clone();
            if let Some(on_partial) = on_partial {
                on_partial(&text);
            }
            Ok(Transcription {
                channel: job.channel,
                t_start_ms: job.t_start_ms,
                t_end_ms: job.t_start_ms + 2_000,
                text,
                language: Some("en".into()),
                language_confidence: Some(0.99),
                language_inherited: false,
                avg_confidence: Some(0.9),
                model_name: Some("test".into()),
                model_revision: Some("1".into()),
                lines: Vec::new(),
            })
        })
    }

    fn forget_meeting(&self, meeting_id: &str) {
        self.forgotten.lock().unwrap().push(meeting_id.to_string());
    }

    fn catch_up<'a>(
        &'a self,
        _db: &'a Db,
        _meeting_id: &'a str,
        _not_before_ms: Option<i64>,
        control: crate::session::ports::CatchUpControl,
    ) -> BoxFuture<'a, Result<CatchUpReport, AsrError>> {
        Box::pin(async move {
            self.catch_up_started.store(true, Ordering::SeqCst);
            while self.catch_up_waits_to_be_stopped.load(Ordering::SeqCst)
                && !control.cancel.as_ref().is_some_and(|c| c())
            {
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
            // A pass that was stopped part-way has one thing it may say about
            // itself, and a count of what it managed is not it.
            if control.cancel.as_ref().is_some_and(|c| c()) {
                return Err(AsrError::Cancelled);
            }
            if let Some(report) = &control.on_progress {
                report(1.0);
            }
            Ok(CatchUpReport {
                segments_written: self.catch_up_segments.load(Ordering::SeqCst),
                unread: self.catch_up_unread.lock().unwrap().clone(),
                ..CatchUpReport::default()
            })
        })
    }

    /// Writes whatever the test said is on disk, and remembers how far it was
    /// asked to read. Only stretches with no final text against them: the real
    /// pass subtracts the coverage, and a double-transcribing mock would hide
    /// exactly the bug that matters here.
    fn catch_up_live<'a>(
        &'a self,
        db: &'a Db,
        meeting_id: &'a str,
        to_ms: i64,
        control: crate::session::ports::CatchUpControl,
    ) -> BoxFuture<'a, Result<u32, AsrError>> {
        Box::pin(async move {
            self.backlog_calls.lock().unwrap().push(to_ms);
            if control.cancel.as_ref().is_some_and(|c| c()) {
                return Err(AsrError::Cancelled);
            }
            let rows = self.backlog_writes.lock().unwrap().clone();
            let covered = crate::db::repo::get_segments(
                db,
                &crate::types::TranscriptQuery {
                    meeting_id: meeting_id.to_string(),
                    include_partial: Some(false),
                    ..Default::default()
                },
            )
            .await
            .unwrap_or_default();
            let mut written = 0;
            for (from_ms, to, text) in rows {
                if to > to_ms {
                    continue;
                }
                if covered
                    .iter()
                    .any(|s| s.t_end_ms > from_ms && s.t_start_ms < to)
                {
                    continue;
                }
                let draft = crate::types::SegmentDraft {
                    meeting_id: meeting_id.to_string(),
                    t_start_ms: from_ms,
                    t_end_ms: to,
                    channel: Channel::Mic,
                    speaker_id: None,
                    text,
                    language: Some("en".into()),
                    avg_confidence: Some(0.9),
                    revision: 1,
                    is_final: true,
                    model_name: Some("test".into()),
                    model_revision: Some("1".into()),
                    corrections: Vec::new(),
                };
                if crate::db::repo::insert_segment(db, &draft).await.is_ok() {
                    written += 1;
                }
            }
            Ok(written)
        })
    }

    fn release<'a>(&'a self) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            self.loaded.store(false, Ordering::SeqCst);
        })
    }

    fn is_loaded(&self) -> bool {
        self.loaded.load(Ordering::SeqCst)
    }

    fn queue_depth(&self) -> u32 {
        0
    }
}

// ---------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------

/// Records the order jobs ran in, and can hold one open so preemption has
/// something to interrupt.
#[derive(Default)]
pub(crate) struct TestExecutor {
    started: std::sync::Mutex<Vec<JobKind>>,
    finished: std::sync::Mutex<Vec<JobKind>>,
    block: AtomicBool,
    parked: AtomicU32,
}

impl TestExecutor {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Jobs from now on sit there until [`TestExecutor::stop_blocking`].
    pub(crate) fn block_until_told(&self) {
        self.block.store(true, Ordering::SeqCst);
    }

    pub(crate) fn stop_blocking(&self) {
        self.block.store(false, Ordering::SeqCst);
    }

    pub(crate) fn finished(&self) -> Vec<JobKind> {
        self.finished.lock().unwrap().clone()
    }

    pub(crate) async fn wait_until_running(&self, kind: JobKind) {
        wait_for(&format!("{kind:?} to start"), || {
            self.started.lock().unwrap().contains(&kind)
        })
        .await;
    }

    pub(crate) async fn wait_for_parked(&self) {
        wait_for("work to be parked", || {
            self.parked.load(Ordering::SeqCst) > 0
        })
        .await;
    }

    pub(crate) async fn wait_for_kinds(&self, count: usize) {
        wait_for(&format!("{count} jobs to finish"), || {
            self.finished.lock().unwrap().len() >= count
        })
        .await;
    }
}

impl JobExecutor for TestExecutor {
    fn run<'a>(&'a self, ctx: &'a JobContext) -> BoxFuture<'a, Result<(), JobFailure>> {
        Box::pin(async move {
            self.started.lock().unwrap().push(ctx.job.kind);
            loop {
                if let Err(interrupted) = ctx.cancel.check() {
                    if matches!(interrupted, JobFailure::Preempted) {
                        self.parked.fetch_add(1, Ordering::SeqCst);
                    }
                    return Err(interrupted);
                }
                if !self.block.load(Ordering::SeqCst) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            ctx.progress.set(0.5).await;
            self.finished.lock().unwrap().push(ctx.job.kind);
            Ok(())
        })
    }
}

async fn wait_for(what: &str, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + PATIENCE;
    while Instant::now() < deadline {
        if done() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("waited too long for {what}");
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/// Keeps everything the session emitted so a test can look at it.
#[derive(Default)]
pub(crate) struct CollectingEvents {
    seen: std::sync::Mutex<Vec<UiEvent>>,
}

impl CollectingEvents {
    pub(crate) fn names(&self) -> Vec<&'static str> {
        self.seen
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|e| e.name())
            .collect()
    }

    /// Did a banner with this machine tag go out?
    pub(crate) fn notice_tagged(&self, tag: &str) -> bool {
        self.seen
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .any(|event| match event {
                UiEvent::Notice(payload) => payload.tag.as_deref() == Some(tag),
                _ => false,
            })
    }

    /// Every banner that went out, in order — the words, not just the tag.
    pub(crate) fn notices(&self) -> Vec<crate::events::NoticePayload> {
        self.seen
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter_map(|event| match event {
                UiEvent::Notice(payload) => Some(payload.clone()),
                _ => None,
            })
            .collect()
    }

    /// Every job announcement, in order.
    pub(crate) fn job_progress(&self) -> Vec<crate::events::JobProgressPayload> {
        self.seen
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter_map(|event| match event {
                UiEvent::JobProgress(payload) => Some(payload.clone()),
                _ => None,
            })
            .collect()
    }

    #[allow(dead_code)]
    pub(crate) fn count(&self, name: &str) -> usize {
        self.names().iter().filter(|n| **n == name).count()
    }

    /// The "everything you hold for this meeting is stale" announcements, in
    /// order.
    pub(crate) fn transcript_revisions(&self) -> Vec<crate::events::TranscriptRevisedPayload> {
        self.seen
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter_map(|event| match event {
                UiEvent::TranscriptRevised(payload) => Some(payload.clone()),
                _ => None,
            })
            .collect()
    }

    /// The speaker-list announcements, in order. Each carries how many people
    /// Echo believes were in the meeting at that moment.
    #[allow(dead_code)]
    pub(crate) fn speaker_updates(&self) -> Vec<crate::events::SpeakersUpdatedPayload> {
        self.seen
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter_map(|event| match event {
                UiEvent::SpeakersUpdated(payload) => Some(payload.clone()),
                _ => None,
            })
            .collect()
    }

    /// Every capture status that went out, in order — the whole story of one
    /// meeting as the screen saw it.
    #[allow(dead_code)]
    pub(crate) fn capture_states(&self) -> Vec<crate::types::CaptureStatus> {
        self.seen
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter_map(|event| match event {
                UiEvent::CaptureState(payload) => Some(payload.clone()),
                _ => None,
            })
            .collect()
    }

    /// The live lines that went out, in the order they were sent — including
    /// the empty `dropped` ones, which are how a line that will never get text
    /// is retired.
    #[allow(dead_code)]
    pub(crate) fn partials(&self) -> Vec<crate::events::TranscriptPartialPayload> {
        self.seen
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter_map(|event| match event {
                UiEvent::TranscriptPartial(payload) => Some(payload.clone()),
                _ => None,
            })
            .collect()
    }

    /// The transcript lines that went out, in the order they were sent.
    pub(crate) fn finals(&self) -> Vec<crate::events::TranscriptFinalPayload> {
        self.seen
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter_map(|event| match event {
                UiEvent::TranscriptFinal(payload) => Some(payload.clone()),
                _ => None,
            })
            .collect()
    }
}

impl EventSink for CollectingEvents {
    fn emit(&self, event: UiEvent) {
        self.seen
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(event);
    }
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

/// A session with every seam mocked, over a real (in-memory) database and a real
/// temporary storage directory.
pub(crate) struct Harness {
    pub(crate) db: Db,
    pub(crate) paths: AppPaths,
    pub(crate) session: SessionManager,
    pub(crate) capture: Arc<MockCapture>,
    #[allow(dead_code)]
    pub(crate) asr: Arc<MockAsr>,
    pub(crate) executor: Arc<TestExecutor>,
    pub(crate) events: Arc<CollectingEvents>,
    _storage: tempfile::TempDir,
}

impl Harness {
    pub(crate) async fn new() -> Self {
        let db = crate::db::connect_in_memory()
            .await
            .expect("in-memory database");
        let storage = tempfile::tempdir().expect("temporary storage");
        let paths = AppPaths::rooted_at(storage.path().to_path_buf(), None);
        paths.ensure().expect("storage directories");

        let capture = Arc::new(MockCapture::new());
        let asr = Arc::new(MockAsr::new());
        capture.watch_engine(asr.clone());
        let executor = Arc::new(TestExecutor::new());
        let events = Arc::new(CollectingEvents::default());
        let bus = EventBus::new();
        bus.set(events.clone());

        let ports = Ports {
            capture: capture.clone(),
            asr: asr.clone(),
            executor: executor.clone(),
            events: bus,
        };
        let session = SessionManager::with_ports(db.clone(), paths.clone(), ports);

        Self {
            db,
            paths,
            session,
            capture,
            asr,
            executor,
            events,
            _storage: storage,
        }
    }

    /// Give the spawned tasks a moment to catch up.
    pub(crate) async fn settle(&self) {
        for _ in 0..4 {
            tokio::task::yield_now().await;
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// Wait for something the spine does on a spawned task.
    pub(crate) async fn wait_until(&self, what: &str, done: impl FnMut() -> bool) {
        wait_for(what, done).await;
    }

    /// Journal one committed chunk, as the writer would once a piece of audio is
    /// flushed and fsynced.
    pub(crate) async fn commit_chunk(&self, meeting_id: &str, t_start_ms: i64, t_end_ms: i64) {
        self.capture.send(CaptureSignal::ChunkCommitted(
            crate::audio::writer::CommittedChunk {
                channel: Channel::Mic,
                seq: 0,
                path: self.paths.chunk_path(meeting_id, Channel::Mic, 0),
                t_start_ms,
                t_end_ms,
            },
        ));
        self.settle().await;
    }

    /// A minute of committed audio: enough that the meeting is obviously worth
    /// keeping, for tests that are about something else.
    pub(crate) async fn record_a_minute(&self, meeting_id: &str) {
        self.commit_chunk(meeting_id, 0, 60_000).await;
    }

    /// The kinds of work queued for a meeting.
    pub(crate) async fn queued_kinds(&self, meeting_id: &str) -> Vec<JobKind> {
        crate::db::repo::list_jobs(
            &self.db,
            &crate::types::JobQuery {
                meeting_id: Some(meeting_id.to_string()),
                ..Default::default()
            },
        )
        .await
        .expect("the work queue")
        .into_iter()
        .map(|job| job.kind)
        .collect()
    }
}
