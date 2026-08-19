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
        }
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
            self.opens.fetch_add(1, Ordering::SeqCst);
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
    calls: AtomicU32,
    catch_up_segments: AtomicU32,
    fail_prewarm: AtomicBool,
    text: std::sync::Mutex<String>,
}

impl MockAsr {
    pub(crate) fn new() -> Self {
        Self {
            loaded: AtomicBool::new(false),
            calls: AtomicU32::new(0),
            catch_up_segments: AtomicU32::new(0),
            fail_prewarm: AtomicBool::new(false),
            text: std::sync::Mutex::new("hello there".to_string()),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn calls(&self) -> u32 {
        self.calls.load(Ordering::SeqCst)
    }

    #[allow(dead_code)]
    pub(crate) fn fail_prewarm(&self) {
        self.fail_prewarm.store(true, Ordering::SeqCst);
    }
}

impl AsrPort for MockAsr {
    fn prewarm<'a>(&'a self) -> BoxFuture<'a, Result<(), AsrError>> {
        Box::pin(async move {
            if self.fail_prewarm.load(Ordering::SeqCst) {
                return Err(AsrError::NotInstalled);
            }
            self.loaded.store(true, Ordering::SeqCst);
            Ok(())
        })
    }

    fn transcribe<'a>(
        &'a self,
        job: TranscribeJob,
        on_partial: Option<PartialFn>,
    ) -> BoxFuture<'a, Result<Transcription, AsrError>> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.loaded.store(true, Ordering::SeqCst);
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
                avg_confidence: Some(0.9),
                model_name: Some("test".into()),
                model_revision: Some("1".into()),
            })
        })
    }

    fn catch_up<'a>(
        &'a self,
        _db: &'a Db,
        _meeting_id: &'a str,
        _not_before_ms: Option<i64>,
        control: crate::session::ports::CatchUpControl,
    ) -> BoxFuture<'a, Result<u32, AsrError>> {
        Box::pin(async move {
            if control.cancel.as_ref().is_some_and(|c| c()) {
                return Err(AsrError::Cancelled);
            }
            if let Some(report) = &control.on_progress {
                report(1.0);
            }
            Ok(self.catch_up_segments.load(Ordering::SeqCst))
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
        self.seen.lock().unwrap().iter().map(|e| e.name()).collect()
    }

    /// Did a banner with this machine tag go out?
    pub(crate) fn notice_tagged(&self, tag: &str) -> bool {
        self.seen.lock().unwrap().iter().any(|event| match event {
            UiEvent::Notice(payload) => payload.tag.as_deref() == Some(tag),
            _ => false,
        })
    }

    #[allow(dead_code)]
    pub(crate) fn count(&self, name: &str) -> usize {
        self.names().iter().filter(|n| **n == name).count()
    }
}

impl EventSink for CollectingEvents {
    fn emit(&self, event: UiEvent) {
        self.seen.lock().unwrap().push(event);
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
}
