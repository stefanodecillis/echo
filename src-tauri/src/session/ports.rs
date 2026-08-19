//! The seams between the session spine and the modules it drives.
//!
//! The spine has to be testable without a microphone and without 1.6 GB of
//! weights on disk, so every module it touches is reached through a small trait:
//!
//! * [`CapturePort`] opens a recording and hands back a [`CaptureFeed`].
//! * [`AsrPort`] turns utterances into text and owns the load/unload lifecycle.
//! * [`EventSink`] carries [`UiEvent`]s to the webview.
//!
//! The real implementations here are thin adapters over [`crate::audio`] and
//! [`crate::asr`]; the mocks live next to the tests. Nothing in this file holds
//! a resource: an adapter is a handful of pointers until something calls `open`
//! or `prewarm` (mantra 1).

use std::sync::Arc;

use futures::future::BoxFuture;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use crate::asr::engine::{EngineWorker, PartialFn};
use crate::asr::{AsrError, TranscribeJob, Transcription};
use crate::audio::{AudioError, CaptureConfig, CaptureSession, CaptureStarted};
use crate::db::Db;
use crate::events;
use crate::types::{AssetKind, Channel};

// ---------------------------------------------------------------------------
// Capture
// ---------------------------------------------------------------------------

/// What a running capture tells the session layer. Owned by [`crate::audio`],
/// which already phrases anything user-visible as a plain sentence (mantra 2).
pub use crate::audio::CaptureSignal;

/// Receiving half of the capture feed, owned by the session pipeline. Unbounded
/// on purpose: the capture threads may never block, and every message is small.
pub type CaptureFeed = UnboundedReceiver<CaptureSignal>;
/// Sending half, owned by the capture backend.
pub type CaptureSender = UnboundedSender<CaptureSignal>;

/// A live capture: what started, how to steer it, and its update feed.
pub struct CaptureStart {
    pub handle: Box<dyn CaptureHandle>,
    pub started: CaptureStarted,
    pub feed: CaptureFeed,
}

/// Steering a running capture.
pub trait CaptureHandle: Send + Sync + 'static {
    /// Offset on the monotonic meeting clock.
    fn elapsed_ms(&self) -> i64;
    /// Smoothed 0.0..=1.0 loudness, mic then system.
    fn levels(&self) -> (f32, f32);
    /// Channels producing audio right now.
    fn active_channels(&self) -> Vec<Channel>;
    /// Utterances capture is holding for the speech engine.
    fn pending_utterances(&self) -> u32;
    /// Idempotent.
    fn pause(&self) -> BoxFuture<'_, Result<(), AudioError>>;
    /// Idempotent.
    fn resume(&self) -> BoxFuture<'_, Result<(), AudioError>>;
    /// Close the devices, flush and commit every open chunk. Returns the
    /// duration on the meeting clock.
    fn stop(self: Box<Self>) -> BoxFuture<'static, Result<i64, AudioError>>;
}

/// Opens recordings. One implementation talks to the real devices; the test one
/// hands the test the sending half of the feed.
pub trait CapturePort: Send + Sync + 'static {
    fn open<'a>(&'a self, cfg: CaptureConfig) -> BoxFuture<'a, Result<CaptureStart, AudioError>>;
}

/// The real devices, via [`crate::audio::CaptureSession`].
pub struct DeviceCapture {
    db: Db,
}

impl DeviceCapture {
    pub fn new(db: Db) -> Self {
        Self { db }
    }
}

impl CapturePort for DeviceCapture {
    fn open<'a>(&'a self, cfg: CaptureConfig) -> BoxFuture<'a, Result<CaptureStart, AudioError>> {
        Box::pin(async move {
            // A missing detector is not a reason to refuse to record: capture
            // falls back to a loudness gate, which is what makes recording on
            // the first launch possible at all.
            let detector = crate::asr::models::installed_path(&self.db, AssetKind::SpeechDetector)
                .await
                .ok()
                .flatten();
            let (session, started) = CaptureSession::start_with_detector(cfg, detector).await?;
            let feed = match session.take_signals() {
                Some(feed) => feed,
                None => {
                    // Someone already took it. Rather than fail a recording over
                    // a missing live transcript, carry on with a dead feed: the
                    // audio still reaches disk and catch-up still runs.
                    tracing::warn!("capture signals were already taken");
                    let (_tx, rx) = tokio::sync::mpsc::unbounded_channel();
                    rx
                }
            };
            Ok(CaptureStart {
                handle: Box::new(DeviceHandle { session }),
                started,
                feed,
            })
        })
    }
}

struct DeviceHandle {
    session: CaptureSession,
}

impl CaptureHandle for DeviceHandle {
    fn elapsed_ms(&self) -> i64 {
        self.session.elapsed_ms()
    }

    fn levels(&self) -> (f32, f32) {
        self.session.levels()
    }

    fn active_channels(&self) -> Vec<Channel> {
        self.session.active_channels()
    }

    fn pending_utterances(&self) -> u32 {
        self.session.pending_utterances()
    }

    fn pause(&self) -> BoxFuture<'_, Result<(), AudioError>> {
        Box::pin(self.session.pause())
    }

    fn resume(&self) -> BoxFuture<'_, Result<(), AudioError>> {
        Box::pin(self.session.resume())
    }

    fn stop(self: Box<Self>) -> BoxFuture<'static, Result<i64, AudioError>> {
        Box::pin(async move { self.session.stop().await })
    }
}

// ---------------------------------------------------------------------------
// Speech
// ---------------------------------------------------------------------------

/// How the job runner steers a catch-up pass while it runs.
///
/// Without this, catch-up is a black box: it cannot be stopped halfway and the
/// person sees a progress bar that never moves. Both halves are optional so a
/// caller that only wants the pass to happen can pass [`CatchUpControl::none`].
#[derive(Clone, Default)]
pub struct CatchUpControl {
    /// Asked between windows. `true` means stop and stay resumable.
    pub cancel: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
    /// 0.0..=1.0 through the pass. Called from the pass's own task, so it must
    /// not block.
    pub on_progress: Option<Arc<dyn Fn(f32) + Send + Sync>>,
}

impl CatchUpControl {
    /// Run to completion, report nothing.
    pub fn none() -> Self {
        Self::default()
    }
}

impl std::fmt::Debug for CatchUpControl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CatchUpControl")
            .field("cancel", &self.cancel.is_some())
            .field("on_progress", &self.on_progress.is_some())
            .finish()
    }
}

/// Turning audio into text, plus the load/unload lifecycle.
pub trait AsrPort: Send + Sync + 'static {
    /// Load now, so the first words of a meeting are not slow. Failing to
    /// prewarm never ends a recording: the audio is still being written.
    fn prewarm<'a>(&'a self) -> BoxFuture<'a, Result<(), AsrError>>;

    /// One utterance. Returns [`AsrError::Cancelled`] when the job was dropped
    /// to keep capture healthy.
    fn transcribe<'a>(
        &'a self,
        job: TranscribeJob,
        on_partial: Option<PartialFn>,
    ) -> BoxFuture<'a, Result<Transcription, AsrError>>;

    /// Transcribe whatever the committed chunks on disk have no text against
    /// yet, per channel. This is what makes "audio on disk is the source of
    /// truth" real, and what makes the catch-up job resumable.
    ///
    /// `not_before_ms` is a floor for callers that know nothing earlier matters;
    /// `None` means "look at the whole meeting". It is deliberately not a
    /// starting point: an utterance the live queue dropped sits *inside* the
    /// transcript, and the two channels reach different lengths, so one offset
    /// for both would quietly lose words that are sitting on disk.
    fn catch_up<'a>(
        &'a self,
        db: &'a Db,
        meeting_id: &'a str,
        not_before_ms: Option<i64>,
        control: CatchUpControl,
    ) -> BoxFuture<'a, Result<u32, AsrError>>;

    /// Give the memory back (mantra 1).
    fn release<'a>(&'a self) -> BoxFuture<'a, ()>;

    fn is_loaded(&self) -> bool;

    /// Utterances waiting. Surfaced to the person only as "catching up".
    fn queue_depth(&self) -> u32;

    /// Drop queued work for a meeting that no longer exists. Default: nothing to
    /// forget.
    fn forget_meeting(&self, _meeting_id: &str) {}

    /// On the way out: stop the engine thread as well as freeing the weights.
    /// Default: the same as [`AsrPort::release`].
    fn shutdown<'a>(&'a self) -> BoxFuture<'a, ()> {
        self.release()
    }

    /// What actually initialised, for Settings → Advanced. `None` when nothing
    /// has been asked to load yet, which is the usual case.
    fn backend(&self) -> Option<crate::asr::engine::BackendReport> {
        None
    }
}

/// The real speech engine, via [`crate::asr::engine::EngineWorker`].
pub struct EngineAsr {
    worker: EngineWorker,
    db: Db,
}

impl EngineAsr {
    /// Creates the worker but loads nothing.
    pub fn new(db: Db, idle_release_minutes: u32) -> Self {
        Self {
            worker: EngineWorker::new(idle_release_minutes),
            db,
        }
    }
}

impl AsrPort for EngineAsr {
    fn prewarm<'a>(&'a self) -> BoxFuture<'a, Result<(), AsrError>> {
        Box::pin(async move {
            // Which weights to load is the speech module's business: it reads the
            // selected quality preset and the installed files itself, so the
            // provenance written onto every segment stays honest.
            self.worker.configure_from_settings(&self.db).await?;
            let backend = self.worker.load_now().await?;
            tracing::info!(?backend, "speech understanding is ready");
            Ok(())
        })
    }

    fn transcribe<'a>(
        &'a self,
        job: TranscribeJob,
        on_partial: Option<PartialFn>,
    ) -> BoxFuture<'a, Result<Transcription, AsrError>> {
        Box::pin(async move { self.worker.submit(job, on_partial).await })
    }

    fn catch_up<'a>(
        &'a self,
        db: &'a Db,
        meeting_id: &'a str,
        not_before_ms: Option<i64>,
        control: CatchUpControl,
    ) -> BoxFuture<'a, Result<u32, AsrError>> {
        Box::pin(async move {
            // `pause_while` is deliberately left unset. A recording preempts this
            // job through the same cancel flag, which parks the row and frees the
            // engine outright; waiting in place would keep the weights resident
            // for the whole meeting (mantra 1).
            let options = crate::asr::catchup::CatchUpOptions {
                from_ms: not_before_ms.map(|ms| ms.max(0)),
                cancel: control.cancel,
                on_progress: control.on_progress,
                ..Default::default()
            };
            let report = crate::asr::catchup::run(
                &self.worker,
                db,
                meeting_id,
                options,
                &crate::asr::catchup::DiskAudio,
            )
            .await?;
            Ok(report.segments_written)
        })
    }

    fn release<'a>(&'a self) -> BoxFuture<'a, ()> {
        Box::pin(async move { self.worker.release().await })
    }

    fn is_loaded(&self) -> bool {
        self.worker.is_loaded()
    }

    fn queue_depth(&self) -> u32 {
        self.worker.queue_depth()
    }

    fn forget_meeting(&self, meeting_id: &str) {
        self.worker.forget_meeting(meeting_id);
    }

    fn shutdown<'a>(&'a self) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            self.worker.release().await;
            // Joins the engine thread. Blocking is only acceptable because this
            // is the last thing that happens before the process goes away.
            self.worker.shutdown();
        })
    }

    fn backend(&self) -> Option<crate::asr::engine::BackendReport> {
        Some(self.worker.backend())
    }
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/// Everything the spine tells the UI. One variant per event name in
/// [`crate::events`]; the payloads are the shared serde types, so the sink only
/// has to pick a name.
#[derive(Debug, Clone)]
pub enum UiEvent {
    CaptureState(events::CaptureStatePayload),
    TranscriptPartial(events::TranscriptPartialPayload),
    TranscriptFinal(events::TranscriptFinalPayload),
    TranscriptRevised(events::TranscriptRevisedPayload),
    AudioLevels(events::AudioLevelsPayload),
    JobProgress(events::JobProgressPayload),
    DownloadProgress(events::DownloadProgressPayload),
    SpeakersUpdated(events::SpeakersUpdatedPayload),
    SummaryReady(events::SummaryReadyPayload),
    ActionItemsUpdated(events::ActionItemsUpdatedPayload),
    MeetingUpdated(events::MeetingUpdatedPayload),
    Notice(events::NoticePayload),
    /// Settings changed somewhere. Any screen showing a setting refetches, so
    /// two windows (or the tray) cannot drift apart.
    SettingsChanged(crate::types::Settings),
    TrayState(crate::types::TrayState),
    RecoveryAvailable(events::RecoveryAvailablePayload),
}

impl UiEvent {
    /// The `echo://` name this goes out under.
    pub fn name(&self) -> &'static str {
        match self {
            UiEvent::CaptureState(_) => events::CAPTURE_STATE,
            UiEvent::TranscriptPartial(_) => events::TRANSCRIPT_PARTIAL,
            UiEvent::TranscriptFinal(_) => events::TRANSCRIPT_FINAL,
            UiEvent::TranscriptRevised(_) => events::TRANSCRIPT_REVISED,
            UiEvent::AudioLevels(_) => events::AUDIO_LEVELS,
            UiEvent::JobProgress(_) => events::JOB_PROGRESS,
            UiEvent::DownloadProgress(_) => events::DOWNLOAD_PROGRESS,
            UiEvent::SpeakersUpdated(_) => events::SPEAKERS_UPDATED,
            UiEvent::SummaryReady(_) => events::SUMMARY_READY,
            UiEvent::ActionItemsUpdated(_) => events::ACTION_ITEMS_UPDATED,
            UiEvent::MeetingUpdated(_) => events::MEETING_UPDATED,
            UiEvent::Notice(_) => events::NOTICE,
            UiEvent::SettingsChanged(_) => events::SETTINGS_CHANGED,
            UiEvent::TrayState(_) => events::TRAY_STATE,
            UiEvent::RecoveryAvailable(_) => events::RECOVERY_AVAILABLE,
        }
    }
}

/// Where [`UiEvent`]s go. Must never block: it is called from the capture
/// pipeline.
pub trait EventSink: Send + Sync + 'static {
    fn emit(&self, event: UiEvent);
}

/// Swappable sink. The session is built before the window exists, so it starts
/// with [`SilentEvents`] and the real sink is attached in `lib.rs`.
pub struct EventBus {
    sink: std::sync::RwLock<Arc<dyn EventSink>>,
}

impl Default for EventBus {
    fn default() -> Self {
        Self {
            sink: std::sync::RwLock::new(Arc::new(SilentEvents)),
        }
    }
}

impl EventBus {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Point the bus at a real sink. Called once, at launch.
    pub fn set(&self, sink: Arc<dyn EventSink>) {
        if let Ok(mut current) = self.sink.write() {
            *current = sink;
        }
    }
}

impl EventSink for EventBus {
    fn emit(&self, event: UiEvent) {
        if let Ok(sink) = self.sink.read() {
            sink.emit(event);
        }
    }
}

/// Drops everything. The default until a window exists.
#[derive(Debug, Default, Clone, Copy)]
pub struct SilentEvents;

impl EventSink for SilentEvents {
    fn emit(&self, _event: UiEvent) {}
}

/// Emits to the webview, and keeps the tray icon in step.
pub struct TauriEvents {
    app: tauri::AppHandle,
}

impl TauriEvents {
    pub fn new(app: tauri::AppHandle) -> Self {
        Self { app }
    }
}

impl EventSink for TauriEvents {
    fn emit(&self, event: UiEvent) {
        use tauri::Emitter;
        let name = event.name();
        let sent = match event {
            UiEvent::CaptureState(p) => self.app.emit(name, p),
            UiEvent::TranscriptPartial(p) => self.app.emit(name, p),
            UiEvent::TranscriptFinal(p) => self.app.emit(name, p),
            UiEvent::TranscriptRevised(p) => self.app.emit(name, p),
            UiEvent::AudioLevels(p) => self.app.emit(name, p),
            UiEvent::JobProgress(p) => self.app.emit(name, p),
            UiEvent::DownloadProgress(p) => self.app.emit(name, p),
            UiEvent::SpeakersUpdated(p) => self.app.emit(name, p),
            UiEvent::SummaryReady(p) => self.app.emit(name, p),
            UiEvent::ActionItemsUpdated(p) => self.app.emit(name, p),
            UiEvent::MeetingUpdated(p) => self.app.emit(name, p),
            UiEvent::Notice(p) => self.app.emit(name, p),
            UiEvent::SettingsChanged(p) => self.app.emit(name, p),
            UiEvent::RecoveryAvailable(p) => self.app.emit(name, p),
            UiEvent::TrayState(state) => {
                // `set_tray_state` emits the event itself.
                crate::set_tray_state(&self.app, state);
                Ok(())
            }
        };
        if let Err(error) = sent {
            tracing::debug!(%error, event = name, "could not reach the window");
        }
    }
}

/// The four seams, passed around as one value.
#[derive(Clone)]
pub struct Ports {
    pub capture: Arc<dyn CapturePort>,
    pub asr: Arc<dyn AsrPort>,
    pub executor: Arc<dyn super::jobs::JobExecutor>,
    pub events: Arc<EventBus>,
}

impl Ports {
    /// The real thing: devices, whisper, the default job handlers.
    pub fn real(db: Db, idle_release_minutes: u32) -> Self {
        Self {
            capture: Arc::new(DeviceCapture::new(db.clone())),
            asr: Arc::new(EngineAsr::new(db, idle_release_minutes)),
            executor: Arc::new(super::jobs::DefaultJobExecutor),
            events: EventBus::new(),
        }
    }
}

impl std::fmt::Debug for Ports {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Ports { .. }")
    }
}
