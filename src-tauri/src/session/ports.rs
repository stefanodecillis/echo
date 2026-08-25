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

use std::panic::AssertUnwindSafe;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use futures::future::BoxFuture;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use crate::asr::engine::{EngineWorker, PartialFn};
use crate::asr::{AsrError, TranscribeJob, Transcription};
use crate::audio::{AudioError, CaptureConfig, CaptureSession, CaptureStarted};
use crate::db::Db;
use crate::events;
use crate::logging::Throttle;
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

    /// One live job, with a plan: whether this is a finished utterance or a
    /// caption of speech that is still going, and any words carried across a
    /// forced cut. See [`crate::asr::engine::DecodePlan`].
    ///
    /// Default: the same as [`AsrPort::transcribe`], so a test double only has
    /// to know about one of them.
    fn transcribe_live<'a>(
        &'a self,
        job: TranscribeJob,
        _plan: crate::asr::engine::DecodePlan,
        on_partial: Option<PartialFn>,
    ) -> BoxFuture<'a, Result<Transcription, AsrError>> {
        self.transcribe(job, on_partial)
    }

    /// Say whether a meeting still needs the engine: audio is being captured, or
    /// its post-meeting work is running or queued.
    ///
    /// While this is true the weights stay in memory however quiet the room gets
    /// (mantra 1's amendment of 2026-08-20). When it goes false the engine's own
    /// grace period starts, and after [`crate::asr::engine::IDLE_GRACE`] with
    /// nothing new the memory goes back. Default: nothing to hold.
    fn hold_resident(&self, _resident: bool) {}

    /// Throw away captions of speech that is still going. Default: nothing to
    /// throw away.
    fn abandon_speculative(&self, _meeting_id: &str) {}

    /// Hand a meeting over to the disk pass, hard: drop every live job, queued
    /// or in flight, so nothing half-finished can race the catch-up pass
    /// (review of 2026-08-20, finding 7). Default: nothing to abandon.
    fn abandon_live(&self, _meeting_id: &str) {}

    /// Transcribe whatever the committed chunks on disk have no text against
    /// yet, per channel. This is what makes "audio on disk is the source of
    /// truth" real, and what makes the catch-up job resumable.
    ///
    /// `not_before_ms` is a floor for callers that know nothing earlier matters;
    /// `None` means "look at the whole meeting". It is deliberately not a
    /// starting point: an utterance the live queue dropped sits *inside* the
    /// transcript, and the two channels reach different lengths, so one offset
    /// for both would quietly lose words that are sitting on disk.
    ///
    /// The count that comes back is a count from a pass that **finished**. A
    /// pass stopped part-way — `control.cancel` went true, usually because a
    /// recording started — answers `Err(AsrError::Cancelled)` and no count at
    /// all, so a caller cannot read "it wrote some lines" as "it is done" (see
    /// [`crate::asr::catchup::cut_short`]).
    fn catch_up<'a>(
        &'a self,
        db: &'a Db,
        meeting_id: &'a str,
        not_before_ms: Option<i64>,
        control: CatchUpControl,
    ) -> BoxFuture<'a, Result<u32, AsrError>>;

    /// [`AsrPort::catch_up`] for a meeting that is **still being recorded**:
    /// everything already on disk up to `to_ms` that has no text against it yet.
    ///
    /// This is what happens when the engine comes up mid-meeting — somebody
    /// pressed Start before the weights were ready, which on a first-ever launch
    /// is minutes of a real conversation. Capture never waits for the engine
    /// (mantra 3), so those minutes are on disk; this reads them back and puts
    /// them in the transcript while the meeting carries on.
    ///
    /// Same code path and the same honest coverage spans as the post-meeting
    /// pass, so a stretch live already wrote down is not read twice. It never
    /// pauses for the recording it belongs to, and its work sits in the engine's
    /// catch-up lane, behind live finals.
    ///
    /// Default: [`AsrPort::catch_up`] over the whole meeting, which is the same
    /// answer for a test double.
    fn catch_up_live<'a>(
        &'a self,
        db: &'a Db,
        meeting_id: &'a str,
        _to_ms: i64,
        control: CatchUpControl,
    ) -> BoxFuture<'a, Result<u32, AsrError>> {
        self.catch_up(db, meeting_id, None, control)
    }

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
    pub fn new(db: Db) -> Self {
        Self {
            worker: EngineWorker::new(),
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

            // The load is over, so whatever this machine had to build for these
            // weights has been built and cached by the OS. This is the one place
            // that can be recorded: every load anything asks for by name — a
            // meeting starting, a download finishing, the setup job — comes
            // through this method.
            //
            // Not quite every load, though: a decode that arrives with nothing
            // loaded is served by `engine::ensure_loaded` on the engine thread,
            // which never passes here, so the marker can stay unwritten after a
            // load that really happened. That costs one redundant setup job,
            // which finds the weights already in memory, returns in
            // milliseconds and writes the marker — the repair, rather than a
            // case to guard against.
            if let Some(file_name) = self
                .worker
                .configured()
                .and_then(|config| config.model_path.file_name().map(|n| n.to_os_string()))
            {
                let file_name = file_name.to_string_lossy().into_owned();
                if let Err(error) = crate::asr::models::mark_warmed(&self.db, &file_name).await {
                    // Nothing is broken by this: the worst it costs is doing the
                    // setup job again, which is fast now.
                    tracing::debug!(
                        %error,
                        model = %file_name,
                        "could not remember that these weights are ready on this machine"
                    );
                }
            }
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

    fn transcribe_live<'a>(
        &'a self,
        job: TranscribeJob,
        plan: crate::asr::engine::DecodePlan,
        on_partial: Option<PartialFn>,
    ) -> BoxFuture<'a, Result<Transcription, AsrError>> {
        Box::pin(async move { self.worker.submit_with(job, plan, on_partial).await })
    }

    fn hold_resident(&self, resident: bool) {
        self.worker.set_resident(resident);
    }

    fn abandon_speculative(&self, meeting_id: &str) {
        self.worker.abandon_speculative(meeting_id);
    }

    fn abandon_live(&self, meeting_id: &str) {
        self.worker.abandon_live(meeting_id);
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

    fn catch_up_live<'a>(
        &'a self,
        db: &'a Db,
        meeting_id: &'a str,
        to_ms: i64,
        control: CatchUpControl,
    ) -> BoxFuture<'a, Result<u32, AsrError>> {
        Box::pin(async move {
            let options = crate::asr::catchup::CatchUpOptions {
                // No floor: a hole is a hole wherever it is, and the whole point
                // of this pass is the stretch before the engine was ready.
                from_ms: None,
                // Everything captured so far, and nothing the live pass is
                // decoding right now.
                to_ms: Some(to_ms.max(0)),
                // Narrower windows than the post-meeting pass uses: a live
                // final waits behind whatever decode is in flight, and this
                // one is in flight during the meeting (see
                // [`crate::asr::catchup::LIVE_PACK_MS`]).
                pack_ms: crate::asr::catchup::LIVE_PACK_MS,
                // `pause_while` stays unset on purpose. The post-meeting pass
                // yields to a live recording; this one *is* the live recording,
                // and yielding to itself would mean never running. It stays out
                // of the way through the queue instead: its jobs are catch-up
                // work, which the engine serves after live finals.
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
    /// The people Echo remembers changed without anybody clicking anything —
    /// the offline pass linked a voice, so somebody was last heard just now, or
    /// it brought a profile up to date with the network in use. Carries the whole
    /// list, like the commands do.
    PeopleUpdated(events::PeopleUpdatedPayload),
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
            UiEvent::PeopleUpdated(_) => events::PEOPLE_UPDATED,
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

/// How long a repeating event-delivery complaint stays quiet after its first
/// line. Long enough that a sink failing on every level meter cannot fill the
/// log; short enough that a transport dying half an hour in still says so
/// within the same half minute.
const EVENT_TROUBLE_INTERVAL: Duration = Duration::from_secs(30);

/// Swappable sink. The session is built before the window exists, so it starts
/// with [`SilentEvents`] and the real sink is attached in `lib.rs`.
///
/// The lock guards the *pointer*, and nothing else. Everything in here is
/// written so that no failure on the delivery side — a slow sink, a panicking
/// sink, a lock some earlier panic poisoned — can stop the next event from
/// going out. On 2026-08-24 a single panic during one emit left this bus
/// dropping every event for the rest of a 35-minute meeting, in silence, while
/// the backend happily carried on writing the transcript to disk.
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

    /// Point the bus at a real sink. Called once, at launch — and again by any
    /// test that wants to watch what goes out.
    ///
    /// Poison is recovered rather than respected. A poisoned lock only means
    /// *some thread panicked while holding it*; the value behind it is a single
    /// `Arc` that no unwind can leave half-written, so there is nothing to
    /// protect the next caller from. Refusing here (what `if let Ok` used to
    /// do) would mean one unrelated panic could permanently stop the window
    /// from ever being attached.
    pub fn set(&self, sink: Arc<dyn EventSink>) {
        let mut current = self.sink.write().unwrap_or_else(|e| e.into_inner());
        *current = sink;
    }
}

impl EventSink for EventBus {
    /// Hand the event to whatever sink is attached, and survive it doing
    /// anything at all.
    ///
    /// Two deliberate shapes here:
    ///
    /// * **The sink runs outside the guard.** The Arc is cloned inside the
    ///   smallest possible scope and the read guard is dropped before delivery.
    ///   Delivery is not cheap — the real sink crosses into Tauri, and the tray
    ///   arm waits on the main thread to repaint an icon — and an `RwLock`
    ///   queues readers behind a waiting writer. Holding the guard across that
    ///   would park capture, the speech thread and the job runner behind one
    ///   slow paint, which is exactly the kind of stall that ends up looking
    ///   like a frozen meeting.
    /// * **The sink runs inside `catch_unwind`.** A notification must never
    ///   take capture down with it (mantra 3). Before this, a panic anywhere
    ///   downstream — Tauri's own listener mutex, the tray animation lock —
    ///   unwound through the emitting thread *and* poisoned this lock on the
    ///   way out, so every later event was dropped forever.
    ///
    /// `AssertUnwindSafe` is honest here: the only things crossing the boundary
    /// are the event (moved in and gone either way) and an `Arc` to a sink that
    /// is `Send + Sync` and owns whatever synchronisation its own state needs.
    ///
    /// Note this net exists only because the crate unwinds. Under
    /// `panic = "abort"` `catch_unwind` catches nothing — the process is
    /// already on its way out. No profile in this repo sets it; if one ever
    /// does, this protection goes with it.
    fn emit(&self, event: UiEvent) {
        let name = event.name();
        let sink = {
            let guard = self.sink.read().unwrap_or_else(|e| e.into_inner());
            Arc::clone(&guard)
        };

        if std::panic::catch_unwind(AssertUnwindSafe(|| sink.emit(event))).is_err() {
            static THROTTLE: OnceLock<Throttle> = OnceLock::new();
            let throttle = THROTTLE.get_or_init(|| Throttle::new(EVENT_TROUBLE_INTERVAL));
            if let Some(missed) = throttle.admit(name) {
                // The panic hook has already written the message and the
                // location; this line says which event was lost to it, which
                // the hook has no way to know.
                tracing::error!(
                    event = name,
                    missed,
                    "the event sink panicked; this event was dropped and the bus carried on"
                );
            }
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
            UiEvent::PeopleUpdated(p) => self.app.emit(name, p),
            UiEvent::SummaryReady(p) => self.app.emit(name, p),
            UiEvent::ActionItemsUpdated(p) => self.app.emit(name, p),
            UiEvent::MeetingUpdated(p) => self.app.emit(name, p),
            UiEvent::Notice(p) => self.app.emit(name, p),
            UiEvent::SettingsChanged(p) => self.app.emit(name, p),
            UiEvent::RecoveryAvailable(p) => self.app.emit(name, p),
            UiEvent::TrayState(state) => {
                // Nothing special about this arm any more. It used to call
                // `set_tray_state`, which emitted its own event and swallowed
                // the result — so a dead window went unnoticed here while the
                // other fifteen arms reported it, and, worse, the icon was
                // painted synchronously from whichever thread was emitting.
                // Now the icon is queued for the main thread and the event goes
                // out through the same path as everything else.
                crate::schedule_tray_state(&self.app, state);
                self.app.emit(name, events::TrayStatePayload { state })
            }
        };
        if let Err(error) = sent {
            // This used to be `debug!`, which in a release build is below the
            // log level and therefore invisible — the transport died on
            // 2026-08-24 and left not one line behind. It is a warning now,
            // throttled per event name so a permanently dead window writes one
            // line every half minute with a count of what it swallowed, rather
            // than one line per level meter.
            static THROTTLE: OnceLock<Throttle> = OnceLock::new();
            let throttle = THROTTLE.get_or_init(|| Throttle::new(EVENT_TROUBLE_INTERVAL));
            if let Some(missed) = throttle.admit(name) {
                tracing::warn!(
                    %error,
                    event = name,
                    missed,
                    "the window is not receiving events"
                );
            }
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
    pub fn real(db: Db) -> Self {
        Self {
            capture: Arc::new(DeviceCapture::new(db.clone())),
            asr: Arc::new(EngineAsr::new(db)),
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicIsize, Ordering};

    /// Keeps what it was given. Deliberately local rather than the session
    /// harness's `CollectingEvents`: these tests are about the bus itself and
    /// should not need a database, a temp directory or a mock engine to run.
    #[derive(Default)]
    struct Collector {
        seen: std::sync::Mutex<Vec<UiEvent>>,
    }

    impl Collector {
        /// The notice text of everything that arrived, which is how these
        /// tests tell one event from the next.
        fn messages(&self) -> Vec<String> {
            self.seen
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .map(|event| match event {
                    UiEvent::Notice(payload) => payload.message.clone(),
                    other => other.name().to_string(),
                })
                .collect()
        }
    }

    impl EventSink for Collector {
        fn emit(&self, event: UiEvent) {
            self.seen
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(event);
        }
    }

    /// Stands in for the tray arm on the day it took the meeting down: blows up
    /// on its first `explosions` deliveries, then behaves like an ordinary sink.
    struct Exploding {
        explosions_left: AtomicIsize,
        delivered: Arc<Collector>,
    }

    impl Exploding {
        fn new(explosions: isize, delivered: Arc<Collector>) -> Self {
            Self {
                explosions_left: AtomicIsize::new(explosions),
                delivered,
            }
        }
    }

    impl EventSink for Exploding {
        fn emit(&self, event: UiEvent) {
            if self.explosions_left.fetch_sub(1, Ordering::SeqCst) > 0 {
                panic!("the sink is having the kind of day the tray had on 2026-08-24");
            }
            self.delivered.emit(event);
        }
    }

    /// A distinguishable event: the message is the label the test reads back.
    fn note(message: &str) -> UiEvent {
        UiEvent::Notice(events::NoticePayload {
            message: message.to_string(),
            ..Default::default()
        })
    }

    #[test]
    fn a_sink_that_panics_does_not_take_the_emitting_thread_with_it() {
        let bus = EventBus::default();
        bus.set(Arc::new(Exploding::new(
            isize::MAX,
            Arc::new(Collector::default()),
        )));

        let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| bus.emit(note("boom"))));

        assert!(
            outcome.is_ok(),
            "emitting must return normally however badly the sink behaves — \
             the caller is usually the capture pipeline"
        );
    }

    #[test]
    fn a_panicking_sink_does_not_stop_the_events_after_it() {
        // The regression test for the incident: one panic used to poison the
        // lock and silence the bus for the rest of the meeting.
        let delivered = Arc::new(Collector::default());
        let bus = EventBus::default();
        bus.set(Arc::new(Exploding::new(1, delivered.clone())));

        for i in 1..=5 {
            bus.emit(note(&format!("line {i}")));
        }

        assert_eq!(
            delivered.messages(),
            vec!["line 2", "line 3", "line 4", "line 5"],
            "only the event the sink blew up on is lost"
        );
    }

    #[test]
    fn the_bus_survives_a_panic_on_another_thread() {
        const THREADS: usize = 8;
        let delivered = Arc::new(Collector::default());
        let bus = EventBus::new();
        bus.set(Arc::new(Exploding::new(1, delivered.clone())));

        let handles: Vec<_> = (0..THREADS)
            .map(|i| {
                let bus = bus.clone();
                std::thread::spawn(move || bus.emit(note(&format!("thread {i}"))))
            })
            .collect();

        for handle in handles {
            assert!(
                handle.join().is_ok(),
                "whichever thread met the panic must still come home"
            );
        }
        assert_eq!(
            delivered.messages().len(),
            THREADS - 1,
            "exactly one emit was lost to the panic; the other threads' events arrived"
        );
    }

    #[test]
    fn a_sink_can_still_be_swapped_after_one_panicked() {
        let bus = EventBus::default();
        bus.set(Arc::new(Exploding::new(
            isize::MAX,
            Arc::new(Collector::default()),
        )));
        bus.emit(note("lost"));

        // What `lib.rs` does when the window comes up — and what a person does
        // by reopening the window after something went wrong.
        let fresh = Arc::new(Collector::default());
        bus.set(fresh.clone());
        bus.emit(note("delivered"));

        assert_eq!(fresh.messages(), vec!["delivered"]);
    }

    #[test]
    fn a_poisoned_bus_still_carries_events_and_still_takes_a_new_sink() {
        // Nothing in the bus can poison this lock any more — that is the point
        // of the `catch_unwind` above. Poison it by hand anyway: Tauri and the
        // tray hold locks of their own, and the recovery here is the last line
        // of defence if some future caller panics inside a guard.
        let bus = EventBus::new();
        let poisoner = bus.clone();
        let _ = std::thread::spawn(move || {
            let _guard = poisoner.sink.write().unwrap();
            panic!("poisoning the bus on purpose");
        })
        .join();
        assert!(bus.sink.is_poisoned(), "the setup did not poison anything");

        let collector = Arc::new(Collector::default());
        bus.set(collector.clone());
        bus.emit(note("after the poison"));

        assert_eq!(collector.messages(), vec!["after the poison"]);
    }

    #[test]
    fn every_event_carries_its_own_name() {
        let table: Vec<(UiEvent, &'static str)> = vec![
            (
                UiEvent::CaptureState(Default::default()),
                events::CAPTURE_STATE,
            ),
            (
                UiEvent::TranscriptPartial(Default::default()),
                events::TRANSCRIPT_PARTIAL,
            ),
            (
                UiEvent::TranscriptFinal(Default::default()),
                events::TRANSCRIPT_FINAL,
            ),
            (
                UiEvent::TranscriptRevised(Default::default()),
                events::TRANSCRIPT_REVISED,
            ),
            (
                UiEvent::AudioLevels(Default::default()),
                events::AUDIO_LEVELS,
            ),
            (
                UiEvent::JobProgress(Default::default()),
                events::JOB_PROGRESS,
            ),
            (
                UiEvent::DownloadProgress(Default::default()),
                events::DOWNLOAD_PROGRESS,
            ),
            (
                UiEvent::SpeakersUpdated(Default::default()),
                events::SPEAKERS_UPDATED,
            ),
            (
                UiEvent::PeopleUpdated(Default::default()),
                events::PEOPLE_UPDATED,
            ),
            (
                UiEvent::SummaryReady(Default::default()),
                events::SUMMARY_READY,
            ),
            (
                UiEvent::ActionItemsUpdated(Default::default()),
                events::ACTION_ITEMS_UPDATED,
            ),
            (
                UiEvent::MeetingUpdated(Default::default()),
                events::MEETING_UPDATED,
            ),
            (UiEvent::Notice(Default::default()), events::NOTICE),
            (
                UiEvent::SettingsChanged(Default::default()),
                events::SETTINGS_CHANGED,
            ),
            (UiEvent::TrayState(Default::default()), events::TRAY_STATE),
            (
                UiEvent::RecoveryAvailable(Default::default()),
                events::RECOVERY_AVAILABLE,
            ),
        ];

        // The compiler cannot make a new variant show up here, so the count
        // does: add a `UiEvent` and this line is what asks you for its name.
        assert_eq!(
            table.len(),
            16,
            "a UiEvent variant is missing from this table"
        );

        let mut names = std::collections::HashSet::new();
        for (event, expected) in &table {
            assert_eq!(&event.name(), expected);
            assert!(
                events::ALL.contains(expected),
                "{expected} is not declared in events::ALL"
            );
            assert!(names.insert(*expected), "{expected} is claimed twice");
        }

        // Naming is done per variant, not per payload, so a sink can key off
        // it before it looks at anything else — which is what the throttles in
        // this file do when delivery goes wrong.
        assert_eq!(note("anything").name(), events::NOTICE);
    }
}
