//! Noticing that a meeting is happening.
//!
//! Two signals, polled every [`POLL_INTERVAL_SECS`] (DESIGN §3):
//! 1. a known meeting app is running (zoom.us, Teams, Webex, Discord, Slack, Meet)
//! 2. something else is using an input device
//!    (macOS `kAudioDevicePropertyDeviceIsRunningSomewhere`, Linux PipeWire
//!    active input streams)
//!
//! Either signal, held for [`DEBOUNCE_POLLS`] consecutive polls, flips the
//! watcher to [`DetectionState::Detected`]: a tray badge, plus exactly one
//! nudge. The nudge is the small floating panel near the menu bar
//! ([`crate::panel`]) — "Meeting detected", how long ago, and a Start button
//! right there. When a window cannot be put on screen at all, it falls back to
//! an OS notification instead (a plain "click opens Echo" one; Tauri's plugin
//! has no action buttons on desktop, review finding 7). Never both: one meeting
//! noticed is one interruption. Closing the panel with ✕ mutes only that
//! episode — see [`DebounceTracker::dismiss`] — and is emphatically not a
//! snooze. While a meeting
//! *is* being recorded, the same signals are watched the other way around:
//! once they have all been clear for [`AUTO_STOP_SUGGEST_SECS`], the watcher
//! sets `suggest_stop` on [`DetectionStatus`] so the UI can offer a gentle
//! "still going?" prompt — it never stops the recording itself.
//!
//! Budget (mantra 1): an idle Echo is this poll and nothing else. No audio
//! stream, no models, no database churn beyond the occasional snooze
//! timestamp. The watcher fully parks — skips the probes entirely — when
//! detection is turned off or snoozed; the one exception is that while a
//! recording is in progress it keeps checking for the auto-stop signal
//! regardless of the enabled/snoozed flags, because "you forgot to stop" is
//! a safety net independent of whether meeting-start notifications are
//! wanted right now.
//!
//! This watcher is its own small state machine, deliberately not part of
//! capture state (review finding 19).

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use chrono::{DateTime, Utc};
use tauri::{AppHandle, Emitter, Manager};
use tokio::task::JoinHandle;

use crate::db::{repo, Db};
use crate::events;
use crate::settings;
use crate::types::{
    CaptureState, DetectionSignal, DetectionSource, DetectionState, DetectionStatus, TrayState,
};

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;

/// How often to look. Not configurable; it is already cheap, and while
/// recording it is the only thing standing between "forgot to stop" and a
/// day of silence on the disk.
pub const POLL_INTERVAL_SECS: u64 = 5;

/// A signal has to hold for this many consecutive polls before we say
/// anything, so alt-tabbing past Zoom does not fire a notification.
pub const DEBOUNCE_POLLS: u32 = 2;

/// The same span, in seconds, for anything that wants to talk about it in
/// wall-clock terms (docs, the idle-poll-stays-cheap test below).
pub const DEBOUNCE_SECS: u64 = POLL_INTERVAL_SECS * DEBOUNCE_POLLS as u64;

/// Once every signal has been clear this long during a recording, suggest
/// stopping.
pub const AUTO_STOP_SUGGEST_SECS: u64 = 120;

/// What "Pause detection" in the tray gives you.
pub const SNOOZE_MINUTES: u64 = 60;

/// Process names that mean a meeting. Matched case-insensitively against the
/// executable name.
pub const MEETING_APPS: &[(&str, &str)] = &[
    ("zoom.us", "Zoom"),
    ("zoom", "Zoom"),
    ("Microsoft Teams", "Teams"),
    ("Teams", "Teams"),
    ("Webex", "Webex"),
    ("Discord", "Discord"),
    ("Slack", "Slack"),
    ("Google Meet", "Google Meet"),
];

#[derive(Debug, thiserror::Error)]
pub enum DetectError {
    #[error("could not look at what is running: {0}")]
    Probe(String),
}

fn auto_stop_threshold_polls() -> u32 {
    (AUTO_STOP_SUGGEST_SECS / POLL_INTERVAL_SECS) as u32
}

// ---------------------------------------------------------------------------
// Probes, real and mocked
// ---------------------------------------------------------------------------

/// Everything the watcher needs from the outside world, behind a trait so
/// tests can hand it fake answers instead of asking `sysinfo`/CoreAudio for
/// real ones.
pub trait SignalSource: Send + Sync {
    /// Friendly names of meeting apps currently running (e.g. "Zoom"),
    /// already deduplicated.
    fn running_meeting_apps(&self) -> Result<Vec<String>, DetectError>;
    /// Is some other app using an input device right now?
    fn input_device_in_use(&self) -> Result<bool, DetectError>;
}

struct SystemSignalSource;

impl SignalSource for SystemSignalSource {
    fn running_meeting_apps(&self) -> Result<Vec<String>, DetectError> {
        running_meeting_apps()
    }

    fn input_device_in_use(&self) -> Result<bool, DetectError> {
        input_device_in_use()
    }
}

/// Which meeting apps are running right now, as friendly names
/// (deduplicated). `sysinfo`, no per-process detail beyond the name.
pub fn running_meeting_apps() -> Result<Vec<String>, DetectError> {
    let mut system = sysinfo::System::new();
    system.refresh_processes(sysinfo::ProcessesToUpdate::All, true);

    let mut found = BTreeSet::new();
    for process in system.processes().values() {
        let name = process.name().to_string_lossy();
        if let Some(friendly) = friendly_app_name(&name) {
            found.insert(friendly.to_string());
        }
    }
    Ok(found.into_iter().collect())
}

/// Is some other app using an input device? CoreAudio on macOS, PipeWire on
/// Linux; `false` on anything else (there is nothing to ask).
#[cfg(target_os = "macos")]
pub fn input_device_in_use() -> Result<bool, DetectError> {
    macos::input_device_in_use()
}

#[cfg(target_os = "linux")]
pub fn input_device_in_use() -> Result<bool, DetectError> {
    linux::input_device_in_use()
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
pub fn input_device_in_use() -> Result<bool, DetectError> {
    Ok(false)
}

/// Map a process name onto a friendly app name, or None when we do not know it.
pub fn friendly_app_name(process_name: &str) -> Option<&'static str> {
    let lower = process_name.to_ascii_lowercase();
    MEETING_APPS
        .iter()
        .find(|(needle, _)| lower.contains(&needle.to_ascii_lowercase()))
        .map(|(_, friendly)| *friendly)
}

// ---------------------------------------------------------------------------
// Debounce / auto-stop state machine (pure, unit-tested directly)
// ---------------------------------------------------------------------------

/// What a poll's result should cause the watcher to *do*, as opposed to what
/// it currently *is* — `Detected`/`SuggestStop` are edges (fire once), while
/// [`DebounceTracker::is_detected`]/[`DebounceTracker::is_suggesting_stop`]
/// are the latched level (what `status()` reports every time).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StepEvent {
    None,
    Detected,
    SuggestStop,
}

#[derive(Debug, Default, Clone, Copy)]
struct DebounceTracker {
    present_streak: u32,
    absent_streak: u32,
    detected_latched: bool,
    stop_suggested: bool,
    /// Counts detection *episodes*: one per false→true flip of the latch. A
    /// meeting that is noticed, waved away, ends, and starts again is a new
    /// episode and gets a fresh nudge.
    episode: u64,
    /// The episode the person closed the panel on. Only that one is muted —
    /// this is not a snooze, detection keeps working exactly as before.
    dismissed_episode: Option<u64>,
}

impl DebounceTracker {
    /// Advance one poll. `signals_present` is whether *any* signal fired
    /// this round; `capture_state` decides which half of the machine is
    /// live (DESIGN §3: detection vs. auto-stop are different concerns).
    fn step(&mut self, capture_state: CaptureState, signals_present: bool) -> StepEvent {
        if capture_state == CaptureState::Recording {
            // Already recording: "Detected" has nothing left to offer, only
            // auto-stop matters.
            if signals_present {
                // The meeting is still going, so this is still the same
                // episode: the latch stays exactly as it was. Otherwise
                // stopping a recording while the call is still open would earn
                // a fresh "meeting detected" nudge ten seconds later, for the
                // meeting the person just decided to stop recording.
                self.absent_streak = 0;
                self.stop_suggested = false;
                StepEvent::None
            } else {
                // Nothing left to hear: the meeting is over as far as we can
                // tell, and the next one is a new episode.
                self.present_streak = 0;
                self.detected_latched = false;
                self.absent_streak = self.absent_streak.saturating_add(1);
                if self.absent_streak >= auto_stop_threshold_polls() && !self.stop_suggested {
                    self.stop_suggested = true;
                    StepEvent::SuggestStop
                } else {
                    StepEvent::None
                }
            }
        } else {
            // Not recording: auto-stop does not apply.
            self.absent_streak = 0;
            self.stop_suggested = false;

            if signals_present {
                self.present_streak = self.present_streak.saturating_add(1);
                if self.present_streak >= DEBOUNCE_POLLS && !self.detected_latched {
                    self.detected_latched = true;
                    self.episode = self.episode.wrapping_add(1);
                    StepEvent::Detected
                } else {
                    StepEvent::None
                }
            } else {
                self.present_streak = 0;
                self.detected_latched = false;
                StepEvent::None
            }
        }
    }

    fn is_detected(&self) -> bool {
        self.detected_latched
    }

    fn is_suggesting_stop(&self) -> bool {
        self.stop_suggested
    }

    /// Which detection episode we are in. Starts at zero, before anything has
    /// ever been detected.
    fn episode(&self) -> u64 {
        self.episode
    }

    /// The person closed the panel: say nothing more about *this* meeting.
    fn dismiss(&mut self) {
        self.dismissed_episode = Some(self.episode);
    }

    /// Has this episode already been waved away?
    fn is_dismissed(&self) -> bool {
        self.dismissed_episode == Some(self.episode)
    }
}

// ---------------------------------------------------------------------------
// The watcher
// ---------------------------------------------------------------------------

struct Inner {
    enabled: bool,
    snoozed_until: Option<DateTime<Utc>>,
    status: DetectionStatus,
}

/// The watcher. Owns the poll timer and the debounce state.
///
/// Construction (`new`) takes only the enabled flag, matching how `lib.rs`
/// builds it before the rest of app state exists. Everything that needs the
/// running app — the database handle for persisting snooze, the
/// [`AppHandle`] for notifications/tray/events, the 5s timer itself — is
/// wired up in [`Watcher::start`], which must be called once app state is
/// managed (see the integrator note in this module's owning task: `lib.rs`
/// does not currently call it, and needs a one-line addition to).
pub struct Watcher {
    inner: Mutex<Inner>,
    debounce: Mutex<DebounceTracker>,
    db: OnceLock<Db>,
    app: Mutex<Option<AppHandle>>,
    task: Mutex<Option<JoinHandle<()>>>,
    requested_notification_permission: AtomicBool,
    /// True while a notification is on screen waiting to be clicked, so a run of
    /// detections cannot pile up threads.
    notification_in_flight: Arc<AtomicBool>,
    source: Box<dyn SignalSource>,
}

impl Watcher {
    /// Create the watcher in the state the settings ask for. Starts no timer.
    pub fn new(enabled: bool) -> Self {
        Self::with_source(enabled, Box::new(SystemSignalSource))
    }

    /// Same as [`Watcher::new`], but with a mocked [`SignalSource`] — for
    /// tests, and for anything else that wants to drive the watcher without
    /// touching `sysinfo`/CoreAudio/PipeWire.
    pub fn with_source(enabled: bool, source: Box<dyn SignalSource>) -> Self {
        Self {
            inner: Mutex::new(Inner {
                enabled,
                snoozed_until: None,
                status: DetectionStatus {
                    enabled,
                    state: initial_state(enabled),
                    ..Default::default()
                },
            }),
            debounce: Mutex::new(DebounceTracker::default()),
            db: OnceLock::new(),
            app: Mutex::new(None),
            task: Mutex::new(None),
            requested_notification_permission: AtomicBool::new(false),
            notification_in_flight: Arc::new(AtomicBool::new(false)),
            source,
        }
    }

    /// Begin polling. Idempotent. Restores a persisted snooze, then spawns
    /// the single 5s interval this whole module is allowed to own (mantra 1).
    pub async fn start(&self, app: AppHandle) -> Result<(), DetectError> {
        if self.task.lock().unwrap().is_some() {
            return Ok(());
        }

        if let Some(state) = app.try_state::<crate::commands::AppState>() {
            let _ = self.db.set(state.db.clone());
        }
        self.restore_snoozed_until().await;

        *self.app.lock().unwrap() = Some(app.clone());
        self.recompute_and_emit(
            self.current_capture_state().await,
            self.last_signals(),
            StepEvent::None,
        )
        .await;

        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(POLL_INTERVAL_SECS));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                interval.tick().await;
                if let Some(state) = app.try_state::<crate::commands::AppState>() {
                    state.detect.tick().await;
                }
            }
        });
        *self.task.lock().unwrap() = Some(handle);
        Ok(())
    }

    /// Stop polling and release anything held. Idempotent.
    pub async fn stop(&self) {
        if let Some(handle) = self.task.lock().unwrap().take() {
            handle.abort();
        }
    }

    pub async fn set_enabled(&self, enabled: bool) -> Result<(), DetectError> {
        {
            let mut inner = self.inner.lock().unwrap();
            inner.enabled = enabled;
        }
        *self.debounce.lock().unwrap() = DebounceTracker::default();
        let capture_state = self.current_capture_state().await;
        self.recompute_and_emit(capture_state, Vec::new(), StepEvent::None)
            .await;
        Ok(())
    }

    /// Mute for `minutes`, then go back to watching by itself.
    pub async fn snooze(&self, minutes: u64) -> Result<(), DetectError> {
        let until = Utc::now() + chrono::Duration::minutes(minutes as i64);
        {
            let mut inner = self.inner.lock().unwrap();
            inner.snoozed_until = Some(until);
        }
        *self.debounce.lock().unwrap() = DebounceTracker::default();

        if let Some(db) = self.db.get() {
            if let Err(err) = repo::set_setting(
                db,
                settings::keys::DETECTION_SNOOZED_UNTIL,
                &until.to_rfc3339(),
            )
            .await
            {
                tracing::warn!(error = %err, "could not remember the snooze past a restart");
            }
        }

        let capture_state = self.current_capture_state().await;
        self.recompute_and_emit(capture_state, Vec::new(), StepEvent::None)
            .await;
        Ok(())
    }

    pub fn status(&self) -> DetectionStatus {
        self.inner.lock().unwrap().status.clone()
    }

    /// Which detection episode is in progress. One per meeting noticed; zero
    /// before anything has been.
    pub fn episode(&self) -> u64 {
        self.debounce.lock().unwrap().episode()
    }

    /// The ✕ on the floating panel: say nothing more about *this* meeting.
    ///
    /// Deliberately not a snooze — detection keeps watching, the tray badge
    /// keeps showing, and the next meeting gets nudged about as usual. All this
    /// mutes is a second panel for the meeting the person just waved away.
    pub fn dismiss_episode(&self) {
        self.debounce.lock().unwrap().dismiss();
    }

    /// Has the meeting we are currently seeing already been waved away?
    pub fn episode_dismissed(&self) -> bool {
        self.debounce.lock().unwrap().is_dismissed()
    }

    /// Poll once, out of band. Used by tests and by the "check now" button in
    /// diagnostics. Does not touch the debounce state — a diagnostic check
    /// should not itself flip the watcher to Detected.
    pub async fn poll_once(&self) -> Result<Vec<DetectionSignal>, DetectError> {
        self.collect_signals()
    }

    // -- internals -----------------------------------------------------

    fn last_signals(&self) -> Vec<DetectionSignal> {
        self.inner.lock().unwrap().status.signals.clone()
    }

    async fn restore_snoozed_until(&self) {
        let Some(db) = self.db.get() else { return };
        let Ok(Some(raw)) = repo::get_setting(db, settings::keys::DETECTION_SNOOZED_UNTIL).await
        else {
            return;
        };
        if raw.is_empty() {
            return;
        }
        let Ok(parsed) = DateTime::parse_from_rfc3339(&raw) else {
            return;
        };
        let until = parsed.with_timezone(&Utc);
        if until > Utc::now() {
            self.inner.lock().unwrap().snoozed_until = Some(until);
        }
    }

    async fn clear_expired_snooze_if_any(&self) {
        let expired = {
            let inner = self.inner.lock().unwrap();
            matches!(inner.snoozed_until, Some(until) if until <= Utc::now())
        };
        if !expired {
            return;
        }
        {
            let mut inner = self.inner.lock().unwrap();
            inner.snoozed_until = None;
        }
        if let Some(db) = self.db.get() {
            let _ = repo::set_setting(db, settings::keys::DETECTION_SNOOZED_UNTIL, "").await;
        }
    }

    async fn current_capture_state(&self) -> CaptureState {
        let app = self.app.lock().unwrap().clone();
        let Some(app) = app else {
            return CaptureState::Idle;
        };
        match app.try_state::<crate::commands::AppState>() {
            Some(state) => state.session.status().await.state,
            None => CaptureState::Idle,
        }
    }

    fn collect_signals(&self) -> Result<Vec<DetectionSignal>, DetectError> {
        let now = Utc::now().to_rfc3339();
        let mut signals = Vec::new();

        for app_name in self.source.running_meeting_apps()? {
            signals.push(DetectionSignal {
                source: DetectionSource::MeetingApp,
                app: Some(app_name),
                since: now.clone(),
                confidence: 1.0,
            });
        }

        if self.source.input_device_in_use()? {
            signals.push(DetectionSignal {
                source: DetectionSource::InputDeviceInUse,
                app: None,
                since: now,
                // Less certain than a named app: something is listening, we
                // just don't know what.
                confidence: 0.6,
            });
        }

        Ok(signals)
    }

    /// One real tick of the timer. Runs the probes only when there is a
    /// reason to (mantra 1): while recording, only the auto-stop half of the
    /// debounce machine matters, and it runs regardless of the
    /// enabled/snoozed flags; otherwise, only when detection is on and not
    /// snoozed.
    async fn tick(&self) {
        self.clear_expired_snooze_if_any().await;
        let capture_state = self.current_capture_state().await;

        let (enabled, snoozed) = {
            let inner = self.inner.lock().unwrap();
            (inner.enabled, inner.snoozed_until.is_some())
        };
        let is_recording = capture_state == CaptureState::Recording;
        let should_probe = is_recording || (enabled && !snoozed);

        if !should_probe {
            // Fully parked: no reason to keep a stale streak around for
            // whenever polling resumes.
            *self.debounce.lock().unwrap() = DebounceTracker::default();
            return;
        }

        let signals = match self.collect_signals() {
            Ok(signals) => signals,
            Err(err) => {
                tracing::warn!(error = %err, "could not check for a meeting this round");
                return;
            }
        };
        let present = !signals.is_empty();
        let edge = self.debounce.lock().unwrap().step(capture_state, present);

        self.recompute_and_emit(capture_state, signals, edge).await;
    }

    async fn recompute_and_emit(
        &self,
        capture_state: CaptureState,
        signals: Vec<DetectionSignal>,
        edge: StepEvent,
    ) {
        let (detected_latched, suggest_stop_latched) = {
            let d = self.debounce.lock().unwrap();
            (d.is_detected(), d.is_suggesting_stop())
        };
        let is_recording = capture_state == CaptureState::Recording;

        let (new_status, previous_state) = {
            let mut inner = self.inner.lock().unwrap();
            let state = if !inner.enabled {
                DetectionState::Off
            } else if inner.snoozed_until.is_some() {
                DetectionState::Snoozed
            } else if is_recording {
                // The Detected/Idle distinction is about whether to nudge the
                // person to start a recording; moot once one is running.
                DetectionState::Idle
            } else if detected_latched {
                DetectionState::Detected
            } else {
                DetectionState::Idle
            };

            let status = DetectionStatus {
                state,
                enabled: inner.enabled,
                signals,
                snoozed_until: inner.snoozed_until.map(|t| t.to_rfc3339()),
                suggest_stop: is_recording && suggest_stop_latched,
            };
            let previous_state = inner.status.state;
            inner.status = status.clone();
            (status, previous_state)
        };

        let Some(app) = self.app.lock().unwrap().clone() else {
            return;
        };

        let _ = app.emit(events::DETECTION, new_status.clone());

        if !is_recording && previous_state != new_status.state {
            crate::set_tray_state(
                &app,
                match new_status.state {
                    DetectionState::Detected => TrayState::Detected,
                    _ => TrayState::Idle,
                },
            );
        }

        if edge == StepEvent::Detected {
            // Recording is locked until the one-time speech download is done,
            // so a "start recording?" nudge would lead nowhere. Stay quiet
            // until Echo can actually follow through.
            let speech_ready = match app.try_state::<crate::AppState>() {
                Some(state) => crate::asr::models::readiness(&state.db, &state.paths)
                    .await
                    .map(|r| r.ready)
                    .unwrap_or(false),
                None => false,
            };
            if speech_ready && !self.episode_dismissed() {
                let detected_app = new_status
                    .signals
                    .iter()
                    .find_map(|signal| signal.app.clone());
                self.nudge_about_the_meeting(&app, detected_app).await;
            }
        }
    }

    /// One nudge per meeting, and only one.
    ///
    /// The floating panel is the good version: it is where the person is
    /// looking, and Start is right there. The OS notification is the fallback
    /// for when a window cannot be put on screen at all. Doing both would be
    /// two interruptions for one meeting.
    async fn nudge_about_the_meeting(&self, app: &AppHandle, detected_app: Option<String>) {
        let detected_at_ms = Utc::now().timestamp_millis();
        if crate::panel::show_detected(app, detected_at_ms, detected_app.clone()).await {
            return;
        }
        tracing::debug!("no floating panel available, falling back to a notification");
        self.show_meeting_notification(app, detected_app);
    }

    fn show_meeting_notification(&self, app: &AppHandle, detected_app: Option<String>) {
        use tauri_plugin_notification::{NotificationExt, PermissionState};

        let notifier = app.notification();
        let granted = match notifier.permission_state() {
            Ok(PermissionState::Granted) => true,
            Ok(_) => {
                if self
                    .requested_notification_permission
                    .swap(true, Ordering::SeqCst)
                {
                    // Already asked once this run; do not re-prompt every
                    // time a meeting is seen.
                    false
                } else {
                    matches!(notifier.request_permission(), Ok(PermissionState::Granted))
                }
            }
            Err(err) => {
                tracing::warn!(error = %err, "could not check notification permission");
                false
            }
        };

        if !granted {
            return;
        }

        show_clickable_notification(
            app.clone(),
            detected_app,
            self.notification_in_flight.clone(),
        );
    }
}

/// What the notification says. Plain words, and it names the thing the person is
/// about to do rather than what Echo noticed (mantra 2).
const NOTIFICATION_TITLE: &str = "Echo";
const NOTIFICATION_BODY: &str = "Sounds like a meeting started — open Echo to record it.";

/// Show the meeting notification and act on a click.
///
/// DESIGN §3 promises "clicking it opens Echo with a prominent Start button".
/// Tauri's notification plugin fires and forgets on desktop — it drops the
/// handle, so the click never reaches us — so Echo shows this one itself and
/// keeps the handle until the person interacts with it.
///
/// A click **opens Echo on Home with the Start button waiting; it does not start
/// recording.** Starting is the person's decision, expressed by pressing a
/// button (DESIGN §3, mantra 4); a notification click is "yes, show me", not
/// "yes, record me". The tray's own "Start recording" item is the explicit
/// press, and that one does start.
fn show_clickable_notification(
    app: AppHandle,
    detected_app: Option<String>,
    in_flight: Arc<AtomicBool>,
) {
    if in_flight.swap(true, Ordering::SeqCst) {
        // One on screen already; a second would only be noise.
        return;
    }
    let spawned = std::thread::Builder::new()
        .name("echo-meeting-notification".into())
        .spawn(move || {
            let _guard = InFlightGuard(in_flight);
            #[cfg(target_os = "macos")]
            {
                // Which app the notification belongs to. Unbundled (development)
                // runs have no identifier of their own to borrow.
                let owner = if tauri::is_dev() {
                    "com.apple.Terminal".to_string()
                } else {
                    app.config().identifier.clone()
                };
                if let Err(error) = notify_rust::set_application(&owner) {
                    tracing::debug!("could not name the notification's owner: {error}");
                }
            }
            let mut notification = notify_rust::Notification::new();
            notification
                .summary(NOTIFICATION_TITLE)
                .body(NOTIFICATION_BODY);
            match notification.show() {
                Ok(handle) => wait_for_notification_click(handle, app, detected_app),
                Err(error) => {
                    tracing::warn!("could not show the meeting notification: {error}")
                }
            }
        });
    if let Err(error) = spawned {
        tracing::warn!(%error, "could not show the meeting notification");
    }
}

/// Clears the "one at a time" flag however the waiting thread ends.
struct InFlightGuard(Arc<AtomicBool>);

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

/// Block until the person clicks or dismisses, then open Echo if they clicked.
///
/// Blocking is fine: this is its own thread, and it exists precisely to outlive
/// the poll that created it.
#[cfg(target_os = "macos")]
fn wait_for_notification_click(
    handle: notify_rust::NotificationHandle,
    app: AppHandle,
    detected_app: Option<String>,
) {
    let _ = handle.wait_for_response(move |response: &notify_rust::NotificationResponse| {
        if response.is_default_action() {
            open_echo_ready_to_start(&app, detected_app.clone());
        }
    });
}

#[cfg(not(target_os = "macos"))]
fn wait_for_notification_click(
    handle: notify_rust::NotificationHandle,
    app: AppHandle,
    detected_app: Option<String>,
) {
    handle.wait_for_action(move |action| {
        if action == "default" {
            open_echo_ready_to_start(&app, detected_app.clone());
        }
    });
}

/// The window, on Home, with the Start button ready — never a recording started
/// behind the person's back.
fn open_echo_ready_to_start(app: &AppHandle, detected_app: Option<String>) {
    tracing::info!("the meeting notification was clicked");
    crate::navigate(
        app,
        events::NavigatePayload {
            target: events::NavigateTarget::Home,
            meeting_id: None,
            detected_app,
        },
    );
}

fn initial_state(enabled: bool) -> DetectionState {
    if enabled {
        DetectionState::Idle
    } else {
        DetectionState::Off
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockSource {
        apps: Vec<&'static str>,
        input_in_use: bool,
    }

    impl SignalSource for MockSource {
        fn running_meeting_apps(&self) -> Result<Vec<String>, DetectError> {
            Ok(self.apps.iter().map(|s| s.to_string()).collect())
        }

        fn input_device_in_use(&self) -> Result<bool, DetectError> {
            Ok(self.input_in_use)
        }
    }

    fn quiet_source() -> Box<dyn SignalSource> {
        Box::new(MockSource {
            apps: Vec::new(),
            input_in_use: false,
        })
    }

    #[test]
    fn process_names_map_to_friendly_names() {
        assert_eq!(friendly_app_name("zoom.us"), Some("Zoom"));
        assert_eq!(
            friendly_app_name("/Applications/Slack.app/Contents/MacOS/Slack"),
            Some("Slack")
        );
        assert_eq!(friendly_app_name("Microsoft Teams (work)"), Some("Teams"));
        assert_eq!(friendly_app_name("Finder"), None);
    }

    #[test]
    fn the_idle_poll_stays_cheap_enough_to_leave_running() {
        assert_eq!(POLL_INTERVAL_SECS, 5);
        const {
            assert!(
                DEBOUNCE_SECS >= POLL_INTERVAL_SECS * 2,
                "debounce must span several polls"
            )
        };
    }

    // -- debounce ---------------------------------------------------------

    #[test]
    fn detection_needs_two_consecutive_present_polls() {
        let mut d = DebounceTracker::default();
        assert_eq!(d.step(CaptureState::Idle, true), StepEvent::None);
        assert!(!d.is_detected(), "one poll is not enough");
        assert_eq!(d.step(CaptureState::Idle, true), StepEvent::Detected);
        assert!(d.is_detected());
    }

    #[test]
    fn detection_does_not_renotify_while_still_present() {
        let mut d = DebounceTracker::default();
        d.step(CaptureState::Idle, true);
        assert_eq!(d.step(CaptureState::Idle, true), StepEvent::Detected);
        // Signal keeps holding: still "detected", but no repeat edge.
        assert_eq!(d.step(CaptureState::Idle, true), StepEvent::None);
        assert_eq!(d.step(CaptureState::Idle, true), StepEvent::None);
        assert!(d.is_detected());
    }

    #[test]
    fn a_single_clear_poll_resets_the_present_streak() {
        let mut d = DebounceTracker::default();
        d.step(CaptureState::Idle, true);
        d.step(CaptureState::Idle, false); // clears before the second present poll lands
        assert!(!d.is_detected());
        // Needs two fresh consecutive present polls again, not just one.
        assert_eq!(d.step(CaptureState::Idle, true), StepEvent::None);
        assert!(!d.is_detected());
        assert_eq!(d.step(CaptureState::Idle, true), StepEvent::Detected);
    }

    #[test]
    fn detected_state_clears_once_signals_are_gone() {
        let mut d = DebounceTracker::default();
        d.step(CaptureState::Idle, true);
        d.step(CaptureState::Idle, true);
        assert!(d.is_detected());
        d.step(CaptureState::Idle, false);
        assert!(!d.is_detected(), "clearing the signal clears Detected");
    }

    // -- episodes and the panel's ✕ ---------------------------------------

    #[test]
    fn nothing_is_dismissed_before_a_meeting_is_ever_noticed() {
        let d = DebounceTracker::default();
        assert_eq!(d.episode(), 0);
        assert!(!d.is_dismissed());
    }

    #[test]
    fn each_time_a_meeting_is_noticed_is_a_new_episode() {
        let mut d = DebounceTracker::default();
        d.step(CaptureState::Idle, true);
        d.step(CaptureState::Idle, true);
        assert_eq!(d.episode(), 1);

        // Still the same meeting: more present polls do not start a new one.
        d.step(CaptureState::Idle, true);
        d.step(CaptureState::Idle, true);
        assert_eq!(d.episode(), 1);

        // Meeting ends, another starts.
        d.step(CaptureState::Idle, false);
        d.step(CaptureState::Idle, true);
        d.step(CaptureState::Idle, true);
        assert_eq!(d.episode(), 2);
    }

    #[test]
    fn closing_the_panel_mutes_only_the_meeting_it_was_about() {
        let mut d = DebounceTracker::default();
        d.step(CaptureState::Idle, true);
        d.step(CaptureState::Idle, true);
        d.dismiss();
        assert!(d.is_dismissed());

        // The signal keeps holding — no second panel for the same meeting.
        d.step(CaptureState::Idle, true);
        assert!(d.is_dismissed());

        // The next meeting is nudged about as if nothing had happened.
        d.step(CaptureState::Idle, false);
        d.step(CaptureState::Idle, true);
        assert_eq!(d.step(CaptureState::Idle, true), StepEvent::Detected);
        assert!(!d.is_dismissed(), "✕ is not a snooze");
    }

    #[test]
    fn stopping_a_recording_mid_meeting_does_not_re_announce_it() {
        let mut d = DebounceTracker::default();
        d.step(CaptureState::Idle, true);
        assert_eq!(d.step(CaptureState::Idle, true), StepEvent::Detected);
        let episode = d.episode();

        // Recording runs while the call is still going.
        for _ in 0..5 {
            assert_eq!(d.step(CaptureState::Recording, true), StepEvent::None);
        }
        // The person stops it, with the call still open.
        assert_eq!(d.step(CaptureState::Idle, true), StepEvent::None);
        assert_eq!(d.step(CaptureState::Idle, true), StepEvent::None);
        assert_eq!(d.episode(), episode, "the same meeting, still");
    }

    #[test]
    fn a_meeting_recorded_and_then_really_over_still_counts_the_next_one() {
        let mut d = DebounceTracker::default();
        d.step(CaptureState::Idle, true);
        d.step(CaptureState::Idle, true);
        d.dismiss();
        d.step(CaptureState::Recording, true);
        // The call ends while recording is still on.
        d.step(CaptureState::Recording, false);
        assert!(!d.is_detected());
        // Later, a different meeting.
        d.step(CaptureState::Idle, true);
        assert_eq!(d.step(CaptureState::Idle, true), StepEvent::Detected);
        assert!(!d.is_dismissed(), "a new meeting is never pre-dismissed");
    }

    #[tokio::test]
    async fn the_watcher_exposes_the_episode_the_panel_dismisses() {
        let watcher = Watcher::with_source(true, quiet_source());
        assert_eq!(watcher.episode(), 0);
        assert!(!watcher.episode_dismissed());
        watcher.dismiss_episode();
        assert!(
            watcher.episode_dismissed(),
            "dismissing before anything is seen still marks the current episode"
        );
        // And it never reaches for the snooze.
        assert_eq!(watcher.status().state, DetectionState::Idle);
        assert!(watcher.status().snoozed_until.is_none());
    }

    // -- auto-stop suggestion ---------------------------------------------

    #[test]
    fn suggest_stop_fires_once_after_the_threshold_while_recording() {
        let mut d = DebounceTracker::default();
        let threshold = auto_stop_threshold_polls();
        for _ in 0..threshold - 1 {
            assert_eq!(d.step(CaptureState::Recording, false), StepEvent::None);
            assert!(!d.is_suggesting_stop());
        }
        assert_eq!(
            d.step(CaptureState::Recording, false),
            StepEvent::SuggestStop
        );
        assert!(d.is_suggesting_stop());
        // Stays latched, does not re-fire the edge every subsequent poll.
        assert_eq!(d.step(CaptureState::Recording, false), StepEvent::None);
        assert!(d.is_suggesting_stop());
    }

    #[test]
    fn suggest_stop_clears_the_moment_a_signal_returns() {
        let mut d = DebounceTracker::default();
        let threshold = auto_stop_threshold_polls();
        for _ in 0..threshold {
            d.step(CaptureState::Recording, false);
        }
        assert!(d.is_suggesting_stop());
        d.step(CaptureState::Recording, true);
        assert!(!d.is_suggesting_stop());
    }

    #[test]
    fn auto_stop_never_fires_outside_a_recording() {
        let mut d = DebounceTracker::default();
        let threshold = auto_stop_threshold_polls();
        for _ in 0..threshold + 5 {
            let event = d.step(CaptureState::Idle, false);
            assert_eq!(event, StepEvent::None);
        }
        assert!(!d.is_suggesting_stop());
    }

    // -- watcher-level: snooze, enable/disable, mocked probes --------------

    #[tokio::test]
    async fn snoozing_reports_snoozed_with_a_future_timestamp() {
        let watcher = Watcher::with_source(true, quiet_source());
        watcher.snooze(60).await.unwrap();
        let status = watcher.status();
        assert_eq!(status.state, DetectionState::Snoozed);
        let until = status.snoozed_until.expect("snooze sets a timestamp");
        let parsed = DateTime::parse_from_rfc3339(&until).unwrap();
        assert!(parsed.with_timezone(&Utc) > Utc::now());
    }

    #[tokio::test]
    async fn disabling_detection_reports_off_regardless_of_prior_state() {
        let watcher = Watcher::with_source(true, quiet_source());
        watcher.snooze(60).await.unwrap();
        watcher.set_enabled(false).await.unwrap();
        assert_eq!(watcher.status().state, DetectionState::Off);
    }

    #[tokio::test]
    async fn a_mocked_meeting_app_shows_up_in_poll_once() {
        let watcher = Watcher::with_source(
            true,
            Box::new(MockSource {
                apps: vec!["Zoom"],
                input_in_use: false,
            }),
        );
        let signals = watcher.poll_once().await.unwrap();
        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0].source, DetectionSource::MeetingApp);
        assert_eq!(signals[0].app.as_deref(), Some("Zoom"));
    }

    #[tokio::test]
    async fn a_mocked_input_device_signal_has_no_app_name() {
        let watcher = Watcher::with_source(
            true,
            Box::new(MockSource {
                apps: Vec::new(),
                input_in_use: true,
            }),
        );
        let signals = watcher.poll_once().await.unwrap();
        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0].source, DetectionSource::InputDeviceInUse);
        assert!(signals[0].app.is_none());
    }

    #[tokio::test]
    async fn poll_once_never_touches_the_debounce_state() {
        let watcher = Watcher::with_source(
            true,
            Box::new(MockSource {
                apps: vec!["Zoom"],
                input_in_use: false,
            }),
        );
        for _ in 0..10 {
            watcher.poll_once().await.unwrap();
        }
        // A diagnostic "check now" is not the watcher deciding a meeting
        // was detected; status is untouched without a real tick.
        assert_eq!(watcher.status().state, DetectionState::Idle);
    }
}
