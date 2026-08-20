//! Noticing that a meeting is happening.
//!
//! Two things are visible from outside a meeting, polled every
//! [`POLL_INTERVAL_SECS`] (DESIGN §3) — and they are emphatically **not worth
//! the same**:
//!
//! 1. **A known meeting app is running** (zoom.us, Teams, Webex, Discord,
//!    Slack, Meet). Corroborating evidence only. On its own it means nothing:
//!    Slack and Discord sit open from breakfast to bedtime, and Zoom lives in
//!    the tray between calls. "Zoom is running" is not "you are in a meeting",
//!    and treating it as one is how Echo used to interrupt people who were
//!    doing nothing at all.
//! 2. **Something other than Echo is holding an input device** (macOS
//!    CoreAudio, Linux PipeWire). This is the signal that means *someone is
//!    talking right now* — but it also trips on dictation, Siri, a voice
//!    message, any app that grabs the mic for a moment.
//!
//! So the microphone is necessary, and a running meeting app only buys speed:
//!
//! | what a poll sees | what the watcher does |
//! |---|---|
//! | meeting app, mic quiet | nothing. Not detected, no badge, no nudge, ever. |
//! | meeting app **and** mic | nudge after [`DEBOUNCE_POLLS`] polls ([`DEBOUNCE_SECS`]s) |
//! | mic alone | nudge after [`MIC_ONLY_DEBOUNCE_POLLS`] polls ([`MIC_ONLY_DEBOUNCE_SECS`]s) |
//! | neither | nothing |
//!
//! The mic-alone wait is picked to sit above dictation and Siri (a few
//! seconds, and never half a minute) and below a meeting anyone would mind
//! missing the top of. It has to exist: Google Meet in a browser tab has no
//! process name Echo can match, so the microphone alone is the *only* way that
//! meeting is ever noticed.
//!
//! Crossing the threshold latches [`DetectionState::Detected`]: a tray badge,
//! plus exactly one nudge. The nudge is the small floating panel near the menu
//! bar ([`crate::panel`]) — "Meeting detected", how long ago, and a Start
//! button right there. When a window cannot be put on screen at all, it falls
//! back to an OS notification instead (a plain "click opens Echo" one; Tauri's
//! plugin has no action buttons on desktop, review finding 7). Never both: one
//! meeting noticed is one interruption. Closing the panel with ✕ mutes only
//! that episode — see [`DebounceTracker::dismiss`] — and is emphatically not a
//! snooze.
//!
//! Once latched, the latch is generous on purpose (hysteresis): a mic that
//! drops out for a moment — a headset swap, a mute button that really does
//! release the device — keeps the meeting alive for up to
//! [`LATCH_GRACE_POLLS`] polls ([`LATCH_GRACE_SECS`]s) as long as a meeting app
//! is still running, so a flickering signal cannot manufacture a second
//! episode and a second nudge for one meeting. It is *bounded* rather than
//! open-ended because of rule 1 above: if the mic never comes back, all that is
//! left is an idle app, and an idle app must never leave the UI or the tray
//! claiming a meeting is happening.
//!
//! While a meeting *is* being recorded, the same signals are watched the other
//! way around. "Signals clear" for the auto-stop question means the
//! corroborated notion — **the microphone is quiet** — and deliberately not
//! "no meeting app is running": a Slack that never quits must not be able to
//! keep the safety net switched off. Once the mic has been quiet for
//! [`AUTO_STOP_SUGGEST_SECS`], the watcher sets `suggest_stop` on
//! [`DetectionStatus`] so the UI can offer a gentle "still going?" prompt — it
//! never stops the recording itself. Echo's own capture is excluded from the
//! mic signal (see the platform probes), without which this half of the
//! machine could never fire at all while Echo holds the microphone.
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

/// Corroborated evidence — a meeting app running *and* the microphone live —
/// has to hold for this many consecutive polls before we say anything, so
/// alt-tabbing past Zoom while a voice message plays does not fire a
/// notification.
pub const DEBOUNCE_POLLS: u32 = 2;

/// The same span, in seconds, for anything that wants to talk about it in
/// wall-clock terms (docs, the idle-poll-stays-cheap test below).
pub const DEBOUNCE_SECS: u64 = POLL_INTERVAL_SECS * DEBOUNCE_POLLS as u64;

/// With no meeting app to corroborate it, the microphone has to hold for this
/// many consecutive polls. Long enough that dictation, Siri and a voice memo
/// are all over before it is reached; short enough that a meeting in a browser
/// tab — which has no process name to match — is still noticed in its opening
/// half-minute.
pub const MIC_ONLY_DEBOUNCE_POLLS: u32 = 6;

/// [`MIC_ONLY_DEBOUNCE_POLLS`] in wall-clock seconds.
pub const MIC_ONLY_DEBOUNCE_SECS: u64 = POLL_INTERVAL_SECS * MIC_ONLY_DEBOUNCE_POLLS as u64;

/// How long an already-noticed meeting survives a quiet microphone while a
/// meeting app is still running. Hysteresis: keeps one flickering meeting from
/// becoming two episodes and two nudges. Bounded, because after this an idle
/// app is all that is left, and an idle app is not a meeting.
pub const LATCH_GRACE_POLLS: u32 = 6;

/// [`LATCH_GRACE_POLLS`] in wall-clock seconds.
pub const LATCH_GRACE_SECS: u64 = POLL_INTERVAL_SECS * LATCH_GRACE_POLLS as u64;

/// Once the microphone has been quiet this long during a recording, suggest
/// stopping. A running meeting app does not count as "not clear" here — see
/// the module docs.
pub const AUTO_STOP_SUGGEST_SECS: u64 = 120;

/// What "Pause detection" in the tray gives you.
pub const SNOOZE_MINUTES: u64 = 60;

/// Process names that hint at a meeting. Matched case-insensitively against
/// the executable name.
///
/// Half of this list is chat apps people never quit, which is exactly why a
/// match here is only ever corroborating evidence for a live microphone and
/// never a meeting by itself.
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
    /// already deduplicated. Corroborating evidence only.
    fn running_meeting_apps(&self) -> Result<Vec<String>, DetectError>;
    /// Is some app *other than Echo* using an input device right now?
    ///
    /// Excluding Echo's own capture is what makes this signal usable while a
    /// recording is running — both for "is the meeting still going" and for the
    /// auto-stop safety net that depends on it.
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

/// Is some app other than Echo using an input device? CoreAudio on macOS,
/// PipeWire on Linux; `false` on anything else (there is nothing to ask).
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

/// What one poll saw, boiled down to the two facts the decision turns on.
///
/// Deliberately not the [`DetectionSignal`] list the UI gets: that one carries
/// names, timestamps and confidences for display, this one carries the
/// evidence. Keeping them apart is what lets the whole heuristic be tested
/// without a clock, a process table or a sound card.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct SignalRead {
    /// A known meeting app is running. Corroborating evidence only.
    meeting_app: bool,
    /// Something other than Echo is holding an input device.
    mic: bool,
}

/// The four states of the world, named, so the tests below read as English.
#[cfg(test)]
impl SignalRead {
    /// Nothing at all: no meeting app, quiet microphone.
    const fn quiet() -> Self {
        Self {
            meeting_app: false,
            mic: false,
        }
    }

    /// Slack has been open since this morning and nobody is talking.
    const fn app_only() -> Self {
        Self {
            meeting_app: true,
            mic: false,
        }
    }

    /// Someone is talking, but to something Echo cannot name — a browser
    /// meeting, or dictation.
    const fn mic_only() -> Self {
        Self {
            meeting_app: false,
            mic: true,
        }
    }

    /// The strong case: a meeting app running *and* the microphone live.
    const fn app_and_mic() -> Self {
        Self {
            meeting_app: true,
            mic: true,
        }
    }
}

impl SignalRead {
    /// Read the two facts back out of what the UI is being told, so there is
    /// one place that decides what a signal list means.
    fn from_signals(signals: &[DetectionSignal]) -> Self {
        Self {
            meeting_app: signals
                .iter()
                .any(|s| s.source == DetectionSource::MeetingApp),
            mic: signals
                .iter()
                .any(|s| s.source == DetectionSource::InputDeviceInUse),
        }
    }

    /// How many consecutive polls of *this* read it takes before Echo says
    /// something, or `None` when this read is never enough on its own however
    /// long it holds.
    const fn polls_to_nudge(self) -> Option<u32> {
        match (self.mic, self.meeting_app) {
            // A live mic with a meeting app behind it: as fast as we dare.
            (true, true) => Some(DEBOUNCE_POLLS),
            // A live mic and nothing to corroborate it: wait out dictation.
            (true, false) => Some(MIC_ONLY_DEBOUNCE_POLLS),
            // An app and no mic is not a meeting, today or in an hour.
            (false, _) => None,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct DebounceTracker {
    /// Consecutive polls with a live microphone. The streak that matters:
    /// nothing is ever detected without it.
    mic_streak: u32,
    /// Consecutive polls with a quiet microphone, for the latch's grace period.
    quiet_streak: u32,
    /// Consecutive *recording* polls with a quiet microphone, for auto-stop.
    /// Kept apart from `quiet_streak` so time spent idle before a recording
    /// cannot suggest stopping the moment one starts.
    absent_streak: u32,
    detected_latched: bool,
    stop_suggested: bool,
    /// Counts detection *episodes*: one per meeting Echo actually announces. A
    /// meeting that is noticed, waved away, ends, and starts again is a new
    /// episode and gets a fresh nudge. Latching during a recording does not
    /// count — there is nothing to announce while Echo is already listening.
    episode: u64,
    /// The episode the person closed the panel on. Only that one is muted —
    /// this is not a snooze, detection keeps working exactly as before.
    dismissed_episode: Option<u64>,
}

impl DebounceTracker {
    /// Advance one poll with what that poll saw. `capture_state` decides which
    /// half of the machine is live (DESIGN §3: detection vs. auto-stop are
    /// different concerns).
    fn step(&mut self, capture_state: CaptureState, read: SignalRead) -> StepEvent {
        let recording = capture_state == CaptureState::Recording;

        if read.mic {
            self.mic_streak = self.mic_streak.saturating_add(1);
            self.quiet_streak = 0;
            // Somebody is talking: whatever else happens, this is not a
            // recording anyone has forgotten about.
            self.absent_streak = 0;
            self.stop_suggested = false;

            if recording {
                // Already recording: a nudge has nothing left to offer. The
                // meeting is a fact, so hold the latch — without counting an
                // episode, because nothing is being announced — so that
                // stopping mid-call is not answered with a "meeting detected"
                // panel ten seconds later, about the meeting the person just
                // decided to stop recording.
                self.detected_latched = true;
                return StepEvent::None;
            }

            let Some(needed) = read.polls_to_nudge() else {
                return StepEvent::None;
            };
            if self.mic_streak >= needed && !self.detected_latched {
                self.detected_latched = true;
                self.episode = self.episode.wrapping_add(1);
                return StepEvent::Detected;
            }
            StepEvent::None
        } else {
            // Quiet microphone. A meeting app may well still be running, and it
            // buys nothing here: it cannot start a meeting (rule 1), and it
            // cannot keep the auto-stop safety net switched off.
            self.mic_streak = 0;
            self.quiet_streak = self.quiet_streak.saturating_add(1);

            if self.detected_latched
                && !(read.meeting_app && self.quiet_streak <= LATCH_GRACE_POLLS)
            {
                // Either there is nothing left at all, or the grace period for
                // a mic that might come back has run out. Whichever it is, we
                // are no longer willing to say a meeting is happening — and the
                // next one is a new episode.
                self.detected_latched = false;
            }

            if !recording {
                // Auto-stop only means something while something is being
                // recorded.
                self.absent_streak = 0;
                self.stop_suggested = false;
                return StepEvent::None;
            }

            self.absent_streak = self.absent_streak.saturating_add(1);
            if self.absent_streak >= auto_stop_threshold_polls() && !self.stop_suggested {
                self.stop_suggested = true;
                StepEvent::SuggestStop
            } else {
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

    /// Look once, and describe honestly what was seen.
    ///
    /// The confidences say what each observation is worth on its own, which is
    /// the whole point of the rework: a running app is a weak hint whatever the
    /// app is, a live microphone is the real signal, and the two together are
    /// as sure as Echo gets without opening the audio stream itself. Meeting
    /// apps come first in the list so that the name shown to the person is the
    /// app's, when there is one.
    fn collect_signals(&self) -> Result<Vec<DetectionSignal>, DetectError> {
        let now = Utc::now().to_rfc3339();
        let mut signals = Vec::new();

        let mic_in_use = self.source.input_device_in_use()?;
        let apps = self.source.running_meeting_apps()?;
        let corroborated = mic_in_use && !apps.is_empty();

        for app_name in apps {
            signals.push(DetectionSignal {
                source: DetectionSource::MeetingApp,
                app: Some(app_name),
                since: now.clone(),
                // Corroborating evidence only. Half this list is chat apps
                // people never quit, so "it is running" is close to worthless
                // by itself — and is treated as worthless by the state machine.
                confidence: 0.25,
            });
        }

        if mic_in_use {
            signals.push(DetectionSignal {
                source: DetectionSource::InputDeviceInUse,
                app: None,
                since: now,
                // Something other than Echo is listening. That alone could be
                // dictation; with a meeting app running too, it is a meeting.
                confidence: if corroborated { 0.9 } else { 0.6 },
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
        let read = SignalRead::from_signals(&signals);
        let edge = self.debounce.lock().unwrap().step(capture_state, read);

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
                // Only ever true with corroborated evidence behind it (a live
                // microphone), so neither this state nor the tray badge it
                // drives can claim a meeting because Slack happens to be open.
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

    #[test]
    fn the_thresholds_sit_where_the_heuristic_needs_them() {
        const {
            assert!(
                MIC_ONLY_DEBOUNCE_SECS > 20,
                "a mic alone must outlast dictation and Siri"
            )
        };
        const {
            assert!(
                MIC_ONLY_DEBOUNCE_SECS <= 40,
                "a browser meeting has nothing else to be noticed by; do not \
                 make it wait"
            )
        };
        const {
            assert!(
                DEBOUNCE_SECS < MIC_ONLY_DEBOUNCE_SECS,
                "corroborated evidence must be the fast path"
            )
        };
        const {
            assert!(
                AUTO_STOP_SUGGEST_SECS > MIC_ONLY_DEBOUNCE_SECS,
                "auto-stop must be slower to speak than detection"
            )
        };
    }

    // -- what each combination of signals is worth ------------------------

    #[test]
    fn a_meeting_app_on_its_own_is_never_enough_however_long_it_runs() {
        assert_eq!(SignalRead::app_only().polls_to_nudge(), None);
    }

    #[test]
    fn a_live_mic_is_faster_when_a_meeting_app_corroborates_it() {
        assert_eq!(
            SignalRead::app_and_mic().polls_to_nudge(),
            Some(DEBOUNCE_POLLS)
        );
        assert_eq!(
            SignalRead::mic_only().polls_to_nudge(),
            Some(MIC_ONLY_DEBOUNCE_POLLS)
        );
    }

    #[test]
    fn a_signal_list_reads_back_as_the_evidence_it_describes() {
        let app = DetectionSignal {
            source: DetectionSource::MeetingApp,
            app: Some("Slack".into()),
            ..Default::default()
        };
        let mic = DetectionSignal {
            source: DetectionSource::InputDeviceInUse,
            ..Default::default()
        };
        assert_eq!(SignalRead::from_signals(&[]), SignalRead::quiet());
        assert_eq!(
            SignalRead::from_signals(std::slice::from_ref(&app)),
            SignalRead::app_only()
        );
        assert_eq!(
            SignalRead::from_signals(std::slice::from_ref(&mic)),
            SignalRead::mic_only()
        );
        assert_eq!(
            SignalRead::from_signals(&[app, mic]),
            SignalRead::app_and_mic()
        );
    }

    // -- the false positive this heuristic exists to kill -------------------

    #[test]
    fn slack_open_all_day_never_fires() {
        let mut d = DebounceTracker::default();
        // Eight hours of Slack sitting there, nobody talking.
        for _ in 0..(8 * 60 * 60 / POLL_INTERVAL_SECS) {
            assert_eq!(
                d.step(CaptureState::Idle, SignalRead::app_only()),
                StepEvent::None
            );
        }
        assert!(
            !d.is_detected(),
            "an app that is merely running is not a meeting"
        );
        assert_eq!(d.episode(), 0, "and it is not an episode either");
    }

    #[test]
    fn fifteen_seconds_of_dictation_never_fires() {
        let mut d = DebounceTracker::default();
        for _ in 0..(15 / POLL_INTERVAL_SECS) {
            assert_eq!(
                d.step(CaptureState::Idle, SignalRead::mic_only()),
                StepEvent::None
            );
        }
        // Dictation ends.
        d.step(CaptureState::Idle, SignalRead::quiet());
        assert!(!d.is_detected());
        assert_eq!(d.episode(), 0);
    }

    #[test]
    fn a_browser_meeting_on_the_mic_alone_fires_within_thirty_five_seconds() {
        let mut d = DebounceTracker::default();
        let mut fired_at_secs = None;
        let mut fires = 0;
        for poll in 1..=(35 / POLL_INTERVAL_SECS) {
            if d.step(CaptureState::Idle, SignalRead::mic_only()) == StepEvent::Detected {
                fires += 1;
                fired_at_secs.get_or_insert(poll * POLL_INTERVAL_SECS);
            }
        }
        assert_eq!(fired_at_secs, Some(30), "Google Meet in a tab, noticed");
        assert_eq!(fired_at_secs, Some(MIC_ONLY_DEBOUNCE_SECS));
        assert_eq!(fires, 1, "exactly one nudge per meeting");
        assert!(d.is_detected());
    }

    #[test]
    fn zoom_plus_a_live_mic_fires_after_ten_seconds() {
        let mut d = DebounceTracker::default();
        assert_eq!(
            d.step(CaptureState::Idle, SignalRead::app_and_mic()),
            StepEvent::None,
            "one poll is not enough"
        );
        assert_eq!(
            d.step(CaptureState::Idle, SignalRead::app_and_mic()),
            StepEvent::Detected
        );
        assert!(d.is_detected());
    }

    #[test]
    fn an_idle_app_cannot_shorten_the_wait_for_a_mic_that_is_not_yet_there() {
        let mut d = DebounceTracker::default();
        // Slack has been open for a while; then the mic comes up. The wait is
        // measured from the mic, not from the app.
        for _ in 0..10 {
            d.step(CaptureState::Idle, SignalRead::app_only());
        }
        assert_eq!(
            d.step(CaptureState::Idle, SignalRead::app_and_mic()),
            StepEvent::None
        );
        assert_eq!(
            d.step(CaptureState::Idle, SignalRead::app_and_mic()),
            StepEvent::Detected
        );
    }

    #[test]
    fn detection_does_not_renotify_while_the_meeting_holds() {
        let mut d = DebounceTracker::default();
        d.step(CaptureState::Idle, SignalRead::app_and_mic());
        assert_eq!(
            d.step(CaptureState::Idle, SignalRead::app_and_mic()),
            StepEvent::Detected
        );
        for _ in 0..20 {
            assert_eq!(
                d.step(CaptureState::Idle, SignalRead::app_and_mic()),
                StepEvent::None
            );
        }
        assert!(d.is_detected());
    }

    #[test]
    fn a_quiet_poll_resets_the_mic_streak() {
        let mut d = DebounceTracker::default();
        d.step(CaptureState::Idle, SignalRead::app_and_mic());
        // Mic drops before the second poll lands: the streak starts over.
        d.step(CaptureState::Idle, SignalRead::app_only());
        assert!(!d.is_detected());
        assert_eq!(
            d.step(CaptureState::Idle, SignalRead::app_and_mic()),
            StepEvent::None
        );
        assert_eq!(
            d.step(CaptureState::Idle, SignalRead::app_and_mic()),
            StepEvent::Detected
        );
    }

    #[test]
    fn detected_clears_once_there_is_nothing_left_to_hear() {
        let mut d = DebounceTracker::default();
        d.step(CaptureState::Idle, SignalRead::mic_only());
        for _ in 1..MIC_ONLY_DEBOUNCE_POLLS {
            d.step(CaptureState::Idle, SignalRead::mic_only());
        }
        assert!(d.is_detected());
        d.step(CaptureState::Idle, SignalRead::quiet());
        assert!(
            !d.is_detected(),
            "no app, no mic: there is nothing to claim a meeting from"
        );
    }

    // -- hysteresis: one meeting stays one meeting -------------------------

    #[test]
    fn a_latched_meeting_survives_the_mic_dropping_briefly() {
        let mut d = DebounceTracker::default();
        d.step(CaptureState::Idle, SignalRead::app_and_mic());
        assert_eq!(
            d.step(CaptureState::Idle, SignalRead::app_and_mic()),
            StepEvent::Detected
        );

        // Headset swap: the mic goes away for a few polls, Zoom stays open.
        for _ in 0..LATCH_GRACE_POLLS {
            assert_eq!(
                d.step(CaptureState::Idle, SignalRead::app_only()),
                StepEvent::None
            );
            assert!(d.is_detected(), "still the same meeting");
        }
        // Mic back: no second episode, so no second nudge.
        assert_eq!(
            d.step(CaptureState::Idle, SignalRead::app_and_mic()),
            StepEvent::None
        );
        assert_eq!(d.episode(), 1);
    }

    #[test]
    fn a_flickering_mic_cannot_manufacture_extra_episodes() {
        let mut d = DebounceTracker::default();
        d.step(CaptureState::Idle, SignalRead::app_and_mic());
        d.step(CaptureState::Idle, SignalRead::app_and_mic());
        for _ in 0..10 {
            d.step(CaptureState::Idle, SignalRead::app_only());
            d.step(CaptureState::Idle, SignalRead::app_and_mic());
        }
        assert_eq!(d.episode(), 1, "one meeting, however jittery the signal");
        assert!(d.is_detected());
    }

    #[test]
    fn the_latch_lets_go_when_only_an_idle_app_is_left() {
        let mut d = DebounceTracker::default();
        d.step(CaptureState::Idle, SignalRead::app_and_mic());
        d.step(CaptureState::Idle, SignalRead::app_and_mic());
        assert!(d.is_detected());

        // The call is over; Slack, as ever, is not.
        for _ in 0..=LATCH_GRACE_POLLS {
            d.step(CaptureState::Idle, SignalRead::app_only());
        }
        assert!(
            !d.is_detected(),
            "grace is for a mic that comes back, not a licence to keep \
             claiming a meeting because an app is open"
        );
    }

    // -- episodes and the panel's ✕ ---------------------------------------

    fn latch_a_meeting(d: &mut DebounceTracker) {
        d.step(CaptureState::Idle, SignalRead::app_and_mic());
        d.step(CaptureState::Idle, SignalRead::app_and_mic());
    }

    #[test]
    fn nothing_is_dismissed_before_a_meeting_is_ever_noticed() {
        let d = DebounceTracker::default();
        assert_eq!(d.episode(), 0);
        assert!(!d.is_dismissed());
    }

    #[test]
    fn each_time_a_meeting_is_noticed_is_a_new_episode() {
        let mut d = DebounceTracker::default();
        latch_a_meeting(&mut d);
        assert_eq!(d.episode(), 1);

        // Still the same meeting: more polls do not start a new one.
        latch_a_meeting(&mut d);
        assert_eq!(d.episode(), 1);

        // Meeting ends properly, another starts later.
        d.step(CaptureState::Idle, SignalRead::quiet());
        latch_a_meeting(&mut d);
        assert_eq!(d.episode(), 2);
    }

    #[test]
    fn closing_the_panel_mutes_only_the_meeting_it_was_about() {
        let mut d = DebounceTracker::default();
        latch_a_meeting(&mut d);
        d.dismiss();
        assert!(d.is_dismissed());

        // The meeting keeps going — no second panel for it.
        d.step(CaptureState::Idle, SignalRead::app_and_mic());
        assert!(d.is_dismissed());

        // The next meeting is nudged about as if nothing had happened.
        d.step(CaptureState::Idle, SignalRead::quiet());
        d.step(CaptureState::Idle, SignalRead::app_and_mic());
        assert_eq!(
            d.step(CaptureState::Idle, SignalRead::app_and_mic()),
            StepEvent::Detected
        );
        assert!(!d.is_dismissed(), "✕ is not a snooze");
    }

    #[test]
    fn a_dismissed_episode_then_a_real_meeting_later_gets_its_own_nudge() {
        let mut d = DebounceTracker::default();
        latch_a_meeting(&mut d);
        d.dismiss();

        // That meeting ends; an hour of Slack-and-silence follows.
        d.step(CaptureState::Idle, SignalRead::quiet());
        for _ in 0..(60 * 60 / POLL_INTERVAL_SECS) {
            assert_eq!(
                d.step(CaptureState::Idle, SignalRead::app_only()),
                StepEvent::None,
                "the waved-away meeting must not come back as a new one"
            );
        }

        // A real meeting, in the browser this time: mic only, and it still
        // gets its own nudge.
        for _ in 1..MIC_ONLY_DEBOUNCE_POLLS {
            assert_eq!(
                d.step(CaptureState::Idle, SignalRead::mic_only()),
                StepEvent::None
            );
        }
        assert_eq!(
            d.step(CaptureState::Idle, SignalRead::mic_only()),
            StepEvent::Detected
        );
        assert_eq!(d.episode(), 2);
        assert!(!d.is_dismissed(), "a new meeting is never pre-dismissed");
    }

    #[test]
    fn stopping_a_recording_mid_meeting_does_not_re_announce_it() {
        let mut d = DebounceTracker::default();
        d.step(CaptureState::Idle, SignalRead::app_and_mic());
        assert_eq!(
            d.step(CaptureState::Idle, SignalRead::app_and_mic()),
            StepEvent::Detected
        );
        let episode = d.episode();

        // Recording runs while the call is still going.
        for _ in 0..5 {
            assert_eq!(
                d.step(CaptureState::Recording, SignalRead::app_and_mic()),
                StepEvent::None
            );
        }
        // The person stops it, with the call still open.
        assert_eq!(
            d.step(CaptureState::Idle, SignalRead::app_and_mic()),
            StepEvent::None
        );
        assert_eq!(
            d.step(CaptureState::Idle, SignalRead::app_and_mic()),
            StepEvent::None
        );
        assert_eq!(d.episode(), episode, "the same meeting, still");
    }

    #[test]
    fn stopping_a_manually_started_recording_mid_meeting_says_nothing_either() {
        let mut d = DebounceTracker::default();
        // Nothing was ever detected: the person pressed Start themselves.
        for _ in 0..10 {
            assert_eq!(
                d.step(CaptureState::Recording, SignalRead::app_and_mic()),
                StepEvent::None
            );
        }
        // They stop while the call is still open. A "meeting detected" panel
        // now would be about the meeting they just stopped recording.
        for _ in 0..5 {
            assert_eq!(
                d.step(CaptureState::Idle, SignalRead::app_and_mic()),
                StepEvent::None
            );
        }
        assert_eq!(d.episode(), 0);
    }

    #[test]
    fn a_meeting_recorded_and_then_really_over_still_counts_the_next_one() {
        let mut d = DebounceTracker::default();
        latch_a_meeting(&mut d);
        d.dismiss();
        d.step(CaptureState::Recording, SignalRead::app_and_mic());
        // The call ends while recording is still on, and the app quits too.
        for _ in 0..=LATCH_GRACE_POLLS {
            d.step(CaptureState::Recording, SignalRead::quiet());
        }
        assert!(!d.is_detected());
        // Later, a different meeting.
        d.step(CaptureState::Idle, SignalRead::app_and_mic());
        assert_eq!(
            d.step(CaptureState::Idle, SignalRead::app_and_mic()),
            StepEvent::Detected
        );
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
            assert_eq!(
                d.step(CaptureState::Recording, SignalRead::quiet()),
                StepEvent::None
            );
            assert!(!d.is_suggesting_stop());
        }
        assert_eq!(
            d.step(CaptureState::Recording, SignalRead::quiet()),
            StepEvent::SuggestStop
        );
        assert!(d.is_suggesting_stop());
        // Stays latched, does not re-fire the edge every subsequent poll.
        assert_eq!(
            d.step(CaptureState::Recording, SignalRead::quiet()),
            StepEvent::None
        );
        assert!(d.is_suggesting_stop());
    }

    #[test]
    fn auto_stop_still_suggests_when_the_call_ends_but_slack_keeps_running() {
        let mut d = DebounceTracker::default();
        latch_a_meeting(&mut d);
        // Recording a real call, mic live.
        for _ in 0..20 {
            d.step(CaptureState::Recording, SignalRead::app_and_mic());
        }
        // Everyone hangs up. Slack, of course, is still open, and Echo is
        // still recording an empty room.
        let threshold = auto_stop_threshold_polls();
        for _ in 0..threshold - 1 {
            assert_eq!(
                d.step(CaptureState::Recording, SignalRead::app_only()),
                StepEvent::None
            );
        }
        assert_eq!(
            d.step(CaptureState::Recording, SignalRead::app_only()),
            StepEvent::SuggestStop,
            "an app nobody ever quits must not switch off the safety net"
        );
        assert!(d.is_suggesting_stop());
    }

    #[test]
    fn suggest_stop_clears_the_moment_the_mic_comes_back() {
        let mut d = DebounceTracker::default();
        let threshold = auto_stop_threshold_polls();
        for _ in 0..threshold {
            d.step(CaptureState::Recording, SignalRead::quiet());
        }
        assert!(d.is_suggesting_stop());
        d.step(CaptureState::Recording, SignalRead::mic_only());
        assert!(!d.is_suggesting_stop(), "somebody is talking again");
    }

    #[test]
    fn auto_stop_never_fires_outside_a_recording() {
        let mut d = DebounceTracker::default();
        let threshold = auto_stop_threshold_polls();
        for _ in 0..threshold + 5 {
            assert_eq!(
                d.step(CaptureState::Idle, SignalRead::quiet()),
                StepEvent::None
            );
        }
        assert!(!d.is_suggesting_stop());
    }

    #[test]
    fn a_long_quiet_idle_stretch_does_not_suggest_stopping_a_fresh_recording() {
        let mut d = DebounceTracker::default();
        // An hour of nothing at all before the person presses Start.
        for _ in 0..(60 * 60 / POLL_INTERVAL_SECS) {
            d.step(CaptureState::Idle, SignalRead::quiet());
        }
        // The recording is seconds old; nobody has forgotten anything yet.
        for _ in 0..3 {
            assert_eq!(
                d.step(CaptureState::Recording, SignalRead::quiet()),
                StepEvent::None
            );
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
    async fn an_idle_meeting_app_never_tells_the_ui_a_meeting_is_happening() {
        let watcher = Watcher::with_source(
            true,
            Box::new(MockSource {
                apps: vec!["Slack"],
                input_in_use: false,
            }),
        );
        // Half an hour of Slack being Slack.
        for _ in 0..(30 * 60 / POLL_INTERVAL_SECS) {
            watcher.tick().await;
        }
        let status = watcher.status();
        assert_eq!(
            status.state,
            DetectionState::Idle,
            "no panel, no notification, and no tray badge either: the badge \
             follows this state"
        );
        assert_eq!(watcher.episode(), 0);

        // The observation is still reported honestly — at what it is worth.
        let app_signal = status
            .signals
            .iter()
            .find(|s| s.source == DetectionSource::MeetingApp)
            .expect("Slack is running and Echo says so plainly");
        assert!(
            app_signal.confidence < 0.5,
            "a running app is a hint, never a meeting"
        );
    }

    #[tokio::test]
    async fn a_zoom_call_with_a_live_mic_reaches_the_ui_after_the_fast_debounce() {
        let watcher = Watcher::with_source(
            true,
            Box::new(MockSource {
                apps: vec!["Zoom"],
                input_in_use: true,
            }),
        );
        watcher.tick().await;
        assert_eq!(watcher.status().state, DetectionState::Idle, "10s, not 5s");
        watcher.tick().await;

        let status = watcher.status();
        assert_eq!(status.state, DetectionState::Detected);
        assert_eq!(watcher.episode(), 1);
        let mic = status
            .signals
            .iter()
            .find(|s| s.source == DetectionSource::InputDeviceInUse)
            .expect("the mic is what makes this a meeting");
        assert!(
            mic.confidence > 0.8,
            "a live mic with a meeting app behind it is as sure as we get"
        );
    }

    #[tokio::test]
    async fn fifteen_seconds_of_dictation_never_reaches_the_ui() {
        let watcher = Watcher::with_source(
            true,
            Box::new(MockSource {
                apps: Vec::new(),
                input_in_use: true,
            }),
        );
        for _ in 0..(15 / POLL_INTERVAL_SECS) {
            watcher.tick().await;
        }
        assert_eq!(watcher.status().state, DetectionState::Idle);

        // Keep going and the same mic, alone, does become a meeting: this is
        // the only way a browser call is ever noticed.
        for _ in 0..MIC_ONLY_DEBOUNCE_POLLS {
            watcher.tick().await;
        }
        assert_eq!(watcher.status().state, DetectionState::Detected);
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
