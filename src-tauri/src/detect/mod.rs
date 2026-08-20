//! Noticing that a meeting is happening.
//!
//! Three things are visible from outside a meeting, polled every
//! [`POLL_INTERVAL_SECS`] (DESIGN §3) — and they are emphatically **not worth
//! the same**:
//!
//! 1. **A known call app is running** (Zoom, Teams, Webex, FaceTime, Discord,
//!    Slack, Meet, and the other names in [`MEETING_APPS`]). Corroborating
//!    evidence only. On its own it means nothing: Slack and Discord sit open
//!    from breakfast to bedtime, and Zoom lives in the tray between calls.
//!    "Zoom is running" is not "you are in a meeting", and treating it as one is
//!    how Echo used to interrupt people who were doing nothing at all.
//! 2. **A browser is running** ([`BROWSERS`]). Weaker still — a browser is
//!    always open — and it is here for one job: a meeting in a tab has no
//!    process name of its own, so a browser is the only thing that makes
//!    "somebody is using the microphone" *possibly* a meeting rather than
//!    certainly not one.
//! 3. **Something other than Echo is holding an input device** (macOS
//!    CoreAudio, Linux PipeWire). This is the signal that means *someone is
//!    talking right now* — but it also trips on dictation, Siri, a voice
//!    message, any app that grabs the mic for a moment.
//!
//! So the microphone is necessary and never sufficient. Something has to make
//! the shape of the evidence *call-shaped* before Echo says a word:
//!
//! | what a poll sees | what the watcher does |
//! |---|---|
//! | call app, mic quiet | nothing. Not detected, no badge, no nudge, ever. |
//! | browser, mic quiet | nothing, obviously. |
//! | call app **and** mic | nudge after [`DEBOUNCE_POLLS`] polls ([`DEBOUNCE_SECS`]s) |
//! | browser **and** mic, no call app | nudge after [`MIC_ONLY_DEBOUNCE_POLLS`] polls ([`MIC_ONLY_DEBOUNCE_SECS`]s) |
//! | mic alone, nothing that can make a call | **nothing, however long it holds** |
//! | neither | nothing |
//!
//! That last row is the difference between Echo and a nuisance. A microphone
//! held with no browser and no call app anywhere on the machine is a dictation
//! utility, a voice memo, a transcription tool, a game — the person is not in a
//! meeting, and no amount of patience makes them in one. Echo used to nudge
//! about it anyway; people reported it as "it thinks every sound is a meeting",
//! and they were right to.
//!
//! The browser-and-mic wait is picked to sit above dictation and Siri (a few
//! seconds, and never half a minute) and below a meeting anyone would mind
//! missing the top of. It has to exist: Google Meet in a browser tab has no
//! process name Echo can match, so the microphone plus an open browser is the
//! *only* way that meeting is ever noticed.
//!
//! None of this can work if the microphone signal itself lies, and it used to:
//! a headset is one duplex device, and asking that device whether it is "running"
//! says yes while music plays out of it. The `macos` module beside this one
//! carries that fix and its reasoning; this file's job is to be un-fooled by
//! anything that gets past it.
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
//! [`LATCH_GRACE_POLLS`] polls ([`LATCH_GRACE_SECS`]s) as long as something that
//! could be carrying the call is still running, so a flickering signal cannot
//! manufacture a second episode and a second nudge for one meeting. It is
//! *bounded* rather than open-ended because of rule 1 above: if the mic never
//! comes back, all that is left is an idle app, and an idle app must never leave
//! the UI or the tray claiming a meeting is happening.
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

/// Process names of apps that can place or take a call. Matched
/// case-insensitively against the executable name, on whole words (see
/// [`friendly_app_name`]).
///
/// Half of this list is chat apps people never quit, which is exactly why a
/// match here is only ever corroborating evidence for a live microphone and
/// never a meeting by itself. The other half — FaceTime, Skype, the VoIP names —
/// is here so that a call in one of them takes the fast path instead of waiting
/// out the browser timer for a meeting Echo could already be sure about.
pub const MEETING_APPS: &[(&str, &str)] = &[
    ("zoom", "Zoom"),
    ("Microsoft Teams", "Teams"),
    ("Teams", "Teams"),
    ("Webex", "Webex"),
    ("Discord", "Discord"),
    ("Slack", "Slack"),
    ("Google Meet", "Google Meet"),
    ("FaceTime", "FaceTime"),
    ("Skype", "Skype"),
    ("WhatsApp", "WhatsApp"),
    ("Telegram", "Telegram"),
    ("Signal", "Signal"),
    ("Chime", "Amazon Chime"),
    ("GoToMeeting", "GoTo"),
    ("GoTo", "GoTo"),
    ("BlueJeans", "BlueJeans"),
    ("RingCentral", "RingCentral"),
    ("Jitsi", "Jitsi"),
    ("Whereby", "Whereby"),
    ("Dialpad", "Dialpad"),
    ("Aircall", "Aircall"),
    ("Zoiper", "Zoiper"),
    ("Linphone", "Linphone"),
];

/// Process names of web browsers, same matching rules.
///
/// The weakest signal Echo has, and the only one that makes a browser meeting
/// noticeable at all: Google Meet, Teams-in-a-tab and every "join from your
/// browser" link run inside one of these and have no process name of their own.
/// A browser is never evidence of anything by itself — it is open right now on
/// the machine reading this — it only decides whether a live microphone is
/// *allowed* to become a meeting on its own.
pub const BROWSERS: &[(&str, &str)] = &[
    ("Google Chrome", "Chrome"),
    ("Chrome", "Chrome"),
    ("Chromium", "Chromium"),
    ("Safari", "Safari"),
    ("Firefox", "Firefox"),
    ("Microsoft Edge", "Edge"),
    ("msedge", "Edge"),
    ("Edge", "Edge"),
    ("Brave Browser", "Brave"),
    ("Brave", "Brave"),
    ("Arc", "Arc"),
    ("Opera", "Opera"),
    ("Vivaldi", "Vivaldi"),
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

/// What is running right now that could possibly be carrying a call.
///
/// Both lists hold friendly names, deduplicated and sorted. Deliberately two
/// lists and not one: they are worth completely different things, and the whole
/// point of this rework is that the difference decides whether Echo speaks.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RunningApps {
    /// Apps that can place or take a call ([`MEETING_APPS`]).
    pub call_apps: Vec<String>,
    /// Browsers ([`BROWSERS`]) — where a meeting with no process name lives.
    pub browsers: Vec<String>,
}

/// Everything the watcher needs from the outside world, behind a trait so
/// tests can hand it fake answers instead of asking `sysinfo`/CoreAudio for
/// real ones.
pub trait SignalSource: Send + Sync {
    /// Which call apps and browsers are running. Corroborating evidence only,
    /// neither list is a meeting by itself.
    fn running_apps(&self) -> Result<RunningApps, DetectError>;
    /// Is some app *other than Echo* using an input device right now?
    ///
    /// Excluding Echo's own capture is what makes this signal usable while a
    /// recording is running — both for "is the meeting still going" and for the
    /// auto-stop safety net that depends on it.
    fn input_device_in_use(&self) -> Result<bool, DetectError>;
}

struct SystemSignalSource;

impl SignalSource for SystemSignalSource {
    fn running_apps(&self) -> Result<RunningApps, DetectError> {
        running_apps()
    }

    fn input_device_in_use(&self) -> Result<bool, DetectError> {
        input_device_in_use()
    }
}

/// Which call apps and browsers are running right now, as friendly names.
/// `sysinfo`, no per-process detail beyond the name, and one walk of the
/// process table for both questions (mantra 1).
pub fn running_apps() -> Result<RunningApps, DetectError> {
    let mut system = sysinfo::System::new();
    system.refresh_processes(sysinfo::ProcessesToUpdate::All, true);

    let mut call_apps = BTreeSet::new();
    let mut browsers = BTreeSet::new();
    for process in system.processes().values() {
        let name = process.name().to_string_lossy();
        if let Some(friendly) = friendly_app_name(&name) {
            call_apps.insert(friendly.to_string());
        }
        if let Some(friendly) = friendly_browser_name(&name) {
            browsers.insert(friendly.to_string());
        }
    }
    Ok(RunningApps {
        call_apps: call_apps.into_iter().collect(),
        browsers: browsers.into_iter().collect(),
    })
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

/// Map a process name onto a friendly call-app name, or None when we do not
/// know it.
pub fn friendly_app_name(process_name: &str) -> Option<&'static str> {
    match_app(process_name, MEETING_APPS)
}

/// The same, for browsers.
pub fn friendly_browser_name(process_name: &str) -> Option<&'static str> {
    match_app(process_name, BROWSERS)
}

fn match_app(process_name: &str, table: &[(&str, &'static str)]) -> Option<&'static str> {
    let name = executable_name(process_name).to_ascii_lowercase();
    table
        .iter()
        .find(|(needle, _)| contains_word(&name, &needle.to_ascii_lowercase()))
        .map(|(_, friendly)| *friendly)
}

/// The last component of whatever the platform called the process, so a full
/// path matches on the executable rather than on a folder somewhere above it.
fn executable_name(process_name: &str) -> &str {
    process_name
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(process_name)
}

/// Does `needle` appear in `haystack` as a whole word?
///
/// Both are already lowercase. Whole-word rather than plain substring because
/// the names Echo has to recognise are short and the process table is long:
/// plain `contains("arc")` matches `searchd`, which is running on every Mac
/// right now, and a browser signal that is always on is not a signal. Word
/// edges are non-alphanumeric, so `Arc Helper`, `zoom.us`,
/// `Microsoft Teams (work)` and `Google Chrome Helper (Renderer)` all match
/// while `chromedriver` and `SafariBookmarksSyncAgent` do not.
fn contains_word(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    haystack.match_indices(needle).any(|(at, _)| {
        let before_is_word = haystack[..at]
            .chars()
            .next_back()
            .is_some_and(|c| c.is_alphanumeric());
        let after_is_word = haystack[at + needle.len()..]
            .chars()
            .next()
            .is_some_and(|c| c.is_alphanumeric());
        !before_is_word && !after_is_word
    })
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

/// One poll's raw answers, before anything is decided about them.
///
/// The two shapes it turns into are on purpose: [`Observation::read`] is the
/// evidence the state machine weighs, [`Observation::signals`] is what the
/// person is shown. The second is a description, not a decision — a running
/// browser is not in it, because "Chrome is open" is not an observation about a
/// meeting and rendering it as one would be noise in the very screen people
/// open when detection has behaved oddly.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Observation {
    call_apps: Vec<String>,
    browsers: Vec<String>,
    mic: bool,
}

impl Observation {
    /// Boil the poll down to the facts the decision turns on.
    fn read(&self) -> SignalRead {
        SignalRead {
            meeting_app: !self.call_apps.is_empty(),
            browser: !self.browsers.is_empty(),
            mic: self.mic,
        }
    }

    /// The same poll as the UI sees it.
    ///
    /// The confidences say what each observation is worth on its own, which is
    /// the whole point of the rework: a running app is a weak hint whatever the
    /// app is, a live microphone is the real signal, and the two together are as
    /// sure as Echo gets without opening the audio stream itself. Call apps come
    /// first in the list so that the name shown to the person is the app's, when
    /// there is one.
    fn signals(&self) -> Vec<DetectionSignal> {
        let now = Utc::now().to_rfc3339();
        let mut signals = Vec::new();
        let corroborated = self.mic && !self.call_apps.is_empty();

        for app_name in &self.call_apps {
            signals.push(DetectionSignal {
                source: DetectionSource::MeetingApp,
                app: Some(app_name.clone()),
                since: now.clone(),
                // Corroborating evidence only. Half this list is chat apps
                // people never quit, so "it is running" is close to worthless
                // by itself — and is treated as worthless by the state machine.
                confidence: 0.25,
            });
        }

        if self.mic {
            signals.push(DetectionSignal {
                source: DetectionSource::InputDeviceInUse,
                app: None,
                since: now,
                // Something other than Echo is listening. On its own that could
                // be dictation — and with nothing on the machine that could
                // carry a call, that is all it can be, which is why this number
                // is a description and the state machine does the deciding.
                confidence: if corroborated { 0.9 } else { 0.6 },
            });
        }

        signals
    }
}

/// What one poll saw, boiled down to the three facts the decision turns on.
///
/// Deliberately not the [`DetectionSignal`] list the UI gets: that one carries
/// names, timestamps and confidences for display, this one carries the
/// evidence. Keeping them apart is what lets the whole heuristic be tested
/// without a clock, a process table or a sound card.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct SignalRead {
    /// A known call app is running. Corroborating evidence only.
    meeting_app: bool,
    /// A browser is running. Weaker still — see the module docs.
    browser: bool,
    /// Something other than Echo is holding an input device.
    mic: bool,
}

/// The states of the world worth naming, so the tests below read as English.
#[cfg(test)]
impl SignalRead {
    /// Nothing at all: no call app, no browser, quiet microphone.
    const fn quiet() -> Self {
        Self {
            meeting_app: false,
            browser: false,
            mic: false,
        }
    }

    /// Slack has been open since this morning and nobody is talking.
    const fn app_only() -> Self {
        Self {
            meeting_app: true,
            browser: false,
            mic: false,
        }
    }

    /// A microphone held by something that cannot possibly be a call: dictation,
    /// a voice memo, a transcription utility. No browser, no call app.
    const fn mic_only() -> Self {
        Self {
            meeting_app: false,
            browser: false,
            mic: true,
        }
    }

    /// Someone is talking into a browser: Google Meet in a tab.
    const fn browser_and_mic() -> Self {
        Self {
            meeting_app: false,
            browser: true,
            mic: true,
        }
    }

    /// A browser open and nobody talking — which is to say, a Tuesday.
    const fn browser_only() -> Self {
        Self {
            meeting_app: false,
            browser: true,
            mic: false,
        }
    }

    /// The strong case: a call app running *and* the microphone live.
    const fn app_and_mic() -> Self {
        Self {
            meeting_app: true,
            browser: false,
            mic: true,
        }
    }
}

impl SignalRead {
    /// How many consecutive polls of *this* read it takes before Echo says
    /// something, or `None` when this read is never enough on its own however
    /// long it holds.
    const fn polls_to_nudge(self) -> Option<u32> {
        match (self.mic, self.meeting_app, self.browser) {
            // A live mic with an app that makes calls behind it: as fast as we
            // dare.
            (true, true, _) => Some(DEBOUNCE_POLLS),
            // A live mic and only a browser to explain it: could be a meeting in
            // a tab, could be dictation. Wait out dictation.
            (true, false, true) => Some(MIC_ONLY_DEBOUNCE_POLLS),
            // A live mic with nothing on the machine that could be carrying a
            // call. Whatever is listening, it is not a meeting.
            (true, false, false) => None,
            // An app and no mic is not a meeting, today or in an hour.
            (false, _, _) => None,
        }
    }

    /// Is there still something running that could be carrying the call?
    ///
    /// Only asked of an already-noticed meeting, to decide whether a microphone
    /// that just dropped out deserves the grace period. A browser counts here
    /// for the same reason it counts above: it may be where the meeting is.
    const fn could_be_carrying_a_call(self) -> bool {
        self.meeting_app || self.browser
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
            // Quiet microphone. A call app or a browser may well still be
            // running, and it buys nothing here: neither can start a meeting
            // (rule 1), and neither can keep the auto-stop safety net switched
            // off.
            self.mic_streak = 0;
            self.quiet_streak = self.quiet_streak.saturating_add(1);

            if self.detected_latched
                && !(read.could_be_carrying_a_call() && self.quiet_streak <= LATCH_GRACE_POLLS)
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
    /// True while a pre-warm started by this watcher is still running, so two
    /// detections cannot ask for the weights twice.
    prewarm_in_flight: Arc<AtomicBool>,
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
            prewarm_in_flight: Arc::new(AtomicBool::new(false)),
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
        Ok(self.observe()?.signals())
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
    fn observe(&self) -> Result<Observation, DetectError> {
        let mic = self.source.input_device_in_use()?;
        let apps = self.source.running_apps()?;
        Ok(Observation {
            call_apps: apps.call_apps,
            browsers: apps.browsers,
            mic,
        })
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

        let observation = match self.observe() {
            Ok(observation) => observation,
            Err(err) => {
                tracing::warn!(error = %err, "could not check for a meeting this round");
                return;
            }
        };
        let signals = observation.signals();
        let edge = self
            .debounce
            .lock()
            .unwrap()
            .step(capture_state, observation.read());

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
            // Noticing a meeting is the strongest hint there is that somebody is
            // about to press Start (mantra 1: "or on an explicit pre-warm").
            // Reading the weights takes seconds on a warm machine and minutes
            // the very first time, when the graphics compiler has work to do —
            // and that first time is exactly the meeting that loses its opening
            // minutes if we wait for the click.
            self.prewarm_speech_in_the_background(&app, speech_ready);

            if speech_ready && !self.episode_dismissed() {
                let detected_app = new_status
                    .signals
                    .iter()
                    .find_map(|signal| signal.app.clone());
                self.nudge_about_the_meeting(&app, detected_app).await;
            }
        }
    }

    /// Start loading speech understanding, without waiting for it.
    ///
    /// Deliberately fire-and-forget: this runs inside the 5-second poll, and a
    /// tick that blocks for a two-minute first load would stop the watcher
    /// noticing the meeting had ended. Failure is not reported anywhere — the
    /// person has not asked for anything yet, and pressing Start does its own
    /// pre-warm with its own message (see `session::SessionManager::start`).
    fn prewarm_speech_in_the_background(&self, app: &AppHandle, speech_ready: bool) {
        let Some(state) = app.try_state::<crate::AppState>() else {
            return;
        };
        if !should_prewarm(
            speech_ready,
            state.session.speech_loaded(),
            self.prewarm_in_flight.load(Ordering::SeqCst),
        ) {
            return;
        }
        // Claim the slot before spawning, so two ticks cannot both start one.
        if self.prewarm_in_flight.swap(true, Ordering::SeqCst) {
            return;
        }
        let in_flight = self.prewarm_in_flight.clone();
        let app = app.clone();
        tokio::spawn(async move {
            if let Some(state) = app.try_state::<crate::AppState>() {
                match state.session.prewarm_speech().await {
                    Ok(()) => tracing::info!(
                        "a meeting was detected, so speech understanding is ready early"
                    ),
                    // Nothing to say to anybody: nobody asked for this yet.
                    Err(error) => tracing::debug!(
                        %error,
                        "could not get speech understanding ready ahead of time"
                    ),
                }
            }
            in_flight.store(false, Ordering::SeqCst);
        });
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

/// Should noticing a meeting load speech understanding now?
///
/// Only when there is something to load (the one-time download has finished),
/// only when it is not already in memory — re-loading would throw away the very
/// thing we are trying to have ready — and only one at a time.
fn should_prewarm(speech_ready: bool, already_loaded: bool, in_flight: bool) -> bool {
    speech_ready && !already_loaded && !in_flight
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

    #[derive(Default)]
    struct MockSource {
        apps: Vec<&'static str>,
        browsers: Vec<&'static str>,
        input_in_use: bool,
    }

    impl SignalSource for MockSource {
        fn running_apps(&self) -> Result<RunningApps, DetectError> {
            Ok(RunningApps {
                call_apps: self.apps.iter().map(|s| s.to_string()).collect(),
                browsers: self.browsers.iter().map(|s| s.to_string()).collect(),
            })
        }

        fn input_device_in_use(&self) -> Result<bool, DetectError> {
            Ok(self.input_in_use)
        }
    }

    fn quiet_source() -> Box<dyn SignalSource> {
        Box::new(MockSource::default())
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
    fn the_apps_that_place_calls_are_recognised_by_name() {
        // The ones that were missing when people reported calls going unnoticed.
        assert_eq!(friendly_app_name("FaceTime"), Some("FaceTime"));
        assert_eq!(friendly_app_name("Skype"), Some("Skype"));
        assert_eq!(friendly_app_name("WhatsApp"), Some("WhatsApp"));
        assert_eq!(friendly_app_name("Signal"), Some("Signal"));
        assert_eq!(friendly_app_name("Cisco Webex Meetings"), Some("Webex"));
    }

    #[test]
    fn browsers_are_recognised_and_kept_apart_from_call_apps() {
        assert_eq!(friendly_browser_name("Google Chrome"), Some("Chrome"));
        assert_eq!(
            friendly_browser_name("Google Chrome Helper (Renderer)"),
            Some("Chrome")
        );
        assert_eq!(friendly_browser_name("Safari"), Some("Safari"));
        assert_eq!(friendly_browser_name("firefox"), Some("Firefox"));
        assert_eq!(friendly_browser_name("Arc"), Some("Arc"));
        assert_eq!(friendly_browser_name("Microsoft Edge"), Some("Edge"));

        // The lists answer different questions and must never bleed.
        assert_eq!(friendly_app_name("Google Chrome"), None);
        assert_eq!(friendly_browser_name("Slack"), None);
    }

    #[test]
    fn a_name_only_matches_on_whole_words() {
        // `searchd` runs on every Mac. Matched loosely it makes "a browser is
        // open" permanently true, which would quietly undo the whole fix.
        assert_eq!(friendly_browser_name("searchd"), None);
        assert_eq!(friendly_browser_name("chromedriver"), None);
        assert_eq!(friendly_browser_name("SafariBookmarksSyncAgent"), None);
        assert_eq!(friendly_app_name("MeetingBar"), None);
        assert_eq!(friendly_app_name("teamsviewerd"), None);
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
    fn an_app_or_a_browser_on_its_own_is_never_enough_however_long_it_runs() {
        assert_eq!(SignalRead::app_only().polls_to_nudge(), None);
        assert_eq!(SignalRead::browser_only().polls_to_nudge(), None);
        assert_eq!(SignalRead::quiet().polls_to_nudge(), None);
    }

    #[test]
    fn a_live_mic_is_faster_when_a_call_app_corroborates_it() {
        assert_eq!(
            SignalRead::app_and_mic().polls_to_nudge(),
            Some(DEBOUNCE_POLLS)
        );
        assert_eq!(
            SignalRead::browser_and_mic().polls_to_nudge(),
            Some(MIC_ONLY_DEBOUNCE_POLLS)
        );
    }

    #[test]
    fn a_mic_with_nothing_that_could_be_a_call_is_never_a_meeting() {
        // The field report, as a single assertion: a microphone held by
        // something that cannot make a call never becomes a nudge, at any
        // patience.
        assert_eq!(SignalRead::mic_only().polls_to_nudge(), None);
    }

    #[test]
    fn a_poll_reads_back_as_the_evidence_it_describes() {
        let observed = |apps: &[&str], browsers: &[&str], mic: bool| Observation {
            call_apps: apps.iter().map(|s| s.to_string()).collect(),
            browsers: browsers.iter().map(|s| s.to_string()).collect(),
            mic,
        };
        assert_eq!(observed(&[], &[], false).read(), SignalRead::quiet());
        assert_eq!(
            observed(&["Slack"], &[], false).read(),
            SignalRead::app_only()
        );
        assert_eq!(observed(&[], &[], true).read(), SignalRead::mic_only());
        assert_eq!(
            observed(&[], &["Chrome"], true).read(),
            SignalRead::browser_and_mic()
        );
        assert_eq!(
            observed(&["Zoom"], &[], true).read(),
            SignalRead::app_and_mic()
        );
    }

    #[test]
    fn what_the_person_is_shown_never_mentions_the_browser() {
        // A browser decides whether a mic *may* become a meeting; it is not
        // itself an observation about one, and the diagnostics list should not
        // read as though Echo thinks Chrome is a meeting.
        let observation = Observation {
            call_apps: Vec::new(),
            browsers: vec!["Chrome".into()],
            mic: true,
        };
        let signals = observation.signals();
        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0].source, DetectionSource::InputDeviceInUse);
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
                d.step(CaptureState::Idle, SignalRead::browser_and_mic()),
                StepEvent::None
            );
        }
        // Dictation ends.
        d.step(CaptureState::Idle, SignalRead::quiet());
        assert!(!d.is_detected());
        assert_eq!(d.episode(), 0);
    }

    #[test]
    fn a_mic_held_by_a_utility_never_fires_however_long_it_holds() {
        // Dictation, a voice memo, a transcription tool, a game: something is
        // listening, and there is nothing on the machine that could be carrying
        // a call. Echo used to nudge about this after half a minute.
        let mut d = DebounceTracker::default();
        for _ in 0..(60 * 60 / POLL_INTERVAL_SECS) {
            assert_eq!(
                d.step(CaptureState::Idle, SignalRead::mic_only()),
                StepEvent::None
            );
        }
        assert!(!d.is_detected(), "no browser, no call app, no meeting");
        assert_eq!(d.episode(), 0, "and not an episode either");
    }

    #[test]
    fn music_on_airpods_never_fires() {
        // Playback through a duplex headset, with a browser open as it always
        // is. The probe is what has to know that playing is not listening (see
        // the `macos` module); this is the state machine's half of the promise —
        // a browser and a quiet microphone are nothing at all.
        let mut d = DebounceTracker::default();
        for _ in 0..(60 * 60 / POLL_INTERVAL_SECS) {
            assert_eq!(
                d.step(CaptureState::Idle, SignalRead::browser_only()),
                StepEvent::None
            );
        }
        assert!(!d.is_detected());
        assert_eq!(d.episode(), 0);
    }

    #[test]
    fn a_browser_meeting_fires_within_thirty_five_seconds() {
        let mut d = DebounceTracker::default();
        let mut fired_at_secs = None;
        let mut fires = 0;
        for poll in 1..=(35 / POLL_INTERVAL_SECS) {
            if d.step(CaptureState::Idle, SignalRead::browser_and_mic()) == StepEvent::Detected {
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
        for _ in 0..MIC_ONLY_DEBOUNCE_POLLS {
            d.step(CaptureState::Idle, SignalRead::browser_and_mic());
        }
        assert!(d.is_detected());
        d.step(CaptureState::Idle, SignalRead::quiet());
        assert!(
            !d.is_detected(),
            "no app, no browser, no mic: there is nothing to claim a meeting from"
        );
    }

    #[test]
    fn a_browser_meeting_survives_the_mic_dropping_briefly_too() {
        // Same hysteresis as a Zoom call: a headset swap mid-tab-meeting is one
        // meeting, not two nudges. Bounded by the same grace period, so a
        // browser that is always open cannot hold the latch open forever.
        let mut d = DebounceTracker::default();
        for _ in 0..MIC_ONLY_DEBOUNCE_POLLS {
            d.step(CaptureState::Idle, SignalRead::browser_and_mic());
        }
        assert_eq!(d.episode(), 1);

        for _ in 0..LATCH_GRACE_POLLS {
            d.step(CaptureState::Idle, SignalRead::browser_only());
            assert!(d.is_detected(), "still the same meeting");
        }
        d.step(CaptureState::Idle, SignalRead::browser_only());
        assert!(
            !d.is_detected(),
            "grace is for a mic that comes back, not a licence to keep claiming \
             a meeting because a browser is open"
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

        // A real meeting, in the browser this time, and it still gets its own
        // nudge.
        for _ in 1..MIC_ONLY_DEBOUNCE_POLLS {
            assert_eq!(
                d.step(CaptureState::Idle, SignalRead::browser_and_mic()),
                StepEvent::None
            );
        }
        assert_eq!(
            d.step(CaptureState::Idle, SignalRead::browser_and_mic()),
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
                ..Default::default()
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
                ..Default::default()
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
                ..Default::default()
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
    async fn a_facetime_call_reaches_the_ui_on_the_fast_path() {
        // The name was missing from the list, so a FaceTime call used to wait
        // out the browser timer — or, with no browser open, never be noticed.
        let watcher = Watcher::with_source(
            true,
            Box::new(MockSource {
                apps: vec!["FaceTime"],
                input_in_use: true,
                ..Default::default()
            }),
        );
        watcher.tick().await;
        assert_eq!(watcher.status().state, DetectionState::Idle, "10s, not 5s");
        watcher.tick().await;
        assert_eq!(watcher.status().state, DetectionState::Detected);
        assert_eq!(watcher.episode(), 1);
    }

    #[tokio::test]
    async fn fifteen_seconds_of_dictation_never_reaches_the_ui() {
        let watcher = Watcher::with_source(
            true,
            Box::new(MockSource {
                browsers: vec!["Chrome"],
                input_in_use: true,
                ..Default::default()
            }),
        );
        for _ in 0..(15 / POLL_INTERVAL_SECS) {
            watcher.tick().await;
        }
        assert_eq!(watcher.status().state, DetectionState::Idle);

        // Keep going and the same mic, with a browser to explain it, does become
        // a meeting: this is the only way a browser call is ever noticed.
        for _ in 0..MIC_ONLY_DEBOUNCE_POLLS {
            watcher.tick().await;
        }
        assert_eq!(watcher.status().state, DetectionState::Detected);
    }

    #[tokio::test]
    async fn a_mic_held_with_no_browser_and_no_call_app_never_reaches_the_ui() {
        // A dictation utility, all afternoon. Nothing that could carry a call is
        // running, so there is nothing for Echo to offer to record.
        let watcher = Watcher::with_source(
            true,
            Box::new(MockSource {
                input_in_use: true,
                ..Default::default()
            }),
        );
        for _ in 0..(30 * 60 / POLL_INTERVAL_SECS) {
            watcher.tick().await;
        }
        assert_eq!(watcher.status().state, DetectionState::Idle);
        assert_eq!(watcher.episode(), 0, "no nudge, no badge, no episode");
    }

    #[tokio::test]
    async fn music_on_airpods_never_reaches_the_ui() {
        // What people reported: sound coming out of a duplex headset while a
        // browser sits open. The probe answers "nothing is listening" once it
        // asks about the input half only (see the `macos` module), and a browser
        // on its own is worth nothing here.
        let watcher = Watcher::with_source(
            true,
            Box::new(MockSource {
                browsers: vec!["Chrome", "Safari"],
                input_in_use: false,
                ..Default::default()
            }),
        );
        for _ in 0..(30 * 60 / POLL_INTERVAL_SECS) {
            watcher.tick().await;
        }
        assert_eq!(watcher.status().state, DetectionState::Idle);
        assert_eq!(watcher.episode(), 0);
        assert!(
            watcher.status().signals.is_empty(),
            "nothing was observed, so nothing is claimed"
        );
    }

    #[tokio::test]
    async fn a_mocked_input_device_signal_has_no_app_name() {
        let watcher = Watcher::with_source(
            true,
            Box::new(MockSource {
                input_in_use: true,
                ..Default::default()
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
                ..Default::default()
            }),
        );
        for _ in 0..10 {
            watcher.poll_once().await.unwrap();
        }
        // A diagnostic "check now" is not the watcher deciding a meeting
        // was detected; status is untouched without a real tick.
        assert_eq!(watcher.status().state, DetectionState::Idle);
    }

    #[test]
    fn noticing_a_meeting_gets_speech_understanding_ready_but_only_when_it_helps() {
        // The case worth having: assets on disk, nothing loaded, nothing running.
        assert!(should_prewarm(true, false, false));

        // Before the one-time download finishes there is nothing to load, and
        // recording is locked anyway.
        assert!(!should_prewarm(false, false, false));
        // Already in memory: loading again would drop the thing we want ready.
        assert!(!should_prewarm(true, true, false));
        // One at a time, however many meetings get noticed.
        assert!(!should_prewarm(true, false, true));
    }
}
