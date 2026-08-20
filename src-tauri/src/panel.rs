//! The little floating panel, and the recording tray icon's pulse.
//!
//! Two surfaces that live outside the main window, both of them costing nothing
//! when they are not on screen (mantra 1):
//!
//! * **The panel** — a small always-on-top window with no frame. It appears in
//!   two situations: top-right of the screen when Echo notices a meeting
//!   ("Meeting detected", how long ago, Start, ✕), and under the tray icon when
//!   someone left-clicks it while a recording is running (elapsed time, Stop).
//!   The window is created the first time it is needed and then reused; it is
//!   hidden, never destroyed, so the second appearance is instant.
//! * **The pulse** — while capture is recording, one 500ms timer swaps the tray
//!   icon between four frames. Paused holds a single frame with no timer at all,
//!   and stopping tears the timer down. There is no timer when nothing is being
//!   recorded.
//!
//! Everything that needs a real window is a thin wrapper around a pure
//! function: [`top_right_of`] and [`under_anchor`] decide where the panel goes,
//! [`motion_for`] decides whether the pulse should be running, and which
//! meeting the ✕ has been used on is [`crate::detect`]'s bookkeeping, not this
//! module's. Those are what the tests drive; no test opens a window.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tauri::{AppHandle, Emitter, Manager};

use crate::events::{self, PanelState};
use crate::types::CaptureState;

// ---------------------------------------------------------------------------
// Shape and placement
// ---------------------------------------------------------------------------

/// The panel window's label. Also the name it is granted permissions under in
/// `capabilities/panel.json` — a much shorter list than the main window's: it
/// hears events and calls Echo's own commands, and cannot move, show, hide or
/// focus a window itself.
pub const LABEL: &str = "panel";

/// Logical size, cut to the card the page draws rather than the other way
/// round: any spare height here is see-through window nobody asked for, and
/// (worse, when the window cannot be transparent) a bare white margin.
///
/// The card is one 36px row of icon-tile, two lines and a pill inside 14px of
/// padding — 66px tall — and both faces of the panel (detected and recording)
/// are built to that same row, so one size fits both. Width is what "Meeting
/// detected", the app's name, the clock and the Start pill need side by side.
/// The remainder is the few pixels the card's shadow falls into: 8 around the
/// top and sides, 12 below, since shadows fall downwards. Keep this in step
/// with `PanelShell`'s outer padding.
pub const WIDTH: f64 = 340.0;
pub const HEIGHT: f64 = 86.0;

/// Breathing room between the panel and the edge of the usable screen.
pub const MARGIN: f64 = 16.0;

/// A detection panel nobody touches takes itself away again. Long enough to
/// notice on the way back from the kitchen, short enough not to become
/// furniture.
pub const DETECTED_AUTO_HIDE_SECS: u64 = 45;

/// A rectangle with a top-left origin, in whatever unit the caller is working
/// in (the callers use physical pixels, the tests use round numbers).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Area {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl Area {
    pub fn new(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

/// Top-right of the usable screen, inset by `margin` on both sides.
pub fn top_right_of(work: Area, panel: (f64, f64), margin: f64) -> (f64, f64) {
    clamp_into(
        work,
        panel,
        margin,
        (work.x + work.width - panel.0 - margin, work.y + margin),
    )
}

/// Centred under `anchor` (the tray icon), pushed back inside the usable screen
/// if that would hang it off an edge — the menu bar item can sit hard against
/// the right of the screen, and half a panel off-screen is a bug people report.
pub fn under_anchor(anchor: Area, work: Area, panel: (f64, f64), margin: f64) -> (f64, f64) {
    let x = anchor.x + anchor.width / 2.0 - panel.0 / 2.0;
    let y = anchor.y + anchor.height + margin / 2.0;
    clamp_into(work, panel, margin, (x, y))
}

/// Keep the whole panel inside `work`, leaving `margin` where there is room. On
/// a screen too small to hold the panel with margins, the top-left corner of the
/// usable area wins: something visible beats something perfectly placed.
fn clamp_into(work: Area, panel: (f64, f64), margin: f64, wanted: (f64, f64)) -> (f64, f64) {
    let clamp = |value: f64, start: f64, extent: f64, size: f64| {
        let low = start + margin;
        let high = start + extent - size - margin;
        if high < low {
            start
        } else {
            value.clamp(low, high)
        }
    };
    (
        clamp(wanted.0, work.x, work.width, panel.0),
        clamp(wanted.1, work.y, work.height, panel.1),
    )
}

// ---------------------------------------------------------------------------
// The tray icon's pulse
// ---------------------------------------------------------------------------

/// How often the recording icon changes frame.
pub const FRAME_INTERVAL_MS: u64 = 500;

/// `icons/tray-recording-0.png` … `-3.png`.
pub const FRAME_COUNT: usize = 4;

/// Whether the tray icon should be moving, still, or left alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IconMotion {
    /// Nothing is being recorded: no timer, and the icon belongs to whoever
    /// last called [`crate::set_tray_state`].
    Off,
    /// Recording: swap frames every [`FRAME_INTERVAL_MS`].
    Running,
    /// Paused: one frame, held. A paused recording is not a moving thing, and a
    /// timer that exists to change nothing is a timer that should not exist.
    Held,
}

/// What capture being in `state` means for the icon.
pub fn motion_for(state: CaptureState) -> IconMotion {
    match state {
        // Degraded is still recording, just with less than we wanted.
        CaptureState::Recording | CaptureState::Degraded => IconMotion::Running,
        CaptureState::Paused => IconMotion::Held,
        CaptureState::Idle
        | CaptureState::Starting
        | CaptureState::Stopping
        | CaptureState::Stopped
        | CaptureState::Failed
        | CaptureState::Recovering => IconMotion::Off,
    }
}

/// Somewhere to put a frame. The app paints the tray; tests count calls.
pub trait FramePainter: Send + Sync + 'static {
    fn paint(&self, frame: usize);
}

/// The real painter: the same atomic icon-and-template swap
/// [`crate::set_tray_state`] does, because setting the icon alone drops the
/// template flag and a non-template icon is a black blob in a dark menu bar.
pub struct TrayPainter {
    app: AppHandle,
}

impl TrayPainter {
    pub fn new(app: AppHandle) -> Self {
        Self { app }
    }
}

impl FramePainter for TrayPainter {
    fn paint(&self, frame: usize) {
        crate::set_tray_frame(&self.app, frame);
    }
}

/// The moving part of the pulse, shared with the timer task: which frame is up,
/// and how to get to the next one.
///
/// [`Pulse::advance`] is the whole animation. The timer calls it every
/// [`FRAME_INTERVAL_MS`]; the tests call it directly, which is why none of them
/// needs a clock.
#[derive(Clone, Default)]
pub struct Pulse {
    frame: Arc<AtomicUsize>,
}

impl Pulse {
    pub fn frame(&self) -> usize {
        self.frame.load(Ordering::Relaxed)
    }

    /// Move to the next frame and put it on screen. Wraps around.
    pub fn advance(&self, painter: &dyn FramePainter) -> usize {
        let next = (self.frame() + 1) % FRAME_COUNT;
        self.frame.store(next, Ordering::Relaxed);
        painter.paint(next);
        next
    }

    fn reset(&self) {
        self.frame.store(0, Ordering::Relaxed);
    }
}

struct AnimationInner {
    task: Option<tokio::task::JoinHandle<()>>,
    motion: IconMotion,
}

/// The pulse's whole existence: at most one timer, and which frame is showing.
///
/// [`TrayAnimation::apply`] has to be called from inside a Tokio context — it
/// spawns onto the *current* runtime deliberately, so tests can pause time.
/// `lib.rs` hops onto Tauri's runtime before calling it.
pub struct TrayAnimation {
    inner: Mutex<AnimationInner>,
    pulse: Pulse,
}

impl Default for TrayAnimation {
    fn default() -> Self {
        Self::new()
    }
}

impl TrayAnimation {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(AnimationInner {
                task: None,
                motion: IconMotion::Off,
            }),
            pulse: Pulse::default(),
        }
    }

    /// The frame counter the timer drives. Handed out so a test can step the
    /// animation by hand instead of waiting for wall-clock time.
    pub fn pulse(&self) -> Pulse {
        self.pulse.clone()
    }

    /// Bring the timer in line with what capture is doing. Idempotent: asking
    /// for the motion that is already running changes nothing, so a stream of
    /// capture-state events does not restart the animation every few seconds.
    pub fn apply(&self, motion: IconMotion, painter: Arc<dyn FramePainter>) {
        let mut inner = self.inner.lock().unwrap();
        if inner.motion == motion {
            return;
        }
        // Whatever was running, stop it: every transition either needs a new
        // timer or none at all.
        if let Some(task) = inner.task.take() {
            task.abort();
        }
        inner.motion = motion;

        match motion {
            IconMotion::Off => self.pulse.reset(),
            IconMotion::Held => {
                // The strongest frame, standing still.
                self.pulse.reset();
                painter.paint(0);
            }
            IconMotion::Running => {
                self.pulse.reset();
                painter.paint(0);
                let pulse = self.pulse.clone();
                inner.task = Some(tokio::spawn(async move {
                    let mut ticker =
                        tokio::time::interval(Duration::from_millis(FRAME_INTERVAL_MS));
                    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                    // The first tick completes immediately and frame 0 is
                    // already on screen.
                    ticker.tick().await;
                    loop {
                        ticker.tick().await;
                        pulse.advance(painter.as_ref());
                    }
                }));
            }
        }
    }

    /// Tear the timer down. Called at stop, and on the way out.
    pub fn stop(&self) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(task) = inner.task.take() {
            task.abort();
        }
        inner.motion = IconMotion::Off;
        self.pulse.reset();
    }

    /// Is a timer alive right now? The lifecycle promise, in one call.
    pub fn is_running(&self) -> bool {
        self.inner.lock().unwrap().task.is_some()
    }

    pub fn motion(&self) -> IconMotion {
        self.inner.lock().unwrap().motion
    }

    pub fn frame(&self) -> usize {
        self.pulse.frame()
    }
}

// ---------------------------------------------------------------------------
// The panel's own state
// ---------------------------------------------------------------------------

#[derive(Default)]
struct PanelInner {
    /// What the panel is showing, so a webview that has only just finished
    /// loading can ask for it instead of missing the event.
    showing: Option<PanelState>,
    /// Bumped on every show and hide. A pending auto-hide carrying an older
    /// number does nothing, so yesterday's timer cannot close today's panel.
    generation: u64,
    auto_hide: Option<tokio::task::JoinHandle<()>>,
    /// Capture state, mirrored from the capture-state event so the tray click
    /// handler can decide what to do without awaiting anything.
    capture: CaptureState,
    /// When the current recording started, in epoch milliseconds.
    started_at_ms: Option<i64>,
}

/// The panel's runtime state, managed by Tauri. Holds no window: the window is
/// looked up by label when it is needed, and created on the first show.
#[derive(Default)]
pub struct Panel {
    inner: Mutex<PanelInner>,
}

impl Panel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Remember what capture is doing. Called from the capture-state listener.
    pub fn note_capture(&self, state: CaptureState, started_at_ms: Option<i64>) {
        let mut inner = self.inner.lock().unwrap();
        inner.capture = state;
        if started_at_ms.is_some() {
            inner.started_at_ms = started_at_ms;
        }
        if motion_for(state) == IconMotion::Off {
            inner.started_at_ms = None;
        }
    }

    pub fn capture(&self) -> CaptureState {
        self.inner.lock().unwrap().capture
    }

    pub fn started_at_ms(&self) -> Option<i64> {
        self.inner.lock().unwrap().started_at_ms
    }

    /// What the panel is showing, for a webview that just loaded.
    pub fn showing(&self) -> Option<PanelState> {
        self.inner.lock().unwrap().showing.clone()
    }

    pub fn is_showing_recording(&self) -> bool {
        matches!(
            self.inner.lock().unwrap().showing,
            Some(PanelState::Recording { .. })
        )
    }

    fn begin_show(&self, state: PanelState) -> u64 {
        let mut inner = self.inner.lock().unwrap();
        if let Some(task) = inner.auto_hide.take() {
            task.abort();
        }
        inner.generation = inner.generation.wrapping_add(1);
        inner.showing = Some(state);
        inner.generation
    }

    fn end_show(&self) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(task) = inner.auto_hide.take() {
            task.abort();
        }
        inner.generation = inner.generation.wrapping_add(1);
        inner.showing = None;
    }

    fn generation(&self) -> u64 {
        self.inner.lock().unwrap().generation
    }

    /// Fold a new paused-ness into the recording panel, if that is what is on
    /// screen. Returns the state to send when it actually changed, so a
    /// capture-state event that says nothing new sends nothing.
    fn update_recording(&self, now_paused: bool) -> Option<PanelState> {
        let mut inner = self.inner.lock().unwrap();
        let Some(PanelState::Recording {
            started_at_ms,
            paused,
        }) = inner.showing
        else {
            return None;
        };
        if paused == now_paused {
            return None;
        }
        let updated = PanelState::Recording {
            started_at_ms,
            paused: now_paused,
        };
        inner.showing = Some(updated.clone());
        Some(updated)
    }

    fn set_auto_hide(&self, task: tokio::task::JoinHandle<()>) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(previous) = inner.auto_hide.replace(task) {
            previous.abort();
        }
    }
}

// ---------------------------------------------------------------------------
// Showing and hiding, for real
// ---------------------------------------------------------------------------

/// Show the "meeting detected" panel at the top-right of the screen.
///
/// Returns whether the panel actually made it onto the screen: the caller uses
/// that to decide between the panel and the OS notification, so exactly one
/// nudge happens per detection episode.
///
/// Must not be called from the main thread — it waits for the window work to
/// run there.
pub async fn show_detected(app: &AppHandle, detected_at_ms: i64, app_name: Option<String>) -> bool {
    let state = PanelState::Detected {
        detected_at_ms,
        app_name,
    };
    let (tx, rx) = tokio::sync::oneshot::channel();
    let handle = app.clone();
    let shown = state.clone();
    if app
        .run_on_main_thread(move || {
            let _ = tx.send(place_and_show(&handle, shown, None));
        })
        .is_err()
    {
        return false;
    }
    // The window work runs on the main thread. If that thread is busy enough to
    // miss this, fall back to the notification rather than wedge the poll loop
    // this is called from.
    let shown = tokio::time::timeout(Duration::from_secs(5), rx)
        .await
        .unwrap_or(Ok(false))
        .unwrap_or(false);
    if !shown {
        return false;
    }

    // Nobody has to close it: a stale "meeting detected" left sitting there
    // would just be litter (see DETECTED_AUTO_HIDE_SECS).
    if let Some(panel) = app.try_state::<Panel>() {
        let generation = panel.generation();
        let handle = app.clone();
        if tokio::runtime::Handle::try_current().is_ok() {
            panel.set_auto_hide(tokio::spawn(async move {
                tokio::time::sleep(Duration::from_secs(DETECTED_AUTO_HIDE_SECS)).await;
                if let Some(panel) = handle.try_state::<Panel>() {
                    if panel.generation() != generation {
                        return;
                    }
                }
                hide(&handle);
            }));
        }
    }
    true
}

/// Left-clicking the tray while recording: show the panel under the icon, or
/// take it away again if it is already there. `anchor` is the tray icon's own
/// rectangle, in physical pixels.
pub fn toggle_recording(app: &AppHandle, anchor: Option<Area>, started_at_ms: i64, paused: bool) {
    if let Some(panel) = app.try_state::<Panel>() {
        if panel.is_showing_recording() {
            hide(app);
            return;
        }
    }
    let handle = app.clone();
    let _ = app.run_on_main_thread(move || {
        place_and_show(
            &handle,
            PanelState::Recording {
                started_at_ms,
                paused,
            },
            anchor,
        );
    });
}

/// The recording the panel is showing just paused or resumed. Keep what it says
/// true, without moving or re-showing anything. Does nothing when the panel is
/// showing something else, or nothing at all.
pub fn refresh_recording(app: &AppHandle, paused: bool) {
    let Some(panel) = app.try_state::<Panel>() else {
        return;
    };
    if let Some(state) = panel.update_recording(paused) {
        let _ = app.emit_to(LABEL, events::PANEL_STATE, Some(state));
    }
}

/// Take the panel off the screen. The window stays alive, hidden, for next time.
pub fn hide(app: &AppHandle) {
    if let Some(panel) = app.try_state::<Panel>() {
        panel.end_show();
    }
    // Tell the page it is done before the window goes: it drops the card and
    // the clock that ticks with it, so nothing keeps counting off screen.
    let _ = app.emit_to(LABEL, events::PANEL_STATE, None::<PanelState>);
    let handle = app.clone();
    let _ = app.run_on_main_thread(move || {
        if let Some(window) = handle.get_webview_window(LABEL) {
            let _ = window.hide();
        }
    });
}

/// Position, fill and show the panel. Main thread only.
fn place_and_show(app: &AppHandle, state: PanelState, anchor: Option<Area>) -> bool {
    let window = match ensure_window(app) {
        Ok(window) => window,
        Err(error) => {
            tracing::warn!(%error, "could not put the meeting panel on screen");
            return false;
        }
    };

    if let Some(panel) = app.try_state::<Panel>() {
        panel.begin_show(state.clone());
    }

    // Everything below is in physical pixels, scaled by the screen the panel is
    // about to appear on rather than the one it was last on — a laptop plugged
    // into an external display has two different answers.
    let (work, scale) = work_area(app, &window, anchor);
    let size = (WIDTH * scale, HEIGHT * scale);
    let margin = MARGIN * scale;
    let (x, y) = match anchor {
        Some(anchor) => under_anchor(anchor, work, size, margin),
        None => top_right_of(work, size, margin),
    };
    let _ = window.set_position(tauri::PhysicalPosition::new(x, y));

    // The webview may still be loading the first time round, in which case this
    // event lands nowhere — that is what `get_panel_state` is for.
    let _ = app.emit_to(LABEL, events::PANEL_STATE, Some(state));

    if let Err(error) = window.show() {
        tracing::warn!(%error, "could not put the meeting panel on screen");
        return false;
    }
    true
}

/// Which screen the panel belongs on — the one holding the tray icon when we
/// know it, otherwise the one the main window is on, otherwise the primary one —
/// as its usable area in physical pixels, plus that screen's scale factor.
fn work_area(app: &AppHandle, window: &tauri::WebviewWindow, anchor: Option<Area>) -> (Area, f64) {
    let monitor = anchor
        .and_then(|a| {
            app.monitor_from_point(a.x + a.width / 2.0, a.y + a.height / 2.0)
                .ok()
                .flatten()
        })
        .or_else(|| {
            app.get_webview_window("main")
                .and_then(|main| main.current_monitor().ok().flatten())
        })
        .or_else(|| window.current_monitor().ok().flatten())
        .or_else(|| app.primary_monitor().ok().flatten());

    match monitor {
        Some(monitor) => {
            let area = monitor.work_area();
            (
                Area::new(
                    area.position.x as f64,
                    area.position.y as f64,
                    area.size.width as f64,
                    area.size.height as f64,
                ),
                monitor.scale_factor(),
            )
        }
        // No monitor to ask. A sane guess beats refusing to show anything.
        None => (
            Area::new(0.0, 0.0, 1440.0, 900.0),
            window.scale_factor().unwrap_or(1.0),
        ),
    }
}

/// The panel window, created the first time it is wanted and reused after that.
/// Main thread only.
fn ensure_window(app: &AppHandle) -> tauri::Result<tauri::WebviewWindow> {
    if let Some(window) = app.get_webview_window(LABEL) {
        return Ok(window);
    }

    // Same bundle, different route: the frontend branches on `?window=panel`.
    #[allow(unused_mut)]
    let mut builder = tauri::WebviewWindowBuilder::new(
        app,
        LABEL,
        tauri::WebviewUrl::App("index.html?window=panel".into()),
    )
    .title("Echo")
    .inner_size(WIDTH, HEIGHT)
    .resizable(false)
    .decorations(false)
    .always_on_top(true)
    .skip_taskbar(true)
    .visible_on_all_workspaces(true)
    .maximizable(false)
    .minimizable(false)
    .closable(false)
    .shadow(true)
    // Never take the keyboard away from whatever the person is doing; the
    // panel is something you glance at and click.
    .focused(false)
    .visible(false);

    // Rounded corners need a see-through window. On macOS that sits behind a
    // private API Tauri gates at compile time, and a crate cannot read its
    // dependency's feature flags — switching it on means adding
    // `macos-private-api = ["tauri/macos-private-api"]` to Cargo.toml,
    // `"macOSPrivateApi": true` to tauri.conf.json, and this platform to the
    // line below. Until then the panel is a plain rectangle on macOS: it works,
    // it is just squarer.
    #[cfg(not(target_os = "macos"))]
    {
        builder = builder.transparent(true);
    }
    #[cfg(target_os = "macos")]
    {
        // The panel's own page makes itself see-through, expecting the window
        // behind it to be too. Until it can be, paint that window Echo's own
        // white rather than leaving it whatever grey the platform picks.
        builder = builder.background_color(tauri::window::Color(255, 255, 255, 255));
    }

    // A click on a window that is not focused should press the button, not just
    // raise the window: Start has to work on the first click.
    #[cfg(target_os = "macos")]
    {
        builder = builder.accept_first_mouse(true);
    }

    // The very first show races the webview booting up, and a panel that missed
    // its one event would sit there empty. Say it again as the page loads. The
    // panel can also ask outright (`get_panel_state`), which is the belt to this
    // pair of braces.
    builder = builder.on_page_load(|window, _| {
        let app = window.app_handle();
        if let Some(showing) = app.try_state::<Panel>().and_then(|panel| panel.showing()) {
            let _ = app.emit_to(LABEL, events::PANEL_STATE, Some(showing));
        }
    });

    let window = builder.build()?;

    // Nothing may close this window: it is reused, and a closed one would have
    // to be rebuilt on the next detection.
    let handle = app.clone();
    window.on_window_event(move |event| {
        if let tauri::WindowEvent::CloseRequested { api, .. } = event {
            api.prevent_close();
            hide(&handle);
        }
    });

    Ok(window)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PANEL: (f64, f64) = (WIDTH, HEIGHT);
    const LAPTOP: Area = Area {
        x: 0.0,
        y: 25.0,
        width: 1440.0,
        height: 875.0,
    };

    // -- placement --------------------------------------------------------

    #[test]
    fn the_detection_panel_sits_inside_the_top_right_corner() {
        let (x, y) = top_right_of(LAPTOP, PANEL, MARGIN);
        assert_eq!(x, 1440.0 - WIDTH - MARGIN);
        assert_eq!(y, 25.0 + MARGIN);
    }

    #[test]
    fn the_detection_panel_respects_a_second_screen_to_the_left() {
        let left = Area::new(-1920.0, 0.0, 1920.0, 1080.0);
        let (x, y) = top_right_of(left, PANEL, MARGIN);
        assert_eq!(x, -1920.0 + 1920.0 - WIDTH - MARGIN);
        assert_eq!(y, MARGIN);
    }

    #[test]
    fn the_tray_panel_is_centred_under_the_icon() {
        let icon = Area::new(700.0, 0.0, 24.0, 24.0);
        let (x, y) = under_anchor(icon, LAPTOP, PANEL, MARGIN);
        assert_eq!(x, 700.0 + 12.0 - WIDTH / 2.0, "centred on the icon");
        // The icon lives in the menu bar, which is not usable screen, so the
        // top of the work area is what actually decides the height — and it
        // puts the panel exactly as far down as the detection one.
        assert_eq!(y, LAPTOP.y + MARGIN);
    }

    #[test]
    fn a_tray_panel_below_the_work_area_top_keeps_its_own_gap() {
        // A system where the menu bar is not carved out of the work area: the
        // panel then really does hang off the bottom of the icon.
        let full = Area::new(0.0, 0.0, 1440.0, 900.0);
        let icon = Area::new(700.0, 4.0, 24.0, 24.0);
        let (_, y) = under_anchor(icon, full, PANEL, MARGIN);
        assert_eq!(y, 4.0 + 24.0 + MARGIN / 2.0);
    }

    #[test]
    fn a_tray_icon_at_the_edge_pulls_the_panel_back_on_screen() {
        // The menu bar item nearest the clock: centring would hang most of the
        // panel off the right of the screen.
        let icon = Area::new(1420.0, 0.0, 20.0, 24.0);
        let (x, _) = under_anchor(icon, LAPTOP, PANEL, MARGIN);
        assert_eq!(x, 1440.0 - WIDTH - MARGIN);
        assert!(x + WIDTH + MARGIN <= LAPTOP.x + LAPTOP.width);
    }

    #[test]
    fn a_screen_too_small_for_the_panel_still_gets_one() {
        let tiny = Area::new(0.0, 0.0, 200.0, 100.0);
        let (x, y) = top_right_of(tiny, PANEL, MARGIN);
        assert_eq!((x, y), (0.0, 0.0), "visible beats perfectly placed");
    }

    // -- pulse lifecycle ---------------------------------------------------

    /// Stands in for the menu bar: remembers every frame it was asked to show.
    #[derive(Default)]
    struct Spy {
        frames: Mutex<Vec<usize>>,
    }

    impl Spy {
        fn frames(&self) -> Vec<usize> {
            self.frames.lock().unwrap().clone()
        }
    }

    impl FramePainter for Spy {
        fn paint(&self, frame: usize) {
            self.frames.lock().unwrap().push(frame);
        }
    }

    #[test]
    fn capture_state_decides_whether_the_icon_moves() {
        assert_eq!(motion_for(CaptureState::Recording), IconMotion::Running);
        assert_eq!(
            motion_for(CaptureState::Degraded),
            IconMotion::Running,
            "degraded is still recording"
        );
        assert_eq!(motion_for(CaptureState::Paused), IconMotion::Held);
        for quiet in [
            CaptureState::Idle,
            CaptureState::Starting,
            CaptureState::Stopping,
            CaptureState::Stopped,
            CaptureState::Failed,
            CaptureState::Recovering,
        ] {
            assert_eq!(motion_for(quiet), IconMotion::Off, "{quiet:?}");
        }
    }

    #[test]
    fn the_frames_run_in_order_and_wrap_around() {
        // The pulse itself, stepped by hand: what the timer does every 500ms,
        // minus the waiting.
        let pulse = Pulse::default();
        let spy = Spy::default();
        assert_eq!(pulse.frame(), 0, "the first frame is the resting one");
        for _ in 0..FRAME_COUNT + 1 {
            pulse.advance(&spy);
        }
        assert_eq!(spy.frames(), vec![1, 2, 3, 0, 1]);
        assert_eq!(pulse.frame(), 1);
    }

    #[tokio::test]
    async fn there_is_no_timer_until_a_recording_starts() {
        let animation = TrayAnimation::new();
        let spy = Arc::new(Spy::default());

        animation.apply(IconMotion::Off, spy.clone());
        assert!(!animation.is_running());
        assert!(spy.frames().is_empty(), "an idle tray is not repainted");
    }

    #[tokio::test]
    async fn recording_starts_one_timer_and_shows_the_first_frame_at_once() {
        let animation = TrayAnimation::new();
        let spy = Arc::new(Spy::default());

        animation.apply(IconMotion::Running, spy.clone());
        assert!(animation.is_running());
        assert_eq!(animation.motion(), IconMotion::Running);
        assert_eq!(spy.frames(), vec![0], "no waiting for the first frame");
    }

    #[tokio::test]
    async fn asking_for_the_motion_already_running_does_not_restart_it() {
        let animation = TrayAnimation::new();
        let spy = Arc::new(Spy::default());

        animation.apply(IconMotion::Running, spy.clone());
        animation.pulse().advance(spy.as_ref()); // the timer's first step
        animation.apply(IconMotion::Running, spy.clone());

        assert_eq!(
            spy.frames(),
            vec![0, 1],
            "a repeated capture-state event must not rewind the pulse"
        );
        assert_eq!(animation.frame(), 1);
    }

    #[tokio::test]
    async fn pausing_holds_a_frame_without_keeping_a_timer() {
        let animation = TrayAnimation::new();
        let spy = Arc::new(Spy::default());

        animation.apply(IconMotion::Running, spy.clone());
        animation.pulse().advance(spy.as_ref());

        animation.apply(IconMotion::Held, spy.clone());
        assert!(!animation.is_running(), "paused keeps no timer (mantra 1)");
        assert_eq!(animation.motion(), IconMotion::Held);
        assert_eq!(
            spy.frames(),
            vec![0, 1, 0],
            "the held frame goes up once and then nothing"
        );
    }

    #[tokio::test]
    async fn resuming_after_a_pause_starts_moving_again() {
        let animation = TrayAnimation::new();
        let spy = Arc::new(Spy::default());

        animation.apply(IconMotion::Running, spy.clone());
        animation.apply(IconMotion::Held, spy.clone());
        assert!(!animation.is_running());
        animation.apply(IconMotion::Running, spy.clone());
        assert!(animation.is_running());
    }

    #[tokio::test]
    async fn stopping_tears_the_timer_down() {
        let animation = TrayAnimation::new();
        let spy = Arc::new(Spy::default());

        animation.apply(IconMotion::Running, spy.clone());
        animation.apply(IconMotion::Off, spy.clone());
        assert!(!animation.is_running());
        assert_eq!(animation.motion(), IconMotion::Off);
        assert_eq!(animation.frame(), 0, "back to the resting frame");
    }

    #[tokio::test]
    async fn stop_is_safe_to_call_when_nothing_is_running() {
        let animation = TrayAnimation::new();
        animation.stop();
        animation.stop();
        assert!(!animation.is_running());
    }

    #[tokio::test]
    async fn the_timer_really_does_move_the_icon_on_its_own() {
        // The one test that waits: everything above drives the pulse by hand,
        // so this is what proves the timer is actually wired to it. One frame
        // interval plus a generous margin.
        let animation = TrayAnimation::new();
        let spy = Arc::new(Spy::default());

        animation.apply(IconMotion::Running, spy.clone());
        tokio::time::sleep(Duration::from_millis(FRAME_INTERVAL_MS + 250)).await;

        let frames = spy.frames();
        assert!(frames.len() >= 2, "the timer never fired: {frames:?}");
        assert_eq!(frames[0], 0);
        assert_eq!(frames[1], 1);

        animation.stop();
        let after_stop = spy.frames();
        tokio::time::sleep(Duration::from_millis(FRAME_INTERVAL_MS + 250)).await;
        assert_eq!(
            spy.frames(),
            after_stop,
            "nothing paints the tray once the recording is over"
        );
    }

    // -- panel bookkeeping -------------------------------------------------

    #[test]
    fn the_panel_mirrors_capture_so_the_tray_click_never_waits() {
        let panel = Panel::new();
        assert_eq!(panel.capture(), CaptureState::Idle);
        assert_eq!(panel.started_at_ms(), None);

        panel.note_capture(CaptureState::Recording, Some(1_700_000_000_000));
        assert_eq!(panel.capture(), CaptureState::Recording);
        assert_eq!(panel.started_at_ms(), Some(1_700_000_000_000));

        // A paused recording still has a start time to count from.
        panel.note_capture(CaptureState::Paused, None);
        assert_eq!(panel.started_at_ms(), Some(1_700_000_000_000));

        panel.note_capture(CaptureState::Stopped, None);
        assert_eq!(panel.started_at_ms(), None, "stopping forgets the clock");
    }

    #[test]
    fn showing_and_hiding_move_the_generation_so_a_stale_timer_is_inert() {
        let panel = Panel::new();
        assert!(panel.showing().is_none());

        let first = panel.begin_show(PanelState::Detected {
            detected_at_ms: 1,
            app_name: Some("Zoom".into()),
        });
        assert!(panel.showing().is_some());
        assert!(!panel.is_showing_recording());

        let second = panel.begin_show(PanelState::Recording {
            started_at_ms: 2,
            paused: false,
        });
        assert_ne!(first, second, "the first panel's auto-hide is now stale");
        assert!(panel.is_showing_recording());

        panel.end_show();
        assert!(panel.showing().is_none());
        assert_ne!(panel.generation(), second);
    }

    #[test]
    fn pausing_changes_what_the_recording_panel_says_exactly_once() {
        let panel = Panel::new();
        assert!(
            panel.update_recording(true).is_none(),
            "nothing on screen, nothing to say"
        );

        panel.begin_show(PanelState::Detected {
            detected_at_ms: 1,
            app_name: None,
        });
        assert!(
            panel.update_recording(true).is_none(),
            "the detection panel is not about a recording"
        );

        panel.begin_show(PanelState::Recording {
            started_at_ms: 2,
            paused: false,
        });
        assert!(
            panel.update_recording(false).is_none(),
            "a capture event that says nothing new sends nothing"
        );
        assert_eq!(
            panel.update_recording(true),
            Some(PanelState::Recording {
                started_at_ms: 2,
                paused: true,
            })
        );
        assert!(panel.update_recording(true).is_none(), "already said");
        assert_eq!(
            panel.update_recording(false),
            Some(PanelState::Recording {
                started_at_ms: 2,
                paused: false,
            }),
            "resuming is the same news the other way round"
        );
    }
}
