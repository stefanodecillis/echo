//! Echo: local-first meeting transcription and recaps.
//!
//! This file wires the app together and does nothing else. Startup order
//! matters, so it is spelled out:
//!
//! 1. resolve paths and create directories
//! 2. start the redacted rotating log
//! 3. open the database and apply migrations
//! 4. requeue jobs a crash left marked running
//! 5. build shared state, the tray and the window
//!
//! Nothing here loads a speech engine, opens an audio device or starts a
//! background worker. An idle Echo is a window, a tray icon and a five-second
//! poll (mantra 1).

pub mod asr;
pub mod audio;
pub mod commands;
pub mod db;
pub mod detect;
pub mod diarize;
pub mod events;
pub mod export;
pub mod logging;
pub mod panel;
pub mod paths;
pub mod secrets;
pub mod session;
pub mod settings;
pub mod summarize;
pub mod types;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tauri::image::Image;
use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Emitter, Listener, Manager, WindowEvent};

use commands::AppState;
use events::PanelState;
use types::{CaptureStatus, TrayAction, TrayState};

/// Tray menu item ids. Also the ids the tray click handler matches on.
mod tray_ids {
    pub const START: &str = "start";
    pub const STOP: &str = "stop";
    pub const OPEN: &str = "open";
    pub const SNOOZE: &str = "snooze";
    pub const QUIT: &str = "quit";
}

/// Three tray icons: waiting, a meeting seems to be happening, recording.
const TRAY_ICON_IDLE: &[u8] = include_bytes!("../icons/tray-idle.png");
const TRAY_ICON_DETECTED: &[u8] = include_bytes!("../icons/tray-detected.png");
const TRAY_ICON_RECORDING: &[u8] = include_bytes!("../icons/tray-recording.png");

/// The recording icon's pulse: the same mark with the dot breathing. Only ever
/// on screen while something is being recorded (see [`panel::TrayAnimation`]).
const TRAY_ICON_RECORDING_FRAMES: [&[u8]; panel::FRAME_COUNT] = [
    include_bytes!("../icons/tray-recording-0.png"),
    include_bytes!("../icons/tray-recording-1.png"),
    include_bytes!("../icons/tray-recording-2.png"),
    include_bytes!("../icons/tray-recording-3.png"),
];

/// Flags the window handlers need synchronously, so they never await.
pub struct UiFlags {
    /// Closing the window hides it instead of quitting.
    pub close_to_tray: AtomicBool,
    /// Set on the way out, so the last close really closes.
    pub quitting: AtomicBool,
}

impl Default for UiFlags {
    fn default() -> Self {
        Self {
            close_to_tray: AtomicBool::new(true),
            quitting: AtomicBool::new(false),
        }
    }
}

/// Swap the tray icon. Called by the session and detection layers.
pub fn set_tray_state(app: &AppHandle, state: TrayState) {
    // The pulse only exists while the tray says "recording". Standing it down
    // here as well as from the capture-state listener closes the one race worth
    // caring about: a frame landing after the icon went back to idle would
    // leave the menu bar claiming a recording that had finished.
    if state != TrayState::Recording {
        if let Some(animation) = app.try_state::<panel::TrayAnimation>() {
            animation.stop();
        }
    }
    let bytes = match state {
        TrayState::Idle => TRAY_ICON_IDLE,
        TrayState::Detected => TRAY_ICON_DETECTED,
        TrayState::Recording => TRAY_ICON_RECORDING,
    };
    paint_tray(app, bytes);
    let _ = app.emit(events::TRAY_STATE, events::TrayStatePayload { state });
}

/// Paint one frame of the recording pulse. Deliberately quiet: the tray *state*
/// has not changed, so no `TRAY_STATE` event goes out — twice a second of "still
/// recording" would be noise on the bus.
pub fn set_tray_frame(app: &AppHandle, frame: usize) {
    paint_tray(app, TRAY_ICON_RECORDING_FRAMES[frame % panel::FRAME_COUNT]);
}

fn paint_tray(app: &AppHandle, bytes: &[u8]) {
    if let Some(tray) = app.tray_by_id("echo-tray") {
        if let Ok(image) = Image::from_bytes(bytes) {
            // Icon and template flag in one go. `set_icon` on its own drops the
            // flag — and a non-template icon is a black blob in a dark menu bar
            // — but setting them one after the other draws the icon twice,
            // which the twice-a-second pulse would show as a flicker. On Linux
            // and Windows this is `set_icon` and nothing else.
            let _ = tray.set_icon_with_as_template(Some(image), true);
        }
    }
}

/// Show the window and focus it. The window is always a first-class way back
/// in, because Linux tray support is not universal (review finding 8).
pub fn focus_main_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

/// Tell the UI where to go. Used by the tray, notification clicks and a second
/// launch of the app.
pub fn navigate(app: &AppHandle, payload: events::NavigatePayload) {
    focus_main_window(app);
    let _ = app.emit(events::NAVIGATE, payload);
}

/// The one way out. Closes a live recording, parks the work queue and gives the
/// speech engine's memory back before the process goes away.
///
/// Blocking here is deliberate: a recording that is not closed properly leaves a
/// meeting to recover on next launch, and quitting is the one moment where
/// waiting a beat is better than being fast.
pub fn quit(app: &AppHandle) {
    if let Some(flags) = app.try_state::<UiFlags>() {
        flags.quitting.store(true, Ordering::SeqCst);
    }
    if let Some(animation) = app.try_state::<panel::TrayAnimation>() {
        animation.stop();
    }
    if let Some(state) = app.try_state::<AppState>() {
        tauri::async_runtime::block_on(state.session.shutdown());
    }
    app.exit(0);
}

fn build_tray(app: &AppHandle) -> tauri::Result<()> {
    // Wording, not commands: "Start recording", not "Start".
    let start = MenuItem::with_id(app, tray_ids::START, "Start recording", true, None::<&str>)?;
    let stop = MenuItem::with_id(app, tray_ids::STOP, "Stop recording", true, None::<&str>)?;
    let open = MenuItem::with_id(app, tray_ids::OPEN, "Open Echo", true, None::<&str>)?;
    let snooze = MenuItem::with_id(
        app,
        tray_ids::SNOOZE,
        "Pause meeting reminders for an hour",
        true,
        None::<&str>,
    )?;
    let quit = MenuItem::with_id(app, tray_ids::QUIT, "Quit Echo", true, None::<&str>)?;
    let separator = PredefinedMenuItem::separator(app)?;

    let menu = Menu::with_items(
        app,
        &[&start, &stop, &separator, &open, &snooze, &separator, &quit],
    )?;

    TrayIconBuilder::with_id("echo-tray")
        .icon(Image::from_bytes(TRAY_ICON_IDLE)?)
        .icon_as_template(true)
        .tooltip("Echo")
        .menu(&menu)
        .show_menu_on_left_click(false)
        .on_menu_event(|app, event| {
            let action = match event.id().as_ref() {
                tray_ids::START => Some(TrayAction::Start),
                tray_ids::STOP => Some(TrayAction::Stop),
                tray_ids::OPEN => Some(TrayAction::Open),
                tray_ids::SNOOZE => Some(TrayAction::PauseDetection),
                tray_ids::QUIT => Some(TrayAction::Quit),
                _ => None,
            };
            let Some(action) = action else { return };

            match action {
                TrayAction::Quit => {
                    let _ = app.emit(events::TRAY_ACTION, events::TrayActionPayload { action });
                    crate::quit(app);
                }
                TrayAction::Open => {
                    navigate(
                        app,
                        events::NavigatePayload {
                            target: events::NavigateTarget::Home,
                            ..Default::default()
                        },
                    );
                }
                TrayAction::Start => {
                    // Open the window too, so there is always somewhere to see
                    // that recording began.
                    navigate(
                        app,
                        events::NavigatePayload {
                            target: events::NavigateTarget::HomeStart,
                            ..Default::default()
                        },
                    );
                    let _ = app.emit(events::TRAY_ACTION, events::TrayActionPayload { action });
                }
                TrayAction::Stop | TrayAction::PauseDetection => {
                    let _ = app.emit(events::TRAY_ACTION, events::TrayActionPayload { action });
                }
            }
        })
        .on_tray_icon_event(|tray, event| {
            let TrayIconEvent::Click {
                button,
                button_state,
                rect,
                ..
            } = event
            else {
                return;
            };
            // macOS reports both halves of the click; act on the release, so one
            // press is one action.
            if button != MouseButton::Left || button_state != MouseButtonState::Up {
                return;
            }
            on_tray_left_click(tray.app_handle(), rect);
        })
        .build(app)?;

    Ok(())
}

/// Left-clicking the tray. While a recording is running that means "show me the
/// recording" — the little panel under the icon, with the clock and Stop.
/// Otherwise it means what it has always meant: bring the window back.
fn on_tray_left_click(app: &AppHandle, rect: tauri::Rect) {
    let Some(state) = app.try_state::<panel::Panel>() else {
        focus_main_window(app);
        return;
    };
    if panel::motion_for(state.capture()) == panel::IconMotion::Off {
        focus_main_window(app);
        return;
    }
    let started_at_ms = state
        .started_at_ms()
        .unwrap_or_else(|| chrono::Utc::now().timestamp_millis());
    let paused = panel::motion_for(state.capture()) == panel::IconMotion::Held;
    panel::toggle_recording(app, Some(tray_anchor(app, &rect)), started_at_ms, paused);
}

/// The tray icon's own rectangle, in physical pixels, so the panel can sit
/// under it. The runtime already hands us physical values; the scale factor is
/// only there in case that ever changes.
fn tray_anchor(app: &AppHandle, rect: &tauri::Rect) -> panel::Area {
    let scale = app
        .get_webview_window("main")
        .and_then(|window| window.scale_factor().ok())
        .unwrap_or(1.0);
    let position = rect.position.to_physical::<f64>(scale);
    let size = rect.size.to_physical::<f64>(scale);
    panel::Area::new(position.x, position.y, size.width, size.height)
}

/// Keep the tray pulse and the floating panel in step with capture, by
/// listening to the same event the UI gets. Nothing in `session` has to know
/// either of them exists.
fn watch_capture(app: &AppHandle) {
    let handle = app.clone();
    app.listen(events::CAPTURE_STATE, move |event| {
        let Ok(status) = serde_json::from_str::<CaptureStatus>(event.payload()) else {
            return;
        };
        let handle = handle.clone();
        // Onto Tauri's runtime: the pulse spawns its timer on whatever runtime
        // is current, and this callback arrives from wherever the session
        // happened to be.
        tauri::async_runtime::spawn(async move { on_capture_state(&handle, status) });
    });
}

fn on_capture_state(app: &AppHandle, status: CaptureStatus) {
    let motion = panel::motion_for(status.state);

    if let Some(state) = app.try_state::<panel::Panel>() {
        state.note_capture(status.state, recording_started_at_ms(&status));

        // Whichever panel is on screen, take it away once it is talking about
        // something that is no longer true: the "meeting detected" nudge the
        // moment recording starts, the recording panel the moment it stops.
        let stale = match state.showing() {
            Some(PanelState::Detected { .. }) => motion != panel::IconMotion::Off,
            Some(PanelState::Recording { .. }) => motion == panel::IconMotion::Off,
            None => false,
        };
        if stale {
            panel::hide(app);
        } else {
            // Still the right panel, but "Listening…" and a ticking clock would
            // be untrue while the recording is paused.
            panel::refresh_recording(app, motion == panel::IconMotion::Held);
        }
    }

    if let Some(animation) = app.try_state::<panel::TrayAnimation>() {
        animation.apply(motion, Arc::new(panel::TrayPainter::new(app.clone())));
    }
}

/// When the running recording started, in epoch milliseconds, so the panel can
/// count on its own instead of being fed a clock tick every second.
fn recording_started_at_ms(status: &CaptureStatus) -> Option<i64> {
    if panel::motion_for(status.state) == panel::IconMotion::Off {
        return None;
    }
    if let Some(started) = status
        .started_at
        .as_deref()
        .and_then(|at| chrono::DateTime::parse_from_rfc3339(at).ok())
    {
        return Some(started.timestamp_millis());
    }
    // No start time on the status: the elapsed clock is the next best answer.
    Some(chrono::Utc::now().timestamp_millis() - status.elapsed_ms)
}

/// Everything that has to happen before the window is usable.
fn setup(app: &AppHandle) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Paths. The configured storage location is read from the database once
    //    it is open; until then the default keeps us going.
    let boot_paths = paths::AppPaths::resolve(None)?;
    boot_paths.ensure()?;

    // 2. Logging, before anything that might fail interestingly.
    logging::init(&boot_paths.log_dir)?;
    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        os = std::env::consts::OS,
        arch = std::env::consts::ARCH,
        "Echo starting"
    );

    // 3. Database. Migrations are transactional, so a half-applied update
    //    cannot happen.
    let db = tauri::async_runtime::block_on(db::connect(&boot_paths.db_path))?;

    // 4. Re-resolve paths against the configured storage location, then clean
    //    up after any crash.
    let app_paths = tauri::async_runtime::block_on(settings::app_paths(&db))?;
    app_paths.ensure()?;

    let loaded = tauri::async_runtime::block_on(settings::load(&db))?;
    let requeued = tauri::async_runtime::block_on(repo_requeue(&db));
    if requeued > 0 {
        tracing::info!(requeued, "picked background work back up after a restart");
    }
    // Nothing else sweeps half-finished downloads, and a partial file left by a
    // crash is dead weight on someone's disk.
    match tauri::async_runtime::block_on(asr::models::clean_stale_partials(&db, &app_paths)) {
        Ok(swept) if swept > 0 => tracing::info!(swept, "cleaned up half-finished downloads"),
        Ok(_) => {}
        Err(error) => tracing::warn!(%error, "could not clean up half-finished downloads"),
    }

    let flags = UiFlags::default();
    flags
        .close_to_tray
        .store(loaded.close_to_tray, Ordering::SeqCst);
    app.manage(flags);

    // Bring the login item in line with what the person asked for. Someone may
    // have removed it in System Settings, or the app may have moved, so the
    // stored value is the intent and this reconciles the world to it.
    apply_launch_at_login(app, loaded.launch_at_login);

    // The floating panel and the tray pulse. Both are pure state until
    // something needs them: no window is created and no timer runs until a
    // meeting is noticed or a recording starts (mantra 1).
    app.manage(panel::Panel::new());
    app.manage(panel::TrayAnimation::new());

    app.manage(AppState {
        session: session::SessionManager::new(
            db.clone(),
            app_paths.clone(),
            loaded.release_after_idle_minutes,
        ),
        detect: detect::Watcher::new(loaded.detection_enabled),
        db,
        paths: app_paths,
    });

    // 5. Point the session's event bus at the window, then start the things
    //    that have to be running for the app to work at all. None of this loads
    //    a speech engine or opens an audio device (mantra 1): the job runner is
    //    an idle 2s poll over an empty table, and the detection watcher parks
    //    itself when detection is off.
    if let Some(state) = app.try_state::<AppState>() {
        state
            .session
            .attach_events(std::sync::Arc::new(session::ports::TauriEvents::new(
                app.clone(),
            )));

        // The meeting watcher: one 5s poll and nothing else.
        if let Err(error) = tauri::async_runtime::block_on(state.detect.start(app.clone())) {
            tracing::warn!(%error, "the meeting watcher could not start");
        }

        // Draining the jobs table, and the recovery scan that puts an
        // interrupted meeting in front of the person.
        let handle = app.clone();
        tauri::async_runtime::spawn(async move {
            let Some(state) = handle.try_state::<AppState>() else {
                return;
            };
            if let Err(error) = state.session.start_job_runner().await {
                tracing::warn!(%error, "background work is not being picked up");
            }
            match state.session.find_interrupted().await {
                Ok(ids) if !ids.is_empty() => {
                    tracing::info!(count = ids.len(), "found meetings to offer finishing");
                }
                Ok(_) => {}
                Err(error) => tracing::warn!(%error, "could not look for interrupted meetings"),
            }
        });
    }

    // 6. Tray, panel and window behaviour.
    watch_capture(app);

    if let Err(error) = build_tray(app) {
        // A missing tray is survivable; the window is the primary surface.
        tracing::warn!(%error, "no tray icon on this system");
    }

    if let Some(window) = app.get_webview_window("main") {
        let handle = app.clone();
        window.on_window_event(move |event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                let Some(flags) = handle.try_state::<UiFlags>() else {
                    return;
                };
                if flags.quitting.load(Ordering::SeqCst) {
                    return;
                }
                if flags.close_to_tray.load(Ordering::SeqCst) {
                    api.prevent_close();
                    if let Some(w) = handle.get_webview_window("main") {
                        let _ = w.hide();
                    }
                }
            }
        });
    }

    Ok(())
}

async fn repo_requeue(db: &db::Db) -> u64 {
    db::repo::requeue_orphaned_jobs(db).await.unwrap_or(0)
}

/// Register or remove the login item. Returns what the operating system now
/// reports, so a refusal never leaves the toggle claiming something untrue.
pub fn set_launch_at_login(app: &AppHandle, wanted: bool) -> Result<bool, String> {
    let manager = app
        .try_state::<tauri_plugin_autostart::AutoLaunchManager>()
        .ok_or_else(|| "starting at login is not available on this system".to_string())?;
    let result = if wanted {
        manager.enable()
    } else {
        manager.disable()
    };
    if let Err(error) = result {
        return Err(error.to_string());
    }
    manager.is_enabled().map_err(|e| e.to_string())
}

/// Same, for launch: log what happened and carry on. Not being able to write a
/// login item is never a reason to refuse to open.
fn apply_launch_at_login(app: &AppHandle, wanted: bool) {
    match set_launch_at_login(app, wanted) {
        Ok(actual) if actual == wanted => {}
        Ok(actual) => tracing::warn!(
            wanted,
            actual,
            "the system did not accept the start-at-login choice"
        ),
        Err(error) => tracing::warn!(%error, "could not set starting at login"),
    }
}

/// Whole-app events, as opposed to per-window ones.
///
/// The one that matters: clicking Echo in the Dock while the window is closed to
/// the tray. macOS sends the application `applicationShouldHandleReopen`, which
/// Tauri surfaces as [`tauri::RunEvent::Reopen`]; without handling it, a person
/// who closed the window has no way back in except the tray, and mantra 4 says
/// no state is a dead end. `has_visible_windows` is the system's own answer, so
/// a click while the window is already up does nothing.
fn on_run_event(app: &AppHandle, event: &tauri::RunEvent) {
    #[cfg(target_os = "macos")]
    if let tauri::RunEvent::Reopen {
        has_visible_windows,
        ..
    } = event
    {
        if !has_visible_windows {
            tracing::debug!("reopened from the Dock");
            focus_main_window(app);
        }
    }
    #[cfg(not(target_os = "macos"))]
    let _ = (app, event);
}

/// Entry point, called from `main.rs`.
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| {
            // A second launch means the person wants the window, not a
            // second copy of Echo.
            navigate(
                app,
                events::NavigatePayload {
                    target: events::NavigateTarget::Home,
                    ..Default::default()
                },
            );
        }))
        .plugin(tauri_plugin_notification::init())
        // "Start Echo when I log in" has to actually register a login item;
        // a toggle that only remembers itself is a lying control.
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent,
            None,
        ))
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(
            // The file log belongs to `logging`; this only carries messages
            // from the UI to the terminal during development.
            tauri_plugin_log::Builder::new()
                .target(tauri_plugin_log::Target::new(
                    tauri_plugin_log::TargetKind::Stdout,
                ))
                .level(if cfg!(debug_assertions) {
                    log::LevelFilter::Debug
                } else {
                    log::LevelFilter::Warn
                })
                .build(),
        )
        .setup(|app| {
            setup(app.handle())?;
            Ok(())
        })
        .invoke_handler(echo_command_handler!())
        // `build` rather than `run`, so the run loop's own events (the Dock
        // click above) reach us.
        .build(tauri::generate_context!())
        .expect("Echo could not start");

    app.run(|app, event| on_run_event(app, &event));
}
