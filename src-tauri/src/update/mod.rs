//! Keeping Echo current, without ever getting in a meeting's way.
//!
//! Every half hour Echo asks GitHub whether there is a newer release. If there
//! is, it downloads it, swaps the app on disk, and says so once — a pill in the
//! corner offering a restart. Nothing restarts on its own.
//!
//! WHY THIS IS NOT A JOB IN THE JOBS TABLE
//! It would have been tempting: a job would inherit parking and resuming for
//! free. But the job runner is strictly serial (see `session::jobs`), so a
//! sixty-megabyte download would sit in front of a just-ended meeting's
//! transcript and recap. That is the exact inversion of the priority rule the
//! whole of that module is built around. So this owns one timer, outside the
//! queue, and yields to the queue instead of competing with it.
//!
//! WHAT "PAUSE AND RESUME LATER" MEANS HERE
//! Not that the meeting's work waits for the update — the opposite. A tick that
//! finds a meeting being recorded, or any work still outstanding for one, does
//! nothing at all and says nothing: no event, no notice, no log a person would
//! ever see. It asks again in half an hour. The posture is copied from
//! `SessionManager::ensure_speech_current`, which leaves older speech weights in
//! place "until this meeting is finished" rather than announcing a deferral
//! nobody can act on.
//!
//! WHY INSTALLING UNDER A RUNNING ECHO IS SAFE
//! Because `asr::models::build_identity` is pinned in a `OnceLock` for exactly
//! this case — its docblock names it: "an updater staging the new app replaces
//! the executable underneath a running Echo". The running process keeps naming
//! the build that actually paid for the engine compile, so the marker it writes
//! is about the binary that did the work. Nothing here needs to defer the swap
//! until exit, and a person who quits by hand simply gets the new version next
//! time.
//!
//! WHAT SECURES THIS
//! Not Apple's signature — the ed25519 one. Every release is signed with a
//! private key that lives in one password manager and one GitHub secret, and the
//! public half is compiled into this binary (`tauri.conf.json`,
//! `plugins.updater.pubkey`). A bundle that does not verify against it is
//! refused before a byte of it is unpacked. Serving the endpoint is therefore
//! not enough to ship code to anybody.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use tauri::async_runtime::JoinHandle;
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_updater::UpdaterExt;
use tokio::sync::Notify;

use crate::db::{repo, Db};
use crate::events::{self, UpdateStatePayload};
use crate::session;
use crate::settings::keys;
use crate::types::{CaptureState, UpdateState};

/// How often Echo asks whether there is a newer release.
///
/// Half an hour, which is the number that was asked for. It is also about the
/// right order of magnitude for the thing being waited on: releases are cut by
/// hand, and a person who has just published one does not need the app to notice
/// inside a minute.
pub const CHECK_INTERVAL: Duration = Duration::from_secs(30 * 60);

/// How long after launch the first check happens.
///
/// Not immediately. Launch is the busiest minute Echo has — migrations, the
/// recovery scan, reconciling the speech weights, possibly a quarter of an hour
/// of engine setup — and an update is the least urgent thing in that list. It is
/// also the minute somebody is most likely to be about to start a meeting.
const FIRST_CHECK_DELAY: Duration = Duration::from_secs(90);

/// How long a recorded "last checked" answer is trusted.
///
/// Restarting Echo repeatedly should not mean asking GitHub repeatedly. The
/// window is the interval itself, so the timer and this agree on what "recently"
/// means.
const RECENT_ENOUGH: Duration = CHECK_INTERVAL;

/// Whether it is a reasonable moment to spend bandwidth and replace the app.
///
/// A pure function of two facts, so the rule can be tested without a session,
/// and so it reads as one sentence rather than being spread over the tick below.
///
/// `engine_is_needed_by_capture` rather than `is_live` deliberately: it also
/// covers `Starting` and `Stopping`, which are the two moments where a meeting
/// exists but is not yet, or no longer, technically recording. Neither is a
/// moment to start rewriting the application bundle.
///
/// The second fact is the one that makes this "pause and resume later" rather
/// than "pause while recording": a meeting that ended five minutes ago still has
/// its transcript and recap to be written, and those get the machine first.
pub fn may_disturb(capture: CaptureState, outstanding_meeting_jobs: usize) -> bool {
    !session::engine_is_needed_by_capture(capture) && outstanding_meeting_jobs == 0
}

/// Follows the one question this module asks, and remembers the answer.
///
/// Shaped like `detect::Watcher` on purpose — it is the codebase's pattern for a
/// module that owns exactly one timer and needs the running app to do its work.
pub struct Watcher {
    state: Mutex<UpdateStatePayload>,
    task: Mutex<Option<JoinHandle<()>>>,
    /// "Ask now", for anything that wants to skip to the next tick.
    wake: Notify,
    db: OnceLock<Db>,
    /// How many checks in a row have failed, for one log line rather than a
    /// hundred. Same reasoning as the capture-state poll on the UI side.
    failures: AtomicU32,
}

impl Default for Watcher {
    fn default() -> Self {
        Self::new()
    }
}

impl Watcher {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(UpdateStatePayload::default()),
            task: Mutex::new(None),
            wake: Notify::new(),
            db: OnceLock::new(),
            failures: AtomicU32::new(0),
        }
    }

    /// What a screen should be showing about updates.
    ///
    /// Asked for outright as well as announced, because the announcement can
    /// happen while no window exists — the lesson the one-time setup pill had to
    /// learn the hard way (`src/hooks/useSetupJob.ts`).
    pub fn state(&self) -> UpdateStatePayload {
        self.state.lock().unwrap().clone()
    }

    /// True once a new version is on disk and only a restart is left.
    pub fn is_ready(&self) -> bool {
        self.state.lock().unwrap().state == UpdateState::Ready
    }

    /// Begin the half-hourly check. Idempotent, like every other start in this
    /// codebase, because a double call must never mean two timers.
    pub async fn start(&self, app: AppHandle) {
        if self.task.lock().unwrap().is_some() {
            return;
        }
        if let Some(state) = app.try_state::<crate::commands::AppState>() {
            let _ = self.db.set(state.db.clone());
        }

        let handle = tauri::async_runtime::spawn(async move {
            tokio::time::sleep(FIRST_CHECK_DELAY).await;
            loop {
                if let Some(state) = app.try_state::<crate::commands::AppState>() {
                    state.updates.tick(&app).await;
                }
                let Some(state) = app.try_state::<crate::commands::AppState>() else {
                    return;
                };
                // Interruptible, so "ask now" does not have to wait out the
                // half hour — the `Notify` + sleep pair `session::jobs` uses.
                tokio::select! {
                    _ = state.updates.wake.notified() => {}
                    _ = tokio::time::sleep(CHECK_INTERVAL) => {}
                }
            }
        });
        *self.task.lock().unwrap() = Some(handle);
    }

    /// Stop checking. Called on the way out, so the task dies with the app
    /// rather than being left for the process to take with it — which is what
    /// the detection watcher does today, and is not a habit worth copying.
    pub fn stop(&self) {
        if let Some(handle) = self.task.lock().unwrap().take() {
            handle.abort();
        }
    }

    /// Ask again at the next opportunity.
    pub fn check_soon(&self) {
        self.wake.notify_one();
    }

    /// One pass: is there a newer Echo, is this a moment to fetch it, and if so
    /// fetch it and say so once.
    async fn tick(&self, app: &AppHandle) {
        // Already downloaded and waiting for a restart. Asking again would
        // achieve nothing and could only replace a true answer with a worse one.
        if self.is_ready() {
            return;
        }

        if !self.moment_is_right(app).await {
            // Deliberately quiet. Nobody can act on "an update is waiting for
            // your meeting to finish", and saying it during a meeting is exactly
            // the kind of interruption mantra 1's amendment rules out.
            tracing::debug!("not a moment to look for a new version; will ask again later");
            return;
        }

        if self.checked_recently().await {
            return;
        }

        self.publish(app, UpdateStatePayload::of(UpdateState::Checking));

        let found = match app.updater() {
            Ok(updater) => updater.check().await,
            Err(error) => Err(error),
        };

        match found {
            Ok(Some(update)) => {
                let version = update.version.clone();
                let notes = update.body.clone();
                tracing::info!(version = %version, "a newer Echo is available");
                self.publish(
                    app,
                    UpdateStatePayload {
                        state: UpdateState::Downloading,
                        version: Some(version.clone()),
                        notes: notes.clone(),
                    },
                );
                // Quietly. It was not asked for, and the corner it would report
                // into belongs to the one-time setup, which is the pill that
                // actually blocks recording.
                match update.download_and_install(|_, _| {}, || {}).await {
                    Ok(()) => {
                        self.failures.store(0, Ordering::SeqCst);
                        self.mark_checked().await;
                        tracing::info!(version = %version, "the new version is installed and waiting for a restart");
                        self.publish(
                            app,
                            UpdateStatePayload {
                                state: UpdateState::Ready,
                                version: Some(version),
                                notes,
                            },
                        );
                    }
                    Err(error) => self.went_wrong(app, error),
                }
            }
            Ok(None) => {
                self.failures.store(0, Ordering::SeqCst);
                self.mark_checked().await;
                tracing::debug!("this is the newest Echo there is");
                self.publish(app, UpdateStatePayload::of(UpdateState::Idle));
            }
            Err(error) => self.went_wrong(app, error),
        }
    }

    /// Both halves of the gate, read as late as possible so the answer is about
    /// now rather than about when this tick was scheduled.
    async fn moment_is_right(&self, app: &AppHandle) -> bool {
        let Some(state) = app.try_state::<crate::commands::AppState>() else {
            return false;
        };
        let capture = state.session.status().await.state;
        let outstanding = session::jobs::outstanding_meeting_jobs(&state.db).await;
        may_disturb(capture, outstanding)
    }

    /// Whether the last check was recent enough to skip this one. A failure to
    /// read the answer means "no": asking once more is cheap, and never asking
    /// is the failure that matters.
    async fn checked_recently(&self) -> bool {
        let Some(db) = self.db.get() else {
            return false;
        };
        let Ok(Some(raw)) = repo::get_setting(db, keys::UPDATE_LAST_CHECKED_AT).await else {
            return false;
        };
        let Ok(then) = chrono::DateTime::parse_from_rfc3339(&raw) else {
            return false;
        };
        let elapsed = chrono::Utc::now().signed_duration_since(then.with_timezone(&chrono::Utc));
        elapsed.num_seconds() >= 0 && (elapsed.num_seconds() as u64) < RECENT_ENOUGH.as_secs()
    }

    async fn mark_checked(&self) {
        let Some(db) = self.db.get() else { return };
        let now = chrono::Utc::now().to_rfc3339();
        if let Err(error) = repo::set_setting(db, keys::UPDATE_LAST_CHECKED_AT, &now).await {
            tracing::debug!(%error, "could not write down when Echo last looked for a new version");
        }
    }

    /// A check or a download failed.
    ///
    /// Never told to the person: there is nothing they can do about it, the app
    /// they have works, and an app that interrupts a meeting to report its own
    /// plumbing is worse than one that carries on. Told to the log, with a count,
    /// because one failure is a blip the next tick fixes and a number that keeps
    /// climbing is something else.
    fn went_wrong(&self, app: &AppHandle, error: tauri_plugin_updater::Error) {
        let failures = self.failures.fetch_add(1, Ordering::SeqCst) + 1;
        tracing::warn!(%error, failures, "could not look for a new version");
        self.publish(app, UpdateStatePayload::of(UpdateState::Failed));
    }

    fn publish(&self, app: &AppHandle, payload: UpdateStatePayload) {
        let ready = payload.state == UpdateState::Ready;
        *self.state.lock().unwrap() = payload.clone();
        // The tray's own item, for somebody whose window is closed. The icon is
        // deliberately untouched: it answers three questions about meetings and
        // an update is not one of them (`session::tray_state_for`).
        crate::show_restart_menu_item(app, ready);
        let _ = app.emit(events::UPDATE_STATE, payload);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_meeting_being_recorded_is_never_a_moment_to_replace_the_app() {
        for state in [
            CaptureState::Recording,
            CaptureState::Paused,
            CaptureState::Degraded,
        ] {
            assert!(!may_disturb(state, 0), "{state:?} should defer");
        }
    }

    #[test]
    fn the_edges_of_a_meeting_count_as_the_meeting() {
        // `Starting` is the weights being loaded for a meeting that is about to
        // happen; `Stopping` is the pipeline still writing down what it heard.
        assert!(!may_disturb(CaptureState::Starting, 0));
        assert!(!may_disturb(CaptureState::Stopping, 0));
    }

    #[test]
    fn work_still_owed_to_a_finished_meeting_defers_it_too() {
        // This is the half that makes the rule "pause and resume later" rather
        // than merely "not while recording": the recap of the meeting somebody
        // just walked out of gets the machine first.
        assert!(!may_disturb(CaptureState::Idle, 1));
        assert!(!may_disturb(CaptureState::Stopped, 3));
    }

    #[test]
    fn an_idle_echo_with_an_empty_queue_is_the_moment() {
        for state in [
            CaptureState::Idle,
            CaptureState::Stopped,
            CaptureState::Failed,
        ] {
            assert!(may_disturb(state, 0), "{state:?} should proceed");
        }
    }
}
