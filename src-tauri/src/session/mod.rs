//! The capture state machine and the thing that drives a recording.
//!
//! Three separate concerns, deliberately not merged (DESIGN §3, review finding
//! 19):
//! * **Capture state** lives here, in memory, mirrored to the UI.
//! * **Jobs** live in the `jobs` table, so they survive a restart ([`jobs`]).
//! * **Detection** is its own watcher in [`crate::detect`].
//!
//! Order of operations when a recording starts:
//! 1. write the meeting row (so a crash leaves something to recover)
//! 2. open capture and start writing chunks from t=0
//! 3. pause background jobs, because recording preempts everything
//! 4. load the speech engine and begin live transcription
//!
//! Steps 3 and 4 may fail without ending the recording. Step 2 failing on one
//! source degrades capture; failing on both is a real error.
//!
//! The speech engine's lifecycle is scoped to *listening*, not to the last
//! decode (mantra 1's amendment of 2026-08-20): it is held from the moment a
//! meeting is detected or started until the last of that meeting's jobs finishes,
//! and only then does its grace period start. Two lines say the whole rule —
//! [`engine_is_needed_by_capture`] and [`engine_stays_resident`] — and one method
//! applies it wherever either fact changes ([`Inner::refresh_engine_residency`]).
//! If the engine comes up *during* a recording, the backlog on disk is read into
//! the transcript before live decoding carries on
//! ([`pipeline::catch_up_backlog`]). That handover splits the meeting clock in
//! two — the recording owns what came before, the live pass what comes after
//! ([`pipeline::Backlog`]) — and the pass doing the reading is a task this
//! module owns, so a stop waits for it before queueing the job that finishes the
//! meeting.
//!
//! Every entry point is idempotent: a second Start while recording returns the
//! meeting already in progress, and a second Stop returns the same result.
//!
//! Everything the spine touches goes through the traits in [`ports`], so the
//! whole thing can be driven in tests with no microphone and no weights on disk.

pub mod jobs;
pub mod pipeline;
pub mod ports;
pub mod recovery;
pub mod retranscribe;
pub mod speaker_count;

#[cfg(test)]
mod mock;

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::task::JoinHandle;

use crate::audio::{AudioError, CaptureConfig};
use crate::db::{repo, Db};
use crate::events::{MeetingUpdatedPayload, NoticeLevel, NoticePayload};
use crate::paths::AppPaths;
use crate::session::ports::{CaptureHandle, CaptureStart, EventBus, EventSink, Ports, UiEvent};
use crate::types::{
    CaptureState, CaptureStatus, Channel, DegradedReason, Id, JobKind, Marker, MarkerKind,
    MeetingStatus, RecoveryAction, SpeechState, StartRecordingOptions, Timestamp, TrayState,
};

/// How long a stop waits for the live pipeline to write what it already has.
/// Bounded, because a stuck engine must never stop a meeting from ending.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(5);

/// Less committed audio than this, with nothing written down, is not a meeting:
/// it is a mis-click, a notification tapped by accident, or a start-then-stop
/// while looking for the right button. See [`recovery::meeting_has_content`].
pub const MIN_KEPT_MS: i64 = 3_000;

#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("not implemented yet")]
    NotImplemented,
    #[error("a recording is already running")]
    AlreadyRecording,
    #[error("nothing is being recorded")]
    NotRecording,
    #[error("Echo could not hear anything: {0}")]
    NoAudioSources(String),
    /// There is no committed audio behind this meeting, so nothing can be read
    /// back off disk. The person deleted the recording and kept the words.
    #[error("no recorded audio for this meeting")]
    NoRecordedAudio,
    /// Work is already running for this meeting and the request would change
    /// what that work is doing. Not a fault, and worth trying again in a moment.
    #[error("this meeting is still being worked on")]
    MeetingBusy,
    #[error("storage problem: {0}")]
    Storage(String),
    #[error("database problem: {0}")]
    Db(#[from] crate::db::DbError),
}

/// What can happen to a capture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureEvent {
    /// The person (or the tray, or a notification) asked to start.
    StartRequested,
    /// At least one source is producing audio.
    Started,
    PauseRequested,
    ResumeRequested,
    StopRequested,
    /// Streams closed and every chunk is committed.
    Stopped,
    /// Capture cannot continue.
    Failed,
    /// One source went away; the other keeps going.
    SourceLost(DegradedReason),
    /// A lost source came back.
    SourceRecovered,
    /// An unfinished meeting was found at launch.
    InterruptedFound,
    /// The person chose finish or discard for an interrupted meeting.
    RecoveryResolved,
}

/// The transition table. `None` means "ignore this event in this state", which
/// is what makes double-clicks safe.
///
/// Returning the current state (rather than `None`) is also fine and means "this
/// was already true".
pub fn next_state(current: CaptureState, event: &CaptureEvent) -> Option<CaptureState> {
    use CaptureEvent as E;
    use CaptureState as S;

    match (current, event) {
        // Starting
        (S::Idle | S::Stopped | S::Failed, E::StartRequested) => Some(S::Starting),
        (S::Starting, E::Started) => Some(S::Recording),
        (S::Starting, E::Failed) => Some(S::Failed),
        // A second Start while already going is a no-op, not an error.
        (S::Starting | S::Recording | S::Degraded | S::Paused, E::StartRequested) => None,

        // Pause and resume
        (S::Recording | S::Degraded, E::PauseRequested) => Some(S::Paused),
        (S::Paused, E::ResumeRequested) => Some(S::Recording),
        (S::Paused, E::PauseRequested) => None,
        (S::Recording | S::Degraded, E::ResumeRequested) => None,

        // Stopping
        (S::Recording | S::Degraded | S::Paused | S::Starting, E::StopRequested) => {
            Some(S::Stopping)
        }
        (S::Stopping, E::Stopped) => Some(S::Stopped),
        (S::Stopping, E::StopRequested) => None,
        (S::Idle | S::Stopped, E::StopRequested) => None,

        // Degrading while live
        (S::Recording, E::SourceLost(_)) => Some(S::Degraded),
        (S::Degraded, E::SourceLost(_)) => None,
        (S::Degraded, E::SourceRecovered) => Some(S::Recording),
        (S::Recording, E::SourceRecovered) => None,

        // Losing everything is a failure, wherever it happens.
        (_, E::Failed) => Some(S::Failed),

        // Recovery at launch
        (S::Idle, E::InterruptedFound) => Some(S::Recovering),
        (S::Recovering, E::RecoveryResolved) => Some(S::Idle),
        (S::Recovering, E::StartRequested) => Some(S::Starting),

        _ => None,
    }
}

/// Can a recording start from here?
pub fn can_start(state: CaptureState) -> bool {
    matches!(
        state,
        CaptureState::Idle
            | CaptureState::Stopped
            | CaptureState::Failed
            | CaptureState::Recovering
    )
}

/// Is audio being written right now?
pub fn is_live(state: CaptureState) -> bool {
    matches!(
        state,
        CaptureState::Recording | CaptureState::Degraded | CaptureState::Paused
    )
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

/// Everything the UI needs about the current capture, plus the recovery list.
/// Guarded by a plain mutex and never held across an await.
#[derive(Debug, Clone, Default)]
pub(crate) struct LiveState {
    pub(crate) state: CaptureState,
    pub(crate) meeting_id: Option<Id>,
    pub(crate) started_at: Option<Timestamp>,
    pub(crate) elapsed_ms: i64,
    pub(crate) active_channels: Vec<Channel>,
    pub(crate) degraded_reason: Option<DegradedReason>,
    /// The engine did not come up for this meeting. Cleared by the next Start,
    /// because a new meeting is a new attempt — the only remembered half of
    /// [`speech_state`], and only because "it failed" is not something the
    /// engine itself keeps.
    pub(crate) speech_load_failed: bool,
    /// Interrupted meetings waiting for a finish-or-discard answer.
    pub(crate) recovering: Vec<Id>,
}

/// The guts of [`SessionManager`], shared with the pipeline and recovery tasks.
pub(crate) struct Inner {
    pub(crate) db: Db,
    pub(crate) paths: AppPaths,
    pub(crate) ports: Ports,
    pub(crate) jobs: Arc<jobs::JobRuntime>,
    pub(crate) live: std::sync::Mutex<LiveState>,
    /// The running capture. `None` whenever nothing is being recorded, which is
    /// also how "no audio device is open while idle" is enforced (mantra 1).
    pub(crate) capture: tokio::sync::Mutex<Option<Box<dyn CaptureHandle>>>,
    /// Serialises start/stop/pause/resume so two clicks cannot interleave.
    pub(crate) command: tokio::sync::Mutex<()>,
    pub(crate) tasks: std::sync::Mutex<Vec<JoinHandle<()>>>,
    /// How the meeting clock is split between the live pass and the pass that
    /// reads this meeting's start off disk, for the recording in progress
    /// ([`pipeline::Backlog`]). Replaced by every start.
    pub(crate) backlog: std::sync::Mutex<Option<Arc<pipeline::Backlog>>>,
    /// The task doing that reading, owned rather than detached: a stop has to be
    /// able to wait for it before the finalize job is queued, or the two passes
    /// fill the same holes (review of 2026-08-20, finding 1).
    pub(crate) backlog_task: std::sync::Mutex<Option<JoinHandle<()>>>,
    /// Utterances waiting for text, for the "catching up" hint.
    pub(crate) pending: AtomicU32,
}

impl Inner {
    pub(crate) fn state(&self) -> CaptureState {
        self.live.lock().expect("capture state lock").state
    }

    /// Feed the state machine. Returns the new state when something changed.
    pub(crate) fn transition(&self, event: &CaptureEvent) -> Option<CaptureState> {
        let mut live = self.live.lock().expect("capture state lock");
        let next = next_state(live.state, event)?;
        if next == live.state {
            return None;
        }
        tracing::info!(from = ?live.state, to = ?next, ?event, "capture state");
        live.state = next;
        Some(next)
    }

    pub(crate) fn status_snapshot(&self) -> CaptureStatus {
        let live = self.live.lock().expect("capture state lock");
        CaptureStatus {
            state: live.state,
            meeting_id: live.meeting_id.clone(),
            elapsed_ms: live.elapsed_ms,
            active_channels: live.active_channels.clone(),
            degraded_reason: live.degraded_reason,
            pending_utterances: self.pending.load(Ordering::SeqCst),
            started_at: live.started_at.clone(),
            // Asked of the engine here, every time, rather than remembered by
            // anything that emits: this status is polled as well as pushed, so
            // a derived answer is one the UI can always get hold of.
            speech: speech_state(
                self.ports.asr.is_loaded(),
                live.speech_load_failed,
                live.state,
            ),
        }
    }

    pub(crate) fn emit_state(&self) {
        self.ports
            .events
            .emit(UiEvent::CaptureState(self.status_snapshot()));
    }

    pub(crate) fn notice(&self, payload: NoticePayload) {
        self.ports.events.emit(UiEvent::Notice(payload));
    }

    pub(crate) fn set_elapsed(&self, t_ms: i64) {
        let mut live = self.live.lock().expect("capture state lock");
        if t_ms > live.elapsed_ms {
            live.elapsed_ms = t_ms;
        }
    }

    /// Capture is still going, but with less than we wanted. The message comes
    /// from the capture layer, which already writes it as a plain sentence.
    ///
    /// Losing a source changes the state to `Degraded`; running low on disk or
    /// falling behind on live text does not, because everything is still being
    /// recorded.
    pub(crate) async fn note_degraded(
        &self,
        channel: Option<Channel>,
        reason: DegradedReason,
        message: &str,
        meeting_id: &str,
    ) {
        tracing::warn!(?channel, ?reason, "capture degraded");
        let source_lost = matches!(
            reason,
            DegradedReason::SystemAudioUnavailable | DegradedReason::MicrophoneUnavailable
        );

        let elapsed = {
            let mut live = self.live.lock().expect("capture state lock");
            if source_lost || matches!(reason, DegradedReason::StorageLow) {
                live.degraded_reason = Some(reason);
            }
            if let Some(channel) = channel.filter(|_| source_lost) {
                live.active_channels.retain(|c| *c != channel);
            }
            live.elapsed_ms
        };

        if source_lost {
            self.transition(&CaptureEvent::SourceLost(reason));
        }
        self.emit_state();
        self.notice(NoticePayload {
            level: match reason {
                DegradedReason::TranscriptBehind => NoticeLevel::Info,
                _ => NoticeLevel::Warning,
            },
            message: message.to_string(),
            persistent: !matches!(reason, DegradedReason::TranscriptBehind),
            meeting_id: Some(meeting_id.to_string()),
            tag: Some(degraded_tag(reason).to_string()),
        });

        if source_lost {
            // Leave a mark in the transcript so the gap is explainable later.
            let _ = repo::insert_marker(
                &self.db,
                meeting_id,
                elapsed,
                MarkerKind::System,
                Some(message),
            )
            .await;
        }
    }

    pub(crate) fn note_source_recovered(&self, channel: Channel, meeting_id: &str) {
        {
            let mut live = self.live.lock().expect("capture state lock");
            if !live.active_channels.contains(&channel) {
                live.active_channels.push(channel);
            }
            live.degraded_reason = None;
        }
        self.transition(&CaptureEvent::SourceRecovered);
        self.emit_state();
        self.notice(NoticePayload {
            level: NoticeLevel::Info,
            message: "Echo is hearing everything again.".into(),
            persistent: false,
            meeting_id: Some(meeting_id.to_string()),
            tag: Some("captureRecovered".into()),
        });
    }

    /// Capture died. Everything already committed is safe, so this ends the
    /// recording the same way a stop would: no dead ends.
    pub(crate) async fn note_fatal(self: &Arc<Self>, detail: &str, meeting_id: &str) {
        tracing::error!(detail, "capture stopped unexpectedly");
        // Do not join the pipeline tasks here: this runs inside one of them.
        let handle = self.capture.lock().await.take();
        if let Some(handle) = handle {
            let _ = handle.stop().await;
        }
        // The backlog pass is a task of its own, so waiting for it here is safe
        // and necessary: this path queues the finalize job too.
        self.join_backlog_pass().await;

        let committed = jobs::committed_end_ms(&self.db, meeting_id)
            .await
            .unwrap_or(0);
        // Literally the predicate a normal stop uses, not a copy of it: a
        // capture that died is the *most* likely moment for the journal to be
        // behind the files on disk, and that fallback look in the folder is the
        // whole reason `meeting_has_content` exists (mantra 3). On any doubt it
        // answers "there is something here", which is the answer we want.
        let worth_keeping = recovery::meeting_has_content(self, meeting_id)
            .await
            .unwrap_or(true);
        let language = repo::language_histogram(&self.db, meeting_id)
            .await
            .ok()
            .and_then(|h| h.first().map(|(l, _)| l.clone()));
        let status = if worth_keeping {
            MeetingStatus::Processing
        } else {
            MeetingStatus::Failed
        };
        let _ = repo::finish_meeting(
            &self.db,
            meeting_id,
            committed,
            language.as_deref(),
            None,
            status,
        )
        .await;

        {
            let mut live = self.live.lock().expect("capture state lock");
            live.active_channels.clear();
            live.elapsed_ms = live.elapsed_ms.max(committed);
        }
        self.transition(&CaptureEvent::Failed);
        self.emit_state();
        self.notice(NoticePayload {
            level: NoticeLevel::Problem,
            message: if worth_keeping {
                "Echo had to stop listening. Everything recorded up to now is saved.".into()
            } else {
                "Echo had to stop listening before it recorded anything. You can start again."
                    .to_string()
            },
            persistent: true,
            meeting_id: worth_keeping.then(|| meeting_id.to_string()),
            tag: Some("captureFailed".into()),
        });
        self.ports.events.emit(UiEvent::TrayState(TrayState::Idle));

        if worth_keeping {
            // Turn what we have into a finished meeting, recap included when
            // that is what the settings say (on unless turned off).
            let _ = self
                .jobs
                .queue(Some(meeting_id), JobKind::TranscribeCatchup)
                .await;
            let _ = self.jobs.queue(Some(meeting_id), JobKind::Diarize).await;
            let _ = self.jobs.queue(Some(meeting_id), JobKind::Mixdown).await;
            let auto_summarize = crate::settings::load(&self.db)
                .await
                .map(|s| s.auto_summarize)
                .unwrap_or(crate::settings::DEFAULT_AUTO_SUMMARIZE);
            if auto_summarize {
                let _ = self.jobs.queue(Some(meeting_id), JobKind::Summarize).await;
            }
        } else {
            // Out of the way rather than in the list. The row is kept (not
            // deleted) because a capture that fell over is worth having in the
            // diagnostics log, and there are no files to reclaim.
            let _ = repo::soft_delete_meeting(&self.db, meeting_id).await;
            self.ports
                .events
                .emit(UiEvent::MeetingUpdated(MeetingUpdatedPayload {
                    meeting_id: meeting_id.to_string(),
                    status: MeetingStatus::Failed,
                    title: None,
                    duration_ms: 0,
                    deleted: true,
                }));
        }
        let _ = self.jobs.release().await;
        // Whatever is left to do for this meeting keeps the engine; if there is
        // nothing, the grace period starts here.
        self.refresh_engine_residency().await;
    }

    /// Stop anything queued or running for this meeting. Used when a meeting is
    /// deleted or discarded: work against a row that is going away is wasted
    /// battery at best and a confusing error at worst.
    pub(crate) async fn cancel_jobs_for(&self, meeting_id: &str) {
        let active = repo::list_jobs(
            &self.db,
            &crate::types::JobQuery {
                meeting_id: Some(meeting_id.to_string()),
                active_only: Some(true),
                ..Default::default()
            },
        )
        .await
        .unwrap_or_default();
        for job in active {
            if let Err(error) = self.jobs.cancel(&job.id).await {
                tracing::debug!(%error, "could not stop work for a meeting that is going away");
            }
        }
        self.refresh_engine_residency().await;
    }

    /// End the pass reading this meeting's start off disk, and wait for it.
    ///
    /// Cancellation is cooperative and already armed: that pass asks between
    /// windows whether this meeting is still the one being recorded, and by the
    /// time anything calls this it is not. So this is the *waiting* half, and it
    /// is the half that matters — the finalize job must not start while another
    /// pass is still writing segments, because "which stretches have no text
    /// against them" is a question with a different answer a second later
    /// (review of 2026-08-20, finding 1).
    ///
    /// Bounded, like every other wait on the stop path: a pass that will not stop
    /// is dropped rather than allowed to hold up the end of a meeting. Dropping
    /// the task cancels it at its next await, which is inside the engine queue,
    /// so nothing is left decoding either.
    pub(crate) async fn join_backlog_pass(&self) {
        let task = self.backlog_task.lock().expect("backlog task lock").take();
        *self.backlog.lock().expect("backlog lock") = None;
        let Some(mut task) = task else {
            return;
        };
        if tokio::time::timeout(DRAIN_TIMEOUT, &mut task)
            .await
            .is_err()
        {
            tracing::warn!("the backlog pass took too long to stop; dropping it");
            task.abort();
        }
    }

    /// A start that got nowhere: back to `Failed`, and let the engine go unless
    /// something else still wants it.
    ///
    /// Every failed start has to come through here. Residency is claimed on the
    /// first line of a start (review of 2026-08-20, finding 3), so a start that
    /// gives up without this would hold the weights for a recording that never
    /// happened.
    pub(crate) async fn start_failed(&self) {
        self.transition(&CaptureEvent::Failed);
        self.emit_state();
        self.refresh_engine_residency().await;
    }

    /// Tell the speech engine whether a meeting still needs it.
    ///
    /// The whole of the listening-scoped lifecycle on this side: the engine is
    /// held while audio is being captured **or** any meeting still has work
    /// running or queued — catch-up, the speaker pass, the recap — and let go
    /// otherwise. Letting go is not unloading: the engine waits out its own
    /// grace period ([`crate::asr::engine::IDLE_GRACE`]) in case the next
    /// meeting starts a minute later.
    ///
    /// Cheap and idempotent, so every path that changes either fact can just
    /// call it: start, stop, a capture that died, a job that finished, a job that
    /// was queued.
    ///
    /// One line, because the rule has to be computed in exactly one place. The
    /// job runtime owns that computation — it is the half that also runs on a
    /// tick — and it reads this capture state through the hook installed in
    /// [`SessionManager::with_ports`].
    pub(crate) async fn refresh_engine_residency(&self) {
        self.jobs.refresh_engine_residency().await;
    }
}

/// Does the capture state on its own mean the engine has to stay?
///
/// Everything from the click to the last chunk being committed: `Starting` counts
/// because the weights are being loaded for this meeting right now, and
/// `Stopping` counts because the pipeline is still writing down what it has.
pub fn engine_is_needed_by_capture(state: CaptureState) -> bool {
    is_live(state) || matches!(state, CaptureState::Starting | CaptureState::Stopping)
}

/// The lifecycle rule, as one line: hold the engine while a meeting is being
/// listened to or has work outstanding, and only then start counting down.
///
/// Deliberately not "is the engine busy right now". A meeting with a long silence
/// in it, or a gap between catch-up finishing and the recap starting, is still
/// one conversation, and unloading three gigabytes in the middle of it only buys
/// a reload (mantra 1's amendment of 2026-08-20).
pub fn engine_stays_resident(capturing: bool, outstanding_meeting_jobs: usize) -> bool {
    capturing || outstanding_meeting_jobs > 0
}

/// What to tell the person about speech, from the two facts that decide it.
///
/// Pure, so the rule can be read and tested without a meeting, a thread or a
/// gigabyte of weights — and called on every status read rather than stored, so
/// nothing here can go stale (see [`crate::types::SpeechState`]).
///
/// The order of the arms *is* the rule:
/// * loaded wins over everything, including a load that failed earlier in this
///   meeting. A later load repairs an earlier failure, and a banner that
///   outlives the problem it described is worse than no banner at all.
/// * a failure is worth saying whether or not a meeting is running: it is the
///   answer to "why is nothing appearing", and the next Start clears it.
/// * otherwise, a meeting that needs the engine and has not got it is
///   `Preparing` — the state that had no way to be shown on 2026-08-24.
pub fn speech_state(loaded: bool, load_failed: bool, capture: CaptureState) -> SpeechState {
    if loaded {
        SpeechState::Ready
    } else if load_failed {
        SpeechState::Unavailable
    } else if engine_is_needed_by_capture(capture) {
        SpeechState::Preparing
    } else {
        SpeechState::Idle
    }
}

// ---------------------------------------------------------------------------
// The manager
// ---------------------------------------------------------------------------

/// Owns the running recording and the background job runner.
pub struct SessionManager(Arc<Inner>);

impl std::fmt::Debug for SessionManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionManager")
            .field("state", &self.0.state())
            .finish()
    }
}

impl SessionManager {
    /// Build the manager. Allocates nothing beyond a few pointers: no device is
    /// opened and no weights are loaded until something asks (mantra 1).
    ///
    /// How long the speech engine is held is deliberately not an argument, and
    /// not a setting either: it is held for as long as a meeting and its
    /// follow-up work need it, then released after a fixed grace
    /// ([`crate::asr::engine::IDLE_GRACE`]) — mantra 1's amendment of
    /// 2026-08-20.
    pub fn new(db: Db, paths: AppPaths) -> Self {
        let ports = Ports::real(db.clone());
        Self::with_ports(db, paths, ports)
    }

    /// Same, with the seams supplied. Tests use this; so could a future
    /// "record from a file" mode.
    pub fn with_ports(db: Db, paths: AppPaths, ports: Ports) -> Self {
        let jobs = jobs::JobRuntime::new(db.clone(), paths.clone(), ports.clone());
        let inner = Arc::new(Inner {
            db,
            paths,
            ports,
            jobs,
            live: std::sync::Mutex::new(LiveState::default()),
            capture: tokio::sync::Mutex::new(None),
            command: tokio::sync::Mutex::new(()),
            tasks: std::sync::Mutex::new(Vec::new()),
            backlog: std::sync::Mutex::new(None),
            backlog_task: std::sync::Mutex::new(None),
            pending: AtomicU32::new(0),
        });
        // The job runtime owns the residency rule, and half of that rule is "is
        // a capture live". This is where it gets to ask. Weak on purpose: the
        // spine owns the runtime, so a strong pointer back would be a cycle
        // nothing ever breaks.
        let capture = Arc::downgrade(&inner);
        inner.jobs.watch_capture(Arc::new(move || {
            capture
                .upgrade()
                .is_some_and(|inner| engine_is_needed_by_capture(inner.state()))
        }));
        Self(inner)
    }

    pub fn db(&self) -> &Db {
        &self.0.db
    }

    pub fn paths(&self) -> &AppPaths {
        &self.0.paths
    }

    /// Point the session's events at the window. Called once at launch, after
    /// the webview exists.
    pub fn attach_events(&self, sink: Arc<dyn EventSink>) {
        self.0.ports.events.set(sink);
    }

    /// The bus other modules can emit on.
    pub fn events(&self) -> Arc<EventBus> {
        self.0.ports.events.clone()
    }

    /// Start recording. Writes the meeting row before opening any device.
    ///
    /// Already recording returns the meeting in progress.
    pub async fn start(&self, opts: StartRecordingOptions) -> Result<Id, SessionError> {
        let inner = self.0.clone();
        let _command = inner.command.lock().await;

        // Idempotent: a second Start is the first one.
        {
            let live = inner.live.lock().expect("capture state lock");
            if is_live(live.state) || matches!(live.state, CaptureState::Starting) {
                if let Some(id) = live.meeting_id.clone() {
                    tracing::debug!("start ignored, already recording");
                    return Ok(id);
                }
            }
        }
        if !can_start(inner.state()) {
            return Err(SessionError::AlreadyRecording);
        }

        let settings = crate::settings::load(&inner.db).await?;
        inner.transition(&CaptureEvent::StartRequested);
        // Before the meeting row, before the devices, before anything that can
        // take a moment: an engine already counting down its grace period has to
        // hear about this recording *now*. Told afterwards, it can unload 1.6 GB
        // between the click and the first chunk and then load it straight back
        // (review of 2026-08-20, finding 3). `Starting` is the state that says
        // so, and this is the first line after it.
        inner.refresh_engine_residency().await;
        {
            let mut live = inner.live.lock().expect("capture state lock");
            live.meeting_id = None;
            live.started_at = None;
            live.elapsed_ms = 0;
            live.active_channels.clear();
            live.degraded_reason = None;
            // A new meeting is a new attempt at the engine: whatever went wrong
            // for the last one is not this meeting's news.
            live.speech_load_failed = false;
        }
        inner.pending.store(0, Ordering::SeqCst);
        inner.emit_state();

        // 1. The meeting row, before any device is touched, so a crash one
        //    millisecond from now still leaves something to recover.
        let title = match opts.title.as_deref().map(str::trim) {
            Some(t) if !t.is_empty() => t.to_string(),
            _ => default_title(opts.detected_app.as_deref()),
        };
        let meeting = match repo::create_meeting(
            &inner.db,
            &title,
            &inner.paths.storage_root().to_string_lossy(),
            opts.detected_app.as_deref(),
        )
        .await
        {
            Ok(meeting) => meeting,
            Err(error) => {
                inner.start_failed().await;
                return Err(error.into());
            }
        };

        let audio_dir = inner.paths.meeting_dir(&meeting.id);
        if let Err(error) = std::fs::create_dir_all(&audio_dir) {
            let _ = repo::set_meeting_status(&inner.db, &meeting.id, MeetingStatus::Failed).await;
            let _ = repo::soft_delete_meeting(&inner.db, &meeting.id).await;
            inner.start_failed().await;
            return Err(SessionError::Storage(format!(
                "{}: {error}",
                audio_dir.display()
            )));
        }
        // `create_meeting` mints the id, so the per-meeting directory can only be
        // recorded now. Chunk rows carry full paths, so this is provenance for
        // anyone reading the row directly.
        if let Err(error) =
            repo::set_meeting_audio_dir(&inner.db, &meeting.id, &audio_dir.to_string_lossy()).await
        {
            tracing::debug!(%error, "could not store the recording folder");
        }

        // 2. Open the sources and start writing chunks from t=0.
        let cfg = CaptureConfig {
            meeting_id: meeting.id.clone(),
            audio_dir,
            capture_system_audio: opts
                .capture_system_audio
                .unwrap_or(settings.capture_system_audio),
            input_device_id: opts.input_device_id.clone().or(settings.input_device_id),
        };
        let opened = inner.ports.capture.open(cfg).await;
        let CaptureStart {
            handle,
            started,
            feed,
        } = match opened {
            Ok(started) => started,
            Err(error) => {
                tracing::warn!(%error, "capture could not start");
                let _ =
                    repo::set_meeting_status(&inner.db, &meeting.id, MeetingStatus::Failed).await;
                let _ = repo::soft_delete_meeting(&inner.db, &meeting.id).await;
                inner.start_failed().await;
                return Err(session_error_for(error));
            }
        };

        if started.channels.is_empty() {
            let _ = handle.stop().await;
            let _ = repo::set_meeting_status(&inner.db, &meeting.id, MeetingStatus::Failed).await;
            let _ = repo::soft_delete_meeting(&inner.db, &meeting.id).await;
            inner.start_failed().await;
            return Err(SessionError::NoAudioSources(
                started
                    .system_audio_error
                    .unwrap_or_else(|| "no source produced audio".into()),
            ));
        }

        // 3. Recording has absolute priority over background work.
        if let Err(error) = inner.jobs.preempt().await {
            tracing::warn!(%error, "could not park background work");
        }

        {
            let mut live = inner.live.lock().expect("capture state lock");
            live.meeting_id = Some(meeting.id.clone());
            live.started_at = Some(meeting.started_at.clone());
            live.active_channels = started.channels.clone();
        }
        *inner.capture.lock().await = Some(handle);

        if let Err(error) =
            repo::set_meeting_status(&inner.db, &meeting.id, MeetingStatus::Recording).await
        {
            tracing::warn!(%error, "could not mark the meeting as recording");
        }
        inner.transition(&CaptureEvent::Started);

        // Got less than we wanted: keep what we have and say so
        // (DESIGN §3 Degraded).
        if let Some(detail) = started.system_audio_error.clone() {
            tracing::warn!(detail, "recording without what the computer plays");
            inner
                .note_degraded(
                    Some(Channel::System),
                    DegradedReason::SystemAudioUnavailable,
                    "Echo is recording you, but it can't hear what this computer plays.",
                    &meeting.id,
                )
                .await;
        }
        if let Some(detail) = started.microphone_error.clone() {
            tracing::warn!(detail, "recording without the microphone");
            inner
                .note_degraded(
                    Some(Channel::Mic),
                    DegradedReason::MicrophoneUnavailable,
                    "Echo is recording what this computer plays, but it can't hear you.",
                    &meeting.id,
                )
                .await;
        }
        if started.speech_detection_degraded {
            // Advanced diagnostics only: the person is not told, because there is
            // nothing for them to do and the recording is fine.
            tracing::info!("speech detection is running on the fallback");
        }

        inner.emit_state();
        inner
            .ports
            .events
            .emit(UiEvent::TrayState(TrayState::Recording));
        inner
            .ports
            .events
            .emit(UiEvent::MeetingUpdated(MeetingUpdatedPayload {
                meeting_id: meeting.id.clone(),
                status: MeetingStatus::Recording,
                title: Some(title),
                duration_ms: 0,
                deleted: false,
            }));

        // Live speaker labels: the microphone is "You", the rest is provisional.
        if let Err(error) = crate::diarize::ensure_channel_speakers(&inner.db, &meeting.id).await {
            tracing::debug!(%error, "provisional speaker labels are not ready");
        }

        // 4a. The live pipeline: journal chunks, write down what is said.
        //
        //     It is handed the split of the meeting clock it shares with the
        //     backlog pass ([`pipeline::Backlog`]). Empty for now, and for most
        //     recordings it stays that way: the engine is usually ready before
        //     the meeting is.
        let backlog = pipeline::Backlog::new();
        *inner.backlog.lock().expect("backlog lock") = Some(backlog.clone());
        let tasks = pipeline::spawn(inner.clone(), meeting.id.clone(), feed, backlog.clone());
        {
            let mut slot = inner.tasks.lock().expect("pipeline task lock");
            *slot = tasks;
        }

        // 4b. The speech engine, last, and off the critical path: failing to
        //     load it must not end a recording. The audio is already on disk.
        //
        //     It was told to stay on the first line of this function, before the
        //     row and the devices, so the weights cannot have gone anywhere in
        //     the meantime (mantra 1's amendment of 2026-08-20).
        //
        //     The task is *owned*, not detached: a stop waits for it before it
        //     queues the finalize job, so there is never a backlog pass still
        //     writing while the post-meeting one works out where the holes are
        //     (review of 2026-08-20, finding 1).
        let engine = inner.clone();
        let engine_meeting = meeting.id.clone();
        let task = tokio::spawn(async move {
            match engine.ports.asr.prewarm().await {
                // Ready — but possibly minutes after the meeting started, if the
                // weights had to be read or the graphics compiler had work to do.
                // Whatever was said in the meantime is on disk, so read it back
                // into the transcript before carrying on live.
                Ok(()) => {
                    // Before the backlog is read, not after it. Reading minutes
                    // of audio back can itself take a while, and the thing the
                    // banner is about — whether Echo can understand speech —
                    // became true the moment the load returned.
                    engine.emit_state();
                    pipeline::catch_up_backlog(engine, engine_meeting, backlog).await
                }
                Err(error) => {
                    tracing::warn!(%error, "speech understanding is not ready");
                    {
                        let mut live = engine.live.lock().expect("capture state lock");
                        live.speech_load_failed = true;
                    }
                    // The banner and the notice say the same thing, and they say
                    // it at the same moment: the notice is a message that can be
                    // missed, this is the state the screen can always ask for.
                    engine.emit_state();
                    engine.notice(NoticePayload {
                        level: NoticeLevel::Warning,
                        message: "Echo is recording, but it can't write the words down yet. It \
                                  will catch up as soon as it can."
                            .into(),
                        persistent: true,
                        meeting_id: Some(engine_meeting),
                        tag: Some("speechNotReady".into()),
                    });
                }
            }
        });
        *inner.backlog_task.lock().expect("backlog task lock") = Some(task);

        tracing::info!(meeting = %meeting.id, channels = ?started.channels, "recording");
        Ok(meeting.id)
    }

    /// Stop recording, commit every chunk, then queue catch-up, the speaker
    /// pass, the mixdown and the recap.
    ///
    /// A meeting that produced nothing worth keeping is deleted here rather than
    /// left in the list as a husk ([`recovery::meeting_has_content`]), in which
    /// case this returns `None` and the UI goes back Home instead of opening a
    /// meeting that no longer exists.
    ///
    /// Not recording returns `None`.
    pub async fn stop(&self) -> Result<Option<Id>, SessionError> {
        let inner = self.0.clone();
        let _command = inner.command.lock().await;

        let meeting_id = {
            let live = inner.live.lock().expect("capture state lock");
            if !(is_live(live.state) || matches!(live.state, CaptureState::Starting)) {
                return Ok(None);
            }
            match live.meeting_id.clone() {
                Some(id) => id,
                None => return Ok(None),
            }
        };

        inner.transition(&CaptureEvent::StopRequested);
        inner.emit_state();

        // Close the devices; the writer flushes and commits its open chunks.
        let handle = inner.capture.lock().await.take();
        let mut duration_ms = 0;
        if let Some(handle) = handle {
            duration_ms = handle.elapsed_ms();
            match handle.stop().await {
                Ok(ms) => duration_ms = duration_ms.max(ms),
                Err(error) => tracing::warn!(%error, "capture did not close cleanly"),
            }
        }

        // Let the pipeline write what it already has, but never wait forever.
        let tasks = {
            let mut slot = inner.tasks.lock().expect("pipeline task lock");
            std::mem::take(&mut *slot)
        };
        for task in tasks {
            if tokio::time::timeout(DRAIN_TIMEOUT, task).await.is_err() {
                tracing::warn!("the live pipeline took too long to finish; carrying on");
            }
        }
        // And the pass reading this meeting's start off disk, before anything
        // queues the one that reads the rest of it.
        inner.join_backlog_pass().await;

        // The journal is the truth about how long this meeting was.
        let committed = jobs::committed_end_ms(&inner.db, &meeting_id)
            .await
            .unwrap_or(0);
        let duration_ms = duration_ms.max(committed);
        let language = repo::language_histogram(&inner.db, &meeting_id)
            .await
            .ok()
            .and_then(|h| h.first().map(|(l, _)| l.clone()));

        repo::finish_meeting(
            &inner.db,
            &meeting_id,
            duration_ms,
            language.as_deref(),
            None,
            MeetingStatus::Processing,
        )
        .await?;

        // Catch-up → speakers → playback file → recap. Or, when there is nothing
        // to work on, no jobs at all and no meeting either.
        let outcome = recovery::queue_finalization(&inner, &meeting_id).await?;
        if let Err(error) = inner.jobs.release().await {
            tracing::warn!(%error, "could not pick background work back up");
        }

        let discarded = matches!(outcome, recovery::Finalized::Discarded);
        {
            let mut live = inner.live.lock().expect("capture state lock");
            live.elapsed_ms = if discarded { 0 } else { duration_ms };
            live.active_channels.clear();
            live.degraded_reason = None;
            if discarded {
                // Nothing left to point at.
                live.meeting_id = None;
                live.started_at = None;
            }
        }
        inner.pending.store(0, Ordering::SeqCst);
        inner.transition(&CaptureEvent::Stopped);
        inner.emit_state();
        inner.ports.events.emit(UiEvent::TrayState(TrayState::Idle));

        if discarded {
            // `queue_finalization` already told the UI the row is gone, and
            // there is no work outstanding for it — so this is where the engine
            // starts its grace period.
            inner.refresh_engine_residency().await;
            tracing::info!(meeting = %meeting_id, duration_ms, "an empty recording was let go");
            return Ok(None);
        }

        inner
            .ports
            .events
            .emit(UiEvent::MeetingUpdated(MeetingUpdatedPayload {
                meeting_id: meeting_id.clone(),
                status: MeetingStatus::Processing,
                title: None,
                duration_ms,
                deleted: false,
            }));

        // The meeting is over but its work is not: catch-up, the speaker pass
        // and the recap all still want the engine, so this holds it rather than
        // letting go (mantra 1's amendment). The grace period starts when the
        // last of those jobs finishes — see `JobRuntime::execute`.
        inner.refresh_engine_residency().await;
        tracing::info!(meeting = %meeting_id, duration_ms, "recording finished");
        Ok(Some(meeting_id))
    }

    /// Idempotent.
    pub async fn pause(&self) -> Result<CaptureStatus, SessionError> {
        let inner = self.0.clone();
        let _command = inner.command.lock().await;
        if !matches!(
            inner.state(),
            CaptureState::Recording | CaptureState::Degraded
        ) {
            return Ok(self.status().await);
        }
        {
            let guard = inner.capture.lock().await;
            if let Some(handle) = guard.as_ref() {
                if let Err(error) = handle.pause().await {
                    tracing::warn!(%error, "could not pause capture");
                }
            }
        }
        inner.transition(&CaptureEvent::PauseRequested);
        inner.emit_state();
        Ok(self.status().await)
    }

    /// Idempotent.
    pub async fn resume(&self) -> Result<CaptureStatus, SessionError> {
        let inner = self.0.clone();
        let _command = inner.command.lock().await;
        if !matches!(inner.state(), CaptureState::Paused) {
            return Ok(self.status().await);
        }
        {
            let guard = inner.capture.lock().await;
            if let Some(handle) = guard.as_ref() {
                if let Err(error) = handle.resume().await {
                    tracing::warn!(%error, "could not resume capture");
                }
            }
        }
        inner.transition(&CaptureEvent::ResumeRequested);
        // A recording that was degraded before the pause is still degraded.
        let reason = inner
            .live
            .lock()
            .expect("capture state lock")
            .degraded_reason;
        if let Some(
            reason @ (DegradedReason::SystemAudioUnavailable
            | DegradedReason::MicrophoneUnavailable),
        ) = reason
        {
            inner.transition(&CaptureEvent::SourceLost(reason));
        }
        inner.emit_state();
        Ok(self.status().await)
    }

    /// Current capture state. Cheap; the UI polls it on mount and then listens
    /// for events.
    pub async fn status(&self) -> CaptureStatus {
        let mut status = self.0.status_snapshot();
        let mut held_by_capture = 0;
        {
            let guard = self.0.capture.lock().await;
            if let Some(handle) = guard.as_ref() {
                status.elapsed_ms = handle.elapsed_ms().max(status.elapsed_ms);
                held_by_capture = handle.pending_utterances();
                let channels = handle.active_channels();
                if !channels.is_empty() {
                    status.active_channels = channels;
                }
            }
        }
        status.pending_utterances = self.0.pending.load(Ordering::SeqCst)
            + self.0.ports.asr.queue_depth()
            + held_by_capture;
        status
    }

    /// Drop a marker at the current position, for the "Flag action item" button.
    pub async fn add_marker(
        &self,
        kind: MarkerKind,
        note: Option<String>,
    ) -> Result<Marker, SessionError> {
        let (meeting_id, mut t_ms) = {
            let live = self.0.live.lock().expect("capture state lock");
            if !is_live(live.state) {
                return Err(SessionError::NotRecording);
            }
            match live.meeting_id.clone() {
                Some(id) => (id, live.elapsed_ms),
                None => return Err(SessionError::NotRecording),
            }
        };
        {
            let guard = self.0.capture.lock().await;
            if let Some(handle) = guard.as_ref() {
                t_ms = handle.elapsed_ms().max(t_ms);
            }
        }
        Ok(repo::insert_marker(&self.0.db, &meeting_id, t_ms, kind, note.as_deref()).await?)
    }

    /// Look for meetings that were recording when the app went away. Returns
    /// their ids so the UI can offer finish or discard. Idempotent.
    pub async fn find_interrupted(&self) -> Result<Vec<Id>, SessionError> {
        recovery::scan(&self.0).await
    }

    /// Quietly remove finished meetings with nothing in them — records from
    /// before the stop-time husk test existed. Once at launch.
    pub async fn tidy_empty_meetings(&self) -> u64 {
        recovery::sweep_husks(&self.0).await
    }

    /// Act on the person's choice for an interrupted meeting.
    pub async fn resolve_interrupted(
        &self,
        meeting_id: &str,
        action: RecoveryAction,
    ) -> Result<(), SessionError> {
        recovery::resolve(&self.0, meeting_id, action).await
    }

    /// "Listen again": throw this meeting's transcript away and write it again
    /// from the audio on disk.
    ///
    /// Turned down while a recording is live; a no-op while one is already
    /// running for this meeting, so a double-click cannot wipe what the running
    /// pass has already written. See [`retranscribe`].
    pub async fn retranscribe(
        &self,
        meeting_id: &str,
    ) -> Result<retranscribe::Retranscribed, SessionError> {
        let inner = self.0.clone();
        // The same lock start and stop take: a wipe must not land in the middle
        // of a recording starting.
        let _command = inner.command.lock().await;
        retranscribe::run(&inner, meeting_id).await
    }

    /// "There were four of us": store the person's own count of how many people
    /// were in this meeting and work out who said what again, cut to exactly
    /// that many voices. `None` puts it back to Echo's own count.
    ///
    /// The number includes whoever was at this computer. Turned down while this
    /// meeting is being recorded, and while the speaker pass or a listen again is
    /// already working on it; asking twice for the same number changes nothing.
    /// See [`speaker_count`].
    pub async fn set_speaker_count(
        &self,
        meeting_id: &str,
        count: Option<u32>,
    ) -> Result<speaker_count::SpeakerCountSet, SessionError> {
        let inner = self.0.clone();
        // The same lock start and stop take: the answer to "is this meeting being
        // recorded" must not change while it is being acted on.
        let _command = inner.command.lock().await;
        speaker_count::run(&inner, meeting_id, count).await
    }

    /// Queue background work. Deduplicated per meeting and kind.
    pub async fn queue_job(
        &self,
        meeting_id: Option<&str>,
        kind: JobKind,
    ) -> Result<Id, SessionError> {
        Ok(self.0.jobs.queue(meeting_id, kind).await?.id)
    }

    /// [`SessionManager::queue_job`], remembering the request behind it — the
    /// recap style and backend a "write this again" button picked, which would
    /// otherwise be lost by the time the job runs.
    pub async fn queue_job_with_payload<T: serde::Serialize>(
        &self,
        meeting_id: Option<&str>,
        kind: JobKind,
        payload: &T,
    ) -> Result<Id, SessionError> {
        let encoded = serde_json::to_string(payload).ok();
        Ok(self
            .0
            .jobs
            .queue_with_payload(meeting_id, kind, encoded.as_deref())
            .await?
            .id)
    }

    /// Cancel a job. Already-finished is success.
    pub async fn cancel_job(&self, job_id: &str) -> Result<(), SessionError> {
        Ok(self.0.jobs.cancel(job_id).await?)
    }

    /// Retry a failed job from where it stopped.
    pub async fn retry_job(&self, job_id: &str) -> Result<(), SessionError> {
        Ok(self.0.jobs.retry(job_id).await?)
    }

    /// Start the loop that drains the jobs table. Called once at launch;
    /// calling it again does nothing.
    pub async fn start_job_runner(&self) -> Result<(), SessionError> {
        self.retry_failed_speech_setup().await;
        // A crash leaves rows marked running, and a crash *during a recording*
        // leaves them parked. Neither can still be true, so both go back in the
        // queue with their progress intact — except a setup row the process died
        // inside, which is settled as failed instead so the compile that took
        // the app down is not started again at this launch and every one after.
        if let Err(error) = repo::requeue_orphaned_jobs(&self.0.db, jobs::SETUP_INTERRUPTED).await {
            tracing::warn!(%error, "could not tidy the work queue");
        }
        // Nothing owns the machine at launch. Saying so out loud clears the
        // in-memory flag and un-parks anything the requeue above did not catch,
        // so no meeting can be left waiting forever for a recording that ended
        // when the process died.
        if let Err(error) = self.0.jobs.release().await {
            tracing::warn!(%error, "could not pick parked work back up");
        }
        self.0.jobs.start();
        Ok(())
    }

    /// Give the one-time speech setup another go, once, at launch.
    ///
    /// The whole automatic escape hatch from the one-attempt policy, and the
    /// reason a single bad minute cannot leave a machine unwarmed for good.
    ///
    /// A setup row that failed without ever reaching the compile took its record
    /// of the attempt back with it (`jobs::prepare_engine`), so the weights still
    /// owe this machine a setup and the row is worth another go — and a launch
    /// is the right distance to try from: far enough apart to be a different
    /// day, a different disk and a different amount of free memory, where asking
    /// again on the readiness polls would be a hundred identical failures a
    /// minute. When the attempt *did* reach the compile, the marker written
    /// before the load is still standing, this reads as "nothing owed", and the
    /// row stays failed — which is the whole point of writing it first.
    async fn retry_failed_speech_setup(&self) {
        match crate::asr::models::warm_up_needed(&self.0.db).await {
            Ok(Some(file_name)) => {
                let Ok(payload) = serde_json::to_string(&file_name) else {
                    return;
                };
                match repo::requeue_failed_setup_jobs(&self.0.db, &payload).await {
                    Ok(n) if n > 0 => tracing::info!(
                        rows = n,
                        model = %file_name,
                        "giving the one-time speech setup another go"
                    ),
                    Ok(_) => {}
                    Err(error) => tracing::warn!(%error, "could not pick the speech setup back up"),
                }
            }
            Ok(None) => {}
            Err(error) => tracing::debug!(%error, "could not tell whether speech needs setting up"),
        }
    }

    /// Make sure Echo has the speech model it wants, and get rid of any it does
    /// not. The one entry point for the self-healing described in
    /// [`crate::asr::reconcile`].
    ///
    /// Called at launch and every time the UI asks whether Echo can understand
    /// speech, so it has to be cheap when there is nothing to do (a `stat` per
    /// catalogued asset and one settings read) and safe to call at any moment.
    ///
    /// Three things can come out of it, at most one of them visible:
    ///
    /// * something wanted is missing → a download job is queued, and the
    ///   progress pill the app already has shows it. Recording is not blocked:
    ///   whatever is on disk keeps serving.
    /// * every wanted file has arrived and older weights are still there → they
    ///   are deleted and the person is told once, quietly.
    /// * nothing to do → silence, which is the common case.
    ///
    /// Returns whether Echo can transcribe right now, which is the only part a
    /// caller has ever needed.
    pub async fn ensure_speech_current(&self) -> bool {
        let plan = match crate::asr::models::plan_reconcile(&self.0.db, &self.0.paths).await {
            Ok(plan) => plan,
            Err(error) => {
                // A reconcile that cannot read the disk changes nothing and
                // blocks nothing: whatever was installed a moment ago still is.
                tracing::warn!(%error, "could not check what Echo has to understand speech with");
                return false;
            }
        };

        // Never take weights away from a meeting in progress. Unlinking a file
        // the engine already has open is survivable, but the engine reloads on
        // its own after three failed decodes, and finding the weights gone at
        // that moment would cost the rest of the meeting's live text. The files
        // are only redundant — they can wait until the meeting is over, which is
        // the next time anything asks.
        let recording = engine_is_needed_by_capture(self.0.state());
        if recording && plan.switching() {
            tracing::info!("leaving older speech weights in place until this meeting is finished");
            return plan.can_serve();
        }
        crate::asr::models::apply_cleanup(&self.0.db, &self.0.paths, &plan).await;

        if plan.switching() {
            // The engine picks the new weights up on its next configure, which
            // is why the old files could go: `installed_path` now resolves to
            // the new model and the old one is not on disk to be loaded.
            tracing::info!(
                serving = plan.serving.unwrap_or("?"),
                "Echo switched to the speech model it wants"
            );
            self.0.notice(NoticePayload {
                level: NoticeLevel::Info,
                message: "Echo upgraded how it understands speech.".into(),
                persistent: false,
                meeting_id: None,
                // Tagged so a second launch that somehow re-observes the switch
                // replaces the notice rather than stacking a duplicate.
                tag: Some("speechUpgraded".into()),
            });
        }

        if !plan.missing.is_empty() {
            // `queue` deduplicates per kind, so a download already queued or
            // running is joined rather than started twice — this can be called
            // several times a second by a UI that is watching readiness.
            match self
                .queue_job_with_payload(None, JobKind::Download, &plan.level_id.to_string())
                .await
            {
                Ok(_) => tracing::info!(
                    missing = plan.missing.len(),
                    bytes = plan.remaining_bytes(),
                    "queued the download of what Echo still needs"
                ),
                Err(error) => tracing::warn!(%error, "could not queue the download"),
            }
        }

        // Everything is here and nothing is being recorded: the moment to pay
        // the one-time setup these weights need on this machine, if it has never
        // been paid. Weights that arrived without a download job — a restore, a
        // copy from another machine, an out-of-band unpack — have nobody else to
        // pay it, and on 2026-08-24 that bill was settled by a real meeting,
        // sixteen minutes of it, with nothing on screen to explain the silence.
        //
        // Not while recording: the meeting in progress is already loading the
        // engine it needs, and a second reason to do the same thing helps
        // nobody. `queue` deduplicates per kind, so the readiness polls that run
        // several times a second join the row that exists rather than stacking
        // rows; once the job has run, the marker it writes is what stops this.
        if plan.can_serve() && plan.missing.is_empty() && !recording {
            match crate::asr::models::warm_up_needed(&self.0.db).await {
                Ok(Some(file_name)) => {
                    if self.setup_already_failed_for(&file_name).await {
                        // Tried, and it did not come off. Asking again here would
                        // ask again a few milliseconds later too, and a few
                        // milliseconds after that, for as long as the app is
                        // open. That row gets its next go at the next launch
                        // (`start_job_runner`), which is far enough away to be a
                        // different day and a different disk.
                    } else {
                        match self
                            .queue_job_with_payload(None, JobKind::PrepareEngine, &file_name)
                            .await
                        {
                            Ok(_) => tracing::info!(
                                model = %file_name,
                                "queued the one-time setup these speech weights need on this machine"
                            ),
                            Err(error) => {
                                tracing::warn!(%error, "could not queue the one-time speech setup")
                            }
                        }
                    }
                }
                Ok(None) => {}
                Err(error) => tracing::debug!(
                    %error,
                    "could not tell whether the speech weights still need setting up"
                ),
            }
        }

        plan.can_serve()
    }

    /// Whether the one-time setup for exactly these weights has already been
    /// asked for and failed.
    ///
    /// A failed attempt that never reached the compile takes its marker back
    /// with it, so [`crate::asr::models::warm_up_needed`] starts saying "still
    /// owed" again the moment it ends — right, but not a reason to try again
    /// this second. The failed row is the memory of the attempt for the rest of
    /// the launch; a fresh one is queued for weights nobody has tried yet, so a
    /// new model is never held back by an old model's bad day.
    ///
    /// Trouble reading the queue counts as "already failed": every other answer
    /// leads to queueing a job, and a database that cannot answer this would
    /// have it queued again on the next poll, and the next.
    async fn setup_already_failed_for(&self, file_name: &str) -> bool {
        let Ok(payload) = serde_json::to_string(file_name) else {
            return true;
        };
        match repo::has_failed_job_with_payload(&self.0.db, JobKind::PrepareEngine, &payload).await
        {
            Ok(failed) => failed,
            Err(error) => {
                tracing::debug!(%error, "could not read the work queue");
                true
            }
        }
    }

    /// Load speech understanding now, because the person asked. Nothing loads on
    /// its own (mantra 1).
    pub async fn prewarm_speech(&self) -> Result<(), crate::asr::AsrError> {
        self.0.ports.asr.prewarm().await
    }

    /// Release the speech engine and any idle resources (mantra 1). Called
    /// before quitting, and by the "give the memory back now" action in
    /// Settings → Advanced.
    ///
    /// Explicit, so it does not wait out the grace period — but it never takes
    /// the engine away from a meeting: not while capture is running, and not
    /// while a meeting's post-meeting work is still queued or running.
    pub async fn release_idle_resources(&self) {
        if engine_is_needed_by_capture(self.0.state()) {
            tracing::debug!("not releasing anything: a recording is running");
            return;
        }
        if jobs::outstanding_meeting_jobs(&self.0.db).await > 0 {
            tracing::debug!("not releasing anything: a meeting is still being worked on");
            return;
        }
        self.0.ports.asr.hold_resident(false);
        self.0.ports.asr.release().await;
    }

    /// True while the engine holds weights in memory. Settings → Advanced only.
    pub fn speech_loaded(&self) -> bool {
        self.0.ports.asr.is_loaded()
    }

    /// What the speech engine actually initialised on. Settings → Advanced only;
    /// `None` until something has asked it to load.
    pub fn speech_backend(&self) -> Option<crate::asr::engine::BackendReport> {
        self.0.ports.asr.backend()
    }

    /// Utterances the engine still has queued. Shown only as "catching up".
    pub fn speech_queue_depth(&self) -> u32 {
        self.0.ports.asr.queue_depth()
    }

    /// A meeting was deleted: drop anything queued for it rather than
    /// transcribing audio that is on its way to the bin.
    pub fn forget_meeting(&self, meeting_id: &str) {
        self.0.ports.asr.forget_meeting(meeting_id);
    }

    /// [`SessionManager::forget_meeting`], and stop the background work too.
    ///
    /// Deleting the row would cascade the `jobs` rows away, but a job that is
    /// *running* holds only the id and would carry on reading files that are
    /// about to disappear. Cancelling first means it stops at its next
    /// checkpoint instead.
    pub async fn drop_work_for_meeting(&self, meeting_id: &str) {
        self.0.ports.asr.forget_meeting(meeting_id);
        self.0.cancel_jobs_for(meeting_id).await;
    }

    /// On the way out: end a recording cleanly, park the queue, free memory.
    pub async fn shutdown(&self) {
        if is_live(self.0.state()) {
            if let Err(error) = self.stop().await {
                tracing::warn!(%error, "the recording did not close cleanly");
            }
        }
        self.0.jobs.shutdown();
        self.0.ports.asr.shutdown().await;
    }
}

/// Audio problems the person can act on keep their meaning; the rest becomes
/// "Echo couldn't hear anything".
fn session_error_for(error: AudioError) -> SessionError {
    match error {
        AudioError::Write(detail) => SessionError::Storage(detail),
        AudioError::PermissionDenied => {
            SessionError::NoAudioSources("microphone permission was not granted".into())
        }
        AudioError::NoInputDevice => SessionError::NoAudioSources("no input device".into()),
        other => SessionError::NoAudioSources(other.to_string()),
    }
}

/// Machine tag on a degraded-capture banner, so the UI can replace rather than
/// stack them.
fn degraded_tag(reason: DegradedReason) -> &'static str {
    match reason {
        DegradedReason::SystemAudioUnavailable => "systemAudioLost",
        DegradedReason::MicrophoneUnavailable => "microphoneLost",
        DegradedReason::TranscriptBehind => "transcriptBehind",
        DegradedReason::StorageLow => "storageLow",
    }
}

/// "Zoom — 19 Aug, 14:32". Names it from the date and, when we know it, the app
/// that made us suggest recording (DESIGN §3 StartRecordingOptions).
fn default_title(detected_app: Option<&str>) -> String {
    let when = chrono::Local::now().format("%-d %b, %H:%M");
    match detected_app.map(str::trim).filter(|a| !a.is_empty()) {
        Some(app) => format!("{} — {when}", friendly_app_name(app)),
        None => format!("Meeting — {when}"),
    }
}

/// Process names are not something a person should have to read.
fn friendly_app_name(app: &str) -> String {
    let lower = app.to_lowercase();
    for (needle, name) in [
        ("zoom", "Zoom"),
        ("teams", "Microsoft Teams"),
        ("webex", "Webex"),
        ("discord", "Discord"),
        ("slack", "Slack"),
        ("meet", "Google Meet"),
    ] {
        if lower.contains(needle) {
            return name.to_string();
        }
    }
    let trimmed = app.trim_end_matches(".exe").trim_end_matches(".us");
    let mut chars = trimmed.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => "Meeting".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::mock::Harness;
    use super::*;
    use crate::audio::vad::Utterance;
    use crate::audio::writer::CommittedChunk;
    use crate::types::{JobQuery, JobStatus, TranscriptQuery};

    // -----------------------------------------------------------------------
    // The state machine on its own
    // -----------------------------------------------------------------------

    #[test]
    fn a_normal_recording_walks_the_expected_path() {
        let mut s = CaptureState::Idle;
        for (event, expected) in [
            (CaptureEvent::StartRequested, CaptureState::Starting),
            (CaptureEvent::Started, CaptureState::Recording),
            (CaptureEvent::PauseRequested, CaptureState::Paused),
            (CaptureEvent::ResumeRequested, CaptureState::Recording),
            (CaptureEvent::StopRequested, CaptureState::Stopping),
            (CaptureEvent::Stopped, CaptureState::Stopped),
        ] {
            s = next_state(s, &event).unwrap_or_else(|| panic!("{event:?} was ignored in {s:?}"));
            assert_eq!(s, expected);
        }
    }

    #[test]
    fn double_clicking_start_and_stop_changes_nothing() {
        assert_eq!(
            next_state(CaptureState::Recording, &CaptureEvent::StartRequested),
            None
        );
        assert_eq!(
            next_state(CaptureState::Starting, &CaptureEvent::StartRequested),
            None
        );
        assert_eq!(
            next_state(CaptureState::Stopping, &CaptureEvent::StopRequested),
            None
        );
        assert_eq!(
            next_state(CaptureState::Idle, &CaptureEvent::StopRequested),
            None
        );
        assert_eq!(
            next_state(CaptureState::Paused, &CaptureEvent::PauseRequested),
            None
        );
    }

    #[test]
    fn losing_one_source_degrades_and_getting_it_back_recovers() {
        let degraded = next_state(
            CaptureState::Recording,
            &CaptureEvent::SourceLost(DegradedReason::SystemAudioUnavailable),
        )
        .unwrap();
        assert_eq!(degraded, CaptureState::Degraded);
        // A second loss does not change anything.
        assert_eq!(
            next_state(
                degraded,
                &CaptureEvent::SourceLost(DegradedReason::StorageLow)
            ),
            None
        );
        assert_eq!(
            next_state(degraded, &CaptureEvent::SourceRecovered),
            Some(CaptureState::Recording)
        );
    }

    #[test]
    fn a_degraded_recording_can_still_be_paused_and_stopped() {
        assert_eq!(
            next_state(CaptureState::Degraded, &CaptureEvent::PauseRequested),
            Some(CaptureState::Paused)
        );
        assert_eq!(
            next_state(CaptureState::Degraded, &CaptureEvent::StopRequested),
            Some(CaptureState::Stopping)
        );
    }

    #[test]
    fn failure_is_reachable_from_anywhere() {
        for state in [
            CaptureState::Idle,
            CaptureState::Starting,
            CaptureState::Recording,
            CaptureState::Degraded,
            CaptureState::Paused,
            CaptureState::Stopping,
        ] {
            assert_eq!(
                next_state(state, &CaptureEvent::Failed),
                Some(CaptureState::Failed)
            );
        }
    }

    #[test]
    fn recovery_can_be_resolved_or_jumped_straight_into_a_recording() {
        let recovering = next_state(CaptureState::Idle, &CaptureEvent::InterruptedFound).unwrap();
        assert_eq!(recovering, CaptureState::Recovering);
        assert_eq!(
            next_state(recovering, &CaptureEvent::RecoveryResolved),
            Some(CaptureState::Idle)
        );
        assert_eq!(
            next_state(recovering, &CaptureEvent::StartRequested),
            Some(CaptureState::Starting)
        );
    }

    #[test]
    fn start_is_allowed_after_a_failure() {
        assert!(can_start(CaptureState::Failed));
        assert!(can_start(CaptureState::Stopped));
        assert!(can_start(CaptureState::Recovering));
        assert!(!can_start(CaptureState::Recording));
        assert!(!can_start(CaptureState::Stopping));
    }

    #[test]
    fn liveness_covers_paused_because_the_devices_are_still_open() {
        assert!(is_live(CaptureState::Recording));
        assert!(is_live(CaptureState::Degraded));
        assert!(is_live(CaptureState::Paused));
        assert!(!is_live(CaptureState::Stopping));
        assert!(!is_live(CaptureState::Idle));
    }

    #[test]
    fn meetings_are_named_without_making_the_person_read_a_process_name() {
        assert!(default_title(Some("zoom.us")).starts_with("Zoom — "));
        assert!(default_title(Some("Microsoft Teams")).starts_with("Microsoft Teams — "));
        assert!(default_title(None).starts_with("Meeting — "));
        assert!(default_title(Some("  ")).starts_with("Meeting — "));
    }

    // -----------------------------------------------------------------------
    // Orchestration, with the audio and speech seams mocked
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn idle_holds_nothing_open() {
        let h = Harness::new().await;
        let status = h.session.status().await;
        assert_eq!(status.state, CaptureState::Idle);
        assert!(status.meeting_id.is_none());
        assert!(status.active_channels.is_empty());
        assert!(!h.session.speech_loaded());
        assert_eq!(h.capture.open_count(), 0);
    }

    /// Mantra 1 at the seam `lib.rs` actually uses: starting the job runner is
    /// what launch does, and it must not pull anything into memory or open a
    /// device on its own.
    #[tokio::test]
    async fn starting_the_job_runner_loads_nothing_and_opens_nothing() {
        let h = Harness::new().await;
        h.session.start_job_runner().await.unwrap();
        // Give the loop a chance to take its first look at the empty table.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        assert!(
            !h.session.speech_loaded(),
            "launch must not put weights in memory"
        );
        assert_eq!(
            h.capture.open_count(),
            0,
            "launch must not open an audio device"
        );
        assert_eq!(h.session.status().await.state, CaptureState::Idle);
        // Starting it twice is what a second launch path would do.
        h.session.start_job_runner().await.unwrap();
        assert!(!h.session.speech_loaded());
    }

    #[tokio::test]
    async fn starting_writes_the_meeting_row_before_it_opens_a_device() {
        let h = Harness::new().await;
        h.capture.fail_with(AudioError::NoInputDevice);

        let error = h.session.start(Default::default()).await.unwrap_err();
        assert!(matches!(error, SessionError::NoAudioSources(_)));
        assert_eq!(h.session.status().await.state, CaptureState::Failed);

        // The row exists even though no device ever opened, which is what makes
        // recovery possible at all.
        let visible = repo::list_meetings(&h.db, &Default::default())
            .await
            .unwrap();
        assert!(
            visible.is_empty(),
            "a start that never happened is not offered as a meeting"
        );
        let all = repo::list_meetings(&h.db, &all_meetings()).await.unwrap();
        assert_eq!(all.len(), 1, "the row is kept, only hidden");
        assert_eq!(all[0].status, MeetingStatus::Failed);
    }

    #[tokio::test]
    async fn a_second_start_returns_the_recording_already_in_progress() {
        let h = Harness::new().await;
        let first = h.session.start(Default::default()).await.unwrap();
        let second = h.session.start(Default::default()).await.unwrap();
        assert_eq!(first, second);
        assert_eq!(h.capture.open_count(), 1, "no second device was opened");
        assert_eq!(
            repo::list_meetings(&h.db, &all_meetings())
                .await
                .unwrap()
                .len(),
            1
        );
        assert_eq!(h.session.status().await.state, CaptureState::Recording);
    }

    #[tokio::test]
    async fn losing_the_computers_audio_degrades_and_banners() {
        let h = Harness::new().await;
        h.capture.without_system_audio("no permission");
        let id = h.session.start(Default::default()).await.unwrap();

        let status = h.session.status().await;
        assert_eq!(status.state, CaptureState::Degraded);
        assert_eq!(
            status.degraded_reason,
            Some(DegradedReason::SystemAudioUnavailable)
        );
        assert!(h.events.notice_tagged("systemAudioLost"));
        // Still a normal recording otherwise.
        assert_eq!(status.meeting_id.as_deref(), Some(id.as_str()));
        h.record_a_minute(&id).await;
        assert!(h.session.stop().await.unwrap().is_some());
    }

    #[tokio::test]
    async fn a_source_lost_mid_meeting_degrades_then_recovers() {
        let h = Harness::new().await;
        h.session.start(Default::default()).await.unwrap();

        h.capture.send(ports::CaptureSignal::Degraded {
            channel: Some(Channel::System),
            reason: DegradedReason::SystemAudioUnavailable,
            message: "Echo stopped hearing what this computer plays.".into(),
        });
        h.settle().await;
        assert_eq!(h.session.status().await.state, CaptureState::Degraded);
        assert!(h.events.notice_tagged("systemAudioLost"));

        h.capture.send(ports::CaptureSignal::Recovered {
            channel: Channel::System,
        });
        h.settle().await;
        assert_eq!(h.session.status().await.state, CaptureState::Recording);
    }

    #[tokio::test]
    async fn pause_and_resume_are_idempotent() {
        let h = Harness::new().await;
        h.session.start(Default::default()).await.unwrap();

        assert_eq!(h.session.pause().await.unwrap().state, CaptureState::Paused);
        assert_eq!(
            h.session.pause().await.unwrap().state,
            CaptureState::Paused,
            "a second pause is the first one"
        );
        assert_eq!(
            h.session.resume().await.unwrap().state,
            CaptureState::Recording
        );
        assert_eq!(
            h.session.resume().await.unwrap().state,
            CaptureState::Recording
        );
    }

    #[tokio::test]
    async fn stopping_while_idle_is_a_no_op() {
        let h = Harness::new().await;
        assert!(h.session.stop().await.unwrap().is_none());
        assert_eq!(h.session.status().await.state, CaptureState::Idle);
    }

    #[tokio::test]
    async fn a_recording_journals_its_audio_and_writes_down_what_was_said() {
        let h = Harness::new().await;
        let id = h.session.start(Default::default()).await.unwrap();

        h.capture
            .send(ports::CaptureSignal::ChunkCommitted(CommittedChunk {
                channel: Channel::Mic,
                seq: 0,
                path: h.paths.chunk_path(&id, Channel::Mic, 0),
                t_start_ms: 0,
                t_end_ms: 30_000,
            }));
        h.capture
            .send(ports::CaptureSignal::UtteranceReady(Utterance {
                channel: Channel::Mic,
                t_start_ms: 1_000,
                t_end_ms: 3_000,
                samples: vec![0.0; 16_000],
                truncated: false,
            }));

        let stopped = h.session.stop().await.unwrap();
        assert_eq!(stopped.as_deref(), Some(id.as_str()));

        let chunks = repo::list_chunks(&h.db, &id, None).await.unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].committed, "a journalled chunk counts as on disk");

        let segments = repo::get_segments(
            &h.db,
            &TranscriptQuery {
                meeting_id: id.clone(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "hello there");
        assert!(segments[0].is_final);
        assert!(
            segments[0].speaker_id.is_some(),
            "the microphone is a speaker"
        );

        let meeting = repo::get_meeting(&h.db, &id).await.unwrap().unwrap();
        assert_eq!(meeting.status, MeetingStatus::Processing);
        assert_eq!(meeting.duration_ms, 30_000);
    }

    #[tokio::test]
    async fn stopping_queues_the_finishing_work_in_order() {
        let h = Harness::new().await;
        let id = h.session.start(Default::default()).await.unwrap();
        h.record_a_minute(&id).await;
        h.session.stop().await.unwrap();

        let queued = h.queued_kinds(&id).await;
        assert!(queued.contains(&JobKind::TranscribeCatchup));
        assert!(queued.contains(&JobKind::Diarize));
        assert!(queued.contains(&JobKind::Mixdown));
        assert!(
            queued.contains(&JobKind::Summarize),
            "the recap is written on its own by default"
        );

        // The queue hands them out catch-up first.
        let next = repo::next_queued_job(&h.db).await.unwrap().unwrap();
        assert_eq!(next.kind, JobKind::TranscribeCatchup);
    }

    #[tokio::test]
    async fn no_recap_is_queued_when_the_person_turned_that_off() {
        let h = Harness::new().await;
        crate::settings::apply(
            &h.db,
            &crate::types::SettingsPatch {
                auto_summarize: Some(false),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        let id = h.session.start(Default::default()).await.unwrap();
        h.record_a_minute(&id).await;
        h.session.stop().await.unwrap();

        let kinds = h.queued_kinds(&id).await;
        assert!(kinds.contains(&JobKind::TranscribeCatchup));
        assert!(
            !kinds.contains(&JobKind::Summarize),
            "a stored 'no' is respected"
        );
    }

    // -----------------------------------------------------------------------
    // Nothing recorded, nothing kept
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn a_start_and_an_immediate_stop_leaves_no_meeting_behind() {
        let h = Harness::new().await;
        let id = h.session.start(Default::default()).await.unwrap();
        let dir = h.paths.meeting_dir(&id);
        assert!(dir.exists());

        // No chunk was ever committed and nothing was said.
        assert!(
            h.session.stop().await.unwrap().is_none(),
            "there is no meeting to open"
        );

        assert!(repo::get_meeting(&h.db, &id).await.unwrap().is_none());
        assert!(repo::list_meetings(&h.db, &all_meetings())
            .await
            .unwrap()
            .is_empty());
        assert!(
            h.queued_kinds(&id).await.is_empty(),
            "no work is queued for a meeting that is gone"
        );
        assert!(!dir.exists(), "its folder went too");
        assert!(h.events.notice_tagged("nothingToKeep"));
        assert_eq!(h.session.status().await.state, CaptureState::Stopped);
        assert!(h.session.status().await.meeting_id.is_none());
    }

    /// The journal is bookkeeping; the files are the truth. A stop that could not
    /// write its chunk rows in time must not cost someone their meeting.
    #[tokio::test]
    async fn audio_on_disk_with_no_journal_rows_survives_a_stop() {
        let h = Harness::new().await;
        let id = h.session.start(Default::default()).await.unwrap();
        let dir = h.paths.meeting_dir(&id);
        {
            let mut writer = crate::audio::writer::ChunkWriter::create(
                &dir,
                Channel::Mic,
                crate::audio::TARGET_SAMPLE_RATE,
            )
            .unwrap();
            writer
                .write_samples(&vec![0.2f32; crate::audio::TARGET_SAMPLE_RATE as usize * 8])
                .unwrap();
            writer.finish().unwrap();
        }
        assert!(repo::list_chunks(&h.db, &id, None)
            .await
            .unwrap()
            .is_empty());

        assert_eq!(
            h.session.stop().await.unwrap().as_deref(),
            Some(id.as_str())
        );
        assert!(repo::get_meeting(&h.db, &id).await.unwrap().is_some());
        assert!(dir.exists());
    }

    #[tokio::test]
    async fn a_couple_of_seconds_of_silence_is_not_a_meeting() {
        let h = Harness::new().await;
        let id = h.session.start(Default::default()).await.unwrap();
        h.commit_chunk(&id, 0, 1_200).await;

        assert!(h.session.stop().await.unwrap().is_none());
        assert!(repo::get_meeting(&h.db, &id).await.unwrap().is_none());
    }

    /// Mantra 3: audio on disk *is* content, even before anything has read it.
    #[tokio::test]
    async fn audio_nobody_has_transcribed_yet_is_never_thrown_away() {
        let h = Harness::new().await;
        let id = h.session.start(Default::default()).await.unwrap();
        h.commit_chunk(&id, 0, 45_000).await;

        assert_eq!(
            h.session.stop().await.unwrap().as_deref(),
            Some(id.as_str())
        );
        let kept = repo::get_meeting(&h.db, &id).await.unwrap().unwrap();
        assert_eq!(kept.status, MeetingStatus::Processing);
        assert!(kept.deleted_at.is_none());
        assert!(h
            .queued_kinds(&id)
            .await
            .contains(&JobKind::TranscribeCatchup));
        assert_eq!(
            repo::count_final_segments(&h.db, &id).await.unwrap(),
            0,
            "kept on the strength of the audio alone"
        );
    }

    /// A very short recording that produced words is a real, if brief, meeting.
    #[tokio::test]
    async fn a_short_meeting_with_words_in_it_is_kept() {
        let h = Harness::new().await;
        let id = h.session.start(Default::default()).await.unwrap();
        h.capture
            .send(ports::CaptureSignal::UtteranceReady(Utterance {
                channel: Channel::Mic,
                t_start_ms: 200,
                t_end_ms: 1_400,
                samples: vec![0.0; 16_000],
                truncated: false,
            }));
        h.settle().await;
        h.commit_chunk(&id, 0, 1_500).await;

        assert_eq!(
            h.session.stop().await.unwrap().as_deref(),
            Some(id.as_str())
        );
        assert!(repo::get_meeting(&h.db, &id).await.unwrap().is_some());
        assert!(repo::count_final_segments(&h.db, &id).await.unwrap() > 0);
    }

    #[tokio::test]
    async fn finishing_an_interrupted_meeting_with_nothing_in_it_bins_it() {
        let h = Harness::new().await;
        let meeting = repo::create_meeting(&h.db, "A false start", "/tmp", None)
            .await
            .unwrap();
        repo::set_meeting_status(&h.db, &meeting.id, MeetingStatus::Recording)
            .await
            .unwrap();
        // Just under a second of audio: enough for the scan to offer it, not
        // enough to be worth keeping.
        let chunk = repo::insert_chunk(
            &h.db,
            &meeting.id,
            Channel::Mic,
            0,
            "/tmp/mic-0.flac",
            0,
            900,
        )
        .await
        .unwrap();
        repo::commit_chunk(&h.db, &chunk, 900).await.unwrap();

        assert_eq!(
            h.session.find_interrupted().await.unwrap(),
            vec![meeting.id.clone()]
        );
        h.session
            .resolve_interrupted(&meeting.id, RecoveryAction::Finish)
            .await
            .unwrap();

        assert!(repo::get_meeting(&h.db, &meeting.id)
            .await
            .unwrap()
            .is_none());
        assert!(h.queued_kinds(&meeting.id).await.is_empty());
        assert_eq!(h.session.status().await.state, CaptureState::Idle);
    }

    #[tokio::test]
    async fn flagging_an_action_item_needs_a_live_meeting() {
        let h = Harness::new().await;
        assert!(matches!(
            h.session.add_marker(MarkerKind::ActionItem, None).await,
            Err(SessionError::NotRecording)
        ));

        let id = h.session.start(Default::default()).await.unwrap();
        h.capture.set_elapsed(42_000);
        let marker = h
            .session
            .add_marker(MarkerKind::ActionItem, Some("chase the invoice".into()))
            .await
            .unwrap();
        assert_eq!(marker.t_ms, 42_000);
        assert_eq!(repo::list_markers(&h.db, &id).await.unwrap().len(), 1);
    }

    // -----------------------------------------------------------------------
    // The speech engine's lifecycle: resident while listening
    // -----------------------------------------------------------------------

    /// Mantra 1's amendment of 2026-08-20, as a rule with no thread and no
    /// weights in it.
    #[test]
    fn the_engine_is_held_while_a_meeting_needs_it_and_not_a_moment_longer() {
        // Anything from the click to the last committed chunk.
        assert!(engine_is_needed_by_capture(CaptureState::Starting));
        assert!(engine_is_needed_by_capture(CaptureState::Recording));
        assert!(engine_is_needed_by_capture(CaptureState::Degraded));
        assert!(engine_is_needed_by_capture(CaptureState::Paused));
        assert!(engine_is_needed_by_capture(CaptureState::Stopping));
        assert!(!engine_is_needed_by_capture(CaptureState::Idle));
        assert!(!engine_is_needed_by_capture(CaptureState::Stopped));

        // Held while capturing, whether or not there is other work.
        assert!(engine_stays_resident(true, 0));
        assert!(engine_stays_resident(true, 3));
        // Held between the stop and the last of the meeting's jobs.
        assert!(engine_stays_resident(false, 1));
        // And only then does the grace period start.
        assert!(!engine_stays_resident(false, 0));
    }

    /// The same predicate guards the model cleanup, and for the same reason: an
    /// upgrade must never reach into a meeting that is happening. See
    /// `SessionManager::ensure_speech_current`.
    #[test]
    fn superseded_weights_are_not_deleted_out_from_under_a_live_meeting() {
        for state in [
            CaptureState::Starting,
            CaptureState::Recording,
            CaptureState::Degraded,
            CaptureState::Paused,
            CaptureState::Stopping,
        ] {
            assert!(
                engine_is_needed_by_capture(state),
                "{state:?} would let the cleanup run during a meeting"
            );
        }
        // Once nothing is being listened to, the leftovers can go.
        assert!(!engine_is_needed_by_capture(CaptureState::Idle));
        assert!(!engine_is_needed_by_capture(CaptureState::Stopped));
    }

    #[tokio::test]
    async fn a_recording_holds_the_engine_through_the_work_that_follows_it() {
        let h = Harness::new().await;
        h.session.start(Default::default()).await.unwrap();
        h.settle().await;
        assert!(h.session.speech_loaded(), "a recording loads it");
        assert!(h.asr.is_resident(), "and holds it while it is listening");

        h.capture.set_elapsed(60_000);
        h.record_a_minute(h.session.status().await.meeting_id.as_deref().unwrap())
            .await;
        h.session.stop().await.unwrap();
        h.settle().await;

        // Catch-up, the speaker pass, the playback file and the recap are all
        // queued and none of them has run: the meeting is not over yet.
        assert!(
            h.asr.is_resident(),
            "the engine waits for the last of the meeting's work"
        );
        h.session.release_idle_resources().await;
        assert!(
            h.session.speech_loaded(),
            "an explicit release must not take the engine off a meeting mid-recap"
        );

        // Once the queue drains, nothing is holding it any more.
        h.session.start_job_runner().await.unwrap();
        h.wait_until("the engine to be let go", || !h.asr.is_resident())
            .await;
        h.session.release_idle_resources().await;
        assert!(!h.session.speech_loaded(), "idle Echo holds no weights");
    }

    /// The edge nobody reported: the last thing a meeting was waiting for is
    /// cancelled, so there is nothing left to do — and nothing left that could
    /// ever produce the edge that lets the engine go
    /// (review of 2026-08-20, finding 2).
    #[tokio::test]
    async fn cancelling_the_last_of_a_meetings_work_lets_the_engine_go() {
        let h = Harness::new().await;
        let meeting = repo::create_meeting(&h.db, "Yesterday", "/tmp", None)
            .await
            .unwrap();
        let job = h
            .session
            .queue_job(Some(&meeting.id), JobKind::Summarize)
            .await
            .unwrap();
        assert!(
            h.asr.is_resident(),
            "queued work for a meeting holds the engine"
        );

        h.session.cancel_job(&job).await.unwrap();
        assert!(
            !h.asr.is_resident(),
            "with nothing left to do, the grace period has to start"
        );
        h.session.release_idle_resources().await;
        assert!(!h.session.speech_loaded(), "idle Echo holds no weights");
    }

    /// And the mirror image: asking for that work again needs the engine again,
    /// whether or not the loop reaches the row this second.
    #[tokio::test]
    async fn retrying_cancelled_work_asks_for_the_engine_again() {
        let h = Harness::new().await;
        let meeting = repo::create_meeting(&h.db, "Yesterday", "/tmp", None)
            .await
            .unwrap();
        let job = h
            .session
            .queue_job(Some(&meeting.id), JobKind::Summarize)
            .await
            .unwrap();
        h.session.cancel_job(&job).await.unwrap();
        assert!(!h.asr.is_resident());

        h.session.retry_job(&job).await.unwrap();
        assert!(
            h.asr.is_resident(),
            "work back in the queue is work the engine is needed for"
        );
    }

    /// A row nothing is running cannot be allowed to hold 1.6 GB for the life of
    /// the process. The loop's own tick is the backstop: the row goes back in the
    /// queue, reaches a terminal state, and the engine is let go.
    #[tokio::test]
    async fn a_row_left_running_by_nothing_is_unstuck_by_the_tick() {
        let h = Harness::new().await;
        let meeting = repo::create_meeting(&h.db, "Yesterday", "/tmp", None)
            .await
            .unwrap();
        let job = repo::ensure_job(&h.db, Some(&meeting.id), JobKind::Summarize)
            .await
            .unwrap();
        // The shape a failed status write, or something dying mid-job, leaves
        // behind: `running`, with nothing running it.
        repo::set_job_status(&h.db, &job.id, JobStatus::Running, None)
            .await
            .unwrap();
        h.session.0.jobs.refresh_engine_residency().await;
        assert!(h.asr.is_resident(), "a non-terminal row counts as work");

        h.session.0.jobs.reconcile().await;
        let after = repo::get_job(&h.db, &job.id).await.unwrap().unwrap();
        assert_eq!(
            after.status,
            JobStatus::Queued,
            "an abandoned row belongs back in the queue"
        );
        assert!(
            h.asr.is_resident(),
            "queued work still needs the engine; it is no longer stuck, that is all"
        );

        // And now that it can finish, it does, and the engine is let go.
        h.session.start_job_runner().await.unwrap();
        h.wait_until("the engine to be let go", || !h.asr.is_resident())
            .await;
    }

    /// An engine part-way through its grace period must not be allowed to unload
    /// between the click and the first chunk (review of 2026-08-20, finding 3).
    #[tokio::test]
    async fn a_start_claims_the_engine_before_it_touches_a_device() {
        let h = Harness::new().await;
        assert_eq!(h.capture.engine_held_when_opened(), None);

        h.session.start(Default::default()).await.unwrap();
        assert_eq!(
            h.capture.engine_held_when_opened(),
            Some(true),
            "the engine has to be claimed before the devices are opened, or it can \
             unload and reload across one Start"
        );
    }

    /// And a start that gets nowhere does not leave the engine held for a
    /// recording that never happened.
    #[tokio::test]
    async fn a_start_that_fails_hands_the_engine_back() {
        let h = Harness::new().await;
        h.capture.fail_with(crate::audio::AudioError::NoInputDevice);
        assert!(h.session.start(Default::default()).await.is_err());
        h.settle().await;
        assert!(
            !h.asr.is_resident(),
            "nothing is being recorded and nothing is queued"
        );
    }

    #[tokio::test]
    async fn a_live_recording_never_has_its_resources_taken_away() {
        let h = Harness::new().await;
        h.session.start(Default::default()).await.unwrap();
        h.settle().await;
        h.session.release_idle_resources().await;
        assert!(h.session.speech_loaded());
        assert!(h.asr.is_resident());
    }

    /// Someone pressed Start before Echo could write anything down — the
    /// first-ever launch, where the weights take minutes to come up. What was
    /// said in the meantime is on disk, and it has to end up in the transcript.
    #[tokio::test]
    async fn what_was_said_before_the_engine_was_ready_is_read_back_into_the_transcript() {
        let h = Harness::new().await;
        // Two stretches on disk with no text against them, either side of one the
        // live pass managed on its own.
        h.asr.backlog_writes(&[
            (0, 4_000, "the bit before"),
            (8_000, 12_000, "and after that"),
        ]);
        h.asr.prewarm_takes(Duration::from_millis(120));
        // Half a minute of this meeting is already captured.
        h.capture.set_elapsed(30_000);

        let meeting_id = h.session.start(Default::default()).await.unwrap();
        // The live pass wrote this stretch down while the backlog was being read.
        repo::insert_segment(
            &h.db,
            &crate::types::SegmentDraft {
                meeting_id: meeting_id.clone(),
                t_start_ms: 4_000,
                t_end_ms: 8_000,
                channel: Channel::Mic,
                speaker_id: None,
                text: "live text".into(),
                language: Some("en".into()),
                avg_confidence: Some(0.9),
                revision: 1,
                is_final: true,
                model_name: Some("test".into()),
                model_revision: Some("1".into()),
            },
        )
        .await
        .unwrap();

        h.wait_until("the backlog to be read back", || {
            h.events.finals().len() >= 2
        })
        .await;

        // Everything captured so far, and nothing that is not.
        assert_eq!(h.asr.backlog_calls(), vec![30_000]);

        let shown = h.events.finals();
        assert_eq!(
            shown.len(),
            2,
            "one line per stretch, and no more: {shown:?}"
        );
        // In the order they were said.
        assert_eq!(shown[0].segment.t_start_ms, 0);
        assert_eq!(shown[1].segment.t_start_ms, 8_000);
        // The stretch the live pass already wrote down is not sent again, and was
        // never read a second time.
        assert!(
            shown.iter().all(|line| line.segment.text != "live text"),
            "text already on screen must not be sent twice: {shown:?}"
        );
        // These lines replace no live line: there was nothing on screen for them.
        assert!(shown.iter().all(|line| line.utterance_id.is_none()));
    }

    /// The whole point of the split: the backlog first, in order, and then live —
    /// with no stretch written twice, not even the one the live pass was still
    /// decoding when the backlog pass claimed it
    /// (review of 2026-08-20, finding 1).
    #[tokio::test]
    async fn the_backlog_is_read_in_order_and_the_live_pass_writes_nothing_over_it() {
        let h = Harness::new().await;
        // What is on disk for the stretch before the engine was ready.
        h.asr.backlog_writes(&[(0, 4_000, "the bit before")]);
        // The engine arrives a moment after Start, and every decode is slow
        // enough that the live one is still in flight when it does.
        h.asr.prewarm_takes(Duration::from_millis(150));
        h.asr.transcribe_takes(Duration::from_millis(250));
        h.capture.set_elapsed(30_000);

        let id = h.session.start(Default::default()).await.unwrap();
        // A live utterance for exactly the stretch the backlog pass is about to
        // claim. It is in the engine when the claim happens, and the mock engine
        // answers it anyway.
        h.capture
            .send(ports::CaptureSignal::UtteranceReady(Utterance {
                channel: Channel::Mic,
                t_start_ms: 1_000,
                t_end_ms: 3_000,
                samples: vec![0.0; 16_000],
                truncated: false,
            }));

        h.wait_until("the backlog to be read back", || {
            h.events
                .finals()
                .iter()
                .any(|line| line.segment.text == "the bit before")
        })
        .await;

        // And now the other half of the race: an utterance for that same stretch
        // reaching the live pass *after* the backlog pass has read what is
        // covered. Written, it would be a second copy of words already in the
        // transcript — the duplicate this whole split exists to prevent.
        h.capture
            .send(ports::CaptureSignal::UtteranceReady(Utterance {
                channel: Channel::Mic,
                t_start_ms: 3_500,
                t_end_ms: 3_900,
                samples: vec![0.0; 6_400],
                truncated: false,
            }));
        tokio::time::sleep(Duration::from_millis(400)).await;
        h.settle().await;

        let below = repo::get_segments(
            &h.db,
            &TranscriptQuery {
                meeting_id: id.clone(),
                to_ms: Some(4_000),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(
            below.len(),
            1,
            "one stretch, one line: the live pass and the backlog pass must not both \
             write it: {below:?}"
        );
        assert_eq!(
            below[0].text, "the bit before",
            "the backlog pass owns everything below the floor"
        );

        // And live carries on above the floor: same recording, same pipeline.
        h.capture.set_elapsed(40_000);
        h.capture
            .send(ports::CaptureSignal::UtteranceReady(Utterance {
                channel: Channel::Mic,
                t_start_ms: 35_000,
                t_end_ms: 37_000,
                samples: vec![0.0; 16_000],
                truncated: false,
            }));
        h.wait_until("live text to carry on after the backlog", || {
            h.events
                .finals()
                .iter()
                .any(|line| line.segment.t_start_ms >= 30_000)
        })
        .await;
    }

    /// The stop handoff owns the backlog pass. It used to be detached, which
    /// meant it could still be filling holes while the finalize job worked out
    /// where the holes were (review of 2026-08-20, finding 1).
    #[tokio::test]
    async fn a_stop_waits_for_the_backlog_pass_before_queueing_the_finalize_job() {
        let h = Harness::new().await;
        h.asr.backlog_writes(&[(0, 4_000, "the bit before")]);
        h.asr.prewarm_takes(Duration::from_millis(50));
        h.capture.set_elapsed(30_000);

        let id = h.session.start(Default::default()).await.unwrap();
        h.wait_until("the backlog pass to start", || {
            !h.asr.backlog_calls().is_empty()
        })
        .await;
        h.record_a_minute(&id).await;
        h.session.stop().await.unwrap();

        assert!(
            h.session
                .0
                .backlog_task
                .lock()
                .expect("backlog task lock")
                .is_none(),
            "the stop owns that task and waits for it"
        );
        assert!(
            h.session.0.backlog.lock().expect("backlog lock").is_none(),
            "and the split goes with the recording it belonged to"
        );
        assert!(
            h.queued_kinds(&id)
                .await
                .contains(&JobKind::TranscribeCatchup),
            "only then does the disk pass for the rest of the meeting get queued"
        );
    }

    #[tokio::test]
    async fn a_meeting_that_is_over_before_the_engine_arrives_is_left_to_the_catch_up_job() {
        let h = Harness::new().await;
        h.asr.backlog_writes(&[(0, 4_000, "too late")]);
        // Longer than the whole start-record-stop it is racing.
        h.asr.prewarm_takes(Duration::from_millis(400));
        h.capture.set_elapsed(30_000);

        let meeting_id = h.session.start(Default::default()).await.unwrap();
        h.record_a_minute(&meeting_id).await;
        h.session.stop().await.unwrap();
        h.settle().await;
        tokio::time::sleep(Duration::from_millis(600)).await;

        assert!(
            h.asr.backlog_calls().is_empty(),
            "a live backlog pass must not start under a meeting that has ended"
        );
        assert!(
            h.queued_kinds(&meeting_id)
                .await
                .contains(&JobKind::TranscribeCatchup),
            "the finalize job owns it from there"
        );
    }

    // -----------------------------------------------------------------------
    // The one-time setup, and what the screen is told about speech
    // -----------------------------------------------------------------------

    /// Every file the level needs, on disk, with nothing in the table saying so
    /// and no download job behind it. That is the machine of 2026-08-24: weights
    /// that arrived some other way, owing this computer a setup nobody had paid.
    async fn the_speech_files_are_simply_there(h: &Harness) {
        crate::asr::models::sync_catalog(&h.db).await.unwrap();
        for id in crate::asr::catalog::preset_asset_ids(crate::asr::catalog::DEFAULT_PRESET_ID) {
            let entry = crate::asr::catalog::entry(id).expect("catalogued");
            let path = crate::asr::models::install_path(&h.paths, entry);
            if entry.is_bundle() {
                std::fs::create_dir_all(&path).unwrap();
                std::fs::write(path.join("coremldata.bin"), b"x").unwrap();
            } else {
                std::fs::write(&path, b"weights").unwrap();
            }
        }
    }

    /// Every setup row there has ever been, finished or not.
    async fn setup_jobs(h: &Harness) -> Vec<crate::types::Job> {
        repo::list_jobs(
            &h.db,
            &JobQuery {
                kind: Some(JobKind::PrepareEngine),
                ..Default::default()
            },
        )
        .await
        .expect("the work queue")
    }

    /// The readiness check runs at launch and again on every question the UI
    /// asks — several times a second while a screen watches it. It has to end up
    /// with exactly one setup row, and then with none at all.
    #[tokio::test]
    async fn the_one_time_speech_setup_is_asked_for_once_and_only_once() {
        let h = Harness::new().await;
        the_speech_files_are_simply_there(&h).await;

        assert!(
            h.session.ensure_speech_current().await,
            "everything the level needs is on disk"
        );
        let queued = setup_jobs(&h).await;
        assert_eq!(queued.len(), 1, "the setup is asked for: {queued:?}");

        h.session.ensure_speech_current().await;
        h.session.ensure_speech_current().await;
        assert_eq!(
            setup_jobs(&h).await.len(),
            1,
            "the readiness polls join the row that exists instead of stacking rows"
        );

        // And once it has been paid, nothing asks again — not even after the row
        // is gone from the queue, which is the only thing dedup was doing.
        repo::set_job_status(&h.db, &queued[0].id, JobStatus::Done, None)
            .await
            .unwrap();
        let file_name = crate::asr::models::installed_speech_file_name(&h.db)
            .await
            .unwrap()
            .expect("something is serving");
        crate::asr::models::mark_warmed(&h.db, &file_name)
            .await
            .unwrap();

        h.session.ensure_speech_current().await;
        assert_eq!(
            setup_jobs(&h).await.len(),
            1,
            "weights that have been through a load here owe nothing"
        );
    }

    /// Mantra 3: a meeting already has the engine's whole attention. Asking for
    /// the same load twice helps nobody, and the row would be parked anyway.
    #[tokio::test]
    async fn the_one_time_speech_setup_is_never_asked_for_during_a_meeting() {
        let h = Harness::new().await;
        the_speech_files_are_simply_there(&h).await;

        h.session.start(Default::default()).await.unwrap();
        h.settle().await;
        h.session.ensure_speech_current().await;
        assert!(
            setup_jobs(&h).await.is_empty(),
            "a meeting in progress is already loading the engine it needs"
        );

        // The moment it is over, the same check asks for it.
        let id = h.session.status().await.meeting_id.expect("a meeting");
        h.record_a_minute(&id).await;
        h.session.stop().await.unwrap();
        h.session.ensure_speech_current().await;
        assert_eq!(setup_jobs(&h).await.len(), 1);
    }

    /// A setup that failed for a reason nobody can see from here — no room on
    /// the disk, an engine that would not start — is asked for again, but at the
    /// next launch and not a moment sooner. Asking on the readiness polls would
    /// be the same failure a hundred times a minute, and never asking again
    /// would hand the sixteen minutes of 2026-08-24 back to a real meeting.
    #[tokio::test]
    async fn a_setup_that_failed_gets_its_next_go_at_the_next_launch() {
        let h = Harness::new().await;
        the_speech_files_are_simply_there(&h).await;

        h.session.ensure_speech_current().await;
        let queued = setup_jobs(&h).await;
        assert_eq!(queued.len(), 1);

        // What a failure that never reached the compile leaves behind: a failed
        // row, and no record of the attempt (`prepare_engine` takes that back).
        repo::set_job_status(&h.db, &queued[0].id, JobStatus::Failed, Some("no"))
            .await
            .unwrap();
        assert!(
            crate::asr::models::warm_up_needed(&h.db)
                .await
                .unwrap()
                .is_some(),
            "the weights still owe this machine a setup"
        );

        for _ in 0..5 {
            h.session.ensure_speech_current().await;
        }
        let after = setup_jobs(&h).await;
        assert_eq!(
            after.len(),
            1,
            "the row that failed is the memory of the attempt for the rest of the launch: {after:?}"
        );
        assert_eq!(after[0].status, JobStatus::Failed);

        // And the next launch hands it back — the whole automatic way out of the
        // one-attempt policy.
        h.session.retry_failed_speech_setup().await;
        let revived = setup_jobs(&h).await;
        assert_eq!(revived.len(), 1, "the same row, not a new one");
        assert_eq!(revived[0].status, JobStatus::Queued);
    }

    /// The other side of it: an attempt that got as far as the compile and then
    /// took the app down with it keeps its marker, and no launch starts that
    /// quarter of an hour again on its own.
    #[tokio::test]
    async fn a_setup_that_died_inside_the_compile_is_not_started_again() {
        let h = Harness::new().await;
        the_speech_files_are_simply_there(&h).await;

        h.session.ensure_speech_current().await;
        let queued = setup_jobs(&h).await;
        assert_eq!(queued.len(), 1);
        // Exactly what the process going down mid-compile leaves: a row still
        // marked running, and the attempt on the record because it was written
        // before the load.
        let file_name = crate::asr::models::installed_speech_file_name(&h.db)
            .await
            .unwrap()
            .expect("something is serving");
        crate::asr::models::mark_warm_attempted(&h.db, &file_name)
            .await
            .unwrap();
        repo::set_job_status(&h.db, &queued[0].id, JobStatus::Running, None)
            .await
            .unwrap();

        // The launch that follows.
        h.session.retry_failed_speech_setup().await;
        repo::requeue_orphaned_jobs(&h.db, jobs::SETUP_INTERRUPTED)
            .await
            .unwrap();
        let after = setup_jobs(&h).await;
        assert_eq!(after.len(), 1);
        assert_eq!(
            after[0].status,
            JobStatus::Failed,
            "an interrupted compile is settled, not handed straight back"
        );

        // And nothing queues a fresh one either, however often the screen asks.
        for _ in 0..5 {
            h.session.ensure_speech_current().await;
        }
        assert_eq!(setup_jobs(&h).await.len(), 1);
    }

    /// The rule the banner is drawn from, with no meeting, no thread and no
    /// weights in it.
    #[test]
    fn what_the_screen_is_told_about_speech_follows_from_what_is_true_now() {
        use SpeechState as S;

        // Nothing is being recorded and nothing is loaded: nothing to say.
        assert_eq!(speech_state(false, false, CaptureState::Idle), S::Idle);
        assert_eq!(speech_state(false, false, CaptureState::Stopped), S::Idle);
        // Idle Echo that happens to be holding the weights is not news either
        // way, but it is honestly ready.
        assert_eq!(speech_state(true, false, CaptureState::Idle), S::Ready);

        for state in [
            CaptureState::Starting,
            CaptureState::Recording,
            CaptureState::Degraded,
            CaptureState::Paused,
            CaptureState::Stopping,
        ] {
            assert_eq!(
                speech_state(false, false, state),
                S::Preparing,
                "{state:?}: a meeting waiting on the engine is the thing nobody could see"
            );
            assert_eq!(speech_state(true, false, state), S::Ready, "{state:?}");
        }

        // A load that failed is worth saying, wherever the capture has got to.
        assert_eq!(
            speech_state(false, true, CaptureState::Recording),
            S::Unavailable
        );
        assert_eq!(
            speech_state(false, true, CaptureState::Idle),
            S::Unavailable
        );

        // And what is loaded wins over what once failed: a later load repairs an
        // earlier failure, and a banner outliving its problem is worse than none.
        assert_eq!(
            speech_state(true, true, CaptureState::Recording),
            S::Ready,
            "the engine is up; nothing on screen may claim otherwise"
        );
        assert_eq!(speech_state(true, true, CaptureState::Idle), S::Ready);
    }

    /// The meeting of 2026-08-24, as the screen would see it now: the engine
    /// takes its time, and the status says so from the click until it is up.
    #[tokio::test]
    async fn a_slow_engine_shows_as_getting_ready_and_then_as_ready() {
        let h = Harness::new().await;
        h.asr.prewarm_takes(Duration::from_millis(120));

        h.session.start(Default::default()).await.unwrap();
        let while_loading = h.events.capture_states();
        assert!(
            while_loading
                .iter()
                .any(|s| s.speech == SpeechState::Preparing),
            "the wait is on the screen while it is happening: {while_loading:?}"
        );
        assert!(
            while_loading.iter().all(|s| s.speech != SpeechState::Ready),
            "nothing may claim to be ready before it is: {while_loading:?}"
        );

        h.wait_until("the engine to be announced as ready", || {
            h.events
                .capture_states()
                .iter()
                .any(|s| s.speech == SpeechState::Ready)
        })
        .await;

        let seen = h.events.capture_states();
        let first_ready = seen
            .iter()
            .position(|s| s.speech == SpeechState::Ready)
            .expect("just waited for it");
        assert!(
            seen[..first_ready]
                .iter()
                .any(|s| s.speech == SpeechState::Preparing),
            "Preparing comes first, and it is the whole point: {seen:?}"
        );
        assert!(
            seen.iter().all(|s| s.speech != SpeechState::Unavailable),
            "nothing failed here: {seen:?}"
        );
        assert_eq!(seen.last().map(|s| s.speech), Some(SpeechState::Ready));
    }

    /// And when it never comes up, the screen is told that too — in the same
    /// breath as the banner, so the two cannot disagree.
    #[tokio::test]
    async fn a_meeting_whose_engine_never_arrives_says_so_on_the_screen() {
        let h = Harness::new().await;
        h.asr.fail_prewarm();

        h.session.start(Default::default()).await.unwrap();
        h.wait_until("the screen to be told", || {
            h.events
                .capture_states()
                .iter()
                .any(|s| s.speech == SpeechState::Unavailable)
        })
        .await;

        assert!(
            h.events.notice_tagged("speechNotReady"),
            "the banner says the same thing at the same moment"
        );
        assert_eq!(
            h.session.status().await.speech,
            SpeechState::Unavailable,
            "and asking for the status gives the same answer as the event did"
        );
        assert!(
            is_live(h.session.status().await.state),
            "mantra 3: the recording carries on regardless"
        );
    }

    // -----------------------------------------------------------------------
    // Recovery
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn an_interrupted_meeting_is_offered_and_can_be_finished() {
        let h = Harness::new().await;
        let meeting = repo::create_meeting(&h.db, "Yesterday", "/tmp", None)
            .await
            .unwrap();
        repo::set_meeting_status(&h.db, &meeting.id, MeetingStatus::Recording)
            .await
            .unwrap();
        let chunk = repo::insert_chunk(
            &h.db,
            &meeting.id,
            Channel::Mic,
            0,
            "/tmp/mic-000000.flac",
            0,
            60_000,
        )
        .await
        .unwrap();
        repo::commit_chunk(&h.db, &chunk, 60_000).await.unwrap();

        let found = h.session.find_interrupted().await.unwrap();
        assert_eq!(found, vec![meeting.id.clone()]);
        assert_eq!(h.session.status().await.state, CaptureState::Recovering);
        assert!(h.events.notice_tagged("recoveryAvailable"));
        // Idempotent.
        assert_eq!(
            h.session.find_interrupted().await.unwrap(),
            vec![meeting.id.clone()]
        );

        h.session
            .resolve_interrupted(&meeting.id, RecoveryAction::Finish)
            .await
            .unwrap();
        assert_eq!(h.session.status().await.state, CaptureState::Idle);

        let kinds: Vec<JobKind> = repo::list_jobs(
            &h.db,
            &JobQuery {
                meeting_id: Some(meeting.id.clone()),
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .into_iter()
        .map(|j| j.kind)
        .collect();
        assert!(kinds.contains(&JobKind::TranscribeCatchup));
        let after = repo::get_meeting(&h.db, &meeting.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(after.status, MeetingStatus::Processing);
        assert_eq!(
            after.duration_ms, 60_000,
            "the journal knows how long it was"
        );
    }

    #[tokio::test]
    async fn discarding_an_interrupted_meeting_keeps_the_audio_on_disk() {
        let h = Harness::new().await;
        let meeting = repo::create_meeting(&h.db, "Yesterday", "/tmp", None)
            .await
            .unwrap();
        repo::set_meeting_status(&h.db, &meeting.id, MeetingStatus::Recording)
            .await
            .unwrap();
        let chunk = repo::insert_chunk(
            &h.db,
            &meeting.id,
            Channel::Mic,
            0,
            "/tmp/mic-000000.flac",
            0,
            5_000,
        )
        .await
        .unwrap();
        repo::commit_chunk(&h.db, &chunk, 5_000).await.unwrap();

        h.session.find_interrupted().await.unwrap();
        h.session
            .resolve_interrupted(&meeting.id, RecoveryAction::Discard)
            .await
            .unwrap();

        assert_eq!(h.session.status().await.state, CaptureState::Idle);
        let after = repo::get_meeting(&h.db, &meeting.id)
            .await
            .unwrap()
            .unwrap();
        assert!(after.deleted_at.is_some(), "out of the way, not shredded");
        assert_eq!(
            repo::list_chunks(&h.db, &meeting.id, None)
                .await
                .unwrap()
                .len(),
            1,
            "committed audio is never thrown away behind the person's back"
        );
        // Not offered again.
        assert!(h.session.find_interrupted().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn an_interrupted_meeting_with_no_audio_is_not_worth_asking_about() {
        let h = Harness::new().await;
        let meeting = repo::create_meeting(&h.db, "Empty", "/tmp", None)
            .await
            .unwrap();
        repo::set_meeting_status(&h.db, &meeting.id, MeetingStatus::Recording)
            .await
            .unwrap();

        assert!(h.session.find_interrupted().await.unwrap().is_empty());
        assert_eq!(h.session.status().await.state, CaptureState::Idle);
        let after = repo::get_meeting(&h.db, &meeting.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(after.status, MeetingStatus::Failed);
        assert!(after.deleted_at.is_some());
    }

    #[tokio::test]
    async fn audio_on_disk_with_no_journal_rows_is_recovered_not_binned() {
        let h = Harness::new().await;
        let meeting = repo::create_meeting(
            &h.db,
            "Crashed before it could write the journal",
            "/tmp",
            None,
        )
        .await
        .unwrap();
        repo::set_meeting_status(&h.db, &meeting.id, MeetingStatus::Recording)
            .await
            .unwrap();
        // Two seconds of real, fsynced audio in the meeting's folder, and not a
        // single row to say so — a crash between the rename and the insert.
        let dir = h.paths.meeting_dir(&meeting.id);
        std::fs::create_dir_all(&dir).unwrap();
        repo::set_meeting_audio_dir(&h.db, &meeting.id, &dir.to_string_lossy())
            .await
            .unwrap();
        {
            let mut writer = crate::audio::writer::ChunkWriter::create(
                &dir,
                Channel::Mic,
                crate::audio::TARGET_SAMPLE_RATE,
            )
            .unwrap();
            writer
                .write_samples(&vec![0.2f32; crate::audio::TARGET_SAMPLE_RATE as usize * 2])
                .unwrap();
            writer.finish().unwrap();
        }
        assert!(repo::list_chunks(&h.db, &meeting.id, None)
            .await
            .unwrap()
            .is_empty());

        let found = h.session.find_interrupted().await.unwrap();
        assert_eq!(
            found,
            vec![meeting.id.clone()],
            "audio on disk means there is something to recover"
        );
        let chunks = repo::list_chunks(&h.db, &meeting.id, None).await.unwrap();
        assert_eq!(chunks.len(), 1, "the journal caught up with the disk");
        assert!(chunks[0].committed);
        assert_eq!(chunks[0].t_start_ms, 0);
        assert_eq!(chunks[0].t_end_ms, 2_000);
        let after = repo::get_meeting(&h.db, &meeting.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(after.status, MeetingStatus::Interrupted);
        assert!(after.deleted_at.is_none());
        assert_eq!(after.duration_ms, 2_000);

        // Running the scan again must not journal the same audio twice.
        h.session.find_interrupted().await.unwrap();
        assert_eq!(
            repo::list_chunks(&h.db, &meeting.id, None)
                .await
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn recovery_never_touches_the_meeting_being_recorded_right_now() {
        let h = Harness::new().await;
        let id = h.session.start(Default::default()).await.unwrap();
        assert!(h.session.find_interrupted().await.unwrap().is_empty());
        assert_eq!(h.session.status().await.state, CaptureState::Recording);
        let meeting = repo::get_meeting(&h.db, &id).await.unwrap().unwrap();
        assert_eq!(meeting.status, MeetingStatus::Recording);
    }

    // -----------------------------------------------------------------------
    // Jobs and preemption
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn a_recording_parks_running_work_which_then_resumes_in_order() {
        let h = Harness::new().await;
        let meeting = repo::create_meeting(&h.db, "Older", "/tmp", None)
            .await
            .unwrap();

        // Something long-running is in flight, plus work behind it.
        h.session
            .queue_job(Some(&meeting.id), JobKind::Diarize)
            .await
            .unwrap();
        h.session
            .queue_job(Some(&meeting.id), JobKind::Summarize)
            .await
            .unwrap();
        let catchup = h
            .session
            .queue_job(Some(&meeting.id), JobKind::TranscribeCatchup)
            .await
            .unwrap();

        h.executor.block_until_told();
        h.session.start_job_runner().await.unwrap();
        h.executor
            .wait_until_running(JobKind::TranscribeCatchup)
            .await;
        repo::set_job_progress(&h.db, &catchup, 0.4).await.unwrap();

        // Recording takes the machine.
        let live = h.session.start(Default::default()).await.unwrap();
        h.executor.wait_for_parked().await;
        let parked = repo::get_job(&h.db, &catchup).await.unwrap().unwrap();
        assert_eq!(parked.status, JobStatus::Paused, "parked, not failed");
        assert_eq!(parked.progress, Some(0.4), "and it kept its place");
        for job in repo::list_jobs(&h.db, &JobQuery::default()).await.unwrap() {
            assert_ne!(
                job.status,
                JobStatus::Running,
                "nothing runs while recording"
            );
        }

        // Recording ends: parked work carries on, catch-up first.
        h.executor.stop_blocking();
        h.record_a_minute(&live).await;
        h.session.stop().await.unwrap();
        // Three from the older meeting, four queued by the stop (the recap is
        // automatic).
        h.executor.wait_for_kinds(7).await;
        let order = h.executor.finished();
        assert_eq!(order[0], JobKind::TranscribeCatchup, "{order:?}");
        let last_catchup = order
            .iter()
            .rposition(|k| *k == JobKind::TranscribeCatchup)
            .expect("catch-up ran");
        let last_speakers = order
            .iter()
            .rposition(|k| *k == JobKind::Diarize)
            .expect("the speaker pass ran");
        let recap = order
            .iter()
            .position(|k| *k == JobKind::Summarize)
            .expect("the recap ran");
        assert!(
            last_catchup < last_speakers && last_speakers < recap,
            "catch-up, then speakers, then the recap: {order:?}"
        );
    }

    #[tokio::test]
    async fn cancelling_a_job_is_idempotent_and_cancelling_a_finished_one_is_fine() {
        let h = Harness::new().await;
        let meeting = repo::create_meeting(&h.db, "Older", "/tmp", None)
            .await
            .unwrap();
        let job = h
            .session
            .queue_job(Some(&meeting.id), JobKind::Summarize)
            .await
            .unwrap();

        h.session.cancel_job(&job).await.unwrap();
        assert_eq!(
            repo::get_job(&h.db, &job).await.unwrap().unwrap().status,
            JobStatus::Cancelled
        );
        h.session.cancel_job(&job).await.unwrap();

        // And it can be put back.
        h.session.retry_job(&job).await.unwrap();
        assert_eq!(
            repo::get_job(&h.db, &job).await.unwrap().unwrap().status,
            JobStatus::Queued
        );
    }

    #[tokio::test]
    async fn asking_for_the_same_work_twice_queues_it_once() {
        let h = Harness::new().await;
        let meeting = repo::create_meeting(&h.db, "Older", "/tmp", None)
            .await
            .unwrap();
        let first = h
            .session
            .queue_job(Some(&meeting.id), JobKind::Summarize)
            .await
            .unwrap();
        let second = h
            .session
            .queue_job(Some(&meeting.id), JobKind::Summarize)
            .await
            .unwrap();
        assert_eq!(first, second);
    }

    // Helpers ---------------------------------------------------------------

    fn all_meetings() -> crate::types::MeetingQuery {
        crate::types::MeetingQuery {
            include_deleted: Some(true),
            ..Default::default()
        }
    }
}
