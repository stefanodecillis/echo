//! Event names and payloads emitted from Rust to the webview.
//!
//! Rules:
//! * Names are `echo://<topic>` and live **only** here, never as a literal
//!   anywhere else in the tree. The TypeScript mirror is `src/lib/ipc.ts`.
//! * Payloads are `camelCase`.
//! * High-frequency topics (`TRANSCRIPT_PARTIAL`, `AUDIO_LEVELS`) are
//!   rate-capped by the emitter, not by the UI (DESIGN §3: "UI events
//!   (rate-capped)"). Emitting must never block capture (mantra 3).
//! * Nothing user-visible in a payload may contain jargon (mantra 2).

use serde::{Deserialize, Serialize};

use crate::types::{
    ActionItem, CaptureStatus, DetectionStatus, Id, Job, MeetingStatus, Segment, Speaker,
    TrayAction, TrayState,
};

// ---------------------------------------------------------------------------
// Event names
// ---------------------------------------------------------------------------

/// Capture state machine changed.
pub const CAPTURE_STATE: &str = "echo://capture-state";
/// Live, not-yet-final text for the current utterance. Rate-capped ~5/s.
pub const TRANSCRIPT_PARTIAL: &str = "echo://transcript-partial";
/// A segment reached its final revision and is persisted.
pub const TRANSCRIPT_FINAL: &str = "echo://transcript-final";
/// Existing segments were replaced by a better pass (catch-up, diarization).
pub const TRANSCRIPT_REVISED: &str = "echo://transcript-revised";
/// Per-channel loudness for the recording indicator. Rate-capped ~10/s.
pub const AUDIO_LEVELS: &str = "echo://audio-levels";
/// A background job was created, advanced, or finished.
pub const JOB_PROGRESS: &str = "echo://job-progress";
/// Download progress for what Echo needs to understand speech.
pub const DOWNLOAD_PROGRESS: &str = "echo://download-progress";
/// The meeting watcher changed its mind.
pub const DETECTION: &str = "echo://detection";
/// Speaker list changed (new speaker, rename, merge).
pub const SPEAKERS_UPDATED: &str = "echo://speakers-updated";
/// A recap finished and is ready to render.
pub const SUMMARY_READY: &str = "echo://summary-ready";
/// Action items were (re)generated or edited.
pub const ACTION_ITEMS_UPDATED: &str = "echo://action-items-updated";
/// A meeting row changed (title, status, duration).
pub const MEETING_UPDATED: &str = "echo://meeting-updated";
/// Settings changed anywhere (including from the tray).
pub const SETTINGS_CHANGED: &str = "echo://settings-changed";
/// A calm, user-facing message for a banner or toast.
pub const NOTICE: &str = "echo://notice";
/// Tray icon appearance changed, for windows that mirror it.
pub const TRAY_STATE: &str = "echo://tray-state";
/// The person picked something in the tray menu.
pub const TRAY_ACTION: &str = "echo://tray-action";
/// Something outside the UI wants the UI to go somewhere (notification click,
/// tray "Open Echo", second-instance launch).
pub const NAVIGATE: &str = "echo://navigate";
/// Interrupted meetings were found at launch; the UI offers finish or discard.
pub const RECOVERY_AVAILABLE: &str = "echo://recovery-available";

/// Every event name, for tests that assert the TS mirror is complete.
pub const ALL: &[&str] = &[
    CAPTURE_STATE,
    TRANSCRIPT_PARTIAL,
    TRANSCRIPT_FINAL,
    TRANSCRIPT_REVISED,
    AUDIO_LEVELS,
    JOB_PROGRESS,
    DOWNLOAD_PROGRESS,
    DETECTION,
    SPEAKERS_UPDATED,
    SUMMARY_READY,
    ACTION_ITEMS_UPDATED,
    MEETING_UPDATED,
    SETTINGS_CHANGED,
    NOTICE,
    TRAY_STATE,
    TRAY_ACTION,
    NAVIGATE,
    RECOVERY_AVAILABLE,
];

// ---------------------------------------------------------------------------
// Payloads
// ---------------------------------------------------------------------------

/// `CAPTURE_STATE`
pub type CaptureStatePayload = CaptureStatus;

/// `TRANSCRIPT_PARTIAL`, cheap and frequent; never persisted as-is.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptPartialPayload {
    pub meeting_id: Id,
    /// Stable id for this in-flight utterance so the UI can replace in place.
    pub utterance_id: String,
    pub t_start_ms: i64,
    pub t_end_ms: i64,
    pub channel: crate::types::Channel,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker_id: Option<Id>,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// True when this utterance is over and there is nothing to write down for
    /// it: it was dropped to protect capture, it turned out to be silence, or
    /// the engine could not read it. **No final will ever arrive for this
    /// utterance**, so the half-written line has to be taken off the screen.
    /// Anything that opens a partial must eventually close it, one way or the
    /// other, or the live view collects "…" lines that never settle.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub dropped: bool,
}

/// `TRANSCRIPT_FINAL`
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptFinalPayload {
    pub meeting_id: Id,
    /// The utterance this replaces, when it came from a live partial.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub utterance_id: Option<String>,
    pub segment: Segment,
}

/// `TRANSCRIPT_REVISED`, batched; the UI refetches the affected window.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptRevisedPayload {
    pub meeting_id: Id,
    pub revision: i64,
    pub from_ms: i64,
    pub to_ms: i64,
    pub segment_ids: Vec<Id>,
}

/// `AUDIO_LEVELS`, 0.0..=1.0 per channel, already smoothed.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioLevelsPayload {
    pub meeting_id: Id,
    pub mic: f32,
    pub system: f32,
    pub t_ms: i64,
}

/// `JOB_PROGRESS`
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobProgressPayload {
    pub job: Job,
    /// Sentence for the UI: "Writing your recap…", "Catching up on the last
    /// few minutes…". Zero jargon.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

/// `DOWNLOAD_PROGRESS`
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadProgressPayload {
    /// Model row id; internal, not shown.
    pub asset_id: Id,
    /// Accuracy level this belongs to, so onboarding can aggregate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level_id: Option<String>,
    pub received_bytes: i64,
    pub total_bytes: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eta_seconds: Option<u64>,
    pub done: bool,
    /// User-facing failure sentence when the download stopped.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// `DETECTION`
pub type DetectionPayload = DetectionStatus;

/// `SPEAKERS_UPDATED`
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakersUpdatedPayload {
    pub meeting_id: Id,
    pub speakers: Vec<Speaker>,
}

/// `SUMMARY_READY`
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SummaryReadyPayload {
    pub meeting_id: Id,
    pub summary_id: Id,
}

/// `ACTION_ITEMS_UPDATED`
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ActionItemsUpdatedPayload {
    pub meeting_id: Id,
    pub items: Vec<ActionItem>,
}

/// `MEETING_UPDATED`
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MeetingUpdatedPayload {
    pub meeting_id: Id,
    pub status: MeetingStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub duration_ms: i64,
    /// The row is gone (deleted).
    pub deleted: bool,
}

/// `NOTICE` severity. Drives colour only; the text carries the meaning.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum NoticeLevel {
    #[default]
    Info,
    /// Something is less good than it should be but work continues.
    Warning,
    /// Something the person needs to act on.
    Problem,
}

/// `NOTICE`, the only channel for spontaneous user-facing messages.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NoticePayload {
    pub level: NoticeLevel,
    /// One short sentence, no jargon, says what the person can do.
    pub message: String,
    /// Sticky banner rather than a toast.
    pub persistent: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meeting_id: Option<Id>,
    /// Machine tag so the UI can dedupe/replace, e.g. "systemAudioLost".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
}

/// `TRAY_STATE`
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrayStatePayload {
    pub state: TrayState,
}

/// `TRAY_ACTION`
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrayActionPayload {
    pub action: TrayAction,
}

/// Where the UI should go.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum NavigateTarget {
    /// Home with a prominent Start button (notification click).
    #[default]
    HomeStart,
    Home,
    Live,
    Meeting,
    Search,
    Settings,
    Onboarding,
}

/// `NAVIGATE`
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NavigatePayload {
    pub target: NavigateTarget,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meeting_id: Option<Id>,
    /// The app that made us suggest starting, for the Start card copy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detected_app: Option<String>,
}

/// `RECOVERY_AVAILABLE`
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecoveryAvailablePayload {
    pub meeting_ids: Vec<Id>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_event_name_is_namespaced_and_unique() {
        let mut seen = std::collections::HashSet::new();
        for name in ALL {
            assert!(name.starts_with("echo://"), "{name} is not namespaced");
            assert!(seen.insert(*name), "{name} is declared twice");
        }
    }

    #[test]
    fn notice_payload_is_camel_case() {
        let json = serde_json::to_string(&NoticePayload {
            level: NoticeLevel::Warning,
            message: "Echo kept your microphone but stopped hearing your computer.".into(),
            persistent: true,
            meeting_id: None,
            tag: Some("systemAudioLost".into()),
        })
        .unwrap();
        assert!(json.contains("\"persistent\":true"), "{json}");
        assert!(json.contains("\"level\":\"warning\""), "{json}");
    }
}
