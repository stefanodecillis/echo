//! Shared serde types crossing the IPC boundary.
//!
//! Contract rules (see `docs/DESIGN.md`):
//! * Every struct is `camelCase` on the wire so the TypeScript mirror in
//!   `src/lib/types.ts` needs no adapters. Keep the two files in lockstep.
//! * Ids are UUID v4 strings (SQLite stores them as TEXT).
//! * Times are milliseconds. Wall-clock timestamps are RFC3339 UTC strings;
//!   in-meeting offsets are `i64` milliseconds on the monotonic meeting clock.
//! * **Zero jargon** (mantra 2): anything a user reads is a sentence, not a
//!   symbol. Enum variants are machine-facing; the UI maps them through
//!   `src/lib/copy.ts`. Never put "Whisper", "VAD", "ONNX", "model",
//!   "connector" or "token" in a string that reaches the UI outside
//!   Settings → Advanced.

use serde::{Deserialize, Serialize};

/// UUID v4, lowercase hyphenated.
pub type Id = String;
/// RFC3339 UTC timestamp.
pub type Timestamp = String;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// The only error shape a command may return.
///
/// `message` is user-facing and must say what the person can *do*; `detail` is
/// the technical cause and is only rendered under Settings → Advanced or
/// written to the redacted log.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UiError {
    pub kind: UiErrorKind,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// Optional suggested next step the UI can render as a button.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action: Option<UiErrorAction>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum UiErrorKind {
    /// A permission the person has to grant in system settings.
    PermissionNeeded,
    /// Something Echo still has to download before it can work.
    NotReady,
    /// Disk full / storage unavailable.
    Storage,
    /// Network or remote service problem.
    Network,
    /// Credential missing, keychain locked, or the person cancelled the prompt.
    Credential,
    /// Bad input from the UI (validated in Rust, never trusted).
    InvalidInput,
    /// Requested thing does not exist.
    NotFound,
    /// The operation was cancelled on purpose.
    Cancelled,
    /// Anything else; message stays generic and calm.
    Unexpected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum UiErrorAction {
    OpenMicrophoneSettings,
    OpenScreenRecordingSettings,
    OpenSpeechSettings,
    OpenSummarySettings,
    OpenStorageSettings,
    Retry,
    RestartApp,
}

impl UiError {
    pub fn new(kind: UiErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            detail: None,
            action: None,
        }
    }

    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }

    pub fn with_action(mut self, action: UiErrorAction) -> Self {
        self.action = Some(action);
        self
    }

    pub fn not_found(what: impl Into<String>) -> Self {
        Self::new(UiErrorKind::NotFound, what)
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(UiErrorKind::InvalidInput, message)
    }

    pub fn unexpected(detail: impl Into<String>) -> Self {
        Self::new(
            UiErrorKind::Unexpected,
            "Something went wrong. Nothing was lost, your recording is safe on this computer.",
        )
        .with_detail(detail)
    }
}

impl std::fmt::Display for UiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)?;
        if let Some(d) = &self.detail {
            write!(f, " ({d})")?;
        }
        Ok(())
    }
}

impl std::error::Error for UiError {}

/// Every command returns this.
pub type CmdResult<T> = Result<T, UiError>;

// ---------------------------------------------------------------------------
// Capture state machine
// ---------------------------------------------------------------------------

/// Orthogonal to jobs and to detection (DESIGN §3 "State model").
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CaptureState {
    #[default]
    Idle,
    Starting,
    Recording,
    Paused,
    Stopping,
    Stopped,
    /// Capture could not start or died unrecoverably.
    Failed,
    /// Still recording, but with less than we wanted (e.g. system audio lost).
    Degraded,
    /// An interrupted meeting was found at launch and awaits a user choice.
    Recovering,
}

/// Which channels are actually flowing right now.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Channel {
    /// Microphone, labelled "You" in the UI.
    #[default]
    Mic,
    /// Everything the computer plays.
    System,
    /// Derived mixdown used for playback only.
    Mixed,
}

impl Channel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Channel::Mic => "mic",
            Channel::System => "system",
            Channel::Mixed => "mixed",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "mic" => Some(Channel::Mic),
            "system" => Some(Channel::System),
            "mixed" => Some(Channel::Mixed),
            _ => None,
        }
    }
}

/// Why capture is degraded, mapped to a calm banner by the UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum DegradedReason {
    /// What this computer plays never reached Echo at all — no permission, no
    /// device, or a stream that opened and delivered nothing. The whole meeting
    /// is in the microphone recording, so the offline pass will separate the
    /// voices in it and the banner is allowed to say so.
    SystemAudioUnavailable,
    /// What this computer plays *was* reaching Echo and stopped. The two halves
    /// of the meeting are not alike: the first has its own channel and the
    /// second does not, which is why this is a reason of its own rather than a
    /// second use of [`Self::SystemAudioUnavailable`]. A banner drawn from that
    /// one would promise a separation of the microphone tail that the offline
    /// pass does not do for a meeting that has any system audio at all
    /// (`diarize::pipeline::voice_channel`), and would claim every line so far
    /// says "You" when the ones from this computer say a real name.
    SystemAudioLost,
    /// Microphone gone, we keep what the computer plays.
    MicrophoneUnavailable,
    /// Nothing is reaching Echo at all: this recording has no microphone in it,
    /// and what the computer plays is not arriving either.
    ///
    /// The state the other two system-audio reasons cannot describe without
    /// lying, because both of their sentences end in a promise about the
    /// microphone recording — one says the offline pass will sort the voices
    /// out of it, the other says Echo is still recording through it. Somebody
    /// who denied the microphone and is capturing this computer alone has no
    /// microphone recording, so a banner drawn from either reason reads as
    /// "carry on, Echo has you" while nothing whatever is being saved.
    NothingIsBeingHeard,
    /// Live text is behind; audio on disk is complete and will catch up.
    TranscriptBehind,
    /// Disk is nearly full.
    StorageLow,
}

/// Whether Echo can write down what is being said right now, in plain states a
/// screen can show without knowing anything about weights or engines.
///
/// Worked out fresh from what the engine is holding every time
/// [`CaptureStatus`] is built, and never remembered anywhere. That is what makes
/// it survive the failure of 2026-08-24, when the window stopped receiving
/// events and every remembered flag on the screen froze with the last one that
/// got through: nothing here can be stale, because there is nothing to go stale.
///
/// Being derived only helps if something re-reads it, so the screen does: while
/// this says `Preparing` or `Unavailable` — the two states a banner is drawn
/// from — the window asks for the status outright every few seconds
/// (`useCaptureState`), and a lost event costs a banner a few seconds, not a
/// meeting.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SpeechState {
    /// Nothing needs it and nothing is loaded. There is nothing to say.
    #[default]
    Idle,
    /// A meeting needs it and it is not up yet: the weights are being read, or
    /// the one-time setup they need on this machine is being paid.
    Preparing,
    /// Loaded. Words appear as they are said.
    Ready,
    /// It was needed and it did not come up. The recording carries on and the
    /// transcript arrives when the meeting ends (mantra 3).
    Unavailable,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptureStatus {
    pub state: CaptureState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meeting_id: Option<Id>,
    /// Elapsed time on the monotonic meeting clock.
    pub elapsed_ms: i64,
    /// Channels currently producing audio.
    pub active_channels: Vec<Channel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub degraded_reason: Option<DegradedReason>,
    /// How many utterances are waiting for text. UI shows this only as
    /// "catching up", never as a number of jobs.
    pub pending_utterances: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub started_at: Option<Timestamp>,
    /// Whether Echo can understand speech right now. Derived on every read from
    /// what the engine is actually holding, never remembered, so no missed event
    /// can leave it stale (see [`SpeechState`]).
    pub speech: SpeechState,
}

/// Options for `start_recording`. All fields optional so the UI can call it
/// with `{}` from the big Start button.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct StartRecordingOptions {
    /// Title the person typed; otherwise Echo names it from date + detected app.
    pub title: Option<String>,
    /// Capture what the computer plays as well as the microphone.
    pub capture_system_audio: Option<bool>,
    /// Override the input device (Settings → General).
    pub input_device_id: Option<String>,
    /// App name that triggered detection, stored for provenance.
    pub detected_app: Option<String>,
}

/// What to do with a meeting that was interrupted by a crash or power loss.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RecoveryAction {
    /// Finish transcribing from the audio already on disk.
    Finish,
    /// Throw the interrupted meeting away.
    Discard,
}

// ---------------------------------------------------------------------------
// Meetings
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MeetingStatus {
    /// Row exists, capture has not started (created before capture, §3).
    #[default]
    Created,
    Recording,
    /// Capture ended, background work still running.
    Processing,
    /// Everything the user asked for is done.
    Complete,
    /// Capture ended abnormally; audio on disk is still authoritative.
    Interrupted,
    Failed,
}

impl MeetingStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            MeetingStatus::Created => "created",
            MeetingStatus::Recording => "recording",
            MeetingStatus::Processing => "processing",
            MeetingStatus::Complete => "complete",
            MeetingStatus::Interrupted => "interrupted",
            MeetingStatus::Failed => "failed",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "created" => Some(MeetingStatus::Created),
            "recording" => Some(MeetingStatus::Recording),
            "processing" => Some(MeetingStatus::Processing),
            "complete" => Some(MeetingStatus::Complete),
            "interrupted" => Some(MeetingStatus::Interrupted),
            "failed" => Some(MeetingStatus::Failed),
            _ => None,
        }
    }
}

/// Row of `meetings`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Meeting {
    pub id: Id,
    pub title: String,
    pub started_at: Timestamp,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<Timestamp>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detected_app: Option<String>,
    /// BCP-47-ish dominant language, detected not configured.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    pub status: MeetingStatus,
    pub audio_dir: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mixed_path: Option<String>,
    pub duration_ms: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deleted_at: Option<Timestamp>,
}

/// What the Home and History lists render.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MeetingSummary {
    pub id: Id,
    pub title: String,
    pub started_at: Timestamp,
    pub duration_ms: i64,
    pub status: MeetingStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// First useful line of the recap, or of the transcript if there is no recap.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
    pub has_recap: bool,
    pub has_audio: bool,
    pub speaker_count: u32,
    pub action_item_count: u32,
}

/// Everything the meeting detail route needs in one round-trip.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MeetingDetail {
    pub meeting: Meeting,
    pub speakers: Vec<Speaker>,
    pub markers: Vec<Marker>,
    pub summaries: Vec<Summary>,
    pub action_items: Vec<ActionItem>,
    pub jobs: Vec<Job>,
    /// Bytes on disk for this meeting's audio.
    pub audio_bytes: u64,
    pub segment_count: u32,
    /// Which channels were actually captured.
    pub captured_channels: Vec<Channel>,
    /// How many people were in this meeting, counting the person at this
    /// computer. Echo's own count unless the person corrected it.
    pub people_count: u32,
    /// True when [`Self::people_count`] is the person's correction rather than
    /// Echo's count. The UI says "detected" for the one and nothing for the
    /// other.
    pub people_count_is_override: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct MeetingQuery {
    pub limit: Option<u32>,
    pub offset: Option<u32>,
    /// Free-text title filter (not FTS; use `search_transcripts` for that).
    pub title_contains: Option<String>,
    pub include_deleted: Option<bool>,
    pub status: Option<MeetingStatus>,
}

/// Deleting is a first-class, explainable operation (DESIGN §3 Provenance).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum DeleteMode {
    /// Remove the recording but keep the transcript and recap.
    AudioOnly,
    /// Remove everything about this meeting.
    Everything,
}

// ---------------------------------------------------------------------------
// Audio chunks (crash-recovery journal)
// ---------------------------------------------------------------------------

/// Row of `audio_chunks`. Raw per-channel audio is the source of truth
/// (mantra 3); a chunk only counts once `committed` is true.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioChunk {
    pub id: Id,
    pub meeting_id: Id,
    pub channel: Channel,
    pub seq: i64,
    pub path: String,
    pub t_start_ms: i64,
    pub t_end_ms: i64,
    pub committed: bool,
}

// ---------------------------------------------------------------------------
// Words Echo should know
// ---------------------------------------------------------------------------

/// One word in the vocabulary, and where it came from.
///
/// The two sources behave differently when somebody deletes one, which is the
/// whole reason this is not a plain list of strings: a typed word is deleted by
/// forgetting it, while a name that comes from an enrolled person has to be
/// remembered *as deleted* or the next launch would derive it all over again.
/// See [`crate::settings::words_to_know`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VocabularyWord {
    pub word: String,
    pub source: VocabularySource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum VocabularySource {
    /// Somebody typed it.
    Typed,
    /// The name of a person whose voice Echo was asked to remember.
    Person,
}

// ---------------------------------------------------------------------------
// Segments
// ---------------------------------------------------------------------------

/// One word Echo put right after the engine wrote it down, and what it had
/// written.
///
/// Kept with the line rather than thrown away, for two reasons that are really
/// the same reason: a person reading a transcript is entitled to know that a
/// word in it is not the word the engine produced, and anything Echo changed on
/// its own has to be reversible. `from` is exactly the run of text that was
/// replaced, `to` exactly what replaced it — put `from` back and the line is the
/// line the engine wrote.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Correction {
    pub from: String,
    pub to: String,
}

/// Row of `segments`. `revision` increases when a later, better pass replaces
/// the text or the speaker (live partial → final → diarization-refined).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Segment {
    pub id: Id,
    pub meeting_id: Id,
    pub t_start_ms: i64,
    pub t_end_ms: i64,
    pub channel: Channel,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker_id: Option<Id>,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_confidence: Option<f32>,
    pub revision: i64,
    pub is_final: bool,
    /// Which speech engine + revision produced this (provenance, §3).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_revision: Option<String>,
    /// Words put right against the list in Settings, empty on the vast majority
    /// of lines. See [`Correction`] and [`crate::asr::glossary`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub corrections: Vec<Correction>,
}

/// A stretch of a meeting that has no words in it because Echo heard it and
/// decided not to write it down.
///
/// Not every such decision is one of these — nearly all of them are the far
/// side written down once instead of twice, and those seconds have words.
/// [`crate::asr::left_out`] is where that difference is argued, and this is
/// only what survives it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LeftOutMoment {
    pub t_start_ms: i64,
    pub t_end_ms: i64,
}

/// A segment as it is being written, before it lands in the database.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SegmentDraft {
    pub meeting_id: Id,
    pub t_start_ms: i64,
    pub t_end_ms: i64,
    pub channel: Channel,
    pub speaker_id: Option<Id>,
    pub text: String,
    pub language: Option<String>,
    pub avg_confidence: Option<f32>,
    pub revision: i64,
    pub is_final: bool,
    pub model_name: Option<String>,
    pub model_revision: Option<String>,
    /// See [`Segment::corrections`].
    #[serde(default)]
    pub corrections: Vec<Correction>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct TranscriptQuery {
    pub meeting_id: Id,
    /// Window on the meeting clock, for the virtualized transcript view.
    pub from_ms: Option<i64>,
    pub to_ms: Option<i64>,
    pub limit: Option<u32>,
    /// Include not-yet-final live text.
    pub include_partial: Option<bool>,
}

// ---------------------------------------------------------------------------
// Speakers
// ---------------------------------------------------------------------------

/// Row of `speakers`. Merging is non-destructive: the merged speaker keeps its
/// row and points at the survivor via `alias_of`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Speaker {
    pub id: Id,
    pub meeting_id: Id,
    /// Stable key from channel attribution or from clustering.
    pub cluster_key: String,
    /// "You", "Speaker 1", or whatever the person renamed it to.
    pub display_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alias_of: Option<Id>,
    /// True for the microphone channel, always "You" until renamed.
    pub is_self: bool,
    /// Total speaking time, for the Info tab.
    pub speaking_ms: i64,
    /// The known person this voice was matched to, when Echo is sure enough to
    /// say so (`diarize::people::TAU_LINK` plus the margin rule).
    ///
    /// `display_name` was copied from the person when the link was made and is
    /// meeting-local from then on: renaming this speaker does **not** rename the
    /// person, and deleting the person does not rewrite this meeting.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub person_id: Option<Id>,
    /// "Looks like Marco — confirm?" — a match that cleared the suggestion bar
    /// but not the linking one. Never an assignment; the person confirms it or
    /// it stays a chip. Look the name up in `listPeople`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggested_person_id: Option<Id>,
    /// How alike the two voices were, `0.0..=1.0`. Kept so the suggestion can be
    /// re-decided if the thresholds move; not for showing to anybody (mantra 2).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggestion_score: Option<f32>,
}

// ---------------------------------------------------------------------------
// Known people (voice enrollment) — DESIGN §1
// ---------------------------------------------------------------------------

/// One remembered voice, as Settings → People shows it.
///
/// Deliberately not a profile: no centroid, no clips, no embedder tag. What
/// leaves the core is a name, when the voice was last heard, how much of it Echo
/// has kept, and whether that material is currently usable.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PersonInfo {
    pub id: Id,
    pub name: String,
    /// How many samples of this voice Echo is keeping (capped — see
    /// `diarize::people::MAX_SAMPLES`).
    pub sample_count: u32,
    /// When this voice was last heard in a meeting, RFC3339. Absent for a person
    /// who has been enrolled but not met since.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_heard_at: Option<Timestamp>,
    /// The stored numbers belong to an older way of listening, so this voice is
    /// not being matched at the moment. Echo redoes it from the clips it kept;
    /// nothing is lost and nobody has to re-enroll. The UI says something calm,
    /// or nothing at all — never a technical reason (mantra 2).
    pub needs_refresh: bool,
}

/// A voice that keeps turning up without a name, offered as "shall I remember
/// this one?".
///
/// Built by cross-matching the per-meeting voice prints the offline pass stores
/// on `speakers`, so it costs no audio reads. `meeting_id` + `speaker_id` are a
/// representative appearance, for Listen (`speaker_sample`) and for enrolling
/// with `enrollSpeakerAsPerson`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SuggestedPerson {
    /// Stable for as long as the same appearances are grouped; not a row id.
    pub id: String,
    /// How many meetings this voice has been in.
    pub appearances: u32,
    /// The most recent of those meetings, RFC3339.
    pub last_heard_at: Timestamp,
    /// Total speech Echo has of this voice across those meetings.
    pub speaking_ms: i64,
    /// A representative appearance: the meeting where this voice said the most.
    pub meeting_id: Id,
    pub speaker_id: Id,
    /// That meeting's title, so the UI can say where the voice is from.
    pub meeting_title: String,
}

// ---------------------------------------------------------------------------
// Markers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MarkerKind {
    /// "Flag action item" button during a live meeting.
    #[default]
    ActionItem,
    /// Generic "remember this".
    Highlight,
    /// Automatic note (capture degraded here, device changed, etc).
    System,
}

impl MarkerKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            MarkerKind::ActionItem => "action_item",
            MarkerKind::Highlight => "highlight",
            MarkerKind::System => "system",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "action_item" => Some(MarkerKind::ActionItem),
            "highlight" => Some(MarkerKind::Highlight),
            "system" => Some(MarkerKind::System),
            _ => None,
        }
    }
}

/// Row of `markers`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Marker {
    pub id: Id,
    pub meeting_id: Id,
    pub t_ms: i64,
    pub kind: MarkerKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

// ---------------------------------------------------------------------------
// Summaries, action items, templates
// ---------------------------------------------------------------------------

/// Where recaps are written. User-facing names live in `copy.ts`
/// ("On this computer" / "Google Gemini").
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Provider {
    /// Local generation, nothing leaves the machine.
    #[default]
    OnThisComputer,
    /// Google Gemini via an AI Studio key; text leaves the machine.
    Gemini,
}

impl Provider {
    pub fn as_str(&self) -> &'static str {
        match self {
            Provider::OnThisComputer => "onThisComputer",
            Provider::Gemini => "gemini",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "onThisComputer" | "ollama" => Some(Provider::OnThisComputer),
            "gemini" => Some(Provider::Gemini),
            _ => None,
        }
    }
}

/// What a summary backend can do. Drives chunk sizing and streaming
/// (DESIGN §3 Connectors).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Caps {
    /// Backend can be asked for strict JSON.
    pub json_mode: bool,
    /// Backend can stream partial text.
    pub streaming: bool,
    /// Usable input size in characters (not tokens, the UI never sees tokens).
    pub context_chars: u32,
    /// Backend can list the choices available to the person.
    pub can_list_models: bool,
    /// Text is sent off this machine.
    pub leaves_machine: bool,
}

impl Default for Caps {
    fn default() -> Self {
        Self {
            json_mode: false,
            streaming: false,
            context_chars: 8_000,
            can_list_models: false,
            leaves_machine: false,
        }
    }
}

/// Row of `templates`, 6 built-ins plus custom ones.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Template {
    pub id: Id,
    pub name: String,
    pub prompt_md: String,
    pub builtin: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct TemplateDraft {
    /// Absent = create, present = update.
    pub id: Option<Id>,
    pub name: String,
    pub prompt_md: String,
}

/// What language the recap is written in (DESIGN §1).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "kind", content = "value")]
pub enum SummaryLanguage {
    /// Same language as the meeting.
    #[default]
    SameAsMeeting,
    English,
    /// A language the person picked.
    Fixed(String),
}

/// Request for a recap. Named `SummaryReq` in DESIGN §3.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct SummaryReq {
    pub meeting_id: Id,
    /// Which template to use; defaults to the general recap.
    pub template_id: Option<Id>,
    /// Override the configured backend (the "write it again with something
    /// else" button).
    pub provider: Option<Provider>,
    /// Ignored, and kept only so a job queued by an older build still
    /// deserializes. Which model writes a recap is part of the chosen backend's
    /// saved configuration ([`crate::summarize::Connector::model`]), not a
    /// per-recap decision — nothing should send this.
    pub model: Option<String>,
    pub language: Option<SummaryLanguage>,
    /// Regenerate even if a recap for this transcript revision exists.
    pub force: Option<bool>,
}

/// Row of `summaries`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Summary {
    pub id: Id,
    pub meeting_id: Id,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub template_id: Option<Id>,
    /// The prompt exactly as used, so an old recap stays explainable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub template_snapshot: Option<String>,
    pub provider: Provider,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Transcript revision this recap was written from.
    pub transcript_revision: i64,
    /// Markdown, already sanitized for render (no raw HTML, no remote assets).
    pub content_md: String,
    pub created_at: Timestamp,
}

/// Row of `action_items`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ActionItem {
    pub id: Id,
    pub meeting_id: Id,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary_id: Option<Id>,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub owner: Option<String>,
    /// Free text as spoken ("before Friday"), never a parsed date.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub due_hint: Option<String>,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub external_url: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ActionItemPatch {
    pub id: Id,
    pub description: Option<String>,
    pub owner: Option<String>,
    pub due_hint: Option<String>,
    pub done: Option<bool>,
    pub external_url: Option<String>,
}

/// Configuration for a summary backend, minus the secret (which lives in the
/// keychain and never crosses IPC).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ProviderConfig {
    pub provider: Provider,
    /// Local backend address. Loopback by default; anything else gets a
    /// "this leaves your machine" warning first.
    pub base_url: Option<String>,
    pub model: Option<String>,
    /// Only meaningful for backends that need a key. `true` if one is stored.
    pub has_key: bool,
    pub enabled: bool,
    /// The person has been shown, and agreed to, "this address is not on your
    /// computer, so what was said leaves it". Required before a non-loopback
    /// address is accepted (DESIGN §3 Connectors); ignored otherwise.
    #[serde(default)]
    pub leaves_machine_acknowledged: bool,
}

/// Result of the "check this works" button on the Summaries settings screen.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderTestResult {
    pub ok: bool,
    /// Plain sentence: "Ready to write recaps." / "Couldn't reach it."
    pub message: String,
    pub caps: Caps,
    pub models: Vec<String>,
    /// True when the address is not loopback, so the UI can warn.
    pub leaves_machine: bool,
}

/// One row of the Summaries settings screen.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderInfo {
    pub provider: Provider,
    pub config: ProviderConfig,
    pub caps: Caps,
    /// Detected as usable right now without any setup.
    pub available: bool,
}

// ---------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum JobKind {
    /// Transcribe the parts that live capture could not keep up with.
    #[default]
    TranscribeCatchup,
    /// The canonical offline speaker pass.
    Diarize,
    Summarize,
    Export,
    /// Fetch what Echo needs to understand speech.
    Download,
    /// Build the mixed file used for playback.
    Mixdown,
    /// Pay the one-time setup a set of speech weights needs on this machine,
    /// before a meeting has to pay it (incident of 2026-08-24).
    PrepareEngine,
}

impl JobKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            JobKind::TranscribeCatchup => "transcribe_catchup",
            JobKind::Diarize => "diarize",
            JobKind::Summarize => "summarize",
            JobKind::Export => "export",
            JobKind::Download => "download",
            JobKind::Mixdown => "mixdown",
            JobKind::PrepareEngine => "prepare_engine",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "transcribe_catchup" => Some(JobKind::TranscribeCatchup),
            "diarize" => Some(JobKind::Diarize),
            "summarize" => Some(JobKind::Summarize),
            "export" => Some(JobKind::Export),
            "download" => Some(JobKind::Download),
            "mixdown" => Some(JobKind::Mixdown),
            "prepare_engine" => Some(JobKind::PrepareEngine),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum JobStatus {
    #[default]
    Queued,
    Running,
    /// Yielded because a recording started (recording preempts everything).
    Paused,
    Done,
    Failed,
    Cancelled,
}

impl JobStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            JobStatus::Queued => "queued",
            JobStatus::Running => "running",
            JobStatus::Paused => "paused",
            JobStatus::Done => "done",
            JobStatus::Failed => "failed",
            JobStatus::Cancelled => "cancelled",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "queued" => Some(JobStatus::Queued),
            "running" => Some(JobStatus::Running),
            "paused" => Some(JobStatus::Paused),
            "done" => Some(JobStatus::Done),
            "failed" => Some(JobStatus::Failed),
            "cancelled" => Some(JobStatus::Cancelled),
            _ => None,
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            JobStatus::Done | JobStatus::Failed | JobStatus::Cancelled
        )
    }
}

/// A stage inside one job that a person would read as a different activity.
///
/// Almost no job needs one: "Writing your recap…" is the whole of what the
/// recap job does. The download is the exception. Its second half is not a
/// download at all — the bytes have arrived and are being made ready to use on
/// this particular machine, which on Apple silicon is a one-off that can take
/// many minutes — and a progress bar that has been sitting at 100% since the
/// bytes landed is not an honest account of it (field report of 2026-08-21: a
/// person watched a bar for eighteen minutes with no idea what was happening,
/// under a sentence about a job that had not started).
///
/// Stored on the job row rather than only announced, because the stage that
/// matters most begins before any screen exists to hear about it — see
/// `migrations/0009_job_stage.sql`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum JobPhase {
    /// The bytes are in; this machine is being got ready to use them. There is
    /// no fraction to report and none can be invented, so the UI shows this as
    /// work in progress without a number.
    PreparingEngine,
}

impl JobPhase {
    pub fn as_str(&self) -> &'static str {
        match self {
            JobPhase::PreparingEngine => "preparing_engine",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "preparing_engine" => Some(JobPhase::PreparingEngine),
            _ => None,
        }
    }
}

/// Row of `jobs`. Persisted so work survives a restart.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Job {
    pub id: Id,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meeting_id: Option<Id>,
    pub kind: JobKind,
    pub status: JobStatus,
    /// 0.0..=1.0, or None when the total is genuinely unknown.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
    /// The stage this job is in right now, when it is in one worth naming.
    ///
    /// Only ever set on a row that is running, and cleared by every write that
    /// changes the status or reports a fraction, so a screen can read it
    /// straight off the row without asking when it was last true.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase: Option<JobPhase>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct JobQuery {
    pub meeting_id: Option<Id>,
    pub kind: Option<JobKind>,
    pub status: Option<JobStatus>,
    pub limit: Option<u32>,
    /// Only jobs that are not finished.
    pub active_only: Option<bool>,
}

// ---------------------------------------------------------------------------
// Speech assets ("what Echo needs to understand speech")
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AssetKind {
    /// Main speech-to-text weights.
    #[default]
    Speech,
    /// Apple-specific encoder companion for the speech asset.
    SpeechAccelerator,
    /// Speech/silence detector.
    SpeechDetector,
    /// Speaker segmentation.
    SpeakerSegmenter,
    /// Speaker fingerprints.
    SpeakerEmbedder,
}

impl AssetKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            AssetKind::Speech => "speech",
            AssetKind::SpeechAccelerator => "speech_accelerator",
            AssetKind::SpeechDetector => "speech_detector",
            AssetKind::SpeakerSegmenter => "speaker_segmenter",
            AssetKind::SpeakerEmbedder => "speaker_embedder",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "speech" => Some(AssetKind::Speech),
            "speech_accelerator" => Some(AssetKind::SpeechAccelerator),
            "speech_detector" => Some(AssetKind::SpeechDetector),
            "speaker_segmenter" => Some(AssetKind::SpeakerSegmenter),
            "speaker_embedder" => Some(AssetKind::SpeakerEmbedder),
            _ => None,
        }
    }
}

/// Row of `models`. Technical detail: only Settings → Advanced renders the
/// name, url or revision.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    pub id: Id,
    pub kind: AssetKind,
    pub name: String,
    pub url: String,
    pub sha256: String,
    pub bytes: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    pub installed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

/// A quality preset as the person sees it: "Everyday", "Careful", "Fastest".
/// Never a model name outside Advanced.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AccuracyLevel {
    /// Stable machine id, e.g. "everyday".
    pub id: String,
    /// "Everyday accuracy"
    pub name: String,
    /// "Good for most meetings. Uses about 1.6 GB of space."
    pub description: String,
    pub download_bytes: i64,
    pub installed: bool,
    /// Currently selected.
    pub selected: bool,
    /// Recommended for this computer.
    pub recommended: bool,
    /// Model ids this level needs (Advanced / internal).
    pub asset_ids: Vec<Id>,
}

/// Answer to "can Echo understand speech right now?".
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeechReadiness {
    /// Everything needed to record and get text is on disk.
    pub ready: bool,
    /// A download is in flight.
    pub downloading: bool,
    /// Bytes still to fetch, for the onboarding progress copy.
    pub remaining_bytes: i64,
    /// Currently selected accuracy level id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level_id: Option<String>,
    /// Loaded in memory right now (mantra 1, normally false when idle).
    pub loaded: bool,
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum DetectionState {
    #[default]
    Idle,
    /// Signals say a meeting is happening.
    Detected,
    /// Muted by the person for a while.
    Snoozed,
    /// Turned off in Settings.
    Off,
}

/// One thing the watcher can see. Not on its own a reason to believe a meeting
/// is happening — see `confidence`, and the decision table in `crate::detect`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DetectionSignal {
    pub source: DetectionSource,
    /// e.g. "zoom.us", shown as "Zoom looks like it's running".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub app: Option<String>,
    pub since: Timestamp,
    /// What this observation is worth on its own: an app merely running is
    /// close to nothing, a live microphone is the real signal, and the two
    /// together are as sure as Echo gets before it opens the audio stream.
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum DetectionSource {
    /// A known meeting app is running. Corroborating evidence only: Slack and
    /// Discord sit open all day, so this can never mean a meeting by itself.
    #[default]
    MeetingApp,
    /// Something other than Echo is using the microphone.
    InputDeviceInUse,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DetectionStatus {
    pub state: DetectionState,
    pub enabled: bool,
    pub signals: Vec<DetectionSignal>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub snoozed_until: Option<Timestamp>,
    /// The microphone has been quiet long enough, while recording, that we
    /// suggest stopping. A meeting app that is merely still running does not
    /// hold this back — that is how "you forgot to stop" gets caught.
    pub suggest_stop: bool,
}

// ---------------------------------------------------------------------------
// Permissions & capabilities
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PermissionState {
    #[default]
    Unknown,
    Granted,
    Denied,
    /// Asked but the person has not answered yet.
    Prompting,
    /// Granted, but this platform needs a restart before it takes effect.
    RestartRequired,
    /// Not a thing on this platform.
    NotApplicable,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PermissionStatus {
    pub microphone: PermissionState,
    /// macOS Screen & System Audio Recording; on Linux, PipeWire availability.
    pub system_audio: PermissionState,
    pub notifications: PermissionState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PermissionTarget {
    Microphone,
    SystemAudio,
    Notifications,
}

/// Settings → Advanced only. Every string here may be technical.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SystemCapabilities {
    pub os: String,
    pub arch: String,
    /// Compiled-in speech backend, e.g. "metal+coreml".
    pub speech_backend: String,
    /// What actually initialised at runtime, e.g. "metal" or "cpu".
    pub speech_backend_active: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_fallback_reason: Option<String>,
    pub cpu_threads: u32,
    pub total_memory_bytes: u64,
    pub system_audio_supported: bool,
    pub tray_supported: bool,
    pub app_version: String,
}

// ---------------------------------------------------------------------------
// Devices & playback
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioDevice {
    pub id: String,
    pub name: String,
    pub is_default: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channels: Option<u16>,
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct SearchQuery {
    /// Raw user input. Escaped before it reaches FTS5 MATCH.
    pub text: String,
    pub meeting_id: Option<Id>,
    pub limit: Option<u32>,
    pub offset: Option<u32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchHit {
    pub meeting_id: Id,
    pub meeting_title: String,
    pub started_at: Timestamp,
    pub segment_id: Id,
    pub t_start_ms: i64,
    /// Snippet with `<mark>`…`</mark>` around matches, HTML-escaped elsewhere.
    pub snippet_html: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker_name: Option<String>,
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ExportFormat {
    #[default]
    Markdown,
    Pdf,
    Docx,
    /// Plain text transcript.
    Text,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ExportRequest {
    pub meeting_id: Id,
    pub format: ExportFormat,
    /// Where to write. If absent the caller must have already picked a path.
    pub destination: Option<String>,
    pub include_recap: Option<bool>,
    pub include_transcript: Option<bool>,
    pub include_action_items: Option<bool>,
    /// Use a specific recap rather than the latest.
    pub summary_id: Option<Id>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportResult {
    pub path: String,
    pub bytes: u64,
    pub format: ExportFormat,
}

// ---------------------------------------------------------------------------
// Storage & settings
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StorageReport {
    pub root: String,
    pub audio_bytes: u64,
    pub database_bytes: u64,
    pub speech_asset_bytes: u64,
    pub log_bytes: u64,
    pub total_bytes: u64,
    pub free_bytes: u64,
    pub meeting_count: u32,
    /// Biggest meetings first, for the "free up space" list.
    pub largest_meetings: Vec<MeetingSummary>,
}

/// Typed view over the `settings` key/value table. Non-secret only.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Settings {
    pub launch_at_login: bool,
    pub detection_enabled: bool,
    /// Where recordings live; configurable (DESIGN §2 Audio files).
    pub storage_dir: String,
    pub capture_system_audio: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_device_id: Option<String>,
    pub summary_language: SummaryLanguage,
    pub summary_provider: Provider,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary_template_id: Option<Id>,
    /// Write a recap as soon as a meeting ends. On unless turned off — see
    /// [`crate::settings::DEFAULT_AUTO_SUMMARIZE`].
    pub auto_summarize: bool,
    /// Quality preset id.
    pub accuracy_level_id: String,
    /// Close the window to the tray instead of quitting.
    pub close_to_tray: bool,
    pub onboarding_complete: bool,
    /// Show technical detail (engine, asset names, diagnostics).
    pub show_advanced: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            launch_at_login: false,
            detection_enabled: true,
            storage_dir: String::new(),
            capture_system_audio: true,
            input_device_id: None,
            summary_language: SummaryLanguage::SameAsMeeting,
            summary_provider: Provider::OnThisComputer,
            summary_template_id: None,
            // One source of truth for this default, so nothing that builds a
            // `Settings` without going through `settings::load` can quietly
            // decide recaps are off.
            auto_summarize: crate::settings::DEFAULT_AUTO_SUMMARIZE,
            accuracy_level_id: "everyday".to_string(),
            close_to_tray: true,
            onboarding_complete: false,
            show_advanced: false,
        }
    }
}

/// Partial update. Absent field = leave alone.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct SettingsPatch {
    pub launch_at_login: Option<bool>,
    pub detection_enabled: Option<bool>,
    pub storage_dir: Option<String>,
    pub capture_system_audio: Option<bool>,
    pub input_device_id: Option<String>,
    pub summary_language: Option<SummaryLanguage>,
    pub summary_provider: Option<Provider>,
    pub summary_template_id: Option<Id>,
    pub auto_summarize: Option<bool>,
    pub accuracy_level_id: Option<String>,
    pub close_to_tray: Option<bool>,
    pub onboarding_complete: Option<bool>,
    pub show_advanced: Option<bool>,
}

// ---------------------------------------------------------------------------
// Onboarding
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum OnboardingStep {
    Welcome,
    Permissions,
    Download,
    Summaries,
    Done,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OnboardingState {
    pub complete: bool,
    pub completed_steps: Vec<String>,
    pub permissions: PermissionStatus,
    pub speech: SpeechReadiness,
    /// A local summary backend was found without any setup.
    pub local_summaries_available: bool,
}

// ---------------------------------------------------------------------------
// Tray
// ---------------------------------------------------------------------------

/// Tray icon appearance, driven by capture, detection and the work queue.
///
/// Which one is showing is never remembered anywhere: it is recomputed from
/// those three facts by [`crate::session::tray_state_for`] every time one of
/// them moves.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TrayState {
    #[default]
    Idle,
    Detected,
    Recording,
    /// The meeting is over and Echo is still finishing it: the rest of the
    /// transcript, who spoke, the playback file, the recap. Worth its own icon
    /// because the work outlives the recording by minutes and, until this
    /// existed, the menu bar said "nothing is happening" throughout.
    Processing,
}

/// How far along Echo is with fetching a newer version of itself.
///
/// Only `Ready` reaches a person. The rest exist so a screen that asks outright
/// gets a truthful answer, and so the log can tell "nothing to do" apart from
/// "could not ask" — which look identical from the outside and mean very
/// different things.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum UpdateState {
    /// Nothing to do: this is the newest Echo there is.
    #[default]
    Idle,
    Checking,
    Downloading,
    /// A new version is on disk. All that is left is a restart, and that is the
    /// person's to ask for.
    Ready,
    /// The last attempt did not work. Deliberately not shown to anybody: the app
    /// they have works, and the next attempt is half an hour away.
    Failed,
}

/// What the person picked in the tray menu, forwarded to the UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TrayAction {
    Start,
    Stop,
    Open,
    PauseDetection,
    Quit,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enums_round_trip_through_their_string_form() {
        for c in [Channel::Mic, Channel::System, Channel::Mixed] {
            assert_eq!(Channel::parse(c.as_str()), Some(c));
        }
        for s in [
            MeetingStatus::Created,
            MeetingStatus::Recording,
            MeetingStatus::Processing,
            MeetingStatus::Complete,
            MeetingStatus::Interrupted,
            MeetingStatus::Failed,
        ] {
            assert_eq!(MeetingStatus::parse(s.as_str()), Some(s));
        }
        for k in [
            JobKind::TranscribeCatchup,
            JobKind::Diarize,
            JobKind::Summarize,
            JobKind::Export,
            JobKind::Download,
            JobKind::Mixdown,
            JobKind::PrepareEngine,
        ] {
            assert_eq!(JobKind::parse(k.as_str()), Some(k));
        }
        for s in [
            JobStatus::Queued,
            JobStatus::Running,
            JobStatus::Paused,
            JobStatus::Done,
            JobStatus::Failed,
            JobStatus::Cancelled,
        ] {
            assert_eq!(JobStatus::parse(s.as_str()), Some(s));
        }
        let stage = JobPhase::PreparingEngine;
        assert_eq!(JobPhase::parse(stage.as_str()), Some(stage));
        for k in [
            MarkerKind::ActionItem,
            MarkerKind::Highlight,
            MarkerKind::System,
        ] {
            assert_eq!(MarkerKind::parse(k.as_str()), Some(k));
        }
        for a in [
            AssetKind::Speech,
            AssetKind::SpeechAccelerator,
            AssetKind::SpeechDetector,
            AssetKind::SpeakerSegmenter,
            AssetKind::SpeakerEmbedder,
        ] {
            assert_eq!(AssetKind::parse(a.as_str()), Some(a));
        }
        for p in [Provider::OnThisComputer, Provider::Gemini] {
            assert_eq!(Provider::parse(p.as_str()), Some(p));
        }
    }

    #[test]
    fn wire_format_is_camel_case() {
        let json = serde_json::to_string(&CaptureStatus {
            state: CaptureState::Recording,
            elapsed_ms: 1234,
            ..Default::default()
        })
        .unwrap();
        assert!(json.contains("\"elapsedMs\":1234"), "{json}");
        assert!(json.contains("\"state\":\"recording\""), "{json}");
    }

    #[test]
    fn summary_language_is_tagged() {
        let json = serde_json::to_string(&SummaryLanguage::Fixed("it".into())).unwrap();
        assert_eq!(json, r#"{"kind":"fixed","value":"it"}"#);
        let same = serde_json::to_string(&SummaryLanguage::SameAsMeeting).unwrap();
        assert_eq!(same, r#"{"kind":"sameAsMeeting"}"#);
    }

    #[test]
    fn job_status_terminality() {
        assert!(JobStatus::Done.is_terminal());
        assert!(!JobStatus::Running.is_terminal());
        assert!(!JobStatus::Paused.is_terminal());
    }
}
