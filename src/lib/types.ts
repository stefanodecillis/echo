/**
 * Mirror of `src-tauri/src/types.rs`.
 *
 * Hand-written on purpose: the Rust side is the contract, and writing this by
 * hand means a change there fails a review here rather than silently going out
 * of sync. If you change one, change the other in the same commit.
 *
 * Conventions:
 *  - fields are camelCase because every Rust struct carries
 *    `#[serde(rename_all = "camelCase")]`
 *  - `Id` is a UUID v4 string
 *  - `Timestamp` is RFC3339 UTC
 *  - in-meeting offsets are milliseconds on the monotonic meeting clock
 *  - enum members are the exact serde strings
 */

export type Id = string;
export type Timestamp = string;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

export type UiErrorKind =
  | "permissionNeeded"
  | "notReady"
  | "storage"
  | "network"
  | "credential"
  | "invalidInput"
  | "notFound"
  | "cancelled"
  | "unexpected";

export type UiErrorAction =
  | "openMicrophoneSettings"
  | "openScreenRecordingSettings"
  | "openSpeechSettings"
  | "openSummarySettings"
  | "openStorageSettings"
  | "retry"
  | "restartApp";

/**
 * The only error shape a command rejects with. `message` is safe to show as-is;
 * `detail` is technical and belongs under Settings > Advanced.
 */
export interface UiError {
  kind: UiErrorKind;
  message: string;
  detail?: string;
  action?: UiErrorAction;
}

// ---------------------------------------------------------------------------
// Capture
// ---------------------------------------------------------------------------

export type CaptureState =
  | "idle"
  | "starting"
  | "recording"
  | "paused"
  | "stopping"
  | "stopped"
  | "failed"
  | "degraded"
  | "recovering";

export type Channel = "mic" | "system" | "mixed";

/**
 * The two system-audio reasons are not one reason twice.
 * `systemAudioUnavailable` means nothing this computer plays ever reached Echo,
 * so the whole meeting is in the microphone recording and the offline pass will
 * separate the voices in it. `systemAudioLost` means it was arriving and
 * stopped — the lines already written carry real names, and the microphone tail
 * is not separated by anything. They get different banners for that reason.
 */
export type DegradedReason =
  | "systemAudioUnavailable"
  | "systemAudioLost"
  | "microphoneUnavailable"
  | "nothingIsBeingHeard"
  | "transcriptBehind"
  | "storageLow";

/**
 * Whether Echo can understand speech right now.
 *
 * The core works this out fresh from what the engine is holding every time a
 * `CaptureStatus` is built, and remembers it nowhere. That is what makes it safe
 * to draw a banner from — but only along with the other half: while it says
 * "preparing" or "unavailable", `useCaptureState` asks for the status outright
 * every few seconds, so a missed event costs the banner a moment rather than the
 * rest of the meeting.
 */
export type SpeechState = "idle" | "preparing" | "ready" | "unavailable";

export interface CaptureStatus {
  state: CaptureState;
  meetingId?: Id;
  elapsedMs: number;
  activeChannels: Channel[];
  degradedReason?: DegradedReason;
  pendingUtterances: number;
  startedAt?: Timestamp;
  /** Always sent by the core — see `SpeechState`. */
  speech: SpeechState;
}

export interface StartRecordingOptions {
  title?: string;
  captureSystemAudio?: boolean;
  inputDeviceId?: string;
  detectedApp?: string;
}

export type RecoveryAction = "finish" | "discard";

// ---------------------------------------------------------------------------
// Meetings
// ---------------------------------------------------------------------------

export type MeetingStatus =
  | "created"
  | "recording"
  | "processing"
  | "complete"
  | "interrupted"
  | "failed";

export interface Meeting {
  id: Id;
  title: string;
  startedAt: Timestamp;
  endedAt?: Timestamp;
  detectedApp?: string;
  language?: string;
  status: MeetingStatus;
  audioDir: string;
  mixedPath?: string;
  durationMs: number;
  deletedAt?: Timestamp;
}

export interface MeetingSummary {
  id: Id;
  title: string;
  startedAt: Timestamp;
  durationMs: number;
  status: MeetingStatus;
  language?: string;
  snippet?: string;
  hasRecap: boolean;
  hasAudio: boolean;
  speakerCount: number;
  actionItemCount: number;
}

export interface MeetingDetail {
  meeting: Meeting;
  speakers: Speaker[];
  markers: Marker[];
  summaries: Summary[];
  actionItems: ActionItem[];
  jobs: Job[];
  audioBytes: number;
  segmentCount: number;
  capturedChannels: Channel[];
  /**
   * How many people were in this meeting, counting whoever was at this
   * computer. Echo's own count unless the person corrected it with
   * `setSpeakerCount`.
   */
  peopleCount: number;
  /** True when `peopleCount` is the person's correction rather than Echo's. */
  peopleCountIsOverride: boolean;
}

export interface MeetingQuery {
  limit?: number;
  offset?: number;
  titleContains?: string;
  includeDeleted?: boolean;
  status?: MeetingStatus;
}

export type DeleteMode = "audioOnly" | "everything";

// ---------------------------------------------------------------------------
// Audio chunks, segments, speakers, markers
// ---------------------------------------------------------------------------

export interface AudioChunk {
  id: Id;
  meetingId: Id;
  channel: Channel;
  seq: number;
  path: string;
  tStartMs: number;
  tEndMs: number;
  committed: boolean;
}

export interface Segment {
  id: Id;
  meetingId: Id;
  tStartMs: number;
  tEndMs: number;
  channel: Channel;
  speakerId?: Id;
  text: string;
  language?: string;
  avgConfidence?: number;
  revision: number;
  isFinal: boolean;
  modelName?: string;
  modelRevision?: string;
  /**
   * Words Echo put right against the list in Settings, absent on nearly every
   * line. `from` is what was written down, `to` what replaced it — enough to
   * show what happened and to put it back.
   */
  corrections?: Correction[];
}

export interface Correction {
  from: string;
  to: string;
}

/** One word in "Words Echo should know", and where it came from. */
export interface VocabularyWord {
  word: string;
  source: VocabularySource;
}

/** `person` means Echo added it itself, from a voice somebody saved. */
export type VocabularySource = "typed" | "person";

export interface TranscriptQuery {
  meetingId: Id;
  fromMs?: number;
  toMs?: number;
  limit?: number;
  includePartial?: boolean;
}

export interface Speaker {
  id: Id;
  meetingId: Id;
  clusterKey: string;
  displayName: string;
  aliasOf?: Id;
  isSelf: boolean;
  speakingMs: number;
  /**
   * The known person this voice was matched to, when Echo is sure enough to say
   * so. `displayName` was copied from that person when the link was made and is
   * meeting-local from then on: renaming this speaker does **not** rename the
   * person, and deleting the person leaves this meeting reading as it does now.
   */
  personId?: Id;
  /**
   * "Looks like Marco — confirm?". A match that cleared the asking bar but not
   * the claiming one, so it is a question and never an assignment. Look the name
   * up in the list from `listPeople`.
   */
  suggestedPersonId?: Id;
  /** How alike the two voices were, 0..1. Internal; never shown (mantra 2). */
  suggestionScore?: number;
}

// ---------------------------------------------------------------------------
// Known people (voice enrollment)
// ---------------------------------------------------------------------------

/** One remembered voice, as Settings > People shows it. */
export interface PersonInfo {
  id: Id;
  name: string;
  /** How many samples of this voice Echo is keeping. Capped. */
  sampleCount: number;
  /** When this voice was last heard in a meeting. Absent if never since. */
  lastHeardAt?: Timestamp;
  /**
   * The stored voice data belongs to an older way of listening, so this person
   * is not being matched at the moment. Echo redoes it from the clips it kept —
   * nothing is lost and nobody has to be enrolled again. Say something calm, or
   * nothing; never a technical reason (mantra 2).
   */
  needsRefresh: boolean;
}

/**
 * A voice that keeps turning up without a name, offered as "shall I remember
 * this one?".
 *
 * `meetingId` + `speakerId` are a representative appearance: use them for
 * `speakerSample` (Listen) and for `acceptSuggestedPerson`.
 */
export interface SuggestedPerson {
  id: string;
  /** How many meetings this voice has been in. At least three. */
  appearances: number;
  lastHeardAt: Timestamp;
  /** Total speech Echo has of this voice across those meetings. */
  speakingMs: number;
  meetingId: Id;
  speakerId: Id;
  meetingTitle: string;
}

export type MarkerKind = "actionItem" | "highlight" | "system";

export interface Marker {
  id: Id;
  meetingId: Id;
  tMs: number;
  kind: MarkerKind;
  note?: string;
}

// ---------------------------------------------------------------------------
// Recaps
// ---------------------------------------------------------------------------

/** Machine names. The words a person reads live in `copy.ts`. */
export type Provider = "onThisComputer" | "gemini";

export interface Caps {
  jsonMode: boolean;
  streaming: boolean;
  contextChars: number;
  canListModels: boolean;
  leavesMachine: boolean;
}

export interface Template {
  id: Id;
  name: string;
  promptMd: string;
  builtin: boolean;
}

export interface TemplateDraft {
  id?: Id;
  name: string;
  promptMd: string;
}

export type SummaryLanguage =
  | { kind: "sameAsMeeting" }
  | { kind: "english" }
  | { kind: "fixed"; value: string };

export interface SummaryReq {
  meetingId: Id;
  templateId?: Id;
  provider?: Provider;
  /**
   * Ignored by the core, and only still here so an older queued job
   * deserializes. Which model writes a recap belongs to the chosen backend's
   * settings, not to one recap — don't send it.
   *
   * @deprecated
   */
  model?: string;
  language?: SummaryLanguage;
  force?: boolean;
}

export interface Summary {
  id: Id;
  meetingId: Id;
  templateId?: Id;
  templateSnapshot?: string;
  provider: Provider;
  model?: string;
  language?: string;
  transcriptRevision: number;
  contentMd: string;
  createdAt: Timestamp;
}

export interface ActionItem {
  id: Id;
  meetingId: Id;
  summaryId?: Id;
  description: string;
  owner?: string;
  dueHint?: string;
  done: boolean;
  externalUrl?: string;
}

export interface ActionItemPatch {
  id: Id;
  description?: string;
  owner?: string;
  dueHint?: string;
  done?: boolean;
  externalUrl?: string;
}

export interface ProviderConfig {
  provider: Provider;
  baseUrl?: string;
  model?: string;
  hasKey: boolean;
  enabled: boolean;
  /**
   * Set only after the person has read and accepted that an address which is
   * not on this computer means what was said leaves it. The core refuses a
   * non-loopback address without it.
   */
  leavesMachineAcknowledged?: boolean;
}

export interface ProviderTestResult {
  ok: boolean;
  message: string;
  caps: Caps;
  models: string[];
  leavesMachine: boolean;
}

export interface ProviderInfo {
  provider: Provider;
  config: ProviderConfig;
  caps: Caps;
  available: boolean;
}

// ---------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------

export type JobKind =
  | "transcribeCatchup"
  | "diarize"
  | "summarize"
  | "export"
  | "download"
  | "mixdown"
  | "prepareEngine";

export type JobStatus =
  | "queued"
  | "running"
  | "paused"
  | "done"
  | "failed"
  | "cancelled";

export interface Job {
  id: Id;
  meetingId?: Id;
  kind: JobKind;
  status: JobStatus;
  /** 0..1, or absent when the total is genuinely unknown. */
  progress?: number;
  error?: string;
  createdAt: Timestamp;
  updatedAt: Timestamp;
  /**
   * The stage this job is in right now, when it is in one worth naming.
   *
   * Part of the row, so a screen that opens in the middle of a stage reads it
   * the same as one that was already open. It used to travel only in the
   * `jobProgress` event, which meant the one stage that matters — the one-time
   * setup, announced at launch before the window has finished loading — was
   * invisible to every screen for the quarter of an hour it lasts.
   *
   * Only ever set on a running job, and cleared by the core the moment the job
   * moves on or has a fraction to report again.
   */
  phase?: JobPhase;
}

export interface JobQuery {
  meetingId?: Id;
  kind?: JobKind;
  status?: JobStatus;
  limit?: number;
  activeOnly?: boolean;
}

// ---------------------------------------------------------------------------
// Speech assets
// ---------------------------------------------------------------------------

export type AssetKind =
  | "speech"
  | "speechAccelerator"
  | "speechDetector"
  | "speakerSegmenter"
  | "speakerEmbedder";

/** Settings > Advanced only. Never render `name`, `url` or `revision` elsewhere. */
export interface ModelInfo {
  id: Id;
  kind: AssetKind;
  name: string;
  url: string;
  sha256: string;
  bytes: number;
  license?: string;
  revision?: string;
  installed: boolean;
  path?: string;
}

/**
 * The speech level as the person sees it. Plain words only.
 *
 * There is exactly one, and the core sends exactly one — Echo ships a single
 * model rather than a choice (see the core's `asr::catalog::PRESETS`). The shape
 * is still a list because `downloadBytes`, `installed` and `assetIds` are what
 * the download screens read, and because the next change of model wants it.
 */
export interface AccuracyLevel {
  id: string;
  name: string;
  description: string;
  downloadBytes: number;
  installed: boolean;
  selected: boolean;
  recommended: boolean;
  assetIds: Id[];
}

export interface SpeechReadiness {
  ready: boolean;
  downloading: boolean;
  remainingBytes: number;
  levelId?: string;
  loaded: boolean;
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

export type DetectionState = "idle" | "detected" | "snoozed" | "off";

export type DetectionSource = "meetingApp" | "inputDeviceInUse";

export interface DetectionSignal {
  source: DetectionSource;
  app?: string;
  since: Timestamp;
  confidence: number;
}

export interface DetectionStatus {
  state: DetectionState;
  enabled: boolean;
  signals: DetectionSignal[];
  snoozedUntil?: Timestamp;
  suggestStop: boolean;
}

// ---------------------------------------------------------------------------
// Permissions and capabilities
// ---------------------------------------------------------------------------

export type PermissionState =
  | "unknown"
  | "granted"
  | "denied"
  | "prompting"
  | "restartRequired"
  | "notApplicable";

export interface PermissionStatus {
  microphone: PermissionState;
  systemAudio: PermissionState;
  notifications: PermissionState;
}

export type PermissionTarget = "microphone" | "systemAudio" | "notifications";

/** Settings > Advanced only. These strings may be technical. */
export interface SystemCapabilities {
  os: string;
  arch: string;
  speechBackend: string;
  speechBackendActive: string;
  gpuFallbackReason?: string;
  cpuThreads: number;
  totalMemoryBytes: number;
  systemAudioSupported: boolean;
  traySupported: boolean;
  appVersion: string;
}

export interface AudioDevice {
  id: string;
  name: string;
  isDefault: boolean;
  sampleRate?: number;
  channels?: number;
}

// ---------------------------------------------------------------------------
// Search, export, storage, settings
// ---------------------------------------------------------------------------

export interface SearchQuery {
  text: string;
  meetingId?: Id;
  limit?: number;
  offset?: number;
}

export interface SearchHit {
  meetingId: Id;
  meetingTitle: string;
  startedAt: Timestamp;
  segmentId: Id;
  tStartMs: number;
  /** Already HTML-escaped, with `<mark>` around matches. Safe to set as HTML. */
  snippetHtml: string;
  speakerName?: string;
}

export type ExportFormat = "markdown" | "pdf" | "docx" | "text";

export interface ExportRequest {
  meetingId: Id;
  format: ExportFormat;
  destination?: string;
  includeRecap?: boolean;
  includeTranscript?: boolean;
  includeActionItems?: boolean;
  summaryId?: Id;
}

export interface ExportResult {
  path: string;
  bytes: number;
  format: ExportFormat;
}

export interface StorageReport {
  root: string;
  audioBytes: number;
  databaseBytes: number;
  speechAssetBytes: number;
  logBytes: number;
  totalBytes: number;
  freeBytes: number;
  meetingCount: number;
  largestMeetings: MeetingSummary[];
}

export interface Settings {
  launchAtLogin: boolean;
  detectionEnabled: boolean;
  storageDir: string;
  captureSystemAudio: boolean;
  inputDeviceId?: string;
  summaryLanguage: SummaryLanguage;
  summaryProvider: Provider;
  summaryTemplateId?: Id;
  autoSummarize: boolean;
  accuracyLevelId: string;
  closeToTray: boolean;
  onboardingComplete: boolean;
  showAdvanced: boolean;
}

/** Absent field means "leave it alone". */
export interface SettingsPatch {
  launchAtLogin?: boolean;
  detectionEnabled?: boolean;
  storageDir?: string;
  captureSystemAudio?: boolean;
  inputDeviceId?: string;
  summaryLanguage?: SummaryLanguage;
  summaryProvider?: Provider;
  summaryTemplateId?: Id;
  autoSummarize?: boolean;
  accuracyLevelId?: string;
  closeToTray?: boolean;
  onboardingComplete?: boolean;
  showAdvanced?: boolean;
}

// ---------------------------------------------------------------------------
// Onboarding and tray
// ---------------------------------------------------------------------------

export type OnboardingStep =
  | "welcome"
  | "permissions"
  | "download"
  | "summaries"
  | "done";

export interface OnboardingState {
  complete: boolean;
  completedSteps: string[];
  permissions: PermissionStatus;
  speech: SpeechReadiness;
  localSummariesAvailable: boolean;
}

/** Mirrors `TrayState` in types.rs. "processing" means the meeting has ended
 *  and Echo is still finishing it off. */
export type TrayState = "idle" | "detected" | "recording" | "processing";

export type TrayAction = "start" | "stop" | "open" | "pauseDetection" | "quit";

// ---------------------------------------------------------------------------
// Event payloads (mirror of src-tauri/src/events.rs)
// ---------------------------------------------------------------------------

export type CaptureStatePayload = CaptureStatus;

export interface TranscriptPartialPayload {
  meetingId: Id;
  /** Stable across updates, so the UI can replace the line in place. */
  utteranceId: string;
  tStartMs: number;
  tEndMs: number;
  channel: Channel;
  speakerId?: Id;
  text: string;
  language?: string;
  /** This utterance ended with nothing to write down — dropped to keep the
   * recording safe, heard as silence, or unreadable. No final will follow, so
   * the half-written line must be removed. */
  dropped?: boolean;
}

export interface TranscriptFinalPayload {
  meetingId: Id;
  utteranceId?: string;
  segment: Segment;
}

export interface TranscriptRevisedPayload {
  meetingId: Id;
  revision: number;
  fromMs: number;
  toMs: number;
  segmentIds: Id[];
}

export interface AudioLevelsPayload {
  meetingId: Id;
  mic: number;
  system: number;
  tMs: number;
}

/**
 * A stage inside one job that a person reads as a different activity.
 *
 * Only the download has one: its second half is not a download, and it is the
 * long half. `job.progress` is absent for a stage with no honest fraction.
 */
export type JobPhase = "preparingEngine";

export interface JobProgressPayload {
  job: Job;
  /** Ready-to-show sentence, e.g. "Writing your recap...". */
  label?: string;
}

export interface DownloadProgressPayload {
  assetId: Id;
  levelId?: string;
  receivedBytes: number;
  totalBytes: number;
  bytesPerSecond?: number;
  etaSeconds?: number;
  done: boolean;
  error?: string;
}

export type DetectionPayload = DetectionStatus;

/**
 * The speaker list changed — and with it, possibly, how many people Echo thinks
 * were in the meeting. The count travels with the rows because the two always
 * move together: the pass finishing is the moment both change.
 */
export interface SpeakersUpdatedPayload {
  meetingId: Id;
  speakers: Speaker[];
  /** People in the meeting, counting whoever was at this computer. */
  peopleCount: number;
  /** True when the count is the person's correction, not Echo's. */
  peopleCountIsOverride: boolean;
  /**
   * How many people the pass could actually tell apart, when this payload comes
   * from a pass that has just run. Absent from every other emitter, which are
   * announcing rows rather than a fresh separation.
   *
   * Below `peopleCount` when someone asked for more people than the recording
   * holds. "Name your speakers" says so beside the number.
   */
  voicesFound?: number;
  /**
   * The best count Echo decided against, when the count is Echo's own automatic
   * reading. Absent when the person set the count, and from every other
   * emitter.
   *
   * A wrong count is the one speaker mistake nothing in the UI can fix after
   * the fact (merges exist; splits do not), so the dialog shows what the
   * runner-up said rather than leaving the decision unarguable.
   */
  alternativeCount?: number;
}

/**
 * The people Echo remembers changed. Carries the whole list, so every view that
 * shows a name lands on the same answer without refetching.
 */
export interface PeopleUpdatedPayload {
  people: PersonInfo[];
}

export interface SummaryReadyPayload {
  meetingId: Id;
  summaryId: Id;
}

export interface ActionItemsUpdatedPayload {
  meetingId: Id;
  items: ActionItem[];
}

export interface MeetingUpdatedPayload {
  meetingId: Id;
  status: MeetingStatus;
  title?: string;
  durationMs: number;
  deleted: boolean;
}

export type NoticeLevel = "info" | "warning" | "problem";

export interface NoticePayload {
  level: NoticeLevel;
  message: string;
  persistent: boolean;
  meetingId?: Id;
  /** Machine tag for deduping, e.g. "systemAudioLost". */
  tag?: string;
}

export interface TrayStatePayload {
  state: TrayState;
}

export interface TrayActionPayload {
  action: TrayAction;
}

export type NavigateTarget =
  | "homeStart"
  | "home"
  | "live"
  | "meeting"
  | "search"
  | "settings"
  | "onboarding";

export interface NavigatePayload {
  target: NavigateTarget;
  meetingId?: Id;
  detectedApp?: string;
}

export interface RecoveryAvailablePayload {
  meetingIds: Id[];
}

/**
 * What the small floating panel is showing. Mirror of `PanelState` in
 * `src-tauri/src/events.rs`: a union tagged on `kind`, so the panel renders one
 * face or the other and never guesses which fields mean anything.
 *
 * Both times are epoch milliseconds on the same clock as `Date.now()`, so the
 * panel can count "3m ago" and the elapsed clock by itself instead of being fed
 * a tick every second.
 */
export type PanelState =
  | { kind: "detected"; detectedAtMs: number; appName?: string }
  | { kind: "recording"; startedAtMs: number; paused: boolean };

/**
 * `null` means the panel has left the screen: it drops what it was drawing, and
 * with it the once-a-second clock, instead of counting where nobody can see.
 */
export type PanelStatePayload = PanelState | null;
