/**
 * Typed wrappers around every command and event in the Rust core.
 *
 * Mirror of `src-tauri/src/commands.rs` and `src-tauri/src/events.rs`. Nothing
 * outside this file calls `invoke` or `listen` directly, so:
 *  - argument names match the Rust parameter names exactly
 *  - every rejection becomes a `UiError` with a message that is safe to show
 *  - event names exist in one place
 */

import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

import type {
  AccuracyLevel,
  ActionItem,
  ActionItemPatch,
  ActionItemsUpdatedPayload,
  AudioDevice,
  AudioLevelsPayload,
  CaptureStatePayload,
  CaptureStatus,
  DeleteMode,
  DetectionPayload,
  DetectionStatus,
  DownloadProgressPayload,
  ExportRequest,
  ExportResult,
  Id,
  Job,
  JobProgressPayload,
  JobQuery,
  Marker,
  MarkerKind,
  Meeting,
  MeetingDetail,
  MeetingQuery,
  MeetingSummary,
  MeetingUpdatedPayload,
  ModelInfo,
  NavigatePayload,
  NoticePayload,
  OnboardingState,
  PanelState,
  PanelStatePayload,
  PermissionState,
  PermissionStatus,
  PermissionTarget,
  Provider,
  ProviderConfig,
  ProviderInfo,
  ProviderTestResult,
  RecoveryAction,
  RecoveryAvailablePayload,
  SearchHit,
  SearchQuery,
  Segment,
  Settings,
  SettingsPatch,
  Speaker,
  SpeakersUpdatedPayload,
  SpeechReadiness,
  StartRecordingOptions,
  StorageReport,
  Summary,
  SummaryReadyPayload,
  SummaryReq,
  SystemCapabilities,
  Template,
  TemplateDraft,
  TranscriptFinalPayload,
  TranscriptPartialPayload,
  TranscriptQuery,
  TranscriptRevisedPayload,
  TrayActionPayload,
  TrayStatePayload,
  UiError,
} from "./types";

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

const FALLBACK_ERROR: UiError = {
  kind: "unexpected",
  message: "Something went wrong. Nothing was lost.",
};

/** Is this value the error shape the Rust side promises? */
export function isUiError(value: unknown): value is UiError {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as UiError).message === "string" &&
    typeof (value as UiError).kind === "string"
  );
}

/**
 * Normalise anything a rejected command throws into a `UiError`.
 *
 * A panic in Rust arrives as a bare string, so callers still get something they
 * can render rather than `[object Object]`.
 */
export function toUiError(err: unknown): UiError {
  if (isUiError(err)) return err;
  if (typeof err === "string" && err.trim() !== "") {
    return { ...FALLBACK_ERROR, detail: err };
  }
  if (err instanceof Error) {
    return { ...FALLBACK_ERROR, detail: err.message };
  }
  return FALLBACK_ERROR;
}

async function call<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  try {
    return await invoke<T>(command, args);
  } catch (err) {
    throw toUiError(err);
  }
}

// ---------------------------------------------------------------------------
// Capture
// ---------------------------------------------------------------------------

/** Safe to call twice: an in-progress recording is returned as-is. */
export const startRecording = (options?: StartRecordingOptions) =>
  call<Id>("start_recording", { options });

/** Resolves to null when nothing was being recorded. */
export const stopRecording = () => call<Id | null>("stop_recording");

export const pauseRecording = () => call<CaptureStatus>("pause_recording");

export const resumeRecording = () => call<CaptureStatus>("resume_recording");

export const getCaptureState = () => call<CaptureStatus>("get_capture_state");

/** The live "Flag action item" button. */
export const addMarker = (kind?: MarkerKind, note?: string) =>
  call<Marker>("add_marker", { kind, note });

export const listInterruptedMeetings = () =>
  call<Meeting[]>("list_interrupted_meetings");

export const resolveInterruptedMeeting = (meetingId: Id, action: RecoveryAction) =>
  call<void>("resolve_interrupted_meeting", { meetingId, action });

// ---------------------------------------------------------------------------
// Meetings
// ---------------------------------------------------------------------------

export const listMeetings = (query?: MeetingQuery) =>
  call<MeetingSummary[]>("list_meetings", { query });

export const getMeeting = (meetingId: Id) =>
  call<MeetingDetail>("get_meeting", { meetingId });

export const updateMeetingTitle = (meetingId: Id, title: string) =>
  call<void>("update_meeting_title", { meetingId, title });

export const deleteMeeting = (meetingId: Id, mode: DeleteMode) =>
  call<void>("delete_meeting", { meetingId, mode });

/** Settings > Data. Removes recordings from disk as well as the rows. */
export const deleteAllData = () => call<void>("delete_all_data");

export const getTranscript = (query: TranscriptQuery) =>
  call<Segment[]>("get_transcript", { query });

/**
 * "Listen again": write this meeting's transcript again from its recording.
 *
 * For a meeting recorded while transcription wasn't working — the sound is on
 * disk, so the words can always be read back out of it. The old transcript and
 * the speakers go; the recording, the recap and its tasks stay.
 *
 * Nothing comes back: progress arrives on `jobProgress`, exactly like the work
 * that follows a normal recording, and the transcript refreshes on
 * `transcriptRevised`. Safe to call twice — a second call while one is running
 * changes nothing. Rejects while a recording is live.
 */
export const retranscribeMeeting = (meetingId: Id) =>
  call<void>("retranscribe_meeting", { meetingId });

export const searchTranscripts = (query: SearchQuery) =>
  call<SearchHit[]>("search_transcripts", { query });

export const getMarkers = (meetingId: Id) =>
  call<Marker[]>("get_markers", { meetingId });

/** Null when the audio has been deleted or not mixed down yet. */
export const getPlaybackPath = (meetingId: Id) =>
  call<string | null>("get_playback_path", { meetingId });

export const getStorageReport = () => call<StorageReport>("get_storage_report");

// ---------------------------------------------------------------------------
// Speakers
// ---------------------------------------------------------------------------

export const listSpeakers = (meetingId: Id) =>
  call<Speaker[]>("list_speakers", { meetingId });

export const renameSpeaker = (speakerId: Id, displayName: string) =>
  call<void>("rename_speaker", { speakerId, displayName });

/** Non-destructive: `fromId` keeps its row and points at `intoId`. */
export const mergeSpeakers = (fromId: Id, intoId: Id) =>
  call<void>("merge_speakers", { fromId, intoId });

export const unmergeSpeaker = (speakerId: Id) =>
  call<void>("unmerge_speaker", { speakerId });

/** Queues the offline pass. Progress arrives on the job-progress event. */
export const refineSpeakers = (meetingId: Id) =>
  call<Id>("refine_speakers", { meetingId });

/**
 * "There were four of us": correct how many people were in this meeting.
 *
 * `count` is the total, **including** whoever was at this computer — that is
 * what the question means to the person answering it. `null` clears the
 * correction and puts the number back to Echo's own count.
 *
 * Setting it works out who said what again, looking for exactly that many
 * voices, which separates them better than any automatic guess. Nothing else
 * changes: the words, the recording and the recap stay as they are. Progress
 * arrives on `jobProgress`, and the new answer — speaker chips and
 * `peopleCount` together — on `speakersUpdated`.
 *
 * Safe to call twice: asking again for the same number changes nothing. Rejects
 * while this meeting is being recorded, while Echo is still working on it, and
 * for a number outside 1–12.
 */
export const setSpeakerCount = (meetingId: Id, count: number | null) =>
  call<void>("set_speaker_count", { meetingId, count });

// ---------------------------------------------------------------------------
// Speech assets
// ---------------------------------------------------------------------------

export const listAccuracyLevels = () =>
  call<AccuracyLevel[]>("list_accuracy_levels");

export const selectAccuracyLevel = (levelId: string) =>
  call<Settings>("select_accuracy_level", { levelId });

export const downloadSpeechAssets = (levelId: string) =>
  call<Id>("download_speech_assets", { levelId });

export const cancelSpeechDownload = (assetId: Id) =>
  call<void>("cancel_speech_download", { assetId });

export const getSpeechReadiness = () =>
  call<SpeechReadiness>("get_speech_readiness");

/** Explicit user action. Nothing loads on its own. */
export const prewarmSpeech = () => call<void>("prewarm_speech");

export const releaseSpeech = () => call<void>("release_speech");

/** Settings > Advanced only. */
export const listSpeechAssets = () => call<ModelInfo[]>("list_speech_assets");

export const removeSpeechAsset = (assetId: Id) =>
  call<void>("remove_speech_asset", { assetId });

// ---------------------------------------------------------------------------
// Recaps
// ---------------------------------------------------------------------------

export const listSummaryProviders = () =>
  call<ProviderInfo[]>("list_summary_providers");

export const testSummaryProvider = (provider: Provider) =>
  call<ProviderTestResult>("test_summary_provider", { provider });

export const saveSummaryProvider = (config: ProviderConfig) =>
  call<void>("save_summary_provider", { config });

/** Goes straight to the keychain. It never comes back out. */
export const setProviderKey = (provider: Provider, key: string) =>
  call<void>("set_provider_key", { provider, key });

export const hasProviderKey = (provider: Provider) =>
  call<boolean>("has_provider_key", { provider });

export const deleteProviderKey = (provider: Provider) =>
  call<void>("delete_provider_key", { provider });

/** Queues the recap. Watch job-progress, then summary-ready. */
export const generateSummary = (request: SummaryReq) =>
  call<Id>("generate_summary", { request });

export const listSummaries = (meetingId: Id) =>
  call<Summary[]>("list_summaries", { meetingId });

export const getSummary = (summaryId: Id) =>
  call<Summary>("get_summary", { summaryId });

export const listActionItems = (meetingId: Id) =>
  call<ActionItem[]>("list_action_items", { meetingId });

export const updateActionItem = (patch: ActionItemPatch) =>
  call<ActionItem>("update_action_item", { patch });

// ---------------------------------------------------------------------------
// Recap styles
// ---------------------------------------------------------------------------

export const listTemplates = () => call<Template[]>("list_templates");

export const saveTemplate = (draft: TemplateDraft) =>
  call<Template>("save_template", { draft });

export const deleteTemplate = (templateId: Id) =>
  call<void>("delete_template", { templateId });

// ---------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------

export const listJobs = (query?: JobQuery) => call<Job[]>("list_jobs", { query });

export const cancelJob = (jobId: Id) => call<void>("cancel_job", { jobId });

export const retryJob = (jobId: Id) => call<void>("retry_job", { jobId });

// ---------------------------------------------------------------------------
// Meeting detection
// ---------------------------------------------------------------------------

export const getDetectionStatus = () =>
  call<DetectionStatus>("get_detection_status");

export const setDetectionEnabled = (enabled: boolean) =>
  call<Settings>("set_detection_enabled", { enabled });

export const snoozeDetection = (minutes?: number) =>
  call<void>("snooze_detection", { minutes });

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

/** `request.destination` must be an absolute path from the save dialog. */
export const exportMeeting = (request: ExportRequest) =>
  call<ExportResult>("export_meeting", { request });

export const exportDiagnostics = (destination: string) =>
  call<ExportResult>("export_diagnostics", { destination });

/** `destination` must be an absolute path from the save dialog. Builds the
 * playback mix first, on demand, if it doesn't exist yet — that wait shows
 * up as this promise taking longer, not as a separate loading state to wire
 * up; the Info tab's own job progress already covers it. */
export const downloadRecording = (meetingId: Id, destination: string) =>
  call<void>("download_recording", { meetingId, destination });

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

export const getSettings = () => call<Settings>("get_settings");

export const updateSettings = (patch: SettingsPatch) =>
  call<Settings>("update_settings", { patch });

/** Resolves to the free space in bytes, or rejects if the folder is unusable. */
export const validateStorageLocation = (path: string) =>
  call<number>("validate_storage_location", { path });

export const listInputDevices = () => call<AudioDevice[]>("list_input_devices");

// ---------------------------------------------------------------------------
// Permissions and onboarding
// ---------------------------------------------------------------------------

export const getPermissionStatus = () =>
  call<PermissionStatus>("get_permission_status");

export const requestPermission = (target: PermissionTarget) =>
  call<PermissionState>("request_permission", { target });

export const openPrivacySettings = (target: PermissionTarget) =>
  call<void>("open_privacy_settings", { target });

export const getOnboardingState = () =>
  call<OnboardingState>("get_onboarding_state");

export const completeOnboarding = () => call<Settings>("complete_onboarding");

/** Settings > Advanced only. */
export const getSystemCapabilities = () =>
  call<SystemCapabilities>("get_system_capabilities");

// ---------------------------------------------------------------------------
// Window
// ---------------------------------------------------------------------------

export const showMainWindow = () => call<void>("show_main_window");

export const quitApp = () => call<void>("quit_app");

// ---------------------------------------------------------------------------
// The floating panel
// ---------------------------------------------------------------------------

/**
 * The ✕ (and Escape) on the "meeting detected" panel: take it away and say
 * nothing more about *this* meeting. Not a snooze — the next meeting gets its
 * panel as usual.
 */
export const panelDismiss = () => call<void>("panel_dismiss");

/**
 * The panel is done because it worked — Start or Stop went through. Same hiding,
 * without telling detection the meeting was unwanted.
 */
export const panelClose = () => call<void>("panel_close");

/**
 * What the panel should be showing right now.
 *
 * The panel is normally told by event, but the very first time the window is
 * still booting and misses it; asking on mount closes that gap.
 */
export const getPanelState = () => call<PanelState | null>("get_panel_state");

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/** Mirror of the constants in `src-tauri/src/events.rs`. */
export const EVENTS = {
  captureState: "echo://capture-state",
  transcriptPartial: "echo://transcript-partial",
  transcriptFinal: "echo://transcript-final",
  transcriptRevised: "echo://transcript-revised",
  audioLevels: "echo://audio-levels",
  jobProgress: "echo://job-progress",
  downloadProgress: "echo://download-progress",
  detection: "echo://detection",
  speakersUpdated: "echo://speakers-updated",
  summaryReady: "echo://summary-ready",
  actionItemsUpdated: "echo://action-items-updated",
  meetingUpdated: "echo://meeting-updated",
  settingsChanged: "echo://settings-changed",
  notice: "echo://notice",
  trayState: "echo://tray-state",
  trayAction: "echo://tray-action",
  navigate: "echo://navigate",
  recoveryAvailable: "echo://recovery-available",
  /** Only ever delivered to the floating panel window. */
  panelState: "echo://panel-state",
} as const;

export type EventName = (typeof EVENTS)[keyof typeof EVENTS];

/** Payload type for each event, so the listeners below stay honest. */
export interface EventPayloads {
  [EVENTS.captureState]: CaptureStatePayload;
  [EVENTS.transcriptPartial]: TranscriptPartialPayload;
  [EVENTS.transcriptFinal]: TranscriptFinalPayload;
  [EVENTS.transcriptRevised]: TranscriptRevisedPayload;
  [EVENTS.audioLevels]: AudioLevelsPayload;
  [EVENTS.jobProgress]: JobProgressPayload;
  [EVENTS.downloadProgress]: DownloadProgressPayload;
  [EVENTS.detection]: DetectionPayload;
  [EVENTS.speakersUpdated]: SpeakersUpdatedPayload;
  [EVENTS.summaryReady]: SummaryReadyPayload;
  [EVENTS.actionItemsUpdated]: ActionItemsUpdatedPayload;
  [EVENTS.meetingUpdated]: MeetingUpdatedPayload;
  [EVENTS.settingsChanged]: Settings;
  [EVENTS.notice]: NoticePayload;
  [EVENTS.trayState]: TrayStatePayload;
  [EVENTS.trayAction]: TrayActionPayload;
  [EVENTS.navigate]: NavigatePayload;
  [EVENTS.recoveryAvailable]: RecoveryAvailablePayload;
  [EVENTS.panelState]: PanelStatePayload;
}

/**
 * Subscribe to one event.
 *
 * Returns a promise for the unsubscribe function. In an effect, guard against
 * unmounting before the promise settles:
 *
 * ```ts
 * useEffect(() => {
 *   let stop: UnlistenFn | undefined;
 *   let cancelled = false;
 *   on(EVENTS.captureState, setStatus).then((fn) => {
 *     if (cancelled) fn();
 *     else stop = fn;
 *   });
 *   return () => {
 *     cancelled = true;
 *     stop?.();
 *   };
 * }, []);
 * ```
 */
export function on<K extends EventName>(
  event: K,
  handler: (payload: EventPayloads[K]) => void,
): Promise<UnlistenFn> {
  return listen<EventPayloads[K]>(event, (e) => handler(e.payload));
}

/** Subscribe to several events with one teardown. */
export async function onMany(
  handlers: { [K in EventName]?: (payload: EventPayloads[K]) => void },
): Promise<UnlistenFn> {
  const entries = Object.entries(handlers) as [
    EventName,
    (payload: unknown) => void,
  ][];
  const stops = await Promise.all(
    entries.map(([event, handler]) =>
      listen(event, (e) => handler(e.payload as unknown)),
    ),
  );
  return () => stops.forEach((stop) => stop());
}

export type { UnlistenFn };
