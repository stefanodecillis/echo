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
  PeopleUpdatedPayload,
  PermissionState,
  PermissionStatus,
  PermissionTarget,
  PersonInfo,
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
  SuggestedPerson,
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
  VocabularyWord,
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

/**
 * A few seconds of this person talking on their own, so a name can be put to
 * the voice.
 *
 * Resolves to **base64 WAV** — no prefix. Play it from a `blob:` URL, not a
 * `data:` one: the app's CSP allows `blob:` under `media-src` and deliberately
 * does not allow `data:` there, so a data URL is silently refused by the
 * webview. `Meeting/lib/audio.ts`'s `base64ToBlobUrl` does the decode:
 *
 * ```ts
 * const clip = await speakerSample(meetingId, speakerId);
 * const url = base64ToBlobUrl(clip); // revoke it when nothing can play it
 * new Audio(url).play();
 * ```
 *
 * Echo picks the longest stretch where only this person is talking, skips the
 * throat-clearing at the start of it, and caps the clip at a few seconds. The
 * same meeting always gives back the same clip, and nothing is written to disk.
 *
 * Rejects with a `notFound` UiError in two ordinary cases — this person never
 * talks on their own for long enough to recognise, and the recording has been
 * deleted — so show `error.message` as an explanation, not as a failure.
 */
export const speakerSample = (meetingId: Id, speakerId: Id) =>
  call<string>("speaker_sample", { meetingId, speakerId });

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
// Known people (voice enrollment)
//
// Enrolment is opt-in, one person at a time. Every call here is something the
// person clicked, and the ones that change anything emit `peopleUpdated`.
// ---------------------------------------------------------------------------

/** Everybody Echo remembers. Cheap: no audio, no models. */
export const listPeople = () => call<PersonInfo[]>("list_people");

/**
 * Forget a voice — the profile, the samples and their clips, all of it.
 *
 * Speaker links go with it. Names already shown in past meetings stay as plain
 * text, so a transcript somebody has read does not rewrite itself; say that
 * plainly if the confirmation dialog needs a sentence.
 */
export const deletePerson = (personId: Id) =>
  call<void>("delete_person", { personId });

/**
 * Rename a remembered person. Names already copied onto past meetings' speakers
 * are left as they are; this is the name future matches will use.
 */
export const renamePerson = (personId: Id, name: string) =>
  call<void>("rename_person", { personId, name });

/**
 * A few seconds of this remembered voice — **base64 WAV**, no prefix, same as
 * `speakerSample`. Play it from a `blob:` URL (`base64ToBlobUrl`), not a `data:`
 * one; the CSP allows only `blob:` under `media-src`.
 *
 * It comes from the clip kept with the profile, so it plays whether or not the
 * meeting it came from still exists. Rejects with `notFound` when Echo has kept
 * no audio for this person yet — an explanation, not a failure.
 */
export const personSampleAudio = (personId: Id) =>
  call<string>("person_sample_audio", { personId });

/**
 * "This is Marco" — or, with `personId: null`, "actually, nobody I have named".
 *
 * Linking is a confirmation, and confirmations are the only thing that ever
 * improves recognition: Echo keeps a few seconds of that voice from this meeting.
 * The person's name is copied onto the speaker row when the row still carries a
 * label Echo made up ("Speaker 2"); a name somebody typed is left alone.
 *
 * Renaming the speaker afterwards does **not** rename the person — the chip is
 * this meeting's display, the person is the identity. Both `speakersUpdated` and
 * `peopleUpdated` follow.
 */
export const linkSpeakerPerson = (
  meetingId: Id,
  speakerId: Id,
  personId: Id | null,
) => call<void>("link_speaker_person", { meetingId, speakerId, personId });

/**
 * "Remember this voice": make a new person out of one of this meeting's
 * speakers, name them, and learn their voice from this recording.
 *
 * Resolves to the person as Settings > People will show them. Rejects, in words
 * worth showing as an explanation, when this speaker never talks on their own
 * for long enough to be recognisable and when the recording has been deleted —
 * in both cases nothing is created.
 */
export const enrollSpeakerAsPerson = (
  meetingId: Id,
  speakerId: Id,
  name: string,
) => call<PersonInfo>("enroll_speaker_as_person", { meetingId, speakerId, name });

/**
 * Voices that have turned up unnamed in three or more meetings.
 *
 * Free to call: built from what the speaker pass already worked out, with no
 * audio read and no model loaded. Use `speakerSample(meetingId, speakerId)` to
 * let the person hear one before naming it.
 */
export const suggestedPeople = () => call<SuggestedPerson[]>("suggested_people");

/**
 * Name one of those recurring voices. Same effect as `enrollSpeakerAsPerson` on
 * the appearance the suggestion carried, which is what accepting means: nothing
 * about a suggestion is stored until somebody names it.
 */
export const acceptSuggestedPerson = (
  meetingId: Id,
  speakerId: Id,
  name: string,
) => call<PersonInfo>("accept_suggested_person", { meetingId, speakerId, name });

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
// Words Echo should know
//
// The list is what somebody typed plus the names of the voices Echo was asked
// to remember. Every one of these hands the whole list back, so the screen that
// shows it never has to ask twice.
// ---------------------------------------------------------------------------

export const listVocabulary = () => call<VocabularyWord[]>("list_vocabulary");

export const addVocabularyWord = (word: string) =>
  call<VocabularyWord[]>("add_vocabulary_word", { word });

export const removeVocabularyWord = (word: string) =>
  call<VocabularyWord[]>("remove_vocabulary_word", { word });

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
  peopleUpdated: "echo://people-updated",
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
  [EVENTS.peopleUpdated]: PeopleUpdatedPayload;
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
