/**
 * Every word a person reads lives here.
 *
 * WHY ONE FILE
 * A string in a component is a string nobody reviews. Keeping them together
 * means the tone can be checked in one pass, and it makes the rule below
 * enforceable rather than aspirational.
 *
 * THE RULE (mantra 2 in docs/DESIGN.md)
 * None of these words may appear in anything a person reads:
 *
 *   Whisper, model, ONNX, VAD, diarization, diarize, embedding, connector,
 *   token, inference, GGML, CoreML, Metal, Vulkan, PipeWire, ScreenCaptureKit,
 *   FLAC, SQLite, FTS, keyring, keychain (as a mechanism), API, endpoint
 *
 * Say what happens to the person instead:
 *
 *   "model download"        -> "Downloading what Echo needs to understand speech"
 *   "transcribing"          -> "Listening…"
 *   "summarizing"           -> "Writing your recap…"
 *   "diarization pass"      -> "Working out who said what"
 *   "VAD detected speech"   -> (say nothing; it is not the person's business)
 *   "connector unreachable" -> "Echo couldn't reach the place that writes your
 *                              recaps. Check it's running."
 *
 * Errors say what the person can do, not what failed inside. Sizes and times are
 * fine; counters and queue depths are not.
 *
 * WHO IS READING (mantra 4)
 * A freelancer or a manager, not an engineer. They have never opened a terminal
 * and they will not read a manual. Assume no prior knowledge, never name a
 * prerequisite they would have to go and install, and make every error end in
 * something they can actually do.
 *
 * THE ONE EXCEPTION
 * Settings > Advanced may name the speech engine, the graphics backend, asset
 * identifiers and file paths. Those strings live under `advanced` below and
 * nowhere else.
 *
 * HOW TO FILL THIS IN
 * Each section starts seeded with what the screen needs to get off the ground.
 * A screen agent adds more keys as its screen grows, as plain sentences with
 * normal punctuation. No exclamation marks, no "Oops", no emoji, no shouting
 * in caps.
 */

/** Words for machine values the UI has to render. */
export const labels = {
  captureState: {
    idle: "Waiting for your next meeting",
    starting: "Getting ready…",
    recording: "Listening…",
    paused: "Paused",
    stopping: "Wrapping up…",
    stopped: "Finished",
    failed: "Something went wrong",
    degraded: "Listening, but not everything",
    recovering: "Picking up where it left off",
  },
  degradedReason: {
    systemAudioUnavailable:
      "Echo can hear you, but not the other people. It will keep recording.",
    microphoneUnavailable:
      "Echo can hear the meeting, but not your own microphone.",
    transcriptBehind: "Catching up on the last few minutes.",
    storageLow: "Running low on space. Recording keeps going — free up room soon.",
  },
  provider: {
    onThisComputer: "Ollama",
    gemini: "Google Gemini",
  },
  jobKind: {
    transcribeCatchup: "Catching up on the transcript",
    diarize: "Working out who said what",
    summarize: "Writing the recap",
    export: "Preparing the file",
    download: "Downloading",
    mixdown: "Preparing playback",
  },
  jobStatus: {
    queued: "Waiting",
    running: "In progress",
    paused: "Paused",
    done: "Done",
    failed: "Didn't finish",
    cancelled: "Cancelled",
  },
  permissionState: {
    unknown: "Not checked yet",
    granted: "Allowed",
    denied: "Not allowed",
    prompting: "Waiting for your answer",
    restartRequired: "Restart Echo to finish",
    notApplicable: "Nothing to set up",
  },
} as const;

/** Words used across more than one screen: dialogs, toasts, generic buttons. */
export const common = {
  close: "Close",
  settingUpLabel: "Getting Echo ready",
  dismiss: "Dismiss",
  cancel: "Cancel",
  save: "Save",
  done: "Done",
  back: "Back",
  next: "Continue",
  skip: "Skip for now",
  retry: "Try again",
  loading: "Loading…",
  search: "Search",
  copy: "Copy",
  copied: "Copied",
  export: "Export",
  delete: "Delete",
  rename: "Rename",
  learnMore: "Learn more",
  openSettings: "Open Settings",
  backToHome: "Back to Home",
  /** On a notice that names a meeting: takes you straight to it. */
  open: "Open",
  /** The 404 route, which should be unreachable in a desktop app. */
  notFoundTitle: "There is nothing here",
  notFoundBody: "Echo took a wrong turn. Your meetings are safe.",
  /** Thrown before React mounts, so it can never reach a rendered screen. */
  noWindowContents: "Echo could not find its window contents.",
} as Record<string, string>;

/** The left sidebar: wordmark, destinations, the live indicator. */
export const nav = {
  wordmark: "Echo",
  newMeeting: "New meeting",
  home: "Home",
  meetings: "Meetings",
  settings: "Settings",
  privacyNote: "Everything stays on this computer.",
  liveLabel: "Recording",
  livePausedLabel: "Paused",
} as Record<string, string>;

/** Home: the status hero, the recent list, empty states. */
export const home = {
  heroIdleTitle: "Waiting for your next meeting",
  heroIdleSubtitle: "Start any time — Echo will listen and write the recap.",
  heroIdleTip: "Echo keeps an eye out for meeting apps and lets you know when it's time.",
  heroPreparingTitle: "Getting Echo ready",
  heroPreparingSubtitle:
    "Finishing the one-time download. Recording and meeting alerts switch on the moment it's done.",
  heroDetectedTitle: "Looks like your meeting is starting",
  heroDetectedSubtitle: "Echo noticed a call getting going — press Start and it'll listen.",
  openLiveButton: "View live",
  startButton: "Start",
  stopButton: "Stop",
  searchPlaceholder: "Search your meetings",
  recentTitle: "Recent meetings",
  viewAll: "View all",
  emptyTitle: "Echo has nothing to echo yet",
  emptyDescription: "Go have a meeting worth remembering, and Echo will write it all down.",
  recoveringTitle: "Echo found a meeting that didn't finish",
  recoveringDescription:
    "This can happen after a crash or a restart. Pick up where it left off, or let it go.",
  recoveringFinish: "Finish it",
  recoveringDiscard: "Let it go",
  /** The quiet trash icon on a meeting row, and its one confirmation. */
  deleteRowLabel: "Delete meeting",
  deleteRowConfirmTitle: "Delete this meeting?",
  /** Word for word the same as `meeting.deleteConfirmDescription`: it is the
   * same action, so it says the same thing wherever it's asked. */
  deleteRowConfirmDescription:
    "The recording, transcript and recap all go with it. This can't be undone.",
} as Record<string, string>;

/** Live: the recording view. */
export const live = {
  title: "Listening",
  waitingForSpeech: "Waiting for someone to start talking…",
  flagButton: "Flag action item",
  flagged: "Flagged",
  pauseButton: "Pause",
  resumeButton: "Resume",
  stopButton: "Stop",
  stopConfirmTitle: "Stop recording?",
  stopConfirmDescription:
    "Echo will finish listening and start writing the recap.",
  micOnlyBanner:
    "Echo can hear you, but not the other people. It will keep recording.",
  systemOnlyBanner: "Echo can hear the meeting, but not your own microphone.",
  unknownSpeaker: "Speaker",
  nothingRecordingTitle: "Nothing is being recorded",
  nothingRecordingDescription: "Start a meeting from Home to see it here.",
  goHomeButton: "Go to Home",
  copyTranscriptButton: "Copy transcript",
} as Record<string, string>;

/** Meeting detail: Recap, Transcript, Info tabs. */
export const meeting = {
  tabRecap: "Recap",
  tabTranscript: "Transcript",
  tabInfo: "Info",
  noRecapTitle: "No recap yet",
  noRecapDescription: "Write one whenever you're ready.",
  writeRecapButton: "Write recap",
  regenerateButton: "Try a different way",
  templateLabel: "Recap style",
  actionItemsTitle: "Action items",
  actionItemsEmpty: "Nothing to follow up on.",
  ownerPlaceholder: "Who's doing this?",
  transcriptSearchPlaceholder: "Search this transcript",
  transcriptEmpty: "Nothing was captured.",
  infoCapturedTitle: "What was captured",
  infoStorageUsed: "Storage used",
  infoDeleteAudio: "Delete audio, keep the transcript",
  infoDeleteAudioDescription:
    "Removes the recording, keeps the words and the recap.",
  infoDeleteAll: "Delete this meeting",
  infoDeleteAllDescription: "Removes everything about this meeting for good.",
  deleteConfirmTitle: "Delete this meeting?",
  deleteConfirmDescription:
    "The recording, transcript and recap all go with it. This can't be undone.",
  renameTitle: "Rename meeting",
  playFromHere: "Play from here",
  languageDetecting: "Detecting…",
  untitledMeeting: "Untitled meeting",
  exportMarkdown: "Markdown",
  exportWord: "Word document",
  exportPdf: "PDF",
  modelPickerLabel: "Written by",
  modelPickerDefault: "Default",
  modelPickerLoading: "Checking…",
  mergeSpeakersButton: "Combine two speakers",
  mergeSpeakersTitle: "Which two are the same person?",
  mergeSpeakersDescription:
    "Pick two speakers below. Everything the second one said becomes part of the first.",
  mergeSpeakersKeepLabel: "Keep this name",
  mergeSpeakersConfirm: "Combine",
  mergeSpeakersNeedTwo: "Pick exactly two speakers to combine.",
  renameSpeakerTitle: "Rename speaker",
  renameSpeakerPlaceholder: "Speaker name",
  deleteAudioConfirmTitle: "Delete the recording?",
  deleteAudioConfirmDescription:
    "Keeps the transcript and recap. The recording itself can't be brought back.",
  infoLanguageLabel: "Language",
  infoDurationLabel: "Length",
  infoLinesLabel: "Lines of transcript",
  infoWorkingTitle: "Still working",
  actionItemMarkDone: "Mark done",
  actionItemMarkNotDone: "Mark not done",
  transcriptNoMatches: "Nothing matches that.",
  transcriptFilterPlaceholder: "Filter this transcript",
  recapWritingTitle: "Writing your recap…",
  recapWritingDescription: "This usually takes less than a minute.",
  infoCapturedLabel: "Captured",
  copyTranscriptButton: "Copy transcript",
  infoSaveRecording: "Save the recording",
  infoSaveRecordingDescription: "Saves the audio as a file you can keep or share.",
  recordingFileType: "Recording",
} as Record<string, string>;

/** History and search. */
export const search = {
  title: "History",
  placeholder: "Search everything Echo has heard",
  emptyTitle: "Nothing found",
  emptyDescription: "Try a different word, or check the spelling.",
  noMeetingsTitle: "No meetings yet",
  noMeetingsDescription: "Once you record one, you'll be able to search it here.",
  resultSingular: "result",
  resultPlural: "results",
  idleTitle: "Search your meetings",
  idleDescription: "Type a few words from anything that was said.",
} as Record<string, string>;

/** Settings, except Advanced. */
export const settings = {
  title: "Settings",
  subtitle: "How Echo starts, where it keeps your recordings, and who writes your recaps.",
  sectionGeneral: "General",
  sectionSpeech: "Speech",
  sectionSummaries: "Summaries",
  sectionTemplates: "Recap styles",
  sectionData: "Data",
  sectionAdvanced: "Advanced",
  launchAtLogin: "Open Echo when you log in",
  detectionEnabled: "Notice when a meeting starts",
  detectionDescription:
    "Echo watches for meeting apps and a microphone in use, and lets you know.",
  storageLocation: "Where recordings are kept",
  storageLocationButton: "Choose folder",
  summaryLanguageLabel: "Recap language",
  summaryLanguageMeeting: "Same as the meeting",
  summaryLanguageEnglish: "English",
  summaryLanguageFixed: "Always this language",
  accuracyLevelLabel: "How carefully Echo listens",
  accuracyLevelDescription:
    "A more careful setting takes a little longer but hears more correctly.",
  summaryProviderOnDevice: "On this computer",
  summaryProviderOnDeviceDescription:
    "Runs an AI model on this Mac through the Ollama app. Nothing leaves your computer.",
  summaryProviderGemini: "Google Gemini",
  summaryProviderGeminiDescription:
    "Your transcript is sent to Google to write the recap.",
  providerKeyLabel: "Key",
  providerKeyPlaceholder: "Paste your key",
  providerKeySaved: "Saved — kept in your computer's secure storage",
  providerTestButton: "Test connection",
  templatesEmpty: "No custom styles yet.",
  addTemplateButton: "Add a style",
  dataStorageReportTitle: "Storage used",
  dataDeleteAllButton: "Delete everything",
  dataDeleteAllConfirmTitle: "Delete every meeting?",
  dataDeleteAllConfirmDescription:
    "Every recording, transcript and recap will be gone for good.",
  advancedIntro:
    "Technical detail, for anyone who wants it. Nothing here changes what Echo does day to day.",

  // General
  storageLocationHint:
    "New recordings are saved here. Pick a folder of Echo's own — the meetings you already have stay where they are.",
  storageFreeSpaceSuffix: "free on this drive",
  summaryLanguageCustomLabel: "Language",
  summaryLanguageCustomPlaceholder: "e.g. French",

  // Speech
  speechCurrentBadge: "Current",
  speechRecommendedBadge: "Recommended",
  speechDownloadButton: "Download",
  speechRemoveButton: "Remove",
  speechCancelButton: "Cancel",
  speechUseButton: "Use this",
  speechInstalledNote: "Ready to use",
  speechDownloadError:
    "That download didn't finish. Check your connection and try again.",

  // Recaps (Summaries)
  /** The one switch above the chooser: recaps happen on their own by default,
   * so this is how someone turns that off. */
  recapsAutomaticTitle: "Write a recap after every meeting",
  recapsAutomaticDescription:
    "Echo writes it on its own as soon as a meeting ends. Turn this off and you can still ask for one from any meeting.",
  recapsOnDeviceStatusDetected: "Ollama is running and ready",
  recapsOnDeviceStatusNotRunning:
    "Ollama isn't running. Get it from ollama.com, open it, then check again.",
  recapsOnDeviceModelLabel: "Model",
  recapsOnDeviceModelPlaceholder: "Pick one",
  recapsOnDeviceAdvancedToggle: "Use a different address",
  recapsOnDeviceAddressLabel: "Address",
  recapsOnDeviceAddressPlaceholder: "http://localhost:11434",
  recapsOnDeviceLeavesMachineWarning:
    "That address isn't on this computer, so your transcript would leave your machine to reach it.",
  recapsGeminiPrivacyParagraph:
    "Turning this on sends your transcript to Google's servers so Gemini can write the recap. Whether Google stores or otherwise uses that data depends on your Google account and billing plan.",
  recapsGeminiPrivacyLink: "Read Google's data terms",
  recapsGeminiModelLabel: "Model",
  recapsGeminiModelPlaceholder: "Pick one",
  recapsSelectedBadge: "Being used for recaps",
  recapsSelectButton: "Use this",
  recapsCheckAgainButton: "Check again",

  // Templates
  templatesBuiltinBadge: "Built-in",
  templatesEditButton: "Edit",
  templatesNameLabel: "Name",
  templatesPromptLabel: "What it asks for",
  templatesPromptHint:
    "Plain instructions for how the recap should be written.",
  templatesNewTitle: "New recap style",
  templatesEditTitle: "Edit recap style",

  // Data
  dataStorageBreakdownAudio: "Recordings",
  dataStorageBreakdownDatabase: "Meetings and transcripts",
  dataStorageBreakdownSpeech: "What Echo uses to understand speech",
  dataStorageBreakdownLogs: "Diagnostics",
  dataFreeSpace: "Free space",
  dataMeetingCount: "Meetings",
  dataDeleteAllTypePrompt: 'Type "delete everything" below to confirm.',
  dataDeleteAllTypePlaceholder: "delete everything",
  dataDeleteAllTypeWord: "delete everything",
} as Record<string, string>;

/**
 * Settings > Advanced. The only place technical words are allowed, because the
 * person went looking for them.
 */
export const advanced = {
  engineLabel: "Speech engine",
  graphicsBackendLabel: "Graphics backend",
  graphicsFallbackReason: "Fell back to the processor because",
  cpuThreadsLabel: "Processor threads in use",
  memoryLabel: "Memory in use",
  modelIdentifierLabel: "Installed asset",
  appVersionLabel: "Echo version",
  diagnosticsExportButton: "Export diagnostics",
  diagnosticsDescription:
    "A log of what Echo did and when — device changes, timings, fallbacks. Never your words.",
  diagnosticsExportSuccess: "Saved.",
  showButton: "Show technical details",
  hideButton: "Hide technical details",
  installedAssetsTitle: "What's installed",
  osLabel: "Operating system",
  systemAudioSupportedLabel: "Can hear other people",
  traySupportedLabel: "Menu bar icon",
  yes: "Yes",
  no: "No",
} as Record<string, string>;

/** Onboarding: Welcome, Permissions, Download, Summaries. */
export const onboarding = {
  welcomeTitle: "Welcome to Echo",
  welcomeSubtitle:
    "A few things to set up once: permission to listen, a one-time download, and where your recaps come from.",
  getStartedButton: "Get started",
  welcomeStepListenTitle: "It listens",
  welcomeStepListenCaption: "During your meetings, on this computer only.",
  welcomeStepTranscriptTitle: "Writes it down",
  welcomeStepTranscriptCaption: "Who said what, in any language.",
  welcomeStepRecapTitle: "Hands you the recap",
  welcomeStepRecapCaption: "Key points and to-dos, ready to share.",
  stepPermissionsTitle: "Let Echo listen",
  microphoneTitle: "Your microphone",
  microphoneDescription: "So Echo can hear you.",
  screenRecordingTitle: "What's on your screen",
  screenRecordingDescription:
    "macOS uses this permission for hearing everything your computer plays, not just your microphone.",
  notificationsTitle: "Notifications",
  notificationsDescription:
    "So Echo can let you know when a meeting starts. macOS asks on its own the first time.",
  permissionAllowButton: "Allow",
  permissionRestartNote: "Restart Echo for this to take effect.",
  permissionDeniedNote:
    "Turned off in System Settings. Turn it on there, then come back.",
  stepDownloadTitle: "Downloading what Echo needs to understand speech",
  stepDownloadSize: "1.6 GB, one time",
  stepDownloadDescription:
    "This happens once. After that, Echo works without sending anything anywhere.",
  chooseSmallerButton: "Use a smaller, faster download instead",
  downloadRetryButton: "Try downloading again",
  stepSummariesTitle: "Who writes your recaps?",
  summariesIntro:
    "Everything Echo records and writes down stays on this Mac either way. This choice is only about which assistant reads the transcript and writes the recap \u2014 you can change it anytime.",
  summariesOnDeviceFound: "Ollama is running \u2014 recaps stay on this Mac, private and free.",
  summariesOnDeviceNotFound:
    "Ollama isn't running. Get it from ollama.com, open it, then check again.",
  summariesGeminiTitle: "Google Gemini",
  summariesGeminiDescription:
    "Google's AI writes the recap. The transcript text is sent to Google for that \u2014 needs a free API key.",
  summariesSkipDescription:
    "You can record and read transcripts without this — set it up whenever you like.",
  finishButton: "Start using Echo",

  stepLabelWelcome: "Welcome",
  stepLabelPermissions: "Permissions",
  stepLabelDownload: "Download",
  stepLabelSummaries: "Recaps",
  quitButton: "Quit Echo",
  downloadContinueAnywayNote: "You can finish this later in Settings.",
  summariesUseOnDevice: "Use Ollama",
  recapsChosenOnDevice: "Recaps will be written by Ollama on this Mac.",
  recapsChosenGemini: "Recaps will be written using Google Gemini.",
} as Record<string, string>;

/** Banners, toasts and confirmations. */
export const notices = {
  somethingWentWrong: "Something went wrong. Nothing was lost.",
  setupDone: "Echo is ready — recording and meeting alerts are on.",
  systemAudioLost:
    "Echo can hear you, but not the other people. It will keep recording.",
  storageLow:
    "Running low on space. Recording keeps going — free up room when you can.",
  storageFull:
    "Out of space. Recording has stopped, but nothing has been lost.",
  meetingDeleted: "Meeting deleted.",
  audioDeleted: "Recording deleted. The transcript is still here.",
  exportedTo: "Saved.",
  copiedToClipboard: "Copied.",
  savedToKeychain: "Saved to your computer's secure storage.",
  keychainLocked:
    "Your computer's secure storage is locked. Unlock it and try again.",
  providerUnreachable:
    "Echo couldn't reach the place that writes your recaps. Check it's running.",
  actionItemUpdated: "Updated.",
} as Record<string, string>;

/**
 * The few strings docs/DESIGN.md quotes directly. They set the tone for
 * everything added later, so they are here from the start.
 */
export const anchors = {
  downloadingSpeech: "Downloading what Echo needs to understand speech",
  downloadingSpeechSize: "1.6 GB, one time",
  listening: "Listening…",
  writingRecap: "Writing your recap…",
  waitingForMeeting: "Waiting for your next meeting",
  systemAudioMeaning: "everything your computer plays",
  catchingUp: "Catching up on the last few minutes",
  workingOutSpeakers: "Working out who said what",
  micOnlyBanner:
    "Echo can still hear you, but not the other people. It will keep recording.",
} as const;

/**
 * The floating panel: a compact window that appears on its own to offer
 * Start when a meeting is detected. Its Start/Stop buttons and "Listening…"
 * line reuse `home`/`anchors` rather than repeating them here — this only
 * holds what's unique to the panel itself.
 */
export const panel = {
  detectedTitle: "Meeting detected",
  /** The line under the title: the app Echo noticed and how long ago, or just
   * how long ago when it cannot name the app. */
  detectedMeta: (ago: string, appName?: string) => (appName ? `${appName} · ${ago}` : ago),
  /** Paused, where the elapsed clock would be a lie: say what is true instead. */
  pausedHint: "Not listening right now",
} as const;

/** Words for the two microphone-versus-computer channels. */
export const channels = {
  mic: "You",
  system: "Everyone else",
  mixed: "Everything",
} as const;

/** A meeting's spoken language, as a plain-language name rather than the raw
 * code the core stores it as. Falls back to the code itself for anything not
 * in the list — still readable, never blank. */
const languageNames: Record<string, string> = {
  en: "English",
  es: "Spanish",
  fr: "French",
  de: "German",
  it: "Italian",
  pt: "Portuguese",
  nl: "Dutch",
  ja: "Japanese",
  zh: "Chinese",
  ko: "Korean",
  ru: "Russian",
  ar: "Arabic",
  hi: "Hindi",
  pl: "Polish",
  sv: "Swedish",
  tr: "Turkish",
};

export function languageLabel(code?: string): string | undefined {
  if (!code) return undefined;
  return languageNames[code.toLowerCase()] ?? code.toUpperCase();
}
