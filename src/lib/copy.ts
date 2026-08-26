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
    /** The one true copy for "nothing this computer plays ever reached Echo, so
     * the whole meeting is in the microphone recording". It used to be
     * hand-copied into four places (`live.micOnlyBanner`,
     * `notices.systemAudioLost`, `anchors.micOnlyBanner`, and this one) and
     * drifted — the incident of 2026-08-24 shipped a banner that claimed Echo
     * couldn't hear the other people at all, while it was transcribing them
     * anyway, mislabelled "You". The others now reference these two instead of
     * repeating them. A Rust test (`SYSTEM_AUDIO_SILENT_MESSAGE` in
     * `src-tauri/src/audio/mod.rs`) pins the backend's copy of the same
     * sentence to this exact string.
     *
     * The last sentence is a promise, and this is the only state that can keep
     * it: the offline pass re-cuts the microphone recording into separate
     * voices for a meeting that has no system audio at all, and pins every
     * microphone line to "You" for one that has any. Hence the separate
     * `systemAudioLost` below rather than one sentence for both. */
    systemAudioUnavailable:
      "Echo is recording through the microphone only, so it can't tell who is speaking — every line says You for now. It will work out who said what once the meeting ends.",
    /** The computer's audio was arriving and stopped. Says what changed and
     * claims nothing else: the lines written while it was working carry real
     * names and stay that way, and the microphone tail is not separated by
     * anything, so neither half of the sentence above would be true here.
     * Pinned to `SYSTEM_AUDIO_LOST_MESSAGE` in `src-tauri/src/audio/mod.rs`. */
    systemAudioLost:
      "Echo stopped hearing what this computer plays. It's still recording through the microphone.",
    microphoneUnavailable:
      "Echo can hear the meeting, but not your own microphone.",
    /** Nothing is arriving at all: this recording has no microphone in it and
     * what the computer plays is not coming through either. Its own sentence
     * because both of the two above end in a promise about the microphone
     * recording — one that the offline pass will sort the voices out of it, one
     * that Echo is still recording through it — and a person who denied the
     * microphone has no microphone recording for either promise to be about.
     * Pinned to `NOTHING_IS_BEING_HEARD_MESSAGE` in
     * `src-tauri/src/audio/mod.rs`. */
    nothingIsBeingHeard:
      "Echo can't hear anything — there's no microphone in this recording, and nothing is coming from this computer. Nothing is being saved, so it's worth stopping and starting again.",
    transcriptBehind: "Catching up on the last few minutes.",
    storageLow: "Running low on space. Recording keeps going — free up room soon.",
  },
  /**
   * Whether Echo can understand speech right now, said while a meeting is
   * running. Only the two states worth interrupting somebody for are here:
   * "ready" needs no words, and "idle" means nothing is asking for it.
   *
   * `unavailable` is the same sentence the core sends as a notice when the
   * engine fails to come up, word for word, so the banner on the screen and the
   * message that slid past agree instead of sounding like two problems.
   */
  speechState: {
    preparing:
      "Echo is finishing a one-time setup. It's recording everything, and the words will fill in as soon as that's done.",
    unavailable:
      "Echo is recording, but it can't write the words down yet. It will catch up as soon as it can.",
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
    /** The one-time setup a set of speech weights needs on this computer, paid
     * before a meeting has to pay it (incident of 2026-08-24). */
    prepareEngine: "Getting Echo ready",
  },
  /**
   * A stage inside a job, named because a person would read it as a different
   * activity. Only the one-time download has one: once the bytes are here,
   * Echo has to get this particular computer ready to use them, which can take
   * a while and has no percentage to report. Saying "Downloading" through that
   * would be a lie, and saying nothing leaves somebody watching a still bar.
   */
  jobPhase: {
    preparingEngine: "Finishing one-time setup",
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

/**
 * The line above a progress bar, for work that is happening and work that is
 * not.
 *
 * The distinction is the whole point (see `lib/jobs.ts`): a job that has not
 * started says so, in words, instead of borrowing the sentence — and the bar —
 * of the one that is running.
 */
/**
 * The note on a line Echo repaired against "Words Echo should know".
 *
 * Deliberately a whole sentence and deliberately quiet — a title, nothing more.
 * The transcript is what a person reads as the record of the meeting, so a word
 * Echo changed on its own has to be able to say so; but the change is right far
 * more often than not, and a badge on every third line would be noise. Says what
 * was written and what it became, in that order, so the sentence reads the way
 * it happened.
 */
export const transcriptNote = {
  corrected: (changes: { from: string; to: string }[]) =>
    changes.map((c) => `Echo wrote “${c.from}” and changed it to “${c.to}”.`).join(" "),
  /**
   * A line Echo was not sure it heard (`src/pages/Meeting/lib/confidence.ts`).
   * Said as what happened, not as a score: "0.42" is a number nobody can act on,
   * and "low confidence" is jargon for the same number.
   */
  unsure: "Echo wasn't sure it heard this line. The words may be wrong.",
  /**
   * The last sentence of the same tooltip, on a line whose repaired words can
   * still be found in it — which is to say, on a line Echo can put back.
   *
   * Says *line*, not *word*, because that is what happens: the note kept
   * against a line records what was replaced and never where, so one repair
   * among several cannot be undone on its own. Nothing new is drawn for this;
   * the underline that already means "a word was put right" is the thing you
   * click, so a line that is also shaky keeps its dimmed text saying only that.
   */
  undoHint: "Click it to put this line back the way Echo first heard it.",
  /** The same offer for anyone not looking at a tooltip. */
  undoLabel: "Put this line back the way Echo first heard it",
  /** Said once, quietly, after the words go back. */
  undone: "Put back the way Echo heard it.",
} as const;

/**
 * The moments Echo heard and chose not to write down, said on the meeting's
 * own screen (`src-tauri/src/asr/left_out.rs` decides which ones these are).
 *
 * Echo takes the microphone's copy of what this computer played out of the
 * transcript so the other people are written down once instead of twice. That
 * is right almost every time and nothing is missing — but it is measurably
 * wrong sometimes, and when it is, the transcript reads exactly like one where
 * nobody spoke. Only the moments that left no words at all reach this: a
 * decision the transcript covers is not a person's problem, and saying "84
 * stretches" about a perfectly good transcript would be noise nobody could act
 * on.
 *
 * Zero-jargon, and specifically: not a word here names the machinery. What a
 * person needs is what happened ("Echo took it for this computer's own sound"),
 * when it happened, and what to do about it.
 */
export const leftOut = {
  title: (count: number) =>
    count === 1 ? "One moment has no words" : `${count} moments have no words`,
  /** Says what Echo did and that nothing else covered those seconds — the two
   * halves of why the transcript is silent there. */
  explanation: (count: number) =>
    count === 1
      ? "Echo heard something through the microphone here and took it for this computer's own sound coming back, so it didn't write it down. Nothing else was written down at that moment either."
      : "Echo heard something through the microphone at these times and took it for this computer's own sound coming back, so it didn't write them down. Nothing else was written down then either.",
  /**
   * The repair, naming the button that performs it so the two can never drift
   * apart. Only shown while the recording still exists: without it there is
   * nothing to read again, and offering a repair that can't run is worse than
   * offering none.
   */
  repair: (buttonLabel: string) =>
    `If somebody was speaking, “${buttonLabel}” reads the whole recording again and decides afresh.`,
  jumpLabel: (time: string) => `Go to ${time} in the transcript`,
  /** The list is a way in, not the record: the count above is the truth, and
   * fifty times in a row is a wall rather than a list. */
  andMore: (count: number) => `and ${count} more`,
} as const;

export const jobLine = {
  running: (label: string) => `${label}…`,
  waiting: (label: string) => `${label} — waiting its turn`,
  /**
   * A pass that stopped without finishing, said on the screen that was waiting
   * for it. The banner during a mic-only recording promises that Echo will work
   * out who said what once the meeting ends; when the pass can't — the speaker
   * files are allowed to arrive after the first recording, so a meeting can
   * happen before they land — this is what keeps that from being a promise
   * quietly broken behind a transcript where every line still says You. The
   * reason underneath it comes from the core, already in plain words.
   */
  stopped: (label: string) => `${label} — didn't finish`,
  /**
   * Work that is set aside because a recording is going on.
   *
   * Recording always gets the machine first, so on a day of back-to-back
   * meetings the previous meeting's work sits still for as long as the next one
   * lasts. Until now the screen kept whatever it last said — a job name over a
   * bar that had stopped moving — which reads as Echo being stuck. This says
   * what is actually true and that it will carry on by itself, so nobody has to
   * decide whether to press anything.
   */
  deferred: (label: string) => `${label} — paused until the recording ends`,
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
  micOnlyBanner: labels.degradedReason.systemAudioUnavailable,
  systemOnlyBanner: "Echo can hear the meeting, but not your own microphone.",
  unknownSpeaker: "Speaker",
  nothingRecordingTitle: "Nothing is being recorded",
  nothingRecordingDescription: "Start a meeting from Home to see it here.",
  goHomeButton: "Go to Home",
  copyTranscriptButton: "Copy transcript",
  /** The small row that sits at the bottom of the transcript with a pulsing
   * dot, for as long as Echo is actively listening. */
  listeningNow: "Listening…",
  /** The pill that appears once someone has scrolled up to read and new
   * speech has come in since — clicking it returns to following along live. */
  jumpToNow: "Jump to now",
} as Record<string, string>;

/** Meeting detail: Recap, Transcript, Info tabs. */
export const meeting = {
  tabRecap: "Recap",
  tabTranscript: "Transcript",
  tabInfo: "Info",
  noRecapTitle: "No recap yet",
  noRecapDescription: "Write one whenever you're ready.",
  /**
   * The same empty tab while Echo is still writing the meeting down.
   *
   * A recap is written *from* the transcript, and Echo already queues it behind
   * that meeting's transcript work — the queue orders by kind, so the recap runs
   * last whatever order things were asked for in. That was always true and the
   * screen never said it: for the several minutes after a meeting ends, and for
   * the whole of one that is still recording, this tab offered "Write one
   * whenever you're ready" beside a live button, inviting a recap of a
   * transcript Echo was still writing.
   *
   * So the button stays: a promise the queue already keeps is better than a
   * disabled control that has to explain itself.
   */
  noRecapWhileTranscribingTitle: "Still writing the meeting down",
  noRecapWhileTranscribingDescription:
    "A recap is written from the transcript, so Echo finishes that first. Ask for one now and it gets written as soon as the transcript is done.",
  /** The pending card when the recap has been asked for but hasn't started. */
  recapWaitingTitle: "Your recap is waiting",
  /** Why it is waiting, when what it is waiting for is the transcript. */
  recapWaitingForTranscriptDescription:
    "Echo is still writing the meeting down. Your recap gets written as soon as that's done.",
  /** Why it is waiting, when it is simply not its turn yet. */
  recapWaitingDescription: "Echo starts on it as soon as it's free.",
  /**
   * The transcript pass stopped before the end.
   *
   * Without this a recap would be written from an incomplete transcript with
   * nothing anywhere saying so. Same shape as the Transcript tab's own notice
   * for the same stopped pass, which carries the reason underneath in the
   * core's own words.
   */
  recapTranscriptStoppedTitle: "The transcript isn't finished",
  recapTranscriptStoppedDescription:
    "Echo stopped before the end of the recording, so a recap written now would miss whatever it didn't reach.",
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
  listenAgainButton: "Listen again",
  listenAgainConfirmTitle: "Rewrite the transcript from the recording?",
  listenAgainConfirmDescription: "The current one is replaced — the recap stays.",
  listenAgainConfirmButton: "Rewrite",

  // People count (Transcript tab)
  peopleCountSingular: "person",
  peopleCountPlural: "people",
  peopleCountDetectedSuffix: "detected",
  peopleCountTriggerTitle: "Change how many people were in this meeting",
  peopleCountEditTitle: "How many people were in this meeting?",
  peopleCountEditDescription: "Echo uses this to work out who said what.",
  peopleCountFewer: "Fewer people",
  peopleCountMore: "More people",
  peopleCountBackToAutomatic: "Back to automatic",
  peopleCountConfirmDescription: "Names you gave speakers may need redoing.",
  peopleCountRedoButton: "Redo",

  // "Name your speakers" dialog (Transcript tab): the participants stepper,
  // the redo it triggers, and one row per speaker with a sample and a name.
  speakersDialogTitle: "Name your speakers",
  speakersDialogDescription: "Play a sample, then type who it is.",
  speakersDialogParticipantsLabel: "Participants",
  /** "Redo", not "Re-run": the question this button answers is worded
   * "Redo who said what…?" (`peopleCount.confirmTitle`), and a button that
   * disagrees with its own question reads as a different action. */
  speakersDialogRerunButton: "Redo",
  speakersDialogListenLabel: "Listen",
  speakersDialogStopLabel: "Stop",
  speakersDialogSampleError: "Echo couldn't play a sample for this voice.",
  speakersDialogEmpty: "Echo hasn't worked out who's speaking yet.",

  // Known people (voice enrollment), inside "Name your speakers": the
  // suggestion chip, the known-people combo on the name field, "remember
  // this voice" enrollment, and a linked row's mark/unlink/local-rename hint.
  speakersDialogSuggestionPrefix: "Looks like",
  speakersDialogSuggestionConfirm: "Confirm",
  speakersDialogKnownPersonHint: "Echo recognizes this voice",
  speakersDialogRowMenuLabel: "More options",
  /** Mantra 2: "Unlink" names the link, which is Echo's idea, not the person's.
   * What they mean is that Echo got the voice wrong. */
  speakersDialogUnlinkAction: "This isn't them",
  /** Shown under the name field while editing a row already linked to a known
   * person — renaming here only relabels this meeting, it doesn't rename the
   * person, and that has to be said plainly rather than discovered later. */
  speakersDialogLocalRenameHint: "Just for this meeting",
  speakersDialogRememberVoice: "Remember this voice",
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
  speechReadyTitle: "Echo can understand speech",
  speechReadyDescription: "It's ready to listen whenever you record.",
  speechReadyBadge: "Ready to use",
  speechNotReadyTitle: "Not set up yet",
  speechNotReadyDescription:
    "Echo needs a one-time download before it can understand speech.",
  /** Appended to the line above with the real total for this computer, so the
   * size a person reads here is the same one onboarding quoted. */
  speechDownloadSizeNote: "It takes about",
  /** Ready *and* still downloading. Two situations reach this, and the sentence
   * has to be true of both: the first download is finishing its last pieces, or
   * a better way to understand speech is arriving to replace what is already
   * here. Either way the only thing the person needs to know is that recording
   * works meanwhile — so that is all it says. Without this line, "Ready to use"
   * next to a progress bar reads as a bug. */
  speechImprovingDescription:
    "Echo is still downloading part of this. Recording works as usual while it finishes — nothing to do.",
  speechStorageLabel: "Storage used",
  speechDownloadButton: "Download",
  speechResumeButton: "Resume download",
  speechRemoveButton: "Remove",
  speechCancelButton: "Cancel",
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

  // People (known voices)
  sectionPeople: "People",
  /** Mantra 2: "profile" is a word about how this is built, not about what it
   * is. What it is, is a bit of somebody's voice, kept here, deletable — and
   * DESIGN §1 asks for exactly those three things in one plain sentence. */
  peopleIntro:
    "Echo keeps a little of these voices on this Mac so it can recognize them again. You can delete any of them at any time.",
  peopleSavedEmptyTitle: "No voices saved yet",
  peopleSavedEmptyDescription:
    "Save someone's voice from a meeting's transcript, and Echo will remember it here.",
  peopleListenLabel: "Listen",
  peopleStopLabel: "Stop",
  peopleSampleError: "Echo couldn't play a sample for this voice.",
  peopleNamePlaceholder: "Name",
  peopleForgetButton: "Forget",
  peopleDeleteConfirmTitle: "Forget this voice?",
  peopleDeleteConfirmDescription: "The saved samples are deleted too.",
  /**
   * Combining two saved voices — the same shape as "Combine two speakers" in a
   * meeting, and deliberately the same words, because it is the same idea said
   * about the list that outlives the meeting.
   *
   * The one difference is that this one cannot be taken back, so it gets a
   * second step that says so before anything happens. What it says there is
   * what the core actually does: both sets of saved sound are kept as one, the
   * overlapping bits are dropped, meetings already read do not change.
   */
  peopleMergeButton: "Combine two voices",
  peopleMergeTitle: "Which two are the same person?",
  peopleMergeDescription:
    "Pick two names below, and the one whose name to keep. Everything Echo has saved for the other moves across.",
  peopleMergeKeepLabel: "Keep this name",
  peopleMergeNeedTwo: "Pick exactly two names to combine.",
  peopleMergeConfirmTitle: "Combine these two voices?",
  peopleMergeConfirmButton: "Combine",
  peopleMergeBackButton: "Back",
  /** `needsRefresh` on a saved person: never actionable, just said once and
   * quietly — Echo runs the refresh itself. */
  peopleRefreshingNote: "Echo is refreshing how it recognizes voices.",
  peopleSuggestedTitle: "People Echo keeps hearing",
  peopleSuggestedDescription:
    "Voices that keep turning up in your meetings, without a name yet.",
  peopleSuggestedEmptyTitle: "Nothing recurring yet",
  peopleSuggestedEmptyDescription:
    "Once the same voice turns up in a few meetings, it'll show up here.",
  peopleSuggestedUnnamed: "Unnamed voice",
  /** The field, not the button: a placeholder says what to type, and "Save as…"
   * next to a Save button said what the button does twice instead. */
  peopleSaveAsPlaceholder: "Their name",
  peopleSaveAsButton: "Save",

  // Words Echo should know
  sectionWords: "Words",
  /** Mantra 2 all the way through: no "vocabulary", no "glossary", no
   * "prompt" — just the names, and what typing one does. The second sentence is
   * the honest limit: this makes those words far more likely to come out right,
   * and promising more than that would be a promise a listener cannot keep. */
  wordsIntro:
    "Names Echo tends to get wrong: a product, a company, a street, someone you work with. Add them here and Echo will watch for them while it writes your meetings down.",
  wordsAddPlaceholder: "A name or a word",
  wordsAddButton: "Add",
  wordsEmptyTitle: "Nothing here yet",
  wordsEmptyDescription:
    "Add the names that come up in your meetings, and Echo will spell them the way you do.",
  /** On a row Echo added itself, from an enrolled voice. Says where it came
   * from, not how it got there — and it can be removed like any other. */
  wordsFromPersonNote: "From a voice you saved",
  wordsRemoveButton: "Remove",
  wordsTooLongError: "That's longer than a name. Add one word or two.",
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
  stepDownloadDescription:
    "This happens once. After that, Echo works without sending anything anywhere.",
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

  /** "4.3 GB, one time", from the number the core reports for this platform.
   * A function rather than a constant because the total differs between macOS
   * and Linux (the speed-up file is Apple-only) and because it moves whenever
   * the catalog does — a hardcoded size here went stale the first time the
   * speech model changed. `anchors.downloadingSpeechSize` is the fallback for
   * the moment before the number has arrived. */
  downloadSize: (formatted?: string) =>
    formatted ? `${formatted}, one time` : anchors.downloadingSpeechSize,

  stepLabelWelcome: "Welcome",
  stepLabelPermissions: "Permissions",
  stepLabelDownload: "Download",
  stepLabelSummaries: "Recaps",
  quitButton: "Quit Echo",
  downloadContinueAnywayNote: "You can finish this later in Settings.",
  summariesUseOnDevice: "Use Ollama",
  recapsChosenOnDevice: "Recaps will be written by Ollama on this Mac.",
  recapsChosenGemini: "Recaps will be written using Google Gemini.",
} as const;

/** Banners, toasts and confirmations. */
export const notices = {
  somethingWentWrong: "Something went wrong. Nothing was lost.",
  setupDone: "Echo is ready — recording and meeting alerts are on.",
  systemAudioLost: labels.degradedReason.systemAudioLost,
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
  /** Echo replaced the speech model with a better one, on its own. Said once,
   * quietly, and only after the new one is in place and working — matches the
   * `speechUpgraded` notice the core emits. */
  speechUpgraded: "Echo upgraded how it understands speech.",
} as Record<string, string>;

/**
 * The few strings docs/DESIGN.md quotes directly. They set the tone for
 * everything added later, so they are here from the start.
 */
export const anchors = {
  downloadingSpeech: "Downloading what Echo needs to understand speech",
  /** The size DESIGN quotes. macOS, where the speed-up file is part of it; the
   * screens all render `downloadSize(bytes)` with the real number instead, and
   * this is only the fallback for the moment before it has loaded. */
  downloadingSpeechSize: "4.3 GB, one time",
  listening: "Listening…",
  writingRecap: "Writing your recap…",
  waitingForMeeting: "Waiting for your next meeting",
  systemAudioMeaning: "everything your computer plays",
  catchingUp: "Catching up on the last few minutes",
  workingOutSpeakers: "Working out who said what",
  micOnlyBanner: labels.degradedReason.systemAudioUnavailable,
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

/**
 * The quiet "N people" control on the Transcript tab: the trigger's label and
 * the one confirmation before a redo. Split out from `meeting` because these
 * two need a number folded in, not just a fixed sentence.
 */
export const peopleCount = {
  /** "3 people", or "3 people (detected)" when nobody has overridden it. */
  triggerLabel: (n: number, detected: boolean) =>
    `${n} ${n === 1 ? meeting.peopleCountSingular : meeting.peopleCountPlural}${
      detected ? ` (${meeting.peopleCountDetectedSuffix})` : ""
    }`,
  /** `null` means "back to automatic" — there is no fixed number to name yet. */
  confirmTitle: (n: number | null) =>
    n === null
      ? "Redo who said what automatically?"
      : `Redo who said what for ${n} ${n === 1 ? meeting.peopleCountSingular : meeting.peopleCountPlural}?`,
  /**
   * Beside the participants stepper when someone asked for more people than
   * the recording holds separable voices. Says what Echo can hear and asks for
   * nothing: the number they typed is about the room they were in, and it
   * stays. Same sentence the backend puts in the message when the pass
   * finishes, so the two never disagree.
   */
  voicesFound: (n: number) =>
    n === 0
      ? "Echo can't tell any voices apart in this recording."
      : n === 1
        ? "Echo can only hear one voice clearly in this recording."
        : `Echo can only hear ${n} distinct voices in this recording.`,
  /**
   * Beside the participants stepper when the count is Echo's own automatic
   * reading. Shows the best count it decided against, because a wrong count is
   * the one speaker mistake nothing here can fix later — merges exist, splits
   * don't — and a decision nobody can see is a decision nobody can argue with.
   * An invitation to correct the number, not a confession of error.
   */
  alternativeCount: (n: number) =>
    n === 1
      ? "One voice was Echo's next best reading. If that's right, lower the number."
      : `${n} was Echo's next best reading. If that's right, set the number to ${n}.`,
} as const;

/**
 * Settings > People: the two bits of copy that need a value folded in — how
 * long since Echo last matched a saved voice, and how many samples it's
 * built from. Split out from `settings` for the same reason `peopleCount`
 * is: those live in a `Record<string, string>` and these are functions.
 */
export const knownPeople = {
  lastHeard: (relative: string) => `Last heard ${relative}`,
  neverHeard: "Hasn't been heard again yet",
  sampleCount: (n: number) => `${n} ${n === 1 ? "sample" : "samples"}`,
  /** A recurring unnamed voice: how many meetings it has turned up in. Without
   * this, two rows of "Unnamed voice" give nobody anything to decide with. */
  heardIn: (n: number) => `Heard in ${n} ${n === 1 ? "meeting" : "meetings"}`,
  /**
   * The second step of combining two saved voices, which names them because
   * the whole risk of the thing is combining the wrong two.
   *
   * Every clause is something the core really does: both sets of saved sound
   * end up behind one name, the overlapping bits are dropped (that dropping is
   * what makes the result one voice rather than two), and meetings somebody has
   * already read are left exactly as they read. It ends on the sentence that
   * decides whether anybody should click.
   */
  combineConfirm: (keep: string, merge: string) =>
    `Echo keeps “${keep}” and puts everything it has saved for “${merge}” behind that one name. Some of it is dropped along the way, where the two voices overlap. Meetings you've already read stay exactly as they read. This can't be undone.`,
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
