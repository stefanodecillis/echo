//! Capture: microphone, system audio, the meeting clock, per-channel audio on
//! disk, and speech detection.
//!
//! Contract (DESIGN §3 pipeline, mantra 3):
//! * Two independent sources produce timestamped frames. Each frame carries an
//!   offset on the **monotonic meeting clock**, never a wall clock.
//! * Frames go straight to per-channel chunks from t=0. That file is the source
//!   of truth; everything else is derived and may be recomputed.
//! * Audio callbacks allocate nothing, lock nothing, touch no database and run
//!   no inference. They push into a bounded [`ringbuf`] and return.
//! * On overflow we drop *live* work, never audio: the ASR catch-up job reads
//!   from disk afterwards.
//! * Losing one source degrades capture (banner) instead of ending it.
//!
//! ## The threads
//!
//! ```text
//! device callback ─► bounded ring ─► pump thread ─┬─► writer (disk, blocking)
//!                    (never blocks)               └─► frame queue ─► speech
//!                                                    (drop oldest)    detection
//!                                                                        │
//!                                                              signal channel
//!                                                            (session pipeline)
//! ```
//!
//! The pump owns the devices and the writers, so disk work can never be starved
//! by inference. Utterances leave through the signal channel the moment they are
//! found; the queue that can fall behind — and that decides what live work is
//! dropped — belongs to the session pipeline, which is the thing actually
//! waiting on the speech engine.

pub mod bleed;
pub mod bleed_guard;
pub mod clock;
pub mod mic;
pub mod resample;
pub mod ring;
pub mod route;
pub mod vad;
pub mod writer;

#[cfg(target_os = "macos")]
pub mod system_macos;

#[cfg(target_os = "linux")]
pub mod system_linux;

#[cfg(target_os = "linux")]
use system_linux as system;
#[cfg(target_os = "macos")]
use system_macos as system;

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

use crate::audio::bleed::{BleedEvidence, LAG_MIN_MS};
use crate::audio::bleed_guard::{BleedGuard, Verdict};
use crate::audio::clock::MeetingClock;
use crate::audio::vad::{OpenSpeech, SpeechDetector, Utterance};
use crate::audio::writer::{ChunkWriter, CommittedChunk};
use crate::types::{AudioDevice, Channel, DegradedReason, PermissionState, PermissionTarget};

/// Everything downstream works at 16 kHz mono per channel.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Frames handed downstream are this long, so latency stays predictable.
pub const FRAME_MS: u32 = 20;

/// One chunk covers this much audio before we roll to the next file.
/// Small enough that a crash loses at most this much, big enough that the
/// journal stays cheap.
pub const CHUNK_SECONDS: u32 = 30;

/// How often the pump looks for new audio. Half a frame, so a frame never waits.
const PUMP_INTERVAL: Duration = Duration::from_millis(10);

/// Frames waiting for speech detection before the oldest start being dropped.
/// Five seconds per channel: enough for any hiccup, short enough that the live
/// transcript never drifts far behind what is being said.
const FRAME_QUEUE_CAPACITY: usize = 5 * 1_000 / FRAME_MS as usize;

/// How long to wait before trying a lost system-audio stream again.
const REOPEN_BACKOFF: Duration = Duration::from_secs(5);

/// New speech before another [`CaptureSignal::SpeechSoFar`] is offered for a
/// channel, measured in audio rather than wall clock.
///
/// The detect thread wakes every 20 ms; a snapshot carries seconds of audio, so
/// offering one per wake would copy megabytes a second for a consumer that
/// decodes a caption every three. This is comfortably finer-grained than the
/// pipeline's own caption step, so the consumer never waits on this side for
/// material — it just has fresher material than it needs, which is the cheap
/// direction to be wrong in.
const SNAPSHOT_OFFER_MS: i64 = 1_000;

/// How long a channel that *started* may deliver nothing before Echo says so.
///
/// Long enough that a slow first buffer, a device switch or a quiet opening are
/// never mistaken for a failure; short enough that the person finds out inside
/// the first minute rather than at the end of the meeting.
const SILENT_CHANNEL_GRACE: Duration = Duration::from_secs(10);

/// Three sibling messages for the three ways the system channel can be
/// missing: never opened, opened but silent, and opened then lost mid-meeting.
/// They must never contradict each other or the banner on screen, so they are
/// named consts in one place rather than three hand-typed string literals —
/// that is exactly how the copy drifted before (four hand-maintained copies of
/// one sentence, one of them wrong, incident of 2026-08-24). A test below
/// sweeps all three together.
///
/// What the person is told when the computer's audio started but carries
/// nothing. Word for word `labels.degradedReason.systemAudioUnavailable` in
/// `src/lib/copy.ts` — a test pins the two together.
///
/// The second sentence is a promise, and it is only keepable in this state.
/// [`SilenceWatchdog`] counts frames for the whole recording and never resets
/// them, so `WentSilent` can only fire when *no* system audio has ever arrived —
/// which is the one case where `diarize::pipeline::voice_channel` picks
/// `Source::MicOnly` and the offline pass really does re-cut the microphone
/// recording into separate voices. A meeting that had system audio for even a
/// minute takes the other branch and pins every microphone line to "You"
/// forever, which is why the stream-died-mid-meeting case below promises
/// nothing.
pub const SYSTEM_AUDIO_SILENT_MESSAGE: &str = "Echo is recording through the microphone only, so it can't tell who is speaking — every line says You for now. It will work out who said what once the meeting ends.";

/// What the person is told when the system channel never opened at all, so
/// the recording started mic-only from t=0 (`session::mod::start`). Same state
/// as [`SYSTEM_AUDIO_SILENT_MESSAGE`] — nothing from this computer will ever be
/// in this recording — said as the thing that just happened rather than as the
/// standing description the banner carries.
pub const SYSTEM_AUDIO_UNAVAILABLE_AT_START: &str =
    "Echo can't hear what this computer plays. It's recording through the microphone only.";

/// What the person is told when a system-audio stream that was working dies
/// mid-meeting. It says what changed and stops there: the labels on the lines
/// already written are real, the ones from here on are not separated by the
/// offline pass, and there is nothing useful to promise about either
/// ([`crate::types::DegradedReason::SystemAudioLost`]).
pub const SYSTEM_AUDIO_LOST_MESSAGE: &str =
    "Echo stopped hearing what this computer plays. It's still recording through the microphone.";

/// What the person is told when there is nothing left to hear: this recording
/// has no microphone in it, and what the computer plays is not arriving either.
///
/// The fourth sibling, and the one that stops the other three from lying. Both
/// sentences above are written for a recording that still has a microphone in
/// it — one promises the offline pass will sort the voices out of the
/// microphone recording, the other says Echo is "still recording through the
/// microphone". Said to somebody who denied the microphone and is capturing
/// only this computer, either one is false, and false in the direction that
/// costs a meeting: it reads as "carry on, Echo has you". So a recording with
/// no microphone and no arriving system audio gets its own reason
/// ([`crate::types::DegradedReason::NothingIsBeingHeard`]) and its own
/// sentence, which promises nothing and says what to do.
///
/// Word for word `labels.degradedReason.nothingIsBeingHeard` in
/// `src/lib/copy.ts`; a test below pins the two together.
pub const NOTHING_IS_BEING_HEARD_MESSAGE: &str = "Echo can't hear anything — there's no microphone in this recording, and nothing is coming from this computer. Nothing is being saved, so it's worth stopping and starting again.";

/// Loudness is reported ten times a second, capped here rather than in the UI.
const LEVELS_INTERVAL: Duration = Duration::from_millis(100);

#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("not implemented yet")]
    NotImplemented,
    #[error("no microphone is available")]
    NoInputDevice,
    #[error("permission to record has not been granted")]
    PermissionDenied,
    #[error("capturing what the computer plays is not available on this system: {0}")]
    SystemAudioUnsupported(String),
    #[error("the audio device went away")]
    DeviceLost,
    #[error("could not write audio to disk: {0}")]
    Write(String),
    #[error("audio backend error: {0}")]
    Backend(String),
}

/// A block of 16 kHz mono audio with its position on the meeting clock.
///
/// Produced by the resampler, consumed by the writer and the speech detector.
#[derive(Debug, Clone, Default)]
pub struct Frame {
    pub channel: Channel,
    /// Offset from the start of the meeting.
    pub t_start_ms: i64,
    /// 16 kHz mono, -1.0..=1.0.
    pub samples: Vec<f32>,
}

impl Frame {
    pub fn duration_ms(&self) -> i64 {
        (self.samples.len() as i64 * 1_000) / i64::from(TARGET_SAMPLE_RATE)
    }

    pub fn t_end_ms(&self) -> i64 {
        self.t_start_ms + self.duration_ms()
    }

    /// Loudest sample in this frame, for the recording indicator.
    pub fn peak(&self) -> f32 {
        self.samples.iter().fold(0.0f32, |m, s| m.max(s.abs()))
    }
}

/// What to capture for one recording.
#[derive(Debug, Clone)]
pub struct CaptureConfig {
    pub meeting_id: String,
    /// Directory for this meeting's chunks; created if missing.
    pub audio_dir: PathBuf,
    /// Capture what the computer plays as well as the microphone.
    pub capture_system_audio: bool,
    /// Specific input device, or the system default when None.
    pub input_device_id: Option<String>,
    /// Where Echo's logs live, which is where the one file that switches bleed
    /// suppression off is looked for
    /// ([`crate::audio::bleed_guard::KILL_SWITCH_FILE`]).
    ///
    /// Plumbed through the config rather than read from a global, for the same
    /// reason `audio_dir` is: this module is handed the folders it may touch
    /// and finds none of them out for itself, so a test can point a recording
    /// anywhere and every path in it stays inside the temporary directory.
    /// `None` — the default, and every test that does not care — means there is
    /// no folder to look in and therefore no kill switch.
    pub log_dir: Option<PathBuf>,
}

impl CaptureConfig {
    /// Everything that is not the two required fields has a safe default.
    pub fn new(meeting_id: impl Into<String>, audio_dir: impl Into<PathBuf>) -> Self {
        Self {
            meeting_id: meeting_id.into(),
            audio_dir: audio_dir.into(),
            capture_system_audio: true,
            input_device_id: None,
            log_dir: None,
        }
    }

    pub fn with_system_audio(mut self, capture: bool) -> Self {
        self.capture_system_audio = capture;
        self
    }

    pub fn with_input_device(mut self, device_id: Option<String>) -> Self {
        self.input_device_id = device_id;
        self
    }

    pub fn with_log_dir(mut self, log_dir: Option<PathBuf>) -> Self {
        self.log_dir = log_dir;
        self
    }
}

/// What actually started, so the session layer can report Degraded honestly.
#[derive(Debug, Clone, Default)]
pub struct CaptureStarted {
    pub channels: Vec<Channel>,
    /// Set when we wanted system audio and could not get it.
    pub system_audio_error: Option<String>,
    /// Set when the microphone could not be opened but system audio could.
    pub microphone_error: Option<String>,
    /// True when speech detection is running on the loudness fallback because
    /// the downloaded detector was not there. Advanced diagnostics only.
    pub speech_detection_degraded: bool,
}

/// Things capture tells the session layer about while it runs.
///
/// Everything user-visible in here is already a plain sentence (mantra 2); the
/// session layer decides whether it becomes a banner or a log line.
#[derive(Debug, Clone)]
pub enum CaptureSignal {
    /// A chunk is durable on disk and can be journalled.
    ChunkCommitted(CommittedChunk),
    /// Speech was found. Whether it gets live text is the pipeline's decision;
    /// the audio is on disk either way.
    UtteranceReady(Utterance),
    /// This stretch of the microphone was the microphone's copy of what the
    /// computer played, so it is not offered for text — the same words are
    /// already going into the transcript from the computer's own side of the
    /// call ([`crate::audio::bleed`]).
    ///
    /// It carries no audio, because there is nothing left to decode: the
    /// recording on disk is untouched and no text is written for these seconds.
    /// Two things are owed instead — retiring the half-written line the live
    /// view has been showing, which the first three fields name, and writing
    /// down that the decision was made, which is what `evidence` is for.
    ///
    /// **The decision is recorded, and that is new.** It used to be that
    /// suppression wrote nothing at all, on the reasoning that the catch-up
    /// pass would reach the same verdict over the same audio and so the seconds
    /// could simply be left blank. The live verification of 2026-08-26 measured
    /// that: six stretches suppressed live, five of them suppressed again
    /// offline, and the sixth written to the microphone channel as a duplicate
    /// nobody said twice. The two passes do not always agree — the live one
    /// judges against a ring on the meeting clock and a delay this machine has
    /// already measured, the offline one reads paged audio off disk and always
    /// searches cold — so the verdict is written down where the offline planner
    /// can read it (`migrations/0007_suppressed_spans.sql`).
    ///
    /// **Suppression is deduplication, never deletion.** If it ever stops being
    /// that — a recording with no system channel, a system channel that is not
    /// live — [`crate::audio::bleed_guard::BleedGuard`] refuses to judge at all
    /// and this signal is never sent.
    UtteranceSuppressed {
        channel: Channel,
        t_start_ms: i64,
        t_end_ms: i64,
        /// What the guard measured. Kept whole rather than flattened into three
        /// numbers, so the row and the log line say the same thing the
        /// correlator said.
        evidence: BleedEvidence,
    },
    /// A look at speech that is *still going*, so the live view can show words
    /// while someone is still talking instead of only once they stop.
    ///
    /// Offered often and cheap to ignore: the pipeline decides which of these
    /// are worth a decode and drops the rest ([`crate::audio::vad::OpenSpeech`]).
    SpeechSoFar(OpenSpeech),
    /// Smoothed loudness for the recording indicator. Rate-capped here, not in
    /// the UI (DESIGN §3).
    Levels { mic: f32, system: f32, t_ms: i64 },
    /// Capture is still going, but with less than we wanted.
    Degraded {
        channel: Option<Channel>,
        reason: DegradedReason,
        /// One sentence, no jargon, says what the person can do.
        message: String,
    },
    /// Something that was lost came back.
    Recovered { channel: Channel },
    /// Audio can no longer be written. Committed chunks are safe.
    StorageFailed { message: String },
}

/// Bounded frame queue that drops its *oldest* entry when full.
///
/// Live work is what gives way (DESIGN §3): audio is already on disk by the time
/// a frame gets here, so losing one costs a few words of live caption and
/// nothing else.
#[derive(Debug)]
struct FrameQueue {
    inner: Mutex<VecDeque<Frame>>,
    capacity: usize,
    dropped: AtomicU32,
}

impl FrameQueue {
    fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(VecDeque::with_capacity(capacity)),
            capacity,
            dropped: AtomicU32::new(0),
        }
    }

    fn push(&self, frame: Frame) {
        let mut queue = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        while queue.len() >= self.capacity {
            queue.pop_front();
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
        queue.push_back(frame);
    }

    fn drain(&self, out: &mut Vec<Frame>) {
        let mut queue = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        out.extend(queue.drain(..));
    }

    fn dropped(&self) -> u32 {
        self.dropped.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// The silent-channel watchdog
// ---------------------------------------------------------------------------

/// What the watchdog decided this tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SilenceVerdict {
    /// The channel started, has produced nothing, and the other channel is
    /// clearly producing — so the clock is running and this really is a
    /// failure. Tell the person once.
    WentSilent,
    /// Frames finally arrived after we had already said so.
    CameBack,
}

/// What stands as proof that this recording is really running, so that "nothing
/// arrived on the watched channel" means a failure rather than a stopped world.
///
/// Chosen once, when the recording opens, from whether there is a second source
/// at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProofOfLife {
    /// The microphone's own frames. The strongest proof there is: another
    /// device, on another thread, handing over audio right now.
    MicrophoneFrames,
    /// There is no microphone in this recording, so there are no frames to
    /// compare against and the meeting clock stands in — the amount of audio
    /// this recording *should* contain by now, paused stretches removed
    /// ([`crate::audio::clock::MeetingClock::audio_ms`]).
    ///
    /// Weaker than the microphone, and deliberately so. It is the only proof
    /// available when somebody grants screen recording, denies the microphone
    /// and records this computer alone — the configuration where the watchdog
    /// used to be unable to fire at all, because its reference was the frames
    /// of a device that was never opened. That person recorded three quarters
    /// of an hour of nothing and was told about it only at the end, and wrongly
    /// (incident of 2026-08-21). Weak proof that can speak beats strong proof
    /// that cannot.
    MeetingClock,
}

/// Notices a channel that opened successfully and then delivered nothing.
///
/// This is the hole the first real meeting fell into: ScreenCaptureKit reported
/// a running stream, no error and no stop, and handed over not one audio buffer.
/// From above the capture layer that is indistinguishable from a meeting where
/// nobody else spoke — so nothing was reported, nothing was written, and the
/// person found out afterwards.
///
/// The proof of life is what makes the judgement safe: "no frames here" only
/// means something when the recording is otherwise going. A machine asleep or a
/// pump that never ran leaves the proof unsatisfied along with the watched
/// channel, and the watchdog stays quiet.
///
/// No clock of its own and no I/O: the caller passes `now` and the meeting
/// clock's reading, which is what makes this testable without waiting ten
/// seconds.
#[derive(Debug)]
struct SilenceWatchdog {
    started_at: Instant,
    grace: Duration,
    proof: ProofOfLife,
    watched_frames: u64,
    reference_frames: u64,
    /// Highest reading of the meeting clock's audio time seen so far. Only read
    /// under [`ProofOfLife::MeetingClock`]; kept as a maximum because the clock
    /// is monotonic and a lock taken under contention should never be able to
    /// walk it backwards.
    recorded_ms: i64,
    /// True once the person has been told, so the banner is sent once.
    reported: bool,
}

impl SilenceWatchdog {
    fn new(started_at: Instant, grace: Duration, proof: ProofOfLife) -> Self {
        Self {
            started_at,
            grace,
            proof,
            watched_frames: 0,
            reference_frames: 0,
            recorded_ms: 0,
            reported: false,
        }
    }

    /// Count frames seen this tick on the watched channel and on the reference
    /// one, and take this tick's reading of the meeting clock.
    ///
    /// `reference` is 0 for a recording with no microphone, and `recorded_ms` is
    /// ignored for one that has a microphone: each proof reads only its own
    /// half. Both are passed every tick anyway so the pump never has to know
    /// which proof this watchdog was built with.
    fn observe(&mut self, watched: u64, reference: u64, recorded_ms: i64) {
        self.watched_frames = self.watched_frames.saturating_add(watched);
        self.reference_frames = self.reference_frames.saturating_add(reference);
        self.recorded_ms = self.recorded_ms.max(recorded_ms);
    }

    fn watched_frames(&self) -> u64 {
        self.watched_frames
    }

    fn reference_frames(&self) -> u64 {
        self.reference_frames
    }

    fn proof(&self) -> ProofOfLife {
        self.proof
    }

    /// `Some(..)` exactly on the tick the answer changes; `None` otherwise.
    fn poll(&mut self, now: Instant) -> Option<SilenceVerdict> {
        if self.reported {
            if self.watched_frames > 0 {
                self.reported = false;
                return Some(SilenceVerdict::CameBack);
            }
            return None;
        }
        let clock_is_running = match self.proof {
            ProofOfLife::MicrophoneFrames => self.reference_frames > 0,
            // A whole grace period's worth of *recorded* audio, not of wall
            // clock: the same ten seconds the patience below is measured in,
            // but with paused stretches taken out. Somebody who starts a
            // recording and pauses it to find the meeting link is not told the
            // computer has gone quiet while they are paused — which is exactly
            // what a bare wall-clock reading would do.
            ProofOfLife::MeetingClock => self.recorded_ms >= self.grace.as_millis() as i64,
        };
        let out_of_patience = now.duration_since(self.started_at) >= self.grace;
        if self.watched_frames == 0 && clock_is_running && out_of_patience {
            self.reported = true;
            return Some(SilenceVerdict::WentSilent);
        }
        None
    }
}

/// Live state the UI polls, all lock-free.
#[derive(Debug)]
struct SessionShared {
    paused: AtomicBool,
    stopping: AtomicBool,
    /// Set by the pump once it has drained the devices for the last time, so
    /// speech detection knows the frame queue will get nothing more.
    pump_finished: AtomicBool,
    elapsed_ms: AtomicI64,
    /// f32 bits, smoothed 0.0..=1.0.
    mic_level: AtomicU32,
    system_level: AtomicU32,
    mic_active: AtomicBool,
    system_active: AtomicBool,
}

impl SessionShared {
    fn store_level(slot: &AtomicU32, value: f32) {
        slot.store(value.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
    }

    fn load_level(slot: &AtomicU32) -> f32 {
        f32::from_bits(slot.load(Ordering::Relaxed))
    }
}

/// A running capture. Dropping it stops capture and commits the open chunks.
#[derive(Debug)]
pub struct CaptureSession {
    meeting_id: String,
    shared: Arc<SessionShared>,
    clock: Arc<Mutex<MeetingClock>>,
    signals: Mutex<Option<UnboundedReceiver<CaptureSignal>>>,
    threads: Mutex<Vec<std::thread::JoinHandle<()>>>,
}

impl CaptureSession {
    /// Open the requested sources and begin writing chunks from t=0.
    ///
    /// Starting with only the microphone is a success with
    /// [`CaptureStarted::system_audio_error`] set, never an error. Only losing
    /// *both* sources is a failure.
    pub async fn start(cfg: CaptureConfig) -> Result<(Self, CaptureStarted), AudioError> {
        Self::start_with_detector(cfg, None).await
    }

    /// Same, with the downloaded speech detector.
    ///
    /// `detector_path` comes from
    /// `asr::models::installed_path(db, AssetKind::SpeechDetector)`. `None` —
    /// not downloaded yet, or the person skipped it — is fine: speech detection
    /// falls back to a loudness gate and the recording still happens, which is
    /// what makes a first-launch recording possible.
    pub async fn start_with_detector(
        cfg: CaptureConfig,
        detector_path: Option<PathBuf>,
    ) -> Result<(Self, CaptureStarted), AudioError> {
        std::fs::create_dir_all(&cfg.audio_dir)
            .map_err(|e| AudioError::Write(format!("could not use the recordings folder ({e})")))?;

        let clock = MeetingClock::start();
        let origin = clock.origin();

        // The microphone first: it is what "You" means, and it is the source we
        // fight hardest to keep.
        let mic = match mic::MicCapture::open_at(cfg.input_device_id.as_deref(), origin) {
            Ok(m) => Some(m),
            Err(e) => {
                tracing::warn!(target: "echo::audio", "could not open the microphone: {e}");
                None
            }
        };
        let microphone_error = if mic.is_none() {
            Some("Echo could not use your microphone, so only what your computer plays is being recorded.".to_string())
        } else {
            None
        };

        let mut system_audio_error = None;
        let system_capture = if cfg.capture_system_audio {
            match open_system_capture(origin).await {
                Ok(s) => Some(s),
                Err(e) => {
                    system_audio_error = Some(describe_system_error(&e));
                    None
                }
            }
        } else {
            None
        };

        if mic.is_none() && system_capture.is_none() {
            let detail = microphone_error
                .clone()
                .or_else(|| system_audio_error.clone())
                .unwrap_or_else(|| "no sound source was available".to_string());
            return Err(AudioError::Backend(detail));
        }

        let mut channels = Vec::new();
        if mic.is_some() {
            channels.push(Channel::Mic);
        }
        if system_capture.is_some() {
            channels.push(Channel::System);
        }

        let detector_available = detector_path.as_deref().is_some_and(|p| p.exists());

        let shared = Arc::new(SessionShared {
            paused: AtomicBool::new(false),
            stopping: AtomicBool::new(false),
            pump_finished: AtomicBool::new(false),
            elapsed_ms: AtomicI64::new(0),
            mic_level: AtomicU32::new(0.0f32.to_bits()),
            system_level: AtomicU32::new(0.0f32.to_bits()),
            mic_active: AtomicBool::new(mic.is_some()),
            system_active: AtomicBool::new(system_capture.is_some()),
        });

        let (signal_tx, signal_rx) = unbounded_channel();
        let frames = Arc::new(FrameQueue::new(FRAME_QUEUE_CAPACITY * 2));
        let clock = Arc::new(Mutex::new(clock));

        // The pump moves frames to disk; the speech thread turns them into
        // utterances. Both are created up front so a failure to spawn either one
        // is a failure to start, not a half-open recording.
        let threads = vec![
            spawn_pump(
                cfg.clone(),
                mic,
                system_capture,
                Arc::clone(&shared),
                Arc::clone(&clock),
                Arc::clone(&frames),
                signal_tx.clone(),
            )?,
            spawn_speech(
                detector_path,
                cfg.log_dir.clone(),
                // Whether a system stream actually opened, not whether one was
                // asked for: with no second copy being written, suppression
                // could only ever be deletion.
                channels.contains(&Channel::System),
                Arc::clone(&shared),
                Arc::clone(&frames),
                signal_tx,
            )?,
        ];

        Ok((
            Self {
                meeting_id: cfg.meeting_id.clone(),
                shared,
                clock,
                signals: Mutex::new(Some(signal_rx)),
                threads: Mutex::new(threads),
            },
            CaptureStarted {
                channels,
                system_audio_error,
                microphone_error,
                speech_detection_degraded: !detector_available,
            },
        ))
    }

    /// The stream of things capture wants to tell the session layer. Available
    /// once; the session layer owns it from then on.
    ///
    /// Unbounded on purpose: the sending side runs on the capture threads, and a
    /// bounded send that could ever block there would put the recording at the
    /// mercy of the receiver. Every message is small and the session layer drains
    /// it continuously; if it ever stops, the chunks are already fsynced on disk.
    pub fn take_signals(&self) -> Option<UnboundedReceiver<CaptureSignal>> {
        self.signals
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
    }

    /// Alias for [`CaptureSession::take_signals`], under the name the session
    /// port asks for.
    pub fn subscribe(&self) -> Option<UnboundedReceiver<CaptureSignal>> {
        self.take_signals()
    }

    pub fn meeting_id(&self) -> &str {
        &self.meeting_id
    }

    /// Stop the streams, flush and commit every open chunk, then release the
    /// devices (mantra 1). Returns the duration on the meeting clock.
    pub async fn stop(self) -> Result<i64, AudioError> {
        let shared = Arc::clone(&self.shared);
        let clock = Arc::clone(&self.clock);
        let threads: Vec<_> =
            std::mem::take(&mut *self.threads.lock().unwrap_or_else(|e| e.into_inner()));
        tokio::task::spawn_blocking(move || {
            shared.stopping.store(true, Ordering::Relaxed);
            for thread in threads {
                let _ = thread.join();
            }
            let clock = clock.lock().unwrap_or_else(|e| e.into_inner());
            clock.audio_ms()
        })
        .await
        .map_err(|e| AudioError::Backend(e.to_string()))
    }

    /// Stop consuming audio but keep the devices open and the clock running,
    /// so resuming is instant. Idempotent.
    ///
    /// The devices stay open on purpose: closing and reopening them mid-meeting
    /// is the slowest and most failure-prone thing capture can do. What gets
    /// written while paused is silence, so the timeline keeps its shape and a
    /// paused stretch reads as exactly that.
    pub async fn pause(&self) -> Result<(), AudioError> {
        if !self.shared.paused.swap(true, Ordering::Relaxed) {
            self.clock.lock().unwrap_or_else(|e| e.into_inner()).pause();
        }
        Ok(())
    }

    /// Idempotent counterpart of [`CaptureSession::pause`].
    pub async fn resume(&self) -> Result<(), AudioError> {
        if self.shared.paused.swap(false, Ordering::Relaxed) {
            self.clock
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .resume();
        }
        Ok(())
    }

    pub fn is_paused(&self) -> bool {
        self.shared.paused.load(Ordering::Relaxed)
    }

    /// Current offset on the meeting clock.
    pub fn elapsed_ms(&self) -> i64 {
        self.shared.elapsed_ms.load(Ordering::Relaxed)
    }

    /// Smoothed 0.0..=1.0 loudness per channel for the recording indicator.
    pub fn levels(&self) -> (f32, f32) {
        (
            SessionShared::load_level(&self.shared.mic_level),
            SessionShared::load_level(&self.shared.system_level),
        )
    }

    /// Channels producing audio right now, shrinks when a device is lost.
    pub fn active_channels(&self) -> Vec<Channel> {
        let mut out = Vec::new();
        if self.shared.mic_active.load(Ordering::Relaxed) {
            out.push(Channel::Mic);
        }
        if self.shared.system_active.load(Ordering::Relaxed) {
            out.push(Channel::System);
        }
        out
    }

    /// How much work capture is holding back from the speech engine: none.
    ///
    /// Capture hands every utterance straight out through the signal channel as
    /// soon as it is found, so nothing queues up here. There *was* a queue in
    /// this layer — pushed on every utterance and popped by nobody (review
    /// finding 6). Its depth only ever went up, so it reached its capacity in
    /// the first busy minute of any meeting and reported "the transcript is
    /// catching up" for the rest of it, whatever the real state of the pipeline.
    ///
    /// The queue that can genuinely fall behind is the session pipeline's, and
    /// its depth is what the status reads now.
    pub fn pending_utterances(&self) -> u32 {
        0
    }
}

impl Drop for CaptureSession {
    fn drop(&mut self) {
        self.shared.stopping.store(true, Ordering::Relaxed);
        let threads: Vec<_> =
            std::mem::take(&mut *self.threads.lock().unwrap_or_else(|e| e.into_inner()));
        for thread in threads {
            let _ = thread.join();
        }
    }
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
async fn open_system_capture(origin: Instant) -> Result<system::SystemCapture, AudioError> {
    system::SystemCapture::open_at(origin).await
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
async fn open_system_capture(_origin: Instant) -> Result<SystemCaptureStub, AudioError> {
    Err(AudioError::SystemAudioUnsupported(
        "this system cannot share what the computer plays".into(),
    ))
}

/// Stand-in so the pump compiles on a platform with no system-audio backend.
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
#[derive(Debug)]
pub struct SystemCaptureStub;

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
impl SystemCaptureStub {
    pub fn drain(&mut self) -> Vec<Frame> {
        Vec::new()
    }
    pub fn flush(&mut self) -> Vec<Frame> {
        Vec::new()
    }
    pub fn is_running(&self) -> bool {
        false
    }
    pub fn stopped_reason(&self) -> Option<String> {
        None
    }
    pub fn diagnostics(&self) -> String {
        "no system-audio backend on this platform".to_string()
    }
    pub fn reopen(&mut self) -> Result<(), AudioError> {
        Err(AudioError::NotImplemented)
    }
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
type SystemHandle = system::SystemCapture;
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
type SystemHandle = SystemCaptureStub;

/// Turn a backend failure into something a person can act on.
fn describe_system_error(error: &AudioError) -> String {
    match error {
        AudioError::PermissionDenied => {
            "Echo needs permission to hear what your computer plays. Open System Settings to allow it, then start again."
                .to_string()
        }
        AudioError::SystemAudioUnsupported(_) => {
            "Echo could not hear what your computer plays, so it is recording your microphone only."
                .to_string()
        }
        _ => {
            "Echo could not hear what your computer plays, so it is recording your microphone only."
                .to_string()
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_pump(
    cfg: CaptureConfig,
    mut mic: Option<mic::MicCapture>,
    mut system_capture: Option<SystemHandle>,
    shared: Arc<SessionShared>,
    clock: Arc<Mutex<MeetingClock>>,
    frames: Arc<FrameQueue>,
    signals: UnboundedSender<CaptureSignal>,
) -> Result<std::thread::JoinHandle<()>, AudioError> {
    let audio_dir = cfg.audio_dir.clone();
    std::thread::Builder::new()
        .name("echo-capture-pump".into())
        .spawn(move || {
            let mut mic_writer = match ChunkWriter::create(&audio_dir, Channel::Mic, TARGET_SAMPLE_RATE) {
                Ok(w) => Some(w),
                Err(e) => {
                    let _ = signals.send(CaptureSignal::StorageFailed {
                        message: e.to_string(),
                    });
                    None
                }
            };
            let mut system_writer = if system_capture.is_some() {
                ChunkWriter::create(&audio_dir, Channel::System, TARGET_SAMPLE_RATE).ok()
            } else {
                None
            };

            let mut mic_level = 0.0f32;
            let mut system_level = 0.0f32;
            let mut mic_reported_lost = false;
            let mut system_reported_lost = false;
            let mut next_reopen: Option<Instant> = None;
            let mut last_levels = Instant::now() - LEVELS_INTERVAL;
            // Whether this recording has a microphone in it at all decides two
            // things below: what proves the recording is running, and which
            // sentence is true when the computer's audio goes quiet. Read once,
            // because `mic` is taken at the end of the loop.
            let has_microphone = mic.is_some();
            // Only watched when the computer's audio actually opened: a channel
            // that never started is already reported by `system_audio_error`.
            let mut system_silence = system_capture.is_some().then(|| {
                SilenceWatchdog::new(
                    Instant::now(),
                    SILENT_CHANNEL_GRACE,
                    if has_microphone {
                        ProofOfLife::MicrophoneFrames
                    } else {
                        ProofOfLife::MeetingClock
                    },
                )
            });

            loop {
                let stopping = shared.stopping.load(Ordering::Relaxed);
                let paused = shared.paused.load(Ordering::Relaxed);

                let mut batch: Vec<Frame> = Vec::new();
                if let Some(m) = mic.as_mut() {
                    // Drain first even when stopping: the ring may still hold
                    // audio the device handed over a moment ago.
                    batch.extend(m.drain());
                    if stopping {
                        batch.extend(m.flush());
                    }
                }
                if let Some(s) = system_capture.as_mut() {
                    batch.extend(s.drain());
                    if stopping {
                        batch.extend(s.flush());
                    }
                }

                if let Some(watchdog) = system_silence.as_mut() {
                    let mut system_frames = 0u64;
                    let mut mic_frames = 0u64;
                    for frame in &batch {
                        match frame.channel {
                            Channel::System => system_frames += 1,
                            _ => mic_frames += 1,
                        }
                    }
                    // `audio_ms`, not `now_ms`: the second reading is what the
                    // watchdog leans on when there is no microphone, and a
                    // paused recording must not look like a running one.
                    let recorded_ms = clock
                        .lock()
                        .map(|c| c.audio_ms())
                        .unwrap_or_else(|e| e.into_inner().audio_ms());
                    watchdog.observe(system_frames, mic_frames, recorded_ms);
                }

                for mut frame in batch {
                    if paused {
                        // Keep the shape of the timeline; write silence rather
                        // than the audio nobody asked us to record.
                        frame.samples.iter_mut().for_each(|s| *s = 0.0);
                    }
                    let peak = frame.peak();
                    let (writer, level) = match frame.channel {
                        Channel::System => (system_writer.as_mut(), &mut system_level),
                        _ => (mic_writer.as_mut(), &mut mic_level),
                    };
                    // Fast attack, slow release: the indicator should jump when
                    // someone speaks and settle gently.
                    *level = if peak > *level {
                        peak
                    } else {
                        *level * 0.85 + peak * 0.15
                    };

                    if let Some(writer) = writer {
                        // `write_frame`, not `write`: the writer places the frame
                        // where its timestamp says it belongs, filling anything
                        // lost upstream, and reports every chunk that became
                        // durable so none goes unjournalled.
                        match writer.write_frame(&frame) {
                            Ok(chunks) => {
                                for chunk in chunks {
                                    let _ = signals.send(CaptureSignal::ChunkCommitted(chunk));
                                }
                            }
                            Err(e) => {
                                let message = e.to_string();
                                tracing::error!(target: "echo::audio", "writing audio failed: {message}");
                                let _ = signals
                                    .send(CaptureSignal::StorageFailed { message });
                                // Stop touching this channel; committed chunks
                                // stay exactly as they are.
                                match frame.channel {
                                    Channel::System => system_writer = None,
                                    _ => mic_writer = None,
                                }
                                break;
                            }
                        }
                    }
                    frames.push(frame);
                }

                SessionShared::store_level(&shared.mic_level, mic_level);
                SessionShared::store_level(&shared.system_level, system_level);
                let elapsed = clock
                    .lock()
                    .map(|c| c.now_ms())
                    .unwrap_or_else(|e| e.into_inner().now_ms());
                shared.elapsed_ms.store(elapsed, Ordering::Relaxed);
                if last_levels.elapsed() >= LEVELS_INTERVAL {
                    last_levels = Instant::now();
                    let _ = signals.send(CaptureSignal::Levels {
                        mic: mic_level,
                        system: system_level,
                        t_ms: elapsed,
                    });
                }

                if stopping {
                    break;
                }

                // Has a source gone away?
                if let Some(m) = mic.as_ref() {
                    if !m.is_running() && !mic_reported_lost {
                        mic_reported_lost = true;
                        shared.mic_active.store(false, Ordering::Relaxed);
                        let _ = signals.send(CaptureSignal::Degraded {
                            channel: Some(Channel::Mic),
                            reason: DegradedReason::MicrophoneUnavailable,
                            message: "Echo stopped hearing your microphone. Everything up to now is saved."
                                .to_string(),
                        });
                    }
                }
                // Started, still claims to be running, and has never produced a
                // frame. The stream-stopped path below owns the banner once the
                // stream admits it is gone; this one is for the case where it
                // never will.
                if let (Some(watchdog), Some(s)) =
                    (system_silence.as_mut(), system_capture.as_ref())
                {
                    if s.is_running() && !system_reported_lost {
                        match watchdog.poll(Instant::now()) {
                            Some(SilenceVerdict::WentSilent) => {
                                shared.system_active.store(false, Ordering::Relaxed);
                                tracing::warn!(
                                    target: "echo::audio",
                                    system_frames = watchdog.watched_frames(),
                                    mic_frames = watchdog.reference_frames(),
                                    proof = ?watchdog.proof(),
                                    has_microphone,
                                    detail = %s.diagnostics(),
                                    "the computer's audio started but has delivered nothing"
                                );
                                let (reason, message) = if has_microphone {
                                    (
                                        DegradedReason::SystemAudioUnavailable,
                                        SYSTEM_AUDIO_SILENT_MESSAGE,
                                    )
                                } else {
                                    // "Recording through the microphone only"
                                    // is not true of a recording that has no
                                    // microphone; nothing at all is arriving.
                                    (
                                        DegradedReason::NothingIsBeingHeard,
                                        NOTHING_IS_BEING_HEARD_MESSAGE,
                                    )
                                };
                                let _ = signals.send(CaptureSignal::Degraded {
                                    channel: Some(Channel::System),
                                    reason,
                                    message: message.to_string(),
                                });
                            }
                            Some(SilenceVerdict::CameBack) => {
                                shared.system_active.store(true, Ordering::Relaxed);
                                tracing::info!(
                                    target: "echo::audio",
                                    system_frames = watchdog.watched_frames(),
                                    "the computer's audio started arriving after all"
                                );
                                let _ = signals.send(CaptureSignal::Recovered {
                                    channel: Channel::System,
                                });
                            }
                            None => {}
                        }
                    }
                }

                if let Some(s) = system_capture.as_mut() {
                    if !s.is_running() {
                        if !system_reported_lost {
                            system_reported_lost = true;
                            shared.system_active.store(false, Ordering::Relaxed);
                            tracing::warn!(
                                target: "echo::audio",
                                "system audio stopped: {:?}",
                                s.stopped_reason()
                            );
                            let (reason, message) = if has_microphone {
                                (
                                    // Not `SystemAudioUnavailable`: the banner
                                    // that reason draws promises a separation
                                    // of the microphone recording that a
                                    // meeting with any system audio in it never
                                    // gets, and calls lines "You" that already
                                    // carry a real name.
                                    DegradedReason::SystemAudioLost,
                                    SYSTEM_AUDIO_LOST_MESSAGE,
                                )
                            } else {
                                // And not `SystemAudioLost` either, for the same
                                // reason the silent case above takes its own
                                // branch: that sentence ends "It's still
                                // recording through the microphone", and this
                                // recording has no microphone to still be
                                // recording through. It has nothing left.
                                (
                                    DegradedReason::NothingIsBeingHeard,
                                    NOTHING_IS_BEING_HEARD_MESSAGE,
                                )
                            };
                            let _ = signals.send(CaptureSignal::Degraded {
                                channel: Some(Channel::System),
                                reason,
                                message: message.to_string(),
                            });
                            next_reopen = Some(Instant::now() + REOPEN_BACKOFF);
                        }
                        // A display change or a Bluetooth switch takes the
                        // stream with it; try to get it back without disturbing
                        // the recording.
                        if next_reopen.is_some_and(|at| Instant::now() >= at) {
                            match s.reopen() {
                                Ok(()) => {
                                    system_reported_lost = false;
                                    next_reopen = None;
                                    shared.system_active.store(true, Ordering::Relaxed);
                                    let _ = signals.send(CaptureSignal::Recovered {
                                        channel: Channel::System,
                                    });
                                }
                                Err(e) => {
                                    tracing::debug!(
                                        target: "echo::audio",
                                        "could not get system audio back yet: {e}"
                                    );
                                    next_reopen = Some(Instant::now() + REOPEN_BACKOFF);
                                }
                            }
                        }
                    }
                }

                std::thread::sleep(PUMP_INTERVAL);
            }

            // Commit whatever is open before the devices go.
            if let Some(writer) = mic_writer.take() {
                match writer.finish() {
                    Ok(Some(chunk)) => {
                        let _ = signals.send(CaptureSignal::ChunkCommitted(chunk));
                    }
                    Ok(None) => {}
                    Err(e) => {
                        let _ = signals.send(CaptureSignal::StorageFailed {
                            message: e.to_string(),
                        });
                    }
                }
            }
            if let Some(writer) = system_writer.take() {
                if let Ok(Some(chunk)) = writer.finish() {
                    let _ = signals.send(CaptureSignal::ChunkCommitted(chunk));
                }
            }
            if let Some(m) = mic.take() {
                m.close();
            }
            drop(system_capture);
            // Only now can speech detection know nothing more is coming.
            shared.pump_finished.store(true, Ordering::Relaxed);
            if frames.dropped() > 0 {
                tracing::info!(
                    target: "echo::audio",
                    "{} frames never reached live speech detection; the recording is complete on disk",
                    frames.dropped()
                );
            }
        })
        .map_err(|e| AudioError::Backend(e.to_string()))
}

fn spawn_speech(
    detector_path: Option<PathBuf>,
    log_dir: Option<PathBuf>,
    has_system_channel: bool,
    shared: Arc<SessionShared>,
    frames: Arc<FrameQueue>,
    signals: UnboundedSender<CaptureSignal>,
) -> Result<std::thread::JoinHandle<()>, AudioError> {
    std::thread::Builder::new()
        .name("echo-speech-detect".into())
        .spawn(move || {
            let mut mic_detector =
                SpeechDetector::load_or_fallback(detector_path.as_deref(), Channel::Mic);
            let mut system_detector =
                SpeechDetector::load_or_fallback(detector_path.as_deref(), Channel::System);
            // The detectors and the bleed guard are built together because they
            // are the same kind of thing: one recording's worth of state, on
            // the one thread that sees every frame and every utterance.
            let mut guard = BleedGuard::new(log_dir.as_deref(), has_system_channel);
            // …and every utterance leaves through here, a tick or two after the
            // detector closed it, because the far side it is judged against is
            // audio from after its own end. See [`FAR_SIDE_WAIT`].
            let mut waiting = WaitingRoom::default();
            let mut batch: Vec<Frame> = Vec::new();
            let mut was_paused = false;
            // Set on resume, cleared per channel by the first frame that
            // arrives after it — which may be several ticks later on a channel
            // that is quiet.
            let mut restart_mic = false;
            let mut restart_system = false;
            // Per channel: which open stretch was last snapshotted, and how far
            // into it that snapshot reached. Both halves matter — a new stretch
            // is a new line and goes out at once, where more of the same
            // stretch waits for [`SNAPSHOT_OFFER_MS`] of new speech.
            let mut offered_mic: Option<(i64, i64)> = None;
            let mut offered_system: Option<(i64, i64)> = None;

            loop {
                // Wait for the pump to say it is done, not just for the stop
                // request: the last frames of a meeting are the ones a person
                // most notices missing from the live transcript.
                let finished = shared.pump_finished.load(Ordering::Relaxed);
                // A paused stretch is written as silence, so the timestamps stay
                // continuous and nothing here notices the gap — but the audio
                // either side of it is not adjacent, and both detectors are
                // carrying state that says it is (review finding 10). Resuming
                // starts them again, from the first frame that arrives after.
                let paused = shared.paused.load(Ordering::Relaxed);
                if was_paused && !paused {
                    restart_mic = true;
                    restart_system = true;
                    // …and the delay goes with them. The audio either side of a
                    // pause is not adjacent, so a delay measured before it is
                    // not a delay to search around after it.
                    guard.forget_lag();
                    tracing::debug!(
                        target: "echo::audio",
                        "recording resumed; speech detection starts again"
                    );
                }
                was_paused = paused;

                // The loudspeaker gate, re-read here and **nowhere else**.
                // `poll_route` does its own two-second throttling; what this
                // call site decides is which thread pays for it. A CoreAudio
                // property read can stall, and the pump is three ring-seconds
                // from losing audio, so it may never be the one to block — the
                // speech thread falling behind costs live text, which is what
                // gives way (mantra 3). `system_active` is passed live because
                // suppression is only deduplication while the far side is
                // actually being written down somewhere else.
                guard.poll_route(
                    shared.elapsed_ms.load(Ordering::Relaxed),
                    shared.system_active.load(Ordering::Relaxed),
                );

                batch.clear();
                frames.drain(&mut batch);
                // **Pass one: the computer's audio into the ring, before either
                // detector runs.**
                //
                // `spawn_pump` builds each tick's batch by draining the
                // microphone first and the system stream second, so a single
                // drain here would push every mic frame of this tick — and any
                // utterance closing on them — through the judge while the ring
                // still ended one tick back. A mic utterance is judged against
                // the far side up to 200 ms *after* its own end (`LAG_MIN_MS`),
                // so a ring twenty milliseconds stale is a coverage refusal on
                // every utterance that closes at the head of the recording.
                // (The other 200 ms of that tail is audio from the future when
                // a stretch closes, which is what the [`WaitingRoom`] is for.)
                //
                // Safe to do first because it touches no detector state at all:
                // the ring is placement by timestamp and nothing else, and the
                // frames are then detected below in exactly the order and with
                // exactly the effects they had before.
                for frame in &batch {
                    if frame.channel == Channel::System {
                        guard.append_system(frame.t_start_ms, &frame.samples);
                    }
                }
                // Pass two: unchanged.
                for frame in batch.drain(..) {
                    let (detector, restart) = match frame.channel {
                        Channel::System => (&mut system_detector, &mut restart_system),
                        _ => (&mut mic_detector, &mut restart_mic),
                    };
                    if std::mem::take(restart) {
                        if let Some(utterance) = detector.reset_at(frame.t_start_ms) {
                            waiting.hold(Instant::now(), utterance);
                        }
                    }
                    for utterance in detector.push(&frame.samples, frame.t_start_ms) {
                        waiting.hold(Instant::now(), utterance);
                    }
                }
                // Pass three: everything whose far side has arrived. This tick's
                // system frames went into the ring above, so a stretch held over
                // from an earlier tick is judged against a ring that has caught
                // up with it.
                waiting.release(&mut guard, &signals, Instant::now());

                // Words on screen while someone is still talking. Only ever a
                // look at what is open: the utterance itself still arrives when
                // the speech ends, and it is the one that gets written down.
                if !paused {
                    for (detector, offered) in [
                        (&mic_detector, &mut offered_mic),
                        (&system_detector, &mut offered_system),
                    ] {
                        let Some(snapshot) = detector.snapshot() else {
                            // Nothing open: the next stretch starts fresh.
                            *offered = None;
                            continue;
                        };
                        let end = snapshot.window_end_ms();
                        // A new stretch is offered straight away, however
                        // little of it there is: it is a new line on screen,
                        // and holding it back is a second of someone talking to
                        // a view that shows nothing. Within one stretch, only
                        // once the speech has actually moved on.
                        let due = match *offered {
                            Some((start, last)) if start == snapshot.t_start_ms => {
                                end - last >= SNAPSHOT_OFFER_MS
                            }
                            _ => true,
                        };
                        if !due {
                            continue;
                        }
                        *offered = Some((snapshot.t_start_ms, end));
                        let _ = signals.send(CaptureSignal::SpeechSoFar(snapshot));
                    }
                }

                if finished {
                    for detector in [&mut mic_detector, &mut system_detector] {
                        if let Some(utterance) = detector.finish() {
                            waiting.hold(Instant::now(), utterance);
                        }
                    }
                    // Nothing more is coming — not another frame of the far
                    // side, and no later tick to release on. Whatever is still
                    // waiting is judged against the whole recording as the ring
                    // holds it, and the last words of the meeting go out now
                    // rather than half a second later.
                    waiting.flush(&mut guard, &signals);
                    break;
                }
                std::thread::sleep(PUMP_INTERVAL * 2);
            }

            report_bleed(&guard.report());
        })
        .map_err(|e| AudioError::Backend(e.to_string()))
}

/// The one door out of speech detection, and the only place an utterance is
/// ever judged.
///
/// There are three places the speech thread takes an utterance from a detector
/// — the ordinary push, the stretch a resume rebases and closes, and the final
/// flush when the pump says nothing more is coming — and a copy of the far side
/// escaping through any one of them is a duplicated line in somebody's
/// transcript. So none of them sends: they all hand it to the
/// [`WaitingRoom`], which hands it here, and there is exactly one
/// `UtteranceReady` in the file to keep honest.
///
/// Everything that is not a measured copy goes on untouched, [`Verdict::Undecided`]
/// included. An unanswered question is not evidence.
fn send_utterance(
    guard: &mut BleedGuard,
    signals: &UnboundedSender<CaptureSignal>,
    utterance: Utterance,
) {
    match guard.judge(&utterance) {
        Verdict::Bleed(evidence) => {
            tracing::debug!(
                target: "echo::audio",
                channel = ?utterance.channel,
                t_start_ms = utterance.t_start_ms,
                t_end_ms = utterance.t_end_ms,
                correlation = evidence.correlation as f64,
                lag_ms = evidence.lag_ms,
                system_voice_ms = evidence.system_voice_ms,
                unexplained_ms = evidence.unexplained_ms,
                span_ms = evidence.span_ms,
                "this stretch of the microphone is what this computer played coming back; \
                 the transcript is getting these words from the computer's own side"
            );
            let _ = signals.send(CaptureSignal::UtteranceSuppressed {
                channel: utterance.channel,
                t_start_ms: utterance.t_start_ms,
                t_end_ms: utterance.t_end_ms,
                evidence,
            });
        }
        Verdict::Undecided(_) | Verdict::Pass => {
            let _ = signals.send(CaptureSignal::UtteranceReady(utterance));
        }
    }
}

/// Longest a stretch waits for the far side it will be judged against.
///
/// The wait is normally two or three ticks of this thread, and it is not
/// optional: a verdict needs the computer's audio up to `LAG_MIN_MS` — 200 ms —
/// *after* the stretch ends, and that audio has not been captured yet at the
/// moment the stretch closes. The speech detector closes a stretch 416 ms after
/// the last voiced window and dates it back to 288 ms after it
/// (`vad::VadSettings::mic`), so about 128 ms of that tail is on hand and the
/// other 72 ms is still in the future. Judging there is not a near miss, it is
/// a structural one: every microphone utterance refused for coverage, for ever,
/// on a ring in perfect health.
///
/// This is the cap for when the rest never arrives — the system stream died and
/// the two-second route poll has not noticed, or the pump is starved. Half a
/// second is several times the honest wait and well under the pause a person
/// would read as the live text having stopped; past it the stretch goes out
/// unjudged, which is the same thing an unanswered question has always meant
/// here.
const FAR_SIDE_WAIT: Duration = Duration::from_millis(500);

/// Utterances on their way out of speech detection, waiting for the far side
/// they will be judged against.
///
/// **One queue for both channels, released strictly from the front.** A
/// microphone stretch waits and a system stretch never does, so releasing each
/// as soon as it is ready would let the far side's own utterances overtake the
/// microphone's by a tick or two — reordering the one sequence the pipeline
/// downstream reads in order. Making the system channel wait behind a
/// microphone stretch that is nearly ready costs it those same few ticks and
/// keeps the order the detectors produced.
#[derive(Debug, Default)]
struct WaitingRoom {
    queue: VecDeque<(Instant, Utterance)>,
}

impl WaitingRoom {
    /// Take one utterance from a detector. Nothing is sent here: releasing is
    /// the loop's business, once this tick's far side is in the ring.
    fn hold(&mut self, now: Instant, utterance: Utterance) {
        self.queue.push_back((now + FAR_SIDE_WAIT, utterance));
    }

    /// Send on everything at the front that can be judged now, or has waited
    /// long enough that nothing more is coming.
    fn release(
        &mut self,
        guard: &mut BleedGuard,
        signals: &UnboundedSender<CaptureSignal>,
        now: Instant,
    ) {
        while self.queue.front().is_some_and(|(deadline, utterance)| {
            now >= *deadline || far_side_on_hand(guard, utterance)
        }) {
            let (_, utterance) = self.queue.pop_front().expect("just looked at the front");
            send_utterance(guard, signals, utterance);
        }
    }

    /// The recording is over: judge what is left against whatever is on hand.
    /// There is no more far side coming, so waiting for it would only hold back
    /// the last words of a meeting.
    fn flush(&mut self, guard: &mut BleedGuard, signals: &UnboundedSender<CaptureSignal>) {
        for (_, utterance) in self.queue.drain(..) {
            send_utterance(guard, signals, utterance);
        }
    }
}

/// Is the far side this stretch would be judged against on hand yet?
///
/// Three ways the answer is yes without looking at the ring at all: the system
/// channel's own utterances are the original and are never judged, an unarmed
/// guard judges nothing, and a guard whose ring is empty is a guard with
/// nothing to wait for. Otherwise the ring has to reach `LAG_MIN_MS` past the
/// end of the stretch, which is exactly what
/// [`crate::audio::bleed::covers_every_lag`] will be asked about a moment
/// later — the same geometry, asked before the measurement rather than after
/// it, so a stretch is held instead of being refused.
fn far_side_on_hand(guard: &BleedGuard, utterance: &Utterance) -> bool {
    if utterance.channel != Channel::Mic || !guard.armed() {
        return true;
    }
    guard
        .system_through_ms()
        .is_some_and(|through_ms| through_ms >= utterance.t_end_ms - LAG_MIN_MS)
}

/// Stretches examined before "and never found one" is worth saying out loud.
///
/// Under this many, a meeting that suppressed nothing is just a meeting where
/// nothing was played out loud, and there is nothing to report.
const ENOUGH_EXAMINED: u64 = 20;

/// …and of those, how many had the far side genuinely audible under them.
///
/// The number that turns "found nothing" into evidence. Ten is what
/// [`crate::asr::catchup_bleed::ENOUGH_MISSED_CHANCES`] uses to decide the same
/// thing about the same meeting from the other end, and for the same reason: on
/// the 2026-08-25 recording every one of the twelve system-channel segments
/// overlapped a mic segment, so a machine that is going to find copies finds
/// them early.
const ENOUGH_CHANCES: u64 = 10;

/// Did this recording look hard for the microphone's copy and never once find
/// one?
///
/// **This is the only thing that makes a machine whose true delay is outside
/// the search window visible.** Such a machine fails silent and safe — nothing
/// is suppressed, the transcript is exactly what it would have been — which is
/// the right way to fail and an impossible way to notice. All three conditions
/// are needed: an unarmed guard was not looking, a handful of stretches is not
/// a sample, and stretches with no far side under them were never a chance to
/// match.
fn never_found_a_copy(report: &bleed_guard::BleedReport) -> bool {
    report.armed
        && report.suppressed == 0
        && report.examined >= ENOUGH_EXAMINED
        && report.opportunities >= ENOUGH_CHANCES
}

/// Was this recording asked over and over and never once *able* to answer?
///
/// The sibling of [`never_found_a_copy`], for the other silent failure: not a
/// delay outside the search window, but the far side for those seconds never
/// being on hand at all — a speech thread far behind the ring, a system stream
/// delivering nothing while it still claims to be live, or a stretch of wiring
/// that asks before the audio it needs exists (which is what [`FAR_SIDE_WAIT`]
/// is there to stop). Every one of those fails safe, and every one of them
/// leaves `examined` at zero, which is precisely where `never_found_a_copy`
/// stops looking: its sample size condition can never be met, so without this
/// the whole feature could be inert for a whole meeting and say nothing.
fn never_had_the_far_side(report: &bleed_guard::BleedReport) -> bool {
    report.armed && report.examined == 0 && report.undecided >= ENOUGH_EXAMINED
}

/// One line per recording about what bleed detection did, and a second one when
/// what it did was nothing.
fn report_bleed(report: &bleed_guard::BleedReport) {
    tracing::info!(
        target: "echo::audio",
        armed = report.armed,
        route = report.route.as_str(),
        examined = report.examined,
        suppressed = report.suppressed,
        too_short = report.too_short,
        undecided = report.undecided,
        opportunities = report.opportunities,
        lag_ms = ?report.lag_ms,
        held_ms = report.held_ms,
        dropped_backwards = report.dropped_backwards,
        "listening for the microphone's copy of what this computer plays is finished"
    );
    if never_found_a_copy(report) {
        tracing::info!(
            target: "echo::audio",
            examined = report.examined,
            chances = report.opportunities,
            "looked for the microphone's copy of what the computer played and never found it"
        );
    }
    if never_had_the_far_side(report) {
        tracing::warn!(
            target: "echo::audio",
            undecided = report.undecided,
            held_ms = report.held_ms,
            "the computer's audio for those seconds was never on hand, so not one stretch \
             of the microphone could be compared against it"
        );
    }
}

// ---------------------------------------------------------------------------
// Devices, permissions, and reading audio back
// ---------------------------------------------------------------------------

/// Input devices for Settings → General. Cheap: no stream is opened.
pub fn list_input_devices() -> Result<Vec<AudioDevice>, AudioError> {
    mic::list_devices()
}

/// Can this build capture what the computer plays at all?
/// macOS 13+ with ScreenCaptureKit, or Linux with PipeWire running.
pub fn system_audio_supported() -> bool {
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    {
        system::supported()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        false
    }
}

/// Current microphone permission, without prompting.
pub async fn microphone_permission() -> PermissionState {
    #[cfg(target_os = "macos")]
    {
        system::microphone_permission_state()
    }
    #[cfg(target_os = "linux")]
    {
        // Nothing gates the microphone at the OS level here; being able to see a
        // device is the only meaningful answer.
        if mic::any_input_device() {
            PermissionState::Granted
        } else {
            PermissionState::NotApplicable
        }
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        PermissionState::Unknown
    }
}

/// Prompt for microphone access. Returns the state after the person answers;
/// `RestartRequired` when the platform needs a relaunch to take effect.
pub async fn request_microphone_permission() -> PermissionState {
    #[cfg(target_os = "macos")]
    {
        tokio::task::spawn_blocking(system::request_microphone_permission_blocking)
            .await
            .unwrap_or(PermissionState::Unknown)
    }
    #[cfg(not(target_os = "macos"))]
    {
        microphone_permission().await
    }
}

/// Current permission for capturing what the computer plays.
pub async fn system_audio_permission() -> PermissionState {
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    {
        system::permission().await
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        PermissionState::NotApplicable
    }
}

/// Ask for permission to capture what the computer plays. On macOS this is
/// Screen & System Audio Recording and usually needs a relaunch.
pub async fn request_system_audio_permission() -> PermissionState {
    #[cfg(target_os = "macos")]
    {
        system::request_permission().await
    }
    #[cfg(not(target_os = "macos"))]
    {
        system_audio_permission().await
    }
}

/// Open the right system settings pane, for the "Open System Settings" button.
pub fn open_privacy_settings(target: PermissionTarget) -> Result<(), AudioError> {
    #[cfg(target_os = "macos")]
    {
        let url = match target {
            PermissionTarget::Microphone => {
                "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"
            }
            PermissionTarget::SystemAudio => {
                return system::open_settings_pane();
            }
            PermissionTarget::Notifications => {
                "x-apple.systempreferences:com.apple.preference.notifications"
            }
        };
        std::process::Command::new("/usr/bin/open")
            .arg(url)
            .status()
            .map_err(|e| AudioError::Backend(e.to_string()))?;
        Ok(())
    }
    #[cfg(target_os = "linux")]
    {
        let _ = target;
        system::open_settings_pane()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = target;
        Err(AudioError::NotImplemented)
    }
}

/// One chunk on disk, with where it belongs on the meeting clock.
///
/// Readers need the offset, not just the file: chunks can have honest gaps
/// between them (a device unplugged mid-meeting), and a chunk that will not
/// decode has to become silence *of the right length* rather than shortening
/// everything after it. Concatenating files and hoping is how catch-up text,
/// speaker turns and click-to-play end up minutes out of step with the audio.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkRef {
    pub path: PathBuf,
    pub channel: Channel,
    pub t_start_ms: i64,
    pub t_end_ms: i64,
}

impl ChunkRef {
    pub fn new(path: impl Into<PathBuf>, channel: Channel, t_start_ms: i64, t_end_ms: i64) -> Self {
        Self {
            path: path.into(),
            channel,
            t_start_ms: t_start_ms.max(0),
            t_end_ms: t_end_ms.max(0),
        }
    }

    /// From a journal row, which is the authority on placement.
    pub fn from_journal(chunk: &crate::types::AudioChunk) -> Self {
        Self::new(&chunk.path, chunk.channel, chunk.t_start_ms, chunk.t_end_ms)
    }

    /// From a chunk that has just been closed.
    pub fn from_committed(chunk: &CommittedChunk) -> Self {
        Self::new(&chunk.path, chunk.channel, chunk.t_start_ms, chunk.t_end_ms)
    }

    pub fn duration_ms(&self) -> i64 {
        (self.t_end_ms - self.t_start_ms).max(0)
    }
}

/// One channel's chunks, in time order.
#[derive(Debug, Clone)]
struct ChannelChunks {
    channel: Channel,
    chunks: Vec<ChunkRef>,
}

/// Group chunks per channel, earliest first.
fn group_by_channel(chunks: &[ChunkRef]) -> Vec<ChannelChunks> {
    let mut groups: Vec<ChannelChunks> = Vec::new();
    for chunk in chunks {
        match groups.iter_mut().find(|g| g.channel == chunk.channel) {
            Some(group) => group.chunks.push(chunk.clone()),
            None => groups.push(ChannelChunks {
                channel: chunk.channel,
                chunks: vec![chunk.clone()],
            }),
        }
    }
    for group in &mut groups {
        group.chunks.sort_by_key(|c| c.t_start_ms);
    }
    groups
}

/// Decode one channel into a single 16 kHz mono timeline that starts at t=0 on
/// the meeting clock, with every chunk at its own offset.
///
/// A chunk that cannot be read leaves silence exactly as long as it was, so
/// everything after it stays where it happened.
fn decode_channel(group: &ChannelChunks) -> Vec<f32> {
    let rate = TARGET_SAMPLE_RATE as i64;
    let ms_to_samples = |ms: i64| (ms.max(0) * rate / 1_000) as usize;
    let length = group
        .chunks
        .iter()
        .map(|c| c.t_end_ms)
        .max()
        .unwrap_or(0)
        .max(0);
    let mut timeline = vec![0.0f32; ms_to_samples(length)];

    for chunk in &group.chunks {
        let at = ms_to_samples(chunk.t_start_ms);
        match writer::read_chunk_blocking(&chunk.path) {
            Ok(samples) => {
                if at + samples.len() > timeline.len() {
                    // The file holds slightly more than the journal claimed;
                    // never truncate real audio.
                    timeline.resize(at + samples.len(), 0.0);
                }
                timeline[at..at + samples.len()].copy_from_slice(&samples);
            }
            Err(e) => {
                // A chunk we cannot read is a hole, not the end of the world:
                // the silence already reserved for it keeps the timeline aligned.
                tracing::warn!(
                    target: "echo::audio",
                    "a chunk could not be read, so it reads back as silence ({}): {e}",
                    chunk.path.display()
                );
            }
        }
    }
    timeline
}

/// Build the mixed file used for playback from the per-channel chunks. Derived,
/// so it can be rebuilt or deleted at any time.
pub async fn build_mixdown(chunks: &[ChunkRef], destination: &Path) -> Result<PathBuf, AudioError> {
    let chunks = chunks.to_vec();
    let destination = destination.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let groups: Vec<ChannelChunks> = group_by_channel(&chunks)
            .into_iter()
            .filter(|g| g.channel != Channel::Mixed)
            .collect();
        if groups.is_empty() {
            return Err(AudioError::Write(
                "there is no audio to build a playable file from".into(),
            ));
        }
        let tracks: Vec<Vec<f32>> = groups.iter().map(decode_channel).collect();
        let length = tracks.iter().map(|t| t.len()).max().unwrap_or(0);
        if length == 0 {
            return Err(AudioError::Write(
                "there is no audio to build a playable file from".into(),
            ));
        }
        let mut mixed = vec![0.0f32; length];
        for track in &tracks {
            for (out, sample) in mixed.iter_mut().zip(track.iter()) {
                *out += *sample;
            }
        }
        // Only pull the level down if the sum actually clipped: quiet meetings
        // should not come back quieter than they were.
        let peak = mixed.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        if peak > 1.0 {
            let scale = 1.0 / peak;
            for s in mixed.iter_mut() {
                *s *= scale;
            }
        }
        writer::write_wav_16k_mono(&destination, &mixed)?;
        Ok(destination)
    })
    .await
    .map_err(|e| AudioError::Backend(e.to_string()))?
}

/// The chunks a window could possibly draw a sample from.
///
/// Everything overlapping `[from_ms, to_ms)`, plus one chunk of slack behind it:
/// a chunk's file may hold a little more audio than the journal claimed (see
/// [`decode_channel`], which resizes rather than truncate), and the tail of the
/// chunk before the window is the only place that overrun could reach into it.
fn chunks_touching(chunks: &[ChunkRef], from_ms: i64, to_ms: i64) -> Vec<ChunkRef> {
    let reach_back = from_ms - i64::from(CHUNK_SECONDS) * 1_000;
    chunks
        .iter()
        .filter(|c| c.t_end_ms > reach_back && c.t_start_ms < to_ms)
        .cloned()
        .collect()
}

/// Read a window of a meeting's audio back as 16 kHz mono, how the ASR
/// catch-up pass reads from disk instead of from the live queue.
///
/// `from_ms`/`to_ms` are offsets on the meeting clock, and the chunks carry
/// their own offsets, so the window that comes back is the audio that really
/// happened then — a gap or an unreadable chunk reads as silence in its own
/// place rather than pulling later audio earlier.
///
/// Only the chunks the window actually touches are opened. That sounds obvious
/// and was not true: this used to decode every chunk of the channel on every
/// call, so a pass over a two-hour meeting decoded the whole recording once per
/// window — hundreds of times over — for the thirty seconds it needed. Chunks
/// outside the window contribute nothing to the answer, so skipping them changes
/// no sample of it.
pub async fn read_window(
    chunks: &[ChunkRef],
    from_ms: i64,
    to_ms: i64,
) -> Result<Vec<f32>, AudioError> {
    let chunks = chunks_touching(chunks, from_ms, to_ms);
    tokio::task::spawn_blocking(move || {
        if to_ms <= from_ms {
            return Ok(Vec::new());
        }
        let groups: Vec<ChannelChunks> = group_by_channel(&chunks)
            .into_iter()
            .filter(|g| g.channel != Channel::Mixed)
            .collect();
        let tracks: Vec<Vec<f32>> = groups.iter().map(decode_channel).collect();
        let rate = TARGET_SAMPLE_RATE as i64;
        let start = (from_ms.max(0) * rate / 1_000) as usize;
        let end = (to_ms * rate / 1_000) as usize;
        let wanted = end - start;
        let mut out = vec![0.0f32; wanted];
        let mut sources = 0usize;
        for track in &tracks {
            if track.len() <= start {
                continue;
            }
            sources += 1;
            let slice = &track[start..track.len().min(end)];
            for (o, s) in out.iter_mut().zip(slice.iter()) {
                *o += *s;
            }
        }
        if sources > 1 {
            let scale = 1.0 / sources as f32;
            for s in out.iter_mut() {
                *s *= scale;
            }
        }
        Ok(out)
    })
    .await
    .map_err(|e| AudioError::Backend(e.to_string()))?
}

/// Everything a meeting's folder holds, per channel, for crash recovery.
pub async fn recover_chunks(audio_dir: &Path) -> Result<Vec<CommittedChunk>, AudioError> {
    let mut all = Vec::new();
    for channel in [Channel::Mic, Channel::System] {
        all.extend(writer::recover(audio_dir, channel).await?);
    }
    Ok(all)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_duration_follows_the_sample_count() {
        let f = Frame {
            channel: Channel::Mic,
            t_start_ms: 1_000,
            samples: vec![0.0; TARGET_SAMPLE_RATE as usize / 50], // 20 ms
        };
        assert_eq!(f.duration_ms(), 20);
        assert_eq!(f.t_end_ms(), 1_020);
    }

    #[test]
    fn an_empty_frame_has_no_duration() {
        assert_eq!(Frame::default().duration_ms(), 0);
    }

    #[test]
    fn a_frames_peak_is_its_loudest_sample() {
        let f = Frame {
            samples: vec![0.1, -0.7, 0.3],
            ..Default::default()
        };
        assert!((f.peak() - 0.7).abs() < 1e-6);
        assert_eq!(Frame::default().peak(), 0.0);
    }

    #[test]
    fn the_frame_queue_drops_the_oldest_and_says_how_many() {
        let q = FrameQueue::new(2);
        for i in 0..5 {
            q.push(Frame {
                t_start_ms: i * 20,
                ..Default::default()
            });
        }
        assert_eq!(q.dropped(), 3);
        let mut out = Vec::new();
        q.drain(&mut out);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].t_start_ms, 60);
        assert_eq!(out[1].t_start_ms, 80);
    }

    // The watchdog carries no clock of its own, so these run instantly and
    // deterministically. `origin` stands in for "when the recording started".
    const GRACE: Duration = Duration::from_secs(10);

    #[test]
    fn a_channel_that_started_but_never_delivers_is_reported_once() {
        let origin = Instant::now();
        let mut w = SilenceWatchdog::new(origin, GRACE, ProofOfLife::MicrophoneFrames);

        // The microphone is producing, the computer's audio is not.
        for tick in 1..=9 {
            w.observe(0, 50, 0);
            assert_eq!(
                w.poll(origin + Duration::from_secs(tick)),
                None,
                "gave up after {tick}s, inside the grace period"
            );
        }

        assert_eq!(
            w.poll(origin + GRACE),
            Some(SilenceVerdict::WentSilent),
            "a channel silent for the whole grace period was never reported"
        );
        // And only once, however long it stays silent.
        for tick in 11..=30 {
            w.observe(0, 50, 0);
            assert_eq!(w.poll(origin + Duration::from_secs(tick)), None);
        }
    }

    #[test]
    fn a_channel_delivering_audio_is_never_reported() {
        let origin = Instant::now();
        let mut w = SilenceWatchdog::new(origin, GRACE, ProofOfLife::MicrophoneFrames);
        for tick in 1..=60 {
            w.observe(50, 50, 0);
            assert_eq!(w.poll(origin + Duration::from_secs(tick)), None);
        }
        assert_eq!(w.watched_frames(), 3_000);
    }

    #[test]
    fn a_stopped_clock_is_not_a_silent_channel() {
        // Neither source is producing: the machine slept, the pump never ran, or
        // the whole recording is broken in a way this watchdog must not
        // misdescribe as "we can hear you but not them".
        let origin = Instant::now();
        let mut w = SilenceWatchdog::new(origin, GRACE, ProofOfLife::MicrophoneFrames);
        for tick in 1..=60 {
            w.observe(0, 0, 0);
            assert_eq!(
                w.poll(origin + Duration::from_secs(tick)),
                None,
                "blamed the computer's audio when nothing at all was arriving"
            );
        }
    }

    #[test]
    fn a_channel_that_comes_back_is_reported_recovered_once() {
        let origin = Instant::now();
        let mut w = SilenceWatchdog::new(origin, GRACE, ProofOfLife::MicrophoneFrames);
        w.observe(0, 50, 0);
        assert_eq!(w.poll(origin + GRACE), Some(SilenceVerdict::WentSilent));

        // The first real frame arrives a minute in.
        w.observe(1, 50, 0);
        assert_eq!(
            w.poll(origin + Duration::from_secs(60)),
            Some(SilenceVerdict::CameBack)
        );
        w.observe(50, 50, 0);
        assert_eq!(
            w.poll(origin + Duration::from_secs(61)),
            None,
            "recovery was announced twice"
        );
    }

    #[test]
    fn a_late_first_frame_inside_the_grace_period_is_not_a_failure() {
        let origin = Instant::now();
        let mut w = SilenceWatchdog::new(origin, GRACE, ProofOfLife::MicrophoneFrames);
        w.observe(0, 50, 0);
        assert_eq!(w.poll(origin + Duration::from_secs(9)), None);
        // Nine and a bit seconds late is a slow start, not a broken channel.
        w.observe(1, 50, 0);
        assert_eq!(w.poll(origin + Duration::from_secs(20)), None);
    }

    // The two-channel tests above pass 0 for the meeting clock throughout, which
    // is the point: with a microphone in the recording the clock is never read,
    // and their timings are the ones that were already asserted before this
    // watchdog knew what a meeting clock was.

    #[test]
    fn a_recording_with_no_microphone_still_notices_the_silence() {
        // Screen recording granted, microphone denied. There is no second
        // channel to compare against, so the meeting clock is the proof that
        // the recording is running — and it has to be enough, because this is
        // the exact configuration that recorded three quarters of an hour of
        // nothing without a word being said about it.
        let origin = Instant::now();
        let mut w = SilenceWatchdog::new(origin, GRACE, ProofOfLife::MeetingClock);

        for tick in 1..=9 {
            w.observe(0, 0, tick as i64 * 1_000);
            assert_eq!(
                w.poll(origin + Duration::from_secs(tick)),
                None,
                "gave up after {tick}s, inside the grace period"
            );
        }

        w.observe(0, 0, 10_000);
        assert_eq!(
            w.poll(origin + GRACE),
            Some(SilenceVerdict::WentSilent),
            "a recording with no microphone never found out it was hearing nothing"
        );
        // Once, as ever.
        for tick in 11..=30 {
            w.observe(0, 0, tick as i64 * 1_000);
            assert_eq!(w.poll(origin + Duration::from_secs(tick)), None);
        }
    }

    #[test]
    fn a_paused_recording_with_no_microphone_is_not_blamed_on_the_computer() {
        // Started, then paused a second in to go and find the meeting link. The
        // wall clock runs on; the amount of audio this recording should contain
        // does not, and that is the one the verdict is made from.
        let origin = Instant::now();
        let mut w = SilenceWatchdog::new(origin, GRACE, ProofOfLife::MeetingClock);
        for tick in 1..=60 {
            w.observe(0, 0, 1_000);
            assert_eq!(
                w.poll(origin + Duration::from_secs(tick)),
                None,
                "said the computer had gone quiet during a pause"
            );
        }
        // Resumed: nine more seconds of real audio, and only the tenth decides.
        for second in 2..=9 {
            w.observe(0, 0, second * 1_000);
            assert_eq!(
                w.poll(origin + Duration::from_secs(60 + second as u64)),
                None
            );
        }
        w.observe(0, 0, 10_000);
        assert_eq!(
            w.poll(origin + Duration::from_secs(70)),
            Some(SilenceVerdict::WentSilent)
        );
    }

    #[test]
    fn a_recording_with_no_microphone_that_gets_audio_says_nothing() {
        let origin = Instant::now();
        let mut w = SilenceWatchdog::new(origin, GRACE, ProofOfLife::MeetingClock);
        for tick in 1..=60 {
            w.observe(50, 0, tick as i64 * 1_000);
            assert_eq!(
                w.poll(origin + Duration::from_secs(tick)),
                None,
                "blamed a channel that was delivering audio all along"
            );
        }
    }

    #[test]
    fn the_recording_that_hears_nothing_at_all_gets_its_own_sentence() {
        // Copy lives in src/lib/copy.ts as
        // labels.degradedReason.nothingIsBeingHeard. If it changes there it
        // changes here.
        assert_eq!(
            NOTHING_IS_BEING_HEARD_MESSAGE,
            "Echo can't hear anything — there's no microphone in this recording, and nothing is coming from this computer. Nothing is being saved, so it's worth stopping and starting again."
        );
        // The whole reason this sentence exists: neither of its two siblings may
        // be said to somebody who has no microphone recording, because both of
        // them make a promise about one.
        assert!(
            !NOTHING_IS_BEING_HEARD_MESSAGE.contains("microphone only")
                && !NOTHING_IS_BEING_HEARD_MESSAGE.contains("still recording")
                && !NOTHING_IS_BEING_HEARD_MESSAGE.contains("who said what"),
            "{NOTHING_IS_BEING_HEARD_MESSAGE} promises something this state cannot keep"
        );
        assert!(
            !NOTHING_IS_BEING_HEARD_MESSAGE.contains('!'),
            "{NOTHING_IS_BEING_HEARD_MESSAGE} shouts at somebody who just lost a recording"
        );
        for jargon in [
            "ScreenCaptureKit",
            "SCStream",
            "OSStatus",
            "queue",
            "buffer",
            "stream",
        ] {
            assert!(
                !NOTHING_IS_BEING_HEARD_MESSAGE.contains(jargon),
                "{NOTHING_IS_BEING_HEARD_MESSAGE} leaks {jargon} to the person"
            );
        }
    }

    #[test]
    fn the_silent_channel_notice_is_the_sentence_the_ui_already_uses() {
        // Copy lives in src/lib/copy.ts as labels.degradedReason.systemAudioUnavailable
        // (and, by reference from there, live.micOnlyBanner and
        // anchors.micOnlyBanner). If it changes there it changes here.
        assert_eq!(
            SYSTEM_AUDIO_SILENT_MESSAGE,
            "Echo is recording through the microphone only, so it can't tell who is speaking — every line says You for now. It will work out who said what once the meeting ends."
        );
        // ...and this one is labels.degradedReason.systemAudioLost (and, by
        // reference, notices.systemAudioLost). The banner the person reads when
        // a working stream dies mid-meeting is drawn from that key, not from the
        // one above, because the sentence above would be false in that state:
        // the lines already written carry real names, and the microphone tail is
        // never re-cut into separate voices (`diarize::pipeline::voice_channel`
        // takes the system branch for any meeting with system audio in it).
        assert_eq!(
            SYSTEM_AUDIO_LOST_MESSAGE,
            "Echo stopped hearing what this computer plays. It's still recording through the microphone."
        );
        assert!(
            !SYSTEM_AUDIO_LOST_MESSAGE.contains("You")
                && !SYSTEM_AUDIO_LOST_MESSAGE.contains("who said what"),
            "{SYSTEM_AUDIO_LOST_MESSAGE} makes a claim about labels it can't keep"
        );
        for jargon in [
            "ScreenCaptureKit",
            "SCStream",
            "OSStatus",
            "queue",
            "buffer",
            "stream",
        ] {
            assert!(
                !SYSTEM_AUDIO_SILENT_MESSAGE.contains(jargon),
                "{SYSTEM_AUDIO_SILENT_MESSAGE} leaks {jargon} to the person"
            );
        }
    }

    // -----------------------------------------------------------------------
    // The microphone's copy of what the computer played
    // -----------------------------------------------------------------------

    /// [`send_utterance`] is the only door out of speech detection, and this is
    /// the guarantee that makes it worth having one: **a stretch the guard calls
    /// a copy never leaves capture as something to write down.**
    ///
    /// `UtteranceReady` is the only way an utterance reaches the engine, so a
    /// suppressed stretch producing none of them is the whole property, asserted
    /// where it is decided rather than three signals downstream.
    #[test]
    fn a_suppressed_stretch_never_leaves_capture_as_one_to_write_down() {
        use crate::audio::bleed::tests::{delayed, speech};
        use crate::audio::route::Route;

        const RATE: usize = TARGET_SAMPLE_RATE as usize;
        /// One frame as the pump produces them: 20 ms.
        const FRAME: usize = RATE * FRAME_MS as usize / 1_000;

        /// A stretch of the microphone as the detector hands it over. `voiced_ms`
        /// is half the duration, which is what a padded real utterance looks
        /// like — see `bleed_guard`'s own tests for why claiming all of it would
        /// not be the conservative choice.
        fn mic(samples: &[f32], from_ms: i64, to_ms: i64) -> Utterance {
            Utterance {
                channel: Channel::Mic,
                t_start_ms: from_ms,
                t_end_ms: to_ms,
                samples: samples[RATE * from_ms as usize / 1_000..RATE * to_ms as usize / 1_000]
                    .to_vec(),
                truncated: false,
                voiced_ms: (to_ms - from_ms) / 2,
            }
        }

        // Twenty seconds of the far side, appended frame by frame exactly as
        // the two-pass drain appends it.
        let system = speech(20_000, 21);
        let mut guard = BleedGuard::new(None, true);
        guard.set_route(Route::Loudspeaker, true);
        for (tick, frame) in system.chunks(FRAME).enumerate() {
            guard.append_system(tick as i64 * 20, frame);
        }

        let (signals, mut received) = unbounded_channel();
        // The microphone's copy: 180 ms late and 25 dB down, a laptop speaker
        // across a desk.
        send_utterance(
            &mut guard,
            &signals,
            mic(&delayed(&system, 180, 0.056), 6_000, 14_000),
        );
        // …and somebody in the room saying something of their own, over the
        // same far side, through the same guard.
        send_utterance(
            &mut guard,
            &signals,
            mic(&speech(20_000, 22), 15_000, 19_500),
        );
        drop(signals);

        let mut sent = Vec::new();
        while let Ok(signal) = received.try_recv() {
            sent.push(signal);
        }
        assert_eq!(sent.len(), 2, "one signal per utterance, always");
        match &sent[0] {
            CaptureSignal::UtteranceSuppressed {
                channel,
                t_start_ms,
                t_end_ms,
                evidence,
            } => {
                assert_eq!(*channel, Channel::Mic);
                assert_eq!((*t_start_ms, *t_end_ms), (6_000, 14_000));
                // The measurement travels with the verdict, because the session
                // layer writes it down and a row that cannot say why it exists
                // is worse than no row.
                assert!(
                    evidence.correlation >= crate::audio::bleed::BLEED_CORRELATION,
                    "the signal carries the correlation that decided it: {evidence:?}"
                );
                assert!(evidence.span_ms > 0, "…and the span it was measured over");
            }
            other => panic!("the copy was offered for text: {other:?}"),
        }
        assert!(
            matches!(&sent[1], CaptureSignal::UtteranceReady(u) if u.t_start_ms == 15_000),
            "one suppression must not put capture in a mood to suppress the next thing it sees"
        );
        assert!(
            !sent.iter().any(
                |signal| matches!(signal, CaptureSignal::UtteranceReady(u) if u.t_start_ms == 6_000)
            ),
            "the suppressed stretch also went out as one to write down"
        );
    }

    /// The geometry that decides whether any of this ever runs: **the far side a
    /// stretch is judged against is audio from after the stretch ended.**
    ///
    /// A verdict needs the computer's audio to `LAG_MIN_MS` — 200 ms — past the
    /// end of the microphone stretch. At the instant the detector closes one,
    /// about 128 ms of that is on hand: it closes 416 ms after the last voiced
    /// window and dates the stretch back to 288 ms after it. Judging there is
    /// not a near miss but a permanent one, and it fails *safe*, which is what
    /// makes it invisible: nothing is deleted, nothing is logged, and the
    /// duplicated lines this exists to remove keep landing. So the stretch
    /// waits the two or three ticks it takes for the rest to arrive.
    #[test]
    fn a_stretch_waits_for_the_far_side_that_comes_after_it() {
        use crate::audio::bleed::tests::{delayed, speech};
        use crate::audio::route::Route;

        const RATE: usize = TARGET_SAMPLE_RATE as usize;
        const FRAME: usize = RATE * FRAME_MS as usize / 1_000;

        /// The far side, and the microphone's copy of it: 180 ms late and 25 dB
        /// down, a laptop speaker across a desk.
        fn far_side() -> (Vec<f32>, Vec<f32>) {
            let system = speech(20_000, 21);
            let mic = delayed(&system, 180, 0.056);
            (system, mic)
        }

        /// The ring as the speech thread has it: frames of the far side, in
        /// order, up to the tick that has arrived.
        fn fill_to(guard: &mut BleedGuard, system: &[f32], appended_ms: &mut i64, to_ms: i64) {
            while *appended_ms + FRAME_MS as i64 <= to_ms {
                let from = RATE * *appended_ms as usize / 1_000;
                guard.append_system(*appended_ms, &system[from..from + FRAME]);
                *appended_ms += FRAME_MS as i64;
            }
        }

        fn cut(source: &[f32], channel: Channel, from_ms: i64, to_ms: i64) -> Utterance {
            Utterance {
                channel,
                t_start_ms: from_ms,
                t_end_ms: to_ms,
                samples: source[RATE * from_ms as usize / 1_000..RATE * to_ms as usize / 1_000]
                    .to_vec(),
                truncated: false,
                voiced_ms: (to_ms - from_ms) / 2,
            }
        }

        let (system, mic) = far_side();
        let mut guard = BleedGuard::new(None, true);
        guard.set_route(Route::Loudspeaker, true);
        let mut appended = 0i64;
        // Exactly what the ring holds at the moment a stretch ending at 10 s is
        // closed: 128 ms of tail, and not one hop more.
        fill_to(&mut guard, &system, &mut appended, 10_128);

        let (signals, mut received) = unbounded_channel();
        let mut waiting = WaitingRoom::default();
        let now = Instant::now();
        waiting.hold(now, cut(&mic, Channel::Mic, 2_000, 10_000));
        // …and the far side's own next utterance behind it, which never waits
        // for anything itself.
        waiting.hold(now, cut(&system, Channel::System, 10_100, 12_000));
        waiting.release(&mut guard, &signals, now);
        assert!(
            received.try_recv().is_err(),
            "judged the moment it closed, against a far side 72 ms short of what \
             every delay in the search needs — which is a refusal every time"
        );

        // Three more ticks of the far side, which is all the wait ever is.
        fill_to(&mut guard, &system, &mut appended, 10_200);
        waiting.release(&mut guard, &signals, now);

        let first = received.try_recv().expect("the wait was over");
        assert!(
            matches!(
                first,
                CaptureSignal::UtteranceSuppressed {
                    channel: Channel::Mic,
                    t_start_ms: 2_000,
                    ..
                }
            ),
            "the copy went out as one to write down after waiting for the far \
             side that proves it is a copy: {first:?}"
        );
        // The system channel's own stretch was behind it in the one queue, and
        // came out behind it: waiting must not reorder the two channels.
        assert!(
            matches!(
                received.try_recv(),
                Ok(CaptureSignal::UtteranceReady(u)) if u.channel == Channel::System
            ),
            "the far side's own utterance overtook the microphone stretch it was queued behind"
        );
    }

    /// …and it is a wait, not a hold: a far side that stops arriving — a system
    /// stream that died while it still claims to be live — must not keep the
    /// microphone's words off the screen.
    #[test]
    fn a_stretch_stops_waiting_when_the_far_side_stops_coming() {
        use crate::audio::bleed::tests::{delayed, speech};
        use crate::audio::route::Route;

        const RATE: usize = TARGET_SAMPLE_RATE as usize;
        let system = speech(20_000, 21);
        let mic = delayed(&system, 180, 0.056);
        let mut guard = BleedGuard::new(None, true);
        guard.set_route(Route::Loudspeaker, true);
        // The far side stops 128 ms past the end of the stretch and never comes
        // back.
        guard.append_system(0, &system[..RATE * 10_128 / 1_000]);

        let (signals, mut received) = unbounded_channel();
        let mut waiting = WaitingRoom::default();
        let now = Instant::now();
        waiting.hold(
            now,
            Utterance {
                channel: Channel::Mic,
                t_start_ms: 2_000,
                t_end_ms: 10_000,
                samples: mic[RATE * 2..RATE * 10].to_vec(),
                truncated: false,
                voiced_ms: 4_000,
            },
        );
        waiting.release(
            &mut guard,
            &signals,
            now + FAR_SIDE_WAIT - Duration::from_millis(1),
        );
        assert!(
            received.try_recv().is_err(),
            "gave up before the wait was up"
        );

        waiting.release(&mut guard, &signals, now + FAR_SIDE_WAIT);
        let sent = received.try_recv().expect("the wait ran out");
        assert!(
            matches!(&sent, CaptureSignal::UtteranceReady(u) if u.t_start_ms == 2_000
                && u.samples.len() == RATE * 8),
            "a stretch nothing could be judged against goes out whole and \
             unjudged, audio and all: {sent:?}"
        );
    }

    /// Nothing waits for a far side that is not being kept: an unarmed guard —
    /// headphones, no system channel, the kill switch — judges nothing, so
    /// holding its utterances would be latency bought for no reason at all.
    #[test]
    fn nothing_waits_when_there_is_nothing_to_judge_against() {
        use crate::audio::route::Route;

        let mut headphones = BleedGuard::new(None, true);
        headphones.set_route(Route::Headphones, true);
        let mic = Utterance {
            channel: Channel::Mic,
            t_start_ms: 2_000,
            t_end_ms: 10_000,
            samples: vec![0.0; TARGET_SAMPLE_RATE as usize * 8],
            truncated: false,
            voiced_ms: 4_000,
        };
        assert!(far_side_on_hand(&headphones, &mic));

        // Armed, but the far side of this stretch is genuinely not there yet.
        let mut armed = BleedGuard::new(None, true);
        armed.set_route(Route::Loudspeaker, true);
        assert!(
            !far_side_on_hand(&armed, &mic),
            "an armed guard with an empty ring has everything to wait for"
        );
        // The computer's own utterances are the original; there is nothing for
        // them to be a copy of, so they never wait.
        assert!(far_side_on_hand(
            &armed,
            &Utterance {
                channel: Channel::System,
                ..mic.clone()
            }
        ));
    }

    /// The silent failure this whole change would otherwise have: a machine
    /// whose true delay is outside the search window suppresses nothing and
    /// transcribes exactly as it did before, which is safe and invisible.
    #[test]
    fn a_meeting_that_looked_hard_and_found_nothing_says_so() {
        let looked_and_found_nothing = bleed_guard::BleedReport {
            armed: true,
            route: route::Route::Loudspeaker,
            examined: 40,
            suppressed: 0,
            undecided: 0,
            too_short: 12,
            opportunities: 30,
            lag_ms: None,
            dropped_backwards: 0,
            held_ms: 30_000,
        };
        assert!(never_found_a_copy(&looked_and_found_nothing));

        // A meeting where nobody played anything out loud offered nothing to
        // find a copy of, and saying "never found it" about it would be noise.
        assert!(!never_found_a_copy(&bleed_guard::BleedReport {
            opportunities: 2,
            ..looked_and_found_nothing
        }));
        // A handful of stretches is not a sample.
        assert!(!never_found_a_copy(&bleed_guard::BleedReport {
            examined: 6,
            opportunities: 6,
            ..looked_and_found_nothing
        }));
        // A guard that was never armed was not looking.
        assert!(!never_found_a_copy(&bleed_guard::BleedReport {
            armed: false,
            ..looked_and_found_nothing
        }));
        // …and one that found copies has nothing to complain about.
        assert!(!never_found_a_copy(&bleed_guard::BleedReport {
            suppressed: 1,
            ..looked_and_found_nothing
        }));

        // The other silent failure, and the one `never_found_a_copy` is blind
        // to by construction: never having the far side to compare against
        // leaves `examined` at zero, so its sample-size condition can never be
        // met however long the meeting runs.
        let never_had_anything_to_compare = bleed_guard::BleedReport {
            examined: 0,
            opportunities: 0,
            undecided: 40,
            ..looked_and_found_nothing
        };
        assert!(!never_found_a_copy(&never_had_anything_to_compare));
        assert!(never_had_the_far_side(&never_had_anything_to_compare));
        // A meeting that was measuring fine and refused a few stretches along
        // the way is an ordinary meeting.
        assert!(!never_had_the_far_side(&bleed_guard::BleedReport {
            undecided: 40,
            ..looked_and_found_nothing
        }));
        // A handful of refusals is not a pattern.
        assert!(!never_had_the_far_side(&bleed_guard::BleedReport {
            undecided: 3,
            ..never_had_anything_to_compare
        }));
        // …and a guard that was never armed was not looking.
        assert!(!never_had_the_far_side(&bleed_guard::BleedReport {
            armed: false,
            ..never_had_anything_to_compare
        }));
    }

    #[test]
    fn every_system_audio_message_says_what_changed_and_claims_nothing_more() {
        // The three sibling messages for "the system channel is missing" must
        // never contradict each other or read like a panic — this sweeps all
        // three in one place so a future edit to any of them stays honest.
        for message in [
            SYSTEM_AUDIO_SILENT_MESSAGE,
            SYSTEM_AUDIO_UNAVAILABLE_AT_START,
            SYSTEM_AUDIO_LOST_MESSAGE,
        ] {
            assert!(message.ends_with('.'), "{message}");
            assert!(!message.contains('!'), "{message}");
            assert!(
                !message.to_lowercase().contains("recording you"),
                "{message} claims Echo is recording the person specifically, which it can't tell"
            );
            for jargon in [
                "ScreenCaptureKit",
                "SCStream",
                "OSStatus",
                "queue",
                "buffer",
                "stream",
            ] {
                assert!(
                    !message.contains(jargon),
                    "{message} leaks {jargon} to the person"
                );
            }
        }
    }

    #[test]
    fn chunks_are_grouped_per_channel_and_put_back_in_time_order() {
        let chunks = vec![
            ChunkRef::new("/m/system-000001.wav", Channel::System, 30_000, 60_000),
            ChunkRef::new("/m/mic-000001.wav", Channel::Mic, 30_000, 60_000),
            ChunkRef::new("/m/mic-000000.wav", Channel::Mic, 0, 30_000),
            ChunkRef::new("/m/system-000000.wav", Channel::System, 0, 30_000),
        ];
        let groups = group_by_channel(&chunks);
        assert_eq!(groups.len(), 2);
        for group in &groups {
            assert_eq!(group.chunks.len(), 2);
            assert_eq!(
                group.chunks[0].t_start_ms, 0,
                "chunks must come back in order"
            );
            assert_eq!(group.chunks[1].t_start_ms, 30_000);
        }
        assert!(groups.iter().any(|g| g.channel == Channel::Mic));
        assert!(groups.iter().any(|g| g.channel == Channel::System));
    }

    #[test]
    fn a_capture_config_needs_only_a_meeting_and_a_folder() {
        let cfg = CaptureConfig::new("abc", "/tmp/echo-test");
        assert!(cfg.capture_system_audio, "system audio is on by default");
        assert!(cfg.input_device_id.is_none());
        assert!(
            cfg.log_dir.is_none(),
            "a capture told about no log folder looks in none, so no test can \
             pick up somebody's real kill-switch file"
        );
        let cfg = cfg.with_system_audio(false);
        assert!(!cfg.capture_system_audio);
    }

    #[test]
    fn every_system_audio_failure_reads_as_something_the_person_can_do() {
        for error in [
            AudioError::PermissionDenied,
            AudioError::SystemAudioUnsupported("SCStream failed with OSStatus -3801".into()),
            AudioError::DeviceLost,
        ] {
            let message = describe_system_error(&error);
            assert!(message.ends_with('.'), "{message}");
            for jargon in [
                "OSStatus",
                "SCStream",
                "PipeWire",
                "ScreenCaptureKit",
                "ONNX",
                "VAD",
                "codec",
            ] {
                assert!(
                    !message.contains(jargon),
                    "{message} leaks {jargon} to the person"
                );
            }
        }
    }

    #[test]
    fn reading_an_empty_window_is_empty_not_an_error() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let out = rt.block_on(read_window(&[], 0, 0)).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn reading_a_window_pads_past_the_end_of_the_recording() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = ChunkWriter::create(dir.path(), Channel::Mic, TARGET_SAMPLE_RATE).unwrap();
        w.write_samples(&vec![0.5f32; TARGET_SAMPLE_RATE as usize])
            .unwrap();
        let chunk = w.finish().unwrap().unwrap();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let out = rt
            .block_on(read_window(&[ChunkRef::from_committed(&chunk)], 500, 2_000))
            .unwrap();
        assert_eq!(out.len(), TARGET_SAMPLE_RATE as usize * 3 / 2);
        // The first half second is real audio, the rest is silence past the end.
        assert!((out[10] - 0.5).abs() < 1e-2, "{}", out[10]);
        assert_eq!(out[out.len() - 1], 0.0);
    }

    #[test]
    fn a_window_after_an_unreadable_chunk_is_still_the_right_audio() {
        let dir = tempfile::tempdir().unwrap();
        // Three seconds in three one-second chunks; the middle one is corrupt.
        let mut chunks = Vec::new();
        for (seq, value) in [(0u64, 0.2f32), (1, 0.4), (2, 0.6)] {
            let mut w = ChunkWriter::create(dir.path(), Channel::Mic, TARGET_SAMPLE_RATE).unwrap();
            w.resume_at(seq, seq as i64 * 1_000);
            w.write_samples(&vec![value; TARGET_SAMPLE_RATE as usize])
                .unwrap();
            chunks.push(ChunkRef::from_committed(&w.finish().unwrap().unwrap()));
        }
        std::fs::write(&chunks[1].path, b"not a recording").unwrap();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        // The third second must still be the third second.
        let out = rt.block_on(read_window(&chunks, 2_000, 3_000)).unwrap();
        assert_eq!(out.len(), TARGET_SAMPLE_RATE as usize);
        assert!((out[100] - 0.6).abs() < 1e-2, "{}", out[100]);
        // And the hole reads as silence where it happened.
        let hole = rt.block_on(read_window(&chunks, 1_000, 2_000)).unwrap();
        assert!(hole.iter().all(|s| *s == 0.0));
    }

    /// A window opens the files it needs and no others.
    ///
    /// Not a micro-optimisation: this used to decode every chunk of the channel
    /// on every call, so the catch-up pass over a long meeting read the whole
    /// recording once per window — a whole recording's worth of decoding for
    /// thirty seconds of audio, hundreds of times over.
    #[test]
    fn a_window_only_opens_the_chunks_it_covers() {
        let hour: Vec<ChunkRef> = (0..120)
            .map(|seq| {
                let from = seq * i64::from(CHUNK_SECONDS) * 1_000;
                ChunkRef::new(
                    format!("/a/mic-{seq:06}.wav"),
                    Channel::Mic,
                    from,
                    from + i64::from(CHUNK_SECONDS) * 1_000,
                )
            })
            .collect();

        // A window in the middle of the meeting: the chunk it lands in, and one
        // behind it in case that file overran its journal entry.
        let touched = chunks_touching(&hour, 1_800_000, 1_824_000);
        assert_eq!(touched.len(), 2, "{touched:?}");
        assert_eq!(touched[0].t_start_ms, 1_770_000);
        assert_eq!(touched[1].t_start_ms, 1_800_000);

        // A window straddling a boundary takes both, and still not the rest.
        let straddling = chunks_touching(&hour, 1_790_000, 1_814_000);
        assert_eq!(straddling.len(), 3);

        // A chunk that ends exactly where the window starts is not in the
        // window — but it is inside the slack, so it is still read.
        assert!(chunks_touching(&hour, 30_000, 60_000)
            .iter()
            .any(|c| c.t_start_ms == 0));
        // And a window before all of the audio takes nothing at all.
        assert!(chunks_touching(&hour, 0, 1).len() <= 1);
    }

    /// The window that comes back is the same audio it always was: the chunks
    /// left out cannot contribute a sample to it.
    #[test]
    fn leaving_chunks_out_changes_no_sample_of_the_window() {
        let dir = tempfile::tempdir().unwrap();
        let mut chunks = Vec::new();
        // Three one-second chunks, a minute apart, so each window is well clear
        // of the others (`chunks_touching` allows a chunk of slack).
        for (seq, value) in [(0u64, 0.2f32), (1, 0.4), (2, 0.6)] {
            let mut w = ChunkWriter::create(dir.path(), Channel::Mic, TARGET_SAMPLE_RATE).unwrap();
            w.resume_at(seq, seq as i64 * 60_000);
            w.write_samples(&vec![value; TARGET_SAMPLE_RATE as usize])
                .unwrap();
            chunks.push(ChunkRef::from_committed(&w.finish().unwrap().unwrap()));
        }
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        for (i, expected) in [0.2f32, 0.4, 0.6].iter().enumerate() {
            let at = i as i64 * 60_000;
            let out = rt.block_on(read_window(&chunks, at, at + 1_000)).unwrap();
            assert_eq!(out.len(), TARGET_SAMPLE_RATE as usize);
            assert!((out[100] - expected).abs() < 1e-2, "{} at {at}", out[100]);
        }
    }

    #[test]
    fn a_gap_between_chunks_reads_back_as_silence_in_its_own_place() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = ChunkWriter::create(dir.path(), Channel::Mic, TARGET_SAMPLE_RATE).unwrap();
        w.write_samples(&vec![0.5f32; TARGET_SAMPLE_RATE as usize])
            .unwrap();
        let first = ChunkRef::from_committed(&w.flush().unwrap().unwrap());
        // The channel came back five minutes later.
        w.resume_at(1, 300_000);
        w.write_samples(&vec![0.5f32; TARGET_SAMPLE_RATE as usize])
            .unwrap();
        let second = ChunkRef::from_committed(&w.finish().unwrap().unwrap());

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let out = rt
            .block_on(read_window(&[first, second], 300_000, 301_000))
            .unwrap();
        assert!(
            (out[100] - 0.5).abs() < 1e-2,
            "audio after a gap moved: {}",
            out[100]
        );
    }

    #[test]
    fn a_mixdown_sums_both_channels_without_clipping() {
        let dir = tempfile::tempdir().unwrap();
        let mut chunks = Vec::new();
        for channel in [Channel::Mic, Channel::System] {
            let mut w = ChunkWriter::create(dir.path(), channel, TARGET_SAMPLE_RATE).unwrap();
            w.write_samples(&vec![0.8f32; TARGET_SAMPLE_RATE as usize / 2])
                .unwrap();
            chunks.push(ChunkRef::from_committed(&w.finish().unwrap().unwrap()));
        }

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let destination = dir.path().join("mixed.wav");
        let out = rt.block_on(build_mixdown(&chunks, &destination)).unwrap();
        assert_eq!(out, destination);

        let mixed = writer::read_wav_16k_mono(&destination).unwrap();
        assert_eq!(mixed.len(), TARGET_SAMPLE_RATE as usize / 2);
        let peak = mixed.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(peak <= 1.0, "the mixdown clipped at {peak}");
        assert!(peak > 0.9, "the mixdown lost the audio: {peak}");
    }

    #[test]
    fn a_mixdown_with_nothing_to_mix_says_so_in_plain_words() {
        let dir = tempfile::tempdir().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let err = rt
            .block_on(build_mixdown(&[], &dir.path().join("mixed.wav")))
            .unwrap_err();
        assert!(err.to_string().contains("no audio"), "{err}");
    }

    #[test]
    fn asking_to_capture_with_no_sources_at_all_is_a_clean_error() {
        let dir = tempfile::tempdir().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        // No microphone selectable and system audio switched off: capture must
        // refuse rather than pretend to record silence.
        let cfg = CaptureConfig::new("test-meeting", dir.path())
            .with_system_audio(false)
            .with_input_device(Some("no-such-device".into()));
        let result = rt.block_on(CaptureSession::start(cfg));
        assert!(result.is_err(), "capture claimed to start with no sources");
    }

    #[test]
    fn recovering_an_empty_folder_finds_nothing_and_does_not_fail() {
        let dir = tempfile::tempdir().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let found = rt.block_on(recover_chunks(dir.path())).unwrap();
        assert!(found.is_empty());
    }
}
