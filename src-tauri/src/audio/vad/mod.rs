//! Speech detection: which stretches of a channel contain someone talking.
//!
//! Silero through `ort`, CPU only. On Linux there is no Vulkan execution
//! provider, so all of this stays CPU-bound (DESIGN §2, review finding 2).
//!
//! Lifecycle (mantra 1): the detector is created when a recording starts and
//! dropped when it stops. Nothing is loaded while Echo sits idle.
//!
//! What this module hands the speech engine is an *utterance*: a stretch of
//! speech padded on both sides, capped at [`MAX_UTTERANCE_MS`] so one long
//! monologue cannot stall the queue (review finding 20), and never shorter than
//! [`MIN_UTTERANCE_MS`], because a window under a second is one the engine
//! cannot read at all.
//!
//! ## Two detectors, one behaviour
//!
//! Segmentation is separate from the decision "is this 32 ms speech?". The
//! decision comes either from Silero (when the small detector asset is on disk)
//! or from a loudness gate (when it is not, and in headless tests). That is not
//! a shortcut: it is what lets a person record on first launch, before anything
//! has finished downloading, and still get a transcript. The segmentation —
//! padding, tails, the long-monologue cut — is identical either way and is what
//! the tests here pin down.
//!
//! The detector asset is fetched at runtime through the speech-asset catalog
//! (`asr::models`, [`crate::types::AssetKind::SpeechDetector`]); nothing is
//! bundled. See [`SpeechDetector::load`].

use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};

use crate::audio::{AudioError, TARGET_SAMPLE_RATE};
use crate::types::Channel;

/// Padding kept before and after detected speech, so words are not clipped.
pub const PAD_MS: i64 = 300;

/// Longest utterance handed downstream. Longer speech is cut at the quietest
/// point found near this limit.
///
/// The speech engine works in 30 s windows, so staying under that with room for
/// padding on both sides means an utterance never has to be split again later.
pub const MAX_UTTERANCE_MS: i64 = 28_000;

/// Silence this long ends an utterance.
pub const SILENCE_TAIL_MS: i64 = 600;

/// Shortest utterance handed downstream.
///
/// The speech engine cannot read a window under a second — whisper.cpp fails to
/// encode it and answers an error instead of text. That is not a reason to throw
/// the words away: a meeting is full of short answers ("yeah", "no, Tuesday"),
/// and each one is a line of the transcript.
///
/// So an utterance that would come out shorter than this is not trimmed back to
/// [`PAD_MS`]. It keeps the silence it already has around it — audio that was
/// captured, held, and until now discarded by [`Segmenter::close`] — until the
/// window is long enough to be read. Nothing is invented and nothing is lost.
///
/// The one case that can still come out shorter is the very end of a recording,
/// where there is no more audio to keep; the engine zero-pads that last window
/// itself.
pub const MIN_UTTERANCE_MS: i64 = 1_100;

/// Silero's window at 16 kHz. Everything here works in whole windows.
pub const WINDOW_SAMPLES: usize = 512;

/// 32 ms.
pub const WINDOW_MS: i64 = WINDOW_SAMPLES as i64 * 1_000 / TARGET_SAMPLE_RATE as i64;

/// The look-behind pad, rounded down to whole windows.
///
/// Keeping every position a whole number of windows is not tidiness: it makes
/// every offset an exact number of milliseconds, so consecutive utterances tile
/// the timeline with no rounding gap between them.
pub const PAD_SAMPLES: usize =
    (PAD_MS as usize * TARGET_SAMPLE_RATE as usize / 1_000) / WINDOW_SAMPLES * WINDOW_SAMPLES;

/// Probability at which speech is considered to have started.
const SPEECH_ENTER: f32 = 0.5;
/// …and the lower bar it has to fall below to be considered over. The gap stops
/// a detector that hovers around the threshold from shredding a sentence.
const SPEECH_EXIT: f32 = 0.35;

/// A stretch of one channel that contains speech.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Utterance {
    pub channel: Channel,
    pub t_start_ms: i64,
    pub t_end_ms: i64,
    /// 16 kHz mono audio for this stretch, padding included.
    pub samples: Vec<f32>,
    /// True when the cut was forced by [`MAX_UTTERANCE_MS`] rather than by
    /// silence, so the engine can carry context across the join.
    pub truncated: bool,
}

impl Utterance {
    pub fn duration_ms(&self) -> i64 {
        self.t_end_ms - self.t_start_ms
    }
}

fn samples_to_ms(samples: usize) -> i64 {
    samples as i64 * 1_000 / i64::from(TARGET_SAMPLE_RATE)
}

fn ms_to_samples(ms: i64) -> usize {
    (ms.max(0) * i64::from(TARGET_SAMPLE_RATE) / 1_000) as usize
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f32 = samples.iter().map(|s| s * s).sum();
    (sum / samples.len() as f32).sqrt()
}

// ---------------------------------------------------------------------------
// Segmentation — pure, deterministic, and the part that decides quality
// ---------------------------------------------------------------------------

struct OpenUtterance {
    /// Absolute sample position of the first sample.
    start_pos: u64,
    samples: Vec<f32>,
    /// Length up to and including the last window that was speech.
    voiced_len: usize,
}

/// Turns a per-window speech decision into padded utterances.
///
/// Independent of how the decision was made, so it can be tested with generated
/// audio and no model on disk.
#[derive(Debug, Default)]
pub struct Segmenter {
    channel: Channel,
    /// Absolute position (in samples) of the next window to arrive.
    pos: u64,
    /// Meeting-clock offset of absolute position 0.
    origin_ms: i64,
    started: bool,
    /// Rolling buffer of the audio just before speech started.
    pre: VecDeque<f32>,
    open: Option<OpenUtterance>,
    /// Consecutive silent windows since the last voiced one.
    silence_windows: usize,
    in_speech: bool,
}

impl std::fmt::Debug for OpenUtterance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenUtterance")
            .field("start_pos", &self.start_pos)
            .field("samples", &self.samples.len())
            .field("voiced_len", &self.voiced_len)
            .finish()
    }
}

impl Segmenter {
    pub fn new(channel: Channel) -> Self {
        Self {
            channel,
            pos: 0,
            origin_ms: 0,
            started: false,
            pre: VecDeque::with_capacity(ms_to_samples(PAD_MS) + WINDOW_SAMPLES),
            open: None,
            silence_windows: 0,
            in_speech: false,
        }
    }

    /// Meeting-clock offset the segmenter believes it is at.
    pub fn position_ms(&self) -> i64 {
        self.origin_ms + samples_to_ms(self.pos as usize)
    }

    /// Is someone talking *now*?
    ///
    /// Not the same as "an utterance is open": once the silence tail has run out
    /// the utterance may stay open a little longer, waiting for enough audio to
    /// be readable (see [`MIN_UTTERANCE_MS`]). The indicator, and the hysteresis
    /// in the per-window decision, both want the honest answer.
    pub fn is_speaking(&self) -> bool {
        self.in_speech && !self.tail_run_out()
    }

    /// Has the run of silence since the last voiced window reached
    /// [`SILENCE_TAIL_MS`]?
    fn tail_run_out(&self) -> bool {
        self.silence_windows as i64 * WINDOW_MS >= SILENCE_TAIL_MS
    }

    /// Anchor the timeline. Called before the first window, and again after a
    /// gap so a discontinuity does not slide every later utterance.
    pub fn rebase(&mut self, t_ms: i64) -> Option<Utterance> {
        let finished = self.finish();
        self.origin_ms = t_ms;
        self.pos = 0;
        self.started = true;
        self.pre.clear();
        self.silence_windows = 0;
        self.in_speech = false;
        finished
    }

    fn pos_to_ms(&self, pos: u64) -> i64 {
        self.origin_ms + samples_to_ms(pos as usize)
    }

    /// Feed exactly one window plus the decision made about it.
    pub fn push_window(&mut self, window: &[f32], is_speech: bool) -> Vec<Utterance> {
        if !self.started {
            self.started = true;
        }
        let mut out = Vec::new();
        let window_pos = self.pos;
        self.pos += window.len() as u64;

        if !self.in_speech {
            if is_speech {
                let pad: Vec<f32> = self.pre.iter().copied().collect();
                self.pre.clear();
                let start_pos = window_pos - pad.len() as u64;
                let mut samples = pad;
                samples.extend_from_slice(window);
                let voiced_len = samples.len();
                self.open = Some(OpenUtterance {
                    start_pos,
                    samples,
                    voiced_len,
                });
                self.in_speech = true;
                self.silence_windows = 0;
            } else {
                self.pre.extend(window.iter().copied());
                while self.pre.len() > PAD_SAMPLES {
                    for _ in 0..WINDOW_SAMPLES {
                        self.pre.pop_front();
                    }
                }
            }
            return out;
        }

        // Already inside an utterance.
        let max_samples = ms_to_samples(MAX_UTTERANCE_MS);
        {
            let open = self
                .open
                .as_mut()
                .expect("in_speech implies an open utterance");
            open.samples.extend_from_slice(window);
            if is_speech {
                open.voiced_len = open.samples.len();
                self.silence_windows = 0;
            } else {
                self.silence_windows += 1;
            }
        }

        let too_long = self
            .open
            .as_ref()
            .is_some_and(|o| o.samples.len() >= max_samples);
        // Speech is over, but the window may still be too short for the engine
        // to read. Hold it open and keep the silence that is arriving anyway
        // rather than emitting a sliver nothing can transcribe.
        let ready = self
            .open
            .as_ref()
            .is_some_and(|o| o.samples.len() >= ms_to_samples(MIN_UTTERANCE_MS));

        if too_long {
            if let Some(u) = self.force_cut() {
                out.push(u);
            }
        } else if self.tail_run_out() && ready {
            if let Some(u) = self.close(false) {
                out.push(u);
            }
        }
        out
    }

    /// Close the open utterance at its natural end: the last voiced window plus
    /// the trailing pad — or more of that trailing silence, when trimming to the
    /// pad would leave a window the engine cannot read ([`MIN_UTTERANCE_MS`]).
    fn close(&mut self, truncated: bool) -> Option<Utterance> {
        let open = self.open.take()?;
        self.in_speech = false;
        self.silence_windows = 0;

        // Normally: the speech plus a trailing pad. But never trim a window back
        // below what the engine can read when the audio to fill it is already in
        // hand — that is the whole reason short answers used to come out as
        // errors instead of text.
        let keep = (open.voiced_len + PAD_SAMPLES)
            .max(ms_to_samples(MIN_UTTERANCE_MS))
            .min(open.samples.len());
        let mut samples = open.samples;
        // Whatever is trimmed off is silence that may lead into the next
        // utterance; keep the last pad of it as look-behind.
        let leftover: Vec<f32> = samples[keep..].to_vec();
        samples.truncate(keep);

        self.pre.clear();
        let from = leftover.len().saturating_sub(PAD_SAMPLES);
        for s in &leftover[from..] {
            self.pre.push_back(*s);
        }

        if samples.is_empty() {
            return None;
        }
        let t_start_ms = self.pos_to_ms(open.start_pos);
        Some(Utterance {
            channel: self.channel,
            t_start_ms,
            t_end_ms: self.pos_to_ms(open.start_pos + samples.len() as u64),
            samples,
            truncated,
        })
    }

    /// The monologue case: cut at the quietest window near the limit and keep
    /// going from there, so long speech becomes several utterances instead of
    /// one that no engine will accept.
    fn force_cut(&mut self) -> Option<Utterance> {
        let open = self.open.as_mut()?;
        let len = open.samples.len();
        // Look for a breath in the last two seconds, on window boundaries so
        // every offset stays an exact number of milliseconds.
        let search_from =
            len.saturating_sub(ms_to_samples(2_000)) / WINDOW_SAMPLES * WINDOW_SAMPLES;
        let mut best = len;
        let mut best_rms = f32::MAX;
        let mut w = search_from;
        while w + WINDOW_SAMPLES <= len {
            let level = rms(&open.samples[w..w + WINDOW_SAMPLES]);
            if level < best_rms {
                best_rms = level;
                best = w + WINDOW_SAMPLES;
            }
            w += WINDOW_SAMPLES;
        }
        let cut = best.clamp(WINDOW_SAMPLES.min(len), len);

        let rest = open.samples.split_off(cut);
        let emitted = std::mem::take(&mut open.samples);
        let start_pos = open.start_pos;
        // The silence the cut was aimed at went out with the piece just emitted.
        // Carrying its count over would close the continuation immediately, as a
        // fragment of a sentence nobody can read.
        self.silence_windows = 0;

        // Carry on from the cut with whatever came after it.
        let rest_len = rest.len();
        open.start_pos = start_pos + cut as u64;
        open.samples = rest;
        open.voiced_len = rest_len;

        let t_start_ms = self.pos_to_ms(start_pos);
        Some(Utterance {
            channel: self.channel,
            t_start_ms,
            t_end_ms: self.pos_to_ms(start_pos + emitted.len() as u64),
            samples: emitted,
            truncated: true,
        })
    }

    /// End of stream: emit whatever is still open.
    ///
    /// This is the one exit that may produce something shorter than
    /// [`MIN_UTTERANCE_MS`], because there is no more audio to hold out for. The
    /// engine pads that last window itself.
    pub fn finish(&mut self) -> Option<Utterance> {
        self.open.as_ref()?;
        self.close(false)
    }
}

// ---------------------------------------------------------------------------
// The per-window decision
// ---------------------------------------------------------------------------

/// Loudness gate used when the detector asset is not on disk yet.
///
/// Tracks the quietest thing it has heard recently as the room's noise floor
/// and calls anything well above it speech.
#[derive(Debug)]
struct LoudnessGate {
    noise_floor: f32,
    active: bool,
}

impl Default for LoudnessGate {
    fn default() -> Self {
        Self {
            noise_floor: 0.0015,
            active: false,
        }
    }
}

impl LoudnessGate {
    fn probability(&mut self, window: &[f32]) -> f32 {
        let level = rms(window);
        let enter = (self.noise_floor * 4.0).max(0.01);
        let exit = (self.noise_floor * 2.5).max(0.005);
        self.active = if self.active {
            level > exit
        } else {
            level > enter
        };
        if !self.active {
            // Creep towards quiet, jump away from loud: a slammed door must not
            // become the new floor.
            self.noise_floor = (self.noise_floor * 0.95 + level * 0.05).clamp(1e-5, 0.05);
        }
        if self.active {
            1.0
        } else {
            0.0
        }
    }
}

struct Silero {
    session: ort::session::Session,
    /// Silero v5 carries a single `[2, 1, 128]` state between windows.
    state: Vec<f32>,
    input_name: String,
    state_name: String,
    rate_name: String,
    prob_output: String,
    state_output: String,
}

impl std::fmt::Debug for Silero {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Silero").finish()
    }
}

/// Silero's recurrent state: two layers, one batch, 128 hidden units.
const SILERO_STATE_LAYERS: usize = 2;
const SILERO_STATE_HIDDEN: usize = 128;
const SILERO_STATE_LEN: usize = SILERO_STATE_LAYERS * SILERO_STATE_HIDDEN;

impl Silero {
    fn load(model_path: &Path) -> Result<Self, AudioError> {
        let mut builder = ort::session::Session::builder()
            .map_err(|e| AudioError::Backend(format!("speech detection unavailable: {e}")))?
            .with_intra_threads(1)
            .map_err(|e| AudioError::Backend(format!("speech detection unavailable: {e}")))?;
        let session = builder
            .commit_from_file(model_path)
            .map_err(|e| AudioError::Backend(format!("speech detection unavailable: {e}")))?;

        let input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        let output_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        let pick = |names: &[String], wanted: &str, fallback: usize| -> Option<String> {
            names
                .iter()
                .find(|n| n.eq_ignore_ascii_case(wanted))
                .cloned()
                .or_else(|| names.get(fallback).cloned())
        };
        let input_name = pick(&input_names, "input", 0).ok_or_else(|| {
            AudioError::Backend("the speech detector has no audio input".to_string())
        })?;
        let state_name = pick(&input_names, "state", 1).ok_or_else(|| {
            AudioError::Backend("the speech detector has an unexpected shape".to_string())
        })?;
        let rate_name = pick(&input_names, "sr", 2).ok_or_else(|| {
            AudioError::Backend("the speech detector has an unexpected shape".to_string())
        })?;
        let prob_output = pick(&output_names, "output", 0).ok_or_else(|| {
            AudioError::Backend("the speech detector produces nothing usable".to_string())
        })?;
        let state_output = pick(&output_names, "stateN", 1).unwrap_or_default();

        Ok(Self {
            session,
            state: vec![0.0; SILERO_STATE_LEN],
            input_name,
            state_name,
            rate_name,
            prob_output,
            state_output,
        })
    }

    fn probability(&mut self, window: &[f32]) -> Result<f32, AudioError> {
        use ort::value::Tensor;

        let audio = Tensor::from_array((vec![1_i64, window.len() as i64], window.to_vec()))
            .map_err(|e| AudioError::Backend(e.to_string()))?;
        let state = Tensor::from_array((vec![2_i64, 1, 128], self.state.clone()))
            .map_err(|e| AudioError::Backend(e.to_string()))?;
        let rate = Tensor::from_array((vec![1_i64], vec![i64::from(TARGET_SAMPLE_RATE)]))
            .map_err(|e| AudioError::Backend(e.to_string()))?;

        let outputs = self
            .session
            .run(ort::inputs![
                self.input_name.as_str() => audio,
                self.state_name.as_str() => state,
                self.rate_name.as_str() => rate,
            ])
            .map_err(|e| AudioError::Backend(e.to_string()))?;

        let probability = outputs[self.prob_output.as_str()]
            .try_extract_tensor::<f32>()
            .map(|(_, data)| data.first().copied().unwrap_or(0.0))
            .map_err(|e| AudioError::Backend(e.to_string()))?;

        if !self.state_output.is_empty() {
            if let Some(value) = outputs.get(self.state_output.as_str()) {
                if let Ok((_, data)) = value.try_extract_tensor::<f32>() {
                    if data.len() == SILERO_STATE_LEN {
                        self.state.copy_from_slice(data);
                    }
                }
            }
        }
        Ok(probability)
    }
}

#[derive(Debug)]
enum Engine {
    Loudness(LoudnessGate),
    Silero(Box<Silero>),
}

/// Per-recording speech detector, one instance per channel.
#[derive(Debug)]
pub struct SpeechDetector {
    engine: Engine,
    segmenter: Segmenter,
    /// Audio that did not fill a whole window yet.
    leftover: Vec<f32>,
    /// Where `leftover` starts on the meeting clock.
    next_t_ms: i64,
    anchored: bool,
    speaking: bool,
    /// How many windows the model refused before we gave up on it.
    engine_failures: u32,
}

impl SpeechDetector {
    /// Load the detector from the downloaded asset and prepare state for one
    /// channel. Called at recording start.
    ///
    /// The asset comes from the speech-asset catalog at runtime:
    /// `asr::models::installed_path(db, AssetKind::SpeechDetector)`. When it is
    /// absent — first launch, download still running, a person who skipped it —
    /// use [`SpeechDetector::without_model`] and keep recording.
    pub fn load(model_path: &Path, channel: Channel) -> Result<Self, AudioError> {
        let engine = Engine::Silero(Box::new(Silero::load(model_path)?));
        Ok(Self {
            engine,
            segmenter: Segmenter::new(channel),
            leftover: Vec::with_capacity(WINDOW_SAMPLES * 2),
            next_t_ms: 0,
            anchored: false,
            speaking: false,
            engine_failures: 0,
        })
    }

    /// Detector for when the asset is not on disk. Recording never waits for a
    /// download.
    pub fn without_model(channel: Channel) -> Self {
        Self {
            engine: Engine::Loudness(LoudnessGate::default()),
            segmenter: Segmenter::new(channel),
            leftover: Vec::with_capacity(WINDOW_SAMPLES * 2),
            next_t_ms: 0,
            anchored: false,
            speaking: false,
            engine_failures: 0,
        }
    }

    /// Load the real detector if the asset is there, fall back if it is not.
    pub fn load_or_fallback(model_path: Option<&Path>, channel: Channel) -> Self {
        match model_path {
            Some(path) if path.exists() => match Self::load(path, channel) {
                Ok(d) => d,
                Err(e) => {
                    tracing::warn!(
                        target: "echo::audio",
                        "falling back to loudness-based speech detection: {e}"
                    );
                    Self::without_model(channel)
                }
            },
            _ => Self::without_model(channel),
        }
    }

    /// True when this detector is using the downloaded model rather than the
    /// loudness fallback. Settings → Advanced only.
    pub fn uses_model(&self) -> bool {
        matches!(self.engine, Engine::Silero(_))
    }

    /// Feed 16 kHz mono audio starting at `t_start_ms`. Returns any utterances
    /// that finished inside this call.
    pub fn push(&mut self, samples: &[f32], t_start_ms: i64) -> Vec<Utterance> {
        let mut out = Vec::new();
        if !self.anchored {
            self.segmenter.rebase(t_start_ms);
            self.next_t_ms = t_start_ms;
            self.anchored = true;
        } else {
            // A hole in the stream: re-anchor rather than let every later
            // utterance inherit the error.
            let expected = self.next_t_ms + samples_to_ms(self.leftover.len());
            if (t_start_ms - expected).abs() > WINDOW_MS {
                if let Some(u) = self.segmenter.rebase(t_start_ms) {
                    out.push(u);
                }
                self.leftover.clear();
                self.next_t_ms = t_start_ms;
            }
        }

        self.leftover.extend_from_slice(samples);
        let mut consumed = 0usize;
        let mut window = vec![0.0f32; WINDOW_SAMPLES];
        while consumed + WINDOW_SAMPLES <= self.leftover.len() {
            window.copy_from_slice(&self.leftover[consumed..consumed + WINDOW_SAMPLES]);
            let is_speech = self.decide(&window);
            out.extend(self.segmenter.push_window(&window, is_speech));
            consumed += WINDOW_SAMPLES;
        }
        if consumed > 0 {
            self.leftover.drain(..consumed);
            self.next_t_ms += samples_to_ms(consumed);
        }
        self.speaking = self.segmenter.is_speaking();
        out
    }

    fn decide(&mut self, window: &[f32]) -> bool {
        let probability = match &mut self.engine {
            Engine::Loudness(gate) => gate.probability(window),
            Engine::Silero(silero) => match silero.probability(window) {
                Ok(p) => p,
                Err(e) => {
                    self.engine_failures += 1;
                    if self.engine_failures == 1 {
                        tracing::warn!(
                            target: "echo::audio",
                            "speech detection failed, falling back to loudness: {e}"
                        );
                    }
                    if self.engine_failures >= 3 {
                        self.engine = Engine::Loudness(LoudnessGate::default());
                    }
                    return rms(window) > 0.01;
                }
            },
        };
        if self.speaking {
            probability > SPEECH_EXIT
        } else {
            probability > SPEECH_ENTER
        }
    }

    /// End of stream: emit whatever is still open.
    pub fn finish(&mut self) -> Option<Utterance> {
        // Whatever did not fill a window is padding at worst; the audio on disk
        // still has it.
        self.leftover.clear();
        let done = self.segmenter.finish();
        self.speaking = false;
        done
    }

    /// Is speech happening right now? Drives the live "someone is talking"
    /// indicator.
    pub fn is_speaking(&self) -> bool {
        self.speaking
    }

    /// Release the session and its memory (mantra 1).
    pub fn unload(self) {
        drop(self);
    }
}

/// Run detection over audio already on disk. This is how the catch-up pass
/// finds utterances it missed while live.
pub async fn detect_offline(
    model_path: &Path,
    samples: &[f32],
    t_offset_ms: i64,
    channel: Channel,
) -> Result<Vec<Utterance>, AudioError> {
    let model_path = model_path.to_path_buf();
    let samples = samples.to_vec();
    tokio::task::spawn_blocking(move || {
        let mut detector = SpeechDetector::load_or_fallback(Some(&model_path), channel);
        let mut out = Vec::new();
        // Half-second pieces: big enough to be cheap, small enough that a very
        // long recording does not sit in one allocation twice.
        let step = TARGET_SAMPLE_RATE as usize / 2;
        for (i, piece) in samples.chunks(step).enumerate() {
            let t = t_offset_ms + samples_to_ms(i * step);
            out.extend(detector.push(piece, t));
        }
        out.extend(detector.finish());
        Ok(out)
    })
    .await
    .map_err(|e| AudioError::Backend(e.to_string()))?
}

// ---------------------------------------------------------------------------
// The bounded queue between detection and the speech engine
// ---------------------------------------------------------------------------

/// How many utterances may wait for text before live work starts being dropped.
pub const DEFAULT_QUEUE_CAPACITY: usize = 24;

/// Bounded queue of work for the speech engine.
///
/// On overflow the *oldest* live job is dropped, never the newest: audio on disk
/// is authoritative, and the catch-up pass will transcribe whatever live
/// transcription could not keep up with. Dropping the oldest keeps the visible
/// transcript close to what is being said right now, which is the only thing the
/// live view is for.
#[derive(Debug)]
pub struct UtteranceQueue {
    inner: Mutex<VecDeque<Utterance>>,
    ready: Condvar,
    capacity: usize,
    dropped: AtomicU64,
    closed: std::sync::atomic::AtomicBool,
}

impl UtteranceQueue {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(VecDeque::with_capacity(capacity.max(1))),
            ready: Condvar::new(),
            capacity: capacity.max(1),
            dropped: AtomicU64::new(0),
            closed: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Queue an utterance. Returns false when something older had to be dropped
    /// to make room, so the caller can tell the UI "the transcript is catching
    /// up" (never a number, mantra 2).
    pub fn push(&self, utterance: Utterance) -> bool {
        let mut queue = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut lost = false;
        while queue.len() >= self.capacity {
            queue.pop_front();
            self.dropped.fetch_add(1, Ordering::Relaxed);
            lost = true;
        }
        queue.push_back(utterance);
        self.ready.notify_one();
        !lost
    }

    pub fn pop(&self) -> Option<Utterance> {
        let mut queue = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        queue.pop_front()
    }

    /// Wait up to `timeout` for work. Returns `None` on timeout or once closed.
    pub fn pop_timeout(&self, timeout: std::time::Duration) -> Option<Utterance> {
        let mut queue = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if queue.is_empty() && !self.is_closed() {
            let (next, _) = self
                .ready
                .wait_timeout(queue, timeout)
                .unwrap_or_else(|e| e.into_inner());
            queue = next;
        }
        queue.pop_front()
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// No more work is coming; wake anything waiting.
    pub fn close(&self) {
        self.closed.store(true, Ordering::Relaxed);
        self.ready.notify_all();
    }

    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Relaxed)
    }
}

impl Default for UtteranceQueue {
    fn default() -> Self {
        Self::new(DEFAULT_QUEUE_CAPACITY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: usize = TARGET_SAMPLE_RATE as usize;

    fn silence(ms: usize) -> Vec<f32> {
        vec![0.0; RATE * ms / 1_000]
    }

    fn tone(ms: usize, hz: f32, amp: f32) -> Vec<f32> {
        let n = RATE * ms / 1_000;
        (0..n)
            .map(|i| (i as f32 / RATE as f32 * hz * std::f32::consts::TAU).sin() * amp)
            .collect()
    }

    /// A little room noise, so the loudness gate has a floor to learn.
    fn room_noise(ms: usize) -> Vec<f32> {
        let n = RATE * ms / 1_000;
        (0..n)
            .map(|i| {
                let x = (i as f32 * 0.37).sin() + (i as f32 * 1.13).sin();
                x * 0.0008
            })
            .collect()
    }

    fn feed(detector: &mut SpeechDetector, audio: &[f32], t0: i64) -> Vec<Utterance> {
        let mut out = Vec::new();
        let step = RATE / 50; // 20 ms, like the capture pipeline
        for (i, piece) in audio.chunks(step).enumerate() {
            out.extend(detector.push(piece, t0 + samples_to_ms(i * step)));
        }
        out
    }

    #[test]
    fn utterance_duration_is_the_window_length() {
        let u = Utterance {
            t_start_ms: 1_000,
            t_end_ms: 4_500,
            ..Default::default()
        };
        assert_eq!(u.duration_ms(), 3_500);
    }

    #[test]
    fn the_utterance_cap_leaves_room_for_padding() {
        const { assert!(MAX_UTTERANCE_MS > PAD_MS * 2) };
        const { assert!(SILENCE_TAIL_MS > 0) };
        // An utterance plus its padding has to fit the speech engine's window.
        const { assert!(MAX_UTTERANCE_MS + PAD_MS * 2 <= 30_000) };
    }

    #[test]
    fn silence_produces_nothing_at_all() {
        let mut d = SpeechDetector::without_model(Channel::Mic);
        let out = feed(&mut d, &silence(5_000), 0);
        assert!(out.is_empty(), "{out:#?}");
        assert!(d.finish().is_none());
        assert!(!d.is_speaking());
    }

    #[test]
    fn quiet_room_noise_is_not_speech() {
        let mut d = SpeechDetector::without_model(Channel::Mic);
        let out = feed(&mut d, &room_noise(5_000), 0);
        assert!(out.is_empty(), "{out:#?}");
    }

    #[test]
    fn one_sentence_between_silences_becomes_one_padded_utterance() {
        let mut d = SpeechDetector::without_model(Channel::Mic);
        let mut audio = room_noise(1_000);
        audio.extend(tone(2_000, 220.0, 0.3));
        audio.extend(room_noise(1_500));

        let mut out = feed(&mut d, &audio, 0);
        out.extend(d.finish());

        assert_eq!(out.len(), 1, "{out:#?}");
        let u = &out[0];
        assert_eq!(u.channel, Channel::Mic);
        assert!(!u.truncated);
        // Speech starts at 1000 ms and the pad reaches back 300 ms.
        assert!(
            (600..=1_050).contains(&u.t_start_ms),
            "utterance started at {}",
            u.t_start_ms
        );
        // It ends at 3000 ms plus the trailing pad.
        assert!(
            (2_950..=3_450).contains(&u.t_end_ms),
            "utterance ended at {}",
            u.t_end_ms
        );
        assert_eq!(
            u.samples.len(),
            ms_to_samples(u.duration_ms()),
            "the audio handed over must match the window it claims"
        );
        // The speech itself is in there, not just padding.
        assert!(rms(&u.samples) > 0.1, "the utterance sounds empty");
    }

    #[test]
    fn two_sentences_with_a_real_gap_become_two_utterances() {
        let mut d = SpeechDetector::without_model(Channel::Mic);
        let mut audio = room_noise(500);
        audio.extend(tone(1_200, 300.0, 0.3));
        audio.extend(room_noise(1_500));
        audio.extend(tone(1_200, 300.0, 0.3));
        audio.extend(room_noise(1_000));

        let mut out = feed(&mut d, &audio, 0);
        out.extend(d.finish());

        assert_eq!(out.len(), 2, "{out:#?}");
        assert!(out[0].t_end_ms <= out[1].t_start_ms, "{out:#?}");
        for u in &out {
            assert!(!u.truncated);
            assert!(u.duration_ms() > 1_000, "{u:#?}");
        }
    }

    #[test]
    fn a_short_pause_inside_a_sentence_does_not_split_it() {
        let mut d = SpeechDetector::without_model(Channel::Mic);
        let mut audio = room_noise(400);
        audio.extend(tone(800, 300.0, 0.3));
        // Shorter than SILENCE_TAIL_MS.
        audio.extend(room_noise(300));
        audio.extend(tone(800, 300.0, 0.3));
        audio.extend(room_noise(1_200));

        let mut out = feed(&mut d, &audio, 0);
        out.extend(d.finish());
        assert_eq!(out.len(), 1, "{out:#?}");
    }

    /// The regression test for the meeting that produced no transcript at all.
    ///
    /// Every utterance handed downstream has to be a window the speech engine
    /// can read. The old segmenter could emit as little as
    /// `2 * PAD_SAMPLES + WINDOW_SAMPLES` — 608 ms — which whisper.cpp refuses
    /// to encode, and every short answer in the meeting landed there.
    #[test]
    fn a_short_answer_is_still_long_enough_for_the_engine_to_read() {
        for burst_ms in [32usize, 100, 200, 300, 400, 700] {
            let mut d = SpeechDetector::without_model(Channel::Mic);
            let mut audio = room_noise(600);
            audio.extend(tone(burst_ms, 300.0, 0.3));
            // Long enough after it for the segmenter to notice speech ended.
            audio.extend(room_noise(2_500));

            let mut out = feed(&mut d, &audio, 0);
            out.extend(d.finish());

            assert_eq!(out.len(), 1, "a {burst_ms} ms answer: {out:#?}");
            let u = &out[0];
            // One second is whisper.cpp's own floor, spelled out here rather
            // than read from MIN_UTTERANCE_MS: the point of the test is that the
            // constant clears the engine's requirement, so it cannot be the
            // thing the requirement is measured against.
            assert!(
                u.duration_ms() >= 1_000,
                "a {burst_ms} ms answer became a {} ms utterance, which the engine cannot read",
                u.duration_ms()
            );
            assert_eq!(
                u.samples.len(),
                ms_to_samples(u.duration_ms()),
                "the audio handed over must match the window it claims"
            );
            // The words themselves are in there, not just the silence we kept.
            assert!(rms(&u.samples) > 0.0, "the utterance came back empty");
        }
    }

    #[test]
    fn the_shortest_window_the_segmenter_can_emit_is_one_the_engine_can_read() {
        // A single voiced window used to be the worst case: one pad either side
        // and 32 ms of speech.
        let floor_ms = samples_to_ms(PAD_SAMPLES * 2 + WINDOW_SAMPLES);
        assert!(
            floor_ms < 1_000,
            "this test exists because the natural floor ({floor_ms} ms) is under a second"
        );
        const {
            assert!(
                MIN_UTTERANCE_MS > 1_000,
                "the floor has to clear whisper.cpp's one-second window, not just touch it"
            )
        };
    }

    #[test]
    fn five_seconds_of_talking_is_one_utterance_not_a_hundred() {
        let mut d = SpeechDetector::without_model(Channel::Mic);
        let mut audio = room_noise(500);
        audio.extend(tone(5_000, 300.0, 0.3));
        audio.extend(room_noise(1_500));

        let mut out = feed(&mut d, &audio, 0);
        out.extend(d.finish());

        assert_eq!(out.len(), 1, "{out:#?}");
        assert!(
            out[0].duration_ms() >= 5_000,
            "five seconds of speech came back as {} ms",
            out[0].duration_ms()
        );
        assert!(!out[0].truncated);
    }

    #[test]
    fn micro_pauses_between_words_do_not_shred_a_sentence() {
        let mut d = SpeechDetector::without_model(Channel::Mic);
        let mut audio = room_noise(500);
        // Six words with 300 ms of breath between them: half the silence tail,
        // so none of them is an utterance of its own.
        for _ in 0..6 {
            audio.extend(tone(400, 300.0, 0.3));
            audio.extend(room_noise(300));
        }
        audio.extend(room_noise(1_500));

        let mut out = feed(&mut d, &audio, 0);
        out.extend(d.finish());

        assert_eq!(out.len(), 1, "{out:#?}");
        assert!(
            out[0].duration_ms() >= 4_000,
            "the sentence came back as {} ms",
            out[0].duration_ms()
        );
    }

    #[test]
    fn a_blip_on_its_own_is_never_handed_over_as_an_unreadable_sliver() {
        // A door, a keystroke, one syllable: whatever it is, the segmenter must
        // hand over a readable window or nothing at all.
        let mut d = SpeechDetector::without_model(Channel::Mic);
        let mut audio = room_noise(400);
        audio.extend(tone(200, 300.0, 0.3));

        // Nothing after it: the recording ends right there.
        let mut out = feed(&mut d, &audio, 0);
        out.extend(d.finish());

        assert!(out.len() <= 1, "{out:#?}");
        for u in &out {
            assert_eq!(u.samples.len(), ms_to_samples(u.duration_ms()));
            // End-of-stream is the one exit that can be short, and the engine
            // pads it: what must never happen is a *silent* zero-length window.
            assert!(!u.samples.is_empty());
        }
    }

    #[test]
    fn a_monologue_is_cut_into_pieces_the_engine_will_accept() {
        let mut d = SpeechDetector::without_model(Channel::Mic);
        let mut audio = room_noise(200);
        audio.extend(tone(70_000, 300.0, 0.3));
        audio.extend(room_noise(1_000));

        let mut out = feed(&mut d, &audio, 0);
        out.extend(d.finish());

        assert!(out.len() >= 3, "70 s of speech became {} pieces", out.len());
        for u in &out {
            assert!(
                u.duration_ms() <= MAX_UTTERANCE_MS + PAD_MS,
                "a piece is {} ms long",
                u.duration_ms()
            );
        }
        // Every cut but the last was forced.
        assert!(out[0].truncated);
        assert!(!out[out.len() - 1].truncated);
        // The pieces tile the monologue without overlapping.
        for pair in out.windows(2) {
            assert_eq!(
                pair[0].t_end_ms, pair[1].t_start_ms,
                "{:#?} then {:#?}",
                pair[0], pair[1]
            );
        }
    }

    #[test]
    fn utterances_carry_the_offset_they_were_given() {
        let mut d = SpeechDetector::without_model(Channel::System);
        let mut audio = room_noise(500);
        audio.extend(tone(1_000, 300.0, 0.3));
        audio.extend(room_noise(1_200));

        let mut out = feed(&mut d, &audio, 600_000);
        out.extend(d.finish());
        assert_eq!(out.len(), 1);
        assert!(
            (600_000..=601_000).contains(&out[0].t_start_ms),
            "{:#?}",
            out[0]
        );
        assert_eq!(out[0].channel, Channel::System);
    }

    #[test]
    fn a_gap_in_the_stream_does_not_slide_later_utterances() {
        let mut d = SpeechDetector::without_model(Channel::Mic);
        let mut out = feed(&mut d, &room_noise(500), 0);
        // The device went away for a minute; capture resumes at a new offset.
        let mut later = room_noise(300);
        later.extend(tone(1_000, 300.0, 0.3));
        later.extend(room_noise(1_200));
        out.extend(feed(&mut d, &later, 60_000));
        out.extend(d.finish());

        assert_eq!(out.len(), 1, "{out:#?}");
        assert!(
            (59_900..=61_500).contains(&out[0].t_start_ms),
            "{:#?}",
            out[0]
        );
    }

    #[test]
    fn without_the_model_the_detector_says_so() {
        let d = SpeechDetector::without_model(Channel::Mic);
        assert!(!d.uses_model());
        // A missing asset is not an error: recording never waits on a download.
        let d =
            SpeechDetector::load_or_fallback(Some(Path::new("/nope/absent.onnx")), Channel::Mic);
        assert!(!d.uses_model());
        let d = SpeechDetector::load_or_fallback(None, Channel::Mic);
        assert!(!d.uses_model());
    }

    #[test]
    fn speaking_is_true_only_while_speech_is_happening() {
        let mut d = SpeechDetector::without_model(Channel::Mic);
        feed(&mut d, &room_noise(400), 0);
        assert!(!d.is_speaking());
        feed(&mut d, &tone(400, 300.0, 0.3), 400);
        assert!(d.is_speaking());
        feed(&mut d, &room_noise(1_500), 800);
        assert!(!d.is_speaking());
    }

    #[test]
    fn the_queue_drops_the_oldest_live_job_when_it_overflows() {
        let q = UtteranceQueue::new(3);
        for i in 0..3 {
            assert!(q.push(Utterance {
                t_start_ms: i * 1_000,
                ..Default::default()
            }));
        }
        assert_eq!(q.len(), 3);
        assert_eq!(q.dropped(), 0);

        assert!(!q.push(Utterance {
            t_start_ms: 3_000,
            ..Default::default()
        }));
        assert_eq!(q.len(), 3);
        assert_eq!(q.dropped(), 1);
        // The oldest went, the newest stayed.
        assert_eq!(q.pop().unwrap().t_start_ms, 1_000);
        assert_eq!(q.pop().unwrap().t_start_ms, 2_000);
        assert_eq!(q.pop().unwrap().t_start_ms, 3_000);
        assert!(q.pop().is_none());
    }

    #[test]
    fn waiting_on_a_closed_queue_returns_instead_of_hanging() {
        let q = UtteranceQueue::new(2);
        q.close();
        assert!(q
            .pop_timeout(std::time::Duration::from_millis(50))
            .is_none());
        assert!(q.is_closed());
    }

    /// The fixture's contents, as `tests/fixtures/README.md` describes them.
    ///
    /// Kept here as well as in the Python generator so the test can stand on its
    /// own: `*.wav` is gitignored at the repo root, so a fresh clone may not have
    /// the file, and a test that fails for that reason would be worse than no
    /// test at all.
    fn fixture_audio() -> Vec<f32> {
        let mut audio = Vec::new();
        let mut pos = 0usize;
        for (voice, ms) in [
            (false, 1_000usize),
            (true, 2_000),
            (false, 1_500),
            (true, 1_500),
            (false, 1_000),
        ] {
            let n = RATE * ms / 1_000;
            for i in pos..pos + n {
                audio.push(if voice {
                    let t = i as f32 / RATE as f32;
                    let tau = std::f32::consts::TAU;
                    let s = (tau * 140.0 * t).sin() * 0.45
                        + (tau * 280.0 * t).sin() * 0.22
                        + (tau * 560.0 * t).sin() * 0.10;
                    s * (0.85 + 0.15 * (tau * 4.5 * t).sin()) * 0.6
                } else {
                    ((i as f32 * 0.37).sin() + (i as f32 * 1.13).sin()) * 0.0008
                });
            }
            pos += n;
        }
        audio
    }

    /// The one test here that reads a real file: it proves the WAV reader, the
    /// resampler and segmentation agree about something none of them produced in
    /// memory. Falls back to writing the fixture itself when the checked-out tree
    /// does not have it.
    #[test]
    fn the_fixture_on_disk_segments_where_its_layout_says_it_should() {
        let committed =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/speech-and-silence-16k.wav");
        let temp = tempfile::tempdir().unwrap();
        let path = if committed.exists() {
            committed
        } else {
            let generated = temp.path().join("speech-and-silence-16k.wav");
            crate::audio::writer::write_wav_16k_mono(&generated, &fixture_audio()).unwrap();
            generated
        };

        let audio = crate::audio::writer::read_wav_16k_mono(&path)
            .expect("the capture fixture must be readable");
        assert_eq!(audio.len(), RATE * 7, "the fixture is not 7 seconds long");

        let mut d = SpeechDetector::without_model(Channel::Mic);
        let mut out = feed(&mut d, &audio, 0);
        out.extend(d.finish());

        assert_eq!(out.len(), 2, "{out:#?}");
        // Stretch A: speech from 1000 ms to 3000 ms, padded both sides.
        assert!((650..=1_050).contains(&out[0].t_start_ms), "{:#?}", out[0]);
        assert!((2_950..=3_450).contains(&out[0].t_end_ms), "{:#?}", out[0]);
        // Stretch B: speech from 4500 ms to 6000 ms.
        assert!(
            (4_150..=4_550).contains(&out[1].t_start_ms),
            "{:#?}",
            out[1]
        );
        assert!((5_950..=6_450).contains(&out[1].t_end_ms), "{:#?}", out[1]);
        for u in &out {
            assert!(!u.truncated);
            assert!(rms(&u.samples) > 0.1, "an utterance came back silent");
        }
    }

    #[test]
    fn offline_detection_finds_the_same_speech() {
        let mut audio = room_noise(500);
        audio.extend(tone(1_500, 300.0, 0.3));
        audio.extend(room_noise(1_500));

        let found = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(detect_offline(
                Path::new("/nope/absent.onnx"),
                &audio,
                10_000,
                Channel::Mic,
            ))
            .unwrap();
        assert_eq!(found.len(), 1, "{found:#?}");
        assert!(
            (9_900..=11_000).contains(&found[0].t_start_ms),
            "{found:#?}"
        );
    }
}
