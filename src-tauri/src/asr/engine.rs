//! The speech engine: whisper-rs (whisper.cpp) in process.
//!
//! IMPLEMENTED-BY: asr agent (M3), informed by spike M0-S3.
//!
//! Build and runtime (DESIGN §2, review finding 1):
//! * The macOS binary compiles the `metal` and `coreml` features; the Linux
//!   binary compiles `vulkan`. Backends are compile-time, so there is one binary
//!   per platform.
//! * At runtime we try to initialise the GPU backend and fall back to CPU if
//!   that fails, recording the reason for Settings → Advanced.
//! * `n_threads` is clamped and defaults low: a meeting recording must stay
//!   responsive, so we leave cores for capture.
//!
//! Concurrency: one engine, one job at a time. The queue is bounded. When it
//! overflows we drop live jobs and let the catch-up pass read from disk
//! (mantra 3, review finding 17).
//!
//! Shape of the thing: the weights live on **one dedicated OS thread**, never on
//! the async runtime. whisper.cpp is a long blocking call that saturates the
//! cores it is given; putting it on a Tokio worker would stall capture and IPC.
//! The async side owns only a bounded channel and some atomics.
//!
//! Lazy by construction (mantra 1): [`EngineWorker::new`] starts a thread that
//! holds nothing. The weights are read on the first job, or on an explicit
//! pre-warm, and dropped again after
//! [`EngineWorker`]'s idle period with no work.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::{Arc, Mutex, Once};
use std::time::{Duration, Instant};

use whisper_rs::{
    FullParams, SamplingStrategy, SegmentCallbackData, WhisperContext, WhisperContextParameters,
    WhisperState,
};

use crate::asr::catalog::{self, DecodeParams};
use crate::asr::language::{Decision, LanguagePolicy};
use crate::asr::{models, AsrError, TranscribeJob, Transcription};
use crate::db::{repo, Db};
use crate::types::AssetKind;

/// Called with partial text while a job decodes. Cheap and non-blocking; the
/// caller rate-caps before it reaches the UI.
pub type PartialFn = Box<dyn Fn(&str) + Send + Sync>;

/// The same thing once it is inside the engine. whisper.cpp keeps the callback
/// we install for longer than the call frame, so it has to be shared rather than
/// owned.
type SharedPartial = Arc<dyn Fn(&str) + Send + Sync>;

/// How many utterances may wait for text. Small on purpose: a deep queue means
/// live captions minutes behind the meeting, which is worse than no captions.
/// Overflow drops *live* work only; the audio is on disk (mantra 3).
pub const QUEUE_CAPACITY: usize = 8;

/// Default idle period before the weights are handed back, when the caller does
/// not say. Ten minutes is long enough to cover the gap between two
/// back-to-back meetings and short enough that an idle Echo is empty.
pub const DEFAULT_IDLE_RELEASE_MINUTES: u32 = 10;

/// Cores kept clear of decoding, so capture and the UI never starve.
const CORES_RESERVED_FOR_CAPTURE: usize = 2;

/// Upper bound on decoding threads. Past this, whisper.cpp scales badly and we
/// are only making the machine hot.
const MAX_DECODE_THREADS: usize = 8;

/// Never fewer than this, however small the computer.
const MIN_DECODE_THREADS: usize = 2;

/// Below this there is no speech to find, only a click or a keystroke. Skipped
/// as "nothing was said", which is an answer, not a failure.
const MIN_JOB_MS: i64 = 80;

/// Shortest window whisper.cpp will actually encode, plus a margin.
///
/// whisper.cpp's own guard only rejects audio under 100 ms outright
/// (`whisper_full_with_state`: *"input is too short — %d ms < 100 ms. consider
/// padding the input audio with silence"*). Between that and a second, the mel
/// it builds is short enough that `whisper_encode_internal` fails, and
/// `whisper_full_with_state` turns that into `return -6` — the
/// `Generic whisper error … Error code: -6` a real meeting produced dozens of
/// times. Taking whisper.cpp's own advice and padding the tail out to this
/// length makes that return unreachable for any non-empty input.
///
/// Timestamps are never taken from the padding: [`Engine::transcribe`] clamps
/// whisper's end-of-speech to the real audio it was handed.
const MIN_DECODE_MS: i64 = 1_100;

/// [`MIN_DECODE_MS`] in 16 kHz mono samples.
const MIN_DECODE_SAMPLES: usize =
    (MIN_DECODE_MS as usize) * (crate::audio::TARGET_SAMPLE_RATE as usize) / 1_000;

/// Whisper reports timestamps in centiseconds.
const CS_TO_MS: i64 = 10;

/// Whisper marks silence with bracketed pseudo-words. They are not speech and
/// must never reach a transcript.
const NOISE_MARKERS: &[&str] = &[
    "[BLANK_AUDIO]",
    "[ Silence ]",
    "[silence]",
    "(silence)",
    "[MUSIC]",
    "[Music]",
    "(music)",
    "[ Pause ]",
    "[INAUDIBLE]",
    "[ Inaudible ]",
];

/// whisper.cpp and GGML write to stderr by default. Route them through our hooks
/// once so a desktop app does not spray a terminal it does not have.
fn quieten_whisper() {
    static ONCE: Once = Once::new();
    ONCE.call_once(whisper_rs::install_logging_hooks);
}

// ---------------------------------------------------------------------------
// Backend selection
// ---------------------------------------------------------------------------

/// Which backend actually came up, for Settings → Advanced.
#[derive(Debug, Clone, Default)]
pub struct BackendReport {
    /// Compiled-in backends, e.g. "metal+coreml".
    pub compiled: String,
    /// What initialised, e.g. "metal" or "cpu".
    pub active: String,
    /// Why the GPU was not used, when it was not.
    pub fallback_reason: Option<String>,
    pub threads: u32,
    /// Whether the Apple encoder companion was found next to the weights. It
    /// only affects speed (review finding 1).
    pub accelerator_present: bool,
    /// Verbatim whisper.cpp build banner. Technical, diagnostics only.
    pub details: String,
}

/// The backends this binary was built with. Compile-time, per DESIGN §2.
pub fn compiled_backends() -> &'static str {
    if cfg!(target_os = "macos") {
        "metal+coreml"
    } else if cfg!(target_os = "linux") {
        "vulkan"
    } else {
        "cpu"
    }
}

/// The GPU backend this platform would try.
fn gpu_backend_name() -> Option<&'static str> {
    if cfg!(target_os = "macos") {
        Some("metal")
    } else if cfg!(target_os = "linux") {
        Some("vulkan")
    } else {
        None
    }
}

/// Decoding threads: leave [`CORES_RESERVED_FOR_CAPTURE`] cores for capture and
/// the UI, cap at [`MAX_DECODE_THREADS`], never go below
/// [`MIN_DECODE_THREADS`].
///
/// Physical cores, not logical: hyperthreads do not help a memory-bound decoder
/// and they do take time away from the audio callbacks.
pub fn recommended_threads() -> u32 {
    let physical = sysinfo::System::physical_core_count().unwrap_or(0);
    let cores = if physical > 0 {
        physical
    } else {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(MIN_DECODE_THREADS)
    };
    clamp_threads(cores)
}

fn clamp_threads(cores: usize) -> u32 {
    cores
        .saturating_sub(CORES_RESERVED_FOR_CAPTURE)
        .clamp(MIN_DECODE_THREADS, MAX_DECODE_THREADS) as u32
}

/// Beam search or greedy, straight from the preset. This *is* the difference
/// between the quality presets, together with which weights they load.
fn sampling_strategy(decode: &DecodeParams) -> SamplingStrategy {
    if decode.uses_beam_search() {
        SamplingStrategy::BeamSearch {
            beam_size: decode.beam_size as i32,
            patience: -1.0,
        }
    } else {
        SamplingStrategy::Greedy {
            best_of: decode.best_of.max(1) as i32,
        }
    }
}

// ---------------------------------------------------------------------------
// What to load
// ---------------------------------------------------------------------------

/// Everything the engine needs to come up. Assembled from the selected quality
/// preset and the `models` table, so provenance travels with the weights.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineConfig {
    pub model_path: PathBuf,
    /// Apple encoder companion. whisper.cpp finds it by name next to
    /// `model_path`; this is only here so we can report whether it is there.
    pub accelerator_path: Option<PathBuf>,
    pub decode: DecodeParams,
    /// Written onto every segment (DESIGN §3 Provenance).
    pub model_name: String,
    pub model_revision: String,
}

impl EngineConfig {
    /// Read the selected preset and the installed files out of the database.
    ///
    /// Fails with [`AsrError::NotInstalled`] rather than loading something else:
    /// silently transcribing with different weights than the person chose would
    /// make the provenance on every segment a lie.
    pub async fn from_settings(db: &Db) -> Result<Self, AsrError> {
        let level_id = repo::get_setting(db, crate::settings::keys::ACCURACY_LEVEL_ID)
            .await?
            .filter(|v| !v.is_empty())
            .unwrap_or_default();
        let preset = catalog::preset_or_default(&level_id);

        let model_path = models::installed_path(db, AssetKind::Speech)
            .await?
            .ok_or(AsrError::NotInstalled)?;
        let accelerator_path = models::installed_path(db, AssetKind::SpeechAccelerator).await?;

        let row = repo::get_model(db, preset.speech_id).await?;
        Ok(Self {
            model_path,
            accelerator_path,
            decode: preset.decode,
            model_name: row
                .as_ref()
                .map(|r| r.name.clone())
                .unwrap_or_else(|| preset.speech_id.to_string()),
            model_revision: row
                .and_then(|r| r.revision)
                .unwrap_or_else(|| catalog::CATALOG_REVISION.to_string()),
        })
    }
}

// ---------------------------------------------------------------------------
// The engine
// ---------------------------------------------------------------------------

/// Live text for the job in flight. Owned by the [`Engine`] and reused, because
/// whisper.cpp keeps the callback we hand it and we would otherwise leak a fresh
/// closure — and everything it captured — once per utterance.
#[derive(Default)]
struct SinkState {
    /// Where live text goes for the current job. `None` = partials not wanted.
    out: Option<SharedPartial>,
    /// Text built up so far, so each partial is the whole utterance rather than
    /// the newest fragment.
    text: String,
}

#[derive(Default)]
struct PartialSink(Mutex<SinkState>);

impl PartialSink {
    fn begin(&self, out: Option<SharedPartial>) {
        let mut s = self.0.lock().expect("partial sink poisoned");
        s.out = out;
        s.text.clear();
    }

    fn push(&self, fragment: &str) {
        let mut s = self.0.lock().expect("partial sink poisoned");
        s.text.push_str(fragment);
        if let Some(out) = &s.out {
            out(s.text.trim());
        }
    }

    fn end(&self) {
        let mut s = self.0.lock().expect("partial sink poisoned");
        s.out = None;
        s.text.clear();
    }
}

/// A loaded speech engine. Dropping it frees the weights.
pub struct Engine {
    ctx: WhisperContext,
    state: WhisperState,
    backend: BackendReport,
    config: EngineConfig,
    sink: Arc<PartialSink>,
    /// Set from outside to stop the decode in flight.
    abort: Arc<AtomicBool>,
    /// Special-token boundary; tokens at or above this are not words.
    first_special_token: i32,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("backend", &self.backend)
            .field("model", &self.config.model_path)
            .finish()
    }
}

impl Engine {
    /// Load the engine from the configured weights.
    ///
    /// Tries the GPU backend first and falls back to CPU, recording why. The
    /// Apple encoder companion is found by whisper.cpp itself, by name, next to
    /// the weights; a missing one only costs speed.
    ///
    /// Blocking on purpose: this reads over a gigabyte from disk and is only
    /// ever called on the engine thread.
    pub fn load(config: EngineConfig, abort: Arc<AtomicBool>) -> Result<Self, AsrError> {
        quieten_whisper();

        if !config.model_path.is_file() {
            return Err(AsrError::NotInstalled);
        }
        let threads = recommended_threads();
        let accelerator_present = config
            .accelerator_path
            .as_deref()
            .is_some_and(accelerator_is_usable);
        if config.accelerator_path.is_some() && !accelerator_present {
            tracing::warn!("the encoder companion is not usable; continuing without it");
        }

        let mut fallback_reason = None;
        let mut active = "cpu".to_string();
        let mut ctx = None;

        if let Some(gpu) = gpu_backend_name() {
            let mut params = WhisperContextParameters::default();
            params.use_gpu(true);
            match WhisperContext::new_with_params(&config.model_path, params) {
                Ok(c) => {
                    active = gpu.to_string();
                    ctx = Some(c);
                }
                Err(e) => {
                    // Not fatal: a machine without a working GPU stack still
                    // has to be able to write down a meeting.
                    tracing::warn!(%e, backend = gpu, "falling back to the processor");
                    fallback_reason = Some(e.to_string());
                }
            }
        } else {
            fallback_reason = Some("this build has no graphics backend".to_string());
        }

        let ctx = match ctx {
            Some(c) => c,
            None => {
                let mut params = WhisperContextParameters::default();
                params.use_gpu(false);
                WhisperContext::new_with_params(&config.model_path, params)
                    .map_err(|e| AsrError::Load(e.to_string()))?
            }
        };

        let state = ctx
            .create_state()
            .map_err(|e| AsrError::Load(e.to_string()))?;
        let first_special_token = ctx.token_eot();

        let backend = BackendReport {
            compiled: compiled_backends().to_string(),
            active,
            fallback_reason,
            threads,
            accelerator_present,
            details: whisper_rs::print_system_info().to_string(),
        };
        tracing::info!(
            backend = %backend.active,
            threads = backend.threads,
            accelerator = backend.accelerator_present,
            "speech engine loaded"
        );

        Ok(Self {
            ctx,
            state,
            backend,
            config,
            sink: Arc::new(PartialSink::default()),
            abort,
            first_special_token,
        })
    }

    /// Transcribe one utterance. Serialised: the caller holds the only handle.
    ///
    /// Transcription only — [`FullParams::set_translate`] is always false, so a
    /// French meeting stays French (DESIGN §1: "transcribe locally, automatic
    /// language detection", never translate).
    pub fn transcribe(
        &mut self,
        job: &TranscribeJob,
        on_partial: Option<PartialFn>,
    ) -> Result<Transcription, AsrError> {
        let duration_ms = job.duration_ms();
        let mut result = Transcription {
            channel: job.channel,
            t_start_ms: job.t_start_ms,
            t_end_ms: job.t_start_ms + duration_ms,
            language: job.language_hint.clone(),
            model_name: Some(self.config.model_name.clone()),
            model_revision: Some(self.config.model_revision.clone()),
            ..Default::default()
        };
        if duration_ms < MIN_JOB_MS {
            return Ok(result);
        }

        let mut params = FullParams::new(sampling_strategy(&self.config.decode));
        params.set_n_threads(self.backend.threads as i32);
        // Transcription, never translation.
        params.set_translate(false);
        // Each utterance stands alone. Carrying decoder context across a silence
        // is how whisper starts repeating itself for ever.
        params.set_no_context(true);
        params.set_single_segment(false);
        params.set_token_timestamps(false);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);
        params.set_temperature(self.config.decode.temperature);
        params.set_temperature_inc(self.config.decode.temperature_inc);
        params.set_entropy_thold(self.config.decode.entropy_thold);
        params.set_logprob_thold(self.config.decode.logprob_thold);
        params.set_no_speech_thold(self.config.decode.no_speech_thold);

        match job.language_hint.as_deref() {
            Some(lang) => params.set_language(Some(lang)),
            None => {
                params.set_language(Some("auto"));
                params.set_detect_language(true);
            }
        }

        self.sink.begin(if job.want_partials {
            on_partial.map(SharedPartial::from)
        } else {
            None
        });
        if job.want_partials {
            let sink = self.sink.clone();
            params.set_segment_callback_safe_lossy(move |data: SegmentCallbackData| {
                sink.push(&data.text);
            });
        }
        self.abort.store(false, Ordering::SeqCst);
        let abort = self.abort.clone();
        params.set_abort_callback_safe(move || abort.load(Ordering::Relaxed));

        // Never the raw utterance: whisper.cpp cannot encode a sub-second window
        // and answers -6 instead of text.
        let audio = padded_for_decode(&job.samples);
        let outcome = self.state.full(params, audio.as_ref());
        self.sink.end();
        if self.abort.load(Ordering::SeqCst) {
            return Err(AsrError::Cancelled);
        }
        if let Err(e) = outcome {
            // A failed encode leaves whisper.cpp's state half-built, and this
            // engine keeps one state for its whole life. Without this, the first
            // bad utterance made every later one fail the same way — which is
            // exactly how one meeting ended up with zero segments.
            self.reset_state();
            return Err(AsrError::Transcribe(e.to_string()));
        }

        // --- collect ------------------------------------------------------
        let mut text = String::new();
        let mut prob_sum = 0.0f64;
        let mut prob_count = 0usize;
        let mut last_end_ms: Option<i64> = None;
        let n = self.state.full_n_segments();
        for i in 0..n {
            let Some(segment) = self.state.get_segment(i) else {
                continue;
            };
            let piece = segment.to_str_lossy().unwrap_or_default();
            let piece = strip_noise_markers(piece.trim());
            if !piece.is_empty() {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(&piece);
            }
            for t in 0..segment.n_tokens() {
                let Some(token) = segment.get_token(t) else {
                    continue;
                };
                // Timestamps, language tags and the like carry probabilities of
                // their own; averaging them in tells us nothing about the words.
                if token.token_id() >= self.first_special_token {
                    continue;
                }
                prob_sum += f64::from(token.token_probability());
                prob_count += 1;
            }
            last_end_ms = Some(segment.end_timestamp() * CS_TO_MS);
        }

        result.text = text;
        if prob_count > 0 {
            result.avg_confidence = Some((prob_sum / prob_count as f64) as f32);
        }
        if job.language_hint.is_none() {
            result.language =
                whisper_rs::get_lang_str(self.state.full_lang_id_from_state()).map(str::to_string);
        }
        // Trust whisper's own end-of-speech over the padded window length, but
        // never let it run past the audio we handed it.
        if let Some(end) = last_end_ms {
            let clamped = end.clamp(0, duration_ms);
            if clamped > 0 {
                result.t_end_ms = job.t_start_ms + clamped;
            }
        }
        Ok(result)
    }

    /// Detect the language from the first stretch of clear speech. Called once
    /// per meeting, then the answer becomes the hint for later jobs
    /// (review finding 21).
    pub fn detect_language(&mut self, samples: &[f32]) -> Result<(String, f32), AsrError> {
        let threads = self.backend.threads as usize;
        // Same window rule as decoding: `whisper_lang_auto_detect_with_state`
        // runs the encoder too, and answers -6 on a window that is too short.
        let audio = padded_for_decode(samples);
        self.state
            .pcm_to_mel(audio.as_ref(), threads)
            .map_err(|e| AsrError::Transcribe(e.to_string()))?;
        let detected = self.state.lang_detect(0, threads);
        let (id, probabilities) = match detected {
            Ok(answer) => answer,
            Err(e) => {
                self.reset_state();
                return Err(AsrError::Transcribe(e.to_string()));
            }
        };
        let language = whisper_rs::get_lang_str(id)
            .ok_or_else(|| AsrError::Transcribe(format!("unknown language id {id}")))?;
        let confidence = probabilities
            .get(id.max(0) as usize)
            .copied()
            .unwrap_or(0.0);
        Ok((language.to_string(), confidence))
    }

    /// Throw away the decoder scratch state and build a fresh one.
    ///
    /// The weights are in the context and are not touched, so this is cheap — no
    /// re-read from disk, no second GPU init. Only the per-decode buffers go.
    ///
    /// Called after any failed decode. whisper.cpp does not unwind a half-built
    /// graph, so a state that has failed once tends to fail for ever, and this
    /// engine holds one state for its whole life. Failing to rebuild is not
    /// fatal: the old state is kept and the next utterance gets one more try.
    fn reset_state(&mut self) {
        match self.ctx.create_state() {
            Ok(fresh) => {
                self.state = fresh;
                tracing::debug!("rebuilt the speech engine's working state after a failed decode");
            }
            Err(e) => tracing::warn!(%e, "could not rebuild the speech engine's working state"),
        }
    }

    pub fn backend(&self) -> BackendReport {
        self.backend.clone()
    }

    /// Identifier and revision written onto every segment for provenance.
    pub fn provenance(&self) -> (String, String) {
        (
            self.config.model_name.clone(),
            self.config.model_revision.clone(),
        )
    }

    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Keeps the context field meaningfully used and gives diagnostics the
    /// whisper.cpp build the weights were loaded by.
    pub fn is_multilingual(&self) -> bool {
        self.ctx.is_multilingual()
    }
}

/// Zero-pad the tail so whisper.cpp always gets a window it can encode.
///
/// Silence at the end costs nothing: the decoder is told to suppress blanks, and
/// the caller clamps whisper's timestamps to the real audio. Handing it a
/// half-second of speech, on the other hand, is how a meeting ends up with
/// nothing written down at all (see [`MIN_DECODE_MS`]).
///
/// Borrows when the audio is already long enough, so the common case does not
/// copy a 28-second utterance.
fn padded_for_decode(samples: &[f32]) -> Cow<'_, [f32]> {
    if samples.len() >= MIN_DECODE_SAMPLES {
        return Cow::Borrowed(samples);
    }
    let mut padded = Vec::with_capacity(MIN_DECODE_SAMPLES);
    padded.extend_from_slice(samples);
    padded.resize(MIN_DECODE_SAMPLES, 0.0);
    Cow::Owned(padded)
}

/// A `.mlmodelc` is a directory bundle, so "the file exists" is the wrong check.
fn accelerator_is_usable(path: &Path) -> bool {
    path.is_dir() || path.is_file()
}

/// Drop whisper's bracketed stand-ins for silence, music and noise. They are
/// not words and a transcript full of "[BLANK_AUDIO]" is worse than a gap.
fn strip_noise_markers(text: &str) -> String {
    let mut out = text.to_string();
    for marker in NOISE_MARKERS {
        out = out.replace(marker, "");
    }
    out.trim().to_string()
}

// ---------------------------------------------------------------------------
// The worker
// ---------------------------------------------------------------------------

enum Msg {
    Load {
        reply: tokio::sync::oneshot::Sender<Result<BackendReport, AsrError>>,
    },
    Job {
        job: Box<TranscribeJob>,
        on_partial: Option<PartialFn>,
        reply: tokio::sync::oneshot::Sender<Result<Transcription, AsrError>>,
    },
    Release,
    Shutdown,
}

/// State the async side reads without touching the engine thread.
#[derive(Default)]
struct Shared {
    loaded: AtomicBool,
    queue_depth: AtomicU32,
    backend: Mutex<BackendReport>,
    /// What to load when something needs the engine. `None` = nothing chosen
    /// yet, which is what a fresh install looks like.
    config: Mutex<Option<EngineConfig>>,
    /// Meetings whose queued work should be thrown away.
    cancelled: Mutex<HashSet<String>>,
    /// Meeting whose job is decoding right now.
    running: Mutex<Option<String>>,
    /// Raised to stop the decode in flight.
    abort: Arc<AtomicBool>,
    /// The language each meeting settled on.
    languages: Mutex<HashMap<String, String>>,
}

/// Serialises jobs onto one loaded engine, loading on demand and unloading when
/// idle (mantra 1).
///
/// The rest of the app only ever talks to this.
pub struct EngineWorker {
    tx: Mutex<Option<SyncSender<Msg>>>,
    shared: Arc<Shared>,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl std::fmt::Debug for EngineWorker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineWorker")
            .field("loaded", &self.is_loaded())
            .field("queue_depth", &self.queue_depth())
            .finish()
    }
}

impl EngineWorker {
    /// Create the worker. Loads nothing yet.
    ///
    /// `idle_release_minutes` of 0 means "use the default"; there is no way to
    /// ask for weights that never go away, because that would break mantra 1.
    pub fn new(idle_release_minutes: u32) -> Self {
        let minutes = if idle_release_minutes == 0 {
            DEFAULT_IDLE_RELEASE_MINUTES
        } else {
            idle_release_minutes
        };
        let idle = Duration::from_secs(u64::from(minutes) * 60);
        let (tx, rx) = std::sync::mpsc::sync_channel(QUEUE_CAPACITY);
        let shared = Arc::new(Shared::default());
        let worker_shared = shared.clone();
        let thread = std::thread::Builder::new()
            .name("echo-speech".to_string())
            .spawn(move || run(rx, worker_shared, idle))
            .ok();

        Self {
            tx: Mutex::new(Some(tx)),
            shared,
            thread: Mutex::new(thread),
        }
    }

    /// Choose what to load. Changing the weights releases whatever is loaded, so
    /// a preset switch never leaves the old ones in memory.
    pub async fn configure(&self, config: EngineConfig) {
        let changed = {
            let mut slot = self.shared.config.lock().expect("engine config poisoned");
            let changed = slot.as_ref() != Some(&config);
            *slot = Some(config);
            changed
        };
        if changed && self.is_loaded() {
            self.release().await;
        }
    }

    /// Read the selected preset out of the database and configure from it.
    pub async fn configure_from_settings(&self, db: &Db) -> Result<(), AsrError> {
        self.configure(EngineConfig::from_settings(db).await?).await;
        Ok(())
    }

    /// What the engine is set up to load, if anything.
    pub fn configured(&self) -> Option<EngineConfig> {
        self.shared
            .config
            .lock()
            .expect("engine config poisoned")
            .clone()
    }

    /// Load now, so the first utterance of a meeting is not slow. Called at
    /// recording start and by the explicit pre-warm action.
    ///
    /// The decoding effort and the provenance come from whichever catalog entry
    /// these weights are, so switching preset behind our back cannot leave a
    /// segment claiming it was written by the previous one. An uncatalogued file
    /// (a developer pointing at their own weights) falls back to the default
    /// preset's numbers and names itself after the file.
    pub async fn prewarm(
        &self,
        model_path: &Path,
        accelerator_path: Option<&Path>,
    ) -> Result<(), AsrError> {
        let entry = model_path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|name| {
                catalog::CATALOG
                    .iter()
                    .find(|e| e.kind == AssetKind::Speech && e.file_name == name)
            });
        let preset = entry.and_then(|e| catalog::PRESETS.iter().find(|p| p.speech_id == e.id));

        let config = EngineConfig {
            model_path: model_path.to_path_buf(),
            accelerator_path: accelerator_path.map(Path::to_path_buf),
            decode: preset.map_or_else(|| catalog::default_preset().decode, |p| p.decode),
            model_name: entry.map_or_else(
                || {
                    model_path
                        .file_stem()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_default()
                },
                |e| e.name.to_string(),
            ),
            model_revision: entry.map_or_else(
                || catalog::CATALOG_REVISION.to_string(),
                |e| e.revision.to_string(),
            ),
        };
        self.configure(config).await;
        self.load_now().await.map(|_| ())
    }

    /// Load whatever is configured and report which backend came up.
    pub async fn load_now(&self) -> Result<BackendReport, AsrError> {
        let (reply, answer) = tokio::sync::oneshot::channel();
        self.send(Msg::Load { reply }, true).await?;
        answer.await.map_err(|_| AsrError::EngineGone)?
    }

    /// Queue a job. Returns [`AsrError::QueueFull`] when the bounded queue is
    /// full and this job was dropped in favour of keeping capture healthy; the
    /// audio is still on disk and the catch-up pass will get to it.
    ///
    /// Jobs marked non-droppable (catch-up work read back from disk) wait for
    /// room instead, because nothing else will pick them up.
    pub async fn submit(
        &self,
        job: TranscribeJob,
        on_partial: Option<PartialFn>,
    ) -> Result<Transcription, AsrError> {
        let droppable = job.droppable;
        let (reply, answer) = tokio::sync::oneshot::channel();
        self.send(
            Msg::Job {
                job: Box::new(job),
                on_partial,
                reply,
            },
            !droppable,
        )
        .await?;
        answer.await.map_err(|_| AsrError::EngineGone)?
    }

    /// Drop the weights now. Called on idle timeout and before quitting.
    pub async fn release(&self) {
        let _ = self.send(Msg::Release, true).await;
    }

    /// Are the weights in memory right now?
    pub fn is_loaded(&self) -> bool {
        self.shared.loaded.load(Ordering::SeqCst)
    }

    /// Utterances waiting. Surfaced to the person only as "catching up".
    pub fn queue_depth(&self) -> u32 {
        self.shared.queue_depth.load(Ordering::SeqCst)
    }

    /// Drop every queued job for one meeting, e.g. when it is deleted.
    ///
    /// Reaches the decode in flight too: waiting for a beam search over thirty
    /// seconds of audio to finish before honouring a delete is not acceptable.
    pub async fn cancel_meeting(&self, meeting_id: &str) {
        self.shared
            .cancelled
            .lock()
            .expect("cancelled set poisoned")
            .insert(meeting_id.to_string());
        let running = self
            .shared
            .running
            .lock()
            .expect("running meeting poisoned")
            .clone();
        if running.as_deref() == Some(meeting_id) {
            self.shared.abort.store(true, Ordering::SeqCst);
        }
    }

    /// Undo [`EngineWorker::cancel_meeting`], for a meeting that comes back
    /// (a retried catch-up job).
    pub fn allow_meeting(&self, meeting_id: &str) {
        self.shared
            .cancelled
            .lock()
            .expect("cancelled set poisoned")
            .remove(meeting_id);
    }

    pub fn is_cancelled(&self, meeting_id: &str) -> bool {
        self.shared
            .cancelled
            .lock()
            .expect("cancelled set poisoned")
            .contains(meeting_id)
    }

    pub fn backend(&self) -> BackendReport {
        self.shared
            .backend
            .lock()
            .expect("backend report poisoned")
            .clone()
    }

    /// The language this meeting settled on, once it has. Written to the meeting
    /// row by the session layer.
    pub fn meeting_language(&self, meeting_id: &str) -> Option<String> {
        self.shared
            .languages
            .lock()
            .expect("language map poisoned")
            .get(meeting_id)
            .cloned()
    }

    /// Forget a finished meeting's language and cancellation state.
    pub fn forget_meeting(&self, meeting_id: &str) {
        self.shared
            .languages
            .lock()
            .expect("language map poisoned")
            .remove(meeting_id);
        self.allow_meeting(meeting_id);
    }

    /// Stop the thread. Idempotent; called before quitting.
    pub fn shutdown(&self) {
        let sender = self.tx.lock().expect("engine sender poisoned").take();
        if let Some(tx) = sender {
            let _ = tx.send(Msg::Shutdown);
        }
        if let Some(handle) = self.thread.lock().expect("engine thread poisoned").take() {
            let _ = handle.join();
        }
    }

    /// Hand a message to the engine thread.
    ///
    /// `wait_for_room` decides what a full queue means: control messages and
    /// catch-up work wait, live utterances are dropped (mantra 3).
    async fn send(&self, msg: Msg, wait_for_room: bool) -> Result<(), AsrError> {
        let mut msg = msg;
        loop {
            let attempt = {
                let guard = self.tx.lock().expect("engine sender poisoned");
                match guard.as_ref() {
                    None => return Err(AsrError::EngineGone),
                    Some(tx) => tx.try_send(msg),
                }
            };
            match attempt {
                Ok(()) => {
                    self.shared.queue_depth.fetch_add(1, Ordering::SeqCst);
                    return Ok(());
                }
                Err(TrySendError::Full(returned)) => {
                    if !wait_for_room {
                        return Err(AsrError::QueueFull);
                    }
                    msg = returned;
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
                Err(TrySendError::Disconnected(_)) => return Err(AsrError::EngineGone),
            }
        }
    }
}

impl Drop for EngineWorker {
    fn drop(&mut self) {
        let sender = self.tx.lock().expect("engine sender poisoned").take();
        drop(sender);
        if let Some(handle) = self.thread.lock().expect("engine thread poisoned").take() {
            let _ = handle.join();
        }
    }
}

/// The engine thread. Owns the weights and nothing else owns them.
fn run(rx: Receiver<Msg>, shared: Arc<Shared>, idle: Duration) {
    let mut engine: Option<Engine> = None;
    let mut last_used = Instant::now();
    let mut policies: HashMap<String, LanguagePolicy> = HashMap::new();
    // Wake often enough to honour the idle period without polling hard.
    let tick = idle
        .min(Duration::from_secs(30))
        .max(Duration::from_secs(1));

    loop {
        match rx.recv_timeout(tick) {
            Ok(Msg::Shutdown) => break,
            Ok(Msg::Release) => {
                shared.queue_depth.fetch_sub(1, Ordering::SeqCst);
                if engine.take().is_some() {
                    shared.loaded.store(false, Ordering::SeqCst);
                    tracing::info!("speech engine released");
                }
            }
            Ok(Msg::Load { reply }) => {
                shared.queue_depth.fetch_sub(1, Ordering::SeqCst);
                let answer = ensure_loaded(&mut engine, &shared).map(|e| e.backend());
                last_used = Instant::now();
                let _ = reply.send(answer);
            }
            Ok(Msg::Job {
                job,
                on_partial,
                reply,
            }) => {
                shared.queue_depth.fetch_sub(1, Ordering::SeqCst);
                let answer = run_job(&mut engine, &shared, &mut policies, *job, on_partial);
                last_used = Instant::now();
                let _ = reply.send(answer);
            }
            Err(RecvTimeoutError::Timeout) => {
                if engine.is_some() && last_used.elapsed() >= idle {
                    engine = None;
                    shared.loaded.store(false, Ordering::SeqCst);
                    tracing::info!("speech engine released after being idle");
                }
            }
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
    drop(engine);
    shared.loaded.store(false, Ordering::SeqCst);
}

fn ensure_loaded<'a>(
    engine: &'a mut Option<Engine>,
    shared: &Shared,
) -> Result<&'a mut Engine, AsrError> {
    let wanted = shared
        .config
        .lock()
        .expect("engine config poisoned")
        .clone()
        .ok_or(AsrError::NotInstalled)?;

    // A preset switch while loaded means these weights are the wrong ones.
    if engine
        .as_ref()
        .is_some_and(|e| e.config().model_path != wanted.model_path)
    {
        *engine = None;
        shared.loaded.store(false, Ordering::SeqCst);
    }
    if engine.is_none() {
        let loaded = Engine::load(wanted, shared.abort.clone())?;
        *shared.backend.lock().expect("backend report poisoned") = loaded.backend();
        *engine = Some(loaded);
        shared.loaded.store(true, Ordering::SeqCst);
    }
    Ok(engine.as_mut().expect("just loaded"))
}

fn run_job(
    engine: &mut Option<Engine>,
    shared: &Shared,
    policies: &mut HashMap<String, LanguagePolicy>,
    mut job: TranscribeJob,
    on_partial: Option<PartialFn>,
) -> Result<Transcription, AsrError> {
    if shared
        .cancelled
        .lock()
        .expect("cancelled set poisoned")
        .contains(&job.meeting_id)
    {
        return Err(AsrError::Cancelled);
    }

    let engine = ensure_loaded(engine, shared)?;
    *shared.running.lock().expect("running meeting poisoned") = Some(job.meeting_id.clone());
    shared.abort.store(false, Ordering::SeqCst);

    // --- language: detect until settled, then pin (review finding 21) ------
    let policy = policies.entry(job.meeting_id.clone()).or_default();
    let speech_ms = job.duration_ms();
    let mut detected_confidence = None;
    if job.language_hint.is_none() {
        // Copy the hint out before doing anything else with the policy, so the
        // borrow ends here rather than covering the branch below.
        let pinned = policy.hint().map(str::to_string);
        if let Some(pinned) = pinned {
            job.language_hint = Some(pinned);
        } else if policy.wants_detection(speech_ms) {
            match engine.detect_language(&job.samples) {
                Ok((language, confidence)) => {
                    detected_confidence = Some(confidence);
                    if let Decision::Pinned(settled) =
                        policy.observe(&language, confidence, speech_ms)
                    {
                        shared
                            .languages
                            .lock()
                            .expect("language map poisoned")
                            .insert(job.meeting_id.clone(), settled);
                    }
                    // Use what we just found for this utterance even before the
                    // meeting as a whole has settled.
                    job.language_hint = Some(language);
                }
                Err(e) => {
                    // Detection is an optimisation. Losing it means whisper
                    // detects internally instead, which is fine.
                    tracing::debug!(%e, "could not work out the language; letting the engine decide");
                }
            }
        }
    }

    let mut result = engine.transcribe(&job, on_partial);
    *shared.running.lock().expect("running meeting poisoned") = None;

    if let Ok(transcription) = &mut result {
        transcription.language_confidence = detected_confidence;
    }
    result
}

// ---------------------------------------------------------------------------
// Catch-up
// ---------------------------------------------------------------------------

/// Transcribe a meeting's audio from disk, filling whatever the live pass
/// missed. Resumable: it starts from the last final segment and stops at the
/// last committed chunk.
///
/// This is the job that makes "audio on disk is the source of truth" real. The
/// implementation lives in [`crate::asr::catchup`].
pub async fn catch_up_from_disk(
    worker: &EngineWorker,
    db: &crate::db::Db,
    meeting_id: &str,
) -> Result<u32, AsrError> {
    catch_up_from_offset(worker, db, meeting_id, None).await
}

/// As [`catch_up_from_disk`], but starting at an explicit offset on the meeting
/// clock. `None` means "wherever the transcript ends", which is what a job
/// picked up after a restart wants.
pub async fn catch_up_from_offset(
    worker: &EngineWorker,
    db: &crate::db::Db,
    meeting_id: &str,
    from_ms: Option<i64>,
) -> Result<u32, AsrError> {
    let report = crate::asr::catchup::run(
        worker,
        db,
        meeting_id,
        crate::asr::catchup::CatchUpOptions {
            from_ms,
            ..Default::default()
        },
        &crate::asr::catchup::DiskAudio,
    )
    .await?;
    Ok(report.segments_written)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threads_leave_room_for_capture_and_never_run_away() {
        // A ten-core machine decodes on eight, which is the cap anyway.
        assert_eq!(clamp_threads(10), 8);
        assert_eq!(clamp_threads(12), 8, "the cap holds on bigger machines");
        // An eight-core machine leaves two alone.
        assert_eq!(clamp_threads(8), 6);
        // Small machines still get a floor, never zero or one.
        assert_eq!(clamp_threads(4), 2);
        assert_eq!(clamp_threads(2), 2);
        assert_eq!(clamp_threads(1), 2);
        assert_eq!(clamp_threads(0), 2);
        // And the real reading obeys the same bounds.
        let real = recommended_threads() as usize;
        assert!((MIN_DECODE_THREADS..=MAX_DECODE_THREADS).contains(&real));
    }

    #[test]
    fn the_quality_preset_decides_beam_search_or_greedy() {
        let everyday = catalog::preset("everyday").unwrap();
        match sampling_strategy(&everyday.decode) {
            SamplingStrategy::BeamSearch { beam_size, .. } => assert_eq!(beam_size, 5),
            other => panic!("the everyday preset should search beams, got {other:?}"),
        }
        let fastest = catalog::preset("fastest").unwrap();
        match sampling_strategy(&fastest.decode) {
            SamplingStrategy::Greedy { best_of } => assert!(best_of >= 1),
            other => panic!("the quickest preset should be greedy, got {other:?}"),
        }
    }

    #[test]
    fn the_compiled_backend_matches_what_this_platform_builds() {
        let compiled = compiled_backends();
        if cfg!(target_os = "macos") {
            assert_eq!(compiled, "metal+coreml");
            assert_eq!(gpu_backend_name(), Some("metal"));
        } else if cfg!(target_os = "linux") {
            assert_eq!(compiled, "vulkan");
            assert_eq!(gpu_backend_name(), Some("vulkan"));
        }
    }

    /// whisper.cpp answers `-6` rather than text when it cannot encode the
    /// window it was handed, and a window under a second is one it cannot
    /// encode. Nothing that reaches `whisper_full` may be shorter than
    /// [`MIN_DECODE_MS`], whichever path it came in on.
    #[test]
    fn nothing_reaches_the_decoder_shorter_than_it_can_read() {
        const RATE: usize = crate::audio::TARGET_SAMPLE_RATE as usize;
        const {
            assert!(
                MIN_DECODE_MS > 1_000,
                "the floor has to clear whisper.cpp's window, not sit on it"
            )
        };

        for ms in [1usize, 10, 80, 200, 608, 640, 999, 1_000, 1_099] {
            let short = vec![0.25f32; RATE * ms / 1_000];
            let padded = padded_for_decode(&short);
            assert!(
                padded.len() >= MIN_DECODE_SAMPLES,
                "{ms} ms went to the decoder as {} samples",
                padded.len()
            );
            // The audio itself is untouched; only the tail is added.
            assert_eq!(&padded[..short.len()], &short[..]);
            assert!(padded[short.len()..].iter().all(|s| *s == 0.0));
        }
    }

    #[test]
    fn audio_long_enough_to_read_is_handed_over_without_copying_it() {
        let long = vec![0.5f32; MIN_DECODE_SAMPLES + 1];
        let same = padded_for_decode(&long);
        assert!(
            matches!(same, Cow::Borrowed(_)),
            "a 28-second utterance must not be cloned to check its length"
        );
        assert_eq!(same.len(), long.len());

        // Exactly at the floor is already readable.
        let exact = vec![0.5f32; MIN_DECODE_SAMPLES];
        assert!(matches!(padded_for_decode(&exact), Cow::Borrowed(_)));
    }

    /// The catch-up pass reads whole holes off disk, and the smallest hole it
    /// bothers with is [`crate::asr::catchup::MIN_GAP_MS`]. That is under a
    /// second, so the live path is not the only one that needs the padding.
    #[test]
    fn the_smallest_catch_up_window_is_padded_too() {
        const RATE: usize = crate::audio::TARGET_SAMPLE_RATE as usize;
        let smallest_hole = crate::asr::catchup::MIN_GAP_MS as usize;
        assert!(
            smallest_hole < 1_000,
            "this test exists because catch-up can hand over a {smallest_hole} ms window"
        );
        let window = vec![0.1f32; RATE * smallest_hole / 1_000];
        assert!(padded_for_decode(&window).len() >= MIN_DECODE_SAMPLES);
    }

    #[test]
    fn a_click_or_a_keystroke_is_nothing_said_rather_than_a_failure() {
        // Below the speech floor the engine answers "nothing was said" without
        // troubling the decoder at all.
        const { assert!(MIN_JOB_MS <= 100, "a short answer is still an answer") };
        let job = TranscribeJob {
            samples: vec![0.0; 16 * 40], // 40 ms
            ..Default::default()
        };
        assert!(job.duration_ms() < MIN_JOB_MS);
    }

    #[test]
    fn whisper_noise_markers_never_reach_a_transcript() {
        assert_eq!(strip_noise_markers("[BLANK_AUDIO]"), "");
        assert_eq!(strip_noise_markers(" [MUSIC] hello "), "hello");
        assert_eq!(
            strip_noise_markers("so [INAUDIBLE] then"),
            "so  then".trim()
        );
        assert_eq!(strip_noise_markers("hello there"), "hello there");
    }

    #[test]
    fn partials_carry_the_whole_utterance_so_far() {
        let seen: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = PartialSink::default();
        let collected = seen.clone();
        sink.begin(Some(Arc::new(move |text: &str| {
            collected.lock().unwrap().push(text.to_string())
        })));
        sink.push(" so ");
        sink.push("we agreed");
        assert_eq!(
            *seen.lock().unwrap(),
            vec!["so".to_string(), "so we agreed".to_string()]
        );

        // Once the job ends nothing more is emitted, and the next job starts
        // from an empty slate.
        sink.end();
        sink.push("leftover");
        assert_eq!(seen.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn a_fresh_worker_holds_nothing_and_says_so() {
        let worker = EngineWorker::new(10);
        assert!(!worker.is_loaded(), "mantra 1: nothing loads on its own");
        assert_eq!(worker.queue_depth(), 0);
        assert!(worker.configured().is_none());
        assert_eq!(worker.backend().active, "");
        // Releasing something that was never loaded is not an error.
        worker.release().await;
        assert!(!worker.is_loaded());
        worker.shutdown();
    }

    #[tokio::test]
    async fn without_a_download_there_is_nothing_to_load() {
        let worker = EngineWorker::new(10);
        let err = worker.load_now().await.unwrap_err();
        assert!(matches!(err, AsrError::NotInstalled), "{err:?}");

        let err = worker
            .submit(
                TranscribeJob {
                    meeting_id: "m1".into(),
                    samples: vec![0.0; 16_000],
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap_err();
        assert!(matches!(err, AsrError::NotInstalled), "{err:?}");
        assert!(!worker.is_loaded());
        worker.shutdown();
    }

    #[tokio::test]
    async fn work_for_a_cancelled_meeting_is_thrown_away_without_loading_anything() {
        let worker = EngineWorker::new(10);
        // Point it at weights that do not exist: if the cancellation check did
        // not come first, this would fail with NotInstalled instead.
        worker
            .configure(EngineConfig {
                model_path: PathBuf::from("/nowhere/ggml-tiny.bin"),
                accelerator_path: None,
                decode: catalog::default_preset().decode,
                model_name: "test".into(),
                model_revision: "test".into(),
            })
            .await;
        worker.cancel_meeting("gone").await;
        assert!(worker.is_cancelled("gone"));

        let err = worker
            .submit(
                TranscribeJob {
                    meeting_id: "gone".into(),
                    samples: vec![0.0; 16_000],
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap_err();
        assert!(matches!(err, AsrError::Cancelled), "{err:?}");
        assert!(!worker.is_loaded());

        // A meeting that comes back is allowed again.
        worker.allow_meeting("gone");
        assert!(!worker.is_cancelled("gone"));
        let err = worker
            .submit(
                TranscribeJob {
                    meeting_id: "gone".into(),
                    samples: vec![0.0; 16_000],
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap_err();
        assert!(
            matches!(err, AsrError::NotInstalled | AsrError::Load(_)),
            "{err:?}"
        );
        worker.shutdown();
    }

    #[tokio::test]
    async fn prewarming_takes_the_provenance_from_whichever_weights_they_are() {
        let worker = EngineWorker::new(10);
        let tiny = catalog::entry(catalog::ids::SPEECH_FASTEST).unwrap();
        let path = PathBuf::from("/nowhere").join(tiny.file_name);

        // Loading fails (the file is not there), but the configuration it chose
        // is what we are checking.
        let _ = worker.prewarm(&path, None).await;
        let cfg = worker
            .configured()
            .expect("prewarm configures before loading");
        assert_eq!(cfg.model_name, tiny.name);
        assert_eq!(cfg.model_revision, tiny.revision);
        assert_eq!(
            cfg.decode,
            catalog::preset("fastest").unwrap().decode,
            "the quickest weights decode with the quickest preset's numbers"
        );

        // Weights nobody catalogued still get a usable name rather than nothing.
        let _ = worker
            .prewarm(Path::new("/tmp/my-own-weights.bin"), None)
            .await;
        let cfg = worker.configured().unwrap();
        assert_eq!(cfg.model_name, "my-own-weights");
        assert_eq!(cfg.decode, catalog::default_preset().decode);
        worker.shutdown();
    }

    #[tokio::test]
    async fn choosing_a_different_preset_replaces_what_would_be_loaded() {
        let worker = EngineWorker::new(10);
        let first = EngineConfig {
            model_path: PathBuf::from("/nowhere/ggml-tiny.bin"),
            accelerator_path: None,
            decode: catalog::preset("fastest").unwrap().decode,
            model_name: "tiny".into(),
            model_revision: "r".into(),
        };
        worker.configure(first.clone()).await;
        assert_eq!(worker.configured().as_ref(), Some(&first));

        let second = EngineConfig {
            model_path: PathBuf::from("/nowhere/ggml-small.bin"),
            ..first.clone()
        };
        worker.configure(second.clone()).await;
        assert_eq!(worker.configured(), Some(second));
        worker.shutdown();
    }

    #[tokio::test]
    async fn a_dead_worker_reports_itself_rather_than_hanging() {
        let worker = EngineWorker::new(10);
        worker.shutdown();
        let err = worker.load_now().await.unwrap_err();
        assert!(matches!(err, AsrError::EngineGone), "{err:?}");
        // Shutting down twice is safe.
        worker.shutdown();
    }

    #[tokio::test]
    async fn the_engine_settings_read_the_selected_preset_and_refuse_to_guess() {
        let db = crate::db::connect_in_memory().await.unwrap();
        models::sync_catalog(&db).await.unwrap();
        let err = EngineConfig::from_settings(&db).await.unwrap_err();
        assert!(
            matches!(err, AsrError::NotInstalled),
            "no download means no engine, not a fallback: {err:?}"
        );

        repo::set_model_installed(
            &db,
            catalog::ids::SPEECH_EVERYDAY,
            true,
            Some("/speech/everyday.bin"),
        )
        .await
        .unwrap();
        let cfg = EngineConfig::from_settings(&db).await.unwrap();
        assert_eq!(cfg.model_path, PathBuf::from("/speech/everyday.bin"));
        assert_eq!(cfg.decode, catalog::preset("everyday").unwrap().decode);
        assert_eq!(
            cfg.model_name,
            catalog::entry(catalog::ids::SPEECH_EVERYDAY).unwrap().name
        );
        assert_eq!(
            cfg.model_revision,
            catalog::entry(catalog::ids::SPEECH_EVERYDAY)
                .unwrap()
                .revision
        );
        assert!(
            cfg.accelerator_path.is_none(),
            "the companion is only used once it is installed"
        );
    }

    #[test]
    fn the_default_idle_period_gives_the_memory_back() {
        assert_eq!(DEFAULT_IDLE_RELEASE_MINUTES, 10);
        // A caller asking for nothing still gets a release, not "never".
        let worker = EngineWorker::new(0);
        assert!(!worker.is_loaded());
        worker.shutdown();
    }
}
