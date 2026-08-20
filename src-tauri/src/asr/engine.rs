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
//! pre-warm.
//!
//! Resident while listening (mantra 1's amendment of 2026-08-20): they are
//! *kept* for as long as the session layer says a meeting is being listened to
//! or has work outstanding — [`EngineWorker::set_resident`]. Only once that is
//! false does the grace period start, and only after [`IDLE_GRACE`] of it with
//! nothing new are the weights handed back. Nothing here decides when a meeting
//! is over; it is told.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
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
use crate::types::{AssetKind, Channel};

/// Called with partial text while a job decodes. Cheap and non-blocking; the
/// caller rate-caps before it reaches the UI.
pub type PartialFn = Box<dyn Fn(&str) + Send + Sync>;

/// The same thing once it is inside the engine. whisper.cpp keeps the callback
/// we install for longer than the call frame, so it has to be shared rather than
/// owned.
type SharedPartial = Arc<dyn Fn(&str) + Send + Sync>;

/// How many jobs may wait for the engine. Small on purpose: a deep queue means
/// live captions minutes behind the meeting, which is worse than no captions.
/// Overflow drops *live* work only, oldest guess first; the audio is on disk
/// (mantra 3).
pub const QUEUE_CAPACITY: usize = 8;

/// How far behind a decode in flight has to be before a newer snapshot of the
/// same speech cancels it outright.
///
/// One cadence plus a margin: a hypothesis that is only three seconds stale is
/// nearly done and worth finishing, one that is four seconds stale is being
/// replaced the moment it lands, so the encoder time is better spent on the
/// newer window (codex: "an in-flight decode should be cancellable if a much
/// newer snapshot exists").
const STALE_SNAPSHOT_MS: i64 = 4_000;

/// How long the weights stay in memory once nothing needs them any more.
///
/// Not a setting, deliberately (product decision of 2026-08-20): a person
/// cannot be asked how many minutes of memory a speech engine should hold, and
/// the honest answer does not depend on them. The engine is resident while a
/// meeting is being listened to or its jobs are outstanding, and this is the
/// grace after that — long enough for the pause between two back-to-back
/// meetings, short enough that an idle Echo is empty.
pub const IDLE_GRACE: Duration = Duration::from_secs(5 * 60);

/// How often the worker wakes to check the grace period.
const IDLE_TICK: Duration = Duration::from_secs(30);

/// Finished utterances queued before new finals narrow their beam.
///
/// The valve behind [`catalog::DecodeParams::live_final_degraded`]. Two channels
/// decoding at the preset's full width is the right trade almost always; when it
/// is not, the symptom is a queue of finished utterances that keeps growing, and
/// the cure is to spend less per utterance until it is empty again. Three is one
/// utterance per channel plus one: below that, the queue is just two people
/// talking at once.
///
/// Counted **at the moment the decision is made**, with the final about to be
/// decoded included — see [`falling_behind`]. The valve used to be read after
/// that final had been popped and against a strict `>`, so it took five queued
/// utterances to open a valve documented to open at three
/// (review of 2026-08-20, finding 6).
pub const FINAL_BACKLOG_DEGRADE_AT: usize = 3;

/// Consecutive failed decodes that mean the engine itself is the problem.
///
/// A single stretch of audio can defeat the decoder; three in a row cannot be
/// about the audio. When the field regression of 2026-08-20 happened there was
/// no rung above [`Engine::reset_state`], so 279 identical failures went by
/// without anything trying anything different. Three is the smallest number that
/// is unmistakably a pattern rather than a bad minute of a meeting.
const RELOAD_AFTER_FAILURES: u32 = 3;

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

/// Gap between the audio handed over and the audio whisper actually read that is
/// worth a log line. Below this it is the trailing pad and the closing silence;
/// above it, words went missing and catch-up will be redoing that stretch.
const COVERAGE_SHORTFALL_MS: i64 = 1_500;

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
// What kind of work this is
// ---------------------------------------------------------------------------

/// What a job is *for*. It decides two things: how hard the engine tries, and
/// who goes first on the one worker.
///
/// This is the whole of the live/catch-up decode split (review of 2026-08-20,
/// §3). One preset, three ways of spending it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum JobKind {
    /// One look at an utterance that is **still open**, decoded so a caption can
    /// appear while somebody is still talking. Replaced wholesale by the next
    /// look and then by the final, so it gets the cheapest decode there is and
    /// the last place in the queue.
    Speculative,
    /// A finished live utterance: the text a person keeps. Greedy or a narrow
    /// beam, one fallback at most, natural segments and timestamps.
    #[default]
    Final,
    /// Read back from disk after the meeting, where latency is invisible: the
    /// preset's full beam and its whole fallback ladder.
    CatchUp,
}

impl JobKind {
    /// Live work is what gives way when the queue is full (mantra 3). Catch-up
    /// work waits instead, because nothing else will pick it up.
    pub fn is_live(self) -> bool {
        matches!(self, JobKind::Speculative | JobKind::Final)
    }

    /// A caption that is about to be replaced anyway.
    pub fn is_speculative(self) -> bool {
        matches!(self, JobKind::Speculative)
    }
}

/// Everything about how to decode one job that does not come from the preset.
///
/// Deliberately small: a kind and, when the previous utterance was cut
/// mid-sentence, the words to carry across the join.
#[derive(Debug, Clone, Default)]
pub struct DecodePlan {
    pub kind: JobKind,
    /// Explicit left context — the tail of the previous final on this channel,
    /// when this audio is the continuation of speech that was force-cut.
    ///
    /// Explicit, never automatic: whisper.cpp's own carry-over
    /// (`no_context = false`) is what locks it into repeating itself, so Echo
    /// keeps that off and hands over the words it chose (codex §3 "Context and
    /// prompts").
    pub prompt: Option<String>,
    /// Set by the worker as the job leaves the queue, never by the caller:
    /// finished utterances are piling up, so this one narrows its beam
    /// ([`FINAL_BACKLOG_DEGRADE_AT`]). Only ever true for [`JobKind::Final`].
    pub(crate) behind: bool,
}

impl DecodePlan {
    /// A finished live utterance.
    pub fn final_utterance() -> Self {
        Self {
            kind: JobKind::Final,
            ..Default::default()
        }
    }

    /// One look at an utterance that is still open.
    pub fn speculative() -> Self {
        Self {
            kind: JobKind::Speculative,
            ..Default::default()
        }
    }

    /// A window read back from disk.
    pub fn catch_up() -> Self {
        Self {
            kind: JobKind::CatchUp,
            ..Default::default()
        }
    }

    /// Carry the tail of the previous final across a forced cut. Blank prompts
    /// are dropped: an empty one still costs tokens.
    pub fn with_prompt(mut self, prompt: Option<String>) -> Self {
        self.prompt = prompt.filter(|p| !p.trim().is_empty());
        self
    }
}

/// The decoding numbers and flags one job runs with.
struct Decoding {
    params: DecodeParams,
    /// One replaceable hypothesis rather than natural segments.
    single_segment: bool,
    /// Off for speculative work: nothing reads the timestamps of a caption that
    /// is replaced three seconds later, and asking for them costs.
    timestamps: bool,
}

impl Decoding {
    /// `behind` is the safety valve: a final decoded while finished utterances
    /// are piling up narrows its beam (see [`FINAL_BACKLOG_DEGRADE_AT`]).
    /// Captions never change — they are already as cheap as a decode gets — and
    /// the disk pass never does either, because nothing is waiting for it.
    fn for_job(kind: JobKind, preset: &DecodeParams, behind: bool) -> Self {
        match kind {
            JobKind::Speculative => Self {
                params: preset.speculative(),
                single_segment: true,
                timestamps: false,
            },
            JobKind::Final => Self {
                params: if behind {
                    preset.live_final_degraded()
                } else {
                    preset.live_final()
                },
                single_segment: false,
                timestamps: true,
            },
            JobKind::CatchUp => Self {
                params: *preset,
                single_segment: false,
                timestamps: true,
            },
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
    /// Read the level and the installed files out of the database.
    ///
    /// Loads whatever is **actually installed**, and names the segments after
    /// it. There is one model in the catalog's level, but there may be an older
    /// one still on disk while the new one downloads, and in that window this
    /// resolves to the old one — that is how a meeting started mid-upgrade gets
    /// transcribed at all (see [`crate::asr::reconcile`]). What it never does is
    /// claim the new model wrote text the old one wrote: `model_name` and
    /// `model_revision` come from the row that was loaded, so the provenance on
    /// every segment stays true through the switch.
    ///
    /// Fails with [`AsrError::NotInstalled`] only when there is nothing at all.
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

        // Which catalog entry that path *is*, rather than which one we hoped
        // for. Decoding numbers come from the level either way: they are about
        // how hard to try, not about which weights, and the level's numbers are
        // what shipped with the weights that are serving.
        let loaded = model_path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|name| {
                catalog::CATALOG
                    .iter()
                    .find(|e| e.kind == AssetKind::Speech && e.file_name == name)
            });
        let asset_id = loaded.map_or(preset.speech_id, |e| e.id);
        let row = repo::get_model(db, asset_id).await?;
        Ok(Self {
            model_path,
            accelerator_path,
            decode: preset.decode,
            model_name: row
                .as_ref()
                .map(|r| r.name.clone())
                .unwrap_or_else(|| asset_id.to_string()),
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
    ///
    /// `plan` decides how much this is allowed to cost and whether any words are
    /// carried in from the utterance before it; see [`JobKind`].
    pub fn transcribe(
        &mut self,
        job: &TranscribeJob,
        plan: &DecodePlan,
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

        let decoding = Decoding::for_job(plan.kind, &self.config.decode, plan.behind);
        let mut params = FullParams::new(sampling_strategy(&decoding.params));
        params.set_n_threads(self.backend.threads as i32);
        // Transcription, never translation.
        params.set_translate(false);
        // Each utterance stands alone. Carrying decoder state across a silence is
        // how whisper starts repeating itself for ever; the only context that
        // travels is the prompt below, which we choose.
        params.set_no_context(true);
        if let Some(prompt) = plan.prompt.as_deref() {
            // whisper.cpp clears its own history for `no_context` first and then
            // pushes this in, so the utterance gets exactly these words and
            // nothing else.
            params.set_initial_prompt(prompt);
        }
        params.set_single_segment(decoding.single_segment);
        params.set_no_timestamps(!decoding.timestamps);
        // Experimental in whisper.cpp, and nothing here reads per-token times.
        params.set_token_timestamps(false);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);
        params.set_temperature(decoding.params.temperature);
        params.set_temperature_inc(decoding.params.temperature_inc);
        params.set_entropy_thold(decoding.params.entropy_thold);
        params.set_logprob_thold(decoding.params.logprob_thold);
        params.set_no_speech_thold(decoding.params.no_speech_thold);

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
        install_abort_callback(&mut params, &self.abort);

        // Never the raw utterance: whisper.cpp cannot encode a sub-second window
        // and answers -6 instead of text.
        let audio = padded_for_decode(&job.samples);
        let outcome = self.state.full(params, audio.as_ref());
        self.sink.end();
        if self.abort.load(Ordering::SeqCst) {
            // An abort leaves whisper.cpp's graph half-built exactly like a
            // failure does, and superseding a stale snapshot is now a routine
            // event rather than a delete. Rebuild rather than let the next
            // utterance inherit it.
            if outcome.is_err() {
                self.reset_state();
            }
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
        // Coverage, honestly (review finding 1). `t_end_ms` is *what whisper
        // read*, not how much audio it was handed: when it stops halfway through
        // a 28-second window, the rest of that window has no text against it and
        // the catch-up pass has to be able to see that. Never past the audio we
        // gave it, though — the padded tail is not speech.
        if let Some(end) = last_end_ms {
            let clamped = end.clamp(0, duration_ms);
            if clamped > 0 {
                result.t_end_ms = job.t_start_ms + clamped;
                let shortfall = duration_ms - clamped;
                if shortfall > COVERAGE_SHORTFALL_MS && !plan.kind.is_speculative() {
                    tracing::debug!(
                        channel = ?job.channel,
                        shortfall_ms = shortfall,
                        window_ms = duration_ms,
                        "the engine stopped before the end of this stretch; the rest is left for the catch-up pass"
                    );
                }
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

/// What whisper.cpp asks at the end of every encode: "should I stop?"
///
/// `whisper_encode_internal` ends with `return !(abort_callback &&
/// abort_callback(data))`, and `whisper_full_with_state` turns a false there
/// into `return -6`. So this one bool *is* the difference between a transcript
/// and `Generic whisper error … Error code: -6`, and it has to answer for the
/// flag we own and nothing else.
///
/// # Safety
/// `user_data` must point at a live [`AtomicBool`]. [`install_abort_callback`]
/// is the only caller-facing way to set it up, and it passes the one inside an
/// `Arc` the engine holds for the whole decode.
unsafe extern "C" fn abort_when_asked(user_data: *mut std::ffi::c_void) -> bool {
    // SAFETY: the contract above. Relaxed is enough: whoever raises the flag
    // only needs this decode to notice soon, not at a particular instruction.
    unsafe { &*(user_data.cast::<AtomicBool>()) }.load(Ordering::Relaxed)
}

/// Point whisper.cpp's abort callback at `flag`, and hand back exactly the pair
/// it will call so the wiring can be tested without loading any weights.
///
/// Not `FullParams::set_abort_callback_safe`, deliberately (root cause of the
/// 2026-08-20 field regression). That helper boxes the closure twice and stores
/// a `*mut Box<dyn FnMut() -> bool>`, but instantiates its trampoline for the
/// *closure's own* type and casts the pointer to it — so the callback reads a
/// bool out of whatever heap byte follows the inner box instead of out of our
/// `AtomicBool`. Nearly every read came back non-zero, whisper.cpp aborted the
/// encode, and a 21-minute meeting produced 279 `-6`s and eight seconds of text.
/// Padding cannot cure that and rebuilding the state cannot recover from it,
/// because nothing was ever wrong with the audio or the state.
///
/// A plain `extern "C"` function over a pointer to the flag has no closure to
/// mistype, allocates nothing, and leaks nothing — the old path leaked its two
/// boxes on every single utterance.
fn install_abort_callback(
    params: &mut FullParams,
    flag: &Arc<AtomicBool>,
) -> (whisper_rs::WhisperAbortCallback, *mut std::ffi::c_void) {
    let user_data = Arc::as_ptr(flag).cast_mut().cast::<std::ffi::c_void>();
    let callback: whisper_rs::WhisperAbortCallback = Some(abort_when_asked);
    // SAFETY: `flag` is an `Arc` field of the engine, so the `AtomicBool` it
    // points at outlives every `whisper_full` call these params are used for,
    // and `abort_when_asked` reads nothing else.
    unsafe {
        params.set_abort_callback(callback);
        params.set_abort_callback_user_data(user_data);
    }
    (callback, user_data)
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
// The queue: one worker, three lanes
// ---------------------------------------------------------------------------
//
// whisper.cpp's context is not safe for two `whisper_full` calls at once, so
// there is exactly one worker and the interesting question is what it does next.
// Three lanes, in this order (codex: "with the current memory footprint, a
// priority scheduler is preferable"):
//
//   1. control   — loading, releasing, shutting down. Never waits behind audio.
//   2. finals    — the text a person keeps, fairly between the two channels.
//   3. catch-up  — read back from disk; nobody is watching it arrive.
//   4. snapshots — captions of speech that is still going, latest only.
//
// Within a lane the two channels take turns: one continuously open microphone
// must not starve the system channel.

/// Control work. Separate from decoding so a release or a shutdown does not wait
/// for a beam search to finish.
enum Control {
    Load {
        reply: tokio::sync::oneshot::Sender<Result<BackendReport, AsrError>>,
    },
    Release,
    Shutdown,
}

/// One queued decode, with the answer's return address.
struct Pending {
    job: Box<TranscribeJob>,
    plan: DecodePlan,
    on_partial: Option<PartialFn>,
    reply: tokio::sync::oneshot::Sender<Result<Transcription, AsrError>>,
    /// Arrival order, so "newer" is a fact rather than a guess.
    seq: u64,
}

impl Pending {
    /// Which stretch of which channel of which meeting this is a look at. Two
    /// jobs with the same key are two looks at the same speech.
    fn key(&self) -> (&str, Channel) {
        (self.job.meeting_id.as_str(), self.job.channel)
    }

    /// End of the audio this job holds, on the meeting clock.
    fn covers_to_ms(&self) -> i64 {
        self.job.t_end_ms()
    }

    /// Tell the caller this hypothesis was thrown away. Not a failure: either a
    /// newer look at the same speech is already queued, or the meeting is over
    /// and the disk pass is authoritative from here.
    fn abandon(self) {
        let _ = self.reply.send(Err(AsrError::Cancelled));
    }
}

/// What the worker should pick up next.
enum Task {
    Control(Control),
    Job(Pending),
}

/// The decode in flight, so a newer snapshot can overtake it and a deleted
/// meeting can stop it.
#[derive(Debug, Clone)]
struct Running {
    meeting_id: String,
    channel: Channel,
    kind: JobKind,
    covers_to_ms: i64,
}

/// How much of a meeting's live work to throw away.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Abandon {
    /// Only the captions of speech that is still going.
    Speculative,
    /// Every live job: captions and queued finals both. Used at the stop
    /// handoff, where the disk pass becomes authoritative (codex finding 7).
    Live,
    /// Everything, including work read back from disk. Used when a meeting is
    /// deleted.
    Everything,
}

impl Abandon {
    fn takes(self, kind: JobKind) -> bool {
        match self {
            Abandon::Speculative => kind.is_speculative(),
            Abandon::Live => kind.is_live(),
            Abandon::Everything => true,
        }
    }
}

/// Why a job could not be queued.
enum PushError {
    /// The engine thread is gone.
    Gone,
    /// No room. The job comes back so the caller can decide: live work is
    /// dropped, catch-up work waits.
    Full(Pending),
}

impl std::fmt::Debug for PushError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PushError::Gone => f.write_str("Gone"),
            PushError::Full(pending) => f
                .debug_struct("Full")
                .field("kind", &pending.plan.kind)
                .field("channel", &pending.job.channel)
                .finish(),
        }
    }
}

/// The scheduling state. Pure: every decision here is testable without a thread,
/// a runtime or 1.6 GB of weights.
#[derive(Default)]
struct QueueState {
    control: VecDeque<Control>,
    finals: VecDeque<Pending>,
    catchup: VecDeque<Pending>,
    /// At most one per (meeting, channel): a newer snapshot **replaces** the
    /// queued older one rather than joining it behind it (review finding 5).
    speculative: Vec<Pending>,
    /// Which channel each lane served last, so the turns alternate.
    last_final: Option<Channel>,
    last_speculative: Option<Channel>,
    running: Option<Running>,
    next_seq: u64,
    closed: bool,
}

impl QueueState {
    fn waiting(&self) -> usize {
        self.finals.len() + self.catchup.len() + self.speculative.len()
    }

    /// Finished utterances still waiting for text. This, and not the total, is
    /// what "falling behind" means: captions are replaceable and the disk pass
    /// has nobody waiting on it.
    fn finals_waiting(&self) -> usize {
        self.finals.len()
    }

    /// Queue one job, or hand it back when there is no room for it.
    ///
    /// Returns whether the decode in flight has been overtaken and should be
    /// cancelled.
    fn push(&mut self, mut pending: Pending) -> Result<bool, PushError> {
        if self.closed {
            return Err(PushError::Gone);
        }
        pending.seq = self.next_seq;
        self.next_seq += 1;

        match pending.plan.kind {
            JobKind::Speculative => {
                // A newer look at the same speech supersedes the queued one:
                // decoding both would spend the encoder twice to show the older
                // answer for a moment and then throw it away.
                if let Some(i) = self
                    .speculative
                    .iter()
                    .position(|queued| queued.key() == pending.key())
                {
                    self.speculative.remove(i).abandon();
                }
                // A snapshot never displaces real speech. When the queue is
                // full of utterances, the caption is the cheapest thing in the
                // room, so it is the thing that goes.
                if self.waiting() >= QUEUE_CAPACITY {
                    pending.abandon();
                    return Ok(false);
                }
                let overtakes = self.running_is_stale_snapshot(&pending);
                self.speculative.push(pending);
                Ok(overtakes)
            }
            JobKind::Final => {
                if self.waiting() >= QUEUE_CAPACITY {
                    // Prefer losing an old guess to losing new speech
                    // (review finding 5): drop the oldest snapshot to make room.
                    if let Some(oldest) = self.oldest_speculative() {
                        self.speculative.remove(oldest).abandon();
                    }
                }
                if self.waiting() >= QUEUE_CAPACITY {
                    return Err(PushError::Full(pending));
                }
                self.finals.push_back(pending);
                Ok(false)
            }
            JobKind::CatchUp => {
                if self.waiting() >= QUEUE_CAPACITY {
                    return Err(PushError::Full(pending));
                }
                self.catchup.push_back(pending);
                Ok(false)
            }
        }
    }

    /// Is the decode in flight a caption this snapshot has left well behind?
    fn running_is_stale_snapshot(&self, pending: &Pending) -> bool {
        self.running.as_ref().is_some_and(|running| {
            running.kind.is_speculative()
                && running.meeting_id == pending.job.meeting_id
                && running.channel == pending.job.channel
                && pending.covers_to_ms() - running.covers_to_ms >= STALE_SNAPSHOT_MS
        })
    }

    fn oldest_speculative(&self) -> Option<usize> {
        self.speculative
            .iter()
            .enumerate()
            .min_by_key(|(_, p)| p.seq)
            .map(|(i, _)| i)
    }

    /// The next thing to do, in lane order.
    fn next(&mut self) -> Option<Task> {
        if let Some(control) = self.control.pop_front() {
            return Some(Task::Control(control));
        }
        if let Some(job) = pop_fair(&mut self.finals, &mut self.last_final) {
            return Some(Task::Job(job));
        }
        if let Some(job) = self.catchup.pop_front() {
            return Some(Task::Job(job));
        }
        if let Some(job) = self.pop_snapshot() {
            return Some(Task::Job(job));
        }
        None
    }

    /// The newest snapshot, taking turns between channels.
    fn pop_snapshot(&mut self) -> Option<Pending> {
        if self.speculative.is_empty() {
            return None;
        }
        let other = self
            .last_speculative
            .and_then(|last| {
                self.speculative
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| p.job.channel != last)
                    .min_by_key(|(_, p)| p.seq)
                    .map(|(i, _)| i)
            })
            .or_else(|| self.oldest_speculative())?;
        let taken = self.speculative.remove(other);
        self.last_speculative = Some(taken.job.channel);
        Some(taken)
    }

    /// Throw away one meeting's queued work. Returns whether the decode in
    /// flight belongs to it too and should be cancelled.
    fn abandon(&mut self, meeting_id: &str, what: Abandon) -> bool {
        let mut taken = Vec::new();
        let mut keep = VecDeque::with_capacity(self.finals.len());
        for pending in self.finals.drain(..) {
            if pending.job.meeting_id == meeting_id && what.takes(pending.plan.kind) {
                taken.push(pending);
            } else {
                keep.push_back(pending);
            }
        }
        self.finals = keep;

        let mut keep = VecDeque::with_capacity(self.catchup.len());
        for pending in self.catchup.drain(..) {
            if pending.job.meeting_id == meeting_id && what.takes(pending.plan.kind) {
                taken.push(pending);
            } else {
                keep.push_back(pending);
            }
        }
        self.catchup = keep;

        let (mine, theirs): (Vec<Pending>, Vec<Pending>) = self
            .speculative
            .drain(..)
            .partition(|p| p.job.meeting_id == meeting_id && what.takes(p.plan.kind));
        self.speculative = theirs;
        taken.extend(mine);

        for pending in taken {
            pending.abandon();
        }

        self.running
            .as_ref()
            .is_some_and(|running| running.meeting_id == meeting_id && what.takes(running.kind))
    }
}

/// Take the front of a lane, unless the other channel has been waiting: one long
/// monologue on the microphone must not push the system channel out of the way.
fn pop_fair(lane: &mut VecDeque<Pending>, last: &mut Option<Channel>) -> Option<Pending> {
    let front = lane.front()?.job.channel;
    let index = if *last == Some(front) {
        lane.iter()
            .position(|p| p.job.channel != front)
            .unwrap_or(0)
    } else {
        0
    };
    let taken = lane.remove(index)?;
    *last = Some(taken.job.channel);
    Some(taken)
}

/// The queue the async side pushes to and the worker thread pulls from.
#[derive(Default)]
struct JobQueue {
    state: Mutex<QueueState>,
    ready: std::sync::Condvar,
}

impl JobQueue {
    fn lock(&self) -> std::sync::MutexGuard<'_, QueueState> {
        self.state.lock().expect("engine queue poisoned")
    }

    fn push_control(&self, control: Control) -> Result<(), AsrError> {
        let mut state = self.lock();
        if state.closed {
            return Err(AsrError::EngineGone);
        }
        state.control.push_back(control);
        drop(state);
        self.ready.notify_one();
        Ok(())
    }

    /// Queue a job. `Ok(true)` means the decode in flight has been overtaken.
    fn push_job(&self, pending: Pending) -> Result<bool, PushError> {
        let mut state = self.lock();
        let overtakes = state.push(pending)?;
        drop(state);
        self.ready.notify_one();
        Ok(overtakes)
    }

    /// Wait for something to do. `None` on timeout, or once the queue is closed
    /// and empty.
    fn take(&self, timeout: Duration) -> Option<Task> {
        let mut state = self.lock();
        if let Some(task) = state.next() {
            return Some(task);
        }
        if state.closed {
            return None;
        }
        let (mut state, _) = self
            .ready
            .wait_timeout(state, timeout)
            .expect("engine queue poisoned");
        state.next()
    }

    /// Mark what is decoding, so a snapshot can overtake it.
    fn begin(&self, running: Running) {
        self.lock().running = Some(running);
    }

    fn finish(&self) {
        self.lock().running = None;
    }

    fn running_meeting(&self) -> Option<String> {
        self.lock().running.as_ref().map(|r| r.meeting_id.clone())
    }

    fn abandon(&self, meeting_id: &str, what: Abandon) -> bool {
        self.lock().abandon(meeting_id, what)
    }

    /// Wake the worker without giving it anything to do, so it re-reads the
    /// lifecycle facts now rather than at the next tick.
    fn wake(&self) {
        self.ready.notify_all();
    }

    /// Refuse anything new and wake the worker so it can stop.
    fn close(&self) {
        let mut state = self.lock();
        state.closed = true;
        // Anything still queued is answered rather than left hanging.
        let mut taken: Vec<Pending> = state.finals.drain(..).collect();
        taken.extend(state.catchup.drain(..));
        taken.append(&mut state.speculative);
        drop(state);
        for pending in taken {
            pending.abandon();
        }
        self.ready.notify_all();
    }

    fn is_closed(&self) -> bool {
        self.lock().closed
    }

    fn finals_waiting(&self) -> usize {
        self.lock().finals_waiting()
    }

    /// Utterances waiting for text. Captions are not counted: a snapshot is not
    /// something that can be behind, and counting it would make Echo claim it is
    /// falling behind while it is keeping up perfectly.
    fn depth(&self) -> u32 {
        let state = self.lock();
        let running = state
            .running
            .as_ref()
            .is_some_and(|r| !r.kind.is_speculative()) as usize;
        (state.finals.len() + state.catchup.len() + running) as u32
    }
}

// ---------------------------------------------------------------------------
// The worker
// ---------------------------------------------------------------------------

/// State the async side reads without touching the engine thread.
#[derive(Default)]
struct Shared {
    loaded: AtomicBool,
    /// True while a meeting is being listened to or still has work outstanding.
    /// The grace period does not even start until this is false
    /// ([`EngineWorker::set_resident`]).
    resident: AtomicBool,
    /// The moment residency was last dropped, stamped by whoever dropped it.
    ///
    /// The grace period is measured from here, and it has to be: the worker only
    /// looks up every [`IDLE_TICK`], so a clock kept by the worker started the
    /// countdown at the *last tick before* the meeting's work finished and
    /// handed the weights back up to thirty seconds early (review of
    /// 2026-08-20, finding 3).
    idle_since: Mutex<Option<Instant>>,
    queue: JobQueue,
    backend: Mutex<BackendReport>,
    /// What to load when something needs the engine. `None` = nothing chosen
    /// yet, which is what a fresh install looks like.
    config: Mutex<Option<EngineConfig>>,
    /// Meetings whose queued work should be thrown away.
    cancelled: Mutex<HashSet<String>>,
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
    /// There is no idle period to pass in any more: the weights are held while
    /// [`EngineWorker::set_resident`] says a meeting needs them and released
    /// [`IDLE_GRACE`] after it stops saying so. Neither half is a setting
    /// (product decision of 2026-08-20).
    pub fn new() -> Self {
        let shared = Arc::new(Shared::default());
        let worker_shared = shared.clone();
        let thread = std::thread::Builder::new()
            .name("echo-speech".to_string())
            .spawn(move || run(worker_shared))
            .ok();

        Self {
            shared,
            thread: Mutex::new(thread),
        }
    }

    /// Say whether a meeting still needs the engine.
    ///
    /// `true` while audio is being captured **or** any of a meeting's
    /// post-meeting work is running or queued: catch-up, the speaker pass, the
    /// recap. While it holds, the weights stay put however long the room is
    /// silent — a meeting with a ten-minute gap in it is still one meeting, and
    /// unloading 1.6 GB in the middle of it only means loading it again
    /// (mantra 1's amendment of 2026-08-20).
    ///
    /// `false` arms the grace period, which starts *now* rather than at the last
    /// decode: the point of reference is when the meeting's work finished, not
    /// when the engine was last busy. The moment is stamped here and the worker
    /// is woken to read it, so the countdown is the whole of [`IDLE_GRACE`]
    /// rather than whatever was left of it since the last tick.
    pub fn set_resident(&self, resident: bool) {
        let was = self.shared.resident.swap(resident, Ordering::SeqCst);
        if was == resident {
            return;
        }
        if resident {
            *self.shared.idle_since.lock().expect("idle edge poisoned") = None;
            tracing::debug!("holding speech understanding for a meeting");
        } else {
            *self.shared.idle_since.lock().expect("idle edge poisoned") = Some(Instant::now());
            tracing::debug!(
                grace_secs = IDLE_GRACE.as_secs(),
                "nothing needs speech understanding; it goes back shortly"
            );
        }
        // Whichever way it went, the worker's idea of when the engine was last
        // needed has just changed. Waking it is what makes the edge the start of
        // the countdown instead of the tick that happens to notice it.
        self.shared.queue.wake();
    }

    /// Is something still holding the engine?
    pub fn is_resident(&self) -> bool {
        self.shared.resident.load(Ordering::SeqCst)
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
    /// The provenance comes from whichever catalog entry these weights are, so
    /// weights changing behind our back cannot leave a segment claiming it was
    /// written by the previous ones. An uncatalogued file (a developer pointing
    /// at their own weights) names itself after the file.
    ///
    /// The decoding numbers come from the level, always. There is one set of
    /// them, and they describe how hard to try rather than which weights to try
    /// with — an older model still serving a meeting is decoded the same way the
    /// new one will be.
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

        let config = EngineConfig {
            model_path: model_path.to_path_buf(),
            accelerator_path: accelerator_path.map(Path::to_path_buf),
            decode: catalog::default_preset().decode,
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
        self.shared.queue.push_control(Control::Load { reply })?;
        answer.await.map_err(|_| AsrError::EngineGone)?
    }

    /// Queue a job read back from disk, or anything else that wants the preset's
    /// full effort. Returns [`AsrError::QueueFull`] when the bounded queue is
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
        self.submit_with(job, DecodePlan::catch_up(), on_partial)
            .await
    }

    /// Queue a job with an explicit plan: what kind of work it is, and any words
    /// carried in from the utterance before it.
    ///
    /// This is what the live pipeline uses. A speculative job answers
    /// [`AsrError::Cancelled`] when a newer look at the same speech overtook it,
    /// which is a normal event and not a failure.
    pub async fn submit_with(
        &self,
        job: TranscribeJob,
        plan: DecodePlan,
        on_partial: Option<PartialFn>,
    ) -> Result<Transcription, AsrError> {
        // Speculative work never waits for room: by the time there is any, the
        // speech it was a guess at has been said, decoded and written down.
        let wait_for_room = !job.droppable && !plan.kind.is_speculative();
        let (reply, answer) = tokio::sync::oneshot::channel();
        let mut pending = Pending {
            job: Box::new(job),
            plan,
            on_partial,
            reply,
            seq: 0,
        };
        loop {
            match self.shared.queue.push_job(pending) {
                Ok(overtook) => {
                    if overtook {
                        // The decode in flight is a caption this snapshot has
                        // left behind. Stop it rather than pay for both.
                        self.shared.abort.store(true, Ordering::SeqCst);
                    }
                    break;
                }
                Err(PushError::Gone) => return Err(AsrError::EngineGone),
                Err(PushError::Full(returned)) => {
                    if !wait_for_room {
                        return Err(AsrError::QueueFull);
                    }
                    pending = returned;
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
            }
        }
        answer.await.map_err(|_| AsrError::EngineGone)?
    }

    /// Drop the weights now. Called on idle timeout and before quitting.
    pub async fn release(&self) {
        let _ = self.shared.queue.push_control(Control::Release);
    }

    /// Are the weights in memory right now?
    pub fn is_loaded(&self) -> bool {
        self.shared.loaded.load(Ordering::SeqCst)
    }

    /// Utterances waiting. Surfaced to the person only as "catching up".
    pub fn queue_depth(&self) -> u32 {
        self.shared.queue.depth()
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
        let running_is_ours = self.shared.queue.abandon(meeting_id, Abandon::Everything);
        if running_is_ours || self.shared.queue.running_meeting().as_deref() == Some(meeting_id) {
            self.shared.abort.store(true, Ordering::SeqCst);
        }
    }

    /// Throw away every caption still queued for a meeting, and stop the one in
    /// flight. The finals are left alone: they are what the transcript is made
    /// of.
    ///
    /// Called when a recording stops, before the disk pass is allowed to think
    /// about holes.
    pub fn abandon_speculative(&self, meeting_id: &str) {
        if self.shared.queue.abandon(meeting_id, Abandon::Speculative) {
            self.shared.abort.store(true, Ordering::SeqCst);
        }
    }

    /// Hand the meeting over to the disk pass, hard (codex finding 7).
    ///
    /// Everything live — captions and queued utterances both — is dropped, and
    /// the live decode in flight is stopped. Whatever had no text against it is
    /// simply a hole on disk, which is exactly what the catch-up pass is for. It
    /// is the *detached* half-finished drain that produces duplicate segments,
    /// not an honest gap.
    ///
    /// Work read back from disk is left alone: this is what makes it
    /// authoritative from here on.
    pub fn abandon_live(&self, meeting_id: &str) {
        if self.shared.queue.abandon(meeting_id, Abandon::Live) {
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
        let _ = self.shared.queue.push_control(Control::Shutdown);
        self.shared.queue.close();
        if let Some(handle) = self.thread.lock().expect("engine thread poisoned").take() {
            let _ = handle.join();
        }
    }
}

impl Default for EngineWorker {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for EngineWorker {
    fn drop(&mut self) {
        self.shared.queue.close();
        if let Some(handle) = self.thread.lock().expect("engine thread poisoned").take() {
            let _ = handle.join();
        }
    }
}

/// Should the weights go back now?
///
/// Pure, so the lifecycle rule is testable without a thread or 1.6 GB on disk:
/// resident beats everything, and the grace is measured from the last moment
/// anything needed the engine — a decode, a load, or a meeting holding it.
fn should_release(loaded: bool, resident: bool, idle_for: Duration) -> bool {
    loaded && !resident && idle_for >= IDLE_GRACE
}

/// When the grace period started: the later of "the engine was last busy" and
/// "the last meeting let go of it".
///
/// Both matter. The edge is normally later — the meeting's last job finishes,
/// then residency drops — and taking it is what gives the countdown its full
/// length instead of whatever was left of it since the last tick. But a decode
/// *after* the edge (a one-off transcription with nothing holding the engine)
/// has to count too, or that work would be measured as idle time.
fn grace_starts_at(last_needed: Instant, idle_since: Option<Instant>) -> Instant {
    match idle_since {
        Some(edge) => last_needed.max(edge),
        None => last_needed,
    }
}

/// Does a final leaving the queue have to narrow its beam? See
/// [`FINAL_BACKLOG_DEGRADE_AT`].
///
/// `queued_including_this_one` is the depth of the finals lane as it was when
/// this final was still in it: the ones still waiting, plus the one now being
/// decided about. That is the number the threshold is written in terms of, and
/// counting it after the pop (against a strict `>`) is what made the valve open
/// two utterances later than documented.
fn falling_behind(queued_including_this_one: usize) -> bool {
    queued_including_this_one >= FINAL_BACKLOG_DEGRADE_AT
}

/// The last rung of decode recovery: throw the weights away and read them again.
///
/// [`Engine::reset_state`] rebuilds the per-decode scratch state, which is the
/// right answer when one decode left a half-built graph behind. It is not an
/// answer to anything wrong with the context itself — a GPU backend that has
/// gone bad, an encoder companion that will not run — and the field has shown
/// that a broken engine can fail every utterance of a meeting while
/// `reset_state` reports success each time.
///
/// Once per episode, deliberately: reloading 1.6 GB is expensive, and an engine
/// that fails again after a fresh load is not going to be fixed by a third one.
/// A single successful decode ends the episode and re-arms this.
fn should_reload_weights(consecutive_failures: u32, already_reloaded: bool) -> bool {
    !already_reloaded && consecutive_failures >= RELOAD_AFTER_FAILURES
}

/// The engine thread. Owns the weights and nothing else owns them.
fn run(shared: Arc<Shared>) {
    let mut engine: Option<Engine> = None;
    let mut last_needed = Instant::now();
    let mut policies: HashMap<String, LanguagePolicy> = HashMap::new();
    // Whether new finals are currently narrowing their beam, so the log gets one
    // line per episode rather than one per utterance.
    let mut behind = false;
    // Decodes that failed back to back, and whether this episode has already had
    // its one reload (see [`should_reload_weights`]).
    let mut consecutive_failures: u32 = 0;
    let mut reloaded_this_episode = false;

    loop {
        match shared.queue.take(IDLE_TICK) {
            Some(Task::Control(Control::Shutdown)) => break,
            Some(Task::Control(Control::Release)) => {
                if engine.take().is_some() {
                    shared.loaded.store(false, Ordering::SeqCst);
                    tracing::info!("speech engine released");
                }
            }
            Some(Task::Control(Control::Load { reply })) => {
                let answer = ensure_loaded(&mut engine, &shared).map(|e| e.backend());
                last_needed = Instant::now();
                let _ = reply.send(answer);
            }
            Some(Task::Job(pending)) => {
                let Pending {
                    job,
                    mut plan,
                    on_partial,
                    reply,
                    ..
                } = pending;
                if plan.kind == JobKind::Final {
                    // This final has already been popped, so the lane it came
                    // from was one deeper than it is now.
                    let queued = shared.queue.finals_waiting() + 1;
                    plan.behind = falling_behind(queued);
                    if plan.behind != behind {
                        behind = plan.behind;
                        if behind {
                            tracing::info!(
                                queued,
                                "live text is falling behind; decoding it more cheaply until it catches up"
                            );
                        } else {
                            tracing::info!("live text caught up; back to full accuracy");
                        }
                    }
                }
                let answer = run_job(&mut engine, &shared, &mut policies, *job, plan, on_partial);
                last_needed = Instant::now();
                // A run of failures is about the engine, not the audio. Rebuild
                // the state after each one (that happens in `Engine`), and after
                // enough of them read the weights again — loudly, because a
                // meeting is being lost while this happens.
                match &answer {
                    Ok(_) => {
                        consecutive_failures = 0;
                        reloaded_this_episode = false;
                    }
                    Err(AsrError::Transcribe(_)) => {
                        consecutive_failures += 1;
                        if should_reload_weights(consecutive_failures, reloaded_this_episode) {
                            tracing::error!(
                                consecutive_failures,
                                "the speech engine could not read {consecutive_failures} stretches in a row; \
                                 loading the weights again"
                            );
                            engine = None;
                            shared.loaded.store(false, Ordering::SeqCst);
                            reloaded_this_episode = true;
                        }
                    }
                    // A cancelled or superseded job says nothing about the
                    // engine's health, and a load failure is reported already.
                    Err(_) => {}
                }
                let _ = reply.send(answer);
            }
            None => {
                if shared.queue.is_closed() {
                    break;
                }
                // A meeting holding the engine counts as needing it, so the grace
                // period only ever starts once nothing does.
                if shared.resident.load(Ordering::SeqCst) {
                    last_needed = Instant::now();
                } else {
                    // The countdown runs from the moment residency was dropped,
                    // which the dropper stamped — not from this tick, and not
                    // from the tick before the edge, which is what used to cut
                    // the grace period short by up to `IDLE_TICK`.
                    let edge = *shared.idle_since.lock().expect("idle edge poisoned");
                    last_needed = grace_starts_at(last_needed, edge);
                    if should_release(engine.is_some(), false, last_needed.elapsed()) {
                        engine = None;
                        shared.loaded.store(false, Ordering::SeqCst);
                        tracing::info!("speech engine released after being idle");
                    }
                }
            }
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
    plan: DecodePlan,
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
    shared.queue.begin(Running {
        meeting_id: job.meeting_id.clone(),
        channel: job.channel,
        kind: plan.kind,
        covers_to_ms: job.t_end_ms(),
    });
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
        } else if plan.kind.is_speculative() {
            // A caption never pays for a detection pass of its own: it borrows
            // whatever the meeting has settled on, and lets whisper decide when
            // there is nothing to borrow yet. Detection is an encode, and this
            // hypothesis is replaced in three seconds.
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

    let mut result = engine.transcribe(&job, &plan, on_partial);
    shared.queue.finish();

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

    /// One model, three lanes. The lane decides beam-versus-greedy; the model
    /// never changes with it (product decision of 2026-08-20).
    #[test]
    fn the_lane_decides_beam_search_or_greedy_and_the_model_never_changes() {
        let level = catalog::default_preset().decode;
        match sampling_strategy(&level) {
            SamplingStrategy::BeamSearch { beam_size, .. } => assert_eq!(beam_size, 5),
            other => panic!("the disk pass should search beams, got {other:?}"),
        }
        match sampling_strategy(&level.live_final()) {
            SamplingStrategy::BeamSearch { beam_size, .. } => assert_eq!(
                beam_size, 5,
                "a live final is decoded at the same width as the disk pass"
            ),
            other => panic!("a live final should search beams, got {other:?}"),
        }
        match sampling_strategy(&level.speculative()) {
            SamplingStrategy::Greedy { best_of } => assert_eq!(best_of, 1),
            other => panic!("a caption should be greedy, got {other:?}"),
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

    /// The regression of 2026-08-20, in one assertion.
    ///
    /// whisper.cpp asks the abort callback once per encode and turns "yes" into
    /// `-6`, so a callback that answers anything other than our own flag turns
    /// a whole meeting into `Generic whisper error … code: -6`. This calls
    /// exactly what whisper.cpp calls, with exactly the pointer whisper.cpp is
    /// given, so a wrongly-typed trampoline or a dangling user-data pointer
    /// fails here instead of in somebody's meeting.
    #[test]
    fn whisper_only_ever_aborts_a_decode_when_we_asked_it_to() {
        let flag = Arc::new(AtomicBool::new(false));
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        let (callback, user_data) = install_abort_callback(&mut params, &flag);
        let callback = callback.expect("an abort callback is installed");

        // SAFETY: `flag` is alive for the whole test, which is the same
        // guarantee `Engine` gives for the whole decode.
        let asked = || unsafe { callback(user_data) };

        // A quiet flag, many times over: one decode of a 28-second window asks
        // this once per encode, and every one of them has to say "carry on".
        for i in 0..10_000 {
            assert!(
                !asked(),
                "call {i} told whisper.cpp to abort an encode nobody cancelled, \
                 which it reports as error -6 and no amount of padding or state \
                 rebuilding can cure"
            );
        }

        flag.store(true, Ordering::SeqCst);
        assert!(asked(), "a real cancellation has to get through");

        flag.store(false, Ordering::SeqCst);
        assert!(!asked(), "and clearing it has to be believed again");
    }

    #[test]
    fn a_run_of_failures_reloads_the_weights_once_and_only_once() {
        // One or two bad stretches are the audio's fault; the state rebuild in
        // `Engine` is the whole response.
        assert!(!should_reload_weights(0, false));
        assert!(!should_reload_weights(1, false));
        assert!(!should_reload_weights(RELOAD_AFTER_FAILURES - 1, false));
        // Three in a row is the engine's fault.
        assert!(should_reload_weights(RELOAD_AFTER_FAILURES, false));
        assert!(should_reload_weights(RELOAD_AFTER_FAILURES + 40, false));
        // But a reload already happened in this episode: an engine that fails
        // after a fresh load will not be fixed by re-reading 1.6 GB per
        // utterance.
        assert!(!should_reload_weights(RELOAD_AFTER_FAILURES, true));
        assert!(!should_reload_weights(279, true));
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
        let worker = EngineWorker::new();
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
        let worker = EngineWorker::new();
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
        let worker = EngineWorker::new();
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
        let worker = EngineWorker::new();
        // Weights Echo no longer wants, which is exactly the case that matters:
        // while the new model downloads these are what a meeting is transcribed
        // with, and the segments have to say so.
        let old = catalog::entry(catalog::ids::SPEECH_TURBO).unwrap();
        let path = PathBuf::from("/nowhere").join(old.file_name);

        // Loading fails (the file is not there), but the configuration it chose
        // is what we are checking.
        let _ = worker.prewarm(&path, None).await;
        let cfg = worker
            .configured()
            .expect("prewarm configures before loading");
        assert_eq!(cfg.model_name, old.name);
        assert_eq!(cfg.model_revision, old.revision);
        assert_eq!(
            cfg.decode,
            catalog::default_preset().decode,
            "one set of decoding numbers, whichever weights are serving"
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
        let worker = EngineWorker::new();
        let first = EngineConfig {
            model_path: PathBuf::from("/nowhere/ggml-tiny.bin"),
            accelerator_path: None,
            decode: catalog::default_preset().decode,
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
        let worker = EngineWorker::new();
        worker.shutdown();
        let err = worker.load_now().await.unwrap_err();
        assert!(matches!(err, AsrError::EngineGone), "{err:?}");
        // Shutting down twice is safe.
        worker.shutdown();
    }

    #[tokio::test]
    async fn the_engine_settings_load_what_is_installed_and_refuse_to_guess() {
        let db = crate::db::connect_in_memory().await.unwrap();
        models::sync_catalog(&db).await.unwrap();
        let err = EngineConfig::from_settings(&db).await.unwrap_err();
        assert!(
            matches!(err, AsrError::NotInstalled),
            "no download means no engine, not a fallback: {err:?}"
        );

        repo::set_model_installed(
            &db,
            catalog::ids::SPEECH,
            true,
            Some("/speech/everyday.bin"),
        )
        .await
        .unwrap();
        let cfg = EngineConfig::from_settings(&db).await.unwrap();
        assert_eq!(cfg.model_path, PathBuf::from("/speech/everyday.bin"));
        assert_eq!(cfg.decode, catalog::default_preset().decode);
        assert_eq!(
            cfg.model_name,
            catalog::entry(catalog::ids::SPEECH).unwrap().name
        );
        assert_eq!(
            cfg.model_revision,
            catalog::entry(catalog::ids::SPEECH).unwrap().revision
        );
        assert!(
            cfg.accelerator_path.is_none(),
            "the companion is only used once it is installed"
        );
    }

    // -----------------------------------------------------------------------
    // Scheduling: who goes next on the one worker
    // -----------------------------------------------------------------------

    type Answer = tokio::sync::oneshot::Receiver<Result<Transcription, AsrError>>;

    /// One queued job, and the handle its caller is waiting on.
    fn queued(
        meeting: &str,
        channel: Channel,
        kind: JobKind,
        t_start_ms: i64,
        duration_ms: i64,
    ) -> (Pending, Answer) {
        let (reply, answer) = tokio::sync::oneshot::channel();
        let samples = vec![0.0f32; (duration_ms as usize) * 16];
        (
            Pending {
                job: Box::new(TranscribeJob {
                    meeting_id: meeting.to_string(),
                    channel,
                    t_start_ms,
                    samples,
                    droppable: kind.is_live(),
                    ..Default::default()
                }),
                plan: DecodePlan {
                    kind,
                    ..Default::default()
                },
                on_partial: None,
                reply,
                seq: 0,
            },
            answer,
        )
    }

    fn taken(task: Option<Task>) -> Pending {
        match task {
            Some(Task::Job(pending)) => pending,
            _ => panic!("expected a job"),
        }
    }

    /// A caption is worth having and worth nothing: it goes last, and it never
    /// delays the utterance a person keeps.
    #[test]
    fn utterances_go_before_captions_and_control_goes_before_both() {
        let mut queue = QueueState::default();
        let (caption, _a) = queued("m", Channel::Mic, JobKind::Speculative, 0, 3_000);
        let (disk, _b) = queued("m", Channel::Mic, JobKind::CatchUp, 0, 20_000);
        let (utterance, _c) = queued("m", Channel::Mic, JobKind::Final, 3_000, 2_000);
        // Queued in exactly the wrong order.
        queue.push(caption).unwrap();
        queue.push(disk).unwrap();
        queue.push(utterance).unwrap();
        let (control_reply, _d) = tokio::sync::oneshot::channel();
        queue.control.push_back(Control::Load {
            reply: control_reply,
        });

        assert!(
            matches!(queue.next(), Some(Task::Control(_))),
            "control first"
        );
        assert_eq!(
            taken(queue.next()).plan.kind,
            JobKind::Final,
            "the text a person keeps comes before anything else"
        );
        assert_eq!(
            taken(queue.next()).plan.kind,
            JobKind::CatchUp,
            "work off the disk beats a caption: it is the transcript"
        );
        assert_eq!(taken(queue.next()).plan.kind, JobKind::Speculative);
        assert!(queue.next().is_none());
    }

    /// The whole point of coalescing: two looks at the same speech are one job.
    #[test]
    fn a_newer_snapshot_replaces_the_one_waiting_rather_than_joining_it() {
        let mut queue = QueueState::default();
        let (first, older) = queued("m", Channel::Mic, JobKind::Speculative, 0, 3_000);
        let (second, _newer) = queued("m", Channel::Mic, JobKind::Speculative, 0, 6_000);
        queue.push(first).unwrap();
        queue.push(second).unwrap();

        assert_eq!(
            queue.speculative.len(),
            1,
            "one channel never has two captions queued"
        );
        assert_eq!(
            queue.speculative[0].covers_to_ms(),
            6_000,
            "the newer look is the one kept"
        );
        // The abandoned one is answered, not left hanging: its caller stops
        // waiting and says nothing to anybody.
        assert!(matches!(
            older.blocking_recv(),
            Ok(Err(AsrError::Cancelled))
        ));

        // The other channel is untouched: it is different speech.
        let (system, _s) = queued("m", Channel::System, JobKind::Speculative, 0, 3_000);
        queue.push(system).unwrap();
        assert_eq!(queue.speculative.len(), 2);
    }

    #[test]
    fn one_channel_that_never_stops_talking_cannot_starve_the_other() {
        let mut queue = QueueState::default();
        let mut waiting = Vec::new();
        for i in 0..3 {
            let (mic, answer) = queued("m", Channel::Mic, JobKind::Final, i * 1_000, 900);
            waiting.push(answer);
            queue.push(mic).unwrap();
        }
        let (system, answer) = queued("m", Channel::System, JobKind::Final, 500, 900);
        waiting.push(answer);
        queue.push(system).unwrap();

        let order: Vec<Channel> = std::iter::from_fn(|| queue.next())
            .map(|task| match task {
                Task::Job(pending) => pending.job.channel,
                Task::Control(_) => panic!("no control queued"),
            })
            .collect();
        assert_eq!(
            order,
            vec![Channel::Mic, Channel::System, Channel::Mic, Channel::Mic],
            "the system channel gets its turn instead of waiting out the monologue"
        );
    }

    /// Review finding 5, exactly: the newest speech must never be the thing that
    /// gets dropped.
    #[test]
    fn a_full_queue_loses_an_old_guess_before_it_loses_new_speech() {
        let mut queue = QueueState::default();
        let (caption, guess) = queued("m", Channel::Mic, JobKind::Speculative, 0, 3_000);
        queue.push(caption).unwrap();
        let mut waiting = Vec::new();
        while queue.waiting() < QUEUE_CAPACITY {
            let (utterance, answer) = queued(
                "m",
                Channel::Mic,
                JobKind::Final,
                queue.waiting() as i64 * 1_000,
                900,
            );
            queue.push(utterance).unwrap();
            waiting.push(answer);
        }

        let (newest, _a) = queued("m", Channel::System, JobKind::Final, 9_000, 900);
        assert!(
            queue.push(newest).is_ok(),
            "new speech is kept; the old guess is what goes"
        );
        assert!(matches!(
            guess.blocking_recv(),
            Ok(Err(AsrError::Cancelled))
        ));
        assert!(queue.speculative.is_empty());

        // With nothing left to give up, the queue says so and the audio on disk
        // becomes catch-up's problem instead.
        let (one_too_many, _b) = queued("m", Channel::Mic, JobKind::Final, 10_000, 900);
        assert!(matches!(queue.push(one_too_many), Err(PushError::Full(_)),));
        // …and a caption offered to a full queue simply does not happen.
        let (late_caption, dropped) = queued("m", Channel::Mic, JobKind::Speculative, 9_000, 3_000);
        assert!(queue.push(late_caption).is_ok());
        assert!(matches!(
            dropped.blocking_recv(),
            Ok(Err(AsrError::Cancelled))
        ));
        assert!(queue.speculative.is_empty());
    }

    #[test]
    fn a_much_newer_snapshot_cancels_the_caption_in_flight() {
        let mut queue = QueueState {
            running: Some(Running {
                meeting_id: "m".into(),
                channel: Channel::Mic,
                kind: JobKind::Speculative,
                covers_to_ms: 3_000,
            }),
            ..Default::default()
        };

        // One step behind: let it finish, it is nearly done.
        let (close, _a) = queued("m", Channel::Mic, JobKind::Speculative, 0, 6_000);
        assert!(!queue.push(close).unwrap());
        // Well behind: stop paying for a hypothesis that is already stale.
        let (far, _b) = queued("m", Channel::Mic, JobKind::Speculative, 0, 8_000);
        assert!(queue.push(far).unwrap());

        // A final in flight is never cancelled by a caption.
        queue.running = Some(Running {
            meeting_id: "m".into(),
            channel: Channel::Mic,
            kind: JobKind::Final,
            covers_to_ms: 3_000,
        });
        let (later, _c) = queued("m", Channel::Mic, JobKind::Speculative, 0, 20_000);
        assert!(!queue.push(later).unwrap());
    }

    /// The stop handoff (codex finding 7): live work is abandoned outright, and
    /// what the disk pass is doing is left alone.
    #[test]
    fn handing_a_meeting_over_drops_every_live_job_and_nothing_else() {
        let mut queue = QueueState::default();
        let (caption, a) = queued("m", Channel::Mic, JobKind::Speculative, 0, 3_000);
        let (utterance, b) = queued("m", Channel::Mic, JobKind::Final, 3_000, 2_000);
        let (disk, c) = queued("m", Channel::Mic, JobKind::CatchUp, 0, 20_000);
        let (other, d) = queued("other", Channel::Mic, JobKind::Final, 0, 2_000);
        queue.push(caption).unwrap();
        queue.push(utterance).unwrap();
        queue.push(disk).unwrap();
        queue.push(other).unwrap();
        queue.running = Some(Running {
            meeting_id: "m".into(),
            channel: Channel::Mic,
            kind: JobKind::Final,
            covers_to_ms: 1_000,
        });

        assert!(
            queue.abandon("m", Abandon::Live),
            "the decode in flight is live work too, so it is stopped"
        );
        assert!(matches!(a.blocking_recv(), Ok(Err(AsrError::Cancelled))));
        assert!(matches!(b.blocking_recv(), Ok(Err(AsrError::Cancelled))));
        assert_eq!(queue.catchup.len(), 1, "the disk pass is authoritative now");
        assert_eq!(
            queue.finals.len(),
            1,
            "another meeting is none of our business"
        );
        drop((c, d));

        // Only the captions, for a meeting that is still recording.
        let mut queue = QueueState::default();
        let (caption, a) = queued("m", Channel::Mic, JobKind::Speculative, 0, 3_000);
        let (utterance, b) = queued("m", Channel::Mic, JobKind::Final, 3_000, 2_000);
        queue.push(caption).unwrap();
        queue.push(utterance).unwrap();
        assert!(!queue.abandon("m", Abandon::Speculative));
        assert!(matches!(a.blocking_recv(), Ok(Err(AsrError::Cancelled))));
        assert_eq!(queue.finals.len(), 1);
        drop(b);
    }

    /// A caption is not something that can be "behind": counting it would have
    /// Echo apologise for keeping up.
    #[test]
    fn captions_are_not_counted_as_a_backlog() {
        let queue = JobQueue::default();
        let (caption, _a) = queued("m", Channel::Mic, JobKind::Speculative, 0, 3_000);
        queue.push_job(caption).ok();
        assert_eq!(queue.depth(), 0);

        let (utterance, _b) = queued("m", Channel::Mic, JobKind::Final, 0, 2_000);
        queue.push_job(utterance).ok();
        assert_eq!(queue.depth(), 1);
    }

    /// The 2026-08-20 product decision, in one test: a caption is still the
    /// cheapest decode there is, and a final is now as good as the disk pass.
    #[test]
    fn a_final_is_decoded_at_catch_up_quality_and_a_caption_stays_a_guess() {
        let preset = catalog::preset("everyday").unwrap().decode;

        let caption = Decoding::for_job(JobKind::Speculative, &preset, false);
        assert_eq!(caption.params.max_attempts(), 1, "no fallback on a guess");
        assert!(!caption.params.uses_beam_search(), "captions stay greedy");
        assert!(caption.single_segment, "one replaceable hypothesis");
        assert!(!caption.timestamps);

        let live = Decoding::for_job(JobKind::Final, &preset, false);
        assert_eq!(live.params.max_attempts(), 2, "one modest fallback");
        assert_eq!(
            live.params.beam_size, preset.beam_size,
            "text a person keeps is decoded as well as the recording would be"
        );
        assert!(!live.single_segment, "natural segments on a final");
        assert!(live.timestamps, "coverage needs them");

        let disk = Decoding::for_job(JobKind::CatchUp, &preset, false);
        assert_eq!(disk.params, preset, "the disk pass keeps the whole preset");
        assert!(disk.timestamps);
    }

    /// The safety valve: while finals pile up, new ones narrow, and captions and
    /// the disk pass are not touched.
    #[test]
    fn finals_narrow_their_beam_only_while_the_queue_is_deep() {
        assert!(!falling_behind(0));
        assert!(
            !falling_behind(FINAL_BACKLOG_DEGRADE_AT - 1),
            "two people talking at once is not a backlog"
        );
        assert!(
            falling_behind(FINAL_BACKLOG_DEGRADE_AT),
            "the threshold is the number in the documentation, exactly"
        );
        assert!(falling_behind(FINAL_BACKLOG_DEGRADE_AT + 1));

        let preset = catalog::preset("everyday").unwrap().decode;
        let behind = Decoding::for_job(JobKind::Final, &preset, true);
        assert_eq!(behind.params.beam_size, catalog::LIVE_DEGRADED_BEAM);
        assert_eq!(
            behind.params.max_attempts(),
            Decoding::for_job(JobKind::Final, &preset, false)
                .params
                .max_attempts(),
            "the valve narrows the beam and changes nothing else"
        );
        assert!(behind.timestamps, "a degraded final is still a final");

        let caption = Decoding::for_job(JobKind::Speculative, &preset, true);
        assert_eq!(
            caption.params,
            preset.speculative(),
            "captions are untouched"
        );
        let disk = Decoding::for_job(JobKind::CatchUp, &preset, true);
        assert_eq!(disk.params, preset, "the disk pass is untouched");
    }

    /// The queue counts what "behind" means: finished utterances waiting.
    #[test]
    fn only_finished_utterances_count_towards_falling_behind() {
        let queue = JobQueue::default();
        let mut answers = Vec::new();
        for i in 0..(FINAL_BACKLOG_DEGRADE_AT + 1) as i64 {
            let (caption, a) = queued("m", Channel::Mic, JobKind::Speculative, i * 3_000, 3_000);
            queue.push_job(caption).ok();
            answers.push(a);
        }
        assert!(
            !falling_behind(queue.finals_waiting()),
            "captions are not a backlog"
        );

        for i in 0..FINAL_BACKLOG_DEGRADE_AT as i64 {
            let (utterance, a) = queued("m", Channel::Mic, JobKind::Final, i * 2_000, 2_000);
            queue.push_job(utterance).ok();
            answers.push(a);
        }
        // Exactly the documented number of finished utterances is queued, and the
        // valve is read the way the worker reads it: after popping the one being
        // decided about, counting that one back in.
        let popped = taken(queue.take(Duration::from_millis(0)));
        assert_eq!(popped.plan.kind, JobKind::Final);
        assert!(
            falling_behind(queue.finals_waiting() + 1),
            "three queued finals is the threshold, not five"
        );
    }

    /// The valve as the worker actually reads it: one utterance per channel is
    /// two people talking, and the third is a queue.
    #[test]
    fn the_valve_opens_at_the_number_it_documents() {
        let queue = JobQueue::default();
        let mut answers = Vec::new();
        for i in 0..(FINAL_BACKLOG_DEGRADE_AT - 1) as i64 {
            let (utterance, a) = queued("m", Channel::Mic, JobKind::Final, i * 2_000, 2_000);
            queue.push_job(utterance).ok();
            answers.push(a);
        }
        let popped = taken(queue.take(Duration::from_millis(0)));
        assert!(
            !falling_behind(queue.finals_waiting() + 1),
            "two queued finals decode at full width"
        );
        drop(popped);
    }

    // -----------------------------------------------------------------------
    // Lifecycle: resident while listening, released after the grace period
    // -----------------------------------------------------------------------

    #[test]
    fn the_grace_period_only_starts_once_nothing_needs_the_engine() {
        // Armed: nothing holds the engine and the grace has run out.
        assert!(should_release(true, false, IDLE_GRACE));
        assert!(should_release(true, false, IDLE_GRACE * 2));
        // Disarmed: a meeting is still being listened to, or its jobs are still
        // running. However long that takes, the weights stay.
        assert!(!should_release(true, true, IDLE_GRACE * 100));
        // Armed but not yet due.
        assert!(!should_release(true, false, IDLE_GRACE / 2));
        // Nothing loaded, nothing to release.
        assert!(!should_release(false, false, IDLE_GRACE * 100));
    }

    /// The grace period starts at the edge, not at the tick that noticed it.
    #[test]
    fn the_countdown_starts_when_the_meeting_let_go_not_when_the_worker_looked() {
        let tick = Instant::now();
        // The worker's last tick was IDLE_TICK ago; the meeting let go just now.
        // Measuring from the tick would hand the weights back `IDLE_TICK` early.
        let long_ago = tick - IDLE_TICK;
        let edge = tick;
        assert_eq!(grace_starts_at(long_ago, Some(edge)), edge);
        assert!(!should_release(
            true,
            false,
            grace_starts_at(long_ago, Some(edge)).elapsed() + IDLE_GRACE - IDLE_TICK
        ));

        // A decode after the edge is the engine being needed again, and it wins.
        let decode = tick + Duration::from_secs(1);
        assert_eq!(grace_starts_at(decode, Some(edge)), decode);
        // Nothing ever let go: the last decode is all there is to go on.
        assert_eq!(grace_starts_at(decode, None), decode);
    }

    #[tokio::test]
    async fn letting_go_stamps_the_moment_the_countdown_starts() {
        let worker = EngineWorker::new();
        assert!(
            worker.shared.idle_since.lock().unwrap().is_none(),
            "nothing has let go of a fresh worker"
        );

        worker.set_resident(true);
        assert!(
            worker.shared.idle_since.lock().unwrap().is_none(),
            "a meeting holding the engine is not a countdown"
        );

        worker.set_resident(false);
        let edge = worker
            .shared
            .idle_since
            .lock()
            .unwrap()
            .expect("letting go stamps the edge");
        assert!(
            edge.elapsed() < Duration::from_secs(1),
            "the edge is now, not the last tick"
        );

        // Taking it back clears the countdown outright.
        worker.set_resident(true);
        assert!(worker.shared.idle_since.lock().unwrap().is_none());
        worker.shutdown();
    }

    #[tokio::test]
    async fn a_worker_holds_nothing_until_something_needs_it_and_says_when_it_is_held() {
        let worker = EngineWorker::new();
        assert!(!worker.is_loaded());
        assert!(!worker.is_resident(), "a fresh worker holds nothing");

        worker.set_resident(true);
        assert!(worker.is_resident());
        // Idempotent: the session layer refreshes this on every job that ends.
        worker.set_resident(true);
        assert!(worker.is_resident());

        worker.set_resident(false);
        assert!(!worker.is_resident());
        worker.shutdown();
    }
}
