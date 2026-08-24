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
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};

use crate::audio::{AudioError, TARGET_SAMPLE_RATE};
use crate::types::Channel;

/// Longest utterance handed downstream. Longer speech is cut near this limit —
/// at a real pause when there is one, otherwise with audio carried over into the
/// continuation (see [`Segmenter::force_cut`]).
///
/// The speech engine works in 30 s windows, so staying under that with room for
/// padding on both sides means an utterance never has to be split again later.
pub const MAX_UTTERANCE_MS: i64 = 24_000;

/// Shortest utterance handed downstream.
///
/// The speech engine cannot read a window under a second — whisper.cpp fails to
/// encode it and answers an error instead of text. That is not a reason to throw
/// the words away: a meeting is full of short answers ("yeah", "no, Tuesday"),
/// and each one is a line of the transcript.
///
/// So an utterance that would come out shorter than this is not trimmed back to
/// its padding. It keeps the silence it already has around it — audio that was
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

/// [`MIN_UTTERANCE_MS`] as whole windows, rounded up.
///
/// Keeping every position a whole number of windows is not tidiness: it makes
/// every offset an exact number of milliseconds, so consecutive utterances tile
/// the timeline with no rounding gap between them — and it lets the segmenter
/// keep one speech/silence decision per window alongside the audio.
pub const MIN_UTTERANCE_SAMPLES: usize =
    (MIN_UTTERANCE_MS as usize * TARGET_SAMPLE_RATE as usize / 1_000).div_ceil(WINDOW_SAMPLES)
        * WINDOW_SAMPLES;

/// How far back a forced cut looks for a real pause to cut at.
const FORCED_CUT_SEARCH_MS: i64 = 3_000;

/// Most audio a [`Segmenter::snapshot`] carries: the tail of what is open.
///
/// Deliberately a little more than the live pipeline's caption window, so the
/// consumer is the one that decides how much of the tail to decode and this
/// side is never the reason a caption is short. Bigger would just be copied and
/// thrown away every tick.
pub const SNAPSHOT_TAIL_MS: i64 = 12_000;

// ---------------------------------------------------------------------------
// Per-channel settings
// ---------------------------------------------------------------------------

/// Everything about *when* speech starts and stops, per channel.
///
/// The two channels are not the same problem. The laptop microphone is far-field
/// and noisy, its level varies with where the person sits, and a false negative
/// there loses words that exist nowhere else — so it is biased towards recall
/// (low thresholds, generous padding, a slower close). What the computer plays
/// arrives clean from a conferencing app, so it can be stricter and close
/// faster; the cost of being wrong is one extra decode, not a missing sentence.
///
/// These are starting values from real Italian meetings, not laws: everything
/// here is meant to be tuned against probability traces, which is why it is one
/// struct rather than a scattering of constants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VadSettings {
    /// Probability at which speech is considered to have started.
    pub enter: f32,
    /// …and the lower bar it has to fall below to be considered over. The gap
    /// stops a detector hovering around the threshold from shredding a sentence.
    pub exit: f32,
    /// Speech has to last this long before an utterance is opened.
    ///
    /// Without it a single 32 ms false positive becomes a padded, second-long
    /// job for the speech engine — a whisper the meeting never contained. Kept
    /// well under Silero's own 250 ms default because Italian acknowledgements
    /// ("sì", "no", "ok") really are that short.
    pub min_voiced_ms: i64,
    /// Silence this long ends an utterance.
    pub silence_tail_ms: i64,
    /// Audio kept *before* speech was detected, so a quiet consonant at the
    /// start of a word is not clipped off.
    pub pre_pad_ms: i64,
    /// Audio kept *after* the last voiced window.
    pub post_pad_ms: i64,
    /// Longest utterance handed downstream before a cut is forced.
    pub max_utterance_ms: i64,
    /// Audio carried into the continuation when a cut has to go through speech,
    /// so the two pieces share a boundary the engine can read across.
    pub forced_overlap_ms: i64,
    /// The shortest real pause a forced cut will settle for instead of cutting
    /// through speech.
    pub min_forced_silence_ms: i64,
}

impl VadSettings {
    /// The microphone: biased towards hearing everything.
    pub const fn mic() -> Self {
        Self {
            enter: 0.38,
            exit: 0.22,
            min_voiced_ms: 96,
            silence_tail_ms: 400,
            pre_pad_ms: 320,
            post_pad_ms: 288,
            max_utterance_ms: MAX_UTTERANCE_MS,
            forced_overlap_ms: 750,
            min_forced_silence_ms: 100,
        }
    }

    /// What the computer plays: cleaner, so stricter and quicker to close.
    pub const fn system() -> Self {
        Self {
            enter: 0.48,
            exit: 0.32,
            min_voiced_ms: 96,
            silence_tail_ms: 320,
            pre_pad_ms: 224,
            post_pad_ms: 288,
            max_utterance_ms: MAX_UTTERANCE_MS,
            forced_overlap_ms: 750,
            min_forced_silence_ms: 100,
        }
    }

    pub const fn for_channel(channel: Channel) -> Self {
        match channel {
            Channel::System => Self::system(),
            _ => Self::mic(),
        }
    }

    /// Whole windows of look-behind. Rounded *down*: padding is a courtesy, and
    /// every offset staying an exact number of windows matters more.
    pub const fn pre_pad_samples(&self) -> usize {
        whole_windows_down(self.pre_pad_ms)
    }

    pub const fn post_pad_samples(&self) -> usize {
        whole_windows_down(self.post_pad_ms)
    }

    /// Voiced windows needed before an utterance opens, at least one.
    pub const fn min_voiced_windows(&self) -> usize {
        let windows = (self.min_voiced_ms / WINDOW_MS) as usize;
        if windows == 0 {
            1
        } else {
            windows
        }
    }

    pub const fn max_samples(&self) -> usize {
        whole_windows_down(self.max_utterance_ms)
    }

    /// Whole windows of audio carried across a forced cut. Rounded down, so
    /// 750 ms is 736 ms of real overlap.
    pub const fn overlap_samples(&self) -> usize {
        whole_windows_down(self.forced_overlap_ms)
    }

    /// Silent windows that make a pause worth cutting at, at least two.
    pub const fn min_forced_silence_windows(&self) -> usize {
        let windows = (self.min_forced_silence_ms as usize).div_ceil(WINDOW_MS as usize);
        if windows < 2 {
            2
        } else {
            windows
        }
    }
}

impl Default for VadSettings {
    fn default() -> Self {
        Self::mic()
    }
}

/// How many milliseconds of the first `samples` of an utterance were speech.
///
/// `voiced` holds one decision per 32 ms window, alongside the audio; every
/// length the segmenter works in is a whole number of windows, so this is a
/// count rather than an interpolation. Windows the emitted piece does not reach
/// are not its business.
fn voiced_ms_of(voiced: &[bool], samples: usize) -> i64 {
    let windows = (samples / WINDOW_SAMPLES).min(voiced.len());
    voiced[..windows].iter().filter(|v| **v).count() as i64 * WINDOW_MS
}

/// Milliseconds as samples, rounded down to whole windows.
const fn whole_windows_down(ms: i64) -> usize {
    if ms <= 0 {
        return 0;
    }
    (ms as usize * TARGET_SAMPLE_RATE as usize / 1_000) / WINDOW_SAMPLES * WINDOW_SAMPLES
}

/// A stretch of one channel that contains speech.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Utterance {
    pub channel: Channel,
    pub t_start_ms: i64,
    pub t_end_ms: i64,
    /// 16 kHz mono audio for this stretch, padding included.
    pub samples: Vec<f32>,
    /// True when this utterance was cut *through speech* at
    /// [`VadSettings::max_utterance_ms`] rather than ending at a pause.
    ///
    /// It means two things at once, and the speech engine needs both: the text
    /// stops mid-thought, so the accepted words should be carried into the next
    /// decode as context; and the next utterance begins with
    /// [`VadSettings::forced_overlap_ms`] of *this* utterance's audio, so the
    /// same words may be transcribed twice and the join has to be reconciled
    /// rather than concatenated.
    ///
    /// A forced cut that found a real pause to land on is not truncated: nothing
    /// was split and no audio is shared.
    pub truncated: bool,
    /// How much of `samples` the detector actually called speech.
    ///
    /// Normally well under the duration: an utterance is padded at both ends
    /// and carries the pause it closed on. What it is *for* is telling a stretch of
    /// real speech apart from a window the detector opened on one cough — which
    /// is what a decoder turns into "Grazie." (see [`crate::asr::phantom`]).
    ///
    /// Zero when nothing measured it, which is why nothing should read this
    /// field directly; [`Utterance::measured_voice_ms`] answers "no measurement"
    /// with "assume it was all speech", so a missing detector can never cost
    /// words.
    pub voiced_ms: i64,
}

impl Utterance {
    pub fn duration_ms(&self) -> i64 {
        self.t_end_ms - self.t_start_ms
    }

    /// How many milliseconds of this stretch were speech.
    ///
    /// An utterance nobody measured — no detector on disk, a window handed over
    /// whole — reads as voice from end to end. Every filter downstream deletes
    /// on *evidence* of silence, never on the absence of evidence.
    pub fn measured_voice_ms(&self) -> i64 {
        let duration = self.duration_ms().max(0);
        if self.voiced_ms <= 0 {
            return duration;
        }
        self.voiced_ms.min(duration)
    }

    /// The same answer as a density, 0.0 to 1.0, so a stretch clipped to part of
    /// itself can carry it — see [`crate::asr::phantom::VoicedSpans`].
    pub fn voiced_ratio(&self) -> f32 {
        let duration = self.duration_ms();
        if duration <= 0 {
            return 1.0;
        }
        (self.measured_voice_ms() as f32 / duration as f32).clamp(0.0, 1.0)
    }
}

/// A look at speech that has **not finished yet**, for a live caption.
///
/// Only the capture layer holds open audio, so only the capture layer can
/// produce this. The contract, in full:
///
/// * `t_start_ms` is the start of the whole open stretch on the meeting clock,
///   and stays the same for as long as that stretch stays open. It is the live
///   line's identity: every look at the same speech replaces the same line, and
///   the final [`Utterance`] — which carries that same start — replaces it for
///   good. After a forced cut the continuation is a new stretch with a new
///   start, so it gets a new line.
/// * `samples` is the *tail* of that stretch, 16 kHz mono, at most
///   [`SNAPSHOT_TAIL_MS`] long. The consumer may keep less.
/// * `window_start_ms` says where `samples` begins, so a caption's timestamps
///   are the meeting's and not the window's.
/// * These are produced whenever it is cheap to. Deciding which are worth
///   decoding, and throwing the rest away, is the consumer's job.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenSpeech {
    pub channel: Channel,
    /// Start of the open stretch, stable while it stays open.
    pub t_start_ms: i64,
    /// Start of the audio in `samples`.
    pub window_start_ms: i64,
    /// 16 kHz mono tail of the open stretch.
    pub samples: Vec<f32>,
}

impl OpenSpeech {
    pub fn duration_ms(&self) -> i64 {
        samples_to_ms(self.samples.len())
    }

    /// End of the audio in `samples`, on the meeting clock.
    pub fn window_end_ms(&self) -> i64 {
        self.window_start_ms + self.duration_ms()
    }
}

fn samples_to_ms(samples: usize) -> i64 {
    samples as i64 * 1_000 / i64::from(TARGET_SAMPLE_RATE)
}

/// Only the tests need this now: every length the segmenter works in is a whole
/// number of windows, and [`whole_windows_down`] is what produces those.
#[cfg(test)]
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
    /// One entry per whole window of `samples`: was that window speech?
    ///
    /// Kept so a forced cut can look for a real pause instead of guessing from
    /// loudness — the quietest 32 ms of a vowel is still a vowel.
    voiced: Vec<bool>,
    /// Length up to and including the last window that was speech.
    voiced_len: usize,
}

impl OpenUtterance {
    /// Windows of silence at the very end, and where the speech before them
    /// stopped.
    fn trailing_silence(&self) -> usize {
        self.voiced.iter().rev().take_while(|v| !**v).count()
    }

    fn recompute_voiced_len(&mut self) {
        let voiced_windows = self.voiced.len() - self.trailing_silence();
        self.voiced_len = voiced_windows * WINDOW_SAMPLES;
    }
}

/// Turns a per-window speech decision into padded utterances.
///
/// Independent of how the decision was made, so it can be tested with generated
/// audio and no model on disk.
#[derive(Debug)]
pub struct Segmenter {
    channel: Channel,
    settings: VadSettings,
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
    /// Consecutive voiced windows while no utterance is open: the debounce that
    /// stops a blip from becoming a padded job.
    voiced_run: usize,
    in_speech: bool,
}

impl Default for Segmenter {
    fn default() -> Self {
        Self::new(Channel::default())
    }
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
    /// With the settings this channel deserves ([`VadSettings::for_channel`]).
    pub fn new(channel: Channel) -> Self {
        Self::with_settings(channel, VadSettings::for_channel(channel))
    }

    pub fn with_settings(channel: Channel, settings: VadSettings) -> Self {
        Self {
            channel,
            settings,
            pos: 0,
            origin_ms: 0,
            started: false,
            pre: VecDeque::with_capacity(settings.pre_pad_samples() + WINDOW_SAMPLES * 4),
            open: None,
            silence_windows: 0,
            voiced_run: 0,
            in_speech: false,
        }
    }

    pub fn settings(&self) -> &VadSettings {
        &self.settings
    }

    /// The most look-behind worth keeping: the pad, plus the run that has to
    /// clear the debounce before any of it counts as speech.
    fn max_pre_samples(&self) -> usize {
        self.settings.pre_pad_samples() + self.settings.min_voiced_windows() * WINDOW_SAMPLES
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

    /// The tail of the utterance that is open right now, for a live caption.
    ///
    /// `None` when nothing is open, or when what is open is not speech yet — a
    /// snapshot of the look-behind pad would be a caption of silence.
    ///
    /// This is a *copy*: the segmenter keeps its own buffer and goes on
    /// extending it, so the same open stretch can be snapshotted again and
    /// again. At most [`SNAPSHOT_TAIL_MS`], cut on a window boundary so the
    /// offsets stay exact.
    pub fn snapshot(&self) -> Option<OpenSpeech> {
        let open = self.open.as_ref()?;
        // Padding with nothing voiced behind it yet is not worth a decode.
        if open.voiced_len == 0 {
            return None;
        }
        let keep = (SNAPSHOT_TAIL_MS as usize * TARGET_SAMPLE_RATE as usize / 1_000)
            / WINDOW_SAMPLES
            * WINDOW_SAMPLES;
        let dropped = open
            .samples
            .len()
            .saturating_sub(keep)
            .div_ceil(WINDOW_SAMPLES)
            * WINDOW_SAMPLES;
        let dropped = dropped.min(open.samples.len());
        Some(OpenSpeech {
            channel: self.channel,
            t_start_ms: self.pos_to_ms(open.start_pos),
            window_start_ms: self.pos_to_ms(open.start_pos + dropped as u64),
            samples: open.samples[dropped..].to_vec(),
        })
    }

    /// Has the run of silence since the last voiced window reached
    /// [`VadSettings::silence_tail_ms`]?
    fn tail_run_out(&self) -> bool {
        self.silence_windows as i64 * WINDOW_MS >= self.settings.silence_tail_ms
    }

    /// Anchor the timeline. Called before the first window, and again after a
    /// gap so a discontinuity does not slide every later utterance.
    ///
    /// Everything the segmenter was holding goes with it: audio from before a
    /// gap is not adjacent to audio after it, so a half-open utterance, the
    /// look-behind buffer and both run counters all start again (review
    /// finding 10). The detector resets its own recurrent state alongside this.
    pub fn rebase(&mut self, t_ms: i64) -> Option<Utterance> {
        let finished = self.finish();
        self.origin_ms = t_ms;
        self.pos = 0;
        self.started = true;
        self.pre.clear();
        self.silence_windows = 0;
        self.voiced_run = 0;
        self.in_speech = false;
        self.open = None;
        finished
    }

    fn pos_to_ms(&self, pos: u64) -> i64 {
        self.origin_ms + samples_to_ms(pos as usize)
    }

    /// Feed exactly one window plus the decision made about it.
    pub fn push_window(&mut self, window: &[f32], is_speech: bool) -> Vec<Utterance> {
        debug_assert_eq!(
            window.len(),
            WINDOW_SAMPLES,
            "the segmenter works in whole windows"
        );
        if !self.started {
            self.started = true;
        }
        let mut out = Vec::new();
        let window_pos = self.pos;
        self.pos += window.len() as u64;

        if !self.in_speech {
            // Everything before speech is confirmed goes into the look-behind
            // buffer, including the run that is being counted: if it clears the
            // debounce, those windows are the start of the utterance.
            self.pre.extend(window.iter().copied());
            let max_pre = self.max_pre_samples();
            while self.pre.len() > max_pre {
                for _ in 0..WINDOW_SAMPLES {
                    self.pre.pop_front();
                }
            }
            self.voiced_run = if is_speech { self.voiced_run + 1 } else { 0 };
            if self.voiced_run >= self.settings.min_voiced_windows() {
                let samples: Vec<f32> = self.pre.drain(..).collect();
                let start_pos = (window_pos + window.len() as u64) - samples.len() as u64;
                let windows = samples.len() / WINDOW_SAMPLES;
                let mut voiced = vec![false; windows];
                for slot in voiced.iter_mut().rev().take(self.voiced_run) {
                    *slot = true;
                }
                let voiced_len = samples.len();
                self.open = Some(OpenUtterance {
                    start_pos,
                    samples,
                    voiced,
                    voiced_len,
                });
                self.in_speech = true;
                self.silence_windows = 0;
                self.voiced_run = 0;
            }
            return out;
        }

        // Already inside an utterance.
        {
            let open = self
                .open
                .as_mut()
                .expect("in_speech implies an open utterance");
            open.samples.extend_from_slice(window);
            open.voiced.push(is_speech);
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
            .is_some_and(|o| o.samples.len() >= self.settings.max_samples());
        // Speech is over, but the window may still be too short for the engine
        // to read. Hold it open and keep the silence that is arriving anyway
        // rather than emitting a sliver nothing can transcribe.
        let ready = self
            .open
            .as_ref()
            .is_some_and(|o| o.samples.len() >= MIN_UTTERANCE_SAMPLES);

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

    /// Keep the last pad of some audio as look-behind for whatever comes next.
    fn keep_as_look_behind(&mut self, leftover: &[f32]) {
        self.pre.clear();
        let pad = self.settings.pre_pad_samples();
        let from = (leftover.len().saturating_sub(pad)) / WINDOW_SAMPLES * WINDOW_SAMPLES;
        for s in &leftover[from..] {
            self.pre.push_back(*s);
        }
    }

    /// Close the open utterance at its natural end: the last voiced window plus
    /// the trailing pad — or more of that trailing silence, when trimming to the
    /// pad would leave a window the engine cannot read ([`MIN_UTTERANCE_MS`]).
    fn close(&mut self, truncated: bool) -> Option<Utterance> {
        let open = self.open.take()?;
        self.in_speech = false;
        self.silence_windows = 0;
        self.voiced_run = 0;

        // Normally: the speech plus a trailing pad. But never trim a window back
        // below what the engine can read when the audio to fill it is already in
        // hand — that is the whole reason short answers used to come out as
        // errors instead of text.
        let keep = (open.voiced_len + self.settings.post_pad_samples())
            .max(MIN_UTTERANCE_SAMPLES)
            .min(open.samples.len());
        let mut samples = open.samples;
        // Whatever is trimmed off is silence that may lead into the next
        // utterance; keep the last pad of it as look-behind.
        let leftover: Vec<f32> = samples[keep..].to_vec();
        samples.truncate(keep);
        self.keep_as_look_behind(&leftover);

        if samples.is_empty() {
            return None;
        }
        let t_start_ms = self.pos_to_ms(open.start_pos);
        let voiced_ms = voiced_ms_of(&open.voiced, samples.len());
        Some(Utterance {
            channel: self.channel,
            t_start_ms,
            t_end_ms: self.pos_to_ms(open.start_pos + samples.len() as u64),
            samples,
            truncated,
            voiced_ms,
        })
    }

    /// The monologue case: someone has been talking for
    /// [`VadSettings::max_utterance_ms`] without a pause the segmenter would
    /// close on, and the engine will not take a longer window.
    ///
    /// Two ways to cut, and which one happened is what `truncated` says:
    ///
    /// * **At a real pause.** If the last few seconds contain a stretch of
    ///   silence at least [`VadSettings::min_forced_silence_ms`] long, the cut
    ///   lands in the middle of it: the piece emitted keeps a tail, the
    ///   continuation keeps a pre-roll, and nothing is split. Looking for the
    ///   *quietest window* instead — what this used to do — is not the same
    ///   thing at all: the quietest 32 ms of a monologue may still be a vowel.
    /// * **Through speech.** Failing that, the cut goes where the cap is, and
    ///   the continuation starts [`VadSettings::forced_overlap_ms`] earlier — it
    ///   re-reads the end of what was just emitted. A word that straddles the
    ///   join is then whole in the second piece, and the engine has real
    ///   acoustic context to start from instead of a cold start mid-syllable.
    fn force_cut(&mut self) -> Option<Utterance> {
        let (cut, at_pause) = self.find_forced_cut()?;
        let overlap = self.settings.overlap_samples();
        let open = self.open.as_mut()?;

        // A pause needs no overlap: nothing was split. A cut through speech
        // carries audio over, so the join can be read across.
        let keep_from = if at_pause {
            cut
        } else {
            cut.saturating_sub(overlap)
        };
        let start_pos = open.start_pos;
        let emitted: Vec<f32> = open.samples[..cut].to_vec();
        // Read before the open utterance is rewritten below: this is the piece
        // going out, not the piece staying behind.
        let emitted_voiced_ms = voiced_ms_of(&open.voiced, cut);
        let rest: Vec<f32> = open.samples[keep_from..].to_vec();
        let rest_voiced: Vec<bool> = open.voiced[keep_from / WINDOW_SAMPLES..].to_vec();

        if rest_voiced.iter().any(|v| *v) {
            open.start_pos = start_pos + keep_from as u64;
            open.samples = rest;
            open.voiced = rest_voiced;
            open.recompute_voiced_len();
            // The silence the cut landed in belongs to both pieces; the
            // continuation inherits however much of it it actually holds, so a
            // speaker who really has stopped closes normally instead of being
            // held open by a counter that was reset for tidiness.
            self.silence_windows = open.trailing_silence();
        } else {
            // Nothing but silence left over: there is no continuation, only
            // look-behind for whoever speaks next.
            self.open = None;
            self.in_speech = false;
            self.silence_windows = 0;
            self.voiced_run = 0;
            self.keep_as_look_behind(&rest);
        }

        if emitted.is_empty() {
            return None;
        }
        let t_start_ms = self.pos_to_ms(start_pos);
        Some(Utterance {
            channel: self.channel,
            t_start_ms,
            t_end_ms: self.pos_to_ms(start_pos + emitted.len() as u64),
            samples: emitted,
            truncated: !at_pause,
            voiced_ms: emitted_voiced_ms,
        })
    }

    /// Where to cut the open utterance, and whether that spot is a real pause.
    ///
    /// Prefers the longest qualifying pause in the search region, latest first
    /// on a tie: the longer the silence, the more likely it is the end of a
    /// sentence rather than a breath between two words.
    fn find_forced_cut(&self) -> Option<(usize, bool)> {
        let open = self.open.as_ref()?;
        let len = open.samples.len();
        let windows = open.voiced.len();
        let search_from = windows.saturating_sub((FORCED_CUT_SEARCH_MS / WINDOW_MS) as usize);
        let need = self.settings.min_forced_silence_windows();

        let mut best: Option<(usize, usize)> = None; // (run length, run start)
        let mut run_start = None;
        for i in search_from..=windows {
            let silent = i < windows && !open.voiced[i];
            match (silent, run_start) {
                (true, None) => run_start = Some(i),
                (false, Some(start)) => {
                    let run = i - start;
                    if run >= need && best.is_none_or(|(best_run, _)| run >= best_run) {
                        best = Some((run, start));
                    }
                    run_start = None;
                }
                _ => {}
            }
        }

        match best {
            // Halfway through the pause: both sides of the join keep some of it.
            // Never zero, so a cut always makes progress.
            Some((run, start)) => Some(((start + run / 2).max(1) * WINDOW_SAMPLES, true)),
            None => Some((len, false)),
        }
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
    /// Forget whether speech was happening. The learned noise floor stays: it
    /// describes the room, and the room is still the room.
    fn reset(&mut self) {
        self.active = false;
    }

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

    /// Zero the recurrent state. What the model remembers about the audio before
    /// a gap is worse than nothing after it.
    fn reset(&mut self) {
        self.state.fill(0.0);
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
    settings: VadSettings,
    /// Audio that did not fill a whole window yet.
    leftover: Vec<f32>,
    /// Where `leftover` starts on the meeting clock.
    next_t_ms: i64,
    anchored: bool,
    /// Hysteresis state, per window: once over [`VadSettings::enter`] it takes
    /// a drop below [`VadSettings::exit`] to go back down.
    above: bool,
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
        Ok(Self::with_engine(engine, channel))
    }

    /// Detector for when the asset is not on disk. Recording never waits for a
    /// download.
    pub fn without_model(channel: Channel) -> Self {
        Self::with_engine(Engine::Loudness(LoudnessGate::default()), channel)
    }

    fn with_engine(engine: Engine, channel: Channel) -> Self {
        let settings = VadSettings::for_channel(channel);
        Self {
            engine,
            segmenter: Segmenter::with_settings(channel, settings),
            settings,
            leftover: Vec::with_capacity(WINDOW_SAMPLES * 2),
            next_t_ms: 0,
            anchored: false,
            above: false,
            speaking: false,
            engine_failures: 0,
        }
    }

    /// The thresholds and paddings this detector is using. Diagnostics and
    /// tuning only.
    pub fn settings(&self) -> &VadSettings {
        &self.settings
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

    /// The tail of the utterance open right now, for a live caption. See
    /// [`Segmenter::snapshot`].
    ///
    /// Only whole windows are ever handed to the segmenter, so the `leftover`
    /// this is not carrying is under 32 ms.
    pub fn snapshot(&self) -> Option<OpenSpeech> {
        self.segmenter.snapshot()
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
                let gap_ms = t_start_ms - expected;
                tracing::debug!(
                    target: "echo::audio",
                    channel = self.segmenter.channel.as_str(),
                    gap_ms,
                    "the audio stream jumped; speech detection starts again from here"
                );
                out.extend(self.reset_at(t_start_ms));
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

    /// Re-anchor the timeline and forget everything recurrent.
    ///
    /// Silero carries state between windows, and so does the loudness gate: both
    /// describe audio that was adjacent to what came next. After a gap, a device
    /// switch or a pause it is not adjacent to anything, and carrying it over is
    /// how a resumed recording starts by mis-hearing (review finding 10). The
    /// segmenter is rebased in the same breath, so both halves of the detector
    /// start again together.
    ///
    /// Returns whatever utterance was still open, so no speech is lost to the
    /// reset.
    pub fn reset_at(&mut self, t_start_ms: i64) -> Option<Utterance> {
        let finished = self.segmenter.rebase(t_start_ms);
        self.leftover.clear();
        self.next_t_ms = t_start_ms;
        self.anchored = true;
        self.above = false;
        self.speaking = false;
        match &mut self.engine {
            Engine::Silero(silero) => silero.reset(),
            Engine::Loudness(gate) => gate.reset(),
        }
        finished
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
                    self.above = rms(window) > 0.01;
                    return self.above;
                }
            },
        };
        // Hysteresis per *window*, not per hand-over: a batch of frames arriving
        // together must be judged the same way as the same audio arriving one
        // frame at a time.
        self.above = if self.above {
            probability > self.settings.exit
        } else {
            probability > self.settings.enter
        };
        self.above
    }

    /// End of stream: emit whatever is still open.
    pub fn finish(&mut self) -> Option<Utterance> {
        // Whatever did not fill a window is padding at worst; the audio on disk
        // still has it.
        self.leftover.clear();
        let done = self.segmenter.finish();
        self.speaking = false;
        self.above = false;
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

// ---------------------------------------------------------------------------
// Detection over audio read back from disk
// ---------------------------------------------------------------------------

enum DetectRequest {
    Push {
        samples: Vec<f32>,
        t_start_ms: i64,
        reply: tokio::sync::oneshot::Sender<Vec<Utterance>>,
    },
    Finish {
        reply: tokio::sync::oneshot::Sender<Vec<Utterance>>,
    },
}

/// One detector, streamed through as much audio as you like, a piece at a time.
///
/// This is what the catch-up pass needs and what it used to lack. Building a
/// fresh detector per window of audio read from disk (review finding 2) means
/// every window boundary is a hard reset: Silero's recurrent state starts from
/// zero, there is no look-behind, and a word straddling the boundary is split in
/// two — an arbitrary line drawn every 24 seconds through a two-hour meeting.
/// Streamed instead, a boundary is nothing at all: the detector never learns
/// that the audio arrived in pieces.
///
/// The detector lives on its own thread for as long as the stretch does. That is
/// not for parallelism — inference here is one window at a time — but because it
/// is the only way state can persist across `await` points without asking
/// anything about the internals of the model runtime.
pub struct OfflineDetector {
    requests: Option<std::sync::mpsc::Sender<DetectRequest>>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl std::fmt::Debug for OfflineDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OfflineDetector")
            .field("running", &self.requests.is_some())
            .finish()
    }
}

impl OfflineDetector {
    /// Start one. `model_path` of `None` — or a path that will not load — falls
    /// back to the loudness gate, exactly as live detection does.
    pub fn open(model_path: Option<PathBuf>, channel: Channel) -> Result<Self, AudioError> {
        let (requests, incoming) = std::sync::mpsc::channel::<DetectRequest>();
        let worker = std::thread::Builder::new()
            .name("echo-speech-offline".into())
            .spawn(move || {
                let mut detector = SpeechDetector::load_or_fallback(model_path.as_deref(), channel);
                while let Ok(request) = incoming.recv() {
                    match request {
                        DetectRequest::Push {
                            samples,
                            t_start_ms,
                            reply,
                        } => {
                            if reply.send(detector.push(&samples, t_start_ms)).is_err() {
                                break;
                            }
                        }
                        DetectRequest::Finish { reply } => {
                            let _ = reply.send(detector.finish().into_iter().collect());
                            break;
                        }
                    }
                }
            })
            .map_err(|e| AudioError::Backend(e.to_string()))?;
        Ok(Self {
            requests: Some(requests),
            worker: Some(worker),
        })
    }

    /// Feed the next piece. Pieces are expected to be contiguous; a jump in
    /// `t_start_ms` is treated as a discontinuity, same as live.
    pub async fn push(
        &mut self,
        samples: Vec<f32>,
        t_start_ms: i64,
    ) -> Result<Vec<Utterance>, AudioError> {
        let (reply, answer) = tokio::sync::oneshot::channel();
        self.ask(
            DetectRequest::Push {
                samples,
                t_start_ms,
                reply,
            },
            answer,
        )
        .await
    }

    /// No more audio: hand back whatever was still open. Idempotent.
    pub async fn finish(&mut self) -> Result<Vec<Utterance>, AudioError> {
        let (reply, answer) = tokio::sync::oneshot::channel();
        let last = self.ask(DetectRequest::Finish { reply }, answer).await;
        self.stop();
        last
    }

    async fn ask(
        &mut self,
        request: DetectRequest,
        answer: tokio::sync::oneshot::Receiver<Vec<Utterance>>,
    ) -> Result<Vec<Utterance>, AudioError> {
        let Some(requests) = self.requests.as_ref() else {
            return Ok(Vec::new());
        };
        let gone = || AudioError::Backend("speech detection stopped unexpectedly".to_string());
        requests.send(request).map_err(|_| gone())?;
        answer.await.map_err(|_| gone())
    }

    /// Release the model and the thread (mantra 1).
    fn stop(&mut self) {
        self.requests = None;
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl Drop for OfflineDetector {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Run detection over one buffer of audio already on disk, start to finish.
///
/// [`OfflineDetector`] is what the catch-up pass uses; this is the one-shot form
/// for a stretch that arrives all at once.
pub async fn detect_offline(
    model_path: &Path,
    samples: &[f32],
    t_offset_ms: i64,
    channel: Channel,
) -> Result<Vec<Utterance>, AudioError> {
    let mut detector = OfflineDetector::open(Some(model_path.to_path_buf()), channel)?;
    let mut out = Vec::new();
    // Half-second pieces: big enough to be cheap, small enough that a very long
    // recording does not sit in one allocation twice.
    let step = TARGET_SAMPLE_RATE as usize / 2;
    for (i, piece) in samples.chunks(step).enumerate() {
        let t = t_offset_ms + samples_to_ms(i * step);
        out.extend(detector.push(piece.to_vec(), t).await?);
    }
    out.extend(detector.finish().await?);
    Ok(out)
}

// ---------------------------------------------------------------------------
// The bounded queue between detection and the speech engine
// ---------------------------------------------------------------------------

/// How many utterances may wait for text before live work starts being dropped.
pub const DEFAULT_QUEUE_CAPACITY: usize = 24;

/// Bounded queue of work for the speech engine.
///
/// Not used by capture: an utterance leaves the capture layer through the signal
/// channel the moment it is found, and the queue that decides what live work is
/// dropped belongs to the session pipeline, which is what actually waits on the
/// speech engine. There used to be one here as well, pushed on every utterance
/// and popped by nobody (review finding 6). This is the primitive, for whoever
/// needs drop-oldest semantics; it is not a second pipeline.
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

    /// Every utterance carries how much of itself was speech, because that is
    /// the only evidence anything downstream has that a line was written over a
    /// pause (2026-08-24; see [`crate::asr::phantom`]).
    #[test]
    fn an_utterance_says_how_much_of_it_was_voice() {
        let settings = VadSettings::mic();
        let mut seg = Segmenter::with_settings(Channel::Mic, settings);
        let quiet = vec![0.0f32; WINDOW_SAMPLES];
        let mut out = Vec::new();

        // A shortest-possible burst: exactly the debounce, then silence until
        // the utterance closes.
        for _ in 0..10 {
            out.extend(seg.push_window(&quiet, false));
        }
        let voiced_windows = settings.min_voiced_windows();
        for _ in 0..voiced_windows {
            out.extend(seg.push_window(&quiet, true));
        }
        for _ in 0..60 {
            out.extend(seg.push_window(&quiet, false));
        }
        out.extend(seg.finish());
        assert_eq!(out.len(), 1, "{out:#?}");
        let blip = &out[0];
        assert_eq!(blip.voiced_ms, voiced_windows as i64 * WINDOW_MS);
        assert!(
            blip.measured_voice_ms() < crate::asr::phantom::TOO_LITTLE_VOICE_MS,
            "a stretch opened by one blip holds {} ms of voice",
            blip.measured_voice_ms()
        );

        // And a stretch somebody actually spoke through does not.
        let mut seg = Segmenter::with_settings(Channel::Mic, settings);
        let mut out = Vec::new();
        for _ in 0..10 {
            out.extend(seg.push_window(&quiet, false));
        }
        for _ in 0..20 {
            out.extend(seg.push_window(&quiet, true));
        }
        for _ in 0..60 {
            out.extend(seg.push_window(&quiet, false));
        }
        out.extend(seg.finish());
        assert_eq!(out.len(), 1, "{out:#?}");
        let spoken = &out[0];
        assert_eq!(spoken.voiced_ms, 20 * WINDOW_MS);
        assert!(
            spoken.measured_voice_ms() > crate::asr::phantom::TOO_LITTLE_VOICE_MS,
            "real speech holds {} ms of voice",
            spoken.measured_voice_ms()
        );
    }

    /// Nothing measured it, so nothing may be deleted because of it.
    #[test]
    fn an_unmeasured_utterance_reads_as_all_voice() {
        let unmeasured = Utterance {
            t_start_ms: 0,
            t_end_ms: 4_000,
            ..Default::default()
        };
        assert_eq!(unmeasured.measured_voice_ms(), 4_000);
        assert_eq!(unmeasured.voiced_ratio(), 1.0);
        assert_eq!(Utterance::default().measured_voice_ms(), 0);
        assert_eq!(Utterance::default().voiced_ratio(), 1.0);
        // And a measurement longer than the stretch cannot exceed it.
        let odd = Utterance {
            t_start_ms: 0,
            t_end_ms: 1_000,
            voiced_ms: 5_000,
            ..Default::default()
        };
        assert_eq!(odd.measured_voice_ms(), 1_000);
        assert_eq!(odd.voiced_ratio(), 1.0);
    }

    #[test]
    fn the_utterance_cap_leaves_room_for_padding() {
        for settings in [VadSettings::mic(), VadSettings::system()] {
            assert!(settings.max_utterance_ms > settings.pre_pad_ms + settings.post_pad_ms);
            assert!(settings.silence_tail_ms > 0);
            // An utterance plus its padding has to fit the speech engine's
            // window, or it would have to be split again later.
            assert!(
                settings.max_utterance_ms + settings.pre_pad_ms + settings.post_pad_ms <= 30_000
            );
        }
    }

    #[test]
    fn each_channel_gets_the_settings_its_audio_deserves() {
        let mic = VadSettings::mic();
        let system = VadSettings::system();
        assert_eq!(VadSettings::for_channel(Channel::Mic), mic);
        assert_eq!(VadSettings::for_channel(Channel::System), system);
        assert_eq!(VadSettings::default(), mic);

        // The microphone is the far-field, noisy, level-variable one: it listens
        // harder, pads more and closes later. What the computer plays is clean,
        // so it is stricter and quicker.
        assert!(
            mic.enter < system.enter,
            "the microphone must be the eager one"
        );
        assert!(mic.exit < system.exit);
        assert!(mic.silence_tail_ms > system.silence_tail_ms);
        assert!(mic.pre_pad_ms > system.pre_pad_ms);
        assert_eq!(mic.max_utterance_ms, system.max_utterance_ms);

        for settings in [mic, system] {
            // Hysteresis, not a single threshold: a detector hovering around the
            // bar must not shred a sentence.
            assert!(settings.exit < settings.enter, "{settings:#?}");
            assert!(settings.enter < 1.0 && settings.exit > 0.0);
            // A debounce short enough for "sì", long enough that one window of
            // noise is not an utterance.
            assert!((WINDOW_MS..250).contains(&settings.min_voiced_ms));
            assert!(settings.min_voiced_windows() >= 2);
            // Every derived length is a whole number of windows, so utterances
            // tile the timeline with no rounding gap.
            for samples in [
                settings.pre_pad_samples(),
                settings.post_pad_samples(),
                settings.max_samples(),
                settings.overlap_samples(),
            ] {
                assert_eq!(samples % WINDOW_SAMPLES, 0, "{settings:#?}");
                assert!(samples > 0);
            }
            assert!(settings.overlap_samples() < settings.max_samples());
        }
    }

    #[test]
    fn a_detector_carries_its_channels_settings() {
        assert_eq!(
            SpeechDetector::without_model(Channel::System).settings(),
            &VadSettings::system()
        );
        assert_eq!(
            SpeechDetector::without_model(Channel::Mic).settings(),
            &VadSettings::mic()
        );
    }

    #[test]
    fn a_thirty_two_millisecond_blip_is_not_an_utterance() {
        // The whole point of the debounce: one window over the threshold — a
        // key, a chair, a click on the line — used to become a padded,
        // second-long job for the speech engine, and a line of invented
        // transcript with it.
        let mut d = SpeechDetector::without_model(Channel::Mic);
        let mut audio = room_noise(600);
        audio.extend(tone(WINDOW_MS as usize, 300.0, 0.3));
        audio.extend(room_noise(2_000));

        let mut out = feed(&mut d, &audio, 0);
        out.extend(d.finish());
        assert!(out.is_empty(), "a 32 ms blip became an utterance: {out:#?}");
    }

    #[test]
    fn speech_that_clears_the_debounce_is_kept_whole_from_before_it_started() {
        let settings = VadSettings::mic();
        let mut d = SpeechDetector::without_model(Channel::Mic);
        let mut audio = room_noise(1_000);
        audio.extend(tone(
            settings.min_voiced_ms as usize + WINDOW_MS as usize,
            300.0,
            0.3,
        ));
        audio.extend(room_noise(2_000));

        let mut out = feed(&mut d, &audio, 0);
        out.extend(d.finish());
        assert_eq!(out.len(), 1, "{out:#?}");
        // The windows spent proving it was speech are *inside* the utterance,
        // not thrown away: the debounce delays the decision, never the audio.
        assert!(
            out[0].t_start_ms <= 1_000 - samples_to_ms(settings.pre_pad_samples()) + WINDOW_MS,
            "the onset was clipped: {:#?}",
            out[0]
        );
    }

    #[test]
    fn the_computer_channel_closes_sooner_than_the_microphone() {
        // The same audio, a pause between the two halves that is longer than the
        // system tail and shorter than the microphone's.
        let mut audio = room_noise(300);
        audio.extend(tone(1_400, 300.0, 0.3));
        audio.extend(room_noise(360));
        audio.extend(tone(1_400, 300.0, 0.3));
        audio.extend(room_noise(1_200));

        let mut mic = SpeechDetector::without_model(Channel::Mic);
        let mut heard = feed(&mut mic, &audio, 0);
        heard.extend(mic.finish());
        assert_eq!(heard.len(), 1, "the microphone split a phrase: {heard:#?}");

        let mut system = SpeechDetector::without_model(Channel::System);
        let mut played = feed(&mut system, &audio, 0);
        played.extend(system.finish());
        assert_eq!(
            played.len(),
            2,
            "the computer's audio held a finished sentence open: {played:#?}"
        );
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
        // Shorter than the microphone's silence tail.
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
    /// can read. The old segmenter could emit as little as one pad either side
    /// of a single voiced window — 608 ms — which whisper.cpp refuses to encode,
    /// and every short answer in the meeting landed there.
    #[test]
    fn a_short_answer_is_still_long_enough_for_the_engine_to_read() {
        for burst_ms in [100usize, 200, 300, 400, 700] {
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
        // The natural floor is the debounce plus a pad either side, which on
        // either channel is well under the second whisper.cpp needs.
        for settings in [VadSettings::mic(), VadSettings::system()] {
            let floor_ms = samples_to_ms(
                settings.pre_pad_samples()
                    + settings.post_pad_samples()
                    + settings.min_voiced_windows() * WINDOW_SAMPLES,
            );
            assert!(
                floor_ms < 1_000,
                "this test exists because the natural floor ({floor_ms} ms) is under a second"
            );
        }
        const {
            assert!(
                MIN_UTTERANCE_MS > 1_000,
                "the floor has to clear whisper.cpp's one-second window, not just touch it"
            )
        };
        // And the floor is kept in whole windows, so holding an utterance open
        // for it cannot leave a fraction of a window behind.
        assert_eq!(MIN_UTTERANCE_SAMPLES % WINDOW_SAMPLES, 0);
        assert!(MIN_UTTERANCE_SAMPLES >= ms_to_samples(MIN_UTTERANCE_MS));
    }

    // -- snapshots of speech that is still going ---------------------------

    #[test]
    fn nothing_open_means_nothing_to_caption() {
        let mut d = SpeechDetector::without_model(Channel::Mic);
        assert!(d.snapshot().is_none(), "no audio at all");
        feed(&mut d, &room_noise(1_000), 0);
        assert!(d.snapshot().is_none(), "room noise is not speech");
    }

    #[test]
    fn speech_can_be_looked_at_before_it_finishes() {
        let mut d = SpeechDetector::without_model(Channel::Mic);
        let mut audio = room_noise(500);
        audio.extend(tone(4_000, 300.0, 0.3));
        // Nothing here closes the utterance: no trailing silence is fed, so the
        // stretch is still open when the snapshot is taken. This is the whole
        // point — text on screen while someone is still talking.
        let out = feed(&mut d, &audio, 0);
        assert!(
            out.is_empty(),
            "the utterance should still be open: {out:#?}"
        );

        let snap = d.snapshot().expect("speech is open");
        assert_eq!(snap.channel, Channel::Mic);
        assert!(rms(&snap.samples) > 0.0, "the caption got silence");
        // The window is inside the speech, and its end is where the audio has
        // actually reached rather than wherever the stretch started.
        assert!(snap.window_start_ms >= snap.t_start_ms);
        assert!(snap.window_end_ms() > snap.window_start_ms);
        assert!(
            snap.window_end_ms() <= 4_500,
            "a caption cannot cover audio that has not arrived: {}",
            snap.window_end_ms()
        );
        assert!(
            snap.duration_ms() >= 3_000,
            "four seconds of speech gave a {} ms caption",
            snap.duration_ms()
        );

        // The identity of the line is stable while the stretch stays open, and
        // the final utterance carries that same start so it replaces the line.
        let before = snap.t_start_ms;
        feed(&mut d, &tone(2_000, 300.0, 0.3), 4_500);
        let later = d.snapshot().expect("still open");
        assert_eq!(
            later.t_start_ms, before,
            "the line changed identity mid-word"
        );
        assert!(
            later.window_end_ms() > snap.window_end_ms(),
            "it did not advance"
        );

        let finished = d.finish().expect("the stretch closes");
        assert_eq!(
            finished.t_start_ms, before,
            "the final has to land on the line the captions were drawn on"
        );
    }

    #[test]
    fn a_caption_never_carries_more_than_the_tail() {
        let mut d = SpeechDetector::without_model(Channel::Mic);
        let mut audio = room_noise(500);
        // Well past the cap, but still under MAX_UTTERANCE_MS so the segmenter
        // has not forced a cut and the stretch is genuinely this long.
        audio.extend(tone(20_000, 300.0, 0.3));
        feed(&mut d, &audio, 0);

        let snap = d.snapshot().expect("speech is open");
        assert!(
            snap.duration_ms() <= SNAPSHOT_TAIL_MS,
            "a caption carried {} ms, cap is {SNAPSHOT_TAIL_MS}",
            snap.duration_ms()
        );
        // Trimming the front moves the window, never the line's identity.
        assert!(snap.window_start_ms > snap.t_start_ms);
        // Whole windows only, so the offsets stay exact and captions tile with
        // the finals rather than drifting by a fraction of a window.
        assert_eq!(snap.samples.len() % WINDOW_SAMPLES, 0);
        assert_eq!((snap.window_start_ms - snap.t_start_ms) % WINDOW_MS, 0);
    }

    #[test]
    fn the_tail_a_caption_gets_covers_the_window_it_wants() {
        // The capture layer offers a little more than the pipeline's caption
        // window so the consumer, not the producer, decides how much to decode.
        const {
            assert!(
                SNAPSHOT_TAIL_MS >= crate::session::pipeline::CAPTION_WINDOW_MS,
                "captions would be short because the snapshot is the thing capping them"
            )
        };
        // And a tail that long still has to fit an utterance, or it would be
        // capping nothing.
        const { assert!(SNAPSHOT_TAIL_MS < MAX_UTTERANCE_MS) };
        assert_eq!(
            (SNAPSHOT_TAIL_MS * TARGET_SAMPLE_RATE as i64 / 1_000) % WINDOW_SAMPLES as i64,
            0,
            "keep the cap a whole number of windows"
        );
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
    fn a_monologue_with_no_pause_is_cut_with_audio_carried_across_the_join() {
        let settings = VadSettings::mic();
        let mut d = SpeechDetector::without_model(Channel::Mic);
        let mut audio = room_noise(200);
        // Not one breath in seventy seconds: there is nowhere good to cut, so
        // every cut has to go through speech.
        audio.extend(tone(70_000, 300.0, 0.3));
        audio.extend(room_noise(1_000));

        let mut out = feed(&mut d, &audio, 0);
        out.extend(d.finish());

        assert!(out.len() >= 3, "70 s of speech became {} pieces", out.len());
        for u in &out {
            assert!(
                u.duration_ms() <= settings.max_utterance_ms + settings.post_pad_ms,
                "a piece is {} ms long",
                u.duration_ms()
            );
            assert_eq!(u.samples.len(), ms_to_samples(u.duration_ms()));
        }
        // Every cut but the last went through speech, and says so — that flag is
        // what tells the speech engine the text stops mid-thought and the next
        // piece starts by repeating audio it has already read.
        let (last, forced) = out.split_last().unwrap();
        assert!(forced.iter().all(|u| u.truncated), "{out:#?}");
        assert!(!last.truncated);

        let overlap_ms = samples_to_ms(settings.overlap_samples());
        assert!(
            overlap_ms >= 700,
            "{overlap_ms} ms is not enough to read across"
        );
        for pair in out.windows(2) {
            assert_eq!(
                pair[1].t_start_ms,
                pair[0].t_end_ms - overlap_ms,
                "the join carried no audio: {:#?} then {:#?}",
                pair[0],
                pair[1]
            );
            // And the carried audio really is the same audio, not silence.
            let carried = ms_to_samples(overlap_ms);
            let tail = &pair[0].samples[pair[0].samples.len() - carried..];
            assert_eq!(
                &pair[1].samples[..carried],
                tail,
                "the overlap is not the same audio"
            );
        }
    }

    #[test]
    fn a_monologue_with_a_real_pause_is_cut_there_and_nothing_is_repeated() {
        let settings = VadSettings::mic();
        let mut d = SpeechDetector::without_model(Channel::Mic);
        let mut audio = room_noise(200);
        // Talking right up to the cap, with one real breath just before it.
        audio.extend(tone(settings.max_utterance_ms as usize - 1_000, 300.0, 0.3));
        audio.extend(room_noise(200));
        audio.extend(tone(3_000, 300.0, 0.3));
        audio.extend(room_noise(1_000));

        let mut out = feed(&mut d, &audio, 0);
        out.extend(d.finish());

        assert_eq!(out.len(), 2, "{out:#?}");
        assert!(
            !out[0].truncated,
            "a cut that landed in a real pause split nothing and needs no overlap"
        );
        assert_eq!(
            out[0].t_end_ms, out[1].t_start_ms,
            "the pieces overlapped even though there was a pause to cut at"
        );
        // The cut landed inside the pause, so both pieces keep some of it.
        let quiet_tail = rms(&out[0].samples[out[0].samples.len() - WINDOW_SAMPLES..]);
        assert!(
            quiet_tail < 0.05,
            "the first piece ends mid-word: {quiet_tail}"
        );
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
    fn resuming_hands_back_what_was_open_and_starts_again_from_there() {
        let mut d = SpeechDetector::without_model(Channel::Mic);
        // Someone is mid-sentence when the recording is paused.
        let mut audio = room_noise(300);
        audio.extend(tone(2_000, 300.0, 0.3));
        let out = feed(&mut d, &audio, 0);
        assert!(out.is_empty(), "the sentence had not finished: {out:#?}");
        assert!(d.is_speaking());

        // Resume: whatever was open comes back rather than being lost, and
        // nothing about the audio before the pause is carried forward.
        let held = d
            .reset_at(90_000)
            .expect("the open sentence must come back");
        assert!(held.t_end_ms <= 2_400, "{held:#?}");
        assert!(!d.is_speaking());

        let mut later = room_noise(300);
        later.extend(tone(1_500, 300.0, 0.3));
        later.extend(room_noise(1_200));
        let mut after = feed(&mut d, &later, 90_000);
        after.extend(d.finish());
        assert_eq!(after.len(), 1, "{after:#?}");
        assert!(
            after[0].t_start_ms >= 89_900,
            "audio from before the pause was glued to audio after it: {:#?}",
            after[0]
        );
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

    /// The catch-up pass reads a meeting back in pieces. A sentence that happens
    /// to straddle two of those pieces has to come out as one utterance, because
    /// where the reads fall is an accident of buffer sizes and has nothing to do
    /// with when anyone stopped talking.
    #[test]
    fn a_sentence_straddling_two_reads_is_still_one_utterance() {
        let mut audio = room_noise(500);
        audio.extend(tone(3_000, 300.0, 0.3));
        audio.extend(room_noise(1_500));

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let found = rt.block_on(async {
            let mut detector = OfflineDetector::open(None, Channel::Mic).unwrap();
            let mut out = Vec::new();
            // Two reads, with the boundary in the middle of the sentence.
            let split = RATE * 2;
            out.extend(detector.push(audio[..split].to_vec(), 0).await.unwrap());
            out.extend(
                detector
                    .push(audio[split..].to_vec(), samples_to_ms(split))
                    .await
                    .unwrap(),
            );
            out.extend(detector.finish().await.unwrap());
            // Finishing twice is not an error, it just has nothing to add.
            assert!(detector.finish().await.unwrap().is_empty());
            out
        });

        assert_eq!(
            found.len(),
            1,
            "the read boundary split a sentence: {found:#?}"
        );
        assert!(found[0].duration_ms() >= 3_000, "{:#?}", found[0]);
    }
}
