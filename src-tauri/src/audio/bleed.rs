//! The microphone's copy of what the computer played, and how to recognise it.
//!
//! When somebody takes a meeting on laptop speakers, the far side comes out of
//! those speakers, crosses thirty centimetres of air, and goes back in through
//! the microphone. Echo then has the same sentence twice: once clean on the
//! system channel, once quiet and late on the mic channel. The transcript shows
//! it twice, and — worse — the second copy is attributed to *you*, because a
//! meeting with any system audio at all pins every mic line to "You"
//! (`audio/mod.rs`, `diarize/pipeline.rs`).
//!
//! The measurement that started this, on the test recording of 2026-08-25:
//! **every one of the twelve system-channel segments overlapped a mic segment in
//! time.** All twelve. Matching the *text* of the two copies caught only three
//! of the eight pairs, because the two channels transcribe the same words
//! differently ("scrolling check line seven" against "Scrolling check line 7").
//! So the strong signal is timing and loudness, not words — which is what this
//! module measures.
//!
//! ## What it does, and what it deliberately does not
//!
//! It compares the **loudness over time** of the two channels — an RMS envelope
//! at 100 Hz, correlated at every plausible delay — and answers with evidence,
//! not a verdict: how well the two shapes matched, at what delay, how much of
//! the aligned stretch the far side was actually audible for, and the longest
//! run of mic loudness the far side cannot account for. [`is_bleed`] then turns
//! that evidence into a yes or no.
//!
//! There is no FFT here, and that is on purpose: `diarize/features.rs` says the
//! codebase avoids one, and it does not need one. A 24 s utterance is 2,400
//! envelope points against 101 candidate delays — about a quarter of a million
//! multiply-adds, well under a millisecond. That is the grain of this code.
//!
//! It is also not echo cancellation. Cancellation would mean leaving cpal for
//! the mic path and accepting automatic gain control, which would perturb the
//! very embeddings the speaker pass depends on. Detect-and-drop is cheaper and
//! reversible: a suppressed copy is not deleted text, it is text that is already
//! in the transcript from a cleaner signal. Suppression here is *deduplication*.
//!
//! ## Its shape is [`crate::asr::phantom`]'s
//!
//! That filter deletes a line only when a stock courtesy phrase **and** near
//! silence under it are true at once, because either alone is wrong in both
//! directions. This one is the same discipline with more conditions: a pure
//! predicate over measured evidence, every condition able only to *prevent* a
//! suppression, and a table of tests that says which condition carried each
//! case. A wrongly suppressed mic line is attributed to nobody at all, which is
//! worse than being attributed to the wrong person — so the whole file leans one
//! way.
//!
//! ## The numbers in here are starting values
//!
//! In exactly the sense [`crate::audio::vad::VadSettings`] says its numbers are:
//! "starting values from real Italian meetings, not laws". Each constant below
//! carries the argument for the value it has, and every one of those arguments
//! is a prediction. `examples/bleed_probe.rs` is what turns them into measured
//! numbers: it runs this predicate over a finished meeting from disk and prints
//! the correlation histogram split by whether a system segment overlapped that
//! utterance in time. That split is the measurement — it is what says whether
//! the predicate separates the two populations at all, and where the line
//! between them actually falls on this machine's audio.

use std::collections::VecDeque;

use crate::audio::TARGET_SAMPLE_RATE;

// ---------------------------------------------------------------------------
// The envelope
// ---------------------------------------------------------------------------

/// Width of one envelope window: 20 ms of audio, root-mean-squared.
pub const ENVELOPE_WINDOW_MS: i64 = 20;

/// How far the window moves between points: 10 ms, so the envelope is 100 Hz.
///
/// Syllables arrive at 4-8 Hz in running speech, so a 100 Hz envelope samples
/// the rhythm this correlator actually reads about five times faster than
/// Nyquist demands. The reason to go that fast anyway is the *lag*: a delay is
/// only ever resolved to a whole hop, and 10 ms is the resolution the whole
/// module is quoted at. Halving it again would double the work for a precision
/// nothing downstream can use — the alignment between the two channels is not
/// trustworthy to tens of milliseconds in the first place (each channel's drift
/// tracker inserts or drops up to 200 ms per correction, `audio/clock.rs`).
pub const ENVELOPE_HOP_MS: i64 = 10;

const WINDOW_SAMPLES: usize = (TARGET_SAMPLE_RATE as usize * ENVELOPE_WINDOW_MS as usize) / 1_000;
const HOP_SAMPLES: usize = (TARGET_SAMPLE_RATE as usize * ENVELOPE_HOP_MS as usize) / 1_000;

/// Loudness over time: one RMS value per [`ENVELOPE_HOP_MS`], **linear**.
///
/// Linear, not decibels, and that is the load-bearing choice. A log envelope
/// spreads the quiet parts out: the difference between a microphone's noise
/// floor and digital silence is enormous in decibels and nothing at all in
/// pressure, so a log correlation ends up dominated by how the two channels'
/// *pauses* happen to wobble — which is exactly the part that carries no shared
/// structure. The structure the two copies share lives in the peaks: the same
/// syllables, in the same order, at the same relative sizes. Linear RMS puts the
/// weight there.
///
/// Windows overlap (20 ms wide, 10 ms apart) so a syllable edge cannot fall
/// between two windows and vanish. Only whole windows are produced; a tail
/// shorter than one window is dropped rather than measured against less audio
/// than everything before it.
///
/// `out` is cleared and reused, so a caller in the capture loop allocates once.
pub fn envelope(samples: &[f32], out: &mut Vec<f32>) {
    out.clear();
    if samples.len() < WINDOW_SAMPLES {
        return;
    }
    let hops = (samples.len() - WINDOW_SAMPLES) / HOP_SAMPLES + 1;
    out.reserve(hops);
    for hop in 0..hops {
        let from = hop * HOP_SAMPLES;
        let window = &samples[from..from + WINDOW_SAMPLES];
        let sum: f32 = window.iter().map(|s| s * s).sum();
        out.push((sum / WINDOW_SAMPLES as f32).sqrt());
    }
}

// ---------------------------------------------------------------------------
// The delay
// ---------------------------------------------------------------------------

/// Earliest delay worth looking at: the mic copy 200 ms **ahead** of the system
/// copy.
///
/// Physically impossible — sound does not arrive before it is played — but
/// alignment is not physics here. The mic pins its first sample to t=0 with no
/// measurement of device latency; the system channel pins to a clock stamped
/// *after* the OS handed over its first buffer; and each channel's drift tracker
/// independently inserts or drops up to 200 ms per correction. A negative
/// measured lag means the bookkeeping drifted, not that time ran backwards, and
/// refusing to look there would simply lose those meetings.
pub const LAG_MIN_MS: i64 = -200;

/// Latest delay worth looking at.
///
/// The acoustic path is nothing — a metre of air is 3 ms. What fills 800 ms is
/// everything else: the conferencing app's own output buffering, the two
/// capture paths' independent queues, and the same two drift trackers. 800 ms is
/// generous rather than measured, and it fails **silent and safe**: a machine
/// whose true lag is longer suppresses nothing and transcribes exactly as it
/// does today. The way that becomes visible is the "never found it" line the
/// live path logs, not a wrong answer here.
pub const LAG_MAX_MS: i64 = 800;

/// How far either side of a known lag a **warm** search looks.
///
/// Once a meeting has produced a few agreeing measurements, the delay is a
/// property of this machine and this route, and it moves only when a drift
/// correction moves it — up to 200 ms, in one step. A ±150 ms window follows
/// ordinary jitter without re-opening the whole search; a correction bigger than
/// the window costs a couple of missed utterances and then re-establishes
/// itself, because [`LagEstimate`] forgets what it knew after two minutes.
pub const LAG_WINDOW_MS: i64 = 150;

/// Less overlap than this and a correlation means nothing.
///
/// Twenty hops is 200 ms — two syllables. Any two signals can be made to agree
/// over a handful of points, and at the far ends of the search the overlap is
/// what shrinks first, so this is the guard that keeps the edges of the search
/// from inventing a match.
const MIN_OVERLAP_HOPS: i64 = 20;

/// Which delays to try.
///
/// **Sign convention, stated once and never re-derived: `lag_ms` is how much the
/// mic copy is delayed relative to the system copy.** Positive means the mic is
/// later, which is the physical case — speaker, then air, then microphone.
/// Everything in this module reads that way, including [`BleedEvidence::lag_ms`]
/// and [`LagEstimate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LagSearch {
    pub from_ms: i64,
    pub to_ms: i64,
}

impl LagSearch {
    /// The whole plausible range: nothing is known about this machine yet.
    pub const fn cold() -> Self {
        Self {
            from_ms: LAG_MIN_MS,
            to_ms: LAG_MAX_MS,
        }
    }

    /// [`LAG_WINDOW_MS`] either side of a lag already measured, clamped into the
    /// cold range so a wild estimate can never widen the search.
    pub fn around(lag_ms: i64) -> Self {
        let cold = Self::cold();
        Self {
            from_ms: (lag_ms - LAG_WINDOW_MS).clamp(cold.from_ms, cold.to_ms),
            to_ms: (lag_ms + LAG_WINDOW_MS).clamp(cold.from_ms, cold.to_ms),
        }
    }

    fn hops(&self) -> std::ops::RangeInclusive<i64> {
        hops_of(self.from_ms)..=hops_of(self.to_ms)
    }
}

/// Milliseconds as whole hops, to the nearest.
fn hops_of(ms: i64) -> i64 {
    let hop = ENVELOPE_HOP_MS;
    if ms >= 0 {
        (ms + hop / 2) / hop
    } else {
        -((-ms + hop / 2) / hop)
    }
}

/// The delay at which the two envelopes agree best, and how well they agree
/// there.
///
/// `sys_lead_hops` is how many hops **earlier** the system envelope begins than
/// the mic envelope: it is the bookkeeping that puts the two buffers on one
/// clock before any delay is considered, and it is separate from the delay
/// itself. With a delay of `L` hops, mic point `i` is a copy of what the system
/// played `L` hops earlier, so it is compared against system point
/// `i + sys_lead_hops - L`.
///
/// The agreement is Pearson correlation, with the mean and the variance computed
/// **over the overlap at that particular delay** — not once over the whole
/// buffer. That is what makes attenuation and any constant offset irrelevant: a
/// copy 25 dB down is a copy scaled by 0.056, and Pearson does not see a scale
/// factor at all. It is why the quiet late copy still reads as a copy, and it is
/// the reason this measures shapes rather than levels.
///
/// Zero variance on either side — digital silence, or a perfectly flat tone —
/// answers 0.0. Never a NaN: a NaN compared with a threshold is false in both
/// directions, and a filter that deletes speech may not have a value in it that
/// nobody can reason about.
pub fn best_lag(
    mic_env: &[f32],
    sys_env: &[f32],
    sys_lead_hops: i64,
    search: LagSearch,
) -> (f32, i64) {
    let mut best = (0.0f32, search.from_ms);
    let mut seen = false;
    for lag in search.hops() {
        let r = correlation_at(mic_env, sys_env, sys_lead_hops - lag);
        if !seen || r > best.0 {
            best = (r, lag * ENVELOPE_HOP_MS);
            seen = true;
        }
    }
    best
}

/// A run of `false` shorter than `bridge`, with `true` either side of it, was a
/// dip and not a stop.
///
/// Leading and trailing runs are never filled: a stretch that has not started
/// yet, or has already ended, is not pausing.
fn close_short_gaps(flags: &mut [bool], bridge: usize) {
    let mut i = 0;
    let mut started = false;
    while i < flags.len() {
        if flags[i] {
            started = true;
            i += 1;
            continue;
        }
        let mut j = i;
        while j < flags.len() && !flags[j] {
            j += 1;
        }
        if started && j < flags.len() && j - i < bridge {
            flags[i..j].fill(true);
        }
        i = j;
    }
}

/// The hop range of `mic_env` that has a partner in `sys_env` at this offset.
fn overlap(mic_len: usize, sys_len: usize, offset: i64) -> Option<(usize, usize)> {
    let start = 0i64.max(-offset);
    let end = (mic_len as i64).min(sys_len as i64 - offset);
    if end - start < MIN_OVERLAP_HOPS {
        return None;
    }
    Some((start as usize, end as usize))
}

/// Pearson correlation of the two envelopes where they overlap at `offset`.
fn correlation_at(mic_env: &[f32], sys_env: &[f32], offset: i64) -> f32 {
    let Some((start, end)) = overlap(mic_env.len(), sys_env.len(), offset) else {
        return 0.0;
    };
    let n = (end - start) as f64;
    let paired = |i: usize| {
        (
            mic_env[i] as f64,
            sys_env[(i as i64 + offset) as usize] as f64,
        )
    };

    let (mut mic_sum, mut sys_sum) = (0.0f64, 0.0f64);
    for i in start..end {
        let (m, s) = paired(i);
        mic_sum += m;
        sys_sum += s;
    }
    let (mic_mean, sys_mean) = (mic_sum / n, sys_sum / n);

    let (mut cov, mut mic_var, mut sys_var) = (0.0f64, 0.0f64, 0.0f64);
    for i in start..end {
        let (m, s) = paired(i);
        let (dm, ds) = (m - mic_mean, s - sys_mean);
        cov += dm * ds;
        mic_var += dm * dm;
        sys_var += ds * ds;
    }
    if mic_var <= 0.0 || sys_var <= 0.0 {
        return 0.0;
    }
    let r = cov / (mic_var * sys_var).sqrt();
    if r.is_finite() {
        r as f32
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// The evidence
// ---------------------------------------------------------------------------

/// How loud the far side has to be, relative to its own loudest moment in this
/// stretch, before it counts as *audibly playing*.
///
/// A fifth of the peak is about 14 dB down — below any syllable the far side
/// actually said and above the tail of one dying away. Relative rather than
/// absolute because the system channel's level is whatever the conferencing app
/// and the volume slider make it, and an absolute bar would arm on one meeting
/// and never on the next.
pub const SYSTEM_PRESENT_REL: f32 = 0.20;

/// …and the floor under that, for a stretch where the far side never played at
/// all.
///
/// Without it, a stretch of pure digital silence has a peak of zero, a relative
/// bar of zero, and every one of its hops counts as "the far side was playing" —
/// which would make the coverage condition below meaningless exactly where it
/// matters most. 0.002 RMS is about 54 dB below full scale: under any room tone
/// a microphone picks up, well over the numerical dust of a silent buffer.
pub const SYSTEM_PRESENT_ABS: f32 = 0.002;

/// How much louder than the far side can account for a hop has to be before it
/// counts as somebody talking into the microphone.
///
/// **The reference is what the far side accounts for, not the microphone's own
/// loudest moment, and that is the whole of the change.** A bar of "a third of
/// the loudest thing in this stretch" reads well until you notice what sets
/// that peak: in a stretch that *is* bleed, the loudest thing in the microphone
/// is the bleed. The person's own voice then has to beat a third of the far
/// side's copy of itself to be seen at all — and a colleague across the table,
/// farther from the microphone than the laptop speakers are, never does.
/// Measured with the generated audio below, a 2.5 s answer spoken into a 3 s
/// pause vanished completely below about nine decibels under the bleed, with
/// the veto reading exactly zero milliseconds. Nothing in this file bounds that
/// ratio, so nothing in this file may depend on it.
///
/// Asked as a comparison instead, the question has an answer at every hop,
/// including the hops the far side is talking through — which is the whole of
/// an interruption, and which a bar against the mic's own peak could never
/// reach. Two uncorrelated sounds add in power, so somebody exactly as loud as
/// the bleed puts √2 ≈ 1.41 times the bleed into the microphone and somebody
/// 5 dB louder puts about 2.0 there. Two is therefore roughly "louder than the
/// far side's copy of itself"; in practice it fires well below that, because
/// the two voices' loud moments do not line up and the person only has to win
/// somewhere.
///
/// It cannot go much lower. Measured over the 2026-08-25 meeting, the ratio
/// between the microphone and the far side wanders by about ±5 dB even across
/// hops where the far side is loud — a room colours every phoneme differently —
/// so a headroom under about 1.7 reads the room as a person and vetoes genuine
/// copies. Two keeps 22 of the 31 stretches that meeting's own audio says are
/// copies, and gives up the other nine as duplicated lines rather than risk the
/// other kind of mistake.
pub const OWN_VOICE_HEADROOM: f32 = 2.0;

/// …and the floor under that: own voice quieter than this fraction of the
/// loudest the far side's copy gets is not worth calling voice.
///
/// The headroom above compares the microphone against the far side hop by hop,
/// so wherever the far side is silent it compares the microphone against almost
/// nothing — and room tone, a chair, a breath would all clear it. This is what
/// stops that. A twelfth is about 22 dB under the bleed's own peak; on the
/// 2026-08-25 recording the microphone's quiet hops sit 33 dB under it, so a
/// real room passes under this with room to spare. It replaces a bar that stood
/// at 0.35 of the *microphone's* peak — 9 dB — which is the number that made
/// real speech invisible.
pub const OWN_VOICE_REL: f32 = 0.08;

/// How long the microphone goes on hearing the far side after the far side
/// stops: the time the room takes to fall by 20 dB.
///
/// A room rings, a laptop lid rings, and the 10 ms grid this module quantises
/// every delay onto smears the rest. Without an allowance, the hop after every
/// far-side consonant reads as "louder than the far side accounts for" — which
/// is the room finishing that syllable, not a person starting a word — and
/// [`OWN_VOICE_GAP_MS`] then bridges those flickers into something the length of
/// a sentence. Measured on the 2026-08-25 meeting, this allowance alone is the
/// difference between recognising 22 of the 31 genuine copies and recognising
/// three of them.
///
/// So the far side's envelope is run through a decay before anything is
/// compared against it: each hop holds the loudest thing the far side has done
/// recently, falling a tenth every 400 ms. 400 ms is a small room's reverberation
/// with the loudspeaker's own overhang on top of it. It cannot cover a word, so
/// it cannot hide one: a word is 120-260 ms of sustained level, and by the end
/// of one the thing it is being compared against has fallen 6-13 dB.
pub const ROOM_DECAY_MS: i64 = 400;

/// Where in the sorted hop-by-hop ratios the bleed's own gain is read off: the
/// lowest fifth.
///
/// The gain is what maps the far side's envelope onto the microphone's, and it
/// has to be measured from a microphone that may also contain a person. Every
/// hop the person talks makes the ratio *bigger*, so a mean or a least-squares
/// fit follows them and then explains their voice away as part of the copy. A
/// low quantile does not: it reads the gain off the quietest fifth of the
/// stretch, which is the fifth the person is least likely to be in. It fails
/// only when somebody talks over more than four fifths of the stretch, which is
/// the gapless-double-talk case this design has always said needs real echo
/// cancellation.
const BLEED_GAIN_QUANTILE: f32 = 0.20;

/// The shortest pause that counts as somebody having stopped.
///
/// Running speech is not continuous. Words are 120-260 ms with 40-120 ms
/// between them, and inside a word the level drops 10-20 dB at every consonant.
/// Read hop by hop with no allowance for that, "the far side was playing" would
/// measure the vowels of the far side and "the person's own voice" would measure
/// one syllable and never a sentence — so both measures below let a short dip
/// pass without calling it a stop.
///
/// This is the far side's number. It is also what makes
/// [`BleedEvidence::system_voice_ms`] comparable with the mic's `voiced_ms` at
/// all: the speech detector's own answer already has an allowance like this
/// built in, and a far more generous one (it waits 400 ms on the mic before
/// closing a stretch, [`crate::audio::vad::VadSettings::mic`]). 150 ms bridges a
/// word boundary and nothing longer.
pub const SHORTEST_STOP_MS: i64 = 150;

/// …and the longest quiet that still leaves two loud moments part of one
/// sentence.
///
/// Twice [`SHORTEST_STOP_MS`], because the two are asked different questions.
/// "The far side is playing" is asked of a channel with no room in it, at a
/// fifth of its own peak, so it survives everything but a real pause. "Somebody
/// is talking into the microphone" is asked of a channel that also contains the
/// far side, and it is answered by comparing the two — which goes quiet at every
/// consonant the *person* makes and again wherever the far side happens to be
/// loud enough to mask them. Read at 150 ms that measures syllables, and
/// [`OWN_VOICE_MS`] would never be reached by anything; 300 ms joins words into
/// a sentence and still cannot join two turns, which are separated by much more
/// than that.
///
/// This errs towards finding own voice rather than missing it, which is the
/// direction the whole file leans: the cost of a run measured too generously is
/// one duplicated line, and the cost of one measured too meanly is a sentence
/// attributed to nobody.
pub const OWN_VOICE_GAP_MS: i64 = 300;

/// What the two channels look like against each other over one stretch.
///
/// Evidence, not a verdict — [`is_bleed`] is the verdict. Kept separate so the
/// probe can print the numbers for a whole meeting and somebody can disagree
/// with where the line is drawn.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BleedEvidence {
    /// How well the two loudness shapes agreed at the best delay, -1.0 to 1.0.
    pub correlation: f32,
    /// That delay: how much the mic copy is late relative to the system copy.
    /// Positive is the physical direction.
    pub lag_ms: i64,
    /// Milliseconds inside the aligned span where the far side was audibly
    /// playing.
    pub system_voice_ms: i64,
    /// The longest run where the mic was louder than the aligned system channel
    /// can account for. That is somebody in the room talking: it is the only
    /// thing that can put loudness into a microphone that the far side did not
    /// put there.
    ///
    /// "Can account for" is measured, not assumed: the far side's envelope is
    /// scaled by the bleed's own gain (see [`BLEED_GAIN_QUANTILE`]), stretched
    /// by [`ROOM_DECAY_MS`] for the room's ringing, and a hop counts only if the
    /// mic beats that by [`OWN_VOICE_HEADROOM`] *and* clears [`OWN_VOICE_REL`]
    /// of the bleed's own peak. Because it is a comparison and not a level,
    /// **it still answers while the far side is talking** — which is the whole
    /// of an interruption, and which a bar against the mic's own peak could
    /// never do.
    ///
    /// A run breathes over a quiet shorter than [`OWN_VOICE_GAP_MS`], so it
    /// measures a sentence rather than a syllable.
    pub unexplained_ms: i64,
    /// How much of the stretch the two buffers actually had in common at that
    /// delay. Everything above is measured over this and nothing else.
    pub span_ms: i64,
}

/// Measure one stretch of mic audio against the system audio around it.
///
/// `sys_lead_ms` is how much earlier `system` begins than `mic` on the meeting
/// clock; positive means the system buffer starts first, which is the normal
/// case because a caller reads back extra system audio either side to give the
/// search room. Sub-hop placement is rounded away — 5 ms is below this module's
/// resolution and far below what the two channels' alignment is worth.
///
/// The envelopes are a few thousand floats for the longest utterance Echo will
/// ever hand over, so this allocates two of them rather than growing an API.
/// [`envelope`] is the reusable half for a caller that measures thousands of
/// stretches in a row.
pub fn examine(mic: &[f32], system: &[f32], sys_lead_ms: i64, search: LagSearch) -> BleedEvidence {
    let mut mic_env = Vec::new();
    let mut sys_env = Vec::new();
    envelope(mic, &mut mic_env);
    envelope(system, &mut sys_env);
    examine_envelopes(&mic_env, &sys_env, sys_lead_ms, search)
}

/// [`examine`] over envelopes already computed.
pub fn examine_envelopes(
    mic_env: &[f32],
    sys_env: &[f32],
    sys_lead_ms: i64,
    search: LagSearch,
) -> BleedEvidence {
    let sys_lead_hops = hops_of(sys_lead_ms);
    let (correlation, lag_ms) = best_lag(mic_env, sys_env, sys_lead_hops, search);
    let offset = sys_lead_hops - hops_of(lag_ms);
    let Some((start, end)) = overlap(mic_env.len(), sys_env.len(), offset) else {
        return BleedEvidence::default();
    };
    let at = |i: usize| sys_env[(i as i64 + offset) as usize];

    let n = end - start;
    let sys_of = |k: usize| at(start + k);
    let mic_of = |k: usize| mic_env[start + k];
    let sys_peak = (0..n).fold(0.0f32, |m, k| m.max(sys_of(k)));
    let sys_bar = (sys_peak * SYSTEM_PRESENT_REL).max(SYSTEM_PRESENT_ABS);

    // Was the far side playing? Allowed to breathe over a dip shorter than
    // [`SHORTEST_STOP_MS`] before anything is counted, so this measures pauses
    // and not consonants.
    let mut playing: Vec<bool> = (0..n).map(|k| sys_of(k) >= sys_bar).collect();
    close_short_gaps(&mut playing, (SHORTEST_STOP_MS / ENVELOPE_HOP_MS) as usize);

    // …and how loud the far side's own copy is inside this microphone. The
    // lowest fifth of the hop-by-hop ratio, over the hops the far side was
    // genuinely above its bar — see [`BLEED_GAIN_QUANTILE`] for why a quantile
    // and not a fit. With nothing above the bar there is no copy to be a copy
    // of, and a gain of zero says exactly that.
    let mut ratios: Vec<f32> = (0..n)
        .filter(|k| sys_of(*k) >= sys_bar)
        .map(|k| mic_of(k) / sys_of(k))
        .collect();
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let gain = if ratios.is_empty() {
        0.0
    } else {
        let last = ratios.len() - 1;
        ratios[(last as f32 * BLEED_GAIN_QUANTILE).round() as usize]
    };

    // What the far side can account for at each hop: its own envelope, at that
    // gain, held forward through the room's ringing (see [`ROOM_DECAY_MS`]) and
    // one hop ahead for the delay's quantisation.
    let per_hop = 10f32.powf(-(ENVELOPE_HOP_MS as f32) / ROOM_DECAY_MS as f32);
    let mut held = 0.0f32;
    let mut room: Vec<f32> = Vec::with_capacity(n);
    for k in 0..n {
        let ahead = sys_of((k + 1).min(n - 1));
        held = held.max(sys_of(k)).max(ahead);
        room.push(held);
        held *= per_hop;
    }
    let explained = |k: usize| gain * room[k];
    // The floor under it, so that where the far side accounts for nothing the
    // room does not get called a person.
    let own_floor = gain * sys_peak * OWN_VOICE_REL;

    let mut mine: Vec<bool> = (0..n)
        .map(|k| mic_of(k) > explained(k) * OWN_VOICE_HEADROOM && mic_of(k) >= own_floor)
        .collect();
    close_short_gaps(&mut mine, (OWN_VOICE_GAP_MS / ENVELOPE_HOP_MS) as usize);

    let system_hops = playing.iter().filter(|p| **p).count() as i64;
    let mut longest = 0i64;
    let mut run = 0i64;
    for m in &mine {
        run = if *m { run + 1 } else { 0 };
        longest = longest.max(run);
    }

    BleedEvidence {
        correlation,
        lag_ms,
        system_voice_ms: system_hops * ENVELOPE_HOP_MS,
        unexplained_ms: longest * ENVELOPE_HOP_MS,
        span_ms: n as i64 * ENVELOPE_HOP_MS,
    }
}

// ---------------------------------------------------------------------------
// The verdict
// ---------------------------------------------------------------------------

/// How well the two loudness shapes have to agree.
///
/// The number to beat is the *null distribution*: two unrelated people talking,
/// correlated at every one of a hundred and one candidate delays, and the best
/// of those hundred and one kept. Taking the best of many tries is what puts a
/// floor under chance. Measured over four thousand mismatched pairs of three
/// seconds each — `unrelated_speech_stays_under_the_bar` in this file — that
/// floor sits at 0.55 for one pair in twenty, 0.61 for one in a hundred, and
/// 0.65 for one in a thousand; the largest of the four thousand was 0.69. A
/// true copy measures 0.8 to 0.9 — not 1.0, because the room, the speakers and
/// the microphone all colour it on the way — and on the real meeting of
/// 2026-08-25 the stretches that overlapped far-side speech averaged 0.77 with
/// eighty of a hundred and sixteen above 0.80, against 0.25 for the ones that
/// did not.
///
/// 0.70 sits between the two with more room above the null than below the
/// copies, which is the right way round for a filter that must never delete a
/// sentence. **What it is not is a wall.** Chance crosses 0.70 in this
/// distribution roughly once in a few thousand three-second stretches, and that
/// is exactly why nothing is suppressed on a correlation alone: the coverage
/// condition and [`OWN_VOICE_MS`] are what stand behind it, and the same test
/// asserts that not one of the four thousand chance pairs was suppressed once
/// all three had their say. A threshold argued only in a comment is a threshold
/// nobody has checked; a threshold defended by the largest of two hundred draws
/// is one somebody checked too gently.
pub const BLEED_CORRELATION: f32 = 0.70;

/// Shortest stretch judgeable with no idea what this machine's delay is.
///
/// Three seconds, because the null distribution is a function of length. Same
/// four thousand pairs, same hundred and one delays, at half the length: the
/// chance floor climbs from 0.61 at the ninety-ninth percentile to 0.76, and
/// one pair in twenty clears 0.70 outright. That is not a reason to raise the
/// correlation bar — it is a reason to refuse short stretches, which is what
/// this is. There is no honest answer to give about a second and a half of
/// audio searched over a second of possible delays.
pub const MIN_SPAN_COLD_MS: i64 = 3_000;

/// …and the shortest once the delay is known.
///
/// A warm search tries thirty-one delays instead of a hundred and one, over a
/// window the meeting has already agreed on, and fewer tries do lower the
/// chance floor. **They do not lower it as far as halving the length raises
/// it**, which is what this constant used to assume. Measured, four thousand
/// pairs each:
///
/// ```text
///                       p99    max    over 0.70
///   cold, 3000 ms      0.61   0.69      0.00 %     <- the bar was set from this
///   cold, 1500 ms      0.76   0.88      5.08 %
///   warm, 1500 ms      0.72   0.82      1.55 %
///   warm, 2000 ms      0.66   0.76      0.28 %
///   warm, 2500 ms      0.62   0.71      0.03 %
/// ```
///
/// The narrower search buys back a factor of three; halving the length costs a
/// factor of fifty. So the warm floor is set where the narrowing has actually
/// paid for it and no lower: at 2.5 s the warm chance rate is the cold one's
/// equal, and at 1.5 s it was fifty times worse than the case the bar was
/// derived from. `unrelated_speech_stays_under_the_bar` measures both arms, so
/// this stops being a claim and starts being a number.
pub const MIN_SPAN_WARM_MS: i64 = 2_500;

/// Least voice the speech detector must have found in the mic stretch.
///
/// Anything shorter is not a sentence anybody would notice twice in a
/// transcript, and it is where the correlator is least reliable. This is the
/// second, *independent* condition in the phantom filter's sense: it comes from
/// Silero and not from anything measured here, so a correlator gone wrong cannot
/// satisfy it.
pub const MIN_MIC_VOICE_MS: i64 = 1_000;

/// Least far-side audio there must have been, whatever the coverage ratio says.
///
/// The ratio (six tenths of the mic's voice) is the real condition; this is the
/// floor under it, so a stretch with almost no measured mic voice cannot clear
/// the ratio by having almost no system voice either. 300 ms is one clearly
/// spoken word — below that there is nothing for the mic to be a copy *of*.
pub const MIN_SYSTEM_VOICE_MS: i64 = 300;

/// Own voice this long inside a stretch, and the stretch is not deleted, however
/// well the rest of it correlates.
///
/// This is the veto, and it is what makes talking over the far side safe. It has
/// to sit above [`crate::asr::phantom::TOO_LITTLE_VOICE_MS`] (200 ms — a cough,
/// a keyboard click, the blip that can open an utterance at all) and below a
/// clearly spoken word (300-400 ms of phonation). 400 ms is the top of that
/// band, chosen high rather than low on purpose: the cost of vetoing too eagerly
/// is one duplicated line, and the cost of vetoing too late is a sentence
/// attributed to nobody.
///
/// **What it covers, measured rather than asserted.** Sixty seeds a row of the
/// generated audio this file's tests use, `own` being the person's own level
/// against the bleed's; the tests below assert the first twelve seeds of each:
///
/// ```text
///   an answer into a far-side pause      own 0.33x          saved 60/60
///                                        own 0.20x          saved 12/60
///   a colleague across the table,
///     2.5 s inside a 3 s pause           own 0.24x          saved 57/60
///   talking straight over a far side
///     that never pauses at all           own 1.0x           saved 51/60
///                                        own 1.5x and up    saved 60/60
/// ```
///
/// The second row is the floor and it is honest about it: an answer a fifth of
/// the bleed's level, into a two-second pause, is still lost. It was lost at
/// *half* the bleed's level before, and at any level at all when the far side
/// did not pause.
///
/// The last row is where it is weakest, and honestly so: while the far side
/// never stops, the person has to be about as loud in the microphone as the far
/// side's own copy of itself before the comparison can see them. That is the
/// gapless-double-talk case this design has said from the start needs real echo
/// cancellation. What has changed is that it is a matter of degree — before,
/// the veto could not fire *at all* while the far side was playing, because own
/// voice was defined as loudness with the far side silent.
pub const OWN_VOICE_MS: i64 = 400;

/// Is this mic stretch the microphone's copy of what the computer played?
///
/// Every condition below can only ever *prevent* a suppression. None of them can
/// cause one on its own, and that is the whole design — same as
/// [`crate::asr::phantom::is_phantom`], with more evidence available.
///
/// 1. **The shapes agree** ([`BLEED_CORRELATION`]). Alone this is not enough:
///    two people saying the same thing at the same pace correlate, and so does
///    one person answering the far side in the rhythm of the far side's
///    question. Correlation says the loudness moved together, not that the
///    sound was the same sound.
/// 2. **The far side was actually playing** — for at least six tenths of the
///    voice the detector found in the mic, and never less than
///    [`MIN_SYSTEM_VOICE_MS`]. Alone this is nothing at all: on a call the far
///    side plays almost continuously, so nearly every mic utterance would pass.
///    It is here to stop the previous condition being satisfied by a lucky
///    alignment against a mostly silent system channel, where a couple of
///    matching bumps are all the evidence there is. Six tenths rather than one:
///    the two numbers are measured by different means — Silero counts voiced
///    windows on the mic, this file counts hops above a loudness bar on the
///    system side — so they are not expected to agree exactly, and the mic's
///    stretch carries padding at both ends that the far side never filled.
/// 3. **Nothing the far side cannot account for** ([`OWN_VOICE_MS`]). Alone
///    this would delete every quiet stretch of a meeting, since silence
///    accounts for itself. It is the veto, and it is what saves an
///    interruption. Note that it is *not* independent of the first condition
///    the way the second one is: both read the same two envelopes, and both get
///    harder as the person gets quieter against the bleed. Condition 2 is the
///    one measured by other means.
///
/// Plus two length floors that are not about the audio but about the
/// measurement: a stretch too short for the correlation to mean anything
/// ([`MIN_SPAN_COLD_MS`] / [`MIN_SPAN_WARM_MS`], `warm` meaning the delay was
/// already known so the search was narrow), and a stretch with too little
/// measured voice in it ([`MIN_MIC_VOICE_MS`]).
///
/// `mic_voiced_ms` is what the speech detector called voice — pass
/// [`crate::audio::vad::Utterance::measured_voice_ms`], which answers "nobody
/// measured" with "assume it was all speech", so a missing detector can never
/// cost words.
pub fn is_bleed(ev: &BleedEvidence, mic_voiced_ms: i64, warm: bool) -> bool {
    let span_floor = if warm {
        MIN_SPAN_WARM_MS
    } else {
        MIN_SPAN_COLD_MS
    };
    let coverage_floor = (mic_voiced_ms * 6 / 10).max(MIN_SYSTEM_VOICE_MS);
    ev.span_ms >= span_floor
        && mic_voiced_ms >= MIN_MIC_VOICE_MS
        && ev.correlation >= BLEED_CORRELATION
        && ev.system_voice_ms >= coverage_floor
        && ev.unexplained_ms < OWN_VOICE_MS
}

// ---------------------------------------------------------------------------
// Remembering the delay
// ---------------------------------------------------------------------------

/// Measurements kept. Five is two more than it takes to follow a step, which
/// leaves room for one outlier on either side of one.
const LAG_MEMORY: usize = 5;

/// Fewest measurements before the estimate is worth using.
///
/// One is not a measurement, it is an observation. Two that agree is the least
/// that can rule out a single wrong answer, and the price of the second one is
/// one more utterance judged with the cold search.
const LAG_MIN_HITS: usize = 2;

/// How long a measurement is worth anything. Two minutes.
///
/// A route change, a device swap, a drift correction — all of them move the true
/// delay, and none of them are visible from here. Forgetting on a timer means
/// the worst a stale number can do is narrow the search wrongly for a couple of
/// minutes, after which the search re-opens on its own. Callers that *do* see a
/// route change call [`LagEstimate::clear`] rather than waiting.
pub const LAG_MEMORY_MS: i64 = 120_000;

/// This machine's delay between what the speakers played and what the microphone
/// heard, as the meeting has measured it so far.
///
/// **The median of the last few full hits, not a running average.** A drift
/// correction moves the true lag by up to 200 ms in a *single step*
/// (`audio/clock.rs`), and the two behave completely differently on that: a
/// median follows a step in three hits and then sits exactly on the new value,
/// while it ignores a single wild reading entirely. An exponential average does
/// neither well — it crawls toward a step for a dozen hits and it is dragged off
/// by every outlier on the way. The thing being estimated is a staircase with
/// occasional nonsense on it, so the estimator is a median.
///
/// No clock of its own: every method takes `now_ms`. The capture layer has the
/// meeting clock and this has no business having a second one.
#[derive(Debug, Default, Clone)]
pub struct LagEstimate {
    hits: VecDeque<(i64, i64)>,
}

impl LagEstimate {
    /// Record one measured delay. **Only full hits belong here** — stretches
    /// that were actually judged bleed. A lag read off a stretch that failed the
    /// predicate is a lag read off noise.
    pub fn record(&mut self, now_ms: i64, lag_ms: i64) {
        self.forget_before(now_ms - LAG_MEMORY_MS);
        self.hits.push_back((now_ms, lag_ms));
        while self.hits.len() > LAG_MEMORY {
            self.hits.pop_front();
        }
    }

    /// The delay to search around, or `None` while there is not enough to go on.
    ///
    /// With an even number of measurements this is the mean of the two middle
    /// ones, which for two hits is simply their midpoint — a compromise between
    /// them rather than a coin toss between them.
    pub fn lag_ms(&self, now_ms: i64) -> Option<i64> {
        let cutoff = now_ms - LAG_MEMORY_MS;
        let mut fresh: Vec<i64> = self
            .hits
            .iter()
            .filter(|(at, _)| *at >= cutoff)
            .map(|(_, lag)| *lag)
            .collect();
        if fresh.len() < LAG_MIN_HITS {
            return None;
        }
        fresh.sort_unstable();
        let mid = fresh.len() / 2;
        Some(if fresh.len() % 2 == 1 {
            fresh[mid]
        } else {
            (fresh[mid - 1] + fresh[mid]) / 2
        })
    }

    /// The search this estimate justifies: narrow if it knows the delay, the
    /// whole range if it does not.
    pub fn search(&self, now_ms: i64) -> LagSearch {
        match self.lag_ms(now_ms) {
            Some(lag) => LagSearch::around(lag),
            None => LagSearch::cold(),
        }
    }

    /// Whether [`search`](Self::search) came back narrow — the `warm` argument
    /// to [`is_bleed`].
    pub fn is_warm(&self, now_ms: i64) -> bool {
        self.lag_ms(now_ms).is_some()
    }

    /// Everything measured so far is about a machine in a state it is no longer
    /// in: the audio route changed, the recording resumed, the system channel
    /// came back after being lost.
    pub fn clear(&mut self) {
        self.hits.clear();
    }

    fn forget_before(&mut self, cutoff_ms: i64) {
        while self.hits.front().is_some_and(|(at, _)| *at < cutoff_ms) {
            self.hits.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asr::phantom;

    const RATE: usize = TARGET_SAMPLE_RATE as usize;

    /// Room tone, so the pauses are a microphone's pauses and not digital zero.
    /// Kept well under [`SYSTEM_PRESENT_ABS`] on purpose: a noise floor is not
    /// the far side playing.
    const NOISE: f32 = 0.0005;

    /// A deterministic pseudo-random source, sixteen lines of it, because the
    /// alternative is a dependency for the sake of shuffling some syllable
    /// lengths. Numbers are Knuth's LCG constants.
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            let mut lcg = Self(seed);
            // Two turns so neighbouring seeds do not start out neighbouring.
            lcg.next_u32();
            lcg.next_u32();
            lcg
        }

        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 32) as u32
        }

        fn between(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * (self.next_u32() as f32 / u32::MAX as f32)
        }
    }

    /// How deep the dip between syllables inside one word goes: to 15% of the
    /// word's own peak, about 16 dB down.
    ///
    /// This is the parameter that decides whether the generated audio is a fair
    /// test of the correlator, so it gets an argument of its own. A burst with a
    /// single smooth hump has all its modulation at the word rate and nothing
    /// above it — and two such signals, having only a dozen features in three
    /// seconds to disagree about, correlate by chance far more than real speech
    /// does (measured here: a null maximum of 0.74 against 0.59 with the dips
    /// in). Real speech is not like that: its modulation spectrum peaks at the
    /// 4-8 Hz syllable rate but carries real structure well past 20 Hz, because
    /// every consonant inside a word closes the vocal tract and drops the level
    /// 10-20 dB. Leaving that out would make this test *flatter* than reality
    /// and the bar look harder to defend than it is.
    const SYLLABLE_DIP: f32 = 0.85;

    /// Something shaped like speech: words as raised-cosine bursts on a
    /// voice-pitched carrier with two harmonics, separated by the gaps running
    /// speech actually has, each word dipping between its own syllables.
    ///
    /// Not speech, and not trying to be. What it has to reproduce is the only
    /// thing this module reads — the *rhythm* of loudness: bursts of 120-260 ms
    /// with 40-120 ms between them, at varying levels, with one to three
    /// syllables inside each. Every parameter comes from `seed` through the LCG
    /// above, so two seeds give two unrelated speakers and the same seed gives
    /// the same audio on every machine, for ever, with no dependency.
    fn speech(ms: usize, seed: u64) -> Vec<f32> {
        let n = RATE * ms / 1_000;
        let mut out: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.37).sin() + (i as f32 * 1.13).sin()) * NOISE)
            .collect();
        let mut rng = Lcg::new(seed);
        let mut at = 0usize;
        let mut phase = 0.0f32;
        while at < n {
            let syllable = (rng.between(120.0, 260.0) as usize) * RATE / 1_000;
            let gap = (rng.between(40.0, 120.0) as usize) * RATE / 1_000;
            let f0 = rng.between(160.0, 205.0);
            let amp = rng.between(0.3, 0.9);
            let syllables = rng.between(1.0, 3.99).floor();
            let onset = rng.between(0.0, 1.0);
            let step = std::f32::consts::TAU * f0 / RATE as f32;
            for k in 0..syllable.min(n - at) {
                let t = k as f32 / syllable as f32;
                let inner = 0.5 - 0.5 * (std::f32::consts::TAU * (t * syllables + onset)).cos();
                let shape = (0.5 - 0.5 * (std::f32::consts::TAU * t).cos())
                    * ((1.0 - SYLLABLE_DIP) + SYLLABLE_DIP * inner);
                let ph = phase + step * k as f32;
                let voiced = ph.sin() + 0.5 * (2.0 * ph).sin() + 0.25 * (3.0 * ph).sin();
                out[at + k] += amp * shape * voiced * 0.5;
            }
            phase += step * syllable as f32;
            at += syllable + gap;
        }
        out
    }

    /// The same audio, `lag_ms` later and `gain` quieter, in a buffer of the same
    /// length on the same clock. A negative lag moves it earlier.
    fn delayed(source: &[f32], lag_ms: i64, gain: f32) -> Vec<f32> {
        let shift = lag_ms * RATE as i64 / 1_000;
        (0..source.len())
            .map(|i| {
                let from = i as i64 - shift;
                if from < 0 || from >= source.len() as i64 {
                    0.0
                } else {
                    source[from as usize] * gain
                }
            })
            .collect()
    }

    /// Silence a stretch of a signal, the way the far side stops talking.
    fn hush(signal: &mut [f32], from_ms: usize, to_ms: usize) {
        let from = RATE * from_ms / 1_000;
        let to = (RATE * to_ms / 1_000).min(signal.len());
        for s in &mut signal[from..to] {
            *s = 0.0;
        }
    }

    /// Add one signal on top of another starting at `at_ms`.
    fn mix_in(base: &mut [f32], other: &[f32], at_ms: usize, gain: f32) {
        let at = RATE * at_ms / 1_000;
        for (i, s) in other.iter().enumerate() {
            if at + i < base.len() {
                base[at + i] += *s * gain;
            }
        }
    }

    // -------------------------------------------------------------------
    // The envelope
    // -------------------------------------------------------------------

    #[test]
    fn the_envelope_is_one_point_every_ten_milliseconds_of_whole_windows() {
        assert_eq!(WINDOW_SAMPLES, 320);
        assert_eq!(HOP_SAMPLES, 160);
        let mut env = Vec::new();
        // One second: (16000 - 320) / 160 + 1 windows fit.
        envelope(&vec![0.5f32; RATE], &mut env);
        assert_eq!(env.len(), 99);
        // RMS of a constant is that constant.
        assert!(env.iter().all(|v| (v - 0.5).abs() < 1e-6));
        // Nothing shorter than one window is measured at all.
        envelope(&vec![0.5f32; WINDOW_SAMPLES - 1], &mut env);
        assert!(env.is_empty());
        // The buffer is reused, never appended to.
        envelope(&vec![0.5f32; RATE], &mut env);
        assert_eq!(env.len(), 99);
    }

    // -------------------------------------------------------------------
    // Finding the delay
    // -------------------------------------------------------------------

    #[test]
    fn an_identical_delayed_quieter_copy_is_bleed() {
        let system = speech(6_000, 11);
        // 25 dB down and 180 ms late: a laptop speaker across a desk.
        let mic = delayed(&system, 180, 0.056);
        let ev = examine(&mic, &system, 0, LagSearch::cold());
        assert!(
            (ev.lag_ms - 180).abs() <= ENVELOPE_HOP_MS,
            "the delay should be found within one hop, was {} ms",
            ev.lag_ms
        );
        assert!(
            ev.correlation > 0.9,
            "a copy of a signal should read as a copy, was {:.3}",
            ev.correlation
        );
        assert_eq!(
            ev.unexplained_ms, 0,
            "nothing in a copy is unexplained by the original"
        );
        assert!(is_bleed(&ev, 3_000, false));
    }

    /// Every delay in the search, including both ends of it and the impossible
    /// negative one that alignment drift makes possible anyway.
    #[test]
    fn the_delay_is_found_wherever_it_is() {
        for lag_ms in [-200, -100, 0, 120, 350, 600, 800] {
            let system = speech(9_000, 4_242);
            let mic = delayed(&system, lag_ms, 0.08);
            let ev = examine(&mic, &system, 0, LagSearch::cold());
            assert!(
                (ev.lag_ms - lag_ms).abs() <= ENVELOPE_HOP_MS,
                "a copy {lag_ms} ms late was read as {} ms late (r={:.3})",
                ev.lag_ms,
                ev.correlation
            );
            assert!(
                ev.correlation > 0.9,
                "r={:.3} at {lag_ms} ms",
                ev.correlation
            );
        }
    }

    /// The two buffers do not have to start at the same moment, and saying so
    /// wrongly must move the answer by exactly as much as the mistake.
    #[test]
    fn a_system_buffer_that_starts_earlier_is_accounted_for_before_the_delay_is() {
        let long = speech(12_000, 77);
        // The system buffer starts 2 s before the mic buffer.
        let system = long[..RATE * 9].to_vec();
        let mic_source = long[RATE * 2..RATE * 11].to_vec();
        let mic = delayed(&mic_source, 200, 0.07);
        let ev = examine(&mic, &system, 2_000, LagSearch::cold());
        assert!(
            (ev.lag_ms - 200).abs() <= ENVELOPE_HOP_MS,
            "was {} ms",
            ev.lag_ms
        );
    }

    /// One arm of the null measurement: `pairs` mismatched speakers, each
    /// stretch `span_ms` long, searched the way a caller of that length would
    /// search it.
    ///
    /// The two speakers are drawn from a pool rather than generated fresh for
    /// every pair, because generating a quarter of an hour of audio per arm is
    /// the whole cost of this test and pairing is what the measurement is
    /// about. Two unrelated seeds are two unrelated speakers however often
    /// either of them is reused.
    fn null_arm(pairs: usize, span_ms: i64, warm: bool) -> (Vec<f32>, usize) {
        const POOL: usize = 64;
        let search = if warm {
            LagSearch::around(180)
        } else {
            LagSearch::cold()
        };
        // The far side as a caller reads it back: the mic stretch plus room
        // either side for the whole search, so every candidate delay sees all of
        // the stretch and none is judged on less evidence than the others.
        let mics: Vec<Vec<f32>> = (0..POOL)
            .map(|i| speech(span_ms as usize, i as u64 * 2 + 1))
            .collect();
        let systems: Vec<Vec<f32>> = (0..POOL)
            .map(|i| {
                speech(
                    (span_ms + LAG_MAX_MS - LAG_MIN_MS) as usize,
                    i as u64 * 2 + 2_000,
                )
            })
            .collect();
        let mut correlations = Vec::with_capacity(pairs);
        let mut suppressed = 0usize;
        for pair in 0..pairs {
            let mic = &mics[pair % POOL];
            let system = &systems[(pair / POOL + pair * 7 + 1) % POOL];
            let ev = examine(mic, system, LAG_MAX_MS, search);
            correlations.push(ev.correlation);
            // Every chance it could have: claim the whole stretch was voice, so
            // only the correlation, the coverage and the veto can refuse.
            if is_bleed(&ev, span_ms, warm) {
                suppressed += 1;
            }
        }
        correlations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        (correlations, suppressed)
    }

    fn percentile(sorted: &[f32], p: f64) -> f32 {
        sorted[(((sorted.len() - 1) as f64) * p).round() as usize]
    }

    /// **Where [`BLEED_CORRELATION`], [`MIN_SPAN_COLD_MS`] and
    /// [`MIN_SPAN_WARM_MS`] are defended.**
    ///
    /// The predecessor of this test drew two hundred pairs, kept the largest
    /// correlation any of them reached, and asserted that it stayed a clear
    /// margin under the bar. That is not a measurement of a tail: the same
    /// generator reaches 0.68 in four thousand draws and would cross 0.70
    /// eventually, so the assertion was true only of the sample size. What this
    /// asserts instead is a *percentile*, which is stable, and a suppression
    /// count over the whole predicate, which is the thing that actually matters
    /// — chance clearing the correlation bar is survivable, chance deleting a
    /// sentence is not.
    ///
    /// It measures both arms. The warm one had never been measured at all, and
    /// when it was, the shorter floor it used to have (1.5 s) turned out to let
    /// chance over the bar fifty times more often than the cold case the bar was
    /// derived from — which is why [`MIN_SPAN_WARM_MS`] moved to 2.5 s.
    #[test]
    fn unrelated_speech_stays_under_the_bar() {
        const PAIRS: usize = 1_000;
        let arms = [
            ("cold", MIN_SPAN_COLD_MS, false),
            ("warm", MIN_SPAN_WARM_MS, true),
        ];
        for (name, span_ms, warm) in arms {
            let (rs, suppressed) = null_arm(PAIRS, span_ms, warm);
            let over = rs.iter().filter(|r| **r >= BLEED_CORRELATION).count();
            println!(
                "unrelated speech, {name}: {PAIRS} pairs at {span_ms} ms over {} delays — \
                 p95 {:.3}, p99 {:.3}, max {:.3}; {over} reached the {BLEED_CORRELATION:.2} bar, \
                 {suppressed} were suppressed",
                if warm {
                    2 * LAG_WINDOW_MS / ENVELOPE_HOP_MS + 1
                } else {
                    (LAG_MAX_MS - LAG_MIN_MS) / ENVELOPE_HOP_MS + 1
                },
                percentile(&rs, 0.95),
                percentile(&rs, 0.99),
                percentile(&rs, 1.0),
            );
            assert!(
                percentile(&rs, 0.99) < BLEED_CORRELATION - 0.05,
                "{name}: chance clears {:.3} once in a hundred, which leaves no room \
                 under the {BLEED_CORRELATION:.2} bar",
                percentile(&rs, 0.99),
            );
            assert_eq!(
                suppressed, 0,
                "{name}: chance deleted {suppressed} of {PAIRS} unrelated stretches"
            );
        }
    }

    /// …and the other half of that argument: the floor the warm arm used to have
    /// does *not* hold, so nobody can quietly lower it back again. Same pairs,
    /// same narrow search, 1.5 s instead of 2.5 s.
    #[test]
    fn the_short_warm_floor_this_file_used_to_have_does_not_hold() {
        let (rs, _) = null_arm(400, 1_500, true);
        let over = rs.iter().filter(|r| **r >= BLEED_CORRELATION).count();
        println!(
            "unrelated speech, warm at 1500 ms: 400 pairs — p99 {:.3}, max {:.3}, \
             {over} of 400 reached the {BLEED_CORRELATION:.2} bar",
            percentile(&rs, 0.99),
            percentile(&rs, 1.0),
        );
        assert!(
            percentile(&rs, 0.99) > BLEED_CORRELATION - 0.05,
            "if chance at 1.5 s now stays clear of the bar, MIN_SPAN_WARM_MS can \
             come back down — and this test should be the thing that says so"
        );
    }

    // -------------------------------------------------------------------
    // The verdict, condition by condition
    // -------------------------------------------------------------------

    /// Two people talking at once. The correlation is what refuses this one, and
    /// the test says so rather than being satisfied by whichever condition
    /// happened to fail first.
    #[test]
    fn both_talking_at_once_is_not_bleed() {
        let system = speech(8_000, 31);
        let mut mic = delayed(&system, 200, 0.05);
        // The person's own voice, into their own microphone: loud, and nothing
        // to do with what the far side is saying.
        mix_in(&mut mic, &speech(8_000, 32), 0, 0.6);
        let ev = examine(&mic, &system, 0, LagSearch::cold());
        assert!(
            ev.correlation < BLEED_CORRELATION,
            "double talk correlated at {:.3}",
            ev.correlation
        );
        assert!(!is_bleed(&ev, 5_000, false));
    }

    /// The case the whole veto exists for: a long stretch of genuine bleed with
    /// one real sentence inside it, said while the far side happened to pause.
    /// The correlation is *above* the bar here — the bleed dominates the stretch
    /// — so the only thing standing between that sentence and deletion is
    /// [`OWN_VOICE_MS`]. The test asserts both halves so a future change that
    /// drops the veto cannot pass by accident.
    ///
    /// The person is **quieter than the bleed here**, at a third of its level,
    /// and that is deliberate. The version of this test that shipped with the
    /// first draft made them 2.17 times *louder* than the bleed, which is the
    /// one thing nothing in this file bounds: the answer was then loud enough to
    /// clear a bar set at a third of the microphone's own peak, and the test
    /// passed for a reason that had nothing to do with the far side. Anything
    /// under about nine decibels below the bleed was deleted, silently, with the
    /// veto reading zero.
    #[test]
    fn one_real_sentence_inside_a_long_bleed_survives() {
        let mut system = speech(18_000, 51);
        // The far side stops for two seconds.
        hush(&mut system, 8_000, 10_000);
        let mut mic = delayed(&system, 200, 0.06);
        // …and the person answers into the gap, quietly — a third of the bleed's
        // level. A shout would break the correlation on its own and prove
        // nothing about the veto.
        mix_in(&mut mic, &speech(1_400, 52), 8_200, 0.02);
        let ev = examine(&mic, &system, 0, LagSearch::cold());
        assert!(
            ev.correlation >= BLEED_CORRELATION,
            "the stretch is still mostly a copy; r={:.3}",
            ev.correlation
        );
        assert!(
            ev.unexplained_ms >= OWN_VOICE_MS,
            "the sentence the far side cannot explain measured {} ms",
            ev.unexplained_ms
        );
        assert!(
            !is_bleed(&ev, 6_000, false),
            "the veto, and only the veto, has to carry this"
        );
    }

    /// The same sentence, over a range of levels against the bleed, because the
    /// level is the variable the old measure was secretly a function of.
    ///
    /// A bar set at a third of the *microphone's* loudest moment is a bar the
    /// bleed itself sets, so an answer quieter than that fraction of the bleed
    /// could not be seen at all — measured on this exact scenario, everything
    /// from 0.5x downwards was deleted with `unexplained_ms` reading 0. The
    /// point of asserting a whole sweep rather than one level is that the
    /// failure was invisible at one level and total three decibels away.
    #[test]
    fn an_answer_into_a_pause_survives_however_quiet_it_is() {
        for tenths in [1, 2, 3, 5, 10, 20] {
            let own = 0.06 * tenths as f32 / 10.0;
            let mut worst: Option<(u64, BleedEvidence)> = None;
            for seed in 0..12u64 {
                let mut system = speech(18_000, 5_000 + seed);
                hush(&mut system, 8_000, 10_000);
                let mut mic = delayed(&system, 200, 0.06);
                mix_in(&mut mic, &speech(1_400, 11_000 + seed), 8_200, own);
                let ev = examine(&mic, &system, 0, LagSearch::cold());
                if worst.is_none_or(|(_, w)| ev.unexplained_ms < w.unexplained_ms) {
                    worst = Some((seed, ev));
                }
            }
            let (seed, ev) = worst.expect("twelve seeds");
            println!(
                "answer at {:.2}x the bleed: worst of twelve is seed {seed}, \
                 r={:.3}, own voice {} ms",
                tenths as f32 / 10.0,
                ev.correlation,
                ev.unexplained_ms
            );
            // A third of the bleed's level and up, every seed keeps its sentence.
            if tenths >= 3 {
                assert!(
                    !is_bleed(&ev, 6_000, false),
                    "an answer at {:.1}x the bleed was deleted: r={:.3}, own voice {} ms",
                    tenths as f32 / 10.0,
                    ev.correlation,
                    ev.unexplained_ms
                );
            }
        }
    }

    /// A colleague across the table, farther from the microphone than the laptop
    /// speakers are, answering inside a three-second pause. Their voice is on no
    /// other channel at all, so a suppression here is not deduplication — it is
    /// two and a half seconds of somebody's speech attributed to nobody.
    ///
    /// This is the case the first draft of this file got wrong in the worst
    /// possible way: it deleted the sentence with `unexplained_ms` reading
    /// exactly 0, because the bar own voice had to clear was 0.35 of a
    /// microphone peak that the bleed itself set.
    #[test]
    fn a_colleague_across_the_table_is_not_deleted() {
        for seed in 0..12u64 {
            let mut system = speech(20_000, 6_000 + seed);
            hush(&mut system, 9_000, 12_000);
            // The far side, out of the speakers, at a quarter of full scale.
            let mut mic = delayed(&system, 200, 0.25);
            // The colleague: a twelfth of that. Quiet, and unmistakably a person.
            mix_in(&mut mic, &speech(2_500, 12_000 + seed), 9_300, 0.06);
            let ev = examine(&mic, &system, 0, LagSearch::cold());
            assert!(
                !is_bleed(&ev, 7_000, false),
                "seed {seed}: a colleague was deleted — r={:.3}, own voice {} ms, \
                 far side {} ms of {} ms",
                ev.correlation,
                ev.unexplained_ms,
                ev.system_voice_ms,
                ev.span_ms
            );
        }
    }

    /// Talking straight over a far side that never pauses — the case the veto
    /// used to be structurally incapable of seeing, because own voice was
    /// defined as microphone loudness *while the far side was silent* and the
    /// far side is never silent on a call.
    ///
    /// The measure now compares rather than gates, so it answers here too. It is
    /// still the weakest case in the file and the test says where the line is:
    /// at half again the bleed's level the interruption is safe, and the design
    /// has always said that perfectly gapless double talk where the person does
    /// not dominate the microphone needs real echo cancellation.
    #[test]
    fn talking_over_a_far_side_that_never_pauses() {
        let mut deleted = 0;
        for seed in 0..12u64 {
            let system = speech(12_000, 4_000 + seed);
            let mut mic = delayed(&system, 180, 0.20);
            // Four seconds of the person, from the fourth second, half again as
            // loud as the far side's copy of itself.
            mix_in(&mut mic, &speech(4_000, 9_000 + seed), 4_000, 0.30);
            let ev = examine(&mic, &system, 0, LagSearch::cold());
            if is_bleed(&ev, 8_000, false) {
                deleted += 1;
                println!(
                    "seed {seed}: r={:.3}, own voice {} ms — deleted",
                    ev.correlation, ev.unexplained_ms
                );
            }
        }
        assert_eq!(
            deleted, 0,
            "{deleted} of twelve interruptions over an unbroken far side were deleted"
        );
    }

    /// …and the other side of the same line: a short interjection is not a
    /// sentence, and it does not save a stretch that is otherwise a copy.
    #[test]
    fn a_short_interjection_does_not_veto() {
        let mut system = speech(18_000, 61);
        hush(&mut system, 8_000, 9_000);
        let mut mic = delayed(&system, 200, 0.06);
        mix_in(&mut mic, &speech(150, 62), 8_300, 0.13);
        let ev = examine(&mic, &system, 0, LagSearch::cold());
        // A 150 ms blip reads as less than 150 ms of own voice: it starts and
        // ends inside the room's own tail from the syllable before it, so part
        // of it is accounted for. Still nowhere near the veto, which is the
        // point — a cough may not save a bleed line.
        assert!(
            ev.unexplained_ms < OWN_VOICE_MS,
            "a 150 ms noise measured {} ms of own voice",
            ev.unexplained_ms
        );
        assert!(is_bleed(&ev, 6_000, false));
    }

    /// Silence is not evidence of anything, and it is never a division by zero.
    #[test]
    fn silence_on_either_side_is_never_bleed_and_never_a_nan() {
        let speech_6s = speech(6_000, 71);
        let quiet = vec![0.0f32; RATE * 6];

        let mic_silent = examine(&quiet, &speech_6s, 0, LagSearch::cold());
        assert!(mic_silent.correlation.is_finite());
        assert_eq!(mic_silent.correlation, 0.0);
        assert_eq!(mic_silent.unexplained_ms, 0);
        assert!(!is_bleed(&mic_silent, 3_000, false));

        let system_silent = examine(&speech_6s, &quiet, 0, LagSearch::cold());
        assert!(system_silent.correlation.is_finite());
        assert_eq!(system_silent.correlation, 0.0);
        assert_eq!(
            system_silent.system_voice_ms, 0,
            "digital silence is not the far side playing"
        );
        assert!(!is_bleed(&system_silent, 3_000, false));

        let both = examine(&quiet, &quiet, 0, LagSearch::cold());
        assert!(both.correlation.is_finite());
        assert!(!is_bleed(&both, 3_000, false));
    }

    /// A flat envelope has no variance to correlate, which is a 0.0 and not a
    /// NaN. A NaN would compare false against the bar in both directions and
    /// nobody could reason about it.
    #[test]
    fn a_flat_envelope_correlates_with_nothing() {
        let flat = vec![0.3f32; RATE * 6];
        let mut env = Vec::new();
        envelope(&flat, &mut env);
        let (r, _) = best_lag(&env, &env, 0, LagSearch::cold());
        assert_eq!(r, 0.0);
        assert!(r.is_finite());
        // …and against real speech, likewise.
        let ev = examine(&flat, &speech(6_000, 81), 0, LagSearch::cold());
        assert!(ev.correlation.is_finite());
        assert_eq!(ev.correlation, 0.0);
    }

    /// How quiet the copy is changes nothing: the correlation is computed on
    /// zero-mean, unit-variance envelopes over the overlap, so a scale factor is
    /// invisible to it. This is why a copy 25 dB down still reads as a copy.
    #[test]
    fn attenuation_does_not_change_the_verdict() {
        let system = speech(8_000, 91);
        let mut verdicts = Vec::new();
        for gain in [0.02f32, 0.5] {
            let mut mic = delayed(&system, 160, gain);
            // A microphone's own noise floor, at the same absolute level either
            // way — so the quiet copy really is the harder one.
            for (i, s) in mic.iter_mut().enumerate() {
                *s += ((i as f32 * 0.61).sin() + (i as f32 * 1.87).sin()) * NOISE;
            }
            let ev = examine(&mic, &system, 0, LagSearch::cold());
            verdicts.push((gain, ev, is_bleed(&ev, 4_000, false)));
        }
        for (gain, ev, verdict) in &verdicts {
            assert!(
                *verdict,
                "gain {gain}: r={:.3} lag={} system_voice={} unexplained={}",
                ev.correlation, ev.lag_ms, ev.system_voice_ms, ev.unexplained_ms
            );
            assert!((ev.lag_ms - 160).abs() <= ENVELOPE_HOP_MS);
        }
        // Twenty-five times quieter, and the two correlations are within a
        // whisker of each other.
        let spread = (verdicts[0].1.correlation - verdicts[1].1.correlation).abs();
        assert!(
            spread < 0.05,
            "attenuation moved the correlation by {spread:.3}"
        );
    }

    /// Two and four fifths of a second is too little to judge against a hundred
    /// and one candidate delays and enough to judge against thirty-one. Same
    /// audio, same evidence, two answers — which is the length floor doing its
    /// job and nothing else.
    #[test]
    fn a_stretch_too_short_to_judge_cold_is_judged_warm() {
        let system = speech(2_800, 101);
        let mic = delayed(&system, 100, 0.07);

        let cold = examine(&mic, &system, 0, LagSearch::cold());
        assert!(cold.span_ms < MIN_SPAN_COLD_MS);
        assert!(cold.span_ms >= MIN_SPAN_WARM_MS);
        assert!(cold.correlation >= BLEED_CORRELATION);
        assert!(!is_bleed(&cold, 1_200, false), "too short to judge cold");

        let warm = examine(&mic, &system, 0, LagSearch::around(100));
        assert!(
            is_bleed(&warm, 1_200, true),
            "r={:.3} span={} system_voice={} unexplained={}",
            warm.correlation,
            warm.span_ms,
            warm.system_voice_ms,
            warm.unexplained_ms
        );
    }

    /// Each condition on its own, so a change that quietly stops one of them
    /// mattering fails here rather than in a meeting.
    #[test]
    fn every_condition_can_refuse_on_its_own() {
        let good = BleedEvidence {
            correlation: 0.85,
            lag_ms: 180,
            system_voice_ms: 4_000,
            unexplained_ms: 40,
            span_ms: 6_000,
        };
        assert!(is_bleed(&good, 3_000, false));

        assert!(!is_bleed(
            &BleedEvidence {
                correlation: BLEED_CORRELATION - 0.01,
                ..good
            },
            3_000,
            false
        ));
        assert!(!is_bleed(
            &BleedEvidence {
                system_voice_ms: 1_799,
                ..good
            },
            3_000,
            false
        ));
        assert!(!is_bleed(
            &BleedEvidence {
                unexplained_ms: OWN_VOICE_MS,
                ..good
            },
            3_000,
            false
        ));
        assert!(!is_bleed(
            &BleedEvidence {
                span_ms: MIN_SPAN_COLD_MS - 10,
                ..good
            },
            3_000,
            false
        ));
        assert!(!is_bleed(&good, MIN_MIC_VOICE_MS - 1, false));
        // The floor under the coverage ratio: a stretch with almost no measured
        // mic voice cannot pass by having almost no system voice either.
        assert!(!is_bleed(
            &BleedEvidence {
                system_voice_ms: MIN_SYSTEM_VOICE_MS - 1,
                ..good
            },
            MIN_MIC_VOICE_MS,
            false
        ));
    }

    // -------------------------------------------------------------------
    // Remembering the delay
    // -------------------------------------------------------------------

    #[test]
    fn one_measurement_is_not_an_estimate() {
        let mut lag = LagEstimate::default();
        assert_eq!(lag.lag_ms(0), None);
        assert!(!lag.is_warm(0));
        assert_eq!(lag.search(0), LagSearch::cold());
        lag.record(1_000, 180);
        assert_eq!(lag.lag_ms(1_000), None);
        lag.record(2_000, 200);
        assert_eq!(lag.lag_ms(2_000), Some(190));
        assert!(lag.is_warm(2_000));
        assert_eq!(lag.search(2_000), LagSearch::around(190));
    }

    /// The two behaviours a median was chosen for, in one test: it ignores a
    /// single wild reading, and it follows a real step within three readings.
    #[test]
    fn the_estimate_ignores_an_outlier_and_follows_a_step() {
        let mut lag = LagEstimate::default();
        for (at, ms) in [(1_000, 180), (2_000, 190), (3_000, 185)] {
            lag.record(at, ms);
        }
        assert_eq!(lag.lag_ms(3_000), Some(185));
        // One nonsense measurement moves it not at all.
        lag.record(4_000, 780);
        assert_eq!(lag.lag_ms(4_000), Some(187));
        // A drift correction: the true delay steps by 200 ms and stays there.
        lag.record(5_000, 385);
        lag.record(6_000, 390);
        lag.record(7_000, 388);
        assert_eq!(
            lag.lag_ms(7_000),
            Some(388),
            "three hits after a step, the median should be on the new value"
        );
    }

    #[test]
    fn an_estimate_is_forgotten_after_two_minutes_and_on_demand() {
        let mut lag = LagEstimate::default();
        lag.record(1_000, 180);
        lag.record(2_000, 200);
        assert!(lag.is_warm(2_000));
        // Both are still fresh two minutes after the *older* of them.
        assert!(lag.is_warm(1_000 + LAG_MEMORY_MS));
        // A millisecond later there is one measurement left, which is not an
        // estimate.
        assert!(!lag.is_warm(1_001 + LAG_MEMORY_MS));
        // A route change does not wait for the timer.
        lag.record(200_000, 180);
        lag.record(201_000, 200);
        assert!(lag.is_warm(201_000));
        lag.clear();
        assert!(!lag.is_warm(201_000));
    }

    #[test]
    fn a_warm_search_can_never_be_wider_than_a_cold_one() {
        for lag in [-5_000, LAG_MIN_MS, 0, 300, LAG_MAX_MS, 5_000] {
            let warm = LagSearch::around(lag);
            assert!(warm.from_ms >= LAG_MIN_MS && warm.to_ms <= LAG_MAX_MS);
            assert!(warm.to_ms - warm.from_ms <= 2 * LAG_WINDOW_MS);
        }
    }

    // -------------------------------------------------------------------
    // The constants against each other
    // -------------------------------------------------------------------

    /// The relationships between the numbers, asserted at compile time, because
    /// each one is an argument that would quietly stop being true if somebody
    /// tuned a single constant on its own.
    #[test]
    fn the_constants_hold_together() {
        const {
            // The veto has to sit above the blip that opens a stretch at all —
            // otherwise a cough would save every bleed line — and below a
            // clearly spoken word.
            assert!(OWN_VOICE_MS > phantom::TOO_LITTLE_VOICE_MS);
            assert!(OWN_VOICE_MS < 700);
            // A warm search buys a shorter stretch, never a longer one.
            assert!(MIN_SPAN_WARM_MS < MIN_SPAN_COLD_MS);
            // …and a stretch has to be long enough to hold the voice required
            // of it.
            assert!(MIN_MIC_VOICE_MS <= MIN_SPAN_WARM_MS);
            // The floor under the coverage ratio has to be under the ratio it
            // is a floor for.
            assert!(MIN_SYSTEM_VOICE_MS < MIN_MIC_VOICE_MS * 6 / 10);
            // The search covers the physical direction and a drift correction's
            // worth of the impossible one.
            assert!(LAG_MIN_MS < 0 && LAG_MAX_MS > 0);
            assert!(-LAG_MIN_MS >= 200);
            // A warm search is a small part of a cold one, or it buys nothing.
            assert!(2 * LAG_WINDOW_MS < (LAG_MAX_MS - LAG_MIN_MS) / 2);
            // Bridging a syllable gap, not a turn.
            assert!(SHORTEST_STOP_MS < OWN_VOICE_GAP_MS);
            assert!(OWN_VOICE_GAP_MS < OWN_VOICE_MS);
            assert!(SHORTEST_STOP_MS % ENVELOPE_HOP_MS == 0);
            assert!(OWN_VOICE_GAP_MS % ENVELOPE_HOP_MS == 0);
            // Windows overlap, so a syllable edge cannot fall between two.
            assert!(ENVELOPE_HOP_MS * 2 == ENVELOPE_WINDOW_MS);
            // Own voice is measured against what the far side accounts for, so
            // the headroom has to be a real margin — and it has to be under the
            // level at which two equally loud sounds add (√2), or somebody
            // talking over the far side at the same level could never be seen.
            assert!(OWN_VOICE_HEADROOM > 1.0);
            assert!(OWN_VOICE_HEADROOM < 3.0);
            // The floor under it is far below the bar it replaced (0.35 of the
            // mic's own peak), which is the point of the whole change.
            assert!(OWN_VOICE_REL < 0.2 && OWN_VOICE_REL > 0.0);
            // The room's ringing may not last as long as a word, or it would
            // account for one.
            assert!(ROOM_DECAY_MS < 500);
            assert!(BLEED_GAIN_QUANTILE > 0.0 && BLEED_GAIN_QUANTILE < 0.5);
            // …and the far side's presence bar sits above the room tone this
            // file's generated audio has, or "the far side is playing" would be
            // true of silence.
            assert!(SYSTEM_PRESENT_ABS > NOISE);
        }
    }
}
