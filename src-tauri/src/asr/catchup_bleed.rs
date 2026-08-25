//! The other half of bleed suppression: the pass that reads the meeting back
//! off disk.
//!
//! [`crate::audio::bleed_guard`] does this while the meeting is happening, over
//! a thirty-second ring of the computer's own audio. That is not enough on its
//! own, and this module is not an optimisation — it is what makes the live half
//! *stick*.
//!
//! **Where this sits in the sequence.** This landed first and
//! [`crate::audio::bleed_guard`]'s wiring into the capture path landed after
//! it, so for one commit this pass was the only half there was — and it earned
//! its place even then, because catch-up reads the microphone's copy of the far
//! side back and writes it into the kept transcript of every meeting taken on
//! loudspeakers. With both halves in place it is what makes the live one
//! *stick*.
//!
//! [`crate::asr::catchup`] plans its work as "the audio on disk, minus the
//! stretches that have text against them". A stretch the live path suppressed
//! has no text against it, **by construction**: suppression is deduplication,
//! so nothing is written down and nothing is meant to be. Catch-up therefore
//! sees exactly those seconds as a hole, reads them back, and transcribes them
//! at the end of every meeting — undoing every suppression the live path made,
//! a few minutes later, in the copy of the transcript people actually keep.
//!
//! So the same question has to be asked again here. **The same question**: this
//! module owns no threshold, no correlation bar and no length floor of its own.
//! [`crate::audio::bleed::examine_envelopes`] and
//! [`crate::audio::bleed::is_bleed`] decide, exactly as they do live, and the
//! three refusals are named with the live half's own constants
//! ([`TOO_SHORT`], [`NOT_ON_HAND`], [`PART_ON_HAND`]). Only where the far side
//! comes from is different: [`crate::asr::catchup::AudioSource::read_window`]
//! over the system channel's committed chunks — which is
//! [`crate::audio::read_window`] in production — instead of the in-memory ring.
//!
//! ## The three things that are genuinely different offline
//!
//! 1. **Judging happens before the audio is packed.** A stretch judged after
//!    packing has already cost an encode; judged before, it costs neither the
//!    decode nor the row, and the packed window stays free of audio nobody will
//!    ever read. See the call site in [`crate::asr::catchup`].
//! 2. **The far side is read a page at a time** ([`SystemPager`]). The naïve
//!    version asks for a fresh ~25 s window of the system channel per mic
//!    utterance; an hour-long meeting with six hundred of them would spend
//!    minutes of wall clock decoding FLAC that overlaps the FLAC it decoded a
//!    moment ago. Utterances arrive in time order, so two thirty-second pages
//!    turn six hundred reads into about a hundred and twenty.
//! 3. **The route cannot be asked after the fact.** Whether the person had
//!    headphones in during a meeting that ended twenty minutes ago is not
//!    knowable — CoreAudio answers about *now* — so this pass calibrates
//!    itself instead: it stays armed until ten stretches that each had a real
//!    chance to match have all missed, and then stops looking (see
//!    [`ENOUGH_MISSED_CHANCES`]). On a headphone meeting that costs ten page
//!    touches and finds nothing, which is what the live route gate would have
//!    bought and no more.
//!
//! Everything else is deliberately identical, including the direction the whole
//! design leans: a stretch this pass cannot judge *in full* is transcribed, not
//! deleted. An unanswered question is not evidence.

use crate::asr::catchup::AudioSource;
use crate::asr::AsrError;
use crate::audio::bleed::{
    covers_every_lag, envelope, examine_envelopes, is_bleed, LagEstimate, LagSearch, LAG_MAX_MS,
    LAG_MIN_MS, MIN_SPAN_COLD_MS, MIN_SPAN_WARM_MS, MIN_SYSTEM_VOICE_MS,
};
use crate::audio::bleed_guard::{Verdict, NOT_ON_HAND, PART_ON_HAND, TOO_SHORT};
use crate::audio::vad::{Utterance, MAX_UTTERANCE_MS};
use crate::audio::{ChunkRef, TARGET_SAMPLE_RATE};
use crate::types::Channel;

/// Exactly sixteen samples per millisecond at 16 kHz — the same arithmetic
/// [`crate::audio::bleed_guard`] leans on, and the reason a page can be cut at
/// a millisecond with no rounding at all.
const SAMPLES_PER_MS: i64 = TARGET_SAMPLE_RATE as i64 / 1_000;

const _: () = assert!(
    SAMPLES_PER_MS * 1_000 == TARGET_SAMPLE_RATE as i64,
    "a page is cut at whole milliseconds, which is only exact because 16 kHz is \
     a whole number of samples per millisecond"
);

/// How much of the far side one page holds.
///
/// The same thirty seconds the live ring holds, for a related but not identical
/// reason. Live, thirty seconds is "the longest utterance plus the lag search,
/// plus headroom for a speech thread that fell behind". Here nothing falls
/// behind — the whole meeting is on disk — and the number is a bargain between
/// two costs instead: a bigger page reads more FLAC than any one utterance
/// needs, and a smaller one has to be replaced more often. Thirty seconds is
/// also, not by accident, one chunk of the recording
/// ([`crate::audio::CHUNK_SECONDS`]), so a page boundary tends to fall where a
/// file boundary already is.
pub const PAGE_MS: i64 = 30_000;

/// The widest stretch of the far side anything will ask for: the longest
/// utterance the detector can emit, plus the whole lag search either side of
/// it.
const WIDEST_SPAN_MS: i64 = MAX_UTTERANCE_MS + LAG_MAX_MS - LAG_MIN_MS;

const _: () = assert!(
    WIDEST_SPAN_MS <= PAGE_MS,
    "two adjacent pages have to cover any span this pass can ask for, or the \
     widest utterances could never be judged at all — and a span that needs \
     three pages is one this pager declines rather than answers partly"
);

/// How many genuine chances to match may all miss before this pass stops
/// looking.
///
/// This is the offline stand-in for the loudspeaker gate. A stretch counts only
/// when the far side
/// was genuinely audible under it — [`MIN_SYSTEM_VOICE_MS`] of it, the same bar
/// [`crate::audio::bleed_guard::BleedReport::opportunities`] uses — so a
/// meeting where the other person barely spoke never disarms anything, because
/// it never offered anything to find a copy of.
///
/// Ten missed chances in a row is a strong statement about a meeting: on the
/// 2026-08-25 recording every one of the twelve system-channel segments
/// overlapped a mic segment, so a machine that was going to find copies finds
/// them early. Ten is what it costs a headphone meeting to work that out: ten
/// stretches judged, at most ten page reads, and then nothing for the rest of
/// the pass.
///
/// **A hit resets it.** The count is a streak, not a total: a meeting that has
/// produced one real suppression has proved the path exists, and a quiet
/// stretch in the middle of it is not evidence against that.
pub const ENOUGH_MISSED_CHANCES: u32 = 10;

/// Nothing was measured, because this pass has stopped looking — the far side
/// was audible ten times over and the microphone never once carried a copy of
/// it. See [`ENOUGH_MISSED_CHANCES`].
pub const NEVER_A_COPY: &str =
    "nothing this microphone recorded turned out to be a copy of what the computer played";

/// Nothing was measured, because the computer's audio would not read back.
///
/// Kept apart from [`NOT_ON_HAND`], which is about seconds the recording never
/// held: this one is a disk that answered with an error, and the two want
/// different things done about them.
pub const COULD_NOT_READ: &str = "the computer's audio for this stretch could not be read back";

// ---------------------------------------------------------------------------
// The pager
// ---------------------------------------------------------------------------

/// One page of the far side, as it was read.
#[derive(Debug)]
struct Page {
    /// Which [`PAGE_MS`]-aligned page this is: it covers
    /// `[index * PAGE_MS, (index + 1) * PAGE_MS)` on the meeting clock.
    index: i64,
    samples: Vec<f32>,
}

impl Page {
    fn starts_at_ms(&self) -> i64 {
        self.index * PAGE_MS
    }
}

/// The computer's own audio, read back from disk in [`PAGE_MS`] pages and kept
/// two at a time.
///
/// **This is the whole performance story of the offline half.** Every mic
/// utterance wants the far side from [`LAG_MAX_MS`] before it to
/// `-`[`LAG_MIN_MS`] after it — up to twenty-five seconds — and asking for that
/// span directly would mean a fresh read per utterance, each one overlapping
/// the last by nearly all of itself. An hour-long meeting has around six
/// hundred mic utterances in it; six hundred reads of twenty-five seconds is
/// fifteen thousand seconds of FLAC decoded to judge three thousand six hundred
/// seconds of meeting, which is minutes of wall clock spent re-reading.
///
/// Utterances arrive in time order, so a page and the page before it are all
/// any of them ever need. Two pages is 3.8 MB, and an hour of meeting becomes
/// about a hundred and twenty sequential reads — one per thirty seconds of
/// recording, which is the least any reader of the whole far side could do.
///
/// **Pages are aligned to the clock, never to the utterance.** Alignment is
/// what makes two overlapping spans share a page: a pager that read "twenty-five
/// seconds around this utterance" would have a different page for every
/// utterance and would never hit.
///
/// A span the two pages cannot cover — before the recording's own far side
/// begins, after it ends, or wider than two pages — is **declined**, never
/// served in part. That is the lesson of
/// [`crate::audio::bleed::covers_every_lag`]: a veto can only veto seconds it
/// was shown, so half an answer is worse than none.
#[derive(Debug)]
pub struct SystemPager {
    /// The system channel's committed chunks, cloned once for the pass.
    chunks: Vec<ChunkRef>,
    /// Where the far side of this recording begins and ends on the meeting
    /// clock. Holes *inside* it read back as silence, which is the honest
    /// answer — the correlator then finds no far side there and declines on the
    /// coverage condition.
    reach: (i64, i64),
    /// The page a span was last served from, and the one before it.
    newer: Option<Page>,
    older: Option<Page>,
    reads: u64,
}

impl SystemPager {
    /// A pager over one meeting's far side, or `None` when there is none.
    pub fn over(chunks: &[ChunkRef]) -> Option<Self> {
        let from = chunks.iter().map(|c| c.t_start_ms).min()?;
        let to = chunks.iter().map(|c| c.t_end_ms).max()?;
        if to <= from {
            return None;
        }
        Some(Self {
            chunks: chunks.to_vec(),
            reach: (from.max(0), to),
            newer: None,
            older: None,
            reads: 0,
        })
    }

    /// How many times this pager has actually gone to disk.
    ///
    /// The number the whole design is about, so it is measured rather than
    /// argued: `an_hour_of_meeting_is_read_once` runs six hundred stretches
    /// past this pager and asserts what it cost — 120 reads, one per thirty
    /// seconds of recording.
    pub fn reads(&self) -> u64 {
        self.reads
    }

    /// The far side over `[from_ms, to_ms)`, or `None` when this pager cannot
    /// answer about all of it.
    ///
    /// `out` is the caller's scratch buffer, cleared and refilled — a judged
    /// utterance allocates nothing, exactly as live.
    async fn read<A: AudioSource>(
        &mut self,
        audio: &A,
        from_ms: i64,
        to_ms: i64,
        out: &mut Vec<f32>,
    ) -> Result<Option<i64>, AsrError> {
        out.clear();
        if to_ms <= from_ms || from_ms < self.reach.0 || to_ms > self.reach.1 {
            return Ok(None);
        }
        let first = from_ms.div_euclid(PAGE_MS);
        let last = (to_ms - 1).div_euclid(PAGE_MS);
        if last - first > 1 {
            // Unreachable while [`WIDEST_SPAN_MS`] holds, and still answered
            // rather than asserted: the cost of being wrong is one stretch
            // transcribed twice, and the cost of assuming is a panic in the
            // middle of somebody's meeting.
            return Ok(None);
        }
        // Oldest first, so that when both have to be fetched the older one is
        // the one that survives in `older`.
        for index in first..=last {
            self.fetch(audio, index).await?;
        }
        for index in first..=last {
            let Some(page) = self.held(index) else {
                out.clear();
                return Ok(None);
            };
            let take_from = from_ms.max(page.starts_at_ms());
            let take_to = to_ms.min(page.starts_at_ms() + PAGE_MS);
            let at = ((take_from - page.starts_at_ms()) * SAMPLES_PER_MS) as usize;
            let end = ((take_to - page.starts_at_ms()) * SAMPLES_PER_MS) as usize;
            if end > page.samples.len() || at > end {
                // A read that came back shorter than it was asked for. Nothing
                // partial is served: the coverage check downstream would refuse
                // it anyway, and refusing here says so one step earlier.
                out.clear();
                return Ok(None);
            }
            out.extend_from_slice(&page.samples[at..end]);
        }
        Ok(Some(from_ms))
    }

    fn held(&self, index: i64) -> Option<&Page> {
        [self.newer.as_ref(), self.older.as_ref()]
            .into_iter()
            .flatten()
            .find(|page| page.index == index)
    }

    /// Make sure page `index` is held, reading it if it is not.
    async fn fetch<A: AudioSource>(&mut self, audio: &A, index: i64) -> Result<(), AsrError> {
        if self.held(index).is_some() {
            return Ok(());
        }
        let from_ms = index * PAGE_MS;
        let samples = audio
            .read_window(&self.chunks, from_ms, from_ms + PAGE_MS)
            .await?;
        self.reads += 1;
        self.older = self.newer.take();
        self.newer = Some(Page { index, samples });
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The judge
// ---------------------------------------------------------------------------

/// Bleed suppression for one catch-up pass.
///
/// Built once per pass and asked about every mic utterance before it is packed.
/// It holds what has to be remembered across utterances: the pages, this
/// machine's delay as the meeting has measured it, and whether looking is still
/// worth it.
#[derive(Debug)]
pub struct OfflineBleed {
    pager: SystemPager,
    lag: LagEstimate,
    /// `None` while armed; otherwise the reason, ready to be handed back as an
    /// [`Verdict::Undecided`].
    disarmed: Option<&'static str>,
    /// Genuine chances to match, in a row, that all missed. Reset by a hit —
    /// see [`ENOUGH_MISSED_CHANCES`].
    missed_in_a_row: u32,
    // Scratch, owned so that judging an utterance allocates nothing.
    system: Vec<f32>,
    mic_env: Vec<f32>,
    sys_env: Vec<f32>,
    examined: u32,
    suppressed: u32,
}

impl OfflineBleed {
    /// One of these for a meeting that has far side to be a copy of, or `None`
    /// for one that has not.
    ///
    /// The mic-only meeting is the first question, asked exactly where
    /// `diarize::pipeline::voice_channel` asks it and answered the same way:
    /// with no system chunks there is no second copy of anything, so
    /// suppression could only ever be deletion. The whole of this module is
    /// then skipped — no pager, no pages, no correlation, no read.
    pub fn over(system_chunks: &[ChunkRef]) -> Option<Self> {
        Some(Self {
            pager: SystemPager::over(system_chunks)?,
            lag: LagEstimate::default(),
            disarmed: None,
            missed_in_a_row: 0,
            system: Vec::with_capacity(WIDEST_SPAN_MS as usize * SAMPLES_PER_MS as usize),
            mic_env: Vec::new(),
            sys_env: Vec::new(),
            examined: 0,
            suppressed: 0,
        })
    }

    pub fn armed(&self) -> bool {
        self.disarmed.is_none()
    }

    pub fn suppressed(&self) -> u32 {
        self.suppressed
    }

    pub fn reads(&self) -> u64 {
        self.pager.reads()
    }

    /// Is this stretch of the microphone already in the transcript, from the
    /// computer's own side of the call?
    ///
    /// The one question the catch-up pass asks. Everything that is not a
    /// measured copy is transcribed exactly as it would have been —
    /// [`Verdict::Undecided`] included, which is the whole of the safety
    /// argument: this pass declines far more often than it suppresses, and
    /// every decline costs a duplicated line rather than a lost sentence.
    pub async fn suppresses<A: AudioSource>(&mut self, audio: &A, utterance: &Utterance) -> bool {
        match self.judge(audio, utterance).await {
            Verdict::Bleed(evidence) => {
                tracing::debug!(
                    target: "echo::asr",
                    t_start_ms = utterance.t_start_ms,
                    t_end_ms = utterance.t_end_ms,
                    correlation = evidence.correlation as f64,
                    lag_ms = evidence.lag_ms,
                    system_voice_ms = evidence.system_voice_ms,
                    unexplained_ms = evidence.unexplained_ms,
                    "this stretch of the microphone is the computer's own audio coming back; \
                     the transcript already has these words from the computer's own side"
                );
                true
            }
            Verdict::Undecided(_) | Verdict::Pass => false,
        }
    }

    /// The verdict itself, with nothing logged: what the tests read.
    ///
    /// The shape is [`crate::audio::bleed_guard::BleedGuard::judge`]'s, refusal
    /// for refusal, because the two have to agree about a meeting or the
    /// offline pass would quietly re-transcribe what the live one suppressed —
    /// and about the one thing they must never do, which is judge part of a
    /// sentence and let the verdict stand for all of it.
    ///
    /// **The search is always cold.** Live, a warm search is worth the
    /// bookkeeping because it is the difference between judging a two-and-a-half
    /// second stretch and refusing it. Here the bottleneck is the FLAC decode
    /// and the packed encode either side of it, not a hundred and one
    /// correlations over an envelope that is already computed, so the cold
    /// search — which cannot be narrowed onto the wrong delay by a drift
    /// correction it has not seen — is simply the more robust of the two at a
    /// price nothing here can feel. The estimate is still kept, because the
    /// *span floor* is a separate question: a meeting that has measured its own
    /// delay twice can judge shorter stretches ([`MIN_SPAN_WARM_MS`]), and
    /// `chance_never_clears_the_bar_at_the_warm_floor_with_a_cold_search`
    /// measures that this remains true when the search that found it was wide.
    pub async fn judge<A: AudioSource>(&mut self, audio: &A, utterance: &Utterance) -> Verdict {
        // The computer's own channel is the *original*; there is nothing for it
        // to be a copy of.
        if utterance.channel != Channel::Mic {
            return Verdict::Pass;
        }
        if let Some(reason) = self.disarmed {
            return Verdict::Undecided(reason);
        }

        let now_ms = utterance.t_end_ms;
        let voiced_ms = utterance.measured_voice_ms();
        let warm = self.lag.is_warm(now_ms);

        let stretch_ms = utterance.samples.len() as i64 / SAMPLES_PER_MS;
        let span_floor = if warm {
            MIN_SPAN_WARM_MS
        } else {
            MIN_SPAN_COLD_MS
        };
        if stretch_ms < span_floor {
            return Verdict::Undecided(TOO_SHORT);
        }

        // Every delay in the search has to see the whole stretch, so the far
        // side is read from LAG_MAX_MS before the utterance starts to
        // LAG_MIN_MS after it ends.
        let read = self
            .pager
            .read(
                audio,
                utterance.t_start_ms - LAG_MAX_MS,
                utterance.t_end_ms - LAG_MIN_MS,
                &mut self.system,
            )
            .await;
        let system_from_ms = match read {
            Ok(Some(from_ms)) => from_ms,
            Ok(None) => return Verdict::Undecided(NOT_ON_HAND),
            Err(error) => {
                tracing::debug!(
                    target: "echo::asr",
                    %error,
                    t_start_ms = utterance.t_start_ms,
                    "could not read the computer's audio for this stretch, so it is \
                     written down as it stands"
                );
                return Verdict::Undecided(COULD_NOT_READ);
            }
        };
        let sys_lead_ms = utterance.t_start_ms - system_from_ms;

        envelope(&utterance.samples, &mut self.mic_env);
        envelope(&self.system, &mut self.sys_env);

        // …and "wide enough" is checked against the arrays that actually came
        // back rather than assumed from the arithmetic above — the same reason
        // the live guard checks it: the own-voice veto can only veto seconds it
        // was shown, so a person talking in a part nothing was read for would
        // be deleted along with the echo, `unexplained_ms` reading zero.
        if !covers_every_lag(
            self.mic_env.len(),
            self.sys_env.len(),
            sys_lead_ms,
            LagSearch::cold(),
        ) {
            return Verdict::Undecided(PART_ON_HAND);
        }

        self.examined += 1;
        let evidence =
            examine_envelopes(&self.mic_env, &self.sys_env, sys_lead_ms, LagSearch::cold());
        if is_bleed(&evidence, voiced_ms, warm) {
            self.suppressed += 1;
            self.missed_in_a_row = 0;
            // Only full hits teach the estimate: a lag read off a stretch that
            // failed the predicate is a lag read off noise.
            self.lag.record(now_ms, evidence.lag_ms);
            return Verdict::Bleed(evidence);
        }
        if evidence.system_voice_ms >= MIN_SYSTEM_VOICE_MS {
            // A real chance to match, and it missed.
            self.missed_in_a_row += 1;
            if self.missed_in_a_row >= ENOUGH_MISSED_CHANCES {
                self.disarmed = Some(NEVER_A_COPY);
                tracing::info!(
                    target: "echo::asr",
                    examined = self.examined,
                    chances = self.missed_in_a_row,
                    "the computer's audio was clearly audible and the microphone never \
                     carried a copy of it, so the rest of this recording is read back \
                     without looking for one"
                );
            }
        }
        Verdict::Pass
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asr::catchup::SpeechStream;
    use crate::audio::bleed::tests::{delayed, speech};
    use crate::audio::bleed::{examine, BLEED_CORRELATION};
    use std::path::Path;
    use std::sync::atomic::{AtomicU64, Ordering};

    const RATE: usize = TARGET_SAMPLE_RATE as usize;

    /// A meeting on disk, generated rather than stored: the far side is one
    /// minute of speech-shaped audio tiled for as long as the test needs, and
    /// the microphone is that same audio 180 ms late and 25 dB down — a laptop
    /// speaker across a desk.
    ///
    /// Tiled at a minute, which is sixty times the whole lag search, so the
    /// repetition can never put a false alignment inside the window the
    /// correlator looks in.
    struct Meeting {
        far_side: Vec<f32>,
        /// Set for the far side to be silent from here on, so that a stretch
        /// can have no chance to match at all.
        quiet_from_ms: i64,
        reads: AtomicU64,
    }

    /// This is *not* [`SystemPager::reads`]: it counts what reached the audio
    /// source, so a test can tell a pager that answered from a held page apart
    /// from one that went back to the source.
    impl Meeting {
        fn new(seed: u64) -> Self {
            Self {
                far_side: speech(60_000, seed),
                quiet_from_ms: i64::MAX,
                reads: AtomicU64::new(0),
            }
        }

        fn quiet_from(mut self, t_ms: i64) -> Self {
            self.quiet_from_ms = t_ms;
            self
        }

        fn reads(&self) -> u64 {
            self.reads.load(Ordering::SeqCst)
        }

        /// The far side at one moment of the meeting clock.
        fn far_at(&self, t_ms: i64, sample: usize) -> f32 {
            if t_ms >= self.quiet_from_ms {
                return 0.0;
            }
            let at = (t_ms.rem_euclid(60_000) as usize * RATE) / 1_000 + sample;
            self.far_side[at % self.far_side.len()]
        }

        /// One span of the far side, as the pager would read it.
        fn far_span(&self, from_ms: i64, to_ms: i64) -> Vec<f32> {
            let count = ((to_ms - from_ms).max(0) as usize * RATE) / 1_000;
            (0..count)
                .map(|i| {
                    let at_ms = from_ms + (i as i64 * 1_000) / RATE as i64;
                    self.far_at(at_ms, i % (RATE / 1_000))
                })
                .collect()
        }

        /// One stretch of the microphone: the far side, late and quiet.
        fn mic_utterance(&self, from_ms: i64, to_ms: i64) -> Utterance {
            let whole = self.far_span(from_ms - 1_000, to_ms);
            let late = delayed(&whole, 180, 0.056);
            let skip = RATE;
            Utterance {
                channel: Channel::Mic,
                t_start_ms: from_ms,
                t_end_ms: to_ms,
                samples: late[skip..].to_vec(),
                voiced_ms: (to_ms - from_ms) / 2,
                ..Default::default()
            }
        }

        /// …and one that is somebody in the room, saying something of their
        /// own.
        fn own_utterance(&self, from_ms: i64, to_ms: i64, seed: u64) -> Utterance {
            Utterance {
                channel: Channel::Mic,
                t_start_ms: from_ms,
                t_end_ms: to_ms,
                samples: speech((to_ms - from_ms) as usize, seed),
                voiced_ms: (to_ms - from_ms) / 2,
                ..Default::default()
            }
        }
    }

    /// Nothing in these tests opens a detector; the far side is what the pager
    /// asks for and the only thing this has to answer.
    struct NoStream;

    impl SpeechStream for NoStream {
        async fn push(
            &mut self,
            _samples: Vec<f32>,
            _t_start_ms: i64,
        ) -> Result<Vec<Utterance>, AsrError> {
            Ok(Vec::new())
        }

        async fn finish(&mut self) -> Result<Vec<Utterance>, AsrError> {
            Ok(Vec::new())
        }
    }

    impl AudioSource for Meeting {
        type Stream = NoStream;

        async fn read_window(
            &self,
            _chunks: &[ChunkRef],
            from_ms: i64,
            to_ms: i64,
        ) -> Result<Vec<f32>, AsrError> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            Ok(self.far_span(from_ms, to_ms))
        }

        fn open_stream(&self, _detector: Option<&Path>, _channel: Channel) -> NoStream {
            NoStream
        }
    }

    /// The far side's chunks, as the plan hands them over: `minutes` minutes of
    /// thirty-second chunks from t=0.
    fn far_side_chunks(minutes: i64) -> Vec<ChunkRef> {
        (0..minutes * 2)
            .map(|seq| {
                ChunkRef::new(
                    format!("/audio/system-{seq:06}.flac"),
                    Channel::System,
                    seq * 30_000,
                    (seq + 1) * 30_000,
                )
            })
            .collect()
    }

    // -------------------------------------------------------------------
    // The pager
    // -------------------------------------------------------------------

    /// The reason this module has a pager at all: consecutive utterances live
    /// in the same thirty seconds of meeting, and the far side under them is
    /// read once.
    #[tokio::test]
    async fn overlapping_stretches_are_served_from_the_pages_already_held() {
        let meeting = Meeting::new(11);
        let mut bleed = OfflineBleed::over(&far_side_chunks(2)).expect("a far side");

        // Four stretches across three pages, each wanting a six-second window
        // of the far side that overlaps the last one's or sits beside it. The
        // second and the fourth straddle a page boundary, which is what the
        // second held page is for.
        for (from, to) in [
            (34_000, 38_000),
            (58_000, 62_000),
            (62_500, 66_500),
            (88_000, 92_000),
        ] {
            assert!(
                bleed
                    .suppresses(&meeting, &meeting.mic_utterance(from, to))
                    .await,
                "the stretch at {from} ms is the far side, late and quiet"
            );
        }
        assert_eq!(
            meeting.reads(),
            3,
            "33.2 s to 92.2 s is three pages of the recording; nothing may be \
             read twice"
        );
        assert_eq!(bleed.reads(), meeting.reads(), "every read is the pager's");

        // …and going back over seconds a held page still covers costs nothing
        // at all: the page the last stretch came out of, and the one before it.
        let before = meeting.reads();
        for (from, to) in [(88_000, 92_000), (62_500, 66_500)] {
            assert!(
                bleed
                    .suppresses(&meeting, &meeting.mic_utterance(from, to))
                    .await
            );
        }
        assert_eq!(meeting.reads(), before);
    }

    /// The measurement the whole design rests on: an hour of meeting is about
    /// one read per thirty seconds of it, however many utterances there are.
    ///
    /// Six hundred stretches over the hour, which is what a real two-sided
    /// meeting produces. Without the pager this would be six hundred reads of
    /// twenty-five seconds each — the failure this module was written to avoid.
    #[tokio::test]
    async fn an_hour_of_meeting_is_read_once() {
        let meeting = Meeting::new(23);
        let mut bleed = OfflineBleed::over(&far_side_chunks(60)).expect("a far side");
        let mut suppressed = 0;
        for i in 0..600i64 {
            let from = 1_000 + i * 6_000;
            if bleed
                .suppresses(&meeting, &meeting.mic_utterance(from, from + 3_200))
                .await
            {
                suppressed += 1;
            }
        }
        println!(
            "an hour, 600 stretches: {} reads, {suppressed} suppressed",
            meeting.reads()
        );
        assert!(
            meeting.reads() <= 125,
            "an hour is 120 pages; {} reads means pages are being fetched twice",
            meeting.reads()
        );
        assert!(
            suppressed > 550,
            "only {suppressed} of 600 copies were recognised"
        );
    }

    /// A stretch the pager cannot cover in full is **left alone**, not judged
    /// on the part it could cover.
    ///
    /// This is commit 2's lesson in its offline form. The microphone runs on
    /// after the computer's audio stops — the person carries on talking after
    /// the call ends, the system channel died mid-meeting — and the last
    /// stretches want seconds of far side that this recording simply does not
    /// have. Judging them against what *is* there would let the own-voice veto
    /// vouch for seconds it was never shown.
    #[tokio::test]
    async fn a_stretch_the_far_side_does_not_reach_is_never_judged() {
        let meeting = Meeting::new(31);
        // The far side stops after one minute; the microphone kept recording.
        let mut bleed = OfflineBleed::over(&far_side_chunks(1)).expect("a far side");

        // Genuinely a copy, and genuinely unanswerable: the window it needs
        // runs 200 ms past the end of the far side.
        let over_the_end = meeting.mic_utterance(55_000, 60_000);
        assert_eq!(
            bleed.judge(&meeting, &over_the_end).await,
            Verdict::Undecided(NOT_ON_HAND)
        );
        // …and the same audio, a few seconds earlier, is recognised — so the
        // refusal above is about the reach of the recording and not about the
        // stretch being unrecognisable.
        assert!(matches!(
            bleed
                .judge(&meeting, &meeting.mic_utterance(50_000, 55_000))
                .await,
            Verdict::Bleed(_)
        ));

        // The other end of the recording: the very first stretch wants 800 ms
        // of far side from before the meeting began.
        let mut bleed = OfflineBleed::over(&far_side_chunks(1)).expect("a far side");
        assert_eq!(
            bleed
                .judge(&meeting, &meeting.mic_utterance(500, 5_000))
                .await,
            Verdict::Undecided(NOT_ON_HAND)
        );
    }

    // -------------------------------------------------------------------
    // The self-calibrating gate
    // -------------------------------------------------------------------

    /// Headphones, or a machine whose delay is outside the search: the far side
    /// is audible under every stretch and not one of them is a copy. After ten
    /// such chances the pass stops looking — and not before ten.
    #[tokio::test]
    async fn ten_missed_chances_stop_the_pass_looking() {
        let meeting = Meeting::new(41);
        let mut bleed = OfflineBleed::over(&far_side_chunks(60)).expect("a far side");

        for i in 0..ENOUGH_MISSED_CHANCES as i64 - 1 {
            let from = 5_000 + i * 6_000;
            let mine = meeting.own_utterance(from, from + 4_000, 900 + i as u64);
            assert_eq!(bleed.judge(&meeting, &mine).await, Verdict::Pass);
            assert!(bleed.armed(), "disarmed after {} chances", i + 1);
        }
        let last = meeting.own_utterance(5_000 + 9 * 6_000, 9_000 + 9 * 6_000, 999);
        assert_eq!(bleed.judge(&meeting, &last).await, Verdict::Pass);
        assert!(!bleed.armed(), "ten missed chances and still looking");

        // …and once it has stopped, it costs nothing at all: no read, no
        // correlation, and every stretch written down as it stands — including
        // one that really is a copy, which is the price of the gate and is
        // exactly what the live route gate costs on headphones.
        let quiet = meeting.reads();
        assert_eq!(
            bleed
                .judge(&meeting, &meeting.mic_utterance(90_000, 95_000))
                .await,
            Verdict::Undecided(NEVER_A_COPY)
        );
        assert_eq!(meeting.reads(), quiet, "a disarmed pass went to disk");
    }

    /// A meeting where the other side barely speaks is not evidence about
    /// anything, so it never disarms the pass however many stretches miss.
    #[tokio::test]
    async fn silence_on_the_far_side_is_not_a_missed_chance() {
        let meeting = Meeting::new(43).quiet_from(0);
        let mut bleed = OfflineBleed::over(&far_side_chunks(60)).expect("a far side");
        for i in 0..(ENOUGH_MISSED_CHANCES as i64 * 2) {
            let from = 5_000 + i * 6_000;
            let mine = meeting.own_utterance(from, from + 4_000, 700 + i as u64);
            assert_eq!(bleed.judge(&meeting, &mine).await, Verdict::Pass);
        }
        assert!(
            bleed.armed(),
            "a meeting with nothing to be a copy of stopped the pass looking"
        );
    }

    /// One real hit resets the streak: a meeting that has proved the path
    /// exists is not disarmed by a quiet patch in the middle of it.
    #[tokio::test]
    async fn a_single_copy_keeps_the_pass_looking() {
        let meeting = Meeting::new(47);
        let mut bleed = OfflineBleed::over(&far_side_chunks(60)).expect("a far side");
        for round in 0..3i64 {
            for i in 0..ENOUGH_MISSED_CHANCES as i64 - 1 {
                let from = 5_000 + (round * 20 + i) * 6_000;
                let mine = meeting.own_utterance(from, from + 4_000, 300 + i as u64);
                bleed.judge(&meeting, &mine).await;
            }
            let from = 5_000 + (round * 20 + 15) * 6_000;
            assert!(matches!(
                bleed
                    .judge(&meeting, &meeting.mic_utterance(from, from + 5_000))
                    .await,
                Verdict::Bleed(_)
            ));
        }
        assert!(bleed.armed());
        assert_eq!(bleed.suppressed(), 3);
    }

    // -------------------------------------------------------------------
    // The predicate, offline
    // -------------------------------------------------------------------

    /// The mic-only meeting: no far side, no pager, no work.
    #[test]
    fn a_meeting_with_no_far_side_has_nothing_to_be_a_copy_of() {
        assert!(OfflineBleed::over(&[]).is_none());
        assert!(SystemPager::over(&[]).is_none());
        // A chunk with no width in it is no far side either.
        let empty = vec![ChunkRef::new("/audio/system-0.flac", Channel::System, 0, 0)];
        assert!(OfflineBleed::over(&empty).is_none());
    }

    /// Somebody in the room talking over the same seconds is not a copy, and
    /// the far side being audible throughout is what makes that a real test
    /// rather than a lucky one.
    #[tokio::test]
    async fn somebody_talking_in_the_room_is_written_down() {
        let meeting = Meeting::new(53);
        let mut bleed = OfflineBleed::over(&far_side_chunks(2)).expect("a far side");
        let mine = meeting.own_utterance(20_000, 26_000, 54);
        assert_eq!(bleed.judge(&meeting, &mine).await, Verdict::Pass);
        assert_eq!(bleed.suppressed(), 0);
    }

    /// A stretch too short for the correlation to mean anything says so, and
    /// never reaches the pager.
    #[tokio::test]
    async fn a_short_answer_is_too_short_to_judge() {
        let meeting = Meeting::new(57);
        let mut bleed = OfflineBleed::over(&far_side_chunks(2)).expect("a far side");
        assert_eq!(
            bleed
                .judge(&meeting, &meeting.mic_utterance(20_000, 22_000))
                .await,
            Verdict::Undecided(TOO_SHORT)
        );
        assert_eq!(
            meeting.reads(),
            0,
            "a stretch nothing can be said about was read"
        );
    }

    /// **Where the warm span floor is defended for an offline pass.**
    ///
    /// [`crate::audio::bleed::MIN_SPAN_WARM_MS`] was measured against a *narrow*
    /// search: 2.5 s over thirty-one delays, where chance clears the bar about
    /// as often as it does at 3 s over a hundred and one. This pass uses the
    /// shorter floor with the *wide* search, which is a combination that table
    /// does not have a row for — so it gets measured here rather than assumed.
    ///
    /// Same instrument as `bleed.rs`'s own null test: unrelated speakers, every
    /// chance the predicate can give them (the whole stretch claimed as voice),
    /// and the only thing that matters asserted — that chance does not delete a
    /// sentence.
    ///
    /// Measured, a thousand pairs, against the two arms `bleed.rs` already has:
    ///
    /// ```text
    ///                          max     over 0.70    suppressed
    ///   cold, 3000 ms         0.69       0.0 %         0/1000
    ///   warm, 2500 ms         0.71       0.03 %        0/1000
    ///   cold, 2500 ms         0.78       0.2 %         0/1000   <- this pass
    /// ```
    ///
    /// So the combination is genuinely the weakest of the three on the
    /// correlation alone — a wide search over a short stretch is the worst of
    /// both — and the assertion is deliberately not about that column. Chance
    /// reaching the correlation bar is survivable, because the coverage
    /// condition and the own-voice veto still have to agree; chance deleting a
    /// sentence is not, and it did not happen in a thousand draws. If it ever
    /// does, the answer is to judge offline at [`MIN_SPAN_COLD_MS`] and give up
    /// the short stretches, which costs duplicated lines and nothing else.
    #[test]
    fn chance_never_clears_the_bar_at_the_warm_floor_with_a_cold_search() {
        const PAIRS: usize = 1_000;
        const POOL: usize = 64;
        let span_ms = MIN_SPAN_WARM_MS;
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
        let mut over = 0usize;
        let mut suppressed = 0usize;
        let mut worst = 0.0f32;
        for pair in 0..PAIRS {
            let mic = &mics[pair % POOL];
            let system = &systems[(pair / POOL + pair * 7 + 1) % POOL];
            let ev = examine(mic, system, LAG_MAX_MS, LagSearch::cold());
            worst = worst.max(ev.correlation);
            if ev.correlation >= BLEED_CORRELATION {
                over += 1;
            }
            // Warm, because the warm floor is the thing being measured.
            if is_bleed(&ev, span_ms, true) {
                suppressed += 1;
            }
        }
        println!(
            "unrelated speech at the warm floor ({span_ms} ms) over the whole cold \
             search: {PAIRS} pairs, max {worst:.3}, {over} reached the \
             {BLEED_CORRELATION:.2} bar, {suppressed} were suppressed"
        );
        assert_eq!(
            suppressed, 0,
            "chance deleted {suppressed} of {PAIRS} unrelated stretches at the \
             floor this pass judges at"
        );
    }
}
