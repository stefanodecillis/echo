//! Which of the stretches Echo decided not to write down left a hole.
//!
//! The bleed guard ([`crate::audio::bleed`] live, [`crate::asr::catchup_bleed`]
//! offline) takes the microphone's copy of what this computer played out of the
//! transcript, so the far side is written down once instead of twice. Nearly
//! every time that is exactly right and **nothing is missing**: the same seconds
//! carry the far side's own words, read off the computer's side of the call. On
//! a meeting held over a loudspeaker the guard fires on most of what the far
//! side says, and a person told "Echo left 84 stretches out" would be told
//! something true about the machinery and false about their transcript.
//!
//! Sometimes it is wrong, and the guard says so with numbers rather than
//! hedging ([`crate::audio::bleed::OWN_VOICE_MS`]): an answer at a fifth of the
//! far side's level, spoken into a pause, is lost 48 times in 60. When that
//! happens the seconds end up with no words against them at all, and the
//! finished transcript is indistinguishable from one where the person never
//! spoke.
//!
//! So the question this module asks is not "what did the guard decide" — the
//! decisions are all on record ([`crate::db::repo::list_suppressed_spans`],
//! `migrations/0007_suppressed_spans.sql`) — but **which of those decisions left
//! the recording with seconds that have no words in them**. Those are the ones
//! worth a person's attention, they are rare, and each one is a place where
//! Echo is the only reason the transcript is silent.
//!
//! # What it cannot see, said plainly
//!
//! The guard's other measured loss mode is sustained double-talk: while the far
//! side never pauses, a person talking over it is lost 9 times in 60. In that
//! case the far side *is* being written down over the same seconds, so the hole
//! test below finds nothing and this module stays quiet about a real loss.
//! Nothing on disk can tell that case apart from a correct suppression — both
//! are "the far side had the floor and the microphone's copy went" — and
//! guessing would put a warning on the case the guard gets right almost always.
//! The repair for it is the same one named on screen: read the recording again.
//!
//! # What a hole is
//!
//! A stretch with **no line of transcript overlapping it at all** — final text
//! or the live guess that was kept over it. Any words at all touching those
//! seconds and the person has something to read there, which is the difference
//! between "this reads oddly" and "this reads as silence". The bar is the
//! strictest one available on purpose: it can only ever fail to report a hole,
//! never invent one, and a sentence about somebody's own transcript has to be
//! true before it is complete.

use crate::audio::bleed::MIN_MIC_VOICE_MS;
use crate::db::repo::{SuppressedSpan, SuppressionReason};
use crate::types::{LeftOutMoment, Segment};

/// Two decisions closer together than this are one hole, not two.
///
/// The guard decides per utterance, and one answer is often several utterances
/// with a breath between them. Counting those separately would say "three
/// moments" about one thing a person experienced once, and the count is the
/// claim this whole surface rests on.
///
/// The gap is [`MIN_MIC_VOICE_MS`] (1 s), the least voice the guard will call a
/// stretch at all — its own reasoning is that anything shorter "is not a
/// sentence anybody would notice twice in a transcript". A gap under that
/// cannot hold anything the guard itself would have counted, so nothing worth
/// naming separately fits between the two decisions.
pub const SAME_HOLE_GAP_MS: i64 = MIN_MIC_VOICE_MS;

/// The stretches of this meeting that have no words in them because Echo chose
/// not to write them down, in clock order.
///
/// `spans` are the decisions that still stand — a "listen again" withdraws
/// them ([`crate::db::repo::clear_transcript`]), and a withdrawn decision is
/// not hiding anything any more. `segments` is the whole transcript, live
/// guesses included: text on the screen is text, whoever wrote it.
///
/// Empty is the normal answer, and the screen says nothing at all for it. A
/// reassuring zero would be a claim about the double-talk case above that this
/// cannot make.
pub fn moments_left_out(spans: &[SuppressedSpan], segments: &[Segment]) -> Vec<LeftOutMoment> {
    // Only the reason whose sentence is written. The copy on screen says what
    // *bleed* means — the microphone's copy of this computer's own sound — and
    // a second reason arriving without its own sentence must not inherit that
    // one, so it is left out here rather than described wrongly there.
    let mut decided: Vec<(i64, i64)> = spans
        .iter()
        .filter(|s| s.reason == SuppressionReason::Bleed && s.t_end_ms > s.t_start_ms)
        .map(|s| (s.t_start_ms, s.t_end_ms))
        .collect();
    decided.sort_unstable();

    let mut merged: Vec<(i64, i64)> = Vec::new();
    for (start, end) in decided {
        match merged.last_mut() {
            Some(last) if start - last.1 < SAME_HOLE_GAP_MS => last.1 = last.1.max(end),
            _ => merged.push((start, end)),
        }
    }

    merged
        .into_iter()
        .filter(|&(start, end)| {
            !segments
                .iter()
                .any(|s| !s.text.trim().is_empty() && s.t_start_ms < end && s.t_end_ms > start)
        })
        .map(|(t_start_ms, t_end_ms)| LeftOutMoment {
            t_start_ms,
            t_end_ms,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::repo::DecidedBy;
    use crate::types::Channel;

    fn decision(t_start_ms: i64, t_end_ms: i64) -> SuppressedSpan {
        SuppressedSpan {
            channel: Channel::Mic,
            t_start_ms,
            t_end_ms,
            reason: SuppressionReason::Bleed,
            decided_by: DecidedBy::Live,
            correlation: Some(0.91),
            lag_ms: Some(210),
            system_voice_ms: Some(t_end_ms - t_start_ms),
        }
    }

    fn line(t_start_ms: i64, t_end_ms: i64, text: &str) -> Segment {
        Segment {
            t_start_ms,
            t_end_ms,
            channel: Channel::System,
            text: text.to_string(),
            is_final: true,
            ..Default::default()
        }
    }

    fn at(moments: &[LeftOutMoment]) -> Vec<(i64, i64)> {
        moments.iter().map(|m| (m.t_start_ms, m.t_end_ms)).collect()
    }

    /// The ordinary loudspeaker meeting: the guard fired a dozen times and the
    /// far side's own words are against every one of those seconds. Nothing was
    /// lost, so nothing is said.
    #[test]
    fn a_decision_the_transcript_covers_is_not_a_hole() {
        let spans: Vec<_> = (0..12)
            .map(|i| decision(i * 30_000, i * 30_000 + 4_000))
            .collect();
        let lines: Vec<_> = (0..12)
            .map(|i| line(i * 30_000 - 500, i * 30_000 + 5_000, "the far side, once"))
            .collect();
        assert!(moments_left_out(&spans, &lines).is_empty());
    }

    /// The measured loss mode: a soft answer into a far-side pause. Nothing else
    /// wrote those seconds down, and the transcript reads as silence.
    #[test]
    fn a_decision_nothing_wrote_down_is_a_hole() {
        let spans = [decision(60_000, 63_000)];
        let lines = [
            line(40_000, 50_000, "before"),
            line(70_000, 80_000, "after"),
        ];
        assert_eq!(at(&moments_left_out(&spans, &lines)), [(60_000, 63_000)]);
    }

    /// A line touching the stretch at all is text a person can read there.
    #[test]
    fn words_that_only_overlap_the_edge_still_count_as_words() {
        let spans = [decision(60_000, 63_000)];
        assert!(moments_left_out(&spans, &[line(62_800, 70_000, "…yes")]).is_empty());
        assert!(moments_left_out(&spans, &[line(50_000, 60_200, "as I was")]).is_empty());
        // Touching end-to-end is not overlapping: those seconds are still bare.
        assert_eq!(
            at(&moments_left_out(
                &spans,
                &[line(50_000, 60_000, "as I was")]
            )),
            [(60_000, 63_000)]
        );
    }

    /// An empty line is what silence read as nothing looks like in the table. It
    /// is not something to read.
    #[test]
    fn a_line_with_no_words_in_it_covers_nothing() {
        let spans = [decision(60_000, 63_000)];
        assert_eq!(
            at(&moments_left_out(&spans, &[line(59_000, 64_000, "   ")])),
            [(60_000, 63_000)]
        );
    }

    /// One answer, three utterances, breaths in between: one hole.
    #[test]
    fn decisions_a_breath_apart_are_one_moment() {
        let spans = [
            decision(60_000, 62_500),
            decision(62_900, 65_000),
            decision(65_000, 67_000),
        ];
        assert_eq!(at(&moments_left_out(&spans, &[])), [(60_000, 67_000)]);
    }

    /// A second of quiet is a break. Two things that happened separately are
    /// counted separately.
    #[test]
    fn decisions_a_full_second_apart_are_two_moments() {
        let spans = [decision(60_000, 62_000), decision(63_000, 65_000)];
        assert_eq!(
            at(&moments_left_out(&spans, &[])),
            [(60_000, 62_000), (63_000, 65_000)]
        );
    }

    /// Rows arrive in whatever order two passes wrote them, and one can sit
    /// inside another after a live and an offline decision meet.
    #[test]
    fn decisions_are_read_in_clock_order_whatever_order_they_arrive_in() {
        let spans = [
            decision(90_000, 93_000),
            decision(60_000, 70_000),
            decision(61_000, 62_000),
        ];
        assert_eq!(
            at(&moments_left_out(&spans, &[])),
            [(60_000, 70_000), (90_000, 93_000)]
        );
    }

    /// Nothing decided, nothing to say — and a stretch with no width was never
    /// a stretch of anybody's recording.
    #[test]
    fn nothing_decided_says_nothing() {
        assert!(moments_left_out(&[], &[line(0, 1_000, "hello")]).is_empty());
        assert!(moments_left_out(&[decision(60_000, 60_000)], &[]).is_empty());
    }
}
