//! When Echo is not sure it heard a line, and who gets told.
//!
//! The engine already reports a confidence per line, and Echo already uses it
//! three times over — [`crate::asr::catchup::HowThisMeetingReads`] retries a
//! window that reads far worse than the rest of the meeting,
//! `diarize::sample` ranks voice samples by it, and
//! [`crate::asr::language`] refuses to vote with a reading below
//! [`crate::asr::language::MIN_USABLE_CONFIDENCE`]. What it has never done is
//! tell **a person**, or tell the thing that writes the recap. Both read the
//! transcript as if every line were equally solid.
//!
//! This module is the one bar those two surfaces ask. It is deliberately a
//! copy of the reasoning already argued in `catchup.rs` rather than a new
//! theory of confidence — with one arm added, because that reasoning has a hole
//! this use has to cover.
//!
//! # Why two arms
//!
//! **Relative alone is not enough.** `catchup.rs` retries anything below
//! `0.75 ×` the meeting's median, and its own doc comment writes down the catch:
//! "when a meeting is read wrongly *throughout*, the median is wrong too and
//! nothing stands out" — the 2026-08-24 meeting, which read at 0.609 from end to
//! end and was garbage from end to end. A purely relative bar marks almost
//! nothing on exactly the meeting a person most needs warning about, because
//! every line agrees with every other line about being wrong.
//!
//! **Absolute alone is not enough either.** A meeting that reads at 0.90 for
//! fifty-five minutes and 0.62 for five — somebody on a bad connection, a
//! stretch in another language, a name nobody in the room pronounces the same
//! way twice — never dips under any floor that leaves quiet-but-fine meetings
//! alone. That stretch is the one worth marking, and only its neighbours can
//! reveal it.
//!
//! So a line is shaky if **either** arm says so. They fail in opposite
//! directions and neither subsumes the other.
//!
//! # What this does not claim
//!
//! [`SHAKY_FLOOR`] is an interpolation between two observed meetings, not a
//! measurement: the 2026-08-24 Danish reading at 0.609 was nonsense, the Italian
//! control at 0.761 was fine, and `language::MIN_USABLE_CONFIDENCE` (0.35)
//! already marks the point below which a reading "carries no information".
//! 0.55 sits between the two ends and below the bad meeting.
//!
//! And that is the limit, stated plainly rather than papered over: **a meeting
//! read wrongly throughout at 0.609 has plenty of lines above 0.55 that are
//! still wrong, and nothing here finds them.** Its median is 0.609 too, so the
//! relative arm stays quiet, and the floor sits below most of it. The repair for
//! that shape is still the one `catchup.rs` names —
//! [`crate::db::repo::clear_transcript`] dropping the language so the next pass
//! detects afresh. This marks the lines it can honestly point at, and says
//! nothing about the rest. An unmarked line is not a promise that it is right;
//! it is the absence of a reason to doubt it.

use crate::types::Segment;

/// Below this, a line is shaky whatever the rest of the meeting looks like.
///
/// See the module comment for where the number comes from: above
/// [`crate::asr::language::MIN_USABLE_CONFIDENCE`] (0.35, "carries no
/// information"), below the 0.609 meeting that was nonsense, well below the
/// 0.761 meeting that was fine.
pub const SHAKY_FLOOR: f32 = 0.55;

/// How far short of the meeting's own middle a line has to fall before it is
/// shaky, when the meeting has a middle worth comparing against.
///
/// The same 0.75 that [`crate::asr::catchup`] already retries a window at. It
/// is deliberately the same number: this is the same question — "is this line
/// out of step with how this meeting reads?" — asked by a different caller, and
/// two different answers to one question would be a bug waiting to be argued
/// about.
pub const SHAKY_BELOW_MEDIAN: f32 = 0.75;

/// Measured lines needed before the meeting has a middle at all.
///
/// Also the same as `catchup`'s. Below this, one unlucky line drags the median
/// down far enough to excuse itself, and the relative arm becomes noise; the
/// floor still applies, so a short meeting is not unprotected, only unjudged
/// against itself.
pub const MEDIAN_NEEDS: usize = 8;

/// How sure the engine was about this meeting, taken as a whole.
///
/// Built once per meeting and asked once per line. Cheap to build (a sort) and
/// free to ask, which is what lets both the screen and the recap use the same
/// bar instead of each inventing one.
#[derive(Debug, Default, Clone)]
pub struct HowSureThisMeetingIs {
    /// The middle of the measured lines, once there are [`MEDIAN_NEEDS`] of
    /// them. `None` means "this meeting has no normal yet", not "this meeting
    /// is fine".
    median: Option<f32>,
}

impl HowSureThisMeetingIs {
    /// Read the meeting off its finished segments.
    ///
    /// Two exclusions, both inherited from
    /// [`crate::asr::catchup::HowThisMeetingReads`]'s `saw`:
    ///
    /// * a line with no text is not the engine being unsure, it is silence, and
    ///   counting it would drag the bar down until nothing could fall below it;
    /// * a line with no confidence contributes nothing — see [`Self::is_shaky`]
    ///   for why "nobody measured it" is never evidence in either direction.
    pub fn of(segments: &[Segment]) -> Self {
        Self::from_readings(segments.iter().filter_map(|s| {
            if s.text.trim().is_empty() {
                None
            } else {
                s.avg_confidence
            }
        }))
    }

    /// The same thing from bare numbers, for callers that have already thrown
    /// the empty lines away (and for tests).
    pub fn from_readings(readings: impl IntoIterator<Item = f32>) -> Self {
        let mut measured: Vec<f32> = readings.into_iter().collect();
        if measured.len() < MEDIAN_NEEDS {
            return Self { median: None };
        }
        measured.sort_by(|a, b| a.total_cmp(b));
        let mid = measured.len() / 2;
        let median = if measured.len().is_multiple_of(2) {
            (measured[mid - 1] + measured[mid]) / 2.0
        } else {
            measured[mid]
        };
        Self {
            median: Some(median),
        }
    }

    /// The middle of this meeting, when it has one. Exposed so a log line or a
    /// test can say what the bar actually was.
    pub fn median(&self) -> Option<f32> {
        self.median
    }

    /// Was Echo unsure it heard this line?
    ///
    /// Either arm is enough: under [`SHAKY_FLOOR`] outright, or under
    /// [`SHAKY_BELOW_MEDIAN`] of how this meeting normally reads.
    ///
    /// **`is_shaky(None)` is always false**, and that is a rule rather than a
    /// convenience. A missing confidence means nothing measured this line — an
    /// engine that does not report one, an older row written before it was
    /// stored. Marking it would be Echo saying "I was unsure" when what happened
    /// is that nobody asked. This is the same principle as
    /// [`crate::audio::vad::Utterance::measured_voice_ms`]: nothing measured it,
    /// so nothing may be claimed because of it. `catchup`'s `far_below` takes
    /// the identical line.
    pub fn is_shaky(&self, confidence: Option<f32>) -> bool {
        let Some(c) = confidence else {
            return false;
        };
        if c < SHAKY_FLOOR {
            return true;
        }
        match self.median {
            Some(median) => c < median * SHAKY_BELOW_MEDIAN,
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(text: &str, confidence: Option<f32>) -> Segment {
        Segment {
            text: text.to_string(),
            avg_confidence: confidence,
            ..Default::default()
        }
    }

    #[test]
    fn a_bad_line_is_shaky_on_its_own_merits() {
        // A meeting that reads well throughout, with one line that does not.
        // The floor catches it without the median needing to be consulted.
        let sure = HowSureThisMeetingIs::from_readings([0.9; 12]);
        assert!(sure.is_shaky(Some(0.40)));
        assert!(sure.is_shaky(Some(0.54)));
        assert!(!sure.is_shaky(Some(0.90)));
    }

    #[test]
    fn the_floor_applies_before_the_meeting_has_a_middle() {
        // Fewer measured lines than MEDIAN_NEEDS: no median, so the relative arm
        // is silent — but a line the engine plainly could not read is still
        // marked. A short meeting is unjudged against itself, not unprotected.
        let sure = HowSureThisMeetingIs::from_readings([0.9; MEDIAN_NEEDS - 1]);
        assert_eq!(sure.median(), None);
        assert!(sure.is_shaky(Some(0.30)));
        assert!(!sure.is_shaky(Some(0.60)));
    }

    #[test]
    fn a_line_far_below_this_meeting_is_shaky_even_though_it_clears_the_floor() {
        // The case the floor cannot see: a meeting that reads at 0.90 with a
        // stretch at 0.62. 0.62 is comfortably above SHAKY_FLOOR and just as
        // comfortably out of step with its neighbours.
        let sure = HowSureThisMeetingIs::from_readings([0.9; 20]);
        assert_eq!(sure.median(), Some(0.9));
        const { assert!(0.62 > SHAKY_FLOOR) };
        assert!(sure.is_shaky(Some(0.62)));
        // ...and a line only slightly below the middle is not marked. Being
        // below average is not the same as being wrong.
        assert!(!sure.is_shaky(Some(0.80)));
    }

    #[test]
    fn nothing_measured_is_never_marked() {
        // Both shapes: with a median and without one. A line nobody measured is
        // never evidence that Echo was unsure — see `is_shaky`.
        let judged = HowSureThisMeetingIs::from_readings([0.9; 20]);
        let unjudged = HowSureThisMeetingIs::default();
        assert!(!judged.is_shaky(None));
        assert!(!unjudged.is_shaky(None));
        // And an unmeasured line does not move the meeting's own middle either.
        let segments = vec![line("measured", Some(0.9)), line("not measured", None)];
        assert!(HowSureThisMeetingIs::of(&segments).median().is_none());
    }

    #[test]
    fn empty_lines_stay_out_of_the_meetings_own_middle() {
        // Silence read as nothing is not the engine being unsure. If those rows
        // counted, a meeting with long pauses would set a median low enough that
        // no real line could ever fall below it.
        let mut segments: Vec<Segment> =
            (0..MEDIAN_NEEDS).map(|_| line("real", Some(0.9))).collect();
        segments.extend((0..30).map(|_| line("   ", Some(0.1))));
        let sure = HowSureThisMeetingIs::of(&segments);
        assert_eq!(sure.median(), Some(0.9));
        assert!(sure.is_shaky(Some(0.62)));
    }

    #[test]
    fn a_meeting_read_wrongly_throughout_does_not_mark_every_line() {
        // The 2026-08-24 meeting: read as the wrong language from end to end, at
        // 0.609 from end to end. This is the limit the module comment writes
        // down rather than papers over — the median is 0.609 too, so nothing
        // stands out, and 0.609 clears the floor. Echo does not find this shape
        // here, and the test exists to state that out loud so nobody later
        // mistakes silence for a clean bill of health.
        let readings: Vec<f32> = (0..40).map(|i| 0.609 + (i % 5) as f32 * 0.01).collect();
        let sure = HowSureThisMeetingIs::from_readings(readings.iter().copied());
        let marked = readings.iter().filter(|c| sure.is_shaky(Some(**c))).count();
        assert_eq!(marked, 0, "the median moved with the meeting, as designed");
    }

    #[test]
    fn a_meeting_with_a_bad_stretch_marks_the_stretch_and_not_the_rest() {
        // The shape this is for, end to end: fifty-five good minutes and five
        // bad ones. Only the bad stretch is marked.
        let mut readings = vec![0.90_f32; 50];
        readings.extend([0.62_f32; 5]);
        let sure = HowSureThisMeetingIs::from_readings(readings.iter().copied());
        let marked: Vec<f32> = readings
            .iter()
            .copied()
            .filter(|c| sure.is_shaky(Some(*c)))
            .collect();
        assert_eq!(marked.len(), 5);
        assert!(marked.iter().all(|c| (*c - 0.62).abs() < f32::EPSILON));
    }

    #[test]
    fn the_bar_is_the_same_one_the_screen_draws() {
        // The transcript screen has to ask this question in the browser, over
        // segments it already has, so the bar is implemented twice. The mirror
        // lives in `src/pages/Meeting/lib/confidence.ts` and these three
        // constants are copied there by hand. If a number changes here it has to
        // change there in the same commit, or a line reads as shaky in the recap
        // and solid on screen.
        assert_eq!(SHAKY_FLOOR, 0.55);
        assert_eq!(SHAKY_BELOW_MEDIAN, 0.75);
        assert_eq!(MEDIAN_NEEDS, 8);
        // And the relative arm is deliberately the same number `catchup` retries
        // a window at — one question, one answer.
        assert_eq!(SHAKY_BELOW_MEDIAN, crate::asr::catchup::RETRY_BELOW_MEDIAN);
    }
}
