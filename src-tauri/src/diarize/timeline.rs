//! Half-open time spans on the meeting clock, and the arithmetic the speaker
//! pass does with them.
//!
//! A span is `(start_ms, end_ms)` with `start < end`. Everything here is pure
//! and cheap; it is the part of diarization that has to be exactly right,
//! because a one-span mistake shows up as a wrong name against a sentence.

/// Half-open `[start, end)` on the monotonic meeting clock.
pub type Span = (i64, i64);

/// Length of a span, never negative.
pub fn span_ms(s: Span) -> i64 {
    (s.1 - s.0).max(0)
}

/// Total covered time. Assumes the spans are already merged; overlapping spans
/// would be double-counted.
pub fn total_ms(spans: &[Span]) -> i64 {
    spans.iter().copied().map(span_ms).sum()
}

/// Milliseconds two spans share.
pub fn overlap_ms(a: Span, b: Span) -> i64 {
    (a.1.min(b.1) - a.0.max(b.0)).max(0)
}

/// Gap between two spans, 0 when they touch or overlap.
pub fn gap_ms(a: Span, b: Span) -> i64 {
    if overlap_ms(a, b) > 0 {
        0
    } else if a.1 <= b.0 {
        b.0 - a.1
    } else {
        a.0 - b.1
    }
}

/// Sort, then fuse spans that overlap or sit within `max_gap_ms` of each other.
pub fn merge_spans(spans: &mut Vec<Span>, max_gap_ms: i64) {
    if spans.len() < 2 {
        return;
    }
    spans.sort_unstable();
    let mut out: Vec<Span> = Vec::with_capacity(spans.len());
    for s in spans.drain(..) {
        match out.last_mut() {
            Some(last) if s.0 - last.1 <= max_gap_ms => last.1 = last.1.max(s.1),
            _ => out.push(s),
        }
    }
    *spans = out;
}

/// Drop spans shorter than `min_ms`.
pub fn drop_short(spans: Vec<Span>, min_ms: i64) -> Vec<Span> {
    spans
        .into_iter()
        .filter(|s| span_ms(*s) >= min_ms)
        .collect()
}

/// `spans` minus `holes`. Used to find the parts of a speech turn where nobody
/// else is talking, which is the only audio worth fingerprinting.
///
/// Both inputs must be sorted and non-overlapping within themselves.
pub fn subtract(spans: &[Span], holes: &[Span]) -> Vec<Span> {
    let mut out = Vec::new();
    for &(start, end) in spans {
        let mut cursor = start;
        for &(hs, he) in holes {
            if he <= cursor {
                continue;
            }
            if hs >= end {
                break;
            }
            if hs > cursor {
                out.push((cursor, hs.min(end)));
            }
            cursor = cursor.max(he);
            if cursor >= end {
                break;
            }
        }
        if cursor < end {
            out.push((cursor, end));
        }
    }
    out.retain(|s| span_ms(*s) > 0);
    out
}

/// Union of several tracks, flattened into one merged track.
pub fn union(tracks: &[Vec<Span>], max_gap_ms: i64) -> Vec<Span> {
    let mut all: Vec<Span> = tracks.iter().flatten().copied().collect();
    merge_spans(&mut all, max_gap_ms);
    all
}

/// Which track a transcript segment belongs to, by how much time they share.
///
/// This is the step that turns speaker turns into `segments.speaker_id`. A
/// segment that touches two speakers goes to whoever holds more of it, which is
/// the right answer far more often than any tie-break on start time.
pub fn assign_by_overlap(segment: Span, tracks: &[Vec<Span>]) -> Option<usize> {
    let mut best: Option<(usize, i64)> = None;
    for (i, track) in tracks.iter().enumerate() {
        let shared: i64 = track.iter().map(|&s| overlap_ms(segment, s)).sum();
        if shared == 0 {
            continue;
        }
        match best {
            Some((_, most)) if shared <= most => {}
            _ => best = Some((i, shared)),
        }
    }
    best.map(|(i, _)| i)
}

/// Nearest track when nothing overlaps, within `tolerance_ms`.
///
/// Speech detection trims silence, so a short segment can fall in the gap
/// between two turns. Reaching for the closest voice is better than leaving the
/// line unattributed, but only if it is genuinely close.
pub fn nearest_track(segment: Span, tracks: &[Vec<Span>], tolerance_ms: i64) -> Option<usize> {
    let mut best: Option<(usize, i64)> = None;
    for (i, track) in tracks.iter().enumerate() {
        for &s in track {
            let d = gap_ms(segment, s);
            if d > tolerance_ms {
                continue;
            }
            match best {
                Some((_, nearest)) if d >= nearest => {}
                _ => best = Some((i, d)),
            }
        }
    }
    best.map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lengths_and_overlaps_are_half_open() {
        assert_eq!(span_ms((100, 400)), 300);
        assert_eq!(span_ms((400, 100)), 0);
        // Touching spans share nothing.
        assert_eq!(overlap_ms((0, 100), (100, 200)), 0);
        assert_eq!(overlap_ms((0, 100), (50, 200)), 50);
        assert_eq!(overlap_ms((0, 100), (200, 300)), 0);
        assert_eq!(gap_ms((0, 100), (250, 300)), 150);
        assert_eq!(gap_ms((250, 300), (0, 100)), 150);
        assert_eq!(gap_ms((0, 100), (50, 300)), 0);
    }

    #[test]
    fn merging_closes_gaps_up_to_the_limit_and_no_further() {
        let mut spans = vec![(0, 100), (150, 200), (600, 700)];
        merge_spans(&mut spans, 50);
        assert_eq!(spans, vec![(0, 200), (600, 700)]);

        let mut spans = vec![(0, 100), (150, 200)];
        merge_spans(&mut spans, 49);
        assert_eq!(spans, vec![(0, 100), (150, 200)]);
    }

    #[test]
    fn merging_handles_unsorted_and_nested_spans() {
        let mut spans = vec![(500, 600), (0, 1_000), (100, 200)];
        merge_spans(&mut spans, 0);
        assert_eq!(spans, vec![(0, 1_000)]);
    }

    #[test]
    fn subtracting_carves_out_the_parts_someone_else_is_talking_over() {
        let speech = vec![(0, 1_000)];
        let others = vec![(200, 300), (800, 1_200)];
        assert_eq!(subtract(&speech, &others), vec![(0, 200), (300, 800)]);
        // A hole that swallows the span leaves nothing.
        assert!(subtract(&speech, &[(0, 2_000)]).is_empty());
        // No holes leaves the span alone.
        assert_eq!(subtract(&speech, &[]), speech);
        // Holes outside the span do nothing.
        assert_eq!(subtract(&speech, &[(2_000, 3_000)]), speech);
    }

    #[test]
    fn subtracting_from_several_spans_keeps_them_separate() {
        let speech = vec![(0, 500), (1_000, 1_500)];
        let others = vec![(400, 1_100)];
        assert_eq!(subtract(&speech, &others), vec![(0, 400), (1_100, 1_500)]);
    }

    #[test]
    fn a_segment_goes_to_whoever_holds_more_of_it() {
        let tracks = vec![vec![(0, 1_000)], vec![(900, 3_000)]];
        // 0..1000 is all speaker 0.
        assert_eq!(assign_by_overlap((0, 1_000), &tracks), Some(0));
        // 950..3000 is mostly speaker 1.
        assert_eq!(assign_by_overlap((950, 3_000), &tracks), Some(1));
        // Nothing overlaps.
        assert_eq!(assign_by_overlap((5_000, 6_000), &tracks), None);
    }

    #[test]
    fn overlap_sums_across_a_speakers_whole_track() {
        // Speaker 0 has two short turns inside the segment, speaker 1 one
        // slightly shorter turn. The sum has to decide it.
        let tracks = vec![vec![(0, 200), (400, 600)], vec![(700, 1_000)]];
        assert_eq!(assign_by_overlap((0, 1_000), &tracks), Some(0));
    }

    #[test]
    fn the_first_track_wins_a_tie_so_the_result_is_deterministic() {
        let tracks = vec![vec![(0, 500)], vec![(500, 1_000)]];
        assert_eq!(assign_by_overlap((0, 1_000), &tracks), Some(0));
    }

    #[test]
    fn a_segment_in_a_gap_reaches_for_the_closest_voice_but_not_far() {
        let tracks = vec![vec![(0, 1_000)], vec![(5_000, 6_000)]];
        assert_eq!(nearest_track((1_100, 1_200), &tracks, 250), Some(0));
        assert_eq!(nearest_track((4_800, 4_900), &tracks, 250), Some(1));
        assert_eq!(nearest_track((3_000, 3_100), &tracks, 250), None);
    }

    #[test]
    fn union_flattens_every_track_into_one_merged_run() {
        let tracks = vec![vec![(0, 200), (600, 700)], vec![(150, 400)]];
        assert_eq!(union(&tracks, 0), vec![(0, 400), (600, 700)]);
    }

    #[test]
    fn total_and_drop_short_agree_with_the_span_lengths() {
        let spans = vec![(0, 100), (500, 900)];
        assert_eq!(total_ms(&spans), 500);
        assert_eq!(drop_short(spans, 200), vec![(500, 900)]);
    }
}
