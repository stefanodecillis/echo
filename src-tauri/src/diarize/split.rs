//! Cutting a transcript line that holds two voices into one line per voice.
//!
//! ## The line this is for
//!
//! Fast dialogue defeats the transcript's own idea of a line. The speech engine
//! decides where a line ends from what the audio sounds like, and when two people
//! answer each other inside four hundred milliseconds it hears one continuous
//! stretch of speech and writes one line:
//!
//! ```text
//! [3:43] "No, io domani sera prendo l'aereo e vengo in Italia. Ah, ok. E domani…"
//! ```
//!
//! "Ah, ok." is the other person. Attribution then has one row to point at two
//! people, and [`super::timeline::assign_by_overlap`] gives the whole line to
//! whoever held more of it — so a line of dialogue turns up entirely on one
//! speaker, and the other one silently loses their words.
//!
//! The separation pass is the one place that can fix this, because it is the only
//! place that holds both the transcript and the speaker turns. So it cuts the
//! line where the turns say the voice changed.
//!
//! ## Where the words get cut, and the honest name for it
//!
//! The *time* of the cut is known well: it is a turn boundary the segmentation
//! model produced, on the same meeting clock the line is on.
//!
//! The *place in the text* is not known. Echo stores one row per whisper
//! sub-segment and nothing finer — no word or token times, because
//! `asr::engine` runs with `token_timestamps` off (they are experimental in
//! whisper.cpp and nothing else reads them). So a line's text has exactly two
//! known time points, its start and its end, and everything in between is
//! interpolation.
//!
//! This module therefore does the following, and calls it what it is:
//!
//! 1. Cut the *time* at the turn boundary — exact.
//! 2. Cut the *text* at the sentence-punctuation boundary nearest to where that
//!    time falls if the words are assumed to run at a constant rate — an
//!    **approximation**, and one that gets the common case right for a reason
//!    worth stating: a voice change nearly always happens at the end of a
//!    sentence, so the punctuation is usually sitting exactly where the cut
//!    belongs. In the line above the cut lands at "Italia. | Ah, ok. | E
//!    domani…", which is right.
//! 3. Refuse if the nearest boundary is further away than
//!    [`MAX_TEXT_DRIFT`] of the line, or if any of the gates below is not met.
//!    A refused split leaves the line exactly as it was: whole, on one speaker,
//!    which is today's behaviour and not a regression.
//!
//! What this cannot do is put the cut inside a sentence one person finished and
//! another interrupted. Nothing can, without word times. When word times arrive
//! this module keeps its shape and step 2 stops being an approximation.
//!
//! ## When it declines
//!
//! Splitting a line is destructive in a way attributing one is not — the old row
//! is deleted and two new ones take its place — so it happens only when the turns
//! are worth believing:
//!
//! * both voices hold at least [`MIN_SHARE`] of the line and at least
//!   [`MIN_PIECE_MS`] of it, so a hundred milliseconds of crosstalk never cuts
//!   anything;
//! * turns cover at least [`MIN_COVERAGE`] of the line, so a boundary is not
//!   being inferred from a line the segmentation mostly did not see;
//! * both voices were heard at [`MIN_TURN_CONFIDENCE`] mean activation or better,
//!   so a cut is not made out of two guesses;
//! * the engine's own confidence in the words is at least
//!   [`MIN_TEXT_CONFIDENCE`], because cutting text nobody trusts is moving noise
//!   from one speaker to another;
//! * the line is at least [`MIN_LINE_MS`] long and has at least
//!   [`MIN_WORDS_PER_PIECE`] words for each piece.
//!
//! Two of those gates are weaker than they read, and it is worth knowing which.
//! [`MIN_TURN_CONFIDENCE`] is checked against a voice's **mean activation over
//! the whole meeting**, because that is the number the clustering cut carries;
//! it catches "this voice was never heard clearly" and not "this particular
//! boundary was a guess". And coverage is measured against the turns as the
//! segmentation drew them, which trims silence — a line with a long pause in the
//! middle can fail it and be left whole. Both err towards leaving the transcript
//! alone, which is the direction to err in.

use super::timeline::{self, Span};

/// Share of a line one voice has to hold before the line is worth cutting.
pub const MIN_SHARE: f32 = 0.25;

/// Speech one voice has to hold inside a line before the line is worth cutting,
/// and the shortest piece a cut may leave behind.
pub const MIN_PIECE_MS: i64 = 1_500;

/// Lines shorter than this are left alone. Two people inside one second of
/// transcript is crosstalk, and cutting it produces two fragments of a sentence
/// rather than two sentences.
pub const MIN_LINE_MS: i64 = 3_000;

/// How much of a line the speaker turns have to cover before a boundary inside
/// it is worth believing.
pub const MIN_COVERAGE: f32 = 0.75;

/// Mean activation both voices need before their shared boundary is used to cut
/// words. Below this the segmentation was unsure who was talking, and an unsure
/// boundary is not a place to cut a sentence.
pub const MIN_TURN_CONFIDENCE: f32 = 0.5;

/// The engine's own mean confidence in the words, below which they are not
/// re-attributed by cutting them up. `None` — an older row with no confidence
/// recorded — passes: absence of a number is not evidence of a bad one.
pub const MIN_TEXT_CONFIDENCE: f32 = 0.3;

/// How far from the proportional point a cut may land, as a share of the line's
/// text. Beyond this the punctuation is not where the voice changed and the
/// honest answer is to leave the line whole.
pub const MAX_TEXT_DRIFT: f32 = 0.25;

/// Fewest words a piece may be left with. One word per speaker is not dialogue
/// worth rewriting a transcript for.
pub const MIN_WORDS_PER_PIECE: usize = 2;

/// One piece of a line that held two voices.
#[derive(Debug, Clone, PartialEq)]
pub struct Piece {
    /// Where the piece sits on the meeting clock. The pieces tile the line
    /// exactly: no gap, no overlap, first starts where the line did and last
    /// ends where it did.
    pub span: Span,
    pub text: String,
    /// Index into the tracks the cut was made from — the speaker this piece is.
    pub speaker: usize,
}

/// What the line's own row knows that decides whether it may be cut.
#[derive(Debug, Clone, Copy, Default)]
pub struct LineFacts {
    pub span: Span,
    /// The engine's mean confidence in these words, when it recorded one.
    pub text_confidence: Option<f32>,
}

/// Cut one line at its turn boundaries, or `None` to leave it exactly as it is.
///
/// `tracks` is the final per-person timeline and `confidence` the mean activation
/// behind each of them — both straight off the clustering cut, so the speaker
/// indices are the ones the caller will attribute with.
pub fn split_line(
    facts: LineFacts,
    text: &str,
    tracks: &[Vec<Span>],
    confidence: &[f32],
) -> Option<Vec<Piece>> {
    let line = facts.span;
    let line_ms = timeline::span_ms(line);
    if line_ms < MIN_LINE_MS || text.trim().is_empty() {
        return None;
    }
    if facts
        .text_confidence
        .is_some_and(|c| c < MIN_TEXT_CONFIDENCE)
    {
        return None;
    }

    // How much of the line the turns actually account for, measured *before*
    // anything is collapsed or stretched. The pieces are made to tile the line at
    // the end, so asking this question of them afterwards would always answer
    // "all of it" and the gate would be no gate at all.
    let owned = order_turns(line, tracks);
    let covered: i64 = owned.iter().map(|r| timeline::span_ms(r.span)).sum();
    if (covered as f32) < MIN_COVERAGE * line_ms as f32 {
        return None;
    }

    let runs = collapse(owned.clone(), line);
    if runs.len() < 2 {
        return None;
    }

    // Two voices, each with a real share of the line and each heard clearly.
    let mut held: Vec<(usize, i64)> = Vec::new();
    for run in owned
        .iter()
        .filter(|r| runs.iter().any(|k| k.speaker == r.speaker))
    {
        let ms = timeline::span_ms(run.span);
        match held.iter_mut().find(|(s, _)| *s == run.speaker) {
            Some((_, total)) => *total += ms,
            None => held.push((run.speaker, ms)),
        }
    }
    if held.len() < 2 {
        return None;
    }
    for (speaker, ms) in &held {
        if *ms < MIN_PIECE_MS || (*ms as f32) < MIN_SHARE * line_ms as f32 {
            return None;
        }
        if confidence.get(*speaker).copied().unwrap_or(0.0) < MIN_TURN_CONFIDENCE {
            return None;
        }
    }

    // The words, cut proportionally and snapped to punctuation. Refuses rather
    // than guesses; see the module docs.
    let weights: Vec<i64> = runs.iter().map(|r| timeline::span_ms(r.span)).collect();
    let parts = cut_text(text, &weights)?;

    Some(
        runs.iter()
            .zip(parts)
            .map(|(run, text)| Piece {
                span: run.span,
                text,
                speaker: run.speaker,
            })
            .collect(),
    )
}

/// One stretch of the line, and whose it is.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Run {
    span: Span,
    speaker: usize,
}

/// Who holds each elementary stretch of the line, in time order.
///
/// Overlap-aware decoding means two voices can claim the same millisecond. A cut
/// needs one owner per stretch, so a contested one goes to whoever also holds the
/// stretch before it — a voice does not stop and restart inside its own
/// sentence — and to the earlier speaker index when there is nothing before it.
fn order_turns(line: Span, tracks: &[Vec<Span>]) -> Vec<Run> {
    let mut edges: Vec<i64> = vec![line.0, line.1];
    for track in tracks {
        for &turn in track {
            for edge in [turn.0, turn.1] {
                if edge > line.0 && edge < line.1 {
                    edges.push(edge);
                }
            }
        }
    }
    edges.sort_unstable();
    edges.dedup();

    let mut out: Vec<Run> = Vec::new();
    for pair in edges.windows(2) {
        let piece: Span = (pair[0], pair[1]);
        if timeline::span_ms(piece) == 0 {
            continue;
        }
        let claimants: Vec<usize> = tracks
            .iter()
            .enumerate()
            .filter(|(_, track)| {
                track
                    .iter()
                    .any(|&turn| timeline::overlap_ms(turn, piece) > 0)
            })
            .map(|(i, _)| i)
            .collect();
        let owner = match claimants.len() {
            0 => continue,
            1 => claimants[0],
            _ => {
                let previous = out.last().map(|r| r.speaker);
                match previous.filter(|p| claimants.contains(p)) {
                    Some(p) => p,
                    None => claimants[0],
                }
            }
        };
        out.push(Run {
            span: piece,
            speaker: owner,
        });
    }
    out
}

/// Fuse neighbouring stretches of one voice, drop the slivers into whoever is
/// next to them, and stretch the result so it tiles the whole line.
///
/// A sliver is a run below [`MIN_PIECE_MS`]: it is not a piece anybody would
/// want as its own line of transcript, and leaving it in would put a cut inside
/// one person's sentence.
fn collapse(runs: Vec<Run>, line: Span) -> Vec<Run> {
    let mut merged: Vec<Run> = Vec::new();
    for run in runs {
        match merged.last_mut() {
            Some(last) if last.speaker == run.speaker => last.span.1 = run.span.1,
            _ => merged.push(run),
        }
    }

    // Slivers go, smallest first, into the longer of their neighbours.
    loop {
        if merged.len() < 2 {
            break;
        }
        let Some(at) = (0..merged.len())
            .filter(|&i| timeline::span_ms(merged[i].span) < MIN_PIECE_MS)
            .min_by_key(|&i| timeline::span_ms(merged[i].span))
        else {
            break;
        };
        let before = at.checked_sub(1);
        let after = (at + 1 < merged.len()).then_some(at + 1);
        let host = match (before, after) {
            (Some(b), Some(a)) => {
                if timeline::span_ms(merged[b].span) >= timeline::span_ms(merged[a].span) {
                    b
                } else {
                    a
                }
            }
            (Some(b), None) => b,
            (None, Some(a)) => a,
            (None, None) => break,
        };
        let gone = merged.remove(at);
        let host = if host > at { host - 1 } else { host };
        merged[host].span.0 = merged[host].span.0.min(gone.span.0);
        merged[host].span.1 = merged[host].span.1.max(gone.span.1);
        // Removing a run can leave two runs of one voice side by side.
        let mut fused: Vec<Run> = Vec::with_capacity(merged.len());
        for run in merged.drain(..) {
            match fused.last_mut() {
                Some(last) if last.speaker == run.speaker => last.span.1 = run.span.1,
                _ => fused.push(run),
            }
        }
        merged = fused;
    }

    // Tile the line: the transcript view has no room for a gap between two
    // halves of what used to be one row, and the pieces have to add back up to
    // the line they replace.
    if let Some(first) = merged.first_mut() {
        first.span.0 = line.0;
    }
    if let Some(last) = merged.last_mut() {
        last.span.1 = line.1;
    }
    for i in 1..merged.len() {
        let seam = merged[i].span.0.min(merged[i - 1].span.1);
        merged[i - 1].span.1 = seam;
        merged[i].span.0 = seam;
    }
    merged.retain(|r| timeline::span_ms(r.span) > 0);
    merged
}

/// A place the text could be cut, and how good a place it is.
#[derive(Debug, Clone, Copy)]
struct Boundary {
    /// Characters before the cut.
    at: usize,
    /// 0 = end of a sentence, 1 = end of a clause, 2 = between two words.
    rank: u8,
}

/// Cut `text` into `weights.len()` pieces whose lengths follow `weights`,
/// snapping each cut to the best boundary near where the time says it falls.
///
/// `None` when there is no honest place to cut: no boundary near enough
/// ([`MAX_TEXT_DRIFT`]), or a piece that would come out empty or shorter than
/// [`MIN_WORDS_PER_PIECE`] words.
fn cut_text(text: &str, weights: &[i64]) -> Option<Vec<String>> {
    let text = text.trim();
    if weights.len() < 2 {
        return None;
    }
    let chars: Vec<char> = text.chars().collect();
    let total_chars = chars.len();
    if total_chars == 0 {
        return None;
    }
    if text.split_whitespace().count() < MIN_WORDS_PER_PIECE * weights.len() {
        return None;
    }

    let boundaries = boundaries(&chars);
    let total_ms: i64 = weights.iter().sum();
    if total_ms <= 0 {
        return None;
    }
    let tolerance = ((total_chars as f32) * MAX_TEXT_DRIFT).ceil() as usize;

    let mut cuts: Vec<usize> = Vec::with_capacity(weights.len() - 1);
    let mut running = 0i64;
    for w in &weights[..weights.len() - 1] {
        running += *w;
        let target = ((running as f64 / total_ms as f64) * total_chars as f64).round() as usize;
        let floor = cuts.last().copied().unwrap_or(0) + 1;
        let cut = nearest(&boundaries, target, tolerance, floor, total_chars)?;
        cuts.push(cut);
    }

    let mut pieces: Vec<String> = Vec::with_capacity(weights.len());
    let mut from = 0usize;
    for to in cuts.iter().copied().chain(std::iter::once(total_chars)) {
        let piece: String = chars[from..to].iter().collect();
        let piece = piece.trim().to_string();
        if piece.split_whitespace().count() < MIN_WORDS_PER_PIECE {
            return None;
        }
        pieces.push(piece);
        from = to;
    }
    Some(pieces)
}

/// Every place the text could be cut, best first by rank.
fn boundaries(chars: &[char]) -> Vec<Boundary> {
    let mut out: Vec<Boundary> = Vec::new();
    for (i, c) in chars.iter().enumerate() {
        if !c.is_whitespace() {
            continue;
        }
        // The cut goes after the run of whitespace, so no piece starts with a
        // space and none ends with one.
        let mut at = i + 1;
        while chars.get(at).is_some_and(|c| c.is_whitespace()) {
            at += 1;
        }
        if at >= chars.len() {
            continue;
        }
        // What the whitespace follows decides how good a cut it is. Closing
        // quotes and brackets sit between the punctuation and the space.
        let ends_sentence = chars[..i]
            .iter()
            .rev()
            .find(|c| !matches!(c, '"' | '\'' | '»' | '”' | '’' | ')' | ']'))
            .copied();
        let rank = match ends_sentence {
            Some('.' | '!' | '?' | '…' | ';') => 0,
            Some(',' | ':' | '—' | '–') => 1,
            _ => 2,
        };
        out.push(Boundary { at, rank });
    }
    out
}

/// The boundary nearest `target`, preferring a sentence end to a clause end and
/// a clause end to a word gap — but only among boundaries close enough to be the
/// place the voice actually changed.
fn nearest(
    boundaries: &[Boundary],
    target: usize,
    tolerance: usize,
    floor: usize,
    ceiling: usize,
) -> Option<usize> {
    let reach = |b: &Boundary| target.abs_diff(b.at);
    for rank in 0..=2u8 {
        let best = boundaries
            .iter()
            .filter(|b| b.rank == rank && b.at >= floor && b.at < ceiling && reach(b) <= tolerance)
            .min_by_key(|b| reach(b));
        if let Some(b) = best {
            return Some(b.at);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The field line from the module docs, with the turns the pass found under
    /// it: one voice, four hundred milliseconds of somebody else, then the first
    /// voice again.
    fn dialogue() -> (&'static str, Vec<Vec<Span>>) {
        let text = "No, io domani sera prendo l'aereo e vengo in Italia. Ah, ok. \
                    E domani la vabbe stacchero prima";
        let tracks = vec![
            vec![(223_072, 226_500), (227_800, 229_632)],
            vec![(226_500, 227_800)],
        ];
        (text, tracks)
    }

    fn facts(span: Span) -> LineFacts {
        LineFacts {
            span,
            text_confidence: Some(0.8),
        }
    }

    #[test]
    fn a_line_with_one_voice_in_it_is_left_alone() {
        let tracks = vec![vec![(0, 10_000)]];
        assert!(split_line(facts((0, 8_000)), "one voice talking here", &tracks, &[0.9]).is_none());
    }

    #[test]
    fn a_line_of_two_voices_is_cut_at_the_turn_boundary() {
        let text = "Prendo l'aereo e vengo in Italia. Ah ok va bene allora ci vediamo.";
        let tracks = vec![vec![(0, 4_000)], vec![(4_000, 8_000)]];
        let pieces = split_line(facts((0, 8_000)), text, &tracks, &[0.9, 0.9]).expect("cut");
        assert_eq!(pieces.len(), 2);
        assert_eq!(pieces[0].speaker, 0);
        assert_eq!(pieces[1].speaker, 1);
        assert_eq!(pieces[0].text, "Prendo l'aereo e vengo in Italia.");
        assert_eq!(pieces[1].text, "Ah ok va bene allora ci vediamo.");
        // The pieces tile the line exactly.
        assert_eq!(pieces[0].span.0, 0);
        assert_eq!(pieces[1].span.1, 8_000);
        assert_eq!(pieces[0].span.1, pieces[1].span.0);
    }

    #[test]
    fn a_short_interjection_between_two_turns_of_one_voice_makes_three_pieces() {
        let (text, tracks) = dialogue();
        let pieces = split_line(facts((223_072, 229_632)), text, &tracks, &[0.9, 0.9]);
        // The interjection is 1.3 s — under MIN_PIECE_MS, so it is not given a
        // line of its own and the whole thing stays on one voice.
        assert!(pieces.is_none(), "{pieces:?}");
    }

    #[test]
    fn a_long_enough_interjection_does_get_its_own_line() {
        let text = "No io domani sera prendo l'aereo e vengo in Italia. \
                    Ah ok ho capito bene. E domani la vabbe stacchero prima.";
        let tracks = vec![vec![(0, 4_000), (7_000, 11_000)], vec![(4_000, 7_000)]];
        let pieces =
            split_line(facts((0, 11_000)), text, &tracks, &[0.9, 0.9]).expect("three pieces");
        assert_eq!(pieces.len(), 3);
        assert_eq!(
            pieces.iter().map(|p| p.speaker).collect::<Vec<_>>(),
            vec![0, 1, 0]
        );
        assert!(pieces[1].text.starts_with("Ah ok"), "{:?}", pieces[1].text);
    }

    #[test]
    fn crosstalk_too_short_to_be_a_turn_never_cuts_a_line() {
        let text = "Prendo l'aereo e vengo in Italia domani sera va bene.";
        // The second voice holds 300 ms of an eight-second line.
        let tracks = vec![vec![(0, 3_800), (4_100, 8_000)], vec![(3_800, 4_100)]];
        assert!(split_line(facts((0, 8_000)), text, &tracks, &[0.9, 0.9]).is_none());
    }

    #[test]
    fn a_voice_holding_less_than_a_quarter_of_the_line_never_cuts_it() {
        let text = "Prendo l'aereo e vengo in Italia domani sera va bene allora ci vediamo.";
        // 1.6 s of ten: past MIN_PIECE_MS, under MIN_SHARE.
        let tracks = vec![vec![(0, 8_400)], vec![(8_400, 10_000)]];
        assert!(split_line(facts((0, 10_000)), text, &tracks, &[0.9, 0.9]).is_none());
    }

    /// Coverage on its own, with both voices clearing every other gate: 8 s of
    /// turns under a 12 s line is a boundary inferred from audio the
    /// segmentation mostly did not see.
    #[test]
    fn a_line_the_turns_barely_cover_is_left_alone() {
        let text = "Prendo l'aereo e vengo in Italia. Ah ok va bene allora ci vediamo.";
        let thin = vec![vec![(0, 4_000)], vec![(4_000, 8_000)]];
        assert!(split_line(facts((0, 12_000)), text, &thin, &[0.9, 0.9]).is_none());
        // The same turns under a line they do cover are cut.
        assert!(split_line(facts((0, 9_000)), text, &thin, &[0.9, 0.9]).is_some());
    }

    #[test]
    fn an_unsure_voice_never_cuts_a_line() {
        let text = "Prendo l'aereo e vengo in Italia. Ah ok va bene allora ci vediamo.";
        let tracks = vec![vec![(0, 4_000)], vec![(4_000, 8_000)]];
        assert!(split_line(facts((0, 8_000)), text, &tracks, &[0.9, 0.2]).is_none());
    }

    #[test]
    fn words_the_engine_does_not_trust_are_not_cut_up() {
        let text = "Prendo l'aereo e vengo in Italia. Ah ok va bene allora ci vediamo.";
        let tracks = vec![vec![(0, 4_000)], vec![(4_000, 8_000)]];
        let unsure = LineFacts {
            span: (0, 8_000),
            text_confidence: Some(0.1),
        };
        assert!(split_line(unsure, text, &tracks, &[0.9, 0.9]).is_none());
        // An older row with no number recorded is not held against it.
        let unknown = LineFacts {
            span: (0, 8_000),
            text_confidence: None,
        };
        assert!(split_line(unknown, text, &tracks, &[0.9, 0.9]).is_some());
    }

    #[test]
    fn a_short_line_is_never_cut_however_clear_the_turns_are() {
        let text = "Si certo. Va bene.";
        let tracks = vec![vec![(0, 1_200)], vec![(1_200, 2_400)]];
        assert!(split_line(facts((0, 2_400)), text, &tracks, &[0.9, 0.9]).is_none());
    }

    #[test]
    fn a_line_with_no_boundary_anywhere_near_the_cut_is_left_whole() {
        // One long sentence with the turn boundary at the halfway point: the only
        // candidates are word gaps, which is rank 2 and still allowed…
        let text = "uno due tre quattro cinque sei sette otto nove dieci undici dodici";
        let tracks = vec![vec![(0, 4_000)], vec![(4_000, 8_000)]];
        let pieces = split_line(facts((0, 8_000)), text, &tracks, &[0.9, 0.9]).expect("word gap");
        assert_eq!(pieces.len(), 2);
        // …and the cut lands on a word gap near the middle rather than inside a
        // word.
        assert!(pieces[0].text.ends_with("sei"), "{:?}", pieces[0].text);

        // Two words is not dialogue worth rewriting a row for.
        assert!(split_line(facts((0, 8_000)), "si no", &tracks, &[0.9, 0.9]).is_none());
    }

    #[test]
    fn a_sentence_end_beats_a_word_gap_that_is_nearer() {
        // Target is inside "quattro cinque"; the sentence end after "tre" is
        // three characters further away and still wins.
        let text = "uno due tre. quattro cinque sei sette";
        let tracks = vec![vec![(0, 4_000)], vec![(4_000, 8_000)]];
        let pieces = split_line(facts((0, 8_000)), text, &tracks, &[0.9, 0.9]).expect("cut");
        assert_eq!(pieces[0].text, "uno due tre.");
    }

    #[test]
    fn the_pieces_always_add_back_up_to_the_line_they_replace() {
        let text = "Prendo l'aereo e vengo in Italia. Ah ok va bene allora ci vediamo.";
        let tracks = vec![vec![(0, 4_000)], vec![(4_000, 8_000)]];
        let pieces = split_line(facts((0, 8_000)), text, &tracks, &[0.9, 0.9]).unwrap();
        let words: usize = pieces
            .iter()
            .map(|p| p.text.split_whitespace().count())
            .sum();
        assert_eq!(words, text.split_whitespace().count());
        assert_eq!(
            pieces
                .iter()
                .map(|p| timeline::span_ms(p.span))
                .sum::<i64>(),
            8_000
        );
    }

    #[test]
    fn a_contested_millisecond_stays_with_the_voice_that_had_the_one_before_it() {
        // Both voices are active over 3_500..4_500; speaker 0 owns what comes
        // before, so the overlap does not become a run of its own.
        let runs = order_turns((0, 8_000), &[vec![(0, 4_500)], vec![(3_500, 8_000)]]);
        assert!(runs.iter().all(|r| r.speaker == 0 || r.span.0 >= 4_500));
        let collapsed = collapse(runs, (0, 8_000));
        assert_eq!(collapsed.len(), 2);
        assert_eq!(collapsed[0].speaker, 0);
        assert_eq!(collapsed[1].speaker, 1);
    }

    #[test]
    fn punctuation_ranks_the_way_the_docs_say() {
        let chars: Vec<char> = "uno. due, tre quattro".chars().collect();
        let ranks: Vec<(usize, u8)> = boundaries(&chars).iter().map(|b| (b.at, b.rank)).collect();
        assert_eq!(ranks, vec![(5, 0), (10, 1), (14, 2)]);
    }

    #[test]
    fn a_closing_quote_does_not_hide_the_full_stop_behind_it() {
        let chars: Vec<char> = "disse \"basta.\" poi ando".chars().collect();
        let ranks: Vec<(usize, u8)> = boundaries(&chars).iter().map(|b| (b.at, b.rank)).collect();
        assert_eq!(ranks, vec![(6, 2), (15, 0), (19, 2)]);
    }
}
