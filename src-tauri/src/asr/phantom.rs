//! Silence that came back as words, and how to recognise it afterwards.
//!
//! The meeting of 2026-08-24 produced 943 segments, 33 of which nobody said.
//! Twenty-six were a bare "Grazie."; the rest were "Grazie a voi.", "Grazie a
//! te.", "Buonanotte." at half past ten in the morning, "Ciao ciao",
//! "Buongiorno.", "Alla prossima." and one "Wow!". Every single one of them sat
//! against dead air — the recording has 68 gaps of fifteen seconds or more, the
//! longest 75 seconds — and every single one was decoded *confidently*.
//!
//! That last word is why this file exists instead of a decoder setting.
//! whisper.cpp only throws a window away as silence when it is both fairly sure
//! there was no speech **and** unsure about the words it produced anyway
//! (`no_speech_prob > no_speech_thold && avg_logprobs < logprob_thold`,
//! whisper.cpp:7585). A courtesy phrase hallucinated over a pause fails the
//! second half: the model is *certain* about "Grazie." — it is one of the most
//! common things in its training data — so no threshold that leaves real quiet
//! speech alone will ever catch it. See [`crate::asr::catalog`] for what the
//! thresholds were tightened to anyway, and why that is a different job.
//!
//! So this is a filter, applied after the decode, and it needs two things to be
//! true at once before it deletes anything a person might have said:
//!
//! 1. the segment is **nothing but** a stock courtesy phrase or a subtitling
//!    watermark — not a sentence containing one; and
//! 2. the recording holds **almost no voice** under it, according to the speech
//!    detector that was already running.
//!
//! One condition alone would be wrong in both directions. Plenty of real
//! meetings end with somebody saying only "Grazie", and plenty of quiet windows
//! contain a real short answer. Two conditions is the whole design.
//!
//! # The sibling rule, and why it is not here
//!
//! The recording of 2026-08-26 produced a second family of lines nobody said:
//! `you` at 160 ms and `Bye.` four times consecutively at 40 ms each, three of
//! them at confidence 0.99 to 1.0. This filter never saw them, and should not
//! have: "you" is not a courtesy phrase, and no deny-list that stayed honest
//! would ever hold it. What gives those lines away is not what they say but how
//! *narrow* they are — forty milliseconds is two frames of mel, and the speech
//! detector will not open a stretch under
//! [`crate::audio::vad::MIN_UTTERANCE_MS`].
//!
//! That is a judgement about a span rather than about words: it needs no list,
//! no detector and no language, so it lives where spans are made, at the
//! splitter — see [`crate::asr::catchup`]'s `MIN_LINE_MS`.

/// Less voice than this under a stock phrase, and nobody said it.
///
/// **Milliseconds of detected voice, deliberately, not a share of the span.** A
/// share was the first thing this was, and it measures the padding as much as
/// the speech. Work it through for [`crate::audio::vad::VadSettings::mic`], in
/// 32 ms windows: an utterance opens carrying 320 ms of look-behind, cannot open
/// at all until three windows in a row are speech, and closes 416 ms after the
/// last voiced one — and is then held to
/// [`crate::audio::vad::MIN_UTTERANCE_MS`] (1120 ms) if it comes out shorter. So
/// **every** short mic utterance carries ~736 ms of guaranteed non-voice, and
/// the same three-syllable answer reads as 0.46 of a stretch that closed at its
/// own pause and 0.09 of a twenty-second one that was cut through the middle.
/// The quarter-share this used to be worked out at 256 ms of detected voice on
/// the mic channel and at something else on every other lane, which is not a
/// threshold anybody could reason about.
///
/// Milliseconds mean the same thing everywhere, so the number can be argued
/// about honestly:
///
/// * what starts a phantom is a cough, a keyboard click, a breath — three to
///   five windows, **96 to 160 ms**, because three windows is what it takes to
///   open an utterance at all and a click does not last longer;
/// * a real "Grazie." is two syllables, half a second of phonation even said
///   quickly, of which a far-field detector marks perhaps two thirds: **300 to
///   400 ms**.
///
/// A fifth of a second sits between them: about twice the blip, about half the
/// word. That is the real headroom — narrow, which is exactly why the deny-list
/// is the other half of the test and why this is never asked on its own.
pub const TOO_LITTLE_VOICE_MS: i64 = 200;

/// Stock phrases a decoder reaches for when it is handed a pause.
///
/// Written as they come out of [`normalized`]: lower case, no punctuation,
/// single spaces. Only whole segments are compared against this list, so
/// "grazie" here cannot touch "grazie, allora vediamo" — that is a sentence, and
/// sentences are never dropped.
///
/// The Italian entries are the ones the 2026-08-24 meeting actually produced,
/// plus the near neighbours of each (a decoder that says "grazie a voi" into
/// silence says "grazie a tutti" into the next silence). The English and French
/// ones are the same phenomenon in the two other languages Echo is most likely
/// to meet.
const COURTESY_LINES: &[&str] = &[
    // --- Italian: every one of these was in the 2026-08-24 transcript, or is
    // the same phrase with a different pronoun.
    "grazie",
    "grazie a voi",
    "grazie a te",
    "grazie a tutti",
    "grazie a lei",
    "grazie mille",
    "grazie di tutto",
    "grazie per l'attenzione",
    "buongiorno",
    "buonasera",
    "buonanotte",
    "ciao ciao",
    "arrivederci",
    "alla prossima",
    "a presto",
    "wow",
    // --- English
    "thank you",
    "thanks",
    "thank you very much",
    "thanks for watching",
    "thank you for watching",
    "goodbye",
    "bye bye",
    "see you next time",
    // --- French
    "merci",
    "merci beaucoup",
    "au revoir",
    "a bientot",
    "à bientôt",
    // --- the shortest form the subtitling credit collapses to
    "sottotitoli",
    "sottotitoli e revisione",
];

/// Fragments of the subtitling credits whisper learned from its training data.
///
/// These are matched as **substrings** of the whole normalized segment rather
/// than exactly, because the credit line comes out worded a dozen different
/// ways ("sottotitoli e revisione a cura di…", "sottotitoli creati dalla
/// comunità amara.org", "subtitles by the amara.org community"). The
/// 2026-08-24 recording happens to contain none of them; they are here because
/// they are the same failure with a longer string, and a filter that only knew
/// about the phrases one meeting produced would be a filter fitted to one
/// meeting.
///
/// Every entry is a whole phrase from the credit, never a single word. "I
/// sottotitoli sono già attivi" is a sentence somebody says in a meeting about
/// a video call; "sottotitoli e revisione" is not.
const WATERMARK_MARKERS: &[&str] = &[
    "amara.org",
    "sottotitoli e revisione",
    "sottotitoli creati",
    "sottotitoli a cura",
    "sous-titrage",
    "sous-titres réalisés",
    "subtitles by",
    "subtitled by",
    "untertitel von",
    "untertitelung im auftrag",
];

/// Lower case, no punctuation, single spaces — the form the lists above are
/// written in.
///
/// Word by word rather than character by character, so "Grazie, a voi." and
/// "grazie a voi" are the same three words. Apostrophes inside a word survive
/// ("l'attenzione"); ones at the edge do not.
fn normalized(text: &str) -> String {
    text.split_whitespace()
        .map(|word| {
            word.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Is this segment *entirely* a stock courtesy phrase or a subtitling
/// watermark?
///
/// Half of the test. On its own it means nothing: a meeting that really ends
/// with "Grazie." produces exactly this. See [`is_phantom`].
pub fn courtesy_or_watermark_only(text: &str) -> bool {
    let text = normalized(text);
    if text.is_empty() {
        return false;
    }
    COURTESY_LINES.contains(&text.as_str())
        || WATERMARK_MARKERS.iter().any(|marker| text.contains(marker))
}

/// The whole predicate: stock phrase **and** next to no voice under it.
///
/// `voiced_ms` is how many milliseconds of this segment's own span the speech
/// detector called speech. Where that is not known — no detector on disk, a
/// window handed over whole — pass the width of the span: no evidence of silence
/// is not evidence of silence, and this must never delete words on a guess.
pub fn is_phantom(text: &str, voiced_ms: i64) -> bool {
    voiced_ms < TOO_LITTLE_VOICE_MS && courtesy_or_watermark_only(text)
}

// ---------------------------------------------------------------------------
// Where the voice was
// ---------------------------------------------------------------------------

/// The stretches a speech detector called speech, so any span of a meeting can
/// be asked how much of it was voice.
///
/// The catch-up pass decodes a **packed window**: several stretches of speech
/// and the pauses between them, handed to the engine as one continuous piece of
/// the recording (see [`crate::asr::catchup`]). When the window comes back as
/// several lines, each has its own place on the clock and is asked about its own
/// seconds — a thirty-second window holding four seconds of speech is perfectly
/// normal and says nothing about any one line in it.
///
/// When it comes back as a single line, though, there is no per-line clock to
/// ask, and that line is left spanning the whole packed window. That is why the
/// answer is a duration and not a share: see [`VoicedSpans::voiced_ms`].
///
/// Within one detected stretch the voice is treated as evenly spread. It is not,
/// but the alternative is carrying a per-32-ms bitmap through the whole pass for
/// a decision made in fifths of a second.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct VoicedSpans {
    /// `(start, end, share of it that was voice)`, in the order they arrived.
    spans: Vec<(i64, i64, f32)>,
}

impl VoicedSpans {
    /// Record one detected stretch and how much of it was voice.
    pub fn add(&mut self, from_ms: i64, to_ms: i64, voiced_ratio: f32) {
        if to_ms > from_ms {
            self.spans
                .push((from_ms, to_ms, voiced_ratio.clamp(0.0, 1.0)));
        }
    }

    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Take everything `other` knows.
    pub fn absorb(&mut self, other: &VoicedSpans) {
        self.spans.extend_from_slice(&other.spans);
    }

    /// How many milliseconds of `[from_ms, to_ms)` were voice. Time no stretch
    /// covers is silence, which is the whole point: that is where the phantoms
    /// live.
    ///
    /// Asked about one line's own seconds it answers about that line. Asked
    /// about a whole packed window — which is what a line whisper gave no
    /// timestamps of its own is left spanning — it answers with every
    /// millisecond of voice the window holds, across all the stretches that were
    /// packed into it. That is the safe way round, and it is why this counts
    /// milliseconds rather than dividing by the width: a window holding three
    /// separate seconds of speech across thirty seconds of clock is *normal*,
    /// and a share of it would read as silence and delete a real answer nobody
    /// could place any more precisely.
    pub fn voiced_ms(&self, from_ms: i64, to_ms: i64) -> i64 {
        if to_ms <= from_ms {
            return 0;
        }
        let voiced: f64 = self
            .spans
            .iter()
            .map(|(start, end, share)| {
                let overlap = end.min(&to_ms) - start.max(&from_ms);
                if overlap <= 0 {
                    0.0
                } else {
                    overlap as f64 * f64::from(*share)
                }
            })
            .sum();
        (voiced.round() as i64).clamp(0, to_ms - from_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 2026-08-24 cases, one row each. The first column is what the segment
    /// said, the second whether the deny-list claims it.
    #[test]
    fn the_deny_list_claims_whole_courtesy_lines_and_nothing_else() {
        let table: &[(&str, bool)] = &[
            // --- the 33 phantoms of 2026-08-24, by wording ------------------
            ("Grazie.", true),
            ("grazie", true),
            ("Grazie!", true),
            ("Grazie a voi.", true),
            ("Grazie, a voi.", true),
            ("Grazie a te.", true),
            ("Grazie a tutti!", true),
            ("Grazie mille.", true),
            ("Buonanotte.", true),
            ("Buongiorno.", true),
            ("Ciao ciao", true),
            ("Alla prossima.", true),
            ("Wow!", true),
            ("  Wow  ", true),
            // --- watermarks: the same failure with a longer string ----------
            ("Sottotitoli e revisione a cura di QTSS", true),
            ("Sottotitoli creati dalla comunità Amara.org", true),
            ("Subtitles by the Amara.org community", true),
            ("Sottotitoli", true),
            // --- sentences that merely contain one of the words -------------
            ("Grazie, allora vediamo domani.", false),
            ("Ti ringrazio, grazie di tutto quello che hai fatto.", false),
            ("Buongiorno a tutti, cominciamo?", false),
            ("Wow, non me lo aspettavo.", false),
            ("Thanks, that works for me.", false),
            ("I sottotitoli sono già attivi sulla registrazione.", false),
            // --- ordinary talk ----------------------------------------------
            ("Ciao", false),
            ("Allora, direi che possiamo procedere.", false),
            ("", false),
            ("...", false),
            ("?!", false),
        ];
        for (text, expected) in table {
            assert_eq!(
                courtesy_or_watermark_only(text),
                *expected,
                "{text:?} was judged wrong"
            );
        }
    }

    /// The rule that keeps the filter honest: it takes both conditions, and a
    /// segment that fails either one is left exactly where it is.
    #[test]
    fn a_lone_grazie_over_dead_air_goes_and_the_same_word_over_speech_stays() {
        // One cough: the three 32 ms windows it takes to open a stretch at all.
        assert!(is_phantom("Grazie.", 96));
        assert!(is_phantom("Buonanotte.", 0));
        // Somebody really said it: half a second of phonation.
        assert!(!is_phantom("Grazie.", 500));
        // A quiet, clipped one survives too — that is what the headroom is for.
        assert!(!is_phantom("Grazie.", 260));
        // A real sentence over near-silence is not this filter's business; the
        // detector heard something, and the words are not a stock phrase.
        assert!(!is_phantom("Grazie, allora vediamo domani.", 0));
        // An unmeasured stretch is passed as its own width and never deletes
        // anything.
        assert!(!is_phantom("Grazie.", 1_200));
    }

    /// The bar is a duration, so it can be checked against the two durations it
    /// sits between rather than against a share of a padded window.
    #[test]
    fn the_bar_sits_between_the_blip_that_opens_a_stretch_and_a_spoken_word() {
        // Nothing shorter than this can become a phantom: it is what the
        // detector needs before it will open a stretch at all.
        let blip_ms = crate::audio::vad::VadSettings::mic().min_voiced_ms;
        assert!(
            TOO_LITTLE_VOICE_MS >= blip_ms * 2,
            "a click that flickers voiced twice as long as the debounce must \
             still be caught"
        );
        // And the quietest real two-syllable answer is above it, with the same
        // sort of room the other way.
        const A_QUIET_GRAZIE_MS: i64 = 400;
        const {
            assert!(
                TOO_LITTLE_VOICE_MS * 2 <= A_QUIET_GRAZIE_MS,
                "a real word must survive"
            )
        };
    }

    #[test]
    fn punctuation_and_case_do_not_hide_a_phantom() {
        assert_eq!(normalized("  Grazie,  a   VOI. "), "grazie a voi");
        assert_eq!(normalized("«Grazie!»"), "grazie");
        assert_eq!(normalized("l'attenzione"), "l'attenzione");
        assert_eq!(normalized("... ?!"), "");
    }

    // -------------------------------------------------------------------
    // Where the voice was
    // -------------------------------------------------------------------

    #[test]
    fn a_line_in_a_gap_between_two_stretches_reads_as_silence() {
        let mut voiced = VoicedSpans::default();
        voiced.add(0, 4_000, 0.6);
        voiced.add(30_000, 34_000, 0.6);
        // A line whisper placed in the twenty-six seconds nobody spoke in.
        assert_eq!(voiced.voiced_ms(10_000, 11_000), 0);
        // …and one inside a stretch that really was speech.
        assert_eq!(voiced.voiced_ms(1_000, 2_000), 600);
        // A line straddling the edge gets the share it actually overlaps.
        assert_eq!(voiced.voiced_ms(3_000, 5_000), 600);
    }

    /// The failure this counts milliseconds to avoid: a window that holds two
    /// far-apart seconds of speech is a normal packed window, and a line nothing
    /// could place inside it must be judged against the voice the window holds,
    /// not against the pause between the stretches.
    #[test]
    fn a_window_of_far_apart_stretches_is_not_silence_just_because_it_is_wide() {
        let mut voiced = VoicedSpans::default();
        for start in [0, 15_000, 27_000] {
            voiced.add(start, start + 1_200, 0.5);
        }
        // A share of the whole window would be 0.06 — silence, on any threshold
        // that leaves a cough caught.
        assert_eq!(voiced.voiced_ms(0, 28_200), 1_800);
        assert!(!is_phantom("Grazie a tutti.", voiced.voiced_ms(0, 28_200)));
        // Line by line, the same window still tells the truth about each line.
        assert!(!is_phantom(
            "Grazie a tutti.",
            voiced.voiced_ms(27_000, 28_200)
        ));
        assert!(is_phantom(
            "Grazie a tutti.",
            voiced.voiced_ms(20_000, 21_200)
        ));
    }

    #[test]
    fn an_empty_or_backwards_span_is_silence_rather_than_a_division_by_zero() {
        let voiced = VoicedSpans::default();
        assert_eq!(voiced.voiced_ms(0, 0), 0);
        assert_eq!(voiced.voiced_ms(500, 100), 0);
        assert!(voiced.is_empty());
    }

    #[test]
    fn spans_can_be_taken_over_wholesale() {
        let mut first = VoicedSpans::default();
        first.add(0, 1_000, 1.0);
        let mut second = VoicedSpans::default();
        second.add(1_000, 2_000, 1.0);
        first.absorb(&second);
        assert_eq!(first.voiced_ms(0, 2_000), 2_000);
    }
}
