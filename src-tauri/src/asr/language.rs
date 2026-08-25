//! Deciding what language a meeting is in.
//!
//! IMPLEMENTED-BY: asr agent (M3), review finding 21.
//!
//! The naive approach — detect once on the first audio that arrives — is
//! brittle: the first three seconds of a call are "hello? can you hear me?"
//! over a bad line, or nothing at all. On 2026-08-24 a 2.3-second utterance
//! read at 0.522 confidence said "Danish", and a 75-minute Italian meeting was
//! written down as Danish from end to end — 880 segments and a recap in Danish
//! — because that first answer was pinned and never questioned again. So:
//!
//! * detection runs on **speech**, and keeps running until enough speech has
//!   been heard, across more than one stretch, with enough agreement — never on
//!   a wall-clock timer, and never on one utterance, however long that one
//!   utterance happens to be;
//! * a stretch that was actually detected keeps its own language and confidence
//!   on the segment row, and one that merely inherited the meeting's answer
//!   keeps nothing, so what a finished transcript says about the languages in it
//!   is a set of observations rather than a copy of the pin;
//! * once settled, the answer becomes the hint the engine is given, which is
//!   both faster and more accurate than making it guess again every few
//!   seconds;
//! * but settled is not final. Every so often, straight away after a reading
//!   that came back unsure, and straight away after an answer that disagreed
//!   with the meeting, the question is asked again — because people do switch
//!   language mid-meeting. A different answer only wins when it clearly beats
//!   the one in place: twice the weight, over seconds of speech, across more
//!   than one stretch. One odd reading can never move it, and neither can a run
//!   of them, because the language in place keeps the standing of the evidence
//!   that settled it even while nothing has re-confirmed it lately.
//! * the settled answer is what goes on the meeting row.
//!
//! This module is deliberately pure: no engine, no database, no clock. That is
//! what makes the policy testable.

use std::collections::HashMap;
use std::collections::VecDeque;

/// Speech that has to be heard before the language can be pinned. Two utterances
/// of ordinary length, roughly.
pub const MIN_SPEECH_MS: i64 = 8_000;

/// Separate stretches that have to have voted before the language is pinned.
///
/// [`MIN_SPEECH_MS`] is a duration, and one stretch can hold all of it: the
/// speech detector lets a live monologue run to twenty-four seconds before it
/// cuts one, and a catch-up window is packed to half a minute by construction.
/// Either arrives as a single detection holding a hundred percent of the vote —
/// which is the exact shape of the 2026-08-24 failure, one answer never
/// questioned. Two is the smallest number of stretches that can disagree with
/// each other, and this is what makes "never on one utterance" true rather than
/// merely intended.
pub const MIN_STRETCHES: usize = 2;

/// A single detection below this confidence is ignored entirely; it carries no
/// information worth voting with.
pub const MIN_USABLE_CONFIDENCE: f32 = 0.35;

/// Share of the weighted vote the winner needs before it is pinned.
pub const MIN_DOMINANCE: f32 = 0.6;

/// Enough speech in one stretch to be worth detecting on at all. Shorter
/// fragments produce coin-flip answers.
pub const MIN_STRETCH_MS: i64 = 1_000;

/// After this much speech with no clear winner, take the one that is ahead.
///
/// A meeting where two languages are genuinely neck and neck is a real meeting —
/// somebody switching back and forth all the way through — and it would
/// otherwise never settle, which means paying for a detection pass on every
/// single utterance of it for an hour. Taking the leader is both cheaper and
/// harmless: it is only ever the hint, every stretch is still read in the
/// language it was heard in, and the rolling re-check below follows whichever
/// language takes over.
pub const SETTLE_ANYWAY_MS: i64 = 30_000;

/// How many finished stretches go by, once the meeting has settled, before the
/// question is asked again.
///
/// Detection is an extra encode, so this is not free — but at one in eight it
/// is a few percent of the decoding a meeting does anyway, and it is the only
/// thing standing between a person switching to English and half an hour of
/// English written down as Italian.
pub const RECHECK_EVERY: u32 = 8;

/// A reading this unsure asks for the language to be looked at again on the
/// next stretch, without waiting for [`RECHECK_EVERY`].
///
/// A decoder forced through the wrong language is exactly what a run of shaky
/// readings looks like, so this is the fast path onto a language change.
///
/// An absolute floor is not enough on its own — see [`RECHECK_BELOW_MEDIAN`],
/// which is the test that actually fires on the meeting this module exists for.
pub const RECHECK_BELOW_CONFIDENCE: f32 = 0.5;

/// How far below the meeting's *own* normal a reading has to sit before the
/// language is looked at again.
///
/// [`RECHECK_BELOW_CONFIDENCE`] is a fixed bar, and this repo already worked out
/// why a fixed bar does not catch a wrong language: a decoder forced through the
/// wrong vocabulary usually stays plausible enough to clear one while being
/// obviously out of step with its neighbours (`asr::catchup`, which reads the
/// same way for the same reason). The 2026-08-24 meeting read at 0.609 and never
/// once fell under 0.5, so the fixed bar never fired and the meeting stayed
/// Danish for seventy-five minutes. Three quarters of 0.609 is 0.457, which the
/// wrong-language stretches do fall under.
pub const RECHECK_BELOW_MEDIAN: f32 = 0.75;

/// How many finished readings there have to be before this meeting has a normal
/// to be far below. Until then the fixed bar is the only trigger — which is
/// what a meeting's first minute had anyway.
pub const READINGS_BEFORE_NORMAL: usize = 8;

/// How many readings the meeting's normal is measured over.
///
/// Bounded, because a policy lives as long as its meeting and an hour of talking
/// is a lot of numbers. Bounded from the *front* rather than as a rolling
/// window, on purpose: a rolling one would drift down to meet a stretch of
/// wrong-language readings and then stop calling them low, which is precisely
/// the failure this test exists to catch.
const READINGS_REMEMBERED: usize = 200;

/// How much more weight a different language needs than everything else heard
/// recently — the language in place and any other guesses — before the meeting
/// changes language.
///
/// Everything else, rather than the incumbent alone, because a stretch of bad
/// line does not guess the same wrong language twice: it guesses Danish, then
/// Norwegian, then Dutch. Measured against the incumbent alone, each of those
/// beats a language nobody has re-confirmed lately; measured against the whole
/// recent picture, none of them comes close.
pub const CHALLENGE_MULTIPLE: f64 = 2.0;

/// And how much speech that weight has to come from. Twice as much of very
/// little is still very little.
pub const CHALLENGE_MIN_SPEECH_MS: i64 = 3_000;

/// And across how many separate stretches. One stretch is an event; two is a
/// conversation. This is what a noisy patch cannot fake.
pub const CHALLENGE_MIN_STRETCHES: usize = 2;

/// And how sure the detector has to have been, on average, across them. A run of
/// barely-usable guesses is what a bad line sounds like; somebody actually
/// speaking another language does not read like that.
pub const CHALLENGE_MIN_CONFIDENCE: f64 = 0.5;

/// How many recent detections the challenge is judged over. Older ones are what
/// pinned the language in the first place; asking them again would mean a
/// meeting could never change language after the first ten minutes.
pub const RECENT_STRETCHES: usize = 6;

/// What a language carried into a policy from the meeting row — a restart, a
/// meeting picked up again — is worth when something challenges it.
///
/// The evidence that settled it belonged to a previous run and is not here to be
/// counted, but the answer is not nothing either: a whole meeting settled on it
/// once. One ordinary stretch at the bar a challenge itself has to clear is the
/// honest way to say that.
const CARRIED_OVER_STANDING: f64 = CHALLENGE_MIN_CONFIDENCE * CHALLENGE_MIN_SPEECH_MS as f64;

/// A language holding less than this share of the speaking time never wins a
/// meeting, however the rest of the transcript is split. One stray sentence
/// read as Chinese is not what the meeting was in.
pub const MIN_LANGUAGE_SHARE: f64 = 0.05;

/// A second language holding more than this much of the speaking time is worth
/// saying out loud — a recap of a meeting that was a third in English should
/// know that before it is written.
pub const ALSO_SPOKEN_SHARE: f64 = 0.25;

/// One detection, kept for as long as it counts as recent.
#[derive(Debug, Clone)]
struct Heard {
    language: String,
    weight: f64,
    speech_ms: i64,
}

/// What one language has been heard to be worth, before anything is settled.
#[derive(Debug, Clone, Copy, Default)]
struct Tally {
    /// Confidence × speech milliseconds, summed.
    weight: f64,
    speech_ms: i64,
    /// Separate stretches this language was heard in. One is an accident.
    stretches: usize,
}

/// Running language decision for one meeting.
#[derive(Debug, Clone, Default)]
pub struct LanguagePolicy {
    /// Weighted votes per language: confidence × speech milliseconds.
    votes: HashMap<String, Tally>,
    /// Speech that produced a usable detection.
    voted_speech_ms: i64,
    /// The answer, once settled.
    pinned: Option<String>,
    /// What the settled answer is worth in a challenge, as one stretch of it —
    /// see [`LanguagePolicy::challenge`]. Nothing re-confirms the language in
    /// place for eight stretches after it settles, and without this the first
    /// challenger would be measured against zero.
    standing: f64,
    /// The last few detections, which is what a language change is judged on.
    recent: VecDeque<Heard>,
    /// Finished stretches decoded since the last time the question was asked.
    since_detection: u32,
    /// The last reading came back unsure, so ask again on the next stretch.
    unsure: bool,
    /// How confidently this meeting reads, so a reading can be judged against
    /// its own meeting rather than only against a fixed number.
    reads: Vec<f32>,
}

/// What the policy decided after seeing one more stretch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    /// Keep detecting: not enough speech, or no clear winner yet.
    KeepDetecting,
    /// The language just became settled. Write it onto the meeting row.
    Pinned(String),
    /// The meeting was in one language and has moved to another. Write that on
    /// the meeting row too — and say so in the log, because a meeting changing
    /// language is the sort of thing somebody will want explained afterwards.
    Changed { from: String, to: String },
    /// Settled, and this stretch did not change it.
    Held,
}

impl LanguagePolicy {
    pub fn new() -> Self {
        Self::default()
    }

    /// Start from a language the meeting already knows, e.g. after a restart
    /// when the meeting row carries one.
    pub fn pinned_to(language: impl Into<String>) -> Self {
        Self {
            pinned: Some(language.into()),
            standing: CARRIED_OVER_STANDING,
            ..Self::default()
        }
    }

    /// The hint to pass the engine. `None` means "detect".
    pub fn hint(&self) -> Option<&str> {
        self.pinned.as_deref()
    }

    pub fn is_settled(&self) -> bool {
        self.pinned.is_some()
    }

    /// Should the engine spend time detecting on this stretch?
    ///
    /// Before the meeting has settled: on everything long enough to mean
    /// anything. After: every so often, and straight away after a reading that
    /// came back unsure — see [`RECHECK_EVERY`] and
    /// [`RECHECK_BELOW_CONFIDENCE`].
    pub fn wants_detection(&self, speech_ms: i64) -> bool {
        if speech_ms < MIN_STRETCH_MS {
            return false;
        }
        if !self.is_settled() {
            return true;
        }
        self.unsure || self.since_detection >= RECHECK_EVERY
    }

    /// One more finished stretch was decoded, with the confidence it came back
    /// at. This is what moves the meeting towards its next re-check.
    ///
    /// Captions and second readings of a stretch are not finished stretches and
    /// do not belong here: one is replaced within seconds, the other is a second
    /// look at seconds that have already had their say.
    pub fn saw_decode(&mut self, avg_confidence: Option<f32>) {
        self.since_detection = self.since_detection.saturating_add(1);
        let Some(confidence) = avg_confidence else {
            return;
        };
        if !confidence.is_finite() {
            self.unsure = true;
            return;
        }
        // Judged against the meeting as it stood *before* this reading, then
        // added to it: a reading is never part of the evidence about itself.
        if confidence < RECHECK_BELOW_CONFIDENCE || self.far_below(confidence) {
            self.unsure = true;
        }
        if self.reads.len() < READINGS_REMEMBERED {
            self.reads.push(confidence);
        }
    }

    /// How confidently this meeting reads, once enough of it has been read for
    /// that to mean anything.
    fn normal(&self) -> Option<f32> {
        if self.reads.len() < READINGS_BEFORE_NORMAL {
            return None;
        }
        let mut sorted = self.reads.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let mid = sorted.len() / 2;
        Some(if sorted.len().is_multiple_of(2) {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        })
    }

    /// Did this reading fall a long way short of how this meeting normally
    /// reads? See [`RECHECK_BELOW_MEDIAN`] for how far "a long way" is.
    fn far_below(&self, confidence: f32) -> bool {
        self.normal()
            .is_some_and(|normal| confidence < normal * RECHECK_BELOW_MEDIAN)
    }

    /// What this stretch should be decoded in, given what the detector just
    /// said about it.
    ///
    /// A usable fresh answer wins for **this stretch only** — that is how a
    /// sentence somebody said in English inside an Italian meeting gets written
    /// down in English rather than as Italian nonsense. An unusable one falls
    /// back to the meeting's own answer.
    pub fn reading(&self, detected: Option<(&str, f32)>) -> Option<String> {
        usable(detected).or_else(|| self.hint().map(str::to_string))
    }

    /// Record what the engine reported for one stretch of speech.
    pub fn observe(&mut self, language: &str, confidence: f32, speech_ms: i64) -> Decision {
        // The question was asked, whatever the answer turned out to be: the
        // countdown to the next re-check starts here.
        self.since_detection = 0;
        self.unsure = false;

        let language = normalise(language);
        if language.is_empty() || speech_ms < MIN_STRETCH_MS {
            return self.nothing_learned();
        }
        if !confidence.is_finite() || confidence < MIN_USABLE_CONFIDENCE {
            return self.nothing_learned();
        }

        let weight = f64::from(confidence.clamp(0.0, 1.0)) * speech_ms as f64;
        if self.is_settled() {
            return self.challenge(language, weight, speech_ms);
        }

        let tally = self.votes.entry(language).or_default();
        tally.weight += weight;
        tally.speech_ms += speech_ms;
        tally.stretches += 1;
        self.voted_speech_ms += speech_ms;

        // Enough speech *and* more than one stretch of it. One long utterance
        // clears the first on its own and holds the whole vote while it does —
        // see [`MIN_STRETCHES`].
        if self.voted_speech_ms < MIN_SPEECH_MS || self.voted_stretches() < MIN_STRETCHES {
            return Decision::KeepDetecting;
        }
        let enough_of_this = self.voted_speech_ms >= SETTLE_ANYWAY_MS;
        match self.leader() {
            Some((lang, share)) if share >= f64::from(MIN_DOMINANCE) || enough_of_this => {
                self.standing = self.standing_of(&lang);
                self.pinned = Some(lang.clone());
                Decision::Pinned(lang)
            }
            _ => Decision::KeepDetecting,
        }
    }

    /// Stretches that have voted for anything at all.
    fn voted_stretches(&self) -> usize {
        self.votes.values().map(|t| t.stretches).sum()
    }

    /// One stretch's worth of what a language has been heard to be worth.
    fn standing_of(&self, language: &str) -> f64 {
        match self.votes.get(language) {
            Some(tally) if tally.stretches > 0 => tally.weight / tally.stretches as f64,
            _ => 0.0,
        }
    }

    /// The best answer available right now, settled or not. Used when a
    /// recording ends before the policy ever settled.
    pub fn dominant(&self) -> Option<String> {
        if let Some(p) = &self.pinned {
            return Some(p.clone());
        }
        self.leader().map(|(lang, _)| lang)
    }

    /// Speech that has contributed a usable vote.
    pub fn voted_speech_ms(&self) -> i64 {
        self.voted_speech_ms
    }

    /// How much of the vote the settled answer holds, for the log line that
    /// says why this meeting is in the language it is in.
    pub fn confidence(&self) -> f32 {
        match self.leader() {
            Some((_, share)) => share as f32,
            None => 0.0,
        }
    }

    fn nothing_learned(&self) -> Decision {
        if self.is_settled() {
            Decision::Held
        } else {
            Decision::KeepDetecting
        }
    }

    /// A settled meeting hearing a stretch. The language in place keeps the
    /// meeting unless this and the other recent stretches clearly say otherwise.
    fn challenge(&mut self, language: String, weight: f64, speech_ms: i64) -> Decision {
        let Some(held) = self.pinned.clone() else {
            return Decision::Held;
        };
        // An answer that disagrees with the meeting brings the next question
        // forward, rather than waiting out the next [`RECHECK_EVERY`]. A
        // challenge needs several stretches to agree with each other, and at one
        // detection in eight that is thirty-two finished stretches — four or
        // five minutes of somebody speaking English written down as Italian
        // before the meeting follows them. Asked on the next stretch instead, a
        // real switch takes seconds; a bad line still cannot get past the bars
        // below, and stops costing a detection as soon as it agrees again.
        if language != held {
            self.unsure = true;
        }
        self.recent.push_back(Heard {
            language,
            weight,
            speech_ms,
        });
        while self.recent.len() > RECENT_STRETCHES {
            self.recent.pop_front();
        }

        // Recent speech, gathered per language: how much weight, over how much
        // speech, across how many separate stretches.
        let mut tally: HashMap<&str, (f64, i64, usize)> = HashMap::new();
        for heard in &self.recent {
            let entry = tally.entry(heard.language.as_str()).or_insert((0.0, 0, 0));
            entry.0 += heard.weight;
            entry.1 += heard.speech_ms;
            entry.2 += 1;
        }
        let recent_weight: f64 = tally.values().map(|e| e.0).sum();

        // The strongest of everything that is not the language in place. Ties
        // break on the name so the answer never depends on hash order.
        let mut best: Option<(&str, f64, i64, usize)> = None;
        for (lang, (weight, ms, stretches)) in &tally {
            if *lang == held.as_str() {
                continue;
            }
            let better = match best {
                None => true,
                Some((best_lang, best_weight, _, _)) => {
                    *weight > best_weight || (*weight == best_weight && *lang < best_lang)
                }
            };
            if better {
                best = Some((lang, *weight, *ms, *stretches));
            }
        }
        let Some((challenger, weight, ms, stretches)) = best else {
            return Decision::Held;
        };
        // Everything that is not the challenger — and the language in place is
        // always part of that, whether or not anything has re-confirmed it
        // lately.
        //
        // Nothing re-confirms it for eight stretches after it settles, and the
        // fast path onto a re-check fires precisely on bad audio, which is when
        // the challenger's readings are bad too. Without the standing of the
        // evidence that settled it, `everything_else` was zero on exactly those
        // stretches, "twice everything else" was "at least nothing", and two
        // borderline readings of the 2026-08-24 shape — 0.52 over 2.3 seconds —
        // flipped a settled meeting.
        let held_recently = tally.get(held.as_str()).map(|e| e.0).unwrap_or(0.0);
        let others = (recent_weight - weight - held_recently).max(0.0);
        let everything_else = others + held_recently.max(self.standing);
        let mean_confidence = if ms > 0 { weight / ms as f64 } else { 0.0 };
        let clearly_wins = weight >= everything_else * CHALLENGE_MULTIPLE
            && ms >= CHALLENGE_MIN_SPEECH_MS
            && stretches >= CHALLENGE_MIN_STRETCHES
            && mean_confidence >= CHALLENGE_MIN_CONFIDENCE;
        if !clearly_wins {
            return Decision::Held;
        }

        let winner = challenger.to_string();
        // The evidence that won stays on the books, on every count: it is what
        // the *next* challenger has to beat, so a meeting cannot flap back and
        // forth between two languages on the strength of one stretch each.
        self.votes.clear();
        self.votes.insert(
            winner.clone(),
            Tally {
                weight,
                speech_ms: ms,
                stretches,
            },
        );
        self.voted_speech_ms = ms;
        self.standing = weight / stretches.max(1) as f64;
        self.recent.clear();
        self.recent.push_back(Heard {
            language: winner.clone(),
            weight,
            speech_ms: ms,
        });
        self.pinned = Some(winner.clone());
        // The question has just been answered, and answered thoroughly.
        self.unsure = false;
        Decision::Changed {
            from: held,
            to: winner,
        }
    }

    fn leader(&self) -> Option<(String, f64)> {
        let total: f64 = self.votes.values().map(|t| t.weight).sum();
        if total <= 0.0 {
            return None;
        }
        // Ties break on the language name so the answer is deterministic.
        let mut best: Option<(&String, f64)> = None;
        for (lang, tally) in &self.votes {
            let weight = tally.weight;
            let better = match best {
                None => true,
                Some((best_lang, best_weight)) => {
                    weight > best_weight || (weight == best_weight && lang < best_lang)
                }
            };
            if better {
                best = Some((lang, weight));
            }
        }
        best.map(|(lang, weight)| (lang.clone(), weight / total))
    }
}

/// What a finished meeting turned out to be in, read off its transcript.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Spoken {
    /// The language most of the meeting was spoken in.
    pub dominant: Option<String>,
    /// A second language a real part of the meeting was in, when there was one.
    pub also: Option<String>,
}

/// Read a meeting's language off the speaking time in its transcript.
///
/// Dominant by speaking time, as before — but with two rules the 2026-08-24
/// meeting made necessary. A language holding a sliver of the transcript never
/// wins, however the rest of it is split: a single sentence misread as Chinese
/// is not what the meeting was in. And when a second language holds a real share
/// of it, that is said out loud rather than averaged away, so a meeting that ran
/// a third in English is not silently written up as if it had not.
///
/// `histogram` is `(language, milliseconds)`, largest first — what
/// `db::repo::language_histogram` returns.
pub fn spoken_in(histogram: &[(String, i64)]) -> Spoken {
    let total: i64 = histogram
        .iter()
        .map(|(_, ms)| *ms)
        .filter(|ms| *ms > 0)
        .sum();
    if total <= 0 {
        return Spoken::default();
    }
    let mut kept: Vec<(&str, i64)> = histogram
        .iter()
        .filter(|(lang, ms)| {
            *ms > 0
                && !normalise(lang).is_empty()
                && (*ms as f64) / (total as f64) >= MIN_LANGUAGE_SHARE
        })
        .map(|(lang, ms)| (lang.as_str(), *ms))
        .collect();
    // Never trust the caller's ordering for something this load-bearing.
    kept.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    let mut spoken = Spoken::default();
    if let Some((lang, _)) = kept.first() {
        spoken.dominant = Some(normalise(lang));
    }
    if let Some((lang, ms)) = kept.get(1) {
        if (*ms as f64) / (total as f64) > ALSO_SPOKEN_SHARE {
            spoken.also = Some(normalise(lang));
        }
    }
    spoken
}

/// A detection worth reading a stretch in: a real language code, said with
/// enough confidence to carry any information at all.
///
/// `None` says the detector answered nothing usable about these seconds — which
/// is a different thing from the meeting's own answer, and the two must not be
/// confused: one is an observation and one is a hand-me-down.
pub fn usable(detected: Option<(&str, f32)>) -> Option<String> {
    let (language, confidence) = detected?;
    let language = normalise(language);
    if language.is_empty() || !confidence.is_finite() || confidence < MIN_USABLE_CONFIDENCE {
        return None;
    }
    Some(language)
}

/// Whisper reports two-letter codes; keep them lowercase and drop anything
/// that is not a language ("auto", empty, whitespace).
fn normalise(language: &str) -> String {
    let lang = language.trim().to_ascii_lowercase();
    if lang.is_empty() || lang == "auto" {
        return String::new();
    }
    lang
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_is_decided_before_enough_speech_has_been_heard() {
        let mut p = LanguagePolicy::new();
        assert!(p.hint().is_none());
        // A whole minute of wall-clock time with only two seconds of speech
        // must not settle anything.
        assert_eq!(p.observe("en", 0.99, 2_000), Decision::KeepDetecting);
        assert!(!p.is_settled());
        assert_eq!(p.voted_speech_ms(), 2_000);
    }

    /// The 2026-08-24 meeting, exactly: one short utterance, read at barely half
    /// confidence, said Danish. Seventy-five minutes of Italian followed.
    #[test]
    fn one_short_shaky_utterance_does_not_pin_a_meeting() {
        let mut p = LanguagePolicy::new();
        assert_eq!(p.observe("da", 0.522, 2_300), Decision::KeepDetecting);
        assert!(!p.is_settled(), "2.3 seconds is not a meeting");
        assert!(
            p.hint().is_none(),
            "and the engine is still asked to detect"
        );
        assert!(p.wants_detection(4_000));
        // The Italian that actually followed takes it, and Danish never gets a
        // hint out of the engine at all.
        p.observe("it", 0.9, 6_000);
        p.observe("it", 0.9, 6_000);
        assert_eq!(p.hint(), Some("it"));
    }

    /// The other way one answer can take a whole meeting: not a short utterance
    /// but a long one. The speech detector cuts a monologue at twenty-four
    /// seconds, so "enough speech" arrives complete inside a single detection,
    /// holding a hundred percent of the vote.
    #[test]
    fn one_long_utterance_does_not_pin_a_meeting_either() {
        let mut p = LanguagePolicy::new();
        assert_eq!(p.observe("da", 0.36, 13_000), Decision::KeepDetecting);
        assert!(!p.is_settled(), "one stretch is one stretch, however long");
        assert!(p.hint().is_none());
        assert!(p.wants_detection(4_000), "so it is asked again");
    }

    /// And the same for a catch-up window, which is packed to half a minute of
    /// speech by construction — so on "listen again", which deliberately starts
    /// with nothing pinned, the very first window used to decide the meeting.
    #[test]
    fn one_catch_up_window_does_not_pin_a_meeting() {
        let mut p = LanguagePolicy::new();
        assert_eq!(p.observe("da", 0.40, 20_000), Decision::KeepDetecting);
        assert!(!p.is_settled());
        // A second window that agrees does settle it: two stretches is the bar,
        // not two minutes.
        assert_eq!(
            p.observe("da", 0.40, 20_000),
            Decision::Pinned("da".to_string())
        );
    }

    #[test]
    fn enough_agreeing_speech_pins_the_language() {
        let mut p = LanguagePolicy::new();
        assert_eq!(p.observe("en", 0.9, 4_000), Decision::KeepDetecting);
        assert_eq!(
            p.observe("en", 0.95, 5_000),
            Decision::Pinned("en".to_string())
        );
        assert!(p.is_settled());
        assert_eq!(p.hint(), Some("en"));
        // The next stretches are decoded on the pinned answer rather than
        // paying for a detection each.
        assert!(!p.wants_detection(30_000));
        // And one stretch of another language, however long and however
        // confident, is one stretch: it does not move the meeting.
        assert_eq!(p.observe("de", 0.99, 30_000), Decision::Held);
        assert_eq!(p.hint(), Some("en"), "one stretch does not flap a meeting");
    }

    #[test]
    fn a_shaky_detection_carries_no_vote() {
        let mut p = LanguagePolicy::new();
        assert_eq!(p.observe("uk", 0.05, 10_000), Decision::KeepDetecting);
        assert_eq!(p.voted_speech_ms(), 0, "low confidence does not count");
        assert!(p.dominant().is_none());
    }

    #[test]
    fn very_short_fragments_are_ignored() {
        let mut p = LanguagePolicy::new();
        assert!(!p.wants_detection(200));
        assert_eq!(p.observe("fr", 0.99, 200), Decision::KeepDetecting);
        assert_eq!(p.voted_speech_ms(), 0);
    }

    #[test]
    fn a_disagreeing_meeting_keeps_detecting_until_one_side_dominates() {
        let mut p = LanguagePolicy::new();
        p.observe("en", 0.8, 5_000);
        p.observe("it", 0.8, 5_000);
        assert!(
            !p.is_settled(),
            "a 50/50 split is not a decision, even with plenty of speech"
        );
        // More Italian tips it over the dominance threshold.
        p.observe("it", 0.9, 12_000);
        assert_eq!(p.hint(), Some("it"));
    }

    /// A meeting somebody spends switching back and forth: no side ever
    /// dominates, and the policy would ask the engine to detect on every
    /// utterance of it for an hour. It takes the one that is ahead instead —
    /// and the rolling re-check is still there to follow a real switch.
    #[test]
    fn a_meeting_with_no_clear_winner_still_stops_paying_for_detection() {
        let mut p = LanguagePolicy::new();
        let mut spoken = 0;
        while !p.is_settled() {
            p.observe("it", 0.9, 5_000);
            p.observe("en", 0.9, 5_000);
            spoken += 10_000;
            assert!(spoken <= SETTLE_ANYWAY_MS * 2, "this has to end");
        }
        // Whichever of the two ended up ahead, the meeting can still follow the
        // other one when it takes over.
        let settled = p.hint().expect("settled").to_string();
        let other = if settled == "it" { "en" } else { "it" };
        p.observe(other, 0.9, 6_000);
        p.observe(other, 0.9, 6_000);
        assert_eq!(p.hint(), Some(other));
    }

    #[test]
    fn the_best_guess_is_available_before_the_policy_settles() {
        let mut p = LanguagePolicy::new();
        p.observe("es", 0.8, 3_000);
        p.observe("en", 0.6, 1_500);
        assert!(!p.is_settled());
        assert_eq!(p.dominant().as_deref(), Some("es"));
    }

    #[test]
    fn junk_language_codes_are_dropped() {
        let mut p = LanguagePolicy::new();
        assert_eq!(p.observe("auto", 0.99, 10_000), Decision::KeepDetecting);
        assert_eq!(p.observe("  ", 0.99, 10_000), Decision::KeepDetecting);
        assert_eq!(p.observe("NaN", f32::NAN, 10_000), Decision::KeepDetecting);
        assert_eq!(p.voted_speech_ms(), 0);
    }

    #[test]
    fn codes_are_compared_case_insensitively() {
        let mut p = LanguagePolicy::new();
        p.observe("EN", 0.9, 5_000);
        assert_eq!(
            p.observe("en", 0.9, 5_000),
            Decision::Pinned("en".to_string())
        );
    }

    #[test]
    fn a_known_language_can_be_carried_over_from_a_restart() {
        let p = LanguagePolicy::pinned_to("nl");
        assert!(p.is_settled());
        assert_eq!(p.hint(), Some("nl"));
        assert!(!p.wants_detection(60_000));
    }

    // -- asking again, once the meeting has settled -------------------------

    #[test]
    fn a_settled_meeting_asks_again_every_so_often() {
        let mut p = settled_on("it");
        for _ in 0..(RECHECK_EVERY - 1) {
            p.saw_decode(Some(0.9));
            assert!(!p.wants_detection(5_000), "not yet");
        }
        p.saw_decode(Some(0.9));
        assert!(p.wants_detection(5_000), "eight stretches on, ask again");
        // Asking resets the count, whatever the answer was.
        p.observe("it", 0.9, 5_000);
        assert!(!p.wants_detection(5_000));
    }

    #[test]
    fn a_reading_that_came_back_unsure_asks_again_straight_away() {
        let mut p = settled_on("it");
        p.saw_decode(Some(0.31));
        assert!(
            p.wants_detection(5_000),
            "a decoder forced through the wrong language reads like this"
        );
        p.observe("it", 0.9, 5_000);
        assert!(!p.wants_detection(5_000), "and the question is answered");
    }

    #[test]
    fn a_meeting_that_switches_language_follows_it() {
        let mut p = settled_on("it");
        // One English stretch is not a switch.
        assert_eq!(p.observe("en", 0.9, 5_000), Decision::Held);
        assert_eq!(p.hint(), Some("it"));
        // A second one, and the meeting is in English now.
        assert_eq!(
            p.observe("en", 0.9, 5_000),
            Decision::Changed {
                from: "it".to_string(),
                to: "en".to_string()
            }
        );
        assert_eq!(p.hint(), Some("en"));
        assert_eq!(p.dominant().as_deref(), Some("en"));
    }

    #[test]
    fn a_noisy_stretch_cannot_flap_the_language() {
        let mut p = settled_on("it");
        // Half a minute of readings that barely clear the usable bar, each one
        // guessing something different — the shape of a bad line, not of a
        // conversation in another language.
        for lang in ["da", "no", "sv", "da", "nl", "de"] {
            assert_eq!(p.observe(lang, 0.4, 2_000), Decision::Held);
        }
        assert_eq!(p.hint(), Some("it"), "the meeting is still Italian");
    }

    #[test]
    fn a_run_of_barely_usable_guesses_cannot_flap_the_language_either() {
        let mut p = settled_on("it");
        // The harder case: the bad line keeps guessing the *same* wrong
        // language, so counting stretches alone would call it a switch. It
        // never reads like somebody speaking Danish, though.
        for _ in 0..RECENT_STRETCHES {
            assert_eq!(p.observe("da", 0.4, 2_000), Decision::Held);
        }
        assert_eq!(p.hint(), Some("it"));
        // Said with any conviction, Danish does take the meeting — the point is
        // the confidence, not the amount.
        p.observe("da", 0.9, 5_000);
        p.observe("da", 0.9, 5_000);
        assert_eq!(p.hint(), Some("da"));
    }

    /// The window a settled meeting is most exposed in: it has just settled, so
    /// nothing has re-confirmed the language yet, and the fast path onto a
    /// re-check fires precisely on the bad audio that produces bad challengers.
    /// Two readings of the exact 2026-08-24 shape — half confidence, a couple of
    /// seconds — used to be enough to take the meeting.
    #[test]
    fn a_meeting_that_has_only_just_settled_still_cannot_be_flipped_by_a_bad_line() {
        let mut p = settled_on("it");
        assert_eq!(p.observe("da", 0.52, 2_300), Decision::Held);
        assert_eq!(p.observe("da", 0.54, 2_500), Decision::Held);
        assert_eq!(p.observe("da", 0.53, 2_400), Decision::Held);
        assert_eq!(p.hint(), Some("it"), "the meeting is still Italian");
    }

    /// A language carried in from the meeting row has none of its own evidence
    /// to hand — but a whole meeting settled on it once, so it does not fall
    /// over to the first two shaky readings after a restart either.
    #[test]
    fn a_language_carried_in_from_the_row_is_not_defenceless() {
        let mut p = LanguagePolicy::pinned_to("it");
        assert_eq!(p.observe("da", 0.52, 2_300), Decision::Held);
        assert_eq!(p.observe("da", 0.54, 2_500), Decision::Held);
        assert_eq!(p.hint(), Some("it"));
    }

    /// An answer that disagrees with the meeting is a question, not a verdict —
    /// and the question is asked again on the very next stretch. At one
    /// detection in eight, a real switch otherwise took thirty-two finished
    /// stretches to take effect: four or five minutes written down in the wrong
    /// language.
    #[test]
    fn an_answer_that_disagrees_brings_the_next_question_forward() {
        let mut p = settled_on("it");
        assert_eq!(p.observe("en", 0.9, 5_000), Decision::Held);
        assert!(
            p.wants_detection(5_000),
            "somebody may have just switched language; ask again now"
        );
        // And an answer that agrees goes back to asking every so often.
        assert_eq!(p.observe("it", 0.9, 5_000), Decision::Held);
        assert!(!p.wants_detection(5_000));
    }

    /// The fixed floor is not what catches a wrong language. The 2026-08-24
    /// meeting read at 0.609 throughout and never once fell under 0.5, so the
    /// floor never fired; what a decoder forced through the wrong vocabulary
    /// looks like is a reading well below how the rest of *this* meeting reads.
    #[test]
    fn a_reading_far_below_this_meetings_own_normal_asks_again() {
        let mut p = settled_on("it");
        for _ in 0..READINGS_BEFORE_NORMAL {
            p.saw_decode(Some(0.609));
        }
        // Eight steady stretches is also when the meeting is due its ordinary
        // re-check; answering it puts the count back to zero, so what follows is
        // about the confidence and nothing else.
        p.observe("it", 0.9, 5_000);
        assert!(!p.wants_detection(5_000), "nothing is wrong yet");
        // Clears the fixed floor comfortably, and is a long way under 0.609.
        p.saw_decode(Some(0.44));
        assert!(
            p.wants_detection(5_000),
            "0.44 passes the floor and still means the language does not fit"
        );
    }

    /// And a meeting that simply reads quietly all the way through is not
    /// treated as one that keeps going wrong.
    #[test]
    fn a_meeting_that_reads_low_throughout_has_a_low_normal() {
        let mut p = settled_on("it");
        for _ in 0..READINGS_BEFORE_NORMAL {
            p.saw_decode(Some(0.55));
        }
        p.observe("it", 0.9, 5_000);
        for _ in 0..(RECHECK_EVERY - 1) {
            p.saw_decode(Some(0.55));
            assert!(!p.wants_detection(5_000), "0.55 is this meeting's normal");
        }
    }

    #[test]
    fn a_language_that_just_won_is_not_given_up_on_the_next_stretch() {
        let mut p = settled_on("it");
        p.observe("en", 0.9, 5_000);
        assert!(matches!(
            p.observe("en", 0.9, 5_000),
            Decision::Changed { .. }
        ));
        // Two Italian stretches straight after would have flipped it back if the
        // winning evidence had been thrown away.
        assert_eq!(p.observe("it", 0.9, 4_000), Decision::Held);
        assert_eq!(p.observe("it", 0.9, 4_000), Decision::Held);
        assert_eq!(p.hint(), Some("en"));
    }

    #[test]
    fn a_stretch_is_read_in_the_language_it_was_heard_in() {
        let p = settled_on("it");
        // A confident English answer for these seconds: read them as English,
        // which is how a bilingual meeting reads as one.
        assert_eq!(p.reading(Some(("en", 0.8))).as_deref(), Some("en"));
        // A shaky one is not evidence of anything; the meeting's answer stands.
        assert_eq!(p.reading(Some(("da", 0.2))).as_deref(), Some("it"));
        assert_eq!(p.reading(None).as_deref(), Some("it"));
        // And with nothing settled and nothing detected, the engine decides.
        assert_eq!(LanguagePolicy::new().reading(None), None);
    }

    // -- what a finished transcript says the meeting was in -----------------

    #[test]
    fn a_stray_sentence_never_wins_the_meeting() {
        let hist = vec![
            ("it".to_string(), 3_600_000),
            ("zh".to_string(), 30_000),
            ("da".to_string(), 4_000),
        ];
        let spoken = spoken_in(&hist);
        assert_eq!(spoken.dominant.as_deref(), Some("it"));
        assert_eq!(
            spoken.also, None,
            "1% of a meeting is not a language it was in"
        );
    }

    #[test]
    fn a_meeting_held_in_two_languages_says_so() {
        let hist = vec![("it".to_string(), 700_000), ("en".to_string(), 300_000)];
        let spoken = spoken_in(&hist);
        assert_eq!(spoken.dominant.as_deref(), Some("it"));
        assert_eq!(spoken.also.as_deref(), Some("en"));
    }

    #[test]
    fn a_meeting_that_is_almost_all_one_language_has_no_second_one() {
        let hist = vec![("it".to_string(), 900_000), ("en".to_string(), 100_000)];
        let spoken = spoken_in(&hist);
        assert_eq!(spoken.dominant.as_deref(), Some("it"));
        assert_eq!(spoken.also, None, "a tenth is a quote, not a language");
    }

    #[test]
    fn a_transcript_with_no_language_on_it_decides_nothing() {
        assert_eq!(spoken_in(&[]), Spoken::default());
        assert_eq!(spoken_in(&[("auto".to_string(), 1_000)]), Spoken::default());
    }

    /// Settled the way a real meeting settles, so the tests above start from a
    /// policy that has actually been through it.
    fn settled_on(language: &str) -> LanguagePolicy {
        let mut p = LanguagePolicy::new();
        p.observe(language, 0.9, 5_000);
        p.observe(language, 0.9, 5_000);
        assert!(p.is_settled());
        p
    }
}
