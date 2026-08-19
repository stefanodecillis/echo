//! Deciding what language a meeting is in.
//!
//! IMPLEMENTED-BY: asr agent (M3), review finding 21.
//!
//! The naive approach — detect once on the first audio that arrives — is
//! brittle: the first three seconds of a call are "hello? can you hear me?"
//! over a bad line, or nothing at all. So:
//!
//! * detection runs on **speech**, and keeps running until enough speech has
//!   been heard with enough agreement — never on a wall-clock timer;
//! * every stretch keeps its own detected language and confidence on the
//!   segment row, so a bilingual meeting is not flattened;
//! * once settled, the dominant language is pinned and passed as a hint, which
//!   is both faster and stops mid-meeting language flapping;
//! * the pinned answer is what goes on the meeting row.
//!
//! This module is deliberately pure: no engine, no database, no clock. That is
//! what makes the policy testable.

use std::collections::HashMap;

/// Speech that has to be heard before the language can be pinned. Two utterances
/// of ordinary length, roughly.
pub const MIN_SPEECH_MS: i64 = 8_000;

/// A single detection below this confidence is ignored entirely; it carries no
/// information worth voting with.
pub const MIN_USABLE_CONFIDENCE: f32 = 0.35;

/// Share of the weighted vote the winner needs before we stop detecting.
pub const MIN_DOMINANCE: f32 = 0.6;

/// Enough speech in one stretch to be worth detecting on at all. Shorter
/// fragments produce coin-flip answers.
pub const MIN_STRETCH_MS: i64 = 1_000;

/// Running language decision for one meeting.
#[derive(Debug, Clone, Default)]
pub struct LanguagePolicy {
    /// Weighted votes per language: confidence × speech milliseconds.
    votes: HashMap<String, f64>,
    /// Speech that produced a usable detection.
    voted_speech_ms: i64,
    /// The answer, once settled.
    pinned: Option<String>,
}

/// What the policy decided after seeing one more stretch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    /// Keep detecting: not enough speech, or no clear winner yet.
    KeepDetecting,
    /// The language just became settled. Write it onto the meeting row.
    Pinned(String),
    /// Already settled; nothing changed.
    AlreadyPinned,
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
    pub fn wants_detection(&self, speech_ms: i64) -> bool {
        !self.is_settled() && speech_ms >= MIN_STRETCH_MS
    }

    /// Record what the engine reported for one stretch of speech.
    pub fn observe(&mut self, language: &str, confidence: f32, speech_ms: i64) -> Decision {
        if self.is_settled() {
            return Decision::AlreadyPinned;
        }
        let language = normalise(language);
        if language.is_empty() || speech_ms < MIN_STRETCH_MS {
            return Decision::KeepDetecting;
        }
        if !confidence.is_finite() || confidence < MIN_USABLE_CONFIDENCE {
            return Decision::KeepDetecting;
        }

        let weight = f64::from(confidence.clamp(0.0, 1.0)) * speech_ms as f64;
        *self.votes.entry(language).or_insert(0.0) += weight;
        self.voted_speech_ms += speech_ms;

        if self.voted_speech_ms < MIN_SPEECH_MS {
            return Decision::KeepDetecting;
        }
        match self.leader() {
            Some((lang, share)) if share >= f64::from(MIN_DOMINANCE) => {
                self.pinned = Some(lang.clone());
                Decision::Pinned(lang)
            }
            _ => Decision::KeepDetecting,
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

    fn leader(&self) -> Option<(String, f64)> {
        let total: f64 = self.votes.values().sum();
        if total <= 0.0 {
            return None;
        }
        // Ties break on the language name so the answer is deterministic.
        let mut best: Option<(&String, f64)> = None;
        for (lang, weight) in &self.votes {
            let better = match best {
                None => true,
                Some((best_lang, best_weight)) => {
                    *weight > best_weight || (*weight == best_weight && lang < best_lang)
                }
            };
            if better {
                best = Some((lang, *weight));
            }
        }
        best.map(|(lang, weight)| (lang.clone(), weight / total))
    }
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
        // Later stretches no longer ask for detection.
        assert!(!p.wants_detection(30_000));
        assert_eq!(p.observe("de", 0.99, 30_000), Decision::AlreadyPinned);
        assert_eq!(p.hint(), Some("en"), "a pinned meeting does not flap");
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
}
