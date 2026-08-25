//! The words a person knows and the decoder does not.
//!
//! The meeting of 2026-08-24 was about a product called **Langola**. The word
//! is in the transcript eight times and it is spelled six different ways, none
//! of them right: *Ingola*, *Langura*, *sull'angolo*, *lana gola*, *Nongula*,
//! *Nongulo*. Obsidara came out three ways, Feedharbour three, Vogliacasa four,
//! "VMC / Voglia Mutui Casa" became "BMC è voglio lui casa", Yomenia became Omeni,
//! Ostri became Anastri, lampo.dev became "Lampo.gr". Every one of them is a word
//! the person could have typed in ten seconds before the call — and some of
//! them Echo already knows, because the voices were enrolled by name.
//!
//! So this module is two halves of one idea, and they are deliberately
//! different in temperament.
//!
//! ## Before the decode: a nudge
//!
//! [`Glossary::context`] builds the text handed to whisper.cpp as
//! `initial_prompt` — the same slot the tail of a cut sentence already travels
//! in (see [`crate::asr::engine::DecodePlan::prompt`]). It is a *bias*: the
//! decoder is more likely to reach for a word it has just read, and no more than
//! that. Nothing here can force a spelling, and nothing here is allowed to grow
//! without limit: past roughly two hundred tokens whisper starts writing the
//! prompt back out into the transcript, which would be a far worse bug than the
//! one this fixes. Hence [`PROMPT_MAX_CHARS`], and a truncation that stops at a
//! whole entry.
//!
//! ## After the decode: a repair, made nervously
//!
//! [`Glossary::correct`] is the dangerous half. It rewrites words in text a
//! person reads as the record of what was said, so every rule in it is written
//! to fail closed:
//!
//! * a word shorter than [`SHORTEST_TOKEN`] is never looked at;
//! * an entry shorter than [`SHORTEST_NEAR_MISS_ENTRY`] can only be matched
//!   letter for letter — "VMC" may fix "vmc" and may never touch "BMC";
//! * a candidate that is an [everyday word](EVERYDAY_WORDS) of one of the
//!   languages Echo meets is never touched, and neither is an entry that is one;
//! * the looser the rule, the longer the entry has to be to use it
//!   ([`Entry::reach`]);
//! * every rule but the tightest asks for something the *decoder* did on top of
//!   what the letters say — a capital in the middle of a sentence, an
//!   apostrophe where an article ran into the next word — because past a
//!   certain point the letters alone stop telling a misheard name from a word;
//! * and everything it changes is recorded, word for word, so the line can say
//!   what happened to it and be put back.
//!
//! Recall is the thing given away here, on purpose, and it has been given away
//! generously: of the six spellings that meeting invented for Langola, this
//! fixes four, and *Ingola* and *Langura* are left alone on the grounds that
//! nothing tight enough to be trusted can reach them. The measure that decided
//! every threshold in here is a false-positive count, not a hit count: with
//! thirty-five plausible names in the list, the rules below rewrite one word in
//! ten thousand of ordinary English and none at all of the frequent Italian
//! vocabulary. Every loosening that was tried and rejected cost hundreds of
//! times that — *whatever* became a product name, *include* became a person,
//! *bianco* became somebody's surname. A meeting where "Langola" is still
//! written *Langura* twice is a meeting somebody skims past; a meeting where the
//! ordinary Italian word *amabile* has been quietly turned into a product name
//! is a meeting nobody can trust again. The tests carry both.

use crate::types::Correction;

/// Most characters of vocabulary handed to the decoder, prompt and all.
///
/// whisper.cpp keeps the last `n_text_ctx / 2` tokens of the prompt (224 for
/// every model Echo ships), and long before that limit a prompt stops being a
/// hint and starts being something the decoder transcribes: it has been trained
/// on text that continues, so a list it cannot place in the audio comes back out
/// in the words. Eight hundred characters is roughly two hundred tokens of
/// names, which is a comfortable distance below both.
pub const PROMPT_MAX_CHARS: usize = 800;

/// Most entries kept at all. Well past what anybody types, and it keeps the
/// per-line matching cost bounded no matter what lands in the database.
pub const MAX_ENTRIES: usize = 200;

/// Shortest decoded word the near-miss matcher will look at.
///
/// Three letters is "poi", "che", "www" — there is no way to be sure enough
/// about a three-letter word to rewrite it, and every wrong answer is a word
/// somebody actually said.
pub const SHORTEST_TOKEN: usize = 4;

/// Shortest entry that may be matched by anything other than its own spelling.
///
/// "VMC" is the reason this exists. A three-letter entry is within one edit of
/// hundreds of real words and abbreviations, so it gets exactly one power: it
/// fixes its own casing.
pub const SHORTEST_NEAR_MISS_ENTRY: usize = 5;

/// Shortest entry that may be matched by something that merely *sounds* like
/// it, rather than by the letters it is written with.
///
/// Six characters is where a name stops being one of the language's own shapes.
/// *Notion* is one edit from *motion*, *nation*, *lotion* and *potion*;
/// *Claude* is one from *clause*; *Vercel* keeps *cercare*'s sounds and
/// *Sentry* keeps *centro*'s. Nothing at that length is far enough from
/// ordinary speech to be worth guessing at, so a six-letter entry gets its own
/// spelling and the one rule that asks to find it written out in full.
pub const SHORTEST_SOUND_ALIKE_ENTRY: usize = 7;

/// Fewest sounds a name must have before [rule 2](Entry::claims) — the one that
/// reads past the vowels — is allowed to look at it at all.
///
/// The sound classes are coarse on purpose, so a short name has a shape the
/// language is full of: *Sentry* and *centro* are the same six sounds bar the
/// first, and so are *Vercel* and *cercare*, *Notion* and *notaio*, *Claude*
/// and *classe*, *Bianchi* and *bianco*. At seven sounds the coincidences stop
/// being coincidences — which is exactly where *Langola* and *Nongula* sit.
const LONG_ENOUGH_TO_MISHEAR: usize = 7;

// ---------------------------------------------------------------------------
// The list
// ---------------------------------------------------------------------------

/// The words Echo has been told about, ready to prompt with and to match
/// against.
///
/// Built once — per meeting for the live pass, once per pass for catch-up — and
/// then read from many times, so everything derived from an entry is derived
/// here rather than per line.
#[derive(Debug, Default, Clone)]
pub struct Glossary {
    entries: Vec<Entry>,
    /// Longest window of decoded words any entry will look at, so the matcher
    /// does not walk window lengths no entry can use.
    widest_window: usize,
}

#[derive(Debug, Clone)]
struct Entry {
    /// Exactly as the person typed it: this is what gets written into the text.
    word: String,
    /// Lower case, unaccented, letters and digits only.
    folded: String,
    /// [`folded`](Self::folded) as sound classes, vowels flattened.
    sounds: String,
    /// [`sounds`](Self::sounds) with the vowels removed: the consonant spine.
    skeleton: String,
    /// How many words the entry itself is ("Voglia Mutui Casa" is three).
    words: usize,
    /// Which rules this entry is long enough to use.
    reach: Reach,
}

/// How far from its own spelling an entry is allowed to reach.
///
/// The rule is one sentence: **the shorter the word, the fewer ways there are
/// to arrive at it.** A four-letter entry is one edit away from most of the
/// language; a nine-letter product name is not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reach {
    /// Its own spelling, ignoring case and punctuation. Nothing else.
    Spelling,
    /// …and a longer word with the entry buried inside it (Anastri → Ostri).
    Buried,
    /// …and everything: near spellings, same-consonant sound-alikes, and two
    /// decoded words that together sound like one entry.
    SoundAlike,
}

impl Glossary {
    /// Take a list of words, in the order they should be offered to the decoder.
    ///
    /// Blank entries go, duplicates go (the first spelling wins, so a typed
    /// "Langola" beats a person named "Langola"), and anything past
    /// [`MAX_ENTRIES`] goes. Order is preserved exactly: the prompt is truncated
    /// from the end, so the caller's order decides what survives a cap, and the
    /// same list always produces the same prompt.
    pub fn new<I, S>(words: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut entries: Vec<Entry> = Vec::new();
        for word in words {
            let word = word.as_ref().trim();
            if word.is_empty() || entries.len() >= MAX_ENTRIES {
                continue;
            }
            let folded = fold(word);
            if folded.is_empty() || entries.iter().any(|e| e.folded == folded) {
                continue;
            }
            let sounds = sounds(&folded);
            let skeleton = skeleton(&sounds);
            let words = word.split_whitespace().count().max(1);
            let reach = Reach::for_length(folded.chars().count());
            entries.push(Entry {
                word: word.to_string(),
                folded,
                sounds,
                skeleton,
                words,
                reach,
            });
        }
        let widest_window = entries
            .iter()
            .map(|e| {
                if e.reach.hears_two_words() {
                    e.words + 1
                } else {
                    e.words
                }
            })
            .max()
            .unwrap_or(0);
        Self {
            entries,
            widest_window,
        }
    }

    /// Nothing to say and nothing to fix. Every path checks this first, because
    /// an empty vocabulary has to leave the app exactly as it was.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// The words, in the order they were given.
    pub fn words(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|e| e.word.as_str())
    }

    /// The text to hand the decoder before it listens: the vocabulary, then
    /// whatever was carried across a forced cut.
    ///
    /// Order matters and this is the way round it has to be. whisper.cpp reads
    /// the prompt as the text immediately *before* this audio, so the words
    /// closest to the end are the ones it treats as the sentence in progress —
    /// which is precisely what the carried tail is. The vocabulary is
    /// background; it goes first.
    ///
    /// With an empty vocabulary this returns the carried tail unchanged, byte
    /// for byte, so nothing about the existing carry-over behaviour moves.
    pub fn context(&self, carried: Option<&str>) -> Option<String> {
        let carried = carried.map(str::trim).filter(|c| !c.is_empty());
        if self.entries.is_empty() {
            return carried.map(str::to_string);
        }
        // The cap covers everything the decoder is handed, so the carried tail —
        // which is the half that must not be dropped — is budgeted for first.
        let tail_cost = carried.map_or(0, |c| c.chars().count() + 1);
        let budget = PROMPT_MAX_CHARS.saturating_sub(tail_cost);

        let mut list = String::new();
        for entry in &self.entries {
            // ", " between entries, "." after the last one.
            let cost = entry.word.chars().count() + if list.is_empty() { 1 } else { 3 };
            if list.chars().count() + cost > budget {
                // Stop at the first entry that does not fit rather than skipping
                // it and trying the next: a prompt whose contents depend on the
                // widths further down the list is a prompt nobody can predict.
                break;
            }
            if !list.is_empty() {
                list.push_str(", ");
            }
            list.push_str(&entry.word);
        }
        if list.is_empty() {
            return carried.map(str::to_string);
        }
        list.push('.');
        match carried {
            Some(tail) => {
                list.push(' ');
                list.push_str(tail);
                Some(list)
            }
            None => Some(list),
        }
    }

    /// The vocabulary on its own, for the lanes that carry nothing across.
    pub fn prompt(&self) -> Option<String> {
        self.context(None)
    }

    /// Put right what the decoder nearly got, and say what was changed.
    ///
    /// `None` means "leave this line alone" — not an empty change list, which
    /// the callers would have to remember to treat as untouched. When it is
    /// `Some`, every byte of the line that was not part of a corrected word is
    /// still exactly where it was: spacing, punctuation and case around the
    /// match are copied through, and only the matched words are replaced.
    pub fn correct(&self, text: &str) -> Option<Corrected> {
        if self.entries.is_empty() || text.is_empty() {
            return None;
        }
        let spans = word_spans(text);
        if spans.is_empty() {
            return None;
        }
        let folded: Vec<String> = spans.iter().map(|s| fold(&text[s.0..s.1])).collect();

        let mut changes: Vec<Correction> = Vec::new();
        let mut out = String::new();
        let mut copied = 0usize;
        let mut at = 0usize;
        while at < spans.len() {
            match self.claim(&folded, at, text, &spans) {
                Some((width, entry)) => {
                    let from = &text[spans[at].0..spans[at + width - 1].1];
                    out.push_str(&text[copied..spans[at].0]);
                    out.push_str(&entry.word);
                    copied = spans[at + width - 1].1;
                    changes.push(Correction {
                        from: from.to_string(),
                        to: entry.word.clone(),
                    });
                    at += width;
                }
                None => at += 1,
            }
        }
        if changes.is_empty() {
            return None;
        }
        out.push_str(&text[copied..]);
        Some(Corrected { text: out, changes })
    }

    /// The best entry for the words starting at `at`, and how many words it
    /// takes.
    ///
    /// Deterministic on purpose, three tie-breaks deep: the widest window wins
    /// (two decoded words that together sound like one entry is a stronger
    /// claim than either of them alone), then the tightest rule, then the
    /// entry that comes first in the list.
    fn claim<'a>(
        &'a self,
        folded: &[String],
        at: usize,
        text: &str,
        spans: &[(usize, usize)],
    ) -> Option<(usize, &'a Entry)> {
        let mut best: Option<(usize, u8, &Entry)> = None;
        let widest = self.widest_window.min(spans.len() - at);
        for width in 1..=widest {
            let parts = &folded[at..at + width];
            // Two decoded words are only ever read as one when both of them are
            // words. A single letter next to a name is "è", "e", "a", "o" — the
            // Italian sentence around it — and gluing it on turns "È lavabile"
            // into a candidate that keeps Langola's consonants exactly.
            if width > 1 && parts.iter().any(|p| p.chars().count() < 2) {
                continue;
            }
            // …and only when the line has nothing between them but the space
            // the decoder put there. A claim is written back over the whole
            // stretch from the first word to the last, so anything else in
            // between — a full stop, a comma, a line break — would be deleted
            // by the replacement: "Non ho tempo. Casa mia è lontana" must not
            // come back as "Non ho Vogliacasa mia è lontana". Nothing wider can
            // be contiguous either once this window is not, hence the break.
            if width > 1 && !spaces_between(text, spans, at, width) {
                break;
            }
            let raw = &text[spans[at].0..spans[at + width - 1].1];
            let candidate = Candidate::of(
                parts.concat(),
                raw,
                width,
                opens_a_sentence(text, spans[at].0),
            );
            for entry in &self.entries {
                let Some(rank) = entry.claims(&candidate) else {
                    continue;
                };
                let better = match best {
                    None => true,
                    Some((best_width, best_rank, _)) => {
                        width > best_width || (width == best_width && rank < best_rank)
                    }
                };
                if better {
                    best = Some((width, rank, entry));
                }
            }
        }
        best.map(|(width, _, entry)| (width, entry))
    }
}

/// One line, put right, and the list of what was changed in it.
#[derive(Debug, Clone, PartialEq)]
pub struct Corrected {
    pub text: String,
    pub changes: Vec<Correction>,
}

impl Reach {
    fn for_length(folded_chars: usize) -> Self {
        if folded_chars < SHORTEST_NEAR_MISS_ENTRY {
            Reach::Spelling
        } else if folded_chars < SHORTEST_SOUND_ALIKE_ENTRY {
            Reach::Buried
        } else {
            Reach::SoundAlike
        }
    }

    fn hears_two_words(self) -> bool {
        self == Reach::SoundAlike
    }
}

/// One run of decoded words, asked about once per entry.
///
/// Everything here is worked out once for the run rather than once per entry,
/// including the two everyday-word questions, which are the expensive ones.
struct Candidate<'a> {
    /// The run, folded and run together: "lana gola" is `lana gola`.
    folded: String,
    /// The same run exactly as it appears in the line.
    raw: &'a str,
    /// How many decoded words it is.
    words: usize,
    /// The run itself is an everyday word. Nothing may touch it.
    everyday: bool,
    /// What follows an elided article is an everyday word: "l'angolo" is *the
    /// oval*, and folded it is `langolo` — one letter from `Langola`. Only the
    /// rules that go by sound are stopped by this; the one that needs the entry
    /// to be present in full is not, which is the whole difference between
    /// "l'angolo" (left alone) and "sull'angolo" (the 2026-08-24 mangling of
    /// Langola, which holds `langolo` inside a longer word).
    everyday_tail: bool,
    /// The decoder wrote this word the way it writes a name: a capital letter,
    /// in the middle of a sentence rather than at the start of one.
    ///
    /// This is the only thing that separates *Nongula* from *whatever* — the
    /// two are the same shape, and one of them is a name whisper had never
    /// heard and wrote down as a name anyway. Every single-token mangling that
    /// meeting produced came back capitalised: *Ingola*, *Langura*, *Nongula*,
    /// *Nongulo*, *Anastri*, *Lampo.gr*, *Omeni*. It is evidence, not proof, so
    /// only [rule 2](Entry::claims) — the one rule that reads past the spelling
    /// entirely — is allowed to lean on it, and a decoder that hands Echo a
    /// line with no capitals in it simply loses that rule rather than
    /// misreading the line.
    named: bool,
    /// The word wears an elided article or preposition: `sull'angolo`, `l'angolo`,
    /// `dell'angolo`. That apostrophe is a decoder telling you it heard the
    /// sentence run into the next word, which is exactly the shape
    /// [rule 3](Entry::claims) is looking for.
    elided: bool,
}

impl<'a> Candidate<'a> {
    fn of(folded: String, raw: &'a str, words: usize, opens_sentence: bool) -> Self {
        let single = words == 1;
        let everyday = single && is_everyday(&folded);
        let elision = single.then(|| elided_tail(raw)).flatten();
        let everyday_tail = elision.as_deref().is_some_and(is_everyday);
        let named = !opens_sentence && raw.chars().next().is_some_and(char::is_uppercase);
        Self {
            folded,
            raw,
            words,
            everyday,
            everyday_tail,
            named,
            elided: elision.is_some(),
        }
    }

    fn len(&self) -> usize {
        self.folded.chars().count()
    }
}

/// Is the word beginning at `from` the first word of a sentence?
///
/// Every language Echo meets capitalises the word after a full stop, so a
/// capital there says nothing about the word. Anything that ends a sentence
/// counts, and so does the start of the line and the start of a new one.
fn opens_a_sentence(text: &str, from: usize) -> bool {
    let before = &text[..from];
    let spaced = before.trim_end();
    // A line break is a full stop as far as this is concerned.
    if before[spaced.len()..].contains('\n') {
        return true;
    }
    match spaced
        .chars()
        .rev()
        .find(|c| !matches!(c, '"' | '\'' | '’' | '«' | '(' | '[' | '“' | '-' | '—'))
    {
        None => true,
        Some(c) => matches!(c, '.' | '!' | '?' | ':' | ';' | '…'),
    }
}

/// What follows an elided article or preposition — the `ovale` of "sull'angolo"
/// — folded, when the word has one.
///
/// Only a short prefix counts, because that is what an elision is: `l'`, `un'`,
/// `dell'`, `sull'`, `quell'`. Anything longer is a word with an apostrophe in
/// it, which is a different thing.
fn elided_tail(raw: &str) -> Option<String> {
    let cut = raw.rfind(['\'', '\u{2019}'])?;
    let prefix = fold(&raw[..cut]);
    let tail = fold(&raw[cut..]);
    (!prefix.is_empty() && prefix.chars().count() <= 5 && tail.chars().count() >= 3).then_some(tail)
}

impl Entry {
    /// Does this entry claim this run of decoded words? `Some(rank)`, where a
    /// lower rank is a tighter rule.
    fn claims(&self, candidate: &Candidate<'_>) -> Option<u8> {
        let entry_len = self.folded.chars().count();
        let candidate_len = candidate.len();
        let folded = candidate.folded.as_str();

        // Rule 0 — the same word, spelled the same way. Nothing to do unless the
        // capitals or the punctuation differ ("Obsidara" → "Obsidara",
        // "vmc" → "VMC"), which is the one thing every entry may fix, however
        // short it is. Two letters is where even that stops: "ai" and "di" are
        // Italian, and an entry that short would recapitalise the language.
        if folded == self.folded {
            return (entry_len >= 3 && candidate.raw != self.word).then_some(0);
        }
        if self.reach == Reach::Spelling || candidate_len < SHORTEST_TOKEN {
            return None;
        }
        // An entry that is itself an everyday word is a word people say. It gets
        // its own spelling and nothing more, whatever its length: somebody who
        // adds "Casa" to the list has not asked for "cosa" to become it.
        if is_everyday(&self.folded) {
            return None;
        }

        if candidate.words > 1 {
            // Two decoded words that were meant to be one ("lana gola",
            // "voglia casa"). The everyday-word guard cannot help here — the
            // halves of this kind of mishearing are nearly always ordinary
            // words — so the run has to look like the entry three separate
            // ways at once, and the consonant test is only one of them.
            //
            // On its own, "the same consonants in the same order" is not a
            // bar at all: a spine is two to five characters, so *la favola*
            // keeps Langola's, *non ho* keeps Yomenia's, and *un attimo*
            // keeps Notion's. What actually separates "lana gola" from
            // "la favola" is the spelling: run together, a real mishearing
            // lands within a few letters of the name and starts the same way,
            // and an ordinary phrase does neither.
            if !self.reach.hears_two_words() {
                return None;
            }
            let same_spine = skeleton(&sounds(folded)) == self.skeleton;
            return (same_spine
                && difference(candidate_len, entry_len) <= 3
                && distance(folded, &self.folded) <= 3
                && shared_prefix(folded, &self.folded) >= 2)
                .then_some(2);
        }

        // Nothing below can reach across more than three characters of length,
        // so anything further away is not compared at all. This is the gate that
        // keeps a list of two hundred names from costing anything measurable on
        // a line of twenty words.
        if difference(candidate_len, entry_len) > 3 {
            return None;
        }

        // From here on it is one decoded word, and it has to be a word nobody
        // recognises. This guard is what stands between the rules below and the
        // meeting language: *amabile* is two edits from *Langola*, and it is a
        // word somebody said.
        if candidate.everyday {
            return None;
        }
        // …and the rules that go by sound are also stopped by an everyday word
        // wearing an elided article. Rule 3 below is not, because it asks for
        // something stronger than a resemblance.
        let sounds_like = !candidate.everyday_tail;
        let sound_alike = self.reach == Reach::SoundAlike && sounds_like;
        let candidate_sounds = sounds(folded);

        // Rule 1 — a near spelling. One insertion, deletion, substitution or
        // swap: "obsidera", "Vogliacasa".
        if sound_alike && distance(folded, &self.folded) <= 1 {
            return Some(1);
        }

        // Rule 2 — the first sound misheard, every sound after it kept. This is
        // Nongula and Nongulo for Langola: *n* for *l*, and then `bbl` exactly,
        // in order. It is the only rule that reads past the spelling altogether,
        // so it is fenced in on four sides.
        //
        // The tail has to be identical, because a *near* spine is not evidence
        // of anything: six sound classes cover the consonants, so a spine is two
        // to five characters long, and one edit inside something that short
        // reaches most of the language — Langola's `lbbl` is one edit from
        // *lovely*, *libera*, *lavabo* and *valuable*, and Yomenia's `nn` is one
        // edit from *Milano*, *mattina*, *Monday* and *nuovo*.
        //
        // There has to be enough of that tail ([`LONG_ENOUGH_TO_MISHEAR`] sounds
        // in the name, at least four of them consonants), because *Sentry* and
        // *centro* are also a first sound apart, and so are *Vercel* and
        // *cercare*.
        //
        // And the word has to have been written down like a name
        // ([`Candidate::named`]). That is what is left when the shape stops
        // being enough: *whatever* and *October* stand in exactly the same
        // relation to *Airtable* that *Nongula* does to *Langola*, and the only
        // thing that tells them apart is that the decoder wrote one of the three
        // with a capital in the middle of a sentence.
        if sound_alike
            && candidate.named
            && difference(candidate_len, entry_len) <= 2
            && candidate_sounds.chars().count() == self.sounds.chars().count()
            && self.sounds.chars().count() >= LONG_ENOUGH_TO_MISHEAR
        {
            let spine = skeleton(&candidate_sounds);
            let entry_spine = self.skeleton.as_str();
            if entry_spine.chars().count() >= 4
                && spine.chars().count() == entry_spine.chars().count()
                && spine.chars().next() != entry_spine.chars().next()
                && spine.chars().skip(1).eq(entry_spine.chars().skip(1))
            {
                return Some(2);
            }
        }

        // Rule 3 — the entry with something stuck to the *front* of it.
        // "sull'angolo" is one token holding "langolo" after `sul`; "Anastri" is
        // one holding "Ostri" after `an`. This is the one rule a five-letter
        // entry may use, because it asks for the entry to be there in full
        // rather than for something that merely sounds like it.
        //
        // Both ends are pinned, and neither is a detail.
        //
        // Something has to come first, or a word that *begins* with the entry
        // and carries on is claimed by it — and a word that begins with a name
        // and carries on is that word's own ending: *milanese* is how you say a
        // firm is from Milano, *veronese* the same for Verona, *marchio* is a
        // trademark and not a person called Marco.
        //
        // And nothing may come after, or the entry is being found in the middle
        // of a word it has nothing to do with — the run has to be the tail of
        // what was decoded, the way a decoder that has glued a preposition onto
        // a name it does not know leaves it.
        //
        // And the word has to be one the decoder itself flagged: written like a
        // name it did not know, or wearing the apostrophe of an article it ran
        // into the next word. Without that, "a name with a syllable in front of
        // it" is a description of ordinary vocabulary — *allowable* is
        // *Langola* behind `al`, *include* is *Claude* behind `in`, *mention*
        // and *attention* are both *Notion* behind something.
        if (candidate.named || candidate.elided)
            && candidate_len > entry_len
            && candidate_len <= entry_len + 3
            && buried_in_tail(&self.folded, folded) <= 1
        {
            return Some(3);
        }
        None
    }
}

fn difference(a: usize, b: usize) -> usize {
    a.abs_diff(b)
}

/// How many characters two folded words open with in common.
fn shared_prefix(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

// ---------------------------------------------------------------------------
// Reading a line as words
// ---------------------------------------------------------------------------

/// Byte ranges of the words in a line, punctuation trimmed off each end.
///
/// A word keeps the apostrophes, hyphens and dots *inside* it, which is the
/// whole reason this is not `split_whitespace`: "sull'angolo" is the token that
/// has to be recognised, and "lampo.dev" is an entry. Anything hanging off the
/// end — the full stop, the closing quote — is left in the line, where it stays
/// untouched when the word is replaced.
pub(crate) fn word_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start: Option<usize> = None;
    for (index, ch) in text.char_indices() {
        if is_word_char(ch) {
            start.get_or_insert(index);
        } else if let Some(from) = start.take() {
            spans.push((from, index));
        }
    }
    if let Some(from) = start {
        spans.push((from, text.len()));
    }
    spans
        .into_iter()
        .filter_map(|(from, to)| trim_to_letters(text, from, to))
        .collect()
}

/// Is every gap inside this window of words made of spaces and nothing else?
///
/// [`Glossary::correct`] replaces the whole stretch between the first and last
/// word of a claim, so this is what keeps punctuation and line breaks — which
/// are part of what somebody said and where they stopped saying it — from being
/// swallowed by a two-word match.
fn spaces_between(text: &str, spans: &[(usize, usize)], at: usize, width: usize) -> bool {
    (at..at + width - 1).all(|index| {
        let gap = &text[spans[index].1..spans[index + 1].0];
        gap.chars()
            .all(|c| c.is_whitespace() && c != '\n' && c != '\r')
    })
}

fn is_word_char(ch: char) -> bool {
    ch.is_alphanumeric() || matches!(ch, '\'' | '’' | '-' | '.' | '_')
}

/// Pull the span in from both ends until it starts and ends on a letter or a
/// digit, so "domani." is "domani" and "«Langola»" is "Langola".
fn trim_to_letters(text: &str, from: usize, to: usize) -> Option<(usize, usize)> {
    let slice = &text[from..to];
    let start = slice.find(|c: char| c.is_alphanumeric())?;
    let end = slice.rfind(|c: char| c.is_alphanumeric())?;
    let end = end + slice[end..].chars().next().map_or(0, char::len_utf8);
    Some((from + start, from + end))
}

// ---------------------------------------------------------------------------
// The three ways a word is written down here
// ---------------------------------------------------------------------------

/// Lower case, accents flattened, everything that is not a letter or a digit
/// dropped: "sull'angolo" and "sullangolo" are the same nine letters.
pub(crate) fn fold(word: &str) -> String {
    word.chars()
        .flat_map(|c| c.to_lowercase())
        .map(unaccent)
        .filter(|c| c.is_alphanumeric())
        .collect()
}

fn unaccent(c: char) -> char {
    match c {
        'à' | 'á' | 'â' | 'ä' | 'ã' | 'å' => 'a',
        'è' | 'é' | 'ê' | 'ë' => 'e',
        'ì' | 'í' | 'î' | 'ï' => 'i',
        'ò' | 'ó' | 'ô' | 'ö' | 'õ' => 'o',
        'ù' | 'ú' | 'û' | 'ü' => 'u',
        'ç' => 'c',
        'ñ' => 'n',
        other => other,
    }
}

/// The word as sound classes, with runs of the same class collapsed.
///
/// Letters that a listener confuses are one character here: b/v/p/f/w are all
/// `b`, d/t are `d`, c/g/k/q/x are `g`, s/z are `s`, m/n are `n`, l/r are `l`.
/// Every vowel is `a`, because which vowel was heard is the least reliable thing
/// in a mangled name — *Nongula*, *Nongulo* and *Langola* differ almost entirely
/// in their vowels. `h` disappears, as it does in Italian.
///
/// This is a deliberately crude stand-in for a phoneme table. It is crude in a
/// direction that is safe: it throws information away, and everything that reads
/// it also has to satisfy a rule that uses the spelling.
fn sounds(folded: &str) -> String {
    let mut out = String::with_capacity(folded.len());
    for ch in folded.chars() {
        let class = match ch {
            'h' => continue,
            'b' | 'v' | 'p' | 'f' | 'w' => 'b',
            'c' | 'g' | 'k' | 'q' | 'x' => 'g',
            'd' | 't' => 'd',
            's' | 'z' => 's',
            'm' | 'n' => 'n',
            'l' | 'r' => 'l',
            'j' | 'y' => 'a',
            c if c.is_numeric() => c,
            c if c.is_alphabetic() => 'a',
            c => c,
        };
        if !out.ends_with(class) {
            out.push(class);
        }
    }
    out
}

/// The consonant spine: [`sounds`] with the vowels taken out.
fn skeleton(sounds: &str) -> String {
    sounds.chars().filter(|c| *c != 'a').collect()
}

// ---------------------------------------------------------------------------
// Distances
// ---------------------------------------------------------------------------

/// Edit distance counting a swap of two neighbours as one mistake (Damerau's
/// optimal string alignment).
///
/// The swap matters here: half of what a decoder does to an unfamiliar name is
/// reorder its letters — "abel" for "able".
fn distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut grid = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for (i, row) in grid.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in grid[0].iter_mut().enumerate() {
        *cell = j;
    }
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            let mut best = (grid[i - 1][j] + 1)
                .min(grid[i][j - 1] + 1)
                .min(grid[i - 1][j - 1] + cost);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                best = best.min(grid[i - 2][j - 2] + 1);
            }
            grid[i][j] = best;
        }
    }
    grid[a.len()][b.len()]
}

/// How far `needle` is from the *end* of `haystack`, given that at least one
/// character of the haystack has to be left in front of it.
///
/// The same table as [`distance`], except that skipping the front of the
/// haystack is free — which is what makes "Langola", read against the tail of
/// "sullangolo", one edit away instead of four.
///
/// Both ends matter, and each is one clause of the table. The front is free
/// only from the second character on, so a haystack that simply *starts* with
/// the needle and continues ("milanese" against "milano") is not a match; and
/// the answer is read out of the last column rather than the smallest one, so
/// the needle has to reach the end of the haystack rather than sit somewhere in
/// its middle.
fn buried_in_tail(needle: &str, haystack: &str) -> usize {
    let needle: Vec<char> = needle.chars().collect();
    let haystack: Vec<char> = haystack.chars().collect();
    if needle.is_empty() || haystack.is_empty() {
        return needle.len().max(haystack.len());
    }
    // Row zero is the cost of consuming `j` haystack characters before the
    // match begins: free from the second character on, impossible at the first.
    let unreachable = needle.len() + haystack.len() + 1;
    let mut previous: Vec<usize> = (0..=haystack.len())
        .map(|j| if j == 0 { unreachable } else { 0 })
        .collect();
    let mut current = vec![0usize; haystack.len() + 1];
    for i in 1..=needle.len() {
        current[0] = unreachable;
        for j in 1..=haystack.len() {
            let cost = usize::from(needle[i - 1] != haystack[j - 1]);
            current[j] = (previous[j] + 1)
                .min(current[j - 1] + 1)
                .min(previous[j - 1] + cost)
                .min(unreachable);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[haystack.len()]
}

// ---------------------------------------------------------------------------
// Words nobody may rewrite
// ---------------------------------------------------------------------------

/// Is this an everyday word of a language Echo is likely to be listening to?
fn is_everyday(folded: &str) -> bool {
    EVERYDAY_WORDS.binary_search(&folded).is_ok()
}

/// The words the near-miss rules are forbidden to touch.
///
/// **This list is load-bearing, and it is not a dictionary.** It is the frequent
/// words of the three languages Echo meets most, written folded (lower case, no
/// accents), sorted so it can be searched by halving. Its job is narrow: the
/// rules above are close enough to reach ordinary language, and this is what
/// stops them. *lavabile* is two edits from *Langola*; *trenta* keeps *Ostri*'s
/// consonants; *felpa* keeps *lampo.dev*'s. Every one of those would be rewritten
/// without this list, and every one of them is a word somebody says.
///
/// Nothing shorter than [`SHORTEST_TOKEN`] needs to be here — those words are
/// never candidates in the first place — so the list starts at four letters.
///
/// It will never be complete, which is why it is one of several guards and not
/// the only one. Adding a word to it can only ever make Echo more careful.
const EVERYDAY_WORDS: &[&str] = &[
    "abile",
    "about",
    "above",
    "acqua",
    "adesso",
    "aesthetic",
    "after",
    "again",
    "against",
    "agree",
    "airliner",
    "allora",
    "allowable",
    "almost",
    "alors",
    "already",
    "also",
    "alti",
    "altri",
    "altro",
    "always",
    "amabile",
    "amico",
    "anche",
    "ancora",
    "andare",
    "another",
    "answer",
    "anything",
    "apart",
    "aperto",
    "appena",
    "april",
    "aprile",
    "around",
    "arrivare",
    "arrivato",
    "aspetta",
    "astride",
    "attention",
    "attimo",
    "aussi",
    "autre",
    "available",
    "avanti",
    "avere",
    "avoir",
    "avvocato",
    "back",
    "banca",
    "basta",
    "bathrobe",
    "beaucoup",
    "beautiful",
    "because",
    "been",
    "before",
    "being",
    "bella",
    "belle",
    "bello",
    "bene",
    "besoin",
    "better",
    "between",
    "bianca",
    "bianco",
    "bien",
    "bigger",
    "both",
    "bravo",
    "bring",
    "build",
    "buona",
    "buono",
    "button",
    "call",
    "called",
    "cambiare",
    "capable",
    "capably",
    "capito",
    "cara",
    "care",
    "carino",
    "caro",
    "casa",
    "cause",
    "cela",
    "cerca",
    "certo",
    "change",
    "check",
    "chez",
    "chiaro",
    "chiedere",
    "chose",
    "cinque",
    "citta",
    "classe",
    "clear",
    "cliente",
    "close",
    "code",
    "colore",
    "colpa",
    "come",
    "comme",
    "company",
    "comune",
    "conclude",
    "cosa",
    "cosi",
    "costi",
    "costo",
    "could",
    "count",
    "credo",
    "dans",
    "data",
    "date",
    "davanti",
    "dave",
    "days",
    "deja",
    "dentro",
    "deplorable",
    "detto",
    "deux",
    "devotion",
    "dice",
    "dietro",
    "difference",
    "different",
    "dire",
    "does",
    "doing",
    "dollar",
    "domanda",
    "domani",
    "done",
    "doorknob",
    "dopo",
    "dove",
    "dovere",
    "down",
    "drop",
    "each",
    "early",
    "easy",
    "elle",
    "else",
    "email",
    "emotion",
    "employable",
    "encore",
    "endeavor",
    "enigma",
    "enough",
    "entrare",
    "entre",
    "entro",
    "errore",
    "essere",
    "even",
    "every",
    "exactly",
    "example",
    "exclude",
    "face",
    "fact",
    "fare",
    "fatto",
    "favola",
    "felice",
    "felpa",
    "feltro",
    "fermare",
    "fine",
    "finire",
    "first",
    "fois",
    "follow",
    "forse",
    "found",
    "four",
    "from",
    "full",
    "function",
    "funziona",
    "gente",
    "gets",
    "getting",
    "gioco",
    "giorno",
    "give",
    "going",
    "good",
    "grande",
    "grani",
    "grazie",
    "great",
    "group",
    "guarda",
    "half",
    "hand",
    "happen",
    "hard",
    "have",
    "having",
    "hear",
    "help",
    "here",
    "high",
    "hold",
    "home",
    "hope",
    "hour",
    "house",
    "however",
    "humane",
    "idea",
    "important",
    "include",
    "insieme",
    "intention",
    "into",
    "invention",
    "issue",
    "junction",
    "just",
    "keep",
    "kind",
    "know",
    "large",
    "last",
    "late",
    "later",
    "lavabile",
    "lavabo",
    "lavare",
    "lavarle",
    "lavorare",
    "lavoro",
    "least",
    "leave",
    "left",
    "less",
    "letto",
    "levare",
    "level",
    "libera",
    "libero",
    "libro",
    "life",
    "like",
    "line",
    "lingua",
    "list",
    "little",
    "livable",
    "livello",
    "locale",
    "long",
    "look",
    "lovely",
    "made",
    "mail",
    "make",
    "making",
    "male",
    "mandare",
    "many",
    "matter",
    "mattina",
    "mean",
    "medieval",
    "meeting",
    "meglio",
    "mention",
    "mercato",
    "mese",
    "metaphor",
    "methodic",
    "mettere",
    "might",
    "milan",
    "mille",
    "minute",
    "minuto",
    "modo",
    "moins",
    "molto",
    "momento",
    "monday",
    "mondo",
    "money",
    "montare",
    "month",
    "more",
    "morphine",
    "most",
    "motion",
    "movable",
    "much",
    "must",
    "mutable",
    "name",
    "nation",
    "need",
    "negozio",
    "nessuno",
    "never",
    "next",
    "niente",
    "nirvana",
    "nome",
    "nonna",
    "nonno",
    "nostro",
    "notable",
    "notably",
    "notaio",
    "notevole",
    "nothing",
    "notte",
    "nous",
    "novella",
    "novembre",
    "number",
    "numero",
    "nuovo",
    "october",
    "office",
    "often",
    "once",
    "only",
    "open",
    "option",
    "order",
    "orthodox",
    "other",
    "outstrip",
    "ovale",
    "over",
    "pagina",
    "palazzo",
    "para",
    "parlare",
    "parlato",
    "parole",
    "part",
    "parte",
    "particolare",
    "pass",
    "passato",
    "pathetic",
    "pause",
    "peggio",
    "pensare",
    "pentola",
    "people",
    "perche",
    "perfume",
    "period",
    "persona",
    "pesante",
    "peut",
    "piano",
    "piccolo",
    "place",
    "plus",
    "point",
    "porta",
    "portale",
    "possible",
    "post",
    "potion",
    "pour",
    "prendere",
    "presently",
    "preventivo",
    "price",
    "prima",
    "primo",
    "probabile",
    "problem",
    "problema",
    "process",
    "product",
    "profane",
    "progetto",
    "project",
    "promotion",
    "pronto",
    "propane",
    "prossimo",
    "provable",
    "punto",
    "quando",
    "quanto",
    "quasi",
    "quattro",
    "quello",
    "questa",
    "queste",
    "questi",
    "questo",
    "quindi",
    "quotable",
    "ragazzo",
    "read",
    "ready",
    "real",
    "really",
    "reason",
    "record",
    "removable",
    "rendere",
    "retention",
    "retrieve",
    "rien",
    "rifare",
    "rifarle",
    "right",
    "ripetere",
    "ripeto",
    "riunione",
    "roba",
    "room",
    "rumore",
    "same",
    "sanction",
    "sapere",
    "sara",
    "scusa",
    "second",
    "secondo",
    "seem",
    "sembra",
    "sempre",
    "sense",
    "sentire",
    "senza",
    "sera",
    "serve",
    "servizio",
    "session",
    "settimana",
    "several",
    "share",
    "should",
    "show",
    "sicuro",
    "similar",
    "simple",
    "since",
    "slots",
    "small",
    "solo",
    "solvable",
    "some",
    "something",
    "sono",
    "sopra",
    "sorry",
    "sotto",
    "sous",
    "space",
    "speak",
    "spesso",
    "stanza",
    "start",
    "state",
    "stato",
    "stigma",
    "still",
    "stipula",
    "stop",
    "story",
    "strada",
    "strani",
    "strano",
    "street",
    "stretto",
    "subito",
    "such",
    "suitable",
    "sure",
    "system",
    "tabella",
    "table",
    "take",
    "talk",
    "tanto",
    "tardi",
    "tavolo",
    "team",
    "tell",
    "tempo",
    "tende",
    "terzo",
    "test",
    "text",
    "than",
    "thank",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "thing",
    "think",
    "this",
    "those",
    "though",
    "three",
    "through",
    "time",
    "tornare",
    "total",
    "tous",
    "tout",
    "town",
    "traffico",
    "train",
    "treni",
    "trenta",
    "tribune",
    "trop",
    "trovare",
    "true",
    "trying",
    "turbine",
    "tutti",
    "tutto",
    "under",
    "understand",
    "until",
    "used",
    "using",
    "vale",
    "valle",
    "valore",
    "valuable",
    "value",
    "variabile",
    "variabili",
    "vedere",
    "vediamo",
    "vendere",
    "venire",
    "verde",
    "vero",
    "verso",
    "very",
    "viale",
    "vicino",
    "viewable",
    "vita",
    "voce",
    "voglio",
    "voir",
    "volevo",
    "volta",
    "volte",
    "vous",
    "vuole",
    "want",
    "waterway",
    "week",
    "well",
    "were",
    "what",
    "whatever",
    "when",
    "where",
    "which",
    "while",
    "widower",
    "will",
    "with",
    "word",
    "work",
    "would",
    "write",
    "year",
    "your",
];

#[cfg(test)]
mod tests {
    use super::*;

    fn Langola() -> Glossary {
        Glossary::new(["Langola"])
    }

    fn corrected(glossary: &Glossary, line: &str) -> String {
        glossary
            .correct(line)
            .map_or_else(|| line.to_string(), |c| c.text)
    }

    // -------------------------------------------------------------------
    // The 2026-08-24 meeting, one row each
    // -------------------------------------------------------------------

    /// The whole reason this file exists: one product name, eight mentions, six
    /// spellings, none of them the name.
    #[test]
    fn every_way_that_meeting_spelled_Langola_comes_back_as_Langola() {
        let glossary = Langola();
        let table: &[&str] = &["sull'angolo", "lana gola", "Nongula", "Nongulo"];
        for mangled in table {
            let line = format!("Allora, {mangled} è quello che usiamo.");
            assert_eq!(
                corrected(&glossary, &line),
                "Allora, Langola è quello che usiamo.",
                "{mangled:?} was left as it was"
            );
        }
    }

    /// The two of the six this file cannot have, and why it is better not to.
    ///
    /// *Ingola* keeps three of Langola's sounds and *Langura* keeps its first
    /// three letters — and every rule loose enough to reach either of them
    /// reaches ordinary language on the way. "The first three letters and a
    /// similar sound" is *bianco* for a person called Bianchi, *classe* for
    /// Claude, *verde* for Vercel, *notte* for Notion and *milanese* for Milano;
    /// letting a sound-alike lose a whole consonant is *troppo* for Stripe,
    /// *locale* for Vercel and *traffico* for Anthropic. Two mangled mentions
    /// left standing is the price of not having any of that, and it is a good
    /// price.
    #[test]
    fn the_two_spellings_that_are_deliberately_out_of_reach() {
        let glossary = Langola();
        for mangled in ["Ingola", "Langura"] {
            let line = format!("Allora, {mangled} è quello che usiamo.");
            assert_eq!(glossary.correct(&line), None, "{mangled:?}");
        }
        // The words that would have come with them.
        for (entry, line) in [
            ("Bianchi", "il muro bianco della cucina"),
            ("Claude", "una bella classe di ragazzi"),
            ("Vercel", "il semaforo verde in fondo"),
            ("Notion", "ci vediamo di notte allora"),
            ("Milano", "una ditta milanese seria"),
            ("Stripe", "è troppo caro per noi"),
            ("Anthropic", "c'è traffico in tangenziale"),
        ] {
            assert_eq!(Glossary::new([entry]).correct(line), None, "{entry:?}");
        }
    }

    /// The other names that meeting lost, as far as this matcher goes. Each row
    /// is `(vocabulary entry, what was written, what should come back)`.
    #[test]
    fn the_other_names_of_that_meeting() {
        let table: &[(&str, &str, &str)] = &[
            // Trailing punctuation and capitals are the line's, not the match's.
            (
                "Ostri",
                "Siamo passati da Anastri",
                "Siamo passati da Ostri",
            ),
            (
                "Obsidara",
                "usiamo obsidera per i dati",
                "usiamo Obsidara per i dati",
            ),
            (
                "Feedharbour",
                "poi feed harbor per i post",
                "poi Feedharbour per i post",
            ),
            ("Vogliacasa", "quelli di vogli casa", "quelli di Vogliacasa"),
            // Casing, which every entry may fix however short it is.
            (
                "VMC",
                "il contratto vmc di ieri",
                "il contratto VMC di ieri",
            ),
        ];
        for (entry, line, expected) in table {
            let glossary = Glossary::new([*entry]);
            assert_eq!(corrected(&glossary, line), *expected, "entry {entry:?}");
        }
    }

    // -------------------------------------------------------------------
    // The half that matters more: what must never be touched
    // -------------------------------------------------------------------

    /// A glossary is a list of words somebody wants spelled right, not a licence
    /// to rewrite the language they were speaking.
    #[test]
    fn ordinary_words_near_an_entry_are_left_exactly_where_they_are() {
        let glossary = Langola();
        let table: &[&str] = &[
            // The one the whole design is aimed at: "amabile" is two edits from
            // "Langola" and it is an ordinary Italian adjective.
            "Un vino amabile, direi.",
            "È lavabile in lavatrice.",
            "Mi ha raccontato una favola.",
            "Questo è un livello notevole.",
            "Non è probabile che succeda.",
            "Devo levare il tavolo dalla valle.",
            "Il lavoro di novembre.",
            "Ha comprato una tabella globale.",
            "Guarda l'angolo del campo.",
            // Words too short to be looked at, whatever they sound like.
            "La vale poco.",
        ];
        for line in table {
            assert_eq!(
                glossary.correct(line),
                None,
                "{line:?} was rewritten; nothing in it is a product name"
            );
        }
    }

    /// The same question asked the way a real meeting asks it: a list somebody
    /// plausibly typed, against sentences somebody plausibly said.
    ///
    /// Every row here was rewritten by an earlier draft of this file, and each
    /// one is a different rule reaching too far — a consonant spine one edit
    /// wide, a name found at the front of a longer word, three shared letters
    /// standing in for a word. Nothing in the list is a licence to touch any of
    /// them.
    #[test]
    fn a_plausible_list_leaves_a_plausible_meeting_alone() {
        let glossary = Glossary::new([
            "Langola",
            "Obsidara",
            "Feedharbour",
            "Anthropic",
            "Claude",
            "Notion",
            "Linear",
            "Vercel",
            "Stripe",
            "Yomenia",
            "lampo.dev",
            "Ostri",
            "Milano",
            "Verona",
            "Marco",
            "Bianchi",
            "Airtable",
            "Sentry",
            "Netlify",
            "Grafana",
        ]);
        let table: &[&str] = &[
            // Italian, the language that meeting was in.
            "Devo levare le tende prima di lavarle.",
            "Bisogna rifarle tutte entro venerdì prossimo.",
            "Ci sono troppe variabili in gioco.",
            "La casa è ancora libera e il lavabo è nuovo.",
            "Il nostro cliente vuole vedere il preventivo.",
            "I costi sono alti e siamo arrivati tardi.",
            "Quel palazzo è stato ristrutturato per colpa della banca.",
            "Il mercato a Milano è fermo, ci vediamo domani mattina.",
            "Ho parlato con l'avvocato del portale del comune.",
            "Il ragazzo è bravo, ha comprato il feltro verde.",
            "La stipula del mutuo è troppo pesante.",
            "Manda una mail a tutti quelli di via Roma.",
            "Il locale è carino ma il livello di servizio è basso.",
            "Sono passato davanti al negozio, c'era traffico in tangenziale.",
            "Una ditta milanese e un prosciutto veronese.",
            "Il marchio registrato è di un muro bianco.",
            "Un attimo, ripeto: la favola della nonna.",
            "Non ho capito bene, è troppo grande per quella stanza.",
            // …and English, which the same meeting drifted into.
            "That was a really valuable session.",
            "The lovely part is the price, and the slots are available.",
            "Can you make the button bigger?",
            "I think we should roll it back on Monday.",
            "The train from Milan was late, whatever.",
            "It should be a suitable and movable solution.",
            "Please mention that in the invention section.",
            "We can include or exclude the beautiful part.",
        ];
        for line in table {
            assert_eq!(
                glossary.correct(line),
                None,
                "{line:?} was rewritten; every word in it is ordinary speech"
            );
        }
    }

    /// Two decoded words are read as one only when the line put nothing but a
    /// space between them.
    ///
    /// A two-word claim is written back over everything from the first word to
    /// the last, so a match that steps over a full stop deletes it and welds two
    /// sentences together — the transcript then says somebody said something
    /// they did not. "Non ho tempo. Casa mia è lontana" is the case: both halves
    /// of "Vogliacasa" are there, in order, with a sentence boundary between them.
    #[test]
    fn a_two_word_match_never_swallows_the_punctuation_between_them() {
        let glossary = Glossary::new(["Vogliacasa"]);
        for line in [
            "Non ho tempo. Casa mia è lontana.",
            "che tempo, casa mia",
            "Non ho tempo\nCasa mia è lontana.",
        ] {
            assert_eq!(glossary.correct(line), None, "{line:?}");
        }
        // With nothing but a space between them it is still the name.
        assert_eq!(
            corrected(&glossary, "quelli di voglia casa mi hanno chiamato"),
            "quelli di Vogliacasa mi hanno chiamato"
        );
    }

    /// A name at the *front* of a longer word is that word's own ending.
    ///
    /// This is how a surname or a town in the list eats its own adjectives, and
    /// it is the difference between "Anastri" — where something the decoder
    /// invented sits in front of Ostri — and "milanese", where Milano sits in
    /// front of an ordinary Italian suffix.
    #[test]
    fn a_word_that_merely_starts_with_an_entry_belongs_to_itself() {
        for (entry, line) in [
            ("Milano", "una ditta milanese"),
            ("Verona", "il prosciutto veronese"),
            ("Marco", "il marchio registrato"),
            ("Langola", "un contratto allowable"),
            ("Ostri", "hanno Ostri la pratica"),
        ] {
            assert_eq!(Glossary::new([entry]).correct(line), None, "{entry:?}");
        }
        assert_eq!(
            corrected(&Glossary::new(["Ostri"]), "Siamo passati da Anastri"),
            "Siamo passati da Ostri"
        );
    }

    /// The narrowest call in the file, and it is a real Italian sentence on one
    /// side and a 2026-08-24 mangling on the other. Both are an elided article
    /// followed by *ovale*; folded, "l'angolo" is one letter from "Langola".
    ///
    /// What separates them is what each rule asks for. A word wearing an
    /// everyday word behind its apostrophe may not be claimed by anything that
    /// merely *sounds* like the entry — but "sull'angolo" is long enough to hold
    /// "langolo" inside it, and the rule that asks for the entry in full is
    /// allowed to say so.
    #[test]
    fn the_oval_of_the_pitch_is_not_a_product_name_and_sullangolo_still_is() {
        let glossary = Langola();
        assert_eq!(glossary.correct("Guarda l'angolo del campo."), None);
        assert_eq!(glossary.correct("Il campo è un ovale."), None);
        assert_eq!(
            corrected(&glossary, "Quelli di sull'angolo ci hanno scritto."),
            "Quelli di Langola ci hanno scritto."
        );
    }

    /// A short entry has one power and it is not fuzzy matching. "BMC è più lui
    /// casa" was the 2026-08-24 mangling of "VMC / Voglia Mutui Casa" — and this
    /// is deliberately not fixed, because whatever would fix it would also
    /// rewrite every other three-letter word in the meeting.
    #[test]
    fn a_three_letter_entry_cannot_claim_short_words() {
        let glossary = Glossary::new(["VMC", "PEC", "IVA"]);
        for line in [
            "il BMC è voglio lui casa",
            "abbiamo la PEC nuova",
            "DMC oppure VNC",
            "time to go",
            "una IVA al 22%",
        ] {
            assert_eq!(glossary.correct(line), None, "{line:?}");
        }
        // The one thing it may do.
        assert_eq!(corrected(&glossary, "manda la pec"), "manda la PEC");
    }

    /// Nothing anywhere in the app may change because somebody has not typed a
    /// word yet. This is the invariant every lane leans on.
    #[test]
    fn an_empty_vocabulary_changes_nothing_at_all() {
        let empty = Glossary::new(Vec::<String>::new());
        assert!(empty.is_empty());
        assert_eq!(empty.prompt(), None);
        assert_eq!(empty.context(None), None);
        // The carried tail travels exactly as it did before this file existed.
        assert_eq!(
            empty.context(Some("e il secondo punto")),
            Some("e il secondo punto".to_string())
        );
        assert_eq!(empty.context(Some("   ")), None);
        for line in [
            "Ingola, Langura, sull'angolo, lana gola, Nongula, Nongulo.",
            "",
            "...",
        ] {
            assert_eq!(empty.correct(line), None, "{line:?}");
        }
        // And a list of blanks is an empty list, not a list of nothing.
        assert!(Glossary::new(["", "   ", "!!"]).is_empty());
    }

    // -------------------------------------------------------------------
    // What the line looks like afterwards
    // -------------------------------------------------------------------

    #[test]
    fn everything_that_was_not_corrected_survives_byte_for_byte() {
        let glossary = Glossary::new(["Langola", "Obsidara"]);
        let line = "  Quindi: «Nongula» — con obsidera, no?  ";
        let fixed = glossary.correct(line).expect("two names to put right");
        assert_eq!(fixed.text, "  Quindi: «Langola» — con Obsidara, no?  ");
        assert_eq!(
            fixed.changes,
            vec![
                Correction {
                    from: "Nongula".into(),
                    to: "Langola".into()
                },
                Correction {
                    from: "obsidera".into(),
                    to: "Obsidara".into()
                },
            ]
        );
    }

    /// A correction has to be reversible from what was recorded, or "undo" is a
    /// word with nothing behind it.
    #[test]
    fn what_was_changed_is_recorded_well_enough_to_put_back() {
        let glossary = Langola();
        let line = "Il team di lana gola ci ha scritto.";
        let fixed = glossary.correct(line).expect("a two-word mishearing");
        assert_eq!(fixed.text, "Il team di Langola ci ha scritto.");
        let mut back = fixed.text.clone();
        for change in &fixed.changes {
            back = back.replacen(&change.to, &change.from, 1);
        }
        assert_eq!(back, line);
    }

    #[test]
    fn a_word_already_spelled_right_is_not_a_correction() {
        let glossary = Langola();
        assert_eq!(glossary.correct("Langola è pronto"), None);
    }

    // -------------------------------------------------------------------
    // The prompt, and the cap on it
    // -------------------------------------------------------------------

    #[test]
    fn the_prompt_is_the_list_then_the_words_carried_across_a_cut() {
        let glossary = Glossary::new(["Langola", "Obsidara"]);
        assert_eq!(glossary.prompt().as_deref(), Some("Langola, Obsidara."));
        assert_eq!(
            glossary.context(Some("e il secondo punto")).as_deref(),
            // The carried tail is last: it is the sentence this audio continues.
            Some("Langola, Obsidara. e il secondo punto")
        );
    }

    /// Past its cap the prompt stops being a hint and starts being something
    /// whisper writes down, so the cap is not allowed to be approximate.
    #[test]
    fn the_cap_holds_and_never_cuts_an_entry_in_half() {
        let many: Vec<String> = (0..MAX_ENTRIES)
            .map(|i| format!("Nomeazienda{i:03}"))
            .collect();
        let glossary = Glossary::new(many.clone());
        let prompt = glossary.prompt().expect("a long list still prompts");
        assert!(
            prompt.chars().count() <= PROMPT_MAX_CHARS,
            "prompt was {} characters",
            prompt.chars().count()
        );
        // Every entry in the prompt is a whole entry, and they are the first
        // ones in the list — the same ones, every time.
        let kept: Vec<&str> = prompt.trim_end_matches('.').split(", ").collect();
        for (index, word) in kept.iter().enumerate() {
            assert_eq!(*word, many[index], "entry {index} arrived cut or reordered");
        }
        assert!(kept.len() < many.len(), "this test needs the cap to bite");
        assert_eq!(glossary.prompt(), Glossary::new(many).prompt());

        // The carried tail is budgeted for first: it is the half that cannot be
        // dropped, so a long list gives way to it rather than the other way
        // round.
        let tail = "e allora il punto principale della riunione di oggi era";
        let with_tail = glossary.context(Some(tail)).expect("context");
        assert!(with_tail.chars().count() <= PROMPT_MAX_CHARS);
        assert!(with_tail.ends_with(tail));
    }

    /// Everything about the vocabulary has to be a function of the list, or a
    /// meeting recorded twice would be prompted differently each time.
    #[test]
    fn the_same_list_always_produces_the_same_prompt_and_the_same_matches() {
        let words = ["Langola", "Langola", "  Langola  ", "Obsidara"];
        let once = Glossary::new(words);
        let twice = Glossary::new(words);
        assert_eq!(once.prompt(), twice.prompt());
        // The duplicates collapsed, first spelling wins.
        assert_eq!(once.len(), 2);
        assert_eq!(
            once.words().collect::<Vec<_>>(),
            vec!["Langola", "Obsidara"]
        );
    }

    #[test]
    fn a_list_longer_than_the_cap_on_entries_is_cut_at_the_cap() {
        let many: Vec<String> = (0..MAX_ENTRIES + 50).map(|i| format!("Word{i}")).collect();
        assert_eq!(Glossary::new(many).len(), MAX_ENTRIES);
    }

    // -------------------------------------------------------------------
    // The pieces underneath
    // -------------------------------------------------------------------

    #[test]
    fn a_word_is_read_with_its_apostrophes_and_dots_but_not_its_full_stop() {
        let spans = word_spans("sull'angolo, poi lampo.dev.");
        let text = "sull'angolo, poi lampo.dev.";
        let words: Vec<&str> = spans.iter().map(|(a, b)| &text[*a..*b]).collect();
        assert_eq!(words, vec!["sull'angolo", "poi", "lampo.dev"]);
        assert!(word_spans("... ?!").is_empty());
    }

    #[test]
    fn folding_flattens_case_accents_and_punctuation() {
        assert_eq!(fold("sull'angolo"), "sullangolo");
        assert_eq!(fold("Città"), "citta");
        assert_eq!(fold("lampo.dev"), "lampodev");
        assert_eq!(fold("«Langola»"), "Langola");
    }

    #[test]
    fn the_sound_classes_are_what_make_Nongula_and_Langola_the_same_shape() {
        assert_eq!(skeleton(&sounds("Langola")), "lbbl");
        assert_eq!(skeleton(&sounds("Nongula")), "nbbl");
        assert_eq!(skeleton(&sounds("lana gola")), "lbbl");
        assert_eq!(skeleton(&sounds("Ingola")), "bbl");
        // …and what keeps an ordinary word from being one: the guard list is
        // what saves these, not the shape.
        assert_eq!(skeleton(&sounds("lavabile")), "lbbl");
    }

    #[test]
    fn a_swap_of_two_letters_is_one_mistake_not_two() {
        assert_eq!(distance("abel", "able"), 1);
        assert_eq!(distance("Langola", "Langola"), 0);
        assert_eq!(distance("", "abc"), 3);
        assert_eq!(buried_in_tail("Langola", "sullangolo"), 1);
        assert_eq!(buried_in_tail("Ostri", "Anastri"), 1);
        assert!(buried_in_tail("Langola", "amabile") > 1);
        // Both ends are pinned: the entry may not start the word, and it may
        // not stop before the word does.
        assert!(buried_in_tail("milano", "milanese") > 1);
        assert!(buried_in_tail("Ostri", "tranquillo") > 1);
    }

    /// The guard list is searched by halving, so it has to be sorted — and it
    /// has to be written in the same form the matcher folds words into, or half
    /// of it would silently never match.
    #[test]
    fn the_everyday_word_list_is_sorted_folded_and_long_enough_to_matter() {
        for pair in EVERYDAY_WORDS.windows(2) {
            assert!(
                pair[0] < pair[1],
                "{:?} and {:?} are out of order",
                pair[0],
                pair[1]
            );
        }
        for word in EVERYDAY_WORDS {
            assert_eq!(&fold(word), word, "{word:?} is not written folded");
            assert!(
                word.chars().count() >= SHORTEST_TOKEN,
                "{word:?} is shorter than anything the matcher looks at"
            );
        }
    }

    /// An entry that is itself an everyday word may only fix its own casing:
    /// somebody who adds "Casa" to the list has not asked for every mention of
    /// a house to be capitalised, and certainly not for "cosa" to become it.
    #[test]
    fn an_entry_that_is_an_ordinary_word_only_fixes_its_own_spelling() {
        let glossary = Glossary::new(["Casa", "Meeting"]);
        assert_eq!(glossary.correct("che cosa fai"), None);
        assert_eq!(glossary.correct("il meating di oggi"), None);
        assert_eq!(
            corrected(&glossary, "il meeting di oggi"),
            "il Meeting di oggi"
        );
    }
}
