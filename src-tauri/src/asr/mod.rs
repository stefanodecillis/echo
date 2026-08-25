//! Turning audio into text.
//!
//! IMPLEMENTED-BY: asr agent (M3).
//!
//! * [`catalog`] is the data: which files Echo fetches, from where and under
//!   what licence — including the ones it has stopped wanting, which stay
//!   catalogued so they can be recognised and removed.
//! * [`models`] fetches and verifies what Echo needs to understand speech.
//! * [`reconcile`] keeps the weights on disk in step with the weights Echo
//!   wants, without ever leaving a person with nothing to record with.
//! * [`engine`] runs it: one loaded engine, one job at a time.
//! * [`language`] decides when the meeting's language is settled.
//! * [`catchup`] transcribes from disk whatever the live pass missed.
//! * [`phantom`] recognises the lines silence talked the decoder into, after
//!   the fact, and drops them.
//!
//! Lifecycle (mantra 1, as amended 2026-08-20): the engine loads the moment a
//! meeting is detected or started and stays resident for the whole conversation
//! *and* its follow-up jobs — while Echo is listening, transcript quality
//! outranks resource thrift. Residency is held by the session
//! ([`crate::session`] calls `hold_resident`), never by a timer over the last
//! decode; once nothing needs it, [`engine::IDLE_GRACE`] of quiet unloads it.
//! There is no setting behind that number. Idle Echo holds no weights.

pub mod catalog;
pub mod catchup;
pub mod catchup_bleed;
pub mod engine;
pub mod glossary;
pub mod language;
pub mod models;
pub mod phantom;
pub mod reconcile;

use crate::types::{Channel, Id};

#[derive(Debug, thiserror::Error)]
pub enum AsrError {
    #[error("not implemented yet")]
    NotImplemented,
    #[error("Echo still needs to download what it uses to understand speech")]
    NotInstalled,
    #[error("could not load what Echo uses to understand speech: {0}")]
    Load(String),
    #[error("transcription failed: {0}")]
    Transcribe(String),
    #[error("cancelled")]
    Cancelled,
    #[error("download failed: {0}")]
    Download(String),
    #[error("the downloaded file did not match the catalog")]
    IntegrityCheckFailed,
    #[error("not enough free space: {needed} bytes needed, {available} available")]
    NotEnoughSpace { needed: u64, available: u64 },
    /// Asked for something the catalog does not describe. Only reachable from a
    /// stale UI or a hand-edited database.
    #[error("unknown speech asset: {0}")]
    UnknownAsset(String),
    /// A catalogued asset that this platform does not use (the Apple encoder
    /// companion on Linux).
    #[error("{0} is not used on this system")]
    NotUsedHere(String),
    #[error("could not write to disk: {0}")]
    Io(String),
    #[error("database problem: {0}")]
    Db(#[from] crate::db::DbError),
    /// The queue was full and this live job was dropped so capture stays
    /// healthy. The audio is on disk; catch-up will get to it (mantra 3).
    #[error("the live queue is full; this audio will be transcribed from disk")]
    QueueFull,
    /// The engine thread is gone (shutdown, or it panicked).
    #[error("the speech engine is not running")]
    EngineGone,
}

impl From<std::io::Error> for AsrError {
    fn from(err: std::io::Error) -> Self {
        AsrError::Io(err.to_string())
    }
}

impl AsrError {
    /// True when this is not a failure at all: a live utterance was dropped to
    /// keep capture healthy, and the catch-up pass will transcribe it from disk
    /// (mantra 3). Callers should log it quietly, never show it to anybody.
    pub fn is_deferred_to_catchup(&self) -> bool {
        matches!(self, AsrError::QueueFull)
    }
}

/// One line the engine wrote inside a decode, with where it falls on the
/// meeting clock.
///
/// This is whisper's own segmentation, mapped through the window's origin by
/// [`crate::asr::engine::Engine::transcribe`] — so a packed catch-up window
/// comes back as the sentences it contained rather than as one wall of text
/// stamped with the window's start (see [`crate::asr::catchup`]).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TranscribedLine {
    /// On the meeting clock, never relative to the window.
    pub t_start_ms: i64,
    pub t_end_ms: i64,
    pub text: String,
    /// Mean token probability for this line alone.
    pub avg_confidence: Option<f32>,
}

/// One transcribed stretch of audio, before it becomes a database row.
#[derive(Debug, Clone, Default)]
pub struct Transcription {
    pub channel: Channel,
    pub t_start_ms: i64,
    pub t_end_ms: i64,
    pub text: String,
    /// The language this stretch was read in, with the confidence of the
    /// detection when there was one.
    pub language: Option<String>,
    pub language_confidence: Option<f32>,
    /// Whether [`Transcription::language`] is the meeting's standing answer
    /// handed down to this stretch rather than something heard in it.
    ///
    /// Handed down is the common case once a meeting has settled — seven
    /// stretches in eight are decoded on the hint without a detection of their
    /// own — and it is what the stretch was read in, so it stays here for the
    /// things that care about that. It must not reach the segment row, though:
    /// a language histogram built from copies of the pin can only ever
    /// re-confirm the pin, which is how a meeting a third of which was in
    /// English gets written up as if it had not been. See
    /// [`Transcription::observed_language`].
    pub language_inherited: bool,
    /// Mean token probability, used to flag shaky passages.
    pub avg_confidence: Option<f32>,
    /// Name and revision of what produced this, stored on the segment.
    pub model_name: Option<String>,
    pub model_revision: Option<String>,
    /// The lines this decode produced, in order, when timestamps were asked
    /// for. Empty for a caption (no timestamps) and for a decode that said
    /// nothing; [`Transcription::per_line`] is how to read it either way.
    pub lines: Vec<TranscribedLine>,
}

impl Transcription {
    /// This decode as one transcription per line, each one ready to become a
    /// row of its own.
    ///
    /// A window that came back as a single line — or as no lines at all,
    /// because the caller asked for no timestamps — is one transcription: the
    /// whole thing, exactly as it was before there were lines. Everything that
    /// is a property of the *decode* rather than of the words (the language it
    /// settled on, the weights that read it) is copied onto every line, because
    /// that is what it is true of.
    pub fn per_line(&self) -> Vec<Transcription> {
        if self.lines.len() < 2 {
            return vec![self.clone()];
        }
        self.lines
            .iter()
            .map(|line| Transcription {
                channel: self.channel,
                t_start_ms: line.t_start_ms,
                t_end_ms: line.t_end_ms,
                text: line.text.clone(),
                language: self.language.clone(),
                language_confidence: self.language_confidence,
                language_inherited: self.language_inherited,
                avg_confidence: line.avg_confidence.or(self.avg_confidence),
                model_name: self.model_name.clone(),
                model_revision: self.model_revision.clone(),
                lines: Vec::new(),
            })
            .collect()
    }

    /// Nothing worth storing: whisper.cpp emits empty strings and lone
    /// punctuation for silence.
    pub fn is_empty(&self) -> bool {
        let meaningful = self
            .text
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '\'' || *c == '’')
            .count();
        meaningful == 0
    }

    /// The language this stretch was *heard* to be in, which is what a segment
    /// row records.
    ///
    /// `None` when the stretch only inherited the meeting's answer. A row
    /// carrying a copy of the pin is not evidence about the meeting — it is the
    /// pin again — and everything downstream of these rows
    /// ([`crate::asr::language::spoken_in`], the recap's bilingual directive,
    /// the write-back that decides the meeting's language after catch-up) is
    /// asking what was *heard*.
    pub fn observed_language(&self) -> Option<String> {
        if self.language_inherited {
            return None;
        }
        self.language.clone()
    }

    /// The row this becomes. Speakers are attached later: live attribution is
    /// channel-based and the offline pass owns the final answer.
    pub fn to_draft(&self, meeting_id: &str) -> crate::types::SegmentDraft {
        crate::types::SegmentDraft {
            meeting_id: meeting_id.to_string(),
            t_start_ms: self.t_start_ms,
            t_end_ms: self.t_end_ms,
            channel: self.channel,
            speaker_id: None,
            text: self.text.clone(),
            language: self.observed_language(),
            avg_confidence: self.avg_confidence,
            revision: 1,
            is_final: true,
            model_name: self.model_name.clone(),
            model_revision: self.model_revision.clone(),
            corrections: Vec::new(),
        }
    }
}

/// A unit of work for the engine: one utterance from the speech detector, or
/// one window read back from disk during catch-up.
#[derive(Debug, Clone, Default)]
pub struct TranscribeJob {
    pub meeting_id: Id,
    /// Stable id so a live partial can be replaced by its final text.
    pub utterance_id: String,
    pub channel: Channel,
    pub t_start_ms: i64,
    /// 16 kHz mono.
    pub samples: Vec<f32>,
    /// Language hint from earlier in this meeting. `None` asks for detection.
    pub language_hint: Option<String>,
    /// Work the language out from this audio alone, whatever the meeting has
    /// settled on.
    ///
    /// `language_hint: None` on its own does *not* mean that: the engine keeps
    /// its own answer for the meeting and fills an empty hint in from it, which
    /// is the right default for every ordinary utterance. It is wrong for
    /// exactly one caller — catch-up reading a stretch a second time *because*
    /// the meeting's answer does not fit it (`catchup::transcribe_with_prior`).
    /// Without this the second
    /// reading is handed the very answer it is trying to get away from, and the
    /// escape hatch is a decode that can never reach a different result.
    pub detect_afresh: bool,
    /// How much of this window the speech detector actually called voice.
    ///
    /// `None` means nobody measured, and the whole window stands in for it.
    /// It matters for one thing: how much a language vote from this window is
    /// worth. A live utterance is padded at both ends and a catch-up window is
    /// mostly the pauses between the things that were said, so weighting a vote
    /// by the window is weighting it by silence.
    pub voiced_ms: Option<i64>,
    /// Emit partial results while decoding. Off for catch-up work.
    pub want_partials: bool,
    /// Live utterances are dropped when the queue is full; catch-up work waits
    /// its turn instead, because nothing else will pick it up.
    pub droppable: bool,
}

impl TranscribeJob {
    pub fn duration_ms(&self) -> i64 {
        (self.samples.len() as i64 * 1_000) / i64::from(crate::audio::TARGET_SAMPLE_RATE)
    }

    pub fn t_end_ms(&self) -> i64 {
        self.t_start_ms + self.duration_ms()
    }

    /// The speech in this window, as far as anything measured it — never more
    /// than the window itself, and the whole window when nothing did.
    pub fn voice_ms(&self) -> i64 {
        let duration = self.duration_ms().max(0);
        match self.voiced_ms {
            Some(measured) if measured > 0 => measured.min(duration),
            _ => duration,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_transcriptions_are_recognised_as_nothing() {
        let mut t = Transcription {
            text: " [BLANK_AUDIO] ".into(),
            ..Default::default()
        };
        // Bracketed markers do contain letters, so they are *not* empty; the
        // engine strips them before this point.
        assert!(!t.is_empty());

        t.text = "  ...  ".into();
        assert!(t.is_empty());
        t.text = " - ".into();
        assert!(t.is_empty());
        t.text = String::new();
        assert!(t.is_empty());
        t.text = "ok".into();
        assert!(!t.is_empty());
    }

    #[test]
    fn a_packed_window_comes_apart_into_the_lines_the_engine_wrote() {
        let window = Transcription {
            channel: Channel::Mic,
            t_start_ms: 60_000,
            t_end_ms: 88_000,
            text: "allora sì, d'accordo".into(),
            language: Some("it".into()),
            avg_confidence: Some(0.7),
            model_name: Some("weights".into()),
            model_revision: Some("rev1".into()),
            lines: vec![
                TranscribedLine {
                    t_start_ms: 60_000,
                    t_end_ms: 64_000,
                    text: "allora".into(),
                    avg_confidence: Some(0.9),
                },
                TranscribedLine {
                    t_start_ms: 70_000,
                    t_end_ms: 73_500,
                    text: "sì, d'accordo".into(),
                    avg_confidence: Some(0.5),
                },
            ],
            ..Default::default()
        };
        let lines = window.per_line();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].t_start_ms, 60_000);
        assert_eq!(lines[1].t_end_ms, 73_500);
        assert_eq!(lines[1].text, "sì, d'accordo");
        // Each line keeps its own confidence and the window's language and
        // provenance: those are true of the decode, not of the sentence.
        assert_eq!(lines[1].avg_confidence, Some(0.5));
        assert!(lines.iter().all(|l| l.language.as_deref() == Some("it")));
        assert!(lines
            .iter()
            .all(|l| l.model_revision.as_deref() == Some("rev1")));
        assert!(lines.iter().all(|l| l.channel == Channel::Mic));
        assert!(
            lines.iter().all(|l| l.lines.is_empty()),
            "a line does not contain lines"
        );

        // One line, or none at all — a caption asks for no timestamps — is the
        // whole thing, exactly as it was before there were lines.
        let single = Transcription {
            text: "ok".into(),
            t_start_ms: 10,
            t_end_ms: 20,
            ..Default::default()
        };
        assert_eq!(single.per_line().len(), 1);
        assert_eq!(single.per_line()[0].t_end_ms, 20);
    }

    #[test]
    fn a_job_knows_how_long_it_is() {
        let job = TranscribeJob {
            t_start_ms: 5_000,
            samples: vec![0.0; 16_000],
            ..Default::default()
        };
        assert_eq!(job.duration_ms(), 1_000);
        assert_eq!(job.t_end_ms(), 6_000);
    }

    /// What a segment row records is what these seconds were *heard* to be in.
    ///
    /// Seven settled stretches in eight are decoded on the meeting's own answer
    /// without a detection of their own, and a row carrying that answer is not
    /// evidence about the meeting — it is the pin written out again. Rows like
    /// that are what the meeting's language is later recomputed from, so keeping
    /// them makes that a closed loop, and drowns a language a real part of the
    /// meeting was in under copies of the one that was pinned.
    #[test]
    fn a_language_a_stretch_only_inherited_never_reaches_the_row() {
        let mut t = Transcription {
            channel: Channel::Mic,
            t_start_ms: 0,
            t_end_ms: 2_000,
            text: "hello".into(),
            language: Some("it".into()),
            language_inherited: true,
            ..Default::default()
        };
        assert_eq!(t.observed_language(), None);
        assert_eq!(t.to_draft("m1").language, None);

        // Heard in these seconds, so it is worth writing down.
        t.language_inherited = false;
        assert_eq!(t.observed_language().as_deref(), Some("it"));
        assert_eq!(t.to_draft("m1").language.as_deref(), Some("it"));
    }

    #[test]
    fn a_transcription_becomes_a_final_segment_with_its_provenance() {
        let t = Transcription {
            channel: Channel::System,
            t_start_ms: 1_000,
            t_end_ms: 3_000,
            text: "hello".into(),
            language: Some("en".into()),
            language_confidence: Some(0.98),
            language_inherited: false,
            avg_confidence: Some(0.87),
            // Deliberately the weights Echo has retired: a segment written
            // while an older model was still serving has to keep saying so
            // (see `crate::asr::reconcile`), so this is the case worth pinning.
            model_name: Some("whisper large-v3-turbo (ggml)".into()),
            model_revision: Some("abc123".into()),
            lines: Vec::new(),
        };
        let d = t.to_draft("m1");
        assert_eq!(d.meeting_id, "m1");
        assert_eq!(d.channel, Channel::System);
        assert!(d.is_final);
        assert_eq!(d.revision, 1);
        assert_eq!(d.language.as_deref(), Some("en"));
        assert_eq!(
            d.model_name.as_deref(),
            Some("whisper large-v3-turbo (ggml)")
        );
        assert!(d.speaker_id.is_none(), "speakers are attached later");
    }
}
