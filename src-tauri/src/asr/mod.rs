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
pub mod engine;
pub mod language;
pub mod models;
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
    /// Detected language for this stretch, with its own confidence.
    pub language: Option<String>,
    pub language_confidence: Option<f32>,
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
            language: self.language.clone(),
            avg_confidence: self.avg_confidence,
            revision: 1,
            is_final: true,
            model_name: self.model_name.clone(),
            model_revision: self.model_revision.clone(),
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

    #[test]
    fn a_transcription_becomes_a_final_segment_with_its_provenance() {
        let t = Transcription {
            channel: Channel::System,
            t_start_ms: 1_000,
            t_end_ms: 3_000,
            text: "hello".into(),
            language: Some("en".into()),
            language_confidence: Some(0.98),
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
