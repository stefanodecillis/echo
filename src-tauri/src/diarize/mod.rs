//! Who said what.
//!
//! Two layers, and the difference matters (DESIGN §1, review findings 9 and 31):
//!
//! 1. **Live, channel-based.** The microphone is "You". Everything on the system
//!    channel is one provisional remote speaker. This costs nothing and is
//!    always right about the person using the computer.
//! 2. **Offline refinement, canonical.** After the recording ends, segmentation
//!    plus speaker fingerprints split the recording into individual people and
//!    stabilise the labels. This pass owns the final answer and bumps the segment
//!    revision.
//!
//! Which recording gets split depends on what the meeting actually captured, and
//! this is the one place the two layers can disagree:
//!
//! * **With a system channel**, the far end is on its own recording. The
//!   microphone holds one known person, so layer 1's "You" stands and the pass
//!   only names the other voices.
//! * **With no system channel** — the person was on speakers, or in the same room
//!   as whoever they were talking to — everybody's voice is in the microphone
//!   recording. The pass splits *that*, the clusters become "Speaker 1..N", and
//!   **nothing is called "You"**: with one channel and no enrolled voice print
//!   Echo cannot tell which voice belongs to the person holding the laptop, and
//!   guessing would be a claim they then have to notice and undo. Layer 1's "You"
//!   row is pruned like any other row the final cut does not produce.
//!   [`pipeline`]'s module docs carry the detail.
//!
//! If the M0-S4 quality gate fails, v1 ships layer 1 live plus this pass
//! offline, and no live clustering — [`LiveClusterHook`] is the seam that would
//! change, and nothing in v1 implements it.
//!
//! ## How the offline pass works
//!
//! | step | module |
//! |---|---|
//! | read the committed chunks of the channel the voices arrived on, a window at a time | [`pcm`] |
//! | segment a 10 s window, decode the powerset output into turns | [`segmentation`] |
//! | log-mel features for the fingerprint network | [`features`] |
//! | fingerprint the audio where one person talks alone | [`embedding`] |
//! | group fingerprints into people; line up window-local labels | [`cluster`] |
//! | cut a transcript line that holds two voices in two | [`split`] |
//! | span arithmetic and the segment-to-speaker mapping | [`timeline`] |
//! | run all of it, write the result | [`pipeline`] |
//! | the same thing as a cancellable row in `jobs` | [`job`] |
//!
//! Everything runs through `ort` on CPU. Sessions are created for the pass and
//! dropped at the end (mantra 1) — nothing here stays loaded while Echo is idle,
//! and a recording starting stops the pass mid-window.
//!
//! ## How many people were in the meeting
//!
//! The pass answers this itself — the number of clusters it ends up with, plus
//! the person at the keyboard — and the answer is shown as "detected". The
//! person can correct it, and a correction is better evidence than any threshold
//! (DESIGN §1), so it re-runs the pass cut to that many voices — and says so
//! when the recording does not hold that many that can be told apart, rather
//! than making the difference up in speaker rows with nothing behind them
//! ([`cluster::ForcedOutcome`]).
//!
//! **How the pass decides, and why it changed.** It used to merge fingerprints
//! until the closest pair was further apart than one calibrated distance. That
//! distance was measured twice and still got a two-person conversation wrong in
//! the field on 2026-08-21, because the band of distances that gets a given
//! meeting right is only a few hundredths wide and a retranscription can slide
//! it. The count is now read out of the *shape* of the merge tree — the cut that
//! best explains the fingerprints — with the calibrated distance kept only as a
//! floor, a ceiling and a tie-break. [`cluster::CountChoice`] carries the
//! criterion, the evidence for it, and what it does not fix.
//!
//! **The number is the total, including whoever was at this computer.** That is
//! what "people in this meeting" means to a human, and the UI asks the question
//! in those words. What it turns into depends on the shape of the meeting, and
//! there are exactly two functions that do the turning:
//!
//! * [`remote_target`] — with a system channel. The microphone is a certain,
//!   separate speaker no clustering is involved in, so an override of 3 asks for
//!   2 remote voices when the microphone caught speech.
//! * [`mic_target`] — with no system channel. Every voice is in the one
//!   recording, the person's own included, so an override of 3 asks for 3
//!   microphone clusters. Nothing is subtracted, because nothing is certain.
//!
//! What survives a change of count, honestly stated:
//!
//! * **"You" survives as long as there is a system channel.** The microphone is
//!   then the person at the keyboard by construction; no count changes that, and a
//!   rename of it is keyed on a cluster key that never moves. On a
//!   microphone-only meeting there is no "You" row for a name to live on — see
//!   above — and a name typed onto one before the offline pass ran goes with the
//!   row.
//! * **A rename survives when the cluster does.** Speaker rows are keyed
//!   `speaker-01`, `speaker-02`, … in the order people first speak, so cutting a
//!   meeting from 4 voices to 3 keeps the first three rows and the names on them
//!   — but the *voice* behind `speaker-02` may not be the same person it was,
//!   because a different cut splits and fuses different pairs. A name on a
//!   cluster that no longer exists is dropped with the row
//!   ([`crate::db::repo::prune_speakers_except`]), because there is nothing left
//!   for it to be the name of.
//! * **A merge survives while both rows do**, and is released when the row it
//!   pointed at stops existing.
//!
//! That is the trade the alias/cluster-key machinery buys: re-running never
//! duplicates a speaker and never rewrites a transcript line, and in exchange a
//! rename is attached to a cluster rather than to a voice.

pub mod cluster;
pub mod embedding;
pub mod features;
pub mod job;
pub mod pcm;
pub mod people;
pub mod pipeline;
pub mod sample;
pub mod segmentation;
pub mod split;
pub mod timeline;

use std::path::Path;

use crate::db::Db;
use crate::types::{Id, Speaker};

pub use cluster::DISTANCE_THRESHOLD;
pub use pipeline::{
    cluster_key, display_name, mic_target, remote_target, DiarizeControl, SELF_CLUSTER_KEY,
    SELF_DISPLAY_NAME,
};

/// Fewest people a meeting can be said to have had in it. One: the person
/// talking to themselves, or reading a voice note back.
pub const MIN_PEOPLE: u32 = 1;

/// Most people Echo will try to tell apart in one meeting, counting the person
/// at this computer.
///
/// The same number as [`cluster::MAX_SPEAKERS`], and for the same reason: past a
/// dozen voices the fingerprints are noise, and a wrong-but-short list of chips
/// is far kinder than a wall of them. An all-hands with forty attendees is not a
/// thing per-speaker attribution can do anything useful with.
pub const MAX_PEOPLE: u32 = cluster::MAX_SPEAKERS as u32;

/// Pull a people count into the range Echo can actually work with.
///
/// A guard rail, not a validator: commands turn an out-of-range request down
/// with a sentence (so nobody's number is silently changed), and this is what
/// keeps a hand-edited database or an older row from reaching the clustering
/// loop with something absurd in it.
pub fn clamp_people(count: u32) -> u32 {
    count.clamp(MIN_PEOPLE, MAX_PEOPLE)
}

#[derive(Debug, thiserror::Error)]
pub enum DiarizeError {
    #[error("not implemented yet")]
    NotImplemented,
    #[error("Echo still needs to download what it uses to tell voices apart")]
    NotInstalled,
    #[error("could not load the speaker models: {0}")]
    Load(String),
    #[error("cancelled")]
    Cancelled,
    /// A recording started. Not a failure: the pass reads only finished audio
    /// off disk, so it can be redone from scratch whenever the machine is free
    /// again (mantra 1, and DESIGN §3 "recording preempts all").
    #[error("paused because a recording started")]
    Yielded,
    #[error("speaker analysis failed: {0}")]
    Failed(String),
    /// Two speakers that cannot be the same person — they belong to different
    /// meetings, or joining them would make a ring of aliases.
    #[error("those two speakers cannot be joined")]
    CannotMerge(String),
    /// Nobody's fault: this person never talks on their own for long enough to
    /// cut a recognisable clip out of. See [`sample`].
    #[error("there is no clear moment of that voice on its own")]
    NoVoiceSample,
    /// A different problem from [`Self::NoVoiceSample`], and one the person can
    /// act on: not one line of this meeting's transcript belongs to that voice,
    /// so there is nothing to look through for a clear moment in the first
    /// place. Telling somebody there is no clear moment of a voice that never
    /// said anything sends them looking for a recording fault that is not there.
    ///
    /// Since 2026-08-24 the pass no longer leaves rows like this behind (see
    /// [`pipeline::persist`]), but a merge, a rename or an older meeting can
    /// still produce one, and a person clicking on it deserves the true answer.
    #[error("no line of this meeting belongs to that voice")]
    NoLines,
    /// The words are still there and the audio is not — the recording was
    /// deleted, or the meeting was kept as a transcript only.
    #[error("that meeting's audio is no longer on this computer")]
    AudioForgotten,
}

/// Acceptance thresholds from spike M0-S4. Below these, live clustering does
/// not ship.
pub mod thresholds {
    /// Highest diarization error rate we accept on the fixture set.
    pub const MAX_DER: f32 = 0.25;
    /// Real-time factor on CPU. 0.3 means a one-hour meeting takes 18 minutes.
    pub const MAX_RTF: f32 = 0.3;
}

/// One stretch attributed to one speaker.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpeakerTurn {
    pub t_start_ms: i64,
    pub t_end_ms: i64,
    /// Cluster index within this meeting, before it gets a name.
    pub cluster: u32,
    pub confidence: f32,
}

impl SpeakerTurn {
    pub fn duration_ms(&self) -> i64 {
        (self.t_end_ms - self.t_start_ms).max(0)
    }
}

/// What the offline pass produced.
#[derive(Debug, Clone, Default)]
pub struct DiarizationResult {
    pub turns: Vec<SpeakerTurn>,
    /// How many distinct voices the pass separated out of the channel it read.
    ///
    /// With a system channel these are the remote voices and the microphone is
    /// not one of them, so the number of people in the meeting is one more. With
    /// no system channel the pass read the microphone instead and this *is*
    /// everybody. Either way [`Self::people_count`] is the number to show.
    pub speaker_count: u32,
    /// The distance the hierarchy was cut at, for diagnostics. Not a threshold
    /// any more in either mode: the automatic pass reads its count out of the
    /// shape of the tree ([`cluster::CountChoice`]) and a count the person gave
    /// us was never a distance at all.
    pub threshold: f32,
    /// How the automatic count was arrived at — the merge ladder, every count
    /// the tree could have been cut at, and what decided between them. `None`
    /// when the count was the person's own, or when there was no audio to
    /// cluster.
    ///
    /// Carried out of the pass so `examples/speakers_probe.rs` can print the
    /// decision and Settings → Advanced could show it. A count nobody can
    /// inspect is a count nobody can argue with.
    pub choice: Option<cluster::CountChoice>,
    /// The speaker rows as they now stand, ready for
    /// [`crate::events::SPEAKERS_UPDATED`].
    pub speakers: Vec<Speaker>,
    /// How many people were in the meeting, counting whoever was at this
    /// computer. This is the number the UI shows.
    pub people_count: u32,
    /// True when [`Self::people_count`] is the person's own correction rather
    /// than Echo's count.
    pub people_count_is_override: bool,
    /// How many people the pass was told to look for, counting whoever was at
    /// this computer. `None` when nobody corrected the count.
    pub voices_asked: Option<u32>,
    /// How many people the pass could actually tell apart, in the same units:
    /// the speaker rows it left behind, every one of them holding at least one
    /// line of the transcript.
    ///
    /// Below [`Self::voices_asked`] when the recording does not hold that many
    /// separable voices. That is a shortfall Echo says out loud rather than
    /// filling with rows nobody can hear (see
    /// [`cluster::ForcedOutcome`](crate::diarize::cluster::ForcedOutcome)).
    pub voices_found: u32,
}

/// Where live clustering would plug in, if the M0-S4 gate ever passes.
///
/// Deliberately unimplemented in v1. Live labels are channel-based and always
/// correct about "You"; guessing at individual remote speakers *while*
/// recording would load a second and third model during the one activity that
/// has absolute resource priority, which is exactly what mantra 1 forbids. The
/// offline pass runs seconds after the meeting ends and is canonical anyway.
pub trait LiveClusterHook: Send + Sync {
    /// Called with one finished utterance from the system channel. Returns the
    /// cluster it belongs to, or `None` for "not sure yet".
    fn on_utterance(&mut self, t_start_ms: i64, t_end_ms: i64, samples: &[f32]) -> Option<u32>;
}

/// Always `None` in v1. See [`LiveClusterHook`].
pub fn live_cluster_hook() -> Option<Box<dyn LiveClusterHook>> {
    None
}

/// The live, channel-based attribution. No models, no cost.
///
/// Creates "You" for the microphone and one provisional remote speaker for the
/// system channel, points any unattributed segments at them, and returns the
/// list.
pub async fn ensure_channel_speakers(
    db: &Db,
    meeting_id: &str,
) -> Result<Vec<Speaker>, DiarizeError> {
    pipeline::pin_channel_speakers(db, meeting_id).await
}

/// The canonical offline pass.
///
/// Reads the channel the voices arrived on from disk — the system channel, or the
/// microphone on a meeting that has no system channel — runs segmentation and
/// fingerprinting with a sliding window, decodes overlaps, clusters with a
/// calibrated threshold (or with the count the person gave us, if they gave us
/// one), then writes speakers and re-points segments at them with a fresh
/// revision.
///
/// Not cancellable on its own — use [`refine_speakers_with`] or [`job::run`] for
/// that. This shape exists for callers that already know the pass should run to
/// completion.
pub async fn refine_speakers(
    db: &Db,
    meeting_id: &str,
    segmenter_path: &Path,
    embedder_path: &Path,
) -> Result<DiarizationResult, DiarizeError> {
    refine_speakers_with(
        db,
        meeting_id,
        segmenter_path,
        embedder_path,
        &DiarizeControl::new(),
    )
    .await
}

/// [`refine_speakers`] with cancellation, pre-emption and progress attached.
pub async fn refine_speakers_with(
    db: &Db,
    meeting_id: &str,
    segmenter_path: &Path,
    embedder_path: &Path,
    control: &DiarizeControl,
) -> Result<DiarizationResult, DiarizeError> {
    pipeline::refine(db, meeting_id, segmenter_path, embedder_path, control).await
}

/// Rename a speaker everywhere. The alias graph means this never rewrites
/// segments.
pub async fn rename(db: &Db, speaker_id: &Id, display_name: &str) -> Result<(), DiarizeError> {
    let name = display_name.trim();
    if name.is_empty() {
        return Err(DiarizeError::Failed("a speaker needs a name".into()));
    }
    crate::db::repo::rename_speaker(db, speaker_id, name)
        .await
        .map_err(|e| DiarizeError::Failed(e.to_string()))
}

/// Merge two speakers by aliasing one to the other. Non-destructive and
/// reversible: `from` keeps its row and points at `into`, so [`unmerge`] puts it
/// back exactly as it was and no transcript line is ever rewritten.
pub async fn merge(db: &Db, from: &Id, into: &Id) -> Result<(), DiarizeError> {
    if from == into {
        return Ok(());
    }
    crate::db::repo::merge_speakers(db, from, into)
        .await
        .map_err(|e| match e {
            // "Different meetings" and "that would make a ring" are refusals,
            // not failures: the person gets told what is wrong, not that
            // something broke.
            crate::db::DbError::Invalid(why) => DiarizeError::CannotMerge(why),
            other => DiarizeError::Failed(other.to_string()),
        })
}

/// Undo a [`merge`].
pub async fn unmerge(db: &Db, speaker_id: &Id) -> Result<(), DiarizeError> {
    crate::db::repo::unmerge_speaker(db, speaker_id)
        .await
        .map_err(|e| DiarizeError::Failed(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::repo;
    use crate::types::{Channel, SegmentDraft};

    async fn db() -> Db {
        let db = crate::db::connect_in_memory().await.expect("in-memory db");
        crate::db::migrate(&db).await.expect("migrations");
        db
    }

    fn line(meeting_id: &str, channel: Channel, from: i64, to: i64) -> SegmentDraft {
        SegmentDraft {
            meeting_id: meeting_id.to_string(),
            t_start_ms: from,
            t_end_ms: to,
            channel,
            text: "hello".into(),
            revision: 1,
            is_final: true,
            ..Default::default()
        }
    }

    #[test]
    fn quality_gate_thresholds_are_the_ones_in_the_design() {
        assert_eq!(thresholds::MAX_DER, 0.25);
        assert_eq!(thresholds::MAX_RTF, 0.3);
    }

    #[test]
    fn a_turn_is_a_half_open_window() {
        let t = SpeakerTurn {
            t_start_ms: 0,
            t_end_ms: 1_000,
            cluster: 1,
            confidence: 0.9,
        };
        assert!(t.t_end_ms > t.t_start_ms);
        assert_eq!(t.duration_ms(), 1_000);
    }

    #[test]
    fn live_clustering_does_not_ship_in_v1() {
        assert!(live_cluster_hook().is_none());
    }

    #[tokio::test]
    async fn channel_attribution_pins_the_microphone_to_you() {
        let db = db().await;
        let meeting = repo::create_meeting(&db, "One to one", "/tmp/echo-test", None)
            .await
            .unwrap();
        repo::insert_segments(
            &db,
            &[
                line(&meeting.id, Channel::Mic, 0, 2_000),
                line(&meeting.id, Channel::Mic, 4_000, 6_000),
            ],
        )
        .await
        .unwrap();

        let speakers = ensure_channel_speakers(&db, &meeting.id).await.unwrap();
        // No system chunks, so no provisional remote speaker was invented.
        assert_eq!(speakers.len(), 1);
        assert_eq!(speakers[0].display_name, "You");
        assert!(speakers[0].is_self);

        let segments = repo::get_segments(
            &db,
            &crate::types::TranscriptQuery {
                meeting_id: meeting.id.clone(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert!(segments
            .iter()
            .all(|s| s.speaker_id.as_deref() == Some(speakers[0].id.as_str())));
    }

    #[tokio::test]
    async fn a_system_channel_gets_one_provisional_remote_speaker() {
        let db = db().await;
        let meeting = repo::create_meeting(&db, "Team sync", "/tmp/echo-test", None)
            .await
            .unwrap();
        let chunk = repo::insert_chunk(
            &db,
            &meeting.id,
            Channel::System,
            0,
            "/tmp/echo-test/system-0.flac",
            0,
            30_000,
        )
        .await
        .unwrap();
        repo::commit_chunk(&db, &chunk, 30_000).await.unwrap();
        repo::insert_segments(
            &db,
            &[
                line(&meeting.id, Channel::Mic, 0, 2_000),
                line(&meeting.id, Channel::System, 3_000, 9_000),
            ],
        )
        .await
        .unwrap();

        let speakers = ensure_channel_speakers(&db, &meeting.id).await.unwrap();
        assert_eq!(speakers.len(), 2);
        let names: Vec<&str> = speakers.iter().map(|s| s.display_name.as_str()).collect();
        assert!(names.contains(&"You"));
        assert!(names.contains(&"Speaker 1"));
        // "You" sorts first, which is what the transcript chips want.
        assert_eq!(speakers[0].display_name, "You");
    }

    #[tokio::test]
    async fn attribution_is_idempotent_so_a_double_click_changes_nothing() {
        let db = db().await;
        let meeting = repo::create_meeting(&db, "Interview", "/tmp/echo-test", None)
            .await
            .unwrap();
        let first = ensure_channel_speakers(&db, &meeting.id).await.unwrap();
        let second = ensure_channel_speakers(&db, &meeting.id).await.unwrap();
        assert_eq!(first.len(), second.len());
        assert_eq!(first[0].id, second[0].id);
    }

    #[tokio::test]
    async fn renaming_leaves_the_transcript_untouched() {
        let db = db().await;
        let meeting = repo::create_meeting(&db, "Retro", "/tmp/echo-test", None)
            .await
            .unwrap();
        repo::insert_segments(&db, &[line(&meeting.id, Channel::Mic, 0, 1_000)])
            .await
            .unwrap();
        let speakers = ensure_channel_speakers(&db, &meeting.id).await.unwrap();
        let before = repo::get_segments(
            &db,
            &crate::types::TranscriptQuery {
                meeting_id: meeting.id.clone(),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        rename(&db, &speakers[0].id, "  Ada  ").await.unwrap();
        let renamed = repo::get_speaker(&db, &speakers[0].id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(renamed.display_name, "Ada");

        let after = repo::get_segments(
            &db,
            &crate::types::TranscriptQuery {
                meeting_id: meeting.id.clone(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(before[0].revision, after[0].revision);
        assert_eq!(before[0].speaker_id, after[0].speaker_id);
    }

    #[tokio::test]
    async fn an_empty_name_is_refused_rather_than_stored() {
        let db = db().await;
        let meeting = repo::create_meeting(&db, "Standup", "/tmp/echo-test", None)
            .await
            .unwrap();
        let speakers = ensure_channel_speakers(&db, &meeting.id).await.unwrap();
        assert!(rename(&db, &speakers[0].id, "   ").await.is_err());
        let unchanged = repo::get_speaker(&db, &speakers[0].id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(unchanged.display_name, "You");
    }

    #[tokio::test]
    async fn merging_is_an_alias_and_survives_an_undo() {
        let db = db().await;
        let meeting = repo::create_meeting(&db, "Workshop", "/tmp/echo-test", None)
            .await
            .unwrap();
        let a = repo::upsert_speaker(&db, &meeting.id, "speaker-01", "Speaker 1", false)
            .await
            .unwrap();
        let b = repo::upsert_speaker(&db, &meeting.id, "speaker-02", "Speaker 2", false)
            .await
            .unwrap();

        merge(&db, &b.id, &a.id).await.unwrap();
        let merged = repo::get_speaker(&db, &b.id).await.unwrap().unwrap();
        assert_eq!(merged.alias_of.as_deref(), Some(a.id.as_str()));
        // The row is still there, which is what makes the undo possible.
        assert_eq!(merged.display_name, "Speaker 2");
        assert_eq!(
            repo::resolve_speaker(&db, &b.id).await.unwrap().unwrap().id,
            a.id
        );

        unmerge(&db, &b.id).await.unwrap();
        let restored = repo::get_speaker(&db, &b.id).await.unwrap().unwrap();
        assert!(restored.alias_of.is_none());
    }

    #[tokio::test]
    async fn merging_a_speaker_into_itself_is_a_no_op() {
        let db = db().await;
        let meeting = repo::create_meeting(&db, "Sync", "/tmp/echo-test", None)
            .await
            .unwrap();
        let a = repo::upsert_speaker(&db, &meeting.id, "speaker-01", "Speaker 1", false)
            .await
            .unwrap();
        merge(&db, &a.id, &a.id).await.unwrap();
        let same = repo::get_speaker(&db, &a.id).await.unwrap().unwrap();
        assert!(same.alias_of.is_none());
    }

    /// A meeting with no audio left on either channel — forgotten, or never
    /// committed. There is nothing to separate, so channel attribution is the
    /// whole answer and no model is touched.
    #[tokio::test]
    async fn a_meeting_with_no_audio_on_disk_needs_no_models_at_all() {
        let db = db().await;
        let meeting = repo::create_meeting(&db, "Voice note", "/tmp/echo-test", None)
            .await
            .unwrap();
        repo::insert_segments(&db, &[line(&meeting.id, Channel::Mic, 0, 5_000)])
            .await
            .unwrap();

        // Paths that do not exist: reaching for a model would fail loudly.
        let result = refine_speakers(
            &db,
            &meeting.id,
            Path::new("/nonexistent/segmenter.onnx"),
            Path::new("/nonexistent/fingerprints.onnx"),
        )
        .await
        .unwrap();

        assert_eq!(result.speaker_count, 0);
        assert!(result.turns.is_empty());
        assert_eq!(result.speakers.len(), 1);
        assert_eq!(result.speakers[0].display_name, "You");
    }

    /// A meeting recorded on speakers: microphone audio, no system channel. The
    /// other person's voice is in that recording, so the pass has to run on it —
    /// which means reaching for the models rather than pinning everything to
    /// "You" and stopping.
    #[tokio::test]
    async fn a_microphone_only_meeting_with_audio_still_gets_separated() {
        let db = db().await;
        let meeting = repo::create_meeting(&db, "Coffee", "/tmp/echo-test", None)
            .await
            .unwrap();
        let chunk = repo::insert_chunk(
            &db,
            &meeting.id,
            Channel::Mic,
            0,
            "/tmp/echo-test/mic-0.wav",
            0,
            30_000,
        )
        .await
        .unwrap();
        repo::commit_chunk(&db, &chunk, 30_000).await.unwrap();
        repo::insert_segments(&db, &[line(&meeting.id, Channel::Mic, 0, 5_000)])
            .await
            .unwrap();

        let err = refine_speakers(
            &db,
            &meeting.id,
            Path::new("/nonexistent/segmenter.onnx"),
            Path::new("/nonexistent/fingerprints.onnx"),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, DiarizeError::NotInstalled), "{err:?}");
    }

    #[tokio::test]
    async fn the_pass_reports_a_missing_download_as_something_to_fetch() {
        let db = db().await;
        let meeting = repo::create_meeting(&db, "Client call", "/tmp/echo-test", None)
            .await
            .unwrap();
        let chunk = repo::insert_chunk(
            &db,
            &meeting.id,
            Channel::System,
            0,
            "/tmp/echo-test/system-0.flac",
            0,
            30_000,
        )
        .await
        .unwrap();
        repo::commit_chunk(&db, &chunk, 30_000).await.unwrap();

        let err = refine_speakers(
            &db,
            &meeting.id,
            Path::new("/nonexistent/segmenter.onnx"),
            Path::new("/nonexistent/fingerprints.onnx"),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, DiarizeError::NotInstalled), "{err:?}");
    }

    /// End-to-end over real audio and real models. Ignored by default: it needs
    /// the two ONNX assets on disk, which CI does not download.
    #[tokio::test]
    #[ignore = "needs the downloaded speaker assets and a fixture recording"]
    async fn two_voices_in_a_fixture_recording_come_out_as_two_people() {
        let db = db().await;
        let (segmenter, embedder) = job::model_paths(&db).await.expect("speaker assets on disk");
        let meeting_id = std::env::var("ECHO_FIXTURE_MEETING")
            .expect("set ECHO_FIXTURE_MEETING to a meeting id in the dev database");

        let result = refine_speakers(&db, &meeting_id, &segmenter, &embedder)
            .await
            .expect("the pass runs");
        assert!(result.speaker_count >= 2, "{result:?}");
        assert!(!result.turns.is_empty());
        // The count came out of the tree, so the recorded number is the height
        // the tree was cut at — inside the bounds, not equal to either.
        let choice = result.choice.as_ref().expect("the automatic count decided");
        assert_eq!(choice.count as u32, result.speaker_count);
        assert!(result.threshold > 0.0 && result.threshold < cluster::FUSE_CEILING);
        // Turns never run backwards and never overlap within one person.
        for pair in result.turns.windows(2) {
            assert!(pair[0].t_start_ms <= pair[1].t_start_ms);
        }
    }
}
