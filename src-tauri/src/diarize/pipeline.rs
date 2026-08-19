//! The offline pass, end to end.
//!
//! ```text
//! committed system-channel chunks
//!   └─ 10 s window, 5 s hop ─┐
//!                            ├─ segmentation → powerset decode → local turns
//!                            ├─ line up local labels with the previous window
//!                            └─ single-voice audio → fingerprint
//!   all fingerprints ─────────► agglomerative clustering (calibrated threshold)
//!   labels ───────────────────► per-person timeline → speakers rows
//!   final segments ───────────► speaker_id by time overlap, new revision
//! ```
//!
//! Three properties this pass has to keep, in order of importance:
//!
//! 1. **It never touches the audio.** Everything is read-only on disk; the pass
//!    can be run again, or abandoned, without losing anything (mantra 3).
//! 2. **It yields.** A recording starting outranks it absolutely, and it drops
//!    both models when it does (mantra 1).
//! 3. **The microphone is always "You".** Channel attribution is right about the
//!    person at the keyboard by construction, so this pass only ever decides who
//!    the *other* voices are.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::db::{repo, Db};
use crate::types::{Channel, Id, Speaker, TranscriptQuery};

use super::cluster::{self, ClusterItem};
use super::embedding::{self, Embedder};
use super::pcm::ChunkPcm;
use super::segmentation::{self, Segmenter};
use super::timeline::{self, Span};
use super::{DiarizationResult, DiarizeError, SpeakerTurn};

/// Cluster key of the microphone speaker. Stable for every meeting, which is
/// what lets a rename of "You" stick.
pub const SELF_CLUSTER_KEY: &str = "you";

/// Display name for the microphone speaker until the person renames it.
pub const SELF_DISPLAY_NAME: &str = "You";

/// A transcript line that falls in the silence between two turns is given to the
/// closest voice if it is this near, and left alone otherwise.
pub const NEAREST_TOLERANCE_MS: i64 = 400;

/// Progress in `0.0..=1.0`. Called often; must be cheap and must not block.
pub type ProgressFn = Arc<dyn Fn(f32) + Send + Sync>;
/// Returns true when a recording has started and the pass must step aside.
pub type YieldFn = Arc<dyn Fn() -> bool + Send + Sync>;

/// Cancellation, pre-emption and progress for one pass.
#[derive(Clone, Default)]
pub struct DiarizeControl {
    cancel: Arc<AtomicBool>,
    progress: Option<ProgressFn>,
    should_yield: Option<YieldFn>,
}

impl std::fmt::Debug for DiarizeControl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiarizeControl")
            .field("cancelled", &self.is_cancelled())
            .field("reports_progress", &self.progress.is_some())
            .field("can_yield", &self.should_yield.is_some())
            .finish()
    }
}

impl DiarizeControl {
    pub fn new() -> Self {
        Self::default()
    }

    /// Share the cancellation flag, so a command can stop the pass.
    pub fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel)
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    pub fn on_progress(mut self, f: ProgressFn) -> Self {
        self.progress = Some(f);
        self
    }

    /// Give the pass a way to notice a recording started.
    pub fn yield_when(mut self, f: YieldFn) -> Self {
        self.should_yield = Some(f);
        self
    }

    fn report(&self, fraction: f32) {
        if let Some(p) = &self.progress {
            p(fraction.clamp(0.0, 1.0));
        }
    }

    /// Stop here if we have been cancelled or a recording wants the machine.
    pub fn checkpoint(&self) -> Result<(), DiarizeError> {
        if self.is_cancelled() {
            return Err(DiarizeError::Cancelled);
        }
        if self.should_yield.as_ref().is_some_and(|f| f()) {
            return Err(DiarizeError::Yielded);
        }
        Ok(())
    }
}

/// How many cores to give the models.
///
/// Half of them, capped at four. This is background work: leaving headroom means
/// the machine stays responsive, and if a recording starts the pass steps aside
/// anyway.
fn inference_threads() -> usize {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2);
    (cores / 2).clamp(1, 4)
}

/// Run a synchronous, CPU-bound block without starving the async runtime.
///
/// Model inference is a few hundred milliseconds of solid arithmetic. On a
/// multi-threaded runtime tokio can hand the rest of the work to another worker
/// while this thread is busy; elsewhere (tests on a current-thread runtime) it
/// just runs.
fn run_blocking<T>(f: impl FnOnce() -> T) -> T {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(f)
        }
        _ => f(),
    }
}

/// What one sliding window produced. Local speaker indices are meaningless
/// outside the window they came from.
#[derive(Debug, Clone, Default)]
struct WindowResult {
    start_ms: i64,
    /// Per local speaker, the stretches it was talking, on the meeting clock.
    tracks: Vec<Vec<Span>>,
    /// Per local speaker, mean activation while it was talking.
    confidence: Vec<f32>,
    /// Per local speaker, its fingerprint's index, when it spoke enough alone.
    fingerprint: Vec<Option<usize>>,
    /// Per local speaker, the previous window's local speaker it continues.
    continues: Vec<Option<usize>>,
}

/// The canonical offline pass. See the module docs for the shape of it.
pub async fn refine(
    db: &Db,
    meeting_id: &str,
    segmenter_path: &std::path::Path,
    embedder_path: &std::path::Path,
    control: &DiarizeControl,
) -> Result<DiarizationResult, DiarizeError> {
    control.checkpoint()?;

    let chunks = repo::list_chunks(db, meeting_id, Some(Channel::System))
        .await
        .map_err(db_failed)?;
    let mut pcm = ChunkPcm::new(chunks);

    // A microphone-only meeting: channel attribution already knows everything
    // there is to know, so pin the transcript to "You" and stop. No model is
    // loaded, which is the point of checking first (mantra 1).
    if pcm.is_empty() {
        let speakers = pin_channel_speakers(db, meeting_id).await?;
        control.report(1.0);
        return Ok(DiarizationResult {
            turns: Vec::new(),
            speaker_count: 0,
            threshold: cluster::DISTANCE_THRESHOLD,
            speakers,
        });
    }

    let threads = inference_threads();
    let mut segmenter = Segmenter::load(segmenter_path, threads)?;
    let mut embedder = Embedder::load(embedder_path, threads)?;

    let covered = pcm.covered();
    let total_ms = pcm.total_ms();
    let step = segmentation::STEP_MS;
    let window_ms = segmentation::WINDOW_MS;
    let planned = ((total_ms + step - 1) / step).max(1);

    let mut items: Vec<ClusterItem> = Vec::new();
    let mut windows: Vec<WindowResult> = Vec::new();

    let mut start_ms = 0i64;
    let mut index = 0i64;
    while start_ms < total_ms {
        control.checkpoint()?;

        let window: Span = (start_ms, start_ms + window_ms);
        // Skip stretches of clock with no audio behind them at all.
        let has_audio = covered.iter().any(|&c| timeline::overlap_ms(window, c) > 0);
        if has_audio {
            let samples = pcm.window(start_ms, window_ms).await?;
            if segmentation::has_signal(&samples) {
                let result = analyse_window(
                    &mut segmenter,
                    &mut embedder,
                    &samples,
                    start_ms,
                    window_ms,
                    windows.last(),
                    &mut items,
                )?;
                windows.push(result);
            }
        }

        index += 1;
        // Reserve the last fifth of the bar for clustering and the database.
        control.report(0.8 * (index as f32 / planned as f32));
        start_ms += step;
    }

    // Both models are done. Drop them and the decoded audio before clustering,
    // so the heaviest part of this pass is not also the part holding two ONNX
    // sessions open (mantra 1).
    drop(segmenter);
    drop(embedder);
    pcm.release();
    control.checkpoint()?;
    control.report(0.85);

    let clustering = cluster::cluster(&items, cluster::DISTANCE_THRESHOLD);
    let labelled = label_windows(&windows, &clustering.labels);
    let (mut tracks, mut confidence) = build_tracks(&windows, &labelled, clustering.cluster_count);

    // Nothing was fingerprintable — every voice talked over every other one, or
    // in snatches too short to identify. Falling through to zero speakers would
    // be a regression on the live labels, which at least said "Speaker 1", so
    // keep exactly that: one remote voice covering the speech we did find.
    if tracks.is_empty() {
        let fallback = fallback_track(&windows, &covered);
        if !fallback.is_empty() {
            tracks = vec![fallback];
            confidence = vec![0.0];
        }
    }

    let turns = turns_from_tracks(&tracks, &confidence);

    control.checkpoint()?;
    control.report(0.9);

    let speakers = persist(db, meeting_id, &tracks).await?;
    control.report(1.0);

    Ok(DiarizationResult {
        speaker_count: tracks.len() as u32,
        turns,
        threshold: clustering.threshold,
        speakers,
    })
}

/// Segment one window, line its labels up with the previous one, and fingerprint
/// whoever spoke alone for long enough.
fn analyse_window(
    segmenter: &mut Segmenter,
    embedder: &mut Embedder,
    samples: &[f32],
    start_ms: i64,
    window_ms: i64,
    previous: Option<&WindowResult>,
    items: &mut Vec<ClusterItem>,
) -> Result<WindowResult, DiarizeError> {
    let activations = run_blocking(|| segmenter.run(samples))?;
    let tracks = segmentation::tracks_from_activations(&activations, start_ms, window_ms);
    let confidence: Vec<f32> = (0..activations.num_speakers)
        .map(|s| activations.confidence(s))
        .collect();

    let continues = match previous {
        Some(prev) => align_with_previous(&tracks, start_ms, prev),
        None => vec![None; tracks.len()],
    };

    let mut fingerprint = vec![None; tracks.len()];
    for local in 0..tracks.len() {
        // Only audio where this voice is alone: a fingerprint of two mixed
        // voices resembles neither of them.
        let others: Vec<Span> = timeline::union(
            &tracks
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != local)
                .map(|(_, t)| t.clone())
                .collect::<Vec<_>>(),
            0,
        );
        let alone = timeline::subtract(&tracks[local], &others);
        if !embedding::worth_embedding(&alone) {
            continue;
        }
        let audio = embedding::select_speech(samples, start_ms, &alone, embedding::MAX_EMBED_MS);
        let weight_ms = timeline::total_ms(&alone).min(embedding::MAX_EMBED_MS);
        if let Some(vector) = run_blocking(|| embedder.embed(&audio))? {
            fingerprint[local] = Some(items.len());
            items.push(ClusterItem {
                embedding: vector,
                weight_ms,
            });
        }
    }

    Ok(WindowResult {
        start_ms,
        tracks,
        confidence,
        fingerprint,
        continues,
    })
}

/// Pair this window's local speakers with the previous window's, over the audio
/// the two windows share.
///
/// Only the shared region counts. Outside it the two windows saw different audio,
/// so agreement there would be meaningless.
fn align_with_previous(
    tracks: &[Vec<Span>],
    start_ms: i64,
    prev: &WindowResult,
) -> Vec<Option<usize>> {
    let shared: Span = (start_ms, prev.start_ms + segmentation::WINDOW_MS);
    if timeline::span_ms(shared) == 0 {
        return vec![None; tracks.len()];
    }
    let overlap: Vec<Vec<i64>> = tracks
        .iter()
        .map(|mine| {
            prev.tracks
                .iter()
                .map(|theirs| shared_ms(mine, theirs, shared))
                .collect()
        })
        .collect();
    cluster::align_permutation(&overlap)
}

/// Time two tracks are both active, counted only inside `region`.
fn shared_ms(a: &[Span], b: &[Span], region: Span) -> i64 {
    let mut total = 0i64;
    for &x in a {
        if timeline::overlap_ms(x, region) == 0 {
            continue;
        }
        for &y in b {
            let both = (x.0.max(y.0).max(region.0), x.1.min(y.1).min(region.1));
            total += timeline::span_ms(both);
        }
    }
    total
}

/// Give every window-local speaker a cluster label.
///
/// A fingerprint answers directly. A local speaker who spoke too little to be
/// fingerprinted inherits the label of the speaker it continues from the
/// previous window — that is what the permutation alignment is for, and it is
/// how a person who says two words keeps their name.
fn label_windows(windows: &[WindowResult], labels: &[usize]) -> Vec<Vec<Option<usize>>> {
    let mut out: Vec<Vec<Option<usize>>> = Vec::with_capacity(windows.len());
    for (w, window) in windows.iter().enumerate() {
        let mut row = vec![None; window.tracks.len()];
        for (local, fp) in window.fingerprint.iter().enumerate() {
            if let Some(index) = fp {
                row[local] = labels.get(*index).copied();
            }
        }
        for (local, slot) in row.iter_mut().enumerate() {
            if slot.is_some() || window.tracks[local].is_empty() {
                continue;
            }
            let previous = w.checked_sub(1).and_then(|p| out.get(p));
            if let (Some(prev_local), Some(prev_row)) =
                (window.continues.get(local).copied().flatten(), previous)
            {
                *slot = prev_row.get(prev_local).copied().flatten();
            }
        }
        out.push(row);
    }
    out
}

/// Collect every window's contribution into one timeline per person, then
/// renumber so "Speaker 1" is the first voice heard.
fn build_tracks(
    windows: &[WindowResult],
    labelled: &[Vec<Option<usize>>],
    cluster_count: usize,
) -> (Vec<Vec<Span>>, Vec<f32>) {
    let mut raw: Vec<Vec<Span>> = vec![Vec::new(); cluster_count];
    let mut weight: Vec<f64> = vec![0.0; cluster_count];
    let mut confidence: Vec<f64> = vec![0.0; cluster_count];

    for (window, row) in windows.iter().zip(labelled) {
        for (local, label) in row.iter().enumerate() {
            let Some(label) = label else { continue };
            if *label >= cluster_count {
                continue;
            }
            let ms = timeline::total_ms(&window.tracks[local]) as f64;
            raw[*label].extend_from_slice(&window.tracks[local]);
            weight[*label] += ms;
            confidence[*label] +=
                ms * f64::from(window.confidence.get(local).copied().unwrap_or(0.0));
        }
    }

    for track in raw.iter_mut() {
        timeline::merge_spans(track, segmentation::MIN_GAP_MS);
        *track = timeline::drop_short(std::mem::take(track), segmentation::MIN_SPEECH_MS);
    }

    // Order by who spoke first, dropping labels that ended up with nothing.
    let mut order: Vec<usize> = (0..cluster_count).filter(|&i| !raw[i].is_empty()).collect();
    order.sort_by_key(|&i| raw[i][0].0);

    let tracks: Vec<Vec<Span>> = order.iter().map(|&i| raw[i].clone()).collect();
    let confidences: Vec<f32> = order
        .iter()
        .map(|&i| {
            if weight[i] > 0.0 {
                (confidence[i] / weight[i]) as f32
            } else {
                0.0
            }
        })
        .collect();
    (tracks, confidences)
}

/// One track covering every stretch of remote speech we found, for the case
/// where clustering could not name anybody. Falls back to the whole of the
/// recorded audio if even segmentation came up empty.
fn fallback_track(windows: &[WindowResult], covered: &[Span]) -> Vec<Span> {
    let all: Vec<Vec<Span>> = windows.iter().flat_map(|w| w.tracks.clone()).collect();
    let mut merged = timeline::union(&all, segmentation::MIN_GAP_MS);
    if merged.is_empty() {
        merged = covered.to_vec();
    }
    merged
}

fn turns_from_tracks(tracks: &[Vec<Span>], confidence: &[f32]) -> Vec<SpeakerTurn> {
    let mut turns: Vec<SpeakerTurn> = Vec::new();
    for (i, track) in tracks.iter().enumerate() {
        for &(start, end) in track {
            turns.push(SpeakerTurn {
                t_start_ms: start,
                t_end_ms: end,
                cluster: i as u32,
                confidence: confidence.get(i).copied().unwrap_or(0.0),
            });
        }
    }
    turns.sort_by_key(|t| (t.t_start_ms, t.cluster));
    turns
}

/// Create the speaker rows and re-point the transcript at them.
///
/// The microphone keeps "You" whatever the models decided. Every other final
/// segment goes to the person who was talking over most of it. Segments nothing
/// covers are left alone rather than guessed at, so an unattributed line stays
/// honestly unattributed.
async fn persist(
    db: &Db,
    meeting_id: &str,
    tracks: &[Vec<Span>],
) -> Result<Vec<Speaker>, DiarizeError> {
    let revision = repo::transcript_revision(db, meeting_id)
        .await
        .map_err(db_failed)?
        + 1;

    let me = repo::upsert_speaker(db, meeting_id, SELF_CLUSTER_KEY, SELF_DISPLAY_NAME, true)
        .await
        .map_err(db_failed)?;

    let mut remote_ids: Vec<Id> = Vec::with_capacity(tracks.len());
    for i in 0..tracks.len() {
        let speaker =
            repo::upsert_speaker(db, meeting_id, &cluster_key(i), &display_name(i), false)
                .await
                .map_err(db_failed)?;
        remote_ids.push(speaker.id);
    }

    let segments = repo::get_segments(
        db,
        &TranscriptQuery {
            meeting_id: meeting_id.to_string(),
            limit: Some(50_000),
            ..Default::default()
        },
    )
    .await
    .map_err(db_failed)?;

    fn push(into: &mut Vec<(Id, Vec<Id>)>, speaker_id: &Id, segment_id: Id) {
        match into.iter_mut().find(|(s, _)| s == speaker_id) {
            Some((_, ids)) => ids.push(segment_id),
            None => into.push((speaker_id.clone(), vec![segment_id])),
        }
    }

    let mut by_speaker: Vec<(Id, Vec<Id>)> = Vec::new();
    for segment in segments {
        let span: Span = (segment.t_start_ms, segment.t_end_ms);
        match segment.channel {
            Channel::Mic => push(&mut by_speaker, &me.id, segment.id),
            Channel::System => {
                let hit = timeline::assign_by_overlap(span, tracks)
                    .or_else(|| timeline::nearest_track(span, tracks, NEAREST_TOLERANCE_MS));
                if let Some(speaker_id) = hit.and_then(|i| remote_ids.get(i)).cloned() {
                    push(&mut by_speaker, &speaker_id, segment.id);
                }
            }
            Channel::Mixed => {}
        }
    }

    for (speaker_id, segment_ids) in by_speaker {
        repo::assign_speaker(db, &segment_ids, &speaker_id, revision)
            .await
            .map_err(db_failed)?;
    }
    repo::recompute_speaking_time(db, meeting_id)
        .await
        .map_err(db_failed)?;

    repo::list_speakers(db, meeting_id).await.map_err(db_failed)
}

/// Channel attribution on its own: "You" for the microphone, and one provisional
/// remote voice when there is a system channel. Costs nothing, always right
/// about the person at the keyboard.
pub async fn pin_channel_speakers(db: &Db, meeting_id: &str) -> Result<Vec<Speaker>, DiarizeError> {
    let me = repo::upsert_speaker(db, meeting_id, SELF_CLUSTER_KEY, SELF_DISPLAY_NAME, true)
        .await
        .map_err(db_failed)?;

    let has_system = !repo::list_chunks(db, meeting_id, Some(Channel::System))
        .await
        .map_err(db_failed)?
        .is_empty();
    let remote = if has_system {
        Some(
            repo::upsert_speaker(db, meeting_id, &cluster_key(0), &display_name(0), false)
                .await
                .map_err(db_failed)?,
        )
    } else {
        None
    };

    let segments = repo::get_segments(
        db,
        &TranscriptQuery {
            meeting_id: meeting_id.to_string(),
            limit: Some(50_000),
            include_partial: Some(true),
            ..Default::default()
        },
    )
    .await
    .map_err(db_failed)?;

    let revision = repo::transcript_revision(db, meeting_id)
        .await
        .map_err(db_failed)?
        .max(1);

    let mine: Vec<Id> = segments
        .iter()
        .filter(|s| s.channel == Channel::Mic && s.speaker_id.is_none())
        .map(|s| s.id.clone())
        .collect();
    repo::assign_speaker(db, &mine, &me.id, revision)
        .await
        .map_err(db_failed)?;

    if let Some(remote) = &remote {
        let theirs: Vec<Id> = segments
            .iter()
            .filter(|s| s.channel == Channel::System && s.speaker_id.is_none())
            .map(|s| s.id.clone())
            .collect();
        repo::assign_speaker(db, &theirs, &remote.id, revision)
            .await
            .map_err(db_failed)?;
    }

    repo::list_speakers(db, meeting_id).await.map_err(db_failed)
}

/// Stable per-meeting cluster key. Stable matters: re-running the pass must
/// reuse the same rows so a rename survives it.
pub fn cluster_key(index: usize) -> String {
    format!("speaker-{:02}", index + 1)
}

/// Zero jargon: the person sees "Speaker 1", never a cluster id (mantra 2).
pub fn display_name(index: usize) -> String {
    format!("Speaker {}", index + 1)
}

fn db_failed(err: crate::db::DbError) -> DiarizeError {
    DiarizeError::Failed(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(start_ms: i64, tracks: Vec<Vec<Span>>) -> WindowResult {
        let n = tracks.len();
        WindowResult {
            start_ms,
            tracks,
            confidence: vec![0.9; n],
            fingerprint: vec![None; n],
            continues: vec![None; n],
        }
    }

    #[test]
    fn cluster_keys_are_stable_and_names_carry_no_jargon() {
        assert_eq!(cluster_key(0), "speaker-01");
        assert_eq!(cluster_key(9), "speaker-10");
        assert_eq!(display_name(0), "Speaker 1");
        assert_eq!(SELF_DISPLAY_NAME, "You");
        for i in 0..5 {
            let name = display_name(i);
            for banned in ["cluster", "model", "embedding", "diariz", "ONNX"] {
                assert!(
                    !name.to_lowercase().contains(&banned.to_lowercase()),
                    "{name} leaks jargon"
                );
            }
        }
    }

    #[test]
    fn a_fingerprinted_speaker_takes_the_label_its_cluster_got() {
        let mut w = window(0, vec![vec![(0, 5_000)], vec![(5_000, 9_000)]]);
        w.fingerprint = vec![Some(0), Some(1)];
        let labelled = label_windows(&[w], &[3, 7]);
        assert_eq!(labelled, vec![vec![Some(3), Some(7)]]);
    }

    #[test]
    fn a_speaker_who_said_too_little_inherits_from_the_window_before() {
        let mut first = window(0, vec![vec![(0, 8_000)]]);
        first.fingerprint = vec![Some(0)];
        // The second window recognises the same voice as its local speaker 1,
        // but has no fingerprint for it.
        let mut second = window(5_000, vec![vec![(9_000, 10_000)], vec![(5_000, 8_000)]]);
        second.continues = vec![None, Some(0)];
        let labelled = label_windows(&[first, second], &[4]);
        assert_eq!(labelled[0], vec![Some(4)]);
        assert_eq!(labelled[1], vec![None, Some(4)]);
    }

    #[test]
    fn inheritance_does_not_reach_past_the_first_window() {
        let mut only = window(0, vec![vec![(0, 3_000)]]);
        only.continues = vec![Some(0)];
        let labelled = label_windows(&[only], &[]);
        assert_eq!(labelled, vec![vec![None]]);
    }

    #[test]
    fn people_are_numbered_in_the_order_they_first_speak() {
        // Cluster 1 speaks first, cluster 0 second.
        let mut w = window(0, vec![vec![(6_000, 9_000)], vec![(0, 4_000)]]);
        w.fingerprint = vec![Some(0), Some(1)];
        let labelled = label_windows(std::slice::from_ref(&w), &[0, 1]);
        let (tracks, conf) = build_tracks(&[w], &labelled, 2);
        assert_eq!(tracks, vec![vec![(0, 4_000)], vec![(6_000, 9_000)]]);
        assert_eq!(conf.len(), 2);
        assert!(conf.iter().all(|c| (*c - 0.9).abs() < 1e-5));
    }

    #[test]
    fn overlapping_windows_of_one_voice_become_a_single_stretch() {
        let mut a = window(0, vec![vec![(1_000, 9_000)]]);
        a.fingerprint = vec![Some(0)];
        let mut b = window(5_000, vec![vec![(6_000, 14_000)]]);
        b.fingerprint = vec![Some(1)];
        let windows = vec![a, b];
        // Both fingerprints landed in the same cluster.
        let labelled = label_windows(&windows, &[0, 0]);
        let (tracks, _) = build_tracks(&windows, &labelled, 1);
        assert_eq!(tracks, vec![vec![(1_000, 14_000)]]);
    }

    #[test]
    fn a_label_nothing_was_attributed_to_leaves_no_speaker_behind() {
        let mut w = window(0, vec![vec![(0, 4_000)]]);
        w.fingerprint = vec![Some(0)];
        let labelled = label_windows(std::slice::from_ref(&w), &[1]);
        // Cluster 0 exists in the clustering but nothing carries its label.
        let (tracks, _) = build_tracks(&[w], &labelled, 2);
        assert_eq!(tracks.len(), 1);
    }

    #[test]
    fn turns_come_out_in_time_order_with_their_cluster() {
        let tracks = vec![vec![(0, 1_000), (4_000, 5_000)], vec![(1_000, 4_000)]];
        let turns = turns_from_tracks(&tracks, &[0.8, 0.6]);
        assert_eq!(
            turns
                .iter()
                .map(|t| (t.t_start_ms, t.cluster))
                .collect::<Vec<_>>(),
            vec![(0, 0), (1_000, 1), (4_000, 0)]
        );
        assert!((turns[1].confidence - 0.6).abs() < 1e-6);
    }

    #[test]
    fn when_nobody_can_be_fingerprinted_one_remote_voice_is_kept() {
        // Two windows of speech, no fingerprints at all.
        let a = window(0, vec![vec![(1_000, 4_000)]]);
        let b = window(5_000, vec![vec![(3_800, 9_000)]]);
        let windows = vec![a, b];
        let track = fallback_track(&windows, &[(0, 30_000)]);
        assert_eq!(track, vec![(1_000, 9_000)]);
    }

    #[test]
    fn with_no_segmentation_at_all_the_fallback_is_the_recorded_audio() {
        assert_eq!(fallback_track(&[], &[(0, 30_000)]), vec![(0, 30_000)]);
        assert!(fallback_track(&[], &[]).is_empty());
    }

    #[test]
    fn shared_time_is_counted_only_inside_the_overlap_region() {
        let mine = vec![(4_000, 9_000)];
        let theirs = vec![(3_000, 6_000)];
        // Region covers the whole intersection.
        assert_eq!(shared_ms(&mine, &theirs, (0, 10_000)), 2_000);
        // Region clips it.
        assert_eq!(shared_ms(&mine, &theirs, (5_000, 10_000)), 1_000);
        // Region excludes it.
        assert_eq!(shared_ms(&mine, &theirs, (7_000, 10_000)), 0);
    }

    #[test]
    fn cancelling_stops_the_pass_at_the_next_checkpoint() {
        let control = DiarizeControl::new();
        assert!(control.checkpoint().is_ok());
        control.cancel();
        assert!(matches!(control.checkpoint(), Err(DiarizeError::Cancelled)));
    }

    #[test]
    fn a_recording_starting_makes_the_pass_step_aside() {
        let recording = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&recording);
        let control =
            DiarizeControl::new().yield_when(Arc::new(move || flag.load(Ordering::Relaxed)));
        assert!(control.checkpoint().is_ok());
        recording.store(true, Ordering::Relaxed);
        assert!(matches!(control.checkpoint(), Err(DiarizeError::Yielded)));
    }

    #[test]
    fn progress_is_reported_clamped_and_monotonically_usable() {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = Arc::clone(&seen);
        let control =
            DiarizeControl::new().on_progress(Arc::new(move |p| sink.lock().unwrap().push(p)));
        control.report(-1.0);
        control.report(0.5);
        control.report(9.0);
        assert_eq!(*seen.lock().unwrap(), vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn the_thread_budget_leaves_the_machine_room() {
        let t = inference_threads();
        assert!((1..=4).contains(&t), "{t} threads");
    }
}
