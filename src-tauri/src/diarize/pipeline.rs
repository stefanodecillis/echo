//! The offline pass, end to end.
//!
//! ```text
//! committed chunks of the channel the voices arrived on
//!   └─ 10 s window, 5 s hop ─┐
//!                            ├─ segmentation → powerset decode → local turns
//!                            ├─ line up local labels with the previous window
//!                            └─ single-voice audio → fingerprint
//!   all fingerprints ─────────► agglomerative clustering — cut at the
//!                               calibrated threshold, or at the count the
//!                               person gave us if they gave us one
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
//! 3. **It never invents a name.** A voice Echo separated out but cannot identify
//!    is "Speaker N", never a guess at who it was.
//!
//! ## Which channel gets separated, and what "You" means
//!
//! There are two shapes of meeting, and the difference decides everything below.
//!
//! **A meeting with a system channel** — headphones, or any call Echo could tap.
//! The far end is on its own recording and the microphone holds exactly one
//! person: whoever is at the keyboard. Channel attribution is right about them by
//! construction, so the microphone is pinned to "You" and the separation pass
//! only ever decides who the *other* voices are. [`remote_target`] owns the
//! arithmetic that turns "there were N of us" into a number of remote voices.
//!
//! **A meeting with no system channel at all** — the person was on speakers, or
//! sitting across a table. Every voice in the room, theirs included, arrived
//! through the one microphone. Pinning that channel to "You" would claim a
//! two-person conversation was a monologue, so instead the same windows, the same
//! fingerprints and the same clustering run on the **microphone** channel, and:
//!
//! * The clusters are the people. A count the person gives maps straight through
//!   — N people means N microphone clusters, no subtraction — see
//!   [`mic_target`].
//! * **Nothing is called "You".** Echo has no way to know which of two voices in
//!   one recording is the person holding the laptop: there is no second channel
//!   to compare against and no enrolled voice print. Claiming to know would be a
//!   lie the person then has to notice and undo, so the honest output is
//!   "Speaker 1" and "Speaker 2" and a rename away from being right.
//! * The "You" row the live channel pass left behind is pruned with every other
//!   row this cut no longer produces, and any line still on it is released to
//!   unattributed rather than left labelled with a voice this pass does not
//!   believe in.
//!
//! Renaming works exactly the same in both shapes: rows are keyed by cluster key,
//! so `speaker-01` keeps the name the person typed on it across a re-run.

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

/// Which channel the pass is separating, and therefore what its labels mean.
///
/// See the module docs. This is decided once, before any model loads, from what
/// is actually on disk — never from a setting, because the setting cannot know
/// whether the person happened to be wearing headphones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    /// The far end has a channel of its own. The microphone is "You" by
    /// construction; the clustering names only the other voices.
    System,
    /// No system channel: everyone audible arrived through the microphone. The
    /// clustering names all of them and nobody is "You".
    MicOnly,
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

/// Everything the models had to say about one meeting, before any decision about
/// how many people were in it.
///
/// The expensive half of [`refine`] — every window segmented, every stretch of a
/// single voice fingerprinted — separated from the cheap half, which is one
/// clustering cut. Splitting them is what lets the calibration tool
/// (`examples/voices_fixture.rs`) try forty thresholds on one pass over the audio
/// instead of forty passes, and it means the number in
/// [`cluster::DISTANCE_THRESHOLD`] was measured through this code rather than
/// through a copy of it.
pub struct Scan {
    items: Vec<ClusterItem>,
    windows: Vec<WindowResult>,
    covered: Vec<Span>,
}

/// One clustering cut of a [`Scan`]: who spoke when, and how many people that is.
#[derive(Debug, Clone, Default)]
pub struct ScanCut {
    /// Per person, the stretches they were talking, ordered by who spoke first.
    pub tracks: Vec<Vec<Span>>,
    /// Per person, mean activation while they were talking.
    pub confidence: Vec<f32>,
    /// The threshold used, or — in fixed-count mode — the height the hierarchy
    /// was cut at.
    pub threshold: f32,
}

impl Scan {
    /// How many fingerprints the pass got out of the audio. Below one per person
    /// no threshold can possibly find them all.
    pub fn fingerprints(&self) -> usize {
        self.items.len()
    }

    /// Length of the fingerprints, or `None` when nothing was fingerprintable.
    /// Reading it off the vectors is how a calibration run proves it measured the
    /// network it thinks it did.
    pub fn fingerprint_dim(&self) -> Option<usize> {
        self.items.first().map(|i| i.embedding.len())
    }

    /// Every fingerprint, with the stretches of the meeting clock its audio came
    /// from.
    ///
    /// Only a calibration run wants this, and only because it already knows who
    /// was talking when: matching a fingerprint back to a span is what turns "the
    /// count came out right" into "these two voices are 0.83 apart and those two
    /// are 0.21" — which is the statement a threshold can actually be checked
    /// against. The clustering itself never looks at where a fingerprint came
    /// from; that is the whole point of clustering.
    ///
    /// The spans are the local speaker's whole activity in its window, which is a
    /// superset of the single-voice audio the fingerprint was computed from
    /// (`analyse_window` subtracts everyone else before embedding). For deciding
    /// *whose* voice a fingerprint is, the superset is the same answer.
    pub fn fingerprints_with_spans(&self) -> Vec<(&[f32], Vec<Span>)> {
        let mut out: Vec<(&[f32], Vec<Span>)> = self
            .items
            .iter()
            .map(|i| (i.embedding.as_slice(), Vec::new()))
            .collect();
        for window in &self.windows {
            for (local, slot) in window.fingerprint.iter().enumerate() {
                if let Some(index) = slot {
                    if let Some(entry) = out.get_mut(*index) {
                        entry.1 = window.tracks[local].clone();
                    }
                }
            }
        }
        out
    }

    /// Cut at a distance: the automatic pass. `None` for `target` means exactly
    /// that; `Some(k)` cuts at a count instead and ignores the threshold, which
    /// is what an override does.
    pub fn cut(&self, threshold: f32, target: Option<usize>) -> ScanCut {
        let clustering = match target {
            Some(k) => cluster::cluster_fixed(&self.items, k),
            None => cluster::cluster(&self.items, threshold),
        };
        let labelled = label_windows(&self.windows, &clustering.labels);
        let (mut tracks, mut confidence) =
            build_tracks(&self.windows, &labelled, clustering.cluster_count);

        // Nothing was fingerprintable — every voice talked over every other one,
        // or in snatches too short to identify. Falling through to zero speakers
        // would be a regression on the live labels, which at least said
        // "Speaker 1", so keep exactly that: one voice covering the speech we did
        // find.
        if tracks.is_empty() {
            let fallback = fallback_track(&self.windows, &self.covered);
            if !fallback.is_empty() {
                tracks = vec![fallback];
                confidence = vec![0.0];
            }
        }

        ScanCut {
            tracks,
            confidence,
            threshold: clustering.threshold,
        }
    }
}

/// The audio the pass will read, and what its labels will therefore mean.
///
/// A system channel means the far end recorded itself and the microphone is one
/// known person; no system channel means the whole conversation is in the
/// microphone recording and every voice in it has to be separated out (module
/// docs). `None` means there is no committed audio on either channel, and
/// therefore nothing to load a model for (mantra 1).
async fn voice_channel(db: &Db, meeting_id: &str) -> Result<Option<(Source, ChunkPcm)>, DiarizeError> {
    let system = ChunkPcm::new(
        repo::list_chunks(db, meeting_id, Some(Channel::System))
            .await
            .map_err(db_failed)?,
    );
    if !system.is_empty() {
        return Ok(Some((Source::System, system)));
    }
    let mic = ChunkPcm::new(
        repo::list_chunks(db, meeting_id, Some(Channel::Mic))
            .await
            .map_err(db_failed)?,
    );
    if mic.is_empty() {
        return Ok(None);
    }
    Ok(Some((Source::MicOnly, mic)))
}

/// Segment and fingerprint one meeting's audio. Writes nothing.
///
/// The expensive half of [`refine`], and the half a calibration run wants on its
/// own. Returns `None` when the meeting has no committed audio on either channel,
/// which is the case where no model is loaded at all (mantra 1).
pub async fn scan(
    db: &Db,
    meeting_id: &str,
    segmenter_path: &std::path::Path,
    embedder_path: &std::path::Path,
    control: &DiarizeControl,
) -> Result<Option<Scan>, DiarizeError> {
    control.checkpoint()?;

    let Some((_source, mut pcm)) = voice_channel(db, meeting_id).await? else {
        return Ok(None);
    };

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

    Ok(Some(Scan {
        items,
        windows,
        covered,
    }))
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

    let Some((source, _)) = voice_channel(db, meeting_id).await? else {
        // Nothing on disk to separate. Channel attribution is the whole answer,
        // and no model is loaded at all.
        let speakers = pin_channel_speakers(db, meeting_id).await?;
        let (people_count, people_count_is_override) = people_count(db, meeting_id).await?;
        control.report(1.0);
        return Ok(DiarizationResult {
            turns: Vec::new(),
            speaker_count: 0,
            threshold: cluster::DISTANCE_THRESHOLD,
            speakers,
            people_count,
            people_count_is_override,
        });
    };

    // How many voices to look for. `None` is the automatic pass and the common
    // case; `Some` is the person having corrected the count, and it is read
    // before any model loads so a correction can never be lost to a pass that
    // was already deciding for itself.
    let target = match source {
        Source::System => remote_cluster_target(db, meeting_id).await?,
        Source::MicOnly => mic_cluster_target(db, meeting_id).await?,
    };

    let Some(scanned) = scan(db, meeting_id, segmenter_path, embedder_path, control).await? else {
        return Err(DiarizeError::Failed(
            "the meeting's audio disappeared while the pass was starting".into(),
        ));
    };
    let cut = scanned.cut(cluster::DISTANCE_THRESHOLD, target);
    let turns = turns_from_tracks(&cut.tracks, &cut.confidence);

    control.checkpoint()?;
    control.report(0.9);

    let speakers = persist(db, meeting_id, &cut.tracks, source).await?;
    let (people_count, people_count_is_override) = people_count(db, meeting_id).await?;
    control.report(1.0);

    Ok(DiarizationResult {
        speaker_count: cut.tracks.len() as u32,
        turns,
        threshold: cut.threshold,
        speakers,
        people_count,
        people_count_is_override,
    })
}

/// How many remote voices this meeting's clustering should aim for, or `None`
/// for "let the calibrated threshold decide".
async fn remote_cluster_target(db: &Db, meeting_id: &str) -> Result<Option<usize>, DiarizeError> {
    let Some(people) = repo::speaker_count_override(db, meeting_id)
        .await
        .map_err(db_failed)?
    else {
        return Ok(None);
    };
    // "Did the person at this computer say anything" is a question about the
    // transcript, not about the devices: the microphone is always open, so
    // asking whether mic audio exists would subtract a person from every
    // meeting, including the ones the person only listened to.
    let mic_has_speech = repo::has_channel_speech(db, meeting_id, Channel::Mic)
        .await
        .map_err(db_failed)?;
    Ok(Some(remote_target(people, mic_has_speech)))
}

/// Turn "there were N people in this meeting" into "look for this many remote
/// voices".
///
/// The person counts themselves — "people in this meeting" includes the one
/// reading the question — and the microphone is a separate, certain speaker that
/// no clustering is involved in. So a count of N asks the clustering for N-1
/// when this computer's microphone caught speech, and for N when it did not
/// (a meeting the person only listened to).
///
/// Never zero: a meeting with remote audio in it has at least one remote voice,
/// whatever the arithmetic says, and returning zero would leave every remote
/// line unattributed to make a subtraction come out right.
pub fn remote_target(people: u32, mic_has_speech: bool) -> usize {
    let people = super::clamp_people(people);
    let remote = if mic_has_speech {
        people.saturating_sub(1)
    } else {
        people
    };
    remote.max(1) as usize
}

/// How many voices a microphone-only meeting's clustering should aim for, or
/// `None` for "let the calibrated threshold decide".
async fn mic_cluster_target(db: &Db, meeting_id: &str) -> Result<Option<usize>, DiarizeError> {
    Ok(repo::speaker_count_override(db, meeting_id)
        .await
        .map_err(db_failed)?
        .map(mic_target))
}

/// Turn "there were N people in this meeting" into "look for this many voices in
/// the microphone recording".
///
/// It maps straight through. With no system channel there is no separate,
/// certain microphone speaker to take off the total: the person at the keyboard
/// is one more voice in the same recording, to be found by the same clustering as
/// everybody else. Subtracting one here — the arithmetic
/// [`remote_target`] does, and correctly, for a meeting with two channels — would
/// answer "there were two of us" with a single cluster covering both voices.
pub fn mic_target(people: u32) -> usize {
    super::clamp_people(people) as usize
}

/// The number the UI shows, and whether it is the person's or Echo's.
///
/// An override wins whenever there is one — it is the person's answer to the
/// question the UI asked, and the control has to read back what they typed. The
/// pass produces *at most* that many voices and may produce fewer when the
/// meeting does not contain that much distinguishable speech; that is a quality
/// shortfall in the transcript, not a reason to argue with the person about how
/// many people were in their meeting.
///
/// Without an override it is Echo's own count: every speaker row that is not
/// merged into another one, which is "You" plus the voices the pass found.
async fn people_count(db: &Db, meeting_id: &str) -> Result<(u32, bool), DiarizeError> {
    repo::people_count(db, meeting_id).await.map_err(db_failed)
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
/// With a system channel the microphone keeps "You" whatever the models decided,
/// and every other final segment goes to the voice that was talking over most of
/// it. Microphone-only, there is no "You" to keep — the microphone segments are
/// the ones being split, and the row the live pass created for "You" is one of
/// the rows this cut no longer produces (module docs).
///
/// Segments nothing covers are left alone rather than guessed at, so an
/// unattributed line stays honestly unattributed.
///
/// **Reuse, never duplicate.** Rows are keyed by cluster key, so running the
/// pass again lands on the rows that are already there — which is what lets a
/// rename survive a re-run — and the last thing it does is forget the rows this
/// run no longer produced. Without that a meeting re-cut from four voices to two
/// would keep two ghost chips with nothing behind them, and Echo's own count of
/// how many people were in the meeting would still say four.
async fn persist(
    db: &Db,
    meeting_id: &str,
    tracks: &[Vec<Span>],
    source: Source,
) -> Result<Vec<Speaker>, DiarizeError> {
    let revision = repo::transcript_revision(db, meeting_id)
        .await
        .map_err(db_failed)?
        + 1;

    let me = match source {
        Source::System => Some(
            repo::upsert_speaker(db, meeting_id, SELF_CLUSTER_KEY, SELF_DISPLAY_NAME, true)
                .await
                .map_err(db_failed)?,
        ),
        Source::MicOnly => None,
    };

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

    // Which channel's lines the clustering is entitled to speak for. The other
    // one is either "You" (a microphone line, when the far end had its own
    // channel) or nothing at all.
    let separated = match source {
        Source::System => Channel::System,
        Source::MicOnly => Channel::Mic,
    };

    let mut by_speaker: Vec<(Id, Vec<Id>)> = Vec::new();
    for segment in segments {
        let span: Span = (segment.t_start_ms, segment.t_end_ms);
        if segment.channel == separated {
            let hit = timeline::assign_by_overlap(span, tracks)
                .or_else(|| timeline::nearest_track(span, tracks, NEAREST_TOLERANCE_MS));
            if let Some(speaker_id) = hit.and_then(|i| remote_ids.get(i)).cloned() {
                push(&mut by_speaker, &speaker_id, segment.id);
            }
        } else if segment.channel == Channel::Mic {
            if let Some(me) = &me {
                push(&mut by_speaker, &me.id, segment.id);
            }
        }
    }

    for (speaker_id, segment_ids) in by_speaker {
        repo::assign_speaker(db, &segment_ids, &speaker_id, revision)
            .await
            .map_err(db_failed)?;
    }

    // Whoever this run did not produce is no longer one of the people in this
    // meeting. Their rows go, which releases any line still pointing at them
    // back to unattributed rather than leaving it labelled with a voice the pass
    // no longer believes in.
    let mut keep: Vec<String> = Vec::with_capacity(tracks.len() + 1);
    if let Some(me) = &me {
        keep.push(me.cluster_key.clone());
    }
    keep.extend((0..tracks.len()).map(cluster_key));
    repo::prune_speakers_except(db, meeting_id, &keep)
        .await
        .map_err(db_failed)?;

    repo::recompute_speaking_time(db, meeting_id)
        .await
        .map_err(db_failed)?;

    repo::list_speakers(db, meeting_id).await.map_err(db_failed)
}

/// Channel attribution on its own: "You" for the microphone, and one provisional
/// remote voice when there is a system channel. Costs nothing.
///
/// This is the live answer, and it is the best one available while a recording is
/// happening. On a meeting that turns out to have no system channel it is also
/// *provisionally wrong* about the microphone — a second person in the room is in
/// that recording too — and [`refine`] is what corrects it, replacing "You" with
/// the voices it separates out. Live, there is nothing better to say: no model is
/// allowed to run during a recording (mantra 1).
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

    // -----------------------------------------------------------------------
    // "There were four of us"
    // -----------------------------------------------------------------------

    /// The arithmetic the whole feature turns on: the number the person gives
    /// counts them, and the microphone is not one of the voices being clustered.
    #[test]
    fn the_person_counts_themselves_so_the_microphone_comes_off_the_total() {
        assert_eq!(remote_target(4, true), 3);
        assert_eq!(remote_target(2, true), 1);
        // A meeting the person only listened to: nobody to subtract.
        assert_eq!(remote_target(4, false), 4);
        assert_eq!(remote_target(1, false), 1);
    }

    /// "Just me" on a meeting with remote audio in it. Zero remote voices would
    /// leave every remote line unattributed to make the subtraction come out
    /// right, which is worse than one voice too many.
    #[test]
    fn a_count_of_one_still_leaves_a_voice_for_the_far_end() {
        assert_eq!(remote_target(1, true), 1);
    }

    #[test]
    fn a_count_outside_what_echo_can_do_is_pulled_into_range() {
        assert_eq!(remote_target(0, false), 1);
        assert_eq!(
            remote_target(9_999, false),
            super::super::MAX_PEOPLE as usize
        );
        assert_eq!(
            remote_target(9_999, true),
            super::super::MAX_PEOPLE as usize - 1
        );
    }

    // -----------------------------------------------------------------------
    // Running the pass again
    // -----------------------------------------------------------------------

    async fn db() -> Db {
        let db = crate::db::connect_in_memory().await.expect("in-memory db");
        crate::db::migrate(&db).await.expect("migrations");
        db
    }

    fn remote_line(meeting_id: &str, from: i64, to: i64) -> crate::types::SegmentDraft {
        crate::types::SegmentDraft {
            meeting_id: meeting_id.to_string(),
            t_start_ms: from,
            t_end_ms: to,
            channel: Channel::System,
            text: "hello".into(),
            revision: 1,
            is_final: true,
            ..Default::default()
        }
    }

    /// The property the whole re-run rests on: a second pass lands on the rows
    /// the first one made, moves the transcript onto them, and leaves nothing
    /// behind. Not one duplicate row, and no ghost of a voice this cut does not
    /// believe in.
    #[tokio::test]
    async fn running_the_pass_again_with_fewer_people_reassigns_instead_of_duplicating() {
        let db = db().await;
        let meeting = repo::create_meeting(&db, "Team sync", "/tmp/echo-test", None)
            .await
            .unwrap();
        repo::insert_segments(
            &db,
            &[
                remote_line(&meeting.id, 0, 4_000),
                remote_line(&meeting.id, 5_000, 9_000),
                remote_line(&meeting.id, 10_000, 14_000),
            ],
        )
        .await
        .unwrap();

        // First pass: three voices.
        let three = vec![
            vec![(0i64, 4_000i64)],
            vec![(5_000, 9_000)],
            vec![(10_000, 14_000)],
        ];
        let after_first = persist(&db, &meeting.id, &three, Source::System)
            .await
            .unwrap();
        assert_eq!(after_first.len(), 4, "You and three voices");
        assert_eq!(repo::count_people(&db, &meeting.id).await.unwrap(), 4);

        // The person renames the second one before deciding there were fewer
        // people than that.
        let dana = after_first
            .iter()
            .find(|s| s.cluster_key == cluster_key(1))
            .expect("the second voice");
        repo::rename_speaker(&db, &dana.id, "Dana").await.unwrap();

        // Second pass, cut to one voice covering the same speech.
        let one = vec![vec![(0i64, 14_000i64)]];
        let after_second = persist(&db, &meeting.id, &one, Source::System)
            .await
            .unwrap();

        // Two rows, not five: the first pass's rows were reused and the two the
        // second pass no longer produces are gone.
        assert_eq!(after_second.len(), 2, "{after_second:?}");
        let keys: Vec<&str> = after_second
            .iter()
            .map(|s| s.cluster_key.as_str())
            .collect();
        assert_eq!(keys, vec![SELF_CLUSTER_KEY, cluster_key(0).as_str()]);
        assert_eq!(repo::count_people(&db, &meeting.id).await.unwrap(), 2);

        // Every line is on the surviving voice, and none is left pointing at a
        // row that no longer exists.
        let survivor = after_second
            .iter()
            .find(|s| s.cluster_key == cluster_key(0))
            .expect("the one voice");
        let segments = repo::get_segments(
            &db,
            &TranscriptQuery {
                meeting_id: meeting.id.clone(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(segments.len(), 3, "the words themselves are untouched");
        assert!(segments
            .iter()
            .all(|s| s.speaker_id.as_deref() == Some(survivor.id.as_str())));
        assert!(segments.iter().all(|s| s.text == "hello"));
    }

    /// A rename survives a re-run for the cluster it was on. The honest other
    /// half: a name on a voice the new cut does not produce goes with the row.
    #[tokio::test]
    async fn a_rename_survives_a_re_run_of_the_cluster_it_was_on() {
        let db = db().await;
        let meeting = repo::create_meeting(&db, "One to one", "/tmp/echo-test", None)
            .await
            .unwrap();
        repo::insert_segments(&db, &[remote_line(&meeting.id, 0, 4_000)])
            .await
            .unwrap();

        let first = persist(&db, &meeting.id, &[vec![(0i64, 4_000i64)]], Source::System)
            .await
            .unwrap();
        let voice = first
            .iter()
            .find(|s| s.cluster_key == cluster_key(0))
            .unwrap();
        repo::rename_speaker(&db, &voice.id, "Ada").await.unwrap();
        // And "You" is renamed too, which must survive everything.
        let me = first.iter().find(|s| s.is_self).unwrap();
        repo::rename_speaker(&db, &me.id, "Stefano").await.unwrap();

        let second = persist(&db, &meeting.id, &[vec![(0i64, 4_000i64)]], Source::System)
            .await
            .unwrap();
        assert_eq!(second.len(), 2);
        let names: Vec<&str> = second.iter().map(|s| s.display_name.as_str()).collect();
        assert!(names.contains(&"Ada"), "{names:?}");
        assert!(names.contains(&"Stefano"), "{names:?}");
        // Same rows, so the ids the UI is holding are still good.
        assert_eq!(second.iter().find(|s| s.is_self).unwrap().id, me.id);
    }

    /// The microphone is never one of the rows a cut can take away.
    #[tokio::test]
    async fn you_survive_a_cut_that_produces_no_remote_voices_at_all() {
        let db = db().await;
        let meeting = repo::create_meeting(&db, "Voice note", "/tmp/echo-test", None)
            .await
            .unwrap();
        repo::upsert_speaker(&db, &meeting.id, &cluster_key(0), "Speaker 1", false)
            .await
            .unwrap();

        let speakers = persist(&db, &meeting.id, &[], Source::System).await.unwrap();
        assert_eq!(speakers.len(), 1);
        assert_eq!(speakers[0].cluster_key, SELF_CLUSTER_KEY);
        assert!(speakers[0].is_self);
    }

    // -----------------------------------------------------------------------
    // Both voices came through the microphone
    // -----------------------------------------------------------------------

    fn mic_line(meeting_id: &str, from: i64, to: i64) -> crate::types::SegmentDraft {
        crate::types::SegmentDraft {
            channel: Channel::Mic,
            ..remote_line(meeting_id, from, to)
        }
    }

    /// The other half of the count arithmetic. Subtracting the microphone is
    /// right when the microphone is one known person on its own channel, and
    /// wrong when it is the whole room: "there were two of us" would then ask for
    /// one cluster and put both voices in it.
    #[test]
    fn on_one_channel_the_count_the_person_gives_is_the_number_of_voices_to_find() {
        assert_eq!(mic_target(2), 2);
        assert_eq!(mic_target(1), 1);
        assert_eq!(mic_target(5), 5);
        // Nothing to subtract, so the two arithmetics disagree by exactly one.
        assert_eq!(mic_target(2), remote_target(2, true) + 1);
    }

    #[test]
    fn a_count_outside_range_is_pulled_in_on_the_microphone_path_too() {
        assert_eq!(mic_target(0), super::super::MIN_PEOPLE as usize);
        assert_eq!(mic_target(9_999), super::super::MAX_PEOPLE as usize);
    }

    /// The whole point of the microphone path: two voices that both arrived
    /// through the one microphone come out as two people, and neither is claimed
    /// to be the person at the keyboard.
    #[tokio::test]
    async fn two_voices_on_the_microphone_become_two_speakers_and_no_one_is_you() {
        let db = db().await;
        let meeting = repo::create_meeting(&db, "Coffee", "/tmp/echo-test", None)
            .await
            .unwrap();
        repo::insert_segments(
            &db,
            &[
                mic_line(&meeting.id, 0, 4_000),
                mic_line(&meeting.id, 5_000, 9_000),
            ],
        )
        .await
        .unwrap();
        // The live channel pass already said "You" and put both lines on it.
        let live = pin_channel_speakers(&db, &meeting.id).await.unwrap();
        assert_eq!(live.len(), 1);
        assert_eq!(live[0].cluster_key, SELF_CLUSTER_KEY);

        let two = vec![vec![(0i64, 4_000i64)], vec![(5_000, 9_000)]];
        let speakers = persist(&db, &meeting.id, &two, Source::MicOnly)
            .await
            .unwrap();

        // Two people, both anonymous, and the live pass's "You" is gone rather
        // than left as a third chip with nothing behind it.
        let keys: Vec<&str> = speakers.iter().map(|s| s.cluster_key.as_str()).collect();
        assert_eq!(keys, vec![cluster_key(0).as_str(), cluster_key(1).as_str()]);
        assert!(
            speakers.iter().all(|s| !s.is_self),
            "Echo cannot know which voice is the person: {speakers:?}"
        );
        assert_eq!(
            speakers
                .iter()
                .map(|s| s.display_name.as_str())
                .collect::<Vec<_>>(),
            vec!["Speaker 1", "Speaker 2"]
        );
        // And that is the number the UI shows: N people is N microphone clusters.
        assert_eq!(repo::count_people(&db, &meeting.id).await.unwrap(), 2);

        // One line each, on the voice that was talking over it.
        let segments = repo::get_segments(
            &db,
            &TranscriptQuery {
                meeting_id: meeting.id.clone(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let first = speakers[0].id.clone();
        let second = speakers[1].id.clone();
        assert_eq!(segments[0].speaker_id.as_deref(), Some(first.as_str()));
        assert_eq!(segments[1].speaker_id.as_deref(), Some(second.as_str()));
        assert!(segments.iter().all(|s| s.text == "hello"));
    }

    /// A microphone-only line the clustering could not cover is released to
    /// unattributed rather than left on the "You" row it was pinned to live —
    /// the row is gone, so keeping the pointer would be a chip for a voice this
    /// pass does not believe in.
    #[tokio::test]
    async fn a_microphone_line_no_voice_covers_ends_up_unattributed() {
        let db = db().await;
        let meeting = repo::create_meeting(&db, "Workshop", "/tmp/echo-test", None)
            .await
            .unwrap();
        repo::insert_segments(
            &db,
            &[
                mic_line(&meeting.id, 0, 4_000),
                // Nowhere near the one track below.
                mic_line(&meeting.id, 60_000, 64_000),
            ],
        )
        .await
        .unwrap();
        pin_channel_speakers(&db, &meeting.id).await.unwrap();

        persist(&db, &meeting.id, &[vec![(0i64, 4_000i64)]], Source::MicOnly)
            .await
            .unwrap();

        let segments = repo::get_segments(
            &db,
            &TranscriptQuery {
                meeting_id: meeting.id.clone(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert!(segments[0].speaker_id.is_some());
        assert!(
            segments[1].speaker_id.is_none(),
            "a line nothing covers stays honestly unattributed"
        );
        assert_eq!(repo::count_people(&db, &meeting.id).await.unwrap(), 1);
    }

    /// Renames are keyed on the cluster key, which the microphone path uses
    /// exactly as the system path does — so naming the two voices in the room
    /// survives a re-run.
    #[tokio::test]
    async fn naming_the_voices_in_the_room_survives_a_re_run() {
        let db = db().await;
        let meeting = repo::create_meeting(&db, "Kitchen table", "/tmp/echo-test", None)
            .await
            .unwrap();
        repo::insert_segments(
            &db,
            &[
                mic_line(&meeting.id, 0, 4_000),
                mic_line(&meeting.id, 5_000, 9_000),
            ],
        )
        .await
        .unwrap();

        let two = vec![vec![(0i64, 4_000i64)], vec![(5_000, 9_000)]];
        let first = persist(&db, &meeting.id, &two, Source::MicOnly)
            .await
            .unwrap();
        repo::rename_speaker(&db, &first[0].id, "Stefano")
            .await
            .unwrap();
        repo::rename_speaker(&db, &first[1].id, "Giulia")
            .await
            .unwrap();

        let second = persist(&db, &meeting.id, &two, Source::MicOnly)
            .await
            .unwrap();
        assert_eq!(second.len(), 2);
        let names: Vec<&str> = second.iter().map(|s| s.display_name.as_str()).collect();
        assert!(names.contains(&"Stefano"), "{names:?}");
        assert!(names.contains(&"Giulia"), "{names:?}");
        // Same rows, so the ids the UI is holding are still good.
        assert_eq!(second[0].id, first[0].id);
        assert_eq!(second[1].id, first[1].id);
    }

    /// The system path is untouched by any of the above: a meeting with a system
    /// channel still pins the microphone to "You" and clusters only the far end.
    #[tokio::test]
    async fn with_a_system_channel_the_microphone_is_still_you() {
        let db = db().await;
        let meeting = repo::create_meeting(&db, "Client call", "/tmp/echo-test", None)
            .await
            .unwrap();
        repo::insert_segments(
            &db,
            &[
                mic_line(&meeting.id, 0, 4_000),
                remote_line(&meeting.id, 5_000, 9_000),
            ],
        )
        .await
        .unwrap();

        let speakers = persist(&db, &meeting.id, &[vec![(5_000i64, 9_000i64)]], Source::System)
            .await
            .unwrap();
        assert_eq!(speakers.len(), 2);
        let me = speakers.iter().find(|s| s.is_self).expect("You");
        assert_eq!(me.display_name, SELF_DISPLAY_NAME);

        let segments = repo::get_segments(
            &db,
            &TranscriptQuery {
                meeting_id: meeting.id.clone(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(segments[0].speaker_id.as_deref(), Some(me.id.as_str()));
        assert_ne!(segments[1].speaker_id.as_deref(), Some(me.id.as_str()));
        assert!(segments[1].speaker_id.is_some());
    }
}
