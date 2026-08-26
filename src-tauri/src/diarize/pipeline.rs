//! The offline pass, end to end.
//!
//! ```text
//! committed chunks of the channel the voices arrived on
//!   └─ 10 s window, 5 s hop ─┐
//!                            ├─ segmentation → powerset decode → local turns
//!                            ├─ line up local labels with the previous window
//!                            └─ single-voice audio → fingerprint
//!   all fingerprints ─────────► agglomerative clustering — the count read out
//!                               of the shape of the merge tree, or the count
//!                               the person gave us if they gave us one
//!   labels ───────────────────► per-person timeline → speakers rows
//!   final segments ───────────► lines holding two voices cut at the turn
//!                               boundary, then speaker_id by time overlap,
//!                               new revision
//! ```
//!
//! Three properties this pass has to keep, in order of importance:
//!
//! 1. **It never touches the audio.** Everything is read-only on disk; the pass
//!    can be run again, or abandoned, without losing anything (mantra 3).
//! 2. **It yields.** A recording starting outranks it absolutely, and it drops
//!    both models when it does (mantra 1). What it worked out first is written
//!    down ([`super::scan_cache`]), so stepping aside costs the meeting a delay
//!    and not the work.
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
//!
//! ## Lines that hold two voices
//!
//! This pass is also the only place that can fix a transcript line holding two
//! people. The speech engine decides where a line ends from what the audio sounds
//! like, so two people answering each other inside four hundred milliseconds come
//! out as one row; attributing that row whole gives one of them the other's
//! words. Because this pass holds both the turns and the transcript, it cuts the
//! row at the turn boundary before attributing anything — [`split`] owns the rule
//! for when that is honest and what it approximates when it is.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::db::{repo, Db};
use crate::types::{Channel, Id, Speaker, TranscriptQuery};

use super::cluster::{self, ClusterItem};
use super::embedding::{self, Embedder};
use super::pcm::ChunkPcm;
use super::people;
use super::scan_cache;
use super::segmentation::{self, Segmenter};
use super::split;
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

/// Why a step that had to be interruptible stopped.
///
/// Asked of the control after the fact, because the clustering only reports
/// *that* it was told to stop. If the reason has already gone away — a yield
/// closure that has since changed its mind — this still parks rather than
/// carrying on, because the work that was in flight has already been dropped.
fn stopped(control: &DiarizeControl) -> DiarizeError {
    control.checkpoint().err().unwrap_or(DiarizeError::Yielded)
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

impl Source {
    /// For the log only. Never shown to anybody (mantra 2).
    fn as_str(self) -> &'static str {
        match self {
            Source::System => "system",
            Source::MicOnly => "microphone-only",
        }
    }
}

/// What one sliding window produced. Local speaker indices are meaningless
/// outside the window they came from.
///
/// Serialisable, and visible to [`super::scan_cache`], because a park keeps
/// these rather than throwing the models' work away.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct WindowResult {
    pub(crate) start_ms: i64,
    /// Per local speaker, the stretches it was talking, on the meeting clock.
    pub(crate) tracks: Vec<Vec<Span>>,
    /// Per local speaker, mean activation while it was talking.
    pub(crate) confidence: Vec<f32>,
    /// Per local speaker, its fingerprint's index, when it spoke enough alone.
    pub(crate) fingerprint: Vec<Option<usize>>,
    /// Per local speaker, the previous window's local speaker it continues.
    pub(crate) continues: Vec<Option<usize>>,
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

/// One clustering cut of a [`Scan`]: who spoke when, how many people that is,
/// and which of them Echo already knows.
#[derive(Debug, Clone, Default)]
pub struct ScanCut {
    /// Per person, the stretches they were talking, ordered by who spoke first.
    pub tracks: Vec<Vec<Span>>,
    /// Per person, mean activation while they were talking.
    pub confidence: Vec<f32>,
    /// The height the hierarchy was cut at, or — for [`Scan::cut`], which is
    /// swept by the calibration tools — the distance it was cut at, which is the
    /// same number.
    pub threshold: f32,
    /// How the automatic count was arrived at, when it was the automatic count
    /// that arrived at it. `None` for a count the person gave us and for a cut
    /// at a distance a sweep chose.
    pub choice: Option<cluster::CountChoice>,
    /// What a count the person gave us actually came back with — and whether it
    /// came back short. `None` for the automatic count and for a cut at a
    /// distance a sweep chose.
    ///
    /// The counterpart of [`Self::choice`]: one of the two is always `None`,
    /// because a cut is either Echo's decision or the person's.
    pub forced: Option<cluster::ForcedOutcome>,
    /// Per person, the voice print of everything they said in this meeting.
    ///
    /// Stored on the speaker row by [`persist`], which is what makes "this
    /// unnamed voice has now been in four meetings" and "does this voice belong
    /// to somebody enrolled last week" answerable without reading a second of
    /// audio back (DESIGN §1).
    pub centroids: Vec<Vec<f32>>,
    /// Per person, the known person this voice turned out to be — or might be.
    /// Filled by [`Scan::cut_guided`] for the voices recognised before
    /// clustering and by [`people::match_remaining`] for the rest.
    pub matches: Vec<Option<people::Match>>,
}

impl Scan {
    /// How many fingerprints the pass got out of the audio. Below one per person
    /// no threshold can possibly find them all.
    pub fn fingerprints(&self) -> usize {
        self.items.len()
    }

    /// Speech behind all the fingerprints together. The mass the clustering has
    /// to divide up, which is what makes a per-cluster figure in the log mean
    /// something.
    pub fn fingerprinted_ms(&self) -> i64 {
        self.items.iter().map(|i| i.weight_ms.max(0)).sum()
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

    /// Which fingerprints belong to somebody Echo already knows, before
    /// anything is counted or clustered.
    ///
    /// This is the step that makes enrollment worth having (DESIGN §1: "enrolled
    /// voices are matched BEFORE the count question, so blind counting applies
    /// only to strangers"). A meeting with two known people and one stranger asks
    /// the merge tree about one voice instead of three, and the part of the
    /// pipeline that gets meetings wrong — deciding how many people there were
    /// from the shape of a tree — is asked a much easier question.
    ///
    /// A person whose pre-assigned fingerprints add up to less than
    /// [`people::MIN_GUIDED_MS`] is released back to the clustering: one strong
    /// fingerprint of a cough is not somebody being in the meeting.
    pub fn guide(&self, enrolled: &[people::Enrolled]) -> Guided {
        let prints: Vec<&[f32]> = self.items.iter().map(|i| i.embedding.as_slice()).collect();
        let mut assignment = people::pre_assign(&prints, enrolled);

        let mut speech: Vec<i64> = vec![0; enrolled.len()];
        for (i, slot) in assignment.iter().enumerate() {
            if let Some(who) = slot {
                speech[*who] += self.items[i].weight_ms.max(0);
            }
        }
        for slot in assignment.iter_mut() {
            if slot.is_some_and(|who| speech[who] < people::MIN_GUIDED_MS) {
                *slot = None;
            }
        }

        // Group order is the order these voices are first heard, which is the
        // same rule `build_tracks` applies to everybody else.
        let mut groups: Vec<usize> = Vec::new();
        for who in assignment.iter().flatten() {
            if !groups.contains(who) {
                groups.push(*who);
            }
        }
        Guided {
            assignment,
            groups,
            people: enrolled.to_vec(),
        }
    }

    /// Cut at one distance, which is what the calibration tools sweep.
    ///
    /// The app does not take this path any more: the count it would produce is
    /// the one that failed in the field, because a single distance always has
    /// that failure available to it (see [`cluster::CountChoice`]). What the
    /// sweep is still for is checking that the tree this criterion reads is the
    /// same tree the old measurements were made against. `Some(k)` cuts at a
    /// count and ignores the distance entirely.
    pub fn cut(&self, threshold: f32, target: Option<usize>) -> ScanCut {
        let nobody = Guided::default();
        match target {
            Some(k) => {
                let (clustering, forced) = cluster::cluster_fixed(&self.items, k);
                self.tracks_of(clustering, None, Some(forced), &nobody)
            }
            None => self.tracks_of(
                cluster::cluster(&self.items, threshold),
                None,
                None,
                &nobody,
            ),
        }
    }

    /// What the app runs. `Some(k)` is the count the person gave us and wins
    /// outright; `None` is the automatic count, read out of the shape of the
    /// merge tree ([`cluster::cluster_auto`]).
    pub fn cut_for(&self, target: Option<usize>) -> ScanCut {
        self.cut_guided(&Guided::default(), target)
    }

    /// [`Self::cut_for`] with the voices Echo already knows taken out first.
    ///
    /// Every fingerprint [`Self::guide`] recognised is held aside as a
    /// ready-made group, and **only the remainder is counted and clustered**.
    /// `target` is still the whole channel's worth of voices — the count the
    /// person gave us, meaning the same thing it always meant — and the known
    /// people come off it here, because that subtraction is only knowable once
    /// the guiding has happened.
    ///
    /// The floor of one stranger when there are leftover fingerprints is the
    /// same judgement [`remote_target`] makes and for the same reason: leaving
    /// real speech unattributed to make an arithmetic come out right is worse
    /// than one voice too many.
    pub fn cut_guided(&self, guided: &Guided, target: Option<usize>) -> ScanCut {
        self.cut_guided_until(guided, target, &DiarizeControl::new())
            .expect("a cut nothing can stop always finishes")
    }

    /// [`Self::cut_guided`], which a recording can interrupt.
    ///
    /// The clustering is the one stretch of the pass that reads nothing off
    /// disk, so without a checkpoint inside it a park would be honoured only
    /// once the whole merge tree had been built — minutes, on a long meeting,
    /// with a recording already running. Stopping gives back
    /// [`DiarizeError::Yielded`] or [`DiarizeError::Cancelled`] and no cut at
    /// all: half a merge tree is not a smaller answer, it is a wrong one.
    pub fn cut_guided_until(
        &self,
        guided: &Guided,
        target: Option<usize>,
        control: &DiarizeControl,
    ) -> Result<ScanCut, DiarizeError> {
        let known = guided.groups.len();
        let leftovers: Vec<usize> = (0..self.items.len())
            .filter(|i| guided.assignment.get(*i).copied().flatten().is_none())
            .collect();
        let items: Vec<ClusterItem> = leftovers.iter().map(|&i| self.items[i].clone()).collect();

        let strangers = target.map(|t| {
            let left = t.saturating_sub(known);
            if left == 0 && !items.is_empty() {
                1
            } else {
                left
            }
        });

        let stop = || control.checkpoint().is_err();
        let (clustering, choice, forced) = match (items.is_empty(), strangers) {
            (true, _) => (cluster::Clustering::default(), None, None),
            (false, Some(0)) => (cluster::Clustering::default(), None, None),
            (false, Some(k)) => {
                let (clustering, forced) = cluster::cluster_fixed_until(&items, k, &stop)
                    .ok_or_else(|| stopped(control))?;
                (clustering, None, Some(forced))
            }
            (false, None) => {
                let (clustering, choice) =
                    cluster::cluster_auto_until(&items, &stop).ok_or_else(|| stopped(control))?;
                (clustering, Some(choice), None)
            }
        };

        // One label space over every fingerprint: the known people first, in the
        // order they are first heard, then whatever the clustering made of the
        // strangers.
        let mut labels = vec![usize::MAX; self.items.len()];
        for (i, slot) in guided.assignment.iter().enumerate() {
            if let Some(who) = slot {
                if let Some(group) = guided.groups.iter().position(|g| g == who) {
                    labels[i] = group;
                }
            }
        }
        // A leftover cluster that is unmistakably more of a voice already
        // recognised goes back to that voice instead of becoming a speaker of
        // its own. Without this, enrolling somebody *adds* a person to the
        // meeting they were enrolled from: their strong fingerprints are the
        // guided group and their weaker ones cluster into a second, very
        // convincing stranger. See `people::absorb_leftovers`.
        let leftover_centroids: Vec<Vec<f32>> = (0..clustering.cluster_count)
            .map(|label| {
                people::centroid_of(
                    clustering
                        .labels
                        .iter()
                        .enumerate()
                        .filter(|(_, l)| **l == label)
                        .map(|(slot, _)| items[slot].embedding.as_slice()),
                )
            })
            .collect();
        let absorbed =
            people::absorb_leftovers(&leftover_centroids, &guided.people, &guided.groups);
        for (slot, &i) in leftovers.iter().enumerate() {
            if let Some(label) = clustering.labels.get(slot) {
                // The vacated label simply ends up with nothing in it, and
                // `build_tracks` drops empty labels — so the count comes out
                // right without any renumbering.
                labels[i] = match absorbed.get(*label).copied().flatten() {
                    Some(group) => group,
                    None => known + label,
                };
            }
        }
        let combined = cluster::Clustering {
            labels,
            cluster_count: known + clustering.cluster_count,
            threshold: clustering.threshold,
        };
        Ok(self.tracks_of(combined, choice, forced, guided))
    }

    /// Turn one clustering of the fingerprints into per-person timelines.
    fn tracks_of(
        &self,
        clustering: cluster::Clustering,
        choice: Option<cluster::CountChoice>,
        forced: Option<cluster::ForcedOutcome>,
        guided: &Guided,
    ) -> ScanCut {
        let labelled = label_windows(&self.windows, &clustering.labels);
        let (mut tracks, mut confidence, order) =
            build_tracks(&self.windows, &labelled, clustering.cluster_count);

        // Per surviving person, the mean of the fingerprints the clustering put
        // in their group. Computed here because this is the only place that holds
        // both the fingerprints and the order the tracks came out in.
        let mut centroids: Vec<Vec<f32>> = order
            .iter()
            .map(|&label| {
                people::centroid_of(
                    clustering
                        .labels
                        .iter()
                        .enumerate()
                        .filter(|(_, l)| **l == label)
                        .map(|(i, _)| self.items[i].embedding.as_slice()),
                )
            })
            .collect();
        // Voices recognised before any of this happened keep their name.
        let mut matches: Vec<Option<people::Match>> = order
            .iter()
            .enumerate()
            .map(|(track, &label)| {
                let who = *guided.groups.get(label)?;
                let person = guided.people.get(who)?;
                Some(people::Match::Linked {
                    person_id: person.person_id.clone(),
                    name: person.name.clone(),
                    score: people::similarity(&centroids[track], &person.centroid),
                })
            })
            .collect();

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
                centroids = vec![Vec::new()];
                matches = vec![None];
            }
        }

        ScanCut {
            tracks,
            confidence,
            threshold: clustering.threshold,
            choice,
            forced,
            centroids,
            matches,
        }
    }
}

/// The voices Echo recognised before it started counting.
///
/// [`Scan::guide`] builds it; [`Scan::cut_guided`] turns it into ready-made
/// groups. Empty — the [`Default`] — is a meeting with nobody enrolled in it,
/// which is every meeting until somebody says "remember this voice", and it makes
/// the guided path exactly the old path.
#[derive(Debug, Clone, Default)]
pub struct Guided {
    /// Per fingerprint, which enrolled person it unmistakably is.
    assignment: Vec<Option<usize>>,
    /// Which enrolled people got a group, in the order they are first heard.
    groups: Vec<usize>,
    /// The enrolment this was decided against.
    people: Vec<people::Enrolled>,
}

impl Guided {
    /// How many voices this meeting no longer has to work out for itself.
    pub fn recognised(&self) -> usize {
        self.groups.len()
    }

    /// How many fingerprints were handed to a known person.
    pub fn fingerprints(&self) -> usize {
        self.assignment.iter().flatten().count()
    }
}

/// The audio the pass will read, and what its labels will therefore mean.
///
/// A system channel means the far end recorded itself and the microphone is one
/// known person; no system channel means the whole conversation is in the
/// microphone recording and every voice in it has to be separated out (module
/// docs). `None` means there is no committed audio on either channel, and
/// therefore nothing to load a model for (mantra 1).
async fn voice_channel(
    db: &Db,
    meeting_id: &str,
) -> Result<Option<(Source, ChunkPcm)>, DiarizeError> {
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

/// What the models do to one window.
///
/// A trait with exactly one real implementation, so the resumable walk below can
/// be exercised without two ONNX sessions — which is the only way a test can
/// state the thing that matters here: after a park and a resume, no window is
/// analysed twice.
trait WindowModels: Send {
    fn analyse(
        &mut self,
        samples: &[f32],
        start_ms: i64,
        window_ms: i64,
        previous: Option<&WindowResult>,
        items: &mut Vec<ClusterItem>,
    ) -> Result<WindowResult, DiarizeError>;
}

/// The real one: the segmentation network and the fingerprint network.
struct OnnxModels {
    segmenter: Segmenter,
    embedder: Embedder,
}

impl WindowModels for OnnxModels {
    fn analyse(
        &mut self,
        samples: &[f32],
        start_ms: i64,
        window_ms: i64,
        previous: Option<&WindowResult>,
        items: &mut Vec<ClusterItem>,
    ) -> Result<WindowResult, DiarizeError> {
        analyse_window(
            &mut self.segmenter,
            &mut self.embedder,
            samples,
            start_ms,
            window_ms,
            previous,
            items,
        )
    }
}

/// How much work may be lost to something that is not a park — a crash, a
/// force-quit, the power going.
///
/// A park writes the scan down on its way out, so this interval is only about
/// the ways of stopping that get no warning. A minute of ONNX arithmetic is a
/// tolerable loss; writing after every window would not be, because the file
/// holds every fingerprint found so far and grows all pass.
const SAVE_EVERY: std::time::Duration = std::time::Duration::from_secs(60);

/// Where the window walk got to, and everything it found on the way.
#[derive(Debug)]
struct Walk {
    items: Vec<ClusterItem>,
    windows: Vec<WindowResult>,
    next_start_ms: i64,
}

impl Walk {
    fn fresh() -> Self {
        Self {
            items: Vec::new(),
            windows: Vec::new(),
            next_start_ms: 0,
        }
    }
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
    scan_resumable(db, meeting_id, segmenter_path, embedder_path, None, control).await
}

/// [`scan`], with somewhere to keep the work.
///
/// `audio_dir` is the meeting's own directory. Given one, the walk picks up
/// where a parked run stopped and writes down where this one gets to; given
/// `None` — the calibration tools — it behaves exactly as it always did.
/// [`super::scan_cache`] carries the argument for why resuming is safe, and what
/// makes a kept scan get thrown away instead.
pub(crate) async fn scan_resumable(
    db: &Db,
    meeting_id: &str,
    segmenter_path: &std::path::Path,
    embedder_path: &std::path::Path,
    audio_dir: Option<&std::path::Path>,
    control: &DiarizeControl,
) -> Result<Option<Scan>, DiarizeError> {
    control.checkpoint()?;

    let Some((source, mut pcm)) = voice_channel(db, meeting_id).await? else {
        return Ok(None);
    };

    // The key first: it decides whether there is anything to resume, and a
    // model file that cannot even be stat'ed switches the whole cache off
    // rather than being guessed at.
    let key = match audio_dir {
        Some(_) => scan_key(db, &pcm, source, segmenter_path, embedder_path).await,
        None => None,
    };
    let slot = audio_dir.zip(key);

    let covered = pcm.covered();
    let total_ms = pcm.total_ms();
    let step = segmentation::STEP_MS;
    let window_ms = segmentation::WINDOW_MS;
    let planned = ((total_ms + step - 1) / step).max(1);

    let mut walk = Walk::fresh();
    if let Some((dir, key)) = &slot {
        if let Some(kept) = scan_cache::load(dir, key).await {
            tracing::info!(
                meeting = %meeting_id,
                from_ms = kept.next_start_ms,
                fingerprints = kept.items.len(),
                "picking this meeting's separation up where it stopped"
            );
            walk = Walk {
                items: kept.items,
                windows: kept.windows,
                next_start_ms: kept.next_start_ms,
            };
        }
    }

    if walk.next_start_ms < total_ms {
        let threads = inference_threads();
        let mut models = OnnxModels {
            // Loaded only when there are windows left to analyse: a scan that
            // was finished before the park goes straight to clustering without
            // opening an ONNX session at all (mantra 1).
            segmenter: Segmenter::load(segmenter_path, threads)?,
            embedder: Embedder::load(embedder_path, threads)?,
        };
        walk = walk_windows(
            &mut pcm,
            &mut models,
            walk,
            &covered,
            total_ms,
            window_ms,
            step,
            planned,
            slot.as_ref().map(|(dir, key)| (*dir, key)),
            control,
        )
        .await?;
    }

    // Both models are done. Drop the decoded audio before clustering, so the
    // heaviest part of this pass is not also the part holding two ONNX sessions
    // open (mantra 1).
    pcm.release();
    control.checkpoint()?;
    control.report(0.85);

    Ok(Some(Scan {
        items: walk.items,
        windows: walk.windows,
        covered,
    }))
}

/// The window walk itself: forward, one step at a time, from wherever it starts.
///
/// Parked at a checkpoint like everything else in this pass — but on the way out
/// it writes down what it has, which is the difference between a meeting that
/// eventually gets its speakers and one that restarts from zero every time
/// somebody takes a call.
#[allow(clippy::too_many_arguments)]
async fn walk_windows(
    pcm: &mut ChunkPcm,
    models: &mut (dyn WindowModels + Send),
    mut walk: Walk,
    covered: &[Span],
    total_ms: i64,
    window_ms: i64,
    step: i64,
    planned: i64,
    slot: Option<(&std::path::Path, &scan_cache::Key)>,
    control: &DiarizeControl,
) -> Result<Walk, DiarizeError> {
    let mut last_saved = std::time::Instant::now();
    while walk.next_start_ms < total_ms {
        if let Err(stopped) = control.checkpoint() {
            // A park is Echo stepping aside and coming back, so what it worked
            // out is written down. A cancel is not: the commonest reason for
            // one is the person deleting this meeting, whose folder is being
            // removed at that very moment — and a write landing in the middle of
            // that recreates the file the delete just took away, leaving this
            // meeting's voice fingerprints behind after somebody asked Echo to
            // forget the recording.
            keep(slot, &walk, covered, control).await;
            return Err(stopped);
        }

        let start_ms = walk.next_start_ms;
        let window: Span = (start_ms, start_ms + window_ms);
        // Skip stretches of clock with no audio behind them at all.
        let has_audio = covered.iter().any(|&c| timeline::overlap_ms(window, c) > 0);
        if has_audio {
            let samples = pcm.window(start_ms, window_ms).await?;
            if segmentation::has_signal(&samples) {
                let result = models.analyse(
                    &samples,
                    start_ms,
                    window_ms,
                    walk.windows.last(),
                    &mut walk.items,
                )?;
                walk.windows.push(result);
            }
        }

        walk.next_start_ms = start_ms + step;
        // Reserve the last fifth of the bar for clustering and the database.
        let done = (walk.next_start_ms + step - 1) / step;
        control.report(0.8 * (done as f32 / planned as f32));

        if last_saved.elapsed() >= SAVE_EVERY {
            keep(slot, &walk, covered, control).await;
            last_saved = std::time::Instant::now();
        }
    }
    // Finished: the file now says so, and a park during the clustering that
    // follows resumes without touching a model again.
    keep(slot, &walk, covered, control).await;
    Ok(walk)
}

/// Write the walk down, when there is somewhere to write it and the meeting is
/// still there to write it for.
///
/// Cancelled means stop, and the reason is usually that this meeting is being
/// deleted — its folder emptied by `commands::delete_meeting` while this runs.
/// Checked here as well as at the checkpoint above, because a save that came due
/// on a timer has no checkpoint in front of it.
async fn keep(
    slot: Option<(&std::path::Path, &scan_cache::Key)>,
    walk: &Walk,
    covered: &[Span],
    control: &DiarizeControl,
) {
    let Some((dir, key)) = slot else {
        return;
    };
    if control.is_cancelled() {
        return;
    }
    scan_cache::store(
        dir,
        &scan_cache::Cached {
            key: key.clone(),
            next_start_ms: walk.next_start_ms,
            items: walk.items.clone(),
            windows: walk.windows.clone(),
            covered: covered.to_vec(),
        },
    )
    .await;
}

/// Everything that has to be the same for a kept scan to still be true of this
/// meeting, or `None` when one of the model files cannot be identified — in
/// which case nothing is kept and nothing is resumed.
async fn scan_key(
    db: &Db,
    pcm: &ChunkPcm,
    source: Source,
    segmenter_path: &std::path::Path,
    embedder_path: &std::path::Path,
) -> Option<scan_cache::Key> {
    Some(scan_cache::Key {
        format: scan_cache::FORMAT,
        source: source.as_str().to_string(),
        audio: pcm.stamp(),
        total_ms: pcm.total_ms(),
        window_ms: segmentation::WINDOW_MS,
        step_ms: segmentation::STEP_MS,
        segmenter: scan_cache::model_stamp(segmenter_path)?,
        embedder: scan_cache::model_stamp(embedder_path)?,
        embedder_tag: people::embedder_tag(db, embedder_path).await,
    })
}

/// Where a meeting's own directory is, for the things that live beside its
/// audio. `None` when the meeting has gone — the pass then keeps nothing and
/// still works.
async fn audio_dir_of(db: &Db, meeting_id: &str) -> Option<std::path::PathBuf> {
    let recorded = match repo::get_meeting(db, meeting_id).await {
        Ok(Some(meeting)) => std::path::PathBuf::from(meeting.audio_dir),
        Ok(None) => return None,
        Err(error) => {
            tracing::debug!(%error, "could not read where this meeting's audio lives");
            return None;
        }
    };
    // Only a folder that names this meeting. The row is written before the
    // per-meeting folder is recorded on it and the second write is allowed to
    // fail (`session::start`), so a row can name the storage root instead — and
    // deleting the meeting only ever removes the folder named after it
    // (`commands::meeting_audio_dir`, which guards the same way for the same
    // reason). Keeping voice fingerprints anywhere else would leave them behind
    // when somebody asks Echo to forget a recording, and would have every such
    // meeting overwriting the same file. Nothing kept is the correct answer
    // here: the pass costs time and loses nothing.
    let names_this_meeting = recorded
        .file_name()
        .is_some_and(|name| name == std::ffi::OsStr::new(meeting_id));
    if !names_this_meeting {
        tracing::debug!(
            meeting = %meeting_id,
            "this meeting has no folder of its own; its separation will not be kept"
        );
        return None;
    }
    Some(recorded)
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
            threshold: 0.0,
            choice: None,
            speakers,
            people_count,
            people_count_is_override,
            voices_asked: None,
            // The same question as below, asked of the rows channel attribution
            // just pinned: how many voices are audible in what was recorded. A
            // row with nothing on it — "You" on a meeting the person only
            // listened to — is not one of them.
            voices_found: repo::count_audible_people(db, meeting_id)
                .await
                .map_err(db_failed)?,
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

    // Who Echo has been told to remember, before anything is counted (DESIGN §1).
    // A voice that is not being matched right now but could be — its numbers
    // came from an older network, or two names were made one and the centroid
    // was dropped rather than averaged — is re-embedded from its kept clips
    // first, quietly, so neither costs anybody their people.
    let embedder_tag = people::embedder_tag(db, embedder_path).await;
    let mut enrolment = people::enrolled(db, &embedder_tag).await?;
    if enrolment.stale > 0 {
        control.checkpoint()?;
        // Never fatal. A profile that cannot be brought up to date stays skipped
        // and stays reported as needing a refresh; working out who said what does
        // not depend on it.
        match people::refresh_profiles(db, embedder_path).await {
            Ok(0) => {}
            Ok(_) => enrolment = people::enrolled(db, &embedder_tag).await?,
            Err(error) => {
                tracing::debug!(%error, "could not bring the remembered voices up to date")
            }
        }
    }

    // The meeting's own directory is where a parked scan is kept, so twelve
    // minutes of ONNX work survives somebody taking the next call.
    let audio_dir = audio_dir_of(db, meeting_id).await;
    let Some(scanned) = scan_resumable(
        db,
        meeting_id,
        segmenter_path,
        embedder_path,
        audio_dir.as_deref(),
        control,
    )
    .await?
    else {
        return Err(DiarizeError::Failed(
            "the meeting's audio disappeared while the pass was starting".into(),
        ));
    };

    let guided = scanned.guide(&enrolment.people);
    if guided.recognised() > 0 {
        tracing::debug!(
            people = guided.recognised(),
            fingerprints = guided.fingerprints(),
            "recognised {} known voice(s) before counting the rest",
            guided.recognised()
        );
    }
    // What the clustering is about to be asked, and what it has to work with.
    // At info, because the forced path used to log nothing at all and a run that
    // under-delivers is exactly the run somebody reads the log for.
    tracing::info!(
        source = source.as_str(),
        target = ?target,
        recognised = guided.recognised(),
        fingerprints = scanned.fingerprints(),
        dim = ?scanned.fingerprint_dim(),
        speech_ms = scanned.fingerprinted_ms(),
        "separating this meeting's voices"
    );
    // Solid arithmetic with no I/O in it, and minutes of it on a long meeting:
    // off the async worker, and interruptible, or a recording starting would
    // wait for the whole merge tree.
    let mut cut = run_blocking(|| scanned.cut_guided_until(&guided, target, control))?;
    people::match_remaining(&mut cut, &enrolment);
    if let Some(choice) = &cut.choice {
        for candidate in &choice.candidates {
            tracing::info!(
                count = candidate.count,
                silhouette = candidate.silhouette,
                verdict = ?candidate.refusal,
                split_gap = candidate.split_gap,
                cut_clusters = candidate.cut_clusters,
                folded = candidate.folded,
                mass_ms = ?candidate.mass_ms,
                "a possible answer: {} people",
                candidate.count
            );
        }
        tracing::info!(
            count = choice.count,
            cut_at = choice.cut_at,
            silhouette = choice.silhouette,
            runner_up = ?choice.runner_up,
            fragment_bar_ms = choice.fragment_bar_ms,
            "the automatic count read {} people out of the merge tree",
            choice.count
        );
    }
    if let Some(forced) = &cut.forced {
        tracing::info!(
            asked = forced.asked,
            got = forced.got,
            cut_at = forced.cut_at,
            silhouette = forced.silhouette,
            mass_ms = ?forced.mass_ms,
            floor_ms = cluster::FORCED_FLOOR_MS,
            "the count this meeting was given came back with {} of {} voice(s)",
            forced.got,
            forced.asked
        );
    }
    let turns = turns_from_tracks(&cut.tracks, &cut.confidence);

    control.checkpoint()?;
    control.report(0.9);

    let speakers = persist(db, meeting_id, &cut, source).await?;
    let (people_count, people_count_is_override) = people_count(db, meeting_id).await?;
    // Counted from the rows the pass just wrote, and only the ones a line of
    // transcript landed on. `people_count` cannot answer this: with an override
    // it reads back the person's own number, which is the question, not the
    // answer. Nor can a plain row count — "You" exists on a two-channel meeting
    // whether or not the microphone caught a word, and on a meeting somebody
    // only listened to, counting it would claim one more audible voice than the
    // recording holds.
    let voices_found = repo::count_audible_people(db, meeting_id)
        .await
        .map_err(db_failed)?;
    control.report(1.0);

    Ok(DiarizationResult {
        speaker_count: cut.tracks.len() as u32,
        turns,
        threshold: cut.threshold,
        choice: cut.choice,
        speakers,
        people_count,
        people_count_is_override,
        // In people, counting whoever was at this computer — the same units the
        // person typed their correction in, so the two are comparable without
        // anybody redoing [`remote_target`]'s arithmetic backwards.
        voices_asked: people_count_is_override.then_some(people_count),
        voices_found,
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
///
/// The third return value is which cluster label each track came from, which is
/// the only way back from a renumbered track to the fingerprints behind it — what
/// [`Scan::tracks_of`] needs to attach a centroid and a known person to a row.
fn build_tracks(
    windows: &[WindowResult],
    labelled: &[Vec<Option<usize>>],
    cluster_count: usize,
) -> (Vec<Vec<Span>>, Vec<f32>, Vec<usize>) {
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
    (tracks, confidences, order)
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
///
/// **The invariant: no voice row without a line of transcript on it.** Who said
/// what is worked out *first*, in track numbers, and only the tracks that ended
/// up owning at least one final segment get a row. This is a safety net and it
/// is deliberately independent of the clustering: whatever a cut hands over —
/// a count the person gave us, the automatic count, a sweep — a voice that wins
/// no line is not a person anybody can do anything with. On 2026-08-24 a forced
/// count of four produced two rows with 630 and 313 lines and two rows with
/// none, and the person met those two in the enrolment dialog, which could only
/// tell them there was no clear moment of a voice that had never spoken.
///
/// The one row this does not apply to is "You", which is not a cluster: it is
/// the microphone channel itself, and it exists on a two-channel meeting
/// whether or not the person happened to say anything.
///
/// Surviving voices are *labelled* from one in the order they first speak, so a
/// meeting whose first cluster won no line still shows a "Speaker 1" — but they
/// are *keyed* by the voice, not by that position, so a re-run in which one
/// voice falls silent cannot slide the row underneath somebody's rename. See
/// [`cluster_key`].
async fn persist(
    db: &Db,
    meeting_id: &str,
    cut: &ScanCut,
    source: Source,
) -> Result<Vec<Speaker>, DiarizeError> {
    let tracks = cut.tracks.as_slice();
    let revision = repo::transcript_revision(db, meeting_id)
        .await
        .map_err(db_failed)?
        + 1;

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

    // Which channel's lines the clustering is entitled to speak for. The other
    // one is either "You" (a microphone line, when the far end had its own
    // channel) or nothing at all.
    let separated = match source {
        Source::System => Channel::System,
        Source::MicOnly => Channel::Mic,
    };

    // ---- Who said what, in track numbers. Nothing is written yet. ----------
    let mut plan: Vec<(crate::types::Segment, Attribution)> = Vec::new();
    let mut mine: Vec<Id> = Vec::new();
    let mut owned_ms: Vec<i64> = vec![0; tracks.len()];
    let mut owned_lines: Vec<usize> = vec![0; tracks.len()];
    let mut unattributed = 0usize;
    for segment in segments {
        let span: Span = (segment.t_start_ms, segment.t_end_ms);
        if segment.channel == separated {
            // A line that holds two voices is cut in two before anything is
            // attributed. Attributing it whole is what put a whole exchange of
            // fast dialogue on one speaker and silently lost the other one's
            // words; see [`split`].
            let pieces = split::split_line(
                split::LineFacts {
                    span,
                    text_confidence: segment.avg_confidence,
                },
                &segment.text,
                tracks,
                &cut.confidence,
            );
            let attribution = match pieces {
                Some(pieces) => Some(Attribution::Split(pieces)),
                None => timeline::assign_by_overlap(span, tracks)
                    .or_else(|| timeline::nearest_track(span, tracks, NEAREST_TOLERANCE_MS))
                    .map(Attribution::Whole),
            };
            match attribution {
                Some(a) => {
                    for (track, ms) in a.claims(span) {
                        if let (Some(slot), Some(lines)) =
                            (owned_ms.get_mut(track), owned_lines.get_mut(track))
                        {
                            *slot += ms;
                            *lines += 1;
                        }
                    }
                    plan.push((segment, a));
                }
                // Nothing covers it, and guessing is worse than saying nothing:
                // an unattributed line stays honestly unattributed.
                None => unattributed += 1,
            }
        } else if segment.channel == Channel::Mic && source == Source::System {
            mine.push(segment.id);
        }
    }

    // ---- The rows, for the tracks that own speech and no others. -----------
    let me = match source {
        Source::System => Some(
            repo::upsert_speaker(db, meeting_id, SELF_CLUSTER_KEY, SELF_DISPLAY_NAME, true)
                .await
                .map_err(db_failed)?,
        ),
        Source::MicOnly => None,
    };

    let surviving: Vec<usize> = (0..tracks.len()).filter(|i| owned_lines[*i] > 0).collect();

    let mut remote_ids: Vec<Option<Id>> = vec![None; tracks.len()];
    for (slot, &track) in surviving.iter().enumerate() {
        // Keyed by the voice, named by where it comes in the list. The key is
        // the row's identity across runs and is never renumbered (see below);
        // the label is what the person reads, and reads best counting from one.
        let speaker = repo::upsert_speaker(
            db,
            meeting_id,
            &cluster_key(track),
            &display_name(slot),
            false,
        )
        .await
        .map_err(db_failed)?;

        // A row that was already here keeps whatever it is called, unless what
        // it is called is a label Echo made up — those are renumbered so the
        // list still reads 1, 2, 3 after a voice drops out of it. A name
        // somebody typed is never touched.
        if speaker.display_name != display_name(slot) && is_numbered_name(&speaker.display_name) {
            repo::rename_speaker(db, &speaker.id, &display_name(slot))
                .await
                .map_err(db_failed)?;
        }

        // This meeting's voice print for this voice, whether or not anybody
        // knows whose it is. Stored either way: it is what a person enrolled
        // next month will be matched against (DESIGN §1).
        let centroid = cut.centroids.get(track).filter(|c| !c.is_empty());
        repo::set_speaker_centroid(db, &speaker.id, centroid.map(|c| c.as_slice()))
            .await
            .map_err(db_failed)?;

        match cut.matches.get(track).and_then(|m| m.as_ref()) {
            Some(people::Match::Linked {
                person_id, name, ..
            }) => {
                repo::set_speaker_person(db, &speaker.id, Some(person_id))
                    .await
                    .map_err(db_failed)?;
                // The name is *copied* onto the row, and only over a label Echo
                // made up. From here it is meeting-local: renaming this speaker
                // does not rename the person, and deleting the person does not
                // rewrite this meeting.
                if people::is_default_name(&speaker.display_name) {
                    repo::rename_speaker(db, &speaker.id, name)
                        .await
                        .map_err(db_failed)?;
                }
            }
            Some(people::Match::Suggested { person_id, score }) => {
                // Only ever a question, and only on a row nobody has answered:
                // a link or a name somebody typed outranks anything the pass
                // suspects.
                if speaker.person_id.is_none() {
                    repo::set_speaker_suggestion(db, &speaker.id, Some(person_id), Some(*score))
                        .await
                        .map_err(db_failed)?;
                }
            }
            None => {
                if speaker.person_id.is_none() {
                    repo::set_speaker_suggestion(db, &speaker.id, None, None)
                        .await
                        .map_err(db_failed)?;
                }
            }
        }
        tracing::info!(
            speaker = %cluster_key(slot),
            lines = owned_lines[track],
            speaking_ms = owned_ms[track],
            "this voice ended up with {} line(s) of the transcript",
            owned_lines[track]
        );
        remote_ids[track] = Some(speaker.id);
    }

    // ---- The writes, now that every piece has a row to point at. -----------
    fn push(into: &mut Vec<(Id, Vec<Id>)>, speaker_id: &Id, segment_id: Id) {
        match into.iter_mut().find(|(s, _)| s == speaker_id) {
            Some((_, ids)) => ids.push(segment_id),
            None => into.push((speaker_id.clone(), vec![segment_id])),
        }
    }

    let mut by_speaker: Vec<(Id, Vec<Id>)> = Vec::new();
    if let Some(me) = &me {
        for id in mine {
            push(&mut by_speaker, &me.id, id);
        }
    }
    let mut lines_split = 0usize;
    for (segment, attribution) in plan {
        let span: Span = (segment.t_start_ms, segment.t_end_ms);
        match attribution {
            Attribution::Split(pieces) => {
                // A note about a word Echo put right belongs to the half of the
                // line that still holds that word, and to exactly one half.
                let mut notes = corrections_by_piece(&segment.corrections, &pieces);
                let drafts: Vec<crate::types::SegmentDraft> = pieces
                    .iter()
                    .zip(notes.drain(..))
                    .map(|(piece, corrections)| crate::types::SegmentDraft {
                        meeting_id: segment.meeting_id.clone(),
                        t_start_ms: piece.span.0,
                        t_end_ms: piece.span.1,
                        channel: segment.channel,
                        speaker_id: remote_ids.get(piece.speaker).and_then(|id| id.clone()),
                        text: piece.text.clone(),
                        // The words were not re-transcribed, so their language,
                        // the engine's confidence in them and their provenance
                        // are the line's. Only the cut is Echo's.
                        language: segment.language.clone(),
                        avg_confidence: segment.avg_confidence,
                        revision,
                        is_final: true,
                        model_name: segment.model_name.clone(),
                        model_revision: segment.model_revision.clone(),
                        corrections,
                    })
                    .collect();
                if drafts.iter().all(|d| d.speaker_id.is_some())
                    && !repo::split_segment(db, &segment.id, &drafts, revision)
                        .await
                        .map_err(db_failed)?
                        .is_empty()
                {
                    lines_split += 1;
                    continue;
                }
                // The cut did not happen after all. The line is still one of
                // these voices' — whoever holds most of it — rather than
                // nobody's.
                let hit = timeline::assign_by_overlap(span, tracks)
                    .or_else(|| timeline::nearest_track(span, tracks, NEAREST_TOLERANCE_MS));
                if let Some(speaker_id) = hit
                    .and_then(|i| remote_ids.get(i))
                    .and_then(|id| id.clone())
                {
                    push(&mut by_speaker, &speaker_id, segment.id);
                }
            }
            Attribution::Whole(track) => {
                if let Some(speaker_id) = remote_ids.get(track).and_then(|id| id.clone()) {
                    push(&mut by_speaker, &speaker_id, segment.id);
                }
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
    let mut keep: Vec<String> = Vec::with_capacity(surviving.len() + 1);
    if let Some(me) = &me {
        keep.push(me.cluster_key.clone());
    }
    keep.extend(surviving.iter().map(|&track| cluster_key(track)));
    let pruned = if keep.is_empty() {
        // Nothing on the microphone belongs to any voice this pass found — a
        // meeting with no transcript yet, or one whose lines all sit outside
        // the speech it detected. There is nobody to keep, and saying so is
        // `prune_speakers_except`'s one refusal, so it is said here instead.
        repo::forget_speakers(db, meeting_id)
            .await
            .map_err(db_failed)?
    } else {
        repo::prune_speakers_except(db, meeting_id, &keep)
            .await
            .map_err(db_failed)?
    };

    tracing::info!(
        voices = tracks.len(),
        rows = surviving.len(),
        silent_voices = tracks.len() - surviving.len(),
        rows_pruned = pruned,
        lines_split,
        unattributed,
        "wrote {} voice row(s); {} voice(s) won no line and got none",
        surviving.len(),
        tracks.len() - surviving.len()
    );

    repo::recompute_speaking_time(db, meeting_id)
        .await
        .map_err(db_failed)?;

    repo::list_speakers(db, meeting_id).await.map_err(db_failed)
}

/// Which voice one final segment turned out to belong to, before any row
/// exists to point it at.
///
/// [`persist`] works this out for every line first and creates rows afterwards,
/// so that a voice nobody's words landed on never becomes a speaker row.
enum Attribution {
    /// The whole line is one voice's.
    Whole(usize),
    /// The line holds more than one voice and is cut at the turn boundary
    /// first; each piece names the track it belongs to.
    Split(Vec<split::Piece>),
}

impl Attribution {
    /// Per track, how much of the line it claims. One entry for a whole line,
    /// one per piece for a split one.
    fn claims(&self, line: Span) -> Vec<(usize, i64)> {
        match self {
            Attribution::Whole(track) => vec![(*track, timeline::span_ms(line))],
            Attribution::Split(pieces) => pieces
                .iter()
                .map(|p| (p.speaker, timeline::span_ms(p.span)))
                .collect(),
        }
    }
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
///
/// **Keyed by the voice the cut produced, never by where that voice comes in the
/// list.** [`persist`] shows only the voices that won a line of transcript, so
/// the list has gaps in it — voice 3 of 4 can be the second row on the screen.
/// Numbering the *rows* instead would mean a re-run in which one voice fell
/// silent handed voice 3's fingerprints, and its lines, to the row voice 2 is
/// on: the person's rename, and the person they linked it to, would end up on
/// somebody else's speech, and the last row would be pruned away with a name on
/// it. So the key follows the voice and [`display_name`] does the counting.
pub fn cluster_key(index: usize) -> String {
    format!("speaker-{:02}", index + 1)
}

/// Zero jargon: the person sees "Speaker 1", never a cluster id (mantra 2).
pub fn display_name(index: usize) -> String {
    format!("Speaker {}", index + 1)
}

/// Is this still one of the labels Echo made up, rather than a name somebody
/// typed? Only those are renumbered when the list of voices changes.
///
/// Narrower than [`people::is_default_name`], which also counts "You": a remote
/// row somebody has named "You" is a name they chose, and this must not take it
/// off them.
/// Hand each of a line's corrections to the one piece of the split that still
/// holds the word it wrote.
///
/// Asking each piece whether it *contains* the corrected word is not enough,
/// and the case it gets wrong is not exotic: one ASR line can hold two separate
/// manglings of the same name — the 2026-08-24 meeting spelled Langola six ways
/// and used two of them within a sentence of each other — and both notes then
/// say "Langola", so both pieces claim both of them. The reader is told the
/// half in front of them was repaired twice when it was repaired once.
///
/// The pieces tile the line left to right with no gap and no overlap
/// ([`split::Piece`]), and the corrections were recorded left to right as the
/// line was read, so the two orders agree: this walks them together and gives
/// each note the next occurrence of its word that no earlier note has taken.
/// A note whose word is nowhere in the pieces is dropped rather than guessed at.
fn corrections_by_piece(
    corrections: &[crate::types::Correction],
    pieces: &[split::Piece],
) -> Vec<Vec<crate::types::Correction>> {
    let mut out = vec![Vec::new(); pieces.len()];
    // How far the sweep has read: which piece, and how far into it.
    let (mut piece, mut at) = (0usize, 0usize);
    for correction in corrections {
        if correction.to.is_empty() {
            continue;
        }
        while piece < pieces.len() {
            let text = &pieces[piece].text;
            match text.get(at..).and_then(|rest| rest.find(&correction.to)) {
                Some(hit) => {
                    out[piece].push(correction.clone());
                    at += hit + correction.to.len();
                    break;
                }
                None => {
                    piece += 1;
                    at = 0;
                }
            }
        }
    }
    out
}

fn is_numbered_name(display_name: &str) -> bool {
    (0..cluster::MAX_SPEAKERS).any(|i| display_name == self::display_name(i))
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

    fn piece(text: &str) -> split::Piece {
        split::Piece {
            span: (0, 0),
            text: text.to_string(),
            speaker: 0,
        }
    }

    fn note(from: &str, to: &str) -> crate::types::Correction {
        crate::types::Correction {
            from: from.to_string(),
            to: to.to_string(),
        }
    }

    /// Two manglings of the same name in one line, and a cut between them.
    ///
    /// Both notes read "Langola", so asking each half whether it contains
    /// "Langola" hands both notes to both halves — and the reader is told a
    /// sentence was repaired twice when one word in it was repaired once. Each
    /// note belongs to the occurrence it actually made.
    #[test]
    fn a_repaired_word_follows_the_half_of_the_line_it_is_actually_in() {
        let pieces = [piece("Langola è ok."), piece("Poi Langola è tornato.")];
        let notes = corrections_by_piece(
            &[note("Nongula", "Langola"), note("lana gola", "Langola")],
            &pieces,
        );
        assert_eq!(
            notes,
            vec![
                vec![note("Nongula", "Langola")],
                vec![note("lana gola", "Langola")]
            ]
        );
    }

    /// The ordinary shapes: one note, and a note whose word the cut left in the
    /// other half entirely.
    #[test]
    fn a_repaired_word_is_never_claimed_by_a_half_that_does_not_hold_it() {
        let pieces = [
            piece("Non lo so."),
            piece("Quelli di Langola hanno scritto."),
        ];
        assert_eq!(
            corrections_by_piece(&[note("Nongula", "Langola")], &pieces),
            vec![vec![], vec![note("Nongula", "Langola")]]
        );
        // Two different names, one in each half, in the order they were read.
        let pieces = [piece("Usiamo Langola."), piece("E poi Obsidara.")];
        assert_eq!(
            corrections_by_piece(
                &[note("Nongula", "Langola"), note("obsidera", "Obsidara")],
                &pieces
            ),
            vec![
                vec![note("Nongula", "Langola")],
                vec![note("obsidera", "Obsidara")]
            ]
        );
        // And a note for a word no piece kept is dropped, not guessed at.
        assert_eq!(
            corrections_by_piece(&[note("Nongula", "Langola")], &[piece("Non lo so.")]),
            vec![Vec::new()]
        );
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
        let (tracks, conf, _) = build_tracks(&[w], &labelled, 2);
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
        let (tracks, _, _) = build_tracks(&windows, &labelled, 1);
        assert_eq!(tracks, vec![vec![(1_000, 14_000)]]);
    }

    #[test]
    fn a_label_nothing_was_attributed_to_leaves_no_speaker_behind() {
        let mut w = window(0, vec![vec![(0, 4_000)]]);
        w.fingerprint = vec![Some(0)];
        let labelled = label_windows(std::slice::from_ref(&w), &[1]);
        // Cluster 0 exists in the clustering but nothing carries its label.
        let (tracks, _, _) = build_tracks(&[w], &labelled, 2);
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

    /// A clustering cut of some tracks, every voice heard clearly. What
    /// [`persist`] takes, and the confidence is what decides whether a line
    /// holding two voices may be cut in two.
    fn cut_of(tracks: &[Vec<Span>]) -> ScanCut {
        ScanCut {
            confidence: vec![0.9; tracks.len()],
            centroids: vec![Vec::new(); tracks.len()],
            matches: vec![None; tracks.len()],
            tracks: tracks.to_vec(),
            threshold: 0.5,
            choice: None,
            forced: None,
        }
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

    // -----------------------------------------------------------------------
    // Known people: recognised before anything is counted
    // -----------------------------------------------------------------------

    /// Room for several orthogonal "voices" plus somewhere to put the leftover
    /// length, so a test can state exactly what every similarity is.
    const DIM: usize = 32;

    /// A fingerprint `similarity` alike to voice `v`, and near-nothing to every
    /// other voice. `wobble` moves it a little without moving it towards anybody.
    fn print_of(v: usize, similarity: f32, wobble: usize) -> Vec<f32> {
        let mut e = vec![0.0f32; DIM];
        e[v * 2] = similarity;
        e[16 + wobble % 8] = (1.0 - similarity * similarity).max(0.0).sqrt();
        cluster::l2_normalize(&mut e);
        e
    }

    fn enrolled_as(v: usize, name: &str, person_id: &str) -> people::Enrolled {
        let mut centroid = vec![0.0f32; DIM];
        centroid[v * 2] = 1.0;
        people::Enrolled {
            person_id: person_id.to_string(),
            name: name.to_string(),
            centroid,
        }
    }

    /// A scan of some fingerprints: one window each, one voice per window, each
    /// talking its own stretch of the meeting clock. Everything the clustering and
    /// the guiding need, and nothing that needs a model.
    fn scan_of(prints: &[(Vec<f32>, i64)]) -> Scan {
        let mut items: Vec<ClusterItem> = Vec::new();
        let mut windows: Vec<WindowResult> = Vec::new();
        for (i, (embedding, ms)) in prints.iter().enumerate() {
            let start = i as i64 * 10_000;
            let mut w = window(start, vec![vec![(start, start + ms)]]);
            w.fingerprint = vec![Some(items.len())];
            windows.push(w);
            items.push(ClusterItem {
                embedding: embedding.clone(),
                weight_ms: *ms,
            });
        }
        Scan {
            items,
            windows,
            covered: vec![(0, prints.len() as i64 * 10_000)],
        }
    }

    /// Three of a known voice and two strangers, which is the shape DESIGN §1 is
    /// about: "enrolled voices are matched BEFORE the count question".
    fn one_known_two_strangers() -> Scan {
        scan_of(&[
            (print_of(0, 0.95, 0), 4_000),
            (print_of(0, 0.93, 1), 4_000),
            (print_of(0, 0.96, 2), 4_000),
            (print_of(1, 0.98, 3), 4_000),
            (print_of(1, 0.97, 4), 4_000),
            (print_of(2, 0.98, 5), 4_000),
            (print_of(2, 0.96, 6), 4_000),
        ])
    }

    /// The point of the whole feature: a voice Echo already knows is taken out
    /// before the count is worked out, so the hard question — how many people
    /// were here — is only ever asked about strangers.
    #[test]
    fn a_known_voice_is_taken_out_before_the_counting_starts() {
        let scanned = one_known_two_strangers();
        let known = vec![enrolled_as(0, "Marco", "person-marco")];

        let guided = scanned.guide(&known);
        assert_eq!(guided.recognised(), 1);
        assert_eq!(
            guided.fingerprints(),
            3,
            "all three of Marco's, and nobody else's"
        );

        let cut = scanned.cut_guided(&guided, None);
        // The automatic count only ever saw the strangers. That is the claim, and
        // this is the only place it can be observed.
        let choice = cut.choice.as_ref().expect("the strangers were counted");
        assert_eq!(choice.count, 2, "the count was asked about Marco as well");
        assert_eq!(cut.tracks.len(), 3, "Marco plus the two strangers");

        // Marco is the first voice heard, so he is the first row, and he arrives
        // with a name rather than a number.
        match cut.matches[0].as_ref().expect("Marco was recognised") {
            people::Match::Linked {
                name,
                person_id,
                score,
            } => {
                assert_eq!(name, "Marco");
                assert_eq!(person_id, "person-marco");
                assert!(*score > people::TAU_LINK, "{score}");
            }
            other => panic!("{other:?}"),
        }
        assert!(cut.matches[1].is_none(), "a stranger must stay a stranger");
        assert!(cut.matches[2].is_none());
        // And every voice, known or not, leaves a print behind for next time.
        assert!(cut.centroids.iter().all(|c| c.len() == DIM));
    }

    /// The same scan with nobody enrolled: three voices, all counted, no names.
    /// The guided path has to be exactly the old path when there is nobody to
    /// recognise, because that is every meeting until somebody says "remember
    /// this voice".
    #[test]
    fn with_nobody_enrolled_the_pass_answers_exactly_as_it_used_to() {
        let scanned = one_known_two_strangers();
        let guided = scanned.guide(&[]);
        assert_eq!(guided.recognised(), 0);

        let cut = scanned.cut_guided(&guided, None);
        assert_eq!(cut.choice.as_ref().unwrap().count, 3, "all three counted");
        assert_eq!(cut.tracks.len(), 3);
        assert!(cut.matches.iter().all(|m| m.is_none()));

        // Byte for byte the same as the unguided entry point.
        let plain = scanned.cut_for(None);
        assert_eq!(plain.tracks, cut.tracks);
        assert_eq!(plain.threshold, cut.threshold);
    }

    /// Enrolling somebody must not add a person to the meeting they were
    /// enrolled from.
    ///
    /// The regression `examples/people_probe` caught on a real two-person
    /// meeting: [`people::pre_assign`] takes the fingerprints that clear
    /// `TAU_STRONG` and the rest of that same person's speech falls through to
    /// the clustering, where it is a perfectly coherent unfamiliar voice. The
    /// meeting went from two speakers to three, the enrolled person split across
    /// two rows, both wearing their name. Here Marco has three fingerprints that
    /// clear the strict bar and three that do not — the shape of any real voice,
    /// where the clean stretches are recognisable and the short answers are not.
    #[test]
    fn enrolling_somebody_does_not_add_a_speaker_to_their_own_meeting() {
        // The weak three share a wobble, so they are each other's nearest
        // neighbour and the clustering does what it did in the field: makes them
        // one convincing voice.
        let scanned = scan_of(&[
            (print_of(0, 0.95, 0), 6_000),
            (print_of(0, 0.94, 1), 6_000),
            (print_of(0, 0.96, 2), 6_000),
            (print_of(0, 0.70, 7), 6_000),
            (print_of(0, 0.70, 7), 6_000),
            (print_of(0, 0.70, 7), 6_000),
            (print_of(1, 0.98, 3), 8_000),
            (print_of(1, 0.97, 4), 8_000),
        ]);
        let guided = scanned.guide(&[enrolled_as(0, "Marco", "person-marco")]);
        assert_eq!(guided.recognised(), 1);
        assert_eq!(
            guided.fingerprints(),
            3,
            "only the three that clear the strict bar pre-assign — which is the \
             whole reason the leftovers have to be dealt with"
        );

        let cut = scanned.cut_guided(&guided, None);
        assert_eq!(
            cut.tracks.len(),
            2,
            "Marco and one stranger — not Marco, Marco again, and one stranger"
        );
        assert_eq!(
            cut.matches.iter().flatten().count(),
            1,
            "one person, one voice"
        );
        match cut.matches[0].as_ref().expect("Marco, heard first") {
            people::Match::Linked { name, .. } => assert_eq!(name, "Marco"),
            other => panic!("{other:?}"),
        }
        assert!(cut.matches[1].is_none(), "the stranger stays a stranger");

        // And the speech went back to Marco rather than being dropped: his track
        // holds the weak stretches too.
        let marco = timeline::total_ms(&cut.tracks[0]);
        let stranger = timeline::total_ms(&cut.tracks[1]);
        assert!(
            marco > stranger,
            "Marco said 36 s and the stranger 16 s, got {marco} and {stranger}"
        );
    }

    /// A single strong fingerprint is not somebody being in the meeting. Two
    /// seconds of a voice that sounds like Marco is a cough, a hold tone, or a
    /// word over the top of somebody else.
    #[test]
    fn a_moment_of_a_known_voice_is_not_enough_to_take_a_row() {
        let scanned = scan_of(&[
            (print_of(0, 0.95, 0), 1_200),
            (print_of(1, 0.98, 1), 8_000),
            (print_of(1, 0.97, 2), 8_000),
        ]);
        let guided = scanned.guide(&[enrolled_as(0, "Marco", "person-marco")]);
        assert_eq!(guided.recognised(), 0, "under MIN_GUIDED_MS");
        // And the fingerprint is back in the clustering rather than thrown away.
        let cut = scanned.cut_guided(&guided, None);
        assert!(cut.matches.iter().all(|m| m.is_none()));
        assert!(!cut.tracks.is_empty());
    }

    /// "There were three of us" with one of the three already known: the count
    /// the person gave still means the same thing, and the arithmetic happens
    /// where the known people are known.
    #[test]
    fn a_count_the_person_gave_has_the_known_people_taken_off_it() {
        let scanned = one_known_two_strangers();
        let guided = scanned.guide(&[enrolled_as(0, "Marco", "person-marco")]);

        let three = scanned.cut_guided(&guided, Some(3));
        assert_eq!(three.tracks.len(), 3, "Marco plus two strangers");
        assert!(
            three.choice.is_none(),
            "a count the person gave is not a guess"
        );

        // "There were two of us": Marco plus one stranger.
        let two = scanned.cut_guided(&guided, Some(2));
        assert_eq!(two.tracks.len(), 2);

        // "Just me and Marco" on a recording that plainly has strangers in it.
        // One voice too many beats leaving real speech unattributed, which is the
        // same judgement `remote_target` makes.
        let one = scanned.cut_guided(&guided, Some(1));
        assert_eq!(one.tracks.len(), 2, "{:?}", one.tracks);
    }

    /// Everybody in the meeting is somebody Echo knows: nothing left to cluster
    /// at all, and no count to work out.
    #[test]
    fn a_meeting_of_nothing_but_known_voices_needs_no_clustering() {
        let scanned = scan_of(&[
            (print_of(0, 0.95, 0), 6_000),
            (print_of(0, 0.94, 1), 6_000),
            (print_of(1, 0.96, 2), 6_000),
            (print_of(1, 0.97, 3), 6_000),
        ]);
        let guided = scanned.guide(&[
            enrolled_as(0, "Marco", "person-marco"),
            enrolled_as(1, "Priya", "person-priya"),
        ]);
        assert_eq!(guided.recognised(), 2);

        let cut = scanned.cut_guided(&guided, None);
        assert!(cut.choice.is_none(), "there was nothing left to count");
        assert_eq!(cut.tracks.len(), 2);
        let names: Vec<&str> = cut
            .matches
            .iter()
            .map(|m| match m.as_ref().expect("both recognised") {
                people::Match::Linked { name, .. } => name.as_str(),
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(names, vec!["Marco", "Priya"]);
    }

    /// A voice that is only *probably* somebody is a question on the row, not a
    /// name — and the same person cannot be two of this meeting's voices.
    #[test]
    fn the_second_stage_asks_rather_than_claims_and_never_names_one_person_twice() {
        // Two voices, both somewhat like Marco, neither unmistakable. The better
        // one gets the question; the other gets nothing, because two voices in one
        // room cannot both be Marco.
        let mut cut = ScanCut {
            tracks: vec![vec![(0, 8_000)], vec![(9_000, 17_000)]],
            confidence: vec![0.9, 0.9],
            centroids: vec![print_of(0, 0.58, 0), print_of(0, 0.55, 1)],
            matches: vec![None, None],
            threshold: 0.5,
            choice: None,
            forced: None,
        };
        let enrolment = people::Enrolment {
            people: vec![enrolled_as(0, "Marco", "person-marco")],
            stale: 0,
        };
        people::match_remaining(&mut cut, &enrolment);

        match cut.matches[0]
            .as_ref()
            .expect("the better one is asked about")
        {
            people::Match::Suggested { person_id, .. } => assert_eq!(person_id, "person-marco"),
            other => panic!("{other:?}"),
        }
        assert!(cut.matches[1].is_none(), "one person, one voice");
    }

    #[test]
    fn a_voice_already_recognised_is_not_decided_twice() {
        let mut cut = ScanCut {
            tracks: vec![vec![(0, 8_000)]],
            confidence: vec![0.9],
            centroids: vec![print_of(0, 0.99, 0)],
            matches: vec![Some(people::Match::Linked {
                person_id: "person-marco".into(),
                name: "Marco".into(),
                score: 0.99,
            })],
            threshold: 0.5,
            choice: None,
            forced: None,
        };
        let enrolment = people::Enrolment {
            people: vec![enrolled_as(0, "Renamed since", "person-marco")],
            stale: 0,
        };
        people::match_remaining(&mut cut, &enrolment);
        match cut.matches[0].as_ref().unwrap() {
            people::Match::Linked { name, .. } => assert_eq!(name, "Marco"),
            other => panic!("{other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // What the pass writes down about a known person
    // -----------------------------------------------------------------------

    /// A cut of one voice, with a known person attached and a voice print to
    /// remember it by.
    ///
    /// Each voice gets four seconds of its own, five seconds apart, so a caller
    /// can give every one of them a line — a voice with no line of transcript on
    /// it no longer gets a row at all ([`persist`]).
    fn cut_with(matches: Vec<Option<people::Match>>) -> ScanCut {
        let tracks: Vec<Vec<Span>> = (0..matches.len().max(1))
            .map(|i| vec![(i as i64 * 5_000, i as i64 * 5_000 + 4_000)])
            .collect();
        ScanCut {
            confidence: vec![0.9; tracks.len()],
            centroids: (0..tracks.len()).map(|i| print_of(i, 0.99, i)).collect(),
            matches,
            tracks,
            threshold: 0.5,
            choice: None,
            forced: None,
        }
    }

    #[tokio::test]
    async fn a_linked_voice_gets_the_persons_name_and_keeps_a_typed_one() {
        let db = db().await;
        let meeting = repo::create_meeting(&db, "Team sync", "/tmp/echo-test", None)
            .await
            .unwrap();
        repo::insert_segments(&db, &[remote_line(&meeting.id, 0, 4_000)])
            .await
            .unwrap();
        let marco = repo::create_person(&db, "Marco").await.unwrap();

        let linked = || {
            vec![Some(people::Match::Linked {
                person_id: marco.id.clone(),
                name: "Marco".into(),
                score: 0.9,
            })]
        };
        let speakers = persist(&db, &meeting.id, &cut_with(linked()), Source::System)
            .await
            .unwrap();
        let voice = speakers
            .iter()
            .find(|s| s.cluster_key == cluster_key(0))
            .unwrap();
        assert_eq!(voice.person_id.as_deref(), Some(marco.id.as_str()));
        assert_eq!(
            voice.display_name, "Marco",
            "the label Echo made up is replaced"
        );
        assert!(voice.suggested_person_id.is_none());

        // The person renames the chip in this meeting. That is meeting-local: the
        // link stays, the person keeps their own name, and a re-run does not
        // overwrite what was typed.
        repo::rename_speaker(&db, &voice.id, "Marco B.")
            .await
            .unwrap();
        let again = persist(&db, &meeting.id, &cut_with(linked()), Source::System)
            .await
            .unwrap();
        let voice = again
            .iter()
            .find(|s| s.cluster_key == cluster_key(0))
            .unwrap();
        assert_eq!(voice.display_name, "Marco B.");
        assert_eq!(voice.person_id.as_deref(), Some(marco.id.as_str()));
        assert_eq!(
            repo::get_person(&db, &marco.id)
                .await
                .unwrap()
                .unwrap()
                .name,
            "Marco",
            "renaming a speaker must never rename the person"
        );
    }

    #[tokio::test]
    async fn a_suggestion_is_stored_as_a_question_and_taken_back_when_it_stops_being_true() {
        let db = db().await;
        let meeting = repo::create_meeting(&db, "Client call", "/tmp/echo-test", None)
            .await
            .unwrap();
        repo::insert_segments(&db, &[remote_line(&meeting.id, 0, 4_000)])
            .await
            .unwrap();
        let marco = repo::create_person(&db, "Marco").await.unwrap();

        let speakers = persist(
            &db,
            &meeting.id,
            &cut_with(vec![Some(people::Match::Suggested {
                person_id: marco.id.clone(),
                score: 0.58,
            })]),
            Source::System,
        )
        .await
        .unwrap();
        let voice = speakers
            .iter()
            .find(|s| s.cluster_key == cluster_key(0))
            .unwrap();
        assert_eq!(
            voice.suggested_person_id.as_deref(),
            Some(marco.id.as_str())
        );
        assert!((voice.suggestion_score.unwrap() - 0.58).abs() < 1e-6);
        assert!(
            voice.person_id.is_none(),
            "a suggestion is never an assignment"
        );
        assert_eq!(voice.display_name, "Speaker 1", "and never a name");

        // A later run that no longer believes it takes the question away rather
        // than leaving it on the screen.
        let speakers = persist(&db, &meeting.id, &cut_with(vec![None]), Source::System)
            .await
            .unwrap();
        let voice = speakers
            .iter()
            .find(|s| s.cluster_key == cluster_key(0))
            .unwrap();
        assert!(voice.suggested_person_id.is_none());
        assert!(voice.suggestion_score.is_none());
    }

    /// Every voice the pass separates leaves its print on the row — which is what
    /// makes a recurring stranger findable later without re-reading any audio
    /// (DESIGN §1).
    #[tokio::test]
    async fn the_pass_leaves_a_voice_print_on_every_row_it_makes() {
        let db = db().await;
        let meeting = repo::create_meeting(&db, "Workshop", "/tmp/echo-test", None)
            .await
            .unwrap();
        repo::insert_segments(
            &db,
            &[
                remote_line(&meeting.id, 0, 4_000),
                remote_line(&meeting.id, 5_000, 9_000),
            ],
        )
        .await
        .unwrap();

        persist(
            &db,
            &meeting.id,
            &cut_with(vec![None, None]),
            Source::System,
        )
        .await
        .unwrap();

        let prints = repo::list_unnamed_voice_prints(&db).await.unwrap();
        assert_eq!(prints.len(), 2, "{prints:?}");
        assert!(prints.iter().all(|p| p.centroid.len() == DIM));
        assert!(prints.iter().all(|p| p.meeting_id == meeting.id));
        // "You" is not one of them: the microphone was never clustered.
        assert!(prints.iter().all(|p| p.display_name.starts_with("Speaker")));
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
        let after_first = persist(&db, &meeting.id, &cut_of(&three), Source::System)
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
        let after_second = persist(&db, &meeting.id, &cut_of(&one), Source::System)
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

        let first = persist(
            &db,
            &meeting.id,
            &cut_of(&[vec![(0i64, 4_000i64)]]),
            Source::System,
        )
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

        let second = persist(
            &db,
            &meeting.id,
            &cut_of(&[vec![(0i64, 4_000i64)]]),
            Source::System,
        )
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

        let speakers = persist(&db, &meeting.id, &cut_of(&[]), Source::System)
            .await
            .unwrap();
        assert_eq!(speakers.len(), 1);
        assert_eq!(speakers[0].cluster_key, SELF_CLUSTER_KEY);
        assert!(speakers[0].is_self);
    }

    /// The safety net, and the shape of the meeting of 2026-08-24: a cut that
    /// hands over a voice which then wins no line of the transcript. Whatever
    /// decided that cut — a count the person gave us, the automatic count — a
    /// voice with nothing behind it must not become a row. Two of them did on
    /// that meeting, and the person met them in the naming dialog.
    #[tokio::test]
    async fn a_voice_that_wins_no_line_gets_no_row() {
        let db = db().await;
        let meeting = repo::create_meeting(&db, "Standup", "/tmp/echo-test", None)
            .await
            .unwrap();
        repo::insert_segments(
            &db,
            &[
                remote_line(&meeting.id, 0, 4_000),
                remote_line(&meeting.id, 5_000, 9_000),
            ],
        )
        .await
        .unwrap();

        // Two voices with the conversation between them, and two slivers off in
        // the silence where nobody said anything.
        let cut = cut_of(&[
            vec![(0i64, 4_000i64)],
            vec![(5_000, 9_000)],
            vec![(30_000, 30_300)],
            vec![(31_000, 31_300)],
        ]);
        let speakers = persist(&db, &meeting.id, &cut, Source::System)
            .await
            .unwrap();

        assert_eq!(speakers.len(), 3, "You and the two voices that spoke");
        let keys: Vec<&str> = speakers.iter().map(|s| s.cluster_key.as_str()).collect();
        assert_eq!(
            keys,
            vec![
                SELF_CLUSTER_KEY,
                cluster_key(0).as_str(),
                cluster_key(1).as_str()
            ],
            "the two silent voices left rows behind"
        );
        // The invariant, stated as the person would meet it: every row has at
        // least one line of the transcript on it.
        let segments = repo::get_segments(
            &db,
            &TranscriptQuery {
                meeting_id: meeting.id.clone(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        for speaker in speakers.iter().filter(|s| !s.is_self) {
            assert!(
                segments
                    .iter()
                    .any(|s| s.speaker_id.as_deref() == Some(speaker.id.as_str())),
                "{} has no lines and should not exist",
                speaker.display_name
            );
        }
        assert_eq!(repo::count_people(&db, &meeting.id).await.unwrap(), 3);
    }

    /// The rows are *labelled* by the voices that survive, so a meeting whose
    /// first cluster is a stray still has a "Speaker 1". Without this the person
    /// is shown "Speaker 2" and "Speaker 3" and left wondering where the first
    /// one went. The key underneath still names the voice, not the position —
    /// [`a_rename_stays_on_the_voice_it_was_given_to`] is what that is for.
    #[tokio::test]
    async fn the_voices_that_survive_are_numbered_from_one() {
        let db = db().await;
        let meeting = repo::create_meeting(&db, "Retro", "/tmp/echo-test", None)
            .await
            .unwrap();
        repo::insert_segments(&db, &[remote_line(&meeting.id, 10_000, 14_000)])
            .await
            .unwrap();

        // The sliver speaks first on the clock, so it is track 0.
        let cut = cut_of(&[vec![(0i64, 300i64)], vec![(10_000, 14_000)]]);
        let speakers = persist(&db, &meeting.id, &cut, Source::System)
            .await
            .unwrap();

        let voices: Vec<&Speaker> = speakers.iter().filter(|s| !s.is_self).collect();
        assert_eq!(voices.len(), 1);
        assert_eq!(voices[0].display_name, display_name(0), "counted from one");
        assert_eq!(
            voices[0].cluster_key,
            cluster_key(1),
            "the key names the voice it came from, not where it sits in the list"
        );
    }

    /// **The thing the key exists for.** A person renames a voice and links it
    /// to somebody they know; a re-run then finds one of the *earlier* voices
    /// silent. The row must still be the row for that voice — same name, same
    /// person, same lines — however the list around it shifts. Numbering the
    /// rows instead of the voices slides the name onto somebody else's speech
    /// and deletes the last row with a name still on it.
    #[tokio::test]
    async fn a_rename_stays_on_the_voice_it_was_given_to() {
        let db = db().await;
        let meeting = repo::create_meeting(&db, "Weekly", "/tmp/echo-test", None)
            .await
            .unwrap();
        repo::insert_segments(
            &db,
            &[
                remote_line(&meeting.id, 0, 4_000),
                remote_line(&meeting.id, 5_000, 9_000),
            ],
        )
        .await
        .unwrap();

        // Two voices, both talking: two rows.
        let both = cut_of(&[vec![(0i64, 4_000i64)], vec![(5_000, 9_000)]]);
        persist(&db, &meeting.id, &both, Source::System)
            .await
            .unwrap();
        let second = repo::list_speakers(&db, &meeting.id)
            .await
            .unwrap()
            .into_iter()
            .find(|s| s.cluster_key == cluster_key(1))
            .expect("the second voice has a row");
        let person = repo::create_person(&db, "Ada").await.unwrap();
        repo::rename_speaker(&db, &second.id, "Ada").await.unwrap();
        repo::set_speaker_person(&db, &second.id, Some(&person.id))
            .await
            .unwrap();

        // A re-run in which the first voice wins nothing at all.
        let only_second = cut_of(&[vec![(30_000i64, 30_300i64)], vec![(5_000, 9_000)]]);
        let speakers = persist(&db, &meeting.id, &only_second, Source::System)
            .await
            .unwrap();

        let voices: Vec<&Speaker> = speakers.iter().filter(|s| !s.is_self).collect();
        assert_eq!(voices.len(), 1, "one voice spoke, so one row");
        assert_eq!(voices[0].cluster_key, cluster_key(1));
        assert_eq!(voices[0].display_name, "Ada", "the name moved or was lost");
        assert_eq!(voices[0].person_id.as_deref(), Some(person.id.as_str()));
        // And Ada still has the lines she said, not the silent voice's.
        let segments = repo::get_segments(
            &db,
            &TranscriptQuery {
                meeting_id: meeting.id.clone(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let hers: Vec<i64> = segments
            .iter()
            .filter(|s| s.speaker_id.as_deref() == Some(voices[0].id.as_str()))
            .map(|s| s.t_start_ms)
            .collect();
        assert_eq!(hers, vec![5_000]);
    }

    /// With nothing on disk to separate there is no pass to report on: the
    /// result carries the rows that exist and no claim about how many voices
    /// were heard.
    #[tokio::test]
    async fn a_meeting_with_no_audio_reports_the_rows_it_has() {
        let db = db().await;
        let meeting = repo::create_meeting(&db, "Notes only", "/tmp/echo-test", None)
            .await
            .unwrap();
        repo::insert_segments(&db, &[mic_line(&meeting.id, 0, 4_000)])
            .await
            .unwrap();

        let missing = std::path::Path::new("/nowhere/at/all");
        let result = refine(&db, &meeting.id, missing, missing, &DiarizeControl::new())
            .await
            .expect("no audio is an answer, not a failure");
        assert_eq!(result.voices_asked, None);
        assert_eq!(result.voices_found, 1, "the microphone row");
        assert_eq!(result.speaker_count, 0);
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

    /// Fast dialogue the engine wrote as one row comes out as one row per voice,
    /// and both halves keep the provenance of the words nobody re-transcribed.
    ///
    /// The other half of the field failure: before this, a line like the one
    /// below went whole to whoever held more of it, and one of the two people in
    /// the exchange silently lost their words.
    #[tokio::test]
    async fn a_line_of_fast_dialogue_is_cut_so_both_voices_keep_their_words() {
        let db = db().await;
        let meeting = repo::create_meeting(&db, "Coffee", "/tmp/echo-test", None)
            .await
            .unwrap();
        repo::insert_segments(
            &db,
            &[crate::types::SegmentDraft {
                meeting_id: meeting.id.clone(),
                t_start_ms: 0,
                t_end_ms: 9_000,
                channel: Channel::Mic,
                text: "Domani sera prendo l'aereo e vengo in Italia. \
                       Ah ok ho capito bene allora ci vediamo."
                    .into(),
                language: Some("it".into()),
                avg_confidence: Some(0.8),
                revision: 3,
                is_final: true,
                model_name: Some("whisper large-v3 (ggml)".into()),
                model_revision: Some("abc123".into()),
                ..Default::default()
            }],
        )
        .await
        .unwrap();

        let two = vec![vec![(0i64, 4_500i64)], vec![(4_500, 9_000)]];
        let speakers = persist(&db, &meeting.id, &cut_of(&two), Source::MicOnly)
            .await
            .unwrap();
        assert_eq!(speakers.len(), 2);

        let segments = repo::get_segments(
            &db,
            &TranscriptQuery {
                meeting_id: meeting.id.clone(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(segments.len(), 2, "the line was not cut: {segments:?}");
        assert_eq!(
            segments[0].speaker_id.as_deref(),
            Some(speakers[0].id.as_str())
        );
        assert_eq!(
            segments[1].speaker_id.as_deref(),
            Some(speakers[1].id.as_str())
        );
        assert!(
            segments[0].text.ends_with("Italia."),
            "{:?}",
            segments[0].text
        );
        assert!(
            segments[1].text.starts_with("Ah ok"),
            "{:?}",
            segments[1].text
        );
        // The pieces tile the line, carry the new revision, and keep the
        // provenance of words that were never re-transcribed.
        assert_eq!((segments[0].t_start_ms, segments[1].t_end_ms), (0, 9_000));
        assert_eq!(segments[0].t_end_ms, segments[1].t_start_ms);
        assert!(segments.iter().all(|s| s.revision == 4
            && s.language.as_deref() == Some("it")
            && s.model_revision.as_deref() == Some("abc123")));

        // Running the pass again does not cut the halves in half: each of them
        // now holds one voice.
        let again = persist(&db, &meeting.id, &cut_of(&two), Source::MicOnly)
            .await
            .unwrap();
        assert_eq!(again.len(), 2);
        assert_eq!(repo::count_segments(&db, &meeting.id).await.unwrap(), 2);
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
        let speakers = persist(&db, &meeting.id, &cut_of(&two), Source::MicOnly)
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

        persist(
            &db,
            &meeting.id,
            &cut_of(&[vec![(0i64, 4_000i64)]]),
            Source::MicOnly,
        )
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
        let first = persist(&db, &meeting.id, &cut_of(&two), Source::MicOnly)
            .await
            .unwrap();
        repo::rename_speaker(&db, &first[0].id, "Stefano")
            .await
            .unwrap();
        repo::rename_speaker(&db, &first[1].id, "Giulia")
            .await
            .unwrap();

        let second = persist(&db, &meeting.id, &cut_of(&two), Source::MicOnly)
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

        let speakers = persist(
            &db,
            &meeting.id,
            &cut_of(&[vec![(5_000i64, 9_000i64)]]),
            Source::System,
        )
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

    /// A meeting the person only listened to: "You" gets a row because they were
    /// there, and it has nothing on it because they said nothing. The count Echo
    /// reports back — the one behind "Echo can only hear N distinct voices" —
    /// must not include it, or a two-voice recording claims three.
    #[tokio::test]
    async fn the_silent_you_row_is_not_one_of_the_voices_echo_can_hear() {
        let db = db().await;
        let meeting = repo::create_meeting(&db, "Webinar", "/tmp/echo-test", None)
            .await
            .unwrap();
        repo::insert_segments(
            &db,
            &[
                remote_line(&meeting.id, 0, 4_000),
                remote_line(&meeting.id, 5_000, 9_000),
            ],
        )
        .await
        .unwrap();

        let cut = cut_of(&[vec![(0i64, 4_000i64)], vec![(5_000, 9_000)]]);
        let speakers = persist(&db, &meeting.id, &cut, Source::System)
            .await
            .unwrap();
        assert_eq!(speakers.len(), 3, "You, silent, plus the two who spoke");
        assert_eq!(
            repo::count_people(&db, &meeting.id).await.unwrap(),
            3,
            "there were three people in the meeting"
        );
        assert_eq!(
            repo::count_audible_people(&db, &meeting.id).await.unwrap(),
            2,
            "and two of them can be heard"
        );
    }

    /// The other end of the same rule: nothing on the microphone belongs to any
    /// voice the pass found, so there is nobody to keep and every row goes.
    /// Microphone-only, so there is no "You" row holding the list open — the
    /// branch that deletes them all outright.
    #[tokio::test]
    async fn a_pass_that_finds_nobody_leaves_nobody_behind() {
        let db = db().await;
        let meeting = repo::create_meeting(&db, "Voice memo", "/tmp/echo-test", None)
            .await
            .unwrap();
        repo::insert_segments(&db, &[mic_line(&meeting.id, 0, 4_000)])
            .await
            .unwrap();
        // The rows a previous run left behind, one of them with a name on it.
        let stale = repo::upsert_speaker(&db, &meeting.id, &cluster_key(0), "Speaker 1", false)
            .await
            .unwrap();
        repo::rename_speaker(&db, &stale.id, "Giulia")
            .await
            .unwrap();
        let existing = repo::get_segments(
            &db,
            &TranscriptQuery {
                meeting_id: meeting.id.clone(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let ids: Vec<Id> = existing.iter().map(|s| s.id.clone()).collect();
        repo::assign_speaker(&db, &ids, &stale.id, 1).await.unwrap();

        // Another meeting's rows, which this must not touch.
        let other = repo::create_meeting(&db, "Standup", "/tmp/echo-test", None)
            .await
            .unwrap();
        repo::upsert_speaker(&db, &other.id, &cluster_key(0), "Speaker 1", false)
            .await
            .unwrap();

        let speakers = persist(&db, &meeting.id, &cut_of(&[]), Source::MicOnly)
            .await
            .unwrap();
        assert!(speakers.is_empty(), "{speakers:?}");
        assert_eq!(repo::count_people(&db, &meeting.id).await.unwrap(), 0);
        assert_eq!(
            repo::count_people(&db, &other.id).await.unwrap(),
            1,
            "the other meeting's speakers were deleted too"
        );
        // The line that pointed at a deleted row reads as unattributed rather
        // than as a voice nobody can look up.
        let segments = repo::get_segments(
            &db,
            &TranscriptQuery {
                meeting_id: meeting.id.clone(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert!(segments.iter().all(|s| s.speaker_id.is_none()));
    }
}

// ===========================================================================
// Keeping the expensive half across a park
// ===========================================================================

#[cfg(test)]
mod resume_tests {
    use super::*;
    use crate::types::Channel;

    async fn db() -> Db {
        let db = crate::db::connect_in_memory().await.expect("in-memory db");
        crate::db::migrate(&db).await.expect("migrations");
        db
    }

    /// Stands in for the two ONNX networks and counts what it was asked to do.
    ///
    /// Every window it sees costs real seconds in the app; the whole point of
    /// the cache is that a parked meeting does not pay for the same window
    /// twice, and a counter is the only way to state that as a test.
    #[derive(Default)]
    struct CountingModels {
        seen: Vec<i64>,
    }

    impl WindowModels for CountingModels {
        fn analyse(
            &mut self,
            _samples: &[f32],
            start_ms: i64,
            _window_ms: i64,
            previous: Option<&WindowResult>,
            items: &mut Vec<ClusterItem>,
        ) -> Result<WindowResult, DiarizeError> {
            self.seen.push(start_ms);
            let continues = previous.map(|_| Some(0usize)).unwrap_or(None);
            let index = items.len();
            items.push(ClusterItem {
                embedding: vec![1.0, start_ms as f32 / 100_000.0],
                weight_ms: 4_000,
            });
            Ok(WindowResult {
                start_ms,
                tracks: vec![vec![(start_ms, start_ms + 4_000)]],
                confidence: vec![0.9],
                fingerprint: vec![Some(index)],
                continues: vec![continues],
            })
        }
    }

    /// A meeting whose audio is one chunk of noise, long enough for several
    /// windows.
    async fn meeting_with_audio(db: &Db, dir: &std::path::Path, ms: i64) -> String {
        let meeting = repo::create_meeting(db, "Standup", &dir.to_string_lossy(), None)
            .await
            .unwrap();
        let path = dir.join("mic-0.wav");
        // Something every window can find a signal in; the fake models never
        // look at it, but `has_signal` does.
        let samples: Vec<f32> = (0..(ms * 16))
            .map(|i| if i % 2 == 0 { 0.4 } else { -0.4 })
            .collect();
        crate::audio::writer::write_wav_16k_mono(&path, &samples).unwrap();
        let id = repo::insert_chunk(
            db,
            &meeting.id,
            Channel::Mic,
            0,
            &path.to_string_lossy(),
            0,
            ms,
        )
        .await
        .unwrap();
        repo::commit_chunk(db, &id, ms).await.unwrap();
        meeting.id
    }

    /// Two files that exist, so the cache key can identify "the models" without
    /// this test loading a hundred megabytes of ONNX.
    fn model_files(dir: &std::path::Path) -> (std::path::PathBuf, std::path::PathBuf) {
        let segmenter = dir.join("segmenter.onnx");
        let embedder = dir.join("embedder.onnx");
        std::fs::write(&segmenter, b"segmenter").unwrap();
        std::fs::write(&embedder, b"embedder").unwrap();
        (segmenter, embedder)
    }

    async fn key_for(
        db: &Db,
        meeting_id: &str,
        segmenter: &std::path::Path,
        embedder: &std::path::Path,
    ) -> scan_cache::Key {
        let (source, pcm) = voice_channel(db, meeting_id).await.unwrap().unwrap();
        scan_key(db, &pcm, source, segmenter, embedder)
            .await
            .expect("both model files are there")
    }

    /// Run the walk over a whole meeting, resuming from whatever was kept, and
    /// with a recording either in the way or not.
    async fn walk(
        db: &Db,
        meeting_id: &str,
        dir: &std::path::Path,
        key: &scan_cache::Key,
        models: &mut CountingModels,
        recording: bool,
    ) -> Result<Walk, DiarizeError> {
        let (_, mut pcm) = voice_channel(db, meeting_id).await.unwrap().unwrap();
        let covered = pcm.covered();
        let total_ms = pcm.total_ms();
        let step = segmentation::STEP_MS;
        let planned = ((total_ms + step - 1) / step).max(1);

        // A recording holding the machine, expressed the way the session layer
        // expresses it: a flag the pass reads at its next checkpoint.
        let control = if recording {
            DiarizeControl::new().yield_when(Arc::new(|| true))
        } else {
            DiarizeControl::new()
        };

        let mut walk = Walk::fresh();
        if let Some(kept) = scan_cache::load(dir, key).await {
            walk = Walk {
                items: kept.items,
                windows: kept.windows,
                next_start_ms: kept.next_start_ms,
            };
        }

        walk_windows(
            &mut pcm,
            models,
            walk,
            &covered,
            total_ms,
            segmentation::WINDOW_MS,
            step,
            planned,
            Some((dir, key)),
            &control,
        )
        .await
    }

    /// The claim of the whole workstream: a park does not throw away the
    /// minutes of model work that were already paid for.
    #[tokio::test]
    async fn a_parked_scan_picks_up_where_it_stopped_instead_of_starting_over() {
        let dir = tempfile::tempdir().unwrap();
        let db = db().await;
        let meeting = meeting_with_audio(&db, dir.path(), 60_000).await;
        let (segmenter, embedder) = model_files(dir.path());
        let key = key_for(&db, &meeting, &segmenter, &embedder).await;

        // First run: park it before it has done anything.
        let mut first = CountingModels::default();
        let parked = walk(&db, &meeting, dir.path(), &key, &mut first, true).await;
        assert!(
            matches!(parked, Err(DiarizeError::Yielded)),
            "a recording has to park the walk"
        );

        // Prove the file is there and says where to pick up.
        let kept = scan_cache::load(dir.path(), &key)
            .await
            .expect("the parked scan was kept");
        assert_eq!(kept.next_start_ms, 0);

        // Second run: no recording in the way, so it finishes.
        let mut second = CountingModels::default();
        let done = walk(&db, &meeting, dir.path(), &key, &mut second, false)
            .await
            .expect("the resumed scan finished");
        assert_eq!(done.next_start_ms, 60_000);

        // Third run over the finished cache: every window is already done, so
        // the models are asked for nothing at all.
        let mut third = CountingModels::default();
        let again = walk(&db, &meeting, dir.path(), &key, &mut third, false)
            .await
            .expect("nothing left to do");
        assert!(
            third.seen.is_empty(),
            "a finished scan was walked again: {:?}",
            third.seen
        );
        assert_eq!(again.windows.len(), done.windows.len());
        assert_eq!(again.items.len(), done.items.len());
    }

    /// Halfway is the case that matters on a real machine: some windows paid
    /// for, the rest still owed. Neither window is analysed twice, and the
    /// fingerprints of both halves end up in one scan.
    #[tokio::test]
    async fn the_windows_already_paid_for_are_not_paid_for_again() {
        let dir = tempfile::tempdir().unwrap();
        let db = db().await;
        let meeting = meeting_with_audio(&db, dir.path(), 60_000).await;
        let (segmenter, embedder) = model_files(dir.path());
        let key = key_for(&db, &meeting, &segmenter, &embedder).await;

        // Do the first half by hand, then keep it exactly as a park would.
        let mut first = CountingModels::default();
        let (_, mut pcm) = voice_channel(&db, &meeting).await.unwrap().unwrap();
        let covered = pcm.covered();
        let control = DiarizeControl::new();
        let half = walk_windows(
            &mut pcm,
            &mut first,
            Walk::fresh(),
            &covered,
            30_000,
            segmentation::WINDOW_MS,
            segmentation::STEP_MS,
            12,
            Some((dir.path(), &key)),
            &control,
        )
        .await
        .unwrap();
        assert_eq!(first.seen, vec![0, 5_000, 10_000, 15_000, 20_000, 25_000]);
        assert_eq!(half.next_start_ms, 30_000);

        let mut second = CountingModels::default();
        let whole = walk(&db, &meeting, dir.path(), &key, &mut second, false)
            .await
            .unwrap();
        assert_eq!(
            second.seen,
            vec![30_000, 35_000, 40_000, 45_000, 50_000, 55_000],
            "the second run redid work the first one had already done"
        );
        assert_eq!(
            whole.windows.len(),
            12,
            "both halves are in the finished scan"
        );
        assert_eq!(whole.items.len(), 12);
    }

    /// A cancel is not a park. The commonest reason for one is somebody
    /// deleting the meeting, and `commands::delete_meeting` cancels the work and
    /// then removes the folder without waiting — so a scan that wrote itself
    /// down on its way out could recreate the very file the delete had just
    /// taken away, leaving this meeting's voice fingerprints on disk after
    /// somebody asked Echo to forget the recording.
    #[tokio::test]
    async fn a_cancelled_scan_leaves_nothing_behind() {
        let dir = tempfile::tempdir().unwrap();
        let db = db().await;
        let meeting = meeting_with_audio(&db, dir.path(), 30_000).await;
        let (segmenter, embedder) = model_files(dir.path());
        let key = key_for(&db, &meeting, &segmenter, &embedder).await;

        let (_, mut pcm) = voice_channel(&db, &meeting).await.unwrap().unwrap();
        let covered = pcm.covered();
        let control = DiarizeControl::new();
        control.cancel();
        let mut models = CountingModels::default();
        let stopped = walk_windows(
            &mut pcm,
            &mut models,
            Walk::fresh(),
            &covered,
            30_000,
            segmentation::WINDOW_MS,
            segmentation::STEP_MS,
            6,
            Some((dir.path(), &key)),
            &control,
        )
        .await;

        assert!(
            matches!(stopped, Err(DiarizeError::Cancelled)),
            "{stopped:?}"
        );
        assert!(
            !scan_cache::path_in(dir.path()).exists(),
            "a cancelled pass wrote itself into a folder that may be being deleted"
        );
    }

    /// Where the fingerprints are allowed to live: a folder named after this
    /// meeting and nothing else.
    ///
    /// A meeting row is created against the storage root and only afterwards
    /// told about its own folder, and that second write is allowed to fail
    /// (`session::start`). A row still naming the root would put this meeting's
    /// voice fingerprints where deleting the meeting never looks — and where
    /// every other such meeting writes the same file.
    #[tokio::test]
    async fn a_meeting_with_no_folder_of_its_own_keeps_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let db = db().await;

        let stray = repo::create_meeting(&db, "Standup", &dir.path().to_string_lossy(), None)
            .await
            .unwrap();
        assert_eq!(
            audio_dir_of(&db, &stray.id).await,
            None,
            "a row naming the storage root must not be written into"
        );

        let own = dir.path().join("meeting-of-its-own");
        std::fs::create_dir_all(&own).unwrap();
        let proper = repo::create_meeting(&db, "Standup", &own.to_string_lossy(), None)
            .await
            .unwrap();
        // The folder has to be named after the meeting; rename it to the id the
        // row was actually given.
        let named = dir.path().join(&proper.id);
        std::fs::rename(&own, &named).unwrap();
        repo::set_meeting_audio_dir(&db, &proper.id, &named.to_string_lossy())
            .await
            .unwrap();
        assert_eq!(audio_dir_of(&db, &proper.id).await, Some(named));
    }

    /// The safety property. A cache that no longer describes this meeting's
    /// audio is worth less than nothing — it would put one person's voice on
    /// another person's lines — so it is thrown away and the work is done
    /// again.
    #[tokio::test]
    async fn audio_that_arrived_late_throws_the_kept_scan_away() {
        let dir = tempfile::tempdir().unwrap();
        let db = db().await;
        let meeting = meeting_with_audio(&db, dir.path(), 30_000).await;
        let (segmenter, embedder) = model_files(dir.path());
        let key = key_for(&db, &meeting, &segmenter, &embedder).await;

        let mut first = CountingModels::default();
        walk(&db, &meeting, dir.path(), &key, &mut first, false)
            .await
            .unwrap();
        assert_eq!(first.seen.len(), 6);

        // A second chunk lands — a recovery after a crash, a late commit — and
        // the meeting is no longer the one that was scanned.
        let path = dir.path().join("mic-1.wav");
        let samples: Vec<f32> = (0..(30_000 * 16))
            .map(|i| if i % 2 == 0 { 0.4 } else { -0.4 })
            .collect();
        crate::audio::writer::write_wav_16k_mono(&path, &samples).unwrap();
        let id = repo::insert_chunk(
            &db,
            &meeting,
            Channel::Mic,
            1,
            &path.to_string_lossy(),
            30_000,
            60_000,
        )
        .await
        .unwrap();
        repo::commit_chunk(&db, &id, 60_000).await.unwrap();

        let fresh_key = key_for(&db, &meeting, &segmenter, &embedder).await;
        assert_ne!(fresh_key, key, "the key has to notice the new audio");
        assert!(
            scan_cache::load(dir.path(), &fresh_key).await.is_none(),
            "the old scan must not be resumed over audio it never saw"
        );

        let mut second = CountingModels::default();
        walk(&db, &meeting, dir.path(), &fresh_key, &mut second, false)
            .await
            .unwrap();
        assert_eq!(
            second.seen.first().copied(),
            Some(0),
            "a stale cache has to mean a full rescan, from the first window"
        );
        assert_eq!(second.seen.len(), 12);
    }

    /// A finished scan is the one case where the pass can answer without a
    /// model at all — which is also how this test can prove it: the model paths
    /// are files no ONNX runtime could ever load, so reaching for them fails
    /// loudly.
    #[tokio::test]
    async fn a_finished_scan_is_read_back_without_loading_a_model() {
        let dir = tempfile::tempdir().unwrap();
        let db = db().await;
        let meeting = meeting_with_audio(&db, dir.path(), 30_000).await;
        let (segmenter, embedder) = model_files(dir.path());
        let key = key_for(&db, &meeting, &segmenter, &embedder).await;

        let mut models = CountingModels::default();
        walk(&db, &meeting, dir.path(), &key, &mut models, false)
            .await
            .unwrap();

        let scanned = scan_resumable(
            &db,
            &meeting,
            &segmenter,
            &embedder,
            Some(dir.path()),
            &DiarizeControl::new(),
        )
        .await
        .expect("the kept scan answered")
        .expect("there is audio");
        assert_eq!(scanned.fingerprints(), 6);

        // And without the cache, the same call has to load the models and
        // fails on these files — which is what makes the assertion above mean
        // "it did not load them".
        std::fs::remove_file(scan_cache::path_in(dir.path())).unwrap();
        let without_the_cache = scan_resumable(
            &db,
            &meeting,
            &segmenter,
            &embedder,
            Some(dir.path()),
            &DiarizeControl::new(),
        )
        .await;
        match without_the_cache {
            Err(DiarizeError::Load(_)) => {}
            Err(other) => panic!("{other:?}"),
            Ok(_) => panic!("these files are not models; loading them cannot succeed"),
        }
    }

    /// A model file that cannot be identified switches the cache off entirely
    /// rather than being guessed at.
    #[tokio::test]
    async fn without_a_model_to_name_nothing_is_kept() {
        let dir = tempfile::tempdir().unwrap();
        let db = db().await;
        let meeting = meeting_with_audio(&db, dir.path(), 10_000).await;
        let (_, pcm) = voice_channel(&db, &meeting).await.unwrap().unwrap();
        assert!(scan_key(
            &db,
            &pcm,
            Source::MicOnly,
            std::path::Path::new("/nonexistent/segmenter.onnx"),
            std::path::Path::new("/nonexistent/embedder.onnx"),
        )
        .await
        .is_none());
    }

    // -----------------------------------------------------------------------
    // A park during the clustering
    // -----------------------------------------------------------------------

    /// Enough fingerprints that building the merge tree is real work.
    fn many_prints(n: usize) -> Scan {
        let mut items = Vec::new();
        let mut windows = Vec::new();
        for i in 0..n {
            let start = i as i64 * 1_000;
            let mut embedding = vec![0.0f32; 64];
            embedding[i % 64] = 1.0;
            embedding[(i * 7) % 64] += 0.5;
            cluster::l2_normalize(&mut embedding);
            windows.push(WindowResult {
                start_ms: start,
                tracks: vec![vec![(start, start + 900)]],
                confidence: vec![0.9],
                fingerprint: vec![Some(items.len())],
                continues: vec![None],
            });
            items.push(ClusterItem {
                embedding,
                weight_ms: 900,
            });
        }
        Scan {
            items,
            windows,
            covered: vec![(0, n as i64 * 1_000)],
        }
    }

    /// The clustering reads nothing off disk, so before this it was minutes
    /// during which a starting recording simply had to wait. Now it stops at the
    /// first thing it does, and stops with no answer rather than half of one.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_recording_starting_during_the_clustering_is_honoured_at_once() {
        let scanned = many_prints(400);
        let guided = Guided::default();

        let undisturbed = std::time::Instant::now();
        let full = scanned.cut_guided(&guided, None);
        let full_took = undisturbed.elapsed();
        assert!(!full.tracks.is_empty());

        let parked = std::time::Instant::now();
        let control = DiarizeControl::new().yield_when(Arc::new(|| true));
        let err = scanned
            .cut_guided_until(&guided, None, &control)
            .expect_err("the recording wins");
        let parked_took = parked.elapsed();
        assert!(matches!(err, DiarizeError::Yielded), "{err:?}");
        assert!(
            parked_took * 4 < full_took,
            "the clustering ran on regardless: {parked_took:?} against {full_took:?}"
        );
    }

    /// The same, for the path a count correction takes.
    #[test]
    fn a_forced_count_stops_for_a_recording_too() {
        let scanned = many_prints(200);
        let control = DiarizeControl::new().yield_when(Arc::new(|| true));
        let err = scanned
            .cut_guided_until(&Guided::default(), Some(3), &control)
            .expect_err("the recording wins");
        assert!(matches!(err, DiarizeError::Yielded), "{err:?}");
    }

    /// Cancelling is not parking, and the two must not be confused: one is the
    /// person saying stop, the other is Echo standing aside.
    #[test]
    fn cancelling_during_the_clustering_says_cancelled() {
        let scanned = many_prints(50);
        let control = DiarizeControl::new();
        control.cancel();
        let err = scanned
            .cut_guided_until(&Guided::default(), None, &control)
            .expect_err("the person stopped it");
        assert!(matches!(err, DiarizeError::Cancelled), "{err:?}");
    }
}
