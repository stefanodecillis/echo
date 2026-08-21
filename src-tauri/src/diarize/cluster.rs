//! Grouping voice fingerprints into people, and lining up window-local labels.
//!
//! Two independent jobs, both pure and both tested against synthetic
//! fingerprints so neither needs a model on disk:
//!
//! * [`cluster_auto`] decides how many people spoke and which fingerprint
//!   belongs to whom. Agglomerative, centroid linkage, cosine distance — and
//!   then the count is read out of the **shape of the merge tree** rather than
//!   off a single calibrated distance. See [`CountChoice`] for the criterion and
//!   [`DISTANCE_THRESHOLD`] for what is left of that distance (a floor and a
//!   ceiling, nothing more).
//! * [`cluster`] is the same hierarchy cut at one distance, which is what the
//!   calibration tools sweep. The app does not call it any more.
//! * [`cluster_fixed`] is the same hierarchy cut at a count the person gave us
//!   instead of at a distance. A number a human states is better evidence than
//!   any threshold, so this path ignores the threshold entirely.
//! * [`align_permutation`] lines up one window's local speaker labels with the
//!   previous window's, using the audio the two windows share. Clustering owns
//!   the final identity; alignment is what keeps a speaker who talks too little
//!   to be fingerprinted from becoming an anonymous gap.

use std::collections::HashMap;

/// Cosine distance above which two fingerprints are different people.
///
/// **This is no longer what decides how many people were in a meeting.** It was,
/// until it got a two-person conversation wrong in the field on 2026-08-21; the
/// automatic count is now read out of the shape of the merge tree
/// ([`CountChoice`]) and this number survives as three smaller things:
///
/// * the pooling radius for a long meeting (half of it — see [`leaves`]),
/// * the tie-break when two counts explain the fingerprints equally well,
/// * and the number [`SPLIT_FLOOR`] and [`FUSE_CEILING`] are placed around.
///
/// Everything below is the measurement it came from, kept because those two
/// bounds are placed by it and because it is the record of why a single
/// calibrated distance was not enough. [`cluster`] still cuts at it, and the
/// calibration tools still sweep it.
///
/// **This constant belongs to one specific fingerprint network**, and to cosine
/// distance with centroid linkage. It is not a taste setting.
///
/// # Which network
///
/// `catalog::ids::EMBEDDER` — WeSpeaker **ResNet34-LM**, 256 dimensions, the
/// voice-print model of the reference pyannote 3.1 and community-1 pipelines.
///
/// Before 2026-08-20 the catalog shipped WeSpeaker CAM++ (512 dimensions, a
/// different embedding space entirely) while this constant held a number
/// published for ResNet34-LM. The one value deciding how many people Echo thought
/// were in the room described a geometry it had never been measured in. Swapping
/// the asset is what fixed that; see the embedder entry in
/// [`crate::asr::catalog::CATALOG`].
///
/// # Where 0.58 comes from
///
/// Not from upstream. pyannote 3.1 ships 0.7046 for this network (0.7153 in
/// 3.0), and **that number does not reproduce here** — measured, not assumed.
/// Both pipelines run "centroid-linkage agglomerative clustering over cosine
/// distance" and the phrase hides real differences: pyannote reaches its
/// cluster-to-cluster numbers through scipy's Lance-Williams update over a cosine
/// distance matrix, where [`cluster`] recomputes each centroid and returns it to
/// the unit sphere, and pyannote discards any cluster holding fewer than twelve
/// embeddings where [`MIN_CLUSTER_MS`] counts milliseconds. Feed those two
/// algorithms the same fingerprints and the same cut lands in a different place.
///
/// So the number was measured against audio whose answer is known, twice, and
/// 0.58 is the middle of what both measurements allow.
///
/// **Synthetic, many people.** `cargo run --release --example voices_fixture`
/// builds meetings out of macOS's own text-to-speech voices — 2, 3, 5 and 7
/// *known* people, three turns each of about twelve seconds, short gaps between
/// them — runs the real pass ([`super::pipeline::scan`], the same code the app
/// runs) over each one, and then cuts the same fingerprints at every threshold
/// from 0.15 to 1.10. Every threshold in **0.2375 … 0.625** gets all four
/// fixtures exactly right, each cluster at least 97.9% one true voice. Above
/// about 0.64 the seven-person fixture starts fusing people (six, then five);
/// below about 0.22 it starts inventing an eighth. Those edges move by one
/// 0.0125 step between runs, because `say` does not render bit-identically twice;
/// the interval above is what two runs agreed on.
///
/// **Real, two people.** `cargo run --release --example speakers_probe -- --sweep`
/// over a real 21-minute recorded meeting: exactly two voices, cleanly split
/// 650 s / 465 s, for every threshold in **0.5625 … 0.6050**. At 0.6075 the two
/// speakers fuse into one — which is what pyannote's own 0.7046 would do to this
/// meeting, and what makes this the measurement that mattered. Below 0.5575 the
/// same two voices are still right, but 4–7 second fragments start surviving
/// [`MIN_CLUSTER_MS`] and showing up as extra people.
///
/// The two windows overlap in 0.5625 … 0.6050, and 0.58 sits in the middle of it
/// — 0.0175 clear of the fragment edge below, 0.0275 clear of the fusion edge
/// above, and a long way inside the synthetic plateau.
///
/// # What this is not
///
/// It is not a diarization error rate. Text-to-speech voices are cleaner and far
/// more self-consistent than people: no room, no overlap, no crosstalk, no
/// laughter, and a synthetic voice does not drift the way a real one does inside
/// a single sentence. And one real meeting is one real meeting — the interval
/// above is 0.04 wide, which is narrow, and a second meeting could move it. What
/// the two measurements together do rule out is the failure this was written to
/// end: a threshold that has never been measured against the network it is
/// deciding with.
///
/// # Which way to be wrong
///
/// Lower splits one person into several, or keeps a fragment as a person
/// ("Speaker 3" turns up halfway through a meeting); higher fuses two people into
/// one. Fusing is the worse failure — a merge is a thing the UI asks people to
/// do, but nothing lets them say "these two were actually different" — so where
/// the interval allows a choice, 0.58 leans low.
///
/// # What happened next
///
/// On 2026-08-21 the same meeting was retranscribed, the segment boundaries moved
/// a little, the fingerprints moved with them, and the band above slid under
/// 0.58: the two speakers fused and Echo reported one voice for a two-person
/// conversation. The forced-count path still split them correctly, which is what
/// made it obvious the fingerprints were fine and the *decision* was not.
///
/// The interval was 0.04 wide and the shipped number had 0.0275 of clearance
/// above it. That is not a number that was measured badly; it is a decision
/// procedure with no margin to give. See [`CountChoice`] for what replaced it.
pub const DISTANCE_THRESHOLD: f32 = 0.58;

/// A cluster holding less speech than this is a fragment, not a person. Its
/// fingerprints are handed to whoever they resemble most instead of becoming a
/// "Speaker 5" the person then has to merge away.
pub const MIN_CLUSTER_MS: i64 = 3_000;

/// Two clusters this near are never told apart by the automatic count.
///
/// The floor half of what is left of [`DISTANCE_THRESHOLD`] once the count comes
/// out of the merge tree instead of out of a single cut (see [`CountChoice`]).
/// It is not a cut: it says only that a cut refusing to merge a pair *closer*
/// than this would be claiming one voice is two, whatever the shape of the tree
/// says. A meeting where nobody is further from anybody than this is one person
/// talking.
///
/// **Where it comes from.** The real two-person meeting in
/// `examples/speakers_probe.rs` puts the two people 0.606 apart and puts every
/// merge *inside* a voice at or below 0.556 — and the synthetic fixtures hold
/// their voices together within 0.46
/// (`examples/voices_fixture.rs`, `--matrix`). 0.50 sits under both real
/// numbers with room to spare and over the synthetic within-voice spread, and
/// it is deliberately far below the distance that decides anything: the
/// criterion, not this number, is what picks the count. Compare the 0.0275 of
/// clearance the shipped 0.58 cut had.
pub const SPLIT_FLOOR: f32 = 0.50;

/// Two clusters this far apart are never made one person by the automatic
/// count, however few people the tree's shape suggests.
///
/// The ceiling half of the prior, and the safety net for the failure the
/// criterion exists to end: fusing two people is the worse way to be wrong
/// (see [`DISTANCE_THRESHOLD`]), so a cut that would swallow a merge this wide
/// is refused outright and the count is forced up. Above pyannote 3.1's own
/// published 0.7046 for this network, so it only ever fires on a pair no
/// published threshold would have merged either.
pub const FUSE_CEILING: f32 = 0.75;

/// A cluster holding less than this share of a meeting's fingerprinted speech is
/// a fragment for the automatic count, whatever [`MIN_CLUSTER_MS`] says.
///
/// [`MIN_CLUSTER_MS`] asks the same question in absolute milliseconds and is
/// kept for what it is good at — a 300 ms sliver in a short recording. This asks
/// it in proportion, which is what the real failure needed. The two-person
/// meeting in `examples/speakers_probe.rs` produces, besides its two people, a
/// tail of ten one-to-four second fingerprints that sit far from both of them —
/// door noise, a cough, half a word over the top of somebody. Every one is past
/// `MIN_CLUSTER_MS`, every one is under 0.2% of the conversation, and between
/// them they take ten of the twelve cluster slots Echo has. Nobody was in that
/// meeting for two seconds.
///
/// 1% of a twenty-minute meeting is about nineteen seconds of speech.
/// [`fragment_bar`] is where this is turned into a number, with
/// [`MAX_FRAGMENT_MS`] stopping it from asking a long meeting for minutes of
/// speech before it will call somebody a person.
///
/// A count the person states is not filtered by this at all: [`cluster_fixed`]
/// folds nothing away, because somebody who says "morning" is still one of the
/// four people in the room when a human has said there were four.
pub const MIN_CLUSTER_SHARE: f32 = 0.01;

/// However long the meeting, a cluster holding this much speech is somebody.
///
/// Without it [`MIN_CLUSTER_SHARE`] would want 108 seconds of a three-hour
/// meeting before it believed in a person, and plenty of real people in real
/// three-hour meetings say less than that.
pub const MAX_FRAGMENT_MS: i64 = 30_000;

/// Below how much speech a cluster is a fragment rather than a person, for a
/// meeting holding `speech_ms` of fingerprinted speech.
///
/// See [`MIN_CLUSTER_SHARE`] and [`MAX_FRAGMENT_MS`]. Never below
/// [`MIN_CLUSTER_MS`], so a short recording keeps the absolute rule it has
/// always had.
pub fn fragment_bar(speech_ms: i64) -> i64 {
    let proportional = (speech_ms as f64 * f64::from(MIN_CLUSTER_SHARE)) as i64;
    proportional.clamp(MIN_CLUSTER_MS, MAX_FRAGMENT_MS)
}

/// Hard ceiling on people per meeting. Real meetings do not have thirty
/// distinct voices; a count that high means the fingerprints are noise, and a
/// wrong-but-small speaker list is far kinder than a wall of chips.
pub const MAX_SPEAKERS: usize = 12;

/// Above this many fingerprints, an exact all-pairs pass gets expensive, so
/// near-identical fingerprints are pooled first at half the threshold. A
/// two-hour meeting produces roughly 1,500.
const MAX_EXACT_ITEMS: usize = 600;

/// One fingerprint with the amount of speech behind it.
#[derive(Debug, Clone, Default)]
pub struct ClusterItem {
    pub embedding: Vec<f32>,
    /// How much speech the fingerprint was computed from. Longer is more
    /// trustworthy, so it weighs more in the centroid.
    pub weight_ms: i64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Clustering {
    /// One label per input item.
    pub labels: Vec<usize>,
    pub cluster_count: usize,
    /// The threshold actually used, recorded for diagnostics.
    pub threshold: f32,
}

/// Scale a vector to unit length. A zero vector is left alone (its distance to
/// everything is then 1.0, which is the honest answer for "no information").
pub fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// `1 - cos(a, b)`, in `0.0..=2.0`. Both vectors are assumed unit length.
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    (1.0 - dot).clamp(0.0, 2.0)
}

/// Internal cluster state during the merge loop.
#[derive(Clone)]
struct Node {
    /// Weighted sum of member fingerprints; the centroid is this, normalised.
    sum: Vec<f32>,
    centroid: Vec<f32>,
    weight: f64,
    weight_ms: i64,
    members: Vec<usize>,
    alive: bool,
}

impl Node {
    fn leaf(index: usize, embedding: &[f32], weight_ms: i64) -> Self {
        let w = (weight_ms.max(1)) as f64;
        let sum: Vec<f32> = embedding.iter().map(|x| x * w as f32).collect();
        Self {
            sum,
            centroid: embedding.to_vec(),
            weight: w,
            weight_ms: weight_ms.max(0),
            members: vec![index],
            alive: true,
        }
    }

    fn recentre(&mut self) {
        self.centroid.clear();
        self.centroid.extend_from_slice(&self.sum);
        l2_normalize(&mut self.centroid);
    }
}

/// Agglomerative clustering, centroid linkage, cosine distance.
///
/// Merges the closest pair of clusters until the closest pair is further apart
/// than `threshold`, then keeps merging if that left more than [`MAX_SPEAKERS`]
/// of them. Fragments below [`MIN_CLUSTER_MS`] are folded into their nearest
/// neighbour afterwards.
///
/// Labels are arbitrary integers; the caller renumbers them in the order people
/// first speak, so "Speaker 1" is the first voice heard.
pub fn cluster(items: &[ClusterItem], threshold: f32) -> Clustering {
    if items.is_empty() {
        return Clustering {
            labels: Vec::new(),
            cluster_count: 0,
            threshold,
        };
    }

    let Some(embeddings) = normalised(items) else {
        // Nothing usable: everyone is one speaker rather than a crash.
        return Clustering {
            labels: vec![0; items.len()],
            cluster_count: 1,
            threshold,
        };
    };

    // Above MAX_EXACT_ITEMS the leaves are pools rather than single
    // fingerprints; the pre-pass groups at half the threshold, so it only ever
    // groups fingerprints the exact pass would certainly have grouped anyway.
    let mut nodes = leaves(items, &embeddings, threshold / 2.0);
    merge_loop(&mut nodes, threshold, MAX_SPEAKERS);
    // The absolute fragment rule, which is the one this path has always had. The
    // automatic count uses the proportional one instead ([`fragment_bar`]); this
    // path keeps the old bar so the sweeps that measured
    // [`DISTANCE_THRESHOLD`] stay comparable with what they measured.
    absorb_fragments(&mut nodes, MIN_CLUSTER_MS);
    finish(nodes, items.len(), threshold)
}

/// Unit-length copies of every fingerprint, or `None` when they cannot be
/// compared at all — no dimensions, or not all the same length. Every distance
/// in this module assumes unit length.
fn normalised(items: &[ClusterItem]) -> Option<Vec<Vec<f32>>> {
    let mut embeddings: Vec<Vec<f32>> = items.iter().map(|i| i.embedding.clone()).collect();
    for e in embeddings.iter_mut() {
        l2_normalize(e);
    }
    let dim = embeddings.first()?.len();
    if dim == 0 || embeddings.iter().any(|e| e.len() != dim) {
        return None;
    }
    Some(embeddings)
}

/// The leaves of the hierarchy: one node per fingerprint, or — above
/// [`MAX_EXACT_ITEMS`], where an exact all-pairs pass gets expensive — one node
/// per pool of near-identical fingerprints.
fn leaves(items: &[ClusterItem], embeddings: &[Vec<f32>], pool_radius: f32) -> Vec<Node> {
    if items.len() > MAX_EXACT_ITEMS {
        return pool(items, embeddings, pool_radius);
    }
    embeddings
        .iter()
        .enumerate()
        .map(|(i, e)| Node::leaf(i, e, items[i].weight_ms))
        .collect()
}

/// Group fingerprints that are within `radius` of each other into single nodes,
/// so the all-pairs merge loop runs over pools rather than over every window.
fn pool(items: &[ClusterItem], embeddings: &[Vec<f32>], radius: f32) -> Vec<Node> {
    let mut pools: Vec<Node> = Vec::new();
    for (i, e) in embeddings.iter().enumerate() {
        let mut best: Option<(usize, f32)> = None;
        for (p, existing) in pools.iter().enumerate() {
            let d = cosine_distance(e, &existing.centroid);
            match best {
                Some((_, bd)) if d >= bd => {}
                _ => best = Some((p, d)),
            }
        }
        match best {
            Some((p, d)) if d < radius => {
                let w = items[i].weight_ms.max(1) as f64;
                for (slot, x) in pools[p].sum.iter_mut().zip(e.iter()) {
                    *slot += x * w as f32;
                }
                pools[p].weight += w;
                pools[p].weight_ms += items[i].weight_ms.max(0);
                pools[p].members.push(i);
                pools[p].recentre();
            }
            _ => pools.push(Node::leaf(i, e, items[i].weight_ms)),
        }
    }
    pools
}

/// The same hierarchy, cut at exactly `k` clusters instead of at a distance.
///
/// This is what an override runs: the person has told Echo how many people were
/// in the meeting, and a number a human states about a conversation they were in
/// beats any threshold measured on somebody else's corpus. So the calibrated
/// distance is not consulted at all — the closest pair keeps merging until `k`
/// are left, however near or far the last merge was.
///
/// Two things this deliberately does *not* do, both because they would quietly
/// overrule the person:
///
/// * **No fragment absorption.** [`absorb_fragments`] exists to stop a sliver of
///   speech becoming a "Speaker 5" nobody asked for. Here somebody *did* ask:
///   a person who only says "morning" is still one of the four people in the
///   room, and folding them away would hand back three.
/// * **No ceiling below `k`.** `k` is clamped into `1..=`[`MAX_SPEAKERS`] by the
///   caller and again here, and nothing else shrinks it.
///
/// Honest about not being able to deliver: with fewer fingerprints than `k`
/// there is no way to show `k` distinct voices, so it returns as many as it
/// actually has rather than inventing the rest.
pub fn cluster_fixed(items: &[ClusterItem], k: usize) -> Clustering {
    let k = k.clamp(1, MAX_SPEAKERS);
    if items.is_empty() {
        return Clustering {
            labels: Vec::new(),
            cluster_count: 0,
            threshold: 0.0,
        };
    }

    let Some(embeddings) = normalised(items) else {
        // Nothing usable to measure with: one speaker rather than a crash, the
        // same answer `cluster` gives.
        return Clustering {
            labels: vec![0; items.len()],
            cluster_count: 1,
            threshold: 0.0,
        };
    };

    // Fewer voices to hand out than the person asked for: give back what exists.
    if items.len() <= k {
        return Clustering {
            labels: (0..items.len()).collect(),
            cluster_count: items.len(),
            threshold: 0.0,
        };
    }

    // Same pre-pass as the automatic path, and for the same reason: an exact
    // all-pairs loop over a two-hour meeting is expensive. Pooling at half the
    // calibrated distance only ever groups fingerprints the exact pass would
    // certainly have grouped, so it cannot change where the cut lands.
    let mut nodes = leaves(items, &embeddings, DISTANCE_THRESHOLD / 2.0);
    let cut = merge_to_k(&mut nodes, k);
    finish(nodes, items.len(), cut)
}

// ===========================================================================
// The automatic count: reading it out of the shape of the merge tree
// ===========================================================================

/// One merge of the hierarchy, recorded so a cut can be replayed without
/// searching for the closest pair all over again.
#[derive(Debug, Clone, Copy)]
struct Merge {
    a: usize,
    b: usize,
    distance: f32,
}

/// One rung of the merge ladder: the two nearest clusters were this far apart
/// when they became one, which left `clusters` of them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rung {
    pub clusters: usize,
    pub distance: f32,
}

/// Why a candidate count is not one the automatic pass will hand back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// It splits a pair closer together than [`SPLIT_FLOOR`] — one voice
    /// disagreeing with itself, not two people.
    OneVoice,
    /// It merges two clusters, both of them big enough to be somebody, at least
    /// [`FUSE_CEILING`] apart — two people made one.
    Fused,
    /// It needs more clusters than [`MAX_SPEAKERS`], counting only the ones big
    /// enough to be people.
    TooMany,
}

/// One candidate answer to "how many people were in this meeting", with
/// everything the criterion weighs against it.
///
/// Every field here is printed by `examples/speakers_probe.rs`. A count that
/// cannot be inspected is a count nobody can argue with, and the number this
/// replaces was wrong in the field for exactly that reason.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    /// People this cut ends up with: clusters left once the ones too small to be
    /// anybody are handed to whoever they resemble.
    pub count: usize,
    /// Clusters the tree is cut into to get there. Larger than `count` whenever
    /// the cut also produces fragments — which on a real meeting is most of the
    /// time, and by a lot; see [`CountChoice`].
    pub cut_clusters: usize,
    /// How many of those clusters were fragments, folded away.
    pub folded: usize,
    /// Height of the last merge the cut accepted; `0.0` when it accepted none.
    pub accepted: f32,
    /// Height of the merge the cut refuses — how far apart the two nearest
    /// clusters still are. `0.0` for the whole tree merged into one.
    pub refused: f32,
    /// `refused / accepted`, the relative gap in the ladder at this rung. Kept
    /// because it is the first thing anybody looks for, and because on the real
    /// meeting it is the column that gets the answer *wrong* — see
    /// [`CountChoice`].
    pub gap: f32,
    /// Weighted mean silhouette of the cut: how much better every fingerprint
    /// fits the voice it was given than the nearest other voice. This is the
    /// criterion.
    pub silhouette: f32,
    /// Speech behind each surviving cluster, largest first.
    pub mass_ms: Vec<i64>,
    /// `None` when the prior allows this count.
    pub refusal: Option<Refusal>,
}

impl Candidate {
    pub fn allowed(&self) -> bool {
        self.refusal.is_none()
    }

    /// How far [`DISTANCE_THRESHOLD`] is from the band of distances this cut is
    /// the answer for. Zero when the old shipped cut would have agreed.
    pub fn distance_from_shipped(&self) -> f32 {
        if self.refused <= 0.0 {
            return (DISTANCE_THRESHOLD - self.accepted).max(0.0);
        }
        (self.accepted - DISTANCE_THRESHOLD)
            .max(DISTANCE_THRESHOLD - self.refused)
            .max(0.0)
    }
}

/// How many people the automatic pass decided there were, and the evidence.
///
/// # Why the count no longer comes off one distance
///
/// It used to: [`cluster`] merged until the closest pair was further apart than
/// [`DISTANCE_THRESHOLD`] and however many clusters were left was the answer.
/// That number was measured, twice, and it still failed in the field on
/// 2026-08-21 — a two-person conversation came back as one voice. Nothing was
/// wrong with the measurement. On that meeting every distance in 0.5625 … 0.6050
/// gives the right answer and 0.6125 gives one voice, so the whole correct answer
/// lived in 0.0275 of cosine distance above the 0.58 that shipped; a
/// retranscription moved the segment boundaries slightly, the fingerprints moved
/// with them, and the band slid under the cut. Any single number has that failure
/// available to it. The margin *is* the bug.
///
/// So the count is read out of the shape of the tree instead, where the same
/// meeting is not marginal at all.
///
/// # The criterion: weighted silhouette of the candidate cuts
///
/// [`cluster_auto`] records the whole hierarchy once, then for every height it
/// could be cut at: cuts there, hands every cluster below [`fragment_bar`] to
/// whoever it resembles, and scores what is left by **weighted mean
/// silhouette** against the cluster centroids — the same geometry the linkage
/// works in. For each fingerprint, `a` is its distance to its own voice's
/// centroid with itself taken out of it, `b` is its distance to the nearest other
/// voice's centroid, and its score is `(b - a) / max(a, b)`: +1 for a fingerprint
/// that plainly belongs where it is, 0 for one that could go either way,
/// negative for one in the wrong place. Fingerprints weigh what their speech
/// weighs. The count with the best score wins.
///
/// This is scale-free in exactly the way the old cut was not. It never asks "are
/// these two 0.58 apart"; it asks "does splitting here explain the fingerprints
/// better than not splitting here".
///
/// # What a real meeting's tree actually looks like
///
/// This matters, because it is not what the shape of the problem looks like from
/// a distance. The 21-minute two-person meeting in `examples/speakers_probe.rs`
/// produces 345 fingerprints holding 1,886 s of speech between them, and its
/// merge ladder near the top is:
///
/// ```text
/// clusters   merge height   relative gap   people, after folding
///        9         0.5482          1.014   -
///        8         0.5560          1.011   -
///        7         0.5624          1.076   2   <- chosen, silhouette 0.705
///        6         0.6052          1.000   -
///        5         0.6051          1.190   1
///        4         0.7200          1.065   -
///        3         0.7665          1.053   -
///        2         0.8071          1.003   -
///        1         0.8096              -   1   silhouette 0
/// ```
///
/// The two people are 1,079 s and 807 s of that speech, and they fuse at 0.6052.
/// Everything above 0.6052 is *fragments merging*: ten one-to-four second
/// fingerprints — a cough, a door, half a word said over the top of somebody —
/// which sit further from everything than the two people sit from each other.
/// They take ten of the twelve cluster slots, so the two-person answer is a cut
/// into **seven** clusters with five of them folded away. That is why the count
/// has to be read after folding and not before, and it is why
/// [`MIN_CLUSTER_SHARE`] exists.
///
/// The winning margin is the whole point: two voices at 0.705 against one voice
/// at 0. Nothing a shifted segment boundary does to the fingerprints crosses
/// that.
///
/// # Why not the largest relative gap
///
/// Because it is measurably wrong here, and wrong in the direction that failed in
/// the field. The largest relative gap over every cut height of that ladder is
/// 1.190, at a cut into five clusters — one person and four fragments, which
/// folds to **one voice**. Cutting at the biggest jump reproduces the bug.
///
/// Worth knowing *why* it looks like a good idea: on the synthetic fixtures in
/// `examples/voices_fixture.rs` the gap is enormous and correct (4.074 at the
/// right cut on the two-voice fixture), because text-to-speech voices produce no
/// fragments at all. A criterion tuned on clean fixtures and a criterion that
/// survives a real recording are not the same criterion, and that is the same
/// mistake in a new place as the threshold this replaces. The gap is kept as a
/// printed column ([`Candidate::gap`]) because it is the first thing a reader
/// reaches for, and because this is the evidence for not reaching for it.
///
/// # What the calibrated distance is still for
///
/// A bound at each end, and nothing else. The criterion cannot be argued into a
/// count by either of them; they only ever remove candidates:
///
/// * [`SPLIT_FLOOR`] — no cut that separates a pair closer than this.
/// * [`FUSE_CEILING`] — no cut that merges two clusters, both big enough to be
///   people, further apart than this.
/// * [`MAX_SPEAKERS`] — no cut with more people than Echo will show.
///
/// When two counts score within [`SILHOUETTE_TIE`] of each other, the one whose
/// band of distances is nearest the old shipped 0.58 wins, and a straight tie
/// after that goes to the larger count: fusing two people is the worse way to be
/// wrong, because the UI lets somebody merge two speakers and never lets them
/// split one.
///
/// # What this does not fix
///
/// It decides *how many* voices there were far more robustly than a distance
/// could. It does not make the fingerprints better: if two people genuinely
/// fingerprint alike, no criterion reading this tree will separate them, and the
/// person's own count ([`cluster_fixed`]) is still the better evidence when they
/// give one.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CountChoice {
    /// Every merge, in the order they happened.
    pub ladder: Vec<Rung>,
    /// One row per number of people the tree could be cut into, ascending. Where
    /// several cut heights fold to the same number of people, the row is the
    /// best-scoring of them.
    pub candidates: Vec<Candidate>,
    /// The count that won.
    pub count: usize,
    /// Height the tree was cut at — the last merge accepted.
    pub cut_at: f32,
    /// The score that won.
    pub silhouette: f32,
    /// The best score among the counts that lost, when there was one.
    pub runner_up: Option<(usize, f32)>,
    /// Speech a cluster had to hold to be a person rather than a fragment, for
    /// this meeting ([`fragment_bar`]).
    pub fragment_bar_ms: i64,
    /// What the largest relative gap in the ladder would have answered — over
    /// **every** cut height, not just the ones that survive into
    /// [`Self::candidates`].
    ///
    /// Printed beside the answer, because on the meeting this criterion was
    /// written for the two disagree and the gap is the one that is wrong.
    pub gap_answer: Option<Candidate>,
}

impl CountChoice {
    /// The winning row, when there was a choice to make at all.
    pub fn chosen(&self) -> Option<&Candidate> {
        self.candidates.iter().find(|c| c.count == self.count)
    }
}

/// Two silhouettes closer than this are a tie, and the tie-break decides.
///
/// Small on purpose: the criterion's claim is that its margins are not
/// hair-thin, so anything inside a hundredth is noise, and the old shipped
/// distance is a better tie-break than noise is.
pub const SILHOUETTE_TIE: f32 = 0.01;

/// How many cut heights to weigh, as a multiple of [`MAX_SPEAKERS`].
///
/// Not one per person: fragments take cluster slots, and on the real meeting the
/// two-person answer is a cut into **seven** clusters, five of which fold away.
/// Three times the ceiling leaves room for a full meeting with a tail of noise in
/// it, and each extra height costs one replay of a recorded hierarchy.
const CUT_HEIGHTS: usize = MAX_SPEAKERS * 3;

/// Group fingerprints into people, deciding for itself how many there were.
///
/// The automatic pass, and what [`crate::diarize::pipeline::refine`] calls when
/// nobody has corrected the count. See [`CountChoice`] for the criterion, what a
/// real meeting's tree looks like, why the answer is not the largest relative
/// gap, and what became of [`DISTANCE_THRESHOLD`].
///
/// Returns the clustering and the whole decision, so a probe can print it and a
/// person can argue with it.
pub fn cluster_auto(items: &[ClusterItem]) -> (Clustering, CountChoice) {
    if items.is_empty() {
        return (
            Clustering {
                labels: Vec::new(),
                cluster_count: 0,
                threshold: 0.0,
            },
            CountChoice::default(),
        );
    }

    let Some(embeddings) = normalised(items) else {
        // Nothing usable to measure with: one speaker rather than a crash, the
        // same answer every other path here gives.
        return (
            Clustering {
                labels: vec![0; items.len()],
                cluster_count: 1,
                threshold: 0.0,
            },
            CountChoice {
                count: 1,
                ..Default::default()
            },
        );
    };

    // The leaves, and the whole hierarchy above them recorded once. Merging all
    // the way down to one cluster costs no more than the old loop did: it stopped
    // a handful of merges short of this, and those last merges are the cheap ones
    // — a pair search over a handful of clusters.
    let leaf_nodes = leaves(items, &embeddings, DISTANCE_THRESHOLD / 2.0);
    let mut working = leaf_nodes.clone();
    let merges = record_merges(&mut working);
    drop(working);

    let bar = fragment_bar(leaf_nodes.iter().map(|l| l.weight_ms.max(0)).sum());
    let ladder = ladder_of(&merges, leaf_nodes.len());
    let (candidates, gap_answer) = weigh_counts(items, &embeddings, &leaf_nodes, &merges, bar);
    let (chosen, runner_up) = pick(&candidates);
    let (count, silhouette, cut_clusters) =
        chosen.map_or((1, 0.0, 1), |c| (c.count, c.silhouette, c.cut_clusters));

    // Replay the recorded merges rather than searching for the pairs again, then
    // hand the fragments away exactly as the candidate that won was scored with.
    let mut nodes = leaf_nodes;
    let cut_at = replay(&mut nodes, &merges, cut_clusters);
    absorb_fragments(&mut nodes, bar);

    (
        finish(nodes, items.len(), cut_at),
        CountChoice {
            ladder,
            candidates,
            count,
            cut_at,
            silhouette,
            runner_up,
            fragment_bar_ms: bar,
            gap_answer,
        },
    )
}

/// Merge the closest pair over and over until one cluster is left, recording
/// every step. Leaves `nodes` fully merged; [`replay`] is what rebuilds a cut.
fn record_merges(nodes: &mut [Node]) -> Vec<Merge> {
    let mut out = Vec::with_capacity(nodes.len().saturating_sub(1));
    while let Some((a, b, distance)) = closest_pair(nodes) {
        out.push(Merge { a, b, distance });
        fold(nodes, a, b);
    }
    out
}

/// Apply a recorded hierarchy's merges to fresh leaves until `keep` clusters are
/// left. Returns the height of the last merge applied, `0.0` when none was.
fn replay(nodes: &mut [Node], merges: &[Merge], keep: usize) -> f32 {
    let mut cut = 0.0f32;
    let mut alive = alive_count(nodes);
    for m in merges {
        if alive <= keep {
            break;
        }
        cut = m.distance;
        fold(nodes, m.a, m.b);
        alive -= 1;
    }
    cut
}

fn ladder_of(merges: &[Merge], leaves: usize) -> Vec<Rung> {
    merges
        .iter()
        .enumerate()
        .map(|(i, m)| Rung {
            clusters: leaves.saturating_sub(i + 1),
            distance: m.distance,
        })
        .collect()
}

/// The first merge that fuses two clusters both big enough to be people, at or
/// beyond [`FUSE_CEILING`].
///
/// Only pairs of real clusters count. A two-second sliver landing 0.9 away from
/// everything is not two people being fused, and letting it fire the ceiling was
/// what made the first version of this criterion answer "one voice" on the very
/// meeting it was written for: every cut was refused, so the fallback answered.
fn ceiling_breach(merges: &[Merge], leaves: &[Node], bar: i64) -> Option<usize> {
    let mut mass: Vec<i64> = leaves.iter().map(|l| l.weight_ms.max(0)).collect();
    for (i, m) in merges.iter().enumerate() {
        if m.distance >= FUSE_CEILING && mass[m.a] >= bar && mass[m.b] >= bar {
            return Some(i);
        }
        mass[m.a] += mass[m.b];
    }
    None
}

/// Score every number of people the tree could be cut into, and — for the
/// record — what the largest relative gap in it would have said.
///
/// One row per resulting count rather than per cut height: several heights fold
/// to the same number of people once fragments are handed away, and the row kept
/// is the best-scoring of them.
fn weigh_counts(
    items: &[ClusterItem],
    embeddings: &[Vec<f32>],
    leaves: &[Node],
    merges: &[Merge],
    bar: i64,
) -> (Vec<Candidate>, Option<Candidate>) {
    let n = leaves.len();
    let weights: Vec<i64> = items.iter().map(|i| i.weight_ms.max(0)).collect();
    // How many clusters the ceiling insists on: enough that the merge which
    // would fuse two people is not one of the merges accepted.
    let forced = ceiling_breach(merges, leaves, bar).map_or(0, |at| n - at);

    let mut out: Vec<Candidate> = Vec::new();
    let mut widest: Option<Candidate> = None;
    for cut_clusters in 1..=CUT_HEIGHTS.min(n) {
        let want = n - cut_clusters;
        let accepted = want.checked_sub(1).map_or(0.0, |i| merges[i].distance);
        let refused = merges.get(want).map_or(0.0, |m| m.distance);

        let mut nodes = leaves.to_vec();
        replay(&mut nodes, merges, cut_clusters);
        absorb_fragments(&mut nodes, bar);
        let (labels, count) = labels_of(&nodes, items.len());
        let mut mass_ms: Vec<i64> = nodes
            .iter()
            .filter(|node| node.alive)
            .map(|node| node.weight_ms)
            .collect();
        mass_ms.sort_unstable_by(|a, b| b.cmp(a));

        let refusal = if count > MAX_SPEAKERS {
            Some(Refusal::TooMany)
        } else if cut_clusters > 1 && refused < SPLIT_FLOOR {
            Some(Refusal::OneVoice)
        } else if cut_clusters < forced && count < MAX_SPEAKERS {
            // The `count < MAX_SPEAKERS` half is what stops the ceiling refusing
            // every cut of a meeting where *everybody* is further apart than it:
            // a room of a dozen plainly different voices has a breaching merge at
            // the very bottom of its tree, and refusing every cut above it would
            // leave nothing to choose from. Echo shows twelve people at most, so
            // twelve is where the insisting stops.
            Some(Refusal::Fused)
        } else {
            None
        };

        let candidate = Candidate {
            count,
            cut_clusters,
            folded: cut_clusters.saturating_sub(count),
            accepted,
            refused,
            gap: if accepted > 1e-6 {
                refused / accepted
            } else {
                0.0
            },
            silhouette: silhouette(embeddings, &weights, &labels, count),
            mass_ms,
            refusal,
        };

        if widest.as_ref().is_none_or(|w| candidate.gap > w.gap) {
            widest = Some(candidate.clone());
        }

        // One row per count. A refused row never displaces an allowed one, and
        // among equals the better score stays.
        match out.iter_mut().find(|c| c.count == candidate.count) {
            Some(kept) => {
                let better = match (kept.allowed(), candidate.allowed()) {
                    (false, true) => true,
                    (true, false) => false,
                    _ => candidate.silhouette > kept.silhouette,
                };
                if better {
                    *kept = candidate;
                }
            }
            None => out.push(candidate),
        }
    }
    out.sort_by_key(|c| c.count);
    (out, widest)
}

/// Weighted mean silhouette of one cut, against cluster centroids.
///
/// Centroids rather than all-pairs means: this is the geometry the linkage itself
/// works in, it costs one pass per cut instead of a distance matrix, and it
/// answers the question that matters — is a fingerprint nearer the voice it was
/// given than any other voice.
///
/// A fingerprint is left out of its own cluster's centroid, or a cluster of one
/// would score a perfect 1.0 for being exactly where it put itself. A cluster of
/// one scores 0 instead: there is no voice there to be near.
fn silhouette(embeddings: &[Vec<f32>], weights: &[i64], labels: &[usize], count: usize) -> f32 {
    if count < 2 || embeddings.is_empty() {
        return 0.0;
    }
    let dim = embeddings[0].len();
    let mut sums = vec![vec![0.0f32; dim]; count];
    let mut mass = vec![0.0f64; count];
    for (i, v) in embeddings.iter().enumerate() {
        let c = labels[i];
        let w = weights[i].max(1) as f64;
        for (slot, x) in sums[c].iter_mut().zip(v.iter()) {
            *slot += x * w as f32;
        }
        mass[c] += w;
    }
    let centroids: Vec<Vec<f32>> = sums
        .iter()
        .map(|s| {
            let mut c = s.clone();
            l2_normalize(&mut c);
            c
        })
        .collect();

    let mut total = 0.0f64;
    let mut weighed = 0.0f64;
    for (i, v) in embeddings.iter().enumerate() {
        let own = labels[i];
        let w = weights[i].max(1) as f64;
        weighed += w;

        let nearest_other = (0..count)
            .filter(|&c| c != own && mass[c] > 0.0)
            .map(|c| cosine_distance(v, &centroids[c]))
            .fold(f32::INFINITY, f32::min);
        if mass[own] - w <= 0.0 || !nearest_other.is_finite() {
            continue; // a cluster of one, or nothing to compare with: scores 0
        }

        let mut rest: Vec<f32> = sums[own]
            .iter()
            .zip(v.iter())
            .map(|(s, x)| s - x * w as f32)
            .collect();
        l2_normalize(&mut rest);
        let own_distance = cosine_distance(v, &rest);

        let scale = own_distance.max(nearest_other);
        if scale > 1e-9 {
            total += w * f64::from((nearest_other - own_distance) / scale);
        }
    }
    if weighed <= 0.0 {
        0.0
    } else {
        (total / weighed) as f32
    }
}

/// The best-scoring allowed count, and the best score among the rest.
///
/// Ties inside [`SILHOUETTE_TIE`] go to the cut nearest the old shipped distance,
/// then to the larger count. See [`CountChoice`].
///
/// When the prior refuses every cut — which takes a tree where every cut either
/// needs more people than Echo shows or splits a pair too close to be two — the
/// best-explained cut wins anyway. A bound is there to keep a criterion honest,
/// not to leave it with nothing to say.
fn pick(candidates: &[Candidate]) -> (Option<&Candidate>, Option<(usize, f32)>) {
    let mut ranked: Vec<&Candidate> = candidates.iter().filter(|c| c.allowed()).collect();
    if ranked.is_empty() {
        ranked = candidates.iter().collect();
    }
    ranked.sort_by(|x, y| {
        if (x.silhouette - y.silhouette).abs() < SILHOUETTE_TIE {
            return x
                .distance_from_shipped()
                .total_cmp(&y.distance_from_shipped())
                .then(y.count.cmp(&x.count));
        }
        y.silhouette.total_cmp(&x.silhouette)
    });
    let winner = ranked.first().copied();
    let runner_up = candidates
        .iter()
        .filter(|c| Some(c.count) != winner.map(|w| w.count))
        .max_by(|x, y| x.silhouette.total_cmp(&y.silhouette))
        .map(|c| (c.count, c.silhouette));
    (winner, runner_up)
}

fn alive_count(nodes: &[Node]) -> usize {
    nodes.iter().filter(|n| n.alive).count()
}

/// The two closest live clusters and how far apart they are. `None` once fewer
/// than two are left.
fn closest_pair(nodes: &[Node]) -> Option<(usize, usize, f32)> {
    let alive: Vec<usize> = (0..nodes.len()).filter(|&i| nodes[i].alive).collect();
    if alive.len() < 2 {
        return None;
    }
    let mut best: Option<(usize, usize, f32)> = None;
    for (a_pos, &a) in alive.iter().enumerate() {
        for &b in &alive[a_pos + 1..] {
            let d = cosine_distance(&nodes[a].centroid, &nodes[b].centroid);
            match best {
                Some((_, _, bd)) if d >= bd => {}
                _ => best = Some((a, b, d)),
            }
        }
    }
    best
}

/// Fold `b` into `a`: `a` keeps every member of both and moves to their shared
/// weighted centroid, `b` stops existing.
fn fold(nodes: &mut [Node], a: usize, b: usize) {
    let (sum_b, weight_b, ms_b, members_b) = {
        let nb = &mut nodes[b];
        nb.alive = false;
        (
            std::mem::take(&mut nb.sum),
            nb.weight,
            nb.weight_ms,
            std::mem::take(&mut nb.members),
        )
    };
    let na = &mut nodes[a];
    for (slot, x) in na.sum.iter_mut().zip(sum_b.iter()) {
        *slot += x;
    }
    na.weight += weight_b;
    na.weight_ms += ms_b;
    na.members.extend(members_b);
    na.recentre();
}

fn merge_loop(nodes: &mut [Node], threshold: f32, max_clusters: usize) {
    while let Some((a, b, d)) = closest_pair(nodes) {
        if d >= threshold && alive_count(nodes) <= max_clusters {
            return;
        }
        fold(nodes, a, b);
    }
}

/// Merge the closest pair over and over until exactly `k` clusters are left,
/// however far apart they end up being.
///
/// Returns the distance of the last merge — the height the dendrogram was cut
/// at, which is the useful diagnostic here in place of a threshold. `0.0` when
/// nothing had to be merged at all.
fn merge_to_k(nodes: &mut [Node], k: usize) -> f32 {
    let mut cut = 0.0f32;
    while alive_count(nodes) > k {
        let Some((a, b, d)) = closest_pair(nodes) else {
            break;
        };
        cut = d;
        fold(nodes, a, b);
    }
    cut
}

/// Hand every cluster holding less than `min_ms` of speech to the cluster it
/// most resembles.
///
/// `min_ms` is [`MIN_CLUSTER_MS`] for a cut at a distance, and [`fragment_bar`]
/// for the automatic count — a fragment is a fragment in proportion to the
/// meeting it is in, and the real failure had ten of them.
fn absorb_fragments(nodes: &mut [Node], min_ms: i64) {
    loop {
        let alive: Vec<usize> = (0..nodes.len()).filter(|&i| nodes[i].alive).collect();
        if alive.len() < 2 {
            return;
        }
        // Smallest cluster first, so a chain of fragments resolves outward.
        let Some(&small) = alive
            .iter()
            .filter(|&&i| nodes[i].weight_ms < min_ms)
            .min_by_key(|&&i| nodes[i].weight_ms)
        else {
            return;
        };
        let Some(&host) = alive.iter().filter(|&&i| i != small).min_by(|&&x, &&y| {
            cosine_distance(&nodes[small].centroid, &nodes[x].centroid)
                .total_cmp(&cosine_distance(&nodes[small].centroid, &nodes[y].centroid))
        }) else {
            return;
        };

        let (sum, weight, ms, members) = {
            let n = &mut nodes[small];
            n.alive = false;
            (
                std::mem::take(&mut n.sum),
                n.weight,
                n.weight_ms,
                std::mem::take(&mut n.members),
            )
        };
        let h = &mut nodes[host];
        for (slot, x) in h.sum.iter_mut().zip(sum.iter()) {
            *slot += x;
        }
        h.weight += weight;
        h.weight_ms += ms;
        h.members.extend(members);
        h.recentre();
    }
}

/// One label per item, numbered in the order the surviving clusters come, and
/// how many of them there are.
fn labels_of(nodes: &[Node], item_count: usize) -> (Vec<usize>, usize) {
    let mut labels = vec![0usize; item_count];
    let mut next = 0usize;
    for node in nodes.iter().filter(|n| n.alive) {
        for &m in &node.members {
            labels[m] = next;
        }
        next += 1;
    }
    (labels, next)
}

fn finish(nodes: Vec<Node>, item_count: usize, threshold: f32) -> Clustering {
    let (labels, count) = labels_of(&nodes, item_count);
    Clustering {
        labels,
        cluster_count: count.max(1),
        threshold,
    }
}

/// Memo for [`align_permutation`]: `(row, bitmask of used columns)` to the best
/// total shared milliseconds and the choices that got there.
type AssignmentMemo = HashMap<(usize, u32), (i64, Vec<Option<usize>>)>;

/// Line up this window's local speakers with the previous window's, using the
/// milliseconds each pair shares in the region the windows have in common.
///
/// `overlap[current][previous]` is that shared time. Returns, for every current
/// local speaker, the previous local speaker it continues, or `None` when it is
/// somebody new. The mapping is injective: two voices in this window can never
/// both be the same voice from the last one.
///
/// Exact rather than greedy, because greedy gets the common two-speaker seam
/// wrong whenever one voice dominates the overlap.
pub fn align_permutation(overlap: &[Vec<i64>]) -> Vec<Option<usize>> {
    let rows = overlap.len();
    if rows == 0 {
        return Vec::new();
    }
    let cols = overlap[0].len();
    if cols == 0 || cols > 16 || rows > 16 {
        return vec![None; rows];
    }

    // best(row, used columns) -> (total shared ms, choices from here on)
    let mut memo: AssignmentMemo = HashMap::new();

    fn best(
        row: usize,
        used: u32,
        rows: usize,
        cols: usize,
        overlap: &[Vec<i64>],
        memo: &mut AssignmentMemo,
    ) -> (i64, Vec<Option<usize>>) {
        if row == rows {
            return (0, Vec::new());
        }
        if let Some(hit) = memo.get(&(row, used)) {
            return hit.clone();
        }
        // Option 1: this speaker is new.
        let (mut top, tail) = best(row + 1, used, rows, cols, overlap, memo);
        let mut choice: Option<usize> = None;
        let mut best_tail = tail;

        for c in 0..cols {
            if used & (1 << c) != 0 || overlap[row][c] <= 0 {
                continue;
            }
            let (sub, tail) = best(row + 1, used | (1 << c), rows, cols, overlap, memo);
            let total = sub + overlap[row][c];
            if total > top {
                top = total;
                choice = Some(c);
                best_tail = tail;
            }
        }

        let mut out = Vec::with_capacity(rows - row);
        out.push(choice);
        out.extend(best_tail);
        memo.insert((row, used), (top, out.clone()));
        (top, out)
    }

    best(0, 0, rows, cols, overlap, &mut memo).1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic fingerprint: a unit spike in one dimension plus a little
    /// jitter, so members of one "voice" are close but not identical.
    fn voice(dim: usize, id: usize, jitter: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[id % dim] = 1.0;
        v[(id + 1 + jitter) % dim] += 0.08 * (jitter as f32 + 1.0);
        l2_normalize(&mut v);
        v
    }

    fn item(embedding: Vec<f32>, weight_ms: i64) -> ClusterItem {
        ClusterItem {
            embedding,
            weight_ms,
        }
    }

    #[test]
    fn normalising_makes_unit_vectors_and_leaves_zero_alone() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
        let mut z = vec![0.0f32; 4];
        l2_normalize(&mut z);
        assert_eq!(z, vec![0.0; 4]);
    }

    #[test]
    fn cosine_distance_is_zero_for_the_same_voice_and_one_for_unrelated_ones() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        assert!(cosine_distance(&a, &a) < 1e-6);
        assert!((cosine_distance(&a, &b) - 1.0).abs() < 1e-6);
        let opposite = vec![-1.0f32, 0.0, 0.0];
        assert!((cosine_distance(&a, &opposite) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn three_tight_groups_come_out_as_three_speakers() {
        let dim = 32;
        let mut items = Vec::new();
        for id in [3usize, 11, 20] {
            for j in 0..4 {
                items.push(item(voice(dim, id, j), 4_000));
            }
        }
        let c = cluster(&items, DISTANCE_THRESHOLD);
        assert_eq!(c.cluster_count, 3, "labels {:?}", c.labels);
        // Every group of four shares a label, and the three labels differ.
        for g in 0..3 {
            let group = &c.labels[g * 4..(g + 1) * 4];
            assert!(
                group.iter().all(|l| *l == group[0]),
                "group {g} split: {group:?}"
            );
        }
        assert_eq!(
            c.labels
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            3
        );
    }

    #[test]
    fn one_voice_recorded_many_times_stays_one_speaker() {
        let dim = 64;
        let items: Vec<_> = (0..12).map(|j| item(voice(dim, 5, j % 3), 5_000)).collect();
        let c = cluster(&items, DISTANCE_THRESHOLD);
        assert_eq!(c.cluster_count, 1);
        assert!(c.labels.iter().all(|l| *l == 0));
    }

    #[test]
    fn a_single_fingerprint_is_one_speaker() {
        let c = cluster(&[item(vec![1.0, 0.0, 0.0], 9_000)], DISTANCE_THRESHOLD);
        assert_eq!(c.cluster_count, 1);
        assert_eq!(c.labels, vec![0]);
    }

    #[test]
    fn no_fingerprints_means_no_speakers() {
        let c = cluster(&[], DISTANCE_THRESHOLD);
        assert_eq!(c.cluster_count, 0);
        assert!(c.labels.is_empty());
    }

    #[test]
    fn a_sliver_of_speech_is_folded_into_the_voice_it_resembles() {
        let dim = 32;
        let mut items = vec![
            item(voice(dim, 2, 0), 20_000),
            item(voice(dim, 2, 1), 20_000),
            item(voice(dim, 25, 0), 20_000),
        ];
        // 300 ms of a fingerprint too far from both voices to be merged on
        // distance alone, but leaning towards the first one. On its own it would
        // become a third speaker the person has to merge away.
        let mut sliver = vec![0.0f32; dim];
        sliver[2] = 0.2;
        sliver[15] = 0.7;
        sliver[16] = 0.7;
        l2_normalize(&mut sliver);
        assert!(
            cosine_distance(&sliver, &items[0].embedding) > DISTANCE_THRESHOLD,
            "the fixture has to be past the threshold or the merge loop eats it"
        );
        items.push(item(sliver, 300));

        let c = cluster(&items, DISTANCE_THRESHOLD);
        assert_eq!(c.cluster_count, 2, "labels {:?}", c.labels);
        assert_eq!(
            c.labels[3], c.labels[0],
            "the sliver joined the wrong voice"
        );
    }

    #[test]
    fn the_speaker_ceiling_is_enforced_even_with_many_distinct_voices() {
        let dim = 128;
        let items: Vec<_> = (0..40)
            .map(|id| item(voice(dim, id * 3, 0), 10_000))
            .collect();
        let c = cluster(&items, DISTANCE_THRESHOLD);
        assert!(
            c.cluster_count <= MAX_SPEAKERS,
            "{} clusters",
            c.cluster_count
        );
        assert!(c.cluster_count > 1);
    }

    #[test]
    fn the_pooling_path_agrees_with_the_exact_path() {
        // Enough items to trip MAX_EXACT_ITEMS, drawn from three voices.
        let dim = 48;
        let mut items = Vec::new();
        for i in 0..(MAX_EXACT_ITEMS + 60) {
            items.push(item(voice(dim, [4usize, 17, 33][i % 3], i % 3), 4_000));
        }
        let pooled = cluster(&items, DISTANCE_THRESHOLD);
        assert_eq!(pooled.cluster_count, 3, "labels differ");
        // Same voice, same label, all the way through.
        for i in 0..items.len() {
            assert_eq!(
                pooled.labels[i],
                pooled.labels[i % 3],
                "item {i} left its voice"
            );
        }
    }

    #[test]
    fn every_label_is_inside_the_cluster_count() {
        let dim = 16;
        let items: Vec<_> = (0..20).map(|i| item(voice(dim, i, i % 4), 6_000)).collect();
        let c = cluster(&items, DISTANCE_THRESHOLD);
        assert_eq!(c.labels.len(), items.len());
        assert!(c.labels.iter().all(|l| *l < c.cluster_count));
    }

    #[test]
    fn a_lower_threshold_splits_and_a_higher_one_fuses() {
        let dim = 32;
        let items: Vec<_> = (0..6)
            .map(|i| item(voice(dim, if i < 3 { 2 } else { 4 }, i), 5_000))
            .collect();
        let tight = cluster(&items, 0.05);
        let loose = cluster(&items, 1.9);
        assert!(tight.cluster_count >= loose.cluster_count);
        assert_eq!(loose.cluster_count, 1);
    }

    // -----------------------------------------------------------------------
    // Cutting at a count the person gave us
    // -----------------------------------------------------------------------

    /// Four voices, four fingerprints each. The automatic pass finds four; the
    /// person can have any number they ask for out of the same hierarchy.
    fn four_voices() -> Vec<ClusterItem> {
        let dim = 32;
        let mut items = Vec::new();
        for id in [3usize, 11, 20, 28] {
            for j in 0..4 {
                items.push(item(voice(dim, id, j), 4_000));
            }
        }
        items
    }

    #[test]
    fn left_alone_the_same_fingerprints_come_out_as_four_people() {
        // The baseline the two cuts below are a correction of.
        assert_eq!(cluster(&four_voices(), DISTANCE_THRESHOLD).cluster_count, 4);
    }

    #[test]
    fn asked_for_two_it_gives_exactly_two() {
        let items = four_voices();
        let c = cluster_fixed(&items, 2);
        assert_eq!(c.cluster_count, 2, "labels {:?}", c.labels);
        assert_eq!(c.labels.len(), items.len());
        assert!(c.labels.iter().all(|l| *l < 2));
        // Fusing happens between whole voices, never inside one: each group of
        // four still shares a label.
        for g in 0..4 {
            let group = &c.labels[g * 4..(g + 1) * 4];
            assert!(
                group.iter().all(|l| *l == group[0]),
                "voice {g} was split to make the count: {group:?}"
            );
        }
    }

    #[test]
    fn asked_for_six_it_gives_exactly_six() {
        let items = four_voices();
        let c = cluster_fixed(&items, 6);
        assert_eq!(c.cluster_count, 6, "labels {:?}", c.labels);
        assert_eq!(c.labels.len(), items.len());
        assert!(c.labels.iter().all(|l| *l < 6));
    }

    /// The whole point of this path: the calibrated distance does not get a vote.
    #[test]
    fn the_calibrated_threshold_does_not_override_the_count() {
        let items = four_voices();
        for k in 1..=8 {
            let c = cluster_fixed(&items, k);
            assert_eq!(c.cluster_count, k, "asked for {k}, got {}", c.cluster_count);
        }
    }

    /// A sliver of speech is a person too, when somebody has said so. The
    /// automatic path folds it away; this one must not.
    #[test]
    fn a_short_speaker_is_kept_when_the_count_asks_for_them() {
        let dim = 32;
        let mut items = vec![
            item(voice(dim, 2, 0), 20_000),
            item(voice(dim, 2, 1), 20_000),
            item(voice(dim, 25, 0), 20_000),
        ];
        let mut sliver = vec![0.0f32; dim];
        sliver[2] = 0.2;
        sliver[15] = 0.7;
        sliver[16] = 0.7;
        l2_normalize(&mut sliver);
        items.push(item(sliver, 300));

        // Left to itself the sliver is absorbed and two people come out.
        assert_eq!(cluster(&items, DISTANCE_THRESHOLD).cluster_count, 2);
        // Asked for three, the sliver stays a person of its own.
        let c = cluster_fixed(&items, 3);
        assert_eq!(c.cluster_count, 3, "labels {:?}", c.labels);
        assert_ne!(
            c.labels[3], c.labels[0],
            "the short speaker was folded away anyway"
        );
    }

    #[test]
    fn fewer_fingerprints_than_asked_for_gives_back_what_exists() {
        let dim = 16;
        let items: Vec<_> = (0..3).map(|i| item(voice(dim, i * 5, 0), 5_000)).collect();
        let c = cluster_fixed(&items, 6);
        assert_eq!(c.cluster_count, 3, "there were only three to hand out");
        assert_eq!(c.labels, vec![0, 1, 2]);
    }

    #[test]
    fn a_count_outside_what_echo_can_do_is_pulled_into_range() {
        let items = four_voices();
        assert_eq!(cluster_fixed(&items, 0).cluster_count, 1);
        assert_eq!(
            cluster_fixed(&items, 999).cluster_count,
            items.len().min(MAX_SPEAKERS)
        );
    }

    #[test]
    fn no_fingerprints_stays_no_speakers_whatever_the_count_says() {
        let c = cluster_fixed(&[], 4);
        assert_eq!(c.cluster_count, 0);
        assert!(c.labels.is_empty());
    }

    #[test]
    fn the_recorded_number_is_the_height_the_hierarchy_was_cut_at() {
        let items = four_voices();
        let tight = cluster_fixed(&items, 4);
        let loose = cluster_fixed(&items, 2);
        // Cutting lower down the tree means the last merge was a closer pair.
        assert!(
            loose.threshold > tight.threshold,
            "cut heights {} then {}",
            tight.threshold,
            loose.threshold
        );
        // Nothing had to be merged, so there is no cut to report.
        assert_eq!(cluster_fixed(&items[..2], 4).threshold, 0.0);
    }

    #[test]
    fn the_pooling_path_still_hits_the_count_exactly() {
        // Enough items to trip MAX_EXACT_ITEMS, drawn from three voices.
        let dim = 48;
        let items: Vec<_> = (0..(MAX_EXACT_ITEMS + 60))
            .map(|i| item(voice(dim, [4usize, 17, 33][i % 3], i % 3), 4_000))
            .collect();
        let c = cluster_fixed(&items, 2);
        assert_eq!(c.cluster_count, 2);
        assert_eq!(c.labels.len(), items.len());
    }

    // -----------------------------------------------------------------------
    // Deciding the count from the shape of the tree
    // -----------------------------------------------------------------------

    /// Fingerprints for two voices whose centroids sit about `apart` in cosine
    /// distance, each with within-voice jitter in a dimension of its own so
    /// members of one voice are near each other without being identical.
    ///
    /// `apart` is what every test below is about. The field failure was a pair
    /// 0.606 apart being cut at 0.58.
    fn pair_apart(apart: f32, jitter: f32, per_voice: usize, weight_ms: i64) -> Vec<ClusterItem> {
        const DIM: usize = 64;
        let cos = 1.0 - apart;
        let sin = (1.0 - cos * cos).max(0.0).sqrt();
        let mut items = Vec::new();
        for (v, (x, y)) in [(1.0f32, 0.0f32), (cos, sin)].into_iter().enumerate() {
            for j in 0..per_voice {
                let mut e = vec![0.0f32; DIM];
                e[0] = x;
                e[1] = y;
                e[2 + (v * per_voice + j) % (DIM - 2)] = jitter;
                l2_normalize(&mut e);
                items.push(item(e, weight_ms));
            }
        }
        items
    }

    /// **The field failure, as a unit test.** Two people closer together than the
    /// distance that shipped: the old cut makes them one person, and reading the
    /// count off the tree makes them two.
    #[test]
    fn two_people_nearer_than_the_old_cut_are_still_two_people() {
        let items = pair_apart(0.55, 0.5, 6, 20_000);

        // What shipped, and what went wrong with it: the pair is inside the
        // calibrated distance, so it fuses.
        assert_eq!(
            cluster(&items, DISTANCE_THRESHOLD).cluster_count,
            1,
            "the fixture has to reproduce the failure or it is testing nothing"
        );

        let (clustering, choice) = cluster_auto(&items);
        assert_eq!(clustering.cluster_count, 2, "{:?}", choice.candidates);
        // And not by a hair: the two-voice cut explains the fingerprints far
        // better than one voice does.
        assert!(choice.silhouette > 0.3, "silhouette {}", choice.silhouette);
        assert_eq!(choice.count, 2);
    }

    /// The same pair moved further apart than the old cut. Both answers agree
    /// here, which is the point: nothing regressed for the meetings that worked.
    #[test]
    fn two_people_further_apart_than_the_old_cut_are_two_people_either_way() {
        let items = pair_apart(0.65, 0.5, 6, 20_000);
        assert_eq!(cluster(&items, DISTANCE_THRESHOLD).cluster_count, 2);
        assert_eq!(cluster_auto(&items).0.cluster_count, 2);
    }

    /// One voice that disagrees with itself is not two people. The floor is the
    /// only thing that can say so — a silhouette will happily score any split of
    /// one blob — and this is the failure it exists to stop.
    #[test]
    fn a_voice_that_disagrees_with_itself_is_not_two_people() {
        let items = pair_apart(0.15, 0.4, 6, 20_000);
        let (clustering, choice) = cluster_auto(&items);
        assert_eq!(clustering.cluster_count, 1, "{:?}", choice.candidates);
        assert!(
            choice
                .candidates
                .iter()
                .any(|c| c.count > 1 && c.refusal == Some(Refusal::OneVoice)),
            "the floor should be the reason, not luck: {:?}",
            choice.candidates
        );
    }

    /// The shape a real meeting actually has: two people, and a tail of slivers
    /// that sit further from everything than the two people sit from each other.
    /// The two-person answer is a cut into *six* clusters, four of which fold.
    #[test]
    fn a_tail_of_slivers_does_not_take_the_cluster_slots_the_people_need() {
        let mut items = pair_apart(0.60, 0.4, 8, 20_000);
        for k in 0..4 {
            items.push(item(voice(64, 10 + k * 7, 0), 500));
        }

        let (clustering, choice) = cluster_auto(&items);
        assert_eq!(clustering.cluster_count, 2, "{:?}", choice.candidates);
        let chosen = choice.chosen().expect("a chosen row");
        assert!(
            chosen.cut_clusters > 2 && chosen.folded > 0,
            "the people are only reachable above the slivers: {chosen:?}"
        );
        // Neither sliver survives as somebody, and both people do.
        assert_eq!(chosen.mass_ms.len(), 2);
        assert!(chosen.mass_ms.iter().all(|ms| *ms >= choice.fragment_bar_ms));
    }

    /// Two voices nothing could confuse. The ceiling refuses to make them one
    /// person whatever else the tree looks like.
    #[test]
    fn two_plainly_different_voices_are_never_made_one_person() {
        let items = pair_apart(1.0, 0.3, 6, 20_000);
        let (clustering, choice) = cluster_auto(&items);
        assert_eq!(clustering.cluster_count, 2);
        let one = choice
            .candidates
            .iter()
            .find(|c| c.count == 1)
            .expect("a one-voice row");
        assert_eq!(one.refusal, Some(Refusal::Fused));
    }

    /// A room of a dozen distinct voices has a ceiling-breaking merge at the very
    /// bottom of its tree. Refusing every cut above it would leave nothing to
    /// choose from, so the ceiling stops insisting at [`MAX_SPEAKERS`].
    #[test]
    fn a_room_full_of_distinct_voices_comes_back_capped_not_collapsed() {
        let items: Vec<_> = (0..40)
            .map(|id| item(voice(128, id * 3, 0), 10_000))
            .collect();
        let (clustering, choice) = cluster_auto(&items);
        assert!(
            clustering.cluster_count > 1 && clustering.cluster_count <= MAX_SPEAKERS,
            "{} clusters, {:?}",
            clustering.cluster_count,
            choice.candidates
        );
    }

    #[test]
    fn nothing_to_cluster_stays_nothing_and_one_fingerprint_is_one_person() {
        let (empty, choice) = cluster_auto(&[]);
        assert_eq!(empty.cluster_count, 0);
        assert!(empty.labels.is_empty());
        assert!(choice.candidates.is_empty() && choice.ladder.is_empty());

        let (single, choice) = cluster_auto(&[item(vec![1.0, 0.0, 0.0], 9_000)]);
        assert_eq!(single.cluster_count, 1);
        assert_eq!(choice.count, 1);
    }

    #[test]
    fn fingerprints_that_cannot_be_compared_are_one_person_rather_than_a_crash() {
        let items = vec![item(vec![1.0, 0.0], 5_000), item(vec![1.0, 0.0, 0.0], 5_000)];
        let (clustering, choice) = cluster_auto(&items);
        assert_eq!(clustering.cluster_count, 1);
        assert_eq!(choice.count, 1);
    }

    #[test]
    fn the_pooling_path_reads_the_same_count_out_of_the_tree() {
        // Enough fingerprints to trip MAX_EXACT_ITEMS, drawn from three voices.
        let dim = 48;
        let items: Vec<_> = (0..(MAX_EXACT_ITEMS + 60))
            .map(|i| item(voice(dim, [4usize, 17, 33][i % 3], i % 3), 4_000))
            .collect();
        let (clustering, choice) = cluster_auto(&items);
        assert_eq!(clustering.cluster_count, 3, "{:?}", choice.candidates);
        for i in 0..items.len() {
            assert_eq!(clustering.labels[i], clustering.labels[i % 3]);
        }
    }

    #[test]
    fn the_evidence_is_complete_enough_to_argue_with() {
        let items = pair_apart(0.60, 0.4, 6, 20_000);
        let (_, choice) = cluster_auto(&items);
        // One rung per merge, counting down to one cluster.
        assert_eq!(choice.ladder.len(), items.len() - 1);
        assert_eq!(choice.ladder.first().map(|r| r.clusters), Some(11));
        assert_eq!(choice.ladder.last().map(|r| r.clusters), Some(1));
        // Distances are what the merges happened at, so the ladder climbs.
        assert!(choice.ladder.first().unwrap().distance <= choice.ladder.last().unwrap().distance);
        // One row per count, ascending, and the chosen one is in it.
        assert!(choice.candidates.windows(2).all(|w| w[0].count < w[1].count));
        assert!(choice.chosen().is_some());
        assert!(choice.gap_answer.is_some());
        assert!(choice.fragment_bar_ms >= MIN_CLUSTER_MS);
    }

    // -- the criterion, on its own ------------------------------------------

    fn row(count: usize, silhouette: f32, gap: f32, accepted: f32, refused: f32) -> Candidate {
        Candidate {
            count,
            cut_clusters: count,
            folded: 0,
            accepted,
            refused,
            gap,
            silhouette,
            mass_ms: vec![10_000; count],
            refusal: None,
        }
    }

    /// The whole reason the criterion is a silhouette and not a gap.
    #[test]
    fn the_best_explained_count_wins_even_when_another_has_the_bigger_gap() {
        let rows = vec![
            row(1, 0.0, 0.0, 0.81, 0.0),
            row(2, 0.705, 1.076, 0.5624, 0.6052),
            row(5, 0.532, 1.190, 0.6051, 0.7200),
        ];
        let (winner, runner_up) = pick(&rows);
        assert_eq!(winner.map(|c| c.count), Some(2));
        assert_eq!(runner_up, Some((5, 0.532)));
    }

    #[test]
    fn a_count_the_prior_refused_never_wins_however_well_it_scores() {
        let mut rows = vec![row(1, 0.0, 0.0, 0.81, 0.0), row(2, 0.9, 1.1, 0.4, 0.45)];
        rows[1].refusal = Some(Refusal::OneVoice);
        assert_eq!(pick(&rows).0.map(|c| c.count), Some(1));
    }

    /// A tie goes to the cut the old shipped distance would have made, which is
    /// the one thing that number is still good for.
    #[test]
    fn a_tie_goes_to_the_cut_nearest_the_distance_that_used_to_ship() {
        // Both score the same. The first is the answer for distances around
        // 0.30, the second for distances around 0.58.
        let rows = vec![row(3, 0.50, 1.1, 0.20, 0.30), row(2, 0.505, 1.1, 0.56, 0.60)];
        assert_eq!(pick(&rows).0.map(|c| c.count), Some(2));
    }

    /// Fusing is the worse way to be wrong, so a dead heat splits.
    #[test]
    fn a_dead_heat_leans_towards_more_people_rather_than_fewer() {
        let a = row(2, 0.5, 1.1, 0.56, 0.60);
        let b = row(3, 0.5, 1.1, 0.56, 0.60);
        assert_eq!(pick(&[a, b]).0.map(|c| c.count), Some(3));
    }

    #[test]
    fn a_bound_that_refuses_everything_still_leaves_an_answer() {
        let mut rows = vec![row(1, 0.0, 0.0, 0.9, 0.0), row(2, 0.4, 1.1, 0.5, 0.55)];
        rows[0].refusal = Some(Refusal::Fused);
        rows[1].refusal = Some(Refusal::TooMany);
        assert_eq!(pick(&rows).0.map(|c| c.count), Some(2));
    }

    /// A cluster is a fragment in proportion to the meeting it is in, and never
    /// below the absolute rule this path has always had.
    #[test]
    fn the_fragment_bar_scales_with_the_meeting_but_only_so_far() {
        // A five-minute meeting keeps the absolute rule.
        assert_eq!(fragment_bar(300_000), MIN_CLUSTER_MS);
        // A twenty-minute one asks for about nineteen seconds.
        assert!((18_800..=18_900).contains(&fragment_bar(1_886_000)));
        // A three-hour one stops asking for more.
        assert_eq!(fragment_bar(10_800_000), MAX_FRAGMENT_MS);
        assert_eq!(fragment_bar(0), MIN_CLUSTER_MS);
    }

    /// The silhouette has to say what it claims to say: a fingerprint plainly in
    /// the right place scores near 1, a cluster of one scores 0.
    #[test]
    fn the_silhouette_scores_a_clean_split_high_and_a_cluster_of_one_at_zero() {
        let embeddings = vec![
            vec![1.0f32, 0.0, 0.0],
            vec![0.99, 0.14, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.14, 0.0, 0.99],
        ];
        let weights = vec![10_000i64; 4];
        let clean = silhouette(&embeddings, &weights, &[0, 0, 1, 1], 2);
        assert!(clean > 0.8, "{clean}");
        // Both voices split down the middle instead: every fingerprint is now as
        // near the other cluster as its own.
        let scrambled = silhouette(&embeddings, &weights, &[0, 1, 0, 1], 2);
        assert!(scrambled < clean, "{scrambled} vs {clean}");
        // A cut that leaves singletons cannot score them.
        assert_eq!(silhouette(&embeddings, &weights, &[0, 1, 2, 3], 4), 0.0);
        // One cluster is not a shape anything can be said about.
        assert_eq!(silhouette(&embeddings, &weights, &[0, 0, 0, 0], 1), 0.0);
    }

    /// The floor and the ceiling are what is left of the calibrated distance, and
    /// both are pinned to what the real meeting measured. Moving either means
    /// re-running `examples/speakers_probe.rs` and moving these numbers with it.
    #[test]
    fn the_bounds_bracket_the_distance_they_replaced_and_the_meeting_that_broke_it() {
        // The real two-person meeting, re-scanned 2026-08-21 with the shipped
        // assets (examples/speakers_probe.rs): the two people fuse at 0.6052 and
        // the cut that keeps them apart refuses that merge having accepted
        // 0.5624.
        const PEOPLE_FUSE_AT: f32 = 0.6052;
        const CUT_REFUSES: f32 = 0.6052;
        const CUT_ACCEPTS: f32 = 0.5624;

        // The floor must not refuse to split that pair…
        const { assert!(SPLIT_FLOOR < CUT_REFUSES) };
        // …with room to spare — this is the margin the old 0.0275 became.
        const { assert!(CUT_REFUSES - SPLIT_FLOOR > 0.05) };
        // …and it must still be high enough to sit above one voice's own spread,
        // which is the only failure it exists to stop.
        const { assert!(SPLIT_FLOOR > 0.4) };

        // The ceiling never fires on a pair as near as these two, and sits above
        // what upstream publishes for this network, so it can only ever fire on a
        // pair no published threshold would have merged either.
        const { assert!(FUSE_CEILING > PEOPLE_FUSE_AT) };
        // pyannote 3.1 publishes 0.7046 for this network.
        const { assert!(FUSE_CEILING > 0.7046) };

        // Both bracket the distance that used to decide everything, which is what
        // makes them a prior around it rather than a replacement for it.
        const { assert!(SPLIT_FLOOR < DISTANCE_THRESHOLD && DISTANCE_THRESHOLD < FUSE_CEILING) };
        const { assert!(CUT_ACCEPTS < CUT_REFUSES) };
    }

    #[test]
    fn alignment_matches_the_pairing_with_the_most_shared_speech() {
        // Current speaker 0 shares most with previous 1, current 1 with previous 0.
        let overlap = vec![vec![100, 900], vec![800, 200]];
        assert_eq!(align_permutation(&overlap), vec![Some(1), Some(0)]);
    }

    #[test]
    fn alignment_prefers_the_best_total_over_the_greedy_choice() {
        // Greedy would give current 0 its 100 and leave current 1 with 10,
        // total 110. The best pairing is 90 + 95 = 185.
        let overlap = vec![vec![100, 90], vec![95, 10]];
        assert_eq!(align_permutation(&overlap), vec![Some(1), Some(0)]);
    }

    #[test]
    fn a_voice_with_nothing_in_common_is_treated_as_new() {
        let overlap = vec![vec![0, 0], vec![500, 0]];
        assert_eq!(align_permutation(&overlap), vec![None, Some(0)]);
    }

    #[test]
    fn alignment_never_gives_one_previous_voice_to_two_current_ones() {
        let overlap = vec![vec![900], vec![800], vec![700]];
        let out = align_permutation(&overlap);
        let taken: Vec<usize> = out.iter().flatten().copied().collect();
        assert_eq!(taken, vec![0]);
        assert_eq!(out.iter().filter(|c| c.is_none()).count(), 2);
    }

    #[test]
    fn alignment_of_nothing_is_nothing() {
        assert!(align_permutation(&[]).is_empty());
        assert_eq!(align_permutation(&[vec![]]), vec![None]);
    }

    /// The measured interval, pinned. Editing [`DISTANCE_THRESHOLD`] to anything
    /// outside it means the measurement was redone — so redo it, and move these
    /// numbers with it.
    #[test]
    fn the_calibrated_threshold_is_inside_both_measured_windows() {
        assert_eq!(DISTANCE_THRESHOLD, 0.58);
        // It has to sit inside the cosine range or nothing would ever merge.
        const { assert!(DISTANCE_THRESHOLD > 0.0 && DISTANCE_THRESHOLD < 2.0) };

        // Where the synthetic fixtures get 2, 3, 5 and 7 known speakers exactly
        // right (examples/voices_fixture.rs, 2026-08-20) — the part two runs
        // agreed on, since text-to-speech does not render bit-identically twice.
        const SYNTHETIC: (f32, f32) = (0.2375, 0.625);
        // Where a real 21-minute two-person meeting comes out as two voices with
        // no fragments (examples/speakers_probe.rs --sweep, 2026-08-20).
        const REAL: (f32, f32) = (0.5625, 0.6050);
        for (low, high) in [SYNTHETIC, REAL] {
            assert!(
                DISTANCE_THRESHOLD >= low && DISTANCE_THRESHOLD <= high,
                "{DISTANCE_THRESHOLD} is outside the measured window {low}..{high}"
            );
        }
        // And it is not sitting on either edge of the tighter one: a threshold
        // that only just works is a threshold that stops working on the next
        // meeting.
        assert!(DISTANCE_THRESHOLD - REAL.0 > 0.01);
        assert!(REAL.1 - DISTANCE_THRESHOLD > 0.01);

        // The number upstream publishes for this network, kept here because it is
        // the thing a reader will reach for. It is outside the interval — see the
        // doc comment for why the two pipelines do not share a threshold.
        const PYANNOTE_3_1: f32 = 0.7046;
        assert!(PYANNOTE_3_1 > REAL.1);
    }

    /// The threshold and the fingerprint network are one decision, spread across
    /// two files. This test is what stops someone swapping the asset — a
    /// one-line change in the catalog — and quietly invalidating the number that
    /// decides how many people Echo thinks were in the room.
    ///
    /// It names exactly one asset, not a list of plausible ones. A list is how
    /// the old version of this test let CAM++ in beside the network the number
    /// actually came from, and the count went unvalidated for a release.
    ///
    /// If this fails, that is the point: run
    /// `cargo run --release --example voices_fixture` against the new asset, put
    /// the measured threshold here, and update [`DISTANCE_THRESHOLD`]'s doc
    /// comment with what the sweep said.
    #[test]
    fn the_threshold_matches_the_fingerprint_asset_it_was_tuned_for() {
        /// The one export 0.7153 was measured on: WeSpeaker ResNet34-LM,
        /// 256 dimensions, DIHARD-tuned upstream and swept over the
        /// text-to-speech fixtures here.
        const CALIBRATED_AGAINST: &str = "speaker-embedder-wespeaker-resnet34-lm";
        assert_eq!(
            crate::asr::catalog::ids::EMBEDDER,
            CALIBRATED_AGAINST,
            "the catalog ships a fingerprint network this threshold has never \
             been measured against — re-run examples/voices_fixture before \
             changing this"
        );
        // And the network Echo is walking away from must not quietly come back
        // as the wanted one.
        assert_ne!(
            crate::asr::catalog::ids::EMBEDDER,
            crate::asr::catalog::ids::EMBEDDER_CAMPLUS
        );
    }
}
