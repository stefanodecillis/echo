//! Grouping voice fingerprints into people, and lining up window-local labels.
//!
//! Two independent jobs, both pure and both tested against synthetic
//! fingerprints so neither needs a model on disk:
//!
//! * [`cluster`] decides how many people spoke and which fingerprint belongs to
//!   whom. Agglomerative, centroid linkage, cosine distance, one calibrated
//!   stopping threshold — see [`DISTANCE_THRESHOLD`] for why that number.
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
pub const DISTANCE_THRESHOLD: f32 = 0.58;

/// A cluster holding less speech than this is a fragment, not a person. Its
/// fingerprints are handed to whoever they resemble most instead of becoming a
/// "Speaker 5" the person then has to merge away.
pub const MIN_CLUSTER_MS: i64 = 3_000;

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

    // Normalise once; every distance below assumes unit length.
    let mut embeddings: Vec<Vec<f32>> = items.iter().map(|i| i.embedding.clone()).collect();
    for e in embeddings.iter_mut() {
        l2_normalize(e);
    }
    let dim = embeddings[0].len();
    if dim == 0 || embeddings.iter().any(|e| e.len() != dim) {
        // Nothing usable: everyone is one speaker rather than a crash.
        return Clustering {
            labels: vec![0; items.len()],
            cluster_count: 1,
            threshold,
        };
    }

    if items.len() > MAX_EXACT_ITEMS {
        return cluster_pooled(items, &embeddings, threshold);
    }
    cluster_exact(items, &embeddings, threshold)
}

fn cluster_exact(items: &[ClusterItem], embeddings: &[Vec<f32>], threshold: f32) -> Clustering {
    let mut nodes: Vec<Node> = embeddings
        .iter()
        .enumerate()
        .map(|(i, e)| Node::leaf(i, e, items[i].weight_ms))
        .collect();
    merge_loop(&mut nodes, threshold, MAX_SPEAKERS);
    absorb_fragments(&mut nodes);
    finish(nodes, items.len(), threshold)
}

/// Pool near-identical fingerprints first, then cluster the pools exactly.
///
/// The pre-pass uses half the threshold, so it only ever groups fingerprints
/// that the exact pass would certainly have grouped anyway.
fn cluster_pooled(items: &[ClusterItem], embeddings: &[Vec<f32>], threshold: f32) -> Clustering {
    let mut pools = pool(items, embeddings, threshold / 2.0);
    merge_loop(&mut pools, threshold, MAX_SPEAKERS);
    absorb_fragments(&mut pools);
    finish(pools, items.len(), threshold)
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

    let mut embeddings: Vec<Vec<f32>> = items.iter().map(|i| i.embedding.clone()).collect();
    for e in embeddings.iter_mut() {
        l2_normalize(e);
    }
    let dim = embeddings[0].len();
    if dim == 0 || embeddings.iter().any(|e| e.len() != dim) {
        // Nothing usable to measure with: one speaker rather than a crash, the
        // same answer `cluster` gives.
        return Clustering {
            labels: vec![0; items.len()],
            cluster_count: 1,
            threshold: 0.0,
        };
    }

    // Fewer voices to hand out than the person asked for: give back what exists.
    if items.len() <= k {
        return Clustering {
            labels: (0..items.len()).collect(),
            cluster_count: items.len(),
            threshold: 0.0,
        };
    }

    let mut nodes: Vec<Node> = if items.len() > MAX_EXACT_ITEMS {
        // Same pre-pass as the automatic path, and for the same reason: an exact
        // all-pairs loop over a two-hour meeting is expensive. Pooling at half
        // the calibrated distance only ever groups fingerprints the exact pass
        // would certainly have grouped, so it cannot change where the cut lands.
        pool(items, &embeddings, DISTANCE_THRESHOLD / 2.0)
    } else {
        embeddings
            .iter()
            .enumerate()
            .map(|(i, e)| Node::leaf(i, e, items[i].weight_ms))
            .collect()
    };
    let cut = merge_to_k(&mut nodes, k);
    finish(nodes, items.len(), cut)
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

/// Hand every fragment's fingerprints to the cluster they most resemble.
fn absorb_fragments(nodes: &mut [Node]) {
    loop {
        let alive: Vec<usize> = (0..nodes.len()).filter(|&i| nodes[i].alive).collect();
        if alive.len() < 2 {
            return;
        }
        // Smallest cluster first, so a chain of fragments resolves outward.
        let Some(&small) = alive
            .iter()
            .filter(|&&i| nodes[i].weight_ms < MIN_CLUSTER_MS)
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

fn finish(nodes: Vec<Node>, item_count: usize, threshold: f32) -> Clustering {
    let mut labels = vec![0usize; item_count];
    let mut next = 0usize;
    for node in nodes.iter().filter(|n| n.alive) {
        for &m in &node.members {
            labels[m] = next;
        }
        next += 1;
    }
    Clustering {
        labels,
        cluster_count: next.max(1),
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
