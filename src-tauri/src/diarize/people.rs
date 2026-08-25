//! The voices Echo was asked to remember.
//!
//! DESIGN §1 "Known people (voice enrollment)". A person is a name, a centroid
//! and a capped, condition-diverse set of samples; matching is **open-set**,
//! which is the whole difficulty. Clustering only ever has to answer "are these
//! two the same?" among voices that are all present. This module has to answer
//! "is this Marco, or is it somebody I have never heard?" — and the second
//! answer has to be available at all times, because most voices in most meetings
//! belong to nobody Echo knows.
//!
//! ## Three places a known voice is used, in order
//!
//! | when | what | why |
//! |---|---|---|
//! | before counting | [`pre_assign`] — fingerprints that are unmistakably one enrolled person | so the count question and the clustering only ever run on strangers |
//! | after clustering | [`decide`] over whole-cluster centroids | a centroid is a much better measurement than any one fingerprint |
//! | on a confirmation | [`learn_from_confirmation`] | the only thing that ever changes a profile |
//!
//! The first of those is the one that makes enrollment worth having at all.
//! Enrolled voices are matched *before* the count question (DESIGN §1), so a
//! meeting with two known people and one stranger asks the merge tree about one
//! voice instead of three — and blind counting, which is the part that gets a
//! meeting wrong, applies only to the people Echo has no other information
//! about.
//!
//! ## The margin rule
//!
//! "Nobody I know" is always a valid answer, so a threshold on its own is not
//! enough: a stranger who happens to sit 0.63 from Marco and 0.61 from Luca is
//! not Marco, they are somebody else entirely, and the tell is that the two
//! numbers are the same. So every decision here needs two things — a score over
//! the bar, **and** daylight between the best candidate and the second one. See
//! [`decide`] for what the bars are and how they were measured.
//!
//! The margin applies to suggestions as well as links, and that is deliberate. A
//! suggestion is a question put to a person — "Looks like Marco?" — and an
//! ambiguous one asks them to arbitrate between two names for a voice they
//! cannot hear. Silence is the better answer.
//!
//! ## Why nothing here trusts a number it did not measure itself
//!
//! A fingerprint means something only inside the network that produced it, and
//! [`super::cluster::DISTANCE_THRESHOLD`] is a long account of what happens when
//! a threshold and a network disagree about which geometry they are in. Profiles
//! are therefore tagged with the network their numbers came from
//! ([`embedder_tag`]); a profile whose tag does not match the network in use is
//! **skipped silently** and reported as needing a refresh, and
//! [`refresh_profiles`] re-embeds its kept clips to bring it back. Nobody is
//! ever asked to enroll again because Echo changed how it listens.
//!
//! ## What is confirmation-gated, and what that protects
//!
//! Profiles improve **only** from confirmations: accepting a suggestion, or
//! linking a speaker to a person by hand. The pass never learns on its own, at
//! any confidence. The reason is that a profile that learns from its own guesses
//! drifts: one wrong match adds a stranger's voice to Marco, which makes the
//! next wrong match more likely, and the failure is invisible until Marco stops
//! being recognised at all. [`curate`] is the other half of that defence — after
//! every confirmation the whole sample set is reviewed, near-duplicates are
//! dropped and a sample that disagrees with the rest of the profile is presumed
//! to be a mislabel and evicted (DESIGN §1: "the app reviews samples over time
//! to improve detection").

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use crate::db::{repo, Db};
use crate::types::{AssetKind, Channel, Id, PersonInfo, SuggestedPerson, TranscriptQuery};

use super::cluster::{self, cosine_distance, l2_normalize};
use super::embedding::Embedder;
use super::sample;
use super::DiarizeError;

// ===========================================================================
// The bars
// ===========================================================================

/// Similarity at which a whole cluster of one meeting's speech **is** a known
/// person, and their name goes on the chip.
///
/// Cosine similarity, `1 - `[`cosine_distance`], against the person's centroid.
/// Like every number in [`super::cluster`] it belongs to one specific network —
/// `catalog::ids::EMBEDDER`, WeSpeaker ResNet34-LM, 256 dimensions — and to
/// nothing else. A profile computed by any other network is not compared at all
/// (see [`embedder_tag`]).
///
/// # How it was measured
///
/// `cargo run --release --example people_fixture`, which is
/// `examples/voices_fixture.rs`'s procedure pointed at the open-set question
/// instead of the clustering one. Three fixtures built from macOS voices, run
/// through the real pass ([`super::pipeline::scan`]):
///
/// 1. **Enrollment.** Two voices, three turns each. Each voice's cluster gives
///    up [`SAMPLES_PER_CONFIRMATION`] fingerprints and [`centroid_of`] makes a
///    profile — so what is measured is a profile Echo could actually have on
///    disk, not an average over everything it heard.
/// 2. **Mixed.** One enrolled voice plus two never-enrolled ones. The right
///    person must be linked; neither stranger may be linked *or* suggested.
/// 3. **Strangers.** Three never-enrolled voices. Nobody linked, nobody
///    suggested.
///
/// Measured twice on 2026-08-21 (Samantha and Daniel enrolled, profiles 0.888
/// apart; Alice, Rishi and Tara as strangers; every fixture's count came out
/// right both times). The two runs agree to the third decimal — `say` does not
/// render bit-identically twice, so that is as stable as this gets:
///
/// ```text
/// enrolled voice against its own profile     0.932
/// stranger against the nearest profile       0.094 … 0.444
/// the gap the bars live in                   0.444 … 0.932
/// every TAU_LINK that gets all three right   0.460 … 0.920
/// ```
///
/// **0.64 sits deliberately above the middle of that plateau**, 0.196 clear of
/// the closest a stranger came and 0.292 below the true match. The two ways of
/// being wrong are not worth the same. Failing to link puts "Speaker 2" on a chip
/// the person fixes in one click — and the fix is worth *more* than the link
/// would have been, because it is a confirmation and it teaches the profile.
/// Linking wrongly puts a real person's name on a stranger's words in a
/// transcript that gets read, exported and summarised, and nothing in the UI
/// shouts "that is not Marco" loudly enough to catch it.
///
/// The stranger that came closest is worth naming: Alice, in the mixed meeting,
/// reached 0.444 against Samantha's profile *with* a healthy margin over the
/// runner-up. That is the case a margin rule cannot catch and only a bar can, and
/// it is why the bar is not lower.
///
/// # What this is not
///
/// Synthetic voices are cleaner and far more self-consistent than people — the
/// honest limits in `examples/voices_fixture.rs` apply here word for word — and
/// three fixtures are three fixtures, with one enrolled voice ever tested against
/// its own profile. A real voice heard down a bad connection will score lower
/// than 0.932, which is what [`TAU_SUGGEST`] is for. What the measurement rules
/// out is the indefensible thing: a bar picked by taste, in a space nobody
/// checked.
pub const TAU_LINK: f32 = 0.64;

/// Similarity at which Echo will *ask* — "Looks like Marco?" — without claiming.
///
/// The medium tier of DESIGN §1's three: above [`TAU_LINK`] the name goes on the
/// row, between the two bars the row carries a question and its score, below
/// this nothing is said at all. It exists for exactly the case the link bar gives
/// up on: a known person heard through a bad connection, or saying six words,
/// whose cluster centroid lands in the 0.5s.
///
/// **Measured, and then moved.** `examples/people_fixture` reports 0.46 as the
/// lowest bar that still says nothing about any stranger — one step above Alice's
/// 0.444. 0.52 is deliberately well above that floor, for two reasons: the
/// strangers in a real meeting scatter more than three text-to-speech voices do,
/// and a wrong suggestion is not free. It asks a person to arbitrate a name on a
/// voice they cannot hear, and the answer they give becomes a confirmation that
/// teaches the profile — so a wrong suggestion that gets accepted is a wrong
/// link with extra steps.
///
/// That leaves the suggestion band 0.52 … 0.64 narrow, which is the honest shape
/// of this: most voices are either recognisably somebody or recognisably nobody,
/// and the space in between is small.
pub const TAU_SUGGEST: f32 = 0.52;

/// How far the best candidate has to beat the second-best before either bar
/// counts for anything.
///
/// The open-set half of the rule, and the half no threshold can do. Two known
/// people who both score 0.66 against one voice do not make it the
/// slightly-higher one of them; they make it a voice that resembles a *kind* of
/// voice — which is what somebody's brother sounds like from in here.
///
/// **What the fixtures can and cannot say about this number.** On the measured
/// run a true match beat its runner-up by 0.787 at the cluster level and by
/// 0.673 … 0.876 per fingerprint, so 0.12 costs a right answer nothing at all.
/// What the fixtures *cannot* demonstrate is the rule earning its keep, and the
/// reason is in `examples/voices_fixture.rs`: the voice pool was deliberately
/// built out of voices at least 0.70 apart, because a fixture made of two voices
/// a network cannot tell apart is not ground truth for anything. The two enrolled
/// profiles sit 0.112 similarity apart — nowhere near close enough to make one
/// voice ambiguous between them.
///
/// So this is insurance, priced to cost nothing: it never fires on the fixtures,
/// and it is the only thing standing between a family of similar voices and a
/// confident wrong name. If two enrolled people ever do turn out to be members of
/// one voice family, the honest outcome is Echo refusing to tell them apart,
/// which is what this produces.
pub const MARGIN: f32 = 0.12;

/// Similarity at which a **single fingerprint** is handed to a known person
/// before any counting or clustering happens.
///
/// Higher than [`TAU_LINK`], and the reason is the unit of measurement.
/// [`TAU_LINK`] judges a cluster centroid: every fingerprint of one voice in the
/// meeting, averaged, with the phonemes averaged away. This judges one 1–8 second
/// fingerprint, which carries whichever sounds happened to be in those seconds as
/// well as the voice, and therefore scatters much more widely. Measured on the
/// same run:
///
/// ```text
/// enrolled voice, one fingerprint at a time   0.814 … 0.988
/// stranger, one fingerprint at a time         0.045 … 0.506
/// ```
///
/// 0.78 sits 0.274 above the closest a stranger's fingerprint came and 0.034
/// below the weakest true one — lopsided on purpose, and the asymmetry is
/// sharper here than anywhere else in this module. A wrongly pre-assigned
/// fingerprint is *removed from the clustering*: it does not merely mislabel a
/// voice, it takes evidence away from the count, which is the one number this
/// whole feature exists to improve. Being wrong the other way costs nothing
/// measurable — an enrolled voice that fails this bar is clustered like a
/// stranger and then linked by [`decide`] a moment later, which is the path it
/// would have taken if this optimisation did not exist.
///
/// At 0.78 the measured run pre-assigned all nine of the enrolled voice's
/// fingerprints and none of the twenty-five strangers'. In a real meeting fewer
/// will clear it, and that is a fallback rather than a failure.
pub const TAU_STRONG: f32 = 0.78;

/// A pre-assigned group holding less speech than this goes back in with the
/// strangers.
///
/// [`super::cluster::MIN_CLUSTER_MS`] asks the same question of a cluster, and
/// for the same reason: three seconds of speech is the least that can be called
/// a person rather than a fragment. Without it, one strong fingerprint of a cough
/// could take a whole speaker row and a name.
pub const MIN_GUIDED_MS: i64 = cluster::MIN_CLUSTER_MS;

/// Most samples Echo keeps per person.
///
/// Each one is about six seconds of audio (~190 KB) plus its numbers, so
/// twenty-four is around 4.5 MB per person — small enough to keep on a laptop
/// forever, large enough to hold both conditions and a spread of days. Past this
/// the marginal sample is a near-duplicate of one already kept, which is exactly
/// what [`choose_evictions`] measures.
pub const MAX_SAMPLES: usize = 24;

/// Most samples one confirmation may contribute.
///
/// DESIGN §1: "capped per meeting". Four is enough to cover both conditions and
/// two different moments; more than that and one long meeting would dominate a
/// profile that is supposed to describe a voice across meetings.
pub const SAMPLES_PER_CONFIRMATION: usize = 4;

/// Below this many samples, no sample is called an outlier.
///
/// With five samples of a voice there is no "the others" to be far from — the
/// median is one sample away from the extremes and evicting on it would just
/// throw away whichever one was recorded in a different room. Eight is where the
/// leave-one-out centroid stops being dominated by the sample being judged.
pub const OUTLIER_MIN_SAMPLES: usize = 8;

/// How far below the median a sample has to sit before it is presumed to be
/// somebody else's voice.
///
/// The outlier rule exists for one specific accident: a confirmation on the
/// wrong row. The person clicks "that's Marco" on a speaker that is not Marco,
/// four samples of a stranger go into Marco's profile, and from then on Marco's
/// centroid is a blend of two people that matches neither of them well. Nothing
/// upstream can prevent it — it is a person's own mistake, made in one click —
/// so the profile has to be able to notice.
///
/// What it notices: each sample's similarity to the centroid *of all the others*.
/// Samples of one voice sit close together and close to that centroid (0.75 …
/// 0.92 on the fixtures); a sample of a different voice sits far below all of
/// them (0.05 … 0.35). A gap of 0.25 under the median is well outside the spread
/// of one voice and well inside the distance to another one. The second condition
/// — the sample must also fail [`TAU_LINK`], i.e. it would not have been called
/// this person in the first place — is what stops a merely unusual sample of the
/// right voice from being thrown away.
pub const OUTLIER_GAP: f32 = 0.25;

/// How many meetings an unnamed voice has to turn up in before Echo offers to
/// remember it.
///
/// DESIGN §1: "a 'suggested people' list of recurring unnamed voices across
/// meetings". Three, because two is a coincidence — one meeting rescheduled, or
/// one conversation Echo happened to separate the same way twice — and offering
/// to remember somebody the person met once is noise in a settings screen.
pub const SUGGEST_MIN_APPEARANCES: u32 = 3;

/// Cosine distance at which two samples of one voice count as fully different
/// moments, for scoring diversity.
///
/// Not a decision, a scale: [`choose_evictions`] divides by it to turn a raw
/// distance into `0.0..=1.0`. [`super::cluster::SPLIT_FLOOR`] is the measured
/// upper end of how far one voice gets from itself, and it is the right end of
/// the ruler here too.
const DIVERSITY_FULL: f32 = cluster::SPLIT_FLOOR;

/// How much a sample's contribution to the spread counts against how new it is,
/// when something has to go. They sum to 1.
///
/// Diversity weighs more because it is the thing this cap is for: keeping the
/// twenty-four samples that describe the widest range of one voice. Recency is
/// the tie-break that matters — a voice changes, and the older of two
/// interchangeable samples is the one to lose.
const W_DIVERSITY: f32 = 0.6;
const W_RECENCY: f32 = 0.4;

// ===========================================================================
// Pure comparison
// ===========================================================================

/// How alike two voice prints are, in `-1.0..=1.0`. Higher is more alike.
///
/// `1 - `[`cosine_distance`], which is the dot product of two unit vectors —
/// every fingerprint in Echo is normalised where it is produced
/// ([`Embedder::embed`]) and every centroid here is normalised by
/// [`centroid_of`]. Vectors that cannot be compared at all (different lengths,
/// or empty — a profile from another network, or a corrupt row) score `-1.0`:
/// the honest answer for "no information", and one that can never clear a bar.
pub fn similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return -1.0;
    }
    1.0 - cosine_distance(a, b)
}

/// The mean of some voice prints, back on the unit sphere.
///
/// What a profile's centroid is, and what a cluster's centroid is. Unweighted:
/// every sample is one observation of the voice, and a sample that happens to
/// come from a longer turn is not more that person than a short one.
pub fn centroid_of<'a>(embeddings: impl IntoIterator<Item = &'a [f32]>) -> Vec<f32> {
    let mut sum: Vec<f32> = Vec::new();
    for e in embeddings {
        if sum.is_empty() {
            sum = e.to_vec();
            continue;
        }
        if e.len() != sum.len() {
            continue;
        }
        for (slot, x) in sum.iter_mut().zip(e) {
            *slot += x;
        }
    }
    l2_normalize(&mut sum);
    sum
}

/// What matching decided about one voice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Verdict {
    /// This is that person. Their name goes on the row.
    Linked { who: usize, score: f32 },
    /// This might be that person. The row carries the question, never the claim.
    Suggested { who: usize, score: f32 },
    /// Nobody Echo knows — always available, and the common answer.
    Nobody,
}

impl Verdict {
    pub fn who(&self) -> Option<usize> {
        match self {
            Verdict::Linked { who, .. } | Verdict::Suggested { who, .. } => Some(*who),
            Verdict::Nobody => None,
        }
    }
}

/// The open-set margin rule, over one voice's similarity to every known person.
///
/// `scores[i]` is how alike this voice and person `i` are. Returns which person
/// it is, or might be, or that it is nobody — see [`TAU_LINK`], [`TAU_SUGGEST`]
/// and [`MARGIN`] for the numbers and where they were measured.
///
/// With a single enrolled person there is no second-best to beat, and the margin
/// is measured against 0.0 — "no resemblance at all" — rather than waived, so the
/// first person ever enrolled is held to the same standard as the fifth.
pub fn decide(scores: &[f32]) -> Verdict {
    let Some((who, score, runner_up)) = top_two(scores) else {
        return Verdict::Nobody;
    };
    if score - runner_up < MARGIN {
        return Verdict::Nobody;
    }
    if score >= TAU_LINK {
        Verdict::Linked { who, score }
    } else if score >= TAU_SUGGEST {
        Verdict::Suggested { who, score }
    } else {
        Verdict::Nobody
    }
}

/// The best candidate, its score, and what it has to beat.
///
/// The floor is 0.0 — "no resemblance at all" — so a single enrolled person still
/// has to clear the margin over nothing, and a score below zero (an incomparable
/// profile, see [`similarity`]) can never win by default.
fn top_two(scores: &[f32]) -> Option<(usize, f32, f32)> {
    let mut best: Option<(usize, f32)> = None;
    let mut runner_up = 0.0f32;
    for (i, &score) in scores.iter().enumerate() {
        match best {
            Some((_, top)) if score > top => {
                runner_up = runner_up.max(top);
                best = Some((i, score));
            }
            Some(_) => runner_up = runner_up.max(score),
            None => best = Some((i, score)),
        }
    }
    best.map(|(who, score)| (who, score, runner_up))
}

/// The same rule at the pre-assignment bar, for one fingerprint.
///
/// Only ever "yes, unmistakably" or "no": there is no suggestion tier before
/// clustering, because a fingerprint that is only probably Marco is worth far
/// more inside the clustering than out of it. See [`TAU_STRONG`].
pub fn decide_strong(scores: &[f32]) -> Option<(usize, f32)> {
    let (who, score, runner_up) = top_two(scores)?;
    (score >= TAU_STRONG && score - runner_up >= MARGIN).then_some((who, score))
}

// ===========================================================================
// Profiles
// ===========================================================================

/// What the pass concluded about one of a meeting's voices, ready to go on the
/// speaker row.
///
/// [`Verdict`] is the decision; this is the decision with the person attached,
/// and it is what [`super::pipeline::ScanCut`] carries.
#[derive(Debug, Clone, PartialEq)]
pub enum Match {
    /// This is that person. `name` is copied onto the row (over a label Echo
    /// made up, never over one somebody typed) and is meeting-local from then on.
    Linked {
        person_id: Id,
        name: String,
        score: f32,
    },
    /// "Looks like Marco?" — the row carries the question and the score, and the
    /// person answers it. No name is copied anywhere.
    Suggested { person_id: Id, score: f32 },
}

impl Match {
    pub fn person_id(&self) -> &str {
        match self {
            Match::Linked { person_id, .. } | Match::Suggested { person_id, .. } => person_id,
        }
    }

    pub fn is_link(&self) -> bool {
        matches!(self, Match::Linked { .. })
    }
}

/// Decide who the voices that were *not* recognised before clustering belong to.
///
/// The second of the two matching stages, and the more reliable one: a whole
/// cluster's centroid is every fingerprint of that voice in the meeting averaged
/// together, which is a far better measurement than any one of them
/// ([`TAU_STRONG`] explains the difference). Leaves the pre-assigned voices
/// alone.
///
/// Two rules beyond [`decide`], both about the same thing — a person can only be
/// one voice in one meeting:
///
/// * a person already claimed by another voice in this meeting is not a candidate
///   for this one, and
/// * if two voices still end up claiming one person, the better score keeps them
///   and the other is left as nobody. Not even a suggestion: two voices in one
///   room that both look like Marco is a statement about the pass's separation,
///   not about Marco, and asking the person to arbitrate would be asking them to
///   confirm something false.
pub fn match_remaining(cut: &mut super::pipeline::ScanCut, enrolment: &Enrolment) {
    if enrolment.is_empty() {
        return;
    }
    let claimed: Vec<String> = cut
        .matches
        .iter()
        .flatten()
        .map(|m| m.person_id().to_string())
        .collect();

    let mut decided: Vec<(usize, Match)> = Vec::new();
    for (track, centroid) in cut.centroids.iter().enumerate() {
        if cut.matches.get(track).is_some_and(|m| m.is_some()) || centroid.is_empty() {
            continue;
        }
        let scores = enrolment.scores(centroid);
        let verdict = decide(&scores);
        let Some(who) = verdict.who() else { continue };
        let person = &enrolment.people[who];
        if claimed.contains(&person.person_id) {
            continue;
        }
        let found = match verdict {
            Verdict::Linked { score, .. } => Match::Linked {
                person_id: person.person_id.clone(),
                name: person.name.clone(),
                score,
            },
            Verdict::Suggested { score, .. } => Match::Suggested {
                person_id: person.person_id.clone(),
                score,
            },
            Verdict::Nobody => continue,
        };
        decided.push((track, found));
    }

    // One person, one voice: where two voices picked the same person, the better
    // score keeps them.
    let mut best_for: HashMap<String, (usize, f32)> = HashMap::new();
    for (track, found) in &decided {
        let score = match found {
            Match::Linked { score, .. } | Match::Suggested { score, .. } => *score,
        };
        let slot = best_for
            .entry(found.person_id().to_string())
            .or_insert((*track, score));
        if score > slot.1 {
            *slot = (*track, score);
        }
    }
    for (track, found) in decided {
        if best_for
            .get(found.person_id())
            .is_some_and(|(t, _)| *t == track)
        {
            if let Some(slot) = cut.matches.get_mut(track) {
                *slot = Some(found);
            }
        }
    }
}

/// An enrolled voice, ready to compare against.
#[derive(Debug, Clone)]
pub struct Enrolled {
    pub person_id: Id,
    /// The name that goes on a linked speaker row, copied at link time.
    pub name: String,
    pub centroid: Vec<f32>,
}

/// Who Echo can match right now, and who it had to leave out.
#[derive(Debug, Clone, Default)]
pub struct Enrolment {
    pub people: Vec<Enrolled>,
    /// Remembered voices that are **not** being matched at the moment and could
    /// be: their numbers came from a different network, or they have no profile
    /// at all. They are skipped silently and reported through
    /// [`PersonInfo::needs_refresh`] (DESIGN §1). Non-zero is what makes the
    /// pass run [`refresh_profiles`] before it counts anybody.
    pub stale: usize,
}

impl Enrolment {
    pub fn is_empty(&self) -> bool {
        self.people.is_empty()
    }

    /// Every score for one voice, in `people` order.
    pub fn scores(&self, print: &[f32]) -> Vec<f32> {
        self.people
            .iter()
            .map(|p| similarity(print, &p.centroid))
            .collect()
    }
}

/// The `models.id` of the voice-print network at `path`.
///
/// The tag a profile is stamped with, and the thing that decides whether it can
/// be compared at all. Read off the `models` table so it is the catalog's own
/// id; a file the table has never heard of falls back to its own name, which is
/// still stable and still changes when the model does. Both beat guessing.
pub async fn embedder_tag(db: &Db, path: &Path) -> String {
    let named = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    match repo::list_models(db, Some(AssetKind::SpeakerEmbedder)).await {
        Ok(rows) => rows
            .into_iter()
            .find(|m| m.path.as_deref().map(Path::new) == Some(path))
            .map(|m| m.id)
            .unwrap_or(named),
        Err(error) => {
            tracing::debug!(%error, "could not read which voice-print network this is");
            named
        }
    }
}

/// Everybody whose profile was computed by the network now in use.
///
/// `stale` is asked of the **people**, not of the profile rows. A voice with no
/// profile at all is exactly as unmatched as one whose numbers came from another
/// network, and [`merge`] makes that state on purpose every time it drops a
/// centroid rather than average two networks together. Counting only mismatched
/// rows left the merge survivor invisible to the one thing that repairs it — the
/// gate in [`super::pipeline`] that runs [`refresh_profiles`] when this is
/// non-zero — so Settings said Echo was refreshing while nothing ever was.
///
/// So this is [`repo::list_person_infos`]'s own `needs_refresh`, the field the
/// person reads, narrowed to the people a refresh can actually put back:
/// re-fingerprinting works off kept clips, and somebody with no samples left
/// (see `a_person_with_nothing_usable_left_stops_being_matched`) would otherwise
/// hold the gate open forever — a model loaded on every pass to finish no work.
pub async fn enrolled(db: &Db, embedder_tag: &str) -> Result<Enrolment, DiarizeError> {
    let profiles = repo::list_person_profiles(db).await.map_err(failed)?;
    let mut out = Enrolment::default();
    for profile in profiles {
        if profile.centroid.is_empty()
            || profile.sample_count == 0
            || profile.embedder_asset_id != embedder_tag
        {
            continue;
        }
        out.people.push(Enrolled {
            person_id: profile.person_id,
            name: profile.name,
            centroid: profile.centroid,
        });
    }
    out.stale = repo::list_person_infos(db, embedder_tag)
        .await
        .map_err(failed)?
        .into_iter()
        .filter(|person| person.needs_refresh && person.sample_count > 0)
        .count();
    Ok(out)
}

/// Which fingerprints are unmistakably somebody Echo already knows.
///
/// One entry per fingerprint: the index into `people`, or `None` for "cluster
/// this one like anybody else". Pure, so the bar in [`TAU_STRONG`] is tested
/// against synthetic geometry as well as measured against audio.
pub fn pre_assign(prints: &[&[f32]], people: &[Enrolled]) -> Vec<Option<usize>> {
    if people.is_empty() {
        return vec![None; prints.len()];
    }
    prints
        .iter()
        .map(|print| {
            let scores: Vec<f32> = people
                .iter()
                .map(|p| similarity(print, &p.centroid))
                .collect();
            decide_strong(&scores).map(|(who, _)| who)
        })
        .collect()
}

/// Which of the leftover clusters are more of a voice that was already
/// recognised before counting.
///
/// # The failure this exists to stop
///
/// [`pre_assign`] works one fingerprint at a time, at the strict
/// [`TAU_STRONG`] bar. A person's speech in a meeting does not clear that bar
/// uniformly: the clean stretches do, and the six-word answers and the sentences
/// with somebody laughing over them do not. So the guided group holds *some* of
/// that person's fingerprints and the rest fall through to the clustering — where
/// they look exactly like a coherent, unfamiliar voice, because they are a
/// coherent voice. It just is not an unfamiliar one.
///
/// Measured on a real two-person meeting
/// (`cargo run --release --example people_probe`): enrolling one of the two
/// voices took the meeting from **two** speakers to **three**, the enrolled
/// person split across two rows at 253 s and 271 s, each linked, each carrying
/// their name. Enrolling somebody made Echo worse at the meeting they were
/// enrolled from — the opposite of what DESIGN §1 promises ("enrolled voices are
/// matched BEFORE the count question, so blind counting applies only to
/// strangers"). The leftovers of a known voice are not strangers, and counting
/// them as one is the whole bug.
///
/// # The rule
///
/// A leftover *cluster's* centroid is a far better measurement than any single
/// fingerprint in it ([`TAU_STRONG`] explains why), so it is judged at the
/// ordinary claiming bar with the ordinary margin — [`decide`], the same rule
/// that puts a name on a row — and only a [`Verdict::Linked`] absorbs. A cluster
/// that merely *might* be a known person stays its own voice and reaches
/// [`match_remaining`] as a suggestion, which is the honest answer: merging
/// speech into somebody's track is a claim about who said it, and a claim needs
/// the claiming bar.
///
/// Only people who already have a guided group can absorb. An enrolled person
/// with no group at all is the normal case that [`match_remaining`] handles after
/// the counting, and pulling that decision forward would hand the clustering an
/// answer it never had to earn.
///
/// Returns, per leftover cluster, the guided group it belongs to — an index into
/// `groups`, which is the label the guided fingerprints already carry.
pub fn absorb_leftovers(
    leftovers: &[Vec<f32>],
    people: &[Enrolled],
    groups: &[usize],
) -> Vec<Option<usize>> {
    if groups.is_empty() {
        return vec![None; leftovers.len()];
    }
    leftovers
        .iter()
        .map(|centroid| {
            if centroid.is_empty() {
                return None;
            }
            let scores: Vec<f32> = people
                .iter()
                .map(|p| similarity(centroid, &p.centroid))
                .collect();
            match decide(&scores) {
                Verdict::Linked { who, .. } => groups.iter().position(|g| *g == who),
                Verdict::Suggested { .. } | Verdict::Nobody => None,
            }
        })
        .collect()
}

// ===========================================================================
// Learning from a confirmation
// ===========================================================================

/// What one confirmation did to a profile. Returned so the tests — and the log —
/// can see it; nothing user-facing carries these numbers (mantra 2).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Learned {
    pub added: usize,
    /// Dropped because the profile was already full of better-spread samples.
    pub evicted: usize,
    /// Dropped because they disagree with the rest of the profile — a mislabel
    /// from a confirmation on the wrong row.
    pub outliers: usize,
    /// What the profile holds now.
    pub kept: usize,
}

/// Learn this person's voice from a meeting they have just been confirmed in.
///
/// The only way a profile ever changes (see the module docs). Takes up to
/// [`SAMPLES_PER_CONFIRMATION`] clips of that speaker talking on their own,
/// preferring different moments and both conditions
/// ([`sample::pick_windows`]), stores each with its fingerprint, then re-curates
/// the whole set and recomputes the centroid.
///
/// **Never called from the pass.** Only from the commands a person's own click
/// reaches: linking a speaker to a person, and enrolling a new one.
///
/// Refuses when this voice never talks alone for long enough to be worth keeping
/// ([`DiarizeError::NoVoiceSample`]) and when the recording is gone
/// ([`DiarizeError::AudioForgotten`]). Both are ordinary answers, and both leave
/// the profile exactly as it was.
pub async fn learn_from_confirmation(
    db: &Db,
    meeting_id: &str,
    speaker_id: &str,
    person_id: &str,
) -> Result<Learned, DiarizeError> {
    let (_, embedder_path) = super::job::model_paths(db).await?;
    let tag = embedder_tag(db, &embedder_path).await;

    // Old numbers and new numbers have to live in the same space before either
    // the centroid or the curation means anything.
    refresh_person(db, person_id, &embedder_path, &tag).await?;

    let speakers = repo::list_speakers(db, meeting_id).await.map_err(failed)?;
    let Some(theirs) = sample::same_person(&speakers, speaker_id) else {
        return Err(DiarizeError::NoVoiceSample);
    };
    let segments = repo::get_segments(
        db,
        &TranscriptQuery {
            meeting_id: meeting_id.to_string(),
            limit: Some(50_000),
            ..Default::default()
        },
    )
    .await
    .map_err(failed)?;

    let picks = sample::pick_windows(&segments, &theirs, SAMPLES_PER_CONFIRMATION);
    if picks.is_empty() {
        // A voice with no lines at all and a voice with no clear moment in its
        // lines are different answers to the person; `why_no_moment` owns the
        // distinction.
        return Err(sample::why_no_moment(&segments, &theirs));
    }
    let clips = sample::clips_for(db, meeting_id, &picks).await?;
    if clips.is_empty() {
        return Err(DiarizeError::AudioForgotten);
    }

    let mut added = 0usize;
    {
        let mut embedder = Embedder::load(&embedder_path, 1)?;
        for (pick, samples) in &clips {
            let Some(print) = embedder.embed(samples)? else {
                continue;
            };
            let wav = sample::encode_wav_16k_mono(samples)?;
            repo::insert_person_sample(
                db,
                &repo::NewPersonSample {
                    person_id,
                    embedding: &print,
                    clip: &wav,
                    condition: condition_of(pick.channel),
                    source_meeting_id: Some(meeting_id),
                    t_start_ms: pick.from_ms,
                    t_end_ms: pick.to_ms,
                },
            )
            .await
            .map_err(failed)?;
            added += 1;
        }
    }
    if added == 0 {
        return Err(DiarizeError::NoVoiceSample);
    }

    let curated = curate(db, person_id, &tag).await?;
    tracing::debug!(
        added,
        evicted = curated.evicted,
        outliers = curated.outliers,
        kept = curated.kept,
        "a confirmation taught Echo a voice"
    );
    Ok(Learned {
        added,
        evicted: curated.evicted,
        outliers: curated.outliers,
        kept: curated.kept,
    })
}

/// Which recording a sample came off. Only two conditions exist as far as a
/// voice is concerned; a mixed-down clip is treated as the microphone, which is
/// what it mostly is.
fn condition_of(channel: Channel) -> Channel {
    match channel {
        Channel::System => Channel::System,
        _ => Channel::Mic,
    }
}

// ===========================================================================
// Curation
// ===========================================================================

/// One sample, as curation sees it. No audio: keeping or dropping a sample is
/// decided entirely on its numbers, its condition and its age.
#[derive(Debug, Clone)]
pub struct SampleFacts {
    pub id: Id,
    pub embedding: Vec<f32>,
    pub condition: Channel,
    /// 0 is the newest sample.
    pub age: usize,
}

/// What a curation pass changed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Curation {
    pub evicted: usize,
    pub outliers: usize,
    pub kept: usize,
}

/// Samples that disagree with the rest of the profile, worst first.
///
/// Each sample is scored against the centroid of *all the others* — leave-one-out,
/// because a sample compared against a centroid it is part of is partly compared
/// against itself, and with few samples that alone would hide a bad one. A sample
/// is an outlier when it sits [`OUTLIER_GAP`] below the median **and** fails
/// [`TAU_LINK`], i.e. it would not have been called this person at all.
///
/// Iterative: dropping the worst one changes every other sample's leave-one-out
/// centroid, which is how a confirmation on the wrong row loses all four of its
/// samples rather than one. Stops as soon as the set falls to
/// [`OUTLIER_MIN_SAMPLES`], where "the others" stops being a meaningful thing to
/// be far from.
pub fn find_outliers(samples: &[SampleFacts]) -> Vec<Id> {
    let mut live: Vec<usize> = (0..samples.len()).collect();
    let mut dropped: Vec<Id> = Vec::new();

    while live.len() >= OUTLIER_MIN_SAMPLES {
        let scores: Vec<f32> = live
            .iter()
            .map(|&i| {
                let rest = centroid_of(
                    live.iter()
                        .filter(|&&j| j != i)
                        .map(|&j| samples[j].embedding.as_slice()),
                );
                similarity(&samples[i].embedding, &rest)
            })
            .collect();

        let median = median_of(&scores);
        let Some((worst_at, worst)) = scores
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1).then(a.0.cmp(&b.0)))
            .map(|(i, s)| (i, *s))
        else {
            break;
        };
        if worst >= median - OUTLIER_GAP || worst >= TAU_LINK {
            break;
        }
        let index = live.remove(worst_at);
        tracing::debug!(
            similarity = worst,
            median,
            "dropping a sample that does not sound like the rest of this voice"
        );
        dropped.push(samples[index].id.clone());
    }
    dropped
}

/// Which samples to let go of to come back down to `keep`, in the order they go.
///
/// Every sample is scored on what it adds:
///
/// * **Spread** — how far it is from the nearest *other* sample. A near-duplicate
///   adds nothing: two clips of the same sentence from the same call are one
///   observation of the voice however many rows they occupy. This is the term the
///   cap is really for, and it weighs most ([`W_DIVERSITY`]).
/// * **Recency** — a voice changes, a room changes, a headset changes. Between
///   two interchangeable samples the older one goes.
///
/// One thing is a rule rather than a weight: **the last sample of a condition is
/// never evicted** while the profile has more than one condition in it. Losing
/// the only clip of somebody down a call would make the profile better at rooms
/// by making it useless at calls, and no weighting is worth trusting with that.
pub fn choose_evictions(samples: &[SampleFacts], keep: usize) -> Vec<Id> {
    let mut live: Vec<usize> = (0..samples.len()).collect();
    let mut out: Vec<Id> = Vec::new();
    let oldest = samples.iter().map(|s| s.age).max().unwrap_or(0).max(1) as f32;

    while live.len() > keep {
        let mut conditions: Vec<Channel> = live.iter().map(|&i| samples[i].condition).collect();
        conditions.sort_by_key(|c| c.as_str());
        conditions.dedup();
        // The candidate to lose: its score, its age, and where it sits in `live`.
        let mut worst: Option<(f32, usize, usize)> = None;

        for (pos, &i) in live.iter().enumerate() {
            let last_of_its_kind = conditions.len() > 1
                && live
                    .iter()
                    .filter(|&&j| samples[j].condition == samples[i].condition)
                    .count()
                    == 1;
            if last_of_its_kind {
                continue;
            }
            let nearest = live
                .iter()
                .filter(|&&j| j != i)
                .map(|&j| 1.0 - similarity(&samples[i].embedding, &samples[j].embedding))
                .fold(f32::INFINITY, f32::min);
            let spread = if nearest.is_finite() {
                (nearest / DIVERSITY_FULL).clamp(0.0, 1.0)
            } else {
                1.0
            };
            let recency = 1.0 - (samples[i].age as f32 / oldest).clamp(0.0, 1.0);
            let score = W_DIVERSITY * spread + W_RECENCY * recency;
            // Lowest score goes. Between equals the older one goes, and between
            // those the earlier row — candidates are visited in order and only a
            // strictly better one takes over, so this is deterministic.
            let takes_it = match worst {
                None => true,
                Some((lowest, oldest, _)) => {
                    score < lowest || (score == lowest && samples[i].age > oldest)
                }
            };
            if takes_it {
                worst = Some((score, samples[i].age, pos));
            }
        }

        match worst {
            Some((_, _, pos)) => {
                let index = live.remove(pos);
                out.push(samples[index].id.clone());
            }
            // Everything left is the last of its condition. Better a profile
            // slightly over the cap than one that can only recognise this person
            // in one of the two ways they turn up.
            None => break,
        }
    }
    out
}

/// Review a person's whole sample set, then rewrite their profile from what is
/// left.
///
/// Order matters: outliers go first, because a mislabelled sample is *also* an
/// unusually well-spread one and would survive the cap on exactly the grounds
/// that make it wrong.
pub async fn curate(
    db: &Db,
    person_id: &str,
    embedder_tag: &str,
) -> Result<Curation, DiarizeError> {
    let rows = repo::list_person_samples(db, person_id)
        .await
        .map_err(failed)?;

    // Rows the current network cannot read at all — a sample stored before a
    // change of network whose clip could not be re-embedded. Not an outlier
    // (nothing can be measured about it), just unusable.
    let dim = modal_dim(&rows);
    let mut doomed: Vec<Id> = rows
        .iter()
        .filter(|r| Some(r.embedding.len()) != dim)
        .map(|r| r.id.clone())
        .collect();

    let facts: Vec<SampleFacts> = rows
        .iter()
        .filter(|r| Some(r.embedding.len()) == dim)
        .enumerate()
        .map(|(age, r)| SampleFacts {
            id: r.id.clone(),
            embedding: r.embedding.clone(),
            condition: r.condition,
            age,
        })
        .collect();

    let outliers = find_outliers(&facts);
    let surviving: Vec<SampleFacts> = facts
        .iter()
        .filter(|f| !outliers.contains(&f.id))
        .cloned()
        .collect();
    let evicted = choose_evictions(&surviving, MAX_SAMPLES);

    doomed.extend(outliers.iter().cloned());
    doomed.extend(evicted.iter().cloned());
    repo::delete_person_samples(db, &doomed)
        .await
        .map_err(failed)?;

    let kept: Vec<&SampleFacts> = surviving
        .iter()
        .filter(|f| !evicted.contains(&f.id))
        .collect();
    if kept.is_empty() {
        // No usable material left. Dropping the profile row is what makes this
        // person stop being matched and start being reported as needing a
        // refresh, which is the truth about them.
        repo::delete_person_profile(db, person_id)
            .await
            .map_err(failed)?;
    } else {
        let centroid = centroid_of(kept.iter().map(|f| f.embedding.as_slice()));
        repo::upsert_person_profile(db, person_id, &centroid, kept.len() as u32, embedder_tag)
            .await
            .map_err(failed)?;
    }

    Ok(Curation {
        evicted: evicted.len(),
        outliers: outliers.len(),
        kept: kept.len(),
    })
}

fn modal_dim(rows: &[repo::PersonSampleRow]) -> Option<usize> {
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for row in rows {
        if row.embedding.is_empty() {
            continue;
        }
        *counts.entry(row.embedding.len()).or_default() += 1;
    }
    // Most common length wins; the largest wins a tie, because a truncated blob
    // is the failure mode and it is never the longer one.
    counts
        .into_iter()
        .max_by_key(|(dim, count)| (*count, *dim))
        .map(|(dim, _)| dim)
}

fn median_of(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

// ===========================================================================
// Keeping profiles usable across a change of network
// ===========================================================================

/// Re-embed every kept clip of every profile whose numbers came from a different
/// network, and rewrite their centroids.
///
/// The whole reason [`repo::NewPersonSample::clip`] exists. Quiet by design: it
/// says nothing to the person, because from where they are sitting nothing has
/// happened — the people they told Echo to remember are still remembered.
///
/// Returns how many people were brought back into service.
pub async fn refresh_profiles(db: &Db, embedder_path: &Path) -> Result<usize, DiarizeError> {
    let tag = embedder_tag(db, embedder_path).await;
    let people = repo::list_people(db).await.map_err(failed)?;
    let profiles = repo::list_person_profiles(db).await.map_err(failed)?;
    let fresh: BTreeSet<&str> = profiles
        .iter()
        .filter(|p| p.embedder_asset_id == tag)
        .map(|p| p.person_id.as_str())
        .collect();

    let stale: Vec<Id> = people
        .iter()
        .filter(|p| !fresh.contains(p.id.as_str()))
        .map(|p| p.id.clone())
        .collect();
    if stale.is_empty() {
        return Ok(0);
    }

    let mut embedder = Embedder::load(embedder_path, 1)?;
    let mut done = 0usize;
    for person_id in stale {
        match re_embed(db, &person_id, &mut embedder).await {
            Ok(0) => {}
            Ok(_) => {
                curate(db, &person_id, &tag).await?;
                done += 1;
            }
            Err(error) => {
                // One unreadable profile must not stop the others. It stays
                // reported as needing a refresh, which is true.
                tracing::debug!(%error, "could not bring a remembered voice up to date");
            }
        }
    }
    if done > 0 {
        tracing::info!(people = done, "brought remembered voices up to date");
    }
    Ok(done)
}

/// [`refresh_profiles`] for one person, and only when they need it.
///
/// Called before a confirmation adds anything, so the new samples and the old
/// ones are measured in the same space.
async fn refresh_person(
    db: &Db,
    person_id: &str,
    embedder_path: &Path,
    tag: &str,
) -> Result<(), DiarizeError> {
    let current = repo::list_person_profiles(db)
        .await
        .map_err(failed)?
        .into_iter()
        .find(|p| p.person_id == person_id);
    if current.is_some_and(|p| p.embedder_asset_id == tag) {
        return Ok(());
    }
    let mut embedder = Embedder::load(embedder_path, 1)?;
    if re_embed(db, person_id, &mut embedder).await? > 0 {
        curate(db, person_id, tag).await?;
    }
    Ok(())
}

/// Fingerprint every clip this person has again. Returns how many worked.
async fn re_embed(
    db: &Db,
    person_id: &str,
    embedder: &mut Embedder,
) -> Result<usize, DiarizeError> {
    let clips = repo::person_sample_clips(db, person_id)
        .await
        .map_err(failed)?;
    let mut done = 0usize;
    for (sample_id, wav) in clips {
        let samples = match sample::decode_wav_16k_mono(&wav) {
            Ok(s) => s,
            Err(error) => {
                tracing::debug!(%error, "a kept clip could not be read back");
                continue;
            }
        };
        if let Some(print) = embedder.embed(&samples)? {
            repo::set_person_sample_embedding(db, &sample_id, &print)
                .await
                .map_err(failed)?;
            done += 1;
        }
    }
    Ok(done)
}

// ===========================================================================
// Recurring unnamed voices
// ===========================================================================

/// Voices that keep turning up without a name.
///
/// Built entirely from the per-meeting voice prints the offline pass leaves on
/// `speakers`, which is why this costs no audio reads at all — the reason those
/// prints are stored (DESIGN §1: "per-meeting speaker centroids are stored to
/// make this possible without re-reading audio").
///
/// The rules, in order:
///
/// * Two prints group together when they are as alike as a link would need
///   ([`TAU_LINK`]). Grouping is not the same decision as naming, but it is the
///   same question — "is this the same voice?" — and using a lower bar here would
///   offer to remember a *kind* of voice.
/// * **Never two speakers from one meeting.** The pass already decided those are
///   different people; a group that swallowed both would be claiming the pass was
///   wrong, which is not this list's job.
/// * At least [`SUGGEST_MIN_APPEARANCES`] different meetings.
/// * A group that already matches somebody enrolled is dropped: that voice is
///   known, and offering to remember them again would produce two Marcos.
pub async fn suggested_people(db: &Db) -> Result<Vec<SuggestedPerson>, DiarizeError> {
    let prints = repo::list_unnamed_voice_prints(db).await.map_err(failed)?;
    let known = repo::list_person_profiles(db).await.map_err(failed)?;

    /// A group under construction.
    struct Group {
        members: Vec<repo::VoicePrint>,
        centroid: Vec<f32>,
    }

    let mut groups: Vec<Group> = Vec::new();
    for print in prints.into_iter().filter(|p| !p.centroid.is_empty()) {
        // Best group that this meeting is not already in.
        let mut best: Option<(usize, f32)> = None;
        for (g, group) in groups.iter().enumerate() {
            if group
                .members
                .iter()
                .any(|m| m.meeting_id == print.meeting_id)
            {
                continue;
            }
            let score = similarity(&print.centroid, &group.centroid);
            if score >= TAU_LINK && best.is_none_or(|(_, top)| score > top) {
                best = Some((g, score));
            }
        }
        match best {
            Some((g, _)) => {
                groups[g].members.push(print);
                groups[g].centroid =
                    centroid_of(groups[g].members.iter().map(|m| m.centroid.as_slice()));
            }
            None => {
                let centroid = print.centroid.clone();
                groups.push(Group {
                    members: vec![print],
                    centroid,
                });
            }
        }
    }

    let mut out: Vec<SuggestedPerson> = Vec::new();
    for group in groups {
        let meetings: BTreeSet<&str> = group
            .members
            .iter()
            .map(|m| m.meeting_id.as_str())
            .collect();
        if (meetings.len() as u32) < SUGGEST_MIN_APPEARANCES {
            continue;
        }
        let already_known = known
            .iter()
            .any(|k| similarity(&group.centroid, &k.centroid) >= TAU_LINK);
        if already_known {
            continue;
        }
        // The meeting this voice said the most in: the best clip to listen to,
        // and the best material to enroll from.
        let Some(representative) = group.members.iter().max_by_key(|m| m.speaking_ms) else {
            continue;
        };
        out.push(SuggestedPerson {
            id: representative.speaker_id.clone(),
            appearances: meetings.len() as u32,
            last_heard_at: group
                .members
                .iter()
                .map(|m| m.started_at.clone())
                .max()
                .unwrap_or_default(),
            speaking_ms: group.members.iter().map(|m| m.speaking_ms).sum(),
            meeting_id: representative.meeting_id.clone(),
            speaker_id: representative.speaker_id.clone(),
            meeting_title: representative.meeting_title.clone(),
        });
    }
    // Most-heard first: the voice in six meetings is the one worth naming.
    out.sort_by(|a, b| {
        b.appearances
            .cmp(&a.appearances)
            .then(b.last_heard_at.cmp(&a.last_heard_at))
            .then(a.id.cmp(&b.id))
    });
    Ok(out)
}

// ===========================================================================
// Commands' half: the plain operations
// ===========================================================================

/// Everybody Echo remembers, with the refresh flag worked out against the
/// network in use. Costs no model load.
pub async fn list_people(db: &Db) -> Result<Vec<PersonInfo>, DiarizeError> {
    // No embedder installed at all: nothing can be matched, so nothing claims to
    // be fresh. An empty tag matches no profile.
    let tag = match crate::asr::models::installed_path(db, AssetKind::SpeakerEmbedder).await {
        Ok(Some(path)) => embedder_tag(db, &path).await,
        _ => String::new(),
    };
    repo::list_person_infos(db, &tag).await.map_err(failed)
}

/// Remember a new voice: create the person, learn from this meeting, link the
/// speaker.
///
/// The order is the whole point. Learning comes **before** the link, so a
/// meeting with no clear moment of that voice on its own — or no recording left
/// — costs nothing at all: the half-made person is removed and the speaker row is
/// never touched. "Remember this voice" either remembers a voice or says why it
/// cannot; what it must never do is leave a name behind with nothing behind the
/// name.
pub async fn enroll(
    db: &Db,
    meeting_id: &str,
    speaker_id: &str,
    name: &str,
) -> Result<PersonInfo, DiarizeError> {
    // Check the row belongs to this meeting before creating anything for it.
    let speaker = this_meetings_speaker(db, meeting_id, speaker_id).await?;

    let person = repo::create_person(db, name).await.map_err(failed)?;
    if let Err(error) = learn_from_confirmation(db, meeting_id, speaker_id, &person.id).await {
        let _ = repo::delete_person(db, &person.id).await;
        return Err(error);
    }
    attach(db, &speaker, &person.id, name).await?;

    list_people(db)
        .await?
        .into_iter()
        .find(|p| p.id == person.id)
        .ok_or_else(|| DiarizeError::Failed("the person disappeared while being created".into()))
}

/// Point a meeting's speaker at a known person, or let it go.
///
/// Linking is a confirmation, so it teaches the profile
/// ([`learn_from_confirmation`]) — and it copies the person's name onto the
/// speaker row **only if that row still carries a name Echo made up**. A name the
/// person typed is theirs; this is also why renaming a linked speaker later does
/// not rename the person (DESIGN §1 keeps meeting-local display separate from
/// identity).
///
/// Unlinking releases the row and leaves the name where it is: the transcript
/// somebody has already read does not rewrite itself.
///
/// A failure to learn does **not** fail the link, and does not undo it. The link
/// is what the person asked for and it is right whether or not this particular
/// meeting happened to contain six clean seconds of that voice — so that case is
/// logged and the profile is left as it was. [`enroll`] is the one caller that
/// cannot take that view, because it has nothing to fall back on.
pub async fn link(
    db: &Db,
    meeting_id: &str,
    speaker_id: &str,
    person_id: Option<&str>,
) -> Result<(), DiarizeError> {
    let speaker = this_meetings_speaker(db, meeting_id, speaker_id).await?;

    let Some(person_id) = person_id else {
        repo::set_speaker_person(db, speaker_id, None)
            .await
            .map_err(failed)?;
        return Ok(());
    };
    // Through the merges, not straight at the row: a window opened before two
    // names were made one still sends the old id, and the person it means is
    // whoever that name is now.
    let person = repo::resolve_person(db, person_id)
        .await
        .map_err(failed)?
        .ok_or_else(|| DiarizeError::Failed("Echo does not remember that voice".into()))?;

    attach(db, &speaker, &person.id, &person.name).await?;
    if let Err(error) = learn_from_confirmation(db, meeting_id, speaker_id, &person.id).await {
        tracing::debug!(%error, "linked a voice there was nothing new to learn from");
    }
    Ok(())
}

/// What became of a voice when two names were made one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceAfterMerge {
    /// Nothing about the surviving voice changed: the merged name had no samples
    /// to give it.
    Unchanged,
    /// The two sets of samples were measured by the same network, so they were
    /// curated into one profile here and now. `kept` is how many samples the
    /// surviving voice is described by.
    Recomputed { kept: usize },
    /// The samples arrived but their numbers were not comparable with the
    /// survivor's, so the profile was dropped rather than averaged across two
    /// different spaces. The person is reported as needing a refresh — which is
    /// true — and [`refresh_profiles`] re-fingerprints every kept clip and puts
    /// them back in service without anybody being asked anything.
    ///
    /// What makes that automatic rather than a promise: a person with no profile
    /// counts towards [`Enrolment::stale`], which is the gate the pass checks
    /// before it counts anybody, so the next meeting's pass rebuilds this voice
    /// before it tries to recognise it.
    NeedsRefresh,
}

/// What one merge did, for the log line and for the tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Merged {
    /// False when the two were already one person. Nothing was written.
    pub merged: bool,
    pub samples_moved: u64,
    pub speakers_relinked: u64,
    pub voice: VoiceAfterMerge,
}

/// "These two are the same person": make one remembered voice out of two.
///
/// The same voice gets enrolled twice — once from a call, once from the room —
/// and until now the only way to tidy that up was to delete one of them, which
/// destroyed that half of the voice: its samples, its clips, and the ability to
/// re-fingerprint it when the network changes. This keeps both halves.
///
/// [`repo::merge_people`] moves the samples and the links and records the merge;
/// this half owns the one thing that layer cannot decide — **which network the
/// numbers belong to**:
///
/// * Both sides measured by the same network → the samples are one set and
///   [`curate`] makes one profile out of them, outliers and near-duplicates
///   dropped exactly as a confirmation would. The surviving voice describes both
///   from the moment this returns.
/// * Anything else — one side never fingerprinted, or fingerprinted by a
///   different network — → the survivor's profile is **dropped**, not averaged.
///   Two centroids from two networks describe geometries neither one has been
///   in (`0004_known_people.sql`), and a blend of them matches nobody. Settings
///   then says this person needs a refresh, and the ordinary refresh re-embeds
///   every kept clip and rebuilds the profile from the union.
///
/// **Not reversible.** The identity is: the merged name keeps its row pointing
/// at the survivor. The voice is not: curating one set out of two deletes the
/// samples it does not keep, and that deletion is what makes the survivor one
/// voice rather than two. Ask before calling this.
///
/// Meetings are left exactly as they read. Speaker rows are re-pointed at the
/// survivor, but the names on them were copied there when the link was made and
/// stay as plain text — the same rule [`link`], [`super::rename`] and
/// `delete_person` all follow, and for the same reason: a meeting somebody has
/// already read does not rewrite itself.
///
/// Safe to call twice, in either order: the second call finds the two already
/// one person and writes nothing.
pub async fn merge(db: &Db, keep_id: &str, merge_id: &str) -> Result<Merged, DiarizeError> {
    // Read both tags while both profile rows still exist — the merge deletes one
    // of them, and after that there is no way to tell what space its numbers
    // were in.
    let keep_root = root_of(db, keep_id).await?;
    let merge_root = root_of(db, merge_id).await?;
    if keep_root == merge_root {
        // Already one person — a second click, or a name somebody's other window
        // merged a moment ago. Nothing to write and nothing to say.
        return Ok(Merged {
            merged: false,
            samples_moved: 0,
            speakers_relinked: 0,
            voice: VoiceAfterMerge::Unchanged,
        });
    }
    let keep_tag = profile_tag(db, &keep_root).await?;
    let merge_tag = profile_tag(db, &merge_root).await?;

    let moved = repo::merge_people(db, &keep_root, &merge_root)
        .await
        .map_err(|e| match e {
            crate::db::DbError::Invalid(why) => DiarizeError::CannotMerge(why),
            other => failed(other),
        })?;
    if !moved.merged {
        return Ok(Merged {
            merged: false,
            samples_moved: 0,
            speakers_relinked: 0,
            voice: VoiceAfterMerge::Unchanged,
        });
    }

    let voice = if moved.samples_moved == 0 {
        VoiceAfterMerge::Unchanged
    } else if keep_tag.is_some() && keep_tag == merge_tag {
        let tag = keep_tag.clone().unwrap_or_default();
        match curate(db, &keep_root, &tag).await {
            Ok(curation) => VoiceAfterMerge::Recomputed {
                kept: curation.kept,
            },
            Err(error) => {
                // The merge itself has landed. Leaving a centroid that now
                // describes the wrong set of samples would be worse than saying
                // the voice needs looking at again, which is true and repairs
                // itself the next time the refresh runs.
                tracing::warn!(%error, "made one voice out of two but could not rebuild its profile");
                repo::delete_person_profile(db, &keep_root)
                    .await
                    .map_err(failed)?;
                VoiceAfterMerge::NeedsRefresh
            }
        }
    } else {
        repo::delete_person_profile(db, &keep_root)
            .await
            .map_err(failed)?;
        VoiceAfterMerge::NeedsRefresh
    };

    tracing::info!(
        samples_moved = moved.samples_moved,
        speakers_relinked = moved.speakers_relinked,
        suggestions_relinked = moved.suggestions_relinked,
        aliases_flattened = moved.aliases_flattened,
        voice = ?voice,
        "made one remembered voice out of two"
    );
    Ok(Merged {
        merged: true,
        samples_moved: moved.samples_moved,
        speakers_relinked: moved.speakers_relinked,
        voice,
    })
}

/// The id this name is kept under now, refusing a name Echo has never heard of.
async fn root_of(db: &Db, person_id: &str) -> Result<Id, DiarizeError> {
    repo::resolve_person(db, person_id)
        .await
        .map_err(failed)?
        .map(|p| p.id)
        .ok_or_else(|| DiarizeError::Failed("Echo does not remember that voice".into()))
}

/// Which network this person's numbers were measured by, if they have any.
async fn profile_tag(db: &Db, person_id: &str) -> Result<Option<String>, DiarizeError> {
    Ok(repo::person_profile(db, person_id)
        .await
        .map_err(failed)?
        .filter(|p| !p.centroid.is_empty() && p.sample_count > 0)
        .map(|p| p.embedder_asset_id))
}

/// The speaker row, if it is really one of this meeting's.
///
/// Ids come from the webview and a speaker id from another meeting would
/// otherwise link a person to a row nobody is looking at.
async fn this_meetings_speaker(
    db: &Db,
    meeting_id: &str,
    speaker_id: &str,
) -> Result<crate::types::Speaker, DiarizeError> {
    let speaker = repo::get_speaker(db, speaker_id)
        .await
        .map_err(failed)?
        .ok_or_else(|| DiarizeError::Failed("that speaker is not there any more".into()))?;
    if speaker.meeting_id != meeting_id {
        return Err(DiarizeError::Failed(
            "that speaker belongs to a different meeting".into(),
        ));
    }
    Ok(speaker)
}

/// Set the link, and copy the name over a label Echo made up.
async fn attach(
    db: &Db,
    speaker: &crate::types::Speaker,
    person_id: &str,
    name: &str,
) -> Result<(), DiarizeError> {
    repo::set_speaker_person(db, &speaker.id, Some(person_id))
        .await
        .map_err(failed)?;
    if is_default_name(&speaker.display_name) {
        repo::rename_speaker(db, &speaker.id, name)
            .await
            .map_err(failed)?;
    }
    Ok(())
}

/// Is this a name Echo made up, or one somebody typed?
///
/// "Speaker 3" and "You" are Echo's own labels ([`super::pipeline::display_name`],
/// [`super::pipeline::SELF_DISPLAY_NAME`]) and may be replaced by a person's real
/// name without asking. Anything else was chosen by a human and outranks
/// anything this module knows.
pub fn is_default_name(display_name: &str) -> bool {
    if display_name == super::pipeline::SELF_DISPLAY_NAME {
        return true;
    }
    (0..cluster::MAX_SPEAKERS).any(|i| display_name == super::pipeline::display_name(i))
}

/// A few seconds of a remembered voice, base64 WAV — the freshest clip kept.
///
/// Straight out of the profile: no audio is read off disk and no meeting has to
/// still exist, which is the point of keeping the clips at all.
pub async fn person_sample_audio(db: &Db, person_id: &str) -> Result<String, DiarizeError> {
    let clip = repo::freshest_person_clip(db, person_id)
        .await
        .map_err(failed)?
        .ok_or(DiarizeError::NoVoiceSample)?;
    Ok(sample::to_base64(&clip))
}

fn failed(err: crate::db::DbError) -> DiarizeError {
    DiarizeError::Failed(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Channel, MeetingStatus, SegmentDraft};

    /// Room for eight orthogonal "people" plus a noise region, which is more than
    /// any test below needs.
    const DIM: usize = 32;

    async fn db() -> Db {
        let db = crate::db::connect_in_memory().await.expect("in-memory db");
        crate::db::migrate(&db).await.expect("migrations");
        db
    }

    /// Person `p`'s centroid: the unit vector along one dimension of its own, so
    /// every person is exactly orthogonal to every other one and a test can state
    /// what the numbers are instead of hoping.
    fn axis(p: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; DIM];
        v[p * 2] = 1.0;
        v
    }

    fn person(p: usize, name: &str) -> Enrolled {
        Enrolled {
            person_id: format!("person-{p}"),
            name: name.to_string(),
            centroid: axis(p),
        }
    }

    /// A voice print whose similarity to person `i` is exactly `scores[i]`.
    ///
    /// Possible because the people are orthogonal: the print is that combination
    /// of their axes plus whatever length is left over, put somewhere nobody
    /// occupies.
    fn print_scoring(scores: &[f32]) -> Vec<f32> {
        let mut v = vec![0.0f32; DIM];
        let mut used = 0.0f32;
        for (i, &s) in scores.iter().enumerate() {
            v[i * 2] = s;
            used += s * s;
        }
        v[DIM - 1] = (1.0 - used).max(0.0).sqrt();
        v
    }

    /// A sample of the voice on `axis`, with a small wobble in a dimension of its
    /// own so two samples of one voice are close without being identical.
    /// The same voice from a genuinely different moment: still much nearer to
    /// person 0 than to anybody else, but a real distance from the other samples
    /// — what a profile with spread in it is made of.
    fn another_moment(k: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; DIM];
        v[0] = 1.0;
        v[k * 2 + 2] = 0.8;
        l2_normalize(&mut v);
        v
    }

    fn sample_near(axis_index: usize, wobble: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; DIM];
        v[axis_index * 2] = 1.0;
        v[16 + wobble % 8] = 0.15;
        l2_normalize(&mut v);
        v
    }

    fn facts(id: &str, embedding: Vec<f32>, condition: Channel, age: usize) -> SampleFacts {
        SampleFacts {
            id: id.to_string(),
            embedding,
            condition,
            age,
        }
    }

    /// A clip that is a real, playable file, because that is what the schema
    /// promises and what a refresh has to be able to read back.
    fn clip() -> Vec<u8> {
        sample::encode_wav_16k_mono(&[0.05f32; 1_600]).expect("a wav")
    }

    async fn person_with_samples(
        db: &Db,
        name: &str,
        samples: &[(Vec<f32>, Channel)],
        tag: &str,
    ) -> Id {
        let row = repo::create_person(db, name).await.unwrap();
        let wav = clip();
        for (embedding, condition) in samples {
            repo::insert_person_sample(
                db,
                &repo::NewPersonSample {
                    person_id: &row.id,
                    embedding,
                    clip: &wav,
                    condition: *condition,
                    source_meeting_id: None,
                    t_start_ms: 0,
                    t_end_ms: 6_000,
                },
            )
            .await
            .unwrap();
        }
        let centroid = centroid_of(samples.iter().map(|(e, _)| e.as_slice()));
        repo::upsert_person_profile(db, &row.id, &centroid, samples.len() as u32, tag)
            .await
            .unwrap();
        row.id
    }

    // -----------------------------------------------------------------------
    // Comparing
    // -----------------------------------------------------------------------

    #[test]
    fn a_voice_is_perfectly_like_itself_and_says_nothing_about_one_it_cannot_compare() {
        let v = sample_near(0, 1);
        assert!((similarity(&v, &v) - 1.0).abs() < 1e-6);
        assert!(similarity(&v, &axis(1)) < 0.05);
        // No information at all is -1.0, which can never clear a bar.
        assert_eq!(similarity(&v, &[]), -1.0);
        assert_eq!(similarity(&v, &[0.5, 0.5]), -1.0);
        assert_eq!(similarity(&[], &[]), -1.0);
    }

    #[test]
    fn a_centroid_is_the_mean_put_back_on_the_unit_sphere() {
        let c = centroid_of([axis(0).as_slice(), axis(1).as_slice()]);
        let length: f32 = c.iter().map(|x| x * x).sum();
        assert!((length - 1.0).abs() < 1e-5, "not unit length: {length}");
        // Halfway between two orthogonal voices, so equally unlike both.
        assert!((similarity(&c, &axis(0)) - similarity(&c, &axis(1))).abs() < 1e-6);
        // A vector of a different length cannot join in, and an empty set is
        // empty rather than a panic.
        assert!(centroid_of(Vec::<&[f32]>::new()).is_empty());
        let mixed = centroid_of([axis(0).as_slice(), [1.0f32, 2.0].as_slice()]);
        assert_eq!(mixed.len(), DIM);
    }

    // -----------------------------------------------------------------------
    // The margin rule
    // -----------------------------------------------------------------------

    #[test]
    fn the_matrix_of_what_one_voice_can_be_decided_to_be() {
        // Over the linking bar, alone in front: that person.
        assert_eq!(
            decide(&[0.90, 0.10]),
            Verdict::Linked {
                who: 0,
                score: 0.90
            }
        );
        // Between the bars: a question, not a claim.
        assert!(matches!(
            decide(&[0.56, 0.10]),
            Verdict::Suggested { who: 0, .. }
        ));
        // Under both bars: nobody, which is always available.
        assert_eq!(decide(&[0.44, 0.10]), Verdict::Nobody);
        // Over the linking bar but level with the runner-up: nobody. This is the
        // whole point of the margin — a voice that resembles a kind of voice is
        // not the slightly-higher one of them.
        assert_eq!(decide(&[0.90, 0.85]), Verdict::Nobody);
        // The same rule in the suggestion band.
        assert_eq!(decide(&[0.58, 0.52]), Verdict::Nobody);
        // Nobody enrolled at all.
        assert_eq!(decide(&[]), Verdict::Nobody);
        // One person enrolled: the margin is measured against no resemblance at
        // all rather than waived, so the first person is held to the same
        // standard as the fifth.
        assert!(matches!(decide(&[0.90]), Verdict::Linked { who: 0, .. }));
        assert_eq!(decide(&[0.08]), Verdict::Nobody);
    }

    #[test]
    fn the_winner_is_the_best_score_wherever_it_sits_in_the_list() {
        assert!(matches!(
            decide(&[0.10, 0.20, 0.91, 0.05]),
            Verdict::Linked { who: 2, .. }
        ));
        // And the runner-up is the second best, not the next one along.
        assert_eq!(decide(&[0.88, 0.10, 0.86]), Verdict::Nobody);
    }

    #[test]
    fn a_negative_score_never_wins_anything() {
        // What an incomparable profile scores. It must not be able to be linked
        // just because it is the only candidate.
        assert_eq!(decide(&[-1.0]), Verdict::Nobody);
        assert!(matches!(
            decide(&[-1.0, 0.9]),
            Verdict::Linked { who: 1, .. }
        ));
    }

    #[test]
    fn one_fingerprint_is_held_to_a_higher_bar_than_a_whole_cluster() {
        // Clears TAU_LINK, nowhere near TAU_STRONG: good enough to name a
        // cluster, not good enough to take a fingerprint out of the clustering.
        assert!(matches!(decide(&[0.70, 0.1]), Verdict::Linked { .. }));
        assert!(decide_strong(&[0.70, 0.1]).is_none());
        assert_eq!(decide_strong(&[0.85, 0.1]), Some((0, 0.85)));
        // The margin applies here too.
        assert!(decide_strong(&[0.85, 0.80]).is_none());
        assert!(decide_strong(&[]).is_none());
    }

    /// The bars against the numbers `examples/people_fixture` actually measured
    /// on 2026-08-21. If a change moves a bar past one of these, it moves it past
    /// something that was observed rather than assumed.
    #[test]
    fn the_bars_still_do_what_the_fixtures_measured() {
        // Samantha's cluster against Samantha's profile, in a meeting with two
        // strangers in it.
        assert!(matches!(
            decide(&[0.932, 0.145]),
            Verdict::Linked { who: 0, .. }
        ));
        // The closest any stranger came at the cluster level — with a healthy
        // margin, which is why only a bar can refuse it.
        assert_eq!(decide(&[0.444, 0.150]), Verdict::Nobody);
        // The other strangers, refused by both rules at once.
        assert_eq!(decide(&[0.293, 0.284]), Verdict::Nobody);
        assert_eq!(decide(&[0.188, 0.109]), Verdict::Nobody);
        assert_eq!(decide(&[0.094, 0.060]), Verdict::Nobody);

        // Per fingerprint: the weakest true one still pre-assigns, the closest
        // stranger one still does not.
        assert!(decide_strong(&[0.814, 0.141]).is_some());
        assert!(decide_strong(&[0.506, 0.100]).is_none());

        // The ordering the whole module's reasoning depends on, and the
        // clearance over the closest stranger ever measured, at both levels.
        const {
            assert!(
                TAU_SUGGEST < TAU_LINK,
                "asking must be easier than claiming"
            );
            assert!(
                TAU_LINK < TAU_STRONG,
                "one fingerprint must be held to a higher bar than a cluster"
            );
            assert!(MARGIN > 0.0 && MARGIN < TAU_SUGGEST);
            assert!(TAU_SUGGEST - 0.444 > 0.05, "too near the nearest stranger");
            assert!(TAU_STRONG - 0.506 > 0.2);
        }
    }

    // -----------------------------------------------------------------------
    // Guiding: recognising a voice before anything is counted
    // -----------------------------------------------------------------------

    #[test]
    fn only_unmistakable_fingerprints_are_handed_to_a_known_person() {
        let known = vec![person(0, "Marco"), person(1, "Luca")];
        let prints = [
            print_scoring(&[0.95, 0.05]), // unmistakably Marco
            print_scoring(&[0.60, 0.05]), // Marco-ish: leave it to the clustering
            print_scoring(&[0.80, 0.75]), // like both of them, so neither
            print_scoring(&[0.05, 0.90]), // Luca
        ];
        let refs: Vec<&[f32]> = prints.iter().map(|p| p.as_slice()).collect();
        assert_eq!(
            pre_assign(&refs, &known),
            vec![Some(0), None, None, Some(1)]
        );
    }

    #[test]
    fn with_nobody_enrolled_nothing_is_pre_assigned() {
        let prints = [print_scoring(&[0.99])];
        let refs: Vec<&[f32]> = prints.iter().map(|p| p.as_slice()).collect();
        assert_eq!(pre_assign(&refs, &[]), vec![None]);
    }

    // -----------------------------------------------------------------------
    // Curation: outliers
    // -----------------------------------------------------------------------

    /// The accident this rule exists for: a confirmation on the wrong row. Ten
    /// samples of one voice and one of somebody else — the one that goes has to be
    /// the intruder, and nothing else may go with it.
    #[test]
    fn a_wrong_voice_among_ten_right_ones_is_the_one_that_gets_dropped() {
        let mut samples: Vec<SampleFacts> = (0..10)
            .map(|i| facts(&format!("right-{i}"), sample_near(0, i), Channel::Mic, i))
            .collect();
        samples.push(facts("wrong", sample_near(3, 1), Channel::Mic, 10));

        let dropped = find_outliers(&samples);
        assert_eq!(dropped, vec!["wrong".to_string()], "{dropped:?}");
    }

    #[test]
    fn a_profile_that_agrees_with_itself_loses_nothing() {
        let samples: Vec<SampleFacts> = (0..12)
            .map(|i| facts(&format!("s{i}"), sample_near(0, i), Channel::Mic, i))
            .collect();
        assert!(find_outliers(&samples).is_empty());
    }

    #[test]
    fn a_confirmation_on_the_wrong_row_loses_all_of_its_samples() {
        // Eight of the right voice and four of another: dropping one changes
        // every other sample's leave-one-out centroid, so the rule has to keep
        // going. It stops at eight, where "the others" stops meaning anything.
        let mut samples: Vec<SampleFacts> = (0..8)
            .map(|i| facts(&format!("right-{i}"), sample_near(0, i), Channel::Mic, i))
            .collect();
        for i in 0..4 {
            samples.push(facts(
                &format!("wrong-{i}"),
                sample_near(3, i),
                Channel::Mic,
                8 + i,
            ));
        }
        let dropped = find_outliers(&samples);
        assert_eq!(dropped.len(), 4, "{dropped:?}");
        assert!(
            dropped.iter().all(|id| id.starts_with("wrong")),
            "{dropped:?}"
        );
    }

    #[test]
    fn with_too_few_samples_nothing_is_called_an_outlier() {
        // Five samples, one of them somebody else. There is no "the others" to be
        // far from yet, and throwing away the odd one out would as likely lose the
        // sample recorded in a different room.
        let mut samples: Vec<SampleFacts> = (0..4)
            .map(|i| facts(&format!("right-{i}"), sample_near(0, i), Channel::Mic, i))
            .collect();
        samples.push(facts("wrong", sample_near(3, 1), Channel::Mic, 4));
        assert!(find_outliers(&samples).is_empty());
        assert!(samples.len() < OUTLIER_MIN_SAMPLES);
    }

    // -----------------------------------------------------------------------
    // Curation: the cap
    // -----------------------------------------------------------------------

    #[test]
    fn near_duplicates_are_what_the_cap_takes_first() {
        // Twenty-six samples of one voice: twenty-four of them nearly identical,
        // two of them genuinely different moments. Coming down to twenty-four has
        // to cost the duplicates.
        let mut samples: Vec<SampleFacts> = (0..24)
            .map(|i| facts(&format!("dup-{i}"), sample_near(0, 0), Channel::Mic, i))
            .collect();
        samples.push(facts("wide-a", another_moment(1), Channel::Mic, 24));
        samples.push(facts("wide-b", another_moment(2), Channel::Mic, 25));

        let evicted = choose_evictions(&samples, MAX_SAMPLES);
        assert_eq!(evicted.len(), 2);
        assert!(
            evicted.iter().all(|id| id.starts_with("dup")),
            "the spread was thrown away instead of the duplicates: {evicted:?}"
        );
    }

    #[test]
    fn between_two_interchangeable_samples_the_older_one_goes() {
        // Two samples that add exactly the same thing to the profile. Recency is
        // the only thing left to choose on, and it chooses the newer.
        let samples = vec![
            facts("new", sample_near(0, 0), Channel::Mic, 0),
            facts("old", sample_near(0, 0), Channel::Mic, 1),
        ];
        assert_eq!(choose_evictions(&samples, 1), vec!["old".to_string()]);
    }

    #[test]
    fn the_only_clip_of_a_condition_is_never_evicted() {
        // Nine samples off the microphone and one off the call. The call one is
        // the least like the others *and* the only one of its kind; losing it
        // would make the profile better in a room by making it useless on a call.
        let mut samples: Vec<SampleFacts> = (0..9)
            .map(|i| facts(&format!("mic-{i}"), sample_near(0, i), Channel::Mic, i))
            .collect();
        samples.push(facts("call", sample_near(0, 0), Channel::System, 9));

        let evicted = choose_evictions(&samples, 4);
        assert_eq!(evicted.len(), 6);
        assert!(!evicted.contains(&"call".to_string()), "{evicted:?}");
    }

    #[test]
    fn a_profile_under_the_cap_is_left_alone() {
        let samples: Vec<SampleFacts> = (0..3)
            .map(|i| facts(&format!("s{i}"), sample_near(0, i), Channel::Mic, i))
            .collect();
        assert!(choose_evictions(&samples, MAX_SAMPLES).is_empty());
        assert!(choose_evictions(&[], MAX_SAMPLES).is_empty());
    }

    // -----------------------------------------------------------------------
    // Leftovers of a voice that was already recognised
    // -----------------------------------------------------------------------

    /// The regression `examples/people_probe` caught on a real meeting: the
    /// weaker half of a recognised person's speech must go back to them, not
    /// become an extra speaker.
    #[test]
    fn more_of_a_recognised_voice_goes_back_to_that_voice() {
        let people = vec![person(0, "Marco"), person(1, "Ada")];
        let groups = vec![0]; // only Marco was recognised before the counting

        let absorbed = absorb_leftovers(
            &[
                print_scoring(&[0.90, 0.05]), // unmistakably more of Marco
                print_scoring(&[0.55, 0.05]), // only a question — stays its own voice
                print_scoring(&[0.20, 0.10]), // nobody
                print_scoring(&[0.05, 0.95]), // Ada, who has no group of her own
                Vec::new(),                   // nothing fingerprintable
            ],
            &people,
            &groups,
        );
        assert_eq!(
            absorbed,
            vec![Some(0), None, None, None, None],
            "only a claim at the linking bar, and only into a group that exists"
        );
    }

    /// Two people apart by less than the margin is the case a bar alone cannot
    /// answer: merging that speech into either track would be a guess.
    #[test]
    fn a_leftover_that_looks_like_two_people_goes_back_to_neither() {
        let people = vec![person(0, "Marco"), person(1, "Ada")];
        let absorbed = absorb_leftovers(&[print_scoring(&[0.70, 0.65])], &people, &[0, 1]);
        assert_eq!(absorbed, vec![None], "{MARGIN} of daylight or nothing");
    }

    /// Nobody enrolled, nothing recognised: the guided path has to be exactly the
    /// old path, cluster for cluster.
    #[test]
    fn with_nobody_recognised_nothing_is_absorbed() {
        assert_eq!(
            absorb_leftovers(&[print_scoring(&[0.99])], &[person(0, "Marco")], &[]),
            vec![None]
        );
        assert!(absorb_leftovers(&[], &[], &[]).is_empty());
    }

    // -----------------------------------------------------------------------
    // Curation, end to end over a database
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn curating_comes_down_to_the_cap_and_rewrites_the_centroid() {
        let db = db().await;
        let mut samples: Vec<(Vec<f32>, Channel)> =
            (0..28).map(|_| (sample_near(0, 0), Channel::Mic)).collect();
        samples[0] = (sample_near(0, 3), Channel::System);
        let person_id = person_with_samples(&db, "Marco", &samples, "old-network").await;

        let curated = curate(&db, &person_id, "this-network").await.unwrap();
        assert_eq!(curated.kept, MAX_SAMPLES);
        assert_eq!(curated.evicted, 28 - MAX_SAMPLES);
        assert_eq!(curated.outliers, 0);
        assert_eq!(
            repo::list_person_samples(&db, &person_id)
                .await
                .unwrap()
                .len(),
            MAX_SAMPLES
        );

        // The profile now describes what is left, and says which network it
        // belongs to.
        let profile = repo::list_person_profiles(&db)
            .await
            .unwrap()
            .into_iter()
            .find(|p| p.person_id == person_id)
            .expect("a profile");
        assert_eq!(profile.sample_count as usize, MAX_SAMPLES);
        assert_eq!(profile.embedder_asset_id, "this-network");
        assert!(similarity(&profile.centroid, &sample_near(0, 0)) > 0.9);
        // Both conditions survived, because one of them only just did.
        let conditions: Vec<Channel> = repo::list_person_samples(&db, &person_id)
            .await
            .unwrap()
            .into_iter()
            .map(|s| s.condition)
            .collect();
        assert!(conditions.contains(&Channel::System), "{conditions:?}");
    }

    #[tokio::test]
    async fn curating_a_profile_with_a_mislabel_in_it_drops_the_mislabel() {
        let db = db().await;
        let mut samples: Vec<(Vec<f32>, Channel)> =
            (0..10).map(|i| (sample_near(0, i), Channel::Mic)).collect();
        samples.push((sample_near(3, 1), Channel::Mic));
        let person_id = person_with_samples(&db, "Marco", &samples, "n").await;

        let curated = curate(&db, &person_id, "n").await.unwrap();
        assert_eq!(curated.outliers, 1);
        assert_eq!(curated.kept, 10);

        // And the centroid is the right voice again, not a blend of two people.
        let profile = repo::list_person_profiles(&db)
            .await
            .unwrap()
            .into_iter()
            .find(|p| p.person_id == person_id)
            .unwrap();
        assert!(similarity(&profile.centroid, &sample_near(0, 0)) > 0.95);
        assert!(similarity(&profile.centroid, &sample_near(3, 1)) < 0.1);
    }

    #[tokio::test]
    async fn a_person_with_nothing_usable_left_stops_being_matched() {
        let db = db().await;
        let person_id = person_with_samples(&db, "Nobody", &[], "n").await;
        let curated = curate(&db, &person_id, "n").await.unwrap();
        assert_eq!(curated.kept, 0);
        assert!(
            repo::list_person_profiles(&db).await.unwrap().is_empty(),
            "an empty profile must not be left behind to match nothing"
        );
        // The person is still there, and honestly reported as needing a refresh.
        let people = repo::list_person_infos(&db, "n").await.unwrap();
        assert_eq!(people.len(), 1);
        assert!(people[0].needs_refresh);
        assert_eq!(people[0].sample_count, 0);
    }

    // -----------------------------------------------------------------------
    // A profile computed by another network
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn a_profile_from_another_network_is_skipped_rather_than_compared() {
        let db = db().await;
        person_with_samples(&db, "Marco", &[(sample_near(0, 1), Channel::Mic)], "old").await;
        person_with_samples(&db, "Luca", &[(sample_near(1, 1), Channel::Mic)], "new").await;

        let enrolment = enrolled(&db, "new").await.unwrap();
        assert_eq!(enrolment.people.len(), 1);
        assert_eq!(enrolment.people[0].name, "Luca");
        assert_eq!(enrolment.stale, 1);

        // And the person says so, in a field the UI can be calm about.
        let people = repo::list_person_infos(&db, "new").await.unwrap();
        let marco = people.iter().find(|p| p.name == "Marco").unwrap();
        assert!(marco.needs_refresh);
        assert!(
            !people
                .iter()
                .find(|p| p.name == "Luca")
                .unwrap()
                .needs_refresh
        );
    }

    #[tokio::test]
    async fn nothing_to_re_embed_with_leaves_the_profile_exactly_as_it_was() {
        let db = db().await;
        let person_id =
            person_with_samples(&db, "Marco", &[(sample_near(0, 1), Channel::Mic)], "old").await;
        let before = repo::list_person_profiles(&db).await.unwrap();

        let outcome = refresh_profiles(&db, Path::new("/nonexistent/fingerprints.onnx")).await;
        assert!(
            matches!(outcome, Err(DiarizeError::NotInstalled)),
            "{outcome:?}"
        );

        let after = repo::list_person_profiles(&db).await.unwrap();
        assert_eq!(after.len(), before.len());
        assert_eq!(after[0].embedder_asset_id, "old");
        assert_eq!(
            repo::list_person_samples(&db, &person_id)
                .await
                .unwrap()
                .len(),
            1,
            "a failed refresh must not cost anybody their samples"
        );
    }

    #[tokio::test]
    async fn a_profile_already_on_this_network_is_not_touched() {
        let db = db().await;
        // A file the models table has never heard of is tagged by its own name,
        // which is still stable and still changes when the model does.
        let path = Path::new("/nonexistent/fingerprints.onnx");
        let tag = embedder_tag(&db, path).await;
        assert_eq!(tag, "fingerprints.onnx");

        person_with_samples(&db, "Marco", &[(sample_near(0, 1), Channel::Mic)], &tag).await;
        // No model is loaded, because there is nothing to do — which is the point:
        // a refresh costs nothing on the overwhelmingly common day.
        assert_eq!(refresh_profiles(&db, path).await.unwrap(), 0);
    }

    /// The full refresh, through the real network: every kept clip fingerprinted
    /// again and the profile put back in service. Ignored by default because it
    /// needs the downloaded asset, which CI does not have.
    #[tokio::test]
    #[ignore = "needs the downloaded speaker assets"]
    async fn a_refresh_re_embeds_every_clip_and_puts_the_profile_back_in_service() {
        let db = db().await;
        crate::asr::models::ensure_catalogued(&db).await.unwrap();
        let embedder = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join("Library/Application Support/Echo/speech")
            .join("wespeaker-en-voxceleb-resnet34-lm.onnx");
        assert!(embedder.exists(), "{}", embedder.display());

        // A profile whose numbers are nonsense and whose clips are real audio:
        // exactly the state a change of network leaves behind.
        let person_id = repo::create_person(&db, "Marco").await.unwrap().id;
        let tone: Vec<f32> = (0..16_000 * 4)
            .map(|i| ((i as f32) * 0.05).sin() * 0.4)
            .collect();
        let wav = sample::encode_wav_16k_mono(&tone).unwrap();
        for _ in 0..2 {
            repo::insert_person_sample(
                &db,
                &repo::NewPersonSample {
                    person_id: &person_id,
                    embedding: &[0.0; 4],
                    clip: &wav,
                    condition: Channel::Mic,
                    source_meeting_id: None,
                    t_start_ms: 0,
                    t_end_ms: 4_000,
                },
            )
            .await
            .unwrap();
        }
        repo::upsert_person_profile(&db, &person_id, &[0.0; 4], 2, "a-different-network")
            .await
            .unwrap();

        let tag = embedder_tag(&db, &embedder).await;
        assert_eq!(enrolled(&db, &tag).await.unwrap().stale, 1);

        assert_eq!(refresh_profiles(&db, &embedder).await.unwrap(), 1);

        let enrolment = enrolled(&db, &tag).await.unwrap();
        assert_eq!(enrolment.stale, 0);
        assert_eq!(enrolment.people.len(), 1);
        assert_eq!(enrolment.people[0].centroid.len(), 256);
        // Every sample's numbers came from the current network too.
        for s in repo::list_person_samples(&db, &person_id).await.unwrap() {
            assert_eq!(s.embedding.len(), 256);
            assert!(similarity(&s.embedding, &enrolment.people[0].centroid) > 0.9);
        }
    }

    // -----------------------------------------------------------------------
    // Names, links, and what deleting a person does
    // -----------------------------------------------------------------------

    #[test]
    fn echos_own_labels_may_be_replaced_by_a_real_name_but_a_typed_one_may_not() {
        assert!(is_default_name("Speaker 1"));
        assert!(is_default_name("Speaker 12"));
        assert!(is_default_name("You"));
        assert!(!is_default_name("Marco"));
        assert!(!is_default_name("Speaker 1 (Marco)"));
        assert!(!is_default_name("Speaker 99"));
    }

    /// A meeting with one speaker who says something, so a link has a row to land
    /// on. No audio, so nothing can be learned from it — which is the case this
    /// helper is for.
    async fn meeting_with_a_speaker(db: &Db, name: &str) -> (Id, Id) {
        let meeting = repo::create_meeting(db, "Team sync", "/tmp/echo-test", None)
            .await
            .unwrap();
        let speaker = repo::upsert_speaker(db, &meeting.id, "speaker-01", name, false)
            .await
            .unwrap();
        repo::insert_segments(
            db,
            &[SegmentDraft {
                meeting_id: meeting.id.clone(),
                t_start_ms: 0,
                t_end_ms: 20_000,
                channel: Channel::System,
                speaker_id: Some(speaker.id.clone()),
                text: "hello".into(),
                revision: 1,
                is_final: true,
                ..Default::default()
            }],
        )
        .await
        .unwrap();
        repo::set_meeting_status(db, &meeting.id, MeetingStatus::Complete)
            .await
            .unwrap();
        (meeting.id, speaker.id)
    }

    // -----------------------------------------------------------------------
    // Two names, one voice
    // -----------------------------------------------------------------------

    /// The case the feature exists for: the same voice enrolled twice, once off
    /// a call and once out of the room. Both sets of samples survive and the
    /// profile that matches against them describes both.
    #[tokio::test]
    async fn merging_two_names_makes_one_voice_out_of_both_sets_of_samples() {
        let db = db().await;
        let keep = person_with_samples(
            &db,
            "Marco",
            &[
                (sample_near(0, 1), Channel::Mic),
                (sample_near(0, 2), Channel::Mic),
            ],
            "wespeaker",
        )
        .await;
        let dup = person_with_samples(
            &db,
            "Marco (call)",
            &[(sample_near(0, 3), Channel::System)],
            "wespeaker",
        )
        .await;

        let done = merge(&db, &keep, &dup).await.unwrap();
        assert!(done.merged);
        assert_eq!(done.samples_moved, 1);
        assert_eq!(done.voice, VoiceAfterMerge::Recomputed { kept: 3 });

        let samples = repo::list_person_samples(&db, &keep).await.unwrap();
        assert_eq!(samples.len(), 3, "nothing was thrown away");
        assert!(
            samples.iter().any(|s| s.condition == Channel::System),
            "the voice heard down the call is in there too"
        );
        let profile = repo::person_profile(&db, &keep).await.unwrap().unwrap();
        assert_eq!(profile.sample_count, 3);
        assert_eq!(profile.embedder_asset_id, "wespeaker");

        // One person, under the name that was kept.
        let people = list_people(&db).await.unwrap();
        assert_eq!(people.len(), 1);
        assert_eq!(people[0].name, "Marco");
        assert_eq!(people[0].sample_count, 3);
    }

    /// Numbers from two different networks describe geometries neither one has
    /// been in, so they are never averaged. The profile is dropped, the person
    /// is honestly reported as needing a refresh, and every clip is still there
    /// for the refresh to re-fingerprint.
    #[tokio::test]
    async fn merging_voices_measured_by_different_networks_asks_for_a_refresh() {
        let db = db().await;
        let keep = person_with_samples(
            &db,
            "Marco",
            &[(sample_near(0, 1), Channel::Mic)],
            "wespeaker",
        )
        .await;
        let dup = person_with_samples(
            &db,
            "Marco (call)",
            &[(sample_near(0, 2), Channel::System)],
            "an-older-network",
        )
        .await;

        let done = merge(&db, &keep, &dup).await.unwrap();
        assert_eq!(done.voice, VoiceAfterMerge::NeedsRefresh);
        assert!(repo::person_profile(&db, &keep).await.unwrap().is_none());
        assert_eq!(
            repo::list_person_samples(&db, &keep).await.unwrap().len(),
            2,
            "both clips survive; only the centroid was given up on"
        );
        let people = list_people(&db).await.unwrap();
        assert_eq!(people.len(), 1);
        assert_eq!(people[0].sample_count, 2);
        assert!(
            people[0].needs_refresh,
            "a person with no profile is honestly reported as needing one"
        );
    }

    /// The other half of that sentence. Settings says Echo is refreshing, and
    /// this is what makes it so: the survivor now has no profile row, and a
    /// person with no profile counts towards the gate the pass checks before it
    /// counts anybody. Without this the merge dropped a centroid that nothing
    /// would ever rebuild — the person quietly stopped being recognised while
    /// the screen promised a refresh was under way.
    #[tokio::test]
    async fn a_merged_voice_with_no_profile_is_what_makes_the_next_pass_rebuild_it() {
        let db = db().await;
        let keep = person_with_samples(
            &db,
            "Marco",
            &[(sample_near(0, 1), Channel::Mic)],
            "wespeaker",
        )
        .await;
        let dup = person_with_samples(
            &db,
            "Marco (call)",
            &[(sample_near(0, 2), Channel::System)],
            "an-older-network",
        )
        .await;
        assert_eq!(
            merge(&db, &keep, &dup).await.unwrap().voice,
            VoiceAfterMerge::NeedsRefresh
        );

        // Not matched at the moment — and counted, which is the gate.
        let enrolment = enrolled(&db, "wespeaker").await.unwrap();
        assert!(enrolment.people.is_empty(), "no centroid to match against");
        assert_eq!(enrolment.stale, 1, "the pass must be told there is work");

        // And the work is real: the refresh gets past its own early return and
        // only stops for the missing network, so with one installed this voice
        // is rebuilt from the clips both names kept.
        let outcome = refresh_profiles(&db, Path::new("/nonexistent/fingerprints.onnx")).await;
        assert!(
            matches!(outcome, Err(DiarizeError::NotInstalled)),
            "{outcome:?}"
        );
    }

    /// The gate is deliberately stricter than the refresh it opens: it counts
    /// only people a refresh could put back. A voice with nothing left to
    /// re-fingerprint would otherwise hold it open on every pass forever, and
    /// each pass would load the network to finish no work.
    #[tokio::test]
    async fn a_voice_with_nothing_left_to_re_fingerprint_does_not_hold_the_gate_open() {
        let db = db().await;
        let person_id = person_with_samples(&db, "Nobody", &[], "wespeaker").await;
        curate(&db, &person_id, "wespeaker").await.unwrap();
        assert!(repo::person_profile(&db, &person_id)
            .await
            .unwrap()
            .is_none());

        // Still honestly reported to the person, and still not work to schedule.
        let people = repo::list_person_infos(&db, "wespeaker").await.unwrap();
        assert!(people[0].needs_refresh);
        assert_eq!(enrolled(&db, "wespeaker").await.unwrap().stale, 0);
    }

    /// A name with no voice behind it has nothing to give, so the surviving
    /// profile is left exactly as it was rather than rebuilt for no reason.
    #[tokio::test]
    async fn merging_a_name_with_no_samples_leaves_the_voice_untouched() {
        let db = db().await;
        let keep = person_with_samples(
            &db,
            "Marco",
            &[(sample_near(0, 1), Channel::Mic)],
            "wespeaker",
        )
        .await;
        let before = repo::person_profile(&db, &keep).await.unwrap().unwrap();
        let dup = repo::create_person(&db, "Marco?").await.unwrap();

        let done = merge(&db, &keep, &dup.id).await.unwrap();
        assert!(done.merged);
        assert_eq!(done.voice, VoiceAfterMerge::Unchanged);
        let after = repo::person_profile(&db, &keep).await.unwrap().unwrap();
        assert_eq!(after.centroid, before.centroid);
        assert_eq!(after.updated_at, before.updated_at);
    }

    /// Clicking it twice is not a second merge, and a window that has been open
    /// since before a merge still means the person that name is now.
    #[tokio::test]
    async fn merging_the_same_two_people_again_changes_nothing() {
        let db = db().await;
        let keep = person_with_samples(
            &db,
            "Marco",
            &[(sample_near(0, 1), Channel::Mic)],
            "wespeaker",
        )
        .await;
        let dup = person_with_samples(
            &db,
            "Marco (call)",
            &[(sample_near(0, 2), Channel::System)],
            "wespeaker",
        )
        .await;
        merge(&db, &keep, &dup).await.unwrap();

        let again = merge(&db, &keep, &dup).await.unwrap();
        assert!(!again.merged);
        let backwards = merge(&db, &dup, &keep).await.unwrap();
        assert!(!backwards.merged);
        assert_eq!(
            repo::list_person_samples(&db, &keep).await.unwrap().len(),
            2
        );
        assert_eq!(list_people(&db).await.unwrap().len(), 1);
    }

    /// A link asked for under the old name lands on the person that name is
    /// now, rather than failing or re-creating the duplicate.
    #[tokio::test]
    async fn linking_by_a_merged_name_links_to_whoever_it_is_now() {
        let db = db().await;
        let (meeting_id, speaker_id) = meeting_with_a_speaker(&db, "Speaker 1").await;
        let keep = person_with_samples(
            &db,
            "Marco",
            &[(sample_near(0, 1), Channel::Mic)],
            "wespeaker",
        )
        .await;
        let dup = person_with_samples(
            &db,
            "Marco (call)",
            &[(sample_near(0, 2), Channel::Mic)],
            "wespeaker",
        )
        .await;
        merge(&db, &keep, &dup).await.unwrap();

        link(&db, &meeting_id, &speaker_id, Some(&dup))
            .await
            .unwrap();

        let speaker = repo::get_speaker(&db, &speaker_id).await.unwrap().unwrap();
        assert_eq!(speaker.person_id.as_deref(), Some(keep.as_str()));
        assert_eq!(
            speaker.display_name, "Marco",
            "the name that got copied on is the surviving one"
        );
    }

    #[tokio::test]
    async fn deleting_a_person_releases_the_link_and_leaves_the_transcript_alone() {
        let db = db().await;
        let (meeting_id, speaker_id) = meeting_with_a_speaker(&db, "Speaker 1").await;
        let person_id =
            person_with_samples(&db, "Marco", &[(sample_near(0, 1), Channel::Mic)], "n").await;

        // Link by hand, the way `link` does, without needing audio to learn from.
        repo::set_speaker_person(&db, &speaker_id, Some(&person_id))
            .await
            .unwrap();
        repo::rename_speaker(&db, &speaker_id, "Marco")
            .await
            .unwrap();

        repo::delete_person(&db, &person_id).await.unwrap();

        let speaker = repo::get_speaker(&db, &speaker_id).await.unwrap().unwrap();
        assert_eq!(speaker.person_id, None, "the link has to go");
        assert_eq!(
            speaker.display_name, "Marco",
            "the name already read in this meeting stays as plain text"
        );
        assert!(repo::list_person_samples(&db, &person_id)
            .await
            .unwrap()
            .is_empty());
        assert!(repo::list_person_profiles(&db).await.unwrap().is_empty());
        assert!(repo::get_person(&db, &person_id).await.unwrap().is_none());
        // And the meeting itself is untouched.
        assert!(repo::get_meeting(&db, &meeting_id).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn deleting_the_meeting_a_sample_came_from_does_not_cost_the_sample() {
        let db = db().await;
        let meeting = repo::create_meeting(&db, "Once", "/tmp/echo-test", None)
            .await
            .unwrap();
        let person = repo::create_person(&db, "Marco").await.unwrap();
        let wav = clip();
        repo::insert_person_sample(
            &db,
            &repo::NewPersonSample {
                person_id: &person.id,
                embedding: &sample_near(0, 1),
                clip: &wav,
                condition: Channel::Mic,
                source_meeting_id: Some(&meeting.id),
                t_start_ms: 0,
                t_end_ms: 6_000,
            },
        )
        .await
        .unwrap();

        repo::delete_meeting(&db, &meeting.id).await.unwrap();

        let samples = repo::list_person_samples(&db, &person.id).await.unwrap();
        assert_eq!(samples.len(), 1, "a profile is not derived from a meeting");
        assert_eq!(samples[0].source_meeting_id, None);
        assert!(!samples[0].embedding.is_empty());
    }

    #[tokio::test]
    async fn unlinking_releases_the_row_without_renaming_it_back() {
        let db = db().await;
        let (meeting_id, speaker_id) = meeting_with_a_speaker(&db, "Speaker 1").await;
        let person_id =
            person_with_samples(&db, "Marco", &[(sample_near(0, 1), Channel::Mic)], "n").await;
        repo::set_speaker_person(&db, &speaker_id, Some(&person_id))
            .await
            .unwrap();
        repo::rename_speaker(&db, &speaker_id, "Marco")
            .await
            .unwrap();

        link(&db, &meeting_id, &speaker_id, None).await.unwrap();

        let speaker = repo::get_speaker(&db, &speaker_id).await.unwrap().unwrap();
        assert_eq!(speaker.person_id, None);
        assert_eq!(speaker.display_name, "Marco");
    }

    #[tokio::test]
    async fn a_speaker_from_another_meeting_cannot_be_linked() {
        let db = db().await;
        let (_, speaker_id) = meeting_with_a_speaker(&db, "Speaker 1").await;
        let (other_meeting, _) = meeting_with_a_speaker(&db, "Speaker 1").await;
        let person_id =
            person_with_samples(&db, "Marco", &[(sample_near(0, 1), Channel::Mic)], "n").await;
        assert!(link(&db, &other_meeting, &speaker_id, Some(&person_id))
            .await
            .is_err());
    }

    #[tokio::test]
    async fn linking_to_a_person_who_does_not_exist_is_refused() {
        let db = db().await;
        let (meeting_id, speaker_id) = meeting_with_a_speaker(&db, "Speaker 1").await;
        assert!(link(&db, &meeting_id, &speaker_id, Some(&repo::new_id()))
            .await
            .is_err());
        let speaker = repo::get_speaker(&db, &speaker_id).await.unwrap().unwrap();
        assert_eq!(speaker.person_id, None);
    }

    #[tokio::test]
    async fn remembering_a_voice_there_is_no_clip_of_creates_nobody() {
        // The meeting has words and no recording, so there is nothing to keep.
        // A person with a name and no voice behind it is not what was asked for.
        let db = db().await;
        let (meeting_id, speaker_id) = meeting_with_a_speaker(&db, "Speaker 1").await;
        let outcome = enroll(&db, &meeting_id, &speaker_id, "Marco").await;
        assert!(outcome.is_err(), "{outcome:?}");
        assert!(repo::list_people(&db).await.unwrap().is_empty());
        // And the speaker is left exactly as it was.
        let speaker = repo::get_speaker(&db, &speaker_id).await.unwrap().unwrap();
        assert_eq!(speaker.display_name, "Speaker 1");
        assert_eq!(speaker.person_id, None);
    }

    #[tokio::test]
    async fn the_freshest_clip_is_what_listening_to_a_person_plays() {
        let db = db().await;
        let person_id =
            person_with_samples(&db, "Marco", &[(sample_near(0, 1), Channel::Mic)], "n").await;
        let encoded = person_sample_audio(&db, &person_id).await.unwrap();
        assert!(
            encoded.starts_with("UklGR"),
            "has to be a WAV: {encoded:.8}"
        );

        let empty = repo::create_person(&db, "Never heard").await.unwrap();
        assert!(matches!(
            person_sample_audio(&db, &empty.id).await,
            Err(DiarizeError::NoVoiceSample)
        ));
    }

    #[test]
    fn a_kept_clip_reads_back_as_the_audio_it_was() {
        // The round trip a refresh depends on: what went in comes out, so a
        // fingerprint computed from it means something.
        let original: Vec<f32> = (0..1_000).map(|i| ((i as f32) * 0.1).sin() * 0.5).collect();
        let wav = sample::encode_wav_16k_mono(&original).unwrap();
        let back = sample::decode_wav_16k_mono(&wav).unwrap();
        assert_eq!(back.len(), original.len());
        for (a, b) in original.iter().zip(&back) {
            assert!((a - b).abs() < 1e-3, "{a} became {b}");
        }
        assert!(sample::decode_wav_16k_mono(b"not a wav at all").is_err());
    }

    /// "Remember this voice", all the way through: real audio on disk, the real
    /// network, four clips kept, a profile that matches them. Ignored by default
    /// because it needs the downloaded asset.
    #[tokio::test]
    #[ignore = "needs the downloaded speaker assets"]
    async fn remembering_a_voice_keeps_a_few_clips_of_it_and_a_profile_that_matches() {
        let db = db().await;
        crate::asr::models::ensure_catalogued(&db).await.unwrap();
        // Point the models table at the installed file, the way a real install
        // does, so `embedder_tag` reads the catalog id rather than a file name.
        let embedder = std::path::PathBuf::from(std::env::var("HOME").unwrap())
            .join("Library/Application Support/Echo/speech")
            .join("wespeaker-en-voxceleb-resnet34-lm.onnx");
        assert!(embedder.exists(), "{}", embedder.display());
        repo::set_model_installed(
            &db,
            crate::asr::catalog::ids::EMBEDDER,
            true,
            Some(&embedder.to_string_lossy()),
        )
        .await
        .unwrap();
        repo::set_model_installed(
            &db,
            crate::asr::catalog::ids::SEGMENTER,
            true,
            Some(
                &embedder
                    .with_file_name("pyannote-segmentation-3.0.onnx")
                    .to_string_lossy(),
            ),
        )
        .await
        .unwrap();

        // Ninety seconds of recording with four separate turns in it.
        let dir = tempfile::tempdir().unwrap();
        let meeting = repo::create_meeting(&db, "Weekly sync", dir.path().to_str().unwrap(), None)
            .await
            .unwrap();
        let tone: Vec<f32> = (0..(16_000 * 90))
            .map(|i| {
                let t = i as f32 / 16_000.0;
                (t * 220.0 * std::f32::consts::TAU).sin() * 0.3
                    + (t * 330.0 * std::f32::consts::TAU).sin() * 0.2
            })
            .collect();
        let path = dir.path().join("mic-000000.wav");
        crate::audio::writer::write_wav_16k_mono(&path, &tone).unwrap();
        let chunk = repo::insert_chunk(
            &db,
            &meeting.id,
            Channel::Mic,
            0,
            path.to_str().unwrap(),
            0,
            90_000,
        )
        .await
        .unwrap();
        repo::commit_chunk(&db, &chunk, 90_000).await.unwrap();

        let speaker = repo::upsert_speaker(&db, &meeting.id, "speaker-01", "Speaker 1", false)
            .await
            .unwrap();
        let turns = [
            (10_000, 20_000),
            (30_000, 40_000),
            (50_000, 60_000),
            (70_000, 80_000),
        ];
        for (from, to) in turns {
            repo::insert_segments(
                &db,
                &[SegmentDraft {
                    meeting_id: meeting.id.clone(),
                    t_start_ms: from,
                    t_end_ms: to,
                    channel: Channel::Mic,
                    speaker_id: Some(speaker.id.clone()),
                    text: "hello".into(),
                    revision: 1,
                    is_final: true,
                    ..Default::default()
                }],
            )
            .await
            .unwrap();
        }
        repo::set_meeting_status(&db, &meeting.id, MeetingStatus::Complete)
            .await
            .unwrap();

        let info = enroll(&db, &meeting.id, &speaker.id, "Marco")
            .await
            .unwrap();
        assert_eq!(info.name, "Marco");
        assert_eq!(
            info.sample_count as usize, SAMPLES_PER_CONFIRMATION,
            "one confirmation keeps up to four moments"
        );
        assert!(!info.needs_refresh, "enrolled with the network in use");
        assert!(info.last_heard_at.is_some(), "the meeting it came from");

        // The row now says who it is, in the person's own name.
        let row = repo::get_speaker(&db, &speaker.id).await.unwrap().unwrap();
        assert_eq!(row.person_id.as_deref(), Some(info.id.as_str()));
        assert_eq!(row.display_name, "Marco");

        // And the profile really does match the samples it was built from.
        let tag = embedder_tag(&db, &embedder).await;
        assert_eq!(tag, crate::asr::catalog::ids::EMBEDDER);
        let enrolment = enrolled(&db, &tag).await.unwrap();
        assert_eq!(enrolment.people.len(), 1);
        assert_eq!(enrolment.stale, 0);
        for sample in repo::list_person_samples(&db, &info.id).await.unwrap() {
            let score = similarity(&sample.embedding, &enrolment.people[0].centroid);
            assert!(score > TAU_LINK, "a kept sample scored {score}");
        }
        // The clips are real audio, playable and re-embeddable.
        for (_, clip) in repo::person_sample_clips(&db, &info.id).await.unwrap() {
            let back = sample::decode_wav_16k_mono(&clip).unwrap();
            assert_eq!(back.len(), sample::CLIP_MS as usize * 16);
        }

        // Confirming again adds more of the same voice without doubling anything
        // it already knows.
        link(&db, &meeting.id, &speaker.id, Some(&info.id))
            .await
            .unwrap();
        let after = list_people(&db).await.unwrap();
        assert!(
            after[0].sample_count as usize <= MAX_SAMPLES,
            "{} samples",
            after[0].sample_count
        );
        assert!(after[0].sample_count >= info.sample_count);
    }

    // -----------------------------------------------------------------------
    // Recurring unnamed voices
    // -----------------------------------------------------------------------

    /// A finished meeting with one fingerprinted, unnamed speaker in it.
    async fn meeting_with_a_voice(
        db: &Db,
        title: &str,
        centroid: &[f32],
        speaking_ms: i64,
    ) -> (Id, Id) {
        let meeting = repo::create_meeting(db, title, "/tmp/echo-test", None)
            .await
            .unwrap();
        let speaker = repo::upsert_speaker(db, &meeting.id, "speaker-01", "Speaker 1", false)
            .await
            .unwrap();
        repo::insert_segments(
            db,
            &[SegmentDraft {
                meeting_id: meeting.id.clone(),
                t_start_ms: 0,
                t_end_ms: speaking_ms,
                channel: Channel::System,
                speaker_id: Some(speaker.id.clone()),
                text: "hello".into(),
                revision: 1,
                is_final: true,
                ..Default::default()
            }],
        )
        .await
        .unwrap();
        repo::recompute_speaking_time(db, &meeting.id)
            .await
            .unwrap();
        repo::set_speaker_centroid(db, &speaker.id, Some(centroid))
            .await
            .unwrap();
        (meeting.id, speaker.id)
    }

    #[tokio::test]
    async fn a_voice_in_three_meetings_is_offered_and_one_in_two_is_not() {
        let db = db().await;
        for i in 0..3 {
            meeting_with_a_voice(
                &db,
                &format!("Sync {i}"),
                &sample_near(0, i),
                60_000 + i as i64,
            )
            .await;
        }
        for i in 0..2 {
            meeting_with_a_voice(&db, &format!("Other {i}"), &sample_near(3, i), 10_000).await;
        }

        let suggested = suggested_people(&db).await.unwrap();
        assert_eq!(suggested.len(), 1, "{suggested:?}");
        assert_eq!(suggested[0].appearances, 3);
        assert_eq!(suggested[0].speaking_ms, 60_000 + 60_001 + 60_002);
        // The representative is the meeting this voice said the most in, which is
        // the best clip to play and the best material to enroll from.
        assert_eq!(suggested[0].meeting_title, "Sync 2");
        assert_eq!(suggested[0].id, suggested[0].speaker_id);
    }

    #[tokio::test]
    async fn two_speakers_in_one_meeting_are_never_the_same_recurring_voice() {
        let db = db().await;
        // Three meetings, each with two speakers whose voice prints are alike.
        // The pass already decided they are different people; this list is not
        // the place to overrule it.
        for i in 0..3 {
            let meeting = repo::create_meeting(&db, &format!("Sync {i}"), "/tmp/echo-test", None)
                .await
                .unwrap();
            for k in 0..2 {
                let speaker = repo::upsert_speaker(
                    &db,
                    &meeting.id,
                    &format!("speaker-0{}", k + 1),
                    "Speaker 1",
                    false,
                )
                .await
                .unwrap();
                repo::set_speaker_centroid(&db, &speaker.id, Some(&sample_near(0, i * 2 + k)))
                    .await
                    .unwrap();
            }
        }
        let suggested = suggested_people(&db).await.unwrap();
        // Two groups of three, not one group of six.
        assert_eq!(suggested.len(), 2, "{suggested:?}");
        assert!(suggested.iter().all(|s| s.appearances == 3));
    }

    #[tokio::test]
    async fn a_voice_that_is_already_a_known_person_is_not_offered_again() {
        let db = db().await;
        for i in 0..3 {
            meeting_with_a_voice(&db, &format!("Sync {i}"), &sample_near(0, i), 60_000).await;
        }
        assert_eq!(suggested_people(&db).await.unwrap().len(), 1);

        person_with_samples(&db, "Marco", &[(sample_near(0, 1), Channel::Mic)], "n").await;
        assert!(
            suggested_people(&db).await.unwrap().is_empty(),
            "offering to remember somebody already remembered makes two of them"
        );
    }

    #[tokio::test]
    async fn a_voice_already_linked_to_a_person_is_not_a_recurring_stranger() {
        let db = db().await;
        let person_id =
            person_with_samples(&db, "Marco", &[(sample_near(4, 1), Channel::Mic)], "n").await;
        for i in 0..3 {
            let (_, speaker_id) =
                meeting_with_a_voice(&db, &format!("Sync {i}"), &sample_near(0, i), 60_000).await;
            repo::set_speaker_person(&db, &speaker_id, Some(&person_id))
                .await
                .unwrap();
        }
        assert!(suggested_people(&db).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_deleted_meeting_stops_counting_as_an_appearance() {
        let db = db().await;
        let mut ids = Vec::new();
        for i in 0..3 {
            let (meeting_id, _) =
                meeting_with_a_voice(&db, &format!("Sync {i}"), &sample_near(0, i), 60_000).await;
            ids.push(meeting_id);
        }
        assert_eq!(suggested_people(&db).await.unwrap().len(), 1);
        repo::soft_delete_meeting(&db, &ids[0]).await.unwrap();
        assert!(suggested_people(&db).await.unwrap().is_empty());
    }
}
