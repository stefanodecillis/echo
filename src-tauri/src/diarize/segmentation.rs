//! Sliding-window speaker segmentation and the decoding of its output.
//!
//! The network is a pyannote-style end-to-end segmenter. It takes a fixed
//! window of raw audio and returns, for every ~17 ms frame, a distribution over
//! the **powerset** of local speakers rather than one probability per speaker.
//! With three local speakers and at most two talking at once that is seven
//! classes:
//!
//! ```text
//! 0: nobody      3: C            6: B+C
//! 1: A           4: A+B
//! 2: B           5: A+C
//! ```
//!
//! That encoding is what makes the model overlap-aware: "A and B together" is
//! its own class, so two people talking is a thing the model predicts directly
//! instead of something we infer from two independent scores.
//!
//! Labels are **local to the window**: speaker A in one window has nothing to
//! do with speaker A in the next. Stitching windows together is
//! [`crate::diarize::cluster::align_permutation`]'s job, and the final identity
//! comes from clustering fingerprints.
//!
//! Everything above [`Segmenter`] is pure arithmetic and unit-tested. The
//! session itself is created for one pass and dropped at the end (mantra 1).

use std::path::Path;

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

use super::features::SAMPLE_RATE;
use super::timeline::{self, Span};
use super::DiarizeError;

/// Window handed to the segmenter, in seconds. The model was trained on 10 s.
pub const WINDOW_MS: i64 = 10_000;
/// Hop between windows. Half the window: enough overlap to stitch labels
/// across the seam, cheap enough to keep the whole pass well under the 0.3
/// real-time budget from the M0-S4 gate (two inferences per 10 s of audio).
pub const STEP_MS: i64 = 5_000;

/// Samples in one window at 16 kHz.
pub const WINDOW_SAMPLES: usize = (WINDOW_MS as usize) * (SAMPLE_RATE as usize) / 1_000;

/// Activations shorter than this are noise, not a speech turn.
pub const MIN_SPEECH_MS: i64 = 200;
/// Gaps shorter than this inside one speaker's activity are breaths, not turns.
pub const MIN_GAP_MS: i64 = 200;

/// Windows whose peak amplitude is below this hold no voice worth running the
/// model over. Skipping them is most of the reason the pass is fast on a
/// meeting where one side is quiet.
pub const SILENCE_PEAK: f32 = 1e-4;

/// Per-frame activations for one window, already converted out of the powerset.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WindowActivations {
    pub num_frames: usize,
    pub num_speakers: usize,
    /// `num_frames × num_speakers`, the summed probability that this speaker is
    /// talking. Kept for confidence, not for the decision.
    pub soft: Vec<f32>,
    /// `num_frames × num_speakers`, the decision: the most likely powerset
    /// class, expanded to the speakers it contains.
    pub hard: Vec<bool>,
}

impl WindowActivations {
    pub fn soft_at(&self, frame: usize, speaker: usize) -> f32 {
        self.soft[frame * self.num_speakers + speaker]
    }

    pub fn hard_at(&self, frame: usize, speaker: usize) -> bool {
        self.hard[frame * self.num_speakers + speaker]
    }

    /// Mean activation over the frames where this speaker is on. 0.0 when the
    /// speaker never speaks in this window.
    pub fn confidence(&self, speaker: usize) -> f32 {
        let mut sum = 0.0f32;
        let mut n = 0usize;
        for f in 0..self.num_frames {
            if self.hard_at(f, speaker) {
                sum += self.soft_at(f, speaker);
                n += 1;
            }
        }
        if n == 0 {
            0.0
        } else {
            sum / n as f32
        }
    }
}

/// All subsets of `n` speakers of size `k`, in lexicographic order.
fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![Vec::new()];
    }
    if k > n {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        out.push(idx.clone());
        // Rightmost position that is not already at its maximum.
        let mut i = k;
        while i > 0 && idx[i - 1] == i - 1 + n - k {
            i -= 1;
        }
        if i == 0 {
            return out;
        }
        idx[i - 1] += 1;
        for j in i..k {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

/// Recover the powerset layout from the number of output classes.
///
/// The model does not tell us how many speakers it models; the class count
/// does. Ordering matches the training-time encoder: the empty set, then every
/// single speaker in order, then every pair, and so on — so class *i* here is
/// class *i* there.
///
/// Returns `None` when no (speakers, simultaneous) pair explains the count,
/// which means the file is not the segmenter we expect.
pub fn powerset_classes(num_classes: usize) -> Option<Vec<Vec<usize>>> {
    for speakers in 1..=8usize {
        let mut classes: Vec<Vec<usize>> = Vec::new();
        for size in 0..=speakers {
            classes.extend(combinations(speakers, size));
            if classes.len() == num_classes {
                return Some(classes);
            }
            if classes.len() > num_classes {
                break;
            }
        }
    }
    None
}

/// Softmax over the class axis of a `frames × classes` matrix.
pub fn softmax_rows(logits: &[f32], num_frames: usize, num_classes: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; num_frames * num_classes];
    for f in 0..num_frames {
        let row = &logits[f * num_classes..(f + 1) * num_classes];
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for (c, x) in row.iter().enumerate() {
            let e = (x - max).exp();
            out[f * num_classes + c] = e;
            sum += e;
        }
        if sum > 0.0 {
            for c in 0..num_classes {
                out[f * num_classes + c] /= sum;
            }
        }
    }
    out
}

/// Turn raw powerset logits into per-speaker activations.
///
/// `soft` sums the probability of every class containing the speaker, which is
/// the overlap-aware marginal. `hard` takes the single most likely class and
/// expands it, which is the decision the pipeline acts on — it can never claim
/// more simultaneous speakers than the model was trained to predict.
pub fn decode_powerset(
    logits: &[f32],
    num_frames: usize,
    num_classes: usize,
) -> Result<WindowActivations, DiarizeError> {
    if num_frames == 0 || num_classes == 0 || logits.len() != num_frames * num_classes {
        return Err(DiarizeError::Failed(format!(
            "segmentation output was {} values, expected {num_frames}×{num_classes}",
            logits.len()
        )));
    }
    let classes = powerset_classes(num_classes).ok_or_else(|| {
        DiarizeError::Load(format!(
            "segmentation model has {num_classes} output classes, which is not a speaker powerset"
        ))
    })?;
    let num_speakers = classes.iter().flatten().copied().max().map_or(0, |m| m + 1);
    if num_speakers == 0 {
        return Err(DiarizeError::Load(
            "segmentation model predicts no speakers".into(),
        ));
    }

    let probs = softmax_rows(logits, num_frames, num_classes);
    let mut soft = vec![0.0f32; num_frames * num_speakers];
    let mut hard = vec![false; num_frames * num_speakers];

    for f in 0..num_frames {
        let row = &probs[f * num_classes..(f + 1) * num_classes];
        for (c, p) in row.iter().enumerate() {
            for &s in &classes[c] {
                soft[f * num_speakers + s] += p;
            }
        }
        let best = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(c, _)| c)
            .unwrap_or(0);
        for &s in &classes[best] {
            hard[f * num_speakers + s] = true;
        }
    }

    Ok(WindowActivations {
        num_frames,
        num_speakers,
        soft,
        hard,
    })
}

/// Convert one window's frame decisions into speech turns on the meeting clock.
///
/// The frame duration comes from the output length rather than a constant, so a
/// segmenter with a different receptive field still lands on the right
/// timestamps. Short activations are dropped and short gaps closed, in that
/// order, because a single flipped frame is never a speech turn.
pub fn tracks_from_activations(
    act: &WindowActivations,
    window_start_ms: i64,
    window_ms: i64,
) -> Vec<Vec<Span>> {
    let frame_ms = window_ms as f64 / act.num_frames as f64;
    let mut out = Vec::with_capacity(act.num_speakers);

    for s in 0..act.num_speakers {
        let mut spans: Vec<Span> = Vec::new();
        let mut run_start: Option<usize> = None;
        for f in 0..act.num_frames {
            let on = act.hard_at(f, s);
            match (on, run_start) {
                (true, None) => run_start = Some(f),
                (false, Some(start)) => {
                    spans.push(frames_to_span(start, f, frame_ms, window_start_ms));
                    run_start = None;
                }
                _ => {}
            }
        }
        if let Some(start) = run_start {
            spans.push(frames_to_span(
                start,
                act.num_frames,
                frame_ms,
                window_start_ms,
            ));
        }
        timeline::merge_spans(&mut spans, MIN_GAP_MS);
        out.push(timeline::drop_short(spans, MIN_SPEECH_MS));
    }
    out
}

fn frames_to_span(from: usize, to: usize, frame_ms: f64, offset_ms: i64) -> Span {
    let start = offset_ms + (from as f64 * frame_ms).round() as i64;
    let end = offset_ms + (to as f64 * frame_ms).round() as i64;
    (start, end.max(start + 1))
}

/// Is there anything in this window worth running the model over?
pub fn has_signal(samples: &[f32]) -> bool {
    samples.iter().any(|x| x.abs() > SILENCE_PEAK)
}

/// How many axes a model declares for one of its tensors.
///
/// Two exports of the same network disagree about whether the batch and channel
/// axes are present, and feeding a rank-2 model a rank-3 tensor fails at run
/// time with nothing a user could act on. Reading the declaration instead means
/// either export works.
pub fn declared_rank(dtype: &ort::value::ValueType) -> Option<usize> {
    match dtype {
        ort::value::ValueType::Tensor { shape, .. } => Some(shape.len()),
        _ => None,
    }
}

/// The segmentation network, loaded for one offline pass.
///
/// Dropping it releases the session and its weights (mantra 1). Nothing here is
/// cached between passes on purpose: a diarization pass happens once per
/// meeting, and holding a session for the rest of the day to save a second of
/// load time is exactly what mantra 1 forbids.
pub struct Segmenter {
    session: Session,
    input_name: String,
    /// Rank the model declares for its input. Exports of the same network differ
    /// on whether the channel axis is there, so read it rather than assume it.
    input_rank: usize,
}

impl std::fmt::Debug for Segmenter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Segmenter")
            .field("input", &self.input_name)
            .finish()
    }
}

impl Segmenter {
    /// Load the segmenter. CPU only: there is no Vulkan execution provider for
    /// this runtime on Linux, so both platforms take the same path and the cost
    /// is honest (DESIGN §2, review finding 2).
    pub fn load(path: &Path, threads: usize) -> Result<Self, DiarizeError> {
        if !path.exists() {
            return Err(DiarizeError::NotInstalled);
        }
        let session = Session::builder()
            .map_err(|e| DiarizeError::Load(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| DiarizeError::Load(e.to_string()))?
            .with_intra_threads(threads.max(1))
            .map_err(|e| DiarizeError::Load(e.to_string()))?
            .commit_from_file(path)
            .map_err(|e| DiarizeError::Load(e.to_string()))?;

        let input = session
            .inputs()
            .first()
            .ok_or_else(|| DiarizeError::Load("segmentation model has no input".into()))?;
        let input_name = input.name().to_string();
        let input_rank = declared_rank(input.dtype()).unwrap_or(3);

        Ok(Self {
            session,
            input_name,
            input_rank,
        })
    }

    /// Run one window. `samples` is 16 kHz mono; shorter input is zero-padded
    /// and longer input truncated, so the caller never has to think about the
    /// tail of a meeting.
    pub fn run(&mut self, samples: &[f32]) -> Result<WindowActivations, DiarizeError> {
        let mut padded = vec![0.0f32; WINDOW_SAMPLES];
        let n = samples.len().min(WINDOW_SAMPLES);
        padded[..n].copy_from_slice(&samples[..n]);

        let samples = WINDOW_SAMPLES as i64;
        let shape = match self.input_rank {
            0 | 1 => vec![samples],
            2 => vec![1, samples],
            _ => vec![1, 1, samples],
        };
        let tensor =
            Tensor::from_array((shape, padded)).map_err(|e| DiarizeError::Failed(e.to_string()))?;
        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => tensor])
            .map_err(|e| DiarizeError::Failed(e.to_string()))?;

        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| DiarizeError::Failed(e.to_string()))?;

        // Expected [1, frames, classes]; tolerate a missing batch axis.
        let dims: Vec<i64> = shape.iter().copied().collect();
        let (num_frames, num_classes) = match dims.as_slice() {
            [1, frames, classes] => (*frames as usize, *classes as usize),
            [frames, classes] => (*frames as usize, *classes as usize),
            other => {
                return Err(DiarizeError::Failed(format!(
                    "segmentation output had shape {other:?}, expected [1, frames, classes]"
                )))
            }
        };
        decode_powerset(data, num_frames, num_classes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build logits that put all the mass on one class per frame.
    fn one_hot(frames: &[usize], num_classes: usize) -> Vec<f32> {
        let mut out = vec![-20.0f32; frames.len() * num_classes];
        for (f, &c) in frames.iter().enumerate() {
            out[f * num_classes + c] = 20.0;
        }
        out
    }

    #[test]
    fn seven_classes_is_three_speakers_two_at_a_time() {
        let classes = powerset_classes(7).expect("7 is a powerset");
        assert_eq!(
            classes,
            vec![
                vec![],
                vec![0],
                vec![1],
                vec![2],
                vec![0, 1],
                vec![0, 2],
                vec![1, 2],
            ]
        );
    }

    #[test]
    fn other_plausible_class_counts_decode_too() {
        // 1 speaker, on or off.
        assert_eq!(powerset_classes(2), Some(vec![vec![], vec![0]]));
        // 2 speakers, both may talk: {}, {0}, {1}, {0,1}.
        assert_eq!(
            powerset_classes(4),
            Some(vec![vec![], vec![0], vec![1], vec![0, 1]])
        );
        // 4 speakers, up to two at once: 1 + 4 + 6 = 11.
        let eleven = powerset_classes(11).expect("11 is a powerset");
        assert_eq!(eleven.len(), 11);
        assert_eq!(eleven.iter().flatten().copied().max(), Some(3));
    }

    #[test]
    fn a_class_count_that_is_not_a_powerset_is_rejected() {
        // Reachable counts are 1 + C(n,1) + … + C(n,k). 10, 12 and 13 are not
        // any such sum, so a file claiming them is not our segmenter.
        for bad in [10usize, 12, 13, 14] {
            assert_eq!(powerset_classes(bad), None, "{bad} should not decode");
        }
        let err = decode_powerset(&[0.0; 10], 1, 10).unwrap_err();
        assert!(matches!(err, DiarizeError::Load(_)), "{err:?}");
    }

    #[test]
    fn the_fewest_speakers_that_explains_the_count_wins() {
        // 7 classes is both "3 speakers, 2 at a time" and "6 speakers, one at a
        // time". The trained model is the former, and it is also the reading
        // that can represent overlap, so it must win.
        let classes = powerset_classes(7).unwrap();
        assert_eq!(classes.iter().flatten().copied().max(), Some(2));
        assert!(classes.iter().any(|c| c.len() == 2));
    }

    #[test]
    fn softmax_rows_sum_to_one_and_survive_large_logits() {
        let logits = vec![1000.0, 1001.0, 999.0, -1000.0];
        let p = softmax_rows(&logits, 1, 4);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "{sum}");
        assert!(p.iter().all(|x| x.is_finite()));
        assert!(p[1] > p[0] && p[0] > p[2] && p[2] > p[3]);
    }

    #[test]
    fn the_decision_expands_the_winning_class_to_its_speakers() {
        // Frame 0 nobody, frame 1 speaker A, frame 2 A and B together.
        let logits = one_hot(&[0, 1, 4], 7);
        let act = decode_powerset(&logits, 3, 7).unwrap();
        assert_eq!(act.num_speakers, 3);
        assert_eq!(
            (0..3).map(|s| act.hard_at(0, s)).collect::<Vec<_>>(),
            vec![false, false, false]
        );
        assert_eq!(
            (0..3).map(|s| act.hard_at(1, s)).collect::<Vec<_>>(),
            vec![true, false, false]
        );
        assert_eq!(
            (0..3).map(|s| act.hard_at(2, s)).collect::<Vec<_>>(),
            vec![true, true, false]
        );
    }

    #[test]
    fn the_soft_marginal_adds_up_every_class_the_speaker_is_in() {
        // Equal mass on "A alone" and "B alone": each has marginal 0.5.
        let mut logits = vec![-40.0f32; 7];
        logits[1] = 0.0;
        logits[2] = 0.0;
        let act = decode_powerset(&logits, 1, 7).unwrap();
        assert!(
            (act.soft_at(0, 0) - 0.5).abs() < 1e-3,
            "{}",
            act.soft_at(0, 0)
        );
        assert!((act.soft_at(0, 1) - 0.5).abs() < 1e-3);
        assert!(act.soft_at(0, 2) < 1e-3);
        // Overlap raises both marginals: "A+B" alone gives each of them 1.0.
        let mut both = vec![-40.0f32; 7];
        both[4] = 0.0;
        let act = decode_powerset(&both, 1, 7).unwrap();
        assert!(act.soft_at(0, 0) > 0.99 && act.soft_at(0, 1) > 0.99);
    }

    #[test]
    fn a_mismatched_output_length_is_an_error_not_a_panic() {
        let err = decode_powerset(&[0.0; 10], 3, 7).unwrap_err();
        assert!(matches!(err, DiarizeError::Failed(_)), "{err:?}");
    }

    #[test]
    fn turns_come_out_on_the_meeting_clock_with_short_blips_dropped() {
        // 100 frames over a 1000 ms window = 10 ms per frame.
        // A speaks frames 0..40 (400 ms); B has a 2-frame blip well clear of it.
        let mut frames = vec![0usize; 100];
        frames[..40].fill(1);
        frames[80] = 2;
        frames[81] = 2;
        let act = decode_powerset(&one_hot(&frames, 7), 100, 7).unwrap();
        let tracks = tracks_from_activations(&act, 30_000, 1_000);
        assert_eq!(tracks.len(), 3);
        // The 400 ms turn survives, shifted onto the meeting clock.
        assert_eq!(tracks[0], vec![(30_000, 30_400)]);
        // The 20 ms blip is below MIN_SPEECH_MS.
        assert!(tracks[1].is_empty(), "{:?}", tracks[1]);
        assert!(tracks[2].is_empty());
    }

    #[test]
    fn short_gaps_inside_one_speakers_turn_are_closed() {
        // A speaks, pauses for 100 ms (under MIN_GAP_MS), speaks again.
        let mut frames = vec![0usize; 100];
        frames[..30].fill(1);
        frames[40..80].fill(1);
        let act = decode_powerset(&one_hot(&frames, 7), 100, 7).unwrap();
        let tracks = tracks_from_activations(&act, 0, 1_000);
        assert_eq!(tracks[0], vec![(0, 800)]);
    }

    #[test]
    fn confidence_averages_only_the_frames_where_the_speaker_is_on() {
        let act = decode_powerset(&one_hot(&[0, 1, 1], 7), 3, 7).unwrap();
        assert!(act.confidence(0) > 0.9);
        assert_eq!(act.confidence(2), 0.0);
    }

    #[test]
    fn silence_detection_is_cheap_and_correct() {
        assert!(!has_signal(&[0.0; 100]));
        assert!(!has_signal(&[SILENCE_PEAK / 2.0; 100]));
        assert!(has_signal(&[0.0, 0.0, 0.2]));
    }

    #[test]
    fn the_window_and_hop_stay_within_the_real_time_budget() {
        // Half-overlap means two inferences per window of audio; anything
        // finer would break the RTF target from the M0-S4 gate.
        const { assert!(STEP_MS * 2 >= WINDOW_MS) };
        assert_eq!(WINDOW_SAMPLES, 160_000);
    }
}
