//! Voice fingerprints: a fixed-length vector per stretch of one person talking.
//!
//! The network is a WeSpeaker ResNet34 export. It does not take audio — it takes
//! the log-mel features [`super::features`] produces, and returns one vector per
//! utterance whose *direction* identifies the voice. Two recordings of the same
//! person point roughly the same way; two different people do not. That is the
//! whole basis for [`super::cluster`], and the calibrated distance threshold
//! there belongs to this exact network.
//!
//! Fingerprints are only computed from audio where **one** person is talking
//! ([`select_speech`] carves out everyone else), because a fingerprint of two
//! mixed voices resembles neither.
//!
//! The session lives for one pass and is dropped at the end (mantra 1).

use std::path::Path;

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

use super::features::{self, NUM_MEL_BINS, SAMPLE_RATE};
use super::timeline::{self, Span};
use super::DiarizeError;

/// Shortest stretch of one voice worth fingerprinting. Below a second the
/// vector is dominated by whatever phonemes happened to be in it rather than by
/// the person, and a bad fingerprint is worse than none: it pulls a cluster
/// centroid off the voice it represents.
pub const MIN_EMBED_MS: i64 = 1_000;

/// Longest stretch fed to the network at once. More audio stops improving the
/// fingerprint well before this, and the cost is linear.
pub const MAX_EMBED_MS: i64 = 8_000;

/// Collect the audio for a set of spans out of one window's samples.
///
/// `samples` starts at `window_start_ms`. Spans outside the window are ignored,
/// spans that straddle its edge are clipped, and the result stops at
/// [`MAX_EMBED_MS`]. Pure, so the span arithmetic is testable without a model.
pub fn select_speech(
    samples: &[f32],
    window_start_ms: i64,
    spans: &[Span],
    max_ms: i64,
) -> Vec<f32> {
    let per_ms = SAMPLE_RATE as i64 / 1_000;
    let cap = (max_ms.max(0) * per_ms) as usize;
    let mut out: Vec<f32> = Vec::new();
    for &(start, end) in spans {
        if out.len() >= cap {
            break;
        }
        let from = ((start - window_start_ms).max(0) * per_ms) as usize;
        let to = ((end - window_start_ms).max(0) * per_ms) as usize;
        if from >= samples.len() || to <= from {
            continue;
        }
        let to = to.min(samples.len()).min(from + (cap - out.len()));
        out.extend_from_slice(&samples[from..to]);
    }
    out
}

/// Is there enough of one voice here to be worth a fingerprint?
pub fn worth_embedding(spans: &[Span]) -> bool {
    timeline::total_ms(spans) >= MIN_EMBED_MS
}

/// The fingerprint network, loaded for one offline pass.
pub struct Embedder {
    session: Session,
    input_name: String,
    /// Rank the model declares for its input: with or without a batch axis.
    input_rank: usize,
    /// Length of the vectors this network produces, learned from its first run.
    dim: Option<usize>,
}

impl std::fmt::Debug for Embedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Embedder")
            .field("input", &self.input_name)
            .field("dim", &self.dim)
            .finish()
    }
}

impl Embedder {
    /// CPU only, same reasoning as the segmenter.
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
            .ok_or_else(|| DiarizeError::Load("fingerprint model has no input".into()))?;
        let input_name = input.name().to_string();
        let input_rank = super::segmentation::declared_rank(input.dtype()).unwrap_or(3);

        Ok(Self {
            session,
            input_name,
            input_rank,
            dim: None,
        })
    }

    /// Length of the fingerprints, once one has been computed.
    pub fn dim(&self) -> Option<usize> {
        self.dim
    }

    /// Fingerprint one stretch of 16 kHz mono audio holding a single voice.
    ///
    /// Returns `Ok(None)` when the audio is too short to produce features —
    /// that is an ordinary outcome, not a failure.
    pub fn embed(&mut self, samples: &[f32]) -> Result<Option<Vec<f32>>, DiarizeError> {
        if samples.len() < features::min_samples() {
            return Ok(None);
        }
        let feats = features::fbank(samples);
        if feats.is_empty() {
            return Ok(None);
        }

        let (frames, bins) = (feats.frames as i64, NUM_MEL_BINS as i64);
        let shape = match self.input_rank {
            0..=2 => vec![frames, bins],
            _ => vec![1, frames, bins],
        };
        let tensor = Tensor::from_array((shape, feats.data))
            .map_err(|e| DiarizeError::Failed(e.to_string()))?;
        let outputs = self
            .session
            .run(ort::inputs![self.input_name.as_str() => tensor])
            .map_err(|e| DiarizeError::Failed(e.to_string()))?;
        let (out_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| DiarizeError::Failed(e.to_string()))?;

        let dims: Vec<i64> = out_shape.iter().copied().collect();
        let dim = match dims.as_slice() {
            [1, d] => *d as usize,
            [d] => *d as usize,
            other => {
                return Err(DiarizeError::Failed(format!(
                    "fingerprint output had shape {other:?}, expected [1, dim]"
                )))
            }
        };
        if dim == 0 || data.len() < dim {
            return Err(DiarizeError::Failed(
                "fingerprint output was empty".to_string(),
            ));
        }

        let mut v = data[..dim].to_vec();
        super::cluster::l2_normalize(&mut v);
        self.dim = Some(dim);
        Ok(Some(v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32).collect()
    }

    #[test]
    fn selected_audio_is_exactly_the_spans_inside_the_window() {
        // 1 s of audio starting at 10 000 ms; 16 samples per ms.
        let samples = ramp(16_000);
        let picked = select_speech(&samples, 10_000, &[(10_100, 10_200)], 8_000);
        assert_eq!(picked.len(), 100 * 16);
        assert_eq!(picked[0], 100.0 * 16.0);
    }

    #[test]
    fn several_spans_are_concatenated_in_order() {
        let samples = ramp(16_000);
        let picked = select_speech(&samples, 0, &[(0, 10), (500, 520)], 8_000);
        assert_eq!(picked.len(), (10 + 20) * 16);
        assert_eq!(picked[0], 0.0);
        assert_eq!(picked[10 * 16], 500.0 * 16.0);
    }

    #[test]
    fn spans_outside_the_window_are_ignored_and_edges_are_clipped() {
        let samples = ramp(16_000);
        // Entirely past the end.
        assert!(select_speech(&samples, 0, &[(2_000, 3_000)], 8_000).is_empty());
        // Straddling the end: only the part we have.
        let clipped = select_speech(&samples, 0, &[(900, 1_200)], 8_000);
        assert_eq!(clipped.len(), 100 * 16);
        // Before the window start.
        let before = select_speech(&samples, 5_000, &[(4_000, 4_500)], 8_000);
        assert!(before.is_empty());
    }

    #[test]
    fn the_cap_stops_collecting_rather_than_truncating_afterwards() {
        let samples = ramp(16_000);
        let picked = select_speech(&samples, 0, &[(0, 400), (400, 800)], 500);
        assert_eq!(picked.len(), 500 * 16);
    }

    #[test]
    fn a_zero_cap_collects_nothing() {
        let samples = ramp(16_000);
        assert!(select_speech(&samples, 0, &[(0, 900)], 0).is_empty());
    }

    #[test]
    fn only_a_second_of_one_voice_is_worth_a_fingerprint() {
        assert!(!worth_embedding(&[]));
        assert!(!worth_embedding(&[(0, 999)]));
        assert!(worth_embedding(&[(0, 1_000)]));
        // Several short pieces of the same voice add up.
        assert!(worth_embedding(&[(0, 400), (1_000, 1_400), (2_000, 2_300)]));
    }

    #[test]
    fn the_minimum_is_long_enough_to_produce_features() {
        let min_samples = MIN_EMBED_MS * i64::from(SAMPLE_RATE) / 1_000;
        assert!(features::num_frames(min_samples as usize) > 50);
        const { assert!(MAX_EMBED_MS > MIN_EMBED_MS) };
    }
}
