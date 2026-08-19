//! Log-mel filterbank features, byte-compatible with Kaldi's `compute-fbank-feats`.
//!
//! The speaker-fingerprint network was trained on features produced by
//! `torchaudio.compliance.kaldi.fbank(..., num_mel_bins=80, frame_length=25,
//! frame_shift=10, dither=0.0, sample_frequency=16000)` followed by
//! mean-normalisation over time. Getting these details wrong does not fail
//! loudly, it just quietly makes every voice look the same, so the constants
//! below are deliberate and each one matches the training recipe:
//!
//! * waveform scaled by `1 << 15`, because Kaldi works in 16-bit units. The
//!   scale is a constant offset in the log domain and is removed again by
//!   mean-normalisation, but it keeps quiet frames off the log floor.
//! * DC removed per frame, then pre-emphasis 0.97, then a Povey window.
//! * 400-sample frame zero-padded to 512 for the transform.
//! * 80 triangular mel filters between 20 Hz and Nyquist, log of the floored
//!   energy.
//! * finally each dimension has its mean over time subtracted (CMN).
//!
//! No allocation-per-frame surprises and no external FFT crate: the transform
//! is a 512-point radix-2 done in place.

/// Everything downstream of capture is 16 kHz mono.
pub const SAMPLE_RATE: u32 = 16_000;
/// Feature dimension the fingerprint network expects.
pub const NUM_MEL_BINS: usize = 80;
/// 25 ms analysis window.
pub const FRAME_LENGTH: usize = 400;
/// 10 ms hop.
pub const FRAME_SHIFT: usize = 160;
/// Next power of two above [`FRAME_LENGTH`].
const FFT_SIZE: usize = 512;
/// Kaldi's `num_fft_bins`: half the padded window, Nyquist excluded.
const NUM_FFT_BINS: usize = FFT_SIZE / 2;
const LOW_FREQ_HZ: f32 = 20.0;
const PREEMPHASIS: f32 = 0.97;
/// Kaldi floors filterbank energies at `numeric_limits<float>::epsilon()`.
const ENERGY_FLOOR: f32 = f32::EPSILON;
/// Kaldi treats the waveform as 16-bit units.
const WAVEFORM_SCALE: f32 = 32_768.0;

/// A `frames × NUM_MEL_BINS` matrix in row-major order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Fbank {
    pub frames: usize,
    pub data: Vec<f32>,
}

impl Fbank {
    pub fn row(&self, frame: usize) -> &[f32] {
        &self.data[frame * NUM_MEL_BINS..(frame + 1) * NUM_MEL_BINS]
    }

    pub fn is_empty(&self) -> bool {
        self.frames == 0
    }
}

/// How many frames Kaldi produces for `n` samples with `snip_edges=true`.
pub fn num_frames(n: usize) -> usize {
    if n < FRAME_LENGTH {
        0
    } else {
        1 + (n - FRAME_LENGTH) / FRAME_SHIFT
    }
}

/// Shortest audio that yields at least one feature frame.
pub fn min_samples() -> usize {
    FRAME_LENGTH
}

fn hz_to_mel(hz: f32) -> f32 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

/// Povey window: a Hann raised to 0.85, what Kaldi uses by default.
fn povey_window() -> Vec<f32> {
    let denom = (FRAME_LENGTH - 1) as f32;
    (0..FRAME_LENGTH)
        .map(|i| {
            let hann = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / denom).cos();
            hann.powf(0.85)
        })
        .collect()
}

/// Triangular mel filters as `(first_bin, weights)` pairs, so the inner loop
/// touches only the bins a filter actually covers.
fn mel_banks() -> Vec<(usize, Vec<f32>)> {
    let nyquist = SAMPLE_RATE as f32 / 2.0;
    let bin_width = SAMPLE_RATE as f32 / FFT_SIZE as f32;
    let mel_low = hz_to_mel(LOW_FREQ_HZ);
    let mel_high = hz_to_mel(nyquist);
    let delta = (mel_high - mel_low) / (NUM_MEL_BINS + 1) as f32;

    let mel_of_bin: Vec<f32> = (0..NUM_FFT_BINS)
        .map(|k| hz_to_mel(bin_width * k as f32))
        .collect();

    let mut banks = Vec::with_capacity(NUM_MEL_BINS);
    for bin in 0..NUM_MEL_BINS {
        let left = mel_low + bin as f32 * delta;
        let center = left + delta;
        let right = left + 2.0 * delta;

        let mut first = None;
        let mut weights: Vec<f32> = Vec::new();
        for (k, &mel) in mel_of_bin.iter().enumerate() {
            if mel <= left || mel >= right {
                continue;
            }
            let w = if mel <= center {
                (mel - left) / delta
            } else {
                (right - mel) / delta
            };
            if first.is_none() {
                first = Some(k);
            }
            weights.push(w);
        }
        banks.push((first.unwrap_or(0), weights));
    }
    banks
}

/// In-place radix-2 decimation-in-time FFT. `re`/`im` must be the same
/// power-of-two length. Twiddles accumulate in f64 so a 512-point transform
/// does not drift.
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());
    debug_assert_eq!(im.len(), n);

    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let angle = -2.0 * std::f64::consts::PI / len as f64;
        let (wr, wi) = (angle.cos(), angle.sin());
        for base in (0..n).step_by(len) {
            let mut cr = 1.0f64;
            let mut ci = 0.0f64;
            for k in 0..half {
                let (ur, ui) = (re[base + k], im[base + k]);
                let (br, bi) = (re[base + k + half], im[base + k + half]);
                let vr = br * cr - bi * ci;
                let vi = br * ci + bi * cr;
                re[base + k] = ur + vr;
                im[base + k] = ui + vi;
                re[base + k + half] = ur - vr;
                im[base + k + half] = ui - vi;
                let next_cr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = next_cr;
            }
        }
        len <<= 1;
    }
}

/// Compute the feature matrix for one stretch of 16 kHz mono audio.
///
/// Returns an empty matrix when the audio is shorter than one frame; callers
/// must treat that as "not enough voice to fingerprint", not as an error.
pub fn fbank(samples: &[f32]) -> Fbank {
    let mut out = fbank_raw(samples);
    mean_normalize(&mut out);
    out
}

/// [`fbank`] without the mean-normalisation step. Only useful for tests that
/// care about absolute energies.
pub fn fbank_raw(samples: &[f32]) -> Fbank {
    let frames = num_frames(samples.len());
    if frames == 0 {
        return Fbank::default();
    }

    let window = povey_window();
    let banks = mel_banks();

    let mut data = vec![0.0f32; frames * NUM_MEL_BINS];
    let mut re = vec![0.0f64; FFT_SIZE];
    let mut im = vec![0.0f64; FFT_SIZE];
    let mut buf = vec![0.0f32; FRAME_LENGTH];

    for f in 0..frames {
        let start = f * FRAME_SHIFT;
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = samples[start + i] * WAVEFORM_SCALE;
        }

        // Remove DC offset.
        let mean = buf.iter().sum::<f32>() / FRAME_LENGTH as f32;
        for x in buf.iter_mut() {
            *x -= mean;
        }

        // Pre-emphasis, walking backwards so each sample sees the original
        // value of its predecessor. Kaldi treats x[-1] as x[0].
        for i in (1..FRAME_LENGTH).rev() {
            buf[i] -= PREEMPHASIS * buf[i - 1];
        }
        buf[0] -= PREEMPHASIS * buf[0];

        re[..FRAME_LENGTH]
            .iter_mut()
            .zip(buf.iter().zip(window.iter()))
            .for_each(|(dst, (x, w))| *dst = f64::from(x * w));
        re[FRAME_LENGTH..].fill(0.0);
        im.fill(0.0);

        fft(&mut re, &mut im);

        // Power spectrum, Kaldi's default for filterbanks.
        let power: Vec<f32> = (0..NUM_FFT_BINS)
            .map(|k| (re[k] * re[k] + im[k] * im[k]) as f32)
            .collect();

        let row = &mut data[f * NUM_MEL_BINS..(f + 1) * NUM_MEL_BINS];
        for (bin, (first, weights)) in banks.iter().enumerate() {
            let mut energy = 0.0f32;
            for (i, w) in weights.iter().enumerate() {
                energy += w * power[first + i];
            }
            row[bin] = energy.max(ENERGY_FLOOR).ln();
        }
    }

    Fbank { frames, data }
}

/// Subtract each dimension's mean over time (CMN). This is what makes the
/// fingerprint indifferent to microphone and room, and it also cancels the
/// constant introduced by [`WAVEFORM_SCALE`].
pub fn mean_normalize(f: &mut Fbank) {
    if f.frames == 0 {
        return;
    }
    let mut means = vec![0.0f32; NUM_MEL_BINS];
    for frame in f.data.chunks_exact(NUM_MEL_BINS) {
        for (mean, value) in means.iter_mut().zip(frame) {
            *mean += value;
        }
    }
    let n = f.frames as f32;
    for m in means.iter_mut() {
        *m /= n;
    }
    for frame in f.data.chunks_exact_mut(NUM_MEL_BINS) {
        for (value, mean) in frame.iter_mut().zip(&means) {
            *value -= mean;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(hz: f32, secs: f32) -> Vec<f32> {
        let n = (SAMPLE_RATE as f32 * secs) as usize;
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * hz * i as f32 / SAMPLE_RATE as f32).sin() * 0.5)
            .collect()
    }

    #[test]
    fn frame_count_follows_the_kaldi_snip_edges_rule() {
        assert_eq!(num_frames(0), 0);
        assert_eq!(num_frames(399), 0);
        assert_eq!(num_frames(400), 1);
        assert_eq!(num_frames(559), 1);
        assert_eq!(num_frames(560), 2);
        // One second of audio is 98 frames with a 25/10 ms window.
        assert_eq!(num_frames(16_000), 98);
    }

    #[test]
    fn audio_shorter_than_a_frame_yields_nothing_rather_than_an_error() {
        let f = fbank(&[0.0; 100]);
        assert!(f.is_empty());
        assert!(f.data.is_empty());
    }

    #[test]
    fn the_matrix_has_one_row_per_frame_and_eighty_columns() {
        let f = fbank(&tone(440.0, 1.0));
        assert_eq!(f.frames, 98);
        assert_eq!(f.data.len(), 98 * NUM_MEL_BINS);
        assert!(f.data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn mean_normalisation_leaves_every_dimension_centred() {
        let f = fbank(&tone(300.0, 0.6));
        for bin in 0..NUM_MEL_BINS {
            let mean: f32 = (0..f.frames).map(|fr| f.row(fr)[bin]).sum::<f32>() / f.frames as f32;
            assert!(mean.abs() < 1e-2, "bin {bin} mean {mean}");
        }
    }

    #[test]
    fn a_pure_tone_peaks_in_the_mel_bin_that_covers_it() {
        fn peak_bin(hz: f32) -> usize {
            let f = fbank_raw(&tone(hz, 0.5));
            let mid = f.row(f.frames / 2);
            mid.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .expect("80 bins")
        }
        // The expected bin is the mel-scale position of the tone.
        fn expected_bin(hz: f32) -> usize {
            let mel_low = hz_to_mel(LOW_FREQ_HZ);
            let delta = (hz_to_mel(SAMPLE_RATE as f32 / 2.0) - mel_low) / (NUM_MEL_BINS + 1) as f32;
            ((hz_to_mel(hz) - mel_low) / delta).round() as usize
        }
        for hz in [300.0f32, 1_000.0, 3_000.0] {
            let (got, want) = (peak_bin(hz), expected_bin(hz));
            assert!(
                got.abs_diff(want) <= 1,
                "{hz} Hz peaked at bin {got}, expected about {want}"
            );
        }
        assert!(peak_bin(300.0) < peak_bin(3_000.0));
    }

    #[test]
    fn mel_filters_are_ordered_and_non_empty() {
        let banks = mel_banks();
        assert_eq!(banks.len(), NUM_MEL_BINS);
        let mut last_first = 0usize;
        for (i, (first, weights)) in banks.iter().enumerate() {
            assert!(!weights.is_empty(), "mel bin {i} covers no transform bin");
            assert!(*first >= last_first, "mel bin {i} runs backwards");
            assert!(first + weights.len() <= NUM_FFT_BINS);
            assert!(weights.iter().all(|w| *w > 0.0 && *w <= 1.000_01));
            last_first = *first;
        }
    }

    #[test]
    fn the_transform_matches_a_direct_dft_on_a_small_case() {
        let n = 8usize;
        let input: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).sin()).collect();
        let mut re = input.clone();
        let mut im = vec![0.0; n];
        fft(&mut re, &mut im);
        for k in 0..n {
            let (mut dr, mut di) = (0.0f64, 0.0f64);
            for (t, x) in input.iter().enumerate() {
                let ang = -2.0 * std::f64::consts::PI * (k * t) as f64 / n as f64;
                dr += x * ang.cos();
                di += x * ang.sin();
            }
            assert!((re[k] - dr).abs() < 1e-9, "bin {k}: {} vs {dr}", re[k]);
            assert!((im[k] - di).abs() < 1e-9, "bin {k}: {} vs {di}", im[k]);
        }
    }

    #[test]
    fn silence_is_finite_because_energies_are_floored() {
        let f = fbank(&vec![0.0; 8_000]);
        assert!(f.frames > 0);
        assert!(f.data.iter().all(|x| x.is_finite()));
    }
}
