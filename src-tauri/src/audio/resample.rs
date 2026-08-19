//! Downmix to mono and resample to 16 kHz, once per channel.
//!
//! Everything downstream — the chunk writer, speech detection, the speech
//! engine — works at [`crate::audio::TARGET_SAMPLE_RATE`] mono. Devices deliver
//! whatever they feel like (44.1 kHz stereo, 48 kHz eight-channel interface),
//! so this is the one place that converts.
//!
//! This runs on a consumer thread, never inside an audio callback (DESIGN §2
//! DSP): it allocates and it does real arithmetic.
//!
//! `rubato`'s FFT resampler is used because the ratio is fixed and rational for
//! every real device rate, and because it band-limits properly — decimating
//! 48 kHz to 16 kHz without a filter would fold cymbals and sibilance down onto
//! the speech band and make the transcript worse.

use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};

use crate::audio::{AudioError, TARGET_SAMPLE_RATE};

/// Roughly how much audio one resampler call should cover. The real chunk is
/// rounded up to something the rate pair allows.
const TARGET_CHUNK_MS: usize = 20;

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.max(1)
}

/// Streaming downmix + resample for one capture channel.
///
/// Feed interleaved frames at the device's native rate and take 16 kHz mono
/// back. Leftover input is held until enough has arrived, so no sample is lost
/// and no silence is invented at a call boundary.
pub struct Resampler16k {
    input_rate: u32,
    input_channels: usize,
    /// `None` when the device already runs at 16 kHz: then this is a pure
    /// downmix and there is nothing to filter.
    inner: Option<Fft<f32>>,
    /// Mono input waiting for a full resampler chunk.
    pending: Vec<f32>,
    /// Scratch for one resampler call's output.
    scratch: Vec<f32>,
    frames_per_call: usize,
    max_out_frames: usize,
    /// The filter's start-up delay, trimmed off the front of the stream so a
    /// frame's timestamp still means what it says.
    delay_to_trim: usize,
    total_in: u64,
    total_out: u64,
}

impl std::fmt::Debug for Resampler16k {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Resampler16k")
            .field("input_rate", &self.input_rate)
            .field("input_channels", &self.input_channels)
            .field("resampling", &self.inner.is_some())
            .field("pending", &self.pending.len())
            .finish()
    }
}

impl Resampler16k {
    pub fn new(input_rate: u32, input_channels: u16) -> Result<Self, AudioError> {
        if input_rate == 0 || input_channels == 0 {
            return Err(AudioError::Backend(format!(
                "device reported an unusable format: {input_rate} Hz, {input_channels} channels"
            )));
        }
        let input_channels = usize::from(input_channels);
        if input_rate == TARGET_SAMPLE_RATE {
            return Ok(Self {
                input_rate,
                input_channels,
                inner: None,
                pending: Vec::new(),
                scratch: Vec::new(),
                frames_per_call: 0,
                max_out_frames: 0,
                delay_to_trim: 0,
                total_in: 0,
                total_out: 0,
            });
        }

        let rate_in = input_rate as usize;
        let rate_out = TARGET_SAMPLE_RATE as usize;
        // The FFT resampler can only work in whole multiples of this.
        let min_block = rate_in / gcd(rate_in, rate_out);
        let wanted = (rate_in * TARGET_CHUNK_MS / 1_000).max(1);
        let chunk = min_block * wanted.div_ceil(min_block.max(1)).max(1);

        let inner = Fft::<f32>::new(rate_in, rate_out, chunk, 1, FixedSync::Input)
            .map_err(|e| AudioError::Backend(format!("could not set up resampling: {e}")))?;
        let frames_per_call = inner.input_frames_next().max(1);
        let max_out_frames = inner.output_frames_max().max(1);
        let delay_to_trim = inner.output_delay();
        Ok(Self {
            input_rate,
            input_channels,
            inner: Some(inner),
            pending: Vec::with_capacity(frames_per_call * 2),
            scratch: vec![0.0; max_out_frames],
            frames_per_call,
            max_out_frames,
            delay_to_trim,
            total_in: 0,
            total_out: 0,
        })
    }

    pub fn input_rate(&self) -> u32 {
        self.input_rate
    }

    pub fn input_channels(&self) -> usize {
        self.input_channels
    }

    /// Convert interleaved native-rate audio to 16 kHz mono.
    ///
    /// Extra samples that do not fill a resampler chunk stay inside and come
    /// out of the next call.
    pub fn push_interleaved(&mut self, interleaved: &[f32]) -> Vec<f32> {
        let before = self.pending.len();
        self.downmix_into_pending(interleaved);
        self.total_in += (self.pending.len() - before) as u64;
        self.drain_pending()
    }

    /// Already-mono native-rate audio (what ScreenCaptureKit and PipeWire hand
    /// us after their own channel sum).
    pub fn push_mono(&mut self, mono: &[f32]) -> Vec<f32> {
        self.pending.extend_from_slice(mono);
        self.total_in += mono.len() as u64;
        self.drain_pending()
    }

    /// End of stream: pad the tail with silence so the last partial chunk comes
    /// out instead of being swallowed, then trim back to the length the input
    /// actually justifies.
    pub fn flush(&mut self) -> Vec<f32> {
        if self.inner.is_none() {
            let out = std::mem::take(&mut self.pending);
            self.total_out += out.len() as u64;
            return out;
        }
        // Pad with silence: enough to fill the last chunk, plus two more so the
        // filter's tail — the part still inside its delay line — comes out. The
        // padding is not counted as input, so the length check below trims the
        // result back to exactly what the real audio justifies.
        let padded = self.pending.len().div_ceil(self.frames_per_call) * self.frames_per_call
            + self.frames_per_call * 2;
        self.pending.resize(padded, 0.0);
        let mut out = self.drain_pending();
        let ratio = f64::from(TARGET_SAMPLE_RATE) / f64::from(self.input_rate);
        let target_total = (self.total_in as f64 * ratio).round() as u64;
        let already = self.total_out - out.len() as u64;
        let allowed = target_total.saturating_sub(already) as usize;
        if out.len() > allowed {
            self.total_out -= (out.len() - allowed) as u64;
            out.truncate(allowed);
        }
        out
    }

    fn downmix_into_pending(&mut self, interleaved: &[f32]) {
        if self.input_channels == 1 {
            self.pending.extend_from_slice(interleaved);
            return;
        }
        let scale = 1.0 / self.input_channels as f32;
        for frame in interleaved.chunks_exact(self.input_channels) {
            let sum: f32 = frame.iter().sum();
            self.pending.push(sum * scale);
        }
    }

    fn drain_pending(&mut self) -> Vec<f32> {
        let Some(inner) = self.inner.as_mut() else {
            let out = std::mem::take(&mut self.pending);
            self.total_out += out.len() as u64;
            return out;
        };
        let mut out = Vec::new();
        let mut consumed = 0usize;
        loop {
            let needed = inner.input_frames_next();
            if self.pending.len() - consumed < needed {
                break;
            }
            let input = match InterleavedSlice::new(
                &self.pending[consumed..consumed + needed],
                1,
                needed,
            ) {
                Ok(i) => i,
                Err(_) => break,
            };
            if self.scratch.len() < self.max_out_frames {
                self.scratch.resize(self.max_out_frames, 0.0);
            }
            let produced = {
                let mut output = match InterleavedSlice::new_mut(
                    &mut self.scratch[..],
                    1,
                    self.max_out_frames,
                ) {
                    Ok(o) => o,
                    Err(_) => break,
                };
                match inner.process_into_buffer(&input, &mut output, None) {
                    Ok((_, produced)) => produced,
                    Err(_) => break,
                }
            };
            let mut slice = &self.scratch[..produced];
            if self.delay_to_trim > 0 {
                let skip = self.delay_to_trim.min(slice.len());
                self.delay_to_trim -= skip;
                slice = &slice[skip..];
            }
            out.extend_from_slice(slice);
            consumed += needed;
        }
        if consumed > 0 {
            self.pending.drain(..consumed);
        }
        self.total_out += out.len() as u64;
        out
    }
}

/// Cheap linear resample of a whole buffer, for reading audio back off disk
/// where never failing matters more than the last decibel.
pub fn resample_buffer(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || samples.is_empty() || from_rate == 0 || to_rate == 0 {
        return samples.to_vec();
    }
    let ratio = f64::from(to_rate) / f64::from(from_rate);
    let out_len = ((samples.len() as f64) * ratio).round().max(1.0) as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f64 / ratio;
        let lo = pos.floor() as usize;
        let frac = (pos - lo as f64) as f32;
        let a = samples.get(lo).copied().unwrap_or(0.0);
        let b = samples.get(lo + 1).copied().unwrap_or(a);
        out.push(a + (b - a) * frac);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(rate: u32, ms: usize, hz: f32) -> Vec<f32> {
        let n = rate as usize * ms / 1_000;
        (0..n)
            .map(|i| (i as f32 / rate as f32 * hz * std::f32::consts::TAU).sin() * 0.5)
            .collect()
    }

    #[test]
    fn a_16k_mono_source_passes_straight_through() {
        let mut r = Resampler16k::new(16_000, 1).unwrap();
        let input = sine(16_000, 100, 440.0);
        let out = r.push_mono(&input);
        assert_eq!(out.len(), input.len());
        assert_eq!(out, input);
    }

    #[test]
    fn stereo_is_summed_to_mono() {
        let mut r = Resampler16k::new(16_000, 2).unwrap();
        let interleaved = vec![1.0, -1.0, 0.5, 0.5, 0.2, 0.4];
        let out = r.push_interleaved(&interleaved);
        assert_eq!(out.len(), 3);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 0.5).abs() < 1e-6);
        assert!((out[2] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn forty_eight_k_becomes_a_third_as_many_samples() {
        let mut r = Resampler16k::new(48_000, 1).unwrap();
        let mut total = 0usize;
        // One second, fed in 20 ms pieces like a real device would.
        for _ in 0..50 {
            total += r.push_mono(&sine(48_000, 20, 300.0)).len();
        }
        total += r.flush().len();
        assert_eq!(
            total, 16_000,
            "one second of 48 kHz became {total} samples at 16 kHz"
        );
    }

    #[test]
    fn forty_four_one_k_stereo_also_lands_on_a_second() {
        let mut r = Resampler16k::new(44_100, 2).unwrap();
        let mono = sine(44_100, 1_000, 300.0);
        let mut interleaved = Vec::with_capacity(mono.len() * 2);
        for s in &mono {
            interleaved.push(*s);
            interleaved.push(*s);
        }
        let mut total = 0usize;
        for piece in interleaved.chunks(44_100 / 50 * 2) {
            total += r.push_interleaved(piece).len();
        }
        total += r.flush().len();
        assert_eq!(
            total, 16_000,
            "one second of 44.1 kHz became {total} samples at 16 kHz"
        );
    }

    #[test]
    fn a_downsampled_tone_keeps_its_shape() {
        let mut r = Resampler16k::new(48_000, 1).unwrap();
        let mut out = Vec::new();
        for _ in 0..50 {
            out.extend(r.push_mono(&sine(48_000, 20, 440.0)));
        }
        // Skip the very start before measuring; a band-limited filter needs a
        // few hundred samples to settle.
        let steady = &out[1_000..out.len().min(15_000)];
        let peak = steady.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(
            (0.4..=0.6).contains(&peak),
            "a 0.5 amplitude tone came back at {peak}"
        );
    }

    #[test]
    fn an_unusable_device_format_is_an_error_not_a_panic() {
        assert!(Resampler16k::new(0, 1).is_err());
        assert!(Resampler16k::new(48_000, 0).is_err());
    }

    #[test]
    fn buffer_resampling_scales_length() {
        let input = sine(48_000, 100, 440.0);
        let out = resample_buffer(&input, 48_000, 16_000);
        assert_eq!(out.len(), 1_600);
        let same = resample_buffer(&input, 16_000, 16_000);
        assert_eq!(same.len(), input.len());
    }
}
