//! Microphone capture via `cpal`. This is the "You" channel.
//!
//! Rules for the callback (DESIGN §2 DSP):
//! * no allocation, no locks, no logging, no database, no inference
//! * copy into a bounded [`ringbuf`] producer and return
//! * stamp the block with the meeting clock offset *on entry*
//! * on overflow, count the drop and keep going, the writer thread is the one
//!   that must never fall behind, and it writes straight to disk
//!
//! Resampling to 16 kHz mono happens on the consumer side (`rubato`), never in
//! the callback.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;

use crate::audio::clock::DriftTracker;
use crate::audio::resample::Resampler16k;
use crate::audio::ring::{hand_over, CaptureCounters, RingConsumer, RingProducer};
use crate::audio::{AudioError, Frame, FRAME_MS, TARGET_SAMPLE_RATE};
use crate::types::{AudioDevice, Channel};

/// The callback converts at most this many samples per pass, so its scratch
/// buffer can be allocated once when the stream opens.
const CALLBACK_SCRATCH: usize = 16_384;

/// An open microphone stream.
pub struct MicCapture {
    /// Dropping this closes the device.
    stream: cpal::Stream,
    consumer: RingConsumer,
    counters: Arc<CaptureCounters>,
    resampler: Resampler16k,
    drift: DriftTracker,
    device_name: String,
    native_rate: u32,
    native_channels: u16,
    /// Raw samples popped from the ring, reused between drains.
    scratch: Vec<f32>,
    /// 16 kHz mono that did not fill a whole frame yet.
    pending: Vec<f32>,
    /// Offset of the next frame handed downstream.
    next_t_ms: i64,
    /// 16 kHz samples emitted so far, the authority for timestamps.
    emitted: u64,
    alive: Arc<AtomicBool>,
}

impl std::fmt::Debug for MicCapture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MicCapture")
            .field("device", &self.device_name)
            .field("native_rate", &self.native_rate)
            .field("native_channels", &self.native_channels)
            .field("running", &self.is_running())
            .field("dropped_blocks", &self.dropped_blocks())
            .finish()
    }
}

impl MicCapture {
    /// Open the named device, or the system default when `device_id` is None.
    ///
    /// The device's native rate and channel count are accepted as-is; we
    /// downmix and resample downstream rather than asking the driver for a
    /// format it may refuse.
    pub fn open(device_id: Option<&str>) -> Result<Self, AudioError> {
        Self::open_at(device_id, Instant::now())
    }

    /// Same, but stamped against an existing meeting clock.
    pub fn open_at(device_id: Option<&str>, origin: Instant) -> Result<Self, AudioError> {
        let host = cpal::default_host();
        let device = match device_id {
            Some(id) => find_device(&host, id).ok_or(AudioError::NoInputDevice)?,
            None => host
                .default_input_device()
                .ok_or(AudioError::NoInputDevice)?,
        };
        let device_name = describe(&device);
        let supported = device.default_input_config().map_err(map_cpal_error)?;
        let native_rate = supported.sample_rate();
        let native_channels = supported.channels();
        let sample_format = supported.sample_format();
        let config = supported.config();

        let resampler = Resampler16k::new(native_rate, native_channels)?;
        let (mut producer, consumer, counters) = hand_over(native_rate, native_channels, origin);
        let alive = Arc::new(AtomicBool::new(true));

        let mut scratch = vec![0.0f32; CALLBACK_SCRATCH];
        let error_alive = Arc::clone(&alive);
        let error_counters = Arc::clone(&counters);
        let stream = device
            .build_input_stream_raw(
                config,
                sample_format,
                move |data, _info| {
                    // Hot path: convert in place into a buffer that already
                    // exists and hand it over. Nothing else.
                    convert_and_push(data, &mut scratch, &mut producer);
                },
                move |err| {
                    // Not the audio thread: cpal calls this from its own error
                    // path, so logging here is fine.
                    let fatal = matches!(
                        err.kind(),
                        cpal::ErrorKind::DeviceNotAvailable | cpal::ErrorKind::BackendError
                    );
                    tracing::warn!(target: "echo::audio", "microphone stream error: {err}");
                    if fatal {
                        error_alive.store(false, Ordering::Relaxed);
                        error_counters.mark_stopped();
                    }
                },
                None,
            )
            .map_err(map_cpal_error)?;
        stream.play().map_err(map_cpal_error)?;

        Ok(Self {
            stream,
            consumer,
            counters,
            resampler,
            drift: DriftTracker::new(TARGET_SAMPLE_RATE),
            device_name,
            native_rate,
            native_channels,
            scratch: Vec::with_capacity(native_rate as usize),
            pending: Vec::with_capacity(TARGET_SAMPLE_RATE as usize),
            next_t_ms: 0,
            emitted: 0,
            alive,
        })
    }

    /// Take whatever 16 kHz mono frames are ready. Never blocks.
    pub fn drain(&mut self) -> Vec<Frame> {
        self.scratch.clear();
        let moved = self.consumer.drain_into(&mut self.scratch);
        let mut resampled = if moved == 0 {
            Vec::new()
        } else {
            let raw = std::mem::take(&mut self.scratch);
            let out = self.resampler.push_interleaved(&raw);
            self.scratch = raw;
            out
        };

        if !resampled.is_empty() {
            self.drift
                .observe(resampled.len(), self.counters.last_clock_ms());
            let correction = self.drift.correction_samples();
            if correction > 0 {
                // The source went quiet or fell behind: pad so later audio keeps
                // its place on the meeting clock.
                let mut padded = vec![0.0f32; correction as usize];
                padded.extend_from_slice(&resampled);
                resampled = padded;
                self.drift.apply_correction(correction);
            } else if correction < 0 {
                let drop = (-correction as usize).min(resampled.len());
                resampled.drain(..drop);
                self.drift.apply_correction(-(drop as i64));
            }
        }

        self.pending.extend_from_slice(&resampled);
        self.cut_frames(Channel::Mic)
    }

    fn cut_frames(&mut self, channel: Channel) -> Vec<Frame> {
        let frame_len = TARGET_SAMPLE_RATE as usize * FRAME_MS as usize / 1_000;
        let mut frames = Vec::new();
        let mut consumed = 0usize;
        while consumed + frame_len <= self.pending.len() {
            frames.push(Frame {
                channel,
                t_start_ms: self.next_t_ms,
                samples: self.pending[consumed..consumed + frame_len].to_vec(),
            });
            consumed += frame_len;
            self.emitted += frame_len as u64;
            self.next_t_ms = self.emitted as i64 * 1_000 / i64::from(TARGET_SAMPLE_RATE);
        }
        if consumed > 0 {
            self.pending.drain(..consumed);
        }
        frames
    }

    /// Everything still held inside, at the end of a recording.
    pub fn flush(&mut self) -> Vec<Frame> {
        let tail = self.resampler.flush();
        self.pending.extend_from_slice(&tail);
        let mut frames = self.cut_frames(Channel::Mic);
        if !self.pending.is_empty() {
            let samples = std::mem::take(&mut self.pending);
            let len = samples.len();
            frames.push(Frame {
                channel: Channel::Mic,
                t_start_ms: self.next_t_ms,
                samples,
            });
            self.emitted += len as u64;
            self.next_t_ms = self.emitted as i64 * 1_000 / i64::from(TARGET_SAMPLE_RATE);
        }
        frames
    }

    /// Blocks dropped because the consumer fell behind. Diagnostics only, the
    /// person is told "the transcript is catching up", never a number.
    pub fn dropped_blocks(&self) -> u64 {
        self.counters.dropped_samples()
    }

    /// True until the device disappears (unplugged, profile switch).
    pub fn is_running(&self) -> bool {
        self.alive.load(Ordering::Relaxed) && self.counters.is_running()
    }

    /// Human-readable device name, for the Info tab and diagnostics.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn native_format(&self) -> (u32, u16) {
        (self.native_rate, self.native_channels)
    }

    /// Gaps this source has been through, for the redacted log.
    pub fn discontinuities(&self) -> usize {
        self.drift.discontinuities().len()
    }

    /// Close the stream and release the device (mantra 1).
    pub fn close(self) {
        // Dropping the stream stops the device; be explicit about the order so
        // the callback cannot run against a half-dropped producer.
        let MicCapture { stream, .. } = self;
        let _ = stream.pause();
        drop(stream);
    }
}

/// Convert one callback's worth of native samples into `f32` and hand them over.
///
/// Allocation-free: `scratch` was sized when the stream opened, and a buffer
/// bigger than it is processed in several passes.
fn convert_and_push(data: &cpal::Data, scratch: &mut [f32], producer: &mut RingProducer) {
    macro_rules! push_converted {
        ($ty:ty, $convert:expr) => {{
            let Some(slice) = data.as_slice::<$ty>() else {
                return;
            };
            for part in slice.chunks(scratch.len()) {
                let out = &mut scratch[..part.len()];
                for (dst, src) in out.iter_mut().zip(part.iter()) {
                    *dst = $convert(*src);
                }
                producer.push(out);
            }
        }};
    }

    match data.sample_format() {
        SampleFormat::F32 => push_converted!(f32, |s: f32| s),
        SampleFormat::F64 => push_converted!(f64, |s: f64| s as f32),
        SampleFormat::I8 => push_converted!(i8, |s: i8| f32::from(s) / 128.0),
        SampleFormat::I16 => push_converted!(i16, |s: i16| f32::from(s) / 32_768.0),
        SampleFormat::I32 => push_converted!(i32, |s: i32| s as f32 / 2_147_483_648.0),
        SampleFormat::I64 => push_converted!(i64, |s: i64| s as f32 / 9.223_372e18),
        SampleFormat::U8 => push_converted!(u8, |s: u8| (f32::from(s) - 128.0) / 128.0),
        SampleFormat::U16 => push_converted!(u16, |s: u16| (f32::from(s) - 32_768.0) / 32_768.0),
        SampleFormat::U32 => {
            push_converted!(u32, |s: u32| (s as f64 - 2_147_483_648.0) as f32
                / 2_147_483_648.0)
        }
        // 24-bit and DSD devices are rare enough that guessing at the packing
        // would be worse than saying honestly that we cannot use them.
        _ => {}
    }
}

fn describe(device: &cpal::Device) -> String {
    device
        .description()
        .map(|d| d.name().to_string())
        .unwrap_or_else(|_| "Microphone".to_string())
}

fn device_id_string(device: &cpal::Device) -> Option<String> {
    device.id().ok().map(|id| id.to_string())
}

fn find_device(host: &cpal::Host, id: &str) -> Option<cpal::Device> {
    let devices = host.input_devices().ok()?;
    let mut by_name = None;
    for device in devices {
        if device_id_string(&device).as_deref() == Some(id) {
            return Some(device);
        }
        if by_name.is_none() && describe(&device) == id {
            by_name = Some(device);
        }
    }
    by_name
}

fn map_cpal_error(err: cpal::Error) -> AudioError {
    match err.kind() {
        cpal::ErrorKind::PermissionDenied => AudioError::PermissionDenied,
        cpal::ErrorKind::DeviceNotAvailable => AudioError::DeviceLost,
        cpal::ErrorKind::HostUnavailable => AudioError::NoInputDevice,
        _ => AudioError::Backend(err.to_string()),
    }
}

/// Enumerate input devices. Opens no stream, so it is safe to call while idle.
pub fn list_devices() -> Result<Vec<AudioDevice>, AudioError> {
    let host = cpal::default_host();
    let default_id = host
        .default_input_device()
        .and_then(|d| device_id_string(&d));
    let devices = match host.input_devices() {
        Ok(d) => d,
        // No microphone at all is an empty list, not a failure: Settings still
        // has to render.
        Err(_) => return Ok(Vec::new()),
    };
    let mut out = Vec::new();
    for device in devices {
        let id = match device_id_string(&device) {
            Some(id) => id,
            None => describe(&device),
        };
        let config = device.default_input_config().ok();
        out.push(AudioDevice {
            is_default: Some(&id) == default_id.as_ref(),
            name: describe(&device),
            id,
            sample_rate: config.as_ref().map(|c| c.sample_rate()),
            channels: config.as_ref().map(|c| c.channels()),
        });
    }
    Ok(out)
}

/// The device the OS would pick.
pub fn default_device() -> Result<Option<AudioDevice>, AudioError> {
    let host = cpal::default_host();
    let Some(device) = host.default_input_device() else {
        return Ok(None);
    };
    let config = device.default_input_config().ok();
    let id = device_id_string(&device).unwrap_or_else(|| describe(&device));
    Ok(Some(AudioDevice {
        id,
        name: describe(&device),
        is_default: true,
        sample_rate: config.as_ref().map(|c| c.sample_rate()),
        channels: config.as_ref().map(|c| c.channels()),
    }))
}

/// Is there any microphone at all on this computer?
pub fn any_input_device() -> bool {
    cpal::default_host().default_input_device().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Nothing here opens a device: CI is headless and a test that needs a
    // microphone is a test that fails for the wrong reason.

    #[test]
    fn enumerating_devices_never_fails_even_with_no_microphone() {
        let devices = list_devices().expect("enumeration must not error");
        for d in &devices {
            assert!(!d.id.is_empty(), "a device with no id cannot be selected");
            assert!(!d.name.is_empty());
        }
        assert!(
            devices.iter().filter(|d| d.is_default).count() <= 1,
            "there can only be one default"
        );
    }

    #[test]
    fn the_default_device_agrees_with_the_list() {
        let default = default_device().expect("must not error");
        match default {
            None => assert!(!any_input_device()),
            Some(d) => {
                assert!(d.is_default);
                assert!(!d.id.is_empty());
            }
        }
    }

    #[test]
    fn asking_for_a_device_that_is_not_there_is_a_clean_error() {
        let err = MicCapture::open(Some("this-device-does-not-exist")).unwrap_err();
        assert!(
            matches!(err, AudioError::NoInputDevice),
            "unexpected error: {err}"
        );
    }
}
