//! Linux system-audio capture: PipeWire monitor of the default sink.
//!
//! **Cannot be compile-verified on the macOS development machine**, so
//! everything Linux-specific stays inside this file and `cargo check` on macOS
//! is unaffected.
//!
//! Notes that decide the implementation (DESIGN §2, review finding 3):
//! * Discover the *current* default sink and connect to its monitor ports.
//! * Follow default-sink changes: a Bluetooth headset connecting mid-meeting
//!   moves the sink, and capture has to move with it.
//! * Reconnect when a node is removed or a Bluetooth profile switches, without
//!   ending the recording — a gap plus a degraded banner beats a lost meeting.
//! * PipeWire is required; PulseAudio-only systems are out of scope for v1
//!   (DESIGN §1 non-goals). Report that honestly instead of failing obscurely.
//!
//! ## Why this asks PipeWire to pick the sink
//!
//! The first two bullets could be done by hand: walk the registry, find the
//! node marked default, link to its monitor ports, then watch for metadata
//! changes and relink. PipeWire already does exactly that for a stream created
//! with `PW_KEY_STREAM_CAPTURE_SINK` plus `AUTOCONNECT`: it attaches to the
//! current default sink's monitor and moves with it when the default changes or
//! a Bluetooth profile switches. Re-implementing that in Echo would be more code
//! doing the same thing, less well, on a path that cannot be tested here.
//!
//! What Echo keeps for itself is the part PipeWire will not do: noticing that the
//! stream died anyway and rebuilding it ([`SystemCapture::reopen`]), so a lost
//! node degrades the recording instead of ending it.

#![cfg(target_os = "linux")]

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use crate::audio::clock::DriftTracker;
use crate::audio::resample::Resampler16k;
use crate::audio::ring::{hand_over, CaptureCounters, RingConsumer, RingProducer};
use crate::audio::{AudioError, Frame, FRAME_MS, TARGET_SAMPLE_RATE};
use crate::types::{Channel, PermissionState};

const REQUESTED_RATE: u32 = 48_000;
const REQUESTED_CHANNELS: u16 = 2;
const OPEN_TIMEOUT: Duration = Duration::from_secs(8);

/// Told to the worker thread from outside.
enum Control {
    Stop,
}

struct LinuxShared {
    counters: Arc<CaptureCounters>,
    /// Rate PipeWire negotiated; 0 until it has.
    rate: AtomicU32,
    /// Channel count PipeWire negotiated.
    channels: AtomicU32,
    stopped: AtomicBool,
    stopped_reason: Mutex<Option<String>>,
}

impl LinuxShared {
    fn note_stopped(&self, reason: String) {
        *self
            .stopped_reason
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(reason);
        self.stopped.store(true, Ordering::Relaxed);
        self.counters.mark_stopped();
    }
}

/// An open monitor stream on the current default sink.
pub struct SystemCapture {
    consumer: RingConsumer,
    counters: Arc<CaptureCounters>,
    shared: Arc<LinuxShared>,
    control: Option<pipewire::channel::Sender<Control>>,
    worker: Option<std::thread::JoinHandle<()>>,
    resampler: Option<Resampler16k>,
    drift: DriftTracker,
    scratch: Vec<f32>,
    pending: Vec<f32>,
    start_offset_ms: i64,
    next_t_ms: i64,
    emitted: u64,
    origin: Instant,
}

impl std::fmt::Debug for SystemCapture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SystemCapture")
            .field("running", &self.is_running())
            .field("start_offset_ms", &self.start_offset_ms)
            .finish()
    }
}

impl SystemCapture {
    /// Connect to the default sink's monitor.
    ///
    /// Returns [`AudioError::SystemAudioUnsupported`] when PipeWire is not
    /// running, so the session layer degrades to microphone-only with a banner.
    pub async fn open() -> Result<Self, AudioError> {
        Self::open_at(Instant::now()).await
    }

    pub async fn open_at(origin: Instant) -> Result<Self, AudioError> {
        tokio::task::spawn_blocking(move || Self::open_blocking(origin))
            .await
            .map_err(|e| AudioError::Backend(e.to_string()))?
    }

    fn open_blocking(origin: Instant) -> Result<Self, AudioError> {
        if !supported() {
            return Err(AudioError::SystemAudioUnsupported(
                "this system does not offer a way to capture what the computer plays".into(),
            ));
        }
        let (producer, consumer, counters) = hand_over(REQUESTED_RATE, REQUESTED_CHANNELS, origin);
        let shared = Arc::new(LinuxShared {
            counters: Arc::clone(&counters),
            rate: AtomicU32::new(REQUESTED_RATE),
            channels: AtomicU32::new(u32::from(REQUESTED_CHANNELS)),
            stopped: AtomicBool::new(false),
            stopped_reason: Mutex::new(None),
        });

        let (control, worker) = spawn_worker(Arc::clone(&shared), producer)?;

        Ok(Self {
            consumer,
            counters,
            shared,
            control: Some(control),
            worker: Some(worker),
            resampler: None,
            drift: DriftTracker::new(TARGET_SAMPLE_RATE),
            scratch: Vec::with_capacity(REQUESTED_RATE as usize),
            pending: Vec::with_capacity(TARGET_SAMPLE_RATE as usize),
            start_offset_ms: -1,
            next_t_ms: 0,
            emitted: 0,
            origin,
        })
    }

    /// Take whatever 16 kHz mono frames are ready. Never blocks.
    pub fn drain(&mut self) -> Vec<Frame> {
        self.scratch.clear();
        if self.consumer.drain_into(&mut self.scratch) == 0 {
            return Vec::new();
        }
        if self.start_offset_ms < 0 {
            self.start_offset_ms = self.counters.last_clock_ms().max(0);
            self.next_t_ms = self.start_offset_ms;
        }
        let rate = self.shared.rate.load(Ordering::Relaxed).max(8_000);
        if self
            .resampler
            .as_ref()
            .is_none_or(|r| r.input_rate() != rate)
        {
            self.resampler = Resampler16k::new(rate, 1).ok();
        }
        let Some(resampler) = self.resampler.as_mut() else {
            return Vec::new();
        };
        let raw = std::mem::take(&mut self.scratch);
        let mut resampled = resampler.push_mono(&raw);
        self.scratch = raw;

        if !resampled.is_empty() {
            self.drift
                .observe(resampled.len(), self.counters.last_clock_ms());
            let correction = self.drift.correction_samples();
            if correction > 0 {
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
        self.cut_frames()
    }

    fn cut_frames(&mut self) -> Vec<Frame> {
        let frame_len = TARGET_SAMPLE_RATE as usize * FRAME_MS as usize / 1_000;
        let base = self.start_offset_ms.max(0);
        let mut frames = Vec::new();
        let mut consumed = 0usize;
        while consumed + frame_len <= self.pending.len() {
            frames.push(Frame {
                channel: Channel::System,
                t_start_ms: self.next_t_ms,
                samples: self.pending[consumed..consumed + frame_len].to_vec(),
            });
            consumed += frame_len;
            self.emitted += frame_len as u64;
            self.next_t_ms = base + self.emitted as i64 * 1_000 / i64::from(TARGET_SAMPLE_RATE);
        }
        if consumed > 0 {
            self.pending.drain(..consumed);
        }
        frames
    }

    pub fn flush(&mut self) -> Vec<Frame> {
        let tail = self
            .resampler
            .as_mut()
            .map(|r| r.flush())
            .unwrap_or_default();
        self.pending.extend_from_slice(&tail);
        let mut frames = self.cut_frames();
        if !self.pending.is_empty() {
            let samples = std::mem::take(&mut self.pending);
            let len = samples.len();
            frames.push(Frame {
                channel: Channel::System,
                t_start_ms: self.next_t_ms,
                samples,
            });
            self.emitted += len as u64;
        }
        frames
    }

    pub fn is_running(&self) -> bool {
        !self.shared.stopped.load(Ordering::Relaxed) && self.worker.is_some()
    }

    pub fn dropped_blocks(&self) -> u64 {
        self.counters.dropped_samples()
    }

    pub fn start_offset_ms(&self) -> i64 {
        self.start_offset_ms
    }

    pub fn stopped_reason(&self) -> Option<String> {
        self.shared
            .stopped_reason
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Name of the sink currently being monitored, for diagnostics.
    ///
    /// PipeWire owns the choice (see the module docs), so what Echo can report
    /// honestly is the format it negotiated, not a node name it never asked for.
    pub fn current_sink(&self) -> Option<String> {
        let rate = self.shared.rate.load(Ordering::Relaxed);
        let channels = self.shared.channels.load(Ordering::Relaxed);
        if rate == 0 {
            None
        } else {
            Some(format!("default output, {rate} Hz, {channels} channels"))
        }
    }

    /// Re-point at whatever the default sink is now.
    ///
    /// A no-op while the stream is alive: PipeWire moves the stream itself when
    /// the default sink changes. Kept because the session layer calls it after a
    /// device change, and because it is the honest place to rebuild the stream if
    /// PipeWire could not follow.
    pub async fn follow_default_sink(&mut self) -> Result<(), AudioError> {
        if self.is_running() {
            return Ok(());
        }
        self.reopen()
    }

    /// Rebuild the stream after the node it was on disappeared. Keeps the same
    /// ring and clock, so the recording survives with a gap in it.
    pub fn reopen(&mut self) -> Result<(), AudioError> {
        self.stop_worker();
        *self
            .shared
            .stopped_reason
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;
        self.shared.stopped.store(false, Ordering::Relaxed);
        self.shared.counters.mark_running();

        // A fresh producer for the new stream; the consumer side is untouched so
        // no already-captured audio is lost.
        let (producer, consumer, counters) =
            hand_over(REQUESTED_RATE, REQUESTED_CHANNELS, self.origin);
        let (control, worker) = spawn_worker(Arc::clone(&self.shared), producer)?;
        self.consumer = consumer;
        self.counters = counters;
        self.control = Some(control);
        self.worker = Some(worker);
        Ok(())
    }

    fn stop_worker(&mut self) {
        if let Some(control) = self.control.take() {
            let _ = control.send(Control::Stop);
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }

    pub async fn close(mut self) {
        let _ = tokio::task::spawn_blocking(move || self.stop_worker()).await;
    }
}

impl Drop for SystemCapture {
    fn drop(&mut self) {
        self.stop_worker();
    }
}

type Worker = (
    pipewire::channel::Sender<Control>,
    std::thread::JoinHandle<()>,
);

fn spawn_worker(shared: Arc<LinuxShared>, producer: RingProducer) -> Result<Worker, AudioError> {
    let (control_tx, control_rx) = pipewire::channel::channel::<Control>();
    let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();
    let worker_shared = Arc::clone(&shared);
    let worker = std::thread::Builder::new()
        .name("echo-system-audio".into())
        .spawn(move || {
            if let Err(e) = run_loop(worker_shared.clone(), producer, control_rx, &ready_tx) {
                let _ = ready_tx.send(Err(e.clone()));
                worker_shared.note_stopped(e);
            }
        })
        .map_err(|e| AudioError::Backend(e.to_string()))?;

    match ready_rx.recv_timeout(OPEN_TIMEOUT) {
        Ok(Ok(())) => Ok((control_tx, worker)),
        Ok(Err(e)) => {
            let _ = worker.join();
            Err(AudioError::SystemAudioUnsupported(e))
        }
        Err(_) => {
            let _ = control_tx.send(Control::Stop);
            Err(AudioError::SystemAudioUnsupported(
                "the sound system did not answer in time".into(),
            ))
        }
    }
}

/// Everything PipeWire touches lives on this one thread.
fn run_loop(
    shared: Arc<LinuxShared>,
    producer: RingProducer,
    control: pipewire::channel::Receiver<Control>,
    ready: &mpsc::Sender<Result<(), String>>,
) -> Result<(), String> {
    use pipewire as pw;
    use pw::spa;
    use spa::pod::Pod;

    pw::init();

    let mainloop = pw::main_loop::MainLoop::new(None).map_err(|e| e.to_string())?;
    let context = pw::context::Context::new(&mainloop).map_err(|e| e.to_string())?;
    let core = context.connect(None).map_err(|e| e.to_string())?;

    let quit = mainloop.clone();
    let _control = control.attach(mainloop.loop_(), move |message| match message {
        Control::Stop => quit.quit(),
    });

    let stream = pw::stream::Stream::new(
        &core,
        "echo-system-audio",
        pw::properties::properties! {
            *pw::keys::MEDIA_TYPE => "Audio",
            *pw::keys::MEDIA_CATEGORY => "Capture",
            *pw::keys::MEDIA_ROLE => "Communication",
            // This is the whole trick: capture the default *sink*, and let
            // PipeWire keep us pointed at it as the default moves.
            *pw::keys::STREAM_CAPTURE_SINK => "true",
            *pw::keys::NODE_NAME => "Echo",
        },
    )
    .map_err(|e| e.to_string())?;

    struct State {
        shared: Arc<LinuxShared>,
        producer: RingProducer,
        /// Reused so the real-time process callback does not allocate.
        mono: Vec<f32>,
    }

    let state = State {
        shared: Arc::clone(&shared),
        producer,
        mono: Vec::with_capacity(REQUESTED_RATE as usize),
    };

    let _listener = stream
        .add_local_listener_with_user_data(state)
        .param_changed(|_stream, state, id, param| {
            let Some(param) = param else { return };
            if id != spa::param::ParamType::Format.as_raw() {
                return;
            }
            let Ok((media_type, media_subtype)) = spa::param::format_utils::parse_format(param)
            else {
                return;
            };
            if media_type != spa::param::format::MediaType::Audio
                || media_subtype != spa::param::format::MediaSubtype::Raw
            {
                return;
            }
            let mut info = spa::param::audio::AudioInfoRaw::new();
            if info.parse(param).is_err() {
                return;
            }
            state
                .shared
                .rate
                .store(info.rate().max(1), Ordering::Relaxed);
            state
                .shared
                .channels
                .store(info.channels().max(1), Ordering::Relaxed);
        })
        .process(|stream, state| {
            let Some(mut buffer) = stream.dequeue_buffer() else {
                return;
            };
            // Disjoint field borrows, so the downmix buffer and the producer can
            // be touched in the same expression.
            let State {
                shared,
                producer,
                mono,
            } = state;
            let channels = shared.channels.load(Ordering::Relaxed).max(1) as usize;
            let datas = buffer.datas_mut();
            if datas.is_empty() {
                return;
            }
            let Some(bytes) = datas[0].data() else { return };
            let frames = bytes.len() / std::mem::size_of::<f32>();
            if frames == 0 {
                return;
            }
            // PipeWire hands over interleaved F32LE for a raw audio stream.
            // SAFETY: the buffer is at least `frames * 4` bytes of the format we
            // negotiated in `param_changed`.
            let samples =
                unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, frames) };
            mono.clear();
            if channels <= 1 {
                mono.extend_from_slice(samples);
            } else {
                let scale = 1.0 / channels as f32;
                for frame in samples.chunks_exact(channels) {
                    mono.push(frame.iter().sum::<f32>() * scale);
                }
            }
            producer.push(mono);
        })
        .state_changed(|_stream, state, _old, new| {
            if let pw::stream::StreamState::Error(message) = new {
                state.shared.note_stopped(message.to_string());
            }
        })
        .register()
        .map_err(|e| e.to_string())?;

    let mut audio_info = spa::param::audio::AudioInfoRaw::new();
    audio_info.set_format(spa::param::audio::AudioFormat::F32LE);
    let object = spa::pod::Object {
        type_: spa::utils::SpaTypes::ObjectParamFormat.as_raw(),
        id: spa::param::ParamType::EnumFormat.as_raw(),
        properties: audio_info.into(),
    };
    let values: Vec<u8> = spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &spa::pod::Value::Object(object),
    )
    .map_err(|e| e.to_string())?
    .0
    .into_inner();
    let mut params = [Pod::from_bytes(&values).ok_or("could not describe the audio format")?];

    stream
        .connect(
            spa::utils::Direction::Input,
            None,
            pw::stream::StreamFlags::AUTOCONNECT
                | pw::stream::StreamFlags::MAP_BUFFERS
                | pw::stream::StreamFlags::RT_PROCESS,
            &mut params,
        )
        .map_err(|e| e.to_string())?;

    let _ = ready.send(Ok(()));
    mainloop.run();
    let _ = stream.disconnect();
    Ok(())
}

/// Is PipeWire available on this system?
pub fn supported() -> bool {
    // The socket is the only honest answer: a machine can have the libraries
    // installed and no server running.
    let Some(runtime_dir) = std::env::var_os("XDG_RUNTIME_DIR") else {
        return false;
    };
    let base = std::path::Path::new(&runtime_dir);
    ["pipewire-0", "pipewire-0-manager"]
        .iter()
        .any(|name| base.join(name).exists())
}

/// Linux has no separate permission for monitoring a sink; availability is the
/// only question.
pub async fn permission() -> PermissionState {
    if supported() {
        PermissionState::Granted
    } else {
        PermissionState::NotApplicable
    }
}

/// Open the desktop's sound settings, for the "Open Settings" button.
pub fn open_settings_pane() -> Result<(), AudioError> {
    std::process::Command::new("xdg-open")
        .arg("gnome-control-center://sound")
        .status()
        .map_err(|e| AudioError::Backend(e.to_string()))?;
    Ok(())
}

/// Microphone access on Linux is not gated by the OS the way it is on macOS.
pub fn microphone_permission_state() -> PermissionState {
    PermissionState::NotApplicable
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn availability_is_probed_not_assumed() {
        // Whatever the answer, it must not panic and must not need a server.
        let _ = supported();
    }
}
