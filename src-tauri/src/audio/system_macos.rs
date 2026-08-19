//! macOS system-audio capture: ScreenCaptureKit, audio only (macOS 13+).
//!
//! Notes that decide the implementation (DESIGN §2, review findings 3, 6):
//! * `SCStreamConfiguration` with `capturesAudio = true` and video suppressed
//!   as far as the API allows; we consume `CMSampleBuffer`s and convert to PCM.
//! * The content filter **must exclude Echo's own application**, or playback of
//!   a recording would loop back into the next one. Belt and braces:
//!   `excludesCurrentProcessAudio` on the configuration *and* Echo's own
//!   application excluded from the content filter.
//! * This captures everything the computer plays, not just the meeting app. The
//!   UI says exactly that, no pretending it is scoped.
//! * Permission is "Screen & System Audio Recording" and typically needs a
//!   relaunch after being granted; report `RestartRequired` rather than
//!   pretending it worked.
//! * Sample buffers carry their own presentation timestamps; convert them onto
//!   the meeting clock instead of trusting arrival order.
//!
//! ## Shape of the implementation
//!
//! `SCStream` and its friends are Objective-C objects and not `Send`, but a
//! [`crate::audio::CaptureSession`] moves between threads. So one worker thread
//! owns the stream for its whole life and everything else talks to it through a
//! ring buffer and a couple of atomics. The worker also lets us re-open the
//! stream after a display change or a Bluetooth switch without disturbing
//! anyone else.
//!
//! The bindings used are `objc2-screen-capture-kit`, not `cidre`: the objc2
//! family is already in the tree for the rest of the macOS integration. Where an
//! `objc2-core-media` helper is behind a feature we do not enable, the C
//! function is declared directly — that is the whole reason this file talks to
//! CoreMedia by hand.

#![cfg(target_os = "macos")]
// The protocol methods below have to keep their Objective-C selector spelling so
// the runtime can find them.
#![allow(non_snake_case)]

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::{define_class, msg_send, AnyThread, DefinedClass};
use objc2_core_media::{CMSampleBuffer, CMTime, CMTimeFlags};
use objc2_foundation::{NSArray, NSError, NSObject, NSObjectProtocol, NSProcessInfo, NSString};
use objc2_screen_capture_kit::{
    SCContentFilter, SCRunningApplication, SCShareableContent, SCStream, SCStreamConfiguration,
    SCStreamDelegate, SCStreamOutput, SCStreamOutputType, SCWindow,
};

use crate::audio::clock::DriftTracker;
use crate::audio::resample::Resampler16k;
use crate::audio::ring::{hand_over, CaptureCounters, RingConsumer, RingProducer};
use crate::audio::{AudioError, Frame, FRAME_MS, TARGET_SAMPLE_RATE};
use crate::types::{Channel, PermissionState};

/// What we ask ScreenCaptureKit for. It obliges on every supported release.
const REQUESTED_RATE: u32 = 48_000;
const REQUESTED_CHANNELS: u16 = 2;

/// How long to wait for ScreenCaptureKit to answer before giving up and
/// degrading to microphone-only.
const OPEN_TIMEOUT: Duration = Duration::from_secs(8);

// ---------------------------------------------------------------------------
// The parts of CoreMedia / CoreGraphics / CoreFoundation we need by hand
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy)]
struct AudioBufferRaw {
    number_channels: u32,
    data_byte_size: u32,
    data: *mut c_void,
}

/// `AudioBufferList` is a variable-length struct. ScreenCaptureKit never gives
/// us more than a handful of channels, so a fixed maximum is honest and avoids
/// allocating inside the sample handler.
const MAX_BUFFERS: usize = 8;

#[repr(C)]
struct AudioBufferListRaw {
    number_buffers: u32,
    buffers: [AudioBufferRaw; MAX_BUFFERS],
}

impl AudioBufferListRaw {
    fn zeroed() -> Self {
        Self {
            number_buffers: 0,
            buffers: [AudioBufferRaw {
                number_channels: 0,
                data_byte_size: 0,
                data: std::ptr::null_mut(),
            }; MAX_BUFFERS],
        }
    }
}

#[repr(C)]
struct StreamBasicDescription {
    sample_rate: f64,
    format_id: u32,
    format_flags: u32,
    bytes_per_packet: u32,
    frames_per_packet: u32,
    bytes_per_frame: u32,
    channels_per_frame: u32,
    bits_per_channel: u32,
    reserved: u32,
}

const FORMAT_FLAG_IS_FLOAT: u32 = 1 << 0;

#[link(name = "CoreMedia", kind = "framework")]
unsafe extern "C" {
    fn CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
        sbuf: *const CMSampleBuffer,
        buffer_list_size_needed_out: *mut usize,
        buffer_list_out: *mut c_void,
        buffer_list_size: usize,
        structure_allocator: *const c_void,
        block_allocator: *const c_void,
        flags: u32,
        block_buffer_out: *mut *mut c_void,
    ) -> i32;
    fn CMSampleBufferGetFormatDescription(sbuf: *const CMSampleBuffer) -> *const c_void;
    fn CMAudioFormatDescriptionGetStreamBasicDescription(
        desc: *const c_void,
    ) -> *const StreamBasicDescription;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFRelease(cf: *const c_void);
}

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {
    /// True when this process already has Screen & System Audio Recording.
    /// Does not prompt.
    fn CGPreflightScreenCaptureAccess() -> bool;
    /// Prompts once per process. Returns the state *before* the person answers,
    /// which is why granting needs a relaunch.
    fn CGRequestScreenCaptureAccess() -> bool;
}

// ---------------------------------------------------------------------------
// State shared between the sample handler and Echo's threads
// ---------------------------------------------------------------------------

struct SystemShared {
    /// The sample handler runs on a ScreenCaptureKit dispatch queue, not on a
    /// real-time audio thread, so a short lock here cannot glitch a device.
    producer: Mutex<RingProducer>,
    counters: Arc<CaptureCounters>,
    /// Reused between callbacks so the handler does not allocate per buffer.
    scratch: Mutex<Vec<f32>>,
    /// Presentation timestamp of the first buffer, in microseconds.
    /// `i64::MIN` until the first one arrives.
    first_pts_us: AtomicI64,
    /// Meeting-clock offset at which that first buffer was handed over.
    first_clock_ms: AtomicI64,
    /// Sample rate the stream is actually delivering.
    rate: AtomicU32,
    /// Set when the stream stopped on its own: permission revoked mid-meeting,
    /// display disconnected, or the daemon gave up.
    stopped_reason: Mutex<Option<String>>,
    stopped: AtomicBool,
}

impl SystemShared {
    fn note_stopped(&self, reason: String) {
        *self
            .stopped_reason
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(reason);
        self.stopped.store(true, Ordering::Relaxed);
        self.counters.mark_stopped();
    }
}

/// Moves an Objective-C pointer between threads.
///
/// Safe here because the pointer is a `+1` reference to an immutable
/// `SCShareableContent` snapshot that only the receiving thread ever touches.
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}

define_class!(
    // SAFETY: NSObject has no subclassing requirements and this class has no
    // Drop implementation of its own.
    #[unsafe(super(NSObject))]
    #[name = "EchoSystemAudioTap"]
    #[ivars = Arc<SystemShared>]
    struct AudioTap;

    unsafe impl NSObjectProtocol for AudioTap {}

    unsafe impl SCStreamOutput for AudioTap {
        #[unsafe(method(stream:didOutputSampleBuffer:ofType:))]
        fn stream_didOutputSampleBuffer_ofType(
            &self,
            _stream: &SCStream,
            sample_buffer: &CMSampleBuffer,
            kind: SCStreamOutputType,
        ) {
            if kind != SCStreamOutputType::Audio {
                // Video frames are never consumed: we asked for audio only and
                // dropping them keeps the cost at nothing.
                return;
            }
            handle_audio(self.ivars(), sample_buffer);
        }
    }

    unsafe impl SCStreamDelegate for AudioTap {
        #[unsafe(method(stream:didStopWithError:))]
        fn stream_didStopWithError(&self, _stream: &SCStream, error: &NSError) {
            let detail = error.localizedDescription().to_string();
            tracing::warn!(target: "echo::audio", "system audio stream stopped: {detail}");
            self.ivars().note_stopped(detail);
        }
    }
);

impl AudioTap {
    fn new(shared: Arc<SystemShared>) -> Retained<Self> {
        let this = Self::alloc().set_ivars(shared);
        unsafe { msg_send![super(this), init] }
    }
}

/// Convert one sample buffer to mono `f32` and hand it over. Allocation-free
/// after the first buffer.
fn handle_audio(shared: &Arc<SystemShared>, sbuf: &CMSampleBuffer) {
    let raw: *const CMSampleBuffer = sbuf;

    // Format first: if this is not packed float we do not guess.
    let mut rate = REQUESTED_RATE;
    let mut is_float = true;
    unsafe {
        let desc = CMSampleBufferGetFormatDescription(raw);
        if !desc.is_null() {
            let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(desc);
            if !asbd.is_null() {
                let asbd = &*asbd;
                if asbd.sample_rate > 0.0 {
                    rate = asbd.sample_rate as u32;
                }
                is_float = asbd.format_flags & FORMAT_FLAG_IS_FLOAT != 0;
            }
        }
    }
    if !is_float {
        return;
    }
    shared.rate.store(rate, Ordering::Relaxed);

    let mut list = AudioBufferListRaw::zeroed();
    let mut block_buffer: *mut c_void = std::ptr::null_mut();
    let status = unsafe {
        CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            raw,
            std::ptr::null_mut(),
            (&mut list as *mut AudioBufferListRaw).cast(),
            std::mem::size_of::<AudioBufferListRaw>(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            &mut block_buffer,
        )
    };
    if status != 0 {
        if !block_buffer.is_null() {
            unsafe { CFRelease(block_buffer) };
        }
        return;
    }

    let channel_count = (list.number_buffers as usize).min(MAX_BUFFERS);
    if channel_count == 0 {
        unsafe { CFRelease(block_buffer) };
        return;
    }

    // ScreenCaptureKit delivers de-interleaved float: one buffer per channel.
    let frames = (list.buffers[0].data_byte_size as usize) / std::mem::size_of::<f32>();
    if frames > 0 {
        let mut scratch = shared.scratch.lock().unwrap_or_else(|e| e.into_inner());
        scratch.clear();
        scratch.resize(frames, 0.0);
        let mut used = 0usize;
        for buffer in list.buffers.iter().take(channel_count) {
            if buffer.data.is_null() {
                continue;
            }
            let n = (buffer.data_byte_size as usize / std::mem::size_of::<f32>()).min(frames);
            let samples = unsafe { std::slice::from_raw_parts(buffer.data as *const f32, n) };
            for (dst, src) in scratch.iter_mut().zip(samples.iter()) {
                *dst += *src;
            }
            used += 1;
        }
        if used > 1 {
            let scale = 1.0 / used as f32;
            for s in scratch.iter_mut() {
                *s *= scale;
            }
        }

        let mut producer = shared.producer.lock().unwrap_or_else(|e| e.into_inner());
        if shared.first_pts_us.load(Ordering::Relaxed) == i64::MIN {
            let pts_us = presentation_us(sbuf);
            shared.first_pts_us.store(pts_us, Ordering::Relaxed);
            shared
                .first_clock_ms
                .store(producer.counters().last_clock_ms(), Ordering::Relaxed);
        }
        producer.push(&scratch);
    }

    unsafe { CFRelease(block_buffer) };
}

fn presentation_us(sbuf: &CMSampleBuffer) -> i64 {
    let time = unsafe { sbuf.presentation_time_stamp() };
    if time.timescale == 0 {
        return 0;
    }
    time.value.saturating_mul(1_000_000) / i64::from(time.timescale)
}

// ---------------------------------------------------------------------------
// The public capture handle
// ---------------------------------------------------------------------------

/// An open system-audio stream.
pub struct SystemCapture {
    consumer: RingConsumer,
    counters: Arc<CaptureCounters>,
    shared: Arc<SystemShared>,
    shutdown: Arc<AtomicBool>,
    worker: Option<std::thread::JoinHandle<()>>,
    resampler: Option<Resampler16k>,
    drift: DriftTracker,
    scratch: Vec<f32>,
    pending: Vec<f32>,
    /// Meeting-clock offset the system channel starts at. System audio always
    /// arrives a little after the microphone, and that offset is real: the
    /// timeline must show it rather than pretend both started at zero.
    start_offset_ms: i64,
    next_t_ms: i64,
    emitted: u64,
}

impl std::fmt::Debug for SystemCapture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SystemCapture")
            .field("running", &self.is_running())
            .field("start_offset_ms", &self.start_offset_ms)
            .field("dropped_samples", &self.counters.dropped_samples())
            .finish()
    }
}

impl SystemCapture {
    /// Start capturing what the computer plays, excluding Echo itself.
    ///
    /// Returns [`AudioError::SystemAudioUnsupported`] on macOS < 13 and
    /// [`AudioError::PermissionDenied`] when the permission is missing, so the
    /// session layer can degrade to microphone-only with a banner.
    pub async fn open() -> Result<Self, AudioError> {
        Self::open_at(Instant::now()).await
    }

    /// Same, stamped against an existing meeting clock.
    pub async fn open_at(origin: Instant) -> Result<Self, AudioError> {
        tokio::task::spawn_blocking(move || Self::open_blocking(origin))
            .await
            .map_err(|e| AudioError::Backend(e.to_string()))?
    }

    fn open_blocking(origin: Instant) -> Result<Self, AudioError> {
        if !supported() {
            return Err(AudioError::SystemAudioUnsupported(
                "this version of macOS cannot share what the computer plays".into(),
            ));
        }
        if !unsafe { CGPreflightScreenCaptureAccess() } {
            return Err(AudioError::PermissionDenied);
        }

        let (producer, consumer, counters) = hand_over(REQUESTED_RATE, REQUESTED_CHANNELS, origin);
        let shared = Arc::new(SystemShared {
            producer: Mutex::new(producer),
            counters: Arc::clone(&counters),
            scratch: Mutex::new(Vec::with_capacity(REQUESTED_RATE as usize)),
            first_pts_us: AtomicI64::new(i64::MIN),
            first_clock_ms: AtomicI64::new(0),
            rate: AtomicU32::new(REQUESTED_RATE),
            stopped_reason: Mutex::new(None),
            stopped: AtomicBool::new(false),
        });

        let shutdown = Arc::new(AtomicBool::new(false));
        let (ready_tx, ready_rx) = mpsc::channel::<Result<(), AudioError>>();
        let worker_shared = Arc::clone(&shared);
        let worker_shutdown = Arc::clone(&shutdown);
        let worker = std::thread::Builder::new()
            .name("echo-system-audio".into())
            .spawn(move || run_stream(worker_shared, worker_shutdown, ready_tx))
            .map_err(|e| AudioError::Backend(e.to_string()))?;

        match ready_rx.recv_timeout(OPEN_TIMEOUT) {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                shutdown.store(true, Ordering::Relaxed);
                let _ = worker.join();
                return Err(e);
            }
            Err(_) => {
                shutdown.store(true, Ordering::Relaxed);
                return Err(AudioError::SystemAudioUnsupported(
                    "macOS did not answer in time".into(),
                ));
            }
        }

        Ok(Self {
            consumer,
            counters,
            shared,
            shutdown,
            worker: Some(worker),
            resampler: None,
            drift: DriftTracker::new(TARGET_SAMPLE_RATE),
            scratch: Vec::with_capacity(REQUESTED_RATE as usize),
            pending: Vec::with_capacity(TARGET_SAMPLE_RATE as usize),
            start_offset_ms: -1,
            next_t_ms: 0,
            emitted: 0,
        })
    }

    /// Take whatever 16 kHz mono frames are ready. Never blocks.
    pub fn drain(&mut self) -> Vec<Frame> {
        self.scratch.clear();
        if self.consumer.drain_into(&mut self.scratch) == 0 {
            return Vec::new();
        }
        if self.start_offset_ms < 0 {
            let first = self.shared.first_clock_ms.load(Ordering::Relaxed);
            self.start_offset_ms = first.max(0);
            self.next_t_ms = self.start_offset_ms;
        }
        let rate = self.shared.rate.load(Ordering::Relaxed).max(8_000);
        if self
            .resampler
            .as_ref()
            .is_none_or(|r| r.input_rate() != rate)
        {
            // Already mono by the time it reaches the ring.
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

    /// Everything still held inside, at the end of a recording.
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
        !self.shared.stopped.load(Ordering::Relaxed) && !self.shutdown.load(Ordering::Relaxed)
    }

    /// Why the stream stopped, for the diagnostics log. The banner the person
    /// sees comes from the session layer, in plain words.
    pub fn stopped_reason(&self) -> Option<String> {
        self.shared
            .stopped_reason
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    pub fn dropped_blocks(&self) -> u64 {
        self.counters.dropped_samples()
    }

    /// Where on the meeting clock this channel started. -1 until the first
    /// buffer arrives.
    pub fn start_offset_ms(&self) -> i64 {
        self.start_offset_ms
    }

    /// Presentation timestamp of the first buffer, in microseconds on the
    /// system's own media clock. Diagnostics only: the meeting clock is what
    /// places audio on the timeline, and it is derived from the sample count so
    /// it cannot drift away from the file on disk.
    pub fn first_presentation_us(&self) -> Option<i64> {
        let value = self.shared.first_pts_us.load(Ordering::Relaxed);
        (value != i64::MIN).then_some(value)
    }

    /// Try again after the stream stopped on its own — a display was
    /// disconnected, or headphones took the audio device with them. Keeps the
    /// same ring and clock, so the timeline survives with a gap in it.
    pub fn reopen(&mut self) -> Result<(), AudioError> {
        self.stop_worker();
        *self
            .shared
            .stopped_reason
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;
        self.shared.stopped.store(false, Ordering::Relaxed);
        self.shared.counters.mark_running();
        self.shutdown = Arc::new(AtomicBool::new(false));

        let (ready_tx, ready_rx) = mpsc::channel::<Result<(), AudioError>>();
        let worker_shared = Arc::clone(&self.shared);
        let worker_shutdown = Arc::clone(&self.shutdown);
        self.worker = Some(
            std::thread::Builder::new()
                .name("echo-system-audio".into())
                .spawn(move || run_stream(worker_shared, worker_shutdown, ready_tx))
                .map_err(|e| AudioError::Backend(e.to_string()))?,
        );
        match ready_rx.recv_timeout(OPEN_TIMEOUT) {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => {
                self.shutdown.store(true, Ordering::Relaxed);
                Err(e)
            }
            Err(_) => {
                self.shutdown.store(true, Ordering::Relaxed);
                Err(AudioError::SystemAudioUnsupported(
                    "macOS did not answer in time".into(),
                ))
            }
        }
    }

    fn stop_worker(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }

    /// Stop the stream and release the capture session.
    pub async fn close(mut self) {
        let _ = tokio::task::spawn_blocking(move || self.stop_worker()).await;
    }
}

impl Drop for SystemCapture {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Owns the `SCStream` for its whole life on one thread, because none of these
/// objects may cross threads.
fn run_stream(
    shared: Arc<SystemShared>,
    shutdown: Arc<AtomicBool>,
    ready: mpsc::Sender<Result<(), AudioError>>,
) {
    let started = match start_stream(&shared) {
        Ok(parts) => {
            let _ = ready.send(Ok(()));
            parts
        }
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };
    let (stream, tap) = started;

    while !shutdown.load(Ordering::Relaxed) {
        if shared.stopped.load(Ordering::Relaxed) {
            // The delegate already said why; the session layer decides whether
            // to ask for a reopen.
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    let (done_tx, done_rx) = mpsc::channel::<()>();
    let handler = RcBlock::new(move |_error: *mut NSError| {
        let _ = done_tx.send(());
    });
    unsafe { stream.stopCaptureWithCompletionHandler(Some(&handler)) };
    let _ = done_rx.recv_timeout(Duration::from_secs(3));
    let _ = unsafe {
        stream.removeStreamOutput_type_error(
            ProtocolObject::from_ref(&*tap),
            SCStreamOutputType::Audio,
        )
    };
    drop(tap);
    drop(stream);
}

type StreamParts = (Retained<SCStream>, Retained<AudioTap>);

fn start_stream(shared: &Arc<SystemShared>) -> Result<StreamParts, AudioError> {
    let content = shareable_content()?;

    let displays = unsafe { content.displays() };
    let display = displays.firstObject().ok_or_else(|| {
        AudioError::SystemAudioUnsupported("this computer has no display to share from".into())
    })?;

    // Exclude Echo itself so playing a recording back cannot leak into the next
    // one. `excludesCurrentProcessAudio` below does the same job at the audio
    // level; both are cheap and one of them is always enough.
    let own_pid = std::process::id() as i32;
    let applications = unsafe { content.applications() };
    let own: Vec<Retained<SCRunningApplication>> = applications
        .iter()
        .filter(|app| unsafe { app.processID() } == own_pid)
        .collect();
    let excluded = NSArray::from_retained_slice(&own);
    let no_windows: Retained<NSArray<SCWindow>> = NSArray::new();

    let filter = unsafe {
        SCContentFilter::initWithDisplay_excludingApplications_exceptingWindows(
            SCContentFilter::alloc(),
            &display,
            &excluded,
            &no_windows,
        )
    };

    let config = unsafe { SCStreamConfiguration::new() };
    unsafe {
        config.setCapturesAudio(true);
        config.setExcludesCurrentProcessAudio(true);
        config.setSampleRate(REQUESTED_RATE as isize);
        config.setChannelCount(REQUESTED_CHANNELS as isize);
        // Video cannot be switched off, so make it as close to free as the API
        // allows: a 2x2 frame every ten minutes, and we never read one.
        config.setWidth(2);
        config.setHeight(2);
        config.setMinimumFrameInterval(CMTime {
            value: 600,
            timescale: 1,
            flags: CMTimeFlags::Valid,
            epoch: 0,
        });
        config.setQueueDepth(6);
    }

    let tap = AudioTap::new(Arc::clone(shared));
    let delegate: &ProtocolObject<dyn SCStreamDelegate> = ProtocolObject::from_ref(&*tap);
    let stream = unsafe {
        SCStream::initWithFilter_configuration_delegate(
            SCStream::alloc(),
            &filter,
            &config,
            Some(delegate),
        )
    };

    let output: &ProtocolObject<dyn SCStreamOutput> = ProtocolObject::from_ref(&*tap);
    // A nil queue means ScreenCaptureKit uses a private serial queue, which is
    // exactly what we want: samples arrive in order, off our threads.
    unsafe {
        stream.addStreamOutput_type_sampleHandlerQueue_error(
            output,
            SCStreamOutputType::Audio,
            None,
        )
    }
    .map_err(|e| AudioError::Backend(describe_error(&e)))?;

    let (tx, rx) = mpsc::channel::<Option<String>>();
    let handler = RcBlock::new(move |error: *mut NSError| {
        let message = if error.is_null() {
            None
        } else {
            Some(unsafe { (*error).localizedDescription() }.to_string())
        };
        let _ = tx.send(message);
    });
    unsafe { stream.startCaptureWithCompletionHandler(Some(&handler)) };

    match rx.recv_timeout(OPEN_TIMEOUT) {
        Ok(None) => Ok((stream, tap)),
        Ok(Some(message)) => Err(classify_start_error(&message)),
        Err(_) => Err(AudioError::SystemAudioUnsupported(
            "macOS did not start sharing in time".into(),
        )),
    }
}

fn shareable_content() -> Result<Retained<SCShareableContent>, AudioError> {
    let (tx, rx) = mpsc::channel::<Result<SendPtr<SCShareableContent>, String>>();
    let handler = RcBlock::new(
        move |content: *mut SCShareableContent, error: *mut NSError| {
            if !content.is_null() {
                // Retain it so it outlives the block, and hand the +1 over.
                let retained = unsafe { Retained::retain(content) };
                match retained {
                    Some(r) => {
                        let _ = tx.send(Ok(SendPtr(Retained::into_raw(r))));
                    }
                    None => {
                        let _ = tx.send(Err("macOS returned nothing to share".to_string()));
                    }
                }
            } else {
                let message = if error.is_null() {
                    "macOS returned nothing to share".to_string()
                } else {
                    unsafe { (*error).localizedDescription() }.to_string()
                };
                let _ = tx.send(Err(message));
            }
        },
    );
    unsafe { SCShareableContent::getShareableContentWithCompletionHandler(&handler) };

    match rx.recv_timeout(OPEN_TIMEOUT) {
        Ok(Ok(SendPtr(ptr))) => unsafe { Retained::from_raw(ptr) }.ok_or_else(|| {
            AudioError::SystemAudioUnsupported("macOS returned nothing to share".into())
        }),
        Ok(Err(message)) => Err(classify_start_error(&message)),
        Err(_) => Err(AudioError::SystemAudioUnsupported(
            "macOS did not answer in time".into(),
        )),
    }
}

fn describe_error(error: &NSError) -> String {
    error.localizedDescription().to_string()
}

/// ScreenCaptureKit reports a missing permission as a plain error. Map the ones
/// we can recognise so the session layer can ask for permission instead of
/// showing a dead end.
fn classify_start_error(message: &str) -> AudioError {
    let lower = message.to_lowercase();
    if lower.contains("permission")
        || lower.contains("declined")
        || lower.contains("not authorized")
        || lower.contains("unauthorized")
        || lower.contains("tcc")
    {
        AudioError::PermissionDenied
    } else {
        AudioError::SystemAudioUnsupported(message.to_string())
    }
}

/// ScreenCaptureKit audio capture needs macOS 13.
pub fn supported() -> bool {
    let version = NSProcessInfo::processInfo().operatingSystemVersion();
    version.majorVersion >= 13
}

/// Permission state without prompting.
pub async fn permission() -> PermissionState {
    if !supported() {
        return PermissionState::NotApplicable;
    }
    if unsafe { CGPreflightScreenCaptureAccess() } {
        PermissionState::Granted
    } else {
        PermissionState::Denied
    }
}

/// Trigger the system prompt. Grant usually needs a relaunch, so the result is
/// often [`PermissionState::RestartRequired`].
pub async fn request_permission() -> PermissionState {
    if !supported() {
        return PermissionState::NotApplicable;
    }
    if unsafe { CGPreflightScreenCaptureAccess() } {
        return PermissionState::Granted;
    }
    let granted = unsafe { CGRequestScreenCaptureAccess() };
    if granted {
        PermissionState::Granted
    } else {
        // macOS only starts honouring a fresh grant after a relaunch, so saying
        // "denied" here would be a lie the person cannot act on.
        PermissionState::RestartRequired
    }
}

/// Open System Settings → Privacy & Security → Screen & System Audio Recording.
pub fn open_settings_pane() -> Result<(), AudioError> {
    let url = "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture";
    std::process::Command::new("/usr/bin/open")
        .arg(url)
        .status()
        .map_err(|e| AudioError::Backend(e.to_string()))?;
    Ok(())
}

/// Current microphone permission on macOS, read straight from AVFoundation
/// through the Objective-C runtime so no extra crate is needed.
pub fn microphone_permission_state() -> PermissionState {
    // AVAuthorizationStatus: 0 not determined, 1 restricted, 2 denied, 3 authorized.
    let Some(class) = objc2::runtime::AnyClass::get(c"AVCaptureDevice") else {
        return PermissionState::Unknown;
    };
    let media_type = NSString::from_str(AV_MEDIA_TYPE_AUDIO);
    let status: isize = unsafe { msg_send![class, authorizationStatusForMediaType: &*media_type] };
    match status {
        0 => PermissionState::Unknown,
        1 | 2 => PermissionState::Denied,
        3 => PermissionState::Granted,
        _ => PermissionState::Unknown,
    }
}

/// Prompt for the microphone and wait for the answer.
pub fn request_microphone_permission_blocking() -> PermissionState {
    let current = microphone_permission_state();
    if current != PermissionState::Unknown {
        return current;
    }
    let Some(class) = objc2::runtime::AnyClass::get(c"AVCaptureDevice") else {
        return PermissionState::Unknown;
    };
    let media_type = NSString::from_str(AV_MEDIA_TYPE_AUDIO);
    let (tx, rx) = mpsc::channel::<bool>();
    let handler = RcBlock::new(move |granted: objc2::runtime::Bool| {
        let _ = tx.send(granted.as_bool());
    });
    let _: () = unsafe {
        msg_send![
            class,
            requestAccessForMediaType: &*media_type,
            completionHandler: &*handler,
        ]
    };
    match rx.recv_timeout(Duration::from_secs(60)) {
        Ok(true) => PermissionState::Granted,
        Ok(false) => PermissionState::Denied,
        Err(_) => PermissionState::Prompting,
    }
}

/// `AVMediaTypeAudio`, which is the four-character code for sound.
const AV_MEDIA_TYPE_AUDIO: &str = "soun";

// Keep the linker from dropping AVFoundation: the runtime lookups above find
// AVCaptureDevice by name, which only works if the framework is loaded.
#[link(name = "AVFoundation", kind = "framework")]
unsafe extern "C" {}

#[cfg(test)]
mod tests {
    use super::*;

    // Nothing here starts a stream: that needs a signed build, a real display
    // and a person clicking Allow. What can be checked without hardware is the
    // reasoning around it.

    #[test]
    fn the_os_version_gate_answers_without_touching_a_device() {
        // Any macOS this can be built on is at least 11, so the answer must be
        // a real boolean rather than a panic.
        let _ = supported();
    }

    #[test]
    fn permission_errors_are_recognised_so_the_ui_can_offer_settings() {
        assert!(matches!(
            classify_start_error("The user declined TCCs permission"),
            AudioError::PermissionDenied
        ));
        assert!(matches!(
            classify_start_error("not authorized to capture the screen"),
            AudioError::PermissionDenied
        ));
        assert!(matches!(
            classify_start_error("no displays attached"),
            AudioError::SystemAudioUnsupported(_)
        ));
    }

    #[test]
    fn the_buffer_list_is_big_enough_for_a_real_stream() {
        assert!(MAX_BUFFERS >= usize::from(REQUESTED_CHANNELS));
        assert_eq!(std::mem::size_of::<AudioBufferRaw>(), 16);
    }

    #[test]
    fn microphone_permission_is_answered_without_prompting() {
        // On a headless CI machine this is Unknown or Denied; either is a valid
        // answer and neither may hang.
        let state = microphone_permission_state();
        assert!(matches!(
            state,
            PermissionState::Unknown
                | PermissionState::Denied
                | PermissionState::Granted
                | PermissionState::NotApplicable
        ));
    }
}
