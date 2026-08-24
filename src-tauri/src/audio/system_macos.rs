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
//! * **A stream output only calls back on the queue it was given.** Echo hands
//!   `addStreamOutput:type:sampleHandlerQueue:` its own serial queues and keeps
//!   them alive for the stream's life. A nil queue there produced exactly one
//!   symptom in the field: a stream that started with no error, never stopped,
//!   and delivered no audio for an entire meeting. Nothing above this layer can
//!   tell that apart from a working stream, so it must be right here.
//! * A screen output is registered next to the audio one and every video frame
//!   is dropped. Not because the frames are wanted, but because an `SCStream`
//!   always has a video side and audio-only registration is not a configuration
//!   Apple documents as working. It costs a 16x16 frame a second.
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
//! family is already in the tree for the rest of the macOS integration. All
//! CoreMedia and CoreAudio types come from the crate bindings too — this file
//! used to declare them by hand, and a mistake in those hand declarations is
//! exactly the kind of thing that made every buffer of a real meeting
//! unreadable (2026-08-24). The only functions still declared directly are the
//! two CoreGraphics permission preflights, which no crate in the tree binds.

#![cfg(target_os = "macos")]
// The protocol methods below have to keep their Objective-C selector spelling so
// the runtime can find them.
#![allow(non_snake_case)]

use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, AtomicU32, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use block2::RcBlock;
use dispatch2::{DispatchQueue, DispatchQueueAttr, DispatchRetained};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::{define_class, msg_send, AnyThread, DefinedClass};
use objc2_core_audio_types::{
    kAudioFormatFlagIsFloat, kAudioFormatFlagIsNonInterleaved, AudioBuffer, AudioBufferList,
};
use objc2_core_foundation::CFRetained;
use objc2_core_media::{
    kCMSampleBufferError_AllocationFailed, kCMSampleBufferError_AlreadyHasDataBuffer,
    kCMSampleBufferError_ArrayTooSmall, kCMSampleBufferError_BufferHasNoSampleSizes,
    kCMSampleBufferError_BufferHasNoSampleTimingInfo, kCMSampleBufferError_BufferNotReady,
    kCMSampleBufferError_CannotSubdivide, kCMSampleBufferError_DataCanceled,
    kCMSampleBufferError_DataFailed, kCMSampleBufferError_InvalidEntryCount,
    kCMSampleBufferError_InvalidMediaFormat, kCMSampleBufferError_InvalidMediaTypeForOperation,
    kCMSampleBufferError_InvalidSampleData, kCMSampleBufferError_Invalidated,
    kCMSampleBufferError_RequiredParameterMissing, kCMSampleBufferError_SampleIndexOutOfRange,
    kCMSampleBufferError_SampleTimingInfoInvalid,
    kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment,
    CMAudioFormatDescriptionGetStreamBasicDescription, CMBlockBuffer, CMSampleBuffer, CMTime,
    CMTimeFlags,
};
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

/// Video cannot be switched off on an `SCStream`: the configuration always
/// describes a video stream, and Apple's own sample registers a screen output
/// next to the audio one. So Echo registers both and throws every video frame
/// away. Small and slow, but not degenerate — a size or interval the framework
/// has to special-case is exactly the kind of thing that makes a stream start
/// and then deliver nothing.
const VIDEO_SIDE: usize = 16;
/// One 16x16 frame a second, all of them dropped. Effectively free.
const VIDEO_FRAME_INTERVAL: CMTime = CMTime {
    value: 1,
    timescale: 1,
    flags: CMTimeFlags::Valid,
    epoch: 0,
};

// ---------------------------------------------------------------------------
// The one part of CoreGraphics we still declare by hand (no crate binds it)
// ---------------------------------------------------------------------------

/// `AudioBufferList` is a variable-length C struct: a count followed by as many
/// `AudioBuffer`s as the stream has channels. The crate binding only declares
/// the first element, so [`AudioBufferListStorage`] reserves the rest.
/// ScreenCaptureKit never gives us more than a handful of channels, so a fixed
/// maximum is honest and avoids allocating inside the sample handler.
const MAX_BUFFERS: usize = 8;

/// Room for an `AudioBufferList` carrying up to [`MAX_BUFFERS`] buffers.
///
/// The crate's [`AudioBufferList`] declares `mBuffers: [AudioBuffer; 1]`
/// faithfully to the C header, where the array is really variable-length and
/// the caller is expected to allocate enough room behind it. `_extra` is that
/// room. `#[repr(C)]` keeps `_extra` contiguous with `list.mBuffers`, which
/// the layout tests pin down — this is exactly the kind of ABI assumption that
/// silently broke a real meeting (2026-08-24), so it stays under test.
#[repr(C)]
struct AudioBufferListStorage {
    list: AudioBufferList,
    _extra: [AudioBuffer; MAX_BUFFERS - 1],
}

impl AudioBufferListStorage {
    fn zeroed() -> Self {
        const EMPTY: AudioBuffer = AudioBuffer {
            mNumberChannels: 0,
            mDataByteSize: 0,
            mData: std::ptr::null_mut(),
        };
        Self {
            list: AudioBufferList {
                mNumberBuffers: 0,
                mBuffers: [EMPTY; 1],
            },
            _extra: [EMPTY; MAX_BUFFERS - 1],
        }
    }

    /// The pointer CoreMedia calls want. Derived from the whole struct, not
    /// just the `list` field, so writes into the extra buffers behind the
    /// header stay inside the pointer's provenance.
    fn list_ptr(&mut self) -> NonNull<AudioBufferList> {
        // SAFETY: `list` is the first field of a #[repr(C)] struct, so a
        // pointer to the struct is a valid pointer to it; `&mut self` is
        // never null.
        unsafe { NonNull::new_unchecked(std::ptr::addr_of_mut!(*self).cast::<AudioBufferList>()) }
    }

    /// The first `n` buffers as one slice. Always go through this rather than
    /// indexing `list.mBuffers` — the crate type declares one element, and
    /// anything past it lives in `_extra`.
    fn buffers(&self, n: usize) -> &[AudioBuffer] {
        let n = n.min(MAX_BUFFERS);
        let base = std::ptr::addr_of!(*self).cast::<u8>();
        // SAFETY: the layout tests pin `list.mBuffers` and `_extra` as one
        // contiguous [AudioBuffer; MAX_BUFFERS] region starting at this
        // offset, and `n` never exceeds MAX_BUFFERS.
        unsafe {
            let first = base.add(Self::BUFFERS_OFFSET).cast::<AudioBuffer>();
            std::slice::from_raw_parts(first, n)
        }
    }

    /// Mutable twin of [`Self::buffers`].
    fn buffers_mut(&mut self, n: usize) -> &mut [AudioBuffer] {
        let n = n.min(MAX_BUFFERS);
        let base = std::ptr::addr_of_mut!(*self).cast::<u8>();
        // SAFETY: same layout argument as `buffers`; `&mut self` guarantees
        // exclusive access to the whole region.
        unsafe {
            let first = base.add(Self::BUFFERS_OFFSET).cast::<AudioBuffer>();
            std::slice::from_raw_parts_mut(first, n)
        }
    }

    /// Where the buffer array starts, inside this struct.
    const BUFFERS_OFFSET: usize =
        std::mem::offset_of!(Self, list) + std::mem::offset_of!(AudioBufferList, mBuffers);
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

/// Names an `OSStatus` CoreMedia handed back from a failed buffer-list call.
/// The number alone means nothing to whoever reads the log next; the constant
/// name is the thing a search engine (or a memory of this file) can work with.
/// Zero-jargon doesn't apply here — this is a tracing line, not something a
/// person in a meeting ever sees.
// The kCMSampleBufferError_* constants keep Apple's own spelling so a search
// for the name in Apple's headers or docs finds it here too.
#[allow(non_upper_case_globals)]
fn sample_buffer_status_name(status: i32) -> &'static str {
    match status {
        kCMSampleBufferError_AllocationFailed => "AllocationFailed",
        kCMSampleBufferError_RequiredParameterMissing => "RequiredParameterMissing",
        kCMSampleBufferError_AlreadyHasDataBuffer => "AlreadyHasDataBuffer",
        kCMSampleBufferError_BufferNotReady => "BufferNotReady",
        kCMSampleBufferError_SampleIndexOutOfRange => "SampleIndexOutOfRange",
        kCMSampleBufferError_BufferHasNoSampleSizes => "BufferHasNoSampleSizes",
        kCMSampleBufferError_BufferHasNoSampleTimingInfo => "BufferHasNoSampleTimingInfo",
        kCMSampleBufferError_ArrayTooSmall => "ArrayTooSmall",
        kCMSampleBufferError_InvalidEntryCount => "InvalidEntryCount",
        kCMSampleBufferError_CannotSubdivide => "CannotSubdivide",
        kCMSampleBufferError_SampleTimingInfoInvalid => "SampleTimingInfoInvalid",
        kCMSampleBufferError_InvalidMediaTypeForOperation => "InvalidMediaTypeForOperation",
        kCMSampleBufferError_InvalidSampleData => "InvalidSampleData",
        kCMSampleBufferError_InvalidMediaFormat => "InvalidMediaFormat",
        kCMSampleBufferError_Invalidated => "Invalidated",
        kCMSampleBufferError_DataFailed => "DataFailed",
        kCMSampleBufferError_DataCanceled => "DataCanceled",
        _ => "unknown",
    }
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
    /// Landing area for the PCM CoreMedia copies out of each sample buffer,
    /// before the mixdown into `scratch`. Grow-only, so after the first buffer
    /// the handler never allocates.
    pcm: Mutex<Vec<f32>>,
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
    /// Counters, not audio. A stream that starts and then delivers nothing
    /// looks identical to a working one from the outside, so the next real
    /// meeting has to be able to say *where* it went quiet: never called at
    /// all, called with video only, or called with buffers we refused.
    counts: BufferCounts,
    /// Set once the first audio buffer's format has been written to the log.
    format_logged: AtomicBool,
    /// Set once the first buffer-list refusal for this stream has been
    /// written to the log. Every subsequent refusal still counts toward
    /// `unreadable`; it just doesn't get its own log line, or a failing
    /// stream would drown the log in a repeat of the same OSStatus.
    read_failure_logged: AtomicBool,
    /// Set once this stream has logged that the fallback read rescued a
    /// buffer the primary read refused. Separate from
    /// `read_failure_logged`, which is reserved for the double refusal: a
    /// stream that is rescued for an hour and then loses a buffer outright
    /// deserves both lines.
    fallback_logged: AtomicBool,
    /// The OSStatus CoreMedia gave back for the most recent refusal of the
    /// primary (copy) read — recorded on *every* copy refusal, whether or
    /// not the fallback then rescued the buffer, so a meeting served
    /// entirely by the fallback still names its reason in the diagnostics.
    /// Zero means never failed. This is deliberately not reset by
    /// `reopen()` — see the comment there.
    last_read_status: AtomicI32,
    /// Same, for the fallback (retained-block-buffer) read. Set only when
    /// the fallback is *also* refused, so it is only ever non-zero together
    /// with `last_read_status`.
    last_list_status: AtomicI32,
}

/// How many sample buffers arrived and what became of them.
#[derive(Debug, Default)]
struct BufferCounts {
    audio: AtomicU64,
    /// Video frames, which we drop on purpose.
    other: AtomicU64,
    /// Not 32-bit float, so we did not guess at it.
    not_float: AtomicU64,
    /// CoreMedia would not let us read it: no usable format description, an
    /// impossible channel or frame count, or both read paths refused.
    unreadable: AtomicU64,
    /// The primary (copy) read was refused but the fallback read rescued the
    /// buffer. No audio was lost — but a stream living off this counter is
    /// degraded, and 2026-08-24 is the proof that a degraded read path must
    /// never be indistinguishable from a healthy one.
    fallback: AtomicU64,
    /// Arrived before its data did. CoreMedia says which; we count it apart
    /// from `unreadable` because "the data never became ready" points at the
    /// producer, not at this file's reading of it.
    not_ready: AtomicU64,
    /// Carried no samples.
    empty: AtomicU64,
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

    /// One line of counters for the diagnostics log. Lives here, not on
    /// [`SystemCapture`], so a test can build a `SystemShared` directly and
    /// check the line without opening a stream.
    fn diagnostics_summary(&self) -> String {
        diagnostics_line(
            self.counts.audio.load(Ordering::Relaxed),
            self.counts.other.load(Ordering::Relaxed),
            self.counts.not_float.load(Ordering::Relaxed),
            self.counts.unreadable.load(Ordering::Relaxed),
            self.counts.fallback.load(Ordering::Relaxed),
            self.counts.not_ready.load(Ordering::Relaxed),
            self.counts.empty.load(Ordering::Relaxed),
            self.counters.pushed_samples(),
            self.counters.dropped_samples(),
            self.rate.load(Ordering::Relaxed),
            self.last_read_status.load(Ordering::Relaxed),
            self.last_list_status.load(Ordering::Relaxed),
        )
    }

    /// Give a fresh stream its own once-per-stream log lines. Called by
    /// [`SystemCapture::reopen`]; lives here so a test can exercise it
    /// without opening a stream.
    ///
    /// A new stream gets to describe its own first buffer: the format can
    /// change under us when the output device does. And it gets to earn its
    /// own read-failure warning rather than staying silent because the last
    /// stream already used its one line. The buffer counts and the last
    /// refusal statuses are deliberately left alone: they are the record of
    /// the whole meeting, not of one particular stream, and the diagnostics
    /// line at the end should still be able to say what went wrong even if
    /// the last reopen happened to succeed.
    fn reset_per_stream_logs(&self) {
        self.format_logged.store(false, Ordering::Relaxed);
        self.read_failure_logged.store(false, Ordering::Relaxed);
        self.fallback_logged.store(false, Ordering::Relaxed);
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
                // Video frames are never consumed: they exist only because an
                // `SCStream` always has a video side, and dropping them keeps
                // the cost at nothing. Counted, because "macOS called us with
                // video but never with audio" is a completely different
                // diagnosis from "macOS never called us".
                self.ivars().counts.other.fetch_add(1, Ordering::Relaxed);
                return;
            }
            self.ivars().counts.audio.fetch_add(1, Ordering::Relaxed);
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
    // Format first: if this is not 32-bit float we do not guess.
    // SAFETY: `sbuf` is a valid sample buffer for the whole callback; the
    // description comes back retained, and the ASBD pointer it hands out is
    // valid as long as the description is — the struct is copied out before
    // the retain is dropped.
    let described = unsafe { sbuf.format_description() }.and_then(|desc| {
        let asbd = unsafe { CMAudioFormatDescriptionGetStreamBasicDescription(&desc) };
        (!asbd.is_null()).then(|| unsafe { *asbd })
    });

    let rate = match described {
        Some(asbd) if asbd.mSampleRate > 0.0 => asbd.mSampleRate as u32,
        _ => REQUESTED_RATE,
    };
    let is_float = described.is_some_and(|asbd| asbd.mFormatFlags & kAudioFormatFlagIsFloat != 0);

    // Exactly once per stream, and never audio content: what macOS actually
    // negotiated, so the next real meeting's log either proves this path works
    // or says which field is wrong. One log line on a dispatch queue that is
    // already taking two mutexes below costs nothing.
    if !shared.format_logged.swap(true, Ordering::Relaxed) {
        match described {
            Some(asbd) => tracing::info!(
                target: "echo::audio",
                rate,
                channels = asbd.mChannelsPerFrame,
                bits = asbd.mBitsPerChannel,
                format_flags = asbd.mFormatFlags,
                is_float,
                "first system-audio buffer arrived"
            ),
            None => tracing::info!(
                target: "echo::audio",
                "first system-audio buffer arrived with no format description"
            ),
        }
    }

    // A buffer that does not say what format it is in cannot be read. The old
    // code silently assumed float and carried on; now it counts and says so.
    let Some(asbd) = described else {
        shared.counts.unreadable.fetch_add(1, Ordering::Relaxed);
        if !shared.read_failure_logged.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                target: "echo::audio",
                "an audio buffer arrived without a format description, so it cannot be read"
            );
        }
        return;
    };

    if !is_float || asbd.mBitsPerChannel != 32 {
        shared.counts.not_float.fetch_add(1, Ordering::Relaxed);
        return;
    }
    let non_interleaved = asbd.mFormatFlags & kAudioFormatFlagIsNonInterleaved != 0;
    let channels = asbd.mChannelsPerFrame as usize;
    if channels == 0 || channels > MAX_BUFFERS {
        shared.counts.unreadable.fetch_add(1, Ordering::Relaxed);
        if !shared.read_failure_logged.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                target: "echo::audio",
                channels,
                max_channels = MAX_BUFFERS,
                "the stream announced a channel count the buffer list cannot hold"
            );
        }
        return;
    }
    shared.rate.store(rate, Ordering::Relaxed);

    // SAFETY: plain accessor on a valid sample buffer.
    if !unsafe { sbuf.data_is_ready() } {
        shared.counts.not_ready.fetch_add(1, Ordering::Relaxed);
        return;
    }
    // SAFETY: plain accessor on a valid sample buffer.
    let num_samples = unsafe { sbuf.num_samples() };
    if num_samples <= 0 {
        shared.counts.empty.fetch_add(1, Ordering::Relaxed);
        return;
    }
    // The copy call takes the frame count as an i32. A count that does not
    // fit is not audio we can represent anyway; refuse it rather than let the
    // cast wrap into a nonsense request.
    let Ok(frames) = i32::try_from(num_samples) else {
        shared.counts.unreadable.fetch_add(1, Ordering::Relaxed);
        if !shared.read_failure_logged.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                target: "echo::audio",
                num_samples,
                "the buffer claims a frame count too large to read"
            );
        }
        return;
    };
    let frames = frames as usize;

    let mut scratch = shared.scratch.lock().unwrap_or_else(|e| e.into_inner());
    let copy_status = {
        let mut pcm = shared.pcm.lock().unwrap_or_else(|e| e.into_inner());
        read_by_copying(
            sbuf,
            non_interleaved,
            channels,
            frames,
            &mut pcm,
            &mut scratch,
        )
    };
    if copy_status != 0 {
        // Record the refusal before anything else: even if the fallback
        // rescues every single buffer, the diagnostics line must still be
        // able to say the primary read never worked and why. A degraded
        // stream that looks healthy is this file's original sin (2026-08-24).
        shared
            .last_read_status
            .store(copy_status, Ordering::Relaxed);
        // The primary read was refused; ask for the buffer list that
        // references the sample buffer's own storage instead. The 2026-08-24
        // meeting failed on (an unsound hand-rolled version of) this second
        // call on every one of 500 buffers, which is why the copy is primary
        // now and why a double refusal names both statuses below.
        match read_via_block_buffer(sbuf, frames, &mut scratch) {
            Ok(()) => {
                shared.counts.fallback.fetch_add(1, Ordering::Relaxed);
                // Once per stream: the audio survived, but the stream is
                // running on its second-choice read and the log has to say
                // so while the meeting is still happening, not only in the
                // end-of-meeting counters.
                if !shared.fallback_logged.swap(true, Ordering::Relaxed) {
                    tracing::warn!(
                        target: "echo::audio",
                        copy_status,
                        copy_status_name = sample_buffer_status_name(copy_status),
                        "macOS refused the primary read of an audio buffer; the fallback read is carrying the audio"
                    );
                }
            }
            Err((list_status, needed)) => {
                shared.counts.unreadable.fetch_add(1, Ordering::Relaxed);
                shared
                    .last_list_status
                    .store(list_status, Ordering::Relaxed);
                // Once per stream, and named rather than numbered. Every later
                // refusal still counts toward `unreadable`; it just doesn't get
                // its own line, or a stream that fails on every buffer would
                // drown the log in the same status a thousand times over.
                if !shared.read_failure_logged.swap(true, Ordering::Relaxed) {
                    tracing::warn!(
                        target: "echo::audio",
                        copy_status,
                        copy_status_name = sample_buffer_status_name(copy_status),
                        list_status,
                        list_status_name = sample_buffer_status_name(list_status),
                        needed_size = needed,
                        storage_size = std::mem::size_of::<AudioBufferListStorage>(),
                        "macOS refused both ways of reading an audio buffer"
                    );
                }
                return;
            }
        }
    }
    if scratch.is_empty() {
        shared.counts.empty.fetch_add(1, Ordering::Relaxed);
        return;
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

/// Primary read: have CoreMedia copy the buffer's PCM into `pcm`, then mix it
/// down to mono in `mono`. Returns the OSStatus of the copy call; `mono` is
/// only meaningful when that is 0. `pcm` grows to fit and never shrinks, so
/// after the first buffer this path allocates nothing.
fn read_by_copying(
    sbuf: &CMSampleBuffer,
    non_interleaved: bool,
    channels: usize,
    frames: usize,
    pcm: &mut Vec<f32>,
    mono: &mut Vec<f32>,
) -> i32 {
    let total = frames * channels;
    if pcm.len() < total {
        pcm.resize(total, 0.0);
    }
    let data = &mut pcm[..total];

    // Describe `data` to CoreMedia the way the format says the samples are
    // laid out: one buffer per channel when non-interleaved, one buffer of
    // frame-sized groups when interleaved.
    let mut storage = AudioBufferListStorage::zeroed();
    if non_interleaved {
        storage.list.mNumberBuffers = channels as u32;
        let byte_size = (frames * std::mem::size_of::<f32>()) as u32;
        for (chunk, buffer) in data
            .chunks_exact_mut(frames)
            .zip(storage.buffers_mut(channels).iter_mut())
        {
            *buffer = AudioBuffer {
                mNumberChannels: 1,
                mDataByteSize: byte_size,
                mData: chunk.as_mut_ptr().cast(),
            };
        }
    } else {
        storage.list.mNumberBuffers = 1;
        storage.buffers_mut(1)[0] = AudioBuffer {
            mNumberChannels: channels as u32,
            mDataByteSize: (total * std::mem::size_of::<f32>()) as u32,
            mData: data.as_mut_ptr().cast(),
        };
    }

    // SAFETY: every buffer in the list points into `data`, which lives for
    // the whole call and is exactly as large as the sizes written above;
    // `frames` was bounds-checked against i32 by the caller.
    let status =
        unsafe { sbuf.copy_pcm_data_into_audio_buffer_list(0, frames as i32, storage.list_ptr()) };
    if status != 0 {
        return status;
    }

    if non_interleaved {
        let mut planes: [&[f32]; MAX_BUFFERS] = [&[]; MAX_BUFFERS];
        for (plane, chunk) in planes.iter_mut().zip(data.chunks_exact(frames)) {
            *plane = chunk;
        }
        mix_planar_into(mono, &planes[..channels], frames);
    } else {
        mix_interleaved_into(mono, data, channels);
    }
    0
}

/// Fallback read: ask CoreMedia for an `AudioBufferList` whose buffers point
/// into the sample buffer's own storage, kept alive by a retained block
/// buffer, and mix that down to mono. On failure returns the OSStatus and the
/// buffer-list size CoreMedia said it actually needed.
///
/// This is the call the 2026-08-24 meeting lost 500 of 500 buffers to, back
/// when a hand-declared version of it was the whole read path. It stays as
/// the fallback — and is exercised directly by tests — so a refusal of the
/// copy call still has a second chance instead of a silent gap.
fn read_via_block_buffer(
    sbuf: &CMSampleBuffer,
    frames: usize,
    mono: &mut Vec<f32>,
) -> Result<(), (i32, usize)> {
    // Ask for the size first. CoreMedia does NOT treat `buffer_list_size` as
    // a capacity: handing it storage larger than the list actually needs
    // fails with ArrayTooSmall, of all things — measured right here on a
    // synthetic stereo buffer (needed=40, provided=136, status=-12737). The
    // exact reported size is accepted. This is the same shape of refusal the
    // 2026-08-24 meeting hit 500 times out of 500: the old hand-rolled call
    // always handed over its full fixed-size storage and lost every buffer.
    let mut needed: usize = 0;
    // SAFETY: a null buffer list with size 0 is the documented way to ask
    // only for the needed size; the out-pointer is valid for the write.
    let probe = unsafe {
        sbuf.audio_buffer_list_with_retained_block_buffer(
            &mut needed,
            std::ptr::null_mut(),
            0,
            None,
            None,
            kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment,
            std::ptr::null_mut(),
        )
    };
    if needed == 0 || needed > std::mem::size_of::<AudioBufferListStorage>() {
        // Either the probe told us nothing, or the list genuinely does not
        // fit (more than MAX_BUFFERS channels). Hand back the probe's status
        // so the log can name it.
        return Err((probe, needed));
    }

    let mut storage = AudioBufferListStorage::zeroed();
    let mut block_buffer: *mut CMBlockBuffer = std::ptr::null_mut();
    // SAFETY: `needed` was checked against the real size of `storage` just
    // above, so CoreMedia writes only into memory we own; the out-pointers
    // are valid for writes.
    let status = unsafe {
        sbuf.audio_buffer_list_with_retained_block_buffer(
            std::ptr::null_mut(),
            storage.list_ptr().as_ptr(),
            needed,
            None,
            None,
            kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment,
            &mut block_buffer,
        )
    };
    // Wrap the block buffer immediately so every return path below releases
    // it. The old hand-rolled CFRelease could be skipped by an early return.
    // SAFETY: when non-null, this is a +1 reference the call handed over.
    let _block_buffer = NonNull::new(block_buffer).map(|b| unsafe { CFRetained::from_raw(b) });
    if status != 0 {
        return Err((status, needed));
    }

    let count = (storage.list.mNumberBuffers as usize).min(MAX_BUFFERS);
    let buffers = storage.buffers(count);
    if count > 1 {
        // One buffer per channel.
        let mut planes: [&[f32]; MAX_BUFFERS] = [&[]; MAX_BUFFERS];
        let mut used = 0usize;
        for buffer in buffers {
            if buffer.mData.is_null() {
                continue;
            }
            let n = (buffer.mDataByteSize as usize / std::mem::size_of::<f32>()).min(frames);
            // SAFETY: the block buffer guard keeps the pointed-to samples
            // alive past the mixdown, and `n` never exceeds what
            // mDataByteSize says is there.
            planes[used] = unsafe { std::slice::from_raw_parts(buffer.mData as *const f32, n) };
            used += 1;
        }
        mix_planar_into(mono, &planes[..used], frames);
    } else if count == 1 && !buffers[0].mData.is_null() {
        // One buffer of interleaved frames (or plain mono).
        let buffer = &buffers[0];
        let channels = (buffer.mNumberChannels as usize).max(1);
        let n = (buffer.mDataByteSize as usize / std::mem::size_of::<f32>()).min(frames * channels);
        // SAFETY: same lifetime and bounds argument as above.
        let samples = unsafe { std::slice::from_raw_parts(buffer.mData as *const f32, n) };
        mix_interleaved_into(mono, samples, channels);
    } else {
        mono.clear();
    }
    Ok(())
}

/// Mix one buffer per channel down to mono: sum the channels and, when more
/// than one contributed, scale by 1/channels. A single channel passes through
/// untouched. Channels shorter than `frames` contribute what they have.
fn mix_planar_into(mono: &mut Vec<f32>, planes: &[&[f32]], frames: usize) {
    mono.clear();
    mono.resize(frames, 0.0);
    for plane in planes {
        for (dst, src) in mono.iter_mut().zip(plane.iter()) {
            *dst += *src;
        }
    }
    if planes.len() > 1 {
        let scale = 1.0 / planes.len() as f32;
        for s in mono.iter_mut() {
            *s *= scale;
        }
    }
}

/// Mix interleaved frames down to mono: average each frame's channels. Mono
/// input passes through untouched. A trailing partial frame is dropped —
/// it was never a whole frame to begin with.
fn mix_interleaved_into(mono: &mut Vec<f32>, samples: &[f32], channels: usize) {
    mono.clear();
    if channels == 0 {
        return;
    }
    if channels == 1 {
        mono.extend_from_slice(samples);
        return;
    }
    let scale = 1.0 / channels as f32;
    mono.extend(
        samples
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() * scale),
    );
}

fn presentation_us(sbuf: &CMSampleBuffer) -> i64 {
    let time = unsafe { sbuf.presentation_time_stamp() };
    if time.timescale == 0 {
        return 0;
    }
    time.value.saturating_mul(1_000_000) / i64::from(time.timescale)
}

/// Builds the one-line diagnostics summary, pulled out of [`SystemCapture::diagnostics`]
/// so the format can be tested without a stream, a display, or anyone's
/// permission. `read_status` (the copy read) and `list_status` (the fallback
/// read) are 0 when that call has never failed on this stream.
#[allow(clippy::too_many_arguments)]
fn diagnostics_line(
    audio_buffers: u64,
    video_buffers: u64,
    not_float: u64,
    unreadable: u64,
    fallback: u64,
    not_ready: u64,
    empty: u64,
    pushed_samples: u64,
    dropped_samples: u64,
    rate: u32,
    read_status: i32,
    list_status: i32,
) -> String {
    let mut line = format!(
        "audio_buffers={audio_buffers} video_buffers={video_buffers} not_float={not_float} unreadable={unreadable} fallback={fallback} not_ready={not_ready} empty={empty} pushed_samples={pushed_samples} dropped_samples={dropped_samples} rate={rate}"
    );
    use std::fmt::Write as _;
    if read_status != 0 {
        let _ = write!(
            line,
            " read_status={read_status}({})",
            sample_buffer_status_name(read_status)
        );
    }
    if list_status != 0 {
        let _ = write!(
            line,
            " list_status={list_status}({})",
            sample_buffer_status_name(list_status)
        );
    }
    line
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
            // Pre-sized to a full second of stereo — far more than one
            // callback's worth — so the handler never allocates once running.
            pcm: Mutex::new(Vec::with_capacity(
                REQUESTED_RATE as usize * usize::from(REQUESTED_CHANNELS),
            )),
            first_pts_us: AtomicI64::new(i64::MIN),
            first_clock_ms: AtomicI64::new(0),
            rate: AtomicU32::new(REQUESTED_RATE),
            stopped_reason: Mutex::new(None),
            stopped: AtomicBool::new(false),
            counts: BufferCounts::default(),
            format_logged: AtomicBool::new(false),
            read_failure_logged: AtomicBool::new(false),
            fallback_logged: AtomicBool::new(false),
            last_read_status: AtomicI32::new(0),
            last_list_status: AtomicI32::new(0),
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

    /// One line of counters for the diagnostics log, so a silent channel can say
    /// *where* it went silent. Never audio content.
    ///
    /// Read it as a funnel: `audio_buffers=0 video_buffers=0` means macOS never
    /// called back at all; `audio_buffers=0 video_buffers>0` means the stream is
    /// running but carries no audio; a non-zero `not_float`/`unreadable`/`empty`
    /// means the buffers arrived and this file threw them away; a non-zero
    /// `fallback` means the audio survived, but only because the second-choice
    /// read rescued buffers the primary read refused (`read_status` says why).
    pub fn diagnostics(&self) -> String {
        self.shared.diagnostics_summary()
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
        // The once-per-stream log lines start over; the counts and last
        // refusal statuses stay, because they are the record of the whole
        // meeting. The reasoning lives on the method.
        self.shared.reset_per_stream_logs();
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
    let parts = match start_stream(&shared) {
        Ok(parts) => {
            let _ = ready.send(Ok(()));
            parts
        }
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };
    let StreamParts {
        stream,
        tap,
        audio_queue,
        video_queue,
        screen_registered,
    } = parts;

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
    if screen_registered {
        let _ = unsafe {
            stream.removeStreamOutput_type_error(
                ProtocolObject::from_ref(&*tap),
                SCStreamOutputType::Screen,
            )
        };
    }
    drop(tap);
    drop(stream);
    // The queues go last: ScreenCaptureKit was dispatching onto them until the
    // outputs came off, and a queue released while a block is still in flight is
    // a crash rather than a missing recording.
    drop(audio_queue);
    drop(video_queue);
}

/// Everything the stream needs to stay alive, held on the worker thread for the
/// life of the capture.
///
/// The queues are in here for a reason. `addStreamOutput:type:sampleHandlerQueue:`
/// is documented as delivering sample buffers "on the provided queue", and
/// ScreenCaptureKit does not keep it alive for you — a queue dropped after start
/// is a stream that runs and delivers nothing, which is precisely the failure
/// this file exists to have fixed.
struct StreamParts {
    stream: Retained<SCStream>,
    tap: Retained<AudioTap>,
    audio_queue: DispatchRetained<DispatchQueue>,
    video_queue: DispatchRetained<DispatchQueue>,
    /// Whether the screen output was accepted, so teardown removes exactly what
    /// was added.
    screen_registered: bool,
}

/// The stream configuration Echo asks for. Separate from [`start_stream`] so a
/// test can check every field round-trips through ScreenCaptureKit without a
/// display, a signed build or anybody's permission.
fn audio_configuration() -> Retained<SCStreamConfiguration> {
    let config = unsafe { SCStreamConfiguration::new() };
    unsafe {
        config.setCapturesAudio(true);
        config.setExcludesCurrentProcessAudio(true);
        config.setSampleRate(REQUESTED_RATE as isize);
        config.setChannelCount(REQUESTED_CHANNELS as isize);
        // Video cannot be switched off, so make it small and slow rather than
        // degenerate, and never read a frame.
        config.setWidth(VIDEO_SIDE);
        config.setHeight(VIDEO_SIDE);
        config.setMinimumFrameInterval(VIDEO_FRAME_INTERVAL);
        config.setQueueDepth(6);
    }
    config
}

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

    let config = audio_configuration();

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

    // ScreenCaptureKit calls a stream output back **on the queue handed to
    // `addStreamOutput:type:sampleHandlerQueue:`**. There is no documented
    // fallback for nil, and Apple's own sample always supplies one — a stream
    // registered with nil starts cleanly, reports no error, never stops, and
    // never delivers a single buffer. Echo owns two serial queues instead, and
    // keeps them alive for the stream's life.
    let audio_queue = DispatchQueue::new("dev.echo.system-audio", DispatchQueueAttr::SERIAL);
    let video_queue = DispatchQueue::new("dev.echo.system-audio.video", DispatchQueueAttr::SERIAL);

    let output: &ProtocolObject<dyn SCStreamOutput> = ProtocolObject::from_ref(&*tap);
    unsafe {
        stream.addStreamOutput_type_sampleHandlerQueue_error(
            output,
            SCStreamOutputType::Audio,
            Some(&audio_queue),
        )
    }
    .map_err(|e| AudioError::Backend(describe_error(&e)))?;

    // The screen output is not wanted, it is insurance: an `SCStream` always has
    // a video side, and Apple's sample registers a screen output alongside the
    // audio one. Audio-only registration is not a configuration Apple documents
    // as working, and this path cannot be re-tested quickly, so register both
    // and throw every frame away. A refusal here is not fatal — audio is the
    // whole point.
    let screen_registered = match unsafe {
        stream.addStreamOutput_type_sampleHandlerQueue_error(
            output,
            SCStreamOutputType::Screen,
            Some(&video_queue),
        )
    } {
        Ok(()) => true,
        Err(e) => {
            tracing::debug!(
                target: "echo::audio",
                "the throwaway video output was refused, continuing with audio only: {}",
                describe_error(&e)
            );
            false
        }
    };

    // `startCaptureWithCompletionHandler:` returns immediately; the *completion*
    // carries the verdict. Treating the call itself as success is how a stream
    // that failed its permission re-check ends up looking like a working one.
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
        Ok(None) => {
            log_negotiated_configuration(&config, screen_registered);
            Ok(StreamParts {
                stream,
                tap,
                audio_queue,
                video_queue,
                screen_registered,
            })
        }
        Ok(Some(message)) => Err(classify_start_error(&message)),
        Err(_) => Err(AudioError::SystemAudioUnsupported(
            "macOS did not start sharing in time".into(),
        )),
    }
}

/// What macOS agreed to, read back off the configuration rather than repeated
/// from what we asked for. This is the line that tells the next real meeting
/// whether the stream was set up right, before any buffer has arrived.
fn log_negotiated_configuration(config: &SCStreamConfiguration, screen_registered: bool) {
    let (sample_rate, channel_count, captures_audio, excludes_own_audio, queue_depth) = unsafe {
        (
            config.sampleRate(),
            config.channelCount(),
            config.capturesAudio(),
            config.excludesCurrentProcessAudio(),
            config.queueDepth(),
        )
    };
    tracing::info!(
        target: "echo::audio",
        sample_rate,
        channel_count,
        captures_audio,
        excludes_own_audio,
        queue_depth,
        screen_output_registered = screen_registered,
        sample_handler_queue = "dev.echo.system-audio",
        "system audio stream started"
    );
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

/// Whether this process has already asked macOS for screen capture. The
/// preflight API cannot tell "never asked" from "asked and refused", and the
/// two need different UI: an Allow button the first time, a road to System
/// Settings after.
static SCREEN_ACCESS_REQUESTED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Permission state without prompting.
pub async fn permission() -> PermissionState {
    if !supported() {
        return PermissionState::NotApplicable;
    }
    if unsafe { CGPreflightScreenCaptureAccess() } {
        PermissionState::Granted
    } else if !SCREEN_ACCESS_REQUESTED.load(Ordering::SeqCst) {
        // Not granted, but nothing has been asked yet this run: the person
        // needs the Allow button, not a hunt through System Settings for an
        // app that is not even listed there.
        PermissionState::Unknown
    } else {
        PermissionState::Denied
    }
}

/// Ask macOS for screen capture. This is also what makes Echo APPEAR in
/// System Settings → Screen & System Audio Recording: an app is only listed
/// there once it has actually attempted capture, so both calls below are made
/// even though they are expected to say no while permission is missing.
pub async fn request_permission() -> PermissionState {
    if !supported() {
        return PermissionState::NotApplicable;
    }
    if unsafe { CGPreflightScreenCaptureAccess() } {
        return PermissionState::Granted;
    }
    SCREEN_ACCESS_REQUESTED.store(true, Ordering::SeqCst);

    // Shows the system dialog (once per launch at most), and registers the
    // app with the permission system.
    let granted = unsafe { CGRequestScreenCaptureAccess() };

    // Belt to that braces: actually asking ScreenCaptureKit for content is
    // the documented trigger for both the prompt and the System Settings
    // listing on newer macOS. Expected to fail while permission is missing —
    // registration is the point. Bounded by OPEN_TIMEOUT, off the async
    // runtime's threads.
    let _ = tokio::task::spawn_blocking(|| {
        let _ = shareable_content();
    })
    .await;

    if granted {
        PermissionState::Granted
    } else {
        // Now Echo is listed in System Settings with its switch off; the
        // Denied card is the one that points there. macOS itself offers to
        // reopen Echo when the switch is flipped.
        PermissionState::Denied
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

    use objc2_core_audio_types::{
        kAudioFormatFlagIsPacked, kAudioFormatFlagIsSignedInteger, kAudioFormatLinearPCM,
        AudioStreamBasicDescription,
    };
    use objc2_core_media::{
        kCMBlockBufferAssureMemoryNowFlag, CMAudioFormatDescriptionCreate, CMFormatDescription,
        CMItemCount, CMSampleTimingInfo,
    };

    // Nothing here starts a stream: that needs a signed build, a real display
    // and a person clicking Allow. But sample buffers themselves are plain
    // CoreMedia objects, so the whole read path — the exact code that lost a
    // real meeting's system channel (2026-08-24) — is exercised below on
    // synthetic buffers, headless.

    /// A `SystemShared` exactly as `open_blocking` builds one, so a test can
    /// feed `handle_audio` and read what comes out the ring's other end.
    fn shared_for_tests() -> (Arc<SystemShared>, RingConsumer) {
        let (producer, consumer, counters) =
            hand_over(REQUESTED_RATE, REQUESTED_CHANNELS, Instant::now());
        let shared = Arc::new(SystemShared {
            producer: Mutex::new(producer),
            counters,
            scratch: Mutex::new(Vec::new()),
            pcm: Mutex::new(Vec::new()),
            first_pts_us: AtomicI64::new(i64::MIN),
            first_clock_ms: AtomicI64::new(0),
            rate: AtomicU32::new(REQUESTED_RATE),
            stopped_reason: Mutex::new(None),
            stopped: AtomicBool::new(false),
            counts: BufferCounts::default(),
            format_logged: AtomicBool::new(false),
            read_failure_logged: AtomicBool::new(false),
            fallback_logged: AtomicBool::new(false),
            last_read_status: AtomicI32::new(0),
            last_list_status: AtomicI32::new(0),
        });
        (shared, consumer)
    }

    fn drained(consumer: &mut RingConsumer) -> Vec<f32> {
        let mut out = Vec::new();
        consumer.drain_into(&mut out);
        out
    }

    fn f32_bytes(samples: &[f32]) -> Vec<u8> {
        samples.iter().flat_map(|s| s.to_ne_bytes()).collect()
    }

    /// The format ScreenCaptureKit actually negotiates: packed 32-bit float
    /// linear PCM (`format_flags=41` with the non-interleaved bit, as the
    /// 2026-08-24 meeting's log recorded).
    fn float_asbd(channels: u32, non_interleaved: bool) -> AudioStreamBasicDescription {
        let mut flags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
        let bytes_per_frame = if non_interleaved { 4 } else { 4 * channels };
        if non_interleaved {
            flags |= kAudioFormatFlagIsNonInterleaved;
        }
        AudioStreamBasicDescription {
            mSampleRate: 48_000.0,
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: flags,
            mBytesPerPacket: bytes_per_frame,
            mFramesPerPacket: 1,
            mBytesPerFrame: bytes_per_frame,
            mChannelsPerFrame: channels,
            mBitsPerChannel: 32,
            mReserved: 0,
        }
    }

    fn asbd_format_description(
        mut asbd: AudioStreamBasicDescription,
    ) -> CFRetained<CMFormatDescription> {
        let mut out: *const CMFormatDescription = std::ptr::null();
        // SAFETY: `asbd` is a valid stack ASBD which the call copies; `out`
        // is valid for the write.
        let status = unsafe {
            CMAudioFormatDescriptionCreate(
                None,
                NonNull::from(&mut asbd),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                None,
                NonNull::from(&mut out),
            )
        };
        assert_eq!(status, 0, "CoreMedia refused the format description");
        // SAFETY: on success the call hands back a +1 reference.
        unsafe { CFRetained::from_raw(NonNull::new(out.cast_mut()).expect("no description out")) }
    }

    fn audio_timing() -> CMSampleTimingInfo {
        let clock = CMTime {
            value: 0,
            timescale: 48_000,
            flags: CMTimeFlags::Valid,
            epoch: 0,
        };
        CMSampleTimingInfo {
            duration: CMTime { value: 1, ..clock },
            presentationTimeStamp: clock,
            decodeTimeStamp: CMTime {
                value: 0,
                timescale: 0,
                flags: CMTimeFlags::empty(),
                epoch: 0,
            },
        }
    }

    /// Builds a real `CMSampleBuffer` the way ScreenCaptureKit would deliver
    /// one: a format description plus PCM attached through the write-side
    /// twin of the read API, so CoreMedia itself owns the packing. `planes`
    /// is one byte buffer per channel when non-interleaved, or a single
    /// interleaved byte buffer; empty `planes` makes a dataless marker
    /// buffer. Attaching data this way also marks it ready.
    fn synthetic_buffer(
        asbd: AudioStreamBasicDescription,
        frames: usize,
        planes: &[&[u8]],
    ) -> CFRetained<CMSampleBuffer> {
        let desc = asbd_format_description(asbd);
        let timing = audio_timing();
        let (timing_count, timing_ptr): (CMItemCount, *const CMSampleTimingInfo) = if frames > 0 {
            (1, &timing)
        } else {
            (0, std::ptr::null())
        };
        let mut raw: *mut CMSampleBuffer = std::ptr::null_mut();
        // SAFETY: the timing array pointer covers `timing_count` entries, the
        // size array is empty, and the out-pointer is valid for the write. A
        // dataless buffer may be created already-ready (a marker buffer).
        let status = unsafe {
            CMSampleBuffer::create(
                None,
                None,
                planes.is_empty(),
                None,
                std::ptr::null_mut(),
                Some(&desc),
                frames as CMItemCount,
                timing_count,
                timing_ptr,
                0,
                std::ptr::null(),
                NonNull::from(&mut raw),
            )
        };
        assert_eq!(status, 0, "CoreMedia refused to create the sample buffer");
        // SAFETY: on success the call hands back a +1 reference.
        let sbuf = unsafe { CFRetained::from_raw(NonNull::new(raw).expect("no buffer out")) };

        if !planes.is_empty() {
            let channels_per_buffer = if planes.len() > 1 {
                1
            } else {
                asbd.mChannelsPerFrame
            };
            let mut storage = AudioBufferListStorage::zeroed();
            storage.list.mNumberBuffers = planes.len() as u32;
            for (buffer, plane) in storage.buffers_mut(planes.len()).iter_mut().zip(planes) {
                *buffer = AudioBuffer {
                    mNumberChannels: channels_per_buffer,
                    mDataByteSize: plane.len() as u32,
                    mData: plane.as_ptr().cast_mut().cast(),
                };
            }
            // SAFETY: every buffer in the list points at a live `planes`
            // slice; the call copies the bytes into its own block buffer.
            let status = unsafe {
                sbuf.set_data_buffer_from_audio_buffer_list(
                    None,
                    None,
                    kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment,
                    storage.list_ptr(),
                )
            };
            assert_eq!(status, 0, "CoreMedia refused to attach the audio data");
        }
        sbuf
    }

    /// Builds a `CMSampleBuffer` whose data buffer exists but was never
    /// marked ready — the shape of a buffer whose producer is still filling
    /// it in. `set_data_buffer_from_audio_buffer_list` can't make one of
    /// these (it marks the data ready itself), so the block buffer is
    /// attached at creation with `data_ready: false`.
    fn not_ready_buffer(
        asbd: AudioStreamBasicDescription,
        frames: usize,
        byte_len: usize,
    ) -> CFRetained<CMSampleBuffer> {
        let desc = asbd_format_description(asbd);
        let mut block_raw: *mut CMBlockBuffer = std::ptr::null_mut();
        // SAFETY: a null memory block asks CoreMedia to allocate `byte_len`
        // bytes itself (assured immediately by the flag); the out-pointer is
        // valid for the write.
        let status = unsafe {
            CMBlockBuffer::create_with_memory_block(
                None,
                std::ptr::null_mut(),
                byte_len,
                None,
                std::ptr::null(),
                0,
                byte_len,
                kCMBlockBufferAssureMemoryNowFlag,
                NonNull::from(&mut block_raw),
            )
        };
        assert_eq!(status, 0, "CoreMedia refused to create the block buffer");
        // SAFETY: on success the call hands back a +1 reference.
        let block = unsafe { CFRetained::from_raw(NonNull::new(block_raw).expect("no block out")) };

        let timing = audio_timing();
        let mut raw: *mut CMSampleBuffer = std::ptr::null_mut();
        // SAFETY: same argument as in `synthetic_buffer`; `data_ready: false`
        // with no make-ready callback is a buffer that never becomes ready.
        let status = unsafe {
            CMSampleBuffer::create(
                None,
                Some(&block),
                false,
                None,
                std::ptr::null_mut(),
                Some(&desc),
                frames as CMItemCount,
                1,
                &timing,
                0,
                std::ptr::null(),
                NonNull::from(&mut raw),
            )
        };
        assert_eq!(status, 0, "CoreMedia refused to create the sample buffer");
        // SAFETY: on success the call hands back a +1 reference.
        unsafe { CFRetained::from_raw(NonNull::new(raw).expect("no buffer out")) }
    }

    /// Whether this process can read PCM out of a fresh, valid sample buffer
    /// at all, via both CoreMedia read calls — with the calls made *inline*,
    /// deliberately not through `read_by_copying`/`read_via_block_buffer`. A
    /// bug in those functions must fail the tests below; it must not be able
    /// to disguise itself as a broken environment and skip them.
    ///
    /// This exists because CoreMedia itself has been observed refusing these
    /// exact calls on freshly built synthetic buffers in bimodal windows: in
    /// one 2026-08-24 review session, 16 consecutive runs of this module
    /// failed — first with RequiredParameterMissing (-12731) from the copy
    /// call, then with refusals of the block-buffer call — and then 47
    /// consecutive runs of the *identical binary* passed, including under
    /// artificial CPU load. Every FFI signature was verified against the
    /// objc2-core-media 0.3.2 bindings; the trigger is environmental
    /// (mediaserverd state, sandboxed execution, or similar), not
    /// deterministic. So when even this canary is refused, the tests that
    /// need CoreMedia to cooperate skip loudly instead of failing a gate on
    /// weather — while any deterministic break in the production read path
    /// still fails them on every run.
    fn coremedia_read_canary() -> Result<(), String> {
        let samples = [0.1f32, 0.2, 0.3, 0.4];
        let bytes = f32_bytes(&samples);
        let sbuf = synthetic_buffer(float_asbd(1, false), samples.len(), &[&bytes]);

        // The copy call, descriptor built inline.
        let mut out = [0.0f32; 4];
        let mut storage = AudioBufferListStorage::zeroed();
        storage.list.mNumberBuffers = 1;
        storage.buffers_mut(1)[0] = AudioBuffer {
            mNumberChannels: 1,
            mDataByteSize: std::mem::size_of_val(&out) as u32,
            mData: out.as_mut_ptr().cast(),
        };
        // SAFETY: the single buffer points at `out`, which lives for the
        // whole call and is exactly as large as the size written above.
        let status = unsafe {
            sbuf.copy_pcm_data_into_audio_buffer_list(0, out.len() as i32, storage.list_ptr())
        };
        if status != 0 {
            return Err(format!(
                "the copy call refused the canary buffer: {status}({})",
                sample_buffer_status_name(status)
            ));
        }
        if out != samples {
            return Err(format!("the copy call wrote back {out:?}"));
        }

        // The block-buffer call: size probe, then the list itself.
        let mut needed = 0usize;
        // SAFETY: a null buffer list with size 0 is the documented way to
        // ask only for the needed size; the out-pointer is valid.
        let probe = unsafe {
            sbuf.audio_buffer_list_with_retained_block_buffer(
                &mut needed,
                std::ptr::null_mut(),
                0,
                None,
                None,
                kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment,
                std::ptr::null_mut(),
            )
        };
        if needed == 0 || needed > std::mem::size_of::<AudioBufferListStorage>() {
            return Err(format!(
                "the size probe refused the canary buffer: {probe}({}), needed={needed}",
                sample_buffer_status_name(probe)
            ));
        }
        let mut storage = AudioBufferListStorage::zeroed();
        let mut block: *mut CMBlockBuffer = std::ptr::null_mut();
        // SAFETY: `needed` was checked against the real size of `storage`;
        // the out-pointers are valid for writes.
        let status = unsafe {
            sbuf.audio_buffer_list_with_retained_block_buffer(
                std::ptr::null_mut(),
                storage.list_ptr().as_ptr(),
                needed,
                None,
                None,
                kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment,
                &mut block,
            )
        };
        // SAFETY: when non-null, this is a +1 reference the call handed over.
        let _block = NonNull::new(block).map(|b| unsafe { CFRetained::from_raw(b) });
        if status != 0 {
            return Err(format!(
                "the block-buffer call refused the canary buffer: {status}({})",
                sample_buffer_status_name(status)
            ));
        }
        Ok(())
    }

    /// True when CoreMedia is cooperating; otherwise says loudly that the
    /// calling test is being skipped, and why, so a quiet green run during a
    /// refusal window is at least visibly quieter in the output.
    fn coremedia_cooperates(test: &str) -> bool {
        match coremedia_read_canary() {
            Ok(()) => true,
            Err(why) => {
                eprintln!(
                    "SKIPPED {test}: CoreMedia is refusing PCM reads in this environment right now — {why}"
                );
                false
            }
        }
    }

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
    fn the_buffer_list_storage_matches_the_c_layout() {
        // The whole fix rests on this layout: the crate's AudioBufferList
        // declares one AudioBuffer and CoreMedia writes up to MAX_BUFFERS,
        // so `_extra` must sit exactly where the C ABI expects buffer #2.
        assert_eq!(std::mem::size_of::<AudioBuffer>(), 16);
        assert_eq!(AudioBufferListStorage::BUFFERS_OFFSET, 8);
        assert_eq!(
            std::mem::size_of::<AudioBufferListStorage>(),
            8 + 16 * MAX_BUFFERS
        );
        assert!(MAX_BUFFERS >= usize::from(REQUESTED_CHANNELS));
    }

    #[test]
    fn a_deinterleaved_stereo_buffer_is_averaged_to_mono() {
        // The exact shape ScreenCaptureKit delivered in the 2026-08-24
        // meeting: 32-bit float, packed, non-interleaved, one buffer per
        // channel. L=0.2 and R=0.6 must come out as 0.4 on the ring.
        if !coremedia_cooperates("a_deinterleaved_stereo_buffer_is_averaged_to_mono") {
            return;
        }
        let (shared, mut consumer) = shared_for_tests();
        let left = f32_bytes(&[0.2; 4]);
        let right = f32_bytes(&[0.6; 4]);
        let sbuf = synthetic_buffer(float_asbd(2, true), 4, &[&left, &right]);
        handle_audio(&shared, &sbuf);
        let mono = drained(&mut consumer);
        assert_eq!(mono.len(), 4, "{}", shared.diagnostics_summary());
        for sample in &mono {
            assert!((sample - 0.4).abs() < 1e-6, "expected 0.4, got {sample}");
        }
        assert_eq!(shared.counts.unreadable.load(Ordering::Relaxed), 0);
        // The samples must have come through the *primary* copy read. A
        // broken copy path whose refusals the fallback quietly rescues
        // produces the right audio and the wrong implementation — mutation
        // testing proved these tests were green with the interleaved copy
        // descriptor completely broken until this assert existed.
        assert_eq!(
            shared.counts.fallback.load(Ordering::Relaxed),
            0,
            "the copy read did not carry this buffer: {}",
            shared.diagnostics_summary()
        );
    }

    #[test]
    fn an_interleaved_stereo_buffer_is_averaged_to_mono() {
        if !coremedia_cooperates("an_interleaved_stereo_buffer_is_averaged_to_mono") {
            return;
        }
        let (shared, mut consumer) = shared_for_tests();
        let frames = f32_bytes(&[0.2, 0.6, 0.2, 0.6, 0.2, 0.6]);
        let sbuf = synthetic_buffer(float_asbd(2, false), 3, &[&frames]);
        handle_audio(&shared, &sbuf);
        let mono = drained(&mut consumer);
        assert_eq!(mono.len(), 3, "{}", shared.diagnostics_summary());
        for sample in &mono {
            assert!((sample - 0.4).abs() < 1e-6, "expected 0.4, got {sample}");
        }
        assert_eq!(shared.counts.unreadable.load(Ordering::Relaxed), 0);
        // Right audio via the wrong path is a fail: the interleaved copy
        // descriptor had zero coverage while the fallback could rescue it.
        assert_eq!(
            shared.counts.fallback.load(Ordering::Relaxed),
            0,
            "the copy read did not carry this buffer: {}",
            shared.diagnostics_summary()
        );
    }

    #[test]
    fn a_mono_buffer_passes_through_unscaled() {
        if !coremedia_cooperates("a_mono_buffer_passes_through_unscaled") {
            return;
        }
        let (shared, mut consumer) = shared_for_tests();
        let samples = [0.25f32, -0.5, 1.0, 0.0];
        let bytes = f32_bytes(&samples);
        let sbuf = synthetic_buffer(float_asbd(1, false), 4, &[&bytes]);
        handle_audio(&shared, &sbuf);
        // Bit-exact: a single channel must not be averaged with anything.
        assert_eq!(drained(&mut consumer), samples.to_vec());
        // And it must have come through the primary copy read, not been
        // rescued by the fallback after a refusal.
        assert_eq!(
            shared.counts.fallback.load(Ordering::Relaxed),
            0,
            "the copy read did not carry this buffer: {}",
            shared.diagnostics_summary()
        );
    }

    #[test]
    fn a_buffer_with_no_samples_counts_as_empty_not_unreadable() {
        let (shared, mut consumer) = shared_for_tests();
        let sbuf = synthetic_buffer(float_asbd(2, true), 0, &[]);
        handle_audio(&shared, &sbuf);
        assert!(drained(&mut consumer).is_empty());
        assert_eq!(shared.counts.empty.load(Ordering::Relaxed), 1);
        assert_eq!(shared.counts.unreadable.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn integer_pcm_is_refused_rather_than_guessed_at() {
        // 16-bit signed integers reinterpreted as f32 would be deafening
        // garbage; the counter has to say the buffer was refused for its
        // format, not thrown away as unreadable.
        let (shared, mut consumer) = shared_for_tests();
        let asbd = AudioStreamBasicDescription {
            mSampleRate: 48_000.0,
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked,
            mBytesPerPacket: 4,
            mFramesPerPacket: 1,
            mBytesPerFrame: 4,
            mChannelsPerFrame: 2,
            mBitsPerChannel: 16,
            mReserved: 0,
        };
        let bytes: Vec<u8> = [100i16, -100, 200, -200, 300, -300, 400, -400]
            .iter()
            .flat_map(|s| s.to_ne_bytes())
            .collect();
        let sbuf = synthetic_buffer(asbd, 4, &[&bytes]);
        handle_audio(&shared, &sbuf);
        assert!(drained(&mut consumer).is_empty());
        assert_eq!(shared.counts.not_float.load(Ordering::Relaxed), 1);
        assert_eq!(shared.counts.unreadable.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn the_fallback_read_hears_the_same_audio_as_the_copy() {
        // The fallback is only ever taken when the copy call is refused,
        // which no test can force CoreMedia to do — so it is called directly
        // here, both to keep it from being dead code and to pin down that
        // switching paths mid-meeting cannot change what is heard.
        if !coremedia_cooperates("the_fallback_read_hears_the_same_audio_as_the_copy") {
            return;
        }
        let left = f32_bytes(&[0.2; 4]);
        let right = f32_bytes(&[0.6; 4]);
        let sbuf = synthetic_buffer(float_asbd(2, true), 4, &[&left, &right]);

        let mut pcm = Vec::new();
        let mut copied = Vec::new();
        let copy_status = read_by_copying(&sbuf, true, 2, 4, &mut pcm, &mut copied);
        assert_eq!(
            copy_status,
            0,
            "the copy read refused a valid buffer the canary could read: {copy_status}({})",
            sample_buffer_status_name(copy_status)
        );
        let mut fallback = Vec::new();
        if let Err((status, needed)) = read_via_block_buffer(&sbuf, 4, &mut fallback) {
            panic!(
                "the fallback read refused a valid buffer the canary could read: \
                 {status}({}), needed={needed}, storage={}",
                sample_buffer_status_name(status),
                std::mem::size_of::<AudioBufferListStorage>()
            );
        }
        assert_eq!(copied, fallback);
        assert!((copied[0] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn a_buffer_whose_data_never_became_ready_is_counted_apart() {
        // "The data never arrived" points at the producer; "we could not
        // read it" points here. The two must never share a counter.
        let (shared, mut consumer) = shared_for_tests();
        // 4 frames of non-interleaved stereo f32: 32 bytes, never made ready.
        let sbuf = not_ready_buffer(float_asbd(2, true), 4, 32);
        handle_audio(&shared, &sbuf);
        assert!(drained(&mut consumer).is_empty());
        assert_eq!(shared.counts.not_ready.load(Ordering::Relaxed), 1);
        assert_eq!(shared.counts.unreadable.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn the_planar_mixdown_averages_channels_and_passes_mono_through() {
        let mut mono = Vec::new();
        mix_planar_into(&mut mono, &[&[0.2, 0.4], &[0.6, 0.0]], 2);
        assert!((mono[0] - 0.4).abs() < 1e-6);
        assert!((mono[1] - 0.2).abs() < 1e-6);

        // One channel: passthrough, bit-exact, no scaling.
        mix_planar_into(&mut mono, &[&[0.5, -0.5]], 2);
        assert_eq!(mono, vec![0.5, -0.5]);

        // A short channel contributes what it has; the frame count holds.
        mix_planar_into(&mut mono, &[&[1.0, 1.0], &[1.0]], 2);
        assert!((mono[0] - 1.0).abs() < 1e-6);
        assert!((mono[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn the_interleaved_mixdown_averages_each_frame() {
        let mut mono = Vec::new();
        mix_interleaved_into(&mut mono, &[0.2, 0.6, -0.2, -0.6], 2);
        assert_eq!(mono.len(), 2);
        assert!((mono[0] - 0.4).abs() < 1e-6);
        assert!((mono[1] + 0.4).abs() < 1e-6);

        // Mono: passthrough, bit-exact.
        mix_interleaved_into(&mut mono, &[0.1, 0.2, 0.3], 1);
        assert_eq!(mono, vec![0.1, 0.2, 0.3]);

        // A trailing partial frame was never a whole frame; it is dropped.
        mix_interleaved_into(&mut mono, &[0.5, 0.5, 0.5], 2);
        assert_eq!(mono.len(), 1);
    }

    #[test]
    fn a_reopened_stream_earns_its_own_log_lines_but_keeps_the_meetings_record() {
        let (shared, _consumer) = shared_for_tests();
        shared.format_logged.store(true, Ordering::Relaxed);
        shared.read_failure_logged.store(true, Ordering::Relaxed);
        shared.fallback_logged.store(true, Ordering::Relaxed);
        shared.counts.fallback.fetch_add(7, Ordering::Relaxed);
        shared
            .last_read_status
            .store(kCMSampleBufferError_ArrayTooSmall, Ordering::Relaxed);
        shared
            .last_list_status
            .store(kCMSampleBufferError_BufferNotReady, Ordering::Relaxed);

        shared.reset_per_stream_logs();

        // The new stream gets its own once-per-stream lines...
        assert!(!shared.format_logged.load(Ordering::Relaxed));
        assert!(!shared.read_failure_logged.load(Ordering::Relaxed));
        assert!(!shared.fallback_logged.load(Ordering::Relaxed));
        // ...but the meeting's record of what went wrong survives.
        assert_eq!(shared.counts.fallback.load(Ordering::Relaxed), 7);
        assert_eq!(
            shared.last_read_status.load(Ordering::Relaxed),
            kCMSampleBufferError_ArrayTooSmall
        );
        assert_eq!(
            shared.last_list_status.load(Ordering::Relaxed),
            kCMSampleBufferError_BufferNotReady
        );
    }

    #[test]
    fn the_configuration_macos_gets_is_the_one_we_asked_for() {
        // A configuration object is plain state: no display, no daemon, no
        // permission. So the one thing that *can* be checked here is that every
        // field really took — a setter that silently did nothing is a stream
        // that starts and delivers nothing.
        let config = audio_configuration();
        unsafe {
            assert!(config.capturesAudio(), "audio was never switched on");
            assert!(
                config.excludesCurrentProcessAudio(),
                "playing a recording back would leak into the next one"
            );
            assert_eq!(config.sampleRate(), REQUESTED_RATE as isize);
            assert_eq!(config.channelCount(), REQUESTED_CHANNELS as isize);
            assert_eq!(config.width(), VIDEO_SIDE);
            assert_eq!(config.height(), VIDEO_SIDE);
            assert!(config.queueDepth() >= 3, "too shallow to absorb a hiccup");
        }
        // And the log line built from it must not panic on any of that.
        log_negotiated_configuration(&config, true);
    }

    #[test]
    fn the_video_side_of_the_stream_is_cheap_but_not_degenerate() {
        // Registered and thrown away. It exists because an SCStream always has a
        // video side; it must never be a size or a rate the framework has to
        // special-case.
        let side = VIDEO_SIDE;
        assert!(side >= 16, "small enough that macOS may refuse it");
        assert_eq!(side % 2, 0, "an odd frame size is asking for trouble");

        let interval = VIDEO_FRAME_INTERVAL;
        assert!(interval.timescale > 0, "a zero timescale is not a duration");
        assert!(
            interval.value > 0,
            "a zero interval means as fast as possible"
        );
        assert!(
            interval.value / i64::from(interval.timescale) <= 60,
            "an interval measured in minutes is not a documented configuration"
        );
    }

    #[test]
    fn audio_and_video_arrive_under_different_labels() {
        // The tap tells them apart by this value alone: if they ever collided,
        // every dropped video frame would be counted as audio and the
        // watchdog would never fire.
        assert_ne!(SCStreamOutputType::Audio, SCStreamOutputType::Screen);
    }

    #[test]
    fn a_serial_queue_for_the_sample_handler_can_be_made() {
        // The bug this file was fixed for: ScreenCaptureKit only calls a stream
        // output back on the queue it was handed, and nil is not a queue.
        let queue = DispatchQueue::new("dev.echo.system-audio.test", DispatchQueueAttr::SERIAL);
        let ran = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&ran);
        queue.exec_sync(move || flag.store(true, Ordering::SeqCst));
        assert!(ran.load(Ordering::SeqCst), "the queue never ran anything");
    }

    #[test]
    fn read_failures_are_named_not_numbered() {
        // The bug this instrumentation exists to fix: a real meeting's log
        // could only say "unreadable=500", never which OSStatus macOS
        // actually returned. Every status the failing branch can plausibly
        // see must come back as a name, and an OSStatus nobody has catalogued
        // yet must say "unknown" rather than panic or lie.
        assert_eq!(
            sample_buffer_status_name(kCMSampleBufferError_ArrayTooSmall),
            "ArrayTooSmall"
        );
        assert_eq!(
            sample_buffer_status_name(kCMSampleBufferError_InvalidMediaTypeForOperation),
            "InvalidMediaTypeForOperation"
        );
        assert_eq!(
            sample_buffer_status_name(kCMSampleBufferError_InvalidMediaFormat),
            "InvalidMediaFormat"
        );
        assert_eq!(sample_buffer_status_name(-1), "unknown");
    }

    #[test]
    fn the_diagnostics_line_carries_the_last_refusal() {
        // `shared_for_tests` is constructed the same way `open_blocking`
        // builds one, so this test breaks the moment a field is added there
        // and forgotten here.
        let (shared, _consumer) = shared_for_tests();

        // Never failed: no status clauses at all, so a healthy meeting's
        // diagnostics line doesn't grow a misleading "read_status=0(unknown)".
        let line = shared.diagnostics_summary();
        assert!(!line.contains("read_status"));
        assert!(!line.contains("list_status"));
        assert!(line.contains("not_ready=0"), "not in the funnel: {line}");
        assert!(line.contains("fallback=0"), "not in the funnel: {line}");

        shared.counts.unreadable.fetch_add(1, Ordering::Relaxed);
        shared
            .last_read_status
            .store(kCMSampleBufferError_ArrayTooSmall, Ordering::Relaxed);
        shared.last_list_status.store(
            kCMSampleBufferError_InvalidMediaTypeForOperation,
            Ordering::Relaxed,
        );
        let line = shared.diagnostics_summary();
        assert!(
            line.contains("read_status=-12737(ArrayTooSmall)"),
            "diagnostics line did not name the last copy refusal: {line}"
        );
        assert!(
            line.contains("list_status=-12741(InvalidMediaTypeForOperation)"),
            "diagnostics line did not name the last fallback refusal: {line}"
        );
    }

    #[test]
    fn a_stream_carried_entirely_by_the_fallback_is_not_mistaken_for_a_healthy_one() {
        // The 2026-08-24 class of failure, one notch less severe: every copy
        // read refused, every buffer rescued by the fallback. No audio was
        // lost — `unreadable` stays 0 — but the line must still say the
        // stream ran on its second-choice read and name the refusal, or a
        // permanently degraded meeting reads exactly like a healthy one.
        let line = diagnostics_line(
            500,
            10,
            0,
            0,
            500,
            0,
            0,
            240_000,
            0,
            48_000,
            kCMSampleBufferError_RequiredParameterMissing,
            0,
        );
        assert!(line.contains("unreadable=0"), "wrong funnel: {line}");
        assert!(
            line.contains("fallback=500"),
            "the rescue is invisible: {line}"
        );
        assert!(
            line.contains("read_status=-12731(RequiredParameterMissing)"),
            "the refusal that forced the fallback is unnamed: {line}"
        );
        // The fallback itself never failed, so no second status clause.
        assert!(
            !line.contains("list_status"),
            "phantom fallback refusal: {line}"
        );
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
