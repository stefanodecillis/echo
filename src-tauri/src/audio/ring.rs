//! The bounded hand-over between an audio callback and Echo's own threads.
//!
//! An audio callback runs on a thread the operating system owns and will not
//! wait for. It may not allocate, lock, log, touch the database or run
//! inference (DESIGN §2 DSP). All it does is copy its samples into a bounded
//! ring and return.
//!
//! When the ring is full the callback drops the block and counts it. That is
//! deliberate: a callback that blocks makes the whole device stutter, and the
//! consumer is the one writing audio to disk, so it is the side that must never
//! fall behind. Drops are diagnostics only — the person is told "the transcript
//! is catching up", never a number (mantra 2).

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use ringbuf::traits::{Consumer, Observer, Producer, Split};
use ringbuf::{HeapCons, HeapProd, HeapRb};

/// How much audio a source's ring holds before it starts dropping. Three
/// seconds is far more than any consumer hiccup and still small in memory
/// (48 kHz stereo ≈ 1.2 MB).
pub const RING_SECONDS: usize = 3;

/// Lock-free state shared between a callback and its consumer.
#[derive(Debug)]
pub struct CaptureCounters {
    /// Samples the callback could not hand over.
    dropped_samples: AtomicU64,
    /// Samples successfully handed over.
    pushed_samples: AtomicU64,
    /// Meeting-clock offset of the most recent hand-over.
    last_clock_ms: AtomicI64,
    /// False once the device goes away.
    running: AtomicBool,
    /// Native rate as discovered at runtime, 0 until known.
    sample_rate: AtomicU32,
    /// Native channel count, 0 until known.
    channels: AtomicU32,
}

impl Default for CaptureCounters {
    fn default() -> Self {
        Self {
            dropped_samples: AtomicU64::new(0),
            pushed_samples: AtomicU64::new(0),
            last_clock_ms: AtomicI64::new(0),
            running: AtomicBool::new(true),
            sample_rate: AtomicU32::new(0),
            channels: AtomicU32::new(0),
        }
    }
}

impl CaptureCounters {
    pub fn dropped_samples(&self) -> u64 {
        self.dropped_samples.load(Ordering::Relaxed)
    }

    pub fn pushed_samples(&self) -> u64 {
        self.pushed_samples.load(Ordering::Relaxed)
    }

    pub fn last_clock_ms(&self) -> i64 {
        self.last_clock_ms.load(Ordering::Relaxed)
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    pub fn mark_stopped(&self) {
        self.running.store(false, Ordering::Relaxed);
    }

    pub fn mark_running(&self) {
        self.running.store(true, Ordering::Relaxed);
    }

    pub fn set_format(&self, sample_rate: u32, channels: u16) {
        self.sample_rate.store(sample_rate, Ordering::Relaxed);
        self.channels.store(u32::from(channels), Ordering::Relaxed);
    }

    /// `None` until the backend has told us what it is actually delivering.
    pub fn format(&self) -> Option<(u32, u16)> {
        let rate = self.sample_rate.load(Ordering::Relaxed);
        let channels = self.channels.load(Ordering::Relaxed);
        if rate == 0 || channels == 0 {
            None
        } else {
            Some((rate, channels as u16))
        }
    }
}

/// The callback half. Everything it does is a memcpy and a few relaxed atomics.
pub struct RingProducer {
    prod: HeapProd<f32>,
    counters: Arc<CaptureCounters>,
    origin: Instant,
}

impl std::fmt::Debug for RingProducer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RingProducer").finish()
    }
}

impl RingProducer {
    /// Hand over a block of native-rate samples. Never blocks, never allocates.
    ///
    /// Returns how many samples made it, which is all of them unless the
    /// consumer fell behind.
    pub fn push(&mut self, samples: &[f32]) -> usize {
        let stamp = self.origin.elapsed().as_millis() as i64;
        self.counters.last_clock_ms.store(stamp, Ordering::Relaxed);
        let written = self.prod.push_slice(samples);
        self.counters
            .pushed_samples
            .fetch_add(written as u64, Ordering::Relaxed);
        if written < samples.len() {
            self.counters
                .dropped_samples
                .fetch_add((samples.len() - written) as u64, Ordering::Relaxed);
        }
        written
    }

    pub fn counters(&self) -> &Arc<CaptureCounters> {
        &self.counters
    }
}

/// Echo's half: pops whatever is there and never waits for more.
pub struct RingConsumer {
    cons: HeapCons<f32>,
    counters: Arc<CaptureCounters>,
}

impl std::fmt::Debug for RingConsumer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RingConsumer")
            .field("available", &self.cons.occupied_len())
            .finish()
    }
}

impl RingConsumer {
    /// Append everything available to `out`. Returns how many samples moved.
    pub fn drain_into(&mut self, out: &mut Vec<f32>) -> usize {
        let available = self.cons.occupied_len();
        if available == 0 {
            return 0;
        }
        let start = out.len();
        out.resize(start + available, 0.0);
        let got = self.cons.pop_slice(&mut out[start..]);
        out.truncate(start + got);
        got
    }

    pub fn counters(&self) -> &Arc<CaptureCounters> {
        &self.counters
    }
}

/// Build a hand-over sized for `sample_rate * channels * RING_SECONDS`.
///
/// `origin` is the meeting clock's start instant, copied into the callback so
/// it can stamp its block without reaching for shared state.
pub fn hand_over(
    sample_rate: u32,
    channels: u16,
    origin: Instant,
) -> (RingProducer, RingConsumer, Arc<CaptureCounters>) {
    let capacity = (sample_rate.max(8_000) as usize) * usize::from(channels.max(1)) * RING_SECONDS;
    let (prod, cons) = HeapRb::<f32>::new(capacity).split();
    let counters = Arc::new(CaptureCounters::default());
    counters.set_format(sample_rate, channels);
    (
        RingProducer {
            prod,
            counters: Arc::clone(&counters),
            origin,
        },
        RingConsumer {
            cons,
            counters: Arc::clone(&counters),
        },
        counters,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_go_through_in_order() {
        let (mut prod, mut cons, counters) = hand_over(16_000, 1, Instant::now());
        assert_eq!(prod.push(&[1.0, 2.0, 3.0]), 3);
        let mut out = Vec::new();
        assert_eq!(cons.drain_into(&mut out), 3);
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
        assert_eq!(counters.dropped_samples(), 0);
        assert_eq!(counters.pushed_samples(), 3);
    }

    #[test]
    fn draining_an_empty_ring_is_free_and_appends_nothing() {
        let (_prod, mut cons, _) = hand_over(16_000, 1, Instant::now());
        let mut out = vec![9.0];
        assert_eq!(cons.drain_into(&mut out), 0);
        assert_eq!(out, vec![9.0]);
    }

    #[test]
    fn overflow_is_counted_and_the_callback_still_returns() {
        // 8 kHz mono is clamped to the 8 kHz floor: 3 s = 24 000 samples.
        let (mut prod, mut cons, counters) = hand_over(8_000, 1, Instant::now());
        let block = vec![0.5f32; 10_000];
        for _ in 0..4 {
            prod.push(&block);
        }
        assert!(
            counters.dropped_samples() > 0,
            "nothing was reported dropped"
        );
        assert_eq!(
            counters.pushed_samples() + counters.dropped_samples(),
            40_000
        );
        // What is in the ring is still readable: a drop loses audio, never the
        // structure.
        let mut out = Vec::new();
        assert_eq!(cons.drain_into(&mut out), 24_000);
        assert!(out.iter().all(|s| *s == 0.5));
    }

    #[test]
    fn the_clock_stamp_advances() {
        let (mut prod, _cons, counters) = hand_over(16_000, 1, Instant::now());
        prod.push(&[0.0; 16]);
        let first = counters.last_clock_ms();
        std::thread::sleep(std::time::Duration::from_millis(5));
        prod.push(&[0.0; 16]);
        assert!(counters.last_clock_ms() >= first);
    }

    #[test]
    fn a_format_is_only_reported_once_it_is_known() {
        let counters = CaptureCounters::default();
        assert_eq!(counters.format(), None);
        counters.set_format(48_000, 2);
        assert_eq!(counters.format(), Some((48_000, 2)));
    }
}
