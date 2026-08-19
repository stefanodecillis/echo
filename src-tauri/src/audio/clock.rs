//! The monotonic meeting clock.
//!
//! Every frame from every source is stamped against this one clock, so mic and
//! system audio stay aligned even when their devices run at slightly different
//! rates (DESIGN §3, review finding 16). Wall-clock time is never used for
//! offsets, a daylight-saving jump or an NTP step must not move a transcript.
//!
//! The clock keeps running while capture is paused so offsets stay comparable;
//! [`MeetingClock::paused_ms`] records how much of that was silence.

use std::time::Instant;

/// Offsets in milliseconds from the moment capture started.
#[derive(Debug)]
pub struct MeetingClock {
    started: Instant,
    paused_at: Option<Instant>,
    paused_total_ms: i64,
}

impl MeetingClock {
    /// Start at zero. Called immediately before the first stream opens.
    pub fn start() -> Self {
        Self {
            started: Instant::now(),
            paused_at: None,
            paused_total_ms: 0,
        }
    }

    /// Current offset. Monotonic, never decreases.
    pub fn now_ms(&self) -> i64 {
        self.started.elapsed().as_millis() as i64
    }

    /// The instant capture started.
    ///
    /// [`Instant`] is `Copy + Send`, so an audio callback can hold a copy and
    /// stamp its buffer with `origin.elapsed()` without touching a lock or the
    /// heap — which is the only reason this is exposed.
    pub fn origin(&self) -> Instant {
        self.started
    }

    /// Offset with paused stretches removed, the amount of real audio.
    pub fn audio_ms(&self) -> i64 {
        (self.now_ms() - self.paused_ms()).max(0)
    }

    /// Total time spent paused so far.
    pub fn paused_ms(&self) -> i64 {
        let extra = self
            .paused_at
            .map(|at| at.elapsed().as_millis() as i64)
            .unwrap_or(0);
        self.paused_total_ms + extra
    }

    /// Idempotent.
    pub fn pause(&mut self) {
        if self.paused_at.is_none() {
            self.paused_at = Some(Instant::now());
        }
    }

    /// Idempotent.
    pub fn resume(&mut self) {
        if let Some(at) = self.paused_at.take() {
            self.paused_total_ms += at.elapsed().as_millis() as i64;
        }
    }

    pub fn is_paused(&self) -> bool {
        self.paused_at.is_some()
    }
}

impl Default for MeetingClock {
    fn default() -> Self {
        Self::start()
    }
}

/// A source stopped delivering audio for a while: device glitch, sleep, or a
/// backend reconnect. Logged (never shown as jargon) and compensated with
/// silence so later audio keeps its place on the meeting clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Discontinuity {
    /// Offset on the meeting clock where the gap started.
    pub at_ms: i64,
    /// How much audio is missing.
    pub gap_ms: i64,
}

/// A source is treated as having skipped when a hand-over arrives this much
/// later than the audio it carries can explain.
pub const DISCONTINUITY_MS: i64 = 200;

/// Below this the drift is not worth touching; correcting jitter would be
/// worse than leaving it.
pub const DRIFT_CORRECTION_THRESHOLD_MS: i64 = 40;

/// Never move more than this much audio in one correction, so a bad estimate
/// cannot mangle a chunk.
pub const MAX_CORRECTION_MS: i64 = 200;

/// Tracks how far one source has drifted from the meeting clock.
///
/// A device that claims 48 kHz but delivers 48 000.3 samples per second gains
/// about a second an hour. The consumer uses the estimate to insert or drop
/// samples at a frame boundary rather than letting the transcript slide.
///
/// The tracker is fed *after* resampling, in [`crate::audio::TARGET_SAMPLE_RATE`]
/// samples, because resampling is deterministic in time: one input second is
/// always one output second, so measuring the cheap side loses nothing.
#[derive(Debug)]
pub struct DriftTracker {
    rate: u32,
    /// Samples delivered since the current baseline.
    audio_samples: i64,
    /// Net correction applied since the current baseline (+inserted, -dropped).
    applied_samples: i64,
    /// Clock offset of the baseline, `None` until the first hand-over.
    origin_ms: Option<i64>,
    /// Clock offset of the most recent hand-over.
    last_clock_ms: i64,
    /// Silence still owed because the source skipped.
    pending_silence_samples: i64,
    discontinuities: Vec<Discontinuity>,
}

impl DriftTracker {
    pub fn new(nominal_sample_rate: u32) -> Self {
        Self {
            rate: nominal_sample_rate.max(1),
            audio_samples: 0,
            applied_samples: 0,
            origin_ms: None,
            last_clock_ms: 0,
            pending_silence_samples: 0,
            discontinuities: Vec::new(),
        }
    }

    fn samples_to_ms(&self, samples: i64) -> i64 {
        samples * 1_000 / i64::from(self.rate)
    }

    fn ms_to_samples(&self, ms: i64) -> i64 {
        ms * i64::from(self.rate) / 1_000
    }

    /// Feed one hand-over: how many samples arrived, and the clock offset at
    /// which they were handed over.
    pub fn observe(&mut self, samples: usize, clock_ms: i64) {
        let samples = samples as i64;
        match self.origin_ms {
            None => {
                // First hand-over defines the baseline; its own samples are not
                // yet evidence of anything.
                self.origin_ms = Some(clock_ms);
                self.last_clock_ms = clock_ms;
                self.audio_samples = 0;
            }
            Some(_) => {
                let carried_ms = self.samples_to_ms(samples);
                let observed_ms = clock_ms - self.last_clock_ms;
                if observed_ms - carried_ms > DISCONTINUITY_MS {
                    let gap_ms = observed_ms - carried_ms;
                    self.discontinuities.push(Discontinuity {
                        at_ms: self.last_clock_ms,
                        gap_ms,
                    });
                    self.pending_silence_samples += self.ms_to_samples(gap_ms);
                    // The old baseline says nothing about a stream that stopped;
                    // start measuring again from here.
                    self.origin_ms = Some(clock_ms);
                    self.last_clock_ms = clock_ms;
                    self.audio_samples = 0;
                    self.applied_samples = 0;
                    return;
                }
                self.audio_samples += samples;
                self.last_clock_ms = clock_ms;
            }
        }
    }

    /// Positive = this source is ahead of the meeting clock.
    pub fn drift_ms(&self) -> i64 {
        let Some(origin) = self.origin_ms else {
            return 0;
        };
        let delivered_ms = self.samples_to_ms(self.audio_samples + self.applied_samples);
        delivered_ms - (self.last_clock_ms - origin)
    }

    /// How many samples to insert (positive) or drop (negative) at the next
    /// frame boundary to bring the source back in line.
    pub fn correction_samples(&self) -> i64 {
        let drift = self.drift_ms();
        let rate_correction = if drift.abs() >= DRIFT_CORRECTION_THRESHOLD_MS {
            let clamped = drift.clamp(-MAX_CORRECTION_MS, MAX_CORRECTION_MS);
            -self.ms_to_samples(clamped)
        } else {
            0
        };
        let silence = self
            .pending_silence_samples
            .min(self.ms_to_samples(MAX_CORRECTION_MS * 25));
        silence + rate_correction
    }

    /// Record that `samples` of correction were actually applied, so the next
    /// estimate accounts for it. Positive = silence inserted.
    pub fn apply_correction(&mut self, samples: i64) {
        if samples > 0 {
            let from_silence = samples.min(self.pending_silence_samples);
            self.pending_silence_samples -= from_silence;
            self.applied_samples += samples - from_silence;
        } else {
            self.applied_samples += samples;
        }
    }

    /// Gaps seen so far, for the redacted diagnostics log.
    pub fn discontinuities(&self) -> &[Discontinuity] {
        &self.discontinuities
    }

    /// Silence still owed because the source stopped delivering for a while.
    pub fn pending_silence_samples(&self) -> i64 {
        self.pending_silence_samples
    }
}

impl Default for DriftTracker {
    fn default() -> Self {
        Self::new(crate::audio::TARGET_SAMPLE_RATE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::TARGET_SAMPLE_RATE;

    const RATE: u32 = TARGET_SAMPLE_RATE;
    /// 20 ms at 16 kHz.
    const BLOCK: usize = 320;

    #[test]
    fn the_clock_starts_at_zero_and_only_moves_forward() {
        let c = MeetingClock::start();
        let a = c.now_ms();
        let b = c.now_ms();
        assert!(a >= 0);
        assert!(b >= a);
    }

    #[test]
    fn pause_and_resume_are_idempotent() {
        let mut c = MeetingClock::start();
        assert!(!c.is_paused());
        c.pause();
        c.pause();
        assert!(c.is_paused());
        c.resume();
        c.resume();
        assert!(!c.is_paused());
        assert!(c.paused_ms() >= 0);
        assert!(c.audio_ms() <= c.now_ms());
    }

    #[test]
    fn a_source_running_exactly_on_time_needs_no_correction() {
        let mut d = DriftTracker::new(RATE);
        for i in 0..500 {
            d.observe(BLOCK, i * 20);
        }
        assert_eq!(d.drift_ms(), 0, "drift {}", d.drift_ms());
        assert_eq!(d.correction_samples(), 0);
        assert!(d.discontinuities().is_empty());
    }

    #[test]
    fn a_fast_source_is_told_to_drop_samples() {
        let mut d = DriftTracker::new(RATE);
        // Delivers 20 ms of audio every 19 ms of real time: 5 % fast.
        for i in 0..101 {
            d.observe(BLOCK, i * 19);
        }
        // 100 blocks of audio = 2000 ms against 1900 ms of clock.
        assert_eq!(d.drift_ms(), 100);
        let correction = d.correction_samples();
        assert!(correction < 0, "expected a drop, got {correction}");
        assert_eq!(correction, -(100 * i64::from(RATE) / 1_000));

        // Applying it brings the source back in line.
        d.apply_correction(correction);
        assert_eq!(d.drift_ms(), 0);
        assert_eq!(d.correction_samples(), 0);
    }

    #[test]
    fn a_slow_source_is_told_to_insert_silence() {
        let mut d = DriftTracker::new(RATE);
        // Delivers 20 ms of audio every 21 ms of real time.
        for i in 0..101 {
            d.observe(BLOCK, i * 21);
        }
        assert_eq!(d.drift_ms(), -100);
        let correction = d.correction_samples();
        assert!(correction > 0, "expected an insert, got {correction}");
        d.apply_correction(correction);
        assert_eq!(d.drift_ms(), 0);
    }

    #[test]
    fn jitter_below_the_threshold_is_left_alone() {
        let mut d = DriftTracker::new(RATE);
        d.observe(BLOCK, 0);
        d.observe(BLOCK, 20);
        // 30 ms late, still under DISCONTINUITY_MS and under the correction
        // threshold once averaged in.
        d.observe(BLOCK, 70);
        assert!(d.drift_ms().abs() < DRIFT_CORRECTION_THRESHOLD_MS);
        assert_eq!(d.correction_samples(), 0);
        assert!(d.discontinuities().is_empty());
    }

    #[test]
    fn a_stalled_source_is_logged_and_padded_with_exactly_the_missing_audio() {
        let mut d = DriftTracker::new(RATE);
        d.observe(BLOCK, 0);
        d.observe(BLOCK, 20);
        // The device went away for five seconds and came back.
        d.observe(BLOCK, 5_020);

        assert_eq!(d.discontinuities().len(), 1);
        let gap = d.discontinuities()[0];
        assert_eq!(gap.at_ms, 20);
        assert_eq!(gap.gap_ms, 4_980);

        let correction = d.correction_samples();
        assert_eq!(correction, 4_980 * i64::from(RATE) / 1_000);
        d.apply_correction(correction);
        assert_eq!(d.pending_silence_samples(), 0);
        assert_eq!(d.correction_samples(), 0);
        // Measurement restarted, so the gap does not keep asking for silence.
        assert_eq!(d.drift_ms(), 0);
    }

    #[test]
    fn a_single_correction_is_capped_so_one_bad_estimate_cannot_mangle_a_chunk() {
        let mut d = DriftTracker::new(RATE);
        d.observe(BLOCK, 0);
        // A wildly fast source: one second of audio in one block.
        d.observe(RATE as usize, 20);
        let correction = d.correction_samples();
        assert_eq!(correction, -(MAX_CORRECTION_MS * i64::from(RATE) / 1_000));
    }
}
