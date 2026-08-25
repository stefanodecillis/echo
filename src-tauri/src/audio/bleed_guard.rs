//! The stateful half of bleed detection: the computer's own audio kept on hand,
//! and the one object the capture path will ever have to see.
//!
//! [`crate::audio::bleed`] is a pure measurement — envelopes in, evidence out,
//! no clock, no state, no filesystem, testable against a finished meeting on
//! disk. That is the whole reason it can be argued with. Everything that has to
//! *remember* something lives here instead: the last thirty seconds of the
//! system channel ([`SystemEcho`]), this machine's measured delay
//! ([`crate::audio::bleed::LagEstimate`]), what the loudspeaker gate last said
//! ([`crate::audio::route`]), and the counters a field report is read off.
//!
//! Nothing in this file is wired into capture yet. It is built to be
//! constructible and testable standing on its own — no threads, no devices, no
//! database — because the wiring is a separate change that should be readable
//! as wiring.
//!
//! ## What suppression is allowed to mean
//!
//! Deduplication, and nothing else. A suppressed mic line is not deleted text:
//! it is text that is already in the transcript from the cleaner system-channel
//! copy. That is what makes it defensible, and it has a precondition — **the
//! other copy has to actually be written**. So two of the disarms in
//! [`BleedGuard`] have nothing to do with the route: no system channel in this
//! recording, and a system channel that is not live right now. On the
//! 2026-08-25 test recording 136 of 137 system-channel lines had a microphone
//! twin, so there is a great deal to deduplicate — but only while the far side
//! is being written down somewhere else.

use std::collections::VecDeque;
use std::path::Path;
use std::time::Duration;

use crate::audio::bleed::{
    covers_every_lag, envelope, examine_envelopes, is_bleed, BleedEvidence, LagEstimate, LagSearch,
    LAG_MAX_MS, LAG_MIN_MS, MIN_SPAN_COLD_MS, MIN_SPAN_WARM_MS, MIN_SYSTEM_VOICE_MS,
};
use crate::audio::route::{self, Route};
use crate::audio::vad::Utterance;
use crate::audio::TARGET_SAMPLE_RATE;
use crate::logging::Throttle;
use crate::types::Channel;

// ---------------------------------------------------------------------------
// The ring
// ---------------------------------------------------------------------------

/// Exactly sixteen samples per millisecond at 16 kHz, and the reason the index
/// below is counted in samples.
const SAMPLES_PER_MS: i64 = TARGET_SAMPLE_RATE as i64 / 1_000;

const _: () = assert!(
    SAMPLES_PER_MS * 1_000 == TARGET_SAMPLE_RATE as i64,
    "the sample index is only exact because 16 kHz is a whole number of \
     samples per millisecond"
);

/// How much of the computer's audio is kept.
///
/// The widest span anything ever asks for is one utterance —
/// [`crate::audio::vad::MAX_UTTERANCE_MS`], 24 s — plus the lag search either
/// side, a second in total: 25 s. An utterance is handed over 400-700 ms after
/// the speech it contains ended. Thirty seconds is that plus real headroom for
/// a speech thread that fell behind, which is the case this ring exists to
/// survive; further behind than that and it declines to answer rather than
/// answering about the wrong seconds.
pub const RING_SECONDS: usize = 30;

/// 480,000 floats — 1.92 MB, allocated once and never again.
pub const CAPACITY_SAMPLES: usize = TARGET_SAMPLE_RATE as usize * RING_SECONDS;

const _: () = assert!(
    CAPACITY_SAMPLES as i64
        >= (crate::audio::vad::MAX_UTTERANCE_MS + LAG_MAX_MS - LAG_MIN_MS) * SAMPLES_PER_MS,
    "the ring has to hold the longest utterance plus the whole lag search, or \
     the longest utterances could never be judged at all"
);

/// The shortest intersection worth copying out.
///
/// [`MIN_SPAN_WARM_MS`] is the shortest stretch anything in `bleed.rs` will
/// judge at all, so an intersection shorter than that cannot produce an answer
/// however it is measured. Declining it here gives the same verdict one step
/// earlier *and names the reason* — [`NOT_ON_HAND`] is a sentence somebody can
/// act on, where a correlation over one second silently falling under a span
/// floor is not. It is a coarse pre-check and not the real floor: the real one
/// is [`crate::audio::bleed::is_bleed`]'s, measured over the actual overlap at
/// the chosen delay.
///
/// It says nothing about the *utterance's* own length, which is a different
/// question with a different answer ([`TOO_SHORT`]): a short utterance on a
/// healthy ring is an ordinary "yes, exactly", not audio gone missing.
/// [`BleedGuard::judge`] asks that one first, so this never has to answer it.
const MIN_READ_SAMPLES: i64 = MIN_SPAN_WARM_MS * SAMPLES_PER_MS;

/// The last [`RING_SECONDS`] of what the computer played, on the meeting clock.
///
/// ## Why the index is in samples and not milliseconds
///
/// `first_index` is the **absolute 16 kHz sample index**, counted from t=0 on
/// the meeting clock, of the sample at the front of the deque. Not a
/// millisecond offset: at 16 kHz there are exactly sixteen samples in a
/// millisecond, so a sample index converts to a time and back with no remainder
/// and no accumulated error — however many thousands of times the ring is
/// trimmed, and whatever odd number of samples each trim removes. A millisecond
/// index would have to round on every trim, and each rounding would be a tiny
/// permanent lie about where in the meeting this audio came from. The entire
/// value of this ring is that the audio in it is at the time it claims to be
/// at; the correlator searches ±10 ms grids and a drift of even a few
/// milliseconds an hour would eventually put the copy outside the search.
///
/// ## Placement is by timestamp, never by accumulation
///
/// Exactly as [`crate::audio::writer::ChunkWriter::write_frame`] does it, and
/// for the same reason: the frame's own `t_start_ms` is the truth, and what has
/// been appended so far is only bookkeeping. A gap ahead is filled with
/// silence, which is the *correct* answer rather than a convenient one — the
/// correlator then finds no far-side energy in those seconds, the coverage
/// condition is not met, and the utterance is left alone.
pub struct SystemEcho {
    samples: VecDeque<f32>,
    /// Absolute 16 kHz sample index of `samples.front()`. See the type docs.
    first_index: i64,
    /// Frames that arrived out of order and were dropped. Reported, because a
    /// ring quietly dropping audio would look exactly like a quiet meeting.
    dropped_backwards: u64,
    filled_gaps: u64,
    rebased: u64,
    complaints: Throttle,
}

/// Hand-written because [`Throttle`] is a mutex and not `Debug`, and because
/// the 480,000 floats are never what somebody printing this wants to see.
impl std::fmt::Debug for SystemEcho {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SystemEcho")
            .field("held_ms", &self.held_ms())
            .field("first_index", &self.first_index)
            .field("dropped_backwards", &self.dropped_backwards)
            .field("filled_gaps", &self.filled_gaps)
            .field("rebased", &self.rebased)
            .finish()
    }
}

impl Default for SystemEcho {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemEcho {
    /// One allocation, here, for the whole meeting.
    pub fn new() -> Self {
        Self {
            samples: VecDeque::with_capacity(CAPACITY_SAMPLES),
            first_index: 0,
            dropped_backwards: 0,
            filled_gaps: 0,
            rebased: 0,
            complaints: Throttle::new(Duration::from_secs(30)),
        }
    }

    /// Place one frame of system audio at the moment it belongs to.
    ///
    /// Three things can happen besides the ordinary case:
    ///
    /// * **A gap ahead** — the frame starts after the audio already held ends.
    ///   The hole is filled with silence so the index stays true. This is the
    ///   honest fill: those milliseconds contained nothing that reached Echo,
    ///   the correlator finds no far side there, and it declines.
    /// * **A frame from the past** — it starts before the audio already held
    ///   ends. Dropped whole, with a throttled debug. Trimming the overlapping
    ///   part and keeping the rest, the way the writer does, would be defensible
    ///   for a file that is only ever appended to; here it would mean rewriting
    ///   seconds that a judgement may already have been made about, and the one
    ///   thing this ring must never do is be wrong about *when*.
    /// * **A jump bigger than the ring** — everything held is older than the
    ///   ring's own span by the time this frame lands, so it is dropped and the
    ///   index re-bases on the new frame.
    pub fn append(&mut self, t_start_ms: i64, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }
        let at = t_start_ms * SAMPLES_PER_MS;
        if self.samples.is_empty() {
            self.first_index = at;
            self.push(samples);
            return;
        }

        let gap = at - self.end_index();
        if gap < 0 {
            self.dropped_backwards += 1;
            if let Some(suppressed) = self.complaints.admit("system echo went backwards") {
                tracing::debug!(
                    target: "echo::audio",
                    behind_ms = -gap / SAMPLES_PER_MS,
                    dropped = self.dropped_backwards,
                    suppressed,
                    "a frame of the computer's audio arrived after audio that comes later; \
                     leaving what is already held alone"
                );
            }
            return;
        }
        if gap as usize >= CAPACITY_SAMPLES {
            self.rebased += 1;
            self.samples.clear();
            self.first_index = at;
            self.push(samples);
            return;
        }
        if gap > 0 {
            self.filled_gaps += 1;
            self.push_silence(gap as usize);
        }
        self.push(samples);
    }

    /// Copy the audio between two moments on the meeting clock into `out`,
    /// answering with the millisecond it actually starts at.
    ///
    /// `None` when what the ring still holds of that span is too short to judge
    /// — a speech thread that fell far behind, a stretch from before the
    /// recording resumed. The caller's business is then to leave the utterance
    /// alone and say why, never to judge it against whatever happened to be
    /// there.
    ///
    /// `out` is the caller's scratch buffer, cleared and refilled. The copy
    /// walks [`VecDeque::as_slices`] rather than calling `make_contiguous`,
    /// which is an O(n) rotate of 480,000 floats — on a thread whose whole job
    /// is to keep up with a meeting, and for no benefit at all, since two
    /// `extend_from_slice` calls copy exactly the same bytes.
    pub fn read_into(&self, from_ms: i64, to_ms: i64, out: &mut Vec<f32>) -> Option<i64> {
        out.clear();
        // Start on a whole millisecond, so the answer this returns is exact
        // rather than the nearest millisecond to a sample boundary the ring
        // happens to have been trimmed to. At most fifteen samples are given
        // up, which is a sixth of one envelope hop.
        let wanted = (from_ms * SAMPLES_PER_MS).max(self.first_index).max(0);
        let from = wanted.div_euclid(SAMPLES_PER_MS) * SAMPLES_PER_MS;
        let from = if from < wanted {
            from + SAMPLES_PER_MS
        } else {
            from
        };
        let to = (to_ms * SAMPLES_PER_MS).min(self.end_index());
        if to - from < MIN_READ_SAMPLES {
            return None;
        }

        let start = (from - self.first_index) as usize;
        let end = (to - self.first_index) as usize;
        let (front, back) = self.samples.as_slices();
        if start < front.len() {
            out.extend_from_slice(&front[start..end.min(front.len())]);
        }
        if end > front.len() {
            let start = start.saturating_sub(front.len());
            out.extend_from_slice(&back[start..end - front.len()]);
        }
        Some(from / SAMPLES_PER_MS)
    }

    /// How much audio is held, in milliseconds — for a log line, not a decision.
    pub fn held_ms(&self) -> i64 {
        self.samples.len() as i64 / SAMPLES_PER_MS
    }

    pub fn dropped_backwards(&self) -> u64 {
        self.dropped_backwards
    }

    /// One past the last sample held, as an absolute index.
    fn end_index(&self) -> i64 {
        self.first_index + self.samples.len() as i64
    }

    fn push_silence(&mut self, count: usize) {
        self.make_room(count);
        for _ in 0..count.min(CAPACITY_SAMPLES) {
            self.samples.push_back(0.0);
        }
    }

    fn push(&mut self, samples: &[f32]) {
        // A single frame longer than the whole ring: only its tail can be kept,
        // and everything held is older than that tail.
        if samples.len() >= CAPACITY_SAMPLES {
            let skipped = samples.len() - CAPACITY_SAMPLES;
            self.first_index = self.end_index() + skipped as i64;
            self.samples.clear();
            self.samples.extend(samples[skipped..].iter().copied());
            return;
        }
        self.make_room(samples.len());
        self.samples.extend(samples.iter().copied());
    }

    /// Drop the oldest samples so `incoming` more fit **without the deque ever
    /// growing past the capacity it was built with**. Trimming before the push
    /// rather than after is the whole of it: a deque that momentarily exceeds
    /// its capacity reallocates 1.92 MB, and then does it again the next time.
    fn make_room(&mut self, incoming: usize) {
        let room = CAPACITY_SAMPLES.saturating_sub(incoming.min(CAPACITY_SAMPLES));
        if self.samples.len() > room {
            let excess = self.samples.len() - room;
            self.samples.drain(..excess);
            self.first_index += excess as i64;
        }
    }
}

// ---------------------------------------------------------------------------
// The guard
// ---------------------------------------------------------------------------

/// A file with this name in the log folder switches bleed suppression off for
/// the whole app, with no rebuild and no terminal.
///
/// Same lever as `logging.rs`'s `log-filter` file and for the same reason: a
/// Finder-launched .app has no environment to set. If this ever deletes a real
/// sentence in somebody's meeting, "make an empty file called
/// `no-bleed-suppression` next to your logs" is an instruction that can be
/// given over a chat window and is in force the next time Echo starts.
pub const KILL_SWITCH_FILE: &str = "no-bleed-suppression";

/// How often the loudspeaker gate is re-read while recording.
///
/// Headphones go in and out mid-meeting. Two seconds is quick enough that at
/// most a couple of utterances are judged on the old answer, and two CoreAudio
/// property reads every two seconds is nothing next to the audio threads
/// already running.
pub const ROUTE_POLL_MS: i64 = 2_000;

/// Nothing was measured, because the stretch is shorter than anything
/// `bleed.rs` will judge — see [`BleedReport::too_short`].
pub const TOO_SHORT: &str = "this stretch is too short to measure against the computer's audio";

/// Nothing was measured, because the ring no longer holds those seconds at all.
pub const NOT_ON_HAND: &str = "the computer's audio for this stretch is no longer on hand";

/// Nothing was measured, because the ring holds only part of those seconds and
/// a verdict about part of an utterance may not stand for all of it — see
/// [`crate::audio::bleed::covers_every_lag`].
pub const PART_ON_HAND: &str =
    "only part of the computer's audio for this stretch is still on hand";

/// What one mic utterance was judged to be.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Verdict {
    /// Send it on. Either there was nothing to judge (the system channel's own
    /// utterances) or it was judged and is not a copy.
    Pass,
    /// Nothing was judged, and this is why. The utterance is sent on untouched
    /// — an unanswered question is not evidence.
    Undecided(&'static str),
    /// The microphone's copy of what the computer played, with the measurement
    /// that says so.
    Bleed(BleedEvidence),
}

/// What a whole recording's worth of judging looked like.
///
/// The `examined`/`suppressed` pair is what the once-per-recording log line is
/// made of, and `opportunities` is what makes the *silent* failure visible:
/// plenty of stretches where the far side was genuinely audible and not one of
/// them ever matched means this machine's delay is outside the search window,
/// which fails safe but must not fail quietly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BleedReport {
    pub armed: bool,
    pub route: Route,
    /// Mic utterances actually measured against the far side.
    pub examined: u64,
    pub suppressed: u64,
    /// Utterances whose stretch of the far side was no longer on hand, whole or
    /// in part — the number this report exists to make visible, because a
    /// speech thread far enough behind that the ring has already rolled past
    /// its utterances would otherwise look exactly like a meeting with no bleed
    /// in it. A guard that was never armed at all counts nothing here: it did
    /// not examine these utterances and it did not fail to, and `armed` is what
    /// says so.
    pub undecided: u64,
    /// Utterances too short for the correlation to mean anything, kept apart
    /// from `undecided` because nothing was wrong when they happened.
    ///
    /// "Yes, exactly" is over [`crate::audio::vad::MIN_UTTERANCE_MS`] and under
    /// [`crate::audio::bleed::MIN_SPAN_COLD_MS`], so a chatty meeting produces
    /// a great many of them on a ring in perfect health. Counting those as far
    /// side gone missing would bury the signal above under ordinary
    /// conversation.
    pub too_short: u64,
    /// Of those examined, how many had a real chance to match: the far side was
    /// audible for at least [`MIN_SYSTEM_VOICE_MS`] of the stretch.
    pub opportunities: u64,
    /// The delay measured by the last full hit, if there ever was one.
    pub lag_ms: Option<i64>,
    pub dropped_backwards: u64,
    pub held_ms: i64,
}

/// Everything bleed detection needs to remember, behind four methods.
///
/// The capture path will hold one of these per recording: hand it system frames
/// as they arrive, tell it what the audio route and the system channel are
/// doing, ask it about each mic utterance, and read the report at the end.
#[derive(Debug)]
pub struct BleedGuard {
    echo: SystemEcho,
    lag: LagEstimate,
    route: Route,
    /// Set once at construction and never cleared: the person asked for this
    /// off.
    kill_switch: bool,
    has_system_channel: bool,
    /// `None` while armed; otherwise the reason, ready to be handed back as an
    /// [`Verdict::Undecided`].
    disarmed: Option<&'static str>,
    route_read_at_ms: Option<i64>,
    // Scratch, owned so that judging an utterance allocates nothing. A fresh
    // 1.6 MB `Vec` per utterance on the speech thread — every few seconds, for
    // the length of a meeting — would be a real regression on the one thread
    // that must keep up with the room.
    system: Vec<f32>,
    mic_env: Vec<f32>,
    sys_env: Vec<f32>,
    examined: u64,
    suppressed: u64,
    undecided: u64,
    too_short: u64,
    opportunities: u64,
    last_hit_lag_ms: Option<i64>,
}

impl BleedGuard {
    /// Build one for a recording.
    ///
    /// `log_dir` is where [`KILL_SWITCH_FILE`] is looked for — `None` when
    /// there is no log folder to look in, which is every test. Starts disarmed:
    /// nothing is judged until [`BleedGuard::poll_route`] (or
    /// [`BleedGuard::set_route`]) has said the system channel is live.
    pub fn new(log_dir: Option<&Path>, has_system_channel: bool) -> Self {
        let kill_switch = log_dir.is_some_and(|dir| dir.join(KILL_SWITCH_FILE).exists());
        if kill_switch {
            tracing::info!(
                target: "echo::audio",
                "a no-bleed-suppression file is in the log folder, so Echo will write down \
                 everything it hears, including the microphone's copy of what this computer plays"
            );
        }
        Self {
            echo: SystemEcho::new(),
            lag: LagEstimate::default(),
            route: Route::default(),
            kill_switch,
            has_system_channel,
            disarmed: disarmed_because(kill_switch, has_system_channel, false, Route::default()),
            route_read_at_ms: None,
            system: Vec::with_capacity(
                (crate::audio::vad::MAX_UTTERANCE_MS + LAG_MAX_MS - LAG_MIN_MS) as usize
                    * SAMPLES_PER_MS as usize,
            ),
            mic_env: Vec::new(),
            sys_env: Vec::new(),
            examined: 0,
            suppressed: 0,
            undecided: 0,
            too_short: 0,
            opportunities: 0,
            last_hit_lag_ms: None,
        }
    }

    /// One frame of what the computer played, at the moment it played it.
    pub fn append_system(&mut self, t_start_ms: i64, samples: &[f32]) {
        self.echo.append(t_start_ms, samples);
    }

    /// Re-read the loudspeaker gate, at most every [`ROUTE_POLL_MS`].
    ///
    /// `system_live` is whether the system channel is delivering audio right
    /// now — `SessionShared::system_active` at the call site. It is not a route
    /// question and it is not a hint: it is the precondition that makes
    /// suppression deduplication rather than deletion.
    pub fn poll_route(&mut self, now_ms: i64, system_live: bool) {
        let due = self
            .route_read_at_ms
            .is_none_or(|at| now_ms - at >= ROUTE_POLL_MS || now_ms < at);
        if !due {
            // The route is unchanged, but whether the far side is being written
            // down somewhere else can change between polls.
            self.set_route(self.route, system_live);
            return;
        }
        self.route_read_at_ms = Some(now_ms);
        // Reading two properties costs nothing, but it is still not worth doing
        // for a recording that can never suppress anything.
        let route = if self.kill_switch || !self.has_system_channel {
            self.route
        } else {
            route::current()
        };
        self.set_route(route, system_live);
    }

    /// Take a route reading from somewhere else.
    ///
    /// [`BleedGuard::poll_route`] is the live caller; tests hand a route
    /// straight in, so no test depends on what this machine happens to be
    /// plugged into.
    pub fn set_route(&mut self, route: Route, system_live: bool) {
        if route != self.route {
            // The delay is a property of this machine *and this route*.
            // Somebody plugging headphones in changes the path the sound takes,
            // so every measurement of it is now about a machine in a state it
            // is no longer in.
            self.lag.clear();
            tracing::info!(
                target: "echo::audio",
                from = self.route.as_str(),
                to = route.as_str(),
                "what this computer plays through changed"
            );
        }
        self.route = route;
        let was = self.disarmed;
        self.disarmed = disarmed_because(
            self.kill_switch,
            self.has_system_channel,
            system_live,
            route,
        );
        if was.is_some() && self.disarmed.is_none() {
            tracing::info!(
                target: "echo::audio",
                route = route.as_str(),
                "listening for the microphone's copy of what this computer plays"
            );
        }
        if was.is_none() {
            if let Some(reason) = self.disarmed {
                tracing::info!(target: "echo::audio", reason, "no longer looking for the microphone's copy");
                self.lag.clear();
            }
        }
    }

    pub fn armed(&self) -> bool {
        self.disarmed.is_none()
    }

    pub fn route(&self) -> Route {
        self.route
    }

    /// Is this stretch of microphone audio the microphone's copy of what the
    /// computer played?
    ///
    /// The mic side is [`Utterance::samples`] and nothing else: it is already
    /// 16 kHz mono spanning exactly this utterance's own start to end, so
    /// nothing is re-read, re-cut, or re-timed for the channel being judged.
    /// Only the far side comes out of the ring, and it is read wide enough for
    /// every delay in the search to see the whole stretch.
    ///
    /// **The search is warm first and cold once.** A warm search — 150 ms
    /// either side of what this meeting has already measured — is narrower,
    /// which is what buys the shorter span floor. When it misses, the search is
    /// retried once over the whole range before concluding anything, because
    /// the commonest reason a warm search misses is that a drift correction
    /// just moved the true delay by up to 200 ms in one step
    /// (`audio/clock.rs`). Retrying recovers that inside a single utterance
    /// instead of waiting out the estimate's two-minute memory. The retry costs
    /// one more correlation pass over envelopes that are already computed.
    ///
    /// **Only full hits teach the estimate.** A lag read off a stretch that
    /// failed the predicate is a lag read off noise, and feeding it back would
    /// let one chance correlation drag the warm window onto a wrong delay and
    /// lock it there — the warm window would then keep finding that same wrong
    /// delay, which is the failure mode a narrowing search has.
    ///
    /// **A verdict is about a whole utterance or it is not made.** Three
    /// refusals come before any measurement, each naming itself rather than
    /// borrowing another one's reason:
    ///
    /// * the guard is not armed — nothing is judged at all, and none of the
    ///   counters move;
    /// * the stretch is shorter than the span floor ([`TOO_SHORT`]), which no
    ///   amount of far side on hand could fix;
    /// * the ring holds none ([`NOT_ON_HAND`]) or only part ([`PART_ON_HAND`])
    ///   of the seconds every delay in the search needs. A partial answer is the
    ///   dangerous one: [`crate::audio::bleed::examine_envelopes`] measures over
    ///   the intersection, so a person talking in the part that was never read
    ///   is invisible to the own-voice veto and would be deleted along with the
    ///   echo, `unexplained_ms` reading zero. Suppressing on a fraction of a
    ///   sentence is exactly the error the whole design is built to refuse.
    pub fn judge(&mut self, utterance: &Utterance) -> Verdict {
        // The system channel's own utterances are the *original*; there is
        // nothing for them to be a copy of.
        if utterance.channel != Channel::Mic {
            return Verdict::Pass;
        }
        if let Some(reason) = self.disarmed {
            return Verdict::Undecided(reason);
        }

        // The meeting clock, from the utterance itself. Nothing here keeps a
        // clock of its own, for the same reason `LagEstimate` does not.
        let now_ms = utterance.t_end_ms;
        let voiced_ms = utterance.measured_voice_ms();
        let warm = self.lag.is_warm(now_ms);

        // Shorter than the span floor and there is no answer to be had, whatever
        // the ring holds: `is_bleed` measures the span over the overlap and
        // refuses it, so this is the same verdict a step earlier. It is here to
        // be *truthful* about which of the two reasons applies — half of what
        // people say in a meeting is "yes, exactly", and blaming the ring for
        // those would drown the number that says the speech thread fell behind
        // in ordinary conversation.
        let stretch_ms = utterance.samples.len() as i64 / SAMPLES_PER_MS;
        let span_floor = if warm {
            MIN_SPAN_WARM_MS
        } else {
            MIN_SPAN_COLD_MS
        };
        if stretch_ms < span_floor {
            self.too_short += 1;
            return Verdict::Undecided(TOO_SHORT);
        }

        // Every delay in the search has to see the whole stretch, so the far
        // side is read from LAG_MAX_MS before the utterance starts to
        // LAG_MIN_MS after it ends.
        let Some(system_from_ms) = self.echo.read_into(
            utterance.t_start_ms - LAG_MAX_MS,
            utterance.t_end_ms - LAG_MIN_MS,
            &mut self.system,
        ) else {
            self.undecided += 1;
            return Verdict::Undecided(NOT_ON_HAND);
        };
        let sys_lead_ms = utterance.t_start_ms - system_from_ms;

        envelope(&utterance.samples, &mut self.mic_env);
        envelope(&self.system, &mut self.sys_env);

        // …and "wide enough" is checked against the arrays that actually came
        // back, not assumed from the arithmetic above. `read_into` answers about
        // partial intersections — the utterance began before the recording did,
        // the system stream died mid-sentence and the two-second route poll has
        // not noticed yet, the speech thread is running late — and a partial
        // answer is exactly what must not be turned into a suppression: the
        // own-voice veto can only veto seconds it was shown, so a person
        // talking in the part nothing was read for would be deleted along with
        // the echo, with `unexplained_ms` reading zero. Cold, because the cold
        // retry below may search the whole range, and cold coverage implies
        // warm.
        if !covers_every_lag(
            self.mic_env.len(),
            self.sys_env.len(),
            sys_lead_ms,
            LagSearch::cold(),
        ) {
            self.undecided += 1;
            return Verdict::Undecided(PART_ON_HAND);
        }

        self.examined += 1;
        let mut evidence = examine_envelopes(
            &self.mic_env,
            &self.sys_env,
            sys_lead_ms,
            self.lag.search(now_ms),
        );
        let mut hit = is_bleed(&evidence, voiced_ms, warm);
        if !hit && warm {
            let cold =
                examine_envelopes(&self.mic_env, &self.sys_env, sys_lead_ms, LagSearch::cold());
            if is_bleed(&cold, voiced_ms, false) {
                evidence = cold;
                hit = true;
            } else if cold.system_voice_ms > evidence.system_voice_ms {
                // Report the wider search's view of how much the far side was
                // audible: it is the one that had every delay to choose from,
                // and `opportunities` below is what says whether this machine
                // ever had a chance to match at all.
                evidence = cold;
            }
        }
        if evidence.system_voice_ms >= MIN_SYSTEM_VOICE_MS {
            self.opportunities += 1;
        }
        if hit {
            self.suppressed += 1;
            self.last_hit_lag_ms = Some(evidence.lag_ms);
            self.lag.record(now_ms, evidence.lag_ms);
            return Verdict::Bleed(evidence);
        }
        Verdict::Pass
    }

    /// What this recording's judging looked like, for one log line.
    pub fn report(&self) -> BleedReport {
        BleedReport {
            armed: self.armed(),
            route: self.route,
            examined: self.examined,
            suppressed: self.suppressed,
            undecided: self.undecided,
            too_short: self.too_short,
            opportunities: self.opportunities,
            lag_ms: self.last_hit_lag_ms,
            dropped_backwards: self.echo.dropped_backwards(),
            held_ms: self.echo.held_ms(),
        }
    }
}

/// Why this guard is not judging anything, or `None` when it is.
///
/// A pure function over four facts, so the arming rule can be read and tested
/// in one place. Two of the three refusals are not about the route at all:
/// suppression is only defensible as *deduplication*, and deduplication needs
/// the other copy to exist. A recording with no system channel has no other
/// copy ever; a system channel that is not live right now has none for these
/// seconds.
///
/// The route is the weakest of the three and deliberately so — see
/// [`crate::audio::route`] for why [`Route::CannotTell`] arms.
fn disarmed_because(
    kill_switch: bool,
    has_system_channel: bool,
    system_live: bool,
    route: Route,
) -> Option<&'static str> {
    if kill_switch {
        return Some("bleed suppression is switched off by a file in the log folder");
    }
    if !has_system_channel {
        return Some("this recording has no audio from the computer to be a copy of");
    }
    if !system_live {
        return Some("the computer's audio is not being recorded right now");
    }
    if !route.arms() {
        return Some("the computer is playing into headphones, so nothing reaches the microphone");
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::bleed::tests::{delayed, speech};
    use crate::audio::bleed::{BLEED_CORRELATION, ENVELOPE_HOP_MS};

    const RATE: usize = TARGET_SAMPLE_RATE as usize;
    /// One frame as the capture path produces them: 20 ms, 320 samples.
    const FRAME_SAMPLES: usize = RATE * 20 / 1_000;

    fn ramp(ms: usize) -> Vec<f32> {
        // Distinguishable at every sample, so a test can say exactly which
        // millisecond came back.
        (0..RATE * ms / 1_000).map(|i| i as f32).collect()
    }

    // -------------------------------------------------------------------
    // The ring
    // -------------------------------------------------------------------

    #[test]
    fn audio_comes_back_at_the_millisecond_it_went_in_at() {
        let mut echo = SystemEcho::new();
        let audio = ramp(10_000);
        echo.append(0, &audio);
        let mut out = Vec::new();

        let at = echo.read_into(3_000, 9_000, &mut out).expect("held");
        assert_eq!(at, 3_000);
        assert_eq!(out.len(), RATE * 6);
        assert_eq!(out[0], audio[RATE * 3]);
        assert_eq!(*out.last().unwrap(), audio[RATE * 9 - 1]);

        // …and a span that runs off both ends comes back clamped to what is
        // held, saying where it really starts.
        let at = echo.read_into(-500, 20_000, &mut out).expect("held");
        assert_eq!(at, 0);
        assert_eq!(out.len(), RATE * 10);
    }

    #[test]
    fn a_span_older_than_the_ring_is_declined_rather_than_answered_wrongly() {
        let mut echo = SystemEcho::new();
        for second in 0..40 {
            echo.append(second * 1_000, &vec![0.5f32; RATE]);
        }
        let mut out = Vec::new();
        // Forty seconds in, the first ten are gone.
        assert_eq!(echo.read_into(0, 8_000, &mut out), None);
        assert!(out.is_empty(), "nothing may be handed back with a None");
        // The boundary: a span that only *partly* survives is answered about
        // the part that survives, and the answer says where that starts.
        let at = echo
            .read_into(8_000, 20_000, &mut out)
            .expect("partly held");
        assert_eq!(at, 10_000);
        assert_eq!(out.len(), RATE * 10);
        // A span shorter than anything `bleed.rs` will judge is declined too.
        assert_eq!(echo.read_into(30_000, 32_000, &mut out), None);
    }

    #[test]
    fn a_gap_is_filled_with_silence_so_the_index_stays_true() {
        let mut echo = SystemEcho::new();
        echo.append(0, &vec![1.0f32; RATE * 2]);
        // Two seconds of the computer's audio never arrived.
        echo.append(4_000, &vec![1.0f32; RATE * 2]);
        let mut out = Vec::new();
        let at = echo.read_into(0, 6_000, &mut out).expect("held");
        assert_eq!(at, 0);
        assert_eq!(out.len(), RATE * 6);
        assert_eq!(out[RATE * 3], 0.0, "the hole is silence");
        assert_eq!(
            out[RATE * 4],
            1.0,
            "and the audio after it is at the moment it claims to be at"
        );
    }

    #[test]
    fn a_backwards_frame_is_dropped_rather_than_corrupting_the_index() {
        let mut echo = SystemEcho::new();
        echo.append(0, &vec![1.0f32; RATE * 4]);
        // A frame from a second ago: the one thing that must not be allowed to
        // move what is already held.
        echo.append(3_000, &vec![-1.0f32; RATE]);
        assert_eq!(echo.dropped_backwards(), 1);
        let mut out = Vec::new();
        let at = echo.read_into(0, 4_000, &mut out).expect("held");
        assert_eq!(at, 0);
        assert_eq!(out.len(), RATE * 4);
        assert!(
            out.iter().all(|s| *s == 1.0),
            "the late frame overwrote audio that was already placed"
        );
        // …and the next in-order frame still lands where it says it does.
        echo.append(4_000, &vec![0.25f32; RATE]);
        let at = echo.read_into(1_000, 5_000, &mut out).expect("held");
        assert_eq!(at, 1_000);
        assert_eq!(*out.last().unwrap(), 0.25);
    }

    #[test]
    fn a_jump_bigger_than_the_ring_rebases_it() {
        let mut echo = SystemEcho::new();
        echo.append(0, &vec![1.0f32; RATE * 4]);
        // A minute later: everything held is older than the ring's own span.
        echo.append(64_000, &vec![0.5f32; RATE * 4]);
        let mut out = Vec::new();
        assert_eq!(echo.read_into(0, 4_000, &mut out), None);
        let at = echo.read_into(64_000, 68_000, &mut out).expect("held");
        assert_eq!(at, 64_000);
        assert!(out.iter().all(|s| *s == 0.5));
    }

    /// The allocation promise, which is the reason this is a `VecDeque` with a
    /// capacity rather than a `Vec` that is drained.
    #[test]
    fn five_minutes_of_appends_never_grow_the_ring() {
        let mut echo = SystemEcho::new();
        let capacity = echo.samples.capacity();
        assert!(capacity >= CAPACITY_SAMPLES);
        let frame = vec![0.3f32; FRAME_SAMPLES];
        for tick in 0..(5 * 60 * 1_000 / 20) {
            echo.append(tick * 20, &frame);
        }
        assert_eq!(
            echo.samples.capacity(),
            capacity,
            "the ring reallocated 1.92 MB under the speech thread"
        );
        assert_eq!(echo.samples.len(), CAPACITY_SAMPLES);
        assert_eq!(echo.held_ms(), RING_SECONDS as i64 * 1_000);
        // …and a frame longer than the ring itself does not grow it either.
        echo.append(400_000, &vec![0.1f32; CAPACITY_SAMPLES * 2]);
        assert_eq!(echo.samples.capacity(), capacity);
        assert_eq!(echo.samples.len(), CAPACITY_SAMPLES);
    }

    // -------------------------------------------------------------------
    // Arming
    // -------------------------------------------------------------------

    #[test]
    fn no_system_channel_is_never_armed() {
        // Whatever the route says, and whatever anything else says: with no
        // system channel there is no second copy, so suppression could only
        // ever be deletion.
        for route in [Route::Loudspeaker, Route::CannotTell, Route::Headphones] {
            for live in [true, false] {
                assert!(
                    disarmed_because(false, false, live, route).is_some(),
                    "armed with no system channel on {route:?}"
                );
            }
        }
        let mut guard = BleedGuard::new(None, false);
        guard.set_route(Route::Loudspeaker, true);
        assert!(!guard.armed());
        let utterance = Utterance {
            channel: Channel::Mic,
            t_start_ms: 0,
            t_end_ms: 6_000,
            samples: speech(6_000, 3),
            ..Default::default()
        };
        assert!(matches!(guard.judge(&utterance), Verdict::Undecided(_)));
    }

    #[test]
    fn a_system_channel_that_is_not_live_is_never_armed() {
        assert!(disarmed_because(false, true, false, Route::Loudspeaker).is_some());
        assert!(disarmed_because(false, true, true, Route::Loudspeaker).is_none());
    }

    #[test]
    fn headphones_disarm_and_cannot_tell_arms() {
        assert!(disarmed_because(false, true, true, Route::Headphones).is_some());
        assert!(
            disarmed_because(false, true, true, Route::CannotTell).is_none(),
            "disarming on cannot-tell would disarm on most Bluetooth, which is \
             exactly where a speakerphone in a room lives"
        );
    }

    #[test]
    fn the_kill_switch_file_disarms_everything() {
        assert!(disarmed_because(true, true, true, Route::Loudspeaker).is_some());
        let dir = tempfile::tempdir().expect("temp dir");
        let mut guard = BleedGuard::new(Some(dir.path()), true);
        guard.set_route(Route::Loudspeaker, true);
        assert!(guard.armed(), "no file, no kill switch");

        std::fs::write(dir.path().join(KILL_SWITCH_FILE), "").expect("write");
        let mut guard = BleedGuard::new(Some(dir.path()), true);
        guard.set_route(Route::Loudspeaker, true);
        assert!(!guard.armed());
    }

    /// A route change throws away everything measured about the old route,
    /// rather than searching around a delay that belonged to a different path.
    #[test]
    fn changing_the_route_forgets_the_delay() {
        let mut guard = BleedGuard::new(None, true);
        guard.set_route(Route::Loudspeaker, true);
        guard.lag.record(1_000, 180);
        guard.lag.record(2_000, 200);
        assert!(guard.lag.is_warm(2_000));
        guard.set_route(Route::Headphones, true);
        assert!(!guard.lag.is_warm(3_000));
    }

    // -------------------------------------------------------------------
    // Judging, with no threads and no devices
    // -------------------------------------------------------------------

    /// The whole guard, from frames to a verdict: twenty seconds of the far
    /// side appended the way the capture path appends it, and a microphone
    /// utterance that is the delayed, quiet copy of part of it.
    fn armed_guard(system: &[f32]) -> BleedGuard {
        let mut guard = BleedGuard::new(None, true);
        guard.set_route(Route::Loudspeaker, true);
        for (tick, frame) in system.chunks(FRAME_SAMPLES).enumerate() {
            guard.append_system(tick as i64 * 20, frame);
        }
        guard
    }

    /// One stretch of the microphone as the detector would hand it over.
    ///
    /// `voiced_ms` is half the duration, which is what a real utterance looks
    /// like: it is padded at both ends and carries the pause it closed on, so
    /// the detector calls well under all of it speech. `bleed.rs`'s own tests
    /// use the same proportion (3 s of voice inside a 6 s stretch). Claiming
    /// every millisecond was voice would not be conservative — it *raises* the
    /// coverage floor the far side has to clear, and would make this test pass
    /// or fail on how densely the generated far side happens to talk.
    fn mic_utterance(samples: &[f32], from_ms: i64, to_ms: i64) -> Utterance {
        let from = RATE * from_ms as usize / 1_000;
        let to = RATE * to_ms as usize / 1_000;
        Utterance {
            channel: Channel::Mic,
            t_start_ms: from_ms,
            t_end_ms: to_ms,
            samples: samples[from..to].to_vec(),
            voiced_ms: (to_ms - from_ms) / 2,
            ..Default::default()
        }
    }

    #[test]
    fn the_microphones_copy_of_what_the_computer_played_is_found() {
        let system = speech(20_000, 21);
        let mut guard = armed_guard(&system);
        // 180 ms late and 25 dB down: a laptop speaker across a desk.
        let mic = delayed(&system, 180, 0.056);
        let utterance = mic_utterance(&mic, 6_000, 14_000);

        match guard.judge(&utterance) {
            Verdict::Bleed(ev) => {
                assert!(
                    (ev.lag_ms - 180).abs() <= ENVELOPE_HOP_MS,
                    "the delay was read as {} ms",
                    ev.lag_ms
                );
                assert!(ev.correlation >= BLEED_CORRELATION);
            }
            other => panic!("a copy of the far side was not recognised: {other:?}"),
        }
        let report = guard.report();
        assert_eq!((report.examined, report.suppressed), (1, 1));
        assert_eq!(
            report.lag_ms.map(|l| (l - 180).abs() <= ENVELOPE_HOP_MS),
            Some(true)
        );

        // The same guard, the same far side, the same seconds — but somebody in
        // the room saying something of their own. One hit does not put the
        // guard in a mood to suppress the next thing it sees.
        let mine = speech(20_000, 22);
        assert_eq!(
            guard.judge(&mic_utterance(&mine, 15_000, 19_500)),
            Verdict::Pass
        );
        assert_eq!(guard.report().suppressed, 1);
    }

    #[test]
    fn somebody_else_talking_over_the_same_seconds_is_not_a_copy() {
        let system = speech(20_000, 21);
        let mut guard = armed_guard(&system);
        // The same seconds of the meeting, but a person in the room saying
        // something of their own.
        let mic = speech(20_000, 22);
        let utterance = mic_utterance(&mic, 6_000, 14_000);
        assert_eq!(guard.judge(&utterance), Verdict::Pass);
        let report = guard.report();
        assert_eq!((report.examined, report.suppressed), (1, 0));
        assert!(
            report.opportunities > 0,
            "the far side was audible throughout, so this utterance was a real \
             chance to match and simply did not"
        );
    }

    /// The system channel's own utterances are the original, and the original
    /// is never judged.
    #[test]
    fn the_computers_own_channel_passes_without_being_measured() {
        let system = speech(20_000, 21);
        let mut guard = armed_guard(&system);
        let mut utterance = mic_utterance(&system, 6_000, 14_000);
        utterance.channel = Channel::System;
        assert_eq!(guard.judge(&utterance), Verdict::Pass);
        assert_eq!(guard.report().examined, 0);
    }

    #[test]
    fn an_utterance_from_before_the_ring_is_undecided_not_judged() {
        let system = speech(20_000, 21);
        let mut guard = armed_guard(&system);
        // Forty seconds into a meeting whose ring only reaches back thirty.
        for tick in 1_000..2_000i64 {
            guard.append_system(tick * 20, &vec![0.0f32; FRAME_SAMPLES]);
        }
        let mic = delayed(&system, 180, 0.056);
        let utterance = mic_utterance(&mic, 2_000, 8_000);
        assert!(matches!(guard.judge(&utterance), Verdict::Undecided(_)));
        let report = guard.report();
        assert_eq!(
            (report.examined, report.suppressed, report.undecided),
            (0, 0, 1)
        );
    }

    /// **Only full hits teach the estimate.** A stretch that failed the
    /// predicate leaves the search cold, however well its envelopes happened to
    /// line up somewhere; a stretch that passed it makes the next search warm.
    #[test]
    fn only_full_hits_teach_the_delay() {
        let system = speech(20_000, 21);

        let mut guard = armed_guard(&system);
        let unrelated = speech(20_000, 22);
        for (from, to) in [(3_000, 11_000), (11_500, 19_500)] {
            assert_eq!(
                guard.judge(&mic_utterance(&unrelated, from, to)),
                Verdict::Pass
            );
        }
        assert!(
            !guard.lag.is_warm(19_500),
            "two refusals taught the estimate a delay it never measured"
        );
        assert_eq!(guard.report().lag_ms, None);

        let mut guard = armed_guard(&system);
        let mic = delayed(&system, 180, 0.056);
        for (from, to) in [(3_000, 11_000), (11_500, 19_500)] {
            assert!(matches!(
                guard.judge(&mic_utterance(&mic, from, to)),
                Verdict::Bleed(_)
            ));
        }
        assert!(guard.lag.is_warm(19_500), "two hits are an estimate");
    }

    /// A warm search that misses is retried cold once, so a drift correction —
    /// which moves the true delay by up to 200 ms in a single step — is
    /// recovered inside one utterance instead of waiting out the estimate's
    /// two-minute memory.
    #[test]
    fn a_warm_search_that_misses_is_retried_cold() {
        let system = speech(24_000, 31);
        let mut guard = armed_guard(&system);
        // The meeting has measured a delay of about 120 ms.
        guard.lag.record(1_000, 120);
        guard.lag.record(2_000, 120);
        assert!(guard.lag.is_warm(2_000));

        // …and then the clock corrected, and the true delay is 600 ms — far
        // outside the ±150 ms window the warm search would look in.
        let mic = delayed(&system, 600, 0.06);
        let utterance = mic_utterance(&mic, 6_000, 14_000);
        match guard.judge(&utterance) {
            Verdict::Bleed(ev) => assert!(
                (ev.lag_ms - 600).abs() <= ENVELOPE_HOP_MS,
                "the retry read the delay as {} ms",
                ev.lag_ms
            ),
            other => panic!("the cold retry never happened: {other:?}"),
        }
    }

    /// **A verdict is only ever made about a whole utterance.**
    ///
    /// The computer's audio stops halfway through a stretch the microphone is
    /// still recording — the stream died, or the speech thread got the frames
    /// late, and the two-second route poll has not yet noticed. What the ring
    /// holds is a genuine copy of the first five seconds; the last three are
    /// the person in the room saying something of their own, and nothing on the
    /// far side can be compared with them at all.
    ///
    /// Suppressing on the strength of the part that *was* compared would delete
    /// a sentence that exists nowhere else, with `unexplained_ms` reading 0
    /// because the veto was never shown the seconds it was needed for.
    #[test]
    fn a_stretch_the_far_side_only_half_covers_is_not_suppressed() {
        let system = speech(20_000, 21);
        let mut guard = BleedGuard::new(None, true);
        guard.set_route(Route::Loudspeaker, true);
        // The computer's audio reaches the ring for six seconds and then stops.
        for (tick, frame) in system[..RATE * 6].chunks(FRAME_SAMPLES).enumerate() {
            guard.append_system(tick as i64 * 20, frame);
        }

        // The microphone hears the far side, 180 ms late and 25 dB down, until
        // 6.2 s — and then somebody in the room talks for 2.8 s.
        let bleed = delayed(&system, 180, 0.056);
        let mine = speech(20_000, 22);
        let mut mic = bleed[RATE..RATE * 9].to_vec();
        let own_from = RATE * 52 / 10;
        mic[own_from..].copy_from_slice(&mine[RATE * 6..RATE * 6 + (RATE * 8 - own_from)]);
        let utterance = Utterance {
            channel: Channel::Mic,
            t_start_ms: 1_000,
            t_end_ms: 9_000,
            samples: mic,
            voiced_ms: 3_000,
            ..Default::default()
        };

        match guard.judge(&utterance) {
            Verdict::Undecided(_) => {}
            other => {
                panic!("2.8 seconds nothing was ever compared against were deleted: {other:?}")
            }
        }
        let report = guard.report();
        assert_eq!((report.examined, report.suppressed), (0, 0));
        assert_eq!(
            report.undecided, 1,
            "a stretch the far side only half covers is a stretch the ring \
             could not answer about"
        );
    }

    /// A stretch too short for anything in `bleed.rs` to judge says *that*, and
    /// does not blame a ring that is holding every second of it.
    ///
    /// Half of what people say in a meeting is "yes, exactly" — over
    /// `vad::MIN_UTTERANCE_MS` and under any span the correlator can measure.
    /// Counting those as far-side audio gone missing would bury the one number
    /// that says the speech thread fell behind.
    #[test]
    fn a_short_answer_is_too_short_to_judge_not_missing_audio() {
        let system = speech(20_000, 21);
        let mut guard = armed_guard(&system);
        let mic = delayed(&system, 180, 0.056);

        assert_eq!(
            guard.judge(&mic_utterance(&mic, 4_000, 5_200)),
            Verdict::Undecided(TOO_SHORT)
        );
        let report = guard.report();
        assert_eq!((report.examined, report.suppressed), (0, 0));
        assert_eq!(
            (report.too_short, report.undecided),
            (1, 0),
            "the ring held all twenty seconds; nothing about the far side was \
             missing"
        );

        // The same guard, the same ring, a stretch long enough to measure.
        assert!(matches!(
            guard.judge(&mic_utterance(&mic, 6_000, 14_000)),
            Verdict::Bleed(_)
        ));
        assert_eq!(guard.report().too_short, 1);
    }
}
