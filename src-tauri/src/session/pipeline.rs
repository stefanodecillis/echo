//! The live pipeline: what happens to audio while a meeting is being recorded.
//!
//! Three tasks, all fed by the bounded capture feed:
//!
//! ```text
//! feed task:  chunk committed ─► journal row (audio_chunks, committed = 1)
//!             utterance       ─► bounded queue ─┐   (full ⇒ drop, never block)
//!             levels          ─► rate-capped event
//!             source lost     ─► degraded state + banner
//!                                                │
//! speech task: ◄─────────────────────────────────┘
//!             utterance ─► speech engine ─► segment draft
//!                       ─► batched insert (every ~2 s or 8 segments)
//!                       ─► one final event per segment
//!
//! caption task: ◄── snapshots of utterances that are still open
//!             last ≤10 s, every ≥3 s of new speech
//!                       ─► cheapest possible decode ─► one partial event,
//!                          replacing the whole line
//! ```
//!
//! Nothing here is allowed to slow capture down (mantra 3). If the speech engine
//! cannot keep up, utterances are dropped on the floor: the audio is already
//! committed to disk and the catch-up job reads it back afterwards.
//!
//! There is a fourth thing, which runs once per meeting rather than per
//! utterance: [`catch_up_backlog`]. Capture starts at t=0 whether or not the
//! engine is ready, so when the engine *does* come up — seconds later on a warm
//! machine, minutes on a first-ever launch — the start of the meeting is on disk
//! with nothing written down against it. That pass reads it back into the live
//! transcript, at lower priority than new speech, and then gets out of the way.
//!
//! The two passes do not overlap, and [`Backlog`] is how: the meeting clock is
//! split once, at the moment the engine came up. Everything before the split is
//! read back off the recording in order and declined by the live pass;
//! everything after it is the live pass's, and the disk pass never looks there.
//! Read that type before changing either pass — a stretch written by both is a
//! paragraph the person reads twice.
//!
//! ## Why a caption task at all
//!
//! Whisper cannot transcribe speech that has not finished, so text used to
//! appear only when somebody stopped talking — a whole sentence late, and much
//! later than that during a monologue. The caption task decodes what has been
//! said *so far* every few seconds and replaces the line wholesale, the way
//! whisper.cpp's own streaming example does (3 s step, 10 s window). Those
//! captions are guesses: they are never written to the database, never counted
//! as coverage, and always replaced by the final utterance's text.
//!
//! ## Where the snapshots come from
//!
//! The capture layer produces them — it is the only thing holding the open
//! utterance's audio. Its speech-detection thread offers one per channel every
//! second or so of new speech as [`CaptureSignal::SpeechSoFar`]; see
//! [`crate::audio::vad::OpenSpeech`] for the contract. They arrive interleaved
//! with everything else on the capture feed, and [`feed_loop`] splits them onto
//! a channel of their own so a caption never queues behind a chunk being
//! journalled. [`spawn`] wires that up; [`spawn_with_captions`] is the seam the
//! tests use to drive one task at a time.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::asr::engine::{DecodePlan, PartialFn};
use crate::asr::{AsrError, TranscribeJob, Transcription};
use crate::audio::vad::Utterance;
use crate::audio::writer::CommittedChunk;
use crate::db::repo;
use crate::events::{NoticeLevel, NoticePayload, TranscriptFinalPayload, TranscriptPartialPayload};
use crate::session::ports::{CaptureFeed, CaptureSignal, CatchUpControl, EventSink, UiEvent};
use crate::session::Inner;
use crate::types::{Channel, Id, Segment, SegmentDraft};

/// Utterances waiting for text. Small on purpose: falling behind should show up
/// as a catch-up job, not as gigabytes of queued audio.
pub const UTTERANCE_QUEUE: usize = 32;
/// Snapshots waiting to be decoded. Tiny: only the newest one per channel is
/// ever worth anything, so a deep queue would only hold stale guesses.
pub const SNAPSHOT_QUEUE: usize = 4;
/// Longest a finished segment waits before it is written.
pub const BATCH_INTERVAL: Duration = Duration::from_secs(2);
/// Or this many segments, whichever comes first.
pub const BATCH_SEGMENTS: usize = 8;
/// Live partial text: at most five a second.
const PARTIAL_INTERVAL: Duration = Duration::from_millis(200);
/// Loudness for the recording indicator: at most ten a second.
const LEVELS_INTERVAL: Duration = Duration::from_millis(100);

/// Most audio a caption is decoded from: the tail of what is open, never the
/// whole thing. whisper.cpp's streaming example uses the same 10 s.
pub const CAPTION_WINDOW_MS: i64 = 10_000;

/// New speech needed before the caption is decoded again on that channel.
///
/// Measured in audio, not wall clock: if a decode takes longer than the step,
/// the next caption simply covers more speech instead of a queue of
/// near-identical windows building up. Same 3 s as whisper.cpp's example.
pub const CAPTION_STEP_MS: i64 = 3_000;

/// Below this there is not enough speech for a caption to be worth the encode.
const CAPTION_MIN_MS: i64 = 1_100;

/// How far speech has to have moved past the caption being decoded before that
/// caption is cancelled outright rather than finished.
///
/// One step plus a margin, the same rule the engine's queue applies to a queued
/// snapshot: three seconds stale is nearly done and worth finishing, five
/// seconds stale is being replaced the moment it lands.
const STALE_CAPTION_MS: i64 = CAPTION_STEP_MS + 1_000;

/// How long the live pipeline may keep decoding after capture has stopped.
///
/// The stop handoff is hard (review of 2026-08-20, finding 7): when this runs
/// out, every live job — queued or in flight — is abandoned and the disk pass
/// becomes the only thing touching this meeting. A detached drain still decoding
/// while catch-up works out where the holes are is how one stretch of a meeting
/// ends up transcribed twice.
///
/// Comfortably inside the session layer's own five-second drain wait, so the
/// pipeline finishes on its own terms rather than being left running.
const LIVE_HANDOFF: Duration = Duration::from_millis(3_500);

/// Words carried across a forced cut, at most. Codex: "at most the last 20 to 40
/// accepted tokens from the preceding final segment".
const CARRY_WORDS: usize = 30;

/// Silence that ends the carry. A forced cut is contiguous by construction, so
/// anything longer than a breath means this is not the continuation of that
/// sentence any more.
const CARRY_MAX_GAP_MS: i64 = 1_500;

/// Text this shaky is not context, it is a guess about a guess.
const CARRY_MIN_CONFIDENCE: f32 = 0.5;

/// After the first engine failure, log only every Nth.
///
/// A backed-up queue draining against a broken engine is dozens of identical
/// lines a second. The first one is the diagnosis; the rest are noise that
/// buries whatever else the log had to say.
const FAILURE_LOG_EVERY: u32 = 25;

/// Consecutive failures before the person is told, once.
///
/// One utterance the engine could not read is not worth interrupting anybody
/// over — it is on disk and catch-up will get it. Ten in a row means live text
/// is not working, and *that* is worth one sentence.
const FAILURES_BEFORE_TELLING: u32 = 10;

// ---------------------------------------------------------------------------
// Snapshots of speech that is still going
// ---------------------------------------------------------------------------

/// A look at an utterance that has **not finished yet**, for a caption.
///
/// Produced by the capture layer, which is the only place the open audio lives —
/// see [`crate::audio::vad::OpenSpeech`] for the contract it arrives under. It
/// reaches this module as [`CaptureSignal::SpeechSoFar`], which [`feed_loop`]
/// forwards onto the caption channel.
pub use crate::audio::vad::OpenSpeech as OpenUtterance;

/// Keep only the last [`CAPTION_WINDOW_MS`] of a snapshot. A caption of the last
/// ten seconds is what a person is reading; the twenty before it are already on
/// the screen as finals or on their way there.
fn trim_to_window(mut snapshot: OpenUtterance) -> OpenUtterance {
    let keep = (CAPTION_WINDOW_MS * i64::from(crate::audio::TARGET_SAMPLE_RATE) / 1_000) as usize;
    if snapshot.samples.len() > keep {
        let dropped = snapshot.samples.len() - keep;
        snapshot.window_start_ms +=
            dropped as i64 * 1_000 / i64::from(crate::audio::TARGET_SAMPLE_RATE);
        snapshot.samples.drain(..dropped);
    }
    snapshot
}

/// Where the capture layer sends its snapshots.
pub type CaptionSender = mpsc::Sender<OpenUtterance>;
/// The pipeline's end of that.
pub type CaptionFeed = mpsc::Receiver<OpenUtterance>;

/// The live line each channel is showing, shared between the caption task and
/// the speech task.
///
/// Every partial Echo opens has to be closed by something. When a final lands it
/// closes its own line; if the caption task happened to be showing a *different*
/// line for that channel, this is how the speech task knows to retire it rather
/// than leave it on screen for the rest of the meeting.
#[derive(Debug, Default, Clone)]
struct LiveLines(Arc<Mutex<Vec<Line>>>);

/// A live line on screen, and the stretch of the meeting it covers.
#[derive(Debug, Clone, PartialEq)]
struct Line {
    channel: Channel,
    id: String,
    t_start_ms: i64,
    t_end_ms: i64,
}

impl LiveLines {
    fn showing(&self, line: Line) {
        let mut lines = self.0.lock().expect("live lines poisoned");
        match lines.iter_mut().find(|open| open.channel == line.channel) {
            Some(entry) => *entry = line,
            None => lines.push(line),
        }
    }

    /// Forget the line for this channel and say what it was.
    fn take(&self, channel: Channel) -> Option<Line> {
        let mut lines = self.0.lock().expect("live lines poisoned");
        let at = lines.iter().position(|open| open.channel == channel)?;
        Some(lines.remove(at))
    }

    fn drain(&self) -> Vec<Line> {
        std::mem::take(&mut *self.0.lock().expect("live lines poisoned"))
    }
}

/// The language this meeting has settled on, shared the same way: a caption
/// borrows it rather than paying for a detection pass of its own.
#[derive(Debug, Default, Clone)]
struct LiveLanguage(Arc<Mutex<Option<String>>>);

impl LiveLanguage {
    fn get(&self) -> Option<String> {
        self.0.lock().expect("live language poisoned").clone()
    }

    fn set(&self, language: &str) {
        *self.0.lock().expect("live language poisoned") = Some(language.to_string());
    }
}

/// Start the tasks for one recording. They end on their own when the capture
/// feed closes, which is what stopping a capture does.
pub(crate) fn spawn(
    inner: Arc<Inner>,
    meeting_id: Id,
    feed: CaptureFeed,
    backlog: Arc<Backlog>,
) -> Vec<JoinHandle<()>> {
    // The capture layer's snapshots arrive interleaved with everything else on
    // the capture feed; this is where they are split back out, so the caption
    // task never waits behind a chunk being journalled.
    let (captions_tx, captions_rx) = mpsc::channel::<OpenUtterance>(SNAPSHOT_QUEUE);
    spawn_with_captions(
        inner,
        meeting_id,
        feed,
        Some(captions_tx),
        Some(captions_rx),
        backlog,
    )
}

/// As [`spawn`], plus live captions of speech that is still going.
///
/// `captions` is the receiving end of the channel snapshots arrive on and
/// `snapshots` the sending end [`feed_loop`] forwards onto. `None` for both
/// means finals only, which is what Echo did before there was anything to
/// snapshot; the tests use that to drive one task at a time.
pub(crate) fn spawn_with_captions(
    inner: Arc<Inner>,
    meeting_id: Id,
    feed: CaptureFeed,
    snapshots: Option<CaptionSender>,
    captions: Option<CaptionFeed>,
    backlog: Arc<Backlog>,
) -> Vec<JoinHandle<()>> {
    let (utterances_tx, utterances_rx) = mpsc::channel::<Utterance>(UTTERANCE_QUEUE);
    // Capture stopping is a fact both speech tasks have to see promptly: it
    // starts the bounded handoff to the disk pass.
    let (over_tx, over_rx) = tokio::sync::watch::channel(false);
    let lines = LiveLines::default();
    let language = LiveLanguage::default();
    let stopping = Arc::new(AtomicBool::new(false));

    let feed_task = {
        let inner = inner.clone();
        let meeting_id = meeting_id.clone();
        tokio::spawn(async move {
            feed_loop(inner, meeting_id, feed, utterances_tx, snapshots).await;
            // The queue closing is what tells the speech task to drain; this is
            // what tells it *when* the drain started.
            let _ = over_tx.send(true);
        })
    };

    let mut tasks = vec![feed_task];
    if let Some(captions) = captions {
        let inner = inner.clone();
        let meeting_id = meeting_id.clone();
        let lines = lines.clone();
        let language = language.clone();
        let stopping = stopping.clone();
        tasks.push(tokio::spawn(async move {
            caption_loop(inner, meeting_id, captions, lines, language, stopping).await
        }));
    }
    tasks.push(tokio::spawn(async move {
        speech_loop(
            inner,
            meeting_id,
            utterances_rx,
            over_rx,
            lines,
            language,
            stopping,
            backlog,
        )
        .await
    }));
    tasks
}

// ---------------------------------------------------------------------------
// The backlog: what was said before the engine was ready
// ---------------------------------------------------------------------------

/// Backlog shorter than this is not worth a pass.
///
/// The engine was ready before the meeting really started — the pre-warm on
/// detection usually gets there first — and the live pass has these two seconds
/// or is about to. Below the bar, reading them back would only cost a decode and
/// blank the caption on screen for a moment.
const BACKLOG_WORTH_READING_MS: i64 = 2_000;

/// How often newly written backlog text is pushed to the live transcript.
///
/// The pass writes rows as it goes, so this is only how promptly they appear.
/// Often enough that the transcript visibly fills in, rare enough to be free.
const BACKLOG_EMIT_INTERVAL: Duration = Duration::from_millis(750);

/// Longest the disk pass waits for the live pass to confirm the handoff.
///
/// The live pass answers within one [`BATCH_INTERVAL`] of being asked, so this is
/// only what happens when it cannot answer at all — it has already stopped, or
/// it is stuck in a decode that will not abort. Reading the backlog is worth more
/// than waiting forever for a confirmation, and the coverage the pass then reads
/// is the truth as the database has it: at worst a stretch is read twice, which
/// is what a *bounded* wait trades against never catching up at all.
const BACKLOG_SETTLE_WAIT: Duration = Duration::from_millis(3_000);

// ---------------------------------------------------------------------------
// The split between the backlog pass and the live pass
// ---------------------------------------------------------------------------

/// Who owns which part of the meeting clock while the start of a meeting is
/// being read back off disk.
///
/// The engine can come up minutes after Start (mantra 1's amendment), and when
/// it does there are two passes with an interest in the same seconds of audio:
/// the live pass, which may be holding finals for them — queued for the engine,
/// in flight, or decoded and waiting in the batch — and the disk pass, about to
/// ask the database which stretches have no text against them. Both writing is
/// how one stretch of a meeting ends up in the transcript twice
/// (review of 2026-08-20, finding 1).
///
/// So the clock is split, once, at a **floor**:
///
/// * **below the floor** is the disk pass's, in order, from the beginning. The
///   live pass declines every utterance that starts there — including one
///   straddling the floor, whose tail becomes an honest hole for the
///   post-meeting job rather than an overlapping second copy.
/// * **at or above the floor** is the live pass's, and the disk pass never looks
///   past it.
///
/// A floor on its own would not be enough, because the live pass can be holding
/// finished text for the stretch below it that the database has not seen yet. So
/// claiming the floor is a handshake: the disk pass waits for the live pass to
/// say it has written down or let go of everything below the floor, and only
/// then asks what is covered. "Backlog in order, then live" is those two facts
/// together.
#[derive(Debug)]
pub(crate) struct Backlog {
    /// Where the split is, once there is one.
    floor: tokio::sync::watch::Sender<Option<i64>>,
    /// Set by the live pass when it has finished with everything below the
    /// floor.
    settled: tokio::sync::watch::Sender<bool>,
}

impl Backlog {
    /// A recording with no split in it yet, which is what every recording starts
    /// as and most stay: the engine is usually ready before the meeting is.
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            floor: tokio::sync::watch::channel(None).0,
            settled: tokio::sync::watch::channel(false).0,
        })
    }

    /// Where the split is, or `None` while the whole meeting is the live pass's.
    pub(crate) fn floor(&self) -> Option<i64> {
        *self.floor.borrow()
    }

    /// Claim everything before `to_ms` for the disk pass. Idempotent-ish: a
    /// second claim can only ever move the floor later, and there is only one
    /// backlog pass per recording, so it never happens.
    ///
    /// `send_replace`, not `send`: the value has to be stored whether or not
    /// anybody is listening yet. `send` fails when there is no live receiver and
    /// leaves the value untouched, which would mean a floor claimed while the
    /// live pass happened to be between borrows was silently no floor at all.
    fn claim(&self, to_ms: i64) {
        self.floor.send_replace(Some(to_ms));
    }

    /// Wait until there is a floor. Cancel-safe, and safe to call after the
    /// claim has already happened.
    async fn claimed(&self) {
        let mut rx = self.floor.subscribe();
        loop {
            if rx.borrow_and_update().is_some() {
                return;
            }
            if rx.changed().await.is_err() {
                // The sender lives in this same value, so this cannot happen —
                // and if it somehow does, never resolving is better than a
                // caller spinning on an answer that will not come.
                std::future::pending::<()>().await;
            }
        }
    }

    /// The live pass has written down or let go of everything below the floor.
    fn done_below_the_floor(&self) {
        self.settled.send_replace(true);
    }

    /// Wait for that, but not forever. `false` means the wait ran out.
    async fn wait_until_settled(&self, longest: Duration) -> bool {
        let mut rx = self.settled.subscribe();
        let wait = async {
            loop {
                if *rx.borrow_and_update() {
                    return;
                }
                if rx.changed().await.is_err() {
                    std::future::pending::<()>().await;
                }
            }
        };
        tokio::time::timeout(longest, wait).await.is_ok()
    }
}

/// Read this meeting's already-captured backlog into the transcript, then leave
/// the live pass to it.
///
/// Called once the engine reports ready, which on a first-ever launch can be
/// minutes after Start: the weights have to be read and the graphics compiler
/// has work to do, and capture never waits for any of it (mantra 3). Everything
/// said in the meantime is on disk, and this is what puts it in the transcript
/// while the meeting is still going — the same pass, the same coverage spans and
/// the same code path the post-meeting catch-up job uses, scoped to what has been
/// captured so far.
///
/// Four things make it safe to run underneath a live meeting, and the first two
/// are what make it "the backlog, in order, and *then* live" rather than two
/// passes racing for the same seconds (review of 2026-08-20, finding 1):
/// * the meeting clock is split at a floor, and the live pass declines
///   everything below it — see [`Backlog`];
/// * the split is a handshake, so the coverage this pass reads already includes
///   everything the live pass had in flight for that stretch;
/// * its decodes are catch-up work, which the engine serves *after* live finals,
///   so new speech never waits behind old;
/// * it stops the moment this meeting is no longer the one being recorded — from
///   there the finalize job owns the meeting, and two passes filling the same
///   holes is how a stretch gets transcribed twice.
pub(crate) async fn catch_up_backlog(inner: Arc<Inner>, meeting_id: Id, backlog: Arc<Backlog>) {
    let Some(to_ms) = backlog_end_ms(&inner, &meeting_id).await else {
        return;
    };
    if to_ms < BACKLOG_WORTH_READING_MS {
        tracing::debug!(
            to_ms,
            "speech understanding was ready in time; nothing to read back"
        );
        return;
    }

    // The split, in three steps that have to happen in this order.
    //
    // 1. The floor: from here the live pass writes nothing below `to_ms`, and
    //    this pass reads nothing above it.
    backlog.claim(to_ms);
    // 2. Everything the live pass has *queued* for that stretch goes. Every live
    //    job at this instant is a look at audio that was captured before `to_ms`
    //    — `to_ms` is "now" — so dropping all of them drops exactly the stretch
    //    being handed over, and the abandoned utterances become holes this pass
    //    fills from the recording.
    inner.ports.asr.abandon_live(&meeting_id);
    // 3. And the handshake: whatever the live pass had already finished with is
    //    written down before this pass looks at what is covered. Without it the
    //    batch waiting to be written is invisible, and invisible text is text
    //    this pass would write a second time.
    if !backlog.wait_until_settled(BACKLOG_SETTLE_WAIT).await {
        tracing::warn!(
            meeting = %meeting_id,
            "the live pass did not confirm the handoff; reading the backlog against \
             the transcript as it stands"
        );
    }

    // Everything that already has text: these lines are on screen and must not
    // be sent again.
    let mut emitted = final_segment_ids(&inner, &meeting_id, to_ms).await;
    tracing::info!(
        meeting = %meeting_id,
        to_ms,
        "reading back what was said before Echo could write it down"
    );

    let watcher = inner.clone();
    let watched = meeting_id.clone();
    let control = CatchUpControl {
        // Between windows, not mid-decode: the pass stops as soon as this
        // meeting is no longer the live one.
        cancel: Some(Arc::new(move || !is_recording_this(&watcher, &watched))),
        on_progress: None,
    };

    let pass = inner
        .ports
        .asr
        .catch_up_live(&inner.db, &meeting_id, to_ms, control);
    tokio::pin!(pass);
    let outcome = loop {
        tokio::select! {
            done = &mut pass => break done,
            _ = tokio::time::sleep(BACKLOG_EMIT_INTERVAL) => {
                emit_new_finals(&inner, &meeting_id, to_ms, &mut emitted).await;
            }
        }
    };
    // Whatever the last windows wrote, cancelled or not: the rows are there and
    // the person should see them.
    let shown = emit_new_finals(&inner, &meeting_id, to_ms, &mut emitted).await;

    match outcome {
        Ok(written) => tracing::info!(
            meeting = %meeting_id,
            written,
            shown,
            "the start of the meeting is in the transcript; live text takes over"
        ),
        // Not a failure: the recording ended, and the finalize job owns the rest.
        Err(AsrError::Cancelled) => {
            tracing::debug!(shown, "the recording ended while its backlog was being read")
        }
        Err(error) => tracing::warn!(
            %error,
            shown,
            "could not read the start of this meeting back; the catch-up job will"
        ),
    }
}

/// Is this meeting the one being recorded right now?
fn is_recording_this(inner: &Arc<Inner>, meeting_id: &str) -> bool {
    let live = inner.live.lock().expect("capture state lock");
    crate::session::is_live(live.state) && live.meeting_id.as_deref() == Some(meeting_id)
}

/// How much of this meeting has been captured so far, or `None` when it is not
/// being recorded any more.
async fn backlog_end_ms(inner: &Arc<Inner>, meeting_id: &str) -> Option<i64> {
    if !is_recording_this(inner, meeting_id) {
        return None;
    }
    let elapsed = inner
        .capture
        .lock()
        .await
        .as_ref()
        .map(|handle| handle.elapsed_ms())
        .unwrap_or(0);
    let live = inner.live.lock().expect("capture state lock").elapsed_ms;
    Some(elapsed.max(live).max(0))
}

/// The final segments this meeting already has up to `to_ms`.
async fn final_segment_ids(
    inner: &Arc<Inner>,
    meeting_id: &str,
    to_ms: i64,
) -> std::collections::HashSet<Id> {
    read_finals(inner, meeting_id, to_ms)
        .await
        .into_iter()
        .map(|segment| segment.id)
        .collect()
}

/// Push whatever the pass has written since the last look, oldest first. Returns
/// how many lines went out.
async fn emit_new_finals(
    inner: &Arc<Inner>,
    meeting_id: &str,
    to_ms: i64,
    emitted: &mut std::collections::HashSet<Id>,
) -> usize {
    let mut shown = 0;
    for segment in read_finals(inner, meeting_id, to_ms).await {
        if !emitted.insert(segment.id.clone()) {
            continue;
        }
        inner
            .ports
            .events
            .emit(UiEvent::TranscriptFinal(TranscriptFinalPayload {
                meeting_id: meeting_id.to_string(),
                // No live line to replace: this stretch happened before there
                // was anything on screen for it.
                utterance_id: None,
                segment,
            }));
        shown += 1;
    }
    shown
}

/// This meeting's final segments up to `to_ms`, in the order they were said.
async fn read_finals(inner: &Arc<Inner>, meeting_id: &str, to_ms: i64) -> Vec<Segment> {
    match repo::get_segments(
        &inner.db,
        &crate::types::TranscriptQuery {
            meeting_id: meeting_id.to_string(),
            include_partial: Some(false),
            to_ms: Some(to_ms),
            ..Default::default()
        },
    )
    .await
    {
        Ok(segments) => segments,
        Err(error) => {
            tracing::debug!(%error, "could not read the transcript back");
            Vec::new()
        }
    }
}

// ---------------------------------------------------------------------------
// Feed
// ---------------------------------------------------------------------------

async fn feed_loop(
    inner: Arc<Inner>,
    meeting_id: Id,
    mut feed: CaptureFeed,
    utterances: mpsc::Sender<Utterance>,
    snapshots: Option<CaptionSender>,
) {
    let mut last_levels: Option<Instant> = None;
    let mut told_them_we_are_behind = false;

    while let Some(update) = feed.recv().await {
        match update {
            CaptureSignal::ChunkCommitted(chunk) => {
                inner.set_elapsed(chunk.t_end_ms);
                journal(&inner, &meeting_id, &chunk).await;
            }
            CaptureSignal::UtteranceReady(utterance) => {
                inner.pending.fetch_add(1, Ordering::SeqCst);
                // A full queue means Echo is dozens of utterances behind, and
                // this one is dropped rather than blocking capture (mantra 3).
                // It is on disk and the catch-up pass reads it back; what a
                // person is *watching* stays current because the captions are a
                // lane of their own and never queue behind this
                // (review finding 5).
                if utterances.try_send(utterance).is_err() {
                    inner.pending.fetch_sub(1, Ordering::SeqCst);
                    if !told_them_we_are_behind {
                        told_them_we_are_behind = true;
                        inner.notice(NoticePayload {
                            level: NoticeLevel::Info,
                            message:
                                "The live text is a little behind. Echo will fill in the rest \
                                      when the meeting ends."
                                    .into(),
                            persistent: false,
                            meeting_id: Some(meeting_id.clone()),
                            tag: Some("transcriptBehind".into()),
                        });
                    }
                    tracing::debug!("live text dropped an utterance; the audio is on disk");
                }
            }
            CaptureSignal::SpeechSoFar(snapshot) => {
                // A caption nobody has room for is a caption not worth having:
                // the newest snapshot always follows within a second, and the
                // caption task supersedes whatever it finds waiting anyway. So
                // this never blocks and never counts as a backlog.
                if let Some(snapshots) = snapshots.as_ref() {
                    let _ = snapshots.try_send(snapshot);
                }
            }
            CaptureSignal::Levels { mic, system, t_ms } => {
                if due(&mut last_levels, LEVELS_INTERVAL) {
                    inner.ports.events.emit(UiEvent::AudioLevels(
                        crate::events::AudioLevelsPayload {
                            meeting_id: meeting_id.clone(),
                            mic,
                            system,
                            t_ms,
                        },
                    ));
                }
            }
            CaptureSignal::Degraded {
                channel,
                reason,
                message,
            } => {
                inner
                    .note_degraded(channel, reason, &message, &meeting_id)
                    .await;
            }
            CaptureSignal::Recovered { channel } => {
                inner.note_source_recovered(channel, &meeting_id);
            }
            CaptureSignal::StorageFailed { message } => {
                inner.note_fatal(&message, &meeting_id).await;
            }
        }
    }
    // Closing the channel is what tells the speech task to drain and stop.
    drop(utterances);
}

/// Journal a chunk as committed. The writer has already flushed and fsynced it,
/// so it counts as audio-on-disk the moment this row lands.
async fn journal(inner: &Arc<Inner>, meeting_id: &str, chunk: &CommittedChunk) {
    let path = chunk.path.to_string_lossy().into_owned();
    match repo::insert_chunk(
        &inner.db,
        meeting_id,
        chunk.channel,
        chunk.seq as i64,
        &path,
        chunk.t_start_ms,
        chunk.t_end_ms,
    )
    .await
    {
        Ok(id) => {
            if let Err(error) = repo::commit_chunk(&inner.db, &id, chunk.t_end_ms).await {
                tracing::warn!(%error, "could not mark a piece of the recording as saved");
            }
            if let Err(error) =
                repo::set_meeting_duration(&inner.db, meeting_id, chunk.t_end_ms).await
            {
                tracing::debug!(%error, "could not update the meeting length");
            }
        }
        Err(error) => {
            // The file itself is on disk; only the journal row failed.
            tracing::warn!(%error, seq = chunk.seq, "could not journal a piece of the recording");
            inner.notice(NoticePayload {
                level: NoticeLevel::Warning,
                message: "Echo is having trouble keeping track of this recording. Check there is \
                          space on the drive you chose."
                    .into(),
                persistent: false,
                meeting_id: Some(meeting_id.to_string()),
                tag: Some("storage".into()),
            });
        }
    }
}

fn due(last: &mut Option<Instant>, interval: Duration) -> bool {
    let now = Instant::now();
    match *last {
        Some(previous) if now.duration_since(previous) < interval => false,
        _ => {
            *last = Some(now);
            true
        }
    }
}

// ---------------------------------------------------------------------------
// Speech
// ---------------------------------------------------------------------------

/// How live transcription is going, so a broken engine is reported once instead
/// of once per utterance.
///
/// Nothing here changes what is recoverable: every failed utterance is still on
/// disk and still picked up by the catch-up pass (mantra 3). This is only about
/// what the log and the person are told.
#[derive(Debug, Default)]
struct FailureRun {
    /// Failures since the last success. Reset by any utterance that works.
    consecutive: u32,
    /// Failures for the whole recording, for the every-Nth log line.
    total: u32,
    /// The person has already been told about this recording.
    told_them: bool,
}

impl FailureRun {
    /// Record a failure. Returns whether this one is worth a log line.
    fn note_failure(&mut self) -> bool {
        self.consecutive += 1;
        self.total += 1;
        self.total == 1 || self.total.is_multiple_of(FAILURE_LOG_EVERY)
    }

    fn note_success(&mut self) {
        self.consecutive = 0;
    }

    /// Is it time to say something out loud, exactly once?
    fn should_tell_them(&mut self) -> bool {
        if self.told_them || self.consecutive < FAILURES_BEFORE_TELLING {
            return false;
        }
        self.told_them = true;
        true
    }
}

/// The stop handoff (review of 2026-08-20, finding 7).
///
/// Capture stopping starts a bounded window: the live pipeline finishes what it
/// already has, and when the window closes every live job is abandoned and the
/// disk pass is the only thing left touching this meeting. Either it drained or
/// it was cancelled — never a detached drain still decoding while catch-up works
/// out where the holes are.
#[derive(Debug, Default)]
struct Handoff {
    deadline: Option<tokio::time::Instant>,
}

impl Handoff {
    fn started(&self) -> bool {
        self.deadline.is_some()
    }

    fn deadline(&self) -> Option<tokio::time::Instant> {
        self.deadline
    }

    /// Start the window. `false` when it was already running.
    fn begin(&mut self) -> bool {
        if self.deadline.is_some() {
            return false;
        }
        self.deadline = Some(tokio::time::Instant::now() + LIVE_HANDOFF);
        true
    }

    fn expired(&self) -> bool {
        self.deadline
            .is_some_and(|at| tokio::time::Instant::now() >= at)
    }
}

/// Capture has stopped: no more captions of speech that is still going, and no
/// half-written line left on screen.
fn stop_captions(inner: &Arc<Inner>, meeting_id: &str, stopping: &AtomicBool, lines: &LiveLines) {
    stopping.store(true, Ordering::SeqCst);
    inner.ports.asr.abandon_speculative(meeting_id);
    for line in lines.drain() {
        close_partial(
            inner,
            meeting_id,
            &line.id,
            line.channel,
            line.t_start_ms,
            line.t_end_ms,
        );
    }
}

/// The window has closed. Everything live goes, and whatever had no text against
/// it is a hole on disk — which is what the catch-up pass is for.
fn hand_over(inner: &Arc<Inner>, meeting_id: &str, stopping: &AtomicBool, lines: &LiveLines) {
    stop_captions(inner, meeting_id, stopping, lines);
    inner.ports.asr.abandon_live(meeting_id);
    tracing::debug!(
        "the live pass handed this meeting over; the rest comes off the recording"
    );
}

#[allow(clippy::too_many_arguments)]
async fn speech_loop(
    inner: Arc<Inner>,
    meeting_id: Id,
    mut utterances: mpsc::Receiver<Utterance>,
    mut capture_over: tokio::sync::watch::Receiver<bool>,
    lines: LiveLines,
    language: LiveLanguage,
    stopping: Arc<AtomicBool>,
    backlog: Arc<Backlog>,
) {
    let mut batch: Vec<(String, SegmentDraft)> = Vec::new();
    let mut last_flush = Instant::now();
    let mut speakers = SpeakerCache::default();
    let mut failures = FailureRun::default();
    let mut carry = ContextCarry::default();
    let mut handoff = Handoff::default();
    // Whether the handshake with the disk pass has been answered yet.
    let mut answered_backlog = false;

    loop {
        if handoff.expired() {
            hand_over(&inner, &meeting_id, &stopping, &lines);
            break;
        }
        // The disk pass has claimed the start of this meeting. Everything the
        // live pass finished before the claim goes to the database now, so the
        // coverage that pass is about to read includes it — and then it is told
        // it may look (see [`Backlog`]).
        if !answered_backlog && backlog.floor().is_some() {
            flush(&inner, &meeting_id, &mut batch).await;
            last_flush = Instant::now();
            backlog.done_below_the_floor();
            answered_backlog = true;
        }
        let until_flush = BATCH_INTERVAL.saturating_sub(last_flush.elapsed());
        let received = tokio::select! {
            received = utterances.recv() => received,
            changed = capture_over.changed(), if !handoff.started() => {
                let _ = changed;
                handoff.begin();
                stop_captions(&inner, &meeting_id, &stopping, &lines);
                continue;
            }
            // Answered promptly rather than at the next flush: the disk pass is
            // waiting on it before it can start.
            () = backlog.claimed(), if !answered_backlog => continue,
            _ = tokio::time::sleep(until_flush) => {
                flush(&inner, &meeting_id, &mut batch).await;
                last_flush = Instant::now();
                continue;
            }
        };
        let Some(utterance) = received else {
            // Capture is over and everything it produced has text against it or
            // has been left to the disk pass on purpose.
            stop_captions(&inner, &meeting_id, &stopping, &lines);
            break;
        };
        // Below the floor is not this pass's stretch any more. Decoding it as
        // well would spend the engine on audio the disk pass is already reading
        // and write the same words a second time.
        if below_the_floor(&backlog, utterance.t_start_ms) {
            tracing::debug!(
                t_start_ms = utterance.t_start_ms,
                "this stretch is the backlog pass's; the live pass lets it go"
            );
            close_partial(
                &inner,
                &meeting_id,
                &live_line_id(utterance.channel, utterance.t_start_ms),
                utterance.channel,
                utterance.t_start_ms,
                utterance.t_end_ms,
            );
            inner.pending.fetch_sub(1, Ordering::SeqCst);
            continue;
        }
        let entry = transcribe(
            &inner,
            &meeting_id,
            utterance,
            &language,
            &lines,
            &mut speakers,
            &mut failures,
            &mut carry,
            &mut capture_over,
            &mut handoff,
            &stopping,
        )
        .await;
        if let Some((utterance_id, draft)) = entry {
            // The same rule again, after the decode rather than before it: this
            // is the one that was already in flight when the floor was claimed.
            // Its abandon may not have caught it in time, and one late row is
            // all a duplicate takes.
            if below_the_floor(&backlog, draft.t_start_ms) {
                tracing::debug!(
                    t_start_ms = draft.t_start_ms,
                    "a live decode landed inside the backlog pass's stretch; not written"
                );
                close_partial(
                    &inner,
                    &meeting_id,
                    &utterance_id,
                    draft.channel,
                    draft.t_start_ms,
                    draft.t_end_ms,
                );
            } else {
                batch.push((utterance_id, draft));
            }
        }
        inner.pending.fetch_sub(1, Ordering::SeqCst);
        if batch.len() >= BATCH_SEGMENTS {
            flush(&inner, &meeting_id, &mut batch).await;
            last_flush = Instant::now();
        }
    }
    flush(&inner, &meeting_id, &mut batch).await;
    // The live pass is over, so it is finished with everything below any floor —
    // claimed or not. Saying so releases a backlog pass still waiting on the
    // handshake instead of leaving it to wait the whole timeout out on a pass
    // that is never going to answer.
    backlog.done_below_the_floor();
}

/// Does this stretch belong to the backlog pass rather than the live one?
///
/// The start decides, not the end: an utterance that straddles the floor is one
/// utterance, and giving it to whichever pass owns its beginning is what keeps
/// the two from overlapping. Its tail is then a hole above the floor, which the
/// post-meeting catch-up job fills from the recording — a gap Echo closes later
/// rather than the same words written twice.
fn below_the_floor(backlog: &Backlog, t_start_ms: i64) -> bool {
    backlog.floor().is_some_and(|floor_ms| t_start_ms < floor_ms)
}

/// Await one live decode, honouring the stop handoff.
///
/// While capture is running this is just the decode. Once capture stops, the
/// decode gets whatever is left of [`LIVE_HANDOFF`] and is then cancelled
/// outright — a 28-second beam search finishing two minutes into the disk pass
/// is worse than the gap it would have filled.
#[allow(clippy::too_many_arguments)]
async fn decode_live(
    inner: &Arc<Inner>,
    meeting_id: &str,
    job: TranscribeJob,
    plan: DecodePlan,
    on_partial: Option<PartialFn>,
    capture_over: &mut tokio::sync::watch::Receiver<bool>,
    handoff: &mut Handoff,
    stopping: &AtomicBool,
    lines: &LiveLines,
) -> Result<Transcription, AsrError> {
    let decoding = inner.ports.asr.transcribe_live(job, plan, on_partial);
    tokio::pin!(decoding);
    loop {
        match handoff.deadline() {
            Some(at) => {
                return tokio::select! {
                    answer = &mut decoding => answer,
                    _ = tokio::time::sleep_until(at) => {
                        hand_over(inner, meeting_id, stopping, lines);
                        (&mut decoding).await
                    }
                };
            }
            None => {
                tokio::select! {
                    answer = &mut decoding => return answer,
                    changed = capture_over.changed() => {
                        let _ = changed;
                        handoff.begin();
                        stop_captions(inner, meeting_id, stopping, lines);
                    }
                }
            }
        }
    }
}

/// One utterance through the engine. `None` when there was nothing to write
/// down, or when the job was dropped to protect capture.
///
/// Whenever it returns `None` it closes the line it opened (see
/// [`close_partial`]): the live view has no other way of knowing that no final
/// is coming, and a "…" line that never settles stays there for the rest of the
/// meeting.
#[allow(clippy::too_many_arguments)]
async fn transcribe(
    inner: &Arc<Inner>,
    meeting_id: &str,
    utterance: Utterance,
    language: &LiveLanguage,
    lines: &LiveLines,
    speakers: &mut SpeakerCache,
    failures: &mut FailureRun,
    carry: &mut ContextCarry,
    capture_over: &mut tokio::sync::watch::Receiver<bool>,
    handoff: &mut Handoff,
    stopping: &AtomicBool,
) -> Option<(String, SegmentDraft)> {
    let channel = utterance.channel;
    let t_start_ms = utterance.t_start_ms;
    let t_end_ms = utterance.t_end_ms;
    let truncated = utterance.truncated;
    // The same line the captions were written on, so the final text replaces
    // them instead of appearing underneath them.
    let utterance_id = live_line_id(channel, t_start_ms);
    let hint = language.get();
    // Only across a forced cut, and only from the same channel in the same
    // language a moment earlier (codex §3 "Context and prompts").
    let prompt = carry.prompt_for(channel, t_start_ms, hint.as_deref());
    // The same words serve twice: as context going in, and as the thing the
    // overlap at the front of this utterance is matched against coming out.
    let carried = prompt.clone();

    // What the captions were showing for this channel is this utterance's
    // business now, one way or another.
    let caption = lines.take(channel);
    // A caption of this very stretch is already on screen, in full. Streaming
    // whisper's partials over it would replace ten seconds of text with the
    // first three and then grow it back — so when there is a caption to keep,
    // the final replaces it in one go instead.
    let captioned = caption.as_ref().is_some_and(|line| line.id == utterance_id);
    if let Some(stale) = caption.filter(|line| line.id != utterance_id) {
        // A stretch that ended up segmented differently than the captions
        // assumed. Retire that line rather than leave it on screen for the rest
        // of the meeting.
        close_partial(
            inner,
            meeting_id,
            &stale.id,
            channel,
            stale.t_start_ms,
            stale.t_end_ms,
        );
    }

    let job = TranscribeJob {
        meeting_id: meeting_id.to_string(),
        utterance_id: utterance_id.clone(),
        channel,
        t_start_ms,
        samples: utterance.samples,
        language_hint: hint,
        want_partials: !captioned,
        // Live work gives way when the queue is full; the catch-up pass picks
        // this stretch up from disk instead (mantra 3).
        droppable: true,
    };
    let on_partial = (!captioned).then(|| {
        partial_events(
            inner,
            meeting_id,
            &utterance_id,
            channel,
            t_start_ms,
            t_end_ms,
        )
    });

    let answer = decode_live(
        inner,
        meeting_id,
        job,
        DecodePlan::final_utterance().with_prompt(prompt),
        on_partial,
        capture_over,
        handoff,
        stopping,
        lines,
    )
    .await;

    let transcription = match answer {
        Ok(t) => t,
        // The queue was full, so this stretch was dropped on purpose. Nothing is
        // lost: it is on disk, and catch-up reads it from there (mantra 3).
        Err(error) if error.is_deferred_to_catchup() => {
            tracing::debug!("an utterance was dropped; the catch-up pass will get it from disk");
            close_partial(inner, meeting_id, &utterance_id, channel, t_start_ms, t_end_ms);
            return None;
        }
        Err(AsrError::Cancelled) => {
            tracing::debug!("the meeting went away mid-utterance");
            close_partial(inner, meeting_id, &utterance_id, channel, t_start_ms, t_end_ms);
            return None;
        }
        Err(error) => {
            // One line for the first, then one every Nth: a queue draining
            // against a broken engine used to write the same sentence dozens of
            // times a second and drown the log it was supposed to explain.
            if failures.note_failure() {
                tracing::warn!(
                    %error,
                    count = failures.total,
                    "could not write down an utterance live"
                );
            }
            if failures.should_tell_them() {
                inner.notice(NoticePayload {
                    level: NoticeLevel::Warning,
                    message: "Echo is having trouble writing things down — it will catch up from \
                              the recording afterwards."
                        .into(),
                    persistent: false,
                    meeting_id: Some(meeting_id.to_string()),
                    tag: Some("liveTextTrouble".into()),
                });
            }
            close_partial(inner, meeting_id, &utterance_id, channel, t_start_ms, t_end_ms);
            return None;
        }
    };
    failures.note_success();

    // The join, reconciled rather than concatenated. When the piece before this
    // one was cut through speech, this one restarted inside it and has just
    // re-read the last of its words; saying them twice is the artefact the
    // overlap trades a sliced word for, and this is where it is paid back.
    let text = match carried.as_deref() {
        Some(carried) => strip_overlap(transcription.text.trim(), carried),
        None => transcription.text.trim().to_string(),
    };
    let text = text.trim();
    if text.is_empty() {
        carry.forget(channel);
        close_partial(inner, meeting_id, &utterance_id, channel, t_start_ms, t_end_ms);
        return None;
    }

    if language.get().is_none() {
        if let Some(detected) = transcription.language.clone() {
            language.set(&detected);
            if let Err(error) = repo::set_meeting_language(&inner.db, meeting_id, &detected).await {
                tracing::debug!(%error, "could not store the meeting language");
            }
        }
    }

    let speaker_id = channel_speaker(inner, meeting_id, channel, speakers).await;
    // What the engine actually read, never what we hoped it read
    // (review finding 1).
    let (span_start_ms, span_end_ms) = transcribed_span(t_start_ms, t_end_ms, &transcription);
    // Hold the tail only when this piece was cut mid-sentence: the next one on
    // this channel is the rest of that sentence.
    carry.remember(
        channel,
        truncated,
        text,
        transcription.language.as_deref(),
        transcription.avg_confidence,
        span_end_ms,
    );

    Some((
        utterance_id,
        SegmentDraft {
            meeting_id: meeting_id.to_string(),
            t_start_ms: span_start_ms,
            t_end_ms: span_end_ms,
            channel,
            speaker_id,
            text: text.to_string(),
            language: transcription.language,
            avg_confidence: transcription.avg_confidence,
            revision: 1,
            is_final: true,
            model_name: transcription.model_name,
            model_revision: transcription.model_revision,
        },
    ))
}

/// The stretch a segment may claim: what the engine read, not what it was handed.
///
/// The pipeline used to write `max(whisper_end, utterance_end)` here
/// (review finding 1). Since the catch-up pass subtracts these spans from the
/// audio on disk to find its holes, claiming the whole padded utterance when
/// whisper stopped halfway through it is how the missing tail of a sentence
/// becomes permanently missing. So: whisper's end, clamped to the audio it was
/// given, and never before the start.
fn transcribed_span(
    utterance_start_ms: i64,
    utterance_end_ms: i64,
    transcription: &Transcription,
) -> (i64, i64) {
    let start = if transcription.t_start_ms > 0 {
        transcription.t_start_ms
    } else {
        utterance_start_ms.max(0)
    }
    .clamp(0, utterance_end_ms.max(0));
    let full = utterance_end_ms.max(start);
    let end = if transcription.t_end_ms > start {
        transcription.t_end_ms.min(full)
    } else {
        // The engine reported no end at all — which it only does when it had no
        // timestamps to give. A zero-length segment would be a worse lie than
        // either answer, so the window stands.
        full
    };
    (start, end)
}

// ---------------------------------------------------------------------------
// Context across a forced cut
// ---------------------------------------------------------------------------

/// The tail of the previous final on a channel, held for its continuation.
///
/// The segmenter cuts a monologue at its hard cap and marks the piece
/// `truncated`. Nothing read that flag (review of 2026-08-20, §3), so the
/// continuation started with no idea what sentence it was in the middle of —
/// which is one direct cause of transcripts full of sentences that stop
/// mid-thought.
///
/// What is carried, and when it is dropped, follows the review exactly: at most
/// [`CARRY_WORDS`] words, same channel, same language, no long silence in
/// between, and nothing that was itself shaky or repetitive.
#[derive(Debug, Default)]
struct ContextCarry {
    held: Vec<(Channel, Carry)>,
}

#[derive(Debug, Clone, PartialEq)]
struct Carry {
    words: String,
    language: Option<String>,
    ends_at_ms: i64,
}

impl ContextCarry {
    /// The prompt for the next utterance on this channel, if the carry still
    /// applies. Consumed either way: a stale one must not resurface later.
    fn prompt_for(
        &mut self,
        channel: Channel,
        starts_at_ms: i64,
        language: Option<&str>,
    ) -> Option<String> {
        let at = self.held.iter().position(|(c, _)| *c == channel)?;
        let (_, carry) = self.held.remove(at);
        // A pause, a lost source, a resumed recording: all of them show up here
        // as a gap far longer than the join we were carrying across.
        if starts_at_ms - carry.ends_at_ms > CARRY_MAX_GAP_MS {
            return None;
        }
        // Somebody switched language mid-meeting. Italian context is worse than
        // no context for an English sentence.
        if let (Some(now), Some(before)) = (language, carry.language.as_deref()) {
            if !now.eq_ignore_ascii_case(before) {
                return None;
            }
        }
        Some(carry.words)
    }

    /// Remember the tail of a final — or deliberately forget, which is most of
    /// the time.
    fn remember(
        &mut self,
        channel: Channel,
        truncated: bool,
        text: &str,
        language: Option<&str>,
        confidence: Option<f32>,
        ends_at_ms: i64,
    ) {
        // Only a forced cut leaves a sentence in mid-air. An utterance that
        // ended in silence is finished, and prompting the next one with it is
        // how whisper starts repeating itself.
        if !truncated
            || confidence.is_some_and(|c| c < CARRY_MIN_CONFIDENCE)
            || looks_repetitive(text)
        {
            self.forget(channel);
            return;
        }
        let words = tail_words(text, CARRY_WORDS);
        if words.is_empty() {
            self.forget(channel);
            return;
        }
        let carry = Carry {
            words,
            language: language.map(str::to_string),
            ends_at_ms,
        };
        match self.held.iter_mut().find(|(c, _)| *c == channel) {
            Some(entry) => entry.1 = carry,
            None => self.held.push((channel, carry)),
        }
    }

    fn forget(&mut self, channel: Channel) {
        self.held.retain(|(c, _)| *c != channel);
    }
}

/// Most words the forced-cut overlap can plausibly hold.
///
/// The overlap is [`crate::audio::vad::VadSettings::forced_overlap_ms`] — 750 ms
/// — and fast speech is around five words a second. Six is generous; looking
/// further back would start matching phrases that were genuinely said twice.
const OVERLAP_MAX_WORDS: usize = 6;

/// Fewest words that count as a join rather than a coincidence. One repeated
/// word is a normal thing to say ("sì, sì"); a repeated pair at exactly the
/// point two decodes were spliced is the splice.
const OVERLAP_MIN_WORDS: usize = 2;

/// Drop the words at the start of a continuation that the piece before it
/// already said.
///
/// A forced cut restarts the continuation
/// [`crate::audio::vad::VadSettings::forced_overlap_ms`] earlier, so a word
/// straddling the join is whole in the second piece instead of being sliced in
/// half — that is what stops the transcript reading like it was cut mid-word.
/// The cost is that the overlapping audio is decoded twice and those words
/// arrive twice, which is what this removes.
///
/// Deliberately exact and word-aligned: the two decodes see different context
/// and often word the overlap differently, and when they disagree this does
/// nothing rather than guess. A duplicated phrase is a blemish; deleting words
/// somebody said is a lie.
fn strip_overlap(text: &str, carried: &str) -> String {
    let before: Vec<String> = normalized_words(carried);
    let after_raw: Vec<&str> = text.split_whitespace().collect();
    let after: Vec<String> = normalized_words(text);
    let most = OVERLAP_MAX_WORDS.min(before.len()).min(after.len());
    for k in (OVERLAP_MIN_WORDS..=most).rev() {
        if before[before.len() - k..] == after[..k] {
            let kept = after_raw[k..].join(" ");
            // Never turn a real utterance into an empty one: if the whole
            // continuation was overlap, the piece before it already has these
            // words and this one has nothing to add, but an empty segment would
            // read as a failed decode. Keep it whole instead.
            if !kept.trim().is_empty() {
                return kept;
            }
        }
    }
    text.to_string()
}

fn normalized_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .collect()
}

/// The last `count` words, which is as much context as is worth carrying.
fn tail_words(text: &str, count: usize) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    let from = words.len().saturating_sub(count);
    words[from..].join(" ")
}

/// Has this text already fallen into a loop?
///
/// Feeding a repetition back in as context is how a stuck decoder stays stuck,
/// so a piece that looks like one is not carried anywhere.
fn looks_repetitive(text: &str) -> bool {
    let words: Vec<String> = text
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
        .filter(|w| !w.is_empty())
        .collect();
    if words.len() < 4 {
        return false;
    }
    // The same word three times over, or the same pair of words three times.
    words.windows(3).any(|w| w[0] == w[1] && w[1] == w[2])
        || words
            .windows(6)
            .any(|w| w[0..2] == w[2..4] && w[2..4] == w[4..6])
}

// ---------------------------------------------------------------------------
// Captions of speech that is still going
// ---------------------------------------------------------------------------

/// The snapshots waiting to be decoded: at most one per channel, and only ever
/// the newest.
///
/// A newer look at the same speech **replaces** the one waiting rather than
/// queueing behind it (codex: "a new snapshot supersedes the queued old one").
/// Decoding both would spend the encoder twice to show the older answer for a
/// moment and then throw it away.
#[derive(Debug, Default)]
struct Speculation {
    waiting: Vec<OpenUtterance>,
    /// Which channel was captioned last, so the turns alternate.
    last: Option<Channel>,
}

impl Speculation {
    fn offer(&mut self, snapshot: OpenUtterance) {
        match self
            .waiting
            .iter_mut()
            .find(|s| s.channel == snapshot.channel)
        {
            Some(slot) => *slot = snapshot,
            None => self.waiting.push(snapshot),
        }
    }

    /// The next snapshot to decode, alternating channels: one continuously open
    /// microphone must not starve the system channel (codex §2).
    fn take_next(&mut self) -> Option<OpenUtterance> {
        if self.waiting.is_empty() {
            return None;
        }
        let index = self
            .last
            .and_then(|last| self.waiting.iter().position(|s| s.channel != last))
            .unwrap_or(0);
        let taken = self.waiting.remove(index);
        self.last = Some(taken.channel);
        Some(taken)
    }
}

/// One caption per channel per [`CAPTION_STEP_MS`] of *new speech*.
///
/// Measured in audio rather than wall clock, so a decode that takes longer than
/// the step turns into one caption covering more speech instead of a backlog of
/// near-identical windows. A stretch that has only just opened is captioned at
/// once: making the first words of an answer wait three seconds is the very
/// thing captions exist to fix.
#[derive(Debug, Default)]
struct Cadence {
    seen: Vec<(Channel, i64, i64)>,
}

impl Cadence {
    fn advanced(&mut self, snapshot: &OpenUtterance) -> bool {
        let end = snapshot.window_end_ms();
        match self
            .seen
            .iter_mut()
            .find(|(channel, _, _)| *channel == snapshot.channel)
        {
            Some(entry) => {
                let new_line = entry.1 != snapshot.t_start_ms;
                if new_line || end - entry.2 >= CAPTION_STEP_MS {
                    entry.1 = snapshot.t_start_ms;
                    entry.2 = end;
                    true
                } else {
                    false
                }
            }
            None => {
                self.seen.push((snapshot.channel, snapshot.t_start_ms, end));
                true
            }
        }
    }
}

/// The live line a channel's current stretch of speech is written on.
///
/// Stable for as long as that stretch stays open, and the same id the final
/// utterance is emitted under, so the final text *replaces* the caption rather
/// than appearing underneath it.
fn live_line_id(channel: Channel, t_start_ms: i64) -> String {
    format!("live-{}-{t_start_ms}", channel_key(channel))
}

fn channel_key(channel: Channel) -> &'static str {
    match channel {
        Channel::Mic => "mic",
        Channel::System => "system",
        Channel::Mixed => "mixed",
    }
}

/// Decode snapshots of speech that is still going and put the words on screen.
///
/// Everything here is disposable: a caption is never written to the database,
/// never counted as coverage, and always replaced — by the next caption, by the
/// final utterance, or by nothing at all when the meeting stops.
async fn caption_loop(
    inner: Arc<Inner>,
    meeting_id: Id,
    mut snapshots: CaptionFeed,
    lines: LiveLines,
    language: LiveLanguage,
    stopping: Arc<AtomicBool>,
) {
    let mut waiting = Speculation::default();
    let mut cadence = Cadence::default();
    let mut closed = false;

    while !stopping.load(Ordering::SeqCst) {
        // Everything that arrived while the last caption was decoding, newest
        // per channel only. The rest never happened.
        while let Ok(snapshot) = snapshots.try_recv() {
            waiting.offer(trim_to_window(snapshot));
        }
        let Some(snapshot) = waiting.take_next() else {
            if closed {
                break;
            }
            match snapshots.recv().await {
                Some(snapshot) => {
                    waiting.offer(trim_to_window(snapshot));
                    continue;
                }
                None => break,
            }
        };
        if !cadence.advanced(&snapshot) || snapshot.duration_ms() < CAPTION_MIN_MS {
            continue;
        }

        let channel = snapshot.channel;
        let covers_to_ms = snapshot.window_end_ms();
        let decoding = caption(&inner, &meeting_id, snapshot, &lines, &language);
        tokio::pin!(decoding);
        loop {
            tokio::select! {
                () = &mut decoding => break,
                received = snapshots.recv(), if !closed => {
                    match received {
                        Some(snapshot) => {
                            let snapshot = trim_to_window(snapshot);
                            // Speech has moved well past the window we are still
                            // waiting on: the answer will be stale the moment it
                            // lands, so stop paying for it and take the newer one
                            // (codex §2 "compute waste").
                            let overtaken = snapshot.channel == channel
                                && snapshot.window_end_ms() - covers_to_ms >= STALE_CAPTION_MS;
                            waiting.offer(snapshot);
                            if overtaken {
                                inner.ports.asr.abandon_speculative(&meeting_id);
                            }
                        }
                        None => closed = true,
                    }
                }
            }
        }
    }
    tracing::debug!(meeting = %meeting_id, "live captions stopped");
}

/// One caption: the cheapest decode there is, and one event that replaces the
/// whole line.
async fn caption(
    inner: &Arc<Inner>,
    meeting_id: &str,
    snapshot: OpenUtterance,
    lines: &LiveLines,
    language: &LiveLanguage,
) {
    let channel = snapshot.channel;
    let line = live_line_id(channel, snapshot.t_start_ms);
    let t_start_ms = snapshot.t_start_ms;
    let t_end_ms = snapshot.window_end_ms();

    let job = TranscribeJob {
        meeting_id: meeting_id.to_string(),
        utterance_id: line.clone(),
        channel,
        t_start_ms: snapshot.window_start_ms,
        samples: snapshot.samples,
        // Whatever the meeting has settled on. A caption never pays for a
        // detection pass of its own.
        language_hint: language.get(),
        // The result *is* the partial; there is nothing to stream out of it.
        want_partials: false,
        droppable: true,
    };

    match inner
        .ports
        .asr
        .transcribe_live(job, DecodePlan::speculative(), None)
        .await
    {
        Ok(transcription) => {
            let text = transcription.text.trim();
            if text.is_empty() {
                return;
            }
            lines.showing(Line {
                channel,
                id: line.clone(),
                t_start_ms,
                t_end_ms,
            });
            inner
                .ports
                .events
                .emit(UiEvent::TranscriptPartial(TranscriptPartialPayload {
                    meeting_id: meeting_id.to_string(),
                    utterance_id: line,
                    t_start_ms,
                    t_end_ms,
                    channel,
                    speaker_id: None,
                    text: text.to_string(),
                    language: transcription.language,
                    dropped: false,
                }));
        }
        // A newer look at the same speech overtook this one, or there was no room
        // for it. Both are the queue working as intended.
        Err(AsrError::Cancelled) => {}
        Err(error) if error.is_deferred_to_catchup() => {}
        Err(error) => {
            // Captions are cosmetic: nobody is told, and the final utterance is
            // still coming.
            tracing::debug!(%error, "a live caption did not come out");
        }
    }
}

/// The two live speakers, looked up once per recording.
#[derive(Debug, Default)]
struct SpeakerCache {
    mic: Option<Id>,
    system: Option<Id>,
}

impl SpeakerCache {
    fn get(&self, channel: Channel) -> Option<&Id> {
        match channel {
            Channel::Mic => self.mic.as_ref(),
            Channel::System => self.system.as_ref(),
            Channel::Mixed => None,
        }
    }

    fn set(&mut self, channel: Channel, id: Id) {
        match channel {
            Channel::Mic => self.mic = Some(id),
            Channel::System => self.system = Some(id),
            Channel::Mixed => {}
        }
    }
}

/// The speaker row a channel's words belong to: cluster key, name, and whether
/// it is the person using the computer.
///
/// These are the *same* keys the diarize pass uses, not lookalikes of them. A
/// speaker row is identified by its cluster key, so keying the microphone as
/// `"mic"` here while [`crate::diarize::pin_channel_speakers`] keyed it as
/// `"you"` produced two rows both called "You" for one person — and, once system
/// audio started flowing, two called "Speaker 1" for the other. Sharing the keys
/// means the live pass, the provisional pass and the offline refine all land on
/// one row per speaker, and a rename survives all three.
fn channel_speaker_identity(channel: Channel) -> Option<(String, String, bool)> {
    match channel {
        Channel::Mic => Some((
            crate::diarize::SELF_CLUSTER_KEY.to_string(),
            crate::diarize::SELF_DISPLAY_NAME.to_string(),
            true,
        )),
        Channel::System => Some((
            crate::diarize::cluster_key(0),
            crate::diarize::display_name(0),
            false,
        )),
        Channel::Mixed => None,
    }
}

/// Live speaker attribution is the channel: the microphone is the person using
/// the computer, everything else is provisional until the offline pass runs.
async fn channel_speaker(
    inner: &Arc<Inner>,
    meeting_id: &str,
    channel: Channel,
    cache: &mut SpeakerCache,
) -> Option<Id> {
    if let Some(id) = cache.get(channel) {
        return Some(id.clone());
    }
    let (cluster_key, display_name, is_self) = channel_speaker_identity(channel)?;
    match repo::upsert_speaker(&inner.db, meeting_id, &cluster_key, &display_name, is_self).await {
        Ok(speaker) => {
            cache.set(channel, speaker.id.clone());
            Some(speaker.id)
        }
        Err(error) => {
            tracing::debug!(%error, "could not name a speaker yet");
            None
        }
    }
}

/// Tell the live view that this utterance is over with nothing to show.
///
/// Every partial Echo opens has to be closed, either by a final or by this. The
/// three ways an utterance ends with no text — dropped to protect capture, heard
/// as silence, or an engine that could not read it — all end up here, so the
/// half-written line disappears instead of sitting in the transcript for the
/// rest of the meeting.
fn close_partial(
    inner: &Arc<Inner>,
    meeting_id: &str,
    utterance_id: &str,
    channel: Channel,
    t_start_ms: i64,
    t_end_ms: i64,
) {
    inner
        .ports
        .events
        .emit(UiEvent::TranscriptPartial(TranscriptPartialPayload {
            meeting_id: meeting_id.to_string(),
            utterance_id: utterance_id.to_string(),
            t_start_ms,
            t_end_ms,
            channel,
            speaker_id: None,
            text: String::new(),
            language: None,
            dropped: true,
        }));
}

fn partial_events(
    inner: &Arc<Inner>,
    meeting_id: &str,
    utterance_id: &str,
    channel: Channel,
    t_start_ms: i64,
    t_end_ms: i64,
) -> PartialFn {
    let events = inner.ports.events.clone();
    let meeting_id = meeting_id.to_string();
    let utterance_id = utterance_id.to_string();
    let last: std::sync::Mutex<Option<Instant>> = std::sync::Mutex::new(None);
    Box::new(move |text: &str| {
        {
            let Ok(mut guard) = last.lock() else { return };
            if !due(&mut guard, PARTIAL_INTERVAL) {
                // Live text is cosmetic; skipping one costs nothing.
                return;
            }
        }
        events.emit(UiEvent::TranscriptPartial(TranscriptPartialPayload {
            meeting_id: meeting_id.clone(),
            utterance_id: utterance_id.clone(),
            t_start_ms,
            t_end_ms,
            channel,
            speaker_id: None,
            text: text.to_string(),
            language: None,
            dropped: false,
        }));
    })
}

/// One transaction for the whole batch, then one event per segment
/// (DESIGN §3: "segments ─► SQLite (batched) + UI events (rate-capped)").
async fn flush(inner: &Arc<Inner>, meeting_id: &str, batch: &mut Vec<(String, SegmentDraft)>) {
    if batch.is_empty() {
        return;
    }
    let drafts: Vec<SegmentDraft> = batch.iter().map(|(_, draft)| draft.clone()).collect();
    match repo::insert_segments(&inner.db, &drafts).await {
        Ok(ids) => {
            for ((utterance_id, draft), id) in batch.drain(..).zip(ids) {
                inner
                    .ports
                    .events
                    .emit(UiEvent::TranscriptFinal(TranscriptFinalPayload {
                        meeting_id: meeting_id.to_string(),
                        utterance_id: Some(utterance_id),
                        segment: segment_of(id, draft),
                    }));
            }
        }
        Err(error) => {
            // The words are lost, the audio is not: catch-up will redo this
            // stretch from disk. The lines still have to be closed, or the live
            // view keeps a half-written one for each of them.
            tracing::warn!(%error, count = batch.len(), "could not store live text");
            for (utterance_id, draft) in batch.drain(..) {
                close_partial(
                    inner,
                    meeting_id,
                    &utterance_id,
                    draft.channel,
                    draft.t_start_ms,
                    draft.t_end_ms,
                );
            }
        }
    }
}

fn segment_of(id: Id, draft: SegmentDraft) -> Segment {
    Segment {
        id,
        meeting_id: draft.meeting_id,
        t_start_ms: draft.t_start_ms,
        t_end_ms: draft.t_end_ms,
        channel: draft.channel,
        speaker_id: draft.speaker_id,
        text: draft.text,
        language: draft.language,
        avg_confidence: draft.avg_confidence,
        revision: draft.revision.max(1),
        is_final: draft.is_final,
        model_name: draft.model_name,
        model_revision: draft.model_revision,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_bad_utterance_is_logged_once_and_told_to_nobody() {
        let mut run = FailureRun::default();
        assert!(run.note_failure(), "the first one always gets a line");
        assert!(
            !run.should_tell_them(),
            "one utterance the engine could not read is not worth interrupting anybody"
        );
        // And an utterance that works clears the run.
        run.note_success();
        assert_eq!(run.consecutive, 0);
        assert!(!run.should_tell_them());
    }

    #[test]
    fn a_broken_engine_writes_one_line_then_every_nth() {
        let mut run = FailureRun::default();
        let logged = (1..=100).filter(|_| run.note_failure()).count();
        // The first, then every FAILURE_LOG_EVERY-th: 1, 25, 50, 75, 100.
        assert_eq!(logged, 1 + 100 / FAILURE_LOG_EVERY as usize);
        assert_eq!(run.total, 100);
    }

    #[test]
    fn after_a_run_of_failures_the_person_is_told_exactly_once() {
        let mut run = FailureRun::default();
        for i in 1..FAILURES_BEFORE_TELLING {
            run.note_failure();
            assert!(!run.should_tell_them(), "too early at {i}");
        }
        run.note_failure();
        assert!(run.should_tell_them(), "ten in a row is worth one sentence");
        // Never twice, however long it goes on.
        for _ in 0..500 {
            run.note_failure();
            assert!(!run.should_tell_them());
        }
    }

    // -----------------------------------------------------------------------
    // The forced-cut join
    // -----------------------------------------------------------------------

    #[test]
    fn words_the_overlap_read_twice_are_only_written_once() {
        // The cut fell after "riunione"; the continuation restarted 750 ms
        // earlier and re-read "della riunione" before carrying on.
        let carried = "il primo punto della riunione";
        let heard = "della riunione è il bilancio";
        assert_eq!(strip_overlap(heard, carried), "è il bilancio");
    }

    #[test]
    fn punctuation_and_case_do_not_stop_the_join_being_found() {
        // The two decodes saw different context, so they punctuated the shared
        // words differently. It is still the same join.
        assert_eq!(
            strip_overlap("Della riunione, è il bilancio", "il primo punto della riunione"),
            "è il bilancio"
        );
    }

    #[test]
    fn a_join_that_does_not_match_is_left_alone() {
        // The overlap was worded differently by the second decode. Guessing
        // here would delete words somebody said.
        let heard = "e poi il bilancio";
        assert_eq!(strip_overlap(heard, "il primo punto della riunione"), heard);
    }

    #[test]
    fn one_repeated_word_is_not_a_join() {
        // "sì" ending one piece and starting the next is a thing people say,
        // not evidence of a splice.
        assert_eq!(strip_overlap("sì certo", "va bene sì"), "sì certo");
    }

    #[test]
    fn a_continuation_that_was_all_overlap_is_kept_whole() {
        // Stripping everything would leave an empty segment, which reads as a
        // failed decode rather than as a join.
        let heard = "della riunione";
        assert_eq!(strip_overlap(heard, "il punto della riunione"), heard);
    }

    #[test]
    fn a_long_repeat_is_not_mistaken_for_the_overlap() {
        // Only the last few words can possibly be shared: the overlap is
        // 750 ms. A phrase repeated further back is somebody repeating himself.
        let carried = "uno due tre quattro cinque sei sette otto";
        let heard = "uno due tre quattro cinque sei sette otto nove";
        // Nothing matches within OVERLAP_MAX_WORDS of the boundary, so it stands.
        assert_eq!(strip_overlap(heard, carried), heard);
        const { assert!(OVERLAP_MAX_WORDS < 8) };
        const {
            assert!(
                OVERLAP_MIN_WORDS >= 2,
                "one shared word is a coincidence, not a splice"
            )
        };
    }

    // -----------------------------------------------------------------------
    // Speakers
    // -----------------------------------------------------------------------

    #[test]
    fn live_speakers_are_the_same_rows_the_diarize_pass_uses() {
        let (mic_key, mic_name, mic_is_self) =
            channel_speaker_identity(Channel::Mic).expect("the microphone is somebody");
        // Keyed identically to `pin_channel_speakers`, or the person gets two
        // rows called "You" and the offline pass cannot reuse either.
        assert_eq!(mic_key, crate::diarize::SELF_CLUSTER_KEY);
        assert_eq!(mic_name, crate::diarize::SELF_DISPLAY_NAME);
        assert!(mic_is_self);

        let (sys_key, sys_name, sys_is_self) =
            channel_speaker_identity(Channel::System).expect("what the computer plays is somebody");
        assert_eq!(sys_key, crate::diarize::cluster_key(0));
        assert_eq!(sys_name, crate::diarize::display_name(0));
        assert!(!sys_is_self, "the far end is not the person recording");
        assert_ne!(mic_key, sys_key, "You and Speaker 1 are two people");

        // Zero jargon in anything a person reads (mantra 2).
        assert_eq!(mic_name, "You");
        assert_eq!(sys_name, "Speaker 1");

        // The mixed playback channel is not a speaker at all.
        assert!(channel_speaker_identity(Channel::Mixed).is_none());
    }

    // -----------------------------------------------------------------------
    // Captions
    // -----------------------------------------------------------------------

    fn snapshot(channel: Channel, start_ms: i64, window_start_ms: i64, ms: i64) -> OpenUtterance {
        OpenUtterance {
            channel,
            t_start_ms: start_ms,
            window_start_ms,
            samples: vec![0.0; (ms as usize) * 16],
        }
    }

    #[test]
    fn a_newer_look_at_the_same_speech_replaces_the_one_waiting() {
        let mut waiting = Speculation::default();
        waiting.offer(snapshot(Channel::Mic, 0, 0, 3_000));
        waiting.offer(snapshot(Channel::Mic, 0, 0, 6_000));
        assert_eq!(
            waiting.waiting.len(),
            1,
            "two looks at one channel are one job, never two"
        );
        assert_eq!(waiting.waiting[0].window_end_ms(), 6_000);

        // Even across a forced cut, where the stretch itself is new: still one
        // caption per channel, and it is the newest one.
        waiting.offer(snapshot(Channel::Mic, 6_000, 6_000, 2_000));
        assert_eq!(waiting.waiting.len(), 1);
        assert_eq!(waiting.waiting[0].t_start_ms, 6_000);
    }

    #[test]
    fn captions_take_turns_so_one_open_microphone_cannot_starve_the_room() {
        let mut waiting = Speculation::default();
        waiting.offer(snapshot(Channel::Mic, 0, 0, 3_000));
        waiting.offer(snapshot(Channel::System, 0, 0, 3_000));
        assert_eq!(waiting.take_next().unwrap().channel, Channel::Mic);

        // The microphone is still going; the system channel is next anyway.
        waiting.offer(snapshot(Channel::Mic, 0, 0, 6_000));
        assert_eq!(waiting.take_next().unwrap().channel, Channel::System);
        assert_eq!(waiting.take_next().unwrap().channel, Channel::Mic);
        assert!(waiting.take_next().is_none());
    }

    #[test]
    fn a_caption_waits_for_new_speech_but_the_first_words_never_do() {
        let mut cadence = Cadence::default();
        // The first look at a stretch is always worth decoding.
        assert!(cadence.advanced(&snapshot(Channel::Mic, 0, 0, 1_500)));
        // A second later there is nothing new to say.
        assert!(!cadence.advanced(&snapshot(Channel::Mic, 0, 0, 2_400)));
        // Three seconds on, there is.
        assert!(cadence.advanced(&snapshot(Channel::Mic, 0, 0, 4_500)));
        // The other channel keeps its own pace.
        assert!(cadence.advanced(&snapshot(Channel::System, 0, 0, 1_200)));
        // And a new stretch — the continuation after a forced cut — is captioned
        // at once rather than waiting out the step.
        assert!(cadence.advanced(&snapshot(Channel::Mic, 4_500, 4_500, 1_200)));
    }

    #[test]
    fn a_caption_is_decoded_from_the_last_ten_seconds_and_no_more() {
        let long = trim_to_window(snapshot(Channel::Mic, 0, 0, 26_000));
        assert_eq!(long.duration_ms(), CAPTION_WINDOW_MS);
        assert_eq!(
            long.window_start_ms,
            16_000,
            "the window moves with the speech, so its timestamps stay the meeting's"
        );
        assert_eq!(long.window_end_ms(), 26_000);
        assert_eq!(
            long.t_start_ms, 0,
            "the line is still the same line: the whole stretch's start"
        );

        // Short enough already: untouched.
        let short = trim_to_window(snapshot(Channel::Mic, 1_000, 1_000, 4_000));
        assert_eq!(short.duration_ms(), 4_000);
        assert_eq!(short.window_start_ms, 1_000);
    }

    /// The final has to land on the line the captions were written on, or a
    /// person sees the same sentence twice.
    #[test]
    fn the_final_text_replaces_the_caption_on_the_same_line() {
        let open = snapshot(Channel::Mic, 4_000, 8_000, 2_000);
        assert_eq!(
            live_line_id(open.channel, open.t_start_ms),
            live_line_id(Channel::Mic, 4_000),
            "the same stretch is the same line, whatever window it was captioned from"
        );
        assert_ne!(
            live_line_id(Channel::Mic, 4_000),
            live_line_id(Channel::System, 4_000)
        );
        assert_ne!(
            live_line_id(Channel::Mic, 4_000),
            live_line_id(Channel::Mic, 32_000),
            "the continuation after a forced cut is a line of its own"
        );
    }

    // -----------------------------------------------------------------------
    // Coverage
    // -----------------------------------------------------------------------

    fn read_up_to(start_ms: i64, end_ms: i64) -> Transcription {
        Transcription {
            t_start_ms: start_ms,
            t_end_ms: end_ms,
            text: "qualcosa".into(),
            ..Default::default()
        }
    }

    /// Review finding 1. The catch-up pass subtracts these spans from the audio
    /// on disk, so a segment that claims audio nobody transcribed is how the
    /// missing half of a sentence becomes permanently missing.
    #[test]
    fn a_segment_claims_only_the_audio_the_engine_actually_read() {
        // 28 seconds handed over, whisper stopped at 12.
        let span = transcribed_span(10_000, 38_000, &read_up_to(10_000, 22_000));
        assert_eq!(
            span,
            (10_000, 22_000),
            "the untranscribed tail stays visible to the catch-up pass"
        );

        // It never claims more than the audio it was given, either.
        assert_eq!(
            transcribed_span(10_000, 38_000, &read_up_to(10_000, 99_000)),
            (10_000, 38_000)
        );

        // Nothing usable reported: the whole window is the only honest answer.
        assert_eq!(
            transcribed_span(10_000, 38_000, &read_up_to(0, 0)),
            (10_000, 38_000)
        );

        // And a start the engine placed later than the utterance is kept.
        assert_eq!(
            transcribed_span(10_000, 38_000, &read_up_to(11_500, 20_000)),
            (11_500, 20_000)
        );
    }

    // -----------------------------------------------------------------------
    // Context across a forced cut
    // -----------------------------------------------------------------------

    #[test]
    fn context_crosses_a_forced_cut_and_nothing_else() {
        let mut carry = ContextCarry::default();

        // An utterance that ended in silence is finished: nothing is carried.
        carry.remember(Channel::Mic, false, "e quindi ci siamo", Some("it"), Some(0.9), 8_000);
        assert!(carry.prompt_for(Channel::Mic, 8_200, Some("it")).is_none());

        // One cut short by the hard cap hands its tail to the continuation.
        carry.remember(
            Channel::Mic,
            true,
            "allora il punto principale della riunione è",
            Some("it"),
            Some(0.9),
            28_000,
        );
        let prompt = carry
            .prompt_for(Channel::Mic, 28_000, Some("it"))
            .expect("the continuation of a cut sentence gets its context");
        assert!(prompt.ends_with("riunione è"));
        // Consumed: it can never resurface later in the meeting.
        assert!(carry.prompt_for(Channel::Mic, 28_000, Some("it")).is_none());
    }

    #[test]
    fn the_carried_context_is_dropped_the_moment_it_stops_applying() {
        let cut = |carry: &mut ContextCarry| {
            carry.remember(Channel::Mic, true, "e il secondo punto invece", Some("it"), Some(0.9), 28_000);
        };

        // A silence longer than a breath: this is not that sentence any more.
        let mut carry = ContextCarry::default();
        cut(&mut carry);
        assert!(carry
            .prompt_for(Channel::Mic, 28_000 + CARRY_MAX_GAP_MS + 1, Some("it"))
            .is_none());

        // Somebody switched language.
        let mut carry = ContextCarry::default();
        cut(&mut carry);
        assert!(carry.prompt_for(Channel::Mic, 28_000, Some("en")).is_none());

        // The other channel is somebody else talking.
        let mut carry = ContextCarry::default();
        cut(&mut carry);
        assert!(carry.prompt_for(Channel::System, 28_000, Some("it")).is_none());
        assert!(carry.prompt_for(Channel::Mic, 28_000, Some("it")).is_some());

        // Text the engine was not sure about is not context.
        let mut carry = ContextCarry::default();
        carry.remember(Channel::Mic, true, "forse qualcosa cosi", Some("it"), Some(0.2), 28_000);
        assert!(carry.prompt_for(Channel::Mic, 28_000, Some("it")).is_none());

        // Neither is a decoder that has started looping.
        let mut carry = ContextCarry::default();
        carry.remember(Channel::Mic, true, "sì sì sì sì", Some("it"), Some(0.95), 28_000);
        assert!(carry.prompt_for(Channel::Mic, 28_000, Some("it")).is_none());

        // An unknown language on either side is not a mismatch; the gap rule
        // still applies.
        let mut carry = ContextCarry::default();
        carry.remember(Channel::Mic, true, "e il secondo punto invece", None, None, 28_000);
        assert!(carry.prompt_for(Channel::Mic, 28_100, Some("it")).is_some());
    }

    #[test]
    fn only_the_tail_of_the_previous_sentence_travels() {
        let long = (1..=60).map(|i| format!("parola{i}")).collect::<Vec<_>>().join(" ");
        let tail = tail_words(&long, CARRY_WORDS);
        assert_eq!(tail.split_whitespace().count(), CARRY_WORDS);
        assert!(tail.ends_with("parola60"));
        assert_eq!(tail_words("due parole", CARRY_WORDS), "due parole");
        assert_eq!(tail_words("   ", CARRY_WORDS), "");
    }

    #[test]
    fn a_loop_is_recognised_before_it_is_fed_back_in() {
        assert!(looks_repetitive("sì sì sì sì"));
        assert!(looks_repetitive("e poi e poi e poi basta"));
        assert!(!looks_repetitive("allora il punto principale è questo"));
        assert!(!looks_repetitive("sì sì"), "twice is emphasis, not a loop");
    }

    // -----------------------------------------------------------------------
    // The stop handoff
    // -----------------------------------------------------------------------

    #[test]
    fn the_handoff_window_starts_once_and_then_runs_out() {
        let mut handoff = Handoff::default();
        assert!(!handoff.started());
        assert!(!handoff.expired(), "nothing is expired before capture stops");

        assert!(handoff.begin());
        assert!(handoff.started());
        assert!(
            !handoff.begin(),
            "a second stop does not hand the pipeline more time"
        );
        assert!(!handoff.expired());
        assert!(
            handoff.deadline().is_some_and(|at| at
                <= tokio::time::Instant::now() + LIVE_HANDOFF),
            "the window is bounded, and inside the session layer's own drain wait"
        );

        // The window closing while the last decode was still running is the case
        // that matters: the live pass gives up on its own rather than being left
        // running behind the disk pass.
        let closed = Handoff {
            deadline: Some(tokio::time::Instant::now() - Duration::from_millis(1)),
        };
        assert!(closed.expired());
        assert!(
            LIVE_HANDOFF < Duration::from_secs(5),
            "the pipeline has to finish before the session layer stops waiting for it"
        );
    }

    #[test]
    fn a_recovery_in_the_middle_resets_the_run_without_re_notifying() {
        let mut run = FailureRun::default();
        for _ in 0..FAILURES_BEFORE_TELLING - 1 {
            run.note_failure();
        }
        run.note_success();
        for _ in 0..FAILURES_BEFORE_TELLING - 1 {
            run.note_failure();
        }
        assert!(
            !run.should_tell_them(),
            "a working utterance in between means live text is not stuck"
        );
    }
    // -----------------------------------------------------------------------
    // The split between the backlog pass and the live pass
    // -----------------------------------------------------------------------

    #[test]
    fn the_start_of_an_utterance_decides_which_pass_owns_it() {
        let backlog = Backlog::new();
        assert!(
            !below_the_floor(&backlog, 0),
            "with no split, the whole meeting is the live pass's"
        );

        backlog.claim(10_000);
        assert!(below_the_floor(&backlog, 0));
        assert!(
            below_the_floor(&backlog, 9_999),
            "an utterance straddling the floor started below it, so it is the disk \
             pass's; its tail is a hole the finalize job fills"
        );
        assert!(
            !below_the_floor(&backlog, 10_000),
            "the floor itself is where live takes over"
        );
        assert!(!below_the_floor(&backlog, 20_000));
    }

    #[tokio::test]
    async fn the_split_holds_the_disk_pass_until_the_live_pass_answers() {
        let backlog = Backlog::new();
        assert_eq!(backlog.floor(), None);
        // Nothing claimed: waiting for a claim waits.
        assert!(
            tokio::time::timeout(Duration::from_millis(20), backlog.claimed())
                .await
                .is_err()
        );

        backlog.claim(30_000);
        assert_eq!(backlog.floor(), Some(30_000));
        // A claim made before anybody was listening is still a claim — the bug
        // that made the whole split a no-op the first time it was written.
        tokio::time::timeout(Duration::from_millis(20), backlog.claimed())
            .await
            .expect("a claim already made is seen straight away");

        // And the disk pass waits, bounded, for the live pass to confirm it has
        // finished with everything below the floor.
        assert!(!backlog.wait_until_settled(Duration::from_millis(20)).await);
        backlog.done_below_the_floor();
        assert!(backlog.wait_until_settled(Duration::from_millis(20)).await);
    }

    // -----------------------------------------------------------------------
    // The backlog pass, through the real catch-up code
    // -----------------------------------------------------------------------

    /// Audio that is always there and always speech: the point of these tests is
    /// which stretches get read, not what is in them.
    struct FakeAudio;

    struct FakeStream {
        channel: Channel,
    }

    impl crate::asr::catchup::AudioSource for FakeAudio {
        type Stream = FakeStream;

        async fn read_window(
            &self,
            _chunks: &[crate::audio::ChunkRef],
            from_ms: i64,
            to_ms: i64,
        ) -> Result<Vec<f32>, AsrError> {
            let samples =
                ((to_ms - from_ms).max(0) * i64::from(crate::audio::TARGET_SAMPLE_RATE)) / 1_000;
            Ok(vec![0.2; samples as usize])
        }

        fn open_stream(
            &self,
            _detector: Option<&std::path::Path>,
            channel: Channel,
        ) -> FakeStream {
            FakeStream { channel }
        }
    }

    impl crate::asr::catchup::SpeechStream for FakeStream {
        async fn push(
            &mut self,
            samples: Vec<f32>,
            t_start_ms: i64,
        ) -> Result<Vec<Utterance>, AsrError> {
            let duration =
                (samples.len() as i64 * 1_000) / i64::from(crate::audio::TARGET_SAMPLE_RATE);
            Ok(vec![Utterance {
                channel: self.channel,
                t_start_ms,
                t_end_ms: t_start_ms + duration,
                samples,
                truncated: false,
            }])
        }

        async fn finish(&mut self) -> Result<Vec<Utterance>, AsrError> {
            Ok(Vec::new())
        }
    }

    /// Remembers every stretch it was asked to read.
    #[derive(Default)]
    struct Recorder {
        asked: Mutex<Vec<(i64, i64)>>,
    }

    impl crate::asr::catchup::Transcriber for Recorder {
        async fn transcribe(&self, job: TranscribeJob) -> Result<Transcription, AsrError> {
            let span = (job.t_start_ms, job.t_end_ms());
            self.asked.lock().expect("recorder").push(span);
            Ok(Transcription {
                channel: job.channel,
                t_start_ms: span.0,
                t_end_ms: span.1,
                text: format!("read {}..{}", span.0, span.1),
                ..Default::default()
            })
        }
    }

    /// The pass the live backlog uses is the post-meeting one, and it reads what
    /// has no text against it — never what the live pass already wrote down.
    #[tokio::test]
    async fn the_backlog_pass_reads_only_the_stretches_with_no_text_against_them() {
        let db = crate::db::connect_in_memory().await.unwrap();
        let meeting = repo::create_meeting(&db, "First ever launch", "/tmp", None)
            .await
            .unwrap();
        // Twenty seconds of committed audio on the microphone.
        let chunk = repo::insert_chunk(
            &db,
            &meeting.id,
            Channel::Mic,
            0,
            "/tmp/mic-000000.flac",
            0,
            20_000,
        )
        .await
        .unwrap();
        repo::commit_chunk(&db, &chunk, 20_000).await.unwrap();
        // The live pass got the middle of it: [4s, 12s].
        repo::insert_segment(
            &db,
            &SegmentDraft {
                meeting_id: meeting.id.clone(),
                t_start_ms: 4_000,
                t_end_ms: 12_000,
                channel: Channel::Mic,
                speaker_id: None,
                text: "live text".into(),
                language: Some("en".into()),
                avg_confidence: Some(0.9),
                revision: 1,
                is_final: true,
                model_name: Some("test".into()),
                model_revision: Some("1".into()),
            },
        )
        .await
        .unwrap();

        let recorder = Recorder::default();
        let report = crate::asr::catchup::run(
            &recorder,
            &db,
            &meeting.id,
            crate::asr::catchup::CatchUpOptions {
                to_ms: Some(20_000),
                ..Default::default()
            },
            &FakeAudio,
        )
        .await
        .unwrap();

        let asked = recorder.asked.lock().unwrap().clone();
        assert!(!asked.is_empty(), "the holes either side have to be read");
        for (from_ms, to_ms) in &asked {
            assert!(
                *to_ms <= 4_000 || *from_ms >= 12_000,
                "{from_ms}..{to_ms} overlaps text that already exists"
            );
        }
        assert_eq!(report.segments_written as usize, asked.len());
        // And the stretch that already had text still has exactly one line.
        let lines = repo::get_segments(
            &db,
            &crate::types::TranscriptQuery {
                meeting_id: meeting.id.clone(),
                from_ms: Some(4_000),
                to_ms: Some(11_999),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(
            lines.iter().filter(|s| s.text == "live text").count(),
            1,
            "the live line must not be duplicated"
        );
    }
}
