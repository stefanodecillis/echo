//! The live pipeline: what happens to audio while a meeting is being recorded.
//!
//! Two tasks, both fed by the bounded capture feed:
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
//! ```
//!
//! Nothing here is allowed to slow capture down (mantra 3). If the speech engine
//! cannot keep up, utterances are dropped on the floor: the audio is already
//! committed to disk and the catch-up job reads it back afterwards.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::asr::engine::PartialFn;
use crate::asr::{AsrError, TranscribeJob};
use crate::audio::vad::Utterance;
use crate::audio::writer::CommittedChunk;
use crate::db::repo;
use crate::events::{NoticeLevel, NoticePayload, TranscriptFinalPayload, TranscriptPartialPayload};
use crate::session::ports::{CaptureFeed, CaptureSignal, EventSink, UiEvent};
use crate::session::Inner;
use crate::types::{Channel, Id, Segment, SegmentDraft};

/// Utterances waiting for text. Small on purpose: falling behind should show up
/// as a catch-up job, not as gigabytes of queued audio.
pub const UTTERANCE_QUEUE: usize = 32;
/// Longest a finished segment waits before it is written.
pub const BATCH_INTERVAL: Duration = Duration::from_secs(2);
/// Or this many segments, whichever comes first.
pub const BATCH_SEGMENTS: usize = 8;
/// Live partial text: at most five a second.
const PARTIAL_INTERVAL: Duration = Duration::from_millis(200);
/// Loudness for the recording indicator: at most ten a second.
const LEVELS_INTERVAL: Duration = Duration::from_millis(100);

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

/// Start the two tasks for one recording. They end on their own when the
/// capture feed closes, which is what stopping a capture does.
pub(crate) fn spawn(inner: Arc<Inner>, meeting_id: Id, feed: CaptureFeed) -> Vec<JoinHandle<()>> {
    let (utterances_tx, utterances_rx) = mpsc::channel::<Utterance>(UTTERANCE_QUEUE);
    let feed_task = {
        let inner = inner.clone();
        let meeting_id = meeting_id.clone();
        tokio::spawn(async move { feed_loop(inner, meeting_id, feed, utterances_tx).await })
    };
    let speech_task =
        tokio::spawn(async move { speech_loop(inner, meeting_id, utterances_rx).await });
    vec![feed_task, speech_task]
}

// ---------------------------------------------------------------------------
// Feed
// ---------------------------------------------------------------------------

async fn feed_loop(
    inner: Arc<Inner>,
    meeting_id: Id,
    mut feed: CaptureFeed,
    utterances: mpsc::Sender<Utterance>,
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

async fn speech_loop(inner: Arc<Inner>, meeting_id: Id, mut utterances: mpsc::Receiver<Utterance>) {
    let mut batch: Vec<(String, SegmentDraft)> = Vec::new();
    let mut last_flush = Instant::now();
    let mut language: Option<String> = None;
    let mut speakers = SpeakerCache::default();
    let mut failures = FailureRun::default();

    loop {
        let until_flush = BATCH_INTERVAL.saturating_sub(last_flush.elapsed());
        tokio::select! {
            received = utterances.recv() => {
                match received {
                    Some(utterance) => {
                        if let Some(entry) = transcribe(
                            &inner,
                            &meeting_id,
                            utterance,
                            &mut language,
                            &mut speakers,
                            &mut failures,
                        )
                        .await
                        {
                            batch.push(entry);
                        }
                        inner.pending.fetch_sub(1, Ordering::SeqCst);
                        if batch.len() >= BATCH_SEGMENTS {
                            flush(&inner, &meeting_id, &mut batch).await;
                            last_flush = Instant::now();
                        }
                    }
                    None => {
                        flush(&inner, &meeting_id, &mut batch).await;
                        return;
                    }
                }
            }
            _ = tokio::time::sleep(until_flush) => {
                flush(&inner, &meeting_id, &mut batch).await;
                last_flush = Instant::now();
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
async fn transcribe(
    inner: &Arc<Inner>,
    meeting_id: &str,
    utterance: Utterance,
    language: &mut Option<String>,
    speakers: &mut SpeakerCache,
    failures: &mut FailureRun,
) -> Option<(String, SegmentDraft)> {
    let utterance_id = repo::new_id();
    let channel = utterance.channel;
    let t_start_ms = utterance.t_start_ms;
    let t_end_ms = utterance.t_end_ms;

    let job = TranscribeJob {
        meeting_id: meeting_id.to_string(),
        utterance_id: utterance_id.clone(),
        channel,
        t_start_ms,
        samples: utterance.samples,
        language_hint: language.clone(),
        want_partials: true,
        // Live work gives way when the queue is full; the catch-up pass picks
        // this stretch up from disk instead (mantra 3).
        droppable: true,
    };
    let on_partial = partial_events(
        inner,
        meeting_id,
        &utterance_id,
        channel,
        t_start_ms,
        t_end_ms,
    );

    let transcription = match inner.ports.asr.transcribe(job, Some(on_partial)).await {
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

    let text = transcription.text.trim();
    if text.is_empty() {
        close_partial(inner, meeting_id, &utterance_id, channel, t_start_ms, t_end_ms);
        return None;
    }

    if language.is_none() {
        if let Some(detected) = transcription.language.clone() {
            *language = Some(detected.clone());
            if let Err(error) = repo::set_meeting_language(&inner.db, meeting_id, &detected).await {
                tracing::debug!(%error, "could not store the meeting language");
            }
        }
    }

    let speaker_id = channel_speaker(inner, meeting_id, channel, speakers).await;

    Some((
        utterance_id,
        SegmentDraft {
            meeting_id: meeting_id.to_string(),
            t_start_ms: if transcription.t_start_ms > 0 {
                transcription.t_start_ms
            } else {
                t_start_ms.max(0)
            },
            t_end_ms: transcription.t_end_ms.max(t_end_ms),
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
    let (cluster_key, display_name, is_self) = match channel {
        Channel::Mic => ("mic", "You", true),
        Channel::System => ("system", "Speaker 1", false),
        Channel::Mixed => return None,
    };
    match repo::upsert_speaker(&inner.db, meeting_id, cluster_key, display_name, is_self).await {
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
}
