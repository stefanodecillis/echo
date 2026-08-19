//! Catching up from disk.
//!
//! IMPLEMENTED-BY: asr agent (M3), mantra 3 and review findings 17 and 18.
//!
//! Live transcription is best-effort. The queue is bounded, so a busy machine
//! drops live utterances rather than letting text run minutes behind the
//! meeting — and a crash or a force-quit drops everything that was in flight.
//! None of that is allowed to lose a word, because the per-channel FLAC on disk
//! is the source of truth. This module is what makes that promise true:
//!
//! * it **fills the holes**, per channel: it works out which stretches of
//!   committed audio have no text against them yet and transcribes exactly
//!   those, stopping at the last **committed** chunk so it never reads a file
//!   that is still being written. "Resume after the last thing written down"
//!   would not do, for two reasons found in review: an utterance the live queue
//!   dropped mid-meeting sits *before* the end of the transcript, so resuming
//!   past it loses those words for good even though the audio is right there;
//!   and the two channels run to different lengths, so one number for both can
//!   skip fifty minutes of the other side of the call.
//! * an explicit `from_ms` is a **floor**, never a per-channel override: a
//!   crash-recovery job can say "nothing before here matters" without hiding a
//!   hole on the other channel.
//! * it is driven by the `jobs` table, so it survives a restart and can be
//!   retried;
//! * it is cancellable, and it **yields to a live recording**: a meeting
//!   happening now always outranks tidying up one that already ended;
//! * it writes final segments only, with the same per-segment language,
//!   confidence and provenance the live pass writes.
//!
//! Everything it touches from outside — reading audio, finding speech,
//! transcribing — arrives through a trait, so the whole path is testable without
//! a gigabyte of weights or a single FLAC file.

use std::future::Future;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use crate::asr::engine::EngineWorker;
use crate::asr::{models, AsrError, TranscribeJob, Transcription};
use crate::audio::vad::Utterance;
use crate::audio::ChunkRef;
use crate::db::{repo, Db};
use crate::types::{AssetKind, Channel, TranscriptQuery};

/// How much audio is read from disk at a time. Matched to the longest utterance
/// the speech detector will emit, so a window never splits one in half.
pub const DEFAULT_WINDOW_MS: i64 = crate::audio::vad::MAX_UTTERANCE_MS;

/// Holes shorter than this are left alone.
///
/// Speech is written down with padding on both sides, so consecutive utterances
/// always leave a short silent gap between them; re-reading every one of those
/// would mean grinding through the whole meeting to find nothing. A hole this
/// short cannot hold a word once the padding either side is taken off.
pub const MIN_GAP_MS: i64 = 800;

/// How long to wait before looking again, while a recording has priority.
const YIELD_INTERVAL: Duration = Duration::from_millis(500);

/// Asked between windows: `true` means stop.
pub type CancelCheck = Arc<dyn Fn() -> bool + Send + Sync>;
/// Asked between windows: `true` means a recording is live, so wait.
pub type PauseCheck = Arc<dyn Fn() -> bool + Send + Sync>;
/// 0.0..=1.0 through the work.
pub type ProgressCheck = Arc<dyn Fn(f32) + Send + Sync>;

#[derive(Clone, Default)]
pub struct CatchUpOptions {
    /// Ignore anything before this offset on the meeting clock. A floor, not a
    /// starting point: holes after it are still filled per channel.
    pub from_ms: Option<i64>,
    /// Stop here instead of at the last committed chunk.
    pub to_ms: Option<i64>,
    /// Read size. 0 means [`DEFAULT_WINDOW_MS`].
    pub window_ms: i64,
    /// Emit live text while catching up. Off by default: nobody is watching.
    pub want_partials: bool,
    pub cancel: Option<CancelCheck>,
    /// Recording preempts everything (DESIGN §3 "Jobs table").
    pub pause_while: Option<PauseCheck>,
    pub on_progress: Option<ProgressCheck>,
}

impl std::fmt::Debug for CatchUpOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CatchUpOptions")
            .field("from_ms", &self.from_ms)
            .field("to_ms", &self.to_ms)
            .field("window_ms", &self.window_ms)
            .field("want_partials", &self.want_partials)
            .finish_non_exhaustive()
    }
}

impl CatchUpOptions {
    fn window(&self) -> i64 {
        if self.window_ms > 0 {
            self.window_ms
        } else {
            DEFAULT_WINDOW_MS
        }
    }

    fn cancelled(&self) -> bool {
        self.cancel.as_ref().is_some_and(|c| c())
    }

    fn should_wait(&self) -> bool {
        self.pause_while.as_ref().is_some_and(|p| p())
    }

    fn report(&self, fraction: f32) {
        if let Some(p) = &self.on_progress {
            p(fraction.clamp(0.0, 1.0));
        }
    }
}

/// What one catch-up pass did.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CatchUpReport {
    pub segments_written: u32,
    pub windows_read: u32,
    /// Where the pass began and ended on the meeting clock, per channel summed
    /// into one span for the log.
    pub from_ms: i64,
    pub to_ms: i64,
    /// True when it stopped early because it was told to.
    pub cancelled: bool,
    /// The language the meeting settled on, when the engine worked one out.
    pub language: Option<String>,
}

// ---------------------------------------------------------------------------
// Seams
// ---------------------------------------------------------------------------

/// Reading a meeting's audio back, and finding the speech in it.
///
/// [`DiskAudio`] is the real one. Everything else about catch-up is exercised
/// against a fake.
pub trait AudioSource: Send + Sync {
    /// 16 kHz mono for one window of one channel, read from the committed
    /// chunks. Each chunk carries where it belongs, so the window that comes
    /// back is the audio that happened then.
    fn read_window(
        &self,
        chunks: &[ChunkRef],
        from_ms: i64,
        to_ms: i64,
    ) -> impl Future<Output = Result<Vec<f32>, AsrError>> + Send;

    /// Speech inside that window. Without a detector this returns the whole
    /// window as one stretch, which is slower but never loses words.
    fn detect_speech(
        &self,
        detector: Option<&Path>,
        samples: &[f32],
        t_offset_ms: i64,
        channel: Channel,
    ) -> impl Future<Output = Result<Vec<Utterance>, AsrError>> + Send;
}

/// Whatever turns audio into text. The real one is [`EngineWorker`].
pub trait Transcriber: Send + Sync {
    fn transcribe(
        &self,
        job: TranscribeJob,
    ) -> impl Future<Output = Result<Transcription, AsrError>> + Send;

    /// The language the meeting settled on, if it has.
    fn settled_language(&self, _meeting_id: &str) -> Option<String> {
        None
    }
}

impl Transcriber for EngineWorker {
    async fn transcribe(&self, job: TranscribeJob) -> Result<Transcription, AsrError> {
        self.submit(job, None).await
    }

    fn settled_language(&self, meeting_id: &str) -> Option<String> {
        self.meeting_language(meeting_id)
    }
}

/// The real audio source: the per-channel FLAC chunks and the speech detector.
#[derive(Debug, Clone, Copy, Default)]
pub struct DiskAudio;

impl AudioSource for DiskAudio {
    async fn read_window(
        &self,
        chunks: &[ChunkRef],
        from_ms: i64,
        to_ms: i64,
    ) -> Result<Vec<f32>, AsrError> {
        crate::audio::read_window(chunks, from_ms, to_ms)
            .await
            .map_err(|e| AsrError::Io(e.to_string()))
    }

    async fn detect_speech(
        &self,
        detector: Option<&Path>,
        samples: &[f32],
        t_offset_ms: i64,
        channel: Channel,
    ) -> Result<Vec<Utterance>, AsrError> {
        let Some(detector) = detector else {
            return Ok(whole_window(samples, t_offset_ms, channel));
        };
        match crate::audio::vad::detect_offline(detector, samples, t_offset_ms, channel).await {
            Ok(found) => Ok(found),
            Err(e) => {
                // Losing the detector costs time, never words: transcribe the
                // whole window instead of skipping it.
                tracing::warn!(%e, "could not pick speech out of the recording; taking it whole");
                Ok(whole_window(samples, t_offset_ms, channel))
            }
        }
    }
}

/// One stretch covering everything we were handed.
fn whole_window(samples: &[f32], t_offset_ms: i64, channel: Channel) -> Vec<Utterance> {
    if samples.is_empty() {
        return Vec::new();
    }
    let duration = (samples.len() as i64 * 1_000) / i64::from(crate::audio::TARGET_SAMPLE_RATE);
    vec![Utterance {
        channel,
        t_start_ms: t_offset_ms,
        t_end_ms: t_offset_ms + duration,
        samples: samples.to_vec(),
        truncated: false,
    }]
}

// ---------------------------------------------------------------------------
// The pass
// ---------------------------------------------------------------------------

/// Fill in whatever the live pass missed, then leave the transcript complete up
/// to the last committed chunk.
pub async fn run<T, A>(
    transcriber: &T,
    db: &Db,
    meeting_id: &str,
    options: CatchUpOptions,
    audio: &A,
) -> Result<CatchUpReport, AsrError>
where
    T: Transcriber,
    A: AudioSource,
{
    if repo::get_meeting(db, meeting_id).await?.is_none() {
        return Err(AsrError::Db(crate::db::DbError::NotFound(format!(
            "meeting {meeting_id}"
        ))));
    }

    // Live partials are guesses that never became text. They are not evidence of
    // anything, and leaving them behind would make this pass skip real audio.
    repo::delete_partial_segments(db, meeting_id).await?;

    let detector = models::installed_path(db, AssetKind::SpeechDetector).await?;
    let mut report = CatchUpReport {
        from_ms: i64::MAX,
        ..Default::default()
    };

    // Plan first, so progress can be a fraction of something real. One entry per
    // hole per channel: the audio that is on disk, minus what already has text
    // against it.
    let floor = options.from_ms.unwrap_or(0).max(0);
    let mut plan: Vec<(Channel, Vec<ChunkRef>, i64, i64)> = Vec::new();
    for channel in [Channel::Mic, Channel::System] {
        let chunks: Vec<ChunkRef> = repo::list_chunks(db, meeting_id, Some(channel))
            .await?
            .iter()
            .filter(|c| c.committed)
            .map(ChunkRef::from_journal)
            .collect();
        if chunks.is_empty() {
            continue;
        }
        let committed_to = repo::last_committed_offset_ms(db, meeting_id, channel).await?;
        let limit = options.to_ms.unwrap_or(committed_to).min(committed_to);
        let on_disk = spans_of_audio(&chunks, floor, limit);
        let written = transcribed_spans(db, meeting_id, channel).await?;
        for (from_ms, to_ms) in subtract(&on_disk, &written) {
            if to_ms - from_ms < MIN_GAP_MS {
                continue;
            }
            plan.push((channel, chunks.clone(), from_ms, to_ms));
        }
    }

    let total_ms: i64 = plan.iter().map(|(_, _, s, e)| e - s).sum();
    if total_ms <= 0 {
        report.from_ms = 0;
        report.report_language(transcriber, meeting_id);
        options.report(1.0);
        return Ok(report);
    }
    let mut done_ms: i64 = 0;
    let window = options.window();

    for (channel, chunks, start, end) in plan {
        report.from_ms = report.from_ms.min(start);
        report.to_ms = report.to_ms.max(end);
        let mut cursor = start;

        while cursor < end {
            if options.cancelled() {
                report.cancelled = true;
                report.finish(transcriber, meeting_id, &options, done_ms, total_ms);
                return Ok(report);
            }
            // A meeting happening now matters more than one that already ended.
            while options.should_wait() {
                if options.cancelled() {
                    report.cancelled = true;
                    report.finish(transcriber, meeting_id, &options, done_ms, total_ms);
                    return Ok(report);
                }
                tokio::time::sleep(YIELD_INTERVAL).await;
            }

            let window_end = (cursor + window).min(end);
            let samples = audio.read_window(&chunks, cursor, window_end).await?;
            report.windows_read += 1;

            let utterances = audio
                .detect_speech(detector.as_deref(), &samples, cursor, channel)
                .await?;
            for utterance in utterances {
                // Only speech that actually falls in this hole. Anything that
                // reaches back into a stretch already written down would say the
                // same words twice.
                let overlaps = utterance.t_end_ms > start && utterance.t_start_ms < end;
                if !overlaps || utterance.samples.is_empty() {
                    continue;
                }
                let job = TranscribeJob {
                    meeting_id: meeting_id.to_string(),
                    utterance_id: format!("catchup-{}-{}", channel.as_str(), utterance.t_start_ms),
                    channel,
                    t_start_ms: utterance.t_start_ms,
                    samples: utterance.samples,
                    language_hint: None,
                    want_partials: options.want_partials,
                    // Catch-up work is the last chance this audio has, so it
                    // waits for the queue instead of being dropped.
                    droppable: false,
                };
                match transcriber.transcribe(job).await {
                    Ok(text) => {
                        if text.is_empty() {
                            continue;
                        }
                        repo::insert_segment(db, &text.to_draft(meeting_id)).await?;
                        report.segments_written += 1;
                    }
                    Err(AsrError::Cancelled) => {
                        report.cancelled = true;
                        report.finish(transcriber, meeting_id, &options, done_ms, total_ms);
                        return Ok(report);
                    }
                    // One bad window must not abandon the rest of the meeting.
                    Err(e) => tracing::warn!(%e, cursor, "skipped a stretch that would not decode"),
                }
            }

            done_ms += window_end - cursor;
            cursor = window_end;
            options.report(done_ms as f32 / total_ms as f32);
        }
    }

    report.finish(transcriber, meeting_id, &options, total_ms, total_ms);
    Ok(report)
}

impl CatchUpReport {
    fn report_language<T: Transcriber>(&mut self, transcriber: &T, meeting_id: &str) {
        self.language = transcriber.settled_language(meeting_id);
    }

    fn finish<T: Transcriber>(
        &mut self,
        transcriber: &T,
        meeting_id: &str,
        options: &CatchUpOptions,
        done_ms: i64,
        total_ms: i64,
    ) {
        if self.from_ms == i64::MAX {
            self.from_ms = 0;
        }
        self.report_language(transcriber, meeting_id);
        if total_ms > 0 {
            options.report(done_ms as f32 / total_ms as f32);
        } else {
            options.report(1.0);
        }
    }
}

/// Where this channel's audio is, as merged spans on the meeting clock, clipped
/// to `[floor, limit]`.
fn spans_of_audio(chunks: &[ChunkRef], floor: i64, limit: i64) -> Vec<(i64, i64)> {
    let mut spans: Vec<(i64, i64)> = chunks
        .iter()
        .map(|c| (c.t_start_ms.max(floor), c.t_end_ms.min(limit)))
        .filter(|(from, to)| to > from)
        .collect();
    merge(&mut spans);
    spans
}

/// The stretches of this channel that already have text against them.
///
/// Only **final** segments count. Live partials are guesses that never became
/// text — they are deleted at the start of the pass — so they never stand in the
/// way of the audio being read properly.
async fn transcribed_spans(
    db: &Db,
    meeting_id: &str,
    channel: Channel,
) -> Result<Vec<(i64, i64)>, AsrError> {
    let segments = repo::get_segments(
        db,
        &TranscriptQuery {
            meeting_id: meeting_id.to_string(),
            include_partial: Some(false),
            limit: Some(200_000),
            ..Default::default()
        },
    )
    .await?;
    let mut spans: Vec<(i64, i64)> = segments
        .iter()
        .filter(|s| s.channel == channel && s.is_final && s.t_end_ms > s.t_start_ms)
        .map(|s| (s.t_start_ms.max(0), s.t_end_ms))
        .collect();
    merge(&mut spans);
    Ok(spans)
}

/// Sort and join overlapping or touching spans, in place.
fn merge(spans: &mut Vec<(i64, i64)>) {
    spans.sort_by_key(|(from, _)| *from);
    let mut merged: Vec<(i64, i64)> = Vec::with_capacity(spans.len());
    for (from, to) in spans.drain(..) {
        match merged.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => merged.push((from, to)),
        }
    }
    *spans = merged;
}

/// `have` minus `covered`: the holes that still need text. Both must be merged
/// and sorted.
fn subtract(have: &[(i64, i64)], covered: &[(i64, i64)]) -> Vec<(i64, i64)> {
    let mut holes = Vec::new();
    for &(from, to) in have {
        let mut cursor = from;
        for &(c_from, c_to) in covered {
            if c_to <= cursor {
                continue;
            }
            if c_from >= to {
                break;
            }
            if c_from > cursor {
                holes.push((cursor, c_from.min(to)));
            }
            cursor = cursor.max(c_to);
            if cursor >= to {
                break;
            }
        }
        if cursor < to {
            holes.push((cursor, to));
        }
    }
    holes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::connect_in_memory;
    use crate::types::{Channel as Ch, SegmentDraft};
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::sync::Mutex;

    const SR: usize = crate::audio::TARGET_SAMPLE_RATE as usize;

    /// Reads back silence of exactly the length asked for, and reports every
    /// window it was asked to read.
    #[derive(Default)]
    struct FakeAudio {
        reads: Mutex<Vec<(Channel, i64, i64)>>,
        /// One utterance per window, or none at all.
        speech: bool,
    }

    /// The reads, as (channel, from, to).
    impl FakeAudio {
        fn reads(&self) -> Vec<(Channel, i64, i64)> {
            self.reads.lock().unwrap().clone()
        }
    }

    impl FakeAudio {
        fn with_speech() -> Self {
            Self {
                speech: true,
                ..Default::default()
            }
        }
    }

    impl AudioSource for FakeAudio {
        async fn read_window(
            &self,
            chunks: &[ChunkRef],
            from_ms: i64,
            to_ms: i64,
        ) -> Result<Vec<f32>, AsrError> {
            let channel = chunks.first().map(|c| c.channel).unwrap_or(Channel::Mic);
            self.reads.lock().unwrap().push((channel, from_ms, to_ms));
            let samples = ((to_ms - from_ms).max(0) as usize * SR) / 1_000;
            Ok(vec![0.0; samples])
        }

        async fn detect_speech(
            &self,
            _detector: Option<&Path>,
            samples: &[f32],
            t_offset_ms: i64,
            channel: Channel,
        ) -> Result<Vec<Utterance>, AsrError> {
            if !self.speech {
                return Ok(Vec::new());
            }
            Ok(whole_window(samples, t_offset_ms, channel))
        }
    }

    #[derive(Default)]
    struct FakeEngine {
        seen: Mutex<Vec<TranscribeJob>>,
        text: String,
        cancel_after: Option<u32>,
        calls: AtomicU32,
    }

    impl FakeEngine {
        fn saying(text: &str) -> Self {
            Self {
                text: text.to_string(),
                ..Default::default()
            }
        }
    }

    impl Transcriber for FakeEngine {
        async fn transcribe(&self, job: TranscribeJob) -> Result<Transcription, AsrError> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if self.cancel_after.is_some_and(|limit| n >= limit) {
                return Err(AsrError::Cancelled);
            }
            let t_end = job.t_end_ms();
            let t_start = job.t_start_ms;
            let channel = job.channel;
            self.seen.lock().unwrap().push(job);
            Ok(Transcription {
                channel,
                t_start_ms: t_start,
                t_end_ms: t_end,
                text: self.text.clone(),
                language: Some("en".into()),
                avg_confidence: Some(0.9),
                model_name: Some("test weights".into()),
                model_revision: Some("rev1".into()),
                ..Default::default()
            })
        }

        fn settled_language(&self, _meeting_id: &str) -> Option<String> {
            Some("en".into())
        }
    }

    /// A meeting with `chunks` committed 30 s chunks on the microphone channel.
    async fn meeting_with_audio(db: &Db, chunks: i64) -> String {
        let id = repo::create_meeting(db, "Catch up", "/audio", None)
            .await
            .unwrap()
            .id;
        commit_chunks(db, &id, Ch::Mic, chunks).await;
        id
    }

    /// `count` committed 30 s chunks on one channel, starting at t=0.
    async fn commit_chunks(db: &Db, meeting_id: &str, channel: Ch, count: i64) {
        for seq in 0..count {
            let chunk_id = repo::insert_chunk(
                db,
                meeting_id,
                channel,
                seq,
                &format!("/audio/{}-{seq:06}.flac", channel.as_str()),
                seq * 30_000,
                (seq + 1) * 30_000,
            )
            .await
            .unwrap();
            repo::commit_chunk(db, &chunk_id, (seq + 1) * 30_000)
                .await
                .unwrap();
        }
    }

    /// A stretch that already has text against it.
    async fn already_written(db: &Db, meeting_id: &str, channel: Ch, from_ms: i64, to_ms: i64) {
        repo::insert_segment(
            db,
            &SegmentDraft {
                meeting_id: meeting_id.to_string(),
                t_start_ms: from_ms,
                t_end_ms: to_ms,
                channel,
                text: "already written".into(),
                is_final: true,
                revision: 1,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn a_meeting_with_no_committed_audio_is_a_no_op() {
        let db = connect_in_memory().await.unwrap();
        let id = repo::create_meeting(&db, "Empty", "/audio", None)
            .await
            .unwrap()
            .id;
        let engine = FakeEngine::saying("hello");
        let report = run(
            &engine,
            &db,
            &id,
            CatchUpOptions::default(),
            &FakeAudio::with_speech(),
        )
        .await
        .unwrap();
        assert_eq!(report.segments_written, 0);
        assert_eq!(report.windows_read, 0);
        assert!(!report.cancelled);
        assert!(engine.seen.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_missing_meeting_is_an_error_not_a_silent_success() {
        let db = connect_in_memory().await.unwrap();
        let err = run(
            &FakeEngine::default(),
            &db,
            "nobody",
            CatchUpOptions::default(),
            &FakeAudio::with_speech(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, AsrError::Db(_)), "{err:?}");
    }

    #[tokio::test]
    async fn everything_committed_is_transcribed_and_written_with_its_provenance() {
        let db = connect_in_memory().await.unwrap();
        let id = meeting_with_audio(&db, 2).await;
        let audio = FakeAudio::with_speech();
        let engine = FakeEngine::saying("we agreed to ship on Friday");

        let report = run(
            &engine,
            &db,
            &id,
            CatchUpOptions {
                window_ms: 30_000,
                ..Default::default()
            },
            &audio,
        )
        .await
        .unwrap();

        assert_eq!(report.windows_read, 2, "two committed chunks, two windows");
        assert_eq!(report.segments_written, 2);
        assert_eq!(report.from_ms, 0);
        assert_eq!(report.to_ms, 60_000);
        assert_eq!(report.language.as_deref(), Some("en"));

        let written = repo::get_segments(
            &db,
            &TranscriptQuery {
                meeting_id: id.clone(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(written.len(), 2);
        assert!(written.iter().all(|s| s.is_final));
        assert!(written.iter().all(|s| s.channel == Ch::Mic));
        assert!(written
            .iter()
            .all(|s| s.model_name.as_deref() == Some("test weights")));
        assert!(written
            .iter()
            .all(|s| s.model_revision.as_deref() == Some("rev1")));
        assert!(written.iter().all(|s| s.language.as_deref() == Some("en")));
        assert_eq!(written[0].t_start_ms, 0);
        assert_eq!(written[1].t_start_ms, 30_000);

        // The jobs it queued could not be dropped: nothing else would pick them
        // up (mantra 3).
        let seen = engine.seen.lock().unwrap();
        assert!(seen.iter().all(|j| !j.droppable));
        assert!(seen.iter().all(|j| !j.want_partials));
    }

    #[tokio::test]
    async fn what_already_has_text_is_not_read_again() {
        let db = connect_in_memory().await.unwrap();
        let id = meeting_with_audio(&db, 3).await;
        // The live pass already got the first minute.
        already_written(&db, &id, Ch::Mic, 0, 60_000).await;

        let audio = FakeAudio::with_speech();
        let report = run(
            &FakeEngine::saying("the rest"),
            &db,
            &id,
            CatchUpOptions {
                window_ms: 30_000,
                ..Default::default()
            },
            &audio,
        )
        .await
        .unwrap();

        assert_eq!(report.from_ms, 60_000, "it fills in, it does not redo");
        assert_eq!(report.to_ms, 90_000);
        assert_eq!(report.windows_read, 1);
        assert_eq!(audio.reads(), vec![(Channel::Mic, 60_000, 90_000)]);
    }

    #[tokio::test]
    async fn an_utterance_the_live_pass_dropped_is_picked_up_from_disk() {
        let db = connect_in_memory().await.unwrap();
        let id = meeting_with_audio(&db, 2).await;
        // The live text has a hole in the middle: the queue was full for ten
        // seconds, and the words after it were written down as usual.
        already_written(&db, &id, Ch::Mic, 0, 20_000).await;
        already_written(&db, &id, Ch::Mic, 30_000, 60_000).await;

        let audio = FakeAudio::with_speech();
        let report = run(
            &FakeEngine::saying("the bit nobody heard"),
            &db,
            &id,
            CatchUpOptions {
                window_ms: 30_000,
                ..Default::default()
            },
            &audio,
        )
        .await
        .unwrap();

        assert_eq!(
            audio.reads(),
            vec![(Channel::Mic, 20_000, 30_000)],
            "the hole in the middle is what needs reading"
        );
        assert_eq!(report.segments_written, 1);
    }

    #[tokio::test]
    async fn each_channel_is_caught_up_on_its_own() {
        let db = connect_in_memory().await.unwrap();
        let id = meeting_with_audio(&db, 2).await;
        // The other side of the call has half a minute of audio and no text at
        // all, while the microphone side is fully written down. One offset for
        // both channels would skip the remote speaker entirely.
        commit_chunks(&db, &id, Ch::System, 1).await;
        already_written(&db, &id, Ch::Mic, 0, 60_000).await;

        let audio = FakeAudio::with_speech();
        let report = run(
            &FakeEngine::saying("what they said"),
            &db,
            &id,
            CatchUpOptions {
                window_ms: 30_000,
                ..Default::default()
            },
            &audio,
        )
        .await
        .unwrap();

        assert_eq!(audio.reads(), vec![(Channel::System, 0, 30_000)]);
        assert_eq!(report.segments_written, 1);
        let written = repo::get_segments(
            &db,
            &TranscriptQuery {
                meeting_id: id.clone(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert!(written.iter().any(|s| s.channel == Ch::System));
    }

    #[tokio::test]
    async fn an_explicit_offset_is_a_floor_not_a_starting_point() {
        let db = connect_in_memory().await.unwrap();
        let id = meeting_with_audio(&db, 2).await;
        let audio = FakeAudio::with_speech();
        run(
            &FakeEngine::saying("x"),
            &db,
            &id,
            CatchUpOptions {
                from_ms: Some(45_000),
                window_ms: 30_000,
                ..Default::default()
            },
            &audio,
        )
        .await
        .unwrap();
        assert_eq!(audio.reads(), vec![(Channel::Mic, 45_000, 60_000)]);
    }

    #[tokio::test]
    async fn a_gap_too_short_to_hold_a_word_is_left_alone() {
        let db = connect_in_memory().await.unwrap();
        let id = meeting_with_audio(&db, 1).await;
        // Ordinary silence between two utterances, padding included.
        already_written(&db, &id, Ch::Mic, 0, 10_000).await;
        already_written(&db, &id, Ch::Mic, 10_000 + MIN_GAP_MS / 2, 30_000).await;

        let audio = FakeAudio::with_speech();
        let report = run(
            &FakeEngine::saying("x"),
            &db,
            &id,
            CatchUpOptions {
                window_ms: 30_000,
                ..Default::default()
            },
            &audio,
        )
        .await
        .unwrap();
        assert!(audio.reads().is_empty(), "{:?}", audio.reads());
        assert_eq!(report.segments_written, 0);
    }

    #[tokio::test]
    async fn uncommitted_audio_is_never_read() {
        let db = connect_in_memory().await.unwrap();
        let id = meeting_with_audio(&db, 1).await;
        // A chunk that was written but never committed: a crash mid-write.
        repo::insert_chunk(
            &db,
            &id,
            Ch::Mic,
            1,
            "/audio/mic-000001.flac",
            30_000,
            60_000,
        )
        .await
        .unwrap();

        let audio = FakeAudio::with_speech();
        let report = run(
            &FakeEngine::saying("x"),
            &db,
            &id,
            CatchUpOptions {
                window_ms: 30_000,
                ..Default::default()
            },
            &audio,
        )
        .await
        .unwrap();
        assert_eq!(report.to_ms, 30_000, "it stops at committed audio");
        assert_eq!(
            audio.reads.lock().unwrap().as_slice(),
            &[(Channel::Mic, 0, 30_000)]
        );
    }

    #[tokio::test]
    async fn already_finished_work_writes_nothing_and_still_reports_cleanly() {
        let db = connect_in_memory().await.unwrap();
        let id = meeting_with_audio(&db, 1).await;
        repo::insert_segment(
            &db,
            &SegmentDraft {
                meeting_id: id.clone(),
                t_start_ms: 0,
                t_end_ms: 30_000,
                channel: Ch::Mic,
                text: "all of it".into(),
                is_final: true,
                revision: 1,
                ..Default::default()
            },
        )
        .await
        .unwrap();

        let seen: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        let report = run(
            &FakeEngine::saying("x"),
            &db,
            &id,
            CatchUpOptions {
                on_progress: Some(Arc::new(move |f| sink.lock().unwrap().push(f))),
                ..Default::default()
            },
            &FakeAudio::with_speech(),
        )
        .await
        .unwrap();
        assert_eq!(report.segments_written, 0);
        assert_eq!(report.windows_read, 0);
        assert_eq!(seen.lock().unwrap().last().copied(), Some(1.0));
    }

    #[tokio::test]
    async fn silence_produces_no_segments() {
        let db = connect_in_memory().await.unwrap();
        let id = meeting_with_audio(&db, 2).await;
        // The detector found nothing at all in this recording.
        let report = run(
            &FakeEngine::saying("should never be called"),
            &db,
            &id,
            CatchUpOptions {
                window_ms: 30_000,
                ..Default::default()
            },
            &FakeAudio::default(),
        )
        .await
        .unwrap();
        assert_eq!(report.windows_read, 2, "the audio was still read");
        assert_eq!(report.segments_written, 0);
    }

    #[tokio::test]
    async fn an_engine_that_hears_nothing_writes_nothing() {
        let db = connect_in_memory().await.unwrap();
        let id = meeting_with_audio(&db, 1).await;
        let report = run(
            &FakeEngine::saying("   ...   "),
            &db,
            &id,
            CatchUpOptions {
                window_ms: 30_000,
                ..Default::default()
            },
            &FakeAudio::with_speech(),
        )
        .await
        .unwrap();
        assert_eq!(
            report.segments_written, 0,
            "punctuation over silence is not a transcript"
        );
    }

    #[tokio::test]
    async fn cancelling_stops_between_windows_and_keeps_what_was_written() {
        let db = connect_in_memory().await.unwrap();
        let id = meeting_with_audio(&db, 4).await;
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        // Let one window through, then ask it to stop.
        let watcher: CancelCheck = Arc::new(move || flag.swap(true, Ordering::SeqCst));

        let report = run(
            &FakeEngine::saying("first bit"),
            &db,
            &id,
            CatchUpOptions {
                window_ms: 30_000,
                cancel: Some(watcher),
                ..Default::default()
            },
            &FakeAudio::with_speech(),
        )
        .await
        .unwrap();

        assert!(report.cancelled);
        assert_eq!(report.windows_read, 1, "it stopped at the next boundary");
        assert_eq!(report.segments_written, 1, "and kept what it had done");
    }

    #[tokio::test]
    async fn a_cancelled_meeting_reported_by_the_engine_stops_the_pass() {
        let db = connect_in_memory().await.unwrap();
        let id = meeting_with_audio(&db, 3).await;
        let engine = FakeEngine {
            text: "gone".into(),
            cancel_after: Some(1),
            ..Default::default()
        };
        let report = run(
            &engine,
            &db,
            &id,
            CatchUpOptions {
                window_ms: 30_000,
                ..Default::default()
            },
            &FakeAudio::with_speech(),
        )
        .await
        .unwrap();
        assert!(report.cancelled);
        assert_eq!(report.segments_written, 1);
    }

    #[tokio::test]
    async fn it_waits_while_a_recording_is_live() {
        let db = connect_in_memory().await.unwrap();
        let id = meeting_with_audio(&db, 1).await;
        let recording = Arc::new(AtomicBool::new(true));
        let asked = Arc::new(AtomicU32::new(0));
        let live = recording.clone();
        let counter = asked.clone();
        let pause: PauseCheck = Arc::new(move || {
            // Say "a meeting is happening" twice, then let it through.
            if counter.fetch_add(1, Ordering::SeqCst) >= 2 {
                live.store(false, Ordering::SeqCst);
            }
            live.load(Ordering::SeqCst)
        });

        let started = std::time::Instant::now();
        let report = run(
            &FakeEngine::saying("later"),
            &db,
            &id,
            CatchUpOptions {
                window_ms: 30_000,
                pause_while: Some(pause),
                ..Default::default()
            },
            &FakeAudio::with_speech(),
        )
        .await
        .unwrap();

        assert_eq!(report.segments_written, 1, "it got there in the end");
        assert!(
            asked.load(Ordering::SeqCst) >= 3,
            "it kept checking rather than pressing on"
        );
        assert!(
            started.elapsed() >= YIELD_INTERVAL,
            "and it actually waited"
        );
    }

    #[tokio::test]
    async fn progress_climbs_to_one() {
        let db = connect_in_memory().await.unwrap();
        let id = meeting_with_audio(&db, 4).await;
        let seen: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        run(
            &FakeEngine::saying("x"),
            &db,
            &id,
            CatchUpOptions {
                window_ms: 30_000,
                on_progress: Some(Arc::new(move |f| sink.lock().unwrap().push(f))),
                ..Default::default()
            },
            &FakeAudio::with_speech(),
        )
        .await
        .unwrap();

        let seen = seen.lock().unwrap();
        assert!(seen.len() >= 4);
        assert!(
            seen.windows(2).all(|w| w[1] >= w[0]),
            "never goes backwards"
        );
        assert_eq!(seen.last().copied(), Some(1.0));
        assert!(seen.iter().all(|f| (0.0..=1.0).contains(f)));
    }

    #[tokio::test]
    async fn live_guesses_are_cleared_before_the_real_text_is_written() {
        let db = connect_in_memory().await.unwrap();
        let id = meeting_with_audio(&db, 1).await;
        repo::insert_segment(
            &db,
            &SegmentDraft {
                meeting_id: id.clone(),
                t_start_ms: 0,
                t_end_ms: 5_000,
                channel: Ch::Mic,
                text: "half heard gue".into(),
                is_final: false,
                revision: 1,
                ..Default::default()
            },
        )
        .await
        .unwrap();

        run(
            &FakeEngine::saying("half heard guess, in full"),
            &db,
            &id,
            CatchUpOptions {
                window_ms: 30_000,
                ..Default::default()
            },
            &FakeAudio::with_speech(),
        )
        .await
        .unwrap();

        let written = repo::get_segments(
            &db,
            &TranscriptQuery {
                meeting_id: id.clone(),
                include_partial: Some(true),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(written.len(), 1);
        assert!(written[0].is_final);
        assert_eq!(written[0].text, "half heard guess, in full");
    }

    #[test]
    fn a_window_with_no_detector_is_taken_whole() {
        let one_second = vec![0.0f32; SR];
        let found = whole_window(&one_second, 12_000, Channel::System);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].t_start_ms, 12_000);
        assert_eq!(found[0].t_end_ms, 13_000);
        assert_eq!(found[0].channel, Channel::System);
        assert!(whole_window(&[], 0, Channel::Mic).is_empty());
    }

    #[test]
    fn overlapping_spans_are_merged_before_anything_is_subtracted() {
        let mut spans = vec![(30_000, 60_000), (0, 10_000), (9_000, 31_000)];
        merge(&mut spans);
        assert_eq!(spans, vec![(0, 60_000)]);
        let mut touching = vec![(0, 1_000), (1_000, 2_000)];
        merge(&mut touching);
        assert_eq!(touching, vec![(0, 2_000)]);
    }

    #[test]
    fn subtracting_what_is_written_leaves_exactly_the_holes() {
        // Nothing written: the whole thing is a hole.
        assert_eq!(subtract(&[(0, 100)], &[]), vec![(0, 100)]);
        // Written in the middle.
        assert_eq!(
            subtract(&[(0, 100)], &[(40, 60)]),
            vec![(0, 40), (60, 100)]
        );
        // Written at both ends.
        assert_eq!(subtract(&[(0, 100)], &[(0, 10), (90, 100)]), vec![(10, 90)]);
        // Fully written.
        assert!(subtract(&[(0, 100)], &[(0, 100)]).is_empty());
        // Text that reaches past the audio does not invent a hole.
        assert!(subtract(&[(0, 100)], &[(0, 500)]).is_empty());
        // Two separate stretches of audio.
        assert_eq!(
            subtract(&[(0, 50), (100, 150)], &[(10, 20)]),
            vec![(0, 10), (20, 50), (100, 150)]
        );
    }

    #[test]
    fn audio_spans_are_clipped_to_the_floor_and_the_committed_end() {
        let chunks = vec![
            ChunkRef::new("/a/mic-000000.wav", Channel::Mic, 0, 30_000),
            ChunkRef::new("/a/mic-000001.wav", Channel::Mic, 30_000, 60_000),
            // A gap, then more audio: an outage mid-meeting.
            ChunkRef::new("/a/mic-000002.wav", Channel::Mic, 90_000, 120_000),
        ];
        assert_eq!(
            spans_of_audio(&chunks, 0, 120_000),
            vec![(0, 60_000), (90_000, 120_000)],
            "a real gap in the audio is not read as though it were there"
        );
        assert_eq!(
            spans_of_audio(&chunks, 45_000, 100_000),
            vec![(45_000, 60_000), (90_000, 100_000)]
        );
    }

    #[test]
    fn the_default_window_never_splits_an_utterance() {
        assert_eq!(DEFAULT_WINDOW_MS, crate::audio::vad::MAX_UTTERANCE_MS);
        assert_eq!(CatchUpOptions::default().window(), DEFAULT_WINDOW_MS);
        assert_eq!(
            CatchUpOptions {
                window_ms: 5_000,
                ..Default::default()
            }
            .window(),
            5_000
        );
    }
}
