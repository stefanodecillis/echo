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
//! * it **packs**: whisper.cpp pads every input out to thirty seconds of mel and
//!   encodes all of it, so a four-second stretch costs what a twenty-eight
//!   second one does. Consecutive stretches of the same channel therefore go to
//!   the engine as one window, silences and all, and the lines that come back
//!   are split onto the transcript at their own times (see [`MAX_PACK_MS`]).
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

use crate::asr::engine::{DecodePlan, EngineWorker};
use crate::asr::glossary::Glossary;
use crate::asr::phantom::VoicedSpans;
use crate::asr::{models, AsrError, TranscribeJob, Transcription};
use crate::audio::vad::Utterance;
use crate::audio::ChunkRef;
use crate::db::{repo, Db};
use crate::types::{AssetKind, Channel, TranscriptQuery};

/// How much audio is read from disk at a time.
///
/// Only a read size: one detector is streamed through a whole stretch of missing
/// text, so where a read happens to end is not a boundary for anything (see
/// [`AudioSource::open_stream`]).
pub const DEFAULT_WINDOW_MS: i64 = crate::audio::vad::MAX_UTTERANCE_MS;

/// The longest packed window handed to one decode.
///
/// **This constant is the whole of the catch-up speed story.** whisper.cpp does
/// not decode the audio it is given: it pads every input out to 30 seconds of
/// mel and runs the encoder over all of it. `log_mel_spectrogram` fills
/// `stage_1_pad = WHISPER_SAMPLE_RATE * 30` samples of zeros past the end of the
/// audio (`whisper.cpp:3189`), and `whisper_encode_internal` then copies
/// `mel_offset .. mel_offset + 2*n_ctx` frames into a fixed-size input tensor it
/// has already zeroed — `n_ctx` being the model's own `n_audio_ctx`, 1500 frames
/// for large-v3 (`whisper.cpp:2044`, `whisper.cpp:2393`). A four-second
/// utterance and a twenty-eight-second one therefore cost the *same encode*.
///
/// The disk pass used to hand over one speech stretch at a time. On a real
/// meeting that averages about four seconds, so the four minutes
/// `catchup_probe` measures cost 59 encodes where 8 carry the same words —
/// seven eighths of the encoder time spent on zeros. Measured on that meeting
/// with the installed large-v3: 157 s of wall clock became 83 s, 0.66x real
/// time became 0.34x, and the transcript came back with more of the words in it
/// rather than fewer (2026-08-21).
///
/// 28 s rather than 30: whisper.cpp starts a second seek when there is more than
/// half a chunk left (`whisper.cpp:7401`, `whisper.cpp:7730`), and a window that
/// lands exactly on the boundary is one rounding error away from paying for two
/// encodes instead of one. Two seconds of headroom makes "one window, one
/// encode" true rather than usually true.
pub const MAX_PACK_MS: i64 = 28_000;

/// The packed-window budget for the pass that runs **while a meeting is being
/// recorded** — the one that reads back the stretch from before the engine was
/// ready.
///
/// Narrower than [`MAX_PACK_MS`] for one reason: the engine serves one decode at
/// a time. A live final that arrives mid-decode waits for it, and the wait is
/// however long that window takes. A 28-second window is around nine seconds of
/// decode on the machine this was measured on, and nine seconds is not a wait a
/// caption can absorb — the engine already treats a hypothesis four seconds
/// stale as not worth finishing ([`crate::asr::engine`]'s `STALE_SNAPSHOT_MS`).
/// Twelve seconds keeps the worst wait in that neighbourhood while still
/// carrying three or four stretches per encode instead of one.
pub const LIVE_PACK_MS: i64 = 12_000;

/// Holes shorter than this are left alone.
///
/// This used to be 800 ms, on the reasoning that a gap that short cannot hold a
/// word. It can. "Sì", "no", a name, a date, the clipped last word of a sentence
/// the live pass gave up on — Italian meetings are full of speech shorter than
/// that, and skipping it means the words are gone for good with the audio
/// sitting right there on disk (review finding 3).
///
/// 250 ms is the other end of the argument: it is roughly the shortest stretch
/// that can hold a word *and* be found by the speech detector at all — Silero's
/// own minimum-speech default is 250 ms, and below that a hole is padding, a
/// rounding difference between two segment rows, or a breath. The cost of the
/// lower bar is reading a few hundred short gaps of silence per meeting, where
/// "reading" means a handful of speech-detector windows that find nothing and
/// never reach the speech engine. That is cheap. Losing a word is not.
pub const MIN_GAP_MS: i64 = 250;

/// Below this mean confidence, a decode is treated as evidence that the language
/// prior was wrong rather than as the truth about the audio.
///
/// Deliberately the same bar the language policy uses to throw a detection away
/// ([`crate::asr::language::MIN_USABLE_CONFIDENCE`]): below it, the answer
/// carries no information worth keeping.
const CONFIDENCE_COLLAPSED: f32 = crate::asr::language::MIN_USABLE_CONFIDENCE;

/// How long to wait before looking again, while a recording has priority.
const YIELD_INTERVAL: Duration = Duration::from_millis(500);

/// The width a row is given when the engine wrote words but no span for them.
///
/// Not a guess at where the words were: a floor, so that real text always lands
/// somewhere the transcript can show it and the coverage machinery can count.
const MIN_ROW_MS: i64 = 200;

/// After the first stretch the engine could not read, log only every Nth.
///
/// A pass over a two-hour meeting with a broken engine is thousands of identical
/// lines. One at the start and one summary at the end say the same thing.
const SKIP_LOG_EVERY: u32 = 25;

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
    /// How wide a packed window may be. 0 means [`MAX_PACK_MS`].
    ///
    /// A knob for the same reason `window_ms` is one: `catchup_probe` runs the
    /// real pass over a real meeting twice, once with this at 1 — where every
    /// stretch is a window of its own, exactly as the pass worked before
    /// packing — so that the before-and-after numbers come from the same audio,
    /// the same weights and the same machine rather than from memory.
    pub pack_ms: i64,
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

    fn pack(&self) -> i64 {
        if self.pack_ms > 0 {
            self.pack_ms
        } else {
            MAX_PACK_MS
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
    /// Speech stretches the detector found and this pass took responsibility
    /// for. Before packing this was also the number of decodes.
    pub stretches_packed: u32,
    /// Packed windows handed to the engine: one decode each, and — because
    /// whisper.cpp pads every input to thirty seconds — one encode each. This
    /// against `stretches_packed` is the whole of what packing bought.
    pub windows_decoded: u32,
    /// Decodes beyond the first for a window: the second reading a stretch gets
    /// when the meeting's language cannot explain it (see
    /// [`transcribe_with_prior`]). Logged per pass so a slow catch-up can be
    /// read off the log rather than guessed at — whisper.cpp's own temperature
    /// ladder is capped at [`crate::asr::catalog::CATCHUP_MAX_FALLBACKS`], and
    /// this is the half of the retry story Echo controls.
    pub fallback_attempts: u32,
    /// Lines the engine wrote that the recording has no voice under: a stock
    /// courtesy phrase over near-silence, dropped rather than written down (see
    /// [`crate::asr::phantom`]). Counted so a pass that starts throwing away
    /// real speech is visible in the log rather than only in the transcript.
    pub phantoms_dropped: u32,
    /// Words put right against the vocabulary — "Nongula" back to "Langola"
    /// (see [`crate::asr::glossary`]). Counted for the same reason the phantoms
    /// are: this pass changes words a person will read, so how often it does
    /// belongs in the log next to how much it wrote.
    pub words_corrected: u32,
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
    /// Finds the speech in one continuous stretch, read a window at a time.
    type Stream: SpeechStream;

    /// 16 kHz mono for one window of one channel, read from the committed
    /// chunks. Each chunk carries where it belongs, so the window that comes
    /// back is the audio that happened then.
    fn read_window(
        &self,
        chunks: &[ChunkRef],
        from_ms: i64,
        to_ms: i64,
    ) -> impl Future<Output = Result<Vec<f32>, AsrError>> + Send;

    /// One detector for one continuous stretch of missing text.
    ///
    /// It is fed every window of that stretch in order and finished at the end
    /// of it. That is the whole point: a fresh detector per window (review
    /// finding 2) makes every read boundary a hard reset — no look-behind, no
    /// recurrent state, and a word straddling the boundary split in two by an
    /// accident of buffer sizes. Streamed, the detector never learns that the
    /// audio arrived in pieces.
    fn open_stream(&self, detector: Option<&Path>, channel: Channel) -> Self::Stream;
}

/// A speech detector with a memory, fed one window at a time.
pub trait SpeechStream: Send {
    fn push(
        &mut self,
        samples: Vec<f32>,
        t_start_ms: i64,
    ) -> impl Future<Output = Result<Vec<Utterance>, AsrError>> + Send;

    /// End of the stretch: whatever speech is still open is still speech.
    fn finish(&mut self) -> impl Future<Output = Result<Vec<Utterance>, AsrError>> + Send;
}

/// Whatever turns audio into text. The real one is [`EngineWorker`].
pub trait Transcriber: Send + Sync {
    /// `prompt` is the text to put in front of this audio — the words Echo was
    /// told about (see [`crate::asr::glossary`]). Nothing is carried across
    /// windows here: this pass reads a recording that already exists, window by
    /// window, and the only context it hands over is context somebody typed.
    fn transcribe(
        &self,
        job: TranscribeJob,
        prompt: Option<String>,
    ) -> impl Future<Output = Result<Transcription, AsrError>> + Send;

    /// The language the meeting settled on, if it has.
    fn settled_language(&self, _meeting_id: &str) -> Option<String> {
        None
    }
}

impl Transcriber for EngineWorker {
    async fn transcribe(
        &self,
        job: TranscribeJob,
        prompt: Option<String>,
    ) -> Result<Transcription, AsrError> {
        self.submit_with(job, DecodePlan::catch_up().with_prompt(prompt), None)
            .await
    }

    fn settled_language(&self, meeting_id: &str) -> Option<String> {
        self.meeting_language(meeting_id)
    }
}

/// The real audio source: the per-channel FLAC chunks and the speech detector.
#[derive(Debug, Clone, Copy, Default)]
pub struct DiskAudio;

impl AudioSource for DiskAudio {
    type Stream = DiskStream;

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

    fn open_stream(&self, detector: Option<&Path>, channel: Channel) -> DiskStream {
        let detector = detector.and_then(|path| {
            match crate::audio::vad::OfflineDetector::open(Some(path.to_path_buf()), channel) {
                Ok(open) => Some(open),
                Err(e) => {
                    tracing::warn!(%e, "could not start speech detection; reading the audio whole");
                    None
                }
            }
        });
        DiskStream { channel, detector }
    }
}

/// The real stream: one speech detector for one stretch of a meeting.
///
/// Without a detector — the asset was never downloaded, or it will not start —
/// every window is handed over whole. That is slower and gives the speech engine
/// silence to chew on, but it never loses words, which is the only promise this
/// module makes.
#[derive(Debug)]
pub struct DiskStream {
    channel: Channel,
    detector: Option<crate::audio::vad::OfflineDetector>,
}

impl SpeechStream for DiskStream {
    async fn push(
        &mut self,
        samples: Vec<f32>,
        t_start_ms: i64,
    ) -> Result<Vec<Utterance>, AsrError> {
        let Some(detector) = self.detector.as_mut() else {
            return Ok(whole_window(&samples, t_start_ms, self.channel));
        };
        // The copy is kept so that losing the detector mid-stretch can still
        // hand this window over whole. A memcpy is nothing next to a decode.
        match detector.push(samples.clone(), t_start_ms).await {
            Ok(found) => Ok(found),
            Err(e) => {
                // Losing the detector costs time, never words.
                tracing::warn!(%e, "could not pick speech out of the recording; taking it whole");
                self.detector = None;
                Ok(whole_window(&samples, t_start_ms, self.channel))
            }
        }
    }

    async fn finish(&mut self) -> Result<Vec<Utterance>, AsrError> {
        match self.detector.as_mut() {
            Some(detector) => Ok(detector.finish().await.unwrap_or_default()),
            None => Ok(Vec::new()),
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
        // Nothing measured this: there is no detector, which is why the window
        // is being handed over whole. Claiming it was all speech is the answer
        // that can never cost words — the courtesy filter downstream deletes on
        // evidence of silence, and there is none here.
        voiced_ms: duration,
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
    let Some(meeting) = repo::get_meeting(db, meeting_id).await? else {
        return Err(AsrError::Db(crate::db::DbError::NotFound(format!(
            "meeting {meeting_id}"
        ))));
    };

    // What language this meeting turned out to be in is the single most useful
    // thing we know about it, and catch-up used to throw it away (review
    // finding 4): every job asked the engine to guess again, from a few seconds
    // of audio at a time, which is exactly the situation where it guesses wrong.
    //
    // A prior, though, not a lock: see [`transcribe_with_prior`].
    let prior = meeting
        .language
        .clone()
        .or_else(|| transcriber.settled_language(meeting_id));

    // Live partials are guesses that never became text. They are not evidence of
    // anything, and leaving them behind would make this pass skip real audio.
    repo::delete_partial_segments(db, meeting_id).await?;

    // The words Echo should know, read once for the whole pass. This lane gets
    // them for the same reason it gets the meeting's language: it is the text
    // people keep, nothing is waiting on it, and its windows are long enough
    // that a couple of hundred tokens of names is a small part of what the
    // decoder is reading.
    let glossary = crate::settings::glossary(db).await;

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
    let budget = options.pack();
    let mut skipped: u32 = 0;

    for (channel, chunks, start, end) in plan {
        report.from_ms = report.from_ms.min(start);
        report.to_ms = report.to_ms.max(end);
        let mut cursor = start;
        // One detector for this whole stretch, however many reads it takes.
        let mut speech = audio.open_stream(detector.as_deref(), channel);
        let mut read_it_all = false;
        // The window being filled with consecutive stretches of this channel.
        let mut pack: Option<Pack> = None;
        // The audio those windows are cut out of, read once.
        let mut held = Held::new(start);
        let work = Work {
            transcriber,
            db,
            meeting_id,
            channel,
            prior: prior.as_deref(),
            glossary: &glossary,
            options: &options,
        };

        while !read_it_all {
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

            let utterances = if cursor < end {
                let window_end = (cursor + window).min(end);
                let samples = audio.read_window(&chunks, cursor, window_end).await?;
                report.windows_read += 1;
                held.push(cursor, &samples);
                let found = speech.push(samples, cursor).await?;
                done_ms += window_end - cursor;
                cursor = window_end;
                options.report(done_ms as f32 / total_ms as f32);
                found
            } else {
                // The stretch is over, and the detector may still be holding the
                // speech that ran up to the end of it.
                read_it_all = true;
                speech.finish().await?
            };

            // Every stretch either extends the window being packed or closes it
            // and opens the next. `read_it_all` is the last word: whatever is
            // still open when the hole runs out is decoded then.
            let mut ready: Vec<Pack> = Vec::new();
            for utterance in utterances {
                if let Some(next) = Pack::of(&utterance, (start, end)) {
                    match pack.as_mut() {
                        Some(open) if open.would_hold(&next, budget) => open.extend(&next),
                        _ => {
                            ready.extend(pack.replace(next));
                        }
                    }
                }
            }
            if read_it_all {
                ready.extend(pack.take());
            }

            for packed in ready {
                let samples = match held.span(packed.from_ms, packed.to_ms) {
                    Some(samples) => samples,
                    // A detector that held speech from further back than this
                    // pass keeps audio for. Rare, and reading those seconds
                    // again is cheaper than keeping every second just in case.
                    None => {
                        audio
                            .read_window(&chunks, packed.from_ms, packed.to_ms)
                            .await?
                    }
                };
                match decode_pack(&work, packed, samples, &mut report, &mut skipped).await? {
                    Outcome::Carried => {}
                    Outcome::Cancelled => {
                        report.cancelled = true;
                        report.finish(transcriber, meeting_id, &options, done_ms, total_ms);
                        return Ok(report);
                    }
                }
            }
            // Nothing before the open window — or before the oldest utterance
            // the detector could still be sitting on — is ever wanted again.
            let keep_from = pack
                .as_ref()
                .map(|open| open.from_ms)
                .unwrap_or(cursor)
                .min(cursor - crate::audio::vad::MAX_UTTERANCE_MS);
            held.forget_before(keep_from);
        }
    }

    if skipped > 0 {
        tracing::warn!(
            skipped,
            written = report.segments_written,
            "some stretches of this recording would not decode"
        );
    }
    // What this pass spent, in the two currencies that decide how long it took:
    // how many times the engine ran, and how many of those were retries. Before
    // packing, `windows_decoded` and `stretches_packed` were the same number.
    tracing::info!(
        target: "echo::asr",
        stretches = report.stretches_packed,
        windows = report.windows_decoded,
        fallbacks = report.fallback_attempts,
        ladder_cap = crate::asr::catalog::CATCHUP_MAX_FALLBACKS + 1,
        written = report.segments_written,
        phantoms_dropped = report.phantoms_dropped,
        words_corrected = report.words_corrected,
        audio_ms = total_ms,
        "catch-up pass finished"
    );
    report.finish(transcriber, meeting_id, &options, total_ms, total_ms);
    Ok(report)
}

/// Everything one packed window needs to become rows, gathered so the
/// transcribe step reads as one thing rather than nine arguments.
struct Work<'a, T: Transcriber> {
    transcriber: &'a T,
    db: &'a Db,
    meeting_id: &'a str,
    channel: Channel,
    /// The meeting's language, when it has one.
    prior: Option<&'a str>,
    /// The words Echo has been told about. Read once for the whole pass.
    glossary: &'a Glossary,
    options: &'a CatchUpOptions,
}

enum Outcome {
    /// Written, empty, or skipped — either way the pass carries on.
    Carried,
    Cancelled,
}

// ---------------------------------------------------------------------------
// Packing
// ---------------------------------------------------------------------------

/// Consecutive speech stretches of one channel, to be decoded as one window.
///
/// A pack is a **span on the meeting clock**, not a bag of audio: it runs from
/// the first stretch's start to the last one's end, and the audio handed to the
/// engine is read back over exactly that span. So the silences *between* the
/// stretches are in the window — the real recording, breaths, pauses and all,
/// rather than utterances glued end to end with the gaps cut out. That matters
/// in both directions: whisper.cpp has been listening to natural pauses since it
/// was trained, and its timestamps only mean anything if the window is a
/// continuous stretch of the meeting. Splicing would also break the mapping
/// back, because there would be no single origin to map through.
#[derive(Debug, Clone, PartialEq)]
struct Pack {
    from_ms: i64,
    to_ms: i64,
    /// How many stretches went in, for the report.
    stretches: u32,
    /// Where inside the window the detector actually heard voice.
    ///
    /// Carried because the window is mostly silence by construction, so "how
    /// voiced was this window" is the wrong question — each line the window
    /// comes back as has to be asked about its own place on the clock. See
    /// [`crate::asr::phantom::VoicedSpans`].
    voiced: VoicedSpans,
}

impl Pack {
    /// The pack one utterance makes on its own, or `None` when this utterance is
    /// not this hole's business.
    ///
    /// The rule is the one the per-stretch pass used: only speech that actually
    /// falls in the hole, because anything reaching back into a stretch that
    /// already has text would say the same words twice.
    fn of(utterance: &Utterance, hole: (i64, i64)) -> Option<Self> {
        let (start, end) = hole;
        if utterance.samples.is_empty()
            || utterance.t_end_ms <= start
            || utterance.t_start_ms >= end
        {
            return None;
        }
        let from_ms = utterance.t_start_ms.max(start);
        let to_ms = utterance.t_end_ms.min(end);
        if to_ms <= from_ms {
            return None;
        }
        let mut voiced = VoicedSpans::default();
        voiced.add(from_ms, to_ms, utterance.voiced_ratio());
        Some(Self {
            from_ms,
            to_ms,
            stretches: 1,
            voiced,
        })
    }

    /// Would this window still be one encode with `next` in it?
    ///
    /// Only the budget is asked about, deliberately. A long silence between two
    /// stretches is not a reason to close the window: carrying it costs nothing
    /// — the encoder is paid per call, not per second — and the alternative is
    /// two encodes for the same words. What closes a window is running out of
    /// [`MAX_PACK_MS`], and a gap wide enough to matter does that by itself.
    fn would_hold(&self, next: &Pack, budget_ms: i64) -> bool {
        next.to_ms - self.from_ms <= budget_ms
    }

    fn extend(&mut self, next: &Pack) {
        self.to_ms = self.to_ms.max(next.to_ms);
        self.stretches += next.stretches;
        self.voiced.absorb(&next.voiced);
    }

    fn len_ms(&self) -> i64 {
        self.to_ms - self.from_ms
    }
}

/// The audio of the hole being worked through, kept for as long as a packed
/// window might still want it.
///
/// A packed window is a *span*, and the audio for it has to be the recording
/// over that span — gaps included. It has already been read once, to be fed to
/// the speech detector, so this holds on to it rather than reading the same
/// seconds off disk twice: FLAC does not decode itself for free, and a second
/// pass over every chunk would give back part of what packing just won.
///
/// Bounded by [`Held::forget_before`]: everything before the open window's start
/// and before the oldest utterance the detector could still be holding is
/// dropped after every read, so this is tens of seconds of audio, never a
/// meeting's worth.
#[derive(Debug)]
struct Held {
    /// Where `samples` begins on the meeting clock.
    from_ms: i64,
    samples: Vec<f32>,
}

impl Held {
    fn new(from_ms: i64) -> Self {
        Self {
            from_ms,
            samples: Vec::new(),
        }
    }

    /// Where the audio held here runs out, on the meeting clock.
    fn end_ms(&self) -> i64 {
        self.from_ms + ms_of_samples(self.samples.len())
    }

    /// Add the next read, which starts at `t_start_ms`.
    ///
    /// A read that does not begin exactly where the last one ended starts the
    /// buffer again from there. That cannot happen the way the hole is walked —
    /// the cursor is contiguous and a read always comes back the length it was
    /// asked for — and being wrong about it would be silent and awful: every
    /// window cut out afterwards would be the wrong audio under the right
    /// timestamps. So it is checked rather than assumed, and the cost of being
    /// wrong is a few windows read off disk again.
    fn push(&mut self, t_start_ms: i64, samples: &[f32]) {
        if t_start_ms != self.end_ms() {
            self.samples.clear();
            self.from_ms = t_start_ms;
        }
        self.samples.extend_from_slice(samples);
    }

    /// The audio over `[from_ms, to_ms)`, or `None` when it is not all here — a
    /// detector that held speech for longer than [`Held::forget_before`] keeps.
    /// The caller reads those few spans off disk instead, so this being an
    /// optimisation rather than the source of truth is the point.
    fn span(&self, from_ms: i64, to_ms: i64) -> Option<Vec<f32>> {
        if from_ms < self.from_ms || to_ms < from_ms {
            return None;
        }
        let first = samples_in(from_ms - self.from_ms);
        let last = samples_in(to_ms - self.from_ms);
        if last > self.samples.len() {
            return None;
        }
        Some(self.samples[first..last].to_vec())
    }

    /// Drop everything before `t_ms`.
    fn forget_before(&mut self, t_ms: i64) {
        if t_ms <= self.from_ms {
            return;
        }
        let drop = samples_in(t_ms - self.from_ms).min(self.samples.len());
        self.samples.drain(..drop);
        self.from_ms += ms_of_samples(drop);
    }
}

/// 16 kHz mono samples in `ms` milliseconds. Exact: one millisecond is sixteen
/// samples.
fn samples_in(ms: i64) -> usize {
    (ms.max(0) * i64::from(crate::audio::TARGET_SAMPLE_RATE) / 1_000) as usize
}

fn ms_of_samples(samples: usize) -> i64 {
    samples as i64 * 1_000 / i64::from(crate::audio::TARGET_SAMPLE_RATE)
}

/// Decode one packed window once, and write down the lines it came back as.
async fn decode_pack<T: Transcriber>(
    work: &Work<'_, T>,
    pack: Pack,
    samples: Vec<f32>,
    report: &mut CatchUpReport,
    skipped: &mut u32,
) -> Result<Outcome, AsrError> {
    if samples.is_empty() {
        return Ok(Outcome::Carried);
    }
    let job = TranscribeJob {
        meeting_id: work.meeting_id.to_string(),
        utterance_id: format!("catchup-{}-{}", work.channel.as_str(), pack.from_ms),
        channel: work.channel,
        t_start_ms: pack.from_ms,
        samples,
        language_hint: work.prior.map(str::to_string),
        want_partials: work.options.want_partials,
        // Catch-up work is the last chance this audio has, so it waits for the
        // queue instead of being dropped.
        droppable: false,
    };
    report.stretches_packed += pack.stretches;
    report.windows_decoded += 1;
    let prompt = work.glossary.prompt();
    match transcribe_with_prior(work.transcriber, job, work.prior, prompt, report).await {
        Ok(text) => {
            if text.is_empty() {
                return Ok(Outcome::Carried);
            }
            // One row per line the engine wrote, each at its own place on the
            // meeting clock. A window that came back as one line is one row,
            // exactly as a single stretch always was.
            for line in split_onto_the_transcript(&text, &pack) {
                // A stock courtesy phrase sitting where the detector heard no
                // voice at all — see the 2026-08-24 phantoms in
                // [`crate::asr::phantom`].
                //
                // The question is asked about this line's own seconds. When
                // whisper gave the window one line and no timestamps, those
                // seconds *are* the whole packed window, and the answer is then
                // all the voice the window holds across every stretch packed
                // into it — which is why this counts milliseconds rather than a
                // share. A packed window is mostly pause by construction, and a
                // share of one would call every multi-stretch window silence and
                // delete the real answer at the end of it.
                //
                // Dropping leaves those seconds with no text against them, so a
                // later pass over the same meeting reads them again and drops
                // the same line again. That is the same thing that already
                // happens to a window that decodes to nothing, and it is the
                // right way round: the alternative would be writing a line
                // nobody said to keep a coverage sum tidy.
                let voiced_ms = pack.voiced.voiced_ms(line.t_start_ms, line.t_end_ms);
                if crate::asr::phantom::is_phantom(&line.text, voiced_ms) {
                    report.phantoms_dropped += 1;
                    tracing::debug!(
                        target: "echo::asr",
                        t_start_ms = line.t_start_ms,
                        voiced_ms,
                        text = %line.text,
                        "dropped a courtesy line the recording has no voice under"
                    );
                    continue;
                }
                // Near misses against the words Echo was told about, put right
                // on the way to the database — and recorded on the row, because
                // this text is what a person reads afterwards as the record of
                // the meeting (see [`crate::asr::glossary`]).
                let mut draft = line.to_draft(work.meeting_id);
                if let Some(fixed) = work.glossary.correct(&draft.text) {
                    report.words_corrected += fixed.changes.len() as u32;
                    draft.text = fixed.text;
                    draft.corrections = fixed.changes;
                }
                repo::insert_segment(work.db, &draft).await?;
                report.segments_written += 1;
            }
        }
        Err(AsrError::Cancelled) => return Ok(Outcome::Cancelled),
        // One bad window must not abandon the rest of the meeting — nor fill
        // the log with one line per window while it does.
        Err(e) => {
            *skipped += 1;
            if *skipped == 1 || skipped.is_multiple_of(SKIP_LOG_EVERY) {
                tracing::warn!(
                    %e,
                    t_start_ms = pack.from_ms,
                    window_ms = pack.len_ms(),
                    count = *skipped,
                    "skipped a window that would not decode"
                );
            }
        }
    }
    Ok(Outcome::Carried)
}

/// The rows one decoded window becomes: whisper's own lines, clipped to the
/// window they were read out of, with anything empty or backwards dropped.
///
/// The engine has already mapped each line through the window's origin (see
/// [`crate::asr::TranscribedLine`]), so this is the sanity check rather than the
/// arithmetic: a line has to sit inside the window and it has to have width, or
/// the coverage machinery would either claim text over audio nobody read or
/// leave a zero-width row that never counts as covered and gets re-read for ever.
fn split_onto_the_transcript(text: &Transcription, pack: &Pack) -> Vec<Transcription> {
    let mut rows = Vec::new();
    for mut line in text.per_line() {
        if line.is_empty() {
            continue;
        }
        line.t_start_ms = line.t_start_ms.clamp(pack.from_ms, pack.to_ms);
        line.t_end_ms = line.t_end_ms.clamp(line.t_start_ms, pack.to_ms);
        if line.t_end_ms <= line.t_start_ms {
            // Whisper can close a window with a line it gives no width — the
            // words are real, the span is not. A row with no width is a row the
            // coverage machinery never counts as covered, so the audio under it
            // would be read again for ever. It gets the rest of the window, or
            // the last [`MIN_ROW_MS`] of it when there is no rest.
            line.t_end_ms = pack.to_ms;
            if line.t_end_ms <= line.t_start_ms {
                line.t_start_ms = (line.t_end_ms - MIN_ROW_MS).max(pack.from_ms);
            }
        }
        if line.t_end_ms <= line.t_start_ms {
            continue;
        }
        rows.push(line);
    }
    rows
}

/// Decode with the meeting's language pinned, and try again without it when the
/// answer falls apart.
///
/// A meeting language is a strong prior and a bad law. Most Italian meetings are
/// Italian throughout, and telling the engine so is worth more than any decoder
/// setting. But people quote an English email, a colleague joins and switches
/// language, someone reads out a product name — and a pinned language turns
/// those stretches into confident nonsense. Confidence collapsing is the signal
/// that the prior does not fit *this* stretch, so the stretch is read again with
/// nothing pinned and the better of the two answers is kept.
async fn transcribe_with_prior<T: Transcriber>(
    transcriber: &T,
    job: TranscribeJob,
    prior: Option<&str>,
    prompt: Option<String>,
    report: &mut CatchUpReport,
) -> Result<Transcription, AsrError> {
    if prior.is_none() {
        return transcriber.transcribe(job, prompt).await;
    }
    // Held back only so the retry can happen; a copy of the audio costs nothing
    // next to a decode of it.
    let retry = TranscribeJob {
        language_hint: None,
        ..job.clone()
    };
    let first = transcriber.transcribe(job, prompt.clone()).await?;
    if !collapsed(&first) {
        return Ok(first);
    }
    let t_start_ms = retry.t_start_ms;
    report.fallback_attempts += 1;
    match transcriber.transcribe(retry, prompt).await {
        Ok(second) if improves_on(&second, &first) => {
            tracing::debug!(
                target: "echo::asr",
                t_start_ms,
                language = second.language.as_deref().unwrap_or("unknown"),
                "this stretch was not in the meeting's language; kept the second reading"
            );
            Ok(second)
        }
        Ok(_) => Ok(first),
        Err(AsrError::Cancelled) => Err(AsrError::Cancelled),
        // The prior's answer was poor, but poor beats nothing.
        Err(e) => {
            tracing::debug!(%e, t_start_ms, "could not read this stretch a second time");
            Ok(first)
        }
    }
}

/// Nothing worth keeping, or a confidence too low to mean anything.
fn collapsed(text: &Transcription) -> bool {
    text.is_empty()
        || text
            .avg_confidence
            .is_some_and(|c| c < CONFIDENCE_COLLAPSED)
}

fn improves_on(second: &Transcription, first: &Transcription) -> bool {
    if second.is_empty() {
        return false;
    }
    if first.is_empty() {
        return true;
    }
    second.avg_confidence.unwrap_or(0.0) > first.avg_confidence.unwrap_or(0.0)
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

    /// What the fake detector claims to hear.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    enum Speech {
        /// Nothing at all.
        #[default]
        None,
        /// One utterance per window handed to it.
        PerWindow,
        /// One utterance covering everything it was fed, handed over when the
        /// stretch ends: a sentence that ran across every read boundary.
        WholeStretch,
        /// Many short utterances with gaps between them: what a real speech
        /// detector makes of a real meeting, and the shape the field regression
        /// of 2026-08-20 came in (281 stretches over 21 minutes).
        ManyShort,
        /// A few short utterances a long way apart, each holding this much
        /// voice: the end of a call, where three exchanges are separated by the
        /// pauses that pack them into one window (2026-08-24).
        FarApart { voiced_ms: i64 },
    }

    /// How long each utterance is when the detector hears [`Speech::ManyShort`],
    /// and the silence it leaves between them. Both taken from the field
    /// meeting, where the median stretch was a little over a second.
    const SHORT_UTTERANCE_MS: i64 = 1_200;
    const SHORT_GAP_MS: i64 = 300;

    /// Clock between the starts of two [`Speech::FarApart`] stretches: long
    /// enough that three of them still pack into one window, and that a share of
    /// that window would read as silence.
    const FAR_APART_STEP_MS: i64 = 12_000;

    /// Reads back audio of exactly the length asked for, and reports every
    /// window it was asked to read.
    ///
    /// Not silence: a stretch of zeros is indistinguishable from a chunk reader
    /// that has stopped finding the files, and one of these tests is about
    /// telling those apart.
    #[derive(Default)]
    struct FakeAudio {
        reads: Mutex<Vec<(Channel, i64, i64)>>,
        /// Every window handed to a detector, as (channel, from, to).
        fed: Arc<Mutex<Vec<(Channel, i64, i64)>>>,
        /// How many detectors were opened: one per continuous stretch, never one
        /// per window.
        detectors: Arc<AtomicU32>,
        speech: Speech,
    }

    impl FakeAudio {
        /// The reads, as (channel, from, to).
        fn reads(&self) -> Vec<(Channel, i64, i64)> {
            self.reads.lock().unwrap().clone()
        }

        fn fed(&self) -> Vec<(Channel, i64, i64)> {
            self.fed.lock().unwrap().clone()
        }

        fn detectors_opened(&self) -> u32 {
            self.detectors.load(Ordering::SeqCst)
        }
    }

    struct FakeStream {
        channel: Channel,
        speech: Speech,
        fed: Arc<Mutex<Vec<(Channel, i64, i64)>>>,
        /// What a `WholeStretch` detector is holding on to.
        held: Vec<f32>,
        held_from_ms: Option<i64>,
    }

    fn ms_of(samples: usize) -> i64 {
        (samples as i64 * 1_000) / i64::from(crate::audio::TARGET_SAMPLE_RATE)
    }

    impl SpeechStream for FakeStream {
        async fn push(
            &mut self,
            samples: Vec<f32>,
            t_start_ms: i64,
        ) -> Result<Vec<Utterance>, AsrError> {
            self.fed.lock().unwrap().push((
                self.channel,
                t_start_ms,
                t_start_ms + ms_of(samples.len()),
            ));
            match self.speech {
                Speech::None => Ok(Vec::new()),
                Speech::PerWindow => Ok(whole_window(&samples, t_start_ms, self.channel)),
                Speech::WholeStretch => {
                    self.held_from_ms.get_or_insert(t_start_ms);
                    self.held.extend_from_slice(&samples);
                    Ok(Vec::new())
                }
                Speech::FarApart { voiced_ms } => {
                    let per = (SHORT_UTTERANCE_MS as usize * SR) / 1_000;
                    let step = (FAR_APART_STEP_MS as usize * SR) / 1_000;
                    let mut found = Vec::new();
                    let mut at = 0usize;
                    while at + per <= samples.len() {
                        found.extend(whole_window(
                            &samples[at..at + per],
                            t_start_ms + ms_of(at),
                            self.channel,
                        ));
                        at += step;
                    }
                    for utterance in &mut found {
                        utterance.voiced_ms = voiced_ms;
                    }
                    Ok(found)
                }
                Speech::ManyShort => {
                    let per = (SHORT_UTTERANCE_MS as usize * SR) / 1_000;
                    let step = ((SHORT_UTTERANCE_MS + SHORT_GAP_MS) as usize * SR) / 1_000;
                    let mut found = Vec::new();
                    let mut at = 0usize;
                    while at + per <= samples.len() {
                        found.extend(whole_window(
                            &samples[at..at + per],
                            t_start_ms + ms_of(at),
                            self.channel,
                        ));
                        at += step;
                    }
                    Ok(found)
                }
            }
        }

        async fn finish(&mut self) -> Result<Vec<Utterance>, AsrError> {
            let from = match self.held_from_ms.take() {
                Some(from) => from,
                None => return Ok(Vec::new()),
            };
            Ok(whole_window(
                &std::mem::take(&mut self.held),
                from,
                self.channel,
            ))
        }
    }

    /// An utterance the live pass could not write down leaves a hole exactly its
    /// own width. If that width were under [`MIN_GAP_MS`] this pass would step
    /// straight over it and the words would be gone for good, audio on disk or
    /// not — so the shortest thing the segmenter emits has to be wider than the
    /// shortest hole this pass will look at.
    #[test]
    fn every_utterance_the_live_pass_loses_leaves_a_hole_this_pass_will_look_at() {
        const { assert!(crate::audio::vad::MIN_UTTERANCE_MS > MIN_GAP_MS) };
    }

    impl FakeAudio {
        fn with_speech() -> Self {
            Self {
                speech: Speech::PerWindow,
                ..Default::default()
            }
        }

        /// One sentence that runs across every read boundary of a stretch.
        fn with_one_long_sentence() -> Self {
            Self {
                speech: Speech::WholeStretch,
                ..Default::default()
            }
        }

        /// A meeting's worth of ordinary short utterances.
        fn with_a_meeting_full_of_speech() -> Self {
            Self {
                speech: Speech::ManyShort,
                ..Default::default()
            }
        }

        /// Three short exchanges twelve seconds apart, each holding `voiced_ms`
        /// of real voice: the end of a call, packed into one window.
        fn with_a_few_exchanges_far_apart(voiced_ms: i64) -> Self {
            Self {
                speech: Speech::FarApart { voiced_ms },
                ..Default::default()
            }
        }
    }

    impl AudioSource for FakeAudio {
        type Stream = FakeStream;

        async fn read_window(
            &self,
            chunks: &[ChunkRef],
            from_ms: i64,
            to_ms: i64,
        ) -> Result<Vec<f32>, AsrError> {
            let channel = chunks.first().map(|c| c.channel).unwrap_or(Channel::Mic);
            self.reads.lock().unwrap().push((channel, from_ms, to_ms));
            let samples = ((to_ms - from_ms).max(0) as usize * SR) / 1_000;
            // A quiet sawtooth: audible, never clipping, and — unlike a buffer
            // of zeros — distinguishable from a chunk reader that has stopped
            // finding the files.
            Ok((0..samples)
                .map(|i| ((i % 160) as f32 / 160.0) * 0.25 - 0.125)
                .collect())
        }

        fn open_stream(&self, _detector: Option<&Path>, channel: Channel) -> FakeStream {
            self.detectors.fetch_add(1, Ordering::SeqCst);
            FakeStream {
                channel,
                speech: self.speech,
                fed: Arc::clone(&self.fed),
                held: Vec::new(),
                held_from_ms: None,
            }
        }
    }

    #[derive(Default)]
    struct FakeEngine {
        seen: Mutex<Vec<TranscribeJob>>,
        text: String,
        cancel_after: Option<u32>,
        calls: AtomicU32,
        /// Answers badly whenever a language is pinned: the stretch of a
        /// bilingual meeting the meeting's language cannot explain.
        confused_by_a_pinned_language: bool,
        /// What [`Transcriber::settled_language`] says, if anything.
        settled: Option<String>,
        /// Sentences to answer with, as offsets into the window it is given:
        /// what a real engine hands back for a packed window.
        lines: Vec<(i64, i64, String)>,
        /// Every prompt it was handed, in order: what the vocabulary put in
        /// front of the audio.
        prompts: Mutex<Vec<Option<String>>>,
    }

    impl FakeEngine {
        fn saying(text: &str) -> Self {
            Self {
                text: text.to_string(),
                settled: Some("en".into()),
                ..Default::default()
            }
        }

        /// Answers every window with these sentences, at these offsets into it.
        fn in_lines(lines: &[(i64, i64, &str)]) -> Self {
            Self {
                text: lines
                    .iter()
                    .map(|(_, _, text)| *text)
                    .collect::<Vec<_>>()
                    .join(" "),
                settled: Some("en".into()),
                lines: lines
                    .iter()
                    .map(|(from, to, text)| (*from, *to, text.to_string()))
                    .collect(),
                ..Default::default()
            }
        }

        fn hints(&self) -> Vec<Option<String>> {
            self.seen
                .lock()
                .unwrap()
                .iter()
                .map(|j| j.language_hint.clone())
                .collect()
        }

        /// Every window it was handed, as (start on the meeting clock, how much
        /// audio came with it).
        fn spans(&self) -> Vec<(i64, i64)> {
            self.seen
                .lock()
                .unwrap()
                .iter()
                .map(|j| (j.t_start_ms, j.duration_ms()))
                .collect()
        }
    }

    impl Transcriber for FakeEngine {
        async fn transcribe(
            &self,
            job: TranscribeJob,
            prompt: Option<String>,
        ) -> Result<Transcription, AsrError> {
            self.prompts.lock().unwrap().push(prompt);
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if self.cancel_after.is_some_and(|limit| n >= limit) {
                return Err(AsrError::Cancelled);
            }
            let t_end = job.t_end_ms();
            let t_start = job.t_start_ms;
            let channel = job.channel;
            let pinned = job.language_hint.clone();
            self.seen.lock().unwrap().push(job);
            if self.confused_by_a_pinned_language && pinned.is_some() {
                return Ok(Transcription {
                    channel,
                    t_start_ms: t_start,
                    t_end_ms: t_end,
                    text: "parole che nessuno ha detto".into(),
                    language: pinned,
                    avg_confidence: Some(0.1),
                    model_name: Some("test weights".into()),
                    model_revision: Some("rev1".into()),
                    ..Default::default()
                });
            }
            Ok(Transcription {
                channel,
                t_start_ms: t_start,
                t_end_ms: t_end,
                text: self.text.clone(),
                language: pinned.or(Some("en".into())),
                language_confidence: None,
                avg_confidence: Some(0.9),
                model_name: Some("test weights".into()),
                model_revision: Some("rev1".into()),
                // Mapped through the window's origin, exactly as the real engine
                // does it (see `Engine::transcribe`).
                lines: self
                    .lines
                    .iter()
                    .map(|(from, to, text)| crate::asr::TranscribedLine {
                        t_start_ms: t_start + from,
                        t_end_ms: t_start + to,
                        text: text.clone(),
                        avg_confidence: Some(0.9),
                    })
                    .collect(),
            })
        }

        fn settled_language(&self, _meeting_id: &str) -> Option<String> {
            self.settled.clone()
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

    /// Verbatim what whisper.cpp answers when it will not encode a window.
    const WHISPER_MINUS_SIX: &str =
        "Generic whisper error. Varies depending on the function. Error code: -6";

    /// Stands in for the whole speech engine, and fails the way it does.
    ///
    /// It pads short windows out before decoding, exactly as
    /// [`crate::asr::engine::Engine::transcribe`] does, so the only thing it
    /// cannot read is a window with nothing in it: an empty buffer, or the
    /// silence a chunk reader hands back when it has lost the files. Both of
    /// those come back as whisper's `-6`, because that is what really happens.
    #[derive(Default)]
    struct WhisperLike {
        asked: AtomicU32,
        refused: AtomicU32,
    }

    impl WhisperLike {
        fn asked(&self) -> u32 {
            self.asked.load(Ordering::SeqCst)
        }

        fn refused(&self) -> u32 {
            self.refused.load(Ordering::SeqCst)
        }
    }

    impl Transcriber for WhisperLike {
        async fn transcribe(
            &self,
            job: TranscribeJob,
            _prompt: Option<String>,
        ) -> Result<Transcription, AsrError> {
            self.asked.fetch_add(1, Ordering::SeqCst);
            let readable = job.samples.iter().any(|s| *s != 0.0);
            if !readable {
                self.refused.fetch_add(1, Ordering::SeqCst);
                return Err(AsrError::Transcribe(WHISPER_MINUS_SIX.to_string()));
            }
            Ok(Transcription {
                channel: job.channel,
                t_start_ms: job.t_start_ms,
                t_end_ms: job.t_end_ms(),
                text: "allora, direi che possiamo procedere".into(),
                language: job.language_hint.clone().or(Some("it".into())),
                avg_confidence: Some(0.9),
                model_name: Some("probe weights".into()),
                model_revision: Some("rev1".into()),
                ..Default::default()
            })
        }
    }

    /// The field regression of 2026-08-20, as a bar this pass has to clear.
    ///
    /// A 21-minute meeting full of speech produced 281 stretches, 279 `-6`s and
    /// eight seconds of transcript. Nothing about the plan, the windowing or the
    /// chunk reading was wrong — but nothing in the test suite noticed either,
    /// because every catch-up test asserted on a handful of stretches and an
    /// engine that always answers. This one asserts on the ratio: a meeting this
    /// full of speech has to come out as text almost all of the way through,
    /// whatever the engine underneath is doing.
    #[tokio::test]
    async fn nearly_every_stretch_of_speech_in_a_meeting_ends_up_as_text() {
        let db = connect_in_memory().await.unwrap();
        // 21 minutes on the microphone channel, like the meeting that was lost.
        let id = meeting_with_audio(&db, 42).await;
        let audio = FakeAudio::with_a_meeting_full_of_speech();
        let engine = WhisperLike::default();

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

        assert!(
            report.stretches_packed > 200,
            "a 21-minute meeting of ordinary speech should be hundreds of stretches, not {}",
            report.stretches_packed
        );
        assert_eq!(
            engine.refused(),
            0,
            "the pass handed the engine {} window(s) with no audio in them",
            engine.refused()
        );
        // The words, not the decodes: what the field regression lost was
        // transcript, and this is the bar it failed.
        let written = repo::get_segments(
            &db,
            &TranscriptQuery {
                meeting_id: id.clone(),
                limit: Some(50_000),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let speech_ms = i64::from(report.stretches_packed) * SHORT_UTTERANCE_MS;
        let covered_ms = covered(
            written
                .iter()
                .map(|s| (s.t_start_ms, s.t_end_ms))
                .collect::<Vec<_>>(),
        );
        assert!(
            covered_ms >= speech_ms,
            "{covered_ms} ms of transcript over {speech_ms} ms of speech; the field saw 8 s of 21 min"
        );
        // And it cost a fraction of the decodes it used to: every window is one
        // encode whatever is in it, so the same words now arrive for a seventh
        // of the encoder time.
        assert!(
            report.windows_decoded * 4 < report.stretches_packed,
            "{} windows for {} stretches is not packing",
            report.windows_decoded,
            report.stretches_packed
        );
        assert_eq!(
            engine.asked(),
            report.windows_decoded,
            "one decode a window"
        );
    }

    /// Total length of a set of spans, overlaps counted once.
    fn covered(mut spans: Vec<(i64, i64)>) -> i64 {
        merge(&mut spans);
        spans.iter().map(|(from, to)| to - from).sum()
    }

    // -----------------------------------------------------------------
    // Packing
    // -----------------------------------------------------------------

    /// The lever this module turns: consecutive stretches of one channel go to
    /// the engine as one window, with the real silence between them in it.
    ///
    /// The audio handed over has to be as long as the window is wide. That is
    /// the assertion that says "the gaps are still there": utterances glued
    /// together with the pauses cut out would be shorter than the span they
    /// claim, and every timestamp inside the window would be a lie.
    #[tokio::test]
    async fn consecutive_stretches_become_one_window_with_the_silence_kept() {
        let db = connect_in_memory().await.unwrap();
        let id = meeting_with_audio(&db, 4).await;
        let engine = FakeEngine::saying("uno, due, tre");
        let report = run(
            &engine,
            &db,
            &id,
            CatchUpOptions {
                window_ms: 30_000,
                ..Default::default()
            },
            &FakeAudio::with_a_meeting_full_of_speech(),
        )
        .await
        .unwrap();

        let jobs = engine.spans();
        assert!(!jobs.is_empty());
        assert_eq!(jobs.len() as u32, report.windows_decoded);
        assert!(
            report.stretches_packed >= report.windows_decoded * 4,
            "{} stretches in {} windows is barely packing",
            report.stretches_packed,
            report.windows_decoded
        );
        for (t_start_ms, duration_ms) in &jobs {
            assert!(
                *duration_ms <= MAX_PACK_MS,
                "a {duration_ms} ms window at {t_start_ms} ms is more than one encode"
            );
            assert!(
                *duration_ms > SHORT_UTTERANCE_MS,
                "a {duration_ms} ms window is one stretch, not a packed one"
            );
        }
        // Most of the budget, most of the time: a window that stops well short
        // of `MAX_PACK_MS` for no reason is encoder time spent on zeros.
        let widest = jobs.iter().map(|(_, d)| *d).max().unwrap_or(0);
        assert!(widest > MAX_PACK_MS / 2, "widest window was {widest} ms");
    }

    // -----------------------------------------------------------------------
    // Words Echo should know
    // -----------------------------------------------------------------------

    /// This pass is the one whose text a person keeps, so it is the one that has
    /// to get the names right: the vocabulary goes in front of the audio, and
    /// what still comes back mangled is put right on the way to the database.
    #[tokio::test]
    async fn the_words_echo_was_told_about_reach_the_engine_and_fix_what_it_still_gets_wrong() {
        let db = connect_in_memory().await.unwrap();
        let id = meeting_with_audio(&db, 1).await;
        crate::settings::add_word_to_know(&db, "Langola")
            .await
            .unwrap();
        // Exactly what the 2026-08-24 recording produced for that word.
        let engine = FakeEngine::saying("Allora Nongula è quello che usiamo.");

        let report = run(
            &engine,
            &db,
            &id,
            CatchUpOptions::default(),
            &FakeAudio::with_speech(),
        )
        .await
        .unwrap();
        assert!(report.segments_written > 0, "the pass wrote nothing");
        assert_eq!(
            report.words_corrected, report.segments_written,
            "every line it wrote held the name once"
        );

        let prompts = engine.prompts.lock().unwrap().clone();
        assert!(!prompts.is_empty(), "the pass decoded nothing");
        for prompt in &prompts {
            assert_eq!(
                prompt.as_deref(),
                Some("Langola."),
                "every window of this pass gets the vocabulary"
            );
        }

        let rows = repo::get_segments(
            &db,
            &TranscriptQuery {
                meeting_id: id.clone(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let row = rows.first().expect("a line was written");
        assert_eq!(row.text, "Allora Langola è quello che usiamo.");
        assert_eq!(
            row.corrections,
            vec![crate::types::Correction {
                from: "Nongula".into(),
                to: "Langola".into()
            }],
            "the row has to say what was changed in it"
        );
    }

    /// The invariant every path leans on: an install where nobody has typed
    /// anything decodes and writes exactly what it did before this existed.
    #[tokio::test]
    async fn with_nothing_in_the_list_the_pass_is_untouched() {
        let db = connect_in_memory().await.unwrap();
        let id = meeting_with_audio(&db, 1).await;
        let engine = FakeEngine::saying("Allora Nongula è quello che usiamo.");

        let report = run(
            &engine,
            &db,
            &id,
            CatchUpOptions::default(),
            &FakeAudio::with_speech(),
        )
        .await
        .unwrap();
        assert_eq!(report.words_corrected, 0);
        for prompt in engine.prompts.lock().unwrap().iter() {
            assert_eq!(
                prompt.as_deref(),
                None,
                "nothing goes in front of the audio"
            );
        }

        let rows = repo::get_segments(
            &db,
            &TranscriptQuery {
                meeting_id: id.clone(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let row = rows.first().expect("a line was written");
        assert_eq!(row.text, "Allora Nongula è quello che usiamo.");
        assert!(row.corrections.is_empty());
    }

    /// The 2026-08-24 failure, on the disk lane: the same courtesy phrase, once
    /// where somebody was speaking and once in the pause between two stretches.
    ///
    /// A packed window is mostly silence by construction, so this is the test
    /// that the question is asked about each *line's* own seconds rather than
    /// about the window it arrived in — the difference between a filter that
    /// works and one that would empty a transcript.
    #[tokio::test]
    async fn a_courtesy_line_over_a_pause_goes_and_the_same_line_over_speech_stays() {
        let db = connect_in_memory().await.unwrap();
        let id = meeting_with_audio(&db, 1).await;
        // `with_a_meeting_full_of_speech` finds a 1200 ms stretch every 1500 ms,
        // so offsets 0..1200 of any packed window are speech and 1200..1500 is
        // the pause after it.
        let engine = FakeEngine::in_lines(&[
            (0, SHORT_UTTERANCE_MS, "Grazie."),
            (
                SHORT_UTTERANCE_MS,
                SHORT_UTTERANCE_MS + SHORT_GAP_MS,
                "Grazie.",
            ),
        ]);
        let report = run(
            &engine,
            &db,
            &id,
            CatchUpOptions {
                window_ms: 30_000,
                ..Default::default()
            },
            &FakeAudio::with_a_meeting_full_of_speech(),
        )
        .await
        .unwrap();

        assert!(report.windows_decoded > 0);
        assert_eq!(
            report.phantoms_dropped, report.windows_decoded,
            "the line over the pause should have been dropped once per window"
        );
        assert_eq!(
            report.segments_written, report.windows_decoded,
            "the line over the speech should have been kept once per window"
        );

        let rows = repo::get_segments(
            &db,
            &TranscriptQuery {
                meeting_id: id.clone(),
                include_partial: Some(false),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(rows.len() as u32, report.segments_written);
        for row in &rows {
            assert_eq!(row.text, "Grazie.");
            assert_eq!(
                row.t_end_ms - row.t_start_ms,
                SHORT_UTTERANCE_MS,
                "the surviving row is the one over the speech"
            );
        }
    }

    /// The dangerous half of the same filter, and the reason it counts
    /// milliseconds of voice instead of a share of the window.
    ///
    /// Three short exchanges twelve seconds apart pack into one window, and
    /// whisper answers it with a single line and no timestamps of its own — the
    /// ordinary shape of a short window, and the one neither test above covers.
    /// That line is left spanning the whole packed window, which is mostly pause
    /// **by construction**: a share of it reads as 0.14, under any bar that
    /// catches a cough. Judged that way, a real "Grazie a tutti." at the end of
    /// a call is deleted, and the catch-up pass is the last read of that audio.
    #[tokio::test]
    async fn a_courtesy_line_the_engine_gave_no_timestamps_is_judged_on_the_voice_the_window_holds()
    {
        // Each exchange really was speech: 600 ms of voice in 1200 ms.
        let real = run_one_line_over_far_apart_speech("Grazie a tutti.", 600, None).await;
        assert_eq!(
            real.phantoms_dropped, 0,
            "the recording holds seconds of voice under that line"
        );
        assert!(real.segments_written > 0, "and the words were written down");

        // And the filter still bites on a window it can see the whole of: one
        // stretch, opened by the three windows of a cough and nothing more.
        let blip = run_one_line_over_far_apart_speech("Grazie a tutti.", 96, Some(2_000)).await;
        assert_eq!(blip.segments_written, 0);
        assert_eq!(blip.phantoms_dropped, 1);

        // Ordinary words are never this filter's business, however quiet.
        let words =
            run_one_line_over_far_apart_speech("allora vediamo domani", 96, Some(2_000)).await;
        assert_eq!(words.phantoms_dropped, 0);
        assert_eq!(words.segments_written, 1);
    }

    /// One meeting of far-apart exchanges, decoded as one line per window with
    /// no per-line timestamps — what a real engine hands back for a window it
    /// heard one sentence in. `to_ms` cuts the pass short, so a test can have
    /// one stretch in the window instead of three.
    async fn run_one_line_over_far_apart_speech(
        said: &str,
        voiced_ms: i64,
        to_ms: Option<i64>,
    ) -> CatchUpReport {
        let db = connect_in_memory().await.unwrap();
        let id = meeting_with_audio(&db, 1).await;
        run(
            &FakeEngine::saying(said),
            &db,
            &id,
            CatchUpOptions {
                window_ms: 30_000,
                to_ms,
                ..Default::default()
            },
            &FakeAudio::with_a_few_exchanges_far_apart(voiced_ms),
        )
        .await
        .unwrap()
    }

    /// A real sentence in the same pause is not this filter's business, and a
    /// window with no phantom in it is not touched at all.
    #[tokio::test]
    async fn ordinary_words_in_a_pause_are_left_exactly_where_they_are() {
        let db = connect_in_memory().await.unwrap();
        let id = meeting_with_audio(&db, 1).await;
        let engine = FakeEngine::in_lines(&[
            (
                0,
                SHORT_UTTERANCE_MS,
                "allora, direi che possiamo procedere",
            ),
            (
                SHORT_UTTERANCE_MS,
                SHORT_UTTERANCE_MS + SHORT_GAP_MS,
                "sì, esatto",
            ),
        ]);
        let report = run(
            &engine,
            &db,
            &id,
            CatchUpOptions {
                window_ms: 30_000,
                ..Default::default()
            },
            &FakeAudio::with_a_meeting_full_of_speech(),
        )
        .await
        .unwrap();

        assert_eq!(report.phantoms_dropped, 0);
        assert_eq!(report.segments_written, report.windows_decoded * 2);
    }

    /// The before-and-after measurement is honest: with the budget at one
    /// millisecond the pass is what it was before packing — one decode per
    /// speech stretch — so `catchup_probe` compares two runs of the same code
    /// over the same audio rather than a number from a notebook.
    #[tokio::test]
    async fn a_budget_of_nothing_is_the_pass_as_it_was_before_packing() {
        let db = connect_in_memory().await.unwrap();
        let id = meeting_with_audio(&db, 2).await;
        let engine = FakeEngine::saying("uno, due, tre");
        let report = run(
            &engine,
            &db,
            &id,
            CatchUpOptions {
                window_ms: 30_000,
                pack_ms: 1,
                ..Default::default()
            },
            &FakeAudio::with_a_meeting_full_of_speech(),
        )
        .await
        .unwrap();

        assert_eq!(
            report.windows_decoded, report.stretches_packed,
            "one decode per stretch is the old pass"
        );
        assert!(report.stretches_packed > 30);
        for (_, duration_ms) in engine.spans() {
            assert_eq!(duration_ms, SHORT_UTTERANCE_MS, "one stretch, no packing");
        }
    }

    /// A window is a span of *this* hole, never a step into text that already
    /// exists. Packing across a boundary would say the same words twice.
    #[tokio::test]
    async fn a_packed_window_stays_inside_the_hole_it_is_filling() {
        let db = connect_in_memory().await.unwrap();
        let id = meeting_with_audio(&db, 4).await;
        // The live pass got the middle minute. Speech on both sides of it.
        already_written(&db, &id, Ch::Mic, 30_000, 60_000).await;

        let engine = FakeEngine::saying("either side");
        run(
            &engine,
            &db,
            &id,
            CatchUpOptions {
                window_ms: 30_000,
                ..Default::default()
            },
            &FakeAudio::with_a_meeting_full_of_speech(),
        )
        .await
        .unwrap();

        for (t_start_ms, duration_ms) in engine.spans() {
            let t_end_ms = t_start_ms + duration_ms;
            let before = t_end_ms <= 30_000;
            let after = t_start_ms >= 60_000;
            assert!(
                before || after,
                "a window ran from {t_start_ms} to {t_end_ms}, across text that was already there"
            );
        }
    }

    /// One row per line the engine wrote, each at its own place on the meeting
    /// clock — not one row for the whole window.
    #[tokio::test]
    async fn the_lines_inside_a_packed_window_land_at_their_own_times() {
        let db = connect_in_memory().await.unwrap();
        let id = meeting_with_audio(&db, 1).await;
        // A twenty-second hole full of short stretches: one packed window.
        already_written(&db, &id, Ch::Mic, 20_000, 30_000).await;
        // Three sentences inside one window, at offsets the engine reports.
        let engine = FakeEngine::in_lines(&[
            (0, 4_000, "allora, partiamo"),
            (6_500, 9_000, "sì, d'accordo"),
            (12_000, 20_000, "ci vediamo giovedì"),
        ]);
        let report = run(
            &engine,
            &db,
            &id,
            CatchUpOptions {
                window_ms: 30_000,
                ..Default::default()
            },
            &FakeAudio::with_a_meeting_full_of_speech(),
        )
        .await
        .unwrap();

        let window_start = engine.spans()[0].0;
        let written: Vec<_> = repo::get_segments(
            &db,
            &TranscriptQuery {
                meeting_id: id.clone(),
                ..Default::default()
            },
        )
        .await
        .unwrap()
        .into_iter()
        .filter(|s| s.t_start_ms < 20_000)
        .collect();
        assert_eq!(report.windows_decoded, 1, "one window");
        assert_eq!(report.segments_written, 3, "three sentences, three rows");
        assert_eq!(written.len(), 3);
        assert_eq!(written[0].t_start_ms, window_start);
        assert_eq!(written[0].t_end_ms, window_start + 4_000);
        assert_eq!(written[1].t_start_ms, window_start + 6_500);
        assert_eq!(written[2].text, "ci vediamo giovedì");
        assert!(written.iter().all(|s| s.is_final));
        assert!(written.iter().all(|s| s.language.as_deref() == Some("en")));
        assert!(written
            .iter()
            .all(|s| s.model_name.as_deref() == Some("test weights")));
    }

    /// Whisper can hand back a line with no width at the very end of a window.
    /// The words are real, so they get the rest of the window rather than a span
    /// the coverage machinery would never count.
    #[test]
    fn a_line_with_no_width_is_given_the_rest_of_the_window() {
        let pack = Pack {
            from_ms: 10_000,
            to_ms: 38_000,
            stretches: 3,
            voiced: VoicedSpans::default(),
        };
        let text = Transcription {
            text: "a b".into(),
            lines: vec![
                crate::asr::TranscribedLine {
                    t_start_ms: 10_000,
                    t_end_ms: 20_000,
                    text: "a".into(),
                    avg_confidence: Some(0.9),
                },
                crate::asr::TranscribedLine {
                    t_start_ms: 38_000,
                    t_end_ms: 38_000,
                    text: "b".into(),
                    avg_confidence: Some(0.9),
                },
            ],
            ..Default::default()
        };
        let rows = split_onto_the_transcript(&text, &pack);
        assert_eq!(rows.len(), 2, "the words at the end were dropped");
        assert_eq!(
            (rows[1].t_start_ms, rows[1].t_end_ms),
            (38_000 - MIN_ROW_MS, 38_000)
        );

        // A line whisper puts past the end of the window is clipped to it, and
        // one with nothing in it is dropped.
        let overshooting = Transcription {
            text: "c".into(),
            lines: vec![
                crate::asr::TranscribedLine {
                    t_start_ms: 30_000,
                    t_end_ms: 99_000,
                    text: "c".into(),
                    avg_confidence: None,
                },
                crate::asr::TranscribedLine {
                    t_start_ms: 31_000,
                    t_end_ms: 32_000,
                    text: "  ...  ".into(),
                    avg_confidence: None,
                },
            ],
            ..Default::default()
        };
        let rows = split_onto_the_transcript(&overshooting, &pack);
        assert_eq!(rows.len(), 1, "punctuation over silence is not a row");
        assert_eq!(rows[0].t_end_ms, pack.to_ms);
    }

    #[test]
    fn a_window_is_full_when_one_more_stretch_would_cost_a_second_encode() {
        let mut pack = Pack {
            from_ms: 0,
            to_ms: 4_000,
            stretches: 1,
            voiced: VoicedSpans::default(),
        };
        let near = Pack {
            from_ms: 20_000,
            to_ms: MAX_PACK_MS,
            stretches: 1,
            voiced: VoicedSpans::default(),
        };
        assert!(pack.would_hold(&near, MAX_PACK_MS));
        pack.extend(&near);
        assert_eq!(pack.len_ms(), MAX_PACK_MS);
        assert_eq!(pack.stretches, 2);
        // One millisecond past the budget is a second encode.
        assert!(!pack.would_hold(
            &Pack {
                from_ms: MAX_PACK_MS,
                to_ms: MAX_PACK_MS + 1,
                stretches: 1,
                voiced: VoicedSpans::default(),
            },
            MAX_PACK_MS
        ));
        // A long silence is not a reason to close a window: carrying it costs
        // nothing, and closing costs a whole encode.
        let after_a_long_pause = Pack {
            from_ms: 20_000,
            to_ms: 24_000,
            stretches: 1,
            voiced: VoicedSpans::default(),
        };
        let short = Pack {
            from_ms: 0,
            to_ms: 1_200,
            stretches: 1,
            voiced: VoicedSpans::default(),
        };
        assert!(short.would_hold(&after_a_long_pause, MAX_PACK_MS));
        // And a budget of one millisecond is the pass as it worked before
        // packing: every stretch a window of its own. `catchup_probe` measures
        // the before-and-after with it.
        assert!(!short.would_hold(&after_a_long_pause, 1));
    }

    /// The pass that runs during a meeting packs, but not as wide: a caption
    /// waits behind whatever decode is in flight.
    #[test]
    fn the_window_a_recording_shares_the_engine_with_is_narrower() {
        const { assert!(LIVE_PACK_MS < MAX_PACK_MS) };
        // Still worth packing: several stretches an encode, not one.
        const { assert!(LIVE_PACK_MS >= 8_000) };
    }

    #[test]
    fn one_packed_window_is_one_encode_of_whisper_cpps_thirty_second_mel() {
        // whisper.cpp pads every input to 30 s (`log_mel_spectrogram`, and the
        // encoder's fixed `2*n_ctx` window), so the budget has to sit under it
        // with room to spare or the saving is spent on a second encode.
        const { assert!(MAX_PACK_MS < 30_000) };
        const { assert!(MAX_PACK_MS >= 24_000) };
        // And the detector's own cap is inside the budget, so no single stretch
        // can overflow a window on its own.
        const { assert!(crate::audio::vad::MAX_UTTERANCE_MS <= MAX_PACK_MS) };
    }

    #[test]
    fn audio_is_kept_only_while_a_window_might_still_want_it() {
        let mut held = Held::new(1_000);
        held.push(1_000, &[0.5f32; SR]); // 1_000..2_000
        held.push(2_000, &[0.25f32; SR]); // 2_000..3_000
        assert_eq!(held.span(1_000, 3_000).map(|s| s.len()), Some(2 * SR));
        assert_eq!(held.span(1_500, 2_500).map(|s| s.len()), Some(SR));
        // Not held: before the start, or past the end.
        assert!(held.span(0, 500).is_none());
        assert!(held.span(2_000, 4_000).is_none());

        held.forget_before(2_000);
        assert!(held.span(1_000, 3_000).is_none(), "it let that go");
        assert_eq!(held.span(2_000, 3_000).map(|s| s.len()), Some(SR));
        // Forgetting backwards is a no-op, never a truncation.
        held.forget_before(0);
        assert_eq!(held.span(2_000, 3_000).map(|s| s.len()), Some(SR));

        // A read that does not carry on from the last one starts again there,
        // rather than pretending the audio in hand is continuous.
        held.push(9_000, &[0.75f32; SR]);
        assert!(held.span(2_000, 3_000).is_none());
        assert_eq!(held.span(9_000, 10_000).map(|s| s.len()), Some(SR));
    }

    /// The second reading a stretch gets when the meeting's language cannot
    /// explain it is a decode of the same audio, and the pass counts it — that
    /// counter is how a future slow catch-up gets diagnosed from a log.
    #[tokio::test]
    async fn a_second_reading_of_a_window_is_counted_as_a_fallback() {
        let db = connect_in_memory().await.unwrap();
        let id = meeting_with_audio(&db, 1).await;
        repo::set_meeting_language(&db, &id, "it").await.unwrap();

        let engine = FakeEngine {
            text: "and then we shipped it".into(),
            confused_by_a_pinned_language: true,
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
        assert_eq!(report.windows_decoded, 1);
        assert_eq!(
            report.fallback_attempts, 1,
            "one window, one second reading"
        );

        // Nothing to explain, nothing to count.
        let db = connect_in_memory().await.unwrap();
        let id = meeting_with_audio(&db, 1).await;
        let clean = run(
            &FakeEngine::saying("chiaro"),
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
        assert_eq!(clean.fallback_attempts, 0);
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
    async fn a_gap_long_enough_to_hold_a_short_word_is_read() {
        let db = connect_in_memory().await.unwrap();
        let id = meeting_with_audio(&db, 1).await;
        // "Sì." — a third of a second the live pass never wrote down. The old
        // 800 ms floor stepped straight over holes like this one.
        already_written(&db, &id, Ch::Mic, 0, 10_000).await;
        already_written(&db, &id, Ch::Mic, 10_300, 30_000).await;

        let audio = FakeAudio::with_speech();
        let report = run(
            &FakeEngine::saying("sì"),
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
        assert_eq!(audio.reads(), vec![(Channel::Mic, 10_000, 10_300)]);
        assert_eq!(report.segments_written, 1);
    }

    #[test]
    fn the_shortest_hole_worth_reading_is_still_shorter_than_a_word() {
        // Long enough that ordinary padding differences are not chased, short
        // enough that a one-syllable answer cannot hide in it.
        assert!((150..=400).contains(&MIN_GAP_MS), "{MIN_GAP_MS} ms");
    }

    #[tokio::test]
    async fn one_detector_is_streamed_through_a_whole_stretch() {
        let db = connect_in_memory().await.unwrap();
        // A minute and a half of audio with no text against it at all, read in
        // 30 s windows: three reads, one continuous stretch.
        let id = meeting_with_audio(&db, 3).await;
        let audio = FakeAudio::with_one_long_sentence();
        let engine = FakeEngine::saying("one long answer that ran across every read");

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

        assert_eq!(
            audio.detectors_opened(),
            1,
            "a fresh detector per window resets speech detection every 30 s and splits words"
        );
        assert_eq!(
            audio.fed(),
            vec![
                (Channel::Mic, 0, 30_000),
                (Channel::Mic, 30_000, 60_000),
                (Channel::Mic, 60_000, 90_000),
            ],
            "the detector was not fed the stretch in order"
        );
        // Speech that straddled both boundaries came back as one utterance, and
        // it is written down even though it only finished when the audio ran out.
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
        assert_eq!(written.len(), 1);
        assert_eq!(written[0].t_start_ms, 0);
        assert_eq!(written[0].t_end_ms, 90_000);
    }

    #[tokio::test]
    async fn a_separate_detector_for_each_separate_stretch() {
        let db = connect_in_memory().await.unwrap();
        let id = meeting_with_audio(&db, 3).await;
        // Two holes with written text between them: two stretches, and audio on
        // either side of that text is not continuous with anything.
        already_written(&db, &id, Ch::Mic, 30_000, 60_000).await;

        let audio = FakeAudio::with_one_long_sentence();
        let report = run(
            &FakeEngine::saying("either side"),
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
        assert_eq!(audio.detectors_opened(), 2);
        assert_eq!(report.segments_written, 2);
    }

    #[tokio::test]
    async fn the_meetings_own_language_is_what_the_engine_is_told() {
        let db = connect_in_memory().await.unwrap();
        let id = meeting_with_audio(&db, 1).await;
        repo::set_meeting_language(&db, &id, "it").await.unwrap();

        let engine = FakeEngine::saying("allora, ci vediamo giovedì");
        run(
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

        assert_eq!(
            engine.hints(),
            vec![Some("it".to_string())],
            "catch-up asked the engine to guess the language again"
        );
    }

    #[tokio::test]
    async fn a_stretch_the_meetings_language_cannot_explain_is_read_again_without_it() {
        let db = connect_in_memory().await.unwrap();
        let id = meeting_with_audio(&db, 1).await;
        repo::set_meeting_language(&db, &id, "it").await.unwrap();

        let engine = FakeEngine {
            text: "and then we shipped it".into(),
            confused_by_a_pinned_language: true,
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

        // Pinned first, then asked again with nothing pinned — a prior, not a
        // lock.
        assert_eq!(engine.hints(), vec![Some("it".to_string()), None]);
        assert_eq!(report.segments_written, 1, "only one of the two was kept");
        let written = repo::get_segments(
            &db,
            &TranscriptQuery {
                meeting_id: id.clone(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(written.len(), 1);
        assert_eq!(
            written[0].text, "and then we shipped it",
            "the confident reading was thrown away for the collapsed one"
        );
    }

    #[test]
    fn a_collapsed_reading_is_one_worth_trying_again() {
        let good = Transcription {
            text: "chiaro".into(),
            avg_confidence: Some(0.8),
            ..Default::default()
        };
        let shaky = Transcription {
            text: "chiaro".into(),
            avg_confidence: Some(0.1),
            ..Default::default()
        };
        let nothing = Transcription::default();
        assert!(!collapsed(&good));
        assert!(collapsed(&shaky));
        assert!(collapsed(&nothing), "silence is worth a second look too");
        // An engine that reports no confidence at all is taken at its word.
        assert!(!collapsed(&Transcription {
            text: "chiaro".into(),
            ..Default::default()
        }));

        assert!(improves_on(&good, &shaky));
        assert!(!improves_on(&shaky, &good));
        assert!(!improves_on(&nothing, &shaky), "nothing never wins");
        assert!(improves_on(&shaky, &nothing));
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

    /// A stop is honoured at the next boundary, and everything already decoded
    /// stays.
    ///
    /// The window still being packed when the stop arrives is dropped rather
    /// than decoded on the way out — a stop is usually a recording starting, and
    /// that wants the machine now, not after one more beam search. Nothing is
    /// lost by it: no text was written over that audio, so it is still a hole
    /// and the next pass reads it (mantra 3).
    #[tokio::test]
    async fn cancelling_stops_between_windows_and_keeps_what_was_written() {
        let db = connect_in_memory().await.unwrap();
        let id = meeting_with_audio(&db, 4).await;
        let reads = Arc::new(AtomicU32::new(0));
        let counted = reads.clone();
        // Let two windows through, then ask it to stop. Two, because each 30 s
        // window is a packed window of its own here and the first one is still
        // open until the second closes it.
        let watcher: CancelCheck = Arc::new(move || counted.fetch_add(1, Ordering::SeqCst) >= 2);

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
        assert_eq!(report.windows_read, 2, "it stopped at the next boundary");
        assert_eq!(report.segments_written, 1, "and kept what it had done");
        let written = repo::get_segments(
            &db,
            &TranscriptQuery {
                meeting_id: id.clone(),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(written.len(), 1);
        assert_eq!(
            written[0].t_start_ms, 0,
            "the window it finished is the one it kept"
        );
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
        assert_eq!(subtract(&[(0, 100)], &[(40, 60)]), vec![(0, 40), (60, 100)]);
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
    fn the_default_read_is_as_long_as_the_longest_utterance() {
        // Not a boundary for anything — one detector is streamed across every
        // read — but a read shorter than an utterance would mean holding the
        // same speech across several reads for no reason.
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
