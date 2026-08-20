//! Per-channel chunked audio on disk, the source of truth (mantra 3).
//!
//! Contract (DESIGN §3 Crash recovery, review findings 18, 22, 34):
//! * Writing starts at t=0 for every channel, before anything downstream runs.
//! * One file per [`crate::audio::CHUNK_SECONDS`] window per channel. Rolling
//!   to a new file is the commit point: flush, fsync, then mark the journal row
//!   committed via [`crate::db::repo::commit_chunk`].
//! * A chunk that is not committed may be incomplete. Recovery trusts committed
//!   chunks only.
//! * Runs on its own thread, fed by the ring buffer. If the disk stalls, the
//!   ring buffer absorbs it and live transcription is dropped first, audio is
//!   the last thing to give way.
//! * Disk-full is handled: stop cleanly, mark the meeting degraded, tell the
//!   person in plain words.
//!
//! ## Why WAV today, FLAC the moment it can be read back
//!
//! DESIGN calls for FLAC, and [`ChunkFormat::Flac`] below writes real FLAC via
//! `flacenc`. It is not the default yet for two reasons:
//!
//! 1. `flacenc` only *encodes* unless its `decode` feature is enabled, and no
//!    other decoder is in the dependency tree. A chunk Echo cannot read back is
//!    useless: the ASR catch-up pass, the mixdown and click-to-play all read
//!    chunks off disk.
//! 2. FLAC has to be encoded a whole block at a time, so a chunk would only
//!    reach the disk when it rolls. WAV is written sample by sample, which means
//!    a crash costs milliseconds instead of a whole chunk window — a better fit
//!    for "raw audio on disk is the source of truth".
//!
//! Flipping to FLAC is one line here plus `features = ["decode"]` on `flacenc`.
//! 16 kHz mono 16-bit is 32 kB/s, about 115 MB an hour per channel, so the cost
//! of waiting is space, not correctness.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::audio::resample::resample_buffer;
use crate::audio::{AudioError, Frame, CHUNK_SECONDS, TARGET_SAMPLE_RATE};
use crate::types::Channel;

/// Result of closing one chunk, ready to be journalled.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CommittedChunk {
    pub channel: Channel,
    pub seq: u64,
    pub path: PathBuf,
    pub t_start_ms: i64,
    pub t_end_ms: i64,
}

impl CommittedChunk {
    pub fn duration_ms(&self) -> i64 {
        self.t_end_ms - self.t_start_ms
    }
}

/// Container for a chunk on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkFormat {
    /// 16-bit PCM, written incrementally, readable with what is already
    /// compiled in.
    Wav,
    /// Lossless and about half the size, but only writable today (see the
    /// module docs).
    Flac,
}

impl ChunkFormat {
    pub fn extension(self) -> &'static str {
        match self {
            ChunkFormat::Wav => "wav",
            ChunkFormat::Flac => "flac",
        }
    }

    fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_ascii_lowercase().as_str() {
            "wav" => Some(ChunkFormat::Wav),
            "flac" => Some(ChunkFormat::Flac),
            _ => None,
        }
    }
}

/// What new recordings are written as.
pub const DEFAULT_CHUNK_FORMAT: ChunkFormat = ChunkFormat::Wav;

/// The chunk file name a channel and sequence number produce.
pub fn chunk_file_name(channel: Channel, seq: u64, format: ChunkFormat) -> String {
    format!("{}-{:06}.{}", channel.as_str(), seq, format.extension())
}

/// Flush the open chunk to the operating system this often, so a crash costs
/// milliseconds of audio rather than a whole chunk window.
const FLUSH_EVERY: std::time::Duration = std::time::Duration::from_millis(500);

/// How far a frame's own timestamp may sit from where the file ends before the
/// writer treats it as lost audio. Two frames' worth: ordinary rounding between
/// resampling and the clock is smaller than this.
const PLACEMENT_TOLERANCE_MS: i64 = crate::audio::FRAME_MS as i64 * 2;

/// The longest hole the writer fills with silence. Anything bigger is a real
/// outage — a device unplugged for minutes — and is left as a gap between chunks
/// instead: every reader places chunks by their journalled offsets, so a gap
/// reads back as silence without costing a megabyte a minute on disk.
const MAX_SILENCE_PAD_MS: i64 = CHUNK_SECONDS as i64 * 1_000 * 2;

enum Sink {
    Wav(hound::WavWriter<BufWriter<File>>),
    /// FLAC has to be encoded in one go, so samples are held until the roll.
    Flac(Vec<i32>),
}

struct OpenChunk {
    seq: u64,
    path: PathBuf,
    t_start_ms: i64,
    samples: usize,
    sink: Sink,
    last_flush: Instant,
}

/// Writes one channel's audio as a sequence of chunk files.
pub struct ChunkWriter {
    audio_dir: PathBuf,
    channel: Channel,
    sample_rate: u32,
    format: ChunkFormat,
    chunk_samples: usize,
    next_seq: u64,
    /// Where the next chunk starts on the meeting clock. Derived from samples
    /// written, never from a wall clock.
    next_t_start_ms: i64,
    open: Option<OpenChunk>,
    bytes_written: u64,
    /// Set once the disk refused us; every later write is a no-op so already
    /// committed chunks survive untouched.
    fatal: Option<String>,
}

impl std::fmt::Debug for ChunkWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChunkWriter")
            .field("channel", &self.channel)
            .field("format", &self.format)
            .field("next_seq", &self.next_seq)
            .field("bytes_written", &self.bytes_written)
            .field("fatal", &self.fatal)
            .finish()
    }
}

impl ChunkWriter {
    /// Prepare to write `channel` into `audio_dir`. Creates the directory.
    pub fn create(
        audio_dir: &Path,
        channel: Channel,
        sample_rate: u32,
    ) -> Result<Self, AudioError> {
        Self::create_with_format(audio_dir, channel, sample_rate, DEFAULT_CHUNK_FORMAT)
    }

    pub fn create_with_format(
        audio_dir: &Path,
        channel: Channel,
        sample_rate: u32,
        format: ChunkFormat,
    ) -> Result<Self, AudioError> {
        if sample_rate == 0 {
            return Err(AudioError::Write("sample rate cannot be zero".into()));
        }
        std::fs::create_dir_all(audio_dir).map_err(|e| write_error(&e))?;
        Ok(Self {
            audio_dir: audio_dir.to_path_buf(),
            channel,
            sample_rate,
            format,
            chunk_samples: sample_rate as usize * CHUNK_SECONDS as usize,
            next_seq: 0,
            next_t_start_ms: 0,
            open: None,
            bytes_written: 0,
            fatal: None,
        })
    }

    /// Continue a recording that was interrupted: the next chunk gets `seq` and
    /// starts at `t_start_ms`.
    pub fn resume_at(&mut self, seq: u64, t_start_ms: i64) {
        self.next_seq = seq;
        self.next_t_start_ms = t_start_ms;
    }

    pub fn channel(&self) -> Channel {
        self.channel
    }

    pub fn format(&self) -> ChunkFormat {
        self.format
    }

    /// Where the audio written so far reaches on the meeting clock.
    ///
    /// This is the writer's own truth: samples on disk, counted. It is what
    /// makes the writer — not whatever upstream stage happened to lose a block —
    /// the authority on where a frame belongs.
    pub fn written_to_ms(&self) -> i64 {
        let open_ms = self
            .open
            .as_ref()
            .map(|open| open.samples as i64 * 1_000 / i64::from(self.sample_rate))
            .unwrap_or(0);
        self.next_t_start_ms + open_ms
    }

    /// Append a frame **at the position it says it belongs**, and report every
    /// chunk that became durable along the way.
    ///
    /// A frame carries an offset on the monotonic meeting clock. If that offset
    /// is later than where this file ends, audio was lost somewhere upstream —
    /// the callback ring dropped blocks under load, a write failed, a device
    /// handed over late — and simply appending would slide the rest of the
    /// meeting earlier than it really happened, permanently, for every segment
    /// and every speaker turn after it. So the difference is filled: with
    /// silence for a small hole, or with a gap between chunks for a long outage
    /// (readers place chunks by their journalled offsets either way).
    pub fn write_frame(&mut self, frame: &Frame) -> Result<Vec<CommittedChunk>, AudioError> {
        if let Some(reason) = &self.fatal {
            return Err(AudioError::Write(reason.clone()));
        }
        let mut committed = Vec::new();
        if frame.samples.is_empty() {
            return Ok(committed);
        }
        let expected = self.written_to_ms();
        let drift = frame.t_start_ms - expected;
        let mut samples: &[f32] = &frame.samples;

        if drift > PLACEMENT_TOLERANCE_MS {
            if drift <= MAX_SILENCE_PAD_MS {
                tracing::debug!(
                    target: "echo::audio",
                    channel = self.channel.as_str(),
                    drift_ms = drift,
                    "filling a hole so the recording stays in step with the clock"
                );
                let silence = vec![0.0f32; self.ms_to_samples(drift)];
                self.write_samples_into(&silence, &mut committed)?;
            } else {
                tracing::warn!(
                    target: "echo::audio",
                    channel = self.channel.as_str(),
                    drift_ms = drift,
                    "a long stretch of this channel went missing; carrying on at the right offset"
                );
                // Close what is open and restart the numbering at the frame's own
                // offset, leaving an honest hole in the journal.
                if let Some(chunk) = self.roll()? {
                    committed.push(chunk);
                }
                self.next_t_start_ms = frame.t_start_ms;
            }
        } else if drift < -PLACEMENT_TOLERANCE_MS {
            // This frame overlaps audio already on disk. What is written stays
            // written; only the part that would double up is dropped.
            let overlap = self.ms_to_samples(-drift).min(samples.len());
            tracing::debug!(
                target: "echo::audio",
                channel = self.channel.as_str(),
                drift_ms = drift,
                "trimming audio that overlaps what is already saved"
            );
            samples = &samples[overlap..];
            if samples.is_empty() {
                return Ok(committed);
            }
        }

        self.write_samples_into(samples, &mut committed)?;
        Ok(committed)
    }

    /// Append a frame. Kept for callers that only care about the last chunk to
    /// become durable; [`ChunkWriter::write_frame`] reports all of them.
    pub fn write(&mut self, frame: &Frame) -> Result<Option<CommittedChunk>, AudioError> {
        Ok(self.write_frame(frame)?.pop())
    }

    fn ms_to_samples(&self, ms: i64) -> usize {
        (ms.max(0) * i64::from(self.sample_rate) / 1_000) as usize
    }

    /// Append raw 16 kHz mono samples at the end of the file, wherever that is.
    /// Prefer [`ChunkWriter::write_frame`] while recording: it knows where the
    /// audio belongs.
    pub fn write_samples(&mut self, samples: &[f32]) -> Result<Option<CommittedChunk>, AudioError> {
        let mut committed = Vec::new();
        self.write_samples_into(samples, &mut committed)?;
        Ok(committed.pop())
    }

    fn write_samples_into(
        &mut self,
        samples: &[f32],
        committed: &mut Vec<CommittedChunk>,
    ) -> Result<(), AudioError> {
        if let Some(reason) = &self.fatal {
            return Err(AudioError::Write(reason.clone()));
        }
        if samples.is_empty() {
            return Ok(());
        }

        let mut remaining = samples;
        while !remaining.is_empty() {
            self.ensure_open()?;
            let open = self.open.as_mut().expect("just opened");
            let room = self.chunk_samples.saturating_sub(open.samples);
            debug_assert!(room > 0, "a full chunk should already have rolled");
            let take = room.min(remaining.len());
            if take == 0 {
                break;
            }
            let piece = &remaining[..take];

            if let Err(e) = append(&mut open.sink, piece) {
                self.fatal = Some(user_facing_write_error(&e));
                return Err(write_error(&e));
            }
            open.samples += piece.len();
            self.bytes_written += (piece.len() * 2) as u64;
            remaining = &remaining[piece.len()..];

            if open.last_flush.elapsed() >= FLUSH_EVERY {
                if let Err(e) = flush(&mut open.sink) {
                    self.fatal = Some(user_facing_write_error(&e));
                    return Err(write_error(&e));
                }
                open.last_flush = Instant::now();
            }

            if open.samples >= self.chunk_samples {
                // Every roll is reported, so padding a hole that spans a chunk
                // boundary still leaves a journal row per file on disk.
                if let Some(chunk) = self.roll()? {
                    committed.push(chunk);
                }
            }
        }
        Ok(())
    }

    /// Close the current chunk early: flush, fsync, return it.
    pub fn flush(&mut self) -> Result<Option<CommittedChunk>, AudioError> {
        if self.open.is_none() {
            return Ok(None);
        }
        self.roll()
    }

    /// Finish: flush and close, returning the last chunk.
    pub fn finish(mut self) -> Result<Option<CommittedChunk>, AudioError> {
        self.flush()
    }

    /// Bytes written so far, for the storage report.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// The offset the next chunk will start at.
    pub fn next_t_start_ms(&self) -> i64 {
        self.next_t_start_ms
    }

    /// Set once the disk gave up. Already committed chunks are untouched.
    pub fn fatal_error(&self) -> Option<&str> {
        self.fatal.as_deref()
    }

    fn ensure_open(&mut self) -> Result<(), AudioError> {
        if self.open.is_some() {
            return Ok(());
        }
        let seq = self.next_seq;
        let path = self
            .audio_dir
            .join(chunk_file_name(self.channel, seq, self.format));
        let sink = match self.format {
            ChunkFormat::Wav => {
                let spec = hound::WavSpec {
                    channels: 1,
                    sample_rate: self.sample_rate,
                    bits_per_sample: 16,
                    sample_format: hound::SampleFormat::Int,
                };
                let file = File::create(&path).map_err(|e| {
                    self.fatal = Some(user_facing_write_error(&e));
                    write_error(&e)
                })?;
                let writer = hound::WavWriter::new(BufWriter::new(file), spec)
                    .map_err(|e| AudioError::Write(e.to_string()))?;
                Sink::Wav(writer)
            }
            ChunkFormat::Flac => Sink::Flac(Vec::with_capacity(self.chunk_samples)),
        };
        self.open = Some(OpenChunk {
            seq,
            path,
            t_start_ms: self.next_t_start_ms,
            samples: 0,
            sink,
            last_flush: Instant::now(),
        });
        Ok(())
    }

    fn roll(&mut self) -> Result<Option<CommittedChunk>, AudioError> {
        let Some(open) = self.open.take() else {
            return Ok(None);
        };
        let OpenChunk {
            seq,
            path,
            t_start_ms,
            samples,
            sink,
            ..
        } = open;

        if samples == 0 {
            // Nothing was written; do not leave an empty file behind.
            let _ = std::fs::remove_file(&path);
            return Ok(None);
        }

        if let Err(e) = finalize(sink, &path, self.sample_rate) {
            self.fatal = Some(user_facing_write_error_str(&e.to_string()));
            return Err(e);
        }

        let duration_ms = samples as i64 * 1_000 / i64::from(self.sample_rate);
        let chunk = CommittedChunk {
            channel: self.channel,
            seq,
            path,
            t_start_ms,
            t_end_ms: t_start_ms + duration_ms,
        };
        self.next_seq = seq + 1;
        self.next_t_start_ms = chunk.t_end_ms;
        Ok(Some(chunk))
    }
}

impl Drop for ChunkWriter {
    fn drop(&mut self) {
        // Losing a recording because nobody called finish() would be
        // unforgivable; commit whatever is open on the way out.
        if self.open.is_some() {
            let _ = self.roll();
        }
    }
}

fn append(sink: &mut Sink, samples: &[f32]) -> std::io::Result<()> {
    match sink {
        Sink::Wav(w) => {
            for s in samples {
                w.write_sample(to_i16(*s))
                    .map_err(|e| std::io::Error::other(e.to_string()))?;
            }
            Ok(())
        }
        Sink::Flac(buf) => {
            buf.extend(samples.iter().map(|s| i32::from(to_i16(*s))));
            Ok(())
        }
    }
}

fn flush(sink: &mut Sink) -> std::io::Result<()> {
    match sink {
        Sink::Wav(w) => w.flush().map_err(|e| std::io::Error::other(e.to_string())),
        Sink::Flac(_) => Ok(()),
    }
}

/// Close the sink, get the bytes on disk, and make them durable before anyone
/// is told the chunk exists.
fn finalize(sink: Sink, path: &Path, sample_rate: u32) -> Result<(), AudioError> {
    match sink {
        Sink::Wav(w) => {
            w.finalize()
                .map_err(|e| AudioError::Write(user_facing_write_error_str(&e.to_string())))?;
        }
        Sink::Flac(buf) => {
            let bytes = encode_flac(&buf, sample_rate)?;
            let mut file = File::create(path).map_err(|e| write_error(&e))?;
            file.write_all(&bytes).map_err(|e| write_error(&e))?;
            file.flush().map_err(|e| write_error(&e))?;
        }
    }
    // fsync the file, then the directory, so both the contents and the name
    // survive a power cut before we call the chunk committed.
    let file = File::open(path).map_err(|e| write_error(&e))?;
    file.sync_all().map_err(|e| write_error(&e))?;
    if let Some(dir) = path.parent() {
        if let Ok(d) = File::open(dir) {
            let _ = d.sync_all();
        }
    }
    Ok(())
}

fn encode_flac(samples: &[i32], sample_rate: u32) -> Result<Vec<u8>, AudioError> {
    use flacenc::component::BitRepr;
    use flacenc::error::Verify;

    let config = flacenc::config::Encoder::default()
        .into_verified()
        .map_err(|e| AudioError::Write(format!("could not set up lossless encoding: {e:?}")))?;
    let block_size = config.block_size;
    let source = flacenc::source::MemSource::from_samples(samples, 1, 16, sample_rate as usize);
    let stream = flacenc::encode_with_fixed_block_size(&config, source, block_size)
        .map_err(|e| AudioError::Write(format!("could not compress this recording: {e:?}")))?;
    let mut sink = flacenc::bitsink::ByteSink::new();
    stream
        .write(&mut sink)
        .map_err(|e| AudioError::Write(format!("could not compress this recording: {e:?}")))?;
    Ok(sink.as_slice().to_vec())
}

fn to_i16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16
}

fn write_error(e: &std::io::Error) -> AudioError {
    AudioError::Write(user_facing_write_error(e))
}

/// Plain words about what the person can do; the technical cause goes to the
/// redacted log, not the banner (mantra 2).
fn user_facing_write_error(e: &std::io::Error) -> String {
    if is_disk_full(e) {
        "This computer has run out of space, so Echo had to stop saving audio. Free some space and start again — everything recorded up to now is safe.".to_string()
    } else {
        format!("Echo could not save audio to this folder ({e}).")
    }
}

fn user_facing_write_error_str(detail: &str) -> String {
    if detail.contains("space") || detail.contains("os error 28") {
        "This computer has run out of space, so Echo had to stop saving audio. Free some space and start again — everything recorded up to now is safe.".to_string()
    } else {
        format!("Echo could not save audio to this folder ({detail}).")
    }
}

fn is_disk_full(e: &std::io::Error) -> bool {
    e.raw_os_error() == Some(28) || e.to_string().to_lowercase().contains("no space")
}

/// Is this the message Echo shows when the disk filled up?
pub fn is_out_of_space(message: &str) -> bool {
    message.contains("run out of space")
}

/// Restore state after a crash: which chunks on disk are usable, and where a
/// resumed recording should continue from.
///
/// Never deletes a committed chunk. A trailing file that cannot be read is
/// reported as absent rather than trusted, because a half-written chunk with a
/// plausible length would silently corrupt the timeline.
pub async fn recover(
    audio_dir: &Path,
    channel: Channel,
) -> Result<Vec<CommittedChunk>, AudioError> {
    let dir = audio_dir.to_path_buf();
    tokio::task::spawn_blocking(move || recover_blocking(&dir, channel))
        .await
        .map_err(|e| AudioError::Backend(e.to_string()))?
}

fn recover_blocking(audio_dir: &Path, channel: Channel) -> Result<Vec<CommittedChunk>, AudioError> {
    let prefix = format!("{}-", channel.as_str());
    let mut found: Vec<(u64, PathBuf, ChunkFormat)> = Vec::new();
    let entries = match std::fs::read_dir(audio_dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(write_error(&e)),
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.starts_with(&prefix) {
            continue;
        }
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        let Some(format) = ChunkFormat::from_extension(ext) else {
            continue;
        };
        let stem = &name[prefix.len()..name.len() - ext.len() - 1];
        let Ok(seq) = stem.parse::<u64>() else {
            continue;
        };
        found.push((seq, path, format));
    }
    found.sort_by_key(|(seq, _, _)| *seq);

    let mut out = Vec::new();
    let mut t = 0i64;
    for (seq, path, format) in found {
        let duration_ms = match chunk_duration_ms(&path, format) {
            Ok(ms) if ms > 0 => ms,
            // Trailing garbage: stop here rather than guess. Whatever follows a
            // hole cannot be placed on the timeline anyway.
            _ => break,
        };
        out.push(CommittedChunk {
            channel,
            seq,
            path,
            t_start_ms: t,
            t_end_ms: t + duration_ms,
        });
        t += duration_ms;
    }
    Ok(out)
}

fn chunk_duration_ms(path: &Path, format: ChunkFormat) -> Result<i64, AudioError> {
    match format {
        ChunkFormat::Wav => {
            let reader =
                hound::WavReader::open(path).map_err(|e| AudioError::Write(e.to_string()))?;
            let spec = reader.spec();
            let frames = reader.duration() as i64;
            if spec.sample_rate == 0 {
                return Ok(0);
            }
            Ok(frames * 1_000 / i64::from(spec.sample_rate))
        }
        ChunkFormat::Flac => Err(flac_read_unsupported()),
    }
}

/// Decode a chunk to 16 kHz mono. Used by the ASR catch-up pass.
pub async fn read_chunk(path: &Path) -> Result<Vec<f32>, AudioError> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || read_chunk_blocking(&path))
        .await
        .map_err(|e| AudioError::Backend(e.to_string()))?
}

pub fn read_chunk_blocking(path: &Path) -> Result<Vec<f32>, AudioError> {
    let format = path
        .extension()
        .and_then(|e| e.to_str())
        .and_then(ChunkFormat::from_extension)
        .ok_or_else(|| AudioError::Write(format!("{} is not a recording", path.display())))?;
    match format {
        ChunkFormat::Wav => read_wav_16k_mono(path),
        ChunkFormat::Flac => Err(flac_read_unsupported()),
    }
}

fn flac_read_unsupported() -> AudioError {
    AudioError::Backend(
        "this build cannot read back compressed recordings; enable the flacenc \"decode\" feature"
            .into(),
    )
}

/// Read any WAV as 16 kHz mono `f32`, whatever it was recorded as.
pub fn read_wav_16k_mono(path: &Path) -> Result<Vec<f32>, AudioError> {
    let mut reader = hound::WavReader::open(path).map_err(|e| AudioError::Write(e.to_string()))?;
    let spec = reader.spec();
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| AudioError::Write(e.to_string()))?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| AudioError::Write(e.to_string()))?
                .into_iter()
                .map(|s| s as f32 * scale)
                .collect()
        }
    };
    let channels = usize::from(spec.channels.max(1));
    let mono: Vec<f32> = if channels == 1 {
        raw
    } else {
        raw.chunks(channels)
            .map(|f| f.iter().sum::<f32>() / channels as f32)
            .collect()
    };
    Ok(resample_buffer(&mono, spec.sample_rate, TARGET_SAMPLE_RATE))
}

/// Write 16 kHz mono samples as one file, for the derived mixdown.
pub fn write_wav_16k_mono(path: &Path, samples: &[f32]) -> Result<u64, AudioError> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| write_error(&e))?;
    }
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: TARGET_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let file = File::create(path).map_err(|e| write_error(&e))?;
    let mut writer = hound::WavWriter::new(BufWriter::new(file), spec)
        .map_err(|e| AudioError::Write(e.to_string()))?;
    for s in samples {
        writer
            .write_sample(to_i16(*s))
            .map_err(|e| AudioError::Write(e.to_string()))?;
    }
    writer
        .finalize()
        .map_err(|e| AudioError::Write(e.to_string()))?;
    let file = File::open(path).map_err(|e| write_error(&e))?;
    file.sync_all().map_err(|e| write_error(&e))?;
    Ok(file.metadata().map(|m| m.len()).unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = TARGET_SAMPLE_RATE;

    fn frame(t_start_ms: i64, ms: usize, value: f32) -> Frame {
        Frame {
            channel: Channel::Mic,
            t_start_ms,
            samples: vec![value; RATE as usize * ms / 1_000],
        }
    }

    #[test]
    fn a_chunk_rolls_exactly_on_the_window_and_commits_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = ChunkWriter::create(dir.path(), Channel::Mic, RATE).unwrap();

        let mut committed = Vec::new();
        // 65 seconds in 20 ms frames.
        let frames = 65 * 50;
        for i in 0..frames {
            if let Some(c) = w.write(&frame(i as i64 * 20, 20, 0.25)).unwrap() {
                committed.push(c);
            }
        }
        if let Some(c) = w.finish().unwrap() {
            committed.push(c);
        }

        let window = i64::from(CHUNK_SECONDS) * 1_000;
        assert_eq!(committed.len(), 3, "{committed:#?}");
        for (i, c) in committed.iter().enumerate() {
            assert_eq!(c.seq, i as u64, "sequence numbers must not skip");
            assert_eq!(c.channel, Channel::Mic);
            assert!(c.path.exists(), "{} was not written", c.path.display());
        }
        assert_eq!(committed[0].t_start_ms, 0);
        assert_eq!(committed[0].t_end_ms, window);
        assert_eq!(committed[1].t_start_ms, window);
        assert_eq!(committed[1].t_end_ms, 2 * window);
        assert_eq!(committed[2].t_start_ms, 2 * window);
        assert_eq!(committed[2].t_end_ms, 65_000);
    }

    #[test]
    fn committed_chunks_are_contiguous_with_no_gap_and_no_overlap() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = ChunkWriter::create(dir.path(), Channel::System, RATE).unwrap();
        let mut committed = Vec::new();
        // Odd frame sizes, on purpose: the boundary must not depend on them.
        // 10 000 frames of 7 ms is 70 s, so more than two chunks have to roll.
        for i in 0..10_000 {
            if let Some(c) = w.write(&frame(i * 7, 7, 0.1)).unwrap() {
                committed.push(c);
            }
        }
        if let Some(c) = w.finish().unwrap() {
            committed.push(c);
        }
        assert!(committed.len() >= 2);
        for pair in committed.windows(2) {
            assert_eq!(
                pair[0].t_end_ms, pair[1].t_start_ms,
                "chunk {} ends at {} but {} starts at {}",
                pair[0].seq, pair[0].t_end_ms, pair[1].seq, pair[1].t_start_ms
            );
            assert!(pair[1].seq == pair[0].seq + 1);
        }
    }

    #[test]
    fn the_samples_written_come_back_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = ChunkWriter::create(dir.path(), Channel::Mic, RATE).unwrap();
        let tone: Vec<f32> = (0..RATE as usize)
            .map(|i| (i as f32 / RATE as f32 * 440.0 * std::f32::consts::TAU).sin() * 0.5)
            .collect();
        w.write_samples(&tone).unwrap();
        let chunk = w.finish().unwrap().unwrap();

        let back = read_chunk_blocking(&chunk.path).unwrap();
        assert_eq!(back.len(), tone.len());
        for (a, b) in tone.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-3, "{a} became {b}");
        }
        assert_eq!(chunk.duration_ms(), 1_000);
    }

    #[test]
    fn audio_lost_upstream_is_filled_in_so_the_timeline_does_not_slide() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = ChunkWriter::create(dir.path(), Channel::Mic, RATE).unwrap();
        // One second of audio, then a frame that says it belongs two seconds in:
        // a whole second went missing between them.
        w.write_frame(&frame(0, 1_000, 0.5)).unwrap();
        w.write_frame(&frame(2_000, 500, 0.5)).unwrap();
        let chunk = w.finish().unwrap().unwrap();

        assert_eq!(
            chunk.t_end_ms, 2_500,
            "the file must end where the meeting clock says"
        );
        let back = read_chunk_blocking(&chunk.path).unwrap();
        assert_eq!(back.len(), RATE as usize * 5 / 2);
        // The missing second reads back as silence, in the right place.
        assert!((back[RATE as usize / 2] - 0.5).abs() < 1e-2);
        assert_eq!(back[RATE as usize + 100], 0.0);
        assert!((back[RATE as usize * 2 + 100] - 0.5).abs() < 1e-2);
    }

    #[test]
    fn a_long_outage_becomes_a_gap_rather_than_hours_of_silence() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = ChunkWriter::create(dir.path(), Channel::Mic, RATE).unwrap();
        let mut committed = w.write_frame(&frame(0, 1_000, 0.4)).unwrap();
        // The device came back five minutes later.
        committed.extend(w.write_frame(&frame(300_000, 1_000, 0.4)).unwrap());
        if let Some(last) = w.finish().unwrap() {
            committed.push(last);
        }

        assert_eq!(committed.len(), 2, "{committed:#?}");
        assert_eq!(committed[0].t_start_ms, 0);
        assert_eq!(committed[0].t_end_ms, 1_000);
        assert_eq!(
            committed[1].t_start_ms, 300_000,
            "placed at the real offset"
        );
        assert_eq!(committed[1].t_end_ms, 301_000);
        // Two seconds of audio on disk, not five minutes of silence.
        let bytes: u64 = committed
            .iter()
            .map(|c| std::fs::metadata(&c.path).unwrap().len())
            .sum();
        assert!(bytes < 200_000, "{bytes} bytes is a padded outage");
    }

    #[test]
    fn a_frame_that_overlaps_what_is_saved_does_not_double_up() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = ChunkWriter::create(dir.path(), Channel::Mic, RATE).unwrap();
        w.write_frame(&frame(0, 1_000, 0.5)).unwrap();
        // Half of this frame is audio already on disk.
        w.write_frame(&frame(500, 1_000, 0.5)).unwrap();
        let chunk = w.finish().unwrap().unwrap();
        assert_eq!(chunk.t_end_ms, 1_500);
    }

    #[test]
    fn padding_across_a_chunk_boundary_reports_every_file_it_closed() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = ChunkWriter::create(dir.path(), Channel::Mic, RATE).unwrap();
        w.write_frame(&frame(0, 1_000, 0.3)).unwrap();
        // A 45 s hole spans a chunk roll; both files have to be reported so both
        // get a journal row (mantra 3).
        let committed = w.write_frame(&frame(46_000, 20, 0.3)).unwrap();
        assert_eq!(committed.len(), 1, "{committed:#?}");
        assert_eq!(committed[0].t_start_ms, 0);
        assert_eq!(committed[0].t_end_ms, 30_000);
        assert_eq!(w.written_to_ms(), 46_020);
    }

    #[test]
    fn nothing_written_leaves_nothing_behind() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = ChunkWriter::create(dir.path(), Channel::Mic, RATE).unwrap();
        assert_eq!(w.flush().unwrap(), None);
        assert_eq!(w.write_samples(&[]).unwrap(), None);
        assert_eq!(w.finish().unwrap(), None);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn dropping_a_writer_still_commits_the_open_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let path = {
            let mut w = ChunkWriter::create(dir.path(), Channel::Mic, RATE).unwrap();
            w.write_samples(&vec![0.2; RATE as usize]).unwrap();
            dir.path()
                .join(chunk_file_name(Channel::Mic, 0, DEFAULT_CHUNK_FORMAT))
        };
        assert!(path.exists(), "the open chunk was lost on drop");
        let back = read_chunk_blocking(&path).unwrap();
        assert_eq!(back.len(), RATE as usize);
    }

    #[test]
    fn recovery_reads_the_chunks_a_crash_left_behind() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = ChunkWriter::create(dir.path(), Channel::Mic, RATE).unwrap();
        for _ in 0..70 {
            w.write_samples(&vec![0.1; RATE as usize]).unwrap();
        }
        drop(w);

        let found = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(recover(dir.path(), Channel::Mic))
            .unwrap();
        assert_eq!(found.len(), 3, "{found:#?}");
        assert_eq!(found[0].t_start_ms, 0);
        assert_eq!(found[2].t_end_ms, 70_000);
        // A different channel's journal is independent.
        let none = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(recover(dir.path(), Channel::System))
            .unwrap();
        assert!(none.is_empty());
    }

    #[test]
    fn recovery_stops_at_a_file_it_cannot_trust() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = ChunkWriter::create(dir.path(), Channel::Mic, RATE).unwrap();
        for _ in 0..35 {
            w.write_samples(&vec![0.1; RATE as usize]).unwrap();
        }
        drop(w);
        // Simulate a chunk that was being written when the power went out.
        std::fs::write(
            dir.path()
                .join(chunk_file_name(Channel::Mic, 5, ChunkFormat::Wav)),
            b"not a recording",
        )
        .unwrap();

        let found = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(recover(dir.path(), Channel::Mic))
            .unwrap();
        assert_eq!(found.len(), 2);
        assert_eq!(found[1].t_end_ms, 35_000);
    }

    #[test]
    fn a_resumed_recording_carries_on_from_where_it_stopped() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = ChunkWriter::create(dir.path(), Channel::Mic, RATE).unwrap();
        w.resume_at(7, 210_000);
        w.write_samples(&vec![0.1; RATE as usize]).unwrap();
        let chunk = w.finish().unwrap().unwrap();
        assert_eq!(chunk.seq, 7);
        assert_eq!(chunk.t_start_ms, 210_000);
        assert_eq!(chunk.t_end_ms, 211_000);
    }

    #[test]
    fn lossless_chunks_can_be_written_even_though_this_build_cannot_read_them() {
        let dir = tempfile::tempdir().unwrap();
        let mut w =
            ChunkWriter::create_with_format(dir.path(), Channel::Mic, RATE, ChunkFormat::Flac)
                .unwrap();
        w.write_samples(&vec![0.3; RATE as usize / 2]).unwrap();
        let chunk = w.finish().unwrap().unwrap();
        assert!(chunk.path.exists());
        assert_eq!(chunk.path.extension().unwrap(), "flac");
        let head = std::fs::read(&chunk.path).unwrap();
        assert_eq!(&head[..4], b"fLaC", "that is not a lossless file");
        assert!(read_chunk_blocking(&chunk.path).is_err());
    }

    #[test]
    fn a_writer_that_lost_the_disk_refuses_further_writes_and_says_so_plainly() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = ChunkWriter::create(dir.path(), Channel::Mic, RATE).unwrap();
        w.write_samples(&vec![0.1; 1_000]).unwrap();
        // Stand in for a full disk.
        w.fatal = Some(user_facing_write_error(&std::io::Error::from_raw_os_error(
            28,
        )));
        let err = w.write_samples(&vec![0.1; 1_000]).unwrap_err();
        let message = err.to_string();
        assert!(is_out_of_space(&message), "{message}");
        assert!(
            !message.contains("errno") && !message.contains("os error"),
            "the person should not read errno: {message}"
        );
    }

    #[test]
    fn a_mixdown_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mixed.wav");
        let samples: Vec<f32> = (0..8_000).map(|i| (i % 100) as f32 / 200.0).collect();
        let bytes = write_wav_16k_mono(&path, &samples).unwrap();
        assert!(bytes > 16_000);
        let back = read_wav_16k_mono(&path).unwrap();
        assert_eq!(back.len(), samples.len());
    }
}
