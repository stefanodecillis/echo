//! Reading a finished meeting's audio back off disk, one window at a time.
//!
//! Raw per-channel audio is the source of truth (mantra 3), which for this pass
//! means: never load a whole meeting into memory. A two-hour recording is about
//! 460 MB of 16 kHz floats, and the speaker pass runs in the background while
//! the person is doing something else.
//!
//! So the pass walks forward in windows and this reader keeps only the couple of
//! chunks a window touches. Because windows advance monotonically, each chunk on
//! disk is decoded once or twice for the entire pass.
//!
//! Only **committed** chunks are read. An uncommitted chunk may be a torn file
//! from a crash, and the journal is what says which is which
//! (DESIGN §3 Crash recovery).

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;

use crate::audio::{self, TARGET_SAMPLE_RATE};
use crate::types::AudioChunk;

use super::timeline::{self, Span};
use super::DiarizeError;

/// How many decoded chunks to hold. Two is enough for a window that straddles a
/// chunk boundary; three gives the seam some slack.
const CACHE_CHUNKS: usize = 3;

/// Samples per millisecond at the pipeline's sample rate.
const PER_MS: i64 = TARGET_SAMPLE_RATE as i64 / 1_000;

/// A meeting channel's audio, addressable by time.
pub struct ChunkPcm {
    chunks: Vec<AudioChunk>,
    cache: VecDeque<(String, Arc<Vec<f32>>)>,
    total_ms: i64,
}

impl std::fmt::Debug for ChunkPcm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChunkPcm")
            .field("chunks", &self.chunks.len())
            .field("total_ms", &self.total_ms)
            .finish()
    }
}

impl ChunkPcm {
    /// Build a reader over the committed chunks of one channel.
    pub fn new(chunks: Vec<AudioChunk>) -> Self {
        let mut chunks: Vec<AudioChunk> = chunks
            .into_iter()
            .filter(|c| c.committed && c.t_end_ms > c.t_start_ms)
            .collect();
        chunks.sort_by_key(|c| (c.t_start_ms, c.seq));
        let total_ms = chunks.iter().map(|c| c.t_end_ms).max().unwrap_or(0);
        Self {
            chunks,
            cache: VecDeque::with_capacity(CACHE_CHUNKS),
            total_ms,
        }
    }

    /// Where the audio on this channel ends, on the meeting clock.
    pub fn total_ms(&self) -> i64 {
        self.total_ms
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// A short, stable digest of exactly which audio this reader would read.
    ///
    /// Every committed chunk's identity, order, file and stretch of the meeting
    /// clock. It is what lets a kept scan be trusted: if any of that has moved —
    /// a chunk recovered after a crash, a late commit, a re-recorded meeting —
    /// the digest changes and the work is done again rather than replayed over
    /// audio it was not computed from (see [`super::scan_cache`]).
    pub fn stamp(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        for c in &self.chunks {
            hasher.update(c.id.as_bytes());
            hasher.update(b"\0");
            hasher.update(c.path.as_bytes());
            hasher.update(b"\0");
            hasher.update(c.seq.to_le_bytes());
            hasher.update(c.t_start_ms.to_le_bytes());
            hasher.update(c.t_end_ms.to_le_bytes());
        }
        format!("{:x}", hasher.finalize())
    }

    /// Stretches of the clock that actually have audio behind them, merged.
    ///
    /// A recording that lost a device mid-meeting has holes; running the model
    /// over a hole wastes time and can only invent speakers.
    pub fn covered(&self) -> Vec<Span> {
        let mut spans: Vec<Span> = self
            .chunks
            .iter()
            .map(|c| (c.t_start_ms, c.t_end_ms))
            .collect();
        timeline::merge_spans(&mut spans, 0);
        spans
    }

    /// Read `len_ms` of audio starting at `from_ms`, zero-filled where the disk
    /// has nothing. The result is always exactly `len_ms` long, so callers do
    /// not special-case the tail of the meeting.
    pub async fn window(&mut self, from_ms: i64, len_ms: i64) -> Result<Vec<f32>, DiarizeError> {
        let len = (len_ms.max(0) * PER_MS) as usize;
        let mut out = vec![0.0f32; len];
        let to_ms = from_ms + len_ms;

        let touching: Vec<AudioChunk> = self
            .chunks
            .iter()
            .filter(|c| c.t_start_ms < to_ms && c.t_end_ms > from_ms)
            .cloned()
            .collect();

        for chunk in touching {
            let samples = self.decode(&chunk).await?;
            // Where this chunk's first sample lands in the output.
            let base = (chunk.t_start_ms - from_ms) * PER_MS;
            for (i, x) in samples.iter().enumerate() {
                let pos = base + i as i64;
                if pos < 0 {
                    continue;
                }
                let pos = pos as usize;
                if pos >= len {
                    break;
                }
                out[pos] = *x;
            }
        }
        Ok(out)
    }

    async fn decode(&mut self, chunk: &AudioChunk) -> Result<Arc<Vec<f32>>, DiarizeError> {
        if let Some(hit) = self
            .cache
            .iter()
            .find(|(path, _)| path == &chunk.path)
            .map(|(_, s)| Arc::clone(s))
        {
            return Ok(hit);
        }
        let samples = audio::writer::read_chunk(Path::new(&chunk.path))
            .await
            .map_err(|e| DiarizeError::Failed(format!("could not read {}: {e}", chunk.path)))?;
        let samples = Arc::new(samples);
        if self.cache.len() >= CACHE_CHUNKS {
            self.cache.pop_front();
        }
        self.cache
            .push_back((chunk.path.clone(), Arc::clone(&samples)));
        Ok(samples)
    }

    /// Drop every decoded chunk. Called between phases so nothing sits in
    /// memory once the pass no longer needs audio (mantra 1).
    pub fn release(&mut self) {
        self.cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Channel;

    fn chunk(seq: i64, from: i64, to: i64, committed: bool) -> AudioChunk {
        AudioChunk {
            id: format!("c{seq}"),
            meeting_id: "m".into(),
            channel: Channel::System,
            seq,
            path: format!("/nonexistent/{seq}.flac"),
            t_start_ms: from,
            t_end_ms: to,
            committed,
        }
    }

    #[test]
    fn only_committed_chunks_count_as_audio_on_disk() {
        let pcm = ChunkPcm::new(vec![
            chunk(0, 0, 30_000, true),
            chunk(1, 30_000, 60_000, false),
        ]);
        assert_eq!(pcm.total_ms(), 30_000);
        assert_eq!(pcm.covered(), vec![(0, 30_000)]);
    }

    #[test]
    fn chunks_arrive_in_time_order_however_they_were_listed() {
        let pcm = ChunkPcm::new(vec![
            chunk(2, 60_000, 90_000, true),
            chunk(0, 0, 30_000, true),
            chunk(1, 30_000, 60_000, true),
        ]);
        assert_eq!(pcm.total_ms(), 90_000);
        // Contiguous chunks merge into one covered stretch.
        assert_eq!(pcm.covered(), vec![(0, 90_000)]);
    }

    #[test]
    fn a_gap_in_the_recording_shows_up_as_two_covered_stretches() {
        let pcm = ChunkPcm::new(vec![
            chunk(0, 0, 30_000, true),
            chunk(5, 120_000, 150_000, true),
        ]);
        assert_eq!(pcm.covered(), vec![(0, 30_000), (120_000, 150_000)]);
    }

    #[test]
    fn a_meeting_with_nothing_on_this_channel_is_empty() {
        let pcm = ChunkPcm::new(Vec::new());
        assert!(pcm.is_empty());
        assert_eq!(pcm.total_ms(), 0);
        assert!(pcm.covered().is_empty());
    }

    #[tokio::test]
    async fn a_window_over_empty_time_is_silence_of_the_right_length() {
        let mut pcm = ChunkPcm::new(Vec::new());
        let w = pcm.window(0, 10_000).await.unwrap();
        assert_eq!(w.len(), 160_000);
        assert!(w.iter().all(|x| *x == 0.0));
    }

    #[tokio::test]
    async fn a_window_that_touches_no_chunk_never_reads_the_disk() {
        // The paths do not exist, so reaching for them would error.
        let mut pcm = ChunkPcm::new(vec![chunk(0, 0, 30_000, true)]);
        let w = pcm.window(60_000, 10_000).await.unwrap();
        assert_eq!(w.len(), 160_000);
        assert!(w.iter().all(|x| *x == 0.0));
    }

    #[tokio::test]
    async fn a_missing_chunk_file_is_reported_rather_than_ignored() {
        let mut pcm = ChunkPcm::new(vec![chunk(0, 0, 30_000, true)]);
        let err = pcm.window(0, 10_000).await.unwrap_err();
        assert!(matches!(err, DiarizeError::Failed(_)), "{err:?}");
    }
}
