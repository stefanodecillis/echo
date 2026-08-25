//! What the models already worked out about a meeting, kept on disk so a
//! recording does not throw it away.
//!
//! ## Why this exists
//!
//! The speaker pass costs about twelve minutes of ONNX arithmetic on a
//! seventy-six minute meeting, and a recording starting outranks it absolutely
//! (mantra 1). Until now "stepping aside" meant starting over from the first
//! window afterwards — so on a day of back-to-back meetings with gaps shorter
//! than the work takes, a meeting could be parked and restarted for ever and
//! never get its speakers at all. Nothing was lost, but nothing was gained
//! either.
//!
//! So [`super::pipeline::scan`] writes down where it got to. The park still
//! happens at the next checkpoint, and the pass still reads only finished audio
//! off disk — it just picks up at the window it stopped on instead of at zero.
//!
//! ## Why a stale one would be worse than none
//!
//! The thing kept here is a set of voice fingerprints tied to stretches of the
//! meeting clock. Replaying yesterday's fingerprints over audio that has since
//! changed — a chunk recovered after a crash, a different fingerprint network —
//! would not fail loudly: it would quietly attribute lines to the wrong person,
//! which is the one output nobody can check without listening to the whole
//! meeting again. So the rule here is: **any doubt at all means scan again.**
//!
//! [`Key`] is compared whole and must match exactly. It carries:
//!
//! * `format` — this module's own version, bumped whenever the meaning of
//!   anything below changes, so an old file is never read with new rules;
//! * `source` — which channel was scanned, because the same meeting scanned on
//!   the microphone means something completely different from the same meeting
//!   scanned on the system channel;
//! * `audio` — a digest of every committed chunk row of that channel: id,
//!   order, path and the milliseconds it covers. A chunk arriving late, a
//!   recovery renaming one, a re-recording — all change it;
//! * `total_ms` — the length the walk was planned against;
//! * `window_ms` / `step_ms` — the geometry of the sliding window, so a change
//!   to either invalidates every window recorded under the old one;
//! * `segmenter` / `embedder` — the size and modification time of the two model
//!   files, so a re-download or a switch of weights invalidates the
//!   fingerprints they produced;
//! * `embedder_tag` — which network the database says the embedder is, the same
//!   identity enrolled voice prints are stored against.
//!
//! **The transcript is deliberately not in the key.** The scan reads audio and
//! nothing else: no segment, no revision, no speaker row is looked at between
//! the first window and the last. What the transcript revision does affect is
//! the *cheap* half — clustering, cutting lines that hold two voices, writing
//! the rows — and that half is read fresh from the database on every run,
//! including a run that resumed. Keying on the revision would only mean that
//! every re-run after a correction paid the twelve minutes again for an answer
//! it already had, and the pass bumps the revision itself, so it would throw
//! its own work away every time.
//!
//! Beyond the key, a file that cannot be read, cannot be parsed, or does not
//! hold together internally ([`Cached::is_coherent`]) is treated exactly like a
//! file that was not there. Never a failure: the pass simply does the work.
//!
//! ## How long it stays
//!
//! Until the meeting goes. It is kept after the pass finishes on purpose: the
//! person correcting "there were four of us" re-runs the whole pass, and this is
//! the difference between that taking a moment and taking twelve minutes for an
//! answer the models already gave. It costs a few megabytes beside a recording
//! many times that size, and because it lives in the meeting's own directory,
//! deleting the meeting — or just its audio — takes the fingerprints with it,
//! which is the only correct answer when somebody asks Echo to forget a
//! recording.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::cluster::ClusterItem;
use super::pipeline::WindowResult;
use super::timeline::Span;

/// Version of everything in this file. Bump it and every cache written by an
/// older Echo is ignored rather than reinterpreted.
pub const FORMAT: u32 = 1;

/// The file, next to the meeting's audio so it is deleted with the meeting and
/// never outlives what it describes.
const FILE_NAME: &str = "speaker-scan.json";

/// Written next to the real file and renamed over it, so a crash or a full disk
/// mid-write leaves the previous cache intact rather than a half-written one.
const PART_NAME: &str = "speaker-scan.json.part";

/// Everything that has to be identical for a kept scan to still be true of this
/// meeting. Compared whole; any difference at all means a full rescan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Key {
    pub format: u32,
    pub source: String,
    pub audio: String,
    pub total_ms: i64,
    pub window_ms: i64,
    pub step_ms: i64,
    pub segmenter: String,
    pub embedder: String,
    pub embedder_tag: String,
}

/// How far the window walk got, and everything it found on the way.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cached {
    pub key: Key,
    /// The first window that has **not** been analysed. At or past `total_ms`
    /// the scan is finished and the resuming pass goes straight to clustering.
    pub next_start_ms: i64,
    pub items: Vec<ClusterItem>,
    pub windows: Vec<WindowResult>,
    pub covered: Vec<Span>,
}

impl Cached {
    /// Whether this file describes something the pass could actually resume
    /// from.
    ///
    /// A matching key says the *inputs* are the same; this says the *contents*
    /// are not nonsense. Every check here guards an index the pass would
    /// otherwise follow into somebody else's fingerprint — a file truncated by
    /// a full disk, or written by a build whose format number somebody forgot
    /// to bump. Cheap enough to run on every load, and a `false` costs one
    /// rescan, never a wrong label.
    pub fn is_coherent(&self) -> bool {
        if self.next_start_ms < 0 {
            return false;
        }
        // Fingerprints have to be comparable with each other: same non-zero
        // length, or the clustering has nothing to measure.
        let dim = self.items.first().map(|i| i.embedding.len()).unwrap_or(0);
        if self.items.iter().any(|i| i.embedding.len() != dim)
            || (dim == 0 && !self.items.is_empty())
        {
            return false;
        }
        let mut previous_tracks: Option<usize> = None;
        let mut last_start = i64::MIN;
        for window in &self.windows {
            let n = window.tracks.len();
            if window.confidence.len() != n
                || window.fingerprint.len() != n
                || window.continues.len() != n
            {
                return false;
            }
            // Windows are walked forward and lined up with the one before, so
            // an out-of-order file would align the wrong two windows.
            if window.start_ms < last_start {
                return false;
            }
            last_start = window.start_ms;
            if window
                .fingerprint
                .iter()
                .flatten()
                .any(|i| *i >= self.items.len())
            {
                return false;
            }
            // "Continues local speaker N of the previous window" — N has to be
            // one the previous window actually had.
            let previous = previous_tracks.unwrap_or(0);
            if window.continues.iter().flatten().any(|i| *i >= previous) {
                return false;
            }
            previous_tracks = Some(n);
        }
        self.covered.iter().all(|(from, to)| from <= to)
    }
}

/// Where this meeting's cache lives, given the directory its audio is in.
pub fn path_in(audio_dir: &Path) -> PathBuf {
    audio_dir.join(FILE_NAME)
}

/// Identity of a model file: its name, its size and when it was last written.
///
/// `None` when the file cannot be stat'ed at all, and a `None` switches the
/// cache off entirely for this run — neither read nor written. A key that
/// cannot say which weights produced a fingerprint is a key that cannot keep
/// its promise, and the pass falling back to twelve minutes of honest work is
/// the right way to be wrong.
pub fn model_stamp(path: &Path) -> Option<String> {
    let meta = std::fs::metadata(path).ok()?;
    let modified = meta
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?;
    let name = path.file_name()?.to_string_lossy().to_string();
    Some(format!("{name}:{}:{}", meta.len(), modified.as_nanos()))
}

/// Read a kept scan back, or `None` for every reason there is not one: no file,
/// an unreadable one, one written under different inputs, one that does not
/// hold together. The caller does the work again in every one of those cases.
pub async fn load(audio_dir: &Path, key: &Key) -> Option<Cached> {
    let path = path_in(audio_dir);
    let bytes = tokio::fs::read(&path).await.ok()?;
    let cached: Cached = match serde_json::from_slice(&bytes) {
        Ok(cached) => cached,
        Err(error) => {
            tracing::debug!(%error, "the kept speaker scan could not be read; scanning again");
            return None;
        }
    };
    if &cached.key != key {
        tracing::debug!("this meeting's audio or models moved on; scanning again");
        return None;
    }
    if !cached.is_coherent() {
        tracing::warn!("the kept speaker scan does not hold together; scanning again");
        return None;
    }
    Some(cached)
}

/// Write the scan down, atomically. Best effort: a cache that cannot be written
/// costs time on a later run and nothing else, so it never fails the pass.
pub async fn store(audio_dir: &Path, cached: &Cached) {
    if let Err(error) = write_atomically(audio_dir, cached).await {
        tracing::debug!(%error, "could not keep the speaker scan for later");
    }
}

async fn write_atomically<T: Serialize>(audio_dir: &Path, value: &T) -> std::io::Result<()> {
    let bytes = serde_json::to_vec(value)?;
    let part = audio_dir.join(PART_NAME);
    tokio::fs::write(&part, &bytes).await?;
    tokio::fs::rename(&part, path_in(audio_dir)).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> Key {
        Key {
            format: FORMAT,
            source: "system".into(),
            audio: "digest".into(),
            total_ms: 30_000,
            window_ms: 10_000,
            step_ms: 5_000,
            segmenter: "seg:1:2".into(),
            embedder: "emb:3:4".into(),
            embedder_tag: "asset-1".into(),
        }
    }

    fn cached() -> Cached {
        Cached {
            key: key(),
            next_start_ms: 10_000,
            items: vec![ClusterItem {
                embedding: vec![1.0, 0.0],
                weight_ms: 4_000,
            }],
            windows: vec![WindowResult {
                start_ms: 0,
                tracks: vec![vec![(0, 4_000)]],
                confidence: vec![0.9],
                fingerprint: vec![Some(0)],
                continues: vec![None],
            }],
            covered: vec![(0, 30_000)],
        }
    }

    #[tokio::test]
    async fn a_scan_written_under_the_same_inputs_comes_back() {
        let dir = tempfile::tempdir().unwrap();
        store(dir.path(), &cached()).await;
        let back = load(dir.path(), &key()).await.expect("the kept scan");
        assert_eq!(back.next_start_ms, 10_000);
        assert_eq!(back.items.len(), 1);
        assert_eq!(back.windows.len(), 1);
    }

    #[tokio::test]
    async fn nothing_written_is_simply_nothing_to_resume_from() {
        let dir = tempfile::tempdir().unwrap();
        assert!(load(dir.path(), &key()).await.is_none());
    }

    /// The whole safety argument in one test: every single thing the key names
    /// invalidates it on its own. A stale fingerprint set is wrong speaker
    /// labels, so this is the list that must not quietly shrink.
    #[tokio::test]
    async fn every_input_the_key_names_invalidates_it_on_its_own() {
        let dir = tempfile::tempdir().unwrap();
        store(dir.path(), &cached()).await;

        /// One way the world can move on from a kept scan.
        type Mutation = (&'static str, Box<dyn Fn(&mut Key)>);

        let mutations: Vec<Mutation> = vec![
            ("format", Box::new(|k: &mut Key| k.format += 1)),
            (
                "channel",
                Box::new(|k: &mut Key| k.source = "microphone-only".into()),
            ),
            ("audio", Box::new(|k: &mut Key| k.audio = "other".into())),
            ("length", Box::new(|k: &mut Key| k.total_ms += 1_000)),
            ("window", Box::new(|k: &mut Key| k.window_ms += 1)),
            ("step", Box::new(|k: &mut Key| k.step_ms += 1)),
            (
                "segmenter",
                Box::new(|k: &mut Key| k.segmenter = "seg:1:9".into()),
            ),
            (
                "embedder",
                Box::new(|k: &mut Key| k.embedder = "emb:3:9".into()),
            ),
            (
                "network",
                Box::new(|k: &mut Key| k.embedder_tag = "asset-2".into()),
            ),
        ];
        for (what, mutate) in mutations {
            let mut changed = key();
            mutate(&mut changed);
            assert!(
                load(dir.path(), &changed).await.is_none(),
                "a different {what} has to mean a full rescan"
            );
        }
        // And the untouched key still matches, so the test above is about the
        // mutations and not about the file being unreadable.
        assert!(load(dir.path(), &key()).await.is_some());
    }

    #[tokio::test]
    async fn a_half_written_file_is_not_resumed_from() {
        let dir = tempfile::tempdir().unwrap();
        store(dir.path(), &cached()).await;
        let path = path_in(dir.path());
        let mut bytes = tokio::fs::read(&path).await.unwrap();
        bytes.truncate(bytes.len() / 2);
        tokio::fs::write(&path, &bytes).await.unwrap();
        assert!(load(dir.path(), &key()).await.is_none());
    }

    #[test]
    fn a_fingerprint_index_pointing_past_the_fingerprints_is_incoherent() {
        let mut c = cached();
        c.windows[0].fingerprint = vec![Some(7)];
        assert!(!c.is_coherent());
    }

    #[test]
    fn a_window_that_continues_a_speaker_the_last_one_never_had_is_incoherent() {
        let mut c = cached();
        c.windows.push(WindowResult {
            start_ms: 5_000,
            tracks: vec![vec![(5_000, 9_000)]],
            confidence: vec![0.5],
            fingerprint: vec![None],
            continues: vec![Some(3)],
        });
        assert!(!c.is_coherent());
    }

    #[test]
    fn windows_out_of_order_are_incoherent() {
        let mut c = cached();
        c.windows.push(WindowResult {
            start_ms: -5_000,
            ..Default::default()
        });
        assert!(!c.is_coherent());
    }

    #[test]
    fn fingerprints_of_different_lengths_cannot_be_compared_so_are_incoherent() {
        let mut c = cached();
        c.items.push(ClusterItem {
            embedding: vec![1.0, 0.0, 0.0],
            weight_ms: 1_000,
        });
        assert!(!c.is_coherent());
    }

    #[test]
    fn a_model_that_is_not_there_has_no_stamp_so_the_cache_stays_off() {
        assert!(model_stamp(Path::new("/nonexistent/segmenter.onnx")).is_none());
    }
}
