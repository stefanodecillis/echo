//! The signed catalog: every file Echo downloads, and the quality presets that
//! group them.
//!
//! IMPLEMENTED-BY: asr agent (M3).
//!
//! This is data, not behaviour, so it lives in one place and is exhaustively
//! tested. Rules that the tests enforce:
//!
//! * Every entry records **provenance and licence** (DESIGN §3 `models`, review
//!   finding 11): where the bytes come from, which upstream revision, and under
//!   what terms we may ship them.
//! * Every entry carries an expected length and, where upstream publishes one, a
//!   SHA-256. When it does not, the downloader trusts the first copy it fetches
//!   and writes the hash it computed into the `models` table, so a later
//!   verification still means something.
//! * URLs are pinned to an immutable upstream revision (a Hugging Face commit,
//!   a git tag, a GitHub release), never to a moving branch. A catalog update
//!   may add an entry or correct a hash; it must never repoint an installed
//!   entry at different bytes.
//! * The presets are **user-facing**, so their names and descriptions contain
//!   no jargon at all (mantra 2). Model identifiers live in
//!   [`CatalogEntry::name`], which only Settings → Advanced renders.
//!
//! "Signed catalog" (DESIGN §3) is satisfied by construction rather than by a
//! detached signature: the catalog is compiled into the binary, and the binary is
//! the thing that is signed and notarized. Nothing is read from the network to
//! decide what to download, so there is no unsigned channel to attack. Shipping
//! a catalog that can update itself between releases would need a real signature
//! check, and would be a change to this comment as much as to the code.

use crate::types::AssetKind;

/// Bumped whenever this file changes. Stored in settings so a newer build can
/// notice it has a newer catalog than the database.
pub const CATALOG_REVISION: &str = "2026-08-19.1";

/// Which platforms need an entry at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    /// Everyone downloads this.
    Any,
    /// Apple silicon only: the encoder companion whisper.cpp picks up next to
    /// the main file. Missing it costs speed, never correctness.
    MacOs,
}

/// What arrives over the wire, and what has to happen to it afterwards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Archive {
    /// The download *is* the installed file.
    None,
    /// A zip holding one directory bundle, unpacked next to the speech file.
    /// `dir_name` is the directory the zip contains at its top level; the
    /// installed path is that directory, and the zip is deleted afterwards.
    ZipBundle { dir_name: &'static str },
}

/// One row of the catalog, and eventually one row of the `models` table.
#[derive(Debug, Clone, Copy)]
pub struct CatalogEntry {
    /// Stable id. Also the primary key in `models`, so it must never be reused
    /// for different bytes.
    pub id: &'static str,
    pub kind: AssetKind,
    /// Technical identifier. Settings → Advanced only.
    pub name: &'static str,
    /// File name under `paths.assets_dir`. For a speech file this is the name
    /// whisper.cpp derives its Apple encoder companion from, so the two must
    /// stay in step.
    pub file_name: &'static str,
    pub url: &'static str,
    /// Expected SHA-256, lowercase hex. `None` means trust-on-first-use.
    pub sha256: Option<&'static str>,
    /// Expected length in bytes. 0 means unknown.
    pub bytes: i64,
    /// Licence of the bytes we distribute a downloader for.
    pub license: &'static str,
    /// Immutable upstream revision the URL is pinned to.
    pub revision: &'static str,
    /// Where this came from and how it was produced. Diagnostics only.
    pub provenance: &'static str,
    pub archive: Archive,
    pub platform: Platform,
}

impl CatalogEntry {
    /// Is this entry needed on the platform we are running on?
    pub fn applies_here(&self) -> bool {
        match self.platform {
            Platform::Any => true,
            Platform::MacOs => cfg!(target_os = "macos"),
        }
    }

    /// Name of the thing that ends up on disk: the file, or the unpacked
    /// directory bundle.
    pub fn installed_name(&self) -> &'static str {
        match self.archive {
            Archive::None => self.file_name,
            Archive::ZipBundle { dir_name } => dir_name,
        }
    }

    /// Directory bundles cannot be re-hashed after unpacking, so verification
    /// checks they exist instead.
    pub fn is_bundle(&self) -> bool {
        matches!(self.archive, Archive::ZipBundle { .. })
    }
}

// ---------------------------------------------------------------------------
// Speech files. whisper.cpp GGML conversions, published by the whisper.cpp
// project on Hugging Face and pinned to one commit of that repository.
// ---------------------------------------------------------------------------

const WHISPER_REPO_REV: &str = "5359861c739e955e79d9a303bcbc70fb988958b1";

const WHISPER_LICENSE: &str =
    "MIT — OpenAI Whisper weights (MIT), GGML conversion by the whisper.cpp project (MIT)";

const WHISPER_PROVENANCE: &str =
    "huggingface.co/ggerganov/whisper.cpp, GGML conversion published by the whisper.cpp project";

const COREML_PROVENANCE: &str = concat!(
    "huggingface.co/ggerganov/whisper.cpp — Core ML encoder companion. ",
    "whisper.cpp loads it automatically when it sits next to the speech file ",
    "under the matching -encoder.mlmodelc name; it accelerates the encoder only."
);

/// Asset ids, so the presets and the rest of the module never spell a string
/// twice.
pub mod ids {
    pub const SPEECH_EVERYDAY: &str = "speech-large-v3-turbo";
    pub const SPEECH_FASTER: &str = "speech-small";
    pub const SPEECH_FASTEST: &str = "speech-tiny";

    pub const ACCEL_EVERYDAY: &str = "speech-accelerator-large-v3-turbo";
    pub const ACCEL_FASTER: &str = "speech-accelerator-small";
    pub const ACCEL_FASTEST: &str = "speech-accelerator-tiny";

    pub const DETECTOR: &str = "speech-detector-silero-v5";
    pub const SEGMENTER: &str = "speaker-segmenter-pyannote-3";
    pub const EMBEDDER: &str = "speaker-embedder-wespeaker-campplus";
}

/// Everything every preset needs: speech detection and the two speaker files.
/// Small enough that we never make the person choose.
pub const SHARED_ASSET_IDS: &[&str] = &[ids::DETECTOR, ids::SEGMENTER, ids::EMBEDDER];

/// The catalog. Order matters only for display.
pub const CATALOG: &[CatalogEntry] = &[
    // --- speech -----------------------------------------------------------
    CatalogEntry {
        id: ids::SPEECH_EVERYDAY,
        kind: AssetKind::Speech,
        name: "whisper large-v3-turbo (ggml)",
        file_name: "ggml-large-v3-turbo.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-large-v3-turbo.bin",
        sha256: Some("1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69"),
        bytes: 1_624_555_275,
        license: WHISPER_LICENSE,
        revision: WHISPER_REPO_REV,
        provenance: WHISPER_PROVENANCE,
        archive: Archive::None,
        platform: Platform::Any,
    },
    CatalogEntry {
        id: ids::SPEECH_FASTER,
        kind: AssetKind::Speech,
        name: "whisper small (ggml)",
        file_name: "ggml-small.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-small.bin",
        sha256: Some("1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b"),
        bytes: 487_601_967,
        license: WHISPER_LICENSE,
        revision: WHISPER_REPO_REV,
        provenance: WHISPER_PROVENANCE,
        archive: Archive::None,
        platform: Platform::Any,
    },
    CatalogEntry {
        id: ids::SPEECH_FASTEST,
        kind: AssetKind::Speech,
        name: "whisper tiny (ggml)",
        file_name: "ggml-tiny.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-tiny.bin",
        sha256: Some("be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21"),
        bytes: 77_691_713,
        license: WHISPER_LICENSE,
        revision: WHISPER_REPO_REV,
        provenance: WHISPER_PROVENANCE,
        archive: Archive::None,
        platform: Platform::Any,
    },
    // --- Apple encoder companions ----------------------------------------
    CatalogEntry {
        id: ids::ACCEL_EVERYDAY,
        kind: AssetKind::SpeechAccelerator,
        name: "whisper large-v3-turbo Core ML encoder",
        file_name: "ggml-large-v3-turbo-encoder.mlmodelc.zip",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-large-v3-turbo-encoder.mlmodelc.zip",
        sha256: Some("84bedfe895bd7b5de6e8e89a0803dfc5addf8c0c5bc4c937451716bf7cf7988a"),
        bytes: 1_173_393_014,
        license: WHISPER_LICENSE,
        revision: WHISPER_REPO_REV,
        provenance: COREML_PROVENANCE,
        archive: Archive::ZipBundle {
            dir_name: "ggml-large-v3-turbo-encoder.mlmodelc",
        },
        platform: Platform::MacOs,
    },
    CatalogEntry {
        id: ids::ACCEL_FASTER,
        kind: AssetKind::SpeechAccelerator,
        name: "whisper small Core ML encoder",
        file_name: "ggml-small-encoder.mlmodelc.zip",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-small-encoder.mlmodelc.zip",
        sha256: Some("de43fb9fed471e95c19e60ae67575c2bf09e8fb607016da171b06ddad313988b"),
        bytes: 163_083_239,
        license: WHISPER_LICENSE,
        revision: WHISPER_REPO_REV,
        provenance: COREML_PROVENANCE,
        archive: Archive::ZipBundle {
            dir_name: "ggml-small-encoder.mlmodelc",
        },
        platform: Platform::MacOs,
    },
    CatalogEntry {
        id: ids::ACCEL_FASTEST,
        kind: AssetKind::SpeechAccelerator,
        name: "whisper tiny Core ML encoder",
        file_name: "ggml-tiny-encoder.mlmodelc.zip",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-tiny-encoder.mlmodelc.zip",
        sha256: Some("c88cbd2648e1f5415092bcf5256add463a0f19943e6938f46e8d4ffdebd47739"),
        bytes: 15_037_446,
        license: WHISPER_LICENSE,
        revision: WHISPER_REPO_REV,
        provenance: COREML_PROVENANCE,
        archive: Archive::ZipBundle {
            dir_name: "ggml-tiny-encoder.mlmodelc",
        },
        platform: Platform::MacOs,
    },
    // --- speech detection -------------------------------------------------
    CatalogEntry {
        id: ids::DETECTOR,
        kind: AssetKind::SpeechDetector,
        name: "silero-vad v5.1.2 (onnx)",
        file_name: "silero-vad-v5.1.2.onnx",
        url: "https://raw.githubusercontent.com/snakers4/silero-vad/v5.1.2/src/silero_vad/data/silero_vad.onnx",
        sha256: Some("2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f"),
        bytes: 2_327_524,
        license: "MIT — Silero VAD, Silero Team",
        revision: "v5.1.2",
        provenance: "github.com/snakers4/silero-vad, tag v5.1.2, ONNX export shipped by upstream",
        archive: Archive::None,
        platform: Platform::Any,
    },
    // --- speakers ---------------------------------------------------------
    CatalogEntry {
        id: ids::SEGMENTER,
        kind: AssetKind::SpeakerSegmenter,
        name: "pyannote segmentation-3.0 (sherpa-onnx export)",
        file_name: "pyannote-segmentation-3.0.onnx",
        url: "https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/9403a6902bb58e3d5ae8c7e77c3422de279db2e0/model.onnx",
        sha256: Some("220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079"),
        bytes: 5_992_913,
        license: "MIT — pyannote/segmentation-3.0 (CNRS); ONNX export by the sherpa-onnx project (MIT)",
        revision: "9403a6902bb58e3d5ae8c7e77c3422de279db2e0",
        provenance: concat!(
            "huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0 — the ",
            "sherpa-onnx project's ONNX conversion of pyannote/segmentation-3.0. ",
            "Chosen over the tar.bz2 release asset because a single file resumes cleanly."
        ),
        archive: Archive::None,
        platform: Platform::Any,
    },
    CatalogEntry {
        id: ids::EMBEDDER,
        kind: AssetKind::SpeakerEmbedder,
        name: "wespeaker en_voxceleb CAM++ (sherpa-onnx export)",
        file_name: "wespeaker-en-voxceleb-campplus.onnx",
        url: "https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/0743f301363dec56491a490f6d6cbc9d67f9a3bf/wespeaker_en_voxceleb_CAM%2B%2B.onnx",
        sha256: Some("c46fad10b5f81e1aa4a60c162714208577093655076c5450f8c469e522ec54ef"),
        bytes: 29_292_684,
        license: "Apache-2.0 — WeSpeaker (wenet-e2e); ONNX export by the sherpa-onnx project (Apache-2.0)",
        revision: "0743f301363dec56491a490f6d6cbc9d67f9a3bf",
        provenance: concat!(
            "huggingface.co/csukuangfj/speaker-embedding-models — the sherpa-onnx ",
            "project's mirror of its speaker-recognition release assets. ",
            "192-dimension CAM++ embeddings trained on VoxCeleb."
        ),
        archive: Archive::None,
        platform: Platform::Any,
    },
];

// ---------------------------------------------------------------------------
// Decoding: the other half of a quality preset.
// ---------------------------------------------------------------------------

/// How hard the engine tries, per preset. A preset is exactly "which speech
/// file" plus "these decoding numbers"; nothing else varies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeParams {
    /// Beam width. 0 means greedy sampling with `best_of` candidates.
    pub beam_size: u32,
    /// Greedy candidates considered. Clamped to at least 1 by whisper.cpp.
    pub best_of: u32,
    /// Starting decoding temperature. 0 is deterministic.
    pub temperature: f32,
    /// How much to raise the temperature on each fallback attempt.
    pub temperature_inc: f32,
    /// Gibberish guard: retry hotter above this entropy.
    pub entropy_thold: f32,
    /// Retry hotter below this average log probability.
    pub logprob_thold: f32,
    /// Treat a window as silence above this probability.
    pub no_speech_thold: f32,
}

/// The beam a live final narrows to when finals are queueing up.
///
/// Anything that lands in the transcript now decodes with the preset's own beam,
/// live or from disk (mantra 1, amendment of 2026-08-20: while Echo is
/// listening, transcript quality outranks resource thrift). A wide beam over two
/// channels can still fall behind on a small machine, so this is the safety
/// valve: while more than [`crate::asr::engine::FINAL_BACKLOG_DEGRADE_AT`]
/// finished utterances are waiting, new ones decode this narrow until the queue
/// drains again.
pub const LIVE_DEGRADED_BEAM: u32 = 2;

/// Temperature step for a live final utterance.
///
/// whisper.cpp builds its fallback ladder as `t, t + inc, … < 1.0`, so from 0.0
/// this one gives exactly two attempts: the deterministic pass and one modest
/// retry. The full `0.0, 0.2, … 1.0` ladder costs up to six decodes of the same
/// audio, which is fine on disk and unpredictable on a caption.
pub const LIVE_TEMPERATURE_INC: f32 = 0.6;

impl DecodeParams {
    pub fn uses_beam_search(&self) -> bool {
        self.beam_size >= 2
    }

    /// How many decodes of the same audio this may cost, worst case.
    ///
    /// Mirrors whisper.cpp's own ladder: `temperature_inc <= 0` means one
    /// attempt and no fallback at all.
    pub fn max_attempts(&self) -> u32 {
        if self.temperature_inc <= 0.0 {
            return 1;
        }
        let mut attempts = 0;
        let mut t = self.temperature;
        while t < 1.0 + 1e-6 {
            attempts += 1;
            t += self.temperature_inc;
        }
        attempts.max(1)
    }

    /// The same preset, decoding a **final live utterance**: the preset's own
    /// beam, with one modest fallback at most.
    ///
    /// This is the text a person keeps, so it is decoded at the quality the disk
    /// pass would give it — the beam is not narrowed for latency any more
    /// (mantra 1's 2026-08-20 amendment: while Echo is listening, quality
    /// outranks thrift). The one thing still trimmed is the temperature ladder:
    /// six decodes of the same audio makes a *caption* unpredictable, and the
    /// deterministic pass plus one retry is where nearly all of the accuracy is.
    ///
    /// [`DecodeParams::live_final_degraded`] is the fallback for a machine that
    /// cannot keep up at this width.
    pub const fn live_final(mut self) -> Self {
        self.temperature = 0.0;
        self.temperature_inc = LIVE_TEMPERATURE_INC;
        self
    }

    /// A live final on a machine that is falling behind: as
    /// [`DecodeParams::live_final`], with the beam narrowed to
    /// [`LIVE_DEGRADED_BEAM`].
    ///
    /// The valve, not the setting. Text that arrives after the meeting has moved
    /// on is worth less than slightly worse text that arrives now, so a queue of
    /// finished utterances buys itself room by narrowing — and goes back to the
    /// full width as soon as it has drained.
    pub const fn live_final_degraded(self) -> Self {
        let mut params = self.live_final();
        if params.beam_size > LIVE_DEGRADED_BEAM {
            params.beam_size = LIVE_DEGRADED_BEAM;
        }
        if params.best_of > LIVE_DEGRADED_BEAM {
            params.best_of = LIVE_DEGRADED_BEAM;
        }
        params
    }

    /// The same preset, decoding a **speculative caption**: greedy, one attempt,
    /// nothing spent on a hypothesis that is about to be replaced.
    pub const fn speculative(mut self) -> Self {
        self.beam_size = 0;
        self.best_of = 1;
        self.temperature = 0.0;
        // No ladder: a caption that arrives late is worse than a caption that is
        // slightly wrong, and the final decode replaces it either way.
        self.temperature_inc = 0.0;
        self
    }
}

/// A quality preset. `name` and `description` are read by people, so they say
/// nothing about models, beams or temperatures (mantra 2).
#[derive(Debug, Clone, Copy)]
pub struct AccuracyPreset {
    /// Stable machine id, stored in settings.
    pub id: &'static str,
    pub name: &'static str,
    /// One plain sentence. The download size is appended at runtime, because it
    /// differs per platform.
    pub blurb: &'static str,
    pub speech_id: &'static str,
    /// Apple encoder companion, or `None` for presets that have none.
    pub accelerator_id: Option<&'static str>,
    pub decode: DecodeParams,
    /// Recommended only on a computer with at least this much memory.
    pub min_memory_bytes: u64,
}

/// The preset a fresh install starts on. Matches `Settings::default()`.
pub const DEFAULT_PRESET_ID: &str = "everyday";

pub const PRESETS: &[AccuracyPreset] = &[
    AccuracyPreset {
        id: DEFAULT_PRESET_ID,
        name: "Everyday accuracy",
        blurb: "Good for most meetings, and understands almost any language.",
        speech_id: ids::SPEECH_EVERYDAY,
        accelerator_id: Some(ids::ACCEL_EVERYDAY),
        decode: DecodeParams {
            beam_size: 5,
            best_of: 5,
            temperature: 0.0,
            temperature_inc: 0.2,
            entropy_thold: 2.4,
            logprob_thold: -1.0,
            no_speech_thold: 0.6,
        },
        // Below roughly 8 GB the everyday level is a stretch alongside a call.
        min_memory_bytes: 8 * 1024 * 1024 * 1024,
    },
    AccuracyPreset {
        id: "faster",
        name: "Faster",
        blurb: "Keeps up more easily and takes far less room. Good when speech is clear.",
        speech_id: ids::SPEECH_FASTER,
        accelerator_id: Some(ids::ACCEL_FASTER),
        decode: DecodeParams {
            beam_size: 2,
            best_of: 2,
            temperature: 0.0,
            temperature_inc: 0.2,
            entropy_thold: 2.4,
            logprob_thold: -1.0,
            no_speech_thold: 0.6,
        },
        min_memory_bytes: 4 * 1024 * 1024 * 1024,
    },
    AccuracyPreset {
        id: "fastest",
        name: "Quickest",
        blurb: "The lightest choice. Expect more mistakes, especially with accents.",
        speech_id: ids::SPEECH_FASTEST,
        accelerator_id: Some(ids::ACCEL_FASTEST),
        decode: DecodeParams {
            beam_size: 0,
            best_of: 1,
            temperature: 0.0,
            temperature_inc: 0.4,
            entropy_thold: 2.8,
            logprob_thold: -1.0,
            no_speech_thold: 0.6,
        },
        min_memory_bytes: 0,
    },
];

// ---------------------------------------------------------------------------
// Lookups
// ---------------------------------------------------------------------------

pub fn entry(asset_id: &str) -> Option<&'static CatalogEntry> {
    CATALOG.iter().find(|e| e.id == asset_id)
}

pub fn preset(level_id: &str) -> Option<&'static AccuracyPreset> {
    PRESETS.iter().find(|p| p.id == level_id)
}

/// The preset for a stored setting, falling back to the default when the stored
/// value is unknown (a downgrade, or a hand-edited database).
pub fn preset_or_default(level_id: &str) -> &'static AccuracyPreset {
    preset(level_id).unwrap_or_else(default_preset)
}

pub fn default_preset() -> &'static AccuracyPreset {
    preset(DEFAULT_PRESET_ID).expect("the default preset is in PRESETS")
}

/// Every asset a preset needs on *this* platform, in the order they should be
/// fetched: speech first so a recording can start, then the detector, then the
/// speed-up and the speaker files.
pub fn preset_asset_ids(level_id: &str) -> Vec<&'static str> {
    let p = preset_or_default(level_id);
    let mut wanted = vec![p.speech_id, ids::DETECTOR];
    if let Some(accel) = p.accelerator_id {
        wanted.push(accel);
    }
    wanted.push(ids::SEGMENTER);
    wanted.push(ids::EMBEDDER);
    wanted.retain(|id| entry(id).is_some_and(CatalogEntry::applies_here));
    wanted
}

/// The subset a recording actually needs before it can produce text. The
/// speaker files can arrive later, because the offline pass runs afterwards.
pub fn preset_required_asset_ids(level_id: &str) -> Vec<&'static str> {
    let p = preset_or_default(level_id);
    let mut wanted = vec![p.speech_id, ids::DETECTOR];
    wanted.retain(|id| entry(id).is_some_and(CatalogEntry::applies_here));
    wanted
}

/// Total bytes a preset costs on this platform.
pub fn preset_total_bytes(level_id: &str) -> i64 {
    preset_asset_ids(level_id)
        .into_iter()
        .filter_map(entry)
        .map(|e| e.bytes)
        .sum()
}

/// Which preset we suggest on this computer, from installed memory alone.
pub fn recommended_preset_id(total_memory_bytes: u64) -> &'static str {
    // Presets are ordered best-first, so the first one this computer can carry
    // is the one to suggest.
    for p in PRESETS {
        if total_memory_bytes == 0 || total_memory_bytes >= p.min_memory_bytes {
            return p.id;
        }
    }
    DEFAULT_PRESET_ID
}

/// "1.6 GB", "488 MB". Used in the sentence a person reads, so it is short and
/// decimal, the way a download is normally described.
pub fn human_bytes(bytes: i64) -> String {
    const KB: f64 = 1_000.0;
    let b = bytes.max(0) as f64;
    if b < KB {
        return format!("{bytes} bytes");
    }
    let units = ["kB", "MB", "GB", "TB"];
    let mut value = b / KB;
    let mut unit = 0;
    while value >= KB && unit + 1 < units.len() {
        value /= KB;
        unit += 1;
    }
    if value < 10.0 {
        format!("{value:.1} {}", units[unit])
    } else {
        format!("{value:.0} {}", units[unit])
    }
}

/// The user-facing description of a preset: the blurb plus how much room it
/// takes. Still no jargon.
pub fn preset_description(level_id: &str) -> String {
    let p = preset_or_default(level_id);
    format!(
        "{} Takes about {} of space, downloaded once.",
        p.blurb,
        human_bytes(preset_total_bytes(p.id))
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn every_id_is_unique_and_every_file_name_is_unique() {
        let mut ids = HashSet::new();
        let mut names = HashSet::new();
        for e in CATALOG {
            assert!(ids.insert(e.id), "duplicate asset id {}", e.id);
            assert!(
                names.insert(e.installed_name()),
                "two entries install as {}",
                e.installed_name()
            );
        }
        let mut preset_ids = HashSet::new();
        for p in PRESETS {
            assert!(preset_ids.insert(p.id), "duplicate preset id {}", p.id);
        }
    }

    #[test]
    fn every_entry_has_provenance_a_licence_a_length_and_a_pinned_url() {
        for e in CATALOG {
            assert!(!e.license.is_empty(), "{} has no licence", e.id);
            assert!(!e.provenance.is_empty(), "{} has no provenance", e.id);
            assert!(!e.revision.is_empty(), "{} has no revision", e.id);
            assert!(e.bytes > 0, "{} has no expected length", e.id);
            assert!(
                e.url.starts_with("https://"),
                "{} is not fetched over https",
                e.id
            );
            // Pinned, not floating: a branch name in the URL would let the
            // bytes change under an installed entry.
            assert!(
                !e.url.contains("/resolve/main/") && !e.url.contains("/refs/heads/"),
                "{} points at a moving revision",
                e.id
            );
            if let Some(sha) = e.sha256 {
                assert_eq!(sha.len(), 64, "{} has a malformed hash", e.id);
                assert!(
                    sha.chars()
                        .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()),
                    "{} hash must be lowercase hex",
                    e.id
                );
            }
        }
    }

    #[test]
    fn accelerators_are_apple_only_bundles_named_the_way_whisper_expects() {
        for e in CATALOG {
            if e.kind != AssetKind::SpeechAccelerator {
                continue;
            }
            assert_eq!(e.platform, Platform::MacOs, "{} must be Apple-only", e.id);
            assert!(e.is_bundle(), "{} must unpack to a bundle", e.id);
            let dir = e.installed_name();
            assert!(
                dir.ends_with("-encoder.mlmodelc"),
                "{dir} is not the name whisper.cpp looks for"
            );
            // whisper.cpp derives "<speech file without .bin>-encoder.mlmodelc",
            // so the bundle name has to match its preset's speech file.
            let preset = PRESETS
                .iter()
                .find(|p| p.accelerator_id == Some(e.id))
                .expect("every accelerator belongs to a preset");
            let speech = entry(preset.speech_id).unwrap();
            let stem = speech.file_name.trim_end_matches(".bin");
            assert_eq!(dir, format!("{stem}-encoder.mlmodelc"));
        }
    }

    #[test]
    fn presets_point_at_speech_files_that_exist() {
        for p in PRESETS {
            let speech = entry(p.speech_id).expect("preset speech file is catalogued");
            assert_eq!(speech.kind, AssetKind::Speech);
            if let Some(a) = p.accelerator_id {
                let accel = entry(a).expect("preset accelerator is catalogued");
                assert_eq!(accel.kind, AssetKind::SpeechAccelerator);
            }
        }
        for id in SHARED_ASSET_IDS {
            assert!(entry(id).is_some(), "{id} is not catalogued");
        }
    }

    #[test]
    fn the_default_preset_is_the_one_settings_starts_on() {
        assert_eq!(
            DEFAULT_PRESET_ID,
            crate::types::Settings::default().accuracy_level_id
        );
        assert_eq!(default_preset().id, DEFAULT_PRESET_ID);
        assert_eq!(preset_or_default("no-such-level").id, DEFAULT_PRESET_ID);
    }

    #[test]
    fn a_preset_needs_speech_detection_and_the_speaker_files() {
        let all = preset_asset_ids(DEFAULT_PRESET_ID);
        assert_eq!(
            all.first(),
            Some(&ids::SPEECH_EVERYDAY),
            "speech comes first"
        );
        assert!(all.contains(&ids::DETECTOR));
        assert!(all.contains(&ids::SEGMENTER));
        assert!(all.contains(&ids::EMBEDDER));
        assert_eq!(
            all.contains(&ids::ACCEL_EVERYDAY),
            cfg!(target_os = "macos"),
            "the Apple companion is only fetched on Apple hardware"
        );

        // Recording only waits for speech plus detection.
        let required = preset_required_asset_ids(DEFAULT_PRESET_ID);
        assert_eq!(required, vec![ids::SPEECH_EVERYDAY, ids::DETECTOR]);
    }

    #[test]
    fn preset_totals_add_up_and_get_smaller_as_the_preset_gets_faster() {
        let everyday = preset_total_bytes("everyday");
        let faster = preset_total_bytes("faster");
        let fastest = preset_total_bytes("fastest");
        assert!(everyday > faster && faster > fastest);
        // The total is exactly the sum of what we would fetch here.
        let manual: i64 = preset_asset_ids("faster")
            .into_iter()
            .map(|id| entry(id).unwrap().bytes)
            .sum();
        assert_eq!(manual, faster);
    }

    #[test]
    fn quality_presets_differ_only_in_speech_file_and_decoding_effort() {
        let everyday = preset("everyday").unwrap();
        let faster = preset("faster").unwrap();
        let fastest = preset("fastest").unwrap();
        assert!(everyday.decode.beam_size > faster.decode.beam_size);
        assert!(faster.decode.uses_beam_search());
        assert!(
            !fastest.decode.uses_beam_search(),
            "the quickest preset samples greedily"
        );
        for p in PRESETS {
            assert_eq!(p.decode.temperature, 0.0, "decoding starts deterministic");
            assert!(p.decode.temperature_inc > 0.0, "fallbacks must be possible");
            assert!(p.decode.best_of >= 1);
        }
    }

    /// The full ladder belongs to the disk pass; the beam belongs to anything
    /// that lands in the transcript, live or not.
    #[test]
    fn a_live_final_keeps_the_presets_beam_and_gives_up_only_the_ladder() {
        for p in PRESETS {
            let catchup = p.decode;
            let live = catchup.live_final();
            let degraded = catchup.live_final_degraded();
            let speculative = catchup.speculative();

            assert_eq!(
                live.beam_size, catchup.beam_size,
                "{} decodes finals narrower than the recording",
                p.id
            );
            assert!(
                degraded.beam_size <= LIVE_DEGRADED_BEAM,
                "{} still searches {} beams with the valve open",
                p.id,
                degraded.beam_size
            );
            assert_eq!(
                degraded.max_attempts(),
                live.max_attempts(),
                "{} changes more than the beam when it falls behind",
                p.id
            );
            assert_eq!(
                live.max_attempts(),
                2,
                "{} allows {} live attempts, not one fallback",
                p.id,
                live.max_attempts()
            );
            assert_eq!(
                speculative.max_attempts(),
                1,
                "{} falls back on a caption that is about to be replaced",
                p.id
            );
            assert!(!speculative.uses_beam_search());
            assert_eq!(speculative.best_of, 1);
            // The thresholds are untouched: they are about what counts as
            // silence and gibberish, not about how hard to try.
            assert_eq!(speculative.no_speech_thold, catchup.no_speech_thold);
            assert_eq!(live.no_speech_thold, catchup.no_speech_thold);
            // And the preset itself still has its full ladder for the disk pass.
            assert!(
                catchup.max_attempts() >= live.max_attempts(),
                "{} would decode less thoroughly from disk than live",
                p.id
            );
        }
        // The everyday preset is the one with a wide beam to keep or give up.
        let everyday = preset("everyday").unwrap().decode;
        assert_eq!(everyday.beam_size, 5, "catch-up keeps the wide beam");
        assert_eq!(
            everyday.live_final().beam_size,
            5,
            "a live final is decoded at catch-up quality"
        );
        assert_eq!(everyday.live_final_degraded().beam_size, 2);
        assert_eq!(everyday.max_attempts(), 6, "0.0, 0.2, … 1.0 on disk");
    }

    #[test]
    fn a_small_computer_is_pointed_at_a_smaller_preset() {
        assert_eq!(recommended_preset_id(32 * 1024 * 1024 * 1024), "everyday");
        assert_eq!(recommended_preset_id(6 * 1024 * 1024 * 1024), "faster");
        assert_eq!(recommended_preset_id(2 * 1024 * 1024 * 1024), "fastest");
        // Unknown memory must not downgrade anybody.
        assert_eq!(recommended_preset_id(0), "everyday");
    }

    #[test]
    fn nothing_a_person_reads_contains_jargon() {
        // Mantra 2. These words may appear in `name`, `license` and
        // `provenance`, which only Settings → Advanced renders.
        const BANNED: &[&str] = &[
            "whisper",
            "model",
            "onnx",
            "ggml",
            "vad",
            "diariz",
            "token",
            "coreml",
            "core ml",
            "beam",
            "temperature",
            "encoder",
            "inference",
            "gpu",
            "metal",
            "vulkan",
            "silero",
            "pyannote",
            "embedding",
        ];
        for p in PRESETS {
            let readable = format!("{} {}", p.name, preset_description(p.id)).to_lowercase();
            for word in BANNED {
                assert!(
                    !readable.contains(word),
                    "preset {} says {word:?} to the person: {readable}",
                    p.id
                );
            }
        }
    }

    #[test]
    fn sizes_read_the_way_a_download_is_described() {
        assert_eq!(human_bytes(1_624_555_275), "1.6 GB");
        assert_eq!(human_bytes(487_601_967), "488 MB");
        assert_eq!(human_bytes(2_327_524), "2.3 MB");
        assert_eq!(human_bytes(512), "512 bytes");
        assert_eq!(human_bytes(-1), "-1 bytes");
    }

    #[test]
    fn descriptions_say_how_much_room_a_preset_takes() {
        for p in PRESETS {
            let d = preset_description(p.id);
            assert!(d.contains("of space"), "{d}");
            assert!(d.starts_with(p.blurb), "{d}");
        }
    }
}
