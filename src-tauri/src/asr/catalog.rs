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
//! * An entry is **never removed** when it stops being wanted. A release that
//!   changes which weights Echo uses leaves the old ones catalogued, because
//!   that is the only way [`crate::asr::models::plan_reconcile`] can recognise
//!   what is on a person's disk and clean it up. See [`OBSOLETE_ASSET_IDS`].
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
pub const CATALOG_REVISION: &str = "2026-08-20.2";

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
    /// For an Apple encoder companion: the speech entry it belongs to.
    ///
    /// Explicit rather than derived from whichever level happens to use them,
    /// because a companion outlives the level: once large-v3 replaced turbo,
    /// turbo's companion still has to be recognised as *turbo's* so it can be
    /// cleaned up with it. `None` for everything that is not a companion.
    pub pairs_with: Option<&'static str>,
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

/// The Apple encoder companion that belongs to a speech entry, if there is one
/// and this platform uses it.
///
/// Pairing matters to the reconcile: while an upgrade is still downloading, the
/// engine keeps serving the *old* weights, and it must load the old weights'
/// companion, not the new one's.
pub fn accelerator_for(speech_id: &str) -> Option<&'static CatalogEntry> {
    CATALOG
        .iter()
        .find(|e| e.pairs_with == Some(speech_id) && e.applies_here())
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
///
/// **These strings are permanent.** They are the primary key of the `models`
/// table on every installed copy of Echo, so the id of a retired asset has to
/// keep naming the same bytes for the cleanup to find them.
pub mod ids {
    /// The speech model Echo uses: full large-v3, and its Apple companion.
    pub const SPEECH: &str = "speech-large-v3";
    pub const ACCEL: &str = "speech-accelerator-large-v3";

    pub const DETECTOR: &str = "speech-detector-silero-v5";
    pub const SEGMENTER: &str = "speaker-segmenter-pyannote-3";
    /// The voice-print network. See [`super::CATALOG`]'s speaker section for why
    /// it is this one and not a newer-sounding one.
    pub const EMBEDDER: &str = "speaker-embedder-wespeaker-resnet34-lm";

    // --- retired: catalogued so they can be recognised and removed ---------
    /// What Echo used before 2026-08-20. Still on many disks.
    pub const SPEECH_TURBO: &str = "speech-large-v3-turbo";
    pub const ACCEL_TURBO: &str = "speech-accelerator-large-v3-turbo";
    pub const SPEECH_SMALL: &str = "speech-small";
    pub const ACCEL_SMALL: &str = "speech-accelerator-small";
    pub const SPEECH_TINY: &str = "speech-tiny";
    pub const ACCEL_TINY: &str = "speech-accelerator-tiny";
    /// The voice-print network Echo used before 2026-08-20, whose embedding
    /// space the clustering threshold had never been measured against.
    pub const EMBEDDER_CAMPLUS: &str = "speaker-embedder-wespeaker-campplus";
}

/// Everything the level needs beyond speech itself: speech detection and the
/// two speaker files. Small enough that we never make the person choose.
pub const SHARED_ASSET_IDS: &[&str] = &[ids::DETECTOR, ids::SEGMENTER, ids::EMBEDDER];

/// Weights Echo has shipped in the past and no longer wants.
///
/// They stay in [`CATALOG`] and they stay listed here for exactly one reason:
/// the reconcile has to be able to look at a person's disk, recognise a file as
/// something *Echo* put there, and delete it. An unrecognised file is left
/// alone — we do not delete things we cannot account for.
///
/// Nothing reads this to decide what to download. It is derived from the
/// catalog and the level in [`obsolete_asset_ids`]; the constant only exists so
/// a test can assert the two agree.
///
/// It is not only speech weights. A change of voice-print network is the same
/// kind of event as a change of speech model, and rides the same reconcile: the
/// old network stays catalogued so it can be found and removed once the new one
/// is verified on disk (see [`crate::asr::reconcile::is_supersedable`]).
pub const OBSOLETE_ASSET_IDS: &[&str] = &[
    ids::SPEECH_TURBO,
    ids::ACCEL_TURBO,
    ids::SPEECH_SMALL,
    ids::ACCEL_SMALL,
    ids::SPEECH_TINY,
    ids::ACCEL_TINY,
    ids::EMBEDDER_CAMPLUS,
];

/// The catalog. Order matters in one place only: the speech entries are listed
/// best-first, which is the order [`crate::asr::models::plan_reconcile`] falls
/// back through when it has to keep serving something older.
pub const CATALOG: &[CatalogEntry] = &[
    // --- speech -----------------------------------------------------------
    //
    // Full large-v3. Every lane uses it: live captions, live finals, the
    // catch-up pass and Listen again (product decision of 2026-08-20). The
    // lengths and hashes below were taken by downloading both files from the
    // pinned revision and hashing what arrived — that download *is* the
    // provenance record for these two lines.
    CatalogEntry {
        id: ids::SPEECH,
        kind: AssetKind::Speech,
        name: "whisper large-v3 (ggml)",
        file_name: "ggml-large-v3.bin",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-large-v3.bin",
        sha256: Some("64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2"),
        bytes: 3_095_033_483,
        license: WHISPER_LICENSE,
        revision: WHISPER_REPO_REV,
        provenance: WHISPER_PROVENANCE,
        archive: Archive::None,
        platform: Platform::Any,
        pairs_with: None,
    },
    // --- speech Echo no longer wants -------------------------------------
    //
    // Kept so the reconcile can find them on disk and remove them. Nothing
    // downloads these any more; see `OBSOLETE_ASSET_IDS`.
    CatalogEntry {
        id: ids::SPEECH_TURBO,
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
        pairs_with: None,
    },
    CatalogEntry {
        id: ids::SPEECH_SMALL,
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
        pairs_with: None,
    },
    CatalogEntry {
        id: ids::SPEECH_TINY,
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
        pairs_with: None,
    },
    // --- Apple encoder companions ----------------------------------------
    CatalogEntry {
        id: ids::ACCEL,
        kind: AssetKind::SpeechAccelerator,
        name: "whisper large-v3 Core ML encoder",
        file_name: "ggml-large-v3-encoder.mlmodelc.zip",
        url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-large-v3-encoder.mlmodelc.zip",
        sha256: Some("47837be7594a29429ec08620043390c4d6d467f8bd362df09e9390ace76a55a4"),
        bytes: 1_175_711_232,
        license: WHISPER_LICENSE,
        revision: WHISPER_REPO_REV,
        provenance: COREML_PROVENANCE,
        archive: Archive::ZipBundle {
            dir_name: "ggml-large-v3-encoder.mlmodelc",
        },
        platform: Platform::MacOs,
        pairs_with: Some(ids::SPEECH),
    },
    CatalogEntry {
        id: ids::ACCEL_TURBO,
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
        pairs_with: Some(ids::SPEECH_TURBO),
    },
    CatalogEntry {
        id: ids::ACCEL_SMALL,
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
        pairs_with: Some(ids::SPEECH_SMALL),
    },
    CatalogEntry {
        id: ids::ACCEL_TINY,
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
        pairs_with: Some(ids::SPEECH_TINY),
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
        pairs_with: None,
    },
    // --- speakers ---------------------------------------------------------
    //
    // These two are the pyannote *community-1* pipeline's own two networks, and
    // that is not a coincidence — it is the result of checking, on 2026-08-20,
    // what "upgrade to community-1" actually means file by file.
    //
    // **Segmentation.** community-1 does not ship a new segmentation network. Its
    // weights are the weights of segmentation-3.0. Three independent ungated ONNX
    // exports of "community-1 segmentation" were downloaded and compared against
    // the checkpoint in pyannote's own ungated mirror
    // (huggingface.co/pyannote-community/speaker-diarization-community-1,
    // segmentation/pytorch_model.bin) and against the export below: all four ONNX
    // files agree bit-for-bit on every convolution, LSTM and projection tensor,
    // and all four return identical logits — sum -24458.902036 over a fixed 10 s
    // signal — from `cargo run --example onnx_contract -- --run`. pyannote's own
    // release notes say the same thing in words: community-1 keeps "the same
    // segmentation performance … as on pyannote.audio 3.1". So the entry below
    // stays exactly as it was; there is no segmentation upgrade to make, and
    // swapping mirrors for a byte-identical network would only add a new file to
    // trust. What community-1 actually changes is the *clustering* — Bayesian HMM
    // (VBx) with a PLDA instead of a cosine cut — which is a change to
    // `diarize::cluster`, not to an asset.
    //
    // **Voice prints.** This is where community-1's win lives, and it is the
    // entry that moved. See the embedder entry below.
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
            "Chosen over the tar.bz2 release asset because a single file resumes cleanly. ",
            "Verified on 2026-08-20 to be the same network, tensor for tensor, as the ",
            "segmentation checkpoint inside pyannote's speaker-diarization-community-1."
        ),
        archive: Archive::None,
        platform: Platform::Any,
        pairs_with: None,
    },
    // WeSpeaker ResNet34-LM: the voice-print network the reference pyannote
    // pipelines use, 3.1 and community-1 alike.
    //
    // It replaced CAM++ because of the threshold, not because of a leaderboard.
    // `diarize::cluster::DISTANCE_THRESHOLD` is a *published* number — the cosine
    // cut that minimises diarization error on DIHARD for centroid-linkage
    // agglomerative clustering of **these** 256-dimension vectors, which is
    // exactly the algorithm `diarize::cluster::cluster` runs. Under CAM++ that
    // number described a different embedding space (512 dimensions, and a
    // different geometry), so the one constant deciding how many people Echo
    // thinks were in the room was uncalibrated. Now it is not.
    //
    // Verified on 2026-08-20 to be community-1's own embedding network: this file
    // and the ONNX export bundled with community-1 agree on 73 of 75 weight
    // tensors bit-for-bit (the two that differ are batch-norm folding) and return
    // fingerprints that agree to four decimal places.
    //
    // Bigger WeSpeaker networks are mirrored next to this one — ResNet152/221/293,
    // 79 to 114 MB, roughly half the equal-error rate. They are deliberately not
    // used: no published clustering threshold exists for them, so taking one would
    // trade a measured number for a guessed one, and the guessed number is what
    // decides the speaker count. A wrong count is the failure a person has to
    // clean up by hand; a tenth of a percent of equal-error rate is not.
    CatalogEntry {
        id: ids::EMBEDDER,
        kind: AssetKind::SpeakerEmbedder,
        name: "wespeaker en_voxceleb ResNet34-LM (sherpa-onnx export)",
        file_name: "wespeaker-en-voxceleb-resnet34-lm.onnx",
        url: "https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/0743f301363dec56491a490f6d6cbc9d67f9a3bf/wespeaker_en_voxceleb_resnet34_LM.onnx",
        sha256: Some("e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012"),
        bytes: 26_530_550,
        license: "Apache-2.0 — WeSpeaker (wenet-e2e); ONNX export by the sherpa-onnx project (Apache-2.0)",
        revision: "0743f301363dec56491a490f6d6cbc9d67f9a3bf",
        provenance: concat!(
            "huggingface.co/csukuangfj/speaker-embedding-models — the sherpa-onnx ",
            "project's mirror of its speaker-recognition release assets, the same ",
            "pinned revision the CAM++ entry used. 256-dimension WeSpeaker ",
            "ResNet34 embeddings, large-margin finetuned, trained on VoxCeleb; the ",
            "voice-print network of the reference pyannote 3.1 and community-1 ",
            "pipelines. Takes 80-bin Kaldi log-mel features, which is what ",
            "diarize::features produces."
        ),
        archive: Archive::None,
        platform: Platform::Any,
        pairs_with: None,
    },
    // --- voice prints Echo no longer wants -------------------------------
    //
    // Catalogued so the reconcile can find it on disk and remove it once
    // ResNet34-LM is verified installed. Nothing downloads this any more.
    CatalogEntry {
        id: ids::EMBEDDER_CAMPLUS,
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
            "512-dimension CAM++ embeddings trained on VoxCeleb. Retired on ",
            "2026-08-20: no clustering threshold was ever measured against this ",
            "embedding space."
        ),
        archive: Archive::None,
        platform: Platform::Any,
        pairs_with: None,
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
    /// Loop guard: an answer whose tokens carry *less* variety than this is
    /// thrown away and decoded again hotter.
    ///
    /// This is whisper.cpp's stand-in for OpenAI's `compression_ratio_threshold`
    /// and it runs the other way round — a decoder stuck in a loop produces a
    /// very predictable token stream, so the failure is entropy **below** the
    /// bar, not above it (`whisper.cpp:7527`). 2.4 is the same number the two
    /// projects agree on, and it is the number Echo has always used; it is
    /// spelled out here because the 2026-08-24 meeting contained three-times
    /// loops ("ma è un po' figgito" three times in one line) and the first
    /// question anybody will ask of this file is whether the guard was on.
    ///
    /// Its limit, honestly: whisper.cpp only applies it to answers longer than
    /// 32 tokens, so a short line that repeats itself twice is under the bar the
    /// guard is measured over. Nothing in the binding can change that.
    pub entropy_thold: f32,
    /// Below this average log probability per token, the answer is not trusted:
    /// it is either decoded again hotter, or — when the model also thinks the
    /// window was silence — dropped.
    ///
    /// One number, two jobs, and they pull in opposite directions. See
    /// [`LIVE_LOGPROB_THOLD`].
    pub logprob_thold: f32,
    /// Above this probability of "there was no speech here", the window is
    /// treated as silence — but only if [`DecodeParams::logprob_thold`] also
    /// says the words were a guess. See [`CAPTION_NO_SPEECH_THOLD`].
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

/// How many fallbacks one catch-up window is allowed: two retries, three
/// decodes of the same audio at most.
///
/// The disk pass used to run the preset's whole `0.0, 0.2, … 1.0` ladder, six
/// decodes of the same window. On clean audio nothing past the first rung ever
/// runs, so it looked free; on the degraded stretches a real meeting is full of
/// — a phone speaker, someone typing, two people at once — it is the difference
/// between one decode and six, on every one of them. Past the third rung the
/// answer is a hotter guess at audio that is not going to resolve, and the words
/// are the same words. Two retries keep the recovery that matters and drop the
/// tail that only costs time.
pub const CATCHUP_MAX_FALLBACKS: u32 = 2;

/// Temperature step for a catch-up window.
///
/// The number of rungs is the *only* thing whisper.cpp's ladder can be capped
/// by: it is built as `for (t = temperature; t < 1.0 + 1e-6; t += inc)` with no
/// count to limit (`whisper.cpp:6851`), so "three attempts from 0.0" and "steps
/// of 0.4" are the same statement. Hence `1.0 / 3` rounded down to something
/// legible: 0.0, 0.4, 0.8.
pub const CATCHUP_TEMPERATURE_INC: f32 = 0.4;

/// The loop guard, everywhere. See [`DecodeParams::entropy_thold`].
///
/// Named rather than inlined so a test can assert every lane still carries it:
/// a lane that quietly lost it would show up as looped text in a transcript
/// months later, which is exactly how the 2026-08-24 loops were found.
pub const LOOP_ENTROPY_THOLD: f32 = 2.4;

/// What a **live final** demands of itself before it believes its own words.
///
/// Raised from whisper.cpp's -1.0 after the 2026-08-24 meeting. The knob does
/// two things at once (`whisper.cpp:7555` and `:7585`) and they trade against
/// each other:
///
/// * it widens the silence gate — a window the model calls silence is only
///   dropped if the words it produced anyway were *also* a guess, and at -1.0
///   almost nothing counts as a guess;
/// * it widens the fallback gate — an answer under the bar is decoded again at
///   a higher temperature, which costs time.
///
/// -0.85 is a deliberately small step, and it is taken on the live lane rather
/// than the disk lane for one reason: a live final that gets dropped leaves no
/// text against that audio, and the catch-up pass reads exactly the stretches
/// that have no text against them. Live, this is a *deferral*. On disk it would
/// be a deletion.
///
/// It will not, on its own, catch the 33 phantoms of 2026-08-24: those were
/// decoded confidently, well above any bar that leaves real quiet speech alone.
/// That is [`crate::asr::phantom`]'s job, and the reason it exists.
///
/// The step is small for a reason worth spelling out: quiet, accented and
/// far-field speech decodes around -0.5 to -0.9, so a bar much above this stops
/// being a guard against nonsense and starts being a guard against soft-spoken
/// people. The caption lane does not take this step at all — see
/// [`CAPTION_NO_SPEECH_THOLD`].
pub const LIVE_LOGPROB_THOLD: f32 = -0.85;

/// How sure the model has to be that a **caption's** window was silence.
///
/// Lowered from whisper.cpp's 0.6, and it is the *only* half of the silence gate
/// this lane moves. That gate is an AND of two beliefs (whisper.cpp:7585):
///
/// ```text
/// no_speech_prob > no_speech_thold && avg_logprobs < logprob_thold
/// ```
///
/// and moving both halves at once is not "more strictness", it is a different
/// gate. The log-probability half was tried at -0.35 here and reverted, because
/// -0.35 sits *inside* the band real speech decodes at — clean speech runs
/// around -0.15 to -0.4, and quiet, accented or far-field speech routinely -0.5
/// to -0.9 — which reduces the gate to `no_speech_prob > 0.5` alone for exactly
/// the person Echo most needs to caption. Blank captions for one soft-spoken
/// participant for a whole call is not the small, self-correcting cost the rest
/// of this lane's reasoning is built on.
///
/// It would not have bought anything either. The phantoms of 2026-08-24 were
/// decoded *confidently* — a hallucinated "Grazie." carries a high
/// `avg_logprobs`, so no bar that leaves real speech alone catches it, on any
/// lane. That is [`crate::asr::phantom`]'s job. What this half still does is
/// drop a caption that is both probably-silence and word-salad, and that costs
/// nobody anything.
///
/// Left at whisper.cpp's measured 0.6 on both lanes that write the transcript.
pub const CAPTION_NO_SPEECH_THOLD: f32 = 0.5;

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
    ///
    /// The one addition of 2026-08-24: a slightly higher bar for believing an
    /// answer, because a live final that is dropped is re-read from disk by the
    /// catch-up pass, and one that is wrong is in the transcript for good. See
    /// [`LIVE_LOGPROB_THOLD`].
    pub const fn live_final(mut self) -> Self {
        self.temperature = 0.0;
        self.temperature_inc = LIVE_TEMPERATURE_INC;
        self.logprob_thold = LIVE_LOGPROB_THOLD;
        self.entropy_thold = LOOP_ENTROPY_THOLD;
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

    /// The same preset, decoding a **window read back from disk**: the preset's
    /// full beam, and at most [`CATCHUP_MAX_FALLBACKS`] retries.
    ///
    /// Nothing is waiting for this, which is why it keeps the wide beam — but
    /// "nothing is waiting" is not "it may cost anything". A pass that takes
    /// longer than the meeting did is a pass that is still running when the
    /// person opens the transcript, so the ladder is capped here for the same
    /// reason the beam is not: the beam buys words, the sixth temperature does
    /// not.
    ///
    /// The thresholds are deliberately *not* tightened here. This pass is the
    /// last chance these words have: nothing reads the audio again afterwards,
    /// so a window dropped here is a window that stays blank for ever. Silence
    /// that talked its way into a line is taken off this lane after the decode
    /// instead, by [`crate::asr::phantom`], which needs two pieces of evidence
    /// rather than one threshold.
    pub const fn catch_up(mut self) -> Self {
        self.temperature = 0.0;
        self.temperature_inc = CATCHUP_TEMPERATURE_INC;
        self.entropy_thold = LOOP_ENTROPY_THOLD;
        self
    }

    /// The same preset, decoding a **speculative caption**: greedy, one attempt,
    /// nothing spent on a hypothesis that is about to be replaced — and the
    /// quickest of the three lanes to call a window silence, though only in the
    /// one way that cannot cost a quiet speaker their captions.
    pub const fn speculative(mut self) -> Self {
        self.beam_size = 0;
        self.best_of = 1;
        self.temperature = 0.0;
        // No ladder: a caption that arrives late is worse than a caption that is
        // slightly wrong, and the final decode replaces it either way.
        self.temperature_inc = 0.0;
        // With one rung, whisper.cpp never consults these for a retry — they
        // only decide whether the window was silence. The half that means "the
        // model heard no speech" moves; the half that means "the words are a
        // guess" stays where whisper.cpp measured it, or a quiet speaker's
        // captions go blank for the whole call. See [`CAPTION_NO_SPEECH_THOLD`].
        self.no_speech_thold = CAPTION_NO_SPEECH_THOLD;
        self.entropy_thold = LOOP_ENTROPY_THOLD;
        self
    }
}

/// The level: which weights, and how hard to decode with them.
///
/// `name` and `blurb` are read by people, so they say nothing about models,
/// beams or temperatures (mantra 2).
///
/// There is exactly one of these (see [`PRESETS`]). The type is still a list of
/// named levels rather than a bare constant because the settings column, the
/// `models` rows and the IPC surface are all keyed by a level id, and because
/// the *next* change of weights wants the same shape this one had.
#[derive(Debug, Clone, Copy)]
pub struct AccuracyPreset {
    /// Stable machine id, stored in settings.
    pub id: &'static str,
    pub name: &'static str,
    /// One plain sentence. The download size is appended at runtime, because it
    /// differs per platform.
    pub blurb: &'static str,
    pub speech_id: &'static str,
    /// Apple encoder companion, or `None` on a platform with none.
    pub accelerator_id: Option<&'static str>,
    pub decode: DecodeParams,
}

/// The only level there is. Matches `Settings::default()`.
pub const DEFAULT_PRESET_ID: &str = "everyday";

/// One level, one model, everywhere (product decision of 2026-08-20).
///
/// Echo used to carry three levels and pick between them by installed memory.
/// Two things were wrong with that. It was a choice nobody could make well —
/// the honest answer does not depend on the person (mantra 1's amendment) — and
/// it meant the transcript a person got depended on how much RAM they happened
/// to have, which is not a promise anybody would agree to if it were stated out
/// loud. So: full large-v3 for live captions, live finals, the catch-up pass and
/// Listen again alike. Nothing routes by lane, by machine or by plan.
///
/// The decoding numbers still differ *by lane* — a caption is greedy, a final
/// keeps the beam, the disk pass keeps the whole ladder — and that is the
/// business of [`DecodeParams::speculative`], [`DecodeParams::live_final`] and
/// this one struct they all start from.
pub const PRESETS: &[AccuracyPreset] = &[AccuracyPreset {
    id: DEFAULT_PRESET_ID,
    name: "Everyday accuracy",
    blurb: "Good for most meetings, and understands almost any language.",
    speech_id: ids::SPEECH,
    accelerator_id: Some(ids::ACCEL),
    decode: DecodeParams {
        beam_size: 5,
        best_of: 5,
        temperature: 0.0,
        temperature_inc: 0.2,
        entropy_thold: LOOP_ENTROPY_THOLD,
        // whisper.cpp's own measured defaults, and the disk pass keeps them: see
        // `DecodeParams::catch_up`. The live lanes tighten them from here.
        logprob_thold: -1.0,
        no_speech_thold: 0.6,
    },
}];

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
/// fetched.
///
/// Speech first, so a recording can start as early as possible, then speech
/// detection. Then the two speaker files, and only then the Apple speed-up —
/// deliberately in that order, even though the speed-up matters more to how Echo
/// feels. The speaker files are 32 MB against the companion's 1.2 GB, and while
/// they are missing a finished meeting cannot be split into people at all. Thirty
/// megabytes of ordering is the difference between "speaker separation works
/// twenty seconds after the upgrade starts" and "it works twenty minutes later".
/// The companion only ever costs speed, never correctness.
pub fn preset_asset_ids(level_id: &str) -> Vec<&'static str> {
    let p = preset_or_default(level_id);
    let mut wanted = vec![p.speech_id, ids::DETECTOR, ids::SEGMENTER, ids::EMBEDDER];
    if let Some(accel) = p.accelerator_id {
        wanted.push(accel);
    }
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

/// Catalogued assets this platform uses that the level does **not** want.
///
/// This is the cleanup's whole idea of "old": derived from the level, so adding
/// a level's replacement weights to the catalog is the only edit a future change
/// of model needs. Order follows [`CATALOG`], which keeps it stable for logs.
///
/// Speech-adjacent only by construction — the detector and the two speaker files
/// are shared by every level, so they never appear here.
pub fn obsolete_asset_ids(level_id: &str) -> Vec<&'static str> {
    let wanted = preset_asset_ids(level_id);
    CATALOG
        .iter()
        .filter(|e| e.applies_here() && !wanted.contains(&e.id))
        .map(|e| e.id)
        .collect()
}

/// Speech entries this platform could load, best first.
///
/// [`CATALOG`] lists speech best-first, and this is the one place that ordering
/// is load-bearing: it is the order the reconcile falls back through when the
/// wanted weights are not on disk yet and something has to serve the meeting.
pub fn speech_ids_best_first() -> Vec<&'static str> {
    CATALOG
        .iter()
        .filter(|e| e.kind == AssetKind::Speech && e.applies_here())
        .map(|e| e.id)
        .collect()
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

    /// The bytes the level downloads were fetched and hashed by hand before
    /// these lines were written. Pinning the numbers here is what makes a later
    /// edit to them a deliberate act rather than a typo.
    #[test]
    fn the_speech_model_is_the_one_that_was_downloaded_and_hashed() {
        let speech = entry(ids::SPEECH).unwrap();
        assert_eq!(speech.file_name, "ggml-large-v3.bin");
        assert_eq!(speech.bytes, 3_095_033_483);
        assert_eq!(
            speech.sha256,
            Some("64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2")
        );
        let accel = entry(ids::ACCEL).unwrap();
        assert_eq!(accel.bytes, 1_175_711_232);
        assert_eq!(
            accel.sha256,
            Some("47837be7594a29429ec08620043390c4d6d467f8bd362df09e9390ace76a55a4")
        );
        // Both come from the same pinned upstream commit.
        assert_eq!(speech.revision, accel.revision);
        assert!(speech.url.contains(WHISPER_REPO_REV));
        assert!(accel.url.contains(WHISPER_REPO_REV));
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
            // so the bundle name has to match the speech file it is paired with.
            let speech_id = e
                .pairs_with
                .expect("every companion names the speech file it belongs to");
            let speech = entry(speech_id).expect("and that speech file is catalogued");
            assert_eq!(speech.kind, AssetKind::Speech);
            let stem = speech.file_name.trim_end_matches(".bin");
            assert_eq!(dir, format!("{stem}-encoder.mlmodelc"));
        }
        // And every speech file has one, wanted or not: an old model that is
        // still serving a meeting needs its own companion, not the new one's.
        for e in CATALOG.iter().filter(|e| e.kind == AssetKind::Speech) {
            let paired = CATALOG.iter().find(|a| a.pairs_with == Some(e.id));
            assert!(paired.is_some(), "{} has no Apple companion", e.id);
        }
    }

    #[test]
    fn only_companions_pair_with_anything() {
        for e in CATALOG {
            if e.kind == AssetKind::SpeechAccelerator {
                assert!(e.pairs_with.is_some(), "{} pairs with nothing", e.id);
            } else {
                assert!(e.pairs_with.is_none(), "{} should pair with nothing", e.id);
            }
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
                assert_eq!(
                    accel.pairs_with,
                    Some(p.speech_id),
                    "the level's companion belongs to the level's speech file"
                );
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

    /// One model, everywhere. This is the test that would fail if somebody
    /// reintroduced a second level, a per-machine choice or a cheaper model for
    /// some lane (product decision of 2026-08-20).
    #[test]
    fn there_is_exactly_one_level_and_it_uses_the_full_model() {
        assert_eq!(PRESETS.len(), 1, "Echo offers no model choice");
        let only = &PRESETS[0];
        assert_eq!(only.id, DEFAULT_PRESET_ID);
        assert_eq!(only.speech_id, ids::SPEECH);
        assert_eq!(entry(ids::SPEECH).unwrap().file_name, "ggml-large-v3.bin");
        // Nothing anywhere may quietly reach for the smaller weights.
        assert!(
            !preset_asset_ids(DEFAULT_PRESET_ID).contains(&ids::SPEECH_TURBO),
            "the level must not want yesterday's weights"
        );
    }

    #[test]
    fn a_preset_needs_speech_detection_and_the_speaker_files() {
        let all = preset_asset_ids(DEFAULT_PRESET_ID);
        assert_eq!(all.first(), Some(&ids::SPEECH), "speech comes first");
        assert!(all.contains(&ids::DETECTOR));
        assert!(all.contains(&ids::SEGMENTER));
        assert!(all.contains(&ids::EMBEDDER));
        assert_eq!(
            all.contains(&ids::ACCEL),
            cfg!(target_os = "macos"),
            "the Apple companion is only fetched on Apple hardware"
        );
        // The 32 MB of speaker files come before the 1.2 GB speed-up: while they
        // are missing, a finished meeting cannot be split into people at all,
        // and a missing speed-up only costs speed. See `preset_asset_ids`.
        if let (Some(embedder), Some(accel)) = (
            all.iter().position(|id| *id == ids::EMBEDDER),
            all.iter().position(|id| *id == ids::ACCEL),
        ) {
            assert!(embedder < accel, "{all:?}");
        }

        // Recording only waits for speech plus detection.
        let required = preset_required_asset_ids(DEFAULT_PRESET_ID);
        assert_eq!(required, vec![ids::SPEECH, ids::DETECTOR]);
    }

    #[test]
    fn the_total_is_exactly_what_would_be_fetched_here() {
        let manual: i64 = preset_asset_ids(DEFAULT_PRESET_ID)
            .into_iter()
            .map(|id| entry(id).unwrap().bytes)
            .sum();
        assert_eq!(manual, preset_total_bytes(DEFAULT_PRESET_ID));
        // Big enough that the sentence has to say so honestly.
        assert!(preset_total_bytes(DEFAULT_PRESET_ID) > 3_000_000_000);
    }

    /// The honest number the UI quotes. If the catalog changes, this test is
    /// where the copy is reminded to change with it.
    #[cfg(target_os = "macos")]
    #[test]
    fn the_download_is_the_size_the_copy_says_it_is() {
        assert_eq!(preset_total_bytes(DEFAULT_PRESET_ID), 4_305_595_702);
        assert_eq!(
            human_bytes(preset_total_bytes(DEFAULT_PRESET_ID)),
            "4.3 GB",
            "src/lib/copy.ts quotes this number; both have to move together"
        );
    }

    // -----------------------------------------------------------------
    // What "old" means
    // -----------------------------------------------------------------

    #[test]
    fn yesterdays_weights_are_still_catalogued_so_they_can_be_found() {
        // Not a nicety: an id that vanished from the catalog is a file on
        // somebody's disk that nothing can account for or delete.
        for id in OBSOLETE_ASSET_IDS {
            let e = entry(id).unwrap_or_else(|| panic!("{id} left the catalog"));
            assert!(
                crate::asr::reconcile::is_supersedable(e.kind),
                "{id} is not a kind of file a newer model replaces, so the \
                 reconcile would never remove it"
            );
        }
    }

    /// The voice-print swap of 2026-08-20, pinned. Yesterday's network stays
    /// catalogued and is what the reconcile removes; the new one is what a fresh
    /// install fetches.
    #[test]
    fn the_voice_print_network_is_resnet34_and_camplus_is_only_there_to_be_removed() {
        let now = entry(ids::EMBEDDER).expect("the shipped voice-print network");
        assert_eq!(now.kind, AssetKind::SpeakerEmbedder);
        assert_eq!(now.file_name, "wespeaker-en-voxceleb-resnet34-lm.onnx");
        assert_eq!(now.bytes, 26_530_550);
        assert_eq!(
            now.sha256,
            Some("e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012")
        );

        let before = entry(ids::EMBEDDER_CAMPLUS).expect("still catalogued");
        assert_eq!(before.kind, AssetKind::SpeakerEmbedder);
        assert!(OBSOLETE_ASSET_IDS.contains(&ids::EMBEDDER_CAMPLUS));
        assert!(!SHARED_ASSET_IDS.contains(&ids::EMBEDDER_CAMPLUS));
        assert!(!preset_asset_ids(DEFAULT_PRESET_ID).contains(&ids::EMBEDDER_CAMPLUS));
        // Same trusted mirror, same pinned revision: the swap changed which file
        // is fetched, not who is trusted to serve it.
        assert_eq!(now.revision, before.revision);
    }

    #[test]
    fn the_obsolete_set_is_derived_from_the_level_and_agrees_with_the_list() {
        // Compared as sets: the derivation follows catalog layout, the constant
        // is grouped by generation, and neither ordering is the other's
        // business.
        let derived: HashSet<&str> = obsolete_asset_ids(DEFAULT_PRESET_ID).into_iter().collect();
        let expected: HashSet<&str> = OBSOLETE_ASSET_IDS
            .iter()
            .copied()
            .filter(|id| entry(id).is_some_and(CatalogEntry::applies_here))
            .collect();
        assert_eq!(derived, expected);
        let derived: Vec<&str> = obsolete_asset_ids(DEFAULT_PRESET_ID);

        // Nothing the level wants is ever obsolete, and nothing shared is.
        for id in preset_asset_ids(DEFAULT_PRESET_ID) {
            assert!(!derived.contains(&id), "{id} is both wanted and obsolete");
        }
        for id in SHARED_ASSET_IDS {
            assert!(
                !derived.contains(id),
                "{id} is shared by every level and must never be swept up"
            );
        }
    }

    #[test]
    fn every_catalogued_asset_is_either_wanted_or_obsolete() {
        let wanted = preset_asset_ids(DEFAULT_PRESET_ID);
        let obsolete = obsolete_asset_ids(DEFAULT_PRESET_ID);
        for e in CATALOG.iter().filter(|e| e.applies_here()) {
            let counted = wanted.contains(&e.id) as usize + obsolete.contains(&e.id) as usize;
            assert_eq!(counted, 1, "{} is in neither camp or in both", e.id);
        }
    }

    #[test]
    fn speech_is_listed_best_first_so_a_fallback_picks_the_best_thing_there() {
        let order = speech_ids_best_first();
        assert_eq!(
            order.first(),
            Some(&ids::SPEECH),
            "the wanted model has to come first, or a fallback would prefer an older one"
        );
        assert_eq!(
            order,
            vec![
                ids::SPEECH,
                ids::SPEECH_TURBO,
                ids::SPEECH_SMALL,
                ids::SPEECH_TINY
            ],
            "biggest first: turbo beats small beats tiny"
        );
    }

    #[test]
    fn a_companion_is_found_for_the_model_that_is_actually_serving() {
        if cfg!(target_os = "macos") {
            assert_eq!(accelerator_for(ids::SPEECH).map(|e| e.id), Some(ids::ACCEL));
            assert_eq!(
                accelerator_for(ids::SPEECH_TURBO).map(|e| e.id),
                Some(ids::ACCEL_TURBO),
                "an old model that is still serving loads its own companion"
            );
        } else {
            assert!(accelerator_for(ids::SPEECH).is_none());
        }
        assert!(accelerator_for("not-a-model").is_none());
    }

    // -----------------------------------------------------------------
    // Decoding
    // -----------------------------------------------------------------

    #[test]
    fn the_level_searches_beams_from_disk_and_falls_back_when_it_has_to() {
        for p in PRESETS {
            assert!(p.decode.uses_beam_search());
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
            let catchup = p.decode.catch_up();
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
            // The disk pass still tries harder than a live final — it just no
            // longer tries six times.
            assert!(
                catchup.max_attempts() > live.max_attempts(),
                "{} would decode less thoroughly from disk than live",
                p.id
            );
            assert_eq!(
                catchup.max_attempts(),
                CATCHUP_MAX_FALLBACKS + 1,
                "{} spends {} decodes on one window of disk audio",
                p.id,
                catchup.max_attempts()
            );
        }
        let everyday = default_preset().decode;
        assert_eq!(everyday.beam_size, 5, "catch-up keeps the wide beam");
        assert_eq!(
            everyday.live_final().beam_size,
            5,
            "a live final is decoded at catch-up quality"
        );
        assert_eq!(everyday.live_final_degraded().beam_size, 2);
        assert_eq!(
            everyday.catch_up().beam_size,
            5,
            "the cap is on the ladder, never on the beam"
        );
        assert_eq!(
            everyday.catch_up().max_attempts(),
            3,
            "0.0, 0.4, 0.8 on disk"
        );
        // The uncapped preset is what the cap is measured against: six decodes
        // of the same window is what the disk pass used to allow itself.
        assert_eq!(everyday.max_attempts(), 6, "0.0, 0.2, … 1.0 uncapped");
    }

    /// The loop guard has to be on every lane, or a transcript full of
    /// "ma è un po' figgito ma è un po' figgito" is what tells us months later
    /// (2026-08-24).
    #[test]
    fn every_lane_carries_the_loop_guard_and_the_two_that_can_retry_have_a_ladder() {
        let preset = default_preset().decode;
        for (lane, params) in [
            ("caption", preset.speculative()),
            ("live final", preset.live_final()),
            ("live final, behind", preset.live_final_degraded()),
            ("disk", preset.catch_up()),
        ] {
            assert_eq!(
                params.entropy_thold, LOOP_ENTROPY_THOLD,
                "the {lane} lane lost the loop guard"
            );
        }
        // A guard that catches a loop is only half of it: something has to
        // decode the window again. Both lanes that write the transcript can.
        assert_eq!(preset.live_final().max_attempts(), 2);
        assert_eq!(preset.catch_up().max_attempts(), 3);
        // The caption cannot, on purpose — it is replaced by the final either
        // way, and a caption nobody waits for is not a caption.
        assert_eq!(preset.speculative().max_attempts(), 1);
    }

    /// The silence gate, lane by lane. The direction of every one of these
    /// numbers is load-bearing, so they are pinned rather than described.
    #[test]
    fn the_lanes_disagree_about_silence_exactly_where_they_are_meant_to() {
        let preset = default_preset().decode;
        let caption = preset.speculative();
        let live = preset.live_final();
        let disk = preset.catch_up();

        // The caption is the quickest to call a window silence — but only by the
        // half of the gate that means "the model heard no speech".
        assert_eq!(caption.no_speech_thold, CAPTION_NO_SPEECH_THOLD);
        assert!(caption.no_speech_thold < live.no_speech_thold);
        // The other half is whisper.cpp's own, and stays there: the gate is an
        // AND, and a bar inside the band real speech decodes at would blank a
        // quiet participant's captions for the whole call rather than for the
        // three seconds until the final replaces them.
        assert_eq!(caption.logprob_thold, disk.logprob_thold);
        assert!(caption.logprob_thold < live.logprob_thold);

        // A live final asks a little more of itself than the disk pass, because
        // what it drops the disk pass reads again.
        assert_eq!(live.logprob_thold, LIVE_LOGPROB_THOLD);
        assert!(live.logprob_thold > disk.logprob_thold);
        assert_eq!(live.no_speech_thold, disk.no_speech_thold);

        // The disk pass is the last chance these words have: whisper.cpp's own
        // measured defaults, untouched.
        assert_eq!(disk.logprob_thold, -1.0);
        assert_eq!(disk.no_speech_thold, 0.6);

        // The safety valve narrows the beam and nothing else.
        let behind = preset.live_final_degraded();
        assert_eq!(behind.logprob_thold, live.logprob_thold);
        assert_eq!(behind.no_speech_thold, live.no_speech_thold);
    }

    // -----------------------------------------------------------------
    // Words a person reads
    // -----------------------------------------------------------------

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
        assert_eq!(human_bytes(4_305_595_702), "4.3 GB");
        assert_eq!(human_bytes(3_095_033_483), "3.1 GB");
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
