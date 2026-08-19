//! The download manager for everything Echo fetches.
//!
//! IMPLEMENTED-BY: asr agent (M3).
//!
//! Download rules (DESIGN §2 HTTP, review finding 15). Every one of these is a
//! requirement, not a nicety:
//! * check free space before starting, with headroom
//! * verify SHA-256 and the byte length from the catalog
//! * resume with Range plus ETag, and start over if the ETag moved
//! * write to a temp file, fsync, then rename into place
//! * clean up stale partial files on launch
//!
//! macOS also needs a matching `*-encoder.mlmodelc` next to each speech asset.
//! It only speeds up the encoder, so a missing one degrades speed, never
//! correctness (DESIGN §2, review finding 1).
//!
//! The `models` table, not [`crate::asr::catalog`], is what a download reads:
//! the catalog seeds the table and the table is the record of what we asked for
//! and what arrived. That is what makes provenance auditable after the fact.
//!
//! User-facing copy lives in the frontend. This module reports bytes and
//! [`crate::types::AssetKind`]; it never writes a sentence for the UI.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use futures::StreamExt;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::asr::catalog::{self, Archive, CatalogEntry};
use crate::asr::AsrError;
use crate::db::{repo, Db};
use crate::paths::AppPaths;
use crate::settings;
use crate::types::{AccuracyLevel, AssetKind, Id, ModelInfo, SpeechReadiness};

/// Progress callback. Called often, so it must be cheap and must not block.
pub type ProgressFn = Box<dyn Fn(DownloadProgress) + Send + Sync>;

/// Never report more often than this, however fast the bytes arrive
/// (DESIGN §3: UI events are rate-capped by the emitter).
const PROGRESS_INTERVAL: Duration = Duration::from_millis(250);

/// Free space we insist on keeping beyond the download itself, so finishing a
/// download never fills the disk a recording is writing to.
const DISK_HEADROOM_BYTES: u64 = 256 * 1024 * 1024;

/// Chunk size for re-hashing the part of a resumed file we already have.
const REHASH_CHUNK: usize = 1024 * 1024;

#[derive(Debug, Clone, Default)]
pub struct DownloadProgress {
    /// Which catalog entry this is about.
    pub asset_id: String,
    /// The preset the download belongs to, when it was started for one.
    pub level_id: Option<String>,
    pub received_bytes: i64,
    pub total_bytes: i64,
    pub bytes_per_second: f64,
    pub done: bool,
}

// ---------------------------------------------------------------------------
// In-flight downloads
//
// Two small registries, both keyed by asset id:
// * a per-asset lock, so a second request attaches to the download already
//   running instead of starting a competing one;
// * a cancel flag, so `cancel_download` can reach a stream that is mid-flight.
// ---------------------------------------------------------------------------

type Slot = Arc<tokio::sync::Mutex<()>>;

static SLOTS: LazyLock<Mutex<HashMap<String, Slot>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

static CANCELS: LazyLock<Mutex<HashMap<String, Arc<AtomicBool>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn slot_for(asset_id: &str) -> Slot {
    let mut slots = SLOTS.lock().expect("download slot registry poisoned");
    slots.entry(asset_id.to_string()).or_default().clone()
}

fn begin_cancellable(asset_id: &str) -> Arc<AtomicBool> {
    let flag = Arc::new(AtomicBool::new(false));
    CANCELS
        .lock()
        .expect("cancel registry poisoned")
        .insert(asset_id.to_string(), flag.clone());
    flag
}

fn end_cancellable(asset_id: &str) {
    CANCELS
        .lock()
        .expect("cancel registry poisoned")
        .remove(asset_id);
}

/// Is any download running right now? Feeds `SpeechReadiness::downloading`.
pub fn any_download_in_flight() -> bool {
    !CANCELS.lock().expect("cancel registry poisoned").is_empty()
}

// ---------------------------------------------------------------------------
// Catalog ⇄ database
// ---------------------------------------------------------------------------

/// Read the signed catalog and write it into the `models` table.
///
/// The catalog ships with the app and is versioned; a newer revision may add
/// assets or correct a hash but must never silently repoint an installed one to
/// a different file. When a hash *does* change under an installed row, the row
/// is marked not-installed so the next readiness check fetches the right bytes
/// instead of loading the wrong ones.
pub async fn sync_catalog(db: &Db) -> Result<Vec<ModelInfo>, AsrError> {
    let mut out = Vec::new();
    for entry in catalog::CATALOG.iter().filter(|e| e.applies_here()) {
        let existing = repo::get_model(db, entry.id).await?;
        let catalog_sha = entry.sha256.unwrap_or_default();

        if let Some(prev) = &existing {
            let moved = !prev.sha256.is_empty()
                && !catalog_sha.is_empty()
                && !prev.sha256.eq_ignore_ascii_case(catalog_sha);
            if moved && prev.installed {
                tracing::warn!(
                    asset = entry.id,
                    "the catalog now expects different bytes for an installed asset; \
                     marking it for a fresh download"
                );
                repo::set_model_installed(db, entry.id, false, None).await?;
            }
        }

        let row = ModelInfo {
            id: entry.id.to_string(),
            kind: entry.kind,
            name: entry.name.to_string(),
            url: entry.url.to_string(),
            sha256: catalog_sha.to_string(),
            bytes: entry.bytes,
            license: Some(entry.license.to_string()),
            revision: Some(entry.revision.to_string()),
            // upsert_model deliberately leaves installed/path alone on
            // conflict, so these only matter for a brand-new row.
            installed: false,
            path: None,
        };
        repo::upsert_model(db, &row).await?;
        if let Some(saved) = repo::get_model(db, entry.id).await? {
            out.push(saved);
        }
    }
    repo::set_setting(
        db,
        settings::keys::MODEL_CATALOG_REVISION,
        catalog::CATALOG_REVISION,
    )
    .await?;
    Ok(out)
}

/// Seed the table from the catalog if it has not been seeded, or if the shipped
/// catalog is newer than the one the database was built from.
///
/// One settings read in the common case. Every read path calls this, so nothing
/// depends on a caller remembering to sync at startup: an empty `models` table
/// would otherwise make a perfectly healthy install report that there is nothing
/// to download and nothing installed.
pub async fn ensure_catalogued(db: &Db) -> Result<(), AsrError> {
    let stored = repo::get_setting(db, settings::keys::MODEL_CATALOG_REVISION).await?;
    if stored.as_deref() != Some(catalog::CATALOG_REVISION) {
        sync_catalog(db).await?;
    }
    Ok(())
}

/// Make the `installed` column agree with the disk. Cheap, and worth running
/// before anything trusts the table: a person can always delete a file behind
/// our back, and a half-finished install must not read as ready.
pub async fn refresh_installed(db: &Db, paths: &AppPaths) -> Result<(), AsrError> {
    ensure_catalogued(db).await?;
    for entry in catalog::CATALOG.iter().filter(|e| e.applies_here()) {
        let row = match repo::get_model(db, entry.id).await? {
            Some(r) => r,
            None => continue,
        };
        let path = install_path(paths, entry);
        let there = exists_on_disk(&path, entry);
        if there != row.installed || (there && row.path.as_deref() != path.to_str()) {
            let recorded = there.then(|| path_str(&path));
            repo::set_model_installed(db, entry.id, there, recorded.as_deref()).await?;
        }
    }
    Ok(())
}

/// The quality presets the person picks between, named in plain words.
///
/// Never a model identifier here; those live behind Settings → Advanced.
pub async fn list_accuracy_levels(db: &Db) -> Result<Vec<AccuracyLevel>, AsrError> {
    ensure_catalogued(db).await?;
    let selected = selected_level_id(db).await?;
    let recommended = catalog::recommended_preset_id(total_memory_bytes());
    let rows = model_rows(db).await?;

    let mut out = Vec::with_capacity(catalog::PRESETS.len());
    for preset in catalog::PRESETS {
        let asset_ids = catalog::preset_asset_ids(preset.id);
        let installed = asset_ids
            .iter()
            .all(|id| rows.get(*id).is_some_and(|m| m.installed));
        out.push(AccuracyLevel {
            id: preset.id.to_string(),
            name: preset.name.to_string(),
            description: catalog::preset_description(preset.id),
            download_bytes: catalog::preset_total_bytes(preset.id),
            installed,
            selected: preset.id == selected,
            recommended: preset.id == recommended,
            asset_ids: asset_ids.iter().map(|s| s.to_string()).collect(),
        });
    }
    Ok(out)
}

/// Everything the selected preset needs, and whether it is on disk.
pub async fn readiness(db: &Db, paths: &AppPaths) -> Result<SpeechReadiness, AsrError> {
    refresh_installed(db, paths).await?;
    let level_id = selected_level_id(db).await?;
    let rows = model_rows(db).await?;

    // Recording only waits for speech plus speech detection; the speaker files
    // are used by a pass that runs after the meeting ends.
    let required = catalog::preset_required_asset_ids(&level_id);
    let ready = required
        .iter()
        .all(|id| rows.get(*id).is_some_and(|m| m.installed));

    let remaining_bytes = catalog::preset_asset_ids(&level_id)
        .iter()
        .filter(|id| !rows.get(**id).is_some_and(|m| m.installed))
        .filter_map(|id| catalog::entry(id))
        .map(|e| e.bytes)
        .sum();

    Ok(SpeechReadiness {
        ready,
        downloading: any_download_in_flight(),
        remaining_bytes,
        level_id: Some(level_id),
        loaded: false,
    })
}

/// Download every asset a preset needs, in order, resuming what is partial.
///
/// Idempotent: calling it while a download is running attaches to that
/// download instead of starting a second one.
pub async fn download_level(
    db: &Db,
    paths: &AppPaths,
    level_id: &str,
    on_progress: ProgressFn,
) -> Result<(), AsrError> {
    let preset =
        catalog::preset(level_id).ok_or_else(|| AsrError::UnknownAsset(level_id.to_string()))?;
    let asset_ids = catalog::preset_asset_ids(preset.id);

    // Progress is reported for the preset as a whole, so onboarding can show one
    // bar for "what Echo needs" rather than one per file.
    let level_total: i64 = asset_ids
        .iter()
        .filter_map(|id| catalog::entry(id))
        .map(|e| e.bytes)
        .sum();
    let mut done_before: i64 = 0;
    let level_id_owned = preset.id.to_string();
    let shared: Arc<dyn Fn(DownloadProgress) + Send + Sync> = Arc::from(on_progress);

    for asset_id in asset_ids {
        let entry_bytes = catalog::entry(asset_id).map(|e| e.bytes).unwrap_or(0);
        let base = done_before;
        let sink = shared.clone();
        let level = level_id_owned.clone();
        let per_asset: ProgressFn = Box::new(move |p: DownloadProgress| {
            (*sink)(DownloadProgress {
                asset_id: p.asset_id.clone(),
                level_id: Some(level.clone()),
                received_bytes: base + p.received_bytes,
                total_bytes: level_total,
                bytes_per_second: p.bytes_per_second,
                done: false,
            });
        });
        let id = asset_id.to_string();
        download_asset(db, paths, &id, per_asset).await?;
        done_before += entry_bytes;
    }

    (*shared)(DownloadProgress {
        asset_id: String::new(),
        level_id: Some(level_id_owned),
        received_bytes: level_total,
        total_bytes: level_total,
        bytes_per_second: 0.0,
        done: true,
    });
    Ok(())
}

/// Download one catalog entry.
pub async fn download_asset(
    db: &Db,
    paths: &AppPaths,
    asset_id: &Id,
    on_progress: ProgressFn,
) -> Result<ModelInfo, AsrError> {
    let entry = catalog::entry(asset_id).ok_or_else(|| AsrError::UnknownAsset(asset_id.clone()))?;
    if !entry.applies_here() {
        return Err(AsrError::NotUsedHere(asset_id.clone()));
    }

    // Attach rather than compete: whoever holds the slot is already fetching
    // these bytes, and when they are finished the check below short-circuits.
    let slot = slot_for(asset_id);
    let _guard = slot.lock().await;

    // The row is the record of what we asked for. Seed it if this is the first
    // time anything touched the catalog.
    ensure_catalogued(db).await?;
    let row = repo::get_model(db, asset_id)
        .await?
        .ok_or_else(|| AsrError::UnknownAsset(asset_id.clone()))?;

    let destination = install_path(paths, entry);
    if exists_on_disk(&destination, entry) {
        mark_installed(db, entry, &destination, &row.sha256).await?;
        on_progress(DownloadProgress {
            asset_id: asset_id.clone(),
            level_id: None,
            received_bytes: row.bytes,
            total_bytes: row.bytes,
            bytes_per_second: 0.0,
            done: true,
        });
        return current_row(db, asset_id).await;
    }

    ensure_dir(&paths.assets_dir).await?;
    ensure_dir(&paths.tmp_dir).await?;

    // A bundle needs room for the archive *and* what comes out of it.
    let space_multiplier = if entry.is_bundle() { 2 } else { 1 };
    let download_target = paths.assets_dir.join(entry.file_name);
    let partial = paths.asset_partial_path(asset_id);
    let expected_sha = (!row.sha256.is_empty()).then_some(row.sha256.as_str());

    let cancel = begin_cancellable(asset_id);
    let reporter_id = asset_id.clone();
    let started = Instant::now();
    let report: Box<dyn Fn(u64, u64) + Send + Sync> = Box::new(move |received, total| {
        let secs = started.elapsed().as_secs_f64();
        on_progress(DownloadProgress {
            asset_id: reporter_id.clone(),
            level_id: None,
            received_bytes: received as i64,
            total_bytes: total as i64,
            bytes_per_second: if secs > 0.0 {
                received as f64 / secs
            } else {
                0.0
            },
            done: false,
        });
    });

    let outcome = fetch_verified(
        &http_client()?,
        &row.url,
        &partial,
        &download_target,
        row.bytes,
        expected_sha,
        row.bytes * space_multiplier,
        &cancel,
        report.as_ref(),
    )
    .await;
    end_cancellable(asset_id);
    let outcome = outcome?;
    tracing::info!(
        asset = %asset_id,
        bytes = outcome.bytes,
        resumed_from = outcome.resumed_from,
        "finished downloading part of what Echo needs to understand speech"
    );

    // Unpack the Apple encoder companion next to the speech file, then throw the
    // archive away: it is a third of a gigabyte we do not need twice.
    if let Archive::ZipBundle { dir_name } = entry.archive {
        let into = paths.assets_dir.clone();
        let archive = download_target.clone();
        let expect = into.join(dir_name);
        extract_bundle(&archive, &into, &expect).await?;
        let _ = tokio::fs::remove_file(&archive).await;
    }

    mark_installed(db, entry, &destination, &outcome.sha256).await?;
    current_row(db, asset_id).await
}

/// Stop an in-flight download. The partial file stays so it can resume.
pub async fn cancel_download(asset_id: &Id) -> Result<(), AsrError> {
    if let Some(flag) = CANCELS
        .lock()
        .expect("cancel registry poisoned")
        .get(asset_id)
    {
        flag.store(true, Ordering::SeqCst);
    }
    Ok(())
}

/// Delete an installed asset to free space.
pub async fn remove_asset(db: &Db, paths: &AppPaths, asset_id: &Id) -> Result<(), AsrError> {
    let entry = catalog::entry(asset_id).ok_or_else(|| AsrError::UnknownAsset(asset_id.clone()))?;
    let path = install_path(paths, entry);
    if entry.is_bundle() {
        let _ = tokio::fs::remove_dir_all(&path).await;
    } else {
        let _ = tokio::fs::remove_file(&path).await;
    }
    // Any half-finished attempt goes too, or a resume would append to bytes
    // nobody wants any more.
    let partial = paths.asset_partial_path(asset_id);
    let _ = tokio::fs::remove_file(&partial).await;
    let _ = tokio::fs::remove_file(meta_path(&partial)).await;
    repo::set_model_installed(db, asset_id, false, None).await?;
    Ok(())
}

/// Path of an installed asset of this kind for the selected preset.
pub async fn installed_path(db: &Db, kind: AssetKind) -> Result<Option<PathBuf>, AsrError> {
    ensure_catalogued(db).await?;
    let level_id = selected_level_id(db).await?;
    let wanted = catalog::preset_asset_ids(&level_id)
        .into_iter()
        .filter_map(catalog::entry)
        .find(|e| e.kind == kind);
    let Some(entry) = wanted else {
        return Ok(None);
    };
    let Some(row) = repo::get_model(db, entry.id).await? else {
        return Ok(None);
    };
    if !row.installed {
        return Ok(None);
    }
    Ok(row.path.map(PathBuf::from))
}

/// Delete leftover `.part` files that no longer match anything in the catalog.
/// Runs at launch.
pub async fn clean_stale_partials(db: &Db, paths: &AppPaths) -> Result<u64, AsrError> {
    let mut removed = 0u64;
    let mut dir = match tokio::fs::read_dir(&paths.tmp_dir).await {
        Ok(d) => d,
        // No scratch directory yet is not a problem, it means nothing to clean.
        Err(_) => return Ok(0),
    };
    while let Ok(Some(item)) = dir.next_entry().await {
        let path = item.path();
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };
        let asset_id = match name.strip_suffix(".part.meta") {
            Some(id) => id,
            None => match name.strip_suffix(".part") {
                Some(id) => id,
                None => continue,
            },
        };
        let known = catalog::entry(asset_id).is_some_and(CatalogEntry::applies_here);
        let already_installed = match repo::get_model(db, asset_id).await {
            Ok(Some(row)) => row.installed,
            _ => false,
        };
        // Keep only partials that still lead somewhere: a catalogued asset we
        // have not installed yet.
        let dead_weight = !known || already_installed;
        if dead_weight && tokio::fs::remove_file(&path).await.is_ok() {
            removed += 1;
        }
    }
    Ok(removed)
}

/// Verify an installed file still matches its catalog hash. Used by the
/// diagnostics screen and after a failed load.
pub async fn verify_asset(db: &Db, asset_id: &Id) -> Result<bool, AsrError> {
    let entry = catalog::entry(asset_id).ok_or_else(|| AsrError::UnknownAsset(asset_id.clone()))?;
    let Some(row) = repo::get_model(db, asset_id).await? else {
        return Ok(false);
    };
    let Some(path) = row.path.as_deref().map(PathBuf::from) else {
        return Ok(false);
    };
    if !row.installed || !exists_on_disk(&path, entry) {
        return Ok(false);
    }
    // A directory bundle cannot be re-hashed once unpacked, so "it is there and
    // it is not empty" is the strongest honest answer.
    if entry.is_bundle() {
        return Ok(true);
    }
    if row.sha256.is_empty() {
        return Ok(true);
    }
    let digest = hash_file(&path).await?;
    Ok(digest.eq_ignore_ascii_case(&row.sha256))
}

// ---------------------------------------------------------------------------
// The download itself
// ---------------------------------------------------------------------------

/// What a completed fetch produced.
#[derive(Debug, Clone)]
pub(crate) struct FetchOutcome {
    pub bytes: u64,
    /// Lowercase hex SHA-256 of the whole file, computed while streaming. This
    /// is what trust-on-first-use records.
    pub sha256: String,
    /// Bytes that were already on disk and did not have to be fetched again.
    pub resumed_from: u64,
}

/// Sidecar next to a `.part`, so a resume knows what it is resuming.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct PartialMeta {
    url: String,
    etag: Option<String>,
    total: Option<u64>,
}

fn meta_path(partial: &Path) -> PathBuf {
    let mut s = partial.as_os_str().to_os_string();
    s.push(".meta");
    PathBuf::from(s)
}

fn http_client() -> Result<reqwest::Client, AsrError> {
    reqwest::Client::builder()
        // Downloads are long; a stalled connection has to give up eventually,
        // but the overall transfer must not be capped.
        .connect_timeout(Duration::from_secs(30))
        .read_timeout(Duration::from_secs(60))
        .user_agent(concat!("Echo/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| AsrError::Download(e.to_string()))
}

/// Fetch `url` into `destination`, resuming and verifying.
///
/// The whole integrity contract lives here:
/// 1. resume from `partial` when its sidecar says it belongs to this URL, using
///    `Range` guarded by `If-Range` so a changed file restarts instead of
///    producing a spliced mess;
/// 2. refuse to start when the disk cannot hold the result plus headroom;
/// 3. hash every byte, including the bytes we resumed over;
/// 4. check the length and (when known) the hash *before* anything is installed;
/// 5. fsync the temp file, then rename it into place, then fsync the directory,
///    so a crash leaves either the old state or the new one.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn fetch_verified(
    client: &reqwest::Client,
    url: &str,
    partial: &Path,
    destination: &Path,
    expected_bytes: i64,
    expected_sha256: Option<&str>,
    space_needed: i64,
    cancel: &AtomicBool,
    on_progress: &(dyn Fn(u64, u64) + Send + Sync),
) -> Result<FetchOutcome, AsrError> {
    let meta = meta_path(partial);

    // --- 1. what do we already have? -------------------------------------
    let mut have = tokio::fs::metadata(partial)
        .await
        .map(|m| m.len())
        .unwrap_or(0);
    let saved: Option<PartialMeta> = match tokio::fs::read(&meta).await {
        Ok(bytes) => serde_json::from_slice(&bytes).ok(),
        Err(_) => None,
    };
    let mut etag = saved
        .as_ref()
        .filter(|m| m.url == url)
        .and_then(|m| m.etag.clone());
    if have > 0 && etag.is_none() {
        // Without an ETag a resume is a guess, and a wrong guess corrupts the
        // file. Start again instead.
        discard_partial(partial, &meta).await;
        have = 0;
    }
    if expected_bytes > 0 && have > expected_bytes as u64 {
        discard_partial(partial, &meta).await;
        have = 0;
        etag = None;
    }

    // --- 2. is there room? -----------------------------------------------
    if space_needed > 0 {
        let dir = partial.parent().unwrap_or(Path::new("."));
        let free = crate::paths::free_space_bytes(dir);
        let needed = (space_needed as u64).saturating_sub(have) + DISK_HEADROOM_BYTES;
        // free == 0 means "could not tell", never "full" (see paths.rs).
        if free > 0 && free < needed {
            return Err(AsrError::NotEnoughSpace {
                needed,
                available: free,
            });
        }
    }

    if cancel.load(Ordering::SeqCst) {
        return Err(AsrError::Cancelled);
    }

    // --- 3. ask for the rest ---------------------------------------------
    let mut request = client
        .get(url)
        // Range and content encoding do not mix; ask for the bytes as stored.
        .header(reqwest::header::ACCEPT_ENCODING, "identity");
    if have > 0 {
        request = request.header(reqwest::header::RANGE, format!("bytes={have}-"));
        if let Some(tag) = &etag {
            request = request.header(reqwest::header::IF_RANGE, tag.clone());
        }
    }
    let response = request
        .send()
        .await
        .map_err(|e| AsrError::Download(e.to_string()))?;

    let status = response.status();
    let mut append = false;
    if status == reqwest::StatusCode::PARTIAL_CONTENT {
        append = true;
    } else if status.is_success() {
        // The server ignored the range, or the file moved and If-Range failed.
        // Either way we are getting the whole thing.
        have = 0;
    } else if status == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
        discard_partial(partial, &meta).await;
        return Err(AsrError::Download(
            "the part already downloaded no longer matches the file".to_string(),
        ));
    } else {
        return Err(AsrError::Download(format!("server answered {status}")));
    }

    let new_etag = response
        .headers()
        .get(reqwest::header::ETAG)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let total = total_length(&response, have, expected_bytes);

    write_meta(
        &meta,
        &PartialMeta {
            url: url.to_string(),
            etag: new_etag.or(etag),
            total: Some(total),
        },
    )
    .await?;

    // --- 4. hash everything, including what we resumed over ---------------
    let mut hasher = Sha256::new();
    let mut file = if append {
        hash_prefix(partial, have, &mut hasher).await?;
        tokio::fs::OpenOptions::new()
            .append(true)
            .open(partial)
            .await?
    } else {
        if let Some(parent) = partial.parent() {
            ensure_dir(parent).await?;
        }
        tokio::fs::File::create(partial).await?
    };

    let mut received = have;
    let mut last_report = Instant::now();
    let mut stream = response.bytes_stream();
    on_progress(received, total);
    while let Some(chunk) = stream.next().await {
        if cancel.load(Ordering::SeqCst) {
            let _ = file.flush().await;
            let _ = file.sync_all().await;
            // The partial and its sidecar stay behind on purpose: that is what
            // makes "resume" mean something.
            return Err(AsrError::Cancelled);
        }
        let chunk = chunk.map_err(|e| AsrError::Download(e.to_string()))?;
        hasher.update(&chunk);
        file.write_all(&chunk).await?;
        received += chunk.len() as u64;
        if last_report.elapsed() >= PROGRESS_INTERVAL {
            last_report = Instant::now();
            on_progress(received, total.max(received));
        }
    }
    file.flush().await?;
    file.sync_all().await?;
    drop(file);
    on_progress(received, total.max(received));

    // --- 5. check before installing ---------------------------------------
    if expected_bytes > 0 && received != expected_bytes as u64 {
        discard_partial(partial, &meta).await;
        return Err(AsrError::Download(format!(
            "expected {expected_bytes} bytes but received {received}"
        )));
    }
    let digest = hex(hasher.finalize().as_slice());
    if let Some(expected) = expected_sha256 {
        if !expected.eq_ignore_ascii_case(&digest) {
            // Damaged or substituted. Throwing the partial away is the point:
            // resuming would keep the bad bytes forever.
            discard_partial(partial, &meta).await;
            return Err(AsrError::IntegrityCheckFailed);
        }
    }

    // --- 6. atomic install -------------------------------------------------
    if let Some(parent) = destination.parent() {
        ensure_dir(parent).await?;
    }
    tokio::fs::rename(partial, destination).await?;
    fsync_dir(destination.parent().unwrap_or(Path::new("."))).await;
    let _ = tokio::fs::remove_file(&meta).await;

    Ok(FetchOutcome {
        bytes: received,
        sha256: digest,
        resumed_from: have,
    })
}

fn total_length(response: &reqwest::Response, have: u64, expected_bytes: i64) -> u64 {
    // Content-Range wins, because on a 206 Content-Length is only the tail.
    if let Some(range) = response
        .headers()
        .get(reqwest::header::CONTENT_RANGE)
        .and_then(|v| v.to_str().ok())
    {
        if let Some(total) = range.rsplit('/').next().and_then(|t| t.parse::<u64>().ok()) {
            return total;
        }
    }
    if let Some(len) = response.content_length() {
        return have + len;
    }
    expected_bytes.max(0) as u64
}

async fn hash_prefix(path: &Path, len: u64, hasher: &mut Sha256) -> Result<(), AsrError> {
    let mut file = tokio::fs::File::open(path).await?;
    let mut left = len;
    let mut buf = vec![0u8; REHASH_CHUNK];
    while left > 0 {
        let want = std::cmp::min(left as usize, buf.len());
        let read = file.read(&mut buf[..want]).await?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
        left -= read as u64;
    }
    Ok(())
}

async fn write_meta(path: &Path, meta: &PartialMeta) -> Result<(), AsrError> {
    let bytes = serde_json::to_vec(meta).map_err(|e| AsrError::Io(e.to_string()))?;
    tokio::fs::write(path, bytes).await?;
    Ok(())
}

async fn discard_partial(partial: &Path, meta: &Path) {
    let _ = tokio::fs::remove_file(partial).await;
    let _ = tokio::fs::remove_file(meta).await;
}

/// Renaming is only atomic once the directory entry itself is on the platter.
async fn fsync_dir(dir: &Path) {
    let dir = dir.to_path_buf();
    let _ = tokio::task::spawn_blocking(move || {
        if let Ok(handle) = std::fs::File::open(&dir) {
            let _ = handle.sync_all();
        }
    })
    .await;
}

/// Unpack a zipped directory bundle. macOS ships `ditto`, which is the only
/// tool guaranteed to reproduce a `.mlmodelc` bundle faithfully; `unzip` is the
/// fallback everywhere else.
async fn extract_bundle(archive: &Path, into: &Path, expect: &Path) -> Result<(), AsrError> {
    let _ = tokio::fs::remove_dir_all(expect).await;
    let archive = archive.to_path_buf();
    let into = into.to_path_buf();
    let expect_owned = expect.to_path_buf();

    let result = tokio::task::spawn_blocking(move || -> Result<(), String> {
        let attempts: &[(&str, Vec<std::ffi::OsString>)] = &[
            #[cfg(target_os = "macos")]
            (
                "/usr/bin/ditto",
                vec![
                    "-x".into(),
                    "-k".into(),
                    archive.clone().into_os_string(),
                    into.clone().into_os_string(),
                ],
            ),
            (
                "unzip",
                vec![
                    "-q".into(),
                    "-o".into(),
                    archive.clone().into_os_string(),
                    "-d".into(),
                    into.clone().into_os_string(),
                ],
            ),
        ];
        let mut last = String::from("no extraction tool available");
        for (program, args) in attempts {
            match std::process::Command::new(program).args(args).status() {
                Ok(status) if status.success() => return Ok(()),
                Ok(status) => last = format!("{program} exited with {status}"),
                Err(e) => last = format!("{program}: {e}"),
            }
        }
        Err(last)
    })
    .await
    .map_err(|e| AsrError::Io(e.to_string()))?;

    result.map_err(AsrError::Io)?;
    if !expect_owned.is_dir() {
        return Err(AsrError::Io(format!(
            "{} was not in the archive",
            expect_owned.display()
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Small shared helpers
// ---------------------------------------------------------------------------

/// Where an entry ends up once installed: the file, or the unpacked bundle.
pub fn install_path(paths: &AppPaths, entry: &CatalogEntry) -> PathBuf {
    paths.assets_dir.join(entry.installed_name())
}

fn exists_on_disk(path: &Path, entry: &CatalogEntry) -> bool {
    if entry.is_bundle() {
        path.is_dir()
            && std::fs::read_dir(path)
                .map(|mut d| d.next().is_some())
                .unwrap_or(false)
    } else {
        path.is_file()
    }
}

fn path_str(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

async fn ensure_dir(dir: &Path) -> Result<(), AsrError> {
    tokio::fs::create_dir_all(dir).await?;
    Ok(())
}

async fn mark_installed(
    db: &Db,
    entry: &CatalogEntry,
    path: &Path,
    sha256: &str,
) -> Result<(), AsrError> {
    // Trust on first use: when the catalog had no hash, the one we computed
    // becomes the hash this install is checked against from now on.
    if !sha256.is_empty() {
        if let Some(mut row) = repo::get_model(db, entry.id).await? {
            if row.sha256.is_empty() {
                row.sha256 = sha256.to_ascii_lowercase();
                repo::upsert_model(db, &row).await?;
            }
        }
    }
    repo::set_model_installed(db, entry.id, true, Some(&path_str(path))).await?;
    Ok(())
}

async fn current_row(db: &Db, asset_id: &str) -> Result<ModelInfo, AsrError> {
    repo::get_model(db, asset_id)
        .await?
        .ok_or_else(|| AsrError::UnknownAsset(asset_id.to_string()))
}

async fn model_rows(db: &Db) -> Result<HashMap<String, ModelInfo>, AsrError> {
    Ok(repo::list_models(db, None)
        .await?
        .into_iter()
        .map(|m| (m.id.clone(), m))
        .collect())
}

async fn selected_level_id(db: &Db) -> Result<String, AsrError> {
    let stored = repo::get_setting(db, settings::keys::ACCURACY_LEVEL_ID).await?;
    let id = stored.filter(|v| !v.is_empty()).unwrap_or_default();
    Ok(catalog::preset_or_default(&id).id.to_string())
}

/// Installed memory, used only to suggest a preset. 0 when it cannot be read,
/// which the caller must treat as "do not downgrade anybody".
pub fn total_memory_bytes() -> u64 {
    let mut system = sysinfo::System::new_with_specifics(
        sysinfo::RefreshKind::nothing()
            .with_memory(sysinfo::MemoryRefreshKind::nothing().with_ram()),
    );
    system.refresh_memory_specifics(sysinfo::MemoryRefreshKind::nothing().with_ram());
    system.total_memory()
}

async fn hash_file(path: &Path) -> Result<String, AsrError> {
    let mut hasher = Sha256::new();
    let len = tokio::fs::metadata(path).await?.len();
    hash_prefix(path, len, &mut hasher).await?;
    Ok(hex(hasher.finalize().as_slice()))
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::connect_in_memory;
    use wiremock::matchers::{header, method, path as path_matcher};
    use wiremock::{Mock, MockServer, Request, ResponseTemplate};

    fn sha_of(bytes: &[u8]) -> String {
        let mut h = Sha256::new();
        h.update(bytes);
        hex(h.finalize().as_slice())
    }

    fn body(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i % 251) as u8).collect()
    }

    fn no_progress() -> Box<dyn Fn(u64, u64) + Send + Sync> {
        Box::new(|_, _| {})
    }

    struct Fixture {
        _dir: tempfile::TempDir,
        paths: AppPaths,
    }

    fn fixture() -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let paths = AppPaths::rooted_at(dir.path().join("app"), None);
        paths.ensure().unwrap();
        Fixture { _dir: dir, paths }
    }

    // -----------------------------------------------------------------
    // fetch_verified: the integrity contract
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn a_download_is_hashed_then_renamed_into_place() {
        let fx = fixture();
        let bytes = body(4096);
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path_matcher("/asset.bin"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(bytes.clone()))
            .mount(&server)
            .await;

        let partial = fx.paths.asset_partial_path("asset");
        let destination = fx.paths.assets_dir.join("asset.bin");
        let out = fetch_verified(
            &http_client().unwrap(),
            &format!("{}/asset.bin", server.uri()),
            &partial,
            &destination,
            bytes.len() as i64,
            Some(&sha_of(&bytes)),
            bytes.len() as i64,
            &AtomicBool::new(false),
            no_progress().as_ref(),
        )
        .await
        .unwrap();

        assert_eq!(out.bytes, bytes.len() as u64);
        assert_eq!(out.resumed_from, 0);
        assert_eq!(tokio::fs::read(&destination).await.unwrap(), bytes);
        assert!(!partial.exists(), "the temp file is gone once renamed");
        assert!(!meta_path(&partial).exists(), "and so is its sidecar");
    }

    #[tokio::test]
    async fn a_resumed_download_asks_only_for_the_rest() {
        let fx = fixture();
        let bytes = body(8192);
        let (first, rest) = bytes.split_at(3000);
        let etag = "\"abc123\"";

        let partial = fx.paths.asset_partial_path("asset");
        tokio::fs::write(&partial, first).await.unwrap();
        write_meta(
            &meta_path(&partial),
            &PartialMeta {
                url: "PLACEHOLDER".into(),
                etag: Some(etag.into()),
                total: Some(bytes.len() as u64),
            },
        )
        .await
        .unwrap();

        let server = MockServer::start().await;
        let url = format!("{}/asset.bin", server.uri());
        // Rewrite the sidecar now that we know the URL.
        write_meta(
            &meta_path(&partial),
            &PartialMeta {
                url: url.clone(),
                etag: Some(etag.into()),
                total: Some(bytes.len() as u64),
            },
        )
        .await
        .unwrap();

        Mock::given(method("GET"))
            .and(path_matcher("/asset.bin"))
            .and(header("range", "bytes=3000-"))
            .and(header("if-range", etag))
            .respond_with(
                ResponseTemplate::new(206)
                    .append_header("Content-Range", "bytes 3000-8191/8192")
                    .append_header("ETag", etag)
                    .set_body_bytes(rest.to_vec()),
            )
            .mount(&server)
            .await;

        let destination = fx.paths.assets_dir.join("asset.bin");
        let out = fetch_verified(
            &http_client().unwrap(),
            &url,
            &partial,
            &destination,
            bytes.len() as i64,
            Some(&sha_of(&bytes)),
            bytes.len() as i64,
            &AtomicBool::new(false),
            no_progress().as_ref(),
        )
        .await
        .unwrap();

        assert_eq!(out.resumed_from, 3000, "the bytes on disk were kept");
        assert_eq!(
            tokio::fs::read(&destination).await.unwrap(),
            bytes,
            "the resumed file hashes and reads as the whole thing"
        );
    }

    #[tokio::test]
    async fn a_partial_without_a_sidecar_starts_over_rather_than_guessing() {
        let fx = fixture();
        let bytes = body(2048);
        let partial = fx.paths.asset_partial_path("asset");
        tokio::fs::write(&partial, b"junk from an older build")
            .await
            .unwrap();

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path_matcher("/asset.bin"))
            .respond_with(move |req: &Request| {
                assert!(
                    !req.headers.contains_key("range"),
                    "no sidecar means no resume attempt"
                );
                ResponseTemplate::new(200).set_body_bytes(body(2048))
            })
            .mount(&server)
            .await;

        let destination = fx.paths.assets_dir.join("asset.bin");
        fetch_verified(
            &http_client().unwrap(),
            &format!("{}/asset.bin", server.uri()),
            &partial,
            &destination,
            bytes.len() as i64,
            Some(&sha_of(&bytes)),
            bytes.len() as i64,
            &AtomicBool::new(false),
            no_progress().as_ref(),
        )
        .await
        .unwrap();
        assert_eq!(tokio::fs::read(&destination).await.unwrap(), bytes);
    }

    #[tokio::test]
    async fn a_file_that_moved_upstream_restarts_instead_of_splicing() {
        let fx = fixture();
        let bytes = body(5000);
        let partial = fx.paths.asset_partial_path("asset");
        tokio::fs::write(&partial, body(1000)).await.unwrap();

        let server = MockServer::start().await;
        let url = format!("{}/asset.bin", server.uri());
        write_meta(
            &meta_path(&partial),
            &PartialMeta {
                url: url.clone(),
                etag: Some("\"stale\"".into()),
                total: Some(1234),
            },
        )
        .await
        .unwrap();

        // If-Range did not match, so the server sends the whole file with 200.
        Mock::given(method("GET"))
            .and(path_matcher("/asset.bin"))
            .respond_with(
                ResponseTemplate::new(200)
                    .append_header("ETag", "\"fresh\"")
                    .set_body_bytes(bytes.clone()),
            )
            .mount(&server)
            .await;

        let destination = fx.paths.assets_dir.join("asset.bin");
        let out = fetch_verified(
            &http_client().unwrap(),
            &url,
            &partial,
            &destination,
            bytes.len() as i64,
            Some(&sha_of(&bytes)),
            bytes.len() as i64,
            &AtomicBool::new(false),
            no_progress().as_ref(),
        )
        .await
        .unwrap();

        assert_eq!(out.resumed_from, 0, "the stale prefix was thrown away");
        assert_eq!(tokio::fs::read(&destination).await.unwrap(), bytes);
    }

    #[tokio::test]
    async fn a_wrong_hash_is_refused_and_the_partial_is_thrown_away() {
        let fx = fixture();
        let bytes = body(3000);
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(bytes.clone()))
            .mount(&server)
            .await;

        let partial = fx.paths.asset_partial_path("asset");
        let destination = fx.paths.assets_dir.join("asset.bin");
        let err = fetch_verified(
            &http_client().unwrap(),
            &format!("{}/asset.bin", server.uri()),
            &partial,
            &destination,
            bytes.len() as i64,
            Some(&"0".repeat(64)),
            bytes.len() as i64,
            &AtomicBool::new(false),
            no_progress().as_ref(),
        )
        .await
        .unwrap_err();

        assert!(matches!(err, AsrError::IntegrityCheckFailed), "{err:?}");
        assert!(!destination.exists(), "nothing damaged is ever installed");
        assert!(
            !partial.exists(),
            "and the bad bytes are not kept for a resume"
        );
    }

    #[tokio::test]
    async fn a_short_body_is_refused_even_when_no_hash_is_known() {
        let fx = fixture();
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body(100)))
            .mount(&server)
            .await;

        let partial = fx.paths.asset_partial_path("asset");
        let destination = fx.paths.assets_dir.join("asset.bin");
        let err = fetch_verified(
            &http_client().unwrap(),
            &format!("{}/asset.bin", server.uri()),
            &partial,
            &destination,
            9_999,
            None,
            9_999,
            &AtomicBool::new(false),
            no_progress().as_ref(),
        )
        .await
        .unwrap_err();

        assert!(matches!(err, AsrError::Download(_)), "{err:?}");
        assert!(!destination.exists());
    }

    #[tokio::test]
    async fn with_no_catalog_hash_the_computed_one_is_reported_back() {
        let fx = fixture();
        let bytes = body(1500);
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(bytes.clone()))
            .mount(&server)
            .await;

        let partial = fx.paths.asset_partial_path("asset");
        let destination = fx.paths.assets_dir.join("asset.bin");
        let out = fetch_verified(
            &http_client().unwrap(),
            &format!("{}/asset.bin", server.uri()),
            &partial,
            &destination,
            bytes.len() as i64,
            None,
            bytes.len() as i64,
            &AtomicBool::new(false),
            no_progress().as_ref(),
        )
        .await
        .unwrap();
        assert_eq!(out.sha256, sha_of(&bytes));
    }

    #[tokio::test]
    async fn a_server_error_is_a_download_failure_not_a_corrupt_install() {
        let fx = fixture();
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(503))
            .mount(&server)
            .await;

        let partial = fx.paths.asset_partial_path("asset");
        let destination = fx.paths.assets_dir.join("asset.bin");
        let err = fetch_verified(
            &http_client().unwrap(),
            &format!("{}/asset.bin", server.uri()),
            &partial,
            &destination,
            10,
            None,
            10,
            &AtomicBool::new(false),
            no_progress().as_ref(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, AsrError::Download(_)), "{err:?}");
        assert!(!destination.exists());
    }

    #[tokio::test]
    async fn cancelling_stops_before_anything_is_installed() {
        let fx = fixture();
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body(4096)))
            .mount(&server)
            .await;

        let partial = fx.paths.asset_partial_path("asset");
        let destination = fx.paths.assets_dir.join("asset.bin");
        let err = fetch_verified(
            &http_client().unwrap(),
            &format!("{}/asset.bin", server.uri()),
            &partial,
            &destination,
            4096,
            None,
            4096,
            &AtomicBool::new(true),
            no_progress().as_ref(),
        )
        .await
        .unwrap_err();

        assert!(matches!(err, AsrError::Cancelled), "{err:?}");
        assert!(!destination.exists());
    }

    #[tokio::test]
    async fn a_download_that_cannot_fit_is_refused_before_it_starts() {
        let fx = fixture();
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(body(16)))
            .mount(&server)
            .await;

        let partial = fx.paths.asset_partial_path("asset");
        let destination = fx.paths.assets_dir.join("asset.bin");
        let err = fetch_verified(
            &http_client().unwrap(),
            &format!("{}/asset.bin", server.uri()),
            &partial,
            &destination,
            16,
            None,
            // Nobody has an exabyte free.
            i64::MAX / 4,
            &AtomicBool::new(false),
            no_progress().as_ref(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, AsrError::NotEnoughSpace { .. }), "{err:?}");
    }

    #[tokio::test]
    async fn progress_is_reported_and_finishes_on_the_total() {
        let fx = fixture();
        let bytes = body(64 * 1024);
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(bytes.clone()))
            .mount(&server)
            .await;

        let seen: Arc<Mutex<Vec<(u64, u64)>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        let report: Box<dyn Fn(u64, u64) + Send + Sync> =
            Box::new(move |r, t| sink.lock().unwrap().push((r, t)));

        let partial = fx.paths.asset_partial_path("asset");
        let destination = fx.paths.assets_dir.join("asset.bin");
        fetch_verified(
            &http_client().unwrap(),
            &format!("{}/asset.bin", server.uri()),
            &partial,
            &destination,
            bytes.len() as i64,
            Some(&sha_of(&bytes)),
            bytes.len() as i64,
            &AtomicBool::new(false),
            report.as_ref(),
        )
        .await
        .unwrap();

        let seen = seen.lock().unwrap();
        assert!(seen.len() >= 2, "a start and an end at the very least");
        assert_eq!(seen.first().unwrap().0, 0);
        assert_eq!(
            *seen.last().unwrap(),
            (bytes.len() as u64, bytes.len() as u64)
        );
    }

    // -----------------------------------------------------------------
    // catalog ⇄ table
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn the_catalog_lands_in_the_table_with_its_provenance() {
        let db = connect_in_memory().await.unwrap();
        let rows = sync_catalog(&db).await.unwrap();
        assert!(!rows.is_empty());
        for row in &rows {
            let entry = catalog::entry(&row.id).expect("row came from the catalog");
            assert_eq!(row.url, entry.url);
            assert_eq!(row.bytes, entry.bytes);
            assert!(row.license.as_ref().is_some_and(|l| !l.is_empty()));
            assert!(row.revision.as_ref().is_some_and(|r| !r.is_empty()));
            assert!(!row.installed, "nothing is installed by cataloguing it");
        }
        assert_eq!(
            repo::get_setting(&db, settings::keys::MODEL_CATALOG_REVISION)
                .await
                .unwrap()
                .as_deref(),
            Some(catalog::CATALOG_REVISION)
        );

        // Running it twice changes nothing.
        let again = sync_catalog(&db).await.unwrap();
        assert_eq!(again.len(), rows.len());
    }

    #[tokio::test]
    async fn cataloguing_never_repoints_an_installed_asset_at_new_bytes() {
        let db = connect_in_memory().await.unwrap();
        sync_catalog(&db).await.unwrap();
        let id = catalog::ids::DETECTOR;
        repo::set_model_installed(&db, id, true, Some("/somewhere/old.onnx"))
            .await
            .unwrap();

        // Pretend the shipped catalog corrected this entry's hash.
        let mut row = repo::get_model(&db, id).await.unwrap().unwrap();
        row.sha256 = "f".repeat(64);
        repo::upsert_model(&db, &row).await.unwrap();

        sync_catalog(&db).await.unwrap();
        let after = repo::get_model(&db, id).await.unwrap().unwrap();
        assert!(
            !after.installed,
            "the stale file must be re-fetched, not loaded"
        );
        assert_eq!(after.sha256, catalog::entry(id).unwrap().sha256.unwrap());
    }

    #[tokio::test]
    async fn the_table_seeds_itself_so_nothing_depends_on_startup_order() {
        let db = connect_in_memory().await.unwrap();
        assert!(repo::list_models(&db, None).await.unwrap().is_empty());

        // Reading the presets on a database nobody catalogued must still answer
        // honestly, not report an install with nothing to download.
        let levels = list_accuracy_levels(&db).await.unwrap();
        assert_eq!(levels.len(), catalog::PRESETS.len());
        assert!(levels.iter().all(|l| l.download_bytes > 0 && !l.installed));
        assert!(!repo::list_models(&db, None).await.unwrap().is_empty());

        // And it does not re-seed on every call.
        assert_eq!(
            repo::get_setting(&db, settings::keys::MODEL_CATALOG_REVISION)
                .await
                .unwrap()
                .as_deref(),
            Some(catalog::CATALOG_REVISION)
        );
    }

    #[tokio::test]
    async fn presets_are_listed_with_their_selected_and_installed_state() {
        let db = connect_in_memory().await.unwrap();
        sync_catalog(&db).await.unwrap();
        let levels = list_accuracy_levels(&db).await.unwrap();
        assert_eq!(levels.len(), catalog::PRESETS.len());

        let selected: Vec<_> = levels.iter().filter(|l| l.selected).collect();
        assert_eq!(selected.len(), 1, "exactly one preset is selected");
        assert_eq!(selected[0].id, catalog::DEFAULT_PRESET_ID);
        assert!(levels.iter().all(|l| !l.installed));
        assert!(levels.iter().all(|l| l.download_bytes > 0));
        assert!(levels.iter().all(|l| !l.asset_ids.is_empty()));
        assert_eq!(levels.iter().filter(|l| l.recommended).count(), 1);

        // A different choice moves the flag, and an unknown one falls back.
        repo::set_setting(&db, settings::keys::ACCURACY_LEVEL_ID, "fastest")
            .await
            .unwrap();
        let levels = list_accuracy_levels(&db).await.unwrap();
        assert!(levels.iter().find(|l| l.id == "fastest").unwrap().selected);

        repo::set_setting(&db, settings::keys::ACCURACY_LEVEL_ID, "banana")
            .await
            .unwrap();
        let levels = list_accuracy_levels(&db).await.unwrap();
        assert!(
            levels
                .iter()
                .find(|l| l.id == catalog::DEFAULT_PRESET_ID)
                .unwrap()
                .selected
        );
    }

    #[tokio::test]
    async fn readiness_counts_what_is_still_missing_and_only_waits_for_essentials() {
        let fx = fixture();
        let db = connect_in_memory().await.unwrap();
        sync_catalog(&db).await.unwrap();

        let before = readiness(&db, &fx.paths).await.unwrap();
        assert!(!before.ready);
        assert!(!before.loaded, "nothing is loaded until something needs it");
        assert_eq!(
            before.remaining_bytes,
            catalog::preset_total_bytes(catalog::DEFAULT_PRESET_ID)
        );
        assert_eq!(before.level_id.as_deref(), Some(catalog::DEFAULT_PRESET_ID));

        // Put the two files a recording actually needs on disk.
        for id in catalog::preset_required_asset_ids(catalog::DEFAULT_PRESET_ID) {
            let entry = catalog::entry(id).unwrap();
            tokio::fs::write(install_path(&fx.paths, entry), b"x")
                .await
                .unwrap();
        }
        let after = readiness(&db, &fx.paths).await.unwrap();
        assert!(
            after.ready,
            "the speaker files arrive later; recording does not wait for them"
        );
        assert!(after.remaining_bytes > 0, "but they are still owed");
        assert!(after.remaining_bytes < before.remaining_bytes);
    }

    #[tokio::test]
    async fn a_file_deleted_behind_our_back_stops_reading_as_installed() {
        let fx = fixture();
        let db = connect_in_memory().await.unwrap();
        sync_catalog(&db).await.unwrap();
        let entry = catalog::entry(catalog::ids::DETECTOR).unwrap();
        let path = install_path(&fx.paths, entry);
        tokio::fs::write(&path, b"x").await.unwrap();

        refresh_installed(&db, &fx.paths).await.unwrap();
        let row = repo::get_model(&db, entry.id).await.unwrap().unwrap();
        assert!(row.installed);
        assert_eq!(row.path.as_deref(), path.to_str());

        tokio::fs::remove_file(&path).await.unwrap();
        refresh_installed(&db, &fx.paths).await.unwrap();
        assert!(
            !repo::get_model(&db, entry.id)
                .await
                .unwrap()
                .unwrap()
                .installed
        );
    }

    #[tokio::test]
    async fn stale_partials_are_swept_up_at_launch() {
        let fx = fixture();
        let db = connect_in_memory().await.unwrap();
        sync_catalog(&db).await.unwrap();

        let live = fx.paths.asset_partial_path(catalog::ids::DETECTOR);
        let orphan = fx.paths.asset_partial_path("speech-from-a-past-release");
        let installed = fx.paths.asset_partial_path(catalog::ids::SEGMENTER);
        for p in [&live, &orphan, &installed] {
            tokio::fs::write(p, b"half a file").await.unwrap();
            tokio::fs::write(meta_path(p), b"{}").await.unwrap();
        }
        // This one finished by another route, so its partial is dead weight.
        repo::set_model_installed(&db, catalog::ids::SEGMENTER, true, Some("/x"))
            .await
            .unwrap();

        let removed = clean_stale_partials(&db, &fx.paths).await.unwrap();
        assert_eq!(
            removed, 4,
            "the orphan and the installed one, file + sidecar"
        );
        assert!(live.exists(), "a resumable download is left alone");
        assert!(meta_path(&live).exists());
        assert!(!orphan.exists());
        assert!(!installed.exists());
    }

    #[tokio::test]
    async fn installed_path_answers_for_the_selected_preset_only() {
        let db = connect_in_memory().await.unwrap();
        sync_catalog(&db).await.unwrap();
        assert!(installed_path(&db, AssetKind::Speech)
            .await
            .unwrap()
            .is_none());

        repo::set_model_installed(
            &db,
            catalog::ids::SPEECH_EVERYDAY,
            true,
            Some("/speech/everyday.bin"),
        )
        .await
        .unwrap();
        assert_eq!(
            installed_path(&db, AssetKind::Speech).await.unwrap(),
            Some(PathBuf::from("/speech/everyday.bin"))
        );

        // Switching preset points at a different file, which is not there.
        repo::set_setting(&db, settings::keys::ACCURACY_LEVEL_ID, "fastest")
            .await
            .unwrap();
        assert!(installed_path(&db, AssetKind::Speech)
            .await
            .unwrap()
            .is_none());
    }

    // -----------------------------------------------------------------
    // download_asset end to end, against a local server
    // -----------------------------------------------------------------

    /// Point a catalogued row at the mock server, with the length and hash of
    /// the body it will serve. Everything else about the entry is untouched.
    async fn redirect(db: &Db, asset_id: &str, url: String, bytes: &[u8], with_hash: bool) {
        let mut row = repo::get_model(db, asset_id).await.unwrap().unwrap();
        row.url = url;
        row.bytes = bytes.len() as i64;
        row.sha256 = if with_hash {
            sha_of(bytes)
        } else {
            String::new()
        };
        repo::upsert_model(db, &row).await.unwrap();
    }

    #[tokio::test]
    async fn downloading_an_asset_installs_it_and_records_where_it_went() {
        let fx = fixture();
        let db = connect_in_memory().await.unwrap();
        sync_catalog(&db).await.unwrap();

        let bytes = body(2222);
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path_matcher("/detector"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(bytes.clone()))
            .mount(&server)
            .await;
        let id = catalog::ids::DETECTOR.to_string();
        redirect(&db, &id, format!("{}/detector", server.uri()), &bytes, true).await;

        let row = download_asset(&db, &fx.paths, &id, Box::new(|_| {}))
            .await
            .unwrap();
        assert!(row.installed);
        let entry = catalog::entry(&id).unwrap();
        let expected = install_path(&fx.paths, entry);
        assert_eq!(row.path.as_deref(), expected.to_str());
        assert_eq!(tokio::fs::read(&expected).await.unwrap(), bytes);
        assert!(verify_asset(&db, &id).await.unwrap());

        // Second call is a no-op that still reports success, and does not need
        // the server at all.
        drop(server);
        let again = download_asset(&db, &fx.paths, &id, Box::new(|_| {}))
            .await
            .unwrap();
        assert!(again.installed);
    }

    #[tokio::test]
    async fn a_hashless_entry_is_trusted_once_and_checked_ever_after() {
        let fx = fixture();
        let db = connect_in_memory().await.unwrap();
        sync_catalog(&db).await.unwrap();

        let bytes = body(777);
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(bytes.clone()))
            .mount(&server)
            .await;
        let id = catalog::ids::EMBEDDER.to_string();
        redirect(&db, &id, format!("{}/e", server.uri()), &bytes, false).await;

        let row = download_asset(&db, &fx.paths, &id, Box::new(|_| {}))
            .await
            .unwrap();
        assert_eq!(
            row.sha256,
            sha_of(&bytes),
            "the hash we computed is stored, so later checks mean something"
        );
        assert!(verify_asset(&db, &id).await.unwrap());

        // Corrupt the installed file: verification must now fail.
        tokio::fs::write(row.path.as_deref().unwrap(), b"tampered")
            .await
            .unwrap();
        assert!(!verify_asset(&db, &id).await.unwrap());
    }

    #[tokio::test]
    async fn removing_an_asset_clears_the_row_and_any_half_download() {
        let fx = fixture();
        let db = connect_in_memory().await.unwrap();
        sync_catalog(&db).await.unwrap();
        let id = catalog::ids::DETECTOR.to_string();
        let entry = catalog::entry(&id).unwrap();
        let path = install_path(&fx.paths, entry);
        tokio::fs::write(&path, b"weights").await.unwrap();
        let partial = fx.paths.asset_partial_path(&id);
        tokio::fs::write(&partial, b"half").await.unwrap();
        repo::set_model_installed(&db, &id, true, Some(&path_str(&path)))
            .await
            .unwrap();

        remove_asset(&db, &fx.paths, &id).await.unwrap();
        assert!(!path.exists());
        assert!(!partial.exists());
        assert!(!repo::get_model(&db, &id).await.unwrap().unwrap().installed);
        assert!(!verify_asset(&db, &id).await.unwrap());
    }

    #[tokio::test]
    async fn an_unknown_asset_is_rejected_rather_than_guessed_at() {
        let fx = fixture();
        let db = connect_in_memory().await.unwrap();
        let err = download_asset(&db, &fx.paths, &"not-a-thing".to_string(), Box::new(|_| {}))
            .await
            .unwrap_err();
        assert!(matches!(err, AsrError::UnknownAsset(_)), "{err:?}");
        assert!(matches!(
            cancel_download(&"not-a-thing".to_string()).await,
            Ok(())
        ));
    }

    #[tokio::test]
    async fn a_whole_preset_downloads_in_order_with_one_progress_bar() {
        let fx = fixture();
        let db = connect_in_memory().await.unwrap();
        sync_catalog(&db).await.unwrap();

        let server = MockServer::start().await;
        let level = "fastest";
        let ids = catalog::preset_asset_ids(level);
        // The Apple companion is a zip bundle needing a real archive; the rest
        // of the preset is plain files. Cover the plain path here.
        let plain: Vec<&str> = ids
            .iter()
            .copied()
            .filter(|id| !catalog::entry(id).unwrap().is_bundle())
            .collect();
        let mut expected_total = 0i64;
        for (n, id) in plain.iter().enumerate() {
            let bytes = body(512 + n * 128);
            expected_total += bytes.len() as i64;
            Mock::given(method("GET"))
                .and(path_matcher(format!("/{id}")))
                .respond_with(ResponseTemplate::new(200).set_body_bytes(bytes.clone()))
                .mount(&server)
                .await;
            redirect(&db, id, format!("{}/{id}", server.uri()), &bytes, true).await;
        }
        // Anything we are not serving must not be asked for.
        for id in ids.iter().filter(|id| !plain.contains(*id)) {
            let entry = catalog::entry(id).unwrap();
            tokio::fs::create_dir_all(install_path(&fx.paths, entry))
                .await
                .unwrap();
            tokio::fs::write(install_path(&fx.paths, entry).join("stub"), b"x")
                .await
                .unwrap();
        }

        let seen: Arc<Mutex<Vec<DownloadProgress>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        download_level(
            &db,
            &fx.paths,
            level,
            Box::new(move |p| sink.lock().unwrap().push(p)),
        )
        .await
        .unwrap();

        for id in &plain {
            let row = repo::get_model(&db, id).await.unwrap().unwrap();
            assert!(row.installed, "{id} should be installed");
        }
        let seen = seen.lock().unwrap();
        let last = seen.last().expect("progress was reported");
        assert!(last.done, "the preset reports itself finished");
        assert_eq!(last.level_id.as_deref(), Some(level));
        assert!(
            seen.iter().all(|p| p.level_id.as_deref() == Some(level)),
            "every report names the preset, not the individual file"
        );
        assert!(
            expected_total > 0,
            "the mock served something (sanity check)"
        );
    }

    #[test]
    fn hex_is_lowercase_and_zero_padded() {
        assert_eq!(hex(&[0x00, 0x0f, 0xff]), "000fff");
    }

    #[test]
    fn the_sidecar_sits_beside_the_partial() {
        let p = PathBuf::from("/tmp/echo/abc.part");
        assert_eq!(meta_path(&p), PathBuf::from("/tmp/echo/abc.part.meta"));
    }

    #[test]
    fn installed_memory_is_either_a_real_number_or_an_honest_zero() {
        // Nothing to assert about the value; the contract is that it does not
        // panic and that 0 means "unknown".
        let _ = total_memory_bytes();
    }
}
