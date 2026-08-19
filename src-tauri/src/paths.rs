//! Where Echo keeps things on disk.
//!
//! Two roots, deliberately separate:
//!
//! * **App root**, database, downloaded speech assets, logs. Lives in the
//!   platform data directory and is not user-configurable.
//! * **Storage root**, recordings. Configurable (Settings → General →
//!   storage location) because recordings are the big, personal, movable part.
//!   Defaults to `<app root>/recordings`.
//!
//! Layout under the storage root:
//!
//! ```text
//! recordings/
//!   <meeting-id>/
//!     mic-000000.wav       per-channel chunks, canonical from t=0 (mantra 3)
//!     mic-000001.wav
//!     system-000000.wav
//!     mixed.wav            derived, playback only
//! ```
//!
//! The extension is whatever [`crate::audio::writer::DEFAULT_CHUNK_FORMAT`] says
//! — WAV today, FLAC once a decoder is in the tree — and this module asks the
//! writer rather than hardcoding it, so the two can never disagree.
//!
//! Nothing here touches the database; `settings` owns the configured value and
//! passes it in.

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use crate::types::Channel;

/// Directory name under the platform data dir.
#[cfg(target_os = "macos")]
pub const APP_DIR_NAME: &str = "Echo";
#[cfg(not(target_os = "macos"))]
pub const APP_DIR_NAME: &str = "echo";

/// Bundle identifier, kept in sync with `tauri.conf.json`.
pub const APP_IDENTIFIER: &str = "app.echo.desktop";

#[derive(Debug, thiserror::Error)]
pub enum PathsError {
    #[error("this computer has no usable home directory")]
    NoHomeDirectory,
    #[error("could not create {path}: {source}")]
    Create {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("{path} is not a directory Echo can write to")]
    NotWritable { path: String },
    #[error("a storage location must be an absolute path")]
    NotAbsolute,
    #[error("{path} holds other things; Echo needs a folder of its own")]
    NotEmpty { path: String },
    #[error("{path} is too important a folder to fill with recordings")]
    TooBroad { path: String },
}

/// Resolved absolute paths for one run of the app.
///
/// The recordings root is behind a shared cell rather than a plain field,
/// because Settings can move it while the app is running and every clone of
/// this value — the session, the job runner, the commands — has to follow. It is
/// read through [`AppPaths::storage_root`] and changed through
/// [`AppPaths::set_storage_root`]; only new recordings are affected, because
/// each meeting row carries the folder it was recorded into.
#[derive(Clone)]
pub struct AppPaths {
    /// Platform data dir: database, speech assets, logs.
    pub app_root: PathBuf,
    /// Where recordings go. May live on another volume, and may change.
    storage_root: Arc<RwLock<PathBuf>>,
    /// SQLite file.
    pub db_path: PathBuf,
    /// Downloaded speech assets.
    pub assets_dir: PathBuf,
    /// Rotating redacted log files.
    pub log_dir: PathBuf,
    /// Scratch space for downloads and export staging (safe to wipe).
    pub tmp_dir: PathBuf,
}

impl std::fmt::Debug for AppPaths {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppPaths")
            .field("app_root", &self.app_root)
            .field("storage_root", &self.storage_root())
            .field("db_path", &self.db_path)
            .field("assets_dir", &self.assets_dir)
            .field("log_dir", &self.log_dir)
            .field("tmp_dir", &self.tmp_dir)
            .finish()
    }
}

impl PartialEq for AppPaths {
    fn eq(&self, other: &Self) -> bool {
        self.app_root == other.app_root
            && self.storage_root() == other.storage_root()
            && self.db_path == other.db_path
            && self.assets_dir == other.assets_dir
            && self.log_dir == other.log_dir
            && self.tmp_dir == other.tmp_dir
    }
}

impl Eq for AppPaths {}

impl AppPaths {
    /// Resolve paths, using `storage_override` when the person moved their
    /// recordings elsewhere. Creates nothing, call [`AppPaths::ensure`].
    pub fn resolve(storage_override: Option<&Path>) -> Result<Self, PathsError> {
        let app_root = default_app_root()?;
        let storage_root = match storage_override {
            Some(p) if !p.as_os_str().is_empty() => {
                if !p.is_absolute() {
                    return Err(PathsError::NotAbsolute);
                }
                p.to_path_buf()
            }
            _ => app_root.join("recordings"),
        };
        Ok(Self {
            db_path: app_root.join("echo.db"),
            assets_dir: app_root.join("speech"),
            log_dir: app_root.join("logs"),
            tmp_dir: app_root.join("tmp"),
            app_root,
            storage_root: Arc::new(RwLock::new(storage_root)),
        })
    }

    /// Build paths rooted at an arbitrary directory. Used by tests and by the
    /// "move my recordings" preflight.
    pub fn rooted_at(app_root: impl Into<PathBuf>, storage_root: Option<PathBuf>) -> Self {
        let app_root = app_root.into();
        let storage_root = storage_root.unwrap_or_else(|| app_root.join("recordings"));
        Self {
            db_path: app_root.join("echo.db"),
            assets_dir: app_root.join("speech"),
            log_dir: app_root.join("logs"),
            tmp_dir: app_root.join("tmp"),
            app_root,
            storage_root: Arc::new(RwLock::new(storage_root)),
        }
    }

    /// Where recordings are being written right now.
    pub fn storage_root(&self) -> PathBuf {
        match self.storage_root.read() {
            Ok(root) => root.clone(),
            // A poisoned lock still holds a perfectly good path.
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }

    /// Point new recordings somewhere else. Every clone of these paths follows,
    /// so the session and the job runner cannot drift from what Settings says.
    /// Recordings already on disk are untouched: their folder is stored on the
    /// meeting row.
    pub fn set_storage_root(&self, root: &Path) {
        let mut guard = match self.storage_root.write() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        *guard = root.to_path_buf();
    }

    /// Create every directory Echo needs. Idempotent.
    pub fn ensure(&self) -> Result<(), PathsError> {
        for dir in [
            &self.app_root,
            &self.storage_root(),
            &self.assets_dir,
            &self.log_dir,
            &self.tmp_dir,
        ] {
            create_dir(dir)?;
        }
        Ok(())
    }

    /// Directory holding one meeting's audio.
    pub fn meeting_dir(&self, meeting_id: &str) -> PathBuf {
        self.storage_root().join(meeting_id)
    }

    /// Path for one per-channel chunk. Zero-padded so the shell sorts right.
    ///
    /// The name and extension come from [`crate::audio::writer`], which is what
    /// actually writes the file — one source of truth, so changing the on-disk
    /// format does not leave this predicting the wrong name.
    pub fn chunk_path(&self, meeting_id: &str, channel: Channel, seq: u64) -> PathBuf {
        self.meeting_dir(meeting_id)
            .join(crate::audio::writer::chunk_file_name(
                channel,
                seq,
                crate::audio::writer::DEFAULT_CHUNK_FORMAT,
            ))
    }

    /// Derived mixdown used for playback only.
    pub fn mixed_path(&self, meeting_id: &str) -> PathBuf {
        self.meeting_dir(meeting_id).join(format!(
            "mixed.{}",
            crate::audio::writer::DEFAULT_CHUNK_FORMAT.extension()
        ))
    }

    /// Partial-download destination for a catalog entry.
    pub fn asset_partial_path(&self, asset_id: &str) -> PathBuf {
        self.tmp_dir.join(format!("{asset_id}.part"))
    }

    /// Final location for a downloaded asset.
    pub fn asset_path(&self, file_name: &str) -> PathBuf {
        self.assets_dir.join(file_name)
    }

    /// SQLite connection string with WAL-friendly options.
    pub fn db_url(&self) -> String {
        format!("sqlite://{}?mode=rwc", self.db_path.display())
    }
}

/// Platform data directory for Echo.
pub fn default_app_root() -> Result<PathBuf, PathsError> {
    // macOS: ~/Library/Application Support/Echo
    // Linux: ~/.local/share/echo  (respects XDG_DATA_HOME)
    let base = dirs::data_dir().ok_or(PathsError::NoHomeDirectory)?;
    Ok(base.join(APP_DIR_NAME))
}

/// Where exports go by default: the person's Documents folder, or home.
pub fn default_export_dir() -> Result<PathBuf, PathsError> {
    if let Some(d) = dirs::document_dir() {
        return Ok(d);
    }
    dirs::home_dir().ok_or(PathsError::NoHomeDirectory)
}

fn create_dir(path: &Path) -> Result<(), PathsError> {
    std::fs::create_dir_all(path).map_err(|source| PathsError::Create {
        path: path.display().to_string(),
        source,
    })
}

/// Check a candidate storage location before we let the person commit to it.
/// Returns the free space in bytes.
///
/// Echo owns whatever folder this is: "delete everything" removes the meetings
/// it put there, and a folder picker makes `~`, `~/Documents` or a drive root
/// entirely plausible choices. So a folder that is somebody's home, a volume
/// root, or already full of other things is refused — the person picks or
/// creates a folder for recordings instead.
pub fn validate_storage_dir(path: &Path) -> Result<u64, PathsError> {
    if !path.is_absolute() {
        return Err(PathsError::NotAbsolute);
    }
    if is_too_broad(path) {
        return Err(PathsError::TooBroad {
            path: path.display().to_string(),
        });
    }
    create_dir(path)?;
    if let Some(foreign) = first_foreign_entry(path) {
        return Err(PathsError::NotEmpty {
            path: foreign.display().to_string(),
        });
    }
    let probe = path.join(".echo-write-test");
    std::fs::write(&probe, b"echo").map_err(|_| PathsError::NotWritable {
        path: path.display().to_string(),
    })?;
    let _ = std::fs::remove_file(&probe);
    Ok(free_space_bytes(path))
}

/// Folders that must never become the recordings root: the filesystem root, a
/// volume's mount point, a home directory, and the well-known folders inside one.
fn is_too_broad(path: &Path) -> bool {
    let path = normalize(path);
    if path.parent().is_none() {
        return true;
    }
    // "/Volumes/Big" and "/media/me/stick" are mount points: Echo may use a
    // folder *on* the drive, not the whole drive.
    if is_volume_root(&path) {
        return true;
    }
    let mut reserved: Vec<PathBuf> = Vec::new();
    if let Some(home) = dirs::home_dir() {
        reserved.push(normalize(&home));
        for name in [
            "Desktop",
            "Documents",
            "Downloads",
            "Library",
            "Movies",
            "Music",
            "Pictures",
            "Public",
        ] {
            reserved.push(normalize(&home.join(name)));
        }
    }
    for dir in [
        dirs::document_dir(),
        dirs::download_dir(),
        dirs::desktop_dir(),
        dirs::picture_dir(),
        dirs::video_dir(),
        dirs::audio_dir(),
    ]
    .into_iter()
    .flatten()
    {
        reserved.push(normalize(&dir));
    }
    reserved.contains(&path)
}

fn is_volume_root(path: &Path) -> bool {
    use sysinfo::Disks;
    let disks = Disks::new_with_refreshed_list();
    disks
        .list()
        .iter()
        .any(|disk| normalize(disk.mount_point()) == path)
}

/// Resolve what we can without requiring the path to exist, and drop a trailing
/// separator, so two spellings of the same folder compare equal.
fn normalize(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

/// The first thing in `path` that Echo did not put there. `None` means the
/// folder is Echo's to manage.
///
/// Echo's own entries are per-meeting folders (named with the meeting id) and
/// the files this module creates. Anything else — someone's tax return, a photo
/// library — means this is not a folder Echo may take over.
fn first_foreign_entry(path: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(path).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') || name == ".echo-write-test" {
            // Hidden bookkeeping (.DS_Store, .Trash) is not somebody's data.
            continue;
        }
        let is_meeting_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false)
            && uuid::Uuid::parse_str(&name).is_ok();
        if !is_meeting_dir {
            return Some(entry.path());
        }
    }
    None
}

/// Free bytes on the volume containing `path`. Returns 0 when unknown, the
/// caller must treat 0 as "can't tell", never as "full".
pub fn free_space_bytes(path: &Path) -> u64 {
    use sysinfo::Disks;

    let disks = Disks::new_with_refreshed_list();
    let mut best: Option<(usize, u64)> = None;
    for disk in disks.list() {
        let mount = disk.mount_point();
        if path.starts_with(mount) {
            let depth = mount.components().count();
            if best.map(|(d, _)| depth > d).unwrap_or(true) {
                best = Some((depth, disk.available_space()));
            }
        }
    }
    best.map(|(_, free)| free).unwrap_or(0)
}

/// Recursive size of a directory in bytes. Missing directory = 0.
pub fn dir_size_bytes(path: &Path) -> u64 {
    fn walk(path: &Path, total: &mut u64) {
        let Ok(entries) = std::fs::read_dir(path) else {
            return;
        };
        for entry in entries.flatten() {
            match entry.file_type() {
                Ok(t) if t.is_dir() => walk(&entry.path(), total),
                Ok(t) if t.is_file() => {
                    if let Ok(meta) = entry.metadata() {
                        *total += meta.len();
                    }
                }
                _ => {}
            }
        }
    }
    let mut total = 0;
    walk(path, &mut total);
    total
}

/// Size of a single file, or 0 when it is not there.
pub fn file_size_bytes(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_paths_are_sortable_and_per_channel() {
        let p = AppPaths::rooted_at("/tmp/echo-test", None);
        assert_eq!(
            p.chunk_path("m1", Channel::Mic, 7),
            PathBuf::from(format!(
                "/tmp/echo-test/recordings/m1/mic-000007.{}",
                crate::audio::writer::DEFAULT_CHUNK_FORMAT.extension()
            ))
        );
        assert_eq!(
            p.chunk_path("m1", Channel::System, 1234),
            PathBuf::from(format!(
                "/tmp/echo-test/recordings/m1/system-001234.{}",
                crate::audio::writer::DEFAULT_CHUNK_FORMAT.extension()
            ))
        );
        // lexical order == time order
        let a = p.chunk_path("m1", Channel::Mic, 9);
        let b = p.chunk_path("m1", Channel::Mic, 10);
        assert!(a < b);
    }

    #[test]
    fn storage_override_must_be_absolute() {
        let err = AppPaths::resolve(Some(Path::new("relative/dir"))).unwrap_err();
        assert!(matches!(err, PathsError::NotAbsolute));
    }

    #[test]
    fn empty_override_falls_back_to_default() {
        let with_empty = AppPaths::resolve(Some(Path::new(""))).unwrap();
        let without = AppPaths::resolve(None).unwrap();
        assert_eq!(with_empty, without);
    }

    #[test]
    fn ensure_creates_everything_and_is_idempotent() {
        let tmp = tempfile::tempdir().unwrap();
        let p = AppPaths::rooted_at(tmp.path().join("root"), None);
        p.ensure().unwrap();
        p.ensure().unwrap();
        assert!(p.assets_dir.is_dir());
        assert!(p.log_dir.is_dir());
        assert!(p.tmp_dir.is_dir());
        assert!(p.storage_root().is_dir());
    }

    #[test]
    fn moving_the_storage_root_is_seen_by_every_clone() {
        let tmp = tempfile::tempdir().unwrap();
        let p = AppPaths::rooted_at(tmp.path().join("root"), None);
        let held_elsewhere = p.clone();
        let moved = tmp.path().join("elsewhere");
        p.set_storage_root(&moved);
        assert_eq!(held_elsewhere.storage_root(), moved);
        assert_eq!(held_elsewhere.meeting_dir("m1"), moved.join("m1"));
    }

    #[test]
    fn a_folder_full_of_someone_elses_things_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("mixed");
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(target.join("tax-return.pdf"), b"mine").unwrap();
        let err = validate_storage_dir(&target).unwrap_err();
        assert!(matches!(err, PathsError::NotEmpty { .. }), "{err:?}");
    }

    #[test]
    fn a_folder_of_recordings_is_still_accepted() {
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("recordings");
        std::fs::create_dir_all(target.join(uuid::Uuid::new_v4().to_string())).unwrap();
        std::fs::write(target.join(".DS_Store"), b"junk").unwrap();
        validate_storage_dir(&target).unwrap();
    }

    #[test]
    fn home_and_volume_roots_are_never_the_recordings_folder() {
        assert!(is_too_broad(Path::new("/")));
        if let Some(home) = dirs::home_dir() {
            assert!(is_too_broad(&home), "{} was accepted", home.display());
            assert!(is_too_broad(&home.join("Documents")));
            assert!(!is_too_broad(&home.join("Echo Recordings")));
        }
    }

    #[test]
    fn dir_size_counts_nested_files() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("a/b")).unwrap();
        std::fs::write(tmp.path().join("a/one"), vec![0u8; 100]).unwrap();
        std::fs::write(tmp.path().join("a/b/two"), vec![0u8; 50]).unwrap();
        assert_eq!(dir_size_bytes(tmp.path()), 150);
        assert_eq!(dir_size_bytes(&tmp.path().join("missing")), 0);
    }

    #[test]
    fn validate_storage_dir_accepts_a_fresh_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("recordings");
        validate_storage_dir(&target).unwrap();
        assert!(target.is_dir());
        assert!(!target.join(".echo-write-test").exists());
    }
}
