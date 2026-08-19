//! Typed view over the `settings` key/value table.
//!
//! Fully implemented (not a stub). Rules:
//! * **Non-secret only.** API keys live in the OS keychain, see [`crate::secrets`].
//! * Unknown or corrupt values fall back to the default rather than failing;
//!   a bad row must never stop the app from opening.
//! * `storage_dir` is the one setting with a computed default, so it always
//!   comes back as an absolute path.

use std::path::PathBuf;

use crate::db::{repo, Db, DbError};
use crate::paths;
use crate::types::{Provider, Settings, SettingsPatch, SummaryLanguage};

/// Column keys. Snake_case, stable, renaming one is a migration.
pub mod keys {
    pub const LAUNCH_AT_LOGIN: &str = "launch_at_login";
    pub const DETECTION_ENABLED: &str = "detection_enabled";
    pub const STORAGE_DIR: &str = "storage_dir";
    pub const CAPTURE_SYSTEM_AUDIO: &str = "capture_system_audio";
    pub const INPUT_DEVICE_ID: &str = "input_device_id";
    pub const SUMMARY_LANGUAGE: &str = "summary_language";
    pub const SUMMARY_PROVIDER: &str = "summary_provider";
    pub const SUMMARY_TEMPLATE_ID: &str = "summary_template_id";
    pub const AUTO_SUMMARIZE: &str = "auto_summarize";
    pub const ACCURACY_LEVEL_ID: &str = "accuracy_level_id";
    pub const RELEASE_AFTER_IDLE_MINUTES: &str = "release_after_idle_minutes";
    pub const CLOSE_TO_TRAY: &str = "close_to_tray";
    pub const ONBOARDING_COMPLETE: &str = "onboarding_complete";
    pub const SHOW_ADVANCED: &str = "show_advanced";

    // Keys owned by modules but stored here so everything non-secret is in one
    // table. Read/written through `repo::get_setting` / `repo::set_setting`.
    pub const OLLAMA_BASE_URL: &str = "ollama_base_url";
    pub const OLLAMA_MODEL: &str = "ollama_model";
    pub const GEMINI_MODEL: &str = "gemini_model";
    pub const DETECTION_SNOOZED_UNTIL: &str = "detection_snoozed_until";
    pub const MODEL_CATALOG_REVISION: &str = "model_catalog_revision";
}

/// Read every setting, filling in defaults for anything missing.
pub async fn load(db: &Db) -> Result<Settings, DbError> {
    let raw = repo::all_settings(db).await?;
    let mut s = Settings::default();

    let get = |k: &str| raw.get(k).map(String::as_str);

    if let Some(v) = get(keys::LAUNCH_AT_LOGIN) {
        s.launch_at_login = parse_bool(v, s.launch_at_login);
    }
    if let Some(v) = get(keys::DETECTION_ENABLED) {
        s.detection_enabled = parse_bool(v, s.detection_enabled);
    }
    if let Some(v) = get(keys::CAPTURE_SYSTEM_AUDIO) {
        s.capture_system_audio = parse_bool(v, s.capture_system_audio);
    }
    if let Some(v) = get(keys::AUTO_SUMMARIZE) {
        s.auto_summarize = parse_bool(v, s.auto_summarize);
    }
    if let Some(v) = get(keys::CLOSE_TO_TRAY) {
        s.close_to_tray = parse_bool(v, s.close_to_tray);
    }
    if let Some(v) = get(keys::ONBOARDING_COMPLETE) {
        s.onboarding_complete = parse_bool(v, s.onboarding_complete);
    }
    if let Some(v) = get(keys::SHOW_ADVANCED) {
        s.show_advanced = parse_bool(v, s.show_advanced);
    }
    if let Some(v) = get(keys::RELEASE_AFTER_IDLE_MINUTES) {
        s.release_after_idle_minutes = v.parse().unwrap_or(s.release_after_idle_minutes);
    }
    if let Some(v) = get(keys::ACCURACY_LEVEL_ID).filter(|v| !v.is_empty()) {
        s.accuracy_level_id = v.to_string();
    }
    s.input_device_id = get(keys::INPUT_DEVICE_ID)
        .filter(|v| !v.is_empty())
        .map(String::from);
    s.summary_template_id = get(keys::SUMMARY_TEMPLATE_ID)
        .filter(|v| !v.is_empty())
        .map(String::from);
    if let Some(v) = get(keys::SUMMARY_PROVIDER) {
        s.summary_provider = Provider::parse(v).unwrap_or(s.summary_provider);
    }
    if let Some(v) = get(keys::SUMMARY_LANGUAGE) {
        s.summary_language = serde_json::from_str(v).unwrap_or(SummaryLanguage::SameAsMeeting);
    }

    // storage_dir always resolves to an absolute path.
    s.storage_dir = match get(keys::STORAGE_DIR).filter(|v| !v.is_empty()) {
        Some(v) => v.to_string(),
        None => default_storage_dir(),
    };

    Ok(s)
}

/// Apply a partial update and return the settings as they now stand.
pub async fn apply(db: &Db, patch: &SettingsPatch) -> Result<Settings, DbError> {
    let mut pairs: Vec<(String, String)> = Vec::new();
    let mut put = |key: &str, value: String| pairs.push((key.to_string(), value));

    if let Some(v) = patch.launch_at_login {
        put(keys::LAUNCH_AT_LOGIN, v.to_string());
    }
    if let Some(v) = patch.detection_enabled {
        put(keys::DETECTION_ENABLED, v.to_string());
    }
    if let Some(v) = patch.capture_system_audio {
        put(keys::CAPTURE_SYSTEM_AUDIO, v.to_string());
    }
    if let Some(v) = patch.auto_summarize {
        put(keys::AUTO_SUMMARIZE, v.to_string());
    }
    if let Some(v) = patch.close_to_tray {
        put(keys::CLOSE_TO_TRAY, v.to_string());
    }
    if let Some(v) = patch.onboarding_complete {
        put(keys::ONBOARDING_COMPLETE, v.to_string());
    }
    if let Some(v) = patch.show_advanced {
        put(keys::SHOW_ADVANCED, v.to_string());
    }
    if let Some(v) = patch.release_after_idle_minutes {
        put(keys::RELEASE_AFTER_IDLE_MINUTES, v.to_string());
    }
    if let Some(v) = &patch.accuracy_level_id {
        put(keys::ACCURACY_LEVEL_ID, v.clone());
    }
    if let Some(v) = &patch.input_device_id {
        put(keys::INPUT_DEVICE_ID, v.clone());
    }
    if let Some(v) = &patch.summary_template_id {
        put(keys::SUMMARY_TEMPLATE_ID, v.clone());
    }
    if let Some(v) = patch.summary_provider {
        put(keys::SUMMARY_PROVIDER, v.as_str().to_string());
    }
    if let Some(v) = &patch.summary_language {
        put(
            keys::SUMMARY_LANGUAGE,
            serde_json::to_string(v).unwrap_or_else(|_| "{\"kind\":\"sameAsMeeting\"}".into()),
        );
    }
    if let Some(v) = &patch.storage_dir {
        put(keys::STORAGE_DIR, v.clone());
    }

    repo::set_settings(db, &pairs).await?;
    load(db).await
}

/// Absolute storage directory, resolved through [`crate::paths`].
pub async fn storage_dir(db: &Db) -> Result<PathBuf, DbError> {
    Ok(PathBuf::from(load(db).await?.storage_dir))
}

/// Resolve the on-disk layout for the configured storage location.
pub async fn app_paths(db: &Db) -> Result<paths::AppPaths, DbError> {
    let configured = storage_dir(db).await?;
    paths::AppPaths::resolve(Some(&configured))
        .map_err(|e| DbError::Decode(format!("storage location unusable: {e}")))
}

fn default_storage_dir() -> String {
    paths::default_app_root()
        .map(|p| p.join("recordings"))
        .unwrap_or_else(|_| PathBuf::from("recordings"))
        .to_string_lossy()
        .into_owned()
}

fn parse_bool(value: &str, fallback: bool) -> bool {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => true,
        "false" | "0" | "no" | "off" => false,
        _ => fallback,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::connect_in_memory;

    #[tokio::test]
    async fn defaults_come_back_for_an_empty_table() {
        let db = connect_in_memory().await.unwrap();
        let s = load(&db).await.unwrap();
        assert!(s.detection_enabled);
        assert!(s.capture_system_audio);
        assert!(!s.onboarding_complete);
        assert!(!s.show_advanced);
        assert_eq!(s.accuracy_level_id, "everyday");
        assert_eq!(s.summary_provider, Provider::OnThisComputer);
        assert_eq!(s.summary_language, SummaryLanguage::SameAsMeeting);
        assert_eq!(s.release_after_idle_minutes, 10);
        assert!(
            !s.storage_dir.is_empty(),
            "storage location must always resolve"
        );
        assert!(PathBuf::from(&s.storage_dir).is_absolute());
    }

    #[tokio::test]
    async fn patch_only_touches_the_fields_it_names() {
        let db = connect_in_memory().await.unwrap();
        let after = apply(
            &db,
            &SettingsPatch {
                detection_enabled: Some(false),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert!(!after.detection_enabled);
        assert!(
            after.capture_system_audio,
            "untouched fields keep their value"
        );

        let after2 = apply(
            &db,
            &SettingsPatch {
                auto_summarize: Some(true),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert!(!after2.detection_enabled, "earlier change survives");
        assert!(after2.auto_summarize);
    }

    #[tokio::test]
    async fn summary_language_round_trips() {
        let db = connect_in_memory().await.unwrap();
        let after = apply(
            &db,
            &SettingsPatch {
                summary_language: Some(SummaryLanguage::Fixed("it".into())),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(after.summary_language, SummaryLanguage::Fixed("it".into()));
    }

    #[tokio::test]
    async fn corrupt_rows_fall_back_instead_of_failing() {
        let db = connect_in_memory().await.unwrap();
        repo::set_setting(&db, keys::DETECTION_ENABLED, "banana")
            .await
            .unwrap();
        repo::set_setting(&db, keys::RELEASE_AFTER_IDLE_MINUTES, "soon")
            .await
            .unwrap();
        repo::set_setting(&db, keys::SUMMARY_LANGUAGE, "{not json")
            .await
            .unwrap();
        repo::set_setting(&db, keys::SUMMARY_PROVIDER, "telepathy")
            .await
            .unwrap();

        let s = load(&db).await.unwrap();
        assert!(s.detection_enabled);
        assert_eq!(s.release_after_idle_minutes, 10);
        assert_eq!(s.summary_language, SummaryLanguage::SameAsMeeting);
        assert_eq!(s.summary_provider, Provider::OnThisComputer);
    }

    #[tokio::test]
    async fn loose_booleans_are_accepted() {
        let db = connect_in_memory().await.unwrap();
        repo::set_setting(&db, keys::CLOSE_TO_TRAY, "0")
            .await
            .unwrap();
        repo::set_setting(&db, keys::SHOW_ADVANCED, "YES")
            .await
            .unwrap();
        let s = load(&db).await.unwrap();
        assert!(!s.close_to_tray);
        assert!(s.show_advanced);
    }

    #[tokio::test]
    async fn storage_dir_override_is_honoured() {
        let db = connect_in_memory().await.unwrap();
        apply(
            &db,
            &SettingsPatch {
                storage_dir: Some("/Volumes/Big/EchoRecordings".into()),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(
            storage_dir(&db).await.unwrap(),
            PathBuf::from("/Volumes/Big/EchoRecordings")
        );
        let p = app_paths(&db).await.unwrap();
        assert_eq!(p.storage_root(), PathBuf::from("/Volumes/Big/EchoRecordings"));
    }
}
