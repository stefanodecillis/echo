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
use crate::types::{
    Provider, Settings, SettingsPatch, SummaryLanguage, VocabularySource, VocabularyWord,
};

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
    /// The speech weights that have been through a full load on this machine,
    /// and the build of Echo that loaded them: `file name @ build identity`,
    /// written by [`crate::asr::models::mark_warmed`]. The record that the
    /// one-time setup those weights need here has been paid: on Apple silicon
    /// the first load of a model builds its encoder for this particular
    /// machine, which took sixteen minutes on 2026-08-24 — in the middle of a
    /// meeting, with nothing on screen to say so.
    ///
    /// The build is part of the value because the cache the OS keeps is keyed on
    /// the binary: a rebuilt app with untouched weights compiled all over again
    /// on 2026-08-26, and a marker naming only the model file called that warm
    /// (see [`crate::asr::models::warm_up_needed`]).
    pub const SPEECH_WARMED_MODEL: &str = "speech_warmed_model";
    /// The weights, and the build, Echo has *tried* to set up — same shape as
    /// [`SPEECH_WARMED_MODEL`]. Written before the load starts, so a crash
    /// halfway through that compile cannot turn into a quarter of an hour of it
    /// at every launch: the row that crash interrupted is settled as failed
    /// rather than requeued ([`crate::db::repo::requeue_orphaned_jobs`]), and
    /// this is what keeps anything from queueing a fresh one. Taken back again
    /// when the attempt turned out to have failed before the compile could have
    /// started (see [`crate::asr::models::mark_warm_attempted`]).
    pub const SPEECH_WARM_ATTEMPTED: &str = "speech_warm_attempted";
    /// The words somebody typed into "Words Echo should know", as a JSON array
    /// of strings, newest last. See [`crate::settings::words_to_know`].
    pub const VOCABULARY_TYPED: &str = "vocabulary_typed";
    /// The words somebody took *off* that list, as a JSON array of strings.
    ///
    /// This exists because half the list is not stored at all: the names of
    /// enrolled people are read from the `people` table every time, so a rename
    /// is picked up for free and a name can never go stale. The cost of deriving
    /// them is that deleting one would achieve nothing — the next launch would
    /// put it straight back. This is the record that it was deleted on purpose.
    pub const VOCABULARY_REMOVED: &str = "vocabulary_removed";
}

/// Recaps are written on their own once a meeting ends. The happy path is
/// "notification → click → Start → recap appears" (mantra 4), which is only true
/// if nobody has to find a switch first. Someone who turned it off has `false`
/// stored, and a stored value always wins over this.
pub const DEFAULT_AUTO_SUMMARIZE: bool = true;

/// Read every setting, filling in defaults for anything missing.
pub async fn load(db: &Db) -> Result<Settings, DbError> {
    let raw = repo::all_settings(db).await?;
    let mut s = Settings {
        auto_summarize: DEFAULT_AUTO_SUMMARIZE,
        ..Settings::default()
    };

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
    if let Some(v) = get(keys::ACCURACY_LEVEL_ID).filter(|v| !v.is_empty()) {
        // Resolved against the catalog, never taken raw. A database written by
        // an older build can name a level that no longer exists — the levels
        // called "faster" and "fastest" were retired on 2026-08-20 when Echo
        // went to one model — and everything downstream of here treats this as
        // a level it can look up. Handing it a name the catalog does not know
        // turns "your speech download" into a job that fails with nothing a
        // person can do about it.
        s.accuracy_level_id = crate::asr::catalog::preset_or_default(v).id.to_string();
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

// ---------------------------------------------------------------------------
// Words Echo should know
// ---------------------------------------------------------------------------

/// Everything in the vocabulary, in the order the decoder should be offered it.
///
/// Two halves, and only one of them is stored:
///
/// * what somebody typed, from [`keys::VOCABULARY_TYPED`], first — a word
///   worth typing is worth more than one Echo guessed at;
/// * then the name of every enrolled person, read live from the `people` table.
///   Those are the words this feature was born for: "Gianluca" came back from
///   the 2026-08-24 meeting as "Jan Luca", and Echo had been told that name
///   months earlier, by somebody saving that voice.
///
/// Minus anything in [`keys::VOCABULARY_REMOVED`], which is how a derived name
/// stays deleted.
///
/// A row that will not parse is an empty list, never an error: nothing about
/// this is worth failing a meeting over.
pub async fn words_to_know(db: &Db) -> Result<Vec<VocabularyWord>, DbError> {
    let typed = stored_list(db, keys::VOCABULARY_TYPED).await;
    let removed = stored_list(db, keys::VOCABULARY_REMOVED).await;
    let people = repo::list_people(db).await?;

    let mut words: Vec<VocabularyWord> = Vec::new();
    let mut push = |word: &str, source: VocabularySource| {
        let word = word.trim();
        if word.is_empty() || words.len() >= crate::asr::glossary::MAX_ENTRIES {
            return;
        }
        if removed.iter().any(|r| same_word(r, word)) {
            return;
        }
        if words.iter().any(|w| same_word(&w.word, word)) {
            return;
        }
        words.push(VocabularyWord {
            word: word.to_string(),
            source,
        });
    };
    for word in &typed {
        push(word, VocabularySource::Typed);
    }
    for person in &people {
        push(&person.name, VocabularySource::Person);
    }
    Ok(words)
}

/// The same list, ready to prompt and match with. Never fails: a database that
/// cannot answer leaves the decoder exactly as it was before this feature.
pub async fn glossary(db: &Db) -> crate::asr::glossary::Glossary {
    match words_to_know(db).await {
        Ok(words) => crate::asr::glossary::Glossary::new(words.into_iter().map(|w| w.word)),
        Err(error) => {
            tracing::warn!(%error, "could not read the words Echo should know");
            crate::asr::glossary::Glossary::default()
        }
    }
}

/// Add a word, and return the list as it now stands.
///
/// Adding a word that was removed earlier takes it off the removed list, so the
/// two commands are exact opposites however many times they are used.
pub async fn add_word_to_know(db: &Db, word: &str) -> Result<Vec<VocabularyWord>, DbError> {
    let word = word.trim();
    let mut typed = stored_list(db, keys::VOCABULARY_TYPED).await;
    let mut removed = stored_list(db, keys::VOCABULARY_REMOVED).await;
    removed.retain(|r| !same_word(r, word));
    if !word.is_empty() && !typed.iter().any(|t| same_word(t, word)) {
        typed.push(word.to_string());
    }
    typed.truncate(crate::asr::glossary::MAX_ENTRIES);
    store_list(db, keys::VOCABULARY_TYPED, &typed).await?;
    store_list(db, keys::VOCABULARY_REMOVED, &removed).await?;
    words_to_know(db).await
}

/// Take a word off the list — whichever half it came from — and return what is
/// left.
pub async fn remove_word_to_know(db: &Db, word: &str) -> Result<Vec<VocabularyWord>, DbError> {
    let word = word.trim();
    let mut typed = stored_list(db, keys::VOCABULARY_TYPED).await;
    let mut removed = stored_list(db, keys::VOCABULARY_REMOVED).await;
    typed.retain(|t| !same_word(t, word));
    if !word.is_empty() && !removed.iter().any(|r| same_word(r, word)) {
        removed.push(word.to_string());
    }
    removed.truncate(crate::asr::glossary::MAX_ENTRIES);
    store_list(db, keys::VOCABULARY_TYPED, &typed).await?;
    store_list(db, keys::VOCABULARY_REMOVED, &removed).await?;
    words_to_know(db).await
}

/// Two spellings of the same word as far as this list is concerned. Case and
/// surrounding space only — "Langola" and "Langola" are one entry, "Langola"
/// and "Lovabile" are two.
fn same_word(a: &str, b: &str) -> bool {
    a.trim().to_lowercase() == b.trim().to_lowercase()
}

async fn stored_list(db: &Db, key: &str) -> Vec<String> {
    repo::get_setting(db, key)
        .await
        .ok()
        .flatten()
        .and_then(|raw| serde_json::from_str::<Vec<String>>(&raw).ok())
        .unwrap_or_default()
}

async fn store_list(db: &Db, key: &str, words: &[String]) -> Result<(), DbError> {
    let json = serde_json::to_string(words).unwrap_or_else(|_| "[]".to_string());
    repo::set_setting(db, key, &json).await
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
        assert!(
            s.auto_summarize,
            "a recap is written on its own unless someone turned that off"
        );
        assert!(
            !s.storage_dir.is_empty(),
            "storage location must always resolve"
        );
        assert!(PathBuf::from(&s.storage_dir).is_absolute());
    }

    /// The upgrade path. Someone who had picked "Faster" before 2026-08-20 has
    /// that name in their database, and it no longer names anything — so it has
    /// to resolve to the level that does, not travel on to the download job as
    /// an id nothing can look up.
    #[tokio::test]
    async fn a_level_from_an_older_build_resolves_to_the_one_that_exists() {
        let db = connect_in_memory().await.unwrap();
        for retired in ["faster", "fastest", "", "something-invented"] {
            repo::set_setting(&db, keys::ACCURACY_LEVEL_ID, retired)
                .await
                .unwrap();
            let s = load(&db).await.unwrap();
            assert_eq!(
                s.accuracy_level_id,
                crate::asr::catalog::DEFAULT_PRESET_ID,
                "{retired:?} must not reach anything that looks levels up"
            );
            // And the thing it reaches can in fact be looked up.
            assert!(crate::asr::catalog::preset(&s.accuracy_level_id).is_some());
        }
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
    async fn turning_automatic_recaps_off_is_remembered() {
        let db = connect_in_memory().await.unwrap();
        let after = apply(
            &db,
            &SettingsPatch {
                auto_summarize: Some(false),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert!(!after.auto_summarize, "a stored 'no' beats the default");
        assert!(!load(&db).await.unwrap().auto_summarize);

        // And it can be turned back on.
        apply(
            &db,
            &SettingsPatch {
                auto_summarize: Some(true),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert!(load(&db).await.unwrap().auto_summarize);
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
        // A row left behind by a setting that no longer exists: every install
        // that ran a build before mantra 1's 2026-08-20 amendment has one of
        // these. An unrecognised key is ignored, never a reason to fail.
        repo::set_setting(&db, "release_after_idle_minutes", "soon")
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

    // -----------------------------------------------------------------------
    // Words Echo should know
    // -----------------------------------------------------------------------

    /// The half nobody has to type: Echo already knows the names of the people
    /// whose voices it was asked to remember, and those are exactly the words it
    /// gets wrong — "Gianluca" came back from the 2026-08-24 meeting as "Joe
    /// Franco".
    #[tokio::test]
    async fn the_names_of_remembered_people_are_in_the_list_without_anybody_typing_them() {
        let db = connect_in_memory().await.unwrap();
        assert!(words_to_know(&db).await.unwrap().is_empty());

        repo::create_person(&db, "Gianluca").await.unwrap();
        let words = words_to_know(&db).await.unwrap();
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].word, "Gianluca");
        assert_eq!(
            words[0].source,
            VocabularySource::Person,
            "a derived name has to be tellable from a typed one"
        );
    }

    /// Typed first, and a typed word is never listed twice because somebody of
    /// that name is also enrolled.
    #[tokio::test]
    async fn what_was_typed_comes_first_and_a_name_is_only_listed_once() {
        let db = connect_in_memory().await.unwrap();
        repo::create_person(&db, "Marco").await.unwrap();
        add_word_to_know(&db, "Langola").await.unwrap();
        add_word_to_know(&db, "marco").await.unwrap();

        let words = words_to_know(&db).await.unwrap();
        assert_eq!(
            words
                .iter()
                .map(|w| (w.word.as_str(), w.source))
                .collect::<Vec<_>>(),
            vec![
                ("Langola", VocabularySource::Typed),
                ("marco", VocabularySource::Typed),
            ],
            "the typed spelling wins, and the enrolled name does not come back a second time"
        );
    }

    /// The reason the removed list exists at all. Half the vocabulary is derived
    /// from the `people` table every time it is read, so deleting a derived name
    /// has to be *remembered* or the next read would put it straight back.
    #[tokio::test]
    async fn removing_a_name_echo_added_itself_makes_it_stay_removed() {
        let db = connect_in_memory().await.unwrap();
        repo::create_person(&db, "Gianluca").await.unwrap();

        let left = remove_word_to_know(&db, "Gianluca").await.unwrap();
        assert!(left.is_empty());
        // The read that would resurrect it, and every one after it.
        assert!(words_to_know(&db).await.unwrap().is_empty());
        assert!(words_to_know(&db).await.unwrap().is_empty());

        // Adding it back is the exact opposite, however many times either
        // happens — and it is typed now, because somebody typed it.
        let back = add_word_to_know(&db, "Gianluca").await.unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].source, VocabularySource::Typed);
    }

    #[tokio::test]
    async fn a_typed_word_is_added_once_and_removed_for_good() {
        let db = connect_in_memory().await.unwrap();
        add_word_to_know(&db, "Obsidara").await.unwrap();
        add_word_to_know(&db, "  Obsidara  ").await.unwrap();
        assert_eq!(words_to_know(&db).await.unwrap().len(), 1);

        assert!(remove_word_to_know(&db, "Obsidara")
            .await
            .unwrap()
            .is_empty());
        assert!(words_to_know(&db).await.unwrap().is_empty());
    }

    /// Nothing about a list of words is worth failing to open the app for.
    #[tokio::test]
    async fn a_corrupt_vocabulary_row_is_an_empty_list_rather_than_an_error() {
        let db = connect_in_memory().await.unwrap();
        repo::set_setting(&db, keys::VOCABULARY_TYPED, "{not json")
            .await
            .unwrap();
        repo::set_setting(&db, keys::VOCABULARY_REMOVED, "[1, 2, 3]")
            .await
            .unwrap();
        assert!(words_to_know(&db).await.unwrap().is_empty());
        assert!(glossary(&db).await.is_empty());

        // …and writing to it puts it back in a state that parses.
        add_word_to_know(&db, "Langola").await.unwrap();
        assert_eq!(words_to_know(&db).await.unwrap().len(), 1);
    }

    /// The list feeds a prompt with a hard cap on it, so the list itself has one
    /// too — otherwise a thousand words would be stored to have fifty read.
    #[tokio::test]
    async fn the_list_stops_at_the_cap() {
        let db = connect_in_memory().await.unwrap();
        for i in 0..crate::asr::glossary::MAX_ENTRIES + 20 {
            add_word_to_know(&db, &format!("Parola{i}")).await.unwrap();
        }
        assert_eq!(
            words_to_know(&db).await.unwrap().len(),
            crate::asr::glossary::MAX_ENTRIES
        );
    }

    /// The one every decode path leans on: with nothing in the list, the
    /// vocabulary is not in the way of anything.
    #[tokio::test]
    async fn an_untouched_install_has_nothing_to_say_to_the_decoder() {
        let db = connect_in_memory().await.unwrap();
        let glossary = glossary(&db).await;
        assert!(glossary.is_empty());
        assert_eq!(glossary.prompt(), None);
        assert_eq!(
            glossary.context(Some("e il secondo punto")).as_deref(),
            Some("e il secondo punto")
        );
        assert_eq!(glossary.correct("Nongula e Ingola"), None);
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
        assert_eq!(
            p.storage_root(),
            PathBuf::from("/Volumes/Big/EchoRecordings")
        );
    }
}
