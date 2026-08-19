//! Every IPC command, in one place.
//!
//! Each command is thin: validate the input, call one module function, map the
//! error to a [`UiError`] the UI can act on. No business logic lives here.
//!
//! Rules:
//! * Commands are **idempotent**. Start while recording returns the recording in
//!   progress; Stop while idle returns `None`. Double-clicking is safe.
//! * Input from the webview is untrusted. Ids, paths and URLs get checked here
//!   before they reach a module (review finding 38).
//! * Errors carry a sentence the person can act on. Technical detail goes in
//!   `detail`, which only Settings → Advanced renders.
//! * The mirror of this file is `src/lib/ipc.ts`; the two change together.

use tauri::{AppHandle, Manager, State};

use crate::db::repo;
use crate::paths;
use crate::types::*;
use crate::{asr, audio, detect, diarize, export, secrets, session, settings, summarize};

/// Shared state, built once at launch in `lib.rs`.
pub struct AppState {
    pub db: crate::db::Db,
    pub paths: paths::AppPaths,
    pub session: session::SessionManager,
    pub detect: detect::Watcher,
}

// ---------------------------------------------------------------------------
// Error plumbing. Every module error becomes a UiError here, so the modules
// themselves never have to know about the UI.
// ---------------------------------------------------------------------------

impl From<audio::AudioError> for UiError {
    fn from(err: audio::AudioError) -> Self {
        use audio::AudioError as A;
        match err {
            A::PermissionDenied => UiError::new(
                UiErrorKind::PermissionNeeded,
                "Echo needs permission to use your microphone before it can listen.",
            )
            .with_action(UiErrorAction::OpenMicrophoneSettings),
            A::NoInputDevice => UiError::new(
                UiErrorKind::NotReady,
                "Echo can't find a microphone. Plug one in, or pick one in Settings.",
            )
            .with_action(UiErrorAction::OpenStorageSettings),
            A::SystemAudioUnsupported(detail) => UiError::new(
                UiErrorKind::NotReady,
                "This computer can't share what it plays, so Echo will record only your microphone.",
            )
            .with_detail(detail),
            A::Write(detail) => UiError::new(
                UiErrorKind::Storage,
                "Echo couldn't save the recording. Check there is space on the drive you chose.",
            )
            .with_detail(detail)
            .with_action(UiErrorAction::OpenStorageSettings),
            other => UiError::unexpected(other.to_string()),
        }
    }
}

impl From<asr::AsrError> for UiError {
    fn from(err: asr::AsrError) -> Self {
        use asr::AsrError as A;
        match err {
            A::NotInstalled => UiError::new(
                UiErrorKind::NotReady,
                "Echo still needs a one-time download before it can write down speech.",
            )
            .with_action(UiErrorAction::OpenSpeechSettings),
            A::NotEnoughSpace { needed, available } => UiError::new(
                UiErrorKind::Storage,
                "There isn't enough room on this computer for that download.",
            )
            .with_detail(format!("needs {needed} bytes, {available} free"))
            .with_action(UiErrorAction::OpenStorageSettings),
            A::IntegrityCheckFailed => UiError::new(
                UiErrorKind::Network,
                "That download arrived damaged. Echo will start it again.",
            )
            .with_action(UiErrorAction::Retry),
            A::Download(detail) => UiError::new(
                UiErrorKind::Network,
                "The download stopped. Echo will pick up where it left off when you try again.",
            )
            .with_detail(detail)
            .with_action(UiErrorAction::Retry),
            A::Cancelled => UiError::new(UiErrorKind::Cancelled, "Stopped."),
            // A dropped live utterance never reaches the UI — catch-up picks it
            // up from disk — but if one ever does, it is not a failure.
            A::QueueFull => UiError::new(UiErrorKind::Cancelled, "Stopped."),
            A::UnknownAsset(_) | A::NotUsedHere(_) => UiError::not_found("Echo doesn't need that."),
            // A full disk has to say "out of space", not "something went wrong".
            A::Db(e) => e.into(),
            other => UiError::unexpected(other.to_string()),
        }
    }
}

impl From<diarize::DiarizeError> for UiError {
    fn from(err: diarize::DiarizeError) -> Self {
        use diarize::DiarizeError as D;
        match err {
            D::NotInstalled => UiError::new(
                UiErrorKind::NotReady,
                "Echo needs a one-time download before it can tell voices apart.",
            )
            .with_action(UiErrorAction::OpenSpeechSettings),
            D::Cancelled => UiError::new(UiErrorKind::Cancelled, "Stopped."),
            D::CannotMerge(detail) => UiError::invalid(
                "Those two can't be the same person — they're from different meetings.",
            )
            .with_detail(detail),
            // Normal deferral, not a fault: the recording gets the machine.
            D::Yielded => UiError::new(
                UiErrorKind::Cancelled,
                "Echo will finish working out who said what once your recording is done.",
            ),
            other => UiError::unexpected(other.to_string()),
        }
    }
}

impl From<summarize::SummarizeError> for UiError {
    fn from(err: summarize::SummarizeError) -> Self {
        use summarize::SummarizeError as S;
        match err {
            S::NoTranscript => UiError::new(
                UiErrorKind::NotReady,
                "There's nothing written down for this meeting yet.",
            ),
            S::Unreachable(detail) => UiError::new(
                UiErrorKind::Network,
                "Echo couldn't reach the place that writes your recaps. Check it's running.",
            )
            .with_detail(detail)
            .with_action(UiErrorAction::OpenSummarySettings),
            S::MissingCredential => UiError::new(
                UiErrorKind::Credential,
                "Echo needs a key for that service before it can write recaps with it.",
            )
            .with_action(UiErrorAction::OpenSummarySettings),
            S::Timeout => UiError::new(
                UiErrorKind::Network,
                "That took too long. Try again, or pick something else to write the recap.",
            )
            .with_action(UiErrorAction::Retry),
            S::BadJson => UiError::new(
                UiErrorKind::Unexpected,
                "Echo got the recap but couldn't read the task list. Try writing it again.",
            )
            .with_action(UiErrorAction::Retry),
            S::Cancelled => UiError::new(UiErrorKind::Cancelled, "Stopped."),
            other => UiError::unexpected(other.to_string()),
        }
    }
}

impl From<export::ExportError> for UiError {
    fn from(err: export::ExportError) -> Self {
        use export::ExportError as E;
        match err {
            E::Empty => UiError::new(
                UiErrorKind::NotReady,
                "There's nothing to save for this meeting yet.",
            ),
            E::Write(detail) => UiError::new(
                UiErrorKind::Storage,
                "Echo couldn't write that file. Try a different place to save it.",
            )
            .with_detail(detail),
            E::PdfUnavailable(detail) => UiError::new(
                UiErrorKind::NotReady,
                "PDF isn't available on this computer. Word or Markdown will work.",
            )
            .with_detail(detail),
            E::Db(e) => e.into(),
            other => UiError::unexpected(other.to_string()),
        }
    }
}

impl From<session::SessionError> for UiError {
    fn from(err: session::SessionError) -> Self {
        use session::SessionError as S;
        match err {
            S::NoAudioSources(detail) => UiError::new(
                UiErrorKind::PermissionNeeded,
                "Echo couldn't hear anything, so it didn't start. Check its microphone permission.",
            )
            .with_detail(detail)
            .with_action(UiErrorAction::OpenMicrophoneSettings),
            S::Storage(detail) => UiError::new(
                UiErrorKind::Storage,
                "Echo couldn't save to the place you chose for recordings.",
            )
            .with_detail(detail)
            .with_action(UiErrorAction::OpenStorageSettings),
            S::Db(e) => e.into(),
            other => UiError::unexpected(other.to_string()),
        }
    }
}

impl From<detect::DetectError> for UiError {
    fn from(err: detect::DetectError) -> Self {
        UiError::unexpected(err.to_string())
    }
}

impl From<paths::PathsError> for UiError {
    fn from(err: paths::PathsError) -> Self {
        UiError::new(
            UiErrorKind::Storage,
            "Echo can't use that folder for recordings. Pick another one.",
        )
        .with_detail(err.to_string())
        .with_action(UiErrorAction::OpenStorageSettings)
    }
}

/// Reject an id that is not a UUID before it reaches a query.
fn check_id(id: &str) -> CmdResult<()> {
    if id.len() == 36 && uuid::Uuid::parse_str(id).is_ok() {
        Ok(())
    } else {
        Err(UiError::invalid("Echo couldn't find that."))
    }
}

fn trim_limited(value: &str, max: usize, what: &str) -> CmdResult<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(UiError::invalid(format!("{what} can't be empty.")));
    }
    if trimmed.chars().count() > max {
        return Err(UiError::invalid(format!("{what} is too long.")));
    }
    Ok(trimmed.to_string())
}

/// Where one meeting's audio actually lives.
///
/// The folder recorded on the meeting row wins over the current storage
/// location, because the person may have moved that location since — and
/// recordings are never moved out from under them (finding: storage location).
/// The row is only trusted when it names this meeting, so a row that recorded
/// the root by accident cannot point deletion at the root.
async fn meeting_audio_dir(state: &AppState, meeting_id: &str) -> std::path::PathBuf {
    if let Ok(Some(meeting)) = repo::get_meeting(&state.db, meeting_id).await {
        let recorded = std::path::PathBuf::from(&meeting.audio_dir);
        let names_this_meeting = recorded
            .file_name()
            .map(|name| name == std::ffi::OsStr::new(meeting_id))
            .unwrap_or(false);
        if names_this_meeting {
            return recorded;
        }
    }
    state.paths.meeting_dir(meeting_id)
}

/// Tell every open screen a setting changed, so two windows (or the tray and the
/// Settings screen) cannot show different answers.
fn announce_settings(state: &AppState, settings: &Settings) {
    use crate::session::ports::{EventSink, UiEvent};
    state
        .session
        .events()
        .emit(UiEvent::SettingsChanged(settings.clone()));
}

// ---------------------------------------------------------------------------
// Capture
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn start_recording(
    state: State<'_, AppState>,
    options: Option<StartRecordingOptions>,
) -> CmdResult<Id> {
    Ok(state.session.start(options.unwrap_or_default()).await?)
}

#[tauri::command]
pub async fn stop_recording(state: State<'_, AppState>) -> CmdResult<Option<Id>> {
    Ok(state.session.stop().await?)
}

#[tauri::command]
pub async fn pause_recording(state: State<'_, AppState>) -> CmdResult<CaptureStatus> {
    Ok(state.session.pause().await?)
}

#[tauri::command]
pub async fn resume_recording(state: State<'_, AppState>) -> CmdResult<CaptureStatus> {
    Ok(state.session.resume().await?)
}

#[tauri::command]
pub async fn get_capture_state(state: State<'_, AppState>) -> CmdResult<CaptureStatus> {
    Ok(state.session.status().await)
}

/// The live "Flag action item" button.
#[tauri::command]
pub async fn add_marker(
    state: State<'_, AppState>,
    kind: Option<MarkerKind>,
    note: Option<String>,
) -> CmdResult<Marker> {
    let note = note.map(|n| n.trim().chars().take(500).collect::<String>());
    Ok(state
        .session
        .add_marker(kind.unwrap_or(MarkerKind::ActionItem), note)
        .await?)
}

#[tauri::command]
pub async fn list_interrupted_meetings(state: State<'_, AppState>) -> CmdResult<Vec<Meeting>> {
    // Through the session, so asking also moves capture into Recovering and
    // emits the recovery event — otherwise the banner never appears for someone
    // who opens the window after launch.
    state.session.find_interrupted().await?;
    Ok(repo::list_interrupted_meetings(&state.db).await?)
}

#[tauri::command]
pub async fn resolve_interrupted_meeting(
    state: State<'_, AppState>,
    meeting_id: Id,
    action: RecoveryAction,
) -> CmdResult<()> {
    check_id(&meeting_id)?;
    Ok(state
        .session
        .resolve_interrupted(&meeting_id, action)
        .await?)
}

// ---------------------------------------------------------------------------
// Meetings, transcript, search
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn list_meetings(
    state: State<'_, AppState>,
    query: Option<MeetingQuery>,
) -> CmdResult<Vec<MeetingSummary>> {
    Ok(repo::list_meetings(&state.db, &query.unwrap_or_default()).await?)
}

#[tauri::command]
pub async fn get_meeting(state: State<'_, AppState>, meeting_id: Id) -> CmdResult<MeetingDetail> {
    check_id(&meeting_id)?;
    let mut detail = repo::get_meeting_detail(&state.db, &meeting_id).await?;
    detail.audio_bytes = paths::dir_size_bytes(&meeting_audio_dir(&state, &meeting_id).await);
    Ok(detail)
}

#[tauri::command]
pub async fn update_meeting_title(
    state: State<'_, AppState>,
    meeting_id: Id,
    title: String,
) -> CmdResult<()> {
    check_id(&meeting_id)?;
    let title = trim_limited(&title, 200, "A meeting name")?;
    Ok(repo::update_meeting_title(&state.db, &meeting_id, &title).await?)
}

/// Delete a meeting, or just its audio. Files are removed after the rows.
#[tauri::command]
pub async fn delete_meeting(
    state: State<'_, AppState>,
    meeting_id: Id,
    mode: DeleteMode,
) -> CmdResult<()> {
    check_id(&meeting_id)?;
    // Stop transcribing audio that is on its way to the bin.
    state.session.forget_meeting(&meeting_id);
    let dir = meeting_audio_dir(&state, &meeting_id).await;
    let orphaned = match mode {
        DeleteMode::AudioOnly => repo::forget_audio(&state.db, &meeting_id).await?,
        DeleteMode::Everything => repo::delete_meeting(&state.db, &meeting_id).await?,
    };
    for path in orphaned {
        let _ = std::fs::remove_file(path);
    }
    if matches!(mode, DeleteMode::Everything) || dir.exists() {
        let _ = std::fs::remove_dir_all(&dir);
    }
    Ok(())
}

/// Settings → Data → delete everything. Recordings on disk go too.
///
/// It removes the folders Echo created — one per meeting, named after it — and
/// the files the journal recorded, never the folder the person chose. That
/// folder may be `~/Documents` or a drive root, and "delete everything" must
/// mean everything *of Echo's*, not everything in there.
#[tauri::command]
pub async fn delete_all_data(state: State<'_, AppState>) -> CmdResult<()> {
    use std::path::PathBuf;

    let meetings = repo::all_meeting_audio_dirs(&state.db).await?;
    let files = repo::all_audio_file_paths(&state.db).await?;
    repo::delete_all_meetings(&state.db).await?;

    let mut removed: Vec<PathBuf> = Vec::new();
    let mut remove_meeting_dir = |candidate: PathBuf, meeting_id: &str| {
        let named_after_the_meeting = candidate
            .file_name()
            .map(|name| name == std::ffi::OsStr::new(meeting_id))
            .unwrap_or(false);
        if named_after_the_meeting && candidate.is_dir() {
            let _ = std::fs::remove_dir_all(&candidate);
            removed.push(candidate);
        }
    };
    for (meeting_id, audio_dir) in &meetings {
        // The recorded folder, and where it would be today if the person moved
        // the storage location since.
        if !audio_dir.is_empty() {
            remove_meeting_dir(PathBuf::from(audio_dir), meeting_id);
        }
        remove_meeting_dir(state.paths.meeting_dir(meeting_id), meeting_id);
    }
    // Anything the journal knew about that was not inside one of those folders,
    // for instance after the storage location moved mid-life.
    for file in files {
        let path = PathBuf::from(file);
        if removed.iter().any(|dir| path.starts_with(dir)) {
            continue;
        }
        let _ = std::fs::remove_file(&path);
    }
    // Folders left behind by a meeting whose row is already gone. Only ones
    // named like a meeting; a stranger's file in there is not Echo's to delete.
    if let Ok(entries) = std::fs::read_dir(state.paths.storage_root()) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let is_meeting_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false)
                && uuid::Uuid::parse_str(&name).is_ok();
            if is_meeting_dir {
                let _ = std::fs::remove_dir_all(entry.path());
            }
        }
    }
    let _ = state.paths.ensure();

    crate::db::pool::checkpoint(&state.db)
        .await
        .map_err(UiError::from)?;
    crate::db::pool::vacuum(&state.db)
        .await
        .map_err(UiError::from)?;
    Ok(())
}

#[tauri::command]
pub async fn get_transcript(
    state: State<'_, AppState>,
    query: TranscriptQuery,
) -> CmdResult<Vec<Segment>> {
    check_id(&query.meeting_id)?;
    Ok(repo::get_segments(&state.db, &query).await?)
}

#[tauri::command]
pub async fn search_transcripts(
    state: State<'_, AppState>,
    query: SearchQuery,
) -> CmdResult<Vec<SearchHit>> {
    if let Some(id) = &query.meeting_id {
        check_id(id)?;
    }
    Ok(repo::search_segments(&state.db, &query).await?)
}

#[tauri::command]
pub async fn get_markers(state: State<'_, AppState>, meeting_id: Id) -> CmdResult<Vec<Marker>> {
    check_id(&meeting_id)?;
    Ok(repo::list_markers(&state.db, &meeting_id).await?)
}

/// Path of the mixed file for playback. `None` when the audio is gone.
#[tauri::command]
pub async fn get_playback_path(
    state: State<'_, AppState>,
    meeting_id: Id,
) -> CmdResult<Option<String>> {
    check_id(&meeting_id)?;
    // What the mixdown recorded wins, then the folder this meeting was recorded
    // into: a meeting captured before the storage location moved still plays.
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(Some(meeting)) = repo::get_meeting(&state.db, &meeting_id).await {
        if let Some(mixed) = meeting.mixed_path.filter(|p| !p.is_empty()) {
            candidates.push(std::path::PathBuf::from(mixed));
        }
    }
    let dir = meeting_audio_dir(&state, &meeting_id).await;
    candidates.push(dir.join(format!(
        "mixed.{}",
        audio::writer::DEFAULT_CHUNK_FORMAT.extension()
    )));
    candidates.push(state.paths.mixed_path(&meeting_id));
    Ok(candidates
        .into_iter()
        .find(|path| path.exists())
        .map(|path| path.to_string_lossy().into_owned()))
}

#[tauri::command]
pub async fn get_storage_report(state: State<'_, AppState>) -> CmdResult<StorageReport> {
    let root = &state.paths.storage_root();
    let audio_bytes = paths::dir_size_bytes(root);
    let database_bytes = crate::db::pool::database_bytes(&state.paths.db_path);
    let speech_asset_bytes = paths::dir_size_bytes(&state.paths.assets_dir);
    let log_bytes = paths::dir_size_bytes(&state.paths.log_dir);
    let largest_meetings = repo::list_meetings(
        &state.db,
        &MeetingQuery {
            limit: Some(10),
            ..Default::default()
        },
    )
    .await?;
    Ok(StorageReport {
        root: root.to_string_lossy().into_owned(),
        audio_bytes,
        database_bytes,
        speech_asset_bytes,
        log_bytes,
        total_bytes: audio_bytes + database_bytes + speech_asset_bytes + log_bytes,
        free_bytes: paths::free_space_bytes(root),
        meeting_count: repo::count_meetings(&state.db).await?,
        largest_meetings,
    })
}

// ---------------------------------------------------------------------------
// Speakers
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn list_speakers(state: State<'_, AppState>, meeting_id: Id) -> CmdResult<Vec<Speaker>> {
    check_id(&meeting_id)?;
    Ok(repo::list_speakers(&state.db, &meeting_id).await?)
}

#[tauri::command]
pub async fn rename_speaker(
    state: State<'_, AppState>,
    speaker_id: Id,
    display_name: String,
) -> CmdResult<()> {
    check_id(&speaker_id)?;
    let name = trim_limited(&display_name, 80, "A name")?;
    // Through `diarize`, not straight to the table: that is where the alias graph
    // and the empty-name rule live.
    Ok(diarize::rename(&state.db, &speaker_id, &name).await?)
}

#[tauri::command]
pub async fn merge_speakers(state: State<'_, AppState>, from_id: Id, into_id: Id) -> CmdResult<()> {
    check_id(&from_id)?;
    check_id(&into_id)?;
    Ok(diarize::merge(&state.db, &from_id, &into_id).await?)
}

#[tauri::command]
pub async fn unmerge_speaker(state: State<'_, AppState>, speaker_id: Id) -> CmdResult<()> {
    check_id(&speaker_id)?;
    Ok(diarize::unmerge(&state.db, &speaker_id).await?)
}

/// Queue the offline speaker pass. The result replaces the provisional labels.
#[tauri::command]
pub async fn refine_speakers(state: State<'_, AppState>, meeting_id: Id) -> CmdResult<Id> {
    check_id(&meeting_id)?;
    Ok(state
        .session
        .queue_job(Some(&meeting_id), JobKind::Diarize)
        .await?)
}

// ---------------------------------------------------------------------------
// Speech assets
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn list_accuracy_levels(state: State<'_, AppState>) -> CmdResult<Vec<AccuracyLevel>> {
    Ok(asr::models::list_accuracy_levels(&state.db).await?)
}

#[tauri::command]
pub async fn select_accuracy_level(
    state: State<'_, AppState>,
    level_id: String,
) -> CmdResult<Settings> {
    let level_id = trim_limited(&level_id, 64, "That choice")?;
    let updated = settings::apply(
        &state.db,
        &SettingsPatch {
            accuracy_level_id: Some(level_id),
            ..Default::default()
        },
    )
    .await?;
    announce_settings(&state, &updated);
    Ok(updated)
}

/// Start (or attach to) the one-time download for a quality preset.
#[tauri::command]
pub async fn download_speech_assets(state: State<'_, AppState>, level_id: String) -> CmdResult<Id> {
    let level_id = trim_limited(&level_id, 64, "That choice")?;
    // The choice travels with the job, so downloading one preset never switches
    // the person over to weights that are not on disk yet — a recording started
    // in the meantime keeps using whatever is installed. Switching presets is
    // `select_accuracy_level`, and that is a separate decision.
    Ok(state
        .session
        .queue_job_with_payload(None, JobKind::Download, &level_id)
        .await?)
}

#[tauri::command]
pub async fn cancel_speech_download(state: State<'_, AppState>, asset_id: Id) -> CmdResult<()> {
    let _ = &state.db;
    Ok(asr::models::cancel_download(&asset_id).await?)
}

#[tauri::command]
pub async fn get_speech_readiness(state: State<'_, AppState>) -> CmdResult<SpeechReadiness> {
    let mut readiness = asr::models::readiness(&state.db, &state.paths).await?;
    // `models` reads the catalog and the disk; only the session knows whether
    // anything is actually in memory right now.
    readiness.loaded = state.session.speech_loaded();
    Ok(readiness)
}

/// Load speech understanding now, so the first words of a meeting are not slow.
/// Explicit user action; nothing loads on its own (mantra 1).
#[tauri::command]
pub async fn prewarm_speech(state: State<'_, AppState>) -> CmdResult<()> {
    Ok(state.session.prewarm_speech().await?)
}

/// Give the memory back.
#[tauri::command]
pub async fn release_speech(state: State<'_, AppState>) -> CmdResult<()> {
    state.session.release_idle_resources().await;
    Ok(())
}

/// Settings → Advanced only.
#[tauri::command]
pub async fn list_speech_assets(state: State<'_, AppState>) -> CmdResult<Vec<ModelInfo>> {
    Ok(repo::list_models(&state.db, None).await?)
}

#[tauri::command]
pub async fn remove_speech_asset(state: State<'_, AppState>, asset_id: Id) -> CmdResult<()> {
    Ok(asr::models::remove_asset(&state.db, &state.paths, &asset_id).await?)
}

// ---------------------------------------------------------------------------
// Recaps
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn list_summary_providers(state: State<'_, AppState>) -> CmdResult<Vec<ProviderInfo>> {
    Ok(summarize::list_providers(&state.db).await?)
}

#[tauri::command]
pub async fn test_summary_provider(
    state: State<'_, AppState>,
    provider: Provider,
) -> CmdResult<ProviderTestResult> {
    Ok(summarize::test_provider(&state.db, provider).await?)
}

#[tauri::command]
pub async fn save_summary_provider(
    state: State<'_, AppState>,
    config: ProviderConfig,
) -> CmdResult<()> {
    if let Some(url) = &config.base_url {
        let url = url.trim();
        if url.is_empty() {
            return Err(UiError::invalid("That address doesn't look right."));
        }
        // An address somewhere other than this computer means what was said
        // travels. The person has to have said yes to that first (DESIGN §3).
        if !summarize::is_loopback_url(url) && !config.leaves_machine_acknowledged {
            return Err(UiError::invalid(
                "That address isn't on this computer, so what was said would be sent there. \
                 Confirm you're happy with that first.",
            ));
        }
    }
    Ok(summarize::save_config(&state.db, &config).await?)
}

/// Save a key in the keychain. The key never comes back out over IPC.
#[tauri::command]
pub async fn set_provider_key(provider: Provider, key: String) -> CmdResult<()> {
    let key = trim_limited(&key, 512, "A key")?;
    let account = match provider {
        Provider::Gemini => secrets::accounts::GEMINI_API_KEY,
        Provider::OnThisComputer => {
            return Err(UiError::invalid(
                "Recaps on this computer don't need a key.",
            ))
        }
    };
    Ok(secrets::set(account, &key).await?)
}

#[tauri::command]
pub async fn has_provider_key(provider: Provider) -> CmdResult<bool> {
    match provider {
        Provider::Gemini => Ok(secrets::has(secrets::accounts::GEMINI_API_KEY).await?),
        Provider::OnThisComputer => Ok(true),
    }
}

#[tauri::command]
pub async fn delete_provider_key(provider: Provider) -> CmdResult<()> {
    match provider {
        Provider::Gemini => Ok(secrets::delete(secrets::accounts::GEMINI_API_KEY).await?),
        Provider::OnThisComputer => Ok(()),
    }
}

/// Queue a recap. Progress arrives on the job-progress event.
#[tauri::command]
pub async fn generate_summary(state: State<'_, AppState>, request: SummaryReq) -> CmdResult<Id> {
    check_id(&request.meeting_id)?;
    if let Some(id) = &request.template_id {
        check_id(id)?;
    }
    let meeting_id = request.meeting_id.clone();
    Ok(state
        .session
        .queue_job_with_payload(Some(&meeting_id), JobKind::Summarize, &request)
        .await?)
}

#[tauri::command]
pub async fn list_summaries(state: State<'_, AppState>, meeting_id: Id) -> CmdResult<Vec<Summary>> {
    check_id(&meeting_id)?;
    Ok(repo::list_summaries(&state.db, &meeting_id).await?)
}

#[tauri::command]
pub async fn get_summary(state: State<'_, AppState>, summary_id: Id) -> CmdResult<Summary> {
    check_id(&summary_id)?;
    repo::get_summary(&state.db, &summary_id)
        .await?
        .ok_or_else(|| UiError::not_found("That recap is gone."))
}

#[tauri::command]
pub async fn list_action_items(
    state: State<'_, AppState>,
    meeting_id: Id,
) -> CmdResult<Vec<ActionItem>> {
    check_id(&meeting_id)?;
    Ok(repo::list_action_items(&state.db, &meeting_id).await?)
}

#[tauri::command]
pub async fn update_action_item(
    state: State<'_, AppState>,
    patch: ActionItemPatch,
) -> CmdResult<ActionItem> {
    check_id(&patch.id)?;
    Ok(repo::patch_action_item(&state.db, &patch).await?)
}

// ---------------------------------------------------------------------------
// Recap styles
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn list_templates(state: State<'_, AppState>) -> CmdResult<Vec<Template>> {
    Ok(repo::list_templates(&state.db).await?)
}

#[tauri::command]
pub async fn save_template(
    state: State<'_, AppState>,
    draft: TemplateDraft,
) -> CmdResult<Template> {
    let name = trim_limited(&draft.name, 80, "A name")?;
    let prompt = trim_limited(
        &draft.prompt_md,
        summarize::templates::MAX_PROMPT_CHARS,
        "Those instructions",
    )?;
    if let Some(id) = &draft.id {
        check_id(id)?;
    }
    Ok(repo::upsert_template(&state.db, draft.id.as_deref(), &name, &prompt).await?)
}

#[tauri::command]
pub async fn delete_template(state: State<'_, AppState>, template_id: Id) -> CmdResult<()> {
    check_id(&template_id)?;
    Ok(repo::delete_template(&state.db, &template_id).await?)
}

// ---------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn list_jobs(state: State<'_, AppState>, query: Option<JobQuery>) -> CmdResult<Vec<Job>> {
    Ok(repo::list_jobs(&state.db, &query.unwrap_or_default()).await?)
}

#[tauri::command]
pub async fn cancel_job(state: State<'_, AppState>, job_id: Id) -> CmdResult<()> {
    check_id(&job_id)?;
    Ok(state.session.cancel_job(&job_id).await?)
}

#[tauri::command]
pub async fn retry_job(state: State<'_, AppState>, job_id: Id) -> CmdResult<()> {
    check_id(&job_id)?;
    Ok(state.session.retry_job(&job_id).await?)
}

// ---------------------------------------------------------------------------
// Meeting detection
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn get_detection_status(state: State<'_, AppState>) -> CmdResult<DetectionStatus> {
    Ok(state.detect.status())
}

#[tauri::command]
pub async fn set_detection_enabled(
    state: State<'_, AppState>,
    enabled: bool,
) -> CmdResult<Settings> {
    state.detect.set_enabled(enabled).await?;
    let updated = settings::apply(
        &state.db,
        &SettingsPatch {
            detection_enabled: Some(enabled),
            ..Default::default()
        },
    )
    .await?;
    announce_settings(&state, &updated);
    Ok(updated)
}

#[tauri::command]
pub async fn snooze_detection(state: State<'_, AppState>, minutes: Option<u64>) -> CmdResult<()> {
    let minutes = minutes.unwrap_or(detect::SNOOZE_MINUTES).clamp(1, 24 * 60);
    Ok(state.detect.snooze(minutes).await?)
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn export_meeting(
    state: State<'_, AppState>,
    request: ExportRequest,
) -> CmdResult<ExportResult> {
    check_id(&request.meeting_id)?;
    let destination = request
        .destination
        .as_deref()
        .ok_or_else(|| UiError::invalid("Pick where to save the file first."))?;
    if !std::path::Path::new(destination).is_absolute() {
        return Err(UiError::invalid("Pick where to save the file first."));
    }
    Ok(export::export_meeting(&state.db, &request).await?)
}

#[tauri::command]
pub async fn export_diagnostics(
    state: State<'_, AppState>,
    destination: String,
) -> CmdResult<ExportResult> {
    let dest = std::path::PathBuf::from(&destination);
    if !dest.is_absolute() {
        return Err(UiError::invalid("Pick where to save the file first."));
    }
    Ok(export::export_diagnostics(&state.paths, &dest).await?)
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn get_settings(state: State<'_, AppState>) -> CmdResult<Settings> {
    Ok(settings::load(&state.db).await?)
}

#[tauri::command]
pub async fn update_settings(
    app: AppHandle,
    state: State<'_, AppState>,
    patch: SettingsPatch,
) -> CmdResult<Settings> {
    if let Some(dir) = &patch.storage_dir {
        paths::validate_storage_dir(std::path::Path::new(dir))?;
    }
    if let Some(id) = &patch.summary_template_id {
        check_id(id)?;
    }
    // The login item is registered *before* the value is stored, so a system
    // that refuses leaves the toggle showing the truth rather than a promise
    // Echo cannot keep.
    if let Some(wanted) = patch.launch_at_login {
        match crate::set_launch_at_login(&app, wanted) {
            Ok(actual) if actual == wanted => {}
            Ok(_) | Err(_) => {
                return Err(UiError::new(
                    UiErrorKind::Unexpected,
                    "This computer wouldn't let Echo change what starts at login.",
                ))
            }
        }
    }
    let updated = settings::apply(&state.db, &patch).await?;
    // Where recordings go takes effect immediately for new ones: the resolved
    // layout is shared, so the session, the job runner and the storage report
    // all follow. Recordings already on disk stay exactly where they are —
    // every meeting row carries its own folder — so nothing already captured
    // becomes unreachable.
    if patch.storage_dir.is_some() {
        let wanted = std::path::PathBuf::from(&updated.storage_dir);
        if wanted != state.paths.storage_root() {
            state.paths.set_storage_root(&wanted);
            let _ = state.paths.ensure();
            use crate::session::ports::EventSink;
            state.session.events().emit(
                crate::session::ports::UiEvent::Notice(crate::events::NoticePayload {
                    level: crate::events::NoticeLevel::Info,
                    message: "New recordings will be saved here. The ones you already have stay \
                              where they are."
                        .into(),
                    persistent: false,
                    meeting_id: None,
                    tag: Some("storageMoved".into()),
                }),
            );
        }
    }
    // Closing to the tray is decided synchronously by the window handler, so it
    // has to be told rather than asked.
    if let Some(close_to_tray) = patch.close_to_tray {
        if let Some(flags) = app.try_state::<crate::UiFlags>() {
            flags
                .close_to_tray
                .store(close_to_tray, std::sync::atomic::Ordering::SeqCst);
        }
    }
    announce_settings(&state, &updated);
    Ok(updated)
}

/// Check a folder before the person commits to it. Returns the free space.
#[tauri::command]
pub async fn validate_storage_location(path: String) -> CmdResult<u64> {
    Ok(paths::validate_storage_dir(std::path::Path::new(&path))?)
}

#[tauri::command]
pub async fn list_input_devices() -> CmdResult<Vec<AudioDevice>> {
    Ok(audio::list_input_devices()?)
}

// ---------------------------------------------------------------------------
// Permissions and onboarding
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn get_permission_status() -> CmdResult<PermissionStatus> {
    Ok(PermissionStatus {
        microphone: audio::microphone_permission().await,
        system_audio: audio::system_audio_permission().await,
        notifications: PermissionState::Unknown,
    })
}

#[tauri::command]
pub async fn request_permission(target: PermissionTarget) -> CmdResult<PermissionState> {
    Ok(match target {
        PermissionTarget::Microphone => audio::request_microphone_permission().await,
        PermissionTarget::SystemAudio => audio::request_system_audio_permission().await,
        PermissionTarget::Notifications => PermissionState::NotApplicable,
    })
}

#[tauri::command]
pub async fn open_privacy_settings(target: PermissionTarget) -> CmdResult<()> {
    Ok(audio::open_privacy_settings(target)?)
}

#[tauri::command]
pub async fn get_onboarding_state(state: State<'_, AppState>) -> CmdResult<OnboardingState> {
    let s = settings::load(&state.db).await?;
    Ok(OnboardingState {
        complete: s.onboarding_complete,
        completed_steps: Vec::new(),
        permissions: PermissionStatus {
            microphone: audio::microphone_permission().await,
            system_audio: audio::system_audio_permission().await,
            notifications: PermissionState::Unknown,
        },
        speech: asr::models::readiness(&state.db, &state.paths)
            .await
            .unwrap_or_default(),
        local_summaries_available: summarize::ollama::OllamaConnector::detect(
            summarize::ollama::DEFAULT_BASE_URL,
        )
        .await,
    })
}

#[tauri::command]
pub async fn complete_onboarding(state: State<'_, AppState>) -> CmdResult<Settings> {
    let updated = settings::apply(
        &state.db,
        &SettingsPatch {
            onboarding_complete: Some(true),
            ..Default::default()
        },
    )
    .await?;
    announce_settings(&state, &updated);
    Ok(updated)
}

/// Settings → Advanced. Technical strings are allowed here and nowhere else.
#[tauri::command]
pub async fn get_system_capabilities(
    app: AppHandle,
    state: State<'_, AppState>,
) -> CmdResult<SystemCapabilities> {
    // The report exists as soon as the worker does; `active` stays empty until
    // something has actually initialised, which is exactly what we want to show
    // (mantra 1: nothing is loaded until it is needed).
    let backend = state.session.speech_backend();
    let fallback_threads = std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1);
    Ok(SystemCapabilities {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        speech_backend: asr::engine::compiled_backends().to_string(),
        speech_backend_active: backend
            .as_ref()
            .map(|b| b.active.clone())
            .unwrap_or_default(),
        gpu_fallback_reason: backend.as_ref().and_then(|b| b.fallback_reason.clone()),
        cpu_threads: backend
            .as_ref()
            .map(|b| b.threads)
            .filter(|t| *t > 0)
            .unwrap_or(fallback_threads),
        total_memory_bytes: asr::models::total_memory_bytes(),
        system_audio_supported: audio::system_audio_supported(),
        tray_supported: cfg!(any(target_os = "macos", target_os = "windows"))
            || cfg!(target_os = "linux"),
        app_version: app.package_info().version.to_string(),
    })
}

// ---------------------------------------------------------------------------
// Window
// ---------------------------------------------------------------------------

/// Bring the window back, from the tray or from a notification click.
#[tauri::command]
pub async fn show_main_window(app: AppHandle) -> CmdResult<()> {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
    Ok(())
}

#[tauri::command]
pub async fn quit_app(app: AppHandle) -> CmdResult<()> {
    // Same path as the tray: close a live recording, park the queue, free the
    // memory, then go.
    crate::quit(&app);
    Ok(())
}

/// Registered with the Tauri builder in `lib.rs`. Every command in this file
/// appears here, and every one has a wrapper in `src/lib/ipc.ts`.
#[macro_export]
macro_rules! echo_command_handler {
    () => {
        tauri::generate_handler![
            // capture
            $crate::commands::start_recording,
            $crate::commands::stop_recording,
            $crate::commands::pause_recording,
            $crate::commands::resume_recording,
            $crate::commands::get_capture_state,
            $crate::commands::add_marker,
            $crate::commands::list_interrupted_meetings,
            $crate::commands::resolve_interrupted_meeting,
            // meetings
            $crate::commands::list_meetings,
            $crate::commands::get_meeting,
            $crate::commands::update_meeting_title,
            $crate::commands::delete_meeting,
            $crate::commands::delete_all_data,
            $crate::commands::get_transcript,
            $crate::commands::search_transcripts,
            $crate::commands::get_markers,
            $crate::commands::get_playback_path,
            $crate::commands::get_storage_report,
            // speakers
            $crate::commands::list_speakers,
            $crate::commands::rename_speaker,
            $crate::commands::merge_speakers,
            $crate::commands::unmerge_speaker,
            $crate::commands::refine_speakers,
            // speech assets
            $crate::commands::list_accuracy_levels,
            $crate::commands::select_accuracy_level,
            $crate::commands::download_speech_assets,
            $crate::commands::cancel_speech_download,
            $crate::commands::get_speech_readiness,
            $crate::commands::prewarm_speech,
            $crate::commands::release_speech,
            $crate::commands::list_speech_assets,
            $crate::commands::remove_speech_asset,
            // recaps
            $crate::commands::list_summary_providers,
            $crate::commands::test_summary_provider,
            $crate::commands::save_summary_provider,
            $crate::commands::set_provider_key,
            $crate::commands::has_provider_key,
            $crate::commands::delete_provider_key,
            $crate::commands::generate_summary,
            $crate::commands::list_summaries,
            $crate::commands::get_summary,
            $crate::commands::list_action_items,
            $crate::commands::update_action_item,
            // recap styles
            $crate::commands::list_templates,
            $crate::commands::save_template,
            $crate::commands::delete_template,
            // jobs
            $crate::commands::list_jobs,
            $crate::commands::cancel_job,
            $crate::commands::retry_job,
            // detection
            $crate::commands::get_detection_status,
            $crate::commands::set_detection_enabled,
            $crate::commands::snooze_detection,
            // export
            $crate::commands::export_meeting,
            $crate::commands::export_diagnostics,
            // settings
            $crate::commands::get_settings,
            $crate::commands::update_settings,
            $crate::commands::validate_storage_location,
            $crate::commands::list_input_devices,
            // permissions and onboarding
            $crate::commands::get_permission_status,
            $crate::commands::request_permission,
            $crate::commands::open_privacy_settings,
            $crate::commands::get_onboarding_state,
            $crate::commands::complete_onboarding,
            $crate::commands::get_system_capabilities,
            // window
            $crate::commands::show_main_window,
            $crate::commands::quit_app,
        ]
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::DbError;

    #[test]
    fn ids_that_are_not_uuids_are_rejected_before_any_query() {
        assert!(check_id(&repo::new_id()).is_ok());
        for bad in [
            "",
            "1",
            "'; DROP TABLE meetings; --",
            "00000000-0000-4000-8000-00000000000",
            "not-a-uuid-at-all-but-36-chars-long!",
        ] {
            assert!(check_id(bad).is_err(), "{bad:?} should be rejected");
        }
    }

    #[test]
    fn text_input_is_trimmed_and_length_checked() {
        assert_eq!(trim_limited("  hello  ", 10, "A name").unwrap(), "hello");
        assert!(trim_limited("   ", 10, "A name").is_err());
        assert!(trim_limited(&"x".repeat(11), 10, "A name").is_err());
    }

    #[test]
    fn errors_reaching_the_person_carry_no_jargon() {
        let cases: Vec<UiError> = vec![
            audio::AudioError::PermissionDenied.into(),
            audio::AudioError::NoInputDevice.into(),
            asr::AsrError::NotInstalled.into(),
            asr::AsrError::IntegrityCheckFailed.into(),
            diarize::DiarizeError::NotInstalled.into(),
            summarize::SummarizeError::MissingCredential.into(),
            summarize::SummarizeError::NoTranscript.into(),
            export::ExportError::Empty.into(),
            session::SessionError::NoAudioSources("x".into()).into(),
            DbError::NotFound("meeting".into()).into(),
        ];
        let banned = [
            "whisper",
            "onnx",
            "vad",
            "diariz",
            "connector",
            "token",
            "model",
            "sqlite",
            "sqlx",
            "keyring",
            "ffi",
            "pipewire",
            "screencapturekit",
        ];
        for err in cases {
            let lower = err.message.to_lowercase();
            for word in banned {
                assert!(!lower.contains(word), "{:?} leaks {word:?}", err.message);
            }
            assert!(
                err.message.ends_with('.') || err.message.ends_with('!'),
                "{:?} should read as a sentence",
                err.message
            );
        }
    }

    #[test]
    fn cancellation_is_not_reported_as_a_failure() {
        let e: UiError = asr::AsrError::Cancelled.into();
        assert_eq!(e.kind, UiErrorKind::Cancelled);
        let e: UiError = summarize::SummarizeError::Cancelled.into();
        assert_eq!(e.kind, UiErrorKind::Cancelled);
    }
}
