//! Recaps and action items.
//!
//! IMPLEMENTED-BY: summarize agent (M5).
//!
//! One trait, two implementations in v1: [`ollama`] for "on this computer" and
//! [`gemini`] for Google AI Studio. The trait is extensible, but v1 compiles
//! exactly these two (DESIGN §1 non-goals, review finding 33).
//!
//! How a recap is produced (DESIGN §3 Connectors):
//! * map-reduce over transcript chunks sized to [`Caps::context_chars`]
//! * action items come back as strict JSON validated locally, with **one**
//!   repair retry, then a graceful "no action items found"
//! * markdown is sanitized before it ever reaches the webview: no raw HTML, no
//!   remote images, no scripts (review finding 38)
//! * every recap records provider, model and transcript revision, so an old
//!   recap stays explainable
//!
//! Privacy is a choice the person makes, not a footnote. The Gemini screen says
//! plainly what leaves the machine. A local address other than loopback needs an
//! explicit "this leaves your machine" confirmation.

pub mod gemini;
pub mod ollama;
pub mod templates;

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

use futures::future::BoxFuture;
use futures::stream::{BoxStream, StreamExt};

use crate::db::{repo, Db, DbError};
use crate::secrets;
use crate::settings;
use crate::types::{
    ActionItem, Caps, Provider, ProviderConfig, ProviderInfo, ProviderTestResult, Summary,
    SummaryLanguage, SummaryReq, TranscriptQuery,
};

#[derive(Debug, thiserror::Error)]
pub enum SummarizeError {
    #[error("not implemented yet")]
    NotImplemented,
    /// Nothing to summarise.
    #[error("this meeting has no transcript yet")]
    NoTranscript,
    /// The local backend is not running, or the address is wrong.
    #[error("could not reach the place that writes recaps: {0}")]
    Unreachable(String),
    #[error("no key is saved for this service")]
    MissingCredential,
    #[error("the service refused the request: {0}")]
    Rejected(String),
    /// The model saved in Settings is not one this service answers for — it was
    /// renamed, retired, or mistyped. The person fixes it by choosing another
    /// one, so the whole sentence is written for them: this string can reach the
    /// screen unchanged.
    #[error("Echo can't write recaps with “{model}” any more. Open Settings and choose a different one.")]
    ModelNotFound { model: String },
    #[error("the service took too long")]
    Timeout,
    #[error("cancelled")]
    Cancelled,
    /// Strict JSON came back malformed twice.
    #[error("the reply could not be read as action items")]
    BadJson,
    #[error("recap failed: {0}")]
    Failed(String),
}

fn db_err(e: DbError) -> SummarizeError {
    SummarizeError::Failed(e.to_string())
}

/// Cooperative cancel. Set the flag and every in-flight request stops at its
/// next checkpoint.
#[derive(Debug, Clone, Default)]
pub struct CancelFlag(Arc<AtomicBool>);

impl CancelFlag {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// One request to a backend.
///
/// Note what is *not* here: a model. Which model writes the recap is part of the
/// backend's saved configuration, resolved once when the connector is built, so
/// a request cannot quietly ask for a different one (see [`Connector::model`]).
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    /// Instructions plus transcript. Already sized to the backend's context.
    pub prompt: String,
    /// Separate system instruction, for backends that take one.
    pub system: Option<String>,
    /// When set, ask for strict JSON matching this schema. Only honoured when
    /// [`Caps::json_mode`] is true; otherwise the prompt carries the schema and
    /// the reply is validated locally either way.
    pub json_schema: Option<serde_json::Value>,
    pub temperature: Option<f32>,
    pub timeout: Duration,
    pub cancel: CancelFlag,
}

impl Default for GenerateRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            system: None,
            json_schema: None,
            temperature: Some(0.2),
            timeout: Duration::from_secs(180),
            cancel: CancelFlag::new(),
        }
    }
}

/// What a backend emits while it works.
#[derive(Debug, Clone, PartialEq)]
pub enum GenerateEvent {
    /// More text. Only sent when [`Caps::streaming`] is true.
    Text(String),
    /// The whole reply, always sent last.
    Done { text: String },
}

/// A place that can write a recap.
///
/// Dyn-compatible on purpose, so the session layer can hold
/// `Box<dyn Connector>` chosen at runtime from settings. That is why the async
/// methods return boxed futures rather than using `async fn`.
pub trait Connector: Send + Sync {
    /// Which entry in the Summaries settings screen this is.
    fn provider(&self) -> Provider;

    /// What this backend can do. Drives chunk sizing, streaming and whether the
    /// UI offers a JSON-strict path.
    fn capabilities(&self) -> Caps;

    /// The model this connector will actually use, resolved from the saved
    /// configuration when it was built (falling back to the backend's default).
    /// `None` only when nothing is configured and the backend has no sensible
    /// default of its own — a local server, where the pulls are the person's own
    /// deliberate choices.
    ///
    /// The recap records this, so an old recap stays explainable even after the
    /// setting changes.
    fn model(&self) -> Option<String>;

    /// Choices the person can pick from. Empty when the backend cannot list
    /// them ([`Caps::can_list_models`] is false).
    fn list_models<'a>(&'a self) -> BoxFuture<'a, Result<Vec<String>, SummarizeError>>;

    /// Cheap round-trip behind the "check this works" button. Never sends
    /// transcript text.
    fn check<'a>(&'a self) -> BoxFuture<'a, Result<ProviderTestResult, SummarizeError>>;

    /// Run one request. The stream ends with [`GenerateEvent::Done`], or with an
    /// error. Dropping the stream cancels the request; so does the
    /// [`CancelFlag`] on the request.
    fn generate<'a>(
        &'a self,
        req: GenerateRequest,
    ) -> BoxStream<'a, Result<GenerateEvent, SummarizeError>>;
}

/// Small `reqwest` helpers shared by [`ollama`] and [`gemini`], kept in one
/// place so "retry once on transient" means the same thing in both.
pub(crate) mod http_util {
    use super::SummarizeError;

    /// A hiccup worth retrying once: the connection never opened, or the
    /// request timed out before anything was sent. Anything else (a bad
    /// status, a parse failure) is not retried — retrying it would just get
    /// the same answer.
    pub fn is_transient(err: &reqwest::Error) -> bool {
        err.is_connect() || err.is_timeout()
    }

    /// The taxonomy the UI reacts to, from a transport-level failure.
    pub fn classify_reqwest_error(err: &reqwest::Error) -> SummarizeError {
        if err.is_connect() {
            SummarizeError::Unreachable(err.to_string())
        } else if err.is_timeout() {
            SummarizeError::Timeout
        } else {
            SummarizeError::Failed(err.to_string())
        }
    }

    /// Run `attempt` once; if it fails with something transient, run it again
    /// exactly once (DESIGN §3 Connector trait: "timeout, retry, cancel").
    pub async fn with_one_retry<T, Fut, F>(mut attempt: F) -> Result<T, SummarizeError>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T, reqwest::Error>>,
    {
        match attempt().await {
            Ok(v) => Ok(v),
            Err(e) if is_transient(&e) => attempt().await.map_err(|e2| classify_reqwest_error(&e2)),
            Err(e) => Err(classify_reqwest_error(&e)),
        }
    }

    /// Truncate an error body before it goes into an error string. Provider
    /// error bodies are small JSON blobs, never transcript text, but there is
    /// no reason to let a misbehaving server hand us megabytes to log.
    pub fn truncate(s: &str, max_chars: usize) -> String {
        if s.chars().count() <= max_chars {
            s.to_string()
        } else {
            let mut out: String = s.chars().take(max_chars).collect();
            out.push('…');
            out
        }
    }
}

/// Build the connector for a provider from settings plus the keychain.
///
/// Returns [`SummarizeError::MissingCredential`] when a backend needs a key
/// that has not been saved.
pub async fn connector_for(
    db: &Db,
    provider: Provider,
) -> Result<Box<dyn Connector>, SummarizeError> {
    match provider {
        Provider::OnThisComputer => {
            let base_url = repo::get_setting(db, settings::keys::OLLAMA_BASE_URL)
                .await
                .map_err(db_err)?
                .filter(|v| !v.trim().is_empty())
                .unwrap_or_else(|| ollama::DEFAULT_BASE_URL.to_string());
            let model = repo::get_setting(db, settings::keys::OLLAMA_MODEL)
                .await
                .map_err(db_err)?
                .filter(|v| !v.trim().is_empty());
            Ok(Box::new(ollama::OllamaConnector::new(base_url, model)))
        }
        Provider::Gemini => {
            let api_key = secrets::get(secrets::accounts::GEMINI_API_KEY)
                .await
                .map_err(|e| match e {
                    secrets::SecretError::Absent(_) => SummarizeError::MissingCredential,
                    other => SummarizeError::Failed(other.to_string()),
                })?;
            let model = repo::get_setting(db, settings::keys::GEMINI_MODEL)
                .await
                .map_err(db_err)?
                .filter(|v| !v.trim().is_empty());
            Ok(Box::new(gemini::GeminiConnector::new(api_key, model)))
        }
    }
}

/// One row per backend for the Summaries settings screen, including whether a
/// local one was found without any setup.
pub async fn list_providers(db: &Db) -> Result<Vec<ProviderInfo>, SummarizeError> {
    let mut out = Vec::with_capacity(2);

    let base_url = repo::get_setting(db, settings::keys::OLLAMA_BASE_URL)
        .await
        .map_err(db_err)?
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| ollama::DEFAULT_BASE_URL.to_string());
    let ollama_model = repo::get_setting(db, settings::keys::OLLAMA_MODEL)
        .await
        .map_err(db_err)?
        .filter(|v| !v.trim().is_empty());
    let ollama_conn = ollama::OllamaConnector::new(base_url.clone(), ollama_model.clone());
    let ollama_available = ollama::OllamaConnector::detect(&base_url).await;
    out.push(ProviderInfo {
        provider: Provider::OnThisComputer,
        config: ProviderConfig {
            provider: Provider::OnThisComputer,
            base_url: Some(base_url),
            model: ollama_model,
            has_key: true,
            enabled: true,
            // Reporting state, not a request: the flag only matters on the way in.
            leaves_machine_acknowledged: false,
        },
        caps: ollama_conn.capabilities(),
        available: ollama_available,
    });

    // Report the model that would actually write a recap right now, not the raw
    // row: with nothing saved that is the default, and the Settings screen should
    // show the person what is in use rather than an empty box.
    let gemini_model = Some(
        repo::get_setting(db, settings::keys::GEMINI_MODEL)
            .await
            .map_err(db_err)?
            .filter(|v| !v.trim().is_empty())
            .map(|v| v.trim().to_string())
            .unwrap_or_else(|| gemini::DEFAULT_MODEL.to_string()),
    );
    let has_key = secrets::has(secrets::accounts::GEMINI_API_KEY)
        .await
        .unwrap_or(false);
    out.push(ProviderInfo {
        provider: Provider::Gemini,
        config: ProviderConfig {
            provider: Provider::Gemini,
            base_url: None,
            model: gemini_model,
            has_key,
            enabled: true,
            leaves_machine_acknowledged: false,
        },
        caps: Caps {
            json_mode: true,
            streaming: true,
            context_chars: gemini::DEFAULT_CONTEXT_CHARS,
            can_list_models: true,
            leaves_machine: true,
        },
        available: has_key,
    });

    Ok(out)
}

/// Save a backend's non-secret configuration. The key goes to the keychain
/// separately and never travels in this struct.
pub async fn save_config(db: &Db, config: &ProviderConfig) -> Result<(), SummarizeError> {
    match config.provider {
        Provider::OnThisComputer => {
            if let Some(url) = &config.base_url {
                let trimmed = url.trim();
                if trimmed.is_empty() {
                    return Err(SummarizeError::Failed("that address can't be empty".into()));
                }
                repo::set_setting(db, settings::keys::OLLAMA_BASE_URL, trimmed)
                    .await
                    .map_err(db_err)?;
            }
            if let Some(model) = &config.model {
                repo::set_setting(db, settings::keys::OLLAMA_MODEL, model.trim())
                    .await
                    .map_err(db_err)?;
            }
        }
        Provider::Gemini => {
            if let Some(model) = &config.model {
                repo::set_setting(db, settings::keys::GEMINI_MODEL, model.trim())
                    .await
                    .map_err(db_err)?;
            }
            // The key itself never travels through here — `set_provider_key`
            // writes straight to the keychain and never touches this table.
        }
    }
    Ok(())
}

/// Behind the "check this works" button.
pub async fn test_provider(
    db: &Db,
    provider: Provider,
) -> Result<ProviderTestResult, SummarizeError> {
    let connector = connector_for(db, provider).await?;
    connector.check().await
}

/// One call to a connector that only cares about the final text, used by both
/// the map and the reduce step.
async fn run_generate(
    connector: &dyn Connector,
    prompt: String,
    cancel: &CancelFlag,
) -> Result<String, SummarizeError> {
    if cancel.is_cancelled() {
        return Err(SummarizeError::Cancelled);
    }
    let req = GenerateRequest {
        prompt,
        cancel: cancel.clone(),
        ..Default::default()
    };
    let mut stream = connector.generate(req);
    let mut last: Option<String> = None;
    while let Some(event) = stream.next().await {
        match event? {
            GenerateEvent::Text(_) => {}
            GenerateEvent::Done { text } => last = Some(text),
        }
    }
    last.ok_or_else(|| SummarizeError::Failed("the reply ended without any text".into()))
}

/// Every final segment for this meeting, one line per segment, with speakers
/// resolved through any merge (DESIGN §3: "transcript (final segments w/
/// speaker names)").
async fn build_transcript_text(db: &Db, meeting_id: &str) -> Result<String, SummarizeError> {
    let segments = repo::get_segments(
        db,
        &TranscriptQuery {
            meeting_id: meeting_id.to_string(),
            include_partial: Some(false),
            ..Default::default()
        },
    )
    .await
    .map_err(db_err)?;

    let mut names: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut out = String::new();
    for seg in &segments {
        if seg.text.trim().is_empty() {
            continue;
        }
        let speaker_name = match &seg.speaker_id {
            Some(id) => {
                if !names.contains_key(id) {
                    let resolved = repo::resolve_speaker(db, id).await.map_err(db_err)?;
                    let name = resolved
                        .map(|s| s.display_name)
                        .unwrap_or_else(|| "Unknown speaker".to_string());
                    names.insert(id.clone(), name);
                }
                names
                    .get(id)
                    .cloned()
                    .unwrap_or_else(|| "Unknown speaker".to_string())
            }
            None => "Unknown speaker".to_string(),
        };
        out.push_str(&speaker_name);
        out.push_str(": ");
        out.push_str(seg.text.trim());
        out.push('\n');
    }
    Ok(out)
}

/// The meeting's dominant language: the one with the most speaking time in
/// final segments, falling back to whatever the meeting row already says.
async fn dominant_language(
    db: &Db,
    meeting_id: &str,
    fallback: Option<&str>,
) -> Result<Option<String>, SummarizeError> {
    let histogram = repo::language_histogram(db, meeting_id)
        .await
        .map_err(db_err)?;
    if let Some((lang, _ms)) = histogram.into_iter().next() {
        return Ok(Some(lang));
    }
    Ok(fallback.map(String::from))
}

/// Write a recap for one meeting.
///
/// Map-reduce over the transcript, then a strict-JSON pass for action items,
/// then sanitize, then store the recap with its provenance. Cancellable, and a
/// recording preempts it.
pub async fn summarize_meeting(
    db: &Db,
    req: &SummaryReq,
    cancel: CancelFlag,
) -> Result<Summary, SummarizeError> {
    if cancel.is_cancelled() {
        return Err(SummarizeError::Cancelled);
    }

    let meeting = repo::get_meeting(db, &req.meeting_id)
        .await
        .map_err(db_err)?
        .ok_or(SummarizeError::NoTranscript)?;

    let revision = repo::transcript_revision(db, &req.meeting_id)
        .await
        .map_err(db_err)?;
    if revision == 0 {
        return Err(SummarizeError::NoTranscript);
    }

    let template = templates::resolve_template(db, req.template_id.as_deref()).await?;

    // Idempotent: the same transcript revision, same style, already has a
    // recap — hand that back rather than spending another round-trip.
    if !req.force.unwrap_or(false) {
        if let Some(existing) = repo::latest_summary(db, &req.meeting_id)
            .await
            .map_err(db_err)?
        {
            if existing.transcript_revision == revision
                && existing.template_id.as_deref() == Some(template.id.as_str())
                && req.provider.map(|p| p == existing.provider).unwrap_or(true)
            {
                return Ok(existing);
            }
        }
    }

    let app_settings = settings::load(db).await.map_err(db_err)?;
    let provider = req.provider.unwrap_or(app_settings.summary_provider);
    // Provider and model are both resolved here, at run time, from what is saved
    // for that backend — the same path for the recap that runs when a meeting
    // ends and for "write it again" from the meeting screen. `req.model` is
    // deliberately ignored: choosing a model is a setting for the backend, not a
    // decision to make once per recap. (The field stays on the wire type so an
    // older queued job still deserializes.)
    let connector = connector_for(db, provider).await?;
    let model = connector.model();
    let context_chars = connector.capabilities().context_chars;

    let transcript_text = build_transcript_text(db, &req.meeting_id).await?;
    if transcript_text.trim().is_empty() {
        return Err(SummarizeError::NoTranscript);
    }

    let speakers: Vec<_> = repo::list_speakers(db, &req.meeting_id)
        .await
        .map_err(db_err)?
        .into_iter()
        .filter(|s| s.alias_of.is_none())
        .collect();

    let output_language = req
        .language
        .clone()
        .unwrap_or_else(|| app_settings.summary_language.clone());
    let meeting_language =
        dominant_language(db, &req.meeting_id, meeting.language.as_deref()).await?;

    let mut ctx = templates::RenderContext {
        meeting_title: meeting.title.clone(),
        started_at: meeting.started_at.clone(),
        duration_ms: meeting.duration_ms,
        meeting_language,
        output_language,
        speakers,
        transcript_chunk: String::new(),
        chunk_index: 0,
        chunk_count: 1,
    };

    // Leave headroom for the instructions themselves, not just the transcript
    // text, so a chunk that "fits" does not actually overflow once the
    // template's own words are added.
    const PROMPT_OVERHEAD_CHARS: u32 = 1_500;
    const MIN_CHUNK_CHARS: u32 = 1_000;
    let budget = context_chars
        .saturating_sub(PROMPT_OVERHEAD_CHARS)
        .max(MIN_CHUNK_CHARS);
    let chunks = chunk_transcript(&transcript_text, budget);
    if chunks.is_empty() {
        return Err(SummarizeError::NoTranscript);
    }

    let content_md = if chunks.len() == 1 {
        ctx.transcript_chunk = chunks[0].clone();
        ctx.chunk_count = 1;
        let prompt = templates::render(&template, &ctx);
        run_generate(connector.as_ref(), prompt, &cancel).await?
    } else {
        let mut notes = Vec::with_capacity(chunks.len());
        for (i, chunk) in chunks.iter().enumerate() {
            if cancel.is_cancelled() {
                return Err(SummarizeError::Cancelled);
            }
            ctx.transcript_chunk = chunk.clone();
            ctx.chunk_index = i as u32;
            ctx.chunk_count = chunks.len() as u32;
            let prompt = templates::render(&template, &ctx);
            let note = run_generate(connector.as_ref(), prompt, &cancel).await?;
            notes.push(note);
        }
        if cancel.is_cancelled() {
            return Err(SummarizeError::Cancelled);
        }
        let reduce_prompt = templates::render_reduce(&template, &notes, &ctx);
        run_generate(connector.as_ref(), reduce_prompt, &cancel).await?
    };

    let sanitized = sanitize_markdown(&content_md);
    let language_code = match &ctx.output_language {
        SummaryLanguage::English => Some("en".to_string()),
        SummaryLanguage::Fixed(l) => Some(l.clone()),
        SummaryLanguage::SameAsMeeting => ctx.meeting_language.clone(),
    };
    let snapshot = templates::prompt_for(&template);

    let summary = repo::insert_summary(
        db,
        &req.meeting_id,
        Some(&template.id),
        Some(&snapshot),
        provider,
        model.as_deref(),
        language_code.as_deref(),
        revision,
        &sanitized,
    )
    .await
    .map_err(db_err)?;

    if !cancel.is_cancelled() {
        // Action items are a bonus on top of a recap that already succeeded —
        // never fail the whole recap because the follow-up JSON pass did not
        // pan out (see the module doc comment: "a graceful 'no action items
        // found'").
        if let Err(e) = extract_action_items(db, &req.meeting_id, &summary.id, cancel.clone()).await
        {
            tracing::warn!("action items not written for summary {}: {e}", summary.id);
        }
    }

    Ok(summary)
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ActionItemsResponse {
    items: Vec<ActionItemRaw>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ActionItemRaw {
    description: String,
    #[serde(default)]
    owner: Option<String>,
    #[serde(rename = "dueHint", default)]
    due_hint: Option<String>,
}

/// Models like to wrap JSON in a code fence, or add a sentence before or
/// after it. Take the outermost `{...}` rather than demanding a perfectly
/// clean reply — the schema check right after this is what actually enforces
/// strictness.
fn extract_json_object(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end < start {
        return None;
    }
    Some(&text[start..=end])
}

/// Parse and validate one reply against the action-item schema. Deliberately
/// strict (`deny_unknown_fields`) so a reply that almost matches still gets
/// the repair retry rather than silently keeping made-up fields.
fn parse_action_items(text: &str) -> Result<Vec<ActionItemRaw>, SummarizeError> {
    let candidate = extract_json_object(text).ok_or(SummarizeError::BadJson)?;
    let parsed: ActionItemsResponse =
        serde_json::from_str(candidate).map_err(|_| SummarizeError::BadJson)?;
    Ok(parsed
        .items
        .into_iter()
        .filter(|item| !item.description.trim().is_empty())
        .collect())
}

/// One attempt: ask the connector, then validate the reply.
async fn run_action_item_pass(
    connector: &dyn Connector,
    prompt: &str,
    schema: &serde_json::Value,
    json_mode: bool,
    cancel: &CancelFlag,
) -> Result<Vec<ActionItemRaw>, SummarizeError> {
    if cancel.is_cancelled() {
        return Err(SummarizeError::Cancelled);
    }
    let req = GenerateRequest {
        prompt: prompt.to_string(),
        json_schema: json_mode.then(|| schema.clone()),
        cancel: cancel.clone(),
        ..Default::default()
    };
    let mut stream = connector.generate(req);
    let mut text: Option<String> = None;
    while let Some(event) = stream.next().await {
        if let GenerateEvent::Done { text: t } = event? {
            text = Some(t);
        }
    }
    let text = text.ok_or(SummarizeError::BadJson)?;
    parse_action_items(&text)
}

/// Extract action items from a finished recap plus the transcript. Strict JSON,
/// validated locally, one repair retry.
pub async fn extract_action_items(
    db: &Db,
    meeting_id: &str,
    summary_id: &str,
    cancel: CancelFlag,
) -> Result<Vec<ActionItem>, SummarizeError> {
    let summary = repo::get_summary(db, summary_id)
        .await
        .map_err(db_err)?
        .ok_or_else(|| SummarizeError::Failed("that recap is gone".into()))?;

    // The recap's provider, but today's model: `summary.model` says which model
    // wrote that recap, and it may since have been changed or retired. Asking a
    // model that no longer exists for the task list would fail for a reason that
    // has nothing to do with the task list.
    let connector = connector_for(db, summary.provider).await?;
    let json_mode = connector.capabilities().json_mode;

    let app_settings = settings::load(db).await.map_err(db_err)?;
    let ctx = templates::RenderContext {
        output_language: app_settings.summary_language.clone(),
        ..Default::default()
    };
    let prompt = templates::render_action_items(&summary.content_md, &ctx);
    let schema = templates::action_item_schema();

    let raw = match run_action_item_pass(connector.as_ref(), &prompt, &schema, json_mode, &cancel)
        .await
    {
        Ok(items) => items,
        Err(SummarizeError::Cancelled) => return Err(SummarizeError::Cancelled),
        Err(_first_err) => {
            let repair_prompt = format!(
                "{prompt}\n\nYour last reply could not be read as that JSON. Reply again with \
                 ONLY the JSON object described above — no markdown fences, no commentary."
            );
            match run_action_item_pass(
                connector.as_ref(),
                &repair_prompt,
                &schema,
                json_mode,
                &cancel,
            )
            .await
            {
                Ok(items) => items,
                Err(SummarizeError::Cancelled) => return Err(SummarizeError::Cancelled),
                // Second failure: graceful "no action items found" rather than
                // losing an already-written recap over this.
                Err(_second_err) => Vec::new(),
            }
        }
    };

    let items: Vec<ActionItem> = raw
        .into_iter()
        .map(|item| ActionItem {
            id: String::new(),
            meeting_id: meeting_id.to_string(),
            summary_id: Some(summary_id.to_string()),
            description: item.description,
            owner: item.owner,
            due_hint: item.due_hint,
            done: false,
            external_url: None,
        })
        .collect();

    repo::replace_action_items(db, meeting_id, Some(summary_id), &items)
        .await
        .map_err(db_err)
}

/// Split a transcript into pieces that fit `context_chars`, breaking on line
/// boundaries (speaker turns) rather than mid-sentence. A single line longer
/// than the budget is hard-split so it can never block chunking entirely.
pub fn chunk_transcript(text: &str, context_chars: u32) -> Vec<String> {
    let limit = context_chars.max(1) as usize;
    if text.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut current = String::new();

    for line in text.split_inclusive('\n') {
        if line.chars().count() > limit {
            if !current.is_empty() {
                chunks.push(std::mem::take(&mut current));
            }
            // Hard-split the oversized line itself; the remainder becomes the
            // start of the next chunk rather than being dropped.
            let mut piece = String::new();
            for ch in line.chars() {
                if piece.chars().count() >= limit {
                    chunks.push(std::mem::take(&mut piece));
                }
                piece.push(ch);
            }
            current = piece;
            continue;
        }

        if !current.is_empty() && current.chars().count() + line.chars().count() > limit {
            chunks.push(std::mem::take(&mut current));
        }
        current.push_str(line);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// Strip anything that could execute or phone home before the webview renders
/// the recap: raw HTML, scripts, remote images, `javascript:` links.
///
/// Two passes, because a recap is Markdown, not HTML, and the two need
/// different tools:
///
/// 1. **`pulldown-cmark`** walks the actual Markdown grammar so we can find
///    the constructs that only make sense at the Markdown level — an `![]()`
///    image, a `[]()` link — and remove/neutralize exactly those byte ranges
///    in the source. A recap never legitimately embeds an image (a remote
///    `src` is exactly the kind of thing that would try to phone home the
///    moment the webview painted it, DESIGN §3 review finding 38), so every
///    image is dropped outright regardless of scheme; a link is dropped only
///    when its destination is not `http(s)`/`mailto`/relative — the common
///    case (a link to somewhere sensible) survives untouched.
/// 2. **`ammonia`** then runs over what is left as a final HTML-level lock:
///    genuine raw HTML embedded in the Markdown (`<script>`, an `onclick`
///    attribute, …) is parsed and stripped rather than passed through, and
///    any stray `<`/`&` that was never markup comes back HTML-entity-escaped
///    (`&lt;`, `&amp;`) — which a CommonMark renderer decodes right back to
///    the original character when it renders the now-safe Markdown, so plain
///    prose is unaffected either way.
///
/// The webview's CSP blocks remote loads too; this is the second lock, not
/// the only one.
pub fn sanitize_markdown(md: &str) -> String {
    use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};

    let mut cuts: Vec<(usize, usize)> = Vec::new();
    let mut image_starts: Vec<usize> = Vec::new();
    let mut link_starts: Vec<(usize, bool)> = Vec::new(); // (start offset, unsafe destination)

    for (event, range) in Parser::new_ext(md, Options::empty()).into_offset_iter() {
        match event {
            Event::Html(_) | Event::InlineHtml(_) => cuts.push((range.start, range.end)),
            Event::Start(Tag::Image { .. }) => image_starts.push(range.start),
            Event::End(TagEnd::Image) => {
                if let Some(start) = image_starts.pop() {
                    cuts.push((start, range.end));
                }
            }
            Event::Start(Tag::Link { dest_url, .. }) => {
                link_starts.push((range.start, !is_safe_link_scheme(&dest_url)));
            }
            Event::End(TagEnd::Link) => {
                if let Some((start, unsafe_dest)) = link_starts.pop() {
                    if unsafe_dest {
                        cuts.push((start, range.end));
                    }
                }
            }
            _ => {}
        }
    }

    // Keep only the outermost of any nested/overlapping ranges (an image's
    // alt text could in principle contain its own inline-HTML event) so the
    // removal pass below never has to reconcile a byte offset that a nested
    // cut already invalidated.
    cuts.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
    let mut kept: Vec<(usize, usize)> = Vec::new();
    for (start, end) in cuts {
        if let Some(&(_, last_end)) = kept.last() {
            if start < last_end {
                continue;
            }
        }
        kept.push((start, end));
    }

    let mut out = md.to_string();
    for (start, end) in kept.into_iter().rev() {
        if start <= end
            && end <= out.len()
            && out.is_char_boundary(start)
            && out.is_char_boundary(end)
        {
            out.replace_range(start..end, "");
        }
    }

    ammonia::Builder::new()
        .rm_tags(&["img"])
        .clean(&out)
        .to_string()
}

/// A link destination that is safe to leave in a recap: an ordinary web
/// address, a mail link, or a same-document/relative reference. Everything
/// else (`javascript:`, `data:`, `vbscript:`, `file:`, …) is not a place a
/// recap should ever point.
fn is_safe_link_scheme(url: &str) -> bool {
    let trimmed = url.trim();
    let lower = trimmed.to_ascii_lowercase();
    lower.starts_with("https://")
        || lower.starts_with("http://")
        || lower.starts_with("mailto:")
        || !lower.contains(':') // relative path or "#section"; no scheme at all
}

/// Is this address on the loopback interface?
///
/// A local backend is loopback-only by default. Anything else means transcript
/// text leaves the machine, and the person has to agree to that first.
///
/// The host is **parsed as an address**, never matched as a prefix: a name like
/// `127.0.0.1.evil.com` or `localhost.evil.com` resolves wherever its owner
/// points it, so treating it as "on this computer" would ship whole transcripts
/// off the machine with no warning. Only the literal name `localhost` and an
/// address the operating system would call loopback count. Anything this
/// function cannot make sense of is treated as *not* loopback, so an odd address
/// asks for confirmation rather than skipping it.
pub fn is_loopback_url(url: &str) -> bool {
    let rest = match url.split_once("://") {
        Some((_, rest)) => rest,
        None => url,
    };
    // The authority ends at the first path, query or fragment character.
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    // Anything before the last '@' is user information, not the host.
    let authority = match authority.rsplit_once('@') {
        Some((_, host)) => host,
        None => authority,
    };
    let host = if let Some(without_bracket) = authority.strip_prefix('[') {
        // Bracketed IPv6: everything up to the closing bracket.
        match without_bracket.split_once(']') {
            // Only a port may follow the closing bracket; `[::1].evil.com` is
            // not an address, it is a name dressed up as one.
            Some((inside, tail)) if tail.is_empty() || tail.starts_with(':') => inside,
            _ => return false,
        }
    } else {
        // A trailing ":port" is not part of the host. A bare IPv6 address here
        // has several colons, so this yields something unparseable — which is
        // the safe answer.
        authority.split(':').next().unwrap_or("")
    };
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    host.parse::<std::net::IpAddr>()
        .map(|ip| ip.is_loopback())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loopback_detection_covers_the_forms_people_type() {
        for url in [
            "http://localhost:11434",
            "http://127.0.0.1:11434",
            "http://127.0.0.1",
            "http://[::1]:11434",
            "http://127.5.5.5:8080/api",
            "localhost:11434",
            "http://[0:0:0:0:0:0:0:1]:11434",
            "http://LocalHost:11434",
        ] {
            assert!(is_loopback_url(url), "{url} should be loopback");
        }
        for url in [
            "http://192.168.1.20:11434",
            "https://ollama.example.com",
            "http://10.0.0.1:11434/api/generate",
            "http://localhost.evil.com:11434",
            // A name that merely *starts* like a loopback address is somebody
            // else's server (review: consent-gate bypass).
            "http://127.0.0.1.evil.com:11434",
            "http://127.evil.com",
            "http://127.0.0.1.evil.com",
            "http://localhost-evil.com",
            "http://[::1].evil.com",
            // Loopback in the user info, not the host.
            "http://127.0.0.1@evil.com/api",
        ] {
            assert!(!is_loopback_url(url), "{url} should not be loopback");
        }
    }

    #[test]
    fn cancel_flag_is_shared_between_clones() {
        let a = CancelFlag::new();
        let b = a.clone();
        assert!(!a.is_cancelled());
        b.cancel();
        assert!(a.is_cancelled());
    }

    #[test]
    fn default_request_is_conservative() {
        let r = GenerateRequest::default();
        assert_eq!(r.timeout, Duration::from_secs(180));
        assert_eq!(r.temperature, Some(0.2));
        assert!(!r.cancel.is_cancelled());
    }

    // -- chunk_transcript ----------------------------------------------

    #[test]
    fn chunking_returns_nothing_for_empty_input() {
        assert!(chunk_transcript("", 1_000).is_empty());
    }

    #[test]
    fn a_transcript_within_the_budget_is_one_chunk() {
        let text = "You: hello\nSpeaker 1: hi there\n";
        let chunks = chunk_transcript(text, 1_000);
        assert_eq!(chunks, vec![text.to_string()]);
    }

    #[test]
    fn chunking_packs_lines_without_exceeding_the_budget() {
        // Ten lines of exactly 10 chars each (including the newline).
        let line = "0123456789";
        let text: String = format!("{line}\n").repeat(10);
        let chunks = chunk_transcript(&text, 35); // room for 3 lines per chunk
        assert!(
            chunks.len() >= 4,
            "expected several chunks, got {}",
            chunks.len()
        );
        for c in &chunks {
            assert!(c.chars().count() <= 35, "chunk exceeded budget: {c:?}");
        }
        // No text lost: every line reassembles to the original.
        assert_eq!(chunks.concat(), text);
    }

    #[test]
    fn chunking_never_produces_a_chunk_over_the_limit_even_with_ragged_lines() {
        let text = "short\na bit longer line here\nx\nanother medium length line\n";
        for limit in [5u32, 10, 20, 40, 1000] {
            let chunks = chunk_transcript(text, limit);
            assert_eq!(chunks.concat(), text, "limit {limit} lost text");
            for c in &chunks {
                assert!(
                    c.chars().count() <= limit.max(1) as usize,
                    "limit {limit}: chunk {c:?} exceeded it"
                );
            }
        }
    }

    #[test]
    fn a_single_line_longer_than_the_budget_is_hard_split_not_dropped() {
        let text = "0123456789012345678901234567890\n"; // 31 chars + newline
        let chunks = chunk_transcript(text, 10);
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert!(c.chars().count() <= 10);
        }
        assert_eq!(chunks.concat(), text);
    }

    // -- sanitize_markdown -----------------------------------------------

    #[test]
    fn plain_markdown_survives_untouched() {
        let md = "## Decisions\n\n- We'll ship on Friday.\n- Ana owns the migration.\n";
        assert_eq!(sanitize_markdown(md), md);
    }

    #[test]
    fn a_script_tag_is_removed() {
        let md = "Notes.\n\n<script>fetch('https://evil.example/steal')</script>\n\nMore notes.";
        let out = sanitize_markdown(md);
        assert!(!out.contains("<script"));
        assert!(!out.contains("evil.example"));
    }

    #[test]
    fn a_remote_image_is_removed() {
        let md = "![tracker](https://evil.example/pixel.png)\n\nRecap text.";
        let out = sanitize_markdown(md);
        assert!(!out.contains("<img"));
        assert!(!out.contains("evil.example"));
        assert!(out.contains("Recap text."));
    }

    #[test]
    fn a_javascript_link_is_removed_entirely() {
        let md = "Before. [click me](javascript:alert(1)) After.";
        let out = sanitize_markdown(md);
        assert!(!out.to_lowercase().contains("javascript:"));
        assert!(!out.contains("click me"));
        assert!(out.contains("Before."));
        assert!(out.contains("After."));
    }

    #[test]
    fn an_ordinary_link_survives() {
        let md = "See [the doc](https://example.com/notes) for details.";
        assert_eq!(sanitize_markdown(md), md);
    }

    #[test]
    fn sanitizing_is_idempotent() {
        let md = "Plain recap with an & and a < in it.";
        let once = sanitize_markdown(md);
        let twice = sanitize_markdown(&once);
        assert_eq!(once, twice);
    }

    // -- action item JSON validation + repair -----------------------------

    #[test]
    fn valid_json_parses_directly() {
        let text =
            r#"{"items":[{"description":"Send the invoice","owner":"Ana","dueHint":"Friday"}]}"#;
        let items = parse_action_items(text).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].description, "Send the invoice");
        assert_eq!(items[0].owner.as_deref(), Some("Ana"));
        assert_eq!(items[0].due_hint.as_deref(), Some("Friday"));
    }

    #[test]
    fn json_wrapped_in_a_code_fence_still_parses() {
        let text = "Sure, here you go:\n```json\n{\"items\":[{\"description\":\"Follow up\"}]}\n```\nHope that helps!";
        let items = parse_action_items(text).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].description, "Follow up");
    }

    #[test]
    fn an_empty_items_list_is_valid() {
        let items = parse_action_items(r#"{"items": []}"#).unwrap();
        assert!(items.is_empty());
    }

    #[test]
    fn items_without_a_description_are_dropped_not_fatal() {
        let text = r#"{"items":[{"description":"  "},{"description":"Real task"}]}"#;
        let items = parse_action_items(text).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].description, "Real task");
    }

    #[test]
    fn unknown_extra_fields_fail_validation_so_a_repair_can_be_tried() {
        let text = r#"{"items":[{"description":"x","extra":"nope"}]}"#;
        assert!(matches!(
            parse_action_items(text),
            Err(SummarizeError::BadJson)
        ));
    }

    #[test]
    fn prose_with_no_json_at_all_fails_validation() {
        assert!(matches!(
            parse_action_items("Sorry, I don't see any tasks."),
            Err(SummarizeError::BadJson)
        ));
    }

    #[test]
    fn malformed_json_fails_validation() {
        assert!(matches!(
            parse_action_items("{\"items\": [ this is not json }"),
            Err(SummarizeError::BadJson)
        ));
    }
}
