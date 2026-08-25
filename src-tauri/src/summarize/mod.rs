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
    /// The key itself was refused: wrong, revoked, or not allowed to do this.
    #[error("the service refused the request: {0}")]
    Rejected(String),
    /// The service turned this *request* down, and not over the key: a schema
    /// keyword it does not know, a shape it does not accept, a feature the
    /// chosen model does not have. Any 4xx that is not about the credential and
    /// not an exhausted allowance.
    ///
    /// Its own variant because [`Self::Rejected`] means "the key was refused"
    /// all the way to the screen. Folded together, a 400 about a schema sent
    /// people to Settings to replace a key that was working perfectly
    /// (review of 2026-08-20, finding 5). `status` is the number, for the
    /// Advanced-only detail line; `detail` is the service's own prose, which
    /// never reaches the screen.
    #[error("the service turned the request down (http {status})")]
    RequestRefused { status: u16, detail: String },
    /// The key is fine — the account's allowance for it is used up (HTTP 429).
    ///
    /// Deliberately its own variant. Folded into [`Self::Rejected`] it became
    /// "that key wasn't accepted", which sent people off to re-paste a key that
    /// was never the problem.
    #[error("that service's allowance is used up for now")]
    QuotaExhausted,
    /// The service answered, and its answer was "no": its own safety or
    /// recitation rules stopped it writing about this meeting. Trying again with
    /// the same text gets the same answer, so this is not a retry.
    ///
    /// `reason` is the service's own word for it (`SAFETY`, `RECITATION`, …),
    /// for the log and the Advanced-only detail line — never for the screen.
    #[error("the service wouldn't write a recap from this meeting's text")]
    Blocked { reason: String },
    /// The call worked, the reply was readable, and there was no recap in it —
    /// no candidates, or candidates with no text. Worth trying again.
    #[error("the reply arrived with nothing written in it")]
    EmptyReply,
    /// The reply arrived in a shape Echo could not read at all: events came
    /// down the wire and not one of them parsed.
    ///
    /// A different bug class from [`Self::EmptyReply`] on purpose — empty means
    /// the service had nothing to say, this means Echo and the service disagree
    /// about the shape of an answer, which is Echo's bug to fix. The telemetry
    /// line logged alongside it carries the counters that say which.
    #[error("the reply arrived in a shape Echo couldn't read")]
    MalformedReply,
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
    /// How adventurous the wording may be, for backends that take a number for
    /// it. Only [`ollama`] reads this: Google asks that recent Gemini models be
    /// left at their own defaults, and a low value there is a documented cause
    /// of degraded replies, so [`gemini`] sends no sampling settings at all.
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
    last.ok_or(SummarizeError::EmptyReply)
}

/// What a line of transcript is written as when Echo was not sure it heard it.
///
/// A marker, not a redaction. See [`RecapInput`].
const UNCLEAR_MARKER: &str = "[unclear]";

/// One line of transcript on its way into a prompt, before it is decided how to
/// write it down.
#[derive(Debug, Clone)]
struct RecapLine {
    speaker: String,
    text: String,
    confidence: Option<f32>,
}

/// The transcript as the recap pass sees it: the prompt text, and the same
/// words split by whether Echo was sure it heard them.
///
/// **Mark, never silently drop.** A line Echo is unsure about is still the only
/// record of that moment, and a recap written from a transcript with holes in it
/// is worse than one written from a transcript that says where it is shaky:
/// dropping is Echo deciding, on the strength of a number, that something
/// nobody can check did not happen. So every word that was in the transcript is
/// still in [`Self::text`] — the shaky ones just carry
/// [`UNCLEAR_MARKER`] after them.
///
/// The two extra copies exist for one job only: [`owner_is_founded`], which
/// needs to know whether a name the model came back with was ever heard
/// clearly. They are never sent anywhere.
#[derive(Debug, Clone, Default)]
pub struct RecapInput {
    /// Every line, in order, `"{name}: {text}"` — with `[unclear]` appended to
    /// the lines Echo was not sure of. This is what the prompt gets.
    pub text: String,
    /// Is there at least one marker in [`Self::text`]? The prompt only explains
    /// the marker when there is one to explain: an instruction about a notation
    /// that never appears is an invitation to hedge a recap that had nothing to
    /// hedge about.
    pub any_unclear: bool,
    /// Speaker names, plus the words of the lines Echo heard clearly.
    ///
    /// Names are here for **every** line, including the shaky ones, and that is
    /// deliberate. A speaker's name comes from the speaker list — diarization
    /// and whatever the person typed on the meeting screen — not from what the
    /// engine thought it heard. Somebody who only ever spoke on lines Echo was
    /// unsure of still really is in this meeting, and clearing them off a task
    /// for that would be the exact false clear [`owner_is_founded`] is biased
    /// against.
    pub confident_text: String,
    /// The words of the lines Echo was not sure of, and nothing else.
    pub shaky_text: String,
}

/// Write the lines out three ways: once for the prompt, twice for the
/// owner check. Pure, so the marking rules can be tested without a database.
fn assemble_recap_input(lines: &[RecapLine]) -> RecapInput {
    let sure = crate::asr::confidence::HowSureThisMeetingIs::from_readings(
        lines.iter().filter_map(|l| l.confidence),
    );

    let mut input = RecapInput::default();
    for line in lines {
        let shaky = sure.is_shaky(line.confidence);
        input.text.push_str(&line.speaker);
        input.text.push_str(": ");
        input.text.push_str(&line.text);
        if shaky {
            input.text.push(' ');
            input.text.push_str(UNCLEAR_MARKER);
            input.any_unclear = true;
        }
        input.text.push('\n');

        // The name is founded either way (see `RecapInput::confident_text`);
        // only the words move.
        input.confident_text.push_str(&line.speaker);
        input.confident_text.push(':');
        if shaky {
            input.shaky_text.push_str(&line.text);
            input.shaky_text.push('\n');
        } else {
            input.confident_text.push(' ');
            input.confident_text.push_str(&line.text);
        }
        input.confident_text.push('\n');
    }
    input
}

/// Every final segment for this meeting, one line per segment, with speakers
/// resolved through any merge (DESIGN §3: "transcript (final segments w/
/// speaker names)") — and with the lines Echo was not sure it heard marked as
/// such (see [`RecapInput`] and [`crate::asr::confidence`]).
async fn build_transcript_text(db: &Db, meeting_id: &str) -> Result<RecapInput, SummarizeError> {
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
    let mut lines: Vec<RecapLine> = Vec::with_capacity(segments.len());
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
        lines.push(RecapLine {
            speaker: speaker_name,
            text: seg.text.trim().to_string(),
            confidence: seg.avg_confidence,
        });
    }
    Ok(assemble_recap_input(&lines))
}

/// What language this meeting was held in, read off the transcript: the one
/// with the most speaking time in final segments, falling back to whatever the
/// meeting row already says.
///
/// Two rules on top of "most speaking time", both from the 2026-08-24 meeting
/// (see [`crate::asr::language::spoken_in`]): a language holding a sliver of the
/// transcript never wins, and a second language holding a real share of it is
/// carried through to the prompt instead of being averaged away.
async fn dominant_language(
    db: &Db,
    meeting_id: &str,
    fallback: Option<&str>,
) -> Result<crate::asr::language::Spoken, SummarizeError> {
    let histogram = repo::language_histogram(db, meeting_id)
        .await
        .map_err(db_err)?;
    let spoken = crate::asr::language::spoken_in(&histogram);
    if spoken.dominant.is_some() {
        return Ok(spoken);
    }
    Ok(crate::asr::language::Spoken {
        dominant: fallback.map(String::from),
        also: None,
    })
}

/// A finished recap and the task list that came with it.
///
/// The two travel together because they are written together, exactly once —
/// see [`summarize_meeting_with_actions`].
#[derive(Debug, Clone)]
pub struct RecapOutcome {
    pub summary: Summary,
    /// Stored task items for this recap, as they now are in the database.
    /// Empty when the extraction pass found nothing or did not pan out; a recap
    /// is never failed over its task list.
    pub action_items: Vec<ActionItem>,
}

/// Write a recap for one meeting. See [`summarize_meeting_with_actions`], which
/// this wraps — the recap and its task list are written in one pass, and this
/// form simply drops the task list for callers that only want the recap.
pub async fn summarize_meeting(
    db: &Db,
    req: &SummaryReq,
    cancel: CancelFlag,
) -> Result<Summary, SummarizeError> {
    summarize_meeting_with_actions(db, req, cancel)
        .await
        .map(|outcome| outcome.summary)
}

/// Write a recap for one meeting, task list included.
///
/// Map-reduce over the transcript, then a strict-JSON pass for action items,
/// then sanitize, then store the recap with its provenance. Cancellable, and a
/// recording preempts it.
///
/// **This function is the only place a recap's task list is written.** It used
/// to run here *and* again in the summarize job, which meant two model calls
/// (four when both took their repair retry), two database writes, and a real
/// chance of the second pass overwriting a good first result with a worse one.
/// Callers that need the items take them from [`RecapOutcome`] rather than
/// calling [`extract_action_items`] a second time.
pub async fn summarize_meeting_with_actions(
    db: &Db,
    req: &SummaryReq,
    cancel: CancelFlag,
) -> Result<RecapOutcome, SummarizeError> {
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
                // The task list that already belongs to it, so a caller that
                // shows both does not have to go and ask for it (and must not
                // re-extract it: nothing about this meeting has changed).
                let action_items = repo::list_action_items(db, &req.meeting_id)
                    .await
                    .map_err(db_err)?;
                return Ok(RecapOutcome {
                    summary: existing,
                    action_items,
                });
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

    let recap_input = build_transcript_text(db, &req.meeting_id).await?;
    if recap_input.text.trim().is_empty() {
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
    let spoken = dominant_language(db, &req.meeting_id, meeting.language.as_deref()).await?;

    let mut ctx = templates::RenderContext {
        meeting_title: meeting.title.clone(),
        started_at: meeting.started_at.clone(),
        duration_ms: meeting.duration_ms,
        meeting_language: spoken.dominant,
        also_spoken: spoken.also,
        output_language,
        speakers,
        transcript_chunk: String::new(),
        chunk_index: 0,
        chunk_count: 1,
        some_lines_unclear: recap_input.any_unclear,
    };

    // Leave headroom for the instructions themselves, not just the transcript
    // text, so a chunk that "fits" does not actually overflow once the
    // template's own words are added.
    const PROMPT_OVERHEAD_CHARS: u32 = 1_500;
    const MIN_CHUNK_CHARS: u32 = 1_000;
    let budget = context_chars
        .saturating_sub(PROMPT_OVERHEAD_CHARS)
        .max(MIN_CHUNK_CHARS);
    let chunks = chunk_transcript(&recap_input.text, budget);
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
        // Honest limit, written down rather than papered over: the working
        // notes come back as the model's own prose, and nothing carries the
        // `[unclear]` markers forward into them. So the reduce step is told
        // that parts of this meeting were unclear, but it cannot re-check
        // *which* parts — it has only the notes, and by then the marking is
        // gone. A long meeting therefore gets weaker protection in the recap's
        // prose than a short one. The task list is not affected: the owner rule
        // below runs in code, over the transcript, on the final owners,
        // whichever path the prose took.
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

    // Action items are a bonus on top of a recap that already succeeded — never
    // fail the whole recap because the follow-up JSON pass did not pan out (see
    // the module doc comment: "a graceful 'no action items found'").
    let action_items = if cancel.is_cancelled() {
        Vec::new()
    } else {
        // The recap's own language, resolved above and now stored on the row —
        // not whatever the global setting says by the time this line runs.
        let language = action_item_language(summary.language.as_deref(), &ctx.output_language);
        match write_action_items(db, &summary, language, &recap_input, cancel.clone()).await {
            Ok(items) => items,
            Err(e) => {
                tracing::warn!("action items not written for summary {}: {e}", summary.id);
                Vec::new()
            }
        }
    };

    Ok(RecapOutcome {
        summary,
        action_items,
    })
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

/// What language the task list is written in.
///
/// The recap's language, which is stored on the recap row the moment it is
/// written — including the meeting's own language when the setting is "same as
/// the meeting". Re-reading the global setting here got this wrong twice over:
/// a recap written in Italian could be handed an English task list because
/// somebody changed the setting in between, and "same as the meeting" lost the
/// meeting language entirely, because nothing else about the render context was
/// filled in.
fn action_item_language(
    recap_language: Option<&str>,
    fallback: &SummaryLanguage,
) -> SummaryLanguage {
    match recap_language.map(str::trim).filter(|l| !l.is_empty()) {
        Some(language) => SummaryLanguage::Fixed(language.to_string()),
        None => fallback.clone(),
    }
}

/// Extract action items from a finished recap. Strict JSON, validated locally,
/// one repair retry.
///
/// Prefer [`summarize_meeting_with_actions`], which already does this as part of
/// writing the recap; this entry point exists for re-running the pass over a
/// recap that is already stored.
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
    if summary.meeting_id != meeting_id {
        return Err(SummarizeError::Failed(
            "that recap belongs to another meeting".into(),
        ));
    }

    let app_settings = settings::load(db).await.map_err(db_err)?;
    let language =
        action_item_language(summary.language.as_deref(), &app_settings.summary_language);
    // Read again, because the owner rule ([`owner_is_founded`]) is a check
    // against the transcript and this entry point starts from a recap that was
    // written some time ago. One extra read of rows that are already indexed by
    // meeting, next to a model round-trip.
    let recap_input = build_transcript_text(db, meeting_id).await?;
    write_action_items(db, &summary, language, &recap_input, cancel).await
}

// ---------------------------------------------------------------------------
// The owner rule
// ---------------------------------------------------------------------------

/// Take the owner off any task whose owner Echo only ever heard on a line it
/// was unsure of. The task stays.
///
/// Runs over the parsed reply, before anything is stored, so there is no path
/// from a shaky line to a name on a task list — including the repair retry, a
/// re-extraction over an old recap, and whatever a future template asks for.
fn ground_owners(items: Vec<ActionItemRaw>, recap: &RecapInput) -> Vec<ActionItemRaw> {
    items
        .into_iter()
        .map(|mut item| {
            let Some(owner) = item.owner.as_deref() else {
                return item;
            };
            if owner_is_founded(owner, &recap.confident_text, &recap.shaky_text) {
                return item;
            }
            tracing::debug!(
                target: "echo::summarize",
                owner,
                "took an owner off a task: that name was only ever heard on a line Echo was unsure of"
            );
            // The task, not the name. Somebody agreed to do this; what Echo
            // cannot stand behind is who.
            item.owner = None;
            item
        })
        .collect()
}

/// May this name be printed as the person responsible for a task?
///
/// The rule the user gave is *never* attribute a task to a name Echo only heard
/// on a line it was unsure of — and a prompt is a request, not a rule. A model
/// asked nicely complies most of the time, and "most of the time" is not what
/// *never* means. So the check runs here, in code, over the transcript, after
/// the reply has been parsed.
///
/// Three answers, and only one of them clears anything:
///
/// * **the name appears in text Echo heard clearly → keep.** Even if it also
///   appears on a shaky line. That is what "only" means: one clear hearing is
///   enough to found the name, and the rule was never about how often it was
///   misheard.
/// * **the name appears on shaky lines and nowhere clear → clear the owner,
///   keep the task.** The task itself came from the recap and was really
///   agreed; it is the *who* that rests on words Echo did not catch.
/// * **the name appears nowhere in the transcript → keep.** This is the common
///   and completely legitimate case: "the team", "Engineering", "whoever picks
///   up the on-call", or a paraphrase of a name that was said differently. None
///   of those are misheard names, and the rule has nothing to say about them.
///
/// The bias is deliberate and it runs one way. **A false clear deletes a
/// correct owner** — the person reads a task with nobody's name on it and has
/// to remember who said they would do it — whereas a false keep leaves a name
/// that is at worst the wrong one, next to a task the person can read and
/// judge. The first is the worse error, so anything ambiguous keeps.
///
/// Matching is by folded words ([`crate::asr::glossary`], the same word reader
/// and the same case/accent flattening the glossary uses to recognise a spoken
/// name — rather than a fourth hand-rolled word matcher), and **the two sides
/// are read with different eyes, because a match means opposite things on each
/// of them.**
///
/// On the confident side a match is a reason to *keep*, the safe direction, so
/// it is read generously: the whole owner string as a run of consecutive words,
/// or **any single word of the name that could identify the person on its own**.
/// "Anna Bianchi" is founded by hearing "Bianchi" clearly, and equally by
/// hearing "Anna" clearly — a first name said plainly is a hearing, and reading
/// only the longest word here was a false clear waiting to happen: the clear
/// lines say "Anna will handle the invoices", one shaky line carries the
/// surname, and the correct owner is deleted. Words shorter than
/// [`SHORTEST_FOUNDING_WORD`] are the exception — "de", "la", an initial —
/// because they say nothing about who was meant and turn up everywhere; a name
/// made only of such words falls back to its longest word so nothing gets
/// *stricter* than reading one word did.
///
/// On the shaky side a match is a reason to *clear*, the dangerous direction, so
/// it stays narrow: the whole run, or the longest word — the part that actually
/// carries the identity. Widening it there would clear more owners, which is the
/// error this rule is biased against.
fn owner_is_founded(owner: &str, confident: &str, shaky: &str) -> bool {
    let wanted = folded_words(owner);
    if wanted.is_empty() {
        // Punctuation, an emoji, an empty string: not a name, nothing to check.
        return true;
    }
    let longest = wanted
        .iter()
        .max_by_key(|w| w.chars().count())
        .expect("checked non-empty");
    // Every word of the name that could stand for the person on its own. Empty
    // for a name made entirely of particles or initials, and then the longest
    // word carries it alone, exactly as it used to.
    let mut founding: Vec<&String> = wanted
        .iter()
        .filter(|w| w.chars().count() >= SHORTEST_FOUNDING_WORD)
        .collect();
    if founding.is_empty() {
        founding.push(longest);
    }

    let confident_words = folded_words(confident);
    if contains_run(&confident_words, &wanted)
        || confident_words.iter().any(|w| founding.contains(&w))
    {
        return true;
    }

    let shaky_words = folded_words(shaky);
    if contains_run(&shaky_words, &wanted) || shaky_words.iter().any(|w| w == longest) {
        return false;
    }
    true
}

/// How many letters a word of a name needs before hearing it clearly can found
/// the whole name.
///
/// Three, which keeps out "de", "la", "van", "d'" and single initials — the
/// pieces of a name that identify nobody and appear in half of any Italian
/// transcript — while letting in every first name and surname short enough to
/// worry about ("Ivo", "Ada", "Li" via the fallback).
const SHORTEST_FOUNDING_WORD: usize = 3;

/// The words of a piece of text, folded the way the glossary folds them.
fn folded_words(text: &str) -> Vec<String> {
    crate::asr::glossary::word_spans(text)
        .into_iter()
        .map(|(from, to)| crate::asr::glossary::fold(&text[from..to]))
        .filter(|w| !w.is_empty())
        .collect()
}

/// Does `needle` appear in `haystack` as a run of consecutive words?
fn contains_run(haystack: &[String], needle: &[String]) -> bool {
    if needle.is_empty() || needle.len() > haystack.len() {
        return false;
    }
    haystack.windows(needle.len()).any(|w| w == needle)
}

/// The one pass that asks for a task list and stores it.
async fn write_action_items(
    db: &Db,
    summary: &Summary,
    language: SummaryLanguage,
    recap_input: &RecapInput,
    cancel: CancelFlag,
) -> Result<Vec<ActionItem>, SummarizeError> {
    // The recap's provider, but today's model: `summary.model` says which model
    // wrote that recap, and it may since have been changed or retired. Asking a
    // model that no longer exists for the task list would fail for a reason that
    // has nothing to do with the task list.
    let connector = connector_for(db, summary.provider).await?;
    let json_mode = connector.capabilities().json_mode;

    let ctx = templates::RenderContext {
        output_language: language,
        ..Default::default()
    };
    let prompt = templates::render_action_items(&summary.content_md, &ctx);
    let schema = templates::action_item_schema();

    let raw = match run_action_item_pass(connector.as_ref(), &prompt, &schema, json_mode, &cancel)
        .await
    {
        Ok(items) => items,
        // The repair retry exists for exactly one failure: a reply that came
        // back and could not be read as the JSON we asked for. Every other
        // failure — a refused key, a used-up allowance, a safety block, a
        // rejected schema, a model that is gone — would get the identical
        // answer a second time, and the repair prompt would tell the service
        // its last reply was unreadable when it never sent one. Those go
        // straight back to the caller, which keeps the recap and logs why the
        // task list is missing.
        Err(SummarizeError::BadJson) => {
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
                Err(second) => {
                    tracing::warn!(error = %second, "the task list came back unreadable twice");
                    Vec::new()
                }
            }
        }
        Err(other) => return Err(other),
    };

    // Between parsing and storing: the owner rule, enforced here rather than
    // asked for in the prompt. See [`owner_is_founded`].
    let raw = ground_owners(raw, recap_input);

    let items: Vec<ActionItem> = raw
        .into_iter()
        .map(|item| ActionItem {
            id: String::new(),
            meeting_id: summary.meeting_id.clone(),
            summary_id: Some(summary.id.clone()),
            description: item.description,
            owner: item.owner,
            due_hint: item.due_hint,
            done: false,
            external_url: None,
        })
        .collect();

    repo::replace_action_items(db, &summary.meeting_id, Some(&summary.id), &items)
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

    fn said(speaker: &str, text: &str, confidence: Option<f32>) -> RecapLine {
        RecapLine {
            speaker: speaker.to_string(),
            text: text.to_string(),
            confidence,
        }
    }

    /// Enough clearly-read lines that the meeting has a middle to be judged
    /// against ([`crate::asr::confidence::MEDIAN_NEEDS`]).
    fn a_meeting_that_reads_well() -> Vec<RecapLine> {
        (0..10)
            .map(|i| said("Marco", &format!("line number {i}"), Some(0.92)))
            .collect()
    }

    #[test]
    fn a_shaky_line_is_marked_and_nothing_is_dropped() {
        let mut lines = a_meeting_that_reads_well();
        lines.push(said("Giulia", "something Echo half caught", Some(0.30)));
        let input = assemble_recap_input(&lines);

        // Every word that was in the transcript is still in the prompt text.
        // This is the whole rule: mark, never silently drop. A recap written
        // from a transcript with holes punched in it is Echo deciding that
        // something nobody can check did not happen.
        for line in &lines {
            assert!(
                input.text.contains(&line.text),
                "{:?} was dropped from the prompt",
                line.text
            );
            assert!(input.text.contains(&line.speaker));
        }
        assert_eq!(
            input.text.lines().count(),
            lines.len(),
            "one line in, one line out"
        );

        // The shaky one carries the marker; the confident ones do not.
        assert!(input.any_unclear);
        let marked: Vec<&str> = input
            .text
            .lines()
            .filter(|l| l.contains(UNCLEAR_MARKER))
            .collect();
        assert_eq!(marked, vec!["Giulia: something Echo half caught [unclear]"]);
    }

    #[test]
    fn a_meeting_echo_heard_clearly_carries_no_markers_at_all() {
        let input = assemble_recap_input(&a_meeting_that_reads_well());
        assert!(!input.any_unclear);
        assert!(!input.text.contains(UNCLEAR_MARKER));
        assert!(input.shaky_text.trim().is_empty());
    }

    #[test]
    fn the_two_side_copies_split_the_words_and_keep_every_name() {
        let mut lines = a_meeting_that_reads_well();
        lines.push(said("Giulia", "handle the invoices", Some(0.30)));
        let input = assemble_recap_input(&lines);

        // Words of the shaky line: on the shaky side only.
        assert!(input.shaky_text.contains("handle the invoices"));
        assert!(!input.confident_text.contains("handle the invoices"));
        // Words of a clear line: on the confident side only.
        assert!(input.confident_text.contains("line number 3"));
        assert!(!input.shaky_text.contains("line number 3"));
        // Both names are founded, because a name comes from the speaker list
        // rather than from what the engine thought it heard.
        assert!(input.confident_text.contains("Marco"));
        assert!(input.confident_text.contains("Giulia"));
    }

    #[test]
    fn an_owner_is_only_cleared_when_the_name_was_never_heard_clearly() {
        let confident = "Marco: I'll take the pricing page\nGiulia:\n";
        let shaky = "Bianchi is going to redo the invoices with Anna\n";

        // (owner, keep?, why)
        let table: &[(&str, bool, &str)] = &[
            // Heard clearly: keep, no question.
            ("Marco", true, "said clearly"),
            ("marco", true, "folding flattens case"),
            // A speaker's name is founded even though every word they said was
            // shaky — the name is from the speaker list, not from the audio.
            ("Giulia", true, "on the speaker list"),
            // Only ever on a line Echo was unsure of: the task is real, the
            // name is not founded.
            ("Bianchi", false, "shaky only"),
            ("Anna", false, "shaky only"),
            ("Anna Bianchi", false, "shaky only, as a whole run"),
            // Nowhere in the transcript at all: keep. "the team" is not a
            // misheard name, and the rule has nothing to say about it.
            ("the team", true, "not a name from the transcript"),
            ("Engineering", true, "a role, not a hearing"),
            ("Unassigned", true, "nowhere in the transcript"),
            // Not a name at all.
            ("", true, "nothing to check"),
            ("—", true, "no words in it"),
        ];
        for (owner, keep, why) in table {
            assert_eq!(
                owner_is_founded(owner, confident, shaky),
                *keep,
                "{owner:?} ({why})"
            );
        }
    }

    #[test]
    fn a_name_heard_clearly_once_is_founded_however_often_it_was_misheard() {
        // "Only" means only. One clear hearing settles it, and the rule was
        // never about how often a name turned up on a shaky line.
        let confident = "Marco: fine by me\n";
        let shaky = "Marco Marco Marco Marco\n";
        assert!(owner_is_founded("Marco", confident, shaky));
    }

    #[test]
    fn a_two_part_name_is_founded_by_its_longest_word() {
        // The whole run is not in the confident text, but the part that
        // actually identifies the person is.
        let confident = "Bianchi said she would send it\n";
        let shaky = "anna will do it\n";
        assert!(owner_is_founded("Anna Bianchi", confident, shaky));
    }

    #[test]
    fn a_two_part_name_is_founded_by_whichever_part_was_heard_clearly() {
        // The clear lines say the first name, one shaky line carries the
        // surname. Reading only the longest word here deleted a correct owner:
        // "Anna" was heard perfectly well, and one clear hearing founds a name.
        let confident = "Anna: I'll send it\nAnna said she would send it\n";
        let shaky = "Bianchi will redo the invoices\n";
        assert!(owner_is_founded("Anna Bianchi", confident, shaky));
    }

    #[test]
    fn a_particle_heard_clearly_founds_nothing() {
        // "de" is in every other Italian sentence and says nothing about who
        // was meant, so hearing it clearly is not a hearing of the name.
        let confident = "Marco: parliamo de visu domani\n";
        let shaky = "anna de rossi si occupa delle fatture\n";
        assert!(!owner_is_founded("Anna de Rossi", confident, shaky));
        // The parts that do identify her still found her.
        assert!(owner_is_founded(
            "Anna de Rossi",
            "Anna: ci penso io\n",
            shaky
        ));
        assert!(owner_is_founded(
            "Anna de Rossi",
            "Rossi: ci penso io\n",
            shaky
        ));
    }

    #[test]
    fn a_name_too_short_to_stand_alone_still_founds_itself() {
        // Nothing may get stricter than reading the longest word did: a name
        // with no word of three letters falls back to it.
        let confident = "Li: I'll take the pricing page\n";
        let shaky = "li does the invoices\n";
        assert!(owner_is_founded("Li", confident, shaky));
        assert!(!owner_is_founded("Li", "Marco: fine by me\n", shaky));
    }

    #[test]
    fn clearing_an_owner_keeps_the_task() {
        let recap = assemble_recap_input(&{
            let mut lines = a_meeting_that_reads_well();
            lines.push(said(
                "Unknown speaker",
                "Bianchi does the invoices",
                Some(0.2),
            ));
            lines
        });
        let items = vec![
            ActionItemRaw {
                description: "Redo the invoices".into(),
                owner: Some("Bianchi".into()),
                due_hint: Some("Friday".into()),
            },
            ActionItemRaw {
                description: "Ship the pricing page".into(),
                owner: Some("Marco".into()),
                due_hint: None,
            },
        ];
        let grounded = ground_owners(items, &recap);

        // The task survives whole — description and due date untouched. Only
        // the who was unfounded, and only the who is gone.
        assert_eq!(grounded.len(), 2);
        assert_eq!(grounded[0].description, "Redo the invoices");
        assert_eq!(grounded[0].due_hint.as_deref(), Some("Friday"));
        assert_eq!(grounded[0].owner, None);
        // ...and a founded owner is left exactly as the model wrote it.
        assert_eq!(grounded[1].owner.as_deref(), Some("Marco"));
    }

    #[test]
    fn a_task_with_no_owner_is_left_alone() {
        let recap = assemble_recap_input(&a_meeting_that_reads_well());
        let items = vec![ActionItemRaw {
            description: "Book the room".into(),
            owner: None,
            due_hint: None,
        }];
        let grounded = ground_owners(items, &recap);
        assert_eq!(grounded.len(), 1);
        assert_eq!(grounded[0].owner, None);
    }

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

    // -- which language the task list is written in ------------------------

    #[test]
    fn the_task_list_follows_the_recap_not_todays_setting() {
        // The recap says Italian, the setting has since been changed to
        // English. The task list belongs to the recap.
        assert_eq!(
            action_item_language(Some("it"), &SummaryLanguage::English),
            SummaryLanguage::Fixed("it".into())
        );
        // "Same as the meeting" was already resolved to a real language when
        // the recap was written, so nothing has to guess it a second time.
        assert_eq!(
            action_item_language(Some("de"), &SummaryLanguage::SameAsMeeting),
            SummaryLanguage::Fixed("de".into())
        );
        // An older recap with no language recorded falls back to the setting.
        assert_eq!(
            action_item_language(None, &SummaryLanguage::English),
            SummaryLanguage::English
        );
        assert_eq!(
            action_item_language(Some("  "), &SummaryLanguage::English),
            SummaryLanguage::English
        );
    }

    // -- what language the meeting was in ----------------------------------

    /// A transcript made of stretches: `(language, milliseconds)`.
    async fn transcript_of(db: &Db, spoken: &[(&str, i64)]) -> String {
        use crate::types::{Channel, SegmentDraft};

        let meeting = repo::create_meeting(db, "Weekly sync", "/tmp/echo-test", None)
            .await
            .unwrap();
        let mut at = 0;
        let drafts: Vec<SegmentDraft> = spoken
            .iter()
            .map(|(language, ms)| {
                let draft = SegmentDraft {
                    meeting_id: meeting.id.clone(),
                    t_start_ms: at,
                    t_end_ms: at + ms,
                    channel: Channel::Mic,
                    speaker_id: None,
                    text: "qualcosa".to_string(),
                    language: Some((*language).to_string()),
                    avg_confidence: Some(0.9),
                    revision: 1,
                    is_final: true,
                    model_name: None,
                    model_revision: None,
                    corrections: Vec::new(),
                };
                at += ms;
                draft
            })
            .collect();
        repo::insert_segments(db, &drafts).await.unwrap();
        meeting.id
    }

    /// One line out of a whole meeting misread as another language decides
    /// nothing — on 2026-08-24 a single stray segment was the sort of thing
    /// that could have taken the recap with it.
    #[tokio::test]
    async fn a_stray_line_never_decides_what_the_recap_is_written_in() {
        let db = crate::db::connect_in_memory().await.unwrap();
        let meeting_id = transcript_of(&db, &[("it", 3_600_000), ("zh", 30_000)]).await;
        let spoken = dominant_language(&db, &meeting_id, None).await.unwrap();
        assert_eq!(spoken.dominant.as_deref(), Some("it"));
        assert_eq!(spoken.also, None, "one percent is not a language it was in");
    }

    /// And a meeting really held in two languages says so, so the recap is
    /// written in one of them rather than averaged between them.
    #[tokio::test]
    async fn a_meeting_held_in_two_languages_reaches_the_prompt_as_both() {
        let db = crate::db::connect_in_memory().await.unwrap();
        let meeting_id = transcript_of(&db, &[("it", 700_000), ("en", 300_000)]).await;
        let spoken = dominant_language(&db, &meeting_id, None).await.unwrap();
        assert_eq!(spoken.dominant.as_deref(), Some("it"));
        assert_eq!(spoken.also.as_deref(), Some("en"));
    }

    /// Nothing on the transcript to go on: whatever the meeting row says.
    #[tokio::test]
    async fn a_transcript_with_no_languages_falls_back_to_the_meeting_row() {
        let db = crate::db::connect_in_memory().await.unwrap();
        let meeting_id = transcript_of(&db, &[]).await;
        let spoken = dominant_language(&db, &meeting_id, Some("de"))
            .await
            .unwrap();
        assert_eq!(spoken.dominant.as_deref(), Some("de"));
    }

    // -- one recap, one task-list pass -------------------------------------

    /// A meeting with a couple of final segments, ready to summarise.
    async fn meeting_with_a_transcript(db: &Db) -> String {
        use crate::types::{Channel, SegmentDraft};

        let meeting = repo::create_meeting(db, "Weekly sync", "/tmp/echo-test", None)
            .await
            .unwrap();
        let drafts: Vec<SegmentDraft> = ["Spediamo venerdì.", "Ana si occupa della migrazione."]
            .iter()
            .enumerate()
            .map(|(i, text)| SegmentDraft {
                meeting_id: meeting.id.clone(),
                t_start_ms: i as i64 * 5_000,
                t_end_ms: i as i64 * 5_000 + 4_000,
                channel: Channel::Mic,
                speaker_id: None,
                text: (*text).to_string(),
                language: Some("it".into()),
                avg_confidence: Some(0.9),
                revision: 1,
                is_final: true,
                model_name: None,
                model_revision: None,
                corrections: Vec::new(),
            })
            .collect();
        repo::insert_segments(db, &drafts).await.unwrap();
        meeting.id
    }

    #[tokio::test]
    async fn a_recap_asks_for_its_task_list_exactly_once() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        // One reply that serves for both passes: readable as a recap, and
        // readable as the task-list JSON.
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                "{\"message\":{\"content\":\"{\\\"items\\\":[{\\\"description\\\":\\\"Send the \
                 invoice\\\"}]}\"},\"done\":true}\n",
                "application/x-ndjson",
            ))
            .mount(&server)
            .await;

        let db = crate::db::connect_in_memory().await.unwrap();
        repo::set_setting(&db, settings::keys::OLLAMA_BASE_URL, &server.uri())
            .await
            .unwrap();
        repo::set_setting(&db, settings::keys::OLLAMA_MODEL, "test-model")
            .await
            .unwrap();
        let meeting_id = meeting_with_a_transcript(&db).await;

        let outcome = summarize_meeting_with_actions(
            &db,
            &SummaryReq {
                meeting_id: meeting_id.clone(),
                ..Default::default()
            },
            CancelFlag::new(),
        )
        .await
        .unwrap();

        // Two calls, not four: one for the recap, one for the task list. The
        // job used to run the task-list pass again on its own.
        let calls = server.received_requests().await.unwrap();
        assert_eq!(calls.len(), 2, "one recap call and one task-list call");

        assert_eq!(outcome.action_items.len(), 1);
        assert_eq!(outcome.action_items[0].description, "Send the invoice");
        assert_eq!(
            outcome.action_items[0].summary_id.as_deref(),
            Some(outcome.summary.id.as_str())
        );

        // Stored once, and the stored rows are the ones handed back.
        let stored = repo::list_action_items(&db, &meeting_id).await.unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].id, outcome.action_items[0].id);

        // The recap's language came from the meeting, and the task-list prompt
        // asked for that language by name.
        assert_eq!(outcome.summary.language.as_deref(), Some("it"));
        let last: serde_json::Value = serde_json::from_slice(&calls[1].body).unwrap();
        let prompt = last["messages"][0]["content"].as_str().unwrap_or_default();
        assert!(prompt.contains("Write every task in Italian."), "{prompt}");
    }

    #[tokio::test]
    async fn asking_again_for_the_same_recap_costs_nothing_and_still_returns_its_tasks() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                "{\"message\":{\"content\":\"{\\\"items\\\":[{\\\"description\\\":\\\"Send the \
                 invoice\\\"}]}\"},\"done\":true}\n",
                "application/x-ndjson",
            ))
            .mount(&server)
            .await;

        let db = crate::db::connect_in_memory().await.unwrap();
        repo::set_setting(&db, settings::keys::OLLAMA_BASE_URL, &server.uri())
            .await
            .unwrap();
        repo::set_setting(&db, settings::keys::OLLAMA_MODEL, "test-model")
            .await
            .unwrap();
        let meeting_id = meeting_with_a_transcript(&db).await;
        let req = SummaryReq {
            meeting_id: meeting_id.clone(),
            ..Default::default()
        };

        let first = summarize_meeting_with_actions(&db, &req, CancelFlag::new())
            .await
            .unwrap();
        let again = summarize_meeting_with_actions(&db, &req, CancelFlag::new())
            .await
            .unwrap();

        assert_eq!(first.summary.id, again.summary.id);
        assert_eq!(again.action_items.len(), 1);
        // Still two calls in total: the second request was answered from what
        // was already written down.
        assert_eq!(server.received_requests().await.unwrap().len(), 2);
    }
}
