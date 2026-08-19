//! Google Gemini through an AI Studio key.
//!
//! IMPLEMENTED-BY: summarize agent (M5).
//!
//! Privacy, stated plainly (DESIGN §3, review finding 14). The setup screen has
//! to say, before the key is saved:
//! * the transcript text of the meetings you summarise is sent to Google
//! * what Google does with it depends on your account and billing tier, with a
//!   link to their terms
//! * "on this computer" recaps send nothing anywhere
//!
//! The key lives in the keychain ([`crate::secrets`]) and never crosses IPC.
//! It travels to Google as the `x-goog-api-key` header rather than a `?key=`
//! query parameter, so it never ends up copied into a URL that shows up in an
//! error string or a log line. Log lines carry neither the key nor transcript
//! text.

use std::time::Duration;

use futures::future::BoxFuture;
use futures::stream::{self, BoxStream, StreamExt};
use serde::Deserialize;

use crate::summarize::http_util::{classify_reqwest_error, truncate, with_one_retry};
use crate::summarize::{CancelFlag, Connector, GenerateEvent, GenerateRequest, SummarizeError};
use crate::types::{Caps, Provider, ProviderTestResult};

pub const API_BASE: &str = "https://generativelanguage.googleapis.com";

/// The model Echo uses when nobody has chosen one.
///
/// This is the only place in the code that names a Gemini model, so moving to a
/// newer default is a one-line change. Flash is the right shape for a recap:
/// fast, cheap, and more than good enough at prose. If Google ever retires it,
/// the next recap comes back as [`SummarizeError::ModelNotFound`] and the
/// person is asked to pick another one in Settings — the job fails politely, it
/// never panics.
pub const DEFAULT_MODEL: &str = "gemini-3.7-flash";

/// Conservative working size for the prompt. Gemini's window is far larger, but
/// smaller chunks give better recaps and cost less.
pub const DEFAULT_CONTEXT_CHARS: u32 = 120_000;

/// The exact sentence shown before a key is saved. Kept in Rust as well as the
/// UI so a test can assert we never quietly drop it.
pub const PRIVACY_NOTICE: &str = "Your meeting text is sent to Google to write the recap. What Google keeps depends on your account and billing plan.";

const PROBE_TIMEOUT: Duration = Duration::from_secs(10);
const HEADER_API_KEY: &str = "x-goog-api-key";

#[derive(Clone)]
pub struct GeminiConnector {
    /// Read from the keychain when the connector is built, never stored.
    api_key: String,
    /// Resolved once, when the connector is built from stored settings. A recap
    /// never carries a model of its own (see [`Connector::model`]).
    pub model: String,
    /// Where requests go. Always [`API_BASE`] in the app; a test points it at a
    /// local mock server so no test can ever reach Google.
    api_base: String,
    client: reqwest::Client,
}

impl GeminiConnector {
    pub fn new(api_key: impl Into<String>, model: Option<String>) -> Self {
        Self {
            api_key: api_key.into(),
            // A blank setting means "not chosen", same as an absent one.
            model: model
                .filter(|m| !m.trim().is_empty())
                .map(|m| m.trim().to_string())
                .unwrap_or_else(|| DEFAULT_MODEL.to_string()),
            api_base: API_BASE.to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Length only. Enough to render a masked value without reading the key.
    pub fn key_len(&self) -> usize {
        self.api_key.len()
    }

    #[cfg(test)]
    fn with_api_base(mut self, base: impl Into<String>) -> Self {
        self.api_base = base.into();
        self
    }
}

#[derive(Debug, Deserialize)]
struct ModelsResponse {
    #[serde(default)]
    models: Vec<ModelEntry>,
}

#[derive(Debug, Deserialize)]
struct ModelEntry {
    name: String,
    #[serde(rename = "supportedGenerationMethods", default)]
    supported_generation_methods: Vec<String>,
}

#[derive(Debug, Default, Deserialize)]
struct GenerateChunk {
    #[serde(default)]
    candidates: Vec<Candidate>,
    #[serde(rename = "promptFeedback", default)]
    prompt_feedback: Option<PromptFeedback>,
}

#[derive(Debug, Deserialize)]
struct PromptFeedback {
    #[serde(rename = "blockReason", default)]
    block_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Candidate {
    #[serde(default)]
    content: Option<Content>,
    #[serde(rename = "finishReason", default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Content {
    #[serde(default)]
    parts: Vec<Part>,
}

#[derive(Debug, Deserialize)]
struct Part {
    #[serde(default)]
    text: String,
}

/// Families that exist to make pictures, video, speech or vectors. None of them
/// can write a recap, and every one of them shows up in the same catalogue as
/// the ones that can.
///
/// Matched against the name's hyphen/dot-separated pieces rather than as a
/// substring, so a fragment as short as `tts` can never knock out a model whose
/// name merely happens to contain those letters.
const NON_TEXT_NAME_PARTS: &[&str] = &[
    "imagen",
    "veo",
    "tts",
    "audio",
    "aqa",
    "embed",
    "image",
    "images",
    "video",
    "banana", // "nano-banana", Google's picture model nickname
];

/// The same idea for names that only read as one word once the punctuation is
/// gone — `text-embedding-004` and `embeddinggemma` both land here.
const NON_TEXT_COLLAPSED: &[&str] = &["embedding", "nanobanana", "texttospeech"];

/// Does this model claim it can answer a `generateContent` call?
///
/// Google reports the methods per model. Either name counts: streaming is the
/// path Echo actually uses, and a model that lists only the streaming form can
/// still write the recap. An entry that reports **no** methods at all is kept —
/// that is a shape we did not expect, and a surprise in the list is cheaper
/// than a list that has silently gone empty.
fn supports_generate_content(methods: &[String]) -> bool {
    methods.is_empty()
        || methods
            .iter()
            .any(|m| m == "generateContent" || m == "streamGenerateContent")
}

/// Could a model with this name plausibly write prose?
///
/// Deliberately narrow. An unfamiliar name stays in the list: a stranger there
/// costs one puzzled look, while a wrongly hidden model costs somebody the
/// exact model they were looking for.
fn is_text_model_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    let collapsed: String = lower.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
    if NON_TEXT_COLLAPSED
        .iter()
        .any(|marker| collapsed.contains(marker))
    {
        return false;
    }
    !lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .any(|part| NON_TEXT_NAME_PARTS.contains(&part))
}

/// Flash first: it is what most people want and what Echo defaults to, so it
/// belongs at the top of the list rather than wherever Google happened to put
/// it. Everything else keeps the order it arrived in.
fn flash_first(names: &mut [String]) {
    names.sort_by_key(|name| u8::from(!name.to_ascii_lowercase().contains("flash")));
}

/// The models from one catalogue response that can write a recap, best first.
fn recap_capable_models(models: Vec<ModelEntry>) -> Vec<String> {
    let mut names: Vec<String> = models
        .into_iter()
        .filter(|m| supports_generate_content(&m.supported_generation_methods))
        .map(|m| m.name.trim_start_matches("models/").to_string())
        .filter(|name| !name.is_empty() && is_text_model_name(name))
        .collect();
    flash_first(&mut names);
    names.dedup();
    names
}

/// Google's answer when the saved model is not one it will answer for: it was
/// renamed, retired, or mistyped. Usually a 404; a 400 saying the same thing
/// happens too. Worth telling apart from every other refusal, because this is
/// the one the person can fix themselves.
fn is_model_not_found(status: reqwest::StatusCode, body: &str) -> bool {
    if !matches!(status.as_u16(), 400 | 404) {
        return false;
    }
    let lower = body.to_ascii_lowercase();
    lower.contains("not found")
        || lower.contains("not_found")
        || lower.contains("is not supported")
        || lower.contains("unsupported model")
        || lower.contains("unexpected model name")
}

fn classify_status(status: reqwest::StatusCode, body: &str) -> SummarizeError {
    match status.as_u16() {
        401 | 403 => SummarizeError::Rejected("the saved key was refused".into()),
        429 => SummarizeError::Rejected("their usage limit was reached; try again later".into()),
        400..=499 => SummarizeError::Rejected(truncate(body, 200)),
        _ => SummarizeError::Failed(format!("http {status}: {}", truncate(body, 200))),
    }
}

/// `SAFETY`, `RECITATION` and friends mean the model refused to answer; only
/// `STOP` (and `MAX_TOKENS`, a truncation, not a refusal) mean it actually
/// wrote something.
fn is_blocked_finish_reason(reason: &str) -> bool {
    !matches!(reason, "STOP" | "MAX_TOKENS" | "")
}

/// Parse one Server-Sent-Events response (`alt=sse`) into the events it
/// describes, accumulating the full reply as it goes.
async fn drain_sse(
    mut resp: reqwest::Response,
    cancel: &CancelFlag,
) -> Result<Vec<GenerateEvent>, SummarizeError> {
    let mut buf = String::new();
    let mut full = String::new();
    let mut events = Vec::new();

    loop {
        if cancel.is_cancelled() {
            return Err(SummarizeError::Cancelled);
        }
        match resp.chunk().await {
            Ok(Some(bytes)) => {
                buf.push_str(&String::from_utf8_lossy(&bytes));
                while let Some(pos) = buf.find("\n\n") {
                    let event_block = buf[..pos].to_string();
                    buf.drain(..pos + 2);
                    let data: String = event_block
                        .lines()
                        .filter_map(|line| line.strip_prefix("data:"))
                        .map(|rest| rest.trim_start())
                        .collect::<Vec<_>>()
                        .join("");
                    if data.is_empty() || data == "[DONE]" {
                        continue;
                    }
                    let chunk: GenerateChunk = match serde_json::from_str(&data) {
                        Ok(c) => c,
                        Err(_) => continue,
                    };
                    if let Some(reason) = chunk.prompt_feedback.and_then(|f| f.block_reason) {
                        return Err(SummarizeError::Rejected(format!(
                            "blocked before it could answer ({reason})"
                        )));
                    }
                    for candidate in &chunk.candidates {
                        if let Some(reason) = &candidate.finish_reason {
                            if is_blocked_finish_reason(reason) {
                                return Err(SummarizeError::Rejected(format!(
                                    "the reply was blocked ({reason})"
                                )));
                            }
                        }
                        if let Some(content) = &candidate.content {
                            for part in &content.parts {
                                if !part.text.is_empty() {
                                    full.push_str(&part.text);
                                    events.push(GenerateEvent::Text(part.text.clone()));
                                }
                            }
                        }
                    }
                }
            }
            Ok(None) => break,
            Err(e) => return Err(classify_reqwest_error(&e)),
        }
    }

    if full.is_empty() {
        return Err(SummarizeError::Failed("the reply came back empty".into()));
    }
    events.push(GenerateEvent::Done { text: full });
    Ok(events)
}

#[allow(clippy::too_many_arguments)]
async fn run_generate_content(
    client: reqwest::Client,
    api_base: String,
    api_key: String,
    model: String,
    system: Option<String>,
    prompt: String,
    json_schema: Option<serde_json::Value>,
    temperature: Option<f32>,
    timeout: Duration,
    cancel: CancelFlag,
) -> Result<Vec<GenerateEvent>, SummarizeError> {
    if cancel.is_cancelled() {
        return Err(SummarizeError::Cancelled);
    }

    let base = api_base.trim_end_matches('/');
    let url = format!("{base}/v1beta/models/{model}:streamGenerateContent?alt=sse");

    let mut generation_config = serde_json::json!({ "temperature": temperature.unwrap_or(0.2) });
    if let Some(schema) = json_schema {
        generation_config["responseMimeType"] =
            serde_json::Value::String("application/json".into());
        generation_config["responseSchema"] = schema;
    }

    let mut body = serde_json::json!({
        "contents": [{ "role": "user", "parts": [{ "text": prompt }] }],
        "generationConfig": generation_config,
    });
    if let Some(sys) = &system {
        body["systemInstruction"] = serde_json::json!({ "parts": [{ "text": sys }] });
    }

    let resp = with_one_retry(|| {
        client
            .post(&url)
            .header(HEADER_API_KEY, &api_key)
            .json(&body)
            .timeout(timeout)
            .send()
    })
    .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        // The saved model is gone or was never right. Say so in words the person
        // can act on instead of handing them Google's sentence about API
        // versions and generation methods.
        if is_model_not_found(status, &text) {
            return Err(SummarizeError::ModelNotFound { model });
        }
        return Err(classify_status(status, &text));
    }

    drain_sse(resp, &cancel).await
}

impl Connector for GeminiConnector {
    fn provider(&self) -> Provider {
        Provider::Gemini
    }

    fn capabilities(&self) -> Caps {
        Caps {
            json_mode: true,
            streaming: true,
            context_chars: DEFAULT_CONTEXT_CHARS,
            can_list_models: true,
            leaves_machine: true,
        }
    }

    fn model(&self) -> Option<String> {
        Some(self.model.clone())
    }

    /// Only the models that can write a recap, flash first.
    ///
    /// Google's catalogue is every model they ship, so the raw list is mostly
    /// picture, video, voice and vector models. See [`recap_capable_models`] for
    /// the two gates and why the second one errs towards keeping a model.
    fn list_models<'a>(&'a self) -> BoxFuture<'a, Result<Vec<String>, SummarizeError>> {
        Box::pin(async move {
            let base = self.api_base.trim_end_matches('/');
            let url = format!("{base}/v1beta/models");
            let resp = with_one_retry(|| {
                self.client
                    .get(&url)
                    .header(HEADER_API_KEY, &self.api_key)
                    .timeout(PROBE_TIMEOUT)
                    .send()
            })
            .await?;
            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                return Err(classify_status(status, &text));
            }
            let parsed: ModelsResponse = resp
                .json()
                .await
                .map_err(|e| SummarizeError::Failed(e.to_string()))?;
            Ok(recap_capable_models(parsed.models))
        })
    }

    fn check<'a>(&'a self) -> BoxFuture<'a, Result<ProviderTestResult, SummarizeError>> {
        Box::pin(async move {
            let caps = self.capabilities();
            match self.list_models().await {
                Ok(models) => Ok(ProviderTestResult {
                    ok: true,
                    message: "Ready to write recaps.".into(),
                    caps,
                    models,
                    leaves_machine: true,
                }),
                Err(SummarizeError::Rejected(_)) => Ok(ProviderTestResult {
                    ok: false,
                    message: "That key wasn't accepted. Check it and try again.".into(),
                    caps,
                    models: Vec::new(),
                    leaves_machine: true,
                }),
                Err(SummarizeError::Unreachable(_)) => Ok(ProviderTestResult {
                    ok: false,
                    message: "Echo couldn't reach Google. Check your internet connection.".into(),
                    caps,
                    models: Vec::new(),
                    leaves_machine: true,
                }),
                Err(other) => Err(other),
            }
        })
    }

    fn generate<'a>(
        &'a self,
        req: GenerateRequest,
    ) -> BoxStream<'a, Result<GenerateEvent, SummarizeError>> {
        let client = self.client.clone();
        let api_base = self.api_base.clone();
        let api_key = self.api_key.clone();
        // The model is the connector's, resolved from Settings when it was
        // built. A single recap never picks its own — there is no per-recap
        // model picker, and a request that carried one would silently disagree
        // with the model recorded alongside the recap.
        let model = self.model.clone();
        let system = req.system.clone();
        let prompt = req.prompt.clone();
        let json_schema = req.json_schema.clone();
        let temperature = req.temperature;
        let timeout = req.timeout;
        let cancel = req.cancel.clone();

        let fut = run_generate_content(
            client,
            api_base,
            api_key,
            model,
            system,
            prompt,
            json_schema,
            temperature,
            timeout,
            cancel,
        );

        Box::pin(
            stream::once(fut)
                .map(|result| match result {
                    Ok(events) => stream::iter(events.into_iter().map(Ok).collect::<Vec<_>>()),
                    Err(e) => stream::iter(vec![Err(e)]),
                })
                .flatten(),
        )
    }
}

/// Hand-written so a stray `{:?}` in a log line can never print the key.
impl std::fmt::Debug for GeminiConnector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeminiConnector")
            .field("model", &self.model)
            .field("api_base", &self.api_base)
            .field("api_key", &"<redacted>")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn gemini_always_reports_that_text_leaves_the_machine() {
        let c = GeminiConnector::new("k", None);
        assert!(c.capabilities().leaves_machine);
        assert_eq!(c.model, DEFAULT_MODEL);
    }

    #[test]
    fn the_privacy_notice_names_google_and_says_text_is_sent() {
        let lower = PRIVACY_NOTICE.to_lowercase();
        assert!(lower.contains("google"));
        assert!(lower.contains("sent"));
    }

    #[test]
    fn the_key_is_not_exposed_by_the_connector() {
        let c = GeminiConnector::new("secret-key-value", None);
        assert_eq!(c.key_len(), "secret-key-value".len());
        let printed = format!("{c:?}");
        assert!(!printed.contains("secret-key-value"), "{printed}");
        assert!(printed.contains("<redacted>"));
    }

    #[test]
    fn a_blocked_finish_reason_is_recognised() {
        assert!(is_blocked_finish_reason("SAFETY"));
        assert!(is_blocked_finish_reason("RECITATION"));
        assert!(!is_blocked_finish_reason("STOP"));
        assert!(!is_blocked_finish_reason("MAX_TOKENS"));
    }

    // Every test below points the connector at a wiremock server, never at
    // Google: `with_api_base` is the only way the base URL ever changes.

    fn entry(name: &str, methods: &[&str]) -> ModelEntry {
        ModelEntry {
            name: name.to_string(),
            supported_generation_methods: methods.iter().map(|m| m.to_string()).collect(),
        }
    }

    // -- the supportedGenerationMethods gate -------------------------------

    #[test]
    fn only_models_that_answer_generate_content_get_through_the_method_gate() {
        assert!(supports_generate_content(&["generateContent".into()]));
        // Streaming is the call Echo actually makes.
        assert!(supports_generate_content(&["streamGenerateContent".into()]));
        assert!(supports_generate_content(&[
            "generateContent".into(),
            "countTokens".into()
        ]));
        assert!(!supports_generate_content(&["embedContent".into()]));
        assert!(!supports_generate_content(&["predict".into()]));
        assert!(!supports_generate_content(&[
            "predictLongRunning".into(),
            "countTokens".into()
        ]));
        assert!(!supports_generate_content(&["bidiGenerateContent".into()]));
    }

    #[test]
    fn a_model_that_reports_no_methods_at_all_is_kept_rather_than_hidden() {
        // An unexpected response shape must not empty the list.
        assert!(supports_generate_content(&[]));
        let kept = recap_capable_models(vec![entry("models/gemini-4-something", &[])]);
        assert_eq!(kept, vec!["gemini-4-something".to_string()]);
    }

    #[test]
    fn the_method_gate_drops_embedding_and_prediction_models() {
        let kept = recap_capable_models(vec![
            entry("models/gemini-3.7-flash", &["generateContent"]),
            entry("models/text-embedding-004", &["embedContent"]),
            entry("models/veo-3.0-generate", &["predictLongRunning"]),
        ]);
        assert_eq!(kept, vec!["gemini-3.7-flash".to_string()]);
    }

    // -- the name-family gate ---------------------------------------------

    #[test]
    fn the_name_gate_drops_picture_video_voice_and_vector_families() {
        for name in [
            "imagen-4.0-generate-001",
            "veo-3.0-fast-generate-preview",
            "gemini-2.5-flash-preview-tts",
            "gemini-2.5-pro-preview-tts",
            "gemini-2.5-flash-native-audio-dialog",
            "gemini-2.0-flash-exp-image-generation",
            "gemini-3-pro-image-preview",
            "gemini-2.5-flash-nano-banana",
            "nanobanana-pro",
            "text-embedding-004",
            "gemini-embedding-001",
            "embeddinggemma-300m",
            "aqa",
        ] {
            assert!(!is_text_model_name(name), "{name} should be hidden");
        }
    }

    #[test]
    fn the_name_gate_keeps_every_model_that_could_write_prose() {
        for name in [
            "gemini-3.7-flash",
            "gemini-3.7-flash-lite",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-flash-latest",
            "gemma-3-27b-it",
            "learnlm-2.0-flash-experimental",
            // Unfamiliar, so it stays: a stranger in the list beats a missing
            // model.
            "gemini-9-turbo-quux",
        ] {
            assert!(is_text_model_name(name), "{name} should be kept");
        }
    }

    #[test]
    fn a_short_fragment_never_knocks_out_a_model_by_accident() {
        // "tts", "veo" and "audio" are matched as whole name pieces, not as
        // letters found somewhere inside one.
        assert!(is_text_model_name("gemini-2.5-wattson-pro"));
        assert!(is_text_model_name("gemini-veolia-pro"));
        assert!(is_text_model_name("gemini-2.5-audiobook-pro"));
    }

    // -- ordering ---------------------------------------------------------

    #[test]
    fn the_flash_family_comes_first_and_the_rest_keep_their_order() {
        let kept = recap_capable_models(vec![
            entry("models/gemini-2.5-pro", &["generateContent"]),
            entry("models/gemini-3.7-flash", &["generateContent"]),
            entry("models/gemma-3-27b-it", &["generateContent"]),
            entry("models/gemini-3.7-flash-lite", &["generateContent"]),
        ]);
        assert_eq!(
            kept,
            vec![
                "gemini-3.7-flash".to_string(),
                "gemini-3.7-flash-lite".to_string(),
                "gemini-2.5-pro".to_string(),
                "gemma-3-27b-it".to_string(),
            ]
        );
    }

    // -- the whole call, against a mock catalogue ---------------------------

    #[tokio::test]
    async fn list_models_returns_only_recap_models_flash_first_without_the_prefix() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1beta/models"))
            .and(header(HEADER_API_KEY, "test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "models": [
                    {"name": "models/gemini-2.5-pro", "supportedGenerationMethods": ["generateContent", "countTokens"]},
                    {"name": "models/imagen-4.0-generate-001", "supportedGenerationMethods": ["predict"]},
                    {"name": "models/gemini-3.7-flash", "supportedGenerationMethods": ["generateContent"]},
                    {"name": "models/veo-3.0-generate-preview", "supportedGenerationMethods": ["predictLongRunning"]},
                    {"name": "models/gemini-2.5-flash-preview-tts", "supportedGenerationMethods": ["generateContent"]},
                    {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
                    {"name": "models/gemini-2.5-flash-native-audio-dialog", "supportedGenerationMethods": ["bidiGenerateContent"]},
                    {"name": "models/gemini-3-pro-image-preview", "supportedGenerationMethods": ["generateContent"]},
                    {"name": "models/aqa", "supportedGenerationMethods": ["generateAnswer"]},
                ]
            })))
            .mount(&server)
            .await;

        let connector = GeminiConnector::new("test-key", None).with_api_base(server.uri());
        let models = connector.list_models().await.unwrap();
        assert_eq!(
            models,
            vec!["gemini-3.7-flash".to_string(), "gemini-2.5-pro".to_string()]
        );
    }

    #[tokio::test]
    async fn check_reports_the_filtered_list_behind_the_this_works_button() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1beta/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "models": [
                    {"name": "models/gemini-3.7-flash", "supportedGenerationMethods": ["generateContent"]},
                    {"name": "models/imagen-4.0-generate-001", "supportedGenerationMethods": ["predict"]},
                ]
            })))
            .mount(&server)
            .await;

        let connector = GeminiConnector::new("test-key", None).with_api_base(server.uri());
        let result = connector.check().await.unwrap();
        assert!(result.ok);
        assert_eq!(result.models, vec!["gemini-3.7-flash".to_string()]);
        assert!(result.leaves_machine);
    }

    // -- the default model -------------------------------------------------

    #[test]
    fn nothing_configured_means_the_flash_default() {
        assert_eq!(DEFAULT_MODEL, "gemini-3.7-flash");
        assert_eq!(GeminiConnector::new("k", None).model, DEFAULT_MODEL);
        assert_eq!(
            Connector::model(&GeminiConnector::new("k", None)).as_deref(),
            Some(DEFAULT_MODEL)
        );
    }

    #[test]
    fn a_blank_or_padded_model_setting_is_treated_as_unset() {
        assert_eq!(
            GeminiConnector::new("k", Some("   ".to_string())).model,
            DEFAULT_MODEL
        );
        assert_eq!(
            GeminiConnector::new("k", Some("".to_string())).model,
            DEFAULT_MODEL
        );
        assert_eq!(
            GeminiConnector::new("k", Some("  gemini-2.5-pro ".to_string())).model,
            "gemini-2.5-pro"
        );
    }

    #[tokio::test]
    async fn with_no_model_configured_the_default_is_the_one_that_gets_called() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(format!(
                "/v1beta/models/{DEFAULT_MODEL}:streamGenerateContent"
            )))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"recap\"}]},\"finishReason\":\"STOP\"}]}\n\n",
                "text/event-stream",
            ))
            .mount(&server)
            .await;

        let connector = GeminiConnector::new("test-key", None).with_api_base(server.uri());
        let events: Vec<_> = connector
            .generate(GenerateRequest {
                prompt: "write a recap".into(),
                ..Default::default()
            })
            .collect()
            .await;
        assert!(
            matches!(events.last(), Some(Ok(GenerateEvent::Done { text })) if text == "recap"),
            "{events:?}"
        );
    }

    // -- a model Google does not know --------------------------------------

    #[test]
    fn model_not_found_is_told_apart_from_every_other_refusal() {
        let not_found = "{\"error\":{\"code\":404,\"message\":\"models/gemini-nope is not found \
                         for API version v1beta, or is not supported for \
                         streamGenerateContent.\",\"status\":\"NOT_FOUND\"}}";
        assert!(is_model_not_found(
            reqwest::StatusCode::NOT_FOUND,
            not_found
        ));
        assert!(is_model_not_found(
            reqwest::StatusCode::BAD_REQUEST,
            not_found
        ));
        // A refused key, a used-up quota and a server having a bad day are all
        // somebody else's problem to explain.
        assert!(!is_model_not_found(
            reqwest::StatusCode::FORBIDDEN,
            not_found
        ));
        assert!(!is_model_not_found(
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            "{\"error\":{\"message\":\"quota\"}}"
        ));
        assert!(!is_model_not_found(
            reqwest::StatusCode::INTERNAL_SERVER_ERROR,
            not_found
        ));
        assert!(!is_model_not_found(
            reqwest::StatusCode::BAD_REQUEST,
            "{\"error\":{\"message\":\"invalid temperature\"}}"
        ));
    }

    #[tokio::test]
    async fn a_model_google_does_not_know_asks_the_person_to_pick_one_in_settings() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1beta/models/gemini-retired:streamGenerateContent"))
            .respond_with(ResponseTemplate::new(404).set_body_json(serde_json::json!({
                "error": {
                    "code": 404,
                    "message": "models/gemini-retired is not found for API version v1beta, or is not supported for streamGenerateContent.",
                    "status": "NOT_FOUND"
                }
            })))
            .mount(&server)
            .await;

        let connector = GeminiConnector::new("test-key", Some("gemini-retired".into()))
            .with_api_base(server.uri());
        let events: Vec<_> = connector
            .generate(GenerateRequest {
                prompt: "write a recap".into(),
                ..Default::default()
            })
            .collect()
            .await;

        let err = match events.into_iter().next() {
            Some(Err(e)) => e,
            other => panic!("expected one error, got {other:?}"),
        };
        assert!(
            matches!(&err, SummarizeError::ModelNotFound { model } if model == "gemini-retired"),
            "{err:?}"
        );
        // Plain words, and it names the place to fix it.
        let message = err.to_string();
        assert!(message.contains("Settings"), "{message}");
        assert!(message.contains("gemini-retired"), "{message}");
        for jargon in ["404", "v1beta", "streamGenerateContent", "NOT_FOUND"] {
            assert!(!message.contains(jargon), "{message} leaked {jargon}");
        }
    }

    #[tokio::test]
    async fn a_refused_key_is_still_reported_as_a_refusal_not_a_missing_model() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(format!(
                "/v1beta/models/{DEFAULT_MODEL}:streamGenerateContent"
            )))
            .respond_with(ResponseTemplate::new(403).set_body_string(
                "{\"error\":{\"message\":\"API key not valid\",\"status\":\"PERMISSION_DENIED\"}}",
            ))
            .mount(&server)
            .await;

        let connector = GeminiConnector::new("bad-key", None).with_api_base(server.uri());
        let events: Vec<_> = connector
            .generate(GenerateRequest {
                prompt: "write a recap".into(),
                ..Default::default()
            })
            .collect()
            .await;
        assert!(
            matches!(events.first(), Some(Err(SummarizeError::Rejected(_)))),
            "{events:?}"
        );
    }

    #[test]
    fn classify_status_maps_auth_and_quota_errors() {
        assert!(matches!(
            classify_status(reqwest::StatusCode::UNAUTHORIZED, "{}"),
            SummarizeError::Rejected(_)
        ));
        assert!(matches!(
            classify_status(reqwest::StatusCode::FORBIDDEN, "{}"),
            SummarizeError::Rejected(_)
        ));
        assert!(matches!(
            classify_status(reqwest::StatusCode::TOO_MANY_REQUESTS, "{}"),
            SummarizeError::Rejected(_)
        ));
        assert!(matches!(
            classify_status(reqwest::StatusCode::INTERNAL_SERVER_ERROR, "{}"),
            SummarizeError::Failed(_)
        ));
    }

    #[tokio::test]
    async fn drain_sse_accumulates_text_across_events_and_emits_done() {
        let server = MockServer::start().await;
        let body = concat!(
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello\"}]}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\", world\"}],\"role\":\"model\"},\"finishReason\":\"STOP\"}]}\n\n",
        );
        Mock::given(method("GET"))
            .and(path("/stream"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(body.as_bytes(), "text/event-stream"),
            )
            .mount(&server)
            .await;

        let resp = reqwest::Client::new()
            .get(format!("{}/stream", server.uri()))
            .send()
            .await
            .unwrap();
        let events = drain_sse(resp, &CancelFlag::new()).await.unwrap();
        let texts: Vec<String> = events
            .iter()
            .filter_map(|e| match e {
                GenerateEvent::Text(t) => Some(t.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["Hello".to_string(), ", world".to_string()]);
        assert!(
            matches!(events.last(), Some(GenerateEvent::Done { text }) if text == "Hello, world")
        );
    }

    #[tokio::test]
    async fn drain_sse_maps_a_safety_block_to_rejected() {
        let server = MockServer::start().await;
        let body = "data: {\"candidates\":[{\"finishReason\":\"SAFETY\"}]}\n\n";
        Mock::given(method("GET"))
            .and(path("/stream"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(body.as_bytes(), "text/event-stream"),
            )
            .mount(&server)
            .await;

        let resp = reqwest::Client::new()
            .get(format!("{}/stream", server.uri()))
            .send()
            .await
            .unwrap();
        let err = drain_sse(resp, &CancelFlag::new()).await.unwrap_err();
        assert!(matches!(err, SummarizeError::Rejected(_)));
    }

    #[tokio::test]
    async fn drain_sse_maps_a_blocked_prompt_to_rejected() {
        let server = MockServer::start().await;
        let body = "data: {\"candidates\":[],\"promptFeedback\":{\"blockReason\":\"SAFETY\"}}\n\n";
        Mock::given(method("GET"))
            .and(path("/stream"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(body.as_bytes(), "text/event-stream"),
            )
            .mount(&server)
            .await;

        let resp = reqwest::Client::new()
            .get(format!("{}/stream", server.uri()))
            .send()
            .await
            .unwrap();
        let err = drain_sse(resp, &CancelFlag::new()).await.unwrap_err();
        assert!(matches!(err, SummarizeError::Rejected(_)));
    }

    #[tokio::test]
    async fn drain_sse_respects_cancellation() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/stream"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hi\"}]}}]}\n\n",
                "text/event-stream",
            ))
            .mount(&server)
            .await;

        let resp = reqwest::Client::new()
            .get(format!("{}/stream", server.uri()))
            .send()
            .await
            .unwrap();
        let cancel = CancelFlag::new();
        cancel.cancel();
        let err = drain_sse(resp, &cancel).await.unwrap_err();
        assert!(matches!(err, SummarizeError::Cancelled));
    }
}
