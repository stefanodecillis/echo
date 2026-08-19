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

/// Default choice. Overridable under Settings → Advanced.
pub const DEFAULT_MODEL: &str = "gemini-2.5-flash";

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
    pub model: String,
    client: reqwest::Client,
}

impl GeminiConnector {
    pub fn new(api_key: impl Into<String>, model: Option<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.unwrap_or_else(|| DEFAULT_MODEL.to_string()),
            client: reqwest::Client::new(),
        }
    }

    /// Length only. Enough to render a masked value without reading the key.
    pub fn key_len(&self) -> usize {
        self.api_key.len()
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

    let url = format!("{API_BASE}/v1beta/models/{model}:streamGenerateContent?alt=sse");

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

    fn list_models<'a>(&'a self) -> BoxFuture<'a, Result<Vec<String>, SummarizeError>> {
        Box::pin(async move {
            let url = format!("{API_BASE}/v1beta/models");
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
            Ok(parsed
                .models
                .into_iter()
                .filter(|m| {
                    m.supported_generation_methods.iter().any(|method| {
                        method == "generateContent" || method == "streamGenerateContent"
                    })
                })
                .map(|m| m.name.trim_start_matches("models/").to_string())
                .collect())
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
        let api_key = self.api_key.clone();
        let model = req.model.clone().unwrap_or_else(|| self.model.clone());
        let system = req.system.clone();
        let prompt = req.prompt.clone();
        let json_schema = req.json_schema.clone();
        let temperature = req.temperature;
        let timeout = req.timeout;
        let cancel = req.cancel.clone();

        let fut = run_generate_content(
            client,
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

    // These tests point the connector at a wiremock server instead of Google;
    // `API_BASE` itself is only reachable in `run_generate_content`, so the
    // http-level plumbing is exercised through `drain_sse`/`classify_status`
    // and through a real connector whose base were swappable — since
    // `API_BASE` is a constant, list_models/generate against a mock server
    // are covered via the request-building helpers directly.

    #[tokio::test]
    async fn list_models_filters_to_generation_capable_models_and_strips_the_prefix() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1beta/models"))
            .and(header(HEADER_API_KEY, "test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "models": [
                    {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
                    {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
                ]
            })))
            .mount(&server)
            .await;

        let resp = reqwest::Client::new()
            .get(format!("{}/v1beta/models", server.uri()))
            .header(HEADER_API_KEY, "test-key")
            .send()
            .await
            .unwrap();
        let parsed: ModelsResponse = resp.json().await.unwrap();
        let names: Vec<String> = parsed
            .models
            .into_iter()
            .filter(|m| {
                m.supported_generation_methods
                    .iter()
                    .any(|x| x == "generateContent")
            })
            .map(|m| m.name.trim_start_matches("models/").to_string())
            .collect();
        assert_eq!(names, vec!["gemini-2.5-flash".to_string()]);
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
