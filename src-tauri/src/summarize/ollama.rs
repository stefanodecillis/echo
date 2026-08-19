//! "On this computer": a local Ollama server.
//!
//! IMPLEMENTED-BY: summarize agent (M5).
//!
//! Rules (DESIGN §3 Connectors):
//! * default address is `http://127.0.0.1:11434`, loopback only
//! * a non-loopback address is allowed only after the person agrees to
//!   "this leaves your machine"; [`super::is_loopback_url`] is the check
//! * detected automatically during onboarding, so most people never see a
//!   setting
//! * small models need help: chunk the transcript, ask for strict JSON, and
//!   offer "write it again with something else"
//! * `context_chars` comes from the model's own reported context where the
//!   server exposes it, and stays conservative otherwise

use std::time::Duration;

use futures::future::BoxFuture;
use futures::stream::{self, BoxStream, StreamExt};
use serde::Deserialize;

use crate::summarize::http_util::{classify_reqwest_error, truncate, with_one_retry};
use crate::summarize::{CancelFlag, Connector, GenerateEvent, GenerateRequest, SummarizeError};
use crate::types::{Caps, Provider, ProviderTestResult};

/// Address used when nothing is configured.
pub const DEFAULT_BASE_URL: &str = "http://127.0.0.1:11434";

/// Assumed usable input size when the server does not say. Deliberately low:
/// a chunk that overflows produces a worse recap than one more chunk.
pub const CONSERVATIVE_CONTEXT_CHARS: u32 = 8_000;

/// How long `detect`/`list_models` wait before deciding nothing is there.
const PROBE_TIMEOUT: Duration = Duration::from_secs(3);

#[derive(Debug, Clone)]
pub struct OllamaConnector {
    pub base_url: String,
    pub model: Option<String>,
    client: reqwest::Client,
}

impl Default for OllamaConnector {
    fn default() -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            model: None,
            client: reqwest::Client::new(),
        }
    }
}

impl OllamaConnector {
    pub fn new(base_url: impl Into<String>, model: Option<String>) -> Self {
        Self {
            base_url: base_url.into(),
            model,
            client: reqwest::Client::new(),
        }
    }

    fn tags_url(&self) -> String {
        format!("{}/api/tags", self.base_url.trim_end_matches('/'))
    }

    /// Is a server answering at this address right now? Used by onboarding to
    /// offer local recaps without asking any questions.
    pub async fn detect(base_url: &str) -> bool {
        let url = format!("{}/api/tags", base_url.trim_end_matches('/'));
        reqwest::Client::new()
            .get(&url)
            .timeout(PROBE_TIMEOUT)
            .send()
            .await
            .map(|resp| resp.status().is_success())
            .unwrap_or(false)
    }
}

#[derive(Debug, Deserialize)]
struct TagsResponse {
    #[serde(default)]
    models: Vec<TagsModel>,
}

#[derive(Debug, Deserialize)]
struct TagsModel {
    name: String,
}

#[derive(Debug, Deserialize)]
struct ChatChunk {
    #[serde(default)]
    message: Option<ChatMessage>,
    #[serde(default)]
    done: bool,
    /// Ollama sometimes answers with a single `{"error": "..."}` line instead
    /// of a chat chunk, e.g. when the model name does not exist.
    #[serde(default)]
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    #[serde(default)]
    content: String,
}

fn classify_status(status: reqwest::StatusCode, body: &str) -> SummarizeError {
    if status.as_u16() == 404 {
        SummarizeError::Rejected(format!("nothing there yet: {}", truncate(body, 200)))
    } else {
        SummarizeError::Failed(format!("http {status}: {}", truncate(body, 200)))
    }
}

/// Read one line-delimited JSON streaming response into the events it
/// describes. A model-not-found or similar failure usually shows up as a
/// single `{"error": ...}` line rather than an HTTP status, so that has to be
/// checked per line, not just once up front.
async fn drain_ndjson(
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
                while let Some(pos) = buf.find('\n') {
                    let line = buf[..pos].trim().to_string();
                    buf.drain(..=pos);
                    if line.is_empty() {
                        continue;
                    }
                    let chunk: ChatChunk = match serde_json::from_str(&line) {
                        Ok(c) => c,
                        Err(_) => continue, // ignore anything that isn't a chunk we recognise
                    };
                    if let Some(err) = chunk.error {
                        return Err(SummarizeError::Rejected(err));
                    }
                    if let Some(message) = chunk.message {
                        if !message.content.is_empty() {
                            full.push_str(&message.content);
                            events.push(GenerateEvent::Text(message.content));
                        }
                    }
                    if chunk.done {
                        events.push(GenerateEvent::Done { text: full });
                        return Ok(events);
                    }
                }
            }
            Ok(None) => break,
            Err(e) => return Err(classify_reqwest_error(&e)),
        }
    }

    if full.is_empty() {
        return Err(SummarizeError::Failed(
            "the reply ended before it said it was done".into(),
        ));
    }
    events.push(GenerateEvent::Done { text: full });
    Ok(events)
}

#[allow(clippy::too_many_arguments)]
async fn run_chat(
    client: reqwest::Client,
    base_url: String,
    model: Option<String>,
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
    let model = model.ok_or_else(|| {
        SummarizeError::Failed("choose which local model to use in Settings first".into())
    })?;

    let mut messages = Vec::new();
    if let Some(sys) = &system {
        messages.push(serde_json::json!({"role": "system", "content": sys}));
    }
    messages.push(serde_json::json!({"role": "user", "content": prompt}));
    let mut body = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": true,
        "options": { "temperature": temperature.unwrap_or(0.2) },
    });
    if let Some(schema) = json_schema {
        body["format"] = schema;
    }

    let url = format!("{}/api/chat", base_url.trim_end_matches('/'));
    let resp = with_one_retry(|| client.post(&url).json(&body).timeout(timeout).send()).await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(classify_status(status, &text));
    }

    drain_ndjson(resp, &cancel).await
}

impl Connector for OllamaConnector {
    fn provider(&self) -> Provider {
        Provider::OnThisComputer
    }

    fn capabilities(&self) -> Caps {
        Caps {
            json_mode: true,
            streaming: true,
            context_chars: CONSERVATIVE_CONTEXT_CHARS,
            can_list_models: true,
            leaves_machine: !crate::summarize::is_loopback_url(&self.base_url),
        }
    }

    fn list_models<'a>(&'a self) -> BoxFuture<'a, Result<Vec<String>, SummarizeError>> {
        Box::pin(async move {
            let resp = with_one_retry(|| {
                self.client
                    .get(self.tags_url())
                    .timeout(PROBE_TIMEOUT)
                    .send()
            })
            .await?;
            if !resp.status().is_success() {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                return Err(classify_status(status, &text));
            }
            let parsed: TagsResponse = resp
                .json()
                .await
                .map_err(|e| SummarizeError::Failed(e.to_string()))?;
            Ok(parsed.models.into_iter().map(|m| m.name).collect())
        })
    }

    fn check<'a>(&'a self) -> BoxFuture<'a, Result<ProviderTestResult, SummarizeError>> {
        Box::pin(async move {
            let caps = self.capabilities();
            let leaves_machine = caps.leaves_machine;
            match self.list_models().await {
                Ok(models) => Ok(ProviderTestResult {
                    ok: true,
                    message: "Ready to write recaps.".into(),
                    caps,
                    models,
                    leaves_machine,
                }),
                Err(SummarizeError::Unreachable(_)) => Ok(ProviderTestResult {
                    ok: false,
                    message: "Echo couldn't reach it. Make sure it's running.".into(),
                    caps,
                    models: Vec::new(),
                    leaves_machine,
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
        let base_url = self.base_url.clone();
        let model = req.model.clone().or_else(|| self.model.clone());
        let system = req.system.clone();
        let prompt = req.prompt.clone();
        let json_schema = req.json_schema.clone();
        let temperature = req.temperature;
        let timeout = req.timeout;
        let cancel = req.cancel.clone();

        let fut = run_chat(
            client,
            base_url,
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

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn the_default_address_is_loopback() {
        let c = OllamaConnector::default();
        assert!(crate::summarize::is_loopback_url(&c.base_url));
        assert!(!c.capabilities().leaves_machine);
    }

    #[test]
    fn a_remote_address_is_flagged_as_leaving_the_machine() {
        let c = OllamaConnector::new("http://192.168.1.5:11434", None);
        assert!(c.capabilities().leaves_machine);
    }

    #[tokio::test]
    async fn detect_is_false_with_nothing_listening() {
        // Port 1 is reserved and nothing will ever answer there quickly.
        assert!(!OllamaConnector::detect("http://127.0.0.1:1").await);
    }

    #[tokio::test]
    async fn detect_is_true_when_tags_answers() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"models": []})),
            )
            .mount(&server)
            .await;
        assert!(OllamaConnector::detect(&server.uri()).await);
    }

    #[tokio::test]
    async fn list_models_parses_the_tags_response() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "models": [{"name": "llama3.1:8b"}, {"name": "qwen2.5:7b"}]
            })))
            .mount(&server)
            .await;
        let conn = OllamaConnector::new(server.uri(), None);
        let models = conn.list_models().await.unwrap();
        assert_eq!(
            models,
            vec!["llama3.1:8b".to_string(), "qwen2.5:7b".to_string()]
        );
    }

    #[tokio::test]
    async fn list_models_reports_not_running_distinctly() {
        let conn = OllamaConnector::new("http://127.0.0.1:1", None);
        let err = conn.list_models().await.unwrap_err();
        assert!(matches!(err, SummarizeError::Unreachable(_)));
    }

    #[tokio::test]
    async fn check_turns_not_running_into_a_friendly_not_ok_result() {
        let conn = OllamaConnector::new("http://127.0.0.1:1", None);
        let result = conn.check().await.unwrap();
        assert!(!result.ok);
        assert!(result.message.to_lowercase().contains("running") || !result.message.is_empty());
    }

    #[tokio::test]
    async fn check_reports_ready_when_the_server_answers() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"models": []})),
            )
            .mount(&server)
            .await;
        let conn = OllamaConnector::new(server.uri(), None);
        let result = conn.check().await.unwrap();
        assert!(result.ok);
    }

    #[tokio::test]
    async fn generate_collects_a_streamed_chat_reply() {
        let server = MockServer::start().await;
        let body = concat!(
            "{\"message\":{\"role\":\"assistant\",\"content\":\"Hello\"},\"done\":false}\n",
            "{\"message\":{\"role\":\"assistant\",\"content\":\", world\"},\"done\":false}\n",
            "{\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true}\n",
        );
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/x-ndjson")
                    .set_body_raw(body.as_bytes(), "application/x-ndjson"),
            )
            .mount(&server)
            .await;

        let conn = OllamaConnector::new(server.uri(), Some("llama3.1:8b".to_string()));
        let mut stream = conn.generate(GenerateRequest {
            prompt: "hi".into(),
            ..Default::default()
        });
        let mut texts = Vec::new();
        let mut final_text = None;
        while let Some(event) = stream.next().await {
            match event.unwrap() {
                GenerateEvent::Text(t) => texts.push(t),
                GenerateEvent::Done { text } => final_text = Some(text),
            }
        }
        assert_eq!(texts, vec!["Hello".to_string(), ", world".to_string()]);
        assert_eq!(final_text.as_deref(), Some("Hello, world"));
    }

    #[tokio::test]
    async fn generate_maps_a_streamed_error_line_to_rejected() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                "{\"error\":\"model 'ghost' not found\"}\n",
                "application/x-ndjson",
            ))
            .mount(&server)
            .await;

        let conn = OllamaConnector::new(server.uri(), Some("ghost".to_string()));
        let mut stream = conn.generate(GenerateRequest {
            prompt: "hi".into(),
            ..Default::default()
        });
        let first = stream.next().await.unwrap();
        assert!(matches!(first, Err(SummarizeError::Rejected(_))));
    }

    #[tokio::test]
    async fn generate_without_a_model_fails_clearly_instead_of_guessing_one() {
        let conn = OllamaConnector::new("http://127.0.0.1:1", None);
        let mut stream = conn.generate(GenerateRequest {
            prompt: "hi".into(),
            ..Default::default()
        });
        let first = stream.next().await.unwrap();
        assert!(first.is_err());
    }
}
