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
    /// Counts only, never content. Logged so a field failure can be read off
    /// the log line alone.
    #[serde(rename = "usageMetadata", default)]
    usage: Option<UsageMetadata>,
    #[serde(rename = "modelVersion", default)]
    model_version: Option<String>,
    #[serde(rename = "responseId", default)]
    response_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PromptFeedback {
    #[serde(rename = "blockReason", default)]
    block_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct UsageMetadata {
    #[serde(rename = "promptTokenCount", default)]
    prompt_tokens: Option<u64>,
    #[serde(rename = "candidatesTokenCount", default)]
    candidate_tokens: Option<u64>,
    #[serde(rename = "thoughtsTokenCount", default)]
    thought_tokens: Option<u64>,
    #[serde(rename = "totalTokenCount", default)]
    total_tokens: Option<u64>,
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
    /// Recent models can return their own reasoning as a part marked this way.
    /// It is not recap text and must never be pasted into one, so it is counted
    /// and dropped.
    #[serde(default)]
    thought: bool,
}

/// Families that exist to make pictures, video, speech or vectors. None of them
/// can write a recap, and every one of them shows up in the same catalogue as
/// the ones that can.
///
/// Matched against the name's hyphen/dot-separated pieces rather than as a
/// substring, so a fragment as short as `tts` can never knock out a model whose
/// name merely happens to contain those letters.
const NON_TEXT_NAME_PARTS: &[&str] = &[
    "imagen", "veo", "tts", "audio", "aqa", "embed", "image", "images", "video",
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
    let collapsed: String = lower
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect();
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

/// Does this refusal read as being about the key itself?
///
/// The status alone cannot decide it: Google's own answer to a bad key is a
/// `400 INVALID_ARGUMENT` saying *"API key not valid"*, not a 401. So the two
/// auth statuses count on their own, and any other 4xx counts only when the
/// body names the credential.
fn is_about_the_key(status: u16, body: &str) -> bool {
    if matches!(status, 401 | 403) {
        return true;
    }
    let lower = body.to_ascii_lowercase();
    lower.contains("api key")
        || lower.contains("api_key")
        || lower.contains("apikey")
        || lower.contains("credential")
        || lower.contains("unauthenticated")
        || lower.contains("unauthorized")
}

fn classify_status(status: reqwest::StatusCode, body: &str) -> SummarizeError {
    let code = status.as_u16();
    match code {
        // Not a bad key: the key worked and the allowance behind it is spent.
        // Told apart because the two need opposite advice — re-check the key,
        // versus wait or raise the limit.
        429 => SummarizeError::QuotaExhausted,
        400..=499 if is_about_the_key(code, body) => {
            SummarizeError::Rejected("the saved key was refused".into())
        }
        // Everything else in the 4xx range is Google turning *this request*
        // down — a schema keyword it does not know, a shape it does not accept,
        // a feature this model does not have. Folding these in with a refused
        // key sent people off to replace a key that was working perfectly
        // (review of 2026-08-20, finding 5).
        400..=499 => SummarizeError::RequestRefused {
            status: code,
            detail: truncate(body, 200),
        },
        _ => SummarizeError::Failed(format!("http {status}: {}", truncate(body, 200))),
    }
}

/// `SAFETY`, `RECITATION` and friends mean the model refused to answer; only
/// `STOP` (and `MAX_TOKENS`, a truncation, not a refusal) mean it actually
/// wrote something. An unspecified reason is not a refusal either — it is a
/// shape we do not recognise, and calling that a block would blame the person's
/// meeting for our own blind spot.
fn is_blocked_finish_reason(reason: &str) -> bool {
    !matches!(
        reason,
        "STOP" | "MAX_TOKENS" | "FINISH_REASON_UNSPECIFIED" | ""
    )
}

/// Where the one structured line per Gemini call goes. Counters only: never
/// prompt, transcript or recap text.
const TELEMETRY_TARGET: &str = "echo::recap";

/// A UTF-8 byte-order mark. Some proxies prepend one to a text/event-stream
/// body; left in place it becomes part of the first field name and every event
/// in the response fails to parse.
const BOM: [u8; 3] = [0xEF, 0xBB, 0xBF];

/// Everything worth knowing about one Gemini reply, and nothing that could be
/// content.
///
/// This exists because the field failure it was written for ("the reply came
/// back empty", three times running) left no evidence at all: the old parser
/// dropped every event it could not deserialize and then reported the same
/// sentence whether Google had sent nothing, sent a refusal, or sent a perfectly
/// good reply in a frame shape Echo could not split. Each of those needs a
/// different fix, so each of them now has a number next to it in the log.
#[derive(Debug, Default)]
struct CallTally {
    /// Event blocks that carried a `data:` payload.
    events: usize,
    /// Blocks that carried none — comments, keep-alives.
    keepalives: usize,
    /// Payloads that deserialized into a reply chunk.
    parsed: usize,
    /// Payloads that did not. The number that tells a framing bug from an
    /// empty answer.
    parse_failures: usize,
    /// The shape of the first deserialization failure: which kind of failure it
    /// was and where in the payload it happened, and nothing else.
    ///
    /// Deliberately **not** `serde_json::Error::to_string()`. Serde's message
    /// for a type mismatch quotes the offending value — `invalid type: string
    /// "…", expected a sequence` — so the "counters only" promise this whole
    /// struct is built on was not one the old field could keep
    /// (review of 2026-08-20, finding 4).
    first_parse_shape: Option<String>,
    candidates: usize,
    parts: usize,
    text_parts: usize,
    thought_parts: usize,
    /// Length of the assembled reply. A length is not content.
    text_chars: usize,
    finish_reasons: Vec<String>,
    block_reason: Option<String>,
    blocking_finish_reason: Option<String>,
    prompt_tokens: Option<u64>,
    candidate_tokens: Option<u64>,
    thought_tokens: Option<u64>,
    total_tokens: Option<u64>,
    model_version: Option<String>,
    response_id: Option<String>,
    /// Bytes still in the buffer when the body ended — an event that never got
    /// its blank line. Parsed anyway; counted so a truncated stream shows up.
    leftover_bytes: usize,
    saw_crlf: bool,
    saw_lf: bool,
    saw_cr: bool,
    bom: bool,
    done_marker: bool,
}

/// A deserialization failure, described without quoting any of the document.
///
/// Serde's own `Display` is not safe to log here: for a type mismatch it prints
/// the value it did not like, which for a Gemini reply is a piece of somebody's
/// meeting. The category and the position say everything a fix needs — "the
/// payload was valid JSON of the wrong shape, twelve lines in" — and carry
/// nothing that could be content.
fn parse_failure_shape(error: &serde_json::Error) -> String {
    use serde_json::error::Category;
    let kind = match error.classify() {
        Category::Io => "io",
        Category::Syntax => "not json",
        Category::Data => "wrong shape",
        Category::Eof => "cut short",
    };
    format!("{kind} at line {} column {}", error.line(), error.column())
}

/// Note which line endings some part of the reply used. The one field that says
/// straight out whether a framing bug is what went wrong.
fn note_line_endings(bytes: &[u8], tally: &mut CallTally) {
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\r' if bytes.get(i + 1) == Some(&b'\n') => {
                tally.saw_crlf = true;
                i += 2;
            }
            b'\r' => {
                tally.saw_cr = true;
                i += 1;
            }
            b'\n' => {
                tally.saw_lf = true;
                i += 1;
            }
            _ => i += 1,
        }
    }
}

impl CallTally {
    fn line_endings(&self) -> &'static str {
        match (self.saw_crlf, self.saw_lf, self.saw_cr) {
            (true, false, false) => "crlf",
            (false, true, false) => "lf",
            (false, false, true) => "cr",
            (false, false, false) => "none",
            _ => "mixed",
        }
    }

    fn note_finish_reason(&mut self, reason: &str) {
        if reason.is_empty() {
            return;
        }
        if !self.finish_reasons.iter().any(|seen| seen == reason) {
            self.finish_reasons.push(reason.to_string());
        }
        if is_blocked_finish_reason(reason) && self.blocking_finish_reason.is_none() {
            self.blocking_finish_reason = Some(reason.to_string());
        }
    }

    /// One line per call, always, whatever the outcome. The next field failure
    /// has to be diagnosable from this alone.
    fn log(&self, model: &str) {
        tracing::info!(
            target: TELEMETRY_TARGET,
            model = %model,
            events = self.events,
            keepalives = self.keepalives,
            parsed = self.parsed,
            parse_failures = self.parse_failures,
            candidates = self.candidates,
            parts = self.parts,
            text_parts = self.text_parts,
            thought_parts = self.thought_parts,
            text_chars = self.text_chars,
            finish_reasons = %self.finish_reasons.join("|"),
            block_reason = self.block_reason.as_deref().unwrap_or("-"),
            prompt_tokens = self.prompt_tokens.unwrap_or_default(),
            candidate_tokens = self.candidate_tokens.unwrap_or_default(),
            thought_tokens = self.thought_tokens.unwrap_or_default(),
            total_tokens = self.total_tokens.unwrap_or_default(),
            model_version = self.model_version.as_deref().unwrap_or("-"),
            response_id = self.response_id.as_deref().unwrap_or("-"),
            leftover_bytes = self.leftover_bytes,
            line_endings = self.line_endings(),
            bom = self.bom,
            done_marker = self.done_marker,
            "gemini reply"
        );
        if self.parse_failures > 0 {
            tracing::warn!(
                target: TELEMETRY_TARGET,
                model = %model,
                events = self.events,
                parsed = self.parsed,
                parse_failures = self.parse_failures,
                line_endings = self.line_endings(),
                bom = self.bom,
                reason = self.first_parse_shape.as_deref().unwrap_or("-"),
                "some of Google's reply could not be read"
            );
        }
    }
}

/// How long the line terminator at `at` is, or `None` if there isn't one there.
///
/// A server-sent-events line ends with CRLF, LF **or** a bare CR. A trailing CR
/// at the very end of what we have received so far is deliberately reported as
/// "not a terminator": it may be the first half of a CRLF still in flight, and
/// splitting there would cut an event in two.
fn terminator_len(buf: &[u8], at: usize) -> Option<usize> {
    match buf.get(at)? {
        b'\r' => match buf.get(at + 1) {
            Some(b'\n') => Some(2),
            // Nothing after it yet: wait for the other half.
            None => None,
            _ => Some(1),
        },
        b'\n' => Some(1),
        _ => None,
    }
}

/// The next complete event in `buf`: how many bytes belong to the event, and how
/// many to drop from the front of the buffer (the event plus its blank line).
///
/// Events are separated by a blank line, and each of the two line terminators
/// may independently be CRLF, LF or CR — Google's streaming responses have been
/// seen using `\r\n\r\n` where the same endpoint documents `\n\n`. Splitting on
/// `\n\n` alone silently reads a whole CRLF response as one unterminated event,
/// which is exactly the "empty reply" this function was rewritten for.
fn next_event(buf: &[u8]) -> Option<(usize, usize)> {
    let mut i = 0;
    while i < buf.len() {
        match terminator_len(buf, i) {
            Some(first) => match terminator_len(buf, i + first) {
                Some(second) => return Some((i, i + first + second)),
                None => i += first,
            },
            None => i += 1,
        }
    }
    None
}

/// Fold one event block into the tally and the reply being assembled.
fn absorb_event(
    block: &[u8],
    tally: &mut CallTally,
    events: &mut Vec<GenerateEvent>,
    full: &mut String,
) {
    let text = String::from_utf8_lossy(block);
    note_line_endings(block, tally);

    // Every `data:` line of one event belongs to the same payload, joined with
    // a newline — that is what the spec says, and a JSON object split across
    // two data lines (which a long recap chunk can be) only survives if it is
    // rejoined that way. The old parser concatenated them with nothing in
    // between, which quietly corrupted exactly those payloads.
    let mut data = String::new();
    let mut saw_data = false;
    for line in text.split(['\r', '\n']) {
        let Some(rest) = line.strip_prefix("data:") else {
            continue;
        };
        // Exactly one optional space after the colon is part of the framing;
        // anything more is payload.
        let rest = rest.strip_prefix(' ').unwrap_or(rest);
        if saw_data {
            data.push('\n');
        }
        data.push_str(rest);
        saw_data = true;
    }

    if !saw_data || data.trim().is_empty() {
        tally.keepalives += 1;
        return;
    }
    tally.events += 1;

    if data.trim() == "[DONE]" {
        tally.done_marker = true;
        return;
    }

    let chunk: GenerateChunk = match serde_json::from_str(&data) {
        Ok(chunk) => {
            tally.parsed += 1;
            chunk
        }
        Err(e) => {
            tally.parse_failures += 1;
            if tally.first_parse_shape.is_none() {
                tally.first_parse_shape = Some(parse_failure_shape(&e));
            }
            return;
        }
    };

    if let Some(reason) = chunk.prompt_feedback.and_then(|f| f.block_reason) {
        if tally.block_reason.is_none() {
            tally.block_reason = Some(reason);
        }
    }
    if let Some(usage) = chunk.usage {
        tally.prompt_tokens = usage.prompt_tokens.or(tally.prompt_tokens);
        tally.candidate_tokens = usage.candidate_tokens.or(tally.candidate_tokens);
        tally.thought_tokens = usage.thought_tokens.or(tally.thought_tokens);
        tally.total_tokens = usage.total_tokens.or(tally.total_tokens);
    }
    if let Some(version) = chunk.model_version {
        tally.model_version = Some(version);
    }
    if let Some(id) = chunk.response_id {
        tally.response_id = Some(id);
    }

    for candidate in &chunk.candidates {
        tally.candidates += 1;
        if let Some(reason) = &candidate.finish_reason {
            tally.note_finish_reason(reason);
        }
        let Some(content) = &candidate.content else {
            continue;
        };
        for part in &content.parts {
            tally.parts += 1;
            if part.thought {
                tally.thought_parts += 1;
                continue;
            }
            if part.text.is_empty() {
                continue;
            }
            tally.text_parts += 1;
            full.push_str(&part.text);
            events.push(GenerateEvent::Text(part.text.clone()));
        }
    }
}

/// Read the whole body, folding each complete event into the tally as it
/// arrives. Bytes are buffered as bytes, not as lossy text: a chunk boundary
/// falls mid-character often enough in any language with accents, and decoding
/// each chunk on its own turns that character into a replacement mark.
async fn read_sse_body(
    resp: &mut reqwest::Response,
    cancel: &CancelFlag,
    tally: &mut CallTally,
    events: &mut Vec<GenerateEvent>,
    full: &mut String,
) -> Result<(), SummarizeError> {
    let mut buf: Vec<u8> = Vec::new();
    let mut bom_checked = false;

    loop {
        if cancel.is_cancelled() {
            return Err(SummarizeError::Cancelled);
        }
        match resp.chunk().await {
            Ok(Some(bytes)) => {
                buf.extend_from_slice(&bytes);
                if !bom_checked && buf.len() >= BOM.len() {
                    if buf.starts_with(&BOM) {
                        buf.drain(..BOM.len());
                        tally.bom = true;
                    }
                    bom_checked = true;
                }
                while let Some((end, consumed)) = next_event(&buf) {
                    // The blank line between events is where the framing shows
                    // itself, and it is about to be thrown away, so read it
                    // first: `crlf` in the log line is the whole diagnosis.
                    let separator = buf[end..consumed].to_vec();
                    let block = buf[..end].to_vec();
                    buf.drain(..consumed);
                    note_line_endings(&separator, tally);
                    absorb_event(&block, tally, events, full);
                }
            }
            Ok(None) => break,
            Err(e) => return Err(classify_reqwest_error(&e)),
        }
    }

    // A last event with no blank line after it is still an event. Google's
    // stream normally ends with the separator, but a proxy that closes the
    // connection promptly does not have to, and dropping the final chunk means
    // dropping the end of the recap — or, when the whole reply arrives as one
    // unterminated block, the entire thing.
    tally.leftover_bytes = buf.len();
    if !buf.is_empty() {
        absorb_event(&buf, tally, events, full);
    }
    Ok(())
}

/// Parse one Server-Sent-Events response (`alt=sse`) into the events it
/// describes, accumulating the full reply as it goes.
///
/// Every path through here logs exactly one telemetry line first, then decides
/// what the outcome was. The three failures are told apart on purpose:
///
/// * **blocked** — Google answered and refused. Nothing to retry.
/// * **empty** — Google answered, Echo read it, there was no text in it. Text
///   that is nothing but whitespace counts as no text: a recap of four newlines
///   is not a recap, and passing it on would put a blank page where the meeting
///   should be.
/// * **malformed** — events arrived and **not one** of them parsed. Echo's bug,
///   not the meeting's, and the counters in the log line say which one. Strictly
///   "none parsed": a reply where some events parsed and some did not is a
///   *partial* reply, and it is reported as the success it is with a loud line
///   in the log rather than thrown away (review of 2026-08-20, finding 4).
async fn drain_sse(
    mut resp: reqwest::Response,
    cancel: &CancelFlag,
    model: &str,
) -> Result<Vec<GenerateEvent>, SummarizeError> {
    let mut tally = CallTally::default();
    let mut events = Vec::new();
    let mut full = String::new();

    let transport = read_sse_body(&mut resp, cancel, &mut tally, &mut events, &mut full).await;
    tally.text_chars = full.chars().count();
    tally.log(model);
    transport?;

    if let Some(reason) = tally
        .block_reason
        .clone()
        .or_else(|| tally.blocking_finish_reason.clone())
    {
        return Err(SummarizeError::Blocked { reason });
    }
    if full.trim().is_empty() {
        // Events came down the wire and Echo could read none of them: the two
        // sides disagree about the shape of an answer, which is Echo's bug.
        // Anything else with no text in it is an empty reply, whatever else went
        // wrong on the way.
        if tally.events > 0 && tally.parsed == 0 {
            return Err(SummarizeError::MalformedReply);
        }
        return Err(SummarizeError::EmptyReply);
    }

    // Some of the reply was unreadable and there is still a recap here. It goes
    // out — a recap missing a paragraph beats no recap at all — but not quietly:
    // this is the line that says the recap somebody is reading is incomplete.
    if tally.parse_failures > 0 {
        tracing::warn!(
            target: TELEMETRY_TARGET,
            model = %model,
            parsed = tally.parsed,
            parse_failures = tally.parse_failures,
            text_chars = tally.text_chars,
            reason = tally.first_parse_shape.as_deref().unwrap_or("-"),
            "this recap was assembled from a reply Echo could only partly read; \
             some of what Google sent is missing from it"
        );
    }

    events.push(GenerateEvent::Done { text: full });
    Ok(events)
}

/// The keys Google's `Schema` actually defines. Everything else in a JSON
/// Schema — `additionalProperties`, `minLength`, `$schema`, `title` — is not
/// part of it, and sending one is a 400 for the whole request.
const SCHEMA_KEYS_KEPT: &[&str] = &[
    "description",
    "enum",
    "format",
    "maxItems",
    "minItems",
    "nullable",
    "required",
];

/// Translate a JSON Schema into the `Schema` shape Google's `responseSchema`
/// takes.
///
/// Two fields can carry a schema. `responseJsonSchema` accepts JSON Schema as
/// written, but only on the newest model families — and Echo lets a person pick
/// any model from Google's catalogue, including the older ones, where that field
/// is not recognised and the request fails. `responseSchema` is understood by
/// every model that can do structured output at all, so that is the one Echo
/// sends, with the schema translated to its shape:
///
/// * a type is a single value, uppercase (`"OBJECT"`), never a list
/// * nullability is `nullable: true`, not `"type": ["string", "null"]`
/// * keys Google does not define are dropped rather than sent and rejected
///
/// Nothing is lost by dropping them: the reply is validated locally against the
/// real schema either way ([`crate::summarize`] parses it with unknown fields
/// denied), which is what actually enforces strictness. The provider schema only
/// has to be good enough to shape the reply.
fn to_gemini_schema(schema: &serde_json::Value) -> serde_json::Value {
    let Some(object) = schema.as_object() else {
        // Not a schema object we understand; send it as it came rather than
        // inventing something.
        return schema.clone();
    };

    let mut out = serde_json::Map::new();
    let mut nullable = false;

    if let Some(ty) = object.get("type") {
        match ty {
            serde_json::Value::Array(members) => {
                for member in members {
                    match member.as_str() {
                        Some("null") => nullable = true,
                        Some(name) => {
                            out.entry("type")
                                .or_insert_with(|| gemini_type_name(name).into());
                        }
                        None => {}
                    }
                }
            }
            serde_json::Value::String(name) => {
                out.insert("type".into(), gemini_type_name(name).into());
            }
            _ => {}
        }
    }

    for key in SCHEMA_KEYS_KEPT {
        if let Some(value) = object.get(*key) {
            out.insert((*key).to_string(), value.clone());
        }
    }
    if nullable {
        out.insert("nullable".into(), true.into());
    }
    if let Some(items) = object.get("items") {
        out.insert("items".into(), to_gemini_schema(items));
    }
    if let Some(serde_json::Value::Object(properties)) = object.get("properties") {
        let translated: serde_json::Map<String, serde_json::Value> = properties
            .iter()
            .map(|(name, value)| (name.clone(), to_gemini_schema(value)))
            .collect();
        out.insert("properties".into(), translated.into());
    }

    out.into()
}

/// Google's `Type` enum spells its members in capitals.
fn gemini_type_name(name: &str) -> String {
    name.to_ascii_uppercase()
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
    timeout: Duration,
    cancel: CancelFlag,
) -> Result<Vec<GenerateEvent>, SummarizeError> {
    if cancel.is_cancelled() {
        return Err(SummarizeError::Cancelled);
    }

    let base = api_base.trim_end_matches('/');
    let url = format!("{base}/v1beta/models/{model}:streamGenerateContent?alt=sse");

    // No sampling settings at all. Echo used to force temperature 0.2 on every
    // request; Google asks that recent Gemini models be left at their defaults
    // and warns that a low temperature is a cause of degraded, looping or empty
    // replies. What Echo wants — a factual recap — is the model's default
    // behaviour, not something to be dialled in. (Ollama still takes the number;
    // that is its own backend's setting.)
    let mut generation_config = serde_json::Map::new();
    if let Some(schema) = &json_schema {
        generation_config.insert(
            "responseMimeType".into(),
            serde_json::Value::String("application/json".into()),
        );
        generation_config.insert("responseSchema".into(), to_gemini_schema(schema));
    }

    let mut body = serde_json::json!({
        "contents": [{ "role": "user", "parts": [{ "text": prompt }] }],
    });
    if !generation_config.is_empty() {
        body["generationConfig"] = generation_config.into();
    }
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
        // One line for a call that never got as far as a reply, so the log
        // still accounts for every request. The status, not the body: an error
        // body is Google's prose, and this line is for counting.
        tracing::info!(
            target: TELEMETRY_TARGET,
            model = %model,
            status = status.as_u16(),
            json_mode = json_schema.is_some(),
            "gemini refused the call"
        );
        // The saved model is gone or was never right. Say so in words the person
        // can act on instead of handing them Google's sentence about API
        // versions and generation methods.
        if is_model_not_found(status, &text) {
            tracing::warn!(
                target: TELEMETRY_TARGET,
                model = %model,
                status = status.as_u16(),
                "the saved model is not one Google answers for"
            );
            return Err(SummarizeError::ModelNotFound { model });
        }
        return Err(classify_status(status, &text));
    }

    drain_sse(resp, &cancel, &model).await
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
                // Google answered, and said no to the request rather than to the
                // key. "That key wasn't accepted" would be a lie about a key
                // that was.
                Err(SummarizeError::RequestRefused { .. }) => Ok(ProviderTestResult {
                    ok: false,
                    message: "That key works, but Google wouldn't take the request. Try again, \
                              or pick something else to write recaps."
                        .into(),
                    caps,
                    models: Vec::new(),
                    leaves_machine: true,
                }),
                // The key is fine. Saying "that key wasn't accepted" here sent
                // people to re-paste a key that was never the problem — what
                // ran out was the allowance behind it.
                Err(SummarizeError::QuotaExhausted) => Ok(ProviderTestResult {
                    ok: false,
                    message: "That key works, but Google says it has no room left right now. \
                              Try again later."
                        .into(),
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
        // `req.temperature` is deliberately not read: see `run_generate_content`.
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
    fn classify_status_tells_a_refused_key_from_a_used_up_allowance() {
        assert!(matches!(
            classify_status(reqwest::StatusCode::UNAUTHORIZED, "{}"),
            SummarizeError::Rejected(_)
        ));
        assert!(matches!(
            classify_status(reqwest::StatusCode::FORBIDDEN, "{}"),
            SummarizeError::Rejected(_)
        ));
        // 429 is its own thing: the key was accepted, the allowance was not.
        assert!(matches!(
            classify_status(reqwest::StatusCode::TOO_MANY_REQUESTS, "{}"),
            SummarizeError::QuotaExhausted
        ));
        assert!(matches!(
            classify_status(reqwest::StatusCode::INTERNAL_SERVER_ERROR, "{}"),
            SummarizeError::Failed(_)
        ));
    }

    /// The regression this taxonomy was rewritten for: every unmatched 4xx used
    /// to become "that key wasn't accepted", so a schema Google did not like sent
    /// people off to replace a key that was working
    /// (review of 2026-08-20, finding 5).
    #[test]
    fn a_4xx_that_is_not_about_the_key_does_not_blame_the_key() {
        let schema_400 = classify_status(
            reqwest::StatusCode::BAD_REQUEST,
            "{\"error\":{\"message\":\"Invalid JSON payload received. Unknown name \
             \\\"additionalProperties\\\"\",\"status\":\"INVALID_ARGUMENT\"}}",
        );
        match schema_400 {
            SummarizeError::RequestRefused { status, detail } => {
                assert_eq!(status, 400);
                assert!(detail.contains("additionalProperties"), "{detail}");
            }
            other => panic!("a schema 400 is not about the key: {other:?}"),
        }

        // Anything else in the range that says nothing about a credential.
        for code in [404u16, 405, 413, 415, 422] {
            let status = reqwest::StatusCode::from_u16(code).unwrap();
            assert!(
                matches!(
                    classify_status(status, "{\"error\":{\"message\":\"no\"}}"),
                    SummarizeError::RequestRefused { .. }
                ),
                "http {code} blamed the key"
            );
        }

        // Google's own answer to a bad key is a 400, not a 401 — so the wording
        // decides, not the number.
        assert!(matches!(
            classify_status(
                reqwest::StatusCode::BAD_REQUEST,
                "{\"error\":{\"message\":\"API key not valid. Please pass a valid API key.\",\
                 \"status\":\"INVALID_ARGUMENT\"}}"
            ),
            SummarizeError::Rejected(_)
        ));
    }

    /// And the sentence that reaches the person says the request was turned
    /// down, not that the key is wrong.
    #[tokio::test]
    async fn a_request_google_turns_down_is_not_reported_as_a_bad_key() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1beta/models"))
            .respond_with(ResponseTemplate::new(400).set_body_string(
                "{\"error\":{\"code\":400,\"message\":\"Unknown name \\\"widget\\\"\",\
                 \"status\":\"INVALID_ARGUMENT\"}}",
            ))
            .mount(&server)
            .await;

        let connector = GeminiConnector::new("good-key", None).with_api_base(server.uri());
        let result = connector.check().await.unwrap();
        assert!(!result.ok);
        let lower = result.message.to_lowercase();
        assert!(
            !lower.contains("wasn't accepted"),
            "a schema 400 must not blame the key: {}",
            result.message
        );
        assert!(lower.contains("key works"), "{}", result.message);
    }

    #[tokio::test]
    async fn a_used_up_allowance_is_never_reported_as_a_bad_key() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1beta/models"))
            .respond_with(ResponseTemplate::new(429).set_body_string(
                "{\"error\":{\"code\":429,\"message\":\"Resource has been exhausted\",\
                 \"status\":\"RESOURCE_EXHAUSTED\"}}",
            ))
            .mount(&server)
            .await;

        let connector = GeminiConnector::new("good-key", None).with_api_base(server.uri());
        let result = connector.check().await.unwrap();
        assert!(!result.ok);
        let lower = result.message.to_lowercase();
        assert!(
            !lower.contains("wasn't accepted") && !lower.contains("check it"),
            "quota exhaustion must not blame the key: {}",
            result.message
        );
        assert!(lower.contains("no room left"), "{}", result.message);
    }

    // -- SSE framing -------------------------------------------------------
    //
    // The bug these were written for: a reply that arrived framed with `\r\n\r\n`
    // was never split into events at all, every event was dropped without a
    // word, and the person was told "the reply came back empty" three times in
    // a row with nothing in the log to say otherwise.

    /// Serve `body` once as a `text/event-stream` and drain it the way a real
    /// call does, over a real socket.
    async fn drain(body: &[u8]) -> Result<Vec<GenerateEvent>, SummarizeError> {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/stream"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(body.to_vec(), "text/event-stream"),
            )
            .mount(&server)
            .await;
        let resp = reqwest::Client::new()
            .get(format!("{}/stream", server.uri()))
            .send()
            .await
            .unwrap();
        drain_sse(resp, &CancelFlag::new(), "test-model").await
    }

    fn done_text(events: &[GenerateEvent]) -> String {
        match events.last() {
            Some(GenerateEvent::Done { text }) => text.clone(),
            other => panic!("expected a Done event last, got {other:?}"),
        }
    }

    fn streamed_text(events: &[GenerateEvent]) -> Vec<String> {
        events
            .iter()
            .filter_map(|e| match e {
                GenerateEvent::Text(t) => Some(t.clone()),
                _ => None,
            })
            .collect()
    }

    fn text_event(text: &str) -> String {
        format!(
            "data: {{\"candidates\":[{{\"content\":{{\"parts\":[{{\"text\":\"{text}\"}}]}}}}]}}"
        )
    }

    #[test]
    fn event_boundaries_are_found_for_every_line_ending() {
        // LF, CRLF, CR, and the mixed pairs a proxy can produce.
        for sep in ["\n\n", "\r\n\r\n", "\r\r", "\n\r\n", "\r\n\n"] {
            let body = format!("data: one{sep}data: two\n\n");
            let (end, consumed) = next_event(body.as_bytes()).expect("an event should be found");
            assert_eq!(&body.as_bytes()[..end], b"data: one", "separator {sep:?}");
            assert_eq!(consumed, end + sep.len(), "separator {sep:?}");
        }
    }

    #[test]
    fn a_trailing_cr_waits_for_the_rest_of_its_line_ending() {
        // "…\n\r" could be a blank line ending in CRLF whose LF is still in
        // flight. Splitting there would cut the next event in half.
        assert!(next_event(b"data: one\n\r").is_none());
        assert!(next_event(b"data: one\n").is_none());
        assert!(next_event(b"data: one").is_none());
    }

    #[tokio::test]
    async fn a_reply_framed_with_crlf_is_read_end_to_end() {
        let body = format!(
            "{}\r\n\r\ndata: {{\"candidates\":[{{\"content\":{{\"parts\":[{{\"text\":\", world\"}}],\
             \"role\":\"model\"}},\"finishReason\":\"STOP\"}}],\"usageMetadata\":\
             {{\"promptTokenCount\":11,\"candidatesTokenCount\":3,\"totalTokenCount\":14}},\
             \"modelVersion\":\"gemini-3.7-flash\"}}\r\n\r\n",
            text_event("Hello")
        );
        let events = drain(body.as_bytes()).await.unwrap();
        assert_eq!(streamed_text(&events), vec!["Hello", ", world"]);
        assert_eq!(done_text(&events), "Hello, world");
    }

    #[tokio::test]
    async fn a_reply_framed_with_lf_still_works() {
        let body = format!("{}\n\n{}\n\n", text_event("Hello"), text_event(" again"));
        let events = drain(body.as_bytes()).await.unwrap();
        assert_eq!(done_text(&events), "Hello again");
    }

    #[tokio::test]
    async fn the_last_event_is_read_even_without_its_blank_line() {
        // The body simply stops after the final event. Everything still in the
        // buffer at that point is an event, not rubbish to throw away.
        let body = format!(
            "{}\r\n\r\n{}",
            text_event("first half "),
            text_event("second half")
        );
        let events = drain(body.as_bytes()).await.unwrap();
        assert_eq!(done_text(&events), "first half second half");
    }

    #[tokio::test]
    async fn a_whole_reply_that_never_gets_a_blank_line_is_still_read() {
        let events = drain(text_event("all of it").as_bytes()).await.unwrap();
        assert_eq!(done_text(&events), "all of it");
    }

    #[tokio::test]
    async fn a_byte_order_mark_in_front_of_the_stream_is_ignored() {
        let mut body = BOM.to_vec();
        body.extend_from_slice(format!("{}\n\n", text_event("with a mark")).as_bytes());
        let events = drain(&body).await.unwrap();
        assert_eq!(done_text(&events), "with a mark");
    }

    #[tokio::test]
    async fn several_data_lines_in_one_event_are_joined_with_a_newline() {
        // One JSON object split across two `data:` lines, which is what the
        // spec says to expect. Concatenating them with nothing in between (the
        // old behaviour) leaves valid-looking JSON only by luck.
        let body = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"line one\\nline two\"}]}}]}\n\n";
        let events = drain(body.as_bytes()).await.unwrap();
        assert_eq!(done_text(&events), "line one\nline two");

        let split = "data: {\"candidates\":[{\"content\":\ndata: {\"parts\":[{\"text\":\"joined\"}]}}]}\n\n";
        let events = drain(split.as_bytes()).await.unwrap();
        assert_eq!(done_text(&events), "joined");
    }

    #[tokio::test]
    async fn comments_and_keep_alives_are_not_mistaken_for_data() {
        let body = format!(": keep-alive\n\nevent: message\n{}\n\n", text_event("hi"));
        let events = drain(body.as_bytes()).await.unwrap();
        assert_eq!(done_text(&events), "hi");
    }

    #[tokio::test]
    async fn a_done_marker_is_not_an_empty_reply() {
        let body = format!("{}\n\ndata: [DONE]\n\n", text_event("finished"));
        let events = drain(body.as_bytes()).await.unwrap();
        assert_eq!(done_text(&events), "finished");
    }

    // -- the three ways a reply can come back with no recap in it -----------

    #[tokio::test]
    async fn a_blocked_prompt_is_blocked_not_empty() {
        let body = "data: {\"candidates\":[],\"promptFeedback\":{\"blockReason\":\"SAFETY\"}}\n\n";
        let err = drain(body.as_bytes()).await.unwrap_err();
        assert!(
            matches!(&err, SummarizeError::Blocked { reason } if reason == "SAFETY"),
            "{err:?}"
        );
    }

    #[tokio::test]
    async fn a_blocked_answer_is_blocked_not_empty() {
        for reason in ["SAFETY", "RECITATION", "PROHIBITED_CONTENT"] {
            let body = format!("data: {{\"candidates\":[{{\"finishReason\":\"{reason}\"}}]}}\n\n");
            let err = drain(body.as_bytes()).await.unwrap_err();
            assert!(
                matches!(&err, SummarizeError::Blocked { reason: r } if r == reason),
                "{err:?}"
            );
        }
    }

    #[tokio::test]
    async fn a_reply_with_no_text_in_it_is_empty() {
        // Read cleanly, nothing in it: candidates with no parts, then none at
        // all. Both are "Google had nothing to say", not a framing bug.
        let bodies = [
            "data: {\"candidates\":[{\"content\":{\"parts\":[]},\"finishReason\":\"STOP\"}]}\n\n",
            "data: {\"candidates\":[]}\n\n",
            "",
        ];
        for body in bodies {
            let err = drain(body.as_bytes()).await.unwrap_err();
            assert!(
                matches!(err, SummarizeError::EmptyReply),
                "{body:?} gave {err:?}"
            );
        }
    }

    #[tokio::test]
    async fn events_that_none_of_them_parse_are_a_malformed_reply_not_an_empty_one() {
        // Events arrived and Echo could not read one of them. That is Echo's
        // bug to fix, and it must not look like the model having nothing to say.
        let body = "data: <html>gateway error</html>\n\ndata: also not json\n\n";
        let err = drain(body.as_bytes()).await.unwrap_err();
        assert!(matches!(err, SummarizeError::MalformedReply), "{err:?}");
    }

    #[tokio::test]
    async fn one_unreadable_event_does_not_lose_the_rest_of_the_reply() {
        let body = format!("data: not json\n\n{}\n\n", text_event("the recap"));
        let events = drain(body.as_bytes()).await.unwrap();
        assert_eq!(done_text(&events), "the recap");
    }

    /// The half-and-half case, which used to be called malformed even though a
    /// perfectly good recap had arrived: some events parsed, some did not. It is
    /// a partial reply, which is a success — loudly logged, never discarded
    /// (review of 2026-08-20, finding 4).
    #[tokio::test]
    async fn a_reply_only_partly_readable_still_produces_the_recap_it_carried() {
        let body = format!(
            "{}\n\ndata: {{\"candidates\":[{{\"content\":\"not an object\"}}]}}\n\n{}\n\n",
            text_event("first half "),
            text_event("second half")
        );
        let events = drain(body.as_bytes()).await.unwrap();
        assert_eq!(done_text(&events), "first half second half");
        assert_eq!(
            streamed_text(&events),
            vec!["first half ".to_string(), "second half".to_string()]
        );
    }

    /// "None parsed" is the definition of malformed, so one event that parsed is
    /// enough to make an answer with no text in it an *empty* one.
    #[tokio::test]
    async fn a_textless_reply_with_one_unreadable_event_is_empty_not_malformed() {
        let body = "data: {\"candidates\":[]}\n\ndata: not json\n\n";
        let err = drain(body.as_bytes()).await.unwrap_err();
        assert!(matches!(err, SummarizeError::EmptyReply), "{err:?}");
    }

    /// A recap of four newlines is not a recap. It used to pass the emptiness
    /// check and land on the screen as a blank page.
    #[tokio::test]
    async fn a_reply_that_is_only_whitespace_is_an_empty_one() {
        for whitespace in ["\\n\\n\\n", " ", "\\t \\n"] {
            let body = format!("{}\n\n", text_event(whitespace));
            let err = drain(body.as_bytes()).await.unwrap_err();
            assert!(
                matches!(err, SummarizeError::EmptyReply),
                "{whitespace:?} gave {err:?}"
            );
        }
    }

    /// The telemetry promise: counters and positions, never a fragment of
    /// somebody's meeting. Serde's own message quotes the value it rejected, so
    /// this is the one thing that must not be logged verbatim.
    #[test]
    fn a_parse_failure_is_described_without_quoting_the_payload() {
        let err = serde_json::from_str::<GenerateChunk>(
            "{\"candidates\":\"acquisition of Contoso for 4.2 million\"}",
        )
        .expect_err("a string where a list belongs");
        let shape = parse_failure_shape(&err);
        assert!(!shape.contains("Contoso"), "{shape}");
        assert!(!shape.contains("acquisition"), "{shape}");
        assert!(shape.contains("wrong shape"), "{shape}");
        assert!(shape.contains("line"), "{shape}");

        assert!(
            parse_failure_shape(
                &serde_json::from_str::<GenerateChunk>("<html>nope</html>").unwrap_err()
            )
            .contains("not json")
        );
        assert!(parse_failure_shape(
            &serde_json::from_str::<GenerateChunk>("{\"candidates\":[").unwrap_err()
        )
        .contains("cut short"));
    }

    #[tokio::test]
    async fn a_models_own_reasoning_never_lands_in_the_recap() {
        let body = "data: {\"candidates\":[{\"content\":{\"parts\":[\
                    {\"text\":\"thinking out loud\",\"thought\":true},\
                    {\"text\":\"the recap\"}]},\"finishReason\":\"STOP\"}]}\n\n";
        let events = drain(body.as_bytes()).await.unwrap();
        assert_eq!(done_text(&events), "the recap");
    }

    #[tokio::test]
    async fn a_truncated_reply_is_kept_rather_than_thrown_away() {
        let body =
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"as far as it got\"}]},\
                    \"finishReason\":\"MAX_TOKENS\"}]}\n\n";
        let events = drain(body.as_bytes()).await.unwrap();
        assert_eq!(done_text(&events), "as far as it got");
    }

    #[tokio::test]
    async fn an_unrecognised_finish_reason_is_not_treated_as_a_refusal() {
        let body = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"fine\"}]},\
                    \"finishReason\":\"FINISH_REASON_UNSPECIFIED\"}]}\n\n";
        let events = drain(body.as_bytes()).await.unwrap();
        assert_eq!(done_text(&events), "fine");
    }

    #[tokio::test]
    async fn drain_sse_respects_cancellation() {
        let cancel = CancelFlag::new();
        cancel.cancel();
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/stream"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_raw(format!("{}\n\n", text_event("hi")), "text/event-stream"),
            )
            .mount(&server)
            .await;
        let resp = reqwest::Client::new()
            .get(format!("{}/stream", server.uri()))
            .send()
            .await
            .unwrap();
        let err = drain_sse(resp, &cancel, "test-model").await.unwrap_err();
        assert!(matches!(err, SummarizeError::Cancelled));
    }

    // -- what the tally counts --------------------------------------------

    #[test]
    fn the_tally_separates_events_from_the_ones_that_parsed() {
        let mut tally = CallTally::default();
        let mut events = Vec::new();
        let mut full = String::new();

        absorb_event(b": keep-alive", &mut tally, &mut events, &mut full);
        absorb_event(b"data: not json", &mut tally, &mut events, &mut full);
        absorb_event(
            text_event("hi").as_bytes(),
            &mut tally,
            &mut events,
            &mut full,
        );

        assert_eq!(tally.keepalives, 1);
        assert_eq!(tally.events, 2);
        assert_eq!(tally.parsed, 1);
        assert_eq!(tally.parse_failures, 1);
        assert!(tally.first_parse_shape.is_some());
        assert_eq!(tally.candidates, 1);
        assert_eq!(tally.parts, 1);
        assert_eq!(tally.text_parts, 1);
        assert_eq!(full, "hi");
    }

    #[test]
    fn the_tally_reports_which_line_endings_the_reply_used() {
        let mut tally = CallTally::default();
        let (mut events, mut full) = (Vec::new(), String::new());
        absorb_event(b"data: not json\r\n", &mut tally, &mut events, &mut full);
        assert_eq!(tally.line_endings(), "crlf");

        let mut tally = CallTally::default();
        absorb_event(b"data: not json\n", &mut tally, &mut events, &mut full);
        assert_eq!(tally.line_endings(), "lf");
    }

    #[test]
    fn the_framing_itself_is_what_the_line_ending_field_reports() {
        // An event can contain no line ending at all — the framing is the blank
        // line between events, which is exactly the byte run that gets thrown
        // away. Reading it is what makes "crlf" show up in the log next to a
        // reply that came back empty.
        for (body, expected) in [
            (&b"data: one\r\n\r\ndata: two\r\n\r\n"[..], "crlf"),
            (&b"data: one\n\ndata: two\n\n"[..], "lf"),
            (&b"data: one\r\rdata: two\r\r"[..], "cr"),
            (&b"data: one\r\n\ndata: two\n\n"[..], "mixed"),
        ] {
            let mut tally = CallTally::default();
            let mut buf = body.to_vec();
            while let Some((end, consumed)) = next_event(&buf) {
                let separator = buf[end..consumed].to_vec();
                note_line_endings(&separator, &mut tally);
                buf.drain(..consumed);
            }
            assert_eq!(tally.line_endings(), expected, "{body:?}");
        }
    }

    #[test]
    fn the_tally_keeps_counts_and_never_content() {
        let mut tally = CallTally::default();
        let (mut events, mut full) = (Vec::new(), String::new());
        absorb_event(
            b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"secret meeting words\"}]},\
              \"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":7,\
              \"candidatesTokenCount\":2,\"thoughtsTokenCount\":1,\"totalTokenCount\":10},\
              \"modelVersion\":\"gemini-3.7-flash\",\"responseId\":\"abc123\"}",
            &mut tally,
            &mut events,
            &mut full,
        );
        assert_eq!(tally.prompt_tokens, Some(7));
        assert_eq!(tally.candidate_tokens, Some(2));
        assert_eq!(tally.thought_tokens, Some(1));
        assert_eq!(tally.total_tokens, Some(10));
        assert_eq!(tally.model_version.as_deref(), Some("gemini-3.7-flash"));
        assert_eq!(tally.response_id.as_deref(), Some("abc123"));
        assert_eq!(tally.finish_reasons, vec!["STOP".to_string()]);
        // Nothing the model wrote is in the tally; the length of it is.
        let printed = format!("{tally:?}");
        assert!(!printed.contains("secret meeting words"), "{printed}");
    }

    // -- the JSON-mode request shape ---------------------------------------

    #[test]
    fn a_json_schema_becomes_the_shape_google_documents() {
        let translated = to_gemini_schema(&crate::summarize::templates::action_item_schema());
        let expected = serde_json::json!({
            "type": "OBJECT",
            "required": ["items"],
            "properties": {
                "items": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "required": ["description"],
                        "properties": {
                            "description": { "type": "STRING" },
                            // `["string","null"]` is not a type Google accepts.
                            "owner": { "type": "STRING", "nullable": true },
                            "dueHint": { "type": "STRING", "nullable": true }
                        }
                    }
                }
            }
        });
        assert_eq!(translated, expected);
        // The keys that made the whole request a 400 are gone.
        let printed = translated.to_string();
        assert!(!printed.contains("additionalProperties"), "{printed}");
        assert!(!printed.contains("minLength"), "{printed}");
    }

    #[tokio::test]
    async fn a_json_mode_call_sends_a_schema_google_accepts_and_no_sampling_settings() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(format!(
                "/v1beta/models/{DEFAULT_MODEL}:streamGenerateContent"
            )))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"{\\\"items\\\":[]}\"}]},\
                 \"finishReason\":\"STOP\"}]}\r\n\r\n",
                "text/event-stream",
            ))
            .mount(&server)
            .await;

        let connector = GeminiConnector::new("test-key", None).with_api_base(server.uri());
        let events: Vec<_> = connector
            .generate(GenerateRequest {
                prompt: "list the tasks".into(),
                json_schema: Some(crate::summarize::templates::action_item_schema()),
                ..Default::default()
            })
            .collect()
            .await;
        assert_eq!(
            done_text(&events.into_iter().map(|e| e.unwrap()).collect::<Vec<_>>()),
            "{\"items\":[]}"
        );

        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        let config = &body["generationConfig"];
        assert_eq!(config["responseMimeType"], "application/json");
        assert_eq!(config["responseSchema"]["type"], "OBJECT");
        assert_eq!(
            config["responseSchema"]["properties"]["items"]["items"]["properties"]["owner"]
                ["nullable"],
            true
        );
        // The legacy field, not the JSON-Schema one: every model that can do
        // structured output understands this one.
        assert!(config.get("responseJsonSchema").is_none(), "{config}");
        // Sampling is the model's business (Google's own guidance for 3.x).
        assert!(config.get("temperature").is_none(), "{config}");
        assert!(config.get("topP").is_none(), "{config}");
        assert!(config.get("topK").is_none(), "{config}");
    }

    #[tokio::test]
    async fn a_plain_recap_call_sends_no_generation_config_at_all() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(format!(
                "/v1beta/models/{DEFAULT_MODEL}:streamGenerateContent"
            )))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"recap\"}]},\
                 \"finishReason\":\"STOP\"}]}\n\n",
                "text/event-stream",
            ))
            .mount(&server)
            .await;

        let connector = GeminiConnector::new("test-key", None).with_api_base(server.uri());
        let _: Vec<_> = connector
            .generate(GenerateRequest {
                prompt: "write a recap".into(),
                // Even when a caller passes one, Gemini does not send it.
                temperature: Some(0.2),
                ..Default::default()
            })
            .collect()
            .await;

        let requests = server.received_requests().await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert!(body.get("generationConfig").is_none(), "{body}");
    }
}
