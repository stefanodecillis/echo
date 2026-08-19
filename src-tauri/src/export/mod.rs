//! Getting a meeting out of Echo: Markdown, DOCX, PDF, plain text.
//!
//! IMPLEMENTED-BY: export agent (M7).
//!
//! Order of work (DESIGN §5 M7, review finding 32):
//! * Markdown first. It is the format everything else is built from.
//! * DOCX with `docx-rs`, walking the same document model.
//! * PDF by printing from a hidden webview to a file. That pipeline gets spiked
//!   before anyone relies on it; there is no pure-Rust PDF path here.
//! * Plain text for pasting into a chat.
//!
//! Exports go where the person picks, through the system save dialog. Echo never
//! writes outside its own directories on its own.
//!
//! ## PDF decision (recorded for the integrator)
//!
//! Tauri 2 / wry expose exactly one print primitive: `Webview::print()` /
//! `WebviewWindow::print()`, which opens the OS's native print dialog —
//! macOS only (`wry`), a no-op elsewhere — with no way to target a file path
//! or run headless. There is no `print_to_pdf(path)` API anywhere in tauri 2,
//! so the "hidden webview print-to-file pipeline" DESIGN §5 M7 anticipates
//! does not exist to build on. `write_pdf` therefore always returns
//! [`ExportError::PdfUnavailable`], which the existing `UiError` mapping in
//! `commands.rs` already turns into "PDF isn't available on this computer.
//! Word or Markdown will work." — the UI can hide the PDF option behind that,
//! or a "coming soon" state, without any change to this module's contract.

use std::collections::HashMap;
use std::path::{Component, Path};

use docx_rs::{Docx, Paragraph, Run, Table, TableAlignmentType, TableCell, TableRow, WidthType};

use crate::db::repo;
use crate::db::Db;
use crate::types::{
    Channel, ExportFormat, ExportRequest, ExportResult, JobKind, JobStatus, Meeting, Speaker,
    TranscriptQuery,
};

#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    #[error("not implemented yet")]
    NotImplemented,
    #[error("this meeting has nothing to export yet")]
    Empty,
    #[error("could not write the file: {0}")]
    Write(String),
    #[error("PDF export is not available on this system: {0}")]
    PdfUnavailable(String),
    #[error("database problem: {0}")]
    Db(#[from] crate::db::DbError),
}

/// The document Echo assembles before choosing a file format.
#[derive(Debug, Clone, Default)]
pub struct ExportDocument {
    pub title: String,
    /// Date and duration, already formatted for reading.
    pub subtitle: String,
    /// The recap, as sanitized markdown.
    pub recap_md: Option<String>,
    /// Action items with owner and timing.
    pub action_items: Vec<(String, Option<String>, Option<String>, bool)>,
    /// Transcript lines: timestamp, speaker, text.
    pub transcript: Vec<(String, String, String)>,
}

// ---------------------------------------------------------------------------
// Document assembly
// ---------------------------------------------------------------------------

/// Build the document from the database. Every format starts here, so all four
/// stay consistent.
pub async fn build_document(db: &Db, req: &ExportRequest) -> Result<ExportDocument, ExportError> {
    let meeting = repo::get_meeting(db, &req.meeting_id)
        .await?
        .ok_or_else(|| {
            ExportError::Db(crate::db::DbError::NotFound(format!(
                "meeting {}",
                req.meeting_id
            )))
        })?;

    let include_recap = req.include_recap.unwrap_or(true);
    let include_action_items = req.include_action_items.unwrap_or(true);
    let include_transcript = req.include_transcript.unwrap_or(false);

    let recap_md = if include_recap {
        let summary = match &req.summary_id {
            Some(id) => repo::get_summary(db, id).await?,
            None => repo::latest_summary(db, &meeting.id).await?,
        };
        summary.map(|s| s.content_md)
    } else {
        None
    };

    let action_items = if include_action_items {
        repo::list_action_items(db, &meeting.id)
            .await?
            .into_iter()
            .map(|a| (a.description, a.owner, a.due_hint, a.done))
            .collect()
    } else {
        Vec::new()
    };

    let transcript = if include_transcript {
        build_transcript_lines(db, &meeting.id).await?
    } else {
        Vec::new()
    };

    if recap_md.is_none() && action_items.is_empty() && transcript.is_empty() {
        return Err(ExportError::Empty);
    }

    Ok(ExportDocument {
        title: meeting_title(&meeting),
        subtitle: meeting_subtitle(&meeting),
        recap_md,
        action_items,
        transcript,
    })
}

/// Never blank: the person always has something to look at in a file name or
/// a document heading.
fn meeting_title(meeting: &Meeting) -> String {
    let trimmed = meeting.title.trim();
    if trimmed.is_empty() {
        "Untitled meeting".to_string()
    } else {
        trimmed.to_string()
    }
}

/// "August 19, 2026 · 42 min" — plain reading copy, no jargon.
fn meeting_subtitle(meeting: &Meeting) -> String {
    let date = format_date(&meeting.started_at);
    let duration = format_duration(meeting.duration_ms);
    match (date, duration) {
        (Some(d), Some(dur)) => format!("{d} · {dur}"),
        (Some(d), None) => d,
        (None, Some(dur)) => dur,
        (None, None) => String::new(),
    }
}

fn format_date(started_at: &str) -> Option<String> {
    chrono::DateTime::parse_from_rfc3339(started_at)
        .ok()
        .map(|dt| dt.format("%B %-d, %Y").to_string())
}

fn format_duration(duration_ms: i64) -> Option<String> {
    if duration_ms <= 0 {
        return None;
    }
    let total_minutes = (duration_ms / 60_000).max(0);
    let hours = total_minutes / 60;
    let minutes = total_minutes % 60;
    Some(if hours > 0 {
        format!("{hours} hr {minutes} min")
    } else {
        format!("{minutes} min")
    })
}

/// `mm:ss`, or `h:mm:ss` once a meeting runs past an hour.
fn format_timestamp(t_ms: i64) -> String {
    let total_secs = (t_ms.max(0)) / 1000;
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes}:{seconds:02}")
    }
}

/// Follow a speaker's `alias_of` chain within an already-fetched map, so a
/// renamed-then-merged speaker still shows its survivor's name.
fn resolve_speaker_name<'a>(
    speakers: &'a HashMap<String, Speaker>,
    mut id: &'a str,
) -> Option<String> {
    let mut hops = 0;
    loop {
        let speaker = speakers.get(id)?;
        match &speaker.alias_of {
            Some(next) if hops < 8 => {
                id = next;
                hops += 1;
            }
            _ => return Some(speaker.display_name.clone()),
        }
    }
}

fn default_speaker_label(channel: Channel) -> &'static str {
    match channel {
        Channel::Mic => "You",
        Channel::System | Channel::Mixed => "Speaker",
    }
}

async fn build_transcript_lines(
    db: &Db,
    meeting_id: &str,
) -> Result<Vec<(String, String, String)>, ExportError> {
    let speakers: HashMap<String, Speaker> = repo::list_speakers(db, meeting_id)
        .await?
        .into_iter()
        .map(|s| (s.id.clone(), s))
        .collect();

    let segments = repo::get_segments(
        db,
        &TranscriptQuery {
            meeting_id: meeting_id.to_string(),
            from_ms: None,
            to_ms: None,
            limit: Some(50_000),
            include_partial: Some(false),
        },
    )
    .await?;

    Ok(segments
        .into_iter()
        .filter(|s| !s.text.trim().is_empty())
        .map(|s| {
            let label = s
                .speaker_id
                .as_deref()
                .and_then(|id| resolve_speaker_name(&speakers, id))
                .unwrap_or_else(|| default_speaker_label(s.channel).to_string());
            (
                format_timestamp(s.t_start_ms),
                label,
                s.text.trim().to_string(),
            )
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Text / Markdown rendering
// ---------------------------------------------------------------------------

/// Render to a string. Markdown and plain text only.
pub fn render_text(doc: &ExportDocument, format: ExportFormat) -> Result<String, ExportError> {
    match format {
        ExportFormat::Markdown => Ok(render_markdown(doc)),
        ExportFormat::Text => render_plain_text(doc),
        ExportFormat::Docx | ExportFormat::Pdf => Err(ExportError::NotImplemented),
    }
}

fn render_markdown(doc: &ExportDocument) -> String {
    let mut out = String::new();
    out.push_str(&format!("# {}\n\n", doc.title));
    if !doc.subtitle.is_empty() {
        out.push_str(&format!("_{}_\n\n", doc.subtitle));
    }

    if let Some(recap) = &doc.recap_md {
        out.push_str("## Recap\n\n");
        out.push_str(recap.trim());
        out.push_str("\n\n");
    }

    if !doc.action_items.is_empty() {
        out.push_str("## Action items\n\n");
        for (description, owner, due_hint, done) in &doc.action_items {
            let checkbox = if *done { "x" } else { " " };
            out.push_str(&format!("- [{checkbox}] {description}"));
            let mut extras = Vec::new();
            if let Some(owner) = owner.as_deref().filter(|s| !s.is_empty()) {
                extras.push(owner.to_string());
            }
            if let Some(due) = due_hint.as_deref().filter(|s| !s.is_empty()) {
                extras.push(due.to_string());
            }
            if !extras.is_empty() {
                out.push_str(&format!(" ({})", extras.join(", ")));
            }
            out.push('\n');
        }
        out.push('\n');
    }

    if !doc.transcript.is_empty() {
        out.push_str("## Transcript\n\n");
        for (ts, speaker, text) in &doc.transcript {
            out.push_str(&format!("**[{ts}] {speaker}:** {text}\n\n"));
        }
    }

    out.trim_end().to_string() + "\n"
}

/// Plain text for pasting into a chat: no markdown syntax, transcript-first.
fn render_plain_text(doc: &ExportDocument) -> Result<String, ExportError> {
    if doc.transcript.is_empty() {
        return Err(ExportError::Empty);
    }
    let mut out = String::new();
    out.push_str(&doc.title);
    out.push('\n');
    if !doc.subtitle.is_empty() {
        out.push_str(&doc.subtitle);
        out.push('\n');
    }
    out.push('\n');
    for (ts, speaker, text) in &doc.transcript {
        out.push_str(&format!("[{ts}] {speaker}: {text}\n"));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Path validation
// ---------------------------------------------------------------------------

/// Reject anything that could walk the write outside the folder the person
/// chose in the save dialog: relative paths, `..` segments, empty paths.
/// `commands.rs` already checks `is_absolute`; this is the export module's own
/// defense so it never trusts a caller that skips that command.
fn validate_destination(path: &Path) -> Result<(), ExportError> {
    if path.as_os_str().is_empty() {
        return Err(ExportError::Write("no destination was chosen".into()));
    }
    if !path.is_absolute() {
        return Err(ExportError::Write(
            "the destination must be a full path".into(),
        ));
    }
    if path.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(ExportError::Write(
            "that destination tries to leave its folder".into(),
        ));
    }
    Ok(())
}

fn write_text_file(path: &Path, contents: &str) -> Result<u64, ExportError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| ExportError::Write(e.to_string()))?;
    }
    std::fs::write(path, contents).map_err(|e| ExportError::Write(e.to_string()))?;
    Ok(contents.len() as u64)
}

// ---------------------------------------------------------------------------
// Export entry point
// ---------------------------------------------------------------------------

/// Write one meeting to `destination` in the requested format. Recorded as a
/// jobs row so History can show export progress like any other background
/// task.
pub async fn export_meeting(db: &Db, req: &ExportRequest) -> Result<ExportResult, ExportError> {
    let destination = req
        .destination
        .as_deref()
        .ok_or_else(|| ExportError::Write("no destination was chosen".into()))?;
    let dest_path = Path::new(destination);
    validate_destination(dest_path)?;

    let job = repo::create_job(db, Some(&req.meeting_id), JobKind::Export).await?;
    repo::set_job_status(db, &job.id, JobStatus::Running, None).await?;

    match export_meeting_inner(db, req, dest_path).await {
        Ok(result) => {
            let _ = repo::set_job_progress(db, &job.id, 1.0).await;
            repo::set_job_status(db, &job.id, JobStatus::Done, None).await?;
            Ok(result)
        }
        Err(err) => {
            let _ =
                repo::set_job_status(db, &job.id, JobStatus::Failed, Some(&err.to_string())).await;
            Err(err)
        }
    }
}

async fn export_meeting_inner(
    db: &Db,
    req: &ExportRequest,
    dest_path: &Path,
) -> Result<ExportResult, ExportError> {
    let doc = build_document(db, req).await?;

    let bytes = match req.format {
        ExportFormat::Markdown | ExportFormat::Text => {
            let text = render_text(&doc, req.format)?;
            write_text_file(dest_path, &text)?
        }
        ExportFormat::Docx => write_docx(&doc, dest_path).await?,
        ExportFormat::Pdf => write_pdf(&doc, dest_path).await?,
    };

    Ok(ExportResult {
        path: dest_path.display().to_string(),
        bytes,
        format: req.format,
    })
}

// ---------------------------------------------------------------------------
// DOCX
// ---------------------------------------------------------------------------

/// DOCX via `docx-rs`.
pub async fn write_docx(doc: &ExportDocument, destination: &Path) -> Result<u64, ExportError> {
    validate_destination(destination)?;

    let mut docx = Docx::new().add_paragraph(
        Paragraph::new()
            .style("Heading1")
            .add_run(Run::new().bold().size(36).add_text(&doc.title)),
    );

    if !doc.subtitle.is_empty() {
        docx = docx.add_paragraph(
            Paragraph::new().add_run(Run::new().italic().color("666666").add_text(&doc.subtitle)),
        );
    }

    if let Some(recap) = &doc.recap_md {
        docx = docx.add_paragraph(
            Paragraph::new()
                .style("Heading2")
                .add_run(Run::new().bold().size(28).add_text("Recap")),
        );
        docx = append_markdown_body(docx, recap);
    }

    if !doc.action_items.is_empty() {
        docx = docx.add_paragraph(
            Paragraph::new()
                .style("Heading2")
                .add_run(Run::new().bold().size(28).add_text("Action items")),
        );
        docx = docx.add_table(action_items_table(&doc.action_items));
    }

    if !doc.transcript.is_empty() {
        docx = docx.add_paragraph(
            Paragraph::new()
                .style("Heading2")
                .add_run(Run::new().bold().size(28).add_text("Transcript")),
        );
        for (ts, speaker, text) in &doc.transcript {
            docx = docx.add_paragraph(
                Paragraph::new()
                    .add_run(Run::new().bold().add_text(format!("[{ts}] {speaker}: ")))
                    .add_run(Run::new().add_text(text)),
            );
        }
    }

    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent).map_err(|e| ExportError::Write(e.to_string()))?;
    }
    let file = std::fs::File::create(destination).map_err(|e| ExportError::Write(e.to_string()))?;
    docx.build()
        .pack(file)
        .map_err(|e| ExportError::Write(e.to_string()))?;

    Ok(std::fs::metadata(destination).map(|m| m.len()).unwrap_or(0))
}

/// A minimal markdown-to-DOCX walk: headings, bullets, plain paragraphs.
/// The recap is Echo's own sanitized markdown (see `summarize::sanitize_markdown`),
/// never raw HTML, so this line-based reading is enough — no need for a full
/// CommonMark renderer just to open a document.
fn append_markdown_body(mut docx: Docx, markdown: &str) -> Docx {
    for raw_line in markdown.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(text) = line.strip_prefix("### ") {
            docx = docx
                .add_paragraph(Paragraph::new().add_run(Run::new().bold().size(24).add_text(text)));
        } else if let Some(text) = line.strip_prefix("## ") {
            docx = docx
                .add_paragraph(Paragraph::new().add_run(Run::new().bold().size(26).add_text(text)));
        } else if let Some(text) = line.strip_prefix("# ") {
            docx = docx
                .add_paragraph(Paragraph::new().add_run(Run::new().bold().size(28).add_text(text)));
        } else if let Some(text) = line.strip_prefix("- ").or_else(|| line.strip_prefix("* ")) {
            docx = docx
                .add_paragraph(Paragraph::new().add_run(Run::new().add_text(format!("•  {text}"))));
        } else {
            docx = docx.add_paragraph(Paragraph::new().add_run(Run::new().add_text(line)));
        }
    }
    docx
}

fn action_items_table(items: &[(String, Option<String>, Option<String>, bool)]) -> Table {
    let header = TableRow::new(vec![
        header_cell("Task"),
        header_cell("Owner"),
        header_cell("Due"),
    ]);
    let mut rows = vec![header];
    for (description, owner, due_hint, done) in items {
        let task_text = if *done {
            format!("{description} (done)")
        } else {
            description.clone()
        };
        rows.push(TableRow::new(vec![
            body_cell(&task_text),
            body_cell(owner.as_deref().unwrap_or("—")),
            body_cell(due_hint.as_deref().unwrap_or("—")),
        ]));
    }
    Table::new(rows)
        .set_grid(vec![5000, 2500, 2500])
        .align(TableAlignmentType::Left)
        .style("TableGrid")
}

fn header_cell(text: &str) -> TableCell {
    TableCell::new()
        .width(5000, WidthType::Dxa)
        .add_paragraph(Paragraph::new().add_run(Run::new().bold().add_text(text)))
}

fn body_cell(text: &str) -> TableCell {
    TableCell::new()
        .width(5000, WidthType::Dxa)
        .add_paragraph(Paragraph::new().add_run(Run::new().add_text(text)))
}

// ---------------------------------------------------------------------------
// PDF
// ---------------------------------------------------------------------------

/// PDF by printing from a hidden webview. Spiked before it ships — see the
/// module-level doc comment: tauri 2 has no headless print-to-file API, only
/// a native print dialog on macOS, so this always reports unavailable.
pub async fn write_pdf(_doc: &ExportDocument, _destination: &Path) -> Result<u64, ExportError> {
    Err(ExportError::PdfUnavailable(
        "tauri 2 has no headless print-to-PDF API to build a file-writing pipeline on; only a macOS-only print dialog (Webview::print) exists".into(),
    ))
}

// ---------------------------------------------------------------------------
// Diagnostics bundle
// ---------------------------------------------------------------------------

/// Bundle the redacted logs plus a machine summary for the diagnostics button.
///
/// Contains no transcript text and no keys (review finding 40). That is a
/// requirement of the format, not a filter applied at the end.
pub async fn export_diagnostics(
    paths: &crate::paths::AppPaths,
    destination: &Path,
) -> Result<ExportResult, ExportError> {
    validate_destination(destination)?;

    let mut out = String::new();
    out.push_str("Echo diagnostics\n");
    out.push_str("================\n\n");
    out.push_str(
        "This file contains rotating capture/backend logs only: no transcript text, no \
         recap content, and no keys or credentials ever pass through here.\n\n",
    );

    let log_dir = &paths.log_dir;
    let mut log_files: Vec<_> = std::fs::read_dir(log_dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|entry| entry.file_type().map(|t| t.is_file()).unwrap_or(false))
        .collect();
    log_files.sort_by_key(|e| e.file_name());

    if log_files.is_empty() {
        out.push_str("(no log files yet)\n");
    }
    for entry in log_files {
        out.push_str(&format!(
            "--- {} ---\n",
            entry.file_name().to_string_lossy()
        ));
        match std::fs::read_to_string(entry.path()) {
            Ok(contents) => out.push_str(&contents),
            Err(e) => out.push_str(&format!("(could not read this log file: {e})\n")),
        }
        out.push_str("\n\n");
    }

    let bytes = write_text_file(destination, &out)?;
    Ok(ExportResult {
        path: destination.display().to_string(),
        bytes,
        format: ExportFormat::Text,
    })
}

/// A filename that is safe on both platforms: no separators, no reserved
/// characters, not empty, not absurdly long.
pub fn safe_file_stem(title: &str, started_at: &str) -> String {
    let date = started_at.chars().take(10).collect::<String>();
    let cleaned: String = title
        .chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | ' ' | '-' | '_' => c,
            _ => '-',
        })
        .collect();
    let trimmed = cleaned.trim().trim_matches('-').trim();
    let stem = if trimmed.is_empty() {
        "Meeting"
    } else {
        trimmed
    };
    let short: String = stem.chars().take(60).collect();
    format!("{date} {}", short.trim())
}

/// Zero-jargon file name for the save dialog's default: "Echo — <meeting
/// title> — 2026-08-19.md".
pub fn export_file_name(title: &str, started_at: &str, format: ExportFormat) -> String {
    let date = started_at.chars().take(10).collect::<String>();
    let cleaned: String = title
        .chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | ' ' | '-' | '_' | '\'' => c,
            _ => ' ',
        })
        .collect();
    let trimmed = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
    let name = if trimmed.is_empty() {
        "Untitled meeting".to_string()
    } else {
        trimmed
    };
    let short: String = name.chars().take(80).collect();
    let ext = match format {
        ExportFormat::Markdown => "md",
        ExportFormat::Docx => "docx",
        ExportFormat::Pdf => "pdf",
        ExportFormat::Text => "txt",
    };
    format!("Echo — {} — {date}.{ext}", short.trim())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ActionItem, Job as _Job, Meeting, MeetingStatus, Summary};
    use docx_rs::read_docx;

    fn sample_doc() -> ExportDocument {
        ExportDocument {
            title: "Q3 planning".to_string(),
            subtitle: "August 19, 2026 · 42 min".to_string(),
            recap_md: Some(
                "## Highlights\n\n- Shipped the export module\n- Everyone is unblocked\n\nGood meeting."
                    .to_string(),
            ),
            action_items: vec![
                (
                    "Send the recap".to_string(),
                    Some("Alex".to_string()),
                    Some("Friday".to_string()),
                    false,
                ),
                ("Book the venue".to_string(), None, None, true),
            ],
            transcript: vec![
                ("0:00".to_string(), "You".to_string(), "Let's get started.".to_string()),
                (
                    "0:05".to_string(),
                    "Speaker 1".to_string(),
                    "Sounds good.".to_string(),
                ),
            ],
        }
    }

    // -- safe_file_stem / export_file_name --------------------------------

    #[test]
    fn file_names_lose_path_separators_and_odd_characters() {
        assert_eq!(
            safe_file_stem("Q3 review / budget", "2026-08-19T10:00:00Z"),
            "2026-08-19 Q3 review - budget"
        );
        assert_eq!(
            safe_file_stem("../../etc/passwd", "2026-01-02T00:00:00Z"),
            "2026-01-02 etc-passwd"
        );
        assert!(!safe_file_stem("...", "2026-01-02T00:00:00Z").contains('/'));
    }

    #[test]
    fn an_empty_title_still_produces_a_name() {
        assert_eq!(
            safe_file_stem("", "2026-08-19T10:00:00Z"),
            "2026-08-19 Meeting"
        );
        assert_eq!(
            safe_file_stem("   ", "2026-08-19T10:00:00Z"),
            "2026-08-19 Meeting"
        );
    }

    #[test]
    fn very_long_titles_are_cut() {
        let long = "a".repeat(500);
        let stem = safe_file_stem(&long, "2026-08-19T10:00:00Z");
        assert!(stem.chars().count() <= 72, "{}", stem.chars().count());
    }

    #[test]
    fn export_file_name_matches_zero_jargon_convention() {
        let name = export_file_name(
            "Q3 planning",
            "2026-08-19T10:00:00Z",
            ExportFormat::Markdown,
        );
        assert_eq!(name, "Echo — Q3 planning — 2026-08-19.md");
    }

    #[test]
    fn export_file_name_never_contains_path_separators() {
        let name = export_file_name(
            "../../etc/passwd",
            "2026-01-02T00:00:00Z",
            ExportFormat::Docx,
        );
        assert!(!name.contains('/'));
        assert!(!name.contains('\\'));
        assert!(name.ends_with(".docx"));
    }

    // -- validate_destination ----------------------------------------------

    #[test]
    fn destination_must_be_absolute() {
        let err = validate_destination(Path::new("relative/file.md")).unwrap_err();
        assert!(matches!(err, ExportError::Write(_)));
    }

    #[test]
    fn destination_cannot_walk_out_of_its_folder() {
        let err =
            validate_destination(Path::new("/tmp/echo-exports/../../etc/passwd")).unwrap_err();
        assert!(matches!(err, ExportError::Write(_)));
    }

    #[test]
    fn a_clean_absolute_destination_is_accepted() {
        validate_destination(Path::new("/tmp/echo-exports/meeting.md")).unwrap();
    }

    // -- markdown golden file -------------------------------------------

    #[test]
    fn markdown_render_matches_golden_output() {
        let doc = sample_doc();
        let rendered = render_text(&doc, ExportFormat::Markdown).unwrap();
        let expected = "\
# Q3 planning

_August 19, 2026 · 42 min_

## Recap

## Highlights

- Shipped the export module
- Everyone is unblocked

Good meeting.

## Action items

- [ ] Send the recap (Alex, Friday)
- [x] Book the venue

## Transcript

**[0:00] You:** Let's get started.

**[0:05] Speaker 1:** Sounds good.
";
        assert_eq!(rendered, expected);
    }

    #[test]
    fn markdown_render_skips_empty_sections() {
        let doc = ExportDocument {
            title: "Quick sync".to_string(),
            subtitle: String::new(),
            recap_md: Some("All good.".to_string()),
            action_items: Vec::new(),
            transcript: Vec::new(),
        };
        let rendered = render_text(&doc, ExportFormat::Markdown).unwrap();
        assert!(rendered.contains("## Recap"));
        assert!(!rendered.contains("## Action items"));
        assert!(!rendered.contains("## Transcript"));
        assert!(!rendered.contains('_'));
    }

    #[test]
    fn plain_text_needs_a_transcript() {
        let doc = ExportDocument {
            title: "Quick sync".to_string(),
            subtitle: String::new(),
            recap_md: Some("All good.".to_string()),
            action_items: Vec::new(),
            transcript: Vec::new(),
        };
        let err = render_text(&doc, ExportFormat::Text).unwrap_err();
        assert!(matches!(err, ExportError::Empty));
    }

    #[test]
    fn plain_text_has_no_markdown_syntax() {
        let doc = sample_doc();
        let rendered = render_text(&doc, ExportFormat::Text).unwrap();
        assert!(!rendered.contains('#'));
        assert!(!rendered.contains("**"));
        assert!(rendered.contains("[0:00] You: Let's get started."));
    }

    // -- DOCX structural validity -----------------------------------------

    #[tokio::test]
    async fn docx_export_produces_an_openable_document() {
        let doc = sample_doc();
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("meeting.docx");

        let bytes_written = write_docx(&doc, &path).await.unwrap();
        assert!(bytes_written > 0);

        let raw = std::fs::read(&path).unwrap();
        assert_eq!(raw.len() as u64, bytes_written);

        // Structural validity: docx-rs can parse its own ZIP+XML back out.
        let parsed = read_docx(&raw).expect("docx-rs should be able to read back what it wrote");
        assert!(!parsed.document.children.is_empty());

        // The document text should carry the title, an action item, and a
        // transcript line, so a person opening it in Word sees real content.
        let mut all_text = String::new();
        collect_docx_text(&parsed, &mut all_text);
        assert!(all_text.contains("Q3 planning"));
        assert!(all_text.contains("Send the recap"));
        assert!(all_text.contains("Let's get started"));
    }

    #[tokio::test]
    async fn docx_export_rejects_traversal_destinations() {
        let doc = sample_doc();
        let err = write_docx(&doc, Path::new("/tmp/../etc/echo.docx"))
            .await
            .unwrap_err();
        assert!(matches!(err, ExportError::Write(_)));
    }

    /// Walk every run's text out of a parsed docx, ignoring the details of
    /// the document tree shape (only structural validity matters here).
    fn collect_docx_text(docx: &docx_rs::Docx, out: &mut String) {
        // docx-rs's Debug output includes every run's text content, which is
        // enough to assert on without depending on its full internal tree
        // API (paragraph/table child enums churn between versions).
        out.push_str(&format!("{:?}", docx.document));
    }

    // -- PDF decision --------------------------------------------------

    #[tokio::test]
    async fn pdf_export_reports_unavailable_not_a_panic_or_empty_file() {
        let doc = sample_doc();
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("meeting.pdf");
        let err = write_pdf(&doc, &path).await.unwrap_err();
        assert!(matches!(err, ExportError::PdfUnavailable(_)));
        assert!(!path.exists());
    }

    // -- silence unused-import lint for types only referenced by name ----
    #[allow(dead_code)]
    fn _type_smoke(_j: _Job, _m: Meeting, _s: Summary, _a: ActionItem, _st: MeetingStatus) {}
}
