//! Local diagnostics. No telemetry, ever.
//!
//! A bounded rotating log file the person can export from Settings → Advanced
//! (DESIGN §4, review finding 40). It records what helps explain a bad
//! recording: which capture backend came up, device changes, queue overflows,
//! timings, why the GPU was not used.
//!
//! It records none of the following, and [`redact`] is the backstop rather than
//! the plan:
//! * transcript text
//! * recap text
//! * keys or passwords
//! * absolute paths inside the person's home directory
//!
//! Five daily files, then the oldest goes. A log that grows forever is a bug.

use std::io::{self, Write};
use std::path::Path;
use std::sync::OnceLock;

use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::fmt::MakeWriter;

/// How many rotated files to keep.
pub const MAX_LOG_FILES: usize = 5;

/// Words that mean the rest of the value must not be written.
const SENSITIVE_KEYS: [&str; 6] = [
    "key",
    "token",
    "secret",
    "password",
    "authorization",
    "bearer",
];

static GUARD: OnceLock<WorkerGuard> = OnceLock::new();

/// Install the subscriber. Safe to call twice; the second call does nothing.
pub fn init(log_dir: &Path) -> io::Result<()> {
    if GUARD.get().is_some() {
        return Ok(());
    }
    std::fs::create_dir_all(log_dir)?;

    let appender = tracing_appender::rolling::Builder::new()
        .rotation(tracing_appender::rolling::Rotation::DAILY)
        .filename_prefix("echo")
        .filename_suffix("log")
        .max_log_files(MAX_LOG_FILES)
        .build(log_dir)
        .map_err(|e| io::Error::other(e.to_string()))?;

    let (non_blocking, guard) = tracing_appender::non_blocking(appender);
    let _ = GUARD.set(guard);

    let level = if cfg!(debug_assertions) {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };

    let subscriber = tracing_subscriber::fmt()
        .with_writer(Redacting(non_blocking))
        .with_ansi(false)
        .with_target(true)
        .with_max_level(level)
        .finish();

    // Another test in the same process may have set one already.
    let _ = tracing::subscriber::set_global_default(subscriber);
    Ok(())
}

/// Wraps a writer and scrubs each line before it lands on disk.
struct Redacting<M>(M);

impl<'a, M: MakeWriter<'a>> MakeWriter<'a> for Redacting<M> {
    type Writer = RedactingWriter<M::Writer>;

    fn make_writer(&'a self) -> Self::Writer {
        RedactingWriter(self.0.make_writer())
    }
}

struct RedactingWriter<W>(W);

impl<W: Write> Write for RedactingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let text = String::from_utf8_lossy(buf);
        self.0.write_all(redact(&text).as_bytes())?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.0.flush()
    }
}

/// Replace home paths with `~` and mask the value after any sensitive key.
///
/// Deliberately crude: this is a safety net for a mistake somewhere else, so it
/// prefers over-masking to cleverness.
pub fn redact(line: &str) -> String {
    let mut out = mask_sensitive(line);
    if let Some(home) = dirs::home_dir() {
        let home = home.to_string_lossy().into_owned();
        if !home.is_empty() {
            out = out.replace(&home, "~");
        }
    }
    out
}

fn mask_sensitive(line: &str) -> String {
    let lower = line.to_ascii_lowercase();
    let mut out = String::with_capacity(line.len());
    let bytes = line.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        let mut matched = None;
        for needle in SENSITIVE_KEYS {
            if lower[i..].starts_with(needle) {
                matched = Some(needle.len());
                break;
            }
        }
        match matched {
            Some(len) => {
                out.push_str(&line[i..i + len]);
                i += len;
                // Copy the separator run, then swallow the rest of the line.
                // Over-masking on purpose: guessing where a value ends is how
                // half a key ends up in a log file.
                while i < bytes.len() && matches!(bytes[i], b'=' | b':' | b' ') {
                    out.push(bytes[i] as char);
                    i += 1;
                }
                let start = i;
                while i < bytes.len() && bytes[i] != b'\n' && bytes[i] != b'\r' {
                    i += 1;
                }
                if i > start {
                    out.push_str("<redacted>");
                }
            }
            None => {
                // Advance one char, not one byte, so multi-byte text survives.
                let ch_len = line[i..].chars().next().map(char::len_utf8).unwrap_or(1);
                out.push_str(&line[i..i + ch_len]);
                i += ch_len;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_and_tokens_never_reach_the_file() {
        assert_eq!(mask_sensitive("api_key=abc123def"), "api_key=<redacted>");
        assert_eq!(mask_sensitive("token: eyJhbGciOi"), "token: <redacted>");
        assert_eq!(
            mask_sensitive("Authorization: Bearer sk-123"),
            "Authorization: <redacted>"
        );
        assert_eq!(
            mask_sensitive("password=\"hunter2\""),
            "password=<redacted>"
        );
        // Over-masking: everything after the key goes, including later fields.
        assert_eq!(
            mask_sensitive("api_key=abc123 device=Built-in"),
            "api_key=<redacted>"
        );
    }

    #[test]
    fn ordinary_lines_pass_through_unchanged() {
        let line = "capture backend=metal channels=2 dropped=0";
        assert_eq!(mask_sensitive(line), line);
    }

    #[test]
    fn non_ascii_text_is_not_corrupted() {
        let line = "device=Mikrofon (Büro) état=ok 会議";
        assert_eq!(mask_sensitive(line), line);
    }

    #[test]
    fn home_paths_are_shortened() {
        if let Some(home) = dirs::home_dir() {
            let line = format!("writing to {}/Echo/recordings/a.flac", home.display());
            let out = redact(&line);
            assert!(out.starts_with("writing to ~/Echo/recordings"), "{out}");
            assert!(!out.contains(&home.to_string_lossy().into_owned()));
        }
    }

    #[test]
    fn a_bare_sensitive_word_at_the_end_of_a_line_is_left_alone() {
        assert_eq!(mask_sensitive("key"), "key");
        assert_eq!(mask_sensitive("key "), "key ");
        // Prose containing the word loses its tail. That is the trade.
        assert_eq!(mask_sensitive("the key is missing"), "the key <redacted>");
    }

    #[test]
    fn each_line_is_masked_on_its_own() {
        assert_eq!(
            mask_sensitive("api_key=abc\ndevice=Built-in\n"),
            "api_key=<redacted>\ndevice=Built-in\n"
        );
    }
}
