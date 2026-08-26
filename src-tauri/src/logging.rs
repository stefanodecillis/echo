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
//!
//! The level is chosen three ways, checked in order: `RUST_LOG` (a developer
//! at a terminal), then a `log-filter` file dropped next to the logs (a
//! Finder-launched .app has no terminal and no way to set an env var — this
//! is the only lever a person capturing diagnostics for us actually has), then
//! a built-in default (debug in dev, info in release). See
//! [`filter_directives`]. Whatever it lands on, [`init`] writes it into the
//! log itself: an empty file has to be distinguishable from a quiet app.
//!
//! Turning one target up never turns the rest off — [`with_default_floor`]
//! keeps the built-in level underneath anything asked for. Only a bare level
//! in the line (`warn`, `off`) lowers the floor, and then only because someone
//! asked for that in so many words.
//!
//! The appender is deliberately lossy: `tracing_appender::non_blocking` drops
//! lines rather than block the caller when its channel is full. A meeting
//! that stutters because the disk is slow and the log queue backed up is a
//! worse bug than a gap in the log. The one line that must never be dropped —
//! a panic — is also written straight to `panics.log` by hand, past the
//! channel, past the filter, past the level (see [`install_panic_hook`]).

use std::collections::HashMap;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::EnvFilter;

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

    let (source, requested) = filter_directives(log_dir);
    let directives = with_default_floor(&requested);
    // `parse`, not `parse_lossy`: lossy parsing throws the broken directive
    // away and says so on a stderr a bundled .app does not have. If the line
    // does not parse we want to know *in the log*, and we want the default
    // level rather than whatever half of it survived.
    let (filter, rejected) = match EnvFilter::builder().parse(&directives) {
        Ok(filter) => (filter, None),
        Err(e) => (EnvFilter::new(default_directive()), Some(e.to_string())),
    };

    let subscriber = tracing_subscriber::fmt()
        .with_writer(Redacting(non_blocking))
        .with_ansi(false)
        .with_target(true)
        .with_env_filter(filter)
        .finish();

    // Another test in the same process may have set one already.
    let _ = tracing::subscriber::set_global_default(subscriber);

    // From here on, a panic anywhere writes one line instead of vanishing
    // into a terminal nobody is watching (2026-08-24 incident: the event bus
    // poisoned mid-meeting and the only trace of why was gone with the
    // process's stderr).
    let _ = PANIC_LOG_PATH.set(log_dir.join(PANIC_LOG_FILE));
    install_panic_hook();

    // Say out loud which lever won and what it resolved to. Someone we asked
    // to drop a `log-filter` file next to the logs has no other way to find
    // out their line was a typo, and a log that came back empty because of it
    // looks exactly like a log from an app that had nothing to say.
    tracing::info!(
        source = source.name(),
        directives = %directives,
        "log filter in force"
    );
    if let Some(problem) = rejected {
        tracing::warn!(
            requested = %requested,
            problem = %problem,
            "that log filter does not parse, so the built-in default is in force instead"
        );
    }
    Ok(())
}

/// Which of the three levers actually decided the level. Carried out of
/// [`choose_directives`] so `init` can name it: "info because the file said
/// so" and "info because nobody asked for anything" read identically in the
/// log otherwise, and they need very different next steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FilterSource {
    Env,
    File,
    Default,
}

impl FilterSource {
    fn name(self) -> &'static str {
        match self {
            FilterSource::Env => "RUST_LOG",
            FilterSource::File => "log-filter file",
            FilterSource::Default => "built-in default",
        }
    }
}

/// Where the max log level comes from, in priority order: `RUST_LOG`, then
/// the first non-blank line of `<log_dir>/log-filter`, then a built-in
/// default. Split from the env/file I/O so the decision itself is a pure,
/// testable function — [`choose_directives`].
fn filter_directives(log_dir: &Path) -> (FilterSource, String) {
    let from_env = std::env::var("RUST_LOG").ok();
    let from_file = std::fs::read_to_string(log_dir.join("log-filter")).ok();
    choose_directives(from_env.as_deref(), from_file.as_deref())
}

/// The three-way fallback itself, with the env var and file contents already
/// read — kept free of I/O so tests don't have to fight the process over a
/// global env var.
fn choose_directives(
    env_rust_log: Option<&str>,
    filter_file: Option<&str>,
) -> (FilterSource, String) {
    if let Some(trimmed) = env_rust_log.map(str::trim).filter(|s| !s.is_empty()) {
        return (FilterSource::Env, trimmed.to_string());
    }
    // One line: a person hand-editing this file is not writing a directive
    // grammar tutorial, and EnvFilter only wants the first line anyway.
    if let Some(trimmed) = filter_file
        .and_then(|f| f.lines().next())
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        return (FilterSource::File, trimmed.to_string());
    }
    (FilterSource::Default, default_directive().to_string())
}

/// Put the built-in level underneath whatever was asked for, unless the line
/// already says what everything else should do.
///
/// `EnvFilter` drops its default level the moment the string parses into any
/// directive at all, so `echo_lib::session=debug` — the very line we ask
/// people to put in `log-filter` when we need more detail — used to mean "and
/// nothing else logs, ever": turning logging *up* produced strictly less than
/// the shipped build, panic hook and audio warnings included. A misspelling
/// was worse, because it is not an error: `tarce` is a perfectly valid
/// directive for a target nobody has, and it silenced the whole app.
///
/// A bare level anywhere in the line (`warn`, `off`, `warn,echo_lib=debug`) is
/// left exactly as written — that person is setting the floor deliberately,
/// including downwards, and we don't second-guess it.
fn with_default_floor(requested: &str) -> String {
    let sets_own_floor = requested
        .split(',')
        .map(str::trim)
        .any(|part| !part.is_empty() && part.parse::<LevelFilter>().is_ok());
    if sets_own_floor {
        requested.to_string()
    } else {
        format!("{},{requested}", default_directive())
    }
}

fn default_directive() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "info"
    }
}

/// Rate-limits a repeating log line by key: the first sighting of a key is
/// always admitted, later ones are swallowed until `interval` has passed, at
/// which point the next call is admitted carrying how many were swallowed in
/// between. Built for once-per-stream warnings (a stuck audio read, a dead
/// event transport, a panicking sink) that would otherwise write one line per
/// buffer/frame/event and drown the file that is supposed to explain them.
pub struct Throttle {
    interval: Duration,
    windows: Mutex<HashMap<String, Window>>,
}

struct Window {
    last: Instant,
    suppressed: u64,
}

impl Throttle {
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            windows: Mutex::new(HashMap::new()),
        }
    }

    /// `Some(suppressed)` when the caller should log now; `None` when this
    /// key is still inside its cooldown window and should stay quiet.
    pub fn admit(&self, key: &str) -> Option<u64> {
        // A poisoned throttle must not take rate-limiting down with it —
        // that would turn "one panic somewhere" into "unlimited log spam
        // everywhere", the opposite of what this type is for.
        let mut windows = self.windows.lock().unwrap_or_else(|e| e.into_inner());

        // Deliberately crude safety net, not a real LRU: keys here are meant
        // to be a small fixed set (a file:line, an event name). If some
        // future caller keys by something unbounded — a message, an id —
        // this bounds the damage to "throttling briefly forgets its
        // history" instead of "the map grows until the process falls over".
        if windows.len() > 64 {
            windows.clear();
        }

        let now = Instant::now();
        match windows.get_mut(key) {
            None => {
                windows.insert(
                    key.to_string(),
                    Window {
                        last: now,
                        suppressed: 0,
                    },
                );
                Some(0)
            }
            Some(window) => {
                if now.duration_since(window.last) >= self.interval {
                    let suppressed = window.suppressed;
                    window.last = now;
                    window.suppressed = 0;
                    Some(suppressed)
                } else {
                    window.suppressed += 1;
                    None
                }
            }
        }
    }
}

/// How long a repeated panic at the same call site stays quiet after the
/// first line. Long enough that a hot panic loop doesn't fill the log; short
/// enough that a second, unrelated crash a minute later still gets its own
/// line.
const PANIC_THROTTLE_INTERVAL: Duration = Duration::from_secs(30);

static PANIC_HOOK_INSTALLED: OnceLock<()> = OnceLock::new();

/// The file panics are written to by hand, next to the rotating logs so the
/// diagnostics export picks it up with everything else.
const PANIC_LOG_FILE: &str = "panics.log";

/// This one does not rotate by date — it starts over when it gets too big.
/// Panics are meant to be rare; a hot panic loop must still not be able to
/// fill the disk, because a log that grows forever is a bug (module docs).
const MAX_PANIC_LOG_BYTES: u64 = 64 * 1024;

/// Set by [`init`] once the log directory is known. Until then the hook still
/// works, it just has nowhere but tracing to put the line — which is the
/// situation in tests, and fine there.
static PANIC_LOG_PATH: OnceLock<PathBuf> = OnceLock::new();

/// Write one panic line straight to disk, synchronously.
///
/// Everything else in this module goes through `tracing_appender`'s
/// non-blocking channel, which drops lines when it is backed up and is flushed
/// by a `WorkerGuard` that lives in a `static` and therefore never runs its
/// destructor. Both are the right trade for a log line about a slow disk. They
/// are the wrong trade for the one line that explains why the process is
/// about to die: a panic that coincides with a backlog, or that takes the
/// process down before the worker thread wakes up, is exactly the panic we
/// most need to read afterwards. So this bypasses the channel entirely.
///
/// No `sync_all`: once `write_all` returns, the bytes are the kernel's problem
/// and survive the process dying. Only the machine losing power loses them,
/// and at that point the panic is not the story.
fn append_panic_line(path: &Path, line: &str) {
    let start_over = std::fs::metadata(path)
        .map(|m| m.len() >= MAX_PANIC_LOG_BYTES)
        .unwrap_or(false);
    let opened = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(!start_over)
        .truncate(start_over)
        .open(path);
    let Ok(mut file) = opened else {
        // Nowhere to complain to: we are already inside the hook that exists
        // because the usual places failed.
        return;
    };
    let _ = file.write_all(redact(line).as_bytes());
    let _ = file.flush();
}

/// Install a panic hook that writes thread, location and message to
/// `panics.log` and to the log at ERROR, before chaining to whatever hook was
/// already registered (Rust's default stderr printer, or a future crash
/// reporter — either way it still runs, this just stops being the only place
/// the panic went).
///
/// Both, not one: the ERROR line puts the panic in chronological order next to
/// what the app was doing at the time, and the hand-written file is the copy
/// that survives a dropped channel, an unflushed worker, or a `log-filter`
/// line that turned the level down (see [`append_panic_line`]).
///
/// Idempotent: the real fix for "no panic hook anywhere" only needs to
/// happen once, but `init` can run more than once in a process (tests, and
/// the "safe to call twice" contract above), and the second call must not
/// stack a duplicate hook that logs the same panic twice.
pub fn install_panic_hook() {
    if PANIC_HOOK_INSTALLED.set(()).is_err() {
        return;
    }
    static THROTTLE: OnceLock<Throttle> = OnceLock::new();
    let throttle = THROTTLE.get_or_init(|| Throttle::new(PANIC_THROTTLE_INTERVAL));

    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let thread = std::thread::current();
        let thread_name = thread.name().unwrap_or("<unnamed>");
        let location = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "<unknown location>".to_string());
        let message = panic_message(info.payload());

        if let Some(suppressed) = throttle.admit(&location) {
            // Disk first, tracing second: if this panic is the one that ends
            // the process, the copy that made it out of this thread is the
            // only one that will be there afterwards.
            if let Some(path) = PANIC_LOG_PATH.get() {
                let at = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
                append_panic_line(
                    path,
                    &format!(
                        "{at} PANIC thread={thread_name} suppressed={suppressed} \
                         at {location}: {message}\n"
                    ),
                );
            }
            tracing::error!(
                thread = thread_name,
                location = %location,
                suppressed,
                "panic: {message}"
            );
        }

        previous(info);
    }));
}

/// A panic payload is `Any`; in practice it is almost always the `&str` from
/// `panic!("literal")` or the `String` from `panic!("{}", x)`. Anything else
/// is a payload some other crate chose to pass through `panic_any` — we
/// don't know its shape, so we say so rather than guess.
pub(crate) fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
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

    #[test]
    fn throttle_admits_first_then_counts_suppressed_then_reopens_after_interval() {
        let throttle = Throttle::new(Duration::from_millis(20));
        assert_eq!(
            throttle.admit("k"),
            Some(0),
            "first sighting of a key is never throttled"
        );
        assert_eq!(throttle.admit("k"), None);
        assert_eq!(throttle.admit("k"), None);
        std::thread::sleep(Duration::from_millis(25));
        assert_eq!(
            throttle.admit("k"),
            Some(2),
            "re-admits once the interval passes, reporting what it swallowed"
        );
    }

    #[test]
    fn throttle_keys_are_independent() {
        let throttle = Throttle::new(Duration::from_secs(30));
        assert_eq!(throttle.admit("a"), Some(0));
        assert_eq!(throttle.admit("a"), None);
        // "b" has never been seen, so it gets its own first-sighting pass
        // even though "a" is deep inside its cooldown window.
        assert_eq!(throttle.admit("b"), Some(0));
    }

    #[test]
    fn filter_directives_default_when_nothing_is_set() {
        assert_eq!(
            choose_directives(None, None),
            (FilterSource::Default, default_directive().to_string())
        );
        // Whitespace-only env/file content is "not set" too.
        assert_eq!(
            choose_directives(Some("  "), Some("\n  \n")),
            (FilterSource::Default, default_directive().to_string())
        );
    }

    #[test]
    fn filter_directives_env_wins_over_file() {
        assert_eq!(
            choose_directives(Some("echo_lib=trace"), Some("info")),
            (FilterSource::Env, "echo_lib=trace".to_string())
        );
    }

    #[test]
    fn filter_directives_falls_back_to_file_then_default() {
        assert_eq!(
            choose_directives(None, Some("debug\nsome comment nobody asked for")),
            (FilterSource::File, "debug".to_string()),
            "only the first line of the filter file is used"
        );
        assert_eq!(
            choose_directives(Some(""), Some("")),
            (FilterSource::Default, default_directive().to_string())
        );
    }

    #[test]
    fn every_source_can_name_itself_for_the_log() {
        // The whole point of carrying the source is that the log line says
        // which lever moved, so none of these may be blank.
        for source in [FilterSource::Env, FilterSource::File, FilterSource::Default] {
            assert!(!source.name().is_empty(), "{source:?}");
        }
    }

    #[test]
    fn turning_one_target_up_leaves_the_rest_where_they_were() {
        assert_eq!(
            with_default_floor("echo_lib::session=debug"),
            format!("{},echo_lib::session=debug", default_directive()),
        );
        // A stray word is a *valid* directive for a target nobody has, which
        // is why this one silenced the app instead of complaining.
        assert_eq!(
            with_default_floor("tarce"),
            format!("{},tarce", default_directive()),
        );
    }

    #[test]
    fn a_line_that_states_its_own_level_is_left_alone() {
        assert_eq!(with_default_floor("warn"), "warn");
        assert_eq!(with_default_floor("off"), "off");
        assert_eq!(
            with_default_floor("WARN,echo_lib=debug"),
            "WARN,echo_lib=debug"
        );
        assert_eq!(
            with_default_floor("echo_lib=debug,warn"),
            "echo_lib=debug,warn"
        );
        // Numbers are levels to EnvFilter too (3 == info).
        assert_eq!(with_default_floor("3"), "3");
    }

    #[test]
    fn a_target_scoped_filter_still_lets_the_rest_of_the_app_speak() {
        // The regression this floor exists for, end to end through a real
        // EnvFilter: the raw line drops an ERROR from an unrelated target,
        // the floored one keeps it.
        fn errors_seen(directives: &str) -> String {
            let buf = std::sync::Arc::new(Mutex::new(Vec::new()));
            let filter = EnvFilter::builder().parse(directives).unwrap();
            let subscriber = tracing_subscriber::fmt()
                .with_writer(SharedWriter(buf.clone()))
                .with_ansi(false)
                .with_env_filter(filter)
                .finish();
            tracing::subscriber::with_default(subscriber, || {
                tracing::error!("the sink panicked and the bus is gone");
            });
            let written = buf.lock().unwrap_or_else(|e| e.into_inner()).clone();
            String::from_utf8(written).unwrap()
        }

        assert!(
            !errors_seen("echo_lib::session=debug").contains("the bus is gone"),
            "this is the bug: a bare target directive drops every other target"
        );
        assert!(
            errors_seen(&with_default_floor("echo_lib::session=debug")).contains("the bus is gone"),
            "asking for more detail somewhere must never mean less detail everywhere"
        );
        assert!(
            errors_seen(&with_default_floor("tarce")).contains("the bus is gone"),
            "a typo in the filter file must not be able to silence a panic"
        );
    }

    #[test]
    fn a_panic_line_reaches_the_disk_without_the_log_channel() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(PANIC_LOG_FILE);

        append_panic_line(&path, "first panic\n");
        append_panic_line(&path, "second panic\n");
        let written = std::fs::read_to_string(&path).unwrap();
        assert_eq!(written, "first panic\nsecond panic\n");

        // Same redaction as every other line: a panic message is as likely to
        // carry a home path as anything else.
        if let Some(home) = dirs::home_dir() {
            append_panic_line(
                &path,
                &format!("panicked reading {}/Echo/db\n", home.display()),
            );
            let written = std::fs::read_to_string(&path).unwrap();
            assert!(written.contains("panicked reading ~/Echo/db"), "{written}");
        }
    }

    #[test]
    fn the_panic_file_starts_over_instead_of_growing_forever() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(PANIC_LOG_FILE);
        std::fs::write(&path, vec![b'x'; MAX_PANIC_LOG_BYTES as usize + 1]).unwrap();

        append_panic_line(&path, "the panic after the flood\n");

        let written = std::fs::read_to_string(&path).unwrap();
        assert_eq!(written, "the panic after the flood\n");
    }

    #[test]
    fn panic_message_reads_str_and_string_payloads_and_admits_the_rest() {
        let str_payload: &(dyn std::any::Any + Send) = &"boom";
        assert_eq!(panic_message(str_payload), "boom");

        let string_payload: &(dyn std::any::Any + Send) = &String::from("kaboom");
        assert_eq!(panic_message(string_payload), "kaboom");

        let other_payload: &(dyn std::any::Any + Send) = &42i32;
        assert_eq!(panic_message(other_payload), "<non-string panic payload>");
    }

    /// Writes every line into a buffer shared with the test, so the test can
    /// read back what the subscriber actually produced.
    #[derive(Clone)]
    struct SharedWriter(std::sync::Arc<Mutex<Vec<u8>>>);

    impl Write for SharedWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl<'a> MakeWriter<'a> for SharedWriter {
        type Writer = SharedWriter;

        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    #[test]
    fn a_caught_panic_writes_an_error_line_with_message_thread_and_location() {
        let buf = std::sync::Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_writer(SharedWriter(buf.clone()))
            .with_ansi(false)
            .finish();

        // Each cargo test runs its own thread named after the test function,
        // so this doubles as "the hook reports the right thread".
        let this_thread = std::thread::current()
            .name()
            .unwrap_or("<unnamed>")
            .to_string();

        tracing::subscriber::with_default(subscriber, || {
            install_panic_hook();
            let result = std::panic::catch_unwind(|| {
                panic!("sentinel panic for the logging test");
            });
            assert!(result.is_err(), "the hook must not stop the unwind");
        });

        let output =
            String::from_utf8(buf.lock().unwrap_or_else(|e| e.into_inner()).clone()).unwrap();
        assert!(
            output.contains("sentinel panic for the logging test"),
            "{output}"
        );
        assert!(output.contains(&this_thread), "{output}");
        assert!(output.contains("logging.rs"), "{output}");
        // At ERROR, specifically: a hook quietly downgraded to warn would leave
        // the next incident invisible at the release level all over again.
        assert!(output.contains("ERROR"), "{output}");
    }
}
