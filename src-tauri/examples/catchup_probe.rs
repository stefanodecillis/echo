//! Catch-up probe: run the **real** catch-up pass over the first few minutes of
//! a real meeting's chunks, against the **real** installed weights, and print
//! what every speech stretch handed to whisper.cpp and what came back.
//!
//!   cargo run --release --example catchup_probe -- <meeting-id> [minutes]
//!
//! Nothing in the app's own storage is written: the chunks are copied to a
//! temporary directory, the database is a fresh temporary file, and the model
//! rows are pointed at the installed files read-only.
//!
//! This exists because a field meeting produced 281 speech stretches and 279
//! `Generic whisper error … code: -6` failures, and a fix must be proved against
//! the audio that produced it rather than against a synthetic tone.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

use echo_lib::asr::catchup::{self, CatchUpOptions, DiskAudio, Transcriber};
use echo_lib::asr::engine::EngineWorker;
use echo_lib::asr::models;
use echo_lib::asr::{AsrError, TranscribeJob, Transcription};
use echo_lib::db::{self, repo};
use echo_lib::types::{AssetKind, Channel, TranscriptQuery};

const CHUNK_MS: i64 = 30_000;
const DEFAULT_MEETING: &str = "13731077-453a-4c75-8e6e-5586fe2ed019";
const DEFAULT_MINUTES: i64 = 4;

/// One decode attempt, as it happened.
struct Attempt {
    t_start_ms: i64,
    duration_ms: i64,
    samples: usize,
    nonzero: usize,
    peak: f32,
    pinned: Option<String>,
    outcome: String,
}

/// Wraps the real engine so every decode it is asked for is reported.
struct Probe<'a> {
    engine: &'a EngineWorker,
    attempts: Mutex<Vec<Attempt>>,
    ok: AtomicU32,
    failed: AtomicU32,
}

impl Transcriber for Probe<'_> {
    async fn transcribe(&self, job: TranscribeJob) -> Result<Transcription, AsrError> {
        let t_start_ms = job.t_start_ms;
        let duration_ms = job.duration_ms();
        let samples = job.samples.len();
        let nonzero = job.samples.iter().filter(|s| **s != 0.0).count();
        let peak = job.samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        let pinned = job.language_hint.clone();

        let answer = self.engine.transcribe(job).await;
        let outcome = match &answer {
            Ok(t) if t.text.trim().is_empty() => "ok (nothing said)".to_string(),
            Ok(t) => format!(
                "ok conf={:.2} lang={} {:?}",
                t.avg_confidence.unwrap_or(0.0),
                t.language.as_deref().unwrap_or("?"),
                first_words(&t.text, 8)
            ),
            Err(e) => format!("ERR {e}"),
        };
        if answer.is_ok() {
            self.ok.fetch_add(1, Ordering::SeqCst);
        } else {
            self.failed.fetch_add(1, Ordering::SeqCst);
        }
        self.attempts.lock().unwrap().push(Attempt {
            t_start_ms,
            duration_ms,
            samples,
            nonzero,
            peak,
            pinned,
            outcome,
        });
        answer
    }

    fn settled_language(&self, meeting_id: &str) -> Option<String> {
        self.engine.settled_language(meeting_id)
    }
}

fn first_words(text: &str, n: usize) -> String {
    let mut out: Vec<&str> = text.split_whitespace().take(n).collect();
    if text.split_whitespace().count() > n {
        out.push("…");
    }
    out.join(" ")
}

fn app_support() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME");
    Path::new(&home).join("Library/Application Support/Echo")
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let meeting = args.next().unwrap_or_else(|| DEFAULT_MEETING.to_string());
    let minutes: i64 = args
        .next()
        .and_then(|m| m.parse().ok())
        .unwrap_or(DEFAULT_MINUTES);
    let chunks_wanted = (minutes * 60_000 / CHUNK_MS).max(1);

    let support = app_support();
    let source = support.join("recordings").join(&meeting);
    let speech = support.join("speech");

    let tmp = tempfile::tempdir()?;
    let audio_dir = tmp.path().join("audio");
    std::fs::create_dir_all(&audio_dir)?;

    // --- copy the audio (read-only source, never written) -----------------
    let mut copied: Vec<(i64, PathBuf)> = Vec::new();
    for seq in 0..chunks_wanted {
        let name = format!("mic-{seq:06}.wav");
        let from = source.join(&name);
        if !from.is_file() {
            break;
        }
        let to = audio_dir.join(&name);
        std::fs::copy(&from, &to)?;
        copied.push((seq, to));
    }
    if copied.is_empty() {
        return Err(format!("no mic chunks under {}", source.display()).into());
    }
    println!(
        "copied {} chunk(s) ({} s) from {}",
        copied.len(),
        copied.len() as i64 * CHUNK_MS / 1_000,
        source.display()
    );

    // --- a throwaway database pointed at the installed weights ------------
    let db = db::connect(&tmp.path().join("echo.db")).await?;
    models::sync_catalog(&db).await?;
    repo::set_setting(&db, echo_lib::settings::keys::ACCURACY_LEVEL_ID, "everyday").await?;
    for (id, path) in [
        (
            "speech-large-v3-turbo",
            speech.join("ggml-large-v3-turbo.bin"),
        ),
        (
            "speech-accelerator-large-v3-turbo",
            speech.join("ggml-large-v3-turbo-encoder.mlmodelc"),
        ),
        (
            "speech-detector-silero-v5",
            speech.join("silero-vad-v5.1.2.onnx"),
        ),
    ] {
        if !path.exists() {
            return Err(format!("{} is not installed at {}", id, path.display()).into());
        }
        repo::set_model_installed(&db, id, true, Some(&path.to_string_lossy())).await?;
    }
    println!(
        "weights: {}",
        models::installed_path(&db, AssetKind::Speech)
            .await?
            .unwrap()
            .display()
    );
    println!(
        "detector: {}",
        models::installed_path(&db, AssetKind::SpeechDetector)
            .await?
            .unwrap()
            .display()
    );

    // --- the meeting and its committed chunks -----------------------------
    let created = repo::create_meeting(
        &db,
        "catch-up probe",
        &audio_dir.to_string_lossy(),
        Some("probe"),
    )
    .await?;
    // The field meeting had settled on Italian, and that prior is part of the
    // path under test (it decides whether a stretch is decoded once or twice).
    repo::set_meeting_language(&db, &created.id, "it").await?;
    for (seq, path) in &copied {
        let id = repo::insert_chunk(
            &db,
            &created.id,
            Channel::Mic,
            *seq,
            &path.to_string_lossy(),
            seq * CHUNK_MS,
            (seq + 1) * CHUNK_MS,
        )
        .await?;
        repo::commit_chunk(&db, &id, (seq + 1) * CHUNK_MS).await?;
    }

    // --- the real engine --------------------------------------------------
    let engine = EngineWorker::new();
    engine.configure_from_settings(&db).await?;
    let backend = engine.load_now().await?;
    println!(
        "engine: backend={} threads={} accelerator={}",
        backend.active, backend.threads, backend.accelerator_present
    );

    let probe = Probe {
        engine: &engine,
        attempts: Mutex::new(Vec::new()),
        ok: AtomicU32::new(0),
        failed: AtomicU32::new(0),
    };

    let started = std::time::Instant::now();
    let report = catchup::run(
        &probe,
        &db,
        &created.id,
        CatchUpOptions::default(),
        &DiskAudio,
    )
    .await?;
    let elapsed = started.elapsed();

    // --- what happened ----------------------------------------------------
    println!();
    println!(
        "{:>9}  {:>8}  {:>8}  {:>8}  {:>6}  {:>4}  outcome",
        "t_start", "dur_ms", "samples", "nonzero", "peak", "lang"
    );
    for a in probe.attempts.lock().unwrap().iter() {
        println!(
            "{:>9}  {:>8}  {:>8}  {:>8}  {:>6.3}  {:>4}  {}",
            a.t_start_ms,
            a.duration_ms,
            a.samples,
            a.nonzero,
            a.peak,
            a.pinned.as_deref().unwrap_or("-"),
            a.outcome
        );
    }

    let attempts = probe.attempts.lock().unwrap().len();
    println!();
    println!(
        "decode attempts={attempts} ok={} failed={} | stretches with text={} windows_read={} in {:.1}s",
        probe.ok.load(Ordering::SeqCst),
        probe.failed.load(Ordering::SeqCst),
        report.segments_written,
        report.windows_read,
        elapsed.as_secs_f32(),
    );

    let written = repo::get_segments(
        &db,
        &TranscriptQuery {
            meeting_id: created.id.clone(),
            ..Default::default()
        },
    )
    .await?;
    println!("segments written = {}", written.len());
    for s in written.iter().take(4) {
        println!(
            "  [{:>7}..{:>7}] {}",
            s.t_start_ms,
            s.t_end_ms,
            first_words(&s.text, 10)
        );
    }

    engine.shutdown();
    Ok(())
}
