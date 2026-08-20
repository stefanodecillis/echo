//! Catch-up probe: run the **real** catch-up pass over a real meeting's chunks,
//! against the **real** installed weights, and print what every speech stretch
//! handed to whisper.cpp and what came back.
//!
//! ```text
//! cargo run --release --example catchup_probe -- <meeting-id> [minutes|all] [flags]
//!
//!   minutes    how much of the meeting to decode; `all` for the whole thing
//!   --table    one line per decode attempt (private audio: 8 words each)
//!   --dump P   write the segments this pass wrote to P, as JSON
//! ```
//!
//! Nothing in the app's own storage is written: the chunks are copied to a
//! temporary directory, the database is a fresh temporary file, and the model
//! rows are pointed at the installed files read-only.
//!
//! This exists because a field meeting produced 281 speech stretches and 279
//! `Generic whisper error … code: -6` failures, and a fix must be proved against
//! the audio that produced it rather than against a synthetic tone.
//!
//! `--dump` is what makes the coverage this proves reusable: `speakers_probe`
//! seeds its own throwaway database from the file and runs the speaker pass over
//! the same words, without decoding twenty-one minutes of audio a second time.

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
/// How many transcript lines to show. Private audio: a glance is the evidence,
/// not the transcript.
const SAMPLES: usize = 5;
/// Words per sample line.
const SAMPLE_WORDS: usize = 10;

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

/// Total length of a set of spans, with overlaps counted once.
fn covered_ms(mut spans: Vec<(i64, i64)>) -> i64 {
    spans.sort_unstable();
    let mut total = 0i64;
    let mut open: Option<(i64, i64)> = None;
    for (from, to) in spans {
        match open {
            Some((start, end)) if from <= end => open = Some((start, end.max(to))),
            Some((start, end)) => {
                total += end - start;
                open = Some((from, to));
            }
            None => open = Some((from, to)),
        }
    }
    if let Some((start, end)) = open {
        total += end - start;
    }
    total
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut positional: Vec<String> = Vec::new();
    let mut want_table = false;
    let mut dump: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--table" => want_table = true,
            "--dump" => dump = args.next().map(PathBuf::from),
            other => positional.push(other.to_string()),
        }
    }
    let meeting = positional
        .first()
        .cloned()
        .unwrap_or_else(|| DEFAULT_MEETING.to_string());
    // `all` is the whole meeting; a number is that many minutes of it.
    let chunks_wanted = match positional.get(1).map(String::as_str) {
        Some("all") | Some("full") => i64::MAX,
        Some(n) => (n.parse::<i64>().unwrap_or(DEFAULT_MINUTES) * 60_000 / CHUNK_MS).max(1),
        None => (DEFAULT_MINUTES * 60_000 / CHUNK_MS).max(1),
    };

    let support = app_support();
    let source = support.join("recordings").join(&meeting);
    // Where the weights are. `ECHO_PROBE_SPEECH_DIR` lets a probe run against
    // assets in a scratch directory — the app's own folder is read-only while a
    // model change is being proved, and a 3 GB download does not belong in it.
    let speech = std::env::var_os("ECHO_PROBE_SPEECH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| support.join("speech"));

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
    // Point the throwaway table at whatever weights are in `speech`.
    //
    // `ECHO_PROBE_SPEECH_ID` picks which catalogued speech entry to run — the
    // whole point of a model-change probe is running the same audio through two
    // of them and reading the two transcripts side by side. Its Apple companion
    // comes from the catalog's own pairing, so it can never be mismatched.
    let speech_id = std::env::var("ECHO_PROBE_SPEECH_ID")
        .unwrap_or_else(|_| echo_lib::asr::catalog::ids::SPEECH.to_string());
    let speech_entry = echo_lib::asr::catalog::entry(&speech_id)
        .ok_or_else(|| format!("{speech_id} is not in the catalog"))?;
    let mut wanted: Vec<(&str, PathBuf)> =
        vec![(speech_entry.id, speech.join(speech_entry.installed_name()))];
    if let Some(accel) = echo_lib::asr::catalog::accelerator_for(speech_entry.id) {
        wanted.push((accel.id, speech.join(accel.installed_name())));
    }
    let detector = echo_lib::asr::catalog::entry(echo_lib::asr::catalog::ids::DETECTOR).unwrap();
    wanted.push((detector.id, speech.join(detector.installed_name())));

    for (id, path) in &wanted {
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
    if want_table {
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
    }

    let written = repo::get_segments(
        &db,
        &TranscriptQuery {
            meeting_id: created.id.clone(),
            limit: Some(50_000),
            ..Default::default()
        },
    )
    .await?;

    let (attempts, stretch_ms) = {
        let held = probe.attempts.lock().unwrap();
        (
            held.len(),
            covered_ms(
                held.iter()
                    .map(|a| (a.t_start_ms, a.t_start_ms + a.duration_ms))
                    .collect(),
            ),
        )
    };
    let ok = probe.ok.load(Ordering::SeqCst);
    let failed = probe.failed.load(Ordering::SeqCst);
    let meeting_ms = copied.len() as i64 * CHUNK_MS;
    let transcribed_ms = covered_ms(written.iter().map(|s| (s.t_start_ms, s.t_end_ms)).collect());
    let pct = |part: i64, whole: i64| {
        if whole > 0 {
            100.0 * part as f64 / whole as f64
        } else {
            0.0
        }
    };

    println!();
    println!("=== coverage ===");
    println!(
        "meeting length            {:>8.1} s",
        meeting_ms as f64 / 1_000.0
    );
    println!(
        "speech stretches found    {attempts:>8}   ({:.1} s, {:.1}% of the meeting)",
        stretch_ms as f64 / 1_000.0,
        pct(stretch_ms, meeting_ms)
    );
    println!(
        "decodes ok / failed       {ok:>8} / {failed}   ({:.1}% ok)",
        pct(i64::from(ok), i64::from(ok + failed))
    );
    println!(
        "speech transcribed        {:>8.1} s   ({:.1}% of the meeting, {:.1}% of the speech found)",
        transcribed_ms as f64 / 1_000.0,
        pct(transcribed_ms, meeting_ms),
        pct(transcribed_ms, stretch_ms)
    );
    println!(
        "segments written          {:>8}   (pass reported {}, windows read {})",
        written.len(),
        report.segments_written,
        report.windows_read
    );
    println!("wall clock                {:>8.1} s", elapsed.as_secs_f32());

    // Five lines spread across the whole meeting, so the evidence is that the
    // end was transcribed as well as the beginning.
    println!();
    println!("=== {SAMPLES} lines across the meeting (first {SAMPLE_WORDS} words) ===");
    for i in 0..SAMPLES.min(written.len()) {
        let last = written.len() - 1;
        let at = if SAMPLES > 1 {
            i * last / (SAMPLES - 1)
        } else {
            0
        };
        let s = &written[at];
        println!(
            "  [{:>3}:{:02}] {}",
            s.t_start_ms / 60_000,
            (s.t_start_ms % 60_000) / 1_000,
            first_words(&s.text, SAMPLE_WORDS)
        );
    }

    if let Some(path) = dump {
        std::fs::write(&path, serde_json::to_vec_pretty(&written)?)?;
        println!();
        println!("wrote {} segments to {}", written.len(), path.display());
    }

    engine.shutdown();
    Ok(())
}
