//! Speculative-cadence probe: how long does the **real** engine take to decode
//! one caption window of real meeting audio?
//!
//! ```text
//! cargo run --release --example speculative_probe -- [meeting-id] [flags]
//!
//!   --windows N   how many windows to decode at each length (default 6)
//!   --lengths L   comma-separated window lengths in seconds (default 6,8,10)
//! ```
//!
//! `ECHO_PROBE_SPEECH_DIR` and `ECHO_PROBE_SPEECH_ID` point it at weights, the
//! same way `catchup_probe` does.
//!
//! # Why this has to be measured rather than reasoned about
//!
//! The live caption task decodes the tail of an open utterance every
//! [`session::pipeline::CAPTION_STEP_MS`] and replaces the line wholesale. That
//! only *feels* live if a decode finishes in roughly a step. Snapshots supersede
//! each other, so a slow decode never queues up — but it does mean the caption
//! on screen is always one decode behind the speech, and past a few seconds of
//! that a person stops believing the captions are live at all.
//!
//! The window length is the only dial that matters here, and it trades directly
//! against quality: less left context is a worse guess. So the number is not a
//! judgement call, it is a measurement, and this is the thing that measures it.
//! Print it against the model that is actually shipping and set
//! `CAPTION_WINDOW_MS` from what comes out.
//!
//! Nothing in the app's storage is touched: the audio is copied to a temporary
//! directory and the database is a throwaway file.

use std::path::{Path, PathBuf};
use std::time::Instant;

use echo_lib::asr::engine::{DecodePlan, EngineWorker};
use echo_lib::asr::{models, TranscribeJob};
use echo_lib::audio::{ChunkRef, TARGET_SAMPLE_RATE};
use echo_lib::db::{self, repo};
use echo_lib::types::{AssetKind, Channel};

const CHUNK_MS: i64 = 30_000;
const DEFAULT_MEETING: &str = "13731077-453a-4c75-8e6e-5586fe2ed019";
/// Enough chunks to find speech in, without decoding the whole meeting.
const CHUNKS: i64 = 16;

fn app_support() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME");
    Path::new(&home).join("Library/Application Support/Echo")
}

/// Percentile of a sorted slice, nearest-rank.
fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let i = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[i.min(sorted.len() - 1)]
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut positional: Vec<String> = Vec::new();
    let mut windows = 6usize;
    let mut lengths: Vec<i64> = vec![6, 8, 10];
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--windows" => {
                windows = args.next().and_then(|v| v.parse().ok()).unwrap_or(windows);
            }
            "--lengths" => {
                if let Some(list) = args.next() {
                    lengths = list.split(',').filter_map(|v| v.trim().parse().ok()).collect();
                }
            }
            other => positional.push(other.to_string()),
        }
    }
    let meeting = positional
        .first()
        .cloned()
        .unwrap_or_else(|| DEFAULT_MEETING.to_string());

    let support = app_support();
    let source = support.join("recordings").join(&meeting);
    let speech = std::env::var_os("ECHO_PROBE_SPEECH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| support.join("speech"));

    let tmp = tempfile::tempdir()?;
    let audio_dir = tmp.path().join("audio");
    std::fs::create_dir_all(&audio_dir)?;

    let mut copied: Vec<(i64, PathBuf)> = Vec::new();
    for seq in 0..CHUNKS {
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

    // --- a throwaway database pointed at the weights ----------------------
    let db = db::connect(&tmp.path().join("echo.db")).await?;
    models::sync_catalog(&db).await?;
    repo::set_setting(
        &db,
        echo_lib::settings::keys::ACCURACY_LEVEL_ID,
        echo_lib::asr::catalog::DEFAULT_PRESET_ID,
    )
    .await?;
    let speech_id = std::env::var("ECHO_PROBE_SPEECH_ID")
        .unwrap_or_else(|_| echo_lib::asr::catalog::ids::SPEECH.to_string());
    let entry = echo_lib::asr::catalog::entry(&speech_id)
        .ok_or_else(|| format!("{speech_id} is not in the catalog"))?;
    let weights = speech.join(entry.installed_name());
    if !weights.exists() {
        return Err(format!("{} is not at {}", entry.id, weights.display()).into());
    }
    repo::set_model_installed(&db, entry.id, true, Some(&weights.to_string_lossy())).await?;

    // The companion is optional on purpose. Leaving it out is how this probe
    // answers "what does the Apple encoder actually buy?", and in the app a
    // missing one costs speed and never correctness.
    if let Some(accel) = echo_lib::asr::catalog::accelerator_for(entry.id) {
        let path = speech.join(accel.installed_name());
        if path.exists() {
            repo::set_model_installed(&db, accel.id, true, Some(&path.to_string_lossy())).await?;
        } else {
            println!("note:     no encoder companion in {} — measuring without it", speech.display());
        }
    }

    let created = repo::create_meeting(
        &db,
        "caption cadence probe",
        &audio_dir.to_string_lossy(),
        Some("probe"),
    )
    .await?;
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
    let chunks: Vec<ChunkRef> = repo::list_chunks(&db, &created.id, Some(Channel::Mic))
        .await?
        .iter()
        .filter(|c| c.committed)
        .map(ChunkRef::from_journal)
        .collect();

    // --- the real engine, loaded the way a recording loads it -------------
    let engine = EngineWorker::new();
    engine.configure_from_settings(&db).await?;
    let load_started = Instant::now();
    let backend = engine.load_now().await?;
    let load_secs = load_started.elapsed().as_secs_f64();
    println!(
        "model:    {}",
        models::installed_path(&db, AssetKind::Speech)
            .await?
            .unwrap()
            .display()
    );
    println!(
        "engine:   backend={} threads={} accelerator={}",
        backend.active, backend.threads, backend.accelerator_present
    );
    println!("load:     {load_secs:.1} s");
    println!(
        "audio:    {} chunk(s), {} s of the meeting's microphone",
        copied.len(),
        copied.len() as i64 * CHUNK_MS / 1_000
    );
    println!();

    // The caption task decodes the *tail* of what is open, so a window is the
    // last N seconds of speech. Take them from evenly spaced points so this is
    // not one unusual passage measured six times.
    let total_ms = copied.len() as i64 * CHUNK_MS;

    println!(
        "{:>7}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "window", "decodes", "median", "p90", "max", "vs step"
    );
    let mut verdicts = Vec::new();
    for secs in &lengths {
        let window_ms = secs * 1_000;
        let mut times: Vec<f64> = Vec::new();
        for n in 0..windows {
            // Spread the samples across the copied audio, leaving room for a
            // whole window before each end point.
            let end_ms =
                window_ms + ((total_ms - window_ms) * n as i64) / windows.max(1) as i64;
            let samples = echo_lib::audio::read_window(&chunks, end_ms - window_ms, end_ms).await?;
            if samples.len() < (window_ms * i64::from(TARGET_SAMPLE_RATE) / 1_000) as usize / 2 {
                continue;
            }
            let job = TranscribeJob {
                meeting_id: created.id.clone(),
                utterance_id: format!("cap-{secs}-{n}"),
                channel: Channel::Mic,
                t_start_ms: end_ms - window_ms,
                samples,
                language_hint: Some("it".into()),
                want_partials: false,
                droppable: true,
            };
            let started = Instant::now();
            let answer = engine
                .submit_with(job, DecodePlan::speculative(), None)
                .await;
            let elapsed = started.elapsed().as_secs_f64();
            if answer.is_ok() {
                times.push(elapsed);
            }
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = pct(&times, 50.0);
        let step = echo_lib::session::pipeline::CAPTION_STEP_MS as f64 / 1_000.0;
        println!(
            "{:>6}s  {:>8}  {:>7.2}s  {:>7.2}s  {:>7.2}s  {:>7.2}x",
            secs,
            times.len(),
            median,
            pct(&times, 90.0),
            times.last().copied().unwrap_or(0.0),
            if step > 0.0 { median / step } else { 0.0 }
        );
        verdicts.push((*secs, median, pct(&times, 90.0)));
    }

    // --- the decision this probe exists to make ---------------------------
    println!();
    println!("=== is the caption still live? ===");
    println!(
        "caption step is {} ms: a decode has to finish in about that long, or the",
        echo_lib::session::pipeline::CAPTION_STEP_MS
    );
    println!("line on screen is always more than one step behind the talking.");
    for (secs, median, p90) in &verdicts {
        let step = echo_lib::session::pipeline::CAPTION_STEP_MS as f64 / 1_000.0;
        let verdict = if *median <= step {
            "keeps up"
        } else if *median <= step * 1.5 {
            "one step behind — acceptable"
        } else {
            "too slow to read as live"
        };
        println!("  {secs:>2}s window  median {median:.2}s  p90 {p90:.2}s  → {verdict}");
    }
    println!();
    println!(
        "currently configured: CAPTION_WINDOW_MS = {} ms",
        echo_lib::session::pipeline::CAPTION_WINDOW_MS
    );

    engine.shutdown();
    Ok(())
}
