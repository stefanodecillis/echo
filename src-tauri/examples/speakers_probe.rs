//! Speaker probe: run the **real** offline speaker pass over a real meeting's
//! microphone chunks, against the **real** installed ONNX weights, and print how
//! many people it found and who said what.
//!
//! ```text
//! cargo run --release --example speakers_probe -- <meeting-id> [flags]
//!
//!   --segments P    transcript to seed the throwaway database with, the JSON
//!                   `catchup_probe --dump` writes. Without it the pass still
//!                   runs, but there are no words to attribute.
//!   --people N      exercise the override path: ask for exactly N voices
//!                   instead of letting the threshold decide.
//!   --minutes M     only the first M minutes, for a quick loop.
//!   --segmenter P   use this ONNX segmenter instead of the installed one.
//!   --embedder P    use this ONNX fingerprint network instead of the installed
//!                   one. How a candidate asset is tried on real audio before it
//!                   goes in the catalog.
//!   --sweep         also print how many people every clustering threshold would
//!                   find on this meeting. The check against having tuned the
//!                   threshold to synthetic voices: the plateau here and the
//!                   plateau in `examples/voices_fixture.rs` have to overlap.
//!   --from --to --step
//!                   narrow the sweep, to find an edge precisely.
//! ```
//!
//! This is the microphone-only half of the speaker pass, on the audio it was
//! written for: a meeting where the person was on speakers, so the other voice
//! arrived through their microphone and there is no system channel at all. See
//! `diarize::pipeline`'s module docs for why nothing here is called "You".
//!
//! Nothing in the app's own storage is written: the chunks are copied to a
//! temporary directory, the database is a fresh temporary file, and the model
//! rows are pointed at the installed files read-only.

use std::path::{Path, PathBuf};

use echo_lib::asr::{catalog, models};
use echo_lib::db::{self, repo};
use echo_lib::diarize::{self, pipeline::DiarizeControl};
use echo_lib::types::{AssetKind, Channel, Segment, SegmentDraft, TranscriptQuery};

const CHUNK_MS: i64 = 30_000;
const DEFAULT_MEETING: &str = "13731077-453a-4c75-8e6e-5586fe2ed019";
/// Lines per speaker. Private audio: two is enough to tell two people apart.
const SAMPLES_PER_SPEAKER: usize = 2;
const SAMPLE_WORDS: usize = 10;

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

fn clock(ms: i64) -> String {
    format!("{:>3}:{:02}", ms / 60_000, (ms % 60_000) / 1_000)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut positional: Vec<String> = Vec::new();
    let mut segments_file: Option<PathBuf> = None;
    let mut people: Option<u32> = None;
    let mut minutes: Option<i64> = None;
    let mut segmenter_arg: Option<PathBuf> = None;
    let mut embedder_arg: Option<PathBuf> = None;
    let mut do_sweep = false;
    let mut sweep_from = 0.15f32;
    let mut sweep_to = 1.10f32;
    let mut sweep_step = 0.0125f32;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--segments" => segments_file = args.next().map(PathBuf::from),
            "--people" => people = args.next().and_then(|n| n.parse().ok()),
            "--minutes" => minutes = args.next().and_then(|n| n.parse().ok()),
            "--segmenter" => segmenter_arg = args.next().map(PathBuf::from),
            "--embedder" => embedder_arg = args.next().map(PathBuf::from),
            "--sweep" => do_sweep = true,
            "--from" => {
                sweep_from = args
                    .next()
                    .and_then(|n| n.parse().ok())
                    .unwrap_or(sweep_from)
            }
            "--to" => sweep_to = args.next().and_then(|n| n.parse().ok()).unwrap_or(sweep_to),
            "--step" => {
                sweep_step = args
                    .next()
                    .and_then(|n| n.parse().ok())
                    .unwrap_or(sweep_step)
            }
            other => positional.push(other.to_string()),
        }
    }
    let meeting = positional
        .first()
        .cloned()
        .unwrap_or_else(|| DEFAULT_MEETING.to_string());
    let chunks_wanted = minutes.map_or(i64::MAX, |m| (m * 60_000 / CHUNK_MS).max(1));

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
    let meeting_ms = copied.len() as i64 * CHUNK_MS;
    println!(
        "copied {} mic chunk(s) ({:.1} s) from {}",
        copied.len(),
        meeting_ms as f64 / 1_000.0,
        source.display()
    );

    // --- a throwaway database pointed at the weights we want to try -------
    //
    // Asset ids and file names come from the catalog rather than being spelled
    // out here, so swapping a model is one edit in one place and this probe
    // cannot drift out of step with the app.
    let db = db::connect(&tmp.path().join("echo.db")).await?;
    models::sync_catalog(&db).await?;
    for (id, over) in [
        (catalog::ids::SEGMENTER, segmenter_arg.as_ref()),
        (catalog::ids::EMBEDDER, embedder_arg.as_ref()),
    ] {
        let entry = catalog::entry(id).expect("catalogued speaker asset");
        let path = over
            .cloned()
            .unwrap_or_else(|| speech.join(entry.file_name));
        if !path.exists() {
            return Err(format!(
                "{id} is not at {} — pass --segmenter/--embedder to point at a candidate",
                path.display()
            )
            .into());
        }
        repo::set_model_installed(&db, id, true, Some(&path.to_string_lossy())).await?;
    }
    let (segmenter, embedder) = diarize::job::model_paths(&db).await?;
    println!("segmenter: {}", segmenter.display());
    println!("embedder:  {}", embedder.display());
    println!(
        "threshold: {:.4}  ({})",
        diarize::DISTANCE_THRESHOLD,
        catalog::entry(catalog::ids::EMBEDDER)
            .map(|e| e.name)
            .unwrap_or("?")
    );
    // The rows must name the assets that are actually loaded, or the pass would
    // silently fall back to something else.
    for kind in [AssetKind::SpeakerSegmenter, AssetKind::SpeakerEmbedder] {
        assert!(
            models::installed_path(&db, kind).await?.is_some(),
            "{kind:?} did not resolve"
        );
    }

    // --- the meeting and its committed mic chunks -------------------------
    let created = repo::create_meeting(
        &db,
        "speaker probe",
        &audio_dir.to_string_lossy(),
        Some("probe"),
    )
    .await?;
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
    // No system chunks at all — which is the whole point: the person was on
    // speakers, so both voices are in the microphone recording.

    // --- the transcript the catch-up probe produced ------------------------
    if let Some(path) = &segments_file {
        let raw = std::fs::read(path)?;
        let loaded: Vec<Segment> = serde_json::from_slice(&raw)?;
        let drafts: Vec<SegmentDraft> = loaded
            .iter()
            .filter(|s| s.t_end_ms <= meeting_ms)
            .map(|s| SegmentDraft {
                meeting_id: created.id.clone(),
                t_start_ms: s.t_start_ms,
                t_end_ms: s.t_end_ms,
                channel: s.channel,
                speaker_id: None,
                text: s.text.clone(),
                language: s.language.clone(),
                avg_confidence: s.avg_confidence,
                revision: 1,
                is_final: true,
                model_name: s.model_name.clone(),
                model_revision: s.model_revision.clone(),
            })
            .collect();
        repo::insert_segments(&db, &drafts).await?;
        println!(
            "seeded {} transcript line(s) from {}",
            drafts.len(),
            path.display()
        );
    }

    // The live channel pass runs first in the real app, and on a meeting like
    // this one it says "You" and nothing else. The offline pass has to correct
    // that, so reproduce it rather than starting from a clean slate.
    let live = diarize::ensure_channel_speakers(&db, &created.id).await?;
    println!(
        "live channel pass: {}",
        live.iter()
            .map(|s| s.display_name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );

    if let Some(n) = people {
        repo::set_speaker_count_override(&db, &created.id, Some(n)).await?;
        println!("people count override: {n}");
    } else {
        println!("people count: automatic");
    }

    // --- the threshold, checked against real audio ------------------------
    //
    // The whole point of running this: the fixture sweep in
    // `examples/voices_fixture.rs` measures synthetic voices, which are cleaner
    // and more self-consistent than people. If the range of thresholds that gets
    // *this* meeting right and the range that gets the fixtures right do not
    // overlap, the fixtures were the wrong evidence and the number has to come
    // from somewhere else.
    if do_sweep {
        let scanned = diarize::pipeline::scan(
            &db,
            &created.id,
            &segmenter,
            &embedder,
            &DiarizeControl::new(),
        )
        .await?
        .expect("the meeting has audio");
        println!();
        println!("=== what every threshold would find on this real meeting ===");
        println!(
            "{} fingerprints of {} dimensions",
            scanned.fingerprints(),
            scanned.fingerprint_dim().unwrap_or(0)
        );
        println!("{:>10}{:>8}   split", "threshold", "voices");
        let mut index = 0usize;
        loop {
            let t = sweep_from + index as f32 * sweep_step.max(1e-4);
            if t > sweep_to + 1e-6 {
                break;
            }
            index += 1;
            let cut = scanned.cut(t, None);
            let per: Vec<String> = cut
                .tracks
                .iter()
                .map(|track| {
                    format!(
                        "{:.0}s",
                        echo_lib::diarize::timeline::total_ms(track) as f64 / 1_000.0
                    )
                })
                .collect();
            println!(
                "{t:>10.4}{:>8}   {}{}",
                cut.tracks.len(),
                per.join(" "),
                if (t - diarize::DISTANCE_THRESHOLD).abs() < 1e-4 {
                    "   <- shipped"
                } else {
                    ""
                }
            );
        }
    }

    // --- the real pass ----------------------------------------------------
    let started = std::time::Instant::now();
    let result = diarize::refine_speakers(&db, &created.id, &segmenter, &embedder).await?;
    let elapsed = started.elapsed();

    println!();
    println!("=== who was in the meeting ===");
    println!(
        "people detected           {:>8}{}",
        result.people_count,
        if result.people_count_is_override {
            "   (the person's own answer)"
        } else {
            "   (Echo's own count)"
        }
    );
    println!("voices separated          {:>8}", result.speaker_count);
    println!("turns                     {:>8}", result.turns.len());
    println!("clustering cut at         {:>8.4}", result.threshold);
    println!(
        "pass took                 {:>8.1} s   ({:.2}x real time)",
        elapsed.as_secs_f32(),
        elapsed.as_secs_f64() / (meeting_ms as f64 / 1_000.0)
    );

    let all = repo::get_segments(
        &db,
        &TranscriptQuery {
            meeting_id: created.id.clone(),
            limit: Some(50_000),
            ..Default::default()
        },
    )
    .await?;

    println!();
    println!("=== per speaker ===");
    for speaker in &result.speakers {
        let mine: Vec<&Segment> = all
            .iter()
            .filter(|s| s.speaker_id.as_deref() == Some(speaker.id.as_str()))
            .collect();
        let turn_ms: i64 = result
            .turns
            .iter()
            .filter(|t| diarize::cluster_key(t.cluster as usize) == speaker.cluster_key)
            .map(|t| t.duration_ms())
            .sum();
        println!(
            "{} [{}]  transcript {:.1} s over {} line(s), turns {:.1} s",
            speaker.display_name,
            speaker.cluster_key,
            speaker.speaking_ms as f64 / 1_000.0,
            mine.len(),
            turn_ms as f64 / 1_000.0
        );
        for i in 0..SAMPLES_PER_SPEAKER.min(mine.len()) {
            let at = if SAMPLES_PER_SPEAKER > 1 {
                i * (mine.len() - 1) / (SAMPLES_PER_SPEAKER - 1)
            } else {
                0
            };
            let s = mine[at];
            println!(
                "    [{}] {}",
                clock(s.t_start_ms),
                first_words(&s.text, SAMPLE_WORDS)
            );
        }
    }

    let unattributed = all.iter().filter(|s| s.speaker_id.is_none()).count();
    let unattributed_ms: i64 = all
        .iter()
        .filter(|s| s.speaker_id.is_none())
        .map(|s| s.t_end_ms - s.t_start_ms)
        .sum();
    println!();
    println!(
        "unattributed              {unattributed:>8} line(s), {:.1} s",
        unattributed_ms as f64 / 1_000.0
    );

    Ok(())
}
