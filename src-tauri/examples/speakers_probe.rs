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
//!   --sweep         also print how many people every clustering *distance*
//!                   would find on this meeting. Kept because it is the picture
//!                   of the failure this probe exists to document: the band of
//!                   distances that gets this meeting right is 0.04 wide, and a
//!                   retranscription moved it under the number that shipped.
//!                   The automatic count no longer reads a distance at all.
//!   --from --to --step
//!                   narrow the sweep, to find an edge precisely.
//! ```
//!
//! ## What it prints, and why that is the point
//!
//! The automatic count comes out of the shape of the merge tree
//! (`diarize::cluster::CountChoice`), so this probe prints the tree: every merge
//! distance near the top, and one row per count the tree could have been cut at
//! with the relative gap, the silhouette and the cluster masses that decided
//! between them. A count nobody can inspect is a count nobody can argue with,
//! and the number this replaces was wrong in the field for exactly that reason.
//!
//! It also prints the lines that used to hold two voices and now hold one each
//! — the second half of the same failure, where fast dialogue came out as one
//! transcript row and attribution had to give the whole exchange to whoever held
//! more of it.
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
use echo_lib::diarize::{self, cluster, pipeline::DiarizeControl};
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

/// How many rungs of the merge ladder are worth reading. The decision is always
/// near the top; the hundreds of merges below it are one voice agreeing with
/// itself.
const LADDER_ROWS: usize = 14;
/// Mixed-voice lines to show cut in two.
const SPLIT_EXAMPLES: usize = 3;

fn verdict(refusal: Option<cluster::Refusal>) -> &'static str {
    match refusal {
        None => "allowed",
        Some(cluster::Refusal::OneVoice) => "one voice",
        Some(cluster::Refusal::Fused) => "two fused",
        Some(cluster::Refusal::TooMany) => "too many",
    }
}

fn masses(mass_ms: &[i64]) -> String {
    mass_ms
        .iter()
        .map(|ms| format!("{:.0}s", *ms as f64 / 1_000.0))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The whole automatic decision, printed so somebody can disagree with it.
fn print_choice(choice: &cluster::CountChoice) {
    println!();
    println!("=== the merge tree the count was read out of ===");
    println!(
        "{} merges, of which the top {} — a cluster under {:.1} s of speech is a \
         fragment in this meeting, not a person",
        choice.ladder.len(),
        LADDER_ROWS.min(choice.ladder.len()),
        choice.fragment_bar_ms as f64 / 1_000.0,
    );
    println!("{:>9}{:>10}{:>8}", "clusters", "distance", "gap");
    // The ladder is recorded in merge order, so the top of the tree is the tail.
    // `gap` is the merge this cut refuses over the last one it accepted — the
    // column a gap criterion would read the answer off.
    let from = choice.ladder.len().saturating_sub(LADDER_ROWS);
    for (i, rung) in choice.ladder.iter().enumerate().skip(from) {
        let accepted = rung.distance;
        let refused = choice.ladder.get(i + 1).map(|r| r.distance);
        println!(
            "{:>9}{:>10.4}{:>8}",
            rung.clusters,
            rung.distance,
            match refused {
                Some(r) if accepted > 1e-6 => format!("{:.3}", r / accepted),
                _ => "-".to_string(),
            }
        );
    }

    println!();
    println!("=== every count the tree could have been cut into ===");
    println!(
        "{:>7}{:>7}{:>7}{:>10}{:>10}{:>8}{:>12}   {:<10} speech per voice",
        "people", "cut", "folded", "accepted", "refused", "gap", "silhouette", "verdict",
    );
    for c in &choice.candidates {
        println!(
            "{:>7}{:>7}{:>7}{:>10.4}{:>10.4}{:>8}{:>12.3}   {:<10} {}{}",
            c.count,
            c.cut_clusters,
            c.folded,
            c.accepted,
            c.refused,
            if c.gap > 0.0 {
                format!("{:.3}", c.gap)
            } else {
                "-".to_string()
            },
            c.silhouette,
            verdict(c.refusal),
            masses(&c.mass_ms),
            if c.count == choice.count {
                "   <- chosen"
            } else {
                ""
            }
        );
    }

    println!();
    match choice.runner_up {
        Some((count, score)) => println!(
            "chose {} people at silhouette {:.3}; next best was {count} at {score:.3} \
             (margin {:.3})",
            choice.count,
            choice.silhouette,
            choice.silhouette - score
        ),
        None => println!(
            "chose {} people at silhouette {:.3}; there was nothing to compare it to",
            choice.count, choice.silhouette
        ),
    }
    if let Some(widest) = &choice.gap_answer {
        println!(
            "the largest relative gap over every cut height is {:.3}, at a cut into {} \
             clusters, which folds to {} — {}",
            widest.gap,
            widest.cut_clusters,
            if widest.count == 1 {
                "one voice".to_string()
            } else {
                format!("{} people", widest.count)
            },
            if widest.count == choice.count {
                "which agrees here, but see cluster::CountChoice for a tree where it does not"
            } else {
                "which is the wrong answer, and is why the criterion is not the gap"
            }
        );
    }
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
        "network:   {}",
        catalog::entry(catalog::ids::EMBEDDER)
            .map(|e| e.name)
            .unwrap_or("?")
    );
    println!(
        "count:     silhouette of the merge tree, floor {:.2} ceiling {:.2} \
         (old shipped cut {:.4}, not used)",
        cluster::SPLIT_FLOOR,
        cluster::FUSE_CEILING,
        diarize::DISTANCE_THRESHOLD,
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
    //
    // Kept as it went in, so the lines the pass cut in two can be found
    // afterwards by looking for the ones that are no longer one row.
    let mut seeded: Vec<(i64, i64, String)> = Vec::new();
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
        seeded = drafts
            .iter()
            .map(|d| (d.t_start_ms, d.t_end_ms, d.text.clone()))
            .collect();
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
                    "   <- what used to ship"
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
    println!("tree cut at               {:>8.4}", result.threshold);
    println!(
        "pass took                 {:>8.1} s   ({:.2}x real time)",
        elapsed.as_secs_f32(),
        elapsed.as_secs_f64() / (meeting_ms as f64 / 1_000.0)
    );

    if let Some(choice) = &result.choice {
        print_choice(choice);
    }

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

    // --- the lines that used to hold two voices ---------------------------
    //
    // A seeded line that is no longer one row was cut by the pass. Showing the
    // halves side by side with the speaker each landed on is the whole claim:
    // before this, one of the two people in every one of these exchanges lost
    // their words to the other.
    let name_of = |id: Option<&str>| -> String {
        id.and_then(|id| result.speakers.iter().find(|s| s.id == id))
            .map(|s| s.display_name.clone())
            .unwrap_or_else(|| "unattributed".to_string())
    };
    let mut shown = 0usize;
    let mut split_total = 0usize;
    let mut report: Vec<String> = Vec::new();
    for (from, to, text) in &seeded {
        let pieces: Vec<&Segment> = all
            .iter()
            .filter(|s| s.t_start_ms >= *from && s.t_end_ms <= *to)
            .collect();
        let voices: std::collections::BTreeSet<&str> = pieces
            .iter()
            .filter_map(|p| p.speaker_id.as_deref())
            .collect();
        if pieces.len() < 2 || voices.len() < 2 {
            continue;
        }
        split_total += 1;
        if shown >= SPLIT_EXAMPLES {
            continue;
        }
        shown += 1;
        report.push(format!(
            "[{}] was one line: {}",
            clock(*from),
            first_words(text, SAMPLE_WORDS)
        ));
        for p in &pieces {
            report.push(format!(
                "    [{}] {:<12} {}",
                clock(p.t_start_ms),
                name_of(p.speaker_id.as_deref()),
                first_words(&p.text, SAMPLE_WORDS)
            ));
        }
    }
    if split_total > 0 {
        println!();
        println!(
            "=== {split_total} line(s) held more than one voice and were cut; {shown} of them ==="
        );
        for line in report {
            println!("{line}");
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
