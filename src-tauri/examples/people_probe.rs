//! Known-people probe: enroll a voice out of a **real** meeting, then run the
//! **real** offline pass over that same meeting again and watch the voice come
//! back with a name on it — while the other voice in the room, and a voice
//! nobody ever enrolled, come back with nothing.
//!
//! ```text
//! cargo run --release --example people_probe -- <meeting-id> [flags]
//!
//!   --segments P    the meeting's transcript, the JSON `catchup_probe --dump`
//!                   writes. **Required for the enrollment step**: a sample is
//!                   taken from a stretch where that speaker is talking on their
//!                   own, and the transcript is what says where those are
//!                   (`diarize::sample::pick_windows`). Without it the pass
//!                   still runs and still stores voice prints, but there is
//!                   nothing to enroll from and the probe says so.
//!   --minutes M     only the first M minutes of the real meeting.
//!   --stranger V    the `say` voice to build the never-enrolled fixture from.
//!                   Default: Daniel, from `people_fixture`'s measured pool.
//!   --segmenter P   ONNX segmenter. Default: the installed one.
//!   --embedder P    ONNX fingerprint network. Default: the installed one.
//! ```
//!
//! ## Why this exists next to `people_fixture`
//!
//! `examples/people_fixture.rs` measures the *bars* — `TAU_LINK`, `TAU_SUGGEST`,
//! `MARGIN` — on synthetic voices, where identity is known because the fixture
//! wrote it. What it cannot do is prove that the app is wired up: that a clip
//! sampled off a real speaker row becomes a profile, that the profile is loaded
//! before the count question, that the pre-assignment and the cluster-level
//! match agree, and that the answer lands on the speaker row the UI reads. Every
//! one of those is a join between two modules, and a bar measured perfectly on
//! either side of a broken join is worth nothing.
//!
//! So this probe runs the whole seam, on real audio, and prints the numbers the
//! decision was made from:
//!
//! 1. **Pass one, nobody enrolled.** The pass must fingerprint the meeting and
//!    store a voice print per speaker — without that, nothing later is possible.
//! 2. **Enroll.** The longest-speaking voice becomes a person, through the same
//!    `people::enroll` the "Remember this voice" checkbox calls.
//! 3. **Pass two, one person enrolled.** That voice must come back **linked**,
//!    with the person's name copied onto the row; every other voice in the same
//!    meeting must come back with no link *and no suggestion*. This is the
//!    open-set half, and it is the half that matters: the run is only honest
//!    because it can fail here.
//! 4. **A voice nobody enrolled.** A `say` voice, in its own meeting, in the same
//!    database as the profile from step 2. Nobody may be linked, nobody
//!    suggested — and the score it actually reached is printed, so "no" is a
//!    number rather than an assertion.
//!
//! Step 3 re-reads the audio the profile was built from, so it is a plumbing
//! proof, not a generalisation result — recognising a voice across *different*
//! meetings is what `people_fixture`'s mixed fixture speaks to. Step 4 is the
//! one that can only come out right by the geometry being right.
//!
//! Nothing in the app's own storage is written: the chunks are copied to a
//! temporary directory, the database is a fresh temporary file, and the model
//! rows are pointed at the installed files read-only.

use std::path::{Path, PathBuf};
use std::process::Command;

use echo_lib::asr::{catalog, models};
use echo_lib::db::{self, repo, Db};
use echo_lib::diarize::{self, people};
use echo_lib::types::{AssetKind, Channel, Id, Segment, SegmentDraft};

const CHUNK_MS: i64 = 30_000;
const DEFAULT_MEETING: &str = "13731077-453a-4c75-8e6e-5586fe2ed019";
/// The name the enrolled voice is remembered under. Not a real name on purpose:
/// this is somebody's actual meeting.
const PROBE_NAME: &str = "Probe Person";
/// A voice from `people_fixture`'s measured pool, used here as somebody Echo has
/// never been told to remember.
const DEFAULT_STRANGER: &str = "Daniel";

fn app_support() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME");
    Path::new(&home).join("Library/Application Support/Echo")
}

/// A speaker row as the UI reads it, plus the voice print behind it.
///
/// Read straight out of `speakers` rather than through `list_speakers`, because
/// the centroid is deliberately not on the type that crosses to the webview and
/// this probe's whole job is to check the numbers under it.
#[derive(Debug, Clone)]
struct Row {
    id: Id,
    cluster_key: String,
    display_name: String,
    speaking_ms: i64,
    person_id: Option<String>,
    suggested_person_id: Option<String>,
    suggestion_score: Option<f32>,
    centroid: Vec<f32>,
}

fn decode_centroid(blob: Option<Vec<u8>>) -> Vec<f32> {
    blob.map(|bytes| {
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    })
    .unwrap_or_default()
}

async fn rows_of(db: &Db, meeting_id: &str) -> Result<Vec<Row>, Box<dyn std::error::Error>> {
    let found = sqlx::query_as::<
        _,
        (
            String,
            String,
            String,
            i64,
            Option<String>,
            Option<String>,
            Option<f32>,
            Option<Vec<u8>>,
        ),
    >(
        "SELECT id, cluster_key, display_name, speaking_ms, person_id, suggested_person_id,
                suggestion_score, centroid
         FROM speakers
         WHERE meeting_id = ?1 AND alias_of IS NULL AND is_self = 0
         ORDER BY cluster_key",
    )
    .bind(meeting_id)
    .fetch_all(db)
    .await?;
    Ok(found
        .into_iter()
        .map(
            |(id, cluster_key, display_name, speaking_ms, person_id, suggested_person_id, score, centroid)| Row {
                id,
                cluster_key,
                display_name,
                speaking_ms,
                person_id,
                suggested_person_id,
                suggestion_score: score,
                centroid: decode_centroid(centroid),
            },
        )
        .collect())
}

/// Enough of an id to tell two of them apart, in a column that has to fit.
fn short(id: &str) -> String {
    id.chars().take(8).collect()
}

/// The header every table below shares.
fn print_rows(label: &str, rows: &[Row], profiles: &[repo::PersonProfileRow]) {
    println!();
    println!("=== {label} ===");
    println!(
        "{:<12}{:<16}{:>9}{:>7}{:>10}{:>11}{:>9}",
        "key", "row", "speech", "print", "link", "suggested", "score"
    );
    for row in rows {
        // What the open-set rule was actually looking at: this voice against
        // every profile, best first, and what the best had to beat.
        let scores: Vec<f32> = profiles
            .iter()
            .map(|p| people::similarity(&row.centroid, &p.centroid))
            .collect();
        let verdict = people::decide(&scores);
        let best = scores
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1));
        println!(
            "{:<12}{:<16}{:>8.1}s{:>7}{:>10}{:>11}{:>9}",
            row.cluster_key,
            row.display_name,
            row.speaking_ms as f64 / 1_000.0,
            if row.centroid.is_empty() {
                "-".to_string()
            } else {
                format!("{}d", row.centroid.len())
            },
            row.person_id.as_deref().map(short).unwrap_or("—".into()),
            row.suggested_person_id.as_deref().map(short).unwrap_or("—".into()),
            row.suggestion_score
                .map(|s| format!("{s:.3}"))
                .unwrap_or_else(|| "—".to_string()),
        );
        if let Some((who, score)) = best {
            // The margin is measured against the next-best profile, floored at
            // 0.0 — "no resemblance at all" — so one enrolled person is held to
            // the same standard as five (see `people::top_two`).
            let runner_up = scores
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != who)
                .map(|(_, s)| *s)
                .fold(0.0f32, f32::max);
            println!(
                "    ↳ nearest {:?} sim {:.3}  margin {:+.3}  →  {}",
                profiles[who].name,
                score,
                score - runner_up,
                match verdict {
                    people::Verdict::Linked { .. } => "link",
                    people::Verdict::Suggested { .. } => "ask",
                    people::Verdict::Nobody => "nobody",
                }
            );
        }
    }
}

/// Register a directory of `mic-NNNNNN.wav` chunks as a meeting, committed, with
/// the live channel pass already run — the state the offline pass really starts
/// from.
async fn add_meeting(
    db: &Db,
    title: &str,
    audio_dir: &Path,
    chunks: &[(i64, PathBuf)],
) -> Result<Id, Box<dyn std::error::Error>> {
    let created =
        repo::create_meeting(db, title, &audio_dir.to_string_lossy(), Some("probe")).await?;
    for (seq, path) in chunks {
        let id = repo::insert_chunk(
            db,
            &created.id,
            Channel::Mic,
            *seq,
            &path.to_string_lossy(),
            seq * CHUNK_MS,
            (seq + 1) * CHUNK_MS,
        )
        .await?;
        repo::commit_chunk(db, &id, (seq + 1) * CHUNK_MS).await?;
    }
    diarize::ensure_channel_speakers(db, &created.id).await?;
    Ok(created.id)
}

// ---------------------------------------------------------------------------
// the never-enrolled voice
// ---------------------------------------------------------------------------

const STRANGER_LINES: &[&str] = &[
    "Thanks for making the time today, I know the week has been busy for everyone.",
    "Before we go further, can we agree on what actually ships at the end of the month?",
    "My worry is that we are solving the second problem before we understand the first one.",
    "I looked at the numbers again last night and they are better than we thought.",
    "Let me say that back to you, because I want to be sure I have understood it.",
    "If we push the date, the only thing that changes is who is disappointed later.",
];

/// One `say` voice reading enough lines to be fingerprintable, as 16 kHz mono
/// chunks — the same shape a real recording arrives in.
fn render_stranger(voice: &str, work: &Path, dir: &Path) -> Vec<(i64, PathBuf)> {
    let aiff = work.join("stranger.aiff");
    let wav = work.join("stranger.wav");
    let text = STRANGER_LINES.join(" ");
    let said = Command::new("say")
        .args(["-v", voice, "-o"])
        .arg(&aiff)
        .arg(&text)
        .status()
        .expect("`say` — this probe's stranger fixture is macOS only");
    assert!(said.success(), "say -v {voice:?} failed");
    let converted = Command::new("afconvert")
        .args(["-f", "WAVE", "-d", "LEI16@16000", "-c", "1"])
        .arg(&aiff)
        .arg(&wav)
        .status()
        .expect("afconvert");
    assert!(converted.success(), "afconvert failed");

    let mut reader = hound::WavReader::open(&wav).expect("read the rendered voice");
    let speech: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.expect("sample") as f32 / 32_768.0)
        .collect();

    // Half a second of room before it starts, and padded out to whole chunks,
    // exactly like `people_fixture` builds its meetings.
    let per_ms = 16usize;
    let mut audio = vec![0.0f32; 500 * per_ms];
    audio.extend_from_slice(&speech);
    let chunk_samples = CHUNK_MS as usize * per_ms;
    let remainder = audio.len() % chunk_samples;
    if remainder != 0 {
        audio.extend(std::iter::repeat_n(0.0f32, chunk_samples - remainder));
    }

    std::fs::create_dir_all(dir).expect("fixture directory");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut out = Vec::new();
    for (seq, block) in audio.chunks(chunk_samples).enumerate() {
        let path = dir.join(format!("mic-{seq:06}.wav"));
        let mut writer = hound::WavWriter::create(&path, spec).expect("create chunk");
        for x in block {
            writer
                .write_sample((x * 32_767.0).clamp(-32_768.0, 32_767.0) as i16)
                .expect("write sample");
        }
        writer.finalize().expect("finalize chunk");
        out.push((seq as i64, path));
    }
    out
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut positional: Vec<String> = Vec::new();
    let mut minutes: Option<i64> = None;
    let mut segments_file: Option<PathBuf> = None;
    let mut stranger = DEFAULT_STRANGER.to_string();
    let mut segmenter_arg: Option<PathBuf> = None;
    let mut embedder_arg: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--minutes" => minutes = args.next().and_then(|n| n.parse().ok()),
            "--segments" => segments_file = args.next().map(PathBuf::from),
            "--stranger" => stranger = args.next().unwrap_or(stranger),
            "--segmenter" => segmenter_arg = args.next().map(PathBuf::from),
            "--embedder" => embedder_arg = args.next().map(PathBuf::from),
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
    println!(
        "copied {} mic chunk(s) ({:.1} s) from {}",
        copied.len(),
        (copied.len() as i64 * CHUNK_MS) as f64 / 1_000.0,
        source.display()
    );

    // --- a throwaway database pointed at the installed weights ------------
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
            return Err(format!("{id} is not at {}", path.display()).into());
        }
        repo::set_model_installed(&db, id, true, Some(&path.to_string_lossy())).await?;
    }
    let (segmenter, embedder) = diarize::job::model_paths(&db).await?;
    for kind in [AssetKind::SpeakerSegmenter, AssetKind::SpeakerEmbedder] {
        assert!(
            models::installed_path(&db, kind).await?.is_some(),
            "{kind:?} did not resolve"
        );
    }
    println!("embedder:  {}", embedder.display());
    println!(
        "network:   {}  (profiles are tagged {:?})",
        catalog::entry(catalog::ids::EMBEDDER)
            .map(|e| e.name)
            .unwrap_or("?"),
        people::embedder_tag(&db, &embedder).await,
    );
    println!(
        "bars:      link {:.2}  ask {:.2}  margin {:.2}  pre-assign {:.2}",
        people::TAU_LINK,
        people::TAU_SUGGEST,
        people::MARGIN,
        people::TAU_STRONG,
    );

    let real = add_meeting(&db, "people probe", &audio_dir, &copied).await?;

    // The meeting's own words, so a sample can be taken from a stretch where one
    // voice is talking on its own — which is how the app does it, and the only
    // reason a clip is worth keeping.
    let meeting_ms = copied.len() as i64 * CHUNK_MS;
    if let Some(path) = &segments_file {
        let loaded: Vec<Segment> = serde_json::from_slice(&std::fs::read(path)?)?;
        let drafts: Vec<SegmentDraft> = loaded
            .iter()
            .filter(|s| s.t_end_ms <= meeting_ms)
            .map(|s| SegmentDraft {
                meeting_id: real.clone(),
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
                corrections: s.corrections.clone(),
            })
            .collect();
        repo::insert_segments(&db, &drafts).await?;
        println!(
            "seeded {} transcript line(s) from {}",
            drafts.len(),
            path.display()
        );
    } else {
        println!(
            "no --segments: the pass will run and store voice prints, but there \
             is nothing to enroll from"
        );
    }

    // === 1. the pass with nobody enrolled =================================
    let first = diarize::refine_speakers(&db, &real, &segmenter, &embedder).await?;
    let before = rows_of(&db, &real).await?;
    print_rows(
        &format!(
            "pass one — nobody enrolled ({} voice(s) detected)",
            first.people_count
        ),
        &before,
        &[],
    );
    assert!(
        before.len() >= 2,
        "this proof needs a meeting with at least two voices; got {}",
        before.len()
    );
    for row in &before {
        assert!(
            !row.centroid.is_empty(),
            "{}: the pass stored no voice print, so nothing downstream can work",
            row.display_name
        );
        assert!(
            row.person_id.is_none() && row.suggested_person_id.is_none(),
            "{}: linked to somebody before anybody was enrolled",
            row.display_name
        );
    }
    println!(
        "every voice print stored ({} rows, {} dimensions) — DESIGN §1's \
         \"per-meeting speaker centroids are stored\"",
        before.len(),
        before[0].centroid.len()
    );

    // Nothing recurring yet: one appearance is not a pattern.
    let suggestions = people::suggested_people(&db).await?;
    println!(
        "recurring unnamed voices: {} (needs {} appearances)",
        suggestions.len(),
        people::SUGGEST_MIN_APPEARANCES
    );

    // === 2. enroll the longest-speaking voice =============================
    let chosen = before
        .iter()
        .max_by_key(|r| r.speaking_ms)
        .expect("a speaker")
        .clone();
    assert!(
        chosen.speaking_ms > 0,
        "no speech is attributed to any row — pass --segments so there is a \
         stretch of this voice talking alone to sample from"
    );
    println!();
    println!(
        "=== remembering {:?} ({:.1} s of speech) as {PROBE_NAME:?} ===",
        chosen.display_name,
        chosen.speaking_ms as f64 / 1_000.0
    );
    let person = people::enroll(&db, &real, &chosen.id, PROBE_NAME).await?;
    println!(
        "person {:?}: {} sample(s), needs refresh {}",
        person.name, person.sample_count, person.needs_refresh
    );
    assert!(person.sample_count > 0, "enrolled with nothing to compare");
    assert!(
        !person.needs_refresh,
        "a profile just built by the network in use should be usable"
    );
    let profiles = repo::list_person_profiles(&db).await?;
    assert_eq!(profiles.len(), 1, "one enrolled person expected");
    let enrolment = people::enrolled(&db, &people::embedder_tag(&db, &embedder).await).await?;
    assert_eq!(
        enrolment.people.len(),
        1,
        "the profile was not loadable by the pass (stale: {})",
        enrolment.stale
    );

    // === 3. the same meeting again, with that person known ================
    let second = diarize::refine_speakers(&db, &real, &segmenter, &embedder).await?;
    let after = rows_of(&db, &real).await?;
    print_rows(
        &format!(
            "pass two — {PROBE_NAME:?} enrolled ({} voice(s) detected)",
            second.people_count
        ),
        &after,
        &profiles,
    );

    let linked: Vec<&Row> = after.iter().filter(|r| r.person_id.is_some()).collect();
    assert_eq!(
        linked.len(),
        1,
        "exactly one voice should have been recognised, {} were",
        linked.len()
    );
    let hit = linked[0];
    assert_eq!(
        hit.person_id.as_deref(),
        Some(person.id.as_str()),
        "linked to the wrong person"
    );
    assert_eq!(
        hit.display_name, PROBE_NAME,
        "the person's name was not copied onto the row Echo had made up a label for"
    );
    println!();
    println!(
        "AUTO-LINKED: {:?} → {:?} ({:.1} s of speech), name copied onto the row",
        hit.id, hit.display_name, hit.speaking_ms as f64 / 1_000.0
    );
    for row in after.iter().filter(|r| r.person_id.is_none()) {
        assert!(
            row.suggested_person_id.is_none(),
            "{}: a stranger in the same room was offered as {PROBE_NAME:?}",
            row.display_name
        );
        println!(
            "STAYED NOBODY: {:?} ({:.1} s) — no link, no question asked",
            row.display_name,
            row.speaking_ms as f64 / 1_000.0
        );
    }

    // === 4. a voice nobody ever enrolled ==================================
    println!();
    println!("=== a voice nobody enrolled: {stranger:?} ===");
    let work = tmp.path().join("work");
    std::fs::create_dir_all(&work)?;
    let fixture_dir = tmp.path().join("stranger");
    let fixture_chunks = render_stranger(&stranger, &work, &fixture_dir);
    let fixture = add_meeting(&db, "stranger", &fixture_dir, &fixture_chunks).await?;
    let third = diarize::refine_speakers(&db, &fixture, &segmenter, &embedder).await?;
    let stranger_rows = rows_of(&db, &fixture).await?;
    print_rows(
        &format!(
            "{stranger:?}, in a database that remembers {PROBE_NAME:?} ({} voice(s) detected)",
            third.people_count
        ),
        &stranger_rows,
        &profiles,
    );
    assert!(
        !stranger_rows.is_empty(),
        "the stranger fixture produced no speakers at all"
    );
    for row in &stranger_rows {
        assert!(
            row.person_id.is_none(),
            "{stranger:?} was claimed as {PROBE_NAME:?}"
        );
        assert!(
            row.suggested_person_id.is_none(),
            "{stranger:?} was offered as {PROBE_NAME:?}"
        );
    }
    println!();
    println!("LINKED TO NOBODY: {stranger:?} — no link and no question, which is the answer");

    println!();
    println!("all four steps held.");
    Ok(())
}
