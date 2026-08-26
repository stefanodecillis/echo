//! Measure the bars in [`echo_lib::diarize::people`] against voices whose
//! identity is known.
//!
//! `examples/voices_fixture.rs` established the procedure and the voice pool;
//! this points the same machinery at a different question. Clustering asks "are
//! these two the same?" among voices that are all present. Enrollment asks "is
//! this Marco, or somebody I have never heard?" — an **open-set** question, where
//! the right answer is usually "nobody", and a threshold measured on the closed
//! question says nothing about it.
//!
//! ```text
//! cargo run --release --example people_fixture -- [flags]
//!
//!   --segmenter P   ONNX segmenter. Default: the installed one.
//!   --embedder P    ONNX fingerprint network. Default: the installed one.
//!   --keep DIR      keep the rendered fixtures, so they can be listened to.
//! ```
//!
//! # The three fixtures
//!
//! 1. **Enrollment.** Two voices, three turns each. Run the real pass, cluster
//!    it, and build a profile per voice out of up to
//!    `people::SAMPLES_PER_CONFIRMATION` of its fingerprints — which is what one
//!    confirmation gives a person, so what is measured is a profile Echo could
//!    actually have on disk.
//! 2. **Mixed.** One of those two voices plus two that were never enrolled. The
//!    right person must be linked; neither stranger may be linked *or*
//!    suggested.
//! 3. **Strangers.** Three unenrolled voices. Nobody linked, nobody suggested.
//!
//! Getting the count right is not what is being measured here — that is
//! `voices_fixture`'s job. What is measured is, for every cluster the pass
//! produced, its similarity to every profile, which is exactly the input
//! `people::decide` sees in the app.
//!
//! # The honest limits
//!
//! Synthetic voices are cleaner and far more self-consistent than people: no
//! room, no crosstalk, no laughter, and `say` does not drift inside a sentence
//! the way a person does. So these fixtures can prove a bar **wrong** and can
//! show where the plateau of right answers begins and ends; they cannot prove one
//! optimal. Three fixtures are three fixtures. What the measurement rules out is
//! the indefensible thing: a bar picked by taste, in a space nobody checked.

use std::path::{Path, PathBuf};
use std::process::Command;

use echo_lib::asr::catalog;
use echo_lib::db::{self, repo};
use echo_lib::diarize::cluster;
use echo_lib::diarize::people;
use echo_lib::diarize::pipeline::{self, DiarizeControl};
use echo_lib::diarize::timeline::{self, Span};
use echo_lib::types::Channel;

const CHUNK_MS: i64 = 30_000;
const GAP_MS: i64 = 450;
const LEAD_MS: i64 = 500;
const TURNS_PER_VOICE: usize = 3;
const SENTENCES_PER_TURN: usize = 3;

/// The measured pool from `examples/voices_fixture.rs`: every pair at least 0.70
/// apart by centroid, every voice holding itself together within 0.46. Order
/// matters — the first two are the ones that get enrolled.
const VOICE_POOL: &[(&str, &str)] = &[
    ("Samantha", "en"),
    ("Daniel", "en"),
    ("Alice", "it"),
    ("Rishi", "en"),
    ("Tara", "en"),
    ("Grandpa (Italian (Italy))", "it"),
    ("Rocko (English (US))", "en"),
    ("Kathy", "en"),
    ("Albert", "en"),
    ("Fred", "en"),
];

const ENGLISH: &[&str] = &[
    "Thanks for making the time today, I know the week has been busy for everyone.",
    "Before we go further, can we agree on what actually ships at the end of the month?",
    "My worry is that we are solving the second problem before we understand the first one.",
    "I looked at the numbers again last night and they are better than we thought.",
    "Let me say that back to you, because I want to be sure I have understood it.",
    "If we push the date, the only thing that changes is who is disappointed later.",
];

const ITALIAN: &[&str] = &[
    "Grazie per il tempo di oggi, so che è stata una settimana molto piena per tutti.",
    "Prima di andare avanti, possiamo decidere che cosa consegniamo entro fine mese?",
    "Il mio dubbio è che stiamo risolvendo il secondo problema senza capire il primo.",
    "Ho guardato di nuovo i numeri ieri sera e sono migliori di quanto pensassimo.",
    "Ti ripeto quello che hai detto, perché voglio essere sicura di aver capito bene.",
    "Se spostiamo la data, l'unica cosa che cambia è chi resterà deluso più tardi.",
];

#[derive(Debug, Clone)]
struct Turn {
    voice: usize,
    span: Span,
}

fn installed_voices() -> Vec<String> {
    let out = Command::new("say")
        .arg("-v")
        .arg("?")
        .output()
        .expect("`say -v ?` — this fixture builder is macOS only");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|line| {
            let locale = line.split_whitespace().position(is_locale)?;
            let name: String = line
                .split_whitespace()
                .take(locale)
                .collect::<Vec<&str>>()
                .join(" ");
            (!name.is_empty()).then_some(name)
        })
        .collect()
}

fn same_voice(listed: &str, wanted: &str) -> bool {
    listed == wanted || listed.starts_with(&format!("{wanted} ("))
}

fn is_locale(token: &str) -> bool {
    let Some((lang, region)) = token.split_once('_') else {
        return false;
    };
    lang.len() == 2
        && lang.chars().all(|c| c.is_ascii_lowercase())
        && (2..=3).contains(&region.len())
        && region
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
}

fn render(voice: &str, text: &str, work: &Path, tag: &str) -> Vec<f32> {
    let aiff = work.join(format!("{tag}.aiff"));
    let wav = work.join(format!("{tag}.wav"));
    let said = Command::new("say")
        .args(["-v", voice, "-o"])
        .arg(&aiff)
        .arg(text)
        .status()
        .expect("run say");
    assert!(said.success(), "say -v {voice:?} failed");
    let _ = std::fs::remove_file(&wav);
    let converted = Command::new("afconvert")
        .args(["-f", "WAVE", "-d", "LEI16@16000", "-c", "1"])
        .arg(&aiff)
        .arg(&wav)
        .status()
        .expect("run afconvert");
    assert!(converted.success(), "afconvert failed for {tag}");
    let samples = echo_lib::audio::writer::read_wav_16k_mono(&wav).expect("read the rendered wav");
    let _ = std::fs::remove_file(&aiff);
    let _ = std::fs::remove_file(&wav);
    trim_silence(&samples)
}

fn trim_silence(samples: &[f32]) -> Vec<f32> {
    const FLOOR: f32 = 3e-3;
    let first = samples.iter().position(|x| x.abs() > FLOOR).unwrap_or(0);
    let last = samples
        .iter()
        .rposition(|x| x.abs() > FLOOR)
        .unwrap_or(samples.len().saturating_sub(1));
    samples[first..=last.max(first)].to_vec()
}

/// Lay the rendered turns end to end, and remember who is where. `salt` rotates
/// the sentences so the same voice never reads the same words in two fixtures.
fn build_meeting(voices: &[&str], work: &Path, salt: usize) -> (Vec<f32>, Vec<Turn>) {
    let per_ms = 16usize;
    let mut audio = vec![0.0f32; LEAD_MS as usize * per_ms];
    let mut turns: Vec<Turn> = Vec::new();

    for round in 0..TURNS_PER_VOICE {
        for (v, voice) in voices.iter().enumerate() {
            let italian = VOICE_POOL
                .iter()
                .any(|(name, lang)| same_voice(name, voice) && *lang == "it");
            let bank = if italian { ITALIAN } else { ENGLISH };
            let first = (salt + round * voices.len() + v) * SENTENCES_PER_TURN;
            let text: String = (0..SENTENCES_PER_TURN)
                .map(|i| bank[(first + i) % bank.len()])
                .collect::<Vec<&str>>()
                .join(" ");
            let speech = render(voice, &text, work, &format!("s{salt}-v{v}-r{round}"));
            let start_ms = (audio.len() / per_ms) as i64;
            audio.extend_from_slice(&speech);
            let end_ms = (audio.len() / per_ms) as i64;
            turns.push(Turn {
                voice: v,
                span: (start_ms, end_ms),
            });
            audio.extend(std::iter::repeat_n(0.0f32, GAP_MS as usize * per_ms));
        }
    }
    let chunk_samples = CHUNK_MS as usize * per_ms;
    let remainder = audio.len() % chunk_samples;
    if remainder != 0 {
        audio.extend(std::iter::repeat_n(0.0f32, chunk_samples - remainder));
    }
    (audio, turns)
}

fn write_chunks(audio: &[f32], dir: &Path) -> Vec<(i64, PathBuf)> {
    std::fs::create_dir_all(dir).expect("fixture directory");
    let per_chunk = CHUNK_MS as usize * 16;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut out = Vec::new();
    for (seq, block) in audio.chunks(per_chunk).enumerate() {
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

/// Attach every fingerprint to the true voice it spent most of its time on, with
/// the spans it came from. Only a fixture can do this, and only because it wrote
/// the turns itself.
fn attribute(
    scanned: &pipeline::Scan,
    turns: &[Turn],
    voices: usize,
) -> Vec<(usize, Vec<f32>, Vec<Span>)> {
    scanned
        .fingerprints_with_spans()
        .into_iter()
        .filter_map(|(embedding, spans)| {
            let mut per_voice = vec![0i64; voices];
            for turn in turns {
                for &span in &spans {
                    per_voice[turn.voice] += timeline::overlap_ms(span, turn.span);
                }
            }
            let (voice, ms) = per_voice
                .iter()
                .copied()
                .enumerate()
                .max_by_key(|(_, ms)| *ms)?;
            (ms > 0).then(|| (voice, embedding.to_vec(), spans))
        })
        .collect()
}

/// One fixture, scanned and cut, with everything the measurements need.
struct Fixture {
    name: String,
    voices: Vec<String>,
    /// Per fingerprint: which true voice, its numbers, where it came from.
    prints: Vec<(usize, Vec<f32>, Vec<Span>)>,
    /// Per cluster the pass produced: its centroid, and which true voice
    /// dominates it.
    clusters: Vec<(Vec<f32>, usize)>,
}

async fn run_fixture(
    name: &str,
    voices: &[&str],
    salt: usize,
    root: &Path,
    work: &Path,
    segmenter: &Path,
    embedder: &Path,
) -> Result<Fixture, Box<dyn std::error::Error>> {
    println!("=== {name}: {}", voices.join(", "));
    let (audio, turns) = build_meeting(voices, work, salt);
    let dir = root.join(format!("fixture-{name}"));
    let chunks = write_chunks(&audio, &dir);

    let db = db::connect(&root.join(format!("{name}.db"))).await?;
    let meeting = repo::create_meeting(&db, name, &dir.to_string_lossy(), Some("fixture")).await?;
    for (seq, path) in &chunks {
        let id = repo::insert_chunk(
            &db,
            &meeting.id,
            Channel::Mic,
            *seq,
            &path.to_string_lossy(),
            seq * CHUNK_MS,
            (seq + 1) * CHUNK_MS,
        )
        .await?;
        repo::commit_chunk(&db, &id, (seq + 1) * CHUNK_MS).await?;
    }

    let started = std::time::Instant::now();
    let scanned = pipeline::scan(
        &db,
        &meeting.id,
        segmenter,
        embedder,
        &DiarizeControl::new(),
    )
    .await?
    .expect("the fixture has audio");
    let cut = scanned.cut_for(None);
    println!(
        "    {} fingerprints of {} dimensions, {} clusters found (of {}), {:.1} s",
        scanned.fingerprints(),
        scanned.fingerprint_dim().unwrap_or(0),
        cut.tracks.len(),
        voices.len(),
        started.elapsed().as_secs_f32(),
    );

    let prints = attribute(&scanned, &turns, voices.len());

    // A cluster's centroid, built the way the pass will store it: the mean of
    // the fingerprints whose audio lies inside its timeline.
    let mut clusters: Vec<(Vec<f32>, usize)> = Vec::new();
    for track in &cut.tracks {
        let mine: Vec<&Vec<f32>> = prints
            .iter()
            .filter(|(_, _, spans)| {
                spans
                    .iter()
                    .any(|&s| track.iter().any(|&t| timeline::overlap_ms(s, t) > 0))
            })
            .map(|(_, e, _)| e)
            .collect();
        if mine.is_empty() {
            continue;
        }
        let centroid = people::centroid_of(mine.iter().map(|e| e.as_slice()));
        // Which true voice this cluster mostly holds.
        let mut per_voice = vec![0i64; voices.len()];
        for turn in &turns {
            for &span in track {
                per_voice[turn.voice] += timeline::overlap_ms(span, turn.span);
            }
        }
        let dominant = per_voice
            .iter()
            .enumerate()
            .max_by_key(|(_, ms)| **ms)
            .map(|(v, _)| v)
            .unwrap_or(0);
        clusters.push((centroid, dominant));
    }

    Ok(Fixture {
        name: name.to_string(),
        voices: voices.iter().map(|v| v.to_string()).collect(),
        prints,
        clusters,
    })
}

fn asset_path(id: &str, override_path: Option<&PathBuf>) -> PathBuf {
    if let Some(p) = override_path {
        return p.clone();
    }
    let home = std::env::var("HOME").expect("HOME");
    let entry = catalog::entry(id).expect("catalogued asset");
    Path::new(&home)
        .join("Library/Application Support/Echo/speech")
        .join(entry.file_name)
}

/// The margin rule at a given pair of bars, so the sweep can try bars the
/// shipped constants do not have.
fn verdict_at(scores: &[f32], tau_link: f32, tau_suggest: f32, margin: f32) -> &'static str {
    let mut best = (usize::MAX, f32::NEG_INFINITY);
    let mut second = f32::NEG_INFINITY;
    for (i, &s) in scores.iter().enumerate() {
        if s > best.1 {
            second = best.1;
            best = (i, s);
        } else if s > second {
            second = s;
        }
    }
    if best.0 == usize::MAX {
        return "nobody";
    }
    let floor = if second.is_finite() { second } else { 0.0 };
    if best.1 - floor.max(0.0) < margin {
        return "nobody";
    }
    if best.1 >= tau_link {
        "link"
    } else if best.1 >= tau_suggest {
        "suggest"
    } else {
        "nobody"
    }
}

fn range(values: &[f32]) -> (f32, f32) {
    let lo = values.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (lo, hi)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut segmenter_arg: Option<PathBuf> = None;
    let mut embedder_arg: Option<PathBuf> = None;
    let mut keep: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--segmenter" => segmenter_arg = args.next().map(PathBuf::from),
            "--embedder" => embedder_arg = args.next().map(PathBuf::from),
            "--keep" => keep = args.next().map(PathBuf::from),
            other => return Err(format!("unknown flag {other}").into()),
        }
    }

    let segmenter = asset_path(catalog::ids::SEGMENTER, segmenter_arg.as_ref());
    let embedder = asset_path(catalog::ids::EMBEDDER, embedder_arg.as_ref());
    for (what, path) in [("segmenter", &segmenter), ("fingerprints", &embedder)] {
        if !path.exists() {
            return Err(format!("{what} is not at {}", path.display()).into());
        }
    }
    println!("segmenter:    {}", segmenter.display());
    println!("fingerprints: {}", embedder.display());

    let available = installed_voices();
    let pool: Vec<&str> = VOICE_POOL
        .iter()
        .map(|(name, _)| *name)
        .filter(|name| available.iter().any(|v| same_voice(v, name)))
        .collect();
    if pool.len() < 5 {
        return Err(format!("only {} of the pool's voices are installed", pool.len()).into());
    }
    println!("voices available: {} of {}", pool.len(), VOICE_POOL.len());
    println!();

    let tmp = tempfile::tempdir()?;
    let root = keep.clone().unwrap_or_else(|| tmp.path().to_path_buf());
    let work = tmp.path().join("render");
    std::fs::create_dir_all(&work)?;

    // 1. enrollment: the first two voices.
    let enrolment = run_fixture(
        "enrollment",
        &pool[0..2],
        0,
        &root,
        &work,
        &segmenter,
        &embedder,
    )
    .await?;

    // Profiles, built the way one confirmation builds them: up to
    // SAMPLES_PER_CONFIRMATION fingerprints of that voice, spread through the
    // meeting rather than taken from one turn.
    let mut profiles: Vec<(String, Vec<f32>)> = Vec::new();
    for (v, name) in enrolment.voices.iter().enumerate() {
        let mut mine: Vec<&Vec<f32>> = enrolment
            .prints
            .iter()
            .filter(|(voice, _, _)| *voice == v)
            .map(|(_, e, _)| e)
            .collect();
        // Spread: take every k-th so the samples come from different turns.
        let step = (mine.len() / people::SAMPLES_PER_CONFIRMATION.max(1)).max(1);
        mine = mine
            .into_iter()
            .step_by(step)
            .take(people::SAMPLES_PER_CONFIRMATION)
            .collect();
        println!(
            "    profile for {name}: {} sample(s) of {} available",
            mine.len(),
            enrolment
                .prints
                .iter()
                .filter(|(voice, _, _)| *voice == v)
                .count()
        );
        profiles.push((
            name.clone(),
            people::centroid_of(mine.iter().map(|e| e.as_slice())),
        ));
    }
    // How far the two enrolled profiles are from each other: the ceiling on any
    // margin that could ever be demanded.
    println!(
        "    the two profiles sit {:.3} apart (similarity {:.3})",
        cluster::cosine_distance(&profiles[0].1, &profiles[1].1),
        people::similarity(&profiles[0].1, &profiles[1].1),
    );
    println!();

    // 2. mixed: one enrolled voice plus two strangers.
    let mixed_voices: Vec<&str> = vec![pool[0], pool[2], pool[3]];
    let mixed = run_fixture(
        "mixed",
        &mixed_voices,
        1,
        &root,
        &work,
        &segmenter,
        &embedder,
    )
    .await?;
    println!();

    // 3. strangers only.
    let stranger_voices: Vec<&str> = vec![pool[2], pool[3], pool[4]];
    let strangers = run_fixture(
        "strangers",
        &stranger_voices,
        2,
        &root,
        &work,
        &segmenter,
        &embedder,
    )
    .await?;
    println!();

    // --- cluster-level scores, which is what `decide` sees ------------------
    println!("=== every cluster against every profile ===");
    println!(
        "{:<12}{:<28}{:>10}{:>10}{:>9}   verdict at the shipped bars",
        "fixture", "cluster is really", "best", "second", "margin"
    );
    let mut true_scores: Vec<f32> = Vec::new();
    let mut stranger_scores: Vec<f32> = Vec::new();
    let mut true_margins: Vec<f32> = Vec::new();
    let mut rows: Vec<(String, bool, Vec<f32>)> = Vec::new();

    for fixture in [&mixed, &strangers] {
        for (centroid, dominant) in &fixture.clusters {
            let scores: Vec<f32> = profiles
                .iter()
                .map(|(_, c)| people::similarity(centroid, c))
                .collect();
            let voice = &fixture.voices[*dominant];
            let enrolled_index = profiles.iter().position(|(name, _)| name == voice);
            let mut sorted = scores.clone();
            sorted.sort_by(|a, b| b.total_cmp(a));
            let best = sorted[0];
            let second = sorted.get(1).copied().unwrap_or(0.0);
            let verdict = verdict_at(
                &scores,
                people::TAU_LINK,
                people::TAU_SUGGEST,
                people::MARGIN,
            );
            let correct = match enrolled_index {
                Some(i) => verdict == "link" && scores[i] == best,
                None => verdict == "nobody",
            };
            println!(
                "{:<12}{:<28}{:>10.3}{:>10.3}{:>9.3}   {verdict}{}",
                fixture.name,
                format!(
                    "{voice}{}",
                    if enrolled_index.is_some() {
                        " (enrolled)"
                    } else {
                        ""
                    }
                ),
                best,
                second,
                best - second,
                if correct { "" } else { "   <- WRONG" },
            );
            match enrolled_index {
                Some(i) => {
                    true_scores.push(scores[i]);
                    true_margins.push(best - second);
                }
                None => stranger_scores.push(best),
            }
            rows.push((fixture.name.clone(), enrolled_index.is_some(), scores));
        }
    }
    println!();
    if !true_scores.is_empty() {
        let (lo, hi) = range(&true_scores);
        println!("enrolled voice against its own profile   {lo:.3} … {hi:.3}");
        let (mlo, mhi) = range(&true_margins);
        println!("its margin over the runner-up            {mlo:.3} … {mhi:.3}");
    }
    if !stranger_scores.is_empty() {
        let (lo, hi) = range(&stranger_scores);
        println!("stranger against the nearest profile     {lo:.3} … {hi:.3}");
    }
    if let (false, false) = (true_scores.is_empty(), stranger_scores.is_empty()) {
        let (_, worst_stranger) = range(&stranger_scores);
        let (weakest_true, _) = range(&true_scores);
        println!(
            "the gap the bars live in                 {worst_stranger:.3} … {weakest_true:.3}"
        );
    }
    println!();

    // --- the plateau -------------------------------------------------------
    println!(
        "=== every TAU_LINK that gets all three fixtures right (MARGIN = {:.2}) ===",
        people::MARGIN
    );
    let mut right: Vec<f32> = Vec::new();
    let mut step = 0usize;
    loop {
        let tau = 0.20f32 + step as f32 * 0.02;
        if tau > 0.96 {
            break;
        }
        step += 1;
        let all_right = rows.iter().all(|(_, enrolled, scores)| {
            let verdict = verdict_at(scores, tau, people::TAU_SUGGEST.min(tau), people::MARGIN);
            if *enrolled {
                verdict == "link"
            } else {
                verdict == "nobody"
            }
        });
        if all_right {
            right.push((tau * 1_000.0).round() / 1_000.0);
        }
    }
    match (right.first(), right.last()) {
        (Some(lo), Some(hi)) => println!(
            "{lo:.3} … {hi:.3}   (shipped {:.2}; margin below {:.3}, above {:.3})",
            people::TAU_LINK,
            people::TAU_LINK - lo,
            hi - people::TAU_LINK
        ),
        _ => println!("nothing in the sweep gets every fixture right"),
    }

    println!();
    println!("=== the lowest TAU_SUGGEST that still says nothing about a stranger ===");
    let mut floor = 0.0f32;
    let mut step = 0usize;
    loop {
        let tau = 0.20f32 + step as f32 * 0.02;
        if tau > people::TAU_LINK {
            break;
        }
        step += 1;
        let quiet = rows
            .iter()
            .filter(|(_, enrolled, _)| !enrolled)
            .all(|(_, _, scores)| {
                verdict_at(scores, people::TAU_LINK, tau, people::MARGIN) == "nobody"
            });
        if quiet && floor == 0.0 {
            floor = tau;
        }
    }
    println!("{floor:.3}   (shipped {:.2})", people::TAU_SUGGEST);

    // --- fingerprint-level scores, which is what `pre_assign` sees ---------
    println!();
    println!("=== single fingerprints against the profiles (TAU_STRONG) ===");
    let mut true_prints: Vec<f32> = Vec::new();
    let mut true_print_margins: Vec<f32> = Vec::new();
    let mut stranger_prints: Vec<f32> = Vec::new();
    let mut wrongly_pre_assigned = 0usize;
    let mut rightly_pre_assigned = 0usize;
    let mut missed = 0usize;

    for fixture in [&mixed, &strangers] {
        for (voice, embedding, _) in &fixture.prints {
            let scores: Vec<f32> = profiles
                .iter()
                .map(|(_, c)| people::similarity(embedding, c))
                .collect();
            let name = &fixture.voices[*voice];
            let enrolled_index = profiles.iter().position(|(n, _)| n == name);
            let mut sorted = scores.clone();
            sorted.sort_by(|a, b| b.total_cmp(a));
            let best = sorted[0];
            let second = sorted.get(1).copied().unwrap_or(0.0);
            let decided = people::decide_strong(&scores);
            match enrolled_index {
                Some(i) => {
                    true_prints.push(scores[i]);
                    true_print_margins.push(best - second);
                    match decided {
                        Some((who, _)) if who == i => rightly_pre_assigned += 1,
                        Some(_) => wrongly_pre_assigned += 1,
                        None => missed += 1,
                    }
                }
                None => {
                    stranger_prints.push(best);
                    if decided.is_some() {
                        wrongly_pre_assigned += 1;
                    }
                }
            }
        }
    }
    if !true_prints.is_empty() {
        let (lo, hi) = range(&true_prints);
        println!("enrolled voice, one fingerprint at a time   {lo:.3} … {hi:.3}");
        let (mlo, mhi) = range(&true_print_margins);
        println!("its margin over the runner-up               {mlo:.3} … {mhi:.3}");
    }
    if !stranger_prints.is_empty() {
        let (lo, hi) = range(&stranger_prints);
        println!("stranger, one fingerprint at a time         {lo:.3} … {hi:.3}");
    }
    println!(
        "at the shipped TAU_STRONG {:.2}: {rightly_pre_assigned} pre-assigned correctly, \
         {missed} left to the clustering, {wrongly_pre_assigned} pre-assigned WRONGLY",
        people::TAU_STRONG
    );
    println!();
    println!(
        "{}",
        if wrongly_pre_assigned == 0 {
            "no fingerprint was ever handed to the wrong person"
        } else {
            "a fingerprint was pre-assigned to the wrong person — that is a release blocker"
        }
    );

    if let Some(dir) = &keep {
        println!();
        println!("fixtures kept in {}", dir.display());
    }
    Ok(())
}
