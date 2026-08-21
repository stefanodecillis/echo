//! Build synthetic meetings out of macOS's own text-to-speech voices, run the
//! real speaker pass over them, and sweep the clustering threshold.
//!
//! This is the procedure behind the number in
//! [`echo_lib::diarize::cluster::DISTANCE_THRESHOLD`]. Its whole reason to exist
//! is that a diarization threshold cannot be reasoned about — it has to be
//! measured against audio where the answer is already known — and real meetings
//! are the one kind of audio we can never ship as a fixture.
//!
//! ```text
//! cargo run --release --example voices_fixture -- [flags]
//!
//!   --segmenter P    ONNX segmenter to use. Default: the installed one.
//!   --embedder P     ONNX fingerprint network. Default: the installed one.
//!                    Point this at a candidate file to calibrate it before it
//!                    goes anywhere near the catalog.
//!   --sizes 2,3,5,7  how many known speakers per fixture.
//!   --keep DIR       write the rendered fixtures here instead of a temp dir,
//!                    so they can be listened to.
//!   --diagnose       also print, turn by turn, which cluster claimed it.
//!   --matrix         also print the full voice-to-voice distance matrix.
//!   --voices "A,B"   use these voices instead of the pool, in this order.
//! ```
//!
//! # How a fixture is made
//!
//! `say -v ?` lists the voices installed on this machine. [`VOICE_POOL`] picks
//! from it: real speaking voices only (no singing or robot novelty voices),
//! deliberately mixed across accent, sex and language — several Italian voices
//! among the English ones, because a fingerprint network trained on VoxCeleb
//! should not be quietly better at one of them.
//!
//! Each voice reads its own set of sentences. Every sentence is rendered with
//! `say -o … .aiff`, converted to 16 kHz mono with `afconvert` — the same rate
//! and layout Echo records at — and then laid end to end in a round-robin of
//! turns with a short gap between each, which is what a conversation looks like
//! to the segmenter. The result is written as 30 s WAV chunks named the way Echo
//! names them, so [`echo_lib::diarize::pipeline::scan`] reads them exactly as it
//! reads a real recording. Nothing is stubbed: same segmenter, same features,
//! same fingerprints, same clustering.
//!
//! Ground truth is the turn list, which is known exactly because we built it.
//!
//! # What the sweep reports
//!
//! For each fixture, [`echo_lib::diarize::pipeline::scan`] runs **once** — the
//! expensive part — and then the same fingerprints are cut at every threshold in
//! the sweep. Per (fixture, threshold) it prints:
//!
//! * **found** — how many people the cut produced, against how many there were.
//! * **purity** — of all the speech a cluster claims, the share that belongs to
//!   its single most-represented true voice, averaged over clusters and weighted
//!   by how much each one holds. 1.00 means every cluster is one person.
//! * **coverage** — the share of true speech that landed on any cluster at all.
//!   It can read slightly over 1.0: where two clusters both claim the same
//!   millisecond — which is what overlap-aware decoding is *for* — that
//!   millisecond is counted once per claimant.
//!
//! Getting the *count* right is necessary but not sufficient: two clusters each
//! holding half of both voices is also "2". Purity is what rules that out.
//!
//! # The honest limits
//!
//! Synthetic voices are cleaner than people. No room, no overlapping speech, no
//! crosstalk, no laughter, and each voice is perfectly consistent from turn to
//! turn — a real person's voice moves more than that in one sentence. So these
//! fixtures can prove a threshold *wrong*, and can show where the plateau of
//! right answers begins and ends, but they cannot prove one optimal. The guard
//! against overfitting to them is `examples/speakers_probe.rs`, which runs the
//! same pass over a real recorded meeting.

use std::path::{Path, PathBuf};
use std::process::Command;

use echo_lib::asr::catalog;
use echo_lib::db::{self, repo};
use echo_lib::diarize::pipeline::{self, DiarizeControl};
use echo_lib::diarize::timeline::{self, Span};
use echo_lib::types::Channel;

/// Chunk length Echo records at, so the fixture is laid out like a real meeting.
const CHUNK_MS: i64 = 30_000;
/// Silence between two turns. Long enough that the segmenter sees two turns
/// rather than one (`segmentation::MIN_GAP_MS` is 200 ms), short enough to look
/// like people talking rather than taking minutes.
const GAP_MS: i64 = 450;
/// Silence at the very start, so the first turn is not clipped by the window.
const LEAD_MS: i64 = 500;
/// How many turns each voice takes. Three separate stretches per voice, spread
/// through the meeting, so no voice is fingerprinted from one continuous piece of
/// audio.
const TURNS_PER_VOICE: usize = 3;

/// Sentences in one turn.
///
/// This one is not cosmetic. The segmenter sees a 10 s window and has three local
/// speaker slots to hand out; if a window holds two short turns by two different
/// people it may well put both in one slot, and the fingerprint taken from that
/// slot is then a blend of two voices. A blended fingerprint makes a voice look
/// like it disagrees with itself, which no threshold can repair.
///
/// Three sentences is about twelve seconds — a turn long enough that most windows
/// hold one person, which is also what a real meeting sounds like. An earlier
/// version of this fixture used one sentence per turn and measured a within-voice
/// spread of 0.68, worse than the distance between different people; that number
/// was an artefact of the fixture, not a fact about the network.
const SENTENCES_PER_TURN: usize = 3;

/// Voices worth building a fixture from, in the order fixtures take them.
///
/// **Chosen by measurement, not by taste.** The first draft of this list was
/// picked by reading `say -v ?` and taking voices that sound different to a
/// person, and it was wrong: run `--matrix` over the whole installed set and it
/// turns out macOS ships families of voices that a voice-print network cannot
/// tell apart at all. Karen and Moira are 0.17 apart. Eddy and Reed are 0.16.
/// The Italian Eddy and the Italian Reed are 0.12 — closer together than any one
/// voice is to itself. A fixture built from a pair like that is not ground truth
/// for anything; it is a demand that the pipeline distinguish two things that are
/// not distinct.
///
/// So the pool below is the surviving subset: every pair at least 0.70 apart by
/// centroid, every voice holding itself together within 0.46. It still spans
/// accents (US, UK, Indian) and two languages, and alternates male and female so
/// a two-person fixture is not two voices of the same pitch — but those are
/// tie-breakers among voices that already passed the measurement.
///
/// Note what this selection is *not*: it is not chosen against
/// [`echo_lib::diarize::DISTANCE_THRESHOLD`]. Separability is a property of the
/// audio, measured before any threshold is applied, and the sweep is free to say
/// the shipped number is wrong.
///
/// Order matters: a fixture of *n* people takes the first *n* installed, so the
/// two-person fixture gets the easiest pair and the seven-person one has to work
/// hardest. Rerun with `--matrix` after a macOS upgrade; the voices change.
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

/// Sentences long enough to fingerprint. Anything under a second is thrown away
/// by `embedding::MIN_EMBED_MS`, so these are deliberately whole thoughts rather
/// than "yes" and "mm".
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

/// One turn of the synthetic meeting: who spoke, and exactly when.
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
            // "Samantha            en_US    # Hello! …", but also
            // "Grandpa (Italian (Italy)) it_IT   # Ciao! …" — a long name
            // overflows the column and leaves only one space, so the reliable
            // landmark is the locale, not the whitespace. The name is everything
            // before it.
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

/// Is `listed` the voice called `wanted`?
///
/// `say -v ?` sometimes prints a bare name ("Samantha") and sometimes a
/// disambiguated one ("Tara (English (India))") for the same voice that `say -v
/// Tara` speaks with. Both forms have to count, or the fixture silently shrinks.
fn same_voice(listed: &str, wanted: &str) -> bool {
    listed == wanted || listed.starts_with(&format!("{wanted} ("))
}

/// "en_US", "ar_001", "zh_CN" — a BCP-ish locale in the shape `say` prints it.
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

/// Render one sentence in one voice as 16 kHz mono `f32`.
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

    // 16 kHz, mono, 16-bit: what Echo records, so the fixture exercises the same
    // resampling-free path a real chunk does.
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

/// Drop the leading and trailing silence `say` leaves around an utterance, so a
/// turn's ground-truth span is the speech and not the padding.
fn trim_silence(samples: &[f32]) -> Vec<f32> {
    const FLOOR: f32 = 3e-3;
    let first = samples.iter().position(|x| x.abs() > FLOOR).unwrap_or(0);
    let last = samples
        .iter()
        .rposition(|x| x.abs() > FLOOR)
        .unwrap_or(samples.len().saturating_sub(1));
    samples[first..=last.max(first)].to_vec()
}

/// Lay the rendered turns end to end into one recording, and remember who is
/// where.
fn build_meeting(voices: &[&str], work: &Path) -> (Vec<f32>, Vec<Turn>) {
    let per_ms = 16usize; // samples per millisecond at 16 kHz
    let mut audio = vec![0.0f32; LEAD_MS as usize * per_ms];
    let mut turns: Vec<Turn> = Vec::new();

    for round in 0..TURNS_PER_VOICE {
        for (v, voice) in voices.iter().enumerate() {
            // Italian voices read Italian. Anything not in the pool is assumed
            // English, which is what an unlisted `--voices` name gets.
            let italian = VOICE_POOL
                .iter()
                .any(|(name, lang)| same_voice(name, voice) && *lang == "it");
            let bank = if italian { ITALIAN } else { ENGLISH };
            // Rotate the sentences per (voice, round) so no two voices ever read
            // the same words at the same point — otherwise a cluster could be
            // agreeing about the sentence rather than about the speaker.
            let first = (round * voices.len() + v) * SENTENCES_PER_TURN;
            let text: String = (0..SENTENCES_PER_TURN)
                .map(|i| bank[(first + i) % bank.len()])
                .collect::<Vec<&str>>()
                .join(" ");

            let speech = render(voice, &text, work, &format!("v{v}-r{round}"));
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
    // Pad to a whole number of chunks, the way a real recording ends.
    let chunk_samples = CHUNK_MS as usize * per_ms;
    let remainder = audio.len() % chunk_samples;
    if remainder != 0 {
        audio.extend(std::iter::repeat_n(0.0f32, chunk_samples - remainder));
    }
    (audio, turns)
}

/// Write the recording out as the 30 s mono WAV chunks Echo produces.
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
            let clamped = (x * 32_767.0).clamp(-32_768.0, 32_767.0) as i16;
            writer.write_sample(clamped).expect("write sample");
        }
        writer.finalize().expect("finalize chunk");
        out.push((seq as i64, path));
    }
    out
}

/// How well one cut of the fingerprints matches the truth.
#[derive(Debug, Clone, Copy, Default)]
struct Score {
    found: usize,
    purity: f32,
    coverage: f32,
}

/// Compare a cut's per-cluster timelines against the turns we built.
fn score(tracks: &[Vec<Span>], turns: &[Turn], voices: usize) -> Score {
    let truth_ms: i64 = turns.iter().map(|t| timeline::span_ms(t.span)).sum();
    let mut attributed = 0i64;
    let mut dominant = 0i64;

    for track in tracks {
        // Time this cluster shares with each true voice.
        let mut per_voice = vec![0i64; voices];
        for turn in turns {
            for &span in track {
                per_voice[turn.voice] += timeline::overlap_ms(span, turn.span);
            }
        }
        let total: i64 = per_voice.iter().sum();
        if total == 0 {
            continue;
        }
        attributed += total;
        dominant += per_voice.iter().copied().max().unwrap_or(0);
    }

    Score {
        found: tracks.len(),
        purity: if attributed > 0 {
            dominant as f32 / attributed as f32
        } else {
            0.0
        },
        coverage: if truth_ms > 0 {
            attributed as f32 / truth_ms as f32
        } else {
            0.0
        },
    }
}

/// What the fingerprints look like as pure geometry, before any threshold gets a
/// vote.
///
/// This is the measurement a threshold is really made of. `within` is how far
/// apart two fingerprints of the *same* voice get; `between` is how close two
/// fingerprints of *different* voices get. Any threshold above the largest
/// `within` and below the smallest `between` separates this fixture perfectly,
/// and the gap between those two numbers is the margin. When they cross, no
/// threshold can do it and the sweep is not the thing at fault.
#[derive(Debug, Clone, Copy, Default)]
struct Geometry {
    within_max: f32,
    between_min: f32,
    /// The pair of voices that came closest, for naming names.
    closest: (usize, usize),
}

/// Attach every fingerprint to the true voice it spent most of its time on.
/// Only a fixture can do this, and only because it wrote the turns itself.
fn attribute(scanned: &pipeline::Scan, turns: &[Turn], voices: usize) -> Vec<(usize, Vec<f32>)> {
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
            (ms > 0).then(|| (voice, embedding.to_vec()))
        })
        .collect()
}

fn geometry(scanned: &pipeline::Scan, turns: &[Turn], voices: usize) -> Option<Geometry> {
    let owned = attribute(scanned, turns, voices);
    if owned.len() < 2 {
        return None;
    }

    let mut within_max = 0.0f32;
    let mut between_min = f32::INFINITY;
    let mut closest = (0usize, 0usize);
    for (i, (vi, a)) in owned.iter().enumerate() {
        for (vj, b) in owned.iter().skip(i + 1) {
            let d = echo_lib::diarize::cluster::cosine_distance(a, b);
            if vi == vj {
                within_max = within_max.max(d);
            } else if d < between_min {
                between_min = d;
                closest = (*vi.min(vj), *vi.max(vj));
            }
        }
    }
    Some(Geometry {
        within_max,
        between_min,
        closest,
    })
}

/// The whole distance matrix between true voices, plus each voice's own spread.
///
/// The tool for *choosing* a fixture rather than scoring one: two voices whose
/// centroids sit closer together than either one sits to itself are not two
/// speakers as far as any voice-print network is concerned, and a fixture built
/// from them cannot be ground truth for anything.
fn print_matrix(scanned: &pipeline::Scan, turns: &[Turn], voices: &[&str]) {
    let owned = attribute(scanned, turns, voices.len());
    let mut centroid: Vec<Vec<f32>> = vec![Vec::new(); voices.len()];
    let mut spread = vec![0.0f32; voices.len()];
    for v in 0..voices.len() {
        let mine: Vec<&Vec<f32>> = owned
            .iter()
            .filter(|(voice, _)| *voice == v)
            .map(|(_, e)| e)
            .collect();
        if mine.is_empty() {
            continue;
        }
        let dim = mine[0].len();
        let mut sum = vec![0.0f32; dim];
        for e in &mine {
            for (slot, x) in sum.iter_mut().zip(e.iter()) {
                *slot += x;
            }
        }
        echo_lib::diarize::cluster::l2_normalize(&mut sum);
        for a in 0..mine.len() {
            for b in (a + 1)..mine.len() {
                spread[v] = spread[v].max(echo_lib::diarize::cluster::cosine_distance(
                    mine[a], mine[b],
                ));
            }
        }
        centroid[v] = sum;
    }

    println!();
    println!(
        "    {:<30}{:>7}  distance to each other voice",
        "voice", "spread"
    );
    for v in 0..voices.len() {
        print!("    {:<30}{:>7.3} ", voices[v], spread[v]);
        for w in 0..voices.len() {
            if v == w || centroid[v].is_empty() || centroid[w].is_empty() {
                print!("    -");
                continue;
            }
            print!(
                " {:.2}",
                echo_lib::diarize::cluster::cosine_distance(&centroid[v], &centroid[w])
            );
        }
        println!();
    }
}

/// Print, turn by turn, which cluster claimed it. The one view that tells a
/// threshold problem apart from a segmentation problem: if two consecutive turns
/// by *different* voices land on one cluster over and over, no threshold will fix
/// it, because the fingerprints themselves were computed from mixed audio.
fn print_turn_map(tracks: &[Vec<Span>], turns: &[Turn], voices: &[&str]) {
    println!("    turn  voice                         cluster");
    for (i, turn) in turns.iter().enumerate() {
        let best = tracks
            .iter()
            .enumerate()
            .map(|(c, track)| {
                let ms: i64 = track
                    .iter()
                    .map(|&s| timeline::overlap_ms(s, turn.span))
                    .sum();
                (c, ms)
            })
            .max_by_key(|(_, ms)| *ms)
            .filter(|(_, ms)| *ms > 0);
        println!(
            "    {i:>4}  {:<28}  {}",
            voices[turn.voice],
            match best {
                Some((c, ms)) => format!(
                    "{c}   ({:.0}% of the turn)",
                    100.0 * ms as f64 / timeline::span_ms(turn.span).max(1) as f64
                ),
                None => "-".to_string(),
            }
        );
    }
}

/// The sweep. Fine around the shipped value, coarse at the ends — the shape of
/// the plateau near 0.7 is what matters, not the far tails.
fn sweep() -> Vec<f32> {
    let mut out: Vec<f32> = Vec::new();
    let mut step = 0usize;
    loop {
        let t = 0.15f32 + step as f32 * 0.0125;
        if t > 1.101 {
            break;
        }
        out.push((t * 10_000.0).round() / 10_000.0);
        step += 1;
    }
    if !out.contains(&echo_lib::diarize::DISTANCE_THRESHOLD) {
        out.push(echo_lib::diarize::DISTANCE_THRESHOLD);
        out.sort_by(f32::total_cmp);
    }
    out
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut segmenter_arg: Option<PathBuf> = None;
    let mut embedder_arg: Option<PathBuf> = None;
    let mut keep: Option<PathBuf> = None;
    let mut sizes: Vec<usize> = vec![2, 3, 5, 7];
    let mut diagnose = false;
    let mut matrix = false;
    let mut voice_override: Option<Vec<String>> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--segmenter" => segmenter_arg = args.next().map(PathBuf::from),
            "--embedder" => embedder_arg = args.next().map(PathBuf::from),
            "--keep" => keep = args.next().map(PathBuf::from),
            "--diagnose" => diagnose = true,
            "--matrix" => matrix = true,
            "--voices" => {
                voice_override = args
                    .next()
                    .map(|list| list.split(',').map(|v| v.trim().to_string()).collect());
            }
            "--sizes" => {
                if let Some(list) = args.next() {
                    sizes = list
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                }
            }
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
    let wanted: Vec<String> = voice_override.clone().unwrap_or_else(|| {
        VOICE_POOL
            .iter()
            .map(|(name, _)| (*name).to_string())
            .collect()
    });
    let pool: Vec<&str> = wanted
        .iter()
        .filter(|name| available.iter().any(|v| same_voice(v, name)))
        .map(|s| s.as_str())
        .collect();
    println!("voices available: {} of {}", pool.len(), wanted.len());
    println!();

    let tmp = tempfile::tempdir()?;
    let root = keep.clone().unwrap_or_else(|| tmp.path().to_path_buf());
    let work = tmp.path().join("render");
    std::fs::create_dir_all(&work)?;

    let thresholds = sweep();
    // fixture size -> threshold -> score
    let mut table: Vec<(usize, Vec<(f32, Score)>)> = Vec::new();
    // fixture size -> what the automatic count made of it
    let mut automatic: Vec<(usize, Score, Option<echo_lib::diarize::cluster::CountChoice>)> =
        Vec::new();

    for &n in &sizes {
        if pool.len() < n {
            println!(
                "skipping the {n}-voice fixture: only {} voices installed",
                pool.len()
            );
            continue;
        }
        let voices: Vec<&str> = pool[..n].to_vec();
        println!("=== {n} known speakers: {}", voices.join(", "));

        let (audio, turns) = build_meeting(&voices, &work);
        let dir = root.join(format!("fixture-{n}"));
        let chunks = write_chunks(&audio, &dir);
        let speech_ms: i64 = turns.iter().map(|t| timeline::span_ms(t.span)).sum();
        println!(
            "    {:.1} s of recording, {:.1} s of speech in {} turns, {} chunk(s)",
            audio.len() as f64 / 16_000.0,
            speech_ms as f64 / 1_000.0,
            turns.len(),
            chunks.len()
        );

        // A throwaway database with the fixture's chunks committed on the
        // microphone channel — a meeting recorded on speakers, which is the shape
        // that makes the clustering responsible for every voice.
        let db = db::connect(&tmp.path().join(format!("fixture-{n}.db"))).await?;
        let meeting = repo::create_meeting(
            &db,
            "voice fixture",
            &dir.to_string_lossy(),
            Some("fixture"),
        )
        .await?;
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
            &segmenter,
            &embedder,
            &DiarizeControl::new(),
        )
        .await?
        .expect("the fixture has audio");
        let elapsed = started.elapsed();
        println!(
            "    {} fingerprints of {} dimensions, {:.1} s ({:.2}x real time)",
            scanned.fingerprints(),
            scanned.fingerprint_dim().unwrap_or(0),
            elapsed.as_secs_f32(),
            elapsed.as_secs_f64() / (audio.len() as f64 / 16_000.0)
        );

        if let Some(g) = geometry(&scanned, &turns, n) {
            println!(
                "    same voice, furthest apart {:.3}   different voices, closest {:.3}  ({} / {})",
                g.within_max, g.between_min, voices[g.closest.0], voices[g.closest.1]
            );
            if g.between_min <= g.within_max {
                println!(
                    "    -> no threshold can separate this fixture: the closest two voices are \
                     nearer than one voice is to itself"
                );
            } else {
                println!(
                    "    -> any threshold in {:.3} … {:.3} separates it exactly",
                    g.within_max, g.between_min
                );
            }
        }

        // What the app actually does now: the count read out of the shape of the
        // merge tree, no threshold consulted. The sweep below is kept because it
        // is what placed the bounds the criterion still uses, but *this* is the
        // line that has to say `n`.
        let auto = scanned.cut_for(None);
        let auto_score = score(&auto.tracks, &turns, n);
        automatic.push((n, auto_score, auto.choice.clone()));
        println!(
            "    automatic: {} people (of {n}), purity {:.3}, coverage {:.3}{}",
            auto_score.found,
            auto_score.purity,
            auto_score.coverage,
            match &auto.choice {
                Some(c) => format!(
                    ", silhouette {:.3}{}",
                    c.silhouette,
                    c.runner_up
                        .map(|(k, s)| format!(" against {k} at {s:.3}"))
                        .unwrap_or_default()
                ),
                None => String::new(),
            }
        );

        let mut row: Vec<(f32, Score)> = Vec::new();
        for &t in &thresholds {
            let cut = scanned.cut(t, None);
            row.push((t, score(&cut.tracks, &turns, n)));
        }
        table.push((n, row));

        if matrix {
            print_matrix(&scanned, &turns, &voices);
        }
        if diagnose {
            let cut = scanned.cut(echo_lib::diarize::DISTANCE_THRESHOLD, None);
            print_turn_map(&cut.tracks, &turns, &voices);
        }
        println!();
    }

    // --- what the app actually does ---------------------------------------
    //
    // The count comes out of the shape of the merge tree
    // (`diarize::cluster::CountChoice`). It has to get every fixture right
    // without being told anything, and it has to have room to spare while doing
    // it — a criterion that only just wins is the failure this replaced.
    println!("=== the automatic count, on fixtures whose answer is known ===");
    println!(
        "{:>7}{:>8}{:>9}{:>10}{:>12}{:>12}{:>8}{:>7}{:>8}   how the cut was reached",
        "true", "found", "purity", "coverage", "silhouette", "runner-up", "margin", "cut", "folded"
    );
    let mut all_right = true;
    for (n, s, choice) in &automatic {
        let runner = choice.as_ref().and_then(|c| c.runner_up);
        let silhouette = choice.as_ref().map_or(0.0, |c| c.silhouette);
        let chosen = choice.as_ref().and_then(|c| c.chosen());
        println!(
            "{n:>7}{:>8}{:>9.3}{:>10.3}{:>12.3}{:>12}{:>8}{:>7}{:>8}   {}",
            s.found,
            s.purity,
            s.coverage,
            silhouette,
            runner.map_or_else(|| "-".to_string(), |(k, v)| format!("{k} at {v:.3}")),
            runner.map_or_else(|| "-".to_string(), |(_, v)| format!("{:.3}", silhouette - v)),
            chosen.map_or_else(|| "-".to_string(), |c| c.cut_clusters.to_string()),
            chosen.map_or_else(|| "-".to_string(), |c| c.folded.to_string()),
            match chosen.map(|c| c.refusal) {
                Some(None) => "the prior allowed it".to_string(),
                Some(Some(why)) => format!("only the fallback left it: {why:?}"),
                None => "no decision recorded".to_string(),
            }
        );
        all_right &= s.found == *n && s.purity > 0.95;
    }
    if let Some((_, _, Some(c))) = automatic.first() {
        println!();
        println!(
            "the largest relative gap on the {}-voice fixture: {}",
            automatic[0].0,
            c.gap_answer.as_ref().map_or_else(
                || "-".to_string(),
                |g| format!(
                    "{:.3} at a cut into {} clusters, which folds to {} people",
                    g.gap, g.cut_clusters, g.count
                )
            )
        );
    }
    println!();
    println!(
        "{}",
        if all_right {
            "every fixture right, with no threshold consulted"
        } else {
            "the automatic count got a fixture wrong — that is a release blocker"
        }
    );
    println!();

    // --- the sweep, as one table ------------------------------------------
    println!("=== how many people each threshold finds ===");
    print!("{:>9}", "threshold");
    for (n, _) in &table {
        print!("{:>12}", format!("{n} people"));
    }
    println!("{:>10}", "verdict");
    for (i, &t) in thresholds.iter().enumerate() {
        let shipped = (t - echo_lib::diarize::DISTANCE_THRESHOLD).abs() < 1e-4;
        print!("{:>9.3}", t);
        let mut all_right = true;
        for (n, row) in &table {
            let s = row[i].1;
            all_right &= s.found == *n && s.purity > 0.95;
            print!("{:>12}", format!("{}  {:.2}", s.found, s.purity));
        }
        println!(
            "{:>10}{}",
            if all_right { "all right" } else { "" },
            if shipped { "   <- shipped" } else { "" }
        );
    }

    println!();
    println!("=== the plateau: every threshold that gets every fixture exactly right ===");
    let right: Vec<f32> = thresholds
        .iter()
        .enumerate()
        .filter(|(i, _)| {
            table
                .iter()
                .all(|(n, row)| row[*i].1.found == *n && row[*i].1.purity > 0.95)
        })
        .map(|(_, t)| *t)
        .collect();
    if table.is_empty() {
        println!("no fixtures ran, so there is nothing to conclude");
        return Ok(());
    }
    match (right.first(), right.last()) {
        (Some(low), Some(high)) => {
            let shipped = echo_lib::diarize::DISTANCE_THRESHOLD;
            println!("{low:.3} … {high:.3}   (shipped {shipped:.4})");
            println!(
                "margin below {:.4}, margin above {:.4}",
                shipped - low,
                high - shipped
            );
            println!("middle of the plateau {:.4}", (low + high) / 2.0);
        }
        _ => println!("nothing in the sweep gets every fixture right"),
    }

    println!();
    println!("=== at the shipped threshold ===");
    let at = thresholds
        .iter()
        .position(|t| (t - echo_lib::diarize::DISTANCE_THRESHOLD).abs() < 1e-4)
        .expect("the shipped threshold is in the sweep");
    println!(
        "{:>10}{:>10}{:>10}{:>10}",
        "true", "found", "purity", "coverage"
    );
    for (n, row) in &table {
        let s = row[at].1;
        println!(
            "{n:>10}{:>10}{:>10.3}{:>10.3}",
            s.found, s.purity, s.coverage
        );
    }

    if let Some(dir) = &keep {
        println!();
        println!("fixtures kept in {}", dir.display());
    }
    Ok(())
}
