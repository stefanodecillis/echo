//! Bleed probe: run the offline bleed detector over a real meeting's two
//! channels and print, for every stretch of microphone speech, how much of it
//! the computer's own audio can account for.
//!
//! ```text
//! cargo run --release --example bleed_probe -- <meeting-id> [flags]
//!
//!   --audio DIR   read the chunk files from DIR instead of from wherever the
//!                 journal says they are (a copied-out meeting, a scratch disk)
//!   --minutes M   only the first M minutes, for a quick loop
//!   --warm        let the delay estimate narrow the search as the meeting goes
//!                 on, the way the live path will. Off by default: a cold
//!                 search on every stretch is the honest measurement.
//!   --quiet       summary only, no per-stretch rows
//! ```
//!
//! ## Why this exists before any of the detection ships
//!
//! Every constant in [`echo_lib::audio::bleed`] is an argument, and every one of
//! those arguments is a prediction about audio nobody has correlated yet. This
//! is the instrument that turns them into measurements, and it runs against
//! meetings already on disk, with no change to any live code path.
//!
//! **The summary's whole point is the split.** The correlation histogram is
//! printed twice: once for the mic stretches that had a system-channel segment
//! overlapping them in time, once for the ones that did not. If envelope
//! correlation separates a copy from a coincidence, those two histograms sit in
//! different places, and where they stop overlapping is where
//! [`echo_lib::audio::bleed::BLEED_CORRELATION`] belongs. If they sit on top of
//! each other, the approach does not work on this machine's audio and nothing
//! should be suppressed on the strength of it.
//!
//! The split is a *proxy*, not ground truth: on the test recording of
//! 2026-08-25 every one of the twelve system segments overlapped a mic segment,
//! which is the signature of bleed but not proof of it for any single pair. Read
//! it as "had the opportunity to be a copy", which is exactly what makes it the
//! right thing to split on.
//!
//! ## What it showed
//!
//! Run against meeting `eaffa793` (2026-08-25, thirty-five minutes on laptop
//! speakers, 194 stretches of microphone speech), the split is not subtle:
//!
//! ```text
//!   overlapped a system segment   116 stretches, mean r 0.765
//!                                 88 of them at or above 0.70
//!                                 80 of them at or above 0.80
//!   none did                       78 stretches, mean r 0.254
//!                                 3 of them above 0.70
//!   the delay                     +150 ms on 89 of the 91 that agreed
//! ```
//!
//! Two populations, a clean gap between them, and a delay that is the same
//! number all meeting long. That is what locked [`echo_lib::audio::bleed`]'s
//! constants: 0.70 sits in the gap, and the +150 ms is what says the search
//! window is the right shape. On `61fda270` (five minutes) every stretch long
//! enough to judge correlated between 0.945 and 0.981 at +170 ms; on
//! `4a98c1e1` there is no system channel at all and the probe says so rather
//! than measuring nothing.
//!
//! Nothing in the app's own storage is written or even opened for writing: the
//! database is copied to a temporary file first and the audio is read
//! read-only.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use echo_lib::audio::bleed::{self, BleedEvidence, LagEstimate, LagSearch};
use echo_lib::audio::vad::{Listening, OfflineDetector};
use echo_lib::audio::{read_window, ChunkRef};
use echo_lib::db::{self, repo};
use echo_lib::types::{AssetKind, Channel, Segment, TranscriptQuery};

/// How much audio to hand the detector at a time. The chunk size on disk, so
/// progress lines count the same things the recording does.
const PAGE_MS: i64 = 30_000;

fn app_support() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME");
    Path::new(&home).join("Library/Application Support/Echo")
}

fn clock(ms: i64) -> String {
    format!("{:>3}:{:02}", ms / 60_000, (ms % 60_000) / 1_000)
}

/// One mic stretch, judged.
struct Row {
    t_start_ms: i64,
    duration_ms: i64,
    voiced_ms: i64,
    evidence: BleedEvidence,
    warm: bool,
    /// A system-channel segment overlapped this stretch in time.
    overlapped: bool,
    verdict: &'static str,
}

/// The verdict, and — when it is no — the first condition that said no.
///
/// Naming the condition is the difference between "this was not suppressed" and
/// "this was not suppressed *because* the far side was barely playing". The
/// order matches [`bleed::is_bleed`].
fn verdict_of(ev: &BleedEvidence, voiced_ms: i64, warm: bool) -> &'static str {
    let span_floor = if warm {
        bleed::MIN_SPAN_WARM_MS
    } else {
        bleed::MIN_SPAN_COLD_MS
    };
    if ev.span_ms < span_floor {
        return "no: too short to judge";
    }
    if voiced_ms < bleed::MIN_MIC_VOICE_MS {
        return "no: too little voice";
    }
    if ev.correlation < bleed::BLEED_CORRELATION {
        return "no: shapes disagree";
    }
    if ev.system_voice_ms < (voiced_ms * 6 / 10).max(bleed::MIN_SYSTEM_VOICE_MS) {
        return "no: far side barely playing";
    }
    if ev.unexplained_ms >= bleed::OWN_VOICE_MS {
        return "no: own voice in it";
    }
    "BLEED"
}

/// Correlations in twentieths, as a printable histogram.
fn histogram(values: &[f32]) -> BTreeMap<i32, usize> {
    let mut bins = BTreeMap::new();
    for v in values {
        let bin = (v / 0.05).floor() as i32;
        *bins.entry(bin.clamp(-4, 19)).or_insert(0) += 1;
    }
    bins
}

fn bar(n: usize, of: usize) -> String {
    if of == 0 {
        return String::new();
    }
    let width = (n * 40).div_ceil(of.max(1)).min(40);
    "#".repeat(width)
}

/// The two histograms side by side. This is the measurement the whole probe is
/// for.
fn print_split(with: &[f32], without: &[f32]) {
    let (a, b) = (histogram(with), histogram(without));
    let peak = a
        .values()
        .chain(b.values())
        .copied()
        .max()
        .unwrap_or(0)
        .max(1);
    println!(
        "{:>12}  {:>5} {:<20}  {:>5} {:<20}",
        "correlation", "over", "(a system segment overlapped)", "alone", "(none did)"
    );
    for bin in -4..20 {
        let (n_with, n_without) = (
            a.get(&bin).copied().unwrap_or(0),
            b.get(&bin).copied().unwrap_or(0),
        );
        if n_with == 0 && n_without == 0 {
            continue;
        }
        let edge = bin as f32 * 0.05;
        println!(
            "{edge:>12.2}  {n_with:>5} {:<20}  {n_without:>5} {:<20}",
            bar(n_with, peak),
            bar(n_without, peak)
        );
    }
}

fn print_lags(rows: &[Row]) {
    let mut bins: BTreeMap<i64, usize> = BTreeMap::new();
    let mut hits = 0usize;
    for row in rows {
        // Only the delays read off a stretch that actually looked like a copy
        // mean anything; the rest are the best of a hundred coin flips.
        if row.evidence.correlation < bleed::BLEED_CORRELATION {
            continue;
        }
        hits += 1;
        *bins
            .entry(row.evidence.lag_ms.div_euclid(50) * 50)
            .or_insert(0) += 1;
    }
    println!();
    println!("=== the delay, over the {hits} stretches whose shapes agreed ===");
    if hits == 0 {
        println!("  nothing correlated above {:.2}", bleed::BLEED_CORRELATION);
        return;
    }
    let peak = bins.values().copied().max().unwrap_or(1);
    for (edge, n) in &bins {
        println!("{edge:>8} ms  {n:>5} {}", bar(*n, peak));
    }
    println!(
        "  the search ran from {} ms to {} ms; a value against either end means \
         the true delay may be outside it",
        bleed::LAG_MIN_MS,
        bleed::LAG_MAX_MS
    );
}

/// Read-only by construction: the app's database is copied before it is opened,
/// so a probe can be run while Echo itself is running and nothing it does can
/// touch a real meeting.
async fn open_copy_of_the_database(into: &Path) -> Result<db::Db, Box<dyn std::error::Error>> {
    let live = app_support().join("echo.db");
    if !live.is_file() {
        return Err(format!("no database at {}", live.display()).into());
    }
    let copy = into.join("echo.db");
    for suffix in ["", "-wal", "-shm"] {
        let from = PathBuf::from(format!("{}{suffix}", live.display()));
        if from.is_file() {
            std::fs::copy(&from, format!("{}{suffix}", copy.display()))?;
        }
    }
    Ok(db::connect(&copy).await?)
}

/// Chunks as the journal placed them, optionally re-pointed at another
/// directory by file name.
fn chunk_refs(
    chunks: &[echo_lib::types::AudioChunk],
    channel: Channel,
    audio_dir: Option<&Path>,
    until_ms: i64,
) -> Vec<ChunkRef> {
    chunks
        .iter()
        .filter(|c| c.committed && c.channel == channel && c.t_start_ms < until_ms)
        .map(|c| {
            let mut r = ChunkRef::from_journal(c);
            if let Some(dir) = audio_dir {
                if let Some(name) = r.path.file_name() {
                    r.path = dir.join(name);
                }
            }
            r
        })
        .collect()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut positional: Vec<String> = Vec::new();
    let mut audio_dir: Option<PathBuf> = None;
    let mut minutes: Option<i64> = None;
    let mut warm_allowed = false;
    let mut quiet = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--audio" => audio_dir = args.next().map(PathBuf::from),
            "--minutes" => minutes = args.next().and_then(|m| m.parse().ok()),
            "--warm" => warm_allowed = true,
            "--quiet" => quiet = true,
            other => positional.push(other.to_string()),
        }
    }
    let meeting = positional
        .first()
        .cloned()
        .ok_or("usage: bleed_probe <meeting-id> [--audio DIR] [--minutes M] [--warm] [--quiet]")?;
    let until_ms = minutes.map(|m| m * 60_000).unwrap_or(i64::MAX);

    let tmp = tempfile::tempdir()?;
    let db = open_copy_of_the_database(tmp.path()).await?;

    let chunks = repo::list_chunks(&db, &meeting, None).await?;
    let mic = chunk_refs(&chunks, Channel::Mic, audio_dir.as_deref(), until_ms);
    let system = chunk_refs(&chunks, Channel::System, audio_dir.as_deref(), until_ms);
    if mic.is_empty() {
        return Err(format!("meeting {meeting} has no committed microphone audio").into());
    }
    let mic_ms = mic.iter().map(|c| c.t_end_ms).max().unwrap_or(0);
    let system_ms = system.iter().map(|c| c.t_end_ms).max().unwrap_or(0);
    println!(
        "meeting {meeting}: {} mic chunk(s) to {}, {} system chunk(s) to {}",
        mic.len(),
        clock(mic_ms),
        system.len(),
        clock(system_ms)
    );
    if system.is_empty() {
        println!();
        println!(
            "There is no system channel in this meeting, so nothing the microphone \
             heard can be a copy of it. Bleed suppression would be disarmed here and \
             this probe has nothing to measure."
        );
        return Ok(());
    }

    // The transcript, for the split. A meeting that has not been transcribed
    // still measures — the split is simply empty on one side, and the probe says
    // so rather than pretending.
    let segments: Vec<Segment> = repo::get_segments(
        &db,
        &TranscriptQuery {
            meeting_id: meeting.clone(),
            limit: Some(200_000),
            ..Default::default()
        },
    )
    .await?;
    let system_segments: Vec<(i64, i64)> = segments
        .iter()
        .filter(|s| s.channel == Channel::System && s.is_final)
        .map(|s| (s.t_start_ms, s.t_end_ms))
        .collect();
    println!(
        "transcript: {} segment(s), {} of them on the system channel",
        segments.len(),
        system_segments.len()
    );
    if system_segments.is_empty() {
        println!(
            "note: no system-channel text, so every stretch below lands in the \
             \"none did\" column and the split says nothing. Run the catch-up pass \
             first if this meeting has never been transcribed."
        );
    }

    let detector_path = echo_lib::asr::models::installed_path(&db, AssetKind::SpeechDetector)
        .await
        .unwrap_or(None);
    match &detector_path {
        Some(p) => println!("detector: {}", p.display()),
        None => println!(
            "detector: none installed — falling back to the loudness gate, exactly \
             as a first-launch recording does"
        ),
    }
    println!(
        "search: {} ms to {} ms, {}",
        bleed::LAG_MIN_MS,
        bleed::LAG_MAX_MS,
        if warm_allowed {
            "narrowing once the meeting agrees on a delay"
        } else {
            "cold on every stretch"
        }
    );
    println!();

    // --- find the speech in the microphone channel ------------------------
    // The settings the post-meeting pass uses, because that is the pass whose
    // numbers this probe exists to predict.
    let mut detection = OfflineDetector::open(detector_path, Channel::Mic, Listening::FromDisk)?;
    let mut utterances = Vec::new();
    let pages = (mic_ms + PAGE_MS - 1) / PAGE_MS;
    for page in 0..pages {
        let from = page * PAGE_MS;
        let to = (from + PAGE_MS).min(mic_ms);
        let audio = read_window(&mic, from, to).await?;
        utterances.extend(detection.push(audio, from).await?);
        // A long meeting is a long wait; a probe that prints nothing looks hung.
        // Progress goes to stderr so piping the run to a file keeps the
        // measurement clean.
        eprint!(
            "\r  reading microphone {}/{} ({}), {} stretch(es) so far   ",
            page + 1,
            pages,
            clock(to),
            utterances.len()
        );
        std::io::stderr().flush().ok();
    }
    utterances.extend(detection.finish().await?);
    eprintln!();
    println!(
        "read {} page(s) of microphone audio; {} speech stretch(es)",
        pages,
        utterances.len()
    );

    // --- judge each one against the system channel ------------------------
    let mut lag = LagEstimate::default();
    let mut rows: Vec<Row> = Vec::new();
    for (i, u) in utterances.iter().enumerate() {
        // Enough of the far side either side of the stretch that every candidate
        // delay sees all of it: none of them is judged on less evidence than the
        // others.
        let from = (u.t_start_ms - bleed::LAG_MAX_MS).max(0);
        let to = u.t_end_ms - bleed::LAG_MIN_MS;
        let far_side = read_window(&system, from, to).await?;
        let (search, warm) = if warm_allowed {
            (lag.search(u.t_start_ms), lag.is_warm(u.t_start_ms))
        } else {
            (LagSearch::cold(), false)
        };
        let evidence = bleed::examine(&u.samples, &far_side, u.t_start_ms - from, search);
        let voiced_ms = u.measured_voice_ms();
        let verdict = verdict_of(&evidence, voiced_ms, warm);
        if verdict == "BLEED" {
            lag.record(u.t_start_ms, evidence.lag_ms);
        }
        rows.push(Row {
            t_start_ms: u.t_start_ms,
            duration_ms: u.duration_ms(),
            voiced_ms,
            evidence,
            warm,
            overlapped: system_segments
                .iter()
                .any(|(s, e)| *s < u.t_end_ms && *e > u.t_start_ms),
            verdict,
        });
        if i % 25 == 0 {
            eprint!("\r  judging {}/{}   ", i + 1, utterances.len());
            std::io::stderr().flush().ok();
        }
    }
    eprintln!();
    println!("judged {} stretch(es)", rows.len());

    // --- one line each ----------------------------------------------------
    if !quiet {
        println!();
        println!(
            "{:>8} {:>8} {:>8} {:>8} {:>6} {:>7} {:>7} {:>7} {:>6} {:>4}  verdict",
            "at",
            "t_start",
            "dur_ms",
            "voiced",
            "r",
            "lag_ms",
            "span_ms",
            "far_ms",
            "own_ms",
            "over"
        );
        for row in &rows {
            println!(
                "{:>8} {:>8} {:>8} {:>8} {:>6.3} {:>7} {:>7} {:>7} {:>6} {:>4}  {}{}",
                clock(row.t_start_ms),
                row.t_start_ms,
                row.duration_ms,
                row.voiced_ms,
                row.evidence.correlation,
                row.evidence.lag_ms,
                row.evidence.span_ms,
                row.evidence.system_voice_ms,
                row.evidence.unexplained_ms,
                if row.overlapped { "yes" } else { "-" },
                row.verdict,
                if row.warm { " (warm)" } else { "" },
            );
        }
    }

    // --- the measurement --------------------------------------------------
    let (with, without): (Vec<&Row>, Vec<&Row>) = rows.iter().partition(|r| r.overlapped);
    let correlations =
        |rows: &[&Row]| -> Vec<f32> { rows.iter().map(|r| r.evidence.correlation).collect() };
    let mean = |v: &[f32]| -> f32 {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f32>() / v.len() as f32
        }
    };
    let with_r = correlations(&with);
    let without_r = correlations(&without);

    println!();
    println!("=== how well the two channels' loudness agreed ===");
    println!(
        "  {} stretch(es) overlapped a system segment (mean r {:.3}), \
         {} did not (mean r {:.3})",
        with.len(),
        mean(&with_r),
        without.len(),
        mean(&without_r)
    );
    println!(
        "  if the predicate separates, these two columns sit in different places; \
         where they stop overlapping is where the bar belongs (it is {:.2} today)",
        bleed::BLEED_CORRELATION
    );
    println!();
    print_split(&with_r, &without_r);

    print_lags(&rows);

    println!();
    println!("=== what would have been suppressed ===");
    let mut by_verdict: BTreeMap<&str, usize> = BTreeMap::new();
    for row in &rows {
        *by_verdict.entry(row.verdict).or_insert(0) += 1;
    }
    for (verdict, n) in &by_verdict {
        println!("{n:>6}  {verdict}");
    }
    let suppressed: Vec<&Row> = rows.iter().filter(|r| r.verdict == "BLEED").collect();
    let suppressed_ms: i64 = suppressed.iter().map(|r| r.duration_ms).sum();
    let total_ms: i64 = rows.iter().map(|r| r.duration_ms).sum();
    println!(
        "{:>6}  of {} stretch(es), {:.1} s of {:.1} s of microphone speech",
        suppressed.len(),
        rows.len(),
        suppressed_ms as f64 / 1_000.0,
        total_ms as f64 / 1_000.0
    );
    // The one number that says the search window is wide enough: a machine whose
    // true delay is past the end of it suppresses nothing and says nothing.
    let opportunities = rows
        .iter()
        .filter(|r| r.evidence.system_voice_ms >= bleed::MIN_SYSTEM_VOICE_MS)
        .count();
    if suppressed.is_empty() && opportunities > 20 {
        println!();
        println!(
            "Nothing was suppressed although the far side was playing under {opportunities} \
             stretches. On a recording made on speakers that means the microphone's copy \
             was looked for and never found — a delay outside {} ms to {} ms would do \
             that, and so would headphones.",
            bleed::LAG_MIN_MS,
            bleed::LAG_MAX_MS
        );
    }

    Ok(())
}
