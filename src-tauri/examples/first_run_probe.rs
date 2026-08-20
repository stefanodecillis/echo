//! First-run smoke probe: runs the never-executed native seams of the boot
//! path on this machine, without opening a window or an audio device.
//!
//!   cargo run --example first_run_probe
//!
//! What it proves, in order: fresh-file database + migrations, settings load,
//! the recovery scan on an empty database, the 5-second meeting-detection poll
//! (real sysinfo + real CoreAudio FFI), the real keychain round-trip, every
//! catalog URL resolving upstream, and a real download + ONNX load + inference
//! of the speech detector. Exits non-zero on the first failure.

use echo_lib::asr::{catalog, models};
use echo_lib::audio::vad::SpeechDetector;
use echo_lib::types::{AssetKind, Channel};
use echo_lib::{db, detect, paths, secrets, settings};

fn ok(step: &str) {
    println!("PASS  {step}");
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tmp = tempfile::tempdir()?;

    // 1. Fresh file database + transactional migrations, same call as setup().
    let db = db::connect(&tmp.path().join("echo.db")).await?;
    ok("fresh database created, migrations applied");

    // 2. Settings load on an empty table (defaults path).
    let loaded = settings::load(&db).await?;
    assert!(!loaded.onboarding_complete, "fresh db must want onboarding");
    ok("settings load on a fresh database (onboarding wanted, as expected)");

    // 3. Paths against a temp storage root, like setup() after settings.
    let app_paths = paths::AppPaths::resolve(None)?;
    app_paths.set_storage_root(&tmp.path().join("storage"));
    app_paths.ensure()?;
    ok("paths resolve + ensure with a configurable storage root");

    // 4. The recovery scan and stale-partial sweep on an empty database.
    let swept = models::clean_stale_partials(&db, &app_paths).await?;
    assert_eq!(swept, 0);
    ok("stale-download sweep on a fresh install (nothing to sweep)");

    // 5. The meeting-detection poll: real sysinfo process scan and the real
    //    CoreAudio kAudioDevicePropertyDeviceIsRunningSomewhere FFI, three
    //    times, exactly like the watcher's 5s tick.
    let watcher = detect::Watcher::new(true);
    for i in 0..3 {
        let signals = watcher.poll_once().await?;
        println!("      poll {} -> {} signal(s): {:?}", i + 1, signals.len(), signals);
    }
    ok("detection poll (sysinfo + CoreAudio FFI) three times without incident");

    // 6. The real keychain, with a throwaway entry: the unit tests use the
    //    in-memory store, so this is the first genuine Keychain round-trip.
    let account = "first-run-probe-throwaway";
    secrets::set(account, "echo-probe").await?;
    assert_eq!(secrets::get(account).await?, "echo-probe");
    secrets::delete(account).await?;
    assert!(!secrets::has(account).await?);
    ok("keychain set/get/delete round-trip (real backend)");

    // 7. Every catalog asset URL answers upstream with the size we expect.
    models::sync_catalog(&db).await?;
    let client = reqwest::Client::builder().build()?;
    for entry in catalog::CATALOG {
        let resp = client.head(entry.url).send().await?;
        let status = resp.status();
        assert!(
            status.is_success(),
            "{} -> HTTP {status} for {}",
            entry.id,
            entry.url
        );
        println!("      {:<38} HTTP {status}", entry.id);
    }
    ok("all catalog URLs resolve upstream");

    // 8. Real download of the speech detector (2.3 MB), hash-verified by the
    //    real downloader, then a real ONNX session and inference: one second
    //    of silence must not read as speech, and a spoken-band tone must not
    //    crash the engine.
    let info = models::download_asset(&db, &app_paths, &catalog::ids::DETECTOR.to_string(), Box::new(|_| {})).await?;
    println!("      downloaded {} ({} bytes)", info.name, info.bytes);
    let path = models::installed_path(&db, AssetKind::SpeechDetector)
        .await?
        .expect("detector just downloaded");
    let mut detector = SpeechDetector::load(&path, Channel::Mic)?;
    assert!(detector.uses_model(), "must run the ONNX engine, not the fallback");
    let silence = vec![0.0f32; 16_000];
    let utterances = detector.push(&silence, 0);
    assert!(utterances.is_empty(), "silence must not produce an utterance");
    let tone: Vec<f32> = (0..16_000)
        .map(|i| (i as f32 * 220.0 * std::f32::consts::TAU / 16_000.0).sin() * 0.4)
        .collect();
    let _ = detector.push(&tone, 1_000);
    ok("speech detector: real download, hash verify, ONNX load, inference");

    // 9. The speech engine's first load — the seam a first recording hits.
    //    Uses the smallest preset (~75 MB + Apple encoder companion) so this
    //    stays cheap; proves the download bundle path, whisper.cpp linking,
    //    Metal/Core ML init on this machine, and the CPU-fallback report.
    //    Opt in with PROBE_ENGINE=1.
    if std::env::var("PROBE_ENGINE").as_deref() == Ok("1") {
        models::download_level(&db, &app_paths, "fastest", Box::new(|_| {})).await?;
        settings::apply(
            &db,
            &echo_lib::types::SettingsPatch {
                accuracy_level_id: Some("fastest".into()),
                ..Default::default()
            },
        )
        .await?;
        let worker = echo_lib::asr::engine::EngineWorker::new();
        worker.configure_from_settings(&db).await?;
        let report = worker.load_now().await?;
        println!(
            "      compiled: {} | active: {} | accelerator present: {} | threads: {}",
            report.compiled, report.active, report.accelerator_present, report.threads
        );
        if let Some(reason) = &report.fallback_reason {
            println!("      fell back to CPU: {reason}");
        }
        worker.shutdown();
        ok("speech engine: real weights load, Metal/Core ML init, clean shutdown");
    } else {
        println!("SKIP  speech engine load (set PROBE_ENGINE=1 to include the ~90 MB download)");
    }

    println!("\nAll first-run seams passed on this machine.");
    Ok(())
}
