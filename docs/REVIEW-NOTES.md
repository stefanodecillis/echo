# Codex review (gpt-5.6-sol) — condensed findings & resolutions

Full review run 2026-08-19 against DESIGN.md v1. All CRITICAL/IMPORTANT items folded into DESIGN.md v2. Key resolutions:

| # | Finding | Resolution in v2 |
|---|---|---|
| 1 | whisper-rs GPU backends are compile-time features | Per-OS binaries (mac: metal+coreml, linux: vulkan); runtime = init attempt + CPU fallback. CoreML encoder artifacts in model catalog |
| 2 | ort has no Vulkan EP → VAD/diarization CPU-bound on Linux | Accepted & stated; benchmarks in M0-S3/S4 |
| 3,5 | SCK & PipeWire capture nontrivial | M0 spikes S1/S2; own-audio exclusion; follow default sink |
| 4 | macOS permission UX (plist strings, restart-after-grant, revocation) | Onboarding states + Degraded capture state |
| 6 | System capture records ALL system audio | Honest copy: "everything your computer plays" |
| 7 | Tauri desktop notifications: no action buttons | Notification click opens app w/ Start CTA; tray Start |
| 8 | Linux tray not universal | Window/launcher always first-class |
| 9,31 | Diarization pipeline complexity understated | Offline pass canonical, live labels provisional, quality-gated (M0-S4) |
| 11 | Model provenance/licensing | models table: url, sha256, license, revision; signed catalog |
| 12 | keyring failure modes on Linux | Distinct absent/locked/cancelled handling, off UI thread, no plaintext fallback |
| 13,33 | Connector trait too weak; custom-connector claim | Caps (json_mode/streaming/context), timeout/retry/cancel; v1 = Ollama+Gemini only |
| 14 | Gemini privacy disclosure | Explicit choice screen, tier-dependent data handling stated |
| 15 | Download integrity | SHA-256, length, Range+ETag resume, fsync+atomic rename, disk preflight |
| 16 | Clock sync across capture sources | Monotonic meeting clock, per-callback timestamps, drift compensation |
| 17 | Backpressure undefined | Bounded queues, no heavy work in callbacks, audio-on-disk authoritative, ASR catch-up from disk |
| 18,22 | Crash recovery / mixed-only audio insufficient | Per-channel chunked FLAC canonical from t=0, chunk journal, meeting row pre-capture, resume from committed offset |
| 19 | State machine conflates concerns | Orthogonal capture state + persisted jobs table; idempotent IPC |
| 20 | VAD segments vs live partials conflict | Revision field on segments; padded/max-duration chunks (impl detail in asr module) |
| 21 | Language detection brittle | Detect from first sufficient speech w/ confidence, per-segment language, dominant meeting language |
| 23 | Segment schema can't express overlap/refinement | revision + per-segment language/confidence; alias-based speakers; overlap accepted limitation v1 |
| 24 | Provenance columns | Added to segments/summaries (model, revision, template_snapshot, transcript_revision) |
| 25 | Missing tables (markers, models, jobs) | Added |
| 26 | FTS multilingual | unicode61 + trigram, escaped MATCH |
| 27 | Resource arbitration | Recording preempts all jobs; clamped threads; lazy load/unload (mantra 1) |
| 28 | Spikes before scaffold | M0 added |
| 29 | Headless CI can't test audio/permissions | Manual hardware matrix + synthetic fixture tests |
| 30 | Onboarding/tray too late | Tray+lifecycle in M1, thin onboarding in M2 |
| 32 | Exports underestimated | MD first; DOCX via docx-rs; PDF via hidden-webview print pipeline, spiked |
| 34 | Long-meeting memory/disk | Streaming everything, bounded buffers, batched writes, disk-full handling; 8h test |
| 35,36 | Signing/notarization; Linux packaging baseline | Added to Distribution |
| 37 | Consent & deletion | Recording indicator, delete meeting/audio-only/all, storage report |
| 38 | Markdown/render hardening | Sanitized render, no raw HTML/remote resources; IPC validation in Rust |
| 39 | Updates/migrations | sqlx transactional migrations; updater plugin; model-catalog versioning |
| 40 | Local diagnostics | Redacted bounded logs, exportable, no telemetry |
