# Echo — Design Document (v2, post codex review)

Local-first meeting transcription, summaries, and action items. macOS (Apple Silicon) + Linux.
Reviewed by codex/gpt-5.6-sol; 40 findings incorporated (see `docs/REVIEW-NOTES.md`).

## 0. Mantras

1. **Use nothing until needed, release it when idle.** No model in RAM, no audio stream open, no ONNX session, no DB-heavy work unless a recording or an explicit user action requires it. Whisper loads at recording start (or on explicit pre-warm) and unloads after a configurable idle period. Idle app = a 5s detection poll and nothing else (<0.5% CPU target).
2. **Zero jargon in the UI.** Never mention Whisper, models, ONNX, VAD, diarization, connectors, tokens. Onboarding: "Downloading what Echo needs to understand speech (1.6 GB, one time)". Recording: "Listening…". Processing: "Writing your recap…". Technical detail (model names, engine, GPU in use) lives only behind Settings → Advanced. Errors say what the user can do, not what failed internally.
3. **Raw audio on disk is the source of truth.** Everything downstream (transcription, diarization, summaries) can be recomputed from it; nothing downstream may block or lose capture.

## 1. Goals

- Capture meetings (microphone + system audio), transcribe locally with Whisper, automatic language detection.
- On-device speaker attribution: live = channel-based ("You" = mic, remote = system channel with provisional labels); after recording ends, an offline diarization refinement pass distinguishes and stabilizes individual remote speakers. Rename + merge speakers (non-destructive, alias-based).
- AI summaries + action items via a pluggable connector trait; v1 ships **Ollama** (local) and **Gemini** (AI Studio API key). Summary output language is a user setting (meeting language / English / fixed custom).
- Automatic meeting detection with an OS notification (clicking it opens Echo with a prominent Start button; tray menu also has Start — desktop notification action buttons are not supported by Tauri's plugin).
- History: browse, search (FTS), re-open past meetings; export Markdown/PDF/DOCX.
- Onboarding wizard: permissions → model download → summaries setup. Non-technical copy throughout.
- Background/tray app; window/launcher always remains a first-class route (Linux tray support is not universal).
- Matches Meetily Pro: diarization, auto-detect, summary templates (6 built-in + custom), exports, transcript search, quality presets.

### Non-goals (v1)
Windows. Calendar integration. Chat-with-meetings (schema leaves room). Team deployment. Video. PulseAudio-only systems (PipeWire required on Linux; PulseAudio fallback post-v1). User-defined arbitrary connectors (the trait is extensible; v1 compiles Ollama + Gemini).

## 2. Stack

| Layer | Choice | Notes |
|---|---|---|
| Shell | Tauri 2.x | Tray, single-instance, notifications, updater plugins |
| Frontend | React 18 + TS + Vite + Tailwind | Sana-style minimal UI |
| ASR | whisper-rs (whisper.cpp) in-process | **Per-OS builds with backends compiled in**: macOS binary = `metal`+`coreml` features; Linux binary = `vulkan` feature. Runtime = GPU *init* attempt with CPU fallback (`n_threads` clamped, conservative default). CoreML needs a matching `*-encoder.mlmodelc` per model — catalogued and downloaded alongside the GGML file; accelerates encoder only. |
| VAD | Silero VAD via `ort` | CPU (tiny). On Linux, all ONNX work is CPU-bound (no Vulkan EP in ort) — accepted, stated in Advanced settings. |
| Diarization | ONNX segmentation (pyannote community ONNX export, license-audited) + speaker embeddings (WeSpeaker/3D-Speaker) via `ort`; sliding-window inference, overlap-aware decoding, clustering with calibrated threshold; **offline pass after recording is canonical**, live labels provisional | Gated by a quality spike (see M0). If quality misses thresholds, v1 ships channel-attribution live + offline diarization only. |
| Mic capture | `cpal` | |
| System audio | macOS: ScreenCaptureKit audio-only via `cidre` (macOS 13+), content filter excludes Echo's own audio; Linux: PipeWire — discover current default sink's monitor ports, follow default-sink changes, reconnect on node removal/BT profile switch | Honest copy: system capture = "everything your computer plays" |
| DSP | `rubato` resample→16kHz mono per channel, `ringbuf` bounded queues, `ebur128` | No allocation/blocking/DB/ONNX inside audio callbacks |
| Storage | SQLite via `sqlx` + FTS5 (unicode61 + trigram tables; MATCH input escaped) | WAL mode, batched writes |
| Audio files | Per-channel chunked FLAC from t=0 (canonical) + mixed FLAC derived for playback | Chunk journal for crash recovery; configurable location |
| Secrets | `keyring` → Keychain / Secret Service | Off UI thread; distinct handling for absent/locked/cancelled; **never** plaintext fallback; recording works without stored credentials |
| HTTP | `reqwest` | Downloads: SHA-256 verify, length check, Range+ETag resume, temp file + fsync + atomic rename, disk-space preflight, stale-partial cleanup |

Distribution: macOS aarch64 `.dmg` (Developer ID signing + notarization + hardened runtime + usage-description plist entries; permission testing on the notarized build); Linux x86_64 `.AppImage` + `.deb` built on oldest supported base (declares WebKitGTK/AppIndicator deps). CI compiles both + runs unit/fixture tests; audio/permission/tray behavior verified on a manual hardware matrix (headless CI cannot).

## 3. Architecture

```
React UI ⇄ IPC ⇄ Rust core
detect ──► notify(click→open app) ──► session ──► pipeline

pipeline (per recording, monotonic meeting clock):
  mic(cpal) ─────► ts-stamped frames ─► resample ─► ch-A FLAC chunks (disk, canonical)
  system(SCK/PW) ► ts-stamped frames ─► resample ─► ch-B FLAC chunks (disk, canonical)
                          │ bounded ringbufs, drift detect/compensate
                          ▼
                 VAD ─► utterance queue (bounded; on overflow: drop live jobs,
                        audio stays authoritative, ASR catches up from disk)
                          ▼
                 whisper worker (1 model, serialized jobs, live partial events)
                          ▼
                 segments ─► SQLite (batched) + UI events (rate-capped)

on stop: jobs (persisted, resumable): finalize-ASR-catchup → diarize-offline
         → summarize (on demand or auto) — each cancellable, recording preempts all
```

### State model (orthogonal, persisted)
- **Capture**: Idle → Starting → Recording ⇄ Paused → Stopping → Stopped (+ Failed, Degraded[lost system audio → banner, keep mic], Recovering[found interrupted session at launch → offer resume/finalize])
- **Jobs table**: transcribe-catchup / diarize / summarize / export, each Queued → Running → Done/Failed/Cancelled with progress + error. Recording has absolute resource priority; background jobs pause.
- Detection is its own small watcher (Idle/Detected/Snoozed), not part of capture state.
- Every IPC transition idempotent (double-click safe).

### Crash recovery
Meeting row created **before** capture starts. Per-channel FLAC chunks journaled as committed. On relaunch: interrupted meeting detected → transcription resumes from last committed audio offset; already-final segments untouched.

### Data model

```sql
meetings(id, title, started_at, ended_at, detected_app, language, status,
         audio_dir, mixed_path, duration_ms, deleted_at)
audio_chunks(id, meeting_id, channel, seq, path, t_start_ms, t_end_ms, committed)
segments(id, meeting_id, t_start_ms, t_end_ms, channel, speaker_id, text,
         language, avg_confidence, revision, is_final)
speakers(id, meeting_id, cluster_key, display_name, alias_of)  -- merge = alias, non-destructive
markers(id, meeting_id, t_ms, kind, note)                       -- live "flag action item"
summaries(id, meeting_id, template_id, template_snapshot, provider, model,
          language, transcript_revision, content_md, created_at)
action_items(id, meeting_id, summary_id, description, owner, due_hint, done, external_url)
templates(id, name, prompt_md, builtin)   -- general recap, standup, client call, retro, 1:1, interview
jobs(id, meeting_id, kind, status, progress, error, created_at, updated_at)
models(id, kind, name, url, sha256, bytes, license, revision, installed, path)  -- signed catalog incl. CoreML artifacts
settings(key, value)                       -- non-secret only
segments_fts(FTS5)
```

Provenance: segments/summaries store the model + revision that produced them. Deletion: per-meeting delete, "delete audio keep text", delete-all, storage-size report in Settings.

### Connectors

```rust
trait Connector {
  fn capabilities(&self) -> Caps;           // json_mode, streaming, context_window
  async fn list_models(&self) -> ...;
  async fn generate(&self, req) -> Stream;  // timeout, retry, cancel
}
```
Summarization = map-reduce over transcript chunks sized to the model's context; strict JSON schema for action items validated locally with one repair-retry; summary markdown sanitized before render (no raw HTML, no remote resources). Small-model quality mitigations: chunking, schema, "regenerate with different model" button. Gemini setup screen states plainly what leaves the machine and that Google's data handling depends on account/billing tier, with a link — an explicit choice, not a footnote. Ollama URL restricted to loopback by default; non-loopback allowed only with a "this leaves your machine" warning.

### Meeting detection
Poll every 5s when idle: (a) running meeting apps (zoom.us, Teams, Webex, Discord, Slack); (b) input-device-in-use (macOS CoreAudio `kAudioDevicePropertyDeviceIsRunningSomewhere`; Linux PipeWire active input streams). Debounced → notification (click opens Echo, big Start button) + tray badge. Auto-stop suggestion after signals clear >2 min. Snooze 1h.

## 4. UI (Sana-style; see mantra 2)

Light theme, white, generous whitespace, rounded-xl cards, 1px #eee borders, near-black text, one accent used sparingly, pill buttons, Inter/system font.

1. **Home** — status hero ("Waiting for your next meeting" / live card with elapsed + live caption), search bar, recent meetings (title, date, duration, language chip, snippet).
2. **Live** — transcript stream with speaker chips ("You" / "Speaker 1"…), visible recording indicator, pause/stop, "Flag action item" marker button, degraded-capture banner when applicable.
3. **Meeting detail** — tabs: Recap (rendered summary + action-items checklist with owner chips, copy/export, regenerate-with-model picker, template picker), Transcript (virtualized, speaker colors, in-page search, click-to-play), Info (what was captured, storage used, delete options).
4. **History/Search** — FTS across meetings.
5. **Settings** — General (launch at login, detection on/off, storage location, summary language), Speech (installed "accuracy levels" = quality presets; sizes shown), Summaries (On this computer [Ollama] / Google Gemini [key, masked, keychain] with plain-language privacy note), Templates (built-ins + custom CRUD), Data (storage report, delete-all), Advanced (engine, GPU in use, model identifiers, diagnostics export).
6. **Onboarding** — Welcome → Permissions (mic; macOS screen-recording with "why" copy, handles restart-required and denied states, "Open System Settings" button) → Download (~1.6 GB one-time, resumable, can pick "smaller & faster" preset) → Summaries (auto-detect Ollama / Gemini key / skip).

Tray: idle/detected/recording states; menu: Start/Stop, Open Echo, Pause detection 1h, Quit. Close-to-tray, but app is always restorable from launcher/window (Linux tray caveat). Single instance.

Local diagnostics (no telemetry): bounded, redacted rotating log — capture backend, device changes, queue overflows, timings, GPU fallback reason; never transcript text or keys; exportable from Advanced.

## 5. Milestones

**M0 — Feasibility spikes (throwaway code, go/no-go):**
S1 macOS SCK audio-only capture (filter excludes own app, CMSampleBuffer→PCM, timestamps, hidden-window behavior). S2 Linux PipeWire monitor capture (follow default sink). S3 whisper-rs Metal/CoreML + Vulkan builds, RTF benchmarks per preset. S4 diarization quality benchmark vs fixture audio (acceptance: stable speaker count, usable DER, RTF<0.3 CPU) → decides live vs offline-only. S5 signed/notarized .dmg + AppImage smoke.

**M1 — Scaffold**: Tauri2+React+Tailwind, migrations, settings, keyring, jobs table, IPC contracts (stub commands/events), CI skeleton, tray + close-to-tray + single-instance, minimal window shell.
**M2 — Capture**: mic + system per OS, meeting clock, per-channel chunked FLAC + journal, VAD, capture state machine, degraded/recovery paths. Thin functional onboarding (permissions + download UI wired to model manager).
**M3 — ASR**: model manager (catalog/SHA-256/resume/CoreML artifacts), whisper worker, live partials + finals, per-segment language + confidence, catch-up-from-disk, crash-safe persistence.
**M4 — Speakers**: offline diarization pipeline + alias rename/merge; live channel labels; (live clustering only if S4 passed).
**M5 — Summaries**: connector trait, Ollama + Gemini, templates, action-item JSON, summary language, sanitized render.
**M6 — Detection**: process + device signals, notifications, tray states, auto-stop suggestion, snooze.
**M7 — Complete UX**: full onboarding polish, history/FTS, exports (MD; DOCX via docx-rs; PDF via dedicated hidden webview print-to-file pipeline — spiked, not assumed), quality presets, deletion/storage UI, diagnostics, Sana-polish pass.

Every milestone compiles on both OSes; capability checks degrade gracefully (no system-audio permission → mic-only + banner).
