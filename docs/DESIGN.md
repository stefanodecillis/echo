# Echo — Design Document (v2, post codex review)

Local-first meeting transcription, summaries, and action items. macOS (Apple Silicon) + Linux.
Reviewed by codex/gpt-5.6-sol; 40 findings incorporated (see `docs/REVIEW-NOTES.md`).

## 0. Mantras

1. **Use nothing until needed, release it when idle.** No model in RAM, no audio stream open, no ONNX session, no DB-heavy work unless a recording or an explicit user action requires it. Idle app = a 5s detection poll and nothing else. **Amendment (2026-08-20, product decision): while Echo is LISTENING, transcript quality outranks resource thrift.** The engine loads the moment a meeting is detected or started and stays resident for the whole conversation and its post-meeting work, with accuracy-first decode settings; it unloads again once the meeting's jobs are done (idle grace applies). Capture never waits for the engine: audio is on disk from second zero, and when the engine comes up mid-meeting it first transcribes this meeting's backlog, then continues live. "Then" is exact, not approximate: the handover splits the meeting clock at one instant, everything before it is read back off the recording in order while the live pass declines it, and everything after it is the live pass's alone. No stretch of a meeting is ever transcribed by both, and the pass reading the backlog is finished with before the post-meeting one is queued. There is no model choice in onboarding or settings, and no per-lane or per-machine model routing: Echo downloads and uses **one** speech model for everything — live captions, live finals, catch-up and Listen again alike (currently full large-v3 + the Apple accelerator). **Amendment (2026-08-20, product decision): the app also keeps its own model state honest.** At launch, and whenever anything asks whether Echo can understand speech, it reconciles what is on disk against what it wants: missing files are queued for download, and weights it has stopped using are deleted — but never before their replacement is verified installed. A release that changes the model therefore upgrades itself, and a meeting started in the middle of that upgrade records and transcribes on the weights that are already there. One quiet notice when the switch happens ("Echo upgraded how it understands speech"); nothing else is said.
2. **Zero jargon in the UI.** Never mention Whisper, models, ONNX, VAD, diarization, connectors, tokens. Onboarding: "Downloading what Echo needs to understand speech (4.3 GB, one time)" — the size is read from the catalog at runtime, never typed into the copy, because it differs per platform and moves whenever the model does. Recording: "Listening…". Processing: "Writing your recap…". Technical detail (model names, engine, GPU in use) lives only behind Settings → Advanced. Errors say what the user can do, not what failed internally.
3. **Raw audio on disk is the source of truth.** Everything downstream (transcription, diarization, summaries) can be recomputed from it; nothing downstream may block or lose capture.
4. **Built for a non-technical user.** The reference persona is a freelancer or manager, not an engineer. Installation, onboarding, and daily use must never require a terminal, a config file, or prior knowledge. Concretely:
   - **Install**: signed/notarized .dmg with drag-to-Applications layout on macOS; .deb/.AppImage that declare every dependency on Linux. Double-click → it runs. No "install Ollama first" prerequisite: summaries setup offers Gemini or a clearly optional local path, and the app is fully usable for recording/transcripts before any summary provider is configured.
   - **First run**: onboarding does everything (permissions with plain "why" copy and Open-Settings buttons, one-time download with progress, summaries choice or skip). A user who clicks "next" through it ends up with a working app.
   - **Daily use**: the happy path is notification → click → Start → recap appears. Every error states what the user can do next in plain words; no state is a dead end; defaults are always safe so Settings never *needs* visiting.

## 1. Goals

- Capture meetings (microphone + system audio), transcribe locally with Whisper, automatic language detection.
- On-device speaker attribution: live = channel-based ("You" = mic, remote = system channel with provisional labels); after recording ends, an offline diarization refinement pass distinguishes and stabilizes individual remote speakers. Rename + merge speakers (non-destructive, alias-based). The pass identifies HOW MANY people were in the meeting on its own (shown as "detected"), and the person can correct that number afterwards — an override re-runs the voice separation constrained to exactly that many people, which segments voices better than any automatic threshold.
- **Known people (voice enrollment).** Echo can remember specific voices the person chooses to save ("favorite people"), and recognize them in later meetings. The mechanics are enrollment-based speaker identification over the voice-prints the separation pass already computes: a person = a name + a centroid + a capped, condition-diverse set of samples (each sample keeps its short audio clip AND its embedding, so profiles survive an embedder upgrade by re-embedding the clips; profiles are tagged with the embedder asset they were computed with, and matching silently skips when the tags disagree until a refresh runs). Matching is OPEN-SET with a margin rule: a meeting speaker links to a known person only when similarity clears a threshold AND beats the second-best candidate by a margin — "nobody I know" is always a valid answer. Confidence tiers: high → pre-filled suggestion in Name-your-speakers ("Looks like Marco — confirm?"); medium → a suggestion chip; low → nothing; never a silent assignment. Profiles improve ONLY from confirmations (accepting a suggestion, or manually linking), capped per meeting, keeping samples from different conditions. Enrolled voices are matched BEFORE the count question, so blind counting applies only to strangers. UI: a People section in Settings (name, last heard, sample count, Listen, Delete — delete destroys profile, samples and clips) with a plain sentence stating profiles are voice data that stays on this Mac and can always be deleted; a "suggested people" list of recurring unnamed voices across meetings (per-meeting speaker centroids are stored to make this possible without re-reading audio); Name-your-speakers rows always offer BOTH choose-a-known-person and type-a-new-name, plus "remember this voice" to enroll. Enrollment is per-person opt-in — never automatic for everyone. `delete_all_data` includes people. Live in-meeting recognition is v2.
- **Words Echo should know.** A per-user vocabulary of the words a listener has no way to guess: product names, companies, streets, colleagues. A meeting on 2026-08-24 wrote one product name six different ways in eight mentions (Langola → Ingola, Langura, sull'angolo, lana gola, Nongula, Nongulo), and the same meeting turned "Gianluca" — a name Echo had already been told, by somebody saving that voice — into "Jan Luca". So the list has two halves: what the person typed (Settings → Words), and the names of enrolled people, derived from the `people` table on every read and therefore never stale (removing one is remembered, or the next read would put it back). It is used twice: as the decoder's initial prompt on the lanes whose text is kept — the live final and the catch-up pass, never the caption lane — capped at ~200 tokens / 800 characters and truncated at a whole entry, because past that whisper starts writing the prompt back out into the transcript; and as a deterministic near-miss pass over each decoded line afterwards, which is conservative by construction (short words never looked at, short entries matched letter-for-letter only, everyday words of the meeting language never touched, the looser the rule the longer the entry has to be, and every rule but the tightest also asks for something the decoder itself did — a capital mid-sentence, an apostrophe where an article ran into the next word — because past a point the letters alone stop telling a misheard name from an ordinary word). It is tuned on a false-positive count rather than a hit count and gives away recall freely: of those six spellings it puts four right and leaves *Ingola* and *Langura* alone, because nothing tight enough to be trusted reaches them without also turning *bianco* into a colleague called Bianchi. Every rewrite is recorded on the segment row (`corrections`), shown as a quiet note on the line, and reversible from what is stored. "Listen again" applies the current list to an old meeting.
- AI summaries + action items via a pluggable connector trait; v1 ships **Ollama** (local) and **Gemini** (AI Studio API key). Summary output language is a user setting (meeting language / English / fixed custom).
- Automatic meeting detection, nudged by a small floating panel near the menu bar ("Meeting detected", how long ago, a Start button, a ✕) — it appears without taking the keyboard away from whatever is on screen, and takes itself away if nobody touches it. The OS notification is the fallback for when no window can be put on screen (clicking it opens Echo with a prominent Start button); tray menu also has Start — desktop notification action buttons are not supported by Tauri's plugin.
- History: browse, search (FTS), re-open past meetings; export Markdown/PDF/DOCX.
- Onboarding wizard: permissions → model download → summaries setup. Non-technical copy throughout.
- Background/tray app; window/launcher always remains a first-class route (Linux tray support is not universal).
- Matches Meetily Pro: diarization, auto-detect, summary templates (6 built-in + custom), exports, transcript search. Quality presets deliberately *not* matched: Echo ships one model and keeps it current itself (see mantra 1).

### Non-goals (v1)
Windows. Calendar integration. Chat-with-meetings (schema leaves room). Team deployment. Video. PulseAudio-only systems (PipeWire required on Linux; PulseAudio fallback post-v1). User-defined arbitrary connectors (the trait is extensible; v1 compiles Ollama + Gemini).

## 2. Stack

| Layer | Choice | Notes |
|---|---|---|
| Shell | Tauri 2.x | Tray, single-instance, notifications, updater plugins |
| Frontend | React 18 + TS + Vite + Tailwind | Sana-style minimal UI |
| ASR | whisper-rs (whisper.cpp) in-process | **Per-OS builds with backends compiled in**: macOS binary = `metal`+`coreml` features; Linux binary = `vulkan` feature. Runtime = GPU *init* attempt with CPU fallback (`n_threads` clamped, conservative default). CoreML needs a matching `*-encoder.mlmodelc` per model — catalogued and downloaded alongside the GGML file, and paired to *its own* GGML file so an older model still serving a meeting loads its own encoder; accelerates encoder only. |
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
         language, avg_confidence, revision, is_final, corrections)  -- corrections = words put right against "Words Echo should know", JSON, NULL when untouched
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
Poll every 5s when idle: (a) running meeting apps (zoom.us, Teams, Webex, Discord, Slack); (b) input-device-in-use *by something other than Echo* (macOS: per-process CoreAudio `kAudioProcessPropertyIsRunningInput`, falling back to `kAudioDevicePropertyDeviceIsRunningSomewhere` on macOS 13; Linux PipeWire active input streams excluding our own).

The two are not worth the same, and the difference is the heuristic: **(a) alone is never a meeting** — Slack and Discord are open all day and Zoom lives in the tray — so it counts only as corroboration. (b) is necessary. A live mic *with* a meeting app running is nudged after 2 polls (10s); a live mic *alone* after 6 polls (30s), which outlasts dictation/Siri while still catching a browser meeting (Google Meet in a tab has no process to match) inside its first half-minute. Once noticed, the latch survives up to 30s of mic silence while a meeting app is still running (hysteresis, so one flickering meeting is not two nudges) and then lets go: an idle app must never leave the UI or the tray claiming a meeting.

Detected → tray badge, plus exactly one nudge per meeting: the floating panel (Start right there; ✕ says nothing more about *that* meeting, and is not a snooze), or the notification (click opens Echo, big Start button) when no window can be put on screen — never both. Auto-stop suggestion after the mic has been clear >2 min while recording — "clear" means the corroborated signal (b) only, so a Slack nobody ever quits cannot switch the safety net off. Snooze 1h.

## 4. UI (Sana-style; see mantra 2)

Light theme, white, generous whitespace, rounded-xl cards, 1px #eee borders, near-black text, one accent used sparingly, pill buttons, Inter/system font.

1. **Home** — status hero ("Waiting for your next meeting" / live card with elapsed + live caption), search bar, recent meetings (title, date, duration, language chip, snippet).
2. **Live** — transcript stream with speaker chips ("You" / "Speaker 1"…), visible recording indicator, pause/stop, "Flag action item" marker button, degraded-capture banner when applicable.
3. **Meeting detail** — tabs: Recap (rendered summary + action-items checklist with owner chips, copy/export, regenerate-with-model picker, template picker), Transcript (virtualized, speaker colors, in-page search, click-to-play), Info (what was captured, storage used, delete options).
4. **History/Search** — FTS across meetings.
5. **Settings** — General (launch at login, detection on/off, storage location, summary language), Speech (a status card: whether Echo can understand speech, and the one-time download's state and size — nothing to choose, since there is one model), Summaries (On this computer [Ollama] / Google Gemini [key, masked, keychain] with plain-language privacy note), Templates (built-ins + custom CRUD), Data (storage report, delete-all), Advanced (engine, GPU in use, model identifiers, diagnostics export).
6. **Onboarding** — Welcome → Permissions (mic; macOS screen-recording with "why" copy, handles restart-required and denied states, "Open System Settings" button) → Download (~4.3 GB one-time, resumable, size read from the catalog; no preset to pick) → Summaries (auto-detect Ollama / Gemini key / skip).

Tray: idle/detected/recording/processing states, recomputed from three facts (is capture on air, is a meeting detected, is any meeting's work still queued) rather than remembered — a live recording outranks outstanding work, which outranks a detected meeting. The recording one gently pulses and the processing one is a separate spinner motif — an arc travelling round the mark, so "still working on the last meeting" is never mistaken for "still listening" (one 500ms timer, alive only while one of the two is on screen; paused holds a frame with no timer). Left-clicking while recording brings the same floating panel up under the icon with the elapsed time and Stop; left-clicking otherwise brings the window back. Menu: Start/Stop, Open Echo, Pause detection 1h, Quit. Close-to-tray, but app is always restorable from launcher/window (Linux tray caveat). Single instance.

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
