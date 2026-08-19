# Echo

Meeting transcripts, recaps and action items that never leave your computer.
macOS on Apple Silicon and Linux on x86_64.

Echo records your microphone and what your computer plays, writes down what was
said, works out who said it, and drafts a recap. The audio, the text and the
database all sit on your own drive. Recaps can be written by a model running on
your machine, or by Google Gemini if you decide that trade is worth it.

## Four rules the code follows

**Use nothing until needed, release it when idle.** No speech engine in memory,
no audio stream open, no inference session alive unless a recording or an explicit
action needs it. An idle Echo is a window, a tray icon and a five-second poll.

**Zero jargon in anything a person reads.** Not "transcribing" but "Listening".
Not "downloading model" but "Downloading what Echo needs to understand speech".
Errors say what you can do about it. Model names, graphics backends and file paths
live under Settings > Advanced and nowhere else. `src/lib/copy.ts` explains the
rule in full and holds every string.

**Raw per-channel audio on disk is the truth.** Transcripts, speakers and recaps
are all derived from it and can be rebuilt. Nothing downstream is allowed to block
or drop capture. If the transcript falls behind, it catches up from the files.

**Built for someone who is not an engineer.** No terminal, no config file, no
prerequisites: double-click the installer, click through onboarding, have a
working app. Recording and transcripts work before any recap backend is set up.
Every error names a next step, no screen is a dead end, and the defaults are safe
enough that Settings is optional.

## Dev setup

You need Node 20 or newer and a stable Rust toolchain (1.90+). On Linux you also
need WebKitGTK, the tray library, ALSA and PipeWire headers;
`.github/workflows/ci.yml` has the exact package list.

```sh
npm install
npm run tauri dev        # run the app
npm run build            # type-check and build the frontend
```

The first Rust build compiles whisper.cpp and ONNX Runtime, which takes several
minutes and a few GB of disk. After that it is cached.

Checks, the same ones CI runs:

```sh
npx tsc --noEmit
cd src-tauri
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
```

### If `npm` and `node` do not work in scripts

On some machines a shell profile that lazy-loads nvm breaks bare `node`, `npm`
and `npx` in non-interactive shells (you get a `_lazy_load_nvm` error). Use the
absolute paths, or put Homebrew first on `PATH`:

```sh
export PATH=/opt/homebrew/bin:$PATH
command npm install
```

Invoking the tools under `node_modules` directly also works and needs no profile
at all:

```sh
/opt/homebrew/bin/node ./node_modules/typescript/bin/tsc --noEmit
/opt/homebrew/bin/node ./node_modules/vite/bin/vite.js build
```

## Where things are

```
docs/DESIGN.md          the spec: schema, module contracts, milestones
docs/REVIEW-NOTES.md    why the design looks like it does
src/                    React 18 + TypeScript + Vite + Tailwind
  lib/types.ts          hand-written mirror of the Rust types
  lib/ipc.ts            typed wrapper for every command and event
  lib/copy.ts           every word a person reads
  pages/                one directory per screen; routes/ are thin re-exports
src-tauri/
  migrations/           SQLite schema
  src/db/               pool and one function per database operation
  src/audio/            microphone, system audio, meeting clock, chunks, speech detection
  src/asr/              downloads and the speech engine
  src/diarize/          who said what
  src/summarize/        the recap backends
  src/detect/           noticing a meeting started
  src/session/          capture state machine and the job runner
  src/export/           Markdown, DOCX, PDF
```

The Rust↔TypeScript boundary is checked by construction: all 66 IPC commands have
a wrapper in `ipc.ts` and an entry in the `echo_command_handler!` macro, and every
event name, payload struct and enum in `events.rs`/`types.rs` has a matching
declaration in `lib/types.ts`.

## Status, honestly

Everything below compiles, `cargo test` is green (420 unit tests, one ignored),
`cargo clippy --all-targets -- -D warnings` is clean, and the debug binary links
against whisper.cpp and ONNX Runtime. **None of it has been exercised against
real audio hardware or a real download**, because that needs a machine with a
microphone, a permission dialog to click, and 1.6 GB of weights.

### Works, and has been tested

- Database, migrations, settings, keychain access, paths, the redacted log.
- Capture state machine: start/stop/pause/resume, degraded paths (lost one
  source), crash recovery, meeting clock, chunk journal.
- The job runner: ordering, cancellation, recording preempting background work,
  resuming parked work after a restart.
- Speech engine plumbing: catalog, SHA-256 verification, resumable downloads,
  live partials, catch-up-from-disk, idle release.
- Speaker attribution: segmentation and fingerprint decoding, clustering,
  overlap-aware timeline, rename/merge/unmerge as a non-destructive alias graph.
- Recaps: map-reduce chunking, strict action-item JSON with one repair retry,
  markdown sanitization, Ollama and Gemini transports.
- Meeting detection on macOS (CoreAudio), debounce, snooze, auto-stop suggestion.
- Exports: Markdown, plain text, DOCX.
- Every screen: Home, Live, Meeting detail, Search, Settings, Onboarding.

### Not verified, or not done

- **Audio capture has never run.** macOS ScreenCaptureKit and the cpal
  microphone path compile and are unit-tested where they can be, but no audio has
  been captured on real hardware. The M0-S1 spike is the first time that happens.
- **Linux is compile-unverified.** `audio/system_linux.rs` (PipeWire monitor
  capture) and `detect/linux.rs` (PipeWire input-in-use probe) were written on a
  macOS machine and could not be compiled at all. They are isolated behind
  `#[cfg(target_os = "linux")]` so they cannot affect the macOS build, but the
  first Linux CI run is where their API usage gets checked. Highest risk: the
  `AudioInfoRaw`/`PodSerializer` format negotiation and the shutdown plumbing.
- **PDF export is unavailable and says so.** Tauri 2 has no headless
  print-to-file API — only a macOS-only native print dialog — so `write_pdf`
  returns a plain "PDF isn't available on this computer. Word or Markdown will
  work." The format is still offered in the menu, which is a rough edge.
- **The speaker threshold is provisional.** `cluster::DISTANCE_THRESHOLD` is
  calibrated to WeSpeaker ResNet34-LM, but the catalog ships CAM++ because the
  ResNet34-LM export sits in a gated Hugging Face repo an unauthenticated
  download cannot reach. The pass runs; the speaker count is unvalidated. The
  M0-S4 fixture benchmark has to be re-run before v1, and a test fails loudly if
  the asset is swapped again.
- **Chunks on disk are WAV, not FLAC.** Deliberate and reversible in about three
  lines: the only FLAC crate in the tree encodes but cannot decode, so FLAC
  chunks would be unreadable by catch-up, the mixdown and click-to-play. WAV also
  writes sample-by-sample, so a crash costs milliseconds instead of a whole 30 s
  window. Cost is space: roughly 115 MB per hour per channel.
- **Model licenses need an audit.** The catalog records a license and revision
  for all nine assets and the URLs were verified by HTTP HEAD, but the speaker
  assets in particular need a human to confirm the terms before shipping.
- **No downloads have been run end to end.** Range + If-Range resume was
  confirmed against Hugging Face's CDN and Silero VAD (2.3 MB) was fetched once
  to compute its hash, but no whisper weights have ever been downloaded.
- Click-to-play in the transcript, live speaker clustering (deliberately out of
  scope for v1), and installer signing/notarization are not done.

CI compiles and tests on both platforms. It cannot check audio capture,
permission prompts or tray behaviour, because a headless runner has no microphone
and nobody to click a permission dialog. Those go on the manual hardware matrix.

## Privacy

Nothing is sent anywhere unless you turn on Gemini recaps, and that screen says
plainly what gets sent before you enter a key. A local recap address that is not
on this computer is refused until you explicitly acknowledge that what was said
would travel. Keys go in the OS keychain, never into a file. There is no
telemetry. The diagnostics log is bounded, rotates, and holds no transcript text
and no keys.

Recording captures everything your computer plays, not only the meeting window.
Echo says so rather than implying otherwise.
