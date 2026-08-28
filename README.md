<p align="center">
  <img src="docs/assets/echo-banner.png" alt="Echo" width="720">
</p>

<h1 align="center">Echo</h1>

<p align="center">
  Meeting transcripts, recaps and action items that never leave your computer.<br>
  macOS on Apple Silicon and Linux on x86_64.
</p>

<p align="center">
  <a href="#license"><img src="https://img.shields.io/badge/license-MIT-black" alt="MIT licensed"></a>
  <img src="https://img.shields.io/badge/local--first-no%20telemetry-black" alt="Local first, no telemetry">
  <img src="https://img.shields.io/badge/Tauri%202-Rust%20%2B%20React-black" alt="Tauri 2, Rust and React">
</p>

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
microphone, a permission dialog to click, and 4.3 GB of weights.

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
- **The speaker threshold is measured, on synthetic voices and one real
  meeting.** `cluster::DISTANCE_THRESHOLD` now belongs to the network the catalog
  actually ships: WeSpeaker ResNet34-LM, the voice-print model of the reference
  pyannote 3.1 and community-1 pipelines, fetched unauthenticated from the
  sherpa-onnx mirror. It replaced CAM++, against which the number had never been
  measured at all. The number itself was chosen by
  `cargo run --release --example voices_fixture`, which builds meetings of 2, 3,
  5 and 7 *known* speakers out of macOS text-to-speech voices and sweeps the
  threshold across the whole cosine range, and confirmed by
  `cargo run --release --example speakers_probe -- --sweep` on a real recorded
  meeting. A test fails loudly if the asset is swapped without redoing that.
  What is still unproven: those fixtures are cleaner than real speech, and one
  real meeting is one real meeting. Diarization error rate against a labelled
  corpus is not something we can measure here.
- **Chunks on disk are WAV, not FLAC.** Deliberate and reversible in about three
  lines: the only FLAC crate in the tree encodes but cannot decode, so FLAC
  chunks would be unreadable by catch-up, the mixdown and click-to-play. WAV also
  writes sample-by-sample, so a crash costs milliseconds instead of a whole 30 s
  window. Cost is space: roughly 115 MB per hour per channel.
- **Model licenses need an audit.** The catalog records a license and revision
  for every asset, and the speaker assets were downloaded unauthenticated and
  hashed from the bytes that arrived. Both are Apache-2.0/MIT as recorded. One
  thing a human should still confirm before shipping: pyannote's own
  `speaker-diarization-community-1` pipeline is CC-BY-4.0, not Apache-2.0, and
  while Echo ships neither its files nor its clustering, anything that borrows
  from it later inherits that licence.
- **The speech weights have been downloaded and hashed** (2026-08-20): both
  files of the current model were fetched from the pinned upstream commit and
  their SHA-256s computed from the bytes that arrived, which is what the catalog
  now records. Note that Hugging Face's ETag on these is *not* the SHA-256, so a
  hash has to come from the download rather than from a HEAD.
  Range + If-Range resume was confirmed separately against the same CDN.
- Click-to-play in the transcript and live speaker clustering are deliberately
  out of scope for v1. Releases are code-signed and update-signed, but **not
  notarized by Apple** — see Installing below for what that means on first
  launch.

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

## Installing

Releases are on the [releases page](https://github.com/stefanodecillis/echo/releases).

**macOS.** Download the `.dmg`, drag Echo to Applications. The first launch says
Apple cannot verify Echo, because Echo is not notarized — notarization needs a paid
Apple Developer account. Right-click Echo and choose **Open**, once, and macOS
remembers. Nothing about that warning is specific to Echo; it is what every
unnotarized app does.

**Linux.** The AppImage keeps itself up to date. The `.deb` does not — download a
newer one when you want it.

## Keeping itself up to date

Echo asks GitHub every half hour whether there is a newer release. If there is, it
downloads and installs it quietly and then offers a restart, in the corner and in
the tray menu. Nothing restarts on its own.

It never does this while a meeting is being recorded, or while a meeting that has
finished still has its transcript or recap outstanding — those get the machine
first, and the check simply comes back later. Restarting mid-work is safe anyway:
parked work is requeued at launch with its progress intact.

Restarting after an update means Echo gets ready to understand speech again, which
takes a few minutes: the compiled speech model is keyed to the app that asked for
it, so a new app means a new compile. Recording still works throughout — audio goes
to disk from the first second and the words fill in afterwards.

Every update carries an ed25519 signature, and Echo refuses any bundle that does
not verify against the public key compiled into it. Being able to serve the release
endpoint is not enough to hand Echo new code.

## Signing

Two certificates, doing two different jobs. Neither is an Apple Developer ID.

**"Echo Local Signing"** — self-signed, in the login keychain of the machine that
builds, named in `bundle.macOS.signingIdentity`. It exists so macOS permission
grants (microphone, screen recording) survive rebuilds: ad-hoc signatures are
re-keyed on every build and macOS forgets the grants. On a new machine either
create one (Keychain Access → Certificate Assistant → Create a Certificate → type
"Code Signing") or set the identity to `"-"` and accept re-prompting.

**"Echo Release Signing"** — self-signed too, but it lives in a password manager
and in this repository's secrets, and CI signs every release with it. It has to
never change: macOS keys those same permission grants to the signing identity, so
a new certificate means every user grants microphone and screen recording again.

What self-signing does **not** do is satisfy Gatekeeper. That needs notarization,
which needs the paid account. The release workflow is ready for it — Tauri
notarizes when `APPLE_ID`, `APPLE_PASSWORD` and `APPLE_TEAM_ID` are set and skips
with a warning when they are not, so switching it on is three secrets and no code.

## Cutting a release

```sh
node scripts/version.mjs 0.2.0     # writes all three files that carry the version
git commit -am "chore: 0.2.0" && git tag v0.2.0 && git push --follow-tags
```

The tag builds both platforms and opens a **draft** release. Check the artifacts,
then publish — publishing is what makes `latest.json` live, and every installed
copy is polling it.

The version lives in `package.json`, `src-tauri/Cargo.toml` and
`src-tauri/tauri.conf.json`, and the updater compares releases against only the
last of those. `scripts/version.mjs` writes all three and CI refuses a tag that
disagrees with them, because that particular mistake makes every installed copy
download the same release every half hour forever.

## License

MIT. See [LICENSE](LICENSE).

The models Echo downloads at runtime are not covered by this licence and carry
their own, recorded next to every entry in `src-tauri/src/asr/catalog.rs`:

| Model | Licence |
| --- | --- |
| Whisper weights (OpenAI, GGML conversion by whisper.cpp) | MIT |
| Silero VAD | MIT |
| pyannote segmentation 3.0 (CNRS), ONNX export by sherpa-onnx | MIT |
| WeSpeaker embeddings (wenet-e2e), ONNX export by sherpa-onnx | Apache-2.0 |

Nothing is bundled in this repository. The app fetches what it needs on first run,
pinned to an exact upstream revision and verified by hash.
