# Capture fixtures

Small, deterministic audio for tests that need a real file on disk rather than a
buffer built in memory. Nothing here is a recording of a person, and nothing here
is large: keep it that way.

Most capture tests generate their audio in code, because a generated buffer is
exact and needs no I/O. A fixture earns its place only when the test has to prove
that reading a *file* works — that the WAV reader, the resampler and speech
detection agree with each other on something none of them produced.

## `speech-and-silence-16k.wav`

16 kHz, mono, 16-bit, 7 seconds. Two voiced stretches separated by silence long
enough to end an utterance:

| from | to | contents |
|---|---|---|
| 0 ms | 1000 ms | room noise (well below any speech threshold) |
| 1000 ms | 3000 ms | voiced stretch A |
| 3000 ms | 4500 ms | room noise |
| 4500 ms | 6000 ms | voiced stretch B |
| 6000 ms | 7000 ms | room noise |

The voiced stretches are a stack of harmonics at 140/280/560 Hz with a slow
amplitude wobble — vowel-shaped rather than a bare sine, so a real speech
detector has something to work with and a person who opens the file hears
something recognisable. It is **not** speech, so it is no use for checking
transcription accuracy; it is for checking timing and segmentation.

The root `.gitignore` currently excludes `*.wav`, so this file may be missing from
a fresh clone. The test that uses it
(`audio::vad::tests::the_fixture_on_disk_segments_where_its_layout_says_it_should`)
writes an identical copy to a temporary directory when the committed one is
absent, so it passes either way. To commit the fixture for real, add an exception
to `.gitignore`:

```gitignore
!src-tauri/tests/fixtures/*.wav
```

Regenerate with:

```sh
python3 src-tauri/tests/fixtures/make_speech_and_silence.py   # run from the repo root
```

The script uses only the Python standard library and writes the same bytes every
time, so a regenerated fixture is a no-op in `git diff`.
