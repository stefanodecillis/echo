"""Generate the capture test fixture: 16 kHz mono WAV, two voiced stretches.

Layout (milliseconds):
    0..1000     room noise
    1000..3000  voiced stretch A
    3000..4500  room noise
    4500..6000  voiced stretch B
    6000..7000  room noise
"""

import math
import struct
import wave

RATE = 16_000
OUT = "src-tauri/tests/fixtures/speech-and-silence-16k.wav"


def room_noise(n, start):
    # Deterministic, tiny, and well below any speech threshold.
    for i in range(start, start + n):
        yield (math.sin(i * 0.37) + math.sin(i * 1.13)) * 0.0008


def voiced(n, start):
    # A vowel-ish stack of harmonics: enough structure to look like speech to a
    # loudness gate and to a real detector, and audible if someone opens it.
    for i in range(start, start + n):
        t = i / RATE
        s = (
            math.sin(2 * math.pi * 140 * t) * 0.45
            + math.sin(2 * math.pi * 280 * t) * 0.22
            + math.sin(2 * math.pi * 560 * t) * 0.10
        )
        # Gentle amplitude wobble, like a held syllable.
        s *= 0.85 + 0.15 * math.sin(2 * math.pi * 4.5 * t)
        yield s * 0.6


def ms(x):
    return RATE * x // 1000


def main():
    samples = []
    pos = 0
    for kind, duration in [
        ("noise", 1000),
        ("voice", 2000),
        ("noise", 1500),
        ("voice", 1500),
        ("noise", 1000),
    ]:
        n = ms(duration)
        gen = voiced(n, pos) if kind == "voice" else room_noise(n, pos)
        samples.extend(gen)
        pos += n

    with wave.open(OUT, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        frames = bytearray()
        for s in samples:
            v = max(-1.0, min(1.0, s))
            frames += struct.pack("<h", int(round(v * 32767)))
        f.writeframes(bytes(frames))
    print(f"wrote {OUT}: {len(samples)} samples, {len(samples) / RATE:.2f} s")


if __name__ == "__main__":
    main()
