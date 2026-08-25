import { describe, expect, it } from "vitest";

import type { Channel, Segment } from "../../lib/types";

import { mergeFinals, type TranscriptLineData } from "./useTranscriptStream";

function segment(
  id: string,
  text: string,
  { tStartMs = 0, channel = "mic" as Channel, speakerId = undefined as string | undefined, isFinal = true } = {},
): Segment {
  return {
    id,
    meetingId: "m1",
    tStartMs,
    tEndMs: tStartMs + 1000,
    channel,
    speakerId,
    text,
    revision: 1,
    isFinal,
  };
}

function lines(...segments: Segment[]): Map<string, TranscriptLineData> {
  return new Map(
    segments.map((s) => [
      s.id,
      {
        id: s.id,
        tStartMs: s.tStartMs,
        channel: s.channel,
        speakerId: s.speakerId,
        text: s.text,
        isFinal: s.isFinal,
      },
    ]),
  );
}

/**
 * The `null` is the whole test.
 *
 * This runs every five seconds for the length of a meeting, and almost every
 * run has nothing new in it. Handing back a fresh Map anyway re-sorts the list,
 * hands `VirtualList` a brand-new array, and trips the "did a line arrive?"
 * effect that follows the view down — a transcript that twitches once a beat,
 * forever. A `null` that should have been a Map is the other half of the same
 * bug and the worse one: the poll is the floor under the events, so a wrong
 * `null` is a Live view frozen for the rest of the meeting while every word is
 * being safely recorded. That froze a real meeting on 2026-08-24.
 */
describe("mergeFinals", () => {
  it("says nothing changed when the poll brings back what is already on screen", () => {
    const prev = lines(segment("a", "Morning"), segment("b", "Morning to you"));
    expect(mergeFinals(prev, [segment("a", "Morning"), segment("b", "Morning to you")])).toBeNull();
  });

  it("says nothing changed for an empty read", () => {
    expect(mergeFinals(lines(segment("a", "Morning")), [])).toBeNull();
    expect(mergeFinals(new Map(), [])).toBeNull();
  });

  it("brings back a new line, and leaves the identity of the old one alone", () => {
    const prev = lines(segment("a", "Morning"));
    const next = mergeFinals(prev, [segment("a", "Morning"), segment("b", "Later")]);
    expect(next).not.toBeNull();
    expect(next).not.toBe(prev);
    expect(next?.size).toBe(2);
    expect(next?.get("a")).toBe(prev.get("a"));
    expect(next?.get("b")?.text).toBe("Later");
  });

  it("notices a line whose words were rewritten", () => {
    const prev = lines(segment("a", "Nongula"));
    const next = mergeFinals(prev, [segment("a", "Langola")]);
    expect(next?.get("a")?.text).toBe("Langola");
  });

  it("notices a line that has been given a speaker", () => {
    const prev = lines(segment("a", "Morning"));
    const next = mergeFinals(prev, [segment("a", "Morning", { speakerId: "s1" })]);
    expect(next?.get("a")?.speakerId).toBe("s1");
  });

  it("notices a line that has moved in time, or settled", () => {
    const prev = lines(segment("a", "Morning"));
    expect(mergeFinals(prev, [segment("a", "Morning", { tStartMs: 500 })])).not.toBeNull();
    expect(mergeFinals(prev, [segment("a", "Morning", { isFinal: false })])).not.toBeNull();
  });

  it("never mutates the map it was handed", () => {
    const prev = lines(segment("a", "Morning"));
    mergeFinals(prev, [segment("b", "Later")]);
    expect(prev.size).toBe(1);
  });

  it("copies the previous lines only once however many segments changed", () => {
    // The second changed segment has to be looked up in the copy being built,
    // not in `prev`, or a read carrying two changes to the same line loses the
    // first one.
    const prev = lines(segment("a", "One"));
    const next = mergeFinals(prev, [segment("a", "Two"), segment("a", "Three")]);
    expect(next?.get("a")?.text).toBe("Three");
    expect(next?.size).toBe(1);
  });

  it("says nothing changed when a re-read repeats a segment it already folded in", () => {
    const prev = lines(segment("a", "One"));
    const next = mergeFinals(prev, [segment("a", "Two"), segment("a", "Two")]);
    expect(next?.get("a")?.text).toBe("Two");
  });
});
