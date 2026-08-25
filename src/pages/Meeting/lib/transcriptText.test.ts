import { describe, expect, it } from "vitest";

import type { Channel, Segment, Speaker } from "@/lib/types";

import { buildTranscriptText, micFallbackLabel } from "./transcriptText";

function segment(
  id: string,
  tStartMs: number,
  channel: Channel,
  text: string,
  speakerId?: string,
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
    isFinal: true,
  };
}

function speaker(id: string, displayName: string, isSelf = false, aliasOf?: string): Speaker {
  return {
    id,
    meetingId: "m1",
    clusterKey: id,
    displayName,
    aliasOf,
    isSelf,
    speakingMs: 1000,
  };
}

/**
 * The mislabel of 2026-08-24, moved from the live banner into the Copy button.
 *
 * On a meeting where nothing this computer played ever reached Echo, the
 * microphone carried the whole room — the far end leaks in through the
 * speakers — so an unattributed microphone line is not safely anybody, and
 * calling it "You" claims Echo knows who spoke when it does not.
 */
describe("micFallbackLabel", () => {
  it('is "You" whenever the computer had its own channel', () => {
    const segments = [
      segment("a", 0, "mic", "Hello"),
      segment("b", 1000, "system", "Hi there"),
    ];
    expect(micFallbackLabel(segments, [])).toBe("You");
  });

  it("does not claim to be anybody on a mic-only meeting", () => {
    const segments = [segment("a", 0, "mic", "Hello"), segment("b", 1000, "mic", "And then")];
    expect(micFallbackLabel(segments, [])).toBe("Speaker");
  });

  it("borrows the name of the self row while that row is still carrying lines", () => {
    // Half a transcript saying "You" and half saying "Speaker" — one voice, one
    // meeting — is worse than an honest over-claim, so while the live pass's
    // self row is still attributed to, the unattributed half matches it.
    const speakers = [speaker("s1", "You", true)];
    const segments = [
      segment("a", 0, "mic", "Hello", "s1"),
      segment("b", 1000, "mic", "And then"),
    ];
    expect(micFallbackLabel(segments, speakers)).toBe("You");
  });

  it("stops borrowing once the offline pass has dropped that row", () => {
    const speakers = [speaker("s1", "Ana")];
    const segments = [
      segment("a", 0, "mic", "Hello", "s1"),
      segment("b", 1000, "mic", "And then"),
    ];
    expect(micFallbackLabel(segments, speakers)).toBe("Speaker");
  });

  it("follows a merge before deciding whether the self row is still there", () => {
    const speakers = [speaker("s1", "You", true), speaker("s2", "Speaker 2", false, "s1")];
    const segments = [segment("a", 0, "mic", "Hello", "s2")];
    expect(micFallbackLabel(segments, speakers)).toBe("You");
  });
});

describe("buildTranscriptText", () => {
  it("writes one timestamped, speaker-labelled line per segment", () => {
    const speakers = [speaker("s1", "Ana"), speaker("s2", "Bea")];
    const segments = [
      segment("a", 0, "mic", "Morning", "s1"),
      segment("b", 65_000, "system", "Morning to you", "s2"),
    ];
    expect(buildTranscriptText(segments, speakers)).toBe(
      "[0:00] Ana: Morning\n[1:05] Bea: Morning to you",
    );
  });

  it("never pastes a raw id for a line with no speaker of its own", () => {
    const segments = [
      segment("a", 0, "mic", "Morning"),
      segment("b", 1000, "system", "Morning to you"),
    ];
    expect(buildTranscriptText(segments, [])).toBe(
      "[0:00] You: Morning\n[0:01] Everyone else: Morning to you",
    );
  });

  it("keeps the order it was given and drops the lines with nothing in them", () => {
    const segments = [
      segment("a", 0, "mic", "Morning", "s1"),
      segment("b", 1000, "mic", "   ", "s1"),
      segment("c", 2000, "mic", "  Later  ", "s1"),
    ];
    expect(buildTranscriptText(segments, [speaker("s1", "Ana")])).toBe(
      "[0:00] Ana: Morning\n[0:02] Ana: Later",
    );
  });

  it("uses the surviving name for a line that was merged into another speaker", () => {
    const speakers = [speaker("s1", "Ana"), speaker("s2", "Speaker 2", false, "s1")];
    expect(buildTranscriptText([segment("a", 0, "mic", "Morning", "s2")], speakers)).toBe(
      "[0:00] Ana: Morning",
    );
  });

  it("has nothing to say about an empty transcript", () => {
    expect(buildTranscriptText([], [])).toBe("");
  });
});
