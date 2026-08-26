import { describe, expect, it } from "vitest";

import type { Speaker } from "@/lib/types";

import { canonicalSpeakers, isEchoOwnLabel, resolveSpeaker } from "./speakers";

function speaker(id: string, displayName: string, aliasOf?: string): Speaker {
  return {
    id,
    meetingId: "m1",
    clusterKey: id,
    displayName,
    aliasOf,
    isSelf: false,
    speakingMs: 1000,
  };
}

describe("resolveSpeaker", () => {
  it("follows a merged speaker to the name actually shown", () => {
    const speakers = [speaker("a", "Ana"), speaker("b", "Speaker 2", "a")];
    expect(resolveSpeaker("b", speakers)?.displayName).toBe("Ana");
  });

  it("follows a chain of merges", () => {
    const speakers = [speaker("a", "Ana"), speaker("b", "Speaker 2", "a"), speaker("c", "Speaker 3", "b")];
    expect(resolveSpeaker("c", speakers)?.displayName).toBe("Ana");
  });

  it("gives up rather than hanging on a cyclic alias chain", () => {
    // Nothing should ever write this, which is exactly why the renderer has to
    // survive it: a hang here is a blank Meeting page with no way back.
    const speakers = [speaker("a", "Ana", "b"), speaker("b", "Bea", "a")];
    expect(() => resolveSpeaker("a", speakers)).not.toThrow();
  });

  it("has no answer for a line with no speaker, or one that is not in the list", () => {
    expect(resolveSpeaker(undefined, [speaker("a", "Ana")])).toBeUndefined();
    expect(resolveSpeaker("gone", [speaker("a", "Ana")])).toBeUndefined();
  });

  it("leaves an unmerged speaker exactly where it is", () => {
    const speakers = [speaker("a", "Ana")];
    expect(resolveSpeaker("a", speakers)).toBe(speakers[0]);
  });
});

describe("canonicalSpeakers", () => {
  it("drops the merged-away rows so nobody is offered twice", () => {
    const speakers = [speaker("a", "Ana"), speaker("b", "Speaker 2", "a")];
    expect(canonicalSpeakers(speakers).map((s) => s.id)).toEqual(["a"]);
  });
});

/**
 * `isEchoOwnLabel` is a hand-maintained mirror of `is_default_name` in
 * `src-tauri/src/diarize/people.rs`. The Rust side is pinned by its own tests
 * and this side, until now, by nothing — so a change on one side could drift
 * silently, and the drift shows up as the "remember this voice" checkbox
 * offering to save a name somebody typed as if Echo had made it up.
 *
 * The two sides are not letter-for-letter identical, and the two places they
 * differ are pinned below so the difference stays a decision rather than a
 * surprise: this side trims surrounding whitespace and accepts any number after
 * "Speaker", where `is_default_name` compares exactly and only knows the twelve
 * labels the separation pass can actually write (`cluster::MAX_SPEAKERS`).
 * Both widenings are unreachable from stored data — a display name is written
 * by `pipeline::display_name` or typed into a trimmed field — and both fail in
 * the safe direction here, towards "Echo made this up" on a string Echo would
 * never have been given.
 */
describe("isEchoOwnLabel", () => {
  it("knows the two labels the separation pass writes", () => {
    expect(isEchoOwnLabel("You")).toBe(true);
    expect(isEchoOwnLabel("Speaker 1")).toBe(true);
    expect(isEchoOwnLabel("Speaker 2")).toBe(true);
    expect(isEchoOwnLabel("Speaker 12")).toBe(true);
  });

  // Divergence 1: the core compares the string exactly. Nothing can store a
  // padded display name, so this is slack rather than disagreement.
  it("ignores the whitespace around one", () => {
    expect(isEchoOwnLabel("  Speaker 3  ")).toBe(true);
    expect(isEchoOwnLabel(" You ")).toBe(true);
  });

  // Divergence 2: the core only knows Speaker 1 to Speaker 12, because twelve
  // is as many voices as it will ever tell apart.
  it("accepts a number past the twelve the core can produce", () => {
    expect(isEchoOwnLabel("Speaker 13")).toBe(true);
  });

  it("leaves a name somebody typed alone", () => {
    expect(isEchoOwnLabel("Ana")).toBe(false);
    expect(isEchoOwnLabel("Marco Rossi")).toBe(false);
    expect(isEchoOwnLabel("")).toBe(false);
  });

  it("does not mistake a name that merely starts like one", () => {
    expect(isEchoOwnLabel("Speaker")).toBe(false);
    expect(isEchoOwnLabel("Speaker 2's laptop")).toBe(false);
    expect(isEchoOwnLabel("Speakerphone")).toBe(false);
    expect(isEchoOwnLabel("You and me")).toBe(false);
    expect(isEchoOwnLabel("Youssef")).toBe(false);
  });

  it("is case-sensitive, exactly as the core is", () => {
    expect(isEchoOwnLabel("speaker 2")).toBe(false);
    expect(isEchoOwnLabel("you")).toBe(false);
  });
});
