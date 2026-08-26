import type { Segment } from "@/lib/types";

/**
 * When Echo is not sure it heard a line.
 *
 * **This is a hand-maintained mirror of `src-tauri/src/asr/confidence.rs`.** The
 * argument for the two arms, the numbers, and the limits they do not cover all
 * live there and are not repeated here — read that file before changing
 * anything in this one. A Rust test names this file and pins the three
 * constants, so a number that moves on one side and not the other fails the
 * build rather than quietly making a line read as shaky in the recap and solid
 * on screen.
 *
 * The short version: a line is shaky if it falls under {@link SHAKY_FLOOR}
 * outright, **or** under {@link SHAKY_BELOW_MEDIAN} of how this meeting normally
 * reads. Neither arm works alone — a purely relative bar marks almost nothing on
 * a meeting that is wrong throughout, and a purely absolute one cannot see a
 * mostly-fine meeting with a bad stretch in it.
 */

/** Below this a line is shaky whatever the rest of the meeting looks like. */
export const SHAKY_FLOOR = 0.55;

/** How far short of this meeting's own middle a line falls before it is shaky. */
export const SHAKY_BELOW_MEDIAN = 0.75;

/** Measured lines needed before the meeting has a middle at all. */
export const MEDIAN_NEEDS = 8;

export interface HowSureThisMeetingIs {
  /** The middle of the measured lines, or `null` when there are too few to have
   * one. `null` means "no normal yet", not "this meeting is fine". */
  readonly median: number | null;
  /** Was Echo unsure it heard this line? `undefined` — nothing measured it — is
   * never shaky: see the Rust module for why that is a rule and not a
   * convenience. */
  isShaky(confidence: number | null | undefined): boolean;
}

/**
 * Read a whole meeting once, then ask it about each line.
 *
 * Only lines that have words count toward the middle: silence read as nothing is
 * not the engine being unsure, and counting it would drag the bar down until no
 * real line could fall below it.
 *
 * **Not for the Live view.** During a meeting the median has no stable
 * population — a line marked at 10:03 would unmark itself at 10:40 as more of
 * the meeting arrives, and a view that changes its mind about the past is worse
 * than one that says nothing.
 */
export function howSureThisMeetingIs(segments: readonly Segment[]): HowSureThisMeetingIs {
  const measured = segments
    .filter((s) => s.text.trim() !== "")
    .map((s) => s.avgConfidence)
    .filter((c): c is number => typeof c === "number")
    .sort((a, b) => a - b);

  let median: number | null = null;
  if (measured.length >= MEDIAN_NEEDS) {
    const mid = Math.floor(measured.length / 2);
    median =
      measured.length % 2 === 0 ? (measured[mid - 1] + measured[mid]) / 2 : measured[mid];
  }

  return {
    median,
    isShaky(confidence) {
      if (typeof confidence !== "number") return false;
      if (confidence < SHAKY_FLOOR) return true;
      if (median === null) return false;
      return confidence < median * SHAKY_BELOW_MEDIAN;
    },
  };
}
