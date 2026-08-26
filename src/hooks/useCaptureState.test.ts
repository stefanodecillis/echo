import { describe, expect, it } from "vitest";

import { complaintIsDue } from "./useCaptureState";

/**
 * The poll this belongs to is the floor under a dead event transport: it exists
 * because on 2026-08-24 the transport died mid-meeting and every screen drawn
 * from events froze on the last one that got through. It used to swallow its
 * own failures whole, so the second time that happened there would have been
 * nothing in the console to tell it apart from a quiet meeting.
 *
 * Saying so has to survive both shapes of failure, which is what this is about:
 * one read that fails is a blip, and one that fails on every tick for half an
 * hour must not bury the line that explains it.
 */
describe("complaintIsDue", () => {
  it("writes the first failure down at once", () => {
    // A log that starts half a minute after the trouble did leaves whoever
    // reads it guessing about the gap.
    expect(complaintIsDue(1, 0)).toBe(true);
  });

  it("stays quiet for the rest of the interval", () => {
    expect(complaintIsDue(2, 3_000)).toBe(false);
    expect(complaintIsDue(10, 29_999)).toBe(false);
  });

  it("speaks up again once the interval is over, however long the run", () => {
    expect(complaintIsDue(2, 30_000)).toBe(true);
    expect(complaintIsDue(600, 600_000)).toBe(true);
  });
});
