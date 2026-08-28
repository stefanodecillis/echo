import { describe, expect, it } from "vitest";

import { pickCorner } from "./statusCorner";

/**
 * The corner holds one thing. These pin the order it picks in, because the
 * order is a product decision and the code that implements it is three lines
 * anybody could reorder without noticing.
 */
describe("pickCorner", () => {
  it("gives the corner to setup over everything", () => {
    // Setup is the only claimant that blocks recording: until it finishes a
    // meeting gets audio but no words.
    expect(pickCorner({ setup: true, update: true, work: true })).toBe("setup");
    expect(pickCorner({ setup: true, update: false, work: true })).toBe("setup");
  });

  it("puts a waiting update above a meeting being finished off", () => {
    // Persistent and actionable beats transient and informational — and the work
    // is already said on every affected row and by the rail's dot.
    expect(pickCorner({ setup: false, update: true, work: true })).toBe("update");
  });

  it("shows the work only when it is the only thing left to say", () => {
    expect(pickCorner({ setup: false, update: false, work: true })).toBe("work");
  });

  it("says nothing when there is nothing to say", () => {
    expect(pickCorner({ setup: false, update: false, work: false })).toBe("none");
  });

  it("never lets setup and an update narrate at once", () => {
    // They would say overlapping things: restarting for an update is what causes
    // the next setup, because the compiled speech model is keyed to the app that
    // asked for it.
    expect(pickCorner({ setup: true, update: true, work: false })).not.toBe("update");
  });
});
