import { describe, expect, it } from "vitest";

import { leftOut, meeting } from "./copy";

/**
 * The sentence Echo says about moments it left out of a transcript.
 *
 * It is the one place in the app that admits a mechanism removed something a
 * person may have said, so it has to be true, plain, and never machinery. The
 * Rust side (`asr/left_out.rs`) decides which moments reach it; these are the
 * words.
 */
describe("what Echo says about the moments it left out", () => {
  it("counts one moment as one, and says the number for the rest", () => {
    expect(leftOut.title(1)).toBe("One moment has no words");
    expect(leftOut.title(3)).toBe("3 moments have no words");
    expect(leftOut.explanation(1)).toContain("at that moment either");
    expect(leftOut.explanation(4)).toContain("at these times");
  });

  it("says both halves of why those seconds are silent", () => {
    for (const count of [1, 5]) {
      const said = leftOut.explanation(count);
      // What Echo did with it…
      expect(said).toContain("through the microphone");
      expect(said).toContain("this computer's own sound coming back");
      // …and that nothing else covered those seconds, which is the whole
      // reason this is worth saying at all.
      expect(said).toContain("Nothing else was written down");
    }
  });

  it("points at the repair by the name of the button that does it", () => {
    expect(leftOut.repair(meeting.listenAgainButton)).toContain("“Listen again”");
    expect(leftOut.repair(meeting.listenAgainButton)).toContain("reads the whole recording again");
  });

  it("names no machinery anywhere", () => {
    const everything = [
      leftOut.title(1),
      leftOut.title(2),
      leftOut.explanation(1),
      leftOut.explanation(2),
      leftOut.repair(meeting.listenAgainButton),
      leftOut.jumpLabel("12:04"),
      leftOut.andMore(3),
    ].join(" ");
    for (const jargon of [
      "bleed",
      "correlation",
      "channel",
      "span",
      "buffer",
      "stream",
      "segment",
      "suppress",
      "audio",
    ]) {
      expect(everything.toLowerCase()).not.toContain(jargon);
    }
  });
});
