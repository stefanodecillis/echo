import { describe, expect, it } from "vitest";

import { correctedSpans } from "./corrections";

/**
 * These cases are the mirror of the ones `asr::glossary::revert` is tested
 * against, and that is the point of them: this function decides which words the
 * transcript offers as "put this back", and the command behind that offer
 * refuses any line whose repairs it cannot place the same way. A word marked
 * here that the core would not place is an undo that always fails.
 */
describe("correctedSpans", () => {
  it("cuts the repaired word out of the line", () => {
    expect(correctedSpans("We talked about Langola today", [{ from: "lana gola", to: "Langola" }]))
      .toEqual([
        { text: "We talked about ", corrected: false },
        { text: "Langola", corrected: true },
        { text: " today", corrected: false },
      ]);
  });

  it("places several repairs left to right", () => {
    const spans = correctedSpans("Langola and Obsidara", [
      { from: "Nongula", to: "Langola" },
      { from: "Obsidera", to: "Obsidara" },
    ]);
    expect(spans).toEqual([
      { text: "Langola", corrected: true },
      { text: " and ", corrected: false },
      { text: "Obsidara", corrected: true },
    ]);
  });

  it("keeps the punctuation that hangs off a repaired word outside it", () => {
    expect(correctedSpans("«Langola».", [{ from: "lana gola", to: "Langola" }])).toEqual([
      { text: "«", corrected: false },
      { text: "Langola", corrected: true },
      { text: "».", corrected: false },
    ]);
  });

  it("does not point at a word buried inside a longer one", () => {
    // "Langolas" is not the word, so the repair is nowhere in this line: some
    // later pass rewrote it, and nothing here can say where the repair was.
    expect(correctedSpans("Langolas everywhere", [{ from: "lana gola", to: "Langola" }])).toBeNull();
  });

  it("refuses a line where the repaired word turns up more often than it was repaired", () => {
    // The ambiguity the core fails closed on: if the decoder wrote "Langola"
    // itself in the same sentence Echo repaired one into, nothing stored says
    // which of the two is the repair, and guessing is a coin toss over a word
    // somebody really said.
    expect(
      correctedSpans("Langola, really Langola", [{ from: "Nongula", to: "Langola" }]),
    ).toBeNull();
  });

  it("accepts the same word repaired twice, once for each occurrence", () => {
    const spans = correctedSpans("Langola and Langola", [
      { from: "Nongula", to: "Langola" },
      { from: "lana gola", to: "Langola" },
    ]);
    expect(spans).toEqual([
      { text: "Langola", corrected: true },
      { text: " and ", corrected: false },
      { text: "Langola", corrected: true },
    ]);
  });

  it("refuses a line the repair has gone from altogether", () => {
    expect(correctedSpans("Nothing like it here", [{ from: "x", to: "Langola" }])).toBeNull();
  });

  it("has nothing to offer on a line that was never repaired", () => {
    expect(correctedSpans("Plain words", [])).toBeNull();
  });

  it("refuses an empty replacement rather than marking the whole line", () => {
    expect(correctedSpans("Plain words", [{ from: "x", to: "" }])).toBeNull();
  });

  it("marks a repair that is the whole of the line", () => {
    expect(correctedSpans("Langola", [{ from: "Nongula", to: "Langola" }])).toEqual([
      { text: "Langola", corrected: true },
    ]);
  });

  it("treats accented letters as letters on both sides of a word", () => {
    expect(correctedSpans("èLangola", [{ from: "x", to: "Langola" }])).toBeNull();
    expect(correctedSpans("è Langola", [{ from: "x", to: "Langola" }])).toEqual([
      { text: "è ", corrected: false },
      { text: "Langola", corrected: true },
    ]);
  });
});
