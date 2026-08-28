import { describe, expect, it } from "vitest";

import { jobLine, labels, leftOut, meeting, nav, update, workChip } from "./copy";

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

/**
 * The words a meeting row uses for work Echo has not finished.
 *
 * The row is the most cramped place in the app that has to say something true
 * about a state, and the four things it can say have to be tellable apart by
 * somebody reading them — not merely by the code choosing between them.
 */
describe("what a meeting row says about unfinished work", () => {
  const label = labels.jobKind.diarize;

  it("keeps the four states in different words", () => {
    const said = [
      jobLine.running(label),
      workChip.waiting,
      workChip.deferred,
      jobLine.stopped(label),
    ];
    expect(new Set(said).size).toBe(said.length);
  });

  it("says of the running one what is happening, and of the others that it is not", () => {
    // Which pass is running is the whole thing somebody wants from a glance, so
    // that one names itself; the two still ones do not, because nothing is
    // happening and the name would be the only moving part of the sentence.
    expect(jobLine.running(label)).toContain(label);
    expect(jobLine.stopped(label)).toContain(label);
    expect(workChip.waiting).not.toContain(label);
    expect(workChip.deferred).not.toContain(label);
  });

  it("counts nothing", () => {
    // The mechanical form of the rule at the top of copy.ts: sizes and times are
    // fine, counters and queue depths are not. This is what catches the
    // well-meaning future edit that appends "(2 ahead)".
    for (const said of [workChip.waiting, workChip.deferred, ...Object.values(labels.meetingStatus)]) {
      expect(said).not.toMatch(/\d/);
    }
  });

  it("names no machinery, and nothing it says is empty", () => {
    const everything = [
      ...Object.values(labels.meetingStatus),
      workChip.waiting,
      workChip.deferred,
      nav.workingLabel,
    ];
    for (const said of everything) {
      expect(said.trim().length).toBeGreaterThan(0);
    }
    const joined = everything.join(" ").toLowerCase();
    for (const jargon of [
      "job",
      "queue",
      "pending",
      "diariz",
      "transcode",
      "worker",
      "task",
      "process ",
      "status",
    ]) {
      expect(joined).not.toContain(jargon);
    }
  });

  it("agrees with the card Home already shows above the list", () => {
    // Two treatments of the same state in one screen have to use one sentence.
    expect(labels.meetingStatus.interrupted).toBe("Didn't finish");
    expect(meeting.infoWorkingTitle).toBe(nav.workingLabel);
  });
});

/**
 * What Echo says when a newer version of itself is waiting.
 *
 * A pill somebody reads once and acts on, so it has to be true, plain, and
 * silent about machinery — where the version came from, what it is called, and
 * how big it was are all none of the reader's business.
 */
describe("what Echo says about a new version", () => {
  it("says what is waiting and what pressing it costs", () => {
    expect(update.readyTitle).toContain("new version");
    expect(update.readyAction.toLowerCase()).toContain("restart");
    // The honest half: restarting means paying for the speech setup again,
    // because the compiled model is keyed to the app that asked for it. A person
    // who is told beforehand can pick their moment.
    expect(update.readySetupHint.toLowerCase()).toContain("getting ready");
  });

  it("promises to wait rather than offering something it will refuse", () => {
    expect(update.waitingForMeeting.toLowerCase()).toContain("meeting");
    expect(update.waitingForMeeting.toLowerCase()).not.toContain("click");
  });

  it("names no machinery and no numbers", () => {
    const everything = Object.values(update);
    for (const said of everything) {
      expect(said.trim().length).toBeGreaterThan(0);
      // No version strings, no sizes, no percentages.
      expect(said).not.toMatch(/\d/);
    }
    const joined = everything.join(" ").toLowerCase();
    for (const jargon of [
      "github",
      "download",
      "bundle",
      "signature",
      "artifact",
      "binary",
      "install",
      "updater",
      "release",
    ]) {
      expect(joined).not.toContain(jargon);
    }
  });
});
