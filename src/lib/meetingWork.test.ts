import { describe, expect, it } from "vitest";

import { meetingWork } from "./meetingWork";
import type { Job, JobKind, JobStatus, MeetingStatus } from "./types";

/**
 * The decision table in `meetingWork`, one case at a time — plus the handful of
 * cases that are the whole reason it is a table rather than a chain of `if`s.
 *
 * Three of them describe a real window in which a plainer version says something
 * false: the instant before a stopped meeting's work is enqueued, a recap asked
 * for by hand on a meeting that finished last week, and a failure left over from
 * before a successful "Listen again".
 */

const MEETING = "m1";

function job(
  kind: JobKind,
  status: JobStatus,
  { id = `${kind}-${status}`, meetingId = MEETING, createdAt = "2026-08-25T10:00:00Z" } = {},
): Job {
  return { id, kind, status, meetingId, createdAt, updatedAt: createdAt };
}

function meeting(status: MeetingStatus) {
  return { id: MEETING, status };
}

describe("meetingWork", () => {
  it("says nothing at all until the first read has landed", () => {
    // The first-paint guarantee. `undefined` is "nobody has asked yet", which is
    // not "nothing is happening" — and a row that cannot tell them apart would
    // pick one and be wrong about half the time.
    expect(meetingWork(meeting("processing"), undefined, [job("diarize", "failed")])).toEqual({
      state: "none",
    });
  });

  it("leaves a recording alone", () => {
    // The Live screen, the rail's live card and the tray already narrate this.
    expect(meetingWork(meeting("recording"), [job("transcribeCatchup", "running")], [])).toEqual({
      state: "none",
    });
  });

  it("names what is running, whatever order the list arrives in", () => {
    // The 2026-08-21 shape: `listJobs` comes back newest first, and a meeting
    // that has just ended queues its work in the order it will run, so the last
    // job to run sits at the front of the list while the first one works.
    const jobs = [
      job("summarize", "queued"),
      job("mixdown", "queued"),
      job("diarize", "queued"),
      job("transcribeCatchup", "running"),
    ];
    const work = meetingWork(meeting("processing"), jobs, []);
    expect(work.state).toBe("running");
    expect(work.state === "running" && work.job.kind).toBe("transcribeCatchup");
  });

  it("counts work parked for a recording as parked, never as waiting", () => {
    // Starting a capture parks every outstanding row at once, so a mixture of
    // the two exists only for the instant `release` takes — and through it,
    // "nothing is moving" is the truer of the two things to say.
    expect(meetingWork(meeting("processing"), [job("diarize", "paused")], [])).toEqual({
      state: "deferred",
    });
    expect(
      meetingWork(meeting("processing"), [job("summarize", "queued"), job("diarize", "paused")], []),
    ).toEqual({ state: "deferred" });
  });

  it("says a queued meeting is waiting", () => {
    expect(meetingWork(meeting("processing"), [job("transcribeCatchup", "queued")], [])).toEqual({
      state: "waiting",
    });
  });

  it("owns up to a meeting whose last pass stopped without finishing", () => {
    // `settle_meeting` only ever runs for work that finished, so this meeting
    // stays marked as being worked on for good with nothing working on it.
    const work = meetingWork(meeting("processing"), [], [job("diarize", "failed")]);
    expect(work.state).toBe("stopped");
    expect(work.state === "stopped" && work.job.kind).toBe("diarize");
  });

  it("does not cry failure in the moment before the work is queued", () => {
    // Stop sets the meeting to processing, and the four jobs are written a beat
    // later. That beat must not flash "didn't finish".
    expect(meetingWork(meeting("processing"), [], [])).toEqual({ state: "unknown" });
  });

  it("still says a job is running on a meeting that had already finished", () => {
    // Asking for a recap by hand queues the work and leaves the meeting
    // `complete` — nothing in the summarize path puts it back to `processing`.
    // Gating the running state on the meeting's status would hide exactly the
    // thing somebody wants confirmed: the button they just pressed did something.
    const work = meetingWork(meeting("complete"), [job("summarize", "running")], []);
    expect(work.state).toBe("running");
    expect(work.state === "running" && work.job.kind).toBe("summarize");
  });

  it("ignores a failure the meeting has since settled past", () => {
    // A failed row from before a successful "Listen again" is suppressed by the
    // meeting's own status, which is what lets the failure map be a partial
    // history rather than the whole of one.
    expect(meetingWork(meeting("complete"), [], [job("diarize", "failed")])).toEqual({
      state: "none",
    });
  });

  it("never lets another meeting's work leak into this row", () => {
    const others = [
      job("transcribeCatchup", "running", { meetingId: "m2" }),
      job("summarize", "queued", { meetingId: "m2" }),
    ];
    expect(meetingWork(meeting("complete"), others, [])).toEqual({ state: "none" });
    expect(
      meetingWork(meeting("processing"), others, [job("diarize", "failed", { meetingId: "m2" })]),
    ).toEqual({ state: "unknown" });
  });

  it("has nothing to say about a finished meeting", () => {
    expect(meetingWork(meeting("complete"), [], [])).toEqual({ state: "none" });
    expect(meetingWork(meeting("created"), [], [])).toEqual({ state: "none" });
  });
});
