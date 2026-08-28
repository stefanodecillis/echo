import { describe, expect, it } from "vitest";

import type { Id, Job, JobKind, JobStatus } from "../lib/types";
import { applyJobAnnouncement, groupByMeeting, sameMaps, sameWork } from "./useActiveJobs";

/**
 * The pure half of the outstanding-work map: how one announcement folds in, and
 * what counts as a change worth re-rendering for.
 *
 * Two of these are contracts nothing else can check. The reference test is the
 * reason the per-meeting selectors are worth having at all — break it and the
 * app still looks right while re-rendering every row in a five-hundred-row list
 * four times a second. And the wholesale-replacement test stands in for the one
 * thing the event stream cannot promise: the core's terminal announcement is
 * best-effort, so a job can leave the queue without a word.
 */

interface JobOverrides {
  id?: string;
  /** Explicitly `undefined` means a job that belongs to no meeting — which is a
   * case under test, so it cannot be read as "not given". Hence the `in` check
   * below rather than a destructuring default, which would swallow it. */
  meetingId?: Id;
  progress?: number;
}

function job(kind: JobKind, status: JobStatus, overrides: JobOverrides = {}): Job {
  return {
    id: overrides.id ?? `${kind}-${status}`,
    kind,
    status,
    meetingId: "meetingId" in overrides ? overrides.meetingId : "m1",
    progress: overrides.progress,
    createdAt: "2026-08-25T10:00:00Z",
    updatedAt: "2026-08-25T10:00:00Z",
  };
}

describe("groupByMeeting", () => {
  it("drops the jobs that belong to no meeting", () => {
    // The one-time download and getting the engine ready are the corner pill's
    // business; a meeting row has nothing to say about either.
    const grouped = groupByMeeting([
      job("download", "running", { meetingId: undefined }),
      job("prepareEngine", "queued", { meetingId: undefined }),
      job("diarize", "queued"),
    ]);
    expect(Object.keys(grouped)).toEqual(["m1"]);
    expect(grouped.m1).toHaveLength(1);
  });

  it("buckets by meeting", () => {
    const grouped = groupByMeeting([
      job("diarize", "queued", { id: "a", meetingId: "m1" }),
      job("summarize", "queued", { id: "b", meetingId: "m2" }),
      job("mixdown", "queued", { id: "c", meetingId: "m1" }),
    ]);
    expect(grouped.m1.map((j) => j.id)).toEqual(["a", "c"]);
    expect(grouped.m2.map((j) => j.id)).toEqual(["b"]);
  });
});

describe("sameWork", () => {
  it("is unmoved by a fraction, which this map does not carry", () => {
    const before = [job("transcribeCatchup", "running", { progress: 0.1 })];
    const after = [job("transcribeCatchup", "running", { progress: 0.9 })];
    expect(sameWork(before, after)).toBe(true);
  });

  it("notices a status, a kind, a stage, an id or a length", () => {
    const base = [job("diarize", "queued")];
    expect(sameWork(base, [job("diarize", "running")])).toBe(false);
    expect(sameWork(base, [job("summarize", "queued")])).toBe(false);
    expect(sameWork(base, [{ ...base[0], phase: "preparingEngine" }])).toBe(false);
    expect(sameWork(base, [{ ...base[0], id: "other" }])).toBe(false);
    expect(sameWork(base, [...base, job("summarize", "queued")])).toBe(false);
    expect(sameWork(base, undefined)).toBe(false);
  });
});

describe("applyJobAnnouncement", () => {
  it("ignores a job with no meeting of its own", () => {
    const active = {};
    const failed = {};
    const out = applyJobAnnouncement(active, failed, job("download", "running", { meetingId: undefined }));
    expect(out.changed).toBe(false);
    expect(out.active).toBe(active);
    expect(out.failed).toBe(failed);
  });

  it("adds work that has just been queued", () => {
    const out = applyJobAnnouncement({}, {}, job("transcribeCatchup", "queued"));
    expect(out.changed).toBe(true);
    expect(out.active.m1.map((j) => j.status)).toEqual(["queued"]);
  });

  it("takes finished work out, and drops the bucket with it", () => {
    // The bucket has to go, not just empty out: "is anything outstanding" is a
    // question about keys.
    for (const ending of ["done", "cancelled"] as const) {
      const active = { m1: [job("diarize", "running", { id: "j1" })] };
      const out = applyJobAnnouncement(active, {}, job("diarize", ending, { id: "j1" }));
      expect(out.changed).toBe(true);
      expect(out.active).toEqual({});
    }
  });

  it("moves a failure out of the work map and into the other one", () => {
    const active = { m1: [job("diarize", "running", { id: "j1" })] };
    const out = applyJobAnnouncement(active, {}, job("diarize", "failed", { id: "j1" }));
    expect(out.changed).toBe(true);
    expect(out.active).toEqual({});
    expect(out.failed.m1.map((j) => j.id)).toEqual(["j1"]);
  });

  it("un-fails a row that has been put back in the queue", () => {
    // `retry` is the one way an id can legitimately appear in both maps. It must
    // not stay in both, or the row would report a failure while working.
    const failed = { m1: [job("diarize", "failed", { id: "j1" })] };
    const out = applyJobAnnouncement({}, failed, job("diarize", "queued", { id: "j1" }));
    expect(out.changed).toBe(true);
    expect(out.active.m1.map((j) => j.id)).toEqual(["j1"]);
    expect(out.failed).toEqual({});
  });

  it("says nothing changed when only the fraction moved, and hands back what it was given", () => {
    // This is what keeps a download from rewriting the store four times a second
    // for a map that draws no bar.
    const active = { m1: [job("transcribeCatchup", "running", { id: "j1", progress: 0.2 })] };
    const failed = {};
    const out = applyJobAnnouncement(
      active,
      failed,
      job("transcribeCatchup", "running", { id: "j1", progress: 0.7 }),
    );
    expect(out.changed).toBe(false);
    expect(out.active).toBe(active);
    expect(out.failed).toBe(failed);
  });

  it("leaves every other meeting's bucket at the reference it came in with", () => {
    // The perf contract behind the per-meeting selectors, and invisible
    // otherwise: a job moving on one meeting must not re-render the rows of all
    // the others.
    const untouched = [job("summarize", "queued", { id: "j9", meetingId: "m2" })];
    const active = { m1: [job("diarize", "queued", { id: "j1" })], m2: untouched };
    const out = applyJobAnnouncement(active, {}, job("diarize", "running", { id: "j1" }));
    expect(out.changed).toBe(true);
    expect(out.active.m2).toBe(untouched);
    expect(out.active.m1).not.toBe(active.m1);
  });
});

describe("sameMaps", () => {
  it("lets a wholesale re-read drop a job the events never said goodbye to", () => {
    // The core announces a terminal status best-effort — it re-reads the row and
    // announces it *if that read succeeds*. When it does not, this is the only
    // thing that clears the job, which is why the read replaces the map instead
    // of merging into it.
    const held = groupByMeeting([job("diarize", "running", { id: "ghost" })]);
    const fresh = groupByMeeting([]);
    expect(sameMaps(held, fresh)).toBe(false);
    expect(Object.keys(fresh)).toEqual([]);
  });

  it("recognises an unchanged answer, so an idle app stays still", () => {
    const jobs = [job("diarize", "queued", { id: "j1" })];
    expect(sameMaps(groupByMeeting(jobs), groupByMeeting(jobs))).toBe(true);
    expect(sameMaps({}, {})).toBe(true);
  });
});

describe("an announcement that carries nothing", () => {
  it("does not declare the maps read", () => {
    // Launching Echo during the one-time download used to do exactly that: the
    // download announces itself four times a second and belongs to no meeting,
    // so every tick wrote "read, and empty" and dropped the opening read with
    // it. `changed` staying false is what the hook checks before it writes, so
    // it is what this pins.
    const out = applyJobAnnouncement({}, {}, job("download", "running", { meetingId: undefined }));
    expect(out.changed).toBe(false);
  });

  it("does not declare them read for a fraction that moved either", () => {
    const active = { m1: [job("summarize", "running", { id: "j1", progress: 0.1 })] };
    const out = applyJobAnnouncement(active, {}, job("summarize", "running", { id: "j1", progress: 0.6 }));
    expect(out.changed).toBe(false);
  });
});
