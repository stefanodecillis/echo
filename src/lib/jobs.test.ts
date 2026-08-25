import { describe, expect, it } from "vitest";

import { inProgressJob, lastFailure } from "./jobs";
import type { Job, JobKind, JobStatus } from "./types";

/**
 * Both functions here were written to fix a specific wrong screen, and both
 * bugs are the kind that looks fine in every hand-test: the list has to arrive
 * in a particular order, or hold a particular pair of rows, before the old code
 * says the wrong thing. So the order and the pairs are what these tests are
 * about.
 */

/** One row as `listJobs` hands it over, with only the fields these two
 * functions read. */
function job(
  kind: JobKind,
  status: JobStatus,
  {
    id = `${kind}-${status}`,
    createdAt = "2026-08-25T10:00:00Z",
    updatedAt,
  }: { id?: string; createdAt?: string; updatedAt?: string } = {},
): Job {
  return { id, kind, status, createdAt, updatedAt: updatedAt ?? createdAt };
}

describe("inProgressJob", () => {
  it("names the job that is running, not the one at the front of the list", () => {
    // The 2026-08-21 shape exactly: the list comes back newest first, and a
    // meeting that has just ended queues its work in the order it will run, so
    // the *last* job to run sits at the front while the first one works.
    const jobs = [
      job("summarize", "queued"),
      job("mixdown", "queued"),
      job("diarize", "queued"),
      job("transcribeCatchup", "running"),
    ];
    expect(inProgressJob(jobs)?.kind).toBe("transcribeCatchup");
  });

  it("falls back to the job that is waiting when nothing is running", () => {
    const jobs = [job("summarize", "queued"), job("diarize", "queued")];
    expect(inProgressJob(jobs)?.kind).toBe("summarize");
  });

  it("counts a job parked for a recording as waiting, never as running", () => {
    const paused = inProgressJob([job("diarize", "paused")]);
    expect(paused?.status).toBe("paused");
    expect(inProgressJob([job("diarize", "paused"), job("summarize", "running")])?.kind).toBe(
      "summarize",
    );
  });

  it("only looks at the kinds the screen is about", () => {
    const jobs = [job("summarize", "running"), job("diarize", "queued")];
    const kinds = new Set(["transcribeCatchup", "diarize"]);
    expect(inProgressJob(jobs, kinds)?.kind).toBe("diarize");
  });

  it("has nothing to say about a list with no unfinished work", () => {
    expect(inProgressJob([job("diarize", "done"), job("summarize", "failed")])).toBeUndefined();
  });
});

describe("lastFailure", () => {
  it("stays quiet once a newer job of the same kind has been queued over it", () => {
    // "Listen again" queues a fresh job beside the old one rather than
    // replacing it. Only the newest of each kind speaks for that kind, or a
    // months-old failure stays pinned to a transcript rewritten twice since.
    const jobs = [
      job("diarize", "queued", { id: "new", createdAt: "2026-08-25T12:00:00Z" }),
      job("diarize", "failed", { id: "old", createdAt: "2026-06-01T09:00:00Z" }),
    ];
    expect(lastFailure(jobs)).toBeUndefined();
  });

  it("owns up to the failure when the newest job of that kind is the one that failed", () => {
    const jobs = [
      job("diarize", "failed", { id: "new", createdAt: "2026-08-25T12:00:00Z" }),
      job("diarize", "done", { id: "old", createdAt: "2026-06-01T09:00:00Z" }),
    ];
    expect(lastFailure(jobs)?.id).toBe("new");
  });

  it("does not depend on the order the list arrives in", () => {
    const newest = job("diarize", "failed", { id: "new", createdAt: "2026-08-25T12:00:00Z" });
    const older = job("diarize", "done", { id: "old", createdAt: "2026-06-01T09:00:00Z" });
    expect(lastFailure([newest, older])?.id).toBe("new");
    expect(lastFailure([older, newest])?.id).toBe("new");
  });

  it("prefers the most recently updated when two kinds have both failed", () => {
    const jobs = [
      job("transcribeCatchup", "failed", {
        createdAt: "2026-08-25T10:00:00Z",
        updatedAt: "2026-08-25T10:05:00Z",
      }),
      job("diarize", "failed", {
        createdAt: "2026-08-25T10:01:00Z",
        updatedAt: "2026-08-25T10:20:00Z",
      }),
    ];
    expect(lastFailure(jobs)?.kind).toBe("diarize");
  });

  it("ignores a failure of a kind this screen is not about", () => {
    const jobs = [job("summarize", "failed")];
    expect(lastFailure(jobs, new Set(["transcribeCatchup", "diarize"]))).toBeUndefined();
  });

  it("survives a timestamp it cannot read rather than throwing", () => {
    // `Date.parse` of a malformed stamp is NaN, which the || 0 in the code
    // turns into "oldest". Nothing here should ever be a crash on a screen.
    const jobs = [
      job("diarize", "failed", { id: "unparseable", createdAt: "not a date" }),
      job("diarize", "done", { id: "real", createdAt: "2026-08-25T12:00:00Z" }),
    ];
    expect(lastFailure(jobs)).toBeUndefined();
  });
});
