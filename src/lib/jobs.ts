import { labels } from "./copy";
import type { Job, JobKind, JobPhase } from "./types";

/**
 * Which background job a screen should be showing, and what it should say
 * about it.
 *
 * WHY THIS EXISTS
 * Every in-progress treatment in the app used to pick its job with
 * `jobs.find(j => j.status === "running" || j.status === "queued")` — which
 * returns whichever of the two the list happens to hold first. The list comes
 * back newest first, and a meeting that has just ended queues its work in the
 * order it will run: catch-up, then speakers, then playback, then the recap.
 * So "the first active job" was reliably the *last* one to run, and the screen
 * named a job that had not started while a different one worked.
 *
 * On 2026-08-21 that cost somebody eighteen minutes: the Meeting page said
 * "Catching up on the transcript…" over a moving bar for the whole of it, while
 * what was actually happening was the one-time setup finishing — which nothing
 * on that screen mentioned, and which the catch-up job had not even begun
 * waiting on yet. Two lies in one box: the wrong name, and a treatment that
 * said "in progress" about work that had not started.
 *
 * THE RULE
 * In-progress treatment names what is running. Queued work may be listed, but
 * it is never dressed up as happening — it says it is waiting, and it never
 * gets a progress bar, because a job that has not started has made no
 * progress. A job that *is* running shows its fraction when it reports one, and
 * shows an unmeasured bar when it honestly has none.
 */

/** Work that has not finished: running now, waiting its turn, or parked. */
export function isActive(job: Job): boolean {
  return job.status === "running" || job.status === "queued" || job.status === "paused";
}

/**
 * The job to put in front of a person: the one that is running, and only
 * failing that the one that is waiting.
 *
 * `kinds` narrows it to the jobs a particular screen is about — the Transcript
 * tab cares about the transcript being rewritten, not about the recap.
 */
export function inProgressJob(jobs: Job[], kinds?: ReadonlySet<string>): Job | undefined {
  const mine = kinds ? jobs.filter((job) => kinds.has(job.kind)) : jobs;
  return (
    mine.find((job) => job.status === "running") ??
    mine.find((job) => job.status === "queued" || job.status === "paused")
  );
}

/**
 * The failure a screen still has to own up to, or `undefined` when there isn't
 * one.
 *
 * A job row survives its own failure, and "Listen again" queues a fresh job
 * beside the old one rather than replacing it — so "any job that failed" would
 * pin a months-old failure to a transcript that has been rewritten twice since.
 * Only the newest job of each kind speaks for that kind: if *that* one failed,
 * nothing since has done the work, and the person is looking at a transcript
 * missing whatever it was going to add.
 *
 * This exists because the failure that matters most here is the quietest one.
 * The speaker files are allowed to arrive after the first recording, so a
 * meeting can be recorded before Echo can tell voices apart; the pass then fails
 * with a perfectly clear sentence that, until now, nothing on any screen showed
 * — while the mic-only banner had already promised the person that Echo would
 * work out who said what once the meeting ended.
 */
export function lastFailure(jobs: Job[], kinds?: ReadonlySet<string>): Job | undefined {
  const mine = kinds ? jobs.filter((job) => kinds.has(job.kind)) : jobs;
  const at = (stamp: string) => Date.parse(stamp) || 0;
  const newestOfEachKind = new Map<string, Job>();
  for (const job of mine) {
    const seen = newestOfEachKind.get(job.kind);
    if (!seen || at(job.createdAt) >= at(seen.createdAt)) newestOfEachKind.set(job.kind, job);
  }
  return [...newestOfEachKind.values()]
    .filter((job) => job.status === "failed")
    .sort((a, b) => at(b.updatedAt) - at(a.updatedAt))[0];
}

/** Running jobs first, then the ones waiting — for a screen that lists them. */
export function runningFirst(jobs: Job[]): Job[] {
  return [
    ...jobs.filter((job) => job.status === "running"),
    ...jobs.filter((job) => job.status !== "running"),
  ];
}

/** What one job looks like on screen right now. */
export interface JobPresentation {
  /** The sentence, with no trailing ellipsis: callers add one where it fits. */
  label: string;
  /** 0..1 when the job reports one, `undefined` for an honest "no idea". */
  fraction?: number;
  /** True while this is actually happening, false while it waits. */
  running: boolean;
}

/**
 * How to show one job: what it is called, how far along it is, and whether it
 * has started.
 *
 * A stage overrides the job's own name, because the stage is the truer
 * description of the moment: the download's second half is not downloading, and
 * a catch-up waiting for the engine to be got ready is not yet catching up on
 * anything. A queued job keeps its name and loses its bar.
 */
export function presentJob(job: Job, phase: JobPhase | undefined = job.phase): JobPresentation {
  const running = job.status === "running";
  const label = phase ? labels.jobPhase[phase] : labels.jobKind[job.kind as JobKind];
  return {
    label,
    fraction: running ? job.progress : undefined,
    running,
  };
}
