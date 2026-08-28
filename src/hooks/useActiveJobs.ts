import { useEffect, useMemo } from "react";

import { EVENTS, listJobs } from "../lib/ipc";
import { isActive } from "../lib/jobs";
import { meetingWork, type MeetingWork } from "../lib/meetingWork";
import { useEchoStore } from "../lib/store";
import type { Id, Job, MeetingSummary } from "../lib/types";
import { complaintIsDue } from "./useCaptureState";
import { useEvent } from "./useEvent";

/**
 * Which meetings Echo still has work outstanding on, kept fresh for the whole
 * app at once.
 *
 * WHY A SHARED MAP
 * Three things need this answer — every row in two different lists, the rail's
 * dot, and the corner pill — and one of those lists renders every meeting a
 * person has. A hook that fetched per consumer would ask the core once per row.
 * So it lives in the store (`src/lib/store.ts`), fed here, and read through
 * selectors narrow enough that a job moving re-renders the one row it belongs to
 * rather than five hundred.
 *
 * WHY THE FRACTION IS THROWN AWAY
 * A running job announces itself every 250ms. Nothing that reads this map draws
 * a progress bar — a list of moving bars is the noise docs/DESIGN.md §4 is
 * written against — so an announcement that only moved the fraction is dropped
 * before it reaches the store (`sameWork` below). Without that, a download would
 * rewrite state four times a second for no visible difference at all.
 *
 * WHY IT RE-READS INSTEAD OF TIMING OUT
 * The core's terminal announcement is best-effort: `execute` re-reads the row and
 * announces it *if that read succeeds* (src-tauri/src/session/jobs.rs). When it
 * doesn't, a job leaves the queue in total silence, and a client holding a map
 * built only from events holds that job forever. `useSetupJob` had to buy a
 * thirty-second staleness timeout for exactly this, because it follows one job by
 * id and has no cheap way to ask.
 *
 * This does have a cheap way to ask, so it asks: `listJobs({ activeOnly: true })`
 * is the authoritative set of unfinished work, and its answer *replaces* the map
 * rather than merging into it. Anything the client was holding that the core did
 * not name is a ghost, and replacement is the only thing that clears one. There
 * is no timeout anywhere in this file, and nothing here waits for an event that
 * may never come.
 */

/** Only the first mount anywhere in the tree does the opening read. Module
 * state, not a ref, because it has to survive across components. */
let didFetchInitial = false;

/**
 * How many times an announcement has actually changed the maps. Read before a
 * request goes out and checked again when it comes back.
 *
 * Same guard, and the same reason, as `stateSeq` in `useCaptureState`: the core
 * samples its answer when the call reaches it and delivers it whenever it gets
 * round to it, so a read fired a moment before a meeting is stopped comes back
 * describing a queue that has since grown. Replacing the map with that answer
 * would drop a job that really exists, until the next tick put it back.
 *
 * Counted on *changes*, deliberately, not on announcements. A download reports
 * itself four times a second and changes nothing here; if every one of those
 * bumped the number, no read would ever land while one was running, and the
 * ghosts this file exists to clear would survive the whole download.
 */
let workSeq = 0;

/**
 * How often the map is asked for again while anything is outstanding.
 *
 * Only while something is outstanding: an idle Echo has an empty map, so there
 * is no timer at all. Chosen against the core's own beats — its idle poll and
 * its reconcile pass are both 30s — so this can never be the slower half of a
 * lag somebody notices.
 */
const WORK_INTERVAL_MS = 15_000;

/** Subscribers currently watching outstanding work, and the one timer they
 * share — one poll for the whole app, however many components are up. */
let watchers = 0;
let watchTimer: number | undefined;
let anyWatchers = 0;

/** How many reads in a row have come back with nothing, and when the last
 * complaint about it went out. */
let failedReads = 0;
let complainedAtMs = 0;

/**
 * Two lists of work that would draw the same row.
 *
 * Progress is deliberately not in it, and neither is `updatedAt`: this map
 * answers *which* work, never *how far*. The fraction belongs to the screen that
 * draws a bar, and no screen that reads this map draws one.
 */
export function sameWork(a: Job[] | undefined, b: Job[] | undefined): boolean {
  if (a === b) return true;
  if (!a || !b || a.length !== b.length) return false;
  return a.every(
    (job, i) =>
      job.id === b[i].id &&
      job.status === b[i].status &&
      job.kind === b[i].kind &&
      job.phase === b[i].phase,
  );
}

/**
 * Jobs bucketed by the meeting they belong to.
 *
 * Jobs with no meeting are dropped: the one-time download and getting the engine
 * ready are the corner pill's business, and a meeting row has nothing to say
 * about either.
 */
export function groupByMeeting(jobs: Job[]): Record<Id, Job[]> {
  const out: Record<Id, Job[]> = {};
  for (const job of jobs) {
    if (!job.meetingId) continue;
    const bucket = out[job.meetingId];
    if (bucket) bucket.push(job);
    else out[job.meetingId] = [job];
  }
  return out;
}

/** One meeting's bucket with this job written into it, every other meeting's
 * bucket returned by the same reference it came in with. */
function withJob(map: Record<Id, Job[]>, id: Id, job: Job): Record<Id, Job[]> {
  const bucket = map[id];
  if (!bucket) return { ...map, [id]: [job] };
  const at = bucket.findIndex((held) => held.id === job.id);
  const next = at === -1 ? [...bucket, job] : bucket.map((held, i) => (i === at ? job : held));
  return { ...map, [id]: next };
}

/** The same, with the job taken out — and the bucket itself dropped once it is
 * empty, so "is anything outstanding" stays a question about keys. */
function withoutJob(map: Record<Id, Job[]>, id: Id, jobId: Id): Record<Id, Job[]> {
  const bucket = map[id];
  if (!bucket) return map;
  const next = bucket.filter((held) => held.id !== jobId);
  if (next.length === bucket.length) return map;
  if (next.length > 0) return { ...map, [id]: next };
  const rest = { ...map };
  delete rest[id];
  return rest;
}

/**
 * One announcement folded into the two maps.
 *
 * `changed` is false when nothing a screen can see is different, and the maps
 * come back by their original references so the caller can skip the write
 * entirely. Buckets belonging to untouched meetings always come back by
 * reference; that is what makes the per-meeting selectors worth having.
 */
export function applyJobAnnouncement(
  active: Record<Id, Job[]>,
  failed: Record<Id, Job[]>,
  job: Job,
): { active: Record<Id, Job[]>; failed: Record<Id, Job[]>; changed: boolean } {
  const id = job.meetingId;
  if (!id) return { active, failed, changed: false };

  let nextActive = active;
  let nextFailed = failed;

  if (isActive(job)) {
    nextActive = withJob(active, id, job);
    // `retry` puts a failed row back in the queue, which is the one way an id
    // can be in both maps at once. It must not be.
    nextFailed = withoutJob(failed, id, job.id);
  } else if (job.status === "failed") {
    nextActive = withoutJob(active, id, job.id);
    nextFailed = withJob(failed, id, job);
  } else {
    // Done or cancelled: it is neither work nor a reason there is none.
    nextActive = withoutJob(active, id, job.id);
    nextFailed = withoutJob(failed, id, job.id);
  }

  const changed =
    !sameWork(active[id], nextActive[id]) || !sameWork(failed[id], nextFailed[id]);
  return changed ? { active: nextActive, failed: nextFailed, changed } : { active, failed, changed };
}

/**
 * A read could not reach the core.
 *
 * Not told to the person: there is nothing they can do, and the screen still
 * holds the last thing that was true. Told to whoever is looking at the console,
 * with a count — one failed read is a blip the next tick fixes, and a number that
 * keeps climbing is a transport that has died, which is the case this read is the
 * floor under.
 */
function readFailed(err: unknown): void {
  failedReads += 1;
  const now = Date.now();
  if (!complaintIsDue(failedReads, now - complainedAtMs)) return;
  complainedAtMs = now;
  console.warn(`Echo: could not read what is still being worked on (${failedReads} in a row)`, err);
}

function readWorked(): void {
  if (failedReads === 0) return;
  const failed = failedReads;
  failedReads = 0;
  console.info(`Echo: reading outstanding work works again, after ${failed} that did not`);
}

/** Ask what is unfinished and replace the map with the answer. */
function refreshActiveWork(): void {
  const asked = workSeq;
  listJobs({ activeOnly: true })
    .then((jobs) => {
      readWorked();
      // An announcement that landed while this was in the air knows more about
      // the queue than an answer worked out before it.
      if (workSeq !== asked) return;
      const store = useEchoStore.getState();
      const next = groupByMeeting(jobs);
      // An idle app answering "still nothing" must not re-render every row.
      if (store.activeJobsByMeeting !== undefined && sameMaps(store.activeJobsByMeeting, next))
        return;
      store.setJobWork({ active: next });
    })
    .catch(readFailed);
}

/** Ask which meetings carry a failure. Rare, never urgent, so this is only ever
 * driven by coming back to the window or by a meeting settling — never by the
 * timer. */
function refreshFailedWork(): void {
  listJobs({ status: "failed", limit: 100 })
    .then((jobs) => {
      const store = useEchoStore.getState();
      const next = groupByMeeting(jobs);
      if (store.failedJobsByMeeting !== undefined && sameMaps(store.failedJobsByMeeting, next))
        return;
      store.setJobWork({ failed: next });
    })
    .catch(readFailed);
}

/** Two maps that would draw the same lists. */
export function sameMaps(a: Record<Id, Job[]>, b: Record<Id, Job[]>): boolean {
  const keys = Object.keys(a);
  if (keys.length !== Object.keys(b).length) return false;
  return keys.every((key) => sameWork(a[key], b[key]));
}

function refreshIfVisible(): void {
  if (document.visibilityState !== "visible") return;
  refreshActiveWork();
}

function refreshBoth(): void {
  refreshActiveWork();
  refreshFailedWork();
}

function refreshBothIfVisible(): void {
  if (document.visibilityState !== "visible") return;
  refreshBoth();
}

function syncWatchTimer(): void {
  const wanted = watchers > 0;
  if (wanted === (watchTimer !== undefined)) return;
  if (watchTimer !== undefined) {
    window.clearInterval(watchTimer);
    watchTimer = undefined;
    return;
  }
  watchTimer = window.setInterval(refreshIfVisible, WORK_INTERVAL_MS);
}

/** Coming back to Echo is when its screens are most likely to be stale and the
 * moment somebody is looking. It is also the only thing that repopulates these
 * maps after a transport that stopped delivering altogether. */
function watchWindowReturns(): () => void {
  anyWatchers += 1;
  if (anyWatchers === 1) {
    window.addEventListener("focus", refreshBoth);
    document.addEventListener("visibilitychange", refreshBothIfVisible);
  }
  return () => {
    anyWatchers -= 1;
    if (anyWatchers === 0) {
      window.removeEventListener("focus", refreshBoth);
      document.removeEventListener("visibilitychange", refreshBothIfVisible);
    }
  };
}

/**
 * Keeps the outstanding-work maps fresh. Mount once, high in the tree — this is
 * the only thing in the app that writes them, and everything else reads.
 *
 * Every reader must still render correctly with the maps unread, because until
 * the first answer lands that is exactly what they are.
 */
export function useActiveJobs(): void {
  const hasWork = useHasOutstandingWork();

  useEffect(() => {
    if (didFetchInitial) return;
    didFetchInitial = true;
    refreshBoth();
  }, []);

  useEvent(EVENTS.jobProgress, (payload) => {
    const store = useEchoStore.getState();
    const { active, failed, changed } = applyJobAnnouncement(
      store.activeJobsByMeeting ?? {},
      store.failedJobsByMeeting ?? {},
      payload.job,
    );
    // Nothing a screen can see moved — most often a running job's fraction,
    // which this map does not carry, or a download, which has no meeting to
    // belong to.
    //
    // Returning here even while the maps are still unread is the whole point,
    // and it was a bug worth keeping a note about: an announcement that carries
    // nothing must not be the thing that declares them read, and must not bump
    // the number below. Otherwise launching Echo during the one-time download —
    // which announces itself four times a second and belongs to no meeting —
    // would set both maps to "read, and empty" and drop the opening read on
    // every single tick, for the whole of a download that can take a quarter of
    // an hour. The app would spend it insisting nothing was outstanding.
    if (!changed) return;
    // The core said so just now, which beats any read still waiting to be heard
    // back from. A read overtaken this way is dropped rather than merged, and
    // the next beat re-asks — the same trade `useCaptureState` makes.
    workSeq += 1;
    store.setJobWork({ active, failed });
  });

  // `settle_meeting` emits this at exactly the moment a meeting's last job
  // finished, so it is a free and exact "the map should shrink now" — and it is
  // what makes the common case instant even when the terminal announcement was
  // the read that failed.
  useEvent(EVENTS.meetingUpdated, refreshBoth);

  useEffect(watchWindowReturns, []);

  useEffect(() => {
    if (!hasWork) return;
    watchers += 1;
    syncWatchTimer();
    return () => {
      watchers -= 1;
      syncWatchTimer();
    };
  }, [hasWork]);
}

/** What one meeting's row should say. Selects only that meeting's own work, so a
 * job moving re-renders its row and nobody else's. */
export function useMeetingWork(meeting: Pick<MeetingSummary, "id" | "status">): MeetingWork {
  const known = useEchoStore((s) => s.activeJobsByMeeting !== undefined);
  const active = useEchoStore((s) => s.activeJobsByMeeting?.[meeting.id]);
  const failed = useEchoStore((s) => s.failedJobsByMeeting?.[meeting.id]);
  const { id, status } = meeting;
  return useMemo(
    () => meetingWork({ id, status }, known ? (active ?? []) : undefined, failed),
    [id, status, known, active, failed],
  );
}

/** Whether any meeting has work outstanding — running, waiting or parked. The
 * rail's dot, which is the one indicator that still says something while a
 * recording has everything parked and nothing is running. */
export function useHasOutstandingWork(): boolean {
  return useEchoStore((s) => {
    const map = s.activeJobsByMeeting;
    if (!map) return false;
    return Object.values(map).some((bucket) => bucket.length > 0);
  });
}

/** The job being worked on right now, if it belongs to a meeting. The core runs
 * one at a time, so there is never more than one to choose between. */
export function useRunningWork(): Job | undefined {
  return useEchoStore((s) => {
    const map = s.activeJobsByMeeting;
    if (!map) return undefined;
    for (const bucket of Object.values(map)) {
      const running = bucket.find((job) => job.status === "running");
      if (running) return running;
    }
    return undefined;
  });
}
