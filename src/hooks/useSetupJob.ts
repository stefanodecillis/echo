import { useEffect, useRef, useState } from "react";

import { EVENTS, listJobs } from "../lib/ipc";
import { SETUP_KINDS, SETUP_PHASE, setupJobUnderWay } from "../lib/jobs";
import type { JobPhase } from "../lib/types";
import { useEvent } from "./useEvent";

/** Where the one-time setup has got to, as the job itself reports it. */
export interface SetupJobState {
  /** True while the job is running. */
  running: boolean;
  /** 0..1 while there is an honest fraction; absent once there is not. */
  fraction?: number;
  /** The stage it is in, when it is in one worth naming. */
  phase?: JobPhase;
}

/**
 * How long a followed job has to have said nothing before another setup job is
 * allowed to take the pill from it.
 *
 * A download reports its fraction constantly, so it is never near this. The one
 * job that is legitimately silent for a long time is the engine being got ready
 * — and the core runs one job at a time, so while that is happening there is
 * nothing else to hear from anyway.
 *
 * What this guards against is the followed job dying without a word: a lost
 * event, or the database read that the terminal announcement is made from
 * failing. Without it the pill sits there saying "Getting Echo ready" for the
 * rest of the session and swallows every later setup job with it, which is worse
 * than what it replaced.
 */
const GONE_QUIET_MS = 30_000;

/**
 * Follows the one-time setup job — the download and, after it, getting this
 * computer ready to use what arrived.
 *
 * Why not just the download events: they stop at "done" when the last byte
 * lands, and on Apple silicon the job then spends minutes — up to eighteen, on
 * the machine this was measured on — preparing the engine for this machine. The
 * corner pill used to vanish at that moment, leaving the longest part of setup
 * invisible and the person to find some other, wronger explanation on whatever
 * screen they were on (field report of 2026-08-21). The job's own events cover
 * the whole of it, phase included, and end with a real terminal status.
 *
 * Why kind *or* stage: weights can land without a download job ever running (a
 * copy, a restore), and then the setup is a `prepareEngine` job of its own; and
 * any job at all can find itself waiting on that same engine load and report
 * the stage. One job is followed at a time, by id, so a job that merely passed
 * through the stage is let go the moment it leaves it — otherwise the pill
 * would keep reporting a catch-up long after the part worth watching ended.
 *
 * And following one job is never allowed to become being stuck on one job: the
 * terminal event that would let it go is best-effort at the other end, so once
 * the followed job has gone quiet for `GONE_QUIET_MS`, the next setup event from
 * anywhere takes the pill over — or clears it. Nothing here waits for an event
 * that may never come.
 *
 * Nor does it wait for one that has already been. The state of play is asked
 * for on mount as well as listened for, because the job whose stage lasts
 * longest is the one whose stage is announced earliest — before the window that
 * would draw it exists at all.
 */
export function useSetupJob(): SetupJobState {
  const [state, setState] = useState<SetupJobState>({ running: false });
  const followingRef = useRef<{ id: string; heardAt: number } | null>(null);
  /** Whether any announcement has been acted on yet. The answer read on mount
   * is only ever the opening position, and an event that beat it knows more. */
  const heardRef = useRef(false);

  // What is already happening, asked for outright.
  //
  // Without this the pill could only ever show a stage it happened to be
  // listening for, and the stage that matters most begins before it exists: an
  // app update queues the one-time setup at launch, the core starts it while
  // the window is still loading, and its announcement reaches nobody. That left
  // a machine compiling the speech engine for a quarter of an hour with no
  // pill, Settings saying Echo was ready, and a person free to start the
  // meeting that would then have to wait for it — the whole of what the setup
  // job was written to prevent.
  useEffect(() => {
    let cancelled = false;
    listJobs({ activeOnly: true })
      .then((jobs) => {
        if (cancelled || heardRef.current) return;
        const job = setupJobUnderWay(jobs);
        if (!job) return;
        heardRef.current = true;
        followingRef.current = { id: job.id, heardAt: Date.now() };
        setState({ running: true, fraction: job.progress, phase: job.phase });
      })
      .catch((err) => {
        // Nothing a person can do about it and nothing worth a banner — the
        // announcements still work for anything that starts from here on. But
        // it means the pill can be blind to a setup already under way, so
        // whoever is looking at the console should see that it went missing
        // (same standard as `useEvent`).
        console.warn("Echo: could not ask what setup is already under way", err);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEvent(EVENTS.jobProgress, (payload) => {
    const { job } = payload;
    const phase = job.phase;
    const isSetup = SETUP_KINDS.has(job.kind) || phase === SETUP_PHASE;
    const following = followingRef.current;

    if (!isSetup) {
      // A job that was only worth following while it sat in the setup stage,
      // and has now moved on to its own work.
      if (following?.id === job.id) {
        heardRef.current = true;
        followingRef.current = null;
        setState({ running: false });
      }
      return;
    }

    // One at a time: another job passing through the stage doesn't take the
    // pill from the one being followed — unless that one has gone quiet for
    // long enough to be gone, in which case anything that has something to say
    // is truer than a sentence nobody is behind any more.
    if (
      following !== null &&
      following.id !== job.id &&
      Date.now() - following.heardAt < GONE_QUIET_MS
    ) {
      return;
    }

    const running = job.status === "running";
    heardRef.current = true;
    followingRef.current = running ? { id: job.id, heardAt: Date.now() } : null;
    setState({
      running,
      fraction: job.progress,
      phase,
    });
  });

  return state;
}
