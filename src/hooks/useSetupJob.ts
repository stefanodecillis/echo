import { useRef, useState } from "react";

import { EVENTS } from "../lib/ipc";
import type { JobKind, JobPhase } from "../lib/types";
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

/** The kinds of job that are setup and nothing else: whatever they are doing,
 * they are the reason Echo is not ready yet. */
const SETUP_KINDS = new Set<JobKind>(["download", "prepareEngine"]);

/** The stage that is setup no matter whose job it belongs to. A catch-up that
 * has to wait for the engine to be got ready is, for those minutes, the
 * one-time setup — and it is the only thing on screen that can say so. */
const SETUP_PHASE: JobPhase = "preparingEngine";

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
 */
export function useSetupJob(): SetupJobState {
  const [state, setState] = useState<SetupJobState>({ running: false });
  const followingRef = useRef<{ id: string; heardAt: number } | null>(null);

  useEvent(EVENTS.jobProgress, (payload) => {
    const { job, phase } = payload;
    const isSetup = SETUP_KINDS.has(job.kind) || phase === SETUP_PHASE;
    const following = followingRef.current;

    if (!isSetup) {
      // A job that was only worth following while it sat in the setup stage,
      // and has now moved on to its own work.
      if (following?.id === job.id) {
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
    followingRef.current = running ? { id: job.id, heardAt: Date.now() } : null;
    setState({
      running,
      fraction: job.progress ?? undefined,
      phase,
    });
  });

  return state;
}
