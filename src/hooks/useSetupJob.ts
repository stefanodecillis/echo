import { useState } from "react";

import { EVENTS } from "../lib/ipc";
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
 */
export function useSetupJob(): SetupJobState {
  const [state, setState] = useState<SetupJobState>({ running: false });

  useEvent(EVENTS.jobProgress, (payload) => {
    if (payload.job.kind !== "download") return;
    setState({
      running: payload.job.status === "running",
      fraction: payload.job.progress ?? undefined,
      phase: payload.phase,
    });
  });

  return state;
}
