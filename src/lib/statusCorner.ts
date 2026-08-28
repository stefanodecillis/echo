/**
 * Which of the things that want the bottom-right corner actually gets it.
 *
 * WHY THIS IS A FUNCTION AND NOT THREE IFS
 * Because it is a priority order, and a priority order written as a chain of
 * early returns inside a component is a rule nobody can test and anybody can
 * reorder by accident. Three claimants is already enough for that to matter.
 *
 * THE ORDER, AND THE REASON FOR EACH STEP
 *
 * 1. `setup` — the one-time work Echo needs before it can understand speech at
 *    all. It wins because it is the only one of the three that is *blocking*:
 *    until it finishes, a recording gets audio but no words. It is also the one
 *    a person is most likely to be waiting on, having just been told Echo was
 *    getting ready.
 *
 * 2. `update` — a new version on disk, waiting for a restart. Above the work
 *    pill because it is persistent and actionable: it will still be true in an
 *    hour, and there is something to press. It sits below setup partly on
 *    importance and partly because the two would otherwise say the same thing
 *    twice — restarting for an update is what *causes* the next setup, since the
 *    compiled speech model is keyed to the app that asked for it.
 *
 * 3. `work` — Echo is finishing off a meeting. Bottom because it is transient,
 *    because nothing is required of anybody, and because it is the only one of
 *    the three that is already said elsewhere: every affected row carries a chip
 *    and the rail carries a dot. Losing it to a pill above costs nothing.
 */
export type CornerClaim = "setup" | "update" | "work" | "none";

export interface CornerClaims {
  /** The one-time setup is running. */
  setup: boolean;
  /** A new version is downloaded and only a restart is left. */
  update: boolean;
  /** A meeting's background work is running right now. */
  work: boolean;
}

export function pickCorner({ setup, update, work }: CornerClaims): CornerClaim {
  if (setup) return "setup";
  if (update) return "update";
  if (work) return "work";
  return "none";
}
