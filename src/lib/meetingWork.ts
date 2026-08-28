import { isActive, lastFailure } from "./jobs";
import type { Job, MeetingSummary } from "./types";

/**
 * What a meeting row should say about Echo still having work to do on it.
 *
 * WHY THIS IS ITS OWN FILE
 * `lib/jobs.ts` is policy about jobs — which one a screen names, which failure
 * it still has to own up to. This is the one question a *list of meetings* asks
 * of that policy, and it is a different question: a row has no room for a
 * sentence, no kind to filter by, and sometimes no job at all to point at while
 * still having something true to say.
 *
 * THE RULE IT INHERITS
 * The one at the top of `lib/jobs.ts`, and it is not negotiable here either:
 * `running` is the only state that may be dressed as something happening.
 * `waiting` says it is waiting, and gets no motion and no bar, because a job
 * that has not started has made no progress.
 *
 * WHAT IT ADDS
 * Two states that are inferences from *absence*, and so are the only two that
 * consult the meeting's own status:
 *
 *  - `stopped` — the meeting is still marked as being worked on, and the last
 *    thing that ran failed. `JobRuntime::settle_meeting` only ever runs for work
 *    that finished (src-tauri/src/session/jobs.rs), so nothing will ever move
 *    this meeting on. Until now no screen in the app said so.
 *  - `unknown` — the meeting is still marked as being worked on and there is
 *    nothing to point at. True, and the most that can honestly be said: the jobs
 *    may be a millisecond from being enqueued, or an announcement may have gone
 *    missing. It claims no activity whatsoever.
 */
export type MeetingWork =
  /** Happening right now. The only state that may move on screen. */
  | { state: "running"; job: Job }
  /** In the queue. Nothing about *which* work, because nothing is happening and
   * a name would be the only moving part of a still sentence. */
  | { state: "waiting" }
  /** Set aside because a recording is going on. */
  | { state: "deferred" }
  /** Stopped without finishing, and nothing will pick it up again. Carries the
   * job because *what* did not finish is the actionable half: it tells a person
   * what is missing from the meeting they are looking at. */
  | { state: "stopped"; job: Job }
  /** Not finished, and nothing to point at. */
  | { state: "unknown" }
  /** Nothing to say. Also what a row shows before the first read lands. */
  | { state: "none" };

/**
 * Which of the six a row is in.
 *
 * The order of the checks is the whole design, so it is written out:
 *
 * | condition                                  | result     |
 * |--------------------------------------------|------------|
 * | `active` not read yet                      | `none`     |
 * | the meeting is recording                   | `none`     |
 * | one of its jobs is running                 | `running`  |
 * | one of its jobs is parked                  | `deferred` |
 * | one of its jobs is queued                  | `waiting`  |
 * | still processing, and its last job failed  | `stopped`  |
 * | still processing                           | `unknown`  |
 * | otherwise                                  | `none`     |
 *
 * Two things that order gets right, and a plainer one would not:
 *
 * `running` / `deferred` / `waiting` are decided by the jobs alone, and are
 * never gated on the meeting's status. Asking for a recap on a meeting that
 * finished last week queues the work and leaves the meeting `complete` —
 * nothing in the summarize path sets it back to `processing`, unlike the
 * transcript and speaker passes. Gating on `processing` would hide exactly the
 * case somebody most wants confirmed: that the button they just pressed did
 * something.
 *
 * `deferred` comes before `waiting` because starting a recording parks every
 * outstanding row at once, so a mixture of the two exists only for the instant
 * `release` takes to put them back — and through that instant "nothing is
 * moving" is the truer of the two things to say.
 *
 * And the failure check comes last, behind the meeting's status, which is what
 * makes a partial failure history safe to reason from: a failed row left over
 * from before a successful "Listen again" is suppressed by the meeting having
 * settled to `complete`, without this needing to know the meeting's whole past.
 */
export function meetingWork(
  meeting: Pick<MeetingSummary, "id" | "status">,
  active: Job[] | undefined,
  failed: Job[] | undefined,
): MeetingWork {
  // Nobody has asked the core yet. Not the same as an answer of "nothing".
  if (active === undefined) return { state: "none" };

  // A recording is the Live screen's, the rail's live card's and the tray's to
  // narrate. A fourth treatment saying the same thing is a fourth thing to keep
  // true.
  if (meeting.status === "recording") return { state: "none" };

  // The buckets are keyed by meeting already; filtering again costs nothing and
  // makes this answerable from any list of jobs, which is how it is tested.
  const mine = active.filter((job) => job.meetingId === meeting.id && isActive(job));

  const running = mine.find((job) => job.status === "running");
  if (running) return { state: "running", job: running };
  if (mine.some((job) => job.status === "paused")) return { state: "deferred" };
  if (mine.some((job) => job.status === "queued")) return { state: "waiting" };

  if (meeting.status === "processing") {
    const stopped = lastFailure((failed ?? []).filter((job) => job.meetingId === meeting.id));
    if (stopped) return { state: "stopped", job: stopped };
    return { state: "unknown" };
  }

  return { state: "none" };
}
