import { Link, useNavigate } from "react-router-dom";

import { useRunningWork } from "../hooks/useActiveJobs";
import { useCaptureState } from "../hooks/useCaptureState";
import { useSetupPillState, type SetupPillState } from "../hooks/useSetupPill";
import { useUpdateState } from "../hooks/useUpdateState";
import { jobLine, update as updateCopy } from "../lib/copy";
import { restartForUpdate } from "../lib/ipc";
import { presentJob } from "../lib/jobs";
import { pickCorner } from "../lib/statusCorner";
import type { Id, Job } from "../lib/types";
import { ProgressBar } from "./ProgressBar";
import { WorkingRing } from "./WorkingRing";
import { DownloadIcon, SettingsIcon, UpdateReadyIcon } from "./icons";

/** The corner holds one thing, so the shell is written once. */
const shellClasses =
  "fixed bottom-4 right-4 z-40 flex w-56 flex-col gap-1.5 rounded-xl border border-hairline bg-surface px-3.5 py-2.5 text-left shadow-md transition-colors hover:bg-surface-sunken";

const lineClasses = "flex items-center gap-2 text-xs font-medium text-ink";

/**
 * The floating corner pill: what Echo is busy with, from any screen.
 *
 * WHY ONE COMPONENT FOR THREE THINGS
 * Because there is one corner. The one-time setup, a waiting update and a
 * meeting's background work all want a persistent sentence in the same 224
 * pixels, and two `fixed` elements at the same coordinates is a collision, not a
 * design. So this picks exactly one, and the order lives in `lib/statusCorner.ts`
 * where it can be tested — with the reasoning for each step. It is also why a
 * catch-up sitting in the engine-loading stage cannot be announced twice with
 * the same words in two corners.
 *
 * WHY THE WORK ARM IS RUNNING-ONLY
 * The rule at the top of `lib/jobs.ts`: an ambient sentence is a sentence about
 * something happening. Work that is queued or parked for a recording says
 * nothing here — the rail's dot is what covers that case, wordlessly.
 *
 * WHY THE WORK ARM HAS NO BAR
 * The map it reads deliberately does not carry a fraction (see
 * `hooks/useActiveJobs.ts`): a running job announces itself four times a second
 * and nothing that reads that map draws a bar. The setup arm keeps its bar,
 * because the download genuinely reports bytes and that is the one place in the
 * app an honest percentage exists.
 */
export function StatusCorner() {
  const setup = useSetupPillState();
  const update = useUpdateState();
  const working = useRunningWork();

  switch (
    pickCorner({
      setup: setup !== null,
      update: update.state === "ready",
      work: working?.meetingId !== undefined,
    })
  ) {
    case "setup":
      return setup && <SetupPill state={setup} />;
    case "update":
      return <UpdatePill />;
    case "work":
      return working?.meetingId ? <WorkPill job={working} meetingId={working.meetingId} /> : null;
    case "none":
      return null;
  }
}

/**
 * A new Echo is on disk; all that is left is a restart, and that is the person's
 * to ask for.
 *
 * Two shapes, because offering a restart in the middle of a recording would be
 * offering something Echo will refuse. While a meeting is being recorded this
 * says what will happen instead and cannot be pressed — the same posture as the
 * waiting chip on a meeting row: if nothing can be done about it right now, do
 * not draw a control.
 *
 * No progress bar and no percentage in either shape. The download is over by the
 * time this exists, and the thing being waited on now is a decision, not work.
 */
function UpdatePill() {
  const capture = useCaptureState();
  const midMeeting =
    capture.state === "recording" ||
    capture.state === "paused" ||
    capture.state === "degraded" ||
    capture.state === "starting" ||
    capture.state === "stopping";

  const line = (
    <span className={lineClasses}>
      <UpdateReadyIcon aria-hidden className="h-3.5 w-3.5 shrink-0 text-ink-soft" />
      <span className="min-w-0 truncate">{updateCopy.readyTitle}</span>
    </span>
  );

  if (midMeeting) {
    return (
      <div className={`${shellClasses} cursor-default`} title={updateCopy.waitingForMeeting}>
        {line}
        <span className="text-xs text-ink-faint">{updateCopy.waitingForMeeting}</span>
      </div>
    );
  }

  return (
    <button
      type="button"
      // Nothing to catch: the command either refuses (and this shape is only
      // drawn when it will not) or this process is gone before a promise
      // settles.
      onClick={() => void restartForUpdate()}
      className={shellClasses}
      title={updateCopy.readyAction}
    >
      {line}
      <span className="text-xs text-ink-faint">{updateCopy.readyAction}</span>
      {/* The cost, said before it is paid rather than discovered afterwards. */}
      <span className="text-xs text-ink-ghost">{updateCopy.readySetupHint}</span>
    </button>
  );
}

function SetupPill({ state }: { state: SetupPillState }) {
  const navigate = useNavigate();
  const percentLabel =
    state.fraction === undefined ? "" : ` · ${Math.round(state.fraction * 100)}%`;
  // The arrow is about bytes arriving. Once that part is over it would be one
  // more small thing on screen saying something untrue.
  const Icon = state.stage === "preparing" ? SettingsIcon : DownloadIcon;

  return (
    <button
      type="button"
      onClick={() => navigate("/settings")}
      className={shellClasses}
      title={jobLine.running(state.label)}
    >
      <span className={lineClasses}>
        <Icon aria-hidden className="h-3.5 w-3.5 shrink-0 text-ink-soft" />
        {state.label}
        {percentLabel}
      </span>
      <ProgressBar value={state.fraction} label={state.label} />
    </button>
  );
}

function WorkPill({ job, meetingId }: { job: Job; meetingId: Id }) {
  const sentence = jobLine.running(presentJob(job).label);
  return (
    <Link to={`/meeting/${meetingId}`} className={shellClasses} title={sentence}>
      <span className={lineClasses}>
        <WorkingRing size="sm" className="shrink-0" />
        <span className="min-w-0 truncate">{sentence}</span>
      </span>
    </Link>
  );
}
