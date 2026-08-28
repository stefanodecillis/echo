import { Link, useNavigate } from "react-router-dom";

import { useRunningWork } from "../hooks/useActiveJobs";
import { useSetupPillState, type SetupPillState } from "../hooks/useSetupPill";
import { jobLine } from "../lib/copy";
import { presentJob } from "../lib/jobs";
import type { Id, Job } from "../lib/types";
import { ProgressBar } from "./ProgressBar";
import { WorkingRing } from "./WorkingRing";
import { DownloadIcon, SettingsIcon } from "./icons";

/** The corner holds one thing, so the shell is written once. */
const shellClasses =
  "fixed bottom-4 right-4 z-40 flex w-56 flex-col gap-1.5 rounded-xl border border-hairline bg-surface px-3.5 py-2.5 text-left shadow-md transition-colors hover:bg-surface-sunken";

const lineClasses = "flex items-center gap-2 text-xs font-medium text-ink";

/**
 * The floating corner pill: what Echo is busy with, from any screen.
 *
 * WHY ONE COMPONENT FOR TWO THINGS
 * Because there is one corner. The one-time setup and a meeting's background
 * work both want a persistent sentence in the same 224 pixels, and two `fixed`
 * elements at the same coordinates is a collision, not a design. So this picks,
 * and **setup always wins**: it is the more important of the two and it is the
 * one blocking recording. It is also why a catch-up sitting in the engine-loading
 * stage cannot be announced twice with the same words in two corners.
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
  const working = useRunningWork();

  if (setup) return <SetupPill state={setup} />;
  if (working?.meetingId) return <WorkPill job={working} meetingId={working.meetingId} />;
  return null;
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
