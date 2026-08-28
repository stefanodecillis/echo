import { useLocation } from "react-router-dom";

import { common, labels, notices } from "../lib/copy";
import { EVENTS } from "../lib/ipc";
import { useEchoStore } from "../lib/store";
import { useDownloadProgress } from "./useDownloadProgress";
import { useEvent } from "./useEvent";
import { useSetupJob } from "./useSetupJob";
import { useSpeechReadiness } from "./useSpeechReadiness";

/** Which half of the one-time setup is happening. They read as two different
 * activities to a person, so they are two states here. */
export type SetupStage = "downloading" | "preparing";

export interface SetupPillState {
  stage: SetupStage;
  label: string;
  /** 0..1 while there is an honest fraction, absent once there is not. */
  fraction?: number;
}

/**
 * Whether the one-time setup has something to say, and what.
 *
 * It names the stage the job is actually in, and that matters most at the end:
 * once the bytes have arrived the job is not downloading any more, it is getting
 * this computer ready to use them, and that part has no percentage and can take
 * many minutes. The pill used to disappear at exactly that point, which is how
 * somebody came to spend eighteen minutes reading a sentence about a different
 * job on another screen (field report of 2026-08-21).
 *
 * Hidden during onboarding, which draws its own progress, and gone the moment
 * Echo is ready.
 *
 * A hook rather than a component because the corner it draws in holds exactly
 * one thing (see `components/StatusCorner.tsx`), and deciding *whether* setup
 * gets that spot has to happen before anything is drawn. The download-finished
 * toast is raised from in here, so it goes out whichever pill ends up on screen —
 * or none.
 */
export function useSetupPillState(): SetupPillState | null {
  const location = useLocation();
  const readiness = useSpeechReadiness();
  const progress = useDownloadProgress();
  const job = useSetupJob();
  const addToast = useEchoStore((s) => s.addToast);

  useEvent(EVENTS.downloadProgress, (payload) => {
    if (payload.done && payload.levelId && !payload.error) {
      addToast({ level: "info", message: notices.setupDone });
    }
  });

  if (location.pathname.startsWith("/onboarding")) return null;

  const downloading = progress
    ? !progress.done && !progress.error
    : (readiness?.downloading ?? false);
  // The job running is the last word: it outlives the bytes arriving.
  if (!downloading && !job.running) return null;

  const bytesFraction =
    progress && progress.totalBytes > 0
      ? progress.receivedBytes / progress.totalBytes
      : undefined;

  // In a stage of its own, the job speaks for itself — and reports no fraction,
  // because there is none. Nothing is invented to fill the bar.
  if (job.running && job.phase !== undefined) {
    return { stage: "preparing", label: labels.jobPhase[job.phase] };
  }
  return {
    stage: "downloading",
    label: common.settingUpLabel,
    fraction: job.fraction ?? bytesFraction,
  };
}
