import { useLocation, useNavigate } from "react-router-dom";

import { useDownloadProgress } from "../hooks/useDownloadProgress";
import { useEvent } from "../hooks/useEvent";
import { useSetupJob } from "../hooks/useSetupJob";
import { useSpeechReadiness } from "../hooks/useSpeechReadiness";
import { common, jobLine, labels, notices } from "../lib/copy";
import { EVENTS } from "../lib/ipc";
import { useEchoStore } from "../lib/store";
import { ProgressBar } from "./ProgressBar";
import { DownloadIcon, SettingsIcon } from "./icons";

/**
 * The floating corner pill that keeps the one-time setup visible while the
 * person gets on with the rest of the app. Hidden during onboarding (that
 * screen has its own progress) and gone the moment Echo is ready.
 *
 * It names the stage the job is actually in. That matters most at the end:
 * once the bytes have arrived the job is not downloading any more, it is
 * getting this computer ready to use them, and that part has no percentage and
 * can take many minutes. This pill used to disappear at exactly that point,
 * which is how somebody came to spend eighteen minutes reading a sentence about
 * a different job on another screen (field report of 2026-08-21).
 */
export function SetupProgress() {
  const location = useLocation();
  const navigate = useNavigate();
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
  const inPhase = job.running && job.phase !== undefined;
  const label = inPhase ? labels.jobPhase[job.phase!] : common.settingUpLabel;
  const fraction = inPhase ? undefined : (job.fraction ?? bytesFraction);
  const percentLabel = fraction === undefined ? "" : ` · ${Math.round(fraction * 100)}%`;
  // The arrow is about bytes arriving. Once that part is over, it would be one
  // more small thing on screen saying something untrue.
  const Icon = inPhase ? SettingsIcon : DownloadIcon;

  return (
    <button
      type="button"
      onClick={() => navigate("/settings")}
      className="fixed bottom-4 right-4 z-40 flex w-56 flex-col gap-1.5 rounded-xl border border-hairline bg-surface px-3.5 py-2.5 text-left shadow-md transition-colors hover:bg-surface-sunken"
      title={jobLine.running(label)}
    >
      <span className="flex items-center gap-2 text-xs font-medium text-ink">
        <Icon aria-hidden className="h-3.5 w-3.5 shrink-0 text-ink-soft" />
        {label}
        {percentLabel}
      </span>
      <ProgressBar value={fraction} label={label} />
    </button>
  );
}
