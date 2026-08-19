import { useLocation, useNavigate } from "react-router-dom";

import { useDownloadProgress } from "../hooks/useDownloadProgress";
import { useEvent } from "../hooks/useEvent";
import { useSpeechReadiness } from "../hooks/useSpeechReadiness";
import { common, notices } from "../lib/copy";
import { EVENTS } from "../lib/ipc";
import { useEchoStore } from "../lib/store";
import { ProgressBar } from "./ProgressBar";
import { DownloadIcon } from "./icons";

/**
 * The floating corner pill that keeps the one-time download visible while the
 * person gets on with the rest of the app. Hidden during onboarding (that
 * screen has its own progress) and gone the moment Echo is ready.
 */
export function SetupProgress() {
  const location = useLocation();
  const navigate = useNavigate();
  const readiness = useSpeechReadiness();
  const progress = useDownloadProgress();
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
  if (!downloading) return null;

  const fraction =
    progress && progress.totalBytes > 0
      ? progress.receivedBytes / progress.totalBytes
      : undefined;
  const percentLabel = fraction === undefined ? "" : ` · ${Math.round(fraction * 100)}%`;

  return (
    <button
      type="button"
      onClick={() => navigate("/settings")}
      className="fixed bottom-4 right-4 z-40 flex w-56 flex-col gap-1.5 rounded-xl border border-hairline bg-surface px-3.5 py-2.5 text-left shadow-md transition-colors hover:bg-surface-sunken"
    >
      <span className="flex items-center gap-2 text-xs font-medium text-ink">
        <DownloadIcon aria-hidden className="h-3.5 w-3.5 shrink-0 text-ink-soft" />
        {common.settingUpLabel}
        {percentLabel}
      </span>
      <ProgressBar value={fraction} label={common.settingUpLabel} />
    </button>
  );
}
