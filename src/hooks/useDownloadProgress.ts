import { useState } from "react";

import { EVENTS } from "../lib/ipc";
import type { DownloadProgressPayload, Id } from "../lib/types";
import { useEvent } from "./useEvent";

/**
 * The latest progress for one download, or the most recent download overall
 * when no `assetId` is given — the onboarding download step only ever has one
 * running at a time.
 */
export function useDownloadProgress(assetId?: Id): DownloadProgressPayload | undefined {
  const [progress, setProgress] = useState<DownloadProgressPayload>();

  useEvent(EVENTS.downloadProgress, (payload) => {
    if (!assetId || payload.assetId === assetId) setProgress(payload);
  });

  return progress;
}
