import { useEffect, useRef, useState } from "react";

import { useEvent } from "../../hooks";
import { EVENTS, getDetectionStatus } from "../../lib/ipc";
import type { DetectionStatus } from "../../lib/types";

const IDLE_DETECTION: DetectionStatus = {
  state: "idle",
  enabled: true,
  signals: [],
  suggestStop: false,
};

/**
 * Whether a meeting looks like it's starting, kept fresh from the detection
 * event. Local to Home: nothing else needs to know this, only whether to
 * swap the quiet idle hero for a prominent Start card.
 */
export function useDetectionStatus(): DetectionStatus {
  const [status, setStatus] = useState<DetectionStatus>(IDLE_DETECTION);
  const fetchedRef = useRef(false);

  useEffect(() => {
    if (fetchedRef.current) return;
    fetchedRef.current = true;
    getDetectionStatus()
      .then(setStatus)
      .catch(() => {
        // Stay on the idle default; the event stream corrects it once the
        // core is ready, and nothing here is worth bothering the user with.
      });
  }, []);

  useEvent(EVENTS.detection, setStatus);

  return status;
}
