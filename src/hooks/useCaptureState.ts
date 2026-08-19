import { useEffect } from "react";

import { EVENTS, getCaptureState } from "../lib/ipc";
import { useEchoStore } from "../lib/store";
import type { CaptureStatus } from "../lib/types";
import { useEvent } from "./useEvent";

/** Only the first mount anywhere in the tree fetches the initial state; every
 * mount after that just subscribes to updates against the same store. Module
 * state, not a ref, because it needs to survive across components. */
let didFetchInitial = false;

/**
 * The current capture state, kept fresh from the capture-state event.
 *
 * Safe to call from as many components as need it — the Live screen, the
 * sidebar's live indicator, a "Stop" button in Settings — since they all read
 * and write the same store entry (`src/lib/store.ts`).
 */
export function useCaptureState(): CaptureStatus {
  const state = useEchoStore((s) => s.captureState);
  const setCaptureState = useEchoStore((s) => s.setCaptureState);

  useEffect(() => {
    if (didFetchInitial) return;
    didFetchInitial = true;
    getCaptureState()
      .then(setCaptureState)
      .catch(() => {
        // Stay on the idle default; the event stream corrects it once the
        // core is ready, and nothing here is worth bothering the user with.
      });
  }, [setCaptureState]);

  useEvent(EVENTS.captureState, setCaptureState);

  return state;
}
