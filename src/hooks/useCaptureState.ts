import { useEffect, useState } from "react";

import { EVENTS, getCaptureState } from "../lib/ipc";
import { useEchoStore } from "../lib/store";
import type { CaptureStatus } from "../lib/types";
import { useEvent } from "./useEvent";

/** Only the first mount anywhere in the tree fetches the initial state; every
 * mount after that just subscribes to updates against the same store. Module
 * state, not a ref, because it needs to survive across components. */
let didFetchInitial = false;

/** When the last capture-state event landed, so the clock below can keep
 * counting between events. Module state for the same reason as above. */
let lastEventAtMs = Date.now();

/**
 * The current capture state, kept fresh from the capture-state event — with a
 * live clock. The core only emits on state changes, so `elapsedMs` as
 * delivered is frozen between events; this hook advances it locally once a
 * second while recording (not while paused), which is what keeps the "2:30"
 * in every header actually moving.
 *
 * Safe to call from as many components as need it — the Live screen, the
 * sidebar's live indicator, a "Stop" button in Settings — since they all read
 * and write the same store entry (`src/lib/store.ts`).
 */
export function useCaptureState(): CaptureStatus {
  const state = useEchoStore((s) => s.captureState);
  const setCaptureState = useEchoStore((s) => s.setCaptureState);
  const [, tick] = useState(0);

  useEffect(() => {
    if (didFetchInitial) return;
    didFetchInitial = true;
    getCaptureState()
      .then((s) => {
        lastEventAtMs = Date.now();
        setCaptureState(s);
      })
      .catch(() => {
        // Stay on the idle default; the event stream corrects it once the
        // core is ready, and nothing here is worth bothering the user with.
      });
  }, [setCaptureState]);

  useEvent(EVENTS.captureState, (payload) => {
    lastEventAtMs = Date.now();
    setCaptureState(payload);
  });

  const ticking = state.state === "recording" || state.state === "degraded";
  useEffect(() => {
    if (!ticking) return;
    const id = window.setInterval(() => tick((n) => n + 1), 1_000);
    return () => window.clearInterval(id);
  }, [ticking]);

  if (!ticking) return state;
  return { ...state, elapsedMs: state.elapsedMs + (Date.now() - lastEventAtMs) };
}
