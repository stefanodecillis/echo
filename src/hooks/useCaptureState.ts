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
 * How often the status is asked for again while Echo is in the middle of
 * something the screen is drawing a banner about.
 *
 * Short enough that the banner clears a moment after the engine is up rather
 * than staying on the screen for the rest of the meeting, which is what a lost
 * event would otherwise cost: on 2026-08-24 the window stopped receiving events
 * mid-meeting and everything drawn from them froze with the last one that got
 * through. Speech state is worked out fresh on every read for exactly this
 * reason, and it is only actually fresh if something re-reads it.
 */
const WATCH_INTERVAL_MS = 3_000;

/** Subscribers currently watching a state worth re-reading, and the one timer
 * they share — one poll for the whole app, however many components are up. */
let watchers = 0;
let watchTimer: number | undefined;

/** Ask the core what is happening and put the answer in the store. */
function refreshCaptureState(): void {
  getCaptureState()
    .then((s) => {
      lastEventAtMs = Date.now();
      useEchoStore.getState().setCaptureState(s);
    })
    .catch(() => {
      // Stay with what is on screen; the next tick or the next event corrects
      // it, and nothing here is worth bothering the user with.
    });
}

/**
 * The current capture state, kept fresh from the capture-state event — with a
 * live clock. The core only emits on state changes, so `elapsedMs` as
 * delivered is frozen between events; this hook advances it locally once a
 * second while recording (not while paused), which is what keeps the "2:30"
 * in every header actually moving.
 *
 * While speech is being got ready, or has failed to come up, the status is also
 * asked for outright every few seconds: those are the two states a banner is
 * drawn from, and a banner that outlives the thing it describes is worse than no
 * banner at all.
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

  // The two states somebody is being shown a sentence about, and so the two
  // that must not be able to get stuck.
  const watching = state.speech === "preparing" || state.speech === "unavailable";
  useEffect(() => {
    if (!watching) return;
    watchers += 1;
    if (watchers === 1) {
      watchTimer = window.setInterval(refreshCaptureState, WATCH_INTERVAL_MS);
    }
    return () => {
      watchers -= 1;
      if (watchers === 0 && watchTimer !== undefined) {
        window.clearInterval(watchTimer);
        watchTimer = undefined;
      }
    };
  }, [watching]);

  const ticking = state.state === "recording" || state.state === "degraded";
  useEffect(() => {
    if (!ticking) return;
    const id = window.setInterval(() => tick((n) => n + 1), 1_000);
    return () => window.clearInterval(id);
  }, [ticking]);

  if (!ticking) return state;
  return { ...state, elapsedMs: state.elapsedMs + (Date.now() - lastEventAtMs) };
}
