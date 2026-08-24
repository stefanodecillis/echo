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
 * How many times something has been written into the store's capture state.
 * Read before a request goes out and checked again when it comes back.
 *
 * The core samples its answer the moment the call reaches it and delivers it
 * whenever it gets round to it, so a poll fired a millisecond before somebody
 * hits Stop comes back saying "recording" after the stopping and idle events
 * have already been applied. Writing that answer in would put a finished
 * meeting back on screen — the Home page watches for a move into a live state
 * and sends you to /live — and set the clock counting again, for the ten
 * seconds until the next poll happened to disagree. A poll that comes back to a
 * number it does not recognise is a poll answering about a moment that has been
 * overtaken, so it says nothing.
 */
let stateSeq = 0;

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

/**
 * How often the status is asked for again while a meeting is under way, banner
 * or no banner.
 *
 * Same lesson, wider: every header's clock, the recording dot, the pause label —
 * all of it is drawn from a status that only ever changes when an event arrives.
 * If events stop, the clock keeps counting (it is extrapolated locally) and the
 * screen goes on claiming a recording is fine long after it might not be. Ten
 * seconds is often enough that nothing on screen can drift far from the truth,
 * and rare enough to be free.
 */
const MEETING_INTERVAL_MS = 10_000;

/** Capture states that mean a meeting is under way and worth re-reading. */
const ACTIVE_STATES = new Set(["starting", "recording", "paused", "degraded", "stopping"]);

/** Subscribers currently watching a state worth re-reading, split by how
 * closely, and the one timer they share — one poll for the whole app, however
 * many components are up. */
let bannerWatchers = 0;
let meetingWatchers = 0;
let anyWatchers = 0;
let watchTimer: number | undefined;
let watchPeriodMs: number | undefined;

/** Two statuses that would draw exactly the same screen. */
function sameStatus(a: CaptureStatus, b: CaptureStatus): boolean {
  return (
    a.state === b.state &&
    a.meetingId === b.meetingId &&
    a.elapsedMs === b.elapsedMs &&
    a.degradedReason === b.degradedReason &&
    a.pendingUtterances === b.pendingUtterances &&
    a.startedAt === b.startedAt &&
    a.speech === b.speech &&
    a.activeChannels.length === b.activeChannels.length &&
    a.activeChannels.every((c, i) => c === b.activeChannels[i])
  );
}

/** Ask the core what is happening and put the answer in the store. */
function refreshCaptureState(): void {
  const asked = stateSeq;
  getCaptureState()
    .then((s) => {
      // Anything that landed while this was in the air knows more than it does.
      if (stateSeq !== asked) return;
      const store = useEchoStore.getState();
      // An idle app answering "still idle" every ten seconds must not re-render
      // every component that reads capture state. Only a genuine change is
      // worth a new object — and only then does the clock get a new anchor,
      // since `elapsedMs` and the moment it was true belong together.
      if (sameStatus(store.captureState, s)) return;
      lastEventAtMs = Date.now();
      stateSeq += 1;
      store.setCaptureState(s);
    })
    .catch(() => {
      // Stay with what is on screen; the next tick or the next event corrects
      // it, and nothing here is worth bothering the user with.
    });
}

/** Poll only while the window is actually on screen — a hidden window catches
 * up the moment it comes back (see the listeners below). */
function refreshIfVisible(): void {
  if (document.visibilityState !== "visible") return;
  refreshCaptureState();
}

/** Start, restart, or stop the shared timer so its beat matches the closest
 * watch anyone is currently keeping. */
function syncWatchTimer(): void {
  const period =
    bannerWatchers > 0 ? WATCH_INTERVAL_MS : meetingWatchers > 0 ? MEETING_INTERVAL_MS : undefined;
  if (period === watchPeriodMs) return;
  if (watchTimer !== undefined) {
    window.clearInterval(watchTimer);
    watchTimer = undefined;
  }
  watchPeriodMs = period;
  if (period !== undefined) watchTimer = window.setInterval(refreshIfVisible, period);
}

/** Coming back to Echo is the moment its screens are most likely to be stale —
 * and the moment somebody is looking. Both listeners are installed once, for as
 * long as anything reads capture state at all. */
function watchWindowReturns(): () => void {
  anyWatchers += 1;
  if (anyWatchers === 1) {
    window.addEventListener("focus", refreshCaptureState);
    document.addEventListener("visibilitychange", refreshIfVisible);
  }
  return () => {
    anyWatchers -= 1;
    if (anyWatchers === 0) {
      window.removeEventListener("focus", refreshCaptureState);
      document.removeEventListener("visibilitychange", refreshIfVisible);
    }
  };
}

/**
 * The current capture state, kept fresh from the capture-state event — with a
 * live clock. The core only emits on state changes, so `elapsedMs` as
 * delivered is frozen between events; this hook advances it locally once a
 * second while recording (not while paused), which is what keeps the "2:30"
 * in every header actually moving.
 *
 * The status is also asked for outright: every few seconds while speech is being
 * got ready or has failed to come up (a banner that outlives the thing it
 * describes is worse than no banner at all), every ten seconds for the whole of
 * a meeting, and whenever somebody comes back to the window. Each answer that
 * differs from what is on screen re-anchors the clock, so the counting above
 * cannot drift away from the truth; an answer that changes nothing is dropped,
 * so an idle Echo stays perfectly still; and an answer the core worked out
 * before an event that has since arrived is dropped as well, so asking can only
 * ever catch the screen up, never wind it back.
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
    const asked = stateSeq;
    getCaptureState()
      .then((s) => {
        // The first event can beat the first answer; when it does, the event is
        // the newer of the two and this one is history.
        if (stateSeq !== asked) return;
        lastEventAtMs = Date.now();
        stateSeq += 1;
        setCaptureState(s);
      })
      .catch(() => {
        // Stay on the idle default; the event stream corrects it once the
        // core is ready, and nothing here is worth bothering the user with.
      });
  }, [setCaptureState]);

  useEvent(EVENTS.captureState, (payload) => {
    lastEventAtMs = Date.now();
    // The core said so just now, which beats anything a poll is still waiting
    // to hear back about.
    stateSeq += 1;
    setCaptureState(payload);
  });

  // Refresh whenever somebody comes back to the window, whatever is happening —
  // that is when a stale screen is both most likely and most visible.
  useEffect(watchWindowReturns, []);

  // The two states somebody is being shown a sentence about, and so the two
  // that must not be able to get stuck.
  const banner = state.speech === "preparing" || state.speech === "unavailable";
  useEffect(() => {
    if (!banner) return;
    bannerWatchers += 1;
    syncWatchTimer();
    return () => {
      bannerWatchers -= 1;
      syncWatchTimer();
    };
  }, [banner]);

  // And the whole of a meeting, at a slower beat: everything the recording
  // screens draw comes from this one status.
  const meeting = state.meetingId != null && ACTIVE_STATES.has(state.state);
  useEffect(() => {
    if (!meeting) return;
    meetingWatchers += 1;
    syncWatchTimer();
    return () => {
      meetingWatchers -= 1;
      syncWatchTimer();
    };
  }, [meeting]);

  const ticking = state.state === "recording" || state.state === "degraded";
  useEffect(() => {
    if (!ticking) return;
    const id = window.setInterval(() => tick((n) => n + 1), 1_000);
    return () => window.clearInterval(id);
  }, [ticking]);

  if (!ticking) return state;
  return { ...state, elapsedMs: state.elapsedMs + (Date.now() - lastEventAtMs) };
}
