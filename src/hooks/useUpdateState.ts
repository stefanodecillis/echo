import { useEffect, useState } from "react";

import { EVENTS, getUpdateState } from "../lib/ipc";
import type { UpdateStatePayload } from "../lib/types";
import { useEvent } from "./useEvent";

/**
 * What Echo knows about a newer version of itself.
 *
 * Asked for on mount as well as listened for, and that is the whole reason this
 * is a hook rather than an event subscription. An update can be found, fetched
 * and installed in the ninety seconds after launch — before the window has
 * finished loading, let alone rendered — and a screen that only listened would
 * miss the one announcement and stay blind for the rest of the session. The same
 * lesson `useSetupJob` records, and the same fix.
 *
 * No staleness timeout, unlike that one: the answer here is a single field with
 * no in-flight state to get stuck in, and the core re-announces it on every
 * check.
 */
export function useUpdateState(): UpdateStatePayload {
  const [state, setState] = useState<UpdateStatePayload>({ state: "idle" });

  useEffect(() => {
    let cancelled = false;
    getUpdateState()
      .then((answer) => {
        if (!cancelled) setState(answer);
      })
      .catch((err) => {
        // Nothing a person can do, and nothing worth a banner — announcements
        // still work for anything that happens from here on. But it means the
        // pill can be blind to an update already waiting, so whoever is looking
        // at the console should see that it went missing.
        console.warn("Echo: could not ask whether a new version is waiting", err);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEvent(EVENTS.updateState, setState);

  return state;
}
