import { useEffect, useRef } from "react";

import { on, type EventName, type EventPayloads } from "../lib/ipc";

/**
 * Subscribe to a Tauri event for the life of the component.
 *
 * `handler` is read through a ref, so passing a fresh arrow function on every
 * render (the normal case) does not tear down and re-create the underlying
 * `listen` call — only a change of `event` itself does that.
 *
 * ```ts
 * useEvent(EVENTS.transcriptFinal, (payload) => {
 *   if (payload.meetingId === meetingId) appendSegment(payload.segment);
 * });
 * ```
 */
export function useEvent<K extends EventName>(
  event: K,
  handler: (payload: EventPayloads[K]) => void,
): void {
  const handlerRef = useRef(handler);

  useEffect(() => {
    handlerRef.current = handler;
  }, [handler]);

  useEffect(() => {
    let stop: (() => void) | undefined;
    let cancelled = false;

    on(event, (payload) => handlerRef.current(payload))
      .then((fn) => {
        if (cancelled) fn();
        else stop = fn;
      })
      .catch((err) => {
        // Subscribing itself failed, so this component will never hear about
        // `event` again — silently, which is how a whole meeting's worth of
        // updates went missing on 2026-08-24. There is nothing the person
        // using Echo can do about it and nothing worth interrupting them with
        // (the screens that matter re-read the core on a timer anyway), but
        // whoever is looking at the console should see which event died.
        console.warn(`Echo: could not subscribe to the "${event}" event`, err);
      });

    return () => {
      cancelled = true;
      stop?.();
    };
  }, [event]);
}
