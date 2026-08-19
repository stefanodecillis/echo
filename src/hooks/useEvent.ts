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

    on(event, (payload) => handlerRef.current(payload)).then((fn) => {
      if (cancelled) fn();
      else stop = fn;
    });

    return () => {
      cancelled = true;
      stop?.();
    };
  }, [event]);
}
