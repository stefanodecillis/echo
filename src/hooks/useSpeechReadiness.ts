import { useEffect, useState } from "react";

import { EVENTS, getSpeechReadiness } from "../lib/ipc";
import type { SpeechReadiness } from "../lib/types";
import { useEvent } from "./useEvent";

/**
 * Whether Echo has everything it needs to record, kept current while the
 * one-time download runs in the background. `undefined` until the first
 * answer arrives — treat that as "don't know yet", not "not ready".
 */
export function useSpeechReadiness(): SpeechReadiness | undefined {
  const [readiness, setReadiness] = useState<SpeechReadiness>();

  useEffect(() => {
    getSpeechReadiness()
      .then(setReadiness)
      .catch(() => {
        // Leave it unknown; the next download event retries below.
      });
  }, []);

  useEvent(EVENTS.downloadProgress, (payload) => {
    if (payload.done || payload.error) {
      getSpeechReadiness()
        .then(setReadiness)
        .catch(() => {});
    }
  });

  return readiness;
}
