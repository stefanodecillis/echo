import { useEffect, useState } from "react";

/**
 * The current time in millis, refreshed once a second — but the interval
 * only runs while this document is actually visible.
 *
 * The panel is a tiny window Rust hides rather than destroys, so its React
 * tree (and any plain `setInterval`) keeps running in the background unless
 * something stops it. That would be a timer ticking for a window nobody can
 * see, which is exactly the kind of idle work mantra 1 rules out. The Page
 * Visibility API is the standard signal for "is this webview actually on
 * screen", so the ticker starts on mount only if visible, and pauses/resumes
 * on every `visibilitychange`.
 *
 * The braces to that belt: Rust sends the panel a `null` state as it hides, so
 * whatever was using this hook unmounts and the interval goes with it, whether
 * or not the platform bothers to report the window as hidden.
 */
export function useNow(intervalMs = 1000): number {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    let id: number | undefined;

    const start = () => {
      if (id !== undefined) return;
      setNow(Date.now());
      id = window.setInterval(() => setNow(Date.now()), intervalMs);
    };
    const stop = () => {
      if (id === undefined) return;
      window.clearInterval(id);
      id = undefined;
    };

    if (document.visibilityState === "visible") start();

    const onVisibility = () => {
      if (document.visibilityState === "visible") start();
      else stop();
    };
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
      stop();
    };
  }, [intervalMs]);

  return now;
}
