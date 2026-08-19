import { useState } from "react";

import { Button, EchoMark } from "../components";
import { home, panel } from "../lib/copy";
import { panelClose, startRecording, toUiError } from "../lib/ipc";
import type { PanelState, UiError } from "../lib/types";
import { formatAgo } from "./format";
import { useNow } from "./useNow";

export interface DetectedPanelProps {
  state: Extract<PanelState, { kind: "detected" }>;
}

/** "A meeting was just detected — want Echo listening?" One button, one exit. */
export function DetectedPanel({ state }: DetectedPanelProps) {
  const now = useNow();
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<UiError>();

  const handleStart = async () => {
    setStarting(true);
    setError(undefined);
    try {
      await startRecording({ detectedApp: state.appName });
      // Success: the panel's job here is done. Rust hides it anyway the moment
      // capture reports it is recording; this only makes the panel go away on
      // the click rather than on the event. `panelClose`, not `panelDismiss` —
      // starting a recording is the opposite of "not this meeting".
      await panelClose().catch(() => {});
    } catch (err) {
      setError(toUiError(err));
    } finally {
      setStarting(false);
    }
  };

  return (
    <div className="flex items-center gap-3">
      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-accent-soft text-accent">
        <EchoMark className="h-5 w-5" />
      </div>
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-semibold text-ink">{panel.detectedTitle}</p>
        <p className="tabular-nums text-xs text-ink-faint">{formatAgo(state.detectedAtMs, now)}</p>
        {error && <p className="mt-1 truncate text-xs text-live">{error.message}</p>}
      </div>
      <Button variant="primary" size="sm" loading={starting} onClick={() => void handleStart()}>
        {home.startButton}
      </Button>
    </div>
  );
}
