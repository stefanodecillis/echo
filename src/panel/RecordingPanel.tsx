import { useState } from "react";

import { Button, RecordingDot } from "../components";
import { formatElapsed } from "../components/lib/format";
import { anchors, home, labels } from "../lib/copy";
import { panelClose, stopRecording, toUiError } from "../lib/ipc";
import type { PanelState, UiError } from "../lib/types";
import { useNow } from "./useNow";

export interface RecordingPanelProps {
  state: Extract<PanelState, { kind: "recording" }>;
}

/**
 * The quick control while Echo is listening: one look, one Stop. No confirm
 * dialog — that lives on the main window's Live screen; this panel exists so
 * a single click here is enough. The recap that follows is the main window's
 * job, not this one's.
 */
export function RecordingPanel({ state }: RecordingPanelProps) {
  const now = useNow();
  const [stopping, setStopping] = useState(false);
  const [error, setError] = useState<UiError>();

  const handleStop = async () => {
    setStopping(true);
    setError(undefined);
    try {
      await stopRecording();
      // The recap is the main window's job from here. Rust also hides this panel
      // when capture reports it has stopped; closing it here means the click
      // itself is what makes it go away.
      await panelClose().catch(() => {});
    } catch (err) {
      setError(toUiError(err));
    } finally {
      setStopping(false);
    }
  };

  const paused = state.paused;

  return (
    <div className="flex items-center gap-3">
      {/* No label: the line beside the dot already says it, and a screen reader
          should hear it once. */}
      <RecordingDot paused={paused} />
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-semibold text-ink">
          {paused ? labels.captureState.paused : anchors.listening}
        </p>
        {/* The clock counts from when the recording began, so it is only true
            while the recording is running. Paused, it says nothing rather than
            something wrong. */}
        {!paused && (
          <p className="tabular-nums text-xs text-ink-faint">
            {formatElapsed(Math.max(0, now - state.startedAtMs))}
          </p>
        )}
        {error && <p className="mt-1 truncate text-xs text-live">{error.message}</p>}
      </div>
      <Button variant="primary" size="sm" loading={stopping} onClick={() => void handleStop()}>
        {home.stopButton}
      </Button>
    </div>
  );
}
