import { useState } from "react";

import { Button, RecordingDot } from "../components";
import { formatElapsed } from "../components/lib/format";
import { anchors, home, labels, panel } from "../lib/copy";
import { panelClose, stopRecording, toUiError } from "../lib/ipc";
import type { PanelState, UiError } from "../lib/types";
import { PanelTile } from "./PanelTile";
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
    <>
      {/* Paused holds still: the tile only breathes while sound is actually
          being kept. */}
      <PanelTile motion={paused ? "still" : "breathe"} />
      <div className="min-w-0 flex-1">
        <p className="truncate text-[13px] font-semibold leading-tight text-ink">
          {paused ? labels.captureState.paused : anchors.listening}
        </p>
        {/* One second line in every state, so the card's height never moves:
            the error if there is one, otherwise the clock — or, paused, the
            plain fact that nothing is being kept. The clock counts from when
            the recording began, so it is only true while it is running. */}
        <div className="mt-1 flex min-w-0 items-center gap-1.5">
          {error ? (
            <p className="truncate text-[11.5px] leading-tight text-live">{error.message}</p>
          ) : paused ? (
            <p className="truncate text-[11.5px] leading-tight text-ink-faint">
              {panel.pausedHint}
            </p>
          ) : (
            <>
              {/* No label: the line above already says "Listening…", and a
                  screen reader should hear it once. */}
              <RecordingDot size="sm" />
              <p className="text-[11.5px] leading-tight tabular-nums text-ink-faint">
                {formatElapsed(Math.max(0, now - state.startedAtMs))}
              </p>
            </>
          )}
        </div>
      </div>
      <Button
        variant="primary"
        size="sm"
        className="px-3.5"
        loading={stopping}
        onClick={() => void handleStop()}
      >
        {home.stopButton}
      </Button>
    </>
  );
}
