import { useState } from "react";

import { Button } from "../components";
import { cx } from "../components/lib/cx";
import { home, panel } from "../lib/copy";
import { panelClose, startRecording, toUiError } from "../lib/ipc";
import type { PanelState, UiError } from "../lib/types";
import { formatAgo } from "./format";
import { PanelTile } from "./PanelTile";
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
    <>
      <PanelTile motion="radiate" />
      <div className="min-w-0 flex-1">
        <p className="truncate text-[13px] font-semibold leading-tight text-ink">
          {panel.detectedTitle}
        </p>
        {/* One second line, never two: an error takes the place of the clock
            rather than stacking under it, so the card is the same height in
            every state and the window never has to guess. `tabular-nums` keeps
            the seconds from nudging the words sideways as they tick. */}
        <p
          className={cx(
            "mt-1 truncate text-[11.5px] leading-tight tabular-nums",
            error ? "text-live" : "text-ink-faint",
          )}
        >
          {error
            ? error.message
            : panel.detectedMeta(formatAgo(state.detectedAtMs, now), state.appName)}
        </p>
      </div>
      <Button
        variant="primary"
        size="sm"
        className="px-3.5"
        loading={starting}
        onClick={() => void handleStart()}
      >
        {home.startButton}
      </Button>
    </>
  );
}
