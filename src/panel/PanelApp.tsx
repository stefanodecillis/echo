import { useEffect, useState } from "react";

import { EVENTS, getPanelState, on, panelClose, panelDismiss } from "../lib/ipc";
import type { PanelState } from "../lib/types";
import { DetectedPanel } from "./DetectedPanel";
import { PanelShell } from "./PanelShell";
import { RecordingPanel } from "./RecordingPanel";

/**
 * The panel window's whole app: one component switching on the state Rust
 * hands it when showing the window. No router — there is exactly one thing
 * to show, chosen by `state.kind`, never a screen a person navigates to.
 */
export function PanelApp() {
  const [state, setState] = useState<PanelState | null>(null);

  // The one and only event this window ever receives. Rust emits it targeted at
  // this window each time it shows the panel.
  //
  // The one ask on mount is for the first show only: the window is built and
  // shown in the same breath, so the very first event can land before this
  // webview has a listener. After that the window is reused and the event is
  // always in time.
  useEffect(() => {
    let stop: (() => void) | undefined;
    let cancelled = false;
    on(EVENTS.panelState, setState).then((fn) => {
      if (cancelled) fn();
      else stop = fn;
    });
    getPanelState()
      .then((showing) => {
        // Never overwrite a live event with the answer to a question asked
        // earlier: whatever arrived by event is newer than this.
        if (!cancelled && showing) setState((current) => current ?? showing);
      })
      .catch(() => {
        // The event is the real path; this was only the safety net.
      });
    return () => {
      cancelled = true;
      stop?.();
    };
  }, []);

  // Escape means the same as the ✕ on the detected panel — "not this meeting" —
  // and only "put this away" on the recording one: pressing Escape over a
  // running recording must not quietly mute the nudge for a later meeting.
  const kind = state?.kind;
  useEffect(() => {
    if (!kind) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.preventDefault();
      const away = kind === "detected" ? panelDismiss : panelClose;
      void away().catch(() => {
        // Nothing useful to show for a failed dismiss on a window this
        // small; worst case the person clicks the ✕ instead.
      });
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [kind]);

  // Nothing to draw yet: stay fully transparent rather than flash an empty
  // card while waiting for the first event.
  if (!state) return null;

  const handleDismiss = () => void panelDismiss().catch(() => {});

  return (
    <PanelShell onDismiss={state.kind === "detected" ? handleDismiss : undefined}>
      {state.kind === "detected" ? (
        <DetectedPanel state={state} />
      ) : (
        <RecordingPanel state={state} />
      )}
    </PanelShell>
  );
}
