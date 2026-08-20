import type { ReactNode } from "react";

import { CloseIcon } from "../components";
import { common } from "../lib/copy";
import "../styles/panel.css";

export interface PanelShellProps {
  /** The card's one row, left to right: tile, lines, action. */
  children: ReactNode;
  /** Present only where the design calls for the ✕ (the detected state). */
  onDismiss?: () => void;
}

/**
 * The floating panel's one painted surface.
 *
 * The window itself is see-through (`index.css` drops the background for
 * `?window=panel`, Rust makes the window transparent), so this card is the only
 * thing anyone sees: a rounded, hairlined, two-layer-shadowed strip. It is the
 * row too — children are its items rather than a nested box — so a state can
 * never accidentally add a second layer of padding inside it.
 *
 * The window is sized to this card plus the few pixels its shadow needs, which
 * is why nothing here scrolls or stretches: at this size a scrollbar would be
 * its own small disaster, and empty space below the card would read as a bug.
 */
export function PanelShell({ children, onDismiss }: PanelShellProps) {
  return (
    <div className="flex h-full w-full items-start justify-center px-2 pb-3 pt-2">
      <div className="animate-panel-in echo-panel-card flex w-full items-center gap-3 rounded-2xl border border-hairline bg-surface p-3.5">
        {children}
        {onDismiss && (
          <button
            type="button"
            aria-label={common.dismiss}
            onClick={onDismiss}
            /* Pulled into the card's top-right corner, but still its own column
               in the row: nothing can ever end up underneath it. */
            className="-mr-1 -mt-1 flex h-6 w-6 shrink-0 items-center justify-center self-start rounded-lg text-ink-ghost transition-colors hover:bg-surface-sunken hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40"
          >
            <CloseIcon className="h-3.5 w-3.5" />
          </button>
        )}
      </div>
    </div>
  );
}
