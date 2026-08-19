import type { ReactNode } from "react";

import { Card, CloseIcon } from "../components";
import { common } from "../lib/copy";

export interface PanelShellProps {
  children: ReactNode;
  /** Present only where the design calls for the ✕ (the detected state). */
  onDismiss?: () => void;
}

/**
 * The floating panel's one painted surface.
 *
 * The window itself is transparent (Rust's job); this card is what actually
 * draws — rounded, hairlined, softly shadowed, sized to fill the ~340x120
 * window exactly. Everything inside must fit without scrolling, since a
 * scrollbar on a window this size would be its own small disaster.
 */
export function PanelShell({ children, onDismiss }: PanelShellProps) {
  return (
    <Card
      padding="sm"
      className="relative box-border flex h-full w-full items-center rounded-2xl shadow-lift"
    >
      {onDismiss && (
        <button
          type="button"
          aria-label={common.dismiss}
          onClick={onDismiss}
          className="absolute right-1.5 top-1.5 flex h-5 w-5 items-center justify-center rounded-full text-ink-faint transition-colors hover:bg-surface-sunken hover:text-ink"
        >
          <CloseIcon className="h-3 w-3" />
        </button>
      )}
      <div className="min-w-0 flex-1">{children}</div>
    </Card>
  );
}
