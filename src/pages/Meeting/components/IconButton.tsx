import { forwardRef } from "react";
import type { ButtonHTMLAttributes, ReactNode } from "react";

import { cx } from "@/components/lib/cx";

export interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  icon: ReactNode;
  /** Required alongside `title` — every icon-only button needs both, since
   * one is for a screen reader and the other is for a mouse hovering over a
   * glyph with no visible word next to it. */
  "aria-label": string;
  title: string;
}

/** A quiet, icon-only square button for a compact toolbar row — the
 * combine-speakers, copy and listen-again actions on the Transcript tab.
 * Same footprint and hover treatment as the icon buttons already used
 * elsewhere (e.g. the delete affordance on a meeting row), just always
 * visible instead of hover-revealed. */
export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(function IconButton(
  { icon, className, disabled, ...props },
  ref,
) {
  return (
    <button
      ref={ref}
      type="button"
      disabled={disabled}
      className={cx(
        "flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-ink-soft outline-none transition-colors",
        "hover:bg-surface-sunken hover:text-ink",
        "focus-visible:ring-2 focus-visible:ring-accent/40",
        "disabled:cursor-not-allowed disabled:text-ink-ghost disabled:hover:bg-transparent",
        className,
      )}
      {...props}
    >
      {/* Icons from the shared set carry no width/height of their own; an
          unsized <svg> falls back to its intrinsic default and draws outside
          this 16px slot — a visibly empty button. Force any svg child to
          fill the slot. */}
      <span
        aria-hidden
        className="flex h-4 w-4 items-center justify-center [&>svg]:h-full [&>svg]:w-full"
      >
        {icon}
      </span>
    </button>
  );
});
