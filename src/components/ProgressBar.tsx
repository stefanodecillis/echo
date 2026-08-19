import "../styles/animations.css";

import { cx } from "./lib/cx";

export interface ProgressBarProps {
  /** 0..1. Omit for an indeterminate bar (job queued, waiting to hear back). */
  value?: number;
  className?: string;
  /** Read by screen readers; the visible label, if any, lives next to the bar. */
  label?: string;
}

/** A thin bar, for downloads and background jobs — never a spinner-heavy
 * modal, per docs/DESIGN.md: recording keeps going, nothing here blocks it. */
export function ProgressBar({ value, className, label }: ProgressBarProps) {
  const indeterminate = value === undefined;
  const pct = indeterminate ? 0 : Math.round(Math.max(0, Math.min(1, value)) * 100);

  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuenow={indeterminate ? undefined : pct}
      aria-valuemin={0}
      aria-valuemax={100}
      className={cx("h-1 w-full overflow-hidden rounded-full bg-surface-sunken", className)}
    >
      <div
        className={cx(
          "h-full rounded-full bg-ink",
          indeterminate ? "w-1/3 animate-progress-indeterminate" : "transition-[width] duration-300",
        )}
        style={indeterminate ? undefined : { width: `${pct}%` }}
      />
    </div>
  );
}
