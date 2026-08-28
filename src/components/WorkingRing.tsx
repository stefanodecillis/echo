import "../styles/animations.css";

import { cx } from "./lib/cx";

export interface WorkingRingProps {
  size?: "sm" | "md";
  className?: string;
  /** Read aloud in place of the shape, which has no words of its own. */
  label?: string;
}

const sizeClasses = {
  sm: "h-3 w-3",
  md: "h-4 w-4",
} as const;

/**
 * The "still working on this" mark: a quiet ring with one darker quadrant,
 * travelling.
 *
 * A sibling to `RecordingDot`, with the same props for the same reason — the two
 * are the app's pair of "something is happening" marks and a screen picks one.
 * They must never be mistaken for each other: a recording is red and pulses, and
 * this is monochrome and goes round. That is the distinction docs/DESIGN.md §4
 * asks the tray to make, made the same way here.
 *
 * Monochrome on purpose. There is no green or amber in this app, and `live` red
 * means recording or error — so background work, which is neither, gets no
 * colour at all and is told apart by its motion and by the words beside it.
 */
export function WorkingRing({ size = "sm", className, label }: WorkingRingProps) {
  return (
    <span className={cx("relative inline-flex shrink-0", sizeClasses[size], className)}>
      <span
        aria-hidden
        className="h-full w-full rounded-full border-[1.5px] border-hairline border-t-ink-soft animate-working-arc"
      />
      {label && <span className="sr-only">{label}</span>}
    </span>
  );
}
