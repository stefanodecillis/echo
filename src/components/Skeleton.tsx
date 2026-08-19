import { cx } from "./lib/cx";

export interface SkeletonProps {
  className?: string;
  variant?: "text" | "block" | "circle";
}

/** A quiet placeholder shape while something is loading. Never shown for
 * longer than it takes to fetch — nothing in Echo blocks on a spinner. */
export function Skeleton({ className, variant = "block" }: SkeletonProps) {
  return (
    <div
      aria-hidden
      className={cx(
        "animate-pulse bg-surface-sunken",
        variant === "text" && "h-3 rounded-md",
        variant === "block" && "rounded-xl",
        variant === "circle" && "rounded-full",
        className,
      )}
    />
  );
}

export interface SkeletonLinesProps {
  count?: number;
  className?: string;
}

/** A paragraph-shaped loading state — the recap while it streams in, a
 * transcript segment before its text arrives. */
export function SkeletonLines({ count = 3, className }: SkeletonLinesProps) {
  return (
    <div className={cx("flex flex-col gap-2", className)}>
      {Array.from({ length: count }, (_, i) => (
        <Skeleton key={i} variant="text" className={i === count - 1 ? "w-2/3" : "w-full"} />
      ))}
    </div>
  );
}
