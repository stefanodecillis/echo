import { cx } from "./lib/cx";

export interface RecordingDotProps {
  size?: "sm" | "md";
  className?: string;
  /** Screen-reader text; visually the dot speaks for itself. */
  label?: string;
  /** A steady dot with no pulse, for the paused state. */
  paused?: boolean;
}

const sizeClasses = { sm: "h-1.5 w-1.5", md: "h-2.5 w-2.5" } as const;

/** The pulsing red dot that means "Echo is listening right now". */
export function RecordingDot({ size = "md", className, label, paused = false }: RecordingDotProps) {
  return (
    <span className={cx("relative inline-flex shrink-0", sizeClasses[size], className)}>
      {!paused && (
        <span
          aria-hidden
          className="absolute inline-flex h-full w-full animate-ping rounded-full bg-live opacity-60"
        />
      )}
      <span
        aria-hidden
        className={cx("relative inline-flex rounded-full bg-live", sizeClasses[size], paused && "opacity-50")}
      />
      {label && <span className="sr-only">{label}</span>}
    </span>
  );
}
