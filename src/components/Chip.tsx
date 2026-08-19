import type { HTMLAttributes, ReactNode } from "react";

import { cx } from "./lib/cx";

export type ChipVariant = "neutral" | "solid" | "outline" | "danger";

export interface ChipProps extends HTMLAttributes<HTMLSpanElement> {
  /** `neutral` (default) for language/status chips, `solid` for "You" style
   * speaker chips, `outline` for a quieter status, `danger` for failed/error. */
  variant?: ChipVariant;
  icon?: ReactNode;
}

const variantClasses: Record<ChipVariant, string> = {
  neutral: "bg-surface-sunken text-ink-faint",
  solid: "bg-ink text-white",
  outline: "border border-hairline text-ink-soft",
  danger: "bg-live/10 text-live",
};

/** A small rounded-full label — language, speaker, or job status. */
export function Chip({ variant = "neutral", icon, className, children, ...props }: ChipProps) {
  return (
    <span className={cx("echo-chip", variantClasses[variant], className)} {...props}>
      {icon && (
        <span aria-hidden className="-ml-0.5 flex h-3 w-3 shrink-0 items-center">
          {icon}
        </span>
      )}
      {children}
    </span>
  );
}
