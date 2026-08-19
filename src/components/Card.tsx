import type { HTMLAttributes } from "react";

import { cx } from "./lib/cx";

export interface CardProps extends HTMLAttributes<HTMLDivElement> {
  /** Adds a hover lift and a pointer cursor, for a card that's a click target. */
  interactive?: boolean;
  padding?: "none" | "sm" | "md" | "lg";
}

const paddingClasses = {
  none: "",
  sm: "p-3",
  md: "p-5",
  lg: "p-8",
} as const;

/** The one rounded-xl, hairline-bordered surface everything else sits on. */
export function Card({
  interactive = false,
  padding = "md",
  className,
  children,
  ...props
}: CardProps) {
  return (
    <div
      className={cx(
        "echo-card",
        paddingClasses[padding],
        interactive && "cursor-pointer transition-shadow hover:shadow-lift",
        className,
      )}
      {...props}
    >
      {children}
    </div>
  );
}
