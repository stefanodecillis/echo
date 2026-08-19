import { forwardRef } from "react";
import type { ButtonHTMLAttributes, ReactNode } from "react";

import { cx } from "./lib/cx";

export type ButtonVariant = "primary" | "secondary" | "ghost";
export type ButtonSize = "sm" | "md";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  /** `primary` = the dark pill, one per screen. `secondary` = the bordered
   * ghost pill. `ghost` = no border, for a quiet inline action. */
  variant?: ButtonVariant;
  size?: ButtonSize;
  leftIcon?: ReactNode;
  rightIcon?: ReactNode;
  /** Shows a spinner in place of `leftIcon` and disables the button. */
  loading?: boolean;
  fullWidth?: boolean;
}

const sizeClasses: Record<ButtonSize, string> = {
  sm: "px-3 py-1.5 text-xs",
  md: "px-4 py-2 text-sm",
};

const variantClasses: Record<ButtonVariant, string> = {
  primary: "bg-ink text-white hover:bg-ink-soft disabled:bg-ink-ghost",
  secondary:
    "border border-hairline text-ink-soft hover:bg-surface-sunken disabled:text-ink-ghost",
  ghost: "text-ink-soft hover:bg-surface-sunken disabled:text-ink-ghost",
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  {
    variant = "secondary",
    size = "md",
    leftIcon,
    rightIcon,
    loading = false,
    fullWidth = false,
    disabled,
    className,
    children,
    ...props
  },
  ref,
) {
  return (
    <button
      ref={ref}
      type={props.type ?? "button"}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={cx(
        "inline-flex items-center justify-center gap-2 rounded-full font-medium transition-colors",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40",
        "disabled:cursor-not-allowed",
        sizeClasses[size],
        variantClasses[variant],
        fullWidth && "w-full",
        className,
      )}
      {...props}
    >
      {loading ? (
        <Spinner light={variant === "primary"} />
      ) : (
        leftIcon && (
          <span aria-hidden className="-ml-0.5 flex h-4 w-4 shrink-0 items-center">
            {leftIcon}
          </span>
        )
      )}
      {children}
      {!loading && rightIcon && (
        <span aria-hidden className="-mr-0.5 flex h-4 w-4 shrink-0 items-center">
          {rightIcon}
        </span>
      )}
    </button>
  );
});

function Spinner({ light }: { light?: boolean }) {
  return (
    <svg
      className={cx("h-3.5 w-3.5 animate-spin", light ? "text-white" : "text-ink-faint")}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden
    >
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
    </svg>
  );
}
