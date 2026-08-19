import { forwardRef } from "react";
import type { InputHTMLAttributes, LabelHTMLAttributes, ReactNode } from "react";

import { cx } from "./lib/cx";

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  invalid?: boolean;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(function Input(
  { invalid = false, className, ...props },
  ref,
) {
  return (
    <input
      ref={ref}
      aria-invalid={invalid || undefined}
      className={cx(
        "w-full rounded-xl border bg-surface px-3 py-2 text-sm text-ink transition-colors",
        "placeholder:text-ink-ghost focus:outline-none focus:ring-2 focus:ring-accent/30",
        invalid ? "border-live/60 focus:ring-live/20" : "border-hairline focus:border-ink-ghost",
        "disabled:cursor-not-allowed disabled:bg-surface-sunken disabled:text-ink-ghost",
        className,
      )}
      {...props}
    />
  );
});

export interface FieldProps extends LabelHTMLAttributes<HTMLLabelElement> {
  label: string;
  hint?: string;
  error?: string;
  children: ReactNode;
}

/** A label, one control, and an optional hint or error line beneath it. */
export function Field({ label, hint, error, children, className, ...props }: FieldProps) {
  return (
    <label className={cx("flex flex-col gap-1.5", className)} {...props}>
      <span className="text-sm font-medium text-ink">{label}</span>
      {children}
      {error ? (
        <span className="text-xs text-live">{error}</span>
      ) : hint ? (
        <span className="text-xs text-ink-faint">{hint}</span>
      ) : null}
    </label>
  );
}
