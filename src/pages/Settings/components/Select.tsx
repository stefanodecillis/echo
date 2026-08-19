import type { SelectHTMLAttributes } from "react";

import { cx } from "../../../components/lib/cx";
import { ChevronDownIcon } from "../../../components/icons";

export interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {}

/** A native `<select>` dressed to match `Input` — the settings screens need a
 * handful of these (recap language, accuracy default, a model to use) and it
 * isn't part of the shared foundation yet. */
export function Select({ className, children, ...props }: SelectProps) {
  return (
    <div className="relative">
      <select
        className={cx(
          "w-full appearance-none rounded-xl border border-hairline bg-surface px-3 py-2 pr-9 text-sm text-ink",
          "transition-colors focus:outline-none focus:ring-2 focus:ring-accent/30",
          "disabled:cursor-not-allowed disabled:bg-surface-sunken disabled:text-ink-ghost",
          className,
        )}
        {...props}
      >
        {children}
      </select>
      <ChevronDownIcon className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-ghost" />
    </div>
  );
}
