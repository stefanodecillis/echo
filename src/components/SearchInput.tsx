import { forwardRef } from "react";
import type { InputHTMLAttributes, KeyboardEvent } from "react";

import { SearchIcon } from "./icons";
import { cx } from "./lib/cx";

export interface SearchInputProps
  extends Omit<InputHTMLAttributes<HTMLInputElement>, "type" | "size"> {
  /** Fires on Enter, with the current value — for a search that navigates
   * rather than filtering as you type. */
  onSubmitValue?: (value: string) => void;
  containerClassName?: string;
}

/** The rounded-full "What would you like to do?" bar from the reference. */
export const SearchInput = forwardRef<HTMLInputElement, SearchInputProps>(
  function SearchInput(
    { className, containerClassName, onSubmitValue, onKeyDown, ...props },
    ref,
  ) {
    const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
      onKeyDown?.(event);
      if (event.key === "Enter") onSubmitValue?.(event.currentTarget.value);
    };

    return (
      <div
        className={cx(
          "flex w-full items-center gap-2.5 rounded-full border border-hairline bg-surface px-4 py-3",
          "shadow-card transition-colors focus-within:border-ink-ghost",
          containerClassName,
        )}
      >
        <SearchIcon className="h-4 w-4 shrink-0 text-ink-ghost" />
        <input
          ref={ref}
          type="text"
          className={cx(
            "w-full bg-transparent text-sm text-ink placeholder:text-ink-ghost focus:outline-none",
            className,
          )}
          onKeyDown={handleKeyDown}
          {...props}
        />
      </div>
    );
  },
);
