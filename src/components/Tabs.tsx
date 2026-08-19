import type { ReactNode } from "react";

import { cx } from "./lib/cx";

export interface TabItem {
  id: string;
  label: string;
  icon?: ReactNode;
}

export interface TabsProps {
  items: TabItem[];
  value: string;
  onChange: (id: string) => void;
  className?: string;
}

/** Underline-style tabs — Recap / Transcript / Info, and the like. */
export function Tabs({ items, value, onChange, className }: TabsProps) {
  return (
    <div role="tablist" className={cx("flex gap-6 border-b border-hairline", className)}>
      {items.map((item) => {
        const active = item.id === value;
        return (
          <button
            key={item.id}
            type="button"
            role="tab"
            aria-selected={active}
            onClick={() => onChange(item.id)}
            className={cx(
              "relative flex items-center gap-1.5 pb-3 text-sm font-medium transition-colors",
              "focus-visible:outline-none",
              active ? "text-ink" : "text-ink-faint hover:text-ink-soft",
            )}
          >
            {item.icon}
            {item.label}
            <span
              aria-hidden
              className={cx(
                "absolute inset-x-0 -bottom-px h-0.5 rounded-full transition-colors",
                active ? "bg-ink" : "bg-transparent",
              )}
            />
          </button>
        );
      })}
    </div>
  );
}
