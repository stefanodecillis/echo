import type { ReactNode } from "react";

import { cx } from "../../../components/lib/cx";

export interface SettingRowProps {
  title: string;
  description?: string;
  control: ReactNode;
  /** Stacks the control under the text instead of beside it, for a control
   * that needs the full row width (a folder path, a dropdown with a label). */
  stacked?: boolean;
  className?: string;
}

/** One labelled setting: a title, an optional plain-language description, and
 * whatever control it needs — a toggle, a button, a select. The same shape
 * every section in Settings uses, so the page reads as one system. */
export function SettingRow({
  title,
  description,
  control,
  stacked = false,
  className,
}: SettingRowProps) {
  return (
    <div
      className={cx(
        "flex gap-4 py-4",
        stacked ? "flex-col" : "items-center justify-between",
        className,
      )}
    >
      <div className="min-w-0">
        <p className="text-sm font-medium text-ink">{title}</p>
        {description && (
          <p className="mt-0.5 text-xs text-ink-faint">{description}</p>
        )}
      </div>
      <div className={cx(stacked ? "w-full" : "shrink-0")}>{control}</div>
    </div>
  );
}

/** A hairline-separated stack of `SettingRow`s inside a `Card`. */
export function SettingList({ children }: { children: ReactNode }) {
  return <div className="divide-y divide-hairline">{children}</div>;
}
