import type { ReactNode } from "react";

import { cx } from "./lib/cx";

export interface EmptyStateProps {
  icon?: ReactNode;
  title: string;
  description?: string;
  action?: ReactNode;
  className?: string;
}

/** No meetings yet, no results, nothing captured — the same quiet shape
 * everywhere in the app so an empty screen never looks like a broken one. */
export function EmptyState({ icon, title, description, action, className }: EmptyStateProps) {
  return (
    <div className={cx("flex flex-col items-center gap-3 px-6 py-16 text-center", className)}>
      {icon && (
        <div aria-hidden className="text-ink-ghost">
          {icon}
        </div>
      )}
      <h3 className="text-base font-semibold text-ink">{title}</h3>
      {description && <p className="max-w-sm text-sm text-ink-faint">{description}</p>}
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}
