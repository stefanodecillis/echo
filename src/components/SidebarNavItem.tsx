import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";

import { cx } from "./lib/cx";

export interface SidebarNavItemProps {
  to: string;
  label: string;
  icon?: ReactNode;
  end?: boolean;
  badge?: ReactNode;
}

/** One destination in the left rail — an icon, a label, the active state. */
export function SidebarNavItem({ to, label, icon, end, badge }: SidebarNavItemProps) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) =>
        cx(
          "flex items-center gap-2.5 rounded-xl px-3 py-2 text-sm transition-colors select-none",
          isActive
            ? "bg-surface-sunken font-medium text-ink"
            : "text-ink-faint hover:bg-surface-sunken hover:text-ink",
        )
      }
    >
      {icon && (
        <span aria-hidden className="flex h-4 w-4 shrink-0 items-center">
          {icon}
        </span>
      )}
      <span className="flex-1 truncate">{label}</span>
      {badge}
    </NavLink>
  );
}
