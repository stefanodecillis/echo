/**
 * "Last heard <relative date>" for a saved voice — same shape as the
 * meetings list's relative date (`Home/format.ts`) so time reads the same
 * everywhere, kept local rather than imported to stay inside this screen's
 * own ownership.
 */
const MINUTE = 60_000;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

/** `"2026-08-19T09:00:00Z"` -> `"3h ago"`, `"Yesterday"`, `"Aug 12"`. Falls
 * back to the raw string rather than "Invalid Date" for something nobody
 * asked to see rendered. */
export function formatLastHeard(at: string, now: Date = new Date()): string {
  const then = new Date(at);
  if (Number.isNaN(then.getTime())) return at;
  const diffMs = now.getTime() - then.getTime();

  if (diffMs < MINUTE) return "Just now";
  if (diffMs < HOUR) return `${Math.floor(diffMs / MINUTE)}m ago`;
  if (diffMs < DAY) return `${Math.floor(diffMs / HOUR)}h ago`;

  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const startOfThen = new Date(then.getFullYear(), then.getMonth(), then.getDate());
  const dayDiff = Math.round((startOfToday.getTime() - startOfThen.getTime()) / DAY);

  if (dayDiff === 1) return "Yesterday";
  if (dayDiff > 1 && dayDiff < 7) return `${dayDiff}d ago`;

  return then.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: then.getFullYear() === now.getFullYear() ? undefined : "numeric",
  });
}
