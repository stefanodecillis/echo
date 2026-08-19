/**
 * Small formatting helpers for the Home screen's recent-meetings list.
 *
 * Not copy — see `src/lib/copy.ts` for words a person reads. This is only how
 * a timestamp or a duration in milliseconds gets shaped into digits.
 */
import type { Timestamp } from "../../lib/types";

const MINUTE = 60_000;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

/**
 * `"2026-08-19T09:00:00Z"` -> `"3h ago"`, `"Yesterday"`, or `"Aug 12"`.
 *
 * `now` is a parameter (not `Date.now()` inline) so a snapshot test can hold
 * time still — the recent-meetings list re-renders on every capture-state and
 * meeting-updated event, and a relative label that silently drifts across
 * those renders would be a subtler bug than a wrong-but-stable one.
 */
export function formatRelativeDate(startedAt: Timestamp, now: Date = new Date()): string {
  const then = new Date(startedAt);
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

/** `2_700_000` -> `"45 min"`, `4_320_000` -> `"1h 12m"`, `12_000` -> `"< 1 min"`. */
export function formatDuration(ms: number): string {
  if (ms < 30_000) return "< 1 min";
  const totalMinutes = Math.round(ms / MINUTE);
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  if (hours === 0) return `${minutes} min`;
  return minutes === 0 ? `${hours}h` : `${hours}h ${minutes}m`;
}
