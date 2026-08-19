/**
 * Not copy — see `src/lib/copy.ts` for words a person reads. This is only how
 * a millisecond timestamp gets shaped into the panel's ticking "ago" line.
 * Mirrors `src/pages/Home/format.ts`'s `formatRelativeDate`, trimmed to the
 * panel's much shorter horizon: a detected meeting is acted on or dismissed
 * within minutes, never days, so there is no need for an hour/day/date tier.
 */

const MINUTE_MS = 60_000;

/** `(now - sinceMs)` -> `"7s ago"` under a minute, `"3m ago"` after. */
export function formatAgo(sinceMs: number, nowMs: number): string {
  const diffMs = Math.max(0, nowMs - sinceMs);
  if (diffMs < MINUTE_MS) return `${Math.floor(diffMs / 1000)}s ago`;
  return `${Math.floor(diffMs / MINUTE_MS)}m ago`;
}
