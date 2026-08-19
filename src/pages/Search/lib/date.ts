/** `"2026-08-19T14:30:00Z"` -> `"Aug 19, 2026"`. Falls back to the raw
 * string rather than showing "Invalid Date" for something a person never
 * asked to see rendered. */
export function formatResultDate(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

/** `123_456` ms -> `"2:03"`, for the timestamp next to a search hit. */
export function formatTimestamp(ms: number): string {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}
