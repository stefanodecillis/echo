/** `"2026-08-19T14:30:00Z"` -> `"Aug 19, 2026 · 2:30 PM"`. Falls back to the
 * raw string if it somehow isn't a valid timestamp, rather than showing
 * "Invalid Date" to a person who never asked for one. */
export function formatMeetingDate(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const day = date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
  const time = date.toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
  });
  return `${day} · ${time}`;
}

/** `65_000` -> `"1m 05s"` at low durations, `"1h 02m"` once it runs an hour
 * or more — the mm:ss digital-clock shape reads as "still counting", which
 * a finished meeting's length is not. */
export function formatDuration(ms: number): string {
  const totalSeconds = Math.max(0, Math.round(ms / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) return `${hours}h ${minutes.toString().padStart(2, "0")}m`;
  if (minutes > 0) return `${minutes}m ${seconds.toString().padStart(2, "0")}s`;
  return `${seconds}s`;
}

/** `123_456` ms -> `"2:03"`, for a timestamp next to a transcript line. */
export function formatTimestamp(ms: number): string {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}
