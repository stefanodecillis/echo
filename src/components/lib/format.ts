/** Small formatting helpers shared by components. Not copy — no words here,
 * just how numbers and durations are shaped for display. */

/** `65_000` -> `"1:05"`. Hours appear only once the meeting runs that long. */
export function formatElapsed(ms: number): string {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  const pad = (n: number) => n.toString().padStart(2, "0");
  return hours > 0
    ? `${hours}:${pad(minutes)}:${pad(seconds)}`
    : `${minutes}:${pad(seconds)}`;
}

/**
 * `4_308_357_836` -> `"4.3 GB"`. Ten and above loses the decimal.
 *
 * Decimal, not binary. Two reasons, and the first is the one that matters: the
 * core describes the same download in `catalog::human_bytes`, which is decimal
 * because that is how a download is normally quoted — this used to divide by
 * 1024 and the two halves of the UI disagreed about how big the same file was
 * (4 GB here, 4.3 GB there). The second is that macOS itself has counted this
 * way since 10.6, so a number Echo shows matches the one Finder shows.
 *
 * Kept in step with `human_bytes`: change one and change the other.
 */
export function formatBytes(bytes: number): string {
  if (bytes < 1000) return `${bytes} B`;
  const units = ["kB", "MB", "GB", "TB"];
  let value = bytes / 1000;
  let unitIndex = 0;
  while (value >= 1000 && unitIndex < units.length - 1) {
    value /= 1000;
    unitIndex += 1;
  }
  const rounded = value >= 10 ? Math.round(value) : Math.round(value * 10) / 10;
  return `${rounded} ${units[unitIndex]}`;
}
