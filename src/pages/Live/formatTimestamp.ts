/** `72_000` -> `"01:12"`. Always minutes:seconds, zero-padded — a transcript
 * timestamp column reads better aligned than `formatElapsed`'s hours-aware
 * header clock does, and a meeting long enough to spill past 99 minutes still
 * renders fine (just a wider column, never a rollover). */
export function formatTimestamp(ms: number): string {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  const pad = (n: number) => n.toString().padStart(2, "0");
  return `${pad(minutes)}:${pad(seconds)}`;
}
