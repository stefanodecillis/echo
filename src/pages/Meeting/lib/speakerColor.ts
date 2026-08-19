/** A small, fixed palette so the same speaker gets the same color every time
 * a transcript renders, without asking the backend to assign one. */
const PALETTE: { bg: string; fg: string }[] = [
  { bg: "#eef2ff", fg: "#4338ca" }, // indigo
  { bg: "#ecfdf5", fg: "#047857" }, // green
  { bg: "#fff7ed", fg: "#c2410c" }, // orange
  { bg: "#fdf2f8", fg: "#be185d" }, // pink
  { bg: "#f0f9ff", fg: "#0369a1" }, // sky
  { bg: "#fefce8", fg: "#854d0e" }, // yellow
  { bg: "#f5f3ff", fg: "#6d28d9" }, // violet
  { bg: "#f0fdfa", fg: "#0f766e" }, // teal
];

function hashString(value: string): number {
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) {
    hash = (hash * 31 + value.charCodeAt(i)) | 0;
  }
  return Math.abs(hash);
}

/** Deterministic color pair (background/foreground) for a speaker id. */
export function speakerColor(speakerId: string): { bg: string; fg: string } {
  return PALETTE[hashString(speakerId) % PALETTE.length];
}
