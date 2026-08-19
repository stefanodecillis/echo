/**
 * A minimal classname joiner — the one thing this design system needs from a
 * package like `clsx`, kept local so the component library doesn't ask the
 * rest of the app to add a dependency for it.
 */
export type ClassValue =
  | string
  | number
  | null
  | undefined
  | false
  | Record<string, boolean | null | undefined>;

export function cx(...values: ClassValue[]): string {
  const out: string[] = [];
  for (const value of values) {
    if (!value) continue;
    if (typeof value === "string" || typeof value === "number") {
      out.push(String(value));
      continue;
    }
    for (const [key, on] of Object.entries(value)) {
      if (on) out.push(key);
    }
  }
  return out.join(" ");
}
