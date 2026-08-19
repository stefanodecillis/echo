import type { Speaker } from "@/lib/types";

/**
 * Follows a merged speaker's `aliasOf` chain to the canonical speaker whose
 * name should actually be shown. Merging is non-destructive (see
 * `mergeSpeakers` in `lib/ipc.ts`): the merged-away row still exists, it just
 * points elsewhere, so anything that renders a speaker has to resolve it.
 *
 * Bounded by the list length so a corrupt or cyclic alias chain can't hang
 * the renderer.
 */
export function resolveSpeaker(
  speakerId: string | undefined,
  speakers: Speaker[],
): Speaker | undefined {
  if (!speakerId) return undefined;
  const byId = new Map(speakers.map((s) => [s.id, s]));
  let current = byId.get(speakerId);
  let hops = 0;
  while (current?.aliasOf && hops < speakers.length) {
    current = byId.get(current.aliasOf);
    hops += 1;
  }
  return current;
}

/** Speakers a person can pick from — merged-away rows are represented by
 * whatever they were folded into, so they don't show up twice. */
export function canonicalSpeakers(speakers: Speaker[]): Speaker[] {
  return speakers.filter((s) => !s.aliasOf);
}
