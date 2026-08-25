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

/**
 * Is this still a label Echo made up, rather than a name somebody typed?
 *
 * The mirror of `is_default_name` in `src-tauri/src/diarize/people.rs`: the
 * separation pass names the voices it finds "Speaker 1", "Speaker 2"… and the
 * microphone channel "You", and both are placeholders standing in for a name
 * nobody has given yet. Anything else in a speaker's name field was chosen by
 * a person.
 *
 * Written here rather than imported from anywhere because the two sides are
 * different languages; the strings are pinned by the backend tests either side
 * of `display_name`.
 */
export function isEchoOwnLabel(name: string): boolean {
  const trimmed = name.trim();
  return trimmed === "You" || /^Speaker \d+$/.test(trimmed);
}
