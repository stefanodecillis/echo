/**
 * Settings > People: the section's local vocabulary for known voices.
 *
 * This file used to carry its own copy of the known-people contract, written
 * against the shape this screen had been given while `src/lib/ipc.ts` was
 * still somebody else's to edit. That contract has landed for real, so there
 * is now exactly one definition of it and this file is a re-export of it —
 * which is the point: two hand-written copies of the same IPC contract
 * compile perfectly and disagree at runtime, and this one did (it asked for
 * `displayName` where the core sends `name`, and sent `displayName` as the
 * argument to `rename_person`, which takes `name`).
 *
 * Everything real lives in `@/lib/ipc` and `@/lib/types`. What stays here is
 * `ListenState`, which is this screen's own idea about a play button and no
 * part of any contract.
 */

export {
  acceptSuggestedPerson,
  deletePerson,
  listPeople,
  mergePeople,
  personSampleAudio,
  renamePerson,
  suggestedPeople,
} from "@/lib/ipc";

/**
 * A saved voice, as this screen shows it: a name, when it was last heard, how
 * many samples are behind it, and whether Echo is currently able to use them.
 * Nothing about how matching works ever reaches this screen.
 */
export type { PersonInfo as Person, SuggestedPerson } from "@/lib/types";

/** Whether this row's sample is idle, being fetched, or playing. */
export type ListenState = "idle" | "loading" | "playing";
