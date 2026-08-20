import type { MeetingDetail, SpeakersUpdatedPayload } from "@/lib/types";

/**
 * These started life as local widenings written against the agreed contract
 * before the shared types carried the fields. The shared `MeetingDetail` and
 * `SpeakersUpdatedPayload` now declare `peopleCount`/`peopleCountIsOverride`
 * for real, so both names are plain aliases kept only so the control's
 * imports read the same.
 */
export type MeetingDetailWithPeopleCount = MeetingDetail;
export type SpeakersUpdatedPayloadWithPeopleCount = SpeakersUpdatedPayload;

/** The editor only ever offers a whole number of people in this range —
 * matches the offline pass, which needs a concrete count to constrain to. */
export const MIN_PEOPLE_COUNT = 1;
export const MAX_PEOPLE_COUNT = 12;
