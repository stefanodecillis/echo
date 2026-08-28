/**
 * The one piece of client state that more than one screen needs at once.
 *
 * Everything else — a form draft, a tab selection, a search query — belongs in
 * the component that owns it. This store exists for the four things that
 * genuinely cross component boundaries:
 *
 *  - `captureState`: so the sidebar's live indicator and the Live screen and
 *    a "Stop" button somewhere else all agree on what's happening, without
 *    each of them polling `get_capture_state` on their own. Kept fresh by the
 *    `useCaptureState` hook (`src/hooks/useCaptureState.ts`).
 *  - `activeMeetingId`: which meeting the user is currently looking at, so a
 *    toast or a background job update can say "this meeting" without a prop
 *    threaded down from the router.
 *  - `toasts`: the banners the app raises, whether they came from the backend
 *    `notice` event (wired up in `App.tsx`) or from a screen reacting to its
 *    own action (e.g. "Copied.").
 *  - `activeJobsByMeeting` / `failedJobsByMeeting`: which meetings still have
 *    background work outstanding, so a row in any list, the rail's dot and the
 *    corner pill all agree without each of them asking the core on its own.
 *    Kept fresh by the `useActiveJobs` hook (`src/hooks/useActiveJobs.ts`).
 */

import { create } from "zustand";

import type { CaptureStatus, Id, Job, NoticeLevel } from "./types";

export interface Toast {
  id: string;
  level: NoticeLevel;
  message: string;
  /** Stays until dismissed by hand, instead of fading on its own. */
  persistent?: boolean;
  /** A repeat with the same tag replaces the old toast instead of stacking. */
  tag?: string;
  meetingId?: Id;
}

const IDLE_CAPTURE_STATE: CaptureStatus = {
  state: "idle",
  elapsedMs: 0,
  activeChannels: [],
  pendingUtterances: 0,
  speech: "idle",
};

function makeToastId(): string {
  return `toast_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

export interface EchoStore {
  captureState: CaptureStatus;
  setCaptureState: (state: CaptureStatus) => void;

  activeMeetingId: Id | null;
  setActiveMeetingId: (id: Id | null) => void;

  /**
   * Every unfinished job in the app, grouped by the meeting it belongs to.
   *
   * `undefined` until the first read lands, which is not the same as "no work" —
   * and the difference is the whole reason it is nullable. A row that cannot
   * tell "nothing is happening" from "nobody has asked yet" would claim one of
   * them on first paint, and it would be wrong roughly as often as it was right.
   *
   * Jobs with no `meetingId` are deliberately absent: the one-time download and
   * getting the engine ready belong to the corner pill, not to a meeting.
   */
  activeJobsByMeeting: Record<Id, Job[]> | undefined;
  /**
   * The failed rows each meeting still carries.
   *
   * Separate from the map above because a failure is not work: it is the reason
   * there is none. It is also only ever a partial history — enough to explain a
   * meeting that has stopped without finishing, never enough to be asked "what
   * has ever gone wrong here".
   */
  failedJobsByMeeting: Record<Id, Job[]> | undefined;
  /**
   * Replaces either map, or both.
   *
   * Whole-map replacement rather than a per-job setter on purpose: the read it
   * is fed from (`listJobs({ activeOnly: true })`) is the authoritative set of
   * unfinished work, and anything the client is holding that the core did not
   * name is a job that ended without saying so. Merging would keep that ghost
   * on screen for the rest of the session.
   */
  setJobWork: (next: { active?: Record<Id, Job[]>; failed?: Record<Id, Job[]> }) => void;

  toasts: Toast[];
  /** Returns the toast's id, e.g. so a caller can dismiss it early. */
  addToast: (toast: Omit<Toast, "id"> & { id?: string }) => string;
  dismissToast: (id: string) => void;
  clearToasts: () => void;
}

export const useEchoStore = create<EchoStore>((set, get) => ({
  captureState: IDLE_CAPTURE_STATE,
  setCaptureState: (state) => set({ captureState: state }),

  activeMeetingId: null,
  setActiveMeetingId: (id) => set({ activeMeetingId: id }),

  activeJobsByMeeting: undefined,
  failedJobsByMeeting: undefined,
  setJobWork: (next) =>
    set({
      ...(next.active !== undefined ? { activeJobsByMeeting: next.active } : {}),
      ...(next.failed !== undefined ? { failedJobsByMeeting: next.failed } : {}),
    }),

  toasts: [],
  addToast: (toast) => {
    const id = toast.id ?? makeToastId();
    const withoutDupes = toast.tag
      ? get().toasts.filter((t) => t.tag !== toast.tag)
      : get().toasts;
    set({ toasts: [...withoutDupes, { ...toast, id }] });
    return id;
  },
  dismissToast: (id) => set({ toasts: get().toasts.filter((t) => t.id !== id) }),
  clearToasts: () => set({ toasts: [] }),
}));

/** Non-hook access to the current capture state, e.g. from a plain callback. */
export function getCaptureStateSnapshot(): CaptureStatus {
  return useEchoStore.getState().captureState;
}
