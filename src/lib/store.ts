/**
 * The one piece of client state that more than one screen needs at once.
 *
 * Everything else — a form draft, a tab selection, a search query — belongs in
 * the component that owns it. This store exists for the three things that
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
 */

import { create } from "zustand";

import type { CaptureStatus, Id, NoticeLevel } from "./types";

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
