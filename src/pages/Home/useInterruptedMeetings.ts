import { useCallback, useEffect, useState } from "react";

import { useEvent } from "../../hooks";
import { EVENTS, listInterruptedMeetings, resolveInterruptedMeeting } from "../../lib/ipc";
import type { Id, Meeting, RecoveryAction } from "../../lib/types";

export interface InterruptedMeetingsState {
  meetings: Meeting[];
  /** The id currently being resolved, so its row can show a spinner instead
   * of letting a second click queue up behind the first. */
  resolving: Id | null;
  resolve: (meetingId: Id, action: RecoveryAction) => Promise<void>;
}

/**
 * Meetings the core found interrupted at launch — a crash, a forced quit, a
 * dead battery — that still need a Finish/Let-it-go decision from the person.
 *
 * Silent on failure: an empty list is always a safe fallback here, the same
 * way `useCaptureState` stays on its idle default rather than surfacing a
 * fetch error nobody asked about.
 */
export function useInterruptedMeetings(): InterruptedMeetingsState {
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [resolving, setResolving] = useState<Id | null>(null);

  const refresh = useCallback(() => {
    listInterruptedMeetings()
      .then(setMeetings)
      .catch(() => setMeetings([]));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEvent(EVENTS.recoveryAvailable, refresh);

  const resolve = useCallback(async (meetingId: Id, action: RecoveryAction) => {
    setResolving(meetingId);
    try {
      await resolveInterruptedMeeting(meetingId, action);
      setMeetings((prev) => prev.filter((m) => m.id !== meetingId));
    } finally {
      setResolving(null);
    }
  }, []);

  return { meetings, resolving, resolve };
}
