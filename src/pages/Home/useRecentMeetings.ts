import { useEffect, useState } from "react";

import { useEvent } from "../../hooks";
import { EVENTS, listMeetings } from "../../lib/ipc";
import type { Id, MeetingSummary, UiError } from "../../lib/types";

const RECENT_LIMIT = 6;

export interface RecentMeetingsState {
  meetings: MeetingSummary[];
  loading: boolean;
  error: UiError | undefined;
  reload: () => void;
  /** Drops a row the instant it's deleted, so it never lingers for the length
   * of a round trip. The core also announces the delete as `meetingUpdated`,
   * which reloads the list right behind this — this is the fast half. */
  removeMeeting: (meetingId: Id) => void;
}

/**
 * The handful of most recent meetings shown on Home. The full, searchable
 * list lives at `/search`.
 *
 * Refetches whenever a meeting's status or duration changes — so a recording
 * that just stopped picks up its real length, and later its recap snippet,
 * without the person having to leave and come back.
 */
export function useRecentMeetings(): RecentMeetingsState {
  const [meetings, setMeetings] = useState<MeetingSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<UiError>();
  const [reloadToken, setReloadToken] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(undefined);
    listMeetings({ limit: RECENT_LIMIT })
      .then((result) => {
        if (!cancelled) setMeetings(result);
      })
      .catch((err: UiError) => {
        if (!cancelled) setError(err);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [reloadToken]);

  const reload = () => setReloadToken((t) => t + 1);

  const removeMeeting = (meetingId: Id) => {
    setMeetings((list) => list.filter((m) => m.id !== meetingId));
  };

  useEvent(EVENTS.meetingUpdated, reload);

  return { meetings, loading, error, reload, removeMeeting };
}
