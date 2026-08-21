import { useCallback, useEffect, useRef, useState } from "react";

import { EVENTS, getMeeting, listSummaries, toUiError } from "@/lib/ipc";
import { useEvent } from "@/hooks/useEvent";
import type { Id, MeetingDetail, UiError } from "@/lib/types";

import type { MeetingDetailWithPeopleCount, SpeakersUpdatedPayloadWithPeopleCount } from "../lib/peopleCount";

export interface UseMeetingDetailResult {
  detail: MeetingDetail | undefined;
  loading: boolean;
  error: UiError | undefined;
  /** Re-fetch everything. Used after a destructive action (delete audio) or
   * when an event says something changed that a patch can't express. */
  reload: () => void;
  /** Local, optimistic patches — kept separate from `reload` so a title edit
   * or a done-toggle doesn't have to wait on a round trip to feel real. */
  setDetail: (updater: (detail: MeetingDetail) => MeetingDetail) => void;
}

/**
 * Loads one meeting's full detail and keeps it fresh from the backend
 * events that can change it while the screen is open: a still-running
 * transcript catch-up renaming/merging speakers, a recap finishing, action
 * items being ticked off from elsewhere.
 */
export function useMeetingDetail(meetingId: Id | undefined): UseMeetingDetailResult {
  const [detail, setDetailState] = useState<MeetingDetail>();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<UiError>();
  const meetingIdRef = useRef(meetingId);
  meetingIdRef.current = meetingId;

  const reload = useCallback(() => {
    if (!meetingId) return;
    setLoading(true);
    setError(undefined);
    getMeeting(meetingId)
      .then((next) => setDetailState(next))
      .catch((err) => setError(toUiError(err)))
      .finally(() => setLoading(false));
  }, [meetingId]);

  useEffect(() => {
    setDetailState(undefined);
    reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meetingId]);

  const setDetail = useCallback((updater: (detail: MeetingDetail) => MeetingDetail) => {
    setDetailState((prev) => (prev ? updater(prev) : prev));
  }, []);

  useEvent(EVENTS.meetingUpdated, (payload) => {
    if (payload.meetingId !== meetingIdRef.current) return;
    setDetail((prev) => ({
      ...prev,
      meeting: {
        ...prev.meeting,
        status: payload.status,
        title: payload.title ?? prev.meeting.title,
        durationMs: payload.durationMs,
        deletedAt: payload.deleted ? prev.meeting.deletedAt ?? new Date().toISOString() : prev.meeting.deletedAt,
      },
    }));
  });

  useEvent(EVENTS.speakersUpdated, (payload) => {
    if (payload.meetingId !== meetingIdRef.current) return;
    // A redo of who-said-what (the Transcript tab's people-count control) can
    // also change the detected count or clear an override; pick that up here
    // too, if the payload carries it, alongside the refreshed speaker list.
    const withPeopleCount = payload as SpeakersUpdatedPayloadWithPeopleCount;
    setDetail((prev) => {
      const prevWithPeopleCount = prev as MeetingDetailWithPeopleCount;
      return {
        ...prev,
        speakers: payload.speakers,
        peopleCount: withPeopleCount.peopleCount ?? prevWithPeopleCount.peopleCount,
        peopleCountIsOverride: withPeopleCount.peopleCountIsOverride ?? prevWithPeopleCount.peopleCountIsOverride,
      };
    });
  });

  useEvent(EVENTS.actionItemsUpdated, (payload) => {
    if (payload.meetingId !== meetingIdRef.current) return;
    setDetail((prev) => ({ ...prev, actionItems: payload.items }));
  });

  useEvent(EVENTS.summaryReady, (payload) => {
    if (payload.meetingId !== meetingIdRef.current) return;
    listSummaries(payload.meetingId)
      .then((summaries) => setDetail((prev) => ({ ...prev, summaries })))
      .catch(() => {
        // The summary-ready toast (if any) already told the person; a
        // refresh will pick this up next time the screen loads.
      });
  });

  useEvent(EVENTS.jobProgress, (payload) => {
    if (payload.job.meetingId !== meetingIdRef.current) return;
    // The stage travels beside the job rather than in it, and it is the truer
    // description of what is happening — a catch-up job waiting for the engine
    // to be got ready is not yet catching up on anything. Folded in here so
    // every screen reading `detail.jobs` gets it (see `lib/jobs.ts`).
    const job = { ...payload.job, phase: payload.phase };
    setDetail((prev) => {
      const jobs = prev.jobs.some((j) => j.id === job.id)
        ? prev.jobs.map((j) => (j.id === job.id ? job : j))
        : [...prev.jobs, job];
      return { ...prev, jobs };
    });
  });

  return { detail, loading, error, reload, setDetail };
}
