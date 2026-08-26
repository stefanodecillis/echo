import { useCallback, useEffect, useRef, useState } from "react";

import { EVENTS, getMeeting, listSummaries, toUiError } from "@/lib/ipc";
import { useEvent } from "@/hooks/useEvent";
import type { Id, MeetingDetail, UiError } from "@/lib/types";

import type { MeetingDetailWithPeopleCount, SpeakersUpdatedPayloadWithPeopleCount } from "../lib/peopleCount";

export interface UseMeetingDetailResult {
  detail: MeetingDetail | undefined;
  loading: boolean;
  error: UiError | undefined;
  /**
   * How many voices the last speaker pass could actually tell apart. Only a
   * finished pass knows this, so it lives here rather than on `detail`: it
   * arrives with the event and is gone again on a page reload, which is right —
   * it describes a run, not the meeting.
   */
  voicesFound: number | undefined;
  /**
   * The best count the last automatic pass decided against, if there was one.
   * Same lifetime as `voicesFound`: it describes a run, not the meeting.
   */
  alternativeCount: number | undefined;
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
  const [voicesFound, setVoicesFound] = useState<number>();
  const [alternativeCount, setAlternativeCount] = useState<number>();
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
    setVoicesFound(undefined);
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
    // Only a pass that has just run sends this. The emitters that are only
    // announcing rows leave it out, and the last pass's answer stands rather
    // than being wiped by, say, a rename.
    if (typeof payload.voicesFound === "number") setVoicesFound(payload.voicesFound);
    // A fresh decision retires the previous runner-up; an emitter that only
    // announces rows leaves the last answer standing, exactly like the count.
    if (payload.alternativeCount != null) setAlternativeCount(payload.alternativeCount);
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
    // The job carries its own stage, so this row replaces the fetched one
    // whole. It used to arrive beside the job and be folded in here, which
    // meant a screen opened after the announcement drew the job's own name over
    // a bar at 0% — a catch-up job waiting for the engine to be got ready is
    // not catching up on anything yet, and said so to nobody.
    const job = payload.job;
    setDetail((prev) => {
      const jobs = prev.jobs.some((j) => j.id === job.id)
        ? prev.jobs.map((j) => (j.id === job.id ? job : j))
        : [...prev.jobs, job];
      return { ...prev, jobs };
    });
  });

  return { detail, loading, error, reload, setDetail, voicesFound, alternativeCount };
}
