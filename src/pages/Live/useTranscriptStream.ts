import { useEffect, useMemo, useRef, useState } from "react";

import { useEvent } from "../../hooks";
import { EVENTS, getTranscript } from "../../lib/ipc";
import type { Channel, Id, Segment, TranscriptPartialPayload } from "../../lib/types";

export interface TranscriptLineData {
  /** `utteranceId` while still partial, the segment's own id once final —
   * stable either way, so React never re-mounts a line just because it
   * settled. */
  id: string;
  tStartMs: number;
  channel: Channel;
  speakerId?: Id;
  text: string;
  isFinal: boolean;
}

/**
 * How long an unsettled line may sit there before it is swept up.
 *
 * The core closes every partial it opens, so this should never fire. It is the
 * belt to that braces: an event lost on the way to the window would otherwise
 * leave a half-written line in the transcript — and a growing map — for the rest
 * of an eight-hour meeting. Comfortably longer than the longest utterance the
 * core will ever send.
 */
const PARTIAL_EXPIRY_MS = 60_000;

/** At most this many unsettled lines at once; the oldest go first. */
const MAX_PARTIALS = 8;

/** Drop lines that are too old, and cap how many can pile up. */
function sweep(
  partials: Map<string, TranscriptLineData>,
  seenAt: Map<string, number>,
  now: number,
): void {
  for (const [id, at] of seenAt) {
    if (now - at > PARTIAL_EXPIRY_MS || !partials.has(id)) {
      partials.delete(id);
      seenAt.delete(id);
    }
  }
  // Maps keep insertion order, so the front of the list is the oldest.
  while (partials.size > MAX_PARTIALS) {
    const oldest = partials.keys().next();
    if (oldest.done) break;
    partials.delete(oldest.value);
    seenAt.delete(oldest.value);
  }
}

function fromSegment(segment: Segment): TranscriptLineData {
  return {
    id: segment.id,
    tStartMs: segment.tStartMs,
    channel: segment.channel,
    speakerId: segment.speakerId,
    text: segment.text,
    isFinal: segment.isFinal,
  };
}

function fromPartial(payload: TranscriptPartialPayload): TranscriptLineData {
  return {
    id: payload.utteranceId,
    tStartMs: payload.tStartMs,
    channel: payload.channel,
    speakerId: payload.speakerId,
    text: payload.text,
    isFinal: false,
  };
}

/**
 * The live transcript for one meeting: finalized segments in time order, plus
 * whichever utterances are still mid-sentence.
 *
 * A partial with the same `utteranceId` replaces that line in place as more
 * of it arrives; a final removes the partial and appends the settled
 * segment. A later revision (the offline pass correcting a stretch of
 * transcript) re-fetches just that range and swaps the affected lines —
 * capture itself never waits on it.
 *
 * Not every utterance produces text: one can be dropped to keep the recording
 * safe, turn out to be silence, or fail to be read. The core says so with
 * `dropped`, and the line goes away. On top of that, anything left unsettled for
 * [`PARTIAL_EXPIRY_MS`] is swept, so a lost event can never leave a "…" line —
 * or an ever-growing map — behind for a whole meeting.
 */
export function useTranscriptStream(meetingId: Id | undefined): TranscriptLineData[] {
  const [finals, setFinals] = useState<Map<string, TranscriptLineData>>(new Map());
  const [partials, setPartials] = useState<Map<string, TranscriptLineData>>(new Map());
  const partialSeenAt = useRef<Map<string, number>>(new Map());

  useEffect(() => {
    setFinals(new Map());
    setPartials(new Map());
    partialSeenAt.current = new Map();
    if (!meetingId) return;
    let cancelled = false;
    getTranscript({ meetingId })
      .then((segments) => {
        if (cancelled) return;
        setFinals(new Map(segments.map((s) => [s.id, fromSegment(s)])));
      })
      .catch(() => {
        // The live events below still carry new lines even if history
        // didn't load — nothing about capture depends on this fetch.
      });
    return () => {
      cancelled = true;
    };
  }, [meetingId]);

  useEvent(EVENTS.transcriptPartial, (payload) => {
    if (payload.meetingId !== meetingId) return;
    setPartials((prev) => {
      const next = new Map(prev);
      const seenAt = partialSeenAt.current;
      if (payload.dropped || !payload.text.trim()) {
        // This utterance is over with nothing to show: take the line away
        // rather than leave it mid-sentence forever.
        next.delete(payload.utteranceId);
        seenAt.delete(payload.utteranceId);
      } else {
        next.set(payload.utteranceId, fromPartial(payload));
        seenAt.set(payload.utteranceId, Date.now());
      }
      sweep(next, seenAt, Date.now());
      return next;
    });
  });

  useEvent(EVENTS.transcriptFinal, (payload) => {
    if (payload.meetingId !== meetingId) return;
    const partialKey = payload.utteranceId ?? payload.segment.id;
    setPartials((prev) => {
      if (!prev.has(partialKey)) return prev;
      const next = new Map(prev);
      next.delete(partialKey);
      partialSeenAt.current.delete(partialKey);
      return next;
    });
    setFinals((prev) => {
      const next = new Map(prev);
      next.set(payload.segment.id, fromSegment(payload.segment));
      return next;
    });
  });

  useEvent(EVENTS.transcriptRevised, (payload) => {
    if (payload.meetingId !== meetingId) return;
    getTranscript({ meetingId, fromMs: payload.fromMs, toMs: payload.toMs })
      .then((segments) => {
        setFinals((prev) => {
          const next = new Map(prev);
          for (const segment of segments) next.set(segment.id, fromSegment(segment));
          return next;
        });
      })
      .catch(() => {
        // A revision that can't be re-fetched just leaves the old text in
        // place; the offline pass will get another chance.
      });
  });

  return useMemo(() => {
    const lines = [...finals.values(), ...partials.values()];
    lines.sort((a, b) => a.tStartMs - b.tStartMs);
    return lines;
  }, [finals, partials]);
}
