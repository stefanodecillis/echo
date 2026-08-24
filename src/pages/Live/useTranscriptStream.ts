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

/**
 * How often a live meeting re-reads the transcript from the core.
 *
 * Events are the fast path and this is the floor under them. On 2026-08-24 the
 * window stopped receiving Tauri events mid-meeting while the core happily kept
 * writing segments to the database: the transcript on screen froze for the rest
 * of the meeting even though every word was safely recorded. Nothing the core
 * can do fixes that from its side — once the bridge into the window is broken,
 * only the window asking again gets the words across. Five seconds is short
 * enough that a broken bridge reads as a small lag rather than a dead screen.
 */
const POLL_INTERVAL_MS = 5_000;

/**
 * How far back before the newest line each poll looks.
 *
 * Segments are not written in perfect time order — a slow utterance can settle
 * after a later one — so asking only for "everything after the newest thing I
 * have" would step over a line that landed just behind it. Half a minute of
 * overlap is far more than the core's own lag and costs a handful of rows.
 */
const POLL_OVERLAP_MS = 30_000;

/**
 * Every this-many-th poll asks for the whole meeting instead of the recent
 * window — the backstop for anything stranded far behind the watermark (a long
 * revision, a catch-up pass filling in the start of the meeting). At a
 * five-second beat that is one full read a minute.
 */
const FULL_SWEEP_EVERY = 12;

/** Whether a line already on screen says exactly what the core now says. */
function sameLine(line: TranscriptLineData, segment: Segment): boolean {
  return (
    line.text === segment.text &&
    line.speakerId === segment.speakerId &&
    line.tStartMs === segment.tStartMs &&
    line.isFinal === segment.isFinal
  );
}

/**
 * Fold freshly-read segments into the finals already on screen, returning
 * `null` when every one of them was already there unchanged.
 *
 * The `null` is the point. This runs every few seconds for the whole length of a
 * meeting, and almost every run has nothing new in it: handing back a new `Map`
 * anyway would re-run the sort below, hand `VirtualList` a brand-new array, and
 * trip the "did a line arrive?" effect in `Live/index.tsx` that follows the view
 * down — a transcript that twitches once a beat, forever. Same identity in,
 * same identity out, and the screen stays still.
 *
 * Pure on purpose: no state, no clock, no fetching — the one piece of this file
 * worth testing on its own.
 */
export function mergeFinals(
  prev: Map<string, TranscriptLineData>,
  segments: Segment[],
): Map<string, TranscriptLineData> | null {
  let next: Map<string, TranscriptLineData> | null = null;
  for (const segment of segments) {
    const existing = (next ?? prev).get(segment.id);
    if (existing && sameLine(existing, segment)) continue;
    if (!next) next = new Map(prev);
    next.set(segment.id, fromSegment(segment));
  }
  return next;
}

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
 *
 * While `live` is true the meeting is also re-read from the core every few
 * seconds, so that if events stop arriving altogether the transcript falls
 * behind by seconds instead of stopping dead (see [`POLL_INTERVAL_MS`]).
 */
export function useTranscriptStream(
  meetingId: Id | undefined,
  live: boolean,
): TranscriptLineData[] {
  const [finals, setFinals] = useState<Map<string, TranscriptLineData>>(new Map());
  const [partials, setPartials] = useState<Map<string, TranscriptLineData>>(new Map());
  const partialSeenAt = useRef<Map<string, number>>(new Map());
  /** The end of the furthest-along line seen so far, in meeting time: where the
   * next catch-up read starts from (minus [`POLL_OVERLAP_MS`] of slack). */
  const latestEnd = useRef(0);

  useEffect(() => {
    setFinals(new Map());
    setPartials(new Map());
    partialSeenAt.current = new Map();
    latestEnd.current = 0;
    if (!meetingId) return;
    let cancelled = false;
    getTranscript({ meetingId })
      .then((segments) => {
        if (cancelled) return;
        for (const s of segments) latestEnd.current = Math.max(latestEnd.current, s.tEndMs);
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
    latestEnd.current = Math.max(latestEnd.current, payload.segment.tEndMs);
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

  // The floor under the events above: while the meeting is running, ask the
  // core for the transcript again every few seconds and fold in anything the
  // window did not hear about.
  //
  // Which lines arrived by event and which by poll makes no difference to what
  // is drawn: both land in the same map keyed by the segment's own id, the memo
  // below sorts by start time, and the core's database is the one source of
  // truth for the text — so a line can be written twice, in either order, and
  // the result is the same. Nothing here needs to know which path a line took.
  useEffect(() => {
    if (!meetingId || !live) return;
    let cancelled = false;
    let passes = 0;

    const tick = () => {
      // A window nobody is looking at does not need to be current; it catches
      // up on the first tick after it comes back.
      if (document.visibilityState !== "visible") return;

      // Retire any "…" line the core never closed. Until now this only ran when
      // a partial event arrived — which is precisely what stops happening when
      // the bridge into the window breaks, leaving a half-written line sitting
      // there for the rest of the meeting.
      setPartials((prev) => {
        if (prev.size === 0) return prev;
        const next = new Map(prev);
        sweep(next, partialSeenAt.current, Date.now());
        return next.size === prev.size ? prev : next;
      });

      passes += 1;
      const fullSweep = passes % FULL_SWEEP_EVERY === 0;
      const fromMs = fullSweep ? undefined : Math.max(0, latestEnd.current - POLL_OVERLAP_MS);
      getTranscript({ meetingId, fromMs })
        .then((segments) => {
          if (cancelled) return;
          for (const s of segments) latestEnd.current = Math.max(latestEnd.current, s.tEndMs);
          setFinals((prev) => mergeFinals(prev, segments) ?? prev);
        })
        .catch(() => {
          // Nothing about capture depends on this read: the words are already
          // in the core's database either way, and the next tick tries again.
        });
    };

    const timer = window.setInterval(tick, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [meetingId, live]);

  return useMemo(() => {
    const lines = [...finals.values(), ...partials.values()];
    lines.sort((a, b) => a.tStartMs - b.tStartMs);
    return lines;
  }, [finals, partials]);
}
