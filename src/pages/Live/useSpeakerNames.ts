import { useEffect, useMemo, useState } from "react";

import { useEvent } from "../../hooks";
import { channels, live } from "../../lib/copy";
import { EVENTS, listSpeakers } from "../../lib/ipc";
import type { Channel, Id, Speaker } from "../../lib/types";

export type SpeakerLabelFor = (speakerId: Id | undefined, channel: Channel) => string;

/**
 * How often the cast list is re-read during a live meeting. Slower than the
 * transcript's own catch-up read (`useTranscriptStream`) because names change
 * far more rarely than lines do — but without it, a transcript that arrives by
 * polling while the event bridge is broken would be a wall of "Speaker".
 */
const REFRESH_INTERVAL_MS = 30_000;

/** Two cast lists that would put the same name on every line. */
function sameSpeakers(a: Speaker[], b: Speaker[]): boolean {
  return (
    a.length === b.length &&
    a.every((s, i) => {
      const other = b[i];
      return s.id === other.id && s.displayName === other.displayName && s.isSelf === other.isSelf;
    })
  );
}

/**
 * "You" for the mic channel, a speaker's current display name for the system
 * channel (the provisional "Speaker 1"-style labels the core assigns live),
 * and a plain placeholder before the first one arrives — never a blank chip
 * while clustering catches up.
 */
export function useSpeakerNames(meetingId: Id | undefined, isLive: boolean): SpeakerLabelFor {
  const [speakers, setSpeakers] = useState<Speaker[]>([]);

  useEffect(() => {
    setSpeakers([]);
    if (!meetingId) return;
    let cancelled = false;
    listSpeakers(meetingId)
      .then((result) => {
        if (!cancelled) setSpeakers(result);
      })
      .catch(() => {
        // The channel-based fallback below covers it.
      });
    return () => {
      cancelled = true;
    };
  }, [meetingId]);

  // The same floor the transcript stands on: if the window stops hearing
  // events, names still turn up, a beat late.
  useEffect(() => {
    if (!meetingId || !isLive) return;
    let cancelled = false;
    const timer = window.setInterval(() => {
      if (document.visibilityState !== "visible") return;
      listSpeakers(meetingId)
        .then((result) => {
          if (cancelled) return;
          // Same list as last time means the same array identity, so nothing
          // that draws a name re-renders on the strength of a poll alone.
          setSpeakers((prev) => (sameSpeakers(prev, result) ? prev : result));
        })
        .catch(() => {
          // Names are a courtesy; the fallback below covers a failed read and
          // nothing about the recording depends on it.
        });
    }, REFRESH_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [meetingId, isLive]);

  useEvent(EVENTS.speakersUpdated, (payload) => {
    if (payload.meetingId === meetingId) setSpeakers(payload.speakers);
  });

  const byId = useMemo(() => new Map(speakers.map((s) => [s.id, s])), [speakers]);

  return (speakerId, channel) => {
    if (channel === "mic") return channels.mic;
    if (speakerId) {
      const speaker = byId.get(speakerId);
      if (speaker) return speaker.isSelf ? channels.mic : speaker.displayName;
    }
    return live.unknownSpeaker;
  };
}
