import { useEffect, useMemo, useState } from "react";

import { useEvent } from "../../hooks";
import { channels, live } from "../../lib/copy";
import { EVENTS, listSpeakers } from "../../lib/ipc";
import type { Channel, Id, Speaker } from "../../lib/types";

export type SpeakerLabelFor = (speakerId: Id | undefined, channel: Channel) => string;

/**
 * "You" for the mic channel, a speaker's current display name for the system
 * channel (the provisional "Speaker 1"-style labels the core assigns live),
 * and a plain placeholder before the first one arrives — never a blank chip
 * while clustering catches up.
 */
export function useSpeakerNames(meetingId: Id | undefined): SpeakerLabelFor {
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
