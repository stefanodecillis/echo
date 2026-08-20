import { channels } from "@/lib/copy";
import type { Channel, Segment, Speaker } from "@/lib/types";

import { formatTimestamp } from "./date";
import { resolveSpeaker } from "./speakers";

/** Same fallback the export module uses server-side (see
 * `default_speaker_label` in `src-tauri/src/export/mod.rs`) when a segment's
 * speaker hasn't been named or resolved yet — never a raw id in something a
 * person is about to paste somewhere. */
function fallbackLabel(channel: Channel): string {
  return channel === "mic" ? channels.mic : channels.system;
}

/**
 * The full speaker-labelled transcript as clean text, for the Transcript
 * tab's Copy button: `[m:ss] Speaker: what they said`, one line per segment,
 * in the same order the tab reads. Kept independent of whatever the person
 * has typed into the filter box — Copy always gets everything.
 */
export function buildTranscriptText(segments: Segment[], speakers: Speaker[]): string {
  return segments
    .filter((s) => s.text.trim().length > 0)
    .map((s) => {
      const speaker = resolveSpeaker(s.speakerId, speakers);
      const label = speaker?.displayName ?? fallbackLabel(s.channel);
      return `[${formatTimestamp(s.tStartMs)}] ${label}: ${s.text.trim()}`;
    })
    .join("\n");
}
