import { channels, live } from "@/lib/copy";
import type { Channel, Segment, Speaker } from "@/lib/types";

import { formatTimestamp } from "./date";
import { resolveSpeaker } from "./speakers";

/**
 * What an unattributed microphone line is called in this meeting — the same
 * answer the export module works out server-side (`mic_fallback_label` in
 * `src-tauri/src/export/mod.rs`), so the Copy button and the exported file read
 * the same way.
 *
 * With a system channel the microphone carried one known person and "You" is
 * safe. Without one it carried the whole room — the far end leaks in through
 * the speakers — so an unattributed microphone line is not safely anybody, and
 * calling it "You" is the mislabel of 2026-08-24 in the Copy button instead of
 * the live banner.
 *
 * The catch, and why this is not simply `micOnly ? "Speaker" : "You"`: only
 * unattributed lines reach a fallback, and on a mic-only meeting the live pass
 * has already pointed the lines it wrote at a "You" row picked from the channel
 * alone, while the catch-up pass leaves its lines unattributed. Pasting a
 * transcript that says "You" on one half and "Speaker" on the other — one
 * voice, one meeting — is worse than an honest over-claim, so while that row is
 * still carrying lines this matches whatever it is called. The offline pass
 * drops it on a mic-only meeting, and from then on nothing says "You".
 */
function micFallbackLabel(segments: Segment[], speakers: Speaker[]): string {
  if (segments.some((s) => s.channel === "system")) return channels.mic;
  for (const segment of segments) {
    if (segment.channel !== "mic" || !segment.speakerId) continue;
    const speaker = resolveSpeaker(segment.speakerId, speakers);
    if (speaker?.isSelf) return speaker.displayName;
  }
  return live.unknownSpeaker;
}

/** The name a line with no speaker of its own carries — never a raw id in
 * something a person is about to paste somewhere. */
function fallbackLabel(channel: Channel, micFallback: string): string {
  return channel === "mic" ? micFallback : channels.system;
}

/**
 * The full speaker-labelled transcript as clean text, for the Transcript
 * tab's Copy button: `[m:ss] Speaker: what they said`, one line per segment,
 * in the same order the tab reads. Kept independent of whatever the person
 * has typed into the filter box — Copy always gets everything.
 */
export function buildTranscriptText(segments: Segment[], speakers: Speaker[]): string {
  const micFallback = micFallbackLabel(segments, speakers);
  return segments
    .filter((s) => s.text.trim().length > 0)
    .map((s) => {
      const speaker = resolveSpeaker(s.speakerId, speakers);
      const label = speaker?.displayName ?? fallbackLabel(s.channel, micFallback);
      return `[${formatTimestamp(s.tStartMs)}] ${label}: ${s.text.trim()}`;
    })
    .join("\n");
}
