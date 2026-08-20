import type { TranscriptLineData } from "./useTranscriptStream";
import type { SpeakerLabelFor } from "./useSpeakerNames";

/**
 * What has been heard so far, as clean text for the Copy button: one line
 * per row currently on screen (finals and whatever's still mid-sentence
 * alike — this is a snapshot of what's visible, not just the settled part),
 * in the same speaker-labelled shape the transcript itself reads in.
 */
export function buildLiveTranscriptText(
  lines: TranscriptLineData[],
  speakerLabelFor: SpeakerLabelFor,
): string {
  return lines
    .filter((line) => line.text.trim().length > 0)
    .map((line) => `${speakerLabelFor(line.speakerId, line.channel)}: ${line.text.trim()}`)
    .join("\n");
}
