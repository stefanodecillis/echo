import { Chip } from "../../components";
import { cx } from "../../components/lib/cx";
import { channels } from "../../lib/copy";
import { formatTimestamp } from "./formatTimestamp";

export interface TranscriptLineProps {
  /** When this line's speech started, relative to the meeting clock. */
  tStartMs: number;
  speakerLabel: string;
  text: string;
  /** Still arriving — shown a shade quieter with a soft shimmer, the way a
   * typing indicator reads as "not settled yet" without needing its own icon. */
  pending: boolean;
}

/** One row of the live transcript: a quiet timestamp, a speaker chip, and the
 * line itself, sized to the fixed row height `VirtualList` renders it in. */
export function TranscriptLine({ tStartMs, speakerLabel, text, pending }: TranscriptLineProps) {
  return (
    <div className="flex h-full items-start gap-3 border-b border-hairline px-5 py-3">
      <span className="mt-1 shrink-0 select-none text-xs tabular-nums text-ink-ghost">
        {formatTimestamp(tStartMs)}
      </span>
      <Chip
        variant={speakerLabel === channels.mic ? "solid" : "neutral"}
        className="mt-0.5 shrink-0"
      >
        {speakerLabel}
      </Chip>
      <p
        className={cx(
          "line-clamp-2 flex-1 text-sm",
          pending ? "animate-pulse text-ink-faint" : "text-ink",
        )}
      >
        {text}
        {pending && <span aria-hidden> …</span>}
      </p>
    </div>
  );
}
