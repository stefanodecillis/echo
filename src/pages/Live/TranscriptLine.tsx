import { Chip } from "../../components";
import { cx } from "../../components/lib/cx";
import { channels } from "../../lib/copy";

export interface TranscriptLineProps {
  speakerLabel: string;
  text: string;
  /** Still arriving — shown a shade quieter, the way a typing indicator reads
   * as "not settled yet" without needing its own icon. */
  pending: boolean;
}

/** One row of the live transcript: a speaker chip and the line itself,
 * sized to the fixed row height `VirtualList` renders it in. */
export function TranscriptLine({ speakerLabel, text, pending }: TranscriptLineProps) {
  return (
    <div className="flex h-full items-start gap-3 border-b border-hairline px-5 py-3">
      <Chip
        variant={speakerLabel === channels.mic ? "solid" : "neutral"}
        className="mt-0.5 shrink-0"
      >
        {speakerLabel}
      </Chip>
      <p className={cx("line-clamp-2 flex-1 text-sm", pending ? "text-ink-faint" : "text-ink")}>
        {text}
      </p>
    </div>
  );
}
