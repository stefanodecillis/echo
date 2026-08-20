import { meeting as copy } from "@/lib/copy";
import type { Speaker } from "@/lib/types";

import { speakerColor } from "../lib/speakerColor";

export interface SpeakerChipProps {
  speaker: Speaker | undefined;
  fallbackId: string | undefined;
  /** Opens the "Name your speakers" dialog — a chip is a shortcut into that
   * flow, not a rename affordance of its own. */
  onOpen: () => void;
}

/** A colored speaker label. Click it to open the naming dialog, where its
 * name (and everyone else's) can actually be changed. */
export function SpeakerChip({ speaker, fallbackId, onOpen }: SpeakerChipProps) {
  if (!speaker) {
    return <span className="text-xs font-medium text-ink-ghost">{fallbackId ?? "—"}</span>;
  }

  const color = speakerColor(speaker.id);

  return (
    <button
      type="button"
      onClick={onOpen}
      title={copy.renameSpeakerTitle}
      className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium transition-opacity hover:opacity-80"
      style={{ backgroundColor: color.bg, color: color.fg }}
    >
      {speaker.displayName}
    </button>
  );
}
