import { useEffect, useRef, useState } from "react";

import { meeting as copy } from "@/lib/copy";
import type { Speaker } from "@/lib/types";

import { speakerColor } from "../lib/speakerColor";

export interface SpeakerChipProps {
  speaker: Speaker | undefined;
  fallbackId: string | undefined;
  onRename: (speakerId: string, name: string) => void;
}

/** A colored speaker label. Click it to rename in place — the "rename on
 * click" affordance the Transcript tab needs, without a separate modal for
 * something this small. */
export function SpeakerChip({ speaker, fallbackId, onRename }: SpeakerChipProps) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(speaker?.displayName ?? "");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!editing) setDraft(speaker?.displayName ?? "");
  }, [speaker?.displayName, editing]);

  useEffect(() => {
    if (editing) inputRef.current?.select();
  }, [editing]);

  if (!speaker) {
    return <span className="text-xs font-medium text-ink-ghost">{fallbackId ?? "—"}</span>;
  }

  const color = speakerColor(speaker.id);

  const commit = () => {
    setEditing(false);
    const trimmed = draft.trim();
    if (trimmed && trimmed !== speaker.displayName) onRename(speaker.id, trimmed);
  };

  if (editing) {
    return (
      <input
        ref={inputRef}
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") commit();
          if (e.key === "Escape") {
            setDraft(speaker.displayName);
            setEditing(false);
          }
        }}
        placeholder={copy.renameSpeakerPlaceholder}
        className="w-28 rounded-full border border-hairline bg-surface px-2 py-0.5 text-xs text-ink focus:outline-none focus:ring-2 focus:ring-accent/30"
      />
    );
  }

  return (
    <button
      type="button"
      onClick={() => setEditing(true)}
      title={copy.renameSpeakerTitle}
      className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium transition-opacity hover:opacity-80"
      style={{ backgroundColor: color.bg, color: color.fg }}
    >
      {speaker.displayName}
    </button>
  );
}
