import { useEffect, useRef, useState } from "react";

import { Button } from "@/components";
import { meeting as copy } from "@/lib/copy";
import type { Speaker } from "@/lib/types";

import { speakerColor } from "../lib/speakerColor";

export type ListenState = "idle" | "loading" | "playing";

export interface SpeakerRowProps {
  speaker: Speaker;
  listenState: ListenState;
  /** A polite, inline explanation when the sample couldn't be fetched or
   * played — shown under this row only, never a dead-end toast. */
  error: string | undefined;
  /** True while the offline pass that could replace this row is running. */
  nameDisabled: boolean;
  onListen: () => void;
  onRename: (name: string) => void;
}

/** A triangle — "play this". Kept local rather than added to the shared icon
 * set, since nothing outside this one row needs it. */
function PlayGlyph() {
  return (
    <svg width="11" height="11" viewBox="0 0 20 20" fill="none" aria-hidden>
      <path d="M6.5 4.3v11.4l9-5.7-9-5.7z" fill="currentColor" />
    </svg>
  );
}

/** A square — "stop this". Pairs with `PlayGlyph` as the toggled state of the
 * same Listen button. */
function StopGlyph() {
  return (
    <svg width="11" height="11" viewBox="0 0 20 20" fill="none" aria-hidden>
      <rect x="5.5" y="5.5" width="9" height="9" rx="1.5" fill="currentColor" />
    </svg>
  );
}

/**
 * One speaker: a colored dot, an editable name, and a button to hear a
 * sample of that voice before deciding what to type. The name commits on
 * blur or Enter, exactly like the old rename-on-chip flow it replaces.
 */
export function SpeakerRow({ speaker, listenState, error, nameDisabled, onListen, onRename }: SpeakerRowProps) {
  const [draft, setDraft] = useState(speaker.displayName);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setDraft(speaker.displayName);
  }, [speaker.displayName]);

  const color = speakerColor(speaker.id);

  const commit = () => {
    const trimmed = draft.trim();
    if (trimmed && trimmed !== speaker.displayName) onRename(trimmed);
    else setDraft(speaker.displayName);
  };

  return (
    <div className="flex items-center gap-3 rounded-xl border border-hairline px-3 py-2.5">
      <span aria-hidden className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ backgroundColor: color.fg }} />
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        <input
          ref={inputRef}
          value={draft}
          disabled={nameDisabled}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              commit();
              inputRef.current?.blur();
            }
            if (e.key === "Escape") {
              setDraft(speaker.displayName);
              inputRef.current?.blur();
            }
          }}
          placeholder={copy.renameSpeakerPlaceholder}
          className="w-full rounded-lg border border-transparent bg-transparent px-1.5 py-1 text-sm font-medium text-ink outline-none transition-colors hover:border-hairline focus:border-hairline focus:bg-surface focus:ring-2 focus:ring-accent/30 disabled:cursor-not-allowed disabled:opacity-60"
        />
        {error && <p className="text-xs text-live">{error}</p>}
      </div>
      <Button
        variant="secondary"
        size="sm"
        loading={listenState === "loading"}
        leftIcon={listenState === "playing" ? <StopGlyph /> : <PlayGlyph />}
        onClick={onListen}
      >
        {listenState === "playing" ? copy.speakersDialogStopLabel : copy.speakersDialogListenLabel}
      </Button>
    </div>
  );
}
