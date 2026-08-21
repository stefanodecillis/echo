import { useState } from "react";

import { Button, Input } from "@/components";
import { knownPeople, settings as copy } from "@/lib/copy";

import { formatLastHeard } from "../lib/peopleFormat";
import type { ListenState, SuggestedPerson } from "../lib/peopleIpc";

/** A triangle — "play this". */
function PlayGlyph() {
  return (
    <svg width="11" height="11" viewBox="0 0 20 20" fill="none" aria-hidden>
      <path d="M6.5 4.3v11.4l9-5.7-9-5.7z" fill="currentColor" />
    </svg>
  );
}

/** A square — "stop this". */
function StopGlyph() {
  return (
    <svg width="11" height="11" viewBox="0 0 20 20" fill="none" aria-hidden>
      <rect x="5.5" y="5.5" width="9" height="9" rx="1.5" fill="currentColor" />
    </svg>
  );
}

export interface SuggestedPersonRowProps {
  suggestion: SuggestedPerson;
  /** 0-based position in the list, only used to tell otherwise-identical
   * unnamed rows apart ("Unnamed voice 1", "Unnamed voice 2"). */
  index: number;
  listenState: ListenState;
  error?: string;
  onListen: () => void;
  /** Resolves once the person is saved; rejecting leaves the name typed in
   * and the message visible, so nothing is lost. */
  onSave: (name: string) => Promise<void>;
}

/**
 * One recurring voice Echo hasn't been told a name for: a representative
 * sample to listen to, and a "Save as…" field that enrolls it in one step.
 */
export function SuggestedPersonRow({
  suggestion,
  index,
  listenState,
  error,
  onListen,
  onSave,
}: SuggestedPersonRowProps) {
  const [name, setName] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string>();

  const handleSave = async () => {
    const trimmed = name.trim();
    if (!trimmed) return;
    setSaving(true);
    setSaveError(undefined);
    try {
      await onSave(trimmed);
      setName("");
    } catch (err) {
      setSaveError((err as { message?: string }).message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="flex flex-col gap-2 rounded-xl border border-hairline px-3 py-2.5 sm:flex-row sm:items-center sm:justify-between sm:gap-3">
      <div className="flex min-w-0 flex-col gap-0.5">
        <p className="text-sm font-medium text-ink-soft">
          {copy.peopleSuggestedUnnamed} {index + 1}
        </p>
        {/* Which voice this is, in the only terms anybody has for it yet: how
            often it has turned up and when it last did. Two rows both reading
            "Unnamed voice" would give nobody anything to choose between. */}
        <div className="flex flex-wrap items-center gap-x-1.5 text-xs text-ink-faint">
          <span>{knownPeople.heardIn(suggestion.appearances)}</span>
          <span aria-hidden>·</span>
          <span>{knownPeople.lastHeard(formatLastHeard(suggestion.lastHeardAt))}</span>
        </div>
      </div>

      <div className="flex flex-1 flex-wrap items-center justify-end gap-2">
        <Button
          variant="secondary"
          size="sm"
          loading={listenState === "loading"}
          leftIcon={listenState === "playing" ? <StopGlyph /> : <PlayGlyph />}
          onClick={onListen}
        >
          {listenState === "playing" ? copy.peopleStopLabel : copy.peopleListenLabel}
        </Button>
        <div className="w-40">
          <Input
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleSave();
            }}
            placeholder={copy.peopleSaveAsPlaceholder}
          />
        </div>
        <Button variant="primary" size="sm" disabled={!name.trim()} loading={saving} onClick={handleSave}>
          {copy.peopleSaveAsButton}
        </Button>
      </div>

      {(error || saveError) && (
        <p className="w-full text-xs text-live sm:text-right">{saveError ?? error}</p>
      )}
    </div>
  );
}
