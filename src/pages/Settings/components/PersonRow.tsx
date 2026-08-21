import { useEffect, useRef, useState } from "react";

import { Button, Modal } from "@/components";
import { common, knownPeople, settings as copy } from "@/lib/copy";

import { formatLastHeard } from "../lib/peopleFormat";
import type { ListenState, Person } from "../lib/peopleIpc";

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

export interface PersonRowProps {
  person: Person;
  listenState: ListenState;
  /** A polite, inline explanation when the sample couldn't be fetched or
   * played — shown under this row only, never a dead-end toast. */
  error?: string;
  onListen: () => void;
  onRename: (name: string) => void;
  /** Resolves once the person is actually gone, so the row can be removed
   * from the list; rejecting leaves the confirm dialog open with the
   * message visible. */
  onDelete: () => Promise<void>;
}

/**
 * One saved voice: an inline-editable name, "last heard" + sample count as a
 * quiet detail, Listen, and Forget behind one confirmation
 * ("Forget this voice? The saved samples are deleted too.") — the whole
 * shape docs/DESIGN.md §1 asks for.
 */
export function PersonRow({ person, listenState, error, onListen, onRename, onDelete }: PersonRowProps) {
  const [draft, setDraft] = useState(person.name);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string>();
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setDraft(person.name);
  }, [person.name]);

  const commit = () => {
    const trimmed = draft.trim();
    if (trimmed && trimmed !== person.name) onRename(trimmed);
    else setDraft(person.name);
  };

  const handleDelete = async () => {
    setDeleting(true);
    setDeleteError(undefined);
    try {
      await onDelete();
      setConfirmOpen(false);
    } catch (err) {
      setDeleteError((err as { message?: string }).message ?? common.retry);
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div className="flex items-center gap-3 rounded-xl border border-hairline px-3 py-2.5">
      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        <input
          ref={inputRef}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              commit();
              inputRef.current?.blur();
            }
            if (e.key === "Escape") {
              setDraft(person.name);
              inputRef.current?.blur();
            }
          }}
          placeholder={copy.peopleNamePlaceholder}
          className="w-full rounded-lg border border-transparent bg-transparent px-1.5 py-1 text-sm font-medium text-ink outline-none transition-colors hover:border-hairline focus:border-hairline focus:bg-surface focus:ring-2 focus:ring-accent/30"
        />
        <div className="flex flex-wrap items-center gap-x-1.5 px-1.5 text-xs text-ink-faint">
          <span>
            {person.lastHeardAt
              ? knownPeople.lastHeard(formatLastHeard(person.lastHeardAt))
              : knownPeople.neverHeard}
          </span>
          <span aria-hidden>·</span>
          <span>{knownPeople.sampleCount(person.sampleCount)}</span>
        </div>
        {person.needsRefresh && (
          <p className="px-1.5 text-xs text-ink-faint">{copy.peopleRefreshingNote}</p>
        )}
        {error && <p className="px-1.5 text-xs text-live">{error}</p>}
      </div>

      <Button
        variant="secondary"
        size="sm"
        loading={listenState === "loading"}
        leftIcon={listenState === "playing" ? <StopGlyph /> : <PlayGlyph />}
        onClick={onListen}
      >
        {listenState === "playing" ? copy.peopleStopLabel : copy.peopleListenLabel}
      </Button>
      <Button variant="ghost" size="sm" onClick={() => setConfirmOpen(true)}>
        {copy.peopleForgetButton}
      </Button>

      <Modal
        open={confirmOpen}
        onClose={() => setConfirmOpen(false)}
        title={copy.peopleDeleteConfirmTitle}
        footer={
          <>
            <Button variant="secondary" onClick={() => setConfirmOpen(false)}>
              {common.cancel}
            </Button>
            <Button variant="primary" loading={deleting} onClick={handleDelete}>
              {copy.peopleForgetButton}
            </Button>
          </>
        }
      >
        {copy.peopleDeleteConfirmDescription}
        {deleteError && <p className="mt-2 text-xs text-live">{deleteError}</p>}
      </Modal>
    </div>
  );
}
