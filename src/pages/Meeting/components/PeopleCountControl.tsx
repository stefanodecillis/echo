import { useState } from "react";

import { Button, Modal } from "@/components";
import { setSpeakerCount, toUiError } from "@/lib/ipc";
import { common, meeting as copy, peopleCount as peopleCountCopy } from "@/lib/copy";
import { useEchoStore } from "@/lib/store";
import type { Id } from "@/lib/types";

import { IconButton } from "./IconButton";
import { MAX_PEOPLE_COUNT, MIN_PEOPLE_COUNT } from "../lib/peopleCount";

export interface PeopleCountControlProps {
  meetingId: Id;
  /** How many people the automatic pass is currently using. */
  peopleCount: number;
  /** True once someone has corrected the count by hand; false while it's
   * still whatever the automatic pass found on its own. */
  isOverride: boolean;
  /** True while a recording or a transcript rewrite is in progress — the
   * offline pass this control re-runs can't overlap either. */
  disabled: boolean;
}

/** What we're about to ask `setSpeakerCount` for, once the one confirmation
 * has been answered. `count: null` is "back to automatic". */
interface PendingChange {
  count: number | null;
}

/**
 * The quiet "N people" control on the Transcript tab. Clicking it opens a
 * stepper (1-12) to correct how many people the automatic pass should assume,
 * plus a way back to letting Echo decide on its own. Either path asks once
 * before acting, since it can undo names given to speakers — the actual redo
 * then shows up as the same job-progress treatment a fresh recording gets.
 */
export function PeopleCountControl({ meetingId, peopleCount, isOverride, disabled }: PeopleCountControlProps) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(peopleCount);
  const [pending, setPending] = useState<PendingChange>();
  const [submitting, setSubmitting] = useState(false);
  const addToast = useEchoStore((s) => s.addToast);

  const openEditor = () => {
    setDraft(peopleCount);
    setEditing(true);
  };

  const closeEditor = () => {
    setEditing(false);
    setPending(undefined);
  };

  const commit = async () => {
    if (!pending) return;
    setSubmitting(true);
    try {
      await setSpeakerCount(meetingId, pending.count);
      closeEditor();
    } catch (err) {
      addToast({ level: "problem", message: toUiError(err).message });
      setPending(undefined);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <>
      <button
        type="button"
        onClick={openEditor}
        disabled={disabled}
        title={copy.peopleCountTriggerTitle}
        className="rounded-full px-2.5 py-1 text-xs font-medium text-ink-faint outline-none transition-colors hover:bg-surface-sunken hover:text-ink-soft focus-visible:ring-2 focus-visible:ring-accent/40 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {peopleCountCopy.triggerLabel(peopleCount, !isOverride)}
      </button>

      <Modal
        open={editing}
        onClose={closeEditor}
        title={copy.peopleCountEditTitle}
        footer={
          <>
            {isOverride && (
              <Button variant="secondary" onClick={() => setPending({ count: null })}>
                {copy.peopleCountBackToAutomatic}
              </Button>
            )}
            <Button
              variant="primary"
              disabled={isOverride && draft === peopleCount}
              onClick={() => setPending({ count: draft })}
            >
              {common.save}
            </Button>
          </>
        }
      >
        <p className="mb-4">{copy.peopleCountEditDescription}</p>
        <div className="flex items-center justify-center gap-4">
          <IconButton
            icon={<span className="text-base leading-none">−</span>}
            aria-label={copy.peopleCountFewer}
            title={copy.peopleCountFewer}
            disabled={draft <= MIN_PEOPLE_COUNT}
            onClick={() => setDraft((d) => Math.max(MIN_PEOPLE_COUNT, d - 1))}
          />
          <span className="w-10 text-center text-2xl font-semibold tabular-nums text-ink">{draft}</span>
          <IconButton
            icon={<span className="text-base leading-none">+</span>}
            aria-label={copy.peopleCountMore}
            title={copy.peopleCountMore}
            disabled={draft >= MAX_PEOPLE_COUNT}
            onClick={() => setDraft((d) => Math.min(MAX_PEOPLE_COUNT, d + 1))}
          />
        </div>
      </Modal>

      <Modal
        open={!!pending}
        onClose={() => setPending(undefined)}
        title={pending ? peopleCountCopy.confirmTitle(pending.count) : undefined}
        footer={
          <>
            <Button variant="secondary" onClick={() => setPending(undefined)}>
              {common.cancel}
            </Button>
            <Button variant="primary" loading={submitting} onClick={commit}>
              {copy.peopleCountRedoButton}
            </Button>
          </>
        }
      >
        {copy.peopleCountConfirmDescription}
      </Modal>
    </>
  );
}
