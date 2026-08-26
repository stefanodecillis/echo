import { useState } from "react";

import { Button, Modal } from "@/components";
import { common, knownPeople, notices, settings as copy } from "@/lib/copy";
import { toUiError } from "@/lib/ipc";
import type { Id } from "@/lib/types";

import { mergePeople, type Person } from "../lib/peopleIpc";

export interface MergePeopleModalProps {
  open: boolean;
  onClose: () => void;
  people: Person[];
  /** Called with the list as the core now has it, so the screen can show the
   * result without asking again. */
  onMerged: (people: Person[]) => void;
}

/**
 * "These two are the same person" for the saved voices in Settings.
 *
 * Deliberately the same shape as `Meeting/components/MergeSpeakersModal`: pick
 * two, say whose name survives, combine. Somebody who has combined two speakers
 * in a meeting already knows how this works, and a second idiom for the same
 * idea would be a second thing to learn.
 *
 * The one difference is the one that matters. Combining two speakers in a
 * meeting is non-destructive — the merged row keeps its place and points
 * elsewhere. Combining two saved voices is not: making one voice out of two
 * throws away the saved sound that turns out to be a near-duplicate, and that
 * is exactly what makes the result one voice rather than a bag of two. So this
 * one has a second step, which names both people and says plainly what happens
 * and that it cannot be taken back, before anything is written.
 */
export function MergePeopleModal({ open, onClose, people, onMerged }: MergePeopleModalProps) {
  const [selected, setSelected] = useState<Id[]>([]);
  const [keepId, setKeepId] = useState<Id>();
  const [confirming, setConfirming] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string>();

  const keep = people.find((p) => p.id === keepId);
  const merge = people.find((p) => selected.includes(p.id) && p.id !== keepId);

  const toggle = (id: Id) => {
    setError(undefined);
    setSelected((prev) => {
      if (prev.includes(id)) {
        const next = prev.filter((s) => s !== id);
        if (keepId === id) setKeepId(undefined);
        return next;
      }
      if (prev.length >= 2) return prev;
      return [...prev, id];
    });
  };

  const reset = () => {
    setSelected([]);
    setKeepId(undefined);
    setConfirming(false);
    setError(undefined);
  };

  const handleClose = () => {
    if (pending) return;
    reset();
    onClose();
  };

  const review = () => {
    if (selected.length !== 2 || !keepId) {
      setError(copy.peopleMergeNeedTwo);
      return;
    }
    setError(undefined);
    setConfirming(true);
  };

  const confirm = async () => {
    if (!keep || !merge) return;
    setPending(true);
    setError(undefined);
    try {
      const updated = await mergePeople(keep.id, merge.id);
      reset();
      onClose();
      onMerged(updated);
    } catch (err) {
      // Back to the picker with the reason showing: whatever the core turned
      // this down for, the answer is a different pair, not the same one again.
      setConfirming(false);
      setError(toUiError(err).message || notices.somethingWentWrong);
    } finally {
      setPending(false);
    }
  };

  if (confirming && keep && merge) {
    return (
      <Modal
        open={open}
        onClose={handleClose}
        title={copy.peopleMergeConfirmTitle}
        footer={
          <>
            <Button variant="secondary" disabled={pending} onClick={() => setConfirming(false)}>
              {copy.peopleMergeBackButton}
            </Button>
            <Button variant="primary" loading={pending} onClick={confirm}>
              {copy.peopleMergeConfirmButton}
            </Button>
          </>
        }
      >
        <p>{knownPeople.combineConfirm(keep.name, merge.name)}</p>
      </Modal>
    );
  }

  return (
    <Modal
      open={open}
      onClose={handleClose}
      title={copy.peopleMergeTitle}
      footer={
        <>
          <Button variant="secondary" onClick={handleClose}>
            {common.cancel}
          </Button>
          <Button
            variant="primary"
            disabled={selected.length !== 2 || !keepId}
            onClick={review}
          >
            {common.next}
          </Button>
        </>
      }
    >
      <p className="mb-4">{copy.peopleMergeDescription}</p>
      <div className="flex flex-col gap-2">
        {people.map((person) => {
          const isSelected = selected.includes(person.id);
          return (
            <div
              key={person.id}
              className="flex items-center justify-between gap-3 rounded-xl border border-hairline px-3 py-2"
            >
              <label className="flex min-w-0 flex-1 items-center gap-2.5 text-sm text-ink">
                <input
                  type="checkbox"
                  checked={isSelected}
                  onChange={() => toggle(person.id)}
                  disabled={!isSelected && selected.length >= 2}
                />
                <span className="truncate">{person.name}</span>
                <span className="shrink-0 text-xs text-ink-faint">
                  {knownPeople.sampleCount(person.sampleCount)}
                </span>
              </label>
              {isSelected && (
                <label className="flex shrink-0 items-center gap-1.5 text-xs text-ink-faint">
                  <input
                    type="radio"
                    name="keep-person"
                    checked={keepId === person.id}
                    onChange={() => setKeepId(person.id)}
                  />
                  {copy.peopleMergeKeepLabel}
                </label>
              )}
            </div>
          );
        })}
      </div>
      {error && <p className="mt-3 text-xs text-live">{error}</p>}
    </Modal>
  );
}
