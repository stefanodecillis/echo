import { useState } from "react";

import { Button, Modal } from "@/components";
import { mergeSpeakers } from "@/lib/ipc";
import { common, meeting as copy, notices } from "@/lib/copy";
import type { Speaker } from "@/lib/types";

import { canonicalSpeakers } from "../lib/speakers";
import { speakerColor } from "../lib/speakerColor";

export interface MergeSpeakersModalProps {
  open: boolean;
  onClose: () => void;
  speakers: Speaker[];
  /** Called once the merge command succeeds, so the caller can refresh. */
  onMerged: () => void;
}

/** "Pick two, keep one's name" — the simplest UI that can express a
 * non-destructive merge without asking anyone to understand alias chains. */
export function MergeSpeakersModal({ open, onClose, speakers, onMerged }: MergeSpeakersModalProps) {
  const [selected, setSelected] = useState<string[]>([]);
  const [keepId, setKeepId] = useState<string>();
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string>();

  const options = canonicalSpeakers(speakers);

  const toggle = (id: string) => {
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
    setError(undefined);
  };

  const handleClose = () => {
    reset();
    onClose();
  };

  const confirm = async () => {
    if (selected.length !== 2 || !keepId) {
      setError(copy.mergeSpeakersNeedTwo);
      return;
    }
    const loserId = selected.find((id) => id !== keepId);
    if (!loserId) return;
    setPending(true);
    try {
      await mergeSpeakers(loserId, keepId);
      reset();
      onClose();
      onMerged();
    } catch {
      setError(notices.somethingWentWrong);
    } finally {
      setPending(false);
    }
  };

  return (
    <Modal
      open={open}
      onClose={handleClose}
      title={copy.mergeSpeakersTitle}
      footer={
        <>
          <Button variant="secondary" onClick={handleClose}>
            {common.cancel}
          </Button>
          <Button
            variant="primary"
            loading={pending}
            disabled={selected.length !== 2 || !keepId}
            onClick={confirm}
          >
            {copy.mergeSpeakersConfirm}
          </Button>
        </>
      }
    >
      <p className="mb-4">{copy.mergeSpeakersDescription}</p>
      <div className="flex flex-col gap-2">
        {options.map((speaker) => {
          const color = speakerColor(speaker.id);
          const isSelected = selected.includes(speaker.id);
          return (
            <div
              key={speaker.id}
              className="flex items-center justify-between gap-3 rounded-xl border border-hairline px-3 py-2"
            >
              <label className="flex flex-1 items-center gap-2.5 text-sm text-ink">
                <input
                  type="checkbox"
                  checked={isSelected}
                  onChange={() => toggle(speaker.id)}
                  disabled={!isSelected && selected.length >= 2}
                />
                <span
                  aria-hidden
                  className="h-2.5 w-2.5 rounded-full"
                  style={{ backgroundColor: color.fg }}
                />
                {speaker.displayName}
              </label>
              {isSelected && (
                <label className="flex items-center gap-1.5 text-xs text-ink-faint">
                  <input
                    type="radio"
                    name="keep-speaker"
                    checked={keepId === speaker.id}
                    onChange={() => setKeepId(speaker.id)}
                  />
                  {copy.mergeSpeakersKeepLabel}
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
