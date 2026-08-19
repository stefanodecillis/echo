import { useState } from "react";

import { Button } from "../../../components/Button";
import { Modal } from "../../../components/Modal";
import { common } from "../../../lib/copy";

export interface TypedConfirmModalProps {
  open: boolean;
  title: string;
  description: string;
  /** The exact phrase (case-insensitive) the person must type to enable the
   * confirm button — the extra friction a "delete everything" needs. */
  confirmWord: string;
  confirmLabel: string;
  onClose: () => void;
  onConfirm: () => Promise<void>;
}

/** A destructive confirmation that isn't satisfied by a single click — used
 * once, for Settings > Data > Delete everything. */
export function TypedConfirmModal({
  open,
  title,
  description,
  confirmWord,
  confirmLabel,
  onClose,
  onConfirm,
}: TypedConfirmModalProps) {
  const [typed, setTyped] = useState("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const matches = typed.trim().toLowerCase() === confirmWord.toLowerCase();

  async function confirm() {
    setRunning(true);
    setError(null);
    try {
      await onConfirm();
      setTyped("");
      onClose();
    } catch (err) {
      setError((err as { message: string }).message);
    } finally {
      setRunning(false);
    }
  }

  return (
    <Modal
      open={open}
      onClose={() => {
        setTyped("");
        onClose();
      }}
      title={title}
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            {common.cancel}
          </Button>
          <Button variant="primary" disabled={!matches} loading={running} onClick={confirm}>
            {confirmLabel}
          </Button>
        </>
      }
    >
      <div className="flex flex-col gap-3">
        <p>{description}</p>
        <input
          autoFocus
          value={typed}
          onChange={(e) => setTyped(e.target.value)}
          placeholder={confirmWord}
          className="w-full rounded-xl border border-hairline bg-surface px-3 py-2 text-sm text-ink placeholder:text-ink-ghost focus:outline-none focus:ring-2 focus:ring-live/30"
        />
        {error && <p className="text-xs text-live">{error}</p>}
      </div>
    </Modal>
  );
}
