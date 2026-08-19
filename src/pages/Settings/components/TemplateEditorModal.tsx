import { useEffect, useState } from "react";

import { Button } from "../../../components/Button";
import { Field } from "../../../components/Input";
import { Modal } from "../../../components/Modal";
import { common, settings as copy } from "../../../lib/copy";
import type { Template, TemplateDraft } from "../../../lib/types";

export interface TemplateEditorModalProps {
  open: boolean;
  /** Present when editing an existing custom style; absent for a new one. */
  template?: Template;
  onClose: () => void;
  onSave: (draft: TemplateDraft) => Promise<void>;
}

/** The "simple editor" the spec calls for: a name and a plain-language
 * instruction for how the recap should read. Nothing fancier — a custom
 * style is meant to be something a non-technical person can actually write. */
export function TemplateEditorModal({
  open,
  template,
  onClose,
  onSave,
}: TemplateEditorModalProps) {
  const [name, setName] = useState(template?.name ?? "");
  const [promptMd, setPromptMd] = useState(template?.promptMd ?? "");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setName(template?.name ?? "");
      setPromptMd(template?.promptMd ?? "");
      setError(null);
    }
  }, [open, template]);

  async function submit() {
    setSaving(true);
    setError(null);
    try {
      await onSave({ id: template?.id, name, promptMd });
      onClose();
    } catch (err) {
      setError((err as { message: string }).message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title={template ? copy.templatesEditTitle : copy.templatesNewTitle}
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            {common.cancel}
          </Button>
          <Button
            variant="primary"
            loading={saving}
            disabled={!name.trim() || !promptMd.trim()}
            onClick={submit}
          >
            {common.save}
          </Button>
        </>
      }
    >
      <div className="flex flex-col gap-4">
        <Field label={copy.templatesNameLabel}>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="w-full rounded-xl border border-hairline bg-surface px-3 py-2 text-sm text-ink placeholder:text-ink-ghost focus:outline-none focus:ring-2 focus:ring-accent/30"
          />
        </Field>
        <Field label={copy.templatesPromptLabel} hint={copy.templatesPromptHint}>
          <textarea
            value={promptMd}
            onChange={(e) => setPromptMd(e.target.value)}
            rows={8}
            className="w-full resize-y rounded-xl border border-hairline bg-surface px-3 py-2 text-sm text-ink placeholder:text-ink-ghost focus:outline-none focus:ring-2 focus:ring-accent/30"
          />
        </Field>
        {error && <p className="text-xs text-live">{error}</p>}
      </div>
    </Modal>
  );
}
