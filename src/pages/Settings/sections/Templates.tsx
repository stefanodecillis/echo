import { useEffect, useState } from "react";

import { Card } from "../../../components/Card";
import { Chip } from "../../../components/Chip";
import { EmptyState } from "../../../components/EmptyState";
import { PlusIcon } from "../../../components/icons";
import { deleteTemplate, listTemplates, saveTemplate } from "../../../lib/ipc";
import { common, settings as copy } from "../../../lib/copy";
import type { Template, TemplateDraft } from "../../../lib/types";
import { TemplateEditorModal } from "../components/TemplateEditorModal";
import type { SectionProps } from "../types";

/** Recap styles: the six built-ins, read-only, plus whatever custom ones the
 * person has written — with a simple create/edit/delete flow for those. */
export function Templates(_props: SectionProps) {
  const [templates, setTemplates] = useState<Template[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState<Template | "new" | null>(null);

  useEffect(() => {
    refresh();
  }, []);

  function refresh() {
    listTemplates()
      .then(setTemplates)
      .catch((err) => setError(err.message));
  }

  async function handleSave(draft: TemplateDraft) {
    await saveTemplate(draft);
    refresh();
  }

  async function handleDelete(id: string) {
    await deleteTemplate(id);
    refresh();
  }

  const builtins = templates?.filter((t) => t.builtin) ?? [];
  const custom = templates?.filter((t) => !t.builtin) ?? [];

  return (
    <div className="flex flex-col gap-4">
      {error && <p className="text-sm text-live">{error}</p>}
      {!templates && !error && <p className="text-sm text-ink-faint">Loading…</p>}

      {templates && (
        <>
          <Card padding="none">
            <div className="divide-y divide-hairline">
              {builtins.map((t) => (
                <div key={t.id} className="flex items-center justify-between gap-3 px-5 py-3.5">
                  <p className="text-sm font-medium text-ink">{t.name}</p>
                  <Chip>{copy.templatesBuiltinBadge}</Chip>
                </div>
              ))}
            </div>
          </Card>

          <div className="flex items-center justify-between">
            <p className="text-sm font-medium text-ink">Your styles</p>
            <button
              type="button"
              onClick={() => setEditing("new")}
              className="echo-pill-quiet"
            >
              <PlusIcon className="h-4 w-4" />
              {copy.addTemplateButton}
            </button>
          </div>

          {custom.length === 0 ? (
            <EmptyState title={copy.templatesEmpty} />
          ) : (
            <Card padding="none">
              <div className="divide-y divide-hairline">
                {custom.map((t) => (
                  <div key={t.id} className="flex items-center justify-between gap-3 px-5 py-3.5">
                    <p className="min-w-0 truncate text-sm font-medium text-ink">{t.name}</p>
                    <div className="flex shrink-0 gap-4">
                      <button
                        type="button"
                        onClick={() => setEditing(t)}
                        className="text-xs font-medium text-ink-soft hover:text-ink"
                      >
                        {copy.templatesEditButton}
                      </button>
                      <button
                        type="button"
                        onClick={() => handleDelete(t.id)}
                        className="text-xs font-medium text-live hover:opacity-80"
                      >
                        {common.delete}
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </>
      )}

      <TemplateEditorModal
        open={editing !== null}
        template={editing === "new" ? undefined : (editing ?? undefined)}
        onClose={() => setEditing(null)}
        onSave={handleSave}
      />
    </div>
  );
}
