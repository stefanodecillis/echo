import { useEffect, useState } from "react";

import { Button, ProgressBar } from "@/components";
import { labels, meeting as copy } from "@/lib/copy";
import { generateSummary, listSummaryProviders, listTemplates, testSummaryProvider } from "@/lib/ipc";
import type { Id, Job, Provider, Template } from "@/lib/types";

export interface RecapControlsProps {
  meetingId: Id;
  /** True once at least one recap already exists — swaps the button's words
   * and forces a fresh write instead of reusing a cached one. */
  hasSummary: boolean;
  /** The in-flight summarize job for this meeting, if any — drives the
   * progress bar and disables the button while it runs. */
  activeJob?: Job;
  onQueued: (jobId: Id) => void;
}

/** Recap style, who writes it, and the button that kicks the whole thing
 * off — everything the Recap tab needs above the fold when there's work
 * still to do. */
export function RecapControls({ meetingId, hasSummary, activeJob, onQueued }: RecapControlsProps) {
  const [templates, setTemplates] = useState<Template[]>([]);
  const [templateId, setTemplateId] = useState<Id>();
  const [providers, setProviders] = useState<{ provider: Provider; available: boolean }[]>([]);
  const [provider, setProvider] = useState<Provider>();
  const [models, setModels] = useState<string[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [model, setModel] = useState<string>();
  const [queuing, setQueuing] = useState(false);

  useEffect(() => {
    listTemplates()
      .then((list) => {
        setTemplates(list);
        setTemplateId((current) => current ?? list.find((t) => t.builtin)?.id ?? list[0]?.id);
      })
      .catch(() => {
        // No styles to offer yet; the button still works with the server's
        // own default.
      });
    listSummaryProviders()
      .then((list) => {
        const mapped = list.map((p) => ({ provider: p.provider, available: p.available }));
        setProviders(mapped);
        setProvider((current) => current ?? mapped.find((p) => p.available)?.provider);
      })
      .catch(() => {
        // Same story: the write-recap button still works without a choice.
      });
  }, []);

  useEffect(() => {
    if (!provider) return;
    setModel(undefined);
    setModels([]);
    setModelsLoading(true);
    testSummaryProvider(provider)
      .then((result) => setModels(result.models))
      .catch(() => setModels([]))
      .finally(() => setModelsLoading(false));
  }, [provider]);

  const running = activeJob?.status === "running" || activeJob?.status === "queued" || queuing;

  const write = async () => {
    setQueuing(true);
    try {
      const jobId = await generateSummary({
        meetingId,
        templateId,
        provider,
        model,
        force: hasSummary,
      });
      onQueued(jobId);
    } finally {
      setQueuing(false);
    }
  };

  return (
    <div className="flex flex-col gap-3 rounded-xl border border-hairline bg-surface-sunken p-4">
      <div className="flex flex-wrap items-center gap-3">
        {templates.length > 0 && (
          <label className="flex items-center gap-2 text-xs text-ink-faint">
            {copy.templateLabel}
            <select
              value={templateId ?? ""}
              onChange={(e) => setTemplateId(e.target.value)}
              disabled={running}
              className="rounded-lg border border-hairline bg-surface px-2 py-1 text-sm text-ink"
            >
              {templates.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.name}
                </option>
              ))}
            </select>
          </label>
        )}

        {providers.length > 0 && (
          <label className="flex items-center gap-2 text-xs text-ink-faint">
            {copy.modelPickerLabel}
            <select
              value={provider ?? ""}
              onChange={(e) => setProvider(e.target.value as Provider)}
              disabled={running}
              className="rounded-lg border border-hairline bg-surface px-2 py-1 text-sm text-ink"
            >
              {providers.map((p) => (
                <option key={p.provider} value={p.provider} disabled={!p.available}>
                  {labels.provider[p.provider]}
                </option>
              ))}
            </select>
          </label>
        )}

        {provider && (modelsLoading || models.length > 0) && (
          <label className="flex items-center gap-2 text-xs text-ink-faint">
            {modelsLoading ? (
              copy.modelPickerLoading
            ) : (
              <select
                value={model ?? ""}
                onChange={(e) => setModel(e.target.value || undefined)}
                disabled={running}
                className="rounded-lg border border-hairline bg-surface px-2 py-1 text-sm text-ink"
              >
                <option value="">{copy.modelPickerDefault}</option>
                {models.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            )}
          </label>
        )}

        <Button variant="primary" loading={running} onClick={write} className="ml-auto">
          {hasSummary ? copy.regenerateButton : copy.writeRecapButton}
        </Button>
      </div>

      {activeJob && (activeJob.status === "running" || activeJob.status === "queued") && (
        <div className="flex flex-col gap-1.5">
          <ProgressBar value={activeJob.progress} label={labels.jobKind.summarize} />
          <span className="text-xs text-ink-faint">{labels.jobKind.summarize}…</span>
        </div>
      )}
    </div>
  );
}
