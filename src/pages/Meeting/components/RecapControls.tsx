import { useEffect, useState } from "react";

import { Button, ProgressBar } from "@/components";
import { labels, meeting as copy } from "@/lib/copy";
import { generateSummary, listTemplates } from "@/lib/ipc";
import type { Id, Job, Template } from "@/lib/types";

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

/** Recap style and the button that kicks the whole thing off — everything
 * the Recap tab needs above the fold when there's work still to do. Who
 * actually writes it (provider + model) is a Settings concern, not this
 * page's. */
export function RecapControls({ meetingId, hasSummary, activeJob, onQueued }: RecapControlsProps) {
  const [templates, setTemplates] = useState<Template[]>([]);
  const [templateId, setTemplateId] = useState<Id>();
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
  }, []);

  const running = activeJob?.status === "running" || activeJob?.status === "queued" || queuing;

  const write = async () => {
    setQueuing(true);
    try {
      const jobId = await generateSummary({
        meetingId,
        templateId,
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

        <Button variant="primary" loading={running} onClick={write} className="ml-auto">
          {hasSummary ? copy.regenerateButton : copy.writeRecapButton}
        </Button>
      </div>

      {/* Only while an existing recap is being rewritten. With no recap yet, the
          tab below this fills its whole panel with "Writing your recap…" and a
          bar of its own, and saying the same thing twice, two lines apart, reads
          like two things are happening. */}
      {hasSummary && activeJob && (activeJob.status === "running" || activeJob.status === "queued") && (
        <div className="flex flex-col gap-1.5">
          <ProgressBar value={activeJob.progress} label={labels.jobKind.summarize} />
          <span className="text-xs text-ink-faint">{labels.jobKind.summarize}…</span>
        </div>
      )}
    </div>
  );
}
