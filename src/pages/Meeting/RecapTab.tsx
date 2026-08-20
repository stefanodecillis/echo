import { useEffect, useMemo, useState } from "react";

import { Button, EmptyState, ProgressBar } from "@/components";
import { CheckIcon, CopyIcon, EchoMark } from "@/components/icons";
import { common, meeting as copy, notices } from "@/lib/copy";
import { updateActionItem } from "@/lib/ipc";
import { useEchoStore } from "@/lib/store";
import type { ActionItem, Id, Job, MeetingDetail } from "@/lib/types";

import { ActionItemRow } from "./components/ActionItemRow";
import { ExportMenu } from "./components/ExportMenu";
import { RecapControls } from "./components/RecapControls";
import { RecapMarkdown } from "./lib/markdown";

export interface RecapTabProps {
  meetingId: Id;
  meetingTitle: string;
  detail: MeetingDetail;
  setDetail: (updater: (detail: MeetingDetail) => MeetingDetail) => void;
}

/** The Recap tab: the rendered summary, the action-items checklist, and the
 * controls to write or rewrite it. */
export function RecapTab({ meetingId, meetingTitle, detail, setDetail }: RecapTabProps) {
  const addToast = useEchoStore((s) => s.addToast);
  const [pendingJobId, setPendingJobId] = useState<Id>();
  const [copied, setCopied] = useState(false);

  const currentSummary = useMemo(() => {
    if (detail.summaries.length === 0) return undefined;
    return [...detail.summaries].sort((a, b) => b.createdAt.localeCompare(a.createdAt))[0];
  }, [detail.summaries]);

  const activeJob: Job | undefined = useMemo(() => {
    if (pendingJobId) {
      const byId = detail.jobs.find((j) => j.id === pendingJobId);
      if (byId) return byId;
    }
    return detail.jobs.find(
      (j) => j.kind === "summarize" && (j.status === "running" || j.status === "queued"),
    );
  }, [detail.jobs, pendingJobId]);

  useEffect(() => {
    if (!pendingJobId) return;
    const job = detail.jobs.find((j) => j.id === pendingJobId);
    if (job && (job.status === "done" || job.status === "failed" || job.status === "cancelled")) {
      if (job.status === "failed") {
        addToast({ level: "problem", message: job.error ?? notices.somethingWentWrong });
      }
      setPendingJobId(undefined);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [detail.jobs, pendingJobId]);

  const copyRecap = async () => {
    if (!currentSummary) return;
    await navigator.clipboard.writeText(currentSummary.contentMd);
    setCopied(true);
    addToast({ level: "info", message: notices.copiedToClipboard });
    setTimeout(() => setCopied(false), 2000);
  };

  const applyActionItemPatch = (
    item: ActionItem,
    patch: Partial<Pick<ActionItem, "done" | "owner">>,
  ) => {
    const optimistic = { ...item, ...patch };
    setDetail((d) => ({
      ...d,
      actionItems: d.actionItems.map((i) => (i.id === item.id ? optimistic : i)),
    }));
    updateActionItem({ id: item.id, ...patch }).catch(() => {
      // The next `actionItemsUpdated` event (or a reload) reconciles this;
      // no need to roll the optimistic update back mid-interaction.
      addToast({ level: "problem", message: notices.somethingWentWrong });
    });
  };

  return (
    <div className="flex flex-col gap-8 px-8 py-6">
      <section className="flex flex-col gap-3">
        {/* Content first: the recap itself, the "still writing" progress, or —
            with nothing yet — the one card that gets one written. Whichever
            it is, it's the thing this tab is for, so it comes before any
            action around it. */}
        {currentSummary ? (
          <div className="echo-card p-6">
            <RecapMarkdown content={currentSummary.contentMd} />
          </div>
        ) : activeJob ? (
          <div className="echo-card flex flex-col items-center gap-4 px-6 py-16 text-center">
            <h3 className="text-base font-semibold text-ink">{copy.recapWritingTitle}</h3>
            <p className="max-w-sm text-sm text-ink-faint">{copy.recapWritingDescription}</p>
            <div className="w-full max-w-xs">
              <ProgressBar value={activeJob.progress} />
            </div>
          </div>
        ) : (
          <div className="echo-card">
            <EmptyState
              icon={<EchoMark className="h-8 w-8" />}
              title={copy.noRecapTitle}
              description={copy.noRecapDescription}
              action={
                <RecapControls
                  meetingId={meetingId}
                  hasSummary={false}
                  activeJob={activeJob}
                  onQueued={setPendingJobId}
                />
              }
            />
          </div>
        )}

        {/* Actions stay quiet underneath: reachable, never louder than the
            content above. Export is always here — exporting the transcript
            never depended on a summary provider being set up (DESIGN §0 —
            recording and transcripts work with none configured), so it
            can't wait behind a recap existing. */}
        <div className="flex flex-wrap items-center justify-end gap-2">
          {currentSummary && (
            <>
              <RecapControls
                meetingId={meetingId}
                hasSummary
                activeJob={activeJob}
                onQueued={setPendingJobId}
                subtle
              />
              <Button
                variant="ghost"
                size="sm"
                leftIcon={copied ? <CheckIcon /> : <CopyIcon />}
                onClick={copyRecap}
              >
                {copied ? common.copied : common.copy}
              </Button>
            </>
          )}
          <ExportMenu
            meetingId={meetingId}
            meetingTitle={meetingTitle}
            summaryId={currentSummary?.id}
            size="sm"
          />
        </div>
      </section>

      {currentSummary && (
        <section className="flex flex-col gap-2">
          <h2 className="text-sm font-semibold text-ink">{copy.actionItemsTitle}</h2>
          {detail.actionItems.length === 0 ? (
            <p className="text-sm text-ink-faint">{copy.actionItemsEmpty}</p>
          ) : (
            <ul className="divide-y divide-hairline">
              {detail.actionItems.map((item) => (
                <ActionItemRow
                  key={item.id}
                  item={item}
                  onToggleDone={(i) => applyActionItemPatch(i, { done: !i.done })}
                  onOwnerChange={(i, owner) => applyActionItemPatch(i, { owner: owner || undefined })}
                />
              ))}
            </ul>
          )}
        </section>
      )}
    </div>
  );
}
