import { useEffect, useMemo, useState } from "react";

import { Button, EmptyState, ProgressBar } from "@/components";
import { CheckIcon, CopyIcon, EchoMark } from "@/components/icons";
import { common, jobLine, labels, meeting as copy, notices } from "@/lib/copy";
import { updateActionItem } from "@/lib/ipc";
import { inProgressJob, isActive, jobSentence, lastFailure, presentJob } from "@/lib/jobs";
import { useEchoStore } from "@/lib/store";
import type { ActionItem, Id, Job, MeetingDetail } from "@/lib/types";

import { ActionItemRow } from "./components/ActionItemRow";
import { ExportMenu } from "./components/ExportMenu";
import { RecapControls } from "./components/RecapControls";
import { RecapMarkdown } from "./lib/markdown";

/** Job kinds that mean "Echo is still writing this meeting down".
 *
 * The same pair the Transcript tab watches, for the same reason: there is no
 * dedicated kind for "the transcript is being written", it is these two passes.
 * The recap depends on their output, and the queue already runs them first — a
 * queued recap is ordered by kind, so it starts after that meeting's transcript
 * work whatever order things were asked for in. Nothing here changes that; this
 * is only the tab finally saying it. */
const TRANSCRIPT_JOB_KINDS = new Set(["transcribeCatchup", "diarize"]);

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

  // Is Echo still writing this meeting down? The one that is *running*, and
  // only failing that the one that is waiting (see `lib/jobs.ts`).
  const transcriptJob = useMemo(
    () =>
      inProgressJob(
        detail.jobs.filter((j) => j.meetingId === meetingId && isActive(j)),
        TRANSCRIPT_JOB_KINDS,
      ),
    [detail.jobs, meetingId],
  );
  const transcriptProgress = transcriptJob ? presentJob(transcriptJob) : undefined;

  // A transcript pass that stopped without finishing. This is the edge that
  // matters most here: the queue's ordering guarantees a recap comes *after*
  // the transcript work, not that the transcript work succeeded. Writing a
  // recap from a half-written transcript and saying nothing is the one outcome
  // a person could not detect. Only while nothing is running — work in progress
  // is the more useful thing to say about the same passes.
  const transcriptFailure = useMemo(
    () =>
      transcriptJob
        ? undefined
        : lastFailure(
            detail.jobs.filter((j) => j.meetingId === meetingId),
            TRANSCRIPT_JOB_KINDS,
          ),
    [detail.jobs, meetingId, transcriptJob],
  );

  /** What the transcript is doing, said underneath whatever the recap is
   * doing. `undefined` once the transcript is simply finished. */
  const transcriptNote = transcriptProgress ? (
    <span className="text-xs text-ink-faint">{jobSentence(transcriptProgress)}</span>
  ) : transcriptFailure ? (
    <>
      <span className="text-xs font-medium text-ink-soft">
        {jobLine.stopped(labels.jobKind[transcriptFailure.kind])}
      </span>
      <span className="text-xs text-ink-faint">
        {transcriptFailure.error ?? notices.somethingWentWrong}
      </span>
    </>
  ) : undefined;

  const recapProgress = activeJob ? presentJob(activeJob) : undefined;

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
        ) : activeJob && recapProgress ? (
          // Asked for, but not necessarily happening. A queued recap used to be
          // labelled "Writing your recap…" over a moving bar, which is the
          // wrong word for waiting — and while the transcript was still being
          // written, the wrong *thing* to be talking about. Names what is
          // actually true, and says what it is waiting for.
          <div className="echo-card flex flex-col items-center gap-4 px-6 py-16 text-center">
            <h3 className="text-base font-semibold text-ink">
              {recapProgress.running ? copy.recapWritingTitle : copy.recapWaitingTitle}
            </h3>
            <p className="max-w-sm text-sm text-ink-faint">
              {transcriptProgress
                ? copy.recapWaitingForTranscriptDescription
                : transcriptFailure
                  ? copy.recapTranscriptStoppedDescription
                  : recapProgress.running
                    ? copy.recapWritingDescription
                    : copy.recapWaitingDescription}
            </p>
            {/* A bar only for work that has started: a job that has not begun
                has made no progress to show (see `lib/jobs.ts`). */}
            {recapProgress.running ? (
              <div className="w-full max-w-xs">
                <ProgressBar value={activeJob.progress} />
              </div>
            ) : (
              <span className="text-xs text-ink-faint">{jobSentence(recapProgress)}</span>
            )}
            {transcriptNote && (
              <div className="flex flex-col items-center gap-1">{transcriptNote}</div>
            )}
          </div>
        ) : (
          <div className="echo-card">
            <EmptyState
              icon={<EchoMark className="h-8 w-8" />}
              title={
                transcriptProgress
                  ? copy.noRecapWhileTranscribingTitle
                  : transcriptFailure
                    ? copy.recapTranscriptStoppedTitle
                    : copy.noRecapTitle
              }
              description={
                transcriptProgress
                  ? copy.noRecapWhileTranscribingDescription
                  : transcriptFailure
                    ? copy.recapTranscriptStoppedDescription
                    : copy.noRecapDescription
              }
              action={
                // Deliberately still enabled while the transcript is being
                // written. The queue orders a recap after that meeting's
                // transcript work, so this button makes a promise Echo already
                // keeps — better than a disabled control that has to explain
                // why it can't be pressed.
                <RecapControls
                  meetingId={meetingId}
                  hasSummary={false}
                  activeJob={activeJob}
                  onQueued={setPendingJobId}
                />
              }
            />
            {transcriptNote && (
              <div className="flex flex-col items-center gap-1 border-t border-hairline px-6 py-3 text-center">
                {transcriptNote}
              </div>
            )}
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
