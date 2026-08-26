import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";

import {
  Button,
  EmptyState,
  Modal,
  ProgressBar,
  SearchInput,
  VirtualList,
  type VirtualListHandle,
} from "@/components";
import { CheckIcon, CombineIcon, CopyIcon, ReplayIcon } from "@/components/icons";
import { cx } from "@/components/lib/cx";
import {
  EVENTS,
  getTranscript,
  leftOutMoments,
  listSpeakers,
  retranscribeMeeting,
  toUiError,
  undoCorrections,
} from "@/lib/ipc";
import {
  common,
  jobLine,
  labels,
  leftOut as leftOutCopy,
  meeting as copy,
  notices,
  peopleCount as peopleCountCopy,
  transcriptNote,
} from "@/lib/copy";
import { inProgressJob, isActive, jobSentence, lastFailure, presentJob } from "@/lib/jobs";
import { useEvent } from "@/hooks/useEvent";
import { useEchoStore } from "@/lib/store";
import type { Id, LeftOutMoment, MeetingDetail, Segment } from "@/lib/types";

import { ExportMenu } from "./components/ExportMenu";
import { IconButton } from "./components/IconButton";
import { MergeSpeakersModal } from "./components/MergeSpeakersModal";
import { SpeakerChip } from "./components/SpeakerChip";
import { SpeakersDialog } from "./components/SpeakersDialog";
import { correctedSpans } from "./lib/corrections";
import { formatTimestamp } from "./lib/date";
import type { MeetingDetailWithPeopleCount } from "./lib/peopleCount";
import { howSureThisMeetingIs } from "./lib/confidence";
import { canonicalSpeakers, resolveSpeaker } from "./lib/speakers";
import { buildTranscriptText } from "./lib/transcriptText";

/** Job kinds that mean "the transcript is being read from the recording
 * again" — the same offline pass a fresh recording gets, re-run on demand.
 * There is no dedicated kind for it; reusing these is the point (the tab
 * already knows how to show them working). */
const TRANSCRIBE_JOB_KINDS = new Set(["transcribeCatchup", "diarize"]);

const ROW_HEIGHT = 96;

/** How many left-out moments are listed by time before the rest are counted
 * instead. The count in the heading is always the whole truth; this list is a
 * way into the transcript, and fifty timestamps in a row is a wall rather than
 * a list. */
const LEFT_OUT_TIMES_SHOWN = 12;

export interface TranscriptTabProps {
  meetingId: Id;
  detail: MeetingDetail;
  setDetail: (updater: (detail: MeetingDetail) => MeetingDetail) => void;
  /** Milliseconds to scroll to once the transcript has loaded — set when
   * arriving from a search hit. Consumed once. */
  jumpToMs?: number;
  onJumpConsumed: () => void;
  /** How many voices the last speaker pass could tell apart, if one has
   * finished since this page opened. Only "Name your speakers" uses it. */
  voicesFound?: number;
  /** The best count that pass decided against, when the count is Echo's own. */
  alternativeCount?: number;
}

function highlightMatches(text: string, query: string): ReactNode {
  if (!query.trim()) return text;
  const lower = text.toLowerCase();
  const needle = query.toLowerCase();
  const parts: ReactNode[] = [];
  let start = 0;
  let index = lower.indexOf(needle, start);
  let key = 0;
  while (index !== -1) {
    if (index > start) parts.push(text.slice(start, index));
    parts.push(
      <mark key={key++} className="rounded bg-accent-soft text-ink">
        {text.slice(index, index + needle.length)}
      </mark>,
    );
    start = index + needle.length;
    index = lower.indexOf(needle, start);
  }
  if (start < text.length) parts.push(text.slice(start));
  return parts;
}

/** The Transcript tab: a virtualized, filterable list of segments with
 * clickable speaker chips for renaming, and a "combine two speakers" flow
 * for the provisional labels an offline pass hasn't stabilized yet. */
export function TranscriptTab({
  meetingId,
  detail,
  setDetail,
  jumpToMs,
  onJumpConsumed,
  voicesFound,
  alternativeCount,
}: TranscriptTabProps) {
  const [segments, setSegments] = useState<Segment[]>();
  const [filter, setFilter] = useState("");
  const [mergeOpen, setMergeOpen] = useState(false);
  const [speakersOpen, setSpeakersOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const [confirmRetranscribe, setConfirmRetranscribe] = useState(false);
  const [retranscribing, setRetranscribing] = useState(false);
  /** The line whose repairs are being put back right now, so its words stop
   * taking clicks while the core answers. One at a time is enough: the click
   * target is a word, and nobody double-clicks two of them. */
  const [undoingId, setUndoingId] = useState<Id>();
  /** Moments Echo heard and decided not to write down that nothing else wrote
   * down either. Empty for almost every meeting, and empty is silence. */
  const [leftOut, setLeftOut] = useState<LeftOutMoment[]>([]);
  const listRef = useRef<VirtualListHandle>(null);
  const addToast = useEchoStore((s) => s.addToast);

  useEffect(() => {
    let cancelled = false;
    getTranscript({ meetingId })
      .then((result) => {
        if (!cancelled) setSegments(result);
      })
      .catch(() => {
        if (!cancelled) setSegments([]);
      });
    return () => {
      cancelled = true;
    };
  }, [meetingId]);

  useEvent(EVENTS.transcriptFinal, (payload) => {
    if (payload.meetingId !== meetingId) return;
    setSegments((prev) => {
      const list = prev ?? [];
      const existingIndex = list.findIndex((s) => s.id === payload.segment.id);
      if (existingIndex === -1) return [...list, payload.segment];
      const next = [...list];
      next[existingIndex] = payload.segment;
      return next;
    });
  });

  // "Listen again" rewrites the transcript wholesale rather than touching a
  // known range, so the simplest correct response to a revision is a full
  // reload rather than trying to patch individual segments in place.
  useEvent(EVENTS.transcriptRevised, (payload) => {
    if (payload.meetingId !== meetingId) return;
    getTranscript({ meetingId })
      .then((result) => setSegments(result))
      .catch(() => {
        // The job-progress event already reflects that something is
        // happening; a later revision or a manual reload picks this up.
      });
  });

  // The one that is *running*, and only failing that the one that is waiting.
  // Picking whichever came first in the list named the job at the back of the
  // queue while another one worked (see `lib/jobs.ts`).
  const transcribeJob = inProgressJob(
    detail.jobs.filter((j) => j.meetingId === meetingId && isActive(j)),
    TRANSCRIBE_JOB_KINDS,
  );
  const transcribeProgress = transcribeJob ? presentJob(transcribeJob) : undefined;

  // A pass that stopped without finishing. Nothing used to show this, so the
  // most common way for Echo to fail to tell voices apart — the speaker files
  // are allowed to land after the first recording — arrived as a transcript
  // where every line said "You" and no explanation anywhere, after a banner
  // that had promised the opposite. Only while nothing is running: work in
  // progress is the more useful thing to say about the same passes.
  const transcribeFailure = lastFailure(
    detail.jobs.filter((j) => j.meetingId === meetingId),
    TRANSCRIBE_JOB_KINDS,
  );

  const filtered = useMemo(() => {
    const list = segments ?? [];
    const query = filter.trim().toLowerCase();
    if (!query) return list;
    return list.filter((s) => s.text.toLowerCase().includes(query));
  }, [segments, filter]);

  // Over the whole meeting, deliberately — not `filtered`. The bar is partly
  // "how does this line compare with the rest of this meeting", and a search
  // box is not a meeting: typing a word must not change which lines read as
  // shaky.
  const sureness = useMemo(() => howSureThisMeetingIs(segments ?? []), [segments]);

  /** Scroll to the first line at or after a moment on the meeting clock. What
   * is on screen is the filtered list, so that is what the index has to count:
   * jumping by position in the unfiltered transcript lands somewhere else
   * entirely while a filter is typed. */
  const scrollToMoment = (ms: number) => {
    if (filtered.length === 0) return;
    const index = filtered.findIndex((s) => s.tStartMs >= ms);
    const target = index === -1 ? filtered.length - 1 : index;
    listRef.current?.scrollToIndex(Math.max(0, target));
  };

  useEffect(() => {
    if (jumpToMs === undefined || !segments || segments.length === 0) return;
    scrollToMoment(jumpToMs);
    onJumpConsumed();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jumpToMs, segments]);

  // Only asked once nothing is being recorded or read back. Both halves of the
  // question move while a pass runs — decisions are still being made, and the
  // words that cover them are still arriving — so an answer read mid-flight
  // would name moments a minute before they were filled in. Coming out of a
  // pass flips this back and asks again.
  const transcriptSettled =
    detail.meeting.status !== "recording" && !transcribeJob && !retranscribing;

  useEffect(() => {
    // Cleared first, always: a moment belongs to one meeting and one reading of
    // it, so nothing from the last of either is left standing on screen while a
    // fresh answer is on its way.
    setLeftOut([]);
    if (!transcriptSettled) return;
    let cancelled = false;
    leftOutMoments(meetingId)
      .then((moments) => {
        if (!cancelled) setLeftOut(moments);
      })
      .catch(() => {
        // A question that couldn't be asked is not an answer: nothing is said,
        // rather than a failed read turning into a claim about the transcript.
      });
    return () => {
      cancelled = true;
    };
  }, [meetingId, transcriptSettled]);

  const handleMerged = () => {
    listSpeakers(meetingId)
      .then((speakers) => setDetail((d) => ({ ...d, speakers })))
      .catch(() => {
        // The speakers-updated event, if it arrives, reconciles this.
      });
  };

  const handleCopyTranscript = async () => {
    if (!segments || segments.length === 0) return;
    const text = buildTranscriptText(segments, detail.speakers);
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      addToast({ level: "info", message: notices.copiedToClipboard });
      setTimeout(() => setCopied(false), 2000);
    } catch {
      addToast({ level: "problem", message: notices.somethingWentWrong });
    }
  };

  /**
   * "No, it heard that right" — the whole line goes back to the words the
   * engine wrote.
   *
   * The core also announces the change on `transcriptRevised`, which this tab
   * answers with a full reload; writing the returned line into place as well
   * means the word changes under the cursor that clicked it rather than a beat
   * later. Both land on the same row, so the order they arrive in makes no
   * difference.
   *
   * A refusal is an explanation, not a failure: the core's sentence says the
   * line has been written down again since, and there is nothing to retry.
   */
  const handleUndoCorrections = async (segmentId: Id) => {
    if (undoingId) return;
    setUndoingId(segmentId);
    try {
      const restored = await undoCorrections(segmentId);
      setSegments((prev) =>
        prev?.map((s) => (s.id === restored.id ? restored : s)) ?? prev,
      );
      addToast({ level: "info", message: transcriptNote.undone });
    } catch (err) {
      addToast({ level: "problem", message: toUiError(err).message });
    } finally {
      setUndoingId(undefined);
    }
  };

  const handleRetranscribe = async () => {
    setConfirmRetranscribe(false);
    setRetranscribing(true);
    try {
      await retranscribeMeeting(meetingId);
    } catch (err) {
      addToast({ level: "problem", message: toUiError(err).message });
    } finally {
      setRetranscribing(false);
    }
  };

  const hasAudio = detail.audioBytes > 0 && !detail.meeting.deletedAt;

  // `getMeeting` gains these two alongside `speakers` (see `lib/peopleCount`);
  // widened locally rather than in the shared type until that lands.
  const detailWithPeopleCount = detail as MeetingDetailWithPeopleCount;
  const rewriteRunning = retranscribing || !!transcribeJob;
  const peopleCountDisabled = detail.meeting.status === "recording" || rewriteRunning;

  if (segments === undefined) {
    return <div className="px-8 py-6 text-sm text-ink-faint">{common.loading}</div>;
  }

  return (
    <div className="flex h-full flex-col gap-4 px-8 py-6">
      <div className="flex flex-wrap items-center gap-3">
        <SearchInput
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder={copy.transcriptFilterPlaceholder}
          containerClassName="max-w-sm"
        />
        {typeof detailWithPeopleCount.peopleCount === "number" && (
          <button
            type="button"
            onClick={() => setSpeakersOpen(true)}
            disabled={peopleCountDisabled}
            title={copy.speakersDialogTitle}
            className="rounded-full px-2.5 py-1 text-xs font-medium text-ink-faint outline-none transition-colors hover:bg-surface-sunken hover:text-ink-soft focus-visible:ring-2 focus-visible:ring-accent/40 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {peopleCountCopy.triggerLabel(
              detailWithPeopleCount.peopleCount,
              !detailWithPeopleCount.peopleCountIsOverride,
            )}
          </button>
        )}
        <div className="ml-auto flex items-center gap-1">
          {detail.speakers.length > 1 && (
            <IconButton
              icon={<CombineIcon />}
              aria-label={copy.mergeSpeakersButton}
              title={copy.mergeSpeakersButton}
              onClick={() => setMergeOpen(true)}
            />
          )}
          <IconButton
            icon={copied ? <CheckIcon /> : <CopyIcon />}
            aria-label={copied ? common.copied : copy.copyTranscriptButton}
            title={copy.copyTranscriptButton}
            disabled={segments.length === 0}
            onClick={handleCopyTranscript}
          />
          <IconButton
            icon={<ReplayIcon />}
            aria-label={copy.listenAgainButton}
            title={copy.listenAgainButton}
            disabled={!hasAudio || retranscribing || !!transcribeJob}
            onClick={() => setConfirmRetranscribe(true)}
          />
          <ExportMenu meetingId={meetingId} meetingTitle={detail.meeting.title} size="sm" />
        </div>
      </div>

      {transcribeProgress && (
        <div className="flex flex-col gap-1.5 rounded-xl border border-hairline bg-surface-sunken p-4">
          {transcribeProgress.running && (
            <ProgressBar
              value={transcribeProgress.fraction}
              label={transcribeProgress.label}
            />
          )}
          <span className="text-xs text-ink-faint">
            {jobSentence(transcribeProgress)}
          </span>
        </div>
      )}

      {!transcribeProgress && transcribeFailure && (
        <div className="flex flex-col gap-1 rounded-xl border border-hairline bg-surface-sunken p-4">
          <span className="text-xs font-medium text-ink-soft">
            {jobLine.stopped(labels.jobKind[transcribeFailure.kind])}
          </span>
          <span className="text-xs text-ink-faint">
            {transcribeFailure.error ?? notices.somethingWentWrong}
          </span>
        </div>
      )}

      {/* Echo left these seconds out and nothing else wrote them down, so the
          transcript reads there exactly like one where nobody spoke. Said
          after the fact, on the meeting's own screen: never in the Live view,
          where the reading isn't finished, and never in an export, which is
          somebody else's document. Nothing at all when there are none. */}
      {leftOut.length > 0 && (
        <div className="flex flex-col gap-1.5 rounded-xl border border-hairline bg-surface-sunken p-4">
          <span className="text-xs font-medium text-ink-soft">
            {leftOutCopy.title(leftOut.length)}
          </span>
          <span className="text-xs text-ink-faint">
            {leftOutCopy.explanation(leftOut.length)}
          </span>
          <div className="flex flex-wrap items-center gap-1.5 pt-0.5">
            {leftOut.slice(0, LEFT_OUT_TIMES_SHOWN).map((moment) => {
              const time = formatTimestamp(moment.tStartMs);
              return (
                <button
                  key={moment.tStartMs}
                  type="button"
                  title={leftOutCopy.jumpLabel(time)}
                  aria-label={leftOutCopy.jumpLabel(time)}
                  onClick={() => scrollToMoment(moment.tStartMs)}
                  className="rounded-full bg-surface px-2 py-0.5 text-xs tabular-nums text-ink-faint outline-none transition-colors hover:text-ink focus-visible:ring-2 focus-visible:ring-accent/40"
                >
                  {time}
                </button>
              );
            })}
            {leftOut.length > LEFT_OUT_TIMES_SHOWN && (
              <span className="text-xs text-ink-ghost">
                {leftOutCopy.andMore(leftOut.length - LEFT_OUT_TIMES_SHOWN)}
              </span>
            )}
          </div>
          {/* Only while there is still a recording to read: "Listen again" is
              disabled without one, and a repair that cannot run is worse than
              none offered. */}
          {hasAudio && (
            <span className="text-xs text-ink-faint">
              {leftOutCopy.repair(copy.listenAgainButton)}
            </span>
          )}
        </div>
      )}

      {filtered.length === 0 ? (
        <EmptyState title={filter ? copy.transcriptNoMatches : copy.transcriptEmpty} />
      ) : (
        <VirtualList
          ref={listRef}
          items={filtered}
          itemHeight={ROW_HEIGHT}
          getKey={(s) => s.id}
          className="flex-1"
          renderItem={(segment) => {
            const speaker = resolveSpeaker(segment.speakerId, detail.speakers);
            const unsure = sureness.isShaky(segment.avgConfidence);
            const corrections = segment.corrections ?? [];
            const corrected = corrections.length > 0;
            // Non-null exactly when the words Echo repaired can still be
            // pointed at in the line — which is exactly when the core will put
            // them back. `null` keeps the mark on the whole line and offers
            // nothing, because there is nothing that can be undone.
            const spans = corrected ? correctedSpans(segment.text, corrections) : null;
            // Both can be true of one line, so both have to be sayable at once.
            const note = [
              unsure ? transcriptNote.unsure : undefined,
              corrected && !spans ? transcriptNote.corrected(corrections) : undefined,
            ]
              .filter(Boolean)
              .join(" ");
            // The tooltip moves onto the word when the word is the thing you
            // click: an inner `title` is the one a browser shows, so it has to
            // carry everything the line's would have said.
            const wordNote = [
              unsure ? transcriptNote.unsure : undefined,
              transcriptNote.corrected(corrections),
              transcriptNote.undoHint,
            ]
              .filter(Boolean)
              .join(" ");
            return (
              <div className="flex gap-4 border-b border-hairline px-1 py-3">
                <span className="w-12 shrink-0 pt-0.5 text-xs tabular-nums text-ink-ghost">
                  {formatTimestamp(segment.tStartMs)}
                </span>
                <div className="flex min-w-0 flex-1 flex-col gap-1.5">
                  <SpeakerChip
                    speaker={speaker}
                    fallbackId={segment.speakerId}
                    onOpen={() => setSpeakersOpen(true)}
                  />
                  <p
                    className={cx(
                      "line-clamp-3 text-sm leading-relaxed",
                      // A line Echo wasn't sure it heard reads a shade quieter,
                      // and says why on hover. `text-ink-faint` is already this
                      // app's word for "less certain" (Live/TranscriptLine uses
                      // it for a line that is still arriving), so nothing new is
                      // being taught here.
                      //
                      // Deliberately *not* the dotted underline below: that
                      // already means "a word was put right", and a line can be
                      // both unsure and corrected. Two meanings on one mark
                      // leaves the hover unable to say which it is. Colour and
                      // underline compose; two underlines do not.
                      unsure ? "text-ink-faint" : "text-ink-soft",
                      // A word Echo put right against the list in Settings. The
                      // whole indication: a dotted underline and a sentence on
                      // hover saying what was written and what it became. The
                      // transcript is not redesigned for this — the repair is
                      // right far more often than not, and a badge on every
                      // third line would be noise.
                      // Only when the repaired words cannot be pointed at: the
                      // line still says something on it was put right, and the
                      // underline stays where it has always been. When they can,
                      // the underline moves onto the words themselves — same
                      // mark, same meaning, now also the thing you click.
                      corrected &&
                        !spans &&
                        "decoration-hairline underline decoration-dotted underline-offset-4",
                    )}
                    title={note || undefined}
                  >
                    {spans
                      ? spans.map((span, index) =>
                          span.corrected ? (
                            <button
                              key={index}
                              type="button"
                              title={wordNote}
                              aria-label={transcriptNote.undoLabel}
                              disabled={undoingId !== undefined}
                              onClick={() => handleUndoCorrections(segment.id)}
                              className="inline rounded-sm underline decoration-hairline decoration-dotted underline-offset-4 transition-colors hover:text-ink hover:decoration-ink-ghost focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40 disabled:cursor-wait"
                            >
                              {highlightMatches(span.text, filter)}
                            </button>
                          ) : (
                            <Fragment key={index}>
                              {highlightMatches(span.text, filter)}
                            </Fragment>
                          ),
                        )
                      : highlightMatches(segment.text, filter)}
                  </p>
                </div>
              </div>
            );
          }}
        />
      )}

      <MergeSpeakersModal
        open={mergeOpen}
        onClose={() => setMergeOpen(false)}
        speakers={detail.speakers}
        onMerged={handleMerged}
      />

      <SpeakersDialog
        open={speakersOpen}
        onClose={() => setSpeakersOpen(false)}
        meetingId={meetingId}
        speakers={detail.speakers}
        peopleCount={
          typeof detailWithPeopleCount.peopleCount === "number"
            ? detailWithPeopleCount.peopleCount
            : canonicalSpeakers(detail.speakers).length
        }
        peopleCountIsOverride={!!detailWithPeopleCount.peopleCountIsOverride}
        voicesFound={voicesFound}
        alternativeCount={alternativeCount}
        job={transcribeJob}
        disabled={peopleCountDisabled}
      />

      <Modal
        open={confirmRetranscribe}
        onClose={() => setConfirmRetranscribe(false)}
        title={copy.listenAgainConfirmTitle}
        footer={
          <>
            <Button variant="secondary" onClick={() => setConfirmRetranscribe(false)}>
              {common.cancel}
            </Button>
            <Button variant="primary" loading={retranscribing} onClick={handleRetranscribe}>
              {copy.listenAgainConfirmButton}
            </Button>
          </>
        }
      >
        {copy.listenAgainConfirmDescription}
      </Modal>
    </div>
  );
}
