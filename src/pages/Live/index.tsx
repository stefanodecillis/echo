import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import {
  Button,
  EmptyState,
  Modal,
  RecordingDot,
  VirtualList,
  type VirtualListHandle,
} from "../../components";
import { CheckIcon } from "../../components/icons";
import { cx } from "../../components/lib/cx";
import { formatElapsed } from "../../components/lib/format";
import { useCaptureState, useCommand } from "../../hooks";
import { common, labels, live, notices } from "../../lib/copy";
import { addMarker, pauseRecording, resumeRecording, stopRecording, toUiError } from "../../lib/ipc";
import { useEchoStore } from "../../lib/store";
import { TranscriptLine } from "./TranscriptLine";
import { buildLiveTranscriptText } from "./transcriptText";
import { useSpeakerNames } from "./useSpeakerNames";
import { useTranscriptStream } from "./useTranscriptStream";

const ROW_HEIGHT = 68;

/** Being within this many pixels of the true bottom still counts as "at the
 * bottom" — enough slack that a stray pixel of scroll jitter never flips
 * someone into "detached" while they're still glued to the newest line. */
const BOTTOM_SLACK_PX = ROW_HEIGHT * 1.5;

/** The class VirtualList's real scrolling element ends up with — its
 * `className` prop lands on that element (see `components/VirtualList.tsx`),
 * so this is how the page finds the actual DOM node to watch scroll
 * position on, without VirtualList needing to expose one of its own. */
const SCROLL_EL_CLASS = "js-live-transcript-scroll";

/** How long the one-time engine setup has to last before the screen mentions
 * it. Long enough that an ordinary load comes and goes unremarked. */
const PREPARING_BANNER_DELAY_MS = 4000;

const ACTIVE_STATES = new Set(["starting", "recording", "paused", "degraded", "stopping"]);

/** States where Echo is actually taking in speech right now — the ones the
 * bottom "Listening…" row should be visible for. Not `starting` (nothing is
 * flowing yet), not `paused`, not `stopping` (already wrapping up). */
const LISTENING_ROW_STATES = new Set(["recording", "degraded"]);

/**
 * Live: the recording view. Reached automatically the moment capture starts
 * (see `routes/Home/index.tsx`'s store-driven watcher, and the tray/
 * notification path handled in `App.tsx`), and safe to open by hand — if
 * nothing is being recorded, it says so instead of showing an empty room.
 */
export default function Live() {
  const navigate = useNavigate();
  const capture = useCaptureState();
  const speakerLabelFor = useSpeakerNames(capture.meetingId);
  const lines = useTranscriptStream(capture.meetingId);
  const addToast = useEchoStore((s) => s.addToast);

  const pauseCmd = useCommand(pauseRecording);
  const resumeCmd = useCommand(resumeRecording);
  const stopCmd = useCommand(stopRecording);
  const markerCmd = useCommand(addMarker);
  const [confirmStop, setConfirmStop] = useState(false);
  const [copied, setCopied] = useState(false);

  const listRef = useRef<VirtualListHandle>(null);
  const scrollWrapperRef = useRef<HTMLDivElement>(null);
  const prevLineCount = useRef(0);

  // Whether the person is (still) glued to the newest line. Starts true —
  // the normal state when there's nothing to scroll up and away from yet.
  const [atBottom, setAtBottom] = useState(true);
  // Whether text has arrived since they scrolled away from the bottom —
  // this, not `!atBottom` alone, is what shows the "Jump to now" pill: being
  // detached to re-read something is quiet until there's actually something
  // new to jump to.
  const [awaitingJump, setAwaitingJump] = useState(false);

  // Whether the one-time engine setup has been going on long enough to be
  // worth a banner. A three-second load is not news; sixteen minutes is the
  // whole story of the meeting somebody spent watching an empty transcript
  // (incident of 2026-08-24), so the banner waits, then stays.
  const [preparingIsNews, setPreparingIsNews] = useState(false);

  const paused = capture.state === "paused";
  const active = capture.meetingId != null && ACTIVE_STATES.has(capture.state);
  const showListeningRow = LISTENING_ROW_STATES.has(capture.state);
  const hasLines = lines.length > 0;

  // Watch the real scrolling element (found by class, not a VirtualList API —
  // see `SCROLL_EL_CLASS`) so the page knows whether new text should follow
  // the person down or wait for them to come back to it.
  useEffect(() => {
    const el = scrollWrapperRef.current?.querySelector<HTMLDivElement>(`.${SCROLL_EL_CLASS}`);
    if (!el) return;
    const handleScroll = () => {
      const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
      const bottom = distance <= BOTTOM_SLACK_PX;
      setAtBottom(bottom);
      if (bottom) setAwaitingJump(false);
    };
    el.addEventListener("scroll", handleScroll, { passive: true });
    return () => el.removeEventListener("scroll", handleScroll);
  }, [hasLines]);

  // A new final lands, or a fresh partial opens: follow it down when the
  // person is already at the bottom; otherwise leave the view exactly where
  // they put it and just flag that there's more waiting below.
  useEffect(() => {
    if (lines.length === prevLineCount.current) return;
    prevLineCount.current = lines.length;
    if (atBottom) {
      listRef.current?.scrollToBottom();
    } else {
      setAwaitingJump(true);
    }
  }, [lines.length, atBottom]);

  // Wait out `PREPARING_BANNER_DELAY_MS` of unbroken "preparing" before saying
  // anything, and drop the wait the moment the engine is up (or gives up) so
  // the banner never outlives the thing it describes.
  useEffect(() => {
    if (capture.speech !== "preparing") {
      setPreparingIsNews(false);
      return;
    }
    const timer = setTimeout(() => setPreparingIsNews(true), PREPARING_BANNER_DELAY_MS);
    return () => clearTimeout(timer);
  }, [capture.speech]);

  const handleJumpToNow = () => {
    listRef.current?.scrollToBottom();
    setAtBottom(true);
    setAwaitingJump(false);
  };

  if (!active) {
    return (
      <EmptyState
        className="mx-auto max-w-md py-24"
        title={live.nothingRecordingTitle}
        description={live.nothingRecordingDescription}
        action={
          <Button variant="primary" onClick={() => navigate("/")}>
            {live.goHomeButton}
          </Button>
        }
      />
    );
  }

  const handleFlag = async () => {
    try {
      await markerCmd.run("actionItem");
      addToast({ level: "info", message: live.flagged });
    } catch (err) {
      addToast({ level: "problem", message: toUiError(err).message });
    }
  };

  const handleCopyTranscript = async () => {
    if (lines.length === 0) return;
    const text = buildLiveTranscriptText(lines, speakerLabelFor);
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      addToast({ level: "info", message: notices.copiedToClipboard });
      setTimeout(() => setCopied(false), 2000);
    } catch {
      addToast({ level: "problem", message: notices.somethingWentWrong });
    }
  };

  const handleTogglePause = () => {
    if (paused) resumeCmd.run().catch(() => {});
    else pauseCmd.run().catch(() => {});
  };

  return (
    <div className="mx-auto flex h-full w-full max-w-3xl flex-col gap-4 px-8 py-8">
      <header className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <RecordingDot paused={paused} label={labels.captureState[capture.state]} />
          <h1 className="text-lg font-semibold tracking-tight text-ink">{live.title}</h1>
        </div>
        <span className="tabular-nums text-sm text-ink-faint">
          {formatElapsed(capture.elapsedMs)}
        </span>
      </header>

      {/* Speech and audio are separate promises, and both can be broken at
          once: this one is about the words, the one below it about what Echo
          can hear. Neither hides the other. */}
      {capture.speech === "unavailable" && (
        <div className="rounded-xl border border-hairline bg-surface-sunken px-4 py-2.5 text-sm text-ink-soft">
          {labels.speechState.unavailable}
        </div>
      )}

      {capture.speech === "preparing" && preparingIsNews && (
        <div className="rounded-xl border border-hairline bg-surface-sunken px-4 py-2.5 text-sm text-ink-soft">
          {labels.speechState.preparing}
        </div>
      )}

      {capture.state === "degraded" && capture.degradedReason && (
        <div className="rounded-xl border border-hairline bg-surface-sunken px-4 py-2.5 text-sm text-ink-soft">
          {labels.degradedReason[capture.degradedReason]}
        </div>
      )}

      <div
        ref={scrollWrapperRef}
        className="relative min-h-0 flex-1 overflow-hidden rounded-xl"
      >
        <VirtualList
          ref={listRef}
          items={lines}
          itemHeight={ROW_HEIGHT}
          getKey={(line) => line.id}
          className={cx(
            SCROLL_EL_CLASS,
            "absolute inset-0 scroll-smooth rounded-xl border border-hairline bg-surface",
            showListeningRow && "pb-12",
          )}
          emptyState={
            <p className="px-5 py-8 text-center text-sm text-ink-ghost">{live.waitingForSpeech}</p>
          }
          renderItem={(line) => (
            <TranscriptLine
              tStartMs={line.tStartMs}
              speakerLabel={speakerLabelFor(line.speakerId, line.channel)}
              text={line.text}
              pending={!line.isFinal}
            />
          )}
        />

        {showListeningRow && (
          <div
            aria-hidden
            className="pointer-events-none absolute inset-x-0 bottom-0 flex items-center gap-2 bg-gradient-to-t from-surface to-transparent px-5 pb-3 pt-6"
          >
            <RecordingDot size="sm" />
            <span className="text-xs text-ink-faint">{live.listeningNow}</span>
          </div>
        )}

        {awaitingJump && !atBottom && (
          <button
            type="button"
            onClick={handleJumpToNow}
            className={cx(
              "absolute left-1/2 -translate-x-1/2 rounded-full border border-hairline bg-surface px-4 py-1.5 text-xs font-medium text-ink-soft shadow-lift transition-colors hover:bg-surface-sunken focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent",
              showListeningRow ? "bottom-11" : "bottom-3",
            )}
          >
            {live.jumpToNow}
          </button>
        )}
      </div>

      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Button variant="secondary" loading={markerCmd.loading} onClick={handleFlag}>
            {live.flagButton}
          </Button>
          <Button
            variant="secondary"
            leftIcon={copied ? <CheckIcon /> : undefined}
            disabled={lines.length === 0}
            onClick={handleCopyTranscript}
          >
            {copied ? common.copied : live.copyTranscriptButton}
          </Button>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="secondary"
            loading={pauseCmd.loading || resumeCmd.loading}
            onClick={handleTogglePause}
          >
            {paused ? live.resumeButton : live.pauseButton}
          </Button>
          <Button
            variant="primary"
            disabled={capture.state === "stopping"}
            onClick={() => setConfirmStop(true)}
          >
            {live.stopButton}
          </Button>
        </div>
      </div>

      <Modal
        open={confirmStop}
        onClose={() => setConfirmStop(false)}
        title={live.stopConfirmTitle}
        footer={
          <>
            <Button variant="secondary" onClick={() => setConfirmStop(false)}>
              {common.cancel}
            </Button>
            <Button
              variant="primary"
              loading={stopCmd.loading}
              onClick={async () => {
                try {
                  const finishedId = await stopCmd.run();
                  setConfirmStop(false);
                  navigate(finishedId ? `/meeting/${finishedId}` : "/");
                } catch {
                  // stopCmd.error already reflects it below; the dialog just
                  // stays open so the person can see it and try again.
                }
              }}
            >
              {live.stopButton}
            </Button>
          </>
        }
      >
        {live.stopConfirmDescription}
        {stopCmd.error && <p className="mt-2 text-xs text-live">{stopCmd.error.message}</p>}
      </Modal>
    </div>
  );
}
