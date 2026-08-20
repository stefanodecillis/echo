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

const ACTIVE_STATES = new Set(["starting", "recording", "paused", "degraded", "stopping"]);

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
  useEffect(() => {
    listRef.current?.scrollToBottom();
  }, [lines.length]);

  const paused = capture.state === "paused";
  const active = capture.meetingId != null && ACTIVE_STATES.has(capture.state);

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

      {capture.state === "degraded" && capture.degradedReason && (
        <div className="rounded-xl border border-hairline bg-surface-sunken px-4 py-2.5 text-sm text-ink-soft">
          {labels.degradedReason[capture.degradedReason]}
        </div>
      )}

      <VirtualList
        ref={listRef}
        items={lines}
        itemHeight={ROW_HEIGHT}
        getKey={(line) => line.id}
        className="min-h-0 flex-1 rounded-xl border border-hairline bg-surface"
        emptyState={
          <p className="px-5 py-8 text-center text-sm text-ink-ghost">{live.waitingForSpeech}</p>
        }
        renderItem={(line) => (
          <TranscriptLine
            speakerLabel={speakerLabelFor(line.speakerId, line.channel)}
            text={line.text}
            pending={!line.isFinal}
          />
        )}
      />

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
