import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { Button, Card, Modal, RecordingDot } from "../../components";
import { formatElapsed } from "../../components/lib/format";
import { useCommand } from "../../hooks";
import { common, home, labels, live } from "../../lib/copy";
import { openPrivacySettings, stopRecording } from "../../lib/ipc";
import type {
  CaptureStatus,
  DetectionStatus,
  PermissionTarget,
  StartRecordingOptions,
} from "../../lib/types";

export interface StatusHeroProps {
  capture: CaptureStatus;
  detection: DetectionStatus;
  onStart: (options?: StartRecordingOptions) => void;
  starting: boolean;
  startErrorMessage?: string;
  startErrorAction?: "openMicrophoneSettings" | "openScreenRecordingSettings";
  /** The most recent line Echo has heard, for the live variant. */
  lastCaption?: string;
}

const ACTIVE_STATES = new Set(["starting", "recording", "paused", "degraded", "stopping"]);

/**
 * Home's single status card: whichever of idle / detected / recording best
 * describes what's happening right now. Never more than one shows at once —
 * a person glancing at Home should see exactly what Echo is doing, not a
 * stack of banners arguing about it.
 */
export function StatusHero({
  capture,
  detection,
  onStart,
  starting,
  startErrorMessage,
  startErrorAction,
  lastCaption,
}: StatusHeroProps) {
  if (ACTIVE_STATES.has(capture.state)) {
    return <RecordingHero capture={capture} lastCaption={lastCaption} />;
  }

  if (detection.state === "detected") {
    return (
      <Card padding="lg" className="flex flex-col items-start gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-ink">
            {home.heroDetectedTitle}
          </h1>
          <p className="mt-1 text-sm text-ink-soft">{home.heroDetectedSubtitle}</p>
        </div>
        <Button
          variant="primary"
          loading={starting}
          onClick={() => onStart({ detectedApp: detection.signals[0]?.app })}
        >
          {home.startButton}
        </Button>
        <StartError message={startErrorMessage} action={startErrorAction} />
      </Card>
    );
  }

  return (
    <Card padding="lg" className="flex flex-col items-start gap-3">
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-ink">{home.heroIdleTitle}</h1>
        <p className="mt-1 text-sm text-ink-soft">{home.heroIdleSubtitle}</p>
      </div>
      <Button variant="secondary" loading={starting} onClick={() => onStart()}>
        {home.startButton}
      </Button>
      <p className="text-xs text-ink-ghost">{home.heroIdleTip}</p>
      <StartError message={startErrorMessage} action={startErrorAction} />
    </Card>
  );
}

function RecordingHero({
  capture,
  lastCaption,
}: {
  capture: CaptureStatus;
  lastCaption?: string;
}) {
  const navigate = useNavigate();
  const [confirmStop, setConfirmStop] = useState(false);
  const stopCmd = useCommand(stopRecording);
  const paused = capture.state === "paused";

  return (
    <Card padding="lg" className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <RecordingDot paused={paused} label={labels.captureState[capture.state]} />
          <span className="text-sm font-medium text-ink">{labels.captureState[capture.state]}</span>
        </div>
        <span className="tabular-nums text-sm text-ink-faint">
          {formatElapsed(capture.elapsedMs)}
        </span>
      </div>

      <p className="line-clamp-2 text-sm text-ink-soft">{lastCaption || live.waitingForSpeech}</p>

      <div className="flex items-center gap-2">
        <Button variant="secondary" onClick={() => navigate("/live")}>
          {home.openLiveButton}
        </Button>
        <Button
          variant="primary"
          disabled={capture.state === "stopping"}
          onClick={() => setConfirmStop(true)}
        >
          {home.stopButton}
        </Button>
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
              {home.stopButton}
            </Button>
          </>
        }
      >
        {live.stopConfirmDescription}
        {stopCmd.error && <p className="mt-2 text-xs text-live">{stopCmd.error.message}</p>}
      </Modal>
    </Card>
  );
}

function StartError({
  message,
  action,
}: {
  message?: string;
  action?: "openMicrophoneSettings" | "openScreenRecordingSettings";
}) {
  if (!message) return null;
  const target: PermissionTarget = action === "openMicrophoneSettings" ? "microphone" : "systemAudio";
  return (
    <div className="flex flex-col items-start gap-2 rounded-xl bg-live/5 px-3 py-2">
      <p className="text-xs text-live">{message}</p>
      {action && (
        <Button size="sm" variant="secondary" onClick={() => openPrivacySettings(target)}>
          {common.openSettings}
        </Button>
      )}
    </div>
  );
}
