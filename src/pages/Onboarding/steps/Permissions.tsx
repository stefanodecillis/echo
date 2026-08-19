import { useEffect, useState } from "react";
import type { ReactNode } from "react";

import { Button } from "../../../components/Button";
import { Card } from "../../../components/Card";
import { BellIcon, CheckIcon, MicIcon, ScreenIcon } from "../../../components/icons";
import {
  getPermissionStatus,
  openPrivacySettings,
  quitApp,
  requestPermission,
} from "../../../lib/ipc";
import { common, labels, onboarding as copy } from "../../../lib/copy";
import type { PermissionState, PermissionStatus, PermissionTarget } from "../../../lib/types";
import { cx } from "../../../components/lib/cx";

export interface PermissionsStepProps {
  onNext: () => void;
  onBack: () => void;
}

const DEFAULT_STATUS: PermissionStatus = {
  microphone: "unknown",
  systemAudio: "unknown",
  notifications: "unknown",
};

/** Step 2 of 4: one card per permission, each in whatever state the OS
 * actually reports — never assumed granted, never a dead end on denied. */
export function PermissionsStep({ onNext, onBack }: PermissionsStepProps) {
  const [status, setStatus] = useState<PermissionStatus>(DEFAULT_STATUS);

  const refresh = () => {
    getPermissionStatus()
      .then(setStatus)
      .catch(() => {
        // Cards fall back to "not checked yet" and can still be actioned.
      });
  };

  useEffect(refresh, []);

  async function requestAndRefresh(target: PermissionTarget) {
    try {
      const state = await requestPermission(target);
      setStatus((prev) => ({ ...prev, [target]: state }));
    } catch {
      refresh();
    }
  }

  return (
    <div className="flex w-full max-w-md flex-col gap-5">
      <h2 className="text-center text-xl font-semibold tracking-tight text-ink">
        {copy.stepPermissionsTitle}
      </h2>

      <div className="flex flex-col gap-3">
        <PermissionCard
          icon={<MicIcon className="h-full w-full" />}
          title={copy.microphoneTitle}
          description={copy.microphoneDescription}
          state={status.microphone}
          onAllow={() => requestAndRefresh("microphone")}
          onOpenSettings={() => openPrivacySettings("microphone")}
        />
        <PermissionCard
          icon={<ScreenIcon className="h-full w-full" />}
          title={copy.screenRecordingTitle}
          description={copy.screenRecordingDescription}
          state={status.systemAudio}
          onAllow={() => requestAndRefresh("systemAudio")}
          onOpenSettings={() => openPrivacySettings("systemAudio")}
        />
        <PermissionCard
          icon={<BellIcon className="h-full w-full" />}
          title={copy.notificationsTitle}
          description={copy.notificationsDescription}
          state={status.notifications}
          onAllow={() => requestAndRefresh("notifications")}
          onOpenSettings={() => openPrivacySettings("notifications")}
        />
      </div>

      <div className="flex justify-between pt-2">
        <Button variant="ghost" onClick={onBack}>
          {common.back}
        </Button>
        <Button variant="primary" onClick={onNext}>
          {common.next}
        </Button>
      </div>
    </div>
  );
}

function PermissionCard({
  icon,
  title,
  description,
  state,
  onAllow,
  onOpenSettings,
}: {
  icon: ReactNode;
  title: string;
  description: string;
  state: PermissionState;
  onAllow: () => void;
  onOpenSettings: () => void;
}) {
  const granted = state === "granted" || state === "notApplicable";

  return (
    <Card className="flex items-start gap-3">
      <span
        aria-hidden
        className={cx(
          "flex h-9 w-9 shrink-0 items-center justify-center rounded-full",
          granted ? "bg-ink text-white" : "bg-surface-sunken text-ink-faint",
        )}
      >
        {granted ? <CheckIcon className="h-4 w-4" /> : <span className="h-4 w-4">{icon}</span>}
      </span>
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium text-ink">{title}</p>
        <p className="mt-0.5 text-xs text-ink-faint">{description}</p>
        <p className="mt-1 text-xs font-medium text-ink-soft">
          {labels.permissionState[state]}
        </p>

        {state === "denied" && (
          <div className="mt-2 flex flex-col gap-1.5">
            <p className="text-xs text-ink-faint">{copy.permissionDeniedNote}</p>
            <button
              type="button"
              onClick={onOpenSettings}
              className="echo-pill-quiet w-fit text-xs"
            >
              {common.openSettings}
            </button>
          </div>
        )}

        {state === "restartRequired" && (
          <div className="mt-2 flex flex-col gap-1.5">
            <p className="text-xs text-ink-faint">{copy.permissionRestartNote}</p>
            <button type="button" onClick={() => quitApp()} className="echo-pill-quiet w-fit text-xs">
              {copy.quitButton}
            </button>
          </div>
        )}

        {(state === "unknown" || state === "prompting") && (
          <button
            type="button"
            onClick={onAllow}
            disabled={state === "prompting"}
            className="echo-pill-quiet mt-2 w-fit text-xs"
          >
            {copy.permissionAllowButton}
          </button>
        )}
      </div>
    </Card>
  );
}
