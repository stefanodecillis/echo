import { useEffect, useMemo, useState } from "react";

import { Button } from "../../../components/Button";
import { ProgressBar } from "../../../components/ProgressBar";
import { DownloadIcon } from "../../../components/icons";
import { formatBytes } from "../../../components/lib/format";
import { useDownloadProgress } from "../../../hooks/useDownloadProgress";
import {
  cancelSpeechDownload,
  downloadSpeechAssets,
  listAccuracyLevels,
} from "../../../lib/ipc";
import { anchors, common, onboarding as copy } from "../../../lib/copy";
import type { AccuracyLevel, Id } from "../../../lib/types";

export interface DownloadStepProps {
  onNext: () => void;
  onBack: () => void;
}

/** Step 3 of 4: the one-time download, started automatically so a person who
 * just clicks "Continue" through onboarding ends up ready to record. */
export function DownloadStep({ onNext, onBack }: DownloadStepProps) {
  const [levels, setLevels] = useState<AccuracyLevel[] | null>(null);
  const [levelId, setLevelId] = useState<string | null>(null);
  const [assetId, setAssetId] = useState<Id | null>(null);
  const [error, setError] = useState<string | null>(null);
  const progress = useDownloadProgress(assetId ?? undefined);

  useEffect(() => {
    listAccuracyLevels()
      .then((all) => {
        setLevels(all);
        const preferred = all.find((l) => l.recommended) ?? all[0];
        if (preferred) beginDownload(preferred);
      })
      .catch((err) => setError(err.message));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function beginDownload(level: AccuracyLevel) {
    setError(null);
    setLevelId(level.id);
    if (level.installed) return;
    try {
      const id = await downloadSpeechAssets(level.id);
      setAssetId(id);
    } catch (err) {
      setError((err as { message: string }).message);
    }
  }

  const currentLevel = useMemo(
    () => levels?.find((l) => l.id === levelId) ?? null,
    [levels, levelId],
  );
  const smaller = useMemo(() => {
    if (!levels || !currentLevel) return null;
    return (
      levels
        .filter((l) => l.id !== currentLevel.id && l.downloadBytes < currentLevel.downloadBytes)
        .sort((a, b) => b.downloadBytes - a.downloadBytes)[0] ?? null
    );
  }, [levels, currentLevel]);

  async function useSmaller() {
    if (!smaller) return;
    if (assetId) await cancelSpeechDownload(assetId).catch(() => {});
    setAssetId(null);
    beginDownload(smaller);
  }

  async function retry() {
    if (currentLevel) beginDownload(currentLevel);
  }

  const done = currentLevel?.installed || progress?.done;
  const fraction =
    progress && progress.totalBytes > 0 ? progress.receivedBytes / progress.totalBytes : undefined;

  return (
    <div className="flex w-full max-w-sm flex-col items-center gap-6 text-center">
      <span aria-hidden className="flex h-12 w-12 items-center justify-center rounded-full bg-surface-sunken text-ink-faint">
        <DownloadIcon className="h-5 w-5" />
      </span>

      <div className="flex flex-col gap-1.5">
        <h2 className="text-xl font-semibold tracking-tight text-ink">
          {anchors.downloadingSpeech}
        </h2>
        <p className="text-sm text-ink-faint">
          {copy.stepDownloadSize} · {copy.stepDownloadDescription}
        </p>
      </div>

      <div className="flex w-full flex-col gap-2">
        <ProgressBar value={done ? 1 : fraction} label={anchors.downloadingSpeech} />
        {!done && fraction !== undefined && (
          <p className="text-xs tabular-nums text-ink-faint">
            {formatBytes(progress?.receivedBytes ?? 0)} / {formatBytes(progress?.totalBytes ?? 0)}
          </p>
        )}
      </div>

      {error && (
        <div className="flex flex-col items-center gap-2">
          <p className="text-xs text-live">{error}</p>
          <button type="button" onClick={retry} className="echo-pill-quiet text-xs">
            {copy.downloadRetryButton}
          </button>
        </div>
      )}

      {smaller && !done && (
        <button type="button" onClick={useSmaller} className="text-xs font-medium text-ink-soft underline underline-offset-2 hover:text-ink">
          {copy.chooseSmallerButton}
        </button>
      )}

      {!done && !error && (
        <p className="text-xs text-ink-ghost">{copy.downloadContinueAnywayNote}</p>
      )}

      <div className="mt-2 flex w-full justify-between">
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
