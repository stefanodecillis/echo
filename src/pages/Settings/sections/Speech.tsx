import { useEffect, useState } from "react";

import { Card } from "../../../components/Card";
import { Chip } from "../../../components/Chip";
import { ProgressBar } from "../../../components/ProgressBar";
import { formatBytes } from "../../../components/lib/format";
import { useDownloadProgress } from "../../../hooks/useDownloadProgress";
import { useSpeechReadiness } from "../../../hooks/useSpeechReadiness";
import {
  cancelSpeechDownload,
  downloadSpeechAssets,
  getStorageReport,
  listAccuracyLevels,
  removeSpeechAsset,
} from "../../../lib/ipc";
import { common, settings as copy } from "../../../lib/copy";
import type { AccuracyLevel, Id } from "../../../lib/types";
import type { SectionProps } from "../types";

/**
 * Speech: a status card, nothing to choose. Echo always downloads and uses
 * the best fit for this computer (docs/DESIGN.md mantra 1's 2026-08-20
 * amendment) — this screen just says whether that's done, the one-time
 * download's state, and how much room it takes. Model names stay in
 * Settings > Advanced, never here.
 *
 * `listAccuracyLevels` is only consulted internally for the one asset id to
 * download or remove; there is nothing here for the person to pick between.
 */
export function Speech(_props: SectionProps) {
  const readiness = useSpeechReadiness();
  const progress = useDownloadProgress();
  const [level, setLevel] = useState<AccuracyLevel | null>(null);
  const [storageBytes, setStorageBytes] = useState<number | null>(null);
  const [assetId, setAssetId] = useState<Id | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    refresh();
  }, []);

  useEffect(() => {
    if (progress?.done) {
      setAssetId(null);
      refresh();
    }
  }, [progress?.done]);

  function refresh() {
    listAccuracyLevels()
      .then((all) => setLevel(all.find((l) => l.recommended) ?? all[0] ?? null))
      .catch((err) => setError(err.message));
    getStorageReport()
      .then((report) => setStorageBytes(report.speechAssetBytes))
      .catch(() => {
        // The status card still works without a storage figure.
      });
  }

  async function startDownload() {
    if (!level) return;
    setError(null);
    try {
      const id = await downloadSpeechAssets(level.id);
      setAssetId(id);
    } catch (err) {
      setError((err as { message: string }).message);
    }
  }

  async function cancel() {
    const id = assetId ?? progress?.assetId;
    if (!id) return;
    await cancelSpeechDownload(id);
    setAssetId(null);
  }

  async function remove() {
    if (!level) return;
    for (const id of level.assetIds) {
      await removeSpeechAsset(id).catch(() => {});
    }
    refresh();
  }

  const ready = readiness?.ready ?? level?.installed ?? false;
  const downloading = readiness?.downloading ?? false;
  const resuming =
    !ready && !downloading && level !== null && (readiness?.remainingBytes ?? 0) > 0 &&
    (readiness?.remainingBytes ?? 0) < level.downloadBytes;
  const fraction =
    progress && progress.totalBytes > 0 ? progress.receivedBytes / progress.totalBytes : undefined;

  return (
    <div className="flex flex-col gap-4">
      <Card className="flex flex-col gap-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <p className="text-sm font-semibold text-ink">
                {ready ? copy.speechReadyTitle : copy.speechNotReadyTitle}
              </p>
              {ready && <Chip variant="solid">{copy.speechReadyBadge}</Chip>}
            </div>
            <p className="mt-1 text-sm text-ink-soft">
              {ready ? copy.speechReadyDescription : copy.speechNotReadyDescription}
            </p>
            {ready && storageBytes !== null && (
              <p className="mt-1 text-xs text-ink-faint">
                {copy.speechStorageLabel} · {formatBytes(storageBytes)}
              </p>
            )}
          </div>
          <div className="flex shrink-0 flex-col items-end gap-2">
            {!ready && !downloading && (
              <button type="button" onClick={startDownload} className="echo-pill-quiet">
                {resuming ? copy.speechResumeButton : copy.speechDownloadButton}
              </button>
            )}
            {downloading && (
              <button type="button" onClick={cancel} className="echo-pill-quiet">
                {copy.speechCancelButton}
              </button>
            )}
            {ready && (
              <button type="button" onClick={remove} className="echo-pill-quiet">
                {copy.speechRemoveButton}
              </button>
            )}
          </div>
        </div>

        {downloading && (
          <div className="flex flex-col gap-1">
            <ProgressBar value={fraction} label={copy.speechDownloadButton} />
            {progress?.error && <p className="text-xs text-live">{copy.speechDownloadError}</p>}
          </div>
        )}

        {error && (
          <div className="flex flex-col items-start gap-2">
            <p className="text-xs text-live">{error}</p>
            <button type="button" onClick={startDownload} className="echo-pill-quiet text-xs">
              {common.retry}
            </button>
          </div>
        )}
      </Card>
    </div>
  );
}
