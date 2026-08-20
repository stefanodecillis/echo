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
 * Speech: a status card, nothing to choose.
 *
 * Echo downloads and uses one way of understanding speech, the best there is,
 * the same on every computer (docs/DESIGN.md mantra 1 and its 2026-08-20
 * amendments). It also keeps that current on its own: if a newer release wants
 * different weights, the core fetches them in the background and swaps them in
 * once they are ready, without ever leaving the person unable to record. So
 * this screen has no decision on it. It says three things — whether Echo can
 * understand speech, whether anything is still arriving, and how much room it
 * all takes.
 *
 * Model names stay in Settings > Advanced, never here.
 *
 * `listAccuracyLevels` returns exactly one level; it is consulted only for the
 * honest download size and the asset ids behind Remove.
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
  /**
   * Echo can already understand speech *and* there is still more to fetch.
   * Either the first download is finishing its last pieces, or the core is
   * replacing what is here with something better (an upgrade never blocks
   * recording — see the reconcile in the core). Worth a sentence either way,
   * because "Ready to use" next to a progress bar otherwise reads as a bug.
   */
  const improving = ready && (downloading || (readiness?.remainingBytes ?? 0) > 0);
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
              {!ready
                ? // The honest total for this computer, from the core, rather
                  // than a size typed into the copy that goes stale the next
                  // time what Echo downloads changes.
                  level
                  ? `${copy.speechNotReadyDescription} ${copy.speechDownloadSizeNote} ${formatBytes(
                      level.downloadBytes,
                    )}.`
                  : copy.speechNotReadyDescription
                : improving
                  ? copy.speechImprovingDescription
                  : copy.speechReadyDescription}
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
            {/* Never next to a download in flight: "Remove" would delete files
                that are still arriving, and the one to offer then is Cancel. */}
            {ready && !downloading && (
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
