import { useEffect, useState } from "react";

import { Card } from "../../../components/Card";
import { Chip } from "../../../components/Chip";
import { ProgressBar } from "../../../components/ProgressBar";
import { formatBytes } from "../../../components/lib/format";
import { useDownloadProgress } from "../../../hooks/useDownloadProgress";
import {
  cancelSpeechDownload,
  downloadSpeechAssets,
  listAccuracyLevels,
  selectAccuracyLevel,
} from "../../../lib/ipc";
import { settings as copy } from "../../../lib/copy";
import type { AccuracyLevel, Id } from "../../../lib/types";
import type { SectionProps } from "../types";

/** Speech: the "how carefully Echo listens" presets. Each one is a plain
 * sentence plus a size, never a model name — that lives in Advanced.
 *
 * Selecting a level goes through `select_accuracy_level` directly rather than
 * the generic `patch`, since the result also carries `accuracyLevelId` back
 * onto the shared `Settings` the other sections read — `index.tsx`'s own
 * fetch on next visit picks that up, so no local prop is needed here. */
export function Speech(_props: SectionProps) {
  const [levels, setLevels] = useState<AccuracyLevel[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [downloadingAssetId, setDownloadingAssetId] = useState<Id | null>(null);
  const [downloadingLevelId, setDownloadingLevelId] = useState<string | null>(null);
  const progress = useDownloadProgress(downloadingAssetId ?? undefined);

  useEffect(() => {
    refresh();
  }, []);

  function refresh() {
    listAccuracyLevels()
      .then(setLevels)
      .catch((err) => setError(err.message));
  }

  useEffect(() => {
    if (progress?.done) {
      setDownloadingAssetId(null);
      setDownloadingLevelId(null);
      refresh();
    }
  }, [progress?.done]);

  async function startDownload(level: AccuracyLevel) {
    setError(null);
    setDownloadingLevelId(level.id);
    try {
      const assetId = await downloadSpeechAssets(level.id);
      setDownloadingAssetId(assetId);
    } catch (err) {
      setDownloadingLevelId(null);
      setError((err as { message: string }).message);
    }
  }

  async function cancel() {
    if (!downloadingAssetId) return;
    await cancelSpeechDownload(downloadingAssetId);
    setDownloadingAssetId(null);
    setDownloadingLevelId(null);
  }

  async function use(level: AccuracyLevel) {
    const updated = await selectAccuracyLevel(level.id);
    setLevels(
      (prev) =>
        prev?.map((l) => ({ ...l, selected: l.id === updated.accuracyLevelId })) ?? prev,
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <p className="text-sm text-ink-faint">{copy.accuracyLevelDescription}</p>
      {error && <p className="text-sm text-live">{error}</p>}
      {!levels && !error && <p className="text-sm text-ink-faint">Loading…</p>}

      <div className="flex flex-col gap-3">
        {levels?.map((level) => {
          const isDownloadingThis = downloadingLevelId === level.id;
          return (
            <Card key={level.id} className="flex flex-col gap-3">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-semibold text-ink">{level.name}</p>
                    {level.selected && <Chip variant="solid">{copy.speechCurrentBadge}</Chip>}
                    {level.recommended && !level.selected && (
                      <Chip>{copy.speechRecommendedBadge}</Chip>
                    )}
                  </div>
                  <p className="mt-1 text-sm text-ink-soft">{level.description}</p>
                  <p className="mt-1 text-xs text-ink-faint">
                    {formatBytes(level.downloadBytes)}
                    {level.installed ? ` · ${copy.speechInstalledNote}` : ""}
                  </p>
                </div>
                <div className="flex shrink-0 flex-col items-end gap-2">
                  {!level.installed && !isDownloadingThis && (
                    <button
                      type="button"
                      onClick={() => startDownload(level)}
                      className="echo-pill-quiet"
                    >
                      {copy.speechDownloadButton}
                    </button>
                  )}
                  {isDownloadingThis && (
                    <button type="button" onClick={cancel} className="echo-pill-quiet">
                      {copy.speechCancelButton}
                    </button>
                  )}
                  {level.installed && !level.selected && (
                    <button
                      type="button"
                      onClick={() => use(level)}
                      className="echo-pill-quiet"
                    >
                      {copy.speechUseButton}
                    </button>
                  )}
                </div>
              </div>
              {isDownloadingThis && (
                <div className="flex flex-col gap-1">
                  <ProgressBar
                    value={
                      progress && progress.totalBytes > 0
                        ? progress.receivedBytes / progress.totalBytes
                        : undefined
                    }
                    label={copy.speechDownloadButton}
                  />
                  {progress?.error && (
                    <p className="text-xs text-live">{copy.speechDownloadError}</p>
                  )}
                </div>
              )}
            </Card>
          );
        })}
      </div>
    </div>
  );
}
