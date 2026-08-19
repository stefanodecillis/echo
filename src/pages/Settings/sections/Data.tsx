import { useEffect, useState } from "react";

import { Card } from "../../../components/Card";
import { formatBytes } from "../../../components/lib/format";
import { deleteAllData, getStorageReport } from "../../../lib/ipc";
import { settings as copy } from "../../../lib/copy";
import type { StorageReport } from "../../../lib/types";
import { TypedConfirmModal } from "../components/TypedConfirmModal";

/** Data: what Echo is holding onto, and the one irreversible action in
 * Settings — gated behind a typed confirmation rather than a single click. */
export function Data() {
  const [report, setReport] = useState<StorageReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState(false);

  useEffect(() => {
    refresh();
  }, []);

  function refresh() {
    getStorageReport()
      .then(setReport)
      .catch((err) => setError(err.message));
  }

  const rows = report
    ? [
        { label: copy.dataStorageBreakdownAudio, bytes: report.audioBytes },
        { label: copy.dataStorageBreakdownDatabase, bytes: report.databaseBytes },
        { label: copy.dataStorageBreakdownSpeech, bytes: report.speechAssetBytes },
        { label: copy.dataStorageBreakdownLogs, bytes: report.logBytes },
      ]
    : [];

  return (
    <div className="flex flex-col gap-4">
      {error && <p className="text-sm text-live">{error}</p>}
      {!report && !error && <p className="text-sm text-ink-faint">Loading…</p>}

      {report && (
        <Card>
          <p className="text-sm font-semibold text-ink">{copy.dataStorageReportTitle}</p>
          <p className="mt-1 text-2xl font-semibold tracking-tight text-ink">
            {formatBytes(report.totalBytes)}
          </p>
          <div className="mt-4 divide-y divide-hairline">
            {rows.map((row) => (
              <div key={row.label} className="flex items-center justify-between py-2 text-sm">
                <span className="text-ink-faint">{row.label}</span>
                <span className="tabular-nums text-ink-soft">{formatBytes(row.bytes)}</span>
              </div>
            ))}
            <div className="flex items-center justify-between py-2 text-sm">
              <span className="text-ink-faint">{copy.dataMeetingCount}</span>
              <span className="tabular-nums text-ink-soft">{report.meetingCount}</span>
            </div>
            <div className="flex items-center justify-between py-2 text-sm">
              <span className="text-ink-faint">{copy.dataFreeSpace}</span>
              <span className="tabular-nums text-ink-soft">{formatBytes(report.freeBytes)}</span>
            </div>
          </div>
        </Card>
      )}

      <Card className="border-live/20">
        <p className="text-sm font-semibold text-ink">{copy.dataDeleteAllButton}</p>
        <p className="mt-1 text-sm text-ink-faint">
          {copy.dataDeleteAllConfirmDescription}
        </p>
        <button
          type="button"
          onClick={() => setConfirming(true)}
          className="echo-pill mt-4 border border-live/30 text-live hover:bg-live/5"
        >
          {copy.dataDeleteAllButton}
        </button>
      </Card>

      <TypedConfirmModal
        open={confirming}
        title={copy.dataDeleteAllConfirmTitle}
        description={`${copy.dataDeleteAllConfirmDescription} ${copy.dataDeleteAllTypePrompt}`}
        confirmWord={copy.dataDeleteAllTypeWord}
        confirmLabel={copy.dataDeleteAllButton}
        onClose={() => setConfirming(false)}
        onConfirm={async () => {
          await deleteAllData();
          refresh();
        }}
      />
    </div>
  );
}
