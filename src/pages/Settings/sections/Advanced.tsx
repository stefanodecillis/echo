import { useEffect, useState } from "react";
import { save } from "@tauri-apps/plugin-dialog";

import { Card } from "../../../components/Card";
import { ChevronDownIcon } from "../../../components/icons";
import { formatBytes } from "../../../components/lib/format";
import { useEchoStore } from "../../../lib/store";
import { exportDiagnostics, getSystemCapabilities, listSpeechAssets } from "../../../lib/ipc";
import { advanced as copy, settings as settingsCopy } from "../../../lib/copy";
import type { ModelInfo, SystemCapabilities } from "../../../lib/types";
import { cx } from "../../../components/lib/cx";

/** Advanced: the one place technical words are allowed, because the person
 * went looking for them. Collapsed by default — nothing here changes what
 * Echo does day to day, so it shouldn't compete with the rest of Settings. */
export function Advanced() {
  const [expanded, setExpanded] = useState(false);
  const [caps, setCaps] = useState<SystemCapabilities | null>(null);
  const [assets, setAssets] = useState<ModelInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const addToast = useEchoStore((s) => s.addToast);

  useEffect(() => {
    if (!expanded || caps) return;
    Promise.all([getSystemCapabilities(), listSpeechAssets()])
      .then(([c, a]) => {
        setCaps(c);
        setAssets(a);
      })
      .catch((err) => setError(err.message));
  }, [expanded, caps]);

  async function exportLogs() {
    const destination = await save({ defaultPath: "echo-diagnostics.zip" });
    if (!destination) return;
    try {
      await exportDiagnostics(destination);
      addToast({ level: "info", message: copy.diagnosticsExportSuccess });
    } catch (err) {
      addToast({ level: "problem", message: (err as { message: string }).message });
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <p className="text-sm text-ink-faint">{settingsCopy.advancedIntro}</p>
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="flex w-fit items-center gap-1.5 text-sm font-medium text-ink-soft hover:text-ink"
      >
        <ChevronDownIcon
          className={cx("h-4 w-4 transition-transform", expanded && "rotate-180")}
        />
        {expanded ? copy.hideButton : copy.showButton}
      </button>

      {expanded && (
        <>
          {error && <p className="text-sm text-live">{error}</p>}
          {!caps && !error && <p className="text-sm text-ink-faint">Loading…</p>}
          {caps && (
            <Card padding="none">
              <div className="divide-y divide-hairline text-sm">
                <Row label={copy.osLabel} value={`${caps.os} (${caps.arch})`} />
                <Row label={copy.engineLabel} value={caps.speechBackend} />
                <Row
                  label={copy.graphicsBackendLabel}
                  value={caps.speechBackendActive || caps.speechBackend}
                />
                {caps.gpuFallbackReason && (
                  <Row
                    label={copy.graphicsFallbackReason}
                    value={caps.gpuFallbackReason}
                  />
                )}
                <Row label={copy.cpuThreadsLabel} value={String(caps.cpuThreads)} />
                <Row label={copy.memoryLabel} value={formatBytes(caps.totalMemoryBytes)} />
                <Row
                  label={copy.systemAudioSupportedLabel}
                  value={caps.systemAudioSupported ? copy.yes : copy.no}
                />
                <Row
                  label={copy.traySupportedLabel}
                  value={caps.traySupported ? copy.yes : copy.no}
                />
                <Row label={copy.appVersionLabel} value={caps.appVersion} />
              </div>
            </Card>
          )}

          {assets && assets.length > 0 && (
            <Card padding="none">
              <p className="px-5 pt-4 text-xs font-semibold uppercase tracking-wide text-ink-ghost">
                {copy.installedAssetsTitle}
              </p>
              <div className="divide-y divide-hairline text-sm">
                {assets.map((asset) => (
                  <Row
                    key={asset.id}
                    label={asset.name}
                    value={asset.installed ? formatBytes(asset.bytes) : "—"}
                  />
                ))}
              </div>
            </Card>
          )}

          <div>
            <button type="button" onClick={exportLogs} className="echo-pill-quiet">
              {copy.diagnosticsExportButton}
            </button>
            <p className="mt-2 text-xs text-ink-faint">{copy.diagnosticsDescription}</p>
          </div>
        </>
      )}
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-4 px-5 py-2.5">
      <span className="text-ink-faint">{label}</span>
      <span className="truncate text-right text-ink-soft">{value}</span>
    </div>
  );
}
