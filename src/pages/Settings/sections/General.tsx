import { useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";

import { Card } from "../../../components/Card";
import { formatBytes } from "../../../components/lib/format";
import { useCommand } from "../../../hooks/useCommand";
import { validateStorageLocation } from "../../../lib/ipc";
import { settings as copy } from "../../../lib/copy";
import type { SummaryLanguage } from "../../../lib/types";
import { SettingList, SettingRow } from "../components/SettingRow";
import { Select } from "../components/Select";
import { Toggle } from "../components/Toggle";
import type { SectionProps } from "../types";

type LanguageChoice = "meeting" | "english" | "custom";

function choiceOf(language: SummaryLanguage): LanguageChoice {
  if (language.kind === "sameAsMeeting") return "meeting";
  if (language.kind === "english") return "english";
  return "custom";
}

/** General: launch at login, meeting detection, storage location, recap
 * language. Every control here is safe to leave alone — Settings never
 * *needs* a visit (mantra 4). */
export function General({ settings, patch }: SectionProps) {
  const [freeBytes, setFreeBytes] = useState<number | null>(null);
  const [pathError, setPathError] = useState<string | null>(null);
  const { run: checkPath, loading: checkingPath } = useCommand(validateStorageLocation);
  const [customLanguage, setCustomLanguage] = useState(
    settings.summaryLanguage.kind === "fixed" ? settings.summaryLanguage.value : "",
  );

  async function choosePath() {
    const picked = await open({ directory: true, multiple: false });
    const path = typeof picked === "string" ? picked : null;
    if (!path) return;
    setPathError(null);
    try {
      const free = await checkPath(path);
      setFreeBytes(free);
      await patch({ storageDir: path });
    } catch (err) {
      setPathError((err as { message: string }).message);
    }
  }

  async function setLanguage(choice: LanguageChoice) {
    if (choice === "meeting") await patch({ summaryLanguage: { kind: "sameAsMeeting" } });
    else if (choice === "english") await patch({ summaryLanguage: { kind: "english" } });
    else await patch({ summaryLanguage: { kind: "fixed", value: customLanguage || "" } });
  }

  async function commitCustomLanguage(value: string) {
    setCustomLanguage(value);
    if (value.trim()) await patch({ summaryLanguage: { kind: "fixed", value } });
  }

  const languageChoice = choiceOf(settings.summaryLanguage);

  return (
    <div className="flex flex-col gap-4">
      <Card>
        <SettingList>
          <SettingRow
            title={copy.launchAtLogin}
            control={
              <Toggle
                label={copy.launchAtLogin}
                checked={settings.launchAtLogin}
                onChange={(v) => patch({ launchAtLogin: v })}
              />
            }
          />
          <SettingRow
            title={copy.detectionEnabled}
            description={copy.detectionDescription}
            control={
              <Toggle
                label={copy.detectionEnabled}
                checked={settings.detectionEnabled}
                onChange={(v) => patch({ detectionEnabled: v })}
              />
            }
          />
        </SettingList>
      </Card>

      <Card>
        <SettingRow
          stacked
          title={copy.storageLocation}
          description={copy.storageLocationHint}
          control={
            <div className="flex flex-col gap-2">
              <div className="flex items-center gap-2">
                <p className="min-w-0 flex-1 truncate rounded-xl border border-hairline bg-surface-sunken px-3 py-2 text-xs text-ink-faint">
                  {settings.storageDir}
                </p>
                <button
                  type="button"
                  onClick={choosePath}
                  disabled={checkingPath}
                  className="echo-pill-quiet shrink-0"
                >
                  {copy.storageLocationButton}
                </button>
              </div>
              {freeBytes !== null && !pathError && (
                <p className="text-xs text-ink-faint">
                  {formatBytes(freeBytes)} {copy.storageFreeSpaceSuffix}
                </p>
              )}
              {pathError && <p className="text-xs text-live">{pathError}</p>}
            </div>
          }
        />
      </Card>

      <Card>
        <SettingRow
          stacked
          title={copy.summaryLanguageLabel}
          control={
            <div className="flex flex-col gap-3">
              <Select
                value={languageChoice}
                onChange={(e) => setLanguage(e.target.value as LanguageChoice)}
              >
                <option value="meeting">{copy.summaryLanguageMeeting}</option>
                <option value="english">{copy.summaryLanguageEnglish}</option>
                <option value="custom">{copy.summaryLanguageFixed}</option>
              </Select>
              {languageChoice === "custom" && (
                <input
                  value={customLanguage}
                  onChange={(e) => commitCustomLanguage(e.target.value)}
                  placeholder={copy.summaryLanguageCustomPlaceholder}
                  aria-label={copy.summaryLanguageCustomLabel}
                  className="w-full rounded-xl border border-hairline bg-surface px-3 py-2 text-sm text-ink placeholder:text-ink-ghost focus:outline-none focus:ring-2 focus:ring-accent/30"
                />
              )}
            </div>
          }
        />
      </Card>
    </div>
  );
}
