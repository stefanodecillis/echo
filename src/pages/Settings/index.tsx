import { useEffect, useState } from "react";

import { Tabs } from "../../components/Tabs";
import { useCommand } from "../../hooks/useCommand";
import { useEvent } from "../../hooks/useEvent";
import { EVENTS, getSettings, updateSettings } from "../../lib/ipc";
import { common, settings as copy } from "../../lib/copy";
import type { Settings, SettingsPatch } from "../../lib/types";
import { Advanced } from "./sections/Advanced";
import { Data } from "./sections/Data";
import { General } from "./sections/General";
import { Recaps } from "./sections/Recaps";
import { Speech } from "./sections/Speech";
import { Templates } from "./sections/Templates";

type SectionId = "general" | "speech" | "recaps" | "templates" | "data" | "advanced";

const sections: { id: SectionId; label: string }[] = [
  { id: "general", label: copy.sectionGeneral },
  { id: "speech", label: copy.sectionSpeech },
  { id: "recaps", label: copy.sectionSummaries },
  { id: "templates", label: copy.sectionTemplates },
  { id: "data", label: copy.sectionData },
  { id: "advanced", label: copy.sectionAdvanced },
];

/**
 * Settings: General, Speech, Recaps, Recap styles, Data, Advanced.
 *
 * One shared `Settings` object is fetched here and handed down; every section
 * that changes a setting goes through `patch`, which optimistically merges the
 * result of `update_settings` back into that one copy so the others stay
 * consistent (e.g. Recaps changing the active provider is reflected the next
 * time General's language row renders).
 */
export default function SettingsPage() {
  const [active, setActive] = useState<SectionId>("general");
  const [settings, setSettings] = useState<Settings | null>(null);
  const { run: load, error: loadError } = useCommand(getSettings);
  const { run: save } = useCommand(updateSettings);

  useEffect(() => {
    load()
      .then(setSettings)
      .catch(() => {
        // loadError below carries the message; nothing else to do.
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Something else changed a setting — the menu bar's "Pause meeting reminders",
  // or onboarding finishing in another window. Take the core's word for it.
  useEvent(EVENTS.settingsChanged, (next) => {
    setSettings(next);
  });

  async function patch(next: SettingsPatch): Promise<Settings> {
    const result = await save(next);
    setSettings(result);
    return result;
  }

  return (
    <section className="mx-auto flex w-full max-w-3xl flex-col gap-6 px-8 py-10">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight text-ink">{copy.title}</h1>
        <p className="mt-1 text-sm text-ink-faint">{copy.subtitle}</p>
      </header>

      <Tabs
        items={sections.map((s) => ({ id: s.id, label: s.label }))}
        value={active}
        onChange={(id) => setActive(id as SectionId)}
      />

      {!settings && !loadError && (
        <p className="text-sm text-ink-faint">{common.loading}</p>
      )}
      {loadError && !settings && (
        <p className="text-sm text-live">{loadError.message}</p>
      )}

      {settings && (
        <div className="pb-10">
          {active === "general" && <General settings={settings} patch={patch} />}
          {active === "speech" && <Speech settings={settings} patch={patch} />}
          {active === "recaps" && <Recaps settings={settings} patch={patch} />}
          {active === "templates" && <Templates settings={settings} patch={patch} />}
          {active === "data" && <Data />}
          {active === "advanced" && <Advanced />}
        </div>
      )}
    </section>
  );
}
