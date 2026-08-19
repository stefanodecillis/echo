import { useEffect, useState } from "react";

import { Button } from "../../../components/Button";
import { Card } from "../../../components/Card";
import { CheckIcon } from "../../../components/icons";
import {
  listSummaryProviders,
  saveSummaryProvider,
  setProviderKey,
  testSummaryProvider,
  updateSettings,
} from "../../../lib/ipc";
import { common, labels, onboarding as copy, settings as settingsCopy } from "../../../lib/copy";
import type { Provider, ProviderInfo } from "../../../lib/types";
import { cx } from "../../../components/lib/cx";

export interface SummariesStepProps {
  onFinish: (chosen: Provider | null) => void;
  onBack: () => void;
}

/** Step 4 of 4: where recaps come from. Every path here — on-device, Gemini,
 * or Skip — leads to "Start using Echo"; none of them is required. */
export function SummariesStep({ onFinish, onBack }: SummariesStepProps) {
  const [providers, setProviders] = useState<ProviderInfo[] | null>(null);
  const [chosen, setChosen] = useState<Provider | null>(null);
  const [checking, setChecking] = useState(false);
  const [onDeviceAvailable, setOnDeviceAvailable] = useState<boolean | null>(null);

  useEffect(() => {
    listSummaryProviders()
      .then((all) => {
        setProviders(all);
        setOnDeviceAvailable(all.find((p) => p.provider === "onThisComputer")?.available ?? false);
      })
      .catch(() => setOnDeviceAvailable(false));
  }, []);

  async function checkAgain() {
    setChecking(true);
    try {
      const result = await testSummaryProvider("onThisComputer");
      setOnDeviceAvailable(result.ok);
    } catch {
      setOnDeviceAvailable(false);
    } finally {
      setChecking(false);
    }
  }

  async function chooseOnDevice() {
    setChosen("onThisComputer");
    await updateSettings({ summaryProvider: "onThisComputer" }).catch(() => {});
  }

  const geminiInfo = providers?.find((p) => p.provider === "gemini");

  return (
    <div className="flex w-full max-w-md flex-col gap-5">
      <h2 className="text-center text-xl font-semibold tracking-tight text-ink">
        {copy.stepSummariesTitle}
      </h2>

      <Card className={cx("flex flex-col gap-3", chosen === "onThisComputer" && "border-ink")}>
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <p className="text-sm font-semibold text-ink">{labels.provider.onThisComputer}</p>
              {chosen === "onThisComputer" && (
                <span className="flex h-4 w-4 items-center justify-center rounded-full bg-ink text-white">
                  <CheckIcon className="h-2.5 w-2.5" />
                </span>
              )}
            </div>
            <p className="mt-1 text-xs text-ink-faint">
              {checking
                ? common.loading
                : onDeviceAvailable
                  ? copy.summariesOnDeviceFound
                  : copy.summariesOnDeviceNotFound}
            </p>
          </div>
        </div>
        <div className="flex gap-2">
          <button type="button" onClick={chooseOnDevice} className="echo-pill-quiet">
            {copy.summariesUseOnDevice}
          </button>
          <button type="button" onClick={checkAgain} disabled={checking} className="echo-pill-quiet">
            {settingsCopy.recapsCheckAgainButton}
          </button>
        </div>
      </Card>

      <GeminiCard
        chosen={chosen === "gemini"}
        onChosen={() => setChosen("gemini")}
        hasKey={geminiInfo?.config.hasKey ?? false}
      />

      <div className="flex justify-between pt-1">
        <Button variant="ghost" onClick={onBack}>
          {common.back}
        </Button>
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => onFinish(null)}
            className="text-sm font-medium text-ink-faint hover:text-ink-soft"
          >
            {common.skip}
          </button>
          <Button variant="primary" onClick={() => onFinish(chosen)}>
            {copy.finishButton}
          </Button>
        </div>
      </div>
    </div>
  );
}

function GeminiCard({
  chosen,
  onChosen,
  hasKey,
}: {
  chosen: boolean;
  onChosen: () => void;
  hasKey: boolean;
}) {
  const [key, setKey] = useState("");
  const [saved, setSaved] = useState(hasKey);
  const [saving, setSaving] = useState(false);

  async function submit() {
    if (!key.trim()) return;
    setSaving(true);
    try {
      await setProviderKey("gemini", key);
      await saveSummaryProvider({ provider: "gemini", hasKey: true, enabled: true });
      setSaved(true);
      setKey("");
      onChosen();
    } finally {
      setSaving(false);
    }
  }

  return (
    <Card className={cx("flex flex-col gap-3", chosen && "border-ink")}>
      <div className="flex items-center gap-2">
        <p className="text-sm font-semibold text-ink">{copy.summariesGeminiTitle}</p>
        {chosen && (
          <span className="flex h-4 w-4 items-center justify-center rounded-full bg-ink text-white">
            <CheckIcon className="h-2.5 w-2.5" />
          </span>
        )}
      </div>
      <p className="text-xs text-ink-faint">{copy.summariesGeminiDescription}</p>
      <p className="text-xs text-ink-faint">{settingsCopy.recapsGeminiPrivacyParagraph}</p>
      <div className="flex items-center gap-2">
        <input
          type="password"
          value={key}
          onChange={(e) => setKey(e.target.value)}
          placeholder={saved ? settingsCopy.providerKeySaved : settingsCopy.providerKeyPlaceholder}
          className="w-full min-w-0 rounded-xl border border-hairline bg-surface px-3 py-2 text-sm text-ink placeholder:text-ink-ghost focus:outline-none focus:ring-2 focus:ring-accent/30"
        />
        <button
          type="button"
          onClick={submit}
          disabled={saving || !key.trim()}
          className="echo-pill-quiet shrink-0"
        >
          {common.save}
        </button>
      </div>
    </Card>
  );
}
