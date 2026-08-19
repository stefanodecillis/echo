import { useEffect, useState } from "react";

import { Button } from "../../../components/Button";
import { Card } from "../../../components/Card";
import { CheckIcon, GeminiIcon, OllamaIcon } from "../../../components/icons";
import {
  listSummaryProviders,
  saveSummaryProvider,
  setProviderKey,
  testSummaryProvider,
  updateSettings,
} from "../../../lib/ipc";
import { common, labels, onboarding as copy, settings as settingsCopy } from "../../../lib/copy";
import { isLoopbackAddress } from "../../../lib/net";
import type { Provider, ProviderConfig, ProviderInfo } from "../../../lib/types";
import { cx } from "../../../components/lib/cx";

export interface SummariesStepProps {
  onFinish: (chosen: Provider | null) => void;
  onBack: () => void;
}

const DEFAULT_OLLAMA_ADDRESS = "http://127.0.0.1:11434";

/** Step 4 of 4: which assistant writes the recaps. Every path here — Ollama,
 * Gemini, or Skip — leads to "Start using Echo"; none of them is required,
 * and none of them changes where recordings live (always this Mac). */
export function SummariesStep({ onFinish, onBack }: SummariesStepProps) {
  const [providers, setProviders] = useState<ProviderInfo[] | null>(null);
  const [chosen, setChosen] = useState<Provider | null>(null);

  useEffect(() => {
    listSummaryProviders()
      .then(setProviders)
      .catch(() => setProviders([]));
  }, []);

  const ollamaInfo = providers?.find((p) => p.provider === "onThisComputer");
  const geminiInfo = providers?.find((p) => p.provider === "gemini");

  return (
    <div className="flex w-full max-w-md flex-col gap-4">
      <div className="text-center">
        <h2 className="text-xl font-semibold tracking-tight text-ink">
          {copy.stepSummariesTitle}
        </h2>
        <p className="mx-auto mt-2 max-w-sm text-xs leading-relaxed text-ink-faint">
          {copy.summariesIntro}
        </p>
      </div>

      <OllamaCard
        info={ollamaInfo}
        chosen={chosen === "onThisComputer"}
        onChosen={() => setChosen("onThisComputer")}
      />

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

function CardTitle({
  icon,
  title,
  chosen,
}: {
  icon: React.ReactNode;
  title: string;
  chosen: boolean;
}) {
  return (
    <div className="flex items-center gap-2">
      <span aria-hidden className="flex h-5 w-5 items-center justify-center text-ink">
        {icon}
      </span>
      <p className="text-sm font-semibold text-ink">{title}</p>
      {chosen && (
        <span className="flex h-4 w-4 items-center justify-center rounded-full bg-ink text-white">
          <CheckIcon className="h-2.5 w-2.5" />
        </span>
      )}
    </div>
  );
}

function OllamaCard({
  info,
  chosen,
  onChosen,
}: {
  info: ProviderInfo | undefined;
  chosen: boolean;
  onChosen: () => void;
}) {
  const [address, setAddress] = useState(info?.config.baseUrl ?? DEFAULT_OLLAMA_ADDRESS);
  const [acknowledged, setAcknowledged] = useState(false);
  const [available, setAvailable] = useState<boolean | null>(info?.available ?? null);
  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState(info?.config.model ?? "");
  const [checking, setChecking] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    if (info) {
      setAvailable(info.available);
      if (info.config.baseUrl) setAddress(info.config.baseUrl);
      if (info.config.model) setModel(info.config.model);
    }
  }, [info]);

  const remote = !isLoopbackAddress(address);

  function config(): ProviderConfig {
    return {
      provider: "onThisComputer",
      baseUrl: address.trim() || DEFAULT_OLLAMA_ADDRESS,
      model: model || undefined,
      hasKey: false,
      enabled: true,
      leavesMachineAcknowledged: remote ? acknowledged : undefined,
    };
  }

  async function check() {
    setChecking(true);
    setMessage(null);
    try {
      await saveSummaryProvider(config());
      const result = await testSummaryProvider("onThisComputer");
      setAvailable(result.ok);
      setModels(result.models);
      if (result.models.length > 0 && !model) setModel(result.models[0]);
      if (!result.ok) setMessage(result.message);
    } catch (err) {
      setAvailable(false);
      setMessage((err as { message?: string }).message ?? null);
    } finally {
      setChecking(false);
    }
  }

  async function useOllama() {
    try {
      await saveSummaryProvider(config());
      await updateSettings({ summaryProvider: "onThisComputer" });
      onChosen();
    } catch (err) {
      setMessage((err as { message?: string }).message ?? null);
    }
  }

  return (
    <Card className={cx("flex flex-col gap-3", chosen && "border-ink")}>
      <CardTitle icon={<OllamaIcon className="h-5 w-5" />} title={labels.provider.onThisComputer} chosen={chosen} />
      <p className="text-xs text-ink-faint">{settingsCopy.summaryProviderOnDeviceDescription}</p>
      <p className="text-xs text-ink-faint">
        {checking
          ? common.loading
          : available
            ? copy.summariesOnDeviceFound
            : copy.summariesOnDeviceNotFound}
      </p>

      <label className="flex flex-col gap-1.5">
        <span className="text-xs font-medium text-ink">
          {settingsCopy.recapsOnDeviceAddressLabel}
        </span>
        <input
          type="text"
          value={address}
          onChange={(e) => setAddress(e.target.value)}
          placeholder={settingsCopy.recapsOnDeviceAddressPlaceholder}
          className="w-full min-w-0 rounded-xl border border-hairline bg-surface px-3 py-2 text-sm text-ink placeholder:text-ink-ghost focus:outline-none focus:ring-2 focus:ring-accent/30"
        />
      </label>

      {remote && (
        <label className="flex items-start gap-2 text-xs text-ink-soft">
          <input
            type="checkbox"
            checked={acknowledged}
            onChange={(e) => setAcknowledged(e.target.checked)}
            className="mt-0.5"
          />
          {settingsCopy.recapsOnDeviceLeavesMachineWarning}
        </label>
      )}

      {models.length > 0 && (
        <label className="flex flex-col gap-1.5">
          <span className="text-xs font-medium text-ink">
            {settingsCopy.recapsOnDeviceModelLabel}
          </span>
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="w-full rounded-xl border border-hairline bg-surface px-3 py-2 text-sm text-ink focus:outline-none focus:ring-2 focus:ring-accent/30"
          >
            {models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>
      )}

      {message && <p className="text-xs text-ink-faint">{message}</p>}

      <div className="flex gap-2">
        <button
          type="button"
          onClick={useOllama}
          disabled={checking || (remote && !acknowledged)}
          className="echo-pill-quiet"
        >
          {copy.summariesUseOnDevice}
        </button>
        <button type="button" onClick={check} disabled={checking || (remote && !acknowledged)} className="echo-pill-quiet">
          {settingsCopy.recapsCheckAgainButton}
        </button>
      </div>
    </Card>
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
      await updateSettings({ summaryProvider: "gemini" });
      setSaved(true);
      setKey("");
      onChosen();
    } finally {
      setSaving(false);
    }
  }

  return (
    <Card className={cx("flex flex-col gap-3", chosen && "border-ink")}>
      <CardTitle icon={<GeminiIcon className="h-5 w-5" />} title={copy.summariesGeminiTitle} chosen={chosen} />
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
