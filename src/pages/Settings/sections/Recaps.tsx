import { useEffect, useState } from "react";
import type { ReactNode } from "react";

import { Card } from "../../../components/Card";
import { Chip } from "../../../components/Chip";
import { CheckIcon } from "../../../components/icons";
import { useCommand } from "../../../hooks/useCommand";
import {
  hasProviderKey,
  listSummaryProviders,
  saveSummaryProvider,
  setProviderKey,
  testSummaryProvider,
} from "../../../lib/ipc";
import { common, labels, settings as copy } from "../../../lib/copy";
import type { Provider, ProviderConfig, ProviderInfo } from "../../../lib/types";
import { Select } from "../components/Select";
import type { SectionProps } from "../types";

/** Recaps: choose where recaps get written. Two cards, exactly one active —
 * "On this computer" (nothing leaves the machine, needs a one-time setup) or
 * Google Gemini (an explicit, up-front privacy choice, never a footnote). */
export function Recaps({ settings, patch }: SectionProps) {
  const [providers, setProviders] = useState<ProviderInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listSummaryProviders()
      .then(setProviders)
      .catch((err) => setError(err.message));
  }, []);

  function updateProvider(provider: Provider, info: ProviderInfo) {
    setProviders((prev) => prev?.map((p) => (p.provider === provider ? info : p)) ?? prev);
  }

  const onDevice = providers?.find((p) => p.provider === "onThisComputer");
  const gemini = providers?.find((p) => p.provider === "gemini");

  return (
    <div className="flex flex-col gap-4">
      {error && <p className="text-sm text-live">{error}</p>}
      {!providers && !error && <p className="text-sm text-ink-faint">Loading…</p>}

      {onDevice && (
        <OnDeviceCard
          info={onDevice}
          active={settings.summaryProvider === "onThisComputer"}
          onSelect={() => patch({ summaryProvider: "onThisComputer" })}
          onUpdate={(info) => updateProvider("onThisComputer", info)}
        />
      )}
      {gemini && (
        <GeminiCard
          info={gemini}
          active={settings.summaryProvider === "gemini"}
          onSelect={() => patch({ summaryProvider: "gemini" })}
          onUpdate={(info) => updateProvider("gemini", info)}
        />
      )}
    </div>
  );
}

interface ProviderCardProps {
  info: ProviderInfo;
  active: boolean;
  onSelect: () => void;
  onUpdate: (info: ProviderInfo) => void;
}

function ProviderCardShell({
  title,
  description,
  active,
  onSelect,
  children,
}: {
  title: string;
  description: string;
  active: boolean;
  onSelect: () => void;
  children?: ReactNode;
}) {
  return (
    <Card className={active ? "border-ink" : undefined}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <p className="text-sm font-semibold text-ink">{title}</p>
            {active && (
              <Chip variant="solid" icon={<CheckIcon className="h-full w-full" />}>
                {copy.recapsSelectedBadge}
              </Chip>
            )}
          </div>
          <p className="mt-1 text-sm text-ink-soft">{description}</p>
        </div>
        {!active && (
          <button type="button" onClick={onSelect} className="echo-pill-quiet shrink-0">
            {copy.recapsSelectButton}
          </button>
        )}
      </div>
      <div className="mt-4 flex flex-col gap-3">{children}</div>
    </Card>
  );
}

function OnDeviceCard({ info, active, onSelect, onUpdate }: ProviderCardProps) {
  const [models, setModels] = useState<string[]>([]);
  const [checking, setChecking] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  async function check() {
    setChecking(true);
    setMessage(null);
    try {
      const result = await testSummaryProvider("onThisComputer");
      setModels(result.models);
      setMessage(result.message);
    } catch (err) {
      setMessage((err as { message: string }).message);
    } finally {
      setChecking(false);
    }
  }

  useEffect(() => {
    check();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function chooseModel(model: string) {
    const config: ProviderConfig = { ...info.config, model };
    await saveSummaryProvider(config);
    onUpdate({ ...info, config });
  }

  return (
    <ProviderCardShell
      title={labels.provider.onThisComputer}
      description={copy.summaryProviderOnDeviceDescription}
      active={active}
      onSelect={onSelect}
    >
      <div className="flex items-center justify-between gap-3">
        <p className="text-xs text-ink-faint">
          {checking
            ? common.loading
            : info.available
              ? copy.recapsOnDeviceStatusDetected
              : copy.recapsOnDeviceStatusNotRunning}
        </p>
        <button type="button" onClick={check} disabled={checking} className="echo-pill-quiet">
          {copy.recapsCheckAgainButton}
        </button>
      </div>
      {message && <p className="text-xs text-ink-faint">{message}</p>}
      {info.available && models.length > 0 && (
        <label className="flex flex-col gap-1.5">
          <span className="text-xs font-medium text-ink">{copy.recapsOnDeviceModelLabel}</span>
          <Select value={info.config.model ?? ""} onChange={(e) => chooseModel(e.target.value)}>
            <option value="" disabled>
              {copy.recapsOnDeviceModelPlaceholder}
            </option>
            {models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </Select>
        </label>
      )}
    </ProviderCardShell>
  );
}

function GeminiCard({ info, active, onSelect, onUpdate }: ProviderCardProps) {
  const [key, setKey] = useState("");
  const [hasKey, setHasKey] = useState(info.config.hasKey);
  const { run: saveKey, loading: savingKey } = useCommand(setProviderKey);
  const { run: test, loading: testing } = useCommand(testSummaryProvider);
  const [testMessage, setTestMessage] = useState<string | null>(null);

  useEffect(() => {
    hasProviderKey("gemini")
      .then(setHasKey)
      .catch(() => {
        // Leave the last known state; the field below still lets them re-enter it.
      });
  }, []);

  async function submitKey() {
    if (!key.trim()) return;
    await saveKey("gemini", key);
    setKey("");
    setHasKey(true);
    const config: ProviderConfig = { ...info.config, hasKey: true };
    await saveSummaryProvider(config);
    onUpdate({ ...info, config });
  }

  async function runTest() {
    setTestMessage(null);
    try {
      const result = await test("gemini");
      setTestMessage(result.message);
    } catch (err) {
      setTestMessage((err as { message: string }).message);
    }
  }

  return (
    <ProviderCardShell
      title={labels.provider.gemini}
      description={copy.summaryProviderGeminiDescription}
      active={active}
      onSelect={onSelect}
    >
      <p className="text-xs text-ink-faint">{copy.recapsGeminiPrivacyParagraph}</p>
      <a
        href="https://ai.google.dev/gemini-api/terms"
        target="_blank"
        rel="noreferrer"
        className="text-xs font-medium text-accent underline underline-offset-2"
      >
        {copy.recapsGeminiPrivacyLink}
      </a>

      <div className="flex flex-col gap-1.5">
        <span className="text-xs font-medium text-ink">{copy.providerKeyLabel}</span>
        <div className="flex items-center gap-2">
          <input
            type="password"
            value={key}
            onChange={(e) => setKey(e.target.value)}
            placeholder={hasKey ? copy.providerKeySaved : copy.providerKeyPlaceholder}
            className="w-full min-w-0 rounded-xl border border-hairline bg-surface px-3 py-2 text-sm text-ink placeholder:text-ink-ghost focus:outline-none focus:ring-2 focus:ring-accent/30"
          />
          <button
            type="button"
            onClick={submitKey}
            disabled={savingKey || !key.trim()}
            className="echo-pill-quiet shrink-0"
          >
            {common.save}
          </button>
        </div>
        {hasKey && !key && <p className="text-xs text-ink-faint">{copy.providerKeySaved}</p>}
      </div>

      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={runTest}
          disabled={testing || !hasKey}
          className="echo-pill-quiet"
        >
          {copy.providerTestButton}
        </button>
        {testMessage && <p className="text-xs text-ink-faint">{testMessage}</p>}
      </div>
    </ProviderCardShell>
  );
}
