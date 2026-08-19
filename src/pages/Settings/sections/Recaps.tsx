import { useEffect, useState } from "react";
import type { ReactNode } from "react";

import { Card } from "../../../components/Card";
import { Chip } from "../../../components/Chip";
import { cx } from "../../../components/lib/cx";
import { CheckIcon, GeminiIcon, OllamaIcon } from "../../../components/icons";
import { useCommand } from "../../../hooks/useCommand";
import {
  hasProviderKey,
  listSummaryProviders,
  saveSummaryProvider,
  setProviderKey,
  testSummaryProvider,
} from "../../../lib/ipc";
import { common, labels, settings as copy } from "../../../lib/copy";
import { isLoopbackAddress } from "../../../lib/net";
import type { Provider, ProviderConfig, ProviderInfo } from "../../../lib/types";
import { Select } from "../components/Select";
import { SettingRow } from "../components/SettingRow";
import { Toggle } from "../components/Toggle";
import type { SectionProps } from "../types";

/**
 * The options for a model picker: whatever the backend listed, plus the model
 * that's actually saved if that isn't among them.
 *
 * The saved one can be missing for good reasons — the core falls back to its own
 * default before anything has been chosen, a local pull was removed, a hosted
 * model was retired — and a `<select>` whose value matches no option shows
 * nothing at all, so the person would be looking at an empty box while a recap
 * is being written by a model with a name. It stays visible either way.
 */
function withCurrent(models: string[], current: string | undefined): string[] {
  const saved = current?.trim();
  if (!saved || models.includes(saved)) return models;
  return [saved, ...models];
}

/** Recaps: choose where recaps get written. A compact, radio-group-style list
 * of connector cards — one row per connector (icon, name, one-line
 * description, "Being used" badge). Only the selected connector expands its
 * details (address/key, test, model picker) below its row, so the section
 * stays short no matter how many connectors Echo ships. */
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

  return (
    <div className="flex flex-col gap-4">
      {/* On by default, so this is the off switch rather than a step someone has
          to find first (mantra 4). It sits above the chooser because it decides
          whether any of the rest of this section ever gets used. */}
      <Card>
        <SettingRow
          title={copy.recapsAutomaticTitle}
          description={copy.recapsAutomaticDescription}
          control={
            <Toggle
              label={copy.recapsAutomaticTitle}
              checked={settings.autoSummarize}
              onChange={(v) => patch({ autoSummarize: v })}
            />
          }
        />
      </Card>

      {error && <p className="text-sm text-live">{error}</p>}
      {!providers && !error && <p className="text-sm text-ink-faint">{common.loading}</p>}

      {providers && (
        <div role="radiogroup" aria-label={copy.sectionSummaries} className="flex flex-col gap-2">
          {providers.map((info) => (
            <ConnectorOption
              key={info.provider}
              info={info}
              active={settings.summaryProvider === info.provider}
              onSelect={() => patch({ summaryProvider: info.provider })}
              onUpdate={(next) => updateProvider(info.provider, next)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

interface ProviderCardProps {
  info: ProviderInfo;
  onUpdate: (info: ProviderInfo) => void;
}

const CONNECTOR_META: Record<Provider, { icon: ReactNode; description: string }> = {
  onThisComputer: {
    icon: <OllamaIcon className="h-full w-full" />,
    description: copy.summaryProviderOnDeviceDescription,
  },
  gemini: {
    icon: <GeminiIcon className="h-full w-full" />,
    description: copy.summaryProviderGeminiDescription,
  },
};

/** One connector, rendered as a radio-style row that expands into its own
 * settings when it's the active one. */
function ConnectorOption({
  info,
  active,
  onSelect,
  onUpdate,
}: ProviderCardProps & { active: boolean; onSelect: () => void }) {
  const meta = CONNECTOR_META[info.provider];

  return (
    <Card className={cx(active && "border-ink")}>
      <button
        type="button"
        role="radio"
        aria-checked={active}
        // Never disabled, even when it's the chosen one: a disabled radio drops
        // out of the tab order, and then the group has no focusable checked
        // option for anyone using a keyboard. Choosing the current one again is
        // simply a no-op.
        onClick={onSelect}
        className="flex w-full cursor-pointer items-center gap-3 rounded-lg text-left"
      >
        <span
          aria-hidden
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-surface-sunken text-ink"
        >
          <span className="h-5 w-5">{meta.icon}</span>
        </span>
        <span className="min-w-0 flex-1">
          <span className="flex items-center gap-2">
            <span className="text-sm font-semibold text-ink">{labels.provider[info.provider]}</span>
            {active && (
              <Chip variant="solid" icon={<CheckIcon className="h-full w-full" />}>
                {copy.recapsSelectedBadge}
              </Chip>
            )}
          </span>
          <span className="mt-0.5 block truncate text-xs text-ink-soft">{meta.description}</span>
        </span>
        {!active && <span className="echo-pill-quiet shrink-0">{copy.recapsSelectButton}</span>}
      </button>

      {active && (
        <div className="mt-4 flex flex-col gap-3 border-t border-hairline pt-4">
          {info.provider === "onThisComputer" ? (
            <OnDeviceDetails info={info} onUpdate={onUpdate} />
          ) : (
            <GeminiDetails info={info} onUpdate={onUpdate} />
          )}
        </div>
      )}
    </Card>
  );
}

const DEFAULT_OLLAMA_ADDRESS = "http://127.0.0.1:11434";

function OnDeviceDetails({ info, onUpdate }: ProviderCardProps) {
  const [models, setModels] = useState<string[]>([]);
  const [checking, setChecking] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [address, setAddress] = useState(info.config.baseUrl ?? DEFAULT_OLLAMA_ADDRESS);
  const [acknowledged, setAcknowledged] = useState(false);

  const remote = !isLoopbackAddress(address);

  async function check() {
    setChecking(true);
    setMessage(null);
    try {
      const trimmed = address.trim() || DEFAULT_OLLAMA_ADDRESS;
      if (trimmed !== (info.config.baseUrl ?? DEFAULT_OLLAMA_ADDRESS)) {
        const config: ProviderConfig = {
          ...info.config,
          baseUrl: trimmed,
          leavesMachineAcknowledged: remote ? acknowledged : undefined,
        };
        await saveSummaryProvider(config);
        onUpdate({ ...info, config });
      }
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

  const modelOptions = withCurrent(models, info.config.model);

  return (
    <>
      <label className="flex flex-col gap-1.5">
        <span className="text-xs font-medium text-ink">{copy.recapsOnDeviceAddressLabel}</span>
        <input
          type="text"
          value={address}
          onChange={(e) => setAddress(e.target.value)}
          placeholder={copy.recapsOnDeviceAddressPlaceholder}
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
          {copy.recapsOnDeviceLeavesMachineWarning}
        </label>
      )}
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
      {modelOptions.length > 0 && (
        <label className="flex flex-col gap-1.5">
          <span className="text-xs font-medium text-ink">{copy.recapsOnDeviceModelLabel}</span>
          <Select value={info.config.model ?? ""} onChange={(e) => chooseModel(e.target.value)}>
            <option value="" disabled>
              {copy.recapsOnDeviceModelPlaceholder}
            </option>
            {modelOptions.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </Select>
        </label>
      )}
    </>
  );
}

function GeminiDetails({ info, onUpdate }: ProviderCardProps) {
  const [key, setKey] = useState("");
  const [hasKey, setHasKey] = useState(info.config.hasKey);
  const [models, setModels] = useState<string[]>([]);
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

  // With a key already saved, ask Google what it will write recaps with as soon
  // as this section opens — the same thing the on-this-computer side does. Left
  // to the Test button, someone with a working key would sit in front of a
  // section that never shows them the choice they came here to make.
  useEffect(() => {
    if (!hasKey) return;
    let cancelled = false;
    testSummaryProvider("gemini")
      .then((result) => {
        if (!cancelled) setModels(result.models);
      })
      .catch(() => {
        // Silent: nobody pressed anything, and the Test button below says
        // exactly what went wrong when they do.
      });
    return () => {
      cancelled = true;
    };
  }, [hasKey]);

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
      // Only the models Google reports as able to write text (test/list
      // already filters out embedding-only models on the Rust side).
      setModels(result.models);
    } catch (err) {
      setTestMessage((err as { message: string }).message);
    }
  }

  async function chooseModel(model: string) {
    const config: ProviderConfig = { ...info.config, model };
    await saveSummaryProvider(config);
    onUpdate({ ...info, config });
  }

  // `list_providers` always reports the model that would actually be used, so
  // this is never blank — but it may be the core's own default, which Google has
  // not listed yet, or one that has since been retired.
  const modelOptions = withCurrent(models, info.config.model);

  return (
    <>
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

      {modelOptions.length > 0 && (
        <label className="flex flex-col gap-1.5">
          <span className="text-xs font-medium text-ink">{copy.recapsGeminiModelLabel}</span>
          <Select value={info.config.model ?? ""} onChange={(e) => chooseModel(e.target.value)}>
            <option value="" disabled>
              {copy.recapsGeminiModelPlaceholder}
            </option>
            {modelOptions.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </Select>
        </label>
      )}
    </>
  );
}
