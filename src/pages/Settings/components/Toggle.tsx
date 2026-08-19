import { cx } from "../../../components/lib/cx";

export interface ToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
  label: string;
}

/**
 * A small pill switch for on/off settings — launch at login, meeting
 * detection, auto-summarize. `label` is read by screen readers only; the
 * visible label sits next to it as a `SettingRow` title.
 */
export function Toggle({ checked, onChange, disabled, label }: ToggleProps) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={cx(
        "relative inline-flex h-6 w-10 shrink-0 items-center rounded-full transition-colors",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40",
        checked ? "bg-ink" : "bg-surface-sunken border border-hairline",
        disabled && "cursor-not-allowed opacity-50",
      )}
    >
      <span
        aria-hidden
        className={cx(
          "inline-block h-4 w-4 transform rounded-full bg-white shadow-card transition-transform",
          checked ? "translate-x-[1.15rem]" : "translate-x-1",
          !checked && "bg-white",
        )}
      />
    </button>
  );
}
