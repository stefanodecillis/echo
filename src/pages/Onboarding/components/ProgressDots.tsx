import { cx } from "../../../components/lib/cx";

export interface ProgressDotsProps {
  steps: string[];
  activeIndex: number;
}

/** The four dots across the top of onboarding — a plain sense of "how much is
 * left", never a percentage or a step counter a person has to do math on. */
export function ProgressDots({ steps, activeIndex }: ProgressDotsProps) {
  return (
    <ol className="flex items-center justify-center gap-2" aria-label="Setup progress">
      {steps.map((label, i) => (
        <li key={label}>
          <span
            aria-current={i === activeIndex ? "step" : undefined}
            aria-label={label}
            className={cx(
              "block h-1.5 rounded-full transition-all",
              i === activeIndex ? "w-6 bg-ink" : i < activeIndex ? "w-1.5 bg-ink-ghost" : "w-1.5 bg-hairline",
            )}
          />
        </li>
      ))}
    </ol>
  );
}
