import { Button } from "../../../components/Button";
import { onboarding as copy } from "../../../lib/copy";

export interface WelcomeStepProps {
  onNext: () => void;
}

/** Step 1 of 4: no permissions asked yet, no download started — just what's
 * about to happen and one button. */
export function WelcomeStep({ onNext }: WelcomeStepProps) {
  return (
    <div className="flex flex-col items-center gap-6 text-center">
      <span aria-hidden className="h-3 w-3 rounded-full border-2 border-ink" />
      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-semibold tracking-tight text-ink">
          {copy.welcomeTitle}
        </h1>
        <p className="max-w-sm text-sm text-ink-faint">{copy.welcomeSubtitle}</p>
      </div>
      <Button variant="primary" size="md" onClick={onNext} className="px-8">
        {copy.getStartedButton}
      </Button>
    </div>
  );
}
