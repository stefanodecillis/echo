import { Button } from "../../../components/Button";
import { CheckIcon, EchoMark, MeetingsIcon, MicIcon } from "../../../components/icons";
import { onboarding as copy } from "../../../lib/copy";

export interface WelcomeStepProps {
  onNext: () => void;
}

const STEPS = [
  { icon: MicIcon, title: copy.welcomeStepListenTitle, caption: copy.welcomeStepListenCaption },
  {
    icon: MeetingsIcon,
    title: copy.welcomeStepTranscriptTitle,
    caption: copy.welcomeStepTranscriptCaption,
  },
  { icon: CheckIcon, title: copy.welcomeStepRecapTitle, caption: copy.welcomeStepRecapCaption },
];

/** Step 1 of 4: no permissions asked yet, no download started — the mark,
 * what's about to happen, and one button. */
export function WelcomeStep({ onNext }: WelcomeStepProps) {
  return (
    <div className="flex flex-col items-center gap-8 text-center">
      <EchoMark aria-hidden className="h-16 w-16 text-ink" />
      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-semibold tracking-tight text-ink">{copy.welcomeTitle}</h1>
        <p className="max-w-sm text-sm text-ink-faint">{copy.welcomeSubtitle}</p>
      </div>
      <div className="grid w-full max-w-xl grid-cols-1 gap-3 sm:grid-cols-3">
        {STEPS.map(({ icon: Icon, title, caption }) => (
          <div
            key={title}
            className="flex flex-col items-center gap-2 rounded-xl border border-hairline px-4 py-5"
          >
            <Icon aria-hidden className="h-5 w-5 text-ink" />
            <span className="text-sm font-medium text-ink">{title}</span>
            <span className="text-xs leading-relaxed text-ink-faint">{caption}</span>
          </div>
        ))}
      </div>
      <Button variant="primary" size="md" onClick={onNext} className="px-8">
        {copy.getStartedButton}
      </Button>
    </div>
  );
}
