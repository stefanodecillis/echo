import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { useCommand } from "../../hooks/useCommand";
import { completeOnboarding } from "../../lib/ipc";
import { onboarding as copy } from "../../lib/copy";
import type { Provider } from "../../lib/types";
import { ProgressDots } from "./components/ProgressDots";
import { DownloadStep } from "./steps/Download";
import { PermissionsStep } from "./steps/Permissions";
import { SummariesStep } from "./steps/Summaries";
import { WelcomeStep } from "./steps/Welcome";

const STEP_IDS = ["welcome", "permissions", "download", "summaries"] as const;
type StepId = (typeof STEP_IDS)[number];

/** Where setup left off, surviving a relaunch — macOS forces one when the
 * screen-recording switch is flipped, and coming back to the Welcome page
 * after that reads as starting over. */
const RESUME_KEY = "echo.onboarding.step";

function savedStep(): StepId {
  try {
    const saved = window.localStorage.getItem(RESUME_KEY);
    return STEP_IDS.includes(saved as StepId) ? (saved as StepId) : "welcome";
  } catch {
    return "welcome";
  }
}

function rememberStep(step: StepId | null) {
  try {
    if (step === null) window.localStorage.removeItem(RESUME_KEY);
    else window.localStorage.setItem(RESUME_KEY, step);
  } catch {
    // Being unable to remember the step only costs a relaunch its bookmark.
  }
}

const STEP_LABELS: Record<StepId, string> = {
  welcome: copy.stepLabelWelcome,
  permissions: copy.stepLabelPermissions,
  download: copy.stepLabelDownload,
  summaries: copy.stepLabelSummaries,
};

/**
 * Welcome -> Permissions -> Download -> Recaps setup -> Home.
 *
 * Four full-screen steps sharing one piece of state (which step is active);
 * each step owns its own data fetching and local UI. Finishing calls
 * `complete_onboarding` — the one thing that makes the app route here only on
 * first run — then leaves for Home the same way a notification click would.
 */
export default function OnboardingPage() {
  const [step, setStep] = useState<StepId>(savedStep);
  const navigate = useNavigate();
  const { run: finish } = useCommand(completeOnboarding);

  const index = STEP_IDS.indexOf(step);

  function goTo(next: StepId) {
    setStep(next);
    rememberStep(next);
  }

  async function handleFinish(_chosen: Provider | null) {
    try {
      await finish();
    } finally {
      // Onboarding is done either way — a failed write to settings shouldn't
      // trap someone who clicked through four screens on a fresh install.
      rememberStep(null);
      navigate("/");
    }
  }

  return (
    <div className="flex h-full w-full flex-col items-center justify-center gap-10 bg-surface px-6 py-10">
      {step !== "welcome" && (
        <ProgressDots steps={STEP_IDS.map((id) => STEP_LABELS[id])} activeIndex={index} />
      )}

      {step === "welcome" && <WelcomeStep onNext={() => goTo("permissions")} />}
      {step === "permissions" && (
        <PermissionsStep onNext={() => goTo("download")} onBack={() => goTo("welcome")} />
      )}
      {step === "download" && (
        <DownloadStep onNext={() => goTo("summaries")} onBack={() => goTo("permissions")} />
      )}
      {step === "summaries" && (
        <SummariesStep onFinish={handleFinish} onBack={() => goTo("download")} />
      )}
    </div>
  );
}
