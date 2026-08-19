import { useEffect, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";

import { SearchInput } from "../../components";
import { useCaptureState, useCommand, useEvent } from "../../hooks";
import { useDownloadProgress } from "../../hooks/useDownloadProgress";
import { useSpeechReadiness } from "../../hooks/useSpeechReadiness";
import { home } from "../../lib/copy";
import { EVENTS, startRecording, toUiError } from "../../lib/ipc";
import { useEchoStore } from "../../lib/store";
import type { CaptureState, Id, RecoveryAction, StartRecordingOptions } from "../../lib/types";
import { RecentMeetings } from "./RecentMeetings";
import { RecoveryBanner } from "./RecoveryBanner";
import { StatusHero } from "./StatusHero";
import { useDetectionStatus } from "./useDetectionStatus";
import { useInterruptedMeetings } from "./useInterruptedMeetings";

/** States that mean "on the way to /live" — the auto-navigate watcher below
 * fires the moment capture crosses into one of these from anything else. */
const LIVE_BOUND_STATES = new Set<CaptureState>(["starting", "recording"]);

/**
 * Home: what Echo is doing right now, and the meetings already captured.
 *
 * `?start=1` — from the sidebar's "New meeting" link, a notification click,
 * or the tray, via `App.tsx`'s navigate-event handling — asks the Start
 * button to fire itself once, as if it had been clicked, so the person's
 * intent survives the trip here.
 */
export default function Home() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const capture = useCaptureState();
  const detection = useDetectionStatus();
  const interrupted = useInterruptedMeetings();
  const addToast = useEchoStore((s) => s.addToast);
  const startCmd = useCommand(startRecording);
  const [lastCaption, setLastCaption] = useState<string>();
  const readiness = useSpeechReadiness();
  const download = useDownloadProgress();
  // Unknown readiness counts as ready here so the hero doesn't flicker into
  // "getting ready" on every visit; the backend still refuses a premature
  // start with a plain sentence either way.
  const preparing = readiness ? !readiness.ready : false;

  useEvent(EVENTS.transcriptPartial, (payload) => {
    if (payload.meetingId === capture.meetingId) setLastCaption(payload.text);
  });
  useEvent(EVENTS.transcriptFinal, (payload) => {
    if (payload.meetingId === capture.meetingId) setLastCaption(payload.segment.text);
  });

  // Store-driven: once capture actually crosses into starting/recording —
  // whether from the button below, the tray, or a notification click — hop
  // to the live view. Comparing against the *previous* render's state (not
  // just "is it recording") means landing on Home mid-meeting never bounces
  // the person straight back out to /live against their will.
  const prevStateRef = useRef<CaptureState | null>(null);
  useEffect(() => {
    const prev = prevStateRef.current;
    prevStateRef.current = capture.state;
    if (prev !== null && !LIVE_BOUND_STATES.has(prev) && LIVE_BOUND_STATES.has(capture.state)) {
      navigate("/live");
    }
  }, [capture.state, navigate]);

  useEffect(() => {
    if (searchParams.get("start") !== "1") return;
    if (preparing) {
      // Echo isn't ready to record yet; drop the request instead of firing a
      // start that the backend would refuse.
      setSearchParams(
        (params) => {
          params.delete("start");
          return params;
        },
        { replace: true },
      );
      return;
    }
    setSearchParams(
      (params) => {
        params.delete("start");
        return params;
      },
      { replace: true },
    );
    startCmd.run().catch(() => {
      // startCmd.error already carries this; the hero below shows it.
    });
    // Only ever reacts to the `start` param arriving, not to every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams]);

  const handleStart = (options?: StartRecordingOptions) => {
    startCmd.run(options).catch(() => {
      // Same as above: the hero reads startCmd.error directly.
    });
  };

  const handleResolveInterrupted = async (meetingId: Id, action: RecoveryAction) => {
    try {
      await interrupted.resolve(meetingId, action);
    } catch (err) {
      addToast({ level: "problem", message: toUiError(err).message });
    }
  };

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-8 px-8 py-10">
      <SearchInput
        placeholder={home.searchPlaceholder}
        onSubmitValue={(value) => {
          const trimmed = value.trim();
          navigate(trimmed ? `/search?q=${encodeURIComponent(trimmed)}` : "/search");
        }}
      />

      {interrupted.meetings.length > 0 ? (
        <RecoveryBanner
          meetings={interrupted.meetings}
          resolving={interrupted.resolving}
          onResolve={handleResolveInterrupted}
        />
      ) : (
        <StatusHero
          capture={capture}
          detection={detection}
          onStart={handleStart}
          starting={startCmd.loading}
          startErrorMessage={startCmd.error?.message}
          startErrorAction={
            startCmd.error?.action === "openMicrophoneSettings" ||
            startCmd.error?.action === "openScreenRecordingSettings"
              ? startCmd.error.action
              : undefined
          }
          lastCaption={lastCaption}
          preparing={preparing}
          preparingFraction={
            download && download.totalBytes > 0
              ? download.receivedBytes / download.totalBytes
              : undefined
          }
        />
      )}

      <RecentMeetings onStart={handleStart} starting={startCmd.loading} />
    </div>
  );
}
