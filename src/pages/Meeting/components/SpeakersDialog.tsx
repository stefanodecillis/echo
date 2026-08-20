import { useEffect, useRef, useState } from "react";

import { Button, Modal, ProgressBar } from "@/components";
import { renameSpeaker, setSpeakerCount, speakerSample, toUiError } from "@/lib/ipc";
import { common, labels, meeting as copy, peopleCount as peopleCountCopy } from "@/lib/copy";
import { useEchoStore } from "@/lib/store";
import type { Id, Job, Speaker } from "@/lib/types";

import { IconButton } from "./IconButton";
import { SpeakerRow, type ListenState } from "./SpeakerRow";
import { base64ToBlobUrl } from "../lib/audio";
import { MAX_PEOPLE_COUNT, MIN_PEOPLE_COUNT } from "../lib/peopleCount";
import { canonicalSpeakers } from "../lib/speakers";

export interface SpeakersDialogProps {
  open: boolean;
  onClose: () => void;
  meetingId: Id;
  speakers: Speaker[];
  /** How many people the automatic pass is currently using. */
  peopleCount: number;
  /** True once someone has corrected the count by hand. */
  peopleCountIsOverride: boolean;
  /** The job re-working out who said what, if one is running right now —
   * shown here as the in-progress state, alongside whatever else already
   * shows it (a fresh recording gets the same treatment). */
  job: Job | undefined;
  /** True while a recording is live or a rewrite is already running — either
   * way, the redo this dialog can trigger can't overlap it. */
  disabled: boolean;
}

/** What we're about to ask `setSpeakerCount` for, once the one confirmation
 * below has been answered. `count: null` is "back to automatic". */
interface PendingChange {
  count: number | null;
}

/**
 * "Name your speakers": one dialog for everything the Transcript tab needs to
 * say about who's who — how many people were there, a sample of each voice,
 * and a name for each. Replaces the old two-modal people-count editor and the
 * separate rename-on-chip flow; a chip now just opens this instead of editing
 * itself in place.
 */
export function SpeakersDialog({
  open,
  onClose,
  meetingId,
  speakers,
  peopleCount,
  peopleCountIsOverride,
  job,
  disabled,
}: SpeakersDialogProps) {
  const [draftCount, setDraftCount] = useState(peopleCount);
  const [pending, setPending] = useState<PendingChange>();
  const [submitting, setSubmitting] = useState(false);
  const [playingId, setPlayingId] = useState<string>();
  const [loadingId, setLoadingId] = useState<string>();
  const [rowError, setRowError] = useState<{ id: string; message: string }>();
  const sampleCache = useRef<Map<string, string>>(new Map());
  const audioRef = useRef<HTMLAudioElement>(null);
  const addToast = useEchoStore((s) => s.addToast);

  const rows = canonicalSpeakers(speakers);
  const canRerun = !disabled;

  // Only pick up a fresh automatic count while the dialog is open — a
  // background update shouldn't clobber a stepper value someone is mid-way
  // through changing.
  useEffect(() => {
    if (open) setDraftCount(Math.max(MIN_PEOPLE_COUNT, Math.min(MAX_PEOPLE_COUNT, peopleCount)));
  }, [open, peopleCount]);

  // Closing mid-play or mid-confirm shouldn't leave either running quietly
  // behind the dialog.
  useEffect(() => {
    if (!open) {
      audioRef.current?.pause();
      setPlayingId(undefined);
      setPending(undefined);
      setRowError(undefined);
    }
  }, [open]);

  // Samples belong to one meeting's speakers; a different meeting means
  // starting the cache over, and nothing should keep the old blob URLs alive.
  useEffect(() => {
    const cache = sampleCache.current;
    setPlayingId(undefined);
    setLoadingId(undefined);
    setRowError(undefined);
    return () => {
      cache.forEach((url) => URL.revokeObjectURL(url));
      cache.clear();
    };
  }, [meetingId]);

  const commitCount = async () => {
    if (!pending) return;
    setSubmitting(true);
    try {
      await setSpeakerCount(meetingId, pending.count);
      setPending(undefined);
    } catch (err) {
      addToast({ level: "problem", message: toUiError(err).message });
      setPending(undefined);
    } finally {
      setSubmitting(false);
    }
  };

  const handleRename = (speakerId: string, name: string) => {
    renameSpeaker(speakerId, name).catch((err) => {
      addToast({ level: "problem", message: toUiError(err).message });
    });
  };

  const play = (url: string, speakerId: string) => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.src = url;
    audio
      .play()
      .then(() => setPlayingId(speakerId))
      .catch(() => setRowError({ id: speakerId, message: copy.speakersDialogSampleError }));
  };

  const handleListen = (speaker: Speaker) => {
    if (playingId === speaker.id) {
      audioRef.current?.pause();
      setPlayingId(undefined);
      return;
    }
    setRowError(undefined);
    const cached = sampleCache.current.get(speaker.id);
    if (cached) {
      play(cached, speaker.id);
      return;
    }
    setLoadingId(speaker.id);
    speakerSample(meetingId, speaker.id)
      .then((base64) => {
        const url = base64ToBlobUrl(base64);
        sampleCache.current.set(speaker.id, url);
        play(url, speaker.id);
      })
      .catch((err) => {
        setRowError({ id: speaker.id, message: toUiError(err).message ?? copy.speakersDialogSampleError });
      })
      .finally(() => setLoadingId((id) => (id === speaker.id ? undefined : id)));
  };

  const listenStateFor = (speakerId: string): ListenState => {
    if (loadingId === speakerId) return "loading";
    if (playingId === speakerId) return "playing";
    return "idle";
  };

  return (
    <Modal
      open={open}
      onClose={onClose}
      title={copy.speakersDialogTitle}
      className="max-w-lg"
      footer={
        <Button variant="primary" onClick={onClose}>
          {common.done}
        </Button>
      }
    >
      <p className="mb-4">{copy.speakersDialogDescription}</p>

      <div className="mb-4 rounded-xl border border-hairline bg-surface-sunken p-3">
        {job ? (
          <div className="flex flex-col gap-1.5">
            <ProgressBar value={job.progress} label={labels.jobKind[job.kind]} />
            <span className="text-xs text-ink-faint">{labels.jobKind[job.kind]}…</span>
          </div>
        ) : pending ? (
          <div className="flex flex-col gap-2">
            <p className="text-sm text-ink">{peopleCountCopy.confirmTitle(pending.count)}</p>
            <p className="text-xs text-ink-faint">{copy.peopleCountConfirmDescription}</p>
            <div className="flex justify-end gap-2">
              <Button variant="secondary" size="sm" onClick={() => setPending(undefined)}>
                {common.cancel}
              </Button>
              <Button variant="primary" size="sm" loading={submitting} onClick={commitCount}>
                {copy.speakersDialogRerunButton}
              </Button>
            </div>
          </div>
        ) : (
          <div className="flex flex-wrap items-center justify-between gap-3">
            <span className="text-xs font-medium uppercase tracking-wide text-ink-faint">
              {copy.speakersDialogParticipantsLabel}
            </span>
            <div className="flex items-center gap-2">
              <IconButton
                icon={<span className="text-base leading-none">−</span>}
                aria-label={copy.peopleCountFewer}
                title={copy.peopleCountFewer}
                disabled={!canRerun || draftCount <= MIN_PEOPLE_COUNT}
                onClick={() => setDraftCount((n) => Math.max(MIN_PEOPLE_COUNT, n - 1))}
              />
              <span className="w-6 text-center text-lg font-semibold tabular-nums text-ink">{draftCount}</span>
              <IconButton
                icon={<span className="text-base leading-none">+</span>}
                aria-label={copy.peopleCountMore}
                title={copy.peopleCountMore}
                disabled={!canRerun || draftCount >= MAX_PEOPLE_COUNT}
                onClick={() => setDraftCount((n) => Math.min(MAX_PEOPLE_COUNT, n + 1))}
              />
              {peopleCountIsOverride && (
                <Button
                  variant="secondary"
                  size="sm"
                  disabled={!canRerun}
                  onClick={() => setPending({ count: null })}
                >
                  {copy.peopleCountBackToAutomatic}
                </Button>
              )}
              <Button
                variant="primary"
                size="sm"
                disabled={!canRerun || (peopleCountIsOverride && draftCount === peopleCount)}
                onClick={() => setPending({ count: draftCount })}
              >
                {copy.speakersDialogRerunButton}
              </Button>
            </div>
          </div>
        )}
      </div>

      {rows.length === 0 ? (
        <p className="rounded-xl border border-dashed border-hairline px-3 py-6 text-center text-sm text-ink-faint">
          {copy.speakersDialogEmpty}
        </p>
      ) : (
        <div className="flex flex-col gap-2">
          {rows.map((speaker) => (
            <SpeakerRow
              key={speaker.id}
              speaker={speaker}
              listenState={listenStateFor(speaker.id)}
              error={rowError?.id === speaker.id ? rowError.message : undefined}
              nameDisabled={!!job}
              onListen={() => handleListen(speaker)}
              onRename={(name) => handleRename(speaker.id, name)}
            />
          ))}
        </div>
      )}

      <audio ref={audioRef} className="hidden" onEnded={() => setPlayingId(undefined)} />
    </Modal>
  );
}
