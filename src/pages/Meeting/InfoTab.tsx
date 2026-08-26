import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { save } from "@tauri-apps/plugin-dialog";

import { Button, Chip, Modal, ProgressBar } from "@/components";
import { formatBytes } from "@/components/lib/format";
import { channels, common, meeting as copy, notices } from "@/lib/copy";
import { isActive, jobSentence, presentJob, runningFirst } from "@/lib/jobs";
import { deleteMeeting, downloadRecording, toUiError } from "@/lib/ipc";
import { useEchoStore } from "@/lib/store";
import type { Channel, Id, MeetingDetail } from "@/lib/types";

import { languageName } from "./lib/language";
import { formatDuration } from "./lib/date";

export interface InfoTabProps {
  meetingId: Id;
  detail: MeetingDetail;
}

const channelOrder: Channel[] = ["mic", "system", "mixed"];

/** What was captured, how much room it takes, and the two ways to let it
 * go — audio-only, or everything. Both ask first. */
export function InfoTab({ meetingId, detail }: InfoTabProps) {
  const navigate = useNavigate();
  const addToast = useEchoStore((s) => s.addToast);
  const [confirmAudio, setConfirmAudio] = useState(false);
  const [confirmAll, setConfirmAll] = useState(false);
  const [pending, setPending] = useState(false);
  const [savingRecording, setSavingRecording] = useState(false);

  const language = languageName(detail.meeting.language);
  const hasAudio = detail.audioBytes > 0 && !detail.meeting.deletedAt;
  // What is happening first, then what is waiting — and the two look
  // different, because they are (see `lib/jobs.ts`). A queued job with a
  // progress bar over it reads as work in flight that has stalled.
  const activeJobs = runningFirst(detail.jobs.filter(isActive));

  /** Builds the playback mix on demand if it isn't ready yet (an older
   * meeting, or one where that pass hasn't finished) — `downloadRecording`
   * waits on that itself, so this button just shows a spinner the whole
   * time. Its actual progress is already visible above, in "Still working",
   * the same way any other background job is. */
  const handleSaveRecording = async () => {
    const safeName = detail.meeting.title.trim().replace(/[/\\:*?"<>|]+/g, " ").trim() || "meeting";
    let destination: string | null;
    try {
      destination = await save({
        title: common.export,
        defaultPath: `${safeName}.wav`,
        filters: [{ name: copy.recordingFileType, extensions: ["wav"] }],
      });
    } catch {
      return;
    }
    if (!destination) return;

    setSavingRecording(true);
    try {
      await downloadRecording(meetingId, destination);
      addToast({ level: "info", message: notices.exportedTo });
    } catch (err) {
      addToast({ level: "problem", message: toUiError(err).message });
    } finally {
      setSavingRecording(false);
    }
  };

  const runDelete = async (mode: "audioOnly" | "everything") => {
    setPending(true);
    try {
      await deleteMeeting(meetingId, mode);
      setConfirmAudio(false);
      setConfirmAll(false);
      if (mode === "everything") {
        addToast({ level: "info", message: notices.meetingDeleted });
        navigate("/");
      } else {
        addToast({ level: "info", message: notices.audioDeleted });
      }
    } catch {
      addToast({ level: "problem", message: notices.somethingWentWrong });
    } finally {
      setPending(false);
    }
  };

  return (
    <div className="flex flex-col gap-8 px-8 py-6">
      <section className="echo-card flex flex-col gap-4 p-6">
        <h2 className="text-sm font-semibold text-ink">{copy.infoCapturedTitle}</h2>
        <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm sm:grid-cols-3">
          <div>
            <dt className="text-ink-faint">{copy.infoCapturedLabel}</dt>
            <dd className="mt-1 flex flex-wrap gap-1.5">
              {channelOrder
                .filter((c) => detail.capturedChannels.includes(c))
                .map((c) => (
                  <Chip key={c}>{channels[c]}</Chip>
                ))}
              {detail.capturedChannels.length === 0 && <span className="text-ink-ghost">—</span>}
            </dd>
          </div>
          <div>
            <dt className="text-ink-faint">{copy.infoLanguageLabel}</dt>
            <dd className="mt-1 text-ink">{language ?? copy.languageDetecting}</dd>
          </div>
          <div>
            <dt className="text-ink-faint">{copy.infoDurationLabel}</dt>
            <dd className="mt-1 text-ink">{formatDuration(detail.meeting.durationMs)}</dd>
          </div>
          <div>
            <dt className="text-ink-faint">{copy.infoStorageUsed}</dt>
            <dd className="mt-1 text-ink">{formatBytes(detail.audioBytes)}</dd>
          </div>
          <div>
            <dt className="text-ink-faint">{copy.infoLinesLabel}</dt>
            <dd className="mt-1 text-ink">{detail.segmentCount}</dd>
          </div>
        </dl>
      </section>

      {activeJobs.length > 0 && (
        <section className="echo-card flex flex-col gap-3 p-6">
          <h2 className="text-sm font-semibold text-ink">{copy.infoWorkingTitle}</h2>
          {activeJobs.map((job) => {
            const shown = presentJob(job);
            return (
              <div key={job.id} className="flex flex-col gap-1.5">
                {shown.running && <ProgressBar value={shown.fraction} label={shown.label} />}
                <span className="text-xs text-ink-faint">
                  {jobSentence(shown)}
                </span>
              </div>
            );
          })}
        </section>
      )}

      <section className="flex flex-col gap-3">
        {hasAudio && (
          <div className="flex items-center justify-between gap-4 rounded-xl border border-hairline p-4">
            <div>
              <p className="text-sm font-medium text-ink">{copy.infoSaveRecording}</p>
              <p className="text-xs text-ink-faint">{copy.infoSaveRecordingDescription}</p>
            </div>
            <Button variant="secondary" loading={savingRecording} onClick={handleSaveRecording}>
              {common.save}
            </Button>
          </div>
        )}
        {hasAudio && (
          <div className="flex items-center justify-between gap-4 rounded-xl border border-hairline p-4">
            <div>
              <p className="text-sm font-medium text-ink">{copy.infoDeleteAudio}</p>
              <p className="text-xs text-ink-faint">{copy.infoDeleteAudioDescription}</p>
            </div>
            <Button variant="secondary" onClick={() => setConfirmAudio(true)}>
              {common.delete}
            </Button>
          </div>
        )}
        <div className="flex items-center justify-between gap-4 rounded-xl border border-hairline p-4">
          <div>
            <p className="text-sm font-medium text-ink">{copy.infoDeleteAll}</p>
            <p className="text-xs text-ink-faint">{copy.infoDeleteAllDescription}</p>
          </div>
          <Button variant="secondary" onClick={() => setConfirmAll(true)}>
            {common.delete}
          </Button>
        </div>
      </section>

      <Modal
        open={confirmAudio}
        onClose={() => setConfirmAudio(false)}
        title={copy.deleteAudioConfirmTitle}
        footer={
          <>
            <Button variant="secondary" onClick={() => setConfirmAudio(false)}>
              {common.cancel}
            </Button>
            <Button variant="primary" loading={pending} onClick={() => runDelete("audioOnly")}>
              {common.delete}
            </Button>
          </>
        }
      >
        {copy.deleteAudioConfirmDescription}
      </Modal>

      <Modal
        open={confirmAll}
        onClose={() => setConfirmAll(false)}
        title={copy.deleteConfirmTitle}
        footer={
          <>
            <Button variant="secondary" onClick={() => setConfirmAll(false)}>
              {common.cancel}
            </Button>
            <Button variant="primary" loading={pending} onClick={() => runDelete("everything")}>
              {common.delete}
            </Button>
          </>
        }
      >
        {copy.deleteConfirmDescription}
      </Modal>
    </div>
  );
}
