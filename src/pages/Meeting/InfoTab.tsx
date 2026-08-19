import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { Button, Chip, Modal, ProgressBar } from "@/components";
import { formatBytes } from "@/components/lib/format";
import { channels, common, labels, meeting as copy, notices } from "@/lib/copy";
import { deleteMeeting } from "@/lib/ipc";
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

  const language = languageName(detail.meeting.language);
  const hasAudio = detail.audioBytes > 0 && !detail.meeting.deletedAt;
  const activeJobs = detail.jobs.filter((j) => j.status === "running" || j.status === "queued");

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
          {activeJobs.map((job) => (
            <div key={job.id} className="flex flex-col gap-1.5">
              <ProgressBar value={job.progress} label={labels.jobKind[job.kind]} />
              <span className="text-xs text-ink-faint">{labels.jobKind[job.kind]}…</span>
            </div>
          ))}
        </section>
      )}

      <section className="flex flex-col gap-3">
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
