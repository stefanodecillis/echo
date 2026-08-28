import { useState } from "react";
import { Link } from "react-router-dom";

import { Button, Card, Chip, Modal, TrashIcon, WorkChip } from "../../components";
import { useCommand, useMeetingWork } from "../../hooks";
import { common, home, languageLabel } from "../../lib/copy";
import { deleteMeeting } from "../../lib/ipc";
import type { Id, MeetingSummary } from "../../lib/types";
import { formatDuration, formatRelativeDate } from "./format";

export interface MeetingRowProps {
  meeting: MeetingSummary;
  /** Called once the meeting is actually gone, so the list can drop the row. */
  onDeleted?: (meetingId: Id) => void;
}

/**
 * One row in a meetings list: title, date, duration, language, snippet, and
 * a quiet delete affordance that only shows up on hover. Shared by Home's
 * Recent meetings and the full Meetings page (`/search`) so both read as the
 * same list, because they are — both come from `listMeetings`.
 *
 * It also says whether Echo has finished with the meeting: which one is being
 * worked on right now, which is waiting its turn, and which stopped without
 * finishing. That answer comes from `useMeetingWork`, which reads a map kept
 * fresh for the whole app — not from the list, which knows nothing about it. So
 * neither list's own reload cycle has anything to do with it, and neither list
 * hook had to change.
 */
export function MeetingRow({ meeting, onDeleted }: MeetingRowProps) {
  const language = languageLabel(meeting.language);
  const work = useMeetingWork(meeting);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const deleteCmd = useCommand(deleteMeeting);

  const handleDelete = async () => {
    try {
      await deleteCmd.run(meeting.id, "everything");
      setConfirmOpen(false);
      onDeleted?.(meeting.id);
    } catch {
      // deleteCmd.error already reflects it below; the dialog stays open so
      // the person can see it and try again.
    }
  };

  return (
    <li className="group relative">
      <Link to={`/meeting/${meeting.id}`} className="block">
        <Card interactive padding="md" className="flex flex-col gap-1.5 pr-11">
          <div className="flex items-center justify-between gap-3">
            <h3 className="truncate text-sm font-medium text-ink">{meeting.title}</h3>
            <span className="shrink-0 text-xs text-ink-faint">
              {formatRelativeDate(meeting.startedAt)}
            </span>
          </div>
          {/* The status is what yields on a narrow window: `min-w-0` here plus
              `truncate` inside the chip, against `shrink-0` on the two facts that
              are always worth their space. Nothing is absolutely positioned, so
              the card's `pr-11` gutter still belongs entirely to the delete
              button below. */}
          <div className="flex min-w-0 items-center gap-2 text-xs text-ink-faint">
            <span className="shrink-0">{formatDuration(meeting.durationMs)}</span>
            {language && <Chip className="shrink-0">{language}</Chip>}
            <WorkChip work={work} />
          </div>
          {meeting.snippet && <p className="truncate text-sm text-ink-soft">{meeting.snippet}</p>}
        </Card>
      </Link>

      <button
        type="button"
        aria-label={home.deleteRowLabel}
        onClick={(event) => {
          event.preventDefault();
          event.stopPropagation();
          setConfirmOpen(true);
        }}
        className="absolute right-3 top-1/2 flex h-8 w-8 -translate-y-1/2 items-center justify-center rounded-lg bg-surface-sunken text-ink-ghost opacity-0 outline-none transition-colors hover:bg-live/10 hover:text-live focus-visible:opacity-100 focus-visible:ring-2 focus-visible:ring-live/40 group-hover:opacity-100 group-focus-within:opacity-100"
      >
        <TrashIcon className="h-4 w-4" />
      </button>

      <Modal
        open={confirmOpen}
        onClose={() => setConfirmOpen(false)}
        title={home.deleteRowConfirmTitle}
        footer={
          <>
            <Button variant="secondary" onClick={() => setConfirmOpen(false)}>
              {common.cancel}
            </Button>
            <Button variant="primary" loading={deleteCmd.loading} onClick={handleDelete}>
              {common.delete}
            </Button>
          </>
        }
      >
        {home.deleteRowConfirmDescription}
        {deleteCmd.error && <p className="mt-2 text-xs text-live">{deleteCmd.error.message}</p>}
      </Modal>
    </li>
  );
}
