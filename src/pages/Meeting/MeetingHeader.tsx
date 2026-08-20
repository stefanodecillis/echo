import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import { Chip } from "@/components";
import { common, meeting as copy } from "@/lib/copy";
import type { Meeting } from "@/lib/types";

import { formatDuration, formatMeetingDate } from "./lib/date";
import { languageName } from "./lib/language";

export interface MeetingHeaderProps {
  meeting: Meeting;
  /** Fires after the title is committed to the backend, so the parent can
   * update its own copy without a full reload. */
  onTitleChange: (title: string) => void;
}

/** Editable title, date, duration and a language chip — the only things
 * every tab agrees are true about this meeting. */
export function MeetingHeader({ meeting, onTitleChange }: MeetingHeaderProps) {
  const navigate = useNavigate();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(meeting.title);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!editing) setDraft(meeting.title);
  }, [meeting.title, editing]);

  useEffect(() => {
    if (editing) inputRef.current?.select();
  }, [editing]);

  const commit = () => {
    setEditing(false);
    const trimmed = draft.trim();
    if (trimmed === "" || trimmed === meeting.title) {
      setDraft(meeting.title);
      return;
    }
    onTitleChange(trimmed);
  };

  const displayTitle = meeting.title.trim() || copy.untitledMeeting;
  const language = languageName(meeting.language);

  return (
    <header className="flex flex-col gap-3 px-8 py-6">
      <button
        type="button"
        onClick={() => navigate(-1)}
        className="w-fit text-xs font-medium text-ink-faint transition-colors hover:text-ink-soft"
      >
        ← {common.back}
      </button>

      {editing ? (
        <input
          ref={inputRef}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === "Enter") commit();
            if (e.key === "Escape") {
              setDraft(meeting.title);
              setEditing(false);
            }
          }}
          aria-label={copy.renameTitle}
          className="w-full max-w-xl rounded-lg border border-hairline bg-surface px-2 py-1 text-2xl font-semibold tracking-tight text-ink focus:outline-none focus:ring-2 focus:ring-accent/30"
        />
      ) : (
        <button
          type="button"
          onClick={() => setEditing(true)}
          className="w-fit rounded-lg px-2 py-1 text-left text-2xl font-semibold tracking-tight text-ink transition-colors hover:bg-surface-sunken"
          title={copy.renameTitle}
        >
          {displayTitle}
        </button>
      )}

      <div className="flex flex-wrap items-center gap-2 px-2">
        <span className="text-sm text-ink-faint">{formatMeetingDate(meeting.startedAt)}</span>
        <span className="text-ink-ghost">·</span>
        <span className="text-sm text-ink-faint">{formatDuration(meeting.durationMs)}</span>
        <Chip variant="outline">{language ?? copy.languageDetecting}</Chip>
      </div>
    </header>
  );
}
