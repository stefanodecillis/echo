import { Button, Card } from "../../components";
import { home } from "../../lib/copy";
import type { Id, Meeting, RecoveryAction } from "../../lib/types";
import { formatRelativeDate } from "./format";

export interface RecoveryBannerProps {
  meetings: Meeting[];
  resolving: Id | null;
  onResolve: (meetingId: Id, action: RecoveryAction) => void;
}

/**
 * Home shows this instead of the usual status hero when the core found one or
 * more meetings that didn't finish — after a crash, a forced quit, a dead
 * battery. Every meeting gets its own Finish/Let-it-go choice; nothing is
 * decided for the person, and a fresh Start stays out of reach until they've
 * dealt with what's here.
 */
export function RecoveryBanner({ meetings, resolving, onResolve }: RecoveryBannerProps) {
  return (
    <Card padding="lg" className="flex flex-col gap-4">
      <div>
        <h1 className="text-lg font-semibold tracking-tight text-ink">{home.recoveringTitle}</h1>
        <p className="mt-1 text-sm text-ink-soft">{home.recoveringDescription}</p>
      </div>
      <ul className="flex flex-col gap-2">
        {meetings.map((meeting) => (
          <li
            key={meeting.id}
            className="flex items-center justify-between gap-3 rounded-xl border border-hairline px-4 py-3"
          >
            <div className="min-w-0">
              <p className="truncate text-sm font-medium text-ink">{meeting.title}</p>
              <p className="text-xs text-ink-faint">{formatRelativeDate(meeting.startedAt)}</p>
            </div>
            <div className="flex shrink-0 items-center gap-2">
              <Button
                size="sm"
                variant="ghost"
                disabled={resolving === meeting.id}
                onClick={() => onResolve(meeting.id, "discard")}
              >
                {home.recoveringDiscard}
              </Button>
              <Button
                size="sm"
                variant="primary"
                loading={resolving === meeting.id}
                onClick={() => onResolve(meeting.id, "finish")}
              >
                {home.recoveringFinish}
              </Button>
            </div>
          </li>
        ))}
      </ul>
    </Card>
  );
}
