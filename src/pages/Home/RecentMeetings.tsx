import { Link } from "react-router-dom";

import { Button, Card, EchoMark, EmptyState, Skeleton } from "../../components";
import { common, home } from "../../lib/copy";
import type { StartRecordingOptions } from "../../lib/types";
import { MeetingRow } from "./MeetingRow";
import { useRecentMeetings } from "./useRecentMeetings";

export interface RecentMeetingsProps {
  /** Wired to the same Start flow as the (now-gone) idle hero, so the
   * all-empty state is never a dead end. */
  onStart?: (options?: StartRecordingOptions) => void;
  starting?: boolean;
}

/** The recent-meetings section: a loading skeleton, an error with a retry,
 * the first-run empty state, or the rows themselves. */
export function RecentMeetings({ onStart, starting }: RecentMeetingsProps) {
  const { meetings, loading, error, reload, removeMeeting } = useRecentMeetings();

  return (
    <section className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-ink">{home.recentTitle}</h2>
        {meetings.length > 0 && (
          <Link to="/search" className="text-xs font-medium text-ink-faint hover:text-ink">
            {home.viewAll}
          </Link>
        )}
      </div>

      {loading ? (
        <div className="flex flex-col gap-2">
          <Skeleton className="h-16 w-full" />
          <Skeleton className="h-16 w-full" />
          <Skeleton className="h-16 w-full" />
        </div>
      ) : error ? (
        <Card padding="md" className="flex items-center justify-between gap-3">
          <p className="text-sm text-ink-faint">{error.message}</p>
          <button
            type="button"
            onClick={reload}
            className="shrink-0 text-sm font-medium text-ink underline-offset-2 hover:underline"
          >
            {common.retry}
          </button>
        </Card>
      ) : meetings.length === 0 ? (
        <EmptyState
          icon={<EchoMark className="h-8 w-8" />}
          title={home.emptyTitle}
          description={home.emptyDescription}
          action={
            onStart ? (
              <Button variant="secondary" loading={starting} onClick={() => onStart()}>
                {home.startButton}
              </Button>
            ) : undefined
          }
        />
      ) : (
        <ul className="flex flex-col gap-2">
          {meetings.map((meeting) => (
            <MeetingRow key={meeting.id} meeting={meeting} onDeleted={removeMeeting} />
          ))}
        </ul>
      )}
    </section>
  );
}
