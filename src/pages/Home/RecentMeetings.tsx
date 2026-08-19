import { Link } from "react-router-dom";

import { Card, Chip, EmptyState, MeetingsIcon, Skeleton } from "../../components";
import { common, home, languageLabel } from "../../lib/copy";
import type { MeetingSummary } from "../../lib/types";
import { formatDuration, formatRelativeDate } from "./format";
import { useRecentMeetings } from "./useRecentMeetings";

/** The recent-meetings section: a loading skeleton, an error with a retry,
 * the first-run empty state, or the rows themselves. */
export function RecentMeetings() {
  const { meetings, loading, error, reload } = useRecentMeetings();

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
          icon={<MeetingsIcon className="h-8 w-8" />}
          title={home.emptyTitle}
          description={home.emptyDescription}
        />
      ) : (
        <ul className="flex flex-col gap-2">
          {meetings.map((meeting) => (
            <MeetingRow key={meeting.id} meeting={meeting} />
          ))}
        </ul>
      )}
    </section>
  );
}

function MeetingRow({ meeting }: { meeting: MeetingSummary }) {
  const language = languageLabel(meeting.language);
  return (
    <li>
      <Link to={`/meeting/${meeting.id}`} className="block">
        <Card interactive padding="md" className="flex flex-col gap-1.5">
          <div className="flex items-center justify-between gap-3">
            <h3 className="truncate text-sm font-medium text-ink">{meeting.title}</h3>
            <span className="shrink-0 text-xs text-ink-faint">
              {formatRelativeDate(meeting.startedAt)}
            </span>
          </div>
          <div className="flex items-center gap-2 text-xs text-ink-faint">
            <span>{formatDuration(meeting.durationMs)}</span>
            {language && <Chip>{language}</Chip>}
          </div>
          {meeting.snippet && <p className="truncate text-sm text-ink-soft">{meeting.snippet}</p>}
        </Card>
      </Link>
    </li>
  );
}
