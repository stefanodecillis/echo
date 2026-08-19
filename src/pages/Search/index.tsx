import { useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";

import { EmptyState, SearchInput, Skeleton } from "@/components";
import { search as copy } from "@/lib/copy";
import { listMeetings, searchTranscripts } from "@/lib/ipc";
import type { Id, SearchHit } from "@/lib/types";

import { formatResultDate, formatTimestamp } from "./lib/date";

interface MeetingGroup {
  meetingId: Id;
  meetingTitle: string;
  startedAt: string;
  hits: SearchHit[];
}

/** Long enough that typing doesn't fire a query per keystroke, short enough
 * that it still feels instant. */
const DEBOUNCE_MS = 300;
const MIN_QUERY_LENGTH = 2;

function groupByMeeting(hits: SearchHit[]): MeetingGroup[] {
  const order: Id[] = [];
  const byMeeting = new Map<Id, MeetingGroup>();
  for (const hit of hits) {
    let group = byMeeting.get(hit.meetingId);
    if (!group) {
      group = {
        meetingId: hit.meetingId,
        meetingTitle: hit.meetingTitle,
        startedAt: hit.startedAt,
        hits: [],
      };
      byMeeting.set(hit.meetingId, group);
      order.push(hit.meetingId);
    }
    group.hits.push(hit);
  }
  return order.map((id) => byMeeting.get(id)!);
}

/**
 * `/search` — full-text search across every meeting Echo has kept, grouped
 * by meeting, with the matching words underlined. Clicking a hit opens that
 * meeting's Transcript tab already scrolled to the moment it was said.
 */
export default function SearchPage() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [query, setQuery] = useState(() => searchParams.get("q") ?? "");
  const [hits, setHits] = useState<SearchHit[]>();
  const [searching, setSearching] = useState(false);
  const [hasAnyMeetings, setHasAnyMeetings] = useState<boolean>();

  // Whether there is anything to search at all, checked once — an empty
  // library reads very differently from "nothing found for that word".
  useEffect(() => {
    listMeetings({ limit: 1 })
      .then((list) => setHasAnyMeetings(list.length > 0))
      .catch(() => setHasAnyMeetings(true));
  }, []);

  // Keep the query in the URL so the search survives a back-navigation or a
  // copied link, without pushing a history entry per keystroke.
  useEffect(() => {
    const trimmed = query.trim();
    const next = new URLSearchParams(searchParams);
    if (trimmed) next.set("q", trimmed);
    else next.delete("q");
    setSearchParams(next, { replace: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query]);

  useEffect(() => {
    const trimmed = query.trim();
    if (trimmed.length < MIN_QUERY_LENGTH) {
      setHits(undefined);
      setSearching(false);
      return;
    }
    setSearching(true);
    const timer = setTimeout(() => {
      searchTranscripts({ text: trimmed, limit: 200 })
        .then(setHits)
        .catch(() => setHits([]))
        .finally(() => setSearching(false));
    }, DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [query]);

  const groups = useMemo(() => (hits ? groupByMeeting(hits) : []), [hits]);
  const resultCount = hits?.length ?? 0;

  const goToHit = (hit: SearchHit) => {
    navigate(`/meeting/${hit.meetingId}?tab=transcript&t=${hit.tStartMs}`);
  };

  const showIdle = hasAnyMeetings !== false && query.trim().length === 0;
  const showEmptyLibrary = hasAnyMeetings === false;
  const showLoading = !showEmptyLibrary && !showIdle && searching;
  const showNoResults = !showEmptyLibrary && !showIdle && !searching && hits !== undefined && groups.length === 0;
  const showResults = !showEmptyLibrary && !showIdle && !searching && groups.length > 0;

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 px-8 py-10">
      <h1 className="text-2xl font-semibold tracking-tight text-ink">{copy.title}</h1>

      <SearchInput
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder={copy.placeholder}
        autoFocus
      />

      {showEmptyLibrary && (
        <EmptyState title={copy.noMeetingsTitle} description={copy.noMeetingsDescription} />
      )}

      {showIdle && <EmptyState title={copy.idleTitle} description={copy.idleDescription} />}

      {showLoading && (
        <div className="flex flex-col gap-3" aria-hidden>
          <Skeleton className="h-16" />
          <Skeleton className="h-16" />
          <Skeleton className="h-16" />
        </div>
      )}

      {showNoResults && <EmptyState title={copy.emptyTitle} description={copy.emptyDescription} />}

      {showResults && (
        <div className="flex flex-col gap-6">
          <p className="text-sm text-ink-faint">
            {resultCount} {resultCount === 1 ? copy.resultSingular : copy.resultPlural}
          </p>
          {groups.map((group) => (
            <MeetingResultGroup key={group.meetingId} group={group} onNavigate={navigate} onHit={goToHit} />
          ))}
        </div>
      )}
    </div>
  );
}

function MeetingResultGroup({
  group,
  onNavigate,
  onHit,
}: {
  group: MeetingGroup;
  onNavigate: (path: string) => void;
  onHit: (hit: SearchHit) => void;
}) {
  return (
    <div className="echo-card overflow-hidden">
      <button
        type="button"
        onClick={() => onNavigate(`/meeting/${group.meetingId}`)}
        className="flex w-full items-baseline justify-between gap-4 border-b border-hairline bg-surface-sunken px-5 py-3 text-left transition-colors hover:bg-hairline/60"
      >
        <span className="truncate text-sm font-semibold text-ink">{group.meetingTitle}</span>
        <span className="shrink-0 text-xs text-ink-faint">{formatResultDate(group.startedAt)}</span>
      </button>
      <ul className="divide-y divide-hairline">
        {group.hits.map((hit) => (
          <li key={hit.segmentId}>
            <button
              type="button"
              onClick={() => onHit(hit)}
              className="flex w-full items-start gap-3 px-5 py-3 text-left transition-colors hover:bg-surface-sunken"
            >
              <span className="w-12 shrink-0 pt-0.5 text-xs tabular-nums text-ink-ghost">
                {formatTimestamp(hit.tStartMs)}
              </span>
              {/* `snippetHtml` is pre-escaped by the backend with only
                  `<mark>` around matches — see the field's contract in
                  `src/lib/types.ts`. Nothing else ever goes through
                  `dangerouslySetInnerHTML` in this app. */}
              <span
                className="min-w-0 flex-1 text-sm text-ink-soft [&_mark]:rounded [&_mark]:bg-accent-soft [&_mark]:px-0.5 [&_mark]:text-ink [&_mark]:not-italic"
                dangerouslySetInnerHTML={{ __html: hit.snippetHtml }}
              />
              {hit.speakerName && (
                <span className="shrink-0 text-xs text-ink-faint">{hit.speakerName}</span>
              )}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
