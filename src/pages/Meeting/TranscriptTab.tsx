import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";

import { Button, EmptyState, SearchInput, VirtualList, type VirtualListHandle } from "@/components";
import { EVENTS, getTranscript, listSpeakers, renameSpeaker } from "@/lib/ipc";
import { common, meeting as copy, notices } from "@/lib/copy";
import { useEvent } from "@/hooks/useEvent";
import { useEchoStore } from "@/lib/store";
import type { Id, MeetingDetail, Segment } from "@/lib/types";

import { MergeSpeakersModal } from "./components/MergeSpeakersModal";
import { SpeakerChip } from "./components/SpeakerChip";
import { formatTimestamp } from "./lib/date";
import { resolveSpeaker } from "./lib/speakers";

const ROW_HEIGHT = 96;

export interface TranscriptTabProps {
  meetingId: Id;
  detail: MeetingDetail;
  setDetail: (updater: (detail: MeetingDetail) => MeetingDetail) => void;
  /** Milliseconds to scroll to once the transcript has loaded — set when
   * arriving from a search hit. Consumed once. */
  jumpToMs?: number;
  onJumpConsumed: () => void;
}

function highlightMatches(text: string, query: string): ReactNode {
  if (!query.trim()) return text;
  const lower = text.toLowerCase();
  const needle = query.toLowerCase();
  const parts: ReactNode[] = [];
  let start = 0;
  let index = lower.indexOf(needle, start);
  let key = 0;
  while (index !== -1) {
    if (index > start) parts.push(text.slice(start, index));
    parts.push(
      <mark key={key++} className="rounded bg-accent-soft text-ink">
        {text.slice(index, index + needle.length)}
      </mark>,
    );
    start = index + needle.length;
    index = lower.indexOf(needle, start);
  }
  if (start < text.length) parts.push(text.slice(start));
  return parts;
}

/** The Transcript tab: a virtualized, filterable list of segments with
 * clickable speaker chips for renaming, and a "combine two speakers" flow
 * for the provisional labels an offline pass hasn't stabilized yet. */
export function TranscriptTab({ meetingId, detail, setDetail, jumpToMs, onJumpConsumed }: TranscriptTabProps) {
  const [segments, setSegments] = useState<Segment[]>();
  const [filter, setFilter] = useState("");
  const [mergeOpen, setMergeOpen] = useState(false);
  const listRef = useRef<VirtualListHandle>(null);
  const addToast = useEchoStore((s) => s.addToast);

  useEffect(() => {
    let cancelled = false;
    getTranscript({ meetingId })
      .then((result) => {
        if (!cancelled) setSegments(result);
      })
      .catch(() => {
        if (!cancelled) setSegments([]);
      });
    return () => {
      cancelled = true;
    };
  }, [meetingId]);

  useEvent(EVENTS.transcriptFinal, (payload) => {
    if (payload.meetingId !== meetingId) return;
    setSegments((prev) => {
      const list = prev ?? [];
      const existingIndex = list.findIndex((s) => s.id === payload.segment.id);
      if (existingIndex === -1) return [...list, payload.segment];
      const next = [...list];
      next[existingIndex] = payload.segment;
      return next;
    });
  });

  const filtered = useMemo(() => {
    const list = segments ?? [];
    const query = filter.trim().toLowerCase();
    if (!query) return list;
    return list.filter((s) => s.text.toLowerCase().includes(query));
  }, [segments, filter]);

  useEffect(() => {
    if (jumpToMs === undefined || !segments || segments.length === 0) return;
    const index = segments.findIndex((s) => s.tStartMs >= jumpToMs);
    const target = index === -1 ? segments.length - 1 : index;
    listRef.current?.scrollToIndex(Math.max(0, target));
    onJumpConsumed();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jumpToMs, segments]);

  const handleRename = (speakerId: string, name: string) => {
    setDetail((d) => ({
      ...d,
      speakers: d.speakers.map((s) => (s.id === speakerId ? { ...s, displayName: name } : s)),
    }));
    renameSpeaker(speakerId, name).catch(() => {
      addToast({ level: "problem", message: notices.somethingWentWrong });
    });
  };

  const handleMerged = () => {
    listSpeakers(meetingId)
      .then((speakers) => setDetail((d) => ({ ...d, speakers })))
      .catch(() => {
        // The speakers-updated event, if it arrives, reconciles this.
      });
  };

  if (segments === undefined) {
    return <div className="px-8 py-6 text-sm text-ink-faint">{common.loading}</div>;
  }

  return (
    <div className="flex h-full flex-col gap-4 px-8 py-6">
      <div className="flex flex-wrap items-center gap-3">
        <SearchInput
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder={copy.transcriptFilterPlaceholder}
          containerClassName="max-w-sm"
        />
        {detail.speakers.length > 1 && (
          <Button variant="secondary" size="sm" onClick={() => setMergeOpen(true)} className="ml-auto">
            {copy.mergeSpeakersButton}
          </Button>
        )}
      </div>

      {filtered.length === 0 ? (
        <EmptyState title={filter ? copy.transcriptNoMatches : copy.transcriptEmpty} />
      ) : (
        <VirtualList
          ref={listRef}
          items={filtered}
          itemHeight={ROW_HEIGHT}
          getKey={(s) => s.id}
          className="flex-1"
          renderItem={(segment) => {
            const speaker = resolveSpeaker(segment.speakerId, detail.speakers);
            return (
              <div className="flex gap-4 border-b border-hairline px-1 py-3">
                <span className="w-12 shrink-0 pt-0.5 text-xs tabular-nums text-ink-ghost">
                  {formatTimestamp(segment.tStartMs)}
                </span>
                <div className="flex min-w-0 flex-1 flex-col gap-1.5">
                  <SpeakerChip speaker={speaker} fallbackId={segment.speakerId} onRename={handleRename} />
                  <p className="line-clamp-3 text-sm leading-relaxed text-ink-soft">
                    {highlightMatches(segment.text, filter)}
                  </p>
                </div>
              </div>
            );
          }}
        />
      )}

      <MergeSpeakersModal
        open={mergeOpen}
        onClose={() => setMergeOpen(false)}
        speakers={detail.speakers}
        onMerged={handleMerged}
      />
    </div>
  );
}
