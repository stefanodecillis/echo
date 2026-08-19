import { useEffect, useState } from "react";
import { Navigate, useParams, useSearchParams } from "react-router-dom";

import { EmptyState, Skeleton, SkeletonLines, Tabs } from "@/components";
import { meeting as copy } from "@/lib/copy";
import { updateMeetingTitle } from "@/lib/ipc";

import { InfoTab } from "./InfoTab";
import { MeetingHeader } from "./MeetingHeader";
import { RecapTab } from "./RecapTab";
import { TranscriptTab } from "./TranscriptTab";
import { useMeetingDetail } from "./hooks/useMeetingDetail";

type TabId = "recap" | "transcript" | "info";

function parseTab(value: string | null): TabId {
  return value === "transcript" || value === "info" ? value : "recap";
}

/**
 * `/meeting/:id` — header (editable title, date, duration, language),
 * Recap/Transcript/Info tabs.
 *
 * `?tab=` and `?t=` (a millisecond offset) let a search hit or another
 * screen land directly on a moment in the transcript; both are consumed
 * once the jump has happened so navigating away and back doesn't repeat it.
 */
export default function MeetingPage() {
  const { id: meetingId } = useParams<{ id: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const { detail, loading, error, setDetail } = useMeetingDetail(meetingId);

  const [activeTab, setActiveTab] = useState<TabId>(() => parseTab(searchParams.get("tab")));
  const [jumpToMs, setJumpToMs] = useState<number | undefined>(() => {
    const raw = searchParams.get("t");
    return raw !== null && !Number.isNaN(Number(raw)) ? Number(raw) : undefined;
  });

  useEffect(() => {
    if (!searchParams.has("tab") && !searchParams.has("t")) return;
    const next = new URLSearchParams(searchParams);
    next.delete("tab");
    next.delete("t");
    setSearchParams(next, { replace: true });
    // Consumed once, on arrival — not a dependency loop.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (!meetingId) return <Navigate to="/" replace />;

  if (error) {
    return (
      <div className="px-8 py-14">
        <EmptyState title={error.message} description={error.detail} />
      </div>
    );
  }

  if (loading || !detail) {
    return (
      <div className="flex flex-col gap-6 px-8 py-8">
        <Skeleton className="h-8 w-72" />
        <Skeleton className="h-4 w-40" />
        <SkeletonLines count={6} />
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <MeetingHeader
        meeting={detail.meeting}
        onTitleChange={(title) => {
          setDetail((d) => ({ ...d, meeting: { ...d.meeting, title } }));
          updateMeetingTitle(detail.meeting.id, title).catch(() => {
            // The header already shows what was typed; a background
            // reconciliation isn't worth interrupting the person for.
          });
        }}
      />

      <div className="px-8">
        <Tabs
          items={[
            { id: "recap", label: copy.tabRecap },
            { id: "transcript", label: copy.tabTranscript },
            { id: "info", label: copy.tabInfo },
          ]}
          value={activeTab}
          onChange={(tabId) => setActiveTab(tabId as TabId)}
        />
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {activeTab === "recap" && (
          <RecapTab
            meetingId={detail.meeting.id}
            meetingTitle={detail.meeting.title}
            detail={detail}
            setDetail={setDetail}
          />
        )}
        {activeTab === "transcript" && (
          <TranscriptTab
            meetingId={detail.meeting.id}
            detail={detail}
            setDetail={setDetail}
            jumpToMs={jumpToMs}
            onJumpConsumed={() => setJumpToMs(undefined)}
          />
        )}
        {activeTab === "info" && <InfoTab meetingId={detail.meeting.id} detail={detail} />}
      </div>
    </div>
  );
}
