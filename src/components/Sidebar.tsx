import { Link } from "react-router-dom";

import { useHasOutstandingWork } from "../hooks/useActiveJobs";
import { useCaptureState } from "../hooks/useCaptureState";
import { nav } from "../lib/copy";
import { formatElapsed } from "./lib/format";
import { EchoMark, HomeIcon, MeetingsIcon, PlusIcon, SettingsIcon } from "./icons";
import { RecordingDot } from "./RecordingDot";
import { SidebarNavItem } from "./SidebarNavItem";

/**
 * The left rail: wordmark, a "New meeting" pill, the three destinations, and
 * — instead of the privacy note — a live card while a meeting is recording.
 *
 * The window has no title bar of its own, so the rail opens with an empty
 * strip: the close/minimise/zoom buttons float over that space, and the strip
 * itself is what you grab to move the window. It has to stay empty — the
 * drag only works on the element that carries the attribute, never on a
 * parent wrapped around buttons.
 *
 * "New meeting" links to `/?start=1` rather than calling `startRecording`
 * itself: the Home screen owns the actual Start button and its confirmation
 * flow (title prompt, device picker), this just gets the person there with
 * the intent already expressed, the same way a notification click does.
 *
 * A dot appears on Meetings while Echo still has work outstanding on any of
 * them. It is the one indicator that still says something while a recording has
 * every outstanding job parked and nothing is running — which is exactly when
 * the corner pill, being a sentence about something happening, has nothing to
 * say. Wordless on purpose: a number here would be a queue depth, and
 * `lib/copy.ts` rules those out.
 */
export function Sidebar() {
  const capture = useCaptureState();
  const isLive = capture.state === "recording" || capture.state === "paused";
  const outstanding = useHasOutstandingWork();

  return (
    <nav
      aria-label={nav.wordmark}
      className="flex h-full w-56 shrink-0 flex-col gap-1 border-r border-hairline bg-surface px-3 pb-4"
    >
      <div data-tauri-drag-region className="h-9 shrink-0 select-none" />

      <div className="mb-4 flex items-center gap-2 px-3">
        <EchoMark className="h-5 w-5 text-ink" />
        <span className="text-sm font-semibold tracking-tight text-ink">{nav.wordmark}</span>
      </div>

      <Link
        to="/?start=1"
        className="echo-pill-primary mb-3 justify-center"
      >
        <PlusIcon className="h-4 w-4" />
        {nav.newMeeting}
      </Link>

      <SidebarNavItem to="/" end label={nav.home} icon={<HomeIcon className="h-full w-full" />} />
      <SidebarNavItem
        to="/search"
        label={nav.meetings}
        icon={<MeetingsIcon className="h-full w-full" />}
        badge={
          outstanding ? (
            <span className="flex items-center">
              {/* `ink-soft`, never `live` — red on the rail reads as "recording",
                  and this is the opposite: the meeting is over and Echo is still
                  tidying up after it. */}
              <span aria-hidden className="h-1.5 w-1.5 shrink-0 rounded-full bg-ink-soft" />
              <span className="sr-only">{nav.workingLabel}</span>
            </span>
          ) : undefined
        }
      />

      <div className="flex-1" />

      <SidebarNavItem to="/settings" label={nav.settings} icon={<SettingsIcon className="h-full w-full" />} />

      <div className="mt-3 px-3">
        {isLive ? (
          <Link
            to="/live"
            className="flex items-center gap-2 rounded-xl border border-hairline px-3 py-2 text-xs text-ink-soft transition-colors hover:bg-surface-sunken"
          >
            <RecordingDot size="sm" paused={capture.state === "paused"} />
            <span className="flex-1 font-medium">
              {capture.state === "paused" ? nav.livePausedLabel : nav.liveLabel}
            </span>
            <span className="tabular-nums text-ink-faint">{formatElapsed(capture.elapsedMs)}</span>
          </Link>
        ) : (
          <p className="text-xs text-ink-ghost">{nav.privacyNote}</p>
        )}
      </div>
    </nav>
  );
}
