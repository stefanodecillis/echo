import { Link } from "react-router-dom";

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
 * "New meeting" links to `/?start=1` rather than calling `startRecording`
 * itself: the Home screen owns the actual Start button and its confirmation
 * flow (title prompt, device picker), this just gets the person there with
 * the intent already expressed, the same way a notification click does.
 */
export function Sidebar() {
  const capture = useCaptureState();
  const isLive = capture.state === "recording" || capture.state === "paused";

  return (
    <nav
      aria-label={nav.wordmark}
      className="flex h-full w-56 shrink-0 flex-col gap-1 border-r border-hairline bg-surface px-3 py-4"
    >
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
      <SidebarNavItem to="/search" label={nav.meetings} icon={<MeetingsIcon className="h-full w-full" />} />

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
