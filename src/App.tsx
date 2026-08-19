import { Route, Routes, useLocation, useNavigate } from "react-router-dom";

import { Sidebar } from "./components/Sidebar";
import { ToastViewport } from "./components/Toast";
import { useEvent } from "./hooks/useEvent";
import { EVENTS, snoozeDetection, stopRecording } from "./lib/ipc";
import { useEchoStore } from "./lib/store";
import type { NavigatePayload } from "./lib/types";
import Home from "./routes/Home";
import Live from "./routes/Live";
import MeetingDetail from "./routes/MeetingDetail";
import NotFound from "./routes/NotFound";
import Onboarding from "./routes/Onboarding";
import Search from "./routes/Search";
import Settings from "./routes/Settings";

/** Where the Rust side can send the UI, and the URL that means. */
function pathFor(payload: NavigatePayload): string {
  switch (payload.target) {
    case "homeStart":
      // Home, with the Start button asking to be pressed.
      return "/?start=1";
    case "home":
      return "/";
    case "live":
      return "/live";
    case "meeting":
      return payload.meetingId ? `/meeting/${payload.meetingId}` : "/";
    case "search":
      return "/search";
    case "settings":
      return "/settings";
    case "onboarding":
      return "/onboarding";
  }
}

/**
 * The window shell: a left rail and one route.
 *
 * Onboarding runs full width, because a wizard with a navigation bar next to it
 * invites you to wander off halfway through.
 */
export default function App() {
  const navigate = useNavigate();
  const location = useLocation();
  const addToast = useEchoStore((s) => s.addToast);

  // The tray, a notification click and a second launch all arrive as one event.
  useEvent(EVENTS.navigate, (payload) => {
    navigate(pathFor(payload));
  });

  // Every banner the core wants shown becomes a toast, wherever the person
  // currently is — the core doesn't know or care which screen is open.
  useEvent(EVENTS.notice, (payload) => {
    addToast({
      level: payload.level,
      message: payload.message,
      persistent: payload.persistent,
      tag: payload.tag,
      meetingId: payload.meetingId,
    });
  });

  // The menu bar menu. "Start recording" and "Quit Echo" are already handled on
  // the Rust side (Start arrives as a navigate to Home with the button primed,
  // Quit closes the app), so only these two need an answer from here.
  useEvent(EVENTS.trayAction, (payload) => {
    switch (payload.action) {
      case "stop":
        // Idempotent: stopping when nothing is recording does nothing.
        void stopRecording().catch(() => {
          // The core banners anything worth saying about a failed stop.
        });
        break;
      case "pauseDetection":
        void snoozeDetection().catch(() => {});
        break;
      default:
        break;
    }
  });

  const bare = location.pathname.startsWith("/onboarding");

  return (
    <div className="flex h-full bg-surface text-ink">
      {!bare && <Sidebar />}
      <main className="h-full min-w-0 flex-1 overflow-y-auto">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/live" element={<Live />} />
          <Route path="/meeting/:id" element={<MeetingDetail />} />
          <Route path="/search" element={<Search />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/onboarding" element={<Onboarding />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>
      <ToastViewport />
    </div>
  );
}
