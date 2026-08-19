import { useEffect } from "react";
import { createPortal } from "react-dom";
import { Link } from "react-router-dom";

import "../styles/animations.css";

import { common } from "../lib/copy";
import { useEchoStore, type Toast } from "../lib/store";
import type { NoticeLevel } from "../lib/types";
import { CloseIcon } from "./icons";
import { cx } from "./lib/cx";

const AUTO_DISMISS_MS = 5000;

const dotClasses: Record<NoticeLevel, string> = {
  info: "bg-ink-ghost",
  warning: "bg-ink",
  problem: "bg-live",
};

const borderClasses: Record<NoticeLevel, string> = {
  info: "border-hairline",
  warning: "border-hairline",
  problem: "border-live/30",
};

/** Mount once, near the root — see `App.tsx`. Renders whatever is in
 * `useEchoStore().toasts`, fed by the backend `notice` event and by screens
 * raising their own (e.g. "Copied."). */
export function ToastViewport() {
  const toasts = useEchoStore((s) => s.toasts);
  const dismissToast = useEchoStore((s) => s.dismissToast);

  if (toasts.length === 0) return null;

  return createPortal(
    <div
      aria-live="polite"
      className="pointer-events-none fixed inset-x-0 bottom-4 z-50 flex flex-col items-center gap-2 px-4"
    >
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onDismiss={dismissToast} />
      ))}
    </div>,
    document.body,
  );
}

function ToastItem({ toast, onDismiss }: { toast: Toast; onDismiss: (id: string) => void }) {
  useEffect(() => {
    if (toast.persistent) return undefined;
    const timer = setTimeout(() => onDismiss(toast.id), AUTO_DISMISS_MS);
    return () => clearTimeout(timer);
  }, [toast.id, toast.persistent, onDismiss]);

  return (
    <div
      role={toast.level === "problem" ? "alert" : "status"}
      className={cx(
        "echo-card animate-toast-in pointer-events-auto flex w-full max-w-sm items-start gap-2.5 border px-4 py-3 shadow-lift",
        borderClasses[toast.level],
      )}
    >
      <span aria-hidden className={cx("mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full", dotClasses[toast.level])} />
      <p className="flex-1 text-sm text-ink-soft">{toast.message}</p>
      {/* A notice about a particular meeting arrives long after the person
          moved on — "Your recap is ready." most of all, now that recaps write
          themselves. Without this the toast is a dead end (mantra 4): it tells
          you something is waiting and gives you no way to reach it. */}
      {toast.meetingId && (
        <Link
          to={`/meeting/${toast.meetingId}`}
          onClick={() => onDismiss(toast.id)}
          className="shrink-0 self-center text-xs font-medium text-ink underline-offset-2 hover:underline"
        >
          {common.open}
        </Link>
      )}
      <button
        type="button"
        onClick={() => onDismiss(toast.id)}
        aria-label={common.dismiss}
        className="text-ink-ghost transition-colors hover:text-ink"
      >
        <CloseIcon className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}
