import { useEffect, useRef, useState } from "react";
import { save } from "@tauri-apps/plugin-dialog";

import { Button } from "@/components";
import { ChevronDownIcon } from "@/components/icons";
import { common, meeting as copy, notices } from "@/lib/copy";
import { exportMeeting, toUiError } from "@/lib/ipc";
import { useEchoStore } from "@/lib/store";
import type { ExportFormat, Id } from "@/lib/types";

interface ExportOption {
  format: ExportFormat;
  label: string;
  extension: string;
}

// The label doubles as the native save dialog's filter name, so there is only
// one place the person's word for each format is written.
//
// PDF is still listed even though the core cannot produce one yet (Tauri 2 has no
// headless print-to-file API). Picking it gets the plain "PDF isn't available on
// this computer. Word or Markdown will work." — a named next step rather than a
// dead end. Hide the row once a capability flag exists to hide it by.
const OPTIONS: ExportOption[] = [
  { format: "markdown", label: copy.exportMarkdown, extension: "md" },
  { format: "docx", label: copy.exportWord, extension: "docx" },
  { format: "pdf", label: copy.exportPdf, extension: "pdf" },
];

export interface ExportMenuProps {
  meetingId: Id;
  meetingTitle: string;
  summaryId?: Id;
}

/** The Export button: a small dropdown of formats, each backed by a native
 * save dialog so `exportMeeting` gets a real destination path. Every export
 * carries the recap, action items, and the full transcript together — there
 * is one export, not per-section ones, so it always includes what the person
 * came here for even if `summaryId` is left unset (no recap yet). Shown on
 * both the Recap tab (always, not only once a recap exists — the app is
 * usable before any recap is configured, DESIGN §0) and the Transcript tab,
 * so getting a meeting out is reachable from wherever a person is looking
 * for it, not just the one tab that happens to have a recap. */
export function ExportMenu({ meetingId, meetingTitle, summaryId }: ExportMenuProps) {
  const [open, setOpen] = useState(false);
  const [pending, setPending] = useState<ExportFormat>();
  const containerRef = useRef<HTMLDivElement>(null);
  const addToast = useEchoStore((s) => s.addToast);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: PointerEvent) => {
      if (!containerRef.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, [open]);

  const runExport = async (option: ExportOption) => {
    setOpen(false);
    const safeName = meetingTitle.trim().replace(/[/\\:*?"<>|]+/g, " ").trim() || "meeting";
    let destination: string | null;
    try {
      destination = await save({
        title: common.export,
        defaultPath: `${safeName}.${option.extension}`,
        filters: [{ name: option.label, extensions: [option.extension] }],
      });
    } catch {
      return;
    }
    if (!destination) return;

    setPending(option.format);
    try {
      await exportMeeting({
        meetingId,
        format: option.format,
        destination,
        includeRecap: true,
        includeActionItems: true,
        includeTranscript: true,
        summaryId,
      });
      addToast({ level: "info", message: notices.exportedTo });
    } catch (err) {
      addToast({ level: "problem", message: toUiError(err).message });
    } finally {
      setPending(undefined);
    }
  };

  return (
    <div ref={containerRef} className="relative">
      <Button
        variant="secondary"
        rightIcon={<ChevronDownIcon className="h-3.5 w-3.5" />}
        loading={pending !== undefined}
        onClick={() => setOpen((v) => !v)}
      >
        {common.export}
      </Button>
      {open && (
        <div className="echo-card absolute right-0 top-full z-20 mt-1.5 w-48 overflow-hidden p-1">
          {OPTIONS.map((option) => (
            <button
              key={option.format}
              type="button"
              onClick={() => runExport(option)}
              className="w-full rounded-lg px-3 py-2 text-left text-sm text-ink-soft transition-colors hover:bg-surface-sunken hover:text-ink"
            >
              {option.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
