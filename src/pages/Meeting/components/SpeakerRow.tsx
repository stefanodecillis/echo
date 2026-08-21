import { useEffect, useRef, useState } from "react";

import { Button, EchoMark } from "@/components";
import { CloseIcon } from "@/components/icons";
import { cx } from "@/components/lib/cx";
import { meeting as copy, common } from "@/lib/copy";
import type { Id, PersonInfo, Speaker } from "@/lib/types";

import { speakerColor } from "../lib/speakerColor";
import { IconButton } from "./IconButton";

export type ListenState = "idle" | "loading" | "playing";

export interface SpeakerRowProps {
  speaker: Speaker;
  listenState: ListenState;
  /** A polite, inline explanation when the sample couldn't be fetched or
   * played — shown under this row only, never a dead-end toast. */
  error: string | undefined;
  /** True while the offline pass that could replace this row is running. */
  nameDisabled: boolean;
  /** Every known person, for the name field's combo. */
  people: PersonInfo[];
  /** The suggested person's name, resolved from `speaker.suggestedPersonId`
   * against `people` — undefined when there's no suggestion, it's already
   * been dismissed this session, or the suggested person can't be found
   * (nothing to name in the chip, so nothing is shown). */
  suggestionName: string | undefined;
  /** True while this row's Confirm or Unlink is in flight. */
  personActionPending: boolean;
  onListen: () => void;
  onRename: (name: string) => void;
  /** "Remember this voice": enroll the typed, not-yet-known name as a new
   * person instead of a plain meeting-local rename. */
  onEnroll: (name: string) => void;
  onConfirmSuggestion: () => void;
  onDismissSuggestion: () => void;
  onLinkPerson: (personId: Id) => void;
  onUnlink: () => void;
}

/** A triangle — "play this". Kept local rather than added to the shared icon
 * set, since nothing outside this one row needs it. */
function PlayGlyph() {
  return (
    <svg width="11" height="11" viewBox="0 0 20 20" fill="none" aria-hidden>
      <path d="M6.5 4.3v11.4l9-5.7-9-5.7z" fill="currentColor" />
    </svg>
  );
}

/** A square — "stop this". Pairs with `PlayGlyph` as the toggled state of the
 * same Listen button. */
function StopGlyph() {
  return (
    <svg width="11" height="11" viewBox="0 0 20 20" fill="none" aria-hidden>
      <rect x="5.5" y="5.5" width="9" height="9" rx="1.5" fill="currentColor" />
    </svg>
  );
}

/** Three dots — a row's "more options" menu trigger. Local for the same
 * reason as the glyphs above: only this row's Unlink action needs it. */
function MoreGlyph() {
  return (
    <svg width="12" height="12" viewBox="0 0 20 20" fill="none" aria-hidden>
      <circle cx="4" cy="10" r="1.6" fill="currentColor" />
      <circle cx="10" cy="10" r="1.6" fill="currentColor" />
      <circle cx="16" cy="10" r="1.6" fill="currentColor" />
    </svg>
  );
}

/** The one-item row menu a linked row gets: Unlink. A single-item menu still
 * earns its own disclosure rather than a bare button, so a row that is
 * *linked* reads differently at a glance from one that is not. */
function RowMenu({
  disabled,
  onUnlink,
}: {
  disabled: boolean;
  onUnlink: () => void;
}) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: MouseEvent) => {
      if (!containerRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  return (
    <div ref={containerRef} className="relative shrink-0">
      <IconButton
        icon={<MoreGlyph />}
        aria-label={copy.speakersDialogRowMenuLabel}
        title={copy.speakersDialogRowMenuLabel}
        disabled={disabled}
        onClick={() => setOpen((v) => !v)}
      />
      {open && (
        <div className="absolute right-0 top-full z-10 mt-1 min-w-[8rem] rounded-lg border border-hairline bg-surface py-1 shadow-lift">
          <button
            type="button"
            disabled={disabled}
            onClick={() => {
              setOpen(false);
              onUnlink();
            }}
            className="block w-full px-3 py-1.5 text-left text-xs text-ink-soft transition-colors hover:bg-surface-sunken hover:text-ink disabled:cursor-not-allowed disabled:opacity-60"
          >
            {copy.speakersDialogUnlinkAction}
          </button>
        </div>
      )}
    </div>
  );
}

const MAX_COMBO_MATCHES = 6;

/**
 * One speaker: a colored dot, a name field that is both a free-typed rename
 * and a combo of known people, and a button to hear a sample of that voice
 * before deciding what to type.
 *
 * Known-people additions (docs/DESIGN.md §1) layer onto the original
 * rename-on-blur-or-Enter behavior rather than replacing it:
 *  - typing a name that matches no known person still just renames, same as
 *    before, unless "Remember this voice" is checked, which enrolls instead;
 *  - picking a person from the dropdown (click, or Enter on a highlighted
 *    match) links instead of renaming;
 *  - a suggestion chip offers a one-click Confirm (also a link) with a quiet
 *    dismiss that only hides the chip, never the row;
 *  - a linked row shows a small mark by its name and gets an Unlink menu, and
 *    is honest that editing its name here is meeting-local, not a rename of
 *    the person.
 */
export function SpeakerRow({
  speaker,
  listenState,
  error,
  nameDisabled,
  people,
  suggestionName,
  personActionPending,
  onListen,
  onRename,
  onEnroll,
  onConfirmSuggestion,
  onDismissSuggestion,
  onLinkPerson,
  onUnlink,
}: SpeakerRowProps) {
  const [draft, setDraft] = useState(speaker.displayName);
  const [editing, setEditing] = useState(false);
  const [comboOpen, setComboOpen] = useState(false);
  const [highlight, setHighlight] = useState(-1);
  const [remember, setRemember] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  // Picking a person calls `blur()` itself, to close the field the same way
  // Escape/Enter do — but the ordinary blur-commits-a-rename behavior would
  // then immediately "rename" the row to the name just picked, double-firing
  // alongside the link. This skips exactly that one blur.
  const skipNextBlurCommit = useRef(false);

  useEffect(() => {
    setDraft(speaker.displayName);
    setRemember(false);
  }, [speaker.displayName]);

  const color = speakerColor(speaker.id);
  const isLinked = !!speaker.personId;

  const trimmed = draft.trim();
  const matchesKnownPerson = people.some(
    (p) => p.name.toLowerCase() === trimmed.toLowerCase(),
  );
  const isNewName = trimmed !== "" && trimmed !== speaker.displayName && !matchesKnownPerson;
  const matches =
    comboOpen && trimmed !== ""
      ? people
          .filter((p) => p.name.toLowerCase().includes(trimmed.toLowerCase()))
          .slice(0, MAX_COMBO_MATCHES)
      : [];

  const closeCombo = () => {
    setComboOpen(false);
    setHighlight(-1);
  };

  const commit = () => {
    closeCombo();
    const value = draft.trim();
    if (!value || value === speaker.displayName) {
      setDraft(speaker.displayName);
      setRemember(false);
      return;
    }
    const isNew = !people.some((p) => p.name.toLowerCase() === value.toLowerCase());
    if (remember && isNew) onEnroll(value);
    else onRename(value);
    setRemember(false);
  };

  const pick = (person: PersonInfo) => {
    setDraft(person.name);
    setRemember(false);
    closeCombo();
    onLinkPerson(person.id);
    skipNextBlurCommit.current = true;
    inputRef.current?.blur();
  };

  return (
    <div className="flex flex-col gap-2 rounded-xl border border-hairline px-3 py-2.5">
      <div className="flex items-center gap-3">
        <span aria-hidden className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ backgroundColor: color.fg }} />
        <div className="relative flex min-w-0 flex-1 flex-col gap-1">
          <div className="flex items-center gap-1.5">
            {isLinked && (
              <span
                role="img"
                aria-label={copy.speakersDialogKnownPersonHint}
                title={copy.speakersDialogKnownPersonHint}
                className="shrink-0 text-accent"
              >
                <EchoMark aria-hidden className="h-3 w-3" />
              </span>
            )}
            <input
              ref={inputRef}
              value={draft}
              disabled={nameDisabled}
              onChange={(e) => {
                setDraft(e.target.value);
                setComboOpen(true);
                setHighlight(-1);
              }}
              onFocus={() => setEditing(true)}
              onBlur={() => {
                setEditing(false);
                if (skipNextBlurCommit.current) {
                  skipNextBlurCommit.current = false;
                  return;
                }
                commit();
              }}
              onKeyDown={(e) => {
                if (comboOpen && matches.length > 0) {
                  if (e.key === "ArrowDown") {
                    e.preventDefault();
                    setHighlight((h) => Math.min(h + 1, matches.length - 1));
                    return;
                  }
                  if (e.key === "ArrowUp") {
                    e.preventDefault();
                    setHighlight((h) => Math.max(h - 1, 0));
                    return;
                  }
                  if (e.key === "Enter" && highlight >= 0) {
                    e.preventDefault();
                    pick(matches[highlight]);
                    return;
                  }
                }
                if (e.key === "Enter") {
                  commit();
                  inputRef.current?.blur();
                }
                if (e.key === "Escape") {
                  setDraft(speaker.displayName);
                  setRemember(false);
                  closeCombo();
                  inputRef.current?.blur();
                }
              }}
              placeholder={copy.renameSpeakerPlaceholder}
              className="w-full rounded-lg border border-transparent bg-transparent px-1.5 py-1 text-sm font-medium text-ink outline-none transition-colors hover:border-hairline focus:border-hairline focus:bg-surface focus:ring-2 focus:ring-accent/30 disabled:cursor-not-allowed disabled:opacity-60"
            />
          </div>

          {matches.length > 0 && (
            <div className="absolute left-0 top-full z-10 mt-1 w-full rounded-lg border border-hairline bg-surface py-1 shadow-lift">
              {matches.map((person, i) => (
                <button
                  key={person.id}
                  type="button"
                  onMouseDown={(e) => {
                    e.preventDefault();
                    pick(person);
                  }}
                  className={cx(
                    "block w-full px-2.5 py-1.5 text-left text-xs",
                    i === highlight ? "bg-surface-sunken text-ink" : "text-ink-soft hover:bg-surface-sunken",
                  )}
                >
                  {person.name}
                </button>
              ))}
            </div>
          )}

          {isNewName && (
            <label className="flex items-center gap-1.5 pl-1.5 text-xs text-ink-faint">
              <input
                type="checkbox"
                checked={remember}
                onChange={(e) => setRemember(e.target.checked)}
              />
              {copy.speakersDialogRememberVoice}
            </label>
          )}
          {isLinked && editing && !isNewName && (
            <p className="pl-1.5 text-xs text-ink-faint">{copy.speakersDialogLocalRenameHint}</p>
          )}
          {error && <p className="pl-1.5 text-xs text-live">{error}</p>}
        </div>
        <Button
          variant="secondary"
          size="sm"
          loading={listenState === "loading"}
          leftIcon={listenState === "playing" ? <StopGlyph /> : <PlayGlyph />}
          onClick={onListen}
        >
          {listenState === "playing" ? copy.speakersDialogStopLabel : copy.speakersDialogListenLabel}
        </Button>
        {isLinked && <RowMenu disabled={personActionPending} onUnlink={onUnlink} />}
      </div>

      {suggestionName && !isLinked && (
        <div className="ml-5 flex items-center gap-2 rounded-lg bg-surface-sunken px-2.5 py-1.5 text-xs text-ink-soft">
          <span className="min-w-0 flex-1 truncate">
            {copy.speakersDialogSuggestionPrefix}{" "}
            <span className="font-medium text-ink">{suggestionName}</span>
          </span>
          <button
            type="button"
            disabled={personActionPending}
            onClick={onConfirmSuggestion}
            className="shrink-0 font-medium text-accent underline-offset-2 transition-colors hover:underline disabled:cursor-not-allowed disabled:opacity-60"
          >
            {copy.speakersDialogSuggestionConfirm}
          </button>
          <button
            type="button"
            aria-label={common.dismiss}
            title={common.dismiss}
            onClick={onDismissSuggestion}
            className="shrink-0 text-ink-ghost transition-colors hover:text-ink-soft"
          >
            <CloseIcon className="h-3 w-3" />
          </button>
        </div>
      )}
    </div>
  );
}
