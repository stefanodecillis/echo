import { useEffect, useRef, useState } from "react";

import { Card, EmptyState, Input, PlusIcon } from "@/components";
import { addVocabularyWord, listVocabulary, removeVocabularyWord, toUiError } from "@/lib/ipc";
import { common, settings as copy } from "@/lib/copy";
import type { VocabularyWord } from "@/lib/types";

/**
 * Settings > Words: the list Echo checks the meeting against.
 *
 * Two kinds of row, and the difference is only ever a small grey note: what
 * somebody typed, and the name of a voice they saved — Echo puts those in by
 * itself, because they are exactly the words it gets wrong (a meeting on
 * 2026-08-24 turned "Gianluca", a name Echo had been told months earlier,
 * into "Jan Luca"). Removing either kind sticks: a name Echo added is
 * remembered as removed rather than quietly put back at the next launch.
 *
 * Every command hands back the whole list, so the screen never has to guess at
 * what the core now holds.
 */
export function Words() {
  const [words, setWords] = useState<VocabularyWord[] | null>(null);
  const [loadError, setLoadError] = useState<string>();
  const [draft, setDraft] = useState("");
  const [addError, setAddError] = useState<string>();
  const [busy, setBusy] = useState(false);
  const fieldRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    listVocabulary()
      .then(setWords)
      .catch((err) => setLoadError(toUiError(err).message));
  }, []);

  function add() {
    const word = draft.trim();
    if (!word || busy) return;
    setBusy(true);
    setAddError(undefined);
    addVocabularyWord(word)
      .then((next) => {
        setWords(next);
        setDraft("");
        // Adding a list of names is a run of typing, so the field keeps focus.
        fieldRef.current?.focus();
      })
      .catch((err) => setAddError(toUiError(err).message))
      .finally(() => setBusy(false));
  }

  function remove(word: string) {
    removeVocabularyWord(word)
      .then(setWords)
      .catch((err) => setAddError(toUiError(err).message));
  }

  return (
    <div className="flex flex-col gap-6">
      <p className="text-sm text-ink-faint">{copy.wordsIntro}</p>

      <div className="flex flex-col gap-1.5">
        <div className="flex gap-2">
          <Input
            ref={fieldRef}
            value={draft}
            placeholder={copy.wordsAddPlaceholder}
            invalid={Boolean(addError)}
            onChange={(e) => {
              setDraft(e.target.value);
              setAddError(undefined);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                add();
              }
            }}
          />
          <button
            type="button"
            onClick={add}
            disabled={!draft.trim() || busy}
            className="echo-pill-quiet shrink-0 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <PlusIcon className="h-4 w-4" />
            {copy.wordsAddButton}
          </button>
        </div>
        {addError && <p className="text-xs text-live">{addError}</p>}
      </div>

      <Card padding="none">
        {words === null && !loadError && (
          <p className="px-5 py-3.5 text-sm text-ink-faint">{common.loading}</p>
        )}
        {loadError && <p className="px-5 py-3.5 text-sm text-live">{loadError}</p>}
        {words && words.length === 0 && (
          <EmptyState title={copy.wordsEmptyTitle} description={copy.wordsEmptyDescription} />
        )}
        {words && words.length > 0 && (
          <div className="divide-y divide-hairline">
            {words.map((entry) => (
              <div
                key={entry.word}
                className="flex items-center justify-between gap-3 px-5 py-3.5"
              >
                <div className="flex min-w-0 flex-col">
                  <p className="truncate text-sm font-medium text-ink">{entry.word}</p>
                  {entry.source === "person" && (
                    <p className="mt-0.5 text-xs text-ink-faint">{copy.wordsFromPersonNote}</p>
                  )}
                </div>
                <button
                  type="button"
                  onClick={() => remove(entry.word)}
                  className="shrink-0 text-xs font-medium text-ink-soft hover:text-ink"
                >
                  {copy.wordsRemoveButton}
                </button>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}
