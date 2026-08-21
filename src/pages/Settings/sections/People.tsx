import { useCallback, useEffect, useRef, useState } from "react";

import { Card, EmptyState } from "@/components";
import { useEvent } from "@/hooks/useEvent";
import { EVENTS, speakerSample, toUiError } from "@/lib/ipc";
import { common, settings as copy } from "@/lib/copy";
import { useEchoStore } from "@/lib/store";
import type { Id } from "@/lib/types";

import { PersonRow } from "../components/PersonRow";
import { SuggestedPersonRow } from "../components/SuggestedPersonRow";
import { base64ToBlobUrl } from "../lib/audio";
import {
  acceptSuggestedPerson,
  deletePerson,
  listPeople,
  personSampleAudio,
  renamePerson,
  suggestedPeople,
  type ListenState,
  type Person,
  type SuggestedPerson,
} from "../lib/peopleIpc";

/** Which row's sample is loading or playing right now — one shared player
 * for the whole screen, so starting a second clip stops the first. */
type PlayKey = `person:${Id}` | `suggestion:${Id}`;

/**
 * Settings > People: the saved-voices list docs/DESIGN.md §1 asks for, plus
 * the "People Echo keeps hearing" list of recurring voices nobody has named
 * yet. One plain sentence at the top says what these are — voice data that
 * stays on this Mac and can always be deleted — and nothing below it ever
 * uses a technical word for how matching works.
 */
export function People() {
  const [people, setPeople] = useState<Person[] | null>(null);
  const [peopleError, setPeopleError] = useState<string>();
  const [suggestions, setSuggestions] = useState<SuggestedPerson[] | null>(null);
  const [suggestionsError, setSuggestionsError] = useState<string>();

  const [playingKey, setPlayingKey] = useState<PlayKey>();
  const [loadingKey, setLoadingKey] = useState<PlayKey>();
  const [rowError, setRowError] = useState<{ key: PlayKey; message: string }>();

  const audioRef = useRef<HTMLAudioElement>(null);
  const sampleCache = useRef<Map<PlayKey, string>>(new Map());
  const addToast = useEchoStore((s) => s.addToast);

  const refresh = useCallback(() => {
    listPeople()
      .then(setPeople)
      .catch((err) => setPeopleError(toUiError(err).message));
    suggestedPeople()
      .then(setSuggestions)
      .catch((err) => setSuggestionsError(toUiError(err).message));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Renaming, enrolling or forgetting elsewhere (or Echo finishing a refresh
  // pass) all land here. The event carries the whole list, so the saved
  // voices come straight off the payload and only the suggestions — which an
  // enrollment removes one of — have to be asked for again.
  useEvent(EVENTS.peopleUpdated, ({ people: updated }) => {
    setPeople(updated);
    setPeopleError(undefined);
    suggestedPeople()
      .then(setSuggestions)
      .catch((err) => setSuggestionsError(toUiError(err).message));
  });

  // Blob URLs are only good for as long as this screen is mounted.
  useEffect(() => {
    const cache = sampleCache.current;
    return () => {
      cache.forEach((url) => URL.revokeObjectURL(url));
      cache.clear();
    };
  }, []);

  const play = (url: string, key: PlayKey) => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.src = url;
    audio
      .play()
      .then(() => setPlayingKey(key))
      .catch(() => setRowError({ key, message: copy.peopleSampleError }));
  };

  const listen = (key: PlayKey, fetchSample: () => Promise<string>) => {
    if (playingKey === key) {
      audioRef.current?.pause();
      setPlayingKey(undefined);
      return;
    }
    setRowError(undefined);
    const cached = sampleCache.current.get(key);
    if (cached) {
      play(cached, key);
      return;
    }
    setLoadingKey(key);
    fetchSample()
      .then((base64) => {
        const url = base64ToBlobUrl(base64);
        sampleCache.current.set(key, url);
        play(url, key);
      })
      .catch((err) => {
        setRowError({ key, message: toUiError(err).message || copy.peopleSampleError });
      })
      .finally(() => setLoadingKey((k) => (k === key ? undefined : k)));
  };

  const listenStateFor = (key: PlayKey): ListenState => {
    if (loadingKey === key) return "loading";
    if (playingKey === key) return "playing";
    return "idle";
  };

  const handleRename = (personId: Id, name: string) => {
    renamePerson(personId, name)
      .then(() =>
        setPeople((prev) =>
          prev ? prev.map((p) => (p.id === personId ? { ...p, name } : p)) : prev,
        ),
      )
      .catch((err) => addToast({ level: "problem", message: toUiError(err).message }));
  };

  const handleDelete = async (personId: Id) => {
    await deletePerson(personId);
    setPeople((prev) => prev?.filter((p) => p.id !== personId) ?? prev);
  };

  const handleSaveSuggestion = async (suggestion: SuggestedPerson, name: string) => {
    const created = await acceptSuggestedPerson(suggestion.meetingId, suggestion.speakerId, name);
    setSuggestions((prev) => prev?.filter((s) => s.id !== suggestion.id) ?? prev);
    setPeople((prev) => (prev ? [...prev, created] : prev));
  };

  return (
    <div className="flex flex-col gap-6">
      <p className="text-sm text-ink-faint">{copy.peopleIntro}</p>

      <Card padding="sm">
        {people === null && !peopleError && (
          <p className="px-2 py-3 text-sm text-ink-faint">{common.loading}</p>
        )}
        {peopleError && <p className="px-2 py-3 text-sm text-live">{peopleError}</p>}
        {people && people.length === 0 && (
          <EmptyState title={copy.peopleSavedEmptyTitle} description={copy.peopleSavedEmptyDescription} />
        )}
        {people && people.length > 0 && (
          <div className="flex flex-col gap-2">
            {people.map((person) => {
              const key: PlayKey = `person:${person.id}`;
              return (
                <PersonRow
                  key={person.id}
                  person={person}
                  listenState={listenStateFor(key)}
                  error={rowError?.key === key ? rowError.message : undefined}
                  onListen={() => listen(key, () => personSampleAudio(person.id))}
                  onRename={(name) => handleRename(person.id, name)}
                  onDelete={() => handleDelete(person.id)}
                />
              );
            })}
          </div>
        )}
      </Card>

      <div className="flex flex-col gap-3">
        <div>
          <h2 className="text-sm font-semibold text-ink">{copy.peopleSuggestedTitle}</h2>
          <p className="mt-0.5 text-xs text-ink-faint">{copy.peopleSuggestedDescription}</p>
        </div>

        <Card padding="sm">
          {suggestions === null && !suggestionsError && (
            <p className="px-2 py-3 text-sm text-ink-faint">{common.loading}</p>
          )}
          {suggestionsError && <p className="px-2 py-3 text-sm text-live">{suggestionsError}</p>}
          {suggestions && suggestions.length === 0 && (
            <EmptyState
              title={copy.peopleSuggestedEmptyTitle}
              description={copy.peopleSuggestedEmptyDescription}
            />
          )}
          {suggestions && suggestions.length > 0 && (
            <div className="flex flex-col gap-2">
              {suggestions.map((suggestion, index) => {
                const key: PlayKey = `suggestion:${suggestion.id}`;
                return (
                  <SuggestedPersonRow
                    key={suggestion.id}
                    suggestion={suggestion}
                    index={index}
                    listenState={listenStateFor(key)}
                    error={rowError?.key === key ? rowError.message : undefined}
                    onListen={() =>
                      listen(key, () => speakerSample(suggestion.meetingId, suggestion.speakerId))
                    }
                    onSave={(name) => handleSaveSuggestion(suggestion, name)}
                  />
                );
              })}
            </div>
          )}
        </Card>
      </div>

      <audio ref={audioRef} className="hidden" onEnded={() => setPlayingKey(undefined)} />
    </div>
  );
}
