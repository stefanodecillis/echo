import type { Correction } from "@/lib/types";

/**
 * One stretch of a transcript line, and whether Echo wrote it or heard it.
 *
 * The line is cut into these so the words Echo put right can carry the mark —
 * and the offer to put them back — instead of the whole sentence carrying it.
 */
export interface TranscriptSpan {
  text: string;
  /** True when this stretch is a word Echo replaced against "Words Echo
   * should know". */
  corrected: boolean;
}

/** Letter or digit, in any language — the only thing a word boundary is about
 * here. Mirrors Rust's `char::is_alphanumeric`. */
const WORD_CHAR = /[\p{L}\p{N}]/u;

/**
 * Where `needle` next sits in `text` as a word of its own, at or after `from`,
 * or `-1`.
 *
 * "A word of its own" is only about letters and digits on either side: a repair
 * is written over a stretch already trimmed to letters, so the line keeps
 * whatever punctuation hung off it — "Langola." and "«Langola»" are both the
 * word, and "Langolas" is not.
 *
 * The mirror of `next_whole_word` in `src-tauri/src/asr/glossary.rs`, and it has
 * to stay one: this decides which words are offered as a click target, and the
 * command behind that click refuses any line whose repairs it cannot place the
 * same way. Marking a word the core would not put back would offer an undo that
 * always fails.
 */
function nextWholeWord(text: string, needle: string, from: number): number {
  if (!needle) return -1;
  let at = from;
  for (;;) {
    const start = text.indexOf(needle, at);
    if (start === -1) return -1;
    const end = start + needle.length;
    const before = start > 0 ? text[start - 1] : "";
    const after = end < text.length ? text[end] : "";
    if (!WORD_CHAR.test(before) && !WORD_CHAR.test(after)) return start;
    // Past this hit, not past the whole of it: overlapping occurrences of a
    // repeated word are still separate words.
    at = start + 1;
  }
}

/** How many times `needle` is a word of its own in `text`. */
function wholeWordHits(text: string, needle: string): number {
  let count = 0;
  let at = 0;
  for (;;) {
    const hit = nextWholeWord(text, needle, at);
    if (hit === -1) return count;
    count += 1;
    at = hit + needle.length;
  }
}

/**
 * Cut a line into the words Echo repaired and everything around them, or `null`
 * when the note kept against the line no longer fits the line.
 *
 * The `null` is the useful half, and it is deliberately the same answer
 * `asr::glossary::revert` gives on the same input: a replacement missing from
 * the line, or in it more often than it was made, means some later pass rewrote
 * the words and nothing stored here can say which occurrence was the repair.
 * The command behind the click refuses exactly those lines, so this refuses to
 * offer the click.
 *
 * The screen's fallback for `null` is the mark it has always had — the whole
 * line underlined, saying that something on it was put right, with nothing
 * offering to undo it. That is honest: there is nothing that can be undone.
 *
 * Corrections were applied left to right over disjoint stretches, so they are
 * placed left to right behind a cursor here, which is the same assignment
 * arrived at from the other end.
 */
export function correctedSpans(
  text: string,
  corrections: readonly Correction[],
): TranscriptSpan[] | null {
  if (corrections.length === 0) return null;

  for (const change of corrections) {
    if (!change.to) return null;
    const made = corrections.filter((other) => other.to === change.to).length;
    if (wholeWordHits(text, change.to) !== made) return null;
  }

  const spans: TranscriptSpan[] = [];
  let cursor = 0;
  for (const change of corrections) {
    const at = nextWholeWord(text, change.to, cursor);
    if (at === -1) return null;
    if (at > cursor) spans.push({ text: text.slice(cursor, at), corrected: false });
    spans.push({ text: change.to, corrected: true });
    cursor = at + change.to.length;
  }
  if (cursor < text.length) spans.push({ text: text.slice(cursor), corrected: false });
  return spans;
}
