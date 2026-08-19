/** ISO 639-1 codes the speech engine can detect, spelled out in plain words
 * for the language chip. Falls back to the bare (uppercased) code for
 * anything not in this short list — still not jargon, just a code a person
 * can recognize as "a language", same as a passport stamp. */
const LANGUAGE_NAMES: Record<string, string> = {
  en: "English",
  es: "Spanish",
  fr: "French",
  de: "German",
  it: "Italian",
  pt: "Portuguese",
  nl: "Dutch",
  pl: "Polish",
  ru: "Russian",
  uk: "Ukrainian",
  sv: "Swedish",
  no: "Norwegian",
  da: "Danish",
  fi: "Finnish",
  tr: "Turkish",
  el: "Greek",
  cs: "Czech",
  ro: "Romanian",
  hu: "Hungarian",
  ja: "Japanese",
  ko: "Korean",
  zh: "Chinese",
  ar: "Arabic",
  hi: "Hindi",
  id: "Indonesian",
  vi: "Vietnamese",
  th: "Thai",
};

/** `"en"` -> `"English"`. `undefined` -> the caller's own "still working"
 * copy, since this function only knows how to name a code, not its absence. */
export function languageName(code?: string): string | undefined {
  if (!code) return undefined;
  return LANGUAGE_NAMES[code.toLowerCase()] ?? code.toUpperCase();
}
