/**
 * Turns a base64 WAV clip into something an `<audio>` element can play: a
 * `blob:` URL, which the app's CSP allows under `media-src` (a `data:` URL is
 * not). Duplicated from `Meeting/lib/audio.ts` rather than imported, to keep
 * this screen's files self-contained under its own ownership — same eight
 * lines either way. Callers own the returned URL and should
 * `URL.revokeObjectURL` it once nothing can play it any more.
 */
export function base64ToBlobUrl(base64: string, mimeType = "audio/wav"): string {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return URL.createObjectURL(new Blob([bytes], { type: mimeType }));
}
