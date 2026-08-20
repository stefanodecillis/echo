/**
 * Turns the base64 WAV `speakerSample` hands back into something an
 * `<audio>` element can actually play: a `blob:` URL, which the CSP already
 * allows for media. Callers own the returned URL and should
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
