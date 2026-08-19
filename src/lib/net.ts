/** Whether an address stays on this computer. Mirrors the core's own check —
 * the UI uses it only to decide when to show the "this leaves your Mac"
 * acknowledgement; the core enforces it regardless. Unparseable input counts
 * as leaving, which errs on the side of showing the warning. */
export function isLoopbackAddress(address: string): boolean {
  const trimmed = address.trim();
  if (!trimmed) return true;
  let url: URL;
  try {
    url = new URL(trimmed.includes("://") ? trimmed : `http://${trimmed}`);
  } catch {
    return false;
  }
  const host = url.hostname.replace(/^\[|\]$/g, "");
  return host === "localhost" || host === "::1" || /^127(\.\d{1,3}){3}$/.test(host);
}
