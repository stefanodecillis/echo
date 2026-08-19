import type { Settings, SettingsPatch } from "../../lib/types";

/** Shared shape for any section that reads or writes the one `Settings`
 * object owned by `index.tsx`. */
export interface SectionProps {
  settings: Settings;
  /** Sends a patch to the core and merges the authoritative result back into
   * the shared copy every section reads from. */
  patch: (next: SettingsPatch) => Promise<Settings>;
}
