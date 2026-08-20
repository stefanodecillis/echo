/**
 * A small, hand-drawn icon set so the component library doesn't need an icon
 * package. Deliberately narrow: only what the foundation itself uses. Screen
 * agents needing more should add to this file rather than pulling in a
 * dependency — `package.json` is out of this agent's owned paths.
 */
import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement>;

function base(props: IconProps) {
  return {
    viewBox: "0 0 20 20",
    fill: "none",
    "aria-hidden": true,
    ...props,
  };
}

export function SearchIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <circle cx="9" cy="9" r="6" stroke="currentColor" strokeWidth="1.6" />
      <path d="M14 14l4 4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}

export function CloseIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M5 5l10 10M15 5L5 15" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}

export function PlusIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M10 4v12M4 10h12" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}

export function HomeIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path
        d="M4 9.5L10 4l6 5.5V16a1 1 0 01-1 1h-3v-5H8v5H5a1 1 0 01-1-1V9.5z"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function MeetingsIcon(props: IconProps) {
  // A page of notes, not a calendar: this list is what was said, not what is
  // coming up.
  return (
    <svg {...base(props)}>
      <rect x="4.5" y="3" width="11" height="14" rx="2" stroke="currentColor" strokeWidth="1.4" />
      <path
        d="M7.3 7.2h5.4M7.3 10h5.4M7.3 12.8h3.2"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function SettingsIcon(props: IconProps) {
  // Sliders: unmistakably "adjust things", unlike a gear whose spokes read as a
  // sun at this size.
  return (
    <svg {...base(props)}>
      <path d="M3.5 7h13M3.5 13h13" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
      {/* Two rails, not three: at the 16px this actually renders at, three sit
          close enough that the strokes and the knobs merge into one smudge. The
          knobs have to hide the rail behind them, so they are filled with the
          page's own white rather than left hollow. */}
      <circle cx="12.5" cy="7" r="1.9" className="fill-surface" stroke="currentColor" strokeWidth="1.4" />
      <circle cx="7.5" cy="13" r="1.9" className="fill-surface" stroke="currentColor" strokeWidth="1.4" />
    </svg>
  );
}

export function ChevronDownIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M5 7.5l5 5 5-5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function CheckIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M4.5 10.5l3.5 3.5 7.5-8" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function MicIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <rect x="7.25" y="3" width="5.5" height="9" rx="2.75" stroke="currentColor" strokeWidth="1.4" />
      <path d="M4.5 9.5a5.5 5.5 0 0011 0M10 15v2.5M7 17.5h6" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}

export function ScreenIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <rect x="3" y="4" width="14" height="9.5" rx="1.6" stroke="currentColor" strokeWidth="1.4" />
      <path d="M7.5 17.5h5M10 13.5v4" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}

export function BellIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path
        d="M6 8.5a4 4 0 018 0v3.2l1.3 2.3H4.7L6 11.7V8.5z"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinejoin="round"
      />
      <path d="M8.3 15.5a1.8 1.8 0 003.4 0" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}

export function DownloadIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path
        d="M10 3.5v9M6.5 9l3.5 3.5L13.5 9M4 16.5h12"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/** A trash can, for the hover-only delete affordance on a meeting row. Same
 * stroke weight as the rest of the set (1.6) so it doesn't read as timid next
 * to Search/Close/Plus. */
export function TrashIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path
        d="M4.25 6.75h11.5M8.1 6.75V5a1.25 1.25 0 011.25-1.25h1.3A1.25 1.25 0 0111.9 5v1.75"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M6.4 6.75l.68 9.1A1.5 1.5 0 008.58 17.3h2.84a1.5 1.5 0 001.5-1.45l.68-9.1"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path d="M8.6 9.4v5.2M11.4 9.4v5.2" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}

/** The Echo mark: a sound and its fading repetitions. Same drawing as the app
 * icon (src-tauri/icons/source/app-icon.svg), sized for chrome. */
export function EchoMark(props: IconProps) {
  return (
    <svg viewBox="0 0 44 44" fill="none" aria-hidden {...props}>
      <circle cx="13" cy="22" r="4.5" fill="currentColor" />
      <path
        d="M 19.43 14.34 A 10 10 0 0 1 19.43 29.66"
        stroke="currentColor"
        strokeWidth="4"
        strokeLinecap="round"
        opacity="0.55"
      />
      <path
        d="M 23.93 8.98 A 17 17 0 0 1 23.93 35.02"
        stroke="currentColor"
        strokeWidth="4"
        strokeLinecap="round"
        opacity="0.28"
      />
    </svg>
  );
}

/** Ollama's llama face, simplified to strokes so it sits with the rest of the
 * set. Shown wherever the Ollama connector is named. */
export function OllamaIcon(props: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden {...props}>
      <rect x="7" y="2.6" width="2.8" height="6" rx="1.4" stroke="currentColor" strokeWidth="1.5" />
      <rect x="14.2" y="2.6" width="2.8" height="6" rx="1.4" stroke="currentColor" strokeWidth="1.5" />
      <rect x="4.8" y="7" width="14.4" height="13" rx="6.2" stroke="currentColor" strokeWidth="1.5" />
      <circle cx="9.6" cy="13" r="1.15" fill="currentColor" />
      <circle cx="14.4" cy="13" r="1.15" fill="currentColor" />
    </svg>
  );
}

/** Google Gemini's four-point spark. Shown wherever the Gemini connector is
 * named. */
export function GeminiIcon(props: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden {...props}>
      <path
        d="M12 2c.55 5.5 4.5 9.45 10 10-5.5.55-9.45 4.5-10 10-.55-5.5-4.5-9.45-10-10 5.5-.55 9.45-4.5 10-10z"
        fill="currentColor"
      />
    </svg>
  );
}
