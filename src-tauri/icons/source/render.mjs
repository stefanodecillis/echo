// Renders the SVG sources into the PNGs the app ships. Run from this folder:
//   /opt/homebrew/bin/node render.mjs
// Needs @resvg/resvg-js (install anywhere on NODE_PATH or alongside this file).
import { createRequire } from "node:module";
import { readFileSync, writeFileSync } from "node:fs";

// Resolve @resvg/resvg-js from wherever it is installed: alongside this file,
// or in the directory named by RESVG_DIR.
const requireFrom = createRequire(
  (process.env.RESVG_DIR ?? import.meta.dirname) + "/noop.js",
);
const { Resvg } = requireFrom("@resvg/resvg-js");

function render(svg, size, out) {
  const png = new Resvg(svg, {
    fitTo: { mode: "width", value: size },
  }).render();
  writeFileSync(out, png.asPng());
  console.log(`${out} ${size}x${size}`);
}

// The mark, tray-sized. Template icons: pure black + alpha, macOS recolors.
// Three states the tray actually uses: waiting, meeting detected, recording.
function traySvg({ dotR, arcAlpha, badge }) {
  const arcs = `
    <path d="M ${13 + 6.43} ${22 - 7.66} A 10 10 0 0 1 ${13 + 6.43} ${22 + 7.66}"
          fill="none" stroke="#000" stroke-width="4" stroke-linecap="round" opacity="${arcAlpha[0]}"/>
    <path d="M ${13 + 10.93} ${22 - 13.02} A 17 17 0 0 1 ${13 + 10.93} ${22 + 13.02}"
          fill="none" stroke="#000" stroke-width="4" stroke-linecap="round" opacity="${arcAlpha[1]}"/>`;
  const badgeDot = badge ? `<circle cx="36" cy="8" r="5.5" fill="#000"/>` : "";
  return `<svg xmlns="http://www.w3.org/2000/svg" width="44" height="44" viewBox="0 0 44 44">
    <circle cx="13" cy="22" r="${dotR}" fill="#000"/>${arcs}${badgeDot}</svg>`;
}

const appIcon = readFileSync("app-icon.svg", "utf8");
render(appIcon, 1024, "icon-1024.png");

render(traySvg({ dotR: 4.5, arcAlpha: [0.55, 0.28], badge: false }), 44, "../tray-idle.png");
render(traySvg({ dotR: 4.5, arcAlpha: [1, 1], badge: true }), 44, "../tray-detected.png");
render(traySvg({ dotR: 7, arcAlpha: [1, 1], badge: false }), 44, "../tray-recording.png");

// Recording pulse animation: the dot breathes 7 -> 5.5 -> 4.5 -> 5.5 -> (loop
// back to 7), read at ~2fps by the tray. Arcs stay fully opaque throughout —
// recording means a strong signal, no fading.
const pulseRadii = [7, 5.5, 4.5, 5.5];
pulseRadii.forEach((dotR, i) => {
  render(
    traySvg({ dotR, arcAlpha: [1, 1], badge: false }),
    44,
    `../tray-recording-${i}.png`,
  );
});

// The working state: the meeting is over and Echo is still finishing it off.
//
// Deliberately a different motif from the recording pulse rather than a
// variant of it. Both animate at the same 2fps in the same corner of the same
// menu bar, so if "working" were the same mark breathing differently, the one
// question the icon exists to answer — is this thing still listening to me? —
// would need a second look. A dot with something going round it cannot be
// mistaken for a dot with arcs beside it.
//
// Echo's dot stays, small (it is the app's mark, not a recording light), and a
// single arc travels a faint track around it, a quarter turn per frame.
const SPIN_CENTRE = 22;
const SPIN_RADIUS = 13;
// How much of the circle is lit at once. Long enough to read as an arc rather
// than a tick, short enough to leave the track visible behind it.
const SPIN_SWEEP = 110;

function trayWorkingSvg(step, steps) {
  // 0° is twelve o'clock, and the arc runs clockwise from there.
  const point = (degrees) => {
    const radians = ((degrees - 90) * Math.PI) / 180;
    return [
      (SPIN_CENTRE + SPIN_RADIUS * Math.cos(radians)).toFixed(2),
      (SPIN_CENTRE + SPIN_RADIUS * Math.sin(radians)).toFixed(2),
    ];
  };
  const from = (step * 360) / steps;
  const [x0, y0] = point(from);
  const [x1, y1] = point(from + SPIN_SWEEP);
  const large = SPIN_SWEEP > 180 ? 1 : 0;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="44" height="44" viewBox="0 0 44 44">
    <circle cx="${SPIN_CENTRE}" cy="${SPIN_CENTRE}" r="${SPIN_RADIUS}"
            fill="none" stroke="#000" stroke-width="4" opacity="0.22"/>
    <path d="M ${x0} ${y0} A ${SPIN_RADIUS} ${SPIN_RADIUS} 0 ${large} 1 ${x1} ${y1}"
          fill="none" stroke="#000" stroke-width="4" stroke-linecap="round"/>
    <circle cx="${SPIN_CENTRE}" cy="${SPIN_CENTRE}" r="4" fill="#000"/></svg>`;
}

// Frame 0 doubles as the still icon, the same way tray-recording.png is the
// first frame of the pulse: whatever paints the state and whatever paints the
// frames then agree, so a repaint mid-animation cannot make the icon jump.
const workingSteps = 4;
for (let i = 0; i < workingSteps; i += 1) {
  render(trayWorkingSvg(i, workingSteps), 44, `../tray-processing-${i}.png`);
}
render(trayWorkingSvg(0, workingSteps), 44, "../tray-processing.png");
