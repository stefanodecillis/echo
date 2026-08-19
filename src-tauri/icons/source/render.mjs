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
