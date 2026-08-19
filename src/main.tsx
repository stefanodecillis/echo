import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";

import App from "./App";
import { common } from "./lib/copy";
import { PanelApp } from "./panel/PanelApp";
import "./index.css";

const root = document.getElementById("root");
if (!root) {
  throw new Error(common.noWindowContents);
}

// Rust opens one extra window — the tiny always-on-top detected/recording
// control — loading this same app with `?window=panel`. It gets a different
// tree entirely: no router, no sidebar, just the one component the panel
// needs.
const isPanel = new URLSearchParams(window.location.search).get("window") === "panel";

if (isPanel) {
  // That window is transparent so the panel can float as a rounded card with
  // nothing behind it; `index.css` paints every window's `body` opaque
  // (`bg-surface`) for the normal case, so this overrides it here rather
  // than touching a file shared with every other screen. Inline styles beat
  // the class regardless of Tailwind's own specificity.
  document.documentElement.style.background = "transparent";
  document.body.style.background = "transparent";
}

ReactDOM.createRoot(root).render(
  <React.StrictMode>
    {isPanel ? (
      <PanelApp />
    ) : (
      <BrowserRouter>
        <App />
      </BrowserRouter>
    )}
  </React.StrictMode>,
);
