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
  // That window is see-through so the panel can float as a rounded card with
  // nothing behind it, while every other screen keeps its painted white
  // background. Stamping the attribute here — before React renders — lets
  // `index.css` scope the transparency to this one window in CSS, where the
  // rest of the app's theming already lives, instead of smuggling three
  // inline styles in from JavaScript.
  document.documentElement.dataset.window = "panel";
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
