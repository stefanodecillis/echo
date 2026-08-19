import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";

import App from "./App";
import { common } from "./lib/copy";
import "./index.css";

const root = document.getElementById("root");
if (!root) {
  throw new Error(common.noWindowContents);
}

ReactDOM.createRoot(root).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>,
);
