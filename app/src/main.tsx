import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { NotificationProvider } from "./context/NotificationContext";
import { ThemeProvider } from "./context/ThemeContext";
import { applyFontScaleToDocument, readStoredFontScale } from "./theme";
import { stealthWindowKind } from "./utils/stealthWindow";
import "@fontsource-variable/inter";
import "@fontsource-variable/source-serif-4";
import "./index.css";

// Apply stored font scale before first paint to avoid flash
applyFontScaleToDocument(readStoredFontScale());

// Stealth mode runs Otto across two borderless, transparent, non-activating
// panels — a Chat panel and a Live Capture panel. Tag the document so each
// reads as a floating macOS panel (rounded corners, transparent background) and
// so per-panel styling can differ.
const stealthKind = stealthWindowKind();
if (stealthKind) {
  document.documentElement.classList.add("stealth-window", `stealth-${stealthKind}`);
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ErrorBoundary>
      <ThemeProvider>
        <BrowserRouter>
          <NotificationProvider>
            <App />
          </NotificationProvider>
        </BrowserRouter>
      </ThemeProvider>
    </ErrorBoundary>
  </React.StrictMode>,
);
