import { useEffect, useState } from "react";
import { Routes, Route, Navigate, useNavigate } from "react-router-dom";
import Layout from "./components/Layout";
import ChatPage from "./pages/ChatPage";
import SettingsPage from "./pages/SettingsPage";
import AgentsPage from "./pages/AgentsPage";
import DashboardPage from "./pages/DashboardPage";
import SchedulesPage from "./pages/SchedulesPage";
import ScheduleRunsPage from "./pages/ScheduleRunsPage";
import TriggersPage from "./pages/TriggersPage";
import TriggerRunsPage from "./pages/TriggerRunsPage";
import ActivityPage from "./pages/ActivityPage";
import AmbientInbox from "./components/ambient/AmbientInbox";
import RunsPage from "./pages/RunsPage";
import RunDetailPage from "./pages/RunDetailPage";
import SetupWizard from "./components/setup/SetupWizard";
import SetupChatPage from "./components/setup/SetupChatPage";
import StealthCaptureWindow from "./components/stealth/StealthCaptureWindow";
import { useBackendReady } from "./hooks/useBackendReady";
import { useFirstRun } from "./hooks/useFirstRun";
import { ConnectionProvider } from "./context/ConnectionContext";
import { api } from "./hooks/useApi";
import { stealthWindowKind } from "./utils/stealthWindow";
import { initChatWindowBridge } from "./utils/askOttoBridge";
import type { MlxCapabilities } from "./types";

const STEALTH_KIND = stealthWindowKind();
const STEALTH = STEALTH_KIND !== null;

export default function App() {
  const { ready, timedOut, backendReachable, elapsedSec } = useBackendReady();
  const firstRun = useFirstRun();
  const navigate = useNavigate();
  const [capabilities, setCapabilities] = useState<MlxCapabilities | null>(null);
  const [capabilitiesLoaded, setCapabilitiesLoaded] = useState(false);
  // When true, user chose "Set up manually" from SetupChatPage
  const [useLegacySetup, setUseLegacySetup] = useState(false);

  // Listen for tray → navigate events (e.g. clicking "Suggestions" in the tray menu).
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    import("@tauri-apps/api/event")
      .then(({ listen }) => listen<string>("navigate", (event) => navigate(event.payload)))
      .then((fn) => { unlisten = fn; })
      .catch(() => {});
    return () => unlisten?.();
  }, [navigate]);

  // Re-apply the saved stealth preference on startup so the native window state
  // matches the user's choice. Only the main window drives this — the stealth
  // panels are consumers of that state, not a source.
  useEffect(() => {
    if (STEALTH) return;
    import("./utils/screenShareVisibility")
      .then(({ getHideFromScreenShare, applyHideFromScreenShare }) => {
        if (getHideFromScreenShare()) return applyHideFromScreenShare(true);
      })
      .catch(() => {}); // no-op in non-Tauri environments (e.g. web dev)
  }, []);

  // Chat panel only: bridge cross-window hand-offs from the Live Capture panel
  // (which lives in a separate webview) into the local chat bus + agent-busy.
  useEffect(() => {
    if (STEALTH_KIND !== "chat") return;
    return initChatWindowBridge();
  }, []);

  // Route external link clicks to the system browser. Without this, clicking a
  // URL in run results / chat would navigate the main Tauri webview away from
  // the React app (unloading the whole UI) with no way to get back. Same-origin
  // links are left alone so react-router and local assets keep working.
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (e.defaultPrevented || e.button !== 0) return;
      const anchor = (e.target as HTMLElement | null)?.closest("a");
      const href = anchor?.getAttribute("href");
      if (!href) return;
      let url: URL;
      try {
        url = new URL(href, window.location.href);
      } catch {
        return;
      }
      if (url.protocol !== "http:" && url.protocol !== "https:") return;
      if (url.origin === window.location.origin) return; // in-app navigation
      e.preventDefault();
      import("@tauri-apps/plugin-shell")
        .then(({ open }) => open(url.href))
        .catch(() => window.open(url.href, "_blank", "noopener,noreferrer"));
    };
    document.addEventListener("click", handler, true);
    return () => document.removeEventListener("click", handler, true);
  }, []);

  useEffect(() => {
    if (ready) {
      api
        .mlxCapabilities()
        .then((c) => {
          setCapabilities(c);
          setCapabilitiesLoaded(true);
        })
        .catch(() => {
          // On failure treat as ineligible — legacy wizard is the safe fallback
          setCapabilities(null);
          setCapabilitiesLoaded(true);
        });
    }
  }, [ready]);

  // A Mac is eligible for the chat setup when it has Apple Silicon and
  // enough RAM for the 1.1 GB Qwen3-1.7B model alongside the app.
  const eligibleForChatSetup =
    !useLegacySetup &&
    capabilities != null &&
    capabilities.apple_silicon &&
    capabilities.ram_gb >= 8;

  if (!ready) {
    return (
      <div className="flex h-screen items-center justify-center bg-th-bg">
        <div className="text-center">
          {timedOut ? (
            <>
              <p className="text-red-500 text-lg font-medium">
                Unable to connect to the backend
              </p>
              <p className="text-th-text-tertiary text-sm mt-2">
                Please check that the server is running on port 18081
              </p>
              <button
                className="mt-4 px-4 py-2 rounded bg-th-surface-hover text-th-text-primary hover:opacity-90 text-sm border border-th-border"
                onClick={() => window.location.reload()}
              >
                Retry
              </button>
            </>
          ) : (
            <>
              <div className="inline-block w-8 h-8 border-2 border-neutral-200 border-t-neutral-500 rounded-full animate-spin mb-4" />
              <p className="text-th-text-secondary text-sm">
                Starting up&hellip;
              </p>
              {elapsedSec >= 5 && (
                <p className="text-th-text-muted text-xs mt-2">
                  {elapsedSec}s — first launch may take a minute while the application scans files
                </p>
              )}
            </>
          )}
        </div>
      </div>
    );
  }

  // Wait for both the first-run probe and the capabilities fetch before
  // rendering any first-run UI.  Both calls are fast (< 200 ms) and
  // blocking here prevents a flash of the wrong wizard variant.
  if (!firstRun.loaded || (firstRun.show && !capabilitiesLoaded && !useLegacySetup)) {
    return (
      <div className="flex h-screen items-center justify-center bg-th-bg">
        <div className="inline-block w-8 h-8 border-2 border-th-border border-t-th-text-secondary rounded-full animate-spin" />
      </div>
    );
  }

  if (firstRun.show && !STEALTH) {
    if (eligibleForChatSetup && capabilities != null) {
      return (
        <SetupChatPage
          capabilities={capabilities}
          onFinish={firstRun.hide}
          onUseLegacy={() => setUseLegacySetup(true)}
          onSkip={firstRun.hide}
        />
      );
    }
    return (
      <SetupWizard
        initialStep={firstRun.currentStep}
        onFinish={firstRun.hide}
        onSkip={firstRun.hide}
        onNavigate={(path) => { firstRun.hide(); navigate(path); }}
      />
    );
  }

  // The Live Capture stealth panel is a dedicated, single-purpose window — it
  // renders only the capture surface (no router / chat page), and hands captured
  // context to the separate Chat panel over the cross-window bridge.
  if (STEALTH_KIND === "capture") {
    return (
      <ConnectionProvider backendReachable={backendReachable}>
        <StealthCaptureWindow />
      </ConnectionProvider>
    );
  }

  return (
    <ConnectionProvider backendReachable={backendReachable}>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Navigate to="/chat" replace />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/chat/:sessionId" element={<ChatPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/tools" element={<Navigate to="/agents" replace />} />
          <Route path="/agents" element={<AgentsPage />} />
          <Route path="/skills" element={<Navigate to="/agents" replace />} />
          <Route path="/schedules" element={<SchedulesPage />} />
          <Route path="/schedules/:id/runs" element={<ScheduleRunsPage />} />
          <Route path="/triggers" element={<TriggersPage />} />
          <Route path="/triggers/:id/runs" element={<TriggerRunsPage />} />
          <Route path="/memory" element={<Navigate to="/settings?tab=Memory" replace />} />
          <Route
            path="/exo"
            element={<Navigate to="/settings?tab=LLM&sub=Cluster" replace />}
          />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/runs" element={<RunsPage />} />
          <Route path="/runs/:id" element={<RunDetailPage />} />
          <Route path="/history" element={<Navigate to="/runs" replace />} />
          <Route path="/activity" element={<ActivityPage />} />
          <Route path="/ambient" element={<AmbientInbox />} />
        </Route>
      </Routes>
    </ConnectionProvider>
  );
}
