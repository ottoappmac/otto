import { useCallback, useEffect, useRef, useState } from "react";
import { Outlet, useNavigate, useLocation } from "react-router-dom";
import Sidebar from "./Sidebar";
import TranscribeDrawer from "./transcribe/TranscribeDrawer";
import StealthTitlebar from "./stealth/StealthTitlebar";
import { ConnectionBanner } from "./ConnectionBanner";
import AmbientNotificationBanner from "./ambient/AmbientNotificationBanner";
import AmbientToast from "./ambient/AmbientToast";
import { useAmbientHints } from "../hooks/useAmbientHints";
import { useNotification } from "../context/NotificationContext";
import { onPendingRoute, nativeNotify } from "../utils/nativeNotify";
import { closeTranscribePanel, isTranscribePanelOpen, subscribeTranscribePanel } from "../utils/transcribePanel";
import { isStealthWindow } from "../utils/stealthWindow";
import type { AmbientHint } from "../types";

const STEALTH = isStealthWindow();

export default function Layout() {
  const navigate = useNavigate();
  const location = useLocation();

  const pathnameRef = useRef(location.pathname);
  pathnameRef.current = location.pathname;

  // Deep-link routing for notification clicks. Bubbles fire outside the Router
  // tree, so they emit through a module-level bus that we drain into navigate().
  useEffect(() => onPendingRoute((path) => navigate(path)), [navigate]);

  // ── Keyboard shortcut: ⌘/Ctrl+N → new chat ──────────────────────────────
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const isNewChat =
        (e.metaKey || e.ctrlKey) &&
        !e.shiftKey &&
        !e.altKey &&
        e.key.toLowerCase() === "n";
      if (!isNewChat) return;
      e.preventDefault();
      localStorage.removeItem("chatDraft");
      navigate("/chat");
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [navigate]);

  // ── Ambient notifications ────────────────────────────────────────────────
  // bannerHints: hints the user hasn't acknowledged yet (shown in banner).
  // toastHints:  same set, shown as a bottom-right slide-up toast.
  const [bannerHints, setBannerHints] = useState<AmbientHint[]>([]);
  const [showToast, setShowToast] = useState(false);
  const [toastCount, setToastCount] = useState(0);

  // Live-transcription side panel — mounted once here (beside the Outlet, next
  // to whatever page is showing, typically Chat) so its WebSocket connection
  // and in-progress recording persist across route changes. Open/close is
  // driven by the Sidebar's "Transcribe" nav item via a shared module store.
  const [showTranscribe, setShowTranscribe] = useState(isTranscribePanelOpen);
  useEffect(() => subscribeTranscribePanel(setShowTranscribe), []);


  const { pendingCount: _pendingCount, markSeen } = useAmbientHints(
    useCallback(
      (count: number, newHints: AmbientHint[]) => {
        // Don't pop a notification when the user is already on the ambient page.
        if (pathnameRef.current === "/ambient") {
          markSeenRef.current?.();
          return;
        }

        // Top banner.
        setBannerHints(newHints);

        // Bottom-right toast.
        setToastCount(count);
        setShowToast(true);

        // OS notification when window is not focused (native + clickable).
        if (!document.hasFocus()) {
          void nativeNotify({
            title: "Ambient Suggestion",
            body:
              count === 1
                ? "Otto has a new suggestion for you."
                : `Otto has ${count} new suggestions for you.`,
            deepLink: "/ambient",
          });
        }
      },
      // eslint-disable-next-line react-hooks/exhaustive-deps
      [],
    ),
  );

  // Stable ref so the callback closure stays stable (empty deps above).
  const markSeenRef = useRef<(() => void) | null>(null);
  useEffect(() => { markSeenRef.current = markSeen; }, [markSeen]);

  // Keep the context subscribed so badge/title updates still fire.
  useNotification();

  // Clear banner + mark hints seen when the user navigates to /ambient.
  useEffect(() => {
    if (location.pathname === "/ambient") {
      setBannerHints([]);
      setShowToast(false);
      markSeen();
    }
  }, [location.pathname, markSeen]);

  const handleViewAmbient = () => {
    setBannerHints([]);
    setShowToast(false);
    navigate("/ambient");
  };

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-th-bg">
      {STEALTH && <StealthTitlebar kind="chat" />}
      <div className="flex flex-1 min-h-0 overflow-hidden">
      <main className="flex-1 flex flex-col overflow-hidden bg-th-bg-secondary">
        <ConnectionBanner />
        {!STEALTH && bannerHints.length > 0 && (
          <AmbientNotificationBanner
            hints={bannerHints}
            onView={handleViewAmbient}
            onDismiss={() => setBannerHints([])}
          />
        )}
        <div className="flex-1 overflow-auto">
          <Outlet />
        </div>
      </main>

      {/* Non-modal transcription panel — sits in-flow so chat stays usable.
          In stealth mode Live Capture is its own separate panel/window, so this
          docked drawer is only for the normal (main) window. */}
      {!STEALTH && <TranscribeDrawer open={showTranscribe} onClose={closeTranscribePanel} />}

      {!STEALTH && <Sidebar />}

      {/* Bottom-right slide-up toast — rendered outside page columns so it's never clipped */}
      {!STEALTH && showToast && (
        <AmbientToast
          count={toastCount}
          onView={handleViewAmbient}
          onDismiss={() => setShowToast(false)}
        />
      )}
      </div>
    </div>
  );
}
