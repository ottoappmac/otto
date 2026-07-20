import { useCallback, useEffect } from "react";
import TranscribeDrawer from "../transcribe/TranscribeDrawer";
import StealthTitlebar from "./StealthTitlebar";
import { initCaptureWindowBridge } from "../../utils/askOttoBridge";

/**
 * The Live Capture stealth panel — audio transcription + screenshots in its own
 * borderless, movable, capture-excluded window. Renders the shared
 * `TranscribeDrawer` in `standalone` mode (fills the window; no docked-drawer
 * chrome) beneath a draggable stealth title bar.
 *
 * "Ask Otto" here hands the captured context to the separate Chat panel over the
 * cross-window bridge; agent-busy is bridged back so auto-send pauses while Otto
 * is working.
 */
export default function StealthCaptureWindow() {
  useEffect(() => initCaptureWindowBridge(), []);

  // Order this panel out — kept alive for a fast re-show via the hotkey or the
  // Chat panel's "Show Live Capture" button.
  const hideSelf = useCallback(() => {
    import("@tauri-apps/api/window")
      .then(({ getCurrentWindow }) => getCurrentWindow().hide())
      .catch(() => {});
  }, []);

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-th-bg">
      <StealthTitlebar kind="capture" />
      <div className="flex-1 min-h-0 overflow-hidden">
        <TranscribeDrawer open standalone onClose={hideSelf} />
      </div>
    </div>
  );
}
