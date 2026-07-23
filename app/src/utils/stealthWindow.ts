// Compact mode (a stealth-gated preference, see `compactMode.ts`) runs Otto in
// borderless, non-activating, capture-excluded NSPanels (see
// `src-tauri/src/stealth.rs`). There are two such panels, each a separate
// webview:
//   • "overlay" — the Chat panel (compact composer + expandable history)
//   • "capture" — the Live Capture panel (audio transcription + screenshots)
// The normal app runs in the "main" window — including while stealth mode is
// on but compact is off. We detect which window we're in by the Tauri window
// label so each renders the right surface and skips main-window-only startup
// side effects.
//
// We read the label synchronously from the internals Tauri injects before app
// JS runs, so it's available at module-eval time and returns null in a plain
// browser (non-Tauri) dev context.

export type StealthKind = "chat" | "capture";

const LABEL_TO_KIND: Record<string, StealthKind> = {
  overlay: "chat",
  capture: "capture",
};

export const CHAT_WINDOW_LABEL = "overlay";
export const CAPTURE_WINDOW_LABEL = "capture";

function currentWindowLabel(): string | null {
  try {
    const internals = (window as unknown as {
      __TAURI_INTERNALS__?: { metadata?: { currentWindow?: { label?: string } } };
    }).__TAURI_INTERNALS__;
    return internals?.metadata?.currentWindow?.label ?? null;
  } catch {
    return null;
  }
}

/** Which stealth panel this webview is, or null for the main window / web dev. */
export function stealthWindowKind(): StealthKind | null {
  const label = currentWindowLabel();
  return label ? LABEL_TO_KIND[label] ?? null : null;
}

/** True in either stealth panel (chat or capture). */
export function isStealthWindow(): boolean {
  return stealthWindowKind() !== null;
}

export function isStealthChatWindow(): boolean {
  return stealthWindowKind() === "chat";
}

export function isStealthCaptureWindow(): boolean {
  return stealthWindowKind() === "capture";
}
