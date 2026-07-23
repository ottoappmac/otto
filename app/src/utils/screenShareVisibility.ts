// Controls Otto's anti-proctoring "stealth mode" (via the native
// `set_hidden_from_capture` Tauri command). When enabled this:
//   1. Excludes Otto's windows from every screen-capture path — the legacy
//      NSWindow.sharingType flag plus the private CGSSetWindowCaptureExcludeShape
//      call that also defeats ScreenCaptureKit / browser getDisplayMedia (what
//      CoderPad's in-browser proctoring uses).
//   2. Hides Otto's menu bar + Dock icons and keeps its (still normal-sized)
//      window floating above others.
// The preference is persisted locally and re-applied on startup.
//
// Stealth is intentionally decoupled from "compact" (`compactMode.ts`), the
// small transparent overlay-panel UI — compact is a separate, stealth-gated
// preference that's only offered once stealth is on, and is always turned off
// alongside stealth here.
//
// Caveat: capture exclusion relies on a private macOS API. Apple's own
// QuickTime and blessed conferencing partners can still capture the window;
// the browser capture path CoderPad relies on cannot.

const STORAGE_KEY = "otto.hideFromScreenShare";

/**
 * Broadcast whenever stealth mode is toggled. Stealth runs in a separate
 * webview (the "overlay" panel) from the main window, so each window keeps its
 * own React state and localStorage writes in one don't re-render the other.
 * We emit this Tauri event on change and re-broadcast it to every window so the
 * Settings toggle stays in sync no matter where the change originated.
 */
export const STEALTH_CHANGED_EVENT = "otto://stealth-changed";

export function getHideFromScreenShare(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}

/** Push the current value to the native window (no-op outside Tauri). */
export async function applyHideFromScreenShare(hidden: boolean): Promise<void> {
  const { invoke } = await import("@tauri-apps/api/core");
  await invoke("set_hidden_from_capture", { hidden });
}

/** Persist the preference and apply it to the native window. */
export async function setHideFromScreenShare(hidden: boolean): Promise<void> {
  try {
    localStorage.setItem(STORAGE_KEY, hidden ? "1" : "0");
  } catch {
    // ignore storage failures — still apply for this session
  }
  await applyHideFromScreenShare(hidden);
  // Compact requires stealth, so turning stealth off always turns compact
  // off too — regardless of where compact's own preference currently stands.
  if (!hidden) {
    try {
      const { setCompactMode } = await import("./compactMode");
      await setCompactMode(false);
    } catch {
      // outside Tauri — nothing to reconcile
    }
  }
  try {
    const { emit } = await import("@tauri-apps/api/event");
    await emit(STEALTH_CHANGED_EVENT, hidden);
  } catch {
    // outside Tauri or event bus unavailable — nothing else to notify
  }
}

/**
 * Subscribe to stealth on/off changes from any window. Returns a cleanup
 * function. No-op (returns a noop cleanup) outside Tauri.
 */
export function onHideFromScreenShareChanged(
  handler: (hidden: boolean) => void,
): () => void {
  let unlisten: (() => void) | undefined;
  let cancelled = false;
  import("@tauri-apps/api/event")
    .then(({ listen }) =>
      listen<boolean>(STEALTH_CHANGED_EVENT, (event) => handler(event.payload)),
    )
    .then((fn) => {
      if (cancelled) fn();
      else unlisten = fn;
    })
    .catch(() => {});
  return () => {
    cancelled = true;
    unlisten?.();
  };
}
