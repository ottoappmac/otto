// Controls whether Otto's window is excluded from screen capture / screen
// sharing (via the native `set_hidden_from_capture` Tauri command, which flips
// the macOS NSWindow.sharingType). The preference is persisted locally and
// re-applied on startup.
//
// Caveat: on macOS 15+ this only hides Otto from CoreGraphics-based capturers
// (notably Google Meet in Chrome). ScreenCaptureKit consumers (Zoom, Teams,
// QuickTime, system screenshots) will still see the window.

const STORAGE_KEY = "otto.hideFromScreenShare";

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
}
