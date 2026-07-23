// Controls Otto's "compact" overlay UI (via the native `set_compact_mode`
// Tauri command) — a separate, stealth-gated preference from
// `screenShareVisibility.ts`'s "hide from screen share" (stealth mode).
//
// Stealth mode alone keeps Otto's normal-sized window around (just excluded
// from screen capture and off the Dock/menu bar). Compact additionally swaps
// that window for two small, transparent, non-activating panels — Chat and
// Live Capture — that float on top and can be moved independently.
//
// Compact only makes sense while stealth is on, so the Settings UI only
// offers this toggle once stealth is enabled, and turning stealth off also
// turns compact off (see `SettingsPage.tsx` / `StealthTitlebar.tsx`).

const STORAGE_KEY = "otto.compactMode";

/**
 * Broadcast whenever compact mode is toggled, so every window (main + both
 * stealth panels) stays in sync no matter where the change originated —
 * mirrors `STEALTH_CHANGED_EVENT` in `screenShareVisibility.ts`.
 */
export const COMPACT_CHANGED_EVENT = "otto://compact-changed";

export function getCompactMode(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}

/** Push the current value to the native window (no-op outside Tauri). */
export async function applyCompactMode(compact: boolean): Promise<void> {
  const { invoke } = await import("@tauri-apps/api/core");
  await invoke("set_compact_mode", { compact });
}

/** Persist the preference and apply it to the native window. */
export async function setCompactMode(compact: boolean): Promise<void> {
  try {
    localStorage.setItem(STORAGE_KEY, compact ? "1" : "0");
  } catch {
    // ignore storage failures — still apply for this session
  }
  await applyCompactMode(compact);
  try {
    const { emit } = await import("@tauri-apps/api/event");
    await emit(COMPACT_CHANGED_EVENT, compact);
  } catch {
    // outside Tauri or event bus unavailable — nothing else to notify
  }
}

/**
 * Subscribe to compact on/off changes from any window. Returns a cleanup
 * function. No-op (returns a noop cleanup) outside Tauri.
 */
export function onCompactModeChanged(
  handler: (compact: boolean) => void,
): () => void {
  let unlisten: (() => void) | undefined;
  let cancelled = false;
  import("@tauri-apps/api/event")
    .then(({ listen }) =>
      listen<boolean>(COMPACT_CHANGED_EVENT, (event) => handler(event.payload)),
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
