// Adjustable fade level for stealth mode's focus-driven transparency — used by
// both the main window's fade (`.stealth-active` rules in `index.css`, wired
// up in `App.tsx`) and the compact overlay panels' fade (`.stealth-window`
// rules, wired up in `StealthTitlebar.tsx`). Applied purely via a CSS custom
// property (`--stealth-fade-opacity`) that every window sets for itself on
// load and keeps in sync via a broadcast event — no native call needed since
// this never touches the actual OS window, just the page's own CSS.

const STORAGE_KEY = "otto.stealthFadeOpacity";
const CSS_VAR = "--stealth-fade-opacity";

export const DEFAULT_STEALTH_FADE_OPACITY = 0.7;
export const MIN_STEALTH_FADE_OPACITY = 0.2;
export const MAX_STEALTH_FADE_OPACITY = 1;

/** Broadcast whenever the fade level changes, so every window (main + both
 * stealth panels) stays in sync no matter where the change originated —
 * mirrors `STEALTH_CHANGED_EVENT` in `screenShareVisibility.ts`. */
export const STEALTH_FADE_OPACITY_CHANGED_EVENT = "otto://stealth-fade-opacity-changed";

function clamp(opacity: number): number {
  if (!Number.isFinite(opacity)) return DEFAULT_STEALTH_FADE_OPACITY;
  return Math.min(MAX_STEALTH_FADE_OPACITY, Math.max(MIN_STEALTH_FADE_OPACITY, opacity));
}

export function getStealthFadeOpacity(): number {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw == null ? DEFAULT_STEALTH_FADE_OPACITY : clamp(Number(raw));
  } catch {
    return DEFAULT_STEALTH_FADE_OPACITY;
  }
}

/** Apply to this window's document only — pure CSS, no native call. */
export function applyStealthFadeOpacity(opacity: number): void {
  document.documentElement.style.setProperty(CSS_VAR, String(clamp(opacity)));
}

/** Persist the preference, apply it here, and broadcast to other windows. */
export async function setStealthFadeOpacity(opacity: number): Promise<void> {
  const clamped = clamp(opacity);
  try {
    localStorage.setItem(STORAGE_KEY, String(clamped));
  } catch {
    // ignore storage failures — still apply for this session
  }
  applyStealthFadeOpacity(clamped);
  try {
    const { emit } = await import("@tauri-apps/api/event");
    await emit(STEALTH_FADE_OPACITY_CHANGED_EVENT, clamped);
  } catch {
    // outside Tauri or event bus unavailable — nothing else to notify
  }
}

/**
 * Subscribe to fade-level changes from any window. Returns a cleanup
 * function. No-op (returns a noop cleanup) outside Tauri.
 */
export function onStealthFadeOpacityChanged(
  handler: (opacity: number) => void,
): () => void {
  let unlisten: (() => void) | undefined;
  let cancelled = false;
  import("@tauri-apps/api/event")
    .then(({ listen }) =>
      listen<number>(STEALTH_FADE_OPACITY_CHANGED_EVENT, (event) => handler(event.payload)),
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
