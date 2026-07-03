/**
 * nativeNotify — desktop notification helper.
 *
 * Prefers the native Tauri notification plugin (clickable, OS-integrated) and
 * gracefully falls back to the Web `Notification` API in dev / non-Tauri
 * contexts. Notifications fire outside the React Router tree, so click-to-route
 * is delivered through a tiny module-level event bus (`onPendingRoute`) that
 * `Layout` subscribes to and feeds into `navigate(...)`.
 */
import { getCurrentWindow } from "@tauri-apps/api/window";

type RouteListener = (path: string) => void;

let routeListener: RouteListener | null = null;
let pendingRoute: string | null = null;

/**
 * Subscribe to deep-link routes emitted by notification clicks. Any route that
 * arrived before a listener was registered is flushed immediately. Returns an
 * unsubscribe function.
 */
export function onPendingRoute(cb: RouteListener): () => void {
  routeListener = cb;
  if (pendingRoute) {
    const p = pendingRoute;
    pendingRoute = null;
    cb(p);
  }
  return () => {
    if (routeListener === cb) routeListener = null;
  };
}

function emitRoute(path: string) {
  if (routeListener) routeListener(path);
  else pendingRoute = path;
}

function isTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

async function focusWindow() {
  if (!isTauri()) return;
  try {
    const win = getCurrentWindow();
    // The window only hides (never closes) per src-tauri/src/lib.rs, so showing
    // + focusing reliably brings it back to the foreground.
    await win.show();
    await win.setFocus();
  } catch (e) {
    console.error("[nativeNotify] focus failed:", e);
  }
}

/** Focus the window and route to `deepLink` — shared by native + web clicks. */
export async function handleNotificationClick(deepLink?: string): Promise<void> {
  await focusWindow();
  if (deepLink) emitRoute(deepLink);
}

let tauriActionRegistered = false;

async function ensureTauriActionListener() {
  if (tauriActionRegistered) return;
  tauriActionRegistered = true;
  try {
    const { onAction } = await import("@tauri-apps/plugin-notification");
    await onAction((notification) => {
      const extra = notification.extra as Record<string, unknown> | undefined;
      const deepLink = extra?.deepLink;
      void handleNotificationClick(typeof deepLink === "string" ? deepLink : undefined);
    });
  } catch {
    // Plugin unavailable (dev / web) — click routing falls back to web below.
  }
}

export interface NativeNotifyOptions {
  title: string;
  body?: string;
  /** Route navigated to when the bubble is clicked (best-effort on desktop). */
  deepLink?: string;
}

/** Fire a desktop notification, requesting permission once if needed. */
export async function nativeNotify({ title, body, deepLink }: NativeNotifyOptions): Promise<void> {
  if (isTauri()) {
    try {
      const plugin = await import("@tauri-apps/plugin-notification");
      let granted = await plugin.isPermissionGranted();
      if (!granted) {
        granted = (await plugin.requestPermission()) === "granted";
      }
      if (!granted) return;
      await ensureTauriActionListener();
      plugin.sendNotification({
        title,
        body,
        extra: deepLink ? { deepLink } : undefined,
      });
      return;
    } catch (e) {
      console.error("[nativeNotify] native notify failed, falling back to web:", e);
    }
  }

  if (!("Notification" in window)) return;
  const fire = () => {
    const n = new Notification(title, { body });
    n.onclick = () => {
      window.focus();
      if (deepLink) emitRoute(deepLink);
      n.close();
    };
  };
  if (Notification.permission === "granted") {
    fire();
  } else if (Notification.permission !== "denied") {
    Notification.requestPermission().then((p) => {
      if (p === "granted") fire();
    });
  }
}
