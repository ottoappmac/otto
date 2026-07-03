import { getCurrentWindow } from "@tauri-apps/api/window";

export type BadgeLevel = "done" | "hitl" | "error";

const COLORS: Record<BadgeLevel, string> = {
  done: "#10b981",   // emerald-500 — success
  hitl: "#fbbf24",   // amber-400  — needs user input
  error: "#ef4444",  // red-500    — failure
};

/**
 * Set the app badge to a specific unread `count` with an optional top-severity
 * `level` driving the colored overlay dot. Pass `count <= 0` (or no level) to
 * clear the badge. This is the single source of truth for the badge — it is
 * driven by the Notification Center's unread count, so the badge naturally
 * clears as items are read rather than on window focus.
 */
export async function setBadge(count: number, level?: BadgeLevel | null): Promise<void> {
  const showCount = count > 0;

  try {
    const win = getCurrentWindow();
    await win.setBadgeCount(showCount ? count : undefined);
  } catch (e) {
    console.error("[appBadge] setBadgeCount failed:", e);
  }

  try {
    const win = getCurrentWindow();
    if (showCount && level) {
      const icon = await getDotIcon(COLORS[level]);
      await win.setOverlayIcon(icon);
    } else {
      await win.setOverlayIcon(undefined);
    }
  } catch (e) {
    console.error("[appBadge] setOverlayIcon failed:", e);
  }
}

const DOT_SIZE = 32;
const iconCache = new Map<string, Uint8Array>();

async function getDotIcon(color: string) {
  const { Image } = await import("@tauri-apps/api/image");

  let bytes = iconCache.get(color);
  if (!bytes) {
    const canvas = document.createElement("canvas");
    canvas.width = DOT_SIZE;
    canvas.height = DOT_SIZE;
    const ctx = canvas.getContext("2d")!;

    ctx.beginPath();
    ctx.arc(DOT_SIZE / 2, DOT_SIZE / 2, DOT_SIZE / 2, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();

    const blob = await new Promise<Blob>((resolve) =>
      canvas.toBlob((b) => resolve(b!), "image/png"),
    );
    bytes = new Uint8Array(await blob.arrayBuffer());
    iconCache.set(color, bytes);
  }

  return Image.fromBytes(bytes);
}
