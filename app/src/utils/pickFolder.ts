/**
 * Pick a directory and return its absolute path.
 *
 * The desktop app uses the native folder dialog when the plugin is
 * installed. A browser preview, or a build that has not been restarted
 * since the plugin was added, returns null so the caller can fall back
 * to the file input.
 */
export function inTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

export async function pickDirectory(): Promise<string | null> {
  if (!inTauri()) return null;
  try {
    const mod = await import("@tauri-apps/plugin-dialog");
    const selected = await mod.open({ directory: true, multiple: false, title: "Allow a folder" });
    if (typeof selected === "string" && selected) return selected;
    return null;
  } catch {
    return null;
  }
}
