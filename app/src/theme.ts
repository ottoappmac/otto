/** UI theme: persisted in localStorage, applied as `html[data-theme]`. */

export const THEME_STORAGE_KEY = "otto-theme";

export type ThemeMode = "light" | "dark";

// ---------------------------------------------------------------------------
// Font size
// ---------------------------------------------------------------------------

export const FONT_SIZE_STORAGE_KEY = "otto-font-size";

/** Steps exposed in the UI; numeric value is the CSS multiplier applied to 16px. */
export const FONT_SIZE_STEPS = [0.8125, 0.875, 0.9375, 1, 1.0625, 1.125, 1.25] as const;
export type FontSizeStep = (typeof FONT_SIZE_STEPS)[number];

export const FONT_SCALE_DEFAULT: FontSizeStep = 1;
export const FONT_SCALE_MIN = FONT_SIZE_STEPS[0];
export const FONT_SCALE_MAX = FONT_SIZE_STEPS[FONT_SIZE_STEPS.length - 1];

export function readStoredFontScale(): FontSizeStep {
  try {
    const raw = localStorage.getItem(FONT_SIZE_STORAGE_KEY);
    if (raw !== null) {
      const n = parseFloat(raw);
      if ((FONT_SIZE_STEPS as readonly number[]).includes(n)) return n as FontSizeStep;
    }
  } catch { /* ignore */ }
  return FONT_SCALE_DEFAULT;
}

export function applyFontScaleToDocument(scale: number): void {
  document.documentElement.style.setProperty("--font-scale", String(scale));
}

export function writeStoredFontScale(scale: number): void {
  try {
    localStorage.setItem(FONT_SIZE_STORAGE_KEY, String(scale));
  } catch { /* ignore */ }
}

export function stepFontScale(current: number, direction: 1 | -1): FontSizeStep {
  const idx = FONT_SIZE_STEPS.indexOf(current as FontSizeStep);
  if (idx === -1) {
    return direction === 1
      ? (FONT_SIZE_STEPS.find((s) => s > current) ?? FONT_SCALE_MAX)
      : ([...FONT_SIZE_STEPS].reverse().find((s) => s < current) ?? FONT_SCALE_MIN);
  }
  const next = idx + direction;
  return FONT_SIZE_STEPS[Math.max(0, Math.min(FONT_SIZE_STEPS.length - 1, next))];
}

export function readStoredTheme(): ThemeMode {
  try {
    const v = localStorage.getItem(THEME_STORAGE_KEY);
    return v === "dark" ? "dark" : "light";
  } catch {
    return "light";
  }
}

/** Apply to `<html>` only (no localStorage write). For first paint + SSR-safe init. */
export function applyThemeToDocument(theme: ThemeMode): void {
  document.documentElement.dataset.theme = theme;
}

export function writeStoredTheme(theme: ThemeMode): void {
  try {
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    /* ignore */
  }
}
