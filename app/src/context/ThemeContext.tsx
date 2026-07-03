import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import {
  applyThemeToDocument, readStoredTheme, writeStoredTheme, type ThemeMode,
  applyFontScaleToDocument, readStoredFontScale, writeStoredFontScale, stepFontScale,
  FONT_SCALE_MIN, FONT_SCALE_MAX,
} from "../theme";

type ThemeContextValue = {
  theme: ThemeMode;
  setTheme: (t: ThemeMode) => void;
  toggleTheme: () => void;
  fontScale: number;
  setFontScale: (scale: number) => void;
  increaseFontScale: () => void;
  decreaseFontScale: () => void;
  canIncrease: boolean;
  canDecrease: boolean;
};

const ThemeContext = createContext<ThemeContextValue | null>(null);

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<ThemeMode>(() => readStoredTheme());
  const [fontScale, setFontScaleState] = useState<number>(() => readStoredFontScale());

  useEffect(() => {
    applyThemeToDocument(theme);
  }, [theme]);

  useEffect(() => {
    applyFontScaleToDocument(fontScale);
  }, [fontScale]);

  const setTheme = useCallback((t: ThemeMode) => {
    writeStoredTheme(t);
    setThemeState(t);
  }, []);

  const toggleTheme = useCallback(() => {
    setThemeState((prev) => {
      const next: ThemeMode = prev === "dark" ? "light" : "dark";
      writeStoredTheme(next);
      return next;
    });
  }, []);

  const setFontScale = useCallback((scale: number) => {
    writeStoredFontScale(scale);
    setFontScaleState(scale);
  }, []);

  const increaseFontScale = useCallback(() => {
    setFontScaleState((prev) => {
      const next = stepFontScale(prev, 1);
      writeStoredFontScale(next);
      return next;
    });
  }, []);

  const decreaseFontScale = useCallback(() => {
    setFontScaleState((prev) => {
      const next = stepFontScale(prev, -1);
      writeStoredFontScale(next);
      return next;
    });
  }, []);

  // Cmd+= / Cmd++ to increase, Cmd+- to decrease, Cmd+0 to reset
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const isMod = e.metaKey || e.ctrlKey;
      if (!isMod) return;
      if (e.key === "=" || e.key === "+") {
        e.preventDefault();
        increaseFontScale();
      } else if (e.key === "-") {
        e.preventDefault();
        decreaseFontScale();
      } else if (e.key === "0") {
        e.preventDefault();
        setFontScale(1);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [increaseFontScale, decreaseFontScale, setFontScale]);

  const value = useMemo(
    () => ({
      theme, setTheme, toggleTheme,
      fontScale, setFontScale, increaseFontScale, decreaseFontScale,
      canIncrease: fontScale < FONT_SCALE_MAX,
      canDecrease: fontScale > FONT_SCALE_MIN,
    }),
    [theme, setTheme, toggleTheme, fontScale, setFontScale, increaseFontScale, decreaseFontScale],
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (!ctx) {
    throw new Error("useTheme must be used within ThemeProvider");
  }
  return ctx;
}
