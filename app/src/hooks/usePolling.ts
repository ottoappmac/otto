import { useEffect, useRef } from "react";

/**
 * Run ``tick`` once on mount and then every ``intervalMs`` while ``enabled``.
 *
 * Polling is paused while the window is hidden (``visibilitychange``) so
 * minimised / backgrounded app windows stop hammering the backend. When the
 * window becomes visible again the tick fires immediately, then resumes the
 * interval. Re-entrant calls are suppressed: if the previous tick is still
 * in flight when the timer fires, we skip that round instead of stacking.
 */
export function usePolling(
  tick: () => void | Promise<void>,
  intervalMs: number,
  enabled: boolean = true,
): void {
  const tickRef = useRef(tick);
  tickRef.current = tick;

  useEffect(() => {
    if (!enabled) return;
    if (intervalMs <= 0) return;

    let cancelled = false;
    let timer: ReturnType<typeof setInterval> | null = null;
    let running = false;

    const runOnce = async () => {
      if (cancelled || running) return;
      if (typeof document !== "undefined" && document.visibilityState === "hidden") {
        return;
      }
      running = true;
      try {
        await tickRef.current();
      } finally {
        running = false;
      }
    };

    const start = () => {
      if (timer !== null) return;
      void runOnce();
      timer = setInterval(() => {
        void runOnce();
      }, intervalMs);
    };

    const stop = () => {
      if (timer !== null) {
        clearInterval(timer);
        timer = null;
      }
    };

    const onVisibility = () => {
      if (document.visibilityState === "visible") {
        start();
      } else {
        stop();
      }
    };

    if (typeof document === "undefined" || document.visibilityState === "visible") {
      start();
    }
    if (typeof document !== "undefined") {
      document.addEventListener("visibilitychange", onVisibility);
    }

    return () => {
      cancelled = true;
      stop();
      if (typeof document !== "undefined") {
        document.removeEventListener("visibilitychange", onVisibility);
      }
    };
  }, [intervalMs, enabled]);
}
