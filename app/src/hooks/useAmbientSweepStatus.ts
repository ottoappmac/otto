/**
 * Polls the ambient status endpoint to detect when a suggestion-generating
 * sweep is actively running, so the UI can show a "Generating suggestions…"
 * indicator.
 *
 * Sweeps are short-lived, so this polls more frequently than the hints poll.
 * It is intended to be mounted exactly once (in Layout) to avoid duplicate
 * polling loops.
 */
import { useEffect, useState } from "react";
import { api } from "./useApi";

const POLL_INTERVAL_MS = 5_000;

export function useAmbientSweepStatus(pollMs: number = POLL_INTERVAL_MS): boolean {
  const [running, setRunning] = useState(false);

  useEffect(() => {
    let cancelled = false;

    const check = async () => {
      try {
        const status = await api.ambientStatus();
        if (!cancelled) setRunning(Boolean(status.sweep_running));
      } catch {
        // Backend not ready or feature disabled — treat as not running.
        if (!cancelled) setRunning(false);
      }
    };

    void check();
    const timer = setInterval(() => void check(), pollMs);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [pollMs]);

  return running;
}
