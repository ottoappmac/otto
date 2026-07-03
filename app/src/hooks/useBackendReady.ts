import { useState, useEffect, useRef } from "react";
import { API_BASE } from "../config/apiBase";
const STARTUP_POLL_MS = 500;
const HEARTBEAT_MS = 5_000;
const MAX_WAIT_MS = 300_000; // 5 min — AV scanning PyInstaller bundle on first launch can be slow
const FAIL_THRESHOLD = 3;

export function useBackendReady() {
  const [ready, setReady] = useState(false);
  const [timedOut, setTimedOut] = useState(false);
  const [backendReachable, setBackendReachable] = useState(true);
  const [elapsedSec, setElapsedSec] = useState(0);
  const stopped = useRef(false);
  const startTime = useRef(Date.now());

  // Phase 1: startup poll — fast cadence until the backend first responds
  useEffect(() => {
    stopped.current = false;
    const deadline = Date.now() + MAX_WAIT_MS;

    async function poll() {
      while (!stopped.current) {
        try {
          const res = await fetch(`${API_BASE}/api/health`);
          if (res.ok) {
            setReady(true);
            return;
          }
        } catch {
          // backend not up yet
        }
        if (Date.now() >= deadline) {
          setTimedOut(true);
          return;
        }
        await new Promise((r) => setTimeout(r, STARTUP_POLL_MS));
      }
    }

    const ticker = setInterval(() => {
      setElapsedSec(Math.floor((Date.now() - startTime.current) / 1000));
    }, 1000);

    poll();
    return () => {
      stopped.current = true;
      clearInterval(ticker);
    };
  }, []);

  // Phase 2: continuous heartbeat once ready
  useEffect(() => {
    if (!ready) return;
    let failures = 0;
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/health`);
        if (res.ok) {
          failures = 0;
          setBackendReachable(true);
        } else {
          failures++;
        }
      } catch {
        failures++;
      }
      if (failures >= FAIL_THRESHOLD) setBackendReachable(false);
    }, HEARTBEAT_MS);
    return () => clearInterval(interval);
  }, [ready]);

  return { ready, timedOut, backendReachable, elapsedSec };
}
