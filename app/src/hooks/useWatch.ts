/**
 * useWatch — connects to /ws/watch for realtime "live watching" of the
 * screen via Gemini Live.
 *
 * The backend streams screen frames (and optional mic audio) to Gemini and
 * relays the model's running text commentary back over the socket. This hook
 * exposes a simple start/stop API plus the accumulated commentary.
 *
 * Usage:
 *   const w = useWatch({ enabled: open });
 *   w.start({ prompt, fps, audio }); w.stop();
 *   w.commentary  — accumulated commentary lines
 *   w.watching    — true while a live session is active
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { WatchWSEvent } from "../types";
import { WS_BASE } from "../config/apiBase";

const WATCH_WS_URL = `${WS_BASE}/ws/watch`;
const RECONNECT_DELAY_MS = 3000;

export interface WatchCommentaryLine {
  id: string;
  text: string;
  ts: number;
}

export interface UseWatchOptions {
  enabled?: boolean;
}

export interface StartWatchParams {
  prompt?: string;
  fps?: number;
  audio?: boolean;
}

export interface UseWatchReturn {
  connected: boolean;
  watching: boolean;
  commentary: WatchCommentaryLine[];
  error: string | null;
  start: (params?: StartWatchParams) => void;
  stop: () => void;
  clear: () => void;
  clearError: () => void;
}

export function useWatch({ enabled = false }: UseWatchOptions = {}): UseWatchReturn {
  const [connected, setConnected] = useState(false);
  const [watching, setWatching] = useState(false);
  const [commentary, setCommentary] = useState<WatchCommentaryLine[]>([]);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const enabledRef = useRef(enabled);
  enabledRef.current = enabled;
  const idRef = useRef(0);

  const send = useCallback((msg: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  const teardownSocket = useCallback(() => {
    const ws = wsRef.current;
    if (ws) {
      ws.onopen = null;
      ws.onmessage = null;
      ws.onerror = null;
      ws.onclose = null;
      try {
        ws.close();
      } catch {
        /* ignore */
      }
    }
    wsRef.current = null;
  }, []);

  const connect = useCallback(() => {
    if (!enabledRef.current) return;
    const existing = wsRef.current;
    if (
      existing &&
      (existing.readyState === WebSocket.OPEN || existing.readyState === WebSocket.CONNECTING)
    ) {
      return;
    }
    teardownSocket();

    const ws = new WebSocket(WATCH_WS_URL);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);

    ws.onmessage = (ev) => {
      let event: WatchWSEvent;
      try {
        event = JSON.parse(ev.data);
      } catch {
        return;
      }
      switch (event.type) {
        case "state":
          setWatching(event.state === "watching");
          break;
        case "commentary":
          if (event.text) {
            const line: WatchCommentaryLine = {
              id: `w-${++idRef.current}`,
              text: event.text,
              ts: Date.now() / 1000,
            };
            setCommentary((prev) => [...prev, line]);
          }
          break;
        case "error":
          if (event.message) setError(event.message);
          break;
        default:
          break;
      }
    };

    ws.onclose = () => {
      if (wsRef.current !== ws) return;
      setConnected(false);
      setWatching(false);
      wsRef.current = null;
      if (enabledRef.current) {
        if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = setTimeout(connect, RECONNECT_DELAY_MS);
      }
    };

    ws.onerror = () => ws.close();
  }, [teardownSocket]);

  useEffect(() => {
    if (enabled) {
      connect();
    } else {
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "stop" }));
      }
      teardownSocket();
      setConnected(false);
      setWatching(false);
    }
    return () => {
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      teardownSocket();
    };
  }, [enabled, connect, teardownSocket]);

  const start = useCallback(
    (params: StartWatchParams = {}) => {
      setError(null);
      send({ type: "start", ...params });
    },
    [send],
  );
  const stop = useCallback(() => {
    send({ type: "stop" });
    setWatching(false);
  }, [send]);
  const clear = useCallback(() => setCommentary([]), []);
  const clearError = useCallback(() => setError(null), []);

  return {
    connected,
    watching,
    commentary,
    error,
    start,
    stop,
    clear,
    clearError,
  };
}
