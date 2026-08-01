/**
 * useWatch — connects to /ws/watch for "live watching" of the screen.
 *
 * The backend captures screen frames and relays a running text commentary
 * back over the socket, either from Gemini Live (realtime) or from a local
 * vision model describing a batch of frames every few seconds — `mode` says
 * which. When the live preview is enabled it also relays each captured frame
 * so the panel can show what the model is being shown.
 *
 * Usage:
 *   const w = useWatch({ enabled: open });
 *   w.start({ prompt, fps, audio }); w.stop();
 *   w.commentary  — accumulated commentary lines
 *   w.frame       — data URL of the most recent frame, if previewing
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
  /** Which backend served the session, known once watching starts. */
  mode: "gemini" | "local" | null;
  commentary: WatchCommentaryLine[];
  /** Data URL of the latest captured frame, when the preview is on. */
  frame: string | null;
  error: string | null;
  start: (params?: StartWatchParams) => void;
  stop: () => void;
  clear: () => void;
  clearError: () => void;
}

export function useWatch({ enabled = false }: UseWatchOptions = {}): UseWatchReturn {
  const [connected, setConnected] = useState(false);
  const [watching, setWatching] = useState(false);
  const [mode, setMode] = useState<"gemini" | "local" | null>(null);
  const [commentary, setCommentary] = useState<WatchCommentaryLine[]>([]);
  const [frame, setFrame] = useState<string | null>(null);
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
          if (event.state !== "watching") setFrame(null);
          break;
        case "mode":
          setMode(event.mode ?? null);
          break;
        case "frame":
          if (event.jpeg_b64) setFrame(`data:image/jpeg;base64,${event.jpeg_b64}`);
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
      setFrame(null);
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
    setFrame(null);
  }, [send]);
  const clear = useCallback(() => setCommentary([]), []);
  const clearError = useCallback(() => setError(null), []);

  return {
    connected,
    watching,
    mode,
    commentary,
    frame,
    error,
    start,
    stop,
    clear,
    clearError,
  };
}
