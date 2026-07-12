/**
 * useTranscribe — connects to /ws/transcribe and exposes the system-audio +
 * microphone transcription API.
 *
 * Accumulates finalized segments (tagged by source), tracks the latest
 * in-progress partial per source, and surfaces per-source audio levels.
 *
 * Usage:
 *   const t = useTranscribe({ enabled: open });
 *   t.start(["system", "mic"]); t.stop();
 *   t.segments  — finalized lines
 *   t.partials  — live in-progress text keyed by source
 *   t.levels    — 0..1 audio level keyed by source
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  TranscribeSource,
  TranscribeState,
  TranscribeWSEvent,
  TranscriptSegment,
} from "../types";
import { WS_BASE } from "../config/apiBase";

const TRANSCRIBE_WS_URL = `${WS_BASE}/ws/transcribe`;
const RECONNECT_DELAY_MS = 3000;

export type SourceMap<T> = Partial<Record<TranscribeSource, T>>;

export interface UseTranscribeOptions {
  /** If false, the WebSocket is not opened. */
  enabled?: boolean;
  /** Called when a final segment arrives. */
  onSegment?: (seg: TranscriptSegment) => void;
}

export interface UseTranscribeReturn {
  state: TranscribeState;
  connected: boolean;
  segments: TranscriptSegment[];
  partials: SourceMap<string>;
  levels: SourceMap<number>;
  error: string | null;
  start: (sources: TranscribeSource[]) => void;
  stop: () => void;
  clear: () => void;
  /** Remove a single finalized segment by id (local only — not sent to the backend). */
  removeSegment: (id: string) => void;
  clearError: () => void;
}

export function useTranscribe({
  enabled = false,
  onSegment,
}: UseTranscribeOptions = {}): UseTranscribeReturn {
  const [state, setState] = useState<TranscribeState>("idle");
  const [connected, setConnected] = useState(false);
  const [segments, setSegments] = useState<TranscriptSegment[]>([]);
  const [partials, setPartials] = useState<SourceMap<string>>({});
  const [levels, setLevels] = useState<SourceMap<number>>({});
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const enabledRef = useRef(enabled);
  enabledRef.current = enabled;
  const segIdRef = useRef(0);
  const onSegmentRef = useRef(onSegment);
  onSegmentRef.current = onSegment;
  // Stable event ids already applied to segments — guards against a segment
  // being appended more than once if multiple sockets deliver the same event.
  const seenEidsRef = useRef<Set<string>>(new Set());

  const send = useCallback((msg: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  // Fully tear down the current socket WITHOUT letting its onclose schedule a
  // reconnect (handlers are detached first).  Keeps us to a single live socket.
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
    // Replace any stale (closing/closed) socket cleanly before opening a new one.
    teardownSocket();

    const ws = new WebSocket(TRANSCRIBE_WS_URL);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);

    ws.onmessage = (ev) => {
      let event: TranscribeWSEvent;
      try {
        event = JSON.parse(ev.data);
      } catch {
        return;
      }
      const source = (event.source ?? "system") as TranscribeSource;
      switch (event.type) {
        case "state":
          if (event.state) setState(event.state);
          break;
        case "segment":
          if (event.text) {
            // Dedupe by the backend's stable event id: if the same event is
            // delivered by more than one socket, only append it once.
            if (event.eid) {
              if (seenEidsRef.current.has(event.eid)) break;
              seenEidsRef.current.add(event.eid);
            }
            const seg: TranscriptSegment = {
              id: event.eid ?? `seg-${++segIdRef.current}`,
              text: event.text,
              ts: event.ts ?? Date.now() / 1000,
              source,
            };
            setSegments((prev) => [...prev, seg]);
            setPartials((prev) => ({ ...prev, [source]: "" }));
            onSegmentRef.current?.(seg);
          }
          break;
        case "partial":
          setPartials((prev) => ({ ...prev, [source]: event.text ?? "" }));
          break;
        case "level":
          setLevels((prev) => ({ ...prev, [source]: event.rms ?? 0 }));
          break;
        case "error":
          if (event.message) setError(event.message);
          break;
        default:
          break;
      }
    };

    ws.onclose = () => {
      // Ignore stale sockets that have been superseded.
      if (wsRef.current !== ws) return;
      setConnected(false);
      setState("idle");
      setLevels({});
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
      setState("idle");
      setLevels({});
    }
    return () => {
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      teardownSocket();
    };
  }, [enabled, connect, teardownSocket]);

  const start = useCallback(
    (sources: TranscribeSource[]) => {
      setError(null);
      setPartials({});
      send({ type: "start", sources });
    },
    [send],
  );
  const stop = useCallback(() => send({ type: "stop" }), [send]);
  const clear = useCallback(() => {
    setSegments([]);
    setPartials({});
    seenEidsRef.current = new Set();
  }, []);
  const removeSegment = useCallback((id: string) => {
    setSegments((prev) => prev.filter((s) => s.id !== id));
  }, []);
  const clearError = useCallback(() => setError(null), []);

  return {
    state,
    connected,
    segments,
    partials,
    levels,
    error,
    start,
    stop,
    clear,
    removeSegment,
    clearError,
  };
}
