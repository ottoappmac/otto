import { useCallback, useEffect, useRef, useState } from "react";
import type { WSMessage } from "../types";
import { WS_BASE } from "../config/apiBase";

const INITIAL_BACKOFF_MS = 1_000;
const MAX_BACKOFF_MS = 30_000;

interface UseWebSocketOptions {
  sessionId: string | null;
  onMessage?: (msg: WSMessage) => void;
}

export function useWebSocket({ sessionId, onMessage }: UseWebSocketOptions) {
  const wsRef = useRef<WebSocket | null>(null);
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;
  const [connected, setConnected] = useState(false);

  const backoffRef = useRef(INITIAL_BACKOFF_MS);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const intentionalClose = useRef(false);

  const connect = useCallback(
    (sid: string) => {
      const ws = new WebSocket(`${WS_BASE}/ws/chat/${sid}`);
      wsRef.current = ws;

      ws.onopen = () => {
        backoffRef.current = INITIAL_BACKOFF_MS;
        setConnected(true);
      };

      ws.onclose = () => {
        setConnected(false);
        if (!intentionalClose.current) {
          const delay = backoffRef.current;
          backoffRef.current = Math.min(delay * 2, MAX_BACKOFF_MS);
          reconnectTimer.current = setTimeout(() => connect(sid), delay);
        }
      };

      ws.onerror = () => {
        // onclose always fires after onerror; reconnect is handled there
      };

      ws.onmessage = (event) => {
        // Guard: drop messages from a stale WebSocket that hasn't fully
        // closed yet.  When the session changes, wsRef is pointed at the
        // new connection before the old one finishes closing — any late
        // messages on the old socket must be discarded.
        if (wsRef.current !== ws) return;

        try {
          const msg: WSMessage = JSON.parse(event.data);
          // Dispatch immediately — no setTimeout / rAF batching.
          // WKWebView (Tauri/macOS) throttles both rAF and short timers
          // aggressively when the window is not the key window, causing
          // hitl_request and other messages to appear frozen until the
          // user navigates away and back.  React 18 automatically batches
          // state updates that occur within the same synchronous event
          // handler, so dispatching here is safe and performant.
          onMessageRef.current?.(msg);
        } catch (e) {
          console.warn("Malformed WebSocket message:", e);
        }
      };
    },
    [],
  );

  useEffect(() => {
    if (!sessionId) return;

    intentionalClose.current = false;
    connect(sessionId);

    return () => {
      intentionalClose.current = true;
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current);
        reconnectTimer.current = null;
      }
      // Null the ref BEFORE closing so the onmessage guard
      // (wsRef.current !== ws) drops any late-arriving messages.
      const stale = wsRef.current;
      wsRef.current = null;
      stale?.close();
      setConnected(false);
    };
  }, [sessionId, connect]);

  const waitForConnection = useCallback(
    (timeout = 5000) =>
      new Promise<void>((resolve, reject) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          resolve();
          return;
        }
        const interval = setInterval(() => {
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            clearInterval(interval);
            resolve();
          }
        }, 50);
        setTimeout(() => {
          clearInterval(interval);
          reject(new Error("WebSocket connection timeout"));
        }, timeout);
      }),
    [],
  );

  const send = useCallback((content: string): boolean => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ content }));
      return true;
    }
    return false;
  }, []);

  const sendEdit = useCallback((messageIndex: number, content: string): boolean => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({ type: "edit", message_index: messageIndex, content }),
      );
      return true;
    }
    return false;
  }, []);

  const sendHitlResponse = useCallback((decisions: Array<Record<string, unknown>>): boolean => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({ type: "hitl_response", decisions }),
      );
      return true;
    }
    return false;
  }, []);

  const sendContext = useCallback((content: string): boolean => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({ type: "add_context", content }),
      );
      return true;
    }
    return false;
  }, []);

  return { connected, send, sendEdit, sendHitlResponse, sendContext, waitForConnection };
}
