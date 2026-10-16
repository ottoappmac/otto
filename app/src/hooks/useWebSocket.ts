import { useCallback, useEffect, useRef, useState } from "react";
import type { WSMessage } from "../types";
import { acquireSessionSocket, sendSessionMessage, sessionSocketOpen } from "./sessionConnection";

interface UseWebSocketOptions {
  sessionId: string | null;
  onMessage?: (msg: WSMessage) => void;
}

export function useWebSocket({ sessionId, onMessage }: UseWebSocketOptions) {
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;
  const [connected, setConnected] = useState(false);
  const sessionRef = useRef(sessionId);
  sessionRef.current = sessionId;

  useEffect(() => {
    if (!sessionId) {
      setConnected(false);
      return;
    }
    return acquireSessionSocket(
      sessionId,
      (msg) => onMessageRef.current?.(msg),
      setConnected,
    );
  }, [sessionId]);

  const waitForConnection = useCallback(
    (timeout = 5000) =>
      new Promise<void>((resolve, reject) => {
        const sid = sessionRef.current;
        if (sid && sessionSocketOpen(sid)) {
          resolve();
          return;
        }
        const interval = setInterval(() => {
          const current = sessionRef.current;
          if (current && sessionSocketOpen(current)) {
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

  const send = useCallback((content: string, depth?: string, mode?: string): boolean => {
    const sid = sessionRef.current;
    if (!sid) return false;
    const payload: Record<string, unknown> = { content };
    if (depth) payload.depth = depth;
    if (mode) payload.mode = mode;
    return sendSessionMessage(sid, payload);
  }, []);

  const sendEdit = useCallback((messageIndex: number, content: string, depth?: string, mode?: string): boolean => {
    const sid = sessionRef.current;
    if (!sid) return false;
    return sendSessionMessage(sid, {
      type: "edit",
      message_index: messageIndex,
      content,
      ...(depth ? { depth } : {}),
      ...(mode ? { mode } : {}),
    });
  }, []);

  const sendHitlResponse = useCallback((
    decisions: Array<Record<string, unknown>>,
    requestId?: string,
  ): boolean => {
    const sid = sessionRef.current;
    if (!sid) return false;
    return sendSessionMessage(sid, {
      type: "hitl_response",
      decisions,
      ...(requestId ? { request_id: requestId } : {}),
    });
  }, []);

  const sendContext = useCallback((content: string): boolean => {
    const sid = sessionRef.current;
    if (!sid) return false;
    return sendSessionMessage(sid, { type: "add_context", content });
  }, []);

  return { connected, send, sendEdit, sendHitlResponse, sendContext, waitForConnection };
}
