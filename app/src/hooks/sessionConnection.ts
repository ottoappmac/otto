import type { WSMessage } from "../types";
import { wsUrl } from "../config/apiBase";

/**
 * One WebSocket per session per window. Chat, Runs, and Code subscribe
 * here so they do not split a single backend queue between them.
 */

const INITIAL_BACKOFF_MS = 1_000;
const MAX_BACKOFF_MS = 30_000;

type Listener = (msg: WSMessage) => void;

interface Entry {
  sessionId: string;
  ws: WebSocket | null;
  listeners: Set<Listener>;
  statusListeners: Set<(connected: boolean) => void>;
  refs: number;
  lastSeq: number;
  reconnectTimer: ReturnType<typeof setTimeout> | null;
  backoff: number;
  intentional: boolean;
}

const entries = new Map<string, Entry>();

function notifyStatus(entry: Entry, connected: boolean) {
  entry.statusListeners.forEach((fn) => fn(connected));
}

function connect(entry: Entry) {
  const since = entry.lastSeq > 0 ? `?since=${entry.lastSeq}` : "?since=live";
  const ws = new WebSocket(wsUrl(`/ws/chat/${entry.sessionId}${since}`));
  entry.ws = ws;

  ws.onopen = () => {
    entry.backoff = INITIAL_BACKOFF_MS;
    notifyStatus(entry, true);
  };

  ws.onclose = () => {
    if (entry.ws === ws) entry.ws = null;
    notifyStatus(entry, false);
    if (!entry.intentional && entry.refs > 0) {
      const delay = entry.backoff;
      entry.backoff = Math.min(delay * 2, MAX_BACKOFF_MS);
      entry.reconnectTimer = setTimeout(() => connect(entry), delay);
    }
  };

  ws.onerror = () => {
    // onclose follows and schedules the reconnect.
  };

  ws.onmessage = (event) => {
    if (entry.ws !== ws) return;
    let msg: WSMessage;
    try {
      msg = JSON.parse(event.data as string) as WSMessage;
    } catch {
      return;
    }
    if (typeof msg.seq === "number") {
      if (msg.seq <= entry.lastSeq) return;
      entry.lastSeq = msg.seq;
    }
    entry.listeners.forEach((listener) => listener(msg));
  };
}

export function acquireSessionSocket(
  sessionId: string,
  listener: Listener,
  onStatus?: (connected: boolean) => void,
): () => void {
  let entry = entries.get(sessionId);
  if (!entry) {
    entry = {
      sessionId,
      ws: null,
      listeners: new Set(),
      statusListeners: new Set(),
      refs: 0,
      lastSeq: 0,
      reconnectTimer: null,
      backoff: INITIAL_BACKOFF_MS,
      intentional: false,
    };
    entries.set(sessionId, entry);
  }
  entry.refs += 1;
  entry.listeners.add(listener);
  if (onStatus) entry.statusListeners.add(onStatus);
  entry.intentional = false;
  if (entry.ws?.readyState === WebSocket.OPEN) {
    onStatus?.(true);
  } else if (!entry.ws || entry.ws.readyState === WebSocket.CLOSED) {
    connect(entry);
  }

  return () => {
    const current = entries.get(sessionId);
    if (!current) return;
    current.listeners.delete(listener);
    if (onStatus) current.statusListeners.delete(onStatus);
    current.refs -= 1;
    if (current.refs <= 0) {
      current.intentional = true;
      if (current.reconnectTimer) clearTimeout(current.reconnectTimer);
      current.ws?.close();
      entries.delete(sessionId);
    }
  };
}

export function sendSessionMessage(sessionId: string, payload: unknown): boolean {
  const entry = entries.get(sessionId);
  if (entry?.ws?.readyState === WebSocket.OPEN) {
    entry.ws.send(JSON.stringify(payload));
    return true;
  }
  return false;
}

export function sessionSocketOpen(sessionId: string): boolean {
  return entries.get(sessionId)?.ws?.readyState === WebSocket.OPEN;
}

/** Test hook. Not used by the app. */
export function resetSessionConnections(): void {
  for (const entry of entries.values()) {
    entry.intentional = true;
    if (entry.reconnectTimer) clearTimeout(entry.reconnectTimer);
    entry.ws?.close();
  }
  entries.clear();
}
