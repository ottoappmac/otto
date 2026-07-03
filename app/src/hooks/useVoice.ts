/**
 * useVoice — connects to /ws/voice and exposes the STT/wake voice API.
 *
 * Manages the WebSocket lifecycle, translates raw voice WS events into
 * React state, and exposes simple controls (pushToTalkStart, pushToTalkStop,
 * startListening, stopListening, configure).
 *
 * Usage:
 *   const voice = useVoice({ enabled: settings.voice.enabled });
 *   voice.pushToTalkStart();
 *   // voice.state === "capturing"
 *   // voice.transcript  — latest final transcript
 *   // voice.partial     — live partial transcript
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { VoiceState, VoiceWSEvent, VoiceConfig } from "../types";
import { WS_BASE } from "../config/apiBase";

// ---------------------------------------------------------------------------
// Chime: two ascending tones played via Web Audio API when recording starts
// ---------------------------------------------------------------------------
function playActivationChime() {
  try {
    const ctx = new AudioContext();
    const now = ctx.currentTime;
    const notes = [880, 1320]; // A5 → E6 — a quick rising two-tone
    notes.forEach((freq, i) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.type = "sine";
      osc.frequency.value = freq;
      const start = now + i * 0.08;
      const end = start + 0.1;
      gain.gain.setValueAtTime(0, start);
      gain.gain.linearRampToValueAtTime(0.25, start + 0.01);
      gain.gain.exponentialRampToValueAtTime(0.001, end);
      osc.start(start);
      osc.stop(end);
    });
    // Close context after both tones finish
    setTimeout(() => ctx.close(), 400);
  } catch {
    // AudioContext not available (e.g. SSR) — ignore silently
  }
}

const VOICE_WS_URL = `${WS_BASE}/ws/voice`;
const RECONNECT_DELAY_MS = 3000;

export interface UseVoiceOptions {
  /** If false, the WebSocket is not opened. */
  enabled?: boolean;
  /**
   * If true, automatically sends {"type":"start"} as soon as the WebSocket
   * connects.  Use this for wake-word mode where the mic should always be
   * open without an explicit user action.
   */
  autoStart?: boolean;
  /** Called when a final transcript arrives — e.g. to feed into handleSend. */
  onTranscript?: (text: string) => void;
  /** Called when the wake word fires. */
  onWake?: () => void;
}

export interface UseVoiceReturn {
  state: VoiceState;
  transcript: string;
  partial: string;
  connected: boolean;
  pushToTalkStart: () => void;
  pushToTalkStop: () => void;
  startListening: () => void;
  stopListening: () => void;
  configure: (patch: Partial<VoiceConfig>) => void;
}

export function useVoice({
  enabled = false,
  autoStart = false,
  onTranscript,
  onWake,
}: UseVoiceOptions = {}): UseVoiceReturn {
  const [state, setState] = useState<VoiceState>("idle");
  const [transcript, setTranscript] = useState("");
  const [partial, setPartial] = useState("");
  const [connected, setConnected] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const enabledRef = useRef(enabled);
  enabledRef.current = enabled;
  const autoStartRef = useRef(autoStart);
  autoStartRef.current = autoStart;

  const send = useCallback((msg: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  const connect = useCallback(() => {
    if (!enabledRef.current) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(VOICE_WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      if (autoStartRef.current) {
        ws.send(JSON.stringify({ type: "start" }));
      }
    };

    ws.onmessage = (ev) => {
      let event: VoiceWSEvent;
      try {
        event = JSON.parse(ev.data);
      } catch {
        return;
      }

      switch (event.type) {
        case "state":
          if (event.state) {
            if (event.state === "capturing") playActivationChime();
            setState(event.state);
          }
          break;
        case "transcript":
          if (event.text) {
            setTranscript(event.text);
            setPartial("");
            onTranscript?.(event.text);
          }
          break;
        case "partial":
          if (event.text) setPartial(event.text);
          break;
        case "wake":
          onWake?.();
          break;
        default:
          break;
      }
    };

    ws.onclose = () => {
      setConnected(false);
      setState("idle");
      wsRef.current = null;
      if (enabledRef.current) {
        reconnectTimerRef.current = setTimeout(connect, RECONNECT_DELAY_MS);
      }
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [onTranscript, onWake]);

  useEffect(() => {
    if (enabled) {
      connect();
    } else {
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
      wsRef.current = null;
      setConnected(false);
      setState("idle");
    }
    return () => {
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
    };
  }, [enabled, connect]);

  const startListening = useCallback(() => send({ type: "start" }), [send]);
  const stopListening = useCallback(() => send({ type: "stop" }), [send]);
  const pushToTalkStart = useCallback(() => send({ type: "ptt_start" }), [send]);
  const pushToTalkStop = useCallback(() => send({ type: "ptt_stop" }), [send]);
  const configure = useCallback(
    (patch: Partial<VoiceConfig>) => send({ type: "configure", config: patch }),
    [send],
  );

  return {
    state,
    transcript,
    partial,
    connected,
    pushToTalkStart,
    pushToTalkStop,
    startListening,
    stopListening,
    configure,
  };
}
