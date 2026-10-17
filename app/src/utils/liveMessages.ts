import type { ChatMessage, WSMessage } from "../types";
import { appendMissingInterrupts, preserveHitlResolved } from "./mergeToolMessages.ts";

/**
 * Fold one live WebSocket frame into the visible transcript.
 *
 * Chat and Code render the same bubbles, so they must build the same rows:
 * `agent_delta` tokens grow one placeholder bubble, the persisted `agent`
 * frame replaces it, a `tool_call` row becomes its own `tool_result` row, and
 * an approval or ask-user card ends the in-progress text. Pure function: no
 * React, no I/O.
 */

export interface LiveContext {
  /** Returns a fresh React key for a row this call appends. */
  nextId: () => string;
  sessionId?: string;
}

function isStreamingAgent(m: ChatMessage | undefined): m is ChatMessage {
  return Boolean(m && m.type === "agent" && m.metadata?.streaming);
}

function isInterrupt(m: ChatMessage): boolean {
  return m.type === "hitl_request" || m.type === "ask_user";
}

/** Stop treating a trailing in-progress bubble as live. */
function finalizeStreaming(prev: ChatMessage[]): ChatMessage[] {
  const last = prev[prev.length - 1];
  if (!isStreamingAgent(last)) return prev;
  const updated = [...prev];
  updated[updated.length - 1] = { ...last, metadata: { ...last.metadata, streaming: false } };
  return updated;
}

function row(msg: WSMessage, ctx: LiveContext, metadata = msg.metadata): ChatMessage {
  return {
    id: ctx.nextId(),
    type: msg.type,
    content: msg.content,
    metadata,
    timestamp: new Date(),
    sessionId: ctx.sessionId,
  };
}

export function applyLiveMessage(prev: ChatMessage[], msg: WSMessage, ctx: LiveContext): ChatMessage[] {
  switch (msg.type) {
    case "done":
      return finalizeStreaming(prev);

    case "stopped":
      return prev.map((m) => {
        if (m.type === "tool_call") {
          return { ...m, type: "tool_result" as const, metadata: { ...m.metadata, stopped: true } };
        }
        if (isStreamingAgent(m)) return { ...m, metadata: { ...m.metadata, streaming: false } };
        return m;
      });

    case "agent_delta": {
      const last = prev[prev.length - 1];
      if (isStreamingAgent(last)) {
        const updated = [...prev];
        updated[updated.length - 1] = { ...last, content: last.content + msg.content };
        return updated;
      }
      return [...prev, row({ ...msg, type: "agent" }, ctx, { streaming: true })];
    }

    case "hitl_request":
    case "ask_user": {
      const base = finalizeStreaming(prev);
      // The backend re-sends a pending interrupt on every (re)connect so a
      // dropped delivery recovers. Do not render the same card twice.
      for (let i = base.length - 1; i >= 0; i--) {
        const m = base[i];
        if (!isInterrupt(m)) continue;
        if (m.metadata?.resolved) break;
        if (
          m.type === msg.type
          && m.content === msg.content
          && JSON.stringify(m.metadata) === JSON.stringify(msg.metadata)
        ) {
          return base;
        }
        break;
      }
      return [...base, row(msg, ctx)];
    }

    case "execute_output": {
      const callId = msg.metadata?.tool_call_id as string | undefined;
      for (let i = prev.length - 1; i >= 0; i--) {
        const m = prev[i];
        if (m.type !== "tool_call" || m.content !== "execute") continue;
        const rowCallId = m.metadata?.tool_call_id as string | undefined;
        if (callId && rowCallId && rowCallId !== callId) continue;
        const existing = (m.metadata?.liveOutput as string[]) ?? [];
        // Keep the last 100 lines so a chatty command cannot grow without bound.
        const next = existing.length >= 100 ? [...existing.slice(-99), msg.content] : [...existing, msg.content];
        const updated = [...prev];
        updated[i] = { ...m, metadata: { ...m.metadata, liveOutput: next } };
        return updated;
      }
      return prev;
    }

    case "tool_result": {
      const name = (msg.metadata?.name as string) ?? "";
      const callId = msg.metadata?.tool_call_id as string | undefined;
      const images = msg.metadata?.images as Array<{ base64: string; mime_type: string }> | undefined;
      for (let i = prev.length - 1; i >= 0; i--) {
        const m = prev[i];
        if (m.type !== "tool_call") continue;
        const rowCallId = m.metadata?.tool_call_id as string | undefined;
        const matched = callId && rowCallId ? rowCallId === callId : m.content === name;
        if (!matched) continue;
        const updated = [...prev];
        updated[i] = {
          ...m,
          type: "tool_result",
          metadata: { ...m.metadata, result: msg.content, ...(images ? { images } : {}) },
        };
        return updated;
      }
      return [...prev, row(msg, ctx)];
    }

    case "run_depth": {
      const depthId = (msg.metadata?.run_depth_id as string) || "";
      if (depthId && prev.some((m) => m.metadata?.run_depth_id === depthId)) return prev;
      return [...prev, row(msg, ctx)];
    }

    case "agent":
    case "tool_call": {
      const last = prev[prev.length - 1];
      const hasPlaceholder = isStreamingAgent(last);
      if (hasPlaceholder && msg.type === "agent") {
        // The persisted text replaces the placeholder the deltas grew.
        const updated = [...prev];
        updated[updated.length - 1] = { ...last, content: msg.content, metadata: msg.metadata };
        return updated;
      }
      let base = prev;
      if (hasPlaceholder) {
        // Keep streamed preamble as a thought above the tool row. An empty
        // placeholder (the stream produced no text) is dropped.
        base = last.content.trim() ? finalizeStreaming(prev) : prev.slice(0, -1);
      }
      return [...base, row(msg, ctx)];
    }

    case "error":
      return [...finalizeStreaming(prev), row(msg, ctx)];

    default:
      // status, memory_*, workspace_* and the rest never become rows.
      return prev;
  }
}

/** True when this frame proves the agent is working right now. */
export function isActivityFrame(msg: WSMessage): boolean {
  return msg.type === "agent_delta" || msg.type === "tool_call" || msg.type === "execute_output";
}

/** The API never persists base64 images; keep the ones already on screen. */
function preserveImages(apiMerged: ChatMessage[], prev: ChatMessage[]): ChatMessage[] {
  return apiMerged.map((m, i) => {
    const local = prev[i];
    if (!local) return m;
    const apiImages = m.metadata?.images as unknown[] | undefined;
    const localImages = local.metadata?.images as unknown[] | undefined;
    if (!apiImages?.length && localImages?.length) {
      return { ...m, metadata: { ...m.metadata, images: localImages } };
    }
    return m;
  });
}

/**
 * Reconcile the on-screen transcript with the persisted one while a run is in
 * progress. This is the safety net for a dropped WebSocket frame.
 *
 * - The API is shorter: local rows are in flight and not saved yet. Keep them,
 *   but still surface a saved approval / ask-user card the socket missed.
 * - Otherwise the API wins, but rows keep their React keys so open cards and
 *   typed answers are not remounted.
 */
export function syncFromApi(prev: ChatMessage[], apiMerged: ChatMessage[]): ChatMessage[] {
  if (apiMerged.length < prev.length) return appendMissingInterrupts(prev, apiMerged) ?? prev;
  const merged = preserveHitlResolved(preserveImages(apiMerged, prev), prev);
  if (merged.length === prev.length) {
    const lastApi = merged[merged.length - 1];
    const lastLocal = prev[prev.length - 1];
    if (
      lastApi?.content === lastLocal?.content
      && lastApi?.type === lastLocal?.type
      && JSON.stringify(lastApi?.metadata) === JSON.stringify(lastLocal?.metadata)
    ) {
      return prev;
    }
  }
  return merged.map((m, i) => (i < prev.length ? { ...m, id: prev[i].id } : m));
}
