import type { TimelineEvent } from "../types";

/**
 * Fold live WebSocket frames into the run-detail timeline.
 *
 * Chat grows ``agent_delta`` tokens into one bubble. The session run page
 * used to append every token as its own row, which rendered as a rail of
 * empty dots. Same contract here: deltas accumulate, the persisted
 * ``agent`` frame replaces the placeholder, and a following tool call
 * keeps any preamble as a completed assistant thought.
 */

export type LiveWsMessage = {
  type: string;
  content?: unknown;
  metadata?: Record<string, unknown>;
};

const SKIP_TYPES = new Set([
  "execute_output",
  "memory_search",
  "memory_context",
  "context_received",
]);

export const TIMELINE_RENDER_TYPES = new Set([
  "user",
  "assistant",
  "tool_call",
  "tool_result",
  "system",
  "status",
  "error",
  "done",
  "stopped",
]);

function isStreamingAssistant(ev: TimelineEvent | undefined): boolean {
  return Boolean(ev && ev.type === "assistant" && ev.meta?.streaming);
}

function toTimelineEvent(raw: LiveWsMessage, extra?: Partial<TimelineEvent>): TimelineEvent {
  const meta = raw.metadata ?? {};
  return {
    type: raw.type === "agent" || raw.type === "agent_delta" ? "assistant" : raw.type,
    content: raw.content,
    subagent: meta.subagent as string | undefined,
    args: meta.args as Record<string, unknown> | undefined,
    tool: raw.type === "tool_result"
      ? (meta.name as string | undefined)
      : (typeof raw.content === "string" ? raw.content : undefined),
    tool_call_id: meta.tool_call_id as string | undefined,
    images: meta.images as { base64: string; mime_type: string }[] | undefined,
    ...extra,
  };
}

function finalizeStreaming(prev: TimelineEvent[]): TimelineEvent[] {
  if (prev.length === 0) return prev;
  const last = prev[prev.length - 1];
  if (!isStreamingAssistant(last)) return prev;
  const content = typeof last.content === "string" ? last.content : "";
  if (!content.trim()) return prev.slice(0, -1);
  const updated = [...prev];
  updated[updated.length - 1] = { ...last, meta: { ...last.meta, streaming: false } };
  return updated;
}

export function applyLiveTimelineMessage(
  prev: TimelineEvent[],
  raw: LiveWsMessage,
): TimelineEvent[] {
  if (raw.type === "agent_delta") {
    const piece = typeof raw.content === "string" ? raw.content : "";
    if (!piece) return prev;
    const last = prev[prev.length - 1];
    if (isStreamingAssistant(last)) {
      const updated = [...prev];
      updated[updated.length - 1] = {
        ...last,
        content: String(last.content ?? "") + piece,
      };
      return updated;
    }
    return [...prev, toTimelineEvent(raw, { content: piece, meta: { streaming: true } })];
  }

  if (SKIP_TYPES.has(raw.type)) return prev;

  if (raw.type === "agent") {
    const last = prev[prev.length - 1];
    if (isStreamingAssistant(last)) {
      const updated = [...prev];
      updated[updated.length - 1] = toTimelineEvent(raw);
      return updated;
    }
  }

  const base = finalizeStreaming(prev);
  return [...base, toTimelineEvent(raw)];
}
