import type { ChatMessage } from "../types";

/**
 * Collapse raw persisted messages so that each tool_call + tool_result pair
 * becomes a single "tool_result" entry (matching the live WebSocket merge
 * behaviour).  Also marks hitl_request entries as resolved when a subsequent
 * tool_result for the same tool exists.
 */
export function mergeToolMessages(messages: ChatMessage[]): ChatMessage[] {
  const merged: ChatMessage[] = [];
  for (const m of messages) {
    if (m.type === "tool_result") {
      const toolName = (m.metadata?.name as string) ?? "";
      const resultTcId = m.metadata?.tool_call_id as string | undefined;
      let found = false;
      for (let i = merged.length - 1; i >= 0; i--) {
        if (merged[i].type !== "tool_call") continue;
        const callTcId = merged[i].metadata?.tool_call_id as string | undefined;
        const matched = resultTcId && callTcId
          ? callTcId === resultTcId
          : merged[i].content === toolName;
        if (matched) {
          const images = m.metadata?.images;
          merged[i] = {
            ...merged[i],
            type: "tool_result",
            metadata: { ...merged[i].metadata, result: m.content, ...(images ? { images } : {}) },
          };
          found = true;
          break;
        }
      }
      if (found) {
        for (let i = merged.length - 1; i >= 0; i--) {
          const mt = merged[i].type;
          if (mt === "hitl_request" && !merged[i].metadata?.resolved) {
            const meta = merged[i].metadata as Record<string, unknown> | undefined;
            // ``request_credential`` interrupts identify themselves via
            // ``metadata.type``, not ``action_requests`` (those come from
            // the langchain HumanInTheLoopMiddleware tool-approval flow).
            if (meta?.type === "request_credential" && toolName === "request_credential") {
              const answer = String(m.content ?? "");
              merged[i] = { ...merged[i], metadata: { ...merged[i].metadata, resolved: true, decisions: [{ type: "credential_provided", answer }] } };
              break;
            }
            const actions = meta?.action_requests as Array<{ name: string }> | undefined;
            if (actions?.some((a) => a.name === toolName)) {
              merged[i] = { ...merged[i], metadata: { ...merged[i].metadata, resolved: true, decisions: [{ type: "approve" }] } };
            }
            break;
          }
          if (mt === "ask_user" && !merged[i].metadata?.resolved && toolName === "ask_user") {
            merged[i] = { ...merged[i], metadata: { ...merged[i].metadata, resolved: true, decisions: [{ type: "ask_user_answer", answer: m.content }] } };
            break;
          }
        }
        continue;
      }
    }
    merged.push(m);
  }
  return merged;
}

function isInterrupt(m: ChatMessage): boolean {
  return m.type === "hitl_request" || m.type === "ask_user";
}

/**
 * Keep client-side HITL/ask_user ``resolved`` when the API transcript has
 * not caught up yet (no tool_result, or the backend stamp is still in
 * flight).  Without this, the 2s streaming poll re-opens the approval
 * card and Always-allow re-sends ``hitl_response``, cancelling the
 * in-flight execute.
 */
export function preserveHitlResolved(
  apiMerged: ChatMessage[],
  prev: ChatMessage[],
): ChatMessage[] {
  return apiMerged.map((m, i) => {
    if (!isInterrupt(m) || m.metadata?.resolved) return m;
    const local = matchingLocalInterrupt(prev, m, i);
    if (!local?.metadata?.resolved) return m;
    return {
      ...m,
      metadata: {
        ...m.metadata,
        resolved: true,
        decisions: local.metadata.decisions ?? m.metadata?.decisions,
      },
    };
  });
}

function matchingLocalInterrupt(
  prev: ChatMessage[],
  apiMsg: ChatMessage,
  index: number,
): ChatMessage | undefined {
  const atIndex = prev[index];
  if (atIndex && atIndex.type === apiMsg.type && atIndex.content === apiMsg.content) {
    return atIndex;
  }
  for (let i = prev.length - 1; i >= 0; i--) {
    const p = prev[i];
    if (p.type === apiMsg.type && p.content === apiMsg.content) return p;
  }
  return undefined;
}

