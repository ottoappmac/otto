import type { ChatMessage } from "../types";

/**
 * How a transcript is laid out: single bubbles, one collapsible block per
 * sub-agent, and one "Thought" accordion per stretch of intermediate agent
 * text. Shared by Chat and Code so both read the same.
 */
export type RenderItem =
  | { kind: "message"; message: ChatMessage; index: number }
  | { kind: "subagent-group"; name: string; messages: ChatMessage[] }
  | { kind: "thought-group"; messages: ChatMessage[] };

export function buildRenderItems(messages: ChatMessage[], thoughtFlags: boolean[]): RenderItem[] {
  const items: RenderItem[] = [];
  const groups = new Map<string, RenderItem & { kind: "subagent-group" }>();
  const subagentPrefixes = new Set<string>();
  let generation = 0;
  let prevWasSubagent = false;
  let thoughtBuf: ChatMessage[] = [];
  const flushThoughts = () => {
    if (thoughtBuf.length === 0) return;
    items.push({ kind: "thought-group", messages: thoughtBuf });
    thoughtBuf = [];
  };
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    // Errors are pinned to the end of the chat by the caller, not shown inline.
    if (msg.type === "error") continue;
    const subagent = msg.metadata?.subagent as string | undefined;
    if (subagent) {
      flushThoughts();
      subagentPrefixes.add(subagent.replace(/ #\d+$/, ""));
      const groupKey = `${subagent}::${generation}`;
      let group = groups.get(groupKey);
      if (!group) {
        group = { kind: "subagent-group", name: subagent, messages: [] };
        groups.set(groupKey, group);
        items.push(group);
      }
      group.messages.push(msg);
      prevWasSubagent = true;
    } else if (msg.type === "agent" && thoughtFlags[i]) {
      if (prevWasSubagent) generation++;
      prevWasSubagent = false;
      thoughtBuf.push(msg);
    } else {
      if (prevWasSubagent) generation++;
      prevWasSubagent = false;
      flushThoughts();
      items.push({ kind: "message", message: msg, index: i });
    }
  }
  flushThoughts();
  if (subagentPrefixes.size === 0) return items;
  // The sub-agent block already shows the `task` call that started it.
  return items.filter((item) => {
    if (item.kind !== "message") return true;
    const msg = item.message;
    if ((msg.type !== "tool_call" && msg.type !== "tool_result") || msg.content !== "task") return true;
    const agentType = (msg.metadata?.args as Record<string, unknown>)?.subagent_type as string | undefined;
    return !agentType || !subagentPrefixes.has(agentType);
  });
}

/** Only the newest todo list is shown expanded. */
export function computeLatestTodoFlags(messages: ChatMessage[]): boolean[] {
  const flags = new Array<boolean>(messages.length).fill(false);
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if ((m.type === "tool_call" || m.type === "tool_result") && m.content === "write_todos" && !m.metadata?.subagent) {
      flags[i] = true;
      break;
    }
  }
  return flags;
}
