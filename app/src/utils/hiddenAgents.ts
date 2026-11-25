/**
 * Agents that are installed but not meant to be picked by hand: benchmark agents (started by their scripts under
 * `scripts/`), session-eval agents, and subagents that cad-agent calls by name. They stay installed; this only keeps
 * them out of the Agents page list and the chat's `/` menu.
 */
export const BENCHMARK_AGENT_IDS: ReadonlySet<string> = new Set([
  "swebench-coding-agent",
  "benchcad-agent",
  "macarena-agent",
  "macarena-desktop-agent",
]);

export const HIDDEN_AGENT_IDS: ReadonlySet<string> = new Set([
  ...BENCHMARK_AGENT_IDS,
  "claude-session-eval-agent",
  "openclaw-session-eval-agent",
  "cad-critic",
  "drawing-reader",
]);

export function isPickableAgent(name: string): boolean {
  return !HIDDEN_AGENT_IDS.has(name);
}
