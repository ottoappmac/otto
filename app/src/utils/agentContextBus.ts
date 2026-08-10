/**
 * agentContextBus — pushes background observations into the chat session's
 * *context* channel from outside the chat route.
 *
 * Distinct from `askOttoBus`, which sends a user message and starts a turn.
 * Context is injected into a run already in flight, or folded into whatever
 * the user asks next, so the Watch panel can narrate a screen for minutes
 * without repeatedly waking the agent.
 *
 * Nothing is queued for a late subscriber: unlike a hand-off the user
 * explicitly asked for, stale commentary about a screen they have since left
 * is worse than no commentary at all.
 */

type ContextListener = (text: string) => void;

let listener: ContextListener | null = null;

/** Subscribe to context hand-offs. Returns an unsubscribe function. */
export function onAgentContext(cb: ContextListener): () => void {
  listener = cb;
  return () => {
    if (listener === cb) listener = null;
  };
}

/** True when a chat session is mounted and able to receive context. */
export function canSendAgentContext(): boolean {
  return listener !== null;
}

export function emitAgentContext(text: string): void {
  const trimmed = text.trim();
  if (!trimmed || !listener) return;
  listener(trimmed);
}
