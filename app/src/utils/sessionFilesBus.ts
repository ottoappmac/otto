/**
 * sessionFilesBus — announces that a session's files changed outside the
 * normal agent turn.
 *
 * `ChatPage` owns the Files panel but only reloads it when a run finishes,
 * so writes made by other surfaces — the Watch panel uploading a video or
 * finalising a screen recording — would otherwise stay invisible until the
 * next agent turn. Mirrors `watchPanel.ts`.
 */

type Listener = (sessionId: string) => void;

const listeners = new Set<Listener>();

export function subscribeSessionFiles(cb: Listener): () => void {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

export function notifySessionFilesChanged(sessionId: string): void {
  for (const cb of listeners) cb(sessionId);
}
