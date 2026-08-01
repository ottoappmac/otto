/**
 * watchPanel — tiny module-level store for the "Watch video" side panel's
 * open/closed state.
 *
 * The panel is mounted once in `Layout` (so its live-watch WebSocket and any
 * in-progress recording persist across route changes) but is triggered from
 * the `Sidebar` nav item. Mirrors `transcribePanel.ts`.
 */

type Listener = (open: boolean) => void;

let open = false;
const listeners = new Set<Listener>();

function notify() {
  for (const cb of listeners) cb(open);
}

export function isWatchPanelOpen(): boolean {
  return open;
}

/** Subscribe to open/close changes. Immediately invoked with the current value. */
export function subscribeWatchPanel(cb: Listener): () => void {
  listeners.add(cb);
  cb(open);
  return () => listeners.delete(cb);
}

export function openWatchPanel(): void {
  if (open) return;
  open = true;
  notify();
}

export function closeWatchPanel(): void {
  if (!open) return;
  open = false;
  notify();
}

export function toggleWatchPanel(): void {
  open = !open;
  notify();
}
