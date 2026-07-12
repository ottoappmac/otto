/**
 * transcribePanel — tiny module-level store for the live-transcription side
 * panel's open/closed state.
 *
 * The panel is mounted once in `Layout` (so its WebSocket connection and
 * recording state persist across route changes) but is triggered from the
 * `Sidebar` nav item. This store lets both subscribe to the same boolean
 * without threading props through the router tree, mirroring the pattern in
 * `askOttoBus.ts` / `nativeNotify.ts`.
 */

type Listener = (open: boolean) => void;

let open = false;
const listeners = new Set<Listener>();

function notify() {
  for (const cb of listeners) cb(open);
}

export function isTranscribePanelOpen(): boolean {
  return open;
}

/** Subscribe to open/close changes. Immediately invoked with the current value. */
export function subscribeTranscribePanel(cb: Listener): () => void {
  listeners.add(cb);
  cb(open);
  return () => listeners.delete(cb);
}

export function openTranscribePanel(): void {
  if (open) return;
  open = true;
  notify();
}

export function closeTranscribePanel(): void {
  if (!open) return;
  open = false;
  notify();
}

export function toggleTranscribePanel(): void {
  open = !open;
  notify();
}
