/**
 * askOttoBus — module-level bridge for sending captured-transcript text (and
 * optional screenshot attachments) to the chat agent from outside the chat
 * route (e.g. the transcription drawer).
 *
 * Mirrors the `onPendingRoute` pattern in `nativeNotify.ts`: the drawer emits
 * a payload via `emitAskOtto`, and `ChatPage` subscribes via `onAskOtto`. Any
 * payload that arrives before `ChatPage` is mounted (the drawer navigates to
 * `/chat` first) is queued and flushed to the listener as soon as it
 * subscribes, so an incremental auto-send never gets dropped during the mount
 * race.
 */

import type { AskImage } from "../types";

export interface AskPayload {
  text: string;
  /** Optional screenshot attachments to upload + send alongside the text. */
  images?: AskImage[];
}

type AskListener = (payload: AskPayload) => void;

let askListener: AskListener | null = null;
const pending: AskPayload[] = [];

/**
 * Subscribe to transcript hand-offs. Drains any payload queued before the
 * listener registered. Returns an unsubscribe function.
 */
export function onAskOtto(cb: AskListener): () => void {
  askListener = cb;
  if (pending.length > 0) {
    const queued = pending.splice(0, pending.length);
    for (const payload of queued) cb(payload);
  }
  return () => {
    if (askListener === cb) askListener = null;
  };
}

/**
 * Send transcript text (and optional screenshots) to the chat agent (queued if
 * no listener yet). A payload with neither text nor images is ignored.
 */
export function emitAskOtto(text: string, images?: AskImage[]): void {
  const trimmed = text.trim();
  const imgs = images?.filter((i) => i.dataUrl) ?? [];
  if (!trimmed && imgs.length === 0) return;
  const payload: AskPayload = { text: trimmed, images: imgs.length > 0 ? imgs : undefined };
  if (askListener) askListener(payload);
  else pending.push(payload);
}
