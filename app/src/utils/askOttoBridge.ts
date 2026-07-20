/**
 * askOttoBridge — carries the "send captured context to chat" hand-off across
 * the two stealth webviews.
 *
 * In the normal (single-window) app, Live Capture and Chat live in the same
 * webview, so `askOttoBus` (a module-level bus) is enough. In stealth mode they
 * are two separate NSPanels/webviews, so the bus can't reach across. This module
 * bridges the gap over Tauri's cross-window event bus:
 *
 *   • Capture panel → `emit("otto://ask", payload)` + brings the Chat panel
 *     forward, so the transcript/screenshots land in the one conversation.
 *   • Chat panel    → listens for `otto://ask` and replays it into the local
 *     `askOttoBus`, which `ChatPage` already consumes.
 *   • Chat panel    → rebroadcasts its local `otto:agent-busy` CustomEvent as
 *     `otto://agent-busy` so the Capture panel can pause auto-send while Otto is
 *     working; the Capture panel turns it back into the local CustomEvent its
 *     drawer already listens for.
 */

import type { AskImage } from "../types";
import { emitAskOtto } from "./askOttoBus";
import { CHAT_WINDOW_LABEL } from "./stealthWindow";

const ASK_EVENT = "otto://ask";
const AGENT_BUSY_EVENT = "otto://agent-busy";

interface AskEventPayload {
  text: string;
  images?: AskImage[];
}

/**
 * Send captured text + screenshots from the Capture panel to the Chat panel and
 * bring the Chat panel forward. Falls back to a no-op outside Tauri.
 */
export async function routeAskToChat(text: string, images?: AskImage[]): Promise<void> {
  try {
    const { emit } = await import("@tauri-apps/api/event");
    await emit(ASK_EVENT, { text, images } satisfies AskEventPayload);
  } catch {
    // outside Tauri — nothing to route to
  }
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    await invoke("focus_stealth_panel", { panel: CHAT_WINDOW_LABEL, makeKey: true });
  } catch {
    // best effort — the Chat panel may already be visible
  }
}

/**
 * Wire the Chat panel side of the bridge. Returns a cleanup function. Safe to
 * call outside Tauri (no-op cleanup).
 */
export function initChatWindowBridge(): () => void {
  let unlistenAsk: (() => void) | undefined;
  let cancelled = false;

  import("@tauri-apps/api/event")
    .then(({ listen, emit }) => {
      // Inbound: capture panel → local chat bus.
      const p = listen<AskEventPayload>(ASK_EVENT, (event) => {
        const { text, images } = event.payload ?? { text: "" };
        emitAskOtto(text ?? "", images);
      });

      // Outbound: local agent-busy → cross-window, so the capture panel can
      // pause auto-send while Otto is streaming.
      const onBusy = (e: Event) => {
        void emit(AGENT_BUSY_EVENT, !!(e as CustomEvent).detail);
      };
      window.addEventListener("otto:agent-busy", onBusy);

      p.then((fn) => {
        if (cancelled) {
          fn();
          window.removeEventListener("otto:agent-busy", onBusy);
        } else {
          unlistenAsk = () => {
            fn();
            window.removeEventListener("otto:agent-busy", onBusy);
          };
        }
      });
    })
    .catch(() => {});

  return () => {
    cancelled = true;
    unlistenAsk?.();
  };
}

/**
 * Wire the Capture panel side of the bridge: turn the cross-window
 * `otto://agent-busy` event back into the local CustomEvent the drawer listens
 * for. Returns a cleanup function.
 */
export function initCaptureWindowBridge(): () => void {
  let unlisten: (() => void) | undefined;
  let cancelled = false;

  import("@tauri-apps/api/event")
    .then(({ listen }) =>
      listen<boolean>(AGENT_BUSY_EVENT, (event) => {
        window.dispatchEvent(
          new CustomEvent("otto:agent-busy", { detail: !!event.payload }),
        );
      }),
    )
    .then((fn) => {
      if (cancelled) fn();
      else unlisten = fn;
    })
    .catch(() => {});

  return () => {
    cancelled = true;
    unlisten?.();
  };
}
