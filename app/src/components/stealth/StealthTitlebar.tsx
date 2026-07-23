import { useCallback, useEffect, useState } from "react";
import { AppWindow, Maximize2, Minimize2, MessageSquare, Minus, Power, Radio } from "lucide-react";
import type { StealthKind } from "../../utils/stealthWindow";
import { CHAT_WINDOW_LABEL, CAPTURE_WINDOW_LABEL } from "../../utils/stealthWindow";

// Compact = a peek of just the latest item (last chat message / last
// transcribed line). Expanded fills the monitor height. The compact heights are
// kept in sync with the initial panel sizes in `src-tauri/src/stealth.rs`.
const EXPAND_CONFIG: Record<StealthKind, { cls: string; compact: number }> = {
  chat: { cls: "stealth-chat-expanded", compact: 300 },
  capture: { cls: "stealth-capture-expanded", compact: 320 },
};

// When expanded, sit this far below the menu bar and this far above the bottom
// edge, so "full height" still leaves the menu bar reachable.
const TOP_INSET = 28;
const BOTTOM_MARGIN = 16;

interface StealthTitlebarProps {
  /** Which stealth panel this bar belongs to. */
  kind: StealthKind;
}

/**
 * Slim, draggable title bar for a borderless stealth panel (the panels have no
 * native title bar). Follows macOS HIG: deferential chrome, a translucent
 * material strip, comfortable hit targets, and a subtle capture-hidden dot so
 * the state is always legible.
 *
 * Provides the per-panel window controls: move (drag anywhere on the bar),
 * summon the *other* panel, expand/collapse height (chat), hide this panel,
 * exit compact mode (restores the normal main window but leaves stealth on),
 * and turn stealth off entirely (which also exits compact and dismisses both
 * panels). Drives focus-based transparency: solid when focused, faded when the
 * user clicks away.
 */
export default function StealthTitlebar({ kind }: StealthTitlebarProps) {
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    const root = document.documentElement;
    import("@tauri-apps/api/window")
      .then(({ getCurrentWindow }) =>
        getCurrentWindow().onFocusChanged(({ payload: focused }) => {
          root.classList.toggle("stealth-unfocused", !focused);
        }),
      )
      .then((fn) => {
        unlisten = fn;
      })
      .catch(() => {});
    return () => unlisten?.();
  }, []);

  const toggleExpand = useCallback(async () => {
    const next = !expanded;
    const cfg = EXPAND_CONFIG[kind];
    // Reveal/hide the fuller view (CSS driven), then grow to full monitor height
    // (or shrink back to the compact peek) to match.
    document.documentElement.classList.toggle(cfg.cls, next);
    setExpanded(next);
    try {
      const { getCurrentWindow, currentMonitor, PhysicalSize, PhysicalPosition } =
        await import("@tauri-apps/api/window");
      const win = getCurrentWindow();
      const scale = await win.scaleFactor();
      const size = await win.innerSize(); // physical; preserve current width

      if (next) {
        const monitor = await currentMonitor();
        if (monitor) {
          const topInset = Math.round(TOP_INSET * scale);
          const bottomMargin = Math.round(BOTTOM_MARGIN * scale);
          const fullHeight = Math.max(
            Math.round(cfg.compact * scale),
            monitor.size.height - topInset - bottomMargin,
          );
          // Anchor to the top of the current monitor so it doesn't run off-screen.
          const pos = await win.outerPosition(); // physical
          await win.setPosition(
            new PhysicalPosition(pos.x, monitor.position.y + topInset),
          );
          await win.setSize(new PhysicalSize(size.width, fullHeight));
        }
      } else {
        await win.setSize(
          new PhysicalSize(size.width, Math.round(cfg.compact * scale)),
        );
      }
    } catch {
      // no-op outside Tauri
    }
  }, [expanded, kind]);

  const showOther = useCallback(() => {
    // Chat summons Live Capture (no key focus); Capture summons Chat (key focus
    // so the composer is ready to type).
    const panel = kind === "chat" ? CAPTURE_WINDOW_LABEL : CHAT_WINDOW_LABEL;
    const makeKey = kind !== "chat";
    import("@tauri-apps/api/core")
      .then(({ invoke }) => invoke("focus_stealth_panel", { panel, makeKey }))
      .catch(() => {});
  }, [kind]);

  const hide = useCallback(() => {
    import("@tauri-apps/api/window")
      .then(({ getCurrentWindow }) => getCurrentWindow().hide())
      .catch(() => {});
  }, []);

  const turnOffStealth = useCallback(() => {
    import("../../utils/screenShareVisibility")
      .then(({ setHideFromScreenShare }) => setHideFromScreenShare(false))
      .catch(() => {});
  }, []);

  // Exit just the compact overlay panels, restoring the normal window while
  // leaving stealth (capture exclusion, no Dock/menu bar icon) on.
  const exitCompact = useCallback(() => {
    import("../../utils/compactMode")
      .then(({ setCompactMode }) => setCompactMode(false))
      .catch(() => {});
  }, []);

  const label = kind === "chat" ? "Chat" : "Live Capture";

  return (
    <div
      data-tauri-drag-region
      className="flex h-8 shrink-0 items-center gap-2 border-b border-white/5 bg-th-bg-secondary/50 pl-3 pr-2 backdrop-blur-2xl select-none"
    >
      {/* Grab handle + label — a wide draggable region so the panel can be
          dragged anywhere across the screen. */}
      <div
        data-tauri-drag-region
        className="flex min-w-0 flex-1 items-center gap-1.5 cursor-grab active:cursor-grabbing"
      >
        <span
          data-tauri-drag-region
          className="h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-400/90 shadow-[0_0_6px_rgba(52,211,153,0.7)]"
        />
        <span
          data-tauri-drag-region
          className="truncate text-[11px] font-medium tracking-tight text-th-text-secondary"
        >
          {label}
        </span>
      </div>

      {/* Controls never shrink, so they can't get squashed on a narrow panel. */}
      <div className="flex shrink-0 items-center gap-0.5">
        <TitleButton
          onClick={showOther}
          title={kind === "chat" ? "Show Live Capture panel" : "Show Chat panel"}
        >
          {kind === "chat" ? <Radio size={13} /> : <MessageSquare size={13} />}
        </TitleButton>
        <TitleButton
          onClick={() => void toggleExpand()}
          title={
            expanded
              ? kind === "chat"
                ? "Collapse to latest message"
                : "Collapse to latest transcription"
              : kind === "chat"
                ? "Expand to show message history"
                : "Expand to show more transcriptions"
          }
        >
          {expanded ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
        </TitleButton>
        <TitleButton onClick={hide} title={"Hide this panel (\u2318\u21e7\\ to bring back)"}>
          <Minus size={13} />
        </TitleButton>
        <TitleButton
          onClick={exitCompact}
          title="Exit compact mode (keeps stealth on, restores the normal window)"
        >
          <AppWindow size={13} />
        </TitleButton>
        <TitleButton
          onClick={turnOffStealth}
          title="Turn off stealth mode and restore the normal window"
          tone="danger"
        >
          <Power size={13} />
        </TitleButton>
      </div>
    </div>
  );
}

function TitleButton({
  onClick,
  title,
  children,
  tone = "default",
}: {
  onClick: () => void;
  title: string;
  children: React.ReactNode;
  tone?: "default" | "danger";
}) {
  const toneClasses =
    tone === "danger"
      ? "text-th-text-muted hover:bg-rose-500/15 hover:text-rose-400"
      : "text-th-text-muted hover:bg-white/10 hover:text-th-text-secondary";
  return (
    <button
      onClick={onClick}
      title={title}
      className={`flex h-6 w-6 items-center justify-center rounded-md transition-colors ${toneClasses}`}
    >
      {children}
    </button>
  );
}
