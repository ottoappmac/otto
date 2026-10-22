import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import { Terminal } from "@xterm/xterm";
import { FitAddon } from "@xterm/addon-fit";
import "@xterm/xterm/css/xterm.css";
import { wsUrl } from "../../config/apiBase";
import { useTheme } from "../../context/ThemeContext";
import { rememberTerminalCopy } from "../../utils/terminalClip";

export interface XTermHandle {
  /** The last `lines` rows of the screen and scrollback, as plain text. */
  readTail: (lines: number) => string;
  focus: () => void;
  getSelection: () => string;
}

interface XTermViewProps {
  sessionId: string;
  terminalId: string;
  /** Only a shell takes keyboard input. A process view is read-only except Ctrl+C. */
  interactive: boolean;
  onExit?: (code: number | null) => void;
  /** Fires as the highlighted text changes. Empty when nothing is selected. */
  onSelectionChange?: (text: string) => void;
  /** ⌘L with text highlighted. */
  onAddSelection?: (text: string) => void;
}

function cssRgb(name: string, fallback: string): string {
  const raw = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return raw ? `rgb(${raw.split(/\s+/).slice(0, 3).join(",")})` : fallback;
}

function palette(dark: boolean) {
  return {
    background: cssRgb("--color-code-bg", dark ? "#0e0e0e" : "#f2f2f7"),
    foreground: cssRgb("--color-text-primary", dark ? "#f5f5f5" : "#0f0f12"),
    cursor: cssRgb("--color-accent", "#0a84ff"),
    selectionBackground: dark ? "rgba(10,132,255,0.35)" : "rgba(0,113,227,0.25)",
  };
}

/** A live terminal bound to one backend PTY over a WebSocket. */
export const XTermView = forwardRef<XTermHandle, XTermViewProps>(function XTermView(
  { sessionId, terminalId, interactive, onExit, onSelectionChange, onAddSelection },
  ref,
) {
  const host = useRef<HTMLDivElement>(null);
  const term = useRef<Terminal | null>(null);
  const { theme } = useTheme();
  const onExitRef = useRef(onExit);
  onExitRef.current = onExit;
  const selectionRef = useRef(onSelectionChange);
  selectionRef.current = onSelectionChange;
  const addRef = useRef(onAddSelection);
  addRef.current = onAddSelection;

  useImperativeHandle(ref, () => ({
    readTail: (lines: number) => {
      const buffer = term.current?.buffer.active;
      if (!buffer) return "";
      const out: string[] = [];
      for (let i = Math.max(0, buffer.length - lines); i < buffer.length; i += 1) {
        out.push(buffer.getLine(i)?.translateToString(true) ?? "");
      }
      return out.join("\n").replace(/\n+$/, "");
    },
    focus: () => term.current?.focus(),
    getSelection: () => term.current?.getSelection() ?? "",
  }));

  useEffect(() => {
    const node = host.current;
    if (!node) return;
    const xterm = new Terminal({
      fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
      fontSize: 12,
      lineHeight: 1.25,
      cursorBlink: interactive,
      disableStdin: !interactive,
      scrollback: 5000,
      allowProposedApi: false,
      theme: palette(theme === "dark"),
    });
    const fit = new FitAddon();
    xterm.loadAddon(fit);
    xterm.open(node);
    term.current = xterm;
    if (interactive) xterm.focus();
    // Copying from a terminal marks the text, so a paste into the chat becomes a chip.
    const onCopy = () => rememberTerminalCopy(xterm.getSelection());
    node.addEventListener("copy", onCopy);
    const selection = xterm.onSelectionChange(() => selectionRef.current?.(xterm.getSelection()));
    xterm.attachCustomKeyEventHandler((event) => {
      const addKey = event.key.toLowerCase() === "l" && (event.metaKey || (event.ctrlKey && event.shiftKey));
      if (event.type === "keydown" && addKey && xterm.hasSelection()) {
        event.preventDefault();
        addRef.current?.(xterm.getSelection());
        return false;
      }
      return true;
    });
    const doFit = () => {
      try { fit.fit(); } catch { /* hidden or zero-sized */ }
    };
    doFit();

    const socket = new WebSocket(wsUrl(`/ws/terminal/${encodeURIComponent(sessionId)}/${encodeURIComponent(terminalId)}`));
    socket.binaryType = "arraybuffer";
    const send = (message: Record<string, unknown>) => {
      if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
    };
    socket.onopen = () => send({ type: "resize", rows: xterm.rows, cols: xterm.cols });
    socket.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        xterm.write(new Uint8Array(event.data));
        return;
      }
      try {
        const message = JSON.parse(String(event.data)) as { type?: string; code?: number | null; content?: string };
        if (message.type === "exit") {
          xterm.write(`\r\n\x1b[2m[process exited${message.code != null ? ` with code ${message.code}` : ""}]\x1b[0m\r\n`);
          onExitRef.current?.(message.code ?? null);
        } else if (message.type === "error") {
          xterm.write(`\r\n\x1b[31m${message.content ?? "Terminal unavailable"}\x1b[0m\r\n`);
        }
      } catch { /* ignore */ }
    };
    const data = xterm.onData((chunk) => {
      if (interactive || chunk === "\x03") send({ type: interactive ? "input" : "interrupt", data: chunk });
    });
    const resize = xterm.onResize(({ rows, cols }) => send({ type: "resize", rows, cols }));
    const observer = new ResizeObserver(doFit);
    observer.observe(node);

    return () => {
      observer.disconnect();
      node.removeEventListener("copy", onCopy);
      selection.dispose();
      data.dispose();
      resize.dispose();
      socket.close();
      xterm.dispose();
      term.current = null;
    };
  }, [sessionId, terminalId, interactive]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (term.current) term.current.options.theme = palette(theme === "dark");
  }, [theme]);

  return <div ref={host} className="h-full w-full overflow-hidden rounded-lg bg-th-code-bg p-1.5" />;
});
