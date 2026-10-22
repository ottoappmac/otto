import { useCallback, useEffect, useRef, useState } from "react";
import { AlertTriangle, Bot, ExternalLink, MessageSquarePlus, Play, Plus, Settings2, Square, SquareTerminal, X } from "lucide-react";
import { api } from "../../hooks/useApi";
import type { DevState, FolderGrant, TerminalInfo } from "../../types";
import { makeSnippet, type Snippet } from "../../utils/terminalClip";
import { Kbd } from "../ui/Kbd";
import { Badge } from "../ui/Badge";
import { Button, IconButton } from "../ui/Button";
import { XTermView, type XTermHandle } from "./XTermView";

interface TerminalPanelProps {
  sessionId: string;
  /** The project folder's saved settings, for the dev server. */
  grant: FolderGrant | null;
  onGrantChange: (grant: FolderGrant) => void;
  /** Put terminal text in the chat box as a chip. */
  onAddToChat: (snippet: Snippet, prefill?: string) => void;
  /** Reports whether any process is running, for the header chip. */
  onActivity?: (running: number) => void;
}

const POLL_MS = 2500;

function errorText(error: unknown): string {
  const text = error instanceof Error ? error.message : "Request failed";
  const at = text.indexOf("{");
  if (at >= 0) {
    try {
      const body = JSON.parse(text.slice(at)) as { error?: string };
      if (body.error) return body.error;
    } catch { /* raw */ }
  }
  return text;
}

/** Shells the user types in, and the commands the agent started, side by side. */
export function TerminalPanel({ sessionId, grant, onGrantChange, onAddToChat, onActivity }: TerminalPanelProps) {
  const [terminals, setTerminals] = useState<TerminalInfo[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selection, setSelection] = useState("");
  const [dev, setDev] = useState<DevState | null>(null);
  const [devOpen, setDevOpen] = useState(false);
  const [devDraft, setDevDraft] = useState("");
  const [starting, setStarting] = useState(false);
  const view = useRef<XTermHandle>(null);
  const activeRef = useRef(activeId);
  activeRef.current = activeId;

  const refresh = useCallback(async () => {
    try {
      const { terminals: list } = await api.listTerminals(sessionId);
      setTerminals(list);
      api.getDevState(sessionId).then(setDev).catch(() => setDev(null));
      onActivity?.(list.filter((item) => item.running && item.kind === "process").length);
      setActiveId((current) => (current && list.some((item) => item.id === current) ? current : list[0]?.id ?? null));
    } catch { /* the backend may be restarting */ }
  }, [sessionId, onActivity]);

  useEffect(() => {
    setTerminals([]);
    setActiveId(null);
    setSelection("");
    setError(null);
    void refresh();
    const timer = window.setInterval(() => { void refresh(); }, POLL_MS);
    return () => window.clearInterval(timer);
  }, [sessionId, refresh]);

  const openShell = async () => {
    setError(null);
    try {
      const info = await api.openTerminal(sessionId, { rows: 24, cols: 100 });
      setTerminals((prev) => [...prev, { ...info, ports: info.ports ?? [] }]);
      setActiveId(info.id);
    } catch (e) {
      setError(errorText(e));
    }
  };

  const close = async (id: string) => {
    try {
      await api.closeTerminal(sessionId, id);
    } catch (e) {
      setError(errorText(e));
      return;
    }
    setTerminals((prev) => prev.filter((item) => item.id !== id));
    if (activeRef.current === id) setActiveId(null);
    void refresh();
  };

  const active = terminals.find((item) => item.id === activeId) ?? null;
  const failed = active ? !active.running && active.exit_code != null && active.exit_code !== 0 : false;

  const sourceLabel = active ? (active.kind === "process" ? active.command ?? active.title : "Terminal") : "Terminal";

  const addSelection = (text: string) => {
    if (!text.trim()) return;
    onAddToChat(makeSnippet(sourceLabel, text));
  };

  const addOutput = () => {
    const tail = view.current?.readTail(60) ?? "";
    if (!tail.trim()) return;
    onAddToChat(makeSnippet(sourceLabel, tail), failed ? "Find the cause of this error and fix it." : undefined);
  };

  const startDev = async () => {
    setStarting(true);
    setError(null);
    try {
      const state = await api.startDev(sessionId);
      setDev(state);
      await refresh();
      if (state.running_id) setActiveId(state.running_id);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setStarting(false);
    }
  };

  const saveGrant = async (patch: { dev_command?: string; dev_autostart?: boolean; network?: boolean }) => {
    if (!grant) return;
    try {
      onGrantChange(await api.updateFolderGrant(grant.id, patch));
      void refresh();
    } catch (e) {
      setError(errorText(e));
    }
  };

  return (
    <div className="flex h-full min-h-0 flex-col gap-2">
      {dev && (dev.command || grant) && (
        <div className="rounded-xl border border-th-border bg-th-surface px-2.5 py-2 text-2xs">
          <div className="flex items-center gap-2">
            <span className="shrink-0 font-medium text-th-text-primary">Dev server</span>
            <code className="min-w-0 flex-1 truncate text-th-text-muted" title={dev.command || "No command"}>
              {dev.command || "No command detected"}
            </code>
            {dev.running_id ? (
              <Badge tone="success">Running</Badge>
            ) : (
              <Button variant="secondary" disabled={!dev.command || starting} onClick={() => { void startDev(); }}>
                <Play size={11} /> {starting ? "Starting…" : "Start"}
              </Button>
            )}
            <IconButton
              label="Dev server settings"
              icon={<Settings2 size={13} />}
              aria-expanded={devOpen}
              onClick={() => { setDevOpen((open) => !open); setDevDraft(dev.configured ? dev.command : ""); }}
            />
          </div>
          {devOpen && grant && (
            <div className="mt-2 space-y-2 border-t border-th-border pt-2">
              <label className="block">
                <span className="text-th-text-muted">Command (leave empty to detect)</span>
                <input
                  value={devDraft}
                  onChange={(event) => setDevDraft(event.target.value)}
                  onBlur={() => { if (devDraft !== (dev.configured ? dev.command : "")) void saveGrant({ dev_command: devDraft }); }}
                  onKeyDown={(event) => { if (event.key === "Enter") (event.target as HTMLInputElement).blur(); }}
                  placeholder={dev.detected || "npm run dev"}
                  spellCheck={false}
                  className="mt-1 w-full rounded-lg border border-th-input-border bg-th-input-bg px-2 py-1 font-mono text-2xs text-th-text-primary"
                />
              </label>
              <label className="flex items-center gap-2 text-th-text-secondary">
                <input
                  type="checkbox"
                  checked={Boolean(grant.dev_autostart)}
                  onChange={(event) => { void saveGrant({ dev_autostart: event.target.checked }); }}
                />
                Start automatically when a session opens in this project
              </label>
              {!grant.network && (
                <p className="flex items-center gap-2 text-th-warning">
                  <span className="flex-1">A server needs Network on for this project to listen on a port.</span>
                  <Button variant="secondary" onClick={() => { void saveGrant({ network: true }); }}>Turn on</Button>
                </p>
              )}
            </div>
          )}
        </div>
      )}
      <div className="flex items-center gap-1" role="tablist" aria-label="Terminals">
        <div className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto">
          {terminals.map((item) => (
            <button
              key={item.id}
              type="button"
              role="tab"
              aria-selected={item.id === activeId}
              onClick={() => setActiveId(item.id)}
              title={item.command ?? item.title}
              className={`flex shrink-0 items-center gap-1.5 rounded-lg px-2 py-1 text-2xs transition-colors duration-fast ${
                item.id === activeId ? "bg-th-surface-hover text-th-text-primary" : "text-th-text-muted hover:text-th-text-secondary"
              }`}
            >
              {item.kind === "process" ? <Bot size={11} /> : <SquareTerminal size={11} />}
              <span className="max-w-[9rem] truncate">{item.title}</span>
              <span
                className={`h-1.5 w-1.5 rounded-full ${item.running ? "bg-th-success" : item.exit_code ? "bg-th-danger" : "bg-th-text-faint"}`}
                role="img"
                aria-label={item.running ? "Running" : `Exited${item.exit_code != null ? ` with code ${item.exit_code}` : ""}`}
              />
            </button>
          ))}
        </div>
        <IconButton label="New terminal" icon={<Plus size={14} />} onClick={() => { void openShell(); }} />
      </div>

      {error && (
        <div role="alert" className="flex items-start gap-2 rounded-lg border border-th-danger/25 bg-th-danger/10 px-2.5 py-2 text-2xs text-th-danger">
          <AlertTriangle size={13} className="mt-px shrink-0" />
          <span className="min-w-0 flex-1 break-words">{error}</span>
          <IconButton label="Dismiss" icon={<X size={11} />} onClick={() => setError(null)} className="!h-4 !w-4" />
        </div>
      )}

      {!active ? (
        <div className="flex flex-1 flex-col items-center justify-center gap-2 text-center text-th-text-muted">
          <SquareTerminal size={22} />
          <p className="max-w-[16rem] text-xs">
            Open a terminal to run commands in the project. Servers the agent starts show up here too.
          </p>
          <Button variant="secondary" onClick={() => { void openShell(); }}><Plus size={12} /> New terminal</Button>
        </div>
      ) : (
        <>
          <div className="flex flex-wrap items-center gap-1.5 text-2xs">
            {active.kind === "process" && <Badge tone="accent"><Bot size={10} /> Started by the agent</Badge>}
            {active.running ? <Badge tone="success">Running</Badge> : <Badge tone={failed ? "danger" : "neutral"}>{failed ? `Exited ${active.exit_code}` : "Exited"}</Badge>}
            {(active.ports ?? []).map((port) => (
              <a
                key={port}
                href={`http://localhost:${port}`}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 rounded-full bg-th-accent/12 px-2 py-0.5 font-medium text-th-accent hover:underline"
                title="Open in your browser"
              >
                localhost:{port} <ExternalLink size={10} />
              </a>
            ))}
            <span className="ml-auto flex items-center gap-1">
              {selection.trim() ? (
                <Button variant="secondary" onClick={() => addSelection(selection)} title="Add the highlighted text to the chat">
                  <MessageSquarePlus size={12} /> Add to chat <Kbd>⌘</Kbd><Kbd>L</Kbd>
                </Button>
              ) : (
                <Button variant={failed ? "secondary" : "ghost"} onClick={addOutput} title="Add the latest output to the chat">
                  <MessageSquarePlus size={12} /> {failed ? "Fix this" : "Add output"}
                </Button>
              )}
              <Button variant="ghost" className="hover:!text-th-danger" onClick={() => { void close(active.id); }}>
                <Square size={11} /> {active.running ? "Stop" : "Close"}
              </Button>
            </span>
          </div>
          <div className="min-h-[10rem] flex-1">
            <XTermView
              key={active.id}
              ref={view}
              sessionId={sessionId}
              terminalId={active.id}
              interactive={active.kind === "shell"}
              onSelectionChange={setSelection}
              onAddSelection={addSelection}
              onExit={() => { void refresh(); }}
            />
          </div>
        </>
      )}
    </div>
  );
}
