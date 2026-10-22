import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { getCurrentWindow } from "@tauri-apps/api/window";
import {
  AlertTriangle, ArrowDown, Bot, Bug, Check, ClipboardList, Clock, Copy, ExternalLink, FileDiff, FileMinus2, FilePen, FilePlus2,
  FileText, Folder, FolderPlus, GitBranch, Globe, Hammer, HelpCircle, Lock, Maximize2, MessageSquare, Minimize2, PanelBottom, PanelLeft, PanelRight, Plus,
  RotateCcw, SendHorizontal, Square, SquareTerminal, Terminal, Trash2, Undo2, X,
} from "lucide-react";
import { Badge, type BadgeTone } from "../components/ui/Badge";
import { Button, IconButton } from "../components/ui/Button";
import { Kbd } from "../components/ui/Kbd";
import { Segmented } from "../components/ui/Segmented";
import { Toast } from "../components/ui/Toast";
import { fuzzyMatch } from "../utils/fuzzy";
import { formatFileSize } from "../utils/formatFileSize";
import { api } from "../hooks/useApi";
import { usePolling } from "../hooks/usePolling";
import { useWebSocket } from "../hooks/useWebSocket";
import { useChatModel } from "../hooks/useChatModel";
import { ModelPicker } from "../components/chat/ModelPicker";
import WorkspaceTree from "../components/chat/WorkspaceTree";
import { MessageBubble } from "../components/chat/MessageBubble";
import { MessageList } from "../components/chat/MessageList";
import { ThinkingIndicator } from "../components/chat/ThinkingIndicator";
import { ArtifactPanel, artifactTypeFromPath } from "../components/chat/ArtifactPanel";
import type { Artifact, ArtifactType } from "../components/chat/ArtifactPanel";
import { RunTimeline } from "../components/runs/RunTimeline";
import { applyLiveTimelineMessage } from "../utils/liveTimeline";
import { applyLiveMessage, isActivityFrame, syncFromApi } from "../utils/liveMessages";
import { buildRenderItems, computeLatestTodoFlags } from "../utils/renderItems";
import {
  computeThoughtFlags,
  hasUnmatchedApprovalTool,
  mergeToolMessages,
  pendingInterrupts,
} from "../utils/mergeToolMessages";
import { screenHighRiskCommand } from "../utils/highRiskCommands";
import { useNotification } from "../context/NotificationContext";
import { pickDirectory } from "../utils/pickFolder";
import { FolderGrantDialog } from "../components/action/FolderGrantDialog";
import { PatchView } from "../components/code/PatchView";
import { FileEditor } from "../components/code/FileEditor";
import { TerminalPanel } from "../components/code/TerminalPanel";
import { splitDroppedPaths } from "../utils/droppedPaths";
import { formatSnippets, isTerminalPaste, makeSnippet, type Snippet } from "../utils/terminalClip";
import InlineUrlInput, { type InlineUrlInputHandle } from "../components/chat/InlineUrlInput";
import type {
  ChatMessage,
  FolderGrant,
  FolderGrantPreview,
  SessionInfo,
  SessionMode,
  TimelineEvent,
  WorkspaceChanges,
  WorkspaceGitStatus,
  WSMessage,
} from "../types";

type Inspector = "changes" | "files" | "terminal" | "output" | "timeline" | "plan";

const LAST_GRANT = "otto.code.lastGrant";
const LAST_MODE = "otto.code.mode";
const LEFT_WIDTH_KEY = "otto.code.leftWidth";
const RIGHT_WIDTH_KEY = "otto.code.rightWidth";
const PLAN_PATH = "output/plan.md";
const SESSIONS_POLL_MS = 4000;
const LEFT_DEFAULT = 176;
const RIGHT_DEFAULT = 288;
const LEFT_MIN = 160;
const LEFT_MAX = 480;
const RIGHT_MIN = 220;
const RIGHT_MAX = 720;
const CENTER_MIN = 280;
const DOCK_KEY = "otto.code.dock";
const BOTTOM_HEIGHT_KEY = "otto.code.bottomHeight";
const BOTTOM_MIN = 160;
// Below this width a side inspector would squeeze the conversation, so it docks under it.
const SIDE_DOCK_MIN_WIDTH = 940;

function storedWidth(key: string, fallback: number, min: number, max: number): number {
  const raw = Number(localStorage.getItem(key));
  if (!Number.isFinite(raw)) return fallback;
  return Math.min(max, Math.max(min, raw));
}

const MODES: { id: SessionMode; label: string; hint: string; icon: ReactNode }[] = [
  { id: "agent", label: "Agent", hint: "Reads and edits the project, runs commands.", icon: <Bot size={12} /> },
  { id: "ask", label: "Ask", hint: "Read-only. Answers questions about the code and cannot change anything.", icon: <HelpCircle size={12} /> },
  { id: "plan", label: "Plan", hint: "Read-only in the project. Saves the plan to a markdown file, then you build it.", icon: <ClipboardList size={12} /> },
  { id: "debug", label: "Debug", hint: "Investigates first: reproduce, hypothesize, log, fix, clean up.", icon: <Bug size={12} /> },
];

function asMode(value: unknown): SessionMode {
  return value === "ask" || value === "plan" || value === "debug" ? value : "agent";
}

function modeBadge(mode: SessionMode): { label: string; tone: BadgeTone } | null {
  if (mode === "ask") return { label: "Ask mode", tone: "accent" };
  if (mode === "plan") return { label: "Plan mode", tone: "accent" };
  if (mode === "debug") return { label: "Debug mode", tone: "warning" };
  return null;
}

function dayBucket(iso: string | undefined): string {
  const then = iso ? new Date(iso) : null;
  if (!then || Number.isNaN(then.getTime())) return "Earlier";
  const start = new Date();
  start.setHours(0, 0, 0, 0);
  const days = Math.floor((start.getTime() - new Date(then).setHours(0, 0, 0, 0)) / 86400000);
  if (days <= 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days < 7) return "This week";
  return "Earlier";
}

function splitPath(path: string): { name: string; dir: string } {
  const at = path.lastIndexOf("/");
  return at < 0 ? { name: path, dir: "" } : { name: path.slice(at + 1), dir: path.slice(0, at) };
}

function highlightMatch(text: string, query: string): ReactNode {
  const match = query ? fuzzyMatch(query, text) : null;
  if (!match || match.indices.length === 0) return text;
  const hit = new Set(match.indices);
  return Array.from(text).map((ch, i) => (hit.has(i) ? <mark key={i} className="bg-transparent font-semibold text-th-accent">{ch}</mark> : ch));
}

const STARTERS: Record<SessionMode, string[]> = {
  agent: ["Add a small improvement and test it", "Find a bug and fix it with a regression test"],
  ask: ["Explain how this project is structured", "Where is the main entry point?"],
  plan: ["Plan how to add a feature", "Plan a refactor of the busiest module"],
  debug: ["Reproduce and fix a failing test", "Investigate an error I will paste"],
};

function relTime(iso: string | undefined): string {
  if (!iso) return "";
  const then = new Date(iso).getTime();
  if (!Number.isFinite(then)) return "";
  const mins = Math.round((Date.now() - then) / 60000);
  if (mins < 1) return "now";
  if (mins < 60) return `${mins}m`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h`;
  return `${Math.round(hours / 24)}d`;
}

function describeRevert(result: { restored?: string[]; conflicts?: { path: string; reason: string }[] }): string {
  const restored = result.restored?.length ?? 0;
  const conflicts = result.conflicts ?? [];
  if (conflicts.length === 0) {
    return restored ? `Restored ${restored} file${restored === 1 ? "" : "s"}.` : "Nothing to restore.";
  }
  const first = conflicts[0];
  return `Restored ${restored}, kept ${conflicts.length}: ${first.path} — ${first.reason}`;
}

function revertFailure(error: unknown): string {
  const text = error instanceof Error ? error.message : "";
  const at = text.indexOf("{");
  if (at >= 0) {
    try {
      const body = JSON.parse(text.slice(at)) as { restored?: string[]; conflicts?: { path: string; reason: string }[]; error?: string };
      if (body.conflicts || body.restored) return describeRevert(body);
      if (body.error) return body.error;
    } catch { /* raw */ }
  }
  return apiMessage(error);
}

function userOrdinal(messages: ChatMessage[], index: number): number {
  let count = 0;
  for (let i = 0; i < index && i < messages.length; i += 1) {
    const item = messages[i];
    if (item.type === "user" && !item.metadata?.isContext) count += 1;
  }
  return count;
}

function commandExit(text: string): number | null {
  const match = text.match(/Exit code:\s*(-?\d+)\s*$/);
  return match ? Number(match[1]) : null;
}

function apiMessage(error: unknown): string {
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

export default function CodePage() {
  const { sessionId } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  const [grants, setGrants] = useState<FolderGrant[]>([]);
  const [grantId, setGrantId] = useState<string>(() => localStorage.getItem(LAST_GRANT) || "");
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [inspector, setInspector] = useState<Inspector>((searchParams.get("tab") as Inspector) || "changes");
  const [changes, setChanges] = useState<WorkspaceChanges | null>(null);
  const [git, setGit] = useState<WorkspaceGitStatus | null>(null);
  const [diffPath, setDiffPath] = useState<string | null>(searchParams.get("file"));
  const [diff, setDiff] = useState("");
  const [artifact, setArtifact] = useState<Artifact | null>(null);
  const [timeline, setTimeline] = useState<TimelineEvent[]>([]);
  const [live, setLive] = useState<TimelineEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<FolderGrantPreview | null>(null);
  const [grantMode, setGrantMode] = useState<"read" | "write">("write");
  const [notice, setNotice] = useState<string | null>(null);
  const [mode, setMode] = useState<SessionMode>(() => asMode(localStorage.getItem(LAST_MODE)));
  const [hasPlan, setHasPlan] = useState(false);
  const [planRev, setPlanRev] = useState(0);
  const [changeTurn, setChangeTurn] = useState<number | null>(null);
  const [diffTruncated, setDiffTruncated] = useState(false);
  const [revertNote, setRevertNote] = useState<string | null>(null);
  const [confirmRevert, setConfirmRevert] = useState(false);
  const [confirmGrantId, setConfirmGrantId] = useState<string | null>(null);
  const [keptKey, setKeptKey] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [hiddenIds, setHiddenIds] = useState<string[]>([]);
  const [undoDelete, setUndoDelete] = useState<{ id: string; title: string } | null>(null);
  const pendingDelete = useRef<{ id: string; timer: number } | null>(null);
  const [showJump, setShowJump] = useState(false);
  const [runningProcs, setRunningProcs] = useState(0);
  const [narrow, setNarrow] = useState(false);
  const [dockPref, setDockPref] = useState<"right" | "bottom">(() => (localStorage.getItem(DOCK_KEY) === "bottom" ? "bottom" : "right"));
  const [bottomHeight, setBottomHeight] = useState(() => storedWidth(BOTTOM_HEIGHT_KEY, 300, BOTTOM_MIN, 900));
  const [maximized, setMaximized] = useState(false);
  const [shellWidth, setShellWidth] = useState(0);
  const dock: "right" | "bottom" = dockPref === "bottom" || (shellWidth > 0 && shellWidth < SIDE_DOCK_MIN_WIDTH) ? "bottom" : "right";
  const compact = shellWidth > 0 && shellWidth < 760;
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [projectsOpen, setProjectsOpen] = useState(true);
  const [mentions, setMentions] = useState<{ path: string; is_dir: boolean }[]>([]);
  const [atOpen, setAtOpen] = useState(false);
  const [atFilter, setAtFilter] = useState("");
  const [atIndex, setAtIndex] = useState(0);
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [pendingFilePaths, setPendingFilePaths] = useState<string[]>([]);
  const [pendingFolders, setPendingFolders] = useState<string[]>([]);
  const [pendingSnippets, setPendingSnippets] = useState<Snippet[]>([]);
  const [openSnippet, setOpenSnippet] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const composerRef = useRef<InlineUrlInputHandle>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const hasAttachments = pendingFiles.length > 0 || pendingFilePaths.length > 0 || pendingFolders.length > 0 || pendingSnippets.length > 0;
  // Same model selection as Chat: it changes the global model in Settings.
  // A model change does not discard the open coding session; the backend
  // notices the new model at the start of the next turn and rebuilds the
  // session's graph with it.
  const { currentModel, availableModels, appSettings, isModelBusy, handleModelChange } = useChatModel();
  const shellRef = useRef<HTMLDivElement>(null);
  const sessionLoad = useRef(0);
  const sessionIdRef = useRef(sessionId);
  sessionIdRef.current = sessionId;
  const msgIdRef = useRef(0);
  // Cards the user already answered. A second click, or an auto-approve pass
  // that runs before the answer is stamped, must not send the answer twice.
  const hitlSubmittedRef = useRef<Set<string>>(new Set());
  // After an answer is sent the backend needs a moment to start the resume.
  // A status poll that lands in that gap still says "awaiting input / not
  // running"; until this deadline it must not flip Stop back to Send.
  const hitlResumeUntilRef = useRef(0);
  const [awaitingPoll, setAwaitingPoll] = useState(false);
  const [starting, setStarting] = useState(false);
  const startingRef = useRef(false);
  const pendingSendRef = useRef<{ sid: string; text: string; mode: SessionMode } | null>(null);
  const [sessionAutoApprove, setSessionAutoApprove] = useState(false);
  const stickToBottomRef = useRef(true);
  const scrollRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { notify, clearSession, watchSession, unwatchSession } = useNotification();
  const [leftWidth, setLeftWidth] = useState(() => storedWidth(LEFT_WIDTH_KEY, LEFT_DEFAULT, LEFT_MIN, LEFT_MAX));
  const [rightWidth, setRightWidth] = useState(() => storedWidth(RIGHT_WIDTH_KEY, RIGHT_DEFAULT, RIGHT_MIN, RIGHT_MAX));
  const widthsRef = useRef({ left: leftWidth, right: rightWidth });
  widthsRef.current = { left: leftWidth, right: rightWidth };

  const fitWidth = useCallback((edge: "left" | "right", next: number) => {
    const total = shellRef.current?.clientWidth ?? window.innerWidth;
    const other = edge === "left" ? widthsRef.current.right : widthsRef.current.left;
    const min = edge === "left" ? LEFT_MIN : RIGHT_MIN;
    const max = edge === "left" ? LEFT_MAX : RIGHT_MAX;
    const room = total - other - CENTER_MIN - 12;
    return Math.round(Math.min(max, Math.max(min, Math.min(next, Math.max(min, room)))));
  }, []);

  const resizePanel = useCallback((edge: "left" | "right", next: number) => {
    const fitted = fitWidth(edge, next);
    if (edge === "left") setLeftWidth(fitted);
    else setRightWidth(fitted);
    localStorage.setItem(edge === "left" ? LEFT_WIDTH_KEY : RIGHT_WIDTH_KEY, String(fitted));
  }, [fitWidth]);

  const startResize = useCallback((edge: "left" | "right", event: React.PointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    const handle = event.currentTarget;
    handle.setPointerCapture(event.pointerId);
    const startX = event.clientX;
    const start = edge === "left" ? widthsRef.current.left : widthsRef.current.right;
    const previousCursor = document.body.style.cursor;
    const previousSelect = document.body.style.userSelect;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    const onMove = (ev: PointerEvent) => {
      const delta = ev.clientX - startX;
      resizePanel(edge, edge === "left" ? start + delta : start - delta);
    };
    const onUp = () => {
      handle.removeEventListener("pointermove", onMove);
      handle.removeEventListener("pointerup", onUp);
      handle.removeEventListener("pointercancel", onUp);
      document.body.style.cursor = previousCursor;
      document.body.style.userSelect = previousSelect;
    };
    handle.addEventListener("pointermove", onMove);
    handle.addEventListener("pointerup", onUp);
    handle.addEventListener("pointercancel", onUp);
  }, [resizePanel]);

  const loadGrants = useCallback(() => {
    api.listFolderGrants().then((rows) => {
      setGrants(rows);
      setGrantId((current) => (rows.some((row) => row.id === current) ? current : (rows[0]?.id || "")));
    }).catch((e) => setError(apiMessage(e)));
  }, []);

  useEffect(() => { loadGrants(); }, [loadGrants]);

  const loadSessions = useCallback(() => {
    if (!grantId) return;
    api.listSessionsForGrant(grantId).then(setSessions).catch(() => {});
  }, [grantId]);

  useEffect(() => {
    if (!grantId) {
      setSessions([]);
      return;
    }
    localStorage.setItem(LAST_GRANT, grantId);
    loadSessions();
  }, [grantId, loadSessions]);

  // Titles and the running dot change after a turn, and a chat started
  // elsewhere can bind this folder. Keep the list current without leaving Code.
  usePolling(loadSessions, SESSIONS_POLL_MS, Boolean(grantId));

  const pollProcesses = useCallback(async () => {
    if (!sessionId) {
      setRunningProcs(0);
      return;
    }
    try {
      const { terminals } = await api.listTerminals(sessionId);
      setRunningProcs(terminals.filter((item) => item.running && item.kind === "process").length);
    } catch {
      setRunningProcs(0);
    }
  }, [sessionId]);
  usePolling(pollProcesses, 4000, Boolean(sessionId));

  const reloadSession = useCallback((sid: string) => {
    const ticket = ++sessionLoad.current;
    const current = () => sessionLoad.current === ticket;
    api.getSession(sid).then((info) => { if (current()) setSession(info); }).catch(() => { if (current()) setSession(null); });
    api.getSessionStatus(sid).then((status) => {
      if (!current()) return;
      if (status.running) {
        setStreaming(true);
        watchSession(sid);
      }
      // A paused run has no live task, so the card has to come from the
      // saved transcript. The poll below keeps looking until it shows up.
      if (status.awaiting_input && !status.running) setAwaitingPoll(true);
    }).catch(() => {});
    api.getSessionMessages(sid).then((rows) => {
      if (!current()) return;
      const saved = mergeToolMessages(rows.map((row, index) => ({
        id: `${sid}-${index}`,
        type: (row.type as ChatMessage["type"]) || "agent",
        content: String(row.content ?? ""),
        metadata: row.metadata as ChatMessage["metadata"],
        timestamp: new Date(),
        sessionId: sid,
      })));
      // Disk is authoritative, with one exception: rows of this same session
      // that are on screen but not saved yet (a first message sent right after
      // the session was created) must not be wiped by a load that started
      // before they existed.
      setMessages((prev) => (
        saved.length < prev.length && prev.every((m) => m.sessionId === sid)
          ? syncFromApi(prev, saved)
          : saved
      ));
    }).catch(() => { if (current()) setMessages([]); });
    api.getSessionTimeline(sid).then((data) => { if (current()) setTimeline(data.events || []); }).catch(() => { if (current()) setTimeline([]); });
  }, [watchSession]);

  useEffect(() => {
    setLive([]);
    setStreaming(false);
    setAwaitingPoll(false);
    setSessionAutoApprove(false);
    hitlSubmittedRef.current = new Set();
    hitlResumeUntilRef.current = 0;
    stickToBottomRef.current = true;
    setNotice(null);
    setError(null);
    setChangeTurn(null);
    setKeptKey(null);
    setRevertNote(null);
    setConfirmRevert(false);
    if (!sessionId) {
      sessionLoad.current += 1;
      setSession(null);
      setMessages([]);
      setTimeline([]);
      return;
    }
    reloadSession(sessionId);
  }, [sessionId, reloadSession]);

  useEffect(() => {
    if (!session?.id) return;
    setSessions((prev) => prev.map((item) => (
      item.id === session.id
        ? { ...item, title: session.title, status: session.status, mode: session.mode, updated_at: session.updated_at }
        : item
    )));
  }, [session]);

  useEffect(() => {
    if (session?.workspace?.grant_id) setGrantId(session.workspace.grant_id);
  }, [session?.workspace?.grant_id]);

  useEffect(() => {
    if (session?.mode) setMode(asMode(session.mode));
  }, [session?.id, session?.mode]);

  const refreshPlan = useCallback(() => {
    if (!sessionId) {
      setHasPlan(false);
      return;
    }
    api.listSessionFiles(sessionId).then((rows) => {
      const found = rows.some((row) => row.path === PLAN_PATH);
      setHasPlan(found);
      if (found) setPlanRev(Date.now());
    }).catch(() => setHasPlan(false));
  }, [sessionId]);

  useEffect(() => { refreshPlan(); }, [refreshPlan]);

  const changeMode = useCallback(async (next: SessionMode) => {
    setMode(next);
    localStorage.setItem(LAST_MODE, next);
    if (next === "debug") setInspector("output");
    if (next === "plan" && hasPlan) setInspector("plan");
    if (!sessionId) return;
    try {
      setSession(await api.setSessionMode(sessionId, next));
    } catch (e) {
      setError(apiMessage(e));
    }
  }, [sessionId, hasPlan]);

  const refreshChanges = useCallback(() => {
    if (!sessionId || !session?.workspace?.grant_id) return;
    const turn = changeTurn ?? undefined;
    api.getWorkspaceChanges(sessionId, turn).then(setChanges).catch((e) => { setChanges(null); setError(apiMessage(e)); });
    api.getWorkspaceGit(sessionId).then(setGit).catch((e) => { setGit(null); setError(apiMessage(e)); });
  }, [sessionId, session?.workspace?.grant_id, changeTurn]);

  // A finished tool call is folded into its row, so count them: an edit that
  // completes does not add a message but should refresh the Changes tab.
  // During a run this waits briefly so a large repo is not diffed on every line.
  const finishedTools = useMemo(() => messages.filter((m) => m.type === "tool_result").length, [messages]);
  useEffect(() => {
    const delay = streaming ? 750 : 0;
    const timer = window.setTimeout(() => refreshChanges(), delay);
    return () => window.clearTimeout(timer);
  }, [refreshChanges, messages.length, finishedTools, streaming]);

  const changeSig = useMemo(
    () => (changes?.files || []).map((file) => `${file.status}:${file.path}:${file.added}:${file.deleted}`).join("|"),
    [changes],
  );
  useEffect(() => {
    if (!sessionId || !diffPath) {
      setDiff("");
      setDiffTruncated(false);
      return;
    }
    api.getWorkspaceDiff(sessionId, diffPath, changeTurn ?? undefined).then((row) => {
      setDiff(row.diff || row.message || "");
      setDiffTruncated(Boolean(row.truncated));
    }).catch((e) => { setDiff(""); setDiffTruncated(false); setError(apiMessage(e)); });
  }, [sessionId, diffPath, changeSig, changeTurn]);

  const onMessage = useCallback((msg: WSMessage) => {
    const sid = sessionIdRef.current ?? undefined;
    if (msg.type === "workspace_checkpoint" && msg.content) setNotice(msg.content);
    if (msg.type === "workspace_changed") refreshChanges();
    if (msg.type === "resync" && sid) reloadSession(sid);
    if (msg.type === "done" || msg.type === "stopped" || msg.type === "error") {
      setStreaming(false);
      setAwaitingPoll(false);
      hitlResumeUntilRef.current = 0;
      refreshPlan();
      if (sid) unwatchSession(sid);
      if (sid && msg.type === "done") notify("done", sid);
      // The title is written at the end of the turn. Pull the list now
      // instead of waiting for the next poll.
      loadSessions();
    } else if (msg.type === "hitl_request" || msg.type === "ask_user") {
      // The run is paused on the user. Show the card and stop "working".
      setStreaming(false);
      setAwaitingPoll(false);
      hitlResumeUntilRef.current = 0;
      stickToBottomRef.current = true;
      if (sid) { unwatchSession(sid); notify("hitl", sid); }
    } else if (isActivityFrame(msg)) {
      // Live output proves the agent is working, whatever a status poll that
      // raced the resume reported.
      setStreaming(true);
    }
    setMessages((prev) => applyLiveMessage(prev, msg, {
      nextId: () => `live-${++msgIdRef.current}`,
      sessionId: sid,
    }));
    setLive((prev) => applyLiveTimelineMessage(prev, msg));
  }, [refreshChanges, refreshPlan, reloadSession, loadSessions, notify, unwatchSession]);

  const { connected, send, sendEdit, sendContext, sendHitlResponse, waitForConnection } = useWebSocket({
    sessionId: sessionId || null,
    onMessage,
  });

  // Safety net for a dropped frame: while a run is going, or paused with its
  // card not on screen yet, compare against the saved transcript every 2s.
  // This is also what recovers the card after a reconnect.
  const syncWithServer = useCallback(async () => {
    const sid = sessionIdRef.current;
    if (!sid) return;
    const [status, rows] = await Promise.all([
      api.getSessionStatus(sid).catch(() => null),
      api.getSessionMessages(sid).catch(() => null),
    ]);
    if (sid !== sessionIdRef.current) return;
    let apiMerged: ChatMessage[] = [];
    if (rows && rows.length > 0) {
      apiMerged = mergeToolMessages(rows.map((row, index) => ({
        id: `${sid}-${index}`,
        type: (row.type as ChatMessage["type"]) || "agent",
        content: String(row.content ?? ""),
        metadata: row.metadata as ChatMessage["metadata"],
        timestamp: new Date(),
        sessionId: sid,
      })));
      setMessages((prev) => syncFromApi(prev, apiMerged));
    }
    // The user just answered a card and the resume is starting. Whatever this
    // poll saw predates it, so wait for the next one.
    if (Date.now() < hitlResumeUntilRef.current) return;
    if (pendingInterrupts(apiMerged).length > 0) {
      setStreaming(false);
      setAwaitingPoll(false);
      return;
    }
    if (!status) return;
    if (status.running) {
      setStreaming(true);
      setAwaitingPoll(false);
      return;
    }
    // Not running. A run paused on an approval is "awaiting input", and its
    // card may not be saved yet, so keep looking for it.
    const awaiting = status.awaiting_input || hasUnmatchedApprovalTool(apiMerged);
    setStreaming(false);
    setAwaitingPoll(awaiting);
    if (!awaiting) {
      unwatchSession(sid);
      refreshPlan();
      loadSessions();
    }
  }, [loadSessions, refreshPlan, unwatchSession]);

  useEffect(() => {
    const node = shellRef.current;
    if (!node) return;
    const observer = new ResizeObserver(() => setShellWidth(node.clientWidth));
    observer.observe(node);
    setShellWidth(node.clientWidth);
    return () => observer.disconnect();
  }, []);

  const chooseDock = (next: "right" | "bottom") => {
    setDockPref(next);
    setMaximized(false);
    try { localStorage.setItem(DOCK_KEY, next); } catch { /* storage unavailable */ }
  };

  const startBottomResize = (event: React.PointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    const handle = event.currentTarget;
    handle.setPointerCapture(event.pointerId);
    const startY = event.clientY;
    const start = bottomHeight;
    const limit = Math.max(BOTTOM_MIN, (shellRef.current?.clientHeight ?? 800) - 260);
    document.body.style.userSelect = "none";
    const onMove = (ev: PointerEvent) => {
      const next = Math.round(Math.min(limit, Math.max(BOTTOM_MIN, start - (ev.clientY - startY))));
      setBottomHeight(next);
      try { localStorage.setItem(BOTTOM_HEIGHT_KEY, String(next)); } catch { /* storage unavailable */ }
    };
    const onUp = () => {
      handle.removeEventListener("pointermove", onMove);
      handle.removeEventListener("pointerup", onUp);
      handle.removeEventListener("pointercancel", onUp);
      document.body.style.userSelect = "";
    };
    handle.addEventListener("pointermove", onMove);
    handle.addEventListener("pointerup", onUp);
    handle.addEventListener("pointercancel", onUp);
  };

  useEffect(() => {
    const query = window.matchMedia("(max-width: 1100px)");
    const apply = () => {
      setNarrow(query.matches);
      setProjectsOpen(!query.matches);
      setInspectorOpen(!query.matches);
    };
    apply();
    query.addEventListener("change", apply);
    return () => query.removeEventListener("change", apply);
  }, []);

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(null), 4000);
    return () => window.clearTimeout(timer);
  }, [toast]);

  useEffect(() => {
    if (!atOpen || !sessionId) {
      setMentions([]);
      return;
    }
    const timer = window.setTimeout(() => {
      api.searchWorkspace(sessionId, atFilter).then((row) => setMentions(row.paths || [])).catch(() => setMentions([]));
    }, 120);
    return () => window.clearTimeout(timer);
  }, [atOpen, atFilter, sessionId]);

  useEffect(() => {
    const next = new URLSearchParams(searchParams);
    if (inspector !== "changes") next.set("tab", inspector);
    else next.delete("tab");
    if (diffPath) next.set("file", diffPath);
    else next.delete("file");
    if (next.toString() !== searchParams.toString()) setSearchParams(next, { replace: true });
  }, [inspector, diffPath, searchParams, setSearchParams]);

  useEffect(() => {
    composerRef.current?.focus();
  }, [sessionId]);

  // Frames carry a seq. A dropped frame arrives as `resync`, which reloads
  // the transcript. The socket replay covers a reconnect. There is no
  // full-transcript poll while a run is in progress.

  // After the socket comes back, frames sent while it was down are gone.
  const wasConnected = useRef(false);
  useEffect(() => {
    const reconnected = connected && !wasConnected.current;
    wasConnected.current = connected;
    if (reconnected && sessionId) void syncWithServer();
  }, [connected, sessionId, syncWithServer]);

  const output = useMemo(() => {
    const rows: { id: string; command: string; text: string; done: boolean; turn: number }[] = [];
    let turn = 0;
    for (const m of messages) {
      if (m.type === "user" && !m.metadata?.isContext) {
        turn += 1;
        continue;
      }
      if (m.type !== "tool_call" && m.type !== "tool_result") continue;
      const isExecute = m.content === "execute" || m.metadata?.name === "execute";
      if (!isExecute) continue;
      const args = m.metadata?.args as { command?: unknown } | undefined;
      const command = typeof args?.command === "string" ? args.command : "";
      // A call row holds the live lines until its result lands in `result`.
      // A result with no call row keeps the output as its own content.
      const live = (m.metadata?.liveOutput as string[] | undefined)?.join("\n") ?? "";
      const text = typeof m.metadata?.result === "string"
        ? m.metadata.result
        : m.content !== "execute" ? m.content : live;
      rows.push({ id: m.id, command, text, done: m.type === "tool_result", turn });
    }
    return rows;
  }, [messages]);

  const thoughtFlags = useMemo(() => computeThoughtFlags(messages), [messages]);
  const latestTodoFlags = useMemo(() => computeLatestTodoFlags(messages), [messages]);
  const renderItems = useMemo(() => buildRenderItems(messages, thoughtFlags), [messages, thoughtFlags]);
  const restorable = useMemo(() => {
    const turns = new Set((changes?.checkpoints || []).map((item) => item.user_index).filter((value) => value != null));
    const indexes = new Set<number>();
    messages.forEach((message, index) => {
      if (message.type === "user" && !message.metadata?.isContext && turns.has(userOrdinal(messages, index))) {
        indexes.add(index);
      }
    });
    return indexes;
  }, [messages, changes]);
  const reviewStats = useMemo(() => {
    const files = changes?.files || [];
    const otto = files.filter((file) => file.origin !== "outside");
    return {
      count: otto.length,
      outside: files.length - otto.length,
      added: otto.reduce((sum, file) => sum + (file.added || 0), 0),
      deleted: otto.reduce((sum, file) => sum + (file.deleted || 0), 0),
    };
  }, [changes]);
  const reviewKey = changes?.checkpoint?.turn != null ? String(changes.checkpoint.turn) : null;
  const showReview = Boolean(
    sessionId && !streaming && reviewKey && keptKey !== reviewKey && (reviewStats.count > 0 || reviewStats.outside > 0),
  );
  const errorMessages = useMemo(() => messages.filter((m) => m.type === "error"), [messages]);
  const resolveSubagentModel = useCallback(
    () => ({ label: currentModel || "model", family: "inherit" }),
    [currentModel],
  );

  const handleHitlDecision = useCallback((
    messageId: string,
    decisions: Array<Record<string, unknown>>,
    requestId?: string,
  ) => {
    if (hitlSubmittedRef.current.has(messageId)) return;
    if (!sendHitlResponse(decisions, requestId)) {
      setError("Not connected to the server. Wait a moment and try again.");
      return;
    }
    hitlSubmittedRef.current.add(messageId);
    hitlResumeUntilRef.current = Date.now() + 5000;
    const sid = sessionIdRef.current;
    if (sid) { clearSession(sid); watchSession(sid); }
    stickToBottomRef.current = true;
    setError(null);
    setMessages((prev) => prev.map((m) => (
      m.id === messageId ? { ...m, metadata: { ...m.metadata, resolved: true, decisions } } : m
    )));
    setStreaming(true);
    setAwaitingPoll(false);
    setSessions((prev) => prev.map((item) => (item.id === sid ? { ...item, status: "running" } : item)));
  }, [sendHitlResponse, clearSession, watchSession]);

  const approveAllThisSession = useCallback((messageId: string, decisions: Array<Record<string, unknown>>) => {
    handleHitlDecision(messageId, decisions);
    setSessionAutoApprove(true);
  }, [handleHitlDecision]);

  // "Always allow" (or the global setting) answers later command approvals
  // without a click. Cards the server marks as always-ask, and high-risk
  // commands, still wait for the user.
  const autoApprove = Boolean(appSettings?.auto_approve_commands) || sessionAutoApprove;
  useEffect(() => {
    if (!autoApprove) return;
    const card = [...messages].reverse().find((m) => m.type === "hitl_request" && !m.metadata?.resolved);
    if (!card || hitlSubmittedRef.current.has(card.id)) return;
    const risk = card.metadata?.risk as { no_auto_approve?: boolean } | undefined;
    if (risk?.no_auto_approve) return;
    const actions = (card.metadata?.action_requests as Array<{ args?: Record<string, unknown> }> | undefined) ?? [];
    if (actions.some((a) => screenHighRiskCommand(a.args?.command).length > 0)) return;
    const requestId = typeof card.metadata?.request_id === "string" ? card.metadata.request_id : undefined;
    handleHitlDecision(card.id, actions.map(() => ({ type: "approve" })), requestId);
  }, [autoApprove, messages, handleHitlDecision]);

  const handleOpenArtifact = useCallback((path: string, fileUrl: string, type: ArtifactType) => {
    const norm = path.replace(/\\/g, "/").replace(/^\//, "");
    const hit = (changes?.files || []).find((file) => norm === file.path || norm.endsWith(`/${file.path}`));
    if (hit) {
      setDiffPath(hit.path);
      setInspector("changes");
      setInspectorOpen(true);
      return;
    }
    setArtifact({ path, fileUrl, type });
    setInspector("files");
    setInspectorOpen(true);
  }, [changes]);

  const handleStop = async () => {
    const sid = sessionIdRef.current;
    if (!sid) return;
    try {
      await api.stopSession(sid);
    } catch (e) {
      setError(apiMessage(e));
      return;
    }
    setStreaming(false);
  };

  // Follow the newest output unless the user scrolled up to read.
  useEffect(() => {
    if (stickToBottomRef.current) messagesEndRef.current?.scrollIntoView({ block: "end" });
  }, [messages, streaming]);

  const startSession = async (forGrantId: string = grantId): Promise<string | null> => {
    if (!forGrantId) return null;
    setError(null);
    let createdId = "";
    try {
      const created = await api.createSession({ agent_name: "coding-agent" });
      createdId = created.id;
      let bound = await api.bindSessionWorkspace(created.id, forGrantId);
      if (mode !== "agent") bound = await api.setSessionMode(created.id, mode);
      setSession(bound);
      setSessions((prev) => [bound, ...prev.filter((item) => item.id !== bound.id)]);
      navigate(`/code/${created.id}`);
      return created.id;
    } catch (e) {
      // Do not leave an empty, unbound session behind when binding fails.
      if (createdId) api.closeSession(createdId).catch(() => {});
      // The selected folder may have been removed elsewhere. Refresh the list
      // so the selection falls back to a folder that still exists.
      loadGrants();
      setError(apiMessage(e));
      return null;
    }
  };

  const allowFolder = async () => {
    const picked = await pickDirectory();
    const path = picked || window.prompt("Folder path to allow");
    if (!path) return;
    try {
      const next = await api.previewFolderGrant(path);
      setGrantMode(next.existing_mode === "read" ? "read" : "write");
      setPreview(next);
    } catch (e) {
      setError(apiMessage(e));
    }
  };

  const confirmGrant = async () => {
    if (!preview) return;
    try {
      const grant = await api.createFolderGrant(preview.realpath, grantMode);
      setPreview(null);
      setGrantId(grant.id);
      loadGrants();
      // The input stays disabled until a session exists, so open one now.
      await startSession(grant.id);
    } catch (e) {
      setError(apiMessage(e));
    }
  };

  const sendText = (text: string, sendMode: SessionMode): boolean => {
    if (!text || !sessionId) return false;
    // Depth is pinned to Deep by the server for Plan and Debug, so only the
    // mode is sent.
    if (!send(text, undefined, sendMode)) {
      setError("Not connected yet.");
      return false;
    }
    setStreaming(true);
    setAwaitingPoll(false);
    stickToBottomRef.current = true;
    watchSession(sessionId);
    setSessions((prev) => prev.map((item) => (item.id === sessionId ? { ...item, status: "running" } : item)));
    setMessages((prev) => [...prev, {
      id: `local-${++msgIdRef.current}`,
      type: "user",
      content: text,
      timestamp: new Date(),
      sessionId,
    }]);
    return true;
  };

  const submit = () => {
    const text = input.trim();
    if (!text && !hasAttachments) return;
    if (uploading || starting) return;
    void submitPrepared(text);
  };

  const submitPrepared = async (text: string) => {
    const deliver = async (sid: string, full: string) => {
      if (streaming) {
        if (!sendContext(full)) {
          setError("Not connected yet.");
          setInput(full);
          return;
        }
        setMessages((prev) => [...prev, {
          id: `local-${++msgIdRef.current}`,
          type: "user",
          content: full,
          metadata: { isContext: true, pending: true },
          timestamp: new Date(),
          sessionId: sid,
        }]);
        setInput("");
        stickToBottomRef.current = true;
        return;
      }
      if (sid === sessionId) {
        if (sendText(full, mode)) setInput("");
        else setInput(full);
        return;
      }
      pendingSendRef.current = { sid, text: full, mode };
      setInput("");
    };
    try {
      if (!sessionId) {
        if (!grantId || startingRef.current) return;
        startingRef.current = true;
        setStarting(true);
        const sid = await startSession();
        if (!sid) return;
        const full = await materializeAttachments(sid, text);
        // The route effect sends this once the new session is on screen. If
        // that render already happened during the upload, send from here.
        pendingSendRef.current = { sid, text: full, mode };
        setInput("");
        if (sessionIdRef.current === sid) {
          pendingSendRef.current = null;
          await waitForConnection();
          if (sessionIdRef.current === sid && !sendTextRef.current(full, mode)) setInput(full);
        }
        return;
      }
      await deliver(sessionId, await materializeAttachments(sessionId, text));
    } catch (e) {
      setError(apiMessage(e));
    } finally {
      startingRef.current = false;
      setStarting(false);
    }
  };

  const runRevert = async (paths?: string[], turn?: number, hunk?: number) => {
    if (!sessionId) return;
    setConfirmRevert(false);
    try {
      const result = await api.revertWorkspace(sessionId, paths, turn, hunk);
      const note = describeRevert(result);
      setRevertNote(note);
      setToast(note);
      refreshChanges();
    } catch (e) {
      const note = revertFailure(e);
      setRevertNote(note);
      setError(note);
    }
  };

  const restoreToMessage = useCallback(async (messageIndex: number) => {
    if (!sessionId || streaming) return;
    const ordinal = userOrdinal(messages, messageIndex);
    const checkpoint = (changes?.checkpoints || []).find((item) => item.user_index === ordinal);
    if (checkpoint?.turn == null) {
      setError("No checkpoint was taken before that message.");
      return;
    }
    await runRevert(undefined, checkpoint.turn);
  }, [sessionId, streaming, messages, changes, refreshChanges]);

  const handleEditMessage = useCallback((messageIndex: number, newContent: string, restore?: boolean) => {
    if (streaming || !sessionId) return;
    const ordinal = userOrdinal(messages, messageIndex);
    const go = async () => {
      if (restore) {
        const checkpoint = (changes?.checkpoints || []).find((item) => item.user_index === ordinal);
        if (checkpoint?.turn != null) {
          try {
            const result = await api.revertWorkspace(sessionId, undefined, checkpoint.turn);
            setToast(describeRevert(result));
          } catch (e) {
            setError(revertFailure(e));
            return;
          }
        }
      }
      const edited: ChatMessage = {
        id: `local-${++msgIdRef.current}`,
        type: "user",
        content: newContent,
        timestamp: new Date(),
        sessionId,
      };
      setMessages((prev) => prev.slice(0, messageIndex).concat(edited));
      if (!sendEdit(ordinal, newContent, undefined, mode)) {
        setError("Not connected to the server. Wait a moment and try again.");
        return;
      }
      setStreaming(true);
      watchSession(sessionId);
    };
    void go();
  }, [streaming, sessionId, messages, changes, sendEdit, mode, watchSession]);

  const pendingCard = useMemo(
    () => [...messages].reverse().find((m) => m.type === "hitl_request" && !m.metadata?.resolved),
    [messages],
  );

  const decidePending = (kind: "approve" | "reject") => {
    const card = pendingCard;
    if (!card || hitlSubmittedRef.current.has(card.id)) return;
    const risk = card.metadata?.risk as { no_auto_approve?: boolean } | undefined;
    if (risk?.no_auto_approve) return;
    const actions = (card.metadata?.action_requests as Array<{ args?: Record<string, unknown> }> | undefined) ?? [];
    if (actions.some((action) => screenHighRiskCommand(action.args?.command).length > 0)) return;
    const requestId = typeof card.metadata?.request_id === "string" ? card.metadata.request_id : undefined;
    handleHitlDecision(card.id, actions.map(() => ({ type: kind })), requestId);
    composerRef.current?.focus();
  };

  const insertMention = (path: string) => {
    composerRef.current?.replaceMention(`@${path} `);
    setAtOpen(false);
  };

  // Same queues as Chat: picked or pasted blobs, and dropped files, are
  // copied into the session on send. Dropped folders are linked.
  const addFiles = useCallback((files: FileList | File[]) => {
    const blobs: File[] = [];
    const nativePaths: string[] = [];
    for (const file of Array.from(files)) {
      const native = (file as File & { path?: string }).path;
      if (typeof native === "string" && (native.startsWith("/") || /^[A-Za-z]:[\\/]/.test(native))) {
        nativePaths.push(native);
        continue;
      }
      if (file.size === 0) continue;
      blobs.push(file);
    }
    if (nativePaths.length) {
      addDroppedPaths(nativePaths);
    }
    if (blobs.length) setPendingFiles((prev) => [...prev, ...blobs]);
  }, []);

  const addDroppedPaths = useCallback((paths: string[]) => {
    void (async () => {
      let kinds = null;
      try {
        kinds = (await api.pathKinds(paths)).paths;
      } catch (e) {
        setError(apiMessage(e));
      }
      const split = splitDroppedPaths(paths, kinds);
      if (split.files.length) setPendingFilePaths((prev) => [...prev, ...split.files]);
      if (split.folders.length) setPendingFolders((prev) => [...prev, ...split.folders]);
    })();
  }, []);

  const tryNativeClipboardPaste = useCallback(async (fallbackFiles: File[] = []) => {
    try {
      const data = await api.clipboardFile();
      const items = data.items?.length
        ? data.items
        : data.path
          ? [{ path: data.path, is_dir: data.is_dir }]
          : [];
      if (items.length) {
        const folders = items.filter((item) => item.is_dir).map((item) => item.path);
        const filePaths = items.filter((item) => !item.is_dir).map((item) => item.path);
        if (filePaths.length) setPendingFilePaths((prev) => [...prev, ...filePaths]);
        if (folders.length) setPendingFolders((prev) => [...prev, ...folders]);
        return;
      }
    } catch {
      // Backend unreachable — fall through to real blobs.
    }
    const blobs = fallbackFiles.filter((file) => file.size > 0);
    if (blobs.length) addFiles(blobs);
  }, [addFiles]);

  useEffect(() => {
    let cancelled = false;
    let unlisten: (() => void) | null = null;
    let win: ReturnType<typeof getCurrentWindow>;
    try {
      win = getCurrentWindow();
    } catch {
      return;
    }
    win.onDragDropEvent((event) => {
      const payload = event.payload as { type: string; paths?: string[] };
      if (payload.type === "hover") setIsDragging(true);
      else if (payload.type === "leave" || payload.type === "cancel") setIsDragging(false);
      else if (payload.type === "drop" && payload.paths) {
        setIsDragging(false);
        addDroppedPaths(payload.paths);
      }
    }).then((fn) => {
      if (cancelled) fn();
      else unlisten = fn;
    }).catch(() => {});
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, [addDroppedPaths]);

  const materializeAttachments = async (sid: string, text: string): Promise<string> => {
    const files = [...pendingFiles];
    const filePaths = [...pendingFilePaths];
    const folders = [...pendingFolders];
    let uploaded: string[] = [];
    if (files.length > 0) {
      setUploading(true);
      try {
        const results = await Promise.all(
          files.map((file) => api.uploadSessionFile(sid, `uploads/${file.name}`, file)),
        );
        uploaded = results.map((row) => `/uploads/${row.path.split("/").pop()}`);
        setPendingFiles((prev) => prev.filter((file) => !files.includes(file)));
      } catch (e) {
        throw new Error(`File upload failed (${e instanceof Error ? e.message : "unknown error"}) — your message was not sent.`);
      } finally {
        setUploading(false);
      }
    }
    let linkedFiles: string[] = [];
    let linkedFolders: string[] = [];
    const dropped = [...filePaths, ...folders];
    if (dropped.length > 0) {
      try {
        const linkResults = await Promise.all(dropped.map((source) => api.createSessionLink(sid, source)));
        linkResults.forEach((row) => {
          if (row.is_dir) linkedFolders.push(row.path);
          else linkedFiles.push(row.path);
        });
        setPendingFilePaths((prev) => prev.filter((path) => !filePaths.includes(path)));
        setPendingFolders((prev) => prev.filter((path) => !folders.includes(path)));
      } catch (e) {
        throw new Error(`Could not attach the dropped files (${e instanceof Error ? e.message : "unknown error"}) — your message was not sent.`);
      }
    }
    const parts: string[] = [];
    const allFiles = [...uploaded, ...linkedFiles];
    if (allFiles.length > 0) parts.push(`[Uploaded files: ${allFiles.join(", ")}]`);
    if (linkedFolders.length > 0) parts.push(`[Context folders: ${linkedFolders.join(", ")}]`);
    if (pendingSnippets.length > 0) {
      parts.push(formatSnippets(pendingSnippets));
      setPendingSnippets([]);
      setOpenSnippet(null);
    }
    return parts.length > 0 ? `${parts.join("\n")}\n\n${text}` : text;
  };

  const addSnippet = useCallback((snippet: Snippet, prefill?: string) => {
    setPendingSnippets((prev) => [...prev, snippet]);
    if (prefill) setInput((current) => (current.trim() ? current : prefill));
    window.setTimeout(() => composerRef.current?.focus(), 0);
  }, []);

  // First message of a session started from the composer. `sendText` is read
  // through a ref so the send uses the render that has the new session open.
  const sendTextRef = useRef(sendText);
  sendTextRef.current = sendText;
  useEffect(() => {
    const pending = pendingSendRef.current;
    if (!pending || pending.sid !== sessionId) return;
    pendingSendRef.current = null;
    waitForConnection()
      .then(() => {
        if (sessionIdRef.current !== pending.sid) return;
        if (!sendTextRef.current(pending.text, pending.mode)) setInput(pending.text);
      })
      .catch(() => {
        setInput(pending.text);
        setError("Not connected to the server. Your message is back in the box; try again.");
      });
  }, [sessionId, waitForConnection]);

  const [planConfirm, setPlanConfirm] = useState(false);
  const [planText, setPlanText] = useState("");

  const buildPlan = async () => {
    if (!sessionId) return;
    setPlanText(planFromChat);
    if (hasPlan) {
      try {
        const response = await fetch(api.getSessionFileUrl(sessionId, PLAN_PATH), { cache: "no-store" });
        if (response.ok) setPlanText(await response.text());
      } catch (e) {
        setError(apiMessage(e));
      }
    }
    setPlanConfirm(true);
  };

  const confirmBuild = async () => {
    if (!sessionId) return;
    setPlanConfirm(false);
    const request = hasPlan ? "Implement the plan in /output/plan.md." : "Implement the plan you just proposed.";
    await changeMode("agent");
    setInspector("changes");
    sendText(request, "agent");
  };

  const cycleMode = () => {
    if (streaming) return;
    const at = MODES.findIndex((item) => item.id === mode);
    void changeMode(MODES[(at + 1) % MODES.length].id);
  };

  const removeProject = async (grant: FolderGrant) => {
    if (streaming && session?.workspace?.grant_id === grant.id) {
      setError("Wait for the run to finish before removing this project.");
      return;
    }
    try {
      await api.revokeFolderGrant(grant.id);
      const rest = grants.filter((item) => item.id !== grant.id);
      setGrants(rest);
      if (grantId === grant.id) {
        const next = rest[0]?.id ?? "";
        setGrantId(next);
        if (!next) {
          localStorage.removeItem(LAST_GRANT);
          setSessions([]);
        }
      }
      if (sessionId && session?.workspace?.grant_id === grant.id) {
        const info = await api.getSession(sessionId).catch(() => null);
        setSession(info ?? (session ? { ...session, workspace: null } : null));
        setChanges(null);
        setGit(null);
        setDiff("");
        setDiffPath(null);
        setArtifact(null);
      }
    } catch (e) {
      setError(apiMessage(e));
    }
  };

  const unmapProject = async () => {
    if (!sessionId || streaming) return;
    try {
      const info = await api.clearSessionWorkspace(sessionId);
      setSession(info);
      setChanges(null);
      setGit(null);
      setDiff("");
      setDiffPath(null);
      setArtifact(null);
      setSessions((prev) => prev.filter((item) => item.id !== sessionId));
    } catch (e) {
      setError(apiMessage(e));
    }
  };

  const finishDelete = useCallback(async (id: string) => {
    try {
      await api.closeSession(id);
      setSessions((prev) => prev.filter((item) => item.id !== id));
    } catch (e) {
      setError(apiMessage(e));
    }
    setHiddenIds((prev) => prev.filter((item) => item !== id));
  }, []);

  // Deleting hides the row at once and removes the session after a short undo window.
  const deleteSession = (id: string) => {
    const title = sessions.find((item) => item.id === id)?.title || "Untitled";
    if (pendingDelete.current) {
      window.clearTimeout(pendingDelete.current.timer);
      void finishDelete(pendingDelete.current.id);
    }
    setHiddenIds((prev) => [...prev, id]);
    setUndoDelete({ id, title });
    pendingDelete.current = { id, timer: window.setTimeout(() => { pendingDelete.current = null; setUndoDelete(null); void finishDelete(id); }, 6000) };
    if (id === sessionIdRef.current) {
      setStreaming(false);
      setAwaitingPoll(false);
      navigate("/code");
    }
  };

  const undoDeleteSession = () => {
    const pending = pendingDelete.current;
    if (!pending) return;
    window.clearTimeout(pending.timer);
    pendingDelete.current = null;
    setHiddenIds((prev) => prev.filter((item) => item !== pending.id));
    setUndoDelete(null);
  };

  useEffect(() => () => {
    const pending = pendingDelete.current;
    if (pending) {
      window.clearTimeout(pending.timer);
      void api.closeSession(pending.id).catch(() => {});
    }
  }, []);

  const openFile = (rel: string) => {
    if (!sessionId) return;
    const type: ArtifactType = artifactTypeFromPath(rel) ?? "code";
    setArtifact({
      path: rel,
      fileUrl: api.getWorkspaceFileUrl(sessionId, rel),
      type,
    });
    setInspector("files");
  };

  // Until Otto has saved /output/plan.md, the final answer of the turn is the plan.
  const planFromChat = useMemo(() => {
    if (mode !== "plan") return "";
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      const item = messages[i];
      if (item.type === "user") return "";
      if (item.type === "agent" && item.content.trim()) return item.content;
    }
    return "";
  }, [messages, mode]);
  const planReady = mode === "plan" && !streaming && (hasPlan || planFromChat !== "");
  const events = [...timeline, ...live];
  const selected = grants.find((g) => g.id === grantId);

  // Start the project's dev server once per session when the user turned that on.
  const autoStartedRef = useRef<string | null>(null);
  const sessionGrant = grants.find((item) => item.id === session?.workspace?.grant_id);
  useEffect(() => {
    if (!sessionId || !connected || !sessionGrant?.dev_autostart || mode !== "agent") return;
    if (session?.workspace?.mode === "read" || autoStartedRef.current === sessionId) return;
    autoStartedRef.current = sessionId;
    api.startDev(sessionId).then((state) => {
      if (state.started) setToast("Started the dev server. It is running in the Terminal tab.");
      void pollProcesses();
    }).catch((e) => {
      const message = apiMessage(e);
      if (!/No dev command|mode cannot/i.test(message)) setToast(`Dev server did not start: ${message}`);
    });
  }, [sessionId, connected, sessionGrant?.dev_autostart, mode, session?.workspace?.mode, pollProcesses]);

  // A terminal is unusable at the default inspector width.
  useEffect(() => {
    if (inspector === "terminal" && inspectorOpen && dock === "right" && !narrow && rightWidth < 460) resizePanel("right", 480);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inspector, inspectorOpen]);

  const inspectorPane = (
    <>
        <div className="flex border-b border-th-border px-1 pt-1 text-xs" role="tablist" aria-label="Inspector">
          {(["changes", "files", "terminal", "output", "timeline", ...(hasPlan || mode === "plan" ? ["plan"] : [])] as Inspector[]).map((tab) => {
            const TabIcon = { changes: FileDiff, files: Folder, terminal: SquareTerminal, output: Terminal, timeline: Clock, plan: ClipboardList }[tab];
            return (
              <button
                key={tab}
                type="button"
                role="tab"
                aria-selected={inspector === tab}
                onClick={() => setInspector(tab)}
                title={tab[0].toUpperCase() + tab.slice(1)}
                className={`relative flex-1 min-w-0 flex items-center justify-center gap-1.5 py-2 capitalize transition-colors duration-fast ${inspector === tab ? "text-th-text-primary font-medium" : "text-th-text-muted hover:text-th-text-secondary"}`}
              >
                <TabIcon size={13} className="shrink-0" />
                <span className={inspector === tab ? "truncate" : "sr-only"}>{tab}</span>
                {tab === "changes" && reviewStats.count > 0 && (
                  <span className="rounded-full bg-th-accent/15 px-1.5 text-2xs font-medium text-th-accent">{reviewStats.count}</span>
                )}
                {tab === "terminal" && runningProcs > 0 && (
                  <span className="rounded-full bg-th-success/15 px-1.5 text-2xs font-medium text-th-success" aria-label={`${runningProcs} running`}>{runningProcs}</span>
                )}
                {inspector === tab && <span className="absolute inset-x-2 -bottom-px h-0.5 rounded-full bg-th-accent" />}
              </button>
            );
          })}
          <span className="ml-1 flex shrink-0 items-center gap-0.5 self-center border-l border-th-border pl-1">
            <IconButton
              label={dock === "bottom" ? "Dock to the side" : "Dock below the conversation"}
              icon={dock === "bottom" ? <PanelRight size={14} /> : <PanelBottom size={14} />}
              onClick={() => chooseDock(dock === "bottom" ? "right" : "bottom")}
            />
            {dock === "bottom" && (
              <IconButton
                label={maximized ? "Restore" : "Fill the window"}
                icon={maximized ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
                aria-pressed={maximized}
                onClick={() => setMaximized((value) => !value)}
              />
            )}
            <IconButton label="Close inspector" icon={<X size={14} />} onClick={() => { setInspectorOpen(false); setMaximized(false); }} />
          </span>
        </div>
        <div className="flex-1 overflow-y-auto p-3 text-xs">
          {inspector === "changes" && (
            <div className="space-y-2">
              {(changes?.checkpoints?.length || 0) > 1 && (
                <label className="flex items-center gap-2 text-th-text-muted">
                  Turn
                  <select
                    className="bg-th-inset-bg border border-th-border rounded px-1 py-0.5 text-th-text-primary"
                    value={changeTurn ?? changes?.checkpoint?.turn ?? ""}
                    onChange={(event) => {
                      const next = Number(event.target.value);
                      const latest = changes?.checkpoints?.[changes.checkpoints.length - 1]?.turn;
                      setChangeTurn(next === latest ? null : next);
                      setDiffPath(null);
                    }}
                  >
                    {(changes?.checkpoints || []).map((item) => (
                      <option key={item.turn} value={item.turn}>Turn {item.turn}</option>
                    ))}
                  </select>
                </label>
              )}
              {changes?.message && <p className="text-th-text-muted">{changes.message}</p>}
              {sessionId && (changes?.files || []).some((file) => file.origin !== "outside") && (
                confirmRevert ? (
                  <div role="alertdialog" aria-label="Revert turn" className="flex flex-wrap items-center gap-2 rounded-xl border border-th-danger/25 bg-th-danger/8 p-2">
                    <span className="flex-1 min-w-[10rem] text-th-text-secondary">Revert Otto's changes? Edits made outside Otto stay.</span>
                    <Button variant="ghost" onClick={() => setConfirmRevert(false)}>Cancel</Button>
                    <Button variant="danger" className="!bg-th-danger/10" onClick={() => { void runRevert(undefined, changeTurn ?? undefined); }}>Revert</Button>
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <span className="flex-1 text-th-text-secondary">
                      <span className="font-medium text-th-text-primary">{reviewStats.count} file{reviewStats.count === 1 ? "" : "s"}</span>
                      <span className="text-th-diff-add"> +{reviewStats.added}</span>
                      <span className="text-th-diff-del"> −{reviewStats.deleted}</span>
                    </span>
                    <Button variant="ghost" onClick={() => setConfirmRevert(true)}><RotateCcw size={12} /> Revert turn</Button>
                  </div>
                )
              )}
              {revertNote && <p className="text-th-text-secondary">{revertNote}</p>}
              {(changes?.files || []).length === 0 && !changes?.message && (
                <p className="text-th-text-muted">No changes since this checkpoint.</p>
              )}
              {(changes?.files || []).map((file) => {
                const outside = file.origin === "outside";
                const { name, dir } = splitPath(file.path);
                const StatusIcon = file.status === "added" ? FilePlus2 : file.status === "deleted" ? FileMinus2 : FilePen;
                const statusTone = file.status === "added" ? "text-th-success" : file.status === "deleted" ? "text-th-danger" : "text-th-warning";
                const open = diffPath === file.path;
                return (
                  <div key={file.path} className="space-y-1.5">
                    <div className={`group flex items-center gap-1.5 rounded-lg px-1.5 py-1 transition-colors duration-fast ${open ? "bg-th-surface-hover" : "hover:bg-th-surface-hover/60"}`}>
                      <StatusIcon size={14} className={`shrink-0 ${statusTone}`} aria-label={file.status} />
                      <button
                        type="button"
                        aria-expanded={open}
                        title={file.path}
                        className="min-w-0 flex-1 truncate text-left"
                        onClick={() => setDiffPath(open ? null : file.path)}
                      >
                        <span className="font-medium text-th-text-primary">{name}</span>
                        {dir && <span className="ml-1.5 text-th-text-muted">{dir}</span>}
                      </button>
                      {outside && <Badge title="Changed outside Otto and not reverted">outside</Badge>}
                      {file.added != null && <span className="shrink-0 text-th-diff-add">+{file.added}</span>}
                      {file.deleted != null && <span className="shrink-0 text-th-diff-del">−{file.deleted}</span>}
                      {file.status !== "deleted" && (
                        <IconButton label={`Open ${file.path}`} icon={<ExternalLink size={12} />} onClick={() => openFile(file.path)} />
                      )}
                      {!outside && (
                        <IconButton
                          label={`Revert ${file.path}`}
                          icon={<Undo2 size={13} />}
                          className="hover:!text-th-danger"
                          onClick={() => { void runRevert([file.path], changeTurn ?? undefined); }}
                        />
                      )}
                    </div>
                    {open && diff && (
                      <PatchView
                        diff={diff}
                        path={file.path}
                        truncated={diffTruncated}
                        onAcceptHunk={() => {}}
                        onUndoHunk={
                          sessionId && changes?.git && !outside
                            ? (hunk) => { void runRevert([file.path], changeTurn ?? undefined, hunk); }
                            : undefined
                        }
                      />
                    )}
                  </div>
                );
              })}
            </div>
          )}
          {inspector === "files" && artifact && sessionId && (artifact.type === "code" || artifact.type === "txt" || artifact.type === "md" || artifact.type === "json") && (
            <FileEditor sessionId={sessionId} path={artifact.path} fileUrl={artifact.fileUrl} revision={artifact.revision} />
          )}
          {inspector === "files" && artifact && !(artifact.type === "code" || artifact.type === "txt" || artifact.type === "md" || artifact.type === "json") && (
            <ArtifactPanel artifact={artifact} onClose={() => setArtifact(null)} />
          )}
          {inspector === "files" && !artifact && (
            sessionId && session?.workspace ? (
              <WorkspaceTree
                sessionId={sessionId}
                fileUrl={(rel) => api.getWorkspaceFileUrl(sessionId, rel)}
                onOpenFile={(rel) => openFile(rel)}
              />
            ) : (
              <p className="text-th-text-muted">
                {sessionId ? "This session has no project folder." : "Start a session to browse the project files."}
              </p>
            )
          )}
          {inspector === "terminal" && (
            sessionId && session?.workspace ? (
              <div className="h-full min-h-[10rem]">
                <TerminalPanel
                  sessionId={sessionId}
                  grant={grants.find((item) => item.id === session.workspace?.grant_id) ?? null}
                  onGrantChange={(grant) => setGrants((prev) => prev.map((item) => (item.id === grant.id ? grant : item)))}
                  onAddToChat={addSnippet}
                />
              </div>
            ) : (
              <p className="text-th-text-muted">{sessionId ? "This session has no project folder." : "Start a session to open a terminal."}</p>
            )
          )}
          {inspector === "output" && (
            <div className="space-y-3">
              {output.length === 0 && <p className="text-th-text-muted">Command output shows up here.</p>}
              {output.map((item, index) => {
                const exit = item.done ? commandExit(item.text) : null;
                const showTurn = item.turn > 0 && (index === 0 || output[index - 1].turn !== item.turn);
                return (
                  <div key={item.id} className="space-y-1">
                    {showTurn && <p className="text-2xs uppercase tracking-wider text-th-text-muted pt-1">Turn {item.turn}</p>}
                    <div className="flex items-center gap-2">
                      {item.command && <p className="font-mono text-2xs text-th-text-muted break-all flex-1">$ {item.command}</p>}
                      {item.done && exit != null && (
                        <span className={exit === 0 ? "text-emerald-400" : "text-red-300"}>
                          {exit === 0 ? "passed" : `exit ${exit}`}
                        </span>
                      )}
                      <button
                        type="button"
                        className="text-th-text-muted hover:text-th-text-primary"
                        title="Copy output"
                        onClick={() => { void navigator.clipboard.writeText(item.text || ""); }}
                      >
                        <Copy size={12} />
                      </button>
                    </div>
                    <pre className="whitespace-pre-wrap font-mono text-2xs text-th-text-secondary bg-th-inset-bg rounded-lg p-2">{item.text || (item.done ? "(no output)" : "Running…")}</pre>
                  </div>
                );
              })}
            </div>
          )}
          {inspector === "timeline" && <RunTimeline events={events} />}
          {inspector === "plan" && sessionId && (
            hasPlan || planFromChat ? (
              <div className="space-y-2">
                <ArtifactPanel
                  artifact={hasPlan
                    ? { path: PLAN_PATH, fileUrl: api.getSessionFileUrl(sessionId, PLAN_PATH), type: "md", revision: planRev }
                    : { path: "plan.md", fileUrl: `data:text/markdown;charset=utf-8,${encodeURIComponent(planFromChat)}`, type: "md" }}
                  onClose={() => setInspector("changes")}
                />
                {planReady && (
                  <button type="button" onClick={() => { void buildPlan(); }} className="flex items-center gap-1 px-2 py-1 rounded-lg border border-th-border text-th-text-primary hover:bg-th-surface-hover">
                    <Hammer size={12} /> Build this plan
                  </button>
                )}
              </div>
            ) : (
              <p className="text-th-text-muted">In Plan mode the plan shows up here.</p>
            )
          )}
        </div>
    </>
  );

  return (
    <div ref={shellRef} className="flex h-full min-h-0 min-w-0 overflow-hidden">
      {projectsOpen && <aside style={{ width: narrow ? 200 : leftWidth }} className="shrink-0 border-r border-th-border flex flex-col min-h-0 min-w-0 overflow-hidden bg-th-bg-secondary">
        <div className="px-3 pt-3 pb-1 flex items-center justify-between">
          <span className="text-2xs font-semibold uppercase tracking-wider text-th-text-muted">Projects</span>
          <IconButton label="Allow a folder" icon={<FolderPlus size={14} />} onClick={() => { void allowFolder(); }} />
        </div>
        <div className="px-2 space-y-0.5 overflow-y-auto max-h-[35%] shrink-0">
          {grants.map((grant) => (
            <div
              key={grant.id}
              className={`group flex items-center rounded-lg transition-colors duration-fast ${grant.id === grantId ? "bg-th-surface-hover" : "hover:bg-th-surface-hover/60"}`}
            >
              <button
                type="button"
                onClick={() => setGrantId(grant.id)}
                aria-current={grant.id === grantId ? "true" : undefined}
                className={`flex-1 min-w-0 flex items-center gap-2 text-left px-2 py-1.5 text-xs ${grant.id === grantId ? "text-th-text-primary font-medium" : "text-th-text-secondary"}`}
                title={grant.realpath}
              >
                <Folder size={13} className="shrink-0 text-th-text-muted" />
                <span className="truncate">{grant.label}</span>
                {grant.mode === "read" && <Lock size={10} className="shrink-0 text-th-text-muted" aria-label="Read only" />}
              </button>
              {grant.id === grantId && confirmGrantId === grant.id && (
                <span className="shrink-0 flex items-center gap-1 pr-1.5 text-2xs">
                  <button type="button" className="font-medium text-th-danger" onClick={() => { setConfirmGrantId(null); void removeProject(grant); }}>
                    Remove
                  </button>
                  <button type="button" className="text-th-text-muted" onClick={() => setConfirmGrantId(null)}>Keep</button>
                </span>
              )}
              {grant.id === grantId && confirmGrantId !== grant.id && (
                <IconButton
                  label="Remove this project. Sessions using it lose the folder. Files on disk stay."
                  icon={<X size={12} />}
                  onClick={() => setConfirmGrantId(grant.id)}
                  className="mr-0.5 opacity-0 group-hover:opacity-100 focus:opacity-100 hover:!text-th-danger"
                />
              )}
            </div>
          ))}
          {grants.length === 0 && <p className="px-2 text-2xs text-th-text-muted">Allow a project folder to start.</p>}
        </div>
        <div className="px-3 pt-3 pb-1 flex items-center justify-between border-t border-th-border mt-2">
          <span className="text-2xs font-semibold uppercase tracking-wider text-th-text-muted">Sessions</span>
          <IconButton label="New coding session" icon={<Plus size={14} />} onClick={() => { void startSession(); }} disabled={!grantId} />
        </div>
        <div className="flex-1 overflow-y-auto px-2 pb-2">
          {(() => {
            let lastBucket = "";
            return sessions.filter((item) => !hiddenIds.includes(item.id)).map((item) => {
              const bucket = dayBucket(item.updated_at);
              const heading = bucket !== lastBucket ? bucket : null;
              lastBucket = bucket;
              const active = item.id === sessionId;
              return (
                <div key={item.id}>
                  {heading && <p className="px-2 pt-2.5 pb-1 text-2xs font-medium text-th-text-faint">{heading}</p>}
                  <div className={`group flex items-center rounded-lg transition-colors duration-fast ${active ? "bg-th-surface-hover" : "hover:bg-th-surface-hover/60"}`}>
                    <button
                      type="button"
                      onClick={() => navigate(`/code/${item.id}`)}
                      aria-current={active ? "page" : undefined}
                      className={`flex-1 min-w-0 flex items-center gap-2 text-left px-2 py-1.5 text-xs ${active ? "text-th-text-primary font-medium" : "text-th-text-secondary"}`}
                    >
                      <span
                        className={`shrink-0 w-1.5 h-1.5 rounded-full ${item.status === "running" ? "bg-th-success animate-pulse" : "bg-transparent border border-th-border-strong"}`}
                        title={item.status === "running" ? "Running" : "Idle"}
                      />
                      <span className="sr-only">{item.status === "running" ? "Running. " : ""}</span>
                      <span className="truncate flex-1">{item.title || "Untitled"}</span>
                      {item.mode && item.mode !== "agent" && <Badge tone="neutral">{item.mode}</Badge>}
                      <span className="shrink-0 text-2xs text-th-text-faint group-hover:hidden group-focus-within:hidden">{relTime(item.updated_at)}</span>
                    </button>
                    <IconButton
                      label={`Delete ${item.title || "Untitled"}`}
                      icon={<Trash2 size={12} />}
                      onClick={() => deleteSession(item.id)}
                      className="mr-0.5 hidden group-hover:inline-flex focus:inline-flex hover:!text-th-danger"
                    />
                  </div>
                </div>
              );
            });
          })()}
          {sessions.length === 0 && <p className="px-2 pt-1 text-2xs text-th-text-muted">No sessions yet. Press + to start one.</p>}
        </div>
        {confirmGrantId && (
          <p className="px-3 py-2 text-2xs text-th-text-muted border-t border-th-border">
            {sessions.length} session{sessions.length === 1 ? "" : "s"} will lose this folder.
            {sessions.some((item) => item.status === "running") ? " A running session will be stopped." : " Files on disk stay."}
          </p>
        )}
      </aside>}
      {projectsOpen && <div
        role="separator"
        aria-orientation="vertical"
        aria-label="Projects width"
        aria-valuemin={LEFT_MIN}
        aria-valuemax={LEFT_MAX}
        aria-valuenow={leftWidth}
        tabIndex={0}
        title="Drag to resize. Double-click to reset."
        onPointerDown={(event) => startResize("left", event)}
        onDoubleClick={() => resizePanel("left", LEFT_DEFAULT)}
        onKeyDown={(event) => {
          if (event.key === "ArrowLeft") resizePanel("left", leftWidth - 16);
          if (event.key === "ArrowRight") resizePanel("left", leftWidth + 16);
          if (event.key === "Home") resizePanel("left", LEFT_DEFAULT);
        }}
        className="w-1.5 shrink-0 cursor-col-resize bg-transparent hover:bg-sky-400/50 active:bg-sky-400/70 focus:outline-none focus:bg-sky-400/40"
      />}

      <section className="flex-1 min-w-0 flex flex-col min-h-0 overflow-hidden">
        <header className="px-3 py-2 border-b border-th-border flex items-center gap-2 text-xs min-w-0 overflow-hidden">
          <IconButton
            label={projectsOpen ? "Hide projects" : "Show projects"}
            icon={<PanelLeft size={15} />}
            aria-pressed={projectsOpen}
            onClick={() => setProjectsOpen((open) => !open)}
          />
          <span className="font-semibold text-sm text-th-text-primary truncate min-w-[4rem]">
            {sessionId && !session?.workspace ? "No project" : (session?.workspace?.name || selected?.label || "Code")}
          </span>
          {session?.workspace && (
            <IconButton
              label="Remove this project from the session"
              icon={<X size={12} />}
              disabled={streaming}
              onClick={() => { void unmapProject(); }}
            />
          )}
          {git?.git && (
            <Badge title="Git branch">
              <GitBranch size={11} />
              <span className="max-w-[10rem] truncate">{git.branch}</span>
              {git.dirty ? <span className="text-th-warning">· {git.dirty} dirty</span> : null}
            </Badge>
          )}
          {modeBadge(mode) && <Badge tone={modeBadge(mode)!.tone}>{modeBadge(mode)!.label}</Badge>}
          {session?.workspace?.mode === "read" && <Badge tone="accent"><Lock size={10} /> Read only</Badge>}
          {!compact && session?.workspace?.mode === "write" && mode === "agent" && <Badge tone="success" title="Edits are checkpointed and can be reverted">Editable</Badge>}
          <span className="ml-auto flex items-center gap-1 shrink-0">
            {selected && (
              <Button
                variant="ghost"
                aria-pressed={Boolean(selected.network)}
                title={selected.network_hosts?.length ? `Hosts: ${selected.network_hosts.join(", ")}` : "Allow shell network for this project while the global switch is off"}
                onClick={() => {
                  void api.updateFolderGrant(selected.id, { network: !selected.network }).then((grant) => {
                    setGrants((prev) => prev.map((item) => item.id === grant.id ? grant : item));
                  }).catch((e) => setError(apiMessage(e)));
                }}
                className={selected.network ? "!text-th-success" : ""}
              >
                <Globe size={13} aria-hidden /> {compact ? <span className="sr-only">{selected.network ? "Network on" : "Network off"}</span> : (selected.network ? "Network on" : "Network off")}
              </Button>
            )}
            {runningProcs > 0 && (
              <Button variant="ghost" onClick={() => { setInspector("terminal"); setInspectorOpen(true); }} title="Servers and commands the agent started">
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-th-success" /> {runningProcs} running
              </Button>
            )}
            <span className="flex items-center gap-1.5 px-1.5 text-2xs text-th-text-muted" role="status">
              <span className={`h-1.5 w-1.5 rounded-full ${connected ? "bg-th-success" : "bg-th-warning animate-pulse"}`} />
              {connected ? "Live" : "Connecting"}
            </span>
            {sessionId && (
              <>
                <Button variant="ghost" onClick={() => navigate(`/chat/${sessionId}`)}>Chat</Button>
                <Button variant="ghost" onClick={() => navigate(`/runs/${sessionId}`)}>Run</Button>
              </>
            )}
            <IconButton
              label={inspectorOpen ? "Hide inspector" : "Show inspector"}
              icon={<PanelRight size={15} />}
              aria-pressed={inspectorOpen}
              onClick={() => setInspectorOpen((open) => !open)}
            />
          </span>
        </header>
        {notice && (
          <p className="px-4 py-1.5 text-2xs bg-th-warning/10 text-th-warning border-b border-th-warning/20">
            No checkpoint for this turn: {notice}
          </p>
        )}
        {error && (
          <div role="alert" className="mx-4 mt-3 flex items-start gap-2 rounded-xl border border-th-danger/25 bg-th-danger/10 px-3 py-2 text-xs text-th-danger animate-panel-in">
            <AlertTriangle size={14} className="mt-px shrink-0" />
            <span className="flex-1 min-w-0 break-words">{error}</span>
            <IconButton label="Dismiss error" icon={<X size={12} />} onClick={() => setError(null)} className="!h-5 !w-5" />
          </div>
        )}
        <div className="sr-only" aria-live="polite">
          {streaming ? "Working" : awaitingPoll ? "Awaiting approval" : toast || revertNote || ""}
        </div>
        {(changes?.debug_markers ?? 0) > 0 && (
          <p className="px-4 py-1.5 text-2xs bg-th-warning/10 text-th-warning border-b border-th-warning/20">
            {changes?.debug_markers} otto-debug log marker{changes?.debug_markers === 1 ? "" : "s"} still in your files. Remove them before you commit.
          </p>
        )}
        <div className={`relative flex-1 min-h-0 flex flex-col ${inspectorOpen && dock === "bottom" && maximized ? "hidden" : ""}`}>
        <div
          ref={scrollRef}
          className="flex-1 min-h-0 overflow-y-auto px-4 py-4 space-y-4"
          onScroll={(e) => {
            const el = e.currentTarget;
            const gap = el.scrollHeight - el.scrollTop - el.clientHeight;
            stickToBottomRef.current = gap < 80;
            setShowJump(gap > 240);
          }}
        >
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center text-center gap-3">
              <MessageSquare className="text-th-text-muted" size={28} />
              <p className="text-sm text-th-text-secondary">
                {sessionId && !session?.workspace
                  ? "This session has no project folder."
                  : mode === "ask" ? "Ask about the code in this folder."
                    : mode === "plan" ? "Describe what to plan."
                      : mode === "debug" ? "Describe the bug to reproduce."
                        : "Describe a change in the allowed folder."}
              </p>
              {session?.workspace && (
                <div className="flex flex-col gap-1.5">
                  {STARTERS[mode].map((prompt) => (
                    <button
                      key={prompt}
                      type="button"
                      className="text-xs text-th-text-tertiary hover:text-th-text-primary px-3 py-1 rounded-lg border border-th-border"
                      onClick={() => { setInput(prompt); composerRef.current?.focus(); }}
                    >
                      {prompt}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
          <MessageList
            items={renderItems}
            thoughtFlags={thoughtFlags}
            latestTodoFlags={latestTodoFlags}
            isStreaming={streaming}
            currentSessionId={sessionId ?? null}
            resolveSubagentModel={resolveSubagentModel}
            onEditMessage={handleEditMessage}
            restorable={restorable}
            onRestoreMessage={(index) => { void restoreToMessage(index); }}
            onHitlDecision={handleHitlDecision}
            onApproveAllSession={approveAllThisSession}
            onOpenArtifact={handleOpenArtifact}
          />
          {streaming && <ThinkingIndicator phase="thinking" withAvatar />}
          {errorMessages.length > 0 && (
            <div className="space-y-3">
              {errorMessages.map((message) => (
                <MessageBubble key={message.id} message={message} sessionId={sessionId} />
              ))}
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        {showJump && (
          <button
            type="button"
            onClick={() => {
              stickToBottomRef.current = true;
              messagesEndRef.current?.scrollIntoView({ block: "end", behavior: "smooth" });
            }}
            className="absolute bottom-3 left-1/2 -translate-x-1/2 flex items-center gap-1.5 rounded-full border border-th-border bg-th-dropdown-bg px-3 py-1.5 text-xs font-medium text-th-text-primary shadow-lg animate-toast-in hover:bg-th-surface-hover"
          >
            <ArrowDown size={12} /> {streaming ? "Jump to latest" : "Scroll to bottom"}
          </button>
        )}
        </div>
        {inspectorOpen && dock === "bottom" && (
          <div
            style={maximized ? undefined : { height: bottomHeight }}
            className={`relative flex min-h-0 flex-col border-t border-th-border bg-th-bg-secondary animate-panel-in ${maximized ? "flex-1" : "shrink-0"}`}
          >
            {!maximized && (
              <div
                role="separator"
                aria-orientation="horizontal"
                aria-label="Inspector height"
                aria-valuemin={BOTTOM_MIN}
                aria-valuenow={bottomHeight}
                tabIndex={0}
                title="Drag to resize. Double-click to reset."
                onPointerDown={startBottomResize}
                onDoubleClick={() => { setBottomHeight(300); try { localStorage.setItem(BOTTOM_HEIGHT_KEY, "300"); } catch { /* storage unavailable */ } }}
                onKeyDown={(event) => {
                  if (event.key === "ArrowUp") setBottomHeight((h) => Math.min(900, h + 24));
                  if (event.key === "ArrowDown") setBottomHeight((h) => Math.max(BOTTOM_MIN, h - 24));
                }}
                className="absolute inset-x-0 -top-1 z-10 h-2 cursor-row-resize bg-transparent hover:bg-sky-400/50 active:bg-sky-400/70 focus:outline-none focus:bg-sky-400/40"
              />
            )}
            {inspectorPane}
          </div>
        )}
        {showReview && sessionId && (
          <div className="mx-3 mb-2 flex items-center gap-3 rounded-xl border border-th-border bg-th-surface px-3 py-2 text-xs shadow-sm animate-panel-in">
            <FileDiff size={15} className="shrink-0 text-th-text-muted" />
            <span className="text-th-text-secondary flex-1 min-w-0 truncate">
              {reviewStats.count > 0 && (
                <>
                  <span className="font-medium text-th-text-primary">{reviewStats.count} file{reviewStats.count === 1 ? "" : "s"} changed</span>
                  <span className="text-th-diff-add"> +{reviewStats.added}</span>
                  <span className="text-th-diff-del"> −{reviewStats.deleted}</span>
                </>
              )}
              {reviewStats.outside > 0 && (
                <span className="text-th-text-muted">
                  {reviewStats.count > 0 ? " · " : ""}
                  {reviewStats.outside} changed outside Otto
                </span>
              )}
            </span>
            {reviewStats.count > 0 && (
              <Button variant="danger" onClick={() => { setInspector("changes"); setInspectorOpen(true); setConfirmRevert(true); }}>Undo all</Button>
            )}
            <Button variant="secondary" onClick={() => { setInspector("changes"); setInspectorOpen(true); }}>Review</Button>
            <Button variant="primary" onClick={() => setKeptKey(reviewKey)}><Check size={12} /> Keep</Button>
          </div>
        )}
        {planReady && sessionId && (
          <div className="mx-3 mb-2 flex items-center gap-3 rounded-xl border border-th-accent/25 bg-th-accent/8 px-3 py-2 text-xs animate-panel-in">
            <ClipboardList size={15} className="shrink-0 text-th-accent" />
            <span className="text-th-text-secondary flex-1 min-w-0 truncate">Plan ready. Review it, then build.</span>
            <Button variant="ghost" onClick={() => setInspector("plan")}>View plan</Button>
            <Button variant="primary" onClick={() => { void buildPlan(); }}><Hammer size={12} /> Build this plan</Button>
          </div>
        )}
        <form
          className={`p-3 space-y-2 transition-colors duration-fast ${isDragging ? "bg-th-accent/8" : ""}`}
          onSubmit={(e) => { e.preventDefault(); submit(); }}
          onPasteCapture={(event) => {
            const pasted = event.clipboardData?.getData("text/plain") ?? "";
            if (!pasted || event.clipboardData?.files?.length || !isTerminalPaste(pasted)) return;
            event.preventDefault();
            event.stopPropagation();
            addSnippet(makeSnippet("Terminal", pasted));
          }}
          onDragOver={(event) => { event.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setIsDragging(false);
            if (event.dataTransfer.files?.length) addFiles(event.dataTransfer.files);
          }}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            className="hidden"
            onChange={(event) => {
              if (event.target.files) addFiles(event.target.files);
              event.target.value = "";
            }}
          />
          {hasAttachments && (
            <div className="flex flex-wrap gap-2">
              {pendingSnippets.map((snippet) => (
                <div key={snippet.id} className="max-w-full">
                  <div className="flex items-center gap-1.5 rounded-lg border border-th-border bg-th-inset-bg px-2.5 py-1.5 text-xs text-th-text-secondary">
                    <SquareTerminal size={12} className="shrink-0 text-th-text-muted" />
                    <button
                      type="button"
                      aria-expanded={openSnippet === snippet.id}
                      onClick={() => setOpenSnippet((current) => (current === snippet.id ? null : snippet.id))}
                      className="flex min-w-0 items-center gap-1.5"
                      title="Show what will be sent"
                    >
                      <span className="max-w-[150px] truncate font-medium text-th-text-primary">{snippet.label}</span>
                      <span className="shrink-0 text-th-text-muted">{snippet.lines} line{snippet.lines === 1 ? "" : "s"}</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => setPendingSnippets((prev) => prev.filter((item) => item.id !== snippet.id))}
                      className="text-th-text-muted hover:text-th-text-primary"
                      aria-label={`Remove ${snippet.label} output`}
                    >
                      <X size={11} />
                    </button>
                  </div>
                  {openSnippet === snippet.id && (
                    <pre className="mt-1 max-h-40 w-full overflow-auto whitespace-pre-wrap rounded-lg border border-th-border bg-th-code-bg p-2 font-mono text-2xs text-th-text-secondary">{snippet.text}</pre>
                  )}
                </div>
              ))}
              {pendingFiles.map((file, index) => (
                <div key={`file-${file.name}-${index}`} className="flex items-center gap-1.5 px-2.5 py-1.5 bg-th-inset-bg border border-th-border rounded-lg text-xs text-th-text-secondary">
                  <FileText size={12} className="text-th-text-muted shrink-0" />
                  <span className="truncate max-w-[150px] text-th-text-primary">{file.name}</span>
                  <span className="text-th-text-muted">{formatFileSize(file.size)}</span>
                  <button type="button" onClick={() => setPendingFiles((prev) => prev.filter((_, i) => i !== index))} className="text-th-text-muted hover:text-th-text-primary" aria-label={`Remove ${file.name}`}>
                    <X size={11} />
                  </button>
                </div>
              ))}
              {pendingFilePaths.map((path, index) => {
                const name = path.replace(/\\/g, "/").split("/").pop() ?? path;
                return (
                  <div key={`path-${path}-${index}`} className="flex items-center gap-1.5 px-2.5 py-1.5 bg-th-inset-bg border border-th-border rounded-lg text-xs text-th-text-secondary" title={path}>
                    <FileText size={12} className="text-th-text-muted shrink-0" />
                    <span className="truncate max-w-[150px] text-th-text-primary">{name}</span>
                    <button type="button" onClick={() => setPendingFilePaths((prev) => prev.filter((_, i) => i !== index))} className="text-th-text-muted hover:text-th-text-primary" aria-label={`Remove ${name}`}>
                      <X size={11} />
                    </button>
                  </div>
                );
              })}
              {pendingFolders.map((path, index) => {
                const name = path.replace(/\\/g, "/").split("/").pop() ?? path;
                return (
                  <div key={`folder-${path}-${index}`} className="flex items-center gap-1.5 px-2.5 py-1.5 bg-blue-500/10 border border-blue-500/20 rounded-lg text-xs text-blue-400" title={path}>
                    <Folder size={12} className="shrink-0" />
                    <span className="truncate max-w-[150px] font-medium">{name}</span>
                    <button type="button" onClick={() => setPendingFolders((prev) => prev.filter((_, i) => i !== index))} className="text-blue-400/60 hover:text-blue-300" aria-label={`Remove ${name}`}>
                      <X size={11} />
                    </button>
                  </div>
                );
              })}
            </div>
          )}
          {isDragging && (
            <div className="flex items-center justify-center py-3 border-2 border-dashed border-blue-500/40 rounded-xl">
              <span className="text-sm text-blue-400">Drop files or folders here</span>
            </div>
          )}
          <div className="flex gap-2 relative">
            {atOpen && mentions.length > 0 && (
              <div role="listbox" aria-label="Files" className="absolute bottom-full left-0 mb-1 z-20 w-full max-w-md rounded-xl border border-th-border bg-th-dropdown-bg p-1 shadow-xl overflow-hidden animate-pop-in">
                {mentions.slice(0, 8).map((item, index) => {
                  const { name, dir } = splitPath(item.path);
                  return (
                    <button
                      key={item.path}
                      type="button"
                      role="option"
                      aria-selected={index === atIndex}
                      className={`w-full flex items-center gap-2 rounded-lg text-left px-2.5 py-1.5 text-xs ${index === atIndex ? "bg-th-surface-hover text-th-text-primary" : "text-th-text-secondary"}`}
                      onMouseEnter={() => setAtIndex(index)}
                      onMouseDown={(event) => { event.preventDefault(); insertMention(item.path); }}
                    >
                      {item.is_dir ? <Folder size={13} className="shrink-0 text-th-accent" /> : <FileText size={13} className="shrink-0 text-th-text-muted" />}
                      <span className="shrink-0 font-medium">{highlightMatch(name, atFilter)}</span>
                      {dir && <span className="truncate text-th-text-muted">{highlightMatch(dir, atFilter)}</span>}
                    </button>
                  );
                })}
              </div>
            )}
            <div className="relative flex-1 min-w-0">
            <InlineUrlInput
              ref={composerRef}
              value={input}
              onChange={setInput}
              onMentionChange={(query) => {
                if (query === null) { setAtOpen(false); return; }
                setAtOpen(true);
                setAtFilter(query);
                setAtIndex(0);
              }}
              onPasteFiles={(files) => { addFiles(files); return true; }}
              onPastePaths={(paths) => { addDroppedPaths(paths); return true; }}
              onPasteMaybeFolder={(files) => { void tryNativeClipboardPaste(files); }}
              onKeyDown={(e) => {
                if (atOpen && mentions.length > 0) {
                  if (e.key === "ArrowDown") { e.preventDefault(); setAtIndex((index) => Math.min(index + 1, Math.min(mentions.length, 8) - 1)); return; }
                  if (e.key === "ArrowUp") { e.preventDefault(); setAtIndex((index) => Math.max(index - 1, 0)); return; }
                  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); insertMention(mentions[Math.min(atIndex, mentions.length - 1)].path); return; }
                  if (e.key === "Escape") { e.preventDefault(); setAtOpen(false); return; }
                }
                if (e.key === "Escape" && streaming) {
                  e.preventDefault();
                  void handleStop();
                  return;
                }
                if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && pendingCard) {
                  e.preventDefault();
                  decidePending("approve");
                  return;
                }
                if ((e.metaKey || e.ctrlKey) && e.key === "Backspace" && pendingCard) {
                  e.preventDefault();
                  decidePending("reject");
                  return;
                }
                if (e.key === "Tab" && e.shiftKey) {
                  e.preventDefault();
                  cycleMode();
                  return;
                }
                // Enter sends; Shift+Enter inserts a newline.
                if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                  e.preventDefault();
                  submit();
                }
              }}
              placeholder={
                streaming ? "Add context for the running turn…"
                  : hasAttachments ? "Add a message about this…"
                  : !sessionId ? (grantId ? `Describe a change to start in ${selected?.label ?? "this project"}` : "Allow a project folder to start")
                  : !session?.workspace ? "No project folder on this session"
                    : mode === "ask" ? "Ask about the code…"
                      : mode === "plan" ? "Describe what to plan…"
                        : mode === "debug" ? "Describe the bug…"
                          : "Ask for a change…  @ to mention a file"
              }
              disabled={!sessionId && (!grantId || starting)}
              className="w-full bg-th-inset-bg border border-th-border rounded-xl pl-9 pr-3 py-2 text-sm text-th-text-primary overflow-y-auto max-h-[200px]"
            />
            <button
              type="button"
              title="Attach files"
              aria-label="Attach files"
              disabled={uploading || (!sessionId && !grantId)}
              onClick={() => fileInputRef.current?.click()}
              className="absolute left-1.5 top-1/2 -translate-y-1/2 w-7 h-7 rounded-lg flex items-center justify-center text-th-text-muted hover:text-th-text-primary hover:bg-th-surface-hover disabled:opacity-40"
            >
              <Plus size={16} />
            </button>
            </div>
            {streaming ? (
              <IconButton
                label="Stop agent"
                keys={["Esc"]}
                icon={<Square size={14} />}
                variant="secondary"
                size="md"
                onClick={() => { void handleStop(); }}
                className="!h-auto !w-10 !rounded-xl !text-th-danger hover:!bg-th-danger/10"
              />
            ) : (
              <IconButton
                type="submit"
                label="Send"
                keys={["⏎"]}
                icon={<SendHorizontal size={15} />}
                variant="primary"
                size="md"
                disabled={uploading || (!sessionId && (!grantId || starting)) || (!input.trim() && !hasAttachments)}
                className="!h-auto !w-10 !rounded-xl"
              />
            )}
          </div>
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1 min-w-0">
            <Segmented
              label="Mode"
              value={mode}
              disabled={streaming}
              onChange={(id) => { void changeMode(id); }}
              options={MODES.map((item) => ({ id: item.id, label: item.label, icon: item.icon, hint: `${item.hint} (Shift+Tab to switch)` }))}
            />
            {(mode === "plan" || mode === "debug") && <Badge title="Plan and Debug always run Deep">Deep</Badge>}
            {pendingCard ? (
              <span className="flex items-center gap-1 text-2xs text-th-text-muted shrink-0" title="High-risk cards still wait for a click">
                <Kbd>⌘</Kbd><Kbd>⏎</Kbd> allow <Kbd>⌘</Kbd><Kbd>⌫</Kbd> reject
              </span>
            ) : (
              <span className="hidden lg:flex items-center gap-1 text-2xs text-th-text-faint shrink-0">
                <Kbd>⇧</Kbd><Kbd>⇥</Kbd> mode <Kbd>@</Kbd> file {streaming && <><Kbd>Esc</Kbd> stop</>}
              </span>
            )}
            <div className="ml-auto max-w-full" title={isModelBusy ? "Model is loading" : "Changes the model for all chats"}>
              <ModelPicker align="right" value={currentModel} models={availableModels} onChange={(id) => { void handleModelChange(id); }} />
            </div>
          </div>
        </form>
      </section>

      {inspectorOpen && dock === "right" && <div
        role="separator"
        aria-orientation="vertical"
        aria-label="Inspector width"
        aria-valuemin={RIGHT_MIN}
        aria-valuemax={RIGHT_MAX}
        aria-valuenow={rightWidth}
        tabIndex={0}
        title="Drag to resize. Double-click to reset."
        onPointerDown={(event) => startResize("right", event)}
        onDoubleClick={() => resizePanel("right", RIGHT_DEFAULT)}
        onKeyDown={(event) => {
          if (event.key === "ArrowLeft") resizePanel("right", rightWidth + 16);
          if (event.key === "ArrowRight") resizePanel("right", rightWidth - 16);
          if (event.key === "Home") resizePanel("right", RIGHT_DEFAULT);
        }}
        className="w-1.5 shrink-0 cursor-col-resize bg-transparent hover:bg-sky-400/50 active:bg-sky-400/70 focus:outline-none focus:bg-sky-400/40"
      />}
      {inspectorOpen && dock === "right" && <aside style={{ width: narrow ? 280 : rightWidth }} className="shrink-0 border-l border-th-border flex flex-col min-h-0 overflow-hidden">
        {inspectorPane}
      </aside>}
      {planConfirm && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/40 backdrop-blur-sm p-6 animate-fade-in" role="dialog" aria-modal="true" aria-label="Build this plan" onKeyDown={(event) => { if (event.key === "Escape") setPlanConfirm(false); }}>
          <div className="w-full max-w-lg max-h-[70vh] flex flex-col rounded-2xl border border-th-border bg-th-bg-secondary shadow-2xl animate-pop-in">
            <div className="px-4 py-3 border-b border-th-border text-sm text-th-text-primary">Build this plan?</div>
            <pre className="flex-1 overflow-auto px-4 py-3 text-2xs whitespace-pre-wrap text-th-text-secondary">{planText || "The plan is in /output/plan.md."}</pre>
            <div className="px-4 py-3 border-t border-th-border flex justify-end gap-2">
              <Button variant="ghost" onClick={() => setPlanConfirm(false)}>Cancel</Button>
              <Button variant="primary" autoFocus onClick={() => { void confirmBuild(); }}><Hammer size={12} /> Build</Button>
            </div>
          </div>
        </div>
      )}
      {undoDelete ? (
        <Toast key={undoDelete.id} duration={0} onDismiss={() => setUndoDelete(null)} action={{ label: "Undo", onClick: undoDeleteSession }}>
          Deleted “{undoDelete.title}”
        </Toast>
      ) : toast ? (
        <Toast duration={0} onDismiss={() => setToast(null)}>{toast}</Toast>
      ) : null}
      {preview && (
        <FolderGrantDialog
          preview={preview}
          mode={grantMode}
          onMode={setGrantMode}
          onCancel={() => setPreview(null)}
          onConfirm={() => { void confirmGrant(); }}
        />
      )}
    </div>
  );
}
