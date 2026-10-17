import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { FolderPlus, Hammer, MessageSquare, Plus, RotateCcw, Send, Square, Trash2, Undo2, X } from "lucide-react";
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

type Inspector = "changes" | "files" | "output" | "timeline" | "plan";

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

function storedWidth(key: string, fallback: number, min: number, max: number): number {
  const raw = Number(localStorage.getItem(key));
  if (!Number.isFinite(raw)) return fallback;
  return Math.min(max, Math.max(min, raw));
}

const MODES: { id: SessionMode; label: string; hint: string }[] = [
  { id: "agent", label: "Agent", hint: "Reads and edits the project, runs commands." },
  { id: "ask", label: "Ask", hint: "Read-only. Answers questions about the code and cannot change anything." },
  { id: "plan", label: "Plan", hint: "Read-only in the project. Writes a plan to review, then you build it." },
  { id: "debug", label: "Debug", hint: "Investigates first: reproduce, hypothesize, log, fix, clean up." },
];

function asMode(value: unknown): SessionMode {
  return value === "ask" || value === "plan" || value === "debug" ? value : "agent";
}

function modeBadge(mode: SessionMode): { label: string; className: string } | null {
  if (mode === "ask") return { label: "Ask mode", className: "text-sky-300" };
  if (mode === "plan") return { label: "Plan mode", className: "text-violet-300" };
  if (mode === "debug") return { label: "Debug mode", className: "text-amber-300" };
  return null;
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
  const [searchParams] = useSearchParams();
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
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [preview, setPreview] = useState<FolderGrantPreview | null>(null);
  const [grantMode, setGrantMode] = useState<"read" | "write">("write");
  const [notice, setNotice] = useState<string | null>(null);
  const [mode, setMode] = useState<SessionMode>(() => asMode(localStorage.getItem(LAST_MODE)));
  const [hasPlan, setHasPlan] = useState(false);
  const [planRev, setPlanRev] = useState(0);
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
    api.getWorkspaceChanges(sessionId).then(setChanges).catch(() => setChanges(null));
    api.getWorkspaceGit(sessionId).then(setGit).catch(() => setGit(null));
  }, [sessionId, session?.workspace?.grant_id]);

  // A finished tool call is folded into its row, so count them: an edit that
  // completes does not add a message but should refresh the Changes tab.
  const finishedTools = useMemo(() => messages.filter((m) => m.type === "tool_result").length, [messages]);
  useEffect(() => { refreshChanges(); }, [refreshChanges, messages.length, finishedTools]);

  useEffect(() => {
    if (!sessionId || !diffPath) {
      setDiff("");
      return;
    }
    api.getWorkspaceDiff(sessionId, diffPath).then((row) => setDiff(row.diff || row.message || "")).catch(() => setDiff(""));
  }, [sessionId, diffPath, changes]);

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

  const { connected, send, sendHitlResponse, waitForConnection } = useWebSocket({
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
    if (!sessionId || (!streaming && !awaitingPoll)) return;
    const timer = setInterval(() => { void syncWithServer(); }, 2000);
    return () => clearInterval(timer);
  }, [sessionId, streaming, awaitingPoll, syncWithServer]);

  // After the socket comes back, frames sent while it was down are gone.
  const wasConnected = useRef(false);
  useEffect(() => {
    const reconnected = connected && !wasConnected.current;
    wasConnected.current = connected;
    if (reconnected && sessionId) void syncWithServer();
  }, [connected, sessionId, syncWithServer]);

  const output = useMemo(() => messages.flatMap((m) => {
    if (m.type !== "tool_call" && m.type !== "tool_result") return [];
    const isExecute = m.content === "execute" || m.metadata?.name === "execute";
    if (!isExecute) return [];
    const args = m.metadata?.args as { command?: unknown } | undefined;
    const command = typeof args?.command === "string" ? args.command : "";
    // A call row holds the live lines until its result lands in `result`.
    // A result with no call row keeps the output as its own content.
    const live = (m.metadata?.liveOutput as string[] | undefined)?.join("\n") ?? "";
    const text = typeof m.metadata?.result === "string"
      ? m.metadata.result
      : m.content !== "execute" ? m.content : live;
    return [{ id: m.id, command, text, done: m.type === "tool_result" }];
  }), [messages]);

  const thoughtFlags = useMemo(() => computeThoughtFlags(messages), [messages]);
  const latestTodoFlags = useMemo(() => computeLatestTodoFlags(messages), [messages]);
  const renderItems = useMemo(() => buildRenderItems(messages, thoughtFlags), [messages, thoughtFlags]);
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
    setArtifact({ path, fileUrl, type });
    setInspector("files");
  }, []);

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
    if (!text) return;
    if (sessionId) {
      if (sendText(text, mode)) setInput("");
      return;
    }
    void submitToNewSession(text);
  };

  // Typing with no session open starts one in the selected project and sends
  // the text as its first message. The socket for the new session opens after
  // the route changes, so the send waits for it (see the effect below).
  const submitToNewSession = async (text: string) => {
    if (!grantId || startingRef.current) return;
    startingRef.current = true;
    setStarting(true);
    try {
      const sid = await startSession();
      if (!sid) return;
      pendingSendRef.current = { sid, text, mode };
      setInput("");
    } finally {
      startingRef.current = false;
      setStarting(false);
    }
  };

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

  const buildPlan = async () => {
    if (!sessionId) return;
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

  const deleteSession = async (id: string) => {
    setConfirmDeleteId(null);
    try {
      await api.closeSession(id);
    } catch (e) {
      setError(apiMessage(e));
      return;
    }
    setSessions((prev) => prev.filter((item) => item.id !== id));
    if (id === sessionIdRef.current) {
      setStreaming(false);
      setAwaitingPoll(false);
      navigate("/code");
    }
  };

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

  // When the model answers with a plan in chat instead of saving
  // /output/plan.md, the final answer of the turn is the plan.
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

  return (
    <div ref={shellRef} className="flex h-full min-h-0 min-w-0 overflow-hidden">
      <aside style={{ width: leftWidth }} className="shrink-0 border-r border-th-border flex flex-col min-h-0 min-w-0 overflow-hidden bg-th-bg-secondary">
        <div className="px-3 py-2 flex items-center justify-between">
          <span className="text-[11px] uppercase tracking-wider text-th-text-muted">Projects</span>
          <button type="button" title="Allow a folder" onClick={() => { void allowFolder(); }} className="text-th-text-muted hover:text-th-text-primary">
            <FolderPlus size={14} />
          </button>
        </div>
        <div className="px-2 space-y-1 overflow-y-auto max-h-40">
          {grants.map((grant) => (
            <div
              key={grant.id}
              className={`flex items-center rounded ${grant.id === grantId ? "bg-th-surface-hover" : ""}`}
            >
              <button
                type="button"
                onClick={() => setGrantId(grant.id)}
                className={`flex-1 min-w-0 text-left px-2 py-1 text-xs truncate ${grant.id === grantId ? "text-th-text-primary" : "text-th-text-secondary"}`}
                title={grant.realpath}
              >
                {grant.label}
                <span className="text-th-text-muted"> {grant.mode === "read" ? "· read" : ""}</span>
              </button>
              {grant.id === grantId && (
                <button
                  type="button"
                  title="Remove this project"
                  aria-label={`Remove ${grant.label}`}
                  onClick={() => { void removeProject(grant); }}
                  className="shrink-0 px-1.5 py-1 text-th-text-muted hover:text-red-400"
                >
                  <X size={12} />
                </button>
              )}
            </div>
          ))}
          {grants.length === 0 && <p className="px-2 text-[11px] text-th-text-muted">Allow a project folder to start.</p>}
        </div>
        <div className="px-3 py-2 flex items-center justify-between border-t border-th-border">
          <span className="text-[11px] uppercase tracking-wider text-th-text-muted">Sessions</span>
          <button type="button" title="New coding session" onClick={() => { void startSession(); }} disabled={!grantId} className="text-th-text-muted hover:text-th-text-primary disabled:opacity-40">
            <Plus size={14} />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto px-2 space-y-1">
          {sessions.map((item) => (
            <div
              key={item.id}
              className={`group flex items-center rounded ${item.id === sessionId ? "bg-th-surface-hover" : ""}`}
            >
              <button
                type="button"
                onClick={() => navigate(`/code/${item.id}`)}
                className={`flex-1 min-w-0 text-left px-2 py-1 text-xs truncate ${item.id === sessionId ? "text-th-text-primary" : "text-th-text-secondary"}`}
              >
                <span className={`inline-block w-1.5 h-1.5 rounded-full mr-1.5 ${item.status === "running" ? "bg-emerald-400" : "bg-th-text-muted"}`} />
                {item.title || "Untitled"}
              </button>
              {confirmDeleteId === item.id ? (
                <div className="shrink-0 flex items-center gap-1 pr-1.5 text-[11px]">
                  <button
                    type="button"
                    onClick={() => { void deleteSession(item.id); }}
                    className="text-red-400 hover:text-red-300"
                  >
                    Delete
                  </button>
                  <button
                    type="button"
                    onClick={() => setConfirmDeleteId(null)}
                    className="text-th-text-muted hover:text-th-text-primary"
                  >
                    Cancel
                  </button>
                </div>
              ) : (
                <button
                  type="button"
                  title="Delete this session"
                  aria-label={`Delete ${item.title || "Untitled"}`}
                  onClick={() => setConfirmDeleteId(item.id)}
                  className="shrink-0 px-1.5 py-1 text-th-text-muted opacity-0 group-hover:opacity-100 focus:opacity-100 hover:text-red-400"
                >
                  <Trash2 size={12} />
                </button>
              )}
            </div>
          ))}
        </div>
        {session?.workspace && sessionId && (
          <div className="border-t border-th-border max-h-64 overflow-y-auto py-1">
            <WorkspaceTree
              sessionId={sessionId}
              fileUrl={(rel) => api.getWorkspaceFileUrl(sessionId, rel)}
              onOpenFile={(rel) => openFile(rel)}
            />
          </div>
        )}
      </aside>
      <div
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
      />

      <section className="flex-1 min-w-0 flex flex-col min-h-0 overflow-hidden">
        <header className="px-4 py-2 border-b border-th-border flex items-center gap-3 text-xs min-w-0 overflow-hidden">
          <span className="font-medium text-th-text-primary truncate min-w-[4rem]">
            {sessionId && !session?.workspace ? "No project" : (session?.workspace?.name || selected?.label || "Code")}
          </span>
          {session?.workspace && (
            <button
              type="button"
              title="Remove this project from the session"
              disabled={streaming}
              onClick={() => { void unmapProject(); }}
              className="text-th-text-muted hover:text-th-text-primary disabled:opacity-40 shrink-0"
            >
              <X size={12} />
            </button>
          )}
          {git?.git && <span className="text-th-text-muted shrink-0">{git.branch}{git.dirty ? ` · ${git.dirty} dirty` : ""}</span>}
          {modeBadge(mode) && <span className={`shrink-0 ${modeBadge(mode)?.className}`}>{modeBadge(mode)?.label}</span>}
          {session?.workspace?.mode === "read" && <span className="text-sky-300 shrink-0">Read only</span>}
          {session?.workspace?.mode === "write" && mode === "agent" && <span className="text-th-text-muted shrink-0" title="Edits are checkpointed and can be reverted">Editable</span>}
          <span className="ml-auto flex items-center gap-2 shrink-0">
            <span className={connected ? "text-emerald-400" : "text-amber-400"}>{connected ? "Live" : "Connecting"}</span>
            {sessionId && (
              <>
                <button type="button" className="text-th-text-tertiary hover:text-th-text-primary" onClick={() => navigate(`/chat/${sessionId}`)}>Chat</button>
                <button type="button" className="text-th-text-tertiary hover:text-th-text-primary" onClick={() => navigate(`/runs/${sessionId}`)}>Run</button>
              </>
            )}
          </span>
        </header>
        {notice && (
          <p className="px-4 py-1.5 text-[11px] bg-amber-500/10 text-amber-300 border-b border-amber-500/20">
            No checkpoint for this turn: {notice}
          </p>
        )}
        {error && <p className="px-4 py-1 text-xs text-red-400">{error}</p>}
        {(changes?.debug_markers ?? 0) > 0 && (
          <p className="px-4 py-1.5 text-[11px] bg-amber-500/10 text-amber-300 border-b border-amber-500/20">
            {changes?.debug_markers} otto-debug log marker{changes?.debug_markers === 1 ? "" : "s"} still in your files. Remove them before you commit.
          </p>
        )}
        <div
          className="flex-1 overflow-y-auto px-4 py-4 space-y-4"
          onScroll={(e) => {
            const el = e.currentTarget;
            stickToBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
          }}
        >
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center text-center">
              <MessageSquare className="text-th-text-muted mb-3" size={28} />
              <p className="text-sm text-th-text-secondary">
                {sessionId && !session?.workspace
                  ? "This session has no project folder."
                  : "Describe a change in the allowed folder."}
              </p>
            </div>
          )}
          <MessageList
            items={renderItems}
            thoughtFlags={thoughtFlags}
            latestTodoFlags={latestTodoFlags}
            isStreaming={streaming}
            currentSessionId={sessionId ?? null}
            resolveSubagentModel={resolveSubagentModel}
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
        {planReady && sessionId && (
          <div className="px-4 py-2 border-t border-th-border flex items-center gap-3 text-xs bg-violet-500/5">
            <span className="text-th-text-secondary flex-1 min-w-0 truncate">Plan ready. Review it, then build.</span>
            <button type="button" onClick={() => setInspector("plan")} className="text-th-text-tertiary hover:text-th-text-primary">View plan</button>
            <button type="button" onClick={() => { void buildPlan(); }} className="flex items-center gap-1 px-2 py-1 rounded-lg border border-th-border text-th-text-primary hover:bg-th-surface-hover">
              <Hammer size={12} /> Build this plan
            </button>
          </div>
        )}
        <form
          className="p-3 border-t border-th-border space-y-2"
          onSubmit={(e) => { e.preventDefault(); submit(); }}
        >
          <div className="flex gap-2">
            <textarea
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              ref={(el) => {
                if (!el) return;
                el.style.height = "auto";
                el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
              }}
              onKeyDown={(e) => {
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
                !sessionId ? (grantId ? `Describe a change to start a session in ${selected?.label ?? "this project"}` : "Allow a project folder to start")
                  : !session?.workspace ? "No project folder on this session"
                    : mode === "ask" ? "Ask about the code…"
                      : mode === "plan" ? "Describe what to plan…"
                        : mode === "debug" ? "Describe the bug…"
                          : "Ask for a change…"
              }
              disabled={!sessionId && (!grantId || starting)}
              className="flex-1 bg-th-inset-bg border border-th-border rounded-xl px-3 py-2 text-sm text-th-text-primary resize-none overflow-y-auto"
            />
            {streaming ? (
              <button
                type="button"
                title="Stop agent"
                aria-label="Stop agent"
                onClick={() => { void handleStop(); }}
                className="px-3 rounded-xl border border-th-border text-red-400 hover:bg-red-500/15"
              >
                <Square size={14} />
              </button>
            ) : (
              <button type="submit" title="Send" aria-label="Send" disabled={(!sessionId && (!grantId || starting)) || !input.trim()} className="px-3 rounded-xl border border-th-border text-th-text-secondary disabled:opacity-40">
                <Send size={14} />
              </button>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1 min-w-0">
            <div className="flex rounded-lg border border-th-border overflow-hidden shrink-0" role="radiogroup" aria-label="Mode">
              {MODES.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  role="radio"
                  aria-checked={mode === item.id}
                  title={`${item.hint} (Shift+Tab to switch)`}
                  disabled={streaming}
                  onClick={() => { void changeMode(item.id); }}
                  className={`px-2.5 py-0.5 text-[11px] disabled:opacity-60 ${mode === item.id ? "bg-th-tab-active-bg text-th-tab-active-fg" : "text-th-text-tertiary hover:text-th-text-primary"}`}
                >
                  {item.label}
                </button>
              ))}
            </div>
            {(mode === "plan" || mode === "debug") && <span className="text-[11px] text-th-text-muted shrink-0">Deep</span>}
            <div className="ml-auto max-w-full" title={isModelBusy ? "Model is loading" : "Changes the model for all chats"}>
              <ModelPicker align="right" value={currentModel} models={availableModels} onChange={(id) => { void handleModelChange(id); }} />
            </div>
          </div>
        </form>
      </section>

      <div
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
      />
      <aside style={{ width: rightWidth }} className="shrink-0 border-l border-th-border flex flex-col min-h-0 overflow-hidden">
        <div className="flex border-b border-th-border text-[11px]">
          {(["changes", "files", "output", "timeline", ...(hasPlan || mode === "plan" ? ["plan"] : [])] as Inspector[]).map((tab) => (
            <button
              key={tab}
              type="button"
              onClick={() => setInspector(tab)}
              className={`flex-1 py-2 capitalize ${inspector === tab ? "text-th-text-primary border-b border-sky-400" : "text-th-text-muted"}`}
            >
              {tab}
            </button>
          ))}
        </div>
        <div className="flex-1 overflow-y-auto p-3 text-xs">
          {inspector === "changes" && (
            <div className="space-y-2">
              {changes?.message && <p className="text-th-text-muted">{changes.message}</p>}
              {(changes?.files || []).map((file) => (
                <div key={file.path} className="flex items-center gap-2">
                  <button type="button" className="truncate text-left flex-1 font-mono hover:text-th-text-primary text-th-text-secondary" onClick={() => setDiffPath(file.path)}>
                    {file.path}
                  </button>
                  <button
                    type="button"
                    title={`Revert ${file.path}`}
                    aria-label={`Revert ${file.path}`}
                    className="shrink-0 p-1 rounded text-th-text-muted hover:text-th-text-primary hover:bg-th-surface-hover"
                    onClick={() => {
                      if (!sessionId) return;
                      api.revertWorkspace(sessionId, [file.path]).then(() => refreshChanges()).catch((e) => setError(apiMessage(e)));
                    }}
                  >
                    <Undo2 size={13} />
                  </button>
                </div>
              ))}
              {diff && (
                <pre className="whitespace-pre-wrap font-mono text-[11px] bg-th-inset-bg rounded-lg p-2 text-th-text-secondary">{diff}</pre>
              )}
              {sessionId && (changes?.files?.length || 0) > 0 && (
                <button
                  type="button"
                  title="Revert all changes from this turn"
                  aria-label="Revert all changes from this turn"
                  className="p-1 rounded text-th-text-muted hover:text-th-text-primary hover:bg-th-surface-hover"
                  onClick={() => api.revertWorkspace(sessionId).then(() => refreshChanges()).catch((e) => setError(apiMessage(e)))}
                >
                  <RotateCcw size={13} />
                </button>
              )}
            </div>
          )}
          {inspector === "files" && artifact && (
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
          {inspector === "output" && (
            <div className="space-y-3">
              {output.length === 0 && <p className="text-th-text-muted">Command output shows up here.</p>}
              {output.map((item) => (
                <div key={item.id} className="space-y-1">
                  {item.command && <p className="font-mono text-[11px] text-th-text-muted break-all">$ {item.command}</p>}
                  <pre className="whitespace-pre-wrap font-mono text-[11px] text-th-text-secondary bg-th-inset-bg rounded-lg p-2">{item.text || (item.done ? "(no output)" : "Running…")}</pre>
                </div>
              ))}
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
      </aside>
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
