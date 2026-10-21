import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { useNavigate, useParams } from "react-router-dom";
import { Send, Square, ChevronUp, ChevronRight, FileText, FolderOpen, ExternalLink, CheckCircle2, Plus, X, Loader2, Calendar, Brain, Cpu, ArrowUpLeft, ArrowLeft, Folder, GitBranch, RefreshCw, MessageSquarePlus, Mic, MicOff } from "lucide-react";
import { api } from "../hooks/useApi";
import { useWebSocket } from "../hooks/useWebSocket";
import { AgentGraph } from "../components/chat/AgentGraph";
import { MessageBubble } from "../components/chat/MessageBubble";
import { MessageList } from "../components/chat/MessageList";
import { ArtifactPanel, artifactTypeFromPath } from "../components/chat/ArtifactPanel";
import { ThinkingIndicator } from "../components/chat/ThinkingIndicator";
import type { Artifact, ArtifactType } from "../components/chat/ArtifactPanel";
import { ModelPicker } from "../components/chat/ModelPicker";
import { DepthPicker, type RunDepth } from "../components/chat/DepthPicker";
import SessionStatsPanel from "../components/chat/SessionStatsPanel";
import SessionFileTree from "../components/chat/SessionFileTree";
import OutputFileGrid from "../components/chat/OutputFileGrid";
import WorkspaceTree from "../components/chat/WorkspaceTree";
import { ChangedFilesStrip, collectChangedFiles, filePathFromToolArgs } from "../components/chat/ChangedFilesStrip";
import InlineUrlInput, { type InlineUrlInputHandle } from "../components/chat/InlineUrlInput";
import { formatFileSize } from "../utils/formatFileSize";
import { mergeToolMessages, preserveHitlResolved, appendMissingInterrupts, pendingInterrupts, hasUnmatchedApprovalTool, computeThoughtFlags } from "../utils/mergeToolMessages";
import { familyChipClasses } from "../utils/subagentModelChip";
import { screenHighRiskCommand } from "../utils/highRiskCommands";
import { useNotification } from "../context/NotificationContext";
import { useConnection } from "../context/ConnectionContext";
import { useTheme } from "../context/ThemeContext";
import { useChatModel, isDistillCatalogId } from "../hooks/useChatModel";
import { CRON_PRESETS } from "../types";
import type { AgentSpec, ChatMessage, FolderGrantPreview, FolderGrant, MlxDownloadJob, SessionInfo, SessionWorkspace, WSMessage } from "../types";
import { FolderGrantDialog } from "../components/action/FolderGrantDialog";
import { ProjectFolderMenu } from "../components/action/ProjectFolderMenu";
import { pickDirectory } from "../utils/pickFolder";
import { useVoice } from "../hooks/useVoice";
import { onAskOtto } from "../utils/askOttoBus";
import { subscribeSessionFiles } from "../utils/sessionFilesBus";
import { getFollowFileEdits } from "../utils/followFileEdits";
import { onAgentContext } from "../utils/agentContextBus";
import { buildFileTree } from "../utils/fileTree";
import { buildRenderItems, computeLatestTodoFlags } from "../utils/renderItems";
import { isActivityFrame } from "../utils/liveMessages";
import type { AskPayload } from "../utils/askOttoBus";
import type { AskImage } from "../types";
import logoDark from "../assets/logo-dark.png";
import logoLight from "../assets/logo-light.png";

/** Decode a base64 data URL into a File for upload. Returns null if malformed. */
function dataUrlToFile(img: AskImage): File | null {
  try {
    const [meta, b64] = img.dataUrl.split(",");
    if (!b64) return null;
    const mime = /data:(.*?)(;|$)/.exec(meta)?.[1] ?? "image/png";
    const bin = atob(b64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    return new File([arr], img.name, { type: mime });
  } catch {
    return null;
  }
}

function workspaceErrorText(error: unknown, fallback: string): string {
  const text = error instanceof Error ? error.message : fallback;
  const at = text.indexOf("{");
  if (at >= 0) {
    try {
      const body = JSON.parse(text.slice(at)) as { error?: string };
      if (body.error) return body.error;
    } catch { /* raw */ }
  }
  return text;
}

const FILE_WRITE_TOOLS = new Set(["edit_file", "write_file", "edit"]);

function normalizeVirtualPath(path: string): string {
  return path.replace(/\\/g, "/").replace(/^\/+/, "");
}

/** Tool paths (`/output/a.txt`) and listing paths (`output/a.txt`) name one file. */
function virtualPathsMatch(a: string, b: string): boolean {
  return normalizeVirtualPath(a) === normalizeVirtualPath(b);
}

function fileListStamp(
  path: string,
  files: { path: string; size: number; modified_at: number }[],
): string | null {
  const file = files.find((f) => virtualPathsMatch(f.path, path));
  return file ? `${file.modified_at}:${file.size}` : null;
}

function latestWriteResultId(path: string, messages: ChatMessage[]): string | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.type !== "tool_result" || !FILE_WRITE_TOOLS.has(m.content)) continue;
    const written = filePathFromToolArgs(m.metadata?.args as Record<string, unknown> | undefined);
    if (written && virtualPathsMatch(written, path)) return m.id;
  }
  return null;
}

// When rebuilding messages from the API (which never persists base64 images),
// carry any images that are already in local state forward into the new array.
function preserveImages(apiMerged: ChatMessage[], prev: ChatMessage[]): ChatMessage[] {
  return apiMerged.map((m, i) => {
    const local = prev[i];
    if (!local) return m;
    const apiImages = (m.metadata?.images as unknown[] | undefined);
    const localImages = (local.metadata?.images as unknown[] | undefined);
    if (!apiImages?.length && localImages?.length) {
      return { ...m, metadata: { ...m.metadata, images: localImages } };
    }
    return m;
  });
}

function reconcileApiMessages(apiMerged: ChatMessage[], prev: ChatMessage[]): ChatMessage[] {
  return preserveHitlResolved(preserveImages(apiMerged, prev), prev);
}

export default function ChatPage() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();
  const { theme } = useTheme();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState(() => localStorage.getItem("chatDraft") ?? "");
  const [isStreaming, setIsStreaming] = useState(false);
  // Keep the 2s transcript poll alive after the agent task stops when a
  // HITL/ask_user card is expected but has not arrived on the WebSocket yet.
  const [pollForInterrupt, setPollForInterrupt] = useState(false);
  const [streamPhase, setStreamPhase] = useState<"thinking" | "memory_search">("thinking");
  const [pendingContext, setPendingContext] = useState<string[]>([]);
  const [agents, setAgents] = useState<AgentSpec[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<string>(() => localStorage.getItem("chatSelectedAgent") ?? "");
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(sessionId ?? null);
  const [showScheduleDialog, setShowScheduleDialog] = useState(false);
  const [scheduleSuccess, setScheduleSuccess] = useState<string | null>(null);
  const [showContextHint, setShowContextHint] = useState(false);
  const hintTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // True once the user dismisses the hint for the current streaming run.
  // Resets to false at the start of each new streaming run so the hint
  // reappears when the user sends their next message.
  const hintDismissedRef = useRef(false);
  // Shadow ref so the timer cycle can read the current input without being a dep.
  const inputValueRef = useRef(input);
  inputValueRef.current = input;
  const [graphOpen, setGraphOpen] = useState(false);
  // When the orchestrator hands off via spawn_followup_session, the
  // child session records its parent here.  Used to render a "← from
  // parent" link badge in the header that jumps the user back.  Null
  // for any session that wasn't spawned (i.e. every root session).
  const [parentSessionId, setParentSessionId] = useState<string | null>(null);
  const [openArtifact, setOpenArtifact] = useState<Artifact | null>(null);
  const [artifactWidth, setArtifactWidth] = useState(480);
  const artifactDragRef = useRef<{ startX: number; startWidth: number } | null>(null);
  const artifactDragCleanupRef = useRef<(() => void) | null>(null);

  const handleOpenArtifact = useCallback((path: string, fileUrl: string, type: ArtifactType) => {
    const sid = sessionIdRef.current;
    const ws = workspaceRef.current;
    let url = fileUrl;
    if (sid && ws) {
      const stripped = path.replace(/^\//, "");
      const prefix = ws.virtual_path.replace(/^\//, "");
      if (stripped === prefix || stripped.startsWith(`${prefix}/`)) {
        const rel = stripped === prefix ? "" : stripped.slice(prefix.length + 1);
        url = api.getWorkspaceFileUrl(sid, rel);
      }
    }
    setOpenArtifact({ path, fileUrl: url, type });
  }, []);

  const onArtifactResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    artifactDragRef.current = { startX: e.clientX, startWidth: artifactWidth };
    const onMove = (ev: MouseEvent) => {
      if (!artifactDragRef.current) return;
      const delta = artifactDragRef.current.startX - ev.clientX;
      const next = Math.max(320, Math.min(1200, artifactDragRef.current.startWidth + delta));
      setArtifactWidth(next);
    };
    const onUp = () => {
      artifactDragRef.current = null;
      artifactDragCleanupRef.current = null;
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    artifactDragCleanupRef.current = onUp;
  }, [artifactWidth]);

  // Remove drag listeners if the page unmounts mid-drag.
  useEffect(() => () => artifactDragCleanupRef.current?.(), []);
  // The agent the backend bound to the active session at create time.
  //   undefined ⇒ unknown (not yet fetched, or no active session)
  //   null      ⇒ session runs in orchestrator (general-purpose) mode
  //   string    ⇒ session is locked to that subagent
  // Once a session exists, the chip renders this value and the X
  // button is hidden — the binding is immutable for the session's
  // lifetime, so we don't pretend the user can change it.
  const [sessionBoundAgent, setSessionBoundAgent] = useState<string | null | undefined>(undefined);
  // Full persisted session metadata, used by the live token-stats panel for
  // token totals + cost (throughput is aggregated client-side from messages).
  const [sessionInfo, setSessionInfo] = useState<SessionInfo | null>(null);



  // Pull session metadata so we can render the "← from parent" link
  // when this session was spawned via spawn_followup_session, and the
  // immutable agent binding so the chip reflects what the backend
  // graph is actually running.  We refetch on every session change
  // (cheap; one small JSON call) and tolerate failure silently — the
  // badge just doesn't render.
  // Refetch session metadata on session change and whenever a stream
  // finishes (so the token-stats panel's persisted token totals + cost
  // reflect the just-completed turn).  ``isStreaming`` is a dependency so the
  // false-edge after a turn triggers a refresh.
  useEffect(() => {
    if (!currentSessionId) {
      setParentSessionId(null);
      setSessionBoundAgent(undefined);
      setSessionInfo(null);
      return;
    }
    let cancelled = false;
    api.getSession(currentSessionId)
      .then((info) => {
        if (cancelled) return;
        setParentSessionId(info.parent_session_id ?? null);
        setSessionBoundAgent(info.agent_name);
        setSessionInfo(info);
      })
      .catch(() => {
        if (cancelled) return;
        setParentSessionId(null);
        setSessionBoundAgent(undefined);
        setSessionInfo(null);
      });
    return () => { cancelled = true; };
  }, [currentSessionId, isStreaming]);
  const [sessionFiles, setSessionFiles] = useState<{ path: string; size: number; modified_at: number }[]>([]);
  const sidePreviewOpen = openArtifact != null;
  const [showFiles, setShowFiles] = useState(false);
  const [showWorkspaceTree, setShowWorkspaceTree] = useState(true);
  const [mappingFolder, setMappingFolder] = useState(false);
  const [workspaceError, setWorkspaceError] = useState<string | null>(null);
  const [grantPreview, setGrantPreview] = useState<FolderGrantPreview | null>(null);
  const [grantMode, setGrantMode] = useState<"read" | "write">("write");
  const lastFollowedToolRef = useRef<string | null>(null);
  // What the open side preview has already shown, so a later write or mtime
  // change reloads it and the first observation of a file does not.
  const previewSyncRef = useRef<{
    path: string;
    writeId: string | null;
    stamp: string | null;
    awaitingStamp: boolean;
  } | null>(null);
  const [downloadedFile, setDownloadedFile] = useState<string | null>(null);
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [pendingFolders, setPendingFolders] = useState<string[]>([]);
  const [pendingFilePaths, setPendingFilePaths] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [runDepth, setRunDepth] = useState<RunDepth>(() => {
    const stored = localStorage.getItem("otto.runDepth");
    return stored === "quick" || stored === "deep" || stored === "auto" ? stored : "auto";
  });
  const runDepthChosen = useRef(localStorage.getItem("otto.runDepth") != null);
  const {
    currentModel,
    availableModels,
    appSettings,
    debugLlm,
    mlxDownloadJob,
    mlxIsDownloading,
    exoModelLoading,
    omlxModelLoading,
    isModelBusy,
    handleModelChange,
  } = useChatModel({
    onSettingsLoaded: (s) => {
      if (runDepthChosen.current) return;
      const d = s.orchestrator?.default_run_depth;
      if (d === "auto" || d === "quick" || d === "deep") setRunDepth(d);
    },
    onSessionInvalidated: () => {
      // The live session graph still has the previous model baked in.
      setCurrentSessionId(null);
      navigate("/chat", { replace: true });
    },
  });
  const voiceEnabled = !!(appSettings?.voice?.enabled);
  const wakeEnabled = !!(appSettings?.voice?.wake_enabled);
  const pendingVoiceTranscriptRef = useRef<string | null>(null);
  // Screenshot attachments handed off from the transcription drawer, read by
  // handleSend synchronously (uploaded like drag-dropped files).
  const pendingTranscriptImagesRef = useRef<File[] | null>(null);
  const handleSendRef = useRef<(() => void) | null>(null);
  // Latest input draft, read by the transcript hand-off so it can restore what
  // the user was composing after handleSend clears the box.
  const latestInputRef = useRef("");
  // Note: always-on wake listening is owned by the global listener in Layout,
  // so we don't autoStart here.  This connection receives the shared manager's
  // broadcasts (state, transcript) and drives the mic button + transcript send.
  const voice = useVoice({ enabled: voiceEnabled || wakeEnabled });
  // Mic button: distinguish quick-click (toggle) from hold (PTT)
  const micMouseDownTimeRef = useRef<number | null>(null);
  const micClickLockRef = useRef(false);
  const MIC_HOLD_THRESHOLD_MS = 300;
  const [slashOpen, setSlashOpen] = useState(false);
  const [slashFilter, setSlashFilter] = useState("");
  const [slashIndex, setSlashIndex] = useState(0);
  // "@file" mention dropdown — lists files/folders from the current
  // session's own output directory so the user can reference one inline.
  const [atOpen, setAtOpen] = useState(false);
  const [atFilter, setAtFilter] = useState("");
  const [atIndex, setAtIndex] = useState(0);
  const [atWorkspaceEntries, setAtWorkspaceEntries] = useState<{ path: string; isDir: boolean; pending: false; group: "workspace" }[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  // Whether the history pane should auto-scroll to the newest content.
  // Starts pinned (bottom); flips to false as soon as the user scrolls up
  // more than a small threshold, and back to true once they scroll back
  // down near the bottom or send a new message. A ref (not state) so
  // scrolling doesn't itself trigger a re-render.
  const isPinnedToBottomRef = useRef(true);
  const inputRef = useRef<InlineUrlInputHandle>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const slashRef = useRef<HTMLDivElement>(null);
  const atRef = useRef<HTMLDivElement>(null);
  const msgIdRef = useRef(0);
  const pendingMemoryTopicsRef = useRef<string[] | null>(null);
  const sendingRef = useRef(false);
  const hitlSubmittedRef = useRef<Set<string>>(new Set());
  // After the user answers an approval / ask_user card the backend needs a
  // moment to start the resume. A status poll that lands in that gap still
  // says "awaiting input / not running". Until this deadline the poll must not
  // flip the Stop button back to Send.
  const hitlResumeUntilRef = useRef(0);
  const justCreatedRef = useRef(false);
  const sessionIdRef = useRef(currentSessionId);
  sessionIdRef.current = currentSessionId;
  const workspaceRef = useRef<SessionWorkspace | null>(null);

  // Sync per-session state with the URL prop *during* render — without
  // this, navigating from /chat/A to /chat/B leaves ``messages``,
  // ``sessionFiles``, ``parentSessionId``, ``sessionBoundAgent``, the
  // pending-files draft chips and ``isStreaming`` populated with A's
  // values for the first frame after the URL changes (the reset
  // useEffects only run after commit).  This is the React-recommended
  // "computing state from props" pattern: detect the change and queue
  // the resets here so the very next render shows a clean session B.
  // ``mlxDownloadJob`` is intentionally NOT reset — it tracks a global
  // download job for the selected model, not a per-session resource.
  const [prevSessionId, setPrevSessionId] = useState<string | null | undefined>(sessionId);
  if (sessionId !== prevSessionId) {
    setPrevSessionId(sessionId);
    setCurrentSessionId(sessionId ?? null);
    setMessages([]);
    setIsStreaming(false);
    setStreamPhase("thinking");
    setSessionFiles([]);
    setShowFiles(false);
    setShowWorkspaceTree(true);
    lastFollowedToolRef.current = null;
    previewSyncRef.current = null;
    setParentSessionId(null);
    setSessionBoundAgent(undefined);
    setSessionInfo(null);
    setPendingFiles([]);
    setPendingFilePaths([]);
    setPendingFolders([]);
    setGraphOpen(false);
  }

  const { notify, clearSession, notifications, watchSession, unwatchSession } = useNotification();
  const { setWsConnected, setActiveSessionId, setLastError, clearError } = useConnection();

  useEffect(() => {
    setActiveSessionId(currentSessionId);
  }, [currentSessionId, setActiveSessionId]);

  // Viewing a session clears its pending notification — done, error, or a
  // HITL "needs feedback" chip — since the user is now looking at it.
  useEffect(() => {
    if (currentSessionId && notifications[currentSessionId]) {
      clearSession(currentSessionId);
    }
  }, [currentSessionId, notifications, clearSession]);

  useEffect(() => {
    const onFocus = () => {
      const sid = sessionIdRef.current;
      if (sid) clearSession(sid);

      // WKWebView (Tauri/macOS) throttles JS and socket callbacks when the
      // window is not the key window.  On regaining focus, re-fetch status
      // and messages so anything that was buffered while throttled appears
      // immediately — the same recovery the manual Refresh button performs.
      if (!sid) return;
      api.getSessionStatus(sid).then((status) => {
        // Bail when the user navigated to a different session while the
        // fetch was in flight — applying stale results would corrupt or
        // blank the visible chat.
        if (sid !== sessionIdRef.current) return;
        if (status.running) setIsStreaming(true);
        api.getSessionMessages(sid).then((msgs) => {
          if (sid !== sessionIdRef.current) return;
          if (!msgs || msgs.length === 0) {
            if (!status.running) setIsStreaming(false);
            return;
          }
          const raw: ChatMessage[] = msgs.map((m) => ({
            id: `msg-${++msgIdRef.current}`,
            type: (m.type as ChatMessage["type"]) ?? "agent",
            content: (m.content as string) ?? "",
            metadata: m.metadata as Record<string, unknown> | undefined,
            timestamp: new Date(),
            sessionId: sid,
          }));
          const apiMerged = mergeToolMessages(raw);
          setMessages((prev) => {
            if (apiMerged.length < prev.length) {
              return appendMissingInterrupts(prev, apiMerged) ?? prev;
            }
            if (apiMerged.length === prev.length) {
              const lastApi = apiMerged[apiMerged.length - 1];
              const lastLocal = prev[prev.length - 1];
              if (
                lastApi?.content === lastLocal?.content &&
                lastApi?.type === lastLocal?.type
              ) return prev;
            }
            return reconcileApiMessages(apiMerged, prev).map((m, i) => i < prev.length ? { ...m, id: prev[i].id } : m);
          });
          if (!status.running) setIsStreaming(false);
        }).catch(() => {});
      }).catch(() => {});
    };

    window.addEventListener("focus", onFocus);
    // visibilitychange fires more reliably than "focus" in WKWebView when the
    // OS app window is brought back to the foreground.
    const onVisibility = () => { if (document.visibilityState === "visible") onFocus(); };
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      window.removeEventListener("focus", onFocus);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [clearSession]);

  // Classify absolute OS filesystem paths into files vs. folders and queue
  // them for the send flow. A file is copied into the session (`uploads/`).
  // A folder is symlinked (`createSessionLink`). Shared by Tauri drag-drop
  // below and by pasting a Finder-copied file/folder (`onPastePaths` on
  // `InlineUrlInput`) — both hand us real paths rather than Blob content,
  // which is the only way to represent a pasted/dropped *folder* at all.
  const queueOsPaths = (entries: { path: string; isDir?: boolean }[]) => {
    const folders: string[] = [];
    const filePaths: string[] = [];
    entries.forEach(({ path, isDir }) => {
      const basename = path.replace(/\\/g, "/").split("/").pop() ?? path;
      const asDir = isDir ?? !basename.includes(".");
      if (asDir) folders.push(path);
      else filePaths.push(path);
    });
    if (filePaths.length) setPendingFilePaths((prev) => [...prev, ...filePaths]);
    if (folders.length) setPendingFolders((prev) => [...prev, ...folders]);
  };

  const addDroppedPaths = (paths: string[]) => {
    queueOsPaths(paths.map((path) => ({ path })));
  };

  // Fired when paste had a directory stand-in (0-byte File, webkit directory
  // entry) and no usable `file://` uri-list. Ask the OS clipboard for the
  // real path so a pasted folder goes through the same queue as drag-drop.
  // Never fall back to uploading the 0-byte blob — that's the "0 B folder"
  // chip, and letting WKWebView treat it as a File is what hangs the chat.
  const tryNativeClipboardPaste = async (fallbackFiles: File[] = []) => {
    try {
      const data = await api.clipboardFile();
      const items = data.items?.length
        ? data.items
        : data.path
          ? [{ path: data.path, is_dir: data.is_dir }]
          : [];
      if (items.length) {
        queueOsPaths(items.map((it) => ({ path: it.path, isDir: it.is_dir })));
        return;
      }
    } catch {
      // Backend unreachable — fall through to real blobs if we have any.
    }
    const blobs = fallbackFiles.filter((f) => f.size > 0);
    if (blobs.length) addFiles(blobs);
  };

  // Tauri intercepts OS-level file/folder drops before they reach the browser's
  // ondrop event. Use the Tauri window API to handle them properly.
  // The unlisten function is returned asynchronously, so we use a `cancelled`
  // flag to handle the case where cleanup runs before the promise resolves
  // (which happens every mount in React StrictMode dev, causing double registration).
  useEffect(() => {
    let cancelled = false;
    let unlistenFn: (() => void) | null = null;
    // No-op in a plain browser: getCurrentWindow throws when Tauri internals
    // are absent, and a throw here takes down the whole chat page.
    let win: ReturnType<typeof getCurrentWindow>;
    try {
      win = getCurrentWindow();
    } catch {
      return;
    }
    win.onDragDropEvent((event) => {
      const payload = event.payload as { type: string; paths?: string[] };
      if (payload.type === "hover") {
        setIsDragging(true);
      } else if (payload.type === "leave" || payload.type === "cancel") {
        setIsDragging(false);
      } else if (payload.type === "drop" && payload.paths) {
        setIsDragging(false);
        addDroppedPaths(payload.paths);
      }
    }).then((fn) => {
      if (cancelled) fn(); // cleanup already ran — unregister immediately
      else unlistenFn = fn;
    }).catch(() => {});
    return () => {
      cancelled = true;
      unlistenFn?.();
    };
  }, []);

  useEffect(() => { api.listAgents().then(setAgents).catch((e) => console.warn("Failed to load agents:", e)); }, []);

  const scrollRaf = useRef(0);
  useEffect(() => {
    // Respect the user's scroll position: only auto-scroll on new/streamed
    // content while they're pinned to the bottom. Otherwise every streamed
    // token would yank the view back down, making it impossible to scroll
    // up (or even hold the view still) while a response is being generated.
    if (!isPinnedToBottomRef.current) return;
    if (scrollRaf.current) cancelAnimationFrame(scrollRaf.current);
    scrollRaf.current = requestAnimationFrame(() => {
      scrollRaf.current = 0;
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    });
  }, [messages]);

  // Tracks whether the user is near the bottom of the history pane, so the
  // effect above knows whether to keep auto-scrolling. A generous threshold
  // (matches roughly one line of text) avoids flapping from sub-pixel
  // smooth-scroll settling or minor layout shifts while streaming.
  const HISTORY_SCROLL_BOTTOM_THRESHOLD = 120;
  const handleHistoryScroll = useCallback(() => {
    const el = messagesContainerRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    isPinnedToBottomRef.current = distanceFromBottom < HISTORY_SCROLL_BOTTOM_THRESHOLD;
  }, []);

  useEffect(() => {
    if (!currentSessionId) { setSessionFiles([]); return; }
    const poll = () => {
      api.listSessionFiles(currentSessionId).then(setSessionFiles).catch((e) => console.warn("Failed to list session files:", e));
    };
    poll();
    // While a file is open in the side panel, poll faster so edits land in
    // the preview. Streaming alone still refreshes the file tree.
    if (!isStreaming && !sidePreviewOpen) return;
    const interval = setInterval(poll, sidePreviewOpen ? 2000 : 5000);
    return () => clearInterval(interval);
  }, [currentSessionId, isStreaming, sidePreviewOpen]);

  useEffect(() => {
    if (!sessionId) return;

    // Switching sessions loads a fresh history — always start pinned to
    // the bottom rather than carrying over whatever scroll state the
    // previously viewed session happened to be left in.
    isPinnedToBottomRef.current = true;

    // Sessions spawned from the ambient "Approve & run" flow are marked in
    // localStorage so we skip the empty-session redirect while kick_off_message
    // is still being queued asynchronously on the backend.
    const ambientRunSession = localStorage.getItem("ambientRunSession");
    if (ambientRunSession === sessionId) {
      localStorage.removeItem("ambientRunSession");
      justCreatedRef.current = true;
    }

    let cancelled = false;
    (async () => {
      const [status, msgs] = await Promise.all([
        api.getSessionStatus(sessionId).catch(() => null),
        api.getSessionMessages(sessionId).catch(() => null),
      ]);

      if (cancelled) return;

      const hasHistory = msgs && msgs.length > 0;
      const isActive = status?.active;

      if (justCreatedRef.current) {
        justCreatedRef.current = false;
      } else if (!isActive && !hasHistory) {
        // Only discard the session when both API calls returned a definitive
        // response.  A null value means the call threw (backend unreachable,
        // event-loop blocked, etc.) — NOT that the session is gone.  Clearing
        // currentSessionId on a transient failure would make handleSend create
        // a brand-new session instead of reusing the one in the URL.
        if (status !== null && msgs !== null) {
            setCurrentSessionId(null);
          setMessages([]);
          navigate("/chat", { replace: true });
        }
        return;
      }

      if (status?.running) {
        setIsStreaming(true);
        watchSession(sessionId);
      }

      if (hasHistory) {
        const raw: ChatMessage[] = msgs.map((m) => ({
          id: `msg-${++msgIdRef.current}`,
          type: (m.type as ChatMessage["type"]) ?? "agent",
          content: (m.content as string) ?? "",
          metadata: m.metadata as Record<string, unknown> | undefined,
          timestamp: new Date(),
          sessionId: sessionId,
        }));
        // Plain replace — disk history is authoritative (messages are persisted
        // before queuing). An additive merge here would blend messages from a
        // previous session that may still be in React state.
        setMessages(mergeToolMessages(raw));
      }
    })();

    return () => { cancelled = true; };
  }, [sessionId, navigate, watchSession]);

  const refreshSessionFiles = useCallback((sid: string) => {
    api.listSessionFiles(sid).then(setSessionFiles).catch(() => {});
  }, []);

  // Surfaces outside the chat (the Watch panel) write into the session
  // sandbox without an agent turn to trigger the usual reload.
  useEffect(
    () => subscribeSessionFiles((sid) => {
      if (sid === currentSessionId) refreshSessionFiles(sid);
    }),
    [currentSessionId, refreshSessionFiles],
  );

  const handleMessage = useCallback((msg: WSMessage) => {
    const sid = sessionIdRef.current ?? "";
    if (msg.type === "resync" && sid) {
      void api.getSessionMessages(sid).then((msgs) => {
        if (!msgs) return;
        setMessages(mergeToolMessages(msgs.map((m, index) => ({
          id: `resync-${index}`,
          type: (m.type as ChatMessage["type"]) ?? "agent",
          content: (m.content as string) ?? "",
          metadata: m.metadata as Record<string, unknown> | undefined,
          timestamp: new Date(),
          sessionId: sid,
        }))));
      }).catch((e) => setLastError({
        code: "resync_failed",
        message: e instanceof Error ? e.message : "Could not reload this chat",
      }));
      return;
    }
    if (msg.type === "done" || msg.type === "stopped" || msg.type === "hitl_request" || msg.type === "ask_user") {
      // The resume we were waiting on has ended or paused again.
      hitlResumeUntilRef.current = 0;
    } else if (isActivityFrame(msg)) {
      // Live output proves the agent is working, whatever a status poll that
      // raced the resume reported.
      setIsStreaming(true);
    }
    if (msg.type === "done") {
      setIsStreaming(false);
      setPendingContext([]);
      // Defensive cleanup: a normal turn always finalizes its streaming
      // placeholder via an "agent"/"tool_call" event before "done" arrives
      // (see session_manager._do_stream), but clear the flag here too in
      // case a turn ends without generating any text.
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (!last || last.type !== "agent" || !last.metadata?.streaming) return prev;
        const updated = [...prev];
        updated[updated.length - 1] = { ...last, metadata: { ...last.metadata, streaming: false } };
        return updated;
      });
      if (sid) { refreshSessionFiles(sid); unwatchSession(sid); notify("done", sid); }
      return;
    }
    if (msg.type === "stopped") {
      setIsStreaming(false);
      setPendingContext([]);
      if (sid) { refreshSessionFiles(sid); unwatchSession(sid); }
      setMessages((prev) =>
        prev.map((m) =>
          m.type === "tool_call"
            ? { ...m, type: "tool_result" as const, metadata: { ...m.metadata, stopped: true } }
            : m.type === "agent" && m.metadata?.streaming
              ? { ...m, metadata: { ...m.metadata, streaming: false } }
              : m,
        ),
      );
      return;
    }
    if (msg.type === "context_received") {
      // The backend confirmed receipt. Remove the first matching entry from
      // pendingContext so the "queued" indicator can stay accurate, and
      // replace the optimistic local bubble (which has metadata.pending) with
      // the server-persisted version (no pending flag).
      setPendingContext((prev) => {
        const idx = prev.indexOf(msg.content);
        if (idx === -1) return prev;
        return [...prev.slice(0, idx), ...prev.slice(idx + 1)];
      });
      setMessages((prev) => {
        // Find the optimistic pending bubble and confirm it
        for (let i = prev.length - 1; i >= 0; i--) {
          const m = prev[i];
          if (m.type === "user" && m.metadata?.isContext && m.metadata?.pending && m.content === msg.content) {
            const updated = [...prev];
            updated[i] = { ...m, metadata: { ...m.metadata, pending: false } };
            return updated;
          }
        }
        // No optimistic bubble found — server echo arrives first; append it
        return [...prev, {
          id: `msg-${++msgIdRef.current}`,
          type: "user" as const,
          content: msg.content,
          metadata: { isContext: true },
          timestamp: new Date(),
          sessionId: sid || undefined,
        }];
      });
      return;
    }
    if (msg.type === "error" && msg.content === "Session not found") {
      setCurrentSessionId(null);
      setMessages([]);
      setIsStreaming(false);
      navigate("/chat", { replace: true });
      return;
    }
    if (msg.type === "error" && msg.metadata?.error_code) {
      setLastError({
        code: msg.metadata.error_code as string,
        message: msg.content,
      });
    }
    if (msg.type === "run_depth") {
      const mode = (msg.metadata?.run_depth as string) || "";
      const source = (msg.metadata?.run_depth_source as string) || "";
      const reason = (msg.metadata?.run_depth_reason as string) || "";
      const depthId = (msg.metadata?.run_depth_id as string) || "";
      setMessages((prev) => {
        if (depthId && prev.some((m) => m.metadata?.run_depth_id === depthId)) return prev;
        let next = prev;
        // Auto still marks the user bubble. A pinned mode is named by the line below.
        if (source === "agent" && (mode === "quick" || mode === "deep")) {
          for (let i = prev.length - 1; i >= 0; i--) {
            const m = prev[i];
            if (m.type !== "user" || m.metadata?.isContext) continue;
            if (m.sessionId && sid && m.sessionId !== sid) continue;
            next = [...prev];
            next[i] = {
              ...m,
              metadata: {
                ...m.metadata,
                run_depth: mode,
                run_depth_source: "agent",
                run_depth_reason: reason,
              },
            };
            break;
          }
        }
        return [...next, {
          id: `msg-${++msgIdRef.current}`,
          type: "run_depth" as const,
          content: msg.content,
          metadata: msg.metadata,
          timestamp: new Date(),
          sessionId: sid || undefined,
        }];
      });
      return;
    }
    if (msg.type === "memory_search") {
      setStreamPhase("memory_search");
      return;
    }
    if (msg.type === "memory_context") {
      pendingMemoryTopicsRef.current = (msg.metadata?.topics as string[]) ?? null;
      setStreamPhase("thinking");
      return;
    }
    if (msg.type === "agent_delta") {
      // Token-level chunk of the model's reply (see session_manager._do_stream).
      // Grow the last bubble in place if it's already an in-progress streaming
      // placeholder, otherwise start one. The authoritative "agent" event that
      // follows replaces this placeholder's content wholesale, so any minor
      // drift between concatenated deltas and the final text self-corrects.
      setStreamPhase("thinking");
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (last && last.type === "agent" && last.metadata?.streaming) {
          const updated = [...prev];
          updated[updated.length - 1] = { ...last, content: last.content + msg.content };
          return updated;
        }
        return [...prev, {
          id: `msg-${++msgIdRef.current}`,
          type: "agent" as const,
          content: msg.content,
          metadata: { streaming: true },
          timestamp: new Date(),
          sessionId: sid || undefined,
        }];
      });
      return;
    }
    if (msg.type === "hitl_request" || msg.type === "ask_user") {
      setIsStreaming(false);
      setPollForInterrupt(false);
      // Approval cards must be visible even if the user had scrolled up
      // while the agent was working.
      isPinnedToBottomRef.current = true;
      if (sid) { unwatchSession(sid); notify("hitl", sid); }
      const chatMsg: ChatMessage = { id: `msg-${++msgIdRef.current}`, type: msg.type, content: msg.content, metadata: msg.metadata, timestamp: new Date(), sessionId: sid || undefined };
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        let base = prev;
        if (last && last.type === "agent" && last.metadata?.streaming) {
          const updated = [...prev];
          updated[updated.length - 1] = { ...last, metadata: { ...last.metadata, streaming: false } };
          base = updated;
        }
        // Dedupe re-emitted interrupts: the backend re-pushes pending
        // interrupts on every WS (re)connect so dropped deliveries
        // recover, but if the client already received it we'd otherwise
        // render two identical approval cards.  Compare against the most
        // recent unresolved interrupt in state.
        for (let i = base.length - 1; i >= 0; i--) {
          const m = base[i];
          if (m.type !== "hitl_request" && m.type !== "ask_user") continue;
          if (m.metadata?.resolved) break;
          if (
            m.type === msg.type
            && m.content === msg.content
            && JSON.stringify(m.metadata) === JSON.stringify(msg.metadata)
          ) {
            return base;
          }
          break;
        }
        return [...base, chatMsg];
      });
      return;
    }
    if (msg.type === "execute_output") {
      const tcId = msg.metadata?.tool_call_id as string | undefined;
      const line = msg.content;
      setMessages((prev) => {
        for (let i = prev.length - 1; i >= 0; i--) {
          const m = prev[i];
          if (m.type !== "tool_call" || m.content !== "execute") continue;
          const callTcId = m.metadata?.tool_call_id as string | undefined;
          if (tcId && callTcId && callTcId !== tcId) continue;
          const updated = [...prev];
          const existing = (m.metadata?.liveOutput as string[]) ?? [];
          // Keep only the last 100 lines to avoid unbounded memory growth
          const next = existing.length >= 100
            ? [...existing.slice(-99), line]
            : [...existing, line];
          updated[i] = { ...m, metadata: { ...m.metadata, liveOutput: next } };
          return updated;
        }
        return prev;
      });
      return;
    }
    if (msg.type === "tool_result") {
      const resultName = (msg.metadata?.name as string) ?? "";
      const resultTcId = msg.metadata?.tool_call_id as string | undefined;
      const images = msg.metadata?.images as Array<{ base64: string; mime_type: string }> | undefined;
      setMessages((prev) => {
        for (let i = prev.length - 1; i >= 0; i--) {
          const m = prev[i];
          if (m.type !== "tool_call") continue;
          const callTcId = m.metadata?.tool_call_id as string | undefined;
          const matched = resultTcId && callTcId
            ? callTcId === resultTcId
            : m.content === resultName;
          if (matched) {
            const updated = [...prev];
            updated[i] = { ...updated[i], type: "tool_result", metadata: { ...updated[i].metadata, result: msg.content, ...(images ? { images } : {}) } };
            return updated;
          }
        }
        return [...prev, { id: `msg-${++msgIdRef.current}`, type: msg.type, content: msg.content, metadata: msg.metadata, timestamp: new Date(), sessionId: sid || undefined }];
      });
      return;
    }
    if (msg.type === "agent" || msg.type === "tool_call") {
      setStreamPhase("thinking");
    }
    let meta = msg.metadata;
    if (msg.type === "agent" && pendingMemoryTopicsRef.current) {
      meta = { ...meta, memory_topics: pendingMemoryTopicsRef.current };
      pendingMemoryTopicsRef.current = null;
    }
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      const hasStreamingPlaceholder = !!(last && last.type === "agent" && last.metadata?.streaming);
      if (hasStreamingPlaceholder && msg.type === "agent") {
        // Finalize the placeholder that "agent_delta" events grew, replacing
        // its content/metadata with the authoritative persisted values
        // instead of appending a second bubble.
        const updated = [...prev];
        updated[updated.length - 1] = { ...last, content: msg.content, metadata: meta };
        return updated;
      }
      let base = prev;
      if (hasStreamingPlaceholder && msg.type === "tool_call") {
        // Keep streamed preamble as a thought above the tool row.  Empty
        // placeholders (token stream produced no text) are still dropped.
        if (last.content.trim()) {
          const updated = [...prev];
          updated[updated.length - 1] = {
            ...last,
            metadata: { ...last.metadata, streaming: false },
          };
          base = updated;
        } else {
          base = prev.slice(0, -1);
        }
      } else if (hasStreamingPlaceholder) {
        // Some other event (e.g. an error) ended the turn without a proper
        // finalizer — stop treating the placeholder as in-progress.
        const updated = [...prev];
        updated[updated.length - 1] = { ...last, metadata: { ...last.metadata, streaming: false } };
        base = updated;
      }
      const chatMsg: ChatMessage = { id: `msg-${++msgIdRef.current}`, type: msg.type, content: msg.content, metadata: meta, timestamp: new Date(), sessionId: sid || undefined };
      return [...base, chatMsg];
    });
  }, [navigate, notify, unwatchSession, refreshSessionFiles, setLastError]);

  const { connected, send, sendEdit, sendHitlResponse, sendContext, waitForConnection } = useWebSocket({ sessionId: currentSessionId, onMessage: handleMessage });

  useEffect(() => {
    setWsConnected(connected);
  }, [connected, setWsConnected]);

  // Live commentary from the Watch panel arrives as context rather than as a
  // message, so narrating a screen never starts a turn on its own.
  useEffect(
    () => onAgentContext((text) => { sendContext(text); }),
    [sendContext],
  );

  const prevConnected = useRef(false);
  useEffect(() => {
    const reconnected = connected && !prevConnected.current && currentSessionId;
    prevConnected.current = connected;
    if (!reconnected) return;

    if (sendingRef.current) return;

    const sid = currentSessionId!;
    api.getSessionStatus(sid).then((status) => {
      // Bail when the user navigated to a different session while the
      // fetch was in flight.
      if (sid !== sessionIdRef.current) return;
      if (status.running) setIsStreaming(true);

      // Always reload messages on reconnect — even when the session has
      // finished — so any WS messages that arrived while the socket was
      // down (or were throttled by WKWebView) surface immediately without
      // requiring a manual Refresh.
      api.getSessionMessages(sid).then((msgs) => {
        if (sid !== sessionIdRef.current) return;
        if (!msgs || msgs.length === 0) {
          if (!status.running && !sendingRef.current) setIsStreaming(false);
          return;
        }
        const raw: ChatMessage[] = msgs.map((m) => ({
          id: `msg-${++msgIdRef.current}`,
          type: (m.type as ChatMessage["type"]) ?? "agent",
          content: (m.content as string) ?? "",
          metadata: m.metadata as Record<string, unknown> | undefined,
          timestamp: new Date(),
          sessionId: sid,
        }));
        const apiMerged = mergeToolMessages(raw);
        // The API fetch may have been issued before the hitl_request was
        // persisted on the backend (timing race: stream ends → interrupt
        // persisted → WS message sent).  If the WS message already updated
        // local state but the API response doesn't include the hitl_request
        // yet, preserve the local pending interrupt instead of wiping it.
        setMessages((prev) => {
          const withLocal = reconcileApiMessages(apiMerged, prev);
          const apiHasPendingHitl = pendingInterrupts(withLocal).length > 0;
          if (!apiHasPendingHitl) {
            const withHitl = appendMissingInterrupts(prev, apiMerged);
            if (withHitl) return withHitl;
            const localPending = pendingInterrupts(prev);
            if (localPending.length > 0) return [...withLocal, ...localPending];
          }
          return withLocal;
        });
        if (status.awaiting_input && pendingInterrupts(apiMerged).length === 0) {
          setPollForInterrupt(true);
          if (!sendingRef.current) setIsStreaming(false);
        } else if (pendingInterrupts(apiMerged).length > 0) {
          setPollForInterrupt(false);
          if (!sendingRef.current) setIsStreaming(false);
        } else if (!status.running && !sendingRef.current) {
          setIsStreaming(false);
        }
      }).catch(() => {
        if (!status.running && !sendingRef.current) setIsStreaming(false);
      });
    }).catch(() => {});
  }, [connected, currentSessionId]);

  // Polling safety-net: while isStreaming=true (or pollForInterrupt after the
  // agent task has paused for HITL), fetch messages + status from the API
  // every 2 s so that any WS message that was lost (hitl_request, agent
  // responses, done) will still appear within a couple of seconds.
  //
  // Design notes:
  //   • No sendingRef.current guard here — the effect re-runs whenever
  //     isStreaming flips to true (which happens both in handleSend and
  //     handleHitlDecision), so the interval always starts.
  //   • Messages are synced on every tick, not only when the agent stops,
  //     so in-progress tool calls and text responses also appear promptly.
  //   • We reuse existing message IDs by position to avoid remounting
  //     components (which would reset any typed credential input state).
  //   • The interval stops once the run is idle *and* no approval card is
  //     still expected.  ``status.running`` goes false as soon as the
  //     stream task exits, which is also when execute HITL pauses — so
  //     awaiting_input / an unmatched execute call keep polling until the
  //     approval card is in the transcript.
  const streamingPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  useEffect(() => {
    if ((!isStreaming && !pollForInterrupt) || !currentSessionId) {
      if (streamingPollRef.current) {
        clearInterval(streamingPollRef.current);
        streamingPollRef.current = null;
      }
      return;
    }
    const sid = currentSessionId;
    streamingPollRef.current = setInterval(async () => {
      try {
        const [status, msgs] = await Promise.all([
          api.getSessionStatus(sid).catch(() => null),
          api.getSessionMessages(sid).catch(() => null),
        ]);

        // Bail when the user navigated to a different session while the
        // fetch was in flight.
        if (sid !== sessionIdRef.current) return;

        let apiMerged: ChatMessage[] = [];
        if (msgs && msgs.length > 0) {
          // Use session-scoped positional IDs so the same backend message
          // always gets the same React key even across multiple poll cycles.
          const raw: ChatMessage[] = msgs.map((m, idx) => ({
            id: `${sid}-${idx}`,
            type: (m.type as ChatMessage["type"]) ?? "agent",
            content: (m.content as string) ?? "",
            metadata: m.metadata as Record<string, unknown> | undefined,
            timestamp: new Date(),
            sessionId: sid,
          }));
          apiMerged = mergeToolMessages(raw);
          setMessages((prev) => {
            // Keep local state when it has more messages than the API (in-flight
            // WS messages not yet persisted).  When counts are equal, compare
            // the last message's content and type: if the API version differs
            // (e.g. a final agent response replaced a "thinking" chunk that
            // arrived via WS before it was persisted) apply the API state.
            // Reuse existing IDs by position so components don't remount.
            if (apiMerged.length < prev.length) {
              return appendMissingInterrupts(prev, apiMerged) ?? prev;
            }
            const merged = reconcileApiMessages(apiMerged, prev);
            if (merged.length === prev.length) {
              const lastApi = merged[merged.length - 1];
              const lastLocal = prev[prev.length - 1];
              if (
                lastApi?.content === lastLocal?.content &&
                lastApi?.type === lastLocal?.type &&
                JSON.stringify(lastApi?.metadata) === JSON.stringify(lastLocal?.metadata)
              ) return prev;
            }
            return merged.map((m, i) =>
              i < prev.length ? { ...m, id: prev[i].id } : m,
            );
          });
        }

        const apiHasHitl = pendingInterrupts(apiMerged).length > 0;
        // The user just answered a card; the resume is starting. Whatever this
        // poll saw is from before it began, so wait for the next one.
        if (Date.now() < hitlResumeUntilRef.current) return;
        const awaiting =
          (!!status?.awaiting_input && !status.running)
          || (!!status && !status.running && hasUnmatchedApprovalTool(apiMerged));
        if (apiHasHitl) {
          setPollForInterrupt(false);
          setIsStreaming(false);
          if (streamingPollRef.current) {
            clearInterval(streamingPollRef.current);
            streamingPollRef.current = null;
          }
          return;
        }
        if (awaiting) {
          setPollForInterrupt(true);
          setIsStreaming(false);
          return;
        }
        if (status && !status.running) {
          setPollForInterrupt(false);
          setIsStreaming(false);
          if (streamingPollRef.current) {
            clearInterval(streamingPollRef.current);
            streamingPollRef.current = null;
          }
        }
      } catch {
        // ignore transient errors — next tick will retry
      }
    }, 2000);
    return () => {
      if (streamingPollRef.current) {
        clearInterval(streamingPollRef.current);
        streamingPollRef.current = null;
      }
    };
  }, [isStreaming, pollForInterrupt, currentSessionId]);

  // Context-hint nudge: after streaming starts, show a timed reminder that
  // the user can type to steer the agent.  Cycle: 4 s delay → visible 5 s →
  // hidden 15 s → repeat.  Hides immediately when the user starts typing.
  // Dismissing stops the cycle for the current run; the next streaming run
  // resets the dismissed flag so the hint reappears.
  useEffect(() => {
    const clearTimer = () => {
      if (hintTimerRef.current) { clearTimeout(hintTimerRef.current); hintTimerRef.current = null; }
    };
    if (!isStreaming) { clearTimer(); setShowContextHint(false); return; }

    // New streaming run — reset dismissed flag so the hint can appear again.
    hintDismissedRef.current = false;

    const scheduleShow = (initialDelay: number) => {
      clearTimer();
      hintTimerRef.current = setTimeout(() => {
        // Only reveal if the user hasn't dismissed and the input is empty.
        if (!hintDismissedRef.current && !inputValueRef.current.trim()) {
          setShowContextHint(true);
        }
        hintTimerRef.current = setTimeout(() => {
          setShowContextHint(false);
          if (!hintDismissedRef.current) scheduleShow(15_000); // cycle again unless dismissed
        }, 5_000);
      }, initialDelay);
    };

    scheduleShow(4_000);
    return () => { clearTimer(); setShowContextHint(false); };
  }, [isStreaming]);

  // Hide the hint the moment the user starts typing.
  useEffect(() => { if (input.trim()) setShowContextHint(false); }, [input]);

  const handleHitlDecision = useCallback((messageId: string, decisions: Array<Record<string, unknown>>, requestId?: string) => {
    if (hitlSubmittedRef.current.has(messageId)) return;
    if (!sendHitlResponse(decisions, requestId)) {
      setMessages((prev) => [
        ...prev,
        { id: `msg-${++msgIdRef.current}`, type: "error" as const, content: "Not connected to server — please wait and try again", timestamp: new Date(), sessionId: currentSessionId ?? sessionId ?? undefined },
      ]);
      return;
    }
    hitlSubmittedRef.current.add(messageId);
    hitlResumeUntilRef.current = Date.now() + 5000;
    if (currentSessionId) { clearSession(currentSessionId); watchSession(currentSessionId); }
    setMessages((prev) => prev.map((m) =>
      m.id === messageId
        ? { ...m, metadata: { ...m.metadata, resolved: true, decisions } }
        : m,
    ));
    setIsStreaming(true);
    setPollForInterrupt(false);
  }, [sendHitlResponse, clearSession, watchSession, currentSessionId, sessionId]);

  // Per-session auto-approve: enabled by the "Approve all" action and
  // automatically reset whenever the user opens a different session, so
  // every new session starts by requiring approval again.
  const [sessionAutoApprove, setSessionAutoApprove] = useState(false);
  useEffect(() => {
    setSessionAutoApprove(false);
    hitlSubmittedRef.current = new Set();
    setPollForInterrupt(false);
  }, [currentSessionId]);

  const handleApproveAllSession = useCallback((messageId: string, decisions: Array<Record<string, unknown>>) => {
    handleHitlDecision(messageId, decisions);
    setSessionAutoApprove(true);
  }, [handleHitlDecision]);

  // Auto-approve non-high-risk HITL requests when either the global setting
  // or the per-session flag is enabled.
  const autoApproveCommands = !!(appSettings?.auto_approve_commands);
  const shouldAutoApprove = autoApproveCommands || sessionAutoApprove;
  useEffect(() => {
    if (!shouldAutoApprove) return;
    const unresolved = [...messages].reverse().find(
      (m) => m.type === "hitl_request" && !m.metadata?.resolved,
    );
    if (!unresolved) return;
    if (hitlSubmittedRef.current.has(unresolved.id)) return;
    const risk = unresolved.metadata?.risk as { no_auto_approve?: boolean } | undefined;
    if (risk?.no_auto_approve) return;
    const actions = (unresolved.metadata?.action_requests as Array<{ name: string; args: Record<string, unknown> }> | undefined) ?? [];
    const hasHighRisk = actions.some((a) => screenHighRiskCommand(a.args?.command).length > 0);
    if (hasHighRisk) return;
    const requestId = typeof unresolved.metadata?.request_id === "string"
      ? unresolved.metadata.request_id
      : undefined;
    handleHitlDecision(unresolved.id, actions.map(() => ({ type: "approve" })), requestId);
  }, [shouldAutoApprove, messages, handleHitlDecision]);

  const startSession = async (agentOverride?: string) => {
    let data: Awaited<ReturnType<typeof api.createSession>>;
    try {
      const distillFromChat = isDistillCatalogId(currentModel) ? currentModel : "";
      const distillFromSettings = appSettings?.llm?.provider === "mlx"
        && isDistillCatalogId(appSettings.llm.mlx?.hf_llm_model_id)
        ? (appSettings.llm.mlx.hf_llm_model_id || "")
        : "";
      // Turbo runs the fused copy via oMLX.  Passing otto-distill/… here
      // used to force in-process MLX and pin the GPU.
      const distillCatalogId = appSettings?.llm?.provider === "mlx"
        ? (distillFromChat || distillFromSettings)
        : "";
      data = await api.createSession({
        agent_name: (agentOverride ?? selectedAgent) || null,
        ...(distillCatalogId ? { distill_catalog_id: distillCatalogId } : {}),
      });
    } catch (err) {
      // Parse privacy-lock 403 so the chat renders the dedicated card
      // instead of a raw "API 403: …" string.
      const msg = err instanceof Error ? err.message : String(err);
      if (msg.includes("403")) {
        try {
          const json = JSON.parse(msg.replace(/^API \d+: /, ""));
          if (json.error === "privacy_lock") {
            const privacyErr = new Error("privacy_lock");
            (privacyErr as Error & { privacyLock: boolean; llmProvider: string }).privacyLock = true;
            (privacyErr as Error & { privacyLock: boolean; llmProvider: string }).llmProvider = json.llm_provider ?? "unknown";
            throw privacyErr;
          }
        } catch (parseErr) {
          if ((parseErr as Error & { privacyLock?: boolean }).privacyLock) throw parseErr;
        }
      }
      throw err;
    }
    const sid = data.id;
    justCreatedRef.current = true;
    // Lock the chip to the bound agent immediately; the metadata
    // useEffect will refetch but pre-populating avoids a one-frame
    // flicker where the X button briefly reappears between
    // ``setCurrentSessionId`` and the ``getSession`` round-trip.
    setSessionBoundAgent(data.agent_name ?? null);
    setCurrentSessionId(sid);
    navigate(`/chat/${sid}`, { replace: true });
    return sid;
  };

  const workspace: SessionWorkspace | null = sessionInfo?.workspace ?? null;
  workspaceRef.current = workspace;

  const resolveFileUrl = useCallback((path: string) => {
    if (!currentSessionId) return "";
    const stripped = path.replace(/^\//, "");
    if (workspace) {
      const prefix = workspace.virtual_path.replace(/^\//, "");
      if (stripped === prefix || stripped.startsWith(prefix + "/")) {
        const rel = stripped === prefix ? "" : stripped.slice(prefix.length + 1);
        return api.getWorkspaceFileUrl(currentSessionId, rel);
      }
    }
    return api.getSessionFileUrl(currentSessionId, stripped);
  }, [currentSessionId, workspace]);

  const bindWorkspace = useCallback(async (bind: (sid: string) => Promise<SessionInfo>, forgetPending?: string) => {
    setWorkspaceError(null);
    setMappingFolder(true);
    try {
      let sid = currentSessionId;
      if (!sid) {
        const agent = selectedAgent || "coding-agent";
        if (!selectedAgent) {
          setSelectedAgent("coding-agent");
          localStorage.setItem("chatSelectedAgent", "coding-agent");
        }
        sid = await startSession(agent);
        await waitForConnection();
      }
      const info = await bind(sid);
      setSessionInfo(info);
      setShowWorkspaceTree(true);
      if (forgetPending) setPendingFolders((prev) => prev.filter((p) => p !== forgetPending));
    } catch (e) {
      setWorkspaceError(workspaceErrorText(e, "Could not map folder"));
    } finally {
      setMappingFolder(false);
    }
  }, [currentSessionId, selectedAgent, startSession, waitForConnection]);

  const mapWorkspace = useCallback(
    (source: string) => bindWorkspace((sid) => api.setSessionWorkspace(sid, source, grantMode), source),
    [bindWorkspace, grantMode],
  );

  // A folder you already allowed: no second confirmation.
  const useGrant = useCallback(
    (grant: FolderGrant) => bindWorkspace((sid) => api.bindSessionWorkspace(sid, grant.id)),
    [bindWorkspace],
  );

  const beginGrant = useCallback(async (source: string) => {
    setWorkspaceError(null);
    try {
      const preview = await api.previewFolderGrant(source);
      setGrantMode(preview.existing_mode === "read" ? "read" : "write");
      setGrantPreview(preview);
    } catch (e) {
      setWorkspaceError(workspaceErrorText(e, "Could not check that folder"));
    }
  }, []);

  // A typed path (browser) or the native dialog (desktop app).
  const chooseFolder = useCallback(async (path?: string) => {
    const picked = path || await pickDirectory();
    if (picked) await beginGrant(picked);
  }, [beginGrant]);

  const revokeGrant = useCallback(async (grant: FolderGrant) => {
    if (!currentSessionId || workspaceRef.current?.grant_id !== grant.id) return;
    try {
      const info = await api.getSession(currentSessionId);
      setSessionInfo(info);
      if (!info.workspace) setOpenArtifact(null);
    } catch (e) {
      setWorkspaceError(workspaceErrorText(e, "Could not refresh this chat"));
    }
  }, [currentSessionId]);

  const unmapWorkspace = useCallback(async () => {
    if (!currentSessionId) return;
    try {
      const info = await api.clearSessionWorkspace(currentSessionId);
      setSessionInfo(info);
      setOpenArtifact(null);
    } catch (e) {
      setWorkspaceError(e instanceof Error ? e.message : "Could not unmap folder");
    }
  }, [currentSessionId]);

  const addFiles = (files: FileList | File[]) => {
    const blobs: File[] = [];
    const nativePaths: string[] = [];
    for (const f of Array.from(files)) {
      const native = (f as File & { path?: string }).path;
      if (typeof native === "string" && (native.startsWith("/") || /^[A-Za-z]:[\\/]/.test(native))) {
        nativePaths.push(native);
        continue;
      }
      // A pasted folder arrives as a 0-byte File. Uploading it is the "0 B"
      // chip; never queue those as attachments.
      if (f.size === 0) continue;
      blobs.push(f);
    }
    if (nativePaths.length) addDroppedPaths(nativePaths);
    if (blobs.length) setPendingFiles((prev) => [...prev, ...blobs]);
  };

  const removeFile = (index: number) => {
    setPendingFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const removeFolder = (index: number) => {
    setPendingFolders((prev) => prev.filter((_, i) => i !== index));
  };

  const removeFilePath = (index: number) => {
    setPendingFilePaths((prev) => prev.filter((_, i) => i !== index));
  };

  // Wire voice transcripts → auto-send
  useEffect(() => {
    if (!voice.transcript) return;
    // Transcription done — release any stale click-lock so next tap starts fresh
    micClickLockRef.current = false;
    setInput(voice.transcript);
    pendingVoiceTranscriptRef.current = voice.transcript;
    // Defer one tick so setInput flushes before handleSend reads `input`
    const t = setTimeout(() => {
      handleSendRef.current?.();
      pendingVoiceTranscriptRef.current = null;
    }, 30);
    return () => clearTimeout(t);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [voice.transcript]);

  // Keep the latest draft in a ref so the transcript hand-off can restore it.
  useEffect(() => { latestInputRef.current = input; });

  // Send captured-transcript text (and optional screenshot attachments) to the
  // agent without clobbering the user's in-progress draft or touching the input
  // box.  handleSend reads pendingVoiceTranscriptRef and pendingTranscriptImagesRef
  // synchronously before its first await, so it's safe to clear right after
  // invoking it.  Each call appends to the CURRENT session (or starts one), so
  // incremental auto-sends stay in one conversation.
  const sendTranscriptPayload = useCallback((payload: AskPayload) => {
    const trimmed = (payload.text ?? "").trim();
    const files = (payload.images ?? [])
      .map(dataUrlToFile)
      .filter((f): f is File => f !== null);
    if (!trimmed && files.length === 0) return;
    const draft = latestInputRef.current;
    pendingVoiceTranscriptRef.current = trimmed;
    pendingTranscriptImagesRef.current = files.length > 0 ? files : null;
    handleSendRef.current?.();
    pendingVoiceTranscriptRef.current = null;
    if (draft) setTimeout(() => setInput(draft), 0);
  }, []);

  // Bridge: transcription drawer ("Ask Otto" / auto-send) → chat.  Drains any
  // payload queued before this page mounted (the drawer navigates to /chat first).
  useEffect(() => onAskOtto(sendTranscriptPayload), [sendTranscriptPayload]);

  // Broadcast whether the agent is busy so the drawer can pause auto-send while
  // Otto is streaming or awaiting an answer to an ask_user / hitl interrupt.
  const agentBusy = useMemo(
    () =>
      isStreaming ||
      messages.some(
        (m) => (m.type === "hitl_request" || m.type === "ask_user") && !m.metadata?.resolved,
      ),
    [isStreaming, messages],
  );
  useEffect(() => {
    window.dispatchEvent(new CustomEvent("otto:agent-busy", { detail: agentBusy }));
  }, [agentBusy]);
  // On unmount (e.g. navigating away mid-stream) clear the busy flag so the
  // drawer doesn't get stuck refusing to auto-send.
  useEffect(() => () => {
    window.dispatchEvent(new CustomEvent("otto:agent-busy", { detail: false }));
  }, []);

  const handleSend = async () => {
    if (sendingRef.current) return;
    const text = (pendingVoiceTranscriptRef.current ?? input).trim();
    // Screenshot attachments handed off from the transcription drawer — read
    // and clear synchronously so they upload with this send only.
    const transcriptImages = pendingTranscriptImagesRef.current ?? [];
    pendingTranscriptImagesRef.current = null;
    if (
      !text &&
      transcriptImages.length === 0 &&
      pendingFiles.length === 0 &&
      pendingFilePaths.length === 0 &&
      pendingFolders.length === 0
    )
      return;

    if (currentSessionId) clearSession(currentSessionId);
    clearError();
    sendingRef.current = true;

    const filesToUpload = [...pendingFiles, ...transcriptImages];
    const filePathsToSend = [...pendingFilePaths];
    const foldersToSend = [...pendingFolders];
    setInput("");
    localStorage.removeItem("chatDraft");
    localStorage.removeItem("chatSelectedAgent");
    setStreamPhase("thinking");
    setIsStreaming(true);
    setPollForInterrupt(false);

    try {
      let sid = currentSessionId;
      if (!sid && sessionId) {
        // The URL still points at a valid session but React state lost it
        // (e.g. a racing effect cleared currentSessionId).  Recover from the
        // URL param instead of creating a duplicate session.
        sid = sessionId;
        setCurrentSessionId(sid);
      }
      if (!sid) {
        sid = await startSession();
        await waitForConnection();
        // ``startSession`` calls ``navigate(/chat/${sid})`` which causes
        // the render-time sync block to fire (sessionId went undefined →
        // sid) and reset ``isStreaming`` to false.  Re-assert it now so
        // the "Thinking…" indicator stays visible until the first agent
        // response or the ``done`` event arrives.
        setIsStreaming(true);
      }

      let uploadedPaths: string[] = [];
      if (filesToUpload.length > 0) {
        setUploading(true);
        try {
          const results = await Promise.all(
            filesToUpload.map((f) => api.uploadSessionFile(sid!, `uploads/${f.name}`, f)),
          );
          uploadedPaths = results.map((r) => `/uploads/${r.path.split("/").pop()}`);
          setPendingFiles([]);
        } catch (e) {
          console.warn("File upload failed:", e);
          // Restore the draft (attachments stay pending) so the user can
          // retry, then surface the failure instead of silently sending the
          // message without its files.
          setInput(text);
          throw new Error(
            `File upload failed (${e instanceof Error ? e.message : "unknown error"}) — your message was not sent. Please try again.`,
          );
        } finally {
          setUploading(false);
        }
      }

      // Attach dropped paths inside the session so the agent can reach them
      // via the virtual-path tools. A file is copied into uploads/; a folder
      // is symlinked under links/. Absolute host paths are not sent: the
      // file tools are scoped to the session dir and would mangle them.
      let linkedFiles: string[] = [];
      let linkedFolders: string[] = [];
      const dropped = [...filePathsToSend, ...foldersToSend];
      if (dropped.length > 0) {
        try {
          const linkResults = await Promise.all(
            dropped.map((source) => api.createSessionLink(sid!, source)),
          );
          linkResults.forEach((r) => {
            if (r.is_dir) linkedFolders.push(r.path);
            else linkedFiles.push(r.path);
          });
          setPendingFilePaths([]);
          setPendingFolders([]);
        } catch (e) {
          console.warn("Symlink creation failed:", e);
          setInput(text);
          throw new Error(
            `Could not attach the dropped files/folders (${e instanceof Error ? e.message : "unknown error"}) — your message was not sent. Please try again.`,
          );
        }
      }

      const parts: string[] = [];
      const allFiles = [...uploadedPaths, ...linkedFiles];
      if (allFiles.length > 0) parts.push(`[Uploaded files: ${allFiles.join(", ")}]`);
      if (linkedFolders.length > 0) parts.push(`[Context folders: ${linkedFolders.join(", ")}]`);
      const fullText = parts.length > 0 ? `${parts.join("\n")}\n\n${text}` : text;

      const userMsg: ChatMessage = { id: `msg-${++msgIdRef.current}`, type: "user", content: fullText, timestamp: new Date(), sessionId: sid ?? undefined };
      // Sending a message should always jump back to the bottom, even if
      // the user had scrolled up to reread earlier context.
      isPinnedToBottomRef.current = true;
      setMessages((prev) => [...prev, userMsg]);

      if (!send(fullText, runDepth)) {
        throw new Error("Not connected to server — please wait and try again");
      }
      watchSession(sid!);
    } catch (err) {
      setIsStreaming(false);
      const isPrivacyLock = (err as Error & { privacyLock?: boolean })?.privacyLock === true;
      const provider = (err as Error & { llmProvider?: string })?.llmProvider;
      const errMsg: ChatMessage = {
        id: `msg-${++msgIdRef.current}`,
        type: "error",
        content: isPrivacyLock
          ? `Privacy Lock is engaged. Provider '${provider ?? "cloud"}' is blocked.`
          : (err instanceof Error ? err.message : "Failed to start session"),
        metadata: isPrivacyLock
          ? { error_code: "privacy_lock", llm_provider: provider ?? "cloud" }
          : undefined,
        timestamp: new Date(),
        sessionId: currentSessionId ?? sessionId ?? undefined,
      };
      setMessages((prev) => [...prev, errMsg]);
    } finally {
      sendingRef.current = false;
    }
  };

  // Keep ref current so the voice transcript effect can call it
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { handleSendRef.current = handleSend; });

  const handleStop = async () => {
    if (currentSessionId) {
      await api.stopSession(currentSessionId);
      setIsStreaming(false);
      setPendingContext([]);
    }
  };

  const handleAddContext = useCallback(() => {
    const text = input.trim();
    if (!text || !currentSessionId) return;

    // Optimistic bubble — marked pending until backend echoes context_received
    const ctxMsg: ChatMessage = {
      id: `msg-${++msgIdRef.current}`,
      type: "user",
      content: text,
      metadata: { isContext: true, pending: true },
      timestamp: new Date(),
      sessionId: currentSessionId,
    };
    setMessages((prev) => [...prev, ctxMsg]);
    setPendingContext((prev) => [...prev, text]);
    setInput("");
    localStorage.removeItem("chatDraft");

    sendContext(text);
  }, [input, currentSessionId, sendContext]);

  const handleRunDepthChange = useCallback((depth: RunDepth) => {
    runDepthChosen.current = true;
    setRunDepth(depth);
    try { localStorage.setItem("otto.runDepth", depth); } catch { /* ignore */ }
  }, []);

  const handleEditMessage = useCallback((messageIndex: number, newContent: string) => {
    if (sendingRef.current || isStreaming) return;
    sendingRef.current = true;

    let userMsgIndex = 0;
    for (let i = 0; i < messages.length; i++) {
      if (i === messageIndex) break;
      if (messages[i].type === "user") userMsgIndex++;
    }

    const edited: ChatMessage = {
      id: `msg-${++msgIdRef.current}`,
      type: "user",
      content: newContent,
      timestamp: new Date(),
      sessionId: currentSessionId ?? sessionId ?? undefined,
    };
    // Functional updater so concurrent WS/poll updates between render and
    // edit aren't silently dropped.
    setMessages((prev) => prev.slice(0, messageIndex).concat(edited));

    if (!sendEdit(userMsgIndex, newContent, runDepth)) {
      setMessages((prev) => [
        ...prev,
        { id: `msg-${++msgIdRef.current}`, type: "error" as const, content: "Not connected to server — please wait and try again", timestamp: new Date(), sessionId: currentSessionId ?? sessionId ?? undefined },
      ]);
      sendingRef.current = false;
      return;
    }
    setStreamPhase("thinking");
    setIsStreaming(true);
    if (currentSessionId) watchSession(currentSessionId);
    sendingRef.current = false;
  }, [messages, isStreaming, sendEdit, watchSession, currentSessionId, sessionId, runDepth]);

  // The slash menu only lists named subagents.  The orchestrator (no
  // subagent) is the default already — users get there by simply not
  // picking, by pressing Escape, or by clicking X on the chip — so an
  // explicit "General Purpose" entry just adds a misleading row.
  const slashItems = agents.filter((a) =>
    !slashFilter || a.name.toLowerCase().includes(slashFilter.toLowerCase()),
  );
  const clampedIndex = Math.max(0, Math.min(slashIndex, slashItems.length - 1));

  useEffect(() => {
    if (slashOpen && slashRef.current) {
      const active = slashRef.current.querySelectorAll("button")[clampedIndex];
      active?.scrollIntoView({ block: "nearest" });
    }
  }, [slashOpen, clampedIndex]);

  const handleSlashSelect = (agentName: string) => {
    setSelectedAgent(agentName);
    localStorage.setItem("chatSelectedAgent", agentName);
    setInput("");
    localStorage.setItem("chatDraft", "");
    setSlashOpen(false);
    setSlashFilter("");
    setSlashIndex(0);
    inputRef.current?.focus();
  };

  const draftSaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleInputChange = (value: string) => {
    setInput(value);
    if (draftSaveTimer.current) clearTimeout(draftSaveTimer.current);
    draftSaveTimer.current = setTimeout(() => { localStorage.setItem("chatDraft", value); }, 300);
    if (sessionMessages.length === 0 && !isStreaming) {
      const match = value.match(/^\/(\S*)$/);
      if (match) {
        setSlashOpen(true);
        setSlashFilter(match[1]);
        setSlashIndex(0);
        return;
      }
    }
    setSlashOpen(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (slashOpen) {
      /* Follow standard menu convention: ArrowDown moves toward the
         bottom of the visible list, ArrowUp toward the top — same as
         every other dropdown the user encounters. The menu being
         anchored above the composer doesn't change list order, only
         vertical position. */
      if (e.key === "ArrowDown") { e.preventDefault(); setSlashIndex((i) => Math.min(i + 1, slashItems.length - 1)); return; }
      if (e.key === "ArrowUp") { e.preventDefault(); setSlashIndex((i) => Math.max(i - 1, 0)); return; }
      if (e.key === "Enter" || e.key === "Tab") {
        e.preventDefault();
        const pick = slashItems[clampedIndex];
        if (pick) handleSlashSelect(pick.name);
        else setSlashOpen(false);
        return;
      }
      if (e.key === "Escape") { e.preventDefault(); setSlashOpen(false); return; }
    }
    if (atOpen) {
      if (e.key === "ArrowDown") { e.preventDefault(); setAtIndex((i) => Math.min(i + 1, atItems.length - 1)); return; }
      if (e.key === "ArrowUp") { e.preventDefault(); setAtIndex((i) => Math.max(i - 1, 0)); return; }
      if (e.key === "Enter" || e.key === "Tab") {
        e.preventDefault();
        const pick = atItems[clampedAtIndex];
        if (pick) handleAtSelect(pick.path);
        else setAtOpen(false);
        return;
      }
      if (e.key === "Escape") { e.preventDefault(); setAtOpen(false); return; }
    }
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (isStreaming && input.trim()) {
        handleAddContext();
      } else if (!isStreaming) {
        handleSend();
      }
    }
  };

  // Only display messages that belong to the session currently in focus.
  // Messages tagged with a different sessionId are silently dropped so that
  // late-arriving WS frames or stale React state from a prior session can
  // never bleed into the current chat view.  Gate on the URL ``sessionId``
  // (or ``currentSessionId`` when there's no URL — i.e. a brand-new session
  // being authored at /chat) rather than purely on ``currentSessionId``
  // state, so the filter never lags the URL by one render during a swap.
  const focusSessionId = sessionId ?? currentSessionId;
  const sessionMessages = useMemo(
    () => messages.filter((m) => !m.sessionId || m.sessionId === focusSessionId),
    [messages, focusSessionId],
  );

  useEffect(() => {
    const last = [...sessionMessages].reverse().find((m) => {
      if (m.type !== "tool_call" && m.type !== "tool_result") return false;
      return m.content === "read_file" || m.content === "edit_file" || m.content === "edit" || m.content === "write_file";
    });
    if (!last || last.id === lastFollowedToolRef.current) return;
    lastFollowedToolRef.current = last.id;
    if (!getFollowFileEdits()) return;
    const args = last.metadata?.args as Record<string, unknown> | undefined;
    const path = filePathFromToolArgs(args);
    if (!path) return;
    const type = artifactTypeFromPath(path) ?? "code";
    const isEdit = last.content === "edit_file" || last.content === "edit";
    const diff = isEdit && typeof args?.old_string === "string"
      ? { oldText: args.old_string, newText: String(args.new_string ?? "") }
      : undefined;
    setOpenArtifact((current) => {
      const same = current != null && virtualPathsMatch(current.path, path);
      return {
        path,
        fileUrl: resolveFileUrl(path),
        type,
        diff,
        // A tool_call arrives before the write. Keep the current revision so
        // the preview doesn't reload the pre-edit bytes; the sync effect
        // below bumps it once the result (or a new mtime) lands.
        revision: same ? current.revision : undefined,
      };
    });
  }, [sessionMessages, resolveFileUrl]);

  // Reload the side preview when the file it is showing is written again.
  useEffect(() => {
    if (!openArtifact) {
      previewSyncRef.current = null;
      return;
    }
    const key = normalizeVirtualPath(openArtifact.path);
    const writeId = latestWriteResultId(openArtifact.path, sessionMessages);
    const stamp = fileListStamp(openArtifact.path, sessionFiles);
    const prev = previewSyncRef.current;
    if (!prev || prev.path !== key) {
      previewSyncRef.current = { path: key, writeId, stamp, awaitingStamp: false };
      return;
    }
    const writeChanged = writeId != null && writeId !== prev.writeId;
    if (writeChanged) {
      // Keep the previous mtime so the listing catch-up doesn't fetch twice.
      // The tool result means the write has already landed on disk.
      previewSyncRef.current = { path: key, writeId, stamp: prev.stamp, awaitingStamp: true };
      setOpenArtifact((current) => {
        if (!current || normalizeVirtualPath(current.path) !== key) return current;
        return { ...current, revision: (current.revision ?? 0) + 1 };
      });
      return;
    }
    const stampChanged = stamp != null && stamp !== prev.stamp;
    if (!stampChanged) return;
    if (prev.awaitingStamp) {
      previewSyncRef.current = { path: key, writeId, stamp, awaitingStamp: false };
      return;
    }
    previewSyncRef.current = { path: key, writeId, stamp, awaitingStamp: false };
    setOpenArtifact((current) => {
      if (!current || normalizeVirtualPath(current.path) !== key) return current;
      return { ...current, revision: (current.revision ?? 0) + 1 };
    });
  }, [openArtifact, sessionMessages, sessionFiles]);

  const memoryHits = useMemo(() => {
    const agent = sessionMessages.filter((m) => m.type === "agent" && !m.metadata?.subagent);
    const withMemory = agent.filter((m) => (m.metadata?.memory_topics as string[] | undefined)?.length);
    return { total: agent.length, hits: withMemory.length };
  }, [sessionMessages]);

  const thoughtFlags = useMemo(
    () => computeThoughtFlags(sessionMessages),
    [sessionMessages],
  );

  const latestTodoFlags = useMemo(
    () => computeLatestTodoFlags(sessionMessages),
    [sessionMessages],
  );

  const visibleSessionFiles = useMemo(
    () => sessionFiles.filter(f => !f.path.startsWith("large_tool_results/")),
    [sessionFiles],
  );

  // "@mention" candidates: attachments the user just dropped/pasted but
  // hasn't sent yet (only a basename is known — they aren't linked into the
  // session dir until `handleSend` runs) listed ahead of files already in
  // this run's own output directory (flattened from the session file tree,
  // folders-before-files, same order `SessionFileTree` uses).
  const atPendingEntries = useMemo(() => {
    const basename = (p: string) => p.replace(/\\/g, "/").split("/").pop() ?? p;
    return [
      ...pendingFiles.map((f) => ({ path: f.name, isDir: false, pending: true as const, group: "pending" as const })),
      ...pendingFilePaths.map((p) => ({ path: basename(p), isDir: false, pending: true as const, group: "pending" as const })),
      ...pendingFolders.map((p) => ({ path: basename(p), isDir: true, pending: true as const, group: "pending" as const })),
    ];
  }, [pendingFiles, pendingFilePaths, pendingFolders]);

  const atSessionEntries = useMemo(() => {
    const out: { path: string; isDir: boolean; pending: false; group: "run" }[] = [];
    const walk = (nodes: ReturnType<typeof buildFileTree>) => {
      for (const node of nodes) {
        out.push({ path: node.path, isDir: !node.file, pending: false, group: "run" as const });
        if (node.children.length > 0) walk(node.children);
      }
    };
    walk(buildFileTree(visibleSessionFiles));
    return out;
  }, [visibleSessionFiles]);

  const atEntries = useMemo(
    () => [...atPendingEntries, ...atWorkspaceEntries, ...atSessionEntries],
    [atPendingEntries, atWorkspaceEntries, atSessionEntries],
  );

  const atItems = useMemo(() => {
    const q = atFilter.trim().toLowerCase();
    const matches = q ? atEntries.filter((e) => e.path.toLowerCase().includes(q)) : atEntries;
    return matches.slice(0, 30);
  }, [atEntries, atFilter]);
  const clampedAtIndex = Math.max(0, Math.min(atIndex, atItems.length - 1));

  useEffect(() => {
    if (!workspace || !currentSessionId) {
      setAtWorkspaceEntries([]);
      return;
    }
    if (!atOpen) return;
    const prefix = atFilter.includes("/") ? atFilter.replace(/\/[^/]*$/, "") : "";
    let cancelled = false;
    api.getWorkspaceTree(currentSessionId, prefix)
      .then((r) => {
        if (cancelled) return;
        setAtWorkspaceEntries(r.entries.map((e) => ({
          path: e.path,
          isDir: e.is_dir,
          pending: false as const,
          group: "workspace" as const,
        })));
      })
      .catch(() => { if (!cancelled) setAtWorkspaceEntries([]); });
    return () => { cancelled = true; };
  }, [atOpen, atFilter, workspace, currentSessionId]);

  useEffect(() => {
    if (atOpen && atRef.current) {
      const active = atRef.current.querySelectorAll("button")[clampedAtIndex];
      active?.scrollIntoView({ block: "nearest" });
    }
  }, [atOpen, clampedAtIndex]);

  const handleAtSelect = (path: string) => {
    inputRef.current?.replaceMention(`@${path} `);
    setAtOpen(false);
    setAtFilter("");
    setAtIndex(0);
  };

  const getViewType = artifactTypeFromPath;

  const viewableSessionFiles = useMemo(
    () => visibleSessionFiles.filter(f => getViewType(f.path) !== null),
    [visibleSessionFiles],
  );

  const resolveSubagentModel = useMemo(() => {
    const agentByName = new Map(agents.map((a) => [a.name, a] as const));
    const orchestrator = appSettings?.orchestrator;
    const llm = appSettings?.llm;
    const mainProvider = llm?.provider ?? "anthropic";
    const mainAnthropicModel = llm?.anthropic?.model_name ?? "";
    const mainOpenAIModel = llm?.openai?.model_name ?? "";
    const mainMlxModel = llm?.mlx?.hf_llm_model_id ?? "";
    const mainExoModel = appSettings?.exo?.model_name ?? "";

    const orchestratorChip = (): { label: string; family: string } => {
      const fam = orchestrator?.llm_family ?? "follow_main";
      if (fam === "frontier") {
        // "frontier" family always means Anthropic
        return { label: mainAnthropicModel || "frontier", family: "frontier" };
      }
      if (fam === "openai") {
        return { label: mainOpenAIModel || "openai", family: "openai" };
      }
      if (fam === "mlx") {
        return { label: orchestrator?.mlx_model || mainMlxModel || "Standard", family: "mlx" };
      }
      if (fam === "exo") {
        return { label: mainExoModel || "Cluster", family: "exo" };
      }
      const ovr = orchestrator?.provider_override?.trim();
      if (ovr) {
        if (ovr.toLowerCase() === "mlx") return { label: mainMlxModel || "Standard", family: "mlx" };
        if (ovr.toLowerCase() === "anthropic") return { label: mainAnthropicModel || "frontier", family: "frontier" };
        if (ovr.toLowerCase() === "openai") return { label: mainOpenAIModel || "openai", family: "openai" };
        if (ovr.toLowerCase() === "exo") return { label: mainExoModel || "Cluster", family: "exo" };
        return { label: ovr, family: "custom" };
      }
      if (mainProvider === "mlx") return { label: mainMlxModel || "Standard", family: "mlx" };
      if (mainProvider === "omlx") {
        const omlxModel = appSettings?.omlx?.model_name ?? "";
        return { label: omlxModel ? (omlxModel.split("/")[1] ?? omlxModel) : "omlx", family: "omlx" };
      }
      if (mainProvider === "exo") return { label: mainExoModel || "Cluster", family: "exo" };
      if (mainProvider === "openai") return { label: mainOpenAIModel || "openai", family: "openai" };
      return { label: mainAnthropicModel || "frontier", family: "frontier" };
    };

    return (rawName: string): { label: string; family: string } => {
      const name = rawName.replace(/ #\d+$/, "");
      const spec = agentByName.get(name);
      if (spec) {
        const fam = spec.subagent_llm_family ?? "inherit";
        if (fam === "frontier") {
          return { label: spec.model_override || mainAnthropicModel || "frontier", family: "frontier" };
        }
        if (fam === "mlx") {
          return { label: spec.mlx_model_id || mainMlxModel || "Standard", family: "mlx" };
        }
        if (fam === "exo") {
          return { label: mainExoModel || "Cluster", family: "exo" };
        }
        if (fam === "custom") {
          return { label: spec.model_override || "custom", family: "custom" };
        }
      }
      return orchestratorChip();
    };
  }, [agents, appSettings]);

  const handleRefresh = useCallback(() => {
    if (!currentSessionId) return;
    const sid = currentSessionId;
    navigate("/chat");
    setTimeout(() => navigate(`/chat/${sid}`), 30);
  }, [currentSessionId, navigate]);

  const renderItems = useMemo(
    () => buildRenderItems(sessionMessages, thoughtFlags),
    [sessionMessages, thoughtFlags],
  );

  // Errors are surfaced at the very end of the chat (after the latest agent
  // turn) rather than inline where they occurred, so they're never buried.
  const errorMessages = useMemo(
    () => sessionMessages.filter((m) => m.type === "error"),
    [sessionMessages],
  );

  const changedFiles = useMemo(
    () => collectChangedFiles(sessionMessages),
    [sessionMessages],
  );

  const codingSelected = (currentSessionId ? sessionBoundAgent : selectedAgent) === "coding-agent";
  const welcomePrompts = codingSelected
    ? ["Explain this repo", "Fix a failing test", "Add a feature"]
    : [
        "Write a Python script",
        "Summarise a document",
        "Research a topic online",
        "Debug this code",
        "Plan a project",
        "Draft an email",
      ];



  return (
    <div className="flex h-full min-h-0">
      <div className="flex flex-col flex-1 min-w-0 h-full">
      <header data-chat-region="header" className="border-b border-th-border px-6 py-3.5 flex flex-wrap items-center justify-between gap-x-3 gap-y-2 shrink-0 bg-th-bg-secondary">
        <div className="flex flex-wrap items-center gap-3 min-w-0">
          {currentSessionId && (
            <span className={`inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full font-medium ${connected ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/25" : "bg-amber-500/15 text-amber-400 border border-amber-500/25"}`}>
              <span className={`w-1.5 h-1.5 rounded-full ${connected ? "bg-emerald-400" : "bg-amber-400 animate-pulse"}`} />
              {connected ? "Live" : "Connecting"}
            </span>
          )}
          {parentSessionId && (
            <button
              onClick={() => navigate(`/chat/${parentSessionId}`)}
              className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full font-medium bg-blue-500/15 text-blue-400 border border-blue-500/25 hover:bg-blue-500/25 transition-colors"
              title="Open the parent session this was spawned from"
            >
              <ArrowUpLeft size={11} />
              From parent
            </button>
          )}
          <ProjectFolderMenu
            workspace={workspace}
            disabled={isStreaming}
            busy={mappingFolder}
            onUseGrant={useGrant}
            onAllowFolder={chooseFolder}
            onClear={unmapWorkspace}
            onRevoked={revokeGrant}
          />
          {sessionInfo?.mode && sessionInfo.mode !== "agent" && (
            <span
              className={`inline-flex items-center text-xs px-2.5 py-1 rounded-full font-medium border ${
                sessionInfo.mode === "plan"
                  ? "bg-violet-500/15 text-violet-300 border-violet-500/25"
                  : sessionInfo.mode === "ask"
                    ? "bg-sky-500/15 text-sky-300 border-sky-500/25"
                    : "bg-amber-500/15 text-amber-300 border-amber-500/25"
              }`}
              title="Set in the Code tab. Ask and Plan are read-only."
            >
              {sessionInfo.mode === "plan" ? "Plan mode" : sessionInfo.mode === "ask" ? "Ask mode" : "Debug mode"}
            </span>
          )}
        </div>
        <div className="flex flex-wrap items-center justify-end gap-3 ml-auto min-w-0">
          {!currentSessionId && !selectedAgent && (
            <span
              className="text-xs text-th-text-tertiary"
              title="Slash to pick a subagent before sending; otherwise the orchestrator will route the message."
            >
              Type <kbd className="px-1.5 py-0.5 bg-th-inset-bg border border-th-border rounded text-2xs font-mono text-th-text-muted">/</kbd> to pick a subagent
            </span>
          )}
          {renderItems.some(item => item.kind === "subagent-group") && (
            <button
              onClick={() => setGraphOpen(o => !o)}
              className={`flex items-center gap-1.5 text-xs transition-colors px-2.5 py-1.5 rounded-lg border ${
                graphOpen
                  ? "text-blue-400 bg-blue-500/10 border-blue-500/25 hover:bg-blue-500/15"
                  : "text-th-text-tertiary hover:text-th-text-primary hover:bg-th-surface-hover border-transparent hover:border-th-border"
              }`}
              title="Toggle agent graph"
            >
              <GitBranch size={13} />
              <span>Graph</span>
            </button>
          )}
          {currentSessionId && (
            <button
              onClick={() => navigate(`/code/${currentSessionId}`)}
              className="flex items-center gap-1.5 text-xs text-th-text-tertiary hover:text-th-text-primary transition-colors px-2.5 py-1.5 rounded-lg hover:bg-th-surface-hover border border-transparent hover:border-th-border"
              title="Open this session in the Code view"
            >
              <GitBranch size={13} />
              <span>Code</span>
            </button>
          )}
          {currentSessionId && (
            <button
              onClick={() => navigate(`/runs/${currentSessionId}`)}
              className="flex items-center gap-1.5 text-xs text-th-text-tertiary hover:text-th-text-primary transition-colors px-2.5 py-1.5 rounded-lg hover:bg-th-surface-hover border border-transparent hover:border-th-border"
              title="View run details"
            >
              <ArrowLeft size={13} />
              <span>Run <span className="font-mono text-th-text-muted/60">#{currentSessionId.slice(0, 8)}</span></span>
            </button>
          )}
          {currentSessionId && sessionMessages.length > 0 && (
            <button
              onClick={() => api.openSessionFilesFolder(currentSessionId).catch(() => {})}
              className="flex items-center gap-1.5 text-xs text-th-text-tertiary hover:text-th-text-primary transition-colors px-2.5 py-1.5 rounded-lg hover:bg-th-surface-hover border border-transparent hover:border-th-border"
              title="Open session folder"
            >
              <FolderOpen size={13} />
              <span>Folder</span>
            </button>
          )}
          {currentSessionId && sessionMessages.length > 0 && (
            <SessionStatsPanel messages={sessionMessages} info={sessionInfo} />
          )}
          {memoryHits.hits > 0 && (
            <span
              className="inline-flex items-center gap-1.5 text-2xs px-2 py-1 rounded-full bg-blue-500/10 text-blue-400/80 border border-blue-500/15"
              title={`Memory injected in ${memoryHits.hits} of ${memoryHits.total} responses`}
            >
              <Brain size={11} />
              {memoryHits.hits}/{memoryHits.total}
            </span>
          )}
        </div>
      </header>

      <div
        data-chat-region="history"
        ref={messagesContainerRef}
        onScroll={handleHistoryScroll}
        className="flex-1 overflow-y-auto px-8 py-8 space-y-7"
      >
        {sessionMessages.length === 0 && (
          <div data-chat-empty className="flex flex-col items-center justify-center h-full -mt-12">
            <img
              src={theme === "dark" ? logoDark : logoLight}
              alt="Otto"
              className="w-20 h-20 mb-5 select-none object-contain opacity-90"
              draggable={false}
            />
            <h3 className="text-xl font-semibold text-th-text-primary mb-2">How can I help?</h3>
            <p className="text-sm text-th-text-tertiary max-w-sm text-center leading-relaxed mb-7">
              {codingSelected
                ? "Choose a project folder, then describe the change. Otto will search, edit, and run commands in that tree."
                : "Your AI agent — browse, write, run code, and more."}
            </p>
            {codingSelected && !workspace && (
              <div className="mb-5">
                <ProjectFolderMenu
                  workspace={workspace}
                  disabled={isStreaming}
                  busy={mappingFolder}
                  onUseGrant={useGrant}
                  onAllowFolder={chooseFolder}
                  onClear={unmapWorkspace}
                  onRevoked={revokeGrant}
                />
              </div>
            )}
            <div className="flex flex-wrap gap-2 justify-center max-w-lg">
              {welcomePrompts.map((prompt) => (
                <button
                  key={prompt}
                  onClick={() => { setInput(prompt); inputRef.current?.focus(); }}
                  className="px-3.5 py-2 text-sm text-th-text-secondary bg-th-surface border border-th-border rounded-xl hover:bg-th-surface-hover hover:text-th-text-primary hover:border-th-border-strong transition-all duration-150"
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        )}
        <MessageList
          items={renderItems}
          thoughtFlags={thoughtFlags}
          latestTodoFlags={latestTodoFlags}
          isStreaming={isStreaming}
          currentSessionId={currentSessionId}
          resolveSubagentModel={resolveSubagentModel}
          onEditMessage={handleEditMessage}
          onHitlDecision={handleHitlDecision}
          onApproveAllSession={handleApproveAllSession}
          onOpenArtifact={handleOpenArtifact}
        />
        {isStreaming && (
          <ThinkingIndicator
            phase={streamPhase}
            pendingContext={pendingContext}
            withAvatar
          />
        )}
        {!isStreaming && changedFiles.length > 0 && currentSessionId && (
          <ChangedFilesStrip
            files={changedFiles}
            fileUrl={resolveFileUrl}
            onOpen={handleOpenArtifact}
          />
        )}
        {!isStreaming && viewableSessionFiles.length > 0 && (
          <div className="mt-4 ml-10 p-3 rounded-xl border border-th-border bg-th-card-bg">
            <p className="text-2xs uppercase tracking-wider font-semibold text-th-text-muted mb-2">
              Output files
            </p>
            <OutputFileGrid
              files={viewableSessionFiles}
              fileUrl={(p) => api.getSessionFileUrl(currentSessionId!, p)}
              onOpenArtifact={handleOpenArtifact}
            />
          </div>
        )}
        {errorMessages.length > 0 && (
          <div className="space-y-3">
            {errorMessages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} sessionId={currentSessionId ?? undefined} />
            ))}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {graphOpen && (
        <AgentGraph
          messages={sessionMessages}
          onClose={() => setGraphOpen(false)}
        />
      )}

      {currentSessionId && (
        <div className="flex justify-end px-6 py-1.5 shrink-0 border-t border-th-border bg-th-bg-secondary">
          <button
            onClick={handleRefresh}
            className="flex items-center gap-1.5 text-xs text-th-text-muted hover:text-th-text-primary transition-colors px-2 py-1 rounded-md hover:bg-th-surface-hover"
            title="Refresh — reload messages and files"
          >
            <RefreshCw size={12} />
            <span>Refresh</span>
          </button>
        </div>
      )}

      {workspace && currentSessionId && (
        <div className="border-t border-th-border px-6 py-2 shrink-0 bg-th-inset-bg/90">
          <div className="flex items-center gap-2 py-1">
            <button
              onClick={() => setShowWorkspaceTree(!showWorkspaceTree)}
              className="flex items-center gap-2 text-xs text-th-text-tertiary hover:text-th-text-primary transition-colors flex-1"
            >
              {showWorkspaceTree ? <ChevronUp size={12} /> : <ChevronRight size={12} />}
              <Folder size={13} className="text-sky-400" />
              <span className="font-medium text-th-text-primary truncate">{workspace.name}</span>
              <span className="text-th-text-muted truncate max-w-[240px]">{workspace.host_path}</span>
            </button>
          </div>
          {showWorkspaceTree && (
            <div className="mt-1.5 mb-1 max-h-48 overflow-y-auto">
              <WorkspaceTree
                sessionId={currentSessionId}
                fileUrl={(rel) => api.getWorkspaceFileUrl(currentSessionId, rel)}
                onOpenFile={(rel, url, type) => handleOpenArtifact(`${workspace.virtual_path}/${rel}`, url, type)}
              />
            </div>
          )}
        </div>
      )}

      {visibleSessionFiles.length > 0 && (
        <div className="border-t border-th-border px-6 py-2 shrink-0 bg-th-inset-bg/90">
          <div className="flex items-center gap-2 py-1">
            <button
              onClick={() => setShowFiles(!showFiles)}
              className="flex items-center gap-2 text-xs text-th-text-tertiary hover:text-th-text-primary transition-colors flex-1"
            >
              {showFiles ? <ChevronUp size={12} /> : <ChevronRight size={12} />}
              <FolderOpen size={13} className="text-th-text-muted" />
              <span className="font-medium text-th-text-primary">Files ({visibleSessionFiles.length})</span>
            </button>
            <button
              onClick={() => { if (currentSessionId) api.openSessionFilesFolder(currentSessionId).catch((e) => console.warn("Failed to open folder:", e)); }}
              className="flex items-center gap-1 text-2xs text-th-text-tertiary hover:text-th-text-primary transition-colors px-1.5 py-0.5 rounded hover:bg-th-surface-hover"
              title="Open in file manager"
            >
              <ExternalLink size={11} />
              <span>Open folder</span>
            </button>
          </div>
          {showFiles && (
            <div className="mt-1.5 mb-1 space-y-1 max-h-40 overflow-y-auto">
              <SessionFileTree
                files={visibleSessionFiles}
                fileUrl={(p) => api.getSessionFileUrl(currentSessionId!, p)}
                downloadedFile={downloadedFile}
                onOpenArtifact={handleOpenArtifact}
                onReveal={(p) =>
                  api.openSessionFilesFolder(currentSessionId!, p)
                    .catch((e) => console.warn("Failed to open folder:", e))
                }
                onDownload={(p) => {
                  setDownloadedFile(p);
                  setTimeout(() => setDownloadedFile((prev) => (prev === p ? null : prev)), 3000);
                }}
                onDelete={(p) =>
                  api.deleteSessionFile(currentSessionId!, p)
                    .then(() => setSessionFiles((prev) => prev.filter((x) => x.path !== p)))
                }
              />
            </div>
          )}
        </div>
      )}

      <div
        data-chat-region="composer"
        className={`px-6 pb-5 pt-3 shrink-0 bg-th-bg-secondary transition-colors ${isDragging ? "border-t-2 border-blue-500/60 bg-blue-500/5" : "border-t border-th-border"}`}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          className="hidden"
          onChange={(e) => { if (e.target.files) addFiles(e.target.files); e.target.value = ""; }}
        />
        {(pendingFiles.length > 0 || pendingFilePaths.length > 0 || pendingFolders.length > 0) && (
          <div className="flex flex-wrap gap-2 mb-3 max-w-4xl mx-auto">
            {pendingFiles.map((f, i) => (
              <div key={`file-${f.name}-${i}`} className="flex items-center gap-1.5 px-2.5 py-1.5 bg-th-inset-bg border border-th-border rounded-lg text-xs text-th-text-secondary">
                <FileText size={12} className="text-th-text-muted shrink-0" />
                <span className="truncate max-w-[150px] text-th-text-primary">{f.name}</span>
                <span className="text-th-text-muted">{formatFileSize(f.size)}</span>
                <button onClick={() => removeFile(i)} className="text-th-text-muted hover:text-th-text-primary transition-colors ml-0.5">
                  <X size={11} />
                </button>
              </div>
            ))}
            {pendingFilePaths.map((p, i) => {
              const basename = p.replace(/\\/g, "/").split("/").pop() ?? p;
              return (
                <div key={`filepath-${p}-${i}`} className="flex items-center gap-1.5 px-2.5 py-1.5 bg-th-inset-bg border border-th-border rounded-lg text-xs text-th-text-secondary" title={p}>
                  <FileText size={12} className="text-th-text-muted shrink-0" />
                  <span className="truncate max-w-[150px] text-th-text-primary">{basename}</span>
                  <button onClick={() => removeFilePath(i)} className="text-th-text-muted hover:text-th-text-primary transition-colors ml-0.5">
                    <X size={11} />
                  </button>
                </div>
              );
            })}
            {pendingFolders.map((name, i) => {
              const basename = name.replace(/\\/g, "/").split("/").pop() ?? name;
              return (
                <div key={`folder-${name}-${i}`} className="flex items-center gap-1.5 px-2.5 py-1.5 bg-blue-500/10 border border-blue-500/20 rounded-lg text-xs text-blue-400" title={name}>
                  <Folder size={12} className="shrink-0" />
                  <span className="truncate max-w-[150px] font-medium">{basename}</span>
                  <button
                    type="button"
                    onClick={() => { void beginGrant(name); }}
                    className="text-2xs px-1.5 py-0.5 rounded bg-sky-500/20 text-sky-300 hover:bg-sky-500/30"
                    title="Map as project workspace"
                  >
                    Map
                  </button>
                  <button onClick={() => removeFolder(i)} className="text-blue-400/60 hover:text-blue-300 transition-colors ml-0.5">
                    <X size={11} />
                  </button>
                </div>
              );
            })}
          </div>
        )}
        {workspaceError && (
          <p className="max-w-4xl mx-auto mb-2 text-xs text-red-400">{workspaceError}</p>
        )}
        {grantPreview && (
          <FolderGrantDialog
            preview={grantPreview}
            mode={grantMode}
            busy={mappingFolder}
            onMode={setGrantMode}
            onCancel={() => setGrantPreview(null)}
            onConfirm={() => {
              const path = grantPreview.realpath;
              setGrantPreview(null);
              void mapWorkspace(path);
            }}
          />
        )}
        {/* Context-injection nudge — slides in/out while agent is running */}
        {isStreaming && (
          <div
            className={`max-w-4xl mx-auto mb-2 transition-all duration-500 ease-out ${
              showContextHint
                ? "opacity-100 translate-y-0 pointer-events-auto"
                : "opacity-0 translate-y-2 pointer-events-none"
            }`}
          >
            <div className="flex items-center justify-between gap-3 px-3.5 py-2 rounded-xl border border-violet-500/25 bg-violet-500/10 backdrop-blur-sm">
              <div className="flex items-center gap-2.5">
                <MessageSquarePlus size={13} className="shrink-0 text-violet-400" />
                <span className="text-xs text-violet-300/90">
                  Type to steer the agent in real time — press{" "}
                  <kbd className="inline-flex items-center px-1.5 py-0.5 rounded-md bg-violet-500/20 border border-violet-400/30 font-mono text-2xs text-violet-300 leading-none">
                    ↵
                  </kbd>{" "}
                  to inject
                </span>
              </div>
              <button
                onClick={() => { hintDismissedRef.current = true; setShowContextHint(false); if (hintTimerRef.current) { clearTimeout(hintTimerRef.current); hintTimerRef.current = null; } }}
                className="shrink-0 text-violet-400/50 hover:text-violet-300 transition-colors"
                aria-label="Dismiss"
              >
                <X size={12} />
              </button>
            </div>
          </div>
        )}
        {isDragging && (
          <div className="flex items-center justify-center py-4 mb-3 border-2 border-dashed border-blue-500/40 rounded-xl max-w-4xl mx-auto">
            <span className="text-sm text-blue-400 font-medium">Drop files or folders here</span>
          </div>
        )}
        <div className="relative max-w-4xl mx-auto">
          {slashOpen && slashItems.length > 0 && (
            <div ref={slashRef} className="absolute bottom-full left-0 mb-2 w-72 max-h-64 overflow-y-auto bg-th-card-bg border border-th-border rounded-lg shadow-xl z-50">
              <div className="px-3 py-2 border-b border-th-border">
                <span className="text-2xs uppercase tracking-wider text-th-text-muted font-semibold">Select Agent</span>
              </div>
              {slashItems.map((item, i) => (
                <button
                  key={item.name}
                  onClick={() => handleSlashSelect(item.name)}
                  onMouseEnter={() => setSlashIndex(i)}
                  className={`w-full text-left px-3 py-2.5 text-sm transition-colors flex flex-col gap-0.5 ${
                    i === clampedIndex
                      ? "bg-th-tab-active-bg text-th-tab-active-fg"
                      : "text-th-text-tertiary hover:bg-th-surface-hover hover:text-th-text-primary"
                  }`}
                >
                  <span className="font-medium">{item.name}</span>
                  {item.description && <span className="text-2xs text-th-text-muted truncate">{item.description}</span>}
                </button>
              ))}
            </div>
          )}
          {atOpen && atItems.length > 0 && (
            <div ref={atRef} className="absolute bottom-full left-0 mb-2 w-80 max-h-64 overflow-y-auto bg-th-card-bg border border-th-border rounded-lg shadow-xl z-50">
              {atItems.map((item, i) => {
                // Section headers: once right before the first item of each
                // group (pending attachments always come first, see atEntries).
                const showHeader = i === 0 || item.group !== atItems[i - 1].group;
                return (
                  <div key={`${item.group}:${item.path}:${i}`}>
                    {showHeader && (
                      <div className="px-3 py-2 border-b border-th-border">
                        <span className="text-2xs uppercase tracking-wider text-th-text-muted font-semibold">
                          {item.group === "pending"
                            ? "Attached (not sent yet)"
                            : item.group === "workspace"
                              ? "Project"
                              : "Files in this run"}
                        </span>
                      </div>
                    )}
                    <button
                      onClick={() => handleAtSelect(item.path)}
                      onMouseEnter={() => setAtIndex(i)}
                      className={`w-full text-left px-3 py-2 text-sm transition-colors flex items-center gap-2 ${
                        i === clampedAtIndex
                          ? "bg-th-tab-active-bg text-th-tab-active-fg"
                          : "text-th-text-tertiary hover:bg-th-surface-hover hover:text-th-text-primary"
                      }`}
                    >
                      {item.isDir ? (
                        <Folder size={13} className="shrink-0 text-sky-400/70" />
                      ) : (
                        <FileText size={13} className="shrink-0 text-th-text-muted" />
                      )}
                      <span className="truncate font-mono text-xs flex-1">{item.path}</span>
                      {item.pending && (
                        <span className="shrink-0 text-2xs px-1 py-0.5 rounded bg-amber-500/10 text-amber-400/80">
                          not sent
                        </span>
                      )}
                    </button>
                  </div>
                );
              })}
            </div>
          )}
          <InlineUrlInput
            ref={inputRef}
            className={`w-full pl-10 pr-12 py-3 bg-th-input-bg border rounded-2xl text-th-text-primary focus:outline-none transition-all min-h-[46px] max-h-[200px] overflow-y-auto cursor-text text-sm leading-5 shadow-sm disabled:opacity-50 disabled:cursor-not-allowed ${
              isStreaming
                ? "border-violet-500/20 focus:border-violet-400/60 focus:ring-2 focus:ring-violet-300/20"
                : "border-th-input-border focus:border-blue-400/60 focus:ring-2 focus:ring-blue-300/20"
            }`}
            placeholder={
              mlxIsDownloading ? "Waiting for model to finish downloading…"
              : exoModelLoading || omlxModelLoading ? "Waiting for model to load…"
              : isStreaming
                ? "Steer the agent with context…"
                : (pendingFiles.length > 0 || pendingFilePaths.length > 0 || pendingFolders.length > 0)
                  ? "Add a message about your context..."
                  : `Message Otto…${messages.length === 0 ? " (/ for agents)" : ""}`
            }
            disabled={isModelBusy}
            value={input}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            onPasteFiles={(files) => { addFiles(files); return true; }}
            onPastePaths={(paths) => { addDroppedPaths(paths); return true; }}
            onPasteMaybeFolder={(files) => { void tryNativeClipboardPaste(files); }}
            onMentionChange={(query) => {
              if (query === null) { setAtOpen(false); return; }
              setAtOpen(true);
              setAtFilter(query);
              setAtIndex(0);
            }}
          />
          <button
            className={`absolute left-2.5 top-1/2 -translate-y-1/2 w-7 h-7 rounded-lg flex items-center justify-center text-th-text-muted hover:text-th-text-primary hover:bg-th-surface-hover transition-all duration-150 disabled:opacity-50 disabled:cursor-not-allowed`}
            onClick={() => fileInputRef.current?.click()}
            disabled={isModelBusy}
            title="Attach files"
          >
            <Plus size={16} strokeWidth={2} />
          </button>
          {voiceEnabled && (
            <button
              className={`absolute right-10 top-1/2 -translate-y-1/2 w-7 h-7 rounded-lg flex items-center justify-center transition-all duration-150 ${
                voice.state === "capturing" ? "text-red-400 animate-pulse"
                : voice.state === "transcribing" ? "text-yellow-400 animate-spin"
                : voice.connected ? "text-th-text-muted hover:text-th-text-primary hover:bg-th-surface-hover"
                : "text-th-text-faint opacity-40"
              }`}
              onMouseDown={(e) => {
                e.preventDefault();
                // Second click in click-lock mode → stop.
                // Don't check voice.state here — WebSocket state updates are
                // async and may not have arrived yet when the user clicks again.
                if (micClickLockRef.current) {
                  micClickLockRef.current = false;
                  micMouseDownTimeRef.current = null;
                  voice.pushToTalkStop();
                  return;
                }
                micMouseDownTimeRef.current = Date.now();
                voice.pushToTalkStart();
              }}
              onMouseUp={() => {
                if (micMouseDownTimeRef.current === null) return;
                if (micClickLockRef.current) return; // already in click-lock, ignore
                const held = Date.now() - micMouseDownTimeRef.current;
                micMouseDownTimeRef.current = null;
                if (held >= MIC_HOLD_THRESHOLD_MS) {
                  // PTT hold — release to stop
                  voice.pushToTalkStop();
                } else {
                  // Quick click — lock recording on until next click
                  micClickLockRef.current = true;
                }
              }}
              onMouseLeave={() => {
                // If holding (not in click-lock), release stops recording
                if (micMouseDownTimeRef.current !== null && !micClickLockRef.current) {
                  const held = Date.now() - micMouseDownTimeRef.current;
                  micMouseDownTimeRef.current = null;
                  if (held >= MIC_HOLD_THRESHOLD_MS) {
                    voice.pushToTalkStop();
                  } else {
                    // Dragged off quickly — treat as click-lock
                    micClickLockRef.current = true;
                  }
                }
              }}
              onTouchStart={(e) => {
                e.preventDefault();
                if (micClickLockRef.current) {
                  micClickLockRef.current = false;
                  micMouseDownTimeRef.current = null;
                  voice.pushToTalkStop();
                  return;
                }
                micMouseDownTimeRef.current = Date.now();
                voice.pushToTalkStart();
              }}
              onTouchEnd={() => {
                if (micMouseDownTimeRef.current === null) return;
                if (micClickLockRef.current) return;
                const held = Date.now() - micMouseDownTimeRef.current;
                micMouseDownTimeRef.current = null;
                if (held >= MIC_HOLD_THRESHOLD_MS) {
                  voice.pushToTalkStop();
                } else {
                  micClickLockRef.current = true;
                }
              }}
              title={
                voice.state === "capturing" && micClickLockRef.current ? "Click to stop"
                : voice.state === "capturing" ? "Release to stop"
                : voice.state === "transcribing" ? "Transcribing…"
                : "Hold or click to speak"
              }
            >
              {voice.state === "capturing" || voice.state === "transcribing"
                ? <MicOff size={14} />
                : <Mic size={14} />
              }
            </button>
          )}
          <button
            className={`absolute right-2.5 top-1/2 -translate-y-1/2 w-7 h-7 rounded-lg flex items-center justify-center transition-all duration-150 ${
              uploading || isModelBusy ? "text-th-text-muted"
              : isStreaming && input.trim() ? "text-violet-400 hover:bg-violet-500/15"
              : isStreaming ? "text-red-400 hover:bg-red-500/15"
              : (input.trim() || pendingFiles.length > 0 || pendingFilePaths.length > 0 || pendingFolders.length > 0) ? "text-th-text-primary hover:bg-th-surface-hover"
              : "text-th-text-faint"
            }`}
            onClick={isStreaming && input.trim() ? handleAddContext : isStreaming ? handleStop : handleSend}
            disabled={(!input.trim() && !isStreaming && pendingFiles.length === 0 && pendingFilePaths.length === 0 && pendingFolders.length === 0) || uploading || isModelBusy}
            title={
              mlxIsDownloading ? "Waiting for MLX model to download…"
              : exoModelLoading ? "Waiting for the cluster model to load…"
              : omlxModelLoading ? "Waiting for oMLX model to load…"
              : isStreaming && input.trim() ? "Add context to running agent (Enter)"
              : isStreaming ? "Stop agent"
              : undefined
            }
          >
            {uploading
              ? <Loader2 size={14} className="animate-spin" />
              : isStreaming && input.trim()
                ? <MessageSquarePlus size={14} />
                : isStreaming
                  ? <Square size={14} />
                  : <Send size={14} />
            }
          </button>
        </div>
        {mlxIsDownloading && mlxDownloadJob && (
          <MlxDownloadBanner job={mlxDownloadJob} />
        )}
        <div className="max-w-4xl mx-auto mt-1.5 flex items-center gap-2">
          <div className="flex items-center gap-2 flex-1 min-w-0">
            <ModelPicker value={currentModel} models={availableModels} onChange={handleModelChange} />
            <DepthPicker value={runDepth} onChange={handleRunDepthChange} />
            {exoModelLoading && (
              <span className="inline-flex items-center gap-1 text-2xs text-amber-400 shrink-0">
                <Loader2 size={11} className="animate-spin" />
                Loading…
              </span>
            )}
            {omlxModelLoading && (
              <span className="inline-flex items-center gap-1 text-2xs text-amber-400 shrink-0">
                <Loader2 size={11} className="animate-spin" />
                Loading…
              </span>
            )}
            {debugLlm && (
              <div className="flex items-center gap-1.5 text-2xs text-th-text-muted truncate">
                <span className="px-1.5 py-0.5 rounded bg-th-inset-bg text-th-text-tertiary border border-th-border">
                  {debugLlm.provider === "exo" ? "Cluster"
                    : debugLlm.provider === "omlx" ? "Turbo"
                    : debugLlm.provider === "mlx" ? "Standard"
                    : debugLlm.provider === "azure" ? "Azure OpenAI"
                    : debugLlm.provider === "openai" ? "OpenAI"
                    : debugLlm.provider === "anthropic" ? "Anthropic"
                    : debugLlm.provider}
                </span>
                {debugLlm.provider === "bedrock" && (
                  <>
                    <span className="text-th-text-faint">·</span>
                    <span className="text-th-text-tertiary">{debugLlm.region}</span>
                    <span className="text-th-text-faint">·</span>
                    <span className="text-th-text-tertiary">{debugLlm.authMode}</span>
                  </>
                )}
                {debugLlm.provider === "azure" && debugLlm.region && (
                  <>
                    <span className="text-th-text-faint">·</span>
                    <span className="text-th-text-tertiary truncate max-w-[180px]">{debugLlm.region}</span>
                  </>
                )}
                {!["mlx", "omlx", "exo"].includes(debugLlm.provider) && (
                  <>
                    <span className="text-th-text-faint">·</span>
                    <span className={debugLlm.hasKeys ? "text-emerald-400" : "text-red-400"}>
                      {debugLlm.hasKeys ? "✓ credentials" : "✗ no credentials"}
                    </span>
                  </>
                )}
              </div>
            )}
          </div>
          {(() => {
            // Only render a chip when an actual subagent is in play:
            //   pre-session: user picked one via slash
            //   in-session: backend bound one at create time
            // The orchestrator (default) needs no chip — the absence of
            // a chip is the signal that "/" is still actionable.
            const sessionLocked = !!currentSessionId && sessionBoundAgent !== undefined;
            const effectiveName = sessionLocked ? sessionBoundAgent : selectedAgent;
            if (!effectiveName) return null;

            const m = resolveSubagentModel(effectiveName);
            const tone = familyChipClasses(m.family);
            return (
              <div className="flex items-center gap-2 shrink-0">
                <span className="text-xs text-th-text-tertiary font-medium">Agent:</span>
                <span
                  className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full font-medium border bg-emerald-500/15 text-emerald-400 border-emerald-500/25"
                  title={
                    sessionLocked
                      ? "This session is locked to this agent until you start a new chat."
                      : `Next session will run with ${effectiveName}.`
                  }
                >
                  {effectiveName}
                  {!sessionLocked && (
                    <button
                      onClick={() => { setSelectedAgent(""); localStorage.removeItem("chatSelectedAgent"); }}
                      className="hover:text-th-text-primary transition-colors"
                      title="Reset to Orchestrator"
                    >
                      <X size={11} />
                    </button>
                  )}
                </span>
                {m.label && (
                  <span
                    className={`inline-flex items-center gap-1 text-2xs px-2 py-0.5 rounded-full border font-medium ${tone}`}
                    title={`Model: ${m.label}`}
                  >
                    <Cpu size={10} />
                    <span className="truncate max-w-[200px]">{m.label}</span>
                  </span>
                )}
              </div>
            );
          })()}
          {currentSessionId && sessionMessages.length > 0 && !isStreaming && (
            <button
              onClick={() => setShowScheduleDialog(true)}
              className="p-2 rounded-lg hover:bg-th-surface-hover text-th-text-muted hover:text-th-text-secondary transition-colors shrink-0"
              title="Schedule this agent run"
            >
              <Calendar size={15} />
            </button>
          )}
        </div>
      </div>

      {scheduleSuccess && (
        <div className="absolute top-16 left-1/2 -translate-x-1/2 z-50 flex items-center gap-3 px-4 py-3 bg-emerald-500/10 border border-emerald-500/20 rounded-lg shadow-lg">
          <CheckCircle2 size={14} className="text-emerald-400 shrink-0" />
          <span className="text-sm text-emerald-400">{scheduleSuccess}</span>
          <button onClick={() => navigate("/schedules")} className="text-xs text-emerald-400/80 hover:text-emerald-300 font-medium underline">View Schedules</button>
          <button onClick={() => setScheduleSuccess(null)} className="text-emerald-400/50 hover:text-emerald-400"><X size={14} /></button>
        </div>
      )}

      {showScheduleDialog && (
        <ChatScheduleDialog
          agentName={selectedAgent || agents.find(a => sessionMessages.some(m => m.metadata?.agent_name === a.name))?.name || null}
          initialPrompt={sessionMessages.find(m => m.type === "user")?.content ?? ""}
          onClose={() => setShowScheduleDialog(false)}
          onCreated={(scheduleId) => {
            setShowScheduleDialog(false);
            setScheduleSuccess(`Schedule "${scheduleId}" created`);
            setTimeout(() => setScheduleSuccess(null), 6000);
          }}
        />
      )}
    </div>

    {/* All viewable artifacts — side split panel */}
    {openArtifact && (
      // The cap keeps the chat column (and its header actions) usable when
      // the window is narrow; the panel gives up width instead of covering it.
      <div className="h-full shrink-0 flex" style={{ width: artifactWidth, maxWidth: "calc(100% - 360px)", minWidth: 240 }}>
        {/* Drag handle */}
        <div
          onMouseDown={onArtifactResizeStart}
          className="w-1 h-full cursor-col-resize shrink-0 hover:bg-blue-500/40 active:bg-blue-500/60 transition-colors"
          title="Drag to resize"
        />
        <div className="flex-1 min-w-0">
          <ArtifactPanel
            artifact={openArtifact}
            onClose={() => setOpenArtifact(null)}
          />
        </div>
      </div>
    )}
  </div>
  );
}


function ChatScheduleDialog({
  agentName,
  initialPrompt,
  onClose,
  onCreated,
}: {
  agentName: string | null;
  initialPrompt: string;
  onClose: () => void;
  onCreated: (scheduleId: string) => void;
}) {
  const [scheduleId, setScheduleId] = useState(() => {
    const base = agentName || "scheduled";
    return `${base}-${Date.now().toString(36)}`.slice(0, 30);
  });
  const [prompt, setPrompt] = useState(initialPrompt);
  const [cron, setCron] = useState("0 9 * * *");
  const [selectedPreset, setSelectedPreset] = useState("daily-9am");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  const handleSave = async () => {
    setSaving(true);
    setError("");
    try {
      await api.createSchedule({
        id: scheduleId,
        agent_name: agentName,
        prompt,
        cron_expression: cron,
      });
      onCreated(scheduleId);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to create schedule");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/30 backdrop-blur-sm flex items-center justify-center z-50 p-8">
      <div className="bg-th-card-bg border border-th-card-border rounded-2xl w-full max-w-lg p-6 shadow-2xl">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-lg font-bold text-th-text-primary">Schedule This Run</h2>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-th-surface-hover text-th-text-muted hover:text-th-text-primary transition-colors"><X size={20} /></button>
        </div>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-th-text-tertiary mb-2">Agent</label>
            <div className="px-4 py-2.5 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-secondary text-sm">
              {agentName || "Orchestrator"}
            </div>
          </div>
          <div>
            <label className="block text-sm font-medium text-th-text-tertiary mb-2">Schedule ID</label>
            <input
              className="w-full px-4 py-2.5 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-300/30 transition-all text-sm"
              value={scheduleId}
              onChange={(e) => setScheduleId(e.target.value.replace(/[^A-Za-z0-9 _-]/g, ""))}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-th-text-tertiary mb-2">Prompt</label>
            <textarea
              className="w-full px-4 py-3 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-300/30 transition-all min-h-[80px] text-sm"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-th-text-tertiary mb-2">Schedule</label>
            <div className="flex flex-wrap gap-2">
              {CRON_PRESETS.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => { setSelectedPreset(p.id); if (p.cron) setCron(p.cron); }}
                  className={`text-xs px-3 py-1.5 rounded-lg font-medium border transition-all duration-150 ${
                    selectedPreset === p.id
                      ? "bg-th-tab-active-bg text-th-tab-active-fg border-th-tab-active-bg"
                      : "bg-th-inset-bg text-th-text-tertiary border-th-border hover:border-blue-500/40 hover:text-th-text-primary"
                  }`}
                >
                  {p.label}
                </button>
              ))}
            </div>
            {selectedPreset === "custom" && (
              <input
                className="w-full mt-2 px-4 py-2.5 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-400 text-sm font-mono"
                value={cron}
                onChange={(e) => setCron(e.target.value)}
                placeholder="0 9 * * 1-5"
              />
            )}
            <p className="text-2xs text-th-text-muted mt-1.5">Cron: {cron}</p>
          </div>
          {error && (
            <div className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">{error}</div>
          )}
          <div className="flex justify-end gap-2 pt-3 border-t border-th-border">
            <button className="px-4 py-2 bg-th-inset-bg border border-th-border text-th-text-secondary hover:text-th-text-primary rounded-lg text-sm font-medium transition-colors" onClick={onClose}>Cancel</button>
            <button
              className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-semibold transition-colors hover:bg-blue-500 disabled:opacity-40"
              onClick={handleSave}
              disabled={!scheduleId || !prompt.trim() || !cron || saving}
            >
              {saving ? "Creating..." : "Create Schedule"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Inline MLX download progress banner shown above the toolbar when a model
// download is in progress for the currently-selected model.
// ---------------------------------------------------------------------------

function bytesToHuman(n: number): string {
  if (!n || n < 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : 1)} ${units[i]}`;
}

function MlxDownloadBanner({ job }: { job: MlxDownloadJob }) {
  const pct = job.bytes_total > 0
    ? Math.min(100, Math.round((job.bytes_done / job.bytes_total) * 100))
    : 0;
  const indeterminate = job.bytes_total === 0;

  const eta = job.eta_seconds != null && job.eta_seconds >= 0
    ? job.eta_seconds < 60
      ? `${job.eta_seconds}s`
      : `${Math.floor(job.eta_seconds / 60)}m ${job.eta_seconds % 60}s`
    : null;

  const rate = job.rate_bps > 0
    ? `${bytesToHuman(job.rate_bps)}/s`
    : null;

  return (
    <div className="max-w-4xl mx-auto mb-1.5 rounded-lg border border-th-border bg-th-surface px-3 py-2 space-y-1.5">
      <div className="flex items-center justify-between gap-2 text-2xs">
        <span className="inline-flex items-center gap-1.5 text-amber-400 font-medium">
          <Loader2 size={11} className="animate-spin" />
          Downloading model…
        </span>
        <span className="text-th-text-muted font-mono truncate max-w-[280px]" title={job.current_file}>
          {job.current_file
            ? job.current_file.split("/").pop()
            : job.repo_id}
        </span>
        <span className="text-th-text-muted font-mono shrink-0">
          {indeterminate
            ? `${bytesToHuman(job.bytes_done)} / ?`
            : `${bytesToHuman(job.bytes_done)} / ${bytesToHuman(job.bytes_total)}`}
          {!indeterminate && <span className="ml-1">· {pct}%</span>}
          {rate && <span className="ml-1.5">· {rate}</span>}
          {eta && <span className="ml-1.5">· {eta}</span>}
        </span>
      </div>
      <div className="relative h-1.5 rounded-full bg-th-inset-bg overflow-hidden">
        {indeterminate ? (
          <div className="absolute inset-y-0 w-1/3 bg-amber-500/50 animate-pulse" />
        ) : (
          <div
            className="absolute inset-y-0 left-0 bg-amber-500 transition-all duration-300"
            style={{ width: `${pct}%` }}
          />
        )}
      </div>
    </div>
  );
}
