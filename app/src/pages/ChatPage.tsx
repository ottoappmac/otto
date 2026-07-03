import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { useNavigate, useParams } from "react-router-dom";
import { Send, Square, ChevronUp, ChevronRight, FileText, Download, FolderOpen, ExternalLink, CheckCircle2, Plus, X, Loader2, Trash2, Calendar, Brain, Cpu, ArrowUpLeft, ArrowLeft, Folder, GitBranch, RefreshCw, MessageSquarePlus, Mic, MicOff, Image as ImageIcon, FileJson } from "lucide-react";
import { api } from "../hooks/useApi";
import { useWebSocket } from "../hooks/useWebSocket";
import { AgentGraph } from "../components/chat/AgentGraph";
import { MessageBubble } from "../components/chat/MessageBubble";
import { SubagentGroup } from "../components/chat/SubagentGroup";
import { ArtifactPanel, artifactTypeFromPath } from "../components/chat/ArtifactPanel";
import { ThinkingIndicator } from "../components/chat/ThinkingIndicator";
import type { Artifact, ArtifactType } from "../components/chat/ArtifactPanel";
import { ModelPicker } from "../components/chat/ModelPicker";
import SessionStatsPanel from "../components/chat/SessionStatsPanel";
import InlineUrlInput, { type InlineUrlInputHandle } from "../components/chat/InlineUrlInput";
import { formatFileSize } from "../utils/formatFileSize";
import { mergeToolMessages } from "../utils/mergeToolMessages";
import { familyChipClasses } from "../utils/subagentModelChip";
import { screenHighRiskCommand } from "../utils/highRiskCommands";
import { useNotification } from "../context/NotificationContext";
import { useConnection } from "../context/ConnectionContext";
import { useTheme } from "../context/ThemeContext";
import { usePolling } from "../hooks/usePolling";
import { CRON_PRESETS } from "../types";
import type { AgentSpec, AppSettings, ChatMessage, ExoCatalogModel, MlxDownloadJob, SessionInfo, WSMessage } from "../types";
import { useVoice } from "../hooks/useVoice";
import logoDark from "../assets/logo-dark.png";
import logoLight from "../assets/logo-light.png";

type RenderItem =
  | { kind: "message"; message: ChatMessage; index: number }
  | { kind: "subagent-group"; name: string; messages: ChatMessage[] };

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

export default function ChatPage() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();
  const { theme } = useTheme();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState(() => localStorage.getItem("chatDraft") ?? "");
  const [isStreaming, setIsStreaming] = useState(false);
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
    setOpenArtifact({ path, fileUrl, type });
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
  const [showFiles, setShowFiles] = useState(false);
  const [downloadedFile, setDownloadedFile] = useState<string | null>(null);
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [pendingFolders, setPendingFolders] = useState<string[]>([]);
  const [pendingFilePaths, setPendingFilePaths] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [currentModel, setCurrentModel] = useState<string>(() => localStorage.getItem("chatModel") ?? "");
  const [availableModels, setAvailableModels] = useState<{ id: string; name: string }[]>([]);
  const [modelsFetched, setModelsFetched] = useState(false);
  // Tracks the live EXO catalog (with downloaded/loaded flags) so that when
  // the user picks a model from the chat ModelPicker we can fire-and-forget
  // a preload for any model that's already on disk but not yet resident.
  const exoCatalogRef = useRef<ExoCatalogModel[]>([]);

  const exoAutoPreloadInFlightRef = useRef(false);
  const [exoModelLoading, setExoModelLoading] = useState(false);
  const [omlxModelLoading, setOmlxModelLoading] = useState(false);
  // Tracks an active MLX download job for the currently-selected model so
  // we can show progress and block sends while weights are still coming down.
  const [mlxDownloadJob, setMlxDownloadJob] = useState<MlxDownloadJob | null>(null);
  const [appSettings, setAppSettings] = useState<AppSettings | null>(null);
  const voiceEnabled = !!(appSettings?.voice?.enabled);
  const wakeEnabled = !!(appSettings?.voice?.wake_enabled);
  const pendingVoiceTranscriptRef = useRef<string | null>(null);
  const handleSendRef = useRef<(() => void) | null>(null);
  // Note: always-on wake listening is owned by the global listener in Layout,
  // so we don't autoStart here.  This connection receives the shared manager's
  // broadcasts (state, transcript) and drives the mic button + transcript send.
  const voice = useVoice({ enabled: voiceEnabled || wakeEnabled });
  // Mic button: distinguish quick-click (toggle) from hold (PTT)
  const micMouseDownTimeRef = useRef<number | null>(null);
  const micClickLockRef = useRef(false);
  const MIC_HOLD_THRESHOLD_MS = 300;
  const [debugLlm, setDebugLlm] = useState<{ provider: string; authMode: string; hasKeys: boolean; region: string } | null>(null);
  const [slashOpen, setSlashOpen] = useState(false);
  const [slashFilter, setSlashFilter] = useState("");
  const [slashIndex, setSlashIndex] = useState(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<InlineUrlInputHandle>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const slashRef = useRef<HTMLDivElement>(null);
  const msgIdRef = useRef(0);
  const pendingMemoryTopicsRef = useRef<string[] | null>(null);
  const sendingRef = useRef(false);
  const justCreatedRef = useRef(false);
  const sessionIdRef = useRef(currentSessionId);
  sessionIdRef.current = currentSessionId;

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
            if (apiMerged.length < prev.length) return prev;
            if (apiMerged.length === prev.length) {
              const lastApi = apiMerged[apiMerged.length - 1];
              const lastLocal = prev[prev.length - 1];
              if (
                lastApi?.content === lastLocal?.content &&
                lastApi?.type === lastLocal?.type
              ) return prev;
            }
            return preserveImages(apiMerged, prev).map((m, i) => i < prev.length ? { ...m, id: prev[i].id } : m);
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

  // Tauri intercepts OS-level file/folder drops before they reach the browser's
  // ondrop event. Use the Tauri window API to handle them properly.
  // The unlisten function is returned asynchronously, so we use a `cancelled`
  // flag to handle the case where cleanup runs before the promise resolves
  // (which happens every mount in React StrictMode dev, causing double registration).
  useEffect(() => {
    let cancelled = false;
    let unlistenFn: (() => void) | null = null;
    getCurrentWindow().onDragDropEvent((event) => {
      const payload = event.payload as { type: string; paths?: string[] };
      if (payload.type === "hover") {
        setIsDragging(true);
      } else if (payload.type === "leave" || payload.type === "cancel") {
        setIsDragging(false);
      } else if (payload.type === "drop" && payload.paths) {
        setIsDragging(false);
        const folders: string[] = [];
        const filePaths: string[] = [];
        payload.paths.forEach((p) => {
          // Heuristic: if the last path segment contains a ".", treat as file
          const basename = p.replace(/\\/g, "/").split("/").pop() ?? p;
          if (basename.includes(".")) {
            filePaths.push(p);
          } else {
            folders.push(p);
          }
        });
        if (filePaths.length) setPendingFilePaths((prev) => [...prev, ...filePaths]);
        if (folders.length) setPendingFolders((prev) => [...prev, ...folders]);
      }
    }).then((fn) => {
      if (cancelled) fn(); // cleanup already ran — unregister immediately
      else unlistenFn = fn;
    });
    return () => {
      cancelled = true;
      unlistenFn?.();
    };
  }, []);

  useEffect(() => { api.listAgents().then(setAgents).catch((e) => console.warn("Failed to load agents:", e)); }, []);

  // Poll for an active MLX download job matching the currently-selected model.
  const syncMlxDownload = useCallback(async () => {
    if (debugLlm?.provider !== "mlx" || !currentModel) {
      setMlxDownloadJob(null);
      return;
    }
    try {
      const r = await api.mlxDownloadList();
      const active = r.jobs.find(
        (j) => j.repo_id === currentModel && (j.status === "running" || j.status === "pending"),
      ) ?? null;
      setMlxDownloadJob(active);
    } catch {
      // best-effort — don't surface polling errors
    }
  }, [debugLlm?.provider, currentModel]);

  const mlxIsDownloading = mlxDownloadJob !== null;
  // Fast poll (800ms) while a download is active for live progress;
  // slow poll (10s) otherwise so we notice when a download starts
  // without hammering the backend from every open chat window.
  usePolling(syncMlxDownload, mlxIsDownloading ? 800 : 10_000, debugLlm?.provider === "mlx");

  const isModelBusy = mlxIsDownloading || exoModelLoading || omlxModelLoading;

  useEffect(() => {
    api.getSettings().then((s) => {
      setAppSettings(s);
      const mlx = s.llm.mlx ?? { hf_llm_model_id: "", hf_vlm_model_id: "", hf_draft_llm_model_id: "", hf_token: "" };
      if (s.llm.provider === "mlx") {
        const mid = mlx.hf_llm_model_id?.trim() || localStorage.getItem("chatModel") || "";
        setCurrentModel(mid);
        if (mid) localStorage.setItem("chatModel", mid);
        setDebugLlm({
          provider: "mlx",
          authMode: mlx.hf_token ? "hub + token" : "hub",
          hasKeys: !!mid,
          region: "",
        });
        // Populate model picker with every model already in the Hub cache.
        // Falls back to just the configured id if the scan fails or returns nothing.
        api.mlxLocalModels()
          .then((r) => {
            const rows = r.models ?? [];
            if (rows.length > 0) {
              setAvailableModels(rows.map((m) => ({ id: m.repo_id, name: m.name })));
            } else if (mid) {
              setAvailableModels([{ id: mid, name: mid }]);
            } else {
              setAvailableModels([]);
            }
          })
          .catch(() => {
            setAvailableModels(mid ? [{ id: mid, name: mid }] : []);
          })
          .finally(() => setModelsFetched(true));
        return;
      }
      if (s.llm.provider === "exo") {
        const mid = s.exo.model_name?.trim() || localStorage.getItem("chatModel") || "";
        setCurrentModel(mid);
        if (mid) localStorage.setItem("chatModel", mid);
        setDebugLlm({
          provider: "exo",
          authMode: s.exo.base_url || `127.0.0.1:${s.exo.api_port}`,
          hasKeys: !!mid && s.exo.enabled,
          region: "",
        });
        // Populate the model picker from the live cluster catalog so the
        // user can switch between any model exo has (downloaded or loaded)
        // without leaving the chat page.  Falls back to just the currently
        // configured id if the cluster is unreachable.
        api.exoModels()
          .then((r) => {
            const catalog = r.reachable ? r.models : [];
            exoCatalogRef.current = catalog;
            const rich = catalog
              .filter((m) => m.downloaded || m.loaded)
              .map((m) => ({ id: m.id, name: m.name }));
            if (rich.length > 0) {
              setAvailableModels(rich);
            } else if (mid) {
              setAvailableModels([{ id: mid, name: mid }]);
            } else {
              setAvailableModels([]);
            }
          })
          .catch((e) => {
            console.warn("Failed to list EXO models:", e);
            exoCatalogRef.current = [];
            setAvailableModels(mid ? [{ id: mid, name: mid }] : []);
          })
          .finally(() => setModelsFetched(true));
        return;
      }
      if (s.llm.provider === "openai") {
        const o = s.llm.openai;
        setCurrentModel(o.model_name);
        localStorage.setItem("chatModel", o.model_name);
        const activeKey = o.model_provider === "azure" ? o.azure_api_key : o.api_key;
        setDebugLlm({
          provider: o.model_provider === "azure" ? "azure" : "openai",
          authMode: o.model_provider === "azure" ? o.azure_endpoint || "azure" : "api_key",
          hasKeys: !!(activeKey),
          region: o.model_provider === "azure" ? (o.azure_endpoint || "") : "",
        });
        if (!modelsFetched) {
          api.listModels({ provider: "openai", api_key: activeKey, model_name: o.model_name, openai_model_provider: o.model_provider, azure_endpoint: o.azure_endpoint, azure_api_version: o.azure_api_version, azure_deployment: o.azure_deployment })
            .then((r) => { if (r.models?.length) setAvailableModels(r.models); })
            .catch((e) => console.warn("Failed to list OpenAI models:", e))
            .finally(() => setModelsFetched(true));
        }
        return;
      }
      if (s.llm.provider === "omlx") {
        const mid = s.omlx?.model_name?.trim() || localStorage.getItem("chatModel") || "";
        setCurrentModel(mid);
        if (mid) localStorage.setItem("chatModel", mid);
        setDebugLlm({
          provider: "omlx",
          authMode: `127.0.0.1:${s.omlx?.api_port ?? 8000}`,
          hasKeys: !!mid && !!(s.omlx?.enabled),
          region: "",
        });
        api.omlxStatus()
          .then((r) => {
            const models = r.models ?? [];
            if (models.length > 0) {
              setAvailableModels(models.map((m: { id: string }) => ({ id: m.id, name: m.id })));
            } else if (mid) {
              setAvailableModels([{ id: mid, name: mid }]);
            } else {
              setAvailableModels([]);
            }
          })
          .catch(() => {
            setAvailableModels(mid ? [{ id: mid, name: mid }] : []);
          })
          .finally(() => setModelsFetched(true));
        return;
      }
      const a = s.llm.anthropic;
      setCurrentModel(a.model_name);
      localStorage.setItem("chatModel", a.model_name);
      setDebugLlm({
        provider: a.model_provider,
        authMode: a.model_provider === "bedrock" ? a.bedrock_auth_mode : "api_key",
        hasKeys: !!(a.aws_access_key_id || a.api_key),
        region: a.bedrock_region,
      });
      if (!modelsFetched) {
        api.listModels({ provider: s.llm.provider, api_key: a.api_key, model_name: a.model_name, model_provider: a.model_provider, bedrock_region: a.bedrock_region, bedrock_auth_mode: a.bedrock_auth_mode, aws_access_key_id: a.aws_access_key_id, aws_secret_access_key: a.aws_secret_access_key })
          .then((r) => { if (r.models?.length) setAvailableModels(r.models); })
          .catch((e) => console.warn("Failed to list models:", e))
          .finally(() => setModelsFetched(true));
      }
    }).catch((e) => console.warn("Failed to load settings:", e));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps
  const scrollRaf = useRef(0);
  useEffect(() => {
    if (scrollRaf.current) cancelAnimationFrame(scrollRaf.current);
    scrollRaf.current = requestAnimationFrame(() => {
      scrollRaf.current = 0;
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    });
  }, [messages]);

  useEffect(() => {
    if (!currentSessionId) { setSessionFiles([]); return; }
    const poll = () => {
      api.listSessionFiles(currentSessionId).then(setSessionFiles).catch((e) => console.warn("Failed to list session files:", e));
    };
    poll();
    if (!isStreaming) return;
    const interval = setInterval(poll, 5000);
    return () => clearInterval(interval);
  }, [currentSessionId, isStreaming]);

  useEffect(() => {
    if (!sessionId) return;

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

  const handleMessage = useCallback((msg: WSMessage) => {
    const sid = sessionIdRef.current ?? "";
    if (msg.type === "done") {
      setIsStreaming(false);
      setPendingContext([]);
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
    if (msg.type === "memory_search") {
      setStreamPhase("memory_search");
      return;
    }
    if (msg.type === "memory_context") {
      pendingMemoryTopicsRef.current = (msg.metadata?.topics as string[]) ?? null;
      setStreamPhase("thinking");
      return;
    }
    if (msg.type === "hitl_request" || msg.type === "ask_user") {
      setIsStreaming(false);
      if (sid) { unwatchSession(sid); notify("hitl", sid); }
      const chatMsg: ChatMessage = { id: `msg-${++msgIdRef.current}`, type: msg.type, content: msg.content, metadata: msg.metadata, timestamp: new Date(), sessionId: sid || undefined };
      setMessages((prev) => {
        // Dedupe re-emitted interrupts: the backend re-pushes pending
        // interrupts on every WS (re)connect so dropped deliveries
        // recover, but if the client already received it we'd otherwise
        // render two identical approval cards.  Compare against the most
        // recent unresolved interrupt in state.
        for (let i = prev.length - 1; i >= 0; i--) {
          const m = prev[i];
          if (m.type !== "hitl_request" && m.type !== "ask_user") continue;
          if (m.metadata?.resolved) break;
          if (
            m.type === msg.type
            && m.content === msg.content
            && JSON.stringify(m.metadata) === JSON.stringify(msg.metadata)
          ) {
            return prev;
          }
          break;
        }
        return [...prev, chatMsg];
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
    const chatMsg: ChatMessage = { id: `msg-${++msgIdRef.current}`, type: msg.type, content: msg.content, metadata: meta, timestamp: new Date(), sessionId: sid || undefined };
    setMessages((prev) => [...prev, chatMsg]);
  }, [navigate, notify, unwatchSession, refreshSessionFiles, setLastError]);

  const { connected, send, sendEdit, sendHitlResponse, sendContext, waitForConnection } = useWebSocket({ sessionId: currentSessionId, onMessage: handleMessage });

  useEffect(() => {
    setWsConnected(connected);
  }, [connected, setWsConnected]);

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
          const withImages = preserveImages(apiMerged, prev);
          const apiHasPendingHitl = withImages.some(
            (m) => (m.type === "hitl_request" || m.type === "ask_user") && !m.metadata?.resolved,
          );
          if (!apiHasPendingHitl) {
            const localPending = prev.filter(
              (m) => (m.type === "hitl_request" || m.type === "ask_user") && !m.metadata?.resolved,
            );
            if (localPending.length > 0) return [...withImages, ...localPending];
          }
          return withImages;
        });
        if (!status.running && !sendingRef.current) setIsStreaming(false);
      }).catch(() => {
        if (!status.running && !sendingRef.current) setIsStreaming(false);
      });
    }).catch(() => {});
  }, [connected, currentSessionId]);

  // Polling safety-net: while isStreaming=true, fetch messages + status from
  // the API every 2 s so that any WS message that was lost (hitl_request,
  // agent responses, done) will still appear within a couple of seconds.
  //
  // Design notes:
  //   • No sendingRef.current guard here — the effect re-runs whenever
  //     isStreaming flips to true (which happens both in handleSend and
  //     handleHitlDecision), so the interval always starts.
  //   • Messages are synced on every tick, not only when the agent stops,
  //     so in-progress tool calls and text responses also appear promptly.
  //   • We reuse existing message IDs by position to avoid remounting
  //     components (which would reset any typed credential input state).
  //   • The interval stops itself once status.running = false to avoid
  //     hammering the API after the session finishes.
  const streamingPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  useEffect(() => {
    if (!isStreaming || !currentSessionId) {
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
          const apiMerged = mergeToolMessages(raw);
          setMessages((prev) => {
            // Keep local state when it has more messages than the API (in-flight
            // WS messages not yet persisted).  When counts are equal, compare
            // the last message's content and type: if the API version differs
            // (e.g. a final agent response replaced a "thinking" chunk that
            // arrived via WS before it was persisted) apply the API state.
            // Reuse existing IDs by position so components don't remount.
            if (apiMerged.length < prev.length) return prev;
            if (apiMerged.length === prev.length) {
              const lastApi = apiMerged[apiMerged.length - 1];
              const lastLocal = prev[prev.length - 1];
              if (
                lastApi?.content === lastLocal?.content &&
                lastApi?.type === lastLocal?.type &&
                JSON.stringify(lastApi?.metadata) === JSON.stringify(lastLocal?.metadata)
              ) return prev;
            }
            return preserveImages(apiMerged, prev).map((m, i) =>
              i < prev.length ? { ...m, id: prev[i].id } : m,
            );
          });
        }

        if (status && !status.running) {
          clearInterval(streamingPollRef.current!);
          streamingPollRef.current = null;
          setIsStreaming(false);
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
  }, [isStreaming, currentSessionId]);

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

  const handleHitlDecision = useCallback((messageId: string, decisions: Array<Record<string, unknown>>) => {
    if (!sendHitlResponse(decisions)) {
      setMessages((prev) => [
        ...prev,
        { id: `msg-${++msgIdRef.current}`, type: "error" as const, content: "Not connected to server — please wait and try again", timestamp: new Date(), sessionId: currentSessionId ?? sessionId ?? undefined },
      ]);
      return;
    }
    if (currentSessionId) { clearSession(currentSessionId); watchSession(currentSessionId); }
    setMessages((prev) => prev.map((m) =>
      m.id === messageId
        ? { ...m, metadata: { ...m.metadata, resolved: true, decisions } }
        : m,
    ));
    setIsStreaming(true);
  }, [sendHitlResponse, clearSession, watchSession, currentSessionId, sessionId]);

  // Per-session auto-approve: enabled by the "Approve all" action and
  // automatically reset whenever the user opens a different session, so
  // every new session starts by requiring approval again.
  const [sessionAutoApprove, setSessionAutoApprove] = useState(false);
  useEffect(() => { setSessionAutoApprove(false); }, [currentSessionId]);

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
    const actions = (unresolved.metadata?.action_requests as Array<{ name: string; args: Record<string, unknown> }> | undefined) ?? [];
    const hasHighRisk = actions.some((a) => screenHighRiskCommand(a.args?.command).length > 0);
    if (hasHighRisk) return;
    handleHitlDecision(unresolved.id, actions.map(() => ({ type: "approve" })));
  }, [shouldAutoApprove, messages, handleHitlDecision]);

  const startSession = async () => {
    let data: Awaited<ReturnType<typeof api.createSession>>;
    try {
      data = await api.createSession({ agent_name: selectedAgent || null });
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

  const addFiles = (files: FileList | File[]) => {
    const arr = Array.from(files);
    if (arr.length) setPendingFiles((prev) => [...prev, ...arr]);
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

  const handleSend = async () => {
    if (sendingRef.current) return;
    const text = (pendingVoiceTranscriptRef.current ?? input).trim();
    if (!text && pendingFiles.length === 0 && pendingFilePaths.length === 0 && pendingFolders.length === 0) return;

    if (currentSessionId) clearSession(currentSessionId);
    clearError();
    sendingRef.current = true;

    const filesToUpload = [...pendingFiles];
    const filePathsToSend = [...pendingFilePaths];
    const foldersToSend = [...pendingFolders];
    setInput("");
    localStorage.removeItem("chatDraft");
    localStorage.removeItem("chatSelectedAgent");
    setStreamPhase("thinking");
    setIsStreaming(true);

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

      // Symlink dropped files/folders into the session so the agent can
      // reach them via standard virtual-path tools (ls, read_file, etc.).
      // We do this instead of sending absolute paths because the deepagents
      // tools are scoped to the session files dir; absolute paths get
      // mangled to <session>/<absolute path> and silently fail.
      let linkedFiles: string[] = [];
      let linkedFolders: string[] = [];
      const dropped = [
        ...filePathsToSend.map((p) => ({ source: p, isFolder: false })),
        ...foldersToSend.map((p) => ({ source: p, isFolder: true })),
      ];
      if (dropped.length > 0) {
        try {
          const linkResults = await Promise.all(
            dropped.map((d) => api.createSessionLink(sid!, d.source)),
          );
          linkResults.forEach((r, i) => {
            if (dropped[i].isFolder || r.is_dir) linkedFolders.push(r.path);
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
      setMessages((prev) => [...prev, userMsg]);

      if (!send(fullText)) {
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

    if (!sendEdit(userMsgIndex, newContent)) {
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
  }, [messages, isStreaming, sendEdit, watchSession, currentSessionId, sessionId]);

  const handleModelChange = async (modelId: string) => {
    setCurrentModel(modelId);
    localStorage.setItem("chatModel", modelId);
    try {
      const s = await api.getSettings() as AppSettings;
      if (s.llm.provider === "mlx") {
        const nextMlx = { ...(s.llm.mlx ?? { hf_llm_model_id: "", hf_vlm_model_id: "", hf_draft_llm_model_id: "", hf_token: "" }), hf_llm_model_id: modelId };
        const nextSettings = { ...s, llm: { ...s.llm, mlx: nextMlx } } as AppSettings;
        await api.updateSettings(nextSettings as unknown as Record<string, unknown>);
        setAppSettings(nextSettings);
        setDebugLlm({
          provider: "mlx",
          authMode: nextMlx.hf_token ? "hub + token" : "hub",
          hasKeys: !!modelId.trim(),
          region: "",
        });
        return;
      }
      if (s.llm.provider === "exo") {
        const nextExo = { ...s.exo, model_name: modelId };
        const nextSettings = { ...s, exo: nextExo } as AppSettings;
        await api.updateSettings(nextSettings as unknown as Record<string, unknown>);
        setAppSettings(nextSettings);
        setDebugLlm({
          provider: "exo",
          authMode: nextExo.base_url || `127.0.0.1:${nextExo.api_port}`,
          hasKeys: !!modelId.trim() && nextExo.enabled,
          region: "",
        });
        // Fetch a fresh catalog to get accurate downloaded/loaded state, then
        // unload any other loaded model and warm-load the selected one.
        // Uses the async job API (same as the settings page) so we can poll
        // progress without blocking the UI thread.
        const trimmedId = modelId.trim();
        if (trimmedId && nextExo.enabled && !exoAutoPreloadInFlightRef.current) {
          exoAutoPreloadInFlightRef.current = true;
          setExoModelLoading(true);
          void (async () => {
            try {
              // Fresh catalog so we don't act on stale downloaded/loaded flags.
              const fresh = await api.exoModels();
              if (fresh.reachable) exoCatalogRef.current = fresh.models;
              const freshCatalog = fresh.reachable ? fresh.models : exoCatalogRef.current;
              const newModel = freshCatalog.find((c) => c.id === trimmedId);
              const loadedOthers = freshCatalog.filter((c) => c.loaded && c.id !== trimmedId);
              for (const prev of loadedOthers) {
                await api.exoUnload(prev.id);
              }
              // Only load if the model is on disk but not yet in cluster memory.
              if (newModel?.downloaded && !newModel.loaded) {
                const job = await api.exoPreloadStart(trimmedId, 1);
                const startedAt = Date.now();
                while (Date.now() - startedAt < 600_000) {
                  await new Promise<void>((res) => setTimeout(res, 2000));
                  const j = await api.exoPreloadStatus(job.job_id);
                  if (j.status === "done" || j.status === "error" || j.status === "cancelled") break;
                }
                const r = await api.exoModels();
                if (r.reachable) exoCatalogRef.current = r.models;
              }
            } catch (e) {
              console.warn("Auto unload/preload of EXO model failed:", e);
            } finally {
              exoAutoPreloadInFlightRef.current = false;
              setExoModelLoading(false);
            }
          })();
        }
        return;
      }
      if (s.llm.provider === "openai") {
        const nextOpenAI = { ...s.llm.openai, model_name: modelId };
        const nextSettings = { ...s, llm: { ...s.llm, openai: nextOpenAI } } as AppSettings;
        await api.updateSettings(nextSettings as unknown as Record<string, unknown>);
        setAppSettings(nextSettings);
        const o = nextOpenAI;
        const activeKey = o.model_provider === "azure" ? o.azure_api_key : o.api_key;
        setDebugLlm({
          provider: o.model_provider === "azure" ? "azure" : "openai",
          authMode: o.model_provider === "azure" ? o.azure_endpoint || "azure" : "api_key",
          hasKeys: !!(activeKey),
          region: o.model_provider === "azure" ? (o.azure_endpoint || "") : "",
        });
        return;
      }
      if (s.llm.provider === "omlx") {
        const nextOmlx = { ...s.omlx, model_name: modelId };
        const nextSettings = { ...s, omlx: nextOmlx } as AppSettings;
        await api.updateSettings(nextSettings as unknown as Record<string, unknown>);
        setAppSettings(nextSettings);
        setDebugLlm({
          provider: "omlx",
          authMode: `127.0.0.1:${nextOmlx.api_port ?? 8000}`,
          hasKeys: !!modelId.trim() && !!(nextOmlx.enabled),
          region: "",
        });
        const trimmedOmlxId = modelId.trim();
        if (trimmedOmlxId && nextOmlx.enabled) {
          setOmlxModelLoading(true);
          void (async () => {
            try {
              const status = await api.omlxStatus();
              const alreadyLoaded = status.models.some((m) => m.id === trimmedOmlxId);
              if (!alreadyLoaded) {
                const job = await api.omlxLoadModel(trimmedOmlxId);
                const startedAt = Date.now();
                while (Date.now() - startedAt < 600_000) {
                  await new Promise<void>((res) => setTimeout(res, 2000));
                  const j = await api.getOmlxJob(job.job_id);
                  if (j.status === "done" || j.status === "error") break;
                }
              }
            } catch (e) {
              console.warn("oMLX model load failed:", e);
            } finally {
              setOmlxModelLoading(false);
            }
          })();
        }
        return;
      }
      s.llm.anthropic.model_name = modelId;
      await api.updateSettings(s as unknown as Record<string, unknown>);
      setAppSettings(s);
      const a = s.llm.anthropic;
      setDebugLlm({
        provider: a.model_provider,
        authMode: a.model_provider === "bedrock" ? a.bedrock_auth_mode : "api_key",
        hasKeys: !!(a.aws_access_key_id || a.api_key),
        region: a.bedrock_region,
      });
    } catch (e) { console.warn("Failed to save model settings:", e); }
  };

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

  const memoryHits = useMemo(() => {
    const agent = sessionMessages.filter((m) => m.type === "agent" && !m.metadata?.subagent);
    const withMemory = agent.filter((m) => (m.metadata?.memory_topics as string[] | undefined)?.length);
    return { total: agent.length, hits: withMemory.length };
  }, [sessionMessages]);

  const thoughtFlags = useMemo(() => {
    const flags = new Array<boolean>(sessionMessages.length).fill(false);
    let hasFollowUp = false;
    for (let i = sessionMessages.length - 1; i >= 0; i--) {
      const m = sessionMessages[i];
      if (m.type === "agent" && !m.metadata?.subagent) {
        flags[i] = hasFollowUp;
        hasFollowUp = true;
      } else if (m.type === "tool_call") {
        hasFollowUp = true;
      }
    }
    return flags;
  }, [sessionMessages]);

  const latestTodoFlags = useMemo(() => {
    const flags = new Array<boolean>(sessionMessages.length).fill(false);
    for (let i = sessionMessages.length - 1; i >= 0; i--) {
      const m = sessionMessages[i];
      if ((m.type === "tool_call" || m.type === "tool_result") && m.content === "write_todos" && !m.metadata?.subagent) {
        flags[i] = true;
        break;
      }
    }
    return flags;
  }, [sessionMessages]);

  const visibleSessionFiles = useMemo(
    () => sessionFiles.filter(f => !f.path.startsWith("large_tool_results/")),
    [sessionFiles],
  );

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

  const renderItems = useMemo((): RenderItem[] => {
    const items: RenderItem[] = [];
    const groups = new Map<string, RenderItem & { kind: "subagent-group" }>();
    const subagentPrefixes = new Set<string>();
    let generation = 0;
    let prevWasSubagent = false;
    for (let i = 0; i < sessionMessages.length; i++) {
      const msg = sessionMessages[i];
      // Errors are pulled out of the inline flow and pinned to the end of the
      // chat (see errorMessages) so they're always visible after the latest turn.
      if (msg.type === "error") continue;
      const subagent = msg.metadata?.subagent as string | undefined;
      if (subagent) {
        subagentPrefixes.add(subagent.replace(/ #\d+$/, ""));
        const groupKey = `${subagent}::${generation}`;
        let group = groups.get(groupKey);
        if (!group) {
          group = { kind: "subagent-group", name: subagent, messages: [] };
          groups.set(groupKey, group);
          items.push(group);
        }
        group.messages.push(msg);
        prevWasSubagent = true;
      } else {
        if (prevWasSubagent) generation++;
        prevWasSubagent = false;
        items.push({ kind: "message", message: msg, index: i });
      }
    }
    if (subagentPrefixes.size === 0) return items;
    return items.filter((item) => {
      if (item.kind !== "message") return true;
      const msg = item.message;
      if ((msg.type !== "tool_call" && msg.type !== "tool_result") || msg.content !== "task") return true;
      const agentType = (msg.metadata?.args as Record<string, unknown>)?.subagent_type as string | undefined;
      return !agentType || !subagentPrefixes.has(agentType);
    });
  }, [sessionMessages]);

  // Errors are surfaced at the very end of the chat (after the latest agent
  // turn) rather than inline where they occurred, so they're never buried.
  const errorMessages = useMemo(
    () => sessionMessages.filter((m) => m.type === "error"),
    [sessionMessages],
  );



  return (
    <div className="flex h-full min-h-0">
      <div className="flex flex-col flex-1 min-w-0 h-full">
      <header className="border-b border-th-border px-6 py-3.5 flex items-center justify-between shrink-0 bg-th-bg-secondary">
        <div className="flex items-center gap-3">
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
        </div>
        <div className="flex items-center gap-3">
          {!currentSessionId && !selectedAgent && (
            <span
              className="text-xs text-th-text-tertiary"
              title="Slash to pick a subagent before sending; otherwise the orchestrator will route the message."
            >
              Type <kbd className="px-1.5 py-0.5 bg-th-inset-bg border border-th-border rounded text-[10px] font-mono text-th-text-muted">/</kbd> to pick a subagent
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
              className="inline-flex items-center gap-1.5 text-[11px] px-2 py-1 rounded-full bg-blue-500/10 text-blue-400/80 border border-blue-500/15"
              title={`Memory injected in ${memoryHits.hits} of ${memoryHits.total} responses`}
            >
              <Brain size={11} />
              {memoryHits.hits}/{memoryHits.total}
            </span>
          )}
        </div>
      </header>

      <div className="flex-1 overflow-y-auto px-8 py-8 space-y-7">
        {sessionMessages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full -mt-12">
            <img
              src={theme === "dark" ? logoDark : logoLight}
              alt="Otto"
              className="w-20 h-20 mb-5 select-none object-contain opacity-90"
              draggable={false}
            />
            <h3 className="text-xl font-semibold text-th-text-primary mb-2">How can I help?</h3>
            <p className="text-sm text-th-text-tertiary max-w-sm text-center leading-relaxed mb-7">
              Your AI agent — browse, write, run code, and more.
            </p>
            <div className="flex flex-wrap gap-2 justify-center max-w-lg">
              {[
                "Write a Python script",
                "Summarise a document",
                "Research a topic online",
                "Debug this code",
                "Plan a project",
                "Draft an email",
              ].map((prompt) => (
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
        {!isStreaming && viewableSessionFiles.length > 0 && (
          <div className="mt-4 ml-10 p-3 rounded-xl border border-th-border bg-th-card-bg">
            <p className="text-[10px] uppercase tracking-wider font-semibold text-th-text-muted mb-2">
              Output files
            </p>
            <div className="grid grid-cols-5 gap-2">
              {viewableSessionFiles.map((f) => {
                const vt = getViewType(f.path)!;
                const name = f.path.split("/").pop() ?? f.path;
                const iconCls =
                  vt === "pdf"  ? "text-red-400" :
                  vt === "csv" || vt === "xlsx" ? "text-emerald-400" :
                  vt === "image" ? "text-purple-400" :
                  vt === "json" ? "text-amber-400" :
                  "text-blue-400";
                const FileIcon = vt === "image" ? ImageIcon : vt === "json" ? FileJson : FileText;
                return (
                  <button
                    key={f.path}
                    onClick={() => handleOpenArtifact(f.path, api.getSessionFileUrl(currentSessionId!, f.path), vt)}
                    className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-th-border bg-th-bg-secondary hover:bg-th-surface-hover hover:border-th-border-strong transition-colors cursor-pointer group text-left min-w-0"
                    title={f.path}
                  >
                    <FileIcon size={12} className={`shrink-0 ${iconCls}`} />
                    <span className="text-xs font-medium text-th-text-secondary group-hover:text-th-text-primary transition-colors truncate flex-1">
                      {name}
                    </span>
                    <ExternalLink size={10} className="text-th-text-muted group-hover:text-blue-400 transition-colors shrink-0" />
                  </button>
                );
              })}
            </div>
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
              className="flex items-center gap-1 text-[10px] text-th-text-tertiary hover:text-th-text-primary transition-colors px-1.5 py-0.5 rounded hover:bg-th-surface-hover"
              title="Open in file manager"
            >
              <ExternalLink size={11} />
              <span>Open folder</span>
            </button>
          </div>
          {showFiles && (
            <div className="mt-1.5 mb-1 space-y-1 max-h-40 overflow-y-auto">
              {visibleSessionFiles.map((f) => {
                const vt = getViewType(f.path);
                const isViewable = vt !== null;
                const iconCls = vt === "pdf"  ? "text-red-400/70 group-hover:text-red-400"
                  : vt === "csv" || vt === "xlsx" ? "text-emerald-400/70 group-hover:text-emerald-400"
                  : vt === "image" ? "text-purple-400/70 group-hover:text-purple-400"
                  : vt === "json" ? "text-amber-400/70 group-hover:text-amber-400"
                  : isViewable ? "text-blue-400/70 group-hover:text-blue-400"
                  : "text-th-text-muted group-hover:text-th-text-secondary";
                const FileIcon = vt === "image" ? ImageIcon : vt === "json" ? FileJson : FileText;
                return (
                <div
                  key={f.path}
                  className={`flex items-center gap-2.5 px-3 py-1.5 rounded-lg hover:bg-th-surface-hover transition-colors group ${isViewable ? "cursor-pointer" : "cursor-default"}`}
                  onClick={isViewable ? () => handleOpenArtifact(f.path, api.getSessionFileUrl(currentSessionId!, f.path), vt!) : undefined}
                >
                  <FileIcon size={13} className={`shrink-0 ${iconCls}`} />
                  <span className="text-xs text-th-text-secondary truncate flex-1 font-mono">{f.path}</span>
                  <span className="text-[10px] text-th-text-muted shrink-0">{formatFileSize(f.size)}</span>
                  {isViewable && (
                    <span className="text-[10px] text-th-text-muted group-hover:text-blue-400 transition-colors shrink-0 font-medium">
                      Open ↗
                    </span>
                  )}
                  <button
                    onClick={(e) => { e.stopPropagation(); api.openSessionFilesFolder(currentSessionId!, f.path).catch((e) => console.warn("Failed to open folder:", e)); }}
                    className="text-th-text-muted hover:text-th-text-secondary transition-colors shrink-0"
                    title="Show in folder"
                  >
                    <FolderOpen size={12} />
                  </button>
                  {downloadedFile === f.path ? (
                    <span className="inline-flex items-center gap-1 text-[10px] text-emerald-400 font-medium shrink-0"><CheckCircle2 size={12} /> Downloaded</span>
                  ) : (
                    <a
                      href={api.getSessionFileUrl(currentSessionId!, f.path)}
                      download
                      onClick={(e) => { e.stopPropagation(); setDownloadedFile(f.path); setTimeout(() => setDownloadedFile((prev) => prev === f.path ? null : prev), 3000); }}
                      className="text-th-text-muted hover:text-th-text-secondary transition-colors shrink-0"
                      title="Download"
                    >
                      <Download size={12} />
                    </a>
                  )}
                  <button
                    onClick={(e) => { e.stopPropagation(); api.deleteSessionFile(currentSessionId!, f.path).then(() => setSessionFiles((prev) => prev.filter((x) => x.path !== f.path))); }}
                    className="text-th-text-muted hover:text-red-400 transition-colors shrink-0"
                    title="Delete"
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      <div
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
                  <button onClick={() => removeFolder(i)} className="text-blue-400/60 hover:text-blue-300 transition-colors ml-0.5">
                    <X size={11} />
                  </button>
                </div>
              );
            })}
          </div>
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
                  <kbd className="inline-flex items-center px-1.5 py-0.5 rounded-md bg-violet-500/20 border border-violet-400/30 font-mono text-[10px] text-violet-300 leading-none">
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
                <span className="text-[10px] uppercase tracking-wider text-th-text-muted font-semibold">Select Agent</span>
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
                  {item.description && <span className="text-[11px] text-th-text-muted truncate">{item.description}</span>}
                </button>
              ))}
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
            {exoModelLoading && (
              <span className="inline-flex items-center gap-1 text-[10px] text-amber-400 shrink-0">
                <Loader2 size={11} className="animate-spin" />
                Loading…
              </span>
            )}
            {omlxModelLoading && (
              <span className="inline-flex items-center gap-1 text-[10px] text-amber-400 shrink-0">
                <Loader2 size={11} className="animate-spin" />
                Loading…
              </span>
            )}
            {debugLlm && (
              <div className="flex items-center gap-1.5 text-[10px] text-th-text-muted truncate">
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
                    className={`inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full border font-medium ${tone}`}
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
      <div className="h-full shrink-0 flex" style={{ width: artifactWidth }}>
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


// ---------------------------------------------------------------------------
// MessageList — memoized so that typing in the input textarea does NOT
// trigger re-renders of the (potentially very long) message history.
// All callback props must be wrapped in useCallback in ChatPage.
// ---------------------------------------------------------------------------

interface MessageListProps {
  items: RenderItem[];
  thoughtFlags: boolean[];
  latestTodoFlags: boolean[];
  isStreaming: boolean;
  currentSessionId: string | null;
  resolveSubagentModel: (name: string) => { label: string; family: string };
  onEditMessage: (idx: number, content: string) => void;
  onHitlDecision: (id: string, decisions: Array<Record<string, unknown>>) => void;
  onApproveAllSession: (id: string, decisions: Array<Record<string, unknown>>) => void;
  onOpenArtifact: (path: string, fileUrl: string, type: ArtifactType) => void;
}

const MessageList = memo(function MessageList({
  items,
  thoughtFlags,
  latestTodoFlags,
  isStreaming,
  currentSessionId,
  resolveSubagentModel,
  onEditMessage,
  onHitlDecision,
  onApproveAllSession,
  onOpenArtifact,
}: MessageListProps) {
  return (
    <>
      {items.map((item) => {
        if (item.kind === "subagent-group") {
          const model = resolveSubagentModel(item.name);
          return (
            <SubagentGroup
              key={`group-${item.name}`}
              name={item.name}
              messages={item.messages}
              modelLabel={model.label}
              modelFamily={model.family}
              sessionId={currentSessionId ?? undefined}
              onOpenArtifact={onOpenArtifact}
            />
          );
        }
        const { message: msg, index: idx } = item;
        return (
          <MessageBubble
            key={msg.id}
            message={msg}
            isThought={thoughtFlags[idx]}
            isLatestTodo={latestTodoFlags[idx]}
            canEdit={msg.type === "user" && !isStreaming}
            sessionId={currentSessionId ?? undefined}
            onEdit={(newContent) => onEditMessage(idx, newContent)}
            onHitlDecision={(decisions) => onHitlDecision(msg.id, decisions)}
            onApproveAllSession={(decisions) => onApproveAllSession(msg.id, decisions)}
            onOpenArtifact={onOpenArtifact}
          />
        );
      })}
    </>
  );
});

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
            <p className="text-[11px] text-th-text-muted mt-1.5">Cron: {cron}</p>
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
      <div className="flex items-center justify-between gap-2 text-[11px]">
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
