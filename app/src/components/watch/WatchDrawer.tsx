import { useCallback, useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  X,
  Video,
  FileVideo,
  Monitor,
  Youtube,
  Radio,
  Circle,
  Square,
  Send,
  Loader2,
  AlertTriangle,
  MessageSquarePlus,
  Mic,
  Trash2,
} from "lucide-react";
import { api } from "../../hooks/useApi";
import { useWatch } from "../../hooks/useWatch";
import { emitAskOtto } from "../../utils/askOttoBus";
import { canSendAgentContext, emitAgentContext } from "../../utils/agentContextBus";
import { notifySessionFilesChanged } from "../../utils/sessionFilesBus";
import { formatFileSize } from "../../utils/formatFileSize";
import type { CapturePermission, VideoAudioDevice } from "../../types";

type SourceTab = "file" | "screen" | "youtube" | "live";
type LiveToAgent = "off" | "on_stop" | "stream";

const FPS_OPTIONS = [0.5, 1, 2, 5];

const SCREEN_SETTINGS_URL =
  "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture";

function openScreenSettings() {
  import("@tauri-apps/plugin-shell")
    .then(({ open }) => open(SCREEN_SETTINGS_URL))
    .catch(() => {
      /* best effort — no-op outside Tauri */
    });
}

/** Privacy lock arrives as a 403 with a JSON body; everything else is raw. */
function describeSessionError(e: unknown): string {
  const msg = e instanceof Error ? e.message : String(e);
  if (msg.includes("403") && msg.includes("privacy_lock")) {
    return "Privacy lock is on, so a chat can't be started. Allow a provider in Settings → Privacy, or switch to a local model.";
  }
  return `Could not start a chat: ${msg}`;
}

interface WatchDrawerProps {
  open: boolean;
  onClose: () => void;
}

export default function WatchDrawer({ open, onClose }: WatchDrawerProps) {
  const location = useLocation();
  const navigate = useNavigate();

  const sessionId = location.pathname.startsWith("/chat/")
    ? location.pathname.split("/chat/")[1]
    : null;

  const [tab, setTab] = useState<SourceTab>("file");
  const [question, setQuestion] = useState("");
  const [fps, setFps] = useState(1);
  const [geminiConfigured, setGeminiConfigured] = useState(false);
  const [livePreview, setLivePreview] = useState(true);
  const [liveToAgent, setLiveToAgent] = useState<LiveToAgent>("off");
  const [flushSecs, setFlushSecs] = useState(20);

  // File
  const [file, setFile] = useState<File | null>(null);
  const [uploadedPath, setUploadedPath] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // YouTube
  const [youtubeUrl, setYoutubeUrl] = useState("");

  // Screen recording. The session is pinned when recording starts so that
  // navigating away mid-recording doesn't strand the file.
  const [permission, setPermission] = useState<CapturePermission | null>(null);
  const [recording, setRecording] = useState(false);
  const [recordedPath, setRecordedPath] = useState<string | null>(null);
  const [recordedSize, setRecordedSize] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [recordAudio, setRecordAudio] = useState(false);
  const [audioDevices, setAudioDevices] = useState<VideoAudioDevice[]>([]);
  const recordingSessionRef = useRef<string | null>(null);

  // Analyze
  const [creatingSession, setCreatingSession] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Live
  const live = useWatch({ enabled: open && tab === "live" });
  const [livePrompt, setLivePrompt] = useState("");

  // ── Load settings (Gemini configured?) + recording status ────────────
  useEffect(() => {
    if (!open) return;
    api
      .getSettings()
      .then((s) => {
        setGeminiConfigured(!!s.llm?.google?.api_key);
        setLivePreview(s.video?.live_preview ?? true);
        setLiveToAgent((s.video?.live_to_agent as LiveToAgent) ?? "off");
        setFlushSecs(s.video?.live_agent_flush_secs ?? 20);
        setRecordAudio(s.video?.record_audio ?? false);
      })
      .catch(() => setGeminiConfigured(false));
    api
      .videoStatus()
      .then((st) => {
        setRecording(st.recording);
        if (st.recording && st.fps) setFps(st.fps);
      })
      .catch(() => {});
  }, [open]);

  // Poll recording elapsed time.
  useEffect(() => {
    if (!recording) {
      setElapsed(0);
      return;
    }
    const t = setInterval(() => {
      api.videoStatus().then((st) => {
        setRecording(st.recording);
        setElapsed(st.elapsed_secs || 0);
      }).catch(() => {});
    }, 1000);
    return () => clearInterval(t);
  }, [recording]);

  // Check screen recording permission when the Screen tab opens.
  useEffect(() => {
    if (open && tab === "screen") {
      api.videoPermission().then(setPermission).catch(() => {});
      api.videoAudioDevices()
        .then((r) => setAudioDevices(r.devices))
        .catch(() => setAudioDevices([]));
    }
  }, [open, tab]);

  // Preview what's being recorded. ffmpeg holds the capture device, so rather
  // than tapping its output the backend grabs its own frame; a cache-busting
  // query param is what actually drives the <img> to refetch.
  const [previewTick, setPreviewTick] = useState(0);
  useEffect(() => {
    if (!open || tab !== "screen" || !recording || !livePreview) return;
    const t = setInterval(() => setPreviewTick((n) => n + 1), 1000);
    return () => clearInterval(t);
  }, [open, tab, recording, livePreview]);

  /**
   * Resolve the session everything here writes into, starting a chat when
   * none is open — videos, recordings and results all live inside a session,
   * so the panel would otherwise be unusable from the home screen.
   */
  const ensureSession = useCallback(async (): Promise<string | null> => {
    if (sessionId) return sessionId;
    setCreatingSession(true);
    try {
      const created = await api.createSession({ agent_name: null });
      navigate(`/chat/${created.id}`);
      return created.id;
    } catch (e) {
      setError(describeSessionError(e));
      return null;
    } finally {
      setCreatingSession(false);
    }
  }, [sessionId, navigate]);

  const handlePickFile = useCallback(async (f: File) => {
    setError(null);
    setResult(null);
    setUploadedPath(null);
    setFile(f);
    setUploading(true);
    try {
      const sid = await ensureSession();
      if (!sid) return;
      const path = `uploads/${f.name}`;
      await api.uploadSessionFile(sid, path, f);
      setUploadedPath(`/${path}`);
      notifySessionFilesChanged(sid);
    } catch (e) {
      setError(`Upload failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setUploading(false);
    }
  }, [ensureSession]);

  const currentSource = useCallback((): string | null => {
    if (tab === "file") return uploadedPath;
    if (tab === "youtube") return youtubeUrl.trim() || null;
    if (tab === "screen") return recordedPath;
    return null;
  }, [tab, uploadedPath, youtubeUrl, recordedPath]);

  const handleAnalyze = useCallback(async () => {
    const source = currentSource();
    if (!source) {
      setError("Pick a video, record the screen, or paste a YouTube URL first.");
      return;
    }
    setAnalyzing(true);
    setError(null);
    setResult(null);
    try {
      const sid = await ensureSession();
      if (!sid) return;
      const res = await api.videoAnalyze({
        session_id: sid,
        source,
        question: question.trim() || undefined,
      });
      if (res.error) setError(res.error);
      else setResult(res.result || "(no result)");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setAnalyzing(false);
    }
  }, [currentSource, ensureSession, question]);

  const handleSendToChat = useCallback(() => {
    const source = currentSource();
    if (!source) {
      setError("Pick a video, record the screen, or paste a YouTube URL first.");
      return;
    }
    const q = question.trim();
    const msg = q
      ? `Watch this video and answer: ${q}\n\nVideo: ${source}`
      : `Watch this video and give me a detailed, timestamped summary.\n\nVideo: ${source}`;
    if (!location.pathname.startsWith("/chat")) navigate("/chat");
    emitAskOtto(msg);
    onClose();
  }, [currentSource, question, location.pathname, navigate, onClose]);

  const handleStartRecording = useCallback(async () => {
    setError(null);
    setRecordedPath(null);
    setRecordedSize(null);
    const sid = await ensureSession();
    if (!sid) return;
    const res = await api.videoRecordStart(sid, fps, recordAudio);
    if (res.error) {
      setError(res.error);
      return;
    }
    recordingSessionRef.current = sid;
    setRecording(true);
  }, [ensureSession, fps, recordAudio]);

  const handleStopRecording = useCallback(async () => {
    const sid = recordingSessionRef.current ?? sessionId;
    if (!sid) return;
    const res = await api.videoRecordStop(sid);
    recordingSessionRef.current = null;
    setRecording(false);
    if (res.error) {
      setError(res.error);
      return;
    }
    if (res.virtual_path) {
      setRecordedPath(res.virtual_path);
      setRecordedSize(res.size_bytes ?? null);
      notifySessionFilesChanged(sid);
    }
  }, [sessionId]);

  // ── Handing live commentary to the agent ─────────────────────────────
  // Commentary goes over the context channel, so it is folded into the next
  // thing the user asks rather than starting a turn of its own. Only lines
  // not yet handed over are sent.
  const handedOverRef = useRef(0);

  const flushCommentary = useCallback(() => {
    const pending = live.commentary.slice(handedOverRef.current);
    if (pending.length === 0) return;
    handedOverRef.current = live.commentary.length;
    emitAgentContext(
      "While watching the screen I observed:\n\n" +
        pending.map((c) => c.text).join("\n\n"),
    );
  }, [live.commentary]);

  // Held in a ref so the flush timer isn't torn down on every new line.
  const flushRef = useRef(flushCommentary);
  flushRef.current = flushCommentary;

  useEffect(() => {
    if (liveToAgent !== "stream" || !live.watching) return;
    const t = setInterval(() => flushRef.current(), Math.max(5, flushSecs) * 1000);
    return () => clearInterval(t);
  }, [liveToAgent, live.watching, flushSecs]);

  // Both hand-off modes flush the tail when watching ends.
  const wasWatchingRef = useRef(false);
  useEffect(() => {
    const was = wasWatchingRef.current;
    wasWatchingRef.current = live.watching;
    if (was && !live.watching && liveToAgent !== "off") flushRef.current();
  }, [live.watching, liveToAgent]);

  const handleToggleLive = useCallback(() => {
    if (live.watching) {
      live.stop();
    } else {
      live.clear();
      handedOverRef.current = 0;
      live.start({ prompt: livePrompt.trim() || undefined, fps: Math.min(1, fps) });
    }
  }, [live, livePrompt, fps]);

  if (!open) return null;

  const tabs: { id: SourceTab; label: string; icon: typeof Video }[] = [
    { id: "file", label: "File", icon: FileVideo },
    { id: "screen", label: "Screen", icon: Monitor },
    { id: "youtube", label: "YouTube", icon: Youtube },
    { id: "live", label: "Live", icon: Radio },
  ];

  return (
    <aside className="relative h-full w-[380px] shrink-0 flex flex-col bg-th-bg/70 backdrop-blur-xl border-l border-th-border shadow-[-8px_0_24px_-8px_rgba(0,0,0,0.08)]">
      {/* Header */}
      <div className="flex items-center justify-between px-4 h-14 border-b border-th-border/70 shrink-0">
        <div className="flex items-center gap-2.5 min-w-0">
          <Video size={18} className="text-sky-400 shrink-0" />
          <span className="text-sm font-semibold text-th-text-primary">Watch video</span>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 rounded-lg text-th-text-muted hover:text-th-text-primary hover:bg-th-surface-hover transition-all"
          title="Close"
        >
          <X size={16} />
        </button>
      </div>

      {/* Source tabs */}
      <div className="flex gap-1 p-2 border-b border-th-border/70 shrink-0">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            type="button"
            onClick={() => { setTab(id); setError(null); }}
            className={`flex-1 flex items-center justify-center gap-1.5 py-2 rounded-lg text-[12px] font-medium transition-all ${
              tab === id
                ? "bg-th-surface text-th-text-primary border border-th-border"
                : "text-th-text-tertiary hover:bg-th-surface-hover hover:text-th-text-primary"
            }`}
          >
            <Icon size={14} className="shrink-0" />
            {label}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {!sessionId && (
          <div className="flex items-start gap-2 text-[11px] text-th-text-tertiary bg-th-surface/60 border border-th-border rounded-lg p-2.5">
            <MessageSquarePlus size={13} className="shrink-0 mt-0.5" />
            <span>No chat is open — a new one will be started, and the video and its results saved into it.</span>
          </div>
        )}

        {/* ── FILE ─────────────────────────────────────────────── */}
        {tab === "file" && (
          <div className="space-y-3">
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) void handlePickFile(f);
              }}
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="w-full flex flex-col items-center justify-center gap-2 py-8 rounded-xl border-2 border-dashed border-th-border hover:border-th-border-strong hover:bg-th-surface-hover transition-all"
            >
              <FileVideo size={24} className="text-th-text-muted" />
              <span className="text-[12px] text-th-text-secondary">
                {file ? file.name : "Choose a video file"}
              </span>
            </button>
            {uploading && (
              <p className="text-[11px] text-th-text-muted flex items-center gap-1.5">
                <Loader2 size={11} className="animate-spin" />
                {creatingSession ? "Starting a chat…" : "Uploading…"}
              </p>
            )}
            {!uploading && uploadedPath && (
              <p className="text-[11px] text-emerald-400">Uploaded — ready to watch.</p>
            )}
          </div>
        )}

        {/* ── SCREEN ───────────────────────────────────────────── */}
        {tab === "screen" && (
          <div className="space-y-3">
            {permission && permission.granted === false && (
              <div className="text-[11px] text-amber-400 bg-amber-500/10 border border-amber-500/20 rounded-lg p-2.5 space-y-1.5">
                <p>Screen Recording permission is required.</p>
                <button
                  type="button"
                  onClick={openScreenSettings}
                  className="underline hover:text-amber-300"
                >
                  Open System Settings
                </button>
              </div>
            )}
            <label className="flex items-start gap-2 cursor-pointer group">
              <input
                type="checkbox"
                checked={recordAudio}
                disabled={recording}
                onChange={(e) => setRecordAudio(e.target.checked)}
                className="mt-0.5 accent-sky-500 disabled:opacity-40"
              />
              <span className="min-w-0">
                <span className="flex items-center gap-1.5 text-[12px] text-th-text-secondary group-hover:text-th-text-primary">
                  <Mic size={12} className="shrink-0" /> Record audio
                </span>
                <span className="block text-[10px] text-th-text-muted leading-relaxed mt-0.5">
                  {audioDevices.some((d) => d.is_loopback)
                    ? "Captures what's playing via your loopback device."
                    : "Captures your microphone. macOS can't record playback without a loopback device such as BlackHole."}
                </span>
              </span>
            </label>
            {!recording ? (
              <button
                type="button"
                onClick={handleStartRecording}
                disabled={creatingSession}
                className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-red-500/90 hover:bg-red-500 text-white text-[13px] font-medium transition-all disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <Circle size={14} className="fill-current" /> Record screen
              </button>
            ) : (
              <button
                type="button"
                onClick={handleStopRecording}
                className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-th-surface border border-th-border hover:bg-th-surface-hover text-th-text-primary text-[13px] font-medium transition-all"
              >
                <Square size={14} className="fill-current text-red-500" />
                Stop — {Math.floor(elapsed)}s
              </button>
            )}
            {recording && livePreview && (
              <div className="relative rounded-lg overflow-hidden border border-th-border bg-black/40 aspect-video">
                <img
                  src={api.videoPreviewUrl(previewTick)}
                  alt="Screen recording preview"
                  className="w-full h-full object-contain"
                />
                <span className="absolute top-1.5 left-1.5 flex items-center gap-1 px-1.5 py-0.5 rounded bg-black/60 text-[9px] font-medium text-white uppercase tracking-wider">
                  <Circle size={6} className="fill-red-500 text-red-500" />
                  Recording
                </span>
              </div>
            )}
            {recordedPath && (
              <a
                href={sessionId ? api.getSessionFileUrl(sessionId, recordedPath.replace(/^\//, "")) : undefined}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-3 py-2 rounded-lg border border-th-border bg-th-surface/60 hover:bg-th-surface-hover transition-colors group"
                title="Play the recording"
              >
                <Video size={13} className="shrink-0 text-rose-400" />
                <span className="text-[12px] text-th-text-secondary group-hover:text-th-text-primary truncate flex-1 font-mono">
                  {recordedPath.split("/").pop()}
                </span>
                {recordedSize !== null && (
                  <span className="text-[10px] text-th-text-muted shrink-0">
                    {formatFileSize(recordedSize)}
                  </span>
                )}
              </a>
            )}
          </div>
        )}

        {/* ── YOUTUBE ──────────────────────────────────────────── */}
        {tab === "youtube" && (
          <div className="space-y-2">
            <input
              type="text"
              value={youtubeUrl}
              onChange={(e) => setYoutubeUrl(e.target.value)}
              placeholder="https://youtube.com/watch?v=…"
              className="w-full px-3 py-2.5 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-400 text-sm"
            />
            <p className="text-[11px] text-th-text-muted">
              {geminiConfigured
                ? "Gemini will watch the URL directly (no download)."
                : "Without a Gemini key, the video is downloaded and watched via frames."}
            </p>
          </div>
        )}

        {/* ── LIVE ─────────────────────────────────────────────── */}
        {tab === "live" && (
          <div className="space-y-3">
            {!geminiConfigured && (
              <div className="flex items-start gap-2 text-[11px] text-th-text-tertiary bg-th-surface/60 border border-th-border rounded-lg p-2.5">
                <Radio size={13} className="shrink-0 mt-0.5" />
                <span>
                  Without a Gemini key this watches in short batches with your
                  local model — it describes the last few seconds every so often
                  rather than reacting instantly, and occupies the model while it
                  does.
                </span>
              </div>
            )}
            {live.watching && livePreview && (
              <div className="relative rounded-lg overflow-hidden border border-th-border bg-black/40 aspect-video">
                {live.frame ? (
                  <img
                    src={live.frame}
                    alt="Live screen preview"
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center text-[11px] text-th-text-muted">
                    Waiting for the first frame…
                  </div>
                )}
                <span className="absolute top-1.5 left-1.5 flex items-center gap-1 px-1.5 py-0.5 rounded bg-black/60 text-[9px] font-medium text-white uppercase tracking-wider">
                  <Circle size={6} className="fill-red-500 text-red-500" />
                  {live.mode === "local" ? "Local" : "Gemini"}
                </span>
              </div>
            )}
            <textarea
              value={livePrompt}
              onChange={(e) => setLivePrompt(e.target.value)}
              placeholder="What should Otto watch for? (optional)"
              rows={2}
              className="w-full px-3 py-2 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-400 text-sm resize-none"
            />
            <button
              type="button"
              onClick={handleToggleLive}
              className={`w-full flex items-center justify-center gap-2 py-3 rounded-xl text-[13px] font-medium transition-all ${
                live.watching
                  ? "bg-th-surface border border-th-border hover:bg-th-surface-hover text-th-text-primary"
                  : "bg-sky-500/90 hover:bg-sky-500 text-white"
              }`}
            >
              {live.watching ? (
                <><Square size={14} className="fill-current" /> Stop watching</>
              ) : (
                <><Radio size={14} /> Start live watching</>
              )}
            </button>
            {liveToAgent !== "off" && (
              <p className="text-[11px] text-th-text-muted">
                {!canSendAgentContext()
                  ? "Commentary can't reach the agent until a chat is open."
                  : liveToAgent === "stream"
                    ? `Commentary is passed to the agent every ${flushSecs}s as context, and folded into whatever you ask next.`
                    : "Commentary is passed to the agent as context when you stop watching."}
              </p>
            )}
            {live.error && (
              <p className="text-[11px] text-red-400">{live.error}</p>
            )}
            {live.commentary.length > 0 && (
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-medium text-th-text-tertiary">Commentary</span>
                  <button
                    type="button"
                    onClick={live.clear}
                    className="text-th-text-muted hover:text-th-text-primary"
                    title="Clear"
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
                <div className="space-y-1 max-h-[240px] overflow-y-auto text-[12px] text-th-text-secondary leading-relaxed">
                  {live.commentary.map((c) => (
                    <p key={c.id}>{c.text}</p>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── Shared controls (non-live) ───────────────────────── */}
        {tab !== "live" && (
          <>
            <div>
              <label className="block text-[11px] font-medium text-th-text-tertiary mb-1.5">
                Question (optional)
              </label>
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="e.g. Summarise the key steps, or find where the error appears."
                rows={2}
                className="w-full px-3 py-2 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-400 text-sm resize-none"
              />
            </div>

            {(tab === "screen") && (
              <div>
                <label className="block text-[11px] font-medium text-th-text-tertiary mb-1.5">
                  Recording frame rate
                </label>
                <div className="flex gap-1.5">
                  {FPS_OPTIONS.map((f) => (
                    <button
                      key={f}
                      type="button"
                      onClick={() => setFps(f)}
                      disabled={recording}
                      className={`flex-1 py-1.5 rounded-lg text-[12px] font-medium transition-all disabled:opacity-40 ${
                        fps === f
                          ? "bg-th-surface text-th-text-primary border border-th-border"
                          : "text-th-text-tertiary hover:bg-th-surface-hover"
                      }`}
                    >
                      {f} fps
                    </button>
                  ))}
                </div>
              </div>
            )}

            <div className="flex gap-2">
              <button
                type="button"
                onClick={handleAnalyze}
                disabled={analyzing || creatingSession || !currentSource()}
                className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl bg-sky-500/90 hover:bg-sky-500 text-white text-[13px] font-medium transition-all disabled:opacity-40 disabled:cursor-not-allowed"
              >
                {analyzing || creatingSession ? <Loader2 size={14} className="animate-spin" /> : <Video size={14} />}
                Analyse
              </button>
              <button
                type="button"
                onClick={handleSendToChat}
                disabled={!currentSource()}
                className="flex items-center justify-center gap-2 px-3 py-2.5 rounded-xl bg-th-surface border border-th-border hover:bg-th-surface-hover text-th-text-primary text-[13px] font-medium transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                title="Send to chat so the agent watches it"
              >
                <Send size={14} />
              </button>
            </div>

            {error && (
              <p className="text-[11px] text-red-400 flex items-start gap-1.5">
                <AlertTriangle size={12} className="shrink-0 mt-0.5" /> {error}
              </p>
            )}

            {result && (
              <div className="space-y-1.5">
                <span className="text-[11px] font-medium text-th-text-tertiary">Result</span>
                <div className="text-[12px] text-th-text-secondary leading-relaxed whitespace-pre-wrap bg-th-surface/60 border border-th-border rounded-lg p-3 max-h-[320px] overflow-y-auto">
                  {result}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </aside>
  );
}
