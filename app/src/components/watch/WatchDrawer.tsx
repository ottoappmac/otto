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
  Trash2,
} from "lucide-react";
import { api } from "../../hooks/useApi";
import { useWatch } from "../../hooks/useWatch";
import { emitAskOtto } from "../../utils/askOttoBus";
import type { CapturePermission } from "../../types";

type SourceTab = "file" | "screen" | "youtube" | "live";

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

  // File
  const [file, setFile] = useState<File | null>(null);
  const [uploadedPath, setUploadedPath] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // YouTube
  const [youtubeUrl, setYoutubeUrl] = useState("");

  // Screen recording
  const [permission, setPermission] = useState<CapturePermission | null>(null);
  const [recording, setRecording] = useState(false);
  const [recordedPath, setRecordedPath] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);

  // Analyze
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
      .then((s) => setGeminiConfigured(!!s.llm?.google?.api_key))
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
    }
  }, [open, tab]);

  const handlePickFile = useCallback(async (f: File) => {
    setError(null);
    setResult(null);
    setUploadedPath(null);
    setFile(f);
    if (!sessionId) {
      setError("Open or start a chat first — the video is uploaded into the session.");
      return;
    }
    setUploading(true);
    try {
      const path = `uploads/${f.name}`;
      await api.uploadSessionFile(sessionId, path, f);
      setUploadedPath(`/${path}`);
    } catch (e) {
      setError(`Upload failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setUploading(false);
    }
  }, [sessionId]);

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
    if (!sessionId) {
      setError("Open or start a chat first so results have somewhere to live.");
      return;
    }
    setAnalyzing(true);
    setError(null);
    setResult(null);
    try {
      const res = await api.videoAnalyze({
        session_id: sessionId,
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
  }, [currentSource, sessionId, question]);

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
    if (!sessionId) {
      setError("Open or start a chat first — recordings are saved into the session.");
      return;
    }
    setError(null);
    setRecordedPath(null);
    const res = await api.videoRecordStart(sessionId, fps);
    if (res.error) {
      setError(res.error);
      return;
    }
    setRecording(true);
  }, [sessionId, fps]);

  const handleStopRecording = useCallback(async () => {
    if (!sessionId) return;
    const res = await api.videoRecordStop(sessionId);
    setRecording(false);
    if (res.error) {
      setError(res.error);
      return;
    }
    if (res.virtual_path) setRecordedPath(res.virtual_path);
  }, [sessionId]);

  const handleToggleLive = useCallback(() => {
    if (live.watching) {
      live.stop();
    } else {
      live.clear();
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
          <div className="flex items-start gap-2 text-[11px] text-amber-400 bg-amber-500/10 border border-amber-500/20 rounded-lg p-2.5">
            <AlertTriangle size={13} className="shrink-0 mt-0.5" />
            <span>Open or start a chat to record and analyse videos. You can still send a file/URL to a new chat.</span>
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
                <Loader2 size={11} className="animate-spin" /> Uploading…
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
            {!recording ? (
              <button
                type="button"
                onClick={handleStartRecording}
                disabled={!sessionId}
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
            {recordedPath && (
              <p className="text-[11px] text-emerald-400">
                Recorded — ready to watch.
              </p>
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
              <div className="flex items-start gap-2 text-[11px] text-amber-400 bg-amber-500/10 border border-amber-500/20 rounded-lg p-2.5">
                <AlertTriangle size={13} className="shrink-0 mt-0.5" />
                <span>Realtime watching needs a Gemini API key (Settings → LLM → Frontier → Google Gemini).</span>
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
              disabled={!geminiConfigured}
              className={`w-full flex items-center justify-center gap-2 py-3 rounded-xl text-[13px] font-medium transition-all disabled:opacity-40 disabled:cursor-not-allowed ${
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
                disabled={analyzing || !currentSource() || !sessionId}
                className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl bg-sky-500/90 hover:bg-sky-500 text-white text-[13px] font-medium transition-all disabled:opacity-40 disabled:cursor-not-allowed"
              >
                {analyzing ? <Loader2 size={14} className="animate-spin" /> : <Video size={14} />}
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
