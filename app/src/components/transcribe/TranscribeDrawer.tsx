import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import JSZip from "jszip";
import {
  X,
  Circle,
  Square,
  Trash2,
  Send,
  ChevronDown,
  AlertTriangle,
  Volume2,
  Mic,
  Sparkles,
  Check,
  Camera,
  Monitor,
  AppWindow,
  MonitorUp,
  Loader2,
  Radio,
  Clock,
  Maximize2,
  Minimize2,
  FolderArchive,
} from "lucide-react";
import { api } from "../../hooks/useApi";
import { useTranscribe } from "../../hooks/useTranscribe";
import { emitAskOtto } from "../../utils/askOttoBus";
import { Popover } from "../ui/Popover";
import WindowPicker from "./WindowPicker";
import type {
  CaptureWindow,
  LoopbackStatus,
  Shot,
  TranscribeSource,
  TranscriptSegment,
} from "../../types";

const SCREEN_SETTINGS_URL =
  "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture";

function openScreenSettings() {
  import("@tauri-apps/plugin-shell")
    .then(({ open }) => open(SCREEN_SETTINGS_URL))
    .catch(() => {
      /* best effort — outside Tauri this simply no-ops */
    });
}

/** A transcript segment or a captured screenshot, unified for the feed. */
type FeedItem =
  | { kind: "seg"; id: string; ts: number; seg: TranscriptSegment }
  | { kind: "shot"; id: string; ts: number; shot: Shot };

/** What subsequent screenshots reference: the whole desktop or a window. */
type CaptureTarget =
  | { mode: "desktop" }
  | { mode: "window"; window_id: number; app: string; title: string };

const DESKTOP_TARGET: CaptureTarget = { mode: "desktop" };

function targetLabel(t: CaptureTarget): string {
  if (t.mode === "desktop") return "Desktop";
  return t.title ? `${t.app} · ${t.title}` : t.app;
}

// Interval choices for periodic screen snapshots (seconds; 0 = off).
const INTERVAL_OPTIONS: { secs: number; label: string }[] = [
  { secs: 0, label: "Off" },
  { secs: 3, label: "Every 3s" },
  { secs: 5, label: "Every 5s" },
  { secs: 10, label: "Every 10s" },
  { secs: 30, label: "Every 30s" },
  { secs: 60, label: "Every 1m" },
  { secs: 300, label: "Every 5m" },
];

function intervalLabel(secs: number): string {
  return INTERVAL_OPTIONS.find((o) => o.secs === secs)?.label ?? `Every ${secs}s`;
}

// Panel width: drag-resizable between these bounds, plus a one-click "expand"
// preset for reviewing screenshots/long transcripts more comfortably.
const MIN_PANEL_WIDTH = 320;
const MAX_PANEL_WIDTH = 900;
const DEFAULT_PANEL_WIDTH = 380;
const VIEWPORT_MARGIN = 240; // keep at least this much room for the rest of the app

interface TranscribeDrawerProps {
  open: boolean;
  onClose: () => void;
}

function fmtElapsed(secs: number): string {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/** iOS-style toggle switch. */
function Switch({
  on,
  onChange,
  disabled,
}: {
  on: boolean;
  onChange: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      disabled={disabled}
      onClick={onChange}
      className={`relative inline-flex h-[22px] w-[38px] shrink-0 items-center rounded-full transition-colors duration-200 disabled:opacity-40 disabled:cursor-not-allowed ${
        on ? "bg-emerald-500" : "bg-th-border-strong"
      }`}
    >
      <span
        className={`inline-block h-[18px] w-[18px] transform rounded-full bg-white shadow-sm transition-transform duration-200 ${
          on ? "translate-x-[18px]" : "translate-x-[2px]"
        }`}
      />
    </button>
  );
}

const SOURCE_META: Record<
  TranscribeSource,
  { label: string; icon: typeof Volume2; chip: string; bar: string }
> = {
  system: {
    label: "System audio",
    icon: Volume2,
    chip: "bg-indigo-500/15 text-indigo-300 border border-indigo-500/25",
    bar: "bg-indigo-500",
  },
  mic: {
    label: "Microphone",
    icon: Mic,
    chip: "bg-emerald-500/15 text-emerald-300 border border-emerald-500/25",
    bar: "bg-emerald-500",
  },
};

// Prepended to the FIRST message of a capture so Otto knows this is passively
// captured context (live audio transcribed on-device and/or screenshots) and
// should ask (with quick options) what to do when my intent is unclear.
const STANDING_INSTRUCTION =
  "I'm sharing context I'm passively capturing: live audio (transcribed " +
  "on-device) and/or screenshots of my screen. If it's clear what I need, just " +
  "help. If it's not obvious, use ask_user to ask me what to do with a short " +
  "list of quick options and always include an \"Other…\" choice so I can type " +
  "something else. For a transcript, good options are e.g. Summarize, Extract " +
  "action items & decisions, Draft a reply, Answer a question about it. For a " +
  "screenshot, good options are e.g. Describe what's on screen, Extract the " +
  "text, Explain this, Answer a question about it, Identify next steps. When " +
  "I've shared both, consider them together.";

function fmtClock(tsSecs: number): string {
  return new Date(tsSecs * 1000).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

/**
 * Render feed items (transcript lines + screenshots) as one chronologically
 * ordered text block, so Otto sees exactly where each screenshot falls
 * relative to what was being said — not just a trailing "N screenshots
 * attached" note.  Screenshot markers are numbered in the same order as the
 * `images` array passed alongside, so Otto can map "Screenshot 2" to the
 * second attached image.
 */
function buildInterleavedBody(items: FeedItem[]): string {
  const multi = new Set(
    items.filter((i): i is Extract<FeedItem, { kind: "seg" }> => i.kind === "seg").map((i) => i.seg.source),
  ).size > 1;
  let shotIdx = 0;
  return items
    .map((item) => {
      if (item.kind === "shot") {
        shotIdx += 1;
        return `[Screenshot ${shotIdx} — ${fmtClock(item.ts)} — ${item.shot.label}]`;
      }
      const s = item.seg;
      return multi ? `[${s.source === "mic" ? "Me" : "System"}] ${s.text}` : s.text;
    })
    .join("\n");
}

export default function TranscribeDrawer({ open, onClose }: TranscribeDrawerProps) {
  const navigate = useNavigate();
  const location = useLocation();
  const t = useTranscribe({ enabled: open });

  const [status, setStatus] = useState<LoopbackStatus | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [elapsed, setElapsed] = useState(0);
  const [atBottom, setAtBottom] = useState(true);

  // Panel width — draggable from the left edge, expandable via the header
  // button, and persisted across sessions.
  const clampWidth = useCallback((w: number) => {
    const viewportMax = typeof window !== "undefined" ? window.innerWidth - VIEWPORT_MARGIN : MAX_PANEL_WIDTH;
    return Math.min(Math.max(w, MIN_PANEL_WIDTH), Math.min(MAX_PANEL_WIDTH, Math.max(MIN_PANEL_WIDTH, viewportMax)));
  }, []);
  const [panelWidth, setPanelWidth] = useState<number>(() => {
    const raw = Number(localStorage.getItem("transcribe.panelWidth"));
    return Number.isFinite(raw) && raw > 0 ? raw : DEFAULT_PANEL_WIDTH;
  });
  useEffect(() => {
    localStorage.setItem("transcribe.panelWidth", String(panelWidth));
  }, [panelWidth]);
  // Re-clamp on mount and whenever the window shrinks, so the panel never
  // crowds out the rest of the app.
  useEffect(() => {
    setPanelWidth((w) => clampWidth(w));
    const onResize = () => setPanelWidth((w) => clampWidth(w));
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [clampWidth]);

  const [isResizing, setIsResizing] = useState(false);
  const startResize = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault();
      const startX = e.clientX;
      const startWidth = panelWidth;
      setIsResizing(true);
      const onMove = (ev: PointerEvent) => {
        // The panel sits on the right edge of the window, so dragging the
        // handle left (negative deltaX) grows it.
        const delta = startX - ev.clientX;
        setPanelWidth(clampWidth(startWidth + delta));
      };
      const onUp = () => {
        setIsResizing(false);
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
      };
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
    },
    [panelWidth, clampWidth],
  );

  // Prevent text selection / show the resize cursor across the whole window
  // while actively dragging, so fast mouse movement doesn't feel jittery.
  useEffect(() => {
    if (!isResizing) return;
    const prevCursor = document.body.style.cursor;
    const prevUserSelect = document.body.style.userSelect;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    return () => {
      document.body.style.cursor = prevCursor;
      document.body.style.userSelect = prevUserSelect;
    };
  }, [isResizing]);

  const isExpanded = panelWidth > DEFAULT_PANEL_WIDTH + 40;
  const toggleExpanded = useCallback(() => {
    setPanelWidth((w) =>
      w > DEFAULT_PANEL_WIDTH + 40
        ? DEFAULT_PANEL_WIDTH
        : clampWidth(Math.round((typeof window !== "undefined" ? window.innerWidth : 1200) * 0.55)),
    );
  }, [clampWidth]);

  // Which sources to capture — persisted between sessions.
  const [wantSystem, setWantSystem] = useState(
    () => (localStorage.getItem("transcribe.system") ?? "true") === "true",
  );
  const [wantMic, setWantMic] = useState(
    () => localStorage.getItem("transcribe.mic") === "true",
  );
  useEffect(() => { localStorage.setItem("transcribe.system", String(wantSystem)); }, [wantSystem]);
  useEffect(() => { localStorage.setItem("transcribe.mic", String(wantMic)); }, [wantMic]);

  // Hands-free auto-send — on by default, persisted like the source toggles.
  const [autoSend, setAutoSend] = useState(
    () => localStorage.getItem("transcribe.autosend") !== "false",
  );
  useEffect(() => { localStorage.setItem("transcribe.autosend", String(autoSend)); }, [autoSend]);

  // Which segments have already been handed to Otto (incremental send).
  const [sentIds, setSentIds] = useState<Set<string>>(() => new Set());
  // Once true, later sends skip the standing instruction (context already set).
  const firstSendDoneRef = useRef(false);

  // Pause auto-send while Otto is streaming or awaiting an ask_user answer.
  const [agentBusy, setAgentBusy] = useState(false);
  useEffect(() => {
    const h = (e: Event) => setAgentBusy(!!(e as CustomEvent).detail);
    window.addEventListener("otto:agent-busy", h);
    return () => window.removeEventListener("otto:agent-busy", h);
  }, []);

  // ── Screenshots ──────────────────────────────────────────────────
  const [shots, setShots] = useState<Shot[]>([]);
  const [lightboxShot, setLightboxShot] = useState<Shot | null>(null);
  const [captureBusy, setCaptureBusy] = useState(false);
  // "screen-permission" | "window-gone" | "unsupported" | null
  const [captureError, setCaptureError] = useState<string | null>(null);
  // Window picker (choosing a window as the capture target).
  const [pickerOpen, setPickerOpen] = useState(false);

  // The capture target: subsequent screenshots (manual + auto) reference this.
  // Defaults to the whole desktop; switch to a window via the target selector.
  const [captureTarget, setCaptureTarget] = useState<CaptureTarget>(() => {
    try {
      const raw = localStorage.getItem("transcribe.captureTarget");
      return raw ? (JSON.parse(raw) as CaptureTarget) : DESKTOP_TARGET;
    } catch {
      return DESKTOP_TARGET;
    }
  });
  useEffect(() => {
    localStorage.setItem("transcribe.captureTarget", JSON.stringify(captureTarget));
  }, [captureTarget]);
  // Mirror in a ref so the stable capture callbacks read the latest target.
  const captureTargetRef = useRef(captureTarget);
  captureTargetRef.current = captureTarget;

  // Transcript-anchored auto-capture: grab one (deduped) shot of the target
  // whenever a transcript batch auto-sends.
  const [includeScreen, setIncludeScreen] = useState(
    () => localStorage.getItem("transcribe.includeScreen") === "true",
  );
  useEffect(() => {
    localStorage.setItem("transcribe.includeScreen", String(includeScreen));
  }, [includeScreen]);

  // Optional interval capture (0 = off): in addition to the transcript-
  // anchored shot, also grab a (deduped) shot every N seconds while recording.
  // Useful when audio is sparse/absent but the screen still matters.
  const [screenIntervalSecs, setScreenIntervalSecs] = useState<number>(() => {
    const raw = Number(localStorage.getItem("transcribe.screenIntervalSecs"));
    return Number.isFinite(raw) && raw > 0 ? raw : 0;
  });
  useEffect(() => {
    localStorage.setItem("transcribe.screenIntervalSecs", String(screenIntervalSecs));
  }, [screenIntervalSecs]);

  // Last perceptual hash of the target, for dedupe across auto-shots.
  const lastHashRef = useRef<string | null>(null);

  const feedRef = useRef<HTMLDivElement | null>(null);
  const startedAtRef = useRef<number | null>(null);

  const recording = t.state === "recording";

  useEffect(() => {
    if (!open) return;
    api.loopbackStatus().then(setStatus).catch(() => setStatus(null));
  }, [open]);

  useEffect(() => {
    if (!recording) {
      startedAtRef.current = null;
      setElapsed(0);
      return;
    }
    startedAtRef.current = Date.now();
    const id = setInterval(() => {
      if (startedAtRef.current) setElapsed((Date.now() - startedAtRef.current) / 1000);
    }, 500);
    return () => clearInterval(id);
  }, [recording]);

  useEffect(() => {
    if (atBottom && feedRef.current) {
      feedRef.current.scrollTop = feedRef.current.scrollHeight;
    }
  }, [t.segments, t.partials, shots, atBottom]);

  const onScroll = useCallback(() => {
    const el = feedRef.current;
    if (!el) return;
    setAtBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 40);
  }, []);

  const toggleSelect = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  // Remove a single feed item (transcribed line or screenshot). Local-only —
  // doesn't affect anything already sent to Otto — and cleans up any
  // selection/sent bookkeeping so removed ids don't linger.
  const removeItem = useCallback(
    (item: FeedItem) => {
      if (item.kind === "seg") t.removeSegment(item.id);
      else setShots((prev) => prev.filter((s) => s.id !== item.id));
      setSelected((prev) => {
        if (!prev.has(item.id)) return prev;
        const next = new Set(prev);
        next.delete(item.id);
        return next;
      });
      setSentIds((prev) => {
        if (!prev.has(item.id)) return prev;
        const next = new Set(prev);
        next.delete(item.id);
        return next;
      });
    },
    [t],
  );

  // Transcript segments + screenshots interleaved chronologically.
  const feed = useMemo<FeedItem[]>(() => {
    const items: FeedItem[] = [
      ...t.segments.map((s) => ({ kind: "seg" as const, id: s.id, ts: s.ts, seg: s })),
      ...shots.map((sh) => ({ kind: "shot" as const, id: sh.id, ts: sh.ts, shot: sh })),
    ];
    items.sort((a, b) => a.ts - b.ts);
    return items;
  }, [t.segments, shots]);

  const allSelected = feed.length > 0 && selected.size === feed.length;
  const toggleSelectAll = () => {
    if (allSelected) setSelected(new Set());
    else setSelected(new Set(feed.map((f) => f.id)));
  };

  // Items not yet handed to Otto — used by "Select unsent" so the footer send
  // button can target just what's new without hand-picking each line/shot.
  const unsentFeedIds = useMemo(
    () => feed.filter((f) => !sentIds.has(f.id)).map((f) => f.id),
    [feed, sentIds],
  );
  const isUnsentSelected =
    unsentFeedIds.length > 0 &&
    selected.size === unsentFeedIds.length &&
    unsentFeedIds.every((id) => selected.has(id));
  const selectUnsent = () => {
    if (isUnsentSelected) setSelected(new Set());
    else setSelected(new Set(unsentFeedIds));
  };

  const systemSupported = (status?.supported ?? true) && (status?.helper_available ?? true);
  const micSupported = status?.mic_available ?? true;

  const activeSources = useMemo<TranscribeSource[]>(() => {
    const arr: TranscribeSource[] = [];
    if (wantSystem && systemSupported) arr.push("system");
    if (wantMic && micSupported) arr.push("mic");
    return arr;
  }, [wantSystem, wantMic, systemSupported, micSupported]);

  const handleStart = () => {
    if (activeSources.length === 0) return;
    // Fresh capture — reset incremental send state.
    setSentIds(new Set());
    firstSendDoneRef.current = false;
    lastHashRef.current = null;
    t.start(activeSources);
  };

  const handleClear = () => {
    setSentIds(new Set());
    firstSendDoneRef.current = false;
    lastHashRef.current = null;
    setShots([]);
    setSelected(new Set());
    t.clear();
  };

  // Hand a set of feed items (transcript + screenshots) to Otto: build one
  // chronologically interleaved text block (so screenshots land at the exact
  // point they were taken, not just a trailing count), mark them sent, ensure
  // ChatPage is mounted, and route through the bus so they append to ONE
  // conversation.
  const sendItems = useCallback(
    (rawItems: FeedItem[]) => {
      if (rawItems.length === 0) return;
      // Defensive re-sort — callers already pass chronological order, but this
      // guarantees the interleaved body and the attached image order line up.
      const items = [...rawItems].sort((a, b) => a.ts - b.ts);
      const shotItems = items
        .filter((i): i is Extract<FeedItem, { kind: "shot" }> => i.kind === "shot")
        .map((i) => i.shot);
      const hasSegs = items.some((i) => i.kind === "seg");
      const first = !firstSendDoneRef.current;
      const leadIn = first ? `${STANDING_INSTRUCTION}\n\n` : "";

      let text: string;
      if (hasSegs) {
        const body = buildInterleavedBody(items);
        const note =
          shotItems.length > 0
            ? " — screenshots are numbered inline at the point they were taken, and attached in that same order"
            : "";
        text = `${leadIn}${
          first ? "Here's what I've captured so far" : "More of what I've captured"
        }${note}:\n\n"""\n${body}\n"""`;
      } else {
        // Screenshot(s) only — no transcript context to interleave with.
        text = `${leadIn}Here ${
          shotItems.length > 1 ? "are screenshots" : "is a screenshot"
        } of what I'm looking at${shotItems.length > 1 ? ", in order" : ""}.`;
      }
      firstSendDoneRef.current = true;

      const images = shotItems.map((s) => ({ name: `${s.id}.png`, dataUrl: s.dataUrl }));

      setSentIds((prev) => {
        const next = new Set(prev);
        for (const i of items) next.add(i.id);
        return next;
      });
      if (!location.pathname.startsWith("/chat")) navigate("/chat");
      emitAskOtto(text, images);
    },
    [location.pathname, navigate],
  );

  const askOtto = (onlySelected: boolean) => {
    const chosen = onlySelected ? feed.filter((f) => selected.has(f.id)) : feed;
    sendItems(chosen);
    setSelected(new Set());
  };

  // ── Save as .zip ─────────────────────────────────────────────────
  const [savingZip, setSavingZip] = useState(false);
  // Toast confirming a completed download, with (best-effort) links to open
  // the file or its folder — null when hidden.
  const [downloadToast, setDownloadToast] = useState<{
    filename: string;
    fullPath: string | null;
    dir: string | null;
  } | null>(null);
  const downloadToastTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const dismissDownloadToast = useCallback(() => {
    if (downloadToastTimerRef.current) clearTimeout(downloadToastTimerRef.current);
    setDownloadToast(null);
  }, []);

  useEffect(() => () => {
    if (downloadToastTimerRef.current) clearTimeout(downloadToastTimerRef.current);
  }, []);

  const openPath = useCallback((path: string) => {
    import("@tauri-apps/plugin-shell")
      .then(({ open }) => open(path))
      .catch(() => {
        /* not running inside Tauri (e.g. plain browser dev) — no-op */
      });
  }, []);

  // Bundle the transcript + screenshots (selected, or everything if nothing
  // is selected) into a single .zip and hand it to the browser/OS download —
  // handy for archiving a capture outside of Otto. Once saved, surfaces a
  // toast naming the file with quick links to open it / its folder.
  const saveZip = useCallback(
    async (onlySelected: boolean) => {
      const chosen = onlySelected ? feed.filter((f) => selected.has(f.id)) : feed;
      if (chosen.length === 0) return;
      const items = [...chosen].sort((a, b) => a.ts - b.ts);
      setSavingZip(true);
      try {
        const zip = new JSZip();
        const hasSegs = items.some((i) => i.kind === "seg");
        if (hasSegs) zip.file("transcript.txt", buildInterleavedBody(items));

        const shotItems = items.filter(
          (i): i is Extract<FeedItem, { kind: "shot" }> => i.kind === "shot",
        );
        await Promise.all(
          shotItems.map(async (item, idx) => {
            const res = await fetch(item.shot.dataUrl);
            const blob = await res.blob();
            const ext = blob.type === "image/jpeg" ? "jpg" : "png";
            const stamp = new Date(item.ts * 1000).toISOString().replace(/[:.]/g, "-");
            zip.file(`screenshot-${String(idx + 1).padStart(2, "0")}-${stamp}.${ext}`, blob);
          }),
        );

        const blob = await zip.generateAsync({ type: "blob" });
        const filename = `otto-capture-${new Date().toISOString().replace(/[:.]/g, "-")}.zip`;
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);

        // Best-effort: the webview saves silently to the OS Downloads folder,
        // so this is *usually* where it landed — not guaranteed if the user
        // has changed their default download location.
        let dir: string | null = null;
        try {
          const { downloadDir } = await import("@tauri-apps/api/path");
          dir = await downloadDir();
        } catch {
          /* not running inside Tauri, or path resolution unavailable */
        }
        if (downloadToastTimerRef.current) clearTimeout(downloadToastTimerRef.current);
        setDownloadToast({ filename, fullPath: dir ? `${dir}/${filename}` : null, dir });
        downloadToastTimerRef.current = setTimeout(dismissDownloadToast, 10000);
      } finally {
        setSavingZip(false);
      }
    },
    [feed, selected, dismissDownloadToast],
  );

  // Build a Shot from a capture result, appending it to the feed. Returns the
  // Shot, or null (surfacing a friendly error) when the capture didn't yield an
  // image.  `dedupe` skips identical frames (used by auto-capture only).
  const shotFromTarget = useCallback(
    async (dedupe: boolean): Promise<Shot | null> => {
      const target = captureTargetRef.current;
      try {
        const r =
          target.mode === "window"
            ? await api.captureScreen(
                "window",
                target.window_id,
                dedupe ? lastHashRef.current ?? undefined : undefined,
              )
            : await api.captureScreen(
                "desktop",
                undefined,
                dedupe ? lastHashRef.current ?? undefined : undefined,
              );
        if (r.unchanged) {
          if (r.hash) lastHashRef.current = r.hash;
          return null;
        }
        if (r.needs_permission) {
          setCaptureError("screen-permission");
          return null;
        }
        if (r.window_gone) {
          // Followed window vanished — fall back to the desktop.
          setCaptureError("window-gone");
          setCaptureTarget(DESKTOP_TARGET);
          lastHashRef.current = null;
          return null;
        }
        if (r.unsupported || !r.image_b64) {
          if (r.unsupported) setCaptureError("unsupported");
          return null;
        }
        if (r.hash) lastHashRef.current = r.hash;
        const shot: Shot = {
          id: `shot-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
          dataUrl: `data:${r.mime_type ?? "image/png"};base64,${r.image_b64}`,
          label: targetLabel(target),
          ts: Date.now() / 1000,
          kind: "shot",
        };
        setShots((prev) => [...prev, shot]);
        setCaptureError(null);
        return shot;
      } catch {
        setCaptureError("unsupported");
        return null;
      }
    },
    [],
  );

  // Manual one-click screenshot of the current target (never deduped).
  const captureNow = useCallback(async () => {
    setCaptureBusy(true);
    try {
      await shotFromTarget(false);
    } finally {
      setCaptureBusy(false);
    }
  }, [shotFromTarget]);

  // Interval capture: in addition to the transcript-anchored shot, grab a
  // (deduped) shot of the target every `screenIntervalSecs` while recording.
  // Shares the same dedupe hash as the anchored capture, so back-to-back
  // unchanged frames from either trigger are still collapsed to one shot.
  useEffect(() => {
    if (!includeScreen || !recording || screenIntervalSecs <= 0) return;
    const id = setInterval(() => {
      shotFromTarget(true);
    }, screenIntervalSecs * 1000);
    return () => clearInterval(id);
  }, [includeScreen, recording, screenIntervalSecs, shotFromTarget]);

  // Feed items not yet handed to Otto (stable identity unless inputs change).
  const unsent = useMemo(
    () => feed.filter((f) => !sentIds.has(f.id)),
    [feed, sentIds],
  );

  const autoSilenceSecs = status?.config.loopback_auto_send_silence_secs ?? 2.5;

  // Silence-debounced auto-send: whenever transcript activity settles for
  // `autoSilenceSecs`, optionally grab a (deduped) screenshot of the target and
  // flush the unsent items.  Any new segment/partial or a busy agent re-runs
  // this effect and resets/cancels the timer.
  useEffect(() => {
    if (!autoSend || !recording || agentBusy || unsent.length === 0) return;
    let cancelled = false;
    const id = setTimeout(async () => {
      let items = unsent;
      if (includeScreen) {
        const shot = await shotFromTarget(true);
        if (cancelled) return;
        if (shot) {
          items = [...items, { kind: "shot", id: shot.id, ts: shot.ts, shot }];
        }
      }
      if (!cancelled) sendItems(items);
    }, Math.max(500, autoSilenceSecs * 1000));
    return () => {
      cancelled = true;
      clearTimeout(id);
    };
  }, [
    autoSend,
    recording,
    agentBusy,
    unsent,
    t.partials,
    autoSilenceSecs,
    sendItems,
    includeScreen,
    shotFromTarget,
  ]);

  // ── Capture target selection ─────────────────────────────────────
  const chooseDesktopTarget = useCallback(() => {
    setCaptureTarget(DESKTOP_TARGET);
    lastHashRef.current = null;
    setCaptureError(null);
  }, []);

  const chooseWindowTarget = useCallback((win: CaptureWindow) => {
    setPickerOpen(false);
    setCaptureTarget({ mode: "window", window_id: win.window_id, app: win.app, title: win.title });
    lastHashRef.current = null;
    setCaptureError(null);
  }, []);

  const hasSelection = selected.size > 0;
  const canSend = feed.length > 0;
  const firstUnsentIdx = feed.findIndex((f) => !sentIds.has(f.id));

  if (!open) return null;

  return (
    <>
    <aside
      style={{ width: panelWidth }}
      className={`relative shrink-0 h-full flex flex-col bg-th-bg/70 backdrop-blur-xl border-l border-th-border shadow-[-8px_0_24px_-8px_rgba(0,0,0,0.08)] ${
        isResizing ? "" : "transition-[width] duration-150"
      }`}
    >
      {/* Resize handle — drag to adjust width */}
      <div
        onPointerDown={startResize}
        className="absolute left-0 top-0 -translate-x-1/2 h-full w-3 cursor-col-resize z-20 flex items-center justify-center group"
        title="Drag to resize"
      >
        <div
          className={`h-full w-px transition-colors ${
            isResizing ? "bg-blue-500" : "bg-transparent group-hover:bg-blue-500/50"
          }`}
        />
        <div
          className={`absolute w-[3px] h-9 rounded-full transition-all ${
            isResizing ? "bg-blue-500/70" : "bg-th-border-strong/0 group-hover:bg-th-border-strong/70"
          }`}
        />
      </div>

      {/* Header */}
      <div className="flex items-center justify-between px-4 h-14 border-b border-th-border/70 shrink-0">
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="relative w-8 h-8 rounded-xl bg-gradient-to-br from-indigo-500/20 to-purple-500/10 border border-indigo-500/20 flex items-center justify-center shrink-0 shadow-sm shadow-black/[0.03]">
            <Radio size={15} className="text-indigo-400" />
            {recording && (
              <span className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-red-500 ring-2 ring-th-bg animate-pulse" />
            )}
          </div>
          <div className="min-w-0">
            <h2 className="text-[13px] font-semibold text-th-text-primary leading-tight tracking-tight">Live Capture</h2>
            <p className="text-[11px] text-th-text-muted leading-tight truncate">
              {recording ? "Recording · on-device & private" : "Audio & screen · on-device"}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-0.5 shrink-0">
          <button
            onClick={toggleExpanded}
            className="p-1.5 rounded-lg text-th-text-muted hover:text-th-text-primary hover:bg-th-surface-hover active:scale-95 transition-all"
            title={isExpanded ? "Collapse panel" : "Expand panel"}
          >
            {isExpanded ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
          </button>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-th-text-muted hover:text-th-text-primary hover:bg-th-surface-hover active:scale-95 transition-all"
            title="Close panel"
          >
            <X size={16} />
          </button>
        </div>
      </div>

      {/* Source selection */}
      <div className="px-4 py-3 space-y-1.5 border-b border-th-border/70 shrink-0">
        <p className="text-[10px] uppercase tracking-wider text-th-text-muted font-semibold px-0.5">Sources</p>
        {(["system", "mic"] as TranscribeSource[]).map((src) => {
          const meta = SOURCE_META[src];
          const Icon = meta.icon;
          const supported = src === "system" ? systemSupported : micSupported;
          const on = src === "system" ? wantSystem : wantMic;
          const setOn = src === "system" ? setWantSystem : setWantMic;
          const level = t.levels[src] ?? 0;
          const activeNow = recording && (src === "system" ? wantSystem : wantMic) && supported;
          return (
            <div
              key={src}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-2xl border transition-all duration-200 ${
                on && supported
                  ? "bg-th-surface border-th-border shadow-sm shadow-black/[0.03]"
                  : "bg-th-bg-secondary/40 border-th-border/50"
              }`}
            >
              <div className={`w-8 h-8 rounded-xl flex items-center justify-center shrink-0 ${meta.chip}`}>
                <Icon size={15} />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-[13px] font-medium text-th-text-primary leading-tight">{meta.label}</p>
                {supported ? (
                  <div className="mt-1.5 h-1 rounded-full bg-th-surface-hover overflow-hidden">
                    <div
                      className={`h-full rounded-full ${meta.bar} transition-[width] duration-100`}
                      style={{ width: `${activeNow ? Math.round(Math.min(1, level) * 100) : 0}%` }}
                    />
                  </div>
                ) : (
                  <p className="text-[11px] text-amber-400/90 leading-tight mt-0.5">
                    {src === "system"
                      ? status && !status.supported
                        ? "Requires macOS 14.4+"
                        : "Helper not available"
                      : "No microphone found"}
                  </p>
                )}
              </div>
              <Switch on={on && supported} onChange={() => setOn(!on)} disabled={!supported || recording} />
            </div>
          );
        })}

        {/* Screen — capture target + auto-include toggle */}
        <div
          className={`flex items-center gap-3 px-3 py-2.5 rounded-2xl border overflow-hidden transition-all duration-200 ${
            includeScreen ? "bg-th-surface border-th-border shadow-sm shadow-black/[0.03]" : "bg-th-bg-secondary/40 border-th-border/50"
          }`}
        >
          <div className="w-8 h-8 rounded-xl bg-purple-500/15 text-purple-300 border border-purple-500/25 flex items-center justify-center shrink-0">
            <MonitorUp size={15} />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-[13px] font-medium text-th-text-primary leading-tight">Screen</p>
            <div className="mt-0.5 space-y-1 text-[11px]">
              <div className="min-w-0">
                <Popover
                  align="left"
                  role="menu"
                  panelClassName="w-56 p-1"
                  trigger={({ toggle, "aria-expanded": expanded, "aria-haspopup": haspopup }) => (
                    <button
                      type="button"
                      onClick={toggle}
                      aria-expanded={expanded}
                      aria-haspopup={haspopup}
                      className="flex items-center gap-1 w-full min-w-0 text-th-text-muted hover:text-th-text-primary transition-colors"
                      title="Change what screenshots capture"
                    >
                      {captureTarget.mode === "window" ? (
                        <AppWindow size={11} className="shrink-0" />
                      ) : (
                        <Monitor size={11} className="shrink-0" />
                      )}
                      <span className="truncate min-w-0">{targetLabel(captureTarget)}</span>
                      <ChevronDown size={11} className="shrink-0 opacity-70" />
                    </button>
                  )}
                >
                  {({ close }) => (
                    <div onClick={(e) => e.stopPropagation()}>
                      <button
                        type="button"
                        onClick={() => {
                          close();
                          chooseDesktopTarget();
                        }}
                        className={`flex w-full items-center gap-2.5 px-2.5 py-1.5 rounded-lg text-[12px] transition-colors ${
                          captureTarget.mode === "desktop"
                            ? "text-th-text-primary bg-th-surface-hover"
                            : "text-th-text-secondary hover:bg-th-surface-hover hover:text-th-text-primary"
                        }`}
                      >
                        <Monitor size={14} /> Entire desktop
                        {captureTarget.mode === "desktop" && (
                          <Check size={13} className="ml-auto text-emerald-400" />
                        )}
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          close();
                          setPickerOpen(true);
                        }}
                        className="flex w-full items-center gap-2.5 px-2.5 py-1.5 rounded-lg text-[12px] text-th-text-secondary hover:bg-th-surface-hover hover:text-th-text-primary transition-colors"
                      >
                        <AppWindow size={14} /> Choose window…
                      </button>
                    </div>
                  )}
                </Popover>
              </div>

              {includeScreen && (
                <div className="min-w-0 flex justify-end">
                  <Popover
                    align="right"
                    role="menu"
                    panelClassName="w-40 p-1"
                    trigger={({ toggle, "aria-expanded": expanded, "aria-haspopup": haspopup }) => (
                      <button
                        type="button"
                        onClick={toggle}
                        aria-expanded={expanded}
                        aria-haspopup={haspopup}
                        className="flex items-center gap-1 min-w-0 max-w-full text-th-text-muted hover:text-th-text-primary transition-colors"
                        title="Also snapshot on a timer while recording"
                      >
                        <Clock size={11} className="shrink-0" />
                        <span className="truncate min-w-0">{intervalLabel(screenIntervalSecs)}</span>
                        <ChevronDown size={11} className="shrink-0 opacity-70" />
                      </button>
                    )}
                  >
                    {({ close }) => (
                      <div onClick={(e) => e.stopPropagation()}>
                        {INTERVAL_OPTIONS.map((opt) => (
                          <button
                            key={opt.secs}
                            type="button"
                            onClick={() => {
                              close();
                              setScreenIntervalSecs(opt.secs);
                            }}
                            className={`flex w-full items-center gap-2.5 px-2.5 py-1.5 rounded-lg text-[12px] transition-colors ${
                              screenIntervalSecs === opt.secs
                                ? "text-th-text-primary bg-th-surface-hover"
                                : "text-th-text-secondary hover:bg-th-surface-hover hover:text-th-text-primary"
                            }`}
                          >
                            {opt.label}
                            {screenIntervalSecs === opt.secs && (
                              <Check size={13} className="ml-auto text-emerald-400" />
                            )}
                          </button>
                        ))}
                      </div>
                    )}
                  </Popover>
                </div>
              )}
            </div>
          </div>
          <Switch on={includeScreen} onChange={() => setIncludeScreen((v) => !v)} />
        </div>
      </div>

      {/* Controls */}
      <div className="px-4 py-3 flex items-center gap-2 border-b border-th-border/70 shrink-0">
        {!recording ? (
          <button
            onClick={handleStart}
            disabled={activeSources.length === 0 || !t.connected}
            className="flex items-center gap-2 px-4 py-2 rounded-full bg-gradient-to-b from-red-500 to-red-600 hover:from-red-500 hover:to-red-500 text-white text-[13px] font-semibold shadow-sm shadow-red-500/30 disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none transition-all active:scale-[0.97]"
          >
            <Circle size={11} className="fill-current" /> Record
          </button>
        ) : (
          <div className="flex items-center gap-2">
            <button
              onClick={t.stop}
              className="flex items-center gap-2 px-4 py-2 rounded-full bg-th-surface-hover hover:bg-th-border text-th-text-primary text-[13px] font-semibold border border-th-border transition-all active:scale-[0.97]"
            >
              <Square size={12} className="fill-current" /> Stop
            </button>
            <span className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-full bg-red-500/10 border border-red-500/20 text-[12px] font-medium text-red-400 tabular-nums">
              <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />
              {fmtElapsed(elapsed)}
            </span>
          </div>
        )}
        <div className="flex-1" />

        {/* Manual screenshot — one click captures the current target. */}
        <button
          type="button"
          onClick={captureNow}
          disabled={captureBusy}
          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-full text-[12px] font-medium border bg-th-surface-hover border-th-border text-th-text-secondary hover:text-th-text-primary hover:border-th-border-strong disabled:opacity-40 disabled:cursor-not-allowed transition-all active:scale-[0.96]"
          title={`Take a screenshot of ${targetLabel(captureTarget)}`}
        >
          {captureBusy ? (
            <Loader2 size={13} className="animate-spin" />
          ) : (
            <Camera size={13} />
          )}
          Shot
        </button>

        {!recording && activeSources.length === 0 ? (
          <span className="text-[11px] text-th-text-muted">Enable a source</span>
        ) : (
          <button
            type="button"
            onClick={() => setAutoSend((v) => !v)}
            className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-full text-[12px] font-medium border transition-all active:scale-[0.96] ${
              autoSend
                ? "bg-blue-500/15 border-blue-500/40 text-blue-300 shadow-sm shadow-blue-500/10"
                : "bg-th-surface-hover border-th-border text-th-text-secondary hover:text-th-text-primary hover:border-th-border-strong"
            }`}
            title="Automatically send new transcript to Otto after you pause"
          >
            <Sparkles size={13} className={autoSend ? "text-blue-300" : ""} />
            Auto
          </button>
        )}
      </div>

      {captureError && (
        <div className="mx-4 mt-3 p-2.5 rounded-2xl border border-amber-500/25 bg-amber-500/10 text-[11px] text-amber-300 flex items-start gap-2 shrink-0 animate-fade-in">
          <AlertTriangle size={13} className="shrink-0 mt-0.5" />
          <span className="flex-1">
            {captureError === "screen-permission"
              ? "Otto needs Screen Recording permission to capture your screen. Grant it, then fully quit and reopen Otto."
              : captureError === "window-gone"
                ? "That window is no longer open — switched back to capturing the desktop."
                : "Screen capture isn’t available on this device."}
          </span>
          {captureError === "screen-permission" && (
            <button
              onClick={openScreenSettings}
              className="shrink-0 underline hover:text-amber-200"
            >
              Settings
            </button>
          )}
          <button
            onClick={() => setCaptureError(null)}
            className="shrink-0 text-amber-300/70 hover:text-amber-200"
          >
            <X size={13} />
          </button>
        </div>
      )}

      {autoSend && (
        <div className="px-4 py-2 border-b border-th-border/70 flex items-center gap-2 text-[11px] shrink-0">
          {agentBusy ? (
            <span className="flex items-center gap-1.5 text-th-text-muted">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
              Waiting for Otto to finish…
            </span>
          ) : recording ? (
            <span className="flex items-center gap-1.5 text-th-text-muted">
              <span
                className={`w-1.5 h-1.5 rounded-full ${
                  unsent.length > 0 ? "bg-blue-400 animate-pulse" : "bg-th-border-strong"
                }`}
              />
              {unsent.length > 0
                ? `Auto-sending ${unsent.length} new line${unsent.length > 1 ? "s" : ""} after you pause…`
                : "Auto-send on — sends new lines when you pause"}
            </span>
          ) : (
            <span className="text-th-text-muted">Auto-send on — starts when you record</span>
          )}
        </div>
      )}

      {t.error && (
        <div className="mx-4 mt-3 p-2.5 rounded-2xl border border-red-500/25 bg-red-500/10 text-[11px] text-red-300 flex items-start gap-2 shrink-0 animate-fade-in">
          <AlertTriangle size={13} className="shrink-0 mt-0.5" />
          <span className="flex-1">{t.error}</span>
          <button onClick={t.clearError} className="text-red-300/70 hover:text-red-200">
            <X size={13} />
          </button>
        </div>
      )}

      {/* Feed */}
      <div className="relative flex-1 min-h-0">
        <div
          ref={feedRef}
          onScroll={onScroll}
          className="absolute inset-0 overflow-y-auto px-3 py-3 space-y-1"
        >
          {feed.length === 0 && Object.values(t.partials).every((p) => !p) && (
            <div className="flex flex-col items-center justify-center h-full text-center px-6">
              <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-th-surface-hover to-th-surface flex items-center justify-center mb-3.5 border border-th-border/70 shadow-sm shadow-black/[0.03]">
                {recording ? (
                  <Radio size={18} className="text-indigo-400" />
                ) : (
                  <Volume2 size={18} className="text-th-text-muted" />
                )}
              </div>
              <p className="text-[12px] text-th-text-muted leading-relaxed max-w-[220px]">
                {recording
                  ? "Listening… play audio or speak to see it transcribed here."
                  : "Choose your sources and press Record to transcribe audio in real time."}
              </p>
            </div>
          )}

          {feed.map((item, idx) => {
            const isSel = selected.has(item.id);
            const isSent = sentIds.has(item.id);
            const divider = idx === firstUnsentIdx && firstUnsentIdx > 0 && (
              <div className="flex items-center gap-2 px-2.5 py-2 select-none">
                <div className="flex-1 h-px bg-th-border/70" />
                <span className="flex items-center gap-1 text-[9px] uppercase tracking-wider text-th-text-muted font-semibold">
                  <Check size={9} className="text-emerald-400/80" /> Sent to Otto
                </span>
                <div className="flex-1 h-px bg-th-border/70" />
              </div>
            );

            if (item.kind === "shot") {
              const shot = item.shot;
              return (
                <div key={item.id} className="group relative animate-fade-in">
                  {divider}
                  <div
                    className={`flex flex-col gap-1.5 px-2.5 py-2 pr-16 rounded-2xl transition-all duration-150 ${
                      isSel
                        ? "bg-blue-500/[0.08] ring-2 ring-blue-500/30"
                        : isSent
                          ? "opacity-50 hover:opacity-90 hover:bg-th-surface-hover"
                          : "hover:bg-th-surface-hover"
                    }`}
                  >
                    <button
                      onClick={() => toggleSelect(item.id)}
                      className="flex items-center gap-1.5 self-start max-w-full min-w-0"
                    >
                      <span className="inline-flex items-center gap-1 px-1.5 py-px rounded-full text-[9px] font-semibold bg-purple-500/15 text-purple-300 border border-purple-500/25 max-w-[200px] min-w-0">
                        <Camera size={9} className="shrink-0" />
                        <span className="truncate min-w-0">{shot.label || "Screenshot"}</span>
                      </span>
                      <span className="text-[10px] text-th-text-muted tabular-nums shrink-0">{fmtClock(item.ts)}</span>
                      {isSent && (
                        <span className="flex items-center justify-center w-3.5 h-3.5 rounded-full bg-emerald-500/15">
                          <Check size={9} className="text-emerald-400" />
                        </span>
                      )}
                    </button>
                    <button
                      onClick={() => setLightboxShot(shot)}
                      className="block rounded-xl overflow-hidden border border-th-border hover:border-th-border-strong shadow-sm shadow-black/[0.03] hover:shadow-md hover:shadow-black/[0.06] transition-all"
                      title="Preview"
                    >
                      <img
                        src={shot.dataUrl}
                        alt={shot.label}
                        className="w-full max-h-40 object-cover object-top bg-th-inset-bg"
                      />
                    </button>
                  </div>
                  <div className="absolute top-1.5 right-1.5 flex items-center gap-0.5 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-all">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        sendItems([item]);
                      }}
                      className="p-1.5 rounded-lg text-th-text-muted hover:text-blue-400 hover:bg-th-surface-hover transition-colors"
                      title="Send this screenshot to Otto"
                    >
                      <Send size={13} />
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        removeItem(item);
                      }}
                      className="p-1.5 rounded-lg text-th-text-muted hover:text-red-400 hover:bg-th-surface-hover transition-colors"
                      title="Remove this screenshot"
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                </div>
              );
            }

            const seg = item.seg;
            const meta = SOURCE_META[seg.source];
            return (
              <div key={item.id} className="group relative animate-fade-in">
                {divider}
                <button
                  onClick={() => toggleSelect(item.id)}
                  className={`w-full text-left flex flex-col gap-1 px-2.5 py-2 pr-16 rounded-2xl text-[13px] transition-all duration-150 ${
                    isSel
                      ? "bg-blue-500/[0.08] ring-2 ring-blue-500/30"
                      : isSent
                        ? "opacity-50 hover:opacity-90 hover:bg-th-surface-hover"
                        : "hover:bg-th-surface-hover"
                  }`}
                >
                  <span className="flex items-center gap-1.5 self-start">
                    <span className={`inline-flex items-center gap-1 px-1.5 py-px rounded-full text-[9px] font-semibold ${meta.chip}`}>
                      {seg.source === "mic" ? "Me" : "System"}
                    </span>
                    <span className="text-[10px] text-th-text-muted tabular-nums">{fmtClock(seg.ts)}</span>
                    {isSent && (
                      <span className="flex items-center justify-center w-3.5 h-3.5 rounded-full bg-emerald-500/15">
                        <Check size={9} className="text-emerald-400" />
                      </span>
                    )}
                  </span>
                  <span className="text-th-text-primary leading-snug">{seg.text}</span>
                </button>
                <div className="absolute top-1.5 right-1.5 flex items-center gap-0.5 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-all">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      sendItems([item]);
                    }}
                    className="p-1.5 rounded-lg text-th-text-muted hover:text-blue-400 hover:bg-th-surface-hover transition-colors"
                    title="Send this line to Otto"
                  >
                    <Send size={13} />
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      removeItem(item);
                    }}
                    className="p-1.5 rounded-lg text-th-text-muted hover:text-red-400 hover:bg-th-surface-hover transition-colors"
                    title="Remove this line"
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              </div>
            );
          })}

          {(["system", "mic"] as TranscribeSource[]).map((src) =>
            t.partials[src] ? (
              <div key={`p-${src}`} className="flex flex-col gap-1 px-2.5 py-2 rounded-2xl bg-th-surface-hover/30 text-[13px]">
                <span className={`inline-flex items-center gap-1 self-start px-1.5 py-px rounded-full text-[9px] font-semibold opacity-60 ${SOURCE_META[src].chip}`}>
                  {src === "mic" ? "Me" : "System"}
                </span>
                <span className="text-th-text-muted italic leading-snug">
                  {t.partials[src]}
                  <span className="inline-block w-[2px] h-3 ml-0.5 -mb-0.5 bg-th-text-muted/60 animate-pulse" />
                </span>
              </div>
            ) : null,
          )}
        </div>

        {!atBottom && (
          <button
            onClick={() => {
              setAtBottom(true);
              if (feedRef.current) feedRef.current.scrollTop = feedRef.current.scrollHeight;
            }}
            className="absolute bottom-3 left-1/2 -translate-x-1/2 flex items-center gap-1 px-3 py-1.5 rounded-full bg-th-surface/95 backdrop-blur-md border border-th-border text-[11px] font-medium text-th-text-secondary shadow-lg shadow-black/10 hover:text-th-text-primary hover:border-th-border-strong transition-all animate-fade-in"
          >
            <ChevronDown size={13} /> Jump to latest
          </button>
        )}
      </div>

      {downloadToast && (
        <div className="px-3 pt-3 shrink-0 animate-fade-in">
          <div className="flex items-center gap-2.5 px-3 py-2.5 rounded-2xl border border-emerald-500/25 bg-emerald-500/10">
            <span className="flex items-center justify-center w-7 h-7 rounded-xl bg-emerald-500/15 shrink-0">
              <FolderArchive size={14} className="text-emerald-400" />
            </span>
            <div className="flex-1 min-w-0">
              <p className="text-[12px] font-medium text-th-text-primary leading-tight truncate">
                Saved {downloadToast.filename}
              </p>
              <div className="mt-0.5 flex items-center gap-2.5 text-[11px]">
                {downloadToast.fullPath ? (
                  <button
                    onClick={() => openPath(downloadToast.fullPath!)}
                    className="text-emerald-300 hover:text-emerald-200 font-medium underline-offset-2 hover:underline transition-colors"
                  >
                    Open file
                  </button>
                ) : (
                  <span className="text-th-text-muted">Check your Downloads folder</span>
                )}
                {downloadToast.dir && (
                  <button
                    onClick={() => openPath(downloadToast.dir!)}
                    className="text-th-text-secondary hover:text-th-text-primary font-medium underline-offset-2 hover:underline transition-colors"
                  >
                    Show folder
                  </button>
                )}
              </div>
            </div>
            <button
              onClick={dismissDownloadToast}
              className="p-1 rounded-lg text-th-text-muted hover:text-th-text-primary hover:bg-th-surface-hover transition-colors shrink-0"
            >
              <X size={13} />
            </button>
          </div>
        </div>
      )}

      {/* Footer actions */}
      <div className="px-3 py-3 border-t border-th-border/70 flex flex-col gap-2 shrink-0">
        <div className="flex items-center gap-0.5 flex-wrap">
          <button
            onClick={toggleSelectAll}
            disabled={feed.length === 0}
            className="px-2 py-1 rounded-lg text-[12px] text-th-text-secondary hover:text-th-text-primary hover:bg-th-surface-hover disabled:opacity-40 disabled:hover:bg-transparent transition-colors"
          >
            {allSelected ? "Deselect" : "Select all"}
          </button>
          <button
            onClick={selectUnsent}
            disabled={unsentFeedIds.length === 0}
            className="px-2 py-1 rounded-lg text-[12px] text-th-text-secondary hover:text-th-text-primary hover:bg-th-surface-hover disabled:opacity-40 disabled:hover:bg-transparent transition-colors"
            title="Select everything not yet sent to Otto"
          >
            {isUnsentSelected ? "Deselect" : `Select unsent${unsentFeedIds.length > 0 ? ` · ${unsentFeedIds.length}` : ""}`}
          </button>

          <div className="flex-1" />

          <button
            onClick={() => saveZip(hasSelection)}
            disabled={!canSend || savingZip}
            className="p-1.5 rounded-lg text-th-text-muted hover:text-th-text-primary hover:bg-th-surface-hover disabled:opacity-40 disabled:hover:bg-transparent transition-colors"
            title={hasSelection ? "Save selected as a .zip" : "Save transcript & screenshots as a .zip"}
          >
            {savingZip ? <Loader2 size={14} className="animate-spin" /> : <FolderArchive size={14} />}
          </button>
          <button
            onClick={handleClear}
            disabled={feed.length === 0}
            className="p-1.5 rounded-lg text-th-text-muted hover:text-th-text-primary hover:bg-th-surface-hover disabled:opacity-40 disabled:hover:bg-transparent transition-colors"
            title="Clear transcript and screenshots"
          >
            <Trash2 size={14} />
          </button>
        </div>

        <button
          onClick={() => askOtto(hasSelection)}
          disabled={!canSend}
          className="w-full flex items-center justify-center gap-1.5 px-4 py-2.5 rounded-full bg-gradient-to-b from-blue-500 to-blue-600 hover:from-blue-500 hover:to-blue-500 text-white text-[13px] font-semibold shadow-sm shadow-blue-500/25 disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none transition-all active:scale-[0.98]"
        >
          <Send size={14} />
          {hasSelection ? `Ask Otto · ${selected.size}` : "Ask Otto"}
        </button>
      </div>
    </aside>

      <WindowPicker
        open={pickerOpen}
        title="Choose"
        onPick={chooseWindowTarget}
        onClose={() => setPickerOpen(false)}
        onOpenSettings={openScreenSettings}
      />

      {lightboxShot && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-8 bg-black/60 backdrop-blur-sm animate-fade-in"
          onClick={() => setLightboxShot(null)}
        >
          <div className="relative max-w-5xl max-h-[90vh] animate-pop-in" onClick={(e) => e.stopPropagation()}>
            <img
              src={lightboxShot.dataUrl}
              alt={lightboxShot.label}
              className="max-w-full max-h-[85vh] rounded-2xl shadow-2xl ring-1 ring-white/10 object-contain"
            />
            <div className="mt-4 flex items-center justify-between gap-3 px-3.5 py-2 rounded-full bg-black/40 backdrop-blur-md ring-1 ring-white/10">
              <span className="text-[12px] text-white/80 flex items-center gap-1.5 min-w-0 truncate">
                <Camera size={13} className="shrink-0" />
                <span className="truncate min-w-0">{lightboxShot.label || "Screenshot"}</span>
                <span className="text-white/50 tabular-nums shrink-0">{fmtClock(lightboxShot.ts)}</span>
              </span>
              <div className="flex items-center gap-1.5 shrink-0">
                <button
                  onClick={() => {
                    const shot = lightboxShot;
                    setLightboxShot(null);
                    removeItem({ kind: "shot", id: shot.id, ts: shot.ts, shot });
                  }}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-white/10 hover:bg-white/20 text-white text-[12px] font-semibold transition-all active:scale-[0.96]"
                >
                  <Trash2 size={13} /> Remove
                </button>
                <button
                  onClick={() => {
                    const shot = lightboxShot;
                    setLightboxShot(null);
                    sendItems([{ kind: "shot", id: shot.id, ts: shot.ts, shot }]);
                  }}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-blue-600 hover:bg-blue-500 text-white text-[12px] font-semibold shadow-sm shadow-blue-500/30 transition-all active:scale-[0.96]"
                >
                  <Send size={13} /> Ask Otto
                </button>
                <button
                  onClick={() => setLightboxShot(null)}
                  className="p-1.5 rounded-full bg-white/10 hover:bg-white/20 text-white transition-all active:scale-[0.96]"
                >
                  <X size={16} />
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
