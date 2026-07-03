import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Cpu,
  Download,
  HardDrive,
  Library,
  Loader2,
  RefreshCw,
  Search,
  Sparkles,
  Square,
  XCircle,
  AlertTriangle,
  X,
} from "lucide-react";
import { api } from "../../hooks/useApi";
import { usePolling } from "../../hooks/usePolling";
import type { MlxCapabilities, MlxCatalogRow, MlxDownloadJob } from "../../types";

type ChooserTab = "library" | "discover" | "custom";

// ---------------------------------------------------------------------------
// MLX Model Chooser — rendered in the On-Device tab when there's no
// model selected yet (or when the user wants to add another).
//
// Reads ``GET /api/mlx/catalog`` which returns the curated list merged
// with dynamically discovered mlx-community models, all scored against
// the local machine.  Filters by fit / family / search.  When a download is in flight, polls
// ``GET /api/mlx/download/{job_id}`` every 800 ms and renders a real
// progress bar.
// ---------------------------------------------------------------------------

export interface ModelChooserProps {
  /** Currently-selected text model repo id (highlighted in the list). */
  selectedRepoId?: string;
  /** Optional HF token to send with the download. */
  hfToken?: string;
  /** Optional override for the Hub cache directory. */
  cacheDir?: string;
  /** Called when a download finishes successfully so the parent can
   *  refresh its bookmark / local-models lists. */
  onDownloadComplete?: (repo_id: string, displayName: string) => void;
  /** Called when the user picks a model that's already cached so the
   *  parent can set it as the active text model. */
  onUseCached?: (repo_id: string, displayName: string) => void;
  /** Called when the user clicks "Unload" to free GPU memory. */
  onUnload?: () => void;
  /** Whether an unload is in progress. */
  unloading?: boolean;
  /** Transient message shown after an unload attempt. */
  unloadMsg?: { ok: boolean; text: string } | null;
}

type FitFilter = "comfortable" | "tight" | "all";

const PAGE_SIZE = 20;

function bytesToHuman(n: number): string {
  if (!n || n < 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : 1)} ${units[i]}`;
}

function formatRate(bps: number): string {
  if (!bps || bps <= 0) return "—";
  return `${bytesToHuman(bps)}/s`;
}

function formatEta(seconds: number | null): string {
  if (seconds == null || seconds < 0) return "—";
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  if (m < 60) return `${m}m ${s}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

function FitBadge({ fits }: { fits: MlxCatalogRow["fits"] }) {
  const map: Record<MlxCatalogRow["fits"], { label: string; cls: string; dot: string }> = {
    comfortable: {
      label: "Comfortable",
      cls: "bg-emerald-500/15 text-emerald-500 ring-1 ring-emerald-500/30",
      dot: "bg-emerald-500",
    },
    tight: {
      label: "Tight",
      cls: "bg-amber-500/15 text-amber-500 ring-1 ring-amber-500/30",
      dot: "bg-amber-500",
    },
    over: {
      label: "Won't fit",
      cls: "bg-rose-500/15 text-rose-500 ring-1 ring-rose-500/30",
      dot: "bg-rose-500",
    },
    unknown: {
      label: "Unknown",
      cls: "bg-th-surface text-th-text-muted ring-1 ring-th-border",
      dot: "bg-neutral-400",
    },
  };
  const v = map[fits];
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] font-medium ${v.cls}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${v.dot}`} />
      {v.label}
    </span>
  );
}

function CapabilitiesBar({ caps }: { caps: MlxCapabilities | null }) {
  if (!caps) return null;
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md bg-th-inset-bg border border-th-border text-[11px]">
        <Cpu size={11} className="text-th-text-muted shrink-0" />
        <span className="font-semibold text-th-text-primary">{caps.chip || "Unknown"}</span>
      </span>
      <span className="inline-flex items-center px-2 py-1 rounded-md bg-th-inset-bg border border-th-border text-[11px] text-th-text-secondary">
        {caps.ram_gb.toFixed(0)} GB RAM
      </span>
      {caps.apple_silicon && (
        <span
          className="inline-flex items-center px-2 py-1 rounded-md bg-th-inset-bg border border-th-border text-[11px] text-th-text-secondary"
          title="Approximate GPU wired-memory ceiling for MLX"
        >
          {caps.wired_limit_gb.toFixed(0)} GB GPU ceiling
        </span>
      )}
      <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md bg-th-inset-bg border border-th-border text-[11px] text-th-text-secondary">
        <HardDrive size={11} className="text-th-text-muted shrink-0" />
        {caps.free_disk_gb.toFixed(0)} GB free
      </span>
      {caps.models_cached > 0 && (
        <span className="inline-flex items-center gap-1 px-2 py-1 rounded-md bg-emerald-500/10 border border-emerald-500/20 text-[11px] text-emerald-500 font-medium">
          {caps.models_cached} cached ({caps.models_cached_size_gb.toFixed(1)} GB)
        </span>
      )}
    </div>
  );
}

function DownloadProgress({
  job,
  onCancel,
  onDismiss,
}: {
  job: MlxDownloadJob;
  onCancel: () => void;
  onDismiss: () => void;
}) {
  const pct = job.bytes_total > 0 ? Math.min(100, Math.round((job.bytes_done / job.bytes_total) * 100)) : 0;
  const indeterminate = job.bytes_total === 0 && (job.status === "running" || job.status === "cancelling");
  const cancelling = job.status === "cancelling";
  const terminal = job.status === "done" || job.status === "error" || job.status === "cancelled";

  return (
    <div className="rounded-xl border border-th-border bg-th-surface p-3 space-y-2">
      <div className="flex items-center justify-between gap-2">
        <div className="min-w-0 flex-1">
          <p className="text-xs font-medium text-th-text-primary truncate">
            {job.repo_id}
          </p>
          <p className="text-[11px] text-th-text-muted truncate">
            {job.status === "done"
              ? "Download complete"
              : job.status === "error"
                ? job.message || "Download failed"
                : job.status === "cancelled"
                  ? "Cancelled"
                  : cancelling
                    ? "Cancelling… (finishing current file)"
                    : job.current_file
                      ? `${job.current_file}${job.files_total ? ` · ${job.files_done}/${job.files_total}` : ""}`
                      : "Starting…"}
          </p>
        </div>
          <div className="flex items-center gap-2 shrink-0">
          <button
            type="button"
            className="text-th-text-muted hover:text-rose-400 disabled:opacity-40 disabled:cursor-not-allowed"
            onClick={terminal ? onDismiss : onCancel}
            disabled={cancelling}
            title={terminal ? "Dismiss" : cancelling ? "Cancelling…" : "Cancel download"}
          >
            <X size={14} />
          </button>
        </div>
      </div>

      <div className="relative h-2 rounded-full bg-th-inset-bg overflow-hidden">
        {indeterminate ? (
          <div className="absolute inset-y-0 w-1/3 bg-th-tab-active-bg/60 animate-pulse" />
        ) : (
          <div
            className={`absolute inset-y-0 left-0 transition-all duration-300 ${
              job.status === "error"
                ? "bg-rose-500"
                : job.status === "cancelled"
                  ? "bg-th-text-muted"
                  : job.status === "done"
                    ? "bg-emerald-500"
                    : "bg-th-tab-active-bg"
            }`}
            style={{ width: `${pct}%` }}
          />
        )}
      </div>

      <div className="flex items-center justify-between text-[11px] text-th-text-muted font-mono">
        <span>
          {bytesToHuman(job.bytes_done)} / {job.bytes_total > 0 ? bytesToHuman(job.bytes_total) : "?"}
          {!indeterminate && job.bytes_total > 0 && <span className="ml-1">· {pct}%</span>}
        </span>
        <span>
          {job.status === "running" && (
            <>
              {formatRate(job.rate_bps)} · ETA {formatEta(job.eta_seconds)}
              {job.use_hf_transfer && <span className="ml-2 text-th-text-tertiary">⚡ hf_transfer</span>}
            </>
          )}
        </span>
      </div>
    </div>
  );
}

function ModelRow({
  row,
  isSelected,
  activeJob,
  ctxLen,
  onDownload,
  onUseCached,
}: {
  row: MlxCatalogRow;
  isSelected: boolean;
  activeJob?: MlxDownloadJob;
  ctxLen: number;
  onDownload: () => void;
  onUseCached: () => void;
}) {
  const breakdownTitle = useMemo(
    () =>
      [
        `Weights: ${row.weights_gb.toFixed(1)} GB`,
        `KV cache @ ${ctxLen.toLocaleString()}t ctx: ${row.kv_gb.toFixed(1)} GB`,
        `Framework + activations: ${row.overhead_gb.toFixed(1)} GB`,
        `≈ Total runtime: ${row.total_gb.toFixed(1)} GB`,
        `Headroom vs comfortable budget: ${row.headroom_gb >= 0 ? "+" : ""}${row.headroom_gb.toFixed(1)} GB`,
      ].join("\n"),
    [row, ctxLen],
  );

  const dlBlocked = row.fits === "over" || !row.disk_ok;
  const isCached = row.already_cached && row.cache_complete;
  const isIncomplete = row.already_cached && !row.cache_complete;

  return (
    <div
      className={`relative rounded-xl border transition-all overflow-hidden ${
        isSelected
          ? "border-th-tab-active-bg/60 bg-th-tab-active-bg/5 ring-1 ring-th-tab-active-bg/20"
          : row.featured
            ? "border-amber-500/25 bg-th-surface hover:border-amber-500/40"
            : "border-th-border bg-th-surface hover:border-th-border-strong"
      }`}
    >
      {/* Featured left accent bar */}
      {row.featured && !isSelected && (
        <div className="absolute left-0 inset-y-0 w-0.5 bg-gradient-to-b from-amber-400/80 via-amber-500/60 to-amber-400/30" />
      )}
      {isSelected && (
        <div className="absolute left-0 inset-y-0 w-0.5 bg-th-tab-active-bg/80" />
      )}

      <div className="pl-4 pr-3.5 py-3.5">
        {/* Row 1: name + status badges | fit badge */}
        <div className="flex items-start justify-between gap-2 mb-1.5">
          <div className="flex items-center gap-1.5 flex-wrap min-w-0">
            <span className="text-sm font-semibold text-th-text-primary leading-snug">
              {row.display_name}
            </span>
            {row.featured && (
              <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-full bg-amber-500/15 text-amber-500 text-[10px] font-semibold ring-1 ring-amber-500/25">
                <Sparkles size={9} />
                featured
              </span>
            )}
            {isCached && (
              <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-full bg-emerald-500/15 text-emerald-500 text-[10px] font-semibold ring-1 ring-emerald-500/25">
                <CheckCircle2 size={9} />
                cached
              </span>
            )}
            {isIncomplete && (
              <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-full bg-amber-500/15 text-amber-500 text-[10px] font-semibold ring-1 ring-amber-500/25">
                <AlertTriangle size={9} />
                incomplete
              </span>
            )}
          </div>
          <FitBadge fits={row.fits} />
        </div>

        {/* Row 2: blurb */}
        {row.blurb && (
          <p className="text-[11px] text-th-text-tertiary line-clamp-1 leading-relaxed mb-2.5 pr-2">
            {row.blurb}
          </p>
        )}

        {/* Row 3: stats + tags | action */}
        <div className="flex items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 text-[11px] text-th-text-muted min-w-0">
            <span title={breakdownTitle} className="cursor-help font-mono shrink-0">
              {row.weights_gb.toFixed(1)} GB · {row.params_b.toFixed(1)}B · {row.quant}
            </span>
            {row.downloads > 0 && (
              <span className="font-mono shrink-0">{row.downloads.toLocaleString()} ↓</span>
            )}
            {row.capability_tags.slice(0, 3).map((t) => (
              <span
                key={t}
                className="px-1.5 py-0.5 rounded-md bg-th-inset-bg border border-th-border/50 text-th-text-secondary text-[10px] font-medium"
              >
                {t}
              </span>
            ))}
          </div>

          <div className="flex items-center gap-1.5 shrink-0">
            {isCached ? (
              <>
                <button
                  type="button"
                  className="px-3 py-1.5 text-[11px] font-semibold rounded-md bg-emerald-600 text-white hover:bg-emerald-500 transition-colors"
                  onClick={onUseCached}
                >
                  Use this model
                </button>
                <button
                  type="button"
                  className="p-1.5 text-th-text-muted hover:text-th-text-secondary rounded-md hover:bg-th-inset-bg transition-colors disabled:opacity-40"
                  onClick={onDownload}
                  disabled={!!activeJob || dlBlocked}
                  title="Re-download"
                >
                  <Download size={12} />
                </button>
              </>
            ) : isIncomplete ? (
              <button
                type="button"
                className="inline-flex items-center gap-1 px-2.5 py-1.5 text-[11px] font-semibold rounded-md bg-amber-500/15 text-amber-500 ring-1 ring-amber-500/30 hover:bg-amber-500/25 disabled:opacity-50 transition-colors"
                onClick={onDownload}
                disabled={!!activeJob || dlBlocked}
                title="Resume the interrupted download"
              >
                <Download size={11} />
                Continue download
              </button>
            ) : activeJob ? (
              <span
                className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-[11px] font-medium rounded-md bg-th-inset-bg text-th-text-tertiary"
                title="Already downloading — see progress card above."
              >
                <Loader2 size={11} className="animate-spin" />
                {activeJob.bytes_total > 0
                  ? `${Math.min(100, Math.round((activeJob.bytes_done / activeJob.bytes_total) * 100))}%`
                  : "Starting…"}
              </span>
            ) : (
              <button
                type="button"
                className="inline-flex items-center gap-1 px-2.5 py-1.5 text-[11px] font-semibold rounded-md bg-th-tab-active-bg text-th-tab-active-fg disabled:opacity-50 hover:opacity-90 transition-opacity"
                onClick={onDownload}
                disabled={dlBlocked}
                title={
                  row.fits === "over"
                    ? `Won't fit — needs ~${row.total_gb.toFixed(1)} GB but ceiling is ${row.ceiling_gb.toFixed(1)} GB`
                    : !row.disk_ok
                      ? "Not enough free disk"
                      : "Download to local Hub cache"
                }
              >
                <Download size={11} />
                Download
              </button>
            )}
          </div>
        </div>

        {/* Warning rows */}
        {!row.disk_ok && (
          <p className="text-[11px] text-rose-400 inline-flex items-center gap-1 mt-2">
            <AlertTriangle size={11} />
            Not enough free disk for {row.weights_gb.toFixed(1)} GB.
          </p>
        )}
        {row.requires_token && (
          <p className="text-[11px] text-amber-400 mt-1.5">Needs an HF token (gated repo).</p>
        )}
      </div>
    </div>
  );
}

export default function ModelChooser({
  selectedRepoId,
  hfToken,
  cacheDir,
  onDownloadComplete,
  onUseCached,
  onUnload,
  unloading,
  unloadMsg,
}: ModelChooserProps) {
  const [tab, setTab] = useState<ChooserTab>("library");

  // ── Your Library tab state ──
  const [localModels, setLocalModels] = useState<{ repo_id: string; name: string; size_mb: number }[] | null>(null);
  const [loadingLocal, setLoadingLocal] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const [librarySearch, setLibrarySearch] = useState("");

  const fetchLocalModels = useCallback(async () => {
    if (localModels !== null) return;
    setLoadingLocal(true);
    setLocalError(null);
    try {
      const res = await api.mlxLocalModels();
      setLocalModels(res.models);
      if (res.error) setLocalError(res.error);
    } catch (e) {
      setLocalError(e instanceof Error ? e.message : String(e));
      setLocalModels([]);
    } finally {
      setLoadingLocal(false);
    }
  }, [localModels]);

  useEffect(() => {
    if (tab === "library") void fetchLocalModels();
  }, [tab, fetchLocalModels]);

  // ── Discover tab state ──
  const [caps, setCaps] = useState<MlxCapabilities | null>(null);
  const [rows, setRows] = useState<MlxCatalogRow[]>([]);
  const [counts, setCounts] = useState<{ comfortable: number; tight: number; over: number; unknown: number } | null>(null);
  const [ctxLen, setCtxLen] = useState<number>(8192);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<FitFilter>("all");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [enriching, setEnriching] = useState(false);
  // Multi-job state.  ``jobs`` is what's currently visible in the
  // progress stack; ``dismissed`` keeps the user's "✕" intent sticky
  // across polls so a dismissed job doesn't bounce back when the
  // backend keeps returning it.
  const [jobs, setJobs] = useState<MlxDownloadJob[]>([]);
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());
  const [advancedRepo, setAdvancedRepo] = useState("");
  const [advancedLabel, setAdvancedLabel] = useState("");

  const refresh = useCallback(
    async (force = false) => {
      setLoading(true);
      setError(null);
      try {
        const r = await api.mlxCatalog({ refresh: force });
        setCaps(r.capabilities);
        setRows(r.models);
        setCounts(r.counts);
        setCtxLen(r.ctx_len ?? 8192);
        setEnriching(r.enriching ?? false);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load catalog");
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  useEffect(() => {
    if (tab === "discover") void refresh(false);
  }, [tab, refresh]);

  // Refs let the polling closure stay stable across re-renders while
  // still seeing the latest props/state.
  const rowsRef = useRef(rows);
  rowsRef.current = rows;
  const onDoneRef = useRef(onDownloadComplete);
  onDoneRef.current = onDownloadComplete;
  const dismissedRef = useRef(dismissed);
  dismissedRef.current = dismissed;
  const refreshRef = useRef(refresh);
  refreshRef.current = refresh;

  // While the backend is discovering models in the background, poll every
  // 4 s.  When enriching transitions from true → false, reload the list
  // so the newly-cached models appear automatically.
  const enrichingRef = useRef(enriching);
  enrichingRef.current = enriching;
  useEffect(() => {
    if (!enriching) return;
    const id = setInterval(async () => {
      try {
        const r = await api.mlxCatalog({});
        const nowEnriching = r.enriching ?? false;
        setEnriching(nowEnriching);
        if (!nowEnriching && enrichingRef.current) {
          // Just finished — reload the full list.
          setCaps(r.capabilities);
          setRows(r.models);
          setCounts(r.counts);
          setCtxLen(r.ctx_len ?? 8192);
        }
      } catch {
        // best-effort poll
      }
    }, 4000);
    return () => clearInterval(id);
  }, [enriching]);

  // ─── Sync with backend job list ─────────────────────────────────
  // The download workers live in the backend, so navigating away
  // from this tab (which unmounts ModelChooser) doesn't actually
  // stop them — only our local job state.  ``GET /api/mlx/downloads``
  // is the source of truth: each poll merges in any active job and
  // any *recently* terminal one (last 60s) so the user sees the
  // "Done" / "Error" state if they come back right after.  Older
  // terminal jobs are dropped to avoid stale ghosts.
  const syncJobs = useCallback(async () => {
    try {
      const r = await api.mlxDownloadList();
      setJobs((prev) => {
        const prevById = new Map(prev.map((j) => [j.job_id, j]));
        const visible = r.jobs.filter((j) => {
          if (dismissedRef.current.has(j.job_id)) return false;
          if (j.status === "running" || j.status === "pending" || j.status === "cancelling") return true;
          if (j.started_at != null && Date.now() / 1000 - j.started_at < 60) return true;
          // Otherwise only keep it if we were already showing it (so
          // a job that's been on screen briefly stays put for the
          // current render rather than vanishing mid-frame).
          return prevById.has(j.job_id);
        });
        // Detect transitions to "done" so we can fire the
        // bookmark-saving callback and refresh catalog cached badges.
        for (const next of visible) {
          const before = prevById.get(next.job_id);
          if (next.status === "done" && before && before.status !== "done") {
            const row = rowsRef.current.find((rr) => rr.repo_id === next.repo_id);
            onDoneRef.current?.(next.repo_id, row?.display_name || next.repo_id);
            void refreshRef.current(false);
          }
        }
        return visible;
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Status polling failed");
    }
  }, []);

  // One-shot sync on mount so re-entering the tab immediately picks
  // up any in-flight download.
  useEffect(() => {
    void syncJobs();
  }, [syncJobs]);

  // Poll while there's anything to track.  When the stack empties
  // (all jobs dismissed or aged-out), we go quiet until the user
  // starts another download.
  const hasActive = jobs.some((j) => j.status === "running" || j.status === "pending");
  usePolling(syncJobs, 800, hasActive || jobs.length > 0);

  const filtered = useMemo(() => {
    const s = search.trim().toLowerCase();
    return rows.filter((r) => {
      if (filter === "comfortable" && r.fits !== "comfortable") return false;
      if (filter === "tight" && !(r.fits === "comfortable" || r.fits === "tight")) return false;
      if (s) {
        const hay = `${r.repo_id} ${r.display_name} ${r.family} ${r.capability_tags.join(" ")}`.toLowerCase();
        if (!hay.includes(s)) return false;
      }
      return true;
    });
  }, [rows, filter, search]);

  // Reset to first page whenever the visible set changes.
  useEffect(() => {
    setPage(1);
  }, [filter, search]);

  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const safePage = Math.min(page, pageCount);
  const paginated = filtered.slice((safePage - 1) * PAGE_SIZE, safePage * PAGE_SIZE);

  const startDownload = useCallback(
    async (repo_id: string) => {
      setError(null);
      try {
        const body: Record<string, unknown> = { repo_id, safetensors_only: true, max_workers: 16, use_hf_transfer: true };
        if (hfToken && hfToken.trim()) body.hf_token = hfToken.trim();
        if (cacheDir && cacheDir.trim()) body.cache_dir = cacheDir.trim();
        const r = await api.mlxDownload(body);
        const optimistic: MlxDownloadJob = {
          job_id: r.job_id,
          status: "pending",
          message: "",
          repo_id: r.repo_id,
          hub_cache: r.hub_cache,
          started_at: Date.now() / 1000,
          bytes_done: 0,
          bytes_total: 0,
          files_done: 0,
          files_total: 0,
          current_file: "",
          current_file_total: 0,
          rate_bps: 0,
          eta_seconds: null,
          use_hf_transfer: true,
          max_workers: 16,
        };
        // Append; the next sync will replace this with the real job.
        // Un-dismiss in case a previous run for the same repo was
        // dismissed earlier.
        setDismissed((prev) => {
          if (!prev.has(r.job_id)) return prev;
          const next = new Set(prev);
          next.delete(r.job_id);
          return next;
        });
        setJobs((prev) => {
          const without = prev.filter((j) => j.job_id !== optimistic.job_id);
          return [optimistic, ...without];
        });
      } catch (e) {
        setError(e instanceof Error ? e.message : "Download failed to start");
      }
    },
    [hfToken, cacheDir],
  );

  const cancelJob = useCallback(async (jobId: string) => {
    // If this job is already "cancelling" (another window already sent the
    // request), just dismiss the card locally — don't send another cancel.
    const alreadyCancelling = jobs.some(
      (j) => j.job_id === jobId && j.status === "cancelling",
    );
    // Dismiss the card immediately — the user clicked X and it should go
    // away.  The dismissed set prevents it re-appearing on the next poll even
    // while the backend is still finishing the current Xet/HTTP chunk.
    setDismissed((prev) => {
      const next = new Set(prev);
      next.add(jobId);
      return next;
    });
    setJobs((prev) => prev.filter((j) => j.job_id !== jobId));
    if (alreadyCancelling) return;
    try {
      await api.mlxDownloadCancel(jobId);
    } catch {
      // best-effort; the worker will set the status itself
    }
  }, [jobs]);

  const dismissJob = useCallback((jobId: string) => {
    setDismissed((prev) => {
      const next = new Set(prev);
      next.add(jobId);
      return next;
    });
    setJobs((prev) => prev.filter((j) => j.job_id !== jobId));
  }, []);

  // O(N) but jobs is bounded; a Map would be overkill.
  const activeJobsByRepo = useMemo(() => {
    const m = new Map<string, MlxDownloadJob>();
    for (const j of jobs) {
      if (j.status === "running" || j.status === "pending") m.set(j.repo_id, j);
    }
    return m;
  }, [jobs]);

  const chooserTabs: { id: ChooserTab; icon: React.ReactNode; label: string }[] = [
    { id: "library",  icon: <Library size={11} />,  label: "Your library" },
    { id: "discover", icon: <Sparkles size={11} />, label: "Discover" },
    { id: "custom",   icon: <HardDrive size={11} />, label: "Custom" },
  ];

  return (
    <div className="space-y-4">
      {/* Capability bar always visible */}
      {caps && <CapabilitiesBar caps={caps} />}

      {/* Selected model chip */}
      {selectedRepoId && (
        <div className="flex items-center gap-2 px-3 py-2 rounded-xl border border-emerald-500/30 bg-emerald-500/[0.08]">
          <CheckCircle2 size={13} className="text-emerald-400 shrink-0" />
          <div className="min-w-0 flex-1">
            <p className="text-[10px] font-medium text-emerald-400 uppercase tracking-wide leading-none mb-0.5">Active model</p>
            <p className="text-xs font-semibold text-th-text-primary truncate font-mono">{selectedRepoId}</p>
          </div>
          {onUnload && (
            <div className="shrink-0 flex flex-col items-end gap-1">
              <button
                type="button"
                onClick={onUnload}
                disabled={unloading}
                title="Evict the in-process MLX model and free Metal GPU memory"
                className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-[10px] font-medium border border-rose-500/30 bg-rose-500/10 text-rose-400 hover:bg-rose-500/20 disabled:opacity-50 transition-colors"
              >
                {unloading ? <Loader2 size={10} className="animate-spin" /> : <Square size={10} />}
                Unload
              </button>
              {unloadMsg && (
                <span className={`text-[10px] ${unloadMsg.ok ? "text-emerald-400" : "text-red-400"}`}>
                  {unloadMsg.text}
                </span>
              )}
            </div>
          )}
        </div>
      )}

      {/* Tab bar */}
      <div className="rounded-lg border border-th-border bg-th-inset-bg overflow-hidden">
        <div className="flex border-b border-th-border">
          {chooserTabs.map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => setTab(t.id)}
              className={`flex-1 px-3 py-2 text-[11px] font-medium flex items-center justify-center gap-1.5 transition-colors
                ${tab === t.id
                  ? "bg-th-tab-active-bg text-white"
                  : "text-th-text-secondary hover:text-th-text-primary hover:bg-th-surface-hover/20"}`}
            >
              {t.icon}
              {t.label}
            </button>
          ))}
        </div>

        {/* ── Your Library tab ── */}
        {tab === "library" && (
          <div className="p-3 space-y-2">
            {/* Search */}
            <div className="relative">
              <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-th-text-muted pointer-events-none" />
              <input
                type="text"
                placeholder="Search cached models…"
                value={librarySearch}
                onChange={(e) => setLibrarySearch(e.target.value)}
                className="w-full pl-7 pr-3 py-1.5 text-[11px] rounded-lg border border-th-border bg-th-surface text-th-text-primary placeholder:text-th-text-muted focus:outline-none focus:ring-1 focus:ring-th-tab-active-bg/40 focus:border-th-tab-active-bg/50 transition-shadow"
              />
            </div>
            <div className="space-y-2 max-h-72 overflow-y-auto">
            {loadingLocal && (
              <div className="flex items-center gap-2 text-[11px] text-th-text-muted py-4 justify-center">
                <Loader2 size={12} className="animate-spin" />
                Scanning HF cache…
              </div>
            )}
            {localError && (
              <p className="text-[10px] text-red-400">{localError}</p>
            )}
            {!loadingLocal && localModels?.length === 0 && (
              <div className="py-4 text-center space-y-1">
                <p className="text-[11px] text-th-text-secondary">No models found in your HF cache.</p>
                <p className="text-[10px] text-th-text-muted">
                  Switch to <strong>Discover</strong> to download a model.
                </p>
              </div>
            )}
            {!loadingLocal && (() => {
              const s = librarySearch.trim().toLowerCase();
              const visible = (localModels ?? []).filter((m) =>
                !s || m.repo_id.toLowerCase().includes(s) || m.name.toLowerCase().includes(s),
              );
              if (visible.length === 0 && s) {
                return (
                  <p className="text-[11px] text-th-text-muted text-center py-4">
                    No models match "{librarySearch}".
                  </p>
                );
              }
              return visible.map((m) => {
                const isActive = m.repo_id === selectedRepoId;
                return (
                  <div
                    key={m.repo_id}
                    className={`flex items-center justify-between gap-2 rounded-lg border px-3 py-2 ${
                      isActive
                        ? "border-emerald-500/40 bg-emerald-500/[0.06] ring-1 ring-emerald-500/20"
                        : "border-th-border bg-th-surface"
                    }`}
                  >
                    <div className="min-w-0 flex-1">
                      <p className="text-[11px] font-medium text-th-text-primary truncate">{m.repo_id}</p>
                      <div className="flex items-center gap-1.5 mt-0.5">
                        <span className="text-[9px] text-th-text-muted">{(m.size_mb / 1024).toFixed(1)} GB</span>
                        <span className="text-[9px] font-semibold px-1 py-0.5 rounded bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
                          MLX
                        </span>
                        {isActive && (
                          <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 inline-flex items-center gap-0.5">
                            <CheckCircle2 size={8} />
                            active
                          </span>
                        )}
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={() => onUseCached?.(m.repo_id, m.name || m.repo_id)}
                      className={`shrink-0 px-2.5 py-1 rounded-md text-[10px] font-medium transition-colors inline-flex items-center gap-1 ${
                        isActive
                          ? "bg-emerald-700/50 text-emerald-200 cursor-default"
                          : "bg-emerald-600 text-white hover:bg-emerald-500"
                      }`}
                    >
                      <CheckCircle2 size={10} />
                      {isActive ? "Selected" : "Use this model"}
                    </button>
                  </div>
                );
              });
            })()}
            </div>
            <button
              type="button"
              className="w-full text-[10px] text-th-text-muted hover:text-th-text-secondary transition-colors pt-1 inline-flex items-center justify-center gap-1"
              onClick={() => { setLocalModels(null); void fetchLocalModels(); }}
            >
              <RefreshCw size={10} />
              Refresh library
            </button>
          </div>
        )}

        {/* ── Discover tab ── */}
        {tab === "discover" && (
          <div className="p-3 space-y-3">
            {/* Header: Refresh */}
            <div className="flex items-center justify-between gap-2">
              <p className="text-[11px] text-th-text-tertiary">
                Hardware-scored catalog — models ranked by fit for your Mac.
              </p>
              <button
                type="button"
                className="inline-flex items-center gap-1.5 px-2.5 py-1 text-[11px] font-medium rounded-md border border-th-border bg-th-surface text-th-text-secondary hover:bg-th-surface-hover disabled:opacity-50 transition-colors shrink-0"
                onClick={() => void refresh(true)}
                disabled={loading}
                title="Force-refresh download counts from Hugging Face"
              >
                {loading ? <Loader2 size={11} className="animate-spin" /> : <RefreshCw size={11} />}
                Refresh
              </button>
            </div>

            {error && (
              <p className="text-xs text-rose-400 inline-flex items-center gap-1.5 px-3 py-2 rounded-lg bg-rose-500/5 border border-rose-500/20">
                <XCircle size={12} className="shrink-0" />
                {error}
              </p>
            )}

            {enriching && (
              <p className="text-[11px] text-th-text-tertiary inline-flex items-center gap-1.5 px-3 py-2 rounded-lg bg-th-inset-bg border border-th-border">
                <Loader2 size={11} className="animate-spin text-th-text-muted shrink-0" />
                Fetching full model list from HuggingFace… list will update automatically.
              </p>
            )}

            {jobs.length > 0 && (
              <div className="space-y-2">
                {jobs.map((j) => (
                  <DownloadProgress
                    key={j.job_id}
                    job={j}
                    onCancel={() => void cancelJob(j.job_id)}
                    onDismiss={() => dismissJob(j.job_id)}
                  />
                ))}
              </div>
            )}

            {/* Filter segmented control + search */}
            <div className="space-y-2">
              <div className="inline-flex items-center gap-0.5 p-0.5 rounded-lg bg-th-inset-bg border border-th-border">
                {(["comfortable", "tight", "all"] as FitFilter[]).map((f) => {
                  const comfortableN = counts?.comfortable ?? 0;
                  const tightOrBetterN = comfortableN + (counts?.tight ?? 0);
                  const label =
                    f === "comfortable"
                      ? `Comfortable (${comfortableN})`
                      : f === "tight"
                        ? `Fits your Mac (${tightOrBetterN})`
                        : `All (${rows.length})`;
                  return (
                    <button
                      key={f}
                      type="button"
                      className={`px-2.5 py-1 text-[11px] font-medium rounded-md transition-all ${
                        filter === f
                          ? "bg-th-surface border border-th-border shadow-sm text-th-text-primary"
                          : "text-th-text-tertiary hover:text-th-text-secondary"
                      }`}
                      onClick={() => setFilter(f)}
                    >
                      {label}
                    </button>
                  );
                })}
              </div>

              <div className="relative">
                <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-th-text-muted pointer-events-none" />
                <input
                  type="text"
                  placeholder="Search by name, family, or tag…"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="w-full pl-7 pr-3 py-1.5 text-[11px] rounded-lg border border-th-border bg-th-surface text-th-text-primary placeholder:text-th-text-muted focus:outline-none focus:ring-1 focus:ring-th-tab-active-bg/40 focus:border-th-tab-active-bg/50 transition-shadow"
                />
              </div>
            </div>

            <div className="space-y-2">
              {!loading && filtered.length === 0 && (
                <p className="text-xs text-th-text-muted text-center py-6">
                  No models match the current filter.
                </p>
              )}
              {paginated.map((row) => (
                <ModelRow
                  key={row.repo_id}
                  row={row}
                  isSelected={row.repo_id === selectedRepoId}
                  activeJob={activeJobsByRepo.get(row.repo_id)}
                  ctxLen={ctxLen}
                  onDownload={() => void startDownload(row.repo_id)}
                  onUseCached={() => onUseCached?.(row.repo_id, row.display_name)}
                />
              ))}
            </div>

            {pageCount > 1 && (
              <div className="flex items-center justify-between gap-2 pt-1">
                <span className="text-[11px] text-th-text-muted">
                  {filtered.length} models · page {safePage} of {pageCount}
                </span>
                <div className="flex items-center gap-1">
                  <button
                    type="button"
                    disabled={safePage <= 1}
                    onClick={() => setPage((p) => Math.max(1, p - 1))}
                    className="p-1 rounded border border-th-border bg-th-surface text-th-text-secondary hover:bg-th-surface-hover disabled:opacity-40 disabled:cursor-not-allowed"
                    title="Previous page"
                  >
                    <ChevronLeft size={13} />
                  </button>
                  {(() => {
                    const pages: (number | "…")[] = [];
                    if (pageCount <= 7) {
                      for (let i = 1; i <= pageCount; i++) pages.push(i);
                    } else {
                      pages.push(1);
                      if (safePage > 3) pages.push("…");
                      for (let i = Math.max(2, safePage - 1); i <= Math.min(pageCount - 1, safePage + 1); i++) pages.push(i);
                      if (safePage < pageCount - 2) pages.push("…");
                      pages.push(pageCount);
                    }
                    return pages.map((p, i) =>
                      p === "…" ? (
                        <span key={`ellipsis-${i}`} className="px-1 text-[11px] text-th-text-muted select-none">…</span>
                      ) : (
                        <button
                          key={p}
                          type="button"
                          onClick={() => setPage(p as number)}
                          className={`min-w-[26px] px-1.5 py-0.5 text-[11px] font-medium rounded border ${
                            p === safePage
                              ? "border-th-tab-active-bg bg-th-tab-active-bg/15 text-th-text-primary"
                              : "border-th-border bg-th-surface text-th-text-tertiary hover:text-th-text-primary"
                          }`}
                        >
                          {p}
                        </button>
                      ),
                    );
                  })()}
                  <button
                    type="button"
                    disabled={safePage >= pageCount}
                    onClick={() => setPage((p) => Math.min(pageCount, p + 1))}
                    className="p-1 rounded border border-th-border bg-th-surface text-th-text-secondary hover:bg-th-surface-hover disabled:opacity-40 disabled:cursor-not-allowed"
                    title="Next page"
                  >
                    <ChevronRight size={13} />
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── Custom tab ── */}
        {tab === "custom" && (
          <div className="p-3 space-y-3">
            <p className="text-[11px] text-th-text-secondary leading-relaxed">
              Download any Hugging Face repo not shown in Discover. Uses the same
              pipeline (safetensors-only filter, hf_transfer acceleration, real progress bar).
            </p>

            {jobs.length > 0 && (
              <div className="space-y-2">
                {jobs.map((j) => (
                  <DownloadProgress
                    key={j.job_id}
                    job={j}
                    onCancel={() => void cancelJob(j.job_id)}
                    onDismiss={() => dismissJob(j.job_id)}
                  />
                ))}
              </div>
            )}

            <div className="space-y-2">
              <input
                type="text"
                placeholder="Display name (optional)"
                value={advancedLabel}
                onChange={(e) => setAdvancedLabel(e.target.value)}
                className="w-full px-2.5 py-1.5 text-[11px] rounded-lg border border-th-border bg-th-surface text-th-text-primary placeholder:text-th-text-muted"
              />
              <div className="flex gap-2">
                <input
                  type="text"
                  placeholder="mlx-community/your-model or /path/to/model"
                  value={advancedRepo}
                  onChange={(e) => setAdvancedRepo(e.target.value)}
                  className="flex-1 px-2.5 py-1.5 text-[11px] rounded-lg border border-th-border bg-th-surface text-th-text-primary placeholder:text-th-text-muted font-mono"
                />
                <button
                  type="button"
                  className="shrink-0 px-3 py-1.5 rounded-lg bg-th-tab-active-bg text-white text-[10px] font-medium disabled:opacity-50 inline-flex items-center gap-1.5"
                  disabled={!advancedRepo.trim() || activeJobsByRepo.has(advancedRepo.trim())}
                  onClick={() => {
                    const r = advancedRepo.trim();
                    if (r) void startDownload(r);
                  }}
                >
                  <Download size={10} />
                  Download
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
