import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Cpu,
  Loader2,
  RefreshCw,
  Server,
  Sparkles,
  XCircle,
  X,
  Zap,
} from "lucide-react";
import { api } from "../../hooks/useApi";
import { usePolling } from "../../hooks/usePolling";
import type {
  ExoCatalogResponse,
  ExoCatalogRow,
  ExoNodeInfo,
  ExoPreloadJob,
  ExoPreloadStage,
} from "../../types";

// ---------------------------------------------------------------------------
// EXO Model Chooser — sibling of ``app/src/components/mlx/ModelChooser``.
//
// Reads ``GET /api/exo/catalog`` (cluster-aware fit scoring derived from
// exo's TOML model cards + live ``/state`` topology) and drives async
// preload via ``POST /api/exo/preload``.  Polls
// ``GET /api/exo/preloads`` while any job is active so the UI re-attaches
// to in-flight preloads when the user navigates away and back.
//
// Differences vs. the MLX chooser:
//   · "Capabilities" here is a *cluster summary* (peer count, RAM
//     bottleneck) rather than a single machine.
//   · Has a ``min_nodes`` slider that re-scores the catalog against
//     hypothetical cluster splits.
//   · "Fits" = "weights + KV + overhead, divided across N nodes,
//     stays under the smallest-node ceiling".  An "over" badge here
//     usually flips to "comfortable" by raising ``min_nodes``.
//   · Preload progress comes from exo's own download events (per-node
//     bytes/files/speed), not Hugging Face downloads.
// ---------------------------------------------------------------------------

export interface ExoModelChooserProps {
  /** Whether EXO is enabled in settings — disables the action button. */
  enabled: boolean;
  /** Currently-selected model id (highlighted in the list). */
  selectedModelId?: string;
  /** Called when a preload finishes successfully. */
  onPreloadComplete?: (modelId: string) => void;
  /** Called when the user picks a model that's already loaded. */
  onUseLoaded?: (modelId: string) => void;
  /**
   * First-run / beginner mode. Surfaces a single recommended model (best
   * comfortable fit) with everything else tucked behind a "See all models"
   * disclosure — no fit filters, search, or min_nodes slider until expanded.
   */
  simple?: boolean;
}

/** Pick the best starting model for a cluster: a featured, comfortable fit if
 *  available, else the first comfortable fit, else an already-downloaded model,
 *  else the first catalog row. */
function pickRecommended(rows: ExoCatalogRow[]): ExoCatalogRow | null {
  if (rows.length === 0) return null;
  return (
    rows.find((r) => r.featured && r.fits === "comfortable") ??
    rows.find((r) => r.fits === "comfortable") ??
    rows.find((r) => r.loaded) ??
    rows.find((r) => r.downloaded) ??
    rows[0]
  );
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

function FitBadge({ fits }: { fits: ExoCatalogRow["fits"] }) {
  const map: Record<ExoCatalogRow["fits"], { label: string; cls: string; dot: string }> = {
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
      label: "Cluster offline",
      cls: "bg-th-surface text-th-text-muted ring-1 ring-th-border",
      dot: "bg-neutral-400",
    },
  };
  const v = map[fits];
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] font-medium ${v.cls}`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${v.dot}`} />
      {v.label}
    </span>
  );
}

function ClusterBar({
  reachable,
  peerCount,
  nodes,
}: {
  reachable: boolean;
  peerCount: number;
  nodes: ExoNodeInfo[];
}) {
  const totalRam = nodes.reduce(
    (acc, n) => acc + (n.memory_total_gb || 0),
    0,
  );
  const minRam = nodes.length
    ? Math.min(...nodes.map((n) => n.memory_total_gb || 0))
    : 0;

  if (!reachable && nodes.length === 0) {
    return (
      <p className="inline-flex items-center gap-1.5 text-[11px] text-th-text-muted">
        <AlertTriangle size={11} />
        Cluster offline — start it to see fit scores.
      </p>
    );
  }

  const totalFreeRam = nodes.reduce(
    (acc, n) => acc + (n.memory_free_gb || 0),
    0,
  );
  const lowMemory = totalFreeRam > 0 && totalFreeRam < minRam * 0.6;

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[11px] text-th-text-tertiary">
      <span className="inline-flex items-center gap-1.5 text-th-text-secondary font-medium">
        <Server size={12} className="text-th-text-muted" />
        {peerCount} {peerCount === 1 ? "node" : "nodes"}
      </span>
      <span>{totalRam.toFixed(0)} GB total</span>
      {totalFreeRam > 0 && (
        <span
          title="Currently free RAM across the cluster — models need to fit within this to load"
          className={lowMemory ? "text-amber-400 font-medium" : ""}
        >
          {totalFreeRam.toFixed(1)} GB free{lowMemory ? " ⚠" : ""}
        </span>
      )}
      <span title="Smallest node — bottleneck for sharded models">
        bottleneck {minRam.toFixed(0)} GB
      </span>
      {nodes.slice(0, 4).map((n) => (
        <span
          key={n.node_id || n.chip || Math.random()}
          className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-th-inset-bg"
          title={`${n.chip || "?"} · ${n.memory_total_gb ?? "?"} GB total · ${n.memory_free_gb?.toFixed(1) ?? "?"} GB free`}
        >
          <Cpu size={10} className="text-th-text-muted" />
          {n.chip || "?"}
        </span>
      ))}
      {nodes.length > 4 && <span>+{nodes.length - 4} more</span>}
    </div>
  );
}

function PreloadProgress({
  job,
  onCancel,
  onDismiss,
  onUnload,
  unloading,
}: {
  job: ExoPreloadJob;
  onCancel: () => void;
  onDismiss: () => void;
  onUnload?: () => void;
  unloading?: boolean;
}) {
  // When loading an already-downloaded model into cluster memory, bytes_done
  // stays at 0 while bytes_total is the model size — show indeterminate instead
  // of a misleading 0% bar.
  const isLoadingStage = job.stage === "loading" || job.stage === "placing";
  const pct =
    job.bytes_total > 0 && !isLoadingStage
      ? Math.min(100, Math.round((job.bytes_done / job.bytes_total) * 100))
      : 0;
  const indeterminate =
    (job.bytes_total === 0 || isLoadingStage) &&
    (job.status === "running" || job.status === "done" && job.stage === "loading");
  const terminal =
    job.status === "done" || job.status === "error" || job.status === "cancelled";

  const stageLabel: Record<ExoPreloadStage, string> = {
    placing: "Placing instance…",
    downloading: `Downloading shards${job.nodes_active ? ` · ${job.nodes_active} node${job.nodes_active === 1 ? "" : "s"}` : ""}`,
    loading: "Loading into memory…",
    done: "Loaded",
    error: job.message || "Failed",
    cancelled: "Cancelled",
  };

  return (
    <div className="rounded-xl border border-th-border bg-th-surface p-3 space-y-2">
      <div className="flex items-center justify-between gap-2">
        <div className="min-w-0 flex-1">
          <p className="text-xs font-medium text-th-text-primary truncate">
            {job.model_id}
          </p>
          <p className="text-[11px] text-th-text-muted truncate">
            {stageLabel[job.stage]}
            {job.min_nodes > 1 && job.stage !== "done" && (
              <span className="ml-1 text-th-text-tertiary">
                · {job.min_nodes}-node split
              </span>
            )}
          </p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <button
            type="button"
            className="text-th-text-muted hover:text-rose-400 disabled:opacity-40"
            disabled={unloading}
            onClick={() => {
              if (!terminal) {
                onCancel();
              } else if (job.status === "done" && onUnload) {
                onUnload();
              } else {
                onDismiss();
              }
            }}
            title={
              !terminal
                ? "Cancel preload"
                : job.status === "done"
                  ? "Unload model from cluster memory"
                  : "Dismiss"
            }
          >
            {unloading ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <X size={14} />
            )}
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
          {/* Hide byte counters during placing/loading — they don't reflect memory load progress */}
          {!isLoadingStage && (
            <>
              {bytesToHuman(job.bytes_done)} /{" "}
              {job.bytes_total > 0 ? bytesToHuman(job.bytes_total) : "?"}
              {!indeterminate && job.bytes_total > 0 && (
                <span className="ml-1">· {pct}%</span>
              )}
              {job.files_total > 0 && (
                <span className="ml-2">
                  {job.files_done}/{job.files_total} files
                </span>
              )}
            </>
          )}
        </span>
        <span>
          {job.status === "running" && !isLoadingStage && (
            <>
              {formatRate(job.rate_bps)} · ETA {formatEta(job.eta_seconds)}
            </>
          )}
          {job.elapsed_seconds > 0 &&
            `${Math.round(job.elapsed_seconds)}s elapsed`}
        </span>
      </div>
    </div>
  );
}

function ModelRow({
  row,
  isSelected,
  activeJob,
  enabled,
  onPreload,
  onUseLoaded,
  onUnload,
  unloading,
}: {
  row: ExoCatalogRow;
  isSelected: boolean;
  activeJob?: ExoPreloadJob;
  enabled: boolean;
  onPreload: () => void;
  onUseLoaded: () => void;
  onUnload: () => void;
  unloading: boolean;
}) {
  const breakdownTitle = useMemo(
    () =>
      [
        `Weights: ${row.weights_gb.toFixed(1)} GB`,
        `KV cache: ${row.kv_gb.toFixed(1)} GB`,
        `Framework + activations: ${row.overhead_gb.toFixed(1)} GB`,
        `≈ Total runtime: ${row.total_gb.toFixed(1)} GB`,
        row.bottleneck_gb > 0
          ? `Per-node share: ${row.per_node_gb.toFixed(1)} GB / ${row.bottleneck_gb.toFixed(0)} GB bottleneck`
          : "",
      ]
        .filter(Boolean)
        .join("\n"),
    [row],
  );

  const dlBlocked = !enabled || row.fits === "over";

  return (
    <div
      className={`rounded-xl border p-3 transition-colors ${
        isSelected
          ? "border-th-tab-active-bg/60 bg-th-tab-active-bg/5"
          : "border-th-border bg-th-surface"
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1 space-y-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm font-medium text-th-text-primary truncate">
              {row.base_model}
            </span>
            {row.featured && (
              <span
                title="Recommended starting point"
                className="inline-flex items-center gap-0.5 text-[10px] font-medium text-amber-500"
              >
                <Sparkles size={10} />
                featured
              </span>
            )}
            {row.loaded && (
              <span className="inline-flex items-center gap-1 text-[10px] font-medium text-emerald-500">
                <CheckCircle2 size={10} />
                loaded
              </span>
            )}
            {!row.loaded && row.downloaded && (
              <span className="inline-flex items-center gap-1 text-[10px] font-medium text-th-text-tertiary">
                <CheckCircle2 size={10} />
                downloaded
              </span>
            )}
          </div>
          <p className="text-[11px] text-th-text-tertiary truncate font-mono">
            {row.model_id}
          </p>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-th-text-muted">
            <span title={breakdownTitle} className="cursor-help font-mono">
              {row.weights_gb.toFixed(1)} GB · {row.params_b.toFixed(0)}B · {row.quant}
            </span>
            <span className="font-mono">
              ctx {row.context_length.toLocaleString()}
            </span>
            {row.capabilities.slice(0, 4).map((t) => (
              <span
                key={t}
                className="px-1.5 py-0.5 rounded bg-th-inset-bg text-th-text-secondary text-[10px]"
              >
                {t}
              </span>
            ))}
          </div>
          {row.fits === "over" && row.min_nodes_required != null && (
            <p className="text-[11px] text-amber-400">
              Would fit comfortably with {row.min_nodes_required} node
              {row.min_nodes_required === 1 ? "" : "s"} — try the slider above.
            </p>
          )}
          {row.fits === "over" && row.min_nodes_required == null && (
            <p className="text-[11px] text-rose-400 inline-flex items-center gap-1">
              <AlertTriangle size={11} />
              No combination of current nodes can hold this model.
            </p>
          )}
        </div>

        <div className="flex flex-col items-end gap-2 shrink-0">
          <FitBadge fits={row.fits} />
          {row.loaded ? (
            <div className="flex flex-col items-end gap-1.5">
              <button
                type="button"
                className="px-2.5 py-1 text-[11px] font-medium rounded-md border border-th-border bg-th-surface text-th-text-secondary hover:bg-th-surface-hover"
                onClick={onUseLoaded}
              >
                Use this model
              </button>
              <button
                type="button"
                className="inline-flex items-center gap-1 px-2.5 py-1 text-[11px] font-medium rounded-md border border-rose-500/30 bg-rose-500/10 text-rose-500 hover:bg-rose-500/20 disabled:opacity-40"
                onClick={onUnload}
                disabled={!enabled || unloading}
                title="Free cluster RAM by removing the in-memory instance (weights stay on disk)"
              >
                {unloading ? (
                  <Loader2 size={10} className="animate-spin" />
                ) : (
                  <X size={10} />
                )}
                Unload
              </button>
            </div>
          ) : activeJob ? (
            <span
              className="inline-flex items-center gap-1 px-2.5 py-1 text-[11px] font-medium rounded-md bg-th-inset-bg text-th-text-tertiary"
              title="Preload in progress — see card above."
            >
              <Loader2 size={11} className="animate-spin" />
              {activeJob.bytes_total > 0
                ? `${Math.min(100, Math.round((activeJob.bytes_done / activeJob.bytes_total) * 100))}%`
                : activeJob.stage === "loading"
                  ? "Loading…"
                  : "Starting…"}
            </span>
          ) : (
            <button
              type="button"
              className="inline-flex items-center gap-1 px-2.5 py-1 text-[11px] font-medium rounded-md bg-th-tab-active-bg text-th-tab-active-fg disabled:opacity-50 hover:opacity-90"
              onClick={onPreload}
              disabled={dlBlocked}
              title={
                !enabled
                  ? "Enable the cluster in settings"
                  : row.fits === "over"
                    ? `Won't fit — needs ~${row.total_gb.toFixed(1)} GB`
                    : row.downloaded
                      ? "Load model into cluster memory"
                      : "Download weights to cluster, then load into memory"
              }
            >
              <Zap size={11} />
              {row.downloaded ? "Load" : "Download"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export default function ExoModelChooser({
  enabled,
  selectedModelId,
  onPreloadComplete,
  onUseLoaded,
  simple = false,
}: ExoModelChooserProps) {
  const [resp, setResp] = useState<ExoCatalogResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<FitFilter>("all");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [minNodes, setMinNodes] = useState<number>(1);
  // In simple mode the full catalog stays hidden until the user asks for it.
  const [expanded, setExpanded] = useState(false);
  // Multi-job state — same shape as the MLX chooser.
  const [jobs, setJobs] = useState<ExoPreloadJob[]>([]);
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());
  // Track which models are currently being unloaded (by model_id).
  const [unloadingModels, setUnloadingModels] = useState<Set<string>>(new Set());

  const refresh = useCallback(
    async (overrides?: { minNodes?: number; force?: boolean }) => {
      setLoading(true);
      setError(null);
      try {
        const r = await api.exoCatalog({
          min_nodes: overrides?.minNodes ?? minNodes,
          refresh: overrides?.force ?? false,
        });
        setResp(r);
        // Keep the slider clamped within the cluster's max range.
        if (r.cluster.max_nodes && minNodes > r.cluster.max_nodes) {
          setMinNodes(r.cluster.max_nodes);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load catalog");
      } finally {
        setLoading(false);
      }
    },
    [minNodes],
  );

  useEffect(() => {
    void refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Refresh when minNodes changes — rescore against the new split.
  useEffect(() => {
    void refresh({ minNodes });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [minNodes]);

  // Stable refs for the polling callback.
  const dismissedRef = useRef(dismissed);
  dismissedRef.current = dismissed;
  const onPreloadDoneRef = useRef(onPreloadComplete);
  onPreloadDoneRef.current = onPreloadComplete;
  const refreshRef = useRef(refresh);
  refreshRef.current = refresh;
  // Keep a stable ref to the latest catalog response so startPreload can
  // read the current loaded state without needing `resp` in its dep array
  // (which would recreate the function on every catalog refresh).
  const respRef = useRef(resp);
  respRef.current = resp;

  const syncJobs = useCallback(async () => {
    try {
      const r = await api.exoPreloadList();
      setJobs((prev) => {
        const prevById = new Map(prev.map((j) => [j.job_id, j]));
        const now = Date.now() / 1000;
        const eligible = r.jobs.filter((j) => {
          if (dismissedRef.current.has(j.job_id)) return false;
          if (j.status === "running") return true;
          if (j.started_at != null && now - j.started_at < 60) return true;
          return prevById.has(j.job_id);
        });
        // For each model keep only the most-recent job so completed
        // cards from prior runs don't stack up in the UI.
        const latestByModel = new Map<string, ExoPreloadJob>();
        for (const j of eligible) {
          const cur = latestByModel.get(j.model_id);
          if (!cur || (j.started_at ?? 0) > (cur.started_at ?? 0)) {
            latestByModel.set(j.model_id, j);
          }
        }
        const visible = Array.from(latestByModel.values());
        for (const next of visible) {
          const before = prevById.get(next.job_id);
          if (next.status === "done" && before && before.status !== "done") {
            onPreloadDoneRef.current?.(next.model_id);
            void refreshRef.current({ force: false });
          }
        }
        return visible;
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Status polling failed");
    }
  }, []);

  useEffect(() => {
    void syncJobs();
  }, [syncJobs]);

  const hasActive = jobs.some((j) => j.status === "running");
  usePolling(syncJobs, 1500, hasActive || jobs.length > 0);

  const filtered = useMemo(() => {
    if (!resp) return [] as ExoCatalogRow[];
    const s = search.trim().toLowerCase();
    return resp.rows.filter((r) => {
      if (filter === "comfortable" && r.fits !== "comfortable") return false;
      if (
        filter === "tight" &&
        !(r.fits === "comfortable" || r.fits === "tight")
      ) {
        return false;
      }
      if (s) {
        const hay = `${r.model_id} ${r.base_model} ${r.family} ${r.capabilities.join(" ")}`.toLowerCase();
        if (!hay.includes(s)) return false;
      }
      return true;
    });
  }, [resp, filter, search]);

  // Reset to first page whenever the visible set changes.
  useEffect(() => {
    setPage(1);
  }, [filter, search, minNodes]);

  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const safePage = Math.min(page, pageCount);
  const paginated = filtered.slice((safePage - 1) * PAGE_SIZE, safePage * PAGE_SIZE);

  const startPreload = useCallback(
    async (modelId: string) => {
      setError(null);
      try {
        // Unload any model that is currently loaded before starting the
        // new placement — only one model in cluster memory at a time.
        // Uses respRef (not `resp` directly) so we always read the latest
        // catalog without including `resp` in the dep array.
        const loaded = (respRef.current?.rows ?? []).filter(
          (r) => r.loaded && r.model_id !== modelId,
        );
        for (const prev of loaded) {
          await api.exoUnload(prev.model_id);
        }
        const j = await api.exoPreloadStart(modelId, minNodes);
        // Optimistic: prepend so the user sees the card immediately.
        setDismissed((prev) => {
          if (!prev.has(j.job_id)) return prev;
          const next = new Set(prev);
          next.delete(j.job_id);
          return next;
        });
        setJobs((prev) => {
          const without = prev.filter((x) => x.job_id !== j.job_id);
          return [j, ...without];
        });
      } catch (e) {
        setError(e instanceof Error ? e.message : "Preload failed to start");
      }
    },
    [minNodes],
  );

  const handleUnload = useCallback(
    async (modelId: string) => {
      setUnloadingModels((prev) => new Set(prev).add(modelId));
      try {
        await api.exoUnload(modelId);
        // Refresh the catalog so the "loaded" badge clears.
        await refresh();
      } catch (e) {
        setError(e instanceof Error ? e.message : "Unload failed");
      } finally {
        setUnloadingModels((prev) => {
          const next = new Set(prev);
          next.delete(modelId);
          return next;
        });
      }
    },
    [refresh],
  );

  const cancelJob = useCallback(async (jobId: string) => {
    try {
      await api.exoPreloadCancel(jobId);
    } catch {
      // Best-effort; the worker will set its own terminal status.
    }
  }, []);

  const dismissJob = useCallback((jobId: string) => {
    setDismissed((prev) => {
      const next = new Set(prev);
      next.add(jobId);
      return next;
    });
    setJobs((prev) => prev.filter((j) => j.job_id !== jobId));
  }, []);

  const activeJobsByModel = useMemo(() => {
    const m = new Map<string, ExoPreloadJob>();
    for (const j of jobs) {
      if (j.status === "running") m.set(j.model_id, j);
    }
    return m;
  }, [jobs]);

  const counts = resp?.counts;
  const cluster = resp?.cluster;
  const recommended = useMemo(() => pickRecommended(resp?.rows ?? []), [resp]);
  // Condensed = simple mode, not yet expanded — show only the recommended pick.
  const condensed = simple && !expanded;

  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between gap-3 flex-wrap">
        <div className="space-y-1">
          <p className="text-xs font-medium text-th-text-secondary uppercase tracking-wide">
            Pick a model
          </p>
          <ClusterBar
            reachable={!!cluster?.reachable}
            peerCount={cluster?.peer_count ?? 0}
            nodes={cluster?.nodes ?? []}
          />
        </div>
        <button
          type="button"
          className="inline-flex items-center gap-1.5 px-2.5 py-1 text-[11px] font-medium rounded-md border border-th-border bg-th-surface text-th-text-secondary hover:bg-th-surface-hover disabled:opacity-50"
          onClick={() => void refresh({ force: true })}
          disabled={loading}
          title="Re-scan model cards and re-fetch cluster status"
        >
          {loading ? (
            <Loader2 size={11} className="animate-spin" />
          ) : (
            <RefreshCw size={11} />
          )}
          Refresh
        </button>
      </div>

      {error && (
        <p className="text-xs text-rose-400 inline-flex items-center gap-1">
          <XCircle size={12} />
          {error}
        </p>
      )}

      {jobs.length > 0 && (
        <div className="space-y-2">
          {jobs.map((j) => (
            <PreloadProgress
              key={j.job_id}
              job={j}
              onCancel={() => void cancelJob(j.job_id)}
              onDismiss={() => dismissJob(j.job_id)}
              onUnload={async () => {
                await handleUnload(j.model_id);
                dismissJob(j.job_id);
              }}
              unloading={unloadingModels.has(j.model_id)}
            />
          ))}
        </div>
      )}

      {/* Simple mode: one recommended model + a disclosure to the full catalog. */}
      {condensed && (
        <div className="space-y-3">
          {recommended ? (
            <>
              <p className="text-[11px] font-medium text-th-text-secondary inline-flex items-center gap-1.5">
                <Sparkles size={12} className="text-amber-500" />
                Recommended for your cluster
              </p>
              <ModelRow
                row={recommended}
                isSelected={recommended.model_id === selectedModelId}
                activeJob={activeJobsByModel.get(recommended.model_id)}
                enabled={enabled}
                onPreload={() => void startPreload(recommended.model_id)}
                onUseLoaded={() => onUseLoaded?.(recommended.model_id)}
                onUnload={() => void handleUnload(recommended.model_id)}
                unloading={unloadingModels.has(recommended.model_id)}
              />
            </>
          ) : (
            !loading && (
              <p className="text-xs text-th-text-muted text-center py-4">
                {cluster?.reachable === false
                  ? "Cluster offline — start it to see a recommendation."
                  : "No models available yet."}
              </p>
            )
          )}
          <button
            type="button"
            onClick={() => setExpanded(true)}
            className="text-[11px] text-th-text-tertiary hover:text-th-text-primary underline underline-offset-2 transition-colors"
          >
            See all {counts?.total ?? 0} models
          </button>
        </div>
      )}

      {/* min_nodes slider — only meaningful when peer_count > 1 */}
      {!condensed && (cluster?.max_nodes ?? 1) > 1 && (
        <div className="rounded-lg border border-th-border bg-th-inset-bg/40 px-3 py-2 space-y-1">
          <div className="flex items-center justify-between text-[11px] text-th-text-secondary">
            <label htmlFor="exo-min-nodes" className="font-medium">
              Split across at least {minNodes} node
              {minNodes === 1 ? "" : "s"}
            </label>
            <span className="text-th-text-muted">
              max {cluster?.max_nodes}
            </span>
          </div>
          <input
            id="exo-min-nodes"
            type="range"
            min={1}
            max={cluster?.max_nodes ?? 1}
            step={1}
            value={minNodes}
            onChange={(e) => setMinNodes(parseInt(e.target.value, 10) || 1)}
            className="w-full"
          />
          <p className="text-[10px] text-th-text-muted">
            Forces pipeline-parallel placement so larger models can fit by
            sharing weights across nodes.
          </p>
        </div>
      )}

      {!condensed && (
      <div className="space-y-2">
        <div className="flex items-center gap-2 flex-wrap">
          {(["comfortable", "tight", "all"] as FitFilter[]).map((f) => {
            const comfortableN = counts?.comfortable ?? 0;
            const tightOrBetterN = comfortableN + (counts?.tight ?? 0);
            const label =
              f === "comfortable"
                ? `Comfortable (${comfortableN})`
                : f === "tight"
                  ? `Fits your cluster (${tightOrBetterN})`
                  : `All (${counts?.total ?? 0})`;
            return (
              <button
                key={f}
                type="button"
                className={`px-2.5 py-1 text-[11px] font-medium rounded-md border ${
                  filter === f
                    ? "border-th-tab-active-bg bg-th-tab-active-bg/15 text-th-text-primary"
                    : "border-th-border bg-th-surface text-th-text-tertiary hover:text-th-text-primary"
                }`}
                onClick={() => setFilter(f)}
              >
                {label}
              </button>
            );
          })}
        </div>
        <input
          type="text"
          placeholder="Search by name, family, or tag…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="flex-1 min-w-[200px] px-2.5 py-1 text-[11px] rounded-md border border-th-border bg-th-surface text-th-text-primary placeholder:text-th-text-muted w-full"
        />
      </div>
      )}

      {!condensed && (
      <div className="space-y-2">
        {!loading && filtered.length === 0 && (
          <p className="text-xs text-th-text-muted text-center py-6">
            {resp?.cluster.reachable === false
              ? "Cluster offline — start it to score the catalog."
              : "No models match the current filter."}
          </p>
        )}
        {paginated.map((row) => (
          <ModelRow
            key={row.model_id}
            row={row}
            isSelected={row.model_id === selectedModelId}
            activeJob={activeJobsByModel.get(row.model_id)}
            enabled={enabled}
            onPreload={() => void startPreload(row.model_id)}
            onUseLoaded={() => onUseLoaded?.(row.model_id)}
            onUnload={() => void handleUnload(row.model_id)}
            unloading={unloadingModels.has(row.model_id)}
          />
        ))}
      </div>
      )}

      {/* Pagination */}
      {!condensed && pageCount > 1 && (
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
  );
}
