import { useCallback, useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  ArrowLeft,
  Ban,
  CheckCircle,
  ChevronLeft,
  ChevronRight,
  FolderOpen,
  Loader2,
  Play,
  Square,
  XCircle,
  Zap,
} from "lucide-react";
import { api } from "../hooks/useApi";
import { usePolling } from "../hooks/usePolling";
import type { TriggerRun, TriggerSpec } from "../types";

const PAGE_SIZE_OPTIONS = [20, 50, 100];
const DEFAULT_PAGE_SIZE = 20;

type TimeRange = "today" | "7d" | "30d" | "90d" | "all";
type StatusFilter = "" | "running" | "success" | "error" | "cancelled";

const TIME_RANGES: { id: TimeRange; label: string }[] = [
  { id: "today", label: "Today" },
  { id: "7d", label: "7 days" },
  { id: "30d", label: "30 days" },
  { id: "90d", label: "90 days" },
  { id: "all", label: "All time" },
];

const STATUS_FILTERS: { value: StatusFilter; label: string }[] = [
  { value: "", label: "All" },
  { value: "running", label: "Running" },
  { value: "success", label: "Success" },
  { value: "error", label: "Error" },
  { value: "cancelled", label: "Cancelled" },
];

function getAfterDate(range: TimeRange): string | undefined {
  const now = new Date();
  switch (range) {
    case "today": {
      const d = new Date(now);
      d.setHours(0, 0, 0, 0);
      return d.toISOString();
    }
    case "7d":
      return new Date(now.getTime() - 7 * 86_400_000).toISOString();
    case "30d":
      return new Date(now.getTime() - 30 * 86_400_000).toISOString();
    case "90d":
      return new Date(now.getTime() - 90 * 86_400_000).toISOString();
    case "all":
      return undefined;
  }
}

function formatDuration(run: TriggerRun): string {
  if (!run.finished_at) return run.status === "running" ? "running…" : "—";
  const ms = new Date(run.finished_at).getTime() - new Date(run.started_at).getTime();
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60_000);
  const s = Math.round((ms % 60_000) / 1000);
  return `${m}m ${s}s`;
}

function StatusIcon({ status }: { status: string }) {
  if (status === "running") return <Loader2 size={13} className="text-emerald-400 animate-spin" />;
  if (status === "success") return <CheckCircle size={13} className="text-emerald-400" />;
  if (status === "error") return <XCircle size={13} className="text-red-400" />;
  if (status === "cancelled") return <Ban size={13} className="text-amber-400" />;
  return <span className="w-3 h-3 rounded-full bg-th-border inline-block" />;
}

export default function TriggerRunsPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [trigger, setTrigger] = useState<TriggerSpec | null>(null);
  const [runs, setRuns] = useState<TriggerRun[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE);
  const [timeRange, setTimeRange] = useState<TimeRange>("all");
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("");
  const [loading, setLoading] = useState(true);
  const [actioning, setActioning] = useState(false);

  const fetchRuns = useCallback(async () => {
    if (!id) return;
    try {
      const after = getAfterDate(timeRange);
      const result = await api.getTriggerRuns(id, {
        limit: pageSize,
        offset: page * pageSize,
        after,
        status: statusFilter || undefined,
      });
      setRuns(result.runs);
      setTotal(result.total);
    } catch (e) {
      console.warn("[TriggerRunsPage] fetch failed:", e);
    } finally {
      setLoading(false);
    }
  }, [id, page, pageSize, timeRange, statusFilter]);

  useEffect(() => {
    setLoading(true);
    fetchRuns();
  }, [fetchRuns]);

  useEffect(() => {
    if (!id) return;
    api
      .listTriggers()
      .then((triggers) => {
        const found = triggers.find((t) => t.id === id);
        if (found) setTrigger(found);
      })
      .catch(() => {});
  }, [id]);

  const isRunning =
    trigger?.last_status === "running" || runs.some((r) => r.status === "running");

  usePolling(fetchRuns, 3000, isRunning);

  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  const rangeStart = total === 0 ? 0 : page * pageSize + 1;
  const rangeEnd = Math.min((page + 1) * pageSize, total);

  const handleTimeRangeChange = (r: TimeRange) => {
    setPage(0);
    setTimeRange(r);
  };

  const handleStatusChange = (s: StatusFilter) => {
    setPage(0);
    setStatusFilter(s);
  };

  const handlePageSizeChange = (size: number) => {
    setPage(0);
    setPageSize(size);
  };

  const handleRunNow = async () => {
    if (!id) return;
    setActioning(true);
    try {
      await api.runTriggerNow(id);
      fetchRuns();
    } catch { /* ignore */ }
    setActioning(false);
  };

  const handleStop = async () => {
    if (!id) return;
    try {
      await api.stopTriggerRun(id);
    } catch { /* ignore */ }
    fetchRuns();
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <header className="border-b border-th-border px-6 py-3.5 bg-th-bg-secondary shrink-0">
        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate("/triggers")}
            className="flex items-center gap-1.5 text-th-text-muted hover:text-th-text-primary transition-colors text-sm shrink-0"
          >
            <ArrowLeft size={14} aria-hidden />
            Triggers
          </button>
          <span className="text-th-border">/</span>
          <div className="flex items-center gap-2 flex-1 min-w-0">
            <Zap size={13} className="text-th-text-muted shrink-0" aria-hidden />
            <h1 className="text-base font-semibold text-th-text-primary truncate">{id}</h1>
            {trigger && (
              <span className="text-xs text-th-text-muted truncate hidden sm:block">
                · {trigger.type} · {trigger.agent_name || "general-purpose"}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2 ml-auto shrink-0">
            {isRunning ? (
              <button
                onClick={handleStop}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-red-500/30 text-xs font-medium text-red-400 hover:bg-red-500/10 transition-all"
              >
                <Square size={12} aria-hidden />
                Stop
              </button>
            ) : (
              <button
                onClick={handleRunNow}
                disabled={actioning}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-th-border text-xs font-medium text-th-text-secondary hover:text-th-text-primary hover:border-th-border-strong transition-all disabled:opacity-50"
              >
                {actioning ? (
                  <Loader2 size={12} className="animate-spin" aria-hidden />
                ) : (
                  <Play size={12} aria-hidden />
                )}
                Run now
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Filter + pagination bar */}
      <div className="border-b border-th-border px-6 py-2 bg-th-bg-secondary shrink-0 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1">
            {TIME_RANGES.map((r) => (
              <button
                key={r.id}
                onClick={() => handleTimeRangeChange(r.id)}
                className={`px-3 py-1 rounded-lg text-xs font-medium transition-all ${
                  timeRange === r.id
                    ? "bg-blue-500/15 text-blue-400 border border-blue-500/30"
                    : "text-th-text-muted hover:text-th-text-secondary hover:bg-th-surface-hover border border-transparent"
                }`}
              >
                {r.label}
              </button>
            ))}
          </div>
          <div className="w-px h-4 bg-th-border/60" />
          <div className="flex items-center gap-1">
            {STATUS_FILTERS.map((s) => (
              <button
                key={s.value}
                onClick={() => handleStatusChange(s.value)}
                className={`px-3 py-1 rounded-lg text-xs font-medium transition-all ${
                  statusFilter === s.value
                    ? "bg-blue-500/15 text-blue-400 border border-blue-500/30"
                    : "text-th-text-muted hover:text-th-text-secondary hover:bg-th-surface-hover border border-transparent"
                }`}
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>

        {total > 0 && (
          <div className="flex items-center gap-3 text-xs text-th-text-muted shrink-0">
            {/* Rows per page */}
            <div className="flex items-center gap-1.5">
              <span>Rows per page</span>
              <div className="relative">
                <select
                  value={pageSize}
                  onChange={(e) => handlePageSizeChange(Number(e.target.value))}
                  className="appearance-none pl-2 pr-5 py-1 rounded-md bg-th-inset-bg border border-th-border text-xs text-th-text-primary focus:outline-none focus:border-th-border-strong cursor-pointer"
                >
                  {PAGE_SIZE_OPTIONS.map((n) => (
                    <option key={n} value={n}>{n}</option>
                  ))}
                </select>
                <ChevronRight size={10} className="absolute right-1.5 top-1/2 -translate-y-1/2 rotate-90 text-th-text-muted pointer-events-none" />
              </div>
            </div>

            {/* Range + pager */}
            <span className="tabular-nums">{rangeStart}–{rangeEnd} of {total}</span>
            <div className="flex items-center gap-0.5">
              <button
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={page === 0}
                className="p-1 rounded hover:bg-th-surface-hover disabled:opacity-30 transition-colors"
                aria-label="Previous page"
              >
                <ChevronLeft size={14} />
              </button>
              <span className="px-1 tabular-nums">
                {page + 1} / {totalPages}
              </span>
              <button
                onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                disabled={page >= totalPages - 1}
                className="p-1 rounded hover:bg-th-surface-hover disabled:opacity-30 transition-colors"
                aria-label="Next page"
              >
                <ChevronRight size={14} />
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {loading ? (
          <div className="flex items-center justify-center h-full">
            <Loader2 size={22} className="animate-spin text-th-text-muted" />
          </div>
        ) : runs.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <Zap size={28} className="text-th-text-muted mb-3" />
            <p className="text-sm text-th-text-muted">
              No runs{statusFilter ? ` with status "${statusFilter}"` : ""}{timeRange !== "all" ? " in this time range" : " yet"}
            </p>
            {(timeRange !== "all" || statusFilter) && (
              <button
                onClick={() => { handleTimeRangeChange("all"); handleStatusChange(""); }}
                className="mt-3 text-xs text-blue-400 hover:text-blue-300 transition-colors"
              >
                Clear filters
              </button>
            )}
          </div>
        ) : (
          <div className="max-w-4xl mx-auto px-6 py-4">
            <div className="bg-th-card-bg border border-th-card-border rounded-xl overflow-hidden">
              {/* Column headers */}
              <div className="grid grid-cols-[20px_1fr_80px_70px_80px_80px] gap-3 px-4 py-2.5 border-b border-th-border/50">
                <div />
                <div className="text-[10px] font-semibold text-th-text-muted uppercase tracking-wider">Started</div>
                <div className="text-[10px] font-semibold text-th-text-muted uppercase tracking-wider">Duration</div>
                <div className="text-[10px] font-semibold text-th-text-muted uppercase tracking-wider">Msgs</div>
                <div className="text-[10px] font-semibold text-th-text-muted uppercase tracking-wider">Status</div>
                <div />
              </div>

              {/* Run rows */}
              <div className="divide-y divide-th-border/30">
                {runs.map((run) => (
                  <div key={run.id} className="hover:bg-th-surface-hover transition-colors">
                    <div className="grid grid-cols-[20px_1fr_80px_70px_80px_80px] gap-3 px-4 pt-3 pb-2 items-center">
                      <div className="flex items-center">
                        <StatusIcon status={run.status} />
                      </div>

                      <div className="text-xs text-th-text-secondary">
                        {new Date(run.started_at).toLocaleString(undefined, {
                          month: "short",
                          day: "numeric",
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </div>

                      <div className="text-xs text-th-text-muted tabular-nums">
                        {formatDuration(run)}
                      </div>

                      <div className="text-xs text-th-text-muted tabular-nums">
                        {run.message_count}
                      </div>

                      <div className="text-xs">
                        <span className="text-th-text-muted capitalize">{run.status}</span>
                      </div>

                      <div className="flex items-center justify-end gap-1 shrink-0">
                        {run.status !== "running" && (
                          <button
                            onClick={() => api.openTriggerRunFolder(id!, run.id).catch(() => {})}
                            className="p-1 rounded text-th-text-muted hover:text-th-text-secondary hover:bg-th-surface-hover transition-colors"
                            title="Open output folder"
                          >
                            <FolderOpen size={12} />
                          </button>
                        )}
                        {run.session_id && (
                          <button
                            onClick={() => navigate(`/runs/${run.session_id}`)}
                            className={`text-[11px] font-medium px-2 py-0.5 rounded transition-colors ${
                              run.status === "running"
                                ? "text-emerald-400 hover:text-emerald-300"
                                : "text-th-text-muted hover:text-th-text-primary hover:bg-th-surface-hover"
                            }`}
                          >
                            {run.status === "running" ? "Live" : "View"}
                          </button>
                        )}
                      </div>
                    </div>

                    {run.error && (
                      <div className="px-4 pb-2.5 pl-11">
                        <p className="text-[11px] text-red-400/80 bg-red-500/[0.06] border border-red-500/15 rounded-md px-3 py-1.5 leading-relaxed break-words">
                          {run.error}
                        </p>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Bottom pagination */}
            {totalPages > 1 && (
              <div className="flex items-center justify-center gap-2 mt-4 text-xs text-th-text-muted">
                <button
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                  disabled={page === 0}
                  className="flex items-center gap-1 px-3 py-1.5 rounded-lg border border-th-border hover:border-th-border-strong hover:text-th-text-secondary disabled:opacity-30 transition-all"
                >
                  <ChevronLeft size={13} /> Previous
                </button>
                <span className="px-2 tabular-nums">
                  Page {page + 1} of {totalPages}
                </span>
                <button
                  onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                  disabled={page >= totalPages - 1}
                  className="flex items-center gap-1 px-3 py-1.5 rounded-lg border border-th-border hover:border-th-border-strong hover:text-th-text-secondary disabled:opacity-30 transition-all"
                >
                  Next <ChevronRight size={13} />
                </button>
              </div>
            )}
            {/* Bottom range summary */}
            {total > 0 && (
              <p className="text-center text-[11px] text-th-text-muted/70 mt-2 tabular-nums">
                Showing {rangeStart}–{rangeEnd} of {total} run{total !== 1 ? "s" : ""}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
