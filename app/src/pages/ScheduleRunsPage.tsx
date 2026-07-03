import { Fragment, useCallback, useEffect, useState } from "react";
import { useNavigate, useParams, useLocation } from "react-router-dom";
import {
  Activity,
  AlertCircle,
  ArrowLeft,
  Ban,
  BarChart3,
  Calendar,
  CheckCircle,
  CheckCircle2,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Clock,
  Footprints,
  FolderOpen,
  Gauge,
  Loader2,
  Play,
  Square,
  XCircle,
  X,
  MoreVertical,
  ExternalLink,
  Radio,
  Rows2,
  Rows3,
  type LucideIcon,
} from "lucide-react";
import { api } from "../hooks/useApi";
import { usePolling } from "../hooks/usePolling";
import type { ScheduleRun, ScheduleRunStats, ScheduleSpec } from "../types";
import { Popover } from "../components/ui/Popover";
import { StatCard } from "../components/runs/StatCard";

const PAGE_SIZE_OPTIONS = [20, 50, 100];
const DEFAULT_PAGE_SIZE = 20;
const DENSITY_KEY = "runs.density";
const STATS_OPEN_KEY = "schedule.runs.statsOpen";

type Density = "comfortable" | "compact";

interface SavedScheduleFilters {
  status?: string;
  dateFrom?: string;
  dateTo?: string;
  page?: number;
  pageSize?: number;
}

const filtersKey = (id?: string) => `schedule.runs.filters.${id ?? ""}`;

function loadScheduleFilters(id?: string): SavedScheduleFilters {
  try {
    return JSON.parse(localStorage.getItem(filtersKey(id)) ?? "{}");
  } catch {
    return {};
  }
}

const TIME_FILTERS = [
  { label: "1d", days: 1 },
  { label: "1w", days: 7 },
  { label: "1mo", days: 30 },
  { label: "All", days: 0 },
] as const;

function toDateStr(date: Date): string {
  return date.toISOString().slice(0, 10);
}

function getQuickFilterDateFrom(days: number): string {
  if (days === 0) return "";
  const d = new Date();
  d.setDate(d.getDate() - days);
  return toDateStr(d);
}

function dateRangeIso(dateFrom: string, dateTo: string): { after?: string; before?: string } {
  const range: { after?: string; before?: string } = {};
  if (dateFrom) {
    const d = new Date(dateFrom);
    d.setHours(0, 0, 0, 0);
    range.after = d.toISOString();
  }
  if (dateTo) {
    const d = new Date(dateTo);
    d.setHours(23, 59, 59, 999);
    range.before = d.toISOString();
  }
  return range;
}

function detectActiveQuickFilter(dateFrom: string, dateTo: string): number | null {
  if (dateTo !== "") return null;
  if (dateFrom === "") return 0;
  for (const f of TIME_FILTERS) {
    if (f.days === 0) continue;
    if (dateFrom === getQuickFilterDateFrom(f.days)) return f.days;
  }
  return null;
}

type StatusFilter = "" | "running" | "success" | "error" | "cancelled";

const STATUS_OPTIONS: { value: StatusFilter; label: string }[] = [
  { value: "", label: "All statuses" },
  { value: "running", label: "Running" },
  { value: "success", label: "Success" },
  { value: "error", label: "Error" },
  { value: "cancelled", label: "Cancelled" },
];

function DateInput({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <div className="relative flex-1">
      <Calendar size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-th-text-muted pointer-events-none" />
      <input
        type="date"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full pl-7 pr-2 py-1.5 rounded-xl bg-th-inset-bg/70 border border-th-border text-sm text-th-text-primary focus:outline-none focus:border-blue-500/40 focus:ring-2 focus:ring-blue-500/10 transition-all cursor-pointer"
      />
    </div>
  );
}

function evalChipClasses(score: number): string {
  if (score >= 0.8) return "text-emerald-400 bg-emerald-500/10 border-emerald-500/20";
  if (score >= 0.5) return "text-amber-400 bg-amber-500/10 border-amber-500/20";
  return "text-red-400 bg-red-500/10 border-red-500/20";
}

function formatDurationMs(ms: number | null | undefined): string {
  if (ms == null) return "—";
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60_000);
  const s = Math.round((ms % 60_000) / 1000);
  return `${m}m ${s}s`;
}

function formatDuration(run: ScheduleRun): string {
  if (!run.finished_at) return run.status === "running" ? "running…" : "—";
  const ms = new Date(run.finished_at).getTime() - new Date(run.started_at).getTime();
  return formatDurationMs(ms);
}

function cronToHuman(cron: string): string {
  const map: Record<string, string> = {
    "0 9 * * *": "Daily at 9 am",
    "0 9 * * 1-5": "Weekdays at 9 am",
    "0 * * * *": "Every hour",
    "*/15 * * * *": "Every 15 min",
    "0 0 * * *": "Daily at midnight",
    "0 0 * * 1": "Weekly on Monday",
    "0 0 1 * *": "Monthly",
  };
  return map[cron] ?? cron;
}

function statusMeta(status: string): { Icon: LucideIcon; iconCls: string; textCls: string; label: string } {
  switch (status) {
    case "running":
      return { Icon: Loader2, iconCls: "text-emerald-400 animate-spin", textCls: "text-emerald-400", label: "Running" };
    case "success":
      return { Icon: CheckCircle, iconCls: "text-emerald-400", textCls: "text-emerald-400", label: "Success" };
    case "error":
      return { Icon: XCircle, iconCls: "text-red-400", textCls: "text-red-400", label: "Error" };
    case "cancelled":
      return { Icon: Ban, iconCls: "text-amber-400", textCls: "text-amber-400", label: "Cancelled" };
    default:
      return { Icon: AlertCircle, iconCls: "text-th-text-muted", textCls: "text-th-text-muted", label: status || "—" };
  }
}

// ---------------------------------------------------------------------------
// Row actions kebab
// ---------------------------------------------------------------------------

function RunActionsMenu({
  run,
  scheduleId,
  onView,
}: {
  run: ScheduleRun;
  scheduleId: string;
  onView: () => void;
}) {
  const itemCls =
    "flex w-full items-center gap-2.5 px-2.5 py-1.5 rounded-lg text-xs text-th-text-secondary hover:bg-th-surface-hover hover:text-th-text-primary transition-colors disabled:opacity-50 disabled:cursor-not-allowed";

  return (
    <Popover
      role="menu"
      align="right"
      panelClassName="w-48 p-1 overflow-hidden"
      trigger={({ toggle, open, ...aria }) => (
        <button
          type="button"
          onClick={(e) => { e.stopPropagation(); toggle(); }}
          className={`p-1 rounded-lg text-th-text-muted/50 hover:text-th-text-primary hover:bg-th-surface-hover transition-all duration-200 active:scale-90 ${
            open ? "opacity-100 bg-th-surface-hover text-th-text-primary" : "opacity-0 group-hover:opacity-100 focus:opacity-100"
          }`}
          title="Actions"
          aria-label="Run actions"
          {...aria}
        >
          <MoreVertical size={14} />
        </button>
      )}
    >
      {({ close }) => (
        <div onClick={(e) => e.stopPropagation()}>
          {run.session_id && (
            <button
              type="button"
              className={itemCls}
              onClick={() => { close(); onView(); }}
            >
              {run.status === "running" ? <Radio size={13} aria-hidden /> : <ExternalLink size={13} aria-hidden />}
              {run.status === "running" ? "View live" : "View run"}
            </button>
          )}
          {run.status !== "running" && (
            <button
              type="button"
              className={itemCls}
              onClick={() => { close(); api.openScheduleRunFolder(scheduleId, run.id).catch(() => {}); }}
            >
              <FolderOpen size={13} aria-hidden />
              Open output folder
            </button>
          )}
        </div>
      )}
    </Popover>
  );
}

export default function ScheduleRunsPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const location = useLocation();

  // Remember this schedule detail page so the sidebar "Schedules" link restores it.
  useEffect(() => {
    localStorage.setItem("schedules.lastDetailPath", location.pathname);
    return () => {
      // Clear when leaving so a deleted/non-existent schedule doesn't get restored.
      // We keep it while mounted so sibling navigations (e.g. opening a run) don't wipe it.
    };
  }, [location.pathname]);

  // Restore persisted page + filters for this schedule (mirrors the Runs page).
  const initialFilters = loadScheduleFilters(id);

  const [schedule, setSchedule] = useState<ScheduleSpec | null>(null);
  const [runs, setRuns] = useState<ScheduleRun[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(() => Math.max(0, initialFilters.page ?? 0));
  const [pageSize, setPageSize] = useState(() =>
    PAGE_SIZE_OPTIONS.includes(initialFilters.pageSize ?? DEFAULT_PAGE_SIZE)
      ? (initialFilters.pageSize ?? DEFAULT_PAGE_SIZE)
      : DEFAULT_PAGE_SIZE,
  );
  const [dateFrom, setDateFrom] = useState(() => initialFilters.dateFrom ?? "");
  const [dateTo, setDateTo] = useState(() => initialFilters.dateTo ?? "");
  const [statusFilter, setStatusFilter] = useState<StatusFilter>(() => (initialFilters.status ?? "") as StatusFilter);
  const [loading, setLoading] = useState(true);
  const [actioning, setActioning] = useState(false);
  const [expandedErrorId, setExpandedErrorId] = useState<string | null>(null);
  const [density, setDensity] = useState<Density>(
    () => (localStorage.getItem(DENSITY_KEY) === "compact" ? "compact" : "comfortable"),
  );
  const [statsOpen, setStatsOpen] = useState<boolean>(() => localStorage.getItem(STATS_OPEN_KEY) === "1");
  const [stats, setStats] = useState<ScheduleRunStats | null>(null);
  const [statsLoading, setStatsLoading] = useState(false);

  // Persist page + filters whenever they change, scoped to this schedule.
  useEffect(() => {
    localStorage.setItem(
      filtersKey(id),
      JSON.stringify({ status: statusFilter, dateFrom, dateTo, page, pageSize }),
    );
  }, [id, statusFilter, dateFrom, dateTo, page, pageSize]);

  const fetchRuns = useCallback(async () => {
    if (!id) return;
    try {
      const { after, before } = dateRangeIso(dateFrom, dateTo);
      const result = await api.getScheduleRuns(id, {
        limit: pageSize,
        offset: page * pageSize,
        after,
        before,
        status: statusFilter || undefined,
      });
      setRuns(result.runs);
      setTotal(result.total);
    } catch (e) {
      console.warn("[ScheduleRunsPage] fetch failed:", e);
    } finally {
      setLoading(false);
    }
  }, [id, page, pageSize, dateFrom, dateTo, statusFilter]);

  useEffect(() => {
    setLoading(true);
    fetchRuns();
  }, [fetchRuns]);

  // Fetch aggregate stats only while the stats strip is open; refresh on filter change.
  const fetchStats = useCallback(async () => {
    if (!id || !statsOpen) return;
    try {
      const { after, before } = dateRangeIso(dateFrom, dateTo);
      const data = await api.getScheduleRunStats(id, {
        after,
        before,
        status: statusFilter || undefined,
      });
      setStats(data);
    } catch (e) {
      console.warn("[ScheduleRunsPage] stats fetch failed:", e);
    } finally {
      setStatsLoading(false);
    }
  }, [id, statsOpen, dateFrom, dateTo, statusFilter]);

  useEffect(() => {
    if (statsOpen) {
      setStatsLoading(true);
      fetchStats();
    }
  }, [statsOpen, fetchStats]);

  useEffect(() => {
    if (!id) return;
    api
      .listSchedules()
      .then((schedules) => {
        const found = schedules.find((s) => s.id === id);
        if (found) setSchedule(found);
      })
      .catch(() => {});
  }, [id]);

  const isRunning =
    schedule?.last_status === "running" || runs.some((r) => r.status === "running");

  usePolling(fetchRuns, 3000, isRunning);
  usePolling(fetchStats, 3000, isRunning && statsOpen);

  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  const rangeStart = total === 0 ? 0 : page * pageSize + 1;
  const rangeEnd = Math.min((page + 1) * pageSize, total);

  const handleStatusChange = (s: string) => { setPage(0); setStatusFilter(s as StatusFilter); };
  const handleDateFromChange = (v: string) => { setPage(0); setDateFrom(v); };
  const handleDateToChange = (v: string) => { setPage(0); setDateTo(v); };
  const clearFilters = () => { setPage(0); setStatusFilter(""); setDateFrom(""); setDateTo(""); };
  const handleQuickFilter = (days: number) => { setPage(0); setDateFrom(getQuickFilterDateFrom(days)); setDateTo(""); };
  const handlePageSizeChange = (size: number) => { setPage(0); setPageSize(size); };

  const toggleDensity = () => {
    setDensity((d) => {
      const next = d === "compact" ? "comfortable" : "compact";
      localStorage.setItem(DENSITY_KEY, next);
      return next;
    });
  };

  const toggleStats = () => {
    setStatsOpen((s) => {
      const next = !s;
      localStorage.setItem(STATS_OPEN_KEY, next ? "1" : "0");
      return next;
    });
  };

  const activeFilters = [statusFilter, dateFrom, dateTo].filter(Boolean).length;
  const rowPad = density === "compact" ? "py-1.5" : "py-2.5";

  const handleRunNow = async () => {
    if (!id) return;
    setActioning(true);
    try {
      await api.runScheduleNow(id);
      fetchRuns();
    } catch { /* ignore */ }
    setActioning(false);
  };

  const handleStop = async () => {
    if (!id) return;
    try {
      await api.stopScheduleRun(id);
    } catch { /* ignore */ }
    fetchRuns();
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <header className="border-b border-th-border/70 px-6 py-3 shrink-0 bg-th-bg-secondary/80 backdrop-blur-xl space-y-2.5">
        {/* Top row: breadcrumb + title + tools */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate("/schedules")}
            className="flex items-center gap-1.5 text-th-text-muted hover:text-th-text-primary transition-colors text-sm shrink-0"
          >
            <ArrowLeft size={14} aria-hidden />
            Schedules
          </button>
          <span className="text-th-border">/</span>
          <div className="flex items-center gap-2 min-w-0">
            <Calendar size={14} className="text-th-text-muted shrink-0" aria-hidden />
            <h1 className="text-[19px] font-semibold tracking-tight text-th-text-primary truncate">{id}</h1>
            {schedule && (
              <span className="text-xs text-th-text-muted truncate hidden sm:block">
                · {schedule.agent_name || "General Purpose"} · {cronToHuman(schedule.cron_expression)}
              </span>
            )}
            {total > 0 && (
              <span className="text-xs text-th-text-muted tabular-nums shrink-0">· {total} run{total !== 1 ? "s" : ""}</span>
            )}
          </div>

          <div className="flex items-center gap-2 ml-auto shrink-0">
            {/* Stats toggle */}
            <button
              onClick={(e) => { e.stopPropagation(); toggleStats(); }}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium border transition-all duration-200 active:scale-[0.97] ${
                statsOpen
                  ? "bg-blue-500/10 border-blue-500/30 text-blue-400"
                  : "bg-th-inset-bg/70 border-th-border text-th-text-muted hover:text-th-text-secondary hover:border-th-border-strong"
              }`}
              title={statsOpen ? "Hide stats" : "Show stats"}
              aria-pressed={statsOpen}
            >
              <BarChart3 size={13} />
              Stats
              <ChevronDown size={11} className={`transition-transform duration-200 ${statsOpen ? "rotate-180" : ""}`} />
            </button>

            {/* Density segmented control */}
            <div className="inline-flex items-center p-0.5 rounded-xl bg-th-inset-bg/70 border border-th-border">
              {(["comfortable", "compact"] as const).map((d) => {
                const active = density === d;
                const DIcon = d === "comfortable" ? Rows3 : Rows2;
                return (
                  <button
                    key={d}
                    onClick={() => { if (!active) toggleDensity(); }}
                    className={`p-1.5 rounded-lg transition-all duration-200 ${
                      active
                        ? "bg-th-bg-secondary text-th-text-primary shadow-sm ring-1 ring-black/[0.04]"
                        : "text-th-text-muted hover:text-th-text-secondary"
                    }`}
                    title={d === "comfortable" ? "Comfortable rows" : "Compact rows"}
                    aria-label={`${d} rows`}
                    aria-pressed={active}
                  >
                    <DIcon size={14} />
                  </button>
                );
              })}
            </div>

            {isRunning ? (
              <button
                onClick={handleStop}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-red-500/30 text-xs font-medium text-red-400 hover:bg-red-500/10 transition-all duration-200 active:scale-[0.97]"
              >
                <Square size={12} aria-hidden />
                Stop
              </button>
            ) : (
              <button
                onClick={handleRunNow}
                disabled={actioning}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-xs font-medium text-white transition-all duration-200 active:scale-[0.97] disabled:opacity-50"
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

        {/* Toolbar row: status + date filters inline */}
        <div className="flex items-center gap-2 flex-wrap">
          {/* Status dropdown */}
          <div className="relative">
            <select
              value={statusFilter}
              onChange={(e) => handleStatusChange(e.target.value)}
              aria-label="Filter by status"
              className={`appearance-none pl-3 pr-8 py-1.5 rounded-xl border text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/10 transition-all cursor-pointer ${
                statusFilter
                  ? "bg-blue-500/10 border-blue-500/30 text-blue-400"
                  : "bg-th-inset-bg/70 border-th-border text-th-text-primary focus:border-blue-500/40"
              }`}
            >
              {STATUS_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
            <ChevronDown size={12} className="absolute right-2.5 top-1/2 -translate-y-1/2 text-th-text-muted pointer-events-none" />
          </div>

          {/* Quick date chips */}
          <div className="flex items-center p-0.5 rounded-xl bg-th-inset-bg/70 border border-th-border">
            {TIME_FILTERS.map((f) => {
              const active = detectActiveQuickFilter(dateFrom, dateTo) === f.days;
              return (
                <button
                  key={f.label}
                  onClick={() => handleQuickFilter(f.days)}
                  className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all duration-200 ${
                    active
                      ? "bg-th-bg-secondary text-th-text-primary shadow-sm ring-1 ring-black/[0.04]"
                      : "text-th-text-muted hover:text-th-text-secondary"
                  }`}
                >
                  {f.label}
                </button>
              );
            })}
          </div>

          {/* Custom date range inputs */}
          <div className="flex items-center gap-1.5">
            <DateInput value={dateFrom} onChange={handleDateFromChange} />
            <span className="text-th-text-muted text-xs">–</span>
            <DateInput value={dateTo} onChange={handleDateToChange} />
          </div>

          {/* Clear button — only when something is active */}
          {activeFilters > 0 && (
            <button
              onClick={clearFilters}
              className="flex items-center gap-1 px-2.5 py-1.5 rounded-xl text-xs text-th-text-muted hover:text-red-400 hover:bg-red-500/10 border border-transparent hover:border-red-500/20 transition-all duration-200 active:scale-[0.98]"
            >
              <X size={12} />
              Clear
            </button>
          )}
        </div>
      </header>

      {/* Stats strip */}
      {statsOpen && (
        <div className="shrink-0 border-b border-th-border/70 bg-th-bg-secondary/60 backdrop-blur-xl px-6 py-3.5 grid grid-cols-2 sm:grid-cols-5 gap-3 animate-slide-up">
          <StatCard
            label="Total runs"
            value={statsLoading && !stats ? "—" : (stats?.total ?? 0).toLocaleString()}
            icon={BarChart3}
            loading={statsLoading && !stats}
          />
          <StatCard
            label="Success rate"
            value={statsLoading && !stats ? "—" : `${stats?.success_rate ?? 0}%`}
            icon={CheckCircle2}
            iconClassName="text-emerald-400"
            loading={statsLoading && !stats}
          />
          <StatCard
            label="Avg duration"
            value={statsLoading && !stats ? "—" : formatDurationMs(stats?.avg_duration_ms)}
            icon={Clock}
            loading={statsLoading && !stats}
          />
          <StatCard
            label="Avg steps / run"
            value={statsLoading && !stats ? "—" : (stats?.avg_steps_per_run ?? 0).toLocaleString()}
            subValue={stats ? `${stats.total_steps.toLocaleString()} total` : undefined}
            icon={Footprints}
            loading={statsLoading && !stats}
          />
          <StatCard
            label="Running now"
            value={statsLoading && !stats ? "—" : (stats?.running_now ?? 0).toLocaleString()}
            icon={Activity}
            iconClassName={stats && stats.running_now > 0 ? "text-emerald-400" : "text-th-text-muted"}
            loading={statsLoading && !stats}
          />
        </div>
      )}

      {/* Table */}
      <div className="flex-1 overflow-auto">
        {loading ? (
          <div className="space-y-1.5 p-2">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="h-10 rounded-lg bg-th-inset-bg animate-pulse" />
            ))}
          </div>
        ) : runs.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-16 text-center">
            <Calendar size={26} className="text-th-text-muted/60 mb-3" />
            <p className="text-sm text-th-text-muted">
              No runs{activeFilters > 0 ? " match the current filters" : " yet"}
            </p>
            {activeFilters > 0 && (
              <button
                onClick={clearFilters}
                className="mt-3 text-xs text-blue-400 hover:text-blue-300 transition-colors"
              >
                Clear filters
              </button>
            )}
          </div>
        ) : (
          <table className="w-full text-sm border-collapse">
            <thead className="sticky top-0 z-10 bg-th-bg-secondary/80 backdrop-blur-xl">
              <tr className="border-b border-th-border">
                <th className="text-left text-[10px] font-semibold uppercase tracking-wider text-th-text-muted px-3 py-2 whitespace-nowrap">Status</th>
                <th className="text-left text-[10px] font-semibold uppercase tracking-wider text-th-text-muted px-3 py-2 whitespace-nowrap">Started</th>
                <th className="text-left text-[10px] font-semibold uppercase tracking-wider text-th-text-muted px-3 py-2 whitespace-nowrap">Duration</th>
                <th className="text-left text-[10px] font-semibold uppercase tracking-wider text-th-text-muted px-3 py-2 whitespace-nowrap">Msgs</th>
                <th className="text-left text-[10px] font-semibold uppercase tracking-wider text-th-text-muted px-3 py-2 whitespace-nowrap">Eval</th>
                <th className="px-2 py-2 w-px" aria-label="Actions" />
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => {
                const meta = statusMeta(run.status);
                const hasError = !!run.error;
                const isErrorExpanded = hasError && expandedErrorId === run.id;
                const clickable = !!run.session_id;
                return (
                  <Fragment key={run.id}>
                    <tr
                      className={`${isErrorExpanded ? "" : "border-b border-th-border/40"} hover:bg-th-surface-hover transition-colors group ${clickable ? "cursor-pointer" : ""}`}
                      onClick={() => { if (clickable) navigate(`/runs/${run.session_id}`); }}
                    >
                      <td className={`px-3 ${rowPad} whitespace-nowrap`}>
                        <div className="flex items-center gap-1.5">
                          <span className={`inline-flex items-center gap-1 font-medium text-[11px] ${meta.textCls}`}>
                            <meta.Icon size={11} className={meta.iconCls} aria-hidden />
                            {meta.label}
                          </span>
                          {hasError && (
                            <button
                              onClick={(e) => { e.stopPropagation(); setExpandedErrorId(isErrorExpanded ? null : run.id); }}
                              className={`flex items-center gap-0.5 rounded px-1 py-0.5 text-[10px] font-medium transition-colors ${
                                isErrorExpanded
                                  ? "text-red-400 bg-red-500/10"
                                  : "text-th-text-muted/60 hover:text-red-400 hover:bg-red-500/10"
                              }`}
                              title={isErrorExpanded ? "Hide error details" : "Show error details"}
                              aria-label={isErrorExpanded ? "Hide error details" : "Show error details"}
                              aria-expanded={isErrorExpanded}
                            >
                              <AlertCircle size={11} aria-hidden />
                              <ChevronDown size={11} aria-hidden className={`transition-transform ${isErrorExpanded ? "rotate-180" : ""}`} />
                            </button>
                          )}
                        </div>
                      </td>
                      <td className={`px-3 ${rowPad} whitespace-nowrap text-xs text-th-text-secondary`}>
                        {new Date(run.started_at).toLocaleString(undefined, {
                          month: "short",
                          day: "numeric",
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </td>
                      <td className={`px-3 ${rowPad} whitespace-nowrap text-xs text-th-text-muted tabular-nums`}>
                        {formatDuration(run)}
                      </td>
                      <td className={`px-3 ${rowPad} whitespace-nowrap text-xs text-th-text-muted tabular-nums`}>
                        {run.message_count}
                      </td>
                      <td className={`px-3 ${rowPad} whitespace-nowrap`}>
                        {run.eval_status === "done" && run.eval_overall_score != null ? (
                          <button
                            className={`inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-full border text-[10px] font-bold hover:opacity-80 transition-opacity ${evalChipClasses(run.eval_overall_score)}`}
                            title={`${run.eval_pass_count ?? 0}/${run.eval_total ?? 0} metrics passed — click to view`}
                            onClick={(e) => {
                              e.stopPropagation();
                              if (run.session_id) navigate(`/runs/${run.session_id}?tab=evaluation`);
                            }}
                          >
                            <Gauge size={10} aria-hidden />
                            {Math.round(run.eval_overall_score * 100)}%
                          </button>
                        ) : run.eval_status === "running" ? (
                          <span className="text-[10px] text-blue-400 animate-pulse">…</span>
                        ) : (
                          <span className="text-[10px] text-th-text-muted/30">—</span>
                        )}
                      </td>
                      <td className={`px-2 ${rowPad} whitespace-nowrap w-px`} onClick={(e) => e.stopPropagation()}>
                        <RunActionsMenu run={run} scheduleId={id!} onView={() => run.session_id && navigate(`/runs/${run.session_id}`)} />
                      </td>
                    </tr>
                    {isErrorExpanded && (
                      <tr className="border-b border-th-border/40">
                        <td colSpan={6} className="px-3 pb-2.5 pt-0">
                          <div className="flex items-start gap-2 text-[11px] text-red-400/90 bg-red-500/[0.06] border border-red-500/15 rounded-md px-3 py-2 leading-relaxed">
                            <AlertCircle size={12} className="shrink-0 mt-0.5" aria-hidden />
                            <p className="flex-1 min-w-0 break-words whitespace-pre-wrap">{run.error}</p>
                            {run.session_id && (
                              <button
                                onClick={() => navigate(`/runs/${run.session_id}`)}
                                className="shrink-0 text-[10px] font-medium text-red-400 hover:text-red-300 underline-offset-2 hover:underline"
                              >
                                View run
                              </button>
                            )}
                          </div>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* Pagination footer */}
      {(total > 0 || loading) && (
        <footer className="shrink-0 border-t border-th-border/70 px-6 py-2.5 bg-th-bg-secondary/80 backdrop-blur-xl flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5">
              <span className="text-xs text-th-text-muted">Rows per page</span>
              <div className="relative">
                <select
                  value={pageSize}
                  onChange={(e) => handlePageSizeChange(Number(e.target.value))}
                  className="appearance-none pl-2 pr-5 py-1 rounded-lg bg-th-inset-bg/70 border border-th-border text-xs text-th-text-primary focus:outline-none focus:border-blue-500/40 focus:ring-2 focus:ring-blue-500/10 transition-all cursor-pointer"
                >
                  {PAGE_SIZE_OPTIONS.map((n) => (
                    <option key={n} value={n}>{n}</option>
                  ))}
                </select>
                <ChevronDown size={10} className="absolute right-1.5 top-1/2 -translate-y-1/2 text-th-text-muted pointer-events-none" />
              </div>
            </div>
            {!loading && total > 0 && (
              <span className="text-xs text-th-text-muted tabular-nums">
                {rangeStart}–{rangeEnd} of {total}
              </span>
            )}
          </div>

          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0 || loading}
              className="p-1.5 rounded-lg text-th-text-muted hover:text-th-text-primary hover:bg-th-surface-hover disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-200 active:scale-95"
              aria-label="Previous page"
            >
              <ChevronLeft size={14} />
            </button>
            <span className="text-xs text-th-text-secondary px-1 tabular-nums">
              {page + 1} / {totalPages}
            </span>
            <button
              onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
              disabled={page >= totalPages - 1 || loading}
              className="p-1.5 rounded-lg text-th-text-muted hover:text-th-text-primary hover:bg-th-surface-hover disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-200 active:scale-95"
              aria-label="Next page"
            >
              <ChevronRight size={14} />
            </button>
          </div>
        </footer>
      )}
    </div>
  );
}
