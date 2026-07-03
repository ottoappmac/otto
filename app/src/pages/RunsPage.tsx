import { useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
  Search,
  X,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Trash2,
  Loader2,
  Calendar,
  Rows2,
  Rows3,
  BarChart3,
  CheckCircle2,
  Clock,
  Activity,
  Footprints,
} from "lucide-react";
import { api } from "../hooks/useApi";
import { usePolling } from "../hooks/usePolling";
import type { RunInfo, RunStats } from "../types";
import { RunTable, type SortKey, type SortDir, type Density } from "../components/runs/RunTable";
import { StatCard } from "../components/runs/StatCard";
const POLL_MS = 6_000;

const PAGE_SIZE_OPTIONS = [20, 50, 100];
const DEFAULT_PAGE_SIZE = 20;

const SORTABLE_KEYS: SortKey[] = ["started_at", "duration_ms", "tokens", "eval"];
const DENSITY_KEY = "runs.density";
const STATS_OPEN_KEY = "runs.statsOpen";

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

function detectActiveQuickFilter(dateFrom: string, dateTo: string): number | null {
  if (dateTo !== "") return null;
  if (dateFrom === "") return 0; // "All"
  for (const f of TIME_FILTERS) {
    if (f.days === 0) continue;
    const expected = getQuickFilterDateFrom(f.days);
    if (dateFrom === expected) return f.days;
  }
  return null;
}

function formatDurationMs(ms: number | null | undefined): string {
  if (ms == null) return "—";
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60_000);
  const s = Math.round((ms % 60_000) / 1000);
  return `${m}m ${s}s`;
}

const STATUS_OPTIONS = [
  { value: "", label: "All statuses" },
  { value: "running", label: "Running" },
  { value: "completed", label: "Completed" },
  { value: "error", label: "Error" },
  { value: "stopped", label: "Stopped" },
  { value: "awaiting_input", label: "Awaiting input" },
];

const SOURCE_OPTIONS = [
  { value: "", label: "All sources" },
  { value: "manual", label: "Manual" },
  { value: "schedule", label: "Schedule" },
  { value: "trigger", label: "Trigger" },
  { value: "ambient", label: "Ambient" },
  { value: "voice", label: "Voice" },
];


interface DateInputProps {
  value: string;
  onChange: (v: string) => void;
}

function DateInput({ value, onChange }: DateInputProps) {
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

const RUNS_FILTER_KEY = "runs.filters";

function saveFilters(params: URLSearchParams) {
  const keys = ["search", "status", "source", "date_from", "date_to", "order_by", "order", "page_size"];
  const saved: Record<string, string> = {};
  for (const k of keys) {
    const v = params.get(k);
    if (v) saved[k] = v;
  }
  localStorage.setItem(RUNS_FILTER_KEY, JSON.stringify(saved));
}

function loadFilters(): Record<string, string> {
  try {
    return JSON.parse(localStorage.getItem(RUNS_FILTER_KEY) ?? "{}");
  } catch {
    return {};
  }
}

export default function RunsPage() {
  const [searchParams, setSearchParams] = useSearchParams();

  // Skip the very first save (mount with empty URL) so we don't overwrite
  // persisted filters before the restore effect has a chance to apply them.
  const skipNextSaveRef = useRef(true);

  // On first mount with no URL params, restore saved filters from localStorage.
  useEffect(() => {
    if (searchParams.toString() === "") {
      const saved = loadFilters();
      if (Object.keys(saved).length > 0) {
        setSearchParams(new URLSearchParams(saved), { replace: true });
        return; // save effect will fire after the re-render with restored params
      }
    }
    // If we get here either params were already in the URL (external link) or
    // there were no saved filters — safe to start saving from the next change.
    skipNextSaveRef.current = false;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Persist filter state whenever URL params change, skipping the initial
  // empty-URL render to avoid wiping stored filters on mount.
  useEffect(() => {
    if (skipNextSaveRef.current) {
      skipNextSaveRef.current = false;
      return;
    }
    saveFilters(searchParams);
  }, [searchParams]);

  const search = searchParams.get("search") ?? "";
  const status = searchParams.get("status") ?? "";
  const source = searchParams.get("source") ?? "";
  const dateFrom = searchParams.get("date_from") ?? "";
  const dateTo = searchParams.get("date_to") ?? "";
  const rawOrderBy = searchParams.get("order_by") ?? "";
  const sortKey = (SORTABLE_KEYS.includes(rawOrderBy as SortKey) ? rawOrderBy : null) as SortKey | null;
  const sortDir: SortDir = searchParams.get("order") === "asc" ? "asc" : "desc";
  const page = Math.max(1, parseInt(searchParams.get("page") ?? "1", 10));
  const pageSize = (() => {
    const v = parseInt(searchParams.get("page_size") ?? "", 10);
    return PAGE_SIZE_OPTIONS.includes(v) ? v : DEFAULT_PAGE_SIZE;
  })();

  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  // Seed from saved filters so the search box isn't empty after restore.
  const [localSearch, setLocalSearch] = useState(() => search || loadFilters()["search"] || "");
  const [clearingAll, setClearingAll] = useState(false);
  const [confirmClearAll, setConfirmClearAll] = useState(false);

  const [density, setDensity] = useState<Density>(
    () => (localStorage.getItem(DENSITY_KEY) === "compact" ? "compact" : "comfortable"),
  );
  const [statsOpen, setStatsOpen] = useState<boolean>(() => localStorage.getItem(STATS_OPEN_KEY) === "1");
  const [stats, setStats] = useState<RunStats | null>(null);
  const [statsLoading, setStatsLoading] = useState(false);

  const setParam = (key: string, value: string) => {
    setSearchParams((prev) => {
      const next = new URLSearchParams(prev);
      if (value) next.set(key, value);
      else next.delete(key);
      // Reset to page 1 when a filter changes
      if (key !== "page") next.delete("page");
      return next;
    });
  };

  const setMultiParams = (updates: Record<string, string>) => {
    setSearchParams((prev) => {
      const next = new URLSearchParams(prev);
      for (const [k, v] of Object.entries(updates)) {
        if (v) next.set(k, v);
        else next.delete(k);
      }
      return next;
    });
  };

  const handleSort = (key: SortKey) => {
    setSearchParams((prev) => {
      const next = new URLSearchParams(prev);
      next.delete("page");
      const curKey = next.get("order_by");
      const curOrder = next.get("order");
      if (curKey !== key) {
        next.set("order_by", key);
        next.set("order", "desc");
      } else if (curOrder !== "asc") {
        next.set("order", "asc");
      } else {
        next.delete("order_by");
        next.delete("order");
      }
      return next;
    });
  };

  const fetchRuns = useCallback(async () => {
    try {
      const data = await api.listRuns({
        search: search || undefined,
        status: status || undefined,
        source: source || undefined,
        date_from: dateFrom || undefined,
        date_to: dateTo || undefined,
        order_by: sortKey || undefined,
        order: sortKey ? sortDir : undefined,
        limit: pageSize,
        offset: (page - 1) * pageSize,
      });
      setRuns(data.runs);
      setTotal(data.total);
    } catch (e) {
      console.warn("[RunsPage] fetch failed:", e);
    } finally {
      setLoading(false);
    }
  }, [search, status, source, dateFrom, dateTo, sortKey, sortDir, page, pageSize]);

  useEffect(() => {
    setLoading(true);
    fetchRuns();
  }, [fetchRuns]);

  usePolling(fetchRuns, POLL_MS);

  // Fetch aggregate stats only while the stats strip is open; refresh on filter change.
  const fetchStats = useCallback(async () => {
    if (!statsOpen) return;
    try {
      const period = dateFrom || dateTo ? "custom" : "all";
      const data = await api.getRunStats(
        period,
        dateFrom || undefined,
        dateTo || undefined,
        search || undefined,
        status || undefined,
        source || undefined,
      );
      setStats(data);
    } catch (e) {
      console.warn("[RunsPage] stats fetch failed:", e);
    } finally {
      setStatsLoading(false);
    }
  }, [statsOpen, dateFrom, dateTo, search, status, source]);

  useEffect(() => {
    if (!statsOpen) return;
    setStatsLoading(true);
    fetchStats();
  }, [statsOpen, fetchStats]);

  usePolling(fetchStats, POLL_MS);

  // Keep the search box in sync when the URL param changes externally
  // (e.g. after filter restore on mount, or when "Clear all filters" fires).
  useEffect(() => {
    setLocalSearch(search);
  // We intentionally only react to the URL param, not localSearch, to avoid loops.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [search]);

  // Debounce search input to URL param
  useEffect(() => {
    const t = setTimeout(() => {
      setParam("search", localSearch);
    }, 300);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [localSearch]);

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

  const handleDeleteRun = async (run: RunInfo) => {
    const sessionId = run.session_id ?? run.id;
    await api.closeSession(sessionId);
    setRuns((prev) => prev.filter((r) => (r.session_id ?? r.id) !== sessionId));
    setTotal((t) => Math.max(0, t - 1));
  };

  const handleClearAll = async () => {
    if (!confirmClearAll) {
      setConfirmClearAll(true);
      return;
    }
    setClearingAll(true);
    setConfirmClearAll(false);
    try {
      await api.clearAllSessions();
      setRuns([]);
      setTotal(0);
    } catch (e) {
      console.warn("[RunsPage] clear all failed:", e);
    } finally {
      setClearingAll(false);
    }
  };

  const clearFilters = () => {
    setLocalSearch("");
    localStorage.removeItem(RUNS_FILTER_KEY);
    setSearchParams((prev) => {
      const next = new URLSearchParams();
      // Preserve sort + pagination prefs; only drop filters + search.
      const order_by = prev.get("order_by");
      const order = prev.get("order");
      const page_size = prev.get("page_size");
      if (order_by) next.set("order_by", order_by);
      if (order) next.set("order", order);
      if (page_size) next.set("page_size", page_size);
      return next;
    });
  };

  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  const activeFilters = [status, source, dateFrom, dateTo].filter(Boolean).length;
  const runningCount = runs.filter((r) => r.status === "running").length;

  const rangeStart = total === 0 ? 0 : (page - 1) * pageSize + 1;
  const rangeEnd = Math.min(page * pageSize, total);

  return (
    <div className="h-full flex flex-col" onClick={() => setConfirmClearAll(false)}>
      {/* Header */}
      <header className="border-b border-th-border/70 px-6 py-3 shrink-0 bg-th-bg-secondary/80 backdrop-blur-xl space-y-2.5">
        {/* Top row: title + tools */}
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 shrink-0">
            <h1 className="text-[19px] font-semibold tracking-tight text-th-text-primary">Runs</h1>
            {!loading && total > 0 && (
              <span className="text-xs text-th-text-muted tabular-nums">{total} total</span>
            )}
            {runningCount > 0 && (
              <span className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-[11px] font-semibold text-emerald-400">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                {runningCount} running
              </span>
            )}
          </div>

          <div className="flex items-center gap-2 shrink-0">
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
                    onClick={(e) => { e.stopPropagation(); if (!active) toggleDensity(); }}
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

            {/* Delete all */}
            {total > 0 && (
              <button
                onClick={(e) => { e.stopPropagation(); handleClearAll(); }}
                disabled={clearingAll}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium border transition-all duration-200 active:scale-[0.97] ${
                  confirmClearAll
                    ? "bg-red-500/15 border-red-500/40 text-red-400 hover:bg-red-500/25"
                    : "bg-th-inset-bg/70 border-th-border text-th-text-muted hover:text-red-400 hover:border-red-500/30 hover:bg-red-500/10"
                }`}
                title="Delete all sessions and runs"
              >
                {clearingAll ? (
                  <Loader2 size={12} className="animate-spin" />
                ) : (
                  <Trash2 size={12} />
                )}
                {confirmClearAll ? "Confirm delete all?" : "Delete all"}
              </button>
            )}
          </div>
        </div>

        {/* Toolbar row: search + status + source + date — all inline */}
        <div className="flex items-center gap-2 flex-wrap">
          {/* Search */}
          <div className="relative min-w-[160px] max-w-xs flex-1">
            <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-th-text-muted pointer-events-none" />
            <input
              type="text"
              value={localSearch}
              onChange={(e) => setLocalSearch(e.target.value)}
              placeholder="Search title or agent…"
              className="w-full pl-8 pr-7 py-1.5 rounded-xl bg-th-inset-bg/70 border border-th-border text-sm text-th-text-primary placeholder:text-th-text-muted focus:outline-none focus:border-blue-500/40 focus:ring-2 focus:ring-blue-500/10 transition-all"
            />
            {localSearch && (
              <button
                onClick={() => { setLocalSearch(""); setParam("search", ""); }}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-th-text-muted hover:text-th-text-secondary"
              >
                <X size={12} />
              </button>
            )}
          </div>

          {/* Status dropdown */}
          <div className="relative">
            <select
              value={status}
              onChange={(e) => setParam("status", e.target.value)}
              aria-label="Filter by status"
              className={`appearance-none pl-3 pr-8 py-1.5 rounded-xl border text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/10 transition-all cursor-pointer ${
                status
                  ? "bg-blue-500/10 border-blue-500/30 text-blue-400"
                  : "bg-th-inset-bg/70 border-th-border text-th-text-primary focus:border-blue-500/40"
              }`}
            >
              {STATUS_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
            <ChevronDown size={12} className="absolute right-2.5 top-1/2 -translate-y-1/2 text-th-text-muted pointer-events-none" />
          </div>

          {/* Source dropdown */}
          <div className="relative">
            <select
              value={source}
              onChange={(e) => setParam("source", e.target.value)}
              aria-label="Filter by source"
              className={`appearance-none pl-3 pr-8 py-1.5 rounded-xl border text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/10 transition-all cursor-pointer ${
                source
                  ? "bg-blue-500/10 border-blue-500/30 text-blue-400"
                  : "bg-th-inset-bg/70 border-th-border text-th-text-primary focus:border-blue-500/40"
              }`}
            >
              {SOURCE_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
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
                  onClick={() => setMultiParams({ date_from: getQuickFilterDateFrom(f.days), date_to: "", page: "" })}
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
            <DateInput value={dateFrom} onChange={(v) => setParam("date_from", v)} />
            <span className="text-th-text-muted text-xs">–</span>
            <DateInput value={dateTo} onChange={(v) => setParam("date_to", v)} />
          </div>

          {/* Clear — only when something is active */}
          {(activeFilters > 0 || search) && (
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
            value={statsLoading && !stats ? "—" : (stats?.total_period ?? 0).toLocaleString()}
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
        <RunTable
          runs={runs}
          loading={loading}
          density={density}
          sortKey={sortKey}
          sortDir={sortDir}
          onSort={handleSort}
          onDelete={handleDeleteRun}
          onRunAgain={fetchRuns}
          emptyMessage={
            activeFilters > 0 || search
              ? "No runs match the current filters"
              : "No runs yet — start a chat and your runs will appear here"
          }
        />
      </div>

      {/* Pagination footer */}
      {(total > 0 || loading) && (
        <footer className="shrink-0 border-t border-th-border/70 px-6 py-2.5 bg-th-bg-secondary/80 backdrop-blur-xl flex items-center justify-between gap-4">
          {/* Rows-per-page + range info */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5">
              <span className="text-xs text-th-text-muted">Rows per page</span>
              <div className="relative">
                <select
                  value={pageSize}
                  onChange={(e) => setMultiParams({ page_size: e.target.value, page: "1" })}
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
              <span className="text-xs text-th-text-muted">
                {rangeStart}–{rangeEnd} of {total}
              </span>
            )}
          </div>

          {/* Page controls */}
          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setParam("page", String(page - 1))}
              disabled={page <= 1 || loading}
              className="p-1.5 rounded-lg text-th-text-muted hover:text-th-text-primary hover:bg-th-surface-hover disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-200 active:scale-95"
              aria-label="Previous page"
            >
              <ChevronLeft size={14} />
            </button>
            <span className="text-xs text-th-text-secondary px-1 tabular-nums">
              {page} / {totalPages}
            </span>
            <button
              onClick={() => setParam("page", String(page + 1))}
              disabled={page >= totalPages || loading}
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
