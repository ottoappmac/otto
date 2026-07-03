import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Activity,
  CheckCircle2,
  Clock,
  DollarSign,
  Play,
  TrendingUp,
  ExternalLink,
  Square,
  CalendarDays,
  MessageCircleQuestion,
  Coins,
  Gauge,
  Zap,
  Database,
  LayoutGrid,
  Footprints,
  Loader2,
} from "lucide-react";
import { api } from "../hooks/useApi";
import { usePolling } from "../hooks/usePolling";
import { formatRelativeTime } from "../utils/formatRelativeTime";
import type { RunStats, RunInfo } from "../types";
import { StatCard } from "../components/runs/StatCard";
import { BreakdownBar } from "../components/runs/BreakdownBar";
import { RunsOverTimeChart } from "../components/runs/RunsOverTimeChart";
import { EvalScoreChart } from "../components/runs/EvalScoreChart";
import { Disclosure } from "../components/ui/Disclosure";
import { getSourceIcon, getSourceLabel } from "../utils/entityIcons";

const POLL_MS = 8_000;

type Period = "24h" | "7d" | "30d" | "all" | "custom";

const PERIOD_OPTIONS: { value: Period; label: string }[] = [
  { value: "24h", label: "Day" },
  { value: "7d", label: "Week" },
  { value: "30d", label: "Month" },
  { value: "all", label: "All time" },
  { value: "custom", label: "Custom" },
];

function formatDuration(ms: number | null | undefined): string {
  if (ms == null) return "—";
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60_000);
  const s = Math.round((ms % 60_000) / 1000);
  return `${m}m ${s}s`;
}

function formatCost(usd: number): string {
  if (!usd) return "$0.00";
  if (usd < 0.001) return "<$0.001";
  return `$${usd.toFixed(3)}`;
}

function formatTokens(n: number | null | undefined): string {
  if (!n) return "0";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

// Live, ticking elapsed time since a start timestamp.
function formatElapsed(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}:${String(s).padStart(2, "0")}`;
  return `${s}s`;
}

// A run that has been going this long is flagged amber as "long-running".
const LONG_RUNNING_MS = 10 * 60 * 1000;

// ---------------------------------------------------------------------------
// Live run card
// ---------------------------------------------------------------------------

function LiveRunCard({ run, now, onChanged }: { run: RunInfo; now: number; onChanged?: () => void }) {
  const navigate = useNavigate();
  const { Icon: SrcIcon, className: srcCls } = getSourceIcon(run.trigger_source);
  const awaiting = run.status === "awaiting_input";
  const sid = run.session_id ?? run.id;
  const [stopping, setStopping] = useState(false);

  const elapsedMs = now - new Date(run.started_at).getTime();
  const longRunning = !awaiting && elapsedMs > LONG_RUNNING_MS;
  const lastTool = run.tools_used.length > 0 ? run.tools_used[run.tools_used.length - 1] : null;

  const handleStop = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (stopping) return;
    setStopping(true);
    try {
      await api.stopSession(sid);
      onChanged?.();
    } catch (err) {
      console.warn("[Dashboard] stop failed:", err);
    } finally {
      setStopping(false);
    }
  };

  return (
    <div
      className={`relative flex flex-col bg-th-card-bg border rounded-2xl overflow-hidden cursor-pointer transition-all duration-200 shadow-sm shadow-black/[0.03] hover:shadow-md hover:shadow-black/[0.06] group ${
        awaiting
          ? "border-amber-500/30 hover:border-amber-500/50"
          : longRunning
            ? "border-amber-500/30 hover:border-amber-500/50"
            : "border-emerald-500/20 hover:border-emerald-500/40"
      }`}
      onClick={() => navigate(awaiting ? `/chat/${sid}` : `/runs/${sid}`)}
    >
      {/* heartbeat sweep */}
      <div className="absolute top-0 inset-x-0 h-0.5 overflow-hidden">
        <div
          className={`h-full w-1/3 animate-live-sweep bg-gradient-to-r from-transparent to-transparent ${
            awaiting ? "via-amber-400/70" : "via-emerald-400/70"
          }`}
        />
      </div>
      <div className="px-3.5 pt-3.5 pb-3">
        <div className="flex items-center gap-2 mb-2">
          {awaiting ? (
            <>
              <MessageCircleQuestion size={12} className="text-amber-400 animate-pulse shrink-0" aria-hidden />
              <span className="text-[11px] font-semibold text-amber-400">Needs feedback</span>
            </>
          ) : (
            <>
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse shrink-0" />
              <span className="text-[11px] font-semibold text-emerald-400">Running</span>
            </>
          )}
          <span className="ml-auto flex items-center gap-1 text-[10px] text-th-text-muted shrink-0">
            <SrcIcon size={10} className={srcCls} aria-hidden />
            {getSourceLabel(run.trigger_source)}
          </span>
        </div>
        <p className="text-sm font-semibold text-th-text-primary truncate group-hover:text-blue-400 transition-colors">
          {run.title || "Untitled"}
        </p>
        <p className="text-[11px] text-th-text-tertiary mt-0.5 truncate">
          {run.agent_name || "General Purpose"}
        </p>
        <div className="mt-2 flex items-center gap-2 text-[10px] text-th-text-muted">
          {awaiting ? (
            <>
              <span className="tabular-nums">{run.message_count} msgs</span>
              <span>·</span>
              <span>{formatRelativeTime(run.started_at)}</span>
            </>
          ) : (
            <>
              <span className="inline-flex items-center gap-1 tabular-nums" title="Agent steps so far">
                <Footprints size={9} aria-hidden />
                {run.step_count}
              </span>
              {lastTool && (
                <>
                  <span>·</span>
                  <span className="truncate max-w-[120px] font-mono" title={`Last tool: ${lastTool}`}>{lastTool}</span>
                </>
              )}
              <span
                className={`ml-auto inline-flex items-center gap-1 tabular-nums shrink-0 ${longRunning ? "text-amber-400 font-medium" : ""}`}
                title={longRunning
                  ? `Running for a while — started ${new Date(run.started_at).toLocaleTimeString()}`
                  : `Started ${new Date(run.started_at).toLocaleTimeString()}`}
              >
                <Clock size={9} aria-hidden />
                {formatElapsed(elapsedMs)}
              </span>
            </>
          )}
        </div>
      </div>
      <div className="px-3 py-2 border-t border-th-border/40 bg-th-inset-bg/30 flex items-center justify-between gap-2">
        <button
          onClick={(e) => { e.stopPropagation(); navigate(`/chat/${sid}`); }}
          className={`text-[10px] flex items-center gap-1 ${
            awaiting ? "text-amber-400 hover:text-amber-300 font-semibold" : "text-blue-400 hover:text-blue-300"
          }`}
        >
          <ExternalLink size={10} aria-hidden />
          {awaiting ? "Reply now" : "Open chat"}
        </button>
        {!awaiting && (
          <button
            onClick={handleStop}
            disabled={stopping}
            className="text-[10px] flex items-center gap-1 text-th-text-muted hover:text-orange-400 transition-colors disabled:opacity-50"
            title="Stop this run"
            aria-label="Stop this run"
          >
            {stopping ? <Loader2 size={10} className="animate-spin" aria-hidden /> : <Square size={10} aria-hidden />}
            Stop
          </button>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Chart card shell
// ---------------------------------------------------------------------------

function ChartCard({
  title,
  icon: Icon,
  iconClassName,
  legend,
  children,
}: {
  title: string;
  icon: typeof TrendingUp;
  iconClassName?: string;
  legend?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="bg-th-card-bg border border-th-card-border rounded-2xl p-4 shadow-sm shadow-black/[0.03]">
      <div className="flex items-center justify-between mb-4 gap-2">
        <h2 className="text-sm font-semibold tracking-tight text-th-text-primary flex items-center gap-1.5">
          <Icon size={14} className={iconClassName} aria-hidden />
          {title}
        </h2>
        {legend}
      </div>
      <div style={{ height: 190 }}>{children}</div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// DashboardPage
// ---------------------------------------------------------------------------

export default function DashboardPage() {
  const navigate = useNavigate();
  const [stats, setStats] = useState<RunStats | null>(null);
  const [liveRuns, setLiveRuns] = useState<RunInfo[]>([]);
  const [awaitingRuns, setAwaitingRuns] = useState<RunInfo[]>([]);
  const [period, setPeriod] = useState<Period>(() => {
    const saved = localStorage.getItem("dashboard.period");
    return (saved as Period) ?? "7d";
  });
  const [loading, setLoading] = useState(true);
  // Drives live, ticking elapsed timers on running cards.
  const [now, setNow] = useState(() => Date.now());

  // Custom date range state
  const today = new Date().toISOString().slice(0, 10);
  const [customFrom, setCustomFrom] = useState<string>(() => {
    const saved = localStorage.getItem("dashboard.customFrom");
    if (saved) return saved;
    const d = new Date();
    d.setDate(d.getDate() - 30);
    return d.toISOString().slice(0, 10);
  });
  const [customTo, setCustomTo] = useState<string>(() => {
    return localStorage.getItem("dashboard.customTo") ?? today;
  });
  // Track applied custom range to avoid re-fetching on every keystroke
  const appliedCustomRef = useRef({ from: customFrom, to: customTo });

  useEffect(() => { localStorage.setItem("dashboard.period", period); }, [period]);
  useEffect(() => { localStorage.setItem("dashboard.customFrom", customFrom); }, [customFrom]);
  useEffect(() => { localStorage.setItem("dashboard.customTo", customTo); }, [customTo]);

  const fetchData = useCallback(async () => {
    try {
      const dateFrom = period === "custom" ? appliedCustomRef.current.from : undefined;
      const dateTo = period === "custom" ? appliedCustomRef.current.to : undefined;
      const [statsData, runsData, awaitingData] = await Promise.all([
        api.getRunStats(period, dateFrom, dateTo),
        api.listRuns({ status: "running", limit: 12 }),
        api.listRuns({ status: "awaiting_input", limit: 12 }),
      ]);
      setStats(statsData);
      setLiveRuns(runsData.runs);
      setAwaitingRuns(awaitingData.runs);
    } catch (e) {
      console.warn("[Dashboard] fetch failed:", e);
    } finally {
      setLoading(false);
    }
  }, [period]);

  useEffect(() => {
    setLoading(true);
    fetchData();
  }, [fetchData]);

  usePolling(fetchData, POLL_MS);

  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  const kpiLoading = loading && !stats;
  const runningNow = stats?.running_now ?? liveRuns.length;
  const totalPeriod = stats?.total_period ?? 0;
  const successRate = stats?.success_rate ?? 0;
  const avgDuration = stats?.avg_duration_ms ?? null;
  const totalCost = stats?.total_cost_usd ?? 0;
  const totalInputTokens = stats?.total_input_tokens ?? 0;
  const totalOutputTokens = stats?.total_output_tokens ?? 0;
  const avgPrefillTps = stats?.avg_prefill_tps ?? null;
  const avgGenerationTps = stats?.avg_generation_tps ?? null;
  const cacheHitRatio = stats?.cache_hit_ratio ?? null;
  const evalCount = stats?.eval_count ?? 0;
  const evalAvgScore = stats?.eval_avg_score ?? null;
  const evalPassRate = stats?.eval_pass_rate ?? null;
  const evalScorePct = evalAvgScore != null ? Math.round(evalAvgScore * 100) : null;
  const evalColor =
    evalScorePct == null
      ? "text-th-text-muted"
      : evalScorePct >= 80
        ? "text-emerald-400"
        : evalScorePct >= 50
          ? "text-amber-400"
          : "text-red-400";

  // Merge awaiting + running into one prioritized activity feed (awaiting first).
  const activity = [...awaitingRuns, ...liveRuns];
  const periodLabel = PERIOD_OPTIONS.find((o) => o.value === period)?.label ?? period;

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Header */}
      <header className="border-b border-th-border/70 px-6 py-4 shrink-0 bg-th-bg-secondary/80 backdrop-blur-xl">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <h1 className="text-[19px] font-semibold tracking-tight text-th-text-primary">Overview</h1>
            {runningNow > 0 && (
              <span className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-[11px] font-semibold text-emerald-400">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                {runningNow} running
              </span>
            )}
            {awaitingRuns.length > 0 && (
              <button
                onClick={() => navigate("/runs?status=awaiting_input")}
                className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-amber-500/10 border border-amber-500/20 text-[11px] font-semibold text-amber-400 hover:bg-amber-500/20 transition-colors"
              >
                <MessageCircleQuestion size={11} className="animate-pulse" aria-hidden />
                {awaitingRuns.length} need{awaitingRuns.length === 1 ? "s" : ""} feedback
              </button>
            )}
          </div>
          <div className="flex items-center gap-2">
            <div className="inline-flex items-center p-0.5 rounded-xl bg-th-inset-bg/70 border border-th-border">
              {PERIOD_OPTIONS.map(({ value, label }) => (
                <button
                  key={value}
                  onClick={() => setPeriod(value)}
                  className={`px-3 py-1 rounded-lg text-xs font-medium transition-all duration-200 flex items-center gap-1 ${
                    period === value
                      ? "bg-th-bg-secondary text-th-text-primary shadow-sm ring-1 ring-black/[0.04]"
                      : "text-th-text-muted hover:text-th-text-secondary"
                  }`}
                >
                  {value === "custom" && <CalendarDays size={10} aria-hidden />}
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>
        {/* Custom date range inputs */}
        {period === "custom" && (
          <div className="mt-3 flex items-center gap-2">
            <input
              type="date"
              value={customFrom}
              max={customTo}
              onChange={(e) => setCustomFrom(e.target.value)}
              className="px-2.5 py-1 rounded-xl bg-th-inset-bg/70 border border-th-border text-xs text-th-text-primary focus:outline-none focus:border-blue-500/40 focus:ring-2 focus:ring-blue-500/10 transition-all"
            />
            <span className="text-xs text-th-text-muted">to</span>
            <input
              type="date"
              value={customTo}
              min={customFrom}
              max={today}
              onChange={(e) => setCustomTo(e.target.value)}
              className="px-2.5 py-1 rounded-xl bg-th-inset-bg/70 border border-th-border text-xs text-th-text-primary focus:outline-none focus:border-blue-500/40 focus:ring-2 focus:ring-blue-500/10 transition-all"
            />
            <button
              onClick={() => {
                appliedCustomRef.current = { from: customFrom, to: customTo };
                setLoading(true);
                fetchData();
              }}
              className="px-3 py-1 rounded-xl bg-blue-600 hover:bg-blue-500 text-xs font-medium text-white transition-all duration-200 active:scale-[0.97]"
            >
              Apply
            </button>
          </div>
        )}
      </header>

      <div className="flex-1 overflow-y-auto px-6 py-5 space-y-6 animate-fade-in">

        {/* Hero KPI row */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <StatCard
            label="Running now"
            value={kpiLoading ? "—" : runningNow}
            icon={Play}
            iconClassName="text-emerald-400"
            loading={kpiLoading}
          />
          <StatCard
            label={`Runs (${periodLabel})`}
            value={kpiLoading ? "—" : totalPeriod}
            icon={Activity}
            iconClassName="text-blue-400"
            loading={kpiLoading}
          />
          <StatCard
            label="Success rate"
            value={kpiLoading ? "—" : `${successRate}%`}
            icon={CheckCircle2}
            iconClassName={successRate >= 90 ? "text-emerald-400" : successRate >= 70 ? "text-amber-400" : "text-red-400"}
            loading={kpiLoading}
          />
          <StatCard
            label="Avg duration"
            value={kpiLoading ? "—" : formatDuration(avgDuration)}
            subValue={totalCost > 0 ? `Est. cost ${formatCost(totalCost)}` : undefined}
            icon={Clock}
            iconClassName="text-th-text-muted"
            loading={kpiLoading}
          />
        </div>

        {/* Live activity — awaiting feedback first, then running */}
        {activity.length > 0 && (
          <div>
            <div className="flex items-center justify-between mb-2.5">
              <h2 className="text-sm font-semibold tracking-tight text-th-text-primary flex items-center gap-1.5">
                <Activity size={14} className="text-emerald-400" aria-hidden />
                Live activity
                <span className="text-xs text-th-text-muted tabular-nums">{activity.length}</span>
              </h2>
              <button
                onClick={() => navigate("/runs?status=running")}
                className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
              >
                See all →
              </button>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-2.5">
              {activity.map((run) => <LiveRunCard key={run.id} run={run} now={now} onChanged={fetchData} />)}
            </div>
          </div>
        )}

        {/* Primary charts, side-by-side */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
          <ChartCard
            title="Runs over time"
            icon={TrendingUp}
            iconClassName="text-blue-400"
            legend={
              <div className="flex items-center gap-3 text-[10px] text-th-text-muted">
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-emerald-400/70" />Completed</span>
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-red-400/70" />Error</span>
              </div>
            }
          >
            {stats ? (
              <RunsOverTimeChart data={stats.time_series} period={period} />
            ) : (
              <div className="h-full bg-th-inset-bg rounded-lg animate-pulse" />
            )}
          </ChartCard>

          <ChartCard
            title="Evaluation performance"
            icon={Gauge}
            iconClassName="text-violet-400"
            legend={
              <div className="flex items-center gap-3 text-[10px] text-th-text-muted">
                {evalScorePct != null && (
                  <span className="flex items-center gap-1.5">
                    <span className={`text-sm font-bold ${evalColor}`}>{evalScorePct}%</span>
                    <span>avg</span>
                  </span>
                )}
                {evalPassRate != null && (
                  <span className="hidden sm:inline">{evalPassRate}% passed</span>
                )}
                <span className="tabular-nums">{evalCount} evaluated</span>
              </div>
            }
          >
            {stats ? (
              <EvalScoreChart data={stats.time_series} period={period} />
            ) : (
              <div className="h-full bg-th-inset-bg rounded-lg animate-pulse" />
            )}
          </ChartCard>
        </div>

        {/* Performance (tokens + throughput) — collapsed by default */}
        <Disclosure title="Performance" storageKey="performance" icon={Zap} iconClassName="text-violet-400">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <StatCard
              label="Tokens"
              value={kpiLoading ? "—" : `${formatTokens(totalInputTokens)} / ${formatTokens(totalOutputTokens)}`}
              subValue="in / out"
              icon={Coins}
              iconClassName="text-amber-400"
              loading={kpiLoading}
            />
            {avgPrefillTps != null && (
              <StatCard
                label="TIPS (prefill)"
                value={`${avgPrefillTps.toFixed(0)} t/s`}
                icon={Gauge}
                iconClassName="text-sky-400"
              />
            )}
            {avgGenerationTps != null && (
              <StatCard
                label="TOPS (generation)"
                value={`${avgGenerationTps.toFixed(0)} t/s`}
                icon={Zap}
                iconClassName="text-violet-400"
              />
            )}
            {cacheHitRatio != null && (
              <StatCard
                label="KV cache hit"
                value={`${Math.round(cacheHitRatio * 100)}%`}
                icon={Database}
                iconClassName="text-emerald-400"
              />
            )}
          </div>
        </Disclosure>

        {/* Breakdowns — collapsed by default */}
        <Disclosure title="Breakdowns" storageKey="breakdowns" icon={LayoutGrid} iconClassName="text-blue-400">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            <BreakdownBar
              title="Top agents"
              items={(stats?.top_agents ?? []).map((a) => ({ label: a.agent, count: a.count }))}
            />
            <BreakdownBar
              title="Top tools"
              items={(stats?.top_tools ?? []).map((t) => ({ label: t.tool, count: t.count, color: "bg-sky-500/60" }))}
            />
            <BreakdownBar
              title="Sources"
              items={(stats?.source_breakdown ?? []).map((s) => ({ label: s.source || "manual", count: s.count, color: "bg-violet-500/60" }))}
            />
            <BreakdownBar
              title="Models"
              items={(stats?.model_breakdown ?? []).map((m) => ({ label: m.model, count: m.count, color: "bg-amber-500/60" }))}
            />
          </div>
        </Disclosure>

        {/* Cost by model — collapsed, only when there is spend */}
        {stats && totalCost > 0 && (
          <Disclosure
            title="Cost by model"
            storageKey="cost"
            icon={DollarSign}
            iconClassName="text-amber-400"
            summary={formatCost(totalCost)}
          >
            <div className="bg-th-card-bg border border-th-card-border rounded-2xl p-4 shadow-sm shadow-black/[0.03]">
              <div className="space-y-2">
                {stats.model_cost_breakdown.map((m) => (
                  <div key={m.model} className="flex items-center gap-3">
                    <span className="text-xs text-th-text-secondary truncate" style={{ minWidth: "8rem", maxWidth: "12rem" }}>{m.model}</span>
                    <div className="flex-1 h-1.5 rounded-full bg-th-inset-bg overflow-hidden">
                      <div
                        className="h-full rounded-full bg-amber-500/50"
                        style={{ width: `${Math.round((m.cost_usd / totalCost) * 100)}%` }}
                      />
                    </div>
                    <span className="text-[11px] text-th-text-muted w-16 text-right shrink-0 tabular-nums">{formatCost(m.cost_usd)}</span>
                  </div>
                ))}
              </div>
            </div>
          </Disclosure>
        )}

        {/* Quick nav to Runs */}
        <div className="flex justify-center pb-2">
          <button
            onClick={() => navigate("/runs")}
            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-th-surface-hover border border-th-border text-sm font-medium text-th-text-secondary hover:text-th-text-primary hover:border-th-border-strong transition-all duration-200 active:scale-[0.98]"
          >
            <Square size={13} aria-hidden />
            View all runs
          </button>
        </div>

      </div>
    </div>
  );
}
