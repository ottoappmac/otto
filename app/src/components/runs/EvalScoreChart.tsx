import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { RunStats } from "../../types";

export type ChartPeriod = "24h" | "7d" | "30d" | "all" | "custom";

interface EvalScoreChartProps {
  data: RunStats["time_series"];
  period: ChartPeriod;
}

function formatTick(ts: string, period: ChartPeriod) {
  const d = new Date(ts);
  if (period === "24h") {
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
  if (period === "all") {
    return d.toLocaleDateString([], { month: "short", year: "2-digit" });
  }
  return d.toLocaleDateString([], { month: "short", day: "numeric" });
}

export function EvalScoreChart({ data, period }: EvalScoreChartProps) {
  // Only buckets that actually contain an evaluated run are meaningful; the
  // backend sends null for empty buckets so the line skips those gaps.
  const points = (data ?? []).map((d) => ({
    ts: d.ts,
    score: d.eval_avg_score != null ? Math.round(d.eval_avg_score * 100) : null,
    count: d.eval_count,
  }));
  const hasAny = points.some((p) => p.score != null);

  if (!hasAny) {
    return (
      <div className="flex items-center justify-center h-full text-xs text-th-text-muted">
        No evaluated runs in this period
      </div>
    );
  }

  const targetTicks = 6;
  const every = Math.max(1, Math.floor(points.length / targetTicks));
  const ticks = points.filter((_, i) => i % every === 0).map((d) => d.ts);

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={points} margin={{ top: 4, right: 4, left: -24, bottom: 0 }}>
        <defs>
          <linearGradient id="gradEvalScore" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#60a5fa" stopOpacity={0.3} />
            <stop offset="95%" stopColor="#60a5fa" stopOpacity={0} />
          </linearGradient>
        </defs>
        <XAxis
          dataKey="ts"
          ticks={ticks}
          tickFormatter={(t) => formatTick(t, period)}
          tick={{ fill: "rgb(var(--color-text-muted, 118 118 128))", fontSize: 10 }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          domain={[0, 100]}
          ticks={[0, 25, 50, 75, 100]}
          tickFormatter={(v) => `${v}%`}
          tick={{ fill: "rgb(var(--color-text-muted, 118 118 128))", fontSize: 10 }}
          axisLine={false}
          tickLine={false}
          width={36}
        />
        <Tooltip
          contentStyle={{
            background: "rgb(var(--color-card-bg))",
            border: "1px solid rgb(var(--color-border))",
            borderRadius: "12px",
            fontSize: 11,
            color: "rgb(var(--color-text-primary))",
            boxShadow: "0 8px 24px rgb(0 0 0 / 0.12)",
          }}
          labelStyle={{ color: "rgb(var(--color-text-secondary))" }}
          itemStyle={{ color: "rgb(var(--color-text-secondary))" }}
          labelFormatter={(l) => formatTick(l as string, period)}
          formatter={(value, _name, item) => {
            const count = (item?.payload as { count?: number } | undefined)?.count ?? 0;
            return [`${value}% · ${count} run${count === 1 ? "" : "s"}`, "Avg score"];
          }}
        />
        <Area
          type="monotone"
          dataKey="score"
          stroke="#60a5fa"
          strokeWidth={1.5}
          fill="url(#gradEvalScore)"
          connectNulls
          dot={{ r: 2, fill: "#60a5fa" }}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
