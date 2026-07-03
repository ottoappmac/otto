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

interface RunsOverTimeChartProps {
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

export function RunsOverTimeChart({ data, period }: RunsOverTimeChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-xs text-th-text-muted">
        No data for this period
      </div>
    );
  }

  // Show ~6 tick labels regardless of data density
  const targetTicks = 6;
  const every = Math.max(1, Math.floor(data.length / targetTicks));
  const ticks = data.filter((_, i) => i % every === 0).map((d) => d.ts);

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 4, right: 4, left: -24, bottom: 0 }}>
        <defs>
          <linearGradient id="gradCompleted" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#34d399" stopOpacity={0.25} />
            <stop offset="95%" stopColor="#34d399" stopOpacity={0} />
          </linearGradient>
          <linearGradient id="gradError" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#f87171" stopOpacity={0.25} />
            <stop offset="95%" stopColor="#f87171" stopOpacity={0} />
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
          allowDecimals={false}
          tick={{ fill: "rgb(var(--color-text-muted, 118 118 128))", fontSize: 10 }}
          axisLine={false}
          tickLine={false}
          width={28}
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
          formatter={(value, name) => [
            value,
            name === "completed" ? "Completed" : name === "error" ? "Error" : "Running",
          ]}
        />
        <Area
          type="monotone"
          dataKey="error"
          stackId="1"
          stroke="#f87171"
          strokeWidth={1.5}
          fill="url(#gradError)"
        />
        <Area
          type="monotone"
          dataKey="completed"
          stackId="1"
          stroke="#34d399"
          strokeWidth={1.5}
          fill="url(#gradCompleted)"
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
