import { useMemo } from "react";
import { Gauge, Zap, Database, Cpu, ArrowDownToLine, ArrowUpFromLine } from "lucide-react";
import type { ChatMessage, MlxStats, SessionInfo } from "../../types";

// ---------------------------------------------------------------------------
// SessionStatsPanel — persistent, live session-level token-throughput bar.
//
// Throughput (TIPS / TOPS / KV cache / peak GPU) is aggregated client-side
// from each turn's `metadata.stats` (forwarded by the backend on both agent
// and tool_call messages) so the values update live while the model streams.
// Token totals + cost come from the persisted SessionInfo, which the backend
// accumulates across the whole run.
// ---------------------------------------------------------------------------

function formatTokens(n: number | null | undefined): string {
  if (!n) return "0";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

interface Aggregate {
  avg_prefill_tps: number | null;
  avg_generation_tps: number | null;
  cache_hit_ratio: number | null;
  peak_memory_gb: number | null;
}

/** Token-weighted aggregation of per-turn MLX stats across all messages. */
function aggregateLiveStats(messages: ChatMessage[]): Aggregate {
  let prefillTokens = 0;
  let prefillTime = 0;
  let genTokens = 0;
  let genTime = 0;
  let cacheTokens = 0;
  let peak = 0;

  for (const m of messages) {
    const stats = m.metadata?.stats as MlxStats | undefined;
    if (!stats) continue;
    const prefilled = stats.tokens_prefilled ?? 0;
    const cached = stats.tokens_from_cache ?? 0;
    const gen = stats.generation_tokens ?? 0;
    const ptps = stats.prompt_tps ?? 0;
    const gtps = stats.generation_tps ?? 0;
    if (prefilled && ptps > 0) {
      prefillTokens += prefilled;
      prefillTime += prefilled / ptps;
    }
    if (gen && gtps > 0) {
      genTokens += gen;
      genTime += gen / gtps;
    }
    cacheTokens += cached;
    if (stats.peak_memory_gb && stats.peak_memory_gb > peak) peak = stats.peak_memory_gb;
  }

  const cacheTotal = cacheTokens + prefillTokens;
  return {
    avg_prefill_tps: prefillTime > 0 ? prefillTokens / prefillTime : null,
    avg_generation_tps: genTime > 0 ? genTokens / genTime : null,
    cache_hit_ratio: cacheTotal > 0 ? cacheTokens / cacheTotal : null,
    peak_memory_gb: peak > 0 ? peak : null,
  };
}

function Stat({
  icon,
  label,
  value,
  title,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  title: string;
}) {
  return (
    <span
      className="inline-flex items-center gap-1.5 text-[11px] text-th-text-secondary"
      title={title}
    >
      <span className="text-th-text-tertiary">{icon}</span>
      <span className="text-th-text-tertiary">{label}</span>
      <span className="font-mono text-th-text-primary">{value}</span>
    </span>
  );
}

export default function SessionStatsPanel({
  messages,
  info,
}: {
  messages: ChatMessage[];
  info?: SessionInfo | null;
}) {
  const live = useMemo(() => aggregateLiveStats(messages), [messages]);

  // Prefer live aggregation; fall back to persisted session stats (e.g. on a
  // freshly-loaded historical session before any new turn streams in).
  const prefillTps = live.avg_prefill_tps ?? info?.avg_prefill_tps ?? null;
  const genTps = live.avg_generation_tps ?? info?.avg_generation_tps ?? null;
  const cacheRatio = live.cache_hit_ratio ?? info?.cache_hit_ratio ?? null;
  const peakMem = live.peak_memory_gb ?? info?.peak_memory_gb ?? null;

  const inTok = info?.input_tokens ?? 0;
  const outTok = info?.output_tokens ?? 0;
  const cost = info?.estimated_cost_usd ?? null;

  const hasTokens = inTok > 0 || outTok > 0;
  const hasThroughput = prefillTps != null || genTps != null || cacheRatio != null || peakMem != null;
  if (!hasTokens && !hasThroughput) return null;

  return (
    <div className="flex items-center flex-wrap gap-x-4 gap-y-1">
      {hasTokens && (
        <Stat
          icon={<ArrowDownToLine size={11} />}
          label="in"
          value={formatTokens(inTok)}
          title={`${inTok.toLocaleString()} input tokens`}
        />
      )}
      {hasTokens && (
        <Stat
          icon={<ArrowUpFromLine size={11} />}
          label="out"
          value={formatTokens(outTok)}
          title={`${outTok.toLocaleString()} output tokens`}
        />
      )}
      {prefillTps != null && (
        <Stat
          icon={<Gauge size={11} />}
          label="TIPS"
          value={`${prefillTps.toFixed(0)} t/s`}
          title={`Prefill (input) throughput: ${prefillTps.toFixed(1)} tokens/sec`}
        />
      )}
      {genTps != null && (
        <Stat
          icon={<Zap size={11} />}
          label="TOPS"
          value={`${genTps.toFixed(0)} t/s`}
          title={`Generation (output) throughput: ${genTps.toFixed(1)} tokens/sec`}
        />
      )}
      {cacheRatio != null && (
        <Stat
          icon={<Database size={11} />}
          label="KV"
          value={`${Math.round(cacheRatio * 100)}%`}
          title={`KV cache hit ratio: ${(cacheRatio * 100).toFixed(1)}%`}
        />
      )}
      {peakMem != null && (
        <Stat
          icon={<Cpu size={11} />}
          label="GPU"
          value={`${peakMem.toFixed(1)} GB`}
          title={`Peak GPU memory: ${peakMem.toFixed(2)} GB`}
        />
      )}
      {cost != null && cost > 0 && (
        <Stat
          icon={<span className="text-[11px]">$</span>}
          label=""
          value={cost < 0.01 ? cost.toFixed(4) : cost.toFixed(2)}
          title={`Estimated cost: $${cost.toFixed(4)}`}
        />
      )}
    </div>
  );
}
