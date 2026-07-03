import { Fragment, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Footprints,
  ChevronDown,
  ChevronUp,
  ChevronsUpDown,
  Trash2,
  Loader2,
  AlertCircle,
  Gauge,
  FlaskConical,
  RotateCw,
  Square,
  MoreVertical,
  ExternalLink,
} from "lucide-react";
import type { RunInfo } from "../../types";
import { RunStatusBadge } from "./RunStatusBadge";
import { getSourceIcon, getSourceLabel, getProviderIcon } from "../../utils/entityIcons";
import { formatRelativeTime } from "../../utils/formatRelativeTime";
import { api } from "../../hooks/useApi";
import { Popover } from "../ui/Popover";

export type SortKey = "started_at" | "duration_ms" | "tokens" | "eval";
export type SortDir = "asc" | "desc";
export type Density = "comfortable" | "compact";

function formatDuration(ms: number | null | undefined): string {
  if (ms == null) return "—";
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60_000);
  const s = Math.round((ms % 60_000) / 1000);
  return `${m}m ${s}s`;
}

function formatTokens(n: number | null | undefined): string {
  if (!n) return "0";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

function evalChipClasses(score: number): string {
  if (score >= 0.8) return "text-emerald-400 bg-emerald-500/10 border-emerald-500/20";
  if (score >= 0.5) return "text-amber-400 bg-amber-500/10 border-amber-500/20";
  return "text-red-400 bg-red-500/10 border-red-500/20";
}

interface RunTableProps {
  runs: RunInfo[];
  loading?: boolean;
  emptyMessage?: string;
  onDelete?: (run: RunInfo) => Promise<void>;
  onRunAgain?: () => void;
  density?: Density;
  sortKey?: SortKey | null;
  sortDir?: SortDir;
  onSort?: (key: SortKey) => void;
}

interface SortableHeaderProps {
  label: string;
  field: SortKey;
  active: boolean;
  dir: SortDir;
  onSort?: (key: SortKey) => void;
  align?: "left" | "right";
}

function SortableHeader({ label, field, active, dir, onSort, align = "left" }: SortableHeaderProps) {
  const Caret = !active ? ChevronsUpDown : dir === "asc" ? ChevronUp : ChevronDown;
  return (
    <button
      type="button"
      onClick={() => onSort?.(field)}
      className={`group/sort inline-flex items-center gap-1 -mx-1 px-1 rounded transition-colors hover:text-th-text-secondary ${
        active ? "text-th-text-secondary" : ""
      } ${align === "right" ? "flex-row-reverse" : ""}`}
      aria-label={`Sort by ${label}`}
    >
      <span>{label}</span>
      <Caret
        size={11}
        aria-hidden
        className={active ? "text-blue-400" : "text-th-text-muted/40 group-hover/sort:text-th-text-muted"}
      />
    </button>
  );
}

interface RunActionsMenuProps {
  run: RunInfo;
  onDelete?: (run: RunInfo) => Promise<void>;
  onRunAgain?: () => void;
}

function RunActionsMenu({ run, onDelete, onRunAgain }: RunActionsMenuProps) {
  const navigate = useNavigate();
  const sid = run.session_id ?? run.id;
  const [busy, setBusy] = useState<null | "stop" | "again" | "eval" | "delete">(null);
  const [confirmDelete, setConfirmDelete] = useState(false);

  const canEvaluate =
    !!run.session_id &&
    run.status === "completed" &&
    run.eval_status !== "done" &&
    run.eval_status !== "running";

  const runAction = async (
    kind: "stop" | "again" | "eval",
    fn: () => Promise<unknown>,
    close: () => void,
  ) => {
    if (busy) return;
    setBusy(kind);
    try {
      await fn();
      onRunAgain?.();
    } catch (err) {
      console.warn(`[RunTable] ${kind} failed:`, err);
    } finally {
      setBusy(null);
      close();
    }
  };

  const handleDelete = async (close: () => void) => {
    if (!confirmDelete) {
      setConfirmDelete(true);
      return;
    }
    setBusy("delete");
    try {
      await onDelete?.(run);
    } finally {
      setBusy(null);
      setConfirmDelete(false);
      close();
    }
  };

  const itemCls =
    "flex w-full items-center gap-2.5 px-2.5 py-1.5 rounded-lg text-xs text-th-text-secondary hover:bg-th-surface-hover hover:text-th-text-primary transition-colors disabled:opacity-50 disabled:cursor-not-allowed";

  return (
    <Popover
      role="menu"
      align="right"
      panelClassName="w-48 p-1 overflow-hidden"
      onClose={() => setConfirmDelete(false)}
      trigger={({ toggle, open, ...aria }) => (
        <button
          type="button"
          onClick={(e) => { e.stopPropagation(); toggle(); }}
          className={`p-1 rounded-lg text-th-text-muted/50 hover:text-th-text-primary hover:bg-th-surface-hover transition-all duration-200 active:scale-90 ${
            open ? "opacity-100 bg-th-surface-hover text-th-text-primary" : "opacity-0 group-hover:opacity-100 focus:opacity-100"
          }`}
          title="Actions"
          aria-label={`Actions for ${run.title || "run"}`}
          {...aria}
        >
          {busy ? <Loader2 size={14} className="animate-spin" /> : <MoreVertical size={14} />}
        </button>
      )}
    >
      {({ close }) => (
        <div onClick={(e) => e.stopPropagation()}>
          <button
            type="button"
            className={itemCls}
            onClick={() => { close(); navigate(`/runs/${sid}`); }}
          >
            <ExternalLink size={13} aria-hidden />
            View run
          </button>
          <button
            type="button"
            className={itemCls}
            disabled={!!busy}
            onClick={() => runAction("again", () => api.runAgain(sid), close)}
          >
            {busy === "again" ? <Loader2 size={13} className="animate-spin" /> : <RotateCw size={13} aria-hidden />}
            Run again
          </button>
          {canEvaluate && (
            <button
              type="button"
              className={itemCls}
              disabled={!!busy}
              onClick={() => runAction("eval", () => api.runRunEvaluation(sid), close)}
            >
              {busy === "eval" ? <Loader2 size={13} className="animate-spin" /> : <FlaskConical size={13} aria-hidden />}
              Evaluate
            </button>
          )}
          {run.status === "running" && (
            <button
              type="button"
              className={`${itemCls} text-orange-400/90 hover:text-orange-300`}
              disabled={!!busy}
              onClick={() => runAction("stop", () => api.stopSession(sid), close)}
            >
              {busy === "stop" ? <Loader2 size={13} className="animate-spin" /> : <Square size={13} aria-hidden />}
              Stop
            </button>
          )}
          {onDelete && (
            <>
              <div className="my-1 mx-1 border-t border-th-border/70" />
              <button
                type="button"
                className={`${itemCls} ${confirmDelete ? "text-red-400 bg-red-500/10" : "text-red-400/90 hover:text-red-300 hover:bg-red-500/10"}`}
                disabled={busy === "delete"}
                onClick={() => handleDelete(close)}
              >
                {busy === "delete" ? <Loader2 size={13} className="animate-spin" /> : <Trash2 size={13} aria-hidden />}
                {confirmDelete ? "Confirm delete?" : "Delete"}
              </button>
            </>
          )}
        </div>
      )}
    </Popover>
  );
}

export function RunTable({
  runs,
  loading = false,
  emptyMessage = "No runs found",
  onDelete,
  onRunAgain,
  density = "comfortable",
  sortKey = null,
  sortDir = "desc",
  onSort,
}: RunTableProps) {
  const navigate = useNavigate();
  const [expandedErrorId, setExpandedErrorId] = useState<string | null>(null);

  if (loading) {
    return (
      <div className="space-y-1.5 p-2">
        {[...Array(6)].map((_, i) => (
          <div key={i} className="h-10 rounded-lg bg-th-inset-bg animate-pulse" />
        ))}
      </div>
    );
  }

  if (runs.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-center">
        <p className="text-sm text-th-text-muted">{emptyMessage}</p>
      </div>
    );
  }

  const rowPad = density === "compact" ? "py-1.5" : "py-2.5";
  const colSpan = 9;

  return (
    <table className="w-full text-sm border-collapse">
      <thead className="sticky top-0 z-10 bg-th-bg-secondary/80 backdrop-blur-xl">
        <tr className="border-b border-th-border">
          <th className="text-left text-[10px] font-semibold uppercase tracking-wider text-th-text-muted px-3 py-2 whitespace-nowrap">Status</th>
          <th className="text-left text-[10px] font-semibold uppercase tracking-wider text-th-text-muted px-3 py-2 whitespace-nowrap">Run</th>
          <th className="text-left text-[10px] font-semibold uppercase tracking-wider text-th-text-muted px-3 py-2 whitespace-nowrap">Agent</th>
          <th className="text-left text-[10px] font-semibold uppercase tracking-wider text-th-text-muted px-3 py-2 whitespace-nowrap">Source</th>
          <th className="text-left text-[10px] font-semibold uppercase tracking-wider text-th-text-muted px-3 py-2 whitespace-nowrap">
            <SortableHeader label="Started" field="started_at" active={sortKey === "started_at"} dir={sortDir} onSort={onSort} />
          </th>
          <th className="text-left text-[10px] font-semibold uppercase tracking-wider text-th-text-muted px-3 py-2 whitespace-nowrap">
            <SortableHeader label="Duration" field="duration_ms" active={sortKey === "duration_ms"} dir={sortDir} onSort={onSort} />
          </th>
          <th className="text-left text-[10px] font-semibold uppercase tracking-wider text-th-text-muted px-3 py-2 whitespace-nowrap">
            <SortableHeader label="Tokens" field="tokens" active={sortKey === "tokens"} dir={sortDir} onSort={onSort} />
          </th>
          <th className="text-left text-[10px] font-semibold uppercase tracking-wider text-th-text-muted px-3 py-2 whitespace-nowrap">
            <SortableHeader label="Eval" field="eval" active={sortKey === "eval"} dir={sortDir} onSort={onSort} />
          </th>
          <th className="px-2 py-2 w-px" aria-label="Actions" />
        </tr>
      </thead>
      <tbody>
        {runs.map((run) => {
          const { Icon: SrcIcon, className: srcCls } = getSourceIcon(run.trigger_source);
          const { Icon: ProvIcon, className: provCls } = getProviderIcon(run.llm_provider);
          const hasError = run.status === "error" && !!run.error;
          const isErrorExpanded = hasError && expandedErrorId === run.id;

          return (
            <Fragment key={run.id}>
              <tr
                className={`${isErrorExpanded ? "" : "border-b border-th-border/40"} hover:bg-th-surface-hover cursor-pointer transition-colors group`}
                onClick={() => navigate(`/runs/${run.session_id ?? run.id}`)}
              >
                <td className={`px-3 ${rowPad} whitespace-nowrap`}>
                  <div className="flex items-center gap-1.5">
                    {run.eval_status === "running" ? (
                      <span className="inline-flex items-center gap-1 text-[11px] font-medium text-blue-400">
                        <FlaskConical size={11} className="animate-pulse" aria-hidden />
                        Evaluating…
                      </span>
                    ) : (
                      <>
                        <RunStatusBadge status={run.status} />
                        {hasError && (
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setExpandedErrorId(isErrorExpanded ? null : run.id);
                            }}
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
                            <ChevronDown
                              size={11}
                              aria-hidden
                              className={`transition-transform ${isErrorExpanded ? "rotate-180" : ""}`}
                            />
                          </button>
                        )}
                      </>
                    )}
                  </div>
                </td>
                <td className={`px-3 ${rowPad} max-w-xs`}>
                  <div className="flex flex-col gap-0.5 min-w-0">
                    <p className="text-th-text-primary font-medium truncate group-hover:text-blue-400 transition-colors">
                      {run.title || "Untitled"}
                    </p>
                    <span className="text-[10px] font-mono text-th-text-muted/50">
                      #{(run.session_id ?? run.id).slice(0, 8)}
                    </span>
                  </div>
                </td>
                <td className={`px-3 ${rowPad} whitespace-nowrap`}>
                  {run.agent_name ? (
                    <span className="inline-flex items-center gap-1 text-xs text-th-text-secondary">
                      {run.llm_provider && (
                        <span title={run.llm_provider}>
                          <ProvIcon size={10} className={provCls} aria-hidden />
                        </span>
                      )}
                      {run.agent_name}
                    </span>
                  ) : (
                    <span className="text-xs text-th-text-muted">—</span>
                  )}
                </td>
                <td className={`px-3 ${rowPad} whitespace-nowrap`}>
                  {run.schedule_id ? (
                    <button
                      onClick={(e) => { e.stopPropagation(); navigate(`/schedules/${encodeURIComponent(run.schedule_id!)}/runs`); }}
                      className="inline-flex items-center gap-1 text-xs text-th-text-secondary hover:text-blue-400 transition-colors"
                      title="View all runs for this schedule"
                    >
                      <SrcIcon size={11} className={srcCls} aria-hidden />
                      {getSourceLabel(run.trigger_source)}
                    </button>
                  ) : run.trigger_id ? (
                    <button
                      onClick={(e) => { e.stopPropagation(); navigate(`/triggers/${encodeURIComponent(run.trigger_id!)}/runs`); }}
                      className="inline-flex items-center gap-1 text-xs text-th-text-secondary hover:text-blue-400 transition-colors"
                      title="View all runs for this trigger"
                    >
                      <SrcIcon size={11} className={srcCls} aria-hidden />
                      {getSourceLabel(run.trigger_source)}
                    </button>
                  ) : (
                    <span className="inline-flex items-center gap-1 text-xs text-th-text-secondary">
                      <SrcIcon size={11} className={srcCls} aria-hidden />
                      {getSourceLabel(run.trigger_source)}
                    </span>
                  )}
                </td>
                <td
                  className={`px-3 ${rowPad} whitespace-nowrap text-xs text-th-text-muted`}
                  title={run.started_at ? new Date(run.started_at).toLocaleString() : undefined}
                >
                  {run.started_at ? formatRelativeTime(run.started_at) : "—"}
                </td>
                <td className={`px-3 ${rowPad} whitespace-nowrap`}>
                  <div className="flex flex-col gap-0.5">
                    <span className="text-xs text-th-text-muted">{formatDuration(run.duration_ms)}</span>
                    <span
                      className="inline-flex items-center gap-1 text-[10px] text-th-text-muted/50"
                      title="Total agent steps (model turns + tool calls), including subagents"
                    >
                      <Footprints size={9} aria-hidden />
                      {run.step_count ?? 0} steps
                    </span>
                  </div>
                </td>
                <td className={`px-3 ${rowPad} whitespace-nowrap`}>
                  {run.input_tokens || run.output_tokens ? (
                    <div className="flex flex-col gap-0.5">
                      <span
                        className="font-mono text-xs text-th-text-muted"
                        title={`${(run.input_tokens ?? 0).toLocaleString()} in / ${(run.output_tokens ?? 0).toLocaleString()} out`}
                      >
                        {formatTokens(run.input_tokens)}
                        <span className="text-th-text-muted/40"> / </span>
                        {formatTokens(run.output_tokens)}
                      </span>
                      {run.avg_generation_tps != null && (
                        <span
                          className="font-mono text-[10px] text-th-text-muted/50"
                          title={`Generation throughput: ${run.avg_generation_tps} tokens/sec`}
                        >
                          {run.avg_generation_tps.toFixed(0)} t/s
                        </span>
                      )}
                    </div>
                  ) : (
                    <span className="text-xs text-th-text-muted/40">—</span>
                  )}
                </td>
                <td className={`px-3 ${rowPad} whitespace-nowrap`}>
                  {run.eval_status === "done" && run.eval_overall_score != null ? (
                    <button
                      className={`inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-full border text-[10px] font-bold hover:opacity-80 transition-opacity ${evalChipClasses(run.eval_overall_score)}`}
                      title={`${run.eval_pass_count ?? 0}/${run.eval_total ?? 0} metrics passed — click to view`}
                      onClick={(e) => {
                        e.stopPropagation();
                        navigate(`/runs/${run.session_id ?? run.id}?tab=evaluation`);
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
                  <RunActionsMenu run={run} onDelete={onDelete} onRunAgain={onRunAgain} />
                </td>
              </tr>
              {isErrorExpanded && (
                <tr key={`${run.id}-error`} className="border-b border-th-border/40">
                  <td colSpan={colSpan} className="px-3 pb-2.5 pt-0">
                    <div className="flex items-start gap-2 text-[11px] text-red-400/90 bg-red-500/[0.06] border border-red-500/15 rounded-md px-3 py-2 leading-relaxed">
                      <AlertCircle size={12} className="shrink-0 mt-0.5" aria-hidden />
                      <p className="flex-1 min-w-0 break-words whitespace-pre-wrap">{run.error}</p>
                      <button
                        onClick={() => navigate(`/runs/${run.session_id ?? run.id}`)}
                        className="shrink-0 text-[10px] font-medium text-red-400 hover:text-red-300 underline-offset-2 hover:underline"
                      >
                        View run
                      </button>
                    </div>
                  </td>
                </tr>
              )}
            </Fragment>
          );
        })}
      </tbody>
    </table>
  );
}
