import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import {
  ArrowLeft,
  ExternalLink,
  Square,
  Trash2,
  MessageSquare,
  GitBranch,
  FileOutput,
  BarChart2,
  Clock,
  Coins,
  Wrench,
  ChevronDown,
  ChevronUp,
  Loader2,
  X,
  FolderOpen,
  File,
  FileText,
  FileCode,
  Image,
  Download,
  FolderOpenDot,
  Gauge,
  Zap,
  Database,
  Cpu,
  Sparkles,
  Brain,
  ListChecks,
  CheckCircle2,
  XCircle,
  CircleDot,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api } from "../hooks/useApi";
import { usePolling } from "../hooks/usePolling";
import { useNotification } from "../context/NotificationContext";
import { formatRelativeTime } from "../utils/formatRelativeTime";
import type { RunInfo, TimelineEvent, SessionTimeline, RunEvaluation } from "../types";
import { RunStatusBadge } from "../components/runs/RunStatusBadge";
import { RunTimeline } from "../components/runs/RunTimeline";
import { AgentGraph } from "../components/chat/AgentGraph";
import { BreakdownBar } from "../components/runs/BreakdownBar";
import { StatCard } from "../components/runs/StatCard";
import { getSourceIcon, getSourceLabel, getProviderIcon } from "../utils/entityIcons";
import { familyChipClasses } from "../utils/subagentModelChip";
import { WS_BASE } from "../config/apiBase";

const POLL_MS = 4_000;

function formatDuration(ms: number | null | undefined): string {
  if (ms == null) return "—";
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60_000);
  const s = Math.round((ms % 60_000) / 1000);
  return `${m}m ${s}s`;
}

function formatCost(usd: number | null | undefined): string {
  if (!usd) return "—";
  if (usd < 0.001) return "<$0.001";
  return `$${usd.toFixed(4)}`;
}

function formatTokens(n: number | null | undefined): string {
  if (n == null) return "—";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

type Tab = "timeline" | "graph" | "results" | "files" | "metrics" | "evaluation";

interface SessionFile {
  path: string;
  size: number;
  modified_at: number;
}

function fileIcon(path: string) {
  const ext = path.split(".").pop()?.toLowerCase() ?? "";
  if (["png", "jpg", "jpeg", "gif", "webp", "svg"].includes(ext)) return Image;
  if (["ts", "tsx", "js", "jsx", "py", "sh", "json", "yaml", "toml", "css", "html"].includes(ext)) return FileCode;
  if (["md", "txt", "csv", "log"].includes(ext)) return FileText;
  return File;
}

function formatBytes(b: number): string {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / (1024 * 1024)).toFixed(1)} MB`;
}

// ---------------------------------------------------------------------------
// RunDetailPage
// ---------------------------------------------------------------------------

// ── Agent turns history (collapsible) ────────────────────────────────────────

function AgentTurns({ events }: { events: TimelineEvent[] }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="bg-th-card-bg border border-th-card-border rounded-2xl overflow-hidden shadow-sm shadow-black/[0.03]">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full px-5 py-3 flex items-center justify-between text-left hover:bg-th-surface-hover transition-colors"
      >
        <span className="text-[11px] font-semibold text-th-text-muted uppercase tracking-wider">
          Previous turns ({events.length})
        </span>
        <ChevronDown size={13} className={`text-th-text-muted transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <div className="divide-y divide-th-border/30">
          {events.map((ev, i) => {
            const text = typeof ev.content === "string" ? ev.content : "";
            return (
              <div key={i} className="px-5 py-3">
                <p className="text-[10px] text-th-text-muted/60 mb-2">
                  Turn {i + 1}{ev.ts ? ` · ${formatRelativeTime(ev.ts)}` : ""}
                </p>
                <div className="prose prose-sm dark:prose-invert max-w-none text-th-text-secondary
                  prose-p:my-1.5 prose-code:text-xs prose-code:bg-th-inset-bg prose-code:px-1 prose-code:rounded
                  prose-code:before:content-none prose-code:after:content-none
                  prose-pre:bg-th-inset-bg prose-pre:text-xs prose-pre:border prose-pre:border-th-border/50">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ── Files tab ─────────────────────────────────────────────────────────────────

function FilesTab({
  sessionId,
  files,
}: {
  sessionId: string;
  files: SessionFile[];
}) {
  const [previewPath, setPreviewPath] = useState<string | null>(null);
  const [previewContent, setPreviewContent] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);

  const isPreviewable = (path: string) => {
    const ext = path.split(".").pop()?.toLowerCase() ?? "";
    return ["md", "txt", "json", "csv", "log", "py", "ts", "tsx", "js", "jsx", "sh", "yaml", "toml", "html", "css", "svg"].includes(ext);
  };
  const isImage = (path: string) => {
    const ext = path.split(".").pop()?.toLowerCase() ?? "";
    return ["png", "jpg", "jpeg", "gif", "webp"].includes(ext);
  };
  const isMarkdown = (path: string) => path.split(".").pop()?.toLowerCase() === "md";
  const isHtml = (path: string) => path.split(".").pop()?.toLowerCase() === "html";

  const openPreview = async (path: string) => {
    if (previewPath === path) { setPreviewPath(null); setPreviewContent(null); return; }
    setPreviewPath(path);
    setPreviewContent(null);
    if (isImage(path)) return; // images render via URL directly
    if (!isPreviewable(path)) return;
    setPreviewLoading(true);
    try {
      const url = api.getSessionFileUrl(sessionId, path);
      const res = await fetch(url);
      const text = await res.text();
      setPreviewContent(text);
    } catch { setPreviewContent("(Failed to load file content)"); }
    finally { setPreviewLoading(false); }
  };

  const openFolder = async () => {
    try { await api.openSessionFilesFolder(sessionId); } catch { /* ignore */ }
  };

  if (files.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-center">
        <FolderOpen size={28} className="text-th-text-muted mb-3" />
        <p className="text-sm text-th-text-muted">No files created in this run</p>
        <p className="text-xs text-th-text-muted/60 mt-1">Files written by the agent appear here</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Toolbar */}
      <div className="flex items-center justify-between">
        <p className="text-[11px] text-th-text-muted">{files.length} file{files.length !== 1 ? "s" : ""}</p>
        <button
          onClick={openFolder}
          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl border border-th-border bg-th-inset-bg/70 text-xs text-th-text-secondary hover:text-th-text-primary hover:border-th-border-strong transition-all duration-200 active:scale-[0.97]"
        >
          <FolderOpenDot size={12} />
          Open folder
        </button>
      </div>

      {/* File list */}
      <div className="bg-th-card-bg border border-th-card-border rounded-2xl divide-y divide-th-border/40 overflow-hidden shadow-sm shadow-black/[0.03]">
        {[...files].sort((a, b) => b.modified_at - a.modified_at).map((f) => {
          const name = f.path.split("/").pop() ?? f.path;
          const IconComp = fileIcon(f.path);
          const isOpen = previewPath === f.path;
          const imgUrl = isImage(f.path) ? api.getSessionFileUrl(sessionId, f.path) : null;
          const downloadUrl = api.getSessionFileUrl(sessionId, f.path);

          return (
            <div key={f.path}>
              {/* File row */}
              <div
                className={`flex items-center gap-3 px-4 py-3 hover:bg-th-surface-hover transition-colors cursor-pointer group ${isOpen ? "bg-th-surface-hover" : ""}`}
                onClick={() => (isPreviewable(f.path) || isImage(f.path)) && openPreview(f.path)}
              >
                <IconComp size={15} className="text-th-text-muted shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-th-text-primary truncate">{name}</p>
                  <p className="text-[10px] text-th-text-muted/70 truncate mt-0.5">{f.path}</p>
                </div>
                <div className="flex items-center gap-3 shrink-0">
                  <span className="text-[11px] text-th-text-muted">{formatBytes(f.size)}</span>
                  <span className="text-[10px] text-th-text-muted/60">
                    {new Date(f.modified_at * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                  </span>
                  <a
                    href={downloadUrl}
                    download={name}
                    onClick={(e) => e.stopPropagation()}
                    className="opacity-0 group-hover:opacity-100 transition-opacity text-th-text-muted hover:text-blue-400"
                    title="Download"
                  >
                    <Download size={13} />
                  </a>
                </div>
              </div>

              {/* Inline preview */}
              {isOpen && (
                <div className="border-t border-th-border/40 bg-th-inset-bg/50">
                  {previewLoading ? (
                    <div className="flex items-center justify-center py-8">
                      <Loader2 size={16} className="animate-spin text-th-text-muted" />
                    </div>
                  ) : imgUrl ? (
                    <div className="p-4 flex justify-center">
                      <img src={imgUrl} alt={name} className="max-h-80 max-w-full rounded-lg object-contain" />
                    </div>
                  ) : previewContent != null ? (
                    isHtml(f.path) ? (
                      <>
                        <div className="flex items-center gap-2 px-4 py-2 border-b border-th-border/40 bg-th-card-bg">
                          <button
                            onClick={() => { setPreviewPath(null); setPreviewContent(null); }}
                            className="flex items-center gap-1.5 text-xs text-th-text-muted hover:text-th-text-primary transition-colors"
                          >
                            <ArrowLeft size={12} aria-hidden />
                            Back to files
                          </button>
                          <span className="text-th-text-muted/40 text-xs">·</span>
                          <span className="text-xs text-th-text-muted/60 truncate">{name}</span>
                        </div>
                        <iframe
                          srcDoc={previewContent}
                          sandbox="allow-scripts allow-same-origin"
                          className="w-full border-0 rounded-b-xl"
                          style={{ height: "520px" }}
                          title={name}
                        />
                      </>
                    ) : isMarkdown(f.path) ? (
                      <div className="px-5 py-4 prose prose-sm dark:prose-invert max-w-none
                        prose-p:my-2 prose-code:text-xs prose-code:bg-th-inset-bg prose-code:px-1 prose-code:rounded
                        prose-code:before:content-none prose-code:after:content-none
                        prose-pre:bg-th-card-bg prose-pre:border prose-pre:border-th-border/50 prose-pre:text-xs
                        text-th-text-primary">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={{
                            a: ({ href, children }) => (
                              <a href={href} target="_blank" rel="noopener noreferrer">
                                {children}
                              </a>
                            ),
                          }}
                        >{previewContent}</ReactMarkdown>
                      </div>
                    ) : (
                      <pre className="px-4 py-3 text-xs text-th-text-secondary font-mono leading-relaxed overflow-x-auto max-h-80 whitespace-pre-wrap break-words">
                        {previewContent.slice(0, 6000)}{previewContent.length > 6000 ? "\n\n… (truncated)" : ""}
                      </pre>
                    )
                  ) : null}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Evaluation tab ─────────────────────────────────────────────────────────────

const EVAL_LABELS: Record<string, string> = {
  answer_relevancy: "Answer Relevancy",
  toxicity: "Toxicity",
  bias: "Bias",
  g_eval: "G-Eval",
  tool_correctness: "Tool Correctness",
  trajectory: "Trajectory",
};

function scoreColor(score: number, passed: boolean | undefined): string {
  if (passed === false) return "text-red-400";
  if (score >= 0.8) return "text-emerald-400";
  if (score >= 0.5) return "text-amber-400";
  return "text-red-400";
}

function stepIcon(kind: string, success?: boolean) {
  switch (kind) {
    case "thought": return Brain;
    case "selection": return ListChecks;
    case "metric_start": return CircleDot;
    case "metric_result": return success === false ? XCircle : CheckCircle2;
    case "suggestion": return Sparkles;
    case "done": return Sparkles;
    default: return CircleDot;
  }
}

function EvalTrace({ steps, running, title = "Evaluator activity" }: { steps: NonNullable<RunEvaluation["steps"]>; running: boolean; title?: string }) {
  return (
    <div className="bg-th-card-bg border border-th-card-border rounded-2xl overflow-hidden shadow-sm shadow-black/[0.03]">
      <div className="px-5 py-3 border-b border-th-border/50 flex items-center gap-2">
        <Sparkles size={13} className="text-blue-400" aria-hidden />
        <p className="text-[11px] font-semibold text-th-text-muted uppercase tracking-wider">{title}</p>
      </div>
      <div className="divide-y divide-th-border/30">
        {steps.map((s, i) => {
          const Icon = stepIcon(s.kind, s.success);
          const isResult = s.kind === "metric_result";
          const tone = s.kind === "thought"
            ? "text-th-text-secondary italic"
            : isResult
              ? (s.success === false ? "text-red-300" : "text-emerald-200")
              : "text-th-text-secondary";
          const iconTone = s.kind === "metric_result"
            ? (s.success === false ? "text-red-400" : "text-emerald-400")
            : s.kind === "thought" ? "text-purple-400"
            : s.kind === "done" ? "text-blue-400"
            : "text-th-text-muted";
          return (
            <div key={i} className="px-5 py-2.5 flex items-start gap-2.5">
              <Icon size={13} className={`mt-0.5 shrink-0 ${iconTone}`} aria-hidden />
              <div className="min-w-0 flex-1">
                {s.metric && (s.kind === "metric_start" || s.kind === "metric_result") && (
                  <span className="text-[10px] font-semibold text-th-text-muted uppercase tracking-wide mr-1.5">
                    {EVAL_LABELS[s.metric] ?? s.metric}
                    {isResult && typeof s.score === "number" ? ` · ${Math.round(s.score * 100)}%` : ""}
                  </span>
                )}
                <span className={`text-xs leading-relaxed ${tone}`}>
                  {typeof s.text === "string" ? s.text : JSON.stringify(s.text)}
                </span>
              </div>
            </div>
          );
        })}
        {running && (
          <div className="px-5 py-2.5 flex items-center gap-2.5 text-th-text-muted">
            <Loader2 size={13} className="animate-spin text-blue-400 shrink-0" aria-hidden />
            <span className="text-xs">Thinking…</span>
          </div>
        )}
      </div>
    </div>
  );
}

function EvaluationTab({
  evaluation,
  evaluating,
  autoEvaluate,
  onEvaluate,
  run,
}: {
  evaluation: RunEvaluation | null;
  evaluating: boolean;
  autoEvaluate: boolean;
  onEvaluate: () => void;
  run: RunInfo | null;
}) {
  const navigate = useNavigate();
  const [runningImproved, setRunningImproved] = useState(false);
  const [updatingPrompt, setUpdatingPrompt] = useState(false);
  const [updateDone, setUpdateDone] = useState(false);
  const status = evaluation?.status ?? "none";
  const steps = evaluation?.steps ?? [];
  const running = evaluating || status === "running";
  const isErrorAnalysis = evaluation?.kind === "error_analysis" || run?.status === "error";
  const traceTitle = isErrorAnalysis ? "Error analysis" : "Evaluator activity";

  // Before anything has streamed in.
  if (status === "none" && !running) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-center">
        <Gauge size={28} className="text-th-text-muted mb-3" />
        <p className="text-sm text-th-text-muted">
          {isErrorAnalysis ? "This failed run hasn't been analyzed yet" : "This run hasn't been evaluated yet"}
        </p>
        {autoEvaluate ? (
          <p className="text-xs text-th-text-muted/60 mt-1">
            {isErrorAnalysis
              ? "Failed-run analysis is enabled — a prompt fix is drafted automatically when applicable."
              : "Auto-evaluation is enabled — scores appear automatically once a run completes."}
          </p>
        ) : (
          <button
            onClick={onEvaluate}
            className="mt-4 flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-blue-500/15 border border-blue-500/30 text-xs font-medium text-blue-400 hover:bg-blue-500/25 transition-all duration-200 active:scale-[0.97]"
          >
            <Gauge size={12} aria-hidden />
            {isErrorAnalysis ? "Analyze failure" : "Evaluate run"}
          </button>
        )}
      </div>
    );
  }

  const results = evaluation?.results ?? [];
  const overall = evaluation?.overall_score;
  const showSummary = status === "done" && results.length > 0;

  return (
    <div className="space-y-4">
      {/* Overall header (once scored) */}
      {showSummary && (
        <div className="bg-th-card-bg border border-th-card-border rounded-2xl px-5 py-4 flex items-center justify-between shadow-sm shadow-black/[0.03]">
          <div>
            <p className="text-[10px] text-th-text-muted uppercase tracking-wider mb-1">Overall score</p>
            <div className="flex items-baseline gap-2">
              <span className={`text-2xl font-bold ${overall != null ? scoreColor(overall, undefined) : "text-th-text-muted"}`}>
                {overall != null ? `${Math.round(overall * 100)}%` : "—"}
              </span>
              {evaluation?.pass_count != null && evaluation?.total != null && (
                <span className="text-xs text-th-text-muted">
                  {evaluation.pass_count}/{evaluation.total} passed
                </span>
              )}
            </div>
          </div>
          {evaluation?.model && (
            <span className="text-[10px] text-th-text-muted/60">judge: {evaluation.model}</span>
          )}
        </div>
      )}

      {/* Suggested prompt improvement (when the run scored below threshold) */}
      {evaluation?.suggested_prompt && (() => {
        const isSchedule = run?.trigger_source === "schedule" && run?.schedule_id;
        const isTrigger  = run?.trigger_source === "trigger"  && run?.trigger_id;
        const sessionId  = run?.session_id ?? run?.id ?? null;

        const handleRunImproved = async () => {
          if (!sessionId) return;
          setRunningImproved(true);
          try {
            const { session_id } = await api.runAgain(sessionId, evaluation.suggested_prompt!);
            navigate(`/runs/${session_id}?tab=timeline`);
          } finally {
            setRunningImproved(false);
          }
        };

        const handleUpdateAutomation = async () => {
          if (updatingPrompt || updateDone) return;
          setUpdatingPrompt(true);
          try {
            if (isSchedule) {
              await api.updateSchedule(run!.schedule_id!, { prompt: evaluation.suggested_prompt });
            } else if (isTrigger) {
              await api.updateTrigger(run!.trigger_id!, { prompt: evaluation.suggested_prompt });
            }
            setUpdateDone(true);
            setTimeout(() => setUpdateDone(false), 3000);
          } finally {
            setUpdatingPrompt(false);
          }
        };

        return (
          <div className="bg-blue-500/5 border border-blue-500/20 rounded-2xl px-5 py-4 shadow-sm shadow-black/[0.03]">
            <div className="flex items-center gap-1.5 mb-2">
              <Sparkles size={13} className="text-blue-400" aria-hidden />
              <p className="text-[10px] font-semibold text-blue-400 uppercase tracking-wider">
                Suggested prompt improvement
              </p>
            </div>
            {evaluation.suggestion_reason && (
              <p className="text-xs text-th-text-secondary leading-relaxed mb-3">
                {evaluation.suggestion_reason}
              </p>
            )}
            <p className="text-sm text-th-text-primary leading-relaxed whitespace-pre-wrap bg-th-inset-bg rounded-lg px-3 py-2.5 border border-th-card-border">
              {evaluation.suggested_prompt}
            </p>
            <div className="mt-3 flex flex-wrap gap-2">
              {/* Run now with improved prompt */}
              {sessionId && (
                <button
                  onClick={handleRunImproved}
                  disabled={runningImproved}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-emerald-500/15 border border-emerald-500/30 text-xs font-medium text-emerald-400 hover:bg-emerald-500/25 disabled:opacity-50 transition-all duration-200 active:scale-[0.97]"
                >
                  {runningImproved
                    ? <Loader2 size={12} className="animate-spin" aria-hidden />
                    : <Zap size={12} aria-hidden />}
                  Run now with improved prompt
                </button>
              )}
              {/* Update schedule / trigger prompt */}
              {(isSchedule || isTrigger) && (
                <button
                  onClick={handleUpdateAutomation}
                  disabled={updatingPrompt || updateDone}
                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl border text-xs font-medium transition-all duration-200 active:scale-[0.97] disabled:opacity-60 ${
                    updateDone
                      ? "bg-emerald-500/10 border-emerald-500/25 text-emerald-400"
                      : "bg-blue-500/10 border-blue-500/25 text-blue-400 hover:bg-blue-500/20"
                  }`}
                >
                  {updatingPrompt
                    ? <Loader2 size={12} className="animate-spin" aria-hidden />
                    : updateDone
                      ? <CheckCircle2 size={12} aria-hidden />
                      : <CheckCircle2 size={12} aria-hidden />}
                  {updateDone
                    ? `${isSchedule ? "Schedule" : "Trigger"} prompt updated`
                    : `Update ${isSchedule ? "schedule" : "trigger"} prompt`}
                </button>
              )}
              {/* View in suggestions */}
              <button
                onClick={() => navigate("/ambient")}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-th-surface-hover border border-th-border text-xs font-medium text-th-text-muted hover:text-th-text-secondary transition-all duration-200 active:scale-[0.97]"
              >
                <Sparkles size={12} aria-hidden />
                View in Suggestions
              </button>
            </div>
          </div>
        );
      })()}

      {/* Live evaluator trace (chat-style) */}
      {(steps.length > 0 || running) && <EvalTrace steps={steps} running={running} title={traceTitle} />}

      {/* Skipped / error notices */}
      {status === "skipped" && (
        <div className="bg-amber-500/5 border border-amber-500/20 rounded-2xl px-5 py-4 shadow-sm shadow-black/[0.03]">
          <p className="text-[10px] font-semibold text-amber-400 uppercase tracking-wider mb-1">Evaluation skipped</p>
          <p className="text-xs text-th-text-secondary">{evaluation?.reason || "No judge model was available."}</p>
        </div>
      )}
      {status === "error" && (
        <div className="bg-red-500/5 border border-red-500/20 rounded-2xl p-4 shadow-sm shadow-black/[0.03]">
          <p className="text-[10px] font-semibold text-red-400 uppercase tracking-wider mb-1">Evaluation error</p>
          <p className="text-sm text-red-300 font-mono leading-relaxed">{evaluation?.error || "Unknown error"}</p>
        </div>
      )}

      {/* Per-metric score cards (once scored) */}
      {showSummary && (
        <div className="space-y-3">
          {results.map((r, i) => {
            const label = EVAL_LABELS[r.evaluator_type] ?? r.evaluator_type;
            if (r.error) {
              return (
                <div key={i} className="bg-th-card-bg border border-th-card-border rounded-2xl px-5 py-4 shadow-sm shadow-black/[0.03]">
                  <div className="flex items-center justify-between mb-1">
                    <p className="text-sm font-semibold text-th-text-primary">{label}</p>
                    <span className="text-[10px] text-red-400">errored</span>
                  </div>
                  <p className="text-xs text-red-300/80 font-mono">{r.error}</p>
                </div>
              );
            }
            const score = r.score ?? 0;
            return (
              <div key={i} className="bg-th-card-bg border border-th-card-border rounded-2xl px-5 py-4 shadow-sm shadow-black/[0.03]">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-sm font-semibold text-th-text-primary">{label}</p>
                  <div className="flex items-center gap-2">
                    <span className={`text-sm font-bold tabular-nums ${scoreColor(score, r.success)}`}>
                      {Math.round(score * 100)}%
                    </span>
                    <span
                      className={`text-[10px] font-medium px-1.5 py-0.5 rounded-full border ${
                        r.success
                          ? "text-emerald-400 bg-emerald-500/10 border-emerald-500/20"
                          : "text-red-400 bg-red-500/10 border-red-500/20"
                      }`}
                    >
                      {r.success ? "Pass" : "Fail"}
                    </span>
                  </div>
                </div>
                <div className="h-1.5 rounded-full bg-th-inset-bg overflow-hidden mb-2">
                  <div
                    className={`h-full rounded-full ${score >= 0.8 ? "bg-emerald-400" : score >= 0.5 ? "bg-amber-400" : "bg-red-400"}`}
                    style={{ width: `${Math.round(score * 100)}%` }}
                  />
                </div>
                {r.reason && (
                  <p className="text-xs text-th-text-secondary leading-relaxed">{r.reason}</p>
                )}
                {r.threshold != null && (
                  <p className="text-[10px] text-th-text-muted/60 mt-1">threshold {Math.round(r.threshold * 100)}%</p>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function RunDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { notifications, clearSession } = useNotification();

  const [run, setRun] = useState<RunInfo | null>(null);
  const [timeline, setTimeline] = useState<SessionTimeline | null>(null);
  const [sessionMessages, setSessionMessages] = useState<Record<string, unknown>[]>([]);
  const [sessionFiles, setSessionFiles] = useState<SessionFile[]>([]);
  const [liveEvents, setLiveEvents] = useState<TimelineEvent[]>([]);
  const [searchParams] = useSearchParams();
  const [tab, setTab] = useState<Tab>(() => {
    const t = searchParams.get("tab");
    return (t === "evaluation" || t === "graph" || t === "results" || t === "files" || t === "metrics")
      ? (t as Tab)
      : "timeline";
  });

  // When navigating between different runs (same component instance, new id param),
  // re-read the tab from the URL so the previous run's active tab doesn't bleed over.
  useEffect(() => {
    const t = searchParams.get("tab");
    setTab(
      (t === "evaluation" || t === "graph" || t === "results" || t === "files" || t === "metrics")
        ? (t as Tab)
        : "timeline",
    );
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);
  const [loading, setLoading] = useState(true);
  const [evaluation, setEvaluation] = useState<RunEvaluation | null>(null);
  const [autoEvaluate, setAutoEvaluate] = useState(false);
  const [analyzeErrors, setAnalyzeErrors] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const timelineEndRef = useRef<HTMLDivElement | null>(null);

  // Fetch run info + timeline + session messages (for graph)
  const fetchData = useCallback(async () => {
    if (!id) return;
    try {
      const [runsData, tlData, msgsData, filesData] = await Promise.all([
        api.listRuns({ limit: 100 }),
        api.getSessionTimeline(id),
        api.getSessionMessages(id).catch(() => [] as Record<string, unknown>[]),
        api.listSessionFiles(id).catch(() => [] as SessionFile[]),
      ]);
      const found = runsData.runs.find((r) => r.session_id === id || r.id === id);
      if (found) setRun(found);
      setTimeline(tlData);
      setSessionMessages(msgsData);
      setSessionFiles(filesData);
    } catch (e) {
      console.warn("[RunDetail] fetch failed:", e);
    } finally {
      setLoading(false);
    }
  }, [id]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Load existing evaluation + the auto-evaluate setting (controls the button).
  useEffect(() => {
    if (!id) return;
    api.getRunEvaluation(id).then(setEvaluation).catch(() => {});
    api.getSettings()
      .then((s) => {
        setAutoEvaluate(Boolean(s?.evaluation?.auto_evaluate));
        setAnalyzeErrors(Boolean(s?.evaluation?.analyze_errors));
      })
      .catch(() => {});
  }, [id]);

  // Whether this run's evaluation is handled automatically: completed runs key
  // off auto_evaluate, failed runs off analyze_errors. Controls the manual
  // button (hidden when auto-handled) and its guard.
  const autoHandled = run?.status === "error" ? analyzeErrors : autoEvaluate;

  const handleEvaluate = useCallback(async () => {
    if (!id || autoHandled) return;
    setEvaluating(true);
    setTab("evaluation");
    try {
      // Kicks off a background eval and returns the initial running state;
      // the polling effect below streams the evaluator's steps in live.
      const result = await api.runRunEvaluation(id);
      setEvaluation(result);
    } catch {
      /* ignore */
    } finally {
      setEvaluating(false);
    }
  }, [id, autoHandled]);

  // Track whether we've seen this run actively running, so auto-eval polling
  // only kicks in for a run that just finished (not stale completed runs).
  const sawRunningRef = useRef(false);
  if (run?.status === "running") sawRunningRef.current = true;

  // Poll the evaluation endpoint to reveal the evaluator's trace live.
  const pollEval = useCallback(async () => {
    if (!id) return;
    const st = evaluation?.status;
    const terminal = st === "done" || st === "skipped" || st === "error";
    // Auto-eval kicks off in the background right as a run completes. Keep
    // polling while we expect it — either because we watched the run finish in
    // this session, or because it completed recently (covers background
    // schedule/trigger runs the user opens after the fact). The recency window
    // bounds polling so we don't hammer the endpoint for long-finished runs.
    const finishedAt = run?.finished_at ? new Date(run.finished_at).getTime() : 0;
    const finishedRecently = finishedAt > 0 && Date.now() - finishedAt < 5 * 60_000;
    const expectingAuto =
      !terminal &&
      ((autoEvaluate && run?.status === "completed") ||
        (analyzeErrors && run?.status === "error")) &&
      (sawRunningRef.current || finishedRecently);
    if (!evaluating && st !== "running" && !expectingAuto) return;
    try {
      const data = await api.getRunEvaluation(id);
      setEvaluation(data);
    } catch {
      /* ignore */
    }
  }, [id, evaluation?.status, evaluating, autoEvaluate, analyzeErrors, run?.status, run?.finished_at]);

  usePolling(pollEval, 1500);

  // Viewing a run clears its pending notification (incl. a HITL "needs
  // feedback" chip) so the bottom-left chip disappears once the user is here.
  useEffect(() => {
    const sid = run?.session_id ?? id;
    if (sid && notifications[sid]) clearSession(sid);
  }, [id, run, notifications, clearSession]);

  // Lightweight status poll while running
  const pollStatus = useCallback(async () => {
    if (!id || !run) return;
    if (run.status !== "running" && run.status !== "awaiting_input") return;
    try {
      const [tlData, runsData, msgsData] = await Promise.all([
        api.getSessionTimeline(id),
        api.listRuns({ limit: 1 }),
        api.getSessionMessages(id).catch(() => null),
      ]);
      setTimeline(tlData);
      const found = runsData.runs.find((r) => r.session_id === id || r.id === id);
      if (found) setRun(found);
      if (msgsData) setSessionMessages(msgsData);
    } catch {/* ignore */}
  }, [id, run]);

  usePolling(pollStatus, POLL_MS);

  // WebSocket for live streaming events
  useEffect(() => {
    if (!id || run?.status !== "running") return;
    const ws = new WebSocket(`${WS_BASE}/ws/chat/${id}`);
    wsRef.current = ws;

    ws.onmessage = (evt) => {
      try {
        const raw = JSON.parse(evt.data as string) as {
          type: string;
          content?: unknown;
          metadata?: Record<string, unknown>;
        };
        // Feed the raw message (nested metadata shape) into sessionMessages so the
        // graph updates immediately instead of waiting for the next poll.
        if (raw.type === "tool_call" || raw.type === "tool_result" || raw.type === "agent") {
          setSessionMessages((prev) => [...prev, raw as unknown as Record<string, unknown>]);
        }
        // Normalise into the flattened TimelineEvent shape the timeline renderer
        // expects (top-level subagent/args/tool, "assistant" instead of "agent").
        const meta = raw.metadata ?? {};
        const ev: TimelineEvent = {
          type: raw.type === "agent" ? "assistant" : raw.type,
          content: raw.content,
          subagent: meta.subagent as string | undefined,
          args: meta.args as Record<string, unknown> | undefined,
          tool: raw.type === "tool_result"
            ? (meta.name as string | undefined)
            : (typeof raw.content === "string" ? raw.content : undefined),
          tool_call_id: meta.tool_call_id as string | undefined,
          images: meta.images as { base64: string; mime_type: string }[] | undefined,
        };
        setLiveEvents((prev) => [...prev, ev]);
      } catch { /* ignore */ }
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [id, run?.status]);

  // Combine persisted + live events
  const allEvents: TimelineEvent[] = [
    ...(timeline?.events ?? []),
    ...liveEvents,
  ];

  // Append a synthetic error event when the run failed but the session
  // transcript doesn't already contain one (schedule/trigger runs write the
  // error into run.json rather than the chat stream).
  const hasErrorEvent = allEvents.some(ev => ev.type === "error");
  if (run?.status === "error" && run.error && !hasErrorEvent) {
    allEvents.push({
      type: "error",
      content: run.error,
      ts: run.finished_at ?? undefined,
    } as TimelineEvent);
  }

  // While the run is live, keep the Timeline pinned to the bottom so the
  // latest agent turn stays visible as new events stream in.
  useEffect(() => {
    if (tab !== "timeline" || run?.status !== "running") return;
    timelineEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [allEvents.length, tab, run?.status]);

  // Per-tool durations (tool_call_id → ms) come from the timeline endpoint,
  // which derives them from transcript timestamps. The graph reuses them.
  const durationByCallId = useMemo(() => {
    const acc: Record<string, number> = {};
    for (const ev of [...(timeline?.events ?? []), ...liveEvents]) {
      if (ev.tool_call_id && ev.duration_ms != null) acc[ev.tool_call_id] = ev.duration_ms;
    }
    return acc;
  }, [timeline, liveEvents]);

  // Use real session messages for AgentGraph — they carry the exact metadata shape
  // (args, subagent, tool_call_id) that buildGraph expects.
  const chatMessages = sessionMessages.map((m, i) => ({
    id: (m.id as string) ?? `msg-${i}`,
    type: (m.type as import("../types").WSMessageType) ?? "agent",
    content: (m.content as string) ?? "",
    metadata: (m.metadata as Record<string, unknown>) ?? {},
    timestamp: new Date(),
  }));

  // Tool usage stats for Metrics tab
  const toolCounts = allEvents.reduce<Record<string, number>>((acc, ev) => {
    if (ev.type === "tool_call" && ev.tool) acc[ev.tool] = (acc[ev.tool] ?? 0) + 1;
    return acc;
  }, {});
  const topTools = Object.entries(toolCounts)
    .sort((a, b) => b[1] - a[1])
    .map(([tool, count]) => ({ label: tool, count }));

  // Average per-tool duration
  const toolDurations = allEvents
    .filter((ev) => ev.type === "tool_result" && ev.duration_ms != null)
    .reduce<Record<string, number[]>>((acc, ev) => {
      const tool = ev.tool ?? "tool";
      if (!acc[tool]) acc[tool] = [];
      acc[tool].push(ev.duration_ms!);
      return acc;
    }, {});

  const { Icon: SrcIcon, className: srcCls } = getSourceIcon(run?.trigger_source ?? null);
  const { Icon: ProvIcon, className: provCls } = getProviderIcon(run?.llm_provider);

  // --- Actions ---
  const handleStop = async () => {
    if (!id) return;
    setStopping(true);
    try { await api.stopSession(id); } catch { /* ignore */ }
    setStopping(false);
    fetchData();
  };

  const handleDelete = async () => {
    if (!id) return;
    setDeleting(true);
    try {
      await api.closeSession(id);
      navigate("/runs");
    } catch { /* ignore */ }
    setDeleting(false);
  };

  const evalRunning = evaluation?.status === "running" || evaluating;
  const tabs: { id: Tab; label: string; icon: typeof MessageSquare; badge?: number; pulse?: boolean }[] = [
    { id: "timeline", label: "Timeline", icon: MessageSquare },
    { id: "graph",    label: "Graph",    icon: GitBranch },
    { id: "results",  label: "Results",  icon: FileOutput },
    { id: "files",    label: "Files",    icon: FolderOpen, badge: sessionFiles.length || undefined },
    { id: "metrics",  label: "Metrics",  icon: BarChart2 },
    { id: "evaluation", label: "Evaluation", icon: Gauge, pulse: evalRunning },
  ];

  return (
    <div className="h-full flex flex-col">
      {/* Page header */}
      <header className="border-b border-th-border/70 px-6 py-3.5 bg-th-bg-secondary/80 backdrop-blur-xl shrink-0">
        {/* Top row: back + title + actions */}
        <div className="flex items-center gap-3 mb-2">
          <button
            onClick={() => navigate("/runs")}
            className="flex items-center gap-1.5 text-th-text-muted hover:text-th-text-primary transition-colors text-sm"
          >
            <ArrowLeft size={14} aria-hidden />
            Runs
          </button>
          <span className="text-th-border">/</span>
          {loading ? (
            <div className="h-5 w-48 bg-th-inset-bg rounded animate-pulse" />
          ) : (
            <h1 className="text-[19px] font-semibold tracking-tight text-th-text-primary truncate flex-1">
              {run?.title || "Untitled"}
            </h1>
          )}
          <div className="flex items-center gap-2 ml-auto shrink-0">
            {run && (
              <button
                onClick={() => navigate(`/chat/${id}`)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-th-border bg-th-inset-bg/70 text-xs font-medium text-th-text-secondary hover:text-th-text-primary hover:border-th-border-strong transition-all duration-200 active:scale-[0.97]"
              >
                <ExternalLink size={12} aria-hidden />
                Open chat
              </button>
            )}
            {run && (
              // Wrapper span carries the tooltip so it still shows when the
              // button is disabled (disabled elements don't fire hover events).
              <span
                title={
                  autoEvaluate
                    ? "Auto-evaluation is enabled — disable it in Settings to evaluate manually."
                    : "Evaluate this run"
                }
              >
                <button
                  onClick={handleEvaluate}
                  disabled={autoEvaluate || evaluating}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl border text-xs font-medium transition-all duration-200 active:scale-[0.97] ${
                    autoEvaluate
                      ? "border-th-border text-th-text-muted/50 cursor-not-allowed opacity-60 active:scale-100"
                      : "border-blue-500/30 text-blue-400 hover:bg-blue-500/10"
                  }`}
                >
                  {evaluating ? <Loader2 size={12} className="animate-spin" /> : <Gauge size={12} aria-hidden />}
                  Evaluate
                </button>
              </span>
            )}
            {run?.status === "running" && (
              <button
                onClick={handleStop}
                disabled={stopping}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl border border-red-500/40 text-xs font-medium text-red-400 hover:bg-red-500/10 transition-all duration-200 active:scale-[0.97] disabled:opacity-50"
              >
                {stopping ? <Loader2 size={12} className="animate-spin" /> : <Square size={12} aria-hidden />}
                Stop
              </button>
            )}
            {confirmDelete ? (
              <div className="flex items-center gap-1">
                <button
                  onClick={handleDelete}
                  disabled={deleting}
                  className="px-2.5 py-1.5 rounded-xl text-xs font-medium bg-red-500/15 text-red-400 hover:bg-red-500/25 transition-all duration-200 active:scale-[0.97]"
                >
                  {deleting ? <Loader2 size={10} className="animate-spin" /> : "Delete"}
                </button>
                <button
                  onClick={() => setConfirmDelete(false)}
                  className="p-1.5 rounded-xl text-th-text-muted hover:text-th-text-secondary transition-all duration-200 active:scale-[0.97]"
                >
                  <X size={12} />
                </button>
              </div>
            ) : (
              <button
                onClick={() => setConfirmDelete(true)}
                className="p-1.5 rounded-xl text-th-text-muted/50 hover:text-red-400 hover:bg-red-500/10 transition-all duration-200 active:scale-[0.97]"
                title="Delete run"
              >
                <Trash2 size={13} />
              </button>
            )}
          </div>
        </div>

        {/* Meta chips row */}
        {run && (
          <div className="flex items-center flex-wrap gap-2 text-xs">
            <RunStatusBadge status={run.status} size="md" />
            <span className="text-th-text-muted">·</span>
            <span className="font-mono text-[11px] text-th-text-muted/60 select-all" title={run.session_id ?? run.id}>
              #{(run.session_id ?? run.id).slice(0, 8)}
            </span>
            <span className="text-th-text-muted">·</span>
            {run.schedule_id ? (
              <button
                onClick={() => navigate(`/schedules/${encodeURIComponent(run.schedule_id!)}/runs`)}
                className="inline-flex items-center gap-1 text-th-text-muted hover:text-blue-400 transition-colors"
                title="View all runs for this schedule"
              >
                <SrcIcon size={11} className={srcCls} aria-hidden />
                {getSourceLabel(run.trigger_source)}
              </button>
            ) : run.trigger_id ? (
              <button
                onClick={() => navigate(`/triggers/${encodeURIComponent(run.trigger_id!)}/runs`)}
                className="inline-flex items-center gap-1 text-th-text-muted hover:text-blue-400 transition-colors"
                title="View all runs for this trigger"
              >
                <SrcIcon size={11} className={srcCls} aria-hidden />
                {getSourceLabel(run.trigger_source)}
              </button>
            ) : (
              <span className="inline-flex items-center gap-1 text-th-text-muted">
                <SrcIcon size={11} className={srcCls} aria-hidden />
                {getSourceLabel(run.trigger_source)}
              </span>
            )}
            {run.agent_name && (
              <>
                <span className="text-th-text-muted">·</span>
                <span className="text-th-text-secondary">{run.agent_name}</span>
              </>
            )}
            {run.llm_provider && (
              <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded border text-[10px] font-medium ${familyChipClasses(run.llm_provider)}`}>
                <ProvIcon size={9} className={provCls} aria-hidden />
                {run.model || run.llm_provider}
              </span>
            )}
            <span className="text-th-text-muted">·</span>
            <span className="inline-flex items-center gap-1 text-th-text-muted">
              <Clock size={10} aria-hidden />
              {formatRelativeTime(run.started_at)}
            </span>
            {run.duration_ms != null && (
              <span className="text-th-text-muted tabular-nums">
                · {formatDuration(run.duration_ms)}
              </span>
            )}
            {(run.input_tokens || run.output_tokens) && (
              <>
                <span className="text-th-text-muted">·</span>
                <span className="inline-flex items-center gap-1 text-th-text-muted tabular-nums">
                  <Coins size={10} aria-hidden />
                  {formatTokens(run.input_tokens)}↑ {formatTokens(run.output_tokens)}↓
                  {run.estimated_cost_usd ? ` · ${formatCost(run.estimated_cost_usd)}` : ""}
                </span>
              </>
            )}
            {evalRunning && (
              <>
                <span className="text-th-text-muted">·</span>
                <button
                  onClick={() => setTab("evaluation")}
                  className="inline-flex items-center gap-1 text-blue-400 hover:text-blue-300 transition-colors"
                  title="Evaluation in progress — click to watch live"
                >
                  <Loader2 size={10} className="animate-spin" aria-hidden />
                  Evaluating…
                </button>
              </>
            )}
          </div>
        )}

        {/* Tabs */}
        <div className="flex items-center gap-1 mt-3 border-b border-th-border -mb-3.5">
          {tabs.map(({ id: tid, label, icon: Icon, badge, pulse }) => (
            <button
              key={tid}
              onClick={() => setTab(tid)}
              className={`flex items-center gap-1.5 px-3 py-2 text-xs font-medium border-b-2 transition-all duration-200 -mb-px rounded-t-lg ${
                tab === tid
                  ? "border-blue-400 text-blue-400"
                  : "border-transparent text-th-text-muted hover:text-th-text-secondary hover:bg-th-surface-hover/60"
              }`}
            >
              <Icon size={12} aria-hidden />
              {label}
              {badge != null && badge > 0 && (
                <span className="ml-0.5 px-1.5 py-px rounded-full text-[9px] font-bold bg-blue-500/15 text-blue-400 border border-blue-500/20 tabular-nums">
                  {badge}
                </span>
              )}
              {pulse && (
                <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse ml-0.5 shrink-0" />
              )}
            </button>
          ))}
        </div>
      </header>

      {/* Graph tab — lives outside the scroll container so height:100% works */}
      {tab === "graph" && (
        <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
          {chatMessages.some(m => m.type === "tool_call") ? (
            <AgentGraph messages={chatMessages} onClose={() => setTab("timeline")} fullPanel durations={durationByCallId} />
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <GitBranch size={28} className="text-th-text-muted mb-3" />
              <p className="text-sm text-th-text-muted">No tool calls found in this run</p>
              <p className="text-xs text-th-text-muted/60 mt-1">Tool calls and subagent delegations appear as nodes</p>
            </div>
          )}
        </div>
      )}

      {/* Tab content (scrollable) */}
      {tab !== "graph" && (
      <div key={tab} className="flex-1 overflow-y-auto animate-fade-in">
        {tab === "timeline" && (
          <div className="max-w-5xl mx-auto px-6 py-4">
            {loading ? (
              <div className="space-y-3">
                {[...Array(5)].map((_, i) => (
                  <div key={i} className="h-12 rounded-lg bg-th-inset-bg animate-pulse" />
                ))}
              </div>
            ) : (
              <>
                <RunTimeline events={allEvents} />
                <div ref={timelineEndRef} />
              </>
            )}
          </div>
        )}

        {tab === "results" && (
          <div className="max-w-5xl mx-auto px-6 py-4">
            {(() => {
              // Collect all assistant messages (not just the last), for full context
              const agentEvents = allEvents.filter(
                (ev) => ev.type === "assistant" && typeof ev.content === "string" && (ev.content as string).trim(),
              );
              if (agentEvents.length === 0) {
                return (
                  <div className="flex flex-col items-center justify-center py-16 text-center">
                    <FileOutput size={28} className="text-th-text-muted mb-3" />
                    <p className="text-sm text-th-text-muted">No response yet</p>
                  </div>
                );
              }
              const finalEvent = agentEvents[agentEvents.length - 1];
              const finalText = typeof finalEvent.content === "string" ? finalEvent.content : "";
              return (
                <div className="space-y-4">
                  {/* Final response card */}
                  <div className="bg-th-card-bg border border-th-card-border rounded-2xl overflow-hidden shadow-sm shadow-black/[0.03]">
                    <div className="px-5 py-3 border-b border-th-border/50 flex items-center justify-between">
                      <p className="text-[11px] font-semibold text-th-text-muted uppercase tracking-wider">Final response</p>
                      <span className="text-[10px] text-th-text-muted/60">
                        {finalEvent.ts ? formatRelativeTime(finalEvent.ts) : ""}
                      </span>
                    </div>
                    <div className="px-5 py-4 prose prose-sm dark:prose-invert max-w-none
                      prose-p:my-2 prose-p:leading-relaxed
                      prose-h1:text-base prose-h1:font-bold prose-h1:mt-4 prose-h1:mb-2
                      prose-h2:text-sm prose-h2:font-semibold prose-h2:mt-3 prose-h2:mb-1.5
                      prose-h3:text-[13px] prose-h3:font-semibold prose-h3:mt-2 prose-h3:mb-1
                      prose-ul:my-2 prose-ul:pl-4 prose-li:my-0.5
                      prose-ol:my-2 prose-ol:pl-4
                      prose-code:text-[12px] prose-code:bg-th-inset-bg prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:before:content-none prose-code:after:content-none
                      prose-pre:bg-th-inset-bg prose-pre:border prose-pre:border-th-border/60 prose-pre:rounded-lg prose-pre:text-xs
                      prose-blockquote:border-blue-400/40 prose-blockquote:text-th-text-secondary
                      prose-a:text-blue-400 prose-a:no-underline hover:prose-a:underline
                      prose-strong:text-th-text-primary prose-strong:font-semibold
                      prose-hr:border-th-border/40
                      text-th-text-primary">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{finalText}</ReactMarkdown>
                    </div>
                  </div>

                  {/* If there were multiple agent turns, show a collapsible history */}
                  {agentEvents.length > 1 && (
                    <AgentTurns events={agentEvents.slice(0, -1)} />
                  )}
                </div>
              );
            })()}
          </div>
        )}

        {tab === "files" && (
          <div className="px-6 py-4">
            <FilesTab sessionId={id!} files={sessionFiles} />
          </div>
        )}

        {tab === "metrics" && run && (
          <div className="max-w-5xl mx-auto px-6 py-4 space-y-4">
            {/* Token + cost summary */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <StatCard label="Input tokens" value={formatTokens(run.input_tokens)} icon={ChevronUp} />
              <StatCard label="Output tokens" value={formatTokens(run.output_tokens)} icon={ChevronDown} />
              <StatCard label="Est. cost" value={formatCost(run.estimated_cost_usd)} icon={Coins} />
              <StatCard label="Duration" value={formatDuration(run.duration_ms)} icon={Clock} />
            </div>

            {/* MLX throughput summary — only render tiles whose value is present */}
            {(() => {
              const mlxTiles = [
                run.avg_prefill_tps != null && { label: "TIPS (prefill)", value: `${run.avg_prefill_tps.toFixed(0)} t/s`, icon: Gauge },
                run.avg_generation_tps != null && { label: "TOPS (generation)", value: `${run.avg_generation_tps.toFixed(0)} t/s`, icon: Zap },
                run.cache_hit_ratio != null && { label: "KV cache hit", value: `${Math.round(run.cache_hit_ratio * 100)}%`, icon: Database },
                run.peak_memory_gb != null && { label: "Peak GPU", value: `${run.peak_memory_gb.toFixed(1)} GB`, icon: Cpu },
              ].filter(Boolean) as { label: string; value: string; icon: typeof Gauge }[];
              if (mlxTiles.length === 0) return null;
              return (
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  {mlxTiles.map(({ label, value, icon }) => (
                    <StatCard key={label} label={label} value={value} icon={icon} />
                  ))}
                </div>
              );
            })()}

            {topTools.length > 0 && (
              <BreakdownBar
                title="Tool usage"
                items={topTools}
              />
            )}

            {Object.keys(toolDurations).length > 0 && (
              <div className="bg-th-card-bg border border-th-card-border rounded-2xl p-4 shadow-sm shadow-black/[0.03]">
                <h3 className="text-[11px] text-th-text-muted font-semibold uppercase tracking-wider mb-3 flex items-center gap-1.5">
                  <Wrench size={11} aria-hidden />
                  Avg tool duration
                </h3>
                <div className="space-y-2">
                  {Object.entries(toolDurations)
                    .map(([tool, durations]) => ({
                      tool,
                      avg: durations.reduce((a, b) => a + b, 0) / durations.length,
                    }))
                    .sort((a, b) => b.avg - a.avg)
                    .slice(0, 10)
                    .map(({ tool, avg }) => (
                      <div key={tool} className="flex items-center gap-2">
                        <span className="text-[11px] text-th-text-secondary truncate" style={{ minWidth: "8rem" }}>{tool}</span>
                        <span className="text-[11px] text-th-text-muted tabular-nums">{formatDuration(avg)}</span>
                      </div>
                    ))}
                </div>
              </div>
            )}

            {run.error && (
              <div className="bg-red-500/5 border border-red-500/20 rounded-2xl p-4 shadow-sm shadow-black/[0.03]">
                <p className="text-[10px] font-semibold text-red-400 uppercase tracking-wider mb-1">Error</p>
                <p className="text-sm text-red-300 font-mono leading-relaxed">{run.error}</p>
              </div>
            )}
          </div>
        )}

        {tab === "evaluation" && (
          <div className="max-w-5xl mx-auto px-6 py-4">
            <EvaluationTab
              evaluation={evaluation}
              evaluating={evaluating}
              autoEvaluate={autoHandled}
              onEvaluate={handleEvaluate}
              run={run}
            />
          </div>
        )}
      </div>
      )}
    </div>
  );
}
