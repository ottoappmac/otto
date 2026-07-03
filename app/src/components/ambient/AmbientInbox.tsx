/**
 * AmbientInbox — full-page inbox for ambient assistant hints.
 *
 * Each hint card shows:
 *   • Title + rationale
 *   • Source chips (memory / sessions / activity / history)
 *   • "Open in chat" button — navigates to /chat with the proposed prompt pre-filled
 *   • "Approve & run" button (shown only when allow_auto_run is enabled)
 *   • Dismiss / Snooze (4h) actions
 *
 * The component is intentionally self-contained so it can also be embedded
 * in a tray popover or a sheet in the future.
 */
import { useEffect, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
  Sparkles,
  MessageSquare,
  Play,
  X,
  Clock,
  RefreshCw,
  Loader2,
  CheckCircle,
  Check,
  ChevronDown,
  ChevronUp,
  Copy,
  Zap,
  Brain,
  Monitor,
  History,
  BookOpen,
  Trash2,
  CalendarClock,
  Webhook,
  ListTodo,
  Gauge,
} from "lucide-react";
import { api } from "../../hooks/useApi";
import { useAmbientHints } from "../../hooks/useAmbientHints";
import { useAmbientSweepStatus } from "../../hooks/useAmbientSweepStatus";
import type { AmbientHint } from "../../types";

// ---------------------------------------------------------------------------
// Source chip
// ---------------------------------------------------------------------------

const SOURCE_META: Record<string, { label: string; icon: typeof Brain }> = {
  memory:    { label: "Memory",    icon: Brain },
  sessions:  { label: "Sessions",  icon: MessageSquare },
  activity:  { label: "Activity",  icon: Monitor },
  history:   { label: "History",   icon: History },
  schedules: { label: "Schedules", icon: CalendarClock },
  triggers:  { label: "Triggers",  icon: Webhook },
  evaluation:{ label: "Evaluation",icon: Gauge },
};

// Show at most this many source chips inline; the rest collapse into a "+N".
const MAX_VISIBLE_SOURCES = 3;

function SourceChip({ source }: { source: string }) {
  const meta = SOURCE_META[source] ?? { label: source, icon: BookOpen };
  const Icon = meta.icon;
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium border bg-th-surface-hover text-th-text-muted border-th-border">
      <Icon size={9} />
      {meta.label}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Sweep progress banner
// ---------------------------------------------------------------------------

const SWEEP_STAGES = [
  { icon: Brain,        label: "Reading long-term memory…" },
  { icon: MessageSquare,label: "Scanning recent sessions…" },
  { icon: Monitor,      label: "Reviewing macOS activity…" },
  { icon: History,      label: "Checking usage history…" },
  { icon: CalendarClock,label: "Checking existing automations…" },
  { icon: Sparkles,     label: "Generating suggestions…" },
];

function SweepProgressBanner() {
  const [stageIdx, setStageIdx] = useState(0);

  useEffect(() => {
    const id = setInterval(() => {
      setStageIdx((i) => (i + 1) % SWEEP_STAGES.length);
    }, 2800);
    return () => clearInterval(id);
  }, []);

  const stage = SWEEP_STAGES[stageIdx];
  const Icon  = stage.icon;

  return (
    <div className="mb-6 rounded-xl border border-blue-500/25 bg-blue-500/8 px-5 py-4">
      <div className="flex items-center gap-3 mb-3">
        {/* Orbital spinner */}
        <div className="relative w-8 h-8 shrink-0">
          <div className="absolute inset-0 rounded-full border-2 border-blue-500/20" />
          <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-blue-400 animate-spin" />
          <div className="absolute inset-1.5 flex items-center justify-center">
            <Icon size={12} className="text-blue-400" />
          </div>
        </div>
        <div className="min-w-0">
          <p className="text-sm font-medium text-blue-300">Sweeping…</p>
          <p
            key={stageIdx}
            className="text-xs text-blue-400/70 mt-0.5 transition-opacity duration-300"
          >
            {stage.label}
          </p>
        </div>
      </div>

      {/* Stage dots */}
      <div className="flex items-center gap-1.5 ml-11">
        {SWEEP_STAGES.map((_, i) => (
          <div
            key={i}
            className={`h-1 rounded-full transition-all duration-500 ${
              i < stageIdx
                ? "w-4 bg-blue-400"
                : i === stageIdx
                  ? "w-4 bg-blue-300 animate-pulse"
                  : "w-1.5 bg-blue-500/30"
            }`}
          />
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Hint card
// ---------------------------------------------------------------------------

interface HintCardProps {
  hint: AmbientHint;
  highlighted?: boolean;
  allowAutoRun: boolean;
  onOpenInChat: (hint: AmbientHint) => void;
  onApproveRun: (hint: AmbientHint) => Promise<void>;
  onApply: (hint: AmbientHint) => Promise<void>;
  onDismiss: (id: string) => Promise<void>;
  onSnooze: (id: string) => Promise<void>;
}

function HintCard({
  hint,
  highlighted = false,
  allowAutoRun,
  onOpenInChat,
  onApproveRun,
  onApply,
  onDismiss,
  onSnooze,
}: HintCardProps) {
  const [busy, setBusy] = useState(false);
  const [confirmingApply, setConfirmingApply] = useState(false);
  const [promptExpanded, setPromptExpanded] = useState(false);
  const [copied, setCopied] = useState(false);
  const isActioned = hint.status === "accepted" || hint.status === "dismissed";
  const isEval = hint.origin === "evaluation";
  const applyTarget = hint.target_kind === "schedule" || hint.target_kind === "trigger"
    ? hint.target_kind
    : null;
  const canApply = isEval && applyTarget !== null;

  const handle = async (fn: () => Promise<void>) => {
    setBusy(true);
    try {
      await fn();
    } finally {
      setBusy(false);
    }
  };

  const KIND_META: Record<string, { label: string; icon: typeof Sparkles; badgeColor: string; iconBg: string; iconColor: string }> = {
    task:       { label: "Task",       icon: ListTodo,      badgeColor: "bg-blue-500/15 text-blue-300 border-blue-500/25",    iconBg: "bg-blue-500/10 border-blue-500/20",    iconColor: "text-blue-400" },
    automation: { label: "Automate",   icon: Zap,           badgeColor: "bg-amber-500/15 text-amber-300 border-amber-500/25", iconBg: "bg-amber-500/10 border-amber-500/20", iconColor: "text-amber-400" },
    schedule:   { label: "Schedule",   icon: CalendarClock, badgeColor: "bg-violet-500/15 text-violet-300 border-violet-500/25", iconBg: "bg-violet-500/10 border-violet-500/20", iconColor: "text-violet-400" },
    trigger:    { label: "Trigger",    icon: Webhook,       badgeColor: "bg-rose-500/15 text-rose-300 border-rose-500/25",    iconBg: "bg-rose-500/10 border-rose-500/20",    iconColor: "text-rose-400" },
  };
  const kindMeta = KIND_META[hint.kind] ?? KIND_META.task;
  const KindIcon = kindMeta.icon;

  return (
    <div
      id={`hint-card-${hint.id}`}
      className={`bg-th-card-bg border rounded-xl p-4 transition-all duration-200 ${
        isActioned ? "opacity-40 pointer-events-none" : ""
      } ${
        highlighted
          ? "border-amber-400/60 ring-2 ring-amber-400/30 shadow-lg shadow-amber-500/10"
          : "border-th-card-border"
      }`}
    >
      {/* Header */}
      <div className="flex items-start gap-3">
        <div className={`p-2 rounded-lg border shrink-0 mt-0.5 ${kindMeta.iconBg}`}>
          <KindIcon size={16} className={kindMeta.iconColor} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap mb-1">
            <h3 className="text-sm font-semibold text-th-text-primary truncate">{hint.title}</h3>
            {hint.kind === "schedule" && hint.schedule_cron && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-mono border bg-neutral-500/10 text-neutral-400 border-neutral-500/20">
                <Clock size={9} />
                {hint.schedule_cron}
              </span>
            )}
          </div>
          <p className="text-xs text-th-text-muted leading-relaxed">{hint.rationale}</p>
          {hint.sources.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-2">
              {hint.sources.slice(0, MAX_VISIBLE_SOURCES).map((s) => <SourceChip key={s} source={s} />)}
              {hint.sources.length > MAX_VISIBLE_SOURCES && (
                <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-medium border bg-th-surface-hover text-th-text-muted border-th-border">
                  +{hint.sources.length - MAX_VISIBLE_SOURCES}
                </span>
              )}
            </div>
          )}
        </div>

        {/* Snooze / dismiss */}
        {!isActioned && (
          <div className="flex items-center gap-1 shrink-0">
            <button
              onClick={() => handle(() => onSnooze(hint.id))}
              disabled={busy}
              title="Snooze 4 hours"
              className="p-1.5 rounded-md text-th-text-muted hover:text-th-text-secondary hover:bg-th-surface-hover transition-all"
            >
              <span className="text-xs font-bold leading-none tracking-tight">zz</span>
            </button>
            <button
              onClick={() => handle(() => onDismiss(hint.id))}
              disabled={busy}
              title="Dismiss"
              className="p-1.5 rounded-md text-th-text-muted hover:text-red-400 hover:bg-th-surface-hover transition-all"
            >
              <X size={14} />
            </button>
          </div>
        )}
      </div>

      {/* Proposed prompt — collapsed by default behind a subtle toggle */}
      <div className="mt-2.5 ml-11">
        <button
          type="button"
          onClick={() => setPromptExpanded((v) => !v)}
          className="inline-flex items-center gap-1 text-[11px] text-th-text-muted hover:text-th-text-secondary transition-colors"
          aria-expanded={promptExpanded}
        >
          {promptExpanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
          {promptExpanded ? "Hide prompt" : "Show suggested prompt"}
        </button>

        {promptExpanded && (
          <div className="mt-2 bg-th-inset-bg border border-th-border rounded-lg overflow-hidden">
            <p className="px-3 py-2 text-xs text-th-text-secondary whitespace-pre-wrap break-words">
              {hint.proposed_prompt}
            </p>
            <div className="flex justify-end px-3 pb-2">
              <button
                type="button"
                onClick={() => {
                  navigator.clipboard.writeText(hint.proposed_prompt).catch(() => {});
                  setCopied(true);
                  setTimeout(() => setCopied(false), 1500);
                }}
                className="inline-flex items-center gap-1 text-[10px] text-th-text-muted hover:text-th-text-secondary transition-colors"
                title="Copy to clipboard"
              >
                {copied ? <Check size={11} className="text-emerald-400" /> : <Copy size={11} />}
                {copied ? "Copied" : "Copy"}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Actions */}
      {!isActioned && canApply && (
        <div className="mt-3 ml-11">
          {confirmingApply ? (
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-[11px] text-th-text-muted">
                Replace the {applyTarget} prompt with this improved version?
              </span>
              <button
                onClick={() => handle(() => onApply(hint))}
                disabled={busy}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-emerald-500/10 text-emerald-300 border border-emerald-500/20 hover:bg-emerald-500/20 transition-all"
              >
                {busy ? <Loader2 size={12} className="animate-spin" /> : <Check size={12} />}
                Confirm
              </button>
              <button
                onClick={() => setConfirmingApply(false)}
                disabled={busy}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-th-text-muted hover:text-th-text-secondary hover:bg-th-surface-hover transition-all"
              >
                Cancel
              </button>
            </div>
          ) : (
            <button
              onClick={() => setConfirmingApply(true)}
              disabled={busy}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-blue-500/10 text-blue-300 border border-blue-500/20 hover:bg-blue-500/20 transition-all"
            >
              {applyTarget === "schedule" ? <CalendarClock size={12} /> : <Webhook size={12} />}
              Apply to {applyTarget}
            </button>
          )}
        </div>
      )}

      {!isActioned && !canApply && (
        <div className="flex items-center gap-2 mt-3 ml-11">
          <button
            onClick={() => onOpenInChat(hint)}
            disabled={busy}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-blue-500/10 text-blue-300 border border-blue-500/20 hover:bg-blue-500/20 transition-all"
          >
            <MessageSquare size={12} />
            {!isEval && (hint.kind === "schedule" || hint.kind === "trigger")
              ? "Set up in chat"
              : "Open in chat"}
          </button>

          {allowAutoRun && (
            <button
              onClick={() => handle(() => onApproveRun(hint))}
              disabled={busy}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-emerald-500/10 text-emerald-300 border border-emerald-500/20 hover:bg-emerald-500/20 transition-all"
            >
              {busy ? <Loader2 size={12} className="animate-spin" /> : <Play size={12} />}
              {isEval ? "Run improved prompt" : "Approve & run"}
            </button>
          )}
        </div>
      )}

      {/* Accepted indicator */}
      {hint.status === "accepted" && (
        <div className="flex items-center gap-1.5 mt-3 ml-11 text-xs text-emerald-400">
          <CheckCircle size={12} />
          Accepted
        </div>
      )}
    </div>
  );
}


const SKIP_MESSAGES: Record<string, { title: string; detail: string }> = {
  no_model: {
    title: "No model configured",
    detail: "Set up an ambient model in Settings → Agent Memory → Ambient.",
  },
  no_context: {
    title: "Not enough context yet",
    detail: "Add memory topics or run a few chat sessions so the agent has data to work with.",
  },
  no_valid_hints: {
    title: "Nothing notable right now",
    detail: "The model didn't find anything worth suggesting. Try again later.",
  },
  empty_response: {
    title: "Model returned no response",
    detail: "Check your ambient model configuration in Settings → Agent Memory → Ambient.",
  },
  error: {
    title: "Sweep encountered an error",
    detail: "Check the backend logs for details.",
  },
};

export default function AmbientInbox() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [allowAutoRun, setAllowAutoRun] = useState(false);
  // Local state tracks a sweep triggered from this page; the backend status
  // hook covers sweeps started elsewhere (e.g. periodic or post-session).
  const [localSweeping, setLocalSweeping] = useState(false);
  const backendSweeping = useAmbientSweepStatus(3000);
  const sweeping = localSweeping || backendSweeping;
  const [runError, setRunError] = useState<string | null>(null);
  const [skipReason, setSkipReason] = useState<string | null>(null);
  const [highlightedId, setHighlightedId] = useState<string | null>(null);
  const [showRecent, setShowRecent] = useState(false);
  const [activeTab, setActiveTab] = useState<"evaluation" | "activity">("evaluation");
  const highlightTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const { hints, pendingCount, quietHours, accept, dismiss, snooze, triggerSweep, markSeen } =
    useAmbientHints();

  // Mark all currently visible pending hints as "seen" so Layout doesn't
  // re-show the notification banner when the user is already on this page.
  useEffect(() => {
    markSeen();
  // markSeen is stable (no deps), fire once on mount.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Scroll to and briefly highlight a hint when navigated here with ?highlight=<id>.
  const highlightParam = searchParams.get("highlight");
  useEffect(() => {
    if (!highlightParam || hints.length === 0) return;

    // Clear the URL param immediately so back-navigation doesn't re-trigger.
    setSearchParams((prev) => {
      const next = new URLSearchParams(prev);
      next.delete("highlight");
      return next;
    }, { replace: true });

    // Give the DOM one frame to render the card before scrolling.
    const frameId = requestAnimationFrame(() => {
      const el = document.getElementById(`hint-card-${highlightParam}`);
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "center" });
      }
      setHighlightedId(highlightParam);
      if (highlightTimerRef.current) clearTimeout(highlightTimerRef.current);
      highlightTimerRef.current = setTimeout(() => setHighlightedId(null), 2200);
    });

    return () => {
      cancelAnimationFrame(frameId);
      if (highlightTimerRef.current) clearTimeout(highlightTimerRef.current);
    };
  // Re-run when hints become available (they may not be loaded yet on first render).
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [highlightParam, hints.length]);

  // Load allow_auto_run from status on mount.
  useEffect(() => {
    api.ambientStatus()
      .then((s) => setAllowAutoRun(s.allow_auto_run))
      .catch(() => {});
  }, []);

  // Navigate to /chat with the proposed prompt pre-filled.
  const handleOpenInChat = (hint: AmbientHint) => {
    accept(hint.id, "chat").catch(() => {});
    // Store the draft in localStorage so ChatPage picks it up.
    localStorage.setItem("chatDraft", hint.proposed_prompt);
    if (hint.suggested_agent) {
      localStorage.setItem("chatSelectedAgent", hint.suggested_agent);
    }
    navigate("/chat");
  };

  const handleApproveRun = async (hint: AmbientHint) => {
    const { session_id } = await accept(hint.id, "run");
    if (session_id) {
      // Mark this session as ambient-spawned so ChatPage bypasses the
      // "no history yet → redirect away" guard while kick_off_message
      // is still being queued on the backend.
      localStorage.setItem("ambientRunSession", session_id);
      navigate(`/chat/${session_id}`);
    }
  };

  // Eval-triggered schedule/trigger suggestion: update the stored prompt.
  const handleApply = async (hint: AmbientHint) => {
    await accept(hint.id, "apply");
  };

  const [clearingAll, setClearingAll] = useState(false);

  const handleManualSweep = async () => {
    setLocalSweeping(true);
    setRunError(null);
    setSkipReason(null);
    try {
      const result = await triggerSweep();
      if (result.hints_added === 0 && result.skipped) {
        setSkipReason(result.skipped);
      }
    } catch (e: unknown) {
      setRunError(e instanceof Error ? e.message : "Sweep failed");
    } finally {
      setLocalSweeping(false);
    }
  };

  const byNewest = (a: AmbientHint, b: AmbientHint) => b.created_at - a.created_at;
  const actionable = hints.filter((h) => h.status === "pending" || h.status === "shown").sort(byNewest);

  // For eval hints, keep only the most-recent suggestion per target so
  // repeated low-scoring runs don't stack multiple cards for the same
  // schedule, trigger, or manual session.
  const evalHintsDeduped = (() => {
    const seen = new Set<string>();
    return actionable.filter((h) => {
      if (h.origin !== "evaluation") return false;
      // Key: target_kind + target_id (or session_id for manual runs).
      const key = h.target_kind && h.target_id
        ? `${h.target_kind}:${h.target_id}`
        : `manual:${h.session_id ?? h.id}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  })();

  const activityHints = actionable.filter((h) => h.origin !== "evaluation");
  // Alias for readability in the render section.
  const evalHints = evalHintsDeduped;
  const recentActioned = hints.filter((h) => h.status === "accepted" || h.status === "dismissed").sort(byNewest).slice(0, 5);

  // Tab/chip switcher between the two suggestion groups. Both tabs are always
  // shown — even when a group is empty — so the switcher persists and users can
  // browse either category at any time.
  const tabs = [
    { id: "evaluation" as const, label: "Evaluation triggered", icon: Gauge, accent: "text-amber-400", count: evalHints.length },
    { id: "activity" as const, label: "Activity", icon: Monitor, accent: "text-emerald-400", count: activityHints.length },
  ];
  const effectiveTab = activeTab;
  const visibleHints = effectiveTab === "evaluation" ? evalHints : activityHints;
  const activeTabLabel = tabs.find((t) => t.id === effectiveTab)?.label ?? "";

  // Clear All only dismisses the hints in the currently active tab so the
  // other group's suggestions are preserved.
  const handleClearAll = async () => {
    if (visibleHints.length === 0) return;
    setClearingAll(true);
    try {
      await Promise.all(visibleHints.map((h) => dismiss(h.id)));
    } finally {
      setClearingAll(false);
    }
  };

  const renderHintCard = (hint: AmbientHint) => (
    <HintCard
      key={hint.id}
      hint={hint}
      highlighted={highlightedId === hint.id}
      allowAutoRun={allowAutoRun}
      onOpenInChat={handleOpenInChat}
      onApproveRun={handleApproveRun}
      onApply={handleApply}
      onDismiss={dismiss}
      onSnooze={snooze}
    />
  );

  return (
    <div className="h-full overflow-y-auto">
      <div className="px-6 py-6">
        {/* Page header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-blue-500/10 border border-blue-500/20">
              <Sparkles size={22} className="text-blue-400" />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-th-text-primary">Suggestions</h1>
              <p className="text-xs text-th-text-muted mt-0.5">
                {sweeping
                  ? "Running sweep…"
                  : pendingCount > 0
                    ? `${pendingCount} suggestion${pendingCount !== 1 ? "s" : ""} waiting`
                    : "No pending suggestions"}
                {!sweeping && quietHours && " · quiet hours active"}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {visibleHints.length > 0 && (
              <button
                onClick={handleClearAll}
                disabled={clearingAll || sweeping}
                className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg text-sm font-medium border transition-all bg-th-surface-hover border-th-border text-th-text-muted hover:text-red-500 hover:border-red-500/30 hover:bg-red-500/5 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {clearingAll
                  ? <Loader2 size={14} className="animate-spin" />
                  : <Trash2 size={14} />}
                {tabs.length > 1 ? "Clear tab" : "Clear all"}
              </button>
            )}
            <button
              onClick={handleManualSweep}
              disabled={sweeping}
              className={`inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg text-sm font-medium border transition-all ${
                sweeping
                  ? "bg-blue-500/10 border-blue-500/25 text-blue-300 cursor-wait"
                  : "bg-th-surface-hover border-th-border text-th-text-secondary hover:text-th-text-primary hover:bg-th-card-bg"
              }`}
            >
              {sweeping
                ? <Loader2 size={14} className="animate-spin" />
                : <RefreshCw size={14} />}
              {sweeping ? "Sweeping…" : "Suggest now"}
            </button>
          </div>
        </div>

        {/* Progress banner — visible while sweep is running */}
        {sweeping && <SweepProgressBanner />}

        {runError && (
          <div className="mb-4 p-3 rounded-lg bg-red-500/10 border border-red-500/25 text-xs text-red-400">
            {runError}
          </div>
        )}

        {skipReason && !sweeping && (() => {
          const msg = SKIP_MESSAGES[skipReason] ?? {
            title: "Sweep returned no suggestions",
            detail: `Reason: ${skipReason}`,
          };
          return (
            <div className="mb-4 p-4 rounded-xl bg-th-card-bg border border-th-card-border">
              <p className="text-sm font-medium text-th-text-secondary">{msg.title}</p>
              <p className="text-xs text-th-text-muted mt-1">{msg.detail}</p>
            </div>
          );
        })()}

        {/* Tab switcher + hints. Tabs persist even when a group has no
            suggestions; the panel below shows a per-tab empty state instead.
            Content is dimmed while sweeping so the progress banner has focus. */}
        <div className={`transition-opacity duration-300 ${sweeping ? "opacity-40 pointer-events-none" : ""}`}>
          <div className="flex items-center gap-2 mb-5">
            {tabs.map((t) => {
              const Icon = t.icon;
              const active = t.id === effectiveTab;
              return (
                <button
                  key={t.id}
                  onClick={() => setActiveTab(t.id)}
                  className={`inline-flex items-center gap-2 px-4 py-2 rounded-full text-base font-semibold border transition-all ${
                    active
                      ? "bg-th-card-bg border-th-border text-th-text-primary"
                      : "bg-transparent border-transparent text-th-text-muted hover:text-th-text-secondary hover:bg-th-surface-hover"
                  }`}
                >
                  <Icon size={18} className={active ? t.accent : ""} aria-hidden />
                  {t.label}
                  <span className="text-xs text-th-text-muted/60">{t.count}</span>
                </button>
              );
            })}
          </div>

          {visibleHints.length > 0 ? (
            <div className="space-y-2.5">{visibleHints.map(renderHintCard)}</div>
          ) : (
            <div className="flex flex-col items-center justify-center py-16 text-center">
              <div className="p-4 rounded-2xl bg-blue-500/10 border border-blue-500/20 mb-4">
                <Sparkles size={28} className="text-blue-400" />
              </div>
              <p className="text-sm font-medium text-th-text-secondary">
                No {activeTabLabel.toLowerCase()} suggestions
              </p>
              <p className="text-xs text-th-text-muted mt-1 max-w-xs">
                The ambient agent will surface ideas based on your memory, sessions, and activity.
                Enable it in Settings → Agent Memory → Ambient.
              </p>
              <button
                onClick={handleManualSweep}
                disabled={sweeping}
                className="mt-4 inline-flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium bg-blue-500/10 text-blue-300 border border-blue-500/20 hover:bg-blue-500/20 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <RefreshCw size={14} />
                Run a sweep now
              </button>
            </div>
          )}
        </div>

        {/* Recently actioned (ghost trail) — collapsed by default */}
        {recentActioned.length > 0 && (
          <div className="mt-8">
            <button
              type="button"
              onClick={() => setShowRecent((v) => !v)}
              className="flex items-center gap-1.5 px-1 mb-3 text-xs uppercase tracking-wider text-th-text-muted font-semibold hover:text-th-text-secondary transition-colors"
              aria-expanded={showRecent}
            >
              {showRecent ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
              Recently actioned
              <span className="text-[10px] text-th-text-muted/60 normal-case font-medium">({recentActioned.length})</span>
            </button>
            {showRecent && (
              <div className="space-y-2">
                {recentActioned.map((hint) => (
                  <div
                    key={hint.id}
                    className="flex items-center gap-3 px-4 py-2.5 bg-th-card-bg border border-th-card-border rounded-xl opacity-40"
                  >
                    <CheckCircle size={14} className="text-emerald-400 shrink-0" />
                    <p className="text-xs text-th-text-secondary truncate">{hint.title}</p>
                    <span className="ml-auto text-[10px] text-th-text-muted shrink-0 capitalize">
                      {hint.status}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
