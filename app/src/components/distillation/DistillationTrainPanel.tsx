import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Ban,
  CheckCircle,
  Clock,
  Loader2,
  Play,
  RefreshCw,
  Square,
  XCircle,
} from "lucide-react";
import { api } from "../../hooks/useApi";
import { usePolling } from "../../hooks/usePolling";
import type { DistillCensus, DistillJobPhase, DistillJobState, DistillJobStatus } from "../../types";

const IDLE_STATUS: DistillJobStatus = {
  state: "idle",
  phase: "",
  started_at: null,
  finished_at: null,
  error: null,
  log_lines: [],
  n_trajectories: 0,
  n_sft: 0,
  adapter_path: null,
  student_id: "",
  teacher_id: "",
  iters: 200,
  iter_current: null,
  warnings: [],
  blocking_sessions: [],
  catalog_id: "",
  display_name: "",
  purpose: "",
  fused_path: "",
  omlx_model_id: "",
};

const WARN_MIN_TRACES = 20;

const PHASE_LABEL: Record<Exclude<DistillJobPhase, "">, string> = {
  collecting: "Collecting trajectories",
  preparing: "Preparing SFT dataset",
  unloading: "Unloading chat model",
  training: "Training LoRA",
  fusing: "Fusing for Turbo",
};

const STATE_BADGE: Record<
  DistillJobState,
  { label: string; color: string; Icon: typeof Clock; spin?: boolean }
> = {
  idle: { label: "Idle", color: "text-th-text-muted bg-neutral-500/10 border-neutral-500/20", Icon: Clock },
  queued: { label: "Queued", color: "text-amber-400 bg-amber-500/10 border-amber-500/20", Icon: Clock },
  running: { label: "Running", color: "text-violet-400 bg-violet-500/10 border-violet-500/20", Icon: Loader2, spin: true },
  success: { label: "Adapter ready", color: "text-emerald-400 bg-emerald-500/10 border-emerald-500/20", Icon: CheckCircle },
  error: { label: "Failed", color: "text-red-400 bg-red-500/10 border-red-500/20", Icon: XCircle },
  cancelled: { label: "Cancelled", color: "text-amber-400 bg-amber-500/10 border-amber-500/20", Icon: Ban },
};

function StatusBadge({ state }: { state: DistillJobState }) {
  const { label, color, Icon, spin } = STATE_BADGE[state];
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${color}`}>
      <Icon size={12} className={spin ? "animate-spin" : ""} />
      {label}
    </span>
  );
}

export function DistillationTrainPanel({
  teacherModelId,
  studentModelId,
  filterByTeacher,
  currentAdapterPath,
  currentModelId,
  onUseAdapter,
  source = "sessions",
  upload = null,
}: {
  teacherModelId: string;
  studentModelId: string;
  filterByTeacher: boolean;
  currentAdapterPath: string;
  currentModelId?: string;
  onUseAdapter: (path: string, catalogId?: string) => void;
  source?: "sessions" | "upload";
  upload?: { id: string; filename: string; n_valid: number } | null;
}) {
  const [status, setStatus] = useState<DistillJobStatus>(IDLE_STATUS);
  const [census, setCensus] = useState<DistillCensus | null>(null);
  const [censusError, setCensusError] = useState<string | null>(null);
  const [triggerError, setTriggerError] = useState<string | null>(null);
  const [iters, setIters] = useState(200);
  const [purpose, setPurpose] = useState("");
  const [starting, setStarting] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const logRef = useRef<HTMLPreElement>(null);

  const loadCensus = useCallback(async () => {
    try {
      const report = await api.getDistillCensus({
        teacher_model_id: teacherModelId,
        student_model_id: studentModelId,
        filter_sessions_by_teacher: filterByTeacher,
      });
      setCensus(report);
      setCensusError(null);
    } catch (e: unknown) {
      setCensusError(e instanceof Error ? e.message : "Census failed");
    }
  }, [teacherModelId, studentModelId, filterByTeacher]);

  useEffect(() => {
    void loadCensus();
  }, [loadCensus]);

  const running = status.state === "running" || status.state === "queued";

  usePolling(
    async () => {
      try {
        setStatus(await api.getDistillStatus());
      } catch {
        /* keep last */
      }
    },
    running ? 1200 : 4000,
    true,
  );

  useEffect(() => {
    if (status.state !== "running") setCancelling(false);
  }, [status.state]);

  useEffect(() => {
    if (status.state === "success" || status.state === "error" || status.state === "cancelled") {
      void loadCensus();
    }
  }, [status.state, loadCensus]);

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [status.log_lines]);

  const blocking = status.blocking_sessions.length
    ? status.blocking_sessions
    : (census?.blocking_sessions ?? []);
  const nTraj = source === "upload"
    ? (upload?.n_valid ?? 0)
    : (census?.n_trajectories ?? 0);
  const canStart = !running && !starting && nTraj > 0 && blocking.length === 0 && (
    source !== "upload" || !!upload?.id
  );
  const adapterInUse =
    (!!status.adapter_path && currentAdapterPath === status.adapter_path)
    || (!!status.catalog_id && currentModelId === status.catalog_id);
  const progress =
    running && status.iters > 0 && status.iter_current != null
      ? Math.min(100, Math.round((status.iter_current / status.iters) * 100))
      : running && status.phase === "training"
        ? 0
        : null;

  const handleStart = async () => {
    setTriggerError(null);
    setStarting(true);
    try {
      const next = await api.startDistillTrain({
        iters,
        purpose: purpose.trim(),
        teacher_model_id: teacherModelId,
        student_model_id: studentModelId,
        source,
        ...(source === "upload" && upload?.id
          ? { upload_id: upload.id, name: "upload" }
          : {}),
      });
      setStatus(next);
    } catch (e: unknown) {
      setTriggerError(e instanceof Error ? e.message : "Failed to start training");
    } finally {
      setStarting(false);
    }
  };

  const handleCancel = async () => {
    setCancelling(true);
    try {
      await api.cancelDistillTrain();
    } catch {
      /* poll will catch the state */
    }
    setTimeout(() => setCancelling(false), 8000);
  };

  const phaseLabel = status.phase ? PHASE_LABEL[status.phase] : null;

  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-sm font-medium text-th-text-primary">Train LoRA</p>
          <p className="text-[12px] text-th-text-tertiary mt-0.5 leading-relaxed">
            {source === "upload"
              ? "Trains on your uploaded JSONL. Unloads the chat model, then runs "
              : "Collects successful sessions, unloads the chat model, then runs "}
            <code className="font-mono">mlx_lm.lora</code> on the student. Stop any
            running chat first so Metal memory is free.
          </p>
        </div>
        <StatusBadge state={status.state} />
      </div>

      <div className="grid grid-cols-3 gap-2">
        <div className="rounded-lg bg-th-inset-bg border border-th-border px-3 py-2">
          <p className="text-[10px] uppercase tracking-wider text-th-text-muted">
            {source === "upload" ? "Valid rows" : "Trajectories"}
          </p>
          <p className="text-sm font-semibold text-th-text-primary mt-0.5 tabular-nums">
            {census ? nTraj : "—"}
          </p>
        </div>
        <div className="rounded-lg bg-th-inset-bg border border-th-border px-3 py-2">
          <p className="text-[10px] uppercase tracking-wider text-th-text-muted">Iters</p>
          <input
            type="number"
            min={1}
            max={50000}
            value={iters}
            disabled={running}
            onChange={(e) => setIters(Math.min(50000, Math.max(1, parseInt(e.target.value, 10) || 1)))}
            className="w-full mt-0.5 bg-transparent text-sm font-semibold text-th-text-primary tabular-nums focus:outline-none disabled:opacity-60"
          />
        </div>
        <div className="rounded-lg bg-th-inset-bg border border-th-border px-3 py-2">
          <p className="text-[10px] uppercase tracking-wider text-th-text-muted">Progress</p>
          <p className="text-sm font-semibold text-th-text-primary mt-0.5 tabular-nums">
            {running && status.iter_current != null
              ? `${status.iter_current}/${status.iters}`
              : phaseLabel || (status.state === "success" ? "Done" : "—")}
          </p>
        </div>
      </div>

      <div>
        <label className="block text-[11px] font-medium text-th-text-tertiary mb-1.5">
          Purpose
        </label>
        <textarea
          value={purpose}
          disabled={running}
          onChange={(e) => setPurpose(e.target.value)}
          rows={2}
          placeholder="e.g. Mail triage and calendar scheduling from my Otto sessions"
          className="w-full px-3 py-2 rounded-lg bg-th-input-bg border border-th-input-border text-sm text-th-text-primary placeholder:text-th-text-muted focus:outline-none focus:border-violet-400 focus:ring-1 focus:ring-violet-300/30 resize-y disabled:opacity-60"
        />
        <p className="text-[11px] text-th-text-muted mt-1 leading-relaxed">
          Written onto the catalog card so you and the agent know when to load this adapter.
        </p>
      </div>

      {progress != null && (
        <div className="h-1.5 rounded-full bg-th-inset-bg overflow-hidden">
          <div
            className="h-full bg-violet-400 transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
      )}

      {nTraj === 0 && !running && source === "upload" && (
        <p className="text-xs text-th-text-tertiary">
          No uploaded file selected. Add a valid JSONL on the Dataset → Files tab,
          then choose Use for train.
        </p>
      )}
      {nTraj === 0 && !running && source === "sessions" && census && (
        <p className="text-xs text-th-text-tertiary">
          No high-quality trajectories yet. Complete some teacher sessions with
          tool use, then refresh. Start stays disabled until at least one trace
          passes the quality filter.
        </p>
      )}
      {source === "upload" && upload && nTraj > 0 && !running && (
        <p className="text-xs text-th-text-secondary">
          Training on <span className="font-medium text-th-text-primary">{upload.filename}</span>
          {" "}({upload.n_valid.toLocaleString()} valid rows).
        </p>
      )}

      {nTraj > 0 && nTraj < WARN_MIN_TRACES && !running && (
        <p className="text-xs text-amber-400 flex items-start gap-1.5">
          <AlertTriangle size={12} className="shrink-0 mt-0.5" />
          Only {nTraj} high-quality traces. Training will run, but {WARN_MIN_TRACES}+
          is the usual floor before the adapter is worth loading.
        </p>
      )}

      {blocking.length > 0 && (
        <p className="text-xs text-amber-400 flex items-start gap-1.5">
          <AlertTriangle size={12} className="shrink-0 mt-0.5" />
          {blocking.length === 1
            ? `“${blocking[0].title || blocking[0].id}” is still running. Stop it before training.`
            : `${blocking.length} chat sessions are running. Stop them before training.`}
        </p>
      )}

      {censusError && (
        <p className="text-xs text-red-400">{censusError}</p>
      )}
      {triggerError && (
        <p className="text-xs text-red-400">{triggerError}</p>
      )}
      {status.state === "error" && status.error && (
        <p className="text-xs text-red-400 flex items-start gap-1.5">
          <XCircle size={12} className="shrink-0 mt-0.5" />
          {status.error}
        </p>
      )}

      <div className="flex items-center gap-2 flex-wrap">
        {running ? (
          <button
            type="button"
            onClick={() => void handleCancel()}
            disabled={cancelling}
            className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-red-500/10 text-red-400 border border-red-500/20 hover:bg-red-500/20 transition-all disabled:opacity-60 disabled:cursor-wait"
          >
            {cancelling
              ? <><Loader2 size={14} className="animate-spin" /> Cancelling…</>
              : <><Square size={14} /> Cancel</>}
          </button>
        ) : (
          <button
            type="button"
            onClick={() => void handleStart()}
            disabled={!canStart}
            className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-violet-500/10 text-violet-400 border border-violet-500/20 hover:bg-violet-500/20 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {starting
              ? <><Loader2 size={14} className="animate-spin" /> Starting…</>
              : <><Play size={14} /> Start training</>}
          </button>
        )}
        <button
          type="button"
          onClick={() => void loadCensus()}
          disabled={running}
          className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium text-th-text-tertiary hover:text-th-text-primary hover:bg-th-surface-hover transition-all disabled:opacity-40"
        >
          <RefreshCw size={12} /> Refresh dataset
        </button>
        {status.state === "success" && status.adapter_path && (
          <button
            type="button"
            onClick={() => onUseAdapter(status.adapter_path!, status.catalog_id || undefined)}
            disabled={adapterInUse}
            className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 hover:bg-emerald-500/20 transition-all disabled:opacity-50"
          >
            <CheckCircle size={14} />
            {adapterInUse ? "In use" : "Use this model"}
          </button>
        )}
      </div>

      {(running || status.log_lines.length > 0) && (
        <pre
          ref={logRef}
          className="max-h-44 overflow-auto rounded-lg bg-black/80 p-3 text-[10px] text-emerald-100/90 font-mono whitespace-pre-wrap leading-relaxed"
        >
          {status.log_lines.join("\n") || "Waiting for output…"}
        </pre>
      )}

      {status.state === "success" && status.adapter_path && (
        <p className="text-[11px] text-th-text-tertiary break-all">
          {status.display_name && (
            <span className="block font-medium text-th-text-primary mb-0.5">
              {status.display_name}
            </span>
          )}
          <span className="font-mono">{status.catalog_id || status.adapter_path}</span>
          <span className="block font-sans mt-1 text-th-text-muted">
            Listed in the MLX catalog. Loads the student plus this LoRA at the next session start.
          </span>
        </p>
      )}
    </div>
  );
}
