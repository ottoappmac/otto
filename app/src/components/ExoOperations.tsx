import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Cpu,
  Loader2,
  Network,
  Play,
  RefreshCw,
  Server,
  Square,
  X,
  XCircle,
} from "lucide-react";
import { api } from "../hooks/useApi";
import { usePolling } from "../hooks/usePolling";
import type {
  ExoInfo,
  ExoJob,
  ExoJobPhase,
  ExoNodeInfo,
  ExoStatus,
} from "../types";
import ExoModelChooser from "./exo/ExoModelChooser";

// Cluster status drives the live topology pills, so we want it fairly fresh
// — but 4s was hammering the backend. 8s is plenty given that node joins /
// leaves typically take longer to settle anyway. Job logs poll faster while
// jobs are actually running.
const STATUS_POLL_MS = 8000;
const JOB_POLL_MS = 2000;

// ---------------------------------------------------------------------------
// Topology node tracking
// ---------------------------------------------------------------------------

type NodeClusterStatus = "online" | "left" | "offline";

interface NodeRecord {
  node: ExoNodeInfo;
  lastSeen: number;
  clusterStatus: NodeClusterStatus;
}

// ---------------------------------------------------------------------------
// Toast / notification
// ---------------------------------------------------------------------------

type ToastKind = "progress" | "success" | "error";
interface Toast { id: number; kind: ToastKind; message: string; }

let _toastSeq = 0;
function mkToast(kind: ToastKind, message: string): Toast {
  return { id: ++_toastSeq, kind, message };
}

function classNames(...xs: (string | false | null | undefined)[]): string {
  return xs.filter(Boolean).join(" ");
}

function fmtRel(epoch: number | null): string {
  if (!epoch) return "—";
  const diff = (Date.now() / 1000 - epoch) | 0;
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

function ToastBanner({
  toasts,
  onDismiss,
}: {
  toasts: Toast[];
  onDismiss: (id: number) => void;
}) {
  if (toasts.length === 0) return null;
  return (
    <div className="flex flex-col gap-1.5">
      {toasts.map((t) => (
        <div
          key={t.id}
          className={classNames(
            "flex items-center gap-2 rounded-md border px-3 py-2 text-[12px] transition-all",
            t.kind === "progress" &&
              "border-blue-500/30 bg-blue-500/10 text-blue-400",
            t.kind === "success" &&
              "border-emerald-500/30 bg-emerald-500/10 text-emerald-400",
            t.kind === "error" &&
              "border-red-500/30 bg-red-500/10 text-red-400",
          )}
        >
          {t.kind === "progress" ? (
            <Loader2 size={13} className="animate-spin shrink-0" />
          ) : t.kind === "success" ? (
            <CheckCircle2 size={13} className="shrink-0" />
          ) : (
            <AlertTriangle size={13} className="shrink-0" />
          )}
          <span className="flex-1">{t.message}</span>
          {t.kind !== "progress" && (
            <button
              type="button"
              onClick={() => onDismiss(t.id)}
              className="opacity-60 hover:opacity-100 shrink-0"
            >
              <X size={12} />
            </button>
          )}
        </div>
      ))}
    </div>
  );
}

function StatusPill({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span
      className={classNames(
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-medium border",
        ok
          ? "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/30"
          : "bg-red-500/15 text-red-600 dark:text-red-400 border-red-500/30",
      )}
    >
      <span
        className={classNames(
          "h-1.5 w-1.5 rounded-full",
          ok ? "bg-emerald-500" : "bg-red-500",
        )}
      />
      {label}
    </span>
  );
}

function PrimaryButton({
  onClick,
  disabled,
  children,
  icon: Icon,
  spinning,
  variant = "primary",
}: {
  onClick: () => void;
  disabled?: boolean;
  children: React.ReactNode;
  icon?: React.ElementType;
  /** Explicitly animate the icon as a spinner. Inferred automatically when icon === Loader2. */
  spinning?: boolean;
  variant?: "primary" | "danger" | "secondary";
}) {
  const styles = {
    primary: "bg-blue-600 text-white hover:bg-blue-500 disabled:bg-blue-600/40",
    danger: "bg-red-600 text-white hover:bg-red-500 disabled:bg-red-600/40",
    secondary:
      "bg-th-surface-hover text-th-text-primary hover:bg-th-border-strong disabled:opacity-40",
  }[variant];
  const doSpin = spinning ?? Icon === Loader2;
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={classNames(
        "inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-[12px] font-medium transition-all duration-150 disabled:cursor-not-allowed",
        styles,
      )}
    >
      {Icon ? <Icon size={14} className={doSpin ? "animate-spin" : undefined} /> : null}
      {children}
    </button>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-th-border-soft bg-th-bg px-3 py-2">
      <div className="text-[10px] uppercase tracking-widest text-th-text-muted">
        {label}
      </div>
      <div
        className="mt-0.5 text-[12px] text-th-text-primary truncate"
        title={value}
      >
        {value}
      </div>
    </div>
  );
}

function JobStatusIcon({ status }: { status: ExoJob["status"] }) {
  if (status === "running" || status === "pending")
    return <Loader2 size={14} className="animate-spin text-blue-500" />;
  if (status === "done")
    return <CheckCircle2 size={14} className="text-emerald-500" />;
  return <XCircle size={14} className="text-red-500" />;
}

interface ExoOperationsProps {
  /** Whether EXO is enabled in settings — disables action buttons when false. */
  enabled: boolean;
  /**
   * Which slice to render. Defaults to all panes for backward-compat, but
   * SettingsPage uses the sub-pill bar to render one pane at a time.
   *
   *   - ``overview`` — local cluster header + topology + active jobs + prereqs
   *   - ``models``   — model catalog with preload buttons
   */
  pane?: "overview" | "models";
}

/**
 * Operational view of the EXO cluster — local status, action buttons,
 * topology, recent jobs, and prereqs. Self-manages polling.
 *
 * Configuration (enable toggle, repo URL, ports, secondaries) lives in the
 * surrounding Settings → LLM → EXO card. Live ``exo.log`` tailing has
 * been removed in favour of the per-job log expanders below; check the
 * file at ``info.log_file`` for full historical logs.
 */
export default function ExoOperations({ enabled, pane }: ExoOperationsProps) {
  const [info, setInfo] = useState<ExoInfo | null>(null);
  const [status, setStatus] = useState<ExoStatus | null>(null);
  const [activeJobs, setActiveJobs] = useState<Record<string, ExoJob>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Models pane is fully owned by <ExoModelChooser/> — it handles the
  // catalog fetch, fit scoring against live cluster nodes, async
  // preload jobs (with progress), and multi-job UI re-attachment.
  const [toasts, setToasts] = useState<Toast[]>([]);
  // Track which op triggered an in-progress toast so we can replace it on
  // completion without affecting unrelated toasts.
  const progressToastRef = useRef<number | null>(null);
  // Track previous reachability so we can fire "started / stopped" toasts
  // when the status flips while we're not in the middle of a busy op.
  const prevReachableRef = useRef<boolean | null>(null);

  // Persistent topology — keeps the last-known node list so we can show
  // "offline" nodes even when the cluster is stopped, and detect join/leave.
  const [knownNodes, setKnownNodes] = useState<Record<string, NodeRecord>>({});
  const knownNodesRef = useRef<Record<string, NodeRecord>>({});
  // Suppress first-poll toasts so we don't spam "joined" on initial load.
  const topologyInitializedRef = useRef(false);

  const showOverview = pane === undefined || pane === "overview";
  const showModels = pane === undefined || pane === "models";

  // (refreshModels was deleted along with the legacy Models pane.)

  // Poll the cheap, frequently-changing status. exoInfo (config / prereqs /
  // git ref) is essentially static so we only fetch it on mount and after
  // user-triggered ops — see refreshFull below.
  const refresh = useCallback(async () => {
    try {
      const st = await api.exoStatus();
      setStatus(st);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load cluster state");
    }
  }, []);

  const refreshFull = useCallback(async () => {
    try {
      const [st, ri] = await Promise.all([api.exoStatus(), api.exoInfo()]);
      setStatus(st);
      setInfo(ri);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load cluster state");
    }
  }, []);

  // ── Toast helpers ────────────────────────────────────────────────────────

  const dismissToast = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
    if (progressToastRef.current === id) progressToastRef.current = null;
  }, []);

  const pushToast = useCallback(
    (kind: ToastKind, message: string, autoDismissMs?: number): number => {
      const t = mkToast(kind, message);
      setToasts((prev) => [...prev, t]);
      if (autoDismissMs) {
        setTimeout(() => dismissToast(t.id), autoDismissMs);
      }
      return t.id;
    },
    [dismissToast],
  );

  const replaceProgressToast = useCallback(
    (kind: "success" | "error", message: string) => {
      const prev = progressToastRef.current;
      if (prev !== null) {
        setToasts((ts) => ts.filter((t) => t.id !== prev));
        progressToastRef.current = null;
      }
      const id = pushToast(kind, message, kind === "success" ? 4000 : undefined);
      return id;
    },
    [pushToast],
  );

  // Watch reachability transitions to notify about auto-start events that
  // happen outside of a user-driven op (e.g. auto_start on boot).
  useEffect(() => {
    const curr = status?.reachable ?? false;
    const prev = prevReachableRef.current;
    if (prev !== null && prev !== curr && progressToastRef.current === null) {
      if (curr) {
        pushToast("success", "Cluster is up and reachable", 4000);
      } else {
        pushToast("success", "Cluster stopped", 4000);
      }
    }
    prevReachableRef.current = curr;
  }, [status?.reachable, pushToast]);

  // Track topology transitions — detect nodes joining/leaving the cluster
  // and keep a persistent "last known" snapshot for offline display.
  useEffect(() => {
    if (!status) return;
    const now = Date.now();
    const prev = knownNodesRef.current;
    const next: Record<string, NodeRecord> = { ...prev };
    const initialized = topologyInitializedRef.current;

    if (!status.reachable) {
      // Cluster went offline — mark every known node as "offline".
      let changed = false;
      Object.keys(next).forEach((id) => {
        if (next[id].clusterStatus !== "offline") {
          next[id] = { ...next[id], clusterStatus: "offline" };
          changed = true;
        }
      });
      if (changed) {
        knownNodesRef.current = next;
        setKnownNodes(next);
      }
      return;
    }

    // Cluster is reachable — reconcile live node list.
    const liveIds = new Set(status.nodes.map((n) => n.node_id));

    // Nodes that were "online" but are no longer reported → "left".
    Object.keys(next).forEach((id) => {
      if (!liveIds.has(id) && next[id].clusterStatus === "online") {
        next[id] = { ...next[id], clusterStatus: "left" };
        if (initialized) {
          const name = next[id].node.friendly_name ?? id.slice(0, 12);
          pushToast("error", `Node left cluster: ${name}`, 6000);
        }
      }
    });

    // Add / refresh live nodes.
    status.nodes.forEach((n) => {
      const prevRecord = prev[n.node_id];
      const wasOffline =
        !prevRecord || prevRecord.clusterStatus !== "online";
      next[n.node_id] = { node: n, lastSeen: now, clusterStatus: "online" };
      if (initialized && wasOffline && prevRecord) {
        const name = n.friendly_name ?? n.node_id.slice(0, 12);
        pushToast("success", `Node (re)joined cluster: ${name}`, 5000);
      }
    });

    topologyInitializedRef.current = true;
    knownNodesRef.current = next;
    setKnownNodes({ ...next });
  }, [status, pushToast]);

  // One-time hydrate of the static info payload; the periodic status poll
  // below is visibility-aware so the backend goes idle when the window is
  // hidden.
  useEffect(() => {
    void refreshFull();
  }, [refreshFull]);

  // Hydrate any in-flight jobs (e.g. auto-start provision kicked off at boot)
  // so the progress wheel shows immediately when the panel is opened.
  useEffect(() => {
    api.listExoJobs()
      .then(({ jobs }) => {
        const active = jobs.filter(
          (j) => j.status === "running" || j.status === "pending",
        );
        if (active.length === 0) return;
        setActiveJobs((prev) => {
          const next = { ...prev };
          active.forEach((j) => { next[j.id] = j; });
          return next;
        });
      })
      .catch(() => { /* non-critical */ });
  }, []);

  usePolling(refresh, STATUS_POLL_MS);

  // Track previous job statuses so we can fire toasts on completion.
  const prevJobStatusRef = useRef<Record<string, ExoJob["status"]>>({});

  useEffect(() => {
    const ids = Object.keys(activeJobs).filter(
      (id) =>
        activeJobs[id].status === "running" ||
        activeJobs[id].status === "pending",
    );
    if (ids.length === 0) return;
    const t = setInterval(async () => {
      const updates = await Promise.all(
        ids.map((id) => api.getExoJob(id).catch(() => null)),
      );
      setActiveJobs((prev) => {
        const next = { ...prev };
        updates.forEach((j) => {
          if (!j) return;
          const prevStatus = prevJobStatusRef.current[j.id];
          const wasRunning = prevStatus === "running" || prevStatus === "pending";
          const nowDone = j.status === "done" || j.status === "error";
          if (wasRunning && nowDone) {
            if (j.status === "done") {
              const label = j.kind === "provision" ? "Provision" : j.kind === "up" ? "Start" : j.kind;
              pushToast("success", `${label} complete (${j.target})`, 5000);
            } else {
              pushToast("error", `${j.kind} failed (${j.target}): ${j.error || "unknown error"}`);
            }
          }
          prevJobStatusRef.current[j.id] = j.status;
          next[j.id] = j;
        });
        return next;
      });
    }, JOB_POLL_MS);
    return () => clearInterval(t);
  }, [activeJobs, pushToast]);

  const trackJob = useCallback((job: ExoJob) => {
    setActiveJobs((prev) => ({ ...prev, [job.id]: job }));
  }, []);

  const runOp = useCallback(
    async (key: string, fn: () => Promise<void>) => {
      setError(null);
      setBusy(key);
      try {
        await fn();
        // Ops can flip prereqs / git_commit state so we re-pull the full
        // info payload here rather than relying on the periodic status poll.
        await refreshFull();
      } catch (e) {
        setError(e instanceof Error ? e.message : `${key} failed`);
      } finally {
        setBusy(null);
      }
    },
    [refreshFull],
  );

  const handleUp = (force = false) => {
    void runOp(force ? "up-force" : "up", async () => {
      try {
        const job = await api.exoUp(force);
        trackJob(job);
        // The job card below the topology section shows live phase progress —
        // no blocking toast needed.
      } catch (e) {
        pushToast("error", `Failed to start cluster: ${e instanceof Error ? e.message : String(e)}`);
        throw e;
      }
    });
  };

  const handleDown = () => {
    const id = pushToast("progress", "Stopping cluster…");
    progressToastRef.current = id;
    void runOp("down", async () => {
      try {
        await api.exoDown();
        replaceProgressToast("success", "Cluster stopped");
      } catch (e) {
        replaceProgressToast("error", `Failed to stop cluster: ${e instanceof Error ? e.message : String(e)}`);
        throw e;
      }
    });
  };

  const handleProvision = (force = false) => {
    const id = pushToast("progress", force ? "Force re-provisioning…" : "Provisioning cluster (installing deps, building…)");
    progressToastRef.current = id;
    void runOp("provision", async () => {
      try {
        const job = await api.exoProvision(force);
        trackJob(job);
        // Job is now tracked — swap the "starting…" toast for a neutral info
        // success that auto-clears, so the user knows the job is running.
        replaceProgressToast("success", "Provision job started — see Recent jobs below");
        progressToastRef.current = null;
      } catch (e) {
        replaceProgressToast("error", `Provision failed: ${e instanceof Error ? e.message : String(e)}`);
        throw e;
      }
    });
  };

  const handleSmoke = () =>
    runOp("smoke", async () => {
      const job = await api.exoSmoke();
      trackJob(job);
    });

  const reachable = status?.reachable ?? false;
  // Delivery mode drives which controls / diagnostics make sense. Prebuilt
  // (the default on Apple Silicon) downloads a notarized runtime — no build
  // toolchain, so we hide the source-only prereqs and provision controls.
  const prebuilt = info?.mode_effective !== "source";
  const runningJobs = useMemo(
    () =>
      Object.values(activeJobs)
        .sort((a, b) => b.started_at - a.started_at)
        .slice(0, 8),
    [activeJobs],
  );

  // True while an up/provision job is in flight AND the cluster isn't yet
  // reachable. Once reachable=true the daemon is confirmed up; any remaining
  // job work (e.g. model preload) is tracked separately and shouldn't keep
  // the UI locked in a "Starting…" state.
  const startupJob = useMemo(
    () =>
      Object.values(activeJobs).find(
        (j) =>
          (j.kind === "up" || j.kind === "provision") &&
          (j.status === "running" || j.status === "pending"),
      ) ?? null,
    [activeJobs],
  );
  const isStartingUp = startupJob !== null && !reachable;

  return (
    <div className="space-y-4">
      {error && (
        <div className="flex items-center gap-2 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-[12px] text-red-600 dark:text-red-400">
          <AlertTriangle size={14} />
          <span className="flex-1">{error}</span>
          <button
            onClick={() => setError(null)}
            className="opacity-60 hover:opacity-100"
          >
            <X size={14} />
          </button>
        </div>
      )}

      <ToastBanner toasts={toasts} onDismiss={dismissToast} />

      {/* Startup progress banner — shown while an up/provision job is in
          flight and the cluster isn't reachable yet. Disappears automatically
          when the job completes or the cluster becomes reachable. */}
      {isStartingUp && !reachable && (() => {
        const activePhase = startupJob?.phases?.find(
          (ph) => ph.status === "running",
        ) ?? startupJob?.phases?.find((ph) => ph.status === "pending") ?? null;
        const phaseLabel = activePhase
          ? activePhase.name === "provision"
            ? "Installing…"
            : activePhase.name === "start"
              ? "Starting daemon…"
              : activePhase.name === "verify"
                ? "Verifying…"
                : activePhase.name
          : startupJob?.kind === "provision"
            ? "Installing…"
            : "Starting up…";
        return (
          <div className="flex items-center gap-3 rounded-lg border border-blue-500/30 bg-blue-500/8 px-4 py-3 text-[13px]">
            <Loader2 size={16} className="animate-spin text-blue-400 shrink-0" />
            <div className="flex-1 min-w-0">
              <span className="font-medium text-blue-300">{phaseLabel}</span>
              {startupJob?.phases && startupJob.phases.length > 0 && (
                <div className="flex items-center gap-1.5 mt-1">
                  {startupJob.phases.map((ph, idx) => (
                    <div key={ph.name} className="flex items-center gap-1">
                      {idx > 0 && (
                        <span className="text-blue-900/60 text-[10px]">›</span>
                      )}
                      <span
                        className={classNames(
                          "text-[11px] font-medium",
                          ph.status === "done" && "text-emerald-400",
                          ph.status === "running" && "text-blue-300",
                          ph.status === "error" && "text-red-400",
                          ph.status === "pending" && "text-th-text-muted",
                        )}
                      >
                        {ph.name}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        );
      })()}

      {showOverview && (
      <>
      {/* Local cluster status */}
      <section className="rounded-lg border border-th-border bg-th-surface">
        <header className="border-b border-th-border px-4 py-3.5 space-y-3">
          {/* Row 1 — identity + reachability pills. Sits above actions
              so on narrower viewports the title never gets pushed off
              screen by the six lifecycle buttons below. */}
          <div className="flex flex-wrap items-center gap-2.5">
            <Activity size={16} className="text-th-text-secondary" />
            <h2 className="text-[13px] font-semibold text-th-text-primary">
              Local cluster (master)
            </h2>
            <StatusPill
              ok={reachable}
              label={reachable ? "reachable" : "offline"}
            />
            {info?.running && <StatusPill ok={true} label="running" />}
          </div>
          {/* Row 2 — actions grouped: lifecycle (Up/Down) up front,
              diagnostics (Smoke/Provision) at the end, with a vertical
              divider in between for visual grouping. */}
          <div className="flex flex-wrap items-center gap-2">
            <PrimaryButton
              variant="secondary"
              icon={RefreshCw}
              onClick={refreshFull}
            >
              Refresh
            </PrimaryButton>
            <PrimaryButton
              onClick={() => handleUp()}
              disabled={!enabled || busy === "up" || isStartingUp || reachable}
              icon={busy === "up" || isStartingUp ? Loader2 : Play}
            >
              {busy === "up" || isStartingUp ? "Starting…" : "Up"}
            </PrimaryButton>
            <PrimaryButton
              variant="danger"
              onClick={handleDown}
              disabled={!enabled || busy === "down" || !reachable}
              icon={busy === "down" ? Loader2 : Square}
            >
              {busy === "down" ? "Stopping…" : "Down"}
            </PrimaryButton>
            <span className="hidden sm:block h-5 w-px bg-th-border mx-1" />
            <PrimaryButton
              variant="secondary"
              onClick={handleSmoke}
              disabled={!enabled || busy === "smoke" || !reachable}
              icon={CheckCircle2}
            >
              Smoke
            </PrimaryButton>
            {/* Prebuilt mode: "Up" already downloads + starts, so a separate
                Provision button just confuses. Offer a single maintenance
                action to force a fresh runtime download. Source mode keeps the
                explicit provision / force re-provision pair since users there
                often want to build without starting. */}
            {prebuilt ? (
              <PrimaryButton
                variant="secondary"
                onClick={() => handleProvision(true)}
                disabled={!enabled || busy === "provision" || isStartingUp}
                icon={busy === "provision" ? Loader2 : RefreshCw}
              >
                Reinstall runtime
              </PrimaryButton>
            ) : (
              <>
                <PrimaryButton
                  variant="secondary"
                  onClick={() => handleProvision(false)}
                  disabled={!enabled || busy === "provision"}
                  icon={busy === "provision" ? Loader2 : Server}
                >
                  Provision
                </PrimaryButton>
                <PrimaryButton
                  variant="secondary"
                  onClick={() => handleProvision(true)}
                  disabled={!enabled || busy === "provision"}
                >
                  Force re-provision
                </PrimaryButton>
              </>
            )}
          </div>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 p-4">
          <Stat
            label="API base URL"
            value={status?.base_url || info?.config?.base_url || "—"}
          />
          <Stat
            label="Master node"
            value={status?.master_node_id?.slice(0, 18) || "—"}
          />
          <Stat
            label="Peers in cluster"
            value={String(status?.peer_count ?? 0)}
          />
          <Stat
            label="Loaded models"
            value={(status?.loaded_models ?? []).join(", ") || "(none)"}
          />
          <Stat
            label="RDMA edges"
            value={String(status?.rdma_connections ?? 0)}
          />
          <Stat
            label="Runtime"
            value={
              prebuilt
                ? info?.prebuilt?.exo_ref
                  ? `${info.prebuilt.exo_ref} (prebuilt)`
                  : info?.installed
                    ? "prebuilt (installed)"
                    : "(not installed)"
                : info?.state?.git_commit
                  ? `${info.state.exo_ref} @ ${info.state.git_commit.slice(0, 8)}`
                  : "(not provisioned)"
            }
          />
        </div>

        {info?.state?.mlx_version_warning ? (
          <div className="border-t border-th-border px-4 py-3 bg-amber-500/5 text-[12px] text-amber-600 leading-relaxed">
            <div className="flex items-start gap-2">
              <span className="font-semibold uppercase text-[10px] tracking-wider mt-0.5">
                MLX preflight
              </span>
              <span className="flex-1">
                {info.state.mlx_version_warning}
                {info.state.mlx_version_pinned && info.state.mlx_version_bundled && (
                  <span className="ml-2 font-mono text-[11px] text-th-text-muted">
                    (pinned {info.state.mlx_version_pinned} · bundled {info.state.mlx_version_bundled})
                  </span>
                )}
              </span>
            </div>
          </div>
        ) : null}

        <div className="border-t border-th-border px-4 py-3.5">
          <div className="flex items-center justify-between mb-2.5">
            <h3 className="text-[11px] uppercase tracking-widest text-th-text-muted">
              Topology
            </h3>
            <span className="text-[11px] text-th-text-muted">
              {(!reachable || busy === "down")
                ? `${Object.keys(knownNodes).length} offline`
                : `${Object.values(knownNodes).filter((r) => r.clusterStatus === "online").length} online${
                    Object.values(knownNodes).filter((r) => r.clusterStatus !== "online").length > 0
                      ? ` · ${Object.values(knownNodes).filter((r) => r.clusterStatus !== "online").length} offline`
                      : ""
                  }`
              }
            </span>
          </div>

          {Object.keys(knownNodes).length === 0 ? (
            <div className="flex items-center gap-2 rounded-md border border-th-border-soft bg-th-bg px-3 py-2.5 text-[12px] text-th-text-muted">
              <Server size={13} className="shrink-0" />
              {busy === "up"
                ? "Starting cluster — waiting for first peer report…"
                : reachable
                  ? "No peers detected yet — cluster is running solo."
                  : "Cluster offline — start it to populate topology."}
            </div>
          ) : (
            <div className="space-y-2">
              {Object.values(knownNodes)
                .sort((a, b) => {
                  // Master first, then online, then left, then offline.
                  const order: Record<NodeClusterStatus, number> = {
                    online: 0, left: 1, offline: 2,
                  };
                  const isMasterA = a.node.node_id === status?.master_node_id ? -1 : 0;
                  const isMasterB = b.node.node_id === status?.master_node_id ? -1 : 0;
                  return isMasterA - isMasterB || order[a.clusterStatus] - order[b.clusterStatus];
                })
                .map(({ node: n, clusterStatus: trackedStatus, lastSeen }) => {
                  const isMaster = n.node_id === status?.master_node_id;
                  // Derive the DISPLAYED status directly from live signals so
                  // the UI responds instantly without waiting for the tracking
                  // effect to propagate through the state cycle:
                  //  • cluster offline OR Down in progress → "offline"
                  //  • otherwise follow the tracking state
                  const clusterStatus: NodeClusterStatus =
                    (!reachable || busy === "down" || busy === "up-force")
                      ? "offline"
                      : trackedStatus;
                  const statusStyles: Record<NodeClusterStatus, { dot: string; label: string; text: string; border: string }> = {
                    online: {
                      dot: "bg-emerald-500",
                      label: "online",
                      text: "text-emerald-500",
                      border: "border-emerald-500/20",
                    },
                    left: {
                      dot: "bg-amber-400",
                      label: "left cluster",
                      text: "text-amber-400",
                      border: "border-amber-400/20",
                    },
                    offline: {
                      dot: "bg-neutral-400",
                      label: "offline",
                      text: "text-th-text-muted",
                      border: "border-th-border-soft",
                    },
                  };
                  const st = statusStyles[clusterStatus];
                  return (
                    <div
                      key={n.node_id}
                      className={classNames(
                        "flex items-center gap-3 rounded-md border bg-th-bg px-3 py-2 text-[12px]",
                        st.border,
                        clusterStatus !== "online" && "opacity-60",
                      )}
                    >
                      <Server
                        size={14}
                        className={classNames(
                          "shrink-0",
                          clusterStatus === "online"
                            ? "text-th-text-secondary"
                            : "text-th-text-muted",
                        )}
                      />
                      {/* Status dot */}
                      <span
                        className={classNames(
                          "h-1.5 w-1.5 rounded-full shrink-0 ring-1 ring-offset-1 ring-th-bg",
                          st.dot,
                          clusterStatus === "online" && "animate-pulse",
                        )}
                        title={st.label}
                      />
                      <div className="min-w-0 flex-1 flex items-center gap-2 flex-wrap">
                        <code className="text-th-text-primary truncate max-w-[160px]">
                          {n.node_id.slice(0, 16)}…
                        </code>
                        {n.friendly_name && (
                          <span className="text-th-text-secondary truncate">
                            {n.friendly_name}
                          </span>
                        )}
                        {isMaster && (
                          <span className="inline-flex items-center rounded-full bg-blue-500/15 text-blue-400 border border-blue-500/30 px-1.5 py-[1px] text-[10px] font-medium">
                            master
                          </span>
                        )}
                        <span className={classNames("text-[11px] font-medium", st.text)}>
                          {st.label}
                        </span>
                      </div>
                      <div className="flex items-center gap-3 ml-auto shrink-0">
                        {n.chip && (
                          <span className="text-th-text-muted">{n.chip}</span>
                        )}
                        {n.memory_total_gb ? (
                          <span className="text-th-text-tertiary">
                            {n.memory_free_gb ?? "?"} / {n.memory_total_gb} GB free
                          </span>
                        ) : null}
                        {clusterStatus !== "online" && lastSeen > 0 && (
                          <span className="text-th-text-muted">
                            last seen {fmtRel(lastSeen / 1000)}
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
            </div>
          )}
        </div>
      </section>
      </>
      )}

      {/* Models catalog — cluster-aware fit scoring + async preload jobs */}
      {showModels && (
      <section className="rounded-lg border border-th-border bg-th-surface">
        <header className="flex items-center justify-between gap-3 border-b border-th-border px-4 py-3">
          <div className="flex items-center gap-2">
            <Cpu size={16} className="text-th-text-secondary" />
            <h2 className="text-[13px] font-semibold text-th-text-primary">
              Models
            </h2>
            <span className="text-[11px] text-th-text-muted">
              fit-scored against your cluster
            </span>
          </div>
        </header>
        <div className="px-4 py-4">
          <ExoModelChooser enabled={enabled && reachable} />
        </div>
      </section>
      )}

      {/* Active jobs — visible regardless of pane so users see in-flight ops */}
      {runningJobs.length > 0 && (
        <section className="rounded-lg border border-th-border bg-th-surface">
          <header className="border-b border-th-border px-4 py-3 flex items-center gap-2">
            <Network size={16} className="text-th-text-secondary" />
            <h2 className="text-[13px] font-semibold text-th-text-primary">
              Recent jobs
            </h2>
          </header>
          <div className="divide-y divide-th-border">
            {runningJobs.map((j) => (
              <div key={j.id}>
                {/* Phase stepper — shown for "up" jobs that carry phases */}
                {j.kind === "up" && j.phases && j.phases.length > 0 ? (
                  <div className="px-4 py-3 space-y-2">
                    <div className="flex items-center gap-3 text-[12px]">
                      <JobStatusIcon status={j.status} />
                      <span className="font-medium text-th-text-primary capitalize">
                        {j.kind === "up" ? "Starting cluster" : j.kind}
                      </span>
                      <code className="text-th-text-tertiary">{j.target}</code>
                      <span className="text-th-text-muted ml-auto">
                        {fmtRel(j.started_at)}
                      </span>
                    </div>
                    {/* Phase pill row */}
                    <div className="flex flex-wrap items-center gap-2 pl-5">
                      {j.phases.map((ph: ExoJobPhase, idx: number) => (
                        <div key={ph.name} className="flex items-center gap-1.5">
                          {idx > 0 && (
                            <span className="text-th-text-muted text-[10px]">›</span>
                          )}
                          <div
                            className={classNames(
                              "flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium border",
                              ph.status === "done" &&
                                "bg-emerald-500/10 text-emerald-500 border-emerald-500/20",
                              ph.status === "running" &&
                                "bg-blue-500/10 text-blue-400 border-blue-500/20",
                              ph.status === "error" &&
                                "bg-red-500/10 text-red-500 border-red-500/20",
                              ph.status === "pending" &&
                                "bg-th-bg text-th-text-muted border-th-border",
                            )}
                          >
                            {ph.status === "running" && (
                              <Loader2 size={10} className="animate-spin" />
                            )}
                            {ph.status === "done" && (
                              <CheckCircle2 size={10} />
                            )}
                            {ph.status === "error" && (
                              <XCircle size={10} />
                            )}
                            <span className="capitalize">{ph.name}</span>
                            {ph.message && (
                              <span className="opacity-70 ml-0.5">
                                · {ph.message}
                              </span>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                    {/* Error banner */}
                    {j.error && (
                      <div className="flex items-start gap-2 rounded-md bg-red-500/10 border border-red-500/20 px-3 py-2 text-[11px] text-red-500 pl-5">
                        <AlertTriangle size={12} className="shrink-0 mt-0.5" />
                        <span>{j.error}</span>
                      </div>
                    )}
                    {/* Log expander */}
                    <details className="pl-5">
                      <summary className="cursor-pointer text-[11px] text-th-text-muted hover:text-th-text-secondary list-none">
                        {j.log_lines.length} log line{j.log_lines.length !== 1 ? "s" : ""}
                      </summary>
                      <pre className="mt-1.5 max-h-52 overflow-auto rounded bg-black/80 p-2.5 text-[11px] text-emerald-100 font-mono whitespace-pre-wrap">
                        {j.log_lines.join("\n") || "(no output yet)"}
                      </pre>
                    </details>
                  </div>
                ) : (
                  /* Generic expander for other job types */
                  <details className="group px-4 py-2">
                    <summary className="flex cursor-pointer list-none items-center gap-3 text-[12px]">
                      <JobStatusIcon status={j.status} />
                      <span className="font-medium text-th-text-primary">
                        {j.kind}
                      </span>
                      <code className="text-th-text-tertiary">{j.target}</code>
                      <span className="text-th-text-muted">
                        started {fmtRel(j.started_at)}
                      </span>
                      {j.error && (
                        <span
                          className="text-red-500 truncate max-w-[240px]"
                          title={j.error}
                        >
                          {j.error}
                        </span>
                      )}
                      <span className="ml-auto text-th-text-muted">
                        {j.status} · {j.log_lines.length} lines
                      </span>
                    </summary>
                    <pre className="mt-2 max-h-72 overflow-auto rounded bg-black/80 p-3 text-[11px] text-emerald-100 font-mono whitespace-pre-wrap">
                      {j.log_lines.join("\n") || "(no output yet)"}
                    </pre>
                  </details>
                )}
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Prereqs — only relevant for the source build path; the prebuilt
          runtime ships its own toolchain so these checks are noise there. */}
      {showOverview && info && !prebuilt && (
        <section className="rounded-lg border border-th-border bg-th-surface">
          <header className="border-b border-th-border px-4 py-3">
            <h2 className="text-[13px] font-semibold text-th-text-primary">
              Prereqs
            </h2>
          </header>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 p-4 text-[12px]">
            {(["brew", "uv", "node", "npm", "git", "rustup", "cargo"] as const).map(
              (k) => {
                const v = info.prereqs[k];
                return (
                  <div key={k} className="flex items-center gap-2">
                    {v ? (
                      <CheckCircle2 size={14} className="text-emerald-500" />
                    ) : (
                      <XCircle size={14} className="text-red-500" />
                    )}
                    <span className="font-medium text-th-text-primary">
                      {k}
                    </span>
                    <code className="text-th-text-muted truncate max-w-[120px]">
                      {v ?? "(missing)"}
                    </code>
                  </div>
                );
              },
            )}
            <div className="flex items-center gap-2">
              {info.prereqs.rust_nightly ? (
                <CheckCircle2 size={14} className="text-emerald-500" />
              ) : (
                <XCircle size={14} className="text-red-500" />
              )}
              <span className="font-medium text-th-text-primary">
                rust-nightly
              </span>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
