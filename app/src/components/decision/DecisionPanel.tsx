import { useCallback, useEffect, useState } from "react";
import { AlertTriangle, CheckCircle2, Download, Loader2, Sparkles, X } from "lucide-react";
import { api } from "../../hooks/useApi";
import type {
  AppSettings,
  DecisionConfig,
  DecisionModelRow,
  DecisionSnapshot as Snapshot,
  MlxDownloadJob,
} from "../../types";

const TYPESAFE_URL = "https://api.typesafe.ai";

function providerOf(url: string): "local" | "typesafe" {
  const trimmed = (url || "").trim();
  if (!trimmed) return "local";
  try {
    const host = new URL(trimmed).hostname.toLowerCase();
    if (host === "127.0.0.1" || host === "localhost" || host === "::1") return "local";
  } catch {
    return "local";
  }
  return "typesafe";
}

function bytesToHuman(n: number): string {
  if (!n || n < 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : 1)} ${units[i]}`;
}

function formatRate(bps: number): string {
  if (!bps || bps <= 0) return "—";
  return `${bytesToHuman(bps)}/s`;
}

function formatEta(seconds: number | null): string {
  if (seconds == null || seconds < 0) return "—";
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  if (m < 60) return `${m}m ${s}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

function FitBadge({ fits }: { fits: DecisionModelRow["fits"] }) {
  const map: Record<string, { label: string; cls: string; dot: string }> = {
    comfortable: {
      label: "Comfortable",
      cls: "bg-emerald-500/15 text-emerald-500 ring-1 ring-emerald-500/30",
      dot: "bg-emerald-500",
    },
    tight: {
      label: "Tight",
      cls: "bg-amber-500/15 text-amber-500 ring-1 ring-amber-500/30",
      dot: "bg-amber-500",
    },
    over: {
      label: "Won't fit",
      cls: "bg-rose-500/15 text-rose-500 ring-1 ring-rose-500/30",
      dot: "bg-rose-500",
    },
    unknown: {
      label: "Unknown",
      cls: "bg-th-surface text-th-text-muted ring-1 ring-th-border",
      dot: "bg-neutral-400",
    },
  };
  const v = map[fits] ?? map.unknown;
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] font-medium ${v.cls}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${v.dot}`} />
      {v.label}
    </span>
  );
}

function DownloadProgress({
  job,
  onCancel,
  onDismiss,
}: {
  job: MlxDownloadJob;
  onCancel: () => void;
  onDismiss: () => void;
}) {
  const pct = job.bytes_total > 0 ? Math.min(100, Math.round((job.bytes_done / job.bytes_total) * 100)) : 0;
  const indeterminate = job.bytes_total === 0 && (job.status === "running" || job.status === "pending" || job.status === "cancelling");
  const cancelling = job.status === "cancelling";
  const terminal = job.status === "done" || job.status === "error" || job.status === "cancelled";

  return (
    <div className="rounded-xl border border-th-border bg-th-surface p-3 space-y-2">
      <div className="flex items-center justify-between gap-2">
        <div className="min-w-0 flex-1">
          <p className="text-xs font-medium text-th-text-primary truncate">{job.repo_id}</p>
          <p className="text-[11px] text-th-text-muted truncate">
            {job.status === "done"
              ? "Download complete"
              : job.status === "error"
                ? job.message || "Download failed"
                : job.status === "cancelled"
                  ? "Cancelled"
                  : cancelling
                    ? "Cancelling… (finishing current file)"
                    : job.current_file
                      ? `${job.current_file}${job.files_total ? ` · ${job.files_done}/${job.files_total}` : ""}`
                      : "Starting…"}
          </p>
        </div>
        <button
          type="button"
          className="text-th-text-muted hover:text-rose-400 disabled:opacity-40"
          onClick={terminal ? onDismiss : onCancel}
          disabled={cancelling}
          title={terminal ? "Dismiss" : "Cancel download"}
        >
          <X size={14} />
        </button>
      </div>
      <div className="relative h-2 rounded-full bg-th-inset-bg overflow-hidden">
        {indeterminate ? (
          <div className="absolute inset-y-0 w-1/3 bg-th-tab-active-bg/60 animate-pulse" />
        ) : (
          <div
            className={`absolute inset-y-0 left-0 transition-all duration-300 ${
              job.status === "error"
                ? "bg-rose-500"
                : job.status === "cancelled"
                  ? "bg-th-text-muted"
                  : job.status === "done"
                    ? "bg-emerald-500"
                    : "bg-th-tab-active-bg"
            }`}
            style={{ width: `${pct}%` }}
          />
        )}
      </div>
      <div className="flex items-center justify-between text-[11px] text-th-text-muted font-mono">
        <span>
          {bytesToHuman(job.bytes_done)} / {job.bytes_total > 0 ? bytesToHuman(job.bytes_total) : "?"}
          {!indeterminate && job.bytes_total > 0 && <span className="ml-1">· {pct}%</span>}
        </span>
        <span>
          {job.status === "running" && (
            <>
              {formatRate(job.rate_bps)} · ETA {formatEta(job.eta_seconds)}
              {job.use_hf_transfer && <span className="ml-2 text-th-text-tertiary">hf_transfer</span>}
            </>
          )}
        </span>
      </div>
    </div>
  );
}

function ModelRow({
  row,
  isSelected,
  activeJob,
  onDownload,
  onUse,
}: {
  row: DecisionModelRow;
  isSelected: boolean;
  activeJob?: MlxDownloadJob;
  onDownload: () => void;
  onUse: () => void;
}) {
  const dlBlocked = row.fits === "over" || !row.disk_ok;
  const isCached = Boolean(row.already_cached && row.cache_complete);
  const isIncomplete = Boolean(row.already_cached && !row.cache_complete);
  const blockTitle = row.fits === "over"
    ? `Won't fit beside a chat model — about ${row.resident_gb.toFixed(0)} GB resident`
    : !row.disk_ok
      ? "Not enough free disk"
      : "Download to the local Hub cache";

  return (
    <div
      className={`relative rounded-xl border transition-all overflow-hidden ${
        isSelected
          ? "border-th-tab-active-bg/60 bg-th-tab-active-bg/5 ring-1 ring-th-tab-active-bg/20"
          : row.featured
            ? "border-amber-500/25 bg-th-surface hover:border-amber-500/40"
            : "border-th-border bg-th-surface hover:border-th-border-strong"
      }`}
    >
      {row.featured && !isSelected && (
        <div className="absolute left-0 inset-y-0 w-0.5 bg-gradient-to-b from-amber-400/80 via-amber-500/60 to-amber-400/30" />
      )}
      {isSelected && <div className="absolute left-0 inset-y-0 w-0.5 bg-th-tab-active-bg/80" />}
      <div className="pl-4 pr-3.5 py-3.5">
        <div className="flex items-start justify-between gap-2 mb-1.5">
          <div className="flex items-center gap-1.5 flex-wrap min-w-0">
            <span className="text-sm font-semibold text-th-text-primary leading-snug">{row.display_name}</span>
            {row.featured && (
              <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-full bg-amber-500/15 text-amber-500 text-[10px] font-semibold ring-1 ring-amber-500/25">
                <Sparkles size={9} />
                featured
              </span>
            )}
            {isCached && (
              <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-full bg-emerald-500/15 text-emerald-500 text-[10px] font-semibold ring-1 ring-emerald-500/25">
                <CheckCircle2 size={9} />
                cached
              </span>
            )}
            {isIncomplete && (
              <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-full bg-amber-500/15 text-amber-500 text-[10px] font-semibold ring-1 ring-amber-500/25">
                <AlertTriangle size={9} />
                incomplete
              </span>
            )}
          </div>
          <FitBadge fits={row.fits} />
        </div>
        {row.blurb && (
          <p className="text-[11px] text-th-text-tertiary line-clamp-1 leading-relaxed mb-2.5 pr-2">{row.blurb}</p>
        )}
        <div className="flex items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 text-[11px] text-th-text-muted min-w-0">
            <span className="font-mono shrink-0" title={`About ${row.resident_gb.toFixed(0)} GB resident while running`}>
              {row.weights_gb.toFixed(1)} GB · {row.quant}
            </span>
            <span className="px-1.5 py-0.5 rounded-md bg-th-inset-bg border border-th-border/50 text-th-text-secondary text-[10px] font-medium">
              {row.license}
            </span>
          </div>
          <div className="flex items-center gap-1.5 shrink-0">
            {isCached ? (
              <>
                <button
                  type="button"
                  className="px-3 py-1.5 text-[11px] font-semibold rounded-md bg-emerald-600 text-white hover:bg-emerald-500 transition-colors disabled:opacity-50"
                  onClick={onUse}
                  disabled={row.fits === "over"}
                  title={row.fits === "over" ? blockTitle : "Use this model for jev"}
                >
                  Use this model
                </button>
                <button
                  type="button"
                  className="p-1.5 text-th-text-muted hover:text-th-text-secondary rounded-md hover:bg-th-inset-bg transition-colors disabled:opacity-40"
                  onClick={onDownload}
                  disabled={!!activeJob || dlBlocked}
                  title="Re-download"
                >
                  <Download size={12} />
                </button>
              </>
            ) : isIncomplete ? (
              <button
                type="button"
                className="inline-flex items-center gap-1 px-2.5 py-1.5 text-[11px] font-semibold rounded-md bg-amber-500/15 text-amber-500 ring-1 ring-amber-500/30 hover:bg-amber-500/25 disabled:opacity-50 transition-colors"
                onClick={onDownload}
                disabled={!!activeJob || dlBlocked}
                title="Resume the interrupted download"
              >
                <Download size={11} />
                Continue download
              </button>
            ) : activeJob ? (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-[11px] font-medium rounded-md bg-th-inset-bg text-th-text-tertiary">
                <Loader2 size={11} className="animate-spin" />
                {activeJob.bytes_total > 0
                  ? `${Math.min(100, Math.round((activeJob.bytes_done / activeJob.bytes_total) * 100))}%`
                  : "Starting…"}
              </span>
            ) : (
              <button
                type="button"
                className="inline-flex items-center gap-1 px-2.5 py-1.5 text-[11px] font-semibold rounded-md bg-th-tab-active-bg text-th-tab-active-fg disabled:opacity-50 hover:opacity-90 transition-opacity"
                onClick={onDownload}
                disabled={dlBlocked}
                title={blockTitle}
              >
                <Download size={11} />
                Download
              </button>
            )}
          </div>
        </div>
        {!row.disk_ok && (
          <p className="text-[11px] text-rose-400 inline-flex items-center gap-1 mt-2">
            <AlertTriangle size={11} />
            Not enough free disk for {row.weights_gb.toFixed(1)} GB.
          </p>
        )}
      </div>
    </div>
  );
}

export default function DecisionPanel({
  settings,
  setSettings,
  onSave,
}: {
  settings: AppSettings;
  setSettings: (fn: (s: AppSettings) => AppSettings) => void;
  onSave: (next: AppSettings) => Promise<void>;
}) {
  const decision = settings.decision;
  const provider = providerOf(decision.endpoint_override);
  const [snap, setSnap] = useState<Snapshot | null>(null);
  const [error, setError] = useState("");
  const [jobNote, setJobNote] = useState("");
  const [download, setDownload] = useState<MlxDownloadJob | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [testSituation, setTestSituation] = useState("The user asked which of two files to edit, and only one was mentioned in the last message.");
  const [testQuestion, setTestQuestion] = useState("Should you ask which file they mean?");
  const [testResult, setTestResult] = useState("");
  const [busy, setBusy] = useState("");

  const refresh = useCallback(async () => {
    try {
      const next = await api.decisionSnapshot();
      setSnap(next);
      setError("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load decision status");
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    let cancelled = false;
    void api.mlxDownloadList().then((res) => {
      if (cancelled) return;
      const active = res.jobs.find((job) =>
        job.repo_id.startsWith("mlx-community/diffusiongemma-26B-A4B-it-")
        && (job.status === "pending" || job.status === "running" || job.status === "cancelling"),
      );
      if (active) setDownload((current) => current ?? active);
    }).catch(() => {
      /* the list still renders; download progress is optional */
    });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!download || download.status === "done" || download.status === "error" || download.status === "cancelled") return;
    const jobId = download.job_id;
    const timer = setInterval(async () => {
      try {
        const job = await api.mlxDownloadStatus(jobId);
        setDownload(job);
        if (job.status === "done" || job.status === "error" || job.status === "cancelled") {
          void refresh();
        }
      } catch {
        /* keep the last progress paint */
      }
    }, 1000);
    return () => clearInterval(timer);
  }, [download, refresh]);

  const patch = (partial: Partial<DecisionConfig>) => {
    setSettings((s) => {
      const next = { ...s, decision: { ...s.decision, ...partial } };
      void onSave(next);
      return next;
    });
  };

  const pollJob = async (jobId: string) => {
    for (let i = 0; i < 600; i++) {
      const job = await api.decisionJob(jobId);
      setJobNote(job.message || job.status);
      if (job.status === "done" || job.status === "error") {
        if (job.status === "error") setError(job.error || job.message);
        await refresh();
        return;
      }
      await new Promise((r) => setTimeout(r, 1000));
    }
  };

  const run = async (label: string, action: () => Promise<{ job_id: string }>) => {
    setBusy(label);
    setError("");
    try {
      const job = await action();
      await pollJob(job.job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : `${label} failed`);
    } finally {
      setBusy("");
    }
  };

  const startDownload = async (repo: string) => {
    setError("");
    try {
      const body: Record<string, unknown> = {
        repo_id: repo,
        safetensors_only: true,
        max_workers: 16,
        use_hf_transfer: true,
      };
      const token = settings.llm.mlx.hf_token?.trim();
      const cache = settings.llm.mlx.hf_hub_cache?.trim();
      if (token) body.hf_token = token;
      if (cache) body.cache_dir = cache;
      const started = await api.mlxDownload(body);
      setDownload({
        job_id: started.job_id,
        status: "pending",
        message: "",
        repo_id: started.repo_id,
        hub_cache: started.hub_cache,
        started_at: Date.now() / 1000,
        bytes_done: 0,
        bytes_total: 0,
        files_done: 0,
        files_total: 0,
        current_file: "",
        current_file_total: 0,
        rate_bps: 0,
        eta_seconds: null,
        use_hf_transfer: true,
        max_workers: 16,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Download failed to start");
    }
  };

  const useModel = (repo: string) => {
    patch({ model_repo: repo });
    void api.decisionSelect(repo).then(() => refresh()).catch((err) => {
      setError(err instanceof Error ? err.message : "Could not select model");
    });
  };

  const models = snap?.models ?? [];
  const status = snap?.status;
  const selected = models.find((m) => m.repo_id === decision.model_repo);

  return (
    <div className="space-y-6 max-w-3xl">
      <section className="bg-th-card-bg border border-th-card-border rounded-xl p-6 shadow-sm">
        <h2 className="text-base font-semibold text-th-text-primary mb-2">Jev</h2>
        <p className="text-xs text-th-text-tertiary leading-relaxed mb-4">
          A local classifier the agent can call when it is unsure between a few options.
          The answer is advisory. It does not approve shell commands, mail, or anything else.
          The 4-bit model is about 17 GB on disk and about 18 GB in memory while it is running.
          It starts on the first use and stops after it has been idle.
          Chats that are already open pick up a change to this setting on their next message.
        </p>
        <label className="flex items-center gap-2 text-sm text-th-text-primary">
          <input
            type="checkbox"
            checked={decision.enabled}
            onChange={(e) => patch({ enabled: e.target.checked })}
          />
          Enable the jev tool for every agent
        </label>
      </section>

      <div className="flex gap-1 bg-th-inset-bg rounded-xl p-1 w-fit border border-th-border">
        {([
          ["local", "Local"],
          ["typesafe", "TypeSafe"],
        ] as const).map(([id, label]) => (
          <button
            key={id}
            type="button"
            onClick={() => patch({ endpoint_override: id === "local" ? "" : TYPESAFE_URL })}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all duration-150 ${
              provider === id
                ? "bg-th-tab-active-bg text-th-tab-active-fg shadow-sm"
                : "text-th-text-tertiary hover:text-th-text-primary hover:bg-th-surface-hover"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {provider === "typesafe" && (
      <section className="bg-th-card-bg border border-th-card-border rounded-xl p-6 shadow-sm space-y-3">
          <h2 className="text-base font-semibold text-th-text-primary">TypeSafe API key</h2>
          <p className="text-xs text-th-text-tertiary">
            Each jev call goes to api.typesafe.ai. The situation leaves this machine.
            Refused while the privacy lock is on.
          </p>
          <div className="flex gap-2">
            <input
              type="password"
              value={apiKey}
              placeholder={status?.remote_key_set ? "API key is stored" : "TypeSafe API key"}
              onChange={(e) => setApiKey(e.target.value)}
              className="flex-1 px-3 py-2 bg-th-input-bg border border-th-input-border rounded-lg text-sm text-th-text-primary disabled:cursor-not-allowed"
            />
            <button
              type="button"
              className="px-3 py-1.5 text-xs rounded-lg border border-th-border text-th-text-primary disabled:opacity-50"
              onClick={() => {
                void api.decisionCredential(apiKey).then(() => {
                  setApiKey("");
                  void refresh();
                }).catch((err) => setError(err instanceof Error ? err.message : "Could not store key"));
              }}
            >
              Store key
            </button>
          </div>
      </section>
      )}

      {provider === "local" && (
      <>
      <section className="bg-th-card-bg border border-th-card-border rounded-xl p-6 shadow-sm space-y-3">
        <h2 className="text-base font-semibold text-th-text-primary">Model</h2>
        <p className="text-xs text-th-text-tertiary">
          Download a model, then use it. The same Hub cache and progress bar as the open-weight picker.
        </p>
        {decision.model_repo && (
          <div className="flex items-center gap-2 px-3 py-2 rounded-xl border border-emerald-500/30 bg-emerald-500/[0.08]">
            <CheckCircle2 size={13} className="text-emerald-400 shrink-0" />
            <div className="min-w-0 flex-1">
              <p className="text-[10px] font-medium uppercase tracking-wide leading-none mb-0.5 text-emerald-400">
                Selected model
              </p>
              <p className="text-xs font-semibold text-th-text-primary truncate font-mono">{decision.model_repo}</p>
            </div>
          </div>
        )}
        {download && (
          <DownloadProgress
            job={download}
            onCancel={() => { void api.mlxDownloadCancel(download.job_id); }}
            onDismiss={() => setDownload(null)}
          />
        )}
        <div className="space-y-2">
          {models.map((row) => (
            <ModelRow
              key={row.repo_id}
              row={row}
              isSelected={row.repo_id === decision.model_repo}
              activeJob={
                download && download.repo_id === row.repo_id
                  && (download.status === "pending" || download.status === "running" || download.status === "cancelling")
                  ? download
                  : undefined
              }
              onDownload={() => void startDownload(row.repo_id)}
              onUse={() => useModel(row.repo_id)}
            />
          ))}
        </div>
      </section>

      <section className="bg-th-card-bg border border-th-card-border rounded-xl p-6 shadow-sm space-y-3">
        <h2 className="text-base font-semibold text-th-text-primary">Local runtime</h2>
        <p className="text-xs text-th-text-tertiary">
          {status?.running
            ? `Running at ${status.endpoint}`
            : status?.installed
              ? `Installed (${(status.installed_ref || "").slice(0, 12) || "unknown"}). Stopped.`
              : "Not installed yet. A new install adds it automatically, in its own environment. Use the button to retry."}
          {status ? ` Fit: ${status.fit}.` : ""}
        </p>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            disabled={!!busy}
            className="px-3 py-1.5 text-xs font-medium rounded-lg border border-th-border text-th-text-primary hover:bg-th-surface-hover disabled:opacity-50"
            onClick={() => void run("install", () => api.decisionInstall())}
          >
            {busy === "install" ? <Loader2 size={12} className="animate-spin inline" /> : null} Install runtime
          </button>
          <button
            type="button"
            disabled={!!busy || selected?.fits === "over"}
            className="px-3 py-1.5 text-xs font-medium rounded-lg border border-th-border text-th-text-primary hover:bg-th-surface-hover disabled:opacity-50"
            onClick={() => void run("start", () => api.decisionStart())}
          >
            Start
          </button>
          <button
            type="button"
            disabled={!!busy}
            className="px-3 py-1.5 text-xs font-medium rounded-lg border border-th-border text-th-text-primary hover:bg-th-surface-hover disabled:opacity-50"
            onClick={() => void run("stop", () => api.decisionStop())}
          >
            Stop
          </button>
        </div>
        {jobNote && <p className="text-xs text-th-text-muted">{jobNote}</p>}
      </section>
      </>
      )}

      <section className="bg-th-card-bg border border-th-card-border rounded-xl p-6 shadow-sm space-y-3">
        <h2 className="text-base font-semibold text-th-text-primary">Thresholds</h2>
        <div className="grid grid-cols-2 gap-3">
          <label className="text-xs text-th-text-tertiary">
            Confident at or above
            <input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={decision.confident_threshold}
              onChange={(e) => patch({ confident_threshold: Number(e.target.value) })}
              className="mt-1 w-full px-3 py-2 bg-th-input-bg border border-th-input-border rounded-lg text-sm text-th-text-primary"
            />
          </label>
          <label className="text-xs text-th-text-tertiary">
            Unsure below
            <input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={decision.unsure_threshold}
              onChange={(e) => patch({ unsure_threshold: Number(e.target.value) })}
              className="mt-1 w-full px-3 py-2 bg-th-input-bg border border-th-input-border rounded-lg text-sm text-th-text-primary"
            />
          </label>
        </div>
        <p className="text-[11px] text-th-text-muted">
          Confidence is how peaked the answer is, not a calibrated probability. Keep these conservative.
        </p>
      </section>

      <section className="bg-th-card-bg border border-th-card-border rounded-xl p-6 shadow-sm space-y-3">
        <h2 className="text-base font-semibold text-th-text-primary">Test</h2>
        <textarea
          value={testSituation}
          onChange={(e) => setTestSituation(e.target.value)}
          rows={3}
          className="w-full px-3 py-2 bg-th-input-bg border border-th-input-border rounded-lg text-sm text-th-text-primary"
        />
        <input
          value={testQuestion}
          onChange={(e) => setTestQuestion(e.target.value)}
          className="w-full px-3 py-2 bg-th-input-bg border border-th-input-border rounded-lg text-sm text-th-text-primary"
        />
        <button
          type="button"
          className="px-3 py-1.5 text-xs font-medium rounded-lg border border-th-border text-th-text-primary hover:bg-th-surface-hover"
          onClick={() => {
            setTestResult("");
            setError("");
            void api.decisionTest({
              situation: testSituation,
              question: testQuestion,
              options: ["yes", "no"],
            }).then((res) => setTestResult(res.text)).catch((err) => {
              setError(err instanceof Error ? err.message : "Test failed");
            });
          }}
        >
          Run test
        </button>
        {testResult && (
          <pre className="text-xs whitespace-pre-wrap text-th-text-secondary bg-th-inset-bg rounded-lg p-3">{testResult}</pre>
        )}
      </section>

      {error && <p className="text-xs text-red-400">{error}</p>}
    </div>
  );
}
