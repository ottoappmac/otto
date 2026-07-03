import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  Brain,
  Play,
  Square,
  CheckCircle,
  XCircle,
  Loader2,
  AlertCircle,
  FileText,
  Clock,
  Database,
  Ban,
  RefreshCw,
  ChevronDown,
  Zap,
  BookOpen,
  Search,
  Trash2,
  PenLine,
  ChevronRight,
  ShieldAlert,
  Layers,
  Download,
  PackageCheck,
  Sparkles,
} from "lucide-react";
import { api } from "../hooks/useApi";
import type { MemoryHitsResponse } from "../hooks/useApi";
import type { AppSettings, MemoryConfig, MemoryTopic } from "../types";

const AUTO_SAVE_DELAY_MS = 600;
const STATUS_POLL_MS = 2000;

const MEMORY_SUBTABS = ["Status", "Topics", "Search Index", "Configuration"] as const;
type MemorySubTab = (typeof MEMORY_SUBTABS)[number];

type RunState = "idle" | "running" | "success" | "error" | "cancelled";

interface MemoryStatus {
  state: RunState;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
  transcripts_processed: number;
}

interface MemoryStats {
  total_transcripts: number;
  pending_transcripts: number;
  memory_files: number;
  last_consolidated_at: number | null;
  retention_days: number;
}

function formatTimeAgo(isoOrMs: string | number | null): string {
  if (!isoOrMs) return "Never";
  const ms = typeof isoOrMs === "number" ? isoOrMs : new Date(isoOrMs).getTime();
  const diff = Date.now() - ms;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "Just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

function StatusBadge({ state }: { state: RunState }) {
  const config: Record<RunState, { icon: typeof CheckCircle; label: string; color: string }> = {
    idle: { icon: Clock, label: "Idle", color: "text-th-text-muted bg-neutral-500/10 border-neutral-500/20" },
    running: { icon: Loader2, label: "Running", color: "text-blue-400 bg-blue-500/10 border-blue-500/20" },
    success: { icon: CheckCircle, label: "Success", color: "text-emerald-400 bg-emerald-500/10 border-emerald-500/20" },
    error: { icon: XCircle, label: "Error", color: "text-red-400 bg-red-500/10 border-red-500/20" },
    cancelled: { icon: Ban, label: "Cancelled", color: "text-amber-400 bg-amber-500/10 border-amber-500/20" },
  };
  const { icon: Icon, label, color } = config[state];
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${color}`}>
      <Icon size={12} className={state === "running" ? "animate-spin" : ""} />
      {label}
    </span>
  );
}

function StatCard({ icon: Icon, label, value }: { icon: typeof FileText; label: string; value: string | number }) {
  return (
    <div className="bg-th-card-bg border border-th-card-border rounded-xl p-4 flex items-center gap-3">
      <div className="p-2 rounded-lg bg-th-inset-bg">
        <Icon size={18} className="text-th-text-muted" />
      </div>
      <div>
        <p className="text-[11px] text-th-text-tertiary uppercase tracking-wider">{label}</p>
        <p className="text-lg font-semibold text-th-text-primary mt-0.5">{value}</p>
      </div>
    </div>
  );
}

export function MemoryPanel() {
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [status, setStatus] = useState<MemoryStatus>({ state: "idle", started_at: null, finished_at: null, error: null, transcripts_processed: 0 });
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [hits, setHits] = useState<MemoryHitsResponse | null>(null);
  const [saveStatus, setSaveStatus] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [triggerError, setTriggerError] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<{ id: string; name: string }[]>([]);
  const [fetchingModels, setFetchingModels] = useState(false);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [mlxLocalModels, setMlxLocalModels] = useState<{ repo_id: string; name: string; size_mb: number }[]>([]);
  const [mlxListLoading, setMlxListLoading] = useState(false);
  const [mlxListError, setMlxListError] = useState<string | null>(null);

  const loadedRef = useRef(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout>>();
  const savedTimerRef = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => {
    api.getSettings()
      .then((s) => { setSettings(s); loadedRef.current = true; })
      .catch((e) => console.warn("Failed to load settings:", e));
  }, []);

  const pickBestHaiku = (models: { id: string; name: string }[]) => {
    const haikus = models.filter((m) => /haiku/i.test(m.id) || /haiku/i.test(m.name));
    haikus.sort((a, b) => b.id.localeCompare(a.id));
    return haikus[0]?.id || "";
  };

  const handleFetchModels = async () => {
    if (!settings) return;
    setFetchingModels(true);
    setModelsError(null);
    try {
      const provider = settings.llm.provider;
      const a = settings.llm.anthropic;
      const o = settings.llm.openai;

      let req: Record<string, unknown>;
      if (provider === "openai") {
        req = {
          provider: "openai",
          api_key: o.model_provider === "azure" ? o.azure_api_key : o.api_key,
          model_name: o.model_name,
          openai_model_provider: o.model_provider,
          azure_endpoint: o.azure_endpoint,
          azure_api_version: o.azure_api_version,
          azure_deployment: o.azure_deployment,
        };
      } else {
        req = {
          provider: "anthropic",
          api_key: a.api_key,
          model_name: a.model_name,
          model_provider: a.model_provider,
          bedrock_region: a.bedrock_region,
          bedrock_auth_mode: a.bedrock_auth_mode,
          aws_access_key_id: a.aws_access_key_id,
          aws_secret_access_key: a.aws_secret_access_key,
        };
      }

      const result = await api.listModels(req);
      if (result.error) setModelsError(result.error);
      const models = result.models || [];
      setAvailableModels(models);
      if (models.length > 0 && !settings.memory.model_name) {
        const best = pickBestHaiku(models);
        if (best) updateMemoryField("model_name", best);
      }
    } catch (e: any) {
      setModelsError(e?.message || "Failed to fetch models");
    } finally {
      setFetchingModels(false);
    }
  };

  const modelsFetchedRef = useRef(false);
  useEffect(() => {
    if (!loadedRef.current || !settings || modelsFetchedRef.current) return;
    modelsFetchedRef.current = true;
    handleFetchModels();
  }, [settings]); // eslint-disable-line react-hooks/exhaustive-deps

  const refreshMlxModels = useCallback(async () => {
    setMlxListLoading(true);
    setMlxListError(null);
    try {
      const r = await api.mlxLocalModels();
      setMlxLocalModels(r.models || []);
      if (r.error) setMlxListError(r.error);
    } catch (e: any) {
      setMlxListError(e?.message || "Failed to list MLX models");
    } finally {
      setMlxListLoading(false);
    }
  }, []);

  const mlxFetchedRef = useRef(false);
  useEffect(() => {
    if (!loadedRef.current || !settings) return;
    if (mlxFetchedRef.current) return;
    if (settings.memory.llm_family !== "mlx" && settings.llm.provider !== "mlx") return;
    mlxFetchedRef.current = true;
    void refreshMlxModels();
  }, [settings, refreshMlxModels]);

  useEffect(() => {
    const poll = () => {
      api.getMemoryStatus().then((s) => setStatus(s as MemoryStatus)).catch(() => {});
      api.getMemoryStats().then(setStats).catch(() => {});
      api.getMemoryHits().then(setHits).catch(() => {});
    };
    poll();
    const interval = setInterval(poll, STATUS_POLL_MS);
    return () => clearInterval(interval);
  }, []);

  const persistSettings = useCallback(async (next: AppSettings) => {
    setSaveStatus("saving");
    try {
      await api.updateSettings(next as unknown as Record<string, unknown>);
      setSaveStatus("saved");
      savedTimerRef.current = setTimeout(() => setSaveStatus("idle"), 2000);
    } catch {
      setSaveStatus("error");
    }
  }, []);

  useEffect(() => {
    if (!loadedRef.current || !settings) return;
    clearTimeout(debounceRef.current);
    clearTimeout(savedTimerRef.current);
    debounceRef.current = setTimeout(() => persistSettings(settings), AUTO_SAVE_DELAY_MS);
    return () => clearTimeout(debounceRef.current);
  }, [settings, persistSettings]);

  const updateMemoryField = <K extends keyof MemoryConfig>(key: K, value: MemoryConfig[K]) => {
    setSettings((prev) => prev ? { ...prev, memory: { ...prev.memory, [key]: value } } : prev);
  };

  const handleRun = async () => {
    setTriggerError(null);
    try {
      const res = await api.triggerMemoryConsolidation();
      if (res.error) setTriggerError(res.error);
      if (res.reason) setTriggerError(res.reason);
    } catch (e: any) {
      setTriggerError(e?.message || "Failed to start");
    }
  };

  const [cancelling, setCancelling] = useState(false);

  const handleCancel = async () => {
    setCancelling(true);
    try { await api.cancelMemoryConsolidation(); } catch {}
    // The status poll will flip state to "cancelled"; clear our flag then.
    // Guard with a timeout so we never get stuck if the request fails silently.
    setTimeout(() => setCancelling(false), 5000);
  };

  // Clear the cancelling flag as soon as we see the status change away from running.
  useEffect(() => {
    if (status.state !== "running") setCancelling(false);
  }, [status.state]);

  const [memSubTab, setMemSubTab] = useState<MemorySubTab>("Status");

  if (!settings) {
    return (
      <div className="flex items-center justify-center h-full py-16">
        <Loader2 className="animate-spin text-th-text-muted" size={24} />
      </div>
    );
  }

  const rcfg = settings.memory;

  return (
    <div className="space-y-4 max-w-3xl">
      {/* Sub-pill bar — same style as LLM sub-pills */}
      <div className="flex gap-1 bg-th-inset-bg rounded-xl p-1 w-fit border border-th-border">
        {MEMORY_SUBTABS.map((t) => (
          <button
            key={t}
            type="button"
            className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all duration-150 ${
              memSubTab === t
                ? "bg-th-tab-active-bg text-th-tab-active-fg shadow-sm"
                : "text-th-text-tertiary hover:text-th-text-primary hover:bg-th-surface-hover"
            }`}
            onClick={() => setMemSubTab(t)}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Status row */}
      <div className="flex items-center gap-2.5">
        <span className={`h-2 w-2 rounded-full ${rcfg.enabled ? "bg-blue-500" : "bg-neutral-400"}`} />
        <span className="text-xs font-medium text-th-text-primary">Memory consolidation</span>
        <StatusBadge state={status.state} />
        {(saveStatus === "saving" || saveStatus === "saved" || saveStatus === "error") && (
          <span className={`text-[11px] ${saveStatus === "saved" ? "text-emerald-500/70" : saveStatus === "error" ? "text-red-500/70" : "text-th-text-muted"}`}>
            {saveStatus === "saving" ? "Saving…" : saveStatus === "saved" ? "Saved" : "Save failed"}
          </span>
        )}
      </div>

      {/* ── Status pill ───────────────────────────────────────── */}
      {memSubTab === "Status" && (
        <div className="space-y-4">
          {/* Stats row */}
          {stats && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <StatCard icon={FileText} label="Transcripts" value={stats.total_transcripts} />
              <StatCard icon={AlertCircle} label="Pending" value={stats.pending_transcripts} />
              <StatCard icon={Database} label="Memories" value={stats.memory_files} />
              <StatCard icon={Clock} label="Last Run" value={formatTimeAgo(stats.last_consolidated_at)} />
            </div>
          )}

          {/* Toggles + run button */}
          <div className="bg-th-card-bg border border-th-card-border rounded-xl p-5 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Toggle checked={rcfg.enabled} onChange={(v) => updateMemoryField("enabled", v)} />
                <div>
                  <p className="text-sm font-medium text-th-text-primary">Consolidate</p>
                  <p className="text-[12px] text-th-text-tertiary mt-0.5">
                    {rcfg.enabled
                      ? "Automatically builds memories from session transcripts"
                      : "Background memory consolidation is paused"}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {status.state === "running" ? (
                  <button
                    onClick={handleCancel}
                    disabled={cancelling}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-red-500/10 text-red-400 border border-red-500/20 hover:bg-red-500/20 transition-all disabled:opacity-60 disabled:cursor-wait"
                  >
                    {cancelling
                      ? <><Loader2 size={14} className="animate-spin" /> Cancelling…</>
                      : <><Square size={14} /> Cancel</>
                    }
                  </button>
                ) : (
                  <button onClick={handleRun} className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-blue-500/10 text-blue-400 border border-blue-500/20 hover:bg-blue-500/20 transition-all">
                    <Play size={14} /> Run Now
                  </button>
                )}
              </div>
            </div>

            <div className="border-t border-th-border pt-4 space-y-4">
              <div className={`flex items-center gap-3 ${!rcfg.enabled ? "opacity-40" : ""}`}>
                <Toggle
                  checked={rcfg.inject_on_session_start || rcfg.inject_enabled}
                  disabled={!rcfg.enabled}
                  onChange={(v) => {
                    setSettings((prev) => prev ? {
                      ...prev,
                      memory: { ...prev.memory, inject_on_session_start: v, inject_enabled: false, inject_realtime: prev.memory.inject_realtime || prev.memory.inject_enabled },
                    } : prev);
                  }}
                />
                <div>
                  <p className="text-sm font-medium text-th-text-primary">Inject at Session Start</p>
                  <p className="text-[12px] text-th-text-tertiary mt-0.5">
                    {!rcfg.enabled
                      ? "Enable Consolidate to use memory injection"
                      : (rcfg.inject_on_session_start || rcfg.inject_enabled)
                        ? "MEMORY.md is loaded into the system prompt once when a session begins"
                        : "Session-start memory injection is off"}
                  </p>
                </div>
              </div>
              <div className={`flex items-center gap-3 ${!rcfg.enabled ? "opacity-40" : ""}`}>
                <Toggle
                  checked={rcfg.inject_realtime || rcfg.inject_enabled}
                  disabled={!rcfg.enabled}
                  onChange={(v) => {
                    setSettings((prev) => prev ? {
                      ...prev,
                      memory: { ...prev.memory, inject_realtime: v, inject_enabled: false, inject_on_session_start: prev.memory.inject_on_session_start || prev.memory.inject_enabled },
                    } : prev);
                  }}
                />
                <div>
                  <p className="text-sm font-medium text-th-text-primary">Inject in Realtime</p>
                  <p className="text-[12px] text-th-text-tertiary mt-0.5">
                    {!rcfg.enabled
                      ? "Enable Consolidate to use memory injection"
                      : (rcfg.inject_realtime || rcfg.inject_enabled)
                        ? "A ranking model picks relevant memories each turn and injects them on the fly"
                        : "Per-turn memory retrieval is off"}
                  </p>
                </div>
              </div>
            </div>

            {triggerError && (
              <div className="mt-3 p-2.5 rounded-lg bg-amber-500/5 border border-amber-500/15 text-amber-400 text-xs flex items-center gap-2">
                <AlertCircle size={13} />{triggerError}
              </div>
            )}
          </div>

          {/* Embedding model install banner — shown when memory is on */}
          <ModelInstallBanner memoryEnabled={rcfg.enabled} />

          {/* Run result banners */}
          {status.state === "running" && (
            <div className="bg-blue-500/5 border border-blue-500/15 rounded-xl p-4 flex items-center gap-3">
              <Loader2 size={18} className="text-blue-400 animate-spin" />
              <div>
                <p className="text-sm text-blue-300 font-medium">Consolidation in progress…</p>
                {status.started_at && <p className="text-[11px] text-blue-400/60 mt-0.5">Started {formatTimeAgo(status.started_at)}</p>}
              </div>
            </div>
          )}
          {status.state === "success" && status.finished_at && (
            <div className="bg-emerald-500/5 border border-emerald-500/15 rounded-xl p-4 flex items-center gap-3">
              <CheckCircle size={18} className="text-emerald-400" />
              <div>
                <p className="text-sm text-emerald-300 font-medium">Last run completed successfully</p>
                <p className="text-[11px] text-emerald-400/60 mt-0.5">{status.transcripts_processed} transcript(s) processed · {formatTimeAgo(status.finished_at)}</p>
              </div>
            </div>
          )}
          {status.state === "error" && (
            <div className="bg-red-500/5 border border-red-500/15 rounded-xl p-4 flex items-center gap-3">
              <XCircle size={18} className="text-red-400" />
              <div>
                <p className="text-sm text-red-300 font-medium">Last run failed</p>
                {status.error && <p className="text-[11px] text-red-400/60 mt-0.5 font-mono break-all">{status.error}</p>}
              </div>
            </div>
          )}
          {status.state === "cancelled" && (
            <div className="bg-amber-500/5 border border-amber-500/15 rounded-xl p-4 flex items-center gap-3">
              <Ban size={18} className="text-amber-400" />
              <div>
                <p className="text-sm text-amber-300 font-medium">Last run was cancelled</p>
                {status.finished_at && <p className="text-[11px] text-amber-400/60 mt-0.5">{formatTimeAgo(status.finished_at)}</p>}
              </div>
            </div>
          )}

          {/* Memory hits */}
          {hits && hits.total_injections > 0 && (
            <div className="bg-th-card-bg border border-th-card-border rounded-xl p-5">
              <h2 className="text-sm font-semibold text-th-text-secondary mb-4 flex items-center gap-2">
                <Zap size={15} className="text-blue-400" /> Memory Hits
              </h2>
              <div className="grid grid-cols-2 gap-3 mb-4">
                <div className="bg-th-inset-bg rounded-lg p-3 text-center">
                  <p className="text-lg font-semibold text-blue-400">{hits.total_injections}</p>
                  <p className="text-[10px] text-th-text-tertiary uppercase tracking-wider mt-0.5">Injections</p>
                </div>
                <div className="bg-th-inset-bg rounded-lg p-3 text-center">
                  <p className="text-lg font-semibold text-th-text-primary">{hits.unique_sessions}</p>
                  <p className="text-[10px] text-th-text-tertiary uppercase tracking-wider mt-0.5">Sessions</p>
                </div>
              </div>
              {hits.top_topics.length > 0 && (
                <div className="mb-4">
                  <p className="text-[11px] text-th-text-tertiary uppercase tracking-wider mb-2">Most Used</p>
                  <div className="space-y-1.5">
                    {hits.top_topics.slice(0, 5).map((t) => {
                      const maxCount = hits.top_topics[0].count;
                      const pct = Math.round((t.count / maxCount) * 100);
                      return (
                        <div key={t.topic} className="flex items-center gap-2">
                          <div className="flex-1 h-5 bg-th-code-bg rounded overflow-hidden relative">
                            <div className="h-full bg-blue-500/20 rounded" style={{ width: `${pct}%` }} />
                            <span className="absolute inset-0 flex items-center px-2 text-[11px] text-th-text-secondary truncate">{t.topic.replace(/\.md$/, "")}</span>
                          </div>
                          <span className="text-[11px] text-th-text-tertiary w-8 text-right tabular-nums">{t.count}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
              {hits.recent.length > 0 && (
                <div>
                  <p className="text-[11px] text-th-text-tertiary uppercase tracking-wider mb-2">Recent</p>
                  <div className="space-y-1 max-h-32 overflow-y-auto">
                    {hits.recent.slice(0, 10).map((r, i) => (
                      <div key={i} className="flex items-center gap-2 text-[11px]">
                        <span className="text-th-text-muted w-12 shrink-0">{r.ts ? new Date(r.ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : ""}</span>
                        <span className="text-th-text-tertiary w-16 shrink-0 truncate font-mono">{r.session_id}</span>
                        <span className="text-th-text-tertiary truncate">{r.topics.map((t) => t.replace(/\.md$/, "")).join(", ")}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ── Topics pill ─────────────────────────────────────── */}
      {memSubTab === "Topics" && <MemoryTopicsPanel memoryEnabled={rcfg.enabled} />}

      {/* ── Search Index pill ────────────────────────────────── */}
      {memSubTab === "Search Index" && (
        <EmbeddingIndexPanel
          memoryEnabled={rcfg.enabled}
          embeddingEnabled={rcfg.embedding?.enabled ?? true}
          onToggleEmbedding={(v) =>
            setSettings((prev) =>
              prev
                ? {
                    ...prev,
                    memory: {
                      ...prev.memory,
                      embedding: { ...(prev.memory.embedding ?? { model_name: "sentence-transformers/all-MiniLM-L6-v2", chunk_size: 1500, chunk_overlap: 150 }), enabled: v },
                    },
                  }
                : prev
            )
          }
        />
      )}

      {/* ── Configuration pill ───────────────────────────────── */}
      {memSubTab === "Configuration" && (
        <div className="bg-th-card-bg border border-th-card-border rounded-xl p-5 space-y-5">
          <FamilySelectField
            value={(rcfg.llm_family || "follow_main")}
            onChange={(v) => {
              updateMemoryField("llm_family", v);
              if (v === "mlx" && mlxLocalModels.length === 0 && !mlxListLoading) void refreshMlxModels();
            }}
            mainProvider={settings.llm.provider}
          />
          {(() => {
            const family = rcfg.llm_family || "follow_main";
            const usesMlx = family === "mlx" || (family === "follow_main" && settings.llm.provider === "mlx");
            if (usesMlx) {
              return (
                <MlxModelPickerField
                  value={rcfg.mlx_model || ""}
                  onChange={(v) => updateMemoryField("mlx_model", v)}
                  models={mlxLocalModels}
                  loading={mlxListLoading}
                  error={mlxListError}
                  onFetch={() => void refreshMlxModels()}
                  fallbackHint={settings.llm.mlx?.hf_llm_model_id || ""}
                />
              );
            }
            return (
              <ModelPickerField
                value={rcfg.model_name}
                onChange={(v) => updateMemoryField("model_name", v)}
                models={availableModels}
                loading={fetchingModels}
                error={modelsError}
                onFetch={handleFetchModels}
              />
            );
          })()}
          <InputField
            label="Minimum Hours Between Runs"
            description="How many hours must pass since the last consolidation before a new one triggers."
            value={String(rcfg.min_hours)}
            onChange={(v) => updateMemoryField("min_hours", Math.max(1, parseInt(v) || 1))}
            type="number" min={1}
          />
          <InputField
            label="Minimum Sessions"
            description="Number of new session transcripts required before auto-triggering."
            value={String(rcfg.min_sessions)}
            onChange={(v) => updateMemoryField("min_sessions", Math.max(1, parseInt(v) || 1))}
            type="number" min={1}
          />
          <InputField
            label="Retention Days"
            description="Transcript files older than this are deleted after consolidation."
            value={String(rcfg.retention_days)}
            onChange={(v) => updateMemoryField("retention_days", Math.max(1, parseInt(v) || 7))}
            type="number" min={1}
          />
          <InputField
            label="Max Memory Files"
            description="Maximum number of topic files in the memory directory. Oldest pruned first."
            value={String(rcfg.max_memory_files)}
            onChange={(v) => updateMemoryField("max_memory_files", Math.max(10, parseInt(v) || 50))}
            type="number" min={10}
          />
          <InputField
            label="Max Index Size (KB)"
            description="Warns if MEMORY.md exceeds this size."
            value={String(rcfg.max_index_kb)}
            onChange={(v) => updateMemoryField("max_index_kb", Math.max(5, parseInt(v) || 25))}
            type="number" min={5}
          />
        </div>
      )}


    </div>
  );
}

// ---------------------------------------------------------------------------
// Ambient assistant settings panel
// ---------------------------------------------------------------------------

interface AmbientPanelProps {
  settings: AppSettings;
  setSettings: React.Dispatch<React.SetStateAction<AppSettings | null>>;
  mlxLocalModels: { repo_id: string; name: string; size_mb: number }[];
  mlxListLoading: boolean;
  mlxListError: string | null;
  onRefreshMlxModels: () => void;
}

export function AmbientPanel({
  settings,
  setSettings,
  mlxLocalModels,
  mlxListLoading,
  mlxListError,
  onRefreshMlxModels,
}: AmbientPanelProps) {
  const ambient = settings.ambient ?? {
    enabled: false, llm_family: "mlx",
    mlx_model: "mlx-community/Qwen3-1.7B-4bit", model_name: "",
    interval_mins: 30, idle_only: true, react_to_session_end: true,
    use_memory: true, use_sessions: true, use_activity: true, use_history: true,
    lookback_hours: 24,
    min_confidence: 0.6, max_hints_per_day: 10, cooldown_hours: 4,
    quiet_hours_start: 22, quiet_hours_end: 8, allow_auto_run: true,
  };

  const updateAmbient = <K extends keyof typeof ambient>(key: K, value: (typeof ambient)[K]) =>
    setSettings((prev) => prev ? { ...prev, ambient: { ...prev.ambient, [key]: value } } : prev);

  const mainProvider = settings.llm.provider;
  const usesMlx =
    ambient.llm_family === "mlx" ||
    (ambient.llm_family === "follow_main" && mainProvider === "mlx");

  const followLabel =
    mainProvider === "mlx"
      ? "Same as default model (On-Device)"
      : mainProvider === "exo"
      ? "Same as default model (Cluster)"
      : "Same as default model (Frontier)";

  const mlxOpts: { value: string; label: string }[] = [
    { value: "", label: ambient.mlx_model ? `(default — ${ambient.mlx_model})` : "(default — Qwen3-1.7B)" },
  ];
  const seen = new Set<string>();
  for (const m of mlxLocalModels) {
    if (!seen.has(m.repo_id)) {
      seen.add(m.repo_id);
      mlxOpts.push({ value: m.repo_id, label: m.name });
    }
  }
  if (ambient.mlx_model && !seen.has(ambient.mlx_model)) {
    mlxOpts.push({ value: ambient.mlx_model, label: ambient.mlx_model });
  }

  return (
    <div className="space-y-5 max-w-2xl">
      {/* Enable card */}
      <div className="bg-th-card-bg border border-th-card-border rounded-xl p-5">
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-center gap-3 min-w-0">
            <div className="p-2 rounded-lg bg-blue-500/10 border border-blue-500/20 shrink-0">
              <Sparkles size={18} className="text-blue-400" />
            </div>
            <div>
              <p className="text-sm font-medium text-th-text-primary">Suggestions</p>
              <p className="text-xs text-th-text-muted mt-0.5 max-w-sm">
                Proactively surfaces actionable suggestions by analysing your memory,
                sessions, and macOS activity in the background.
              </p>
            </div>
          </div>
          <Toggle
            checked={ambient.enabled}
            onChange={(v) => updateAmbient("enabled", v)}
          />
        </div>
      </div>

      {ambient.enabled && (
        <>
          {/* Model selection */}
          <div className="bg-th-card-bg border border-th-card-border rounded-xl p-5 space-y-4">
            <p className="text-sm font-medium text-th-text-primary">Smaller model for hints</p>
            <p className="text-xs text-th-text-muted -mt-2">
              The ambient agent uses its own lightweight model so hint generation is cheap and
              doesn't compete with your main chat model.
            </p>

            {/* Family picker */}
            <div>
              <label className="block text-sm text-th-text-secondary font-medium mb-1">Model stack</label>
              <div className="relative">
                <select
                  className="w-full appearance-none px-3 py-2 pr-10 bg-th-input-bg border border-th-input-border rounded-lg focus:outline-none focus:border-blue-400 transition-colors text-sm text-th-text-primary"
                  value={ambient.llm_family}
                  onChange={(e) => updateAmbient("llm_family", e.target.value)}
                >
                  <option value="follow_main">{followLabel}</option>
                  <option value="frontier">Frontier (Anthropic / Bedrock)</option>
                  <option value="mlx">On-Device (MLX) — recommended</option>
                  <option value="exo">Cluster</option>
                </select>
                <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-th-text-muted pointer-events-none" />
              </div>
            </div>

            {/* MLX model picker */}
            {usesMlx && (
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="block text-sm text-th-text-secondary font-medium">On-device model</label>
                  <button
                    type="button"
                    className="text-th-text-tertiary hover:text-th-text-secondary transition-colors flex items-center gap-1 text-xs"
                    onClick={onRefreshMlxModels}
                    disabled={mlxListLoading}
                  >
                    <RefreshCw size={12} className={mlxListLoading ? "animate-spin" : ""} />
                    {mlxListLoading ? "Fetching…" : "Refresh"}
                  </button>
                </div>
                <p className="text-[11px] text-th-text-muted mb-2">
                  Pick a small cached model. The default (Qwen3-1.7B-4bit, ~1 GB) is already
                  downloaded on most Apple Silicon Macs after first-run setup.
                </p>
                <div className="relative">
                  <select
                    className="w-full appearance-none px-3 py-2 pr-10 bg-th-input-bg border border-th-input-border rounded-lg focus:outline-none focus:border-blue-400 transition-colors text-sm text-th-text-primary"
                    value={mlxOpts.some((o) => o.value === ambient.mlx_model) ? ambient.mlx_model : ""}
                    onChange={(e) => updateAmbient("mlx_model", e.target.value)}
                  >
                    {mlxOpts.map((o) => (
                      <option key={o.value || "__default__"} value={o.value}>{o.label}</option>
                    ))}
                  </select>
                  <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-th-text-muted pointer-events-none" />
                </div>
                {mlxListError && <p className="mt-1.5 text-xs text-red-400">{mlxListError}</p>}
              </div>
            )}
          </div>

          {/* Cadence */}
          <div className="bg-th-card-bg border border-th-card-border rounded-xl p-5 space-y-4">
            <p className="text-sm font-medium text-th-text-primary">Cadence</p>
            <InputField
              label="Check interval (minutes)"
              description="How often the ambient agent runs a sweep in the background."
              value={String(ambient.interval_mins)}
              onChange={(v) => updateAmbient("interval_mins", Math.max(5, parseInt(v) || 30))}
              type="number" min={5}
            />
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-sm text-th-text-secondary font-medium">Only run when idle</p>
                <p className="text-[11px] text-th-text-muted mt-0.5">
                  Skip sweeps while you're actively typing or using the computer.
                </p>
              </div>
              <Toggle checked={ambient.idle_only} onChange={(v) => updateAmbient("idle_only", v)} />
            </div>
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-sm text-th-text-secondary font-medium">Sweep after each session</p>
                <p className="text-[11px] text-th-text-muted mt-0.5">
                  Trigger an additional sweep shortly after a chat session completes.
                </p>
              </div>
              <Toggle checked={ambient.react_to_session_end} onChange={(v) => updateAmbient("react_to_session_end", v)} />
            </div>
          </div>

          {/* Context sources */}
          <div className="bg-th-card-bg border border-th-card-border rounded-xl p-5 space-y-3">
            <p className="text-sm font-medium text-th-text-primary">Context sources</p>
            {(
              [
                { key: "use_memory", label: "Long-term memory", desc: "Include memory topics and the MEMORY.md index." },
                { key: "use_sessions", label: "Recent sessions", desc: "Include sessions within the lookback window." },
                { key: "use_activity", label: "macOS activity", desc: "Include app/window activity within the lookback window." },
                { key: "use_history", label: "Usage history", desc: "Include tool usage frequency across sessions in the lookback window." },
              ] as const
            ).map(({ key, label, desc }) => (
              <div key={key} className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-sm text-th-text-secondary font-medium">{label}</p>
                  <p className="text-[11px] text-th-text-muted mt-0.5">{desc}</p>
                </div>
                <Toggle
                  checked={Boolean(ambient[key])}
                  onChange={(v) => updateAmbient(key, v)}
                />
              </div>
            ))}
            <InputField
              label="Lookback window (hours)"
              description="How far back sessions, activity, and history gatherers look. Memory is a static index and is not time-filtered."
              value={String(ambient.lookback_hours ?? 24)}
              onChange={(v) => updateAmbient("lookback_hours", Math.max(1, Math.min(168, parseInt(v) || 24)))}
              type="number" min={1} max={168}
            />
          </div>

          {/* Rate limits */}
          <div className="bg-th-card-bg border border-th-card-border rounded-xl p-5 space-y-4">
            <p className="text-sm font-medium text-th-text-primary">Rate limits & quality</p>
            <InputField
              label="Max hints per day"
              description="Hard cap on new suggestions generated per 24-hour period."
              value={String(ambient.max_hints_per_day)}
              onChange={(v) => updateAmbient("max_hints_per_day", Math.max(1, parseInt(v) || 10))}
              type="number" min={1}
            />
            <InputField
              label="Cooldown (hours)"
              description="Minimum hours before a similar suggestion can resurface."
              value={String(ambient.cooldown_hours)}
              onChange={(v) => updateAmbient("cooldown_hours", Math.max(1, parseInt(v) || 4))}
              type="number" min={1}
            />
            <InputField
              label="Quiet hours start (24h)"
              description="Hour at which notifications are suppressed (e.g. 22 = 10 pm)."
              value={String(ambient.quiet_hours_start)}
              onChange={(v) => updateAmbient("quiet_hours_start", Math.min(23, Math.max(0, parseInt(v) || 22)))}
              type="number" min={0} max={23}
            />
            <InputField
              label="Quiet hours end (24h)"
              description="Hour at which notifications resume (e.g. 8 = 8 am)."
              value={String(ambient.quiet_hours_end)}
              onChange={(v) => updateAmbient("quiet_hours_end", Math.min(23, Math.max(0, parseInt(v) || 8)))}
              type="number" min={0} max={23}
            />
          </div>

          {/* Approval */}
          <div className="bg-th-card-bg border border-th-card-border rounded-xl p-5">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-sm font-medium text-th-text-secondary">Allow one-click approve &amp; run</p>
                <p className="text-[11px] text-th-text-muted mt-0.5 max-w-sm">
                  When enabled, each hint gains an "Approve &amp; run" button that spawns a
                  background session immediately.  All risky tool calls still require your
                  approval via the normal HITL flow.
                </p>
              </div>
              <Toggle
                checked={ambient.allow_auto_run}
                onChange={(v) => updateAmbient("allow_auto_run", v)}
              />
            </div>
          </div>
        </>
      )}
    </div>
  );
}

/** Standalone page wrapper — kept for the /memory route redirect. */
export default function MemoryPage() {
  return (
    <div className="h-full overflow-y-auto">
      <div className="px-6 py-6">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2.5 rounded-xl bg-blue-500/10 border border-blue-500/20">
            <Brain size={22} className="text-blue-400" />
          </div>
          <h1 className="text-lg font-semibold text-th-text-primary">Memory</h1>
        </div>
        <MemoryPanel />
      </div>
    </div>
  );
}

/* ---------- Local UI components (matching SettingsPage patterns) ---------- */

function Toggle({ checked, onChange, disabled }: { checked: boolean; onChange: (v: boolean) => void; disabled?: boolean }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => !disabled && onChange(!checked)}
      className={`relative w-11 h-6 rounded-full transition-all duration-200 border-0 ${
        disabled
          ? "opacity-35 cursor-not-allowed bg-neutral-600/50"
          : checked ? "bg-blue-500" : "bg-neutral-600/50"
      }`}
    >
      <span
        className={`absolute top-0.5 left-0.5 w-[18px] h-[18px] rounded-full transition-all duration-200 shadow ${
          checked && !disabled ? "translate-x-[20px] bg-white" : "translate-x-0 bg-neutral-400"
        }`}
      />
    </button>
  );
}

function InputField({
  label,
  description,
  value,
  onChange,
  type = "text",
  min,
  max,
}: {
  label: string;
  description?: string;
  value: string;
  onChange: (v: string) => void;
  type?: string;
  min?: number;
  max?: number;
}) {
  return (
    <div>
      <label className="block text-sm text-th-text-secondary font-medium mb-1">{label}</label>
      {description && <p className="text-[11px] text-th-text-muted mb-2">{description}</p>}
      <input
        type={type}
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-th-input-bg border border-th-input-border rounded-lg px-3 py-2 text-sm text-th-text-primary focus:outline-none focus:border-blue-400 transition-colors"
      />
    </div>
  );
}

function ModelPickerField({ value, onChange, models, loading, error, onFetch }: {
  value: string;
  onChange: (v: string) => void;
  models: { id: string; name: string }[];
  loading: boolean;
  error: string | null;
  onFetch: () => void;
}) {
  const [showAll, setShowAll] = useState(false);
  const haikuModels = models.filter((m) => /haiku/i.test(m.id) || /haiku/i.test(m.name));
  const displayModels = showAll ? models : haikuModels;
  const hasModels = displayModels.length > 0;
  const currentInList = displayModels.some((m) => m.id === value);

  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <label className="block text-sm text-th-text-secondary font-medium">Ranking Model</label>
        <div className="flex items-center gap-2">
          {models.length > 0 && (
            <button type="button" className="text-xs text-th-text-tertiary hover:text-th-text-secondary transition-colors" onClick={() => setShowAll(!showAll)}>
              {showAll ? "Haiku only" : "Show all models"}
            </button>
          )}
          <button type="button" className="text-th-text-tertiary hover:text-th-text-secondary transition-colors flex items-center gap-1 text-xs" onClick={onFetch} disabled={loading}>
            <RefreshCw size={12} className={loading ? "animate-spin" : ""} />
            {loading ? "Fetching…" : "Refresh"}
          </button>
        </div>
      </div>
      <p className="text-[11px] text-th-text-muted mb-2">Fast, low-cost model used for consolidation and per-turn memory relevance ranking.</p>
      {hasModels ? (
        <div className="relative">
          <select
            className={`w-full appearance-none px-3 py-2 pr-10 bg-th-input-bg border rounded-lg focus:outline-none focus:border-blue-400 transition-colors text-sm ${value ? "border-th-input-border text-th-text-primary" : "border-th-input-border text-th-text-muted"}`}
            value={currentInList ? value : ""}
            onChange={(e) => { if (e.target.value) onChange(e.target.value); }}
          >
            {!currentInList && value && <option value="">{value} (custom)</option>}
            {!value && <option value="">Select a model…</option>}
            {displayModels.map((m) => <option key={m.id} value={m.id}>{m.name} ({m.id})</option>)}
          </select>
          <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-th-text-muted pointer-events-none" />
        </div>
      ) : (
        <div className="flex items-center gap-2 text-xs text-th-text-muted py-2">
          {loading ? (
            <><Loader2 size={12} className="animate-spin" /> Loading models…</>
          ) : (
            <span>{value || "Auto-detected on first use"}</span>
          )}
        </div>
      )}
      {error && <p className="mt-1.5 text-xs text-red-400">{error}</p>}
    </div>
  );
}

function FamilySelectField({ value, onChange, mainProvider }: {
  value: string;
  onChange: (v: string) => void;
  mainProvider: string;
}) {
  const followLabel =
    mainProvider === "mlx"
      ? "Same as default model (On-Device)"
      : mainProvider === "exo"
      ? "Same as default model (Cluster)"
      : "Same as default model (Frontier)";
  return (
    <div>
      <label className="block text-sm text-th-text-secondary font-medium mb-1">Consolidation stack</label>
      <p className="text-[11px] text-th-text-muted mb-2">
        Which model runs the consolidation pipeline and per-turn memory ranking. On-Device runs locally; Frontier uses Anthropic / Bedrock; Cluster uses the distributed inference nodes.
      </p>
      <div className="relative">
        <select
          className="w-full appearance-none px-3 py-2 pr-10 bg-th-input-bg border border-th-input-border rounded-lg focus:outline-none focus:border-blue-400 transition-colors text-sm text-th-text-primary"
          value={value}
          onChange={(e) => onChange(e.target.value)}
        >
          <option value="follow_main">{followLabel}</option>
          <option value="frontier">Frontier (Anthropic / Bedrock)</option>
          <option value="mlx">On-Device (MLX)</option>
          <option value="exo">Cluster</option>
        </select>
        <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-th-text-muted pointer-events-none" />
      </div>
    </div>
  );
}

function MlxModelPickerField({ value, onChange, models, loading, error, onFetch, fallbackHint }: {
  value: string;
  onChange: (v: string) => void;
  models: { repo_id: string; name: string; size_mb: number }[];
  loading: boolean;
  error: string | null;
  onFetch: () => void;
  fallbackHint: string;
}) {
  const seen = new Set<string>();
  const opts: { value: string; label: string }[] = [
    { value: "", label: fallbackHint ? `(default — ${fallbackHint})` : "(default — global MLX text model)" },
  ];
  for (const m of models) {
    if (!seen.has(m.repo_id)) {
      seen.add(m.repo_id);
      opts.push({ value: m.repo_id, label: m.name });
    }
  }
  if (value && !seen.has(value)) {
    opts.push({ value, label: `${value} (not in cache listing yet)` });
  }
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <label className="block text-sm text-th-text-secondary font-medium">On-device consolidation model</label>
        <button type="button" className="text-th-text-tertiary hover:text-th-text-secondary transition-colors flex items-center gap-1 text-xs" onClick={onFetch} disabled={loading}>
          <RefreshCw size={12} className={loading ? "animate-spin" : ""} />
          {loading ? "Fetching…" : "Refresh"}
        </button>
      </div>
      <p className="text-[11px] text-th-text-muted mb-2">
        Pick a local on-device model from the cache. Leave on default to reuse the chat model.
      </p>
      <div className="relative">
        <select
          className="w-full appearance-none px-3 py-2 pr-10 bg-th-input-bg border border-th-input-border rounded-lg focus:outline-none focus:border-blue-400 transition-colors text-sm text-th-text-primary"
          value={opts.some((o) => o.value === value) ? value : ""}
          onChange={(e) => onChange(e.target.value)}
        >
          {opts.map((o) => <option key={o.value || "__default__"} value={o.value}>{o.label}</option>)}
        </select>
        <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-th-text-muted pointer-events-none" />
      </div>
      {error && <p className="mt-1.5 text-xs text-red-400">{error}</p>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Embedding model install banner
// ---------------------------------------------------------------------------

interface ModelStatus {
  installed: boolean;
  model_name: string;
  downloading: boolean;
  bytes_downloaded: number;
  total_bytes: number;
  error: string | null;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
}

function ModelInstallBanner({ memoryEnabled }: { memoryEnabled: boolean }) {
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [checking, setChecking] = useState(true);
  const pollRef = useRef<ReturnType<typeof setInterval>>();
  // Track whether we've already kicked off an auto-download this mount so we
  // don't fire it twice if the component re-renders before the request lands.
  const autoStartedRef = useRef(false);

  const startDownload = useCallback(async () => {
    try {
      await api.startModelDownload();
    } catch (e: any) {
      setModelStatus((prev) => prev ? { ...prev, error: e?.message ?? "Failed to start download" } : prev);
    }
  }, []);

  const fetchStatus = useCallback(async () => {
    try {
      const s = await api.getEmbeddingModelStatus();
      setModelStatus(s);
      setChecking(false);
      // Stop polling once fully installed or failed (not currently downloading)
      if (s.installed || (s.error && !s.downloading)) {
        clearInterval(pollRef.current);
      }
    } catch {
      setChecking(false);
    }
  }, []);

  // On mount (or when memory is toggled on): fetch status, then auto-start if needed.
  useEffect(() => {
    if (!memoryEnabled) return;

    const init = async () => {
      try {
        const s = await api.getEmbeddingModelStatus();
        setModelStatus(s);
        setChecking(false);

        if (!s.installed && !s.downloading && !autoStartedRef.current) {
          autoStartedRef.current = true;
          await startDownload();
        }
      } catch {
        setChecking(false);
      }
    };

    void init();
    pollRef.current = setInterval(fetchStatus, 1500);
    return () => clearInterval(pollRef.current);
  }, [memoryEnabled, fetchStatus, startDownload]);

  const handleRetry = async () => {
    autoStartedRef.current = false; // allow re-trigger
    await startDownload();
    clearInterval(pollRef.current);
    pollRef.current = setInterval(fetchStatus, 1000);
    void fetchStatus();
  };

  if (!memoryEnabled || checking || !modelStatus) return null;

  if (modelStatus.installed) {
    return (
      <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-emerald-500/5 border border-emerald-500/15 text-[12px]">
        <PackageCheck size={14} className="text-emerald-400 shrink-0" />
        <span className="text-emerald-300 font-medium">Embedding model installed</span>
        <span className="text-emerald-500/60 truncate ml-1">{modelStatus.model_name}</span>
      </div>
    );
  }

  // Downloading (auto-started or previously triggered)
  if (modelStatus.downloading) {
    const pct = modelStatus.total_bytes > 0
      ? Math.min(100, Math.round((modelStatus.bytes_downloaded / modelStatus.total_bytes) * 100))
      : null;
    return (
      <div className="rounded-xl bg-th-card-bg border border-th-card-border p-4 space-y-3">
        <div className="flex items-center gap-2">
          <Loader2 size={15} className="text-blue-400 animate-spin shrink-0" />
          <p className="text-sm font-medium text-th-text-primary">Downloading embedding model…</p>
        </div>
        <div className="space-y-1.5">
          <div className="w-full h-1.5 bg-th-inset-bg rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 rounded-full transition-all duration-300"
              style={{ width: pct !== null ? `${pct}%` : "30%" }}
            />
          </div>
          <div className="flex items-center justify-between text-[11px] text-th-text-tertiary">
            <span>
              {formatBytes(modelStatus.bytes_downloaded)}
              {modelStatus.total_bytes > 0 && ` / ${formatBytes(modelStatus.total_bytes)}`}
            </span>
            {pct !== null && <span className="tabular-nums">{pct}%</span>}
          </div>
        </div>
        <p className="text-[11px] text-th-text-muted">
          {modelStatus.model_name} · Required for semantic memory search · Only downloads once
        </p>
      </div>
    );
  }

  // Not installed, not downloading — either initial state briefly visible before
  // the auto-start kicks in, or a failed download awaiting retry.
  return (
    <div className="rounded-xl bg-th-card-bg border border-amber-500/20 p-4 space-y-2">
      <div className="flex items-start gap-3">
        <div className="p-2 rounded-lg bg-amber-500/10 shrink-0 mt-0.5">
          <Download size={14} className="text-amber-400" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-th-text-primary">
            {modelStatus.error ? "Embedding model download failed" : "Preparing embedding model…"}
          </p>
          <p className="text-[12px] text-th-text-tertiary mt-0.5">
            {modelStatus.error
              ? "The download didn't complete. Check your connection and try again."
              : "all-MiniLM-L6-v2 (~90 MB) · Required for semantic search"}
          </p>
          {modelStatus.error && (
            <p className="text-[11px] text-red-400 mt-1 font-mono break-all">{modelStatus.error}</p>
          )}
        </div>
      </div>
      {modelStatus.error && (
        <button
          type="button"
          onClick={handleRetry}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium bg-blue-500/10 text-blue-400 border border-blue-500/20 hover:bg-blue-500/20 transition-all ml-11"
        >
          <RefreshCw size={12} /> Retry Download
        </button>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Memory Topics Panel
// ---------------------------------------------------------------------------

const TYPE_COLORS: Record<string, string> = {
  user: "text-blue-400 bg-blue-500/10 border-blue-500/20",
  feedback: "text-amber-400 bg-amber-500/10 border-amber-500/20",
  project: "text-blue-400 bg-blue-500/10 border-blue-500/20",
  reference: "text-emerald-400 bg-emerald-500/10 border-emerald-500/20",
};

const CONFIDENCE_COLORS: Record<string, string> = {
  high: "text-emerald-400",
  medium: "text-amber-400",
  low: "text-red-400",
};

function TopicTypeBadge({ type }: { type: string }) {
  const color = TYPE_COLORS[type] || "text-th-text-muted bg-neutral-500/10 border-neutral-500/20";
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-medium border ${color}`}>
      {type || "—"}
    </span>
  );
}

function MemoryTopicsPanel({ memoryEnabled }: { memoryEnabled: boolean }) {
  const [topics, setTopics] = useState<MemoryTopic[]>([]);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<MemoryTopic | null>(null);
  const [editing, setEditing] = useState(false);
  const [editContent, setEditContent] = useState("");
  const [correctionText, setCorrectionText] = useState("");
  const [saving, setSaving] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
  const [topicError, setTopicError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setTopicError(null);
    try {
      const res = await api.listMemoryTopics();
      setTopics(res.topics ?? []);
    } catch (e: any) {
      setTopicError(e?.message ?? "Failed to load topics");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const openTopic = async (t: MemoryTopic) => {
    const full = await api.getMemoryTopic(t.filename);
    setSelected(full);
    setEditing(false);
    setCorrectionText("");
  };

  const saveEdit = async () => {
    if (!selected) return;
    setSaving(true);
    try {
      await api.updateMemoryTopic(selected.filename, editContent);
      await load();
      setEditing(false);
      setSelected(null);
    } catch (e: any) {
      setTopicError(e?.message ?? "Save failed");
    } finally {
      setSaving(false);
    }
  };

  const addCorrection = async () => {
    if (!selected || !correctionText.trim()) return;
    setSaving(true);
    try {
      await api.addMemoryCorrection(selected.filename, correctionText.trim());
      const updated = await api.getMemoryTopic(selected.filename);
      setSelected(updated);
      setCorrectionText("");
      await load();
    } catch (e: any) {
      setTopicError(e?.message ?? "Failed to add correction");
    } finally {
      setSaving(false);
    }
  };

  const deleteTopic = async (filename: string) => {
    setSaving(true);
    try {
      await api.deleteMemoryTopic(filename);
      setSelected(null);
      setDeleteConfirm(null);
      await load();
    } catch (e: any) {
      setTopicError(e?.message ?? "Delete failed");
    } finally {
      setSaving(false);
    }
  };

  if (!memoryEnabled) {
    return (
      <div className="bg-th-card-bg border border-th-card-border rounded-xl p-8 text-center space-y-2">
        <Brain size={28} className="text-th-text-muted mx-auto" />
        <p className="text-sm text-th-text-secondary">Memory consolidation is off</p>
        <p className="text-xs text-th-text-tertiary">Enable Memory in the Status tab to start building topics.</p>
      </div>
    );
  }

  if (selected) {
    return (
      <div className="space-y-4 max-w-3xl">
        <button type="button" onClick={() => { setSelected(null); setEditing(false); }} className="text-th-text-tertiary hover:text-th-text-primary text-xs flex items-center gap-1 transition-colors">
          ← Topics
        </button>
        <div className="bg-th-card-bg border border-th-card-border rounded-xl p-5 space-y-4">
          <div className="flex items-start justify-between gap-3">
            <div className="space-y-1.5">
              <div className="flex items-center gap-2 flex-wrap">
                <h3 className="text-base font-semibold text-th-text-primary">{selected.name}</h3>
                <TopicTypeBadge type={selected.type} />
                {selected.confidence && (
                  <span className={`text-[11px] font-medium ${CONFIDENCE_COLORS[selected.confidence] ?? "text-th-text-muted"}`}>{selected.confidence} confidence</span>
                )}
              </div>
              <p className="text-xs text-th-text-tertiary">{selected.description}</p>
              <div className="flex flex-wrap gap-3 text-[11px] text-th-text-muted">
                {selected.created_at && <span>Created {formatTimeAgo(selected.created_at)}</span>}
                {selected.updated_at && <span>· Updated {formatTimeAgo(selected.updated_at)}</span>}
                {selected.source_sessions.length > 0 && (
                  <span className="flex items-center gap-1">
                    · Sessions: {selected.source_sessions.slice(0, 4).map((s) => (
                      <code key={s} className="bg-th-code-bg px-1 rounded text-[10px]">{s.slice(0, 8)}</code>
                    ))}
                    {selected.source_sessions.length > 4 && <span>+{selected.source_sessions.length - 4}</span>}
                  </span>
                )}
              </div>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              {!editing && (
                <button type="button" onClick={() => { setEditing(true); setEditContent(selected.content ?? ""); }} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-th-inset-bg border border-th-border text-th-text-secondary hover:text-th-text-primary transition-colors">
                  <PenLine size={12} /> Edit
                </button>
              )}
              {deleteConfirm === selected.filename ? (
                <div className="flex items-center gap-1.5">
                  <span className="text-[11px] text-red-400">Delete?</span>
                  <button type="button" onClick={() => deleteTopic(selected.filename)} className="px-2.5 py-1 rounded text-[11px] font-medium bg-red-500/10 text-red-400 border border-red-500/20 hover:bg-red-500/20">Yes</button>
                  <button type="button" onClick={() => setDeleteConfirm(null)} className="px-2.5 py-1 rounded text-[11px] text-th-text-muted hover:text-th-text-primary">No</button>
                </div>
              ) : (
                <button type="button" onClick={() => setDeleteConfirm(selected.filename)} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-red-500/5 border border-red-500/15 text-red-400 hover:bg-red-500/10 transition-colors">
                  <Trash2 size={12} />
                </button>
              )}
            </div>
          </div>

          {editing ? (
            <div className="space-y-3">
              <textarea className="w-full h-64 bg-th-code-bg border border-th-border rounded-lg p-3 text-xs font-mono text-th-text-primary resize-y focus:outline-none focus:border-blue-400" value={editContent} onChange={(e) => setEditContent(e.target.value)} />
              <div className="flex justify-end gap-2">
                <button type="button" onClick={() => setEditing(false)} className="px-3 py-1.5 rounded-lg text-xs text-th-text-tertiary hover:text-th-text-primary">Cancel</button>
                <button type="button" onClick={saveEdit} disabled={saving} className="px-4 py-1.5 rounded-lg text-xs font-medium bg-blue-500/10 text-blue-400 border border-blue-500/20 hover:bg-blue-500/20 disabled:opacity-50">
                  {saving ? "Saving…" : "Save"}
                </button>
              </div>
            </div>
          ) : (
            <pre className="bg-th-code-bg border border-th-border rounded-lg p-3 text-xs font-mono text-th-text-secondary whitespace-pre-wrap overflow-x-auto max-h-72 overflow-y-auto">{selected.content}</pre>
          )}

          <div className="border-t border-th-border pt-4 space-y-3">
            <p className="text-xs font-semibold text-th-text-secondary flex items-center gap-1.5">
              <ShieldAlert size={13} className="text-amber-400" /> Corrections
            </p>
            <p className="text-[11px] text-th-text-muted">Override what OTTO believes. Corrections survive consolidation runs.</p>
            <div className="flex gap-2">
              <input type="text" className="flex-1 bg-th-input-bg border border-th-input-border rounded-lg px-3 py-2 text-xs text-th-text-primary focus:outline-none focus:border-amber-400 placeholder:text-th-text-muted" placeholder="e.g. I prefer TypeScript, not JavaScript — correct this" value={correctionText} onChange={(e) => setCorrectionText(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); addCorrection(); } }} />
              <button type="button" onClick={addCorrection} disabled={!correctionText.trim() || saving} className="px-3 py-2 rounded-lg text-xs font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20 hover:bg-amber-500/20 disabled:opacity-50">Add</button>
            </div>
          </div>
        </div>
        {topicError && <p className="text-xs text-red-400">{topicError}</p>}
      </div>
    );
  }

  return (
    <div className="space-y-3 max-w-3xl">
      <div className="flex items-center justify-between">
        <p className="text-xs text-th-text-tertiary">{topics.length} topic{topics.length !== 1 ? "s" : ""}</p>
        <button type="button" onClick={load} className="text-th-text-tertiary hover:text-th-text-secondary text-xs flex items-center gap-1 transition-colors">
          <RefreshCw size={11} className={loading ? "animate-spin" : ""} /> Refresh
        </button>
      </div>
      {topicError && <p className="text-xs text-red-400">{topicError}</p>}
      {loading && topics.length === 0 && <div className="flex justify-center py-8"><Loader2 className="animate-spin text-th-text-muted" size={20} /></div>}
      {!loading && topics.length === 0 && (
        <div className="bg-th-card-bg border border-th-card-border rounded-xl p-8 text-center">
          <BookOpen size={28} className="text-th-text-muted mx-auto mb-3" />
          <p className="text-sm text-th-text-secondary">No memory topics yet</p>
          <p className="text-xs text-th-text-tertiary mt-1">Run a consolidation to distill sessions into memory.</p>
        </div>
      )}
      <div className="space-y-2">
        {topics.map((t) => (
          <div
            key={t.filename}
            role="button"
            tabIndex={0}
            onClick={() => deleteConfirm !== t.filename && openTopic(t)}
            onKeyDown={(e) => e.key === "Enter" && deleteConfirm !== t.filename && openTopic(t)}
            className="w-full text-left bg-th-card-bg border border-th-card-border rounded-xl p-4 hover:border-blue-500/30 hover:bg-blue-500/5 transition-all group cursor-pointer"
          >
            <div className="flex items-start justify-between gap-3">
              <div className="space-y-1.5 min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-sm font-medium text-th-text-primary truncate">{t.name}</span>
                  <TopicTypeBadge type={t.type} />
                  {t.confidence && <span className={`text-[10px] ${CONFIDENCE_COLORS[t.confidence] ?? "text-th-text-muted"}`}>{t.confidence}</span>}
                </div>
                <p className="text-xs text-th-text-tertiary line-clamp-1">{t.description}</p>
                <div className="flex flex-wrap gap-3 text-[11px] text-th-text-muted">
                  {t.updated_at && <span>Updated {formatTimeAgo(t.updated_at)}</span>}
                  {t.source_sessions.length > 0 && <span>{t.source_sessions.length} session{t.source_sessions.length !== 1 ? "s" : ""}</span>}
                  <span>{(t.size_bytes / 1024).toFixed(1)} KB</span>
                </div>
              </div>
              <div className="flex items-center gap-1.5 shrink-0 mt-0.5">
                {deleteConfirm === t.filename ? (
                  <div className="flex items-center gap-1.5" onClick={(e) => e.stopPropagation()}>
                    <span className="text-[11px] text-red-400">Delete?</span>
                    <button
                      type="button"
                      onClick={() => deleteTopic(t.filename)}
                      disabled={saving}
                      className="px-2 py-0.5 rounded text-[11px] font-medium bg-red-500/10 text-red-400 border border-red-500/20 hover:bg-red-500/20 disabled:opacity-50"
                    >
                      Yes
                    </button>
                    <button
                      type="button"
                      onClick={() => setDeleteConfirm(null)}
                      className="px-2 py-0.5 rounded text-[11px] text-th-text-muted hover:text-th-text-primary"
                    >
                      No
                    </button>
                  </div>
                ) : (
                  <>
                    <button
                      type="button"
                      onClick={(e) => { e.stopPropagation(); setDeleteConfirm(t.filename); }}
                      className="opacity-0 group-hover:opacity-100 p-1 rounded text-th-text-muted hover:text-red-400 transition-all"
                      title="Delete topic"
                    >
                      <Trash2 size={13} />
                    </button>
                    <ChevronRight size={14} className="text-th-text-muted group-hover:text-blue-400 transition-colors" />
                  </>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Embedding Index Panel
// ---------------------------------------------------------------------------

function EmbeddingIndexPanel({
  memoryEnabled,
  embeddingEnabled,
  onToggleEmbedding,
}: {
  memoryEnabled: boolean;
  embeddingEnabled: boolean;
  onToggleEmbedding: (v: boolean) => void;
}) {
  const [embStatus, setEmbStatus] = useState<{
    enabled: boolean;
    total_chunks: number;
    sources: { source_path: string; source_type: string; chunk_count: number; indexed_at: number }[];
    error?: string;
  } | null>(null);
  const [embLoading, setEmbLoading] = useState(false);
  const [indexPath, setIndexPath] = useState("");
  const [indexing, setIndexing] = useState(false);
  const [embError, setEmbError] = useState<string | null>(null);
  const [deleteConfirmSource, setDeleteConfirmSource] = useState<string | null>(null);
  const [deletingSource, setDeletingSource] = useState(false);
  const [reindexing, setReindexing] = useState(false);

  const loadStatus = useCallback(async () => {
    if (!memoryEnabled) return;
    setEmbLoading(true);
    try { setEmbStatus(await api.getEmbeddingStatus()); } catch {}
    finally { setEmbLoading(false); }
  }, [memoryEnabled]);

  useEffect(() => { loadStatus(); }, [loadStatus]);

  const handleIndex = async () => {
    if (!indexPath.trim()) return;
    setIndexing(true);
    setEmbError(null);
    try {
      await api.indexPath(indexPath.trim());
      setTimeout(loadStatus, 1500);
    } catch (e: any) {
      setEmbError(e?.message ?? "Indexing failed");
    } finally {
      setIndexing(false);
    }
  };

  if (!memoryEnabled) {
    return (
      <div className="bg-th-card-bg border border-th-card-border rounded-xl p-8 text-center space-y-2">
        <Search size={28} className="text-th-text-muted mx-auto" />
        <p className="text-sm text-th-text-secondary">Memory must be enabled for semantic search</p>
        <p className="text-xs text-th-text-tertiary">Turn on Memory in the Status tab first.</p>
      </div>
    );
  }

  return (
    <div className="space-y-4 max-w-3xl">
      <div className="bg-th-card-bg border border-th-card-border rounded-xl p-5 space-y-4">
        <div className="flex items-center gap-3">
          <Toggle checked={embeddingEnabled} onChange={onToggleEmbedding} />
          <div>
            <p className="text-sm font-medium text-th-text-primary flex items-center gap-1.5">
              <Layers size={14} className="text-blue-400" /> Semantic Search Index
            </p>
            <p className="text-[12px] text-th-text-tertiary mt-0.5">
              {embeddingEnabled
                ? "Memory files, transcripts, and uploads are indexed with all-MiniLM-L6-v2"
                : "Semantic search is off — only BM25 keyword search is available"}
            </p>
          </div>
        </div>

        {embeddingEnabled && embStatus && (
          <div className="border-t border-th-border pt-4 space-y-2">
            <div className="flex items-center gap-4 text-xs text-th-text-tertiary">
              <span><span className="font-medium text-th-text-primary">{embStatus.total_chunks.toLocaleString()}</span> chunks</span>
              <span><span className="font-medium text-th-text-primary">{embStatus.sources.length}</span> sources</span>
              <button type="button" onClick={loadStatus} className="ml-auto text-th-text-muted hover:text-th-text-secondary flex items-center gap-1 transition-colors">
                <RefreshCw size={11} className={embLoading ? "animate-spin" : ""} /> Refresh
              </button>
            </div>
            {embStatus.error && <p className="text-xs text-amber-400 bg-amber-500/5 border border-amber-500/15 rounded-lg px-3 py-2">{embStatus.error}</p>}
            <button
              type="button"
              disabled={reindexing}
              onClick={async () => {
                setReindexing(true);
                try {
                  await api.reindexMemory();
                  await loadStatus();
                } finally {
                  setReindexing(false);
                }
              }}
              className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <RefreshCw size={11} className={reindexing ? "animate-spin" : ""} />
              {reindexing ? "Re-indexing…" : "Re-index memory topics"}
            </button>
          </div>
        )}
      </div>

      {embeddingEnabled && (
        <div className="bg-th-card-bg border border-th-card-border rounded-xl p-5 space-y-3">
          <p className="text-sm font-medium text-th-text-primary">Index a local path</p>
          <p className="text-[12px] text-th-text-tertiary">Point OTTO at a file or folder. Like Spotlight Privacy — only what you add is indexed.</p>
          <div className="flex gap-2">
            <input type="text" className="flex-1 bg-th-input-bg border border-th-input-border rounded-lg px-3 py-2 text-xs font-mono text-th-text-primary focus:outline-none focus:border-blue-400 placeholder:text-th-text-muted" placeholder="/Users/me/Documents/notes" value={indexPath} onChange={(e) => setIndexPath(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") handleIndex(); }} />
            <button type="button" onClick={handleIndex} disabled={!indexPath.trim() || indexing} className="px-4 py-2 rounded-lg text-xs font-medium bg-blue-500/10 text-blue-400 border border-blue-500/20 hover:bg-blue-500/20 disabled:opacity-50">
              {indexing ? <Loader2 size={12} className="animate-spin" /> : "Index"}
            </button>
          </div>
          {embError && <p className="text-xs text-red-400">{embError}</p>}
        </div>
      )}

      {embeddingEnabled && embStatus && embStatus.sources.length > 0 && (
        <div className="bg-th-card-bg border border-th-card-border rounded-xl p-5 space-y-3">
          <p className="text-sm font-medium text-th-text-secondary">Indexed Sources</p>
          <div className="space-y-1.5 max-h-64 overflow-y-auto">
            {embStatus.sources.map((s) => (
              <div key={s.source_path} className="flex items-center gap-2 text-[11px] group">
                <span className={`shrink-0 px-1.5 py-0.5 rounded text-[10px] font-medium ${s.source_type === "memory" ? "bg-blue-500/10 text-blue-400" : s.source_type === "transcript" ? "bg-blue-500/10 text-blue-400" : "bg-emerald-500/10 text-emerald-400"}`}>{s.source_type}</span>
                <span className="flex-1 text-th-text-tertiary truncate font-mono">{s.source_path.split("/").slice(-2).join("/")}</span>
                <span className="text-th-text-muted tabular-nums shrink-0">{s.chunk_count} chunk{s.chunk_count !== 1 ? "s" : ""}</span>
                {deleteConfirmSource === s.source_path ? (
                  <div className="flex items-center gap-1.5 shrink-0">
                    <span className="text-[11px] text-red-400">Remove?</span>
                    <button
                      type="button"
                      disabled={deletingSource}
                      onClick={async () => {
                        setDeletingSource(true);
                        try {
                          await api.removeEmbeddingSource(s.source_path);
                          await loadStatus();
                        } finally {
                          setDeletingSource(false);
                          setDeleteConfirmSource(null);
                        }
                      }}
                      className="px-2 py-0.5 rounded text-[11px] font-medium bg-red-500/10 text-red-400 border border-red-500/20 hover:bg-red-500/20 disabled:opacity-50"
                    >
                      Yes
                    </button>
                    <button
                      type="button"
                      onClick={() => setDeleteConfirmSource(null)}
                      className="px-2 py-0.5 rounded text-[11px] text-th-text-muted hover:text-th-text-primary"
                    >
                      No
                    </button>
                  </div>
                ) : (
                  <button
                    type="button"
                    onClick={() => setDeleteConfirmSource(s.source_path)}
                    className="p-1 rounded text-th-text-muted hover:text-red-400 transition-colors shrink-0"
                    title="Remove from index"
                  >
                    <Trash2 size={11} />
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
