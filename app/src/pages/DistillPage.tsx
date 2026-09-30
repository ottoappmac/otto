import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  AlertTriangle,
  Check,
  CheckCircle2,
  ChevronRight,
  Database,
  FileText,
  Filter,
  Loader2,
  MessageSquare,
  RefreshCw,
  Search,
  Wrench,
  XCircle,
} from "lucide-react";
import { DistillationTrainPanel } from "../components/distillation/DistillationTrainPanel";
import { DistilledModelsTab } from "../components/distillation/DistilledModelsTab";
import { DistillUploadsPanel } from "../components/distillation/DistillUploadsPanel";
import { api } from "../hooks/useApi";
import type {
  AppSettings,
  DistillAdapter,
  DistillationConfig,
  DistillDataset,
  DistillDatasetDetail,
  DistillDatasetSession,
  DistillUpload,
} from "../types";

const AUTO_SAVE_DELAY_MS = 600;
const SUBTABS = ["Dataset", "Train", "Models"] as const;
type SubTab = (typeof SUBTABS)[number];
type FilterId = "all" | "eligible" | "excluded" | "missing_model" | "unanswered" | "eval_fail";

const REASON_LABEL: Record<string, string> = {
  not_completed: "Not completed",
  has_error: "Ended in error",
  too_few_tools: "Too few tools",
  teacher_mismatch: "Wrong teacher",
  eval_failed: "Eval failed",
};

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  const mb = kb / 1024;
  return `${mb.toFixed(1)} MB`;
}

function formatWhen(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function shortModel(id: string): string {
  const raw = (id || "").trim();
  if (!raw) return "unknown";
  return raw.includes("/") ? raw.split("/").pop() || raw : raw;
}

function patchDistilledOntoMlx(
  s: AppSettings,
  adapter: Pick<DistillAdapter, "catalog_id" | "adapter_path">,
): AppSettings {
  return {
    ...s,
    llm: {
      ...s.llm,
      provider: "mlx",
      mlx: {
        ...s.llm.mlx,
        hf_llm_model_id: adapter.catalog_id,
        adapter_path: adapter.adapter_path,
      },
    },
  };
}

function StatCard({
  icon: Icon,
  label,
  value,
  hint,
  warn,
}: {
  icon: typeof FileText;
  label: string;
  value: string | number;
  hint?: string;
  warn?: boolean;
}) {
  return (
    <div className="bg-th-card-bg border border-th-card-border rounded-xl p-4 flex items-center gap-3">
      <div className={`p-2 rounded-lg ${warn ? "bg-amber-500/10" : "bg-th-inset-bg"}`}>
        <Icon size={18} className={warn ? "text-amber-400" : "text-th-text-muted"} />
      </div>
      <div className="min-w-0">
        <p className="text-[11px] text-th-text-tertiary uppercase tracking-wider">{label}</p>
        <p className={`text-lg font-semibold tabular-nums mt-0.5 ${warn ? "text-amber-300" : "text-th-text-primary"}`}>
          {value}
        </p>
        {hint && <p className="text-[10px] text-th-text-muted mt-0.5 truncate">{hint}</p>}
      </div>
    </div>
  );
}

function Toggle({
  label,
  description,
  checked,
  onChange,
}: {
  label: string;
  description?: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex items-start gap-3 cursor-pointer">
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`relative w-10 h-[22px] rounded-full transition-all duration-200 border shrink-0 mt-0.5 ${
          checked ? "bg-violet-600 border-violet-600" : "bg-th-inset-bg border-th-border"
        }`}
      >
        <span
          className={`absolute top-0.5 left-0.5 w-[16px] h-[16px] rounded-full transition-all duration-200 ${
            checked ? "translate-x-[18px] bg-white" : "bg-neutral-400"
          }`}
        />
      </button>
      <div className="min-w-0">
        <span className="text-sm text-th-text-secondary">{label}</span>
        {description && <p className="text-[11px] text-th-text-muted mt-0.5 leading-relaxed">{description}</p>}
      </div>
    </label>
  );
}

function InputField({
  label,
  value,
  onChange,
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}) {
  return (
    <div>
      <label className="block text-sm font-medium text-th-text-tertiary mb-2">{label}</label>
      <input
        className="w-full px-4 py-2.5 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-violet-400 focus:ring-1 focus:ring-violet-300/30 transition-all text-sm"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
      />
    </div>
  );
}

function EvalBadge({ row }: { row: DistillDatasetSession }) {
  if (row.eval_verdict === true) {
    return (
      <span className="text-[10px] font-medium px-1.5 py-0.5 rounded-full bg-emerald-500/15 text-emerald-400 border border-emerald-500/25">
        eval {row.eval_overall_score != null ? row.eval_overall_score.toFixed(2) : "pass"}
      </span>
    );
  }
  if (row.eval_verdict === false) {
    return (
      <span className="text-[10px] font-medium px-1.5 py-0.5 rounded-full bg-red-500/15 text-red-400 border border-red-500/25">
        eval {row.eval_overall_score != null ? row.eval_overall_score.toFixed(2) : "fail"}
      </span>
    );
  }
  return <span className="text-[10px] text-th-text-muted">no eval</span>;
}

export default function DistillPage() {
  const [tab, setTab] = useState<SubTab>("Dataset");
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [saveStatus, setSaveStatus] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [dataset, setDataset] = useState<DistillDataset | null>(null);
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const [datasetLoading, setDatasetLoading] = useState(false);
  const [filter, setFilter] = useState<FilterId>("all");
  const [search, setSearch] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<DistillDatasetDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [engine, setEngine] = useState<"mlx" | "omlx">("mlx");
  const [useNote, setUseNote] = useState<string | null>(null);
  const [dataKind, setDataKind] = useState<"sessions" | "files">("sessions");
  const [selectedUpload, setSelectedUpload] = useState<DistillUpload | null>(null);

  const loadedRef = useRef(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout>>();
  const persistChainRef = useRef(Promise.resolve());
  const persistLatestRef = useRef<AppSettings | null>(null);
  const savedTimerRef = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => {
    api.getSettings()
      .then((s) => {
        setSettings(s);
        loadedRef.current = true;
      })
      .catch((e) => console.warn("Failed to load settings:", e));
  }, []);

  const persistSettings = useCallback(async (next: AppSettings) => {
    persistLatestRef.current = next;
    setSaveStatus("saving");
    persistChainRef.current = persistChainRef.current
      .catch(() => undefined)
      .then(async () => {
        const snap = persistLatestRef.current;
        if (!snap) return;
        persistLatestRef.current = null;
        try {
          await api.updateSettings(snap as unknown as Record<string, unknown>);
          setSaveStatus("saved");
          savedTimerRef.current = setTimeout(() => setSaveStatus("idle"), 2000);
        } catch {
          setSaveStatus("error");
        }
      });
    await persistChainRef.current;
  }, []);

  useEffect(() => {
    if (!loadedRef.current || !settings) return;
    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => persistSettings(settings), AUTO_SAVE_DELAY_MS);
    return () => clearTimeout(debounceRef.current);
  }, [settings, persistSettings]);

  const dist = settings?.distillation;
  const query = useMemo(
    () =>
      dist
        ? {
            teacher_model_id: dist.teacher_model_id,
            student_model_id: dist.student_model_id,
            filter_sessions_by_teacher: dist.filter_sessions_by_teacher,
          }
        : undefined,
    [dist?.teacher_model_id, dist?.student_model_id, dist?.filter_sessions_by_teacher],
  );

  const loadDataset = useCallback(async () => {
    setDatasetLoading(true);
    try {
      const next = await api.getDistillDataset(query);
      setDataset(next);
      setDatasetError(null);
    } catch (e: unknown) {
      setDatasetError(e instanceof Error ? e.message : "Failed to load dataset");
    } finally {
      setDatasetLoading(false);
    }
  }, [query]);

  useEffect(() => {
    if (tab === "Dataset" && dataKind === "sessions") void loadDataset();
  }, [tab, dataKind, loadDataset]);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    let cancelled = false;
    setDetailLoading(true);
    api.getDistillDatasetSession(selectedId, query)
      .then((row) => {
        if (!cancelled) setDetail(row);
      })
      .catch(() => {
        if (!cancelled) setDetail(null);
      })
      .finally(() => {
        if (!cancelled) setDetailLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedId, query]);

  const updateDistillationField = <K extends keyof DistillationConfig>(
    field: K,
    value: DistillationConfig[K],
  ) => {
    setSettings((s) =>
      s ? { ...s, distillation: { ...s.distillation, [field]: value } } : s,
    );
  };

  const activateAdapter = async (adapter: DistillAdapter, nextEngine: "mlx" | "omlx") => {
    if (!settings) return;
    setUseNote(null);
    if (nextEngine === "omlx" && adapter.omlx_model_id) {
      const next = {
        ...settings,
        llm: { ...settings.llm, provider: "omlx" as const },
        omlx: { ...settings.omlx, enabled: true, model_name: adapter.omlx_model_id },
      };
      setSettings(next);
      await persistSettings(next);
      setUseNote(`Turbo will load “${adapter.omlx_model_id}” on the next chat.`);
      return;
    }
    const next = patchDistilledOntoMlx(settings, adapter);
    setSettings(next);
    await persistSettings(next);
    setUseNote(`Standard will load ${adapter.display_name || adapter.catalog_id} on the next chat.`);
  };

  const visible = useMemo(() => {
    const rows = dataset?.sessions ?? [];
    const q = search.trim().toLowerCase();
    return rows.filter((row) => {
      if (filter === "eligible" && !row.eligible) return false;
      if (filter === "excluded" && row.eligible) return false;
      if (filter === "missing_model" && row.model) return false;
      if (filter === "unanswered" && row.n_unanswered <= 0) return false;
      if (filter === "eval_fail" && row.eval_verdict !== false) return false;
      if (!q) return true;
      const hay = `${row.title} ${row.preview} ${row.session_id} ${row.model} ${row.llm_provider} ${row.tools_used.join(" ")}`.toLowerCase();
      return hay.includes(q);
    });
  }, [dataset, filter, search]);

  if (!settings || !dist) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="animate-spin text-th-text-muted" size={24} />
      </div>
    );
  }

  const stats = dataset?.stats;
  const persisted = dataset?.persisted;
  const lastTrain = persisted?.trajectories.exists ? persisted.trajectories : null;

  return (
    <div className="h-full flex flex-col">
      <header className="border-b border-th-border px-6 py-4 flex items-center justify-between shrink-0 bg-th-bg-secondary">
        <div className="min-w-0">
          <h1 className="text-lg font-bold text-th-text-primary">Distill</h1>
          <p className="text-xs text-th-text-tertiary mt-0.5">
            Inspect session traces, then train a student LoRA. Training unloads chat weights first.
          </p>
        </div>
        <div className="flex items-center gap-1.5 text-xs font-medium min-w-[80px] justify-end">
          {saveStatus === "saving" && (
            <>
              <Loader2 size={12} className="animate-spin text-th-text-muted" />
              <span className="text-th-text-muted">Saving…</span>
            </>
          )}
          {saveStatus === "saved" && (
            <>
              <Check size={12} className="text-emerald-400" />
              <span className="text-emerald-400">Saved</span>
            </>
          )}
          {saveStatus === "error" && (
            <>
              <XCircle size={12} className="text-red-400" />
              <span className="text-red-400">Save failed</span>
            </>
          )}
        </div>
      </header>

      <div className="flex-1 overflow-y-auto p-6">
        <div className="flex gap-1 mb-6 bg-th-inset-bg rounded-xl p-1 w-fit border border-th-border">
          {SUBTABS.map((t) => (
            <button
              key={t}
              type="button"
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-150 ${
                tab === t
                  ? "bg-th-tab-active-bg text-th-tab-active-fg shadow-sm"
                  : "text-th-text-tertiary hover:text-th-text-primary hover:bg-th-surface-hover"
              }`}
              onClick={() => setTab(t)}
            >
              {t}
            </button>
          ))}
        </div>

        {tab === "Dataset" && (
          <div className="space-y-5 max-w-6xl">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex gap-1 bg-th-inset-bg rounded-xl p-1 w-fit border border-th-border">
                {(["sessions", "files"] as const).map((id) => (
                  <button
                    key={id}
                    type="button"
                    className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                      dataKind === id
                        ? "bg-th-tab-active-bg text-th-tab-active-fg shadow-sm"
                        : "text-th-text-tertiary hover:text-th-text-primary"
                    }`}
                    onClick={() => setDataKind(id)}
                  >
                    {id === "sessions" ? "Otto sessions" : "Your files"}
                  </button>
                ))}
              </div>
              {dataKind === "sessions" && (
                <button
                  type="button"
                  onClick={() => void loadDataset()}
                  className="shrink-0 text-th-text-tertiary hover:text-th-text-secondary text-xs flex items-center gap-1 transition-colors"
                >
                  <RefreshCw size={11} className={datasetLoading ? "animate-spin" : ""} /> Refresh
                </button>
              )}
            </div>

            {dataKind === "files" && (
              <DistillUploadsPanel
                selectedId={selectedUpload?.id ?? null}
                onSelect={setSelectedUpload}
              />
            )}

            {dataKind === "sessions" && (
            <>
            <p className="text-sm text-th-text-secondary leading-relaxed max-w-2xl">
              Every completed chat can become a training example. Eligible traces need a finished
              session and at least {dist.min_tool_calls} tools. Review rejects before you train.
            </p>

            {lastTrain && (
              <div className="rounded-xl border border-violet-500/25 bg-violet-500/5 px-4 py-3 text-xs text-th-text-secondary">
                Last training snapshot: {lastTrain.rows.toLocaleString()} traces
                {persisted?.sft.exists ? ` · ${persisted.sft.rows.toLocaleString()} SFT rows` : ""}
                {" · "}
                {formatBytes(lastTrain.bytes)}
                {lastTrain.updated_at ? ` · ${formatWhen(lastTrain.updated_at)}` : ""}
              </div>
            )}

            {datasetError && <p className="text-xs text-red-400">{datasetError}</p>}

            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              <StatCard
                icon={CheckCircle2}
                label="Eligible"
                value={stats ? stats.n_eligible : "—"}
                hint={stats ? `${stats.n_sessions} sessions scanned` : undefined}
              />
              <StatCard
                icon={Filter}
                label="Excluded"
                value={stats ? stats.n_rejected : "—"}
                hint="won't enter the next train"
                warn={!!stats && stats.n_rejected > 0 && stats.n_eligible === 0}
              />
              <StatCard
                icon={MessageSquare}
                label="Messages"
                value={stats ? stats.n_messages.toLocaleString() : "—"}
                hint={stats ? `median ${Math.round(stats.median_messages)} / session` : undefined}
              />
              <StatCard
                icon={Wrench}
                label="Median tools"
                value={stats ? Math.round(stats.median_tool_calls) : "—"}
                hint="tool calls per session"
              />
              <StatCard
                icon={AlertTriangle}
                label="Missing model"
                value={stats ? stats.n_missing_model : "—"}
                hint="session meta has no model id"
                warn={!!stats && stats.n_missing_model > 0}
              />
              <StatCard
                icon={XCircle}
                label="Eval failed"
                value={stats ? stats.n_eval_fail : "—"}
                hint={stats ? `${stats.n_eval_none} not scored` : undefined}
                warn={!!stats && stats.n_eval_fail > 0}
              />
              <StatCard
                icon={AlertTriangle}
                label="Unanswered tools"
                value={stats ? stats.n_unanswered : "—"}
                hint="tool call with no result"
                warn={!!stats && stats.n_unanswered > 0}
              />
              <StatCard
                icon={Database}
                label="Last SFT"
                value={persisted?.sft.exists ? persisted.sft.rows.toLocaleString() : "—"}
                hint={persisted?.sft.exists ? formatBytes(persisted.sft.bytes) : "no snapshot yet"}
              />
            </div>

            {stats && Object.keys(stats.reject_histogram).length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {Object.entries(stats.reject_histogram).map(([reason, n]) => (
                  <span
                    key={reason}
                    className="text-[11px] px-2 py-1 rounded-lg bg-th-inset-bg border border-th-border text-th-text-tertiary"
                  >
                    {REASON_LABEL[reason] || reason} · {n}
                  </span>
                ))}
              </div>
            )}

            {stats && Object.keys(stats.tool_histogram).length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {Object.entries(stats.tool_histogram).slice(0, 12).map(([name, n]) => (
                  <span
                    key={name}
                    className="text-[11px] px-2 py-1 rounded-lg bg-violet-500/8 border border-violet-500/20 text-violet-300 font-mono"
                  >
                    {name} · {n}
                  </span>
                ))}
              </div>
            )}

            <div className="flex flex-wrap items-center gap-2">
              {(
                [
                  ["all", "All"],
                  ["eligible", "Eligible"],
                  ["excluded", "Excluded"],
                  ["missing_model", "Missing model"],
                  ["unanswered", "Unanswered"],
                  ["eval_fail", "Eval fail"],
                ] as const
              ).map(([id, label]) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setFilter(id)}
                  className={`px-2.5 py-1 rounded-lg text-[11px] font-medium border transition-colors ${
                    filter === id
                      ? "bg-violet-500/15 text-violet-300 border-violet-500/30"
                      : "bg-th-inset-bg text-th-text-tertiary border-th-border hover:text-th-text-primary"
                  }`}
                >
                  {label}
                </button>
              ))}
              <div className="relative ml-auto min-w-[200px]">
                <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-th-text-muted" />
                <input
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search sessions…"
                  className="w-full pl-7 pr-3 py-1.5 text-[12px] rounded-lg border border-th-border bg-th-surface text-th-text-primary placeholder:text-th-text-muted focus:outline-none focus:ring-1 focus:ring-violet-400/40"
                />
              </div>
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_380px] gap-4 items-start">
              <div className="bg-th-card-bg border border-th-card-border rounded-xl overflow-hidden">
                {datasetLoading && !dataset && (
                  <div className="flex justify-center py-12">
                    <Loader2 className="animate-spin text-th-text-muted" size={20} />
                  </div>
                )}
                {dataset && visible.length === 0 && (
                  <div className="py-10 text-center space-y-1">
                    <p className="text-sm text-th-text-secondary">No sessions match this view.</p>
                    <p className="text-xs text-th-text-muted">
                      Complete a chat with tool use, then refresh.
                    </p>
                  </div>
                )}
                {visible.length > 0 && (
                  <table className="w-full text-left border-collapse">
                    <thead>
                      <tr className="border-b border-th-border/50">
                        <th className="pl-4 pr-2 py-2 text-[10px] font-semibold uppercase tracking-wider text-th-text-muted">Session</th>
                        <th className="px-2 py-2 text-[10px] font-semibold uppercase tracking-wider text-th-text-muted text-right">Msgs</th>
                        <th className="px-2 py-2 text-[10px] font-semibold uppercase tracking-wider text-th-text-muted text-right">Tools</th>
                        <th className="px-2 py-2 text-[10px] font-semibold uppercase tracking-wider text-th-text-muted">Eval</th>
                        <th className="pl-2 pr-4 py-2 text-[10px] font-semibold uppercase tracking-wider text-th-text-muted">Gate</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-th-border/20">
                      {visible.map((row) => {
                        const selected = selectedId === row.session_id;
                        return (
                          <tr
                            key={row.session_id}
                            onClick={() => setSelectedId(row.session_id)}
                            className={`cursor-pointer transition-colors ${
                              selected ? "bg-violet-500/10" : "hover:bg-th-surface-hover/40"
                            }`}
                          >
                            <td className="pl-4 pr-2 py-2.5 min-w-0">
                              <p className="text-[12px] font-medium text-th-text-primary truncate">
                                {row.title || row.preview || row.session_id.slice(0, 8)}
                              </p>
                              <p className="text-[10px] text-th-text-muted truncate mt-0.5">
                                {row.preview && row.title !== row.preview ? row.preview : row.session_id.slice(0, 8)}
                                {row.model ? ` · ${shortModel(row.model)}` : " · no model"}
                                {row.llm_provider ? ` · ${row.llm_provider}` : ""}
                              </p>
                            </td>
                            <td className="px-2 py-2.5 text-right text-[12px] tabular-nums text-th-text-secondary">
                              {row.n_messages}
                            </td>
                            <td className="px-2 py-2.5 text-right text-[12px] tabular-nums text-th-text-secondary">
                              {row.n_tool_calls}
                              {row.n_unanswered > 0 && (
                                <span className="block text-[9px] text-amber-400">{row.n_unanswered} open</span>
                              )}
                            </td>
                            <td className="px-2 py-2.5"><EvalBadge row={row} /></td>
                            <td className="pl-2 pr-4 py-2.5">
                              {row.eligible ? (
                                <span className="text-[10px] font-medium text-emerald-400">in</span>
                              ) : (
                                <span className="text-[10px] text-amber-400">
                                  {REASON_LABEL[row.reject_reasons[0]] || "out"}
                                </span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                )}
                {dataset && (
                  <p className="px-4 py-2 text-[10px] text-th-text-muted border-t border-th-border/40">
                    Showing {visible.length} of {dataset.sessions.length}
                    {dist.filter_sessions_by_teacher ? " · teacher filter on" : ""}
                  </p>
                )}
              </div>

              <SessionDetailPanel
                selectedId={selectedId}
                detail={detail}
                loading={detailLoading}
              />
            </div>
            </>
            )}
          </div>
        )}

        {tab === "Train" && (
          <div className="space-y-5 max-w-2xl">
            <div className="bg-th-card-bg border border-th-card-border rounded-xl p-5 space-y-3">
              <p className="text-sm font-medium text-th-text-primary">Training data</p>
              <div className="flex gap-1 bg-th-inset-bg rounded-xl p-1 w-fit border border-th-border">
                {(["sessions", "files"] as const).map((id) => (
                  <button
                    key={id}
                    type="button"
                    className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                      dataKind === id
                        ? "bg-th-tab-active-bg text-th-tab-active-fg shadow-sm"
                        : "text-th-text-tertiary hover:text-th-text-primary"
                    }`}
                    onClick={() => setDataKind(id)}
                  >
                    {id === "sessions" ? "Otto sessions" : "Your files"}
                  </button>
                ))}
              </div>
              {dataKind === "files" ? (
                <p className="text-xs text-th-text-tertiary">
                  {selectedUpload
                    ? `Using ${selectedUpload.filename} (${selectedUpload.n_valid.toLocaleString()} valid rows). Change it under Dataset → Your files.`
                    : "No file selected. Open Dataset → Your files, upload a JSONL, then Use for train."}
                </p>
              ) : (
                <p className="text-xs text-th-text-tertiary">
                  Uses eligible completed sessions from the Dataset tab.
                </p>
              )}
            </div>
            <div className="bg-th-card-bg border border-th-card-border rounded-xl p-6 space-y-4">
              <div>
                <h2 className="text-base font-semibold text-th-text-primary flex items-center gap-2.5">
                  <span className="w-2 h-2 rounded-full bg-violet-400" />
                  Pair
                </h2>
                <p className="text-xs text-th-text-tertiary mt-2 leading-relaxed">
                  Teacher and student are MLX catalog repo ids. Changing them here does not switch
                  the live chat model. The adapter only loads when Standard is that student.
                </p>
              </div>
              <Toggle
                label="Collect trajectories from completed sessions"
                checked={dist.enabled}
                onChange={(v) => updateDistillationField("enabled", v)}
              />
              <InputField
                label="Teacher model (repo id)"
                value={dist.teacher_model_id}
                onChange={(v) => updateDistillationField("teacher_model_id", v)}
                placeholder="mlx-community/Qwen3-32B-4bit"
              />
              <InputField
                label="Student model (repo id)"
                value={dist.student_model_id}
                onChange={(v) => updateDistillationField("student_model_id", v)}
                placeholder="mlx-community/Qwen3-8B-4bit"
              />
              <Toggle
                label="Only use sessions run with the teacher model"
                description="Leave off if most chats ran on Turbo or another provider — those traces still teach the student."
                checked={dist.filter_sessions_by_teacher}
                onChange={(v) => updateDistillationField("filter_sessions_by_teacher", v)}
              />
            </div>

            <div className="bg-th-card-bg border border-th-card-border rounded-xl p-6">
              <DistillationTrainPanel
                teacherModelId={dist.teacher_model_id}
                studentModelId={dist.student_model_id}
                filterByTeacher={dist.filter_sessions_by_teacher}
                currentAdapterPath={settings.llm.mlx.adapter_path ?? ""}
                currentModelId={settings.llm.mlx.hf_llm_model_id}
                source={dataKind === "files" ? "upload" : "sessions"}
                upload={selectedUpload
                  ? { id: selectedUpload.id, filename: selectedUpload.filename, n_valid: selectedUpload.n_valid }
                  : null}
                onUseAdapter={(path, catalogId) => {
                  void activateAdapter(
                    {
                      catalog_id: catalogId || "",
                      adapter_path: path,
                      display_name: catalogId || path,
                      base_repo_id: dist.student_model_id,
                      teacher_model_id: dist.teacher_model_id,
                      dataset_sha: "",
                      trained_at: "",
                      size_mb: 0,
                    },
                    "mlx",
                  );
                }}
              />
              {useNote && <p className="text-xs text-emerald-400 mt-3">{useNote}</p>}
            </div>
          </div>
        )}

        {tab === "Models" && (
          <div className="space-y-5 max-w-2xl">
            <p className="text-sm text-th-text-secondary leading-relaxed">
              Trained adapters appear here and in the Standard / Turbo pickers. Standard attaches the
              LoRA. Turbo needs a fused copy first.
            </p>
            <div className="flex gap-1 bg-th-inset-bg rounded-xl p-1 w-fit border border-th-border">
              {(["mlx", "omlx"] as const).map((id) => (
                <button
                  key={id}
                  type="button"
                  className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                    engine === id
                      ? "bg-th-tab-active-bg text-th-tab-active-fg shadow-sm"
                      : "text-th-text-tertiary hover:text-th-text-primary"
                  }`}
                  onClick={() => setEngine(id)}
                >
                  {id === "mlx" ? "Standard" : "Turbo"}
                </button>
              ))}
            </div>
            <div className="bg-th-card-bg border border-th-card-border rounded-xl p-4">
              <DistilledModelsTab
                engine={engine}
                selectedCatalogId={
                  engine === "omlx"
                    ? settings.omlx?.model_name
                    : settings.llm.mlx.hf_llm_model_id
                }
                selectedAdapterPath={engine === "mlx" ? settings.llm.mlx.adapter_path : undefined}
                className="space-y-2"
                canEditPurpose
                onUse={(adapter) => void activateAdapter(adapter, engine)}
              />
              {useNote && <p className="text-xs text-emerald-400 mt-3">{useNote}</p>}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function SessionDetailPanel({
  selectedId,
  detail,
  loading,
}: {
  selectedId: string | null;
  detail: DistillDatasetDetail | null;
  loading: boolean;
}) {
  if (!selectedId) {
    return (
      <div className="bg-th-card-bg border border-th-card-border rounded-xl p-8 text-center">
        <FileText size={22} className="text-th-text-muted mx-auto mb-2" />
        <p className="text-sm text-th-text-secondary">Select a session</p>
        <p className="text-xs text-th-text-muted mt-1">Messages, tools, and the quality gate.</p>
      </div>
    );
  }
  if (loading && !detail) {
    return (
      <div className="bg-th-card-bg border border-th-card-border rounded-xl p-8 flex justify-center">
        <Loader2 className="animate-spin text-th-text-muted" size={20} />
      </div>
    );
  }
  if (!detail) {
    return (
      <div className="bg-th-card-bg border border-th-card-border rounded-xl p-6 text-sm text-th-text-tertiary">
        Could not load this session.
      </div>
    );
  }
  return (
    <div className="bg-th-card-bg border border-th-card-border rounded-xl p-4 space-y-3 xl:sticky xl:top-0">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="text-sm font-medium text-th-text-primary truncate">{detail.title}</p>
          <p className="text-[10px] text-th-text-muted font-mono truncate">{detail.session_id}</p>
        </div>
        <Link
          to={`/chat/${detail.session_id}`}
          className="shrink-0 text-[11px] text-violet-400 hover:text-violet-300 inline-flex items-center gap-0.5"
        >
          Open chat <ChevronRight size={12} />
        </Link>
      </div>
      <div className="flex flex-wrap gap-1.5">
        <span className={`text-[10px] px-1.5 py-0.5 rounded-full border ${
          detail.eligible
            ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/25"
            : "bg-amber-500/10 text-amber-400 border-amber-500/25"
        }`}>
          {detail.eligible ? "Eligible" : (REASON_LABEL[detail.reject_reasons[0]] || "Excluded")}
        </span>
        <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-th-inset-bg border border-th-border text-th-text-tertiary">
          {detail.n_messages} msgs
        </span>
        <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-th-inset-bg border border-th-border text-th-text-tertiary">
          {detail.n_tool_calls} tools
        </span>
        {detail.n_unanswered > 0 && (
          <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/25">
            {detail.n_unanswered} unanswered
          </span>
        )}
        <EvalBadge row={detail} />
      </div>
      <p className="text-[11px] text-th-text-muted">
        {detail.model ? shortModel(detail.model) : "No model id"}
        {detail.llm_provider ? ` · ${detail.llm_provider}` : ""}
      </p>
      {detail.tools_used.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {detail.tools_used.map((t) => (
            <span key={t} className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-th-inset-bg text-th-text-tertiary">
              {t}
            </span>
          ))}
        </div>
      )}
      <div className="max-h-[28rem] overflow-y-auto space-y-2 pr-1">
        {detail.messages.map((msg, i) => (
          <div key={`${msg.role}-${i}`} className="rounded-lg border border-th-border/50 bg-th-inset-bg/50 px-2.5 py-2">
            <div className="flex items-center gap-2 mb-1">
              <span className={`text-[10px] font-semibold uppercase tracking-wider ${
                msg.role === "user"
                  ? "text-blue-400"
                  : msg.role === "assistant"
                    ? "text-violet-300"
                    : "text-amber-400"
              }`}>
                {msg.role}
                {msg.name ? ` · ${msg.name}` : ""}
              </span>
              {msg.tool_calls?.map((tc) => (
                <span key={tc.id || tc.name} className="text-[9px] font-mono text-th-text-muted">
                  {tc.name}
                </span>
              ))}
            </div>
            <p className="text-[11px] text-th-text-secondary whitespace-pre-wrap break-words leading-relaxed line-clamp-8">
              {msg.content || (msg.tool_calls?.length ? "(tool call)" : "—")}
            </p>
            {msg.truncated && <p className="text-[10px] text-th-text-muted mt-1">Truncated</p>}
          </div>
        ))}
      </div>
    </div>
  );
}
