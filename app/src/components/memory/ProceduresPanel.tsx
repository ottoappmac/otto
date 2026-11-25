import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  ArrowLeft,
  Ban,
  BookOpen,
  CheckCircle,
  Clock,
  Layers,
  Lightbulb,
  ListOrdered,
  Loader2,
  RefreshCw,
  Search,
  ThumbsDown,
  ThumbsUp,
  Trash2,
  Workflow,
} from "lucide-react";
import { api } from "../../hooks/useApi";
import type {
  AppSettings,
  ProceduralMemoryConfig,
  ProcedureEpisode,
  ProcedureEpisodeDetail,
  ProcedureEpisodeStatus,
  ProcedureItem,
} from "../../types";

// Mirrors backend/config.py ProceduralMemoryConfig, used when an older config has no section yet.
export const PROCEDURAL_MEMORY_DEFAULTS: ProceduralMemoryConfig = {
  enabled: false,
  serve: true,
  similarity_floor: 0.45,
  max_tokens: 800,
  judge_success_trusted: false,
  judge_failure_trusted: false,
  segments_from_failed_runs: true,
  enrich_with_llm: true,
  max_saved_runs: 500,
};

const MAX_RUNS_CEILING = 100000;
const FLOOR_MIN = 0.2;
const FLOOR_MAX = 0.9;

type Filter = "all" | ProcedureEpisodeStatus;

const FILTERS: { id: Filter; label: string }[] = [
  { id: "all", label: "All" },
  { id: "pending", label: "Needs review" },
  { id: "admitted", label: "In use" },
  { id: "excluded", label: "Excluded" },
];

const STATUS_META: Record<ProcedureEpisodeStatus, { label: string; hint: string; color: string; icon: typeof Clock }> = {
  pending: {
    label: "Needs review",
    hint: "Saved, but not used yet. Worked adds it to memory; Didn't work keeps it out.",
    color: "text-amber-400 bg-amber-500/10 border-amber-500/20",
    icon: Clock,
  },
  admitted: {
    label: "In use",
    hint: "The macOS desktop agent can recall what it learned from this run.",
    color: "text-emerald-400 bg-emerald-500/10 border-emerald-500/20",
    icon: CheckCircle,
  },
  retired: {
    label: "Retired",
    hint: "No longer offered to the agent.",
    color: "text-th-text-muted bg-neutral-500/10 border-neutral-500/20",
    icon: Ban,
  },
  excluded: {
    label: "Excluded",
    hint: "Marked as not working, so nothing from it is used. Mark it Worked to use it after all.",
    color: "text-th-text-muted bg-neutral-500/10 border-neutral-500/20",
    icon: Ban,
  },
};

const KIND_META: Record<string, { label: string; icon: typeof Workflow }> = {
  recipe: { label: "Recipes", icon: ListOrdered },
  fact: { label: "App facts", icon: Lightbulb },
  lesson: { label: "Lessons", icon: BookOpen },
};

function timeAgo(epochSecs: number | null | undefined): string {
  if (!epochSecs) return "";
  const mins = Math.floor((Date.now() - epochSecs * 1000) / 60000);
  if (mins < 1) return "Just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 30) return `${days}d ago`;
  return new Date(epochSecs * 1000).toLocaleDateString();
}

function fullDate(epochSecs: number | null | undefined): string {
  return epochSecs ? new Date(epochSecs * 1000).toLocaleString() : "";
}

function Switch({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={() => onChange(!checked)}
      className={`relative w-11 h-6 shrink-0 rounded-full transition-all duration-200 border-0 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 ${
        checked ? "bg-blue-500" : "bg-neutral-600/50"
      }`}
    >
      <span
        className={`absolute top-0.5 left-0.5 w-[18px] h-[18px] rounded-full transition-all duration-200 shadow ${
          checked ? "translate-x-[20px] bg-white" : "translate-x-0 bg-neutral-400"
        }`}
      />
    </button>
  );
}

function StatusBadge({ status }: { status: ProcedureEpisodeStatus }) {
  const m = STATUS_META[status] ?? STATUS_META.pending;
  const Icon = m.icon;
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-2xs font-medium border whitespace-nowrap ${m.color}`}>
      <Icon size={11} />
      {m.label}
    </span>
  );
}

function Stat({ label, value, tone }: { label: string; value: number; tone?: string }) {
  return (
    <div className="bg-th-card-bg border border-th-card-border rounded-xl p-3.5">
      <p className="text-2xs text-th-text-tertiary uppercase tracking-wider">{label}</p>
      <p className={`text-lg font-semibold mt-0.5 ${tone ?? "text-th-text-primary"}`}>{value}</p>
    </div>
  );
}

function AppChips({ apps }: { apps: string[] }) {
  if (!apps?.length) return null;
  return (
    <span className="flex flex-wrap gap-1">
      {apps.slice(0, 3).map((a) => (
        <span key={a} className="px-1.5 py-0.5 rounded bg-th-inset-bg border border-th-border text-2xs text-th-text-secondary">{a}</span>
      ))}
      {apps.length > 3 && <span className="text-2xs text-th-text-muted">+{apps.length - 3}</span>}
    </span>
  );
}

function asStrings(v: unknown): string[] {
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === "string" && x.trim() !== "") : [];
}

function ItemCard({ item }: { item: ProcedureItem }) {
  const b = item.body ?? {};
  const lines = asStrings(b.lines);
  const outline = asStrings(b.outline);
  const text = typeof b.text === "string" ? b.text : "";
  const sentence = typeof b.sentence === "string" ? b.sentence : "";
  const content = typeof b.content === "string" ? b.content : "";
  const evidence = typeof b.evidence === "string" ? b.evidence : "";
  return (
    <div className="bg-th-inset-bg border border-th-border rounded-lg p-3 space-y-2">
      <div className="flex items-start justify-between gap-2">
        <p className="text-xs font-medium text-th-text-primary leading-snug">{item.intent || "(no description)"}</p>
        {item.from_outcome === "failure" && (
          <span className="shrink-0 text-2xs text-amber-400" title="Learned from a run that failed overall; only the confirmed part is kept">
            from a failed run
          </span>
        )}
      </div>
      {outline.length > 0 && (
        <ol className="list-decimal list-inside text-2xs text-th-text-secondary space-y-0.5">
          {outline.map((o, i) => <li key={i}>{o}</li>)}
        </ol>
      )}
      {lines.length > 0 && (
        <pre className="bg-th-code-bg border border-th-border rounded p-2 text-2xs font-mono text-th-text-secondary whitespace-pre-wrap overflow-x-auto max-h-48 overflow-y-auto">
          {lines.join("\n")}
        </pre>
      )}
      {text && <p className="text-xs text-th-text-secondary">{text}</p>}
      {(sentence || content) && <p className="text-xs text-th-text-secondary">{[sentence, content].filter(Boolean).join(" ")}</p>}
      {evidence && <p className="text-2xs text-th-text-muted">Checked by: {evidence}</p>}
      {b.has_script === true && <p className="text-2xs text-th-text-muted">Includes a script</p>}
    </div>
  );
}

function EpisodeDetail({
  id,
  memoryOn,
  onBack,
  onChanged,
}: {
  id: string;
  memoryOn: boolean;
  onBack: () => void;
  onChanged: () => void;
}) {
  const [detail, setDetail] = useState<ProcedureEpisodeDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);

  const load = useCallback(async () => {
    setError(null);
    try {
      setDetail(await api.getProcedureEpisode(id));
    } catch (e: any) {
      setError(e?.message || "Could not load this run");
    }
  }, [id]);

  useEffect(() => { void load(); }, [load]);

  const act = async (key: string, fn: () => Promise<unknown>, after: "reload" | "back" = "reload") => {
    setBusy(key);
    setError(null);
    try {
      await fn();
      onChanged();
      if (after === "back") onBack();
      else await load();
    } catch (e: any) {
      setError(e?.message || "That did not work");
    } finally {
      setBusy(null);
    }
  };

  const back = (
    <button type="button" onClick={onBack} className="text-th-text-tertiary hover:text-th-text-primary text-xs flex items-center gap-1 transition-colors">
      <ArrowLeft size={12} /> Saved runs
    </button>
  );

  if (!detail) {
    return (
      <div className="space-y-4">
        {back}
        {error ? (
          <p className="text-xs text-red-400 flex items-center gap-1.5"><AlertCircle size={13} />{error}</p>
        ) : (
          <div className="flex justify-center py-12"><Loader2 className="animate-spin text-th-text-muted" size={20} /></div>
        )}
      </div>
    );
  }

  const ep = detail.episode;
  const groups = Object.entries(
    detail.items.reduce<Record<string, ProcedureItem[]>>((acc, it) => {
      (acc[it.kind] ||= []).push(it);
      return acc;
    }, {}),
  ).sort(([a], [b]) => ["recipe", "fact", "lesson"].indexOf(a) - ["recipe", "fact", "lesson"].indexOf(b));
  const canLabel = memoryOn;
  const labelHint = memoryOn ? undefined : "Turn on procedural memory to review runs";

  return (
    <div className="space-y-4">
      {back}
      <div className="bg-th-card-bg border border-th-card-border rounded-xl p-5 space-y-4">
        <div className="space-y-2">
          <div className="flex items-center gap-2 flex-wrap">
            <StatusBadge status={ep.status} />
            <AppChips apps={ep.apps} />
          </div>
          <h3 className="text-sm font-semibold text-th-text-primary leading-snug whitespace-pre-wrap break-words">
            {ep.task_text || ep.intent || "(no task text)"}
          </h3>
          <p className="text-2xs text-th-text-muted flex flex-wrap gap-x-3 gap-y-1">
            <span title={fullDate(ep.created_at)}>Saved {timeAgo(ep.created_at)}</span>
            <span>{ep.agent}</span>
            {ep.model && <span>{ep.model}</span>}
            {ep.self_report && ep.self_report !== "unknown" && <span>Agent reported: {ep.self_report}</span>}
            {ep.label !== "unknown" && <span>Marked {ep.label === "success" ? "worked" : "didn't work"}{ep.label_source ? ` (${ep.label_source})` : ""}</span>}
          </p>
          <p className="text-xs text-th-text-tertiary">{STATUS_META[ep.status]?.hint}</p>
        </div>

        <div className="flex flex-wrap items-center gap-2 border-t border-th-border pt-4">
          <span className="text-xs text-th-text-secondary mr-1">Did this run do the task?</span>
          <button
            type="button"
            disabled={!canLabel || busy !== null}
            title={labelHint}
            onClick={() => act("success", () => api.answerProcedureEpisode(ep.id, true))}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${
              ep.label === "success" ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/30" : "bg-emerald-500/5 text-emerald-400 border-emerald-500/20 hover:bg-emerald-500/15"
            }`}
          >
            {busy === "success" ? <Loader2 size={12} className="animate-spin" /> : <ThumbsUp size={12} />} Worked
          </button>
          <button
            type="button"
            disabled={!canLabel || busy !== null}
            title={labelHint}
            onClick={() => act("failure", () => api.answerProcedureEpisode(ep.id, false))}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${
              ep.label === "failure" ? "bg-amber-500/20 text-amber-300 border-amber-500/30" : "bg-amber-500/5 text-amber-400 border-amber-500/20 hover:bg-amber-500/15"
            }`}
          >
            {busy === "failure" ? <Loader2 size={12} className="animate-spin" /> : <ThumbsDown size={12} />} Didn't work
          </button>
          <span className="flex-1" />
          {confirmDelete ? (
            <span className="flex items-center gap-1.5">
              <span className="text-2xs text-red-400">Delete permanently?</span>
              <button
                type="button"
                disabled={busy !== null}
                onClick={() => act("delete", () => api.deleteProcedureEpisode(ep.id), "back")}
                className="px-2.5 py-1 rounded text-2xs font-medium bg-red-500/10 text-red-400 border border-red-500/20 hover:bg-red-500/20"
              >
                Delete
              </button>
              <button type="button" onClick={() => setConfirmDelete(false)} className="px-2.5 py-1 rounded text-2xs text-th-text-muted hover:text-th-text-primary">
                Cancel
              </button>
            </span>
          ) : (
            <button
              type="button"
              onClick={() => setConfirmDelete(true)}
              aria-label="Delete this run"
              title="Delete this run"
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-red-500/5 border border-red-500/15 text-red-400 hover:bg-red-500/10 transition-colors"
            >
              <Trash2 size={12} />
            </button>
          )}
        </div>
        {error && <p className="text-xs text-red-400 flex items-center gap-1.5"><AlertCircle size={13} />{error}</p>}
      </div>

      <div className="space-y-3">
        <h4 className="text-xs font-semibold text-th-text-secondary flex items-center gap-1.5">
          <Layers size={13} /> What the agent learned
        </h4>
        {groups.length === 0 ? (
          <p className="text-xs text-th-text-tertiary bg-th-card-bg border border-th-card-border rounded-xl p-4">
            {ep.status === "pending"
              ? "Nothing yet. Steps, app facts and lessons are pulled from the run once you mark whether it worked."
              : "Nothing was kept from this run."}
          </p>
        ) : (
          groups.map(([kind, items]) => {
            const m = KIND_META[kind] ?? { label: kind, icon: Workflow };
            const Icon = m.icon;
            return (
              <section key={kind} className="space-y-2">
                <p className="text-2xs uppercase tracking-wider text-th-text-tertiary flex items-center gap-1.5">
                  <Icon size={11} /> {m.label} <span className="text-th-text-muted">({items.length})</span>
                </p>
                {items.map((it) => <ItemCard key={it.id} item={it} />)}
              </section>
            );
          })
        )}
      </div>

      {detail.labels.length > 0 && (
        <details className="bg-th-card-bg border border-th-card-border rounded-xl p-4 text-xs">
          <summary className="cursor-pointer text-th-text-secondary font-medium">Review history ({detail.labels.length})</summary>
          <ul className="mt-2 space-y-1 text-2xs text-th-text-tertiary">
            {detail.labels.map((l, i) => (
              <li key={i}>
                <span title={fullDate(l.at)}>{timeAgo(l.at)}</span> · {l.source}: {l.value === "success" ? "worked" : l.value === "failure" ? "didn't work" : l.value}
                {l.note ? ` — ${l.note}` : ""}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

export function ProceduresPanel({
  settings,
  setSettings,
}: {
  settings: AppSettings;
  setSettings: React.Dispatch<React.SetStateAction<AppSettings | null>>;
}) {
  const cfg = { ...PROCEDURAL_MEMORY_DEFAULTS, ...(settings.procedural_memory ?? {}) };
  const [episodes, setEpisodes] = useState<ProcedureEpisode[] | null>(null);
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState<Filter>("all");
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<string | null>(null);

  const patchCfg = (patch: Partial<ProceduralMemoryConfig>) =>
    setSettings((prev) => prev ? { ...prev, procedural_memory: { ...PROCEDURAL_MEMORY_DEFAULTS, ...(prev.procedural_memory ?? {}), ...patch } } : prev);
  const setEnabled = (v: boolean) => patchCfg({ enabled: v });

  // The limit is edited as text and saved on blur / Enter, so typing "250" does not save 2, then 25.
  const [limitDraft, setLimitDraft] = useState(String(cfg.max_saved_runs));
  useEffect(() => { setLimitDraft(String(cfg.max_saved_runs)); }, [cfg.max_saved_runs]);
  // The slider moves freely and saves when released, so dragging does not save every step on the way.
  const [floorDraft, setFloorDraft] = useState(cfg.similarity_floor);
  useEffect(() => { setFloorDraft(cfg.similarity_floor); }, [cfg.similarity_floor]);
  const commitFloor = () => {
    const v = Math.round(Math.max(FLOOR_MIN, Math.min(FLOOR_MAX, floorDraft)) * 100) / 100;
    if (v !== cfg.similarity_floor) patchCfg({ similarity_floor: v });
  };
  const commitLimit = () => {
    const n = parseInt(limitDraft, 10);
    const v = Number.isFinite(n) ? Math.max(0, Math.min(MAX_RUNS_CEILING, n)) : cfg.max_saved_runs;
    setLimitDraft(String(v));
    if (v !== cfg.max_saved_runs) patchCfg({ max_saved_runs: v });
  };

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.listProcedureEpisodes();
      setEpisodes(res.episodes);
      setCounts(res.counts ?? {});
    } catch (e: any) {
      setError(e?.message || "Could not load saved runs");
      setEpisodes((prev) => prev ?? []);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void load(); }, [load, cfg.enabled]);

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    return (episodes ?? []).filter((e) => {
      if (filter !== "all" && e.status !== filter && !(filter === "excluded" && e.status === "retired")) return false;
      if (!q) return true;
      return [e.task_text, e.intent, ...(e.apps ?? [])].some((s) => (s || "").toLowerCase().includes(q));
    });
  }, [episodes, filter, query]);

  const total = Object.entries(counts).reduce((n, [k, v]) => (k.startsWith("episodes_") ? n + v : n), 0) || (episodes?.length ?? 0);
  const pending = counts.episodes_pending ?? 0;
  const admitted = counts.episodes_admitted ?? 0;
  const learned = (counts.items_recipe ?? 0) + (counts.items_fact ?? 0) + (counts.items_lesson ?? 0);

  if (selected) {
    return (
      <EpisodeDetail
        id={selected}
        memoryOn={cfg.enabled}
        onBack={() => setSelected(null)}
        onChanged={() => void load()}
      />
    );
  }

  return (
    <div className="space-y-4">
      <div className="bg-th-card-bg border border-th-card-border rounded-xl p-5 space-y-3">
        <div className="flex items-start gap-3">
          <Switch checked={cfg.enabled} onChange={setEnabled} label="Procedural memory" />
          <div className="min-w-0">
            <p className="text-sm font-medium text-th-text-primary">Procedural memory</p>
            <p className="text-[12px] text-th-text-tertiary mt-0.5 leading-relaxed">
              {cfg.enabled
                ? "The macOS desktop agent saves how it did each task and, before a multi-step part of a new task, looks up steps from earlier runs (recall_procedure)."
                : "Off. The macOS desktop agent does not save runs or get the recall_procedure tool."}
            </p>
          </div>
        </div>
        <ul className="text-2xs text-th-text-muted space-y-1 pl-14 list-disc">
          <li>A saved run is only used after you mark it as having worked (or not) below.</li>
          <li>Changes apply to new sessions.</li>
        </ul>

        <div className="border-t border-th-border pt-4 space-y-2">
          <div className="flex items-baseline justify-between gap-4">
            <label htmlFor="procmem-floor" className="text-xs font-medium text-th-text-primary">Similarity threshold</label>
            <output htmlFor="procmem-floor" className="text-xs tabular-nums text-th-text-secondary">{floorDraft.toFixed(2)}</output>
          </div>
          <input
            id="procmem-floor"
            type="range"
            min={FLOOR_MIN}
            max={FLOOR_MAX}
            step={0.05}
            value={floorDraft}
            onChange={(e) => setFloorDraft(Number(e.target.value))}
            onPointerUp={commitFloor}
            onKeyUp={commitFloor}
            onBlur={commitFloor}
            aria-describedby="procmem-floor-help"
            className="w-full accent-blue-500"
          />
          <div className="flex justify-between text-2xs text-th-text-muted" aria-hidden="true">
            <span>More matches</span>
            <span>Closer matches only</span>
          </div>
          <p id="procmem-floor-help" className="text-2xs text-th-text-tertiary leading-relaxed">
            How close a saved procedure must be to the step the agent is about to do before it is offered (cosine
            similarity, 0 to 1). Lower recalls more, including loose matches; higher recalls less but closer.
            App facts and lessons use 85% of this. Default {PROCEDURAL_MEMORY_DEFAULTS.similarity_floor.toFixed(2)}.
            {floorDraft !== PROCEDURAL_MEMORY_DEFAULTS.similarity_floor && (
              <>
                {" "}
                <button
                  type="button"
                  onClick={() => { setFloorDraft(PROCEDURAL_MEMORY_DEFAULTS.similarity_floor); patchCfg({ similarity_floor: PROCEDURAL_MEMORY_DEFAULTS.similarity_floor }); }}
                  className="text-blue-400 hover:underline"
                >
                  Reset
                </button>
              </>
            )}
          </p>
        </div>

        <div className="border-t border-th-border pt-4 flex flex-wrap items-start gap-x-4 gap-y-2">
          <label htmlFor="procmem-max-runs" className="min-w-0 flex-1">
            <span className="block text-xs font-medium text-th-text-primary">Saved runs limit</span>
            <span className="block text-2xs text-th-text-tertiary mt-0.5 leading-relaxed">
              {cfg.max_saved_runs > 0
                ? "When a new run is saved over the limit, the oldest are removed: excluded runs first, then ones awaiting review, and runs in use last."
                : "No limit: runs are kept until you delete them."}
              {" "}0 = no limit. Default {PROCEDURAL_MEMORY_DEFAULTS.max_saved_runs}.
            </span>
          </label>
          <input
            id="procmem-max-runs"
            type="number"
            inputMode="numeric"
            min={0}
            max={MAX_RUNS_CEILING}
            step={50}
            value={limitDraft}
            onChange={(e) => setLimitDraft(e.target.value)}
            onBlur={commitLimit}
            onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); }}
            className="w-28 bg-th-input-bg border border-th-input-border rounded-lg px-3 py-1.5 text-xs text-th-text-primary text-right tabular-nums focus:outline-none focus:border-blue-400"
          />
          {cfg.max_saved_runs > 0 && total > cfg.max_saved_runs && (
            <p className="basis-full text-2xs text-amber-400 flex items-center gap-1.5">
              <AlertCircle size={12} />
              {total - cfg.max_saved_runs} saved run(s) over the limit. The oldest are removed when the next run is saved.
            </p>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <Stat label="Saved runs" value={total} />
        <Stat label="Needs review" value={pending} tone={pending ? "text-amber-400" : undefined} />
        <Stat label="In use" value={admitted} tone={admitted ? "text-emerald-400" : undefined} />
        <Stat label="Learned items" value={learned} />
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <div className="flex gap-1 bg-th-inset-bg rounded-xl p-1 border border-th-border" role="tablist" aria-label="Filter saved runs">
          {FILTERS.map((f) => (
            <button
              key={f.id}
              type="button"
              role="tab"
              aria-selected={filter === f.id}
              onClick={() => setFilter(f.id)}
              className={`px-3 py-1 rounded-lg text-xs font-medium transition-all ${
                filter === f.id ? "bg-th-tab-active-bg text-th-tab-active-fg shadow-sm" : "text-th-text-tertiary hover:text-th-text-primary"
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>
        <label className="relative flex-1 min-w-[160px]">
          <Search size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-th-text-muted" />
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search tasks or apps"
            aria-label="Search saved runs"
            className="w-full bg-th-input-bg border border-th-input-border rounded-lg pl-8 pr-3 py-1.5 text-xs text-th-text-primary focus:outline-none focus:border-blue-400 placeholder:text-th-text-muted"
          />
        </label>
        <button
          type="button"
          onClick={() => void load()}
          aria-label="Refresh"
          title="Refresh"
          className="p-2 rounded-lg bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary"
        >
          <RefreshCw size={13} className={loading ? "animate-spin" : ""} />
        </button>
      </div>

      {error && <p className="text-xs text-red-400 flex items-center gap-1.5"><AlertCircle size={13} />{error}</p>}

      {episodes === null ? (
        <div className="flex justify-center py-12"><Loader2 className="animate-spin text-th-text-muted" size={20} /></div>
      ) : total === 0 ? (
        <div className="bg-th-card-bg border border-th-card-border rounded-xl p-8 text-center space-y-2">
          <Workflow size={26} className="text-th-text-muted mx-auto" />
          <p className="text-sm text-th-text-secondary">No saved runs yet</p>
          <p className="text-xs text-th-text-tertiary">
            {cfg.enabled
              ? "When the macOS desktop agent finishes a task, the run shows up here for review."
              : "Turn on procedural memory, then runs of the macOS desktop agent show up here."}
          </p>
        </div>
      ) : visible.length === 0 ? (
        <p className="text-xs text-th-text-tertiary text-center py-8">No runs match.</p>
      ) : (
        <ul className="bg-th-card-bg border border-th-card-border rounded-xl divide-y divide-th-border overflow-hidden">
          {visible.map((e) => (
            <li key={e.id}>
              <button
                type="button"
                onClick={() => setSelected(e.id)}
                className="w-full text-left px-4 py-3 hover:bg-th-surface-hover focus:outline-none focus-visible:bg-th-surface-hover transition-colors flex items-start gap-3"
              >
                <div className="min-w-0 flex-1 space-y-1.5">
                  <p className="text-xs text-th-text-primary line-clamp-2 break-words">{e.task_text || e.intent || "(no task text)"}</p>
                  <div className="flex flex-wrap items-center gap-2">
                    <AppChips apps={e.apps} />
                    <span className="text-2xs text-th-text-muted" title={fullDate(e.created_at)}>{timeAgo(e.created_at)}</span>
                  </div>
                </div>
                <StatusBadge status={e.status} />
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
