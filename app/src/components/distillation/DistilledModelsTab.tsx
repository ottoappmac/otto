import { useCallback, useEffect, useMemo, useState } from "react";
import { CheckCircle2, Loader2, Pencil, RefreshCw } from "lucide-react";
import { api } from "../../hooks/useApi";
import type { DistillAdapter } from "../../types";

export type DistillPickerEngine = "mlx" | "omlx" | "exo";

function studentShort(repoId: string): string {
  const raw = (repoId || "").trim().replace(/\/+$/, "");
  if (!raw) return "student";
  return raw.includes("/") ? raw.split("/").pop() || raw : raw;
}

function formatTrainedAt(iso: string): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

const ENGINE_NOTE: Record<DistillPickerEngine, string> = {
  mlx: "LoRA adapters trained from your sessions. Use attaches the adapter to the student via mlx_lm and makes it the live Standard model.",
  omlx: "oMLX cannot load a LoRA sidecar. Use fuses the adapter into a standalone MLX model (same size as the student, extra disk) and loads it into Turbo.",
  exo: "Distilled LoRAs are mlx_lm adapters. Use switches Otto to Standard (in-process MLX) and attaches this LoRA. The cluster cannot load adapters.",
};

export function DistilledModelsTab({
  selectedCatalogId,
  selectedAdapterPath,
  onUse,
  engine,
  className = "p-3 space-y-2",
  canUse = true,
  canEditPurpose = false,
}: {
  selectedCatalogId?: string;
  selectedAdapterPath?: string;
  onUse: (adapter: DistillAdapter) => void;
  engine: DistillPickerEngine;
  className?: string;
  canUse?: boolean;
  canEditPurpose?: boolean;
}) {
  const [adapters, setAdapters] = useState<DistillAdapter[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [fuseErr, setFuseErr] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [purposeDraft, setPurposeDraft] = useState("");
  const [savingPurpose, setSavingPurpose] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.listDistillAdapters();
      setAdapters(res.adapters ?? []);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setAdapters([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const visible = useMemo(() => {
    const s = search.trim().toLowerCase();
    const rows = adapters ?? [];
    if (!s) return rows;
    return rows.filter((a) => {
      const hay = `${a.display_name} ${a.catalog_id} ${a.base_repo_id} ${a.teacher_model_id} ${a.purpose ?? ""} ${a.kind ?? ""}`.toLowerCase();
      return hay.includes(s);
    });
  }, [adapters, search]);

  const handleUse = async (adapter: DistillAdapter) => {
    if (engine !== "omlx") {
      onUse(adapter);
      return;
    }
    setFuseErr(null);
    setBusyId(adapter.catalog_id);
    try {
      let next = adapter;
      if (!adapter.omlx_model_id) {
        const started = await api.startDistillFuse(adapter.catalog_id);
        let st = started;
        const deadline = Date.now() + 600_000;
        while (st.state === "running" || st.state === "queued") {
          if (Date.now() > deadline) throw new Error("Fusing the distilled model timed out.");
          await new Promise((r) => setTimeout(r, 1_500));
          st = await api.getDistillStatus();
        }
        if (st.state !== "success" || !st.omlx_model_id) {
          throw new Error(st.error || "Fuse failed — Turbo still cannot load this LoRA.");
        }
        next = {
          ...adapter,
          fused_path: st.fused_path || adapter.fused_path,
          omlx_model_id: st.omlx_model_id,
        };
        void refresh();
      }
      onUse(next);
    } catch (e) {
      setFuseErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusyId(null);
    }
  };

  const startEditPurpose = (adapter: DistillAdapter) => {
    setEditingId(adapter.catalog_id);
    setPurposeDraft(adapter.purpose ?? "");
  };

  const savePurpose = async (catalogId: string) => {
    setSavingPurpose(true);
    try {
      const next = await api.setDistillPurpose(catalogId, purposeDraft.trim());
      setAdapters((prev) =>
        (prev ?? []).map((row) =>
          row.catalog_id === catalogId ? { ...row, purpose: next.purpose ?? purposeDraft.trim() } : row,
        ),
      );
      setEditingId(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSavingPurpose(false);
    }
  };

  return (
    <div className={className}>
      <p className="text-[11px] text-th-text-secondary leading-relaxed">{ENGINE_NOTE[engine]}</p>
      {fuseErr && <p className="text-[10px] text-red-400">{fuseErr}</p>}
      <div className="relative">
        <input
          type="text"
          placeholder="Search distilled adapters…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full px-2.5 py-1.5 text-[11px] rounded-lg border border-th-border bg-th-surface text-th-text-primary placeholder:text-th-text-muted focus:outline-none focus:ring-1 focus:ring-th-tab-active-bg/40 focus:border-th-tab-active-bg/50"
        />
      </div>
      <div className="space-y-2 max-h-72 overflow-y-auto">
        {loading && (
          <div className="flex items-center gap-2 text-[11px] text-th-text-muted py-4 justify-center">
            <Loader2 size={12} className="animate-spin" />
            Scanning trained adapters…
          </div>
        )}
        {error && <p className="text-[10px] text-red-400">{error}</p>}
        {!loading && adapters?.length === 0 && (
          <div className="py-4 text-center space-y-1">
            <p className="text-[11px] text-th-text-secondary">No distilled adapters yet.</p>
            <p className="text-[10px] text-th-text-muted">
              Train one on the Distill page.
            </p>
          </div>
        )}
        {!loading && adapters && adapters.length > 0 && visible.length === 0 && (
          <p className="text-[11px] text-th-text-muted text-center py-4">
            No adapters match “{search}”.
          </p>
        )}
        {!loading &&
          visible.map((a) => {
            const isActive = engine === "omlx"
              ? !!(a.omlx_model_id && a.omlx_model_id === selectedCatalogId)
              : (
                a.catalog_id === selectedCatalogId ||
                (!!selectedAdapterPath && a.adapter_path === selectedAdapterPath)
              );
            const teacher = a.teacher_model_id ? studentShort(a.teacher_model_id) : "";
            const when = formatTrainedAt(a.trained_at);
            return (
              <div
                key={a.catalog_id}
                className={`flex items-center justify-between gap-2 rounded-lg border px-3 py-2 ${
                  isActive
                    ? "border-emerald-500/40 bg-emerald-500/[0.06] ring-1 ring-emerald-500/20"
                    : "border-th-border bg-th-surface"
                }`}
              >
                <div className="min-w-0 flex-1">
                  <p className="text-[11px] font-medium text-th-text-primary truncate">
                    {a.display_name || a.catalog_id}
                  </p>
                    {canEditPurpose && editingId === a.catalog_id ? (
                      <div className="mt-1.5 space-y-1.5">
                        <textarea
                          value={purposeDraft}
                          onChange={(e) => setPurposeDraft(e.target.value)}
                          rows={2}
                          className="w-full px-2 py-1.5 text-[11px] rounded-md bg-th-input-bg border border-th-input-border text-th-text-primary focus:outline-none focus:border-violet-400"
                          placeholder="What this adapter is for…"
                        />
                        <div className="flex items-center gap-1.5">
                          <button
                            type="button"
                            disabled={savingPurpose}
                            onClick={() => void savePurpose(a.catalog_id)}
                            className="px-2 py-0.5 rounded text-[10px] font-medium bg-violet-500/15 text-violet-300 border border-violet-500/30 disabled:opacity-50"
                          >
                            {savingPurpose ? "Saving…" : "Save"}
                          </button>
                          <button
                            type="button"
                            onClick={() => setEditingId(null)}
                            className="px-2 py-0.5 rounded text-[10px] text-th-text-muted hover:text-th-text-primary"
                          >
                            Cancel
                          </button>
                        </div>
                      </div>
                    ) : (
                      <p className="text-[10px] text-th-text-secondary mt-1 leading-snug line-clamp-2">
                        {a.purpose || (canEditPurpose ? "No purpose set" : "")}
                      </p>
                    )}
                    <div className="flex items-center gap-1.5 mt-0.5 flex-wrap">
                    {a.omlx_model_id && (
                      <span className="text-[9px] font-semibold px-1 py-0.5 rounded bg-blue-500/15 text-blue-400 border border-blue-500/30">
                        turbo-ready
                      </span>
                    )}
                    <span className="text-[9px] text-th-text-muted truncate">
                      {studentShort(a.base_repo_id)}
                      {teacher ? ` ← ${teacher}` : ""}
                    </span>
                    <span className="text-[9px] text-th-text-muted">
                      {typeof a.size_mb === "number" ? `${a.size_mb.toFixed(1)} MB` : ""}
                    </span>
                    {when && <span className="text-[9px] text-th-text-muted">{when}</span>}
                    {isActive && (
                      <span className="text-[9px] font-semibold px-1.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 inline-flex items-center gap-0.5">
                        <CheckCircle2 size={8} />
                        active
                      </span>
                    )}
                    {canEditPurpose && editingId !== a.catalog_id && (
                      <button
                        type="button"
                        onClick={() => startEditPurpose(a)}
                        className="text-[9px] text-th-text-muted hover:text-violet-300 inline-flex items-center gap-0.5"
                      >
                        <Pencil size={8} />
                        {a.purpose ? "Edit purpose" : "Add purpose"}
                      </button>
                    )}
                  </div>
                </div>
                <button
                  type="button"
                  disabled={!canUse || busyId === a.catalog_id}
                  onClick={() => { if (canUse) void handleUse(a); }}
                  title={canUse ? undefined : "Open Settings → Standard to attach this adapter"}
                  className={`shrink-0 px-2.5 py-1 rounded-md text-[10px] font-medium transition-colors inline-flex items-center gap-1 ${
                    !canUse
                      ? "bg-th-surface text-th-text-muted border border-th-border cursor-not-allowed"
                      : isActive
                        ? "bg-emerald-700/50 text-emerald-200 cursor-default"
                        : "bg-emerald-600 text-white hover:bg-emerald-500"
                  }`}
                >
                  {busyId === a.catalog_id ? <Loader2 size={10} className="animate-spin" /> : <CheckCircle2 size={10} />}
                  {busyId === a.catalog_id
                    ? (a.omlx_model_id ? "Loading…" : "Fusing…")
                    : isActive
                      ? (engine === "omlx" ? "Loaded in Turbo" : "Selected")
                      : engine === "mlx"
                        ? "Use this model"
                        : engine === "omlx"
                          ? (a.omlx_model_id ? "Load in Turbo" : "Fuse & load in Turbo")
                          : "Use in Standard"}
                </button>
              </div>
            );
          })}
      </div>
      <button
        type="button"
        className="w-full text-[10px] text-th-text-muted hover:text-th-text-secondary transition-colors pt-1 inline-flex items-center justify-center gap-1"
        onClick={() => void refresh()}
      >
        <RefreshCw size={10} />
        Refresh distilled
      </button>
    </div>
  );
}
