import { useCallback, useEffect, useState } from "react";
import { Download, HardDrive, Library, Loader2, Search, Sparkles } from "lucide-react";
import { api } from "../../hooks/useApi";
import type { OmlxLocalModel, OmlxModelCatalogRow } from "../../types";

type PickerTab = "library" | "discover" | "custom";

export function FitBadge({ fits }: { fits: string }) {
  const map: Record<string, { label: string; cls: string }> = {
    comfortable: { label: "Fits",  cls: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30" },
    tight:       { label: "Tight", cls: "bg-amber-500/15  text-amber-400  border-amber-500/30" },
    over:        { label: "Large", cls: "bg-red-500/15    text-red-400    border-red-500/30" },
    unknown:     { label: "?",     cls: "bg-th-surface    text-th-text-muted border-th-border" },
  };
  const { label, cls } = map[fits] ?? map["unknown"];
  return (
    <span className={`text-[9px] font-semibold px-1.5 py-0.5 rounded border ${cls}`}>
      {label}
    </span>
  );
}

/**
 * Three-tab model browser for oMLX.
 *
 * - **Your library** — scans the local HF cache for already-downloaded models.
 * - **Discover**     — hardware-scored curated catalog; shows fit badge + Get / Load.
 * - **Custom**       — free-form HF repo id or local path.
 *
 * The `onLoad` callback is called with the chosen model id; callers are
 * responsible for invoking `api.omlxLoadModel` and updating settings.
 *
 * When `serverRunning` is false the action buttons say "Select" instead of
 * "Load" — the caller's `onLoad` can save the model id as the intended
 * default so it's used when the server starts later.
 */
export function OmlxModelPicker({
  onLoad,
  serverRunning = true,
}: {
  onLoad: (modelId: string) => Promise<void>;
  serverRunning?: boolean;
}) {
  const [tab, setTab] = useState<PickerTab>("library");
  const [localModels, setLocalModels] = useState<OmlxLocalModel[] | null>(null);
  const [catalog, setCatalog] = useState<OmlxModelCatalogRow[] | null>(null);
  const [loadingLocal, setLoadingLocal] = useState(false);
  const [loadingCatalog, setLoadingCatalog] = useState(false);
  const [customInput, setCustomInput] = useState("");
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [scanError, setScanError] = useState<string | null>(null);
  const [librarySearch, setLibrarySearch] = useState("");
  const [discoverSearch, setDiscoverSearch] = useState("");
  const [hubResults, setHubResults] = useState<OmlxModelCatalogRow[]>([]);
  const [hubSearching, setHubSearching] = useState(false);
  const [hubError, setHubError] = useState<string | null>(null);

  const fetchLocal = useCallback(async () => {
    if (localModels !== null) return;
    setLoadingLocal(true);
    setScanError(null);
    try {
      const res = await api.omlxLocalModels();
      setLocalModels(res.models);
      if (res.error) setScanError(res.error);
    } catch (e) {
      setScanError(e instanceof Error ? e.message : String(e));
      setLocalModels([]);
    } finally {
      setLoadingLocal(false);
    }
  }, [localModels]);

  const fetchCatalog = useCallback(async () => {
    if (catalog !== null) return;
    setLoadingCatalog(true);
    try {
      const res = await api.omlxModelCatalog();
      setCatalog(res.models);
    } catch {
      setCatalog([]);
    } finally {
      setLoadingCatalog(false);
    }
  }, [catalog]);

  useEffect(() => {
    if (tab === "library") void fetchLocal();
    if (tab === "discover") void fetchCatalog();
  }, [tab, fetchLocal, fetchCatalog]);

  // Search-as-you-type fallthrough: when the local catalog has no match for
  // the query, ask the HF Hub directly so any mlx-community repo is reachable.
  useEffect(() => {
    if (tab !== "discover") return;
    const s = discoverSearch.trim().toLowerCase();
    const localMatch = (catalog ?? []).some(
      (m) =>
        m.repo_id.toLowerCase().includes(s) ||
        m.display_name.toLowerCase().includes(s) ||
        (m.blurb ?? "").toLowerCase().includes(s),
    );
    if (s.length < 2 || localMatch) {
      setHubResults([]);
      setHubError(null);
      setHubSearching(false);
      return;
    }
    setHubSearching(true);
    setHubError(null);
    let cancelled = false;
    const handle = setTimeout(() => {
      void (async () => {
        try {
          const res = await api.omlxSearchModels(discoverSearch.trim());
          if (!cancelled) setHubResults(res.models);
        } catch (e) {
          if (!cancelled) {
            setHubError(e instanceof Error ? e.message : String(e));
            setHubResults([]);
          }
        } finally {
          if (!cancelled) setHubSearching(false);
        }
      })();
    }, 450);
    return () => {
      cancelled = true;
      clearTimeout(handle);
    };
  }, [discoverSearch, catalog, tab]);

  const handleLoad = async (modelId: string) => {
    setLoadingId(modelId);
    try {
      await onLoad(modelId);
    } finally {
      setLoadingId(null);
    }
  };

  const renderCatalogRow = (m: OmlxModelCatalogRow) => (
    <div
      key={m.repo_id}
      className={`flex items-center justify-between gap-2 rounded-lg border px-3 py-2
        ${m.fits === "over"
          ? "border-red-500/20 bg-red-500/5"
          : "border-th-border bg-th-surface"}`}
    >
      <div className="min-w-0 flex-1 space-y-0.5">
        <div className="flex items-center gap-1.5 flex-wrap">
          <p className="text-[11px] font-medium text-th-text-primary truncate">{m.display_name}</p>
          <FitBadge fits={m.fits} />
          {m.already_cached && (
            <span className="text-[9px] font-semibold px-1 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              Cached
            </span>
          )}
        </div>
        {m.blurb && <p className="text-[10px] text-th-text-muted">{m.blurb}</p>}
        <p className="text-[9px] text-th-text-tertiary">
          {m.weights_gb} GB · {m.params_b}B · {m.quant}
        </p>
      </div>
      <button
        type="button"
        disabled={loadingId !== null || m.fits === "over"}
        onClick={() => void handleLoad(m.repo_id)}
        title={m.fits === "over" ? "Model may exceed available RAM" : undefined}
        className="shrink-0 px-2.5 py-1 rounded-md bg-th-tab-active-bg text-white text-[10px] font-medium disabled:opacity-50 inline-flex items-center gap-1"
      >
        {loadingId === m.repo_id
          ? <Loader2 size={10} className="animate-spin" />
          : m.already_cached ? <Sparkles size={10} /> : <Download size={10} />}
        {m.already_cached
          ? (serverRunning ? "Load" : "Select")
          : "Get"}
      </button>
    </div>
  );

  const tabs: { id: PickerTab; icon: React.ReactNode; label: string }[] = [
    { id: "library",  icon: <Library size={11} />,  label: "Your library" },
    { id: "discover", icon: <Sparkles size={11} />, label: "Discover" },
    { id: "custom",   icon: <HardDrive size={11} />, label: "Custom" },
  ];

  return (
    <div className="rounded-lg border border-th-border bg-th-inset-bg overflow-hidden">
      {/* Tab bar */}
      <div className="flex border-b border-th-border">
        {tabs.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => setTab(t.id)}
            className={`flex-1 px-3 py-2 text-[11px] font-medium flex items-center justify-center gap-1.5 transition-colors
              ${tab === t.id
                ? "bg-th-tab-active-bg text-white"
                : "text-th-text-secondary hover:text-th-text-primary hover:bg-th-surface-hover/20"}`}
          >
            {t.icon}
            {t.label}
          </button>
        ))}
      </div>

      <div className="p-3 space-y-2">
        {/* ── Library tab ── */}
        {tab === "library" && (
          <>
            <div className="relative">
              <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-th-text-muted pointer-events-none" />
              <input
                type="text"
                placeholder="Search cached models…"
                value={librarySearch}
                onChange={(e) => setLibrarySearch(e.target.value)}
                className="w-full pl-7 pr-3 py-1.5 text-[11px] rounded-lg border border-th-border bg-th-surface text-th-text-primary placeholder:text-th-text-muted focus:outline-none focus:ring-1 focus:ring-th-tab-active-bg/40 focus:border-th-tab-active-bg/50 transition-shadow"
              />
            </div>
            <div className="space-y-2 max-h-60 overflow-y-auto">
            {loadingLocal && (
              <div className="flex items-center gap-2 text-[11px] text-th-text-muted py-4 justify-center">
                <Loader2 size={12} className="animate-spin" />
                Scanning HF cache…
              </div>
            )}
            {scanError && (
              <p className="text-[10px] text-red-400">{scanError}</p>
            )}
            {!loadingLocal && localModels?.length === 0 && (
              <div className="py-4 text-center space-y-1">
                <p className="text-[11px] text-th-text-secondary">No models found in your HF cache.</p>
                <p className="text-[10px] text-th-text-muted">
                  Switch to <strong>Discover</strong> to download a model, or set a custom cache path in Settings → MLX.
                </p>
              </div>
            )}
            {!loadingLocal && (() => {
              const s = librarySearch.trim().toLowerCase();
              const visible = (localModels ?? []).filter((m) =>
                !s || m.repo_id.toLowerCase().includes(s),
              );
              if (visible.length === 0 && s) {
                return (
                  <p className="text-[11px] text-th-text-muted text-center py-4">
                    No models match "{librarySearch}".
                  </p>
                );
              }
              return visible.map((m) => (
              <div
                key={m.repo_id}
                className="flex items-center justify-between gap-2 rounded-lg border border-th-border bg-th-surface px-3 py-2"
              >
                <div className="min-w-0 flex-1">
                  <p className="text-[11px] font-medium text-th-text-primary truncate">{m.repo_id}</p>
                  <div className="flex items-center gap-1.5 mt-0.5">
                    <span className="text-[9px] text-th-text-muted">{m.size_gb} GB</span>
                    {m.is_mlx && (
                      <span className="text-[9px] font-semibold px-1 py-0.5 rounded bg-blue-500/15 text-blue-400 border border-blue-500/30">
                        MLX
                      </span>
                    )}
                  </div>
                </div>
                <button
                  type="button"
                  disabled={loadingId !== null}
                  onClick={() => void handleLoad(m.repo_id)}
                  className="shrink-0 px-2.5 py-1 rounded-md bg-th-tab-active-bg text-white text-[10px] font-medium disabled:opacity-50 inline-flex items-center gap-1"
                >
                  {loadingId === m.repo_id
                    ? <Loader2 size={10} className="animate-spin" />
                    : <Sparkles size={10} />}
                  {serverRunning ? "Load" : "Select"}
                </button>
              </div>
              ));
            })()}
            </div>
          </>
        )}

        {/* ── Discover tab ── */}
        {tab === "discover" && (
          <>
            <div className="relative">
              <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-th-text-muted pointer-events-none" />
              <input
                type="text"
                placeholder="Search models…"
                value={discoverSearch}
                onChange={(e) => setDiscoverSearch(e.target.value)}
                className="w-full pl-7 pr-3 py-1.5 text-[11px] rounded-lg border border-th-border bg-th-surface text-th-text-primary placeholder:text-th-text-muted focus:outline-none focus:ring-1 focus:ring-th-tab-active-bg/40 focus:border-th-tab-active-bg/50 transition-shadow"
              />
            </div>
            <div className="space-y-2 max-h-60 overflow-y-auto">
            {loadingCatalog && (
              <div className="flex items-center gap-2 text-[11px] text-th-text-muted py-4 justify-center">
                <Loader2 size={12} className="animate-spin" />
                Scoring models against your hardware…
              </div>
            )}
            {!loadingCatalog && (() => {
              const s = discoverSearch.trim().toLowerCase();
              const visible = (catalog ?? []).filter((m) =>
                !s || m.repo_id.toLowerCase().includes(s) || m.display_name.toLowerCase().includes(s) || (m.blurb ?? "").toLowerCase().includes(s),
              );
              if (visible.length > 0) {
                return visible.map(renderCatalogRow);
              }
              // No local match — fall through to live Hugging Face search.
              return (
                <>
                  {hubSearching && (
                    <div className="flex items-center gap-2 text-[11px] text-th-text-muted py-4 justify-center">
                      <Loader2 size={12} className="animate-spin" />
                      Searching Hugging Face…
                    </div>
                  )}
                  {!hubSearching && hubResults.length > 0 && (
                    <>
                      <p className="text-[9px] text-th-text-tertiary px-0.5">
                        Results from Hugging Face
                      </p>
                      {hubResults.map(renderCatalogRow)}
                    </>
                  )}
                  {hubError && (
                    <p className="text-[10px] text-red-400 text-center py-2">{hubError}</p>
                  )}
                  {!hubSearching && !hubError && hubResults.length === 0 && discoverSearch.trim() && (
                    <p className="text-[11px] text-th-text-muted text-center py-4">
                      No models match "{discoverSearch}".{discoverSearch.trim().length < 2 && " Type a bit more to search Hugging Face."}
                    </p>
                  )}
                </>
              );
            })()}
            </div>
          </>
        )}

        {/* ── Custom tab ── */}
        {tab === "custom" && (
          <div className="space-y-3 py-1">
            <p className="text-[11px] text-th-text-secondary leading-relaxed">
              Enter a Hugging Face repo ID (e.g.{" "}
              <span className="font-mono">mlx-community/Qwen3-8B-4bit</span>) or an absolute path
              to a local model directory.
            </p>
            <div className="flex gap-2">
              <input
                type="text"
                value={customInput}
                onChange={(e) => setCustomInput(e.target.value)}
                placeholder="mlx-community/your-model or /path/to/model"
                className="flex-1 px-3 py-2 rounded-lg border border-th-border bg-th-surface text-[11px] text-th-text-primary placeholder:text-th-text-muted"
              />
              <button
                type="button"
                disabled={!customInput.trim() || loadingId !== null}
                onClick={() => void handleLoad(customInput.trim())}
                className="shrink-0 px-3 py-2 rounded-lg bg-th-tab-active-bg text-white text-[10px] font-medium disabled:opacity-50 inline-flex items-center gap-1.5"
              >
                {loadingId === customInput.trim()
                  ? <Loader2 size={10} className="animate-spin" />
                  : <Sparkles size={10} />}
                Load
              </button>
            </div>
            <p className="text-[10px] text-th-text-muted leading-relaxed">
              {serverRunning
                ? <>Otto will stop the current server and restart it with{" "}
                    <span className="font-mono">omlx serve --model &lt;id&gt;</span>.
                    If the model isn't cached locally oMLX will download it first.</>
                : <>This repo id will be saved as the default model. When you start the server it will be loaded automatically.</>}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
