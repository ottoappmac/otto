import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./useApi";
import { usePolling } from "./usePolling";
import type { AppSettings, ExoCatalogModel, MlxDownloadJob } from "../types";

// Model selection shared by Chat and Code. Choosing a model rewrites the
// global model in Settings for the active provider, so it applies to every
// chat. This is the logic that used to live inline in ChatPage; it moved
// here unchanged so both pages behave the same.

export function isDistillCatalogId(id: string | undefined | null): boolean {
  return (id || "").trim().startsWith("otto-distill/");
}

export type ChatModelOption = { id: string; name: string; adapter_path?: string };

async function distilledChatOptions(engine: "mlx" | "omlx" | "exo" = "mlx"): Promise<ChatModelOption[]> {
  try {
    const res = await api.listDistillAdapters();
    return (res.adapters ?? []).flatMap((a) => {
      const name = (a.display_name || a.catalog_id).trim();
      const label = name.toLowerCase().includes("distill") ? name : `${name} (distilled)`;
      if (engine === "omlx") {
        const oid = (a.omlx_model_id || "").trim();
        if (!oid) return [];
        return [{ id: oid, name: label, adapter_path: a.adapter_path }];
      }
      return [{ id: a.catalog_id, name: label, adapter_path: a.adapter_path }];
    });
  } catch {
    return [];
  }
}

function mergeModelOptions(primary: ChatModelOption[], extra: ChatModelOption[]): ChatModelOption[] {
  const seen = new Set<string>();
  const out: ChatModelOption[] = [];
  for (const row of [...extra, ...primary]) {
    if (!row.id || seen.has(row.id)) continue;
    seen.add(row.id);
    out.push(row);
  }
  return out;
}

export interface UseChatModelOptions {
  /** Called once with the loaded settings (Chat uses it for the default depth). */
  onSettingsLoaded?: (settings: AppSettings) => void;
  /**
   * Called when a model change leaves the open session on the old model.
   * Chat starts a new chat. Code leaves the session alone.
   */
  onSessionInvalidated?: () => void;
}

export function useChatModel({ onSettingsLoaded, onSessionInvalidated }: UseChatModelOptions = {}) {
  const onSettingsLoadedRef = useRef(onSettingsLoaded);
  const onSessionInvalidatedRef = useRef(onSessionInvalidated);
  onSettingsLoadedRef.current = onSettingsLoaded;
  onSessionInvalidatedRef.current = onSessionInvalidated;

  const [currentModel, setCurrentModel] = useState<string>(() => localStorage.getItem("chatModel") ?? "");
  const [availableModels, setAvailableModels] = useState<ChatModelOption[]>([]);
  const [modelsFetched, setModelsFetched] = useState(false);
  // Tracks the live EXO catalog (with downloaded/loaded flags) so that when
  // the user picks a model from the chat ModelPicker we can fire-and-forget
  // a preload for any model that's already on disk but not yet resident.
  const exoCatalogRef = useRef<ExoCatalogModel[]>([]);

  const exoAutoPreloadInFlightRef = useRef(false);
  const [exoModelLoading, setExoModelLoading] = useState(false);
  const [omlxModelLoading, setOmlxModelLoading] = useState(false);
  // Tracks an active MLX download job for the currently-selected model so
  // we can show progress and block sends while weights are still coming down.
  const [mlxDownloadJob, setMlxDownloadJob] = useState<MlxDownloadJob | null>(null);
  const [appSettings, setAppSettings] = useState<AppSettings | null>(null);
  const [debugLlm, setDebugLlm] = useState<{ provider: string; authMode: string; hasKeys: boolean; region: string } | null>(null);

  // Poll for an active MLX download job matching the currently-selected model.
  const syncMlxDownload = useCallback(async () => {
    if (debugLlm?.provider !== "mlx" || !currentModel) {
      setMlxDownloadJob(null);
      return;
    }
    try {
      const r = await api.mlxDownloadList();
      const active = r.jobs.find(
        (j) => j.repo_id === currentModel && (j.status === "running" || j.status === "pending"),
      ) ?? null;
      setMlxDownloadJob(active);
    } catch {
      // best-effort — don't surface polling errors
    }
  }, [debugLlm?.provider, currentModel]);

  const mlxIsDownloading = mlxDownloadJob !== null;
  // Fast poll (800ms) while a download is active for live progress;
  // slow poll (10s) otherwise so we notice when a download starts
  // without hammering the backend from every open chat window.
  usePolling(syncMlxDownload, mlxIsDownloading ? 800 : 10_000, debugLlm?.provider === "mlx");

  const isModelBusy = mlxIsDownloading || exoModelLoading || omlxModelLoading;

  useEffect(() => {
    api.getSettings().then((s) => {
      setAppSettings(s);
      onSettingsLoadedRef.current?.(s);
      const mlx = s.llm.mlx ?? { hf_llm_model_id: "", hf_vlm_model_id: "", hf_draft_llm_model_id: "", hf_token: "" };
      if (s.llm.provider === "mlx") {
        const mid = mlx.hf_llm_model_id?.trim() || localStorage.getItem("chatModel") || "";
        setCurrentModel(mid);
        if (mid) localStorage.setItem("chatModel", mid);
        setDebugLlm({
          provider: "mlx",
          authMode: mlx.hf_token ? "hub + token" : "hub",
          hasKeys: !!mid,
          region: "",
        });
        // Populate model picker with every model already in the Hub cache.
        // Falls back to just the configured id if the scan fails or returns nothing.
        api.mlxLocalModels()
          .then(async (r) => {
            const rows = r.models ?? [];
            const mapped: ChatModelOption[] = rows.map((m) => ({
              id: m.repo_id,
              name: m.source === "distill"
                ? ((m.name || m.repo_id).toLowerCase().includes("distill") ? (m.name || m.repo_id) : `${m.name || m.repo_id} (distilled)`)
                : (m.name || m.repo_id),
              adapter_path: m.adapter_path,
            }));
            const extras = mapped.some((m) => isDistillCatalogId(m.id))
              ? []
              : await distilledChatOptions("mlx");
            const merged = mergeModelOptions(
              mapped.filter((m) => !isDistillCatalogId(m.id)),
              mapped.filter((m) => isDistillCatalogId(m.id)).concat(extras),
            );
            if (merged.length > 0) {
              setAvailableModels(merged);
            } else if (mid) {
              setAvailableModels([{ id: mid, name: mid }]);
            } else {
              setAvailableModels([]);
            }
          })
          .catch(() => {
            setAvailableModels(mid ? [{ id: mid, name: mid }] : []);
          })
          .finally(() => setModelsFetched(true));
        return;
      }
      if (s.llm.provider === "exo") {
        const mid = s.exo.model_name?.trim() || localStorage.getItem("chatModel") || "";
        setCurrentModel(mid);
        if (mid) localStorage.setItem("chatModel", mid);
        setDebugLlm({
          provider: "exo",
          authMode: s.exo.base_url || `127.0.0.1:${s.exo.api_port}`,
          hasKeys: !!mid && s.exo.enabled,
          region: "",
        });
        // Populate the model picker from the live cluster catalog so the
        // user can switch between any model exo has (downloaded or loaded)
        // without leaving the chat page.  Falls back to just the currently
        // configured id if the cluster is unreachable.
        api.exoModels()
          .then(async (r) => {
            const catalog = r.reachable ? r.models : [];
            exoCatalogRef.current = catalog;
            const rich: ChatModelOption[] = catalog
              .filter((m) => m.downloaded || m.loaded)
              .map((m) => ({ id: m.id, name: m.name }));
            const distilled = await distilledChatOptions("exo");
            const merged = mergeModelOptions(rich, distilled);
            if (merged.length > 0) {
              setAvailableModels(merged);
            } else if (mid) {
              setAvailableModels([{ id: mid, name: mid }]);
            } else {
              setAvailableModels([]);
            }
          })
          .catch((e) => {
            console.warn("Failed to list EXO models:", e);
            exoCatalogRef.current = [];
            setAvailableModels(mid ? [{ id: mid, name: mid }] : []);
          })
          .finally(() => setModelsFetched(true));
        return;
      }
      if (s.llm.provider === "openai") {
        const o = s.llm.openai;
        setCurrentModel(o.model_name);
        localStorage.setItem("chatModel", o.model_name);
        const activeKey = o.model_provider === "azure" ? o.azure_api_key : o.api_key;
        setDebugLlm({
          provider: o.model_provider === "azure" ? "azure" : "openai",
          authMode: o.model_provider === "azure" ? o.azure_endpoint || "azure" : "api_key",
          hasKeys: !!(activeKey),
          region: o.model_provider === "azure" ? (o.azure_endpoint || "") : "",
        });
        if (!modelsFetched) {
          api.listModels({ provider: "openai", api_key: activeKey, model_name: o.model_name, openai_model_provider: o.model_provider, azure_endpoint: o.azure_endpoint, azure_api_version: o.azure_api_version, azure_deployment: o.azure_deployment })
            .then((r) => { if (r.models?.length) setAvailableModels(r.models); })
            .catch((e) => console.warn("Failed to list OpenAI models:", e))
            .finally(() => setModelsFetched(true));
        }
        return;
      }
      if (s.llm.provider === "omlx") {
        const mid = s.omlx?.model_name?.trim() || localStorage.getItem("chatModel") || "";
        setCurrentModel(mid);
        if (mid) localStorage.setItem("chatModel", mid);
        setDebugLlm({
          provider: "omlx",
          authMode: `127.0.0.1:${s.omlx?.api_port ?? 8000}`,
          hasKeys: !!mid && !!(s.omlx?.enabled),
          region: "",
        });
        api.omlxStatus()
          .then(async (r) => {
            const models = r.models ?? [];
            const hub: ChatModelOption[] = models.length > 0
              ? models.map((m: { id: string }) => ({ id: m.id, name: m.id }))
              : (mid ? [{ id: mid, name: mid }] : []);
            const distilled = await distilledChatOptions("omlx");
            const merged = mergeModelOptions(hub, distilled);
            if (merged.length > 0) {
              setAvailableModels(merged);
            } else {
              setAvailableModels([]);
            }
          })
          .catch(() => {
            setAvailableModels(mid ? [{ id: mid, name: mid }] : []);
          })
          .finally(() => setModelsFetched(true));
        return;
      }
      const a = s.llm.anthropic;
      setCurrentModel(a.model_name);
      localStorage.setItem("chatModel", a.model_name);
      setDebugLlm({
        provider: a.model_provider,
        authMode: a.model_provider === "bedrock" ? a.bedrock_auth_mode : "api_key",
        hasKeys: !!(a.aws_access_key_id || a.api_key),
        region: a.bedrock_region,
      });
      if (!modelsFetched) {
        api.listModels({ provider: s.llm.provider, api_key: a.api_key, model_name: a.model_name, model_provider: a.model_provider, bedrock_region: a.bedrock_region, bedrock_auth_mode: a.bedrock_auth_mode, aws_access_key_id: a.aws_access_key_id, aws_secret_access_key: a.aws_secret_access_key })
          .then((r) => { if (r.models?.length) setAvailableModels(r.models); })
          .catch((e) => console.warn("Failed to list models:", e))
          .finally(() => setModelsFetched(true));
      }
    }).catch((e) => console.warn("Failed to load settings:", e));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleModelChange = async (modelId: string) => {
    setCurrentModel(modelId);
    localStorage.setItem("chatModel", modelId);
    try {
      const s = await api.getSettings() as AppSettings;
      const picked = availableModels.find((m) => m.id === modelId);
      if (isDistillCatalogId(modelId)) {
        if (s.llm.provider === "omlx") {
          const adapters = await api.listDistillAdapters().catch(() => ({ adapters: [] }));
          const row = (adapters.adapters ?? []).find((a) => a.catalog_id === modelId);
          const omlxId = (row?.omlx_model_id || "").trim();
          if (omlxId) {
            modelId = omlxId;
            setCurrentModel(omlxId);
            localStorage.setItem("chatModel", omlxId);
          } else {
            console.warn("Distilled adapter is not fused for Turbo yet:", modelId);
            return;
          }
        } else {
          const prevProvider = s.llm.provider;
          const omlxModel = s.omlx?.model_name?.trim() ?? "";
          const nextMlx = {
            ...(s.llm.mlx ?? { hf_llm_model_id: "", hf_vlm_model_id: "", hf_draft_llm_model_id: "", hf_token: "" }),
            hf_llm_model_id: modelId,
            adapter_path: picked?.adapter_path || s.llm.mlx.adapter_path || "",
          };
          const nextSettings = {
            ...s,
            llm: { ...s.llm, provider: "mlx" as const, mlx: nextMlx },
          } as AppSettings;
          await api.updateSettings(nextSettings as unknown as Record<string, unknown>);
          setAppSettings(nextSettings);
          setDebugLlm({
            provider: "mlx",
            authMode: nextMlx.hf_token ? "hub + token" : "hub",
            hasKeys: !!modelId.trim(),
            region: "",
          });
          if (prevProvider === "omlx") {
            void (async () => {
              try {
                if (omlxModel) await api.omlxUnloadModel(omlxModel);
                await api.omlxStop();
              } catch {
                /* Turbo unload is best-effort; Standard chat still uses mlx_lm. */
              }
            })();
          }
          return;
        }
      }
      if (s.llm.provider === "mlx") {
        const nextMlx = {
          ...(s.llm.mlx ?? { hf_llm_model_id: "", hf_vlm_model_id: "", hf_draft_llm_model_id: "", hf_token: "" }),
          hf_llm_model_id: modelId,
          adapter_path: "",
        };
        const nextSettings = { ...s, llm: { ...s.llm, mlx: nextMlx } } as AppSettings;
        await api.updateSettings(nextSettings as unknown as Record<string, unknown>);
        setAppSettings(nextSettings);
        setDebugLlm({
          provider: "mlx",
          authMode: nextMlx.hf_token ? "hub + token" : "hub",
          hasKeys: !!modelId.trim(),
          region: "",
        });
        return;
      }
      if (s.llm.provider === "exo") {
        const nextExo = { ...s.exo, model_name: modelId };
        const nextSettings = { ...s, exo: nextExo } as AppSettings;
        await api.updateSettings(nextSettings as unknown as Record<string, unknown>);
        setAppSettings(nextSettings);
        setDebugLlm({
          provider: "exo",
          authMode: nextExo.base_url || `127.0.0.1:${nextExo.api_port}`,
          hasKeys: !!modelId.trim() && nextExo.enabled,
          region: "",
        });
        // Fetch a fresh catalog to get accurate downloaded/loaded state, then
        // unload any other loaded model and warm-load the selected one.
        // Uses the async job API (same as the settings page) so we can poll
        // progress without blocking the UI thread.
        const trimmedId = modelId.trim();
        if (trimmedId && nextExo.enabled && !exoAutoPreloadInFlightRef.current) {
          exoAutoPreloadInFlightRef.current = true;
          setExoModelLoading(true);
          void (async () => {
            try {
              // Fresh catalog so we don't act on stale downloaded/loaded flags.
              const fresh = await api.exoModels();
              if (fresh.reachable) exoCatalogRef.current = fresh.models;
              const freshCatalog = fresh.reachable ? fresh.models : exoCatalogRef.current;
              const newModel = freshCatalog.find((c) => c.id === trimmedId);
              const loadedOthers = freshCatalog.filter((c) => c.loaded && c.id !== trimmedId);
              for (const prev of loadedOthers) {
                await api.exoUnload(prev.id);
              }
              // Only load if the model is on disk but not yet in cluster memory.
              if (newModel?.downloaded && !newModel.loaded) {
                const job = await api.exoPreloadStart(trimmedId, 1);
                const startedAt = Date.now();
                while (Date.now() - startedAt < 600_000) {
                  await new Promise<void>((res) => setTimeout(res, 2000));
                  const j = await api.exoPreloadStatus(job.job_id);
                  if (j.status === "done" || j.status === "error" || j.status === "cancelled") break;
                }
                const r = await api.exoModels();
                if (r.reachable) exoCatalogRef.current = r.models;
              }
            } catch (e) {
              console.warn("Auto unload/preload of EXO model failed:", e);
            } finally {
              exoAutoPreloadInFlightRef.current = false;
              setExoModelLoading(false);
            }
          })();
        }
        return;
      }
      if (s.llm.provider === "openai") {
        const nextOpenAI = { ...s.llm.openai, model_name: modelId };
        const nextSettings = { ...s, llm: { ...s.llm, openai: nextOpenAI } } as AppSettings;
        await api.updateSettings(nextSettings as unknown as Record<string, unknown>);
        setAppSettings(nextSettings);
        const o = nextOpenAI;
        const activeKey = o.model_provider === "azure" ? o.azure_api_key : o.api_key;
        setDebugLlm({
          provider: o.model_provider === "azure" ? "azure" : "openai",
          authMode: o.model_provider === "azure" ? o.azure_endpoint || "azure" : "api_key",
          hasKeys: !!(activeKey),
          region: o.model_provider === "azure" ? (o.azure_endpoint || "") : "",
        });
        return;
      }
      if (s.llm.provider === "omlx") {
        const adapters = await api.listDistillAdapters().catch(() => ({ adapters: [] }));
        const fusedIds = new Set(
          (adapters.adapters ?? []).map((a) => (a.omlx_model_id || "").trim()).filter(Boolean),
        );
        const pickingFused = fusedIds.has(modelId);
        const nextMlx = pickingFused
          ? s.llm.mlx
          : {
              ...s.llm.mlx,
              hf_llm_model_id: isDistillCatalogId(s.llm.mlx?.hf_llm_model_id) ? "" : (s.llm.mlx?.hf_llm_model_id || ""),
              adapter_path: "",
            };
        const nextOmlx = { ...s.omlx, model_name: modelId };
        const nextSettings = { ...s, llm: { ...s.llm, mlx: nextMlx }, omlx: nextOmlx } as AppSettings;
        await api.updateSettings(nextSettings as unknown as Record<string, unknown>);
        setAppSettings(nextSettings);
        setDebugLlm({
          provider: "omlx",
          authMode: `127.0.0.1:${nextOmlx.api_port ?? 8000}`,
          hasKeys: !!modelId.trim() && !!(nextOmlx.enabled),
          region: "",
        });
        // The live session graph still has the previous model baked in.
        onSessionInvalidatedRef.current?.();
        const trimmedOmlxId = modelId.trim();
        if (trimmedOmlxId && nextOmlx.enabled) {
          setOmlxModelLoading(true);
          void (async () => {
            try {
              const status = await api.omlxStatus();
              for (const loaded of status.loaded_models ?? []) {
                if (loaded.id && loaded.id !== trimmedOmlxId) {
                  await api.omlxUnloadModel(loaded.id).catch(() => undefined);
                }
              }
              const alreadyLoaded = (status.loaded_models ?? []).some((m) => m.id === trimmedOmlxId);
              if (!alreadyLoaded) {
                const job = await api.omlxLoadModel(trimmedOmlxId);
                const startedAt = Date.now();
                while (Date.now() - startedAt < 600_000) {
                  await new Promise<void>((res) => setTimeout(res, 2000));
                  const j = await api.getOmlxJob(job.job_id);
                  if (j.status === "done" || j.status === "error") break;
                }
              }
            } catch (e) {
              console.warn("oMLX model load failed:", e);
            } finally {
              setOmlxModelLoading(false);
            }
          })();
        }
        return;
      }
      s.llm.anthropic.model_name = modelId;
      await api.updateSettings(s as unknown as Record<string, unknown>);
      setAppSettings(s);
      const a = s.llm.anthropic;
      setDebugLlm({
        provider: a.model_provider,
        authMode: a.model_provider === "bedrock" ? a.bedrock_auth_mode : "api_key",
        hasKeys: !!(a.aws_access_key_id || a.api_key),
        region: a.bedrock_region,
      });
    } catch (e) { console.warn("Failed to save model settings:", e); }
  };

  return {
    currentModel,
    availableModels,
    appSettings,
    debugLlm,
    mlxDownloadJob,
    mlxIsDownloading,
    exoModelLoading,
    omlxModelLoading,
    isModelBusy,
    handleModelChange,
  };
}
