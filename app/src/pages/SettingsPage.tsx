import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { getVersion } from "@tauri-apps/api/app";
import { useLocation } from "react-router-dom";
import { Eye, EyeOff, CheckCircle, CheckCircle2, AlertTriangle, XCircle, RefreshCw, Loader2, Check, ChevronDown, Plus, Trash2, Server, Play, Square, X, Wifi, PlugZap, ShieldCheck, ShieldOff, ShieldAlert, Copy, ClipboardCheck, Lock, Unlock, Mic, Zap, Wand2 } from "lucide-react";
import { api } from "../hooks/useApi";
import { WS_BASE } from "../config/apiBase";
import { usePolling } from "../hooks/usePolling";
import ClusterSetupFlow from "../components/cluster/ClusterSetupFlow";
import ExoSetupSteps, { type ExoSetupCompletion } from "../components/exo/ExoSetupSteps";
import ExoModelChooser from "../components/exo/ExoModelChooser";
import ExoRuntimeSource from "../components/exo/ExoRuntimeSource";
import ModelChooser from "../components/mlx/ModelChooser";
import { OmlxModelPicker } from "../components/omlx/OmlxModelPicker";
import { MemoryPanel, AmbientPanel } from "./MemoryPage";
import VoiceModelChooser from "../components/voice/VoiceModelChooser";
import type { AppSettings, ExoCatalogModel, ExoConfig, ExoJob, ExoNodeInfo, ExoRemote, ExoStatus, LanSshHost, MlxHfConfig, OpenAIConfig, OrchestratorConfig, PrivacyAuditEntry, PrivacyStatus, SshConfigHost, VoiceConfig } from "../types";

const TABS = ["LLM", "Agent Memory", "Suggestions", "macOS Activity", "Voice", "Advanced", "Observability", "Privacy & Security", "About"] as const;
type Tab = (typeof TABS)[number];

const LLM_SUBTABS = ["Model Provider", "Standard", "Turbo", "Cluster", "Frontier"] as const;
type LlmSubTab = (typeof LLM_SUBTABS)[number];

const EXO_SUBTABS = ["Cluster", "Advanced"] as const;
type ExoSubTab = (typeof EXO_SUBTABS)[number];

const STANDARD_SUBTABS = ["Overview", "Model"] as const;
type StandardSubTab = (typeof STANDARD_SUBTABS)[number];

const AUTO_SAVE_DELAY_MS = 800;

/**
 * Load an oMLX model and wait for the background job to finish.
 *
 * `api.omlxLoadModel` only kicks off a job; on its own the caller can't
 * tell whether the model actually loaded (e.g. oMLX may reject a model
 * whose architecture it doesn't support). This polls the job to a
 * terminal state and throws with the job's error so the UI can surface
 * it instead of silently saving an unloadable default.
 */
async function loadOmlxModelAndWait(
  modelId: string,
  { timeoutMs = 600_000, intervalMs = 1_500 } = {},
): Promise<void> {
  const { job_id } = await api.omlxLoadModel(modelId);
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, intervalMs));
    const job = await api.getOmlxJob(job_id);
    if (job.status === "done") return;
    if (job.status === "error") {
      throw new Error(job.error || `Failed to load model '${modelId}'.`);
    }
  }
  throw new Error(
    `Loading '${modelId}' timed out. Check the oMLX setup screen for details.`,
  );
}

/**
 * Start the oMLX server and poll the resulting job to a terminal state.
 *
 * No-op on the backend when the server is already reachable, so callers can
 * invoke it unconditionally. Throws with the job's error message on failure
 * or timeout so the provider switcher can surface it to the user.
 */
async function startOmlxServerAndWait(
  { timeoutMs = 120_000, intervalMs = 1_500 } = {},
): Promise<void> {
  const { job_id } = await api.omlxStart();
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, intervalMs));
    const job = await api.getOmlxJob(job_id);
    if (job.status === "done") return;
    if (job.status === "error") {
      throw new Error(job.error || "Failed to start the oMLX server.");
    }
  }
  throw new Error("Starting the oMLX server timed out.");
}

/**
 * Stop the oMLX server, polling the job to a terminal state. Best-effort —
 * resolves on either done or error so a stuck stop never blocks a switch.
 */
async function stopOmlxServerAndWait(
  { timeoutMs = 30_000, intervalMs = 1_000 } = {},
): Promise<void> {
  const { job_id } = await api.omlxStop();
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, intervalMs));
    const job = await api.getOmlxJob(job_id);
    if (job.status === "done" || job.status === "error") return;
  }
}

/**
 * Start the Exo cluster (provision + start + verify) and poll the job to a
 * terminal state. Streams the latest log line via `onProgress` so the UI can
 * reflect provisioning/start progress. Throws on failure or timeout.
 */
async function startExoServerAndWait(
  onProgress?: (line: string) => void,
  { timeoutMs = 600_000, intervalMs = 3_000 } = {},
): Promise<void> {
  const { job_id } = await api.exoUp();
  const deadline = Date.now() + timeoutMs;
  let lastLineCount = 0;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, intervalMs));
    const job = await api.getExoJob(job_id);
    if (onProgress && job.log_lines && job.log_lines.length > lastLineCount) {
      const newLines = job.log_lines.slice(lastLineCount);
      lastLineCount = job.log_lines.length;
      const readable = [...newLines].reverse().find((l) => l.trim().length > 0);
      if (readable) onProgress(readable.trim());
    }
    if (job.status === "done") return;
    if (job.status === "error") {
      throw new Error(job.error || "Failed to start the Exo cluster.");
    }
  }
  throw new Error("Starting the Exo cluster timed out.");
}

function mlxSelectOptions(
  models: { repo_id: string; name: string }[],
  current: string,
  includeEmpty: boolean,
  loading = false,
  fetched = false,
): { value: string; label: string; notInCache?: boolean }[] {
  if (loading) {
    const base: { value: string; label: string }[] = includeEmpty ? [{ value: "", label: "(none)" }] : [];
    if (current) base.push({ value: current, label: `${current} (loading list…)` });
    else if (!includeEmpty) base.push({ value: "", label: "(loading…)" });
    return base;
  }
  const opts: { value: string; label: string; notInCache?: boolean }[] = includeEmpty ? [{ value: "", label: "(none)" }] : [];
  const seen = new Set<string>();
  for (const m of models) {
    if (!seen.has(m.repo_id)) {
      seen.add(m.repo_id);
      opts.push({ value: m.repo_id, label: m.name });
    }
  }
  if (current && !seen.has(current)) {
    opts.push({ value: current, label: current, notInCache: fetched });
  }
  if (!includeEmpty && opts.length === 0) {
    opts.push({ value: "", label: fetched ? "(no models in cache — click Refresh)" : "(loading…)" });
  }
  return opts;
}

const DEFAULT_SETTINGS: AppSettings = {
  llm: {
    provider: "anthropic",
    anthropic: { model_provider: "anthropic", api_key: "", model_name: "claude-sonnet-4-6", bedrock_region: "us-east-1", bedrock_auth_mode: "keys", aws_access_key_id: "", aws_secret_access_key: "", max_tokens: 8192, thinking_enabled: false, thinking_budget: 2048, tool_efficient: true },
    openai: { model_provider: "openai", api_key: "", azure_api_key: "", model_name: "gpt-4o", azure_endpoint: "", azure_api_version: "2024-12-01-preview", azure_deployment: "", max_tokens: 16384, temperature: 0.0 },
    mlx: {
      hf_llm_model_id: "mlx-community/quantized-gemma-2b-it",
      hf_vlm_model_id: "",
      hf_draft_llm_model_id: "",
      hf_token: "",
      hf_hub_cache: "huggingface/hub",
      mlx_bookmarks: [],
      mlx_max_tokens: 8192,
      mlx_temp: 0,
      mlx_verbose: false,
      mlx_thinking: false,
      mlx_prompt_cache: false,
      mlx_system_prompt_cache: false,
      mlx_kv_bits: null,
      mlx_kv_group_size: 64,
      mlx_repetition_penalty: 1.1,
      mlx_prompt_cache_max_tokens: 32768,
      turbo_level: "off",
      turbo_ssd_dir: "",
      turbo_ssd_max_gb: 50,
      turbo_tq_bits: 4,
      turbo_block_size: 256,
    },
  },
  orchestrator: {
    llm_family: "follow_main",
    mlx_model: "",
    mlx_model_type: "llm",
    provider_override: null,
    prompt_mode: "auto",
    recursion_limit: 10000,
  },
  mcp_servers: [],
  observability: { langsmith: { enabled: false, api_key: "", endpoint: "https://api.smith.langchain.com", project: "Research" }, log_level: "INFO" },
  claude_hook: { enabled: true, http_hooks_enabled: false, quality_gate_enabled: false, quality_gate_threshold: 0.5, auto_monitor_enabled: false, max_auto_sessions: 3, auto_monitor_agent: "claude-session-eval-agent", hooks: [] },
  evaluation: { auto_evaluate: false, analyze_errors: false, llm_family: "follow_main", max_metrics: 4, threshold: 0.5 },
  openclaw: { enabled: false, mode: "local" as const, state_dir: "~/.openclaw", ssh_host: "", ssh_user: "ubuntu", ssh_key_path: "", ssh_port: 22, watcher_enabled: false, watcher_poll_interval: 10, auto_monitor_enabled: false, max_auto_sessions: 3, auto_monitor_agent: "openclaw-session-eval-agent" },
  memory: { enabled: false, inject_enabled: false, inject_on_session_start: false, inject_realtime: false, model_name: "", llm_family: "follow_main", mlx_model: "", min_hours: 24, min_sessions: 5, retention_days: 30, max_memory_files: 200, max_index_kb: 25, embedding: { enabled: true, model_name: "sentence-transformers/all-MiniLM-L6-v2", chunk_size: 1500, chunk_overlap: 150 } },
  exo: { enabled: false, mode: "prebuilt", prebuilt_url: "", repo_url: "https://github.com/exo-explore/exo.git", repo_ref: "v1.0.71", api_port: 52415, libp2p_port: 0, base_url: "", model_name: "", auto_start: false, auto_provision: true, no_terminal_wrap: false, min_nodes: 1, max_tokens: 8192, enable_thinking: false, sharding: "Pipeline", instance_meta: "MlxRing", remotes: [] },
  omlx: { enabled: false, api_port: 52414, base_url: "", model_name: "", auto_start: false, brew_tap: "jundot/omlx", brew_tap_url: "https://github.com/jundot/omlx", brew_formula: "omlx", cli_path: "", model_dirs: ["~/.cache/huggingface/hub"], max_context_window: 131072, thinking_enabled: false, max_tokens: 8192 },
  activity: { enabled: false, interval_secs: 5, retain_days: 30, exclude_apps: [], idle_threshold_secs: 60, min_span_secs: 5, max_span_secs: 300, context_max_chars: 500, field_val_max_chars: 200, browser_text_max_chars: 500, ax_walk_max_chars: 2000, ax_walk_max_depth: 5, max_db_mb: 500 },
  privacy: { enabled: false, local_only_providers: ["mlx", "omlx", "exo"], allowed_hosts: [], allow_loopback: true, allow_mdns: true, pf_anchor: "otto.privacy", engaged_at: "", audit_token: "" },
  auto_approve_commands: false,
  ambient_suggest_recurrence: false,
  ambient: {
    enabled: false,
    llm_family: "mlx",
    mlx_model: "mlx-community/Qwen3-1.7B-4bit",
    model_name: "",
    interval_mins: 30,
    idle_only: true,
    react_to_session_end: true,
    use_memory: true,
    use_sessions: true,
    use_activity: true,
    use_history: true,
    lookback_hours: 24,
    min_confidence: 0.6,
    max_hints_per_day: 10,
    cooldown_hours: 4,
    quiet_hours_start: 22,
    quiet_hours_end: 8,
    allow_auto_run: false,
  },
  voice: {
    enabled: false,
    activation_mode: "ptt",
    ptt_hotkey: "",
    stt_enabled: true,
    stt_model: "mlx-community/whisper-large-v3-turbo",
    stt_language: "",
    wake_enabled: false,
    wake_model: "hey_otto",
    vad_silence_secs: 1.0,
    mic_device: "",
    loopback_enabled: false,
    loopback_vad_silence_secs: 0.7,
    loopback_max_segment_secs: 12.0,
    loopback_live_partials: true,
    loopback_partial_interval_secs: 1.5,
    loopback_auto_send_silence_secs: 2.5,
  },
};

type SaveStatus = "idle" | "saving" | "saved" | "error";

export default function SettingsPage() {
  const location = useLocation();
  const { initialTab, initialSub } = useMemo(() => {
    const params = new URLSearchParams(location.search);
    const t = params.get("tab");
    const s = params.get("sub")?.trim() ?? "";
    const normalizedSub =
      s.toLowerCase() === "exo"
        ? "Cluster"
        : s.toLowerCase() === "on-device"
          ? "Standard"
          : s;
    const savedTab = sessionStorage.getItem("otto:settings:tab");
    const savedSub = sessionStorage.getItem("otto:settings:llmSubTab");
    const tab: Tab = (TABS as readonly string[]).includes(t ?? "")
      ? (t as Tab)
        : (TABS as readonly string[]).includes(savedTab ?? "")
        ? (savedTab as Tab)
        : "LLM";
    const sub: LlmSubTab = (LLM_SUBTABS as readonly string[]).includes(normalizedSub)
      ? (normalizedSub as LlmSubTab)
      : (LLM_SUBTABS as readonly string[]).includes(savedSub ?? "")
        ? (savedSub as LlmSubTab)
        : "Model Provider";
    return { initialTab: tab, initialSub: sub };
  }, [location.search]);
  const [tab, setTab] = useState<Tab>(initialTab);
  const [llmSubTab, setLlmSubTab] = useState<LlmSubTab>(initialSub);
  const [appVersion, setAppVersion] = useState<string>("");

  useEffect(() => {
    getVersion().then(setAppVersion).catch(() => setAppVersion(""));
  }, []);

  useEffect(() => {
    setTab(initialTab);
    setLlmSubTab(initialSub);
  }, [initialTab, initialSub]);

  useEffect(() => { sessionStorage.setItem("otto:settings:tab", tab); }, [tab]);
  useEffect(() => { sessionStorage.setItem("otto:settings:llmSubTab", llmSubTab); }, [llmSubTab]);
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_SETTINGS);
  const [saveStatus, setSaveStatus] = useState<SaveStatus>("idle");
  const [testResult, setTestResult] = useState<{ success: boolean; message: string } | null>(null);
  const [availableModels, setAvailableModels] = useState<{ id: string; name: string }[]>([]);
  const [fetchingModels, setFetchingModels] = useState(false);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [mlxHubDefaultPath, setMlxHubDefaultPath] = useState("");
  const [mlxHubCacheRoot, setMlxHubCacheRoot] = useState("");
  const [mlxHubDefaultSuffix, setMlxHubDefaultSuffix] = useState("huggingface/hub");
  const [mlxLocalModels, setMlxLocalModels] = useState<{ repo_id: string; name: string; size_mb: number }[]>([]);
  const [mlxListError, setMlxListError] = useState<string | null>(null);
  const [mlxListLoading, setMlxListLoading] = useState(false);
  const [mlxSettingsLoaded, setMlxSettingsLoaded] = useState(false);
  const [mlxListFetched, setMlxListFetched] = useState(false);
  const [exoUnloading, setExoUnloading] = useState(false);
  const [exoUnloadMsg, setExoUnloadMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [exoBusy, setExoBusy] = useState<string | null>(null);
  const [exoErr, setExoErr] = useState<string | null>(null);
  const [providerSwitching, setProviderSwitching] = useState(false);
  const [switchingStatus, setSwitchingStatus] = useState<string | null>(null);
  const [switchError, setSwitchError] = useState<string | null>(null);
  // Per-operation notices: key → { kind, message }. Used to show inline
  // "Starting…" / "Started ✓" / error messages on each remote row.
  const [exoOpNotices, setExoOpNotices] = useState<
    Record<string, { kind: "progress" | "success" | "error"; message: string }>
  >({});
  // Live job tracking for remote "up" operations — alias → job
  const [remoteUpJobs, setRemoteUpJobs] = useState<Record<string, ExoJob>>({});
  const remoteUpJobsRef = useRef<Record<string, ExoJob>>({});
  const [exoConfirmRemove, setExoConfirmRemove] = useState<string | null>(null);
  const [exoShowAddRemote, setExoShowAddRemote] = useState(false);
  // "alias": classic add-by-existing-SSH-alias flow.
  // "wizard": from-scratch setup wizard (probe → key install → ssh-config).
  const [exoAddMode, setExoAddMode] = useState<"alias" | "wizard">("alias");
  const [exoAddAlias, setExoAddAlias] = useState("");
  const [exoAddUser, setExoAddUser] = useState("");
  const [exoAddLabel, setExoAddLabel] = useState("");
  const [exoAddDataDir, setExoAddDataDir] = useState("");
  const [exoSshHosts, setExoSshHosts] = useState<SshConfigHost[]>([]);
  const [exoLanHosts, setExoLanHosts] = useState<LanSshHost[]>([]);
  const [exoLanScanning, setExoLanScanning] = useState(false);
  const [exoLanScanned, setExoLanScanned] = useState(false);
  const [exoSshTesting, setExoSshTesting] = useState(false);
  const [exoSshTestResult, setExoSshTestResult] = useState<
    { ok: boolean; hint: string; stderr: string } | null
  >(null);
  const [exoCatalog, setExoCatalog] = useState<ExoCatalogModel[]>([]);
  const [exoCatalogLoading, setExoCatalogLoading] = useState(false);
  const [exoCatalogErr, setExoCatalogErr] = useState<string | null>(null);
  const [exoCatalogReachable, setExoCatalogReachable] = useState(false);
  // Inline "download a new model" picker on the Model sub-tab — pulls
  // from the catalog filtered to ``!downloaded`` so the user can grab a
  // model without leaving Settings.
  const [exoSubTab, setExoSubTab] = useState<ExoSubTab>(() => {
    const saved = sessionStorage.getItem("otto:settings:exoSubTab");
    return (EXO_SUBTABS as readonly string[]).includes(saved ?? "") ? (saved as ExoSubTab) : "Cluster";
  });
  useEffect(() => { sessionStorage.setItem("otto:settings:exoSubTab", exoSubTab); }, [exoSubTab]);
  const [standardSubTab, setStandardSubTab] = useState<StandardSubTab>(() => {
    const saved = sessionStorage.getItem("otto:settings:standardSubTab");
    return (STANDARD_SUBTABS as readonly string[]).includes(saved ?? "") ? (saved as StandardSubTab) : "Overview";
  });
  useEffect(() => { sessionStorage.setItem("otto:settings:standardSubTab", standardSubTab); }, [standardSubTab]);
  const [turboSubTab, setTurboSubTab] = useState<"Overview" | "Model">("Overview");
  // Live cluster status — used to derive a per-remote "in cluster" badge
  // and a header summary on the Overview pill. Polls every 4 s while the
  // user is on the EXO sub-tab and EXO is enabled.
  const [exoLiveStatus, setExoLiveStatus] = useState<ExoStatus | null>(null);

  const loadedRef = useRef(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout>>();
  const savedTimerRef = useRef<ReturnType<typeof setTimeout>>();
  const anthAutoKeyRef = useRef("");
  const modelCacheRef = useRef<Record<string, string>>({});

  useEffect(() => {
    api.getSettings()
      .then((s) => {
        const mlx = { ...DEFAULT_SETTINGS.llm.mlx, ...(s.llm?.mlx ?? {}) };
        const orch: OrchestratorConfig = {
          ...DEFAULT_SETTINGS.orchestrator,
          ...(s.orchestrator ?? {}),
        };
        const merged: AppSettings = {
          ...DEFAULT_SETTINGS,
          ...s,
          orchestrator: orch,
          llm: {
            ...DEFAULT_SETTINGS.llm,
            ...(s.llm ?? {}),
            anthropic: { ...DEFAULT_SETTINGS.llm.anthropic, ...(s.llm?.anthropic ?? {}) },
            mlx: { ...mlx, mlx_bookmarks: mlx.mlx_bookmarks ?? [] },
          },
        };
        setSettings(merged);
        loadedRef.current = true;
        setMlxSettingsLoaded(true);
      })
      .catch((e) => console.warn("Failed to load settings:", e));
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
    if (!loadedRef.current) return;
    clearTimeout(debounceRef.current);
    clearTimeout(savedTimerRef.current);
    debounceRef.current = setTimeout(() => persistSettings(settings), AUTO_SAVE_DELAY_MS);
    return () => clearTimeout(debounceRef.current);
  }, [settings, persistSettings]);

  const pickBestModel = (models: { id: string; name: string }[]) => {
    const sonnet46 = models.find((m) => /sonnet.*4[._-]?6/i.test(m.id) || /sonnet.*4[._-]?6/i.test(m.name));
    return sonnet46?.id || models[0]?.id || "";
  };


  const handleFetchAnthropicModels = async () => {
    setFetchingModels(true); setModelsError(null);
    try {
      const a = settings.llm.anthropic;
      const result = await api.listModels({
        provider: "anthropic",
        api_key: a.api_key,
        model_name: a.model_name,
        model_provider: a.model_provider,
        bedrock_region: a.bedrock_region,
        bedrock_auth_mode: a.bedrock_auth_mode,
        aws_access_key_id: a.aws_access_key_id,
        aws_secret_access_key: a.aws_secret_access_key,
      });
      if (result.error) { setModelsError(result.error); }
      const models = result.models || [];
      setAvailableModels(models);
      if (models.length > 0 && settings.llm.provider === "anthropic") {
        const currentInList = a.model_name && models.some((m) => m.id === a.model_name);
        if (!currentInList) {
          const cached = modelCacheRef.current[cacheKey(a)];
          const cachedInList = cached && models.some((m) => m.id === cached);
          updateAnthropicField("model_name", cachedInList ? cached : pickBestModel(models));
        }
      }
    } catch (e) { setModelsError(e instanceof Error ? e.message : "Failed to fetch models"); } finally { setFetchingModels(false); }
  };

  const handleFetchOpenAIModels = async () => {
    setFetchingModels(true); setModelsError(null);
    try {
      const o = settings.llm.openai;
      const result = await api.listModels({
        provider: "openai",
        api_key: o.model_provider === "azure" ? o.azure_api_key : o.api_key,
        model_name: o.model_name,
        openai_model_provider: o.model_provider,
        azure_endpoint: o.azure_endpoint,
        azure_api_version: o.azure_api_version,
        azure_deployment: o.azure_deployment,
      });
      if (result.error) { setModelsError(result.error); }
      const models = result.models || [];
      setAvailableModels(models);
      if (models.length > 0 && !models.some((m) => m.id === o.model_name)) {
        updateOpenAIField("model_name", models[0]?.id || o.model_name);
      }
    } catch (e) { setModelsError(e instanceof Error ? e.message : "Failed to fetch models"); } finally { setFetchingModels(false); }
  };


  const resetModels = () => {
    setAvailableModels([]);
    setModelsError(null);
    setTestResult(null);
  };

  const cacheKey = (a: { model_provider: string; bedrock_auth_mode: string }) =>
    `${a.model_provider}:${a.bedrock_auth_mode}`;

  const switchProviderMode = (field: string, value: string) => {
    const a = settings.llm.anthropic;
    if (a.model_name) {
      modelCacheRef.current[cacheKey(a)] = a.model_name;
    }
    updateAnthropicField(field, value);
    resetModels();
    anthAutoKeyRef.current = "";
  };

  useEffect(() => {
    if (!loadedRef.current) return;
    const a = settings.llm.anthropic;
    const key = `${a.model_provider}:${a.bedrock_auth_mode}`;
    if (key === anthAutoKeyRef.current) return;
    anthAutoKeyRef.current = key;
    void handleFetchAnthropicModels();
  }, [settings.llm.anthropic.model_provider, settings.llm.anthropic.bedrock_auth_mode]); // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-fetch models whenever the user switches the top-level provider to "anthropic".
  // The existing effect above only watches Anthropic sub-fields; this one catches the
  // provider flip (e.g. mlx → anthropic) where those sub-fields haven't changed.
  useEffect(() => {
    if (!loadedRef.current) return;
    if (settings.llm.provider !== "anthropic") return;
    const a = settings.llm.anthropic;
    const key = `${a.model_provider}:${a.bedrock_auth_mode}`;
    if (key === anthAutoKeyRef.current) return;
    anthAutoKeyRef.current = key;
    void handleFetchAnthropicModels();
  }, [settings.llm.provider]); // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-fetch models whenever the user switches the top-level provider to "openai"
  // or changes the OpenAI model_provider (native ↔ azure).
  useEffect(() => {
    if (!loadedRef.current) return;
    if (settings.llm.provider !== "openai") return;
    void handleFetchOpenAIModels();
  }, [settings.llm.provider, settings.llm.openai.model_provider]); // eslint-disable-line react-hooks/exhaustive-deps


  const updateAnthropicField = (field: string, value: string | number | boolean) => {
    setSettings((s) => ({ ...s, llm: { ...s.llm, anthropic: { ...s.llm.anthropic, [field]: value } } }));
  };

  const updateOpenAIField = (field: keyof OpenAIConfig, value: string | number | boolean) => {
    setSettings((s) => ({ ...s, llm: { ...s.llm, openai: { ...s.llm.openai, [field]: value } } }));
  };

  const updateMlxField = (field: string, value: string) => {
    setSettings((s) => ({ ...s, llm: { ...s.llm, mlx: { ...s.llm.mlx, [field]: value } } }));
  };

  // onPatch handler for the shared ClusterSetupFlow (settings variant). Merges
  // the patch into local state (the debounced autosave then persists it) and
  // also persists immediately so reads-after-write (e.g. fresh remotes) are
  // durable.
  const handleClusterPatch = useCallback(
    async (patch: Partial<AppSettings>) => {
      const next = {
        ...settings,
        ...patch,
        llm: { ...settings.llm, ...(patch.llm ?? {}) },
        exo: patch.exo ? { ...settings.exo, ...patch.exo } : settings.exo,
      } as AppSettings;
      setSettings(next);
      await persistSettings(next);
    },
    [settings, persistSettings],
  );

  const updateExoPartial = (partial: Partial<ExoConfig>) => {
    setSettings((s) => ({ ...s, exo: { ...s.exo, ...partial } }));
  };

  /**
   * Switch the active LLM provider with automatic unload/load of on-device models.
   *
   * 1. Unloads the current on-device provider's model (MLX/oMLX/Exo) — skips
   *    if it's a frontier provider (Anthropic/OpenAI) or same provider.
   * 2. Updates the provider in settings state.
   * 3. Auto-loads the incoming on-device provider's configured model.
   *
   * oMLX and Exo load are polled to a terminal state so the UI can show
   * progress. MLX loads lazily on the first chat session so no HTTP call is
   * needed here.
   */
  const handleProviderSwitch = useCallback(
    async (nextRaw: string, currentSettings: AppSettings) => {
      const next = nextRaw === "frontier" ? "anthropic" : nextRaw;
      const prev = currentSettings.llm.provider;

      setSwitchError(null);
      setSettings((s) => ({
        ...s,
        llm: { ...s.llm, provider: next },
        // Picking a local server as the active provider implies it should be
        // enabled so warnings, agent tools, and auto-start stay consistent.
        ...(next === "omlx" ? { omlx: { ...s.omlx, enabled: true } } : {}),
        ...(next === "exo" ? { exo: { ...s.exo, enabled: true } } : {}),
      }));
      resetModels();
      anthAutoKeyRef.current = "";

      const prevOnDevice = ["mlx", "omlx", "exo"].includes(prev);
      const nextOnDevice = ["mlx", "omlx", "exo"].includes(next);

      if (!prevOnDevice && !nextOnDevice) return;
      if (prev === next) return;

      setProviderSwitching(true);

      try {
        // ── Step 1: unload outgoing model, then stop its server ───────────
        if (prevOnDevice) {
          try {
            if (prev === "mlx") {
              setSwitchingStatus("Unloading MLX model…");
              await api.mlxUnload();
            } else if (prev === "omlx") {
              const modelId = currentSettings.omlx?.model_name?.trim() ?? "";
              if (modelId) {
                setSwitchingStatus(`Unloading oMLX model "${modelId}"…`);
                const { job_id } = await api.omlxUnloadModel(modelId);
                // Poll until done or error (max 30 s).
                const deadline = Date.now() + 30_000;
                while (Date.now() < deadline) {
                  await new Promise((r) => setTimeout(r, 1_000));
                  const job = await api.getOmlxJob(job_id);
                  if (job.status === "done" || job.status === "error") break;
                }
              }
              // Stop the oMLX server now that we're leaving Turbo.
              setSwitchingStatus("Stopping oMLX server…");
              await stopOmlxServerAndWait();
            } else if (prev === "exo") {
              const modelId = currentSettings.exo?.model_name?.trim() ?? "";
              if (modelId) {
                setSwitchingStatus(`Unloading Exo model "${modelId}"…`);
                await api.exoUnload(modelId);
              }
              // Stop the local Exo daemon now that we're leaving Cluster.
              setSwitchingStatus("Stopping Exo cluster…");
              await api.exoDown();
            }
          } catch (e) {
            // Unload / stop failure is non-fatal — log and continue.
            console.warn("Provider switch: unload/stop failed (continuing):", e);
          }
        }

        // ── Step 2: start incoming server (if stopped), then load model ───
        if (nextOnDevice && next !== "mlx") {
          if (next === "omlx") {
            // Auto-start the oMLX server if it isn't already reachable.
            try {
              const status = await api.omlxStatus().catch(() => null);
              if (!status?.reachable) {
                setSwitchingStatus("Starting oMLX server…");
                await startOmlxServerAndWait();
              }
            } catch (e) {
              const msg = e instanceof Error ? e.message : "Failed to start the oMLX server.";
              setSwitchError(msg);
              return;
            }
            const modelId = currentSettings.omlx?.model_name?.trim() ?? "";
            if (modelId) {
              try {
                setSwitchingStatus(`Loading oMLX model "${modelId}"…`);
                await loadOmlxModelAndWait(modelId, { timeoutMs: 600_000 });
              } catch (e) {
                const msg = e instanceof Error ? e.message : "Failed to load model";
                setSwitchError(msg);
              }
            }
          } else if (next === "exo") {
            // Auto-start the Exo cluster if it isn't already reachable.
            try {
              const status = await api.exoStatus().catch(() => null);
              if (!status?.reachable) {
                setSwitchingStatus("Starting Exo cluster…");
                await startExoServerAndWait((line) =>
                  setSwitchingStatus(`Starting Exo cluster… ${line}`),
                );
              }
            } catch (e) {
              const msg = e instanceof Error ? e.message : "Failed to start the Exo cluster.";
              setSwitchError(msg);
              return;
            }
            const modelId = currentSettings.exo?.model_name?.trim() ?? "";
            if (modelId) {
              try {
                setSwitchingStatus(`Loading Exo model "${modelId}"…`);
                await api.exoPreloadModel(modelId);
              } catch (e) {
                const msg = e instanceof Error ? e.message : "Failed to load model";
                setSwitchError(msg);
              }
            }
          }
        }
        // MLX loads lazily on first session — no HTTP call needed.
      } finally {
        setProviderSwitching(false);
        setSwitchingStatus(null);
      }
    },
    [], // eslint-disable-line react-hooks/exhaustive-deps
  );

  const refreshExoRemotes = useCallback(async () => {
    try {
      const r = await api.listExoRemotes();
      setSettings((s) => ({ ...s, exo: { ...s.exo, remotes: r.remotes } }));
    } catch (e) {
      setExoErr(e instanceof Error ? e.message : "Failed to load remotes");
    }
  }, []);

  const setOpNotice = useCallback(
    (key: string, kind: "progress" | "success" | "error" | null, message = "") => {
      setExoOpNotices((prev) => {
        if (kind === null) {
          const next = { ...prev };
          delete next[key];
          return next;
        }
        return { ...prev, [key]: { kind, message } };
      });
    },
    [],
  );

  const runExoOp = useCallback(
    async (key: string, fn: () => Promise<void>, progressMsg?: string, successMsg?: string) => {
      setExoErr(null);
      setExoBusy(key);
      if (progressMsg) setOpNotice(key, "progress", progressMsg);
      try {
        await fn();
        if (successMsg) {
          setOpNotice(key, "success", successMsg);
          // Auto-clear the success notice after 4 s.
          setTimeout(() => setOpNotice(key, null), 4000);
        } else if (progressMsg) {
          setOpNotice(key, null);
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : `${key} failed`;
        setExoErr(msg);
        if (progressMsg) setOpNotice(key, "error", msg);
      } finally {
        setExoBusy(null);
      }
    },
    [setOpNotice],
  );

  // Poll in-flight remote "up" jobs every 2 s so the progress card stays live.
  useEffect(() => {
    const activeAliases = Object.keys(remoteUpJobs).filter(
      (a) => remoteUpJobs[a].status === "running" || remoteUpJobs[a].status === "pending",
    );
    if (activeAliases.length === 0) return;
    const t = setInterval(async () => {
      const updates = await Promise.all(
        activeAliases.map((a) => api.getExoJob(remoteUpJobs[a].id).catch(() => null)),
      );
      setRemoteUpJobs((prev) => {
        const next = { ...prev };
        updates.forEach((j, i) => {
          if (!j) return;
          next[activeAliases[i]] = j;
          remoteUpJobsRef.current[activeAliases[i]] = j;
        });
        return next;
      });
    }, 2000);
    return () => clearInterval(t);
  }, [remoteUpJobs]);

  const handleExoAddRemote = () => {
    const rawAlias = exoAddAlias.trim();
    if (!rawAlias) return;
    const user = exoAddUser.trim();
    // If a username is set and the alias doesn't already embed one (user@host),
    // prefix it so SSH connects as the right user.
    const alias = user && !rawAlias.includes("@") ? `${user}@${rawAlias}` : rawAlias;
    void runExoOp("add", async () => {
      const res = await api.addExoRemote({
        ssh_alias: alias,
        label: exoAddLabel.trim(),
        app_data_dir: exoAddDataDir.trim(),
        enabled: true,
      });
      setSettings((s) => ({ ...s, exo: { ...s.exo, remotes: res.remotes } }));
      setExoAddAlias("");
      setExoAddUser("");
      setExoAddLabel("");
      setExoAddDataDir("");
      setExoShowAddRemote(false);
    });
  };

  // Called from <ExoSetupSteps onComplete>. The wizard already created
  // the keypair, installed the public key, and wrote the ~/.ssh/config
  // block, so all we have left to do here is persist the new remote in
  // ExoConfig.remotes and refresh the list.
  const handleExoSetupComplete = useCallback(async (r: ExoSetupCompletion) => {
    await runExoOp("add", async () => {
      const res = await api.addExoRemote({
        ssh_alias: r.ssh_alias,
        label: r.label || "",
        app_data_dir: "",
        enabled: true,
      });
      // Backend currently doesn't read identity_file from the create
      // payload (kept simple), so apply it as a separate update so the
      // wizard's choice is persisted in ExoRemoteConfig for future
      // diagnostics.
      if (r.identity_file) {
        try {
          await api.updateExoRemote(r.ssh_alias, {
            identity_file: r.identity_file,
          } as unknown as Partial<{ label: string; app_data_dir: string; enabled: boolean }>);
        } catch {
          // Non-fatal — the alias works regardless of whether the
          // identity_file metadata is recorded.
        }
      }
      setSettings((s) => ({ ...s, exo: { ...s.exo, remotes: res.remotes } }));
      setExoShowAddRemote(false);
      setExoAddMode("alias");
      await refreshExoRemotes();
    });
  }, [runExoOp, refreshExoRemotes]);

  const handleExoRemoveRemote = (alias: string) =>
    runExoOp(`remove-${alias}`, async () => {
      await api.removeExoRemote(alias);
      setExoConfirmRemove(null);
      await refreshExoRemotes();
    });

  const handleExoToggleRemote = (alias: string, enabled: boolean) =>
    runExoOp(`toggle-${alias}`, async () => {
      await api.updateExoRemote(alias, { enabled });
      await refreshExoRemotes();
    });

  const handleExoRemoteUp = (alias: string) => {
    setExoErr(null);
    setExoBusy(`up-${alias}`);
    void (async () => {
      try {
        const job = await api.exoRemoteUp(alias, false);
        setRemoteUpJobs((prev) => {
          const next = { ...prev, [alias]: job };
          remoteUpJobsRef.current = next;
          return next;
        });
      } catch (e) {
        const msg = e instanceof Error ? e.message : "Start failed";
        setExoErr(msg);
        setOpNotice(`up-${alias}`, "error", msg);
      } finally {
        setExoBusy(null);
      }
    })();
  };

  const handleExoRemoteDown = (alias: string) =>
    runExoOp(
      `down-${alias}`,
      async () => { await api.exoRemoteDown(alias); },
      `Stopping ${alias}…`,
      `${alias} stopped`,
    );

  const refreshExoSshHosts = useCallback(async () => {
    try {
      const r = await api.exoDiscoverSshConfig();
      setExoSshHosts(r.hosts);
    } catch (e) {
      setExoErr(e instanceof Error ? e.message : "Failed to read ~/.ssh/config");
    }
  }, []);

  const handleExoLanScan = () => {
    if (exoLanScanning) return;
    setExoLanScanning(true);
    setExoErr(null);
    void (async () => {
      try {
        const r = await api.exoDiscoverLan(3.0);
        setExoLanHosts(r.hosts);
        setExoLanScanned(true);
      } catch (e) {
        setExoErr(e instanceof Error ? e.message : "LAN scan failed");
      } finally {
        setExoLanScanning(false);
      }
    })();
  };

  const handleExoPickSshHost = (h: SshConfigHost) => {
    setExoAddAlias(h.alias);
    if (!exoAddLabel.trim() && h.hostname && h.hostname !== h.alias) {
      setExoAddLabel(h.hostname);
    }
    setExoSshTestResult(null);
  };

  const handleExoPickLanHost = (h: LanSshHost) => {
    if (h.matches_alias) {
      setExoAddAlias(h.matches_alias);
      // Alias already in ~/.ssh/config — user is encoded there; clear the field.
      setExoAddUser("");
    } else {
      // No matching ~/.ssh/config entry. Preference order:
      //   1. Thunderbolt-bridge IP (dedicated point-to-point link —
      //      orders of magnitude faster than Wi-Fi, never NAT'd).
      //   2. First IPv4 — far more deterministic than ``.local``,
      //      which can collide and get suffixed (``foo-3.local``)
      //      or fail outright when mDNS is in a bad state.
      //   3. First IPv6.
      //   4. The ``.local`` hostname as a last resort.
      const tb = (h.thunderbolt_addresses || [])[0];
      const ipv4 = (h.addresses || []).find((a) => /^\d+\.\d+\.\d+\.\d+$/.test(a));
      const ipv6 = (h.addresses || []).find((a) => a.includes(":"));
      setExoAddAlias(tb || ipv4 || ipv6 || h.hostname || `${h.name}.local`);
    }
    if (!exoAddLabel.trim()) setExoAddLabel(h.name);
    setExoSshTestResult(null);
  };

  const handleExoTestSsh = () => {
    const alias = exoAddAlias.trim();
    if (!alias || exoSshTesting) return;
    setExoSshTesting(true);
    setExoSshTestResult(null);
    void (async () => {
      try {
        const r = await api.exoTestSsh(alias, 6.0);
        setExoSshTestResult({ ok: r.ok, hint: r.hint, stderr: r.stderr });
      } catch (e) {
        setExoSshTestResult({
          ok: false,
          hint: e instanceof Error ? e.message : "Test failed",
          stderr: "",
        });
      } finally {
        setExoSshTesting(false);
      }
    })();
  };

  const updateMlxPartial = (partial: Partial<MlxHfConfig>) => {
    setSettings((s) => ({ ...s, llm: { ...s.llm, mlx: { ...s.llm.mlx, ...partial } } }));
  };

  const handleExoUnload = useCallback(async () => {
    const modelId = settings.exo.model_name?.trim() ?? "";
    if (!modelId) return;
    setExoUnloading(true);
    setExoUnloadMsg(null);
    try {
      const res = await api.exoUnload(modelId);
      setExoUnloadMsg({ ok: true, text: `Unloaded — ${res.instances_removed} instance${res.instances_removed === 1 ? "" : "s"} removed.` });
    } catch (e) {
      setExoUnloadMsg({ ok: false, text: e instanceof Error ? e.message : String(e) });
    } finally {
      setExoUnloading(false);
      setTimeout(() => setExoUnloadMsg(null), 4000);
    }
  }, [settings.exo.model_name]);

  const refreshMlxModels = useCallback(async () => {
    setMlxListLoading(true);
    setMlxListError(null);
    try {
      const cd = settings.llm.mlx.hf_hub_cache?.trim();
      const q = cd ? `?cache_dir=${encodeURIComponent(cd)}` : "";
      const r = await api.mlxLocalModels(q);
      if (r.error) setMlxListError(r.error);
      setMlxLocalModels(r.models || []);
      setMlxListFetched(true);
    } catch (e) {
      setMlxListError(e instanceof Error ? e.message : "Failed to list models");
      setMlxLocalModels([]);
    } finally {
      setMlxListLoading(false);
    }
  }, [settings.llm.mlx.hf_hub_cache]);

  const refreshExoCatalog = useCallback(async () => {
    setExoCatalogLoading(true);
    setExoCatalogErr(null);
    try {
      const r = await api.exoModels();
      setExoCatalogReachable(r.reachable);
      setExoCatalog(r.models || []);
      if (!r.reachable) setExoCatalogErr(r.error || "Cluster offline");
    } catch (e) {
      setExoCatalogErr(
        e instanceof Error ? e.message : "Failed to list cluster models",
      );
      setExoCatalog([]);
      setExoCatalogReachable(false);
    } finally {
      setExoCatalogLoading(false);
    }
  }, []);

  // Whether an auto-load job is currently running. Prevents duplicate
  // starts from the dropdown AND from the page-load effect below.
  const exoAutoPreloadInFlightRef = useRef(false);

  // Core function: unload any other loaded model, then preload `id`.
  // Only one model should be in cluster memory at a time.
  // Uses min_nodes=1 so it never blocks on a multi-node placement.
  const exoAutoLoad = useCallback(
    async (id: string) => {
      if (exoAutoPreloadInFlightRef.current) return;
      const m = exoCatalog.find((c) => c.id === id);
      if (!m?.downloaded || m.loaded) return;
      exoAutoPreloadInFlightRef.current = true;
      const others = exoCatalog.filter((c) => c.loaded && c.id !== id);
      setTestResult({
        success: true,
        message: others.length > 0
          ? `Unloading '${others.map((c) => c.id).join(", ")}', then loading '${id}'…`
          : `Loading '${id}'…`,
      });
      try {
        for (const prev of others) {
          await api.exoUnload(prev.id);
        }
        const job = await api.exoPreloadStart(id, 1);
        setTestResult({ success: true, message: `Loading '${id}'… (started)` });
        const startedAt = Date.now();
        while (Date.now() - startedAt < 600_000) {
          await new Promise((r) => setTimeout(r, 1500));
          try {
            const j = await api.exoPreloadStatus(job.job_id);
            if (j.status === "done") {
              setTestResult({
                success: true,
                message: `Model '${id}' loaded in ${Math.round(j.elapsed_seconds)}s`,
              });
              void refreshExoCatalog();
              return;
            }
            if (j.status === "error" || j.status === "cancelled") {
              setTestResult({
                success: false,
                message: `Load of '${id}' ${j.status}: ${j.message || "unknown"}`,
              });
              return;
            }
          } catch {
            // transient poll failure — keep trying
          }
        }
      } catch (e) {
        setTestResult({
          success: false,
          message: `Auto-load of '${id}' failed: ${e instanceof Error ? e.message : "unknown"}`,
        });
      } finally {
        exoAutoPreloadInFlightRef.current = false;
      }
    },
    [exoCatalog, refreshExoCatalog],
  );

  // When the user picks a model from the SelectField, update settings
  // and immediately auto-load it if it's downloaded but not yet in memory.
  const handleExoModelSelect = useCallback(
    (modelId: string) => {
      setSettings((s) => ({ ...s, exo: { ...s.exo, model_name: modelId } }));
      const id = modelId?.trim();
      if (id && exoCatalogReachable) void exoAutoLoad(id);
    },
    [exoAutoLoad, exoCatalogReachable],
  );

  const exoModelSelectOptions = useMemo(() => {
    const opts: { value: string; label: string; loaded?: boolean }[] = [
      { value: "", label: "(pick a model)" },
    ];
    for (const m of exoCatalog.filter((m) => m.downloaded)) {
      opts.push({
        value: m.id,
        label: m.id,
        loaded: m.loaded,
      });
    }
    // Keep the currently-selected value even if it's no longer in the
    // downloaded list (e.g. cluster offline and catalog wasn't fetched).
    const cur = settings.exo.model_name?.trim() ?? "";
    if (cur && !exoCatalog.some((m) => m.downloaded && m.id === cur)) {
      opts.push({ value: cur, label: `${cur} (not downloaded)` });
    }
    return opts;
  }, [exoCatalog, settings.exo.model_name]);

  useEffect(() => {
    if (!loadedRef.current) return;
    const onExoSub = tab === "LLM" && llmSubTab === "Cluster";
    if (settings.llm.provider === "exo" || onExoSub) {
      void refreshExoCatalog();
    }
  }, [settings.llm.provider, tab, llmSubTab, refreshExoCatalog]);

  useEffect(() => {
    if (!loadedRef.current) return;
    void (async () => {
      try {
        const d = await api.mlxHubDefault();
        setMlxHubDefaultPath(d.path);
        setMlxHubCacheRoot(d.cache_root || "");
        setMlxHubDefaultSuffix(d.default_suffix || "huggingface/hub");
      } catch {
        setMlxHubDefaultPath("");
      }
      await refreshMlxModels();
    })();
  }, [settings.llm.mlx.hf_hub_cache, refreshMlxModels, mlxSettingsLoaded]);

  useEffect(() => {
    if (!loadedRef.current) return;
    if (tab !== "LLM" || llmSubTab !== "Cluster") return;
    void refreshExoRemotes();
  }, [tab, llmSubTab, refreshExoRemotes]);

  // Poll live cluster status while the user is on the EXO sub-tab. The
  // status drives both the Overview summary and the per-remote
  // "in cluster" badge. We bail out if EXO is disabled (the master URL
  // would just be unreachable). Polling is paused when the window is
  // hidden — see ``usePolling``.
  const exoPollEnabled =
    loadedRef.current && tab === "LLM" && llmSubTab === "Cluster" && settings.exo.enabled;

  useEffect(() => {
    if (!loadedRef.current) return;
    if (tab !== "LLM" || llmSubTab !== "Cluster") return;
    if (!settings.exo.enabled) setExoLiveStatus(null);
  }, [tab, llmSubTab, settings.exo.enabled]);

  usePolling(
    async () => {
      try {
        setExoLiveStatus(await api.exoStatus());
      } catch {
        setExoLiveStatus(null);
      }
    },
    8000,
    exoPollEnabled,
  );

  // Auto-tune ``min_nodes`` (placement width) to track real cluster
  // membership: when a peer joins, bump it by one; when a peer leaves,
  // drop it by one (floor 1). We track node IDs across polls rather
  // than the raw count so that a transient cluster-offline blip doesn't
  // collapse the value to 1 and a subsequent reconnect doesn't double-
  // count returning peers as new joins. ``null`` means "no reachable
  // observation yet" — the first reachable poll only seeds the snapshot
  // without mutating settings.
  const exoSeenNodeIdsRef = useRef<Set<string> | null>(null);

  useEffect(() => {
    if (!loadedRef.current) return;
    if (!settings.exo.enabled) {
      // Disabling EXO clears our snapshot so the next enable starts
      // fresh — otherwise a long-disabled cluster would re-appear with
      // stale "left" deltas waiting to fire.
      exoSeenNodeIdsRef.current = null;
      return;
    }
    if (!exoLiveStatus?.reachable) {
      // Cluster offline. Hold the snapshot so we can compare against
      // the first reachable poll once it comes back, instead of
      // treating every reconnect as a fresh seed.
      return;
    }
    const liveIds = new Set(exoLiveStatus.nodes.map((n) => n.node_id));
    const prev = exoSeenNodeIdsRef.current;
    if (prev === null) {
      exoSeenNodeIdsRef.current = liveIds;
      return;
    }
    let added = 0;
    let removed = 0;
    liveIds.forEach((id) => {
      if (!prev.has(id)) added += 1;
    });
    prev.forEach((id) => {
      if (!liveIds.has(id)) removed += 1;
    });
    if (added === 0 && removed === 0) return;
    exoSeenNodeIdsRef.current = liveIds;
    setSettings((s) => {
      const cur = s.exo.min_nodes ?? 1;
      const next = Math.max(1, cur + added - removed);
      if (next === cur) return s;
      return { ...s, exo: { ...s.exo, min_nodes: next } };
    });
  }, [exoLiveStatus, settings.exo.enabled]);

  // When min_nodes changes (manually or via the auto-tune join/leave
  // effect above), re-place any currently-loaded model so the new node
  // count takes effect immediately — otherwise the change only applies
  // on the next explicit preload.
  //
  // Design choices to avoid the "stacking instances" problem:
  //
  // 1. exoLiveStatus is read via a ref (not in the dep array) so that the
  //    8-second poll doesn't re-fire the effect and accidentally trigger
  //    a duplicate place_instance call.
  // 2. A 900ms debounce coalesces rapid slider / keyboard changes into a
  //    single call (the debounce timer is longer than the auto-save at
  //    800ms so the persisted value is already settled when we fire).
  // 3. lastPlacedMinNodesRef tracks the min_nodes value we most recently
  //    sent to exo — we skip any call where next === lastPlaced, so a
  //    join/leave auto-tune that ends up back at the same value is a
  //    no-op, and a component re-mount after an in-flight place can't
  //    re-fire the same value.
  const exoLiveStatusRef = useRef<typeof exoLiveStatus>(exoLiveStatus);
  exoLiveStatusRef.current = exoLiveStatus;

  const lastPlacedMinNodesRef = useRef<number | null>(null);
  const minNodesDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const minNodesReplaceInFlightRef = useRef(false);

  useEffect(() => {
    const next = settings.exo.min_nodes ?? 1;

    // Skip if we already sent this exact value to exo.
    if (lastPlacedMinNodesRef.current === next) return;
    if (!loadedRef.current) return;
    if (!settings.exo.enabled) return;

    // Debounce: clear any pending timer and wait for the value to settle.
    if (minNodesDebounceRef.current !== null) {
      clearTimeout(minNodesDebounceRef.current);
    }
    minNodesDebounceRef.current = setTimeout(async () => {
      minNodesDebounceRef.current = null;

      const status = exoLiveStatusRef.current;
      if (!status?.reachable) return;
      if (minNodesReplaceInFlightRef.current) return;

      const settled = settings.exo.min_nodes ?? 1;
      // Guard again after the debounce delay in case it changed further.
      if (lastPlacedMinNodesRef.current === settled) return;

      const modelId =
        (status.loaded_models ?? [])[0]?.trim() ||
        settings.exo.model_name?.trim();
      if (!modelId) return;

      lastPlacedMinNodesRef.current = settled;
      minNodesReplaceInFlightRef.current = true;
      setTestResult({
        success: true,
        message: `Re-placing '${modelId}' across ${settled} node${settled === 1 ? "" : "s"}…`,
      });
      try {
        const r = await api.exoPreloadModel(modelId, settled);
        const replacedNote =
          r.replaced && r.replaced > 0
            ? ` (replaced ${r.replaced} prior instance${r.replaced === 1 ? "" : "s"})`
            : "";
        setTestResult({
          success: r.ok,
          message: r.ok
            ? `'${modelId}' placed across ${settled} node${settled === 1 ? "" : "s"} in ${r.elapsed_seconds}s${replacedNote}`
            : `Re-place failed: ${r.error || "unknown error"}`,
        });
        if (r.ok) void refreshExoCatalog();
      } catch (e) {
        setTestResult({
          success: false,
          message: `Re-place failed: ${e instanceof Error ? e.message : "unknown error"}`,
        });
        // Reset so a retry is possible.
        lastPlacedMinNodesRef.current = null;
      } finally {
        minNodesReplaceInFlightRef.current = false;
      }
    }, 900);

    return () => {
      if (minNodesDebounceRef.current !== null) {
        clearTimeout(minNodesDebounceRef.current);
        minNodesDebounceRef.current = null;
      }
    };
  }, [settings.exo.min_nodes, settings.exo.enabled, settings.exo.model_name, refreshExoCatalog]);

  // One-shot: when the page loads (or provider switches to "exo") and
  // the catalog becomes available, auto-load the configured model if
  // it is downloaded but not yet in memory. This is a one-time check;
  // subsequent selections are handled directly in handleExoModelSelect.
  const pageLoadAutoLoadDoneRef = useRef(false);
  useEffect(() => {
    if (settings.llm.provider !== "exo") { pageLoadAutoLoadDoneRef.current = false; return; }
    if (!exoCatalogReachable) return;
    if (!loadedRef.current) return;
    if (pageLoadAutoLoadDoneRef.current) return;
    const id = settings.exo.model_name?.trim();
    if (!id) return;
    pageLoadAutoLoadDoneRef.current = true;
    void exoAutoLoad(id);
  }, [settings.llm.provider, settings.exo.model_name, exoCatalogReachable, exoAutoLoad]);

  // Best-effort match between a remote (SSH alias / label) and a peer
  // reported by the master. exo's friendlyName comes from the remote
  // mac's computer name so it usually contains the hostname or alias.
  const matchPeer = useCallback(
    (remote: ExoRemote, nodes: ExoNodeInfo[]): ExoNodeInfo | null => {
      const candidates: string[] = [
        remote.ssh_alias,
        remote.label,
        // strip trailing .local for hostname matches
        remote.ssh_alias.replace(/\.local$/i, ""),
      ]
        .map((s) => (s || "").trim().toLowerCase())
        .filter(Boolean);
      for (const n of nodes) {
        const fn = (n.friendly_name || "").toLowerCase();
        if (!fn) continue;
        if (candidates.some((c) => fn.includes(c) || c.includes(fn))) {
          return n;
        }
      }
      return null;
    },
    [],
  );

  /**
   * Full peer assignment: ssh_alias → ExoNodeInfo.
   *
   * Pass 1 — name-based: try matching each enabled remote to a peer whose
   *   friendly_name contains (or is contained by) the alias/label.
   * Pass 2 — deduction: if exactly ONE enabled remote is still unmatched
   *   AND exactly ONE non-master peer is still unclaimed, assign them. This
   *   handles the common case where the SSH alias is an IP address (e.g.
   *   169-254-180-115 via Thunderbolt) that bears no resemblance to the
   *   EXO friendly name reported by the remote (e.g. "Eugene's MacBook Air").
   */
  const remotePeerMap = useMemo<Map<string, ExoNodeInfo>>(() => {
    const map = new Map<string, ExoNodeInfo>();
    if (!exoLiveStatus?.reachable) return map;

    // Non-master peers only — the master is the local machine.
    const peers = exoLiveStatus.nodes.filter(
      (n) => n.node_id !== exoLiveStatus.master_node_id,
    );

    const claimed = new Set<string>();

    // Pass 1: name-based matching.
    for (const r of settings.exo.remotes) {
      if (!r.enabled) continue;
      const match = matchPeer(r, peers.filter((p) => !claimed.has(p.node_id)));
      if (match) {
        claimed.add(match.node_id);
        map.set(r.ssh_alias, match);
      }
    }

    // Pass 2: deduction.
    const unmatched = settings.exo.remotes.filter(
      (r) => r.enabled && !map.has(r.ssh_alias),
    );
    const unclaimed = peers.filter((p) => !claimed.has(p.node_id));
    if (unmatched.length === 1 && unclaimed.length === 1) {
      map.set(unmatched[0].ssh_alias, unclaimed[0]);
    }

    return map;
  }, [exoLiveStatus, settings.exo.remotes, matchPeer]);

  useEffect(() => {
    if (!exoShowAddRemote) return;
    void refreshExoSshHosts();
  }, [exoShowAddRemote, refreshExoSshHosts]);

  return (
    <div className="h-full flex flex-col">
      <header className="border-b border-th-border px-6 py-4 flex items-center justify-between shrink-0 bg-th-bg-secondary">
        <h1 className="text-lg font-bold text-th-text-primary">Settings</h1>
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={async () => {
              try {
                await api.setupReset();
                window.location.reload();
              } catch (e) {
                console.warn("setup reset failed", e);
              }
            }}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg bg-gradient-to-r from-blue-500/15 to-sky-500/15 border border-blue-500/30 text-blue-400 hover:from-blue-500/25 hover:to-sky-500/25 hover:border-blue-500/50 hover:text-blue-300 transition-all duration-150 shadow-sm"
            title="Re-open the setup wizard"
          >
            <Wand2 size={12} className="shrink-0" /> Setup wizard
          </button>
          <div className="flex items-center gap-1.5 text-xs font-medium min-w-[80px] justify-end">
            {saveStatus === "saving" && <><Loader2 size={12} className="animate-spin text-th-text-muted" /><span className="text-th-text-muted">Saving…</span></>}
            {saveStatus === "saved" && <><Check size={12} className="text-emerald-400" /><span className="text-emerald-400">Saved</span></>}
            {saveStatus === "error" && <><XCircle size={12} className="text-red-400" /><span className="text-red-400">Save failed</span></>}
          </div>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto p-6">
        <div className="flex gap-1 mb-6 bg-th-inset-bg rounded-xl p-1 w-fit border border-th-border">
          {TABS.map((t) => (
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

        {tab === "LLM" && (
          <div className="space-y-6 max-w-5xl">
            <div className="flex gap-1 bg-th-inset-bg rounded-xl p-1 w-fit border border-th-border">
              {LLM_SUBTABS.map((t) => (
                <button
                  key={t}
                  type="button"
                  className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all duration-150 ${
                    llmSubTab === t
                      ? "bg-th-tab-active-bg text-th-tab-active-fg shadow-sm"
                      : "text-th-text-tertiary hover:text-th-text-primary hover:bg-th-surface-hover"
                  }`}
                  onClick={() => setLlmSubTab(t)}
                >
                  {t}
                </button>
              ))}
            </div>

            {llmSubTab === "Model Provider" && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-stretch">
            <Card title="Default model" dot="bg-emerald-500">
              <div className="space-y-4">
                <p className="text-xs text-th-text-tertiary">
                  Primary stack used for chat and (by default) the orchestrator and subagents. Configure Frontier, On-Device, and Cluster defaults so you can switch here without losing credentials or cache paths.
                </p>
                {/* Provider chip-group picker */}
                <div>
                  <label className="block text-sm font-medium text-th-text-tertiary mb-2">Provider</label>
                  <div className="grid grid-cols-2 gap-2">
                    {([
                      { value: "mlx"      as const, name: "Standard",  tag: "MLX",         ring: "ring-emerald-500/40",  bg: "bg-emerald-500/10", border: "border-emerald-500/40", tagCls: "bg-emerald-500/20 border-emerald-500/30 text-emerald-400", downloaded: mlxLocalModels.length },
                      { value: "omlx"     as const, name: "Turbo",     tag: "oMLX",        ring: "ring-blue-500/40",   bg: "bg-blue-500/10",  border: "border-blue-500/40",  tagCls: "bg-blue-500/20 border-blue-500/30 text-blue-400",           downloaded: mlxLocalModels.length },
                      { value: "exo"      as const, name: "Cluster",   tag: "Distributed", ring: "ring-orange-500/40",   bg: "bg-orange-500/10",  border: "border-orange-500/40",  tagCls: "bg-orange-500/20 border-orange-500/30 text-orange-400",   downloaded: exoCatalogReachable ? exoCatalog.filter((m) => m.downloaded).length : 0 },
                      { value: "frontier" as const, name: "Frontier",  tag: "Cloud API",   ring: "ring-sky-500/40",      bg: "bg-sky-500/10",     border: "border-sky-500/40",     tagCls: "bg-sky-500/20 border-sky-500/30 text-sky-400",            downloaded: 0 },
                    ]).map(({ value, name, tag, ring, bg, border, tagCls, downloaded }) => {
                      const current =
                        settings.llm.provider === "anthropic" || settings.llm.provider === "openai"
                          ? "frontier"
                          : settings.llm.provider;
                      const selected = current === value;
                      return (
                        <button
                          key={value}
                          type="button"
                          disabled={providerSwitching}
                          onClick={() => void handleProviderSwitch(value, settings)}
                          className={`flex items-start gap-2 px-3 py-2.5 rounded-xl border text-left transition-all disabled:opacity-60 disabled:cursor-wait ${
                            selected
                              ? `${bg} ${border} ring-1 ${ring}`
                              : "bg-th-surface border-th-border hover:bg-th-surface-hover hover:border-th-border-strong"
                          }`}
                        >
                          <div className="flex-1 min-w-0">
                            <span className="text-sm font-medium text-th-text-primary">{name}</span>
                            {downloaded > 0 && (
                              <span className="block mt-0.5 text-[9px] font-medium text-emerald-500/80">
                                {downloaded} downloaded
                              </span>
                            )}
                          </div>
                          <span className={`mt-0.5 px-1.5 py-0.5 rounded-md text-[10px] font-semibold border leading-none shrink-0 ${
                            selected ? tagCls : "bg-th-surface-hover border-th-border text-th-text-muted"
                          }`}>
                            {tag}
                          </span>
                        </button>
                      );
                    })}
                  </div>

                  {/* Provider-switch inline progress / error */}
                  {providerSwitching && switchingStatus && (
                    <div className="flex items-center gap-2 text-xs text-th-text-secondary pt-1">
                      <Loader2 size={12} className="animate-spin shrink-0" />
                      <span>{switchingStatus}</span>
                    </div>
                  )}
                  {!providerSwitching && switchError && (
                    <div className="flex items-start gap-2 text-xs text-amber-400 pt-1">
                      <AlertTriangle size={12} className="shrink-0 mt-0.5" />
                      <span>{switchError}</span>
                    </div>
                  )}
                </div>

                {(settings.llm.provider === "anthropic" || settings.llm.provider === "openai") && (
                  <>
                    <SelectField
                      label="Frontier model source"
                      value={settings.llm.provider}
                      onChange={(v) => {
                        setSettings((s) => ({ ...s, llm: { ...s.llm, provider: v } }));
                        resetModels();
                        anthAutoKeyRef.current = "";
                      }}
                      options={[
                        { value: "anthropic", label: "Anthropic (API or Bedrock)" },
                        { value: "openai", label: "OpenAI (API or Azure)" },
                      ]}
                    />

                    {settings.llm.provider === "anthropic" && (
                      <div className="space-y-2">
                        <ModelPickerField
                          value={settings.llm.anthropic.model_name}
                          onChange={(v) => updateAnthropicField("model_name", v)}
                          models={availableModels}
                          loading={fetchingModels}
                          error={modelsError}
                          onFetch={handleFetchAnthropicModels}
                        />
                        <p className="text-[11px] text-th-text-muted leading-relaxed">
                          Click <em>Fetch models</em> to list the IDs your{" "}
                          {settings.llm.anthropic.model_provider === "bedrock"
                            ? "Bedrock account"
                            : "Anthropic API key"}{" "}
                          can call. Configure credentials under the{" "}
                          <button
                            type="button"
                            onClick={() => setLlmSubTab("Frontier")}
                            className="underline hover:text-th-text-primary"
                          >
                            Frontier
                          </button>{" "}
                          sub-tab.
                        </p>
                      </div>
                    )}

                    {settings.llm.provider === "openai" && (
                      <div className="space-y-2">
                        <ModelPickerField
                          value={settings.llm.openai.model_name}
                          onChange={(v) => updateOpenAIField("model_name", v)}
                          models={availableModels}
                          loading={fetchingModels}
                          error={modelsError}
                          onFetch={handleFetchOpenAIModels}
                        />
                        <p className="text-[11px] text-th-text-muted leading-relaxed">
                          Click <em>Fetch models</em> to list available models. Configure API key
                          and advanced settings under the{" "}
                          <button
                            type="button"
                            onClick={() => setLlmSubTab("Frontier")}
                            className="underline hover:text-th-text-primary"
                          >
                            Frontier
                          </button>{" "}
                          sub-tab.
                        </p>
                      </div>
                    )}
                  </>
                )}

                {settings.llm.provider === "mlx" && (
                  <div className="space-y-2">
                    <SelectField
                      label="MLX text model (HF repo id)"
                      value={settings.llm.mlx.hf_llm_model_id}
                      onChange={(v) => updateMlxField("hf_llm_model_id", v)}
                      options={mlxSelectOptions(
                        mlxLocalModels.filter((m) => m.repo_id.toLowerCase().includes("mlx")),
                        settings.llm.mlx.hf_llm_model_id,
                        false,
                      )}
                    />
                    <div className="flex items-center justify-between">
                      <p className="text-[11px] text-th-text-muted leading-relaxed">
                        Listing models cached under{" "}
                        <code className="font-mono">{mlxHubDefaultPath || mlxHubDefaultSuffix}</code>.
                        Manage downloads, draft &amp; vision models, and turbo
                        settings under the{" "}
                        <button
                          type="button"
                          onClick={() => setLlmSubTab("Standard")}
                          className="underline hover:text-th-text-primary"
                        >
                          Standard
                        </button>{" "}
                        sub-tab.
                      </p>
                      <button
                        type="button"
                        className="ml-2 inline-flex items-center gap-1 px-2 py-1 text-[11px] font-medium rounded-md border border-th-border bg-th-surface text-th-text-secondary hover:bg-th-surface-hover disabled:opacity-50"
                        onClick={() => void refreshMlxModels()}
                        disabled={mlxListLoading}
                        title="Re-scan the MLX Hub cache"
                      >
                        {mlxListLoading ? (
                          <Loader2 size={11} className="animate-spin" />
                        ) : (
                          <RefreshCw size={11} />
                        )}
                        Refresh
                      </button>
                    </div>
                    {mlxListError && (
                      <p className="text-[11px] text-rose-400">{mlxListError}</p>
                    )}
                  </div>
                )}

                {settings.llm.provider === "omlx" && (
                  <OmlxModelSelector
                    settings={settings}
                    setSettings={setSettings}
                    onGoToOnDevice={() => setLlmSubTab("Turbo")}
                  />
                )}

                {settings.llm.provider === "exo" && (
                  <div className="space-y-2">
                    <ExoModelSelectField
                      label="Active model"
                      value={settings.exo.model_name}
                      onChange={handleExoModelSelect}
                      options={exoModelSelectOptions}
                    />
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-[11px] text-th-text-muted leading-relaxed">
                        {exoCatalogReachable ? (
                          <>
                            Cluster reports{" "}
                            <strong>{exoCatalog.filter((m) => m.downloaded).length}</strong>{" "}
                            downloaded of <strong>{exoCatalog.length}</strong> models. You
                            can also set this id under{" "}
                            <button
                              type="button"
                              onClick={() => setLlmSubTab("Cluster")}
                              className="underline hover:text-th-text-primary"
                            >
                              LLM → Cluster
                            </button>
                            . Preload from the operations panel there.
                          </>
                        ) : (
                          <>
                            {exoCatalogErr || "Cluster offline"} — start the cluster from{" "}
                            <button
                              type="button"
                              onClick={() => setLlmSubTab("Cluster")}
                              className="underline hover:text-th-text-primary"
                            >
                              LLM → Cluster
                            </button>
                            .
                          </>
                        )}
                      </p>
                      <button
                        type="button"
                        className="ml-2 inline-flex items-center gap-1 px-2 py-1 text-[11px] font-medium rounded-md border border-th-border bg-th-surface text-th-text-secondary hover:bg-th-surface-hover disabled:opacity-50"
                        onClick={() => void refreshExoCatalog()}
                        disabled={exoCatalogLoading}
                        title="Re-fetch the cluster catalog"
                      >
                        {exoCatalogLoading ? (
                          <Loader2 size={11} className="animate-spin" />
                        ) : (
                          <RefreshCw size={11} />
                        )}
                        Refresh
                      </button>
                    </div>
                  </div>
                )}

                <div className="pt-2 border-t border-th-border space-y-3">
                  {(() => {
                    const warnings: string[] = [];
                    if (settings.llm.provider === "mlx") {
                      if (!settings.llm.mlx.hf_llm_model_id?.trim()) warnings.push("HF LLM model (repo ID) is required for the MLX default model.");
                    } else if (settings.llm.provider === "omlx") {
                      if (!settings.omlx?.enabled)
                        warnings.push("Enable oMLX above and start the server before selecting it as the default provider.");
                      if (!settings.omlx?.model_name?.trim())
                        warnings.push("Pick a default oMLX model id (it must be loaded in the running server).");
                    } else if (settings.llm.provider === "exo") {
                      if (!settings.exo.enabled)
                        warnings.push("Enable the cluster (Settings → LLM → Cluster) before selecting it as the default provider.");
                      if (!settings.exo.model_name?.trim())
                        warnings.push("Pick a cluster model id above.");
                    } else if (settings.llm.provider === "openai") {
                      const o = settings.llm.openai;
                      if (o.model_provider === "openai" && !o.api_key) warnings.push("OpenAI API key is required — set it under Frontier → OpenAI.");
                      if (o.model_provider === "azure" && !o.azure_endpoint) warnings.push("Azure OpenAI endpoint is required — set it under Frontier → OpenAI.");
                      if (!o.model_name) warnings.push("Select or enter a model name.");
                    } else {
                      const a = settings.llm.anthropic;
                      if (a.model_provider === "anthropic" && !a.api_key) warnings.push("Anthropic API key is required — set it under Frontier → Anthropic.");
                      if (a.model_provider === "bedrock" && (!a.aws_access_key_id || !a.aws_secret_access_key)) {
                        warnings.push("AWS Bedrock requires an Access Key ID and Secret — set them under Frontier → Anthropic.");
                      }
                      if (!a.model_name) warnings.push("Select or enter a default model name.");
                    }
                    return warnings.length > 0 ? (
                      <div className="space-y-1">
                        {warnings.map((w, i) => (
                          <p key={i} className="text-xs text-amber-400 flex items-center gap-1.5">
                            <XCircle size={12} />
                            {w}
                          </p>
                        ))}
                      </div>
                    ) : null;
                  })()}
                  {testResult && (
                    <div className={`flex items-center gap-2 text-sm font-medium ${testResult.success ? "text-emerald-400" : "text-red-400"}`}>
                      {testResult.success ? <CheckCircle size={16} /> : <XCircle size={16} />}
                      {testResult.message}
                    </div>
                  )}
                </div>
              </div>
            </Card>

            <Card title="Orchestrator Agent" dot="bg-amber-500">
              <div className="space-y-4">
                <p className="text-xs text-th-text-tertiary">
                  Planner that delegates to subagents. Use the same stack as the default model, or pick Frontier, On-Device, or the shared Cluster independently.
                </p>
                <SelectField
                  label="Orchestrator stack"
                  value={settings.orchestrator.llm_family}
                  onChange={(v) =>
                    setSettings((s) => ({
                      ...s,
                      orchestrator: {
                        ...s.orchestrator,
                        llm_family: v,
                        ...(v !== "mlx" ? { mlx_model: "" } : {}),
                      },
                    }))
                  }
                  options={[
                    { value: "follow_main", label: "Same as default model" },
                    { value: "frontier", label: "Frontier (Anthropic / Bedrock)" },
                    { value: "openai", label: "OpenAI (API or Azure)" },
                    { value: "mlx", label: "On-Device (Apple Silicon)" },
                    { value: "exo", label: "Cluster (shared with default model)" },
                  ]}
                />
                {settings.orchestrator.llm_family === "exo" && (
                  <p className="text-[11px] text-th-text-muted leading-relaxed">
                    Reuses the cluster URL and model id picked in{" "}
                    <button
                      type="button"
                      onClick={() => setLlmSubTab("Model Provider")}
                      className="underline hover:text-th-text-primary"
                    >
                      LLM → Model Provider
                    </button>
                    . If the main provider is also the <code>Cluster</code>, the
                    orchestrator shares the same in-process client.
                  </p>
                )}
                {settings.orchestrator.llm_family === "mlx" && (
                  <>
                    <SelectField
                      label="Orchestrator MLX repo (optional)"
                      value={settings.orchestrator.mlx_model}
                      onChange={(v) => setSettings((s) => ({ ...s, orchestrator: { ...s.orchestrator, mlx_model: v } }))}
                      options={mlxSelectOptions(mlxLocalModels, settings.orchestrator.mlx_model, true)}
                    />
                    <SelectField
                      label="Orchestrator model type"
                      value={settings.orchestrator.mlx_model_type}
                      onChange={(v) => setSettings((s) => ({ ...s, orchestrator: { ...s.orchestrator, mlx_model_type: v } }))}
                      options={[
                        { value: "llm", label: "Text (LLM)" },
                        { value: "vlm", label: "Vision-language (VLM)" },
                      ]}
                    />
                  </>
                )}
                <SelectField
                  label="System prompt size"
                  value={settings.orchestrator.prompt_mode || "auto"}
                  onChange={(v) => {
                    const mode = (v === "full" || v === "lite" || v === "auto" ? v : "auto") as "auto" | "full" | "lite";
                    setSettings((s) => ({ ...s, orchestrator: { ...s.orchestrator, prompt_mode: mode } }));
                  }}
                  options={[
                    { value: "auto", label: "Auto — full on all models (recommended)" },
                    { value: "full", label: "Force full (~1.7K tokens, Claude-tuned)" },
                    { value: "lite", label: "Force lite (~300 tokens, OSS-friendly)" },
                  ]}
                />
                <p className="text-[11px] text-th-text-tertiary mt-1">
                  Full mode is used by default for all models, including local (mlx/omlx/exo).
                  Lite mode keeps the four behaviour-critical rules (subagent dispatch, /output writes,
                  execute path safety, action confirmation) and drops the long Claude-tuned blocks.
                  Path safety and subagent dispatch are also enforced in middleware regardless of mode.
                </p>
              </div>
            </Card>
            </div>
            )}

            {llmSubTab === "Frontier" && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-stretch">
            <Card title="Anthropic (API or Bedrock)" dot="bg-sky-500">
              <div className="space-y-4">
                <p className="text-xs text-th-text-tertiary">
                  Credentials for Anthropic Claude — direct API or AWS Bedrock. Used when the default model or orchestrator is set to <em>Frontier</em>.
                </p>
                <SelectField label="API Mode" value={settings.llm.anthropic.model_provider} onChange={(v) => switchProviderMode("model_provider", v)} options={[{ value: "anthropic", label: "Direct API" }, { value: "bedrock", label: "AWS Bedrock" }]} />
                {settings.llm.anthropic.model_provider === "anthropic" && <SecretField label="API Key" value={settings.llm.anthropic.api_key} onChange={(v) => updateAnthropicField("api_key", v)} placeholder="sk-ant-..." />}
                {settings.llm.anthropic.model_provider === "bedrock" && (
                  <>
                    <SecretField label="AWS Access Key ID" value={settings.llm.anthropic.aws_access_key_id} onChange={(v) => updateAnthropicField("aws_access_key_id", v)} />
                    <SecretField label="AWS Secret Access Key" value={settings.llm.anthropic.aws_secret_access_key} onChange={(v) => updateAnthropicField("aws_secret_access_key", v)} />
                    <InputField label="Region" value={settings.llm.anthropic.bedrock_region} onChange={(v) => updateAnthropicField("bedrock_region", v)} />
                  </>
                )}
                <ModelPickerField
                  value={settings.llm.anthropic.model_name}
                  onChange={(v) => updateAnthropicField("model_name", v)}
                  models={availableModels}
                  loading={fetchingModels}
                  error={modelsError}
                  onFetch={handleFetchAnthropicModels}
                />
                <InputField label="Max Tokens" value={String(settings.llm.anthropic.max_tokens)} onChange={(v) => updateAnthropicField("max_tokens", parseInt(v) || 8192)} type="number" />
              </div>
            </Card>

            <Card title="OpenAI (API or Azure)" dot="bg-emerald-400">
              <div className="space-y-4">
                <p className="text-xs text-th-text-tertiary">
                  Credentials for OpenAI models — native API or Azure OpenAI Service. Used when the default model or orchestrator is set to <em>OpenAI</em>.
                </p>
                <SelectField
                  label="API Mode"
                  value={settings.llm.openai.model_provider}
                  onChange={(v) => {
                    updateOpenAIField("model_provider", v);
                    resetModels();
                  }}
                  options={[
                    { value: "openai", label: "OpenAI API (native)" },
                    { value: "azure", label: "Azure OpenAI Service" },
                  ]}
                />
                {settings.llm.openai.model_provider === "openai" && (
                  <SecretField
                    label="API Key"
                    value={settings.llm.openai.api_key}
                    onChange={(v) => updateOpenAIField("api_key", v)}
                    placeholder="sk-..."
                  />
                )}
                {settings.llm.openai.model_provider === "azure" && (
                  <>
                    <SecretField
                      label="API Key"
                      value={settings.llm.openai.azure_api_key}
                      onChange={(v) => updateOpenAIField("azure_api_key", v)}
                      placeholder="Azure OpenAI API key"
                    />
                    <InputField
                      label="Endpoint"
                      value={settings.llm.openai.azure_endpoint}
                      onChange={(v) => updateOpenAIField("azure_endpoint", v)}
                      placeholder="https://<resource>.openai.azure.com"
                    />
                    <InputField
                      label="Deployment name"
                      value={settings.llm.openai.azure_deployment}
                      onChange={(v) => updateOpenAIField("azure_deployment", v)}
                      placeholder="Leave blank to use model name"
                    />
                    <InputField
                      label="API version"
                      value={settings.llm.openai.azure_api_version}
                      onChange={(v) => updateOpenAIField("azure_api_version", v)}
                      placeholder="2024-12-01-preview"
                    />
                  </>
                )}
                <ModelPickerField
                  value={settings.llm.openai.model_name}
                  onChange={(v) => updateOpenAIField("model_name", v)}
                  models={availableModels}
                  loading={fetchingModels}
                  error={modelsError}
                  onFetch={handleFetchOpenAIModels}
                />
                <InputField
                  label="Max Tokens"
                  value={String(settings.llm.openai.max_tokens)}
                  onChange={(v) => updateOpenAIField("max_tokens", parseInt(v) || 16384)}
                  type="number"
                />
                <InputField
                  label="Temperature"
                  value={String(settings.llm.openai.temperature)}
                  onChange={(v) => updateOpenAIField("temperature", parseFloat(v) || 0.0)}
                  type="number"
                />
              </div>
            </Card>
            </div>
            )}

            {llmSubTab === "Standard" && (
            <div className="space-y-4">
            {/* Subpill nav */}
            <div className="flex gap-1 bg-th-inset-bg rounded-xl p-1 w-fit border border-th-border">
              {(["Overview", "Model"] as const).map((id) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => setStandardSubTab(id)}
                  className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all duration-150 ${
                    standardSubTab === id
                      ? "bg-th-tab-active-bg text-th-tab-active-fg shadow-sm"
                      : "text-th-text-tertiary hover:text-th-text-primary hover:bg-th-surface-hover"
                  }`}
                >
                  {id}
                </button>
              ))}
            </div>

            {standardSubTab === "Model" && (
            <Card title="Pick an MLX model" dot="bg-blue-400">
              <p className="text-xs text-th-text-tertiary mb-4">
                Browse, download, and select models from HuggingFace Hub. The chosen model is used by
                both Standard (in-process) and Turbo (oMLX server) inference.
              </p>
              <ModelChooser
                selectedRepoId={settings.llm.mlx.hf_llm_model_id}
                hfToken={settings.llm.mlx.hf_token}
                cacheDir={settings.llm.mlx.hf_hub_cache}
                onDownloadComplete={(repo, label) => {
                  setSettings((s) => {
                    const bm = s.llm.mlx.mlx_bookmarks ?? [];
                    const rest = bm.filter((b) => b.repo_id !== repo);
                    return {
                      ...s,
                      llm: {
                        ...s.llm,
                        mlx: { ...s.llm.mlx, mlx_bookmarks: [...rest, { repo_id: repo, label }] },
                      },
                    };
                  });
                  void refreshMlxModels();
                  void api.omlxStart().catch(() => undefined);
                }}
                onUseCached={(repo) => {
                  updateMlxField("hf_llm_model_id", repo);
                  setSettings((s) => ({
                    ...s,
                    omlx: { ...s.omlx, default_model: repo },
                  }));
                }}
              />
            </Card>
            )}

            {standardSubTab === "Overview" && (
            <Card title="On-Device defaults" dot="bg-blue-500">
              <div className="space-y-4">
                <p className="text-xs text-th-text-tertiary">
                  Runs models directly in-process on Apple Silicon. Hub cache uses env <code className="text-th-text-secondary">HF_HUB_CACHE</code>. Path is relative to{" "}
                  <code className="font-mono text-th-text-secondary">~/.cache/</code> (e.g. <code className="font-mono text-th-text-secondary">{mlxHubDefaultSuffix}</code>) or absolute. Resolves to:{" "}
                  <span className="font-mono text-th-text-secondary break-all">{mlxHubDefaultPath || "…"}</span>
                  {mlxHubCacheRoot ? (
                    <span className="block mt-1 text-th-text-muted">
                      Cache root: <code className="font-mono">{mlxHubCacheRoot}</code>
                    </span>
                  ) : null}
                </p>
                <InputField
                  label="Model cache folder (optional)"
                  value={settings.llm.mlx.hf_hub_cache}
                  onChange={(v) => updateMlxField("hf_hub_cache", v)}
                  placeholder={mlxHubDefaultSuffix}
                />
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    type="button"
                    className="px-3 py-1.5 text-xs font-medium rounded-lg border border-th-border bg-th-surface text-th-text-secondary hover:bg-th-surface-hover"
                    onClick={() => updateMlxField("hf_hub_cache", mlxHubDefaultSuffix)}
                  >
                    Use default (~/.cache/{mlxHubDefaultSuffix})
                  </button>
                  <button
                    type="button"
                    className="px-3 py-1.5 text-xs font-medium rounded-lg border border-th-border bg-th-surface text-th-text-secondary hover:bg-th-surface-hover inline-flex items-center gap-1.5"
                    onClick={() => void refreshMlxModels()}
                    disabled={mlxListLoading}
                  >
                    {mlxListLoading ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}
                    Refresh model list
                  </button>
                  {!mlxListLoading && !mlxListError && mlxLocalModels.length > 0 && (
                    <span className="text-xs text-th-text-muted">{mlxLocalModels.length} model{mlxLocalModels.length !== 1 ? "s" : ""} found</span>
                  )}
                </div>
                {mlxListError && <p className="text-xs text-red-500">{mlxListError}</p>}
                {(() => {
                  const activeModelOpts = mlxSelectOptions(mlxLocalModels, settings.llm.mlx.hf_llm_model_id, false, mlxListLoading, mlxListFetched);
                  const activeModelNotInCache = activeModelOpts.find((o) => o.value === settings.llm.mlx.hf_llm_model_id)?.notInCache;
                  return (
                    <div className="space-y-1.5">
                      <SelectField
                        label="Active model"
                        value={settings.llm.mlx.hf_llm_model_id}
                        onChange={(v) => updateMlxField("hf_llm_model_id", v)}
                        options={activeModelOpts}
                      />
                      {activeModelNotInCache && (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20">
                          not in cache listing yet
                        </span>
                      )}
                    </div>
                  );
                })()}
                <SelectField
                  label="Vision model (optional)"
                  value={settings.llm.mlx.hf_vlm_model_id}
                  onChange={(v) => updateMlxField("hf_vlm_model_id", v)}
                  options={mlxSelectOptions(mlxLocalModels, settings.llm.mlx.hf_vlm_model_id, true, mlxListLoading, mlxListFetched)}
                />
                <SelectField
                  label="Draft model (optional)"
                  value={settings.llm.mlx.hf_draft_llm_model_id}
                  onChange={(v) => updateMlxField("hf_draft_llm_model_id", v)}
                  options={mlxSelectOptions(mlxLocalModels, settings.llm.mlx.hf_draft_llm_model_id, true, mlxListLoading, mlxListFetched)}
                />
                <SecretField label="HF Token (optional)" value={settings.llm.mlx.hf_token} onChange={(v) => updateMlxField("hf_token", v)} placeholder="hf_… for gated repos" />
                <div className="rounded-xl border border-th-border bg-th-inset-bg p-4 space-y-4">
                  <p className="text-xs font-medium text-th-text-secondary uppercase tracking-wide">Generation &amp; KV cache</p>
                  <p className="text-xs text-th-text-tertiary">
                    These map to <span className="font-mono text-th-text-secondary">MLX_*</span> environment variables for local inference (prompt reuse, KV quantization, sampling).
                  </p>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <InputField label="Max tokens" value={String(settings.llm.mlx.mlx_max_tokens)} onChange={(v) => updateMlxPartial({ mlx_max_tokens: Math.max(1, parseInt(v, 10) || 8192) })} type="number" />
                    <InputField label="Temperature" value={String(settings.llm.mlx.mlx_temp)} onChange={(v) => updateMlxPartial({ mlx_temp: Number.isFinite(parseFloat(v)) ? parseFloat(v) : 0 })} type="number" />
                    <InputField label="Repetition penalty" value={String(settings.llm.mlx.mlx_repetition_penalty)} onChange={(v) => updateMlxPartial({ mlx_repetition_penalty: Math.max(0.1, parseFloat(v) || 1.1) })} type="number" />
                    <SelectField
                      label="KV cache bits"
                      value={settings.llm.mlx.mlx_kv_bits == null ? "" : String(settings.llm.mlx.mlx_kv_bits)}
                      onChange={(v) => updateMlxPartial({ mlx_kv_bits: v === "" ? null : (parseInt(v, 10) === 8 ? 8 : 4) })}
                      options={[
                        { value: "", label: "Full precision (default)" },
                        { value: "4", label: "4-bit KV" },
                        { value: "8", label: "8-bit KV" },
                      ]}
                    />
                    <SelectField
                      label="KV group size"
                      value={String(settings.llm.mlx.mlx_kv_group_size ?? 64)}
                      onChange={(v) => updateMlxPartial({ mlx_kv_group_size: parseInt(v, 10) })}
                      options={[
                        { value: "32", label: "32 (better quality)" },
                        { value: "64", label: "64 (default)" },
                        { value: "128", label: "128 (less overhead)" },
                      ]}
                    />
                    <InputField
                      label="KV cache cap (tokens, 0 = unbounded)"
                      type="number"
                      value={String(settings.llm.mlx.mlx_prompt_cache_max_tokens ?? 32768)}
                      onChange={(v) => updateMlxPartial({ mlx_prompt_cache_max_tokens: Math.max(0, parseInt(v, 10) || 0) })}
                    />
                  </div>
                  <p className="text-[11px] text-th-text-muted leading-relaxed">
                    The KV cache stores the model's attention state for the conversation so far. Without a cap it grows for as long as the agent runs and can push your Mac into swap or trigger out-of-memory crashes after long autonomous sessions. The default 32 768 tokens caps the cache around 1 GB on a 7B 4-bit model. When the cap is hit, the cache trims back to roughly half before the next turn (one slower turn, then back to normal). Set to 0 only if you know you have headroom.
                  </p>
                  <div className="flex flex-col gap-3">
                    <Toggle label="Verbose MLX logging (Thought/Action/Observation)" checked={settings.llm.mlx.mlx_verbose} onChange={(v) => updateMlxPartial({ mlx_verbose: v })} />
                    <Toggle label="Chain-of-thought thinking (Qwen-style models)" checked={settings.llm.mlx.mlx_thinking} onChange={(v) => updateMlxPartial({ mlx_thinking: v })} />
                    <Toggle
                      label="KV prompt cache across turns"
                      checked={settings.llm.mlx.mlx_prompt_cache}
                      onChange={(v) => updateMlxPartial({ mlx_prompt_cache: v })}
                    />
                    <Toggle
                      label="Reuse static system/tool prefix (requires prompt cache)"
                      checked={settings.llm.mlx.mlx_system_prompt_cache}
                      onChange={(v) => updateMlxPartial({ mlx_system_prompt_cache: v })}
                    />
                  </div>
                </div>
                {/* Turbo mode was an in-process reimplementation of oMLX's paged
                    KV cache + prefix sharing.  Now that Otto integrates oMLX
                    directly (Settings → LLM → Turbo), the in-process
                    turbo path is redundant.  The setting is preserved in
                    config.json for backwards compatibility but ignored by the
                    runtime — see model_factory._build_mlx_chat. */}
              </div>
            </Card>
            )}
            </div>
            )}

            {llmSubTab === "Turbo" && (
            <div className="space-y-4">
              <div className="flex gap-1 bg-th-inset-bg rounded-xl p-1 w-fit border border-th-border">
                {(["Overview", "Model"] as const).map((id) => (
                  <button
                    key={id}
                    type="button"
                    onClick={() => setTurboSubTab(id)}
                    className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all duration-150 ${
                      turboSubTab === id
                        ? "bg-th-tab-active-bg text-th-tab-active-fg shadow-sm"
                        : "text-th-text-tertiary hover:text-th-text-primary hover:bg-th-surface-hover"
                    }`}
                  >
                    {id}
                  </button>
                ))}
              </div>
              <Card title={turboSubTab === "Model" ? "Pick an MLX model" : "Turbo inference"} dot="bg-sky-500">
                <p className="text-xs text-th-text-tertiary mb-4">
                  Runs a dedicated local inference server with a paged KV allocator, continuous batching, and
                  prefix-tree cache sharing — higher throughput than standard in-process inference.
                  Models are shared from the same hub cache as the Standard tab.
                </p>
                <OmlxQuickPanel settings={settings} setSettings={setSettings} turboSubTab={turboSubTab} />
              </Card>
            </div>
            )}

            {llmSubTab === "Cluster" && (
            <div className="space-y-5">
              {/* Unified header: tab nav left, compact live status right */}
              <div className="flex items-center justify-between gap-4 flex-wrap border-b border-th-border pb-3">
                <nav className="flex items-center gap-0.5">
                  {EXO_SUBTABS.map((t) => {
                    const badge =
                      t === "Cluster" && settings.exo.remotes.length > 0
                        ? settings.exo.remotes.length + 1
                        : null;
                    const active = exoSubTab === t;
                    return (
                      <button
                        key={t}
                        type="button"
                        className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-[13px] font-medium rounded-md transition-colors ${
                          active
                            ? "text-th-text-primary bg-th-inset-bg"
                            : "text-th-text-tertiary hover:text-th-text-primary hover:bg-th-surface-hover"
                        }`}
                        onClick={() => setExoSubTab(t)}
                      >
                        {t}
                        {badge !== null && (
                          <span className={`min-w-[17px] h-[17px] flex items-center justify-center px-1 rounded-full text-[10px] font-semibold leading-none ${
                            active
                              ? "bg-th-tab-active-bg text-th-tab-active-fg"
                              : "bg-th-surface text-th-text-secondary border border-th-border"
                          }`}>
                            {badge}
                          </span>
                        )}
                      </button>
                    );
                  })}
                </nav>

                <span
                  className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium border ${
                    exoLiveStatus?.reachable
                      ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-500"
                      : settings.exo.enabled
                        ? "border-amber-500/30 bg-amber-500/10 text-amber-500"
                        : "border-th-border bg-th-inset-bg text-th-text-muted"
                  }`}
                  title={
                    exoLiveStatus?.reachable
                      ? `${exoLiveStatus.peer_count} node${exoLiveStatus.peer_count === 1 ? "" : "s"} · ${exoLiveStatus.loaded_models.length} model${exoLiveStatus.loaded_models.length === 1 ? "" : "s"} loaded`
                      : settings.exo.enabled
                        ? "Waiting for the daemon to respond"
                        : "Cluster is disabled"
                  }
                >
                  <span className={`h-1.5 w-1.5 rounded-full ${
                    exoLiveStatus?.reachable
                      ? "bg-emerald-500"
                      : settings.exo.enabled
                        ? "bg-amber-400 animate-pulse"
                        : "bg-neutral-400"
                  }`} />
                  {exoLiveStatus?.reachable
                    ? `Online · ${exoLiveStatus.peer_count} node${exoLiveStatus.peer_count === 1 ? "" : "s"}`
                    : settings.exo.enabled
                      ? "Connecting…"
                      : "Offline"}
                </span>
              </div>

              {exoSubTab === "Cluster" && (
                <div className="space-y-5">
                  <div className="rounded-xl border border-th-border bg-th-inset-bg p-4">
                    <Toggle
                      label="Cluster enabled"
                      description="Master switch. When off, auto-start is skipped and cluster controls are disabled."
                      checked={settings.exo.enabled}
                      onChange={(v) => updateExoPartial({ enabled: v })}
                    />
                  </div>
                  <ClusterSetupFlow
                    variant="settings"
                    settings={settings}
                    onPatch={handleClusterPatch}
                  />
                </div>
              )}

              {exoSubTab === "Advanced" && (
              <div className="space-y-5">

                {(
                <>
                <p className="text-xs text-th-text-tertiary leading-relaxed">
                  Runtime delivery, ports, and lifecycle for the local daemon. Start/stop and per-node controls live on the <strong>Cluster</strong> tab.
                </p>

                <ExoRuntimeSource
                  mode={settings.exo.mode}
                  prebuiltUrl={settings.exo.prebuilt_url}
                  onChange={updateExoPartial}
                  sourceFields={
                    <>
                      <InputField
                        label="Repo URL"
                        value={settings.exo.repo_url}
                        onChange={(v) => updateExoPartial({ repo_url: v })}
                        placeholder="https://github.com/exo-explore/exo.git"
                      />
                      <InputField
                        label="Repo ref (tag / branch / sha)"
                        value={settings.exo.repo_ref}
                        onChange={(v) => updateExoPartial({ repo_ref: v })}
                        placeholder="v1.0.71"
                      />
                      <ExoReleaseCheckButton currentRef={settings.exo.repo_ref} />
                    </>
                  }
                />

                <div className="space-y-1">
                  <p className="text-[10px] font-semibold text-th-text-muted uppercase tracking-widest px-0.5">Networking</p>
                  <div className="rounded-xl border border-th-border bg-th-inset-bg p-4 space-y-3">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      <InputField
                        label="API port"
                        type="number"
                        value={String(settings.exo.api_port)}
                        onChange={(v) => updateExoPartial({ api_port: parseInt(v, 10) || settings.exo.api_port })}
                      />
                      <InputField
                        label="libp2p port (0 = OS-assigned)"
                        type="number"
                        value={String(settings.exo.libp2p_port)}
                        onChange={(v) => updateExoPartial({ libp2p_port: parseInt(v, 10) || 0 })}
                      />
                    </div>
                    <InputField
                      label="Base URL override"
                      value={settings.exo.base_url}
                      onChange={(v) => updateExoPartial({ base_url: v })}
                      placeholder="(blank ⇒ derived from API port)"
                    />
                  </div>
                </div>

                <div className="space-y-1">
                  <p className="text-[10px] font-semibold text-th-text-muted uppercase tracking-widest px-0.5">Lifecycle</p>
                  <div className="rounded-xl border border-th-border bg-th-inset-bg p-4 space-y-3">
                    <Toggle
                      label="Auto-start on boot"
                      description="Start the local daemon and every enabled remote node when the backend boots."
                      checked={settings.exo.auto_start}
                      onChange={(v) => updateExoPartial({ auto_start: v })}
                    />
                    {settings.exo.mode === "source" && (
                    <Toggle
                      label="Auto-provision dependencies"
                      description="Auto-install missing prerequisites (brew / uv / node / rust) when setting up nodes."
                      checked={settings.exo.auto_provision}
                      onChange={(v) => updateExoPartial({ auto_provision: v })}
                    />
                    )}
                    <Toggle
                      label="Skip Terminal.app wrapper"
                      description="Only enable in a signed Otto.app with Local-Network privacy. Leave off in dev so libp2p mDNS multicast works."
                      checked={settings.exo.no_terminal_wrap}
                      onChange={(v) => updateExoPartial({ no_terminal_wrap: v })}
                    />
                  </div>
                </div>
                </>
                )}

                {(
                <>
                <p className="text-xs text-th-text-tertiary">
                  Default model id used for chat / orchestrator / subagents when the
                  provider is set to <strong>Cluster</strong>, plus the placement strategy
                  exo applies when loading it. Browse and preload models on the{" "}
                  <strong>Catalog</strong> pill.
                </p>

                <div className="space-y-1">
                  <p className="text-[10px] font-semibold text-th-text-muted uppercase tracking-widest px-0.5">Active Model</p>
                  <div className="rounded-xl border border-th-border bg-th-inset-bg p-4 space-y-3">
                    <p className="text-[11px] text-th-text-tertiary leading-relaxed">
                      Used when{" "}
                      <button type="button" onClick={() => setLlmSubTab("Model Provider")} className="underline hover:text-th-text-primary">
                        Model Provider
                      </button>{" "}
                      is set to Cluster, and for any orchestrator / subagent stack set to Cluster.
                    </p>
                    <ExoModelSelectField
                      label="Active model"
                      value={settings.exo.model_name}
                      onChange={handleExoModelSelect}
                      options={exoModelSelectOptions}
                    />
                    <div className="flex items-center justify-between gap-2 flex-wrap">
                      <p className="text-[11px] text-th-text-muted leading-relaxed">
                        {exoCatalogReachable ? (
                          <>
                            <strong>{exoCatalog.filter((m) => m.downloaded).length}</strong>{" "}
                            of <strong>{exoCatalog.length}</strong> catalog models downloaded.
                          </>
                        ) : (
                          <span className="text-amber-400/90">{exoCatalogErr || "Cluster offline"} — start the cluster first.</span>
                        )}
                      </p>
                      <div className="flex items-center gap-1.5 shrink-0">
                        {settings.exo.model_name?.trim() && exoCatalogReachable && (
                          <button
                            type="button"
                            className="inline-flex items-center gap-1 px-2 py-1 text-[11px] font-medium rounded-md border border-red-500/30 bg-red-500/10 text-red-400 hover:bg-red-500/20 disabled:opacity-50 transition-colors"
                            onClick={() => void handleExoUnload()}
                            disabled={exoUnloading}
                            title="Evict model instances from cluster RAM (weights stay on disk)"
                          >
                            {exoUnloading ? <Loader2 size={11} className="animate-spin" /> : <Square size={11} />}
                            Unload
                          </button>
                        )}
                        <button
                          type="button"
                          className="inline-flex items-center gap-1 px-2 py-1 text-[11px] font-medium rounded-md border border-th-border bg-th-surface text-th-text-secondary hover:bg-th-surface-hover disabled:opacity-50 transition-colors"
                          onClick={() => void refreshExoCatalog()}
                          disabled={exoCatalogLoading}
                          title="Re-fetch the cluster catalog"
                        >
                          {exoCatalogLoading ? <Loader2 size={11} className="animate-spin" /> : <RefreshCw size={11} />}
                          Refresh
                        </button>
                      </div>
                    </div>
                    {exoUnloadMsg && (
                      <p className={`text-[11px] ${exoUnloadMsg.ok ? "text-emerald-400" : "text-red-400"}`}>
                        {exoUnloadMsg.ok ? "✓ " : "✗ "}{exoUnloadMsg.text}
                      </p>
                    )}
                  </div>
                </div>

                <div className="space-y-1">
                  <p className="text-[10px] font-semibold text-th-text-muted uppercase tracking-widest px-0.5">Download a Model</p>
                  <div className="rounded-xl border border-th-border bg-th-inset-bg p-4 space-y-3">
                    <p className="text-[11px] text-th-text-tertiary leading-relaxed">
                      Browse the cluster catalog, then download &amp; load any model. The cluster shards weights across placement nodes — a 4-bit 30B model is ~20 GB.
                    </p>
                    <ExoModelChooser
                      enabled={settings.exo.enabled && exoCatalogReachable}
                      selectedModelId={settings.exo.model_name}
                      onUseLoaded={handleExoModelSelect}
                      onPreloadComplete={() => void refreshExoCatalog()}
                    />
                  </div>
                </div>

                <div className="space-y-1">
                  <p className="text-[10px] font-semibold text-th-text-muted uppercase tracking-widest px-0.5">Placement Strategy</p>
                  <div className="rounded-xl border border-th-border bg-th-inset-bg p-4 space-y-3">
                    <p className="text-[11px] text-th-text-tertiary leading-relaxed">
                      Minimum nodes that must hold a shard. <strong>1</strong> (default) = fastest single-node. Set <strong>2+</strong> to pipeline-parallel split for larger models or longer contexts. Note: cross-node tensor traffic costs tokens/sec.
                    </p>
                    <CommitInputField
                      label="Minimum nodes (min_nodes)"
                      type="number"
                      min={1}
                      value={String(settings.exo.min_nodes ?? 1)}
                      onChange={(v) => {
                        const n = Math.max(1, parseInt(v, 10) || 1);
                        updateExoPartial({ min_nodes: n });
                      }}
                    />
                    {settings.exo.min_nodes > settings.exo.remotes.length + 1 && (
                      <div className="flex items-start gap-1.5 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2">
                        <AlertTriangle size={12} className="text-amber-400 shrink-0 mt-0.5" />
                        <p className="text-[11px] text-amber-400 leading-relaxed">
                          Only {settings.exo.remotes.length + 1} node{settings.exo.remotes.length + 1 === 1 ? "" : "s"} configured (this Mac + {settings.exo.remotes.length} remote{settings.exo.remotes.length === 1 ? "" : "s"}). The cluster will reject placement requests with min_nodes greater than reachable peers.
                        </p>
                      </div>
                    )}

                    <label className="flex flex-col gap-1">
                      <span className="text-[11px] font-medium text-th-text-secondary">Sharding strategy</span>
                      <select
                        value={settings.exo.sharding ?? "Pipeline"}
                        onChange={(e) => updateExoPartial({ sharding: e.target.value })}
                        className="bg-th-input-bg border border-th-border rounded px-2 py-1 text-[11px] text-th-text-primary focus:outline-none focus:ring-1 focus:ring-th-tab-active-bg"
                      >
                        <option value="Pipeline">Pipeline — best single-request latency (default)</option>
                        <option value="Tensor">Tensor — higher throughput on 2-4 nodes</option>
                      </select>
                      <span className="text-[10px] text-th-text-muted leading-relaxed">
                        Pipeline splits layers across devices (lowest latency for one request at a time, works for every model). Tensor splits each layer for up to 1.8x/3.2x throughput on 2/4 nodes, but only tensor-capable models support it. No effect on single-node placements.
                      </span>
                    </label>

                    <label className="flex flex-col gap-1">
                      <span className="text-[11px] font-medium text-th-text-secondary">Collective backend</span>
                      <select
                        value={settings.exo.instance_meta ?? "MlxRing"}
                        onChange={(e) => updateExoPartial({ instance_meta: e.target.value })}
                        className="bg-th-input-bg border border-th-border rounded px-2 py-1 text-[11px] text-th-text-primary focus:outline-none focus:ring-1 focus:ring-th-tab-active-bg"
                      >
                        <option value="MlxRing">MlxRing — universal (default)</option>
                        <option value="MlxJaccl">MlxJaccl — Thunderbolt 5 / RDMA only</option>
                      </select>
                      <span className="text-[10px] text-th-text-muted leading-relaxed">
                        MlxRing works over any network. MlxJaccl has lower inter-node latency but requires Thunderbolt 5 / RDMA-capable hardware (M4 Pro/Max, macOS 26.2+) — only enable it on a qualified cluster.
                      </span>
                    </label>
                  </div>
                </div>

                <div className="space-y-1">
                  <p className="text-[10px] font-semibold text-th-text-muted uppercase tracking-widest px-0.5">Generation</p>
                  <div className="rounded-xl border border-th-border bg-th-inset-bg p-4 space-y-3">
                    <p className="text-[11px] text-th-text-tertiary leading-relaxed">
                      Per-request generation knobs sent to the cluster. Lower <strong>max tokens</strong> and disabling <strong>thinking</strong> are the two biggest latency wins for an interactive agent.
                    </p>
                    <CommitInputField
                      label="Max output tokens"
                      type="number"
                      min={256}
                      value={String(settings.exo.max_tokens ?? 8192)}
                      onChange={(v) => {
                        const n = Math.max(1, parseInt(v, 10) || 8192);
                        updateExoPartial({ max_tokens: n });
                      }}
                    />
                    <label className="flex items-center justify-between gap-2 cursor-pointer select-none">
                      <span className="text-[11px] text-th-text-secondary">
                        Thinking mode
                        <span className="text-[10px] text-th-text-muted ml-1">(chain-of-thought; Qwen3 / DeepSeek / GLM)</span>
                      </span>
                      <input
                        type="checkbox"
                        checked={settings.exo.enable_thinking ?? false}
                        onChange={(e) => updateExoPartial({ enable_thinking: e.target.checked })}
                        className="w-3.5 h-3.5 accent-th-tab-active-bg cursor-pointer"
                      />
                    </label>
                  </div>
                </div>
                </>
                )}

                {(
                <>
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2.5 flex-wrap">
                    <span className="text-xs text-th-text-secondary">
                      <strong className="text-th-text-primary">{settings.exo.remotes.length}</strong>{" "}
                      remote{settings.exo.remotes.length === 1 ? "" : "s"} configured
                    </span>
                    {exoLiveStatus?.reachable && (
                      <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-medium bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">
                        <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
                        {Math.max(0, exoLiveStatus.peer_count - 1)} joined
                      </span>
                    )}
                  </div>
                  <button
                    type="button"
                    className="inline-flex items-center gap-1.5 text-xs font-medium rounded-lg border border-th-border bg-th-surface px-3 py-1.5 text-th-text-secondary hover:bg-th-surface-hover transition-colors"
                    onClick={() => setExoShowAddRemote((v) => !v)}
                  >
                    <Plus size={12} />
                    {exoShowAddRemote ? "Cancel" : "Add node"}
                  </button>
                </div>
                <div className="rounded-xl border border-th-border bg-th-inset-bg p-4 space-y-3">
                  <p className="text-[11px] text-th-text-tertiary leading-relaxed">
                    Each node is an SSH alias from <code className="font-mono text-th-text-secondary">~/.ssh/config</code> with key-based (non-interactive) auth. "Start" deploys the cluster helper and launches the daemon on the remote, mirroring <code className="font-mono text-th-text-secondary">exo up --remote</code>.
                  </p>

                  {exoShowAddRemote && (
                    <div className="space-y-3 rounded-md border border-th-border bg-th-bg p-3">
                      {/* Mode toggle: classic alias vs from-scratch wizard. */}
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="text-[11px] font-medium text-th-text-secondary">
                          Add a remote by:
                        </span>
                        <div className="inline-flex rounded-md border border-th-border bg-th-surface p-0.5 text-[11px]">
                          <button
                            type="button"
                            onClick={() => setExoAddMode("alias")}
                            className={`px-2.5 py-1 rounded transition-colors ${
                              exoAddMode === "alias"
                                ? "bg-th-bg text-th-text-primary shadow-sm"
                                : "text-th-text-secondary hover:text-th-text-primary"
                            }`}
                          >
                            Existing SSH alias
                          </button>
                          <button
                            type="button"
                            onClick={() => setExoAddMode("wizard")}
                            className={`px-2.5 py-1 rounded transition-colors ${
                              exoAddMode === "wizard"
                                ? "bg-th-bg text-th-text-primary shadow-sm"
                                : "text-th-text-secondary hover:text-th-text-primary"
                            }`}
                            title="Set up a brand-new node: probe, generate key, install on remote, append ~/.ssh/config block"
                          >
                            Set up new node from scratch
                          </button>
                        </div>
                      </div>

                      {exoAddMode === "wizard" && (
                        <ExoSetupSteps
                          sshHosts={exoSshHosts}
                          lanHosts={exoLanHosts}
                          existingAliases={settings.exo.remotes.map((r) => r.ssh_alias)}
                          onComplete={handleExoSetupComplete}
                          onCancel={() => {
                            setExoShowAddRemote(false);
                            setExoAddMode("alias");
                          }}
                        />
                      )}

                      {exoAddMode === "alias" && (
                        <>
                      {/* Suggestions */}
                      {(() => {
                        const existing = new Set(
                          settings.exo.remotes.map((r) => r.ssh_alias),
                        );
                        const aliases = exoSshHosts.filter((h) => !existing.has(h.alias));
                        const lan = exoLanHosts.filter(
                          (h) => !h.matches_alias || !existing.has(h.matches_alias),
                        );
                        if (aliases.length === 0 && lan.length === 0 && !exoLanScanned) {
                          return null;
                        }
                        return (
                          <div className="space-y-2">
                            {aliases.length > 0 && (
                              <div>
                                <p className="text-[11px] font-medium text-th-text-secondary uppercase tracking-wide mb-1">
                                  From your ~/.ssh/config
                                </p>
                                <div className="flex flex-wrap gap-1.5">
                                  {aliases.map((h) => (
                                    <button
                                      key={`ssh-${h.alias}`}
                                      type="button"
                                      onClick={() => handleExoPickSshHost(h)}
                                      className={`text-xs rounded-md border px-2 py-1 transition-colors ${
                                        exoAddAlias === h.alias
                                          ? "border-blue-500 bg-blue-500/10 text-blue-400"
                                          : "border-th-border bg-th-surface text-th-text-secondary hover:bg-th-surface-hover"
                                      }`}
                                      title={
                                        `${h.user ? h.user + "@" : ""}${h.hostname || h.alias}` +
                                        (h.port !== 22 ? `:${h.port}` : "") +
                                        (h.source_file ? `\n(from ${h.source_file})` : "")
                                      }
                                    >
                                      <span className="font-mono">{h.alias}</span>
                                      {h.hostname && h.hostname !== h.alias && (
                                        <span className="text-th-text-muted ml-1">
                                          → {h.hostname}
                                        </span>
                                      )}
                                    </button>
                                  ))}
                                </div>
                              </div>
                            )}
                            {(lan.length > 0 || exoLanScanned) && (
                              <div>
                                <p className="text-[11px] font-medium text-th-text-secondary uppercase tracking-wide mb-1">
                                  Discovered on this LAN
                                </p>
                                {lan.length === 0 ? (
                                  <p className="text-[11px] text-th-text-muted">
                                    No <code className="font-mono">_ssh._tcp</code> services
                                    advertised on this network.
                                  </p>
                                ) : (
                                  <div className="flex flex-wrap gap-1.5">
                                    {lan.map((h) => {
                                      const tb = (h.thunderbolt_addresses || [])[0];
                                      const ipv4 = (h.addresses || []).find((a) =>
                                        /^\d+\.\d+\.\d+\.\d+$/.test(a),
                                      );
                                      const picked = tb || ipv4;
                                      const titleLines = [
                                        `${h.hostname || h.name}:${h.port}`,
                                        h.addresses && h.addresses.length > 0
                                          ? `addresses: ${h.addresses.join(", ")}`
                                          : "",
                                        h.thunderbolt_addresses && h.thunderbolt_addresses.length > 0
                                          ? `thunderbolt: ${h.thunderbolt_addresses.join(", ")}`
                                          : "",
                                        !h.matches_alias && picked
                                          ? `Picking this will use ${picked}${tb ? " (Thunderbolt Bridge)" : ""} instead of the .local hostname.`
                                          : "",
                                      ].filter(Boolean);
                                      return (
                                        <button
                                          key={`lan-${h.name}`}
                                          type="button"
                                          onClick={() => handleExoPickLanHost(h)}
                                          className={`text-xs rounded-md border px-2 py-1 transition-colors ${
                                            h.matches_alias && exoAddAlias === h.matches_alias
                                              ? "border-blue-500 bg-blue-500/10 text-blue-400"
                                              : "border-th-border bg-th-surface text-th-text-secondary hover:bg-th-surface-hover"
                                          }`}
                                          title={titleLines.join("\n")}
                                        >
                                          <Wifi size={10} className="inline mr-1 -mt-0.5" />
                                          <span className="font-mono">{h.name}</span>
                                          {h.matches_alias ? (
                                            <span className="text-emerald-500 ml-1">
                                              ✓ {h.matches_alias}
                                            </span>
                                          ) : tb ? (
                                            <>
                                              <span className="ml-1 px-1 rounded-sm bg-amber-500/20 text-amber-400 text-[10px] font-semibold uppercase tracking-wide">
                                                TB
                                              </span>
                                              <span className="text-th-text-muted ml-1 font-mono">
                                                {tb}
                                              </span>
                                            </>
                                          ) : ipv4 ? (
                                            <span className="text-th-text-muted ml-1 font-mono">
                                              → {ipv4}
                                            </span>
                                          ) : null}
                                        </button>
                                      );
                                    })}
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                        );
                      })()}

                      <div className="flex items-end gap-2">
                        <div className="flex-1 min-w-0">
                          <InputField
                            label="SSH alias / host"
                            value={exoAddAlias}
                            onChange={(v) => {
                              setExoAddAlias(v);
                              setExoSshTestResult(null);
                            }}
                            placeholder="e.g. mini1 or 169.254.144.39"
                          />
                        </div>
                        <div className="w-32 shrink-0">
                          <InputField
                            label="Username"
                            value={exoAddUser}
                            onChange={setExoAddUser}
                            placeholder={(() => {
                              // Try to extract user from alias if it already has @
                              const m = exoAddAlias.match(/^([^@]+)@/);
                              return m ? m[1] : "e.g. alice";
                            })()}
                          />
                        </div>
                      </div>
                      <div className="flex items-end gap-2">
                        <div className="flex-1">
                          {/* invisible spacer to align buttons with alias field */}
                        </div>
                        <button
                          type="button"
                          onClick={handleExoTestSsh}
                          disabled={!exoAddAlias.trim() || exoSshTesting}
                          className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-md border border-th-border bg-th-surface text-th-text-secondary hover:bg-th-surface-hover disabled:opacity-50"
                          title="Run `ssh -o BatchMode=yes <alias> echo ok` to verify the alias before saving"
                        >
                          {exoSshTesting ? (
                            <Loader2 size={12} className="animate-spin" />
                          ) : (
                            <PlugZap size={12} />
                          )}
                          {exoSshTesting ? "Testing…" : "Test SSH"}
                        </button>
                        <button
                          type="button"
                          onClick={handleExoLanScan}
                          disabled={exoLanScanning}
                          className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-md border border-th-border bg-th-surface text-th-text-secondary hover:bg-th-surface-hover disabled:opacity-50"
                          title="Browse the LAN for advertised SSH services (~3 seconds)"
                        >
                          {exoLanScanning ? (
                            <Loader2 size={12} className="animate-spin" />
                          ) : (
                            <Wifi size={12} />
                          )}
                          {exoLanScanning ? "Scanning…" : "Scan LAN"}
                        </button>
                      </div>
                      {(() => {
                        const a = exoAddAlias.trim();
                        if (!a) return null;
                        const isCfgAlias = exoSshHosts.some(
                          (h) => h.alias.toLowerCase() === a.toLowerCase(),
                        );
                        if (isCfgAlias || a.includes("/")) return null;
                        const looksLikeHostname =
                          a.includes(".") || /\d+\.\d+\.\d+\.\d+/.test(a);
                        if (!looksLikeHostname) return null;
                        return (
                          <p className="text-[11px] text-amber-500 leading-relaxed -mt-1">
                            <strong>{a}</strong> isn't a Host alias in{" "}
                            <code>~/.ssh/config</code>. SSH will dial this hostname
                            directly — make sure it resolves and your key is
                            authorized on it. (Tip: add a{" "}
                            <code>Host …</code> block for a friendlier name.)
                          </p>
                        );
                      })()}
                      {exoSshTestResult && (
                        <div
                          className={`rounded-md border px-3 py-2 text-[11px] leading-relaxed ${
                            exoSshTestResult.ok
                              ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-400"
                              : "border-rose-500/40 bg-rose-500/10 text-rose-400"
                          }`}
                        >
                          <div className="font-medium">
                            {exoSshTestResult.ok
                              ? "SSH OK — host is reachable with key auth."
                              : "SSH check failed."}
                          </div>
                          {exoSshTestResult.hint && (
                            <div className="mt-1 text-th-text-secondary">
                              {exoSshTestResult.hint}
                            </div>
                          )}
                          {!exoSshTestResult.ok && exoSshTestResult.stderr && (
                            <pre className="mt-1 max-h-24 overflow-auto whitespace-pre-wrap font-mono text-[10px] text-th-text-muted">
                              {exoSshTestResult.stderr.trim()}
                            </pre>
                          )}
                        </div>
                      )}
                      <InputField
                        label="Label (optional)"
                        value={exoAddLabel}
                        onChange={setExoAddLabel}
                        placeholder="Studio Mac mini"
                      />
                      <InputField
                        label="OTTO_APP_DATA_DIR override (optional)"
                        value={exoAddDataDir}
                        onChange={setExoAddDataDir}
                        placeholder="(defaults to platform standard)"
                      />
                      <p className="text-[11px] text-th-text-muted leading-relaxed">
                        We use the SSH alias to <code>scp</code> the cluster helper and{" "}
                        <code>ssh</code> in for install &amp; start. The alias must
                        authenticate non-interactively (key-based). LAN-discovered hosts
                        without a matching alias will need a{" "}
                        <code className="font-mono">Host …</code> block in{" "}
                        <code className="font-mono">~/.ssh/config</code> first.
                      </p>
                      <div className="flex justify-end gap-2 pt-1">
                        <button
                          type="button"
                          className="px-3 py-1.5 text-xs font-medium rounded-md border border-th-border bg-th-surface text-th-text-secondary hover:bg-th-surface-hover"
                          onClick={() => {
                            setExoShowAddRemote(false);
                            setExoAddAlias("");
                            setExoAddUser("");
                            setExoAddLabel("");
                            setExoAddDataDir("");
                          }}
                        >
                          Cancel
                        </button>
                        <button
                          type="button"
                          className="px-3 py-1.5 text-xs font-semibold rounded-md bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-40"
                          onClick={handleExoAddRemote}
                          disabled={!exoAddAlias.trim() || exoBusy === "add"}
                        >
                          {exoBusy === "add" ? "Adding…" : "Add"}
                        </button>
                      </div>
                        </>
                      )}
                    </div>
                  )}

                  {settings.exo.remotes.length === 0 ? (
                    <div className="flex flex-col items-center gap-2 py-8 text-center">
                      <Server size={28} className="text-th-text-muted opacity-40" />
                      <p className="text-xs font-medium text-th-text-secondary">No remote nodes</p>
                      <p className="text-[11px] text-th-text-muted max-w-xs leading-relaxed">
                        Add a secondary Mac or Linux machine via SSH to distribute model shards across multiple devices.
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      {settings.exo.remotes.map((r: ExoRemote) => {
                        const peer = remotePeerMap.get(r.ssh_alias) ?? null;
                        const localKnown =
                          settings.exo.enabled && exoLiveStatus !== null;
                        let badgeKind: "joined" | "absent" | "disabled" | "unknown";
                        if (!r.enabled) badgeKind = "disabled";
                        else if (peer) badgeKind = "joined";
                        else if (localKnown && exoLiveStatus?.reachable)
                          badgeKind = "absent";
                        else badgeKind = "unknown";
                        const badgeStyles: Record<typeof badgeKind, string> = {
                          joined:
                            "bg-emerald-500/15 text-emerald-500 border-emerald-500/30",
                          absent:
                            "bg-amber-500/15 text-amber-500 border-amber-500/30",
                          disabled:
                            "bg-neutral-500/10 text-th-text-muted border-th-border",
                          unknown:
                            "bg-th-surface text-th-text-tertiary border-th-border",
                        };
                        const badgeLabel: Record<typeof badgeKind, string> = {
                          joined: "in cluster",
                          absent: "not joined",
                          disabled: "disabled",
                          unknown: "—",
                        };
                        return (
                        <div
                          key={r.ssh_alias}
                          className={`rounded-xl border transition-colors ${
                            badgeKind === "joined"
                              ? "border-emerald-500/20 bg-emerald-500/5"
                              : badgeKind === "absent"
                                ? "border-amber-500/20 bg-th-surface"
                                : "border-th-border bg-th-surface"
                          }`}
                        >
                          {/* Node header */}
                          <div className="flex flex-wrap items-center gap-3 px-3.5 py-3">
                            <div className={`h-8 w-8 rounded-lg flex items-center justify-center shrink-0 ${
                              badgeKind === "joined" ? "bg-emerald-500/15"
                                : badgeKind === "absent" ? "bg-amber-500/10"
                                : "bg-th-inset-bg"
                            }`}>
                              <Server size={15} className={
                                badgeKind === "joined" ? "text-emerald-500"
                                  : badgeKind === "absent" ? "text-amber-400"
                                  : "text-th-text-tertiary"
                              } />
                            </div>
                            <div className="min-w-0 flex-1">
                              <div className="flex items-center gap-2 flex-wrap">
                                <span className="text-xs font-semibold text-th-text-primary truncate">
                                  {r.label || r.ssh_alias}
                                </span>
                                {r.label && (
                                  <code className="text-[10px] font-mono text-th-text-muted bg-th-inset-bg px-1.5 py-0.5 rounded">
                                    {r.ssh_alias}
                                  </code>
                                )}
                                <span
                                  className={`inline-flex items-center gap-1 rounded-full border px-1.5 py-[1px] text-[10px] font-medium ${badgeStyles[badgeKind]}`}
                                  title={
                                    peer
                                      ? `Reported as ${peer.friendly_name ?? peer.node_id.slice(0, 12)}…${peer.memory_total_gb ? ` · ${peer.memory_free_gb ?? "?"} / ${peer.memory_total_gb} GB free` : ""}`
                                      : badgeKind === "absent"
                                        ? "Cluster is reachable but this remote isn't reporting as a peer. Try Start."
                                        : badgeKind === "disabled"
                                          ? "Disabled — master will skip it on auto-start."
                                          : "Cluster offline or status unknown."
                                  }
                                >
                                  <span className={`h-1.5 w-1.5 rounded-full ${
                                    badgeKind === "joined" ? "bg-emerald-500"
                                      : badgeKind === "absent" ? "bg-amber-500"
                                      : "bg-neutral-400"
                                  }`} />
                                  {badgeLabel[badgeKind]}
                                </span>
                              </div>
                              {(peer?.chip || peer?.memory_total_gb || r.app_data_dir) && (
                                <div className="flex items-center gap-2 mt-0.5 flex-wrap">
                                  {peer?.chip && <span className="text-[10px] text-th-text-muted">{peer.chip}</span>}
                                  {peer?.memory_total_gb && (
                                    <span className="text-[10px] text-th-text-muted">{peer.memory_free_gb ?? "?"} / {peer.memory_total_gb} GB free</span>
                                  )}
                                  {r.app_data_dir && (
                                    <span className="text-[10px] text-th-text-muted font-mono truncate max-w-[200px]">{r.app_data_dir}</span>
                                  )}
                                </div>
                              )}
                            </div>

                            {/* Actions */}
                            <div className="flex items-center gap-1.5 shrink-0 flex-wrap">
                              <label className="flex items-center gap-1 text-[11px] text-th-text-tertiary cursor-pointer select-none">
                                <input
                                  type="checkbox"
                                  checked={r.enabled}
                                  disabled={exoBusy === `toggle-${r.ssh_alias}`}
                                  onChange={(e) => handleExoToggleRemote(r.ssh_alias, e.target.checked)}
                                  className="h-3.5 w-3.5 rounded border-th-border accent-blue-600"
                                />
                                {r.enabled ? "on" : "off"}
                              </label>
                              <button
                                type="button"
                                className="inline-flex items-center gap-1 text-xs font-medium rounded-lg bg-blue-600 text-white px-2.5 py-1.5 hover:bg-blue-500 disabled:opacity-40 transition-colors"
                                onClick={() => handleExoRemoteUp(r.ssh_alias)}
                                disabled={!settings.exo.enabled || !r.enabled || exoBusy === `up-${r.ssh_alias}`}
                                title="Deploy the cluster helper and start the remote daemon"
                              >
                                {exoBusy === `up-${r.ssh_alias}` ? <Loader2 size={11} className="animate-spin" /> : <Play size={11} />}
                                Start
                              </button>
                              <button
                                type="button"
                                className="inline-flex items-center gap-1 text-xs font-medium rounded-lg bg-th-surface border border-th-border text-th-text-secondary px-2.5 py-1.5 hover:bg-th-surface-hover disabled:opacity-40 transition-colors"
                                onClick={() => handleExoRemoteDown(r.ssh_alias)}
                                disabled={!settings.exo.enabled || exoBusy === `down-${r.ssh_alias}`}
                              >
                                {exoBusy === `down-${r.ssh_alias}` ? <Loader2 size={11} className="animate-spin" /> : <Square size={11} />}
                                Stop
                              </button>
                              {exoConfirmRemove === r.ssh_alias ? (
                                <div className="flex items-center gap-1">
                                  <button
                                    type="button"
                                    onClick={() => handleExoRemoveRemote(r.ssh_alias)}
                                    disabled={exoBusy === `remove-${r.ssh_alias}`}
                                    className="rounded-lg bg-red-500/15 text-red-500 hover:bg-red-500/25 p-1.5 transition-colors"
                                    title="Confirm remove"
                                  >
                                    <Trash2 size={12} />
                                  </button>
                                  <button
                                    type="button"
                                    onClick={() => setExoConfirmRemove(null)}
                                    className="rounded-lg p-1.5 text-th-text-muted hover:bg-th-surface-hover transition-colors"
                                  >
                                    <X size={12} />
                                  </button>
                                </div>
                              ) : (
                                <button
                                  type="button"
                                  onClick={() => setExoConfirmRemove(r.ssh_alias)}
                                  className="rounded-lg p-1.5 text-th-text-muted hover:text-red-500 hover:bg-red-500/10 transition-colors"
                                  title="Remove this node"
                                >
                                  <Trash2 size={12} />
                                </button>
                              )}
                            </div>
                          </div>

                          {/* Per-remote operation notice */}
                          {(exoOpNotices[`up-${r.ssh_alias}`] ?? exoOpNotices[`down-${r.ssh_alias}`]) && !remoteUpJobs[r.ssh_alias] && (() => {
                            const notice = exoOpNotices[`up-${r.ssh_alias}`] ?? exoOpNotices[`down-${r.ssh_alias}`];
                            return (
                              <div className={`mx-3.5 mb-3 flex items-center gap-1.5 px-3 py-2 rounded-lg text-[11px] border ${
                                notice.kind === "progress"
                                  ? "border-blue-500/20 bg-blue-500/5 text-blue-400"
                                  : notice.kind === "success"
                                    ? "border-emerald-500/20 bg-emerald-500/5 text-emerald-400"
                                    : "border-red-500/20 bg-red-500/5 text-red-400"
                              }`}>
                                {notice.kind === "progress" ? <Loader2 size={11} className="animate-spin shrink-0" />
                                  : notice.kind === "success" ? <CheckCircle2 size={11} className="shrink-0" />
                                  : <AlertTriangle size={11} className="shrink-0" />}
                                {notice.message}
                              </div>
                            );
                          })()}

                          {/* Live job progress for "Provision & start" */}
                          {remoteUpJobs[r.ssh_alias] && (() => {
                            const job = remoteUpJobs[r.ssh_alias];
                            const running = job.status === "running" || job.status === "pending";
                            const done = job.status === "done";
                            const failed = job.status === "error";
                            return (
                              <div className={`mx-3.5 mb-3 rounded-lg border text-[11px] ${
                                running ? "border-blue-500/20 bg-blue-500/5"
                                  : done ? "border-emerald-500/20 bg-emerald-500/5"
                                  : "border-red-500/20 bg-red-500/5"
                              }`}>
                                <div className="flex items-center gap-2 px-3 py-2">
                                  {running ? <Loader2 size={12} className="animate-spin text-blue-400 shrink-0" />
                                    : done ? <CheckCircle2 size={12} className="text-emerald-500 shrink-0" />
                                    : <AlertTriangle size={12} className="text-red-500 shrink-0" />}
                                  <span className={`font-medium ${running ? "text-blue-400" : done ? "text-emerald-500" : "text-red-500"}`}>
                                    {running ? "Provisioning & starting…" : done ? "Started successfully" : "Start failed"}
                                  </span>
                                  {!running && (
                                    <button
                                      type="button"
                                      onClick={() => setRemoteUpJobs((p) => { const n = { ...p }; delete n[r.ssh_alias]; return n; })}
                                      className="ml-auto text-th-text-muted hover:text-th-text-primary"
                                    >
                                      <X size={11} />
                                    </button>
                                  )}
                                </div>
                                {job.log_lines.length > 0 && (
                                  <details open={running || failed}>
                                    <summary className="cursor-pointer px-3 pb-1 text-th-text-muted hover:text-th-text-secondary list-none text-[10px]">
                                      {job.log_lines.length} log line{job.log_lines.length !== 1 ? "s" : ""}
                                    </summary>
                                    <pre className="mx-3 mb-2 max-h-40 overflow-auto rounded-lg bg-black/80 p-2 text-[10px] text-emerald-100 font-mono whitespace-pre-wrap">
                                      {job.log_lines.join("\n")}
                                    </pre>
                                  </details>
                                )}
                                {failed && job.error && (
                                  <div className="flex items-start gap-1.5 px-3 pb-2 text-red-400">
                                    <AlertTriangle size={10} className="shrink-0 mt-0.5" />
                                    <span>{job.error}</span>
                                  </div>
                                )}
                              </div>
                            );
                          })()}
                        </div>
                        );
                      })}
                    </div>
                  )}

                  {exoErr && (
                    <p className="text-xs text-red-500 flex items-center gap-1">
                      <XCircle size={12} />
                      {exoErr}
                    </p>
                  )}
                </div>
                </>
                )}

              </div>
              )}
            </div>
            )}
          </div>
        )}

        {tab === "Agent Memory" && (
          <MemoryPanel />
        )}

        {tab === "Suggestions" && (
          <AmbientPanel
            settings={settings}
            setSettings={(updater) => {
              setSettings((prev) => {
                const next = typeof updater === "function" ? updater(prev) : updater;
                return next ?? prev;
              });
            }}
            mlxLocalModels={mlxLocalModels}
            mlxListLoading={mlxListLoading}
            mlxListError={mlxListError}
            onRefreshMlxModels={() => void refreshMlxModels()}
          />
        )}

        {tab === "macOS Activity" && (() => {
          const ACTIVITY_DEFAULTS = {
            enabled: false,
            interval_secs: 15,
            retain_days: 30,
            exclude_apps: [] as string[],
            idle_threshold_secs: 180,
            min_span_secs: 3,
            max_span_secs: 600,
            context_max_chars: 4096,
            field_val_max_chars: 8000,
            browser_text_max_chars: 4000,
            ax_walk_max_chars: 3000,
            ax_walk_max_depth: 25,
            max_db_mb: 5120,
          };
          const a = settings.activity ?? ACTIVITY_DEFAULTS;
          const setA = (patch: Partial<typeof ACTIVITY_DEFAULTS>) =>
            setSettings((s) => ({
              ...s,
              activity: { ...ACTIVITY_DEFAULTS, ...(s.activity ?? {}), ...patch },
            }));
          return (
          <div className="space-y-6 max-w-2xl">
            <Card title="Activity timeline" dot={a.enabled ? "bg-emerald-400" : "bg-neutral-400"}>
              <div className="space-y-4">
                <p className="text-xs text-th-text-tertiary leading-relaxed">
                  Records the foreground app, window title, and browser URL on a timer.
                  No screenshots — only metadata, stored locally in SQLite. The agent can
                  search this timeline via <span className="font-mono">search_screen_history</span>.
                  macOS only; requires Accessibility permission for window titles.
                </p>
                <Toggle
                  label="Enable activity tracking"
                  checked={!!a.enabled}
                  onChange={(v) => setA({ enabled: v })}
                />
                <InputField
                  label="Polling interval (seconds)"
                  type="number"
                  min={5}
                  max={300}
                  value={String(a.interval_secs ?? 15)}
                  onChange={(v) => {
                    const n = Math.max(5, Math.min(300, parseInt(v) || 15));
                    setA({ interval_secs: n });
                  }}
                />
                <InputField
                  label="Retention (days, 0 = keep forever)"
                  type="number"
                  min={0}
                  max={3650}
                  value={String(a.retain_days ?? 30)}
                  onChange={(v) => {
                    const n = Math.max(0, Math.min(3650, parseInt(v) || 0));
                    setA({ retain_days: n });
                  }}
                />
                <InputField
                  label="Storage cap (MB, 0 = unlimited)"
                  type="number"
                  min={0}
                  max={102400}
                  value={String(a.max_db_mb ?? 5120)}
                  onChange={(v) => {
                    const n = Math.max(0, Math.min(102400, parseInt(v) || 0));
                    setA({ max_db_mb: n });
                  }}
                />
                <p className="text-[10px] text-th-text-muted -mt-3">
                  When the database exceeds this size, the oldest records are automatically deleted until it fits. Runs hourly alongside the retention pass. 0 = no cap.
                </p>
                <InputField
                  label="Idle threshold (seconds, 0 = always record)"
                  type="number"
                  min={0}
                  max={3600}
                  value={String(a.idle_threshold_secs ?? 180)}
                  onChange={(v) => {
                    const n = Math.max(0, Math.min(3600, parseInt(v) || 0));
                    setA({ idle_threshold_secs: n });
                  }}
                />
                <p className="text-[10px] text-th-text-muted -mt-3">
                  Skip recording when there's been no keyboard or mouse input for this long. Big DB-size win for users who leave the laptop on overnight.
                </p>
                <InputField
                  label="Minimum span (seconds, 0 = keep all)"
                  type="number"
                  min={0}
                  max={300}
                  value={String(a.min_span_secs ?? 3)}
                  onChange={(v) => {
                    const n = Math.max(0, Math.min(300, parseInt(v) || 0));
                    setA({ min_span_secs: n });
                  }}
                />
                <p className="text-[10px] text-th-text-muted -mt-3">
                  Drop spans shorter than this — cleans up tab-flicking noise without losing the windows you actually used.
                </p>
                <InputField
                  label="Max span before sharding (seconds, 0 = no cap)"
                  type="number"
                  min={0}
                  max={86400}
                  value={String(a.max_span_secs ?? 600)}
                  onChange={(v) => {
                    const n = Math.max(0, Math.min(86400, parseInt(v) || 0));
                    setA({ max_span_secs: n });
                  }}
                />
                <p className="text-[10px] text-th-text-muted -mt-3">
                  Force a new row after this many seconds on the same window. Without sharding, a 90-min focus session collapses into one row that loses everything that happened inside. 600s ≈ 6 rows/hour during deep focus.
                </p>
                <InputField
                  label="Context history size (chars, 0 = overwrite-only)"
                  type="number"
                  min={0}
                  max={32768}
                  value={String(a.context_max_chars ?? 4096)}
                  onChange={(v) => {
                    const n = Math.max(0, Math.min(32768, parseInt(v) || 0));
                    setA({ context_max_chars: n });
                  }}
                />
                <p className="text-[10px] text-th-text-muted -mt-3">
                  Each row keeps a rolling log of distinct selections / typed inputs seen during the span. When this limit is hit, the oldest entries are trimmed from the front. Set to 0 to keep only the latest snapshot (legacy behaviour).
                </p>
                <InputField
                  label="Field value capture limit (chars, 0 = disabled)"
                  type="number"
                  min={0}
                  max={65536}
                  value={String(a.field_val_max_chars ?? 8000)}
                  onChange={(v) => {
                    const n = Math.max(0, Math.min(65536, parseInt(v) || 0));
                    setA({ field_val_max_chars: n });
                  }}
                />
                <p className="text-[10px] text-th-text-muted -mt-3">
                  Max characters captured from the active text field each tick. Native apps like Notes, Mail, Pages, and Xcode expose their full document content here. 8000 ≈ one full page of writing. Set to 0 to disable.
                </p>
                <InputField
                  label="Browser page text capture (chars, 0 = disabled)"
                  type="number"
                  min={0}
                  max={20000}
                  value={String(a.browser_text_max_chars ?? 4000)}
                  onChange={(v) => {
                    const n = Math.max(0, Math.min(20000, parseInt(v) || 0));
                    setA({ browser_text_max_chars: n });
                  }}
                />
                <p className="text-[10px] text-th-text-muted -mt-3">
                  Pulls the visible page body text (title + headings + body) from the active tab in Safari, Chrome, Brave, and Arc via AppleScript. Massively richer than URL alone.
                  <br />
                  <span className="text-amber-400">Safari requires</span> Develop → "Allow JavaScript from Apple Events".
                  <span className="text-amber-400"> Chrome/Brave require</span> View → Developer → "Allow JavaScript from Apple Events". Set to 0 to disable.
                </p>
                <InputField
                  label="UI tree walk capture (chars, 0 = disabled)"
                  type="number"
                  min={0}
                  max={10000}
                  value={String(a.ax_walk_max_chars ?? 2000)}
                  onChange={(v) => {
                    const n = Math.max(0, Math.min(10000, parseInt(v) || 0));
                    setA({ ax_walk_max_chars: n });
                  }}
                />
                <p className="text-[10px] text-th-text-muted -mt-3">
                  Walks the Accessibility tree of non-browser apps and harvests visible text. Captures rich context for Electron apps (Cursor, VS Code, Slack, Discord, Notion) and AppKit apps with multi-pane layouts. Tab labels, file tree entries, status bar — anything AX exposes.
                </p>
                <InputField
                  label="UI tree walk max depth"
                  type="number"
                  min={1}
                  max={40}
                  value={String(a.ax_walk_max_depth ?? 25)}
                  onChange={(v) => {
                    const n = Math.max(1, Math.min(40, parseInt(v) || 25));
                    setA({ ax_walk_max_depth: n });
                  }}
                />
                <p className="text-[10px] text-th-text-muted -mt-3">
                  Recursion depth when descending the AX tree. Electron apps (Cursor, Slack, Discord, VS Code, Notion) nest text inside 15-25 layers of empty Chromium AXGroup wrappers before reaching the actual AXStaticText leaves. Native AppKit apps are much shallower (5-8 levels). 25 is the sweet spot.
                </p>
                <div>
                  <label className="block text-xs font-medium text-th-text-secondary mb-1.5">
                    Excluded apps (one per line, case-insensitive substring match)
                  </label>
                  <textarea
                    rows={4}
                    value={(a.exclude_apps ?? []).join("\n")}
                    onChange={(e) => {
                      const list = e.target.value
                        .split("\n")
                        .map((x) => x.trim())
                        .filter(Boolean);
                      setA({ exclude_apps: list });
                    }}
                    className="w-full px-3 py-2 rounded-lg bg-th-input-bg border border-th-input-border text-sm text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-400 transition-colors font-mono"
                    placeholder="1Password\nBitwarden\nBanking"
                  />
                  <p className="text-[10px] text-th-text-muted mt-1">
                    Apps whose name contains any of these strings will never be recorded.
                  </p>
                </div>
              </div>
            </Card>
          </div>
          );
        })()}

        {tab === "Voice" && settings && (
          <VoiceSettingsPanel settings={settings} setSettings={setSettings} onSave={persistSettings} />
        )}

        {tab === "Advanced" && (
          <div className="space-y-6 max-w-2xl">
            <Card title="Setup wizard" dot="bg-th-tab-active-bg">
              <div className="space-y-3">
                <p className="text-xs text-th-text-tertiary leading-relaxed">
                  Re-open the first-run setup wizard. Useful if you skipped it
                  before or want to walk through model / memory / activity choices
                  again. Your existing settings remain unchanged until you save a
                  step.
                </p>
                <button
                  type="button"
                  onClick={async () => {
                    try {
                      await api.setupReset();
                      window.location.reload();
                    } catch (e) {
                      console.warn("setup reset failed", e);
                    }
                  }}
                  className="inline-flex items-center gap-2 px-4 py-2 text-sm font-semibold rounded-xl bg-gradient-to-r from-blue-500/15 to-sky-500/15 border border-blue-500/30 text-blue-400 hover:from-blue-500/25 hover:to-sky-500/25 hover:border-blue-500/50 hover:text-blue-300 transition-all duration-150 shadow-sm"
                >
                  <Wand2 size={14} className="shrink-0" /> Re-run setup wizard
                </button>
              </div>
            </Card>
            {settings.llm.provider === "anthropic" && (
              <>
                <Card title="Extended Thinking" dot="bg-neutral-400">
                  <div className="space-y-4">
                    <Toggle label="Enable extended thinking" checked={settings.llm.anthropic.thinking_enabled} onChange={(v) => updateAnthropicField("thinking_enabled", v)} />
                    {settings.llm.anthropic.thinking_enabled && (
                      <InputField
                        label="Budget Tokens"
                        value={String(settings.llm.anthropic.thinking_budget)}
                        onChange={(v) => {
                          const maxBudget = settings.llm.anthropic.max_tokens - 1;
                          const parsed = parseInt(v) || 1024;
                          updateAnthropicField("thinking_budget", Math.min(Math.max(parsed, 1024), maxBudget));
                        }}
                        type="number"
                        min={1024}
                        max={settings.llm.anthropic.max_tokens - 1}
                      />
                    )}
                  </div>
                </Card>
                <Card title="Efficiency" dot="bg-emerald-400">
                  <Toggle label="Efficient tool calls (reduces token usage)" checked={settings.llm.anthropic.tool_efficient} onChange={(v) => updateAnthropicField("tool_efficient", v)} />
                </Card>
              </>
            )}
            {settings.llm.provider === "mlx" && (
              <p className="text-sm text-gray-500">Extended thinking and efficient tool calls apply to Anthropic / Bedrock only. MLX uses Hub model IDs from the LLM tab.</p>
            )}
            <Card title="Agent execution" dot="bg-sky-400">
              <div className="space-y-4">
                <p className="text-xs text-th-text-tertiary">
                  Universal limit applied to the orchestrator and every subagent (general-purpose, web-voyager, computer-voyager). Raise for very long autonomous tasks; lower to catch runaway loops early.
                </p>
                <InputField
                  label="Recursion limit (max steps per run)"
                  value={String(settings.orchestrator.recursion_limit ?? 10000)}
                  onChange={(v) => {
                    const parsed = parseInt(v) || 1000;
                    setSettings((s) => ({
                      ...s,
                      orchestrator: {
                        ...s.orchestrator,
                        recursion_limit: Math.min(Math.max(parsed, 1), 10000),
                      },
                    }));
                  }}
                  type="number"
                  min={1}
                  max={10000}
                />
                <div className="pt-1 border-t border-th-border">
                  <Toggle
                    label="Auto-approve commands"
                    checked={settings.auto_approve_commands ?? false}
                    onChange={(v) => setSettings((s) => ({ ...s, auto_approve_commands: v }))}
                  />
                  <p className="text-xs text-th-text-tertiary mt-2 leading-relaxed">
                    Automatically approve shell commands without prompting. Commands flagged as high-risk (e.g. <code className="font-mono text-th-text-secondary">rm -rf</code>, force-push, raw disk writes) are always held for manual review.
                  </p>
                </div>
              </div>
            </Card>
            <Card title="Legacy orchestrator override" dot="bg-neutral-400">
              <p className="text-xs text-th-text-tertiary mb-3">
                When set, overrides the LLM tab orchestrator choice and maps directly to{" "}
                <code className="font-mono text-th-text-secondary">DEEP_AGENT_LLM_PROVIDER</code>. Leave empty to use the LLM tab setting.
              </p>
              <InputField
                label="DEEP_AGENT provider override"
                value={settings.orchestrator.provider_override ?? ""}
                onChange={(v) =>
                  setSettings((s) => ({
                    ...s,
                    orchestrator: { ...s.orchestrator, provider_override: v.trim() ? v.trim().toLowerCase() : null },
                  }))
                }
                placeholder="e.g. anthropic or mlx (optional)"
              />
            </Card>
            <Card title="Ambient scheduling" dot="bg-blue-400">
              <div className="space-y-3">
                <Toggle
                  label="Suggest scheduling for repeatable tasks"
                  checked={settings.ambient_suggest_recurrence ?? false}
                  onChange={(v) => setSettings((s) => ({ ...s, ambient_suggest_recurrence: v }))}
                />
                <p className="text-xs text-th-text-tertiary leading-relaxed">
                  When enabled, Otto asks at the end of responses whether you'd like to automate
                  repeatable tasks — scheduling them to run on a cron, triggering them on file
                  events, or repeating them once. Otto uses its scheduling tools directly when
                  you say yes, with no extra steps.
                </p>
              </div>
            </Card>
          </div>
        )}

        {tab === "Observability" && (
          <div className="space-y-6 max-w-2xl">
            <Card title="Run Evaluation" dot="bg-blue-400">
              <div className="space-y-4">
                <Toggle
                  label="Auto-evaluate completed runs"
                  description="When a run finishes, an LLM picks suitable metrics and scores it automatically. Turn this off to evaluate runs manually with the Evaluate button on each run."
                  checked={settings.evaluation?.auto_evaluate ?? false}
                  onChange={(v) => setSettings((s) => ({ ...s, evaluation: { ...s.evaluation, auto_evaluate: v } }))}
                />
                <Toggle
                  label="Analyze failed runs"
                  description="When a run ends in an error, classify the failure and — only when a better prompt could plausibly help — draft a stronger prompt, surfaced on the run's Evaluation tab and in the Suggestions inbox."
                  checked={settings.evaluation?.analyze_errors ?? false}
                  onChange={(v) => setSettings((s) => ({ ...s, evaluation: { ...s.evaluation, analyze_errors: v } }))}
                />
                <SelectField
                  label="Judge model"
                  value={settings.evaluation?.llm_family ?? "follow_main"}
                  onChange={(v) => setSettings((s) => ({ ...s, evaluation: { ...s.evaluation, llm_family: v } }))}
                  options={[
                    { value: "follow_main", label: "Follow main provider" },
                    { value: "frontier", label: "Frontier (Anthropic / Bedrock)" },
                    { value: "mlx", label: "Local MLX" },
                  ]}
                />
                <div className="grid grid-cols-2 gap-4">
                  {/* Stepper: Max metrics */}
                  <div>
                    <label className="block text-sm font-medium text-th-text-tertiary mb-2">Max metrics</label>
                    <div className="flex items-center border border-th-input-border rounded-lg bg-th-input-bg overflow-hidden focus-within:border-blue-400 focus-within:ring-1 focus-within:ring-blue-300/30 transition-all">
                      <button
                        type="button"
                        className="px-3 py-2.5 text-th-text-secondary hover:text-th-text-primary hover:bg-th-surface-hover transition-colors text-sm font-bold select-none shrink-0"
                        onClick={() => setSettings((s) => ({ ...s, evaluation: { ...s.evaluation, max_metrics: Math.max(1, (s.evaluation?.max_metrics ?? 4) - 1) } }))}
                      >−</button>
                      <input
                        type="number"
                        min={1}
                        max={10}
                        className="flex-1 text-center text-sm text-th-text-primary bg-transparent focus:outline-none tabular-nums [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                        value={settings.evaluation?.max_metrics ?? 4}
                        onChange={(e) => {
                          const n = Math.max(1, Math.min(10, parseInt(e.target.value, 10) || 1));
                          setSettings((s) => ({ ...s, evaluation: { ...s.evaluation, max_metrics: n } }));
                        }}
                      />
                      <button
                        type="button"
                        className="px-3 py-2.5 text-th-text-secondary hover:text-th-text-primary hover:bg-th-surface-hover transition-colors text-sm font-bold select-none shrink-0"
                        onClick={() => setSettings((s) => ({ ...s, evaluation: { ...s.evaluation, max_metrics: Math.min(10, (s.evaluation?.max_metrics ?? 4) + 1) } }))}
                      >+</button>
                    </div>
                  </div>
                  {/* Stepper: Pass threshold */}
                  <div>
                    <label className="block text-sm font-medium text-th-text-tertiary mb-2">Pass threshold</label>
                    <div className="flex items-center border border-th-input-border rounded-lg bg-th-input-bg overflow-hidden focus-within:border-blue-400 focus-within:ring-1 focus-within:ring-blue-300/30 transition-all">
                      <button
                        type="button"
                        className="px-3 py-2.5 text-th-text-secondary hover:text-th-text-primary hover:bg-th-surface-hover transition-colors text-sm font-bold select-none shrink-0"
                        onClick={() => setSettings((s) => ({ ...s, evaluation: { ...s.evaluation, threshold: Math.max(0, Math.round(((s.evaluation?.threshold ?? 0.5) - 0.05) * 100) / 100) } }))}
                      >−</button>
                      <div className="flex-1 flex items-center justify-center gap-0.5">
                        <input
                          type="number"
                          min={0}
                          max={100}
                          className="w-10 text-center text-sm text-th-text-primary bg-transparent focus:outline-none tabular-nums [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                          value={Math.round((settings.evaluation?.threshold ?? 0.5) * 100)}
                          onChange={(e) => {
                            const n = Math.max(0, Math.min(100, parseInt(e.target.value, 10) || 0));
                            setSettings((s) => ({ ...s, evaluation: { ...s.evaluation, threshold: Math.round(n) / 100 } }));
                          }}
                        />
                        <span className="text-sm text-th-text-muted">%</span>
                      </div>
                      <button
                        type="button"
                        className="px-3 py-2.5 text-th-text-secondary hover:text-th-text-primary hover:bg-th-surface-hover transition-colors text-sm font-bold select-none shrink-0"
                        onClick={() => setSettings((s) => ({ ...s, evaluation: { ...s.evaluation, threshold: Math.min(1, Math.round(((s.evaluation?.threshold ?? 0.5) + 0.05) * 100) / 100) } }))}
                      >+</button>
                    </div>
                  </div>
                </div>
              </div>
            </Card>
            <Card title="LangSmith Tracing" dot="bg-neutral-400">
              <div className="space-y-4">
                <Toggle label="Enable LangSmith tracing" checked={settings.observability.langsmith.enabled} onChange={(v) => setSettings((s) => ({ ...s, observability: { ...s.observability, langsmith: { ...s.observability.langsmith, enabled: v } } }))} />
                {settings.observability.langsmith.enabled && (
                  <div className="space-y-4">
                    <SecretField label="API Key" value={settings.observability.langsmith.api_key} onChange={(v) => setSettings((s) => ({ ...s, observability: { ...s.observability, langsmith: { ...s.observability.langsmith, api_key: v } } }))} />
                    <InputField label="Project" value={settings.observability.langsmith.project} onChange={(v) => setSettings((s) => ({ ...s, observability: { ...s.observability, langsmith: { ...s.observability.langsmith, project: v } } }))} />
                  </div>
                )}
              </div>
            </Card>
            <Card title="Logging" dot="bg-neutral-400">
              <SelectField label="Log Level" value={settings.observability.log_level} onChange={(v) => setSettings((s) => ({ ...s, observability: { ...s.observability, log_level: v } }))} options={[{ value: "DEBUG", label: "DEBUG" }, { value: "INFO", label: "INFO" }, { value: "WARNING", label: "WARNING" }, { value: "ERROR", label: "ERROR" }]} />
            </Card>
          </div>
        )}

        {tab === "Privacy & Security" && (
          <PrivacyTab />
        )}

        {tab === "About" && (
          <div className="max-w-2xl">
            <Card>
              <div className="flex items-center gap-4 mb-5">
                <div>
                  <h2 className="text-lg font-bold text-gray-900 tracking-widest">OTTO</h2>
                  <p className="text-sm text-gray-500">{appVersion ? `Version ${appVersion}` : ""}</p>
                </div>
              </div>
              <p className="text-sm text-gray-500 leading-relaxed">Agent Platform</p>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// oMLX quick panel — minimal status / install / start surface used inside the
// Default-model card when ``provider === "omlx"``.  Full UX (model browser,
// continuous-batching stats, etc.) lives in the setup wizard for now; we
// surface enough here that a returning user can re-pin a default model and
// flip the server up/down without leaving Settings.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// ExoReleaseCheckButton — one-shot probe of the upstream exo-explore/exo repo
// to see whether the user's pinned ``repo_ref`` lags behind the latest tag.
// We deliberately do NOT auto-update the field — bumping the ref also forces
// a re-provision and may trip the MLX preflight, so the user gets a "yes
// there's a newer tag, here's the link" nudge and stays in control.
// ---------------------------------------------------------------------------

function ExoReleaseCheckButton({ currentRef }: { currentRef: string }) {
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<
    | { ok: true; latest: string; newer: boolean; url: string; published: string }
    | { ok: false; error: string }
    | null
  >(null);

  const runCheck = async () => {
    setBusy(true);
    setResult(null);
    try {
      const r = await api.exoReleaseCheck();
      if (r.ok) {
        setResult({
          ok: true,
          latest: r.latest_tag || "",
          newer: !!r.newer_available,
          url: r.html_url || "",
          published: r.published_at || "",
        });
      } else {
        setResult({ ok: false, error: r.error || "unknown error" });
      }
    } catch (exc) {
      setResult({ ok: false, error: String(exc) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex items-start gap-3">
      <button
        type="button"
        onClick={runCheck}
        disabled={busy}
        className="inline-flex items-center gap-1.5 rounded-md border border-th-border bg-th-bg px-2.5 py-1.5 text-[11px] font-medium text-th-text-secondary hover:bg-th-inset-bg disabled:opacity-50"
      >
        {busy ? (
          <Loader2 size={12} className="animate-spin" />
        ) : (
          <RefreshCw size={12} />
        )}
        Check for a new cluster release
      </button>
      {result && (
        <div className="flex-1 text-[11px] leading-relaxed">
          {result.ok ? (
            result.newer ? (
              <span className="text-amber-600">
                Newer release available: <span className="font-mono">{result.latest}</span>
                {result.published && (
                  <span className="text-th-text-muted">
                    {" "}· {new Date(result.published).toLocaleDateString()}
                  </span>
                )}{" "}
                — your pin is <span className="font-mono">{currentRef || "(unset)"}</span>.{" "}
                {result.url && (
                  <a
                    href={result.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-th-accent underline hover:opacity-80"
                  >
                    view release notes
                  </a>
                )}
              </span>
            ) : (
              <span className="text-emerald-600">
                You're on the latest release ({result.latest || currentRef}).
              </span>
            )
          ) : (
            <span className="text-th-text-muted">
              Couldn't reach GitHub: {result.error}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

// Thin model-selector shown on the Model Provider tab when oMLX is active.
// The full server panel lives on the Turbo tab.
function OmlxModelSelector({
  settings,
  setSettings,
  onGoToOnDevice,
}: {
  settings: AppSettings;
  setSettings: React.Dispatch<React.SetStateAction<AppSettings>>;
  onGoToOnDevice: () => void;
}) {
  // Options are built from the local HF cache (proper repo IDs), NOT from the
  // oMLX server's /v1/models which returns internal short-hashes that don't
  // match what was saved as omlx.model_name during setup.
  const [localModels, setLocalModels] = useState<{ value: string; label: string }[]>([]);
  const [reachable, setReachable] = useState(false);
  const [loadErr, setLoadErr] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const s = await api.omlxStatus();
      setReachable(!!s.reachable);
    } catch { /* ignore */ }
    try {
      const { models } = await api.omlxLocalModels();
      const mlx = models.filter((m) => m.is_mlx && m.repo_id.includes("/"));
      setLocalModels(mlx.map((m) => ({
        value: m.repo_id,
        label: `${m.repo_id.split("/")[1] ?? m.repo_id} (${m.size_gb.toFixed(1)} GB)`,
      })));
    } catch { /* ignore */ }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);

  const handleLoad = useCallback(async (modelId: string) => {
    setLoadErr(null);
    try {
      await loadOmlxModelAndWait(modelId);
      setSettings((s) => ({ ...s, omlx: { ...s.omlx, enabled: true, model_name: modelId } }));
    } catch (e) {
      setLoadErr(e instanceof Error ? e.message : String(e));
      throw e;
    }
  }, [setSettings]);

  // Build the options list: local cache + always include the saved model_name
  // so it shows as selected even if the cache scan missed it.
  const savedModel = settings.omlx?.model_name?.trim() ?? "";
  const options = (() => {
    const base = localModels.length > 0
      ? localModels
      : savedModel
        ? []
        : [{ value: "", label: reachable ? "(no local models cached)" : "(server offline)" }];
    // Ensure the saved model is always present as an option
    if (savedModel && !base.some((o) => o.value === savedModel)) {
      const shortName = savedModel.split("/")[1] ?? savedModel;
      return [
        { value: savedModel, label: `${shortName}${reachable ? "" : " (saved)"}` },
        ...base,
      ];
    }
    return base;
  })();

  // Server is running but nothing is loaded yet — show the full picker
  if (reachable && !savedModel) {
    return (
      <div className="space-y-2">
        <p className="text-[11px] font-medium text-th-text-primary">Load a model</p>
        <p className="text-[10px] text-th-text-muted leading-relaxed">
          The oMLX server is running but no model has been set. Pick one below.
        </p>
        <OmlxModelPicker onLoad={handleLoad} />
        {loadErr && (
          <p className="text-[10px] text-red-400 leading-relaxed">{loadErr}</p>
        )}
        <p className="text-[11px] text-th-text-muted leading-relaxed">
          Manage the server and KV cache under the{" "}
          <button type="button" onClick={onGoToOnDevice} className="underline hover:text-th-text-primary">Turbo</button>{" "}sub-tab.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium text-th-text-tertiary">oMLX model</label>
      <OmlxModelDropdown
        options={options}
        value={savedModel}
        reachable={reachable}
        onChange={(id) => {
          setSettings((s) => ({ ...s, omlx: { ...s.omlx, enabled: true, model_name: id } }));
          setLoadErr(null);
          if (reachable && id) {
            void loadOmlxModelAndWait(id).catch((e) =>
              setLoadErr(e instanceof Error ? e.message : String(e)),
            );
          }
        }}
      />
      {loadErr && (
        <p className="text-[10px] text-red-400 leading-relaxed">{loadErr}</p>
      )}
      <p className="text-[11px] text-th-text-muted leading-relaxed">
        Manage the server, context window, and KV cache under the{" "}
        <button type="button" onClick={onGoToOnDevice} className="underline hover:text-th-text-primary">Turbo</button>{" "}sub-tab.
      </p>
    </div>
  );
}

function OmlxModelDropdown({
  options,
  value,
  reachable,
  onChange,
}: {
  options: { value: string; label: string }[];
  value: string;
  reachable: boolean;
  onChange: (id: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handler(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const selected = options.find((o) => o.value === value);
  const displayLabel = selected
    ? (selected.value.split("/")[1] ?? selected.value)
    : value
      ? (value.split("/")[1] ?? value)
      : "(no model set)";

  return (
    <div ref={ref} className="relative">
      {/* Trigger */}
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between gap-2 px-4 py-2.5 bg-th-input-bg border border-th-input-border rounded-lg text-sm text-left focus:outline-none hover:border-blue-400/40 transition-colors"
      >
        <span className="flex items-center gap-2 min-w-0">
          <span className="truncate text-th-text-primary">{displayLabel}</span>
          {value && !reachable && (
            <span className="shrink-0 px-1.5 py-0.5 rounded-full text-[10px] font-medium bg-th-surface-hover border border-th-border text-th-text-muted">
              Offline — saved
            </span>
          )}
          {value && reachable && (
            <span className="shrink-0 px-1.5 py-0.5 rounded-full text-[10px] font-medium bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
              Active
            </span>
          )}
        </span>
        <ChevronDown size={14} className={`shrink-0 text-th-text-muted transition-transform ${open ? "rotate-180" : ""}`} />
      </button>

      {/* Dropdown */}
      {open && options.length > 0 && (
        <div className="absolute z-20 mt-1 w-full rounded-xl border border-th-border bg-th-surface shadow-lg overflow-y-auto max-h-56">
          {options.map((opt, i) => {
            const shortLabel = opt.value ? (opt.value.split("/")[1] ?? opt.value) : opt.label;
            const sizeMatch = opt.label.match(/\(([0-9.]+\s*GB)\)/);
            const size = sizeMatch?.[1];
            const isSelected = opt.value === value;
            return (
              <button
                key={opt.value || i}
                type="button"
                onClick={() => { onChange(opt.value); setOpen(false); }}
                className={`w-full flex items-center justify-between gap-2 px-3 py-2.5 text-sm text-left transition-colors
                  ${isSelected ? "bg-blue-500/10 text-blue-400" : "text-th-text-primary hover:bg-th-surface-hover"}
                  ${i > 0 ? "border-t border-th-border/50" : ""}
                `}
              >
                <span className="flex items-center gap-2 min-w-0">
                  {isSelected
                    ? <CheckCircle2 size={12} className="shrink-0 text-blue-400" />
                    : <span className="w-3 shrink-0" />}
                  <span className="truncate font-medium">{shortLabel}</span>
                </span>
                {size && (
                  <span className={`shrink-0 px-1.5 py-0.5 rounded-full text-[10px] font-medium border
                    ${isSelected ? "bg-blue-500/10 border-blue-500/30 text-blue-400" : "bg-th-surface-hover border-th-border text-th-text-muted"}`}>
                    {size}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

function OmlxQuickPanel({
  settings,
  setSettings,
  turboSubTab,
}: {
  settings: AppSettings;
  setSettings: React.Dispatch<React.SetStateAction<AppSettings>>;
  turboSubTab: "Overview" | "Model";
}) {
  const [info, setInfo] = useState<import("../types").OmlxInfo | null>(null);
  const [status, setStatus] = useState<import("../types").OmlxStatus | null>(null);
  const [busy, setBusy] = useState<"install" | "upgrade" | "start" | "stop" | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [detecting, setDetecting] = useState(true);
  const [localModels, setLocalModels] = useState<{ value: string; label: string }[]>([]);
  const [versionInfo, setVersionInfo] = useState<import("../types").OmlxVersionInfo | null>(null);
  // Cache / turbo-mode settings
  const [cache, setCache] = useState<import("../types").OmlxCacheSettings | null>(null);
  const [cacheSaving, setCacheSaving] = useState(false);
  const [cacheErr, setCacheErr] = useState<string | null>(null);

  // Live cache performance stats
  const [cacheStats, setCacheStats] = useState<import("../types").OmlxCacheStats | null>(null);
  const [statsLoading, setStatsLoading] = useState(false);
  const [clearingCache, setClearingCache] = useState<"hot" | "ssd" | null>(null);

  const refreshVersion = useCallback(async () => {
    try {
      const v = await api.omlxVersionInfo();
      setVersionInfo(v);
    } catch { /* ignore when offline */ }
  }, []);

  const refresh = useCallback(async () => {
    setDetecting(true);
    try {
      const [i, s] = await Promise.all([api.omlxInfo(), api.omlxStatus()]);
      setInfo(i); setStatus(s);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
    try {
      const { models } = await api.omlxLocalModels();
      const mlx = models.filter((m) => m.is_mlx && m.repo_id.includes("/"));
      setLocalModels(mlx.map((m) => ({
        value: m.repo_id,
        label: `${m.repo_id.split("/")[1] ?? m.repo_id} (${m.size_gb.toFixed(1)} GB)`,
      })));
    } catch { /* ignore */ }
    setDetecting(false);
  }, []);

  const refreshCache = useCallback(async () => {
    try {
      const c = await api.omlxGetCache();
      setCache(c);
    } catch {
      // server may not be running yet — silently ignore
    }
  }, []);

  const refreshModelConfig = useCallback(async (repoId?: string) => {
    try {
      const mc = await api.omlxGetModelConfig(repoId);
      if (mc.max_context_window != null) {
        setCache((prev) => prev ? { ...prev, max_context_window: mc.max_context_window! } : prev);
        try {
          await api.omlxSetCache({ max_context_window: mc.max_context_window });
        } catch { /* server may not be running yet */ }
      }
    } catch {
      // model config not found — silently ignore
    }
  }, []);

  const refreshStats = useCallback(async () => {
    setStatsLoading(true);
    try {
      const s = await api.omlxGetCacheStats();
      setCacheStats(s);
    } catch {
      // silently ignore when server is offline
    } finally {
      setStatsLoading(false);
    }
  }, []);

  const clearCache = useCallback(async (kind: "hot" | "ssd") => {
    setClearingCache(kind);
    try {
      if (kind === "hot") await api.omlxClearHotCache();
      else await api.omlxClearSsdCache();
      await refreshStats();
    } catch (e) {
      setCacheErr(e instanceof Error ? e.message : String(e));
    } finally {
      setClearingCache(null);
    }
  }, [refreshStats]);

  const saveCache = useCallback(async (patch: Partial<import("../types").OmlxCacheSettings>) => {
    setCacheSaving(true); setCacheErr(null);
    try {
      const updated = await api.omlxSetCache(patch);
      setCache(updated);
    } catch (e) {
      setCacheErr(e instanceof Error ? e.message : String(e));
    } finally {
      setCacheSaving(false);
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);
  useEffect(() => { void refreshCache(); }, [refreshCache]);
  useEffect(() => { void refreshStats(); }, [refreshStats]);
  // Version check runs once on mount; the GitHub API is cached server-side.
  useEffect(() => { void refreshVersion(); }, [refreshVersion]);

  useEffect(() => {
    const t = setInterval(() => void refresh(), 5000);
    return () => clearInterval(t);
  }, [refresh]);

  const installed = !!info?.detection.installed && !!info.detection.cli_path;
  const reachable = !!status?.reachable;
  const homebrew = !!info?.detection.homebrew;

  const run = async (kind: "install" | "start" | "stop") => {
    setBusy(kind); setErr(null);
    try {
      if (kind === "install") {
        const { job_id } = await api.omlxInstall();
        // Install runs as a background job — keep the busy/in-progress state
        // until the job actually settles, not just until it's been kicked off.
        for (let i = 0; i < 240; i++) {
          await new Promise((r) => setTimeout(r, 3000));
          const job = await api.getOmlxJob(job_id);
          if (job.status === "done" || job.status === "error") {
            if (job.status === "error") throw new Error(job.error ?? "Install failed");
            break;
          }
        }
        await refreshVersion();
      } else if (kind === "start") await api.omlxStart();
      else await api.omlxStop();
      await refresh();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const runUpgrade = async () => {
    setBusy("upgrade"); setErr(null);
    try {
      const { job_id } = await api.omlxUpgrade();
      // Poll until the job settles, then refresh version info.
      for (let i = 0; i < 120; i++) {
        await new Promise((r) => setTimeout(r, 3000));
        const job = await api.getOmlxJob(job_id);
        if (job.status === "done" || job.status === "error") {
          if (job.status === "error") throw new Error(job.error ?? "Upgrade failed");
          break;
        }
      }
      await Promise.all([refresh(), refreshVersion()]);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };


  const handleLoadModel = useCallback(async (modelId: string) => {
    setErr(null);
    try {
      await loadOmlxModelAndWait(modelId);
      setSettings((s) => ({ ...s, omlx: { ...s.omlx, enabled: true, model_name: modelId } }));
      void refreshModelConfig(modelId);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setErr(msg);
      throw e;
    } finally {
      await refresh();
    }
  }, [refresh, setSettings, refreshModelConfig]);

  // When the server is offline, "Select" just saves the model name as the default
  // (it will be loaded when the server next starts). No HTTP call is made.
  const handleSelectModelOffline = useCallback(async (modelId: string) => {
    setSettings((s) => ({ ...s, omlx: { ...s.omlx, enabled: true, model_name: modelId } }));
    void refreshModelConfig(modelId);
  }, [setSettings, refreshModelConfig]);


  return (
    <div className="space-y-3 rounded-lg border border-th-border bg-th-inset-bg p-3">
      {turboSubTab === "Overview" && (<>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <p className="text-[12px] font-medium text-th-text-primary">oMLX server</p>
          {detecting && <Loader2 size={11} className="animate-spin text-th-text-tertiary" />}
        </div>
        <div className="flex items-center gap-1.5 text-[10px]">
          {!detecting && (
            <>
              <span className={`inline-block w-1.5 h-1.5 rounded-full ${reachable ? "bg-emerald-500" : installed ? "bg-amber-500" : "bg-th-text-muted"}`} />
              <span className="text-th-text-tertiary">
                {reachable ? "running" : installed ? "installed, stopped" : "not installed"}
              </span>
            </>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[10px] text-th-text-tertiary">
        <div className="truncate">CLI: <span className="text-th-text-secondary">{info?.detection.cli_path ?? "—"}</span></div>
        <div className="truncate">Brew: <span className="text-th-text-secondary">{homebrew ? "yes" : "no"}</span></div>
        <div className="truncate">Service: <span className="text-th-text-secondary">{info?.detection.brew_service_state ?? "—"}</span></div>
        <div className="truncate">Port: <span className="text-th-text-secondary">{settings.omlx?.api_port ?? 8000}</span></div>
        <div className="truncate col-span-2">
          Version:{" "}
          <span className="text-th-text-secondary">
            {versionInfo?.installed_version ?? info?.detection.cli_version ?? "—"}
          </span>
          {versionInfo?.latest_version && (
            <span className="ml-1 text-th-text-muted">
              (latest: {versionInfo.latest_version})
            </span>
          )}
        </div>
      </div>

      {/* Upgrade banner — shown only when an update is available via Homebrew */}
      {versionInfo?.upgrade_available && versionInfo.homebrew && (
        <div className="flex items-center justify-between rounded-md border border-blue-500/30 bg-blue-500/10 px-2.5 py-1.5 text-[10px]">
          <span className="text-blue-300">
            Update available: {versionInfo.installed_version} → {versionInfo.latest_version}
          </span>
          <button
            onClick={() => void runUpgrade()}
            disabled={busy === "upgrade"}
            className="ml-3 flex items-center gap-1 rounded px-2 py-0.5 text-[10px] font-medium bg-blue-500/20 text-blue-300 border border-blue-500/40 hover:bg-blue-500/30 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {busy === "upgrade" ? (
              <><Loader2 size={10} className="animate-spin" /> Upgrading…</>
            ) : (
              "Upgrade"
            )}
          </button>
        </div>
      )}

      <div className="space-y-1">
        <label className="block text-sm font-medium text-th-text-tertiary">Active model</label>
        <OmlxModelDropdown
          options={(() => {
            const savedModel = settings.omlx?.model_name?.trim() ?? "";
            if (!savedModel && localModels.length === 0)
              return [{ value: "", label: reachable ? "(no local models cached)" : "(server offline)" }];
            if (savedModel && !localModels.some((o) => o.value === savedModel)) {
              const shortName = savedModel.split("/")[1] ?? savedModel;
              return [{ value: savedModel, label: `${shortName}${reachable ? "" : " (saved)"}` }, ...localModels];
            }
            return localModels;
          })()}
          value={settings.omlx?.model_name?.trim() ?? ""}
          reachable={reachable}
          onChange={(id) => {
            setSettings((s) => ({ ...s, omlx: { ...s.omlx, enabled: true, model_name: id } }));
            void refreshModelConfig(id);
            if (reachable && id) void handleLoadModel(id);
            else void handleSelectModelOffline(id);
          }}
        />
      </div>
      </>)}

      {turboSubTab === "Model" && (
        <div className="space-y-3">
          <p className="text-[11px] text-th-text-secondary leading-relaxed">
            Browse, download, and load models from HuggingFace Hub. Downloads go to the shared
            HF cache used by both Standard and Turbo.
            {reachable
              ? " Picking a cached model loads it into the running oMLX server."
              : " Server offline — picking a model saves it to load when the server next starts."}
          </p>
          <ModelChooser
            selectedRepoId={settings.omlx?.model_name?.trim() || settings.llm.mlx.hf_llm_model_id}
            hfToken={settings.llm.mlx.hf_token}
            cacheDir={settings.llm.mlx.hf_hub_cache}
            onDownloadComplete={(repo) => {
              void refresh();
              if (reachable) void handleLoadModel(repo).catch(() => undefined);
              else void handleSelectModelOffline(repo);
            }}
            onUseCached={(repo) => {
              if (reachable) void handleLoadModel(repo).catch(() => undefined);
              else void handleSelectModelOffline(repo);
            }}
          />
        </div>
      )}

      {turboSubTab === "Overview" && (<>
      <div className="flex items-center gap-2">
        <span className="text-[11px] text-th-text-secondary w-28 shrink-0">Context window</span>
        <input
          type="number"
          min={4096}
          step={4096}
          value={cache?.max_context_window ?? 131072}
          disabled={cacheSaving}
          onChange={(e) => {
            const v = parseInt(e.target.value, 10) || 131072;
            if (cache) setCache({ ...cache, max_context_window: v });
          }}
          onBlur={(e) => {
            const v = parseInt(e.target.value, 10) || 131072;
            void saveCache({ max_context_window: v });
          }}
          className="w-28 bg-th-input-bg border border-th-border rounded px-2 py-0.5 text-[11px] text-th-text-primary focus:outline-none focus:ring-1 focus:ring-th-tab-active-bg disabled:opacity-40"
        />
        <span className="text-[10px] text-th-text-muted">tokens</span>
      </div>

      <div className="flex items-center gap-2">
        <span className="text-[11px] text-th-text-secondary w-28 shrink-0">Max output tokens</span>
        <input
          type="number"
          min={256}
          step={256}
          value={settings.omlx.max_tokens ?? 8192}
          onChange={(e) => {
            const v = parseInt(e.target.value, 10) || 8192;
            setSettings((s) => ({ ...s, omlx: { ...s.omlx, max_tokens: v } }));
          }}
          className="w-28 bg-th-input-bg border border-th-border rounded px-2 py-0.5 text-[11px] text-th-text-primary focus:outline-none focus:ring-1 focus:ring-th-tab-active-bg"
        />
        <span className="text-[10px] text-th-text-muted">tokens</span>
      </div>

      <label className="flex items-center justify-between gap-2 cursor-pointer select-none">
        <span className="text-[11px] text-th-text-secondary">
          Thinking mode
          <span className="text-[10px] text-th-text-muted ml-1">(chain-of-thought; Qwen3 etc.)</span>
        </span>
        <input
          type="checkbox"
          checked={settings.omlx.thinking_enabled ?? false}
          onChange={(e) =>
            setSettings((s) => ({ ...s, omlx: { ...s.omlx, thinking_enabled: e.target.checked } }))
          }
          className="w-3.5 h-3.5 accent-th-tab-active-bg cursor-pointer"
        />
      </label>

      <div className="flex flex-wrap gap-2">
        {!installed && (
          <button
            type="button"
            onClick={() => void run("install")}
            disabled={busy !== null}
            className="px-3 py-1.5 rounded-md bg-th-tab-active-bg text-white text-[11px] font-medium disabled:opacity-50 inline-flex items-center gap-1.5"
            title={homebrew ? "Run brew tap + brew install" : "Download + install from the official oMLX release (no Homebrew needed)"}
          >
            {busy === "install" ? <Loader2 size={11} className="animate-spin" /> : <Play size={11} />}
            Install
          </button>
        )}
        {installed && !reachable && (
          <button
            type="button"
            onClick={() => void run("start")}
            disabled={busy !== null}
            className="px-3 py-1.5 rounded-md bg-th-tab-active-bg text-white text-[11px] font-medium disabled:opacity-50 inline-flex items-center gap-1.5"
          >
            {busy === "start" ? <Loader2 size={11} className="animate-spin" /> : <Play size={11} />}
            Start
          </button>
        )}
        {reachable && (
          <button
            type="button"
            onClick={() => void run("stop")}
            disabled={busy !== null}
            className="px-3 py-1.5 rounded-md border border-th-border text-th-text-secondary text-[11px] font-medium hover:bg-th-surface-hover/30 inline-flex items-center gap-1.5"
          >
            {busy === "stop" ? <Loader2 size={11} className="animate-spin" /> : <Square size={11} />}
            Stop
          </button>
        )}
        <button
          type="button"
          onClick={() => void refresh()}
          className="px-3 py-1.5 rounded-md border border-th-border text-th-text-secondary text-[11px] font-medium hover:bg-th-surface-hover/30 inline-flex items-center gap-1.5"
        >
          <RefreshCw size={11} />
          Refresh
        </button>
      </div>

      {err && (
        <p className="text-[11px] text-rose-400 leading-relaxed">{err}</p>
      )}

      {/* KV cache / turbo mode */}
      <div className="border-t border-th-border pt-2 space-y-2">
        <div className="flex items-center justify-between">
          <p className="text-[11px] font-medium text-th-text-secondary">KV Cache (turbo mode)</p>
          {cacheSaving && <Loader2 size={10} className="animate-spin text-th-text-tertiary" />}
        </div>

        {/* Enable KV cache toggle */}
        <label className="flex items-center justify-between gap-2 cursor-pointer select-none">
          <span className="text-[11px] text-th-text-secondary">Enable KV cache</span>
          <input
            type="checkbox"
            checked={cache?.cache_enabled ?? true}
            disabled={cacheSaving}
            onChange={(e) => void saveCache({ cache_enabled: e.target.checked })}
            className="w-3.5 h-3.5 accent-th-tab-active-bg cursor-pointer"
          />
        </label>

        {/* SSD cold tier toggle */}
        <label className="flex items-center justify-between gap-2 cursor-pointer select-none">
          <span className="text-[11px] text-th-text-secondary">SSD cold tier</span>
          <input
            type="checkbox"
            checked={cache ? !cache.hot_cache_only : false}
            disabled={cacheSaving || !(cache?.cache_enabled ?? true)}
            onChange={(e) => void saveCache({ hot_cache_only: !e.target.checked })}
            className="w-3.5 h-3.5 accent-th-tab-active-bg cursor-pointer disabled:opacity-40"
          />
        </label>

        {/* SSD dir — shown only when SSD tier is enabled */}
        {cache && !cache.hot_cache_only && cache.cache_enabled && (
          <>
            <div className="flex items-center gap-2">
              <span className="text-[11px] text-th-text-secondary w-24 shrink-0">SSD directory</span>
              <input
                type="text"
                value={cache.ssd_cache_dir}
                placeholder="e.g. /tmp/omlx-cache"
                disabled={cacheSaving}
                onChange={(e) => setCache({ ...cache, ssd_cache_dir: e.target.value })}
                onBlur={(e) => void saveCache({ ssd_cache_dir: e.target.value })}
                className="flex-1 min-w-0 bg-th-input-bg border border-th-border rounded px-2 py-0.5 text-[11px] text-th-text-primary placeholder-th-text-muted focus:outline-none focus:ring-1 focus:ring-th-tab-active-bg"
              />
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[11px] text-th-text-secondary w-24 shrink-0">SSD max size</span>
              <input
                type="text"
                value={cache.ssd_cache_max_size}
                placeholder="e.g. 20GB or auto"
                disabled={cacheSaving}
                onChange={(e) => setCache({ ...cache, ssd_cache_max_size: e.target.value })}
                onBlur={(e) => void saveCache({ ssd_cache_max_size: e.target.value })}
                className="flex-1 min-w-0 bg-th-input-bg border border-th-border rounded px-2 py-0.5 text-[11px] text-th-text-primary placeholder-th-text-muted focus:outline-none focus:ring-1 focus:ring-th-tab-active-bg"
              />
            </div>
          </>
        )}

        {/* Hot cache size + initial cache blocks — only meaningful when cache is on */}
        {cache && cache.cache_enabled && (
          <>
            <div className="flex items-center gap-2">
              <span className="text-[11px] text-th-text-secondary w-24 shrink-0">Hot cache size</span>
              <input
                type="text"
                value={cache.hot_cache_max_size}
                placeholder="0 = unlimited, or e.g. 8GB"
                disabled={cacheSaving}
                onChange={(e) => setCache({ ...cache, hot_cache_max_size: e.target.value })}
                onBlur={(e) => void saveCache({ hot_cache_max_size: e.target.value })}
                className="flex-1 min-w-0 bg-th-input-bg border border-th-border rounded px-2 py-0.5 text-[11px] text-th-text-primary placeholder-th-text-muted focus:outline-none focus:ring-1 focus:ring-th-tab-active-bg"
              />
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[11px] text-th-text-secondary w-24 shrink-0">Initial blocks</span>
              <input
                type="number"
                min={1}
                step={1}
                value={cache.initial_cache_blocks}
                disabled={cacheSaving}
                onChange={(e) => setCache({ ...cache, initial_cache_blocks: parseInt(e.target.value, 10) || 0 })}
                onBlur={(e) => {
                  const v = parseInt(e.target.value, 10);
                  if (Number.isFinite(v) && v > 0) void saveCache({ initial_cache_blocks: v });
                }}
                className="w-24 bg-th-input-bg border border-th-border rounded px-2 py-0.5 text-[11px] text-th-text-primary focus:outline-none focus:ring-1 focus:ring-th-tab-active-bg disabled:opacity-40"
              />
              <span className="text-[10px] text-th-text-muted">KV blocks pre-allocated</span>
            </div>
          </>
        )}

        {/* Batching — continuous-batching concurrency cap */}
        <div className="flex items-center gap-2">
          <span className="text-[11px] text-th-text-secondary w-24 shrink-0">Batch size</span>
          <input
            type="number"
            min={1}
            step={1}
            value={cache?.max_concurrent_requests ?? 8}
            disabled={cacheSaving || !cache}
            onChange={(e) => {
              const v = parseInt(e.target.value, 10) || 1;
              if (cache) setCache({ ...cache, max_concurrent_requests: v });
            }}
            onBlur={(e) => {
              const v = parseInt(e.target.value, 10);
              if (Number.isFinite(v) && v > 0) void saveCache({ max_concurrent_requests: v });
            }}
            className="w-24 bg-th-input-bg border border-th-border rounded px-2 py-0.5 text-[11px] text-th-text-primary focus:outline-none focus:ring-1 focus:ring-th-tab-active-bg disabled:opacity-40"
          />
          <span className="text-[10px] text-th-text-muted">max concurrent requests</span>
        </div>

        {cacheErr && (
          <p className="text-[10px] text-rose-400 leading-relaxed">{cacheErr}</p>
        )}
      </div>

      {/* Cache performance & disk usage */}
      <div className="border-t border-th-border pt-2 space-y-2">
        <div className="flex items-center justify-between">
          <p className="text-[11px] font-medium text-th-text-secondary">Cache performance</p>
          <button
            type="button"
            onClick={() => void refreshStats()}
            disabled={statsLoading}
            className="inline-flex items-center gap-1 text-[10px] text-th-text-muted hover:text-th-text-secondary disabled:opacity-40"
          >
            {statsLoading ? <Loader2 size={10} className="animate-spin" /> : <RefreshCw size={10} />}
            Refresh
          </button>
        </div>

        {cacheStats?.reachable ? (
          <>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[10px]">
              <div className="text-th-text-muted">Efficiency</div>
              <div className="text-th-text-secondary font-medium">{cacheStats.cache_efficiency_pct}%</div>

              <div className="text-th-text-muted">Tokens served</div>
              <div className="text-th-text-secondary">{cacheStats.total_tokens_served.toLocaleString()}</div>

              <div className="text-th-text-muted">Tokens from cache</div>
              <div className="text-th-text-secondary">{cacheStats.total_cached_tokens.toLocaleString()}</div>

              <div className="text-th-text-muted">Requests</div>
              <div className="text-th-text-secondary">{cacheStats.total_requests}</div>

              <div className="text-th-text-muted">Prefill speed</div>
              <div className="text-th-text-secondary">{cacheStats.avg_prefill_tps.toLocaleString()} tok/s</div>

              <div className="text-th-text-muted">Generation speed</div>
              <div className="text-th-text-secondary">{cacheStats.avg_generation_tps} tok/s</div>

              <div className="text-th-text-muted">Disk usage</div>
              <div className="text-th-text-secondary">{cacheStats.disk_gb} GB
                <span className="text-th-text-muted ml-1 font-mono text-[9px] break-all">{cacheStats.cache_dir}</span>
              </div>
            </div>

            <div className="flex gap-2 pt-1">
              <button
                type="button"
                onClick={() => void clearCache("hot")}
                disabled={clearingCache !== null}
                className="px-2.5 py-1 rounded-md border border-th-border text-th-text-secondary text-[10px] font-medium hover:bg-th-surface-hover/30 inline-flex items-center gap-1 disabled:opacity-40"
              >
                {clearingCache === "hot" ? <Loader2 size={10} className="animate-spin" /> : <Trash2 size={10} />}
                Clear hot cache
              </button>
              <button
                type="button"
                onClick={() => void clearCache("ssd")}
                disabled={clearingCache !== null}
                className="px-2.5 py-1 rounded-md border border-th-border text-th-text-secondary text-[10px] font-medium hover:bg-th-surface-hover/30 inline-flex items-center gap-1 disabled:opacity-40"
              >
                {clearingCache === "ssd" ? <Loader2 size={10} className="animate-spin" /> : <Trash2 size={10} />}
                Clear SSD cache
              </button>
            </div>
          </>
        ) : (
          <p className="text-[10px] text-th-text-muted">
            {statsLoading
              ? "Loading…"
              : reachable
                ? "Server is online but admin stats are unavailable — Otto needs an admin API key. Re-run the setup wizard to provision one."
                : "Server offline — start oMLX to see cache stats"}
          </p>
        )}
      </div>

      <p className="text-[10px] text-th-text-muted leading-relaxed">
        For first-time install or a richer setup flow with live install logs,
        use Settings → Reset setup wizard, or run{" "}
        <span className="font-mono">brew tap jundot/omlx https://github.com/jundot/omlx && brew install omlx</span> manually.
      </p>
      </>)}
    </div>
  );
}

function Card({ title, dot, children }: { title?: string; dot?: string; children: React.ReactNode }) {
  return (
    <div className="bg-th-card-bg border border-th-card-border rounded-xl p-6 shadow-sm h-full">
      {title && <h2 className="text-base font-semibold text-th-text-primary mb-5 flex items-center gap-2.5">{dot && <span className={`w-2 h-2 rounded-full ${dot}`} />}{title}</h2>}
      {children}
    </div>
  );
}

function InputField({ label, value, onChange, placeholder, type = "text", min, max }: { label: string; value: string; onChange: (v: string) => void; placeholder?: string; type?: string; min?: number; max?: number }) {
  return (
    <div>
      <label className="block text-sm font-medium text-th-text-tertiary mb-2">{label}</label>
      <input className="w-full px-4 py-2.5 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-300/30 transition-all text-sm" type={type} value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder} min={min} max={max} />
    </div>
  );
}

/**
 * Like ``InputField`` but commits the typed value only on blur or
 * Enter — never on every keystroke. Used for settings whose ``onChange``
 * has expensive side-effects (here: re-placing the loaded model on the
 * EXO cluster) where firing on every digit typed would issue a stack of
 * doomed-then-resolved place_instance calls.
 *
 * The displayed value tracks ``value`` (so external updates — e.g. the
 * auto-tune effect bumping ``min_nodes`` when a peer joins — stay
 * visible) except while the user is actively editing the field, at
 * which point local state owns the display until they commit or revert.
 */
function CommitInputField({
  label, value, onChange, placeholder, type = "text", min,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  type?: string;
  min?: number;
}) {
  const [draft, setDraft] = useState(value);
  const [editing, setEditing] = useState(false);
  useEffect(() => {
    if (!editing) setDraft(value);
  }, [value, editing]);

  const commit = () => {
    setEditing(false);
    if (draft !== value) onChange(draft);
  };
  const revert = () => {
    setEditing(false);
    setDraft(value);
  };

  return (
    <div>
      <label className="block text-sm font-medium text-th-text-tertiary mb-2">{label}</label>
      <input
        className="w-full px-4 py-2.5 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-300/30 transition-all text-sm"
        type={type}
        min={min}
        value={draft}
        placeholder={placeholder}
        onFocus={() => setEditing(true)}
        onChange={(e) => { setEditing(true); setDraft(e.target.value); }}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            (e.target as HTMLInputElement).blur();
          } else if (e.key === "Escape") {
            e.preventDefault();
            revert();
            (e.target as HTMLInputElement).blur();
          }
        }}
      />
    </div>
  );
}

function ExoModelSelectField({ label, value, onChange, options }: { label: string; value: string; onChange: (v: string) => void; options: { value: string; label: string; loaded?: boolean }[] }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const selected = options.find((o) => o.value === value);

  return (
    <div>
      <label className="block text-sm font-medium text-th-text-tertiary mb-2">{label}</label>
      <div className="relative" ref={ref}>
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="w-full flex items-center justify-between px-4 py-2.5 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-300/30 transition-all text-sm"
        >
          <span className="flex items-center gap-2 min-w-0">
            <span className="truncate">{selected?.label ?? (value || "(pick a model)")}</span>
            {selected?.loaded && (
              <span className="inline-flex items-center px-1.5 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500/15 text-emerald-400 border border-emerald-500/25 shrink-0">
                loaded
              </span>
            )}
          </span>
          <ChevronDown size={14} className="text-th-text-muted shrink-0 ml-2" />
        </button>
        {open && (
          <div className="absolute z-50 w-full mt-1 bg-th-input-bg border border-th-input-border rounded-lg shadow-lg overflow-auto max-h-60">
            {options.map((o) => (
              <button
                key={o.value === "" ? "__empty" : o.value}
                type="button"
                onClick={() => { onChange(o.value); setOpen(false); }}
                className={`w-full flex items-center justify-between px-4 py-2 text-sm text-left hover:bg-th-surface-hover transition-colors ${o.value === value ? "text-th-text-primary bg-th-surface-hover/50" : "text-th-text-secondary"}`}
              >
                <span className="truncate">{o.label}</span>
                {o.loaded && (
                  <span className="inline-flex items-center px-1.5 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500/15 text-emerald-400 border border-emerald-500/25 shrink-0 ml-2">
                    loaded
                  </span>
                )}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function SelectField({ label, value, onChange, options }: { label: string; value: string; onChange: (v: string) => void; options: { value: string; label: string }[] }) {
  return (
    <div>
      <label className="block text-sm font-medium text-th-text-tertiary mb-2">{label}</label>
      <div className="relative">
        <select className="w-full appearance-none px-4 py-2.5 pr-10 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-300/30 transition-all text-sm" value={value} onChange={(e) => onChange(e.target.value)}>
          {options.map((o) => <option key={o.value === "" ? "__empty" : o.value} value={o.value}>{o.label}</option>)}
        </select>
        <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-th-text-muted pointer-events-none" />
      </div>
    </div>
  );
}

function SecretField({ label, value, onChange, placeholder }: { label: string; value: string; onChange: (v: string) => void; placeholder?: string }) {
  const [visible, setVisible] = useState(false);
  return (
    <div>
      <label className="block text-sm font-medium text-th-text-tertiary mb-2">{label}</label>
      <div className="relative">
        <input className="w-full px-4 py-2.5 pr-10 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-300/30 transition-all text-sm" type={visible ? "text" : "password"} value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder} />
        <button type="button" className="absolute right-3 top-1/2 -translate-y-1/2 text-th-text-muted hover:text-th-text-secondary transition-colors" onClick={() => setVisible(!visible)}>{visible ? <EyeOff size={16} /> : <Eye size={16} />}</button>
      </div>
    </div>
  );
}

function Toggle({ label, description, checked, onChange }: { label: string; description?: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex items-start gap-3 cursor-pointer">
      <button type="button" role="switch" aria-checked={checked} onClick={() => onChange(!checked)}
        className={`relative w-10 h-[22px] rounded-full transition-all duration-200 border shrink-0 mt-0.5 ${checked ? "bg-blue-600 border-blue-600" : "bg-th-inset-bg border-th-border"}`}>
        <span className={`absolute top-0.5 left-0.5 w-[16px] h-[16px] rounded-full transition-all duration-200 ${checked ? "translate-x-[18px] bg-white" : "bg-neutral-400"}`} />
      </button>
      <div className="min-w-0">
        <span className="text-sm text-th-text-secondary">{label}</span>
        {description && <p className="text-[11px] text-th-text-muted mt-0.5 leading-relaxed">{description}</p>}
      </div>
    </label>
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
  const [open, setOpen] = useState(false);
  const [manualMode, setManualMode] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const hasModels = models.length > 0;
  const currentInList = models.some((m) => m.id === value);
  const selectedModel = models.find((m) => m.id === value);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <label className="block text-sm font-medium text-th-text-tertiary">Model</label>
        <div className="flex items-center gap-2">
          {hasModels && (
            <button
              type="button"
              className="text-xs text-th-text-muted hover:text-th-text-secondary transition-colors"
              onClick={() => { setManualMode(!manualMode); setOpen(false); }}
            >
              {manualMode ? "Pick from list" : "Enter manually"}
            </button>
          )}
          <button
            type="button"
            className="text-th-text-muted hover:text-th-text-secondary transition-colors flex items-center gap-1 text-xs"
            onClick={onFetch}
            disabled={loading}
          >
            <RefreshCw size={12} className={loading ? "animate-spin" : ""} />
            {loading ? "Fetching..." : "Fetch models"}
          </button>
        </div>
      </div>

      {hasModels && !manualMode ? (
        <div className="relative" ref={dropdownRef}>
          {/* Trigger button */}
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            className={`w-full flex items-center justify-between px-3 py-2.5 bg-th-input-bg border rounded-lg focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-300/30 transition-all text-sm text-left ${value ? "border-th-input-border" : "border-amber-500/30"}`}
          >
            <span className="flex items-center gap-2 min-w-0">
              {value ? (
                <>
                  <span className="text-th-text-primary truncate">
                    {selectedModel ? selectedModel.name : value}
                  </span>
                  <span className="shrink-0 inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono font-medium bg-th-inset-bg border border-th-border text-th-text-muted">
                    {value}
                  </span>
                </>
              ) : (
                <span className="text-th-text-muted">Select a model…</span>
              )}
            </span>
            <ChevronDown size={14} className={`shrink-0 ml-2 text-th-text-muted transition-transform ${open ? "rotate-180" : ""}`} />
          </button>

          {/* Dropdown list */}
          {open && (
            <div className="absolute z-50 mt-1 w-full bg-th-bg-secondary border border-th-border rounded-lg shadow-lg overflow-hidden">
              <div className="max-h-56 overflow-y-auto py-1">
                {value && !currentInList && (
                  <div className="flex items-center justify-between gap-2 px-3 py-2 bg-amber-500/5 border-b border-th-border">
                    <span className="text-xs text-th-text-secondary truncate">{value}</span>
                    <span className="shrink-0 inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono bg-amber-500/10 border border-amber-500/30 text-amber-300">custom</span>
                  </div>
                )}
                {models.map((m) => {
                  const active = m.id === value;
                  return (
                    <button
                      key={m.id}
                      type="button"
                      onClick={() => { onChange(m.id); setOpen(false); }}
                      className={`w-full flex items-center justify-between gap-2 px-3 py-2 text-sm text-left transition-colors hover:bg-th-surface-hover ${active ? "bg-blue-500/10" : ""}`}
                    >
                      <span className={`truncate ${active ? "text-blue-300 font-medium" : "text-th-text-primary"}`}>
                        {m.name}
                      </span>
                      <span className={`shrink-0 inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono border ${active ? "bg-blue-500/20 border-blue-500/40 text-blue-300" : "bg-th-inset-bg border-th-border text-th-text-muted"}`}>
                        {m.id}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      ) : (
        <input
          className={`w-full px-4 py-2.5 bg-th-input-bg border rounded-lg placeholder-th-text-muted focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-300/30 transition-all text-sm ${value ? "border-th-input-border text-th-text-primary" : "border-amber-500/30 text-th-text-muted"}`}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Enter model ID (e.g. claude-sonnet-4-6)"
        />
      )}

      {!value && (
        <p className="mt-1.5 text-xs text-amber-600/80">
          No model selected — fetch the list or enter a model ID manually
        </p>
      )}
      {error && <p className="mt-1.5 text-xs text-red-500">{error}</p>}
      {value && !hasModels && !loading && !error && (
        <p className="mt-1.5 text-xs text-th-text-muted">
          Click &quot;Fetch models&quot; to load available models
        </p>
      )}
    </div>
  );
}

// ===========================================================================
// Privacy & Security tab
// ===========================================================================

function PrivacyTab() {
  const [status, setStatus] = useState<PrivacyStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [toggling, setToggling] = useState(false);
  const [audit, setAudit] = useState<PrivacyAuditEntry[]>([]);
  const [auditLoading, setAuditLoading] = useState(false);
  const [pfTemplate, setPfTemplate] = useState<{ pf_template: string; install_command: string } | null>(null);
  const [pfLoading, setPfLoading] = useState(false);
  const [copied, setCopied] = useState<"cmd" | "template" | null>(null);
  const [newHost, setNewHost] = useState("");
  const [savingHosts, setSavingHosts] = useState(false);
  const [showAudit, setShowAudit] = useState(false);
  const [showPf, setShowPf] = useState(false);
  const [hideFromShare, setHideFromShare] = useState(false);
  const [hideBusy, setHideBusy] = useState(false);

  useEffect(() => {
    import("../utils/screenShareVisibility")
      .then(({ getHideFromScreenShare }) => setHideFromShare(getHideFromScreenShare()))
      .catch(() => undefined);
  }, []);

  const toggleHideFromShare = async () => {
    const next = !hideFromShare;
    setHideBusy(true);
    try {
      const { setHideFromScreenShare } = await import("../utils/screenShareVisibility");
      await setHideFromScreenShare(next);
      setHideFromShare(next);
    } finally {
      setHideBusy(false);
    }
  };

  const refresh = () => {
    setLoading(true);
    api.privacyStatus()
      .then(setStatus)
      .catch(() => undefined)
      .finally(() => setLoading(false));
  };

  useEffect(() => { refresh(); }, []);

  const handleToggle = async () => {
    if (!status) return;
    setToggling(true);
    try {
      const fn = status.engaged ? api.privacyDisengage : api.privacyEngage;
      const next = await fn();
      setStatus(next);
    } finally {
      setToggling(false);
    }
  };

  const loadAudit = async () => {
    setAuditLoading(true);
    try {
      const r = await api.privacyAudit(100);
      setAudit(r.events);
      setShowAudit(true);
    } finally {
      setAuditLoading(false);
    }
  };

  const loadPf = async () => {
    setPfLoading(true);
    try {
      const r = await api.privacyPfTemplate();
      setPfTemplate(r);
      setShowPf(true);
    } finally {
      setPfLoading(false);
    }
  };

  const copyText = (text: string, key: "cmd" | "template") => {
    void navigator.clipboard.writeText(text).then(() => {
      setCopied(key);
      setTimeout(() => setCopied(null), 2000);
    });
  };

  const addHost = async () => {
    const h = newHost.trim();
    if (!h || !status) return;
    const hosts = [...(status.allowed_hosts ?? []), h];
    setSavingHosts(true);
    try {
      const next = await api.privacyUpdate({ allowed_hosts: hosts });
      setStatus(next);
      setNewHost("");
    } finally {
      setSavingHosts(false);
    }
  };

  const removeHost = async (host: string) => {
    if (!status) return;
    const hosts = (status.allowed_hosts ?? []).filter((h) => h !== host);
    setSavingHosts(true);
    try {
      const next = await api.privacyUpdate({ allowed_hosts: hosts });
      setStatus(next);
    } finally {
      setSavingHosts(false);
    }
  };

  const engaged = status?.engaged ?? false;
  const pfActive = status?.pf?.available && status.pf.has_block_rule;

  return (
    <div className="max-w-2xl space-y-4">

      {/* ── Main toggle card ─────────────────────────────────────────────── */}
      <Card title="Privacy Lock" dot={engaged ? "bg-emerald-500" : "bg-th-text-muted"}>
        <div className="space-y-4">
          {loading ? (
            <div className="flex items-center gap-2 text-sm text-th-text-muted">
              <Loader2 size={14} className="animate-spin" />
              Loading…
            </div>
          ) : (
            <>
              {/* Big status banner */}
              <div className={`flex items-center gap-3 rounded-xl border p-4 ${
                engaged
                  ? "border-emerald-500/30 bg-emerald-500/5"
                  : "border-th-border bg-th-inset-bg"
              }`}>
                <div className="shrink-0">
                  {engaged
                    ? <ShieldCheck size={28} className="text-emerald-400" />
                    : <ShieldOff size={28} className="text-th-text-muted" />
                  }
                </div>
                <div className="flex-1 min-w-0">
                  <p className={`text-sm font-semibold ${engaged ? "text-emerald-400" : "text-th-text-secondary"}`}>
                    {engaged ? "Privacy Lock engaged" : "Privacy Lock off"}
                  </p>
                  <p className="text-xs text-th-text-muted mt-0.5 leading-relaxed">
                    {engaged
                      ? `Cloud LLMs blocked since ${status?.engaged_at ? new Date(status.engaged_at).toLocaleString() : "—"}. Only local providers (${status?.local_only_providers?.join(", ") ?? "—"}) are allowed.`
                      : "All providers allowed. Enable to restrict Otto to on-device inference only."}
                  </p>
                  {engaged && status?.audit_token && (
                    <p className="text-[10px] font-mono text-th-text-muted mt-1 truncate">
                      Audit token: {status.audit_token}
                    </p>
                  )}
                </div>
                <button
                  type="button"
                  className={`shrink-0 inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all disabled:opacity-50 ${
                    engaged
                      ? "bg-rose-500/10 border border-rose-500/30 text-rose-400 hover:bg-rose-500/20"
                      : "bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/20"
                  }`}
                  onClick={() => void handleToggle()}
                  disabled={toggling}
                >
                  {toggling
                    ? <Loader2 size={14} className="animate-spin" />
                    : engaged ? <Unlock size={14} /> : <Lock size={14} />
                  }
                  {toggling ? "…" : engaged ? "Disengage" : "Engage"}
                </button>
              </div>

              {/* PF kernel layer status */}
              {status?.pf && (
                <div className={`flex items-center gap-2 text-xs rounded-lg border px-3 py-2 ${
                  pfActive
                    ? "border-emerald-500/20 bg-emerald-500/5 text-emerald-400"
                    : "border-th-border bg-th-inset-bg text-th-text-muted"
                }`}>
                  {pfActive
                    ? <ShieldCheck size={12} className="shrink-0" />
                    : <ShieldAlert size={12} className="shrink-0" />
                  }
                  <span>
                    {pfActive
                      ? `Kernel firewall active — anchor "${status.pf.anchor}", ${status.pf.rule_count} rules loaded.`
                      : `Kernel firewall not loaded${status.pf.reason ? ` (${status.pf.reason})` : ""}. App-layer guard is still active.`
                    }
                  </span>
                </div>
              )}
            </>
          )}
        </div>
      </Card>

      {/* ── Screen sharing visibility ────────────────────────────────────── */}
      <Card title="Screen sharing visibility" dot={hideFromShare ? "bg-emerald-500" : "bg-th-text-muted"}>
        <div className="space-y-4">
          <div className={`flex items-center gap-3 rounded-xl border p-4 ${
            hideFromShare
              ? "border-emerald-500/30 bg-emerald-500/5"
              : "border-th-border bg-th-inset-bg"
          }`}>
            <div className="shrink-0">
              {hideFromShare
                ? <EyeOff size={28} className="text-emerald-400" />
                : <Eye size={28} className="text-th-text-muted" />
              }
            </div>
            <div className="flex-1 min-w-0">
              <p className={`text-sm font-semibold ${hideFromShare ? "text-emerald-400" : "text-th-text-secondary"}`}>
                {hideFromShare ? "Hidden from screen share" : "Visible in screen share"}
              </p>
              <p className="text-xs text-th-text-muted mt-0.5 leading-relaxed">
                {hideFromShare
                  ? "Otto's window is excluded from screen capture, and its menu bar and Dock icons are hidden — while staying visible on your own display."
                  : "Otto appears normally when you share or record your screen, with its menu bar and Dock icons shown."}
              </p>
            </div>
            <button
              type="button"
              className={`shrink-0 inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all disabled:opacity-50 ${
                hideFromShare
                  ? "bg-rose-500/10 border border-rose-500/30 text-rose-400 hover:bg-rose-500/20"
                  : "bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/20"
              }`}
              onClick={() => void toggleHideFromShare()}
              disabled={hideBusy}
            >
              {hideBusy
                ? <Loader2 size={14} className="animate-spin" />
                : hideFromShare ? <Eye size={14} /> : <EyeOff size={14} />
              }
              {hideBusy ? "…" : hideFromShare ? "Show" : "Hide"}
            </button>
          </div>
          <div className="flex items-start gap-2 text-[11px] text-th-text-muted leading-relaxed rounded-lg border border-th-border bg-th-inset-bg px-3 py-2">
            <ShieldAlert size={12} className="shrink-0 mt-0.5" />
            <span>
              Works against CoreGraphics-based capture, including <strong className="text-th-text-secondary">Google Meet in Chrome</strong>. On macOS 15+ it does <strong className="text-th-text-secondary">not</strong> hide Otto's window from ScreenCaptureKit apps (Zoom, Teams, QuickTime, system screenshots) — for those, share a single window or a display Otto isn't on. While hidden, reach Otto by clicking its window; if you close it, re-open from this app (there's no menu bar or Dock icon until you turn this off).
            </span>
          </div>
        </div>
      </Card>

      {/* ── What it does ─────────────────────────────────────────────────── */}
      <Card title="How it works">
        <div className="space-y-3 text-xs text-th-text-tertiary leading-relaxed">
          <div className="flex gap-2.5">
            <span className="shrink-0 mt-0.5 text-emerald-500">1.</span>
            <p><strong className="text-th-text-secondary">App-layer guard</strong> — refuses to construct any cloud LLM (Anthropic, OpenAI) before a network call is ever made. The refusal is immediate and logged.</p>
          </div>
          <div className="flex gap-2.5">
            <span className="shrink-0 mt-0.5 text-emerald-500">2.</span>
            <p><strong className="text-th-text-secondary">Audit log</strong> — every engage, disengage, and blocked attempt is appended to a local JSONL file with a session token, making the no-cloud claim verifiable after the fact.</p>
          </div>
          <div className="flex gap-2.5">
            <span className="shrink-0 mt-0.5 text-emerald-500">3.</span>
            <p><strong className="text-th-text-secondary">Optional kernel firewall</strong> — generates a macOS <code className="font-mono">pf</code> anchor that blocks all outbound traffic at the OS level. Requires one <code className="font-mono">sudo</code> command — we never run it silently.</p>
          </div>
        </div>
      </Card>

      {/* ── Allowed hosts ────────────────────────────────────────────────── */}
      <Card title="Allowed hosts (app-layer allowlist)">
        <div className="space-y-3">
          <p className="text-xs text-th-text-tertiary leading-relaxed">
            These hosts bypass the app-layer block even when the lock is engaged. Use for on-premise LLM servers or private endpoints. Format: <code className="font-mono">host[:port]</code>.
          </p>
          {(status?.allowed_hosts ?? []).length > 0 ? (
            <ul className="space-y-1.5">
              {(status?.allowed_hosts ?? []).map((h) => (
                <li key={h} className="flex items-center justify-between gap-2 px-3 py-1.5 rounded-lg bg-th-inset-bg border border-th-border text-xs font-mono text-th-text-secondary">
                  {h}
                  <button
                    type="button"
                    onClick={() => void removeHost(h)}
                    disabled={savingHosts}
                    className="text-th-text-muted hover:text-rose-400 transition-colors"
                  >
                    <X size={12} />
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-xs text-th-text-muted italic">No exceptions — all cloud traffic blocked when engaged.</p>
          )}
          <div className="flex gap-2">
            <input
              className="flex-1 px-3 py-1.5 bg-th-input-bg border border-th-input-border rounded-lg text-xs text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-400"
              placeholder="api.mycompany.com:443"
              value={newHost}
              onChange={(e) => setNewHost(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") void addHost(); }}
            />
            <button
              type="button"
              className="px-3 py-1.5 text-xs font-medium rounded-lg border border-th-border bg-th-surface text-th-text-secondary hover:bg-th-surface-hover disabled:opacity-50 inline-flex items-center gap-1"
              onClick={() => void addHost()}
              disabled={!newHost.trim() || savingHosts}
            >
              {savingHosts ? <Loader2 size={11} className="animate-spin" /> : <Plus size={11} />}
              Add
            </button>
          </div>
        </div>
      </Card>

      {/* ── Kernel firewall (pf) ─────────────────────────────────────────── */}
      <Card title="Kernel firewall (macOS pf)">
        <div className="space-y-3">
          <p className="text-xs text-th-text-tertiary leading-relaxed">
            For a hardware-level guarantee, install the generated <code className="font-mono">pf</code> anchor. This blocks all outbound traffic at the OS kernel — independent of Otto. Requires one <code className="font-mono">sudo</code> command in Terminal.
          </p>
          <button
            type="button"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border border-th-border bg-th-surface text-th-text-secondary hover:bg-th-surface-hover disabled:opacity-50"
            onClick={() => void loadPf()}
            disabled={pfLoading}
          >
            {pfLoading ? <Loader2 size={11} className="animate-spin" /> : <Server size={11} />}
            {pfLoading ? "Generating…" : showPf ? "Refresh template" : "Show pf rules"}
          </button>

          {showPf && pfTemplate && (
            <div className="space-y-2">
              {/* Install command */}
              <div className="rounded-lg border border-th-border bg-th-inset-bg">
                <div className="flex items-center justify-between px-3 py-1.5 border-b border-th-border">
                  <span className="text-[10px] font-medium text-th-text-muted uppercase tracking-wide">Install command (run in Terminal)</span>
                  <button
                    type="button"
                    className="inline-flex items-center gap-1 text-[10px] text-th-text-muted hover:text-th-text-secondary"
                    onClick={() => copyText(pfTemplate.install_command, "cmd")}
                  >
                    {copied === "cmd" ? <ClipboardCheck size={11} className="text-emerald-400" /> : <Copy size={11} />}
                    {copied === "cmd" ? "Copied" : "Copy"}
                  </button>
                </div>
                <pre className="px-3 py-2 text-[11px] font-mono text-th-text-secondary overflow-x-auto whitespace-pre-wrap">{pfTemplate.install_command}</pre>
              </div>
              {/* Rules preview */}
              <div className="rounded-lg border border-th-border bg-th-inset-bg">
                <div className="flex items-center justify-between px-3 py-1.5 border-b border-th-border">
                  <span className="text-[10px] font-medium text-th-text-muted uppercase tracking-wide">Generated pf rules</span>
                  <button
                    type="button"
                    className="inline-flex items-center gap-1 text-[10px] text-th-text-muted hover:text-th-text-secondary"
                    onClick={() => copyText(pfTemplate.pf_template, "template")}
                  >
                    {copied === "template" ? <ClipboardCheck size={11} className="text-emerald-400" /> : <Copy size={11} />}
                    {copied === "template" ? "Copied" : "Copy"}
                  </button>
                </div>
                <pre className="px-3 py-2 text-[11px] font-mono text-th-text-tertiary overflow-x-auto max-h-64 overflow-y-auto leading-relaxed whitespace-pre-wrap">{pfTemplate.pf_template}</pre>
              </div>
              <p className="text-[10px] text-th-text-muted leading-relaxed">
                Verify after install: <code className="font-mono">sudo pfctl -a otto.privacy -s rules</code>.
                Remove: <code className="font-mono">sudo pfctl -a otto.privacy -F all</code>.
              </p>
            </div>
          )}
        </div>
      </Card>

      {/* ── Audit log ────────────────────────────────────────────────────── */}
      <Card title="Audit log">
        <div className="space-y-3">
          <p className="text-xs text-th-text-tertiary leading-relaxed">
            Append-only JSONL log of every engage, disengage, and blocked provider attempt. Each entry is stamped with a session audit token so you can prove which sessions ran under the lock.
          </p>
          <button
            type="button"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg border border-th-border bg-th-surface text-th-text-secondary hover:bg-th-surface-hover disabled:opacity-50"
            onClick={() => void loadAudit()}
            disabled={auditLoading}
          >
            {auditLoading ? <Loader2 size={11} className="animate-spin" /> : <RefreshCw size={11} />}
            {auditLoading ? "Loading…" : showAudit ? "Refresh" : "Load audit log"}
          </button>

          {showAudit && (
            audit.length === 0 ? (
              <p className="text-xs text-th-text-muted italic">No audit events yet.</p>
            ) : (
              <div className="rounded-lg border border-th-border overflow-hidden">
                <table className="w-full text-[11px]">
                  <thead>
                    <tr className="border-b border-th-border bg-th-inset-bg">
                      <th className="text-left px-3 py-1.5 font-medium text-th-text-muted">Time</th>
                      <th className="text-left px-3 py-1.5 font-medium text-th-text-muted">Event</th>
                      <th className="text-left px-3 py-1.5 font-medium text-th-text-muted hidden sm:table-cell">Detail</th>
                    </tr>
                  </thead>
                  <tbody>
                    {audit.map((e, i) => (
                      <tr key={i} className="border-b border-th-border last:border-0 hover:bg-th-surface-hover">
                        <td className="px-3 py-1.5 font-mono text-th-text-muted whitespace-nowrap">
                          {new Date(e.ts).toLocaleTimeString()}
                        </td>
                        <td className="px-3 py-1.5">
                          <span className={`inline-flex items-center gap-1 font-medium ${
                            e.event === "engage"           ? "text-emerald-400" :
                            e.event === "disengage"        ? "text-blue-400"    :
                            e.event === "refuse_provider"  ? "text-rose-400"    :
                                                             "text-th-text-secondary"
                          }`}>
                            {e.event === "engage"          && <Lock size={10} />}
                            {e.event === "disengage"       && <Unlock size={10} />}
                            {e.event === "refuse_provider" && <AlertTriangle size={10} />}
                            {e.event}
                          </span>
                        </td>
                        <td className="px-3 py-1.5 text-th-text-muted hidden sm:table-cell">
                          {e.provider ? `blocked: ${e.provider}` : ""}
                          {e.rotated === true ? "new session" : ""}
                          {e.was_engaged === false ? "was already off" : ""}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )
          )}
        </div>
      </Card>
    </div>
  );
}

// ---------------------------------------------------------------------------
// VoiceSettingsPanel — rendered for the "Voice" tab in SettingsPage
// ---------------------------------------------------------------------------

type VoiceSubTab = "stt" | "wake";

function FeatureToggle({
  enabled,
  onToggle,
}: {
  enabled: boolean;
  onToggle: () => void;
}) {
  return (
    <button
      onClick={onToggle}
      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500/50 ${
        enabled ? "bg-blue-500" : "bg-zinc-600"
      }`}
    >
      <span
        className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform ${
          enabled ? "translate-x-6" : "translate-x-1"
        }`}
      />
    </button>
  );
}

function SettingRow({
  label,
  description,
  children,
}: {
  label: string;
  description?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between gap-4">
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-th-text-primary">{label}</p>
        {description && (
          <p className="text-xs text-th-text-secondary mt-0.5 leading-relaxed">{description}</p>
        )}
      </div>
      <div className="flex-shrink-0">{children}</div>
    </div>
  );
}

function SectionCard({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`rounded-xl border border-th-border bg-th-surface p-4 space-y-4 ${className}`}>
      {children}
    </div>
  );
}

function SectionDivider() {
  return <div className="border-t border-th-border" />;
}

function VoiceSettingsPanel({
  settings,
  setSettings,
  onSave,
}: {
  settings: AppSettings;
  setSettings: React.Dispatch<React.SetStateAction<AppSettings>>;
  onSave: (next: AppSettings) => Promise<void>;
}) {
  const voice: VoiceConfig = {
    ...(settings.voice ?? {}),
    enabled: settings.voice?.enabled ?? false,
    activation_mode: settings.voice?.activation_mode ?? "ptt",
    ptt_hotkey: settings.voice?.ptt_hotkey ?? "",
    stt_model: settings.voice?.stt_model ?? "mlx-community/whisper-large-v3-turbo",
    stt_language: settings.voice?.stt_language ?? "",
    stt_enabled: settings.voice?.stt_enabled ?? true,
    wake_model: settings.voice?.wake_model ?? "hey_otto",
    wake_enabled: settings.voice?.wake_enabled ?? false,
    vad_silence_secs: settings.voice?.vad_silence_secs ?? 1.0,
    mic_device: settings.voice?.mic_device ?? "",
    loopback_enabled: settings.voice?.loopback_enabled ?? false,
    loopback_vad_silence_secs: settings.voice?.loopback_vad_silence_secs ?? 0.7,
    loopback_max_segment_secs: settings.voice?.loopback_max_segment_secs ?? 12.0,
    loopback_live_partials: settings.voice?.loopback_live_partials ?? true,
    loopback_partial_interval_secs: settings.voice?.loopback_partial_interval_secs ?? 1.5,
    loopback_auto_send_silence_secs: settings.voice?.loopback_auto_send_silence_secs ?? 2.5,
  };

  const patch = async (partial: Partial<VoiceConfig>) => {
    const next: AppSettings = { ...settings, voice: { ...voice, ...partial } };
    setSettings(next);
    await onSave(next);
  };

  const [voiceSubTab, setVoiceSubTab] = useState<VoiceSubTab>(() => {
    const saved = sessionStorage.getItem("otto:settings:voiceSubTab");
    return saved === "wake" ? "wake" : "stt";
  });
  useEffect(() => { sessionStorage.setItem("otto:settings:voiceSubTab", voiceSubTab); }, [voiceSubTab]);
  const [inputDevices, setInputDevices] = useState<Array<{ index: number; name: string; default: boolean }>>([]);
  const [loopbackStatus, setLoopbackStatus] = useState<import("../types").LoopbackStatus | null>(null);

  useEffect(() => {
    api.voiceStatus().then((s) => {
      setInputDevices(s.input_devices ?? []);
    }).catch(() => {});
    api.loopbackStatus().then(setLoopbackStatus).catch(() => {});
  }, []);

  // STT test state
  const [sttRecording, setSttRecording] = useState(false);
  const [sttPhase, setSttPhase] = useState<"idle" | "downloading" | "transcribing">("idle");
  const [sttResult, setSttResult] = useState<string | null>(null);
  const [sttError, setSttError] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const startSttRecording = async () => {
    setSttResult(null);
    setSttError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      mediaRecorderRef.current = mr;
      audioChunksRef.current = [];
      mr.ondataavailable = (e) => { if (e.data.size > 0) audioChunksRef.current.push(e.data); };
      mr.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(audioChunksRef.current, { type: mr.mimeType });
        try {
          const arrayBuffer = await blob.arrayBuffer();
          const audioCtx = new AudioContext();
          const decoded = await audioCtx.decodeAudioData(arrayBuffer);
          await audioCtx.close();
          const targetSr = 16_000;
          const offlineCtx = new OfflineAudioContext(1, Math.ceil(decoded.duration * targetSr), targetSr);
          const src = offlineCtx.createBufferSource();
          src.buffer = decoded;
          src.connect(offlineCtx.destination);
          src.start(0);
          const rendered = await offlineCtx.startRendering();
          const float32 = rendered.getChannelData(0);
          const pcmBytes = new Uint8Array(float32.buffer);
          let binary = "";
          const chunkSize = 8192;
          for (let i = 0; i < pcmBytes.length; i += chunkSize) {
            binary += String.fromCharCode(...pcmBytes.subarray(i, i + chunkSize));
          }
          const b64 = btoa(binary);
          const { cached } = await api.voiceModelCached(voice.stt_model);
          setSttPhase(cached ? "transcribing" : "downloading");
          const res = await api.voiceSttTest(b64, voice.stt_language || "en");
          if (res.error) setSttError(res.error);
          else setSttResult(res.text ?? "");
        } catch (err: unknown) {
          setSttError(err instanceof Error ? err.message : String(err));
        } finally {
          setSttPhase("idle");
        }
      };
      mr.start();
      setSttRecording(true);
    } catch (err: unknown) {
      setSttError(err instanceof Error ? err.message : String(err));
    }
  };

  const stopSttRecording = () => {
    mediaRecorderRef.current?.stop();
    setSttRecording(false);
    setSttPhase("transcribing");
  };

  const wakePhrase = "Hey Otto";

  // Wake word test state
  const [wakeTestPhase, setWakeTestPhase] = useState<"idle" | "listening" | "detected" | "error">("idle");
  const [wakeTestError, setWakeTestError] = useState<string | null>(null);
  const wakeWsRef = useRef<WebSocket | null>(null);
  const wakeTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const wakeTestPhaseRef = useRef<"idle" | "listening" | "detected" | "error">("idle");
  const setWakePhase = (p: "idle" | "listening" | "detected" | "error") => {
    wakeTestPhaseRef.current = p;
    setWakeTestPhase(p);
  };

  const stopWakeTest = () => {
    if (wakeTimeoutRef.current) { clearTimeout(wakeTimeoutRef.current); wakeTimeoutRef.current = null; }
    if (wakeWsRef.current) {
      try { wakeWsRef.current.send(JSON.stringify({ type: "stop" })); } catch { /* ignore */ }
      wakeWsRef.current.close();
      wakeWsRef.current = null;
    }
  };

  const startWakeTest = () => {
    setWakeTestError(null);
    setWakePhase("listening");
    stopWakeTest();
    try {
      const ws = new WebSocket(`${WS_BASE}/ws/voice`);
      wakeWsRef.current = ws;
      ws.onopen = () => {
        ws.send(JSON.stringify({ type: "configure", config: { activation_mode: "wakeword", wake_model: "hey_otto" } }));
        ws.send(JSON.stringify({ type: "start" }));
      };
      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data);
          if (msg.type === "wake") {
            setWakePhase("detected");
            stopWakeTest();
          } else if (msg.type === "error") {
            setWakeTestError(msg.message ?? "Unknown error");
            setWakePhase("error");
            stopWakeTest();
          }
        } catch { /* ignore */ }
      };
      ws.onerror = () => { setWakeTestError("WebSocket error"); setWakePhase("error"); };
      ws.onclose = () => { if (wakeTestPhaseRef.current === "listening") setWakePhase("idle"); };
      // Auto-stop after 15 seconds
      wakeTimeoutRef.current = setTimeout(() => {
        setWakePhase("idle");
        stopWakeTest();
      }, 15_000);
    } catch (err: unknown) {
      setWakeTestError(err instanceof Error ? err.message : String(err));
      setWakePhase("error");
    }
  };

  // Cleanup on unmount
  useEffect(() => () => stopWakeTest(), []);

  const subTabs: Array<{
    id: VoiceSubTab;
    label: string;
    icon: React.ReactNode;
    enabled: boolean;
  }> = [
    {
      id: "stt",
      label: "Speech to Text",
      icon: <Mic className="w-3.5 h-3.5" />,
      enabled: voice.stt_enabled,
    },
    {
      id: "wake",
      label: "Wake Word",
      icon: <Zap className="w-3.5 h-3.5" />,
      enabled: voice.wake_enabled,
    },
  ];

  return (
    <div className="space-y-5 max-w-2xl">
      {/* ── Master enable card ── */}
      <SectionCard>
        <SettingRow
          label="Voice Mode"
          description="Adds a mic button to chat and a hands-free Voice screen."
        >
          <FeatureToggle enabled={voice.enabled} onToggle={() => patch({ enabled: !voice.enabled })} />
        </SettingRow>
      </SectionCard>

      {/* ── System audio (loopback) transcription ── */}
      <SectionCard>
        <SettingRow
          label="System Audio Transcription"
          description="Transcribe what your Mac is playing (meetings, calls, media) via a native macOS tap. Opens from the sidebar's Transcribe audio panel."
        >
          <FeatureToggle
            enabled={voice.loopback_enabled}
            onToggle={() => patch({ loopback_enabled: !voice.loopback_enabled })}
          />
        </SettingRow>

        {loopbackStatus && !loopbackStatus.supported && (
          <p className="mt-2 text-xs text-amber-400">
            Requires macOS 14.4 or later.
          </p>
        )}
        {loopbackStatus && loopbackStatus.supported && !loopbackStatus.helper_available && (
          <p className="mt-2 text-xs text-amber-400">
            Capture helper not found — build it with <code>app/src-tauri/build-audiotap.sh</code>.
          </p>
        )}

        {voice.loopback_enabled && (
          <div className="mt-4 space-y-4">
            <div>
              <label className="block text-xs font-medium text-th-text-secondary mb-1.5">
                Silence threshold <span className="opacity-50 font-normal">(seconds before finalising a sentence)</span>
              </label>
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  min="0.3"
                  max="2"
                  step="0.1"
                  value={voice.loopback_vad_silence_secs}
                  onChange={(e) => patch({ loopback_vad_silence_secs: parseFloat(e.target.value) })}
                  className="flex-1 accent-blue-500"
                />
                <span className="text-sm font-medium text-th-text-primary w-12 text-right tabular-nums">
                  {voice.loopback_vad_silence_secs.toFixed(1)}s
                </span>
              </div>
            </div>

            <SettingRow
              label="Live partial transcripts"
              description="Show text while audio is still playing. Uses more CPU."
            >
              <FeatureToggle
                enabled={voice.loopback_live_partials}
                onToggle={() => patch({ loopback_live_partials: !voice.loopback_live_partials })}
              />
            </SettingRow>

            <div>
              <label className="block text-xs font-medium text-th-text-secondary mb-1.5">
                Auto-send pause <span className="opacity-50 font-normal">(silence before new lines are sent to Otto)</span>
              </label>
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  min="1"
                  max="8"
                  step="0.5"
                  value={voice.loopback_auto_send_silence_secs}
                  onChange={(e) => patch({ loopback_auto_send_silence_secs: parseFloat(e.target.value) })}
                  className="flex-1 accent-blue-500"
                />
                <span className="text-sm font-medium text-th-text-primary w-12 text-right tabular-nums">
                  {voice.loopback_auto_send_silence_secs.toFixed(1)}s
                </span>
              </div>
              <p className="text-xs text-th-text-muted mt-1">
                Only applies when you turn on <span className="font-medium">Auto</span> in the
                transcription panel. Longer pauses batch more speech per message.
              </p>
            </div>

            <p className="text-xs text-th-text-muted">
              System audio is captured in software before it reaches your speakers or
              headphones — you keep hearing everything normally. All transcription
              runs on-device. macOS will ask for System Audio Recording permission the
              first time you start.
            </p>
          </div>
        )}
      </SectionCard>

      {voice.enabled && (
        <>
          {/* ── Sub-pill navigation ── */}
          <div className="flex items-center gap-1.5 p-1 rounded-xl bg-th-surface border border-th-border">
            {subTabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setVoiceSubTab(tab.id)}
                className={`flex-1 flex items-center justify-center gap-2 py-2 px-3 rounded-lg text-xs font-semibold transition-all duration-150 ${
                  voiceSubTab === tab.id
                    ? "bg-blue-600 text-white shadow-sm"
                    : "text-th-text-secondary hover:text-th-text-primary hover:bg-th-input-bg"
                }`}
              >
                {tab.icon}
                <span className="hidden sm:inline">{tab.label}</span>
                {/* Status dot */}
                <span
                  className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
                    tab.enabled
                      ? voiceSubTab === tab.id
                        ? "bg-white/70"
                        : "bg-green-400"
                      : voiceSubTab === tab.id
                      ? "bg-white/30"
                      : "bg-zinc-600"
                  }`}
                />
              </button>
            ))}
          </div>

          {/* ── STT tab ── */}
          {voiceSubTab === "stt" && (
            <div className="space-y-4">
              {/* Feature toggle */}
              <SectionCard>
                <SettingRow
                  label="Enable Speech to Text"
                  description="Transcribe your voice to text using an on-device Whisper model."
                >
                  <FeatureToggle
                    enabled={voice.stt_enabled}
                    onToggle={() => patch({ stt_enabled: !voice.stt_enabled })}
                  />
                </SettingRow>
              </SectionCard>

              {voice.stt_enabled && (
                <>
                  {/* Input mode */}
                  <SectionCard>
                    <p className="text-xs font-semibold text-th-text-secondary uppercase tracking-wider">Input mode</p>
                    <div className="grid grid-cols-2 gap-2">
                      {(["ptt", "wakeword"] as const).map((mode) => (
                        <button
                          key={mode}
                          onClick={() => patch({ activation_mode: mode })}
                          className={`rounded-lg border p-3 text-left transition-all duration-150 ${
                            voice.activation_mode === mode
                              ? "border-blue-500/60 bg-blue-500/10 ring-1 ring-blue-500/20"
                              : "border-th-border bg-th-input-bg hover:border-th-border-hover"
                          }`}
                        >
                          <div className="flex items-center gap-2 mb-0.5">
                            {mode === "ptt"
                              ? <Mic className={`w-3.5 h-3.5 ${voice.activation_mode === mode ? "text-blue-400" : "text-th-text-secondary"}`} />
                              : <Zap className={`w-3.5 h-3.5 ${voice.activation_mode === mode ? "text-blue-400" : "text-th-text-secondary"}`} />
                            }
                            <p className={`text-sm font-medium ${voice.activation_mode === mode ? "text-blue-400" : "text-th-text-primary"}`}>
                              {mode === "ptt" ? "Hold to speak" : "Wake word"}
                            </p>
                          </div>
                          <p className="text-xs text-th-text-secondary leading-relaxed">
                            {mode === "ptt" ? "Press the mic button or use a hotkey" : `Say "${wakePhrase}" to activate`}
                          </p>
                        </button>
                      ))}
                    </div>

                    {voice.activation_mode === "ptt" && (
                      <>
                        <SectionDivider />
                        <div>
                          <label className="block text-xs font-medium text-th-text-secondary mb-1.5">
                            Global hotkey <span className="opacity-50 font-normal">(optional)</span>
                          </label>
                          <input
                            className="w-full bg-th-input-bg border border-th-input-border rounded-lg px-3 py-2 text-sm text-th-text-primary focus:outline-none focus:border-blue-400/60 focus:ring-1 focus:ring-blue-400/20 transition-colors"
                            value={voice.ptt_hotkey}
                            onChange={(e) => patch({ ptt_hotkey: e.target.value })}
                            placeholder="e.g. Control+Space"
                          />
                        </div>
                      </>
                    )}
                  </SectionCard>

                  {/* Language & mic */}
                  <SectionCard>
                    <p className="text-xs font-semibold text-th-text-secondary uppercase tracking-wider">Settings</p>
                    <div>
                      <label className="block text-xs font-medium text-th-text-secondary mb-1.5">Language</label>
                      <input
                        className="w-full bg-th-input-bg border border-th-input-border rounded-lg px-3 py-2 text-sm text-th-text-primary focus:outline-none focus:border-blue-400/60 focus:ring-1 focus:ring-blue-400/20 transition-colors"
                        value={voice.stt_language}
                        onChange={(e) => patch({ stt_language: e.target.value })}
                        placeholder="Default: en — set to fr, ja, de, etc."
                      />
                    </div>

                    {inputDevices.length > 0 && (
                      <>
                        <SectionDivider />
                        <div>
                          <label className="block text-xs font-medium text-th-text-secondary mb-1.5">Microphone</label>
                          <select
                            className="w-full bg-th-input-bg border border-th-input-border rounded-lg px-3 py-2 text-sm text-th-text-primary focus:outline-none focus:border-blue-400/60 transition-colors"
                            value={voice.mic_device}
                            onChange={(e) => patch({ mic_device: e.target.value })}
                          >
                            <option value="">System default</option>
                            {inputDevices.map((d) => (
                              <option key={d.index} value={String(d.index)}>
                                {d.name}{d.default ? " (default)" : ""}
                              </option>
                            ))}
                          </select>
                        </div>
                      </>
                    )}
                  </SectionCard>

                  {/* STT test */}
                  <SectionCard>
                    <p className="text-xs font-semibold text-th-text-secondary uppercase tracking-wider">Test</p>
                    <div className="space-y-3">
                      <div className="flex items-center gap-2 flex-wrap">
                        <button
                          onClick={sttRecording ? stopSttRecording : startSttRecording}
                          disabled={sttPhase !== "idle" && !sttRecording}
                          className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition-all disabled:opacity-50 ${
                            sttRecording
                              ? "bg-red-500/15 border border-red-500/40 text-red-400 hover:bg-red-500/25"
                              : "bg-th-input-bg border border-th-input-border text-th-text-primary hover:border-th-border-hover"
                          }`}
                        >
                          {sttPhase !== "idle" && !sttRecording ? (
                            <Loader2 size={13} className="animate-spin" />
                          ) : (
                            <Mic size={13} className={sttRecording ? "animate-pulse" : ""} />
                          )}
                          {sttPhase === "downloading"
                            ? "Downloading model…"
                            : sttPhase === "transcribing"
                            ? "Transcribing…"
                            : sttRecording
                            ? "Stop recording"
                            : "Record & transcribe"}
                        </button>
                        {sttRecording && (
                          <span className="flex items-center gap-1.5 text-xs text-red-400">
                            <span className="w-2 h-2 rounded-full bg-red-400 animate-pulse" />
                            Recording…
                          </span>
                        )}
                      </div>
                      {sttPhase === "downloading" && (
                        <p className="text-xs text-th-text-secondary">
                          First run — downloading Whisper to your Mac. This may take a few minutes.
                        </p>
                      )}
                      {sttResult !== null && (
                        <div className="rounded-lg bg-th-input-bg border border-th-input-border px-3 py-2 text-sm text-th-text-primary">
                          {sttResult || <span className="italic text-th-text-secondary">No speech detected</span>}
                        </div>
                      )}
                      {sttError && <p className="text-xs text-red-400">{sttError}</p>}
                    </div>
                  </SectionCard>

                  {/* STT model chooser */}
                  <SectionCard>
                    <p className="text-xs font-semibold text-th-text-secondary uppercase tracking-wider">Model</p>
                    <VoiceModelChooser
                      config={voice}
                      onSelectStt={(id) => patch({ stt_model: id })}
                      kinds={["stt"]}
                      hfToken={settings.llm?.mlx?.hf_token ?? ""}
                    />
                  </SectionCard>
                </>
              )}
            </div>
          )}

          {/* ── Wake Word tab ── */}
          {voiceSubTab === "wake" && (
            <div className="space-y-4">
              {/* Feature toggle */}
              <SectionCard>
                <SettingRow
                  label="Enable Wake Word"
                  description={`Always-on listening — say "${wakePhrase}" to activate hands-free.`}
                >
                  <FeatureToggle
                    enabled={voice.wake_enabled}
                    onToggle={() => {
                      const next = !voice.wake_enabled;
                      patch({
                        wake_enabled: next,
                        activation_mode: next ? "wakeword" : "ptt",
                        wake_model: "hey_otto",
                      });
                    }}
                  />
                </SettingRow>
              </SectionCard>

              {voice.wake_enabled && (
                <>
                  {/* VAD silence setting */}
                  <SectionCard>
                    <p className="text-xs font-semibold text-th-text-secondary uppercase tracking-wider">Settings</p>
                    <div>
                      <label className="block text-xs font-medium text-th-text-secondary mb-1.5">
                        Silence threshold <span className="opacity-50 font-normal">(seconds before stopping)</span>
                      </label>
                      <div className="flex items-center gap-3">
                        <input
                          type="range"
                          min="0.3"
                          max="3"
                          step="0.1"
                          value={voice.vad_silence_secs}
                          onChange={(e) => patch({ vad_silence_secs: parseFloat(e.target.value) })}
                          className="flex-1 accent-blue-500"
                        />
                        <span className="text-sm font-medium text-th-text-primary w-12 text-right tabular-nums">
                          {voice.vad_silence_secs.toFixed(1)}s
                        </span>
                      </div>
                    </div>
                  </SectionCard>

                  {/* Wake phrase — fixed, no user selection needed */}
                  <SectionCard>
                    <p className="text-xs font-semibold text-th-text-secondary uppercase tracking-wider">Wake phrase</p>
                    <div className="flex items-center gap-3 py-1">
                      <span className="text-sm font-medium text-th-text-primary">"Hey Otto"</span>
                      <span className="text-xs text-th-text-secondary">Built-in on-device model · no download required</span>
                    </div>
                  </SectionCard>

                  {/* Wake word test */}
                  <SectionCard>
                    <p className="text-xs font-semibold text-th-text-secondary uppercase tracking-wider mb-3">Test wake word</p>
                    <div className="flex items-center gap-3">
                      {wakeTestPhase === "idle" || wakeTestPhase === "error" ? (
                        <button
                          onClick={startWakeTest}
                          className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-th-bg-secondary border border-th-border text-sm font-medium text-th-text-primary hover:bg-th-bg-tertiary transition-colors"
                        >
                          <Mic className="w-3.5 h-3.5" />
                          Start listening
                        </button>
                      ) : wakeTestPhase === "listening" ? (
                        <button
                          onClick={() => { setWakePhase("idle"); stopWakeTest(); }}
                          className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-red-500/10 border border-red-500/30 text-sm font-medium text-red-400 hover:bg-red-500/20 transition-colors"
                        >
                          <span className="w-2 h-2 rounded-full bg-red-400 animate-pulse" />
                          Stop
                        </button>
                      ) : null}

                      <span className="text-sm text-th-text-secondary">
                        {wakeTestPhase === "idle" && <>Press start, then say &ldquo;Hey Otto&rdquo;</>}
                        {wakeTestPhase === "listening" && (
                          <span className="flex items-center gap-1.5">
                            <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-pulse inline-block" />
                            Listening… say <span className="font-medium text-th-text-primary">"Hey Otto"</span>
                          </span>
                        )}
                        {wakeTestPhase === "detected" && (
                          <span className="flex items-center gap-1.5 text-green-400 font-medium">
                            <span className="text-base">✓</span> Wake word detected!
                          </span>
                        )}
                        {wakeTestPhase === "error" && (
                          <span className="text-red-400 text-xs">{wakeTestError}</span>
                        )}
                      </span>

                      {wakeTestPhase === "detected" && (
                        <button
                          onClick={() => setWakePhase("idle")}
                          className="ml-auto text-xs text-th-text-secondary hover:text-th-text-primary transition-colors"
                        >
                          Reset
                        </button>
                      )}
                    </div>
                    {wakeTestPhase === "listening" && (
                      <p className="text-xs text-th-text-secondary mt-2 opacity-60">Times out after 15 seconds</p>
                    )}
                  </SectionCard>
                </>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
