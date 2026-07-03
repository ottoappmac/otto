/**
 * SetupChatPage — conversational first-run setup for eligible Apple Silicon Macs.
 *
 * Step sequence mirrors SetupWizard.tsx.  Config mutations go through the same
 * existing APIs the wizard uses:
 *   PUT /api/settings                    — provider, keys, memory, activity
 *   POST /api/settings/test-connection   — validates cloud API keys
 *   POST /api/mlx/download               — pulls the user's chosen local model
 *   POST /api/setup/complete             — marks first-run done
 *
 * POST /api/setup/chat is called only to generate natural-language reply text.
 * It has no side effects on config.
 *
 * Azure OpenAI, AWS Bedrock, and oMLX are redirected to the legacy wizard
 * because they require multiple credential fields that don't map cleanly to a
 * single chat exchange.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { Settings2, Loader2, Sparkles, ArrowRight, Lock, Cpu, Cloud, Send, Check, ChevronDown, ChevronUp, ChevronLeft, Info, Terminal } from "lucide-react";
import { api } from "../../hooks/useApi";
import { useTheme } from "../../context/ThemeContext";
import type { AppSettings, MlxCapabilities, OmlxInfo, OmlxStatus } from "../../types";
import logoDark from "../../assets/logo-dark.png";
import logoLight from "../../assets/logo-light.png";
import ModelChooser from "../mlx/ModelChooser";
import ClusterSetupFlow from "../cluster/ClusterSetupFlow";
import { OmlxModelPicker } from "../omlx/OmlxModelPicker";
import { ThinkingIndicator } from "../chat/ThinkingIndicator";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Phase = "picker" | "preparing" | "chat";

type SetupStep =
  | "provider"
  | "cloud_sub"
  | "cloud_key"
  | "cloud_model"
  | "local_model"
  | "azure_endpoint"       // Azure OpenAI endpoint URL
  | "azure_key"            // Azure OpenAI API key
  | "azure_deployment"     // Azure deployment name
  | "azure_api_version"    // Azure API version
  | "bedrock_key_id"       // AWS Access Key ID inline input
  | "bedrock_secret"       // AWS Secret Access Key inline input
  | "bedrock_region"       // AWS region inline input
  | "exo_remote_confirm"   // "Add this Mac to cluster? Yes / Skip"
  | "exo_username"         // confirm / override the remote username before probing
  | "exo_ssh_password"     // secure password entry — never forwarded to agent
  | "memory"
  | "memory_inject"
  | "activity"
  | "ambient"
  | "evaluation"
  | "done";

type ProviderChoice = "mlx" | "anthropic" | "openai" | "azure" | "bedrock" | "omlx" | "exo" | "cloud" | null;
type CloudSub = "anthropic" | "openai" | null;

interface ModelPickerOption {
  id: string;
  label: string;
  sizeGb: string;
  /** Visual badge shown alongside the size chip */
  badge?: { text: string; color: "green" | "yellow" | "red" | "gray" };
}

interface Message {
  role: "assistant" | "user";
  content: string;
  isLoading?: boolean;
  /** Clickable quick-reply chips shown below the message. Cleared once used. */
  quickReplies?: string[];
  /** Optional tooltip text keyed by quick-reply value, shown via an info icon. */
  quickReplyTooltips?: Record<string, string>;
  /** Inline text / password input embedded in the bubble. Cleared once submitted. */
  inlineInput?: {
    type: "text" | "password";
    placeholder?: string;
    prefill?: string;
  };
  /** Inline model-picker dropdown shown below the message. Cleared once used. */
  modelPicker?: {
    options: ModelPickerOption[];
    /** Default selected index */
    defaultIdx?: number;
    /** Whether to show a "Skip, choose later" link */
    allowSkip?: boolean;
  };
  /**
   * Rich full-catalog model picker — renders ExoModelChooser, ModelChooser, or
   * OmlxModelPicker inline in the chat bubble (same UI as Settings → Cluster →
   * Model and Settings → On Device → Models).  Cleared once a model is confirmed.
   */
  richModelPicker?: {
    type: "exo" | "mlx" | "omlx";
    /** Whether to show a "Skip, choose later" link */
    allowSkip?: boolean;
  };
  /**
   * Mounts the shared ClusterSetupFlow inline in the bubble — the single
   * Cluster setup experience (also used by SetupWizard + Settings). Replaces
   * the old bespoke chat-driven EXO pipeline. Cleared once setup finishes.
   */
  clusterSetup?: boolean;
  /**
   * Structured command/job output shown as a collapsible terminal panel below
   * the message bubble.  Lines are appended as the job streams progress.
   */
  commandOutput?: {
    lines: string[];
    status: "running" | "done" | "error";
  };
}

/** Snapshot of conversational state captured before each step is processed,
 *  enabling the user to go back one exchange at a time. */
interface HistoryEntry {
  messages: Message[];
  step: SetupStep;
  providerChoice: ProviderChoice;
  cloudSub: CloudSub;
  inputType: "text" | "password";
  inlineInputValues: Record<number, string>;
}

interface SetupChatPageProps {
  onFinish: () => void;
  /** Called when the user selects a provider that needs the full wizard
   * (Azure, Bedrock, oMLX) or explicitly clicks "Set up manually". */
  onUseLegacy: () => void;
  /** Called when the user skips setup entirely. */
  onSkip: () => void;
  capabilities: MlxCapabilities;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SETUP_MODEL_ID = "mlx-community/Qwen3-1.7B-4bit";

interface ModelOption { id: string; label: string; sizeGb: string; minRam: number }

const MLX_CATALOG: ModelOption[] = [
  { id: "mlx-community/Qwen3-32B-4bit",  label: "Qwen3-32B — best quality",   sizeGb: "18.2", minRam: 48 },
  { id: "mlx-community/Qwen3-14B-4bit",  label: "Qwen3-14B — great quality",  sizeGb: "8.5",  minRam: 24 },
  { id: "mlx-community/Qwen3-8B-4bit",   label: "Qwen3-8B — balanced",        sizeGb: "4.6",  minRam: 12 },
  { id: "mlx-community/Qwen3-4B-4bit",   label: "Qwen3-4B — light",           sizeGb: "2.4",  minRam: 8  },
  { id: "mlx-community/Qwen3-1.7B-4bit", label: "Qwen3-1.7B — very light",    sizeGb: "1.1",  minRam: 0  },
];

function recommendedMainModel(ramGb: number): ModelOption {
  return (
    MLX_CATALOG.find((m) => m.minRam <= ramGb) ?? MLX_CATALOG[MLX_CATALOG.length - 1]
  );
}

/** Returns up to 4 models that comfortably fit in available RAM, top-first. */
function mlxModelOptions(ramGb: number): ModelOption[] {
  return MLX_CATALOG.filter((m) => m.minRam <= ramGb).slice(0, 4);
}

/** Curated cloud models shown before the live list is fetched. */
const CURATED_CLOUD_MODELS: Record<"anthropic" | "openai", { id: string; label: string }[]> = {
  anthropic: [
    { id: "claude-sonnet-4-6",         label: "Claude Sonnet 4.6 (recommended)" },
    { id: "claude-opus-4-1",           label: "Claude Opus 4.1" },
    { id: "claude-haiku-4-5",          label: "Claude Haiku 4.5 (fast & cheap)" },
    { id: "claude-3-7-sonnet-latest",  label: "Claude 3.7 Sonnet" },
    { id: "claude-3-5-sonnet-latest",  label: "Claude 3.5 Sonnet" },
  ],
  openai: [
    { id: "gpt-4o",       label: "GPT-4o (recommended)" },
    { id: "gpt-4o-mini",  label: "GPT-4o mini (fast & cheap)" },
    { id: "gpt-4.1",      label: "GPT-4.1" },
    { id: "gpt-5",        label: "GPT-5" },
    { id: "o3",           label: "o3 (reasoning)" },
  ],
};

const OPENING_MESSAGE =
  "Welcome to Otto! How would you like to run your AI?";

const OPENING_REPLIES = [
  "Standard (MLX)",
  "Turbo (oMLX)",
  "Cluster",
  "Frontier",
];

const OPENING_TOOLTIPS: Record<string, string> = {
  "Standard (MLX)":
    "Runs fully offline on this Mac using Apple Silicon's unified memory. Private, free, and no internet required.",
  "Turbo (oMLX)":
    "A local inference server with continuous batching and paged KV cache — faster for heavy workloads, still 100% on-device.",
  "Cluster":
    "Splits a large model across two or more Macs over Thunderbolt or LAN. Best for models too big for one machine.",
  "Frontier":
    "Connects to a cloud AI provider (Anthropic Claude or OpenAI GPT). Requires an API key; data leaves this Mac.",
};

// ---------------------------------------------------------------------------
// Natural-language extraction helpers
// ---------------------------------------------------------------------------

function extractProvider(msg: string): ProviderChoice {
  const m = msg.toLowerCase();
  if (/\b(exo|cluster|distributed|thunderbolt|multi.?mac)\b/.test(m)) return "exo";
  if (/\b(turbo|omlx|omlx server|server|advanced)\b/.test(m)) return "omlx";
  if (/\b(local|mac|on.?device|apple silicon|mlx|offline|private|locally)\b/.test(m)) return "mlx";
  if (/\b(claude|anthropic)\b/.test(m)) return "anthropic";
  if (/\b(gpt|chatgpt|openai)\b/.test(m)) return "openai";
  if (/\b(azure)\b/.test(m)) return "azure";
  if (/\b(bedrock|aws|amazon)\b/.test(m)) return "bedrock";
  if (/\b(frontier|cloud|api key|api)\b/.test(m)) return "cloud";
  return null;
}

function extractCloudSub(msg: string): CloudSub | "azure" | "bedrock" {
  const m = msg.toLowerCase();
  // Azure must be checked before OpenAI — "Azure OpenAI" contains the word "openai"
  if (/\b(azure)\b/.test(m)) return "azure";
  if (/\b(bedrock|aws|amazon)\b/.test(m)) return "bedrock";
  if (/\b(anthropic|claude)\b/.test(m)) return "anthropic";
  if (/\b(openai|gpt|chatgpt)\b/.test(m)) return "openai";
  return null as unknown as "anthropic"; // caller checks for null via `== null` pattern
}

function extractBool(msg: string): boolean | null {
  const m = msg.toLowerCase().trim();
  if (/\b(yes|y|sure|ok(ay)?|please|enable|on|definitely|yep|yup|absolutely|go ahead)\b/.test(m))
    return true;
  if (/\b(no|n|nope|nah|skip|off|disable|not really|no thanks|pass)\b/.test(m)) return false;
  return null;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function SetupChatPage({
  onFinish,
  onUseLegacy,
  onSkip,
  capabilities,
}: SetupChatPageProps) {
  const { theme } = useTheme();
  const logo = theme === "dark" ? logoDark : logoLight;

  // "picker"   — choice screen (auto vs manual)
  // "preparing" — model needed but not cached; downloading with progress
  // "chat"     — conversational setup
  const [phase, setPhase] = useState<Phase>("picker");

  // Full settings object loaded on mount — used to do proper deep-merges
  // before sending to the backend (which does model_validate, not a merge).
  const [settings, setSettings] = useState<AppSettings | null>(null);

  useEffect(() => {
    api.getSettings().then(setSettings).catch(() => undefined);
  }, []);

  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", content: OPENING_MESSAGE, quickReplies: OPENING_REPLIES, quickReplyTooltips: OPENING_TOOLTIPS },
  ]);
  const [input, setInput] = useState("");
  const [step, setStep] = useState<SetupStep>("provider");
  const [providerChoice, setProviderChoice] = useState<ProviderChoice>(null);
  const [cloudSub, setCloudSub] = useState<CloudSub>(null);
  /** Live model list fetched after a successful API key test */
  const [cloudModelList, setCloudModelList] = useState<{ id: string; label: string }[]>([]);
  const [isBusy, setIsBusy] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [downloadJobId, setDownloadJobId] = useState<string | null>(null);
  const [downloadProgress, setDownloadProgress] = useState<number | null>(null);
  const [inputType, setInputType] = useState<"text" | "password">("text");
  /** Per-message inline input values keyed by message index */
  const [inlineInputValues, setInlineInputValues] = useState<Record<number, string>>({});
  /** Undo stack — each entry is a snapshot taken just before a step is processed. */
  const [historyStack, setHistoryStack] = useState<HistoryEntry[]>([]);
  /** Set of message indices whose commandOutput panel is collapsed. All outputs start expanded. */
  const [collapsedOutputs, setCollapsedOutputs] = useState<Set<number>>(new Set<number>());

  // Azure OpenAI inline-flow credential state
  const [azureEndpoint, setAzureEndpoint] = useState("");
  const [azureApiKey, setAzureApiKey] = useState("");
  const [azureDeployment, setAzureDeployment] = useState("");
  const [azureApiVersion, setAzureApiVersion] = useState("2024-12-01-preview");

  // Bedrock inline-flow credential state
  const [bedrockKeyId, setBedrockKeyId] = useState("");
  const [bedrockSecret, setBedrockSecret] = useState("");
  const [bedrockRegion, setBedrockRegion] = useState("us-east-1");

  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // Mirrors downloadJobId so callbacks can access the current value without
  // stale-closure problems.
  const activeJobIdRef = useRef<string | null>(null);

  // Keep the ref in sync whenever the state changes.
  useEffect(() => {
    activeJobIdRef.current = downloadJobId;
  }, [downloadJobId]);

  // Always scroll to the bottom as new messages arrive
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Focus the input whenever the chat is ready
  useEffect(() => {
    if (phase === "chat" && !isBusy) inputRef.current?.focus();
  }, [phase, isBusy]);

  // Transition from "preparing" to "chat" once the download finishes
  useEffect(() => {
    if (modelReady && phase === "preparing") setPhase("chat");
  }, [modelReady, phase]);

  // Cancel any in-flight download when the component unmounts (last-resort
  // safety net — the wrapped handleUseLegacy below is the primary path).
  useEffect(() => {
    return () => {
      const jid = activeJobIdRef.current;
      if (jid) api.mlxDownloadCancel(jid).catch(() => undefined);
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  // -------------------------------------------------------------------------
  // Setup model helpers
  // -------------------------------------------------------------------------

  // Cancel any active setup-model download and clear the poll interval.
  // Called before navigating away and on unmount.
  const cancelActiveDownload = useCallback(async () => {
    const jid = activeJobIdRef.current;
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    if (jid) {
      activeJobIdRef.current = null;
      setDownloadJobId(null);
      setDownloadProgress(null);
      try {
        await api.mlxDownloadCancel(jid);
      } catch {
        // Best-effort — the backend will clean up on its own eventually
      }
    }
  }, []);

  // Wrapped navigation: always cancel any in-flight download first so we
  // don't waste bandwidth when the user switches to the manual wizard.
  const handleUseLegacy = useCallback(async () => {
    await cancelActiveDownload();
    onUseLegacy();
  }, [cancelActiveDownload, onUseLegacy]);

  const handleSkip = useCallback(async () => {
    try { await api.setupSkip(); } catch { /* best-effort */ }
    onSkip();
  }, [onSkip]);

  // Called when the user clicks "Let Otto guide me" from the picker.
  // 1. Returns immediately if the model is already cached.
  // 2. Checks whether a download is already running for this model and
  //    re-attaches to it rather than spawning a duplicate job.
  // 3. Only starts a fresh download if nothing is in flight.
  const handleAutoSetup = useCallback(async () => {
    // Already cached — skip straight to chat
    try {
      const { models } = await api.mlxLocalModels();
      if (models.some((m) => m.repo_id === SETUP_MODEL_ID)) {
        setModelReady(true);
        setPhase("chat");
        return;
      }
    } catch {
      // Cache check failed — continue to download path
    }

    setPhase("preparing");

    // Re-attach to any already-running job for this model (covers the case
    // where the user returns to auto-setup after a partial cancel that didn't
    // fully stop, or navigated away and back within the same backend session).
    try {
      const { jobs } = await api.mlxDownloadList();
      const existing = jobs.find(
        (j) =>
          j.repo_id === SETUP_MODEL_ID &&
          (j.status === "pending" || j.status === "running"),
      );
      if (existing) {
        setDownloadJobId(existing.job_id);
        return; // polling effect will take it from here
      }
    } catch {
      // Can't list jobs — fall through to fresh download
    }

    // Start a fresh download
    try {
      const job = await api.mlxDownload({ repo_id: SETUP_MODEL_ID });
      setDownloadJobId(job.job_id);
    } catch {
      // Download failed to start — templates handle replies fine
      setPhase("chat");
    }
  }, []);

  // Kicked once from handleSend to kick off any remaining download in the
  // background (guards against duplicate jobs via a ref).
  const downloadRequestedRef = useRef(false);

  const startModelDownloadIfNeeded = useCallback(async () => {
    if (downloadRequestedRef.current || modelReady) return;
    downloadRequestedRef.current = true;
    try {
      const { models } = await api.mlxLocalModels();
      if (models.some((m) => m.repo_id === SETUP_MODEL_ID)) {
        setModelReady(true);
        return;
      }
      const job = await api.mlxDownload({ repo_id: SETUP_MODEL_ID });
      setDownloadJobId(job.job_id);
    } catch {
      downloadRequestedRef.current = false;
    }
  }, [modelReady]);

  // Poll download job until complete or failed
  useEffect(() => {
    if (!downloadJobId) return;
    pollRef.current = setInterval(async () => {
      try {
        const status = await api.mlxDownloadStatus(downloadJobId);
        if (status.bytes_total && status.bytes_done) {
          setDownloadProgress(
            Math.round((status.bytes_done / status.bytes_total) * 100),
          );
        }
        if (status.status === "done") {
          clearInterval(pollRef.current!);
          setModelReady(true);
          setDownloadProgress(null);
        } else if (status.status === "error" || status.status === "cancelled") {
          clearInterval(pollRef.current!);
          setDownloadProgress(null);
          // Fall through to chat — templates handle replies without the model
          setPhase("chat");
        }
      } catch {
        clearInterval(pollRef.current!);
        // Network error — fall through to chat with template replies
        setPhase("chat");
      }
    }, 2000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [downloadJobId]);

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------

  // -------------------------------------------------------------------------
  // Message list helpers
  // -------------------------------------------------------------------------

  function pushUser(content: string) {
    // Clear any pending interactive widgets when the user sends a message
    setMessages((prev) => [
      ...prev.map((m) =>
        m.quickReplies || m.modelPicker
          ? { ...m, quickReplies: undefined, modelPicker: undefined }
          : m,
      ),
      { role: "user" as const, content },
    ]);
  }

  function pushAssistant(
    content: string,
    quickReplies?: string[],
    modelPicker?: Message["modelPicker"],
    inlineInput?: Message["inlineInput"],
    richModelPicker?: Message["richModelPicker"],
  ) {
    setMessages((prev) => [
      ...prev,
      { role: "assistant" as const, content, quickReplies, modelPicker, inlineInput, richModelPicker },
    ]);
  }

  function pushAssistantLoading() {
    setMessages((prev) => [...prev, { role: "assistant" as const, content: "", isLoading: true }]);
  }

  function resolveLastAssistant(
    content: string,
    quickReplies?: string[],
    modelPicker?: Message["modelPicker"],
    inlineInput?: Message["inlineInput"],
    richModelPicker?: Message["richModelPicker"],
  ) {
    setMessages((prev) =>
      prev.map((m, i) =>
        i === prev.length - 1 && m.isLoading
          ? { role: "assistant" as const, content, quickReplies, modelPicker, inlineInput, richModelPicker }
          : m,
      ),
    );
  }

  /** Clear the inlineInput from the last assistant message (once submitted). */
  function clearLastInlineInput() {
    setMessages((prev) =>
      prev.map((m, i) =>
        i === prev.length - 1 ? { ...m, inlineInput: undefined } : m,
      ),
    );
  }

  /** Save current conversational state so the user can step back to it. */
  function pushHistory() {
    setHistoryStack(prev => [...prev, { messages, step, providerChoice, cloudSub, inputType, inlineInputValues }]);
  }

  /** Restore the most recent snapshot, undoing the last exchange. */
  function handleBack() {
    if (isBusy || historyStack.length === 0) return;
    const entry = historyStack[historyStack.length - 1];
    setHistoryStack(h => h.slice(0, -1));
    setMessages(entry.messages);
    setStep(entry.step);
    setProviderChoice(entry.providerChoice);
    setCloudSub(entry.cloudSub);
    setInputType(entry.inputType);
    setInlineInputValues(entry.inlineInputValues);
    setInput("");
  }

  /** Replace the content of the last assistant message in-place (for live progress). */
  function updateLastAssistant(content: string, commandStatus?: "done" | "error") {
    setMessages((prev) =>
      prev.map((m, i) =>
        i === prev.length - 1 && m.role === "assistant"
          ? {
              ...m,
              content,
              isLoading: false,
              ...(m.commandOutput && commandStatus != null
                ? { commandOutput: { ...m.commandOutput, status: commandStatus } }
                : {}),
            }
          : m,
      ),
    );
  }

  /**
   * Append a streaming log line to the last assistant message's commandOutput
   * panel, setting the label text as the main message content.
   */
  function updateLastAssistantCommand(label: string, newLine: string) {
    setMessages((prev) =>
      prev.map((m, i) =>
        i === prev.length - 1 && m.role === "assistant"
          ? {
              ...m,
              content: label,
              isLoading: false,
              commandOutput: {
                lines: [...(m.commandOutput?.lines ?? []), newLine],
                status: "running" as const,
              },
            }
          : m,
      ),
    );
  }

  // Handle a quick-reply chip click — same as typing and sending that text
  async function handleQuickReply(reply: string) {
    if (isBusy || step === "done") return;
    setInput("");
    pushUser(reply);

    // Global escape hatches — always available regardless of current step
    if (/^set up manually$/i.test(reply.trim())) {
      await handleUseLegacy();
      return;
    }

    pushHistory();
    setIsBusy(true);
    void startModelDownloadIfNeeded();
    try {
      switch (step) {
        case "provider":            await handleProvider(reply);           break;
        case "cloud_sub":           await handleCloudSub(reply);          break;
        case "cloud_key":           await handleCloudKey(reply);          break;
        case "cloud_model":         await handleCloudModel(reply);        break;
        case "local_model":         await handleLocalModel(reply);        break;
        case "azure_endpoint":      await handleAzureEndpoint(reply);     break;
        case "azure_key":           await handleAzureKey(reply);          break;
        case "azure_deployment":    await handleAzureDeployment(reply);   break;
        case "azure_api_version":   await handleAzureApiVersion(reply);   break;
        case "memory":              await handleMemory(reply);            break;
        case "memory_inject":       await handleMemoryInject(reply);      break;
        case "activity":            await handleActivity(reply);          break;
        case "ambient":             await handleAmbient(reply);           break;
        case "evaluation":          await handleEvaluation(reply);        break;
      }
    } finally {
      setIsBusy(false);
    }
  }

  // -------------------------------------------------------------------------
  // Settings deep-merge helper (mirrors SetupWizard's patchSettings)
  //
  // PUT /api/settings does AppConfig.model_validate(payload) — NOT a merge.
  // Sending a partial object resets all omitted fields to Pydantic defaults.
  // We must always send the full merged object.
  // -------------------------------------------------------------------------

  const patchSettings = useCallback(
    async (patch: Record<string, unknown>) => {
      const base = settings ?? ({} as AppSettings);
      // Deep-merge one level for nested sub-objects, mirroring what SetupWizard does.
      // The backend does model_validate(payload) — not a merge — so we must always
      // send the full object or omitted fields reset to Pydantic defaults.
      const pllm = (patch.llm ?? {}) as Partial<AppSettings['llm']>;
      const next: AppSettings = {
        ...base,
        ...(patch as Partial<AppSettings>),
        llm: {
          ...base.llm,
          ...pllm,
          mlx:      { ...(base.llm?.mlx      ?? {}), ...(pllm.mlx      ?? {}) },
          anthropic:{ ...(base.llm?.anthropic ?? {}), ...(pllm.anthropic ?? {}) },
          openai:   { ...(base.llm?.openai   ?? {}), ...(pllm.openai   ?? {}) },
        },
        memory:   { ...(base.memory   ?? {}), ...((patch.memory   ?? {}) as Partial<AppSettings['memory']>) },
        activity: { ...(base.activity ?? {}), ...((patch.activity ?? {}) as Partial<AppSettings['activity']>) },
        omlx:     { ...(base.omlx     ?? {}), ...((patch.omlx     ?? {}) as Partial<AppSettings['omlx']>) },
        ambient:  { ...(base.ambient  ?? {}), ...((patch.ambient  ?? {}) as Partial<AppSettings['ambient']>) },
        evaluation: { ...(base.evaluation ?? {}), ...((patch.evaluation ?? {}) as Partial<AppSettings['evaluation']>) },
      } as AppSettings;
      setSettings(next);
      await api.updateSettings(next as unknown as Record<string, unknown>);
    },
    [settings],
  );

  // -------------------------------------------------------------------------
  // Backend reply generation (pure text; no config side-effects)
  // -------------------------------------------------------------------------

  async function fetchReply(
    currentStep: SetupStep,
    userMessage: string,
    extracted: string | null,
  ): Promise<string> {
    try {
      const { reply } = await api.setupChat({
        step: currentStep,
        user_message: userMessage,
        extracted,
        context: { chip: capabilities.chip, ram_gb: capabilities.ram_gb },
        model_ready: modelReady,
      });
      return reply;
    } catch {
      return "Got it — let's continue.";
    }
  }

  // -------------------------------------------------------------------------
  // Step handlers — each applies config via existing APIs then fetches reply
  // -------------------------------------------------------------------------

  // ---------------------------------------------------------------------------
  // oMLX helpers
  // ---------------------------------------------------------------------------

  async function awaitOmlxJob(
    jobId: string,
    maxSeconds = 180,
    onProgress?: (latestLine: string) => void,
  ): Promise<"done" | "error"> {
    const deadline = Date.now() + maxSeconds * 1000;
    let lastLineCount = 0;
    while (Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 2000));
      try {
        const j = await api.getOmlxJob(jobId);
        // Fire the callback when new log lines arrive
        if (onProgress && j.log_lines && j.log_lines.length > lastLineCount) {
          // Find the last non-empty, human-readable line (skip blank lines)
          const newLines = j.log_lines.slice(lastLineCount);
          lastLineCount = j.log_lines.length;
          const readable = [...newLines].reverse().find((l) => l.trim().length > 0);
          if (readable) onProgress(readable.trim());
        }
        if (j.status === "done") return "done";
        if (j.status === "error") return "error";
      } catch {
        // transient — keep polling
      }
    }
    return "error";
  }

  /** Show the oMLX model picker — sets step to "local_model" so handleLocalModel picks it up. */
  async function showOmlxModelPicker(preamble: string) {
    setStep("local_model");
    pushAssistant(
      preamble,
      undefined,
      undefined,
      undefined,
      { type: "omlx", allowSkip: true },
    );
  }

  async function handleOmlxSetup() {
    // ── 1. Probe current state ──────────────────────────────────────────────
    let info: OmlxInfo | null = null;
    let status: OmlxStatus | null = null;
    try {
      [info, status] = await Promise.all([api.omlxInfo(), api.omlxStatus()]);
    } catch {
      resolveLastAssistant(
        "I couldn't reach the oMLX service — there may be a backend issue. " +
          "Falling back to MLX (built-in). You can switch to oMLX later in Settings.",
      );
      await fallbackToMlx();
      return;
    }

    const installed = !!info?.detection.installed && !!info.detection.cli_path;
    const homebrew   = !!info?.detection.homebrew;
    // Only trust reachable=true if oMLX is actually installed — otherwise it's
    // a false positive from another service (e.g. a dev server) on the same port.
    const reachable  = !!status?.reachable && installed;

    // ── 2. Install (if needed) ──────────────────────────────────────────────
    if (!reachable && !installed) {
      resolveLastAssistant(
        homebrew
          ? "oMLX isn't installed yet — installing via Homebrew (1–2 minutes)…"
          : "oMLX isn't installed yet — downloading it from the official release (no Homebrew required, 1–2 minutes)…",
      );
      try {
        const installJob = await api.omlxInstall();
        const result = await awaitOmlxJob(installJob.id, 300, (line) =>
          updateLastAssistantCommand("Installing oMLX…", line),
        );
        if (result === "error") {
          pushAssistant("Installation failed. Check Settings → LLM → oMLX for logs. Falling back to MLX.");
          await fallbackToMlx();
          return;
        }
        pushAssistant("✓ oMLX installed.");
      } catch {
        pushAssistant("Couldn't install oMLX. Falling back to MLX — switch back from Settings anytime.");
        await fallbackToMlx();
        return;
      }
    } else if (!reachable && installed) {
      resolveLastAssistant("oMLX is installed but not running yet.");
    } else {
      // Already reachable — show the URL so the user can verify
      resolveLastAssistant(
        `oMLX is already running — you can verify with:\n\ncurl http://127.0.0.1:52414/v1/models`,
      );
    }

    // ── 3. Start server (if not already reachable) ─────────────────────────
    if (!reachable) {
      pushAssistant("Starting the oMLX server…");
      try {
        const startJob = await api.omlxStart();
        const result = await awaitOmlxJob(startJob.id, 60, (line) =>
          updateLastAssistantCommand("Starting oMLX server…", line),
        );
        if (result === "error") {
          pushAssistant("Server failed to start. You can retry in Settings → LLM → oMLX. Falling back to MLX for now.");
          await fallbackToMlx();
          return;
        }
        pushAssistant("✓ oMLX server is running.");
      } catch {
        pushAssistant("Couldn't start the server. Falling back to MLX.");
        await fallbackToMlx();
        return;
      }
    }

    // ── 4. Verify server is still reachable, then show model picker ────────
    // Re-probe so a crash between the initial check and here is caught early.
    try {
      const confirm = await api.omlxStatus();
      if (!confirm.reachable) {
        pushAssistant("The oMLX server stopped unexpectedly. Attempting to restart…");
        const restartJob = await api.omlxStart();
        const restartResult = await awaitOmlxJob(restartJob.id, 60);
        if (restartResult === "error") {
          pushAssistant("Restart failed. You can retry from Settings → LLM → oMLX. Falling back to MLX for now.");
          await fallbackToMlx();
          return;
        }
        pushAssistant("✓ oMLX server restarted.");
      }
    } catch {
      // If the re-probe itself fails, try to proceed — omlxLoadModel will catch real errors
    }
    await patchSettings({ llm: { provider: "omlx" }, omlx: { enabled: true } }).catch(() => undefined);
    await showOmlxModelPicker("Which model would you like to load into oMLX?");
  }

  /** Switch provider to MLX and show the model picker as a graceful fallback. */
  async function fallbackToMlx() {
    await patchSettings({ llm: { provider: "mlx" } }).catch(() => undefined);
    setProviderChoice("mlx");
    setStep("local_model");
    pushAssistant(
      `Falling back to built-in MLX. Browse the model catalog below and pick a model for your ${capabilities.chip}:`,
      undefined,
      undefined,
      undefined,
      { type: "mlx", allowSkip: true },
    );
  }

  async function handleProvider(userMessage: string) {
    const extracted = extractProvider(userMessage);
    pushAssistantLoading();

    if (!extracted) {
      resolveLastAssistant(await fetchReply("provider", userMessage, null));
      return;
    }

    setProviderChoice(extracted);

    // Cluster — run models across one or more Macs. Hands off to the shared
    // ClusterSetupFlow, mounted inline below, which owns runtime start, adding
    // Macs, and model selection.
    if (extracted === "exo") {
      try {
        await patchSettings({
          exo: { ...(settings?.exo ?? {}), enabled: true },
          llm: { provider: "exo" },
        });
      } catch {
        /* non-fatal — the flow re-applies this on mount */
      }
      resolveLastAssistant(
        "A cluster runs models across one or more Macs. Let's set it up — " +
          "I'll start it on this Mac, then you can add more.",
      );
      setMessages((prev) => [
        ...prev,
        { role: "assistant" as const, content: "", clusterSetup: true },
      ]);
      return;
    }

    // Azure / Bedrock — multi-field credential, redirect to wizard
    if (extracted === "azure" || extracted === "bedrock") {
      resolveLastAssistant(
        `${extracted === "azure" ? "Azure OpenAI" : "AWS Bedrock"} needs several credential fields — I'll open the step-by-step wizard for you.`,
      );
      await new Promise((r) => setTimeout(r, 1200));
      await handleUseLegacy();
      return;
    }

    // oMLX — handle fully in-chat
    if (extracted === "omlx") {
      await handleOmlxSetup();
      return;
    }

    // MLX / Anthropic / OpenAI — save provider immediately so the rest of setup builds on it.
    // "cloud" is a transient UI choice meaning "some cloud provider, TBD" — don't persist it.
    if (extracted !== "cloud") {
      try {
        await patchSettings({ llm: { provider: extracted } });
      } catch {
        // Non-fatal
      }
    }

    resolveLastAssistant(
      extracted === "mlx"
        ? `Got it — running locally on your ${capabilities.chip}.`
        : extracted === "openai"
        ? "Great — OpenAI it is."
        : extracted === "anthropic"
        ? "Great — Anthropic Claude it is."
        : "Great — a cloud provider it is.",
    );

    if (extracted === "mlx") {
      setStep("local_model");
      pushAssistant(
        `Your ${capabilities.chip} has ${capabilities.ram_gb} GB RAM. ` +
          `Browse the full model catalog below and pick a model — or skip to choose later in Settings.`,
        undefined,
        undefined,
        undefined,
        { type: "mlx", allowSkip: true },
      );
    } else {
      setStep("cloud_sub");
      pushAssistant(
        "Which cloud service would you like to use?",
        ["Anthropic (direct)", "OpenAI (direct)", "AWS Bedrock", "Azure OpenAI"],
      );
    }
  }

  async function handleCloudSub(userMessage: string) {
    const extracted = extractCloudSub(userMessage);
    pushAssistantLoading();

    if (extracted == null) {
      resolveLastAssistant(
        "I didn't catch that — which service would you like?",
        ["Anthropic (direct)", "OpenAI (direct)", "AWS Bedrock", "Azure OpenAI"],
      );
      return;
    }

    // Azure OpenAI — handled inline
    if (extracted === "azure") {
      setProviderChoice("azure");
      resolveLastAssistant(
        "Enter your Azure OpenAI endpoint URL:",
        undefined, undefined,
        { type: "text", placeholder: "https://your-resource.openai.azure.com" },
      );
      setStep("azure_endpoint");
      return;
    }

    // Bedrock — handled inline
    if (extracted === "bedrock") {
      setProviderChoice("bedrock");
      resolveLastAssistant(
        "Enter your AWS Access Key ID:",
        undefined, undefined,
        { type: "text", placeholder: "AKIA…" },
      );
      setStep("bedrock_key_id");
      return;
    }

  setCloudSub(extracted as CloudSub);
  resolveLastAssistant(
    extracted === "anthropic"
      ? "Please paste your Anthropic API key. It's stored only on this Mac."
      : "Please paste your OpenAI API key. It's stored only on this Mac.",
    undefined,
    undefined,
    { type: "password", placeholder: extracted === "openai" ? "sk-…" : "sk-ant-…" },
  );
  setStep("cloud_key");
  }

  async function handleCloudKey(userMessage: string) {
    const provider = cloudSub ?? (providerChoice as CloudSub);
    if (!provider) {
      resolveLastAssistant("Something went wrong — let me open the manual setup.");
      await handleUseLegacy();
      return;
    }

    const apiKey = userMessage.trim();
    pushAssistantLoading();

    // Validate the key first
    let success = false;
    try {
      const result = await api.testConnection({
        provider,
        api_key: apiKey,
        model_provider: provider,
        openai_model_provider: provider === "openai" ? "openai" : undefined,
      });
      success = result.success;
    } catch {
      success = false;
    }

    if (!success) {
      resolveLastAssistant(
        "That key didn't work — the connection test failed. Please double-check it and try again.",
        undefined,
        undefined,
        { type: "password", placeholder: provider === "openai" ? "sk-…" : "sk-ant-…" },
      );
      return;
    }

    // Key works — persist it alongside the provider
    try {
      await patchSettings({
        llm: {
          provider,
          [provider]: {
            model_provider: provider,
            api_key: apiKey,
          },
        },
      });
    } catch {
      // Non-fatal — settings saved on best-effort basis
    }

    // Fetch live model list (best-effort; fall back to curated list)
    let modelList = CURATED_CLOUD_MODELS[provider as "anthropic" | "openai"] ?? [];
    try {
      const result = await api.listModels({
        provider,
        api_key: apiKey,
        model_provider: provider,
      });
      if (result.models?.length) {
        modelList = result.models.map((m: { id: string; name: string }) => ({
          id: m.id,
          label: m.name || m.id,
        }));
      }
    } catch {
      // Use curated list
    }
    setCloudModelList(modelList);

    resolveLastAssistant(
      "Connection verified! Pick a model:",
      undefined,
      {
        options: modelList.map((m) => ({ id: m.id, label: m.label, sizeGb: "" })),
        defaultIdx: 0,
        allowSkip: true,
      },
    );
    setStep("cloud_model");
  }

  async function handleCloudModel(userMessage: string) {
    const provider = cloudSub ?? (providerChoice as CloudSub);
    if (!provider) {
      advanceToMemory();
      return;
    }
    pushAssistantLoading();

    // Match chip label to model id
    const modelList =
      cloudModelList.length > 0
        ? cloudModelList
        : (CURATED_CLOUD_MODELS[provider as "anthropic" | "openai"] ?? []);

    const matched = modelList.find(
      (m) =>
        userMessage.toLowerCase().includes(m.label.toLowerCase()) ||
        userMessage.toLowerCase().includes(m.id.toLowerCase()),
    );

    // Fallback: if user typed a raw model ID-like string use it directly
    const modelId =
      matched?.id ??
      (userMessage.trim().match(/^[a-z0-9._-]+[:/][a-z0-9._-]+/i)
        ? userMessage.trim()
        : modelList[0]?.id ?? "");

    if (!modelId) {
      resolveLastAssistant(
        "I didn't recognise that model — which model would you like?",
        [...modelList.slice(0, 5).map((m) => m.label), "Other…"],
      );
      return;
    }

    try {
      if (providerChoice === "azure") {
        await patchSettings({
          llm: {
            provider: "openai",
            openai: {
              model_provider: "azure",
              azure_api_key: azureApiKey,
              model_name: modelId || azureDeployment,
              azure_endpoint: azureEndpoint,
              azure_deployment: azureDeployment,
              azure_api_version: azureApiVersion,
            },
          },
        });
      } else if (providerChoice === "bedrock") {
        await patchSettings({
          llm: {
            provider: "anthropic",
            anthropic: {
              model_provider: "bedrock",
              model_name: modelId,
              bedrock_region: bedrockRegion,
              bedrock_auth_mode: "keys",
              aws_access_key_id: bedrockKeyId,
              aws_secret_access_key: bedrockSecret,
            },
          },
        });
      } else {
        await patchSettings({
          llm: {
            provider,
            [provider]: {
              model_provider: provider,
              model_name: modelId,
            },
          },
        });
      }
    } catch {
      // Non-fatal
    }

    resolveLastAssistant(`Got it — I'll use **${modelId}** as your model.`);
    advanceToMemory();
  }

  async function handleLocalModel(userMessage: string) {
    pushAssistantLoading();

    const m = userMessage.toLowerCase().trim();
    const options = mlxModelOptions(capabilities.ram_gb);
    const rec = recommendedMainModel(capabilities.ram_gb);

    // Resolve the model ID from the user message (chip label, HF repo ID, or free text)
    let modelId = rec.id;
    const chipMatch = options.find((o) =>
      userMessage.toLowerCase().includes(o.label.toLowerCase().split(" ")[0].toLowerCase()),
    );
    if (chipMatch) {
      modelId = chipMatch.id;
    } else if (userMessage.includes("/")) {
      // Raw HF repo ID
      modelId = userMessage.trim();
    } else if (/\b(yes|y|sure|ok(ay)?|recommended|default|sounds good|use recommended)\b/.test(m)) {
      modelId = rec.id;
    }
    // else: unrecognised free text → fall back to recommended

    const shortName = modelId.split("/")[1] ?? modelId;

    // ── EXO path: save model_name so EXO uses it on next inference ────────
    if (providerChoice === "exo") {
      try {
        await patchSettings({ llm: { provider: "exo" }, exo: { enabled: true, model_name: modelId } });
      } catch { /* non-fatal */ }
      resolveLastAssistant(
        `Got it — **${shortName}** will be used when you start a cluster inference session. ` +
          "You can add remote nodes from Settings → LLM → Cluster.",
      );
      advanceToMemory();
      return;
    }

    // ── oMLX path: download then load into the running server ──────────────
    if (providerChoice === "omlx") {
      resolveLastAssistant(`Downloading **${shortName}** and loading it into oMLX — this may take a few minutes…`);
      try {
        await patchSettings({ llm: { provider: "omlx" }, omlx: { enabled: true, model_name: modelId } });
      } catch { /* non-fatal */ }
      try {
        await api.mlxDownload({ repo_id: modelId });
        const loadJob = await api.omlxLoadModel(modelId);
        const result = await awaitOmlxJob(loadJob.id, 300, (line) =>
          updateLastAssistantCommand(`Loading **${shortName}** into oMLX…`, line),
        );
        if (result === "done") {
          pushAssistant(`✓ **${shortName}** loaded into oMLX. You're all set!`);
        } else {
          pushAssistant(`The load is taking longer than expected — you can track it in Settings → LLM → oMLX.`);
        }
      } catch {
        pushAssistant(`Couldn't load the model automatically — you can do that from Settings → LLM → oMLX.`);
      }
      advanceToMemory();
      return;
    }

    // ── MLX / EXO fallback path ─────────────────────────────────────────────
    try {
      await patchSettings({
        llm: { provider: "mlx", mlx: { hf_llm_model_id: modelId } },
      });
    } catch { /* non-fatal */ }

    try {
      await api.mlxDownload({ repo_id: modelId });
    } catch {
      // May already be cached — ignore
    }

    resolveLastAssistant(
      `Got it — I'll use **${shortName}** and download it in the background.`,
    );
    advanceToMemory();
  }

  // -------------------------------------------------------------------------
  // Azure OpenAI inline credential flow
  // -------------------------------------------------------------------------

  async function handleAzureEndpoint(value: string) {
    const endpoint = value.trim();
    setAzureEndpoint(endpoint);
    pushAssistant(
      "Your Azure OpenAI API key:",
      undefined, undefined,
      { type: "password", placeholder: "Azure OpenAI key…" },
    );
    setStep("azure_key");
  }

  async function handleAzureKey(value: string) {
    setAzureApiKey(value.trim());
    pushAssistant(
      "Deployment name (the name you gave this deployment in Azure):",
      undefined, undefined,
      { type: "text", placeholder: "gpt-4o" },
    );
    setStep("azure_deployment");
  }

  async function handleAzureDeployment(value: string) {
    setAzureDeployment(value.trim());
    pushAssistant(
      "API version:",
      undefined, undefined,
      { type: "text", placeholder: "2024-12-01-preview", prefill: "2024-12-01-preview" },
    );
    setStep("azure_api_version");
  }

  async function handleAzureApiVersion(value: string) {
    const apiVersion = value.trim() || "2024-12-01-preview";
    setAzureApiVersion(apiVersion);
    pushAssistantLoading();

    // Test connection
    let testOk = false;
    let testMsg = "";
    try {
      const r = await api.testConnection({
        provider: "openai",
        api_key: azureApiKey,
        model_name: azureDeployment || "gpt-4o",
        openai_model_provider: "azure",
        azure_endpoint: azureEndpoint,
        azure_api_version: apiVersion,
        azure_deployment: azureDeployment,
      });
      testOk = r.success;
      testMsg = r.message ?? "";
    } catch (e) {
      testMsg = e instanceof Error ? e.message : "Connection failed";
    }

    if (!testOk) {
      resolveLastAssistant(
        `Connection failed: ${testMsg}. Double-check your endpoint, key, and deployment name.`,
        ["Try again", "Set up manually"],
      );
      // "Try again" will re-enter at azure_endpoint; "Set up manually" handled by extractProvider returning null → fetchReply
      setStep("azure_endpoint");
      return;
    }

    // Persist credentials now; model name saved when user picks in cloud_model step
    try {
      await patchSettings({
        llm: {
          provider: "openai",
          openai: {
            model_provider: "azure",
            azure_api_key: azureApiKey,
            azure_endpoint: azureEndpoint,
            azure_deployment: azureDeployment,
            azure_api_version: apiVersion,
          },
        },
      });
    } catch { /* non-fatal */ }

    // Model options — static list (Azure doesn't expose a list endpoint)
    const modelOptions = [
      { id: "gpt-5",        label: "GPT-5",           sizeGb: "" },
      { id: "gpt-4o",       label: "GPT-4o",          sizeGb: "" },
      { id: "gpt-4o-mini",  label: "GPT-4o mini",     sizeGb: "" },
      { id: "gpt-4.1",      label: "GPT-4.1",         sizeGb: "" },
      { id: "gpt-4.1-mini", label: "GPT-4.1 mini",    sizeGb: "" },
      { id: "gpt-4.1-nano", label: "GPT-4.1 nano",    sizeGb: "" },
      { id: "o3",           label: "o3",               sizeGb: "" },
      { id: "o4-mini",      label: "o4 mini",          sizeGb: "" },
    ];
    setCloudModelList(modelOptions.map((m) => ({ id: m.id, label: m.label })));

    resolveLastAssistant(
      "Connected! Which base model is your deployment serving?",
      undefined,
      { options: modelOptions, defaultIdx: 0, allowSkip: true },
    );
    setStep("cloud_model");
  }

  // -------------------------------------------------------------------------
  // Bedrock inline credential flow
  // -------------------------------------------------------------------------

  async function handleBedrockKeyId(value: string) {
    setBedrockKeyId(value.trim());
    pushAssistant(
      "And your AWS Secret Access Key:",
      undefined, undefined,
      { type: "password", placeholder: "••••••••" },
    );
    setStep("bedrock_secret");
  }

  async function handleBedrockSecret(value: string) {
    setBedrockSecret(value.trim());
    pushAssistant(
      "Which AWS region?",
      undefined, undefined,
      { type: "text", placeholder: "us-east-1", prefill: "us-east-1" },
    );
    setStep("bedrock_region");
  }

  async function handleBedrockRegion(value: string) {
    const region = value.trim() || "us-east-1";
    setBedrockRegion(region);
    pushAssistantLoading();

    // Test connection
    let testOk = false;
    let testMsg = "";
    try {
      const r = await api.testConnection({
        provider: "anthropic",
        api_key: "",
        model_provider: "bedrock",
        bedrock_region: region,
        bedrock_auth_mode: "keys",
        aws_access_key_id: bedrockKeyId,
        aws_secret_access_key: bedrockSecret,
      });
      testOk = r.success;
      testMsg = r.message ?? "";
    } catch (e) {
      testMsg = e instanceof Error ? e.message : "Connection failed";
    }

    if (!testOk) {
      resolveLastAssistant(
        `Connection failed: ${testMsg}. Check your credentials and try again.`,
        ["Try again", "Set up manually"],
      );
      setStep("bedrock_key_id");
      return;
    }

    // Fetch inference profiles
    let modelOptions: { id: string; label: string; sizeGb: string }[] = [];
    try {
      const list = await api.listModels({
        provider: "anthropic",
        api_key: "",
        model_provider: "bedrock",
        bedrock_region: region,
        bedrock_auth_mode: "keys",
        aws_access_key_id: bedrockKeyId,
        aws_secret_access_key: bedrockSecret,
      });
      if (list.models?.length) {
        setCloudModelList(list.models.map((m) => ({ id: m.id, label: m.name || m.id })));
        modelOptions = list.models.map((m) => ({ id: m.id, label: m.name || m.id, sizeGb: "" }));
      }
    } catch { /* use fallback */ }

    if (modelOptions.length === 0) {
      modelOptions = [
        { id: "us.anthropic.claude-sonnet-4-20250514-v1:0", label: "Claude Sonnet 4 (US)", sizeGb: "" },
        { id: "us.anthropic.claude-3-7-sonnet-20250219-v1:0", label: "Claude 3.7 Sonnet (US)", sizeGb: "" },
        { id: "us.anthropic.claude-3-5-sonnet-20241022-v2:0", label: "Claude 3.5 Sonnet v2 (US)", sizeGb: "" },
        { id: "us.anthropic.claude-3-5-haiku-20241022-v1:0", label: "Claude 3.5 Haiku (US)", sizeGb: "" },
      ];
    }

    // Persist credentials (model name saved when user picks in cloud_model step)
    try {
      await patchSettings({
        llm: {
          provider: "anthropic",
          anthropic: {
            model_provider: "bedrock",
            bedrock_region: region,
            bedrock_auth_mode: "keys",
            aws_access_key_id: bedrockKeyId,
            aws_secret_access_key: bedrockSecret,
          },
        },
      });
} catch { /* non-fatal */ }

resolveLastAssistant(
      "Connected! Pick an inference profile:",
      undefined,
      { options: modelOptions, defaultIdx: 0, allowSkip: true },
    );
    setStep("cloud_model");
  }

  /** Remove the embedded Cluster setup flow from the thread once it's done. */
  function clearClusterSetup() {
    setMessages((prev) =>
      prev.map((m) => (m.clusterSetup ? { ...m, clusterSetup: false } : m)),
    );
  }

  /** Called when ClusterSetupFlow finishes — continue to the next setup step. */
  function finishClusterSetup() {
    clearClusterSetup();
    advanceToMemory();
  }

  /** Called when the user backs out of the Cluster flow — re-offer providers. */
  function backFromClusterSetup() {
    clearClusterSetup();
    setProviderChoice(null);
    setStep("provider");
    setMessages((prev) => [
      ...prev,
      {
        role: "assistant" as const,
        content: "No problem — which would you like instead?",
        quickReplies: OPENING_REPLIES,
        quickReplyTooltips: OPENING_TOOLTIPS,
      },
    ]);
  }

  function advanceToMemory() {
    setStep("memory");
    pushAssistant(
      "Would you like Otto to remember things across conversations — your preferences, past context, and topics?",
      ["Yes, enable memory", "No thanks"],
    );
  }

  async function handleMemory(userMessage: string) {
    const extracted = extractBool(userMessage);
    pushAssistantLoading();

    if (extracted === null) {
      resolveLastAssistant(
        "Just to confirm — would you like Otto to remember things across conversations?",
        ["Yes, enable memory", "No thanks"],
      );
      return;
    }

    const isLocal = providerChoice === "mlx" || providerChoice === "omlx" || providerChoice === "exo";
    try {
      await patchSettings({
        memory: {
          enabled: extracted,
          // Injection is off by default; user chooses in the next step.
          inject_enabled: false,
          inject_on_session_start: false,
          inject_realtime: false,
          llm_family: isLocal ? "mlx" : "follow_main",
        } as AppSettings["memory"],
      });
    } catch {
      // Non-fatal
    }

    if (!extracted) {
      resolveLastAssistant("No problem — memory is off. You can enable it anytime in Settings.");
      setStep("activity");
      pushAssistant(
        "Should Otto track your Mac activity in the background to give better context? This requires macOS Accessibility permission.",
        ["Yes, enable activity tracking", "No thanks"],
      );
      return;
    }

    resolveLastAssistant(
      "Memory enabled — Otto will summarise each session into notes for future reference.",
    );

    // Check embedding model; download inline if missing
    try {
      const modelStatus = await api.getEmbeddingModelStatus();
      if (!modelStatus.installed) {
        pushAssistantLoading();
        try {
          await api.startModelDownload();
        } catch {
          // If start fails the poll will catch the error state
        }

        await new Promise<void>((resolve) => {
          const embPollRef = setInterval(async () => {
            try {
              const s = await api.getEmbeddingModelStatus();

              if (s.installed) {
                clearInterval(embPollRef);
                updateLastAssistant("✓ Embedding model ready — semantic memory search is active.", "done");
                resolve();
                return;
              }

              if (s.error && !s.downloading) {
                clearInterval(embPollRef);
                updateLastAssistant(
                  `Could not download the embedding model: ${s.error}\n` +
                    "You can retry from **Settings → Memory → Search Index**.",
                  "error",
                );
                resolve();
                return;
              }

              const pct =
                s.total_bytes > 0
                  ? Math.min(100, Math.round((s.bytes_downloaded / s.total_bytes) * 100))
                  : null;
              const mbDone = (s.bytes_downloaded / 1048576).toFixed(0);
              const mbTotal =
                s.total_bytes > 0 ? ` / ${(s.total_bytes / 1048576).toFixed(0)} MB` : " MB";
              const pctLabel = pct !== null ? ` ${pct}%` : "";
              updateLastAssistantCommand(
                "Downloading semantic search model (all-MiniLM-L6-v2, ~90 MB)…",
                `${mbDone}${mbTotal}${pctLabel}`,
              );
            } catch {
              // transient — keep polling
            }
          }, 1500);
        });
      }
    } catch {
      // Embedding check is best-effort; continue setup regardless
    }

    setStep("memory_inject");
    pushAssistant(
      "How would you like memory to be injected into conversations?\n\n" +
        "• **At session start** — MEMORY.md is loaded into the system prompt once when a session begins\n" +
        "• **In realtime** — a ranking model picks relevant memories each turn and injects them on the fly\n" +
        "• **Both** — session start injection + realtime injection\n" +
        "• **Off for now** — consolidation runs silently; you can turn injection on in Settings later",
      ["At session start", "In realtime", "Both", "Off for now"],
    );
  }

  async function handleMemoryInject(userMessage: string) {
    const msg = userMessage.toLowerCase();
    pushAssistantLoading();

    const wantsSession =
      msg.includes("session") || msg.includes("start") || msg.includes("both");
    const wantsRealtime =
      msg.includes("realtime") || msg.includes("real-time") || msg.includes("real time") ||
      msg.includes("both");
    const wantsOff =
      !wantsSession && !wantsRealtime &&
      (msg.includes("off") || msg.includes("later") || msg.includes("manual") ||
       msg.includes("neither") || msg.includes("no"));

    if (!wantsSession && !wantsRealtime && !wantsOff) {
      resolveLastAssistant(
        "Choose how memories should be injected — or pick **Off for now** to configure later.",
        ["At session start", "In realtime", "Both", "Off for now"],
      );
      return;
    }

    try {
      await patchSettings({
        memory: {
          inject_on_session_start: wantsSession,
          inject_realtime: wantsRealtime,
        } as AppSettings["memory"],
      });
    } catch {
      // Non-fatal
    }

    if (wantsOff) {
      resolveLastAssistant(
        "Got it — injection is off for now. You can turn it on anytime under **Settings → Memory**.",
      );
    } else if (wantsSession && wantsRealtime) {
      resolveLastAssistant(
        "Both modes enabled — MEMORY.md will be loaded at session start, and relevant snippets will be injected in realtime.",
      );
    } else if (wantsSession) {
      resolveLastAssistant(
        "Session-start injection enabled — MEMORY.md will be loaded into the system prompt once when each session begins.",
      );
    } else {
      resolveLastAssistant(
        "Realtime injection enabled — relevant memories will be ranked and injected on the fly each turn.",
      );
    }

    setStep("activity");
    pushAssistant(
      "Should Otto track your Mac activity in the background to give better context? This requires macOS Accessibility permission.",
      ["Yes, enable activity tracking", "No thanks"],
    );
  }

  async function handleActivity(userMessage: string) {
    const extracted = extractBool(userMessage);
    pushAssistantLoading();

    if (extracted === null) {
      resolveLastAssistant(
        "Just to confirm — would you like Otto to track Mac activity for better context?",
        ["Yes, enable activity tracking", "No thanks"],
      );
      return;
    }

    try {
      await patchSettings({ activity: { enabled: extracted } });
    } catch {
      // Non-fatal
    }

    if (extracted) {
      // Prompt macOS for Accessibility permission — mirrors what the wizard does.
      // This triggers the system dialog; we don't block on the result.
      try { await api.promptAccessibilityPermission(); } catch { /* ignore */ }
      resolveLastAssistant(
        "Activity tracking enabled — a macOS permission dialog may have appeared. " +
          "If you didn't see it, go to System Settings → Privacy & Security → Accessibility and toggle Otto on.",
      );
    } else {
      resolveLastAssistant("No problem — activity tracking is off. You can enable it anytime in Settings.");
    }

    setStep("ambient");
    pushAssistant(
      "One last question — would you like Otto to proactively surface suggestions " +
        "based on your sessions and Mac activity? It uses a tiny on-device model and " +
        "only notifies you when you're idle.",
      ["Yes, enable ambient suggestions", "No thanks"],
    );
  }

  async function handleAmbient(userMessage: string) {
    const extracted = extractBool(userMessage);
    pushAssistantLoading();

    if (extracted === null) {
      resolveLastAssistant(
        "Would you like proactive ambient suggestions? They run on-device only when you're idle. (Yes / No)",
        ["Yes, enable ambient suggestions", "No thanks"],
      );
      return;
    }

    try {
      await patchSettings({
        ambient: {
          ...(settings?.ambient ?? {}),
          enabled: extracted,
        },
      });
    } catch {
      // Non-fatal
    }

    if (extracted) {
      resolveLastAssistant(
        "Ambient suggestions enabled — Otto will quietly analyse your work and surface ideas when you're free.",
      );
    } else {
      resolveLastAssistant("No problem — you can enable ambient suggestions anytime in Settings → Agent Memory → Ambient.");
    }

    setStep("evaluation");
    pushAssistant(
      "One last thing — when a run finishes, should Otto automatically evaluate it? " +
        "An LLM picks suitable metrics and scores the result so you can track quality over time. " +
        "You can always run evaluations manually instead.",
      ["Yes, auto-evaluate runs", "No thanks"],
    );
  }

  async function handleEvaluation(userMessage: string) {
    const extracted = extractBool(userMessage);
    pushAssistantLoading();

    if (extracted === null) {
      resolveLastAssistant(
        "Just to confirm — would you like Otto to automatically evaluate each completed run? (Yes / No)",
        ["Yes, auto-evaluate runs", "No thanks"],
      );
      return;
    }

    try {
      await patchSettings({
        evaluation: {
          ...(settings?.evaluation ?? {}),
          auto_evaluate: extracted,
        },
      });
    } catch {
      // Non-fatal
    }

    if (extracted) {
      resolveLastAssistant(
        "Auto-evaluation enabled — each completed run will be scored automatically. You can tune the metrics and model in Settings → Observability.",
      );
    } else {
      resolveLastAssistant(
        "No problem — runs won't be evaluated automatically. You can score any run with the Evaluate button, or turn this on later in Settings → Observability.",
      );
    }

    setStep("done");
    try { await api.setupComplete(); } catch { /* ignore */ }
    try { await api.setupMarkStep("done"); } catch { /* ignore */ }

    pushAssistant(
      "You're all set! Everything can be changed later in Settings. Click 'Open Otto' to get started.",
    );
  }

  // Called from the inline ModelPickerCard when the user confirms a model
  async function handleModelPick(modelId: string, label: string) {
    if (isBusy || step === "done") return;
    pushUser(label);
    setIsBusy(true);
    void startModelDownloadIfNeeded();
    try {
      if (step === "local_model") await handleLocalModel(modelId);
      else if (step === "cloud_model") await handleCloudModel(modelId);
    } finally {
      setIsBusy(false);
    }
  }

  // Called when the user clicks "Skip, choose later"
  async function handleModelSkip() {
    if (isBusy || step === "done") return;
    pushUser("Skip — I'll choose later in Settings");
    setIsBusy(true);
    try {
      if (step === "local_model") {
        if (providerChoice === "exo") {
          // EXO: just note the skip; model is chosen at inference time
          pushAssistant("OK — you can pick a model from Settings → LLM → Cluster when you're ready.");
          advanceToMemory();
        } else if (providerChoice === "omlx") {
          // oMLX: skip model load, advance
          pushAssistant("OK — you can load a model from Settings → LLM → On-Device → oMLX when you're ready.");
          advanceToMemory();
        } else {
          // MLX: use the recommended model silently
          const rec = recommendedMainModel(capabilities.ram_gb);
          await patchSettings({ llm: { provider: "mlx", mlx: { hf_llm_model_id: rec.id } } }).catch(() => undefined);
          pushAssistant(`OK — I'll default to ${rec.label}. You can change it anytime in Settings → LLM.`);
          advanceToMemory();
        }
      } else if (step === "cloud_model") {
        // Use the first model in the list silently
        const provider = cloudSub ?? (providerChoice as CloudSub);
        const list = cloudModelList.length > 0
          ? cloudModelList
          : CURATED_CLOUD_MODELS[provider as "anthropic" | "openai"] ?? [];
        const first = list[0];
        if (first) {
          await patchSettings({ llm: { provider, [provider!]: { model_provider: provider, model_name: first.id } } }).catch(() => undefined);
          pushAssistant(`OK — I'll default to ${first.label}. You can change it anytime in Settings → LLM.`);
        }
        advanceToMemory();
      }
    } finally {
      setIsBusy(false);
    }
  }

  // -------------------------------------------------------------------------
  // Rich model picker callbacks (ExoModelChooser / ModelChooser / OmlxModelPicker)
  // -------------------------------------------------------------------------

  /**
   * Called by the embedded ExoModelChooser when the user selects a model that
   * is already loaded (`onUseLoaded`) or when a preload finishes
   * (`onPreloadComplete`).  Saves the model to settings and advances setup.
   */
  /**
   * Called by the embedded ModelChooser when the user picks a cached model
   * (`onUseCached`) or a download completes (`onDownloadComplete`).
   * Saves the model to settings and advances setup.
   */
  async function handleMlxRichSelect(repoId: string, displayName: string) {
    if (isBusy || step === "done") return;
    setIsBusy(true);
    const shortName = displayName || (repoId.split("/")[1] ?? repoId);
    try {
      await patchSettings({ llm: { provider: providerChoice === "exo" ? "exo" : "mlx", mlx: { hf_llm_model_id: repoId } } });
    } catch { /* non-fatal */ }
    pushAssistant(`Got it — I'll use **${shortName}**.`);
    setIsBusy(false);
    advanceToMemory();
  }

  /**
   * Called by the embedded OmlxModelPicker when the user clicks Load / Get /
   * Select.  The picker shows its own row-level spinner while this promise is
   * pending, so we do the actual model load here and advance when it resolves.
   */
  async function handleOmlxRichLoad(modelId: string): Promise<void> {
    const shortName = modelId.split("/")[1] ?? modelId;
    try {
      await patchSettings({ llm: { provider: "omlx" }, omlx: { enabled: true, model_name: modelId } });
      const loadJob = await api.omlxLoadModel(modelId);
      const result = await awaitOmlxJob(loadJob.id, 300, (line) =>
        updateLastAssistantCommand(`Loading **${shortName}** into oMLX…`, line),
      );
      if (result === "done") {
        pushAssistant(`✓ **${shortName}** loaded into oMLX.`);
      } else {
        pushAssistant(`Loading is taking longer than expected — you can track it in Settings → LLM → oMLX.`);
      }
    } catch {
      pushAssistant(`Couldn't load **${shortName}** automatically — you can do that from Settings → LLM → oMLX.`);
    }
    advanceToMemory();
  }

  // -------------------------------------------------------------------------
  // Send dispatcher
  // -------------------------------------------------------------------------

  async function handleSend() {
    const text = input.trim();
    if (!text || isBusy || step === "done") return;

    setInput("");
    // For password steps, show masked text in chat — the raw value is only
    // passed directly to the API handler and never stored in message history.
    pushUser(inputType === "password" ? "••••••••" : text);
    pushHistory();
    setIsBusy(true);

    void startModelDownloadIfNeeded();

    try {
      switch (step) {
        case "provider":            await handleProvider(text);           break;
        case "cloud_sub":           await handleCloudSub(text);          break;
        case "cloud_key":           await handleCloudKey(text);          break;
        case "cloud_model":         await handleCloudModel(text);        break;
        case "local_model":         await handleLocalModel(text);        break;
        case "azure_endpoint":      await handleAzureEndpoint(text);     break;
        case "azure_key":           await handleAzureKey(text);          break;
        case "azure_deployment":    await handleAzureDeployment(text);   break;
        case "azure_api_version":   await handleAzureApiVersion(text);   break;
        case "memory":              await handleMemory(text);            break;
        case "memory_inject":       await handleMemoryInject(text);      break;
        case "activity":            await handleActivity(text);          break;
        case "ambient":             await handleAmbient(text);           break;
      }
    } finally {
      setIsBusy(false);
    }
  }

  /** Called when the user submits an inline input embedded inside a message bubble. */
  async function handleInlineSubmit(msgIdx: number, value: string, inputCfg: NonNullable<Message["inlineInput"]>) {
    if (!value.trim() || isBusy) return;
    // Clear the inline input widget and its tracked value
    clearLastInlineInput();
    setInlineInputValues((prev) => {
      const next = { ...prev };
      delete next[msgIdx];
      return next;
    });
    const display = inputCfg.type === "password" ? "••••••••" : value.trim();
    pushUser(display);
    pushHistory();
    setIsBusy(true);
    void startModelDownloadIfNeeded();
    try {
      if (step === "cloud_key") await handleCloudKey(value.trim());
      else if (step === "azure_endpoint") await handleAzureEndpoint(value.trim());
      else if (step === "azure_key") await handleAzureKey(value.trim());
      else if (step === "azure_deployment") await handleAzureDeployment(value.trim());
      else if (step === "azure_api_version") await handleAzureApiVersion(value.trim());
      else if (step === "bedrock_key_id") await handleBedrockKeyId(value.trim());
      else if (step === "bedrock_secret") await handleBedrockSecret(value.trim());
      else if (step === "bedrock_region") await handleBedrockRegion(value.trim());
    } finally {
      setIsBusy(false);
    }
  }

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  const lastMsg = messages[messages.length - 1];
  const hasInlineInput = !!(lastMsg?.inlineInput);
  const inputDisabled = isBusy || step === "done" || hasInlineInput;
  const isDone = step === "done";

  const STEPS: SetupStep[] = ["provider", "local_model", "memory", "memory_inject", "activity", "ambient", "evaluation", "done"];
  const stepIdx = STEPS.indexOf(step);
  // ---------------------------------------------------------------------------
  // Phase: picker — choose auto (chat) or manual (wizard)
  // ---------------------------------------------------------------------------
  if (phase === "picker") {
    return (
      <div className="flex flex-col h-screen bg-th-bg items-center justify-center px-6">
        <div className="w-full max-w-sm">
          {/* Logo — same size/treatment as the main app empty state */}
          <div className="flex justify-center mb-6">
            <img
              src={logo}
              alt="Otto"
              className="w-20 h-20 select-none object-contain"
              draggable={false}
            />
          </div>

          <h1 className="text-xl font-semibold text-th-text-primary text-center mb-1">
            Welcome to Otto
          </h1>
          <p className="text-sm text-th-text-secondary text-center mb-8">
            Your AI agent for macOS. How would you like to get set up?
          </p>

          {/* Choice cards */}
          <div className="flex flex-col gap-2.5">
            <button
              className="group w-full text-left rounded-xl border border-th-border bg-th-surface hover:bg-th-surface-hover hover:border-blue-500/50 transition-all duration-150 px-4 py-4"
              onClick={() => void handleAutoSetup()}
            >
              <div className="flex items-center gap-3.5">
                <div className="w-8 h-8 rounded-lg bg-blue-500/10 group-hover:bg-blue-500/15 flex items-center justify-center shrink-0 transition-colors">
                  <Sparkles className="w-4 h-4 text-blue-500" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-th-text-primary">Let Otto guide me</p>
                  <p className="text-xs text-th-text-tertiary mt-0.5">Conversational setup</p>
                </div>
                <ArrowRight className="w-3.5 h-3.5 text-th-text-muted group-hover:text-th-text-tertiary group-hover:translate-x-0.5 transition-all shrink-0" />
              </div>
            </button>

            <button
              className="group w-full text-left rounded-xl border border-th-border bg-th-surface hover:bg-th-surface-hover transition-all duration-150 px-4 py-4"
              onClick={() => void handleUseLegacy()}
            >
              <div className="flex items-center gap-3.5">
                <div className="w-8 h-8 rounded-lg bg-th-surface-hover group-hover:bg-th-surface-active flex items-center justify-center shrink-0 transition-colors">
                  <Settings2 className="w-4 h-4 text-th-text-secondary" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-th-text-primary">Set up manually</p>
                  <p className="text-xs text-th-text-tertiary mt-0.5">Step-by-step wizard with full control</p>
                </div>
                <ArrowRight className="w-3.5 h-3.5 text-th-text-muted group-hover:text-th-text-tertiary group-hover:translate-x-0.5 transition-all shrink-0" />
              </div>
            </button>
          </div>

          {/* Feature pills */}
          <div className="flex items-center justify-center gap-2 mt-7 flex-wrap">
            {[
              { icon: Lock, label: "Private" },
              { icon: Cpu, label: "Runs locally" },
              { icon: Cloud, label: "Or cloud" },
            ].map(({ icon: Icon, label }) => (
              <span
                key={label}
                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-th-surface border border-th-border text-[11px] text-th-text-muted"
              >
                <Icon className="w-3 h-3" />
                {label}
              </span>
            ))}
          </div>

          <p className="text-[11px] text-th-text-muted text-center mt-5">
            Everything can be changed later in Preferences.
          </p>

          <div className="flex justify-center mt-5">
            <button
              className="text-[11px] text-th-text-muted hover:text-th-text-tertiary underline underline-offset-2 transition-colors"
              onClick={() => void handleSkip()}
            >
              Skip setup
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // Phase: preparing — model download in progress before chat can start
  // ---------------------------------------------------------------------------
  if (phase === "preparing") {
    return (
      <div className="flex flex-col h-screen bg-th-bg items-center justify-center px-6">
        <div className="w-full max-w-xs text-center">
          <div className="flex justify-center mb-6">
            <img
              src={logo}
              alt="Otto"
              className="w-16 h-16 select-none object-contain"
              draggable={false}
            />
          </div>

          <h2 className="text-base font-semibold text-th-text-primary mb-1.5">
            Preparing your assistant
          </h2>
          <p className="text-xs text-th-text-tertiary mb-7 leading-relaxed">
            Downloading the setup model (1.1 GB).
            This only happens once.
          </p>

          {/* Progress bar */}
          <div className="w-full h-1.5 bg-th-surface-hover rounded-full overflow-hidden mb-2.5">
            {downloadProgress !== null ? (
              <div
                className="h-full bg-blue-500 rounded-full transition-all duration-700"
                style={{ width: `${downloadProgress}%` }}
              />
            ) : (
              <div className="h-full w-2/5 bg-blue-500/60 rounded-full animate-pulse" />
            )}
          </div>

          <div className="flex items-center justify-between text-[11px] text-th-text-muted">
            <span className="flex items-center gap-1.5">
              <Loader2 className="w-3 h-3 animate-spin text-blue-500" />
              {downloadProgress !== null ? `${downloadProgress}% downloaded` : "Starting…"}
            </span>
            <span>1.1 GB</span>
          </div>

          <button
            className="mt-9 text-[11px] text-th-text-muted hover:text-th-text-tertiary underline underline-offset-2 transition-colors"
            onClick={() => void handleUseLegacy()}
          >
            Set up manually instead
          </button>
        </div>
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // Phase: chat
  // ---------------------------------------------------------------------------
  return (
    <div className="flex flex-col h-screen bg-th-bg">

      {/* ── Header ─────────────────────────────────────────────────── */}
      <header className="border-b border-th-border px-5 py-3 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <img src={logo} alt="Otto" className="w-5 h-5 select-none object-contain" draggable={false} />
          <span className="text-sm font-semibold text-th-text-primary">Setup</span>
        </div>

        {/* Progress pills */}
        <div className="flex items-center gap-1">
          {STEPS.slice(0, -1).map((s, i) => (
            <div key={s} className={`h-1 rounded-full transition-all duration-300 ${
              i < stepIdx ? "w-5 bg-blue-500" : i === stepIdx ? "w-5 bg-blue-500/35" : "w-2 bg-th-border"
            }`} />
          ))}
        </div>

        <button
          className="text-[11px] text-th-text-muted hover:text-th-text-secondary transition-colors px-2 py-1"
          onClick={() => void handleUseLegacy()}
        >
          Manual setup
        </button>
      </header>

      {/* ── Setup-model download bar ────────────────────────────────── */}
      {!modelReady && downloadJobId && (
        <div className="px-5 py-1.5 border-b border-th-border/60 flex items-center gap-2.5 shrink-0">
          <Loader2 className="w-3 h-3 animate-spin text-blue-500 shrink-0" />
          <div className="flex-1 h-0.5 bg-th-surface-hover rounded-full overflow-hidden">
            {downloadProgress !== null && (
              <div className="h-full bg-blue-500 rounded-full transition-all duration-500"
                   style={{ width: `${downloadProgress}%` }} />
            )}
          </div>
          <span className="text-[10px] text-th-text-muted whitespace-nowrap">
            {downloadProgress !== null ? `${downloadProgress}%` : "Downloading assistant model…"}
          </span>
        </div>
      )}

      {/* ── Message list ────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto px-5 py-5 space-y-4 min-h-0">
        {messages.map((msg, i) => {
          if (msg.role === "user") {
            return (
              /* ── User message ── */
              <div key={i} className="flex justify-end">
                <div className="max-w-[75%] bg-blue-600 rounded-2xl rounded-br-sm px-4 py-2.5 shadow-sm">
                  <p className="text-sm text-white leading-relaxed">{msg.content}</p>
                </div>
              </div>
            );
          }

          /* ── Assistant message ── */
          const hasQuickReplies = !!(msg.quickReplies?.length && !isBusy && !msg.isLoading);
          const hasModelPicker  = !!(msg.modelPicker && !isBusy && !msg.isLoading);
          const showAvatar      = i === 0 || messages[i - 1]?.role !== "assistant";

          const isCliLine = (line: string) => /^\s*(\[|✓|✗|•|>|\$)/.test(line) || line.trim().startsWith("`");
          const contentLines = msg.content.split("\n");
          // Only use legacy CLI-block rendering for messages that don't have the structured commandOutput field
          const hasCliOutput  = !msg.isLoading && !msg.commandOutput && contentLines.some(isCliLine);

          /** Render inline **bold** and `code` tokens */
          const renderInline = (text: string, _key: number) =>
            text.split(/(\*\*[^*]+\*\*|`[^`]+`)/).map((part, pi) => {
              if (part.startsWith("**") && part.endsWith("**"))
                return <strong key={pi} className="font-semibold text-th-text-primary">{part.slice(2, -2)}</strong>;
              if (part.startsWith("`") && part.endsWith("`"))
                return <code key={pi} className="font-mono text-[11px] bg-th-code-bg px-1.5 py-0.5 rounded text-th-text-secondary">{part.slice(1, -1)}</code>;
              return <span key={pi}>{part}</span>;
            });

          return (
            <div key={i} className="flex gap-3 items-start">
              {/* Avatar — shown once per consecutive assistant block, spacer otherwise */}
              {showAvatar ? (
                <div className="w-7 h-7 rounded-full border border-th-border/70 bg-th-inset-bg flex items-center justify-center shrink-0 mt-0.5">
                  <img src={logo} alt="Otto" className="w-4 h-4 object-contain" draggable={false} />
                </div>
              ) : (
                <div className="w-7 shrink-0" />
              )}

              {/* Content */}
              <div className="flex flex-col gap-2 flex-1 min-w-0">
                {hasQuickReplies ? (
                  /* ── Quick-reply card ── */
                  <div className="bg-th-surface border border-th-border/60 rounded-2xl rounded-tl-sm px-4 py-3.5 shadow-sm">
                    <p className="text-sm text-th-text-primary leading-relaxed mb-3 whitespace-pre-wrap">
                      {contentLines.map((line, li) => <span key={li}>{li > 0 && <br />}{renderInline(line, li)}</span>)}
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {msg.quickReplies!.map((reply) => {
                        const parenMatch = reply.match(/^(.+?)\s+\(([^)]+)\)$/);
                        const tooltip = msg.quickReplyTooltips?.[reply];
                        return (
                          <div key={reply} className="relative group/chip">
                            <button
                              onClick={() => void handleQuickReply(reply)}
                              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-blue-500/8 text-blue-400 border border-blue-500/25 hover:bg-blue-500/20 hover:border-blue-500/50 transition-colors"
                            >
                              {parenMatch ? (
                                <>
                                  <span>{parenMatch[1]}</span>
                                  <span className="px-1.5 py-0.5 rounded-md bg-blue-500/20 border border-blue-500/30 text-[10px] font-semibold text-blue-300 leading-none">
                                    {parenMatch[2]}
                                  </span>
                                </>
                              ) : reply}
                            </button>
                            {tooltip && (
                              <button
                                type="button"
                                className="absolute -top-1.5 -right-1.5 w-4 h-4 rounded-full bg-th-surface border border-th-border flex items-center justify-center text-th-text-muted hover:text-th-text-secondary transition-colors z-10"
                                tabIndex={-1}
                                aria-label={`About ${reply}`}
                              >
                                <Info size={9} />
                                {/* Tooltip popover */}
                                <span className="pointer-events-none absolute top-full left-1/2 -translate-x-1/2 mt-2 w-52 rounded-xl border border-th-border bg-th-bg shadow-lg px-3 py-2.5 text-left opacity-0 group-hover/chip:opacity-100 transition-opacity duration-150 z-20">
                                  <span className="block text-[11px] font-semibold text-th-text-primary mb-1">{reply}</span>
                                  <span className="block text-[11px] text-th-text-tertiary leading-relaxed">{tooltip}</span>
                                </span>
                              </button>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ) : (
                  /* ── Regular bubble ── */
                  <div className="rounded-2xl rounded-tl-sm bg-th-surface border border-th-border/60 px-4 py-3 shadow-sm">
                    {msg.isLoading ? (
                      <ThinkingIndicator />
                    ) : hasCliOutput ? (
                      /* Message has mixed narrative + CLI output — split rendering */
                      <div className="space-y-2.5">
                        {(() => {
                          const groups: Array<{ type: "text" | "cli"; lines: string[] }> = [];
                          for (const line of contentLines) {
                            const type = isCliLine(line) ? "cli" : "text";
                            if (!groups.length || groups[groups.length - 1].type !== type) {
                              groups.push({ type, lines: [line] });
                            } else {
                              groups[groups.length - 1].lines.push(line);
                            }
                          }
                          return groups.map((g, gi) => {
                            if (g.type === "text") {
                              return (
                                <p key={gi} className="text-sm text-th-text-primary leading-relaxed whitespace-pre-wrap">
                                  {g.lines.map((line, li) => <span key={li}>{li > 0 && <br />}{renderInline(line, li)}</span>)}
                                </p>
                              );
                            }
                            const stillRunning = gi === groups.length - 1 && isBusy && i === messages.length - 1;
                            return (
                              <div key={gi} className="rounded-xl bg-th-code-bg border border-th-border px-3 py-2.5 overflow-x-auto">
                                <div className="flex items-center gap-1.5 mb-2 pb-1.5 border-b border-th-border">
                                  <div className="flex gap-1">
                                    <span className="w-2 h-2 rounded-full bg-red-500/40" />
                                    <span className="w-2 h-2 rounded-full bg-yellow-500/40" />
                                    <span className="w-2 h-2 rounded-full bg-green-500/40" />
                                  </div>
                                  {stillRunning ? (
                                    <>
                                      <Loader2 size={10} className="animate-spin text-th-text-muted ml-1" />
                                      <span className="text-[10px] text-th-text-muted font-mono">running…</span>
                                    </>
                                  ) : (
                                    <>
                                      <Check size={10} className="text-emerald-400/60 ml-1" />
                                      <span className="text-[10px] text-emerald-400/60 font-mono">done</span>
                                    </>
                                  )}
                                </div>
                                <pre className="text-[11px] font-mono text-th-text-secondary leading-5 whitespace-pre-wrap break-all">
                                  {g.lines.join("\n").replace(/^`|`$/gm, "")}
                                </pre>
                              </div>
                            );
                          });
                        })()}
                      </div>
                    ) : (
                      /* Plain narrative message */
                      <p className="text-sm text-th-text-primary leading-relaxed whitespace-pre-wrap">
                        {contentLines.map((line, li) => <span key={li}>{li > 0 && <br />}{renderInline(line, li)}</span>)}
                      </p>
                    )}
                  </div>
                )}

                {/* ── Inline model picker (simple dropdown) ── */}
                {hasModelPicker && (
                  <ModelPickerCard
                    options={msg.modelPicker!.options}
                    defaultIdx={msg.modelPicker!.defaultIdx ?? 0}
                    allowSkip={msg.modelPicker!.allowSkip ?? false}
                    onSelect={(id, label) => void handleModelPick(id, label)}
                    onSkip={() => void handleModelSkip()}
                  />
                )}

                {/* ── Embedded Cluster setup flow ── */}
                {msg.clusterSetup && !msg.isLoading && settings && (
                  <div className="w-full rounded-xl border border-th-border bg-th-surface p-3">
                    <ClusterSetupFlow
                      variant="onboarding"
                      settings={settings}
                      onPatch={(p) => patchSettings(p as Record<string, unknown>)}
                      onComplete={finishClusterSetup}
                      onBack={backFromClusterSetup}
                    />
                  </div>
                )}

                {/* ── Rich full-catalog model picker ── */}
                {msg.richModelPicker && !msg.isLoading && (
                  <div className="w-full rounded-xl border border-th-border bg-th-surface p-3 space-y-3">
                    {msg.richModelPicker.type === "mlx" && (
                      <ModelChooser
                        selectedRepoId={settings?.llm?.mlx?.hf_llm_model_id}
                        hfToken={settings?.llm?.mlx?.hf_token}
                        cacheDir={settings?.llm?.mlx?.hf_hub_cache}
                        onDownloadComplete={(repoId, displayName) => void handleMlxRichSelect(repoId, displayName)}
                        onUseCached={(repoId, displayName) => void handleMlxRichSelect(repoId, displayName)}
                      />
                    )}
                    {msg.richModelPicker.type === "omlx" && (
                      <OmlxModelPicker
                        onLoad={(modelId) => handleOmlxRichLoad(modelId)}
                        serverRunning={true}
                      />
                    )}
                    {msg.richModelPicker.allowSkip && !isBusy && (
                      <button
                        onClick={() => void handleModelSkip()}
                        className="text-[11px] text-th-text-muted hover:text-th-text-secondary underline underline-offset-2 transition-colors"
                      >
                        Skip, choose later in Settings
                      </button>
                    )}
                  </div>
                )}

                {/* ── Inline text / password input ── */}
                {msg.inlineInput && (
                  <InlineInputWidget
                    cfg={msg.inlineInput}
                    value={inlineInputValues[i] ?? msg.inlineInput.prefill ?? ""}
                    onChange={(v) => setInlineInputValues((prev) => ({ ...prev, [i]: v }))}
                    disabled={isBusy}
                    onSubmit={(v) => void handleInlineSubmit(i, v, msg.inlineInput!)}
                  />
                )}

                {/* ── Collapsible command output panel ── */}
                {msg.commandOutput && (
                  <div className={`rounded-xl overflow-hidden border transition-colors ${
                    msg.commandOutput.status === "error"
                      ? "border-red-500/30"
                      : msg.commandOutput.status === "done"
                      ? "border-emerald-500/20"
                      : "border-th-border"
                  }`}>
                    {/* Header / toggle row */}
                    <button
                      type="button"
                      onClick={() => {
                        setCollapsedOutputs(prev => {
                          const next = new Set(prev);
                          if (next.has(i)) next.delete(i); else next.add(i);
                          return next;
                        });
                      }}
                      className="w-full flex items-center gap-2 px-3 py-2 bg-th-code-bg hover:bg-th-surface-hover transition-colors text-left group"
                    >
                      {/* Traffic-light dots */}
                      <div className="flex gap-1 shrink-0">
                        <span className="w-2 h-2 rounded-full bg-red-500/40" />
                        <span className="w-2 h-2 rounded-full bg-yellow-500/40" />
                        <span className="w-2 h-2 rounded-full bg-green-500/40" />
                      </div>
                      <Terminal size={10} className="text-th-text-muted shrink-0" />
                      {/* Status icon */}
                      {msg.commandOutput.status === "running" ? (
                        <Loader2 size={10} className="animate-spin text-blue-400 shrink-0" />
                      ) : msg.commandOutput.status === "done" ? (
                        <Check size={10} className="text-emerald-400 shrink-0" />
                      ) : (
                        <span className="text-[10px] text-red-400 shrink-0 leading-none font-bold">✗</span>
                      )}
                      {/* Latest line preview */}
                      <span className="text-[10px] font-mono truncate flex-1 text-th-text-muted">
                        {msg.commandOutput.lines[msg.commandOutput.lines.length - 1] ?? "…"}
                      </span>
                      {/* Line count + status badge */}
                      <span className={`text-[9px] font-semibold px-1.5 py-0.5 rounded-full shrink-0 whitespace-nowrap ${
                        msg.commandOutput.status === "running"
                          ? "bg-blue-500/10 text-blue-400"
                          : msg.commandOutput.status === "done"
                          ? "bg-emerald-500/10 text-emerald-400"
                          : "bg-red-500/10 text-red-400"
                      }`}>
                        {msg.commandOutput.lines.length} {msg.commandOutput.lines.length === 1 ? "line" : "lines"}
                      </span>
                      {/* Expand / collapse chevron */}
                      {collapsedOutputs.has(i)
                        ? <ChevronDown size={12} className="text-th-text-muted shrink-0 group-hover:text-th-text-secondary transition-colors" />
                        : <ChevronUp   size={12} className="text-th-text-muted shrink-0 group-hover:text-th-text-secondary transition-colors" />
                      }
                    </button>
                    {/* Output body — hidden when collapsed */}
                    {!collapsedOutputs.has(i) && (
                      <div className="max-h-56 overflow-y-auto px-3 py-2.5 bg-th-code-bg border-t border-th-border/60">
                        <pre className="text-[11px] font-mono text-th-text-secondary leading-[1.65] whitespace-pre-wrap break-all">
                          {msg.commandOutput.lines.join("\n")}
                        </pre>
                        {msg.commandOutput.status === "running" && (
                          <span className="inline-block w-1.5 h-3.5 bg-th-text-secondary/60 animate-pulse rounded-sm ml-0.5 align-middle" />
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}
        <div ref={bottomRef} />
      </div>

      {/* ── Input footer ────────────────────────────────────────────── */}
      <div className="border-t border-th-border px-5 pb-5 pt-3 shrink-0">
        {historyStack.length > 0 && !isDone && !isBusy && (
          <button
            onClick={handleBack}
            className="flex items-center gap-1 text-xs text-th-text-muted hover:text-th-text-secondary transition-colors mb-2"
          >
            <ChevronLeft size={13} />
            Back
          </button>
        )}
        {isDone ? (
          <button
            className="w-full py-3 rounded-2xl bg-blue-600 hover:bg-blue-500 active:bg-blue-700 text-white text-sm font-semibold transition-colors shadow-sm"
            onClick={onFinish}
          >
            Open Otto
          </button>
        ) : (
          <div className="flex items-center gap-2 bg-th-input-bg border border-th-input-border rounded-2xl px-3.5 py-1.5 focus-within:border-blue-400/60 focus-within:ring-2 focus-within:ring-blue-300/20 transition-all shadow-sm">
            <input
              ref={inputRef}
              type={inputType}
              className="flex-1 bg-transparent py-1.5 text-sm text-th-text-primary placeholder-th-text-muted focus:outline-none disabled:opacity-40"
              placeholder={hasInlineInput ? "Reply above ↑" : inputType === "password" ? "Paste your API key…" : "Message Otto…"}
              value={input}
              disabled={inputDisabled}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void handleSend();
                }
              }}
            />
            <button
              onClick={() => void handleSend()}
              disabled={!input.trim() || inputDisabled}
              className={`w-7 h-7 rounded-lg flex items-center justify-center shrink-0 transition-all ${
                input.trim() && !inputDisabled
                  ? "bg-blue-600 text-white hover:bg-blue-500"
                  : "text-th-text-faint"
              }`}
            >
              <Send size={13} />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// InlineInputWidget — text / password input embedded inside a message bubble
// ---------------------------------------------------------------------------

function InlineInputWidget({
  cfg,
  value,
  onChange,
  disabled,
  onSubmit,
}: {
  cfg: NonNullable<Message["inlineInput"]>;
  value: string;
  onChange: (v: string) => void;
  disabled: boolean;
  onSubmit: (v: string) => void;
}) {
  const [showPw, setShowPw] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Auto-focus inline input when it appears
    inputRef.current?.focus();
    inputRef.current?.select();
  }, []);

  const effectiveType = cfg.type === "password" && !showPw ? "password" : "text";

  return (
    <div className="flex items-center gap-2 bg-th-input-bg border border-th-input-border rounded-2xl px-3.5 py-1.5 focus-within:border-blue-400/60 focus-within:ring-2 focus-within:ring-blue-300/20 transition-all shadow-sm">
      <input
        ref={inputRef}
        type={effectiveType}
        className="flex-1 bg-transparent py-1.5 text-sm text-th-text-primary placeholder-th-text-muted focus:outline-none disabled:opacity-40"
        placeholder={cfg.placeholder ?? (cfg.type === "password" ? "Password…" : "Type here…")}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            onSubmit(value);
          }
        }}
      />
      {cfg.type === "password" && (
        <button
          type="button"
          onMouseDown={(e) => { e.preventDefault(); setShowPw((v) => !v); }}
          className="text-th-text-muted hover:text-th-text-primary transition-colors"
          tabIndex={-1}
        >
          {showPw
            ? <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9.88 9.88a3 3 0 1 0 4.24 4.24"/><path d="M10.73 5.08A10.43 10.43 0 0 1 12 5c7 0 10 7 10 7a13.16 13.16 0 0 1-1.67 2.68"/><path d="M6.61 6.61A13.526 13.526 0 0 0 2 12s3 7 10 7a9.74 9.74 0 0 0 5.39-1.61"/><line x1="2" x2="22" y1="2" y2="22"/></svg>
            : <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>
          }
        </button>
      )}
      <button
        onClick={() => onSubmit(value)}
        disabled={!value.trim() || disabled}
        className={`w-7 h-7 rounded-lg flex items-center justify-center shrink-0 transition-all ${
          value.trim() && !disabled
            ? "bg-blue-600 text-white hover:bg-blue-500"
            : "text-th-text-faint"
        }`}
      >
        <Send size={13} />
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Badge chip — coloured status pill used inside the model picker
// ---------------------------------------------------------------------------

function BadgeChip({ badge }: { badge: NonNullable<ModelPickerOption["badge"]> }) {
  const styles: Record<string, string> = {
    green:  "bg-emerald-500/10 border-emerald-500/30 text-emerald-400",
    yellow: "bg-amber-500/10 border-amber-500/30 text-amber-400",
    red:    "bg-red-500/10 border-red-500/30 text-red-400",
    gray:   "bg-th-surface-hover border-th-border text-th-text-muted",
  };
  return (
    <span className={`px-1.5 py-0.5 rounded-full text-[10px] font-medium border shrink-0 ${styles[badge.color] ?? styles.gray}`}>
      {badge.text}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Inline model picker card — dropdown + confirm + skip
// ---------------------------------------------------------------------------

interface ModelPickerCardProps {
  options: ModelPickerOption[];
  defaultIdx: number;
  allowSkip: boolean;
  onSelect: (id: string, label: string) => void;
  onSkip: () => void;
}

function ModelPickerCard({ options, defaultIdx, allowSkip, onSelect, onSkip }: ModelPickerCardProps) {
  const [selectedIdx, setSelectedIdx] = useState(defaultIdx);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const selected = options[selectedIdx];

  // Close on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  if (!options.length) return null;

  return (
    <div className="w-full rounded-xl border border-th-border bg-th-surface p-3 space-y-2.5">

      {/* Custom dropdown trigger */}
      <div ref={ref} className="relative">
        <button
          onClick={() => setOpen((o) => !o)}
          className="w-full flex items-center justify-between gap-2 bg-th-input-bg border border-th-input-border rounded-lg px-3 py-2 text-sm text-left focus:outline-none focus:border-blue-400/60 hover:border-blue-400/40 transition-colors"
        >
          {(() => {
            const [baseName, qualifier] = (selected?.label ?? "Pick a model").split(" — ");
            return (
              <span className="flex items-center gap-2 min-w-0">
                <span className="truncate text-th-text-primary font-medium">{baseName}</span>
                {qualifier && (
                  <span className="shrink-0 px-1.5 py-0.5 rounded-full text-[10px] font-medium bg-th-surface-hover border border-th-border text-th-text-muted">
                    {qualifier}
                  </span>
                )}
                {selected?.badge && <BadgeChip badge={selected.badge} />}
                {selected?.sizeGb && (
                  <span className="shrink-0 px-1.5 py-0.5 rounded-full text-[10px] font-medium bg-th-surface-hover border border-th-border text-th-text-muted">
                    {selected.sizeGb} GB
                  </span>
                )}
              </span>
            );
          })()}
          <ChevronDown size={13} className={`shrink-0 text-th-text-muted transition-transform ${open ? "rotate-180" : ""}`} />
        </button>

        {/* Dropdown list */}
        {open && (
          <div className="absolute z-20 mt-1 w-full rounded-xl border border-th-border bg-th-surface shadow-lg overflow-y-auto max-h-64">
            {options.map((opt, i) => {
              const [baseName, qualifier] = opt.label.split(" — ");
              const isSelected = i === selectedIdx;
              return (
                <button
                  key={opt.id}
                  onClick={() => { setSelectedIdx(i); setOpen(false); }}
                  className={`w-full flex items-center justify-between gap-2 px-3 py-2.5 text-sm text-left transition-colors
                    ${isSelected ? "bg-blue-500/10 text-blue-400" : "text-th-text-primary hover:bg-th-surface-hover"}
                    ${i > 0 ? "border-t border-th-border/50" : ""}
                  `}
                >
                  <span className="flex items-center gap-2 min-w-0">
                    {isSelected
                      ? <Check size={12} className="shrink-0 text-blue-400" />
                      : <span className="w-3 shrink-0" />}
                    <span className="truncate font-medium">{baseName}</span>
                    {qualifier && (
                      <span className={`shrink-0 px-1.5 py-0.5 rounded-full text-[10px] font-medium border
                        ${isSelected
                          ? "bg-blue-500/10 border-blue-500/30 text-blue-400"
                          : "bg-th-surface-hover border-th-border text-th-text-muted"
                        }`}>
                        {qualifier}
                      </span>
                    )}
                  </span>
                  <span className="flex items-center gap-1.5 shrink-0">
                    {opt.badge && <BadgeChip badge={opt.badge} />}
                    {opt.sizeGb && (
                      <span className={`px-1.5 py-0.5 rounded-full text-[10px] font-medium border
                        ${isSelected
                          ? "bg-blue-500/10 border-blue-500/30 text-blue-400"
                          : "bg-th-surface-hover border-th-border text-th-text-muted"
                        }`}>
                        {opt.sizeGb} GB
                      </span>
                    )}
                  </span>
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Model ID in mono — subtle */}
      {selected && (
        <p className="text-[10px] font-mono text-th-text-muted/70 truncate px-0.5">{selected.id}</p>
      )}

      {/* Actions */}
      <div className="flex items-center gap-2">
        <button
          onClick={() => { setOpen(false); selected && onSelect(selected.id, selected.label); }}
          className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold transition-colors"
        >
          <Check size={12} /> Use this model
        </button>
        {allowSkip && (
          <button
            onClick={onSkip}
            className="text-[11px] text-th-text-muted hover:text-th-text-secondary underline underline-offset-2 transition-colors"
          >
            Skip, choose later in Settings
          </button>
        )}
      </div>
    </div>
  );
}
