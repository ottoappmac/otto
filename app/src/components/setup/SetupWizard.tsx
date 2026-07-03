// ---------------------------------------------------------------------------
// First-run Setup Wizard.
//
// A full-screen overlay shown on first launch (and any time the user
// has neither completed nor explicitly skipped the wizard).  Drives the
// user through:
//
//   0. Welcome              — sets the tone, shows the machine summary
//   1. Provider             — on-device vs cloud (EXO deferred to Settings)
//   2a. Local model picker  — wraps ModelChooser
//   2b. Cloud provider      — API key + model dropdown + test
//   3. Agent Memory         — toggle with reassurance pills
//   4. macOS Activity       — toggle + Accessibility permission probe
//   5. Done                 — summary + "Open Otto"
//
// Each step writes its config via PUT /api/settings as it advances,
// and pings POST /api/setup/step so a force-quit resumes in the right
// place. Final-screen "Open Otto" calls POST /api/setup/complete.
// "Skip setup" calls POST /api/setup/skip.
//
// Visual grammar borrowed from ExoSetupSteps — numbered stepper,
// traffic lights, Back/Continue at the bottom, persistent top-right
// skip.
// ---------------------------------------------------------------------------

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  Brain,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Cloud,
  Cpu,
  Eye,
  EyeOff,
  HardDrive,
  Loader2,
  Lock,
  Mic,
  RefreshCw,
  Server,
  Shield,
  ShieldCheck,
  Sparkles,
  TestTube,
  X,
  XCircle,
  Activity as ActivityIcon,
} from "lucide-react";
import { api } from "../../hooks/useApi";
import ModelChooser from "../mlx/ModelChooser";
import ClusterSetupFlow from "../cluster/ClusterSetupFlow";
import VoiceModelChooser from "../voice/VoiceModelChooser";

import type {
  AppSettings,
  MlxCapabilities,
  OmlxJob,
  VoiceConfig,
} from "../../types";

// ---------------------------------------------------------------------------
// Step model
// ---------------------------------------------------------------------------

type StepId =
  | "welcome"
  | "provider"
  | "exo_setup"
  | "local_model"
  | "cloud_provider"
  | "omlx_setup"
  | "voice"
  | "memory"
  | "activity"
  | "ambient"
  | "evaluation"
  | "done";

interface StepMeta {
  id: StepId;
  /** Short label rendered in the top stepper.  Omitted for the welcome
   *  and done screens, which don't appear in the stepper. */
  label?: string;
}

const ALL_STEPS: StepMeta[] = [
  { id: "welcome" },
  { id: "provider", label: "Model" },
  { id: "exo_setup", label: "Cluster" },
  { id: "local_model", label: "Local model" },
  { id: "cloud_provider", label: "Cloud" },
  { id: "omlx_setup", label: "oMLX" },
  { id: "voice", label: "Voice" },
  { id: "memory", label: "Memory" },
  { id: "activity", label: "Activity" },
  { id: "evaluation", label: "Evaluate" },
  { id: "done" },
];

function classNames(...xs: (string | false | null | undefined)[]): string {
  return xs.filter(Boolean).join(" ");
}

// ---------------------------------------------------------------------------
// Public props
// ---------------------------------------------------------------------------

export interface SetupWizardProps {
  /** Optional cursor from the backend so resumed sessions land on the
   *  right screen. */
  initialStep?: string;
  /** Called once the user clicks "Open Otto" on the final screen. */
  onFinish: () => void;
  /** Called once the user confirms the Skip dialog. */
  onSkip: () => void;
  /** Called when the wizard wants to close and navigate to a specific path. */
  onNavigate?: (path: string) => void;
}

// ---------------------------------------------------------------------------
// Top-level component
// ---------------------------------------------------------------------------

export default function SetupWizard({
  initialStep = "welcome",
  onFinish,
  onSkip,
  onNavigate: _onNavigate,
}: SetupWizardProps) {
  const [step, setStep] = useState<StepId>(coerceStep(initialStep));
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [settingsLoaded, setSettingsLoaded] = useState(false);
  const [skipping, setSkipping] = useState(false);
  const [skipConfirm, setSkipConfirm] = useState(false);
  const [finishing, setFinishing] = useState(false);

  // Load current settings once so each screen has a baseline to patch.
  useEffect(() => {
    api.getSettings()
      .then((s) => {
        setSettings(s);
        setSettingsLoaded(true);
      })
      .catch((e) => {
        console.warn("SetupWizard: failed to load settings", e);
        setSettingsLoaded(true);
      });
  }, []);

  // Persist the current cursor on every step change.
  useEffect(() => {
    if (!settingsLoaded) return;
    void api.setupMarkStep(step, false).catch(() => undefined);
  }, [step, settingsLoaded]);

  // ---------- patching helpers ----------

  const patchSettings = useCallback(
    async (patch: Partial<AppSettings>) => {
      if (!settings) return;
      const next: AppSettings = {
        ...settings,
        ...patch,
        llm: { ...settings.llm, ...(patch.llm ?? {}) },
        orchestrator: { ...settings.orchestrator, ...(patch.orchestrator ?? {}) },
        memory: { ...settings.memory, ...(patch.memory ?? {}) },
        activity: { ...settings.activity, ...(patch.activity ?? {}) },
        omlx: { ...settings.omlx, ...(patch.omlx ?? {}) },
        ambient: { ...(settings.ambient ?? {}), ...(patch.ambient ?? {}) },
        voice: { ...(settings.voice ?? {}), ...(patch.voice ?? {}) },
        evaluation: { ...(settings.evaluation ?? {}), ...(patch.evaluation ?? {}) },
      } as AppSettings;
      setSettings(next);
      await api.updateSettings(next as unknown as Record<string, unknown>);
    },
    [settings],
  );

  const advance = useCallback(
    async (target: StepId) => {
      try {
        await api.setupMarkStep(step, true);
      } catch {
        /* best effort */
      }
      setStep(target);
    },
    [step],
  );

  // ---------- step routing ----------

  /** Pick the next forward step from the current one given the user's
   *  chosen provider. The wizard skips ``local_model`` for cloud users
   *  and ``cloud_provider`` for local users. */
  const nextStepFrom = useCallback(
    (cur: StepId): StepId => {
      const provider = settings?.llm.provider ?? "anthropic";
      switch (cur) {
        case "welcome":
          return "provider";
        case "provider":
          if (provider === "mlx") return "local_model";
          if (provider === "omlx") return "omlx_setup";
          if (provider === "exo") return "exo_setup";
          return "cloud_provider";
        case "exo_setup":
          return "memory";
        case "local_model":
          return "memory";
        case "cloud_provider":
          return "memory";
        case "omlx_setup":
          return "memory";
        case "memory":
          return "activity";
        case "activity":
          return "voice";
        case "voice":
          return "evaluation";
        case "evaluation":
          return "done";
        default:
          return "done";
      }
    },
    [settings?.llm.provider],
  );

  const prevStepFrom = useCallback(
    (cur: StepId): StepId => {
      const provider = settings?.llm.provider ?? "anthropic";
      switch (cur) {
        case "provider":
          return "welcome";
        case "exo_setup":
          return "provider";
        case "local_model":
          return "provider";
        case "cloud_provider":
          return "provider";
        case "omlx_setup":
          return "provider";
        case "memory":
          if (provider === "mlx") return "local_model";
          if (provider === "omlx") return "omlx_setup";
          if (provider === "exo") return "exo_setup";
          return "cloud_provider";
        case "activity":
          return "memory";
        case "voice":
          return "activity";
        case "evaluation":
          return "voice";
        case "done":
          return "evaluation";
        default:
          return "welcome";
      }
    },
    [settings?.llm.provider],
  );

  const goNext = useCallback(() => {
    void advance(nextStepFrom(step));
  }, [step, advance, nextStepFrom]);

  const goBack = useCallback(() => {
    setStep(prevStepFrom(step));
  }, [step, prevStepFrom]);

  // ---------- finish / skip ----------

  const handleFinish = useCallback(async () => {
    setFinishing(true);
    try {
      await api.setupComplete();
      onFinish();
    } catch (e) {
      console.warn("Setup complete failed:", e);
      setFinishing(false);
    }
  }, [onFinish]);

  const handleSkip = useCallback(async () => {
    setSkipping(true);
    try {
      await api.setupSkip();
      onSkip();
    } catch (e) {
      console.warn("Setup skip failed:", e);
      setSkipping(false);
    }
  }, [onSkip]);

  // ---------- render ----------

  if (!settingsLoaded || !settings) {
    return (
      <div className="fixed inset-0 z-[100] flex items-center justify-center bg-th-bg">
        <div className="inline-block w-8 h-8 border-2 border-th-border border-t-th-text-secondary rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="fixed inset-0 z-[100] bg-th-bg overflow-y-auto">
      {/* Top bar */}
      <header className="sticky top-0 z-10 bg-th-bg/90 backdrop-blur-sm border-b border-th-border">
        <div className="max-w-3xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sparkles size={16} className="text-th-tab-active-bg" />
            <span className="text-sm font-semibold text-th-text-primary">
              Otto setup
            </span>
          </div>
          <button
            type="button"
            onClick={() => setSkipConfirm(true)}
            disabled={skipping || finishing}
            className="text-xs font-medium text-th-text-tertiary hover:text-th-text-primary disabled:opacity-40 transition-colors"
          >
            Skip setup
          </button>
        </div>
        <Stepper step={step} />
      </header>

      {/* Content */}
      <main className="max-w-2xl mx-auto px-6 py-10 animate-fade-in">
        {step === "welcome" && (
          <WelcomeScreen onContinue={goNext} />
        )}
        {step === "provider" && (
          <ProviderScreen
            current={settings.llm.provider}
            onPick={async (p) => {
              if (p === "exo") {
                // Keep the user in the wizard — advance to the dedicated EXO
                // setup step which handles provision, start, peer discovery,
                // and model selection inline.
                await patchSettings({
                  llm: { ...settings.llm, provider: "exo" },
                  exo: { ...settings.exo, enabled: true },
                });
                await advance("exo_setup");
                return;
              }
              await patchSettings({
                llm: { ...settings.llm, provider: p },
              });
              // Move forward immediately — picking a provider is the
              // commitment. The user can come back via the stepper if
              // they change their mind.
              const target: StepId =
                p === "mlx" ? "local_model"
                : p === "omlx" ? "omlx_setup"
                : "cloud_provider";
              await advance(target);
            }}
            onBack={goBack}
          />
        )}
        {step === "local_model" && (
          <LocalModelScreen
            settings={settings}
            onChosen={async (repoId) => {
              await patchSettings({
                llm: {
                  ...settings.llm,
                  provider: "mlx",
                  mlx: { ...settings.llm.mlx, hf_llm_model_id: repoId },
                },
              });
            }}
            onBack={goBack}
            onContinue={goNext}
          />
        )}
        {step === "cloud_provider" && (
          <CloudProviderScreen
            settings={settings}
            onPatch={patchSettings}
            onBack={goBack}
            onContinue={goNext}
          />
        )}
        {step === "exo_setup" && (
          <ExoSetupScreen
            settings={settings}
            onPatch={patchSettings}
            onBack={goBack}
            onContinue={goNext}
          />
        )}
        {step === "omlx_setup" && (
          <OmlxSetupScreen
            settings={settings}
            onPatch={patchSettings}
            onBack={goBack}
            onContinue={goNext}
          />
        )}
        {step === "voice" && (
          <VoiceSetupScreen
            settings={settings}
            onPatch={patchSettings}
            onBack={goBack}
            onContinue={goNext}
          />
        )}
        {step === "memory" && (
          <MemoryScreen
            settings={settings}
            onPatch={patchSettings}
            onBack={goBack}
            onContinue={goNext}
          />
        )}
        {step === "activity" && (
          <ActivityScreen
            settings={settings}
            onPatch={patchSettings}
            onBack={goBack}
            onContinue={goNext}
          />
        )}
        {step === "ambient" && (
          <AmbientScreen
            settings={settings}
            onPatch={patchSettings}
            onBack={goBack}
            onContinue={goNext}
          />
        )}
        {step === "evaluation" && (
          <EvaluationScreen
            settings={settings}
            onPatch={patchSettings}
            onBack={goBack}
            onContinue={goNext}
          />
        )}
        {step === "done" && (
          <DoneScreen
            settings={settings}
            finishing={finishing}
            onFinish={handleFinish}
            onBack={goBack}
          />
        )}
      </main>

      {/* Skip confirmation */}
      {skipConfirm && (
        <SkipDialog
          onCancel={() => setSkipConfirm(false)}
          onConfirm={handleSkip}
          busy={skipping}
        />
      )}
    </div>
  );
}

function coerceStep(s: string): StepId {
  const ids = ALL_STEPS.map((x) => x.id);
  return (ids.includes(s as StepId) ? s : "welcome") as StepId;
}

// ---------------------------------------------------------------------------
// Stepper — visible from the Provider step onward; hidden on welcome/done.
// ---------------------------------------------------------------------------


function Stepper({ step }: { step: StepId }) {
  if (step === "welcome" || step === "done") return null;

  // Stable ordering for the merged list (exo_setup / local_model / cloud_provider /
  // omlx_setup all occupy slot 2). Build [provider, modelSlot, memory, activity].
  const slot2: { id: StepId; label: string } =
    step === "exo_setup"
      ? { id: "exo_setup", label: "Cluster" }
      : step === "local_model"
        ? { id: "local_model", label: "Local model" }
        : step === "omlx_setup"
          ? { id: "omlx_setup", label: "oMLX" }
          : { id: "cloud_provider", label: "Cloud" };
  const merged: { id: StepId; label: string }[] = [
    { id: "provider", label: "Model" },
    slot2,
    { id: "memory", label: "Memory" },
    { id: "activity", label: "Activity" },
    { id: "voice", label: "Voice" },
    { id: "evaluation", label: "Evaluate" },
  ];

  const rawIdx = merged.findIndex((m) => m.id === step);
  const idx = rawIdx === -1 ? 0 : rawIdx;

  return (
    <div className="max-w-3xl mx-auto px-6 pb-3">
      <div className="flex items-center gap-1.5">
        {merged.map((m, i) => {
          const done = i < idx;
          const active = i === idx;
          return (
            <div key={m.id} className="flex items-center gap-1.5 min-w-0">
              <div
                className={classNames(
                  "w-5 h-5 rounded-full border text-[10px] font-semibold flex items-center justify-center shrink-0 transition-colors",
                  done && "bg-emerald-500/20 border-emerald-500/60 text-emerald-500",
                  !done && active && "bg-th-tab-active-bg/20 border-th-tab-active-bg/60 text-th-tab-active-bg",
                  !done && !active && "bg-th-surface border-th-border text-th-text-muted",
                )}
              >
                {done ? <Check size={11} /> : i + 1}
              </div>
              <span
                className={classNames(
                  "text-[11px] font-medium hidden sm:inline",
                  (done || active) ? "text-th-text-primary" : "text-th-text-muted",
                )}
              >
                {m.label}
              </span>
              {i < merged.length - 1 && (
                <div className="w-6 h-px bg-th-border mx-1" />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Screen 0 — Welcome
// ---------------------------------------------------------------------------

function WelcomeScreen({ onContinue }: { onContinue: () => void }) {
  const [caps, setCaps] = useState<MlxCapabilities | null>(null);

  useEffect(() => {
    api.mlxCapabilities().then(setCaps).catch(() => undefined);
  }, []);

  return (
    <div className="space-y-8 text-center">
      <div className="space-y-3">
        <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-th-tab-active-bg/15 text-th-tab-active-bg">
          <Sparkles size={28} />
        </div>
        <h1 className="text-2xl font-semibold text-th-text-primary tracking-tight">
          Welcome to Otto.
        </h1>
        <p className="text-sm text-th-text-tertiary max-w-md mx-auto leading-relaxed">
          Let's get you set up. Takes about two minutes. Three quick choices —
          which model runs your agents, whether they remember past sessions,
          and whether they can see what's on screen.
        </p>
        <p className="text-xs text-th-text-muted max-w-md mx-auto">
          Everything you configure lives on this Mac. No telemetry, no cloud sync.
        </p>
      </div>

      <MachineSummaryStrip caps={caps} />

      <div className="flex items-center justify-center gap-3 pt-2">
        <button
          type="button"
          onClick={onContinue}
          className="px-5 py-2.5 text-sm font-semibold rounded-lg bg-th-tab-active-bg text-th-tab-active-fg hover:opacity-90 inline-flex items-center gap-2 transition-opacity"
        >
          Get started <ChevronRight size={14} />
        </button>
      </div>
    </div>
  );
}

function MachineSummaryStrip({ caps }: { caps: MlxCapabilities | null }) {
  if (!caps) {
    return (
      <div className="h-9 inline-flex items-center gap-2 px-3 rounded-full border border-th-border bg-th-surface text-[11px] text-th-text-muted">
        <Loader2 size={11} className="animate-spin" /> Reading your Mac…
      </div>
    );
  }
  return (
    <div className="inline-flex items-center gap-x-5 gap-y-1 px-4 py-2 rounded-full border border-th-border bg-th-surface text-[11px] text-th-text-secondary">
      <span className="inline-flex items-center gap-1.5 font-medium text-th-text-primary">
        <Cpu size={11} /> {caps.chip || caps.platform}
      </span>
      <span>{Math.round(caps.ram_gb)} GB RAM</span>
      <span className="inline-flex items-center gap-1.5">
        <HardDrive size={11} /> {Math.round(caps.free_disk_gb)} GB free
      </span>
      {caps.models_cached > 0 && (
        <span className="text-th-text-muted">
          {caps.models_cached} model{caps.models_cached > 1 ? "s" : ""} cached
        </span>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Screen 1 — Provider picker
// ---------------------------------------------------------------------------

function ProviderScreen({
  current,
  onPick,
  onBack,
}: {
  current: string;
  onPick: (p: "mlx" | "anthropic" | "openai" | "omlx" | "exo") => Promise<void>;
  onBack: () => void;
}) {
  const [busy, setBusy] = useState<string | null>(null);

  const pick = async (p: "mlx" | "anthropic" | "openai" | "omlx" | "exo") => {
    setBusy(p);
    try {
      await onPick(p);
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="space-y-6">
      <ScreenHeader
        title="Pick a model provider"
        subtitle="You can change this anytime in Settings."
      />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <ProviderTile
          icon={<Cpu size={20} />}
          title="Local"
          tagline="Private · offline · no cost"
          description="Run MLX-quantised models directly on this Mac. Nothing leaves your machine."
          bullets={[
            "Works offline",
            "No API costs",
            "Private by default",
          ]}
          selected={current === "mlx"}
          onClick={() => pick("mlx")}
          busy={busy === "mlx"}
          accent="emerald"
        />
        <ProviderTile
          icon={<HardDrive size={20} />}
          title="Turbo"
          tagline="Faster · batched · OpenAI API"
          description="Continuous batching & paged KV cache via a local HTTP server (installs without Homebrew)."
          bullets={[
            "Faster on heavy workloads",
            "OpenAI-compatible API",
            "Still fully local",
          ]}
          selected={current === "omlx"}
          onClick={() => pick("omlx")}
          busy={busy === "omlx"}
          accent="blue"
        />
        <ProviderTile
          icon={<Server size={20} />}
          title="Cluster"
          tagline="Distributed · multi-Mac · big models"
          description="Split a large model across two or more Macs over Thunderbolt or LAN."
          bullets={[
            "Runs models too big for one Mac",
            "Thunderbolt or LAN",
            "Auto peer discovery",
          ]}
          selected={current === "exo"}
          onClick={() => pick("exo")}
          busy={busy === "exo"}
          accent="orange"
        />
        <ProviderTile
          icon={<Cloud size={20} />}
          title="Frontier"
          tagline="Frontier models · max capability"
          description="Anthropic Claude, OpenAI GPT, or AWS Bedrock. Bring your own API key."
          bullets={[
            "Most capable frontier models",
            "No local hardware demands",
            "Needs an API key",
          ]}
          selected={current === "anthropic" || current === "openai"}
          onClick={() => pick("anthropic")}
          busy={busy === "anthropic"}
          accent="sky"
        />
      </div>

      <FooterBar onBack={onBack} backLabel="Back" continueDisabled />
    </div>
  );
}

function ProviderTile({
  icon,
  title,
  tagline,
  description,
  bullets,
  selected,
  onClick,
  busy,
  accent,
}: {
  icon: React.ReactNode;
  title: string;
  tagline: string;
  description: string;
  bullets: string[];
  selected: boolean;
  onClick: () => void;
  busy: boolean;
  accent: "emerald" | "sky" | "blue" | "orange";
}) {
  const accentRing =
    accent === "emerald" ? "ring-emerald-500/40 bg-emerald-500/5"
    : accent === "blue" ? "ring-blue-500/40 bg-blue-500/5"
    : accent === "orange" ? "ring-orange-500/40 bg-orange-500/5"
    : "ring-sky-500/40 bg-sky-500/5";
  const accentIcon =
    accent === "emerald" ? "text-emerald-500 bg-emerald-500/15"
    : accent === "blue" ? "text-blue-500 bg-blue-500/15"
    : accent === "orange" ? "text-orange-500 bg-orange-500/15"
    : "text-sky-500 bg-sky-500/15";

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={busy}
      className={classNames(
        "group text-left rounded-2xl border p-5 transition-all relative",
        "hover:border-th-border-strong hover:bg-th-surface-hover/30",
        selected ? `border-transparent ring-2 ${accentRing}` : "border-th-border bg-th-surface",
        busy && "opacity-60 cursor-wait",
      )}
    >
      <div className="flex items-start gap-3 mb-3">
        <div className={classNames("w-10 h-10 rounded-xl flex items-center justify-center shrink-0", accentIcon)}>
          {icon}
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-semibold text-th-text-primary">
            {title}
          </h3>
          <p className="text-[11px] text-th-text-tertiary mt-0.5">{tagline}</p>
        </div>
        {busy && <Loader2 size={14} className="animate-spin text-th-text-muted" />}
      </div>
      <p className="text-xs text-th-text-secondary leading-relaxed mb-3">
        {description}
      </p>
      <ul className="space-y-1.5">
        {bullets.map((b) => (
          <li key={b} className="text-[11px] text-th-text-tertiary flex items-center gap-1.5">
            <Check size={11} className="text-th-text-muted shrink-0" />
            {b}
          </li>
        ))}
      </ul>
    </button>
  );
}

// ---------------------------------------------------------------------------
// Screen 2A — Local model picker
// ---------------------------------------------------------------------------

function LocalModelScreen({
  settings,
  onChosen,
  onBack,
  onContinue,
}: {
  settings: AppSettings;
  onChosen: (repoId: string) => Promise<void>;
  onBack: () => void;
  onContinue: () => void;
}) {
  const [picked, setPicked] = useState<string>(settings.llm.mlx.hf_llm_model_id || "");
  const [pickedLabel, setPickedLabel] = useState<string>("");

  const handleUseCached = useCallback(
    async (repoId: string, displayName: string) => {
      setPicked(repoId);
      setPickedLabel(displayName);
      await onChosen(repoId);
    },
    [onChosen],
  );

  const handleDownloadComplete = useCallback(
    async (repoId: string, displayName: string) => {
      setPicked(repoId);
      setPickedLabel(displayName);
      await onChosen(repoId);
    },
    [onChosen],
  );

  return (
    <div className="space-y-6">
      <div className="space-y-3">
        <button
          type="button"
          onClick={onBack}
          className="inline-flex items-center gap-1 text-xs font-medium text-th-text-tertiary hover:text-th-text-primary transition-colors"
        >
          <ChevronLeft size={13} /> Back
        </button>
        <ScreenHeader
          title="Pick a model to run on this Mac"
          subtitle="We've sorted models by how comfortably they fit your machine. Featured picks are good starting points."
        />
      </div>

      {picked && (
        <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/5 px-4 py-3 flex items-center gap-3">
          <CheckCircle2 size={15} className="text-emerald-500 shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="text-xs font-semibold text-th-text-primary truncate">
              {pickedLabel || picked}
            </p>
            {pickedLabel && pickedLabel !== picked && (
              <p className="text-[10px] text-th-text-muted font-mono truncate mt-0.5">{picked}</p>
            )}
          </div>
          <button
            type="button"
            onClick={onContinue}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg bg-emerald-600 text-white hover:bg-emerald-500 transition-colors shrink-0"
          >
            Continue <ChevronRight size={12} />
          </button>
        </div>
      )}

      <ModelChooser
        selectedRepoId={picked}
        hfToken={settings.llm.mlx.hf_token}
        cacheDir={settings.llm.mlx.hf_hub_cache}
        onUseCached={(id, name) => void handleUseCached(id, name)}
        onDownloadComplete={(id, name) => void handleDownloadComplete(id, name)}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Screen 2B — Cloud provider
// ---------------------------------------------------------------------------

type CloudProvider = "anthropic" | "openai" | "azure" | "bedrock";

/** Return true if a model_name looks like a Bedrock ARN / region-prefixed id
 *  (e.g. "us.anthropic.claude-…" or "anthropic.claude-…").  These must not
 *  be sent to the direct Anthropic API. */
function isBedRockModelName(name: string): boolean {
  return /^(us|eu|ap)?\.*anthropic\./i.test(name);
}

/** Safe Anthropic model name — strips Bedrock-prefixed names and falls back
 *  to the current Sonnet default. */
function safeAnthropicModel(name: string): string {
  if (!name || isBedRockModelName(name)) return "claude-sonnet-4-6";
  return name;
}

/** Curated default model lists for each cloud provider.  Used as the
 *  initial dropdown contents so users can pick a model *before* testing.
 *  After a successful "Test connection", the discovered list replaces
 *  these.  Conservative selection — only well-known stable IDs. */
const CURATED_MODELS: Record<CloudProvider, { id: string; name: string }[]> = {
  anthropic: [
    { id: "claude-sonnet-4-6", name: "Claude Sonnet 4.6" },
    { id: "claude-opus-4-1", name: "Claude Opus 4.1" },
    { id: "claude-haiku-4-5", name: "Claude Haiku 4.5" },
    { id: "claude-3-7-sonnet-latest", name: "Claude 3.7 Sonnet" },
    { id: "claude-3-5-sonnet-latest", name: "Claude 3.5 Sonnet" },
    { id: "claude-3-5-haiku-latest", name: "Claude 3.5 Haiku" },
  ],
  openai: [
    { id: "gpt-5", name: "GPT-5" },
    { id: "gpt-5-mini", name: "GPT-5 mini" },
    { id: "gpt-4.1", name: "GPT-4.1" },
    { id: "gpt-4.1-mini", name: "GPT-4.1 mini" },
    { id: "gpt-4o", name: "GPT-4o" },
    { id: "gpt-4o-mini", name: "GPT-4o mini" },
    { id: "o3", name: "o3" },
    { id: "o3-mini", name: "o3 mini" },
    { id: "o4-mini", name: "o4 mini" },
    { id: "o1", name: "o1" },
  ],
  azure: [
    { id: "gpt-5", name: "GPT-5" },
    { id: "gpt-4o", name: "GPT-4o" },
    { id: "gpt-4o-mini", name: "GPT-4o mini" },
    { id: "gpt-4.1", name: "GPT-4.1" },
    { id: "gpt-4.1-mini", name: "GPT-4.1 mini" },
    { id: "gpt-4.1-nano", name: "GPT-4.1 nano" },
    { id: "o3", name: "o3" },
    { id: "o3-mini", name: "o3 mini" },
    { id: "o4-mini", name: "o4 mini" },
    { id: "o1", name: "o1" },
    { id: "o1-mini", name: "o1 mini" },
  ],
  bedrock: [
    { id: "us.anthropic.claude-sonnet-4-20250514-v1:0", name: "Claude Sonnet 4 (US)" },
    { id: "us.anthropic.claude-opus-4-20250514-v1:0", name: "Claude Opus 4 (US)" },
    { id: "us.anthropic.claude-3-7-sonnet-20250219-v1:0", name: "Claude 3.7 Sonnet (US)" },
    { id: "us.anthropic.claude-3-5-sonnet-20241022-v2:0", name: "Claude 3.5 Sonnet v2 (US)" },
    { id: "us.anthropic.claude-3-5-haiku-20241022-v1:0", name: "Claude 3.5 Haiku (US)" },
    { id: "eu.anthropic.claude-3-5-sonnet-20240620-v1:0", name: "Claude 3.5 Sonnet (EU)" },
  ],
};

/** Merge curated + discovered models, de-duped by ID, preserving order
 *  (curated first, then any newly discovered IDs).  Returned list always
 *  includes the currently selected ``modelId`` so it remains visible in
 *  the dropdown even if it's a custom ID. */
function mergedModels(
  prov: CloudProvider,
  discovered: { id: string; name: string }[],
  selected: string,
): { id: string; name: string }[] {
  const base = discovered.length > 0 ? discovered : CURATED_MODELS[prov];
  const seen = new Set<string>();
  const out: { id: string; name: string }[] = [];
  for (const m of base) {
    if (seen.has(m.id)) continue;
    seen.add(m.id);
    out.push(m);
  }
  if (selected && !seen.has(selected)) {
    out.unshift({ id: selected, name: `${selected} (custom)` });
  }
  return out;
}

function CloudProviderScreen({
  settings,
  onPatch,
  onBack,
  onContinue,
}: {
  settings: AppSettings;
  onPatch: (patch: Partial<AppSettings>) => Promise<void>;
  onBack: () => void;
  onContinue: () => void;
}) {
  const initialProv: CloudProvider = useMemo(() => {
    const p = settings.llm.provider;
    if (p === "openai") {
      return settings.llm.openai.model_provider === "azure" ? "azure" : "openai";
    }
    if (p === "anthropic") {
      // model_provider can be "anthropic_bedrock" or "bedrock" depending on
      // which code path saved it last.
      const mp = settings.llm.anthropic.model_provider;
      if (mp === "anthropic_bedrock" || mp === "bedrock") return "bedrock";
    }
    return "anthropic";
  }, [settings.llm.provider, settings.llm.anthropic.model_provider, settings.llm.openai.model_provider]);

  const [prov, setProv] = useState<CloudProvider>(initialProv);
  const [apiKey, setApiKey] = useState<string>(() => {
    if (initialProv === "openai") return settings.llm.openai.api_key;
    if (initialProv === "azure") return settings.llm.openai.azure_api_key;
    return settings.llm.anthropic.api_key;
  });
  const [showKey, setShowKey] = useState(false);
  const [models, setModels] = useState<{ id: string; name: string }[]>([]);
  const [modelId, setModelId] = useState<string>(() => {
    if (initialProv === "openai") return settings.llm.openai.model_name || "gpt-4o";
    if (initialProv === "azure") return settings.llm.openai.model_name || "gpt-4o";
    if (initialProv === "bedrock") {
      return settings.llm.anthropic.model_name || "us.anthropic.claude-sonnet-4-20250514-v1:0";
    }
    return safeAnthropicModel(settings.llm.anthropic.model_name);
  });
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{ ok: boolean; message: string } | null>(null);
  const [fetchingProfiles, setFetchingProfiles] = useState(false);

  // Azure-specific fields
  const [azureEndpoint, setAzureEndpoint] = useState(
    () => settings.llm.openai.azure_endpoint || "",
  );
  const [azureDeployment, setAzureDeployment] = useState(
    () => settings.llm.openai.azure_deployment || "",
  );
  const [azureApiVersion, setAzureApiVersion] = useState(
    () => settings.llm.openai.azure_api_version || "2024-12-01-preview",
  );

  // Bedrock-specific fields
  const [bedrockRegion, setBedrockRegion] = useState(
    () => settings.llm.anthropic.bedrock_region || "us-east-1",
  );
  const [awsKeyId, setAwsKeyId] = useState(() => settings.llm.anthropic.aws_access_key_id || "");
  const [awsSecret, setAwsSecret] = useState(
    () => settings.llm.anthropic.aws_secret_access_key || "",
  );
  const [showAwsSecret, setShowAwsSecret] = useState(false);

  // When the user switches provider tab, reset the key/model from settings.
  useEffect(() => {
    if (prov === "openai") {
      setApiKey(settings.llm.openai.api_key);
      setModelId(settings.llm.openai.model_name || "gpt-4o");
    } else if (prov === "azure") {
      setApiKey(settings.llm.openai.azure_api_key);
      setModelId(settings.llm.openai.model_name || "gpt-4o");
      setAzureEndpoint(settings.llm.openai.azure_endpoint || "");
      setAzureDeployment(settings.llm.openai.azure_deployment || "");
      setAzureApiVersion(settings.llm.openai.azure_api_version || "2024-12-01-preview");
    } else if (prov === "bedrock") {
      setApiKey("");
      setModelId(settings.llm.anthropic.model_name || "us.anthropic.claude-sonnet-4-20250514-v1:0");
      setBedrockRegion(settings.llm.anthropic.bedrock_region || "us-east-1");
      setAwsKeyId(settings.llm.anthropic.aws_access_key_id || "");
      setAwsSecret(settings.llm.anthropic.aws_secret_access_key || "");
    } else {
      // "anthropic" — strip any Bedrock ARN that may have been persisted
      setApiKey(settings.llm.anthropic.api_key);
      setModelId(safeAnthropicModel(settings.llm.anthropic.model_name));
    }
    setModels([]);
    setTestResult(null);
  }, [prov, settings.llm.anthropic.api_key, settings.llm.anthropic.model_name, settings.llm.openai.api_key, settings.llm.openai.azure_api_key, settings.llm.openai.model_name]);

  const fetchBedrockProfiles = useCallback(async (opts: {
    region: string;
    keyId: string;
    secret: string;
  }) => {
    setFetchingProfiles(true);
    try {
      const result = await api.listModels({
        provider: "anthropic",
        api_key: "",
        model_provider: "bedrock",
        bedrock_region: opts.region,
        bedrock_auth_mode: "keys",
        aws_access_key_id: opts.keyId.trim(),
        aws_secret_access_key: opts.secret.trim(),
      });
      if (result.models?.length) {
        setModels(result.models);
        setModelId((prev) => {
          const inList = result.models.some((m) => m.id === prev);
          return inList ? prev : (result.models[0]?.id ?? prev);
        });
      }
    } catch {
      // silently ignore — user can still type a model ID manually
    } finally {
      setFetchingProfiles(false);
    }
  }, [api]);

  // Auto-fetch Bedrock inference profiles once credentials are complete.
  useEffect(() => {
    if (prov !== "bedrock") return;
    if (!awsKeyId.trim() || !awsSecret.trim()) return;
    const timer = setTimeout(() => {
      void fetchBedrockProfiles({
        region: bedrockRegion,
        keyId: awsKeyId,
        secret: awsSecret,
      });
    }, 600);
    return () => clearTimeout(timer);
  }, [prov, bedrockRegion, awsKeyId, awsSecret, fetchBedrockProfiles]);

  const handleTest = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      if (prov === "openai") {
        const r = await api.testConnection({
          provider: "openai",
          api_key: apiKey,
          model_name: modelId || "gpt-4o",
          openai_model_provider: "openai",
        });
        setTestResult({ ok: r.success, message: r.message });
        if (r.success) {
          const list = await api.listModels({ provider: "openai", api_key: apiKey });
          if (list.models?.length) {
            setModels(list.models);
            if (!modelId) setModelId(list.models[0].id);
          }
        }
      } else if (prov === "azure") {
        if (!azureEndpoint.trim()) {
          setTestResult({ ok: false, message: "Azure endpoint URL is required." });
          return;
        }
        const r = await api.testConnection({
          provider: "openai",
          api_key: apiKey,
          model_name: modelId || azureDeployment || "gpt-4o",
          openai_model_provider: "azure",
          azure_endpoint: azureEndpoint.trim(),
          azure_api_version: azureApiVersion,
          azure_deployment: azureDeployment.trim(),
        });
        setTestResult({ ok: r.success, message: r.message });
      } else if (prov === "anthropic") {
        const safeModel = safeAnthropicModel(modelId);
        // Sync the field in case the saved name was a Bedrock ARN
        if (safeModel !== modelId) setModelId(safeModel);
        const r = await api.testConnection({
          provider: "anthropic",
          api_key: apiKey,
          model_name: safeModel,
          model_provider: "anthropic",
        });
        setTestResult({ ok: r.success, message: r.message });
        if (r.success) {
          const list = await api.listModels({
            provider: "anthropic",
            api_key: apiKey,
            model_provider: "anthropic",
          });
          if (list.models?.length) {
            setModels(list.models);
            // pick best, or keep what we already have
            if (!modelId || isBedRockModelName(modelId)) {
              setModelId(pickBestAnthropic(list.models));
            }
          }
        }
      } else {
        // bedrock
        if (!awsKeyId.trim() || !awsSecret.trim()) {
          setTestResult({
            ok: false,
            message: "Enter your AWS Access Key ID and Secret Access Key.",
          });
          return;
        }
        const bedrockReq = {
          provider: "anthropic",
          api_key: "",
          model_name: modelId || "us.anthropic.claude-sonnet-4-20250514-v1:0",
          model_provider: "bedrock",
          bedrock_region: bedrockRegion,
          bedrock_auth_mode: "keys" as const,
          aws_access_key_id: awsKeyId.trim(),
          aws_secret_access_key: awsSecret.trim(),
        };
        const r = await api.testConnection(bedrockReq);
        setTestResult({ ok: r.success, message: r.message });
        // If the test returned a fresh profile list, update it
        if (r.success && r.models?.length) setModels(r.models);
      }
    } catch (e) {
      setTestResult({ ok: false, message: e instanceof Error ? e.message : "Test failed" });
    } finally {
      setTesting(false);
    }
  };

  const persistAndContinue = async () => {
    if (prov === "openai") {
      await onPatch({
        llm: {
          ...settings.llm,
          provider: "openai",
          openai: {
            ...settings.llm.openai,
            model_provider: "openai",
            api_key: apiKey,
            model_name: modelId,
          },
        },
      });
    } else if (prov === "azure") {
      await onPatch({
        llm: {
          ...settings.llm,
          provider: "openai",
          openai: {
            ...settings.llm.openai,
            model_provider: "azure",
            azure_api_key: apiKey,
            model_name: modelId || azureDeployment,
            azure_endpoint: azureEndpoint.trim(),
            azure_deployment: azureDeployment.trim(),
            azure_api_version: azureApiVersion,
          },
        },
      });
    } else if (prov === "anthropic") {
      await onPatch({
        llm: {
          ...settings.llm,
          provider: "anthropic",
          anthropic: {
            ...settings.llm.anthropic,
            model_provider: "anthropic",
            api_key: apiKey,
            model_name: modelId,
          },
        },
      });
    } else {
      // bedrock
      await onPatch({
        llm: {
          ...settings.llm,
          provider: "anthropic",
          anthropic: {
            ...settings.llm.anthropic,
            model_provider: "bedrock",
            model_name: modelId,
            bedrock_region: bedrockRegion,
            bedrock_auth_mode: "keys",
            aws_access_key_id: awsKeyId.trim(),
            aws_secret_access_key: awsSecret.trim(),
          },
        },
      });
    }
    onContinue();
  };

  const bedrockKeysOk =
    prov !== "bedrock" ||
    (!!awsKeyId.trim() && !!awsSecret.trim());
  const azureOk = prov !== "azure" || !!azureEndpoint.trim();
  const canContinue = testResult?.ok === true && !!modelId && bedrockKeysOk && azureOk;

  return (
    <div className="space-y-6">
      <ScreenHeader
        title="Connect a cloud provider"
        subtitle="Your API key is stored locally and only used to call the provider's API."
      />

      {/* Provider sub-tabs */}
      <div className="flex flex-wrap gap-2">
        {(["anthropic", "openai", "azure", "bedrock"] as CloudProvider[]).map((p) => (
          <button
            key={p}
            type="button"
            onClick={() => setProv(p)}
            className={classNames(
              "px-3 py-1.5 text-xs font-medium rounded-md border transition-colors",
              prov === p
                ? "border-th-tab-active-bg bg-th-tab-active-bg/15 text-th-text-primary"
                : "border-th-border bg-th-surface text-th-text-tertiary hover:text-th-text-primary",
            )}
          >
            {p === "anthropic" ? "Anthropic" : p === "openai" ? "OpenAI" : p === "azure" ? "Azure OpenAI" : "AWS Bedrock"}
          </button>
        ))}
      </div>

      <div className="space-y-4 rounded-2xl border border-th-border bg-th-surface p-5">
        {/* Anthropic + OpenAI: model picker, then API key + inline Test button */}
        {(prov === "anthropic" || prov === "openai") && (
          <>
            <div>
              <label className="block text-[11px] font-medium text-th-text-tertiary mb-1.5">
                Model
              </label>
              <select
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
                className="w-full px-3 py-2 bg-th-input-bg border border-th-input-border rounded-md text-sm text-th-text-primary"
              >
                {mergedModels(prov, models, modelId).map((m) => (
                  <option key={m.id} value={m.id}>{m.name}</option>
                ))}
              </select>
              <p className="text-[10px] text-th-text-muted mt-1.5">
                After a successful test we'll fetch the live model list from your account.
              </p>
            </div>
            <div>
              <label className="block text-[11px] font-medium text-th-text-tertiary mb-1.5">
                API key
              </label>
              <div className="flex items-center gap-2">
                <div className="relative flex-1">
                  <input
                    type={showKey ? "text" : "password"}
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    placeholder={prov === "openai" ? "sk-…" : "sk-ant-…"}
                    className="w-full px-3 py-2 pr-9 bg-th-input-bg border border-th-input-border rounded-md text-sm text-th-text-primary placeholder-th-text-muted font-mono"
                  />
                  <button
                    type="button"
                    onClick={() => setShowKey((v) => !v)}
                    className="absolute right-2 top-1/2 -translate-y-1/2 text-th-text-muted hover:text-th-text-primary"
                    title={showKey ? "Hide" : "Show"}
                  >
                    {showKey ? <EyeOff size={14} /> : <Eye size={14} />}
                  </button>
                </div>
                <button
                  type="button"
                  onClick={() => void handleTest()}
                  disabled={!apiKey || !modelId || testing}
                  className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-md bg-th-tab-active-bg text-th-tab-active-fg disabled:opacity-40 hover:opacity-90"
                >
                  {testing ? <Loader2 size={12} className="animate-spin" /> : <TestTube size={12} />}
                  {testing ? "Testing…" : "Test"}
                </button>
              </div>
            </div>
          </>
        )}

        {/* Azure OpenAI */}
        {prov === "azure" && (
          <div className="space-y-3">
            <div>
              <label className="block text-[11px] font-medium text-th-text-tertiary mb-1.5">
                Model
              </label>
              <select
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
                className="w-full px-3 py-2 bg-th-input-bg border border-th-input-border rounded-md text-sm text-th-text-primary"
              >
                {mergedModels("azure", models, modelId).map((m) => (
                  <option key={m.id} value={m.id}>{m.name}</option>
                ))}
              </select>
              <p className="text-[10px] text-th-text-muted mt-1.5">
                Pick the base model your Azure deployment is serving.
              </p>
            </div>
            <div>
              <label className="block text-[11px] font-medium text-th-text-tertiary mb-1.5">
                Endpoint URL
              </label>
              <input
                type="text"
                value={azureEndpoint}
                onChange={(e) => setAzureEndpoint(e.target.value)}
                placeholder="https://your-resource.openai.azure.com"
                className="w-full px-3 py-2 bg-th-input-bg border border-th-input-border rounded-md text-sm font-mono text-th-text-primary placeholder-th-text-muted"
              />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-[11px] font-medium text-th-text-tertiary mb-1.5">
                  Deployment name
                </label>
                <input
                  type="text"
                  value={azureDeployment}
                  onChange={(e) => setAzureDeployment(e.target.value)}
                  placeholder={modelId || "gpt-4o"}
                  className="w-full px-3 py-2 bg-th-input-bg border border-th-input-border rounded-md text-sm font-mono text-th-text-primary placeholder-th-text-muted"
                />
              </div>
              <div>
                <label className="block text-[11px] font-medium text-th-text-tertiary mb-1.5">
                  API version
                </label>
                <input
                  type="text"
                  value={azureApiVersion}
                  onChange={(e) => setAzureApiVersion(e.target.value)}
                  className="w-full px-3 py-2 bg-th-input-bg border border-th-input-border rounded-md text-sm font-mono text-th-text-primary"
                />
              </div>
            </div>
            <div>
              <label className="block text-[11px] font-medium text-th-text-tertiary mb-1.5">
                API key
              </label>
              <div className="relative">
                <input
                  type={showKey ? "text" : "password"}
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="Azure OpenAI key"
                  className="w-full px-3 py-2 pr-9 bg-th-input-bg border border-th-input-border rounded-md text-sm font-mono text-th-text-primary placeholder-th-text-muted"
                />
                <button
                  type="button"
                  onClick={() => setShowKey((v) => !v)}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-th-text-muted hover:text-th-text-primary"
                >
                  {showKey ? <EyeOff size={14} /> : <Eye size={14} />}
                </button>
              </div>
            </div>
            <button
              type="button"
              onClick={() => void handleTest()}
              disabled={testing || !azureEndpoint.trim() || !apiKey.trim() || !modelId}
              className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-md bg-th-tab-active-bg text-th-tab-active-fg disabled:opacity-40 hover:opacity-90"
            >
              {testing ? <Loader2 size={12} className="animate-spin" /> : <TestTube size={12} />}
              {testing ? "Testing…" : "Test connection"}
            </button>
          </div>
        )}

        {prov === "bedrock" && (
          <div className="space-y-4">
            {/* AWS keys */}
            <div className="space-y-3">
              <div>
                <label className="block text-[11px] font-medium text-th-text-tertiary mb-1.5">
                  AWS Access Key ID
                </label>
                <input
                  type="text"
                  value={awsKeyId}
                  onChange={(e) => setAwsKeyId(e.target.value)}
                  placeholder="AKIA…"
                  className="w-full px-3 py-2 bg-th-input-bg border border-th-input-border rounded-md text-sm font-mono text-th-text-primary placeholder-th-text-muted"
                />
              </div>
              <div>
                <label className="block text-[11px] font-medium text-th-text-tertiary mb-1.5">
                  AWS Secret Access Key
                </label>
                <div className="relative">
                  <input
                    type={showAwsSecret ? "text" : "password"}
                    value={awsSecret}
                    onChange={(e) => setAwsSecret(e.target.value)}
                    placeholder="••••••••"
                    className="w-full px-3 py-2 pr-9 bg-th-input-bg border border-th-input-border rounded-md text-sm font-mono text-th-text-primary placeholder-th-text-muted"
                  />
                  <button
                    type="button"
                    onClick={() => setShowAwsSecret((v) => !v)}
                    className="absolute right-2 top-1/2 -translate-y-1/2 text-th-text-muted hover:text-th-text-primary"
                  >
                    {showAwsSecret ? <EyeOff size={14} /> : <Eye size={14} />}
                  </button>
                </div>
              </div>
            </div>

            {/* Region */}
            <div>
              <label className="block text-[11px] font-medium text-th-text-tertiary mb-1.5">
                Region
              </label>
              <input
                type="text"
                value={bedrockRegion}
                onChange={(e) => setBedrockRegion(e.target.value)}
                placeholder="us-east-1"
                className="w-full px-3 py-2 bg-th-input-bg border border-th-input-border rounded-md text-sm font-mono text-th-text-primary placeholder-th-text-muted"
              />
            </div>

            {/* Inference profile */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="block text-[11px] font-medium text-th-text-tertiary">
                  Inference profile
                </label>
                {fetchingProfiles && (
                  <span className="flex items-center gap-1 text-[10px] text-th-text-muted">
                    <Loader2 size={10} className="animate-spin" /> Fetching profiles…
                  </span>
                )}
              </div>
              <select
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
                disabled={fetchingProfiles}
                className="w-full px-3 py-2 bg-th-input-bg border border-th-input-border rounded-md text-sm text-th-text-primary disabled:opacity-60"
              >
                {mergedModels("bedrock", models, modelId).map((m) => (
                  <option key={m.id} value={m.id}>{m.name}</option>
                ))}
              </select>
              {!fetchingProfiles && models.length === 0 && (
                <p className="text-[10px] text-th-text-muted mt-1.5">
                  Fill in your credentials above — profiles will load automatically.
                </p>
              )}
            </div>

            <button
              type="button"
              onClick={() => void handleTest()}
              disabled={testing || fetchingProfiles || !modelId || (!awsKeyId.trim() || !awsSecret.trim())}
              className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-md bg-th-tab-active-bg text-th-tab-active-fg disabled:opacity-40 hover:opacity-90"
            >
              {testing ? <Loader2 size={12} className="animate-spin" /> : <TestTube size={12} />}
              {testing ? "Testing…" : "Test connection"}
            </button>
          </div>
        )}

        {testResult && (
          <div
            className={classNames(
              "rounded-md border px-3 py-2 text-xs flex items-start gap-2",
              testResult.ok
                ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-500"
                : "border-rose-500/40 bg-rose-500/10 text-rose-500",
            )}
          >
            {testResult.ok ? <CheckCircle2 size={12} className="mt-0.5 shrink-0" /> : <XCircle size={12} className="mt-0.5 shrink-0" />}
            <span>{testResult.message}</span>
          </div>
        )}

      </div>

      <FooterBar
        onBack={onBack}
        onContinue={() => void persistAndContinue()}
        continueDisabled={!canContinue}
        continueLabel="Continue"
      />
    </div>
  );
}

function pickBestAnthropic(models: { id: string; name: string }[]): string {
  const sonnet = models.find((m) => /sonnet.*4[._-]?6/i.test(m.id));
  return sonnet?.id || models[0]?.id || "claude-sonnet-4-6";
}
// ---------------------------------------------------------------------------
// Screen 2C — oMLX install + start
//
// Shown when the user picks the "On this Mac (oMLX server)" tile on the
// Provider screen.  oMLX is an external macOS process — see
// ``backend/omlx_provisioner.py``. This screen detects existing installs,
// offers a one-click ``brew install`` flow, then ``brew services start``
// ---------------------------------------------------------------------------
// OmlxModelPicker
//
// Shown when the oMLX server is running but has no models loaded.
// Three tabs:
//   Library  — HF hub cache scan → one-click Load from existing downloads
//   Discover — hardware-scored curated catalog → one-click Load / Download
//   Custom   — text input for any HF repo id or local path
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// OmlxSetupScreen
//
// (or a direct ``omlx serve`` spawn) and a smoke probe of /v1/models.
//
// Honest about failure modes:
//  * No Homebrew → render a manual download fallback (link to releases).
//  * brew install failures → surface the tail of the install log, not a
//    blanket success.
//  * Install ok but server unreachable → keep the user here; don't pretend
//    the model is ready.
// ---------------------------------------------------------------------------

function OmlxSetupScreen({
  settings,
  onPatch,
  onBack,
  onContinue,
}: {
  settings: AppSettings;
  onPatch: (patch: Partial<AppSettings>) => Promise<void>;
  onBack: () => void;
  onContinue: () => void;
}) {
  const [info, setInfo] = useState<import("../../types").OmlxInfo | null>(null);
  const [status, setStatus] = useState<import("../../types").OmlxStatus | null>(null);
  const [job, setJob] = useState<OmlxJob | null>(null);
  const [busy, setBusy] = useState<"install" | "start" | "stop" | "load" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [detecting, setDetecting] = useState(true);
  const [showLog, setShowLog] = useState(false);
  const [showDiagnostics, setShowDiagnostics] = useState(false);

  const refresh = useCallback(async () => {
    setDetecting(true);
    try {
      const [i, s] = await Promise.all([api.omlxInfo(), api.omlxStatus()]);
      setInfo(i);
      setStatus(s);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setDetecting(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Keep settings.omlx.enabled in sync with this screen — once the user
  // lands here, they've committed to oMLX as their provider.  The
  // backend session_manager gates omlx_tools on this flag.
  useEffect(() => {
    if (!settings.omlx?.enabled) {
      void onPatch({ omlx: { ...settings.omlx, enabled: true } });
    }
  }, [settings.omlx, onPatch]);

  // Poll the active job until it finishes.
  useEffect(() => {
    if (!job || job.status === "done" || job.status === "error") return;
    const t = setInterval(async () => {
      try {
        const j = await api.getOmlxJob(job.id);
        setJob(j);
        if (j.status === "done" || j.status === "error") {
          await refresh();
        }
      } catch {
        /* ignore transient */
      }
    }, 1500);
    return () => clearInterval(t);
  }, [job, refresh]);

  const installed = !!info?.detection.installed && !!info.detection.cli_path;
  const reachable = !!status?.reachable;
  const homebrew = !!info?.detection.homebrew;
  const firstModelId = status?.models?.[0]?.id ?? "";
  const modelCount = status?.models?.length ?? 0;
  const jobRunning = job?.status === "running";

  const handleInstall = async () => {
    setBusy("install");
    setError(null);
    setShowLog(true);
    try {
      const j = await api.omlxInstall();
      setJob(j);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const handleStart = async () => {
    setBusy("start");
    setError(null);
    setShowLog(true);
    try {
      const j = await api.omlxStart();
      setJob(j);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const handleStop = async () => {
    setBusy("stop");
    setError(null);
    try {
      const j = await api.omlxStop();
      setJob(j);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const handleAdoptModel = async (modelId: string) => {
    if (!modelId) return;
    await onPatch({
      omlx: { ...settings.omlx, model_name: modelId, enabled: true },
    });
  };

  const canContinue = reachable && !!settings.omlx?.model_name;

  const adoptFirstAndContinue = async () => {
    if (!reachable) return;
    if (!settings.omlx?.model_name && firstModelId) {
      await handleAdoptModel(firstModelId);
    }
    onContinue();
  };

  // Derive which pipeline step is "active" (the next incomplete step).
  const activeStep: 1 | 2 | 3 = !installed ? 1 : !reachable ? 2 : 3;

  return (
    <div className="space-y-5">
      <ScreenHeader
        icon={<Server size={16} />}
        title="Set up oMLX"
        subtitle="Apple Silicon inference server — continuous batching, paged KV cache. Otto manages it as a background process."
      />

      {/* ── Step pipeline ─────────────────────────────────────────── */}
      <div className="space-y-2">

        {/* Step 1 — Install CLI */}
        <OmlxStep
          num={1}
          title="Install the oMLX CLI"
          status={installed ? "done" : activeStep === 1 ? "active" : "pending"}
          doneDetail={info?.detection.cli_path ?? "installed"}
        >
          {/* Homebrew path */}
          {homebrew ? (
            <div className="space-y-3">
              <p className="text-[11px] text-th-text-secondary leading-relaxed">
                We'll run{" "}
                <code className="bg-th-inset-bg px-1 py-0.5 rounded text-[10px]">
                  brew tap jundot/omlx &amp;&amp; brew install omlx
                </code>
                . The first run resolves MLX dependencies and can take a few minutes.
              </p>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => void handleInstall()}
                  disabled={busy !== null || jobRunning}
                  className="px-4 py-2 rounded-lg bg-th-tab-active-bg text-th-tab-active-fg text-xs font-semibold disabled:opacity-50 inline-flex items-center gap-2 transition-opacity hover:opacity-90"
                >
                  {busy === "install" || (job?.kind === "install" && jobRunning) ? (
                    <Loader2 size={12} className="animate-spin" />
                  ) : (
                    <Sparkles size={12} />
                  )}
                  Install oMLX
                </button>
                <a
                  href="https://github.com/jundot/omlx/releases"
                  target="_blank"
                  rel="noreferrer"
                  className="text-[11px] text-th-text-tertiary hover:text-th-text-secondary underline underline-offset-2"
                >
                  Download manually ↗
                </a>
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <p className="text-[11px] text-th-text-secondary leading-relaxed">
                Homebrew isn't on your PATH — no problem. We'll download oMLX
                from its{" "}
                <a href="https://github.com/jundot/omlx/releases" target="_blank" rel="noreferrer" className="underline underline-offset-2">
                  official release
                </a>{" "}
                and install it for you (no admin password required). Without
                Homebrew the server won't auto-restart on reboot.
              </p>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => void handleInstall()}
                  disabled={busy !== null || jobRunning}
                  className="px-4 py-2 rounded-lg bg-th-tab-active-bg text-th-tab-active-fg text-xs font-semibold disabled:opacity-50 inline-flex items-center gap-2 transition-opacity hover:opacity-90"
                >
                  {busy === "install" || (job?.kind === "install" && jobRunning) ? (
                    <Loader2 size={12} className="animate-spin" />
                  ) : (
                    <Sparkles size={12} />
                  )}
                  Install oMLX
                </button>
                <a
                  href="https://github.com/jundot/omlx/releases"
                  target="_blank"
                  rel="noreferrer"
                  className="text-[11px] text-th-text-tertiary hover:text-th-text-secondary underline underline-offset-2"
                >
                  Download manually ↗
                </a>
              </div>
            </div>
          )}
        </OmlxStep>

        {/* Step 2 — Start server */}
        <OmlxStep
          num={2}
          title="Start the server"
          status={reachable ? "done" : activeStep === 2 ? "active" : "pending"}
          doneDetail={`port :${settings.omlx?.api_port ?? 52414} · ${modelCount} model${modelCount !== 1 ? "s" : ""} loaded`}
          doneAction={
            <button
              type="button"
              onClick={() => void handleStop()}
              disabled={busy !== null}
              className="text-[11px] text-th-text-tertiary hover:text-red-400 inline-flex items-center gap-1 transition-colors disabled:opacity-40"
            >
              {busy === "stop" ? <Loader2 size={10} className="animate-spin" /> : <X size={10} />}
              Stop
            </button>
          }
        >
          <div className="space-y-3">
            <p className="text-[11px] text-th-text-secondary leading-relaxed">
              {homebrew
                ? <>We'll use <code className="bg-th-inset-bg px-1 py-0.5 rounded text-[10px]">brew services start omlx</code> so it auto-restarts on crash.</>
                : "We'll spawn omlx serve directly. Without Homebrew it won't auto-restart on reboot."}
            </p>
            <button
              type="button"
              onClick={() => void handleStart()}
              disabled={busy !== null || jobRunning}
              className="px-4 py-2 rounded-lg bg-th-tab-active-bg text-th-tab-active-fg text-xs font-semibold disabled:opacity-50 inline-flex items-center gap-2 transition-opacity hover:opacity-90"
            >
              {busy === "start" || (job?.kind === "start" && jobRunning) ? (
                <Loader2 size={12} className="animate-spin" />
              ) : (
                <Sparkles size={12} />
              )}
              Start server
            </button>
          </div>
        </OmlxStep>

        {/* Step 3 — Choose default model */}
        <OmlxStep
          num={3}
          title="Choose a default model"
          status={canContinue ? "done" : activeStep === 3 ? "active" : "pending"}
          doneDetail={settings.omlx?.model_name ?? ""}
        >
          <div className="space-y-3">
            {modelCount > 0 ? (
              <div className="space-y-1.5">
                <select
                  value={settings.omlx?.model_name || firstModelId || ""}
                  onChange={(e) => void handleAdoptModel(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg border border-th-border bg-th-inset-bg text-xs text-th-text-primary"
                >
                  {(status?.models ?? []).map((m) => (
                    <option key={m.id} value={m.id}>{m.id}</option>
                  ))}
                </select>
                <p className="text-[10px] text-th-text-tertiary">
                  Otto uses this model when oMLX is the provider. You can change it any time in Settings.
                </p>
              </div>
            ) : (
              <p className="text-[11px] text-th-text-tertiary">
                No models loaded yet — start the server first, then refresh.
              </p>
            )}

            <ModelChooser
              selectedRepoId={settings.omlx?.model_name ?? ""}
              hfToken={settings.llm.mlx.hf_token}
              cacheDir={settings.llm.mlx.hf_hub_cache}
              onUseCached={(repoId) => void handleAdoptModel(repoId)}
              onDownloadComplete={(repoId) => void handleAdoptModel(repoId)}
            />
          </div>
        </OmlxStep>
      </div>

      {/* ── Job log ───────────────────────────────────────────────── */}
      {job && (job.status === "running" || job.status === "error") && (
        <div className="rounded-xl border border-th-border bg-th-inset-bg overflow-hidden">
          <button
            type="button"
            onClick={() => setShowLog((v) => !v)}
            className="w-full flex items-center justify-between px-3 py-2 text-[11px] font-medium text-th-text-secondary hover:bg-th-surface-hover/20 transition-colors"
          >
            <span className="inline-flex items-center gap-1.5">
              {job.status === "running" && <Loader2 size={11} className="animate-spin text-th-text-muted" />}
              {job.kind === "install" ? "brew install" : job.kind}
              {" · "}
              <span className={job.status === "error" ? "text-red-400" : "text-th-text-muted"}>{job.status}</span>
            </span>
            <ChevronDown size={11} className={classNames("transition-transform", showLog ? "rotate-180" : "")} />
          </button>
          {showLog && (
            <div className="border-t border-th-border px-3 pb-3">
              <pre className="text-[10px] text-th-text-tertiary whitespace-pre-wrap max-h-40 overflow-auto font-mono leading-relaxed pt-2">
                {(job.log_lines.slice(-40).join("\n")) || "(no output yet)"}
              </pre>
              {job.error && (
                <p className="text-[11px] text-red-400 mt-1.5 leading-relaxed">{job.error}</p>
              )}
            </div>
          )}
        </div>
      )}

      {/* ── Error banner ──────────────────────────────────────────── */}
      {error && (
        <div className="rounded-xl border border-red-500/30 bg-red-500/5 px-3 py-2.5 text-[11px] text-red-400 leading-relaxed">
          {error}
        </div>
      )}

      {/* ── Diagnostics (collapsible) ─────────────────────────────── */}
      <div className="rounded-xl border border-th-border overflow-hidden">
        <button
          type="button"
          onClick={() => setShowDiagnostics((v) => !v)}
          className="w-full flex items-center justify-between px-3 py-2 text-[11px] font-medium text-th-text-tertiary hover:text-th-text-secondary hover:bg-th-surface-hover/20 transition-colors"
        >
          <span className="inline-flex items-center gap-1.5">
            {detecting && <Loader2 size={10} className="animate-spin" />}
            Diagnostics
          </span>
          <span className="inline-flex items-center gap-2">
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); void refresh(); }}
              disabled={detecting}
              className="text-th-text-tertiary hover:text-th-text-secondary disabled:opacity-40 inline-flex items-center gap-1"
            >
              <RefreshCw size={10} className={detecting ? "animate-spin" : ""} />
              Refresh
            </button>
            <ChevronDown size={10} className={classNames("transition-transform", showDiagnostics ? "rotate-180" : "")} />
          </span>
        </button>
        {showDiagnostics && (
          <div className="border-t border-th-border bg-th-surface p-3 grid grid-cols-1 sm:grid-cols-2 gap-2">
            <StatusRow
              ok={installed}
              warn={!installed}
              label="oMLX CLI"
              detail={info?.detection.cli_path ?? "not found on PATH"}
            />
            <StatusRow
              ok={homebrew}
              warn={!homebrew}
              label="Homebrew"
              detail={homebrew ? "found" : "missing — manual install required"}
            />
            <StatusRow
              ok={reachable}
              warn={installed && !reachable}
              label={`Server :${settings.omlx?.api_port ?? 52414}`}
              detail={reachable ? `${modelCount} model(s) loaded` : (status?.error || "not running")}
            />
            <StatusRow
              ok={info?.detection.brew_service_state === "started"}
              warn={info?.detection.brew_service_state === "stopped"}
              label="brew services"
              detail={info?.detection.brew_service_state ?? "—"}
            />
          </div>
        )}
      </div>

      {/* ── Third-party note ─────────────────────────────────────── */}
      <p className="text-[10px] text-th-text-tertiary leading-relaxed px-0.5">
        <AlertTriangle size={10} className="inline text-amber-500 mr-1 -mt-0.5" />
        oMLX is a third-party project. If install fails with a dylib relink error, run{" "}
        <code className="font-mono">brew untap --force jundot/omlx &amp;&amp; brew reinstall omlx</code> in your terminal.
      </p>

      <FooterBar
        onBack={onBack}
        onContinue={adoptFirstAndContinue}
        continueDisabled={!reachable || (!settings.omlx?.model_name && !firstModelId)}
        continueLabel="Continue"
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// OmlxStep — a single row in the 3-step install pipeline
// ---------------------------------------------------------------------------

function OmlxStep({
  num,
  title,
  status,
  doneDetail,
  doneAction,
  children,
}: {
  num: number;
  title: string;
  status: "done" | "active" | "pending";
  doneDetail?: string;
  doneAction?: React.ReactNode;
  children?: React.ReactNode;
}) {
  const isDone = status === "done";
  const isActive = status === "active";

  return (
    <div
      className={classNames(
        "rounded-xl border transition-colors overflow-hidden",
        isDone
          ? "border-emerald-500/30 bg-emerald-500/5"
          : isActive
            ? "border-th-tab-active-bg/40 bg-th-surface"
            : "border-th-border bg-th-surface opacity-50",
      )}
    >
      {/* Header row */}
      <div className="flex items-center gap-3 px-4 py-3">
        {/* Step badge */}
        <span
          className={classNames(
            "inline-flex items-center justify-center w-6 h-6 rounded-full text-[11px] font-bold shrink-0",
            isDone
              ? "bg-emerald-500 text-white"
              : isActive
                ? "bg-th-tab-active-bg text-th-tab-active-fg"
                : "bg-th-inset-bg text-th-text-tertiary border border-th-border",
          )}
        >
          {isDone ? <Check size={12} /> : num}
        </span>

        <div className="flex-1 min-w-0">
          <p
            className={classNames(
              "text-sm font-semibold",
              isDone ? "text-emerald-600 dark:text-emerald-400" : isActive ? "text-th-text-primary" : "text-th-text-tertiary",
            )}
          >
            {title}
          </p>
          {isDone && doneDetail && (
            <p className="text-[10px] text-th-text-tertiary truncate mt-0.5">{doneDetail}</p>
          )}
        </div>

        {isDone && doneAction && (
          <div className="shrink-0">{doneAction}</div>
        )}
      </div>

      {/* Active body */}
      {isActive && children && (
        <div className="border-t border-th-border/60 px-4 py-3">
          {children}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Screen 2D — Cluster setup
//
// Thin wrapper around the shared ClusterSetupFlow (the single Cluster setup
// experience, also used by SetupChatPage and Settings). All runtime, node
// discovery, and model-selection logic lives there now.
// ---------------------------------------------------------------------------

function ExoSetupScreen({
  settings,
  onPatch,
  onBack,
  onContinue,
}: {
  settings: AppSettings;
  onPatch: (patch: Partial<AppSettings>) => Promise<void>;
  onBack: () => void;
  onContinue: () => void;
}) {
  return (
    <div className="space-y-5">
      <ScreenHeader
        icon={<Server size={16} />}
        title="Set up your Cluster"
        subtitle="Run models across one or more Macs. We'll start the cluster on this Mac, then you can add more whenever you like."
      />
      <ClusterSetupFlow
        variant="onboarding"
        settings={settings}
        onPatch={onPatch}
        onBack={onBack}
        onComplete={onContinue}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// StatusRow (shared)
// ---------------------------------------------------------------------------

function StatusRow({
  ok,
  warn,
  label,
  detail,
}: {
  ok: boolean;
  warn?: boolean;
  label: string;
  detail: string;
}) {
  const Icon = ok ? CheckCircle2 : warn ? AlertTriangle : XCircle;
  const color = ok ? "text-emerald-500" : warn ? "text-amber-500" : "text-th-text-muted";
  return (
    <div className="flex items-start gap-2">
      <Icon size={12} className={`${color} mt-0.5 shrink-0`} />
      <div className="min-w-0">
        <p className="text-[11px] font-medium text-th-text-primary">{label}</p>
        <p className="text-[10px] text-th-text-tertiary truncate">{detail}</p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Screen 3 — Agent Memory
// ---------------------------------------------------------------------------

function MemoryScreen({
  settings,
  onPatch,
  onBack,
  onContinue,
}: {
  settings: AppSettings;
  onPatch: (patch: Partial<AppSettings>) => Promise<void>;
  onBack: () => void;
  onContinue: () => void;
}) {
  const provider = settings.llm.provider;
  const isLocal = provider === "mlx" || provider === "exo";

  const [enabled, setEnabled] = useState(settings.memory?.enabled ?? false);
  const [forceLocalSummariser, setForceLocalSummariser] = useState<boolean>(() => {
    // Default: when the main model is cloud, prefer a local summariser
    // for privacy. When the main model is already local, follow it.
    if (isLocal) return false;
    return (settings.memory?.llm_family ?? "follow_main") === "mlx";
  });

  const persist = async () => {
    const patch: Partial<AppSettings> = {
      memory: {
        ...settings.memory,
        enabled,
        // Injection is off by default; users configure it in Settings → Memory.
        inject_enabled: false,
        inject_on_session_start: false,
        inject_realtime: false,
        llm_family: isLocal
          ? "mlx"
          : forceLocalSummariser
            ? "mlx"
            : "follow_main",
      },
    };
    await onPatch(patch);
    onContinue();
  };

  return (
    <div className="space-y-6">
      <ScreenHeader
        icon={<Brain size={18} />}
        title="Agent Memory"
        subtitle="Let agents remember context between sessions — preferences, recurring projects, frequent tasks."
      />

      <div className="rounded-2xl border border-th-border bg-th-surface p-5 space-y-5">
        <ToggleRow
          label="Enable Agent Memory"
          description="After each session, a small local model summarises what happened into a markdown note. Future sessions can search those notes."
          checked={enabled}
          onChange={setEnabled}
        />

        <ReassurancePills
          pills={[
            { icon: <Lock size={11} />, label: "Private" },
            { icon: <HardDrive size={11} />, label: "Stays on this Mac" },
            { icon: <Cpu size={11} />, label: "Local LLM only" },
          ]}
        />

        {enabled && !isLocal && (
          <div className="rounded-xl border border-th-border bg-th-inset-bg p-3 space-y-2">
            <label className="flex items-start gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={forceLocalSummariser}
                onChange={(e) => setForceLocalSummariser(e.target.checked)}
                className="mt-0.5 accent-th-tab-active-bg"
              />
              <div className="text-xs">
                <div className="font-medium text-th-text-primary">
                  Summarise with a local model
                </div>
                <div className="text-th-text-muted mt-0.5 leading-relaxed">
                  Recommended. Keeps session transcripts off your cloud
                  provider — only the final memory note is ever consulted
                  by the main model.
                </div>
              </div>
            </label>
          </div>
        )}

        <details className="text-[11px] text-th-text-tertiary">
          <summary className="cursor-pointer select-none hover:text-th-text-secondary">
            How it works
          </summary>
          <div className="pt-2 space-y-1 leading-relaxed">
            <p>
              After each session ends, a local model reads the transcript
              and writes a short markdown note under
              {" "}<code className="font-mono">~/.otto/memory/</code>. On the
              next session, a relevance ranker pulls only the notes that
              match what you're asking about.
            </p>
            <p>
              Nothing in this loop touches a cloud provider unless you
              explicitly disable the option above. Delete anytime from
              Settings → Agent Memory.
            </p>
          </div>
        </details>
      </div>

      <FooterBar
        onBack={onBack}
        onContinue={() => void persist()}
        continueLabel={enabled ? "Continue" : "Skip and continue"}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Screen 4 — macOS Activity
// ---------------------------------------------------------------------------

function ActivityScreen({
  settings,
  onPatch,
  onBack,
  onContinue,
}: {
  settings: AppSettings;
  onPatch: (patch: Partial<AppSettings>) => Promise<void>;
  onBack: () => void;
  onContinue: () => void;
}) {
  const [enabled, setEnabled] = useState(settings.activity?.enabled ?? false);
  const [permission, setPermission] = useState<{
    supported: boolean;
    granted: boolean;
    can_prompt: boolean;
  } | null>(null);
  const [probing, setProbing] = useState(false);
  const [prompted, setPrompted] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval>>();

  const probe = useCallback(async () => {
    setProbing(true);
    try {
      const r = await api.accessibilityPermission();
      setPermission(r);
    } catch (e) {
      console.warn("permission probe failed:", e);
    } finally {
      setProbing(false);
    }
  }, []);

  useEffect(() => {
    void probe();
  }, [probe]);

  // While enabled but ungranted, poll every 2 s so the moment the user
  // flips Accessibility on in System Settings, the wizard catches up.
  useEffect(() => {
    if (!enabled) return;
    if (!permission?.supported) return;
    if (permission.granted) return;
    intervalRef.current = setInterval(() => void probe(), 2000);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [enabled, permission?.supported, permission?.granted, probe]);

  const requestPermission = async () => {
    setPrompted(true);
    try {
      await api.promptAccessibilityPermission();
    } catch (e) {
      console.warn("permission prompt failed:", e);
    }
    void probe();
  };

  const persist = async () => {
    await onPatch({
      activity: {
        ...settings.activity,
        enabled,
      },
    });
    onContinue();
  };

  const blockedOnPermission =
    enabled && permission?.supported && !permission.granted;

  return (
    <div className="space-y-6">
      <ScreenHeader
        icon={<ActivityIcon size={18} />}
        title="macOS Activity"
        subtitle="Let agents answer “what was I working on this morning?” by recording the foreground app, window title, and active URL on a timer."
      />

      <div className="rounded-2xl border border-th-border bg-th-surface p-5 space-y-5">
        <ToggleRow
          label="Enable Activity tracking"
          description="Stores app, window title, and browser URL every 15 seconds. No screenshots, no page contents by default."
          checked={enabled}
          onChange={setEnabled}
        />

        <ReassurancePills
          pills={[
            { icon: <Shield size={11} />, label: "No screenshots" },
            { icon: <HardDrive size={11} />, label: "SQLite on this Mac" },
            { icon: <Lock size={11} />, label: "Local search only" },
          ]}
        />

        {enabled && permission?.supported && (
          <div
            className={classNames(
              "rounded-xl border p-3 flex items-start gap-3",
              permission.granted
                ? "border-emerald-500/30 bg-emerald-500/5"
                : "border-amber-500/40 bg-amber-500/5",
            )}
          >
            {permission.granted ? (
              <CheckCircle2 size={14} className="text-emerald-500 mt-0.5 shrink-0" />
            ) : (
              <AlertTriangle size={14} className="text-amber-500 mt-0.5 shrink-0" />
            )}
            <div className="flex-1 min-w-0">
              {permission.granted ? (
                <p className="text-xs text-th-text-primary">
                  Accessibility permission granted — window titles will be captured.
                </p>
              ) : (
                <>
                  <p className="text-xs font-medium text-th-text-primary">
                    Accessibility permission required
                  </p>
                  <p className="text-[11px] text-th-text-muted mt-0.5 leading-relaxed">
                    macOS requires Accessibility access to read window
                    titles. {prompted
                      ? <>If no dialog appeared, open <span className="font-mono">System Settings → Privacy &amp; Security → Accessibility</span> and toggle Otto on.</>
                      : <>Click below to grant — you'll see a one-time system prompt.</>}
                  </p>
                  <div className="flex items-center gap-2 mt-2">
                    {!prompted && (
                      <button
                        type="button"
                        onClick={() => void requestPermission()}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-medium rounded-md bg-th-tab-active-bg text-th-tab-active-fg hover:opacity-90"
                      >
                        <ShieldCheck size={11} /> Grant permission
                      </button>
                    )}
                    <button
                      type="button"
                      onClick={() => void probe()}
                      disabled={probing}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-medium rounded-md border border-th-border bg-th-surface text-th-text-secondary hover:bg-th-surface-hover disabled:opacity-40"
                    >
                      {probing ? <Loader2 size={11} className="animate-spin" /> : <RefreshCw size={11} />}
                      Re-check
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        {enabled && permission && !permission.supported && (
          <div className="rounded-xl border border-th-border bg-th-inset-bg p-3 text-[11px] text-th-text-muted">
            Activity tracking is macOS-only. The toggle will be saved
            but won't have any effect on this platform.
          </div>
        )}

        <details className="text-[11px] text-th-text-tertiary">
          <summary className="cursor-pointer select-none hover:text-th-text-secondary">
            What's recorded
          </summary>
          <ul className="pt-2 space-y-1 list-disc list-inside leading-relaxed">
            <li>App name and window title</li>
            <li>URL of the active browser tab (no page contents)</li>
            <li>Idle skip after 3 minutes of no keyboard/mouse input</li>
            <li>30-day rolling retention, 5 GB cap (configurable)</li>
            <li>All data lives in a local SQLite DB; clear anytime from Settings</li>
          </ul>
        </details>
      </div>

      <FooterBar
        onBack={onBack}
        onContinue={() => void persist()}
        continueDisabled={blockedOnPermission}
        continueLabel={enabled ? "Continue" : "Skip and continue"}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Screen 5 — Ambient
// ---------------------------------------------------------------------------

function AmbientScreen({
  settings,
  onPatch,
  onBack,
  onContinue,
}: {
  settings: AppSettings;
  onPatch: (patch: Partial<AppSettings>) => Promise<void>;
  onBack: () => void;
  onContinue: () => void;
}) {
  const [enabled, setEnabled] = useState(settings?.ambient?.enabled ?? false);
  const [saving, setSaving] = useState(false);

  const persist = async () => {
    setSaving(true);
    try {
      await onPatch({ ambient: { ...(settings?.ambient ?? {}), enabled } });
      await api.setupMarkStep("ambient", true);
      onContinue();
    } catch {
      /* non-fatal */
      onContinue();
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="flex flex-col gap-6 max-w-lg mx-auto w-full">
      <div>
        <h2 className="text-xl font-semibold text-th-text-primary mb-1">Ambient suggestions</h2>
        <p className="text-sm text-th-text-muted leading-relaxed">
          Otto can quietly run a small background model to surface helpful ideas based on
          your sessions, memory, and Mac activity — only when you're idle.
        </p>
      </div>

      <div className="bg-th-card-bg border border-th-card-border rounded-xl p-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-sm font-medium text-th-text-primary">Enable ambient assistant</p>
            <p className="text-[11px] text-th-text-muted mt-0.5 max-w-xs">
              Uses a tiny on-device model (Qwen3-1.7B, ~1 GB — already downloaded).
              Shows suggestions in the sidebar; never acts without your approval.
            </p>
          </div>
          <button
            type="button"
            role="switch"
            aria-checked={enabled}
            onClick={() => setEnabled((v) => !v)}
            className={`relative shrink-0 h-6 w-11 rounded-full transition-colors duration-200 ${
              enabled ? "bg-blue-500" : "bg-neutral-600"
            }`}
          >
            <span
              className={`block h-4 w-4 rounded-full bg-white shadow-sm absolute top-1 transition-transform duration-200 ${
                enabled ? "translate-x-6" : "translate-x-1"
              }`}
            />
          </button>
        </div>
      </div>

      <FooterBar
        onBack={onBack}
        onContinue={() => void persist()}
        continueDisabled={saving}
        continueLabel={enabled ? "Enable and continue" : "Skip and continue"}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Screen 6 — Auto-evaluation
// ---------------------------------------------------------------------------

function EvaluationScreen({
  settings,
  onPatch,
  onBack,
  onContinue,
}: {
  settings: AppSettings;
  onPatch: (patch: Partial<AppSettings>) => Promise<void>;
  onBack: () => void;
  onContinue: () => void;
}) {
  const [enabled, setEnabled] = useState(
    settings?.evaluation?.auto_evaluate ?? false,
  );
  const [saving, setSaving] = useState(false);

  const persist = async () => {
    setSaving(true);
    try {
      await onPatch({
        evaluation: {
          ...settings.evaluation,
          auto_evaluate: enabled,
        },
      });
      onContinue();
    } catch {
      onContinue();
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-6">
      <ScreenHeader
        icon={<TestTube size={18} />}
        title="Auto-evaluate runs"
        subtitle="When a run finishes, Otto can automatically score it — an LLM picks suitable metrics and grades the result so you can spot regressions over time."
      />

      <div className="rounded-2xl border border-th-border bg-th-surface p-5 space-y-5">
        <ToggleRow
          label="Auto-evaluate completed runs"
          description="When a run finishes, an LLM picks suitable metrics and scores it automatically. Turn this off to evaluate runs manually with the Evaluate button on each run."
          checked={enabled}
          onChange={setEnabled}
        />

        <ReassurancePills
          pills={[
            { icon: <Sparkles size={11} />, label: "Auto metric selection" },
            { icon: <HardDrive size={11} />, label: "Scores stored locally" },
            { icon: <Check size={11} />, label: "Change anytime" },
          ]}
        />
      </div>

      <FooterBar
        onBack={onBack}
        onContinue={() => void persist()}
        continueDisabled={saving}
        continueLabel={enabled ? "Enable and continue" : "Skip and continue"}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Screen 7 — Done
// ---------------------------------------------------------------------------

function DoneScreen({
  settings,
  finishing,
  onFinish,
  onBack,
}: {
  settings: AppSettings;
  finishing: boolean;
  onFinish: () => void;
  onBack: () => void;
}) {
  const provider = settings.llm.provider;
  const providerLabel =
    provider === "mlx"
      ? "On-device (MLX)"
      : provider === "openai"
        ? "OpenAI"
        : provider === "exo"
          ? "Cluster"
          : settings.llm.anthropic.model_provider === "bedrock"
            ? "AWS Bedrock"
            : "Anthropic";
  const modelLabel =
    provider === "mlx"
      ? settings.llm.mlx.hf_llm_model_id || "(not set)"
      : provider === "exo"
        ? settings.exo?.model_name || "(configure in Settings → LLM → Cluster)"
        : provider === "openai"
          ? settings.llm.openai.model_name || "(not set)"
          : settings.llm.anthropic.model_name || "(not set)";

  return (
    <div className="space-y-8 text-center">
      <div className="space-y-3">
        <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-emerald-500/15 text-emerald-500">
          <CheckCircle2 size={28} />
        </div>
        <h1 className="text-2xl font-semibold text-th-text-primary tracking-tight">
          You're ready.
        </h1>
        <p className="text-sm text-th-text-tertiary max-w-md mx-auto">
          Everything below is editable from Settings whenever you want.
        </p>
      </div>

      <div className="rounded-2xl border border-th-border bg-th-surface p-5 text-left max-w-md mx-auto space-y-3 text-sm">
        <SummaryRow label="Provider" value={providerLabel} />
        <SummaryRow label="Model" value={modelLabel} mono />
        <SummaryRow
          label="Agent Memory"
          value={settings.memory?.enabled ? "Enabled — local summariser" : "Disabled"}
          good={settings.memory?.enabled}
        />
        <SummaryRow
          label="macOS Activity"
          value={settings.activity?.enabled ? "Enabled" : "Disabled"}
          good={settings.activity?.enabled}
        />
        <SummaryRow
          label="Auto-evaluate"
          value={settings.evaluation?.auto_evaluate ? "Enabled" : "Disabled"}
          good={settings.evaluation?.auto_evaluate}
        />
      </div>

      <div className="flex items-center justify-center gap-3 pt-2">
        <button
          type="button"
          onClick={onBack}
          disabled={finishing}
          className="px-3 py-2 text-xs font-medium text-th-text-tertiary hover:text-th-text-primary inline-flex items-center gap-1.5 disabled:opacity-40"
        >
          <ChevronLeft size={12} /> Back
        </button>
        <button
          type="button"
          onClick={onFinish}
          disabled={finishing}
          className="px-5 py-2.5 text-sm font-semibold rounded-lg bg-emerald-600 text-white hover:bg-emerald-500 inline-flex items-center gap-2 disabled:opacity-40"
        >
          {finishing ? <Loader2 size={14} className="animate-spin" /> : <ChevronRight size={14} />}
          {finishing ? "Opening…" : "Open Otto"}
        </button>
      </div>
    </div>
  );
}

function SummaryRow({
  label,
  value,
  mono,
  good,
}: {
  label: string;
  value: string;
  mono?: boolean;
  good?: boolean;
}) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="text-[11px] uppercase tracking-wider text-th-text-muted shrink-0">
        {label}
      </span>
      <span
        className={classNames(
          "text-xs text-th-text-primary truncate",
          mono && "font-mono",
          good && "text-emerald-500",
        )}
        title={value}
      >
        {value}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Shared UI primitives
// ---------------------------------------------------------------------------

function ScreenHeader({
  icon,
  title,
  subtitle,
}: {
  icon?: React.ReactNode;
  title: string;
  subtitle?: string;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2">
        {icon && (
          <span className="inline-flex items-center justify-center w-7 h-7 rounded-lg bg-th-tab-active-bg/15 text-th-tab-active-bg">
            {icon}
          </span>
        )}
        <h2 className="text-lg font-semibold text-th-text-primary">{title}</h2>
      </div>
      {subtitle && (
        <p className="text-xs text-th-text-tertiary leading-relaxed">
          {subtitle}
        </p>
      )}
    </div>
  );
}

function ToggleRow({
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
    <label className="flex items-start justify-between gap-4 cursor-pointer">
      <div className="flex-1 min-w-0">
        <span className="block text-sm font-medium text-th-text-primary">
          {label}
        </span>
        {description && (
          <span className="block text-[11px] text-th-text-tertiary mt-0.5 leading-relaxed">
            {description}
          </span>
        )}
      </div>
      <span
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={classNames(
          "relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200",
          checked ? "bg-blue-500" : "bg-neutral-600/50",
        )}
      >
        <span
          className={classNames(
            "pointer-events-none inline-block h-5 w-5 transform rounded-full shadow ring-0 transition-all duration-200",
            checked ? "translate-x-5 bg-white" : "translate-x-0 bg-neutral-400",
          )}
        />
      </span>
    </label>
  );
}

function ReassurancePills({
  pills,
}: {
  pills: { icon: React.ReactNode; label: string }[];
}) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {pills.map((p) => (
        <span
          key={p.label}
          className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-th-inset-bg border border-th-border text-[11px] font-medium text-th-text-secondary"
        >
          <span className="text-emerald-500">{p.icon}</span>
          {p.label}
        </span>
      ))}
    </div>
  );
}

function FooterBar({
  onBack,
  onContinue,
  continueDisabled,
  continueLabel = "Continue",
  backLabel = "Back",
}: {
  onBack: () => void;
  onContinue?: () => void;
  continueDisabled?: boolean;
  continueLabel?: string;
  backLabel?: string;
}) {
  return (
    <div className="flex items-center justify-between pt-2">
      <button
        type="button"
        onClick={onBack}
        className="px-3 py-2 text-xs font-medium text-th-text-tertiary hover:text-th-text-primary inline-flex items-center gap-1.5"
      >
        <ChevronLeft size={12} /> {backLabel}
      </button>
      {onContinue && (
        <button
          type="button"
          onClick={onContinue}
          disabled={continueDisabled}
          className="px-4 py-2 text-xs font-semibold rounded-lg bg-th-tab-active-bg text-th-tab-active-fg disabled:opacity-40 hover:opacity-90 inline-flex items-center gap-1.5 transition-opacity"
        >
          {continueLabel} <ChevronRight size={12} />
        </button>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Skip confirmation dialog
// ---------------------------------------------------------------------------

function SkipDialog({
  onCancel,
  onConfirm,
  busy,
}: {
  onCancel: () => void;
  onConfirm: () => void;
  busy: boolean;
}) {
  return (
    <div className="fixed inset-0 z-[110] flex items-center justify-center bg-black/40 backdrop-blur-sm animate-fade-in">
      <div className="bg-th-bg-secondary border border-th-border rounded-2xl shadow-xl max-w-md w-full mx-4 p-6 space-y-4">
        <div className="flex items-start gap-3">
          <div className="w-10 h-10 rounded-xl bg-amber-500/15 text-amber-500 flex items-center justify-center shrink-0">
            <AlertTriangle size={18} />
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="text-sm font-semibold text-th-text-primary">
              Skip setup?
            </h3>
            <p className="text-xs text-th-text-tertiary mt-1 leading-relaxed">
              You can configure your model, memory, and activity tracking
              anytime from <span className="font-medium">Settings</span>. Otto
              won't be functional until you connect a model.
            </p>
          </div>
          <button
            type="button"
            onClick={onCancel}
            className="text-th-text-muted hover:text-th-text-primary"
          >
            <X size={16} />
          </button>
        </div>
        <div className="flex items-center justify-end gap-2 pt-1">
          <button
            type="button"
            onClick={onCancel}
            disabled={busy}
            className="px-3 py-1.5 text-xs font-medium rounded-md border border-th-border bg-th-surface text-th-text-secondary hover:bg-th-surface-hover disabled:opacity-40"
          >
            Keep setting up
          </button>
          <button
            type="button"
            onClick={onConfirm}
            disabled={busy}
            className="px-3 py-1.5 text-xs font-semibold rounded-md bg-amber-600 text-white hover:bg-amber-500 inline-flex items-center gap-1.5 disabled:opacity-40"
          >
            {busy ? <Loader2 size={11} className="animate-spin" /> : null}
            Skip for now
          </button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// VoiceSetupScreen — optional voice configuration step
// ---------------------------------------------------------------------------

function VoiceToggle({ enabled, onToggle }: { enabled: boolean; onToggle: () => void }) {
  return (
    <button
      role="switch"
      aria-checked={enabled}
      onClick={onToggle}
      className={`relative inline-flex h-6 w-11 flex-shrink-0 items-center rounded-full transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 ${
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

type VoiceProvider = "mlx" | "omlx" | "exo" | "cloud";

function voiceProviderType(provider: string): VoiceProvider {
  if (provider === "mlx") return "mlx";
  if (provider === "omlx") return "omlx";
  if (provider === "exo") return "exo";
  return "cloud";
}

const VOICE_PROVIDER_META: Record<VoiceProvider, {
  badge: string;
  badgeColor: string;
  privacyNote: string;
  headerNote: string;
}> = {
  mlx: {
    badge: "On-device · MLX",
    badgeColor: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
    privacyNote: "Whisper STT and your LLM — all on your Mac. Zero cloud dependency.",
    headerNote: "Speak to Otto using on-device Whisper. Your voice never leaves your Mac.",
  },
  omlx: {
    badge: "On-device · oMLX",
    badgeColor: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
    privacyNote: "Whisper STT and your LLM — all on your Mac. Zero cloud dependency.",
    headerNote: "Speak to Otto using on-device Whisper. Your voice never leaves your Mac.",
  },
  exo: {
    badge: "Cluster",
    badgeColor: "bg-blue-500/15 text-blue-400 border-blue-500/30",
    privacyNote: "Whisper STT runs on your Mac. Only transcribed text is sent to the cluster.",
    headerNote: "Whisper STT runs on your Mac — only the transcribed text travels to your cluster.",
  },
  cloud: {
    badge: "Cloud LLM",
    badgeColor: "bg-amber-500/15 text-amber-400 border-amber-500/30",
    privacyNote: "Whisper STT is always on-device. Only the transcribed text is sent to your cloud LLM.",
    headerNote: "Whisper STT is 100% on-device — only the transcribed text reaches your cloud provider.",
  },
};

function SectionNumber({ n }: { n: number }) {
  return (
    <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-th-bg-secondary border border-th-border text-[10px] font-bold text-th-text-secondary shrink-0">
      {n}
    </span>
  );
}

function VoiceSetupScreen({
  settings,
  onPatch,
  onBack,
  onContinue,
}: {
  settings: AppSettings;
  onPatch: (patch: Partial<AppSettings>) => Promise<void>;
  onBack: () => void;
  onContinue: () => void;
}) {
  const voice = settings.voice ?? {} as VoiceConfig;
  const providerType = voiceProviderType(settings.llm?.provider ?? "anthropic");
  const meta = VOICE_PROVIDER_META[providerType];
  const hfToken = settings.llm?.mlx?.hf_token ?? "";

  const [voiceEnabled, setVoiceEnabled] = useState(voice.enabled ?? false);
  const [sttModel, setSttModel] = useState(voice.stt_model || "mlx-community/whisper-large-v3-turbo");
  const [saving, setSaving] = useState(false);

  const handleContinue = async () => {
    setSaving(true);
    try {
      await onPatch({
        voice: {
          ...voice,
          enabled: voiceEnabled,
          stt_model: sttModel,
          stt_enabled: true,
        },
      });
    } finally {
      setSaving(false);
    }
    onContinue();
  };

  return (
    <div className="space-y-5 max-w-lg mx-auto w-full">

      {/* ── Header ─────────────────────────────────────────────── */}
      <div>
        <div className="flex items-center gap-3 mb-2">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-blue-500/15 border border-blue-500/20">
            <Mic size={18} className="text-blue-400" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-th-text-primary leading-none">Voice</h2>
            <span className={`inline-flex items-center gap-1 mt-1 text-[10px] font-medium px-1.5 py-0.5 rounded-full border ${meta.badgeColor}`}>
              {meta.badge}
            </span>
          </div>
        </div>
        <p className="text-sm text-th-text-secondary leading-relaxed">{meta.headerNote}</p>
      </div>

      {/* ── Enable voice toggle ─────────────────────────────────── */}
      <button
        onClick={() => setVoiceEnabled((v) => !v)}
        className={`w-full flex items-center justify-between rounded-xl border p-4 text-left transition-all duration-150 ${
          voiceEnabled
            ? "border-blue-500/40 bg-blue-500/[0.07] ring-1 ring-blue-500/20"
            : "border-th-border bg-th-surface hover:border-th-border-hover"
        }`}
      >
        <div className="flex items-center gap-3">
          <div className={`flex h-8 w-8 items-center justify-center rounded-lg ${voiceEnabled ? "bg-blue-500/20" : "bg-th-bg-secondary"}`}>
            <Mic size={15} className={voiceEnabled ? "text-blue-400" : "text-th-text-muted"} />
          </div>
          <div>
            <p className="text-sm font-semibold text-th-text-primary">Enable Voice Mode</p>
            <p className="text-xs text-th-text-secondary mt-0.5">
              Adds a mic button to chat. Hold or click to dictate messages using Whisper STT.
            </p>
          </div>
        </div>
        <VoiceToggle enabled={voiceEnabled} onToggle={() => {}} />
      </button>

      {/* ── STT model chooser (only when enabled) ──────────────── */}
      {voiceEnabled && (
        <div className="space-y-5">
          <div className="rounded-xl border border-th-border bg-th-surface overflow-hidden">
            <div className="flex items-center gap-2 px-4 pt-3 pb-2 border-b border-th-border/60">
              <SectionNumber n={1} />
              <div className="flex-1 min-w-0">
                <p className="text-xs font-semibold text-th-text-primary">Speech-to-text model</p>
                <p className="text-[11px] text-th-text-secondary mt-0.5">
                  Whisper runs on-device. Your voice is transcribed privately on your Mac.
                </p>
              </div>
            </div>
            <div className="p-4">
              <VoiceModelChooser
                config={{ stt_model: sttModel }}
                onSelectStt={(id) => setSttModel(id)}
                kinds={["stt"]}
                hfToken={hfToken}
              />
            </div>
          </div>

          {/* ── Privacy callout ──────────────────────────────── */}
          <div className={`flex items-start gap-2.5 rounded-xl border px-3 py-2.5 ${
            providerType === "cloud"
              ? "border-amber-500/20 bg-amber-500/[0.04]"
              : "border-emerald-500/20 bg-emerald-500/[0.04]"
          }`}>
            <ShieldCheck size={14} className={`mt-0.5 shrink-0 ${providerType === "cloud" ? "text-amber-400" : "text-emerald-400"}`} />
            <p className={`text-[11px] leading-relaxed ${providerType === "cloud" ? "text-amber-300/80" : "text-emerald-300/80"}`}>
              <span className={`font-semibold ${providerType === "cloud" ? "text-amber-300" : "text-emerald-300"}`}>
                {providerType === "cloud" ? "STT stays on-device." : "Fully private."}
              </span>{" "}
              {meta.privacyNote}
            </p>
          </div>
        </div>
      )}

      {/* ── Footer ─────────────────────────────────────────────── */}
      <div className="flex items-center justify-between pt-1">
        <button
          onClick={onBack}
          className="flex items-center gap-1.5 text-sm text-th-text-secondary hover:text-th-text-primary transition-colors"
        >
          <ChevronLeft size={16} /> Back
        </button>
        <div className="flex items-center gap-3">
          {!voiceEnabled && (
            <button
              onClick={onContinue}
              className="text-xs text-th-text-secondary hover:text-th-text-primary underline-offset-2 hover:underline transition-colors"
            >
              Skip for now
            </button>
          )}
          <button
            onClick={() => void handleContinue()}
            disabled={saving}
            className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium disabled:opacity-50 transition-colors"
          >
            {saving ? <Loader2 size={14} className="animate-spin" /> : null}
            {voiceEnabled ? "Save & continue" : "Continue"}
            <ChevronRight size={16} />
          </button>
        </div>
      </div>
    </div>
  );
}
