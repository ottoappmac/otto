import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Check,
  CheckCircle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  ClipboardCheck,
  Copy,
  Edit3,
  FileSearch,
  GitBranch,
  Globe,
  History,
  Loader2,
  Play,
  Plus,
  RefreshCw,
  Sparkles,
  Square,
  Terminal,
  TestTube,
  Trash2,
  X,
  XCircle,
  Zap,
} from "lucide-react";
import { api } from "../hooks/useApi";
import { usePolling } from "../hooks/usePolling";
import { useNotification } from "../context/NotificationContext";
import { useEvalSuggestionsByTarget } from "../hooks/useAmbientHints";
import {
  MAX_TRIGGERS,
  MAX_POLL_SECONDS,
  MIN_POLL_SECONDS,
  type AgentSpec,
  type AppSettings,
  type FileOsWatch,
  type HookDefinition,
  type HttpMethod,
  type HttpMode,
  type OsaLanguage,
  type ShellMode,
  type TriggerSpec,
  type TriggerType,
  type AmbientHint,
} from "../types";

const AUTO_SAVE_DELAY_MS = 800;
const STATUS_POLL_MS = 5_000;

// ── Types ────────────────────────────────────────────────────────────────────

type ClaudeHookStatus = {
  enabled: boolean;
  active_sessions: number;
  auto_monitor: { enabled: boolean; active: Record<string, string> };
};

type OpenClawStatus = {
  enabled: boolean;
  running: boolean;
  poll_interval?: number;
  tracked_agents?: number;
  tracked_sessions?: number;
  buffered_events?: number;
  auto_monitor?: { active: Record<string, string>; count: number };
  error?: string;
};

type SaveStatus = "idle" | "saving" | "saved" | "error";

// ── Defaults ─────────────────────────────────────────────────────────────────

const DEFAULT_CLAUDE_HOOK: AppSettings["claude_hook"] = {
  enabled: true,
  http_hooks_enabled: false,
  quality_gate_enabled: false,
  quality_gate_threshold: 0.5,
  auto_monitor_enabled: false,
  max_auto_sessions: 3,
  auto_monitor_agent: "claude-session-eval-agent",
  hooks: [],
};

const DEFAULT_OPENCLAW: AppSettings["openclaw"] = {
  enabled: false,
  mode: "local",
  state_dir: "~/.openclaw",
  ssh_host: "",
  ssh_user: "ubuntu",
  ssh_key_path: "",
  ssh_port: 22,
  watcher_enabled: false,
  watcher_poll_interval: 10,
  auto_monitor_enabled: false,
  max_auto_sessions: 3,
  auto_monitor_agent: "openclaw-session-eval-agent",
};

// ── Page ─────────────────────────────────────────────────────────────────────

export default function TriggersPage() {
  const navigate = useNavigate();
  // Settings state
  const [fullSettings, setFullSettings] = useState<AppSettings | null>(null);
  const [claudeHook, setClaudeHook] = useState(DEFAULT_CLAUDE_HOOK);
  const [openclaw, setOpenclaw] = useState(DEFAULT_OPENCLAW);
  const [saveStatus, setSaveStatus] = useState<SaveStatus>("idle");

  // Agent list for auto-monitor selector
  const [agents, setAgents] = useState<AgentSpec[]>([]);

  // Live status
  const [claudeStatus, setClaudeStatus] = useState<ClaudeHookStatus | null>(null);
  const [openclawStatus, setOpenclawStatus] = useState<OpenClawStatus | null>(null);
  const [statusLoading, setStatusLoading] = useState(true);

  // Action state
  const [claudeInstalling, setClaudeInstalling] = useState(false);
  const [claudeInstallResult, setClaudeInstallResult] = useState<{ status: string; message: string } | null>(null);
  const [openclawTesting, setOpenclawTesting] = useState(false);
  const [openclawTestResult, setOpenclawTestResult] = useState<{ success: boolean; message: string } | null>(null);
  const [claudeConfigOpen, setClaudeConfigOpen] = useState(false);
  const [openclawConfigOpen, setOpenclawConfigOpen] = useState(false);

  // Triggers (custom + managed)
  const [triggers, setTriggers] = useState<TriggerSpec[]>([]);
  const [triggerModalOpen, setTriggerModalOpen] = useState(false);
  const [triggerEditing, setTriggerEditing] = useState<TriggerSpec | null>(null);
  const [triggerError, setTriggerError] = useState<string | null>(null);
  const [triggerBusy, setTriggerBusy] = useState<string | null>(null);

  const evalSuggestions = useEvalSuggestionsByTarget();

  const { clearTriggerNotifications } = useNotification();
  useEffect(() => {
    clearTriggerNotifications();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const customTriggers = triggers.filter((t) => !t.builtin);
  const managedTriggers = triggers.filter((t) => t.builtin);

  const refreshTriggers = useCallback(async () => {
    try {
      const list = await api.listTriggers();
      setTriggers(list);
    } catch (e) {
      console.warn("Failed to load triggers:", e);
    }
  }, []);

  useEffect(() => {
    refreshTriggers();
  }, [refreshTriggers]);

  usePolling(refreshTriggers, STATUS_POLL_MS);

  const loadedRef = useRef(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout>>();
  const savedTimerRef = useRef<ReturnType<typeof setTimeout>>();

  // Load settings and agents on mount
  useEffect(() => {
    Promise.all([api.getSettings(), api.listAgents()])
      .then(([s, agentList]) => {
        setFullSettings(s);
        setClaudeHook({ ...DEFAULT_CLAUDE_HOOK, ...(s.claude_hook ?? {}) });
        setOpenclaw({ ...DEFAULT_OPENCLAW, ...(s.openclaw ?? {}) });
        setAgents(agentList);
        loadedRef.current = true;
      })
      .catch((e) => console.warn("Failed to load settings:", e));
  }, []);

  // Debounced settings save
  const persistSettings = useCallback(
    async (hook: AppSettings["claude_hook"], claw: AppSettings["openclaw"]) => {
      if (!fullSettings) return;
      setSaveStatus("saving");
      try {
        await api.updateSettings({
          ...fullSettings,
          claude_hook: hook,
          openclaw: claw,
        } as unknown as Record<string, unknown>);
        setSaveStatus("saved");
        savedTimerRef.current = setTimeout(() => setSaveStatus("idle"), 2000);
      } catch {
        setSaveStatus("error");
      }
    },
    [fullSettings],
  );

  useEffect(() => {
    if (!loadedRef.current) return;
    clearTimeout(debounceRef.current);
    clearTimeout(savedTimerRef.current);
    debounceRef.current = setTimeout(
      () => persistSettings(claudeHook, openclaw),
      AUTO_SAVE_DELAY_MS,
    );
    return () => clearTimeout(debounceRef.current);
  }, [claudeHook, openclaw, persistSettings]);

  // Live status polling
  const refreshStatus = useCallback(async () => {
    try {
      const [cs, ocs] = await Promise.all([
        api.claudeHookStatus().catch(() => null),
        api.openclawStatus().catch(() => null),
      ]);
      setClaudeStatus(cs);
      setOpenclawStatus(ocs);
    } finally {
      setStatusLoading(false);
    }
  }, []);

  usePolling(refreshStatus, STATUS_POLL_MS);

  // Handlers
  const handleInstallClaudeHooks = async () => {
    setClaudeInstalling(true);
    setClaudeInstallResult(null);
    try {
      const result = await api.installClaudeHooks();
      setClaudeInstallResult(result);
      setTimeout(() => setClaudeInstallResult(null), 6000);
    } catch (e) {
      setClaudeInstallResult({ status: "error", message: e instanceof Error ? e.message : "Unknown error" });
    } finally {
      setClaudeInstalling(false);
    }
  };

  const handleTestOpenClaw = async () => {
    setOpenclawTesting(true);
    setOpenclawTestResult(null);
    try {
      const result = await api.testOpenClawConnection();
      setOpenclawTestResult(result);
      setTimeout(() => setOpenclawTestResult(null), 6000);
    } catch (e) {
      setOpenclawTestResult({ success: false, message: e instanceof Error ? e.message : "Unknown error" });
    } finally {
      setOpenclawTesting(false);
    }
  };

  // Derived status
  const claudeReceiving = claudeStatus?.enabled ?? false;
  const claudeAutoMonitorCount = claudeStatus
    ? Object.keys(claudeStatus.auto_monitor?.active ?? {}).length
    : 0;

  const openclawRunning = openclawStatus?.running ?? false;
  const openclawEnabled = openclawStatus?.enabled ?? openclaw.enabled;

  return (
    <div className="h-full flex flex-col">
      <header className="border-b border-th-border px-6 py-4 flex items-center justify-between shrink-0 bg-th-bg-secondary">
        <h1 className="text-lg font-bold text-th-text-primary">Triggers</h1>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 text-xs font-medium min-w-[80px] justify-end">
            {saveStatus === "saving" && (
              <><Loader2 size={12} className="animate-spin text-th-text-muted" /><span className="text-th-text-muted">Saving…</span></>
            )}
            {saveStatus === "saved" && (
              <><Check size={12} className="text-emerald-400" /><span className="text-emerald-400">Saved</span></>
            )}
            {saveStatus === "error" && (
              <><XCircle size={12} className="text-red-400" /><span className="text-red-400">Save failed</span></>
            )}
          </div>
          <button
            className="px-3 py-2 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-sm font-medium transition-colors flex items-center gap-1.5"
            onClick={refreshStatus}
          >
            <RefreshCw size={14} /> Refresh
          </button>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto p-6">
        {statusLoading && !claudeStatus && !openclawStatus && (
          <div className="flex items-center justify-center h-32">
            <Loader2 size={24} className="text-th-text-muted animate-spin" />
          </div>
        )}

        <div className="space-y-4">

          {/* Custom section divider */}
          <div className="flex items-center gap-3">
            <div className="flex-1 border-t border-th-border" />
            <span className="text-sm uppercase tracking-widest text-th-text-muted font-semibold flex items-center gap-1.5">
              <Zap size={14} /> Custom
            </span>
            <div className="flex-1 border-t border-th-border" />
          </div>

          <CustomTriggersSection
            triggers={customTriggers}
            agents={agents}
            busyId={triggerBusy}
            evalSuggestions={evalSuggestions}
            onCreate={() => { setTriggerEditing(null); setTriggerError(null); setTriggerModalOpen(true); }}
            onEdit={(t) => { setTriggerEditing(t); setTriggerError(null); setTriggerModalOpen(true); }}
            onViewRuns={(id) => navigate(`/triggers/${encodeURIComponent(id)}/runs`)}
            onToggle={async (id) => {
              setTriggerBusy(id);
              try { await api.toggleTrigger(id); await refreshTriggers(); }
              finally { setTriggerBusy(null); }
            }}
            onRunNow={async (id) => {
              setTriggerBusy(id);
              try { await api.runTriggerNow(id); await refreshTriggers(); }
              finally { setTriggerBusy(null); }
            }}
            onDelete={async (id) => {
              setTriggerBusy(id);
              try { await api.deleteTrigger(id); await refreshTriggers(); }
              finally { setTriggerBusy(null); }
            }}
          />

          {triggerModalOpen && (
            <TriggerModal
              initial={triggerEditing}
              agents={agents}
              error={triggerError}
              onClose={() => setTriggerModalOpen(false)}
              onSubmit={async (req, isUpdate) => {
                setTriggerError(null);
                try {
                  if (isUpdate && triggerEditing) {
                    await api.updateTrigger(triggerEditing.id, req);
                  } else {
                    await api.createTrigger(req);
                  }
                  setTriggerModalOpen(false);
                  await refreshTriggers();
                } catch (e) {
                  const msg = e instanceof Error ? e.message : "Unknown error";
                  setTriggerError(msg);
                }
              }}
            />
          )}

          {/* Managed section divider */}
          <div className="flex items-center gap-3 pt-4">
            <div className="flex-1 border-t border-th-border" />
            <span className="text-sm uppercase tracking-widest text-th-text-muted font-semibold flex items-center gap-1.5">
              <Zap size={14} /> Managed
            </span>
            <div className="flex-1 border-t border-th-border" />
          </div>

          {/* ── Managed trigger catalog ──────────────────────────── */}
          {managedTriggers.length > 0 && (
            <ManagedTriggersSection
              triggers={managedTriggers}
              busyId={triggerBusy}
              evalSuggestions={evalSuggestions}
              onEdit={(t) => { setTriggerEditing(t); setTriggerError(null); setTriggerModalOpen(true); }}
              onViewRuns={(id) => navigate(`/triggers/${encodeURIComponent(id)}/runs`)}
              onToggle={async (id) => {
                setTriggerBusy(id);
                try { await api.toggleTrigger(id); await refreshTriggers(); }
                finally { setTriggerBusy(null); }
              }}
              onRunNow={async (id) => {
                setTriggerBusy(id);
                try { await api.runTriggerNow(id); await refreshTriggers(); }
                finally { setTriggerBusy(null); }
              }}
            />
          )}

          {/* ── Claude Hook card ─────────────────────────────────── */}
          <div className="bg-th-card-bg border border-th-card-border rounded-xl p-5 hover:border-th-border-strong transition-all duration-200">
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-3">
                <div className={`w-10 h-10 rounded-lg flex items-center justify-center shrink-0 ${claudeReceiving ? "bg-emerald-500/15 border border-emerald-500/20" : "bg-th-inset-bg border border-th-border"}`}>
                  <Zap size={20} className={claudeReceiving ? "text-emerald-400" : "text-th-text-muted"} />
                </div>
                <div>
                  <h3 className="font-semibold text-th-text-primary">Claude Hook</h3>
                  <p className="text-xs text-th-text-tertiary mt-0.5">
                    {claudeReceiving
                      ? `Receiving — ${claudeStatus?.active_sessions ?? 0} active session${(claudeStatus?.active_sessions ?? 0) !== 1 ? "s" : ""}${claudeAutoMonitorCount > 0 ? `, ${claudeAutoMonitorCount} auto-monitor` : ""}`
                      : claudeHook.enabled
                      ? "Enabled — HTTP hook receiver off"
                      : "Disabled — not receiving events"}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {claudeReceiving ? (
                  <span className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full font-medium bg-emerald-500/15 text-emerald-400 border border-emerald-500/25">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                    Receiving
                  </span>
                ) : claudeHook.enabled ? (
                  <span className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full font-medium bg-sky-500/10 text-sky-400 border border-sky-500/25">
                    <span className="w-1.5 h-1.5 rounded-full bg-sky-400" />
                    Enabled
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full font-medium bg-th-inset-bg text-th-text-tertiary border border-th-border">
                    <span className="w-1.5 h-1.5 rounded-full bg-th-text-muted" />
                    Off
                  </span>
                )}
              </div>
            </div>

            {/* Live stats bar */}
            {claudeReceiving && claudeStatus && (
              <div className="mt-3 flex flex-wrap gap-2">
                <StatPill label="Active sessions" value={claudeStatus.active_sessions} />
                {claudeAutoMonitorCount > 0 && <StatPill label="Auto-monitor" value={claudeAutoMonitorCount} />}
              </div>
            )}

            {/* Install result inline */}
            {claudeInstallResult && (
              <div className={`mt-3 flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium ${claudeInstallResult.status === "ok" ? "bg-emerald-500/10 border border-emerald-500/20 text-emerald-400" : "bg-red-500/10 border border-red-500/20 text-red-400"}`}>
                {claudeInstallResult.status === "ok" ? <CheckCircle2 size={13} /> : <XCircle size={13} />}
                {claudeInstallResult.message}
              </div>
            )}

            {/* Actions */}
            <div className="mt-4 flex flex-wrap gap-2 pt-4 border-t border-th-border">
              {claudeHook.enabled ? (
                <button
                  className="px-3 py-1.5 bg-red-500/10 border border-red-500/20 text-red-400 hover:bg-red-500/20 rounded-lg text-xs font-medium transition-colors flex items-center gap-1.5"
                  onClick={() => setClaudeHook((h) => ({ ...h, enabled: false }))}
                >
                  <Square size={12} /> Disable
                </button>
              ) : (
                <button
                  className="px-3 py-1.5 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 hover:bg-emerald-500/20 rounded-lg text-xs font-medium transition-colors flex items-center gap-1.5"
                  onClick={() => setClaudeHook((h) => ({ ...h, enabled: true }))}
                >
                  <Play size={12} /> Enable
                </button>
              )}
              {claudeHook.enabled && !claudeHook.http_hooks_enabled && (
                <button
                  className="px-3 py-1.5 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 hover:bg-emerald-500/20 rounded-lg text-xs font-medium transition-colors flex items-center gap-1.5"
                  onClick={() => setClaudeHook((h) => ({ ...h, http_hooks_enabled: true }))}
                >
                  <Play size={12} /> Start Receiver
                </button>
              )}
              {claudeHook.enabled && claudeHook.http_hooks_enabled && (
                <button
                  className="px-3 py-1.5 bg-red-500/10 border border-red-500/20 text-red-400 hover:bg-red-500/20 rounded-lg text-xs font-medium transition-colors flex items-center gap-1.5"
                  onClick={() => setClaudeHook((h) => ({ ...h, http_hooks_enabled: false }))}
                >
                  <Square size={12} /> Stop Receiver
                </button>
              )}
              <button
                className="px-3 py-1.5 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-xs font-medium transition-colors flex items-center gap-1.5 disabled:opacity-50"
                onClick={handleInstallClaudeHooks}
                disabled={claudeInstalling}
              >
                {claudeInstalling ? <Loader2 size={12} className="animate-spin" /> : <CheckCircle size={12} />}
                {claudeInstalling ? "Installing…" : "Install to ~/.claude"}
              </button>
              <button
                className="px-3 py-1.5 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-xs font-medium transition-colors flex items-center gap-1.5"
                onClick={() => setClaudeConfigOpen((v) => !v)}
              >
                Configure
                <ChevronRight size={12} className={`transition-transform ${claudeConfigOpen ? "rotate-90" : ""}`} />
              </button>
            </div>

            {/* Configuration panel */}
            {claudeConfigOpen && (
              <div className="mt-4 pt-4 border-t border-th-border space-y-4">

                <div className="border-t border-th-border pt-4">
                  <p className="text-xs text-th-text-muted font-semibold uppercase tracking-wider mb-3">HTTP Hook Receiver</p>
                  <Toggle
                    label="Enable HTTP Hook Receiver"
                    checked={claudeHook.http_hooks_enabled}
                    onChange={(v) => setClaudeHook((h) => ({ ...h, http_hooks_enabled: v }))}
                  />
                  <p className="text-xs text-th-text-tertiary mt-1.5">Receives events directly from Claude Code via HTTP — eliminates polling delay for live session monitoring.</p>

                  {claudeHook.http_hooks_enabled && (
                    <div className="mt-3">
                      <ClaudeHookSnippet customHooks={claudeHook.hooks} />
                    </div>
                  )}
                </div>

                <div className="border-t border-th-border pt-4">
                  <p className="text-xs text-th-text-muted font-semibold uppercase tracking-wider mb-3">Quality Gate <span className="normal-case font-normal text-th-text-tertiary">(experimental)</span></p>
                  <Toggle
                    label="Enable Quality Gate at Stop"
                    checked={claudeHook.quality_gate_enabled}
                    onChange={(v) => setClaudeHook((h) => ({ ...h, quality_gate_enabled: v }))}
                  />
                  <p className="text-xs text-th-text-tertiary mt-1.5">When Claude finishes, check tool error rate. If above threshold, Claude is told to keep working.</p>
                  {claudeHook.quality_gate_enabled && (
                    <div className="mt-3 space-y-2">
                      <label className="block text-xs text-th-text-tertiary">
                        Min success rate: {Math.round(claudeHook.quality_gate_threshold * 100)}%
                      </label>
                      <input
                        type="range" min="0" max="100" step="5"
                        value={Math.round(claudeHook.quality_gate_threshold * 100)}
                        onChange={(e) => setClaudeHook((h) => ({ ...h, quality_gate_threshold: parseInt(e.target.value) / 100 }))}
                        className="w-full accent-emerald-500"
                      />
                      <p className="text-xs text-amber-400/80">Adds ~1-3s latency to every Claude Code stop event.</p>
                    </div>
                  )}
                </div>

                <div className="border-t border-th-border pt-4">
                  <p className="text-xs text-th-text-muted font-semibold uppercase tracking-wider mb-3">Auto-Monitor</p>
                  <Toggle
                    label="Auto-start eval session on new Claude session"
                    checked={claudeHook.auto_monitor_enabled}
                    onChange={(v) => setClaudeHook((h) => ({ ...h, auto_monitor_enabled: v }))}
                  />
                  <p className="text-xs text-th-text-tertiary mt-1.5">Automatically creates an eval agent when Claude Code starts a new session.</p>
                  {claudeHook.auto_monitor_enabled && (
                    <div className="mt-3 space-y-3">
                      <div className="space-y-1">
                        <label className="block text-xs text-th-text-tertiary">
                          Max concurrent auto-sessions: {claudeHook.max_auto_sessions}
                        </label>
                        <input
                          type="range" min="1" max="10" step="1"
                          value={claudeHook.max_auto_sessions}
                          onChange={(e) => setClaudeHook((h) => ({ ...h, max_auto_sessions: parseInt(e.target.value) }))}
                          className="w-full accent-emerald-500"
                        />
                      </div>
                      <AgentSelect
                        label="Agent"
                        value={claudeHook.auto_monitor_agent}
                        agents={agents}
                        onChange={(v) => setClaudeHook((h) => ({ ...h, auto_monitor_agent: v }))}
                      />
                    </div>
                  )}
                </div>

                <div className="border-t border-th-border pt-4">
                  <p className="text-xs text-th-text-muted font-semibold uppercase tracking-wider mb-3">Custom Hooks</p>
                  <p className="text-xs text-th-text-tertiary mb-3">Inject prompts, gate tool calls, or add context. These are merged into the snippet above.</p>
                  <HooksEditor
                    hooks={claudeHook.hooks}
                    events={CLAUDE_HOOK_EVENTS}
                    onChange={(hooks) => setClaudeHook((h) => ({ ...h, hooks }))}
                  />
                </div>
              </div>
            )}
          </div>

          {/* ── OpenClaw card ────────────────────────────────────── */}
          <div className="bg-th-card-bg border border-th-card-border rounded-xl p-5 hover:border-th-border-strong transition-all duration-200">
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-3">
                <div className={`w-10 h-10 rounded-lg flex items-center justify-center shrink-0 ${openclawRunning ? "bg-emerald-500/15 border border-emerald-500/20" : openclawEnabled ? "bg-sky-500/10 border border-sky-500/20" : "bg-th-inset-bg border border-th-border"}`}>
                  <Zap size={20} className={openclawRunning ? "text-emerald-400" : openclawEnabled ? "text-sky-400" : "text-th-text-muted"} />
                </div>
                <div>
                  <h3 className="font-semibold text-th-text-primary">OpenClaw</h3>
                  <p className="text-xs text-th-text-tertiary mt-0.5">
                    {openclawRunning
                      ? `Watching — ${openclawStatus?.tracked_agents ?? 0} agent${(openclawStatus?.tracked_agents ?? 0) !== 1 ? "s" : ""}, ${openclawStatus?.tracked_sessions ?? 0} session${(openclawStatus?.tracked_sessions ?? 0) !== 1 ? "s" : ""}`
                      : openclaw.enabled
                      ? openclaw.mode === "ssh"
                        ? `Enabled — SSH to ${openclaw.ssh_host || "unconfigured"}`
                        : "Enabled — watcher stopped"
                      : "Disabled — not watching"}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {openclawRunning ? (
                  <span className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full font-medium bg-emerald-500/15 text-emerald-400 border border-emerald-500/25">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                    Watching
                  </span>
                ) : openclaw.enabled ? (
                  <span className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full font-medium bg-sky-500/10 text-sky-400 border border-sky-500/25">
                    <span className="w-1.5 h-1.5 rounded-full bg-sky-400" />
                    Enabled
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full font-medium bg-th-inset-bg text-th-text-tertiary border border-th-border">
                    <span className="w-1.5 h-1.5 rounded-full bg-th-text-muted" />
                    Off
                  </span>
                )}
              </div>
            </div>

            {/* Live stats */}
            {openclawStatus && openclaw.enabled && (
              <div className="mt-3 flex flex-wrap gap-2">
                {openclawRunning && (
                  <>
                    <StatPill label="Agents" value={openclawStatus.tracked_agents ?? 0} />
                    <StatPill label="Sessions" value={openclawStatus.tracked_sessions ?? 0} />
                    <StatPill label="Buffered events" value={openclawStatus.buffered_events ?? 0} />
                    {openclawStatus.poll_interval !== undefined && (
                      <StatPill label="Poll" value={`${openclawStatus.poll_interval}s`} />
                    )}
                  </>
                )}
                {(openclawStatus.auto_monitor?.count ?? 0) > 0 && (
                  <StatPill label="Auto-monitor" value={openclawStatus.auto_monitor!.count} />
                )}
              </div>
            )}

            {/* Test result inline */}
            {openclawTestResult && (
              <div className={`mt-3 flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium ${openclawTestResult.success ? "bg-emerald-500/10 border border-emerald-500/20 text-emerald-400" : "bg-red-500/10 border border-red-500/20 text-red-400"}`}>
                {openclawTestResult.success ? <CheckCircle2 size={13} /> : <XCircle size={13} />}
                {openclawTestResult.message}
              </div>
            )}

            {/* Actions */}
            <div className="mt-4 flex flex-wrap gap-2 pt-4 border-t border-th-border">
              {openclaw.enabled ? (
                <button
                  className="px-3 py-1.5 bg-red-500/10 border border-red-500/20 text-red-400 hover:bg-red-500/20 rounded-lg text-xs font-medium transition-colors flex items-center gap-1.5"
                  onClick={() => setOpenclaw((o) => ({ ...o, enabled: false, watcher_enabled: false }))}
                >
                  <Square size={12} /> Disable
                </button>
              ) : (
                <button
                  className="px-3 py-1.5 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 hover:bg-emerald-500/20 rounded-lg text-xs font-medium transition-colors flex items-center gap-1.5"
                  onClick={() => setOpenclaw((o) => ({ ...o, enabled: true }))}
                >
                  <Play size={12} /> Enable
                </button>
              )}
              {openclaw.enabled && !openclaw.watcher_enabled && (
                <button
                  className="px-3 py-1.5 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 hover:bg-emerald-500/20 rounded-lg text-xs font-medium transition-colors flex items-center gap-1.5 disabled:opacity-50"
                  disabled={openclaw.mode === "ssh" && !openclaw.ssh_host}
                  title={openclaw.mode === "ssh" && !openclaw.ssh_host ? "SSH host required" : undefined}
                  onClick={() => setOpenclaw((o) => ({ ...o, watcher_enabled: true }))}
                >
                  <Play size={12} /> Start Watcher
                </button>
              )}
              {openclaw.enabled && openclaw.watcher_enabled && (
                <button
                  className="px-3 py-1.5 bg-red-500/10 border border-red-500/20 text-red-400 hover:bg-red-500/20 rounded-lg text-xs font-medium transition-colors flex items-center gap-1.5"
                  onClick={() => setOpenclaw((o) => ({ ...o, watcher_enabled: false }))}
                >
                  <Square size={12} /> Stop Watcher
                </button>
              )}
              <button
                className="px-3 py-1.5 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-xs font-medium transition-colors flex items-center gap-1.5 disabled:opacity-50"
                onClick={handleTestOpenClaw}
                disabled={openclawTesting || (openclaw.mode === "ssh" && !openclaw.ssh_host)}
              >
                {openclawTesting ? <Loader2 size={12} className="animate-spin" /> : <TestTube size={12} />}
                {openclawTesting ? "Testing…" : "Test"}
              </button>
              <button
                className="px-3 py-1.5 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-xs font-medium transition-colors flex items-center gap-1.5"
                onClick={() => setOpenclawConfigOpen((v) => !v)}
              >
                Configure
                <ChevronRight size={12} className={`transition-transform ${openclawConfigOpen ? "rotate-90" : ""}`} />
              </button>
            </div>

            {/* Configuration panel */}
            {openclawConfigOpen && (
              <div className="mt-4 pt-4 border-t border-th-border space-y-4">

                <div>
                  <p className="text-xs text-th-text-muted font-semibold uppercase tracking-wider mb-3">Connection</p>
                  <div className="space-y-3">
                    <SelectField
                      label="Access Mode"
                      value={openclaw.mode}
                      onChange={(v) => setOpenclaw((o) => ({ ...o, mode: v as "local" | "ssh" }))}
                      options={[
                        { value: "local", label: "Local — read files from this machine" },
                        { value: "ssh", label: "SSH — read files from a remote host" },
                      ]}
                    />
                    <InputField
                      label="State Directory"
                      value={openclaw.state_dir}
                      onChange={(v) => setOpenclaw((o) => ({ ...o, state_dir: v }))}
                      placeholder="~/.openclaw"
                    />
                    {openclaw.mode === "ssh" && (
                      <div className="space-y-3 p-4 bg-th-inset-bg rounded-lg border border-th-border">
                        <p className="text-xs text-th-text-muted font-semibold uppercase tracking-wider">SSH</p>
                        <InputField label="Host" value={openclaw.ssh_host} onChange={(v) => setOpenclaw((o) => ({ ...o, ssh_host: v }))} placeholder="3.27.207.149" />
                        <InputField label="User" value={openclaw.ssh_user} onChange={(v) => setOpenclaw((o) => ({ ...o, ssh_user: v }))} placeholder="ubuntu" />
                        <InputField label="Key Path" value={openclaw.ssh_key_path} onChange={(v) => setOpenclaw((o) => ({ ...o, ssh_key_path: v }))} placeholder="~/.ssh/openclaw.pem" />
                        <InputField label="Port" value={String(openclaw.ssh_port)} onChange={(v) => setOpenclaw((o) => ({ ...o, ssh_port: parseInt(v) || 22 }))} type="number" />
                        {!openclaw.ssh_host && <p className="text-xs text-amber-400">SSH host is required.</p>}
                        {!openclaw.ssh_key_path && <p className="text-xs text-amber-400">SSH key path is required.</p>}
                      </div>
                    )}
                  </div>
                </div>

                <div className="border-t border-th-border pt-4">
                  <p className="text-xs text-th-text-muted font-semibold uppercase tracking-wider mb-3">Session Watcher</p>
                  <Toggle
                    label="Enable session watcher"
                    checked={openclaw.watcher_enabled}
                    onChange={(v) => setOpenclaw((o) => ({ ...o, watcher_enabled: v }))}
                  />
                  <p className="text-xs text-th-text-tertiary mt-1.5">Periodically scans the sessions directory and pushes events to the eval pipeline.</p>
                  {openclaw.watcher_enabled && (
                    <div className="mt-3 space-y-2">
                      <label className="block text-xs text-th-text-tertiary">
                        Poll interval: {openclaw.watcher_poll_interval}s
                      </label>
                      <input
                        type="range" min="5" max="60" step="5"
                        value={openclaw.watcher_poll_interval}
                        onChange={(e) => setOpenclaw((o) => ({ ...o, watcher_poll_interval: parseInt(e.target.value) }))}
                        className="w-full accent-emerald-500"
                      />
                    </div>
                  )}
                </div>

                <div className="border-t border-th-border pt-4">
                  <p className="text-xs text-th-text-muted font-semibold uppercase tracking-wider mb-3">Auto-Monitor</p>
                  <Toggle
                    label="Auto-start eval session on new OpenClaw session"
                    checked={openclaw.auto_monitor_enabled}
                    onChange={(v) => setOpenclaw((o) => ({ ...o, auto_monitor_enabled: v }))}
                  />
                  <p className="text-xs text-th-text-tertiary mt-1.5">Automatically creates an eval agent when a new OpenClaw session is detected.</p>
                  {openclaw.auto_monitor_enabled && (
                    <div className="mt-3 space-y-3">
                      <div className="space-y-1">
                        <label className="block text-xs text-th-text-tertiary">
                          Max concurrent auto-sessions: {openclaw.max_auto_sessions}
                        </label>
                        <input
                          type="range" min="1" max="10" step="1"
                          value={openclaw.max_auto_sessions}
                          onChange={(e) => setOpenclaw((o) => ({ ...o, max_auto_sessions: parseInt(e.target.value) }))}
                          className="w-full accent-emerald-500"
                        />
                      </div>
                      <AgentSelect
                        label="Agent"
                        value={openclaw.auto_monitor_agent}
                        agents={agents}
                        onChange={(v) => setOpenclaw((o) => ({ ...o, auto_monitor_agent: v }))}
                      />
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Constants ────────────────────────────────────────────────────────────────

const CLAUDE_HOOK_EVENTS = [
  "PreToolUse", "PostToolUse", "PostToolUseFailure",
  "Stop", "SubagentStart", "SubagentStop",
  "SessionStart", "SessionEnd",
  "BeforeShellExecution", "AfterShellExecution",
  "BeforeSubmitPrompt", "BeforeReadFile", "AfterFileEdit",
  "BeforeMCPExecution", "AfterMCPExecution",
  "PreCompact", "AfterAgentResponse", "AfterAgentThought",
] as const;

const SYSTEM_HOOKS: Record<string, Array<{ matcher?: string; hooks: Array<Record<string, unknown>> }>> = {
  PostToolUse: [{ matcher: "*", hooks: [{ type: "http", url: "http://localhost:18081/hooks/claude/post-tool-use", timeout: 10 }] }],
  PostToolUseFailure: [{ matcher: "*", hooks: [{ type: "http", url: "http://localhost:18081/hooks/claude/post-tool-use-failure", timeout: 10 }] }],
  Stop: [{ hooks: [{ type: "http", url: "http://localhost:18081/hooks/claude/stop", timeout: 10 }] }],
  SubagentStop: [{ hooks: [{ type: "http", url: "http://localhost:18081/hooks/claude/subagent-stop", timeout: 10 }] }],
  SessionStart: [{ hooks: [{ type: "command", command: "curl -sfS -X POST http://localhost:18081/hooks/claude/session-start -H 'Content-Type: application/json' -d \"$(jq -c '.')\" 2>/dev/null || true", timeout: 10 }] }],
  SessionEnd: [{ hooks: [{ type: "http", url: "http://localhost:18081/hooks/claude/session-end", timeout: 10 }] }],
};

function buildClaudeSnippet(customHooks: HookDefinition[], includeSystem: boolean): string {
  const groups: Record<string, Array<{ matcher?: string; hooks: Array<Record<string, unknown>> }>> = {};
  if (includeSystem) {
    for (const [event, defs] of Object.entries(SYSTEM_HOOKS)) {
      groups[event] = [...defs];
    }
  }
  for (const hook of customHooks) {
    if (!hook.enabled) continue;
    if (!groups[hook.event]) groups[hook.event] = [];
    const hookDef: Record<string, unknown> = { type: hook.type, timeout: hook.timeout };
    if (hook.type === "http") hookDef.url = hook.url;
    else if (hook.type === "command") hookDef.command = hook.command;
    else if (hook.type === "prompt") hookDef.prompt = hook.prompt;
    const group: { matcher?: string; hooks: Array<Record<string, unknown>> } = { hooks: [hookDef] };
    if (hook.matcher) group.matcher = hook.matcher;
    groups[hook.event].push(group);
  }
  return JSON.stringify({ hooks: groups }, null, 2);
}

// ── Sub-components ───────────────────────────────────────────────────────────

function StatPill({ label, value }: { label: string; value: number | string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-[11px] px-2 py-0.5 rounded-full bg-th-inset-bg border border-th-border text-th-text-tertiary font-medium">
      <span className="text-th-text-primary font-semibold">{value}</span>
      {label}
    </span>
  );
}

function ClaudeHookSnippet({ customHooks }: { customHooks: HookDefinition[] }) {
  const [copied, setCopied] = useState(false);
  const snippet = buildClaudeSnippet(customHooks, true);
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(snippet);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch { /* non-secure context */ }
  };
  return (
    <div className="space-y-2">
      <p className="text-xs text-th-text-tertiary">
        Add to <code className="bg-th-inset-bg px-1 py-0.5 rounded text-th-text-secondary">~/.claude/settings.json</code>:
      </p>
      <div className="relative group">
        <pre className="p-3 bg-th-inset-bg border border-th-border rounded-lg text-[11px] leading-relaxed text-th-text-secondary overflow-x-auto max-h-48">{snippet}</pre>
        <button
          type="button"
          onClick={handleCopy}
          className="absolute top-2 right-2 p-1.5 rounded-md bg-th-card-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary transition-colors opacity-0 group-hover:opacity-100"
        >
          {copied ? <ClipboardCheck size={13} /> : <Copy size={13} />}
        </button>
      </div>
      {copied && <p className="text-xs text-emerald-400">Copied to clipboard</p>}
    </div>
  );
}

function HooksEditor({ hooks, events, onChange }: {
  hooks: HookDefinition[];
  events: readonly string[];
  onChange: (hooks: HookDefinition[]) => void;
}) {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const addHook = () => {
    const h: HookDefinition = {
      id: Math.random().toString(36).slice(2, 10),
      event: events[0] ?? "PreToolUse",
      type: "prompt",
      url: "",
      command: "",
      prompt: "",
      matcher: "",
      timeout: 10,
      enabled: true,
    };
    onChange([...hooks, h]);
    setExpandedId(h.id);
  };

  const removeHook = (id: string) => {
    onChange(hooks.filter((h) => h.id !== id));
    if (expandedId === id) setExpandedId(null);
  };

  const updateHook = (id: string, patch: Partial<HookDefinition>) => {
    onChange(hooks.map((h) => (h.id === id ? { ...h, ...patch } : h)));
  };

  return (
    <div className="space-y-3">
      {hooks.length === 0 && (
        <p className="text-xs text-th-text-muted italic">No custom hooks configured.</p>
      )}
      {hooks.map((hook) => {
        const isExpanded = expandedId === hook.id;
        return (
          <div key={hook.id} className="border border-th-border rounded-lg bg-th-inset-bg overflow-hidden">
            <div
              className="flex items-center gap-2 px-3 py-2 cursor-pointer hover:bg-th-surface-hover transition-colors"
              onClick={() => setExpandedId(isExpanded ? null : hook.id)}
            >
              <ChevronRight size={14} className={`text-th-text-tertiary transition-transform ${isExpanded ? "rotate-90" : ""}`} />
              <span className={`text-xs font-mono px-1.5 py-0.5 rounded ${hook.type === "prompt" ? "bg-blue-500/15 text-blue-400" : hook.type === "command" ? "bg-blue-500/15 text-blue-400" : "bg-emerald-500/15 text-emerald-400"}`}>
                {hook.type}
              </span>
              <span className="text-sm text-th-text-primary font-medium">{hook.event}</span>
              {hook.matcher && <span className="text-xs text-th-text-muted font-mono truncate max-w-[120px]">({hook.matcher})</span>}
              <div className="flex-1" />
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); updateHook(hook.id, { enabled: !hook.enabled }); }}
                className={`w-7 h-4 rounded-full transition-all border flex-shrink-0 relative ${hook.enabled ? "bg-blue-600 border-blue-600" : "bg-th-inset-bg border-th-border"}`}
              >
                <span className={`absolute top-[1px] left-[1px] w-[12px] h-[12px] rounded-full transition-all ${hook.enabled ? "translate-x-[12px] bg-white" : "bg-neutral-500"}`} />
              </button>
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); removeHook(hook.id); }}
                className="text-th-text-muted hover:text-red-400 transition-colors p-0.5"
              >
                <Trash2 size={13} />
              </button>
            </div>
            {isExpanded && (
              <div className="px-3 pb-3 pt-1 space-y-3 border-t border-th-border bg-th-card-bg">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-xs text-th-text-tertiary mb-1">Event</label>
                    <div className="relative">
                      <select
                        className="w-full appearance-none px-3 py-1.5 pr-8 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary text-xs focus:outline-none focus:border-blue-400"
                        value={hook.event}
                        onChange={(e) => updateHook(hook.id, { event: e.target.value })}
                      >
                        {events.map((ev) => <option key={ev} value={ev}>{ev}</option>)}
                      </select>
                      <ChevronDown size={12} className="absolute right-2 top-1/2 -translate-y-1/2 text-th-text-muted pointer-events-none" />
                    </div>
                  </div>
                  <div>
                    <label className="block text-xs text-th-text-tertiary mb-1">Type</label>
                    <div className="relative">
                      <select
                        className="w-full appearance-none px-3 py-1.5 pr-8 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary text-xs focus:outline-none focus:border-blue-400"
                        value={hook.type}
                        onChange={(e) => updateHook(hook.id, { type: e.target.value as HookDefinition["type"] })}
                      >
                        <option value="prompt">Prompt</option>
                        <option value="command">Command</option>
                        <option value="http">HTTP</option>
                      </select>
                      <ChevronDown size={12} className="absolute right-2 top-1/2 -translate-y-1/2 text-th-text-muted pointer-events-none" />
                    </div>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-xs text-th-text-tertiary mb-1">Matcher <span className="text-th-text-muted">(regex, optional)</span></label>
                    <input
                      className="w-full px-3 py-1.5 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary placeholder-th-text-muted text-xs focus:outline-none focus:border-blue-400"
                      value={hook.matcher}
                      onChange={(e) => updateHook(hook.id, { matcher: e.target.value })}
                      placeholder="e.g. Shell|Write"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-th-text-tertiary mb-1">Timeout (sec)</label>
                    <input
                      type="number"
                      className="w-full px-3 py-1.5 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary text-xs focus:outline-none focus:border-blue-400"
                      value={hook.timeout}
                      onChange={(e) => updateHook(hook.id, { timeout: parseInt(e.target.value) || 10 })}
                    />
                  </div>
                </div>
                {hook.type === "prompt" && (
                  <div>
                    <label className="block text-xs text-th-text-tertiary mb-1">Prompt <span className="text-th-text-muted">(use $ARGUMENTS)</span></label>
                    <textarea
                      className="w-full px-3 py-2 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary placeholder-th-text-muted text-xs focus:outline-none focus:border-blue-400 min-h-[60px] resize-y"
                      value={hook.prompt}
                      onChange={(e) => updateHook(hook.id, { prompt: e.target.value })}
                      placeholder="Does this tool call look safe? $ARGUMENTS"
                      rows={3}
                    />
                  </div>
                )}
                {hook.type === "command" && (
                  <div>
                    <label className="block text-xs text-th-text-tertiary mb-1">Command</label>
                    <input
                      className="w-full px-3 py-1.5 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary placeholder-th-text-muted text-xs focus:outline-none focus:border-blue-400 font-mono"
                      value={hook.command}
                      onChange={(e) => updateHook(hook.id, { command: e.target.value })}
                      placeholder=".cursor/hooks/my-hook.sh"
                    />
                  </div>
                )}
                {hook.type === "http" && (
                  <div>
                    <label className="block text-xs text-th-text-tertiary mb-1">URL</label>
                    <input
                      className="w-full px-3 py-1.5 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary placeholder-th-text-muted text-xs focus:outline-none focus:border-blue-400 font-mono"
                      value={hook.url}
                      onChange={(e) => updateHook(hook.id, { url: e.target.value })}
                      placeholder="http://localhost:8080/my-hook"
                    />
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })}
      <button
        type="button"
        onClick={addHook}
        className="flex items-center gap-1.5 text-xs text-th-text-tertiary hover:text-th-text-primary transition-colors py-1"
      >
        <Plus size={13} />
        Add hook
      </button>
    </div>
  );
}

function InputField({ label, value, onChange, placeholder, type = "text" }: {
  label: string; value: string; onChange: (v: string) => void; placeholder?: string; type?: string;
}) {
  return (
    <div>
      <label className="block text-sm font-medium text-th-text-tertiary mb-2">{label}</label>
      <input
        className="w-full px-4 py-2.5 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-300/30 transition-all text-sm"
        type={type} value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder}
      />
    </div>
  );
}

function SelectField({ label, value, onChange, options }: {
  label: string; value: string; onChange: (v: string) => void; options: { value: string; label: string }[];
}) {
  return (
    <div>
      <label className="block text-sm font-medium text-th-text-tertiary mb-2">{label}</label>
      <div className="relative">
        <select
          className="w-full appearance-none px-4 py-2.5 pr-10 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-300/30 transition-all text-sm"
          value={value} onChange={(e) => onChange(e.target.value)}
        >
          {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
        </select>
        <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-th-text-muted pointer-events-none" />
      </div>
    </div>
  );
}

function AgentSelect({ label, value, agents, onChange }: {
  label: string;
  value: string;
  agents: AgentSpec[];
  onChange: (v: string) => void;
}) {
  return (
    <div>
      <label className="block text-xs text-th-text-tertiary mb-1">{label}</label>
      <div className="relative">
        <select
          className="w-full appearance-none px-3 py-1.5 pr-8 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary text-xs focus:outline-none focus:border-blue-400"
          value={value}
          onChange={(e) => onChange(e.target.value)}
        >
          {agents.length === 0 && (
            <option value={value}>{value}</option>
          )}
          {agents.map((a) => (
            <option key={a.name} value={a.name}>{a.name}</option>
          ))}
        </select>
        <ChevronDown size={12} className="absolute right-2 top-1/2 -translate-y-1/2 text-th-text-muted pointer-events-none" />
      </div>
    </div>
  );
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex items-center gap-3 cursor-pointer">
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`relative w-10 h-[22px] rounded-full transition-all duration-200 border ${checked ? "bg-blue-600 border-blue-600" : "bg-th-inset-bg border-th-border"}`}
      >
        <span className={`absolute top-0.5 left-0.5 w-[16px] h-[16px] rounded-full transition-all duration-200 ${checked ? "translate-x-[18px] bg-white" : "bg-neutral-400"}`} />
      </button>
      <span className="text-sm text-th-text-secondary">{label}</span>
    </label>
  );
}

// ── Managed Triggers catalog ─────────────────────────────────────────────────

const MANAGED_TRIGGER_META: Record<string, { label: string; description: string }> = {
  "new-download":      { label: "New Download",        description: "Fires when a file arrives in ~/Downloads" },
  "new-screenshot":    { label: "New Screenshot",       description: "Fires when a screenshot is saved to the Desktop" },
  "mail-unread":       { label: "Mail Unread",          description: "Fires when the Mail unread count changes" },
  "calendar-upcoming": { label: "Calendar Upcoming",    description: "Fires when an event starts within 30 minutes" },
  "reminders-overdue": { label: "Reminders Overdue",    description: "Fires when overdue reminders are detected" },
  "zoom-meeting":      { label: "Zoom Meeting",         description: "Fires when Zoom starts or stops" },
  "slack-active":      { label: "Slack Active",         description: "Fires when Slack launches or quits" },
  "battery-low":       { label: "Battery Low",          description: "Fires when battery drops below 20%" },
  "icloud-changed":    { label: "iCloud Drive Changed", description: "Fires when new files sync into iCloud Drive" },
  "app-switch":        { label: "App Switch",           description: "Fires when the frontmost app changes" },
};

function ManagedTriggersSection({
  triggers, busyId, evalSuggestions, onEdit, onViewRuns, onToggle, onRunNow,
}: {
  triggers: TriggerSpec[];
  busyId: string | null;
  evalSuggestions: Map<string, AmbientHint>;
  onEdit: (t: TriggerSpec) => void;
  onViewRuns: (id: string) => void;
  onToggle: (id: string) => Promise<void> | void;
  onRunNow: (id: string) => Promise<void> | void;
}) {
  return (
    <div className="bg-th-card-bg border border-th-card-border rounded-xl p-5">
      <div className="mb-3">
        <h3 className="font-semibold text-th-text-primary">macOS Triggers</h3>
        <p className="text-xs text-th-text-tertiary mt-0.5">
          Built-in triggers for macOS apps and system events. Enable the ones you want — they cannot be deleted.
        </p>
      </div>
      <div className="space-y-2">
        {triggers.map((t) => {
          const meta = MANAGED_TRIGGER_META[t.id];
          return (
            <ManagedTriggerRow
              key={t.id}
              t={t}
              label={meta?.label ?? t.id}
              description={meta?.description ?? ""}
              busy={busyId === t.id}
              evalSuggestion={evalSuggestions.get(t.id)}
              onEdit={() => onEdit(t)}
              onViewRuns={() => onViewRuns(t.id)}
              onToggle={() => onToggle(t.id)}
              onRunNow={() => onRunNow(t.id)}
            />
          );
        })}
      </div>
    </div>
  );
}

function ManagedTriggerRow({ t, label, description, busy, evalSuggestion, onEdit, onViewRuns, onToggle, onRunNow }: {
  t: TriggerSpec;
  label: string;
  description: string;
  busy: boolean;
  evalSuggestion?: AmbientHint;
  onEdit: () => void;
  onViewRuns: () => void;
  onToggle: () => Promise<void> | void;
  onRunNow: () => Promise<void> | void;
}) {
  const navigate = useNavigate();
  const TypeIcon =
    t.type === "fileos" ? FileSearch
    : t.type === "macostool" ? Terminal
    : t.type === "http" ? Globe
    : t.type === "git" ? GitBranch
    : Terminal;

  const statusPill =
    t.last_status === "running"
      ? { cls: "bg-amber-500/15 text-amber-400 border-amber-500/25", label: "running" }
      : t.last_status === "success"
        ? { cls: "bg-emerald-500/15 text-emerald-400 border-emerald-500/25", label: "ok" }
        : t.last_status === "error"
          ? { cls: "bg-red-500/15 text-red-400 border-red-500/25", label: "error" }
          : null;

  return (
    <div className={`border border-th-border rounded-lg bg-th-inset-bg px-3 py-2.5 ${t.enabled ? "" : "opacity-60"}`}>
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-md bg-th-card-bg border border-th-border flex items-center justify-center shrink-0">
          <TypeIcon size={14} className="text-th-text-tertiary" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-th-text-primary truncate">{label}</span>
            {statusPill && (
              <span className={`inline-flex items-center text-[10px] px-1.5 py-0.5 rounded-full font-medium border ${statusPill.cls}`}>
                {statusPill.label}
              </span>
            )}
          </div>
          <p className="text-xs text-th-text-tertiary truncate mt-0.5">{description}</p>
          {t.last_run && (
            <div className="flex items-center gap-1.5 mt-0.5 flex-wrap">
              <span className="text-xs text-th-text-muted">Last run: {formatRelative(t.last_run)}</span>
              {t.last_status && (
                <span className={`inline-flex items-center px-1.5 py-0.5 rounded-full text-[10px] font-semibold border ${
                  t.last_status === "success"
                    ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                    : t.last_status === "error"
                    ? "bg-red-500/10 text-red-400 border-red-500/20"
                    : t.last_status === "running"
                    ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20 animate-pulse"
                    : "bg-amber-500/10 text-amber-400 border-amber-500/20"
                }`}>
                  {t.last_status}
                </span>
              )}
            </div>
          )}
        </div>
        <div className="flex items-center gap-1 shrink-0">
          {/* Run now */}
          <button
            className="p-1.5 rounded-md hover:bg-th-surface-hover text-th-text-tertiary hover:text-emerald-400 transition-colors disabled:opacity-50"
            onClick={onRunNow}
            disabled={busy}
            title="Run now"
          >
            {busy ? <Loader2 size={13} className="animate-spin" /> : <Play size={13} />}
          </button>
          {/* Edit prompt / agent */}
          <button
            className="p-1.5 rounded-md hover:bg-th-surface-hover text-th-text-tertiary hover:text-th-text-primary transition-colors"
            onClick={onEdit}
            title="Edit prompt / agent"
          >
            <Edit3 size={13} />
          </button>
          {/* View runs */}
          <button
            className="p-1.5 rounded-md hover:bg-th-surface-hover text-th-text-tertiary hover:text-th-text-primary transition-colors"
            onClick={onViewRuns}
            title="View runs"
          >
            <History size={13} />
          </button>
          {/* Enable / disable toggle */}
          <button
            type="button"
            role="switch"
            aria-checked={t.enabled}
            onClick={onToggle}
            title={t.enabled ? "Disable" : "Enable"}
            className={`relative w-8 h-[18px] rounded-full transition-all border flex-shrink-0 ${
              t.enabled ? "bg-blue-600 border-blue-600" : "bg-th-inset-bg border-th-border"
            }`}
          >
            <span className={`absolute top-[1px] left-[1px] w-[14px] h-[14px] rounded-full transition-all ${
              t.enabled ? "translate-x-[14px] bg-white" : "bg-neutral-500"
            }`} />
          </button>
        </div>
      </div>
      {t.last_error && (
        <p className="text-[11px] text-red-400/90 mt-2 truncate" title={t.last_error}>
          {t.last_error}
        </p>
      )}
      {evalSuggestion && (
        <button
          onClick={() => navigate(`/ambient?highlight=${encodeURIComponent(evalSuggestion.id)}`)}
          className="mt-2 inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-amber-500/10 border border-amber-500/25 text-[11px] font-medium text-amber-300 hover:bg-amber-500/20 transition-all"
          title="An improved prompt was suggested for this trigger"
        >
          <Sparkles size={11} aria-hidden />
          Prompt suggestion available
        </button>
      )}
    </div>
  );
}

// ── Custom Triggers ──────────────────────────────────────────────────────────

function CustomTriggersSection({
  triggers, agents, busyId, evalSuggestions, onCreate, onEdit, onViewRuns, onToggle, onRunNow, onDelete,
}: {
  triggers: TriggerSpec[];
  agents: AgentSpec[];
  busyId: string | null;
  evalSuggestions: Map<string, AmbientHint>;
  onCreate: () => void;
  onEdit: (t: TriggerSpec) => void;
  onViewRuns: (id: string) => void;
  onToggle: (id: string) => Promise<void> | void;
  onRunNow: (id: string) => Promise<void> | void;
  onDelete: (id: string) => Promise<void> | void;
}) {
  const atCap = triggers.length >= MAX_TRIGGERS;
  return (
    <div className="bg-th-card-bg border border-th-card-border rounded-xl p-5">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="font-semibold text-th-text-primary">Custom Triggers</h3>
          <p className="text-xs text-th-text-tertiary mt-0.5">
            Fire an agent on file changes or AppleScript output. {triggers.length}/{MAX_TRIGGERS} used.
          </p>
        </div>
        <button
          className="px-3 py-1.5 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 hover:bg-emerald-500/20 disabled:opacity-50 rounded-lg text-xs font-medium transition-colors flex items-center gap-1.5"
          onClick={onCreate}
          disabled={atCap}
          title={atCap ? `Cap of ${MAX_TRIGGERS} reached — delete one first.` : undefined}
        >
          <Plus size={12} /> New trigger
        </button>
      </div>

      {triggers.length === 0 ? (
        <p className="text-xs text-th-text-muted italic py-4">
          No custom triggers configured. Click "New trigger" or ask the trigger-builder agent in chat.
        </p>
      ) : (
        <div className="space-y-2">
          {triggers.map((t) => (
            <TriggerRow
              key={t.id}
              t={t}
              agents={agents}
              busy={busyId === t.id}
              evalSuggestion={evalSuggestions.get(t.id)}
              onEdit={() => onEdit(t)}
              onViewRuns={() => onViewRuns(t.id)}
              onToggle={() => onToggle(t.id)}
              onRunNow={() => onRunNow(t.id)}
              onDelete={() => onDelete(t.id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function TriggerRow({ t, busy, evalSuggestion, onEdit, onViewRuns, onToggle, onRunNow, onDelete }: {
  t: TriggerSpec;
  agents: AgentSpec[];
  busy: boolean;
  evalSuggestion?: AmbientHint;
  onEdit: () => void;
  onViewRuns: () => void;
  onToggle: () => Promise<void> | void;
  onRunNow: () => Promise<void> | void;
  onDelete: () => Promise<void> | void;
}) {
  const navigate = useNavigate();
  const [confirmDelete, setConfirmDelete] = useState(false);

  const TypeIcon =
    t.type === "fileos"
      ? FileSearch
      : t.type === "macostool"
        ? Terminal
        : t.type === "http"
          ? Globe
          : t.type === "git"
            ? GitBranch
            : Terminal; // shell
  const subtitle =
    t.type === "fileos"
      ? `${t.watch} on ${t.path || "?"}${t.glob ? ` (${t.glob})` : ""}`
      : t.type === "macostool"
        ? `${t.language}${t.match ? ` /${t.match}/` : ""}`
        : t.type === "http"
          ? `${t.method} ${t.url || "?"} · ${t.http_mode}`
          : t.type === "git"
            ? `${t.repo_path || "?"} @ ${t.branch}`
            : `${(t.command || "").slice(0, 60) || "?"} · ${t.shell_mode}`;
  const statusPill =
    t.last_status === "running"
      ? { cls: "bg-amber-500/15 text-amber-400 border-amber-500/25", label: "running" }
      : t.last_status === "success"
        ? { cls: "bg-emerald-500/15 text-emerald-400 border-emerald-500/25", label: "ok" }
        : t.last_status === "error"
          ? { cls: "bg-red-500/15 text-red-400 border-red-500/25", label: "error" }
          : { cls: "bg-th-inset-bg text-th-text-tertiary border-th-border", label: t.last_status || "idle" };

  return (
    <div className={`border border-th-border rounded-lg bg-th-inset-bg px-3 py-2.5 ${t.enabled ? "" : "opacity-60"}`}>
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-md bg-th-card-bg border border-th-border flex items-center justify-center shrink-0">
          <TypeIcon size={14} className="text-th-text-tertiary" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="font-mono text-sm text-th-text-primary truncate">{t.id}</span>
            <span className={`inline-flex items-center text-[10px] px-1.5 py-0.5 rounded-full font-medium border ${statusPill.cls}`}>
              {statusPill.label}
            </span>
            <span className="text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded bg-th-card-bg text-th-text-muted border border-th-border">
              {t.type}
            </span>
          </div>
          <p className="text-xs text-th-text-tertiary truncate mt-0.5">
            {subtitle} · every {formatPollDuration(t.poll_seconds)} · agent: {t.agent_name || "general-purpose"}
          </p>
          {t.last_run && (
            <div className="flex items-center gap-1.5 mt-0.5 flex-wrap">
              <span className="text-xs text-th-text-muted">Last run: {formatRelative(t.last_run)}</span>
              {t.last_status && (
                <span className={`inline-flex items-center px-1.5 py-0.5 rounded-full text-[10px] font-semibold border ${
                  t.last_status === "success"
                    ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                    : t.last_status === "error"
                    ? "bg-red-500/10 text-red-400 border-red-500/20"
                    : t.last_status === "running"
                    ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20 animate-pulse"
                    : "bg-amber-500/10 text-amber-400 border-amber-500/20"
                }`}>
                  {t.last_status}
                </span>
              )}
            </div>
          )}
        </div>
        <div className="flex items-center gap-1 shrink-0">
          <button
            className="p-1.5 rounded-md hover:bg-th-surface-hover text-th-text-tertiary hover:text-emerald-400 transition-colors disabled:opacity-50"
            onClick={onRunNow}
            disabled={busy}
            title="Run now"
          >
            {busy ? <Loader2 size={13} className="animate-spin" /> : <Play size={13} />}
          </button>
          <button
            className="p-1.5 rounded-md hover:bg-th-surface-hover text-th-text-tertiary hover:text-th-text-primary transition-colors"
            onClick={onEdit}
            title="Edit"
          >
            <Edit3 size={13} />
          </button>
          {confirmDelete ? (
            <>
              <button
                className="px-2 py-1 rounded-md bg-red-500/15 border border-red-500/30 text-red-400 hover:bg-red-500/25 text-[11px] font-medium transition-colors"
                onClick={() => { setConfirmDelete(false); void onDelete(); }}
                title="Confirm delete"
              >
                Delete?
              </button>
              <button
                className="p-1.5 rounded-md hover:bg-th-surface-hover text-th-text-tertiary hover:text-th-text-primary transition-colors"
                onClick={() => setConfirmDelete(false)}
                title="Cancel"
              >
                <X size={13} />
              </button>
            </>
          ) : (
            <button
              className="p-1.5 rounded-md hover:bg-th-surface-hover text-th-text-tertiary hover:text-red-400 transition-colors"
              onClick={() => setConfirmDelete(true)}
              title="Delete"
            >
              <Trash2 size={13} />
            </button>
          )}
          <button
            className="p-1.5 rounded-md hover:bg-th-surface-hover text-th-text-tertiary hover:text-th-text-primary transition-colors"
            onClick={onViewRuns}
            title="View runs"
          >
            <History size={13} />
          </button>
          <button
            className="p-1.5 rounded-md hover:bg-th-surface-hover text-th-text-tertiary hover:text-th-text-primary transition-colors"
            onClick={onToggle}
            title={t.enabled ? "Pause" : "Enable"}
          >
            {t.enabled ? <Square size={13} /> : <CheckCircle2 size={13} />}
          </button>
        </div>
      </div>
      {t.last_error && (
        <p className="text-[11px] text-red-400/90 mt-2 truncate" title={t.last_error}>
          {t.last_error}
        </p>
      )}
      {evalSuggestion && (
        <button
          onClick={() => navigate(`/ambient?highlight=${encodeURIComponent(evalSuggestion.id)}`)}
          className="mt-2 inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-amber-500/10 border border-amber-500/25 text-[11px] font-medium text-amber-300 hover:bg-amber-500/20 transition-all"
          title="An improved prompt was suggested for this trigger"
        >
          <Sparkles size={11} aria-hidden />
          Prompt suggestion available
        </button>
      )}
    </div>
  );
}

function formatRelative(dateStr: string | null): string {
  if (!dateStr) return "never";
  const d = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const mins = Math.floor(diffMs / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

function formatPollDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return s === 0 ? `${m}m` : `${m}m ${s}s`;
  }
  const h = Math.floor(seconds / 3600);
  const rem = seconds % 3600;
  const m = Math.floor(rem / 60);
  return m === 0 ? `${h}h` : `${h}h ${m}m`;
}

function TriggerModal({ initial, agents, error, onClose, onSubmit }: {
  initial: TriggerSpec | null;
  agents: AgentSpec[];
  error: string | null;
  onClose: () => void;
  onSubmit: (req: Record<string, unknown>, isUpdate: boolean) => Promise<void>;
}) {
  const isUpdate = initial !== null;
  const [type, setType] = useState<TriggerType>(initial?.type ?? "fileos");
  const [id, setId] = useState(initial?.id ?? "");
  const [pollSeconds, setPollSeconds] = useState(initial?.poll_seconds ?? 60);
  const [agentName, setAgentName] = useState(initial?.agent_name ?? "");
  const [prompt, setPrompt] = useState(initial?.prompt ?? "");
  // fileos
  const [path, setPath] = useState(initial?.path ?? "");
  const [watch, setWatch] = useState<FileOsWatch>(initial?.watch ?? "mtime");
  const [glob, setGlob] = useState(initial?.glob ?? "");
  // macostool
  const [script, setScript] = useState(initial?.script ?? "");
  const [language, setLanguage] = useState<OsaLanguage>(initial?.language ?? "AppleScript");
  const [match, setMatch] = useState(initial?.match ?? "");
  // http
  const [url, setUrl] = useState(initial?.url ?? "");
  const [httpMode, setHttpMode] = useState<HttpMode>(initial?.http_mode ?? "body_hash");
  const [httpMethod, setHttpMethod] = useState<HttpMethod>(initial?.method ?? "GET");
  const [jsonPath, setJsonPath] = useState(initial?.json_path ?? "");
  // git
  const [repoPath, setRepoPath] = useState(initial?.repo_path ?? "");
  const [branch, setBranch] = useState(initial?.branch ?? "HEAD");
  const [authorFilter, setAuthorFilter] = useState(initial?.author_filter ?? "");
  const [pathFilter, setPathFilter] = useState(initial?.path_filter ?? "");
  // shell
  const [command, setCommand] = useState(initial?.command ?? "");
  const [shellMode, setShellMode] = useState<ShellMode>(initial?.shell_mode ?? "stdout_change");
  const [cwd, setCwd] = useState(initial?.cwd ?? "");
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async () => {
    setSubmitting(true);
    try {
      const req: Record<string, unknown> = {
        prompt,
        poll_seconds: pollSeconds,
        agent_name: agentName || null,
      };
      if (!isUpdate) {
        req.id = id;
        req.type = type;
      }
      if (type === "fileos") {
        req.path = path;
        req.watch = watch;
        req.glob = glob || null;
      } else if (type === "macostool") {
        req.script = script;
        req.language = language;
        req.match = match || null;
      } else if (type === "http") {
        req.url = url;
        req.http_mode = httpMode;
        req.method = httpMethod;
        req.json_path = jsonPath || null;
        req.match = match || null;
      } else if (type === "git") {
        req.repo_path = repoPath;
        req.branch = branch || "HEAD";
        req.author_filter = authorFilter || null;
        req.path_filter = pathFilter || null;
      } else if (type === "shell") {
        req.command = command;
        req.shell_mode = shellMode;
        req.cwd = cwd || null;
        req.match = match || null;
      }
      await onSubmit(req, isUpdate);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center p-4" onClick={onClose}>
      <div
        className="bg-th-card-bg border border-th-card-border rounded-xl w-full max-w-lg max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 py-4 border-b border-th-border">
          <h2 className="font-semibold text-th-text-primary">
            {isUpdate ? `Edit trigger: ${initial!.id}` : "New trigger"}
          </h2>
          <button onClick={onClose} className="p-1 rounded hover:bg-th-surface-hover text-th-text-tertiary">
            <X size={16} />
          </button>
        </div>

        <div className="p-5 space-y-4">
          {!isUpdate && (
            <>
              <InputField label="ID" value={id} onChange={setId} placeholder="e.g. downloads-pdf-watch" />
              <SelectField
                label="Type"
                value={type}
                onChange={(v) => setType(v as TriggerType)}
                options={[
                  { value: "fileos", label: "File / OS path" },
                  { value: "macostool", label: "macOS osascript" },
                  { value: "http", label: "HTTP endpoint" },
                  { value: "git", label: "Git repo" },
                  { value: "shell", label: "Shell command" },
                ]}
              />
            </>
          )}

          <AgentSelect
            label="Worker agent"
            value={agentName}
            agents={[{ name: "", description: "" } as AgentSpec, ...agents]}
            onChange={setAgentName}
          />

          <div>
            <label className="block text-sm font-medium text-th-text-tertiary mb-2">Prompt</label>
            <textarea
              className="w-full px-4 py-2.5 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-400 text-sm min-h-[80px] resize-y"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="What should the agent do when this trigger fires?  The event payload is appended automatically."
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-th-text-tertiary mb-2">
              Poll interval
            </label>
            <div className="flex items-center gap-2">
              <input
                type="number"
                min={MIN_POLL_SECONDS}
                max={MAX_POLL_SECONDS}
                step={1}
                value={pollSeconds}
                onChange={(e) => {
                  const v = parseInt(e.target.value, 10);
                  if (!isNaN(v)) setPollSeconds(Math.max(MIN_POLL_SECONDS, Math.min(MAX_POLL_SECONDS, v)));
                }}
                className="w-28 px-3 py-2 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-300/30 transition-all text-sm text-right"
              />
              <span className="text-sm text-th-text-muted">seconds</span>
              <span className="text-xs text-th-text-tertiary ml-1">({formatPollDuration(pollSeconds)})</span>
            </div>
            <p className="text-xs text-th-text-muted mt-1">Min {MIN_POLL_SECONDS}s · Max {MAX_POLL_SECONDS / 3600}h</p>
          </div>

          {type === "fileos" && (
            <>
              <InputField label="Path" value={path} onChange={setPath} placeholder="~/Downloads" />
              <SelectField
                label="Watch"
                value={watch}
                onChange={(v) => setWatch(v as FileOsWatch)}
                options={[
                  { value: "mtime", label: "mtime — fire when modified time changes" },
                  { value: "size", label: "size — fire when file size changes" },
                  { value: "exists", label: "exists — fire when path appears or disappears" },
                  { value: "new_files", label: "new_files — fire on new matching files in directory" },
                ]}
              />
              {watch === "new_files" && (
                <InputField label="Glob" value={glob} onChange={setGlob} placeholder="*.pdf" />
              )}
            </>
          )}

          {type === "macostool" && (
            <>
              <SelectField
                label="Language"
                value={language}
                onChange={(v) => setLanguage(v as OsaLanguage)}
                options={[
                  { value: "AppleScript", label: "AppleScript" },
                  { value: "JavaScript", label: "JavaScript (JXA)" },
                ]}
              />
              <div>
                <label className="block text-sm font-medium text-th-text-tertiary mb-2">Script</label>
                <textarea
                  className="w-full px-4 py-2.5 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-400 text-xs font-mono min-h-[120px] resize-y"
                  value={script}
                  onChange={(e) => setScript(e.target.value)}
                  placeholder={'tell application "System Events" to get name of every process whose visible is true'}
                />
              </div>
              <InputField
                label="Match regex (optional)"
                value={match}
                onChange={setMatch}
                placeholder="e.g. ^Battery on AC: false"
              />
            </>
          )}

          {type === "http" && (
            <>
              <InputField label="URL" value={url} onChange={setUrl} placeholder="https://api.example.com/status" />
              <SelectField
                label="Method"
                value={httpMethod}
                onChange={(v) => setHttpMethod(v as HttpMethod)}
                options={[
                  { value: "GET", label: "GET" },
                  { value: "POST", label: "POST" },
                  { value: "HEAD", label: "HEAD" },
                ]}
              />
              <SelectField
                label="Mode"
                value={httpMode}
                onChange={(v) => setHttpMode(v as HttpMode)}
                options={[
                  { value: "body_hash", label: "body_hash — fire when response body changes" },
                  { value: "status_change", label: "status_change — fire when HTTP status changes" },
                  { value: "json_value", label: "json_value — fire when extracted JSON value changes" },
                  { value: "regex", label: "regex — fire when regex matches body (rising edge)" },
                ]}
              />
              {httpMode === "json_value" && (
                <InputField
                  label="JSON path"
                  value={jsonPath}
                  onChange={setJsonPath}
                  placeholder="data.items.0.id"
                />
              )}
              {httpMode === "regex" && (
                <InputField
                  label="Match regex"
                  value={match}
                  onChange={setMatch}
                    placeholder={`e.g. "status":\\s*"down"`}
                />
              )}
            </>
          )}

          {type === "git" && (
            <>
              <InputField
                label="Repo path"
                value={repoPath}
                onChange={setRepoPath}
                placeholder="~/code/my-project"
              />
              <InputField
                label="Branch"
                value={branch}
                onChange={setBranch}
                placeholder="HEAD or main"
              />
              <InputField
                label="Author filter regex (optional)"
                value={authorFilter}
                onChange={setAuthorFilter}
                placeholder="alice@example.com"
              />
              <InputField
                label="Path filter (optional)"
                value={pathFilter}
                onChange={setPathFilter}
                placeholder="backend/**"
              />
            </>
          )}

          {type === "shell" && (
            <>
              <div>
                <label className="block text-sm font-medium text-th-text-tertiary mb-2">Command</label>
                <textarea
                  className="w-full px-4 py-2.5 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-400 text-xs font-mono min-h-[80px] resize-y"
                  value={command}
                  onChange={(e) => setCommand(e.target.value)}
                  placeholder={'pgrep -x Zoom'}
                />
              </div>
              <SelectField
                label="Mode"
                value={shellMode}
                onChange={(v) => setShellMode(v as ShellMode)}
                options={[
                  { value: "stdout_change", label: "stdout_change — fire when stdout changes" },
                  { value: "exit_code_change", label: "exit_code_change — fire when exit code changes" },
                  { value: "regex", label: "regex — fire when regex matches stdout (rising edge)" },
                ]}
              />
              <InputField
                label="Working directory (optional)"
                value={cwd}
                onChange={setCwd}
                placeholder="~/projects/foo"
              />
              {shellMode === "regex" && (
                <InputField
                  label="Match regex"
                  value={match}
                  onChange={setMatch}
                  placeholder="e.g. ERROR|FAIL"
                />
              )}
            </>
          )}

          {error && (
            <div className="px-3 py-2 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-xs">
              {error}
            </div>
          )}
        </div>

        <div className="flex items-center justify-end gap-2 px-5 py-4 border-t border-th-border">
          <button
            className="px-3 py-1.5 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-xs font-medium transition-colors"
            onClick={onClose}
          >
            Cancel
          </button>
          <button
            className="px-3 py-1.5 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 hover:bg-emerald-500/20 disabled:opacity-50 rounded-lg text-xs font-medium transition-colors flex items-center gap-1.5"
            onClick={handleSubmit}
            disabled={submitting || (!isUpdate && !id) || !prompt}
          >
            {submitting ? <Loader2 size={12} className="animate-spin" /> : <Check size={12} />}
            {isUpdate ? "Save changes" : "Create trigger"}
          </button>
        </div>
      </div>
    </div>
  );
}
