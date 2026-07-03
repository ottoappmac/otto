import { useEffect, useMemo, useState } from "react";
import { useNavigate, type NavigateFunction } from "react-router-dom";
import { Workflow, Plus, Trash2, Edit3, X, FileJson, Loader2, MessageSquare, Lock, Zap } from "lucide-react";
import { api } from "../hooks/useApi";
import type { AgentSpec } from "../types";
import SkillsPage from "./SkillsPage";
import ToolsPage from "./ToolsPage";

type AgentsTab = "agents" | "skills" | "tools";

const TABS: { id: AgentsTab; label: string }[] = [
  { id: "agents", label: "Agents" },
  { id: "skills", label: "Skills" },
  { id: "tools", label: "Tools" },
];

export default function AgentsPage() {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<AgentsTab>("agents");
  const [agents, setAgents] = useState<AgentSpec[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [editingAgent, setEditingAgent] = useState<AgentSpec | null>(null);
  const [showJsonEditor, setShowJsonEditor] = useState(false);
  const [confirmDeleteName, setConfirmDeleteName] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const refresh = () => {
    api.listAgents().then(setAgents).catch((e) => console.warn("Failed to load agents:", e)).finally(() => setLoading(false));
  };

  useEffect(() => {
    refresh();
  }, []);

  const handleDelete = async (name: string) => {
    setDeleteError(null);
    try {
      await api.deleteAgent(name);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Delete failed";
      setDeleteError(msg.replace(/^API \d+:\s*/, "").replace(/^\{.*"error":\s*"/, "").replace(/"\}$/, ""));
    }
    setConfirmDeleteName(null);
    refresh();
  };

  const HIDDEN_AGENT_IDS = new Set(["claude-session-eval-agent", "openclaw-session-eval-agent"]);
  const customAgents = useMemo(() => agents.filter((a) => !a.builtin && !HIDDEN_AGENT_IDS.has(a.name)), [agents]);
  const managedAgents = useMemo(() => agents.filter((a) => a.builtin && !HIDDEN_AGENT_IDS.has(a.name)), [agents]);

  return (
    <div className="h-full flex flex-col">
      <header className="border-b border-th-border px-6 py-4 flex items-center justify-between shrink-0 bg-th-bg-secondary">
        <div className="flex items-center gap-4">
          <h1 className="text-lg font-bold text-th-text-primary">Agents</h1>
          <div className="flex gap-1 bg-th-inset-bg rounded-lg p-1 border border-th-border">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id)}
                className={`px-3.5 py-1.5 rounded-md text-xs font-medium transition-all duration-150 ${
                  activeTab === tab.id
                    ? "bg-th-tab-active-bg text-th-tab-active-fg shadow-sm"
                    : "text-th-text-tertiary hover:text-th-text-primary hover:bg-th-surface-hover"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
        {activeTab === "agents" && (
          <div className="flex items-center gap-2">
            <button className="px-3 py-2 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-sm font-medium transition-colors flex items-center gap-1.5" onClick={() => setShowJsonEditor(true)}><FileJson size={14} /> Edit JSON</button>
            <button className="px-4 py-2 bg-th-tab-active-bg text-th-tab-active-fg rounded-lg text-sm font-semibold transition-opacity hover:opacity-90 flex items-center gap-2" onClick={() => { setEditingAgent(null); setShowCreate(true); }}><Plus size={15} /> Create Agent</button>
          </div>
        )}
      </header>

      {activeTab === "agents" && (
        <div className="flex-1 overflow-y-auto p-6">
          {deleteError && (
            <div className="mb-4 flex items-center justify-between gap-3 px-4 py-3 bg-red-500/10 border border-red-500/20 rounded-lg">
              <span className="text-sm text-red-400">{deleteError}</span>
              <button className="text-red-400/60 hover:text-red-400 transition-colors" onClick={() => setDeleteError(null)}><X size={16} /></button>
            </div>
          )}
          {loading && agents.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full -mt-12">
              <Loader2 size={32} className="text-th-text-muted animate-spin mb-4" />
              <p className="text-sm text-th-text-tertiary">Loading agents...</p>
            </div>
          )}
          {!loading && agents.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full -mt-12">
              <div className="w-20 h-20 rounded-2xl bg-th-card-bg border border-th-card-border flex items-center justify-center mb-6">
                <Workflow size={36} className="text-th-text-muted" />
              </div>
              <h3 className="text-lg font-semibold text-th-text-secondary mb-2">No agents yet</h3>
              <p className="text-sm text-th-text-tertiary mb-6 max-w-xs text-center">
                Create your first agent to automate tasks with AI-powered workflows.
              </p>
              <button
                className="px-5 py-2.5 bg-th-tab-active-bg text-th-tab-active-fg rounded-lg text-sm font-semibold transition-opacity hover:opacity-90 flex items-center gap-2"
                onClick={() => { setEditingAgent(null); setShowCreate(true); }}
              >
                <Plus size={15} /> Create your first agent
              </button>
            </div>
          )}
          <div className="space-y-3">
            {customAgents.map((agent) => (
              <AgentListCard
                key={agent.name}
                agent={agent}
                navigate={navigate}
                onEdit={() => { setEditingAgent(agent); setShowCreate(true); }}
                confirmDeleteName={confirmDeleteName}
                setConfirmDeleteName={setConfirmDeleteName}
                handleDelete={handleDelete}
                deletable
              />
            ))}
            {managedAgents.length > 0 && (
              <>
                <div className="flex items-center gap-3 mt-6 mb-3">
                  <div className="flex-1 border-t border-th-border" />
                  <span className="text-sm uppercase tracking-widest text-th-text-muted font-semibold flex items-center gap-1.5"><Zap size={14} /> Managed</span>
                  <div className="flex-1 border-t border-th-border" />
                </div>
                {managedAgents.map((agent) => (
                  <AgentListCard
                    key={agent.name}
                    agent={agent}
                    navigate={navigate}
                    onEdit={() => { setEditingAgent(agent); setShowCreate(true); }}
                    confirmDeleteName={confirmDeleteName}
                    setConfirmDeleteName={setConfirmDeleteName}
                    handleDelete={handleDelete}
                    deletable={false}
                  />
                ))}
              </>
            )}
          </div>
        </div>
      )}

      {activeTab === "skills" && (
        <div className="flex-1 flex flex-col overflow-hidden">
          <SkillsPage embedded />
        </div>
      )}

      {activeTab === "tools" && (
        <div className="flex-1 flex flex-col overflow-hidden">
          <ToolsPage embedded />
        </div>
      )}

      {showCreate && (
        <AgentDialog agent={editingAgent} onClose={() => setShowCreate(false)} onSaved={() => { setShowCreate(false); refresh(); }} />
      )}
      {showJsonEditor && <EditAgentsJsonDialog agents={agents} onClose={() => setShowJsonEditor(false)} onSaved={() => { setShowJsonEditor(false); refresh(); }} />}
    </div>
  );
}

interface PickerOption {
  id: string;
  label: string;
  detail?: string;
}

type SubFamily = "inherit" | "frontier" | "mlx" | "exo" | "custom";

function deriveSubFamily(a: AgentSpec | null): SubFamily {
  if (!a) return "inherit";
  const f = a.subagent_llm_family;
  if (
    f === "inherit" ||
    f === "frontier" ||
    f === "mlx" ||
    f === "exo" ||
    f === "custom"
  )
    return f;
  if (a.model_override) return "custom";
  return "inherit";
}

function AgentListCard({
  agent,
  navigate,
  onEdit,
  confirmDeleteName,
  setConfirmDeleteName,
  handleDelete,
  deletable,
}: {
  agent: AgentSpec;
  navigate: NavigateFunction;
  onEdit: () => void;
  confirmDeleteName: string | null;
  setConfirmDeleteName: (name: string | null) => void;
  handleDelete: (name: string) => void | Promise<void>;
  deletable: boolean;
}) {
  return (
    <div className="bg-th-card-bg border border-th-card-border rounded-xl p-5 hover:border-th-border-strong transition-all duration-200">
      <div className="flex items-start justify-between">
        <div className="flex items-start gap-3">
          <div className="w-10 h-10 rounded-lg bg-th-inset-bg border border-th-border flex items-center justify-center mt-0.5 shrink-0">
            <Workflow size={20} className="text-th-text-secondary" />
          </div>
          <div>
            <h3 className="font-semibold text-th-text-primary">{agent.name}</h3>
            <p className="text-sm text-th-text-tertiary mt-1 leading-relaxed">
              {agent.description}
            </p>
            <p className="text-xs text-th-text-muted mt-2">
              Task model:{" "}
              <span className="text-th-text-secondary font-medium">
                {deriveSubFamily(agent)}
                {agent.subagent_llm_family === "mlx" && agent.mlx_model_id ? ` — ${agent.mlx_model_id}` : ""}
                {deriveSubFamily(agent) === "custom" && agent.model_override ? ` — ${agent.model_override}` : ""}
              </span>
            </p>
            <div className="flex flex-wrap gap-2 mt-3">
              {agent.tools.map((t) => (
                <span key={t} className="text-xs px-2.5 py-1 rounded-md bg-th-inset-bg text-th-text-secondary border border-th-border font-medium">
                  {t}
                </span>
              ))}
              {agent.skills.map((s) => (
                <span key={s} className="text-xs px-2.5 py-1 rounded-md bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-medium">
                  {s}
                </span>
              ))}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-1 shrink-0 ml-4">
          <button
            type="button"
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium bg-th-tab-active-bg text-th-tab-active-fg hover:opacity-90 transition-opacity"
            title={`Start a new chat with ${agent.name}`}
            onClick={() => {
              localStorage.setItem("chatSelectedAgent", agent.name);
              navigate("/chat");
            }}
          >
            <MessageSquare size={13} />
            Chat
          </button>
          <button type="button" className="p-2 rounded-lg hover:bg-th-surface-hover text-th-text-muted hover:text-th-text-secondary transition-colors" onClick={onEdit}>
            <Edit3 size={15} />
          </button>
          {!deletable ? (
            <span
              className="p-2 rounded-lg text-th-text-faint cursor-not-allowed"
              title="Built-in agent — managed by the app and cannot be deleted."
            >
              <Lock size={15} />
            </span>
          ) : confirmDeleteName === agent.name ? (
            <span className="flex items-center gap-1.5">
              <span className="text-xs text-red-400">Delete?</span>
              <button type="button" className="px-2.5 py-1.5 bg-red-500/20 border border-red-500/30 text-red-400 hover:bg-red-500/30 rounded-lg text-xs font-semibold transition-colors" onClick={() => handleDelete(agent.name)}>Yes</button>
              <button type="button" className="px-2.5 py-1.5 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-xs font-medium transition-colors" onClick={() => setConfirmDeleteName(null)}>No</button>
            </span>
          ) : (
            <button type="button" className="p-2 rounded-lg hover:bg-red-500/10 text-th-text-muted hover:text-red-400 transition-colors" onClick={() => setConfirmDeleteName(agent.name)}>
              <Trash2 size={15} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function AgentDialog({ agent, onClose, onSaved }: { agent: AgentSpec | null; onClose: () => void; onSaved: () => void }) {
  const [name, setName] = useState(agent?.name ?? "");
  const [description, setDescription] = useState(agent?.description ?? "");
  const [systemPrompt, setSystemPrompt] = useState(agent?.system_prompt ?? "");
  const [selectedTools, setSelectedTools] = useState<string[]>(agent?.tools ?? []);
  const [selectedSkills, setSelectedSkills] = useState<string[]>(agent?.skills ?? []);
  const [subFamily, setSubFamily] = useState<SubFamily>(() => deriveSubFamily(agent));
  const [mlxModelId, setMlxModelId] = useState(agent?.mlx_model_id ?? "");
  const [modelOverride, setModelOverride] = useState(agent?.model_override ?? "");
  const [saving, setSaving] = useState(false);

  const [availableMCP, setAvailableMCP] = useState<PickerOption[]>([]);
  const [availableSkills, setAvailableSkills] = useState<PickerOption[]>([]);
  const [mlxLocalAgents, setMlxLocalAgents] = useState<{ repo_id: string; name: string }[]>([]);

  useEffect(() => {
    setName(agent?.name ?? "");
    setDescription(agent?.description ?? "");
    setSystemPrompt(agent?.system_prompt ?? "");
    setSelectedTools(agent?.tools ?? []);
    setSelectedSkills(agent?.skills ?? []);
    setSubFamily(deriveSubFamily(agent));
    setMlxModelId(agent?.mlx_model_id ?? "");
    setModelOverride(agent?.model_override ?? "");
  }, [agent]);

  useEffect(() => {
    api.listMCPServers().then((servers) =>
      setAvailableMCP(servers.map((s) => ({ id: s.id as string, label: s.name as string })))
    );
    api.listSkills().then((skills) =>
      setAvailableSkills(skills.map((s) => ({ id: s.name as string, label: s.name as string, detail: s.description as string })))
    );
  }, []);

  useEffect(() => {
    api
      .getSettings()
      .then((s) => {
        const cd = s.llm?.mlx?.hf_hub_cache?.trim();
        const q = cd ? `?cache_dir=${encodeURIComponent(cd)}` : "";
        return api.mlxLocalModels(q);
      })
      .then((r) => setMlxLocalAgents(r.models ?? []))
      .catch(() => setMlxLocalAgents([]));
  }, []);

  const mlxAgentOptions = (() => {
    const opts: { value: string; label: string }[] = [{ value: "", label: "(default — global MLX text model)" }];
    const seen = new Set<string>();
    for (const m of mlxLocalAgents) {
      if (!seen.has(m.repo_id)) {
        seen.add(m.repo_id);
        opts.push({ value: m.repo_id, label: m.name });
      }
    }
    const cur = mlxModelId.trim();
    if (cur && !seen.has(cur)) opts.push({ value: cur, label: `${cur} (not listed)` });
    return opts;
  })();

  const handleSave = async () => {
    setSaving(true);
    try {
      const payload: Record<string, unknown> = {
        name,
        description,
        system_prompt: systemPrompt,
        tools: selectedTools,
        skills: selectedSkills,
      };
      if (subFamily === "inherit") {
        payload.subagent_llm_family = null;
        payload.mlx_model_id = null;
        payload.model_override = null;
      } else if (subFamily === "frontier") {
        payload.subagent_llm_family = "frontier";
        payload.mlx_model_id = null;
        payload.model_override = null;
      } else if (subFamily === "mlx") {
        payload.subagent_llm_family = "mlx";
        payload.mlx_model_id = mlxModelId.trim() || null;
        payload.model_override = null;
      } else if (subFamily === "exo") {
        payload.subagent_llm_family = "exo";
        payload.mlx_model_id = null;
        payload.model_override = null;
      } else {
        payload.subagent_llm_family = "custom";
        payload.model_override = modelOverride.trim() || null;
        payload.mlx_model_id = null;
      }
      if (agent) {
        await api.updateAgent(agent.name, payload);
      } else {
        await api.createAgent(payload);
      }
      onSaved();
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/30 backdrop-blur-sm flex items-center justify-center z-50 p-8">
      <div className="bg-th-card-bg border border-th-card-border rounded-2xl w-full max-w-2xl max-h-[90vh] overflow-y-auto p-6 shadow-2xl">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-lg font-bold text-th-text-primary">{agent ? "Edit Agent" : "Create Agent"}</h2>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-th-surface-hover text-th-text-muted hover:text-th-text-secondary transition-colors"><X size={20} /></button>
        </div>
        <div className="space-y-4">
          <Field label="Name" value={name} onChange={setName} placeholder="my-agent-name" />
          <Field label="Description" value={description} onChange={setDescription} placeholder="What this agent does" />
          <div>
            <label className="block text-sm font-medium text-th-text-tertiary mb-2">System Prompt</label>
            <textarea className="w-full px-4 py-3 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-300/30 transition-all min-h-[200px] font-mono text-xs" value={systemPrompt} onChange={(e) => setSystemPrompt(e.target.value)} />
          </div>
          <ChipPicker label="MCP Tools" options={availableMCP} selected={selectedTools} onChange={setSelectedTools} emptyText="No MCP servers configured" />
          <ChipPicker label="Skills" options={availableSkills} selected={selectedSkills} onChange={setSelectedSkills} emptyText="No skills available" />
          <div>
            <label className="block text-sm font-medium text-th-text-tertiary mb-2">Task subagent model</label>
            <p className="text-xs text-th-text-muted mb-2">
              When this agent runs as a delegated subagent (general-purpose mode), choose whether it inherits the main chat model, uses the configured frontier stack, MLX, the cluster, or a custom LangChain model id string.
            </p>
            <select
              className="w-full px-4 py-2.5 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary text-sm"
              value={subFamily}
              onChange={(e) => setSubFamily(e.target.value as typeof subFamily)}
            >
              <option value="inherit">Inherit main chat model</option>
              <option value="frontier">Frontier (Anthropic / Bedrock)</option>
              <option value="mlx">MLX (local Hub)</option>
              <option value="exo">Cluster (shared model)</option>
              <option value="custom">Custom (model id string)</option>
            </select>
            {subFamily === "exo" && (
              <p className="mt-2 text-[11px] text-th-text-muted">
                All cluster subagents share the same cluster URL and model id
                from <span className="font-mono">Settings → LLM → Cluster</span>.
              </p>
            )}
            {subFamily === "mlx" && (
              <div className="mt-3">
                <label className="block text-xs font-medium text-th-text-tertiary mb-1.5">MLX repo (optional)</label>
                <select
                  className="w-full px-4 py-2.5 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary text-sm"
                  value={mlxModelId}
                  onChange={(e) => setMlxModelId(e.target.value)}
                >
                  {mlxAgentOptions.map((o) => (
                    <option key={o.value || "__empty"} value={o.value}>
                      {o.label}
                    </option>
                  ))}
                </select>
              </div>
            )}
            {subFamily === "custom" && (
              <div className="mt-3">
                <label className="block text-xs font-medium text-th-text-tertiary mb-1.5">Model override</label>
                <input
                  className="w-full px-4 py-2.5 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary text-sm font-mono"
                  value={modelOverride}
                  onChange={(e) => setModelOverride(e.target.value)}
                  placeholder="e.g. anthropic:claude-sonnet-4-20250514"
                />
              </div>
            )}
          </div>
          <div className="flex justify-end gap-2 pt-3 border-t border-th-border">
            <button className="px-4 py-2 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-sm font-medium transition-colors" onClick={onClose}>Cancel</button>
            <button className="px-4 py-2 bg-th-tab-active-bg text-th-tab-active-fg rounded-lg text-sm font-semibold transition-opacity hover:opacity-90 disabled:opacity-40" onClick={handleSave} disabled={!name || saving}>{saving ? "Saving..." : "Save Agent"}</button>
          </div>
        </div>
      </div>
    </div>
  );
}

function ChipPicker({ label, options, selected, onChange, emptyText }: {
  label: string;
  options: PickerOption[];
  selected: string[];
  onChange: (v: string[]) => void;
  emptyText: string;
}) {
  const toggle = (id: string) => {
    onChange(selected.includes(id) ? selected.filter((s) => s !== id) : [...selected, id]);
  };

  return (
    <div>
      <label className="block text-sm font-medium text-th-text-tertiary mb-2">{label}</label>
      {options.length === 0 ? (
        <p className="text-xs text-th-text-muted italic">{emptyText}</p>
      ) : (
        <div className="flex flex-wrap gap-2">
          {options.map((opt) => {
            const active = selected.includes(opt.id);
            return (
              <button
                key={opt.id}
                type="button"
                onClick={() => toggle(opt.id)}
                title={opt.detail}
                className={`text-xs px-3 py-1.5 rounded-lg font-medium border transition-all duration-150 ${
                  active
                    ? "bg-th-tab-active-bg text-th-tab-active-fg border-th-tab-active-bg"
                    : "bg-th-inset-bg text-th-text-tertiary border-th-border hover:border-th-border-strong hover:text-th-text-primary"
                }`}
              >
                {opt.label}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

function EditAgentsJsonDialog({ agents, onClose, onSaved }: { agents: AgentSpec[]; onClose: () => void; onSaved: () => void }) {
  const buildJson = () => {
    const obj: Record<string, Record<string, unknown>> = {};
    for (const a of agents) {
      const entry: Record<string, unknown> = {
        description: a.description,
        system_prompt: a.system_prompt,
      };
      if (a.tools?.length) entry.tools = a.tools;
      if (a.skills?.length) entry.skills = a.skills;
      if (a.model_override) entry.model_override = a.model_override;
      if (a.subagent_llm_family) entry.subagent_llm_family = a.subagent_llm_family;
      if (a.mlx_model_id) entry.mlx_model_id = a.mlx_model_id;
      obj[a.name] = entry;
    }
    return JSON.stringify({ agents: obj }, null, 2);
  };

  const [jsonText, setJsonText] = useState(buildJson);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [details, setDetails] = useState<string[]>([]);

  const handleSave = async () => {
    setError("");
    setDetails([]);
    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(jsonText);
    } catch (e: unknown) {
      setError("Invalid JSON: " + (e instanceof SyntaxError ? e.message : String(e)));
      return;
    }
    if (!parsed.agents) {
      setError('JSON must contain an "agents" key');
      return;
    }
    setSaving(true);
    try {
      const res = await api.saveAgentsJson(parsed);
      if (res.error) {
        setError(res.error);
        setDetails(res.details ?? []);
      } else {
        onSaved();
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Save failed");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/30 backdrop-blur-sm flex items-center justify-center z-50 p-8">
      <div className="bg-th-card-bg border border-th-card-border rounded-2xl w-full max-w-2xl p-6 shadow-2xl max-h-[85vh] flex flex-col">
        <div className="flex items-center justify-between mb-4 shrink-0">
          <div>
            <h2 className="text-lg font-bold text-th-text-primary">Edit Agents (JSON)</h2>
            <p className="text-xs text-th-text-tertiary mt-1">{agents.length} agent{agents.length !== 1 ? "s" : ""}. Saving replaces all agents.</p>
          </div>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-th-surface-hover text-th-text-muted hover:text-th-text-secondary transition-colors"><X size={20} /></button>
        </div>

        <textarea
          className="flex-1 w-full px-4 py-3 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-300/30 transition-all font-mono text-xs min-h-[300px] resize-y"
          value={jsonText}
          onChange={(e) => setJsonText(e.target.value)}
          spellCheck={false}
        />

        {error && (
          <div className="mt-3 text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2 shrink-0">
            <p className="font-semibold">{error}</p>
            {details.length > 0 && <ul className="mt-1 list-disc list-inside space-y-0.5 text-red-400/80">{details.map((d, i) => <li key={i}>{d}</li>)}</ul>}
          </div>
        )}

        <div className="flex justify-end gap-2 mt-4 pt-4 border-t border-th-border shrink-0">
          <button className="px-4 py-2 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-sm font-medium transition-colors" onClick={onClose}>Cancel</button>
          <button
            className="px-4 py-2 bg-th-tab-active-bg text-th-tab-active-fg rounded-lg text-sm font-semibold transition-opacity hover:opacity-90 disabled:opacity-40"
            onClick={handleSave}
            disabled={!jsonText.trim() || saving}
          >
            {saving ? "Saving..." : "Save"}
          </button>
        </div>
      </div>
    </div>
  );
}

function Field({ label, value, onChange, placeholder }: { label: string; value: string; onChange: (v: string) => void; placeholder?: string }) {
  return (
    <div>
      <label className="block text-sm font-medium text-th-text-tertiary mb-2">{label}</label>
      <input className="w-full px-4 py-2.5 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-300/30 transition-all text-sm" value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder} />
    </div>
  );
}
