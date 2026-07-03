import { useEffect, useMemo, useState } from "react";
import { BookOpen, Plus, Trash2, Edit3, Sparkles, X, FileJson, Loader2, Lock, Zap } from "lucide-react";
import { api } from "../hooks/useApi";
import type { SkillSpec } from "../types";

export default function SkillsPage({ embedded }: { embedded?: boolean } = {}) {
  const [skills, setSkills] = useState<SkillSpec[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [editingSkill, setEditingSkill] = useState<SkillSpec | null>(null);
  const [showJsonEditor, setShowJsonEditor] = useState(false);
  const [confirmDeleteName, setConfirmDeleteName] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const refresh = () => { api.listSkills().then(setSkills).catch((e) => console.warn("Failed to load skills:", e)).finally(() => setLoading(false)); };
  useEffect(() => { refresh(); }, []);

  const handleDelete = async (name: string) => {
    setDeleteError(null);
    try {
      await api.deleteSkill(name);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Delete failed";
      setDeleteError(msg.replace(/^API \d+:\s*/, "").replace(/^\{.*"error":\s*"/, "").replace(/"\}$/, ""));
    }
    setConfirmDeleteName(null);
    refresh();
  };

  const HIDDEN_SKILL_IDS = new Set(["claude-session-eval", "openclaw-session-eval"]);
  const customSkills = useMemo(() => skills.filter((s) => !s.builtin && !HIDDEN_SKILL_IDS.has(s.name)), [skills]);
  const managedSkills = useMemo(() => skills.filter((s) => s.builtin && !HIDDEN_SKILL_IDS.has(s.name)), [skills]);

  const body = (
    <>
      {!embedded && (
        <header className="border-b border-th-border px-6 py-4 flex items-center justify-between shrink-0 bg-th-bg-secondary">
          <h1 className="text-lg font-bold text-th-text-primary">Skills</h1>
          <div className="flex items-center gap-2">
            <button className="px-3 py-2 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-sm font-medium transition-colors flex items-center gap-1.5" onClick={() => setShowJsonEditor(true)}><FileJson size={14} /> Edit JSON</button>
            <button className="px-4 py-2 bg-th-tab-active-bg text-th-tab-active-fg rounded-lg text-sm font-semibold transition-opacity hover:opacity-90 flex items-center gap-2" onClick={() => { setEditingSkill(null); setShowCreate(true); }}><Plus size={15} /> Create Skill</button>
          </div>
        </header>
      )}
      {embedded && (
        <div className="px-6 pt-4 pb-3 flex items-center justify-end gap-2 border-b border-th-border shrink-0">
          <button className="px-3 py-2 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-sm font-medium transition-colors flex items-center gap-1.5" onClick={() => setShowJsonEditor(true)}><FileJson size={14} /> Edit JSON</button>
          <button className="px-4 py-2 bg-th-tab-active-bg text-th-tab-active-fg rounded-lg text-sm font-semibold transition-opacity hover:opacity-90 flex items-center gap-2" onClick={() => { setEditingSkill(null); setShowCreate(true); }}><Plus size={15} /> Create Skill</button>
        </div>
      )}
      <div className="flex-1 overflow-y-auto p-6">
        {deleteError && (
          <div className="mb-4 flex items-center justify-between gap-3 px-4 py-3 bg-red-500/10 border border-red-500/20 rounded-lg">
            <span className="text-sm text-red-400">{deleteError}</span>
            <button className="text-red-400/60 hover:text-red-400 transition-colors" onClick={() => setDeleteError(null)}><X size={16} /></button>
          </div>
        )}
        {loading && skills.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full -mt-12">
            <Loader2 size={32} className="text-th-text-muted animate-spin mb-4" />
            <p className="text-sm text-th-text-tertiary">Loading skills...</p>
          </div>
        )}
        {!loading && skills.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full -mt-12">
            <div className="w-20 h-20 rounded-2xl bg-th-card-bg border border-th-card-border flex items-center justify-center mb-6"><BookOpen size={36} className="text-th-text-muted" /></div>
            <h3 className="text-lg font-semibold text-th-text-secondary mb-2">No skills yet</h3>
            <p className="text-sm text-th-text-tertiary mb-6 max-w-xs text-center">Create skills to capture domain knowledge and procedures for your agents.</p>
            <button className="px-5 py-2.5 bg-th-tab-active-bg text-th-tab-active-fg rounded-lg text-sm font-semibold transition-opacity hover:opacity-90 flex items-center gap-2" onClick={() => { setEditingSkill(null); setShowCreate(true); }}><Plus size={15} /> Create your first skill</button>
          </div>
        )}
        <div className="space-y-3">
          {customSkills.map((skill) => (
            <SkillListCard
              key={skill.name}
              skill={skill}
              onEdit={() => { setEditingSkill(skill); setShowCreate(true); }}
              confirmDeleteName={confirmDeleteName}
              setConfirmDeleteName={setConfirmDeleteName}
              handleDelete={handleDelete}
              deletable
            />
          ))}
          {managedSkills.length > 0 && (
            <>
              <div className="flex items-center gap-3 mt-6 mb-3">
                <div className="flex-1 border-t border-th-border" />
                <span className="text-sm uppercase tracking-widest text-th-text-muted font-semibold flex items-center gap-1.5"><Zap size={14} /> Managed</span>
                <div className="flex-1 border-t border-th-border" />
              </div>
              {managedSkills.map((skill) => (
                <SkillListCard
                  key={skill.name}
                  skill={skill}
                  onEdit={() => { setEditingSkill(skill); setShowCreate(true); }}
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
      {showCreate && <SkillDialog skill={editingSkill} onClose={() => setShowCreate(false)} onSaved={() => { setShowCreate(false); refresh(); }} />}
      {showJsonEditor && <EditSkillsJsonDialog skills={skills} onClose={() => setShowJsonEditor(false)} onSaved={() => { setShowJsonEditor(false); refresh(); }} />}
    </>
  );

  if (embedded) return <>{body}</>;
  return <div className="h-full flex flex-col">{body}</div>;
}

function SkillListCard({
  skill,
  onEdit,
  confirmDeleteName,
  setConfirmDeleteName,
  handleDelete,
  deletable,
}: {
  skill: SkillSpec;
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
          <div className="w-10 h-10 rounded-lg bg-th-inset-bg border border-th-border flex items-center justify-center mt-0.5 shrink-0"><BookOpen size={20} className="text-th-text-secondary" /></div>
          <div>
            <h3 className="font-semibold text-th-text-primary">{skill.name}</h3>
            <p className="text-sm text-th-text-tertiary mt-1 leading-relaxed">{skill.description}</p>
          </div>
        </div>
        <div className="flex gap-1 shrink-0 ml-4 items-center">
          <button type="button" className="p-2 rounded-lg hover:bg-th-surface-hover text-th-text-muted hover:text-th-text-secondary transition-colors" onClick={onEdit}><Edit3 size={15} /></button>
          {!deletable ? (
            <span
              className="p-2 rounded-lg text-th-text-faint cursor-not-allowed"
              title="Built-in skill — managed by the app and cannot be deleted."
            >
              <Lock size={15} />
            </span>
          ) : confirmDeleteName === skill.name ? (
            <span className="flex items-center gap-1.5">
              <span className="text-xs text-red-400">Delete?</span>
              <button type="button" className="px-2.5 py-1.5 bg-red-500/20 border border-red-500/30 text-red-400 hover:bg-red-500/30 rounded-lg text-xs font-semibold transition-colors" onClick={() => handleDelete(skill.name)}>Yes</button>
              <button type="button" className="px-2.5 py-1.5 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-xs font-medium transition-colors" onClick={() => setConfirmDeleteName(null)}>No</button>
            </span>
          ) : (
            <button type="button" className="p-2 rounded-lg hover:bg-red-500/10 text-th-text-muted hover:text-red-400 transition-colors" onClick={() => setConfirmDeleteName(skill.name)}><Trash2 size={15} /></button>
          )}
        </div>
      </div>
    </div>
  );
}

function SkillDialog({ skill, onClose, onSaved }: { skill: SkillSpec | null; onClose: () => void; onSaved: () => void }) {
  const [step, setStep] = useState<"describe" | "edit">(skill ? "edit" : "describe");
  const [userDescription, setUserDescription] = useState("");
  const [generating, setGenerating] = useState(false);
  const [name, setName] = useState(skill?.name ?? "");
  const [description, setDescription] = useState(skill?.description ?? "");
  const [content, setContent] = useState(skill?.content ?? "");
  const [saving, setSaving] = useState(false);

  const handleGenerate = async () => {
    setGenerating(true);
    try {
      const data = await api.generateSkill(userDescription);
      if (data.content) { setContent(data.content); const lines = data.content.split("\n"); for (const line of lines) { if (line.startsWith("name:")) setName(line.split(":")[1].trim()); if (line.startsWith("description:")) setDescription(line.split(":").slice(1).join(":").trim()); } }
      setStep("edit");
    } catch (e) { alert(e instanceof Error ? e.message : "Generation failed"); } finally { setGenerating(false); }
  };

  const handleSave = async () => {
    setSaving(true);
    try { const payload = { name, description, content }; if (skill) { await api.updateSkill(skill.name, payload); } else { await api.createSkill(payload); } onSaved(); } finally { setSaving(false); }
  };

  return (
    <div className="fixed inset-0 bg-black/30 backdrop-blur-sm flex items-center justify-center z-50 p-8">
      <div className="bg-th-card-bg border border-th-card-border rounded-2xl w-full max-w-2xl max-h-[90vh] overflow-y-auto p-6 shadow-2xl">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-lg font-bold text-th-text-primary">{skill ? "Edit Skill" : "Create Skill"}</h2>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-th-surface-hover text-th-text-muted hover:text-th-text-secondary transition-colors"><X size={20} /></button>
        </div>
        {step === "describe" && (
          <div className="space-y-4">
            <div><label className="block text-sm font-medium text-th-text-tertiary mb-2">What knowledge should this skill capture?</label><textarea className="w-full px-4 py-3 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-400 transition-all min-h-[120px] text-sm" value={userDescription} onChange={(e) => setUserDescription(e.target.value)} placeholder="Describe the domain knowledge..." /></div>
            <div className="flex justify-between">
              <button className="px-4 py-2 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-sm font-medium transition-colors" onClick={() => setStep("edit")}>Skip — manual setup</button>
              <button className="px-4 py-2 bg-th-tab-active-bg text-th-tab-active-fg rounded-lg text-sm font-semibold transition-opacity hover:opacity-90 flex items-center gap-2 disabled:opacity-40" onClick={handleGenerate} disabled={!userDescription.trim() || generating}><Sparkles size={14} />{generating ? "Generating..." : "Generate Skill"}</button>
            </div>
          </div>
        )}
        {step === "edit" && (
          <div className="space-y-4">
            <Field label="Name" value={name} onChange={setName} placeholder="my-skill-name" />
            <Field label="Description" value={description} onChange={setDescription} />
            <div><label className="block text-sm font-medium text-th-text-tertiary mb-2">SKILL.md Content</label><textarea className="w-full px-4 py-3 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-400 transition-all min-h-[300px] font-mono text-xs" value={content} onChange={(e) => setContent(e.target.value)} /></div>
            <div className="flex justify-end gap-2 pt-3 border-t border-th-border">
              <button className="px-4 py-2 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-sm font-medium transition-colors" onClick={onClose}>Cancel</button>
              <button className="px-4 py-2 bg-th-tab-active-bg text-th-tab-active-fg rounded-lg text-sm font-semibold transition-opacity hover:opacity-90 disabled:opacity-40" onClick={handleSave} disabled={!name || saving}>{saving ? "Saving..." : "Save Skill"}</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function EditSkillsJsonDialog({ skills, onClose, onSaved }: { skills: SkillSpec[]; onClose: () => void; onSaved: () => void }) {
  const buildJson = () => {
    const obj: Record<string, Record<string, unknown>> = {};
    for (const s of skills) {
      const entry: Record<string, unknown> = { description: s.description, content: s.content };
      obj[s.name] = entry;
    }
    return JSON.stringify({ skills: obj }, null, 2);
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
    if (!parsed.skills) {
      setError('JSON must contain a "skills" key');
      return;
    }
    setSaving(true);
    try {
      const res = await api.saveSkillsJson(parsed);
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
            <h2 className="text-lg font-bold text-th-text-primary">Edit Skills (JSON)</h2>
            <p className="text-xs text-th-text-tertiary mt-1">{skills.length} skill{skills.length !== 1 ? "s" : ""}. Saving replaces all skills.</p>
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
  return (<div><label className="block text-sm font-medium text-th-text-tertiary mb-2">{label}</label><input className="w-full px-4 py-2.5 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-400 transition-all text-sm" value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder} /></div>);
}
