import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Calendar,
  Plus,
  Trash2,
  Edit3,
  X,
  Play,
  Square,
  ChevronRight,
  Loader2,
  FolderOpen,
  Sparkles,
  Paperclip,
  Search,
  ArrowUpDown,
} from "lucide-react";
import { api } from "../hooks/useApi";
import { usePolling } from "../hooks/usePolling";
import { useNotification } from "../context/NotificationContext";
import { useEvalSuggestionsByTarget } from "../hooks/useAmbientHints";
import { CRON_PRESETS, MAX_SCHEDULES } from "../types";
import type { AgentSpec, AmbientHint, ScheduleAttachment, ScheduleSpec } from "../types";

function cronToHuman(cron: string): string {
  const parts = cron.split(" ");
  if (parts.length !== 5) return cron;
  const preset = CRON_PRESETS.find((p) => p.cron === cron);
  if (preset) return preset.label;
  return cron;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
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

type SortField = "name" | "last_run" | "created_at" | "updated_at";
type SortDir = "asc" | "desc";

export default function SchedulesPage() {
  const [schedules, setSchedules] = useState<ScheduleSpec[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [editingSchedule, setEditingSchedule] = useState<ScheduleSpec | null>(null);
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [sortField, setSortField] = useState<SortField>("name");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const { clearScheduleNotifications } = useNotification();
  const navigate = useNavigate();
  const evalSuggestions = useEvalSuggestionsByTarget();

  const refresh = () => {
    api.listSchedules()
      .then(setSchedules)
      .catch((e) => console.warn("Failed to load schedules:", e))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    clearScheduleNotifications();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  usePolling(refresh, 10000);

  const filteredAndSorted = useMemo(() => {
    const q = search.trim().toLowerCase();
    const filtered = q
      ? schedules.filter(
          (s) =>
            s.id.toLowerCase().includes(q) ||
            (s.agent_name ?? "").toLowerCase().includes(q) ||
            s.prompt.toLowerCase().includes(q),
        )
      : schedules;

    return [...filtered].sort((a, b) => {
      let cmp = 0;
      if (sortField === "name") {
        cmp = a.id.localeCompare(b.id);
      } else if (sortField === "last_run") {
        const ta = a.last_run ? new Date(a.last_run).getTime() : 0;
        const tb = b.last_run ? new Date(b.last_run).getTime() : 0;
        cmp = ta - tb;
      } else if (sortField === "created_at") {
        cmp = new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
      } else if (sortField === "updated_at") {
        cmp = new Date(a.updated_at).getTime() - new Date(b.updated_at).getTime();
      }
      return sortDir === "asc" ? cmp : -cmp;
    });
  }, [schedules, search, sortField, sortDir]);

  const toggleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortField(field);
      setSortDir("asc");
    }
  };

  const handleDelete = async (id: string) => {
    setError(null);
    try {
      await api.deleteSchedule(id);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Delete failed");
    }
    setConfirmDeleteId(null);
    refresh();
  };

  const handleToggle = async (id: string) => {
    try {
      await api.toggleSchedule(id);
      refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Toggle failed");
    }
  };

  const handleRunNow = async (id: string) => {
    setSchedules((prev) =>
      prev.map((s) => (s.id === id ? { ...s, last_status: "running" } : s)),
    );
    try {
      await api.runScheduleNow(id);
      refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Run failed");
      refresh();
    }
  };

  const handleStop = async (id: string) => {
    try {
      await api.stopScheduleRun(id);
    } catch {
      // 409 = no active run — silently ignore and refresh to sync UI
    }
    refresh();
  };

  const SortButton = ({ field, label }: { field: SortField; label: string }) => (
    <button
      onClick={() => toggleSort(field)}
      className={`flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-medium border transition-all ${
        sortField === field
          ? "bg-th-tab-active-bg text-th-tab-active-fg border-th-tab-active-bg"
          : "bg-th-inset-bg text-th-text-tertiary border-th-border hover:border-th-border-strong hover:text-th-text-primary"
      }`}
    >
      {label}
      <ArrowUpDown size={11} className={sortField === field ? "opacity-100" : "opacity-40"} />
      {sortField === field && (
        <span className="text-[10px] opacity-70">{sortDir === "asc" ? "↑" : "↓"}</span>
      )}
    </button>
  );

  return (
    <div className="h-full flex flex-col">
      <header className="border-b border-th-border px-6 py-4 shrink-0 bg-th-bg-secondary">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <h1 className="text-lg font-bold text-th-text-primary">Schedules</h1>
            <span className="text-xs text-th-text-tertiary font-medium">{schedules.length}/{MAX_SCHEDULES}</span>
          </div>
          <button
            className="px-4 py-2 bg-th-tab-active-bg text-th-tab-active-fg rounded-lg text-sm font-semibold transition-opacity hover:opacity-90 flex items-center gap-2 disabled:opacity-40"
            onClick={() => { setEditingSchedule(null); setShowCreate(true); }}
            disabled={schedules.length >= MAX_SCHEDULES}
          >
            <Plus size={15} /> Create Schedule
          </button>
        </div>
        <div className="flex items-center gap-3 flex-wrap">
          <div className="relative flex-1 min-w-[180px] max-w-xs">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-th-text-muted pointer-events-none" />
            <input
              type="text"
              placeholder="Search schedules..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full pl-8 pr-3 py-1.5 bg-th-input-bg border border-th-input-border rounded-lg text-sm text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-300/30 transition-all"
            />
            {search && (
              <button
                onClick={() => setSearch("")}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-th-text-muted hover:text-th-text-secondary"
              >
                <X size={13} />
              </button>
            )}
          </div>
          <div className="flex items-center gap-1.5 text-xs text-th-text-muted">
            <span className="shrink-0">Sort:</span>
            <SortButton field="name" label="Name" />
            <SortButton field="last_run" label="Last run" />
            <SortButton field="created_at" label="Created" />
            <SortButton field="updated_at" label="Updated" />
          </div>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto p-6">
        {error && (
          <div className="mb-4 flex items-center justify-between gap-3 px-4 py-3 bg-red-500/10 border border-red-500/20 rounded-lg">
            <span className="text-sm text-red-400">{error}</span>
            <button className="text-red-400/60 hover:text-red-400 transition-colors" onClick={() => setError(null)}><X size={16} /></button>
          </div>
        )}

        {loading && schedules.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full -mt-12">
            <Loader2 size={32} className="text-th-text-muted animate-spin mb-4" />
            <p className="text-sm text-th-text-tertiary">Loading schedules...</p>
          </div>
        )}

        {!loading && schedules.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full -mt-12">
            <div className="w-20 h-20 rounded-2xl bg-th-card-bg border border-th-card-border flex items-center justify-center mb-6">
              <Calendar size={36} className="text-th-text-muted" />
            </div>
            <h3 className="text-lg font-semibold text-th-text-secondary mb-2">No schedules yet</h3>
            <p className="text-sm text-th-text-tertiary mb-6 max-w-xs text-center">
              Create a schedule to run agents automatically on a cron schedule.
            </p>
            <button
              className="px-5 py-2.5 bg-th-tab-active-bg text-th-tab-active-fg rounded-lg text-sm font-semibold transition-opacity hover:opacity-90 flex items-center gap-2"
              onClick={() => { setEditingSchedule(null); setShowCreate(true); }}
            >
              <Plus size={15} /> Create your first schedule
            </button>
          </div>
        )}

        {!loading && schedules.length > 0 && filteredAndSorted.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full -mt-12">
            <Search size={32} className="text-th-text-muted mb-4" />
            <p className="text-sm text-th-text-tertiary">No schedules match <span className="text-th-text-secondary font-medium">"{search}"</span></p>
            <button className="mt-3 text-xs text-blue-400 hover:text-blue-300 transition-colors" onClick={() => setSearch("")}>Clear search</button>
          </div>
        )}

        <div className="space-y-3">
          {filteredAndSorted.map((schedule) => (
            <ScheduleCard
              key={schedule.id}
              schedule={schedule}
              evalSuggestion={evalSuggestions.get(schedule.id)}
              onToggle={() => handleToggle(schedule.id)}
              onRunNow={() => handleRunNow(schedule.id)}
              onStop={() => handleStop(schedule.id)}
              onEdit={() => { setEditingSchedule(schedule); setShowCreate(true); }}
              onOpenFolder={() => api.openScheduleFolder(schedule.id).catch(() => {})}
              onViewRuns={() => navigate(`/schedules/${encodeURIComponent(schedule.id)}/runs`)}
              onDelete={() => confirmDeleteId === schedule.id ? handleDelete(schedule.id) : setConfirmDeleteId(schedule.id)}
              confirmingDelete={confirmDeleteId === schedule.id}
              onCancelDelete={() => setConfirmDeleteId(null)}
            />
          ))}
        </div>
      </div>

      {showCreate && (
        <ScheduleDialog
          schedule={editingSchedule}
          onClose={() => setShowCreate(false)}
          onSaved={() => { setShowCreate(false); refresh(); }}
        />
      )}
    </div>
  );
}


function StatusDot({ status, enabled }: { status: string | null; enabled: boolean }) {
  if (!enabled) return <span className="w-2 h-2 rounded-full bg-neutral-600" />;
  if (status === "running") return <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />;
  if (status === "success") return <span className="w-2 h-2 rounded-full bg-emerald-400" />;
  if (status === "error") return <span className="w-2 h-2 rounded-full bg-red-400" />;
  if (status === "cancelled") return <span className="w-2 h-2 rounded-full bg-amber-400" />;
  return <span className="w-2 h-2 rounded-full bg-amber-400" />;
}


function ScheduleCard({
  schedule,
  evalSuggestion,
  onToggle,
  onRunNow,
  onStop,
  onEdit,
  onDelete,
  onOpenFolder,
  onViewRuns,
  confirmingDelete,
  onCancelDelete,
}: {
  schedule: ScheduleSpec;
  evalSuggestion?: AmbientHint;
  onToggle: () => void;
  onRunNow: () => void;
  onStop: () => void;
  onEdit: () => void;
  onDelete: () => void;
  onOpenFolder: () => void;
  onViewRuns: () => void;
  confirmingDelete: boolean;
  onCancelDelete: () => void;
}) {
  const navigate = useNavigate();
  const isRunning = schedule.last_status === "running";
  return (
    <div className={`bg-th-card-bg border rounded-xl transition-all duration-200 ${schedule.enabled ? "border-th-card-border hover:border-th-border-strong" : "border-th-border/60 opacity-60"}`}>
      <div className="p-5">
        <div className="flex items-start justify-between">
          <div className="flex items-start gap-3 flex-1 min-w-0 cursor-pointer" onClick={onViewRuns}>
            <div className="w-10 h-10 rounded-lg bg-th-inset-bg border border-th-border flex items-center justify-center mt-0.5 shrink-0">
              <Calendar size={20} className="text-th-text-secondary" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <StatusDot status={schedule.last_status} enabled={schedule.enabled} />
                <h3 className="font-semibold text-th-text-primary truncate">{schedule.id}</h3>
                {isRunning && (
                  <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500/15 text-emerald-400 border border-emerald-500/25 animate-pulse shrink-0">
                    Running
                  </span>
                )}
              </div>
              <p className="text-sm text-th-text-tertiary mt-1">
                <span className="text-th-text-secondary font-medium">{schedule.agent_name || "General Purpose"}</span>
                <span className="mx-2 text-th-text-faint">·</span>
                {cronToHuman(schedule.cron_expression)}
              </p>
              <p className="text-xs text-th-text-tertiary mt-1.5 truncate">{schedule.prompt}</p>
              {schedule.last_run && (
                <div className="flex items-center gap-1.5 mt-1 flex-wrap">
                  <span className="text-xs text-th-text-muted">Last run: {formatRelative(schedule.last_run)}</span>
                  {schedule.last_status && (
                    <span className={`inline-flex items-center px-1.5 py-0.5 rounded-full text-[10px] font-semibold border ${
                      schedule.last_status === "success"
                        ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                        : schedule.last_status === "error"
                        ? "bg-red-500/10 text-red-400 border-red-500/20"
                        : schedule.last_status === "running"
                        ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20 animate-pulse"
                        : "bg-amber-500/10 text-amber-400 border-amber-500/20"
                    }`}>
                      {schedule.last_status}
                    </span>
                  )}
                </div>
              )}
              {schedule.last_error && (
                <p className="text-[11px] text-red-400/80 mt-1 line-clamp-2 break-words">{schedule.last_error}</p>
              )}
            </div>
          </div>

          <div className="flex items-center gap-1 shrink-0 ml-4">
            {isRunning ? (
              <button
                onClick={onStop}
                className="p-2 rounded-lg hover:bg-red-500/10 text-red-400 hover:text-red-300 transition-colors"
                title="Stop running job"
              >
                <Square size={15} />
              </button>
            ) : (
              <button
                onClick={onRunNow}
                className="p-2 rounded-lg hover:bg-th-surface-hover text-th-text-muted hover:text-emerald-400 transition-colors"
                title="Run now"
              >
                <Play size={15} />
              </button>
            )}
            <button onClick={onEdit} className="p-2 rounded-lg hover:bg-th-surface-hover text-th-text-muted hover:text-th-text-secondary transition-colors" title="Edit">
              <Edit3 size={15} />
            </button>
            <button onClick={onOpenFolder} className="p-2 rounded-lg hover:bg-th-surface-hover text-th-text-muted hover:text-th-text-secondary transition-colors" title="Open schedule folder">
              <FolderOpen size={15} />
            </button>
            {confirmingDelete ? (
              <span className="flex items-center gap-1.5">
                <span className="text-xs text-red-400">Delete?</span>
                <button className="px-2.5 py-1.5 bg-red-500/20 border border-red-500/30 text-red-400 hover:bg-red-500/30 rounded-lg text-xs font-semibold transition-colors" onClick={onDelete}>Yes</button>
                <button className="px-2.5 py-1.5 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-xs font-medium transition-colors" onClick={onCancelDelete}>No</button>
              </span>
            ) : (
              <button onClick={onDelete} className="p-2 rounded-lg hover:bg-red-500/10 text-th-text-muted hover:text-red-400 transition-colors">
                <Trash2 size={15} />
              </button>
            )}
            <button
              onClick={onViewRuns}
              className="p-2 rounded-lg hover:bg-th-surface-hover text-th-text-muted hover:text-th-text-secondary transition-colors"
              title="View runs"
            >
              <ChevronRight size={15} />
            </button>
          </div>
        </div>

        <div className="flex items-center justify-between mt-3">
          {evalSuggestion ? (
            <button
              onClick={() => navigate(`/ambient?highlight=${encodeURIComponent(evalSuggestion.id)}`)}
              className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-amber-500/10 border border-amber-500/25 text-[11px] font-medium text-amber-300 hover:bg-amber-500/20 transition-all"
              title="An improved prompt was suggested for this schedule"
            >
              <Sparkles size={11} aria-hidden />
              Prompt suggestion available
            </button>
          ) : (
            <span />
          )}
          <button
            onClick={onToggle}
            className="flex items-center gap-2 group"
            title={schedule.enabled ? "Disable schedule" : "Enable schedule"}
          >
            <span className="text-[11px] text-th-text-muted group-hover:text-th-text-tertiary transition-colors">
              {schedule.enabled ? "Enabled" : "Disabled"}
            </span>
            <div className={`w-8 h-[18px] rounded-full relative transition-colors duration-200 ${schedule.enabled ? "bg-emerald-500" : "bg-neutral-700"}`}>
              <div className={`absolute top-[2px] w-[14px] h-[14px] rounded-full bg-white shadow-sm transition-transform duration-200 ${schedule.enabled ? "translate-x-[16px]" : "translate-x-[2px]"}`} />
            </div>
          </button>
        </div>
      </div>
    </div>
  );
}



function ScheduleDialog({
  schedule,
  onClose,
  onSaved,
}: {
  schedule: ScheduleSpec | null;
  onClose: () => void;
  onSaved: () => void;
}) {
  const [id, setId] = useState(schedule?.id ?? "");
  const [agentName, setAgentName] = useState(schedule?.agent_name ?? "");
  const [prompt, setPrompt] = useState(schedule?.prompt ?? "");
  const [cron, setCron] = useState(schedule?.cron_expression ?? "0 9 * * *");
  const [selectedPreset, setSelectedPreset] = useState(() => {
    const match = CRON_PRESETS.find((p) => p.cron === (schedule?.cron_expression ?? "0 9 * * *"));
    return match?.id ?? "custom";
  });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [agents, setAgents] = useState<AgentSpec[]>([]);
  const [attachments, setAttachments] = useState<ScheduleAttachment[]>([]);
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    api.listAgents().then(setAgents).catch(() => {});
  }, []);

  useEffect(() => {
    if (schedule) {
      api.listScheduleAttachments(schedule.id).then(setAttachments).catch(() => {});
    }
  }, [schedule]);

  const handleAddFiles = (files: FileList | null) => {
    if (!files) return;
    setPendingFiles((prev) => [...prev, ...Array.from(files)]);
  };

  const removePendingFile = (idx: number) => {
    setPendingFiles((prev) => prev.filter((_, i) => i !== idx));
  };

  const removeExistingAttachment = async (path: string) => {
    if (!schedule) return;
    try {
      await api.deleteScheduleAttachment(schedule.id, path);
      setAttachments((prev) => prev.filter((a) => a.path !== path));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Delete failed");
    }
  };

  const uploadPendingFiles = async (scheduleId: string) => {
    for (const file of pendingFiles) {
      await api.uploadScheduleAttachment(scheduleId, file.name, file);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    setUploading(pendingFiles.length > 0);
    setError("");
    try {
      let targetId: string;
      if (schedule) {
        await api.updateSchedule(schedule.id, {
          prompt,
          cron_expression: cron,
          agent_name: agentName || null,
        });
        targetId = schedule.id;
      } else {
        await api.createSchedule({
          id,
          agent_name: agentName || null,
          prompt,
          cron_expression: cron,
        });
        targetId = id;
      }
      await uploadPendingFiles(targetId);
      onSaved();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Save failed");
    } finally {
      setSaving(false);
      setUploading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/30 backdrop-blur-sm flex items-center justify-center z-50 p-8">
      <div className="bg-th-card-bg border border-th-card-border rounded-2xl w-[900px] max-w-[95vw] min-w-[400px] h-[80vh] max-h-[92vh] min-h-[400px] overflow-auto resize p-6 shadow-2xl">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-lg font-bold text-th-text-primary">{schedule ? "Edit Schedule" : "Create Schedule"}</h2>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-th-surface-hover text-th-text-muted hover:text-th-text-secondary transition-colors"><X size={20} /></button>
        </div>

        <div className="space-y-4">
          {!schedule && (
            <div>
              <label className="block text-sm font-medium text-th-text-tertiary mb-2">Schedule ID</label>
              <input
                className="w-full px-4 py-2.5 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-300/30 transition-all text-sm"
                value={id}
                onChange={(e) => setId(e.target.value.replace(/[^A-Za-z0-9 _-]/g, ""))}
                placeholder="daily-regression"
              />
            </div>
          )}

          <div>
            <label className="block text-sm font-medium text-th-text-tertiary mb-2">Agent</label>
            <select
              className="w-full px-4 py-2.5 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary focus:outline-none focus:border-blue-400 text-sm"
              value={agentName}
              onChange={(e) => setAgentName(e.target.value)}
            >
              <option value="">General Purpose</option>
              {agents.map((a) => <option key={a.name} value={a.name}>{a.name}</option>)}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-th-text-tertiary mb-2">Prompt</label>
            <textarea
              className="w-full px-4 py-3 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-300/30 transition-all min-h-[140px] resize-y text-sm"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="What should the agent do each time it runs?"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-th-text-tertiary mb-2">Schedule</label>
            <div className="flex flex-wrap gap-2">
              {CRON_PRESETS.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => { setSelectedPreset(p.id); if (p.cron) setCron(p.cron); }}
                  className={`text-xs px-3 py-1.5 rounded-lg font-medium border transition-all duration-150 ${
                    selectedPreset === p.id
                      ? "bg-th-tab-active-bg text-th-tab-active-fg border-th-tab-active-bg"
                      : "bg-th-inset-bg text-th-text-tertiary border-th-border hover:border-th-border-strong hover:text-th-text-primary"
                  }`}
                >
                  {p.label}
                </button>
              ))}
            </div>
            {selectedPreset === "custom" && (
              <input
                className="w-full mt-2 px-4 py-2.5 bg-th-input-bg border border-th-input-border rounded-lg text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-400 text-sm font-mono"
                value={cron}
                onChange={(e) => setCron(e.target.value)}
                placeholder="0 9 * * 1-5"
              />
            )}
            <p className="text-[11px] text-th-text-muted mt-1.5">Cron: {cron}</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-th-text-tertiary mb-2">Attachments</label>
            <p className="text-[11px] text-th-text-muted mb-2">
              Files attached here are stored once per schedule and made available read-only to every run under <code className="font-mono">/attachments/</code>.
            </p>
            {(attachments.length > 0 || pendingFiles.length > 0) && (
              <div className="space-y-1.5 mb-2">
                {attachments.map((a) => (
                  <div key={a.path} className="flex items-center gap-2 px-3 py-1.5 bg-th-inset-bg border border-th-border rounded-lg text-sm">
                    <Paperclip size={13} className="text-th-text-muted shrink-0" />
                    <span className="text-th-text-secondary truncate flex-1">{a.path}</span>
                    <span className="text-[11px] text-th-text-muted shrink-0">{formatBytes(a.size)}</span>
                    <button
                      type="button"
                      onClick={() => removeExistingAttachment(a.path)}
                      className="p-1 rounded hover:bg-red-500/10 text-th-text-muted hover:text-red-400 transition-colors shrink-0"
                      title="Remove attachment"
                    >
                      <X size={13} />
                    </button>
                  </div>
                ))}
                {pendingFiles.map((f, i) => (
                  <div key={`pending-${i}`} className="flex items-center gap-2 px-3 py-1.5 bg-blue-500/5 border border-blue-500/20 rounded-lg text-sm">
                    <Paperclip size={13} className="text-blue-400 shrink-0" />
                    <span className="text-th-text-secondary truncate flex-1">{f.name}</span>
                    <span className="text-[11px] text-th-text-muted shrink-0">{formatBytes(f.size)}</span>
                    <span className="text-[10px] text-blue-400 shrink-0">pending</span>
                    <button
                      type="button"
                      onClick={() => removePendingFile(i)}
                      className="p-1 rounded hover:bg-red-500/10 text-th-text-muted hover:text-red-400 transition-colors shrink-0"
                      title="Remove"
                    >
                      <X size={13} />
                    </button>
                  </div>
                ))}
              </div>
            )}
            <label className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary hover:border-th-border-strong rounded-lg text-xs font-medium transition-colors cursor-pointer">
              <Plus size={13} />
              Add files
              <input
                type="file"
                multiple
                className="hidden"
                onChange={(e) => { handleAddFiles(e.target.files); e.target.value = ""; }}
              />
            </label>
          </div>

          {error && (
            <div className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
              {error}
            </div>
          )}

          <div className="flex justify-end gap-2 pt-3 border-t border-th-border">
            <button className="px-4 py-2 bg-th-inset-bg border border-th-border text-th-text-tertiary hover:text-th-text-primary rounded-lg text-sm font-medium transition-colors" onClick={onClose}>Cancel</button>
            <button
              className="px-4 py-2 bg-th-tab-active-bg text-th-tab-active-fg rounded-lg text-sm font-semibold transition-opacity hover:opacity-90 disabled:opacity-40"
              onClick={handleSave}
              disabled={(!schedule && !id) || !prompt.trim() || !cron || saving}
            >
              {uploading ? "Uploading..." : saving ? "Saving..." : schedule ? "Save" : "Create Schedule"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
