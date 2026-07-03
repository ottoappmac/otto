import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { AlertTriangle, Calendar, Clock, MessageSquare, Search, Wrench, Trash2, X, Loader2, Zap } from "lucide-react";
import { api } from "../hooks/useApi";
import { useNotification } from "../context/NotificationContext";
import { formatRelativeTime } from "../utils/formatRelativeTime";
import type { SessionInfo } from "../types";

const TRIGGER_META: Record<string, { label: string; detail: string; color: string }> = {
  "claude-hook": { label: "Trigger", detail: "Claude hook", color: "bg-blue-500/15 text-blue-400 border-blue-500/25" },
  "oc-watcher-new": { label: "Trigger", detail: "OC watcher · new session", color: "bg-blue-500/15 text-blue-400 border-blue-500/25" },
  "oc-watcher-activity": { label: "Trigger", detail: "OC watcher · activity", color: "bg-blue-500/15 text-blue-400 border-blue-500/25" },
  "schedule": { label: "Schedule", detail: "Scheduled run", color: "bg-blue-500/15 text-blue-400 border-blue-500/25" },
};

export default function HistoryPage() {
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [confirmClearAll, setConfirmClearAll] = useState(false);
  const [clearingAll, setClearingAll] = useState(false);
  const [search, setSearch] = useState("");
  const navigate = useNavigate();
  const { notifications } = useNotification();

  const refresh = () => { api.listSessions().then(setSessions).catch((e) => console.warn("Failed to load sessions:", e)).finally(() => setLoading(false)); };
  useEffect(() => { refresh(); }, []);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return sessions;
    return sessions.filter((s) => {
      const title = s.title.toLowerCase();
      const agent = (s.agent_name || "general purpose").toLowerCase();
      const trigger = (s.trigger_source && TRIGGER_META[s.trigger_source]?.detail || "").toLowerCase();
      const scheduleId = (s.schedule_id || "").toLowerCase();
      return title.includes(q) || agent.includes(q) || trigger.includes(q) || scheduleId.includes(q);
    });
  }, [sessions, search]);

  const handleClose = async (id: string, e: React.MouseEvent) => { e.stopPropagation(); try { await api.closeSession(id); } catch (err) { console.warn("Failed to delete session:", err); } setConfirmDeleteId(null); refresh(); };

  const handleClearAll = async () => {
    setClearingAll(true);
    try {
      await api.clearAllSessions();
      setSessions([]);
      setSearch("");
    } catch (err) {
      console.warn("Failed to clear sessions:", err);
    } finally {
      setClearingAll(false);
      setConfirmClearAll(false);
    }
  };

  const formatDate = (dateStr: string) => formatRelativeTime(dateStr, "long");

  return (
    <div className="h-full flex flex-col">
      <header className="border-b border-th-border px-6 py-4 flex items-center justify-between gap-4 shrink-0 bg-th-bg-secondary">
        <h1 className="text-lg font-bold text-th-text-primary shrink-0">Session History</h1>
        {sessions.length > 0 && (
          <div className="flex items-center gap-3 flex-1 justify-end">
            <div className="relative max-w-xs w-full">
              <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-th-text-muted pointer-events-none" />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search by title, agent, or tag…"
                className="w-full pl-8 pr-8 py-1.5 rounded-lg bg-th-input-bg border border-th-input-border text-sm text-th-text-primary placeholder-th-text-muted focus:outline-none focus:border-blue-400 transition-colors"
              />
              {search && (
                <button onClick={() => setSearch("")} className="absolute right-2 top-1/2 -translate-y-1/2 text-th-text-muted hover:text-th-text-secondary transition-colors">
                  <X size={14} />
                </button>
              )}
            </div>
            {!confirmClearAll ? (
              <button
                onClick={() => setConfirmClearAll(true)}
                className="shrink-0 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-th-border text-xs font-medium text-th-text-muted hover:border-rose-500/40 hover:text-rose-400 hover:bg-rose-500/5 transition-colors"
                title="Delete all sessions"
              >
                <Trash2 size={13} />
                Clear all
              </button>
            ) : (
              <span className="shrink-0 inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border border-rose-500/30 bg-rose-500/8 text-xs">
                <AlertTriangle size={13} className="text-rose-400 shrink-0" />
                <span className="text-rose-300 font-medium">Delete all {sessions.length} sessions?</span>
                <button
                  onClick={() => void handleClearAll()}
                  disabled={clearingAll}
                  className="ml-1 px-2 py-0.5 rounded bg-rose-600 text-white font-semibold hover:bg-rose-500 disabled:opacity-50 transition-colors"
                >
                  {clearingAll ? <Loader2 size={11} className="animate-spin" /> : "Yes, delete"}
                </button>
                <button
                  onClick={() => setConfirmClearAll(false)}
                  disabled={clearingAll}
                  className="p-0.5 text-rose-400/70 hover:text-rose-300 transition-colors"
                  title="Cancel"
                >
                  <X size={13} />
                </button>
              </span>
            )}
          </div>
        )}
      </header>

      <div className="flex-1 overflow-y-auto p-6">
        {loading && sessions.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full -mt-12">
            <Loader2 size={32} className="text-th-text-muted animate-spin mb-4" />
            <p className="text-sm text-th-text-tertiary">Loading sessions...</p>
          </div>
        )}
        {!loading && sessions.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full -mt-12">
            <div className="w-20 h-20 rounded-2xl bg-th-card-bg border border-th-card-border flex items-center justify-center mb-6"><Clock size={36} className="text-th-text-muted" /></div>
            <h3 className="text-lg font-semibold text-th-text-secondary mb-2">No sessions yet</h3>
            <p className="text-sm text-th-text-tertiary max-w-xs text-center">Start a chat to create your first session. Previous conversations will appear here.</p>
          </div>
        )}
        {!loading && sessions.length > 0 && filtered.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full -mt-12">
            <Search size={36} className="text-th-text-muted mb-4" />
            <h3 className="text-lg font-semibold text-th-text-secondary mb-2">No matching sessions</h3>
            <p className="text-sm text-th-text-tertiary max-w-xs text-center">No sessions match "{search}". Try a different search term.</p>
          </div>
        )}
        <div className="space-y-2">
          {filtered.map((session) => (
            <div key={session.id} className="bg-th-card-bg border border-th-card-border rounded-xl px-5 py-4 hover:border-th-border-strong transition-all duration-200 cursor-pointer group" onClick={() => navigate(`/chat/${session.id}`)}>
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-3 flex-1 min-w-0">
                  <div className="relative w-8 h-8 rounded-lg bg-th-inset-bg border border-th-border flex items-center justify-center mt-0.5 shrink-0">
                    <MessageSquare size={14} className="text-th-text-muted" />
                    {notifications[session.id] && (
                      <span
                        className={`absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full border-2 border-th-card-bg ${
                          notifications[session.id] === "error"
                            ? "bg-red-500 animate-pulse"
                            : notifications[session.id] === "hitl"
                              ? "bg-amber-400 animate-pulse"
                              : "bg-emerald-500"
                        }`}
                      />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="font-medium text-th-text-primary truncate group-hover:text-th-text-primary transition-colors">{session.title}</h3>
                    <div className="flex items-center gap-4 mt-1.5 text-xs text-th-text-tertiary">
                      <span className="flex items-center gap-1"><MessageSquare size={11} /> {session.message_count} messages</span>
                      <span className="text-th-text-secondary font-medium">{session.agent_name || "General Purpose"}</span>
                      {session.trigger_source && TRIGGER_META[session.trigger_source] && (
                        <span
                          className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[10px] font-medium border ${TRIGGER_META[session.trigger_source].color}`}
                          title={TRIGGER_META[session.trigger_source].detail}
                        >
                          <Zap size={9} />
                          {TRIGGER_META[session.trigger_source].label}
                        </span>
                      )}
                      {session.tools_used.length > 0 && <span className="flex items-center gap-1"><Wrench size={11} /> {session.tools_used.join(", ")}</span>}
                      {session.schedule_id && (
                        <span
                          className="flex items-center gap-1 text-blue-400 hover:text-blue-300 transition-colors"
                          onClick={(e) => { e.stopPropagation(); navigate("/schedules"); }}
                        >
                          <Calendar size={11} /> {session.schedule_id}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2 ml-4 shrink-0">
                  <span className="text-xs text-th-text-tertiary font-medium">{formatDate(session.updated_at)}</span>
                  {confirmDeleteId === session.id ? (
                    <span className="flex items-center gap-1">
                      <button className="p-1.5 rounded-lg bg-red-500/10 text-red-400 hover:bg-red-500/20 transition-colors" onClick={(e) => handleClose(session.id, e)} title="Confirm delete"><Trash2 size={14} /></button>
                      <button className="p-1.5 rounded-lg text-th-text-muted hover:text-th-text-secondary hover:bg-th-surface-hover transition-colors" onClick={(e) => { e.stopPropagation(); setConfirmDeleteId(null); }} title="Cancel"><X size={14} /></button>
                    </span>
                  ) : (
                    <button className="p-1.5 rounded-lg hover:bg-red-500/10 text-th-text-muted hover:text-red-400 transition-colors opacity-0 group-hover:opacity-100" onClick={(e) => { e.stopPropagation(); setConfirmDeleteId(session.id); }}><Trash2 size={14} /></button>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
