import { useState } from "react";
import { NavLink, useNavigate, useLocation } from "react-router-dom";
import {
  MessageSquare,
  LayoutDashboard,
  Settings,
  Workflow,
  Calendar,
  Plus,
  X,
  Trash2,
  PanelRightClose,
  PanelRightOpen,
  Power,
  Zap,
  Moon,
  Sun,
  Activity,
  Monitor,
  Sparkles,
  Mic,
} from "lucide-react";
import { api } from "../hooks/useApi";
import { usePolling } from "../hooks/usePolling";
import { useNotification } from "../context/NotificationContext";
import { useTheme } from "../context/ThemeContext";
import { formatRelativeTime } from "../utils/formatRelativeTime";
import { useAmbientHints } from "../hooks/useAmbientHints";
import { useAmbientSweepStatus } from "../hooks/useAmbientSweepStatus";
import NotificationCenter from "./NotificationCenter";

const NAV_ITEMS = [
  { to: "/dashboard", icon: LayoutDashboard, label: "Dashboard" },
  { to: "/runs", icon: Activity, label: "Runs" },
  { to: "/chat", icon: MessageSquare, label: "Chat" },
  { to: "/ambient", icon: Sparkles, label: "Suggestions" },
  { to: "/agents", icon: Workflow, label: "Agents" },
  { to: "/schedules", icon: Calendar, label: "Schedules" },
  { to: "/triggers", icon: Zap, label: "Triggers" },
  { to: "/activity", icon: Monitor, label: "Activity" },
] as const;

interface RecentSession {
  id: string;
  title: string;
  agent_name: string | null;
  trigger_source: string | null;
  created_at: string;
}

const MAX_RECENT_SESSIONS = 5;
const RECENT_POLL_MS = 10_000;

const isMac =
  typeof navigator !== "undefined" &&
  /mac|iphone|ipad|ipod/i.test(navigator.platform || navigator.userAgent || "");

function triggerMeta(source: string | null): { label: string; color: string } | null {
  if (!source) return null;
  if (source === "schedule") {
    return { label: "Schedule", color: "bg-blue-500/15 text-blue-300 border border-blue-500/25" };
  }
  if (source === "ambient") {
    return { label: "Ambient", color: "bg-blue-500/15 text-blue-300 border border-blue-500/25" };
  }
  if (source === "voice") {
    return { label: "Voice", color: "bg-violet-500/15 text-violet-300 border border-violet-500/25" };
  }
  return { label: "Trigger", color: "bg-blue-500/15 text-blue-300 border border-blue-500/25" };
}

export default function Sidebar() {
  const navigate = useNavigate();
  const location = useLocation();
  const { theme, toggleTheme } = useTheme();
  const { notifications, hasAny, hasHitl, hasError, scheduleRunning, scheduleFailed, triggerRunning, triggerFailed } = useNotification();
  // Notifications (banner + toast + TTS) are handled globally in Layout.
  // Sidebar only needs pendingCount for the nav badge.
  const { pendingCount: ambientPending } = useAmbientHints();
  const suggestionsRunning = useAmbientSweepStatus();
  const [recentSessions, setRecentSessions] = useState<RecentSession[]>([]);
  const [runningCount, setRunningCount] = useState(0);
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);
  const [collapsed, setCollapsed] = useState(() => localStorage.getItem("sidebarCollapsed") === "true");
  const [confirmQuit, setConfirmQuit] = useState(false);

  const toggleCollapsed = () => {
    setCollapsed((prev) => {
      const next = !prev;
      localStorage.setItem("sidebarCollapsed", String(next));
      return next;
    });
  };

  const handleNewChat = () => {
    navigate("/chat");
  };

  const handleQuit = async () => {
    try {
      await api.shutdown().catch(() => {});
    } catch { /* backend may already be gone */ }
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke("kill_backend");
      const { exit } = await import("@tauri-apps/plugin-process");
      await exit(0);
    } catch {
      window.close();
    }
  };

  usePolling(async () => {
    try {
      const data = await api.listSessions();

      const sessions = data
        .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
        .slice(0, MAX_RECENT_SESSIONS);
      setRecentSessions(sessions);

      // Count running sessions for the Dashboard badge (check first batch only)
      const statuses = await Promise.allSettled(
        data.slice(0, 20).map((s) => api.getSessionStatus(s.id)),
      );
      const running = statuses.filter(
        (r) => r.status === "fulfilled" && r.value.running,
      ).length;
      setRunningCount(running);
    } catch (e) {
      console.warn("Failed to load sidebar data:", e);
    }
  }, RECENT_POLL_MS);

  const currentSessionId = location.pathname.startsWith("/chat/")
    ? location.pathname.split("/chat/")[1]
    : null;

  return (
    <aside className={`bg-th-sidebar-bg border-l border-th-sidebar-border flex flex-col shrink-0 transition-all duration-200 ${collapsed ? "w-[60px]" : "w-56"}`}>
      {/* Navigation */}
      <nav className="flex flex-col gap-1 px-2 py-4">
        <div className={`flex items-center mb-2 ${collapsed ? "flex-col gap-2" : "justify-between px-3"}`}>
          <NotificationCenter collapsed={collapsed} />
          <button
            onClick={toggleCollapsed}
            className="p-1 rounded text-th-text-muted hover:text-th-text-secondary hover:bg-th-surface-hover transition-all"
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {collapsed ? <PanelRightOpen size={14} /> : <PanelRightClose size={14} />}
          </button>
        </div>
        {NAV_ITEMS.map(({ to, icon: Icon, label }) => {
          // "Schedules" restores the last visited detail page when one is saved.
          const resolvedTo =
            label === "Schedules"
              ? (localStorage.getItem("schedules.lastDetailPath") ?? to)
              : to;
          return (
          <div key={to} className="flex items-center">
            <NavLink
              to={resolvedTo}
              end={false}
              className={({ isActive }) =>
                `flex-1 flex items-center ${collapsed ? "justify-center px-2" : "gap-3 px-3"} py-2.5 rounded-lg text-[13px] font-medium transition-all duration-150 ${
                  isActive
                    ? theme === "dark"
                      ? "bg-neutral-800 text-white shadow-[inset_-3px_0_0_0_rgb(115,115,115)]"
                      : "bg-neutral-900 text-white shadow-[inset_-3px_0_0_0_#111]"
                    : "text-th-text-tertiary hover:bg-th-surface-hover hover:text-th-text-primary"
                }`
              }
              title={collapsed ? label : undefined}
            >
              <span className="relative shrink-0">
                <Icon size={18} />
                {label === "Suggestions" && ambientPending > 0 && (
                  <span className="absolute -top-1 -right-1.5 min-w-[18px] h-[18px] flex items-center justify-center rounded-full border-2 border-th-sidebar-bg bg-blue-500 text-[9px] font-bold text-white px-0.5">
                    {ambientPending > 9 ? "9+" : ambientPending}
                  </span>
                )}
                {label === "Suggestions" && suggestionsRunning && collapsed && (
                  <span
                    className="absolute -top-1 -right-1.5 w-2.5 h-2.5 rounded-full border-2 border-th-sidebar-bg bg-th-text-muted animate-pulse"
                    title="Generating suggestions…"
                  />
                )}
                {label === "Dashboard" && runningCount > 0 && (
                  <span className="absolute -top-1 -right-1.5 w-2.5 h-2.5 rounded-full border-2 border-th-sidebar-bg bg-emerald-500 animate-pulse" />
                )}
                {label === "Runs" && hasAny && (
                  <span
                    className={`absolute -top-1 -right-1.5 w-2.5 h-2.5 rounded-full border-2 border-th-sidebar-bg ${
                      hasError
                        ? "bg-red-500 animate-pulse"
                        : hasHitl
                          ? "bg-amber-400 animate-pulse"
                          : "bg-emerald-500"
                    }`}
                  />
                )}
                {label === "Schedules" && (scheduleRunning || scheduleFailed) && (
                  <span
                    className={`absolute -top-1 -right-1.5 w-2.5 h-2.5 rounded-full border-2 border-th-sidebar-bg ${
                      scheduleFailed ? "bg-red-500" : "bg-emerald-500 animate-pulse"
                    }`}
                  />
                )}
                {label === "Triggers" && (triggerRunning || triggerFailed) && (
                  <span
                    className={`absolute -top-1 -right-1.5 w-2.5 h-2.5 rounded-full border-2 border-th-sidebar-bg ${
                      triggerFailed ? "bg-red-500" : "bg-emerald-500 animate-pulse"
                    }`}
                  />
                )}
              </span>
              {!collapsed && label}
              {!collapsed && label === "Suggestions" && suggestionsRunning && (
                <span className="ml-auto flex items-center gap-1 text-[9px] font-semibold text-th-text-muted bg-th-surface-hover border border-th-border rounded-full px-1.5 py-0.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-th-text-muted animate-pulse shrink-0" />
                  Running
                </span>
              )}
            </NavLink>
            {!collapsed && label === "Chat" && (
              <button
                onClick={handleNewChat}
                className="p-1.5 rounded-md text-th-text-muted hover:text-th-text-primary hover:bg-th-surface-hover transition-all duration-150"
                title={`New chat (${isMac ? "⌘N" : "Ctrl+N"})`}
              >
                <Plus size={14} />
              </button>
            )}
          </div>
          );
        })}
      </nav>

      {/* Recent sessions — hidden when collapsed */}
      {!collapsed && (
        <div className="flex flex-col flex-1 min-h-0 border-t border-th-sidebar-border/50 mt-1 pt-2">
          <p className="px-3 pb-1.5 text-[10px] uppercase tracking-wider text-th-text-muted font-semibold shrink-0">
            Recent sessions
          </p>
          <div className="overflow-y-auto px-2 pb-2 flex-1 min-h-0">
            {recentSessions.length === 0 ? (
              <p className="text-[11px] text-th-text-muted px-3 pt-1">No recent sessions</p>
            ) : (
              <div className="flex flex-col gap-0.5">
                {recentSessions.map((s) => {
                  const isActive = currentSessionId === s.id;
                  const sessionNotif = notifications[s.id];
                  return (
                    <div
                      key={s.id}
                      className={`flex items-center rounded transition-all duration-150 group ${
                        isActive
                          ? "border-r-2 border-th-text-secondary/60"
                          : "border-r-2 border-transparent hover:border-th-border-strong"
                      }`}
                    >
                      <button
                        onClick={() => navigate(`/chat/${s.id}`)}
                        className="flex-1 text-left px-3 py-1.5 min-w-0"
                      >
                        <div className="flex items-center gap-1.5">
                          {sessionNotif && (
                            <span
                              className={`w-2 h-2 rounded-full shrink-0 ${
                                sessionNotif === "error"
                                  ? "bg-red-500 animate-pulse"
                                  : sessionNotif === "hitl"
                                    ? "bg-blue-400 animate-pulse"
                                    : "bg-emerald-500"
                              }`}
                            />
                          )}
                          <p className={`text-[11px] truncate ${isActive ? "text-th-text-primary font-medium" : "text-th-text-tertiary group-hover:text-th-text-secondary"}`}>
                            {s.title || s.agent_name || "New Session"}
                          </p>
                        </div>
                        <p className="text-[10px] text-th-text-muted mt-0.5 truncate flex items-center gap-1">
                          <span className="truncate">{s.agent_name || "General"}</span>
                          {(() => {
                            const meta = triggerMeta(s.trigger_source);
                            return meta && (
                              <span className={`inline-flex items-center gap-0.5 shrink-0 px-1 py-px rounded-full text-[9px] font-medium border ${meta.color}`}>
                                {s.trigger_source === "voice" ? <Mic size={7} /> : <Zap size={7} />}
                                {meta.label}
                              </span>
                            );
                          })()}
                          <span className="shrink-0">· {formatRelativeTime(s.created_at)}</span>
                        </p>
                      </button>
                      {confirmDeleteId === s.id ? (
                        <span className="flex items-center gap-0.5 mr-1 shrink-0">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              api.closeSession(s.id).then(() => {
                                setRecentSessions((prev) => prev.filter((r) => r.id !== s.id));
                                setConfirmDeleteId(null);
                                if (isActive) {
                                  navigate("/chat");
                                }
                              }).catch((err) => console.warn("Failed to delete session:", err));
                            }}
                            className="p-1 rounded text-red-500 hover:bg-red-500/10 transition-all"
                            title="Confirm delete"
                          >
                            <Trash2 size={11} />
                          </button>
                          <button
                            onClick={(e) => { e.stopPropagation(); setConfirmDeleteId(null); }}
                            className="p-1 rounded text-th-text-muted hover:text-th-text-secondary hover:bg-th-surface-hover transition-all"
                            title="Cancel"
                          >
                            <X size={11} />
                          </button>
                        </span>
                      ) : (
                        <button
                          onClick={(e) => { e.stopPropagation(); setConfirmDeleteId(s.id); }}
                          className="p-1 mr-1 rounded opacity-0 group-hover:opacity-100 text-th-text-muted hover:text-red-500 hover:bg-red-500/10 transition-all shrink-0"
                          title="Delete session"
                        >
                          <X size={12} />
                        </button>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Spacer when collapsed */}
      {collapsed && <div className="flex-1" />}

      {/* Bottom section */}
      <div className="px-2 pb-3 pt-2 border-t border-th-sidebar-border flex flex-col gap-1">
        <button
          type="button"
          onClick={toggleTheme}
          className={`flex items-center ${collapsed ? "justify-center px-2" : "gap-3 px-3"} py-2.5 rounded-lg text-[13px] font-medium transition-all duration-150 text-th-text-tertiary hover:bg-th-surface-hover hover:text-th-text-primary`}
          title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode (blue)"}
        >
          {theme === "dark" ? <Sun size={18} className="shrink-0 text-amber-300/90" /> : <Moon size={18} className="shrink-0 text-blue-600" />}
          {!collapsed && (theme === "dark" ? "Light mode" : "Dark mode")}
        </button>
        <NavLink
          to="/settings"
          className={({ isActive }) =>
            `flex items-center ${collapsed ? "justify-center px-2" : "gap-3 px-3"} py-2.5 rounded-lg text-[13px] font-medium transition-all duration-150 ${
              isActive
                ? theme === "dark"
                  ? "bg-neutral-800 text-white shadow-[inset_-3px_0_0_0_rgb(115,115,115)]"
                  : "bg-neutral-900 text-white shadow-[inset_-3px_0_0_0_#111]"
                : "text-th-text-tertiary hover:bg-th-surface-hover hover:text-th-text-primary"
            }`
          }
          title={collapsed ? "Settings" : undefined}
        >
          <Settings size={18} className="shrink-0" />
          {!collapsed && "Settings"}
        </NavLink>

        {confirmQuit ? (
          <div className={`flex items-center ${collapsed ? "justify-center" : "gap-2 px-3"} py-2`}>
            {!collapsed && <span className="text-[12px] text-red-500">Quit app?</span>}
            <button
              onClick={handleQuit}
              className="px-2 py-1 rounded text-[11px] font-medium bg-red-500/15 text-red-400 hover:bg-red-500/25 transition-all"
              title="Confirm quit"
            >
              {collapsed ? <Power size={14} /> : "Yes"}
            </button>
            <button
              onClick={() => setConfirmQuit(false)}
              className="px-2 py-1 rounded text-[11px] font-medium text-th-text-tertiary hover:text-th-text-primary hover:bg-th-surface-hover transition-all"
              title="Cancel"
            >
              {collapsed ? <X size={14} /> : "No"}
            </button>
          </div>
        ) : (
          <button
            onClick={() => setConfirmQuit(true)}
            className={`flex items-center ${collapsed ? "justify-center px-2" : "gap-3 px-3"} py-2.5 rounded-lg text-[13px] font-medium transition-all duration-150 text-th-text-tertiary hover:bg-red-500/10 hover:text-red-400`}
            title={collapsed ? "Quit" : undefined}
          >
            <Power size={18} className="shrink-0" />
            {!collapsed && "Quit"}
          </button>
        )}
      </div>
    </aside>
  );
}
