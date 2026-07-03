import {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  type ReactNode,
} from "react";
import { api } from "../hooks/useApi";
import { usePolling } from "../hooks/usePolling";
import { setBadge, type BadgeLevel } from "../utils/appBadge";
import { nativeNotify } from "../utils/nativeNotify";

export type NotificationType = "done" | "hitl" | "error";

/**
 * A persistent, deep-linked entry in the Notification Center inbox. Stored in
 * localStorage so returning to the app always answers "what was that badge for".
 */
export interface NotificationItem {
  /** Dedupe key, e.g. `schedule:daily:2026-...` or `session:<id>:hitl`. */
  id: string;
  kind: NotificationType;
  category: "schedule" | "trigger" | "session" | "ambient" | "memory";
  /** e.g. `Schedule "daily-report" completed`. */
  title: string;
  body?: string;
  /** e.g. `/schedules/daily-report/runs`. */
  deepLink: string;
  createdAt: number;
  read: boolean;
}

/** Fields callers supply to `pushEvent`; the rest are filled in automatically. */
type PushEventInput = Omit<NotificationItem, "createdAt" | "read">;

type NotificationMap = Record<string, NotificationType>;

interface NotifyOpts {
  /** Custom browser-notification title/body (overrides the generic ones). */
  title?: string;
  body?: string;
}

export interface CompletedJob {
  /** Unique key: `${type}:${id}:${last_run}` */
  key: string;
  id: string;
  status: "success" | "error" | "cancelled";
  type: "schedule" | "trigger";
}

/** A session paused awaiting human feedback (HITL). */
export interface HitlSession {
  id: string;
  title?: string;
}

interface NotificationContextValue {
  notifications: NotificationMap;
  notify: (type: NotificationType, sessionId: string, opts?: NotifyOpts) => void;
  clearSession: (sessionId: string) => void;
  watchSession: (sessionId: string) => void;
  unwatchSession: (sessionId: string) => void;
  hasAny: boolean;
  hasHitl: boolean;
  /** Sessions currently paused awaiting human feedback, with titles when known. */
  hitlSessions: HitlSession[];
  hasError: boolean;
  scheduleRunning: boolean;
  scheduleFailed: boolean;
  runningScheduleIds: string[];
  clearScheduleNotifications: () => void;
  triggerRunning: boolean;
  triggerFailed: boolean;
  runningTriggerIds: string[];
  clearTriggerNotifications: () => void;
  memoryConsolidating: boolean;
  completedJobs: CompletedJob[];
  dismissCompletedJob: (key: string) => void;
  // ── Notification Center (persistent inbox) ──────────────────────────────
  items: NotificationItem[];
  unreadCount: number;
  markRead: (id: string) => void;
  markAllRead: () => void;
  clear: () => void;
  removeItem: (id: string) => void;
}

const NotificationContext = createContext<NotificationContextValue>({
  notifications: {},
  notify: () => {},
  clearSession: () => {},
  watchSession: () => {},
  unwatchSession: () => {},
  hasAny: false,
  hasHitl: false,
  hitlSessions: [],
  hasError: false,
  scheduleRunning: false,
  scheduleFailed: false,
  runningScheduleIds: [],
  clearScheduleNotifications: () => {},
  triggerRunning: false,
  triggerFailed: false,
  runningTriggerIds: [],
  clearTriggerNotifications: () => {},
  memoryConsolidating: false,
  completedJobs: [],
  dismissCompletedJob: () => {},
  items: [],
  unreadCount: 0,
  markRead: () => {},
  markAllRead: () => {},
  clear: () => {},
  removeItem: () => {},
});

const BASE_TITLE = "OTTO";
const POLL_INTERVAL_MS = 3000;
const MAX_WATCHED = 20;
const MAX_NOTIFICATIONS = 50;
const MAX_ITEMS = 100;
const STORAGE_KEY = "ottoNotifications";

// Background (non-interactive) sessions we auto-watch so their completion and
// HITL pauses surface even when the user isn't on that chat. Schedule/trigger
// runs are included only for HITL (their completion/error is already covered by
// the schedule/trigger status polls below — see the watch handler).
const BG_WATCH_SOURCES = new Set([
  "schedule",
  "trigger",
  "ambient",
  "spawn",
  "claude-hook",
  "oc-watcher-new",
  "oc-watcher-activity",
]);
const MAX_SEEN_BG = 300;
const BG_SCAN_LIMIT = 40;

function isEvalSource(source: string | undefined): boolean {
  return source === "claude-hook" || (source?.startsWith("oc-watcher") ?? false);
}

function bgNotifLabel(type: NotificationType, source: string | undefined): NotifyOpts {
  const evalSession = isEvalSource(source);
  if (type === "hitl") {
    return {
      title: "Approval Required",
      body: evalSession
        ? "A background evaluation needs your approval."
        : "A background task needs your approval.",
    };
  }
  if (type === "error") {
    return {
      title: evalSession ? "Evaluation Error" : "Background Task Error",
      body: evalSession
        ? "A background evaluation encountered an error."
        : "A background task encountered an error.",
    };
  }
  return {
    title: evalSession ? "Evaluation Complete" : "Background Task Complete",
    body: evalSession
      ? "A background evaluation has finished."
      : "A background task has finished.",
  };
}

export function NotificationProvider({ children }: { children: ReactNode }) {
  const [notifications, setNotifications] = useState<NotificationMap>({});
  const [scheduleRunning, setScheduleRunning] = useState(false);
  const [scheduleFailed, setScheduleFailed] = useState(false);
  const [runningScheduleIds, setRunningScheduleIds] = useState<string[]>([]);
  const [triggerRunning, setTriggerRunning] = useState(false);
  const [triggerFailed, setTriggerFailed] = useState(false);
  const [runningTriggerIds, setRunningTriggerIds] = useState<string[]>([]);
  const [memoryConsolidating, setMemoryConsolidating] = useState(false);
  const [completedJobs, setCompletedJobs] = useState<CompletedJob[]>([]);
  // sessionId -> title, used to label HITL "needs feedback" chips/cards.
  const [sessionTitles, setSessionTitles] = useState<Record<string, string>>({});

  // ── Persistent Notification Center inbox ──────────────────────────────────
  const [items, setItems] = useState<NotificationItem[]>(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? (parsed as NotificationItem[]) : [];
    } catch {
      return [];
    }
  });
  // Tracks every id ever pushed so we never re-add (and re-alert) a known event.
  // Seeded from hydrated items so a restart doesn't replay the persisted inbox.
  const seenItemIdsRef = useRef<Set<string>>(new Set(items.map((i) => i.id)));
  // Latest session titles, read inside the stable `notify`/`pushEvent` closures.
  const sessionTitlesRef = useRef<Record<string, string>>({});
  useEffect(() => {
    sessionTitlesRef.current = sessionTitles;
  }, [sessionTitles]);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
    } catch {
      // Storage full / unavailable — the in-memory list still works.
    }
  }, [items]);

  // Single entry point for all notification events: dedupes by id, prepends to
  // the inbox, and (only when the window is unfocused) fires a native bubble.
  // Auto-read only when focused AND already on the relevant page — so the bell
  // badge (and Dock badge) still appear when a background event fires while the
  // user is on a different page.
  const pushEvent = useCallback((input: PushEventInput) => {
    if (seenItemIdsRef.current.has(input.id)) return;
    seenItemIdsRef.current.add(input.id);

    const focused = typeof document !== "undefined" && document.hasFocus();
    const deepLinkBase = input.deepLink.split("?")[0];
    const onRelevantPage =
      typeof window !== "undefined" &&
      window.location.pathname.startsWith(deepLinkBase);
    const item: NotificationItem = {
      ...input,
      createdAt: Date.now(),
      read: focused && onRelevantPage,
    };

    setItems((prev) => [item, ...prev].slice(0, MAX_ITEMS));

    if (!focused) {
      void nativeNotify({ title: item.title, body: item.body, deepLink: item.deepLink });
    }
  }, []);

  const markRead = useCallback((id: string) => {
    setItems((prev) =>
      prev.some((i) => i.id === id && !i.read)
        ? prev.map((i) => (i.id === id ? { ...i, read: true } : i))
        : prev,
    );
  }, []);

  const markAllRead = useCallback(() => {
    setItems((prev) =>
      prev.some((i) => !i.read) ? prev.map((i) => ({ ...i, read: true })) : prev,
    );
  }, []);

  const clear = useCallback(() => {
    setItems((prev) => (prev.length ? [] : prev));
  }, []);

  const removeItem = useCallback((id: string) => {
    setItems((prev) => prev.filter((i) => i.id !== id));
  }, []);

  const watchedRef = useRef<Set<string>>(new Set());
  const pollingRef = useRef(false);
  const notifiedSchedulesRef = useRef<Set<string>>(new Set());
  const notifiedTriggersRef = useRef<Set<string>>(new Set());
  // Background sessions we've already decided about (watched or skipped) so we
  // don't re-notify for runs that finished before we ever saw them.
  const seenBgRef = useRef<Set<string>>(new Set());
  // sessionId -> trigger_source, for tailoring the completion notification.
  const bgSourceRef = useRef<Map<string, string>>(new Map());


  const notify = useCallback((type: NotificationType, sessionId: string, opts?: NotifyOpts) => {
    if (!sessionId) return;
    setNotifications((prev) => {
      if (prev[sessionId] === type) return prev;
      const next = { ...prev, [sessionId]: type };
      const keys = Object.keys(next);
      if (keys.length > MAX_NOTIFICATIONS) {
        delete next[keys[0]];
      }
      return next;
    });

    const label = sessionTitlesRef.current[sessionId]
      ? `"${sessionTitlesRef.current[sessionId]}"`
      : "A session";
    const title =
      opts?.title ??
      (type === "hitl"
        ? `${label} needs your input`
        : type === "error"
          ? `${label} encountered an error`
          : `${label} completed`);
    const body =
      opts?.body ??
      (type === "hitl"
        ? "An agent action needs your approval."
        : type === "error"
          ? "The agent encountered an error."
          : "The agent has finished its task.");

    pushEvent({
      id: `session:${sessionId}:${type}`,
      kind: type,
      category: "session",
      title,
      body,
      deepLink: `/chat/${encodeURIComponent(sessionId)}`,
    });
  }, [pushEvent]);

  const clearSession = useCallback((sessionId: string) => {
    setNotifications((prev) => {
      if (!(sessionId in prev)) return prev;
      const next = { ...prev };
      delete next[sessionId];
      return next;
    });
  }, []);

  const watchSession = useCallback((sessionId: string) => {
    if (!sessionId) return;
    if (watchedRef.current.size >= MAX_WATCHED) {
      const oldest = watchedRef.current.values().next().value;
      if (oldest) watchedRef.current.delete(oldest);
    }
    watchedRef.current.add(sessionId);
  }, []);

  const unwatchSession = useCallback((sessionId: string) => {
    watchedRef.current.delete(sessionId);
  }, []);

  const clearScheduleNotifications = useCallback(() => {
    setScheduleFailed(false);
  }, []);

  const clearTriggerNotifications = useCallback(() => {
    setTriggerFailed(false);
  }, []);

  const dismissCompletedJob = useCallback((key: string) => {
    setCompletedJobs((prev) => prev.filter((j) => j.key !== key));
  }, []);

  usePolling(async () => {
    if (pollingRef.current) return;
    const ids = Array.from(watchedRef.current);
    if (ids.length === 0) return;

    pollingRef.current = true;
    try {
      await Promise.allSettled(ids.map(async (sid) => {
        try {
          const status = await api.getSessionStatus(sid);
          if (status.running) return;

          watchedRef.current.delete(sid);

          const msgs = await api.getSessionMessages(sid);
          const lastType = msgs.length > 0
            ? (msgs[msgs.length - 1].type as string)
            : "done";

          console.debug("[notify] session", sid, "lastType:", lastType);

          const mapped: NotificationType =
            lastType === "hitl_request" || lastType === "ask_user"
              ? "hitl"
              : lastType === "error"
                ? "error"
                : "done";

          const source = bgSourceRef.current.get(sid);
          if (source) {
            bgSourceRef.current.delete(sid);
            // Schedule/trigger completion + errors are already reported by their
            // dedicated status polls — only surface the HITL gap here.
            if ((source === "schedule" || source === "trigger") && mapped !== "hitl") {
              return;
            }
            notify(mapped, sid, bgNotifLabel(mapped, source));
          } else {
            notify(mapped, sid);
          }
        } catch {
          watchedRef.current.delete(sid);
          bgSourceRef.current.delete(sid);
        }
      }));
    } finally {
      pollingRef.current = false;
    }
  }, POLL_INTERVAL_MS);

  usePolling(async () => {
    try {
      const status = await api.getScheduleStatus();
      setScheduleRunning(status.running.length > 0);
      setRunningScheduleIds(status.running);

      const hasFailure = status.recently_completed.some((s) => s.status === "error");
      setScheduleFailed(hasFailure);

      const newScheduleCompletions: CompletedJob[] = [];
      for (const entry of status.recently_completed) {
        const key = `schedule:${entry.id}:${entry.last_run}`;
        if (notifiedSchedulesRef.current.has(key)) continue;
        notifiedSchedulesRef.current.add(key);

        newScheduleCompletions.push({
          key,
          id: entry.id,
          status: entry.status as CompletedJob["status"],
          type: "schedule",
        });

        const failed = entry.status === "error";
        pushEvent({
          id: key,
          kind: failed ? "error" : "done",
          category: "schedule",
          title: failed
            ? `Schedule "${entry.id}" failed`
            : `Schedule "${entry.id}" completed`,
          body: failed
            ? `Schedule "${entry.id}" encountered an error.`
            : `Schedule "${entry.id}" finished successfully.`,
          deepLink: `/schedules/${encodeURIComponent(entry.id)}/runs`,
        });
      }
      if (newScheduleCompletions.length > 0) {
        setCompletedJobs((prev) => [...prev, ...newScheduleCompletions].slice(-10));
      }

      if (notifiedSchedulesRef.current.size > 100) {
        const entries = Array.from(notifiedSchedulesRef.current);
        notifiedSchedulesRef.current = new Set(entries.slice(-50));
      }
    } catch {
      // Backend not ready or schedules endpoint not available
    }
  }, 10000);

  // ── Trigger run status (mirrors schedules) ────────────────────────────────
  usePolling(async () => {
    try {
      const status = await api.getTriggerStatus();
      setTriggerRunning(status.running.length > 0);
      setRunningTriggerIds(status.running);

      const hasFailure = status.recently_completed.some((s) => s.status === "error");
      setTriggerFailed(hasFailure);

      const newTriggerCompletions: CompletedJob[] = [];
      for (const entry of status.recently_completed) {
        const key = `trigger:${entry.id}:${entry.last_run}`;
        if (notifiedTriggersRef.current.has(key)) continue;
        notifiedTriggersRef.current.add(key);

        newTriggerCompletions.push({
          key,
          id: entry.id,
          status: entry.status as CompletedJob["status"],
          type: "trigger",
        });

        const failed = entry.status === "error";
        pushEvent({
          id: key,
          kind: failed ? "error" : "done",
          category: "trigger",
          title: failed
            ? `Trigger "${entry.id}" failed`
            : `Trigger "${entry.id}" completed`,
          body: failed
            ? `Trigger "${entry.id}" encountered an error.`
            : `Trigger "${entry.id}" finished successfully.`,
          deepLink: "/triggers",
        });
      }
      if (newTriggerCompletions.length > 0) {
        setCompletedJobs((prev) => [...prev, ...newTriggerCompletions].slice(-10));
      }

      if (notifiedTriggersRef.current.size > 100) {
        const entries = Array.from(notifiedTriggersRef.current);
        notifiedTriggersRef.current = new Set(entries.slice(-50));
      }
    } catch {
      // Backend not ready or triggers endpoint not available
    }
  }, 10000);

  // ── Auto-watch background sessions (ambient / eval / spawn + HITL for ──────
  //    schedule & trigger). The 3s watch poll above then notifies on
  //    completion / HITL / error. We only watch sessions found *running* so we
  //    never re-notify for runs that finished before this client connected.
  usePolling(async () => {
    try {
      const sessions = await api.listSessions();

      // Cache titles so HITL chips/cards can show a human-readable label.
      setSessionTitles((prev) => {
        let changed = false;
        const next = { ...prev };
        for (const s of sessions) {
          if (s.title && next[s.id] !== s.title) {
            next[s.id] = s.title;
            changed = true;
          }
        }
        return changed ? next : prev;
      });

      const bg = sessions
        .filter((s) => s.trigger_source && BG_WATCH_SOURCES.has(s.trigger_source))
        .sort(
          (a, b) =>
            new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
        );

      // Only probe the most recent background sessions; older unseen ones are
      // marked handled so we never notify retroactively for historical runs.
      for (const s of bg.slice(0, BG_SCAN_LIMIT)) {
        if (seenBgRef.current.has(s.id)) continue;

        let running = false;
        try {
          running = (await api.getSessionStatus(s.id)).running;
        } catch {
          seenBgRef.current.add(s.id);
          continue;
        }

        if (running) {
          bgSourceRef.current.set(s.id, s.trigger_source as string);
          watchSession(s.id);
        }
        seenBgRef.current.add(s.id);
      }

      for (const s of bg.slice(BG_SCAN_LIMIT)) {
        seenBgRef.current.add(s.id);
      }

      if (seenBgRef.current.size > MAX_SEEN_BG) {
        const entries = Array.from(seenBgRef.current);
        seenBgRef.current = new Set(entries.slice(-Math.floor(MAX_SEEN_BG / 2)));
      }
    } catch {
      // Backend not ready or sessions endpoint not available
    }
  }, 9000);

  // ── Memory consolidation (passive indicator only — no badge/OS alert) ──────
  usePolling(async () => {
    try {
      const status = await api.getMemoryStatus();
      setMemoryConsolidating(status.state === "running");
    } catch {
      setMemoryConsolidating(false);
    }
  }, 10000);

  // ── App badge driven by unread inbox count + top severity ─────────────────
  // (Badge clears as items are read, so it survives returning to the app.)
  const unreadCount = useMemo(() => items.reduce((n, i) => n + (i.read ? 0 : 1), 0), [items]);
  const topUnreadLevel = useMemo<BadgeLevel | null>(() => {
    let level: BadgeLevel | null = null;
    for (const i of items) {
      if (i.read) continue;
      if (i.kind === "error") return "error";
      if (i.kind === "hitl") level = "hitl";
      else if (level === null) level = "done";
    }
    return level;
  }, [items]);

  useEffect(() => {
    void setBadge(unreadCount, topUnreadLevel);
  }, [unreadCount, topUnreadLevel]);

  const { hasAny, hasHitl, hasError } = useMemo(() => {
    const vals = Object.values(notifications);
    return {
      hasAny: vals.length > 0,
      hasHitl: vals.includes("hitl"),
      hasError: vals.includes("error"),
    };
  }, [notifications]);

  const hitlSessions = useMemo<HitlSession[]>(
    () =>
      Object.entries(notifications)
        .filter(([, type]) => type === "hitl")
        .map(([id]) => ({ id, title: sessionTitles[id] })),
    [notifications, sessionTitles],
  );

  useEffect(() => {
    if (hasError) {
      document.title = `❌ Error — ${BASE_TITLE}`;
    } else if (hasHitl) {
      document.title = `⏳ Approval needed — ${BASE_TITLE}`;
    } else if (hasAny) {
      document.title = `✅ Complete — ${BASE_TITLE}`;
    } else {
      document.title = BASE_TITLE;
    }
  }, [hasAny, hasHitl, hasError]);

  const value = useMemo(() => ({
    notifications, notify, clearSession,
    watchSession, unwatchSession, hasAny, hasHitl, hitlSessions, hasError,
    scheduleRunning, scheduleFailed, runningScheduleIds, clearScheduleNotifications,
    triggerRunning, triggerFailed, runningTriggerIds, clearTriggerNotifications,
    memoryConsolidating,
    completedJobs, dismissCompletedJob,
    items, unreadCount, markRead, markAllRead, clear, removeItem,
  }), [notifications, notify, clearSession, watchSession, unwatchSession, hasAny, hasHitl, hitlSessions, hasError, scheduleRunning, scheduleFailed, runningScheduleIds, clearScheduleNotifications, triggerRunning, triggerFailed, runningTriggerIds, clearTriggerNotifications, memoryConsolidating, completedJobs, dismissCompletedJob, items, unreadCount, markRead, markAllRead, clear, removeItem]);

  return (
    <NotificationContext.Provider value={value}>
      {children}
    </NotificationContext.Provider>
  );
}

export function useNotification() {
  return useContext(NotificationContext);
}
