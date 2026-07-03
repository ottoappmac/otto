/**
 * Polls the ambient hints endpoint and exposes actions for each hint.
 *
 * Poll interval: 30 seconds (mirrors the schedule-status poll).
 * On each successful fetch the hook calls `onNewHints` ONLY when there are
 * pending hints whose IDs have not been seen in a previous notification.
 * Seen IDs are persisted in localStorage so the notification is not
 * re-shown after a page refresh.
 *
 * When new hints are discovered, a `ambient_last_sweep` summary is also
 * written to localStorage so the AmbientInbox can show a "last sweep" line.
 *
 * The `onNewHints` callback is captured in a ref so it never needs to be a
 * stable reference — callers can safely pass inline arrow functions without
 * causing the polling loop to restart on every render.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./useApi";
import type { AmbientHint } from "../types";

const POLL_INTERVAL_MS = 30_000;
export const NOTIFIED_IDS_KEY = "ambient_notified_ids";

export interface SweepResult {
  hints_added: number;
  skipped: string | null;
}

export interface UseAmbientHintsReturn {
  hints: AmbientHint[];
  pendingCount: number;
  quietHours: boolean;
  loading: boolean;
  refresh: () => Promise<void>;
  accept: (id: string, mode?: "chat" | "run" | "apply", agentName?: string) => Promise<{ session_id: string | null }>;
  dismiss: (id: string) => Promise<void>;
  snooze: (id: string, hours?: number) => Promise<void>;
  triggerSweep: () => Promise<SweepResult>;
  /** Mark all current pending hints as "seen" so they don't re-trigger notifications. */
  markSeen: () => void;
}

// ---------------------------------------------------------------------------
// localStorage helpers
// ---------------------------------------------------------------------------

function getNotifiedIds(): Set<string> {
  try {
    const stored = localStorage.getItem(NOTIFIED_IDS_KEY);
    return new Set(stored ? (JSON.parse(stored) as string[]) : []);
  } catch {
    return new Set();
  }
}

function saveNotifiedIds(ids: Set<string>) {
  localStorage.setItem(NOTIFIED_IDS_KEY, JSON.stringify([...ids]));
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useAmbientHints(
  onNewHints?: (count: number, hints: AmbientHint[]) => void,
): UseAmbientHintsReturn {
  const [hints, setHints] = useState<AmbientHint[]>([]);
  const [quietHours, setQuietHours] = useState(false);
  const [loading] = useState(false);
  const pollingRef = useRef(false);
  const hintsRef = useRef<AmbientHint[]>([]);

  const onNewHintsRef = useRef(onNewHints);
  useEffect(() => { onNewHintsRef.current = onNewHints; });

  // Stable refresh — no dependencies that change on re-render.
  const refresh = useCallback(async () => {
    if (pollingRef.current) return;
    pollingRef.current = true;
    try {
      const data = await api.ambientHints();
      hintsRef.current = data.hints;
      setHints(data.hints);
      setQuietHours(data.quiet_hours);

      const pending = data.hints.filter(
        (h) => h.status === "pending" || h.status === "shown",
      );

      // Only notify about hints we haven't fired a notification for yet.
      const notifiedIds = getNotifiedIds();
      const unnotified = pending.filter((h) => !notifiedIds.has(h.id));

      if (unnotified.length > 0) {
        unnotified.forEach((h) => notifiedIds.add(h.id));
        saveNotifiedIds(notifiedIds);
        onNewHintsRef.current?.(unnotified.length, unnotified);
      }
    } catch {
      // Backend not ready or feature not enabled — no-op.
    } finally {
      pollingRef.current = false;
    }
  }, []); // intentionally empty — stability is the goal

  // Initial fetch + single polling loop.
  useEffect(() => {
    void refresh();
    const timer = setInterval(() => void refresh(), POLL_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [refresh]);

  const accept = useCallback(
    async (id: string, mode: "chat" | "run" | "apply" = "chat", agentName?: string) => {
      const result = await api.ambientAccept(id, mode, agentName);
      const updated = (prev: AmbientHint[]) =>
        prev.map((h) => (h.id === id ? { ...h, status: "accepted" as const } : h));
      hintsRef.current = updated(hintsRef.current);
      setHints(updated);
      return { session_id: result.session_id ?? null };
    },
    [],
  );

  const dismiss = useCallback(async (id: string) => {
    await api.ambientDismiss(id);
    const updated = (prev: AmbientHint[]) =>
      prev.map((h) => (h.id === id ? { ...h, status: "dismissed" as const } : h));
    hintsRef.current = updated(hintsRef.current);
    setHints(updated);
  }, []);

  const snooze = useCallback(async (id: string, hours = 4) => {
    await api.ambientSnooze(id, hours);
    const updated = (prev: AmbientHint[]) =>
      prev.map((h) => (h.id === id ? { ...h, status: "snoozed" as const } : h));
    hintsRef.current = updated(hintsRef.current);
    setHints(updated);
  }, []);

  const triggerSweep = useCallback(async (): Promise<SweepResult> => {
    const result = await api.ambientRun();
    const data = await api.ambientHints();
    hintsRef.current = data.hints;
    setHints(data.hints);
    setQuietHours(data.quiet_hours);

    const pending = data.hints.filter(
      (h) => h.status === "pending" || h.status === "shown",
    );
    const notifiedIds = getNotifiedIds();
    const unnotified = pending.filter((h) => !notifiedIds.has(h.id));

    if (unnotified.length > 0) {
      unnotified.forEach((h) => notifiedIds.add(h.id));
      saveNotifiedIds(notifiedIds);
      onNewHintsRef.current?.(unnotified.length, unnotified);
    }

    return { hints_added: result.hints_added ?? 0, skipped: result.skipped ?? null };
  }, []);

  const markSeen = useCallback(() => {
    const pending = hintsRef.current.filter(
      (h) => h.status === "pending" || h.status === "shown",
    );
    if (pending.length === 0) return;
    const notifiedIds = getNotifiedIds();
    pending.forEach((h) => notifiedIds.add(h.id));
    saveNotifiedIds(notifiedIds);
  }, []);

  const pendingCount = hints.filter(
    (h) => h.status === "pending" || h.status === "shown",
  ).length;

  // Keep tray menu badge in sync.
  const prevTrayCountRef = useRef(-1);
  useEffect(() => {
    if (prevTrayCountRef.current === pendingCount) return;
    prevTrayCountRef.current = pendingCount;
    import("@tauri-apps/api/core")
      .then(({ invoke }) => invoke("set_ambient_count", { count: pendingCount }))
      .catch(() => {}); // no-op in non-Tauri environments (e.g. web dev)
  }, [pendingCount]);

  return { hints, pendingCount, quietHours, loading, refresh, accept, dismiss, snooze, triggerSweep, markSeen };
}

/**
 * Fetches pending eval-triggered suggestions and returns a stable map of
 * `target_id -> AmbientHint`. Polls every 30 s alongside the main inbox.
 * Useful for surfacing a "View suggestion" indicator on schedule/trigger cards.
 */
export function useEvalSuggestionsByTarget(): Map<string, AmbientHint> {
  const [byTarget, setByTarget] = useState<Map<string, AmbientHint>>(new Map());

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        const { hints } = await api.ambientHints();
        if (cancelled) return;
        const map = new Map<string, AmbientHint>();
        for (const h of hints) {
          if (
            h.origin === "evaluation" &&
            h.target_id &&
            (h.status === "pending" || h.status === "shown")
          ) {
            // Keep the most-recent one per target.
            const existing = map.get(h.target_id);
            if (!existing || h.created_at > existing.created_at) {
              map.set(h.target_id, h);
            }
          }
        }
        setByTarget(map);
      } catch {
        // Backend unavailable — no-op.
      }
    };

    void load();
    const timer = setInterval(() => void load(), POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  return byTarget;
}
