import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { CheckCircle2, Loader2, ThumbsDown, ThumbsUp, Workflow, X } from "lucide-react";
import { api } from "../../hooks/useApi";

type State = Awaited<ReturnType<typeof api.getProcedureSessionFeedback>>;
type Phase = "hidden" | "ask" | "sending" | "saved" | "not_saved" | "declined" | "error";

// The run is saved in the background after the turn's "done" event, so the first check can come too early.
const RETRY_DELAYS_MS = [1500, 4000];
const SETTINGS_LINK = "/settings?tab=Agent+Memory&sub=Procedures";

function dismissKey(sessionId: string, ids: string[]): string {
  return `otto:procmem:dismissed:${sessionId}:${ids.join(",")}`;
}

function readDismissed(key: string): boolean {
  try { return sessionStorage.getItem(key) === "1"; } catch { return false; }
}

function writeDismissed(key: string): void {
  try { sessionStorage.setItem(key, "1"); } catch { /* private mode: dismiss for this render only */ }
}

/**
 * After a turn in which the macOS desktop agent ran (and procedural memory is on), asks whether it worked.
 * Thumbs up saves what the agent did to procedural memory; thumbs down excludes the run so it is never used.
 */
export function ProcedureFeedbackCard({ sessionId, isStreaming }: { sessionId: string | null; isStreaming: boolean }) {
  const [state, setState] = useState<State | null>(null);
  const [phase, setPhase] = useState<Phase>("hidden");
  const [note, setNote] = useState<string>("");
  const timers = useRef<ReturnType<typeof setTimeout>[]>([]);

  const pendingIds = (s: State | null) =>
    (s?.episodes ?? []).filter((e) => e.user_label === null && e.status === "pending").map((e) => e.id);

  const check = useCallback(async (sid: string): Promise<boolean> => {
    try {
      const s = await api.getProcedureSessionFeedback(sid);
      setState(s);
      const ids = pendingIds(s);
      if (s.ask && ids.length && !readDismissed(dismissKey(sid, ids))) {
        setPhase("ask");
        return true;
      }
      setPhase((p) => (p === "saved" || p === "not_saved" || p === "declined" ? p : "hidden"));
      return s.ask || !s.enabled;
    } catch {
      setPhase((p) => (p === "ask" ? p : "hidden"));
      return true; // the endpoint failed: do not keep retrying
    }
  }, []);

  useEffect(() => {
    timers.current.forEach(clearTimeout);
    timers.current = [];
    if (!sessionId || isStreaming) {
      if (isStreaming) setPhase((p) => (p === "ask" ? "hidden" : p));
      return;
    }
    setPhase("hidden");
    let cancelled = false;
    void check(sessionId).then((done) => {
      if (done || cancelled) return;
      for (const ms of RETRY_DELAYS_MS) {
        timers.current.push(setTimeout(() => { if (!cancelled) void check(sessionId); }, ms));
      }
    });
    return () => {
      cancelled = true;
      timers.current.forEach(clearTimeout);
      timers.current = [];
    };
  }, [sessionId, isStreaming, check]);

  const answer = async (worked: boolean) => {
    if (!sessionId) return;
    setPhase("sending");
    try {
      const res = await api.sendProcedureSessionFeedback(sessionId, worked);
      if (!worked) {
        setPhase("not_saved");
        setNote("Got it. This run won't be saved to memory.");
      } else if (res.saved > 0) {
        setPhase("saved");
        setNote("Saved. The macOS desktop agent can reuse these steps on similar tasks.");
      } else {
        setPhase("saved");
        setNote("Marked as worked. It will be added to memory once the embedding model is available.");
      }
    } catch (e: any) {
      setPhase("error");
      setNote(e?.message?.includes("off") ? "Procedural memory is off." : "Couldn't save your answer. Try again.");
    }
  };

  const dismiss = () => {
    if (sessionId) writeDismissed(dismissKey(sessionId, pendingIds(state)));
    setPhase("declined");
  };

  if (phase === "hidden" || phase === "declined") return null;

  const eps = (state?.episodes ?? []).filter((e) => e.user_label === null && e.status === "pending");
  const latest = eps[eps.length - 1];
  const apps = Array.from(new Set(eps.flatMap((e) => e.apps ?? []))).slice(0, 3);

  if (phase === "saved" || phase === "not_saved") {
    return (
      <div className="mt-3 ml-10 max-w-xl flex items-center gap-2 text-xs text-th-text-tertiary" role="status" aria-live="polite">
        <CheckCircle2 size={13} className={phase === "saved" ? "text-emerald-400" : "text-th-text-muted"} />
        <span>{note}</span>
        <Link to={SETTINGS_LINK} className="text-blue-400 hover:underline whitespace-nowrap">View procedures</Link>
      </div>
    );
  }

  return (
    <section
      aria-label="Did the macOS desktop agent's run work?"
      className="mt-3 ml-10 max-w-xl rounded-xl border border-th-border bg-th-card-bg px-4 py-3"
    >
      <div className="flex items-start gap-3">
        <Workflow size={16} className="text-blue-400 mt-0.5 shrink-0" />
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium text-th-text-primary">Did the macOS desktop agent get this right?</p>
          <p className="text-[12px] text-th-text-tertiary mt-0.5 leading-relaxed">
            If it worked, Otto saves the steps to procedural memory and reuses them on similar tasks
            {apps.length ? ` in ${apps.join(", ")}` : ""}.
          </p>
          {latest?.task_text && (
            <p className="text-2xs text-th-text-muted mt-1 line-clamp-1" title={latest.task_text}>
              Task: {latest.task_text}
            </p>
          )}
          <div className="flex flex-wrap items-center gap-2 mt-3">
            <button
              type="button"
              onClick={() => void answer(true)}
              disabled={phase === "sending"}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border bg-emerald-500/5 text-emerald-400 border-emerald-500/20 hover:bg-emerald-500/15 transition-colors disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400"
            >
              {phase === "sending" ? <Loader2 size={12} className="animate-spin" /> : <ThumbsUp size={12} />} Worked
            </button>
            <button
              type="button"
              onClick={() => void answer(false)}
              disabled={phase === "sending"}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border bg-th-inset-bg text-th-text-secondary border-th-border hover:text-th-text-primary transition-colors disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400"
            >
              <ThumbsDown size={12} /> Didn't work
            </button>
            {phase === "error" && <span className="text-2xs text-red-400" role="alert">{note}</span>}
          </div>
        </div>
        <button
          type="button"
          onClick={dismiss}
          aria-label="Not now"
          title="Not now (you can still review it in Settings → Agent Memory → Procedures)"
          className="p-1 rounded text-th-text-muted hover:text-th-text-primary shrink-0"
        >
          <X size={14} />
        </button>
      </div>
    </section>
  );
}
