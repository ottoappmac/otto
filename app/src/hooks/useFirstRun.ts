import { useCallback, useEffect, useState } from "react";
import { api } from "./useApi";

export interface FirstRunState {
  /** Whether the wizard should be presented now. */
  show: boolean;
  /** Backend's last reported step (used to resume mid-wizard). */
  currentStep: string;
  /** Step ids the user has already touched. */
  completedSteps: string[];
  /** True once we've heard back from the backend. */
  loaded: boolean;
  /** Hide the wizard locally (used by the parent after completion/skip).
   *  Backend still owns the persistent flag; this is a render gate so the
   *  wizard tears down immediately on Finish/Skip without waiting for a
   *  refetch. */
  hide: () => void;
  /** Force a re-check from the backend (e.g. after user picked "Re-run
   *  setup wizard" from Settings). */
  refresh: () => Promise<void>;
}

/** Drive the first-run setup wizard gate.
 *
 * The hook is intentionally cheap — one ``GET /api/settings/first-run``
 * on mount and that's it. Re-renders only when the state actually
 * changes. The wizard's screens save through the existing settings API,
 * so this hook does not poll.
 */
export function useFirstRun(): FirstRunState {
  const [show, setShow] = useState(false);
  const [currentStep, setCurrentStep] = useState("welcome");
  const [completedSteps, setCompletedSteps] = useState<string[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [forced, setForced] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const r = await api.isFirstRun();
      setShow(r.first_run);
      setCurrentStep(r.current_step || "welcome");
      setCompletedSteps(r.completed_steps || []);
    } catch (e) {
      // If the backend isn't reachable yet, fall back to "don't show"
      // — the App-level backend gate will retry separately. Better to
      // show no wizard than to flash one on every boot when the backend
      // is briefly unreachable.
      console.warn("first-run probe failed:", e);
      setShow(false);
    } finally {
      setLoaded(true);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const hide = useCallback(() => setForced(true), []);

  return {
    show: show && !forced,
    currentStep,
    completedSteps,
    loaded,
    hide,
    refresh,
  };
}
