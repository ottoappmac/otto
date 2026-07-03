/**
 * AmbientToast — in-app slide-up notification for new ambient suggestions.
 *
 * Appears in the bottom-right corner whenever new hints arrive, regardless
 * of whether the window is focused.  Auto-dismisses after 8 seconds with a
 * draining progress bar.  Includes a "View" button that navigates to /ambient
 * and a close button to dismiss immediately.
 */
import { useEffect, useRef, useState } from "react";
import { Sparkles, X, ChevronRight } from "lucide-react";

const DURATION_MS = 8000;

interface Props {
  count: number;
  onView: () => void;
  onDismiss: () => void;
}

export default function AmbientToast({ count, onView, onDismiss }: Props) {
  const [visible, setVisible] = useState(false);
  const [progress, setProgress] = useState(100);
  const startRef = useRef<number>(0);
  const rafRef = useRef<number>(0);

  // Slide in on mount.
  useEffect(() => {
    const tid = setTimeout(() => setVisible(true), 10);
    return () => clearTimeout(tid);
  }, []);

  // Drain the progress bar over DURATION_MS then auto-dismiss.
  useEffect(() => {
    startRef.current = Date.now();

    const tick = () => {
      const elapsed = Date.now() - startRef.current;
      const remaining = Math.max(0, 100 - (elapsed / DURATION_MS) * 100);
      setProgress(remaining);
      if (remaining > 0) {
        rafRef.current = requestAnimationFrame(tick);
      } else {
        handleDismiss();
      }
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleDismiss = () => {
    setVisible(false);
    // Wait for slide-out animation before unmounting.
    setTimeout(onDismiss, 300);
  };

  const handleView = () => {
    handleDismiss();
    onView();
  };

  return (
    <div
      className={`
        fixed bottom-6 right-6 z-50 w-72
        bg-th-card-bg border border-blue-500/30 rounded-xl shadow-2xl
        transition-all duration-300 ease-out overflow-hidden
        ${visible ? "translate-y-0 opacity-100" : "translate-y-4 opacity-0"}
      `}
    >
      {/* Progress bar */}
      <div className="h-0.5 bg-blue-500/20">
        <div
          className="h-full bg-blue-400 transition-none"
          style={{ width: `${progress}%` }}
        />
      </div>

      <div className="px-4 py-3">
        <div className="flex items-start gap-3">
          {/* Icon */}
          <div className="p-2 rounded-lg bg-blue-500/15 border border-blue-500/25 shrink-0">
            <Sparkles size={14} className="text-blue-400" />
          </div>

          {/* Text */}
          <div className="flex-1 min-w-0 pt-0.5">
            <p className="text-sm font-semibold text-th-text-primary leading-tight">
              {count === 1 ? "New suggestion" : `${count} new suggestions`}
            </p>
            <p className="text-xs text-th-text-muted mt-0.5">
              Otto has something for you to review.
            </p>
          </div>

          {/* Close */}
          <button
            onClick={handleDismiss}
            className="p-1 rounded-md text-th-text-muted hover:text-th-text-secondary hover:bg-th-surface-hover transition-all shrink-0 -mt-0.5 -mr-1"
          >
            <X size={13} />
          </button>
        </div>

        {/* View button */}
        <button
          onClick={handleView}
          className="mt-3 w-full flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-blue-500/10 text-blue-300 border border-blue-500/20 hover:bg-blue-500/20 transition-all"
        >
          View suggestions
          <ChevronRight size={12} />
        </button>
      </div>
    </div>
  );
}
