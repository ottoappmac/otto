/**
 * AmbientNotificationBanner — a slim top-of-page banner that appears
 * whenever new ambient hints arrive, on any page of the app.
 *
 * Renders just below the ConnectionBanner in Layout so it's always
 * visible regardless of which page the user is on.  Auto-dismisses
 * when the user navigates to /ambient.
 */
import { useEffect, useState } from "react";
import { Sparkles, X, ArrowRight } from "lucide-react";
import type { AmbientHint } from "../../types";

interface Props {
  hints: AmbientHint[];
  onView: () => void;
  onDismiss: () => void;
}

export default function AmbientNotificationBanner({ hints, onView, onDismiss }: Props) {
  const [visible, setVisible] = useState(false);
  const count = hints.length;
  const firstTitle = hints[0]?.title;

  // Slide in from the top on mount.
  useEffect(() => {
    const tid = setTimeout(() => setVisible(true), 10);
    return () => clearTimeout(tid);
  }, []);

  const handleDismiss = () => {
    setVisible(false);
    setTimeout(onDismiss, 200);
  };

  const handleView = () => {
    setVisible(false);
    setTimeout(onView, 200);
  };

  return (
    <div
      className={`flex items-center gap-3 px-4 py-2 bg-blue-500/10 border-b border-blue-500/20 overflow-hidden transition-all duration-200 ${
        visible ? "max-h-12 opacity-100" : "max-h-0 opacity-0"
      }`}
    >
      {/* Icon */}
      <div className="p-1 rounded-md bg-blue-500/15 border border-blue-500/25 shrink-0">
        <Sparkles size={12} className="text-blue-400" />
      </div>

      {/* Message */}
      <p className="flex-1 min-w-0 text-blue-400 text-xs">
        <span className="font-semibold">
          {count === 1 ? "1 new suggestion" : `${count} new suggestions`}
        </span>
        {firstTitle && (
          <span className="text-blue-400/70 ml-1.5">
            — {firstTitle}{count > 1 && ` +${count - 1} more`}
          </span>
        )}
      </p>

      {/* View button */}
      <button
        onClick={handleView}
        className="shrink-0 inline-flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-medium bg-blue-500/15 text-blue-400 border border-blue-500/25 hover:bg-blue-500/25 transition-all"
      >
        View in Suggestions
        <ArrowRight size={11} />
      </button>

      {/* Dismiss */}
      <button
        onClick={handleDismiss}
        className="shrink-0 p-1 rounded-md text-blue-400/70 hover:text-blue-400 hover:bg-blue-500/15 transition-all"
        title="Dismiss"
      >
        <X size={13} />
      </button>
    </div>
  );
}
