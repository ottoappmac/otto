import { useEffect, type ReactNode } from "react";

interface ToastProps {
  children: ReactNode;
  /** Optional action, e.g. Undo. */
  action?: { label: string; onClick: () => void };
  onDismiss: () => void;
  /** Auto-dismiss after this many ms. 0 keeps it until dismissed. */
  duration?: number;
}

export function Toast({ children, action, onDismiss, duration = 5000 }: ToastProps) {
  useEffect(() => {
    if (!duration) return;
    const timer = window.setTimeout(onDismiss, duration);
    return () => window.clearTimeout(timer);
  }, [duration, onDismiss, children]);
  return (
    <div
      role="status"
      className="fixed bottom-4 right-4 z-40 flex max-w-sm animate-toast-in items-center gap-3 rounded-xl border border-th-border bg-th-dropdown-bg px-3.5 py-2.5 text-xs text-th-text-primary shadow-xl"
    >
      <span className="min-w-0 flex-1">{children}</span>
      {action && (
        <button
          type="button"
          onClick={() => { action.onClick(); onDismiss(); }}
          className="shrink-0 font-medium text-th-accent hover:underline"
        >
          {action.label}
        </button>
      )}
    </div>
  );
}
