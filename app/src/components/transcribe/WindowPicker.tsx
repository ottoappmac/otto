import { useCallback, useEffect, useState } from "react";
import { AppWindow, Loader2, RefreshCw, X, ShieldAlert } from "lucide-react";
import { api } from "../../hooks/useApi";
import type { CaptureWindow } from "../../types";

interface WindowPickerProps {
  open: boolean;
  /** Verb shown in the header ("Capture" or "Follow"). */
  title: string;
  onPick: (win: CaptureWindow) => void;
  onClose: () => void;
  onOpenSettings: () => void;
}

/**
 * Modal grid of open windows. Used both for one-shot window capture and for
 * choosing the window to "follow" for transcript-anchored auto-capture.
 */
export default function WindowPicker({
  open,
  title,
  onPick,
  onClose,
  onOpenSettings,
}: WindowPickerProps) {
  const [windows, setWindows] = useState<CaptureWindow[]>([]);
  const [loading, setLoading] = useState(false);
  const [supported, setSupported] = useState(true);

  const load = useCallback(() => {
    setLoading(true);
    api
      .captureWindows(true)
      .then((r) => {
        setSupported(r.supported);
        setWindows(r.windows ?? []);
      })
      .catch(() => {
        setSupported(false);
        setWindows([]);
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (open) load();
  }, [open, load]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  // Thumbnails require Screen Recording; if none came back, we likely lack it.
  const noThumbs = windows.length > 0 && windows.every((w) => !w.thumb_b64);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-8 bg-black/40 backdrop-blur-sm animate-fade-in"
      onClick={onClose}
    >
      <div
        className="w-full max-w-2xl max-h-[80vh] flex flex-col bg-th-card-bg border border-th-card-border rounded-2xl shadow-2xl overflow-hidden animate-pop-in"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 h-14 border-b border-th-border/70 shrink-0">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-th-surface-hover flex items-center justify-center">
              <AppWindow size={15} className="text-th-text-secondary" />
            </div>
            <div>
              <h2 className="text-[13px] font-semibold text-th-text-primary leading-tight">
                {title} a window
              </h2>
              <p className="text-[11px] text-th-text-muted leading-tight">
                Choose an open window
              </p>
            </div>
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={load}
              className="p-1.5 rounded-lg text-th-text-muted hover:text-th-text-primary hover:bg-th-surface-hover transition-colors"
              title="Refresh"
            >
              <RefreshCw size={15} className={loading ? "animate-spin" : ""} />
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-th-text-muted hover:text-th-text-primary hover:bg-th-surface-hover transition-colors"
            >
              <X size={16} />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          {!supported ? (
            <div className="flex flex-col items-center justify-center py-14 text-center px-8">
              <ShieldAlert size={22} className="text-amber-400 mb-3" />
              <p className="text-[13px] text-th-text-secondary leading-relaxed">
                Screen capture isn&apos;t available on this device.
              </p>
            </div>
          ) : loading && windows.length === 0 ? (
            <div className="flex items-center justify-center py-16 text-th-text-muted">
              <Loader2 size={20} className="animate-spin" />
            </div>
          ) : windows.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-center px-8 gap-3">
              <ShieldAlert size={22} className="text-amber-400" />
              <p className="text-[13px] text-th-text-secondary leading-relaxed">
                No windows found. This usually means Otto needs Screen Recording
                permission.
              </p>
              <button
                onClick={onOpenSettings}
                className="px-3.5 py-2 rounded-lg bg-th-tab-active-bg text-th-tab-active-fg text-[12px] font-semibold hover:opacity-90 transition-opacity"
              >
                Open System Settings
              </button>
            </div>
          ) : (
            <>
              {noThumbs && (
                <div className="mb-3 p-2.5 rounded-xl border border-amber-500/30 bg-amber-500/10 text-[11px] text-amber-300 flex items-center gap-2">
                  <ShieldAlert size={13} className="shrink-0" />
                  <span className="flex-1">
                    Grant Screen Recording permission to see window previews and
                    capture pixels.
                  </span>
                  <button
                    onClick={onOpenSettings}
                    className="shrink-0 underline hover:text-amber-200"
                  >
                    Settings
                  </button>
                </div>
              )}
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                {windows.map((win) => (
                  <button
                    key={win.window_id}
                    onClick={() => onPick(win)}
                    className="group flex flex-col text-left rounded-xl border border-th-border bg-th-surface hover:border-th-border-strong hover:bg-th-surface-hover transition-colors overflow-hidden focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                  >
                    <div className="aspect-[16/10] bg-th-inset-bg flex items-center justify-center overflow-hidden">
                      {win.thumb_b64 ? (
                        <img
                          src={`data:image/png;base64,${win.thumb_b64}`}
                          alt=""
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <AppWindow size={22} className="text-th-text-faint" />
                      )}
                    </div>
                    <div className="px-2.5 py-2 min-w-0">
                      <p className="text-[12px] font-medium text-th-text-primary truncate">
                        {win.app}
                      </p>
                      {win.title && (
                        <p className="text-[11px] text-th-text-muted truncate">
                          {win.title}
                        </p>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
