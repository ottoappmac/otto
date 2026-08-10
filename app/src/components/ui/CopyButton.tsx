import { useCallback, useEffect, useRef, useState } from "react";
import { Check, Copy, Loader2, X } from "lucide-react";
import { copyText } from "../../utils/clipboard";

interface CopyButtonProps {
  /** Plain text to copy. Ignored when `onCopy` is given. */
  text?: string | null;
  /**
   * Custom copy routine returning whether it succeeded. Use for payloads that
   * must be fetched or encoded on demand (file contents, images).
   */
  onCopy?: () => boolean | Promise<boolean>;
  /** Optional caption rendered next to the icon. */
  label?: string;
  title?: string;
  /** Icon size in px. */
  size?: number;
  className?: string;
  disabled?: boolean;
}

type State = "idle" | "busy" | "copied" | "failed";

/** Icon button that copies to the clipboard and confirms with a tick. */
export function CopyButton({
  text,
  onCopy,
  label,
  title = "Copy",
  size = 13,
  className = "",
  disabled,
}: CopyButtonProps) {
  const [state, setState] = useState<State>("idle");
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);

  // Re-arm on every mount: StrictMode runs the cleanup once before the real
  // mount, which would otherwise leave the flag false for the whole lifetime.
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  const handleClick = useCallback(async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (state === "busy") return;
    setState("busy");
    let ok = false;
    try {
      ok = onCopy ? await onCopy() : await copyText(text ?? "");
    } catch {
      ok = false;
    }
    if (!mountedRef.current) return;
    setState(ok ? "copied" : "failed");
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      if (mountedRef.current) setState("idle");
    }, 1800);
  }, [onCopy, state, text]);

  const isEmpty = !onCopy && !text;
  const Icon = state === "busy" ? Loader2 : state === "copied" ? Check : state === "failed" ? X : Copy;

  return (
    <button
      type="button"
      onClick={handleClick}
      disabled={disabled || isEmpty}
      title={state === "copied" ? "Copied" : state === "failed" ? "Copy failed" : title}
      aria-label={title}
      className={`inline-flex items-center gap-1 rounded-md transition-colors disabled:opacity-40 disabled:cursor-default ${
        state === "copied"
          ? "text-emerald-400"
          : state === "failed"
            ? "text-red-400"
            : "text-th-text-muted hover:text-th-text-primary hover:bg-th-surface-hover"
      } ${className}`}
    >
      <Icon size={size} className={state === "busy" ? "animate-spin" : undefined} />
      {label && <span>{state === "copied" ? "Copied" : label}</span>}
    </button>
  );
}
