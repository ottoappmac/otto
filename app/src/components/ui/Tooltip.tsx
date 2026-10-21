import { cloneElement, useEffect, useRef, useState, type ReactElement, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { Kbd } from "./Kbd";

interface TooltipProps {
  label: ReactNode;
  /** Keys shown as keycaps, e.g. ["⌘", "S"]. */
  keys?: string[];
  side?: "top" | "bottom";
  children: ReactElement;
}

/**
 * Fast hover/focus tooltip rendered in a portal so overflow-hidden panels never
 * clip it. Replaces the slow native `title` attribute on icon-only controls.
 */
export function Tooltip({ label, keys, side = "bottom", children }: TooltipProps) {
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);
  const timer = useRef<number | null>(null);
  const id = useRef(`tip-${Math.random().toString(36).slice(2, 8)}`).current;

  const clear = () => {
    if (timer.current !== null) window.clearTimeout(timer.current);
    timer.current = null;
  };
  useEffect(() => clear, []);

  const show = (el: HTMLElement) => {
    clear();
    timer.current = window.setTimeout(() => {
      const r = el.getBoundingClientRect();
      setPos({ x: r.left + r.width / 2, y: side === "bottom" ? r.bottom + 6 : r.top - 6 });
    }, 350);
  };
  const hide = () => {
    clear();
    setPos(null);
  };

  const child = cloneElement(children, {
    "aria-describedby": pos ? id : undefined,
    onMouseEnter: (e: React.MouseEvent<HTMLElement>) => { children.props.onMouseEnter?.(e); show(e.currentTarget); },
    onMouseLeave: (e: React.MouseEvent<HTMLElement>) => { children.props.onMouseLeave?.(e); hide(); },
    onFocus: (e: React.FocusEvent<HTMLElement>) => { children.props.onFocus?.(e); show(e.currentTarget); },
    onBlur: (e: React.FocusEvent<HTMLElement>) => { children.props.onBlur?.(e); hide(); },
    onPointerDown: (e: React.PointerEvent<HTMLElement>) => { children.props.onPointerDown?.(e); hide(); },
  });

  return (
    <>
      {child}
      {pos &&
        createPortal(
          <div
            id={id}
            role="tooltip"
            style={{ left: pos.x, top: pos.y, transform: `translate(-50%, ${side === "top" ? "-100%" : "0"})` }}
            className="pointer-events-none fixed z-[100] flex max-w-xs animate-fade-in items-center gap-1.5 rounded-md bg-neutral-900 px-2 py-1 text-2xs text-white shadow-lg"
          >
            <span>{label}</span>
            {keys && <span className="flex gap-0.5">{keys.map((k) => <Kbd key={k}>{k}</Kbd>)}</span>}
          </div>,
          document.body,
        )}
    </>
  );
}
