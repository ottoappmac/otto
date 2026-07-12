import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";

interface PopoverRenderProps {
  open: boolean;
  toggle: () => void;
  close: () => void;
}

interface PopoverProps {
  /** Render the trigger element. Wire its onClick to `toggle`. */
  trigger: (props: PopoverRenderProps & { "aria-expanded": boolean; "aria-haspopup": "menu" | "dialog" }) => ReactNode;
  /** Render the panel contents. Use `close` to dismiss after an action. */
  children: (props: PopoverRenderProps) => ReactNode;
  /** Horizontal alignment of the panel relative to the trigger. */
  align?: "left" | "right";
  /** Extra classes for the floating panel. */
  panelClassName?: string;
  /** Semantic role of the panel. */
  role?: "menu" | "dialog";
  /** Fired whenever the popover closes. */
  onClose?: () => void;
}

interface Coords {
  top: number;
  left?: number;
  right?: number;
}

export function Popover({
  trigger,
  children,
  align = "left",
  panelClassName = "",
  role = "dialog",
  onClose,
}: PopoverProps) {
  const [open, setOpen] = useState(false);
  const [coords, setCoords] = useState<Coords | null>(null);
  const triggerRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  const close = useCallback(() => {
    setOpen(false);
    onClose?.();
  }, [onClose]);

  const toggle = useCallback(() => {
    setOpen((o) => {
      if (o) onClose?.();
      return !o;
    });
  }, [onClose]);

  const reposition = useCallback(() => {
    const el = triggerRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const panelH = panelRef.current?.offsetHeight ?? 0;
    // Flip above the trigger when there isn't room below.
    let top = r.bottom + 6;
    if (panelH && top + panelH > window.innerHeight - 8) {
      top = Math.max(8, r.top - panelH - 6);
    }
    if (align === "right") {
      setCoords({ top, right: Math.max(8, window.innerWidth - r.right) });
    } else {
      setCoords({ top, left: r.left });
    }
  }, [align]);

  useLayoutEffect(() => {
    if (!open) return;
    reposition();
    // Second pass once the panel has mounted so we know its height (for flipping).
    const raf = requestAnimationFrame(reposition);
    return () => cancelAnimationFrame(raf);
  }, [open, reposition]);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: PointerEvent) => {
      const t = e.target as Node;
      if (
        triggerRef.current?.contains(t) ||
        panelRef.current?.contains(t)
      ) {
        return;
      }
      close();
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    const onReflow = () => reposition();
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    window.addEventListener("resize", onReflow);
    // Capture phase so scrolling in any ancestor repositions the panel.
    window.addEventListener("scroll", onReflow, true);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("resize", onReflow);
      window.removeEventListener("scroll", onReflow, true);
    };
  }, [open, close, reposition]);

  const renderProps: PopoverRenderProps = { open, toggle, close };

  return (
    <div ref={triggerRef} className="inline-flex min-w-0">
      {trigger({
        ...renderProps,
        "aria-expanded": open,
        "aria-haspopup": role,
      })}
      {open && coords &&
        createPortal(
          <div
            ref={panelRef}
            role={role}
            style={{
              position: "fixed",
              top: coords.top,
              left: coords.left,
              right: coords.right,
            }}
            className={`z-50 rounded-2xl border border-th-border/70 bg-th-bg-secondary/90 backdrop-blur-xl shadow-2xl shadow-black/10 ring-1 ring-black/[0.04] animate-pop-in ${
              align === "right" ? "origin-top-right" : "origin-top-left"
            } ${panelClassName}`}
          >
            {children(renderProps)}
          </div>,
          document.body,
        )}
    </div>
  );
}
