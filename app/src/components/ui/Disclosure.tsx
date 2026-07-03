import { useState, type ReactNode } from "react";
import { ChevronDown, type LucideIcon } from "lucide-react";

interface DisclosureProps {
  title: string;
  /** Stable key for persisting open state in localStorage. */
  storageKey: string;
  defaultOpen?: boolean;
  icon?: LucideIcon;
  iconClassName?: string;
  /** Optional summary text shown on the right of the header (e.g. a count). */
  summary?: ReactNode;
  children: ReactNode;
}

export function Disclosure({
  title,
  storageKey,
  defaultOpen = false,
  icon: Icon,
  iconClassName = "text-th-text-muted",
  summary,
  children,
}: DisclosureProps) {
  const fullKey = `dashboard.disclosure.${storageKey}`;
  const [open, setOpen] = useState<boolean>(() => {
    const v = localStorage.getItem(fullKey);
    return v == null ? defaultOpen : v === "1";
  });

  const toggle = () => {
    setOpen((o) => {
      const next = !o;
      localStorage.setItem(fullKey, next ? "1" : "0");
      return next;
    });
  };

  return (
    <section>
      <button
        type="button"
        onClick={toggle}
        className="flex w-full items-center gap-2 py-1.5 group text-left transition-all duration-200 active:scale-[0.995]"
        aria-expanded={open}
      >
        {Icon && <Icon size={14} className={iconClassName} aria-hidden />}
        <h2 className="text-sm font-semibold tracking-tight text-th-text-primary">{title}</h2>
        {summary != null && (
          <span className="text-xs text-th-text-muted tabular-nums">{summary}</span>
        )}
        <ChevronDown
          size={16}
          aria-hidden
          className={`ml-auto text-th-text-muted transition-transform duration-200 ${open ? "rotate-180" : ""}`}
        />
      </button>
      {open && <div className="mt-2.5 animate-slide-up">{children}</div>}
    </section>
  );
}
