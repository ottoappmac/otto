import { useEffect, useRef, useState } from "react";
import { Check, ChevronDown, ChevronUp } from "lucide-react";

interface ModelPickerProps {
  value: string;
  models: { id: string; name: string }[];
  onChange: (id: string) => void;
}

export function ModelPicker({ value, models, onChange }: ModelPickerProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const displayName = models.find((m) => m.id === value)?.name || value || "Select model";

  return (
    <div ref={ref} className="relative inline-block">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 px-2 py-0.5 text-[11px] text-th-text-tertiary hover:text-th-text-primary hover:bg-th-surface-hover rounded-md transition-all"
      >
        <span className="truncate max-w-[200px]">{displayName}</span>
        {open ? <ChevronDown size={10} /> : <ChevronUp size={10} />}
      </button>
      {open && (
        <div className="absolute bottom-full left-0 mb-1 w-72 max-h-64 overflow-y-auto bg-th-card-bg border border-th-border rounded-lg shadow-xl z-50">
          {models.length === 0 ? (
            <div className="px-3 py-2 text-xs text-th-text-muted">No models available. Configure credentials in Settings.</div>
          ) : (
            models.map((m) => (
              <button
                key={m.id}
                onClick={() => { onChange(m.id); setOpen(false); }}
                className={`w-full text-left px-3 py-2 text-xs transition-colors flex items-center justify-between gap-2 ${
                  m.id === value
                    ? "bg-th-tab-active-bg text-th-tab-active-fg"
                    : "text-th-text-tertiary hover:bg-th-surface-hover hover:text-th-text-primary"
                }`}
              >
                <span className="truncate">{m.name}</span>
                {m.id === value && <Check size={12} className="text-emerald-400 shrink-0" />}
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}
