import { ALLOWLIST_OPTIONS } from "./actionCategories";

export function AllowlistField({
  value,
  onChange,
}: {
  value: string[];
  onChange: (next: string[]) => void;
}) {
  const selected = new Set(value);
  const toggle = (id: string) => {
    const next = new Set(selected);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    onChange(ALLOWLIST_OPTIONS.map((option) => option.id).filter((id) => next.has(id)));
  };
  return (
    <div>
      <p className="text-sm font-medium text-th-text-tertiary mb-1">Allowed while nobody is here</p>
      <p className="text-xs text-th-text-muted mb-2">
        Anything not selected is refused when this runs on its own. Scheduling more runs is never allowed.
      </p>
      <div className="flex flex-wrap gap-1.5">
        {ALLOWLIST_OPTIONS.map((option) => {
          const on = selected.has(option.id);
          return (
            <button
              key={option.id}
              type="button"
              onClick={() => toggle(option.id)}
              className={`px-2 py-1 text-2xs rounded-md border transition-colors ${
                on
                  ? "bg-th-tab-active-bg text-th-tab-active-fg border-transparent"
                  : "bg-th-inset-bg text-th-text-tertiary border-th-border hover:text-th-text-secondary"
              }`}
            >
              {option.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
