import type { ReactNode } from "react";

export interface SegmentedOption<T extends string> {
  id: T;
  label: string;
  icon?: ReactNode;
  hint?: string;
}

interface SegmentedProps<T extends string> {
  value: T;
  options: SegmentedOption<T>[];
  onChange: (id: T) => void;
  label: string;
  disabled?: boolean;
  /** Hide the text and show only icons (labels stay as accessible names). */
  iconOnly?: boolean;
}

/** Apple-style segmented control with arrow-key navigation. */
export function Segmented<T extends string>({ value, options, onChange, label, disabled, iconOnly }: SegmentedProps<T>) {
  const move = (from: number, step: number) => {
    const next = options[(from + step + options.length) % options.length];
    onChange(next.id);
  };
  return (
    <div role="radiogroup" aria-label={label} className="flex shrink-0 rounded-lg bg-th-bg-tertiary p-0.5">
      {options.map((option, index) => {
        const active = option.id === value;
        return (
          <button
            key={option.id}
            type="button"
            role="radio"
            aria-checked={active}
            aria-label={iconOnly ? option.label : undefined}
            tabIndex={active ? 0 : -1}
            disabled={disabled}
            title={option.hint}
            onClick={() => onChange(option.id)}
            onKeyDown={(event) => {
              if (event.key === "ArrowRight" || event.key === "ArrowDown") { event.preventDefault(); move(index, 1); }
              if (event.key === "ArrowLeft" || event.key === "ArrowUp") { event.preventDefault(); move(index, -1); }
            }}
            className={`inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-2xs font-medium transition-all duration-fast disabled:opacity-60 ${
              active ? "bg-th-surface text-th-text-primary shadow-sm" : "text-th-text-muted hover:text-th-text-primary"
            }`}
          >
            {option.icon}
            {!iconOnly && option.label}
          </button>
        );
      })}
    </div>
  );
}
