export type RunDepth = "auto" | "quick" | "deep";

const OPTIONS: { id: RunDepth; label: string; title: string }[] = [
  { id: "auto", label: "Auto", title: "Let Otto decide Quick or Deep for each message" },
  { id: "quick", label: "Quick", title: "Short answer, few tool calls" },
  { id: "deep", label: "Deep", title: "Full research and planning" },
];

interface DepthPickerProps {
  value: RunDepth;
  onChange: (depth: RunDepth) => void;
}

export function DepthPicker({ value, onChange }: DepthPickerProps) {
  return (
    <div className="inline-flex items-center rounded-md border border-th-border p-0.5 shrink-0">
      {OPTIONS.map((option) => {
        const selected = option.id === value;
        return (
          <button
            key={option.id}
            type="button"
            title={option.title}
            onClick={() => onChange(option.id)}
            className={`px-1.5 py-0.5 text-[10px] rounded transition-colors ${
              selected
                ? "bg-th-tab-active-bg text-th-tab-active-fg"
                : "text-th-text-muted hover:text-th-text-primary"
            }`}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
