interface BarItem {
  label: string;
  count: number;
  color?: string;
}

interface BreakdownBarProps {
  items: BarItem[];
  title: string;
  emptyMessage?: string;
  max?: number;
}

export function BreakdownBar({ items, title, emptyMessage = "No data", max }: BreakdownBarProps) {
  const topN = items.slice(0, 8);
  const maxCount = max ?? Math.max(...topN.map((i) => i.count), 1);

  return (
    <div className="bg-th-card-bg border border-th-card-border rounded-2xl p-4 shadow-sm shadow-black/[0.03]">
      <h3 className="text-[11px] text-th-text-muted font-semibold uppercase tracking-wider mb-3">{title}</h3>
      {topN.length === 0 ? (
        <p className="text-xs text-th-text-muted">{emptyMessage}</p>
      ) : (
        <div className="space-y-2">
          {topN.map((item) => {
            const pct = Math.round((item.count / maxCount) * 100);
            return (
              <div key={item.label} className="flex items-center gap-2">
                <span
                  className="text-[11px] text-th-text-secondary truncate shrink-0"
                  style={{ minWidth: "7rem", maxWidth: "10rem" }}
                  title={item.label}
                >
                  {item.label}
                </span>
                <div className="flex-1 h-1.5 rounded-full bg-th-inset-bg overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all ${item.color ?? "bg-blue-500/60"}`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <span className="text-[11px] text-th-text-muted w-7 text-right shrink-0">{item.count}</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
