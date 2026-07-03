import type { LucideIcon } from "lucide-react";

interface StatCardProps {
  label: string;
  value: string | number;
  subValue?: string;
  icon: LucideIcon;
  iconClassName?: string;
  trend?: "up" | "down" | "neutral";
  loading?: boolean;
}

export function StatCard({
  label,
  value,
  subValue,
  icon: Icon,
  iconClassName = "text-th-text-muted",
  loading = false,
}: StatCardProps) {
  return (
    <div className="bg-th-card-bg border border-th-card-border rounded-2xl px-4 py-3.5 flex items-start gap-3 shadow-sm shadow-black/[0.03] transition-all duration-200 hover:shadow-md hover:shadow-black/[0.06]">
      <div className="w-9 h-9 rounded-xl bg-th-inset-bg border border-th-border flex items-center justify-center shrink-0">
        <Icon size={17} className={iconClassName} aria-hidden />
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-[11px] text-th-text-muted font-medium uppercase tracking-wider truncate">{label}</p>
        {loading ? (
          <div className="mt-1 h-6 w-16 rounded bg-th-inset-bg animate-pulse" />
        ) : (
          <p className="text-xl font-semibold tracking-tight text-th-text-primary leading-tight mt-0.5 tabular-nums">{value}</p>
        )}
        {subValue && !loading && (
          <p className="text-[11px] text-th-text-muted mt-0.5 truncate">{subValue}</p>
        )}
      </div>
    </div>
  );
}
