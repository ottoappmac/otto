import type { ReactNode } from "react";

export type BadgeTone = "neutral" | "accent" | "success" | "warning" | "danger";

const TONES: Record<BadgeTone, string> = {
  neutral: "bg-th-surface-hover text-th-text-tertiary",
  accent: "bg-th-accent/12 text-th-accent",
  success: "bg-th-success/12 text-th-success",
  warning: "bg-th-warning/15 text-th-warning",
  danger: "bg-th-danger/12 text-th-danger",
};

export function Badge({ tone = "neutral", children, title }: { tone?: BadgeTone; children: ReactNode; title?: string }) {
  return (
    <span title={title} className={`inline-flex shrink-0 items-center gap-1 rounded-full px-2 py-0.5 text-2xs font-medium ${TONES[tone]}`}>
      {children}
    </span>
  );
}
