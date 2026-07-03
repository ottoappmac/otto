import type { RunStatus } from "../../types";
import { getStatusIcon, getStatusLabel, getStatusColor } from "../../utils/entityIcons";

interface RunStatusBadgeProps {
  status: RunStatus | string;
  showLabel?: boolean;
  size?: "sm" | "md";
}

export function RunStatusBadge({ status, showLabel = true, size = "sm" }: RunStatusBadgeProps) {
  const { Icon, className } = getStatusIcon(status);
  const label = getStatusLabel(status);
  const textColor = getStatusColor(status);

  const iconSize = size === "md" ? 13 : 11;
  const textSize = size === "md" ? "text-xs" : "text-[11px]";

  return (
    <span className={`inline-flex items-center gap-1 font-medium ${textSize} ${textColor}`}>
      <Icon size={iconSize} className={className} aria-hidden />
      {showLabel && <span>{label}</span>}
    </span>
  );
}
