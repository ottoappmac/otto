/**
 * Format an ISO date string as a human-readable relative time.
 *
 * @param iso  ISO-8601 date string
 * @param style "short" returns "just now / 3m ago / 2h ago", "long" adds "Yesterday" and "3d ago"
 */
export function formatRelativeTime(
  iso: string,
  style: "short" | "long" = "short",
): string {
  const d = new Date(iso);
  const now = new Date();
  const diffMs = now.getTime() - d.getTime();
  const diffMins = Math.floor(diffMs / 60000);

  if (diffMins < 1) return "Just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  const diffHrs = Math.floor(diffMins / 60);
  if (diffHrs < 24) return `${diffHrs}h ago`;

  if (style === "long") {
    const diffDays = Math.floor(diffHrs / 24);
    if (diffDays === 1) return "Yesterday";
    if (diffDays < 7) return `${diffDays}d ago`;
    return d.toLocaleDateString();
  }

  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}
