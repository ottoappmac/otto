/**
 * Tailwind classes for the small "model used" chip displayed next to a
 * delegated subagent in the chat UI. Matches the family of LLM that
 * actually runs the subagent (frontier / mlx / EXO / custom / inherited).
 */
export function familyChipClasses(family: string): string {
  switch (family) {
    case "frontier":
      return "bg-sky-500/10 text-sky-300 border-sky-500/25";
    case "openai":
      return "bg-green-500/10 text-green-300 border-green-500/25";
    case "mlx":
      return "bg-blue-500/10 text-blue-300 border-blue-500/25";
    case "omlx":
      return "bg-violet-500/10 text-violet-300 border-violet-500/25";
    case "exo":
      return "bg-emerald-500/10 text-emerald-300 border-emerald-500/25";
    case "custom":
      return "bg-amber-500/10 text-amber-300 border-amber-500/25";
    default:
      return "bg-th-surface text-th-text-tertiary border-th-border";
  }
}
