// Detection of high-risk shell commands shown in the human-in-the-loop
// approval UI. Mirrors `backend/safety_middleware.py:_HIGH_RISK_PATTERNS`.
// Keep the two in sync when adding a pattern — the backend logs each
// match at WARNING; the frontend uses the result to render a prominent
// red badge on the approval card so a fast click-through can't slip a
// destructive command past the user.

export type HighRiskLabel =
  | "rm-rf-root"
  | "rm-rf-flag-split"
  | "rm-rf-flag-then-/"
  | "force-push"
  | "force-push-main"
  | "hard-reset"
  | "dd-of-device"
  | "mkfs"
  | "chmod-recursive-/"
  | "curl-pipe-shell"
  | "sudo-rm"
  | "shutdown";

const HIGH_RISK_PATTERNS: ReadonlyArray<readonly [HighRiskLabel, RegExp]> = [
  ["rm-rf-root", /\brm\s+(?:-[a-zA-Z]*\s+)*(?:-rf|-fr)\s+(?:\/|~|\$HOME)(?=$|\s|\/)/],
  ["rm-rf-flag-split", /\brm\s+(?:-[a-zA-Z]*r[a-zA-Z]*\s+)(?:-[a-zA-Z]*f[a-zA-Z]*\s+)(?:\/|~|\$HOME)(?=$|\s|\/)/],
  ["force-push", /\bgit\s+push\s+(?:--force|-f)\b/],
  ["force-push-main", /\bgit\s+push\s+.*--force.*\b(main|master)\b/i],
  ["hard-reset", /\bgit\s+reset\s+--hard\b/],
  ["dd-of-device", /\bdd\b[^\n]*\bof=\/dev\//],
  ["mkfs", /\bmkfs(?:\.[a-z0-9]+)?\b/],
  ["chmod-recursive-/", /\bchmod\s+-R\s+[0-7]+\s+\/\s/],
  ["curl-pipe-shell", /\b(?:curl|wget)\b[^\n|]*\|\s*(?:sh|bash|zsh)\b/],
  ["sudo-rm", /\bsudo\s+rm\b/],
  ["shutdown", /\b(?:shutdown|reboot|halt|poweroff)\b/],
];

const RISK_DESCRIPTIONS: Record<HighRiskLabel, string> = {
  "rm-rf-root": "Recursive delete from / or home",
  "rm-rf-flag-split": "Recursive delete from / or home",
  "rm-rf-flag-then-/": "Recursive delete from /",
  "force-push": "Force-push rewrites remote history",
  "force-push-main": "Force-push to main/master",
  "hard-reset": "Hard reset discards uncommitted work",
  "dd-of-device": "Direct write to a block device",
  "mkfs": "Filesystem format wipes data",
  "chmod-recursive-/": "Recursive permission change from /",
  "curl-pipe-shell": "Pipe download into shell",
  "sudo-rm": "Privileged delete",
  "shutdown": "Power state change",
};

/**
 * Return all high-risk labels matched by the given shell command.
 * Empty array when the command is safe (or empty / non-string).
 */
export function screenHighRiskCommand(command: unknown): HighRiskLabel[] {
  if (typeof command !== "string" || !command) return [];
  return HIGH_RISK_PATTERNS.filter(([, p]) => p.test(command)).map(([label]) => label);
}

/** Human-readable explanation for a label, for use in tooltips / banners. */
export function describeHighRiskLabel(label: HighRiskLabel): string {
  return RISK_DESCRIPTIONS[label];
}
