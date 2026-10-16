/** Categories the user can pre-authorise on a schedule or trigger.
 *  persistence is absent on purpose: a run must not create more runs. */
export const ALLOWLIST_OPTIONS: { id: string; label: string }[] = [
  { id: "shell", label: "Shell" },
  { id: "applescript", label: "AppleScript" },
  { id: "desktop_control", label: "Desktop control" },
  { id: "email_send", label: "Send email" },
  { id: "mail_read", label: "Read email" },
  { id: "message_send", label: "Send message" },
  { id: "message_read", label: "Read messages" },
  { id: "destructive", label: "Delete or modify" },
  { id: "mcp_authoring", label: "Install tool servers" },
  { id: "settings_write", label: "Change settings" },
  { id: "pim_write", label: "Calendar, reminders, notes" },
  { id: "work_tools_write", label: "Work tools" },
  { id: "browser_automation", label: "Browser" },
  { id: "unclassified", label: "Unreviewed tools" },
];

export const GRANT_NOUN: Record<string, string> = {
  shell: "shell commands",
  mail_read: "reading email",
  message_read: "reading messages",
  pim_write: "calendar and notes changes",
  work_tools_write: "work-tool writes",
};

const RISK_BADGES: Record<string, string> = {
  "macos-native": "Controls this Mac",
  "macos-osascript": "Runs AppleScript",
  "macos-mail": "Can send email",
  "macos-messages": "Can send messages",
  "macos-calendar": "Changes calendar",
  "macos-reminders": "Changes reminders",
  "macos-notes": "Changes notes",
  slack: "Can send messages",
  discord: "Can send messages",
  "microsoft-teams": "Can send messages",
  atlassian: "Writes to Jira and Confluence",
  "microsoft-onedrive": "Writes to OneDrive",
  blender: "Runs Blender code",
  "playwright-mcp": "Drives the browser",
};

export function mcpRiskBadge(serverId: string, generated: boolean): string | null {
  if (RISK_BADGES[serverId]) return RISK_BADGES[serverId];
  if (generated) return "Unreviewed tools";
  return null;
}
