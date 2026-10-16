"""What each tool is allowed to mean, independent of the user's settings.

A tool is either explicitly classified (a category, or ``None`` for a
read that is not gated) or it falls through to ``unclassified``. Unknown
tools on a server Otto did not ship — generated MCPs, external MCPs —
are unclassified on purpose. Unknown bare tools (research helpers and
the like) are not gated; there are too many of them to treat a new
read as a side effect.

Content patterns (``do shell script``, ``osascript`` inside ``execute``)
are labels plus an extra category. They are not a sandbox. A regex does
not stop a determined command; the execute sandbox does.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Stable order is the order Settings renders.
CATEGORIES: tuple[str, ...] = (
    "shell",
    "applescript",
    "desktop_control",
    "email_send",
    "mail_read",
    "message_send",
    "message_read",
    "destructive",
    "persistence",
    "mcp_authoring",
    "settings_write",
    "pim_write",
    "work_tools_write",
    "browser_automation",
    "workspace_write",
    "unclassified",
)

CATEGORY_TITLES: dict[str, str] = {
    "shell": "Run command?",
    "applescript": "Run AppleScript?",
    "desktop_control": "Control this Mac?",
    "email_send": "Send email?",
    "mail_read": "Read email?",
    "message_send": "Send message?",
    "message_read": "Read messages?",
    "destructive": "Delete or modify?",
    "persistence": "Schedule an unattended run?",
    "mcp_authoring": "Install a tool server?",
    "settings_write": "Change Otto settings?",
    "pim_write": "Change calendar, reminders, or notes?",
    "work_tools_write": "Post to a work tool?",
    "browser_automation": "Drive the browser?",
    "workspace_write": "Edit the allowed folder?",
    "unclassified": "Run an unreviewed tool?",
}

CATEGORY_BLURB: dict[str, str] = {
    "shell": "Shell commands (execute), including inside sub-agents.",
    "applescript": "AppleScript and JavaScript for Automation.",
    "desktop_control": "Mouse, keyboard, and launching apps.",
    "email_send": "Sending mail or creating a draft that can be sent.",
    "mail_read": "Listing or opening mail, including the inbox.",
    "message_send": "iMessage, Slack, Discord, and similar sends.",
    "message_read": "Listing chats or reading message history.",
    "destructive": "Deleting mail, events, reminders, or notes.",
    "persistence": "Schedules and triggers, which run with nobody watching.",
    "mcp_authoring": "Creating, connecting, or registering tool servers.",
    "settings_write": "Changing settings, including which tool servers are on.",
    "pim_write": "Creating or editing calendar events, reminders, and notes.",
    "work_tools_write": "Jira, Confluence, and OneDrive writes.",
    "browser_automation": "Playwright clicks, typing, and navigation.",
    "workspace_write": "write_file and edit_file inside a folder you allowed. Denied outside that folder even when this is Allow.",
    "unclassified": "Tools Otto does not recognise (generated or third-party).",
}

# A session grant may cover later calls. Invocation grants last for one
# agent run only and are not offered as a button.
SESSION_GRANTABLE = frozenset({
    "shell", "pim_write", "work_tools_write", "mail_read", "message_read",
})
# These can be granted for the rest of the chat, but the first call in a
# session still shows a card. The global auto-approve setting must not skip it.
NO_AUTO_APPROVE = frozenset({"mail_read", "message_read"})
INVOCATION_GRANTABLE = frozenset({"desktop_control"})

# After the agent has read untrusted content, these ignore grants and
# cannot stay on "allow".
OUTBOUND_CATEGORIES = frozenset({
    "email_send", "message_send", "work_tools_write",
})

# macOS high-risk servers. The kill switch drops them at attach time and
# the gate refuses them at call time.
MACOS_SERVER_IDS = frozenset({
    "macos-native",
    "macos-osascript",
    "macos-mail",
    "macos-messages",
    "macos-calendar",
    "macos-reminders",
    "macos-notes",
})

# Built-in servers whose tools we expect to have classified. A new tool
# on one of these that is missing from the map becomes ``unclassified``
# at runtime, and the catalog test fails in CI.
KNOWN_SERVER_IDS = MACOS_SERVER_IDS | frozenset({
    "slack",
    "discord",
    "atlassian",
    "microsoft-teams",
    "microsoft-onedrive",
    "edgar-sec",
    "blender",
    "playwright-mcp",
    "agent-eval-service",
    "claude-eval-hook",
    "openclaw-eval-hook",
})

# Servers that must NOT be seatbelt-wrapped: they exist to send Apple
# Events or drive the UI, which a sandbox would make impossible.
UNSANDBOXED_SERVER_IDS = MACOS_SERVER_IDS


def drop_macos_servers(servers, *, macos_automation_enabled: bool):
    """Drop macOS high-risk servers when the kill switch is off.

    Used when a session graph is built. A tool that is already attached
    is still refused at call time by the middleware.
    """
    if macos_automation_enabled:
        return list(servers)
    return [
        server for server in servers
        if getattr(server, "id", None) not in MACOS_SERVER_IDS
    ]


# (server_id, tool_name) -> category. None means "reviewed, not gated".
_SERVER_TOOLS: dict[tuple[str, str], str | None] = {}


def _read(server: str, *names: str) -> None:
    for name in names:
        _SERVER_TOOLS[(server, name)] = None


def _cat(server: str, category: str, *names: str) -> None:
    for name in names:
        _SERVER_TOOLS[(server, name)] = category


_cat("macos-osascript", "applescript", "run_osascript", "run_osascript_file")
_cat("macos-osascript", "desktop_control", "dump_ax_tree")
_cat("macos-osascript", "mail_read", "query_mail_store", "query_outlook_store")
_read("macos-osascript", "inspect_app_dictionary")

_cat("macos-mail", "email_send", "send_message", "create_draft")
_cat("macos-mail", "destructive", "update_message", "delete_message")
_cat(
    "macos-mail", "mail_read",
    "list_accounts", "list_mailboxes", "list_messages", "get_message",
    "search_messages",
)

_cat("macos-messages", "message_send", "send_message")
_cat("macos-messages", "message_read", "list_chats", "list_buddies", "read_messages")

_cat("macos-calendar", "pim_write", "create_event", "update_event")
_cat("macos-calendar", "destructive", "delete_event")
_read("macos-calendar", "list_calendars", "list_events", "get_event")

_cat("macos-reminders", "pim_write", "create_reminder", "update_reminder")
_cat("macos-reminders", "destructive", "delete_reminder")
_read("macos-reminders", "list_lists", "list_reminders", "get_reminder")

_cat("macos-notes", "pim_write", "create_note", "update_note")
_cat("macos-notes", "destructive", "delete_note")
_read("macos-notes", "list_folders", "list_notes", "get_note", "search_notes")

_cat("slack", "message_send", "send_message", "add_reaction", "join_channel")
_read(
    "slack",
    "list_channels", "get_channel_info", "get_channel_history",
    "get_thread_replies", "list_users", "get_user_info",
)

_cat("discord", "message_send", "send_message", "add_reaction")
_read(
    "discord",
    "list_guilds", "get_guild_info", "list_guild_members", "list_channels",
    "get_channel_messages",
)

_cat(
    "atlassian", "work_tools_write",
    "confluence_create_page", "confluence_update_page", "confluence_add_comment",
    "jira_create_issue", "jira_update_issue", "jira_add_comment",
    "jira_transition_issue",
)
_read(
    "atlassian",
    "confluence_search", "confluence_list_spaces", "confluence_get_page",
    "confluence_get_page_by_title", "jira_search_issues", "jira_get_issue",
    "jira_list_projects", "jira_get_transitions",
)

_read(
    "microsoft-teams",
    "list_teams", "list_channels", "get_channel_info", "list_channel_members",
    "get_channel_messages", "list_users", "get_user_info",
)

_cat("microsoft-onedrive", "work_tools_write", "create_folder", "upload_file")
_read(
    "microsoft-onedrive",
    "list_drive_items", "get_drive_item", "search_files", "download_file",
    "list_sites", "get_site", "list_site_drives", "list_site_items",
    "search_site_files", "get_auth_status",
)

_read(
    "edgar-sec",
    "search_filings", "get_company_submissions", "search_company_by_ticker",
    "get_company_facts", "get_filing_document", "get_xbrl_frames",
)

_cat("blender", "shell", "execute_blender_code")
_read(
    "blender",
    "ping", "get_addon_info", "get_scene_info", "get_object_info",
    "get_viewport_screenshot",
)

_cat("macos-native", "desktop_control",
     "open_app", "activate_app", "press_control", "type_into_control",
     "click", "double_click", "right_click", "scroll", "click_app_dom",
     "click_text", "click_at", "double_click_at", "right_click_at",
     "scroll_at", "type_text", "hotkey", "spotlight_search", "batch_actions",
     "launch_app")
_read(
    "macos-native",
    "list_apps", "get_screen_controls", "wait_for_controls",
    "get_control_value", "read_app_dom", "read_screen", "find_text_on_screen",
)

_cat(
    "playwright-mcp", "browser_automation",
    "browser_navigate", "browser_click", "browser_type", "browser_fill_form",
    "browser_tabs", "browser_go_back", "browser_go_forward",
    "browser_select_option", "browser_hover", "browser_drag",
    "browser_press_key", "browser_close", "browser_file_upload",
)
_read(
    "playwright-mcp",
    "browser_snapshot", "browser_take_screenshot", "browser_console_messages",
    "browser_network_requests", "browser_wait",
)

_cat("claude-eval-hook", "pim_write", "save_eval_state", "save_report")
_read(
    "claude-eval-hook",
    "list_platforms", "list_projects", "list_sessions", "check_session_activity",
    "wait_for_activity", "get_hook_status", "load_eval_state",
    "parse_session_turns", "get_session_summary", "get_cowork_audit_info",
)
_cat("openclaw-eval-hook", "pim_write", "save_eval_state", "save_report")
_read(
    "openclaw-eval-hook",
    "list_agents", "list_sessions", "check_session_activity",
    "wait_for_activity", "parse_session_turns", "get_session_summary",
    "get_watcher_status", "load_eval_state",
)
_read(
    "agent-eval-service",
    "list_evaluators", "evaluate", "evaluate_trajectory", "batch_evaluate",
)

# Tools that are not MCP servers. None = reviewed, not gated.
_BARE_TOOLS: dict[str, str | None] = {
    "execute": "shell",
    "write_file": "workspace_write",
    "edit_file": "workspace_write",
    "edit": "workspace_write",
    "create_schedule": "persistence",
    "update_schedule": "persistence",
    "delete_schedule": "persistence",
    "run_schedule_now": "persistence",
    "stop_schedule": "persistence",
    "create_trigger": "persistence",
    "update_trigger": "persistence",
    "delete_trigger": "persistence",
    "run_trigger_now": "persistence",
    "stop_trigger": "persistence",
    "spawn_followup_session": "persistence",
    "create_mcp_server": "mcp_authoring",
    "delete_mcp_server": "mcp_authoring",
    "connect_mcp_server": "mcp_authoring",
    "register_external_mcp_server": "mcp_authoring",
    "toggle_mcp_server": "settings_write",
    "update_orchestrator_settings": "settings_write",
    "update_generation_params": "settings_write",
    "update_memory_settings": "settings_write",
    "update_activity_settings": "settings_write",
    "switch_model_provider": "settings_write",
    "update_exo_cluster_params": "settings_write",
    "toggle_ambient_scheduling": "settings_write",
    "engage_privacy_lock": "settings_write",
    "disengage_privacy_lock": "settings_write",
    "exo_start": "settings_write",
    "exo_stop": "settings_write",
    "exo_load_model": "settings_write",
    "omlx_start": "settings_write",
    "omlx_stop": "settings_write",
    "create_agent_config": "settings_write",
    "update_agent_config": "settings_write",
    "create_skill": "settings_write",
    "update_skill": "settings_write",
    "start_distill_train": "settings_write",
    "use_distilled_model": "persistence",
    "clear_distilled_model": "persistence",
}

# Reading these is what makes later outbound actions ask-each-time.
INGEST_TOOLS = frozenset({
    "browser_snapshot", "browser_navigate", "browser_click",
    "get_message", "search_messages", "read_messages", "list_messages",
    "get_channel_history", "get_thread_replies", "get_channel_messages",
    "web_research", "doc_research", "semantic_search",
    "query_mail_store", "query_outlook_store",
})

# Advisory only. The execute sandbox is the control for shell; these
# labels raise the card and, when they imply another category, escalate.
_EXECUTE_ESCALATIONS: tuple[tuple[str, re.Pattern[str], str | None], ...] = (
    ("osascript", re.compile(r"\bosascript\b"), "applescript"),
    ("shortcuts", re.compile(r"\bshortcuts\b"), "applescript"),
    ("screencapture", re.compile(r"\bscreencapture\b"), "desktop_control"),
    ("sudo", re.compile(r"\bsudo\b"), None),
    ("launchctl", re.compile(r"\blaunchctl\b"), "persistence"),
    ("crontab", re.compile(r"\bcrontab\b"), "persistence"),
    ("security-find", re.compile(r"\bsecurity\s+find-(?:generic|internet)-password\b"), None),
    ("messages-db", re.compile(r"chat\.db|Messages/chat"), "message_send"),
    ("mail-db", re.compile(r"Mail/Envelope Index|Library/Mail"), "email_send"),
    ("local-api", re.compile(r"127\.0\.0\.1:18081|localhost:18081"), "settings_write"),
    ("defaults-write", re.compile(r"\bdefaults\s+write\b"), None),
)

_OSA_ESCALATIONS: tuple[tuple[str, re.Pattern[str], str | None], ...] = (
    ("do-shell-script", re.compile(r"\bdo\s+shell\s+script\b", re.I), "shell"),
    ("admin-privileges", re.compile(r"\bwith\s+administrator\s+privileges\b", re.I), "shell"),
    ("keystroke", re.compile(r"\bkeystroke\b|\bkey\s+code\b", re.I), "desktop_control"),
    ("click", re.compile(r"\bclick\b", re.I), "desktop_control"),
)


@dataclass(frozen=True)
class Classification:
    """One tool call after the catalog and the content screen."""

    category: str | None
    also: tuple[str, ...]
    labels: tuple[str, ...]
    server_id: str | None
    reviewed: bool

    @property
    def categories(self) -> tuple[str, ...]:
        if self.category is None:
            return self.also
        return (self.category, *self.also)


def server_tools() -> dict[tuple[str, str], str | None]:
    """Copy of the static map. Tests use this to check completeness."""
    return dict(_SERVER_TOOLS)


def bare_tools() -> dict[str, str | None]:
    return dict(_BARE_TOOLS)


def _screen(text: str, rules: tuple[tuple[str, re.Pattern[str], str | None], ...]) -> tuple[list[str], list[str]]:
    labels: list[str] = []
    extra: list[str] = []
    for label, pattern, category in rules:
        if pattern.search(text):
            labels.append(label)
            if category and category not in extra:
                extra.append(category)
    return labels, extra


def classify_tool(
    name: str,
    *,
    server_id: str | None = None,
    args: dict | None = None,
) -> Classification:
    """Classify one call. ``category is None`` and ``reviewed`` means not gated."""
    args = args or {}
    key = (server_id or "", name)
    reviewed = False
    category: str | None = None

    if server_id and (server_id, name) in _SERVER_TOOLS:
        category = _SERVER_TOOLS[(server_id, name)]
        reviewed = True
    elif not server_id and name in _BARE_TOOLS:
        category = _BARE_TOOLS[name]
        reviewed = True
    elif server_id in KNOWN_SERVER_IDS:
        category = "unclassified"
        reviewed = False
    elif server_id:
        # Generated or third-party MCP.
        category = "unclassified"
        reviewed = False
    else:
        # A bare helper we have not listed. Not a side effect by default.
        return Classification(None, (), (), None, False)

    labels: list[str] = []
    extra: list[str] = []
    if name == "execute" or category == "shell":
        command = args.get("command")
        if isinstance(command, str):
            labels, extra = _screen(command, _EXECUTE_ESCALATIONS)
    if category == "applescript" or name in ("run_osascript", "run_osascript_file"):
        script = args.get("script") or args.get("path") or ""
        if isinstance(script, str):
            more_labels, more_extra = _screen(script, _OSA_ESCALATIONS)
            labels.extend(more_labels)
            for item in more_extra:
                if item not in extra and item != category:
                    extra.append(item)
    if name == "execute_blender_code":
        labels.append("blender-code")

    # Drop an escalation that merely repeats the primary category.
    extra = [item for item in extra if item != category]
    return Classification(category, tuple(extra), tuple(labels), server_id, reviewed)


def balanced_categories() -> dict[str, str]:
    return {
        "shell": "ask",
        "applescript": "ask",
        "desktop_control": "ask",
        "email_send": "ask",
        "mail_read": "ask",
        "message_send": "ask",
        "message_read": "ask",
        "destructive": "ask",
        "persistence": "ask",
        "mcp_authoring": "ask",
        "settings_write": "ask",
        "pim_write": "ask",
        "work_tools_write": "ask",
        "browser_automation": "allow",
        "workspace_write": "allow",
        "unclassified": "ask",
    }


def locked_down_categories() -> dict[str, str]:
    cats = {name: "deny" for name in CATEGORIES}
    cats["shell"] = "ask"
    cats["browser_automation"] = "ask"
    cats["workspace_write"] = "ask"
    return cats


def trusting_categories() -> dict[str, str]:
    cats = {name: "allow" for name in CATEGORIES}
    # Irreversible or outward-facing stays a prompt even in Trusting.
    for name in (
        "applescript", "email_send", "mail_read", "message_send", "message_read",
        "destructive", "persistence", "unclassified",
    ):
        cats[name] = "ask"
    return cats


PRESETS: dict[str, dict[str, str]] = {
    "locked_down": {},  # filled below, after the functions exist
    "balanced": {},
    "trusting": {},
}
PRESETS["locked_down"] = locked_down_categories()
PRESETS["balanced"] = balanced_categories()
PRESETS["trusting"] = trusting_categories()


def default_rate_limits() -> dict[str, int]:
    """Per-session caps. 0 means no cap."""
    return {
        "email_send": 5,
        "message_send": 10,
        "destructive": 10,
    }


def describe_action(name: str, args: dict | None, category: str | None) -> str:
    """A short, server-built summary. Never the model's own prose."""
    args = args or {}

    def _pick(*keys: str) -> str:
        for key in keys:
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    if name == "execute":
        return _pick("command") or "(empty command)"
    if name in ("run_osascript", "run_osascript_file"):
        return _pick("script", "path") or "(empty script)"
    if category == "email_send":
        to = _pick("to", "recipient", "recipients")
        subject = _pick("subject")
        body = _pick("content", "body", "message")
        parts = [p for p in (f"To: {to}" if to else "", f"Subject: {subject}" if subject else "", body[:400]) if p]
        return "\n".join(parts) or name
    if category == "message_send":
        target = _pick("channel_id", "channel", "to", "recipient", "buddy", "chat_id")
        text = _pick("text", "content", "message", "body")
        head = f"To: {target}\n" if target else ""
        return (head + text[:400]).strip() or name
    if category in (
        "destructive", "pim_write", "work_tools_write", "persistence",
        "settings_write", "mcp_authoring", "mail_read", "message_read",
        "workspace_write",
    ):
        bits = []
        for key, value in list(args.items())[:8]:
            if isinstance(value, str) and len(value) > 180:
                value = value[:180] + "…"
            bits.append(f"{key}={value}")
        return "\n".join(bits) or name
    if name == "execute_blender_code":
        return _pick("code")[:800] or "(empty code)"
    text = _pick("command", "script", "text", "content", "url")
    return text[:500] or name
