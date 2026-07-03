"""Built-in managed trigger catalog.

These triggers are seeded into the app-data triggers directory on startup and
cannot be deleted via the API (``builtin=True``).  Users can enable / disable
them and customise their prompt and agent, but the trigger type and core
parameters are fixed by this catalog.

All triggers are **disabled by default** so nothing fires out of the box —
the user explicitly opts in to each one.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Catalog definition
# ---------------------------------------------------------------------------

# Each entry must satisfy TriggerSpec field requirements.  The `builtin` flag
# is injected automatically by `seed_managed_triggers` — do not set it here.
# All triggers start disabled (enabled=False).

MANAGED_CATALOG: list[dict[str, Any]] = [
    # ── Downloads watcher ────────────────────────────────────────────────────
    {
        "id": "new-download",
        "type": "fileos",
        "path": "~/Downloads",
        "watch": "new_files",
        "glob": "*",
        "poll_seconds": 30,
        "enabled": False,
        "prompt": (
            "New file(s) arrived in ~/Downloads. "
            "Look at the new paths in the event payload and give a one-line summary "
            "of what was downloaded (name, type, likely purpose). "
            "If it's a PDF, extract the title if you can."
        ),
    },
    # ── Screenshots ──────────────────────────────────────────────────────────
    {
        "id": "new-screenshot",
        "type": "fileos",
        "path": "~/Desktop",
        "watch": "new_files",
        "glob": "Screenshot*",
        "poll_seconds": 15,
        "enabled": False,
        "prompt": (
            "A new screenshot was saved to the Desktop. "
            "Read the image and describe what is visible. "
            "Note any text, UI elements, or errors shown."
        ),
    },
    # ── Mail unread count ────────────────────────────────────────────────────
    {
        "id": "mail-unread",
        "type": "macostool",
        "language": "AppleScript",
        "script": (
            'tell application "Mail"\n'
            "    set n to unread count of inbox\n"
            '    return "unread: " & n\n'
            "end tell"
        ),
        "poll_seconds": 60,
        "enabled": False,
        "prompt": (
            "The unread email count in Mail changed. "
            "The new count is in the event payload stdout field. "
            "If the count increased, note that new mail arrived and suggest checking it."
        ),
    },
    # ── Calendar upcoming ────────────────────────────────────────────────────
    {
        "id": "calendar-upcoming",
        "type": "macostool",
        "language": "AppleScript",
        "script": (
            'tell application "Calendar"\n'
            "    set now to current date\n"
            "    set cutoff to now + 30 * minutes\n"
            '    set result to ""\n'
            "    repeat with c in every calendar\n"
            "        try\n"
            "            repeat with e in (every event of c whose start date >= now and start date <= cutoff)\n"
            '                set result to result & (summary of e) & " @ " & ((start date of e) as string) & "\\n"\n'
            "            end repeat\n"
            "        end try\n"
            "    end repeat\n"
            '    if result is "" then return "none"\n'
            "    return result\n"
            "end tell"
        ),
        "poll_seconds": 60,
        "enabled": False,
        "prompt": (
            "A Calendar event is starting within the next 30 minutes. "
            "The event title and time are in the event payload stdout field. "
            "Prepare a brief one-paragraph meeting briefing: recap the event name, "
            "time, and any relevant context you can find."
        ),
    },
    # ── Reminders overdue ────────────────────────────────────────────────────
    {
        "id": "reminders-overdue",
        "type": "macostool",
        "language": "AppleScript",
        "script": (
            'tell application "Reminders"\n'
            "    set now to current date\n"
            '    set result to ""\n'
            "    repeat with r in (every reminder whose completed is false)\n"
            "        try\n"
            "            if (due date of r) < now then\n"
            '                set result to result & (name of r) & "\\n"\n'
            "            end if\n"
            "        end try\n"
            "    end repeat\n"
            '    if result is "" then return "none"\n'
            "    return result\n"
            "end tell"
        ),
        "poll_seconds": 300,
        "enabled": False,
        "prompt": (
            "There are overdue Reminders. "
            "The list of overdue items is in the event payload stdout field. "
            "Triage the list: group by urgency, suggest which to do now vs defer."
        ),
    },
    # ── Zoom meeting start / stop ────────────────────────────────────────────
    {
        "id": "zoom-meeting",
        "type": "shell",
        "command": "pgrep -x zoom.us >/dev/null 2>&1 && echo running || echo stopped",
        "shell_mode": "stdout_change",
        "poll_seconds": 15,
        "enabled": False,
        "prompt": (
            "Zoom's running state just changed (check stdout in the event payload). "
            "If it now says 'running', a meeting just started — prepare a blank meeting-notes "
            "template with today's date. "
            "If it says 'stopped', the meeting ended — note the time."
        ),
    },
    # ── Slack running ────────────────────────────────────────────────────────
    {
        "id": "slack-active",
        "type": "shell",
        "command": "pgrep -x Slack >/dev/null 2>&1 && echo running || echo stopped",
        "shell_mode": "stdout_change",
        "poll_seconds": 30,
        "enabled": False,
        "prompt": (
            "Slack's running state changed (check stdout in the event payload). "
            "Update context: note whether async communication is now available or not."
        ),
    },
    # ── Battery low ─────────────────────────────────────────────────────────
    {
        "id": "battery-low",
        "type": "shell",
        "command": "pmset -g batt",
        "shell_mode": "regex",
        "match": r"([1-9]|1[0-9])%;",   # 1 %–19 % — rising edge only
        "poll_seconds": 120,
        "enabled": False,
        "prompt": (
            "Battery is below 20%. "
            "The pmset output is in the event payload stdout field. "
            "Remind the user to plug in and list any long-running background tasks "
            "that should be paused or saved."
        ),
    },
    # ── iCloud Drive changed ─────────────────────────────────────────────────
    {
        "id": "icloud-changed",
        "type": "fileos",
        "path": "~/Library/Mobile Documents/com~apple~CloudDocs",
        "watch": "new_files",
        "glob": "*",
        "poll_seconds": 60,
        "enabled": False,
        "prompt": (
            "New file(s) synced into iCloud Drive. "
            "List the filenames and suggest whether any need attention."
        ),
    },
    # ── Frontmost app changed ────────────────────────────────────────────────
    {
        "id": "app-switch",
        "type": "macostool",
        "language": "AppleScript",
        "script": (
            "tell application \"System Events\"\n"
            "    set frontApp to name of first application process whose frontmost is true\n"
            "    return frontApp\n"
            "end tell"
        ),
        "poll_seconds": 30,
        "enabled": False,
        "prompt": (
            "The frontmost macOS application just changed. "
            "The new app name is in the event payload stdout field. "
            "Log the app switch with a timestamp and note any relevant context "
            "(e.g. switched to Xcode → coding session, switched to Slack → comms break)."
        ),
    },
]


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def seed_managed_triggers() -> None:
    """Write any missing managed triggers to disk.

    Idempotent — skips triggers that already exist on disk so user
    customisations (prompt, agent, poll_seconds, enabled state) survive
    restarts.  The ``builtin`` flag is always (re-)written so it can't be
    cleared by a manual file edit.
    """
    from backend.trigger_manager import load_trigger, save_trigger
    from backend.schemas import TriggerSpec

    seeded = 0
    for entry in MANAGED_CATALOG:
        trigger_id = entry["id"]
        existing = load_trigger(trigger_id)
        if existing is None:
            spec = TriggerSpec(
                builtin=True,
                **{k: v for k, v in entry.items()},
            )
            save_trigger(spec)
            seeded += 1
        elif not existing.builtin:
            # A user-created trigger has the same id — stamp it as builtin so
            # the delete guard covers it going forward.
            existing.builtin = True
            save_trigger(existing)

    if seeded:
        logger.info("Seeded %d managed trigger(s)", seeded)
