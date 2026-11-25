"""Step accounting for the Otto-native track: which of Otto's macOS desktop tools use a step.

Stdlib only: the bridge (MacArena's env) and the guest tool server (inside the VM) both import it.

The split is Otto's own: ``backend/action_policy/catalog.py`` files the macos-native tools under
``desktop_control`` (they act) or as reads. Reads are free here, like the free screenshot in the standard
track, and capped by the bridge. Every acting tool call is one step; ``batch_actions`` is one step per acting
call inside it, so batching cannot stretch the budget. ``tests/test_macarena_native.py`` checks these sets
against the real toolkit and the action-policy catalog.
"""

from __future__ import annotations

from typing import Callable, Iterable

OBSERVATION = frozenset({
    "list_apps", "get_screen_controls", "wait_for_controls", "get_control_value",
    "read_app_dom", "read_screen", "find_text_on_screen",
    "capture_app_screenshot",                       # vision variant only; a read
    # Structured reads (macos_structured_tools). menu_search opens and closes menus to read them; it chooses
    # nothing.
    "observe", "menu_search", "read_dialog", "read_table", "get_control_values", "wait_until", "app_guide",
})

ACTION = frozenset({
    "open_app", "activate_app", "launch_app", "press_control", "type_into_control",
    "click", "double_click", "right_click", "scroll", "click_app_dom", "click_text",
    "click_at", "double_click_at", "right_click_at", "scroll_at", "type_text", "hotkey",
    "spotlight_search", "batch_actions", "menu_select", "fill_dialog",
    "drag", "set_text_at", "write_cells", "select_text",
})

# Otto's AppleScript tools (the macos-osascript server's VM set), split the same way by the action-policy
# catalog: ``run_osascript`` (applescript) and ``dump_ax_tree`` (desktop_control: it activates the app) act;
# ``inspect_app_dictionary`` and ``query_mail_store`` only read.
OSASCRIPT_OBSERVATION = frozenset({"inspect_app_dictionary", "query_mail_store"})
OSASCRIPT_ACTION = frozenset({"run_osascript", "dump_ax_tree"})
OSASCRIPT_TOOLS = OSASCRIPT_OBSERVATION | OSASCRIPT_ACTION

OBSERVATION = OBSERVATION | OSASCRIPT_OBSERVATION
ACTION = ACTION | OSASCRIPT_ACTION
ALL_TOOLS = OBSERVATION | ACTION

# Acting tools whose result is only a one-line confirmation ("Pressed control #8"). Runs showed the next call after
# an action was a read 73% of the time, so with ``observe_after`` the guest adds the frontmost app's controls to the
# result. Left out: tools that already return the screen (launch_app, open_app), AppleScript, and batch_actions
# (it can include its own reads).
OBSERVE_AFTER = frozenset({
    "activate_app", "press_control", "type_into_control", "click", "double_click", "right_click", "scroll",
    "click_app_dom", "click_text", "click_at", "double_click_at", "right_click_at", "scroll_at", "type_text",
    "hotkey", "spotlight_search", "menu_select", "fill_dialog", "drag", "set_text_at", "select_text",
})
OBSERVE_AFTER_MAX_CHARS = 8000


def succeeded(res: dict) -> bool:
    """Did a tool call work? False when the call failed, and when an AppleScript tool ran but its script failed (its
    result is the server's ``{"ok": false, ...}`` dict)."""
    if not res.get("ok"):
        return False
    result = res.get("result")
    return not (isinstance(result, dict) and result.get("ok") is False)


def cost(tool: str, args: dict, parse_batch: Callable[[str], Iterable[tuple[str, dict]]] | None = None) -> int:
    """Steps this call uses. 0 for a read, 1 for an action, and for ``batch_actions`` one per acting call
    inside it (at least 1). ``parse_batch`` is the toolkit's own parser; without it a batch costs 1, which is
    why only the guest, which has the parser, is asked for a batch's cost."""
    if tool in OBSERVATION:
        return 0
    if tool == "batch_actions" and parse_batch is not None:
        try:
            calls = list(parse_batch(str(args.get("steps", ""))))
        except Exception:  # noqa: BLE001 — an unparsable batch fails in the tool; it still costs a step
            return 1
        return max(1, sum(1 for name, _ in calls if name not in OBSERVATION))
    return 1
