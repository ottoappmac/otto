"""Structured agent actions -> the command strings MacArena's VM server runs.

The agent never sends code. It sends a small dict ({"type": "click", "x": 10, "y": 20}), the bridge
validates it here, and only a command built by this module reaches ``env.step``. Text is quoted with
``repr`` so it cannot break out of the string literal. Anything not listed here is refused, so the agent
cannot reach a shell, the file system or AppleScript in the guest (those would let it change the state
the evaluators read without using the GUI).

Pure Python, no third-party imports: the bridge runs in MacArena's own environment.
"""

from __future__ import annotations

import re
from typing import Any

MAX_TEXT = 2000
MAX_SCROLL = 50
MAX_WAIT_S = 10.0

# pyautogui key names the agent may use (a subset of MacArena's KEYBOARD_KEYS: no media/IME/launch keys).
_NAMED_KEYS = frozenset({
    "enter", "return", "tab", "esc", "escape", "space", "backspace", "delete", "del",
    "up", "down", "left", "right", "home", "end", "pageup", "pagedown",
    "command", "option", "ctrl", "alt", "shift", "fn", "capslock",
    *(f"f{i}" for i in range(1, 13)),
})
_SINGLE = re.compile(r"^[a-z0-9`\-=\[\]\\;',./]$")
_BUTTONS = frozenset({"left", "right", "middle"})

ACTION_TYPES = frozenset({
    "click", "double_click", "right_click", "move_to", "drag_to", "scroll",
    "type", "press", "hotkey", "wait", "done", "fail",
})
TERMINAL = {"done": "DONE", "fail": "FAIL", "wait": "WAIT"}


class ActionError(ValueError):
    """The action is malformed or not allowed. The message is shown to the agent."""


def _num(action: dict, key: str, lo: float, hi: float) -> float:
    if key not in action:
        raise ActionError(f"{action.get('type')}: missing '{key}'")
    try:
        v = float(action[key])
    except (TypeError, ValueError):
        raise ActionError(f"{action.get('type')}: '{key}' must be a number") from None
    if v != v or not (lo <= v <= hi):          # NaN fails both
        raise ActionError(f"{action.get('type')}: '{key}'={action[key]} is outside the screen (0..{hi:g})")
    return v


def _xy(action: dict, size: tuple[int, int], scale: tuple[float, float], xk="x", yk="y") -> tuple[int, int]:
    w, h = size
    x = _num(action, xk, 0, w - 1)
    y = _num(action, yk, 0, h - 1)
    return round(x * scale[0]), round(y * scale[1])


def key_ok(key: Any) -> str:
    k = str(key).strip().lower()
    if k in _NAMED_KEYS or _SINGLE.match(k):
        return k
    raise ActionError(f"unknown key {key!r}. Use a single character or one of: {', '.join(sorted(_NAMED_KEYS))}")


def to_command(action: dict, size: tuple[int, int], scale: tuple[float, float] = (1.0, 1.0)) -> str:
    """The string for ``env.step``: 'WAIT' / 'DONE' / 'FAIL' or a pyautogui command.

    ``size`` is the screenshot size the agent sees (its coordinate space). ``scale`` converts those pixels
    to the VM's pyautogui coordinates (1.0 unless the screenshot is larger than the logical screen).
    """
    if not isinstance(action, dict):
        raise ActionError("action must be an object")
    t = action.get("type")
    if t not in ACTION_TYPES:
        raise ActionError(f"unknown action type {t!r}. Allowed: {', '.join(sorted(ACTION_TYPES))}")
    if t in TERMINAL:
        if t == "wait":
            _num({"type": t, "seconds": action.get("seconds", 1)}, "seconds", 0, MAX_WAIT_S)
        return TERMINAL[t]

    if t in ("click", "double_click", "right_click", "move_to"):
        x, y = _xy(action, size, scale)
        if t == "move_to":
            return f"pyautogui.moveTo({x}, {y})"
        if t == "double_click":
            return f"pyautogui.doubleClick({x}, {y})"
        button = "right" if t == "right_click" else str(action.get("button", "left")).lower()
        if button not in _BUTTONS:
            raise ActionError(f"button must be one of {sorted(_BUTTONS)}")
        return f"pyautogui.click({x}, {y}, button={button!r})"

    if t == "drag_to":
        x1, y1 = _xy(action, size, scale, "from_x", "from_y")
        x2, y2 = _xy(action, size, scale, "x", "y")
        return f"pyautogui.moveTo({x1}, {y1}); pyautogui.dragTo({x2}, {y2}, duration=0.6, button='left')"

    if t == "scroll":
        amount = int(_num(action, "amount", -MAX_SCROLL, MAX_SCROLL))      # + up, - down
        if "x" in action and "y" in action:
            x, y = _xy(action, size, scale)
            return f"pyautogui.moveTo({x}, {y}); pyautogui.scroll({amount})"
        return f"pyautogui.scroll({amount})"

    if t == "type":
        text = action.get("text")
        if not isinstance(text, str) or not text:
            raise ActionError("type: 'text' must be a non-empty string")
        if len(text) > MAX_TEXT:
            raise ActionError(f"type: text is longer than {MAX_TEXT} characters")
        # Exactly `pyautogui.typewrite('...')`: MacArena's env rewrites '<' (pyautogui types '>') only for
        # a call of this shape, and its own agents emit the same. `write(...)` or an `interval=` would skip that fix.
        return f"pyautogui.typewrite({text!r})"

    if t == "press":
        return f"pyautogui.press({key_ok(action.get('key'))!r})"

    # hotkey
    keys = action.get("keys")
    if not isinstance(keys, list) or not 1 <= len(keys) <= 4:
        raise ActionError("hotkey: 'keys' must be a list of 1 to 4 keys, e.g. [\"command\", \"c\"]")
    return "pyautogui.hotkey(" + ", ".join(repr(key_ok(k)) for k in keys) + ")"
