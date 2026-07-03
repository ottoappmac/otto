"""Standalone LangChain tools for macOS desktop interaction.

All macOS API calls are self-contained — this module does NOT import anything
from ``macos_navigation.py``.

Layers
------
ApplicationServices  Accessibility API — inspect the UI element tree, trigger
                     actions, read/set values.  Works for native macOS apps
                     (Calculator, Finder, Safari, Mail, …) and Electron apps
                     (Slack, VS Code, Cursor) via AXManualAccessibility.
AppKit / NSWorkspace Launch apps, list running processes, bring windows forward.
pyautogui            Mouse clicks + keyboard fallback for any app whose AX tree
                     is insufficient (coordinate-based interaction).
subprocess           ``open -a <AppName>`` — the correct macOS command to launch
                     an application by name.

How macOS UI interaction works
------------------------------
1. Every app exposes a tree of ``AXUIElement`` objects.
2. Each element has attributes (role, title, value, position, size, …) and
   supports actions (AXPress, AXFocus, …).
3. ``AXUIElementCreateApplication(pid)`` returns the root element for a process.
4. We walk ``kAXChildrenAttribute`` of the root to collect windows, the menu
   bar, and any other top-level elements.
5. For Electron/Chromium apps we set ``AXManualAccessibility = True`` on first
   contact, which activates Chromium's accessibility bridge and exposes the
   full web DOM as AX elements (sidebar items, buttons, text fields, etc.).
6. Every interactive element is assigned a stable integer index stored in
   ``_element_registry`` so subsequent tools can act by index.

Tool call order
---------------
Launch:       open_app  →  activate_app  →  wait_for_controls
Interact:     get_screen_controls  →  press_control / type_into_control / get_control_value
Fallback:     click / double_click / hotkey / type_text  (index-based mouse click)
"""
from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import ctypes
import io
import logging
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import NamedTuple

import pyautogui
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Annotated

from utilities.environment import Environment

from ApplicationServices import (
    AXUIElementCreateApplication,
    AXUIElementCopyAttributeValue,
    AXUIElementCopyMultipleAttributeValues,
    AXUIElementGetPid,
    AXUIElementPerformAction,
    AXUIElementSetAttributeValue,
    kAXChildrenAttribute,
    kAXDescriptionAttribute,
    kAXEnabledAttribute,
    kAXFocusedAttribute,
    kAXMainWindowAttribute,
    kAXMinimizedAttribute,
    kAXPositionAttribute,
    kAXPressAction,
    kAXRoleAttribute,
    kAXSizeAttribute,
    kAXTitleAttribute,
    kAXValueAttribute,
    kAXWindowsAttribute,
)
from AppKit import (
    NSApplicationActivateIgnoringOtherApps,
    NSBitmapImageFileTypePNG,
    NSBitmapImageRep,
    NSScreen,
    NSWorkspace,
)

# On-device OCR (macOS Vision framework) — the read-and-act fallback for apps
# whose Accessibility API is switched off (e.g. Slack). Imported lazily so the
# module still loads if pyobjc-framework-Vision is missing; tools degrade to a
# clear "OCR unavailable" message instead of crashing.
try:
    from Quartz import (
        CGRectNull,
        CGWindowListCopyWindowInfo,
        CGWindowListCreateImage,
        kCGNullWindowID,
        kCGWindowImageBoundsIgnoreFraming,
        kCGWindowListExcludeDesktopElements,
        kCGWindowListOptionIncludingWindow,
        kCGWindowListOptionOnScreenOnly,
    )
    try:
        from Quartz import CGPreflightScreenCaptureAccess
    except Exception:  # pragma: no cover - older macOS without the symbol
        CGPreflightScreenCaptureAccess = None
    from Vision import (
        VNImageRequestHandler,
        VNRecognizeTextRequest,
        VNRequestTextRecognitionLevelAccurate,
    )
    _VISION_AVAILABLE = True
except Exception:  # pragma: no cover - depends on host frameworks
    _VISION_AVAILABLE = False

# Process-targeted keyboard synthesis (CGEventPostToPid).  Unlike pyautogui's
# global keyboard, these events are posted to a *specific* process and do NOT
# depend on or move the system keyboard focus — so multiple agents can type
# into different apps in parallel without a shared-focus race, and typing works
# even when the target window is not frontmost.  Imported lazily so the module
# still loads if the symbols are unavailable; callers fall back to pyautogui.
try:
    from Quartz import (
        CGEventCreateKeyboardEvent,
        CGEventKeyboardSetUnicodeString,
        CGEventPostToPid,
        CGEventSourceCreate,
        kCGEventSourceStateHIDSystemState,
    )
    _CGEVENT_AVAILABLE = True
except Exception:  # pragma: no cover - depends on host frameworks
    _CGEVENT_AVAILABLE = False

_kVK_Return = 0x24  # virtual keycode for the Return key

_logger = logging.getLogger(__name__)

# Default ceiling (seconds) for waiting on a Chromium/Electron app to build
# its AX tree after AXManualAccessibility is flipped.  Electron exposes its
# DOM lazily and slowly, and under cross-session AX contention a single
# fixed sleep is often too short — so we poll up to this budget.  Overridable
# per-toolkit via the ``ax_bridge_max_wait`` constructor arg.
_DEFAULT_AX_BRIDGE_MAX_WAIT = 6.0

# ---------------------------------------------------------------------------
# Cross-session AX-bridge coordination (process-wide, shared by every toolkit
# instance / agent session in this process).
#
# Two desktop agents share one host and one Accessibility subsystem.  The
# Chromium/Electron bridge (AXManualAccessibility) is serviced on the target
# app's main thread, so two sessions flipping it at the same instant can
# collide.  ``_AX_BRIDGE_LOCK`` serializes ONLY the flip step (a single IPC
# call), not the tree walk, so native-app read parallelism is unaffected.
# ---------------------------------------------------------------------------
_AX_BRIDGE_LOCK = threading.Lock()

# AXError codes worth distinguishing when an app exposes no tree.  The full
# list lives in <HIServices/AXError.h>; we only special-case the ones that
# change what we tell the agent.
_kAXErrorAPIDisabled = -25211      # app's accessibility interface is OFF
_kAXErrorCannotComplete = -25204   # app busy / not responding to AX right now

# ---------------------------------------------------------------------------
# Low-level AX helpers (stateless — shared across all toolkit instances)
# ---------------------------------------------------------------------------
_hiserv = ctypes.CDLL(
    "/System/Library/Frameworks/ApplicationServices.framework"
    "/Versions/A/Frameworks/HIServices.framework/HIServices"
)
_hiserv.AXValueGetValue.restype = ctypes.c_bool
_hiserv.AXValueGetType.restype = ctypes.c_uint32
_hiserv.AXValueGetType.argtypes = [ctypes.c_void_p]

_kAXValueAXErrorType = 5


class _CGPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


class _CGSize(ctypes.Structure):
    _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]


def _to_void_p(obj) -> ctypes.c_void_p:
    if hasattr(obj, "__c_void_p__"):
        return obj.__c_void_p__()
    import objc
    return ctypes.c_void_p(objc.pyobjc_id(obj))


def _ax_point(val) -> tuple[int, int]:
    pt = _CGPoint()
    if _hiserv.AXValueGetValue(_to_void_p(val), 1, ctypes.byref(pt)):
        return int(pt.x), int(pt.y)
    return 0, 0


def _ax_size(val) -> tuple[int, int]:
    sz = _CGSize()
    if _hiserv.AXValueGetValue(_to_void_p(val), 2, ctypes.byref(sz)):
        return int(sz.width), int(sz.height)
    return 0, 0


def _is_ax_error(val) -> bool:
    """Return True if *val* is an AXValueRef wrapping an error sentinel.

    ``AXUIElementCopyMultipleAttributeValues`` returns AXValueRef objects
    for attributes that don't exist on an element instead of None.
    Legitimate AXValueRef objects (CGPoint=1, CGSize=2) have type codes
    below 5; error sentinels have type code 5 (kAXValueAXErrorType).
    """
    if type(val).__name__ != "AXValueRef":
        return False
    try:
        return _hiserv.AXValueGetType(_to_void_p(val)) == _kAXValueAXErrorType
    except Exception:
        return False


def _ax_get(element, attr):
    """Read one attribute from an AX element; return None on any error."""
    err, value = AXUIElementCopyAttributeValue(element, attr, None)
    return value if err == 0 else None


def _ax_pid(element) -> int:
    """Return the process id that owns an AX element, or 0 if unknown."""
    try:
        err, pid = AXUIElementGetPid(element, None)
    except Exception:  # pragma: no cover - defensive against pyobjc quirks
        return 0
    return int(pid) if (err == 0 and pid) else 0


def _press_state_token(element) -> str:
    """Cheap, comparable snapshot of the app state around *element*.

    Combines the owning app's main-window title with the element's own value.
    ``press_control`` reads this before and after an ``AXPress`` to detect a
    *silent no-op*: when the action reports success (err=0) yet the token is
    unchanged, the press was almost certainly swallowed. This is the norm for
    Electron/Chromium controls (Slack, Discord, VS Code) whose accessibility
    elements accept ``AXPress`` but never fire the real DOM click handler.

    Returns "" when no usable signal can be read, so callers skip the check
    rather than warn on a false negative.
    """
    parts: list[str] = []
    pid = _ax_pid(element)
    if pid:
        try:
            root = AXUIElementCreateApplication(pid)
            win = _ax_get(root, kAXMainWindowAttribute)
            if win is None:
                windows = _ax_get(root, kAXWindowsAttribute)
                if windows:
                    win = windows[0]
            if win is not None:
                title = _ax_get(win, kAXTitleAttribute)
                if title:
                    parts.append(str(title))
        except Exception:  # pragma: no cover - defensive against pyobjc quirks
            pass
    val = _ax_get(element, kAXValueAttribute)
    if val is not None:
        parts.append(str(val))
    return "\u241f".join(parts)


def _type_to_pid(pid: int, text: str, submit: bool = False) -> bool:
    """Type *text* into process *pid* via CGEventPostToPid — focus-free.

    Posts keyboard events directly to the target process, so they neither
    depend on nor move the global keyboard focus. This is what lets several
    agents drive different apps in parallel without stealing focus from one
    another, and it works even when the target window is not frontmost.

    Each character is sent as its own key-down/key-up pair carrying a Unicode
    string (no virtual-keycode mapping needed), which is the most broadly
    compatible way to inject arbitrary text. Returns True if the events were
    posted, False if the CGEvent APIs are unavailable (caller should fall back
    to pyautogui).
    """
    if not _CGEVENT_AVAILABLE or not pid:
        return False
    src = CGEventSourceCreate(kCGEventSourceStateHIDSystemState)
    for ch in text:
        for key_down in (True, False):
            ev = CGEventCreateKeyboardEvent(src, 0, key_down)
            if ev is None:
                return False
            # len(ch) is the UTF-16 unit count; 1 for BMP chars.
            CGEventKeyboardSetUnicodeString(ev, len(ch), ch)
            CGEventPostToPid(pid, ev)
        time.sleep(0.004)
    if submit:
        for key_down in (True, False):
            ev = CGEventCreateKeyboardEvent(src, _kVK_Return, key_down)
            if ev is not None:
                CGEventPostToPid(pid, ev)
    return True


# ── Electron / Chromium structured read via Chrome DevTools Protocol ────────
#
# Electron apps (Slack, Discord, VS Code, Cursor, Notion, …) frequently switch
# the macOS Accessibility API OFF (kAXErrorAPIDisabled), so the AX tree is
# unreadable and we otherwise fall back to OCR. If such an app was launched with
# ``--remote-debugging-port=<N>`` we can instead read its full accessibility
# tree directly from Chromium over CDP — structured text, no screenshot, no
# vision tokens, and it works while the window is in the background.
_CDP_PORT_RE = re.compile(r"--remote-debugging-port[= ](\d+)")

# Roles worth surfacing as actionable controls in the formatted CDP dump.
_CDP_INTERACTIVE_ROLES = frozenset({
    "button", "link", "textbox", "searchbox", "checkbox", "radio", "switch",
    "menuitem", "menuitemcheckbox", "menuitemradio", "tab", "combobox",
    "option", "slider", "listitem", "treeitem", "spinbutton",
})


def _find_cdp_port(pid: int) -> int:
    """Return the ``--remote-debugging-port`` an Electron process is using, or 0.

    Scans the process table for the given pid's command line. A non-zero result
    means we can read the app's DOM/AX tree over CDP instead of OCR.
    """
    if not pid:
        return 0
    try:
        out = subprocess.run(
            ["ps", "-Ao", "pid=,command="],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return 0
    for line in out.splitlines():
        line = line.strip()
        head, _, rest = line.partition(" ")
        if not head.isdigit() or int(head) != pid:
            continue
        m = _CDP_PORT_RE.search(rest)
        if m:
            return int(m.group(1))
    return 0


def _cdp_format_ax_tree(nodes: list[dict], max_nodes: int = 400) -> str:
    """Format CDP ``Accessibility.getFullAXTree`` nodes into readable text.

    Interactive controls are tagged with their role (e.g. ``[button] Send``);
    static/heading text is emitted plain so the model can read message content.
    """
    lines: list[str] = []
    for node in nodes:
        if node.get("ignored"):
            continue
        role = (node.get("role") or {}).get("value") or ""
        name = (node.get("name") or {}).get("value") or ""
        value = (node.get("value") or {}).get("value") or ""
        text = (name or value).strip()
        if not text:
            continue
        if role in _CDP_INTERACTIVE_ROLES:
            entry = f"[{role}] {text}"
            if value and value != name:
                entry += f" = {value!r}"
            lines.append(entry)
        elif role in ("StaticText", "text", "heading", "paragraph", "cell"):
            lines.append(text)
        if len(lines) >= max_nodes:
            lines.append(f"… (truncated at {max_nodes} nodes)")
            break
    return "\n".join(lines)


async def _cdp_read_ax_tree(port: int, timeout: float = 8.0) -> str | None:
    """Read the full accessibility tree from a Chromium target over CDP.

    Connects to ``http://127.0.0.1:<port>``, picks the first page target, and
    issues ``Accessibility.enable`` + ``Accessibility.getFullAXTree``. Returns
    formatted text, or None on any failure (caller falls back to OCR).
    """
    if not port:
        return None
    try:
        import aiohttp
    except Exception:
        return None

    base = f"http://127.0.0.1:{port}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{base}/json/list",
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                targets = await resp.json()
            pages = [
                t for t in targets
                if t.get("type") == "page" and t.get("webSocketDebuggerUrl")
            ]
            if not pages:
                return None
            ws_url = pages[0]["webSocketDebuggerUrl"]
            async with session.ws_connect(
                ws_url, timeout=aiohttp.ClientTimeout(total=timeout),
            ) as ws:
                await ws.send_json({"id": 1, "method": "Accessibility.enable"})
                await ws.send_json({"id": 2, "method": "Accessibility.getFullAXTree"})
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    msg = await ws.receive(timeout=timeout)
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    payload = msg.json()
                    if payload.get("id") == 2:
                        nodes = (payload.get("result") or {}).get("nodes")
                        if not nodes:
                            return None
                        return _cdp_format_ax_tree(nodes)
    except Exception as exc:  # pragma: no cover - network/proto variability
        _logger.warning("CDP AX read failed on port %s: %s", port, exc)
        return None
    return None


async def _cdp_click(
    port: int, text: str, role: str = "", timeout: float = 8.0,
) -> str:
    """Click a control in a Chromium target over CDP — no OS focus required.

    Finds the accessibility node whose name matches *text* (case-insensitive
    substring, optionally filtered by *role*), scrolls it into view, then
    dispatches a real ``Input.dispatchMouseEvent`` press/release at its box
    centre. Because CDP delivers input straight to the renderer, this works
    while the window is in the background — i.e. when another app (the host)
    holds the OS keyboard/mouse focus. Returns a human-readable result string.
    """
    if not port:
        return "no CDP port"
    try:
        import aiohttp
    except Exception:
        return "aiohttp is not installed, so CDP input is unavailable"

    base = f"http://127.0.0.1:{port}"
    want = text.strip().lower()
    role_want = role.strip().lower()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{base}/json/list", timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                targets = await resp.json()
            pages = [
                t for t in targets
                if t.get("type") == "page" and t.get("webSocketDebuggerUrl")
            ]
            if not pages:
                return "no CDP page target to click into"
            ws_url = pages[0]["webSocketDebuggerUrl"]
            async with session.ws_connect(
                ws_url, timeout=aiohttp.ClientTimeout(total=timeout),
            ) as ws:
                _msg_id = 0

                async def _cmd(method: str, params: dict | None = None) -> dict:
                    nonlocal _msg_id
                    _msg_id += 1
                    mid = _msg_id
                    await ws.send_json(
                        {"id": mid, "method": method, "params": params or {}}
                    )
                    deadline = time.monotonic() + timeout
                    while time.monotonic() < deadline:
                        msg = await ws.receive(timeout=timeout)
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue
                        payload = msg.json()
                        if payload.get("id") == mid:
                            return payload
                    return {}

                await _cmd("Accessibility.enable")
                await _cmd("DOM.enable")
                tree = await _cmd("Accessibility.getFullAXTree")
                nodes = (tree.get("result") or {}).get("nodes") or []

                match = None
                for node in nodes:
                    if node.get("ignored"):
                        continue
                    if node.get("backendDOMNodeId") is None:
                        continue
                    name = ((node.get("name") or {}).get("value") or "").strip()
                    nrole = ((node.get("role") or {}).get("value") or "").strip().lower()
                    if not name or want not in name.lower():
                        continue
                    if role_want and role_want != nrole:
                        continue
                    match = node
                    # Exact-name + (matching role, if requested) is the best hit.
                    if name.lower() == want and (not role_want or role_want == nrole):
                        break

                if match is None:
                    return f"no CDP node matching {text!r}"

                backend_id = match["backendDOMNodeId"]
                matched_name = (match.get("name") or {}).get("value") or ""
                matched_role = (match.get("role") or {}).get("value") or ""

                resolved = await _cmd(
                    "DOM.resolveNode", {"backendNodeId": backend_id}
                )
                obj_id = ((resolved.get("result") or {}).get("object") or {}).get("objectId")
                if obj_id:
                    await _cmd("Runtime.callFunctionOn", {
                        "objectId": obj_id,
                        "functionDeclaration":
                            "function(){this.scrollIntoView({block:'center',inline:'center'});}",
                    })

                box = await _cmd("DOM.getBoxModel", {"backendNodeId": backend_id})
                model = (box.get("result") or {}).get("model")

                if not model:
                    if not obj_id:
                        return (
                            f"matched {matched_name!r} but could not resolve a "
                            f"clickable box or node"
                        )
                    await _cmd("Runtime.callFunctionOn", {
                        "objectId": obj_id,
                        "functionDeclaration": "function(){this.click();}",
                    })
                    return (
                        f"clicked {matched_name!r} [{matched_role}] via CDP "
                        f"element.click() (focus-free)"
                    )

                quad = model["content"]
                cx = (quad[0] + quad[2] + quad[4] + quad[6]) / 4.0
                cy = (quad[1] + quad[3] + quad[5] + quad[7]) / 4.0
                for ev_type, extra in (
                    ("mouseMoved", {}),
                    ("mousePressed", {"button": "left", "buttons": 1, "clickCount": 1}),
                    ("mouseReleased", {"button": "left", "buttons": 0, "clickCount": 1}),
                ):
                    params = {"type": ev_type, "x": cx, "y": cy}
                    params.update(extra)
                    await _cmd("Input.dispatchMouseEvent", params)
                return (
                    f"clicked {matched_name!r} [{matched_role}] at page "
                    f"({cx:.0f}, {cy:.0f}) via CDP (focus-free)"
                )
    except Exception as exc:  # pragma: no cover - network/proto variability
        _logger.warning("CDP click failed on port %s: %s", port, exc)
        return f"CDP click error: {exc}"


def _logical_screen_size() -> tuple[int, int]:
    frame = NSScreen.mainScreen().frame()
    return int(frame.size.width), int(frame.size.height)


# Shown whenever a pixel capture fails or comes back blank — by far the most
# common cause is the host app lacking the Screen Recording permission, which
# makes screencapture exit non-zero and CGWindowListCreateImage return an empty
# image (so OCR then finds no text).
_SCREEN_RECORDING_HINT = (
    "This usually means the app running this agent does NOT have Screen "
    "Recording permission. Grant it in System Settings → Privacy & Security → "
    "Screen Recording, enable the host app (e.g. Otto/Terminal), then fully "
    "quit and reopen it (the permission only takes effect after a restart)."
)


def _screen_recording_denied() -> bool:
    """Best-effort check that Screen Recording permission is explicitly absent.

    Returns True only when the preflight API is available AND reports no access.
    Returns False when access is granted OR the result is unknown — the API is
    process-scoped and can disagree with the actual capture, so callers should
    treat True as a strong hint, not the sole gate.
    """
    if not _VISION_AVAILABLE or CGPreflightScreenCaptureAccess is None:
        return False
    try:
        return not bool(CGPreflightScreenCaptureAccess())
    except Exception:  # pragma: no cover - defensive
        return False


def _rank_text_matches(results: list[dict], text: str) -> list[dict]:
    """Filter+rank OCR results that contain *text*, best match first.

    Ranks exact match > whole-word match > substring-in-larger-word, then by
    OCR confidence, then by shortest label (tighter click target). Shared by
    ``find_text_on_screen`` and ``click_text`` so they agree on "best".
    """
    needle = text.strip().lower()

    def _score(r: dict) -> tuple[int, float, int]:
        t = r["text"].strip().lower()
        if t == needle:
            quality = 3
        elif re.search(rf"\b{re.escape(needle)}\b", t):
            quality = 2
        elif needle in t:
            quality = 1
        else:
            quality = 0
        return (quality, r["conf"], -len(t))

    matches = [r for r in results if needle in r["text"].lower()]
    matches.sort(key=_score, reverse=True)
    return matches


def _format_ocr_lines(results: list[dict], label: str) -> str:
    """Group OCR results into reading-order text lines for *label*.

    Sorts by a coarse y-band then x so words on the same visual row join into
    one line, top-to-bottom. Shared by both the text and vision ``read_screen``
    variants so they always render identical text.
    """
    if not results:
        return f"No text recognised on screen for {label}."
    ordered = sorted(results, key=lambda r: (round(r["y"] / 12), r["x"]))
    lines: list[str] = []
    current_band: int | None = None
    buf: list[str] = []
    for r in ordered:
        band = round(r["y"] / 12)
        if current_band is None or band == current_band:
            buf.append(r["text"])
        else:
            lines.append(" ".join(buf))
            buf = [r["text"]]
        current_band = band
    if buf:
        lines.append(" ".join(buf))
    return f"OCR text for {label}:\n" + "\n".join(lines)


class _Rect(NamedTuple):
    """Axis-aligned bounding box used for viewport clipping."""
    x: int
    y: int
    w: int
    h: int

    def intersects(self, ox: int, oy: int, ow: int, oh: int) -> bool:
        """Return True if rectangle (ox, oy, ow, oh) overlaps this rect."""
        return (
            ox + ow > self.x
            and ox < self.x + self.w
            and oy + oh > self.y
            and oy < self.y + self.h
        )


_NO_CLIP = _Rect(-99999, -99999, 199998, 199998)


# Attributes fetched in a single IPC round-trip by _ax_get_multi.
_WALK_ATTRS = [
    kAXRoleAttribute,
    kAXPositionAttribute,
    kAXSizeAttribute,
    kAXTitleAttribute,
    kAXDescriptionAttribute,
    kAXValueAttribute,
    kAXEnabledAttribute,
    kAXChildrenAttribute,
]

_INTERACTIVE_ROLES = {
    "AXButton", "AXTextField", "AXTextArea", "AXCheckBox",
    "AXRadioButton", "AXComboBox", "AXSlider", "AXLink",
    "AXMenuItem", "AXMenuBarItem", "AXPopUpButton",
    "AXStaticText", "AXCell", "AXRow", "AXTable",
}

_STRUCTURAL_ROLES = frozenset({
    "AXWindow", "AXToolbar", "AXSplitGroup", "AXScrollArea",
    "AXTabGroup", "AXList", "AXOutline",
    "AXSheet", "AXDrawer", "AXMenuBar", "AXMenu", "AXWebArea",
    "AXGroup",
})

_MAX_STRUCTURAL_DEPTH = 3

_ROLE_CODES: dict[str, str] = {
    "Button": "B", "TextField": "TF", "TextArea": "TA",
    "CheckBox": "CB", "RadioButton": "RB", "ComboBox": "CX",
    "Slider": "SL", "Link": "LN", "MenuItem": "MI",
    "MenuBarItem": "MB", "PopUpButton": "PB", "StaticText": "ST",
    "Cell": "CE", "Row": "RW", "Table": "TB", "Group": "G",
}

# ---------------------------------------------------------------------------
# Agent-facing control interaction rules (constant — not config-dependent)
# ---------------------------------------------------------------------------
CONTROL_INTERACTION_RULES: str = """\
<control_interaction_rules>
  Code │ Role            │ Valid tools
  ─────┼─────────────────┼──────────────────────────────────────────────────
  B    │ Button          │ press_control
  CB   │ CheckBox        │ press_control
  RB   │ RadioButton     │ press_control
  LN   │ Link            │ press_control
  MI   │ MenuItem        │ press_control
  MB   │ MenuBarItem     │ press_control
  PB   │ PopUpButton     │ press_control
  SL   │ Slider          │ click
  G    │ Group           │ click
  TF   │ TextField       │ type_into_control, get_control_value
  TA   │ TextArea        │ type_into_control, get_control_value
  CX   │ ComboBox        │ type_into_control, press_control, get_control_value
  ST   │ StaticText      │ get_control_value
  CE   │ Cell            │ click, get_control_value
  RW   │ Row             │ click
  TB   │ Table           │ get_control_value
       │ Image           │ click
</control_interaction_rules>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Input schemas — Pydantic models used as args_schema for every @tool so that
# the LLM JSON schema explicitly marks all required fields and carries per-field
# descriptions.  Optional arguments (timeouts, scroll amount) retain defaults.
# Constrained types reject empty strings and out-of-range numbers before the
# function body is ever reached.
# ═══════════════════════════════════════════════════════════════════════════════

_AppName   = Annotated[str, Field(min_length=1)]
_ControlIndex = Annotated[int, Field(ge=0)]
_ScrollDir = Annotated[str, Field(min_length=1, pattern=r"^(up|down|left|right)$")]


class _GetScreenControlsInput(BaseModel):
    app_name: _AppName = Field(description="Exact app name as returned by list_apps(). Must not be empty.")
    role: str = Field(default="", description="Optional role filter (e.g. 'TextField', 'Button'). Case-insensitive. Only controls matching this role are returned.")
    label: str = Field(default="", description="Optional label search string. Case-insensitive substring match. Only controls whose label contains this string are returned.")


class _OpenAppInput(BaseModel):
    app_name: _AppName = Field(description="App name, .app bundle path, or URL to open. Must not be empty.")


class _ActivateAppInput(BaseModel):
    app_name: _AppName = Field(description="Exact app name as returned by list_apps(). Must not be empty.")


class _WaitForControlsInput(BaseModel):
    app_name: _AppName = Field(description="Exact app name as returned by list_apps(). Must not be empty.")
    timeout: Annotated[int, Field(ge=1, le=60)] = Field(default=10, description="Seconds to poll (1–60).")


class _LaunchAppInput(BaseModel):
    app_name: _AppName = Field(description="App name, .app bundle path, or URL to launch. Must not be empty.")
    timeout: Annotated[int, Field(ge=1, le=60)] = Field(default=10, description="Seconds to wait for controls (1–60).")


class _PressControlInput(BaseModel):
    index: _ControlIndex = Field(description="Control index from the most recent get_screen_controls() scan.")


class _TypeIntoControlInput(BaseModel):
    index: _ControlIndex = Field(description="Control index from the most recent get_screen_controls() scan.")
    text: str = Field(description="Text to type into the control.")
    submit: bool = Field(
        default=False,
        description=(
            "Press Return/Enter after typing to commit the value in the SAME call. "
            "Set true for search boxes and single-line fields that submit on Enter. "
            "Leave false for multi-line text areas or fields with a separate submit button."
        ),
    )


class _GetControlValueInput(BaseModel):
    index: _ControlIndex = Field(description="Control index from the most recent get_screen_controls() scan.")


class _ClickInput(BaseModel):
    index: _ControlIndex = Field(description="Control index from the most recent get_screen_controls() scan.")


class _DoubleClickInput(BaseModel):
    index: _ControlIndex = Field(description="Control index from the most recent get_screen_controls() scan.")


class _RightClickInput(BaseModel):
    index: _ControlIndex = Field(description="Control index from the most recent get_screen_controls() scan.")


class _ScrollInput(BaseModel):
    index:     _ControlIndex = Field(description="Control index from the most recent get_screen_controls() scan.")
    direction: _ScrollDir    = Field(description="Scroll direction: up | down | left | right.")
    amount:    Annotated[int, Field(gt=0)] = Field(default=300, description="Pixels to scroll (must be > 0).")


class _TypeTextInput(BaseModel):
    text: str = Field(description="Text to type into the currently focused field.")
    submit: bool = Field(
        default=False,
        description=(
            "Press Return/Enter after typing to commit the value in the SAME call. "
            "Set true for search boxes and single-line fields that submit on Enter. "
            "Leave false for multi-line text areas or fields with a separate submit button."
        ),
    )
    app_name: str = Field(
        default="",
        description=(
            "Optional focus guard: the app that must be frontmost before typing. "
            "If set, the app is activated and confirmed frontmost first; if focus "
            "cannot be obtained, nothing is typed. Use for AX-disabled apps (Slack)."
        ),
    )


class _HotkeyInput(BaseModel):
    keys: _AppName = Field(description="Key combo with '+' separators, e.g. 'command+space'. Must not be empty.")


class _SpotlightSearchInput(BaseModel):
    query: _AppName = Field(description="Search query to type into Spotlight. Must not be empty.")


class _ListAppsInput(BaseModel):
    query: _AppName = Field(description="Case-insensitive substring to filter running apps, e.g. 'slack'. Must not be empty.")


class _BatchActionsInput(BaseModel):
    steps: str = Field(
        min_length=1,
        description=(
            "One or more tool calls as plain text, separated by newlines. "
            "Example: press_control(index=5)\\ntype_into_control(index=3, text='hello')"
        )
    )


class _ReadScreenInput(BaseModel):
    app_name: str = Field(
        default="",
        description="App whose window to read via OCR. Empty = full screen.",
    )


class _ReadAppDomInput(BaseModel):
    app_name: _AppName = Field(
        description="Electron/Chromium app to read via CDP (e.g. 'Slack', 'Cursor').",
    )


class _ClickAppDomInput(BaseModel):
    app_name: _AppName = Field(
        description="Electron/Chromium app to click in via CDP (e.g. 'Slack', 'Cursor').",
    )
    text: str = Field(
        description="Visible label of the control to click, as shown by "
        "read_app_dom (case-insensitive substring, e.g. 'DMs', 'Search').",
    )
    role: str = Field(
        default="",
        description="Optional CDP role filter to disambiguate (e.g. 'button', "
        "'tab', 'menuitemradio'). Leave empty to match any role.",
    )


class _FindTextOnScreenInput(BaseModel):
    text: str = Field(description="Text to locate on screen (case-insensitive substring).")
    app_name: str = Field(
        default="",
        description="App whose window to search via OCR. Empty = full screen.",
    )


class _ClickTextInput(BaseModel):
    text: str = Field(description="On-screen text to click (located via OCR, case-insensitive).")
    app_name: _AppName = Field(description="App to focus and click in. Must not be empty.")


class _ClickAtInput(BaseModel):
    x: Annotated[int, Field(ge=0)] = Field(description="Screen x in logical points (e.g. from find_text_on_screen).")
    y: Annotated[int, Field(ge=0)] = Field(description="Screen y in logical points.")


class _ScrollAtInput(BaseModel):
    x:         Annotated[int, Field(ge=0)] = Field(description="Screen x in logical points.")
    y:         Annotated[int, Field(ge=0)] = Field(description="Screen y in logical points.")
    direction: _ScrollDir = Field(description="Scroll direction: up | down | left | right.")
    amount:    Annotated[int, Field(gt=0)] = Field(default=300, description="Pixels to scroll (must be > 0).")


_BATCH_TOOL_NAMES = {
    "press_control", "type_into_control", "get_control_value",
    "click", "double_click", "right_click", "scroll", "type_text", "hotkey",
}

_BATCH_SPLIT_RE = re.compile(
    r"(" + "|".join(re.escape(t) for t in sorted(_BATCH_TOOL_NAMES, key=len, reverse=True)) + r")\(",
)


def _parse_batch_actions(raw: str) -> list[tuple[str, dict]]:
    """Parse a string of tool calls into (tool_name, kwargs) tuples.

    Accepts formats like::

        press_control(index=5)
        type_into_control(index=3, text='hello world')
        hotkey(keys='return')

    Uses known tool names as split anchors so delimiter characters inside
    text values (semicolons, pipes, parentheses) don't cause mis-splits.
    """
    parts = _BATCH_SPLIT_RE.split(raw)
    results: list[tuple[str, dict]] = []

    i = 1
    while i < len(parts) - 1:
        tool_name = parts[i].strip()
        arg_block = parts[i + 1].strip()

        # Strip trailing ')' — find the last one that closes this call.
        # Everything after it (whitespace, newlines, separators) is ignored.
        close = arg_block.rfind(")")
        if close >= 0:
            arg_block = arg_block[:close]

        kwargs: dict = {}
        if arg_block.strip():
            for m in re.finditer(
                r"(\w+)\s*=\s*(?:"
                r"'((?:[^'\\]|\\.)*)'"      # single-quoted value
                r"|\"((?:[^\"\\]|\\.)*)\""   # double-quoted value
                r"|([^,\)]+))",              # unquoted value
                arg_block,
            ):
                key = m.group(1)
                quoted = m.group(2) is not None or m.group(3) is not None
                val = m.group(2) if m.group(2) is not None else (
                    m.group(3) if m.group(3) is not None else m.group(4).strip()
                )
                if not quoted and isinstance(val, str) and val.lower() in ("true", "false"):
                    val = val.lower() == "true"
                else:
                    try:
                        val = int(val)
                    except (ValueError, TypeError):
                        try:
                            val = float(val)
                        except (ValueError, TypeError):
                            pass
                kwargs[key] = val

        results.append((tool_name, kwargs))
        i += 2

    # Dedup: local models sometimes degenerate into repeating the same steps.
    # Preserve order, keep only the first occurrence of each (name, args) pair.
    seen: set[str] = set()
    deduped: list[tuple[str, dict]] = []
    for entry in results:
        key = repr(entry)
        if key not in seen:
            seen.add(key)
            deduped.append(entry)
    return deduped


# ═══════════════════════════════════════════════════════════════════════════════
# MacOSToolkit — config, state, helpers, and LangChain tool creation
# ═══════════════════════════════════════════════════════════════════════════════

class MacOSToolkit:
    """Encapsulates macOS Accessibility tools with configurable parameters.

    Config values are resolved from ``Environment`` (which reads ``.env`` /
    ``os.getenv``) by default.  Pass explicit values to override::

        tk = MacOSToolkit(scan_max_workers=8, scan_max_elements=1000)
        agent_tools = tk.tools          # list of LangChain tools
        vision_tools = tk.vision_tools  # screenshot tool

    A default singleton is created at module level so the legacy
    ``MACOS_TOOLS`` / ``MACOS_VISION_TOOLS`` lists keep working.
    """

    def __init__(
        self,
        *,
        ax_ipc_timeout: float | None = None,
        scan_max_depth: int | None = None,
        scan_max_elements: int | None = None,
        scan_max_workers: int | None = None,
        scan_time_budget: float | None = None,
        ax_bridge_init_delay: float | None = None,
        ax_bridge_max_wait: float | None = None,
        launch_delay: float = 0.0,
        activate_delay: float = 0.5,
        action_delay: float = 0.3,
        scan_cache_ttl: float = 2.0,
        skip_apps: set[str] | None = None,
    ):
        # ── Config (fall back to Environment for any value not supplied) ──
        self.ax_ipc_timeout = ax_ipc_timeout if ax_ipc_timeout is not None else Environment.get_ax_ipc_timeout()
        self.scan_max_depth = scan_max_depth if scan_max_depth is not None else Environment.get_scan_depth()
        self.scan_max_elements = scan_max_elements if scan_max_elements is not None else Environment.get_scan_max_elements()
        self.scan_max_workers = scan_max_workers if scan_max_workers is not None else Environment.get_scan_max_workers()
        self.scan_time_budget = scan_time_budget if scan_time_budget is not None else Environment.get_scan_time_budget()
        self.ax_bridge_init_delay = ax_bridge_init_delay if ax_bridge_init_delay is not None else Environment.get_ax_bridge_init_delay()
        # Total budget for polling an Electron app's tree to populate after the
        # bridge flip. Always at least the single init delay so behaviour never
        # regresses below the old fixed-sleep path.
        _bridge_max_wait = ax_bridge_max_wait if ax_bridge_max_wait is not None else _DEFAULT_AX_BRIDGE_MAX_WAIT
        self.ax_bridge_max_wait = max(_bridge_max_wait, self.ax_bridge_init_delay)
        self.launch_delay = launch_delay
        self.activate_delay = activate_delay
        self.action_delay = action_delay
        self.scan_cache_ttl = scan_cache_ttl
        self.skip_apps = skip_apps or set()

        # True when the vision-combo read_screen (text+image) has been swapped
        # in for a vision-capable agent.  Flips the AX-disabled fallback hint so
        # get_screen_controls tells the model read_screen ALSO returns a
        # screenshot to look at, instead of framing it as text-only OCR.  Set by
        # the session builder when it applies the vision variant.
        self.vision_mode: bool = False

        # ── Mutable state ─────────────────────────────────────────────────
        self._element_registry: dict[int, object] = {}
        self._coord_registry: dict[int, tuple[int, int]] = {}
        self._scan_caches: dict[str, tuple[list[dict], dict[int, object], float]] = {}
        self._fg_apps_cache: tuple[float, dict[str, int]] = (0.0, {})
        self._ax_bridge_activated: set[int] = set()
        # Track AXWebArea elements that have had AXManualAccessibility enabled.
        # Keyed by id(element) so we only pay the init delay once per WebArea.
        self._web_area_activated: set[int] = set()

        self._ax_thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(2, self.scan_max_workers + 1),
            thread_name_prefix="ax-ipc",
        )

        # ── Build LangChain tools ─────────────────────────────────────────
        self.tools, self.vision_tools, self._batch_dispatch = self._build_tools()

    # -------------------------------------------------------------------
    # AX IPC helpers (instance-bound for timeout + pool config)
    # -------------------------------------------------------------------

    def _ax_get_safe(self, element, attr, timeout: float | None = None):
        timeout = timeout if timeout is not None else self.ax_ipc_timeout
        future = self._ax_thread_pool.submit(_ax_get, element, attr)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            _logger.warning("AX IPC timed out reading %s (%.1fs)", attr, timeout)
            return None

    def _ax_set_safe(self, element, attr, value, timeout: float | None = None) -> int:
        timeout = timeout if timeout is not None else self.ax_ipc_timeout

        def _do():
            return AXUIElementSetAttributeValue(element, attr, value)
        future = self._ax_thread_pool.submit(_do)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            _logger.warning("AX IPC timed out setting %s (%.1fs)", attr, timeout)
            return -1

    def _ax_get_multi(self, element, attrs: list = _WALK_ATTRS, timeout: float | None = None) -> dict | None:
        timeout = timeout if timeout is not None else self.ax_ipc_timeout

        def _do():
            err, values = AXUIElementCopyMultipleAttributeValues(
                element, attrs, 0, None,
            )
            if err != 0 or values is None:
                return None
            out: dict = {}
            for attr, val in zip(attrs, values):
                out[attr] = None if (val is None or _is_ax_error(val)) else val
            return out
        future = self._ax_thread_pool.submit(_do)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            _logger.warning("AX IPC batch read timed out (%.1fs)", timeout)
            return None

    # -------------------------------------------------------------------
    # Scan cache
    # -------------------------------------------------------------------

    def _invalidate_scan_cache(self, app_name: str | None = None) -> None:
        if app_name is None:
            self._scan_caches.clear()
            # Element IDs are process-lifetime — clear so new WebArea instances
            # (e.g. after app restart) are re-activated on the next scan.
            self._web_area_activated.clear()
        else:
            self._scan_caches.pop(app_name.lower(), None)
            self._scan_caches.pop("", None)

    def _cache_get(self, app_filter: str) -> tuple[list[dict], dict[int, object]] | None:
        entry = self._scan_caches.get(app_filter.lower())
        if entry is None:
            return None
        result, registry, expires_at = entry
        if time.monotonic() > expires_at:
            self._scan_caches.pop(app_filter.lower(), None)
            return None
        return result, registry

    def _cache_put(self, app_filter: str, result: list[dict], registry: dict[int, object]) -> None:
        self._scan_caches[app_filter.lower()] = (
            result, registry, time.monotonic() + self.scan_cache_ttl,
        )

    # -------------------------------------------------------------------
    # Live process list
    # -------------------------------------------------------------------

    def _get_foreground_apps(self, max_age: float = 2.0) -> dict[str, int]:
        now = time.monotonic()
        if now - self._fg_apps_cache[0] < max_age:
            return self._fg_apps_cache[1]

        result: dict[str, int] = {}

        script = (
            'tell application "System Events"\n'
            '    set output to ""\n'
            '    repeat with p in (every process whose background only is false)\n'
            '        set output to output & (unix id of p as string) & "|" & (name of p) & "\\n"\n'
            '    end repeat\n'
            '    return output\n'
            'end tell'
        )
        try:
            proc = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=6,
            )
            for line in proc.stdout.strip().splitlines():
                if "|" in line:
                    pid_str, name = line.split("|", 1)
                    try:
                        result[name.strip()] = int(pid_str.strip())
                    except ValueError:
                        pass
        except Exception:
            pass

        try:
            proc = subprocess.run(
                ["ps", "-axco", "pid,comm"],
                capture_output=True, text=True, timeout=4,
            )
            for line in proc.stdout.strip().splitlines()[1:]:
                parts = line.split(None, 1)
                if len(parts) == 2:
                    pid_str, comm = parts
                    name = comm.strip()
                    if name and name not in result:
                        try:
                            result[name] = int(pid_str.strip())
                        except ValueError:
                            pass
        except Exception:
            pass

        self._fg_apps_cache = (now, result)
        return result

    @staticmethod
    def _frontmost_app_name() -> str:
        """Name of the app currently owning the keyboard focus, or "".

        Synthesized keystrokes land on whatever is frontmost *at that
        instant*.  When two agents share the host, a second session can steal
        focus between this agent's calls, so ``type_text`` / ``hotkey`` report
        the frontmost app back to the caller as a cheap drift check.
        """
        try:
            app = NSWorkspace.sharedWorkspace().frontmostApplication()
            return str(app.localizedName() or "") if app is not None else ""
        except Exception:
            return ""

    def _activate_and_verify(
        self, app_name: str, timeout: float = 1.6,
    ) -> tuple[bool, str]:
        """Bring *app_name* to the front and CONFIRM it actually got focus.

        Returns ``(success, frontmost_name)``. ``NSApplicationActivateIgnoringOtherApps``
        is deprecated/ineffective on modern macOS for cross-app activation, so we
        drive AppleScript ``activate`` (reliable) and the NS API together, then
        poll the frontmost app until it matches or the timeout elapses. A False
        result means another app is holding focus — synthesized clicks/keystrokes
        would land there, so callers must NOT type/click and should fall back to
        the focus-free read path.
        """
        display = app_name
        ns_app = None
        ws = NSWorkspace.sharedWorkspace()
        for app in ws.runningApplications():
            if str(app.localizedName() or "").lower() == app_name.lower():
                display = str(app.localizedName() or app_name)
                ns_app = app
                break

        try:
            subprocess.run(
                ["osascript", "-e", f'tell application "{display}" to activate'],
                capture_output=True, timeout=5,
            )
        except Exception:
            pass
        if ns_app is not None:
            try:
                ns_app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
            except Exception:
                pass

        deadline = time.monotonic() + timeout
        front = ""
        while time.monotonic() < deadline:
            front = self._frontmost_app_name()
            if front.lower() == display.lower():
                return True, front
            time.sleep(0.15)
        return False, front

    def _resolve_pid(self, app_name: str) -> int:
        """Return the pid for *app_name* (case-insensitive), or 0 if not found.

        Checks NSWorkspace's running applications first, then the foreground-app
        cache. Shared by the focus-free typing path and the OCR window target so
        pid resolution stays consistent.
        """
        if not app_name:
            return 0
        ws = NSWorkspace.sharedWorkspace()
        for app in ws.runningApplications():
            if str(app.localizedName() or "").lower() == app_name.lower():
                return int(app.processIdentifier())
        for name, pid in self._get_foreground_apps(max_age=0).items():
            if name.lower() == app_name.lower():
                return int(pid)
        return 0

    # -------------------------------------------------------------------
    # Tree walk + scan
    # -------------------------------------------------------------------

    def _window_viewport(self, window_element) -> _Rect:
        """Derive a clipping rect from a window's AXPosition + AXSize.

        Returns ``_NO_CLIP`` if the attributes cannot be read so that
        callers never need to special-case failures.
        """
        pos = self._ax_get_safe(window_element, kAXPositionAttribute)
        size = self._ax_get_safe(window_element, kAXSizeAttribute)
        if pos is None or size is None:
            return _NO_CLIP
        try:
            x, y = _ax_point(pos)
            w, h = _ax_size(size)
            if w <= 0 or h <= 0:
                return _NO_CLIP
            return _Rect(x, y, w, h)
        except Exception:
            return _NO_CLIP

    def _walk(
        self, element, results: list[dict], app_name: str,
        viewport: _Rect,
        deadline: float,
        depth: int = 0,
        structural_depth: int = 0,
    ) -> None:
        if (
            depth > self.scan_max_depth
            or len(results) >= self.scan_max_elements
            or time.monotonic() > deadline
        ):
            return

        attrs = self._ax_get_multi(element)
        if attrs is None:
            return

        role = attrs.get(kAXRoleAttribute) or ""
        pos_raw = attrs.get(kAXPositionAttribute)

        include = role in _INTERACTIVE_ROLES
        is_structural = role in _STRUCTURAL_ROLES

        if not include and role == "AXGroup" and pos_raw is not None:
            include = bool(
                attrs.get(kAXTitleAttribute)
                or attrs.get(kAXDescriptionAttribute)
            )
            if include:
                is_structural = False

        if include and role == "AXRow":
            has_label = bool(
                attrs.get(kAXTitleAttribute)
                or attrs.get(kAXDescriptionAttribute)
                or attrs.get(kAXValueAttribute)
            )
            if not has_label:
                include = False

        if include and pos_raw is not None:
            x, y = _ax_point(pos_raw)
            size_raw = attrs.get(kAXSizeAttribute)
            w, h = _ax_size(size_raw) if size_raw else (0, 0)

            if not viewport.intersects(x, y, max(w, 1), max(h, 1)):
                include = False

            if include:
                label = (
                    attrs.get(kAXTitleAttribute)
                    or attrs.get(kAXDescriptionAttribute)
                    or attrs.get(kAXValueAttribute)
                    or ""
                )
                if not label and role == "AXButton":
                    fallback = (
                        self._ax_get_safe(element, "AXHelp")
                        or (self._ax_get_safe(element, "AXSubrole") or "").replace("AX", "")
                        or ""
                    )
                    if fallback.lower() not in ("", "button"):
                        label = fallback
                if not label:
                    include = False

            if include:
                results.append({
                    "app": app_name,
                    "role": role.replace("AX", ""),
                    "label": str(label).strip()[:100 if role == "AXCell" else 500],
                    "x": x, "y": y, "w": w, "h": h,
                    "cx": x + w // 2,
                    "cy": y + h // 2,
                    "enabled": attrs.get(kAXEnabledAttribute),
                    "_element": element,
                    "_depth": structural_depth,
                })

        emit_structural = (
            is_structural
            and not include
            and structural_depth < _MAX_STRUCTURAL_DEPTH
        )
        if emit_structural:
            struct_label = (
                attrs.get(kAXTitleAttribute)
                or attrs.get(kAXDescriptionAttribute)
                or ""
            )
            results.append({
                "_structural": True,
                "app": app_name,
                "role": role.replace("AX", "").lower(),
                "label": str(struct_label).strip()[:80] if struct_label else "",
                "_depth": structural_depth,
            })

        if len(results) >= self.scan_max_elements:
            return

        children = attrs.get(kAXChildrenAttribute)

        if role == "AXWebArea":
            elem_id = id(element)
            if elem_id not in self._web_area_activated:
                err = self._ax_set_safe(element, "AXManualAccessibility", True)
                self._web_area_activated.add(elem_id)
                if err == 0:
                    time.sleep(0.3)
                    children = self._ax_get_safe(element, kAXChildrenAttribute) or children

        if not children:
            return
        child_struct_depth = structural_depth + (1 if emit_structural else 0)
        for child in children:
            if len(results) >= self.scan_max_elements or time.monotonic() > deadline:
                return
            self._walk(
                child, results, app_name, viewport, deadline,
                depth + 1, child_struct_depth,
            )

    @staticmethod
    def _ax_root_error(pid: int) -> int:
        """Return the AXError from reading an app root's role (0 == ok).

        Used to explain an empty scan: ``-25211`` (kAXErrorAPIDisabled) means
        the app has its accessibility interface switched off entirely (Slack
        does this), which no amount of bridge-flipping or polling can fix —
        distinct from a slow tree that just needs more time.
        """
        try:
            root = AXUIElementCreateApplication(pid)
            err, _ = AXUIElementCopyAttributeValue(root, kAXRoleAttribute, None)
            return err
        except Exception:
            return 0

    def _ensure_ax_bridge(self, root, name: str, pid: int) -> list:
        """Return an app's root AX children, enabling the Chromium/Electron bridge.

        macOS native apps (Outlook, Finder, Calculator) expose their AX tree
        directly.  Electron/Chromium apps (Slack, Discord, VS Code, Teams) keep
        their web DOM hidden until ``AXManualAccessibility`` is set on the app
        ROOT.  Critically, the root still reports its *window shells* as
        children while the DOM is hidden — so reading children first and
        skipping the flip when they're non-empty (the regression that broke
        Slack) leaves the actual content invisible: ``_walk`` only ever sees the
        window chrome and the scan comes back "no controls".

        So we ALWAYS flip the bridge once per (session, pid) before trusting any
        read.  The flip is idempotent and harmless for native apps.  The lock
        serializes just the flip+settle so two concurrent sessions don't race it
        on the same app; the tree walk itself stays parallel.  On first contact
        we settle for ``ax_bridge_init_delay``; if the tree still isn't up
        (slow Electron bring-up or AX contention) we poll with backoff up to
        ``ax_bridge_max_wait``, re-asserting the bridge each iteration.

        Returns whatever children were found (possibly empty if the tree never
        populated within the budget).
        """
        flip_ok = False
        if pid not in self._ax_bridge_activated:
            # Fast-fail apps that switch their AX interface OFF entirely (Slack):
            # the root can't even report its role, so flipping/polling the bridge
            # is pure latency. Bail immediately and let the caller surface the
            # kAXErrorAPIDisabled message.
            if self._ax_root_error(pid) == _kAXErrorAPIDisabled:
                self._ax_bridge_activated.add(pid)
                return []
            with _AX_BRIDGE_LOCK:
                err = self._ax_set_safe(root, "AXManualAccessibility", True)
            self._ax_bridge_activated.add(pid)
            flip_ok = err == 0
            if err == -1:
                _logger.warning(
                    "AX bridge flip timed out for %s (pid %d); reading anyway",
                    name, pid,
                )
            elif flip_ok:
                # err == 0 means the app accepted AXManualAccessibility, i.e. it
                # is a real Chromium/Electron app — give it time to build the DOM.
                # Native apps return -25205 (AttributeUnsupported); their tree is
                # already live, so we skip the settle delay entirely.
                time.sleep(self.ax_bridge_init_delay)

        children = self._ax_get_safe(root, kAXChildrenAttribute) or []
        if children:
            return children

        # Empty tree. Only worth polling if this app actually owns a Chromium
        # bridge (flip succeeded) — otherwise an empty native tree just means no
        # open windows and polling would waste the whole budget.
        if not flip_ok:
            return children
        deadline = time.monotonic() + self.ax_bridge_max_wait
        delay = max(self.ax_bridge_init_delay, 0.2)
        while time.monotonic() < deadline:
            self._ax_set_safe(root, "AXManualAccessibility", True)
            remaining = deadline - time.monotonic()
            time.sleep(max(0.0, min(delay, remaining)))
            children = self._ax_get_safe(root, kAXChildrenAttribute) or []
            if children:
                return children
            delay = min(delay * 1.5, 1.0)  # back off, capped at 1s per poll
        return children

    def _scan(
        self,
        skip_apps: set[str] | None = None,
        app_filter: str = "",
        force: bool = False,
    ) -> list[dict]:
        if not force:
            cached = self._cache_get(app_filter)
            if cached is not None:
                result, registry = cached
                self._element_registry = registry
                self._coord_registry = {
                    c["index"]: (c["cx"], c["cy"])
                    for c in result if not c.get("_structural")
                }
                return result

        skip = skip_apps or self.skip_apps
        ws = NSWorkspace.sharedWorkspace()
        results: list[dict] = []

        apps: dict[str, int] = {}
        for app in ws.runningApplications():
            if app.activationPolicy() > 1:
                continue
            n = str(app.localizedName() or "")
            if n:
                apps[n] = app.processIdentifier()

        for n, pid in self._get_foreground_apps().items():
            if n not in apps:
                apps[n] = pid

        for name, pid in apps.items():
            if name in skip:
                continue
            if app_filter and name.lower() != app_filter.lower():
                continue
            root = AXUIElementCreateApplication(pid)

            children = self._ensure_ax_bridge(root, name, pid)
            if not children:
                _logger.debug(
                    "No AX children for %s (pid %d) after bridge retry", name, pid,
                )
                continue

            visible = [
                c for c in children
                if not self._ax_get_safe(c, kAXMinimizedAttribute)
            ]

            # Wall-clock budget for walking this app's tree. Set after the
            # one-time AX bridge init sleep above so the delay isn't charged
            # against the walk. Guarantees the scan returns within a bounded
            # time even for huge Electron trees (e.g. Slack), yielding whatever
            # controls were gathered so far.
            deadline = time.monotonic() + self.scan_time_budget

            if self.scan_max_workers <= 1 or len(visible) <= 1:
                for child in visible:
                    if len(results) >= self.scan_max_elements or time.monotonic() > deadline:
                        break
                    viewport = self._window_viewport(child)
                    self._walk(child, results, name, viewport, deadline)
            else:
                per_child: list[list[dict]] = [[] for _ in visible]
                viewports = [self._window_viewport(c) for c in visible]
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.scan_max_workers,
                    thread_name_prefix="ax-walk",
                ) as walk_pool:
                    futs = [
                        walk_pool.submit(
                            self._walk, child, per_child[i], name,
                            viewports[i], deadline,
                        )
                        for i, child in enumerate(visible)
                    ]
                    for fut in concurrent.futures.as_completed(futs):
                        try:
                            fut.result()
                        except Exception:
                            _logger.debug("Parallel walk error", exc_info=True)
                for partial in per_child:
                    for c in partial:
                        if len(results) >= self.scan_max_elements:
                            break
                        results.append(c)

        seen: dict[tuple[int, int], dict] = {}
        for c in results:
            if c.get("_structural"):
                continue
            key = (c["cx"], c["cy"])
            if key not in seen or len(c["label"]) > len(seen[key]["label"]):
                seen[key] = c

        surviving_ids = {id(c) for c in seen.values()}

        deduped: list[dict] = []
        for c in results:
            if c.get("_structural") or id(c) in surviving_ids:
                deduped.append(c)

        new_registry: dict[int, object] = {}
        idx = 0
        for c in deduped:
            if not c.get("_structural"):
                idx += 1
                c["index"] = idx
                new_registry[idx] = c.pop("_element")

        self._element_registry = new_registry
        self._coord_registry = {
            c["index"]: (c["cx"], c["cy"])
            for c in deduped if not c.get("_structural")
        }
        self._cache_put(app_filter, deduped, new_registry)
        return deduped

    @staticmethod
    def _fmt(controls: list[dict]) -> str:
        """Render controls as an indented ARIA-style tree.

        Structural containers provide hierarchy (indentation).
        Interactive elements show as ``[index]RC'label'`` where *RC* is a
        short role code from ``_ROLE_CODES``.
        """
        if not controls:
            return ""

        apps: dict[str, list[dict]] = {}
        app_order: list[str] = []
        for c in controls:
            app = c["app"]
            if app not in apps:
                apps[app] = []
                app_order.append(app)
            apps[app].append(c)

        lines: list[str] = []
        for app in app_order:
            lines.append(app)
            nodes = apps[app]

            has_structure = any(n.get("_structural") for n in nodes)

            # Pass 1: identify structural nodes with interactive descendants
            has_children: set[int] = set()
            stack: list[tuple[int, int]] = []
            for i, n in enumerate(nodes):
                d = n.get("_depth", 0)
                if n.get("_structural"):
                    while stack and stack[-1][1] >= d:
                        stack.pop()
                    stack.append((i, d))
                else:
                    for idx, _ in stack:
                        has_children.add(idx)

            # Pass 2: render tree
            i = 0
            while i < len(nodes):
                n = nodes[i]

                if n.get("_structural"):
                    if i not in has_children:
                        i += 1
                        continue
                    indent = " " * (n["_depth"] + 1)
                    lbl = f" '{n['label']}'" if n.get("label") else ""
                    lines.append(f"{indent}{n['role']}{lbl}")
                    i += 1
                    continue

                # Batch consecutive same-depth same-role interactive nodes
                batch = [n]
                while (
                    i + 1 < len(nodes)
                    and not nodes[i + 1].get("_structural")
                    and nodes[i + 1].get("_depth", 0) == n.get("_depth", 0)
                    and nodes[i + 1]["role"] == n["role"]
                ):
                    i += 1
                    batch.append(nodes[i])

                depth = n.get("_depth", 0) if has_structure else 0
                indent = " " * (depth + 1)
                rc = _ROLE_CODES.get(n["role"], n["role"][:2])
                parts = [f"[{b['index']}]{rc}'{b['label']}'" for b in batch]

                one_line = f"{indent}{' '.join(parts)}"
                if len(one_line) <= 120:
                    lines.append(one_line)
                else:
                    for p in parts:
                        lines.append(f"{indent}{p}")

                i += 1

        return "\n".join(lines)

    def _resolve_click_coords(self, index: int) -> tuple[int, int] | str:
        coords = self._coord_registry.get(index)
        if coords is not None:
            return coords
        if self._element_registry.get(index) is None:
            return (
                f"Control #{index} not in registry. "
                "Call get_screen_controls() or wait_for_controls() first."
            )
        return f"Control #{index} has no coordinates in the last scan."

    # -------------------------------------------------------------------
    # Vision OCR fallback (for apps with the Accessibility API switched off)
    # -------------------------------------------------------------------

    def _window_target(
        self, app_name: str,
    ) -> tuple[int | None, tuple[int, int, int, int]] | None:
        """Return ``(window_id, (x, y, w, h))`` for an app's largest window.

        Coordinates are logical points (top-left origin). ``window_id`` is None
        for the whole-screen case (empty *app_name*). Uses CoreGraphics' window
        list (NOT Accessibility), so it works even for apps that switch AX off
        (Slack) and for windows that aren't frontmost. Returns None if the app
        isn't running or has no visible normal window.
        """
        if not app_name:
            sw, sh = _logical_screen_size()
            return (None, (0, 0, sw, sh))
        if not _VISION_AVAILABLE:
            return None

        pid = self._resolve_pid(app_name)
        if not pid:
            return None

        info = CGWindowListCopyWindowInfo(
            kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements,
            kCGNullWindowID,
        )
        best: tuple[int, tuple[int, int, int, int]] | None = None
        best_area = 0
        for win in info or []:
            if int(win.get("kCGWindowOwnerPID", -1)) != pid:
                continue
            if int(win.get("kCGWindowLayer", 0)) != 0:  # 0 == normal window
                continue
            bounds = win.get("kCGWindowBounds")
            if not bounds:
                continue
            x = int(bounds.get("X", 0))
            y = int(bounds.get("Y", 0))
            w = int(bounds.get("Width", 0))
            h = int(bounds.get("Height", 0))
            area = w * h
            if w > 0 and h > 0 and area > best_area:
                best = (int(win.get("kCGWindowNumber")), (x, y, w, h))
                best_area = area
        return best

    @staticmethod
    def _cgimage_to_png_bytes(cg_image) -> bytes:
        """PNG-encode a CGImage via NSBitmapImageRep (no extra screen capture)."""
        rep = NSBitmapImageRep.alloc().initWithCGImage_(cg_image)
        data = rep.representationUsingType_properties_(NSBitmapImageFileTypePNG, {})
        return bytes(data)

    @staticmethod
    def _downscale_png_b64(png_bytes: bytes, max_side: int = 1568) -> str:
        """Return base64 PNG with the longest side capped at *max_side* px.

        Caps token cost for the returned image. Click coordinates are unaffected
        because they come from OCR logical bounds, not the returned resolution.
        """
        from PIL import Image  # bundled with pyautogui (Pillow)

        img = Image.open(io.BytesIO(png_bytes))
        w, h = img.size
        longest = max(w, h)
        if longest > max_side:
            scale = max_side / longest
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _grab_window(
        self, window_id: int | None, region: tuple[int, int, int, int],
    ) -> tuple[object | None, bytes | None]:
        """Capture a window (or the screen) ONCE.

        Returns ``(vn_handler, png_bytes)`` so a single capture serves both OCR
        and the returned screenshot. ``(None, None)`` if the window image could
        not be created.

        For a window it uses CGWindowListCreateImage (by window ID), so it reads
        the window's OWN pixels even when another window is on top. For the
        whole screen (``window_id is None``) it falls back to ``pyautogui``.
        """
        ox, oy, w, h = region
        if window_id is None:
            try:
                img = pyautogui.screenshot(region=(ox, oy, w, h))
            except Exception as exc:
                # screencapture exits non-zero without Screen Recording access.
                _logger.warning("full-screen capture failed: %s", exc)
                return None, None
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            raw = buf.getvalue()
            from Foundation import NSData  # local import: full-screen path only
            data = NSData.dataWithBytes_length_(raw, len(raw))
            handler = VNImageRequestHandler.alloc().initWithData_options_(data, {})
            return handler, raw
        cg_image = CGWindowListCreateImage(
            CGRectNull,
            kCGWindowListOptionIncludingWindow,
            window_id,
            kCGWindowImageBoundsIgnoreFraming,
        )
        if cg_image is None:
            return None, None
        handler = VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, {})
        return handler, self._cgimage_to_png_bytes(cg_image)

    @staticmethod
    def _run_ocr(handler, region: tuple[int, int, int, int]) -> list[dict]:
        """Run Vision text recognition on *handler*, mapping boxes to click coords.

        Returns ``[{"text", "conf", "x", "y"}]`` with x/y as logical screen
        points ready for ``click_at``. DPI-proof: Vision yields normalised
        (0–1, bottom-left) boxes mapped onto the window's logical size + origin,
        so the Retina backing-scale cancels out.
        """
        ox, oy, w, h = region
        request = VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(True)
        ok, _err = handler.performRequests_error_([request], None)
        if not ok:
            return []
        out: list[dict] = []
        for obs in (request.results() or []):
            cands = obs.topCandidates_(1)
            if not cands:
                continue
            box = obs.boundingBox()  # normalised, bottom-left origin
            cx_frac = box.origin.x + box.size.width / 2.0
            cy_frac = 1.0 - (box.origin.y + box.size.height / 2.0)  # flip Y
            out.append({
                "text": str(cands[0].string()),
                "conf": float(cands[0].confidence()),
                "x": int(ox + cx_frac * w),
                "y": int(oy + cy_frac * h),
            })
        return out

    def _ocr_window(
        self, window_id: int | None, region: tuple[int, int, int, int],
    ) -> list[dict] | None:
        """OCR a single window (or the screen) and return text + click coords.

        Returns ``[{"text", "conf", "x", "y"}]``. Returns None if Vision is
        unavailable, ``[]`` if the window image could not be captured.
        """
        if not _VISION_AVAILABLE:
            return None
        handler, _png = self._grab_window(window_id, region)
        if handler is None:
            return []
        return self._run_ocr(handler, region)

    def _capture_for_read(
        self,
        window_id: int | None,
        region: tuple[int, int, int, int],
        want_image: bool,
    ) -> tuple[list[dict] | None, str | None]:
        """OCR + optional downscaled screenshot from ONE capture.

        Returns ``(results, png_b64 | None)``. ``results`` is None when Vision
        is unavailable, ``[]`` when the capture failed. ``png_b64`` is only
        populated when *want_image* is set and the capture succeeded.
        """
        if not _VISION_AVAILABLE:
            return None, None
        handler, png = self._grab_window(window_id, region)
        if handler is None:
            return [], None
        results = self._run_ocr(handler, region)
        b64 = None
        if want_image and png is not None:
            b64 = self._downscale_png_b64(png)
        return results, b64

    # -------------------------------------------------------------------
    # Tool builder — creates @tool-decorated closures bound to this instance
    # -------------------------------------------------------------------

    def _build_tools(self):
        tk = self

        # ── Observation ───────────────────────────────────────────────────

        @tool("list_apps", args_schema=_ListAppsInput)
        async def list_apps(query: str) -> str:
            """Find running apps whose name contains the search query (case-insensitive).

            Always call this before launch_app / activate_app / get_screen_controls
            to get the exact localised app name.

            Returns every matching app name on its own line, or a not-running
            message if nothing matches.

            Examples:
              list_apps('slack')   → 'Slack'
              list_apps('chrome')  → 'Google Chrome'
            """
            def _running_app_names() -> list[str]:
                ws = NSWorkspace.sharedWorkspace()
                ws_names = {
                    str(app.localizedName() or "")
                    for app in ws.runningApplications()
                    if app.activationPolicy() <= 1 and (app.localizedName() or "")
                }
                fg_names = set(tk._get_foreground_apps(max_age=0).keys())
                return sorted(ws_names | fg_names)

            all_names = await asyncio.to_thread(_running_app_names)
            q = query.strip().lower()
            matches = [n for n in all_names if q in n.lower()]
            if not matches:
                return (
                    f"No running app matches '{query}'.\n"
                    f"The app may not be open yet — use launch_app('{query}') to open it, "
                    f"or try a shorter search term."
                )
            lines = [f"<matches query='{query}'>"]
            for name in matches:
                lines.append(f"  <app>{name}</app>")
            lines.append("</matches>")
            return "\n".join(lines)

        @tool("get_screen_controls", args_schema=_GetScreenControlsInput)
        async def get_screen_controls(
            app_name: str, role: str = "", label: str = "",
        ) -> str:
            """Return interactive UI controls visible on screen for the named app.

            You MUST provide the app name (use list_apps to find exact names).

            Optional filters (applied to cached scan results):
              role  — case-insensitive role name, e.g. 'TextField', 'Button'
              label — case-insensitive substring match on control label

            First call for an app: omit filters to see everything.
            Follow-up calls: pass role/label to narrow results and save tokens.

            Examples:
              get_screen_controls('Outlook')
              get_screen_controls('Outlook', role='TextField')
              get_screen_controls('Outlook', label='Subject')
              get_screen_controls('Outlook', role='Button', label='Send')
            """
            # The AX tree scan is blocking IPC (and can sleep for bridge init),
            # so run the whole synchronous body in a worker thread — otherwise it
            # stalls the event loop and freezes every other running session.
            def _collect() -> str:
                controls = tk._scan(app_filter=app_name)

                if app_name and not controls:
                    ws = NSWorkspace.sharedWorkspace()
                    app_pid = 0
                    for app in ws.runningApplications():
                        if (
                            app.activationPolicy() <= 1
                            and str(app.localizedName() or "").lower() == app_name.lower()
                        ):
                            app_pid = app.processIdentifier()
                            break
                    if not app_pid:
                        fg = tk._get_foreground_apps()
                        for n, p in fg.items():
                            if n.lower() == app_name.lower():
                                app_pid = p
                                break
                    if not app_pid:
                        return (
                            f"'{app_name}' is not running.\n"
                            f"Call list_apps() to see what is running, "
                            f"then open_app('{app_name}') to launch it."
                        )

                    # Distinguish "accessibility switched off for this app"
                    # (unfixable via AX) from "tree slow/empty" (retry helps).
                    ax_err = tk._ax_root_error(app_pid)
                    if ax_err == _kAXErrorAPIDisabled:
                        if tk.vision_mode:
                            read_screen_hint = (
                                f"  2. read_screen('{app_name}') — returns BOTH the "
                                f"on-screen text AND a screenshot of the window. LOOK at "
                                f"the image to understand layout, icons, and which item "
                                f"is selected; use the text for exact labels.\n"
                            )
                        else:
                            read_screen_hint = (
                                f"  2. read_screen('{app_name}') — read the actual "
                                f"on-screen text.\n"
                            )
                        # Electron apps launched with a remote-debugging port can
                        # be read structurally over CDP — far better than OCR.
                        cdp_port = _find_cdp_port(app_pid)
                        if cdp_port:
                            cdp_hint = (
                                f"  1. read_app_dom('{app_name}') — BEST: reads the "
                                f"app's full UI as structured text over CDP "
                                f"(remote-debugging port {cdp_port} is live). No OCR "
                                f"guesswork.\n"
                            )
                        else:
                            cdp_hint = (
                                f"  1. read_app_dom('{app_name}') — structured CDP read "
                                f"(needs the app relaunched with "
                                f"--remote-debugging-port; the tool prints how). "
                                f"Preferred over OCR if you can enable it.\n"
                            )
                        return (
                            f"'{app_name}' has its accessibility interface DISABLED "
                            f"(AXError {ax_err} / kAXErrorAPIDisabled). The app refuses "
                            f"to expose any AX tree, so get_screen_controls / "
                            f"wait_for_controls cannot read it — retrying will NOT help. "
                            f"Slack is the common example.\n"
                            f"Use a fallback that works without AX:\n"
                            f"{cdp_hint}"
                            f"{read_screen_hint}"
                            f"  3. find_text_on_screen('<text>', '{app_name}') — locate a "
                            f"button/label/placeholder, then click_at(x, y) on it.\n"
                            f"  4. To type: find the field's placeholder/label, click_at it "
                            f"to focus, then type_text('...').\n"
                            f"  5. If this app has an AppleScript dictionary, the "
                            f"macos-applescript route is an alternative.\n"
                            f"Do NOT report window titles or screen history as message "
                            f"content. Report ONLY what read_app_dom/read_screen returns; "
                            f"if you cannot read it, say so explicitly."
                        )

                    # Do NOT fall back to a full scan here — scanning every running
                    # app is very expensive and can hang for minutes if any app's
                    # AX IPC is unresponsive (e.g. Electron apps before bridge init).
                    return (
                        f"'{app_name}' is running but exposed no AX controls "
                        f"(AXError {ax_err}).\n"
                        f"For Electron/Chromium apps the bridge can be slow. Try, in order:\n"
                        f"  1. activate_app('{app_name}') → wait_for_controls('{app_name}', timeout=20)\n"
                        f"  2. read_screen('{app_name}') — OCR fallback to read the "
                        f"actual on-screen text; find_text_on_screen + click_at to act.\n"
                        f"  3. Check System Settings → Privacy & Security → Accessibility.\n"
                        f"Do NOT report screen/window history as message content — if you "
                        f"cannot read the actual content, say so explicitly."
                    )

                if not controls:
                    return "No controls found. Call list_apps() to see what is running."

                filtered = controls
                if role or label:
                    filtered = [c for c in filtered if not c.get("_structural")]
                if role:
                    role_lower = role.lower()
                    filtered = [c for c in filtered if c["role"].lower() == role_lower]
                if label:
                    label_lower = label.lower()
                    filtered = [c for c in filtered if label_lower in c["label"].lower()]

                if not filtered:
                    filters = []
                    if role:
                        filters.append(f"role='{role}'")
                    if label:
                        filters.append(f"label='{label}'")
                    return (
                        f"No controls match {' & '.join(filters)} in '{app_name}'.\n"
                        "Try broader filters or omit them to see all controls."
                    )

                return tk._fmt(filtered)

            # Hard cap on total wall-clock time for the entire collect phase
            # (bridge init sleep + pre-walk IPC + walk). asyncio.wait_for raises
            # TimeoutError and the coroutine is cancelled, but the background
            # thread keeps running until its own deadline fires. This guarantees
            # the tool always returns to the agent within the budget regardless
            # of slow or unresponsive AX IPC (e.g. Accessibility not yet granted).
            hard_limit = tk.ax_bridge_max_wait + tk.scan_time_budget + tk.ax_ipc_timeout * 4
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(_collect),
                    timeout=hard_limit,
                )
            except asyncio.TimeoutError:
                _logger.warning(
                    "get_screen_controls timed out after %.0fs for '%s'",
                    hard_limit, app_name,
                )
                return (
                    f"'{app_name}' scan timed out after {hard_limit:.0f}s.\n"
                    "The Accessibility API may be unresponsive. Check:\n"
                    "  • System Settings → Privacy & Security → Accessibility — "
                    "ensure this app has access.\n"
                    "  • Try activate_app(app_name) then wait_for_controls(app_name).\n"
                    "  • Call capture_app_screenshot(app_name) to see the screen."
                )

        @tool
        async def capture_app_screenshot(app_name: str = "") -> list:
            """Capture a pixel screenshot of an app's main window.

            Returns a multimodal content list containing a text caption and the
            image as a base64 data-URL.  Vision-language models can interpret this
            directly; text-only models should NOT have this tool.

            When *app_name* is provided the image is cropped to that app's window
            bounds (via the Accessibility API).  When omitted, a full-screen capture
            is returned.
            """
            def _compute_region() -> tuple[int, int, int, int] | str | None:
                ws = NSWorkspace.sharedWorkspace()
                pid = None
                for app in ws.runningApplications():
                    if str(app.localizedName() or "").lower() == app_name.lower():
                        pid = app.processIdentifier()
                        break
                if pid is None:
                    fg_apps = tk._get_foreground_apps(max_age=0)
                    for name, p in fg_apps.items():
                        if name.lower() == app_name.lower():
                            pid = p
                            break
                if pid is None:
                    return f"App not found: {app_name}. Call list_apps() first."

                root = AXUIElementCreateApplication(pid)
                win = _ax_get(root, kAXMainWindowAttribute)
                if win is None:
                    windows = _ax_get(root, kAXWindowsAttribute)
                    if windows:
                        win = windows[0]
                if win is not None:
                    pos_raw = _ax_get(win, kAXPositionAttribute)
                    size_raw = _ax_get(win, kAXSizeAttribute)
                    if pos_raw and size_raw:
                        x, y = _ax_point(pos_raw)
                        w, h = _ax_size(size_raw)
                        if w > 0 and h > 0:
                            return (x, y, w, h)
                return None

            region = None
            if app_name:
                region = await asyncio.to_thread(_compute_region)
                if isinstance(region, str):
                    return region

            label = app_name or "full screen"
            try:
                img = await asyncio.to_thread(pyautogui.screenshot, region=region)
            except Exception as exc:
                # pyautogui shells out to `screencapture`, which exits non-zero
                # ("could not create image from display/rect") when the host app
                # lacks Screen Recording permission. Surface that as actionable
                # text instead of letting the raw subprocess error bubble up.
                return (
                    f"Could not capture a screenshot of {label}: {exc}. "
                    f"{_SCREEN_RECORDING_HINT}"
                )
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            return [
                {"type": "text", "text": f"Screenshot of {label}"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]

        # ── App lifecycle ─────────────────────────────────────────────────

        @tool("open_app", args_schema=_OpenAppInput)
        async def open_app(app_name: str) -> str:
            """Launch an application, open a URL, or open a file path.

            For app names and .app bundles uses ``open -a <name>`` which searches
            /Applications and /System/Applications.
            For URLs (http/https) uses ``open <url>`` directly.

            After calling this, always call:
              activate_app('<AppName>') → wait_for_controls('<AppName>')

            Examples:
              open_app('Calculator')
              open_app('https://google.com')
              open_app('/System/Applications/Calculator.app')
            """
            is_url = app_name.startswith(("http://", "https://", "file://"))
            if is_url:
                cmd = ["open", app_name]
                delay = 0.5
            elif app_name.startswith("/") and app_name.endswith(".app"):
                bundle_name = Path(app_name).stem
                cmd = ["open", "-a", bundle_name]
                delay = tk.launch_delay
            else:
                cmd = ["open", "-a", app_name]
                delay = tk.launch_delay

            target = app_name.lower().removesuffix(".app").split("/")[-1]

            def _find_app(fresh: bool = False) -> str | None:
                for n in tk._get_foreground_apps(max_age=0 if fresh else 2.0):
                    if n.lower() == target or n.lower().startswith(target):
                        return n
                return None

            already = await asyncio.to_thread(_find_app, False)
            if already:
                return (
                    f"Launched '{already}' — use activate_app('{already}') "
                    f"then wait_for_controls('{already}')"
                )

            proc = await asyncio.to_thread(
                subprocess.run, cmd, capture_output=True, text=True
            )
            if proc.returncode != 0 and proc.stderr:
                return f"open failed: {proc.stderr.strip()}"

            deadline = time.monotonic() + delay + 5.0
            found_name: str | None = None
            while time.monotonic() < deadline:
                await asyncio.sleep(0.5)
                found_name = await asyncio.to_thread(_find_app, True)
                if found_name:
                    break

            if found_name:
                return (
                    f"Launched '{found_name}' — use activate_app('{found_name}') "
                    f"then wait_for_controls('{found_name}')"
                )
            running_apps = await asyncio.to_thread(tk._get_foreground_apps, 0)
            return (
                f"open command succeeded for '{app_name}' but app did not appear within "
                f"{delay + 5:.0f} s. The app may have crashed or require permissions.\n"
                f"Running apps: {', '.join(sorted(running_apps))}"
            )

        @tool("activate_app", args_schema=_ActivateAppInput)
        async def activate_app(app_name: str) -> str:
            """Bring an application window to the foreground.

            Always call this after open_app and before wait_for_controls or any
            keyboard action. Uses the exact localized name shown by list_apps.

            Example: activate_app('Calculator')
            """
            def _window_offscreen(pid: int) -> bool:
                root = AXUIElementCreateApplication(pid)
                win = _ax_get(root, kAXMainWindowAttribute)
                if win is None:
                    return False
                pos_raw = _ax_get(win, kAXPositionAttribute)
                if pos_raw is None:
                    return False
                wx, wy = _ax_point(pos_raw)
                sw, sh = _logical_screen_size()
                return wy < 0 or wy > sh or wx < -sw or wx > sw * 2

            async def _reposition_and_return(pid: int, display_name: str) -> str:
                if await asyncio.to_thread(_window_offscreen, pid):
                    await asyncio.to_thread(
                        subprocess.run,
                        ["osascript", "-e",
                         f'tell application "{display_name}" to set position of window 1 to {{100, 100}}'],
                        capture_output=True,
                    )
                    await asyncio.sleep(0.3)
                return f"Activated: {display_name} (frontmost confirmed)."

            ws = NSWorkspace.sharedWorkspace()
            display_name = ""
            pid = 0
            for app in ws.runningApplications():
                if str(app.localizedName() or "").lower() == app_name.lower():
                    display_name = str(app.localizedName() or app_name)
                    pid = app.processIdentifier()
                    break
            if not display_name:
                fg_apps = await asyncio.to_thread(tk._get_foreground_apps, 0)
                for name, p in fg_apps.items():
                    if name.lower() == app_name.lower():
                        display_name, pid = name, p
                        break

            if not display_name:
                fg_apps = await asyncio.to_thread(tk._get_foreground_apps, 0)
                running = sorted(
                    set(str(a.localizedName() or "") for a in ws.runningApplications()
                        if a.activationPolicy() <= 1 and (a.localizedName() or ""))
                    | set(fg_apps.keys())
                )
                return (
                    f"App '{app_name}' not found in running processes.\n"
                    f"Running apps: {', '.join(running)}\n"
                    f"Tip: use the exact name from list_apps()."
                )

            ok, front = await asyncio.to_thread(tk._activate_and_verify, display_name)
            if ok:
                return await _reposition_and_return(pid, display_name)

            # Focus did NOT move — be honest so the agent stops trying to type/click.
            return (
                f"Could NOT bring '{display_name}' to the foreground — '{front or 'another app'}' "
                f"still holds focus. Synthesized clicks and keystrokes will go to "
                f"'{front or 'that app'}', NOT '{display_name}'.\n"
                f"Do NOT use click_at / type_text / hotkey now — they will hit the wrong app.\n"
                f"read_screen('{display_name}') still works WITHOUT focus, so read and "
                f"report what is visible instead. Retrying activate_app rarely helps if a host "
                f"app keeps grabbing focus.\n"
                f"If '{display_name}' is an Electron app (Slack, Discord, Cursor, VS Code, "
                f"Notion), prefer the focus-free CDP path: read_app_dom('{display_name}') to "
                f"read it and click_app_dom('{display_name}', '<label>') to act on it — both "
                f"work while it stays in the background."
            )

        @tool("wait_for_controls", args_schema=_WaitForControlsInput)
        async def wait_for_controls(app_name: str, timeout: int = 10) -> str:
            """Poll the AX tree until an app has visible controls, then return them.

            Use immediately after open_app + activate_app instead of manually looping
            get_screen_controls. Populates the element registry on success.

            Each poll bypasses the scan cache (force=True) so it always reflects the
            live AX state as the app finishes launching.

            Example: wait_for_controls('Calculator')
            """
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                controls = await asyncio.to_thread(
                    tk._scan, app_filter=app_name, force=True,
                )
                if any(not c.get("_structural") for c in controls):
                    return tk._fmt(controls)
                await asyncio.sleep(1.0)
            return (
                f"Timed out after {timeout}s — '{app_name}' still has no visible AX controls.\n"
                f"Escalation options:\n"
                f"  1. open_app('/System/Applications/{app_name}.app') — try full path\n"
                f"  2. spotlight_search('{app_name}') → hotkey('return') — force re-launch\n"
                f"  3. list_apps() — check the exact localized name of the app"
            )

        # ── Element-based interaction ─────────────────────────────────────

        @tool("press_control", args_schema=_PressControlInput)
        async def press_control(index: int) -> str:
            """Trigger a UI control by its index from get_screen_controls.

            Uses AXUIElementPerformAction(kAXPressAction) — no mouse movement.
            Works for: Button, CheckBox, RadioButton, Link, MenuItem, PopUpButton.

            Call get_screen_controls or wait_for_controls first to populate the registry.
            Example: press_control(5)
            """
            elem = tk._element_registry.get(index)
            if elem is None:
                return (
                    f"Control #{index} not in registry. "
                    "Call get_screen_controls() or wait_for_controls() first."
                )
            before = await asyncio.to_thread(_press_state_token, elem)
            err = await asyncio.to_thread(AXUIElementPerformAction, elem, kAXPressAction)
            await asyncio.sleep(tk.action_delay)
            if err != 0:
                coords = tk._resolve_click_coords(index)
                if isinstance(coords, str):
                    return f"AXPress failed for #{index} (err={err}) and click fallback failed: {coords}"
                x, y = coords
                await asyncio.to_thread(pyautogui.click, x, y)
                await asyncio.sleep(tk.action_delay)
                return f"Pressed control #{index} (AXPress err={err}; clicked at ({x}, {y}) instead)"
            # AXPress reported success, but Chromium/Electron controls accept the
            # action and silently ignore it. Compare a cheap state token before
            # and after: if nothing changed AND the owning app is Electron (has a
            # live CDP port), surface the likely no-op instead of reporting false
            # success so the agent stops re-pressing the same row. Native Cocoa
            # apps press reliably and often leave the title unchanged on a valid
            # action, so we do not warn there to avoid false positives.
            after = await asyncio.to_thread(_press_state_token, elem)
            if before and after and before == after:
                pid = await asyncio.to_thread(_ax_pid, elem)
                cdp_port = await asyncio.to_thread(_find_cdp_port, pid) if pid else 0
                if cdp_port:
                    return (
                        f"Pressed control #{index}, but the app state did not change "
                        f"(window title/value identical before and after). This is an "
                        f"Electron/Chromium app (Slack, Discord, VS Code) whose controls "
                        f"accept AXPress but never fire the real click, so the press was "
                        f"likely ignored. Do NOT repeat this press. Use the focus-free "
                        f"CDP path instead: read_app_dom() to read the control's label, "
                        f"then click_app_dom(app, '<label>') — this clicks even when the "
                        f"app is NOT frontmost. (click({index}) also works but needs the "
                        f"app focused/visible.)"
                    )
            return f"Pressed control #{index}"

        @tool("type_into_control", args_schema=_TypeIntoControlInput)
        async def type_into_control(index: int, text: str, submit: bool = False) -> str:
            """Focus a text field by index then type into it.

            Focuses via AXUIElementSetAttributeValue(kAXFocusedAttribute, True) then
            types using the keyboard — no click needed.
            Works for: TextField, TextArea, ComboBox.

            Set submit=True to press Return after typing — do this for search boxes and
            single-line fields that submit on Enter. Leave submit=False for multi-line
            text areas or forms with a separate submit button.

            Example: type_into_control(3, 'hello world')
            Example: type_into_control(3, 'cats', submit=True)  # search box
            """
            elem = tk._element_registry.get(index)
            if elem is None:
                return (
                    f"Control #{index} not in registry. "
                    "Call get_screen_controls() first."
                )
            err = await asyncio.to_thread(
                AXUIElementSetAttributeValue, elem, kAXFocusedAttribute, True,
            )
            if err != 0:
                return f"Focus failed for #{index} (err={err})."
            await asyncio.sleep(0.15)
            # Prefer process-targeted CGEvent typing: it posts keystrokes to the
            # element's own process without touching global focus, so it stays
            # correct even if another agent steals the foreground mid-type. Fall
            # back to pyautogui's global keyboard if the pid is unknown or the
            # CGEvent APIs are unavailable.
            pid = await asyncio.to_thread(_ax_pid, elem)
            typed = await asyncio.to_thread(_type_to_pid, pid, text, submit)
            if not typed:
                await asyncio.to_thread(pyautogui.typewrite, text, interval=0.04)
                if submit:
                    await asyncio.to_thread(pyautogui.press, "return")
            await asyncio.sleep(tk.action_delay)
            suffix = " and pressed Return" if submit else ""
            return f"Typed {text!r} into control #{index}{suffix}"

        @tool("get_control_value", args_schema=_GetControlValueInput)
        async def get_control_value(index: int) -> str:
            """Read the current value or text of a control by its index.

            Reads kAXValueAttribute directly — no screenshot needed.
            Useful for: calculator display, text field contents, checkbox state.

            Example: get_control_value(2)
            """
            elem = tk._element_registry.get(index)
            if elem is None:
                return (
                    f"Control #{index} not in registry. "
                    "Call get_screen_controls() first."
                )
            val = await asyncio.to_thread(_ax_get, elem, kAXValueAttribute)
            return str(val) if val is not None else "(no value attribute on this element)"

        # ── Coordinate-based fallbacks ────────────────────────────────────

        @tool("click", args_schema=_ClickInput)
        async def click(index: int) -> str:
            """Click at a control's screen coordinates using the mouse.

            Looks up the control by index from get_screen_controls, then clicks
            at its centre point.  Use when press_control (AXPress) fails or for
            controls that only respond to mouse events.

            Example: click(5)
            """
            result = tk._resolve_click_coords(index)
            if isinstance(result, str):
                return result
            x, y = result
            await asyncio.to_thread(pyautogui.click, x, y)
            await asyncio.sleep(tk.action_delay)
            return f"Clicked control #{index} at ({x}, {y})"

        @tool("double_click", args_schema=_DoubleClickInput)
        async def double_click(index: int) -> str:
            """Double-click at a control's screen coordinates.

            Example: double_click(5)
            """
            result = tk._resolve_click_coords(index)
            if isinstance(result, str):
                return result
            x, y = result
            await asyncio.to_thread(pyautogui.doubleClick, x, y)
            await asyncio.sleep(tk.action_delay)
            return f"Double-clicked control #{index} at ({x}, {y})"

        @tool("right_click", args_schema=_RightClickInput)
        async def right_click(index: int) -> str:
            """Right-click at a control's screen coordinates to open a context menu.

            Example: right_click(5)
            """
            result = tk._resolve_click_coords(index)
            if isinstance(result, str):
                return result
            x, y = result
            await asyncio.to_thread(pyautogui.rightClick, x, y)
            await asyncio.sleep(tk.action_delay)
            return f"Right-clicked control #{index} at ({x}, {y})"

        @tool("scroll", args_schema=_ScrollInput)
        async def scroll(index: int, direction: str, amount: int = 300) -> str:
            """Scroll at a control's screen coordinates.

            Args:
                index:     Control index from get_screen_controls.
                direction: up | down | left | right
                amount:    Pixels to scroll (default 300).

            Example: scroll(5, 'down', 300)
            """
            result = tk._resolve_click_coords(index)
            if isinstance(result, str):
                return result
            x, y = result
            direction = direction.strip().lower()
            clicks = max(1, amount // 50)
            if direction in ("up", "down"):
                await asyncio.to_thread(pyautogui.scroll, clicks if direction == "up" else -clicks, x=x, y=y)
            else:
                await asyncio.to_thread(pyautogui.hscroll, clicks if direction == "right" else -clicks, x=x, y=y)
            await asyncio.sleep(tk.action_delay)
            return f"Scrolled {direction} {amount}px at control #{index} ({x}, {y})"

        # ── Electron structured read via CDP (preferred over OCR) ──────────

        @tool("read_app_dom", args_schema=_ReadAppDomInput)
        async def read_app_dom(app_name: str) -> str:
            """Read an Electron app's UI as structured text via Chrome DevTools.

            For Electron/Chromium apps (Slack, Discord, Cursor, VS Code, Notion)
            whose Accessibility API is DISABLED, this reads the full accessibility
            tree straight from Chromium — structured text with control roles, no
            screenshot and no vision tokens, and it works while the window is in
            the background. Prefer this over read_screen (OCR) when it is
            available.

            Requires the app to have been launched with a remote-debugging port.
            If none is detected this returns the exact command to enable it.

            Example: read_app_dom('Slack')
            """
            pid = await asyncio.to_thread(tk._resolve_pid, app_name)
            if not pid:
                return (
                    f"'{app_name}' is not running. open_app('{app_name}') first."
                )
            port = await asyncio.to_thread(_find_cdp_port, pid)
            if not port:
                return (
                    f"'{app_name}' was not launched with remote debugging, so its "
                    f"DOM cannot be read over CDP. To enable it: quit '{app_name}' "
                    f"completely, then relaunch with\n"
                    f"  open -a \"{app_name}\" --args --remote-debugging-port=9222\n"
                    f"and retry read_app_dom('{app_name}'). Otherwise fall back to "
                    f"read_screen('{app_name}') (OCR)."
                )
            text = await _cdp_read_ax_tree(port)
            if not text:
                return (
                    f"Connected to '{app_name}' on CDP port {port} but read no "
                    f"accessibility nodes. Fall back to read_screen('{app_name}')."
                )
            return (
                f"{app_name} (CDP port {port}):\n{text}\n"
                f"To act on these WITHOUT focus, use "
                f"click_app_dom('{app_name}', '<label>')."
            )

        @tool("click_app_dom", args_schema=_ClickAppDomInput)
        async def click_app_dom(app_name: str, text: str, role: str = "") -> str:
            """Click a control in an Electron app over CDP — works WITHOUT focus.

            The focus-free actuator for Electron/Chromium apps (Slack, Discord,
            Cursor, VS Code, Notion). Use this instead of click / click_at /
            click_text when activate_app reports the app could NOT be brought to
            the foreground (the host app holds focus): CDP delivers the click
            straight to the renderer, so the window does not need to be frontmost.

            Pair it with read_app_dom: read the DOM, then pass the control's
            visible label as *text* (add *role* to disambiguate duplicates).

            Example: click_app_dom('Slack', 'DMs', role='tab')
            """
            pid = await asyncio.to_thread(tk._resolve_pid, app_name)
            if not pid:
                return f"'{app_name}' is not running. open_app('{app_name}') first."
            port = await asyncio.to_thread(_find_cdp_port, pid)
            if not port:
                return (
                    f"'{app_name}' was not launched with remote debugging, so it "
                    f"cannot be clicked over CDP. Relaunch with\n"
                    f"  open -a \"{app_name}\" --args --remote-debugging-port=9222\n"
                    f"or fall back to click_text('{text}', '{app_name}') (needs focus)."
                )
            return await _cdp_click(port, text, role)

        # ── Vision OCR fallback (AX-disabled apps, e.g. Slack) ─────────────
        # These return TEXT, so they work for any model (no VLM required) and
        # let the agent READ and ACT on apps whose Accessibility API is off.

        async def _ocr_payload(
            app_name: str, want_image: bool,
        ) -> str | tuple[list[dict], str | None]:
            """Shared capture for the read tools.

            Returns an error string, or ``(results, png_b64 | None)`` where the
            base64 PNG is only present when *want_image* is set.
            """
            target = await asyncio.to_thread(tk._window_target, app_name)
            if app_name and target is None:
                return (
                    f"'{app_name}' has no on-screen window to read "
                    f"(it may be minimised, hidden, or on another Space). "
                    f"Call activate_app('{app_name}') to bring it forward, then retry. "
                    f"If it isn't running, open_app('{app_name}') first."
                )
            window_id, region = target
            results, b64 = await asyncio.to_thread(
                tk._capture_for_read, window_id, region, want_image,
            )
            if results is None:
                return (
                    "OCR is unavailable on this host "
                    "(pyobjc-framework-Vision is not installed)."
                )
            return results, b64

        async def _ocr(app_name: str) -> list[dict] | str:
            payload = await _ocr_payload(app_name, want_image=False)
            if isinstance(payload, str):
                return payload
            results, _b64 = payload
            return results

        @tool("read_screen", args_schema=_ReadScreenInput)
        async def read_screen(app_name: str = "") -> str:
            """Read on-screen text via OCR — the fallback for AX-disabled apps.

            Use when get_screen_controls reports the app's accessibility interface
            is DISABLED (e.g. Slack) or returns no controls. Captures the app's
            window (or the full screen when app_name is empty) and returns the
            recognised text in roughly top-to-bottom, left-to-right reading order.

            Returns actual on-screen text. Report ONLY what appears here — never
            invent content you cannot read.

            Example: read_screen('Slack')
            """
            results = await _ocr(app_name)
            if isinstance(results, str):
                return results
            label = app_name or "full screen"
            if not results:
                hint = (
                    f"No text recognised for {label}. The window may be blank or "
                    f"still loading"
                )
                if _screen_recording_denied():
                    return f"{hint}, but more likely: {_SCREEN_RECORDING_HINT}"
                return (
                    f"{hint}. If the window clearly has content, the host app may "
                    f"lack Screen Recording permission (the capture comes back "
                    f"blank). {_SCREEN_RECORDING_HINT}"
                )
            return _format_ocr_lines(results, label)

        @tool("read_screen", args_schema=_ReadScreenInput)
        async def read_screen_vision(app_name: str = "") -> list | str:
            """Read on-screen text via OCR AND return a screenshot to inspect.

            Vision-model variant of read_screen: returns the OCR text (literal,
            high-fidelity) PLUS the captured window image so you can also see
            icons, badges, colours, and layout the OCR text can't convey.

            Use when get_screen_controls reports the app's accessibility interface
            is DISABLED (e.g. Slack) or returns no controls. The text and image
            are the SAME occlusion-safe window capture, so they always agree.

            Report ONLY what you can actually read or see — never invent content.

            Example: read_screen('Slack')
            """
            payload = await _ocr_payload(app_name, want_image=True)
            if isinstance(payload, str):
                return payload
            results, b64 = payload
            text = _format_ocr_lines(results, app_name or "full screen")
            if b64 is None:
                return text
            return [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]

        @tool("find_text_on_screen", args_schema=_FindTextOnScreenInput)
        async def find_text_on_screen(text: str, app_name: str = "") -> str:
            """Locate text on screen via OCR and return its click coordinates.

            The fallback locator for AX-disabled apps: finds where *text* appears
            and returns logical screen coordinates you can pass to click_at(x, y).
            To type into a field, find its placeholder or label text, click_at it
            to focus, then type_text(...).

            Returns each match as 'TEXT' at (x, y) [confidence].

            Prefer click_text(text, app_name) when you just want to click matched
            text — it focuses the app first so the click can't land on the wrong app.

            Example: find_text_on_screen('Send', 'Slack') → click_at(x, y)
            """
            results = await _ocr(app_name)
            if isinstance(results, str):
                return results
            matches = _rank_text_matches(results, text)
            if not matches:
                where = app_name or "full screen"
                return (
                    f"No on-screen text matching {text!r} in {where}. "
                    f"Try read_screen('{app_name}') to see what is visible."
                )
            lines = [
                f"{r['text']!r} at ({r['x']}, {r['y']}) [conf {r['conf']:.2f}]"
                for r in matches[:5]
            ]
            return (
                f"Found {len(matches)} match(es) for {text!r}. "
                f"Best first — use click_at(x, y) on one of:\n" + "\n".join(lines)
            )

        @tool("click_text", args_schema=_ClickTextInput)
        async def click_text(text: str, app_name: str) -> str:
            """Focus an app, find on-screen text via OCR, and click it — atomically.

            The preferred actuator for AX-disabled apps (e.g. Slack). It first
            brings *app_name* to the foreground and CONFIRMS it is frontmost, so
            the click cannot land on the wrong app (the failure mode of separate
            activate_app + click_at calls). If focus cannot be obtained it does
            NOT click — it tells you to read instead.

            Example: click_text('Search', 'Slack')
            """
            ok, front = await asyncio.to_thread(tk._activate_and_verify, app_name)
            if not ok:
                return (
                    f"Did NOT click — could not focus '{app_name}' ('{front or 'another app'}' "
                    f"holds focus, so the click would hit the wrong app). "
                    f"Use read_screen('{app_name}') to read it without focus instead."
                )
            target = await asyncio.to_thread(tk._window_target, app_name)
            if target is None:
                return (
                    f"'{app_name}' is frontmost but has no readable window to locate "
                    f"text in (minimised or empty)."
                )
            window_id, region = target
            results = await asyncio.to_thread(tk._ocr_window, window_id, region)
            if results is None:
                return "OCR is unavailable (pyobjc-framework-Vision not installed)."
            matches = _rank_text_matches(results, text)
            if not matches:
                return (
                    f"Focused '{app_name}' but found no on-screen text matching {text!r}. "
                    f"Call read_screen('{app_name}') to see what is visible."
                )
            best = matches[0]
            await asyncio.to_thread(pyautogui.click, best["x"], best["y"])
            await asyncio.sleep(tk.action_delay)
            return (
                f"Focused '{app_name}' and clicked {best['text']!r} "
                f"at ({best['x']}, {best['y']})."
            )

        @tool("click_at", args_schema=_ClickAtInput)
        async def click_at(x: int, y: int) -> str:
            """Click at absolute screen coordinates (logical points).

            The coordinate-based actuator for AX-disabled apps. Use coordinates
            from find_text_on_screen. After clicking a text field to focus it,
            call type_text(...) to enter text.

            Example: click_at(640, 820)
            """
            front = await asyncio.to_thread(tk._frontmost_app_name)
            await asyncio.to_thread(pyautogui.click, x, y)
            await asyncio.sleep(tk.action_delay)
            focus = f" (frontmost app: '{front}')" if front else ""
            return f"Clicked at ({x}, {y}){focus}."

        @tool("double_click_at", args_schema=_ClickAtInput)
        async def double_click_at(x: int, y: int) -> str:
            """Double-click at absolute screen coordinates (logical points).

            Example: double_click_at(640, 820)
            """
            await asyncio.to_thread(pyautogui.doubleClick, x, y)
            await asyncio.sleep(tk.action_delay)
            return f"Double-clicked at ({x}, {y})."

        @tool("right_click_at", args_schema=_ClickAtInput)
        async def right_click_at(x: int, y: int) -> str:
            """Right-click at absolute screen coordinates to open a context menu.

            Example: right_click_at(640, 820)
            """
            await asyncio.to_thread(pyautogui.rightClick, x, y)
            await asyncio.sleep(tk.action_delay)
            return f"Right-clicked at ({x}, {y})."

        @tool("scroll_at", args_schema=_ScrollAtInput)
        async def scroll_at(x: int, y: int, direction: str, amount: int = 300) -> str:
            """Scroll at absolute screen coordinates (logical points).

            Args:
                x, y:      Screen point to scroll over.
                direction: up | down | left | right
                amount:    Pixels to scroll (default 300).

            Example: scroll_at(640, 500, 'down', 300)
            """
            direction = direction.strip().lower()
            clicks = max(1, amount // 50)
            if direction in ("up", "down"):
                await asyncio.to_thread(pyautogui.scroll, clicks if direction == "up" else -clicks, x=x, y=y)
            else:
                await asyncio.to_thread(pyautogui.hscroll, clicks if direction == "right" else -clicks, x=x, y=y)
            await asyncio.sleep(tk.action_delay)
            return f"Scrolled {direction} {amount}px at ({x}, {y})."

        # ── Keyboard ──────────────────────────────────────────────────────

        @tool("type_text", args_schema=_TypeTextInput)
        async def type_text(text: str, submit: bool = False, app_name: str = "") -> str:
            """Type text into the currently focused field.

            Prefer type_into_control(index, text) to target a specific field.
            For AX-disabled apps (e.g. Slack) pass app_name to FOCUS-GUARD the
            type: the app is activated and confirmed frontmost first, and if focus
            cannot be obtained NOTHING is typed (so keystrokes never leak into the
            wrong app). Without app_name, keystrokes go to whatever is frontmost.

            Set submit=True to press Return after typing — do this for search boxes and
            single-line fields that submit on Enter. Leave submit=False for multi-line
            text areas or forms with a separate submit button.
            """
            if app_name:
                ok, front = await asyncio.to_thread(tk._activate_and_verify, app_name)
                if not ok:
                    return (
                        f"Did NOT type — could not focus '{app_name}' ('{front or 'another app'}' "
                        f"holds focus, so keystrokes would land in the wrong app). "
                        f"Bring '{app_name}' to the front manually or read it with "
                        f"read_screen('{app_name}') instead."
                    )
                # With a confirmed target, post keystrokes straight to its pid so
                # they can't leak into another app even if focus drifts mid-type.
                pid = await asyncio.to_thread(tk._resolve_pid, app_name)
                if await asyncio.to_thread(_type_to_pid, pid, text, submit):
                    await asyncio.sleep(tk.action_delay)
                    suffix = " and pressed Return" if submit else ""
                    return (
                        f"Typed: {text!r}{suffix} into '{app_name}' "
                        f"(posted to pid {pid}, focus-free)."
                    )
            front = await asyncio.to_thread(tk._frontmost_app_name)
            await asyncio.to_thread(pyautogui.typewrite, text, interval=0.04)
            if submit:
                await asyncio.to_thread(pyautogui.press, "return")
            await asyncio.sleep(tk.action_delay)
            suffix = " and pressed Return" if submit else ""
            focus = f" (frontmost app: '{front}')" if front else ""
            return (
                f"Typed: {text!r}{suffix}{focus}. "
                f"Keystrokes go to the frontmost app — if that is not your target, "
                f"call activate_app() and retry."
            )

        @tool("hotkey", args_schema=_HotkeyInput)
        async def hotkey(keys: str) -> str:
            """Press a keyboard shortcut.

            Separate modifier keys with '+'.
            Examples:
              hotkey('command+space')    — Spotlight
              hotkey('command+tab')      — switch app
              hotkey('return')           — Enter
              hotkey('escape')           — Escape
              hotkey('command+a')        — select all
              hotkey('command+c')        — copy
              hotkey('command+v')        — paste
              hotkey('command+w')        — close window
              hotkey('command+q')        — quit app
            """
            parts = [k.strip().lower() for k in keys.replace("+", " ").split()]
            front = await asyncio.to_thread(tk._frontmost_app_name)
            await asyncio.to_thread(pyautogui.hotkey, *parts)
            await asyncio.sleep(tk.action_delay)
            focus = f" (frontmost app: '{front}')" if front else ""
            return (
                f"Hotkey: {keys}{focus}. "
                f"Shortcuts go to the frontmost app — if that is not your target, "
                f"call activate_app() and retry."
            )

        @tool("spotlight_search", args_schema=_SpotlightSearchInput)
        async def spotlight_search(query: str) -> str:
            """Open Spotlight and type a search query.

            Follow with hotkey('return') to open the top result.
            Example: spotlight_search('Calculator') → hotkey('return')
            """
            await asyncio.to_thread(pyautogui.hotkey, "command", "space")
            await asyncio.sleep(0.5)
            await asyncio.to_thread(pyautogui.typewrite, query, interval=0.05)
            await asyncio.sleep(0.3)
            return f"Spotlight: {query}"

        # ── Batch actions ─────────────────────────────────────────────────

        batch_dispatch: dict[str, object] = {
            "press_control": press_control,
            "type_into_control": type_into_control,
            "get_control_value": get_control_value,
            "click": click,
            "double_click": double_click,
            "right_click": right_click,
            "scroll": scroll,
            "type_text": type_text,
            "hotkey": hotkey,
        }

        @tool("batch_actions", args_schema=_BatchActionsInput)
        async def batch_actions(steps: str) -> str:
            """Execute a sequence of steps in one call without re-scanning between them.

            Pass steps as plain text, one per line (max 10). Each line is a tool
            call with keyword arguments, using the same syntax as individual calls.

            Allowed tools:
              press_control, type_into_control, get_control_value,
              click, double_click, right_click, scroll, type_text, hotkey

            Examples:
              batch_actions(steps="press_control(index=7)\\npress_control(index=3)")

              batch_actions(steps="type_into_control(index=99, text='user@example.com')\\nclick(index=105)\\ntype_into_control(index=105, text='Subject line')")

              batch_actions(steps="click(index=5)\\nscroll(index=10, direction='down', amount=300)")

              batch_actions(steps="type_into_control(index=4, text='cats', submit=True)")  # search + Enter
            """
            parsed = _parse_batch_actions(steps)
            if not parsed:
                return (
                    "No valid steps parsed. If batch_actions keeps failing, "
                    "call each tool individually instead.\n"
                    "Format: one tool call per line, e.g.:\n"
                    "press_control(index=5)\ntype_into_control(index=3, text='hello')"
                )

            _MAX_BATCH = 10
            if len(parsed) > _MAX_BATCH:
                parsed = parsed[:_MAX_BATCH]

            results: list[str] = []
            for i, (tool_name, args) in enumerate(parsed):
                tool_fn = batch_dispatch.get(tool_name)
                if tool_fn is None:
                    results.append(
                        f"[{i + 1}] ERROR: unknown tool '{tool_name}'. "
                        f"Allowed: {sorted(batch_dispatch)}"
                    )
                    continue
                try:
                    result = await tool_fn.ainvoke(args)
                    results.append(f"[{i + 1}] {tool_name}: {result}")
                except Exception as exc:
                    results.append(f"[{i + 1}] {tool_name}: ERROR — {exc}")

            return "\n".join(results)

        # ── Launch (composite) ────────────────────────────────────────────

        @tool("launch_app", args_schema=_LaunchAppInput)
        async def launch_app(app_name: str, timeout: int = 10) -> str:
            """Launch an app, activate it, and wait for AX controls — all in one step.

            Preferred over calling open_app → activate_app → wait_for_controls
            separately, as it avoids two extra LLM round-trips.

            Works for any app name accepted by open_app (short name, full .app path,
            or URL).  URLs are opened but not activated/waited (they open in the
            default browser which is already running).

            Examples:
              launch_app('Calculator')
              launch_app('TextEdit')
              launch_app('/System/Applications/Calculator.app')
            """
            is_url = app_name.startswith(("http://", "https://", "file://"))
            if is_url:
                return await open_app.ainvoke({"app_name": app_name})

            open_result = await open_app.ainvoke({"app_name": app_name})
            if "failed" in open_result.lower():
                return open_result

            import re as _re
            m = _re.search(r"Launched '([^']+)'", open_result)
            resolved = m.group(1) if m else app_name

            activate_result = await activate_app.ainvoke({"app_name": resolved})
            if "not found" in activate_result.lower():
                return f"Launched but could not activate '{resolved}': {activate_result}"

            controls = await wait_for_controls.ainvoke({"app_name": resolved, "timeout": timeout})
            return controls

        # ── Assemble tool lists ───────────────────────────────────────────

        tools = [
            list_apps,
            get_screen_controls,
            wait_for_controls,
            press_control,
            type_into_control,
            get_control_value,
            batch_actions,
            launch_app,
            open_app,
            activate_app,
            click,
            double_click,
            right_click,
            scroll,
            read_app_dom,
            click_app_dom,
            read_screen,
            find_text_on_screen,
            click_text,
            click_at,
            double_click_at,
            right_click_at,
            scroll_at,
            type_text,
            hotkey,
            spotlight_search,
        ]

        vision_tools = [
            capture_app_screenshot,
        ]

        # Vision-combo read_screen (text + screenshot). Swapped in for the
        # text-only read_screen by the navigator when the model is vision-capable.
        self.read_screen_vision = read_screen_vision

        return tools, vision_tools, batch_dispatch


# ═══════════════════════════════════════════════════════════════════════════════
# Default singleton + backward-compatible module-level exports
# ═══════════════════════════════════════════════════════════════════════════════

_default_toolkit = MacOSToolkit()

MACOS_TOOLS = _default_toolkit.tools
MACOS_VISION_TOOLS = _default_toolkit.vision_tools
MACOS_READ_SCREEN_VISION = _default_toolkit.read_screen_vision
