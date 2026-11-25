"""Structured desktop tools: one call reads or sets what took many before.

``observe`` says where the agent is (which app, which window and file, any dialog or alert on top). The menu,
dialog and table tools work through the app's Accessibility tree: ``menu_select`` picks a menu item by its path,
``read_dialog`` / ``fill_dialog`` read and set a dialog's fields in one call each, ``read_table`` reads the cells on
screen, ``get_control_values`` reads several controls, ``wait_until`` waits for a window, dialog or text, and
``app_guide`` returns the app's card (``macos_app_cards``).

The logic is in ``macos_ax_struct`` (no macOS imports, tested with fake trees). This module binds it to the live
system through :class:`LiveEnv`; tests pass their own env. The toolkit adds these tools to its list in
``MacOSToolkit._build_tools``.
"""

from __future__ import annotations

import asyncio
import json
import posixpath
import re
import time
from typing import Any, Callable
from urllib.parse import unquote, urlparse

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from . import macos_ax_struct as axs

# The file-name box of Save / Export sheets, under the labels apps and agents give it.
_NAME_FIELD = re.compile(r"\b(save as|export as|file ?name|name)\b", re.I)
from .macos_app_cards import card_for, card_key, known_apps


# ── Live system binding ─────────────────────────────────────────────────────

class LiveAccess:
    """:class:`macos_ax_struct.Access` over the real Accessibility API, through the toolkit's IPC helpers (which
    time out instead of hanging on a stuck app)."""

    def __init__(self, tk: Any):
        self.tk = tk

    def get(self, el: Any, attr: str) -> Any:
        return self.tk._ax_get_safe(el, attr)

    def children(self, el: Any) -> list:
        return list(self.tk._ax_get_safe(el, "AXChildren") or [])

    def set(self, el: Any, attr: str, value: Any) -> bool:
        return self.tk._ax_set_safe(el, attr, value) == 0

    def act(self, el: Any, action: str) -> bool:
        from ApplicationServices import AXUIElementPerformAction
        fut = self.tk._ax_thread_pool.submit(AXUIElementPerformAction, el, action)
        try:
            return fut.result(timeout=max(self.tk.ax_ipc_timeout, 2.0)) == 0
        except Exception:
            return False

    def escape(self) -> None:
        import pyautogui
        pyautogui.press("esc")
        time.sleep(0.08)

    def same(self, a: Any, b: Any) -> bool:
        try:
            return bool(a == b)
        except Exception:
            return False


class LiveEnv:
    """What the tools need from the system besides the AX tree."""

    def __init__(self, tk: Any):
        self.tk = tk
        self.access = LiveAccess(tk)

    def running_apps(self) -> list[dict]:
        from AppKit import NSWorkspace
        out = []
        for app in NSWorkspace.sharedWorkspace().runningApplications():
            if app.activationPolicy() != 0:          # regular apps only (Dock icon)
                continue
            name = str(app.localizedName() or "")
            if not name:
                continue
            exe = app.executableURL()
            out.append({
                "name": name,
                "pid": int(app.processIdentifier()),
                "bundle_id": str(app.bundleIdentifier() or ""),
                "executable": str(exe.lastPathComponent()) if exe is not None else "",
                "active": bool(app.isActive()),
            })
        return out

    def app_element(self, pid: int) -> Any:
        from ApplicationServices import AXUIElementCreateApplication
        return AXUIElementCreateApplication(pid)

    def window_list(self) -> list[dict]:
        try:
            from Quartz import (CGWindowListCopyWindowInfo, kCGNullWindowID,
                                kCGWindowListExcludeDesktopElements, kCGWindowListOptionOnScreenOnly)
        except Exception:
            return []
        info = CGWindowListCopyWindowInfo(
            kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements, kCGNullWindowID) or []
        return [dict(w) for w in info]

    def frame(self, el: Any) -> tuple[int, int, int, int] | None:
        from .macos_tools import _ax_point, _ax_size
        pos, size = self.tk._ax_get_safe(el, "AXPosition"), self.tk._ax_get_safe(el, "AXSize")
        if pos is None or size is None:
            return None
        (x, y), (w, h) = _ax_point(pos), _ax_size(size)
        return x, y, w, h

    def activate(self, name: str) -> bool:
        ok, _ = self.tk._activate_and_verify(name)
        return ok

    def press_key(self, key: str) -> None:
        import pyautogui
        pyautogui.press(key)

    def replace_text(self, el: Any, text: str) -> None:
        """Type over the focused field's text with the keyboard, then Tab out so the app commits it."""
        import pyautogui

        from .macos_tools import _ax_pid, _type_to_pid
        pyautogui.hotkey("command", "a")
        time.sleep(0.05)
        if not _type_to_pid(_ax_pid(el), text, False):
            pyautogui.typewrite(text, interval=0.03)
        time.sleep(0.05)
        pyautogui.press("tab")

    def type_text(self, pid: int, text: str) -> None:
        """Type ``text`` into process ``pid`` (focus-free when possible)."""
        import pyautogui

        from .macos_tools import _type_to_pid
        if not _type_to_pid(pid, text, False):
            pyautogui.typewrite(text, interval=0.03)

    def hotkey(self, *keys: str) -> None:
        import pyautogui
        pyautogui.hotkey(*keys)

    def set_selection(self, el: Any, start: int, length: int) -> bool:
        """Select characters ``start`` .. ``start + length`` of a text element (AXSelectedTextRange)."""
        try:
            import ApplicationServices as AS
            kind = getattr(AS, "kAXValueTypeCFRange", None) or AS.kAXValueCFRangeType
            value = AS.AXValueCreate(kind, (int(start), int(length)))
        except Exception:  # pragma: no cover - depends on host frameworks
            return False
        return self.tk._ax_set_safe(el, "AXSelectedTextRange", value) == 0

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


# ── Shared helpers ──────────────────────────────────────────────────────────

def _clip(v: Any, n: int = 80) -> str:
    s = " ".join(str(v).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _doc_path(raw: Any) -> str:
    """AXDocument is a file URL ('file:///Users/me/a.xlsx'); return the path."""
    s = str(raw or "")
    if s.startswith("file://"):
        return unquote(urlparse(s).path)
    return s


def _pick_app(apps: list[dict], name: str) -> dict | None:
    """The running app ``name`` means: its shown name, process name or bundle id; the frontmost app when empty."""
    if not name:
        return next((a for a in apps if a["active"]), apps[0] if apps else None)
    want = axs.norm_label(name)
    for key in ("name", "executable", "bundle_id"):
        for a in apps:
            if axs.norm_label(a.get(key, "")) == want:
                return a
    idx, _ = axs.match_label(name, [a["name"] for a in apps])
    return apps[idx] if idx is not None else None


def _not_running(name: str, apps: list[dict]) -> str:
    names = ", ".join(sorted(a["name"] for a in apps)) or "none"
    return f"'{name}' is not running. Running apps: {names}. Open it with open_app('{name}')."


def _parse_fields(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


# ── Schemas ─────────────────────────────────────────────────────────────────

class _ObserveInput(BaseModel):
    app_name: str = Field(default="", description="App to describe. Empty = the frontmost app.")


class _MenuSelectInput(BaseModel):
    app_name: str = Field(description="App whose menu bar to use, as observe() or list_apps() names it.")
    path: str = Field(min_length=1, description="Menu titles from the menu bar down, separated by '>', e.g. "
                      "'Format > Cells…' or 'Image > Mode > Grayscale'. Trailing '…' and case do not matter.")


class _MenuSearchInput(BaseModel):
    app_name: str = Field(description="App whose menus to search.")
    query: str = Field(min_length=1, description="Words to look for in menu item titles, e.g. 'freeze' or 'export pdf'.")


class _ReadDialogInput(BaseModel):
    app_name: str = Field(default="", description="App that shows the dialog. Empty = the frontmost app.")


class _FillDialogInput(BaseModel):
    app_name: str = Field(default="", description="App that shows the dialog. Empty = the frontmost app.")
    fields: dict[str, Any] = Field(default_factory=dict, description=(
        "Field label -> value, labels as read_dialog shows them. Text fields take text; checkboxes take "
        "true/false; pop-ups and radio buttons take the option's title (or true for a radio button); a tab "
        "takes true to switch to it. Example: {\"Decimal places\": \"2\", \"Negative numbers red\": true}"))
    press: str = Field(default="", description="Button to press after setting the fields, e.g. 'OK'. Empty = none.")
    path: str = Field(default="", description=(
        "Open or Save dialog only: the folder (or file) to go to first, e.g. '~/Desktop' or "
        "'~/Documents/report.pdf'. Typed into the dialog's Go to Folder (⇧⌘G) box, so no sidebar clicks are needed. "
        "Then set the name with fields={'Save As': 'name.pdf'} and press='Save'."))

    @field_validator("fields", mode="before")
    @classmethod
    def _json_fields(cls, v: Any) -> Any:
        return _parse_fields(v)


class _WriteCellsInput(BaseModel):
    app_name: str = Field(default="", description="Spreadsheet app (LibreOffice). Empty = the frontmost app.")
    anchor: str = Field(min_length=2, description="Top-left cell of the block, e.g. 'B2'.")
    rows: list[list[Any]] = Field(min_length=1, max_length=200, description=(
        "The block, one inner list per row, left to right: [[\"Year\", \"Total\"], [2019, \"=SUM(B2:B5)\"]]. "
        "Values and formulas are typed as written; null or \"\" leaves a cell as it is."))

    @field_validator("rows", mode="before")
    @classmethod
    def _json_rows(cls, v: Any) -> Any:
        return _parse_fields(v)


class _SelectTextInput(BaseModel):
    app_name: str = Field(default="", description="App with the document (e.g. LibreOffice). Empty = frontmost.")
    text: str = Field(min_length=1, description="The exact text to select, as it appears in the document.")
    occurrence: int = Field(default=1, ge=1, le=200, description="Which match to select when it occurs "
                            "more than once (1 = the first, in reading order).")
    paragraph: bool = Field(default=False, description="Select the whole paragraph (or field) that holds the "
                            "text, instead of the text alone.")


class _ReadTableInput(BaseModel):
    app_name: str = Field(default="", description="App to read. Empty = the frontmost app.")
    range_ref: str = Field(default="", description="Spreadsheet range to show, e.g. 'A1:F20'. Empty = every "
                           "cell on screen.")


class _GetControlValuesInput(BaseModel):
    indices: list[int] = Field(min_length=1, max_length=80, description="Control indices from the last "
                               "get_screen_controls or read_dialog, e.g. [4, 7, 12].")

    @field_validator("indices", mode="before")
    @classmethod
    def _json_indices(cls, v: Any) -> Any:
        return _parse_fields(v)


class _WaitUntilInput(BaseModel):
    app_name: str = Field(default="", description="App to watch. Empty = the frontmost app.")
    title: str = Field(default="", description="Wait until a window title contains this.")
    text: str = Field(default="", description="Wait until this text shows in the front window.")
    dialog: str = Field(default="", description="'open' to wait for a dialog or sheet, 'closed' for it to go.")
    timeout: float = Field(default=10, ge=0.5, le=60, description="Seconds to wait at most.")


class _AppGuideInput(BaseModel):
    app: str = Field(min_length=1, description="App name, e.g. 'LibreOffice', 'GIMP', 'Google Chrome'.")


# ── The tools ───────────────────────────────────────────────────────────────

MENU_CACHE_TTL = 60.0


def build_structured_tools(tk: Any, env: Any | None = None) -> dict[str, Any]:
    """The structured tools bound to toolkit ``tk``, by name."""
    env = env or LiveEnv(tk)
    ax = env.access
    menu_cache: dict[int, tuple[float, list[axs.MenuHit]]] = {}

    def run(fn: Callable[[], str]) -> Any:
        return asyncio.to_thread(fn)

    def target(app_name: str) -> tuple[dict | None, list[dict], str]:
        apps = env.running_apps()
        app = _pick_app(apps, app_name)
        if app is None:
            return None, apps, _not_running(app_name, apps) if app_name else "No app is running."
        return app, apps, ""

    def front_window(app_el: Any) -> Any:
        return ax.get(app_el, "AXFocusedWindow") or ax.get(app_el, "AXMainWindow") or next(
            iter(ax.get(app_el, "AXWindows") or []), None)

    def ensure_front(app: dict) -> str:
        if app["active"]:
            return ""
        if env.activate(app["name"]):
            app["active"] = True
            return ""
        return (f"Could not bring '{app['name']}' to the front; another window holds focus. "
                "Dismiss it (observe() shows what is on top) and try again.")

    def register(controls: list[axs.Control]) -> None:
        """Give dialog controls indices after the last scan's, so press_control / type_into_control /
        get_control_value / click work on them too."""
        reg = tk._element_registry
        base = max(reg, default=0)
        for n, c in enumerate(controls, start=1):
            c.index = base + n
            reg[c.index] = c.el
            f = env.frame(c.el)
            if f:
                x, y, w, h = f
                tk._coord_registry[c.index] = (x + w // 2, y + h // 2)

    def dialog_brief(app_el: Any) -> str:
        d = axs.find_dialog(ax, app_el)
        if d is None:
            return ""
        el, desc = d
        return axs.describe_dialog_brief(desc, axs.collect_controls(ax, el, max_nodes=250),
                                         messages=axs.dialog_messages(ax, el))

    # observe ---------------------------------------------------------------

    def _observe(app_name: str) -> str:
        app, apps, err = target(app_name)
        if app is None:
            return err
        el = env.app_element(app["pid"])
        lines = []
        state = "frontmost" if app["active"] else "not frontmost"
        lines.append(f"App: {app['name']} ({state}; process '{app['executable'] or app['name']}', "
                     f"bundle {app['bundle_id'] or '?'})")
        wins = list(ax.get(el, "AXWindows") or [])
        focused, main = ax.get(el, "AXFocusedWindow"), ax.get(el, "AXMainWindow")
        if not wins:
            lines.append("Windows: none open (the app may be starting, or every window is closed)")
        else:
            lines.append("Windows:")
            for w in wins[:8]:
                tags = [t for t, other in (("focused", focused), ("main", main))
                        if other is not None and ax.same(w, other)]
                if ax.get(w, "AXMinimized") is True:
                    tags.append("minimized")
                doc = _doc_path(ax.get(w, "AXDocument"))
                s = f"  - '{_clip(ax.get(w, 'AXTitle') or '', 100)}'"
                if tags:
                    s += f" ({', '.join(tags)})"
                if doc:
                    s += f" file: {doc}"
                lines.append(s)
        fe = ax.get(el, "AXFocusedUIElement")
        if fe is not None:
            role = str(ax.get(fe, "AXRole") or "")[2:]
            label = str(ax.get(fe, "AXTitle") or ax.get(fe, "AXDescription") or "")
            val = ax.get(fe, "AXValue")
            s = f"Focus: {role}" + (f" '{_clip(label, 60)}'" if label else "")
            if val is not None and not isinstance(val, (list, tuple)):
                s += f" = {_clip(val, 60)!r}"
            lines.append(s)
        brief = dialog_brief(el)
        lines.append(f"Dialog: {brief} — read_dialog() lists its fields" if brief else "Dialog: none")
        if app["active"]:
            above, notices = axs.split_notices(axs.windows_above(env.window_list(), app["name"], app["pid"]))
            if above:
                lines.append("On top of it: " + "; ".join(f"{o} '{t}'" if t else o for o, t in above)
                             + " — deal with this first; clicks and keys may land there")
            if notices:
                lines.append("Also on screen: " + "; ".join(f"{o} '{t}'" if t else o for o, t in notices)
                             + f" — {axs.NOTICE_NOTE}")
        others = [a["name"] for a in apps if a["pid"] != app["pid"]]
        if others:
            lines.append("Other running apps: " + ", ".join(sorted(others)))
        if card_key(app["name"]) or card_key(app["executable"]):
            lines.append(f"app_guide('{app['name']}') has notes on working this app.")
        return "\n".join(lines)

    @tool("observe", args_schema=_ObserveInput)
    async def observe(app_name: str = "") -> str:
        """Where you are, in one call: the app (frontmost by default) with its process name and bundle id, its
        windows with the file each shows, the focused field, any open dialog or sheet, alerts drawn over it, and
        the other running apps. Call it first, and whenever you are unsure what is on screen."""
        return await run(lambda: _observe(app_name))

    # menus -----------------------------------------------------------------

    def _menu_select(app_name: str, path: str) -> str:
        app, _, err = target(app_name)
        if app is None:
            return err
        parts = axs.parse_menu_path(path)
        if blocked := ensure_front(app):
            return blocked
        el = env.app_element(app["pid"])
        bar = ax.get(el, "AXMenuBar")
        if bar is None:
            return f"'{app['name']}' exposes no menu bar through Accessibility."
        res = axs.select_menu(ax, bar, parts)
        if not res.ok:
            msg = f"Menu not chosen: {res.error}."
            if res.options:
                msg += "\nItems there: " + " | ".join(res.options[:40])
            if not res.chosen or "no item" in res.error:
                msg += f"\nmenu_search('{app['name']}', '<word>') finds an item anywhere in the menus."
            return msg
        menu_cache.pop(app["pid"], None)
        tk._invalidate_scan_cache(app["name"])
        brief = ""
        for _ in range(6):                      # a dialog the command opens takes a moment to appear
            env.sleep(0.25)
            brief = dialog_brief(el)
            if brief:
                break
        out = f"Chose {' ▸ '.join(res.chosen)} in {app['name']}."
        if brief:
            out += f"\nNow open: {brief}. read_dialog() lists its fields; fill_dialog() sets them."
        return out

    @tool("menu_select", args_schema=_MenuSelectInput)
    async def menu_select(app_name: str, path: str) -> str:
        """Choose a menu item by its path, e.g. menu_select('LibreOffice', 'Format > Cells…'). Works the app's
        own menu bar through Accessibility, so it needs no AppleScript. Says which dialog opened, if any. If a
        title is wrong it lists the items that are there."""
        return await run(lambda: _menu_select(app_name, path))

    def _menu_search(app_name: str, query: str) -> str:
        app, _, err = target(app_name)
        if app is None:
            return err
        cached = menu_cache.get(app["pid"])
        if cached and time.monotonic() - cached[0] < MENU_CACHE_TTL:
            hits = cached[1]
        else:
            if blocked := ensure_front(app):
                return blocked
            bar = ax.get(env.app_element(app["pid"]), "AXMenuBar")
            if bar is None:
                return f"'{app['name']}' exposes no menu bar through Accessibility."
            hits = axs.walk_menus(ax, bar)
            menu_cache[app["pid"]] = (time.monotonic(), hits)
        found = axs.search_menus(hits, query)
        if not found:
            return (f"No menu item in {app['name']} matches '{query}' ({len(hits)} items searched). Try another "
                    "word; some commands live in dialogs or the app's own search.")
        lines = [f"{len(found)} menu item(s) in {app['name']} matching '{query}' (pass a path to menu_select):"]
        lines += [f"  {h.line()}" for h in found]
        return "\n".join(lines)

    @tool("menu_search", args_schema=_MenuSearchInput)
    async def menu_search(app_name: str, query: str) -> str:
        """Find menu items by words in their titles, across all of an app's menus and submenus, with their
        full paths and keyboard shortcuts, e.g. menu_search('GIMP', 'grayscale')."""
        return await run(lambda: _menu_search(app_name, query))

    # dialogs ---------------------------------------------------------------

    def dialog_or_window(app: dict) -> tuple[Any, str, bool] | None:
        el = env.app_element(app["pid"])
        d = axs.find_dialog(ax, el)
        if d is not None:
            return d[0], d[1], True
        w = front_window(el)
        if w is None:
            return None
        return w, f"window '{_clip(ax.get(w, 'AXTitle') or '', 100)}'", False

    def _read_dialog(app_name: str) -> str:
        app, _, err = target(app_name)
        if app is None:
            return err
        found = dialog_or_window(app)
        if found is None:
            return f"{app['name']} has no open window."
        root, desc, is_dialog = found
        controls = axs.collect_controls(ax, root)
        register(controls)
        text = axs.format_dialog(desc, controls, axs.dialog_messages(ax, root) if is_dialog else ())
        if not is_dialog:
            text = f"No dialog or sheet is open in {app['name']}; these are the front window's fields.\n" + text
        return text + ("\nSet several fields with fill_dialog(fields={...}, press='OK'); the indices also work "
                       "with press_control and type_into_control.")

    @tool("read_dialog", args_schema=_ReadDialogInput)
    async def read_dialog(app_name: str = "") -> str:
        """Read the open dialog or sheet (or the front window when none is open): its tabs, every field with
        its label and current value, and its buttons, each with an index. Use it right after a command opens a
        dialog, instead of reading controls one at a time."""
        return await run(lambda: _read_dialog(app_name))

    def set_popup(c: axs.Control, value: str) -> str:
        if not ax.act(c.el, "AXPress"):
            return "could not open the pop-up"
        env.sleep(0.2)
        menus = [k for k in ax.children(c.el) if str(ax.get(k, "AXRole") or "") == "AXMenu"]
        items = axs.menu_items(ax, menus[0]) if menus else []
        titles = [t for t, _ in items]
        if not titles:
            # The menu is not in the tree (LibreOffice list boxes): close it, then step through the options with
            # the arrow keys on the focused pop-up, reading its value after each key.
            ax.escape()
            if not ax.set(c.el, "AXFocused", True):
                return "no option list could be read, and the pop-up did not take focus"

            def step(key: str) -> None:
                env.press_key(key)
                env.sleep(0.08)
            chosen, seen = axs.cycle_to_option(lambda: ax.get(c.el, "AXValue") or ax.get(c.el, "AXTitle"),
                                               step, value)
            if chosen is not None:
                return ""
            return f"no option '{value}' (options: {' | '.join(o for o in seen[:30] if o) or 'none readable'})"
        i, tied = axs.match_label(value, titles)
        if i is None:
            ax.escape()
            opts = [titles[j] for j in tied] if tied else titles
            return f"no option '{value}' (options: {' | '.join(opts[:30]) or 'none readable'})"
        if not ax.act(items[i][1], "AXPress"):
            ax.escape()
            return f"could not choose '{titles[i]}'"
        return ""

    def set_one(c: axs.Control, value: Any) -> str:
        """Set one control; '' on success, else why not."""
        if not c.enabled:
            return "the field is disabled"
        if c.kind in ("text", "combo"):
            ax.set(c.el, "AXFocused", True)
            if ax.set(c.el, "AXValue", str(value)) and str(ax.get(c.el, "AXValue")) == str(value):
                return ""
            # LibreOffice's spin fields accept AXValue and keep their old text; typing works.
            env.replace_text(c.el, str(value))
            env.sleep(0.2)
            now = ax.get(c.el, "AXValue")
            if now is not None and str(now) != str(value):
                return f"the field shows {str(now)!r} after setting and typing"
            return ""
        if c.kind == "slider" or c.kind == "stepper":
            try:
                num = float(value)
            except (TypeError, ValueError):
                return f"needs a number, got {value!r}"
            return "" if ax.set(c.el, "AXValue", num) else "the app refused the value"
        if c.kind in ("checkbox", "radio", "tab"):
            want = axs.coerce_bool(value)
            if want is None:
                return f"needs true or false, got {value!r}"
            on = c.value == "on"
            if want == on:
                return ""
            if c.kind != "checkbox" and not want:
                return "a radio button or tab is turned off by choosing another one"
            if c.kind == "tab":
                return "" if ax.act(c.el, "AXPress") else "the press was not accepted"

            def is_on() -> bool | None:
                return axs.coerce_bool(ax.get(c.el, "AXValue"))
            # LibreOffice accepts AXPress on its checkboxes and radio buttons and ignores it; Space on the focused
            # control works. So each way is checked by reading the state back.
            ax.act(c.el, "AXPress")
            env.sleep(0.15)
            if is_on() is want:
                return ""
            if ax.set(c.el, "AXFocused", True):
                env.press_key("space")
                env.sleep(0.15)
                if is_on() is want:
                    return ""
            return f"the app did not change it (still {'on' if on else 'off'})"
        if c.kind == "popup":
            return set_popup(c, str(value))
        if c.kind == "list":
            titles = [t for t, _ in c.items]
            i, tied = axs.match_label(str(value), titles)
            if i is None:
                opts = [titles[j] for j in tied] if tied else titles
                return f"no option '{value}' (options: {' | '.join(opts[:30])})"
            item = c.items[i][1]
            # LibreOffice's list boxes take AXSelectedChildren on the list; AXSelected on an item and AXPress are
            # accepted and ignored there, so each way is checked by reading the selection back.
            for attempt in (lambda: ax.set(c.el, "AXSelectedChildren", [item]),
                            lambda: ax.set(item, "AXSelected", True),
                            lambda: ax.act(item, "AXPress")):
                if attempt() and ax.get(item, "AXSelected") is True:
                    return ""
            return f"the app did not select '{titles[i]}' (still {c.value!r})"
        return f"cannot set a {c.kind}"

    def _focused(pid: int) -> Any:
        try:
            return ax.get(env.app_element(pid), "AXFocusedUIElement")
        except Exception:  # noqa: BLE001 — focus unknown: the caller goes on as before
            return None

    def _go_to_path(app: dict, path: str) -> str:
        """Type ``path`` into the Open/Save dialog's Go to Folder box (⇧⌘G). '' on success, else why not.

        When focus is readable and ⇧⌘G left it where it was, the box did not open (an Automator Save sheet,
        2026-11-24): typing then would put the path into the name field and Return would save under that name."""
        why = ensure_front(app)
        if why:
            return why
        before = _focused(app["pid"])
        env.hotkey("command", "shift", "g")
        env.sleep(0.6)
        after = _focused(app["pid"])
        if before is not None and after is not None and before == after:
            return (f"Go to Folder (⇧⌘G) did not open in {app['name']}, so nothing was typed (it would have gone "
                    "into the name field). Pick the folder in the dialog's sidebar or Where: menu instead.")
        env.type_text(app["pid"], path)
        env.sleep(0.3)
        env.press_key("return")
        env.sleep(0.8)
        return ""

    def _fill_dialog(app_name: str, fields: dict[str, Any], press: str, path: str = "") -> str:
        app, _, err = target(app_name)
        if app is None:
            return err
        # A full path given as the file name: the folder goes to ``path``, the name to the field. Typed whole, the
        # name field takes it literally (a file called '~:Library:Services:Hello.workflow').
        split_note = ""
        for label, value in list(fields.items()):
            if isinstance(value, str) and "/" in value.rstrip("/") and _NAME_FIELD.search(str(label)):
                folder, name = posixpath.split(value.rstrip("/"))
                if folder and name:
                    path = path or folder
                    fields[label] = name
                    split_note = f"Took the folder {folder!r} from the name {value!r}; the name is {name!r}.\n"
        went = split_note
        if path.strip():
            why = _go_to_path(app, path.strip())
            if why:
                return why
            went += f"Went to {path.strip()!r} with Go to Folder (⇧⌘G).\n"
            if not fields and not press:
                after = dialog_brief(env.app_element(app["pid"]))
                return went + (f"Now open: {after}." if after else "No dialog is open now.")
        found = dialog_or_window(app)
        if found is None:
            return f"{app['name']} has no open window."
        root, desc, _ = found
        controls = axs.collect_controls(ax, root)
        register(controls)
        fieldish = [c for c in controls if c.kind != "button"]
        lines = ([went.rstrip()] if went else []) + [f"In {desc}:"]
        failed = 0
        radios = {"radio", "tab"}
        for label, value in fields.items():
            c, tied = axs.pick_control(fieldish, label)
            if c is None and not tied:
                c = _name_field(fieldish, str(label))
            if c is None and isinstance(value, str):
                # {"Alignment": "Center"} where the choices are radio buttons titled by their option.
                c, tied = axs.pick_control([x for x in fieldish if x.kind in radios], value)
                value = True if c is not None else value
            if c is None:
                failed += 1
                if tied:
                    lines.append(f"  ✗ '{label}': matches several fields: "
                                 + ", ".join(f"[{t.index}] '{t.label}'" for t in tied[:8]))
                else:
                    lines.append(f"  ✗ '{label}': no such field (fields: "
                                 + ", ".join(f"'{x.label}'" for x in fieldish[:25] if x.label) + ")")
                continue
            why = set_one(c, value)
            if why:
                failed += 1
                lines.append(f"  ✗ [{c.index}] '{c.label}': {why}")
            else:
                lines.append(f"  ✓ [{c.index}] '{c.label}' = {value!r}")
            if c.kind == "tab":                 # a new tab shows other fields: read them again
                env.sleep(0.2)
                controls = axs.collect_controls(ax, root)
                register(controls)
                fieldish = [x for x in controls if x.kind != "button"]
        if press:
            if failed:
                lines.append(f"Did not press '{press}': {failed} field(s) above were not set.")
                return "\n".join(lines)
            buttons = [c for c in controls if c.kind == "button"]
            b, tied = axs.pick_control(buttons, press)
            if b is None:
                names = ", ".join(f"'{x.label}'" for x in (tied or buttons)[:15])
                lines.append(f"No button '{press}' (buttons: {names}).")
                return "\n".join(lines)
            if not ax.act(b.el, "AXPress"):
                lines.append(f"The press on '{b.label}' was not accepted.")
                return "\n".join(lines)
            tk._invalidate_scan_cache(app["name"])
            env.sleep(0.4)
            after = dialog_brief(env.app_element(app["pid"]))
            lines.append(f"Pressed '{b.label}'. " + (f"Now open: {after}." if after else "No dialog is open now."))
        return "\n".join(lines)

    def _name_field(controls: list, label: str) -> Any:
        """The file-name box of a Save / Export sheet when the label given is not its own: a generic name
        ("Name", "File name") or the box's current text ("Untitled")."""
        texts = [c for c in controls if c.kind in ("text", "combo")]
        named = [c for c in texts if _NAME_FIELD.search(c.label or "")]
        if _NAME_FIELD.search(label) and len(named) == 1:
            return named[0]
        same = [c for c in texts if (c.value or "").strip() and (c.value or "").strip() == label.strip()]
        return same[0] if len(same) == 1 else None

    @tool("fill_dialog", args_schema=_FillDialogInput)
    async def fill_dialog(app_name: str = "", fields: dict[str, Any] | None = None, press: str = "",
                          path: str = "") -> str:
        """Set several fields of the open dialog (or front window) in one call, by their labels, then
        optionally press a button: fill_dialog('LibreOffice', {"Decimal places": "2"}, press='OK'). Says which
        fields were set; nothing is pressed if a field could not be set. In an Open or Save dialog, path='~/Desktop'
        goes to that folder first: fill_dialog('Numbers', {'Save As': 'out.csv'}, press='Save', path='~/Desktop')."""
        return await run(lambda: _fill_dialog(app_name, dict(fields or {}), press, path))

    # writing cells, selecting text ----------------------------------------

    def _goto_cell(box: Any, pid: int, ref: str) -> bool:
        if not ax.set(box, "AXFocused", True):
            return False
        env.sleep(0.05)
        env.hotkey("command", "a")
        env.type_text(pid, ref)
        env.press_key("return")
        env.sleep(0.12)
        return True

    def _write_cells(app_name: str, anchor: str, rows: list[list[Any]]) -> str:
        app, _, err = target(app_name)
        if app is None:
            return err
        wanted = axs.block_cells(anchor, rows)
        if wanted is None:
            return f"'{anchor}' is not one cell; give the block's top-left cell, e.g. 'B2'."
        if not wanted:
            return "No values given: rows is a list of rows, e.g. [[\"Q1\", 120], [\"Q2\", 95]]."
        if len(wanted) > 1000:
            return f"{len(wanted)} cells is too many for one call (at most 1000); split the block."
        why = ensure_front(app)
        if why:
            return why
        pid = app["pid"]
        w = front_window(env.app_element(pid))
        box = axs.find_named_field(ax, w, "Name Box") if w is not None else None
        if box is None:
            return (f"No Name Box in {app['name']}'s front window, so the cells cannot be reached. write_cells "
                    "works in a spreadsheet window (LibreOffice Calc) with no dialog open.")
        c0, r0 = axs.cell_address(anchor)
        for dr, row in enumerate(rows):
            if not _goto_cell(box, pid, f"{axs.col_letters(c0)}{r0 + dr}"):
                tk._invalidate_scan_cache(app["name"])
                return f"Stopped at row {dr + 1}: the Name Box did not take focus. Rows before it were written."
            for v in row:
                text = "" if v is None else str(v)
                if text:
                    env.type_text(pid, text)
                env.press_key("tab")            # commits the cell and moves right; an empty value skips it
            env.sleep(0.05)
        _goto_cell(box, pid, anchor)            # back to the block, so it is on screen for the check
        env.sleep(0.3)
        tk._invalidate_scan_cache(app["name"])
        w = front_window(env.app_element(pid))
        shown = axs.read_cells(ax, w) if w is not None else {}
        bad = axs.cell_mismatches(wanted, shown)
        c1, r1 = c0 + max(len(r) for r in rows) - 1, r0 + len(rows) - 1
        where = f"{anchor.upper()}:{axs.col_letters(c1)}{r1}"
        unseen = sum(1 for k, v in wanted.items() if k not in shown and v and not v.startswith("="))
        lines = [f"Typed {sum(1 for v in wanted.values() if v)} cell(s) into {where}."]
        if bad:
            lines.append("These cells do not show what was typed (Calc's AutoInput may have completed a word, or "
                         "the value was read as a date or number):\n  " + "\n  ".join(bad[:15]))
        elif unseen:
            lines.append(f"{unseen} typed cell(s) are off screen and were not checked; read_table after scrolling.")
        else:
            lines.append("Every typed cell on screen shows what was typed (formulas show their results).")
        return "\n".join(lines)

    @tool("write_cells", args_schema=_WriteCellsInput)
    async def write_cells(app_name: str = "", anchor: str = "A1", rows: list[list[Any]] | None = None) -> str:
        """Type a block of values or formulas into a spreadsheet in one call, row by row from `anchor`, then read
        the cells back and list any that differ: write_cells('LibreOffice', 'B2', [["Q1", 120], ["Q2", 95]]).
        Use it instead of typing cell by cell."""
        return await run(lambda: _write_cells(app_name, anchor, [list(r) for r in (rows or [])]))

    def _select_text(app_name: str, text: str, occurrence: int, paragraph: bool) -> str:
        app, _, err = target(app_name)
        if app is None:
            return err
        why = ensure_front(app)
        if why:
            return why
        w = front_window(env.app_element(app["pid"]))
        hit = axs.find_text(ax, w, text, occurrence) if w is not None else None
        if hit is None:
            return (f"{text!r} (match {occurrence}) is not in the text {app['name']} shows through Accessibility "
                    "(only the visible part of a document is). Scroll to it, or use Find & Replace (⌘H): type the "
                    "text, press Find All (it selects every match), then Close.")
        el, i, val = hit
        start, length = (0, len(val)) if paragraph else (i, len(text))
        ax.set(el, "AXFocused", True)
        env.sleep(0.05)
        if not env.set_selection(el, start, length):
            return ("The app did not accept the selection through Accessibility. Use Find & Replace (⌘H) > Find "
                    "All, or click at the start and ⇧-click at the end.")
        env.sleep(0.1)
        got = ax.get(el, "AXSelectedText")
        want = val[start:start + length]
        if isinstance(got, str) and got == want:
            words = len(want.split())
            return f"Selected {words} word(s), {len(want)} character(s): {_clip(want, 120)!r}."
        if isinstance(got, str):
            return (f"Warning: the selection reads {_clip(got, 120)!r}, not {_clip(want, 120)!r}. Fix it before "
                    "formatting.")
        return f"Selected {len(want)} character(s) (the app does not report the selection back): {_clip(want, 120)!r}."

    @tool("select_text", args_schema=_SelectTextInput)
    async def select_text(app_name: str = "", text: str = "", occurrence: int = 1, paragraph: bool = False) -> str:
        """Select exact text in a document (LibreOffice Writer, TextEdit, a text field) through Accessibility, and
        say what got selected: select_text('LibreOffice', 'Grammar test 2'), or paragraph=True for its whole
        paragraph. Then format it (menus, shortcuts). Clicking lands mid-paragraph; this does not."""
        return await run(lambda: _select_text(app_name, text, occurrence, paragraph))

    # tables ----------------------------------------------------------------

    def _read_table(app_name: str, range_ref: str) -> str:
        app, _, err = target(app_name)
        if app is None:
            return err
        w = front_window(env.app_element(app["pid"]))
        if w is None:
            return f"{app['name']} has no open window."
        rng = None
        if range_ref.strip():
            rng = axs.parse_range(range_ref)
            if rng is None:
                return f"'{range_ref}' is not a cell range; write it like 'A1:F20' or 'B3'."
        cells = axs.read_cells(ax, w)
        if cells:
            text = axs.format_cells(cells, rng)
            if rng and not any(rng[0] <= c <= rng[2] and rng[1] <= r <= rng[3] for c, r in cells):
                text += ("\nNone of that range is on screen. Go to it first (Name Box, or scroll), then read "
                         "again.")
            return text
        tables = axs.read_ax_tables(ax, w)
        if tables:
            return "\n\n".join(axs.format_ax_table(*t) for t in tables)
        return (f"No table or cells readable through Accessibility in {app['name']}'s front window. Use "
                "read_screen for the visible text, or the app's scripting/export.")

    @tool("read_table", args_schema=_ReadTableInput)
    async def read_table(app_name: str = "", range_ref: str = "") -> str:
        """Read the spreadsheet cells or table rows on screen in one call, as tab-separated rows with column
        letters and row numbers, e.g. read_table('LibreOffice', 'A1:F20'). Only what the window shows is in
        the Accessibility tree; move there first for cells off screen."""
        return await run(lambda: _read_table(app_name, range_ref))

    def _get_control_values(indices: list[int]) -> str:
        lines = []
        for i in indices:
            el = tk._element_registry.get(i)
            if el is None:
                lines.append(f"[{i}] not in the last read; call get_screen_controls or read_dialog again")
                continue
            role = str(ax.get(el, "AXRole") or "")[2:]
            label = str(ax.get(el, "AXTitle") or ax.get(el, "AXDescription") or "")
            val = ax.get(el, "AXValue")
            v = "(no value)" if val is None or isinstance(val, (list, tuple)) else repr(_clip(val, 200))
            lines.append(f"[{i}] {role}" + (f" '{_clip(label, 50)}'" if label else "") + f" = {v}")
        return "\n".join(lines)

    @tool("get_control_values", args_schema=_GetControlValuesInput)
    async def get_control_values(indices: list[int]) -> str:
        """Read the current values of several controls at once by index, e.g. get_control_values([4, 7, 12]).
        One call instead of one get_control_value per control."""
        return await run(lambda: _get_control_values(list(indices)))

    def _has_text(root: Any, needle: str, budget: int = 1500) -> bool:
        want = axs.norm_label(needle)
        stack, seen = [root], 0
        while stack and seen < budget:
            el = stack.pop()
            seen += 1
            for attr in ("AXValue", "AXTitle", "AXDescription"):
                v = ax.get(el, attr)
                if isinstance(v, str) and want in axs.norm_label(v):
                    return True
            stack.extend(reversed(ax.children(el)))
        return False

    def _wait_until(app_name: str, title: str, text: str, dialog: str, timeout: float) -> str:
        dialog = axs.norm_label(dialog)
        if dialog not in ("", "open", "closed"):
            return "dialog must be 'open', 'closed' or empty."
        if not (title or text or dialog):
            return "Give at least one of title, text or dialog to wait for."
        start = time.monotonic()
        last = ""
        while True:
            app, _, err = target(app_name)
            if app is not None:
                el = env.app_element(app["pid"])
                titles = [str(ax.get(w, "AXTitle") or "") for w in (ax.get(el, "AXWindows") or [])]
                has_dialog = axs.find_dialog(ax, el) is not None
                ok = True
                if title:
                    ok &= any(axs.norm_label(title) in axs.norm_label(t) for t in titles)
                if dialog:
                    ok &= has_dialog == (dialog == "open")
                if ok and text:
                    w = front_window(el)
                    ok &= w is not None and _has_text(w, text)
                last = (f"windows: {', '.join(repr(_clip(t, 60)) for t in titles[:5]) or 'none'}; "
                        f"dialog {'open' if has_dialog else 'none'}")
                if ok:
                    return f"Done after {time.monotonic() - start:.1f}s ({last})."
            else:
                last = err
            if time.monotonic() - start >= timeout:
                return f"Not yet after {timeout:g}s ({last})."
            env.sleep(0.4)

    @tool("wait_until", args_schema=_WaitUntilInput)
    async def wait_until(app_name: str = "", title: str = "", text: str = "", dialog: str = "",
                         timeout: float = 10) -> str:
        """Wait until a window title contains `title`, `text` shows in the front window, and/or a dialog is
        `open` or `closed`, for at most `timeout` seconds. Use it after starting something slow (opening a
        file, an export, a page load) instead of reading again and again."""
        return await run(lambda: _wait_until(app_name, title, text, dialog, timeout))

    # app guide -------------------------------------------------------------

    @tool("app_guide", args_schema=_AppGuideInput)
    async def app_guide(app: str) -> str:
        """Notes on working an app on macOS: its process name, whether it can be scripted, where common
        commands are in its menus, how it saves, and what it cannot do. Call it before working in an app."""
        card = card_for(app)
        if card:
            return card
        return (f"No guide for '{app}'. Guides exist for: {', '.join(known_apps())}. For another app: "
                "inspect_app_dictionary shows whether it can be scripted, menu_search finds its commands, and "
                "read_dialog reads its dialogs.")

    return {t.name: t for t in (observe, menu_select, menu_search, read_dialog, fill_dialog, read_table,
                                 get_control_values, wait_until, app_guide, write_cells, select_text)}


STRUCTURED_TOOL_NAMES = ("observe", "menu_select", "menu_search", "read_dialog", "fill_dialog", "read_table",
                         "get_control_values", "wait_until", "app_guide", "write_cells", "select_text")
