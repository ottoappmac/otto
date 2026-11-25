"""Structured reads and intent-level actions over a macOS Accessibility tree.

The desktop agent used to reach menus, dialogs and tables one control at a time, or by writing System Events
paths by hand. Benchmark runs showed what that costs: eight failed menu scripts in a row on one zoom task (wrong
element form, wrong process name), 93 single-cell reads in another. This module does those jobs in one call:

* ``select_menu`` / ``search_menus``: choose a menu item by its titles, or find it by a word.
* ``find_dialog`` / ``collect_controls``: the open sheet or dialog with every field's label and value.
* ``read_cells`` / ``read_ax_tables``: the cells on screen as a grid.
* ``controls_delta``: what changed since the last controls listing, so an action's result can stay short.
* ``windows_above``: windows of other apps sitting over the frontmost one (alerts, permission prompts).

Everything here works through an ``Access`` object (read an attribute, list children, set a value, perform an
action, press Escape), never through PyObjC directly. ``macos_structured_tools`` passes one backed by the real
AX API; tests pass a tree of dicts. Nothing in this module imports a macOS framework.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol, Sequence


class Access(Protocol):
    """What this module needs from the Accessibility API."""

    def get(self, el: Any, attr: str) -> Any: ...
    def children(self, el: Any) -> list: ...
    def set(self, el: Any, attr: str, value: Any) -> bool: ...
    def act(self, el: Any, action: str) -> bool: ...
    def escape(self) -> None: ...
    def same(self, a: Any, b: Any) -> bool: ...


# ── Labels ──────────────────────────────────────────────────────────────────

_TRAILING_DOTS = re.compile(r"(?:\.{2,}|…)\s*$")
_SPACES = re.compile(r"\s+")


def norm_label(text: Any) -> str:
    """A label as compared for matching: case, a trailing ellipsis ("…" or "..."), a trailing colon, curly quotes and
    repeated spaces do not matter. "Zoom…", "zoom..." and "Zoom" are the same label."""
    s = unicodedata.normalize("NFC", str(text or ""))
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = _TRAILING_DOTS.sub("", s.strip())
    s = s.rstrip(":").strip()
    return _SPACES.sub(" ", s).casefold()


def match_label(want: str, candidates: Sequence[str]) -> tuple[int | None, list[int]]:
    """Index of the candidate that ``want`` names, and the indices that tied when it is ambiguous.

    Exact (normalized) match first, then a unique prefix match, then a unique substring match."""
    w = norm_label(want)
    if not w:
        return None, []
    normed = [norm_label(c) for c in candidates]
    exact = [i for i, c in enumerate(normed) if c == w]
    if exact:
        return exact[0], []
    for test in (lambda c: c.startswith(w), lambda c: w in c):
        hits = [i for i, c in enumerate(normed) if c and test(c)]
        if len(hits) == 1:
            return hits[0], []
        if len(hits) > 1:
            return None, hits
    return None, []


_PATH_SPLIT = re.compile(r"\s*(?:>|▸|›|→|->)\s*")


def parse_menu_path(path: Any) -> list[str]:
    """``"Format > Cells…"``, ``"View ▸ Zoom ▸ Zoom…"`` or a list of titles, as a list of titles."""
    if isinstance(path, (list, tuple)):
        parts = [str(p) for p in path]
    else:
        parts = _PATH_SPLIT.split(str(path or ""))
    return [p.strip() for p in parts if p and p.strip()]


def _title(ax: Access, el: Any) -> str:
    return str(ax.get(el, "AXTitle") or "").strip()


def _role(ax: Access, el: Any) -> str:
    return str(ax.get(el, "AXRole") or "")


# ── Menus ───────────────────────────────────────────────────────────────────

# AXMenuItemCmdModifiers bits: shift 1, option 2, control 4, and 8 means "no Command key".
def format_shortcut(char: Any, mods: Any) -> str:
    """The shortcut as macOS draws it in a menu ("⇧⌘S"), or "" when the item has none."""
    c = str(char or "").strip()
    if not c:
        return ""
    try:
        m = int(mods or 0)
    except (TypeError, ValueError):
        m = 0
    out = ""
    if m & 4:
        out += "⌃"
    if m & 2:
        out += "⌥"
    if m & 1:
        out += "⇧"
    if not m & 8:
        out += "⌘"
    return out + c.upper()


def menu_items(ax: Access, container: Any) -> list[tuple[str, Any]]:
    """The titled items under a menu bar, a menu, a menu bar item or a menu item (separators left out)."""
    role = _role(ax, container)
    if role in ("AXMenuBar", "AXMenu"):
        items = ax.children(container)
    else:
        menus = [k for k in ax.children(container) if _role(ax, k) == "AXMenu"]
        items = ax.children(menus[0]) if menus else []
    return [(t, el) for el in items if (t := _title(ax, el))]


def _has_submenu(ax: Access, el: Any) -> bool:
    return any(_role(ax, k) == "AXMenu" for k in ax.children(el))


@dataclass
class MenuResult:
    ok: bool
    chosen: list[str] = field(default_factory=list)
    error: str = ""
    options: list[str] = field(default_factory=list)


def select_menu(ax: Access, menubar: Any, path: Sequence[str]) -> MenuResult:
    """Choose the menu item at ``path`` (titles from the menu bar down) and press it.

    Menus are read without opening them where the app exposes their items; a menu that reads empty (built on
    demand) is opened first. Any menu this opened is closed again when the path cannot be followed."""
    if not path:
        return MenuResult(False, error="give the menu path, e.g. 'Format > Cells…'")
    container = menubar
    chosen: list[str] = []
    opened = False

    def close() -> None:
        if opened:
            ax.escape()
            ax.escape()

    for depth, want in enumerate(path):
        items = menu_items(ax, container)
        if not items and depth > 0:
            # Built on demand: open it (the top-level menu first, then this item), then read again.
            ax.act(container, "AXPress")
            opened = True
            items = menu_items(ax, container)
        titles = [t for t, _ in items]
        i, tied = match_label(want, titles)
        if i is None:
            close()
            where = " ▸ ".join(chosen) or "the menu bar"
            if tied:
                return MenuResult(False, chosen, f"'{want}' matches several items in {where}",
                                  [titles[j] for j in tied])
            return MenuResult(False, chosen, f"no item '{want}' in {where}", titles)
        chosen.append(titles[i])
        container = items[i][1]
    leaf = container
    if len(path) == 1 or _has_submenu(ax, leaf):
        subs = [t for t, _ in menu_items(ax, leaf)]
        close()
        return MenuResult(False, chosen, f"{' ▸ '.join(chosen)} is a menu, not an item; add one of its items", subs)
    if ax.get(leaf, "AXEnabled") is False:
        close()
        return MenuResult(False, chosen, f"{' ▸ '.join(chosen)} is disabled (greyed out) right now. It usually "
                          "needs a selection or a different mode first")
    if not ax.act(leaf, "AXPress"):
        close()
        return MenuResult(False, chosen, f"the app did not accept the press on {' ▸ '.join(chosen)}")
    return MenuResult(True, chosen)


@dataclass
class MenuHit:
    path: list[str]
    shortcut: str
    enabled: bool

    def line(self) -> str:
        s = " ▸ ".join(self.path)
        if self.shortcut:
            s += f"   {self.shortcut}"
        if not self.enabled:
            s += "   (disabled now)"
        return s


def walk_menus(ax: Access, menubar: Any, *, max_depth: int = 4, max_items: int = 3000,
               open_empty: bool = True) -> list[MenuHit]:
    """Every titled item in the app's menus, as paths. The Apple menu is left out. A top-level menu that reads empty
    is opened for the read and closed again when ``open_empty``."""
    out: list[MenuHit] = []

    def visit(container: Any, prefix: list[str], depth: int) -> None:
        if depth > max_depth or len(out) >= max_items:
            return
        for title, el in menu_items(ax, container):
            if len(out) >= max_items:
                return
            path = prefix + [title]
            if _has_submenu(ax, el):
                visit(el, path, depth + 1)
            else:
                out.append(MenuHit(path, format_shortcut(ax.get(el, "AXMenuItemCmdChar"),
                                                         ax.get(el, "AXMenuItemCmdModifiers")),
                                   ax.get(el, "AXEnabled") is not False))

    for title, top in menu_items(ax, menubar):
        if norm_label(title) == "apple":
            continue
        if not menu_items(ax, top) and open_empty:
            ax.act(top, "AXPress")
            visit(top, [title], 1)
            ax.escape()
        else:
            visit(top, [title], 1)
    return out


def search_menus(hits: Iterable[MenuHit], query: str, limit: int = 25) -> list[MenuHit]:
    """The menu items whose path contains every word of ``query``; items whose own title matches come first."""
    words = [w for w in norm_label(query).split() if w]
    if not words:
        return []
    scored: list[tuple[int, int, MenuHit]] = []
    for n, h in enumerate(hits):
        full = norm_label(" ".join(h.path))
        if not all(w in full for w in words):
            continue
        leaf = norm_label(h.path[-1])
        score = 0 if leaf == " ".join(words) else 1 if all(w in leaf for w in words) else 2
        scored.append((score, n, h))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [h for _, _, h in scored[:limit]]


# ── Dialogs ─────────────────────────────────────────────────────────────────

_DIALOG_SUBROLES = frozenset({"AXDialog", "AXSystemDialog"})


def find_dialog(ax: Access, app_el: Any) -> tuple[Any, str] | None:
    """The open sheet or dialog of an app as ``(element, description)``, or None.

    A sheet attached to a window comes first, then a window macOS marks as a dialog or as modal, then a focused
    window that is not the main window, then a front window with no document while another window has one.

    The last rule is LibreOffice's (seen in the benchmark VM): its dialogs (Chart Wizard, ...) are plain standard
    windows, not modal, and become the focused *and* main window; only the missing ``AXDocument`` tells them from
    the document window. Minimized windows are skipped: a minimized LibreOffice document reports the AXDialog
    subrole."""
    windows = [w for w in (ax.get(app_el, "AXWindows") or []) if ax.get(w, "AXMinimized") is not True]
    focused = ax.get(app_el, "AXFocusedWindow")
    main = ax.get(app_el, "AXMainWindow")
    if focused is not None and ax.get(focused, "AXMinimized") is True:
        focused = None
    if focused is not None and not any(ax.same(focused, w) for w in windows):
        windows.insert(0, focused)
    elif focused is not None:
        windows = [focused] + [w for w in windows if not ax.same(w, focused)]
    for w in windows:
        for k in ax.children(w):
            if _role(ax, k) == "AXSheet":
                return k, f"sheet on '{_title(ax, w)}'"
    for w in windows:
        if str(ax.get(w, "AXSubrole") or "") in _DIALOG_SUBROLES or ax.get(w, "AXModal") is True:
            return w, f"dialog '{_title(ax, w)}'"
    if focused is not None and main is not None and not ax.same(focused, main):
        return focused, f"window '{_title(ax, focused)}' (in front of the main window)"
    if windows and len(windows) > 1:
        front = windows[0]
        if not ax.get(front, "AXDocument") and any(ax.get(w, "AXDocument") for w in windows[1:]):
            return front, f"window '{_title(ax, front)}' (in front of the document)"
    return None


_KIND = {
    "AXTextField": "text", "AXTextArea": "text", "AXSecureTextField": "text", "AXSearchField": "text",
    "AXComboBox": "combo", "AXPopUpButton": "popup", "AXMenuButton": "popup",
    "AXCheckBox": "checkbox", "AXRadioButton": "radio", "AXSlider": "slider", "AXIncrementor": "stepper",
    "AXButton": "button", "AXDisclosureTriangle": "button", "AXLink": "button",
}
_CODE = {"text": "TF", "combo": "CX", "popup": "PB", "checkbox": "CB", "radio": "RB", "tab": "TAB",
         "slider": "SL", "stepper": "ST", "button": "B", "list": "LS"}
# List boxes: LibreOffice draws them as an AXMenu of AXMenuItem/AXOutlineRow inside the dialog (Chart Wizard's
# chart types); other apps use AXList / AXOutline of rows.
_LIST_ROLES = frozenset({"AXMenu", "AXList", "AXOutline"})
_ITEM_ROLES = frozenset({"AXMenuItem", "AXRow", "AXOutlineRow", "AXStaticText", "AXCell"})


@dataclass
class Control:
    kind: str
    label: str
    value: str
    enabled: bool
    el: Any = field(repr=False, default=None)
    index: int = 0
    items: list = field(repr=False, default_factory=list)    # list boxes: [(title, element)]

    def line(self) -> str:
        v = ""
        if self.kind == "list":
            opts = [t for t, _ in self.items]
            shown = " | ".join(opts[:15]) + (f" | … ({len(opts)} in all)" if len(opts) > 15 else "")
            v = f" = {self.value!r} (options: {shown})"
        elif self.kind in ("text", "combo", "popup", "slider"):
            v = f" = {self.value!r}"
        elif self.kind in ("checkbox", "radio", "tab"):
            v = " = on" if self.value == "on" else " = off"
        off = "" if self.enabled else " (disabled)"
        return f"[{self.index}]{_CODE.get(self.kind, '?')} '{self.label}'{v}{off}"


def _value_text(kind: str, raw: Any) -> str:
    if kind in ("checkbox", "radio", "tab"):
        try:
            return "on" if int(raw or 0) else "off"
        except (TypeError, ValueError):
            return "on" if raw else "off"
    if raw is None:
        return ""
    if isinstance(raw, float) and raw.is_integer():
        return str(int(raw))
    return str(raw)[:300]


def collect_controls(ax: Access, root: Any, *, max_nodes: int = 600, max_depth: int = 14) -> list[Control]:
    """The fields, choices and buttons under ``root`` in reading order, each with its label and current value.

    A control with no title of its own takes the nearest static text before it as its label, which is how most
    dialogs lay labels out ("Decimal places:" then a text field)."""
    out: list[Control] = []
    pending = [""]
    seen = [0]

    def label_of(el: Any) -> str:
        for attr in ("AXTitle", "AXDescription"):
            t = str(ax.get(el, attr) or "").strip()
            if t:
                return t
        tui = ax.get(el, "AXTitleUIElement")
        if tui is not None:
            t = str(ax.get(tui, "AXValue") or ax.get(tui, "AXTitle") or "").strip()
            if t:
                return t
        for attr in ("AXPlaceholderValue", "AXHelp"):
            t = str(ax.get(el, attr) or "").strip()
            if t:
                return t
        return ""

    def visit(el: Any, depth: int, in_tabs: bool) -> None:
        if depth > max_depth or seen[0] >= max_nodes:
            return
        seen[0] += 1
        role = _role(ax, el)
        if role == "AXStaticText":
            t = str(ax.get(el, "AXValue") or ax.get(el, "AXTitle") or "").strip()
            if t:
                pending[0] = t
            return
        if role in _LIST_ROLES and depth > 0:
            items = []
            for k in ax.children(el):
                if _role(ax, k) in _ITEM_ROLES:
                    t = _title(ax, k) or str(ax.get(k, "AXValue") or "").strip() or _cell_text(ax, k)
                    if t:
                        items.append((t, k))
            if items:
                chosen = [t for t, k in items if ax.get(k, "AXSelected") is True]
                label = label_of(el) if (_title(ax, el) or ax.get(el, "AXDescription")) else (pending[0] or label_of(el))
                out.append(Control("list", label, ", ".join(chosen), ax.get(el, "AXEnabled") is not False, el,
                                   items=items))
                pending[0] = ""
                return
        kind = _KIND.get(role)
        if kind == "radio" and in_tabs:
            kind = "tab"
        elif kind is None and role == "AXUnknown" and _title(ax, el) and ax.get(el, "AXValue") is not None:
            kind = "text"           # LibreOffice's spin fields ("Decimal places:") have no role but a value
        if kind == "tab":
            # LibreOffice puts the shown page's fields inside its tab, and reports every tab as off; the tab that
            # holds the page is the selected one.
            kids = ax.children(el)
            label = label_of(el)
            value = _value_text(kind, ax.get(el, "AXValue"))
            if kids and value == "off" and not any(c.kind == "tab" and c.value == "on" for c in out):
                value = "on"
            out.append(Control(kind, label, value, ax.get(el, "AXEnabled") is not False, el))
            pending[0] = ""
            for k in kids:
                visit(k, depth + 1, False)
            return
        if kind:
            label = label_of(el)
            if not label and kind != "button":
                label = pending[0]
            if label or kind != "button":
                raw = ax.get(el, "AXValue")
                if kind == "popup" and not isinstance(raw, str):
                    # LibreOffice's pop-ups report 0 and show their choice in a text child.
                    shown = [ax.get(k, "AXValue") for k in ax.children(el)
                             if _role(ax, k) in ("AXTextArea", "AXTextField", "AXStaticText")]
                    raw = next((v for v in shown if isinstance(v, str) and v), None)
                    if raw is None:
                        raw = ax.get(el, "AXTitle") if ax.get(el, "AXValue") is None else ax.get(el, "AXValue")
                out.append(Control(kind, label, _value_text(kind, raw), ax.get(el, "AXEnabled") is not False, el))
            if kind not in ("combo",):
                pending[0] = ""
                return
        # Only the tab group's own radio buttons are its tabs; radio buttons inside a tab's page are choices.
        tabs = role == "AXTabGroup"
        for k in ax.children(el):
            visit(k, depth + 1, tabs)

    visit(root, 0, False)
    return out


def dialog_messages(ax: Access, root: Any, *, max_nodes: int = 250, max_depth: int = 8) -> list[str]:
    """The sentences a dialog shows ("You must enter a valid reference..."): static texts long enough to be a
    message rather than a field label or a group heading. An alert is mostly its message and an OK button."""
    out: list[str] = []
    seen = [0]

    def visit(el: Any, depth: int) -> None:
        if depth > max_depth or seen[0] >= max_nodes:
            return
        seen[0] += 1
        if _role(ax, el) == "AXStaticText":
            t = " ".join(str(ax.get(el, "AXValue") or "").split())
            if (len(t) >= 30 or t[-1:] in ".?!") and len(t.split()) >= 3 and t not in out:
                out.append(t[:400])
            return
        for k in ax.children(el):
            visit(k, depth + 1)

    visit(root, 0)
    return out[:6]


def format_dialog(desc: str, controls: Sequence[Control], messages: Sequence[str] = ()) -> str:
    """The dialog as the agent reads it: its message, tabs, then fields, then buttons, each with an index for the
    next call."""
    tabs = [c for c in controls if c.kind == "tab"]
    buttons = [c for c in controls if c.kind == "button"]
    fields_ = [c for c in controls if c.kind not in ("tab", "button")]
    lines = [f"Open: {desc}"] + [f"Message: {m}" for m in messages]
    if tabs:
        lines.append("Tabs: " + "  ".join(c.line() for c in tabs))
    if fields_:
        lines.append("Fields:")
        lines += [f"  {c.line()}" for c in fields_]
    if buttons:
        lines.append("Buttons: " + "  ".join(c.line() for c in buttons))
    if not (tabs or fields_ or buttons):
        lines.append("(no fields or buttons readable through Accessibility: the window is open but its controls "
                     "are drawn without Accessibility, as in GTK apps like GIMP. Read it with read_screen(app), which "
                     "reads the front window by OCR, and act with click_at / type_text at the positions it gives)")
    return "\n".join(lines)


def describe_dialog_brief(desc: str, controls: Sequence[Control], limit: int = 6,
                          messages: Sequence[str] = ()) -> str:
    """One line naming the dialog and its buttons, for an observation or an action's result."""
    names = [c.label for c in controls if c.kind == "button" and c.label][:limit]
    said = f" saying {messages[0][:160]!r}" if messages else ""
    return f"{desc}{said}" + (f" (buttons: {', '.join(names)})" if names else "")


def coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = norm_label(value)
    if s in ("on", "true", "yes", "1", "checked", "selected"):
        return True
    if s in ("off", "false", "no", "0", "unchecked"):
        return False
    return None


def pick_control(controls: Sequence[Control], label: str, kinds: Iterable[str] | None = None) -> tuple[Control | None, list[Control]]:
    """The control a field name in ``fill_dialog`` means, and the tied ones when it is ambiguous."""
    pool = [c for c in controls if kinds is None or c.kind in set(kinds)]
    i, tied = match_label(label, [c.label for c in pool])
    if i is not None:
        return pool[i], []
    return None, [pool[j] for j in tied]


# ── Tables ──────────────────────────────────────────────────────────────────

CELL_ADDR = re.compile(r"^\$?([A-Za-z]{1,3})\$?([0-9]{1,7})$")
_RANGE = re.compile(r"^\s*\$?([A-Za-z]{1,3})\$?([0-9]{1,7})\s*(?::\s*\$?([A-Za-z]{1,3})\$?([0-9]{1,7}))?\s*$")


def col_number(letters: str) -> int:
    n = 0
    for ch in letters.upper():
        n = n * 26 + (ord(ch) - 64)
    return n


def col_letters(n: int) -> str:
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def parse_range(text: str) -> tuple[int, int, int, int] | None:
    """``"B2:D9"`` or ``"C4"`` as (first column, first row, last column, last row), 1-based; None if not a range."""
    m = _RANGE.match(text or "")
    if not m:
        return None
    c1, r1 = col_number(m.group(1)), int(m.group(2))
    c2, r2 = (col_number(m.group(3)), int(m.group(4))) if m.group(3) else (c1, r1)
    return min(c1, c2), min(r1, r2), max(c1, c2), max(r1, r2)


def cell_address(label: str) -> tuple[int, int] | None:
    """(column, row) of a cell named like ``"B7"``, or None."""
    m = CELL_ADDR.match(str(label or "").strip())
    if not m:
        return None
    return col_number(m.group(1)), int(m.group(2))


def _clip_cell(v: Any, n: int = 40) -> str:
    s = " ".join(str(v if v is not None else "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def read_cells(ax: Access, root: Any, *, max_nodes: int = 20000, max_depth: int = 12) -> dict[tuple[int, int], str]:
    """Every element under ``root`` named like a spreadsheet cell ("A1") with its value, by (column, row).

    This is how LibreOffice Calc exposes its grid: only the cells currently on screen are in the tree."""
    cells: dict[tuple[int, int], str] = {}
    seen = [0]

    def visit(el: Any, depth: int) -> None:
        if depth > max_depth or seen[0] >= max_nodes:
            return
        seen[0] += 1
        for attr in ("AXDescription", "AXTitle"):
            label = ax.get(el, attr)
            addr = cell_address(label)
            if addr is not None:
                val = ax.get(el, "AXValue")
                # Calc's Name Box is named by the cell it shows ("A1") and holds that same text: not a cell.
                if str(val or "").replace("$", "").strip().upper() == str(label).replace("$", "").strip().upper():
                    return
                cells[addr] = str(val if val is not None else "")
                return
        for k in ax.children(el):
            visit(k, depth + 1)

    visit(root, 0)
    return cells


def format_cells(cells: dict[tuple[int, int], str], rng: tuple[int, int, int, int] | None = None,
                 max_rows: int = 200, max_cols: int = 30) -> str:
    """The cells as a grid with column letters and row numbers. Empty cells show as blank."""
    if rng is not None:
        c1, r1, c2, r2 = rng
        cells = {k: v for k, v in cells.items() if c1 <= k[0] <= c2 and r1 <= k[1] <= r2}
    if not cells:
        return ""
    cols = sorted({c for c, _ in cells})
    rows = sorted({r for _, r in cells})
    # Empty columns and rows before the data are left out, so the first column shown holds data: with three blank
    # columns in front, an agent read column D as the first column and wrote over it. Only for a read from the
    # sheet's start (no range, or one starting at A1); a range the agent picked is shown as asked.
    skipped = []
    filled = [k for k, v in cells.items() if str(v).strip()]
    if filled and (rng is None or (rng[0] == 1 and rng[1] == 1)):
        c0, r0 = min(c for c, _ in filled), min(r for _, r in filled)
        if c0 > cols[0]:
            skipped.append(f"columns {col_letters(cols[0])}–{col_letters(c0 - 1)}" if c0 - 1 > cols[0]
                           else f"column {col_letters(cols[0])}")
            cols = [c for c in cols if c >= c0]
        if r0 > rows[0]:
            skipped.append(f"rows {rows[0]}–{r0 - 1}" if r0 - 1 > rows[0] else f"row {rows[0]}")
            rows = [r for r in rows if r >= r0]
    cut = []
    if len(cols) > max_cols:
        cols, cut = cols[:max_cols], cut + [f"columns after {col_letters(cols[max_cols - 1])}"]
    if len(rows) > max_rows:
        rows, cut = rows[:max_rows], cut + [f"rows after {rows[max_rows - 1]}"]
    lines = ["\t".join([""] + [col_letters(c) for c in cols])]
    for r in rows:
        lines.append("\t".join([str(r)] + [_clip_cell(cells.get((c, r), "")) for c in cols]))
    if cut:
        lines.append(f"(cut: {', '.join(cut)})")
    if skipped:
        lines.append(f"({' and '.join(skipped)} are empty and not shown. The letters and numbers above are the "
                     "cells' real addresses: the first column shown is {}.)".format(col_letters(cols[0])))
    return "\n".join(lines)


_TABLE_ROLES = frozenset({"AXTable", "AXOutline", "AXGrid"})


def _cell_text(ax: Access, el: Any, depth: int = 0) -> str:
    for attr in ("AXValue", "AXTitle", "AXDescription"):
        v = ax.get(el, attr)
        if v not in (None, "") and not isinstance(v, (list, tuple)):
            return str(v)
    if depth >= 3:
        return ""
    parts = [t for k in ax.children(el) if (t := _cell_text(ax, k, depth + 1))]
    return " ".join(parts)


def read_ax_tables(ax: Access, root: Any, *, max_tables: int = 4, max_rows: int = 200,
                   max_cols: int = 30, max_depth: int = 12) -> list[tuple[str, list[str], list[list[str]]]]:
    """The tables, outlines and grids under ``root`` as (name, header, rows) — for apps whose tables are real AX
    tables (Finder lists, Numbers, settings lists, web tables in Chrome)."""
    found: list[Any] = []

    def visit(el: Any, depth: int) -> None:
        if depth > max_depth or len(found) >= max_tables:
            return
        if _role(ax, el) in _TABLE_ROLES:
            found.append(el)
            return
        for k in ax.children(el):
            visit(k, depth + 1)

    visit(root, 0)
    out = []
    for t in found:
        rows_el = ax.get(t, "AXRows")
        if rows_el is None:
            rows_el = [k for k in ax.children(t) if _role(ax, k) == "AXRow"]
        header: list[str] = []
        hdr = ax.get(t, "AXHeader")
        if hdr is not None:
            header = [_clip_cell(_cell_text(ax, k)) for k in ax.children(hdr)][:max_cols]
        else:
            cols = ax.get(t, "AXColumns") or []
            titles = [_clip_cell(_title(ax, c) or str(ax.get(c, "AXDescription") or "")) for c in cols][:max_cols]
            if any(titles):
                header = titles
        rows = []
        for r in list(rows_el)[:max_rows]:
            cells = [k for k in ax.children(r)][:max_cols]
            rows.append([_clip_cell(_cell_text(ax, c)) for c in cells])
        name = _title(ax, t) or str(ax.get(t, "AXDescription") or "") or _role(ax, t)[2:].lower()
        out.append((name, header, rows))
    return out


def format_ax_table(name: str, header: list[str], rows: list[list[str]]) -> str:
    lines = [f"Table '{name}': {len(rows)} row(s) shown"]
    if header:
        lines.append("\t".join(header))
    lines += ["\t".join(r) for r in rows]
    return "\n".join(lines)


# ── Observation deltas ─────────────────────────────────────────────────────

Item = tuple[int, str, str]     # (index, role, label) of one listed control


def controls_delta(prev: Sequence[Item] | None, cur: Sequence[Item], fmt: Callable[[Item], str],
                   max_share: float = 0.5) -> str | None:
    """What changed between two control listings of the same window, or None when the full listing must be shown.

    The short form is only safe when every control that is in both listings kept its index: the agent goes on
    using the indices it already has. So any shift of a kept control's index, or a change touching more than
    ``max_share`` of the controls, means the full listing."""
    if prev is None:
        return None
    before: dict[tuple[str, str], list[int]] = {}
    for i, r, l in prev:
        before.setdefault((r, l), []).append(i)
    after: dict[tuple[str, str], list[int]] = {}
    for i, r, l in cur:
        after.setdefault((r, l), []).append(i)
    added: list[Item] = []
    removed: list[tuple[str, str]] = []
    kept = 0
    for key in set(before) | set(after):
        b, a = before.get(key, []), after.get(key, [])
        n = min(len(a), len(b))
        if sorted(a)[:n] != sorted(b)[:n]:
            return None
        kept += n
        added += [(i, key[0], key[1]) for i in sorted(a)[n:]]
        removed += [key] * max(0, len(b) - n)
    changed = len(added) + len(removed)
    if changed and changed > max_share * max(len(cur), 1):
        return None
    if not changed:
        return (f"Same {kept} controls as the last read, at the same indices (values and on/off states are not "
                "compared: read_dialog or get_control_values shows them).")
    lines = [f"Controls added or gone since the last read ({kept} others kept their indices; values are not "
             "compared):"]
    lines += [f"  + {fmt(it)}" for it in sorted(added)]
    lines += [f"  - {r} '{l}' (gone)" for r, l in removed]
    return "\n".join(lines)


# ── Interrupts: windows over the frontmost app ─────────────────────────────

# Owners whose windows are system chrome, never something to dismiss.
_CHROME_OWNERS = frozenset({"Window Server", "Dock", "Control Center", "SystemUIServer", "Spotlight",
                            "TextInputMenuAgent", "Wallpaper", "WindowManager", "screencaptureui"})
# Owners whose windows are prompts or alerts wherever they sit.
_ALERT_OWNERS = frozenset({"UserNotificationCenter", "CoreServicesUIAgent", "SecurityAgent", "NotificationCenter",
                           "universalAccessAuthWarn", "System Events", "loginwindow", "coreautha"})


# Owners of system notices that are not part of any task and have no button an agent should press: macOS's
# "Accessibility Access" warning (universalAccessAuthWarn) offers only "Open System Settings" and "Deny", which change
# privacy settings. Benchmark runs lost 8-12 steps each trying to activate or close it.
NOTICE_OWNERS = frozenset({"universalAccessAuthWarn"})
NOTICE_NOTE = ("a macOS permission notice, not part of the task: leave it, do not activate or close it, and carry on "
               "(only if a click lands on it, move the click)")


def split_notices(windows: Sequence[tuple[str, str]]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """``windows`` as (owner, title) split into (windows to deal with, system notices to ignore)."""
    act = [w for w in windows if w[0] not in NOTICE_OWNERS]
    return act, [w for w in windows if w[0] in NOTICE_OWNERS]


def windows_above(window_list: Sequence[dict], front_owner: str, front_pid: int = 0) -> list[tuple[str, str]]:
    """Windows of other apps drawn over the frontmost app's first window, as (owner, title), front to back.

    ``window_list`` is the on-screen window list in front-to-back order (CGWindowListCopyWindowInfo dicts:
    kCGWindowOwnerName, kCGWindowName, kCGWindowLayer, kCGWindowOwnerPID). Alert and prompt owners count at any
    layer; other apps only at the normal window layer."""
    out: list[tuple[str, str]] = []
    for w in window_list:
        owner = str(w.get("kCGWindowOwnerName") or "")
        pid = int(w.get("kCGWindowOwnerPID") or 0)
        layer = int(w.get("kCGWindowLayer") or 0)
        if (front_pid and pid == front_pid) or (not front_pid and owner == front_owner):
            break
        if not owner or owner in _CHROME_OWNERS:
            continue
        if owner in _ALERT_OWNERS or layer == 0:
            out.append((owner, str(w.get("kCGWindowName") or "")))
    return out[:5]


# ── Named fields, text search, cell blocks (write_cells, select_text) ──────

_TEXT_FIELD_ROLES = ("AXTextArea", "AXTextField", "AXComboBox")


def find_named_field(ax: Access, root: Any, name: str, *, max_nodes: int = 4000, max_depth: int = 10) -> Any:
    """The text field inside the group named ``name`` (Calc's Name Box: an AXGroup described "Name Box" holding an
    AXTextArea titled with the cell it shows), or a text field labelled ``name`` itself; None when there is none.
    Searched breadth-first, so toolbars near the window's top come before a large grid."""
    want = norm_label(name)
    queue: list[tuple[Any, int, bool]] = [(root, 0, False)]
    seen = 0
    while queue and seen < max_nodes:
        el, depth, inside = queue.pop(0)
        seen += 1
        role = str(ax.get(el, "AXRole") or "")
        label = norm_label(ax.get(el, "AXDescription") or ax.get(el, "AXTitle") or "")
        if role in _TEXT_FIELD_ROLES and (inside or label == want):
            return el
        here = inside or (role == "AXGroup" and label == want)
        if depth < max_depth:
            queue.extend((k, depth + 1, here) for k in ax.children(el))
    return None


_TEXTY_ROLES = ("AXTextArea", "AXTextField", "AXStaticText", "AXParagraph")


def find_text(ax: Access, root: Any, needle: str, occurrence: int = 1, *, max_nodes: int = 6000,
              max_depth: int = 14) -> tuple[Any, int, str] | None:
    """``(element, offset, its text)`` of the ``occurrence``-th element (in reading order) whose value holds
    ``needle``, or None. Case matters; whitespace runs in ``needle`` match the element's own spacing only exactly."""
    if not needle:
        return None
    count, seen = 0, [0]
    found: list[tuple[Any, int, str]] = []

    def visit(el: Any, depth: int) -> bool:
        nonlocal count
        if depth > max_depth or seen[0] >= max_nodes:
            return False
        seen[0] += 1
        role = str(ax.get(el, "AXRole") or "")
        if role in _TEXTY_ROLES:
            val = ax.get(el, "AXValue")
            if isinstance(val, str):
                start = 0
                while True:
                    i = val.find(needle, start)
                    if i < 0:
                        break
                    count += 1
                    if count == occurrence:
                        found.append((el, i, val))
                        return True
                    start = i + 1
        return any(visit(k, depth + 1) for k in ax.children(el))

    visit(root, 0)
    return found[0] if found else None


def block_cells(anchor: str, rows: Sequence[Sequence[Any]]) -> dict[tuple[int, int], str] | None:
    """The cells a block of ``rows`` fills from ``anchor`` (its top-left cell), as {(column, row): text}; None when
    ``anchor`` is not one cell."""
    addr = cell_address(anchor)
    if addr is None:
        return None
    c0, r0 = addr
    out: dict[tuple[int, int], str] = {}
    for dr, row in enumerate(rows):
        for dc, v in enumerate(row):
            out[(c0 + dc, r0 + dr)] = "" if v is None else str(v)
    return out


def _as_number(s: str) -> float | None:
    t = str(s).strip().replace(",", "").replace(" ", "")
    pct = t.endswith("%")
    try:
        n = float(t.rstrip("%"))
    except ValueError:
        return None
    return n / 100.0 if pct else n


def cell_mismatches(wanted: dict[tuple[int, int], str], shown: dict[tuple[int, int], str]) -> list[str]:
    """Cells whose shown value is not what was typed, as "B3 shows 'x' (typed 'y')". Formulas (typed with '=')
    are skipped (the cell shows the result), numbers compare as numbers ('1,000' is 1000), and cells not on
    screen are not judged."""
    out = []
    for (c, r), want in sorted(wanted.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if want.startswith("=") or (c, r) not in shown:
            continue
        got = shown[(c, r)]
        if got.strip() == want.strip():
            continue
        a, b = _as_number(got), _as_number(want)
        if a is not None and b is not None and abs(a - b) <= 1e-9 * max(1.0, abs(b)):
            continue
        out.append(f"{col_letters(c)}{r} shows {_clip_cell(got)!r} (typed {_clip_cell(want)!r})")
    return out


def cycle_to_option(read: Callable[[], Any], step: Callable[[str], None], want: str,
                    max_steps: int = 80) -> tuple[str | None, list[str]]:
    """Choose ``want`` in a focused pop-up with the arrow keys: ``(the option chosen, every option seen)``; the
    option is None when none matched (the pop-up is put back on the option it had).

    For pop-ups whose menu is not in the Accessibility tree (LibreOffice's list boxes): Up to the first option,
    Down through the rest reading each, then Up to the top and Down to the best match (``match_label``)."""
    def top() -> None:
        for _ in range(max_steps):
            before = str(read() or "")
            step("up")
            if str(read() or "") == before:
                return

    start = str(read() or "")
    top()
    options: list[str] = []
    for _ in range(max_steps):
        cur = str(read() or "")
        if cur in options:
            break
        options.append(cur)
        step("down")
        if str(read() or "") == cur:            # the last option
            break
    i, _tied = match_label(want, options)
    target = options[i] if i is not None else start
    top()
    for _ in range(options.index(target) if target in options else 0):
        step("down")
    return (target if i is not None and str(read() or "") == target else None), options
