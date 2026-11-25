"""The structured desktop tools (observe, menus, dialogs, tables) and the desktop run policies, on fake AX trees.

Nothing here touches the real Accessibility API: the logic lives in ``macos_ax_struct`` and the tools take an env
(``macos_structured_tools.build_structured_tools(tk, env)``), so a fake tree stands in for an app.
"""

from __future__ import annotations

import asyncio

import pytest

from tools.navigation.computer import macos_ax_struct as axs
from tools.navigation.computer.macos_app_cards import CARDS, card_for, card_key
from tools.navigation.computer.macos_structured_tools import STRUCTURED_TOOL_NAMES, build_structured_tools


# ── A fake AX tree ──────────────────────────────────────────────────────────

class N:
    def __init__(self, role: str, *children: "N", **attrs):
        self.a = {"AXRole": role, **attrs}
        self.kids = list(children)

    def __repr__(self):
        return f"N({self.a.get('AXRole')}, {self.a.get('AXTitle')!r})"


class FakeAX:
    def __init__(self):
        self.pressed: list[str] = []
        self.escapes = 0

    def get(self, el, attr):
        return el.a.get(attr)

    def children(self, el):
        return list(el.kids)

    def set(self, el, attr, value):
        if el.a.get("readonly"):
            return False
        if el.a.get("ignore_set") and attr != "AXFocused":      # accepted and ignored, like LibreOffice's
            return True
        el.a[attr] = value
        if attr == "AXSelectedChildren":                # like LibreOffice: the list's selection drives its items
            for k in el.kids:
                k.a["AXSelected"] = any(k is v for v in value)
        return True

    def act(self, el, action):
        self.pressed.append(str(el.a.get("AXTitle") or el.a.get("AXDescription") or el.a.get("AXRole")))
        cb = el.a.get("on_press")
        if cb:
            cb(el)
        elif el.a.get("AXRole") in ("AXCheckBox", "AXRadioButton"):
            el.a["AXValue"] = 0 if el.a.get("AXValue") else 1
        return el.a.get("press_ok", True)

    def escape(self):
        self.escapes += 1

    def same(self, a, b):
        return a is b


def item(title, *kids, **attrs):
    return N("AXMenuItem", *kids, AXTitle=title, **attrs)


def menu(*items):
    return N("AXMenu", *items)


def menubar():
    return N("AXMenuBar",
             N("AXMenuBarItem", menu(item("About This Mac")), AXTitle="Apple"),
             N("AXMenuBarItem", menu(item("New", AXMenuItemCmdChar="n"), item("Save As…", AXMenuItemCmdChar="s",
                                                                                AXMenuItemCmdModifiers=1)),
               AXTitle="File"),
             N("AXMenuBarItem", menu(
                 item("Cells…", AXMenuItemCmdChar="1"),
                 item("Merge and Unmerge Cells", menu(item("Merge Cells"), item("Unmerge Cells", AXEnabled=False))),
                 item("Freeze Rows"), item("Freeze Columns")), AXTitle="Format"))


# ── Labels and menus ────────────────────────────────────────────────────────

def test_labels_match_without_ellipsis_case_or_colon():
    assert axs.norm_label("Cells…") == axs.norm_label("cells...") == axs.norm_label("Cells") == "cells"
    assert axs.norm_label("Decimal places:") == "decimal places"
    assert axs.match_label("cells", ["Cells…", "Cell Styles"])[0] == 0
    assert axs.match_label("Freeze", ["Freeze Rows", "Freeze Columns"]) == (None, [0, 1])
    assert axs.parse_menu_path("Format > Merge and Unmerge Cells ▸ Merge Cells") == [
        "Format", "Merge and Unmerge Cells", "Merge Cells"]


def test_shortcuts_are_drawn_like_menus():
    assert axs.format_shortcut("s", 1) == "⇧⌘S"
    assert axs.format_shortcut("x", 8) == "X"
    assert axs.format_shortcut("", 0) == ""


def test_select_menu_presses_the_leaf():
    ax, bar = FakeAX(), menubar()
    res = axs.select_menu(ax, bar, ["format", "merge and unmerge cells", "merge cells"])
    assert res.ok and res.chosen == ["Format", "Merge and Unmerge Cells", "Merge Cells"]
    assert ax.pressed == ["Merge Cells"]


def test_select_menu_explains_what_went_wrong():
    ax, bar = FakeAX(), menubar()
    res = axs.select_menu(ax, bar, ["Format", "Freeze"])
    assert not res.ok and "several" in res.error and res.options == ["Freeze Rows", "Freeze Columns"]
    res = axs.select_menu(ax, bar, ["Format", "Sort"])
    assert not res.ok and "no item 'Sort'" in res.error and "Cells…" in res.options
    res = axs.select_menu(ax, bar, ["Format", "Merge and Unmerge Cells"])
    assert not res.ok and "is a menu" in res.error and res.options == ["Merge Cells", "Unmerge Cells"]
    res = axs.select_menu(ax, bar, ["Format", "Merge and Unmerge Cells", "Unmerge Cells"])
    assert not res.ok and "disabled" in res.error
    assert ax.pressed == []


def test_select_menu_opens_a_menu_built_on_demand_and_closes_it_on_failure():
    lazy = N("AXMenuBarItem", AXTitle="Image")

    def fill(el):
        el.kids = [menu(item("Mode", menu(item("Grayscale"))))]
    lazy.a["on_press"] = fill
    bar = N("AXMenuBar", lazy)
    ax = FakeAX()
    assert axs.select_menu(ax, bar, ["Image", "Mode", "Grayscale"]).ok
    ax2 = FakeAX()
    lazy.kids = []
    assert not axs.select_menu(ax2, bar, ["Image", "Nope"]).ok
    assert ax2.escapes == 2


def test_walk_and_search_menus():
    ax, bar = FakeAX(), menubar()
    hits = axs.walk_menus(ax, bar)
    paths = [" > ".join(h.path) for h in hits]
    assert "Apple > About This Mac" not in paths
    assert "Format > Merge and Unmerge Cells > Merge Cells" in paths
    found = axs.search_menus(hits, "save")
    assert found[0].line() == "File ▸ Save As…   ⇧⌘S"
    unmerge = axs.search_menus(hits, "unmerge")[0]
    assert "(disabled now)" in unmerge.line()
    assert [h.path[-1] for h in axs.search_menus(hits, "freeze")] == ["Freeze Rows", "Freeze Columns"]


# ── Dialogs ─────────────────────────────────────────────────────────────────

def format_cells_dialog():
    return N("AXWindow",
             N("AXTabGroup",
               N("AXRadioButton", AXTitle="Numbers", AXValue=1),
               N("AXRadioButton", AXTitle="Font", AXValue=0),
               N("AXGroup",
                 N("AXStaticText", AXValue="Decimal places:"),
                 N("AXTextField", AXValue="0"),
                 N("AXCheckBox", AXTitle="Negative numbers red", AXValue=0),
                 N("AXRadioButton", AXTitle="Left", AXValue=1),
                 N("AXRadioButton", AXTitle="Center", AXValue=0),
                 N("AXPopUpButton", menu(item("General"), item("Percent"), item("Currency")),
                   AXTitle="Category", AXValue="General"),
                 )),
             N("AXButton", AXTitle="OK"), N("AXButton", AXTitle="Cancel"),
             AXTitle="Format Cells", AXSubrole="AXDialog")


def test_find_dialog_prefers_sheets_then_dialogs_then_a_front_window():
    ax = FakeAX()
    main = N("AXWindow", AXTitle="Doc")
    sheet_win = N("AXWindow", N("AXSheet"), AXTitle="Doc2")
    app = N("AXApplication", AXWindows=[main, sheet_win], AXMainWindow=main, AXFocusedWindow=main)
    assert axs.find_dialog(ax, app)[1] == "sheet on 'Doc2'"
    dlg = format_cells_dialog()
    app = N("AXApplication", AXWindows=[main, dlg], AXMainWindow=main, AXFocusedWindow=main)
    assert axs.find_dialog(ax, app)[1] == "dialog 'Format Cells'"
    other = N("AXWindow", AXTitle="Chart Wizard")
    app = N("AXApplication", AXWindows=[main, other], AXMainWindow=main, AXFocusedWindow=other)
    assert "Chart Wizard" in axs.find_dialog(ax, app)[1]
    app = N("AXApplication", AXWindows=[main], AXMainWindow=main, AXFocusedWindow=main)
    assert axs.find_dialog(ax, app) is None


def test_collect_controls_labels_fields_from_the_text_before_them():
    ax = FakeAX()
    cs = axs.collect_controls(ax, format_cells_dialog())
    kinds = [(c.kind, c.label, c.value) for c in cs]
    assert ("tab", "Numbers", "on") in kinds and ("tab", "Font", "off") in kinds
    assert ("text", "Decimal places:", "0") in kinds
    assert ("radio", "Left", "on") in kinds          # inside a tab's page: a choice, not a tab
    assert ("popup", "Category", "General") in kinds
    text = axs.format_dialog("dialog 'Format Cells'", cs)
    assert text.startswith("Open: dialog 'Format Cells'") and "Buttons:" in text and "Fields:" in text


# ── Tables ──────────────────────────────────────────────────────────────────

def test_ranges_and_cells():
    assert axs.parse_range("B2:D9") == (2, 2, 4, 9)
    assert axs.parse_range("$AA$10") == (27, 10, 27, 10)
    assert axs.parse_range("nope") is None
    assert axs.col_letters(28) == "AB"
    grid = N("AXTable", *[N("AXCell", AXDescription=f"{c}{r}", AXValue=f"{c}{r}v")
                          for r in (1, 2, 3) for c in "ABC"])
    cells = axs.read_cells(FakeAX(), N("AXWindow", grid))
    assert len(cells) == 9
    out = axs.format_cells(cells, axs.parse_range("B2:C3")).splitlines()
    assert out == ["\tB\tC", "2\tB2v\tC2v", "3\tB3v\tC3v"]


def test_the_name_box_is_not_read_as_a_cell():
    grid = N("AXWindow", N("AXTextArea", AXDescription="A1", AXValue="A1"),        # the Name Box showing A1
             N("AXTable", N("AXCell", AXDescription="A1", AXValue="Sales Rep")))
    assert axs.read_cells(FakeAX(), grid) == {(1, 1): "Sales Rep"}


def test_ax_tables_read_rows_and_header():
    ax = FakeAX()
    rows = [N("AXRow", N("AXCell", N("AXStaticText", AXValue=n)), N("AXCell", AXValue=s)) for n, s in
            (("a.txt", "1 KB"), ("b.txt", "2 KB"))]
    t = N("AXTable", *rows, AXRows=rows, AXColumns=[N("AXColumn", AXTitle="Name"), N("AXColumn", AXTitle="Size")])
    name, header, data = axs.read_ax_tables(ax, N("AXWindow", t))[0]
    assert header == ["Name", "Size"] and data == [["a.txt", "1 KB"], ["b.txt", "2 KB"]]


# ── Deltas and interrupts ───────────────────────────────────────────────────

def fmt(it):
    return f"[{it[0]}] {it[1]} '{it[2]}'"


def test_controls_delta_only_when_indices_hold():
    prev = [(1, "B", "OK"), (2, "TF", "Name"), (3, "B", "Cancel")]
    assert axs.controls_delta(None, prev, fmt) is None
    assert axs.controls_delta(prev, prev, fmt).startswith("Same 3 controls")
    grown = prev + [(4, "ST", "Saved")]
    d = axs.controls_delta(prev, grown, fmt)
    assert "+ [4] ST 'Saved'" in d
    shifted = [(1, "B", "OK"), (2, "ST", "New"), (3, "TF", "Name"), (4, "B", "Cancel")]
    assert axs.controls_delta(prev, shifted, fmt) is None
    replaced = [(1, "X", "a"), (2, "X", "b"), (3, "X", "c")]
    assert axs.controls_delta(prev, replaced, fmt) is None


def test_windows_above_finds_alerts_and_other_apps():
    wl = [
        {"kCGWindowOwnerName": "Dock", "kCGWindowLayer": 20},
        {"kCGWindowOwnerName": "UserNotificationCenter", "kCGWindowName": "Allow?", "kCGWindowLayer": 8},
        {"kCGWindowOwnerName": "Finder", "kCGWindowName": "Copy", "kCGWindowLayer": 0, "kCGWindowOwnerPID": 9},
        {"kCGWindowOwnerName": "LibreOffice", "kCGWindowLayer": 0, "kCGWindowOwnerPID": 5},
        {"kCGWindowOwnerName": "Safari", "kCGWindowLayer": 0},
    ]
    assert axs.windows_above(wl, "LibreOffice", 5) == [("UserNotificationCenter", "Allow?"), ("Finder", "Copy")]


# ── App cards ───────────────────────────────────────────────────────────────

def test_cards_resolve_shown_and_process_names_only():
    assert card_key("soffice") == card_key("LibreOffice Calc") == card_key("LibreOffice") == "libreoffice"
    assert card_key("GIMP-3.2") == card_key("gimp") == "gimp"
    assert card_key("Calculator") is None and card_key("Chess") is None
    assert "Use Excel 2007-365!" in card_for("LibreOffice")


def test_cards_hold_no_benchmark_material():
    text = "\n".join(CARDS.values()).lower()
    for word in ("osworld", "macosworld", "checker", "evaluator", "/users/", "task_id"):
        assert word not in text


# ── The tools, with a fake env ──────────────────────────────────────────────

class FakeTK:
    def __init__(self):
        self._element_registry = {1: object(), 2: object()}
        self._coord_registry = {}
        self.invalidated = []

    def _invalidate_scan_cache(self, app_name=None):
        self.invalidated.append(app_name)


class FakeEnv:
    def __init__(self, app_el, apps=None, windows=None):
        self.access = FakeAX()
        self.app_el = app_el
        self.apps = apps or [{"name": "LibreOffice", "pid": 5, "bundle_id": "org.libreoffice.script",
                              "executable": "soffice", "active": True},
                             {"name": "Finder", "pid": 9, "bundle_id": "com.apple.finder",
                              "executable": "Finder", "active": False}]
        self.windows = windows or []
        self.activated = []

    def running_apps(self):
        return [dict(a) for a in self.apps]

    def app_element(self, pid):
        return self.app_el

    def window_list(self):
        return self.windows

    def frame(self, el):
        return (10, 20, 100, 30)

    def activate(self, name):
        self.activated.append(name)
        return True

    def press_key(self, key):
        self.keys = getattr(self, "keys", []) + [key]
        focused = getattr(self, "focused_checkbox", None)
        if key == "space" and focused is not None:
            focused.a["AXValue"] = 0 if focused.a.get("AXValue") else 1

    def replace_text(self, el, text):
        self.typed = getattr(self, "typed", []) + [text]
        el.a["AXValue"] = text

    def sleep(self, s):
        pass


def call(tools, name, **args):
    return asyncio.run(tools[name].ainvoke(args))


def calc_app(dialog=None):
    addrs = [(c, r) for r in (1, 2) for c in "AB"]
    main = N("AXWindow", N("AXTable", *[N("AXCell", AXDescription=f"{c}{r}", AXValue=str(i))
                                       for i, (c, r) in enumerate(addrs)]),
             AXTitle="data.xlsx - LibreOffice Calc", AXDocument="file:///Users/u/Desktop/data%20set.xlsx")
    wins = [main] + ([dialog] if dialog else [])
    return N("AXApplication", AXWindows=wins, AXMainWindow=main, AXFocusedWindow=dialog or main,
             AXMenuBar=menubar(), AXFocusedUIElement=N("AXTextField", AXDescription="Name Box", AXValue="A1"))


def test_tool_names_are_the_declared_set():
    tools = build_structured_tools(FakeTK(), FakeEnv(calc_app()))
    assert tuple(tools) == STRUCTURED_TOOL_NAMES


def test_observe_reports_app_process_file_focus_dialog_and_alerts():
    env = FakeEnv(calc_app(format_cells_dialog()),
                  windows=[{"kCGWindowOwnerName": "SecurityAgent", "kCGWindowName": "", "kCGWindowLayer": 9},
                           {"kCGWindowOwnerName": "LibreOffice", "kCGWindowOwnerPID": 5, "kCGWindowLayer": 0}])
    out = call(build_structured_tools(FakeTK(), env), "observe")
    assert "App: LibreOffice (frontmost; process 'soffice'" in out
    assert "file: /Users/u/Desktop/data set.xlsx" in out
    assert "Focus: TextField 'Name Box' = 'A1'" in out
    assert "Dialog: dialog 'Format Cells' (buttons: OK, Cancel)" in out
    assert "On top of it: SecurityAgent" in out
    assert "Other running apps: Finder" in out and "app_guide('LibreOffice')" in out


def test_menu_select_reports_the_dialog_it_opened():
    app = calc_app()
    dlg = format_cells_dialog()

    def open_dialog(el):
        app.a["AXWindows"].append(dlg)
        app.a["AXFocusedWindow"] = dlg
    app.a["AXMenuBar"].kids[2].kids[0].kids[0].a["on_press"] = open_dialog
    tk = FakeTK()
    tools = build_structured_tools(tk, FakeEnv(app))
    out = call(tools, "menu_select", app_name="soffice", path="Format > Cells")
    assert out.startswith("Chose Format ▸ Cells… in LibreOffice.")
    assert "Now open: dialog 'Format Cells'" in out and tk.invalidated == ["LibreOffice"]
    bad = call(tools, "menu_select", app_name="LibreOffice", path="Format > Sort")
    assert "no item 'Sort'" in bad and "menu_search" in bad


def test_menu_select_activates_an_app_in_the_background():
    env = FakeEnv(calc_app())
    env.apps[0]["active"] = False
    call(build_structured_tools(FakeTK(), env), "menu_select", app_name="LibreOffice", path="Format > Freeze Rows")
    assert env.activated == ["LibreOffice"]


def test_menu_search_lists_paths():
    out = call(build_structured_tools(FakeTK(), FakeEnv(calc_app())), "menu_search", app_name="LibreOffice",
               query="merge")
    assert "Format ▸ Merge and Unmerge Cells ▸ Merge Cells" in out


def test_read_dialog_registers_indices_after_the_scan():
    tk = FakeTK()
    out = call(build_structured_tools(tk, FakeEnv(calc_app(format_cells_dialog()))), "read_dialog")
    assert "[3]TAB 'Numbers' = on" in out and "TF 'Decimal places:' = '0'" in out
    assert max(tk._element_registry) > 3 and tk._coord_registry[3] == (60, 35)


def test_read_dialog_falls_back_to_the_front_window():
    out = call(build_structured_tools(FakeTK(), FakeEnv(calc_app())), "read_dialog")
    assert out.startswith("No dialog or sheet is open in LibreOffice")


def test_fill_dialog_sets_text_checkbox_radio_popup_then_presses():
    dlg = format_cells_dialog()
    app = calc_app(dlg)

    def close(el):
        app.a["AXWindows"].remove(dlg)
        app.a["AXFocusedWindow"] = app.a["AXMainWindow"]
    ok_button = dlg.kids[1]
    ok_button.a["on_press"] = close
    env = FakeEnv(app)
    out = call(build_structured_tools(FakeTK(), env), "fill_dialog",
               fields={"Decimal places": "2", "Negative numbers red": True, "Center": True, "Category": "percent"},
               press="OK")
    assert out.count("✓") == 4, out
    assert "Pressed 'OK'. No dialog is open now." in out
    group = dlg.kids[0].kids[2]
    assert group.kids[1].a["AXValue"] == "2"
    assert group.kids[2].a["AXValue"] == 1
    assert group.kids[4].a["AXValue"] == 1
    assert "Percent" in env.access.pressed


def test_fill_dialog_does_not_press_when_a_field_fails():
    dlg = format_cells_dialog()
    out = call(build_structured_tools(FakeTK(), FakeEnv(calc_app(dlg))), "fill_dialog",
               fields='{"Thousands separator": true}', press="OK")
    assert "no such field" in out and "Did not press 'OK'" in out


def test_fill_dialog_matches_a_radio_by_its_option():
    dlg = format_cells_dialog()
    out = call(build_structured_tools(FakeTK(), FakeEnv(calc_app(dlg))), "fill_dialog",
               fields={"Alignment": "Center"})
    assert "✓" in out and dlg.kids[0].kids[2].kids[4].a["AXValue"] == 1


def test_read_table_reads_on_screen_cells():
    tools = build_structured_tools(FakeTK(), FakeEnv(calc_app()))
    assert call(tools, "read_table").splitlines() == ["\tA\tB", "1\t0\t1", "2\t2\t3"]
    assert "None of that range is on screen" in call(tools, "read_table", range_ref="Z90:Z99")
    assert "not a cell range" in call(tools, "read_table", range_ref="hello")


def test_get_control_values_reads_many():
    tk = FakeTK()
    tk._element_registry = {4: N("AXTextField", AXTitle="Name", AXValue="x"), 5: N("AXCheckBox", AXValue=1)}
    out = call(build_structured_tools(tk, FakeEnv(calc_app())), "get_control_values", indices="[4, 5, 6]")
    assert out.splitlines() == ["[4] TextField 'Name' = 'x'", "[5] CheckBox = '1'",
                                "[6] not in the last read; call get_screen_controls or read_dialog again"]


def test_wait_until_title_and_dialog():
    tools = build_structured_tools(FakeTK(), FakeEnv(calc_app()))
    assert call(tools, "wait_until", title="data.xlsx").startswith("Done")
    assert call(tools, "wait_until", dialog="open", timeout=0.5).startswith("Not yet")
    assert call(tools, "wait_until", text="Nothing here", timeout=0.5).startswith("Not yet")


def test_app_guide():
    tools = build_structured_tools(FakeTK(), FakeEnv(calc_app()))
    assert "soffice" in call(tools, "app_guide", app="LibreOffice Calc")
    assert "No guide for 'Chess'" in call(tools, "app_guide", app="Chess")


# ── Run policies ────────────────────────────────────────────────────────────

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from backend import desktop_run_policy_middleware as pol  # noqa: E402


def test_parse_text_calls_hermes_and_xml():
    assert pol.parse_text_calls("plain answer") == []
    assert pol.parse_text_calls('<tool_call>{"name": "observe", "arguments": {}}</tool_call>') == [
        {"name": "observe", "args": {}}]
    xml = "<tool_call>\n<function=menu_select>\n<parameter=app_name>\nLibreOffice\n</parameter>\n" \
          "<parameter=path>\nFormat > Cells\n</parameter>\n</function>\n</tool_call>"
    assert pol.parse_text_calls(xml) == [{"name": "menu_select",
                                          "args": {"app_name": "LibreOffice", "path": "Format > Cells"}}]
    assert pol.parse_text_calls("<tool_call>{broken</tool_call>") is None


class _Tool:
    def __init__(self, name, result="App: X"):
        self.name, self.result, self.calls = name, result, 0

    def invoke(self, args):
        self.calls += 1
        return self.result

    async def ainvoke(self, args):
        return self.invoke(args)


class _Req:
    def __init__(self, messages, tools):
        self.messages, self.tools = messages, tools

    def override(self, **kw):
        r = _Req(kw.get("messages", self.messages), kw.get("tools", self.tools))
        return r


def _resp(msg):
    from langchain.agents.middleware.types import ModelResponse
    return ModelResponse(result=[msg])


def test_text_tool_call_becomes_a_real_call():
    mw = pol.DesktopRunPolicyMiddleware(pol.BENCH_AGENT)
    seen = []

    def handler(req):
        seen.append(req)
        return _resp(AIMessage(content='Let me look. <tool_call>{"name": "observe", "arguments": {}}</tool_call>'))
    out = mw.wrap_model_call(_Req([HumanMessage(content="task", id="h1")], [_Tool("observe")]), handler)
    ai = out.result[0]
    assert ai.tool_calls[0]["name"] == "observe" and ai.content == "Let me look."
    assert len(seen) == 1
    assert pol.START_TAG in seen[0].messages[1].content          # the start screen follows the request


def test_unknown_text_call_and_empty_first_turn_get_one_retry():
    mw = pol.DesktopRunPolicyMiddleware(pol.BENCH_AGENT)
    replies = [AIMessage(content="<tool_call>{\"name\": \"rm_rf\", \"arguments\": {}}</tool_call>"),
               AIMessage(content="still nothing")]
    seen = []

    def handler(req):
        seen.append(req)
        return _resp(replies[len(seen) - 1])
    out = mw.wrap_model_call(_Req([HumanMessage(content="task", id="h2")], [_Tool("observe")]), handler)
    assert len(seen) == 2 and out.result[0].content == "still nothing"
    assert seen[1].messages[-1].content.startswith("[harness] Your last reply wrote a tool call as text")

    mw2 = pol.DesktopRunPolicyMiddleware(pol.BENCH_AGENT)
    seen.clear()
    replies[:] = [AIMessage(content="Here is my plan."), AIMessage(content="", tool_calls=[
        {"name": "observe", "args": {}, "id": "c1", "type": "tool_call"}])]
    out = mw2.wrap_model_call(_Req([HumanMessage(content="task", id="h3")], [_Tool("observe")]), handler)
    assert len(seen) == 2 and out.result[0].tool_calls


def test_app_agent_final_answer_is_left_alone():
    mw = pol.DesktopRunPolicyMiddleware(pol.APP_AGENT)
    calls = []
    out = mw.wrap_model_call(_Req([HumanMessage(content="hi", id="h4")], [_Tool("observe")]),
                             lambda r: calls.append(r) or _resp(AIMessage(content="Done.")))
    assert len(calls) == 1 and out.result[0].content == "Done."


def test_other_agents_are_untouched():
    mw = pol.DesktopRunPolicyMiddleware("coding-agent")
    tool = _Tool("observe")
    req = _Req([HumanMessage(content="x")], [tool])
    mw.wrap_model_call(req, lambda r: _resp(AIMessage(content="<tool_call>{}</tool_call>")))
    assert tool.calls == 0


def test_start_state_is_read_once_per_request():
    mw = pol.DesktopRunPolicyMiddleware(pol.APP_AGENT)
    tool = _Tool("observe", "App: Notes")
    h = HumanMessage(content="do it", id="h5")
    mw.wrap_model_call(_Req([h], [tool]), lambda r: _resp(AIMessage(content="ok")))
    seen = []
    later = [h, AIMessage(content="", tool_calls=[{"name": "observe", "args": {}, "id": "c", "type": "tool_call"}]),
             ToolMessage(content="x", tool_call_id="c", name="observe")]
    mw.wrap_model_call(_Req(later, [tool]), lambda r: seen.append(r) or _resp(AIMessage(content="ok")))
    assert tool.calls == 1 and "App: Notes" in seen[0].messages[1].content


def test_failure_streak_hint_after_two():
    msgs = [HumanMessage(content="t"),
            AIMessage(content="", tool_calls=[{"name": "run_osascript", "args": {}, "id": "a", "type": "tool_call"}]),
            ToolMessage(content='{"ok": false, "error": "-1728"}', tool_call_id="a", name="run_osascript")]
    assert pol.fail_streak(msgs, "run_osascript") == 1

    class Req:
        tool_call = {"name": "run_osascript"}
        state = {"messages": msgs}
    mw = pol.DesktopRunPolicyMiddleware(pol.APP_AGENT)
    out = mw.wrap_tool_call(Req(), lambda r: ToolMessage(content="execution error: nope", tool_call_id="b",
                                                         name="run_osascript"))
    assert "failed 2 times in a row" in out.content and "menu_select" in out.content
    ok = mw.wrap_tool_call(Req(), lambda r: ToolMessage(content="fine", tool_call_id="b", name="run_osascript"))
    assert "[harness]" not in ok.content


def test_budget_note_warns_when_three_or_fewer_steps_are_left():
    from backend.macarena_native import budget_note
    assert "ONLY 3 LEFT" in budget_note({"steps_used": 12, "steps_left": 3})
    assert "vm_fail" in budget_note({"steps_used": 14, "steps_left": 1})
    assert "LEFT" not in budget_note({"steps_used": 2, "steps_left": 13})
    assert "RUN OVER" in budget_note({"steps_used": 15, "steps_left": 0, "finished": "budget"})


# ── LibreOffice as seen in the benchmark VM (probe of 2026-11-21) ─────────────

def chart_wizard():
    return N("AXWindow",
             N("AXButton", AXTitle="Next >"), N("AXButton", AXTitle="Finish"),
             N("AXGroup",
               N("AXStaticText", AXValue="Choose a Chart Type"),
               N("AXMenu", *[N("AXMenuItem", AXTitle=t, AXSubrole="AXOutlineRow", AXSelected=(t == "Column"))
                             for t in ("Column", "Bar", "Pie", "Line")], AXHelp="Select a basic chart type."),
               N("AXCheckBox", AXTitle="3D Look", AXValue=0)),
             AXTitle="Chart Wizard", AXSubrole="AXStandardWindow")


def test_libreoffice_dialog_is_a_docless_window_that_is_also_main():
    ax = FakeAX()
    doc = N("AXWindow", AXTitle="Sheet.xlsx", AXSubrole="AXStandardWindow", AXDocument="file:///Users/a/Sheet.xlsx")
    wiz = chart_wizard()
    app = N("AXApplication", AXWindows=[wiz, doc], AXMainWindow=wiz, AXFocusedWindow=wiz)
    assert axs.find_dialog(ax, app) == (wiz, "window 'Chart Wizard' (in front of the document)")
    app = N("AXApplication", AXWindows=[doc], AXMainWindow=doc, AXFocusedWindow=doc)
    assert axs.find_dialog(ax, app) is None


def test_a_minimized_document_reporting_axdialog_is_not_a_dialog():
    ax = FakeAX()
    doc = N("AXWindow", AXTitle="Sheet.xlsx", AXSubrole="AXDialog", AXMinimized=True, AXDocument="file:///a.xlsx")
    app = N("AXApplication", AXWindows=[doc], AXMainWindow=None, AXFocusedWindow=None)
    assert axs.find_dialog(ax, app) is None


def test_list_boxes_are_read_and_set_by_option():
    wiz = chart_wizard()
    cs = axs.collect_controls(FakeAX(), wiz)
    lst = next(c for c in cs if c.kind == "list")
    assert lst.label == "Choose a Chart Type" and lst.value == "Column"
    assert "options: Column | Bar | Pie | Line" in lst.line()
    doc = N("AXWindow", AXTitle="Sheet.xlsx", AXDocument="file:///a.xlsx")
    app = N("AXApplication", AXWindows=[wiz, doc], AXMainWindow=wiz, AXFocusedWindow=wiz, AXMenuBar=menubar())
    out = call(build_structured_tools(FakeTK(), FakeEnv(app)), "fill_dialog",
               fields={"Chart Type": "line"}, press="Next >")
    assert "✓" in out and "Pressed 'Next >'" in out, out
    assert wiz.kids[2].kids[1].a["AXSelectedChildren"] == [wiz.kids[2].kids[1].kids[3]]


def test_a_selection_the_app_ignores_is_reported_not_claimed():
    wiz = chart_wizard()
    doc = N("AXWindow", AXTitle="Sheet.xlsx", AXDocument="file:///a.xlsx")
    app = N("AXApplication", AXWindows=[wiz, doc], AXMainWindow=wiz, AXFocusedWindow=wiz, AXMenuBar=menubar())
    for item in wiz.kids[2].kids[1].kids:
        item.a["readonly"] = True
        item.a["press_ok"] = True
    wiz.kids[2].kids[1].a["readonly"] = True
    out = call(build_structured_tools(FakeTK(), FakeEnv(app)), "fill_dialog", fields={"Chart Type": "Line"},
               press="Next >")
    assert "did not select 'Line'" in out and "Did not press" in out


def test_libreoffice_tab_pages_inside_the_tab_and_spin_fields_are_read():
    """Live Format Cells: the Numbers page sits inside its AXRadioButton, every tab reports 0, spin fields are
    AXUnknown with a title and a value."""
    ax = FakeAX()
    dlg = N("AXWindow",
            N("AXTabGroup",
              N("AXRadioButton",
                N("AXGroup",
                  N("AXStaticText", AXValue="Category"),
                  N("AXMenu", N("AXMenuItem", AXTitle="Number"), N("AXMenuItem", AXTitle="Percent"),
                    AXTitle="Category"),
                  N("AXStaticText", AXValue="Decimal places:"),
                  N("AXUnknown", AXTitle="Decimal places:", AXValue="2"),
                  N("AXTextArea", AXTitle="Format Code", AXValue="General")),
                AXTitle="Numbers", AXValue=0),
              N("AXRadioButton", AXTitle="Font", AXValue=0)),
            N("AXButton", AXTitle="OK"), AXTitle="Format Cells")
    got = {(c.kind, c.label): c.value for c in axs.collect_controls(ax, dlg)}
    assert got[("tab", "Numbers")] == "on" and got[("tab", "Font")] == "off"
    assert got[("text", "Decimal places:")] == "2" and got[("text", "Format Code")] == "General"
    assert ("list", "Category") in got


def test_fill_dialog_falls_back_to_keys_when_libreoffice_ignores_ax_sets():
    """Live Format Cells: AXPress on a checkbox and AXValue on a spin field are accepted and change nothing;
    Space on the focused checkbox and typing over the field do."""
    cb = N("AXCheckBox", AXTitle="Thousands separator", AXValue=0, ignore_set=True, on_press=lambda el: None)
    spin = N("AXUnknown", AXTitle="Decimal places:", AXValue="", ignore_set=True)
    dlg = N("AXWindow", N("AXGroup", spin, cb), N("AXButton", AXTitle="OK"), AXTitle="Format Cells")
    doc = N("AXWindow", AXTitle="Sheet.xlsx", AXDocument="file:///a.xlsx")
    app = N("AXApplication", AXWindows=[dlg, doc], AXMainWindow=dlg, AXFocusedWindow=dlg, AXMenuBar=menubar())
    env = FakeEnv(app)
    env.focused_checkbox = cb
    out = call(build_structured_tools(FakeTK(), env), "fill_dialog",
               fields={"Decimal places": "3", "Thousands separator": True})
    assert out.count("✓") == 2, out
    assert spin.a["AXValue"] == "3" and cb.a["AXValue"] == 1 and env.keys == ["space"]


def test_a_checkbox_that_stays_off_is_reported():
    cb = N("AXCheckBox", AXTitle="Wrap", AXValue=0, ignore_set=True, on_press=lambda el: None)
    dlg = N("AXWindow", cb, N("AXButton", AXTitle="OK"), AXTitle="Format Cells")
    doc = N("AXWindow", AXTitle="Sheet.xlsx", AXDocument="file:///a.xlsx")
    app = N("AXApplication", AXWindows=[dlg, doc], AXMainWindow=dlg, AXFocusedWindow=dlg, AXMenuBar=menubar())
    out = call(build_structured_tools(FakeTK(), FakeEnv(app)), "fill_dialog", fields={"Wrap": True}, press="OK")
    assert "did not change it (still off)" in out and "Did not press" in out


def test_a_libreoffice_popup_shows_its_choice_not_zero():
    pb = N("AXPopUpButton", N("AXTextArea", AXTitle="Language", AXValue="Default - English (USA)"),
           AXTitle="Language", AXValue=0)
    got = axs.collect_controls(FakeAX(), N("AXWindow", pb))
    assert [(c.kind, c.value) for c in got] == [("popup", "Default - English (USA)")]


def test_an_alert_shows_its_message():
    """Live: LibreOffice's 'Information' alert read as just an OK button."""
    alert = N("AXWindow", N("AXStaticText", AXValue="You must enter a valid reference or type a valid name for "
                                                    "the selected range."),
              N("AXButton", AXTitle="OK"), AXTitle="Information")
    doc = N("AXWindow", AXTitle="Sheet.xlsx", AXDocument="file:///a.xlsx")
    app = N("AXApplication", AXWindows=[alert, doc], AXMainWindow=alert, AXFocusedWindow=alert, AXMenuBar=menubar())
    out = call(build_structured_tools(FakeTK(), FakeEnv(app)), "read_dialog")
    assert "Message: You must enter a valid reference" in out
    ax = FakeAX()
    assert axs.dialog_messages(ax, format_cells_dialog()) == []         # labels and headings are not messages


# --- the Name Box and formula bar fields of LibreOffice Calc are listed under their names -------------------------

def _walk_fake_tree(monkeypatch, tree, max_elements=100):
    """Run MacOSToolkit._walk over a fake AX tree of (role, {attr: value}, [children]) tuples."""
    from tools.navigation.computer import macos_tools as mt
    key = {"role": mt.kAXRoleAttribute, "title": mt.kAXTitleAttribute, "desc": mt.kAXDescriptionAttribute,
           "value": mt.kAXValueAttribute, "kids": mt.kAXChildrenAttribute, "pos": mt.kAXPositionAttribute,
           "size": mt.kAXSizeAttribute}
    kit = mt.MacOSToolkit.__new__(mt.MacOSToolkit)
    kit.scan_max_depth, kit.scan_max_elements = 20, 100
    kit._web_area_activated = set()
    monkeypatch.setattr(mt, "_ax_point", lambda v: (10, 10))
    monkeypatch.setattr(mt, "_ax_size", lambda v: (100, 20))

    def multi(el, attrs=None, timeout=None):
        role, a, kids = el
        out = {key["role"]: role, key["pos"]: "p", key["size"]: "s", key["kids"]: kids or None}
        out.update({key[k]: v for k, v in a.items()})
        return out
    monkeypatch.setattr(kit, "_ax_get_multi", multi, raising=False)
    monkeypatch.setattr(kit, "_ax_get_safe", lambda el, attr, timeout=None: el[0] if attr == key["role"] else None,
                        raising=False)
    kit.scan_max_elements = max_elements
    results: list = []
    kit._walk(tree, results, "LibreOffice", mt._Rect(0, 0, 1000, 1000), 1e12)
    return [(r["role"], r["label"]) for r in results if not r.get("_structural")]


def test_calc_name_box_and_formula_bar_fields_are_listed_under_their_names(monkeypatch):
    name_box = ("AXGroup", {"desc": "Name Box"}, [
        ("AXComboBox", {"title": "G13", "desc": "G13", "value": "G13"}, [
            ("AXTextArea", {"title": "G13", "value": "G13"}, [])])])
    input_line = ("AXGroup", {"desc": "Input line"}, [
        ("AXGroup", {}, [("AXTextArea", {"desc": "Paragraph: 0 ", "value": ""}, [])])])
    other = ("AXTextArea", {"desc": "Font Name", "value": "Liberation Sans"}, [])
    got = _walk_fake_tree(monkeypatch, ("AXWindow", {}, [name_box, input_line, other]))
    assert ("TextArea", "Name Box") in got
    assert ("TextArea", "Formula bar") in got
    assert ("TextArea", "G13") not in got and ("TextArea", "Paragraph: 0") not in got
    assert ("TextArea", "Font Name") in got      # fields outside those groups keep their own names


# --- a cell reference typed into a Calc cell's own field gets a note pointing at the Name Box ----------------------

_CALC_LABELS = {3: ("TextField", "A1"), 4: ("TextField", "B1"), 405: ("Group", "Name Box"),
                406: ("TextArea", "Name Box"), 410: ("TextArea", "Formula bar"), 436: ("TextArea", "Font Name")}


@pytest.mark.parametrize("text", ["A2:B30", "C12", "$A$1", "Sheet1.A1:A20", "Sheet2.A1", "'My sheet'.B3",
                                  "Sheet2!A1", " d2:d30 "])
def test_a_reference_typed_into_a_cell_points_to_the_name_box(text):
    from tools.navigation.computer.macos_tools import cell_ref_hint
    hint = cell_ref_hint(3, text, _CALC_LABELS)
    assert "#3 is cell A1 itself, not the Name Box" in hint
    assert "Name Box (#406)" in hint                     # the field, not the group around it


@pytest.mark.parametrize("index,text", [
    (3, "=SUM(A1:A3)"), (3, "Total"), (3, "42"), (3, "Year_Profit"), (3, "Q1 2024"),   # ordinary cell contents
    (406, "A2:B30"),                                     # the Name Box itself
    (410, "C12"),                                        # the formula bar
    (436, "A1"),                                         # a field that is not a cell
    (99, "A1"),                                          # unknown index
])
def test_no_note_for_contents_or_for_other_fields(index, text):
    from tools.navigation.computer.macos_tools import cell_ref_hint
    assert cell_ref_hint(index, text, _CALC_LABELS) == ""


def test_the_note_says_how_to_find_the_name_box_when_it_was_not_listed():
    from tools.navigation.computer.macos_tools import cell_ref_hint
    hint = cell_ref_hint(3, "A2:B30", {3: ("TextField", "A1")})
    assert "get_screen_controls(label='Name Box')" in hint


def test_the_name_box_is_listed_even_when_the_grid_fills_the_scan(monkeypatch):
    # Calc's formula toolbar comes after the grid in the tree; a big sheet used to use up the whole scan first.
    grid = ("AXGroup", {}, [("AXTextField", {"title": f"{c}{r}", "value": ""}, [])
                            for r in range(1, 40) for c in "ABCDEFGH"])
    toolbar = ("AXToolbar", {"title": "Formula Tool Bar"}, [
        ("AXGroup", {}, [("AXGroup", {"desc": "Name Box"}, [("AXTextArea", {"title": "A1", "value": "A1"}, [])])]),
        ("AXButton", {"title": "Function Wizard"}, [])])
    window = ("AXWindow", {}, [("AXGroup", {}, [grid, toolbar])])
    got = _walk_fake_tree(monkeypatch, window, max_elements=60)
    assert ("TextArea", "Name Box") in got
    assert got.index(("TextArea", "Name Box")) < got.index(("TextField", "A1"))


def _save_sheet_app(name="Untitled"):
    sheet = N("AXSheet",
              N("AXStaticText", AXValue="Save As:"), N("AXTextField", AXValue=name),
              N("AXButton", AXTitle="Save"), N("AXButton", AXTitle="Cancel"))
    main = N("AXWindow", sheet, AXTitle="Untitled (Quick Action)")
    return N("AXApplication", AXWindows=[main], AXMainWindow=main, AXFocusedWindow=main, AXMenuBar=menubar(),
             AXFocusedUIElement=sheet.kids[1]), sheet.kids[1]


class _SaveEnv(FakeEnv):
    """⇧⌘G either moves focus to a Go to Folder box (opens=True) or leaves it in the name field."""

    def __init__(self, app_el, opens):
        super().__init__(app_el)
        self.opens, self.go_box, self.typed_paths = opens, N("AXTextField", AXValue=""), []

    def hotkey(self, *keys):
        if keys == ("command", "shift", "g") and self.opens:
            self.app_el.a["AXFocusedUIElement"] = self.go_box

    def type_text(self, pid, text):
        self.typed_paths.append(text)


def test_fill_dialog_finds_the_name_box_by_its_current_text():
    app, box = _save_sheet_app()
    out = call(build_structured_tools(FakeTK(), FakeEnv(app)), "fill_dialog", app_name="", fields={"Untitled": "Hello"})
    assert "✓" in out and box.a["AXValue"] == "Hello", out


def test_fill_dialog_moves_the_folder_of_a_full_path_name_to_go_to_folder():
    app, box = _save_sheet_app()
    env = _SaveEnv(app, opens=True)
    out = call(build_structured_tools(FakeTK(), env), "fill_dialog",
               fields={"Save As": "~/Library/Services/Hello.workflow"})
    assert env.typed_paths == ["~/Library/Services"] and box.a["AXValue"] == "Hello.workflow", out
    assert "Took the folder" in out


def test_fill_dialog_types_nothing_when_go_to_folder_does_not_open():
    app, box = _save_sheet_app()
    env = _SaveEnv(app, opens=False)
    out = call(build_structured_tools(FakeTK(), env), "fill_dialog", fields={"Save As": "Hello"},
               path="~/Library/Services/", press="Save")
    assert "did not open" in out and env.typed_paths == [] and box.a["AXValue"] == "Untitled"
