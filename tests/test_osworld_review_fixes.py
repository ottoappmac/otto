"""Fixes from the review of the OSWorld run ``20261122125242-osworld-full-native-100step`` (54 of 155 passed).

The review is ``docs/reports/osworld-native-100step-failure-review.md``. Each test pins one tool or harness
behaviour that run showed going wrong, in a form that holds for any app or task.
"""
from __future__ import annotations

import sys

import pytest

mt = pytest.importorskip("tools.navigation.computer.macos_tools") if sys.platform == "darwin" else None
needs_mac = pytest.mark.skipif(mt is None, reason="macOS-only (pyobjc)")

# Calc's formula toolbar as the scan lists it: each field's group comes right before the field.
CALC_LABELS = {
    1: ("Group", "Name Box"), 2: ("TextArea", "Name Box"),
    3: ("Button", "Function Wizard"), 5: ("Group", "Input line"), 6: ("TextArea", "Formula bar"),
    10: ("TextField", "A1"),
}


# ── T1: the Name Box group is not the Name Box ──────────────────────────────

@needs_mac
def test_name_box_group_redirects_to_its_field():
    assert mt.field_group_redirect(1, CALC_LABELS) == 2
    assert mt.field_group_redirect(5, CALC_LABELS) == 6


@needs_mac
def test_only_named_field_groups_redirect():
    assert mt.field_group_redirect(2, CALC_LABELS) is None          # already the field
    assert mt.field_group_redirect(3, CALC_LABELS) is None
    assert mt.field_group_redirect(7, {7: ("Group", "Sidebar")}) is None
    assert mt.field_group_redirect(1, {1: ("Group", "Name Box")}) is None     # no field listed with it


@needs_mac
def test_buttons_and_bare_groups_refuse_typing():
    assert "does not take text" in mt.untypeable_refusal(3, CALC_LABELS)
    assert "not a text field" in mt.untypeable_refusal(7, {7: ("Group", "Sidebar")})
    assert mt.untypeable_refusal(2, CALC_LABELS) == ""
    assert mt.untypeable_refusal(10, CALC_LABELS) == ""


@needs_mac
@pytest.mark.parametrize("typed,shown", [("D2:D29", "D2:D29"), ("c2", "C2"), ("Sheet1.$A$1", "A1"),
                                         ("B3:B3", "B3")])
def test_name_box_readback_confirms_the_selection(typed, shown):
    assert mt.name_box_readback_note(typed, shown).startswith("Selected")


@needs_mac
def test_name_box_readback_warns_when_the_selection_did_not_move():
    note = mt.name_box_readback_note("D2:D29", "A1")
    assert note.startswith("Warning") and "'A1'" in note


@needs_mac
def test_name_box_readback_is_quiet_for_names_and_unreadable_values():
    assert mt.name_box_readback_note("Totals", "Totals") == ""
    assert mt.name_box_readback_note("A1", None) == ""
    assert mt.name_box_readback_note("A1:B2", "MyRange") == "The Name Box now shows 'MyRange'."


# ── Typing: hotkeys, appended text, arguments ───────────────────────────────

KEYS = frozenset({"command", "shift", "option", "ctrl", "return", "tab", "esc", "a", "z", "=", "+", "f5", "left"})


@needs_mac
@pytest.mark.parametrize("keys,want", [
    ("cmd+shift+z", ["command", "shift", "z"]), ("Command+A", ["command", "a"]), ("enter", ["return"]),
    ("command++", ["command", "+"]), ("command+=", ["command", "="]), ("ctrl tab", ["ctrl", "tab"]),
    ("escape", ["esc"]), ("alt+left", ["option", "left"]),
])
def test_hotkey_names_are_normalised(keys, want):
    assert mt.parse_hotkey(keys, KEYS) == (want, "")


@needs_mac
@pytest.mark.parametrize("keys", ["{return}", "<parameter=command>\nz", "", "command+" + "x" * 80, "cmd+frob"])
def test_junk_is_not_sent_as_a_hotkey(keys):
    parts, why = mt.parse_hotkey(keys, KEYS)
    assert parts == [] and why


@needs_mac
def test_old_text_left_after_the_typed_text_is_flagged():
    note = mt.typed_value_note("0;0", "@General", "0;0@General", append=False, cleared=True)
    assert note.startswith("Warning") and "'@General'" in note
    assert not mt.typed_value_note("0;0", "@General", "0;0", append=False, cleared=True).startswith("Warning")


@needs_mac
def test_args_take_the_types_the_tool_declares():
    out = mt.coerce_args(mt._TypeIntoControlInput, {"index": "7", "text": 2015, "submit": True})
    assert out == {"index": 7, "text": "2015", "submit": True}
    assert mt.coerce_args(mt._ClickAtInput, {"x": "12", "y": "x9"}) == {"x": 12, "y": "x9"}   # left for pydantic
    assert mt.coerce_args(object, {"a": 1}) == {"a": 1}


@needs_mac
def test_batch_keeps_quoted_numbers_as_text_and_knows_the_pointer_tools():
    steps = mt._parse_batch_actions("type_into_control(index=3, text='2015'); click_at(x=40, y=50); "
                                    "set_text_at(x=1, y=2, text='0.5'); drag(from_index=4, to_index=9)")
    assert [n for n, _ in steps] == ["type_into_control", "click_at", "set_text_at", "drag"]
    assert steps[0][1] == {"index": 3, "text": "2015"}
    assert steps[1][1] == {"x": 40, "y": 50}
    assert steps[2][1]["text"] == "0.5" and steps[3][1] == {"from_index": 4, "to_index": 9}


# ── Clicks: off-window, focus loss, OCR points ──────────────────────────────

@needs_mac
def test_points_and_window_rects():
    rects = [(0, 0, 100, 50), (200, 200, 10, 10)]
    assert mt.point_in_rects(5, 5, rects) and mt.point_in_rects(205, 209, rects)
    assert not mt.point_in_rects(100, 10, rects) and not mt.point_in_rects(12, 665, [])
    wl = [{"kCGWindowOwnerPID": 5, "kCGWindowBounds": {"X": 1, "Y": 2, "Width": 30, "Height": 40}},
          {"kCGWindowOwnerPID": 6, "kCGWindowBounds": {"X": 0, "Y": 0, "Width": 9, "Height": 9}},
          {"kCGWindowOwnerPID": 5, "kCGWindowBounds": {"X": 0, "Y": 0, "Width": 0, "Height": 9}}]
    assert mt.window_rects_of(wl, 5) == [(1, 2, 30, 40)]


@needs_mac
def test_a_click_that_moves_focus_to_another_app_is_reported():
    assert mt.front_change_note("LibreOffice", "LibreOffice") == ""
    assert mt.front_change_note("", "Finder") == ""
    note = mt.front_change_note("LibreOffice", "Finder")
    assert "'Finder'" in note and "activate_app('LibreOffice')" in note


@needs_mac
def test_ocr_click_lands_on_the_word_not_the_line_centre():
    line = {"text": "H2O and CO2", "x": 150, "y": 10, "left": 100, "width": 110}    # 10 points a character
    assert mt.substring_point(line, "CO2") == (195, 10)
    assert mt.substring_point(line, "H2O") == (115, 10)
    assert mt.substring_point(line, "missing") == (150, 10)
    assert mt.substring_point({"text": "x", "x": 3, "y": 4}, "x") == (3, 4)               # no line box


@needs_mac
def test_update_balloon_is_not_taken_for_the_front_window():
    def win(n, w, h, name=""):
        return {"kCGWindowOwnerPID": 5, "kCGWindowLayer": 0, "kCGWindowNumber": n, "kCGWindowName": name,
                "kCGWindowBounds": {"X": 0, "Y": 0, "Width": w, "Height": h}}
    picked = mt.pick_app_windows([win(1, 420, 140), win(2, 1200, 800, "data.xlsx - LibreOffice Calc")], 5)
    assert [p[0] for p in picked] == [2]
    assert [p[0] for p in mt.pick_app_windows([win(1, 420, 140)], 5)] == [1]              # the only window: kept
    assert [p[0] for p in mt.pick_app_windows([win(1, 420, 140, "Find"), win(2, 900, 700)], 5)] == [1, 2]


# ── Structured tools: Name Box, cell blocks, text selection, pop-ups ────────

from tests.test_macos_structured_tools import FakeEnv, FakeTK, N, call  # noqa: E402
from tools.navigation.computer import macos_ax_struct as axs  # noqa: E402
from tools.navigation.computer.macos_structured_tools import build_structured_tools  # noqa: E402

AX = axs  # the helpers take any object with get/children; FakeEnv's FakeAX is one


def _ax():
    return FakeEnv(N("AXApplication")).access


def test_the_name_box_field_is_found_inside_its_group():
    field = N("AXTextArea", AXTitle="A1", AXValue="A1")
    win = N("AXWindow", N("AXToolbar", N("AXGroup", field, AXDescription="Name Box")),
            N("AXTextArea", AXDescription="Formula bar"))
    assert axs.find_named_field(_ax(), win, "Name Box") is field
    bare = N("AXTextField", AXDescription="Name Box")
    assert axs.find_named_field(_ax(), N("AXWindow", bare), "name box") is bare
    assert axs.find_named_field(_ax(), N("AXWindow", N("AXGroup", AXDescription="Name Box")), "Name Box") is None


def test_text_is_found_by_occurrence():
    a, b = N("AXStaticText", AXValue="the cat"), N("AXTextArea", AXValue="a cat and a cat")
    win = N("AXWindow", a, N("AXGroup", b))
    assert axs.find_text(_ax(), win, "cat") == (a, 4, "the cat")
    assert axs.find_text(_ax(), win, "cat", 3) == (b, 12, "a cat and a cat")
    assert axs.find_text(_ax(), win, "cat", 4) is None and axs.find_text(_ax(), win, "") is None


def test_block_cells_and_mismatches():
    wanted = axs.block_cells("B2", [["Q1", 1000, "=B2*2"], ["Q2", None, "5%"]])
    assert wanted == {(2, 2): "Q1", (3, 2): "1000", (4, 2): "=B2*2", (2, 3): "Q2", (3, 3): "", (4, 3): "5%"}
    assert axs.block_cells("B2:C3", [[1]]) is None
    shown = {(2, 2): "Q1st", (3, 2): "1,000", (4, 2): "2000", (2, 3): "Q2", (3, 3): "", (4, 3): "0.05"}
    assert axs.cell_mismatches(wanted, shown) == ["B2 shows 'Q1st' (typed 'Q1')"]
    assert axs.cell_mismatches(wanted, {}) == []                         # nothing on screen: nothing judged


def _popup(options, start):
    pos = [options.index(start)]
    return (lambda: options[pos[0]],
            lambda key: pos.__setitem__(0, max(0, pos[0] - 1) if key == "up" else min(len(options) - 1, pos[0] + 1)),
            pos)


def test_a_popup_without_a_readable_menu_is_stepped_to_the_option():
    opts = ["Automatic", "Bar", "Line", "Line and Points"]
    read, step, pos = _popup(opts, "Bar")
    assert axs.cycle_to_option(read, step, "line", max_steps=10) == ("Line", opts)
    assert opts[pos[0]] == "Line"
    read, step, pos = _popup(opts, "Bar")
    assert axs.cycle_to_option(read, step, "Pie", max_steps=10) == (None, opts)
    assert opts[pos[0]] == "Bar"                # no match: put back on the option it had


class SheetEnv(FakeEnv):
    """Calc with a Name Box and a 5x6 grid: typing goes to the Name Box while it has focus, else to the cell."""

    def __init__(self, autoinput=None):
        self.box = N("AXTextArea", AXTitle="A1", AXValue="A1")
        self.cells = {(c, r): N("AXCell", AXDescription=f"{axs.col_letters(c)}{r}", AXValue="")
                      for r in range(1, 7) for c in range(1, 6)}
        win = N("AXWindow", N("AXGroup", self.box, AXDescription="Name Box"), N("AXTable", *self.cells.values()),
                AXTitle="data.ods - LibreOffice Calc")
        super().__init__(N("AXApplication", AXWindows=[win], AXMainWindow=win, AXFocusedWindow=win))
        self.cursor, self.buf, self.autoinput = (1, 1), "", autoinput or {}

    def hotkey(self, *keys):
        if keys == ("command", "a"):
            self.buf = ""

    def type_text(self, pid, text):
        self.buf += text

    def press_key(self, key):
        if self.box.a.get("AXFocused") and key == "return":
            self.cursor = axs.cell_address(self.buf)
            self.box.a.update(AXFocused=False, AXTitle=self.buf, AXValue=self.buf)
        elif key == "tab":
            if self.buf:
                self.cells[self.cursor].a["AXValue"] = self.autoinput.get(self.buf, self.buf)
            self.cursor = (self.cursor[0] + 1, self.cursor[1])
        self.buf = ""


def test_write_cells_types_the_block_and_checks_it():
    env = SheetEnv()
    out = call(build_structured_tools(FakeTK(), env), "write_cells", app_name="LibreOffice", anchor="B2",
               rows=[["Name", "Score"], ["Ann", 90], ["Bob", None]])
    assert "Typed 5 cell(s) into B2:C4" in out and "Every typed cell" in out
    assert env.cells[(2, 3)].a["AXValue"] == "Ann" and env.cells[(3, 3)].a["AXValue"] == "90"
    assert env.cells[(3, 4)].a["AXValue"] == ""                                       # None skips the cell


def test_write_cells_lists_cells_autoinput_changed():
    env = SheetEnv(autoinput={"Bo": "Bob"})
    out = call(build_structured_tools(FakeTK(), env), "write_cells", app_name="LibreOffice", anchor="A1",
               rows=[["Bob"], ["Bo"]])
    assert "A2 shows 'Bob' (typed 'Bo')" in out


@pytest.mark.parametrize("anchor,rows,want", [("B2:C3", [[1]], "not one cell"), ("A1", [[]], "No values")])
def test_write_cells_refuses_bad_input(anchor, rows, want):
    assert want in call(build_structured_tools(FakeTK(), SheetEnv()), "write_cells", app_name="LibreOffice",
                        anchor=anchor, rows=rows)


class DocEnv(FakeEnv):
    def __init__(self, text, shift=0):
        self.para = N("AXTextArea", AXValue=text)
        win = N("AXWindow", self.para, AXTitle="essay.docx - LibreOffice Writer")
        super().__init__(N("AXApplication", AXWindows=[win], AXMainWindow=win, AXFocusedWindow=win))
        self.shift = shift          # an app that selects a few characters off

    def set_selection(self, el, start, length):
        v = el.a["AXValue"]
        el.a["AXSelectedText"] = v[start + self.shift:start + self.shift + length]
        return True


def test_select_text_selects_and_reads_the_selection_back():
    tools = build_structured_tools(FakeTK(), DocEnv("Intro. Grammar test 2 is here."))
    assert call(tools, "select_text", app_name="LibreOffice", text="Grammar test 2") == \
        "Selected 3 word(s), 14 character(s): 'Grammar test 2'."
    out = call(tools, "select_text", app_name="LibreOffice", text="test", paragraph=True)
    assert "30 character(s)" in out


def test_select_text_warns_on_a_wrong_selection_and_a_missing_text():
    out = call(build_structured_tools(FakeTK(), DocEnv("Intro. Grammar test 2.", shift=2)), "select_text",
               app_name="LibreOffice", text="Grammar")
    assert out.startswith("Warning: the selection reads 'ammar t'")
    out = call(build_structured_tools(FakeTK(), DocEnv("Intro.")), "select_text", app_name="LibreOffice",
               text="Grammar")
    assert "is not in the text" in out and "Find & Replace" in out


# ── run_osascript: shell commands without hand quoting, refused bookmark scripts ──

def _osa(monkeypatch):
    import asyncio

    from backend.builtin_mcps.macos_osascript import server
    seen: list[list[str]] = []

    async def fake_run(args, timeout):
        seen.append(list(args))
        return {"ok": True, "exit_code": 0, "stdout": "ok\n", "stdout_truncated": False, "stderr": "",
                "stderr_truncated": False, "timed_out": False, "duration_ms": 1}

    monkeypatch.setattr(server, "_run_osascript_args", fake_run)
    monkeypatch.setenv("OTTO_DESKTOP_LEASE_BY_HOST", "1")
    return (lambda **kw: asyncio.run(server.run_osascript(**kw))), seen


def test_shell_command_is_wrapped_with_its_quoting_done(monkeypatch):
    run, seen = _osa(monkeypatch)
    out = run(shell_command='python3 -c "print(\'a\\\\b\')" | wc -c')
    assert out["ok"] and seen
    assert seen[-1][-1] == 'do shell script "python3 -c \\"print(\'a\\\\\\\\b\')\\" | wc -c"'
    assert not run(script="return 1", shell_command="ls")["ok"]                      # one or the other
    assert len(seen) == 1


@needs_mac
def test_shell_command_script_round_trips_through_osascript():
    import subprocess

    from backend.builtin_mcps.macos_osascript._helpers import shell_command_script
    cmd = """printf '%s|' "a \\"b\\"" '$HOME' "back\\\\slash" """
    script = shell_command_script(cmd)
    out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=20)
    assert out.stdout.strip() == 'a "b"|$HOME|back\\slash|'


def test_chrome_bookmark_scripts_are_refused_before_they_run(monkeypatch):
    from backend.builtin_mcps.macos_osascript._helpers import refused_script_reason
    bad = 'tell application "Google Chrome"\n  make new bookmark folder at bookmark folder "Bookmarks Bar"\nend tell'
    assert "Bookmark Manager" in refused_script_reason(bad)
    assert refused_script_reason('tell application "Google Chrome" to get URL of active tab of window 1') == ""
    assert refused_script_reason('tell application "Safari" to make new bookmark item') == ""
    run, seen = _osa(monkeypatch)
    out = run(script=bad)
    assert out["ok"] is False and out.get("refused") and not seen


# ── MacArena harness: repeats, nags, finish nudges, pooling ─────────────────

def _harness():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    ma = root / "scripts" / "macarena"
    added = [p for p in (str(ma), str(ma / "guest")) if p not in sys.path]
    sys.path[:0] = added
    try:
        import importlib.util

        import bridge
        import report
        spec = importlib.util.spec_from_file_location("macarena_run_t", ma / "run.py")
        run = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run)
    finally:
        sys.path[:] = [p for p in sys.path if p not in added]
    return bridge, run, report


BR, RUN, REPORT = _harness()


def _native_bridge(tmp_path, **kw):
    g = BR.FakeGuest()
    b = BR.Bridge(BR.FakeEnv(), max_steps=0, settle=0.0, pause=0.0, guest=g, **kw)
    b.reset({"id": "t", "instruction": "do it", "evaluator": []}, tmp_path / "run_1")
    return b, g


def _act(b, tool, **args):
    return b.native({"tool": tool, "tool_id": f"tools:{tool}", "args": args, "config": {}})


def test_repeat_notice_every_third_identical_action(tmp_path):
    b, _ = _native_bridge(tmp_path)
    got = ["notice" in _act(b, "click", index=4) for _ in range(6)]
    assert got == [False, False, True, False, False, True] and b.state()["repeat_notices"] == 2
    assert "same click call 3 times" in [_act(b, "click", index=5) for _ in range(3)][-1]["notice"]
    assert not any("notice" in _act(b, "hotkey", keys="down") for _ in range(6))      # moving through a list
    assert not any("notice" in _act(b, "scroll", direction="down") for _ in range(6))


def test_different_actions_reset_the_repeat_count_and_zero_turns_it_off(tmp_path):
    b, _ = _native_bridge(tmp_path)
    for i in range(6):
        assert "notice" not in _act(b, "click", index=i % 2)
    b0, _ = _native_bridge(tmp_path / "b0", repeat_notice_after=0)
    assert not any("notice" in _act(b0, "click", index=1) for _ in range(6))


def test_reset_closes_update_nags_on_the_native_track_only(tmp_path):
    b, g = _native_bridge(tmp_path)
    assert len(g.setup_calls) == 1 and g.calls == [] and b.state()["dismiss_nags"] is True
    assert "Can't update Chrome" in g.setup_calls[0]["args"]["script"]
    b2, g2 = _native_bridge(tmp_path / "keep", dismiss_nags=False)
    assert g2.setup_calls == [] and b2.state()["dismiss_nags"] is False
    std = BR.Bridge(BR.FakeEnv(), max_steps=3, settle=0.0, pause=0.0)
    assert std.dismiss_nags is False


def test_script_refusal_checks_shell_commands_too():
    port = BR.script_refusal("run_osascript", {"shell_command": "curl -s http://localhost:1337/json/list"}, None)
    assert port and "harness" in port
    assert BR.script_refusal("run_osascript", {"shell_command": "ls ~/Desktop"}, None) is None


@pytest.mark.parametrize("state,nudged,left,want", [
    ({"steps": 10, "max_steps": 100}, 0, 600, True),
    ({"steps": 10, "max_steps": 100}, 1, 600, False),             # nudges used up
    ({"steps": 10, "max_steps": 100, "finished": "done"}, 0, 600, False),
    ({"steps": 100, "max_steps": 100}, 0, 600, False),            # no steps left
    ({"steps": 10, "max_steps": 0}, 0, 600, True),                # no cap
    ({"steps": 10, "max_steps": 100}, 0, 30, False),              # no time for another turn
])
def test_finish_nudge_only_when_the_run_is_open(state, nudged, left, want):
    assert RUN.should_nudge(state, nudged, 1, left) is want


def test_report_refuses_to_pool_different_harness_settings():
    row = {"id": "a", "run": 1, "score": 1.0, "status": "ok", "source": "osworld", "domain": "d", "steps": 3,
           "track": "native", "observe_after": True, "fail_notice_after": 4, "repeat_notice_after": 3,
           "finish_nudge": 1, "dismiss_nags": True}
    s = REPORT.summarize([row, {**row, "id": "b"}])
    assert s["harness"]["dismiss_nags"] is True and s["harness"]["finish_nudge"] == 1
    assert "update nags closed" in REPORT.render(s)
    for k, v in (("finish_nudge", 0), ("repeat_notice_after", 0), ("dismiss_nags", False)):
        with pytest.raises(ValueError):
            REPORT.summarize([row, {**row, "id": "b", k: v}])


# ── Old controls lists leave the history ────────────────────────────────────

def test_old_controls_lists_are_cut_and_recent_results_kept():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from middleware import old_controls_pruning as ocp

    def result(i, extra=""):
        return ToolMessage(content=f"Pressed {i}." + extra, tool_call_id=f"c{i}", id=f"t{i}")

    controls = "\n\nControls now (LibreOffice), read after the action:\n" + "[1] B'OK'\n" * 400
    msgs = [HumanMessage(content="task")]
    for i in range(9):
        msgs += [AIMessage(content="", id=f"a{i}"), result(i, controls)]
    msgs.append(ToolMessage(content=[{"type": "text", "text": "Dialog x" + controls}], tool_call_id="cz", id="tz"))
    mw = ocp.OldControlsPruningMiddleware(keep_recent=6)
    out = mw.prune(msgs)
    tools = [m for m in out if isinstance(m, ToolMessage)]
    assert [m.id for m in out] == [m.id for m in msgs] and len(out) == len(msgs)
    assert all(m.content == f"Pressed {i}." + ocp.NOTE_CONTROLS for i, m in enumerate(tools[:4]))
    assert all("Controls now" in (m.content if isinstance(m.content, str) else m.content[0]["text"])
               for m in tools[4:])
    assert mw.prune(msgs[:8]) is None                                # few results: nothing to do
    assert mw.prune(out) is None                                     # idempotent: already short


def test_long_old_results_without_controls_are_cut_with_a_note():
    from middleware import old_controls_pruning as ocp
    assert ocp.shorten("x" * 5000, 1500) == "x" * 1500 + ocp.NOTE_CUT
    assert ocp.shorten("short") == "short"
    assert ocp.maybe_for_agent("macarena-desktop-agent") is not None and ocp.maybe_for_agent("research") is None


def test_cards_carry_the_lessons_from_the_second_rerun():
    from src.tools.navigation.computer.macos_app_cards import card_for
    assert "After current sheet" in card_for("LibreOffice Calc")
    gimp = card_for("GIMP")
    assert "copy under the new name" in gimp and "do not give up with vm_fail" in gimp
    assert "shift-click" in card_for("Google Chrome") and "clearBrowserData" in card_for("Google Chrome")


def test_finish_nudge_says_vm_fail_is_a_tool_call():
    import importlib.util, sys
    from pathlib import Path
    ma = Path(__file__).resolve().parents[1] / "scripts" / "macarena"
    sys.path.insert(0, str(ma))
    try:
        spec = importlib.util.spec_from_file_location("macarena_run_nudge", ma / "run.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(ma))
    assert "vm_fail tool" in mod.FINISH_NUDGE and "does nothing" in mod.FINISH_NUDGE
