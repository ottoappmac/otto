"""Fixes from the review of the MacArena run ``20261122094747-sample-3-per-group-seed1-60step`` (12 of 24 passed).

Each test pins one tool or harness behaviour that run showed going wrong, in a form that holds for any app:
typing that appended to an address bar or went into the Name Box, OCR that read the app's largest window instead
of the dialog in front, a stale index that minimised a window, an unchecked save, a sheet read with blank leading
columns, and the benchmark agent finishing with an unchecked or stand-in result.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend import macarena_guard_middleware as guard
from tools.navigation.computer import macos_ax_struct as axs

ROOT = Path(__file__).resolve().parents[1]


# ── Typing: Name Box guard, clearing single-line fields, read-back ─────────

mt = pytest.importorskip("tools.navigation.computer.macos_tools") if sys.platform == "darwin" else None
needs_mac = pytest.mark.skipif(mt is None, reason="macOS-only (pyobjc)")


@needs_mac
@pytest.mark.parametrize("text", ["A1", "A2:B30", "$A$1", "Sheet2.A1", "'My sheet'.B3", "A:C", "2:5", "Totals"])
def test_name_box_takes_references_and_range_names(text):
    assert mt.name_box_refusal(2, text, {2: ("TextArea", "Name Box")}) == ""


@needs_mac
@pytest.mark.parametrize("text", ["Total Revenue", "=SUM(Sheet1.A2:A20)", "42", "Above average"])
def test_name_box_refuses_values_labels_and_formulas(text):
    msg = mt.name_box_refusal(2, text, {2: ("TextArea", "Name Box")})
    assert msg.startswith("Not typed") and "type_text" in msg


@needs_mac
def test_name_box_guard_leaves_other_fields_alone():
    assert mt.name_box_refusal(3, "Total Revenue", {3: ("TextField", "Address and search bar")}) == ""


@needs_mac
def test_only_single_line_fields_are_cleared_before_typing():
    assert mt.replaces_contents("AXTextField", "Address and search bar")
    assert mt.replaces_contents("AXComboBox", "Font Name")
    assert not mt.replaces_contents("AXTextArea", "Document")          # a document body: insert at the cursor
    assert not mt.replaces_contents("AXTextField", "B12")              # a spreadsheet cell: typing replaces it
    assert not mt.replaces_contents("AXTextField", "Name Box")


@needs_mac
def test_typed_value_note_flags_text_left_in_front():
    warn = mt.typed_value_note("arxiv.org", "arxiv.org", "arxiv.orgarxiv.org", append=False, cleared=False)
    assert warn.startswith("Warning") and "could not be cleared" in warn
    warn = mt.typed_value_note("cats", "dogs ", "dogs cats", append=False, cleared=True)
    assert warn.startswith("Warning")
    # Inline autocomplete after the typed text is fine; so is appending on request.
    assert mt.typed_value_note("arxiv.org", "", "arxiv.org/list", append=False, cleared=True).startswith(
        "The field now reads")
    assert mt.typed_value_note("b", "a", "ab", append=True, cleared=True).startswith("The field now reads")
    assert mt.typed_value_note("q", "", "", append=False, cleared=True).startswith("Warning: the field is empty")


@needs_mac
def test_type_into_control_refuses_text_for_the_name_box_without_typing(monkeypatch):
    tk = mt.MacOSToolkit()
    tk._element_registry = {2: object()}
    tk._label_registry = {2: ("TextArea", "Name Box")}
    typed = []
    monkeypatch.setattr(mt, "_type_to_pid", lambda *a: typed.append(a) or True)
    tool = next(t for t in tk.tools if t.name == "type_into_control")
    out = asyncio.run(tool.ainvoke({"index": 2, "text": "Total Revenue", "submit": True}))
    assert out.startswith("Not typed") and typed == []


@needs_mac
def test_type_into_control_clears_and_reports_the_field(monkeypatch):
    tk = mt.MacOSToolkit()
    elem = object()
    tk._element_registry = {7: elem}
    tk._label_registry = {7: ("TextField", "Address and search bar")}
    field = {"value": "old text"}
    calls = []

    def fake_get(el, attr):
        return {"AXRole": "AXTextField", "AXValue": field["value"]}.get(attr)

    def fake_set(el, attr, value):
        if attr == "AXValue":
            field["value"] = value
        return 0

    def fake_type(pid, text, submit=False):
        calls.append((text, submit))
        field["value"] += text
        return True

    monkeypatch.setattr(mt, "_ax_get", fake_get)
    monkeypatch.setattr(mt, "AXUIElementSetAttributeValue", fake_set)
    monkeypatch.setattr(mt, "_ax_pid", lambda el: 123)
    monkeypatch.setattr(mt, "_type_to_pid", fake_type)
    tk.action_delay = 0
    tool = next(t for t in tk.tools if t.name == "type_into_control")
    out = asyncio.run(tool.ainvoke({"index": 7, "text": "arxiv.org", "submit": True}))
    assert "Before Return the field read 'arxiv.org'" in out
    assert calls == [("arxiv.org", False), ("", True)]                 # Return only after the read-back

    field["value"] = "abc"
    out = asyncio.run(tool.ainvoke({"index": 7, "text": "def", "append": True}))
    assert "The field now reads 'abcdef'" in out


# ── OCR reads the front window, and the windows behind it ───────────────────

def _win(num, pid, x, y, w, h, name="", layer=0):
    return {"kCGWindowNumber": num, "kCGWindowOwnerPID": pid, "kCGWindowLayer": layer, "kCGWindowName": name,
            "kCGWindowBounds": {"X": x, "Y": y, "Width": w, "Height": h}}


@needs_mac
def test_pick_app_windows_keeps_front_to_back_order_and_skips_tiny_and_other_layers():
    info = [
        _win(9, 5, 0, 0, 40, 20, "tooltip"),
        _win(8, 7, 0, 0, 900, 700, "other app"),
        _win(7, 5, 300, 200, 500, 400, "Preferences"),            # the dialog in front, smaller than the main window
        _win(6, 5, 0, 0, 1400, 900, "Image", layer=3),
        _win(5, 5, 0, 0, 1400, 900, "GNU Image Manipulation Program"),
    ]
    picked = mt.pick_app_windows(info, 5)
    assert [w[0] for w in picked] == [7, 5]
    assert picked[0][2] == "Preferences"


@needs_mac
def test_read_screen_reads_each_app_window_in_its_own_section(monkeypatch):
    tk = mt.MacOSToolkit()
    monkeypatch.setattr(tk, "_window_target", lambda app: (7, (300, 200, 500, 400)))
    monkeypatch.setattr(tk, "_app_windows", lambda app: [(7, (300, 200, 500, 400), "Preferences"),
                                                          (5, (0, 0, 1400, 900), "Main")])
    texts = {7: "Undo levels", 5: "Paintbrush"}
    monkeypatch.setattr(tk, "_capture_for_read", lambda wid, region, want_image: (
        [{"text": texts[wid], "conf": 1.0, "x": 10, "y": 10}], None))
    tool = next(t for t in tk.tools if t.name == "read_screen")
    out = asyncio.run(tool.ainvoke({"app_name": "GIMP"}))
    front, behind = out.split("\n\n")
    assert "'Preferences' (front)" in front and "Undo levels" in front
    assert "behind the front window" in behind and "Paintbrush" in behind


# ── Stale indices on window buttons ─────────────────────────────────────────

@needs_mac
def test_window_button_whose_index_changed_is_refused_once(monkeypatch):
    tk = mt.MacOSToolkit()
    elem = object()
    tk._set_labels({35: ("Button", "Close")})                          # a dialog's Close, in the older list
    tk._set_labels({35: ("Button", "MinimizeButton")})                 # the latest list: now the window's button
    tk._element_registry = {35: elem}
    monkeypatch.setattr(mt, "_ax_get", lambda el, attr: "AXMinimizeButton" if attr == "AXSubrole" else None)
    what, refused = tk._index_guard(35)
    assert "MinimizeButton" in what and refused.startswith("Not done") and "'Close'" in refused
    assert tk._index_guard(35)[1] == ""                                # asked again: the agent means it
    tk._set_labels({35: ("Button", "OK")})
    monkeypatch.setattr(mt, "_ax_get", lambda el, attr: None)
    assert tk._index_guard(35) == (" (Button 'OK')", "")               # ordinary controls are never refused


# ── Save shortcut result ────────────────────────────────────────────────────

@needs_mac
def test_save_result_note():
    assert mt.save_result_note("/x/a.docx", 1.0, 2.0, "").startswith("Saved: a.docx")
    assert mt.save_result_note("/x/a.docx", 1.0, 1.0, "").startswith("Not saved yet: a.docx")
    assert "is not saved until it is answered" in mt.save_result_note("/x/a.xlsx", 1.0, 1.0, "'Keep format'")
    assert "no saved file" in mt.save_result_note("", None, None, "")


# ── Tables: leading blank columns ───────────────────────────────────────────

def test_format_cells_drops_blank_leading_columns_and_rows_with_real_addresses():
    cells = {(c, r): "" for c in range(1, 7) for r in range(1, 5)}
    cells.update({(4, 2): "Students", (5, 2): "Marks", (6, 2): "Message", (4, 3): "Ann", (5, 3): "42"})
    out = axs.format_cells(cells).splitlines()
    assert out[0] == "\tD\tE\tF" and out[1].startswith("2\tStudents")
    assert "columns A–C and row 1 are empty" in out[-1] and "first column shown is D" in out[-1]
    # A range the agent chose is shown as asked.
    assert axs.format_cells(cells, axs.parse_range("B2:F3")).splitlines()[0] == "\tB\tC\tD\tE\tF"


# ── System notices ──────────────────────────────────────────────────────────

def test_permission_notice_is_split_from_windows_to_deal_with():
    act, notices = axs.split_notices([("universalAccessAuthWarn", "Accessibility Access"), ("Finder", "Copy")])
    assert act == [("Finder", "Copy")] and notices == [("universalAccessAuthWarn", "Accessibility Access")]


def test_start_state_marks_the_permission_notice_as_not_part_of_the_task():
    sys.path.insert(0, str(ROOT / "scripts" / "macarena"))
    try:
        import bridge as br
    finally:
        sys.path.pop(0)
    out = """##APPS
ASN:0x0-0xaa0aa-"Google Chrome":
##WINDOWS
[["universalAccessAuthWarn","Accessibility Access"],["Google Chrome","Page"],["Finder","Copy"]]
##TABS
[]
##END
"""
    text = br.parse_start_state(out)
    assert '- Other windows on screen: "Copy" (Finder)' in text
    assert '"Accessibility Access" (universalAccessAuthWarn), a macOS permission notice' in text


# ── Benchmark-agent checks in the guard middleware ──────────────────────────

def _osa(script):
    return {"script": script}


@pytest.mark.parametrize("script", [
    'do shell script "cat > ~/Desktop/a.webloc <<EOF\nx\nEOF"',
    'do shell script "echo x >> ~/f"',
    'do shell script "pip install numpy"',
    "do shell script \"sed -i '' s/a/b/ ~/main.py\"",
    'do shell script "python3 -c \'print(1)\'"',
    'do shell script "sips -Z 100 ~/a.jpg"',
    'do shell script "soffice --headless --convert-to csv ~/a.xlsx"',
])
def test_shell_changes_are_recognised(script):
    assert guard.shell_write("run_osascript", _osa(script))


@pytest.mark.parametrize("script", [
    'do shell script "ls -la ~/Desktop 2>/dev/null"',
    'do shell script "grep -n x ~/f >/dev/null && cat ~/f 2>&1"',
    'do shell script "mdfind -name x | head"',
    'tell application "Finder" to activate',
])
def test_shell_reads_are_not_checked(script):
    assert guard.shell_write("run_osascript", _osa(script)) is None


def _tool_msg(text):
    return SimpleNamespace(type="tool", content=text)


def test_shell_change_is_checked_once_per_task():
    script = _osa('do shell script "echo x > ~/f"')
    first = guard.native_check("run_osascript", script, [])
    assert first and first.startswith(guard.SHELL_WRITE_NOTE) and "same call again" in first
    assert guard.native_check("run_osascript", script, [_tool_msg(first)]) is None


def test_vm_done_is_checked_once_and_never_near_the_end_of_the_budget():
    first = guard.native_check("vm_done", {}, [_tool_msg("ok\n\n[benchmark] steps used 10, steps left 50")])
    assert first and first.startswith(guard.DONE_CHECK_NOTE) and "vm_fail" in first
    assert guard.native_check("vm_done", {}, [_tool_msg(first)]) is None
    low = [_tool_msg("ok\n\n[benchmark] steps used 57, steps left 3")]
    assert guard.native_check("vm_done", {}, low) is None
    assert guard.native_check("vm_fail", {}, []) is None
    assert guard.native_check("vm_done", {}, []) is None                # no budget note: the bridge decides
    assert guard.native_check("vm_done", {}, [_tool_msg("[benchmark] steps used 3, no step limit")])


def test_steps_left_reads_the_last_budget_note():
    msgs = [_tool_msg("[benchmark] steps used 1, steps left 59"), SimpleNamespace(type="ai", content="x"),
            _tool_msg("[benchmark] steps used 9, steps left 51")]
    assert guard.steps_left(msgs) == 51
    assert guard.steps_left([_tool_msg("[benchmark] steps used 9, no step limit")]) is None


def test_native_check_answers_without_reaching_the_bridge(monkeypatch):
    ticks = []
    monkeypatch.setattr(guard.macarena_native, "tick", lambda *a, **k: ticks.append(a) or {"ok": True})
    mw = guard.MacarenaGuardMiddleware(agent_name=guard.NATIVE_AGENT)
    msgs = [_tool_msg("ok\n\n[benchmark] steps used 10, steps left 50")]
    req = SimpleNamespace(tool_call={"name": "vm_done", "args": {}, "id": "c1"}, state={"messages": msgs}, tool=None)
    out = mw.wrap_tool_call(req, lambda r: pytest.fail("the call must not run"))
    assert out.content.startswith(guard.DONE_CHECK_NOTE) and ticks == []


# ── Prompt and cards ────────────────────────────────────────────────────────

def test_infeasible_note_ties_vm_fail_to_concrete_signs():
    sys.path.insert(0, str(ROOT / "scripts" / "macarena"))
    try:
        import importlib
        run = importlib.import_module("run")
    finally:
        sys.path.pop(0)
        sys.modules.pop("run", None)
    assert "contradicts itself" in run.INFEASIBLE_NOTE and "rather than make a stand-in" in run.INFEASIBLE_NOTE


def test_new_card_facts_are_reachable():
    from tools.navigation.computer.macos_app_cards import card_for
    assert "Config Editor" in card_for("Thunderbird")
    assert "Pivot Table" in card_for("LibreOffice Calc") and "Cross-reference" in card_for("LibreOffice")
    assert "chip" in card_for("Google Chrome")


# ── Live status of a run in progress ────────────────────────────────────────

def _report():
    sys.path.insert(0, str(ROOT / "scripts" / "macarena"))
    try:
        import report
    finally:
        sys.path.pop(0)
    return report


def test_live_status_counts_done_passed_partial_failed_and_left_by_group(tmp_path):
    report = _report()
    planned = [{"id": f"t{i}", "source": "osworld", "domain": "calc" if i < 3 else "chrome"} for i in range(5)]
    rows = [
        {"id": "t0", "run": 1, "source": "osworld", "domain": "calc", "score": 1.0, "status": "ok", "steps": 10},
        {"id": "t1", "run": 1, "source": "osworld", "domain": "calc", "score": 0.4, "status": "ok", "steps": 30,
         "agent_status": "run_over"},
        {"id": "t3", "run": 1, "source": "osworld", "domain": "chrome", "score": 0.0, "status": "ok", "steps": 5},
        {"id": "t4", "run": 1, "source": "osworld", "domain": "chrome", "score": 0.0, "status": "infra_error"},
        {"id": "other", "run": 1, "source": "osworld", "domain": "calc", "score": 1.0, "status": "ok"},
    ]
    s = report.live_status(planned, 1, rows, {"id": "t2", "source": "osworld", "domain": "calc", "run": 1},
                           started_at=0, now=3600)
    o = s["overall"]
    assert (o["planned_runs"], o["finished_runs"], o["remaining_runs"]) == (5, 4, 1)
    assert (o["passed"], o["partial"], o["failed"], o["infra_error_runs"]) == (1, 1, 1, 1)
    assert o["score_sum"] == 1.4 and o["mean_score"] == round(1.4 / 3, 4)      # infra errors are not scored
    assert s["by_domain"]["calc"]["finished_runs"] == 2 and s["by_domain"]["calc"]["remaining_runs"] == 1
    assert s["by_domain"]["chrome"]["infra_error_runs"] == 1
    assert s["by_domain"]["calc"]["step_limit_runs"] == 1
    assert s["elapsed_s"] == 3600


def test_live_status_files_are_written_whole_and_never_raise(tmp_path, capsys):
    report = _report()
    planned = [{"id": "t0", "source": "osworld", "domain": "os"}]
    report.write_live(tmp_path, planned, 2, [], {"id": "t0", "source": "osworld", "domain": "os", "run": 1,
                                                  "instruction": "do it"}, started_at=0)
    data = json.loads((tmp_path / "live_status.json").read_text())
    assert data["overall"]["planned_runs"] == 2 and data["by_domain"]["os"]["remaining_runs"] == 2
    assert "running now: t0" in (tmp_path / "live_status.txt").read_text()
    assert not list(tmp_path.glob("*.tmp"))
    report.write_live(tmp_path / "missing" / "dir", planned, 1, [], None, started_at=0)   # a bad folder must not raise
    assert "could not write the live status" in capsys.readouterr().err


@needs_mac
def test_a_path_typed_into_a_save_name_box_is_refused_with_the_split():
    msg = mt.save_name_path_refusal(7, "~/Library/Services/Hello.workflow", {7: ("TextField", "Save As:")})
    assert msg.startswith("Not typed") and "'Hello.workflow'" in msg and "path='~/Library/Services'" in msg
    assert mt.save_name_path_refusal(7, "Hello.workflow", {7: ("TextField", "Save As:")}) == ""
    assert mt.save_name_path_refusal(3, "~/Desktop/a.txt", {3: ("TextField", "Address and search bar")}) == ""


def test_finder_card_says_how_to_get_the_front_windows_folder():
    # 90828d3f (2026-11-24): the window "Title" was open, but the agent searched ~ with find and missed the folder.
    from tools.navigation.computer.macos_app_cards import card_for
    card = card_for("Finder")
    assert "target of front window" in card and "POSIX path" in card
