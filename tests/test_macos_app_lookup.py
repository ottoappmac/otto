"""Suggestions for a failed app launch (macos_tools._similar_installed_apps)."""

from tools.navigation.computer.macos_tools import (_is_helper, _is_regular_app, _new_regular_apps, _pick_running_app,
                                                    _similar_installed_apps, _squash)


def apps(tmp_path, *names):
    for n in names:
        (tmp_path / f"{n}.app").mkdir()
    return [tmp_path]


def test_hyphen_and_space_compare_equal(tmp_path):
    d = apps(tmp_path, "Anaconda-Navigator", "Safari", "OBS")
    assert _similar_installed_apps("Anaconda Navigator", d) == ["Anaconda-Navigator"]
    assert _similar_installed_apps("anaconda-navigator", d)[0] == "Anaconda-Navigator"


def test_partial_names_match(tmp_path):
    d = apps(tmp_path, "Visual Studio Code", "Safari")
    assert _similar_installed_apps("Visual Studio", d) == ["Visual Studio Code"]


def test_missing_app_has_no_suggestion(tmp_path):
    d = apps(tmp_path, "Safari", "Keka", "OBS")
    assert _similar_installed_apps("Keynote", d) == []
    assert _similar_installed_apps("Pages", d) == []


def test_path_argument_uses_the_file_name_and_other_files_are_ignored(tmp_path):
    d = apps(tmp_path, "Anaconda-Navigator")
    (tmp_path / "notes.txt").write_text("x")
    assert _similar_installed_apps("/Applications/Anaconda Navigator.app", d) == ["Anaconda-Navigator"]


def test_unreadable_dir_and_empty_name_are_safe(tmp_path):
    assert _similar_installed_apps("Safari", [tmp_path / "missing"]) == []
    assert _similar_installed_apps("", [tmp_path]) == []


def test_limit_and_order(tmp_path):
    d = apps(tmp_path, *[f"Note{i}" for i in range(9)], "Notes")
    out = _similar_installed_apps("Notes", d, limit=3)
    assert out[0] == "Notes" and len(out) == 3


def test_squash():
    assert _squash("Anaconda-Navigator.app") == "anacondanavigator"


# ── the open_app tool itself ────────────────────────────────────────────────

import asyncio
import subprocess
import types

from tools.navigation.computer import macos_tools


def tk_of(tool):
    return tool._tk


def make_open_app(monkeypatch, tmp_path, returncode=1, stderr="Unable to find application named 'X'"):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=returncode, stderr=stderr, stdout="")

    monkeypatch.setattr(macos_tools.subprocess, "run", fake_run)
    monkeypatch.setattr(macos_tools, "_APP_DIRS", (tmp_path,))
    tk = macos_tools.MacOSToolkit()
    monkeypatch.setattr(tk, "_get_foreground_apps", lambda *a, **k: [])
    tool = next(t for t in tk.tools if t.name == "open_app")
    tool._tk = tk
    return tool, calls


def test_failed_open_lists_the_installed_name(monkeypatch, tmp_path):
    (tmp_path / "Anaconda-Navigator.app").mkdir()
    tool, calls = make_open_app(monkeypatch, tmp_path)
    out = asyncio.run(tool.ainvoke({"app_name": "Anaconda Navigator"}))
    assert out.startswith("open failed:") and "Anaconda-Navigator" in out
    assert calls == [["open", "-a", "Anaconda Navigator"]]                  # still the same command


def test_failed_open_of_missing_app_says_so(monkeypatch, tmp_path):
    (tmp_path / "Safari.app").mkdir()
    tool, _ = make_open_app(monkeypatch, tmp_path)
    out = asyncio.run(tool.ainvoke({"app_name": "Keynote"}))
    assert out.startswith("open failed:") and "may not be installed" in out


def test_existing_app_path_is_opened_by_path(monkeypatch, tmp_path):
    bundle = tmp_path / "Anaconda-Navigator.app"
    bundle.mkdir()
    tool, calls = make_open_app(monkeypatch, tmp_path, returncode=0, stderr="")
    # not running before the call (the lookup uses a cached list), running once the fresh list is read
    monkeypatch.setattr(tk_of(tool), "_get_foreground_apps",
                        lambda max_age=2.0, **k: ["Anaconda-Navigator"] if max_age == 0 else [])
    asyncio.run(tool.ainvoke({"app_name": str(bundle)}))
    assert calls[0] == ["open", "-a", str(bundle)]


def test_missing_app_path_keeps_the_name_lookup(monkeypatch, tmp_path):
    tool, calls = make_open_app(monkeypatch, tmp_path)
    asyncio.run(tool.ainvoke({"app_name": "/Applications/Nope Nope.app"}))
    assert calls[0] == ["open", "-a", "Nope Nope"]


# ── which running process is "the app" ──────────────────────────────────────

def test_exact_name_beats_a_prefix_match():
    assert _pick_running_app("reminders", ["RemindersWidgetExtension", "Reminders"]) == "Reminders"
    assert _pick_running_app("code", ["Code Helper", "Code"]) == "Code"
    assert _pick_running_app("notes", ["Notes Plus", "Notes"]) == "Notes"          # a real app that merely starts alike


def test_a_helper_process_is_not_the_app():
    assert _pick_running_app("reminders", ["RemindersWidgetExtension"]) is None
    assert _pick_running_app("safari", ["SafariBookmarksSyncAgent", "Safari Networking (Renderer)"]) is None
    assert _pick_running_app("notes", ["NotesWidgetExtension", "Finder"]) is None


def test_a_real_prefix_match_still_works():
    assert _pick_running_app("visual studio", ["Visual Studio Code", "Finder"]) == "Visual Studio Code"
    assert _pick_running_app("text", ["TextEdit"]) == "TextEdit"
    assert _pick_running_app("calc", ["Finder"]) is None


def test_is_helper():
    for n in ("RemindersWidgetExtension", "Code Helper (GPU)", "AMPDeviceDiscoveryAgent", "Dock Helper"):
        assert _is_helper(n), n
    for n in ("Reminders", "Finder", "Visual Studio Code", "System Settings", "Anaconda-Navigator"):
        assert not _is_helper(n), n


def test_open_app_does_not_return_early_on_a_helper(monkeypatch, tmp_path):
    """The Reminders failure: only the widget extension is running, so open_app must still run `open`."""
    tool, calls = make_open_app(monkeypatch, tmp_path, returncode=0, stderr="")
    state = {"opened": False}
    real_run = macos_tools.subprocess.run

    def run(cmd, **kw):
        state["opened"] = True
        return real_run(cmd, **kw) if False else types.SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(macos_tools.subprocess, "run", run)
    monkeypatch.setattr(tk_of(tool), "_get_foreground_apps",
                        lambda max_age=2.0, **k: {"RemindersWidgetExtension": 1, **({"Reminders": 2} if state["opened"] else {})})
    out = asyncio.run(tool.ainvoke({"app_name": "Reminders"}))
    assert state["opened"] and "Launched 'Reminders'" in out


# ── background daemons whose name is not a helper word (contactsd) ──────────

REGULAR = {10, 20}                              # pids the stand-in reports as regular apps


def regular(pid):
    return pid in REGULAR


def test_a_daemon_that_starts_like_the_app_is_not_the_app():
    running = {"contactsd": 5, "contactsdonationagent": 6}
    assert _pick_running_app("contacts", running, regular) is None
    assert _pick_running_app("contacts", {**running, "Contacts": 10}, regular) == "Contacts"


def test_a_prefix_match_needs_a_regular_app_when_the_process_can_be_asked():
    assert _pick_running_app("visual studio", {"Visual Studio Code": 10}, regular) == "Visual Studio Code"
    assert _pick_running_app("visual studio", {"Visual Studio Code": 99}, regular) is None
    assert _pick_running_app("visual studio", {"Visual Studio Code": 99}, lambda pid: None) == "Visual Studio Code"


def test_an_exact_name_is_taken_even_if_the_process_is_a_background_one():
    assert _pick_running_app("dock", {"Dock": 99}, regular) == "Dock"           # the user typed its exact name


def test_the_real_policy_check_on_this_mac():
    import os
    assert _is_regular_app(1) is False                                          # launchd: no NSRunningApplication
    assert _is_regular_app(os.getpid()) is False                                # a Python process is not an app
    from AppKit import NSWorkspace
    finder = next(a for a in NSWorkspace.sharedWorkspace().runningApplications() if a.localizedName() == "Finder")
    assert _is_regular_app(int(finder.processIdentifier())) is True


def test_open_app_opens_contacts_when_only_the_daemon_is_running(monkeypatch, tmp_path):
    tool, calls = make_open_app(monkeypatch, tmp_path, returncode=0, stderr="")
    state = {"opened": False}

    def run(cmd, **kw):
        state["opened"] = True
        return types.SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(macos_tools.subprocess, "run", run)
    monkeypatch.setattr(tk_of(tool), "_get_foreground_apps",
                        lambda max_age=2.0, **k: {"contactsd": 1, **({"Contacts": 2} if state["opened"] else {})})
    out = asyncio.run(tool.ainvoke({"app_name": "Contacts"}))
    assert state["opened"] and "Launched 'Contacts'" in out


# ── an app that opens under another name (Anaconda-Navigator shows up as "python") ──

def test_new_regular_apps_are_the_ones_that_appeared_and_have_windows():
    before = {"Finder": 1, "Dock": 2}
    after = {"Finder": 1, "Dock": 2, "python": 10, "python3": 20, "mdworker": 30}
    only_python = lambda pid: pid == 10
    assert _new_regular_apps(before, after, only_python) == ["python"]
    assert _new_regular_apps(before, before, only_python) == []
    assert _new_regular_apps({}, {"python": 10}, lambda pid: None) == []         # cannot tell: say nothing


def test_open_app_reports_the_new_name_when_the_app_opens_as_something_else(monkeypatch, tmp_path):
    tool, calls = make_open_app(monkeypatch, tmp_path, returncode=0, stderr="")
    state = {"opened": False}

    def run(cmd, **kw):
        state["opened"] = True
        return types.SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(macos_tools.subprocess, "run", run)
    monkeypatch.setattr(macos_tools, "_is_regular_app", lambda pid: pid == 10)
    monkeypatch.setattr(macos_tools, "_new_regular_apps",
                        lambda before, after, is_regular=None: [n for n, p in after.items()
                                                                 if n not in before and p == 10])
    monkeypatch.setattr(tk_of(tool), "_get_foreground_apps",
                        lambda max_age=2.0, **k: {"Finder": 1, **({"python": 10} if state["opened"] else {})})
    tk_of(tool).launch_delay = 0
    clock = iter(range(0, 10_000, 3))                       # the 5 s wait for the app ends after a few polls
    monkeypatch.setattr(macos_tools, "time", types.SimpleNamespace(monotonic=lambda: next(clock), sleep=lambda s: None))
    out = asyncio.run(tool.ainvoke({"app_name": "Anaconda-Navigator"}))
    assert "appears as 'python'" in out and "activate_app('python')" in out
