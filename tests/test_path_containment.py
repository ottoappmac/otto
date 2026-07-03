"""Path-containment regression tests.

Covers the traversal fixes:
- ``load_source`` (doc_reader) must not read outside the session files dir.
- ``schedule_dir`` / ``trigger_dir`` must reject traversal-capable IDs.
- ``read_schedule_run_output`` must reject unsafe schedule/run IDs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.utils import (
    is_resolved_path_allowed,
    is_safe_path_segment,
    remap_to_virtual_path,
)


# ---------------------------------------------------------------------------
# is_safe_path_segment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("segment", ["daily-report", "run 20260613", "a", "x_1.json"])
def test_safe_segments_accepted(segment):
    assert is_safe_path_segment(segment)


@pytest.mark.parametrize(
    "segment",
    ["", ".", "..", "../x", "a/b", "a\\b", "a\x00b", "..\\..", "/etc"],
)
def test_unsafe_segments_rejected(segment):
    assert not is_safe_path_segment(segment)


# ---------------------------------------------------------------------------
# remap_to_virtual_path
# ---------------------------------------------------------------------------

def test_remap_absolute_under_root_becomes_virtual(tmp_path):
    root = tmp_path / "session" / "files"
    root.mkdir(parents=True)
    assert remap_to_virtual_path(str(root / "output" / "x.md"), root) == "/output/x.md"
    assert remap_to_virtual_path(str(root / "report.md"), root) == "/report.md"
    # Root itself maps to "/".
    assert remap_to_virtual_path(str(root), root) == "/"


def test_remap_leaves_relative_and_virtual_paths_unchanged(tmp_path):
    root = tmp_path / "files"
    root.mkdir()
    assert remap_to_virtual_path("report.md", root) == "report.md"
    assert remap_to_virtual_path("/output/x.md", root) == "/output/x.md"


def test_remap_leaves_outside_paths_unchanged(tmp_path):
    root = tmp_path / "files"
    root.mkdir()
    # Real absolute paths outside the root are NOT remapped (caller's guard
    # rejects them) — the boundary is never widened.
    assert remap_to_virtual_path("/etc/passwd", root) == "/etc/passwd"
    assert remap_to_virtual_path(str(tmp_path / "sibling.md"), root) == str(
        tmp_path / "sibling.md"
    )


# ---------------------------------------------------------------------------
# is_resolved_path_allowed (symlink-escape hardening)
# ---------------------------------------------------------------------------

def test_resolved_in_root_is_allowed(tmp_path):
    root = tmp_path / "files"
    (root / "output").mkdir(parents=True)
    full = root / "output" / "report.md"
    assert is_resolved_path_allowed(full, root, "/output/report.md")
    # Not-yet-created file under an in-root dir is still allowed.
    assert is_resolved_path_allowed(root / "output" / "new.md", root, "/output/new.md")


def test_links_symlink_escape_is_allowed(tmp_path):
    root = tmp_path / "files"
    (root / "links").mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    (external / "photo.png").write_text("img", encoding="utf-8")

    link = root / "links" / "photo.png"
    link.symlink_to(external / "photo.png")
    assert is_resolved_path_allowed(link, root, "/links/photo.png")

    # Sub-path inside a linked directory also allowed.
    dir_link = root / "links" / "userdir"
    dir_link.symlink_to(external)
    assert is_resolved_path_allowed(
        root / "links" / "userdir" / "photo.png", root, "/links/userdir/photo.png"
    )


def test_non_links_symlink_escape_is_rejected(tmp_path):
    root = tmp_path / "files"
    (root / "output").mkdir(parents=True)
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret", encoding="utf-8")

    # A symlink planted outside /links/ that points outside the root must be
    # rejected even though it lexically lives under the session root.
    evil = root / "output" / "evil"
    evil.symlink_to(secret)
    assert not is_resolved_path_allowed(evil, root, "/output/evil")


# ---------------------------------------------------------------------------
# scheduler / trigger directory helpers
# ---------------------------------------------------------------------------

def test_schedule_dir_rejects_traversal_id(tmp_path, monkeypatch):
    from backend import scheduler

    monkeypatch.setattr(scheduler, "_schedules_dir", lambda: tmp_path)
    with pytest.raises(ValueError):
        scheduler.schedule_dir("..")
    with pytest.raises(ValueError):
        scheduler.schedule_dir("../../etc")
    assert scheduler.schedule_dir("ok") == tmp_path / "ok"


def test_load_schedule_returns_none_for_unsafe_id(tmp_path, monkeypatch):
    from backend import scheduler

    monkeypatch.setattr(scheduler, "_schedules_dir", lambda: tmp_path)
    assert scheduler.load_schedule("..") is None
    assert scheduler.load_schedule("../../x") is None


def test_trigger_dir_rejects_traversal_id(tmp_path, monkeypatch):
    from backend import trigger_manager

    monkeypatch.setattr(trigger_manager, "_triggers_dir", lambda: tmp_path)
    with pytest.raises(ValueError):
        trigger_manager.trigger_dir("..")
    assert trigger_manager.load_trigger("..") is None
    assert trigger_manager.trigger_dir("ok") == tmp_path / "ok"


# ---------------------------------------------------------------------------
# read_schedule_run_output tool
# ---------------------------------------------------------------------------

def _make_schedule(tmp_path: Path, schedule_id: str = "daily") -> Path:
    from backend.schemas import ScheduleSpec

    d = tmp_path / schedule_id
    d.mkdir(parents=True)
    spec = ScheduleSpec(id=schedule_id, prompt="p", cron_expression="0 9 * * *")
    (d / "schedule.json").write_text(spec.model_dump_json(), encoding="utf-8")
    return d


def _get_tool(name: str):
    from backend.schedule_tools import build_schedule_tools

    return next(t for t in build_schedule_tools() if t.name == name)


def test_read_run_output_rejects_unsafe_run_id(tmp_path, monkeypatch):
    from backend import scheduler

    monkeypatch.setattr(scheduler, "_schedules_dir", lambda: tmp_path)
    sched = _make_schedule(tmp_path)
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret", encoding="utf-8")
    # A files dir reachable only via traversal through run_id.
    (sched / "runs").mkdir()

    tool = _get_tool("read_schedule_run_output")
    out = tool.invoke({"schedule_id": "daily", "run_id": "../..", "file_path": "secret.txt"})
    assert "top secret" not in out
    assert "not found" in out.lower()


def test_read_run_output_rejects_unsafe_schedule_id(tmp_path, monkeypatch):
    from backend import scheduler

    monkeypatch.setattr(scheduler, "_schedules_dir", lambda: tmp_path)
    tool = _get_tool("read_schedule_run_output")
    out = tool.invoke({"schedule_id": "../../etc", "run_id": "r1", "file_path": "passwd"})
    assert "not found" in out.lower()


def test_read_run_output_happy_path_and_truncation(tmp_path, monkeypatch):
    from backend import scheduler
    from backend.run_output import MAX_OUTPUT_FILE_BYTES

    monkeypatch.setattr(scheduler, "_schedules_dir", lambda: tmp_path)
    sched = _make_schedule(tmp_path)
    files = sched / "runs" / "r1" / "files"
    files.mkdir(parents=True)
    (files / "report.txt").write_text("hello world", encoding="utf-8")
    (files / "big.txt").write_text("x" * (MAX_OUTPUT_FILE_BYTES + 100), encoding="utf-8")

    tool = _get_tool("read_schedule_run_output")

    listing = tool.invoke({"schedule_id": "daily", "run_id": "r1"})
    assert "report.txt" in listing and "big.txt" in listing

    out = tool.invoke({"schedule_id": "daily", "run_id": "r1", "file_path": "report.txt"})
    assert "hello world" in out

    big = tool.invoke({"schedule_id": "daily", "run_id": "r1", "file_path": "big.txt"})
    assert "truncated" in big
    assert f"{MAX_OUTPUT_FILE_BYTES + 100} bytes total" in big

    escape = tool.invoke({"schedule_id": "daily", "run_id": "r1", "file_path": "../../../secret.txt"})
    assert "outside the run output directory" in escape


# ---------------------------------------------------------------------------
# read_trigger_run_output tool
# ---------------------------------------------------------------------------

def _make_trigger(tmp_path: Path, trigger_id: str = "watcher") -> Path:
    from backend.schemas import TriggerSpec

    d = tmp_path / trigger_id
    d.mkdir(parents=True)
    spec = TriggerSpec(id=trigger_id, type="fileos", prompt="p", path="/tmp")
    (d / "trigger.json").write_text(spec.model_dump_json(), encoding="utf-8")
    return d


def _get_trigger_tool(name: str):
    from backend.trigger_tools import build_trigger_tools

    return next(t for t in build_trigger_tools() if t.name == name)


def test_trigger_read_run_output_rejects_unsafe_run_id(tmp_path, monkeypatch):
    from backend import trigger_manager

    monkeypatch.setattr(trigger_manager, "_triggers_dir", lambda: tmp_path)
    trig = _make_trigger(tmp_path)
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret", encoding="utf-8")
    (trig / "runs").mkdir()

    tool = _get_trigger_tool("read_trigger_run_output")
    out = tool.invoke({"trigger_id": "watcher", "run_id": "../..", "file_path": "secret.txt"})
    assert "top secret" not in out
    assert "not found" in out.lower()


def test_trigger_read_run_output_rejects_unsafe_trigger_id(tmp_path, monkeypatch):
    from backend import trigger_manager

    monkeypatch.setattr(trigger_manager, "_triggers_dir", lambda: tmp_path)
    tool = _get_trigger_tool("read_trigger_run_output")
    out = tool.invoke({"trigger_id": "../../etc", "run_id": "r1", "file_path": "passwd"})
    assert "not found" in out.lower()


def test_trigger_read_run_output_happy_path_and_truncation(tmp_path, monkeypatch):
    from backend import trigger_manager
    from backend.run_output import MAX_OUTPUT_FILE_BYTES

    monkeypatch.setattr(trigger_manager, "_triggers_dir", lambda: tmp_path)
    trig = _make_trigger(tmp_path)
    files = trig / "runs" / "r1" / "files"
    files.mkdir(parents=True)
    (files / "report.txt").write_text("hello world", encoding="utf-8")
    (files / "big.txt").write_text("x" * (MAX_OUTPUT_FILE_BYTES + 100), encoding="utf-8")

    tool = _get_trigger_tool("read_trigger_run_output")

    listing = tool.invoke({"trigger_id": "watcher", "run_id": "r1"})
    assert "report.txt" in listing and "big.txt" in listing

    out = tool.invoke({"trigger_id": "watcher", "run_id": "r1", "file_path": "report.txt"})
    assert "hello world" in out

    big = tool.invoke({"trigger_id": "watcher", "run_id": "r1", "file_path": "big.txt"})
    assert "truncated" in big
    assert f"{MAX_OUTPUT_FILE_BYTES + 100} bytes total" in big

    escape = tool.invoke({"trigger_id": "watcher", "run_id": "r1", "file_path": "../../../secret.txt"})
    assert "outside the run output directory" in escape


# ---------------------------------------------------------------------------
# load_source (doc_reader)
# ---------------------------------------------------------------------------

async def test_load_source_reads_inside_files_dir(tmp_path):
    from tools.research._loaders import load_source

    (tmp_path / "notes.txt").write_text("inside content", encoding="utf-8")
    docs = await load_source("notes.txt", files_dir=tmp_path)
    assert docs and "inside content" in docs[0].page_content

    # Session-style absolute path ("/uploads/...") resolves under files_dir.
    (tmp_path / "uploads").mkdir()
    (tmp_path / "uploads" / "a.txt").write_text("uploaded", encoding="utf-8")
    docs = await load_source("/uploads/a.txt", files_dir=tmp_path)
    assert docs and "uploaded" in docs[0].page_content


async def test_load_source_rejects_relative_escape(tmp_path):
    from tools.research._loaders import load_source

    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    files_dir = tmp_path / "session"
    files_dir.mkdir()

    with pytest.raises(ValueError, match="outside the session directory"):
        await load_source("../outside.txt", files_dir=files_dir)


async def test_load_source_rejects_absolute_outside_path(tmp_path):
    from tools.research._loaders import load_source

    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    files_dir = tmp_path / "session"
    files_dir.mkdir()

    with pytest.raises(ValueError, match="outside the session directory"):
        await load_source(str(outside), files_dir=files_dir)


async def test_load_source_unrestricted_without_files_dir(tmp_path):
    from tools.research._loaders import load_source

    f = tmp_path / "free.txt"
    f.write_text("no sandbox", encoding="utf-8")
    docs = await load_source(str(f))
    assert docs and "no sandbox" in docs[0].page_content
