"""Live smoke tests: agent writes files into the virtual session filesystem.

These tests exercise the path-remapping and symlink-hardening changes end-to-end:
- The agent can save a file using the virtual path  (/output/...)
- The agent can save a file using the absolute session-files path ({files_dir}/output/...) —
  the form it naturally builds when the prompt supplies the real SESSION_FILES path.
- Escape paths are rejected and do NOT create real files.

Run with::

    pytest -m live tests/live/test_file_write_smoke.py -v
"""

from __future__ import annotations

import pytest

from backend.session_manager import _session_files_dir
from tests.live.conftest import run_session

pytestmark = pytest.mark.live


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _done_events(events: list[dict]) -> list[dict]:
    return [e for e in events if e.get("type") == "done"]


def _tool_calls(events: list[dict], name: str | None = None) -> list[dict]:
    calls = [e for e in events if e.get("type") == "tool_call"]
    if name:
        calls = [e for e in calls if e.get("content") == name]
    return calls


def _tool_results(events: list[dict], name: str | None = None) -> list[dict]:
    results = [e for e in events if e.get("type") == "tool_result"]
    if name:
        results = [e for e in results if e.get("metadata", {}).get("name") == name]
    return results


# ---------------------------------------------------------------------------
# 1. Write via virtual path — the normal/recommended form
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_writes_file_via_virtual_path(session_manager, live_session):
    """Agent saves a file using the /output/... virtual path.

    This is the baseline: confirms the fix didn't break the path that already
    worked.  The file must appear on disk under the session files directory.
    """
    events = await run_session(
        session_manager,
        live_session.id,
        "Use write_file to create /output/smoke-virtual.txt containing exactly: SMOKE_VIRTUAL_OK",
    )

    assert _done_events(events), f"Stream did not complete cleanly. Events: {events}"
    assert _tool_calls(events, "write_file"), (
        f"Expected a write_file tool call; got: {[e.get('content') for e in _tool_calls(events)]}"
    )

    out = _session_files_dir(live_session.id) / "output" / "smoke-virtual.txt"
    assert out.exists(), (
        f"Expected file at {out}; files in output dir: "
        f"{list(out.parent.glob('*')) if out.parent.exists() else '(no output/ dir)'}"
    )
    assert "SMOKE_VIRTUAL_OK" in out.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 2. Write via absolute session-files path — the remapping fix
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_writes_file_via_absolute_session_path(session_manager, live_session):
    """Agent saves a file using the absolute {files_dir}/output/... path.

    This is the scenario that broke before the remap fix: the agent was handed
    the real files_dir in the system prompt and built an absolute path, which
    the old guard rejected.  After the fix, such paths are transparently remapped
    to their virtual equivalent and the write succeeds.
    """
    files_dir = _session_files_dir(live_session.id)
    abs_path = str(files_dir / "output" / "smoke-absolute.txt")

    events = await run_session(
        session_manager,
        live_session.id,
        f"Use write_file to create {abs_path} containing exactly: SMOKE_ABSOLUTE_OK",
    )

    assert _done_events(events), f"Stream did not complete cleanly. Events: {events}"

    # The write_file tool result should NOT contain the old rejection message.
    wf_results = _tool_results(events, "write_file")
    for r in wf_results:
        content = str(r.get("content", ""))
        assert "looks like a real filesystem path" not in content, (
            f"write_file rejected the absolute path with the old error: {content}"
        )
        assert "outside the session files directory" not in content, (
            f"write_file rejected the absolute path: {content}"
        )

    out = files_dir / "output" / "smoke-absolute.txt"
    assert out.exists(), (
        f"Expected file at {out}; files in output dir: "
        f"{list(out.parent.glob('*')) if out.parent.exists() else '(no output/ dir)'}"
    )
    assert "SMOKE_ABSOLUTE_OK" in out.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 3. Read back a previously written file — round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_reads_back_file_it_wrote(session_manager, live_session):
    """Write a file then read it back in the same session.

    Confirms that the virtual path is stable across write and read and that
    the remap works consistently in both directions.
    """
    events_write = await run_session(
        session_manager,
        live_session.id,
        "Use write_file to create /output/roundtrip.txt containing exactly: ROUNDTRIP_42",
    )
    assert _done_events(events_write), f"Write did not complete. Events: {events_write}"

    events_read = await run_session(
        session_manager,
        live_session.id,
        "Read the file /output/roundtrip.txt and tell me its exact contents.",
    )
    assert _done_events(events_read), f"Read did not complete. Events: {events_read}"

    agent_replies = " ".join(
        e["content"] for e in events_read if e.get("type") == "agent"
    )
    assert "ROUNDTRIP_42" in agent_replies, (
        f"Agent did not echo back file contents. Reply: {agent_replies[:500]}"
    )


# ---------------------------------------------------------------------------
# 4. Escape paths are rejected (security regression)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_cannot_write_to_path_outside_session_root(session_manager, live_session):
    """Asking the agent to write to /etc/passwd or a real OS path outside the
    session root should be blocked — the file must not be created on disk.

    We assert two things:
    1. The on-disk path was not written.
    2. The tool result contains an error message (not a silent success).
    """
    escape_path = "/etc/otto-smoke-test-should-not-exist.txt"

    events = await run_session(
        session_manager,
        live_session.id,
        f"Use write_file to create {escape_path} containing: ESCAPED",
    )

    # The real OS file must not exist.
    from pathlib import Path
    assert not Path(escape_path).exists(), (
        f"Security regression: agent was able to write to {escape_path}"
    )

    # The write_file tool result (if it was called) should contain an error.
    wf_results = _tool_results(events, "write_file")
    if wf_results:
        for r in wf_results:
            content = str(r.get("content", ""))
            assert any(kw in content.lower() for kw in ("error", "outside", "cannot", "not allowed")), (
                f"Expected an error in write_file result for escape path; got: {content}"
            )
