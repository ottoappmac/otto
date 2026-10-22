import sys
import time

import pytest

from backend import terminal

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="PTY needs a POSIX system")


def _wait(predicate, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def session():
    sid = "t-session"
    yield sid
    terminal.close_session(sid)


def _open(sid, tmp_path, **kw):
    kw.setdefault("sandbox", False)
    kw.setdefault("network", True)
    result = terminal.create(sid, tmp_path, **kw)
    assert result["ok"], result
    return result["terminal"]


def test_process_output_and_exit_code(session, tmp_path):
    term = _open(session, tmp_path, kind="process", command="echo hi; exit 3")
    assert _wait(lambda: not term.alive)
    assert _wait(lambda: "hi" in term.text_since(0)[0])
    assert term.exit_code == 3


def test_shell_run_returns_output_and_status(session, tmp_path):
    term = _open(session, tmp_path, kind="shell")
    time.sleep(1.0)
    first = terminal.run_in_terminal(term, "echo one && pwd", timeout=10)
    assert first["exit_code"] == 0
    assert first["output"].splitlines()[0] == "one"
    assert str(tmp_path.resolve()) in first["output"] or tmp_path.name in first["output"]
    failed = terminal.run_in_terminal(term, "false", timeout=10)
    assert failed["exit_code"] == 1


def test_shell_keeps_state_between_runs(session, tmp_path):
    term = _open(session, tmp_path, kind="shell")
    time.sleep(1.0)
    terminal.run_in_terminal(term, "export OTTO_X=42", timeout=10)
    out = terminal.run_in_terminal(term, "echo $OTTO_X", timeout=10)
    assert out["output"] == "42"


def test_secrets_are_not_inherited(session, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    term = _open(session, tmp_path, kind="process", command="env")
    assert _wait(lambda: not term.alive)
    assert "sk-secret" not in term.text_since(0)[0]


def test_limit_per_session(session, tmp_path):
    for _ in range(terminal.MAX_PER_SESSION):
        _open(session, tmp_path, kind="process", command="sleep 30")
    over = terminal.create(session, tmp_path, kind="process", command="sleep 30", sandbox=False, network=True)
    assert not over["ok"] and "running terminals" in over["error"]


def test_close_kills_process(session, tmp_path):
    term = _open(session, tmp_path, kind="process", command="sleep 60")
    assert terminal.close(session, term.id)
    assert _wait(lambda: not term.alive)
    assert terminal.get(session, term.id) is None


def test_attach_replays_scrollback_without_duplicates(session, tmp_path):
    term = _open(session, tmp_path, kind="process", command="echo before; sleep 1; echo after")
    assert _wait(lambda: "before" in term.text_since(0)[0])
    seen = []
    backlog = term.attach(seen.append)
    assert b"before" in backlog
    assert _wait(lambda: b"after" in b"".join(seen))
    assert b"before" not in b"".join(seen)


def test_strip_ansi():
    assert terminal.strip_ansi("\x1b[31mred\x1b[0m\r\nok") == "red\nok"


@pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt is macOS only")
def test_sandbox_blocks_writes_outside_project_and_network(session, tmp_path):
    term = _open(session, tmp_path, kind="shell", sandbox=True, network=False, write_paths=[tmp_path])
    time.sleep(1.5)
    ok = terminal.run_in_terminal(term, "echo x > inside.txt && echo wrote", timeout=10)
    assert "wrote" in ok["output"]
    denied = terminal.run_in_terminal(term, "echo x > /Users/Shared/otto-denied.txt; echo rc=$?", timeout=10)
    assert "rc=0" not in denied["output"]


@pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt is macOS only")
def test_readonly_terminal_cannot_write_project(session):
    # Temp folders are always writable in the profile, so use a folder outside them.
    import shutil
    from pathlib import Path

    folder = Path(__file__).parent / "_readonly_terminal_tmp"
    folder.mkdir(exist_ok=True)
    try:
        term = _open(session, folder, kind="shell", sandbox=True, network=False, write_paths=[])
        time.sleep(1.5)
        out = terminal.run_in_terminal(term, "echo x > blocked.txt; echo rc=$?", timeout=10)
        assert "rc=0" not in out["output"]
        assert not (folder / "blocked.txt").exists()
    finally:
        terminal.close_session(session)
        shutil.rmtree(folder, ignore_errors=True)
