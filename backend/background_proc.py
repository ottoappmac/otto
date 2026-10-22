"""Long-running shell processes for a session.

``execute`` waits for the command. A dev server or a watch script needs to
stay up. Each one runs on a terminal (see ``backend.terminal``), so it keeps
its colours and the user sees the same live output in the Terminal tab. The
sandbox, network grant and environment scrubbing are the ones ``execute`` uses.
A session keeps a handful of them.
"""

from __future__ import annotations

from backend import terminal

_MAX_OUTPUT = 200_000


def start(session_id: str, command: str) -> dict:
    """Start *command* in the session's project. ``{ok, id, pid}`` or ``{ok: False, error}``."""
    from backend.terminal_context import open_terminal

    result = open_terminal(session_id, kind="process", command=command)
    if not result.get("ok"):
        return {"ok": False, "error": result.get("error") or "Could not start the process."}
    term = result["terminal"]
    return {"ok": True, "id": term.id, "pid": term.popen.pid}


def output(session_id: str, proc_id: str, tail: int = _MAX_OUTPUT) -> dict:
    term = terminal.get(session_id, proc_id)
    if term is None:
        return {"ok": False, "error": "No such background process."}
    text, _ = term.text_since(0)
    return {
        "ok": True,
        "id": proc_id,
        "running": term.alive,
        "exit_code": term.exit_code,
        "output": terminal.strip_ansi(text)[-tail:],
    }


def stop(session_id: str, proc_id: str) -> dict:
    if not terminal.close(session_id, proc_id):
        return {"ok": False, "error": "No such background process."}
    return {"ok": True, "id": proc_id}


def ports(session_id: str, proc_id: str) -> dict:
    term = terminal.get(session_id, proc_id)
    if term is None:
        return {"ok": False, "error": "No such background process."}
    import shutil
    if not shutil.which("lsof"):
        return {"ok": True, "id": proc_id, "ports": [], "note": "lsof is not installed."}
    return {"ok": True, "id": proc_id, "ports": terminal.listening_ports(term)}


def stop_session(session_id: str) -> None:
    terminal.close_session(session_id)
