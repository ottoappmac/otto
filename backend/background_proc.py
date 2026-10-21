"""Long-running shell processes for a session.

``execute`` waits for the command. A dev server or a watch script needs
to stay up. Each process is sandboxed with the same profile as
``execute`` when the sandbox is on. A session keeps a handful of them.
"""

from __future__ import annotations

import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

_MAX_PER_SESSION = 4
_MAX_OUTPUT = 200_000

_lock = threading.Lock()
_procs: dict[str, dict[str, "BackgroundProcess"]] = {}


@dataclass
class BackgroundProcess:
    id: str
    command: str
    popen: subprocess.Popen[str]
    output: list[str] = field(default_factory=list)
    _size: int = 0

    def append(self, text: str) -> None:
        if not text or self._size >= _MAX_OUTPUT:
            return
        room = _MAX_OUTPUT - self._size
        chunk = text[:room]
        self.output.append(chunk)
        self._size += len(chunk)


def _reader(proc: BackgroundProcess) -> None:
    stream = proc.popen.stdout
    if stream is None:
        return
    try:
        for line in stream:
            with _lock:
                proc.append(line)
    except Exception:
        return


def start(session_id: str, command: str, cwd: Path, *, sandbox: bool, network: bool, network_hosts: list[str], write_paths: list[Path]) -> dict:
    """Start *command* in *cwd*. Returns ``{ok, id}`` or ``{ok: False, error}``."""
    if not command or not command.strip():
        return {"ok": False, "error": "Command must be a non-empty string."}
    with _lock:
        current = _procs.setdefault(session_id, {})
        if len(current) >= _MAX_PER_SESSION:
            return {"ok": False, "error": f"This session already has {_MAX_PER_SESSION} background processes."}
    argv = ["/bin/sh", "-c", command]
    if sandbox:
        from backend.exec_sandbox import _sandbox_exec, render_profile
        import sys
        if sys.platform != "darwin":
            return {"ok": False, "error": "The sandbox is on, and this system has no Seatbelt. The process was not started."}
        binary = _sandbox_exec()
        if not binary:
            return {"ok": False, "error": "sandbox-exec was not found. The process was not started."}
        profile = cwd / f".bg-{uuid.uuid4().hex}.sb"
        profile.write_text(
            render_profile(
                write_paths=write_paths or [cwd],
                network=network,
                network_hosts=network_hosts,
                session_paths=write_paths,
            ),
            encoding="utf-8",
        )
        argv = [binary, "-f", str(profile), *argv]
    try:
        popen = subprocess.Popen(
            argv,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        return {"ok": False, "error": str(exc)}
    proc = BackgroundProcess(id=uuid.uuid4().hex[:8], command=command, popen=popen)
    threading.Thread(target=_reader, args=(proc,), daemon=True).start()
    with _lock:
        _procs.setdefault(session_id, {})[proc.id] = proc
    return {"ok": True, "id": proc.id, "pid": popen.pid}


def output(session_id: str, proc_id: str) -> dict:
    with _lock:
        proc = _procs.get(session_id, {}).get(proc_id)
        if proc is None:
            return {"ok": False, "error": "No such background process."}
        text = "".join(proc.output)
        running = proc.popen.poll() is None
        code = proc.popen.returncode
    return {"ok": True, "id": proc_id, "running": running, "exit_code": code, "output": text}


def stop(session_id: str, proc_id: str) -> dict:
    with _lock:
        proc = _procs.get(session_id, {}).pop(proc_id, None)
    if proc is None:
        return {"ok": False, "error": "No such background process."}
    if proc.popen.poll() is None:
        try:
            import os
            import signal
            os.killpg(proc.popen.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            proc.popen.terminate()
    return {"ok": True, "id": proc_id}


def ports(session_id: str, proc_id: str) -> dict:
    """Listening TCP ports for a background process, via ``lsof`` when it exists."""
    with _lock:
        proc = _procs.get(session_id, {}).get(proc_id)
        if proc is None:
            return {"ok": False, "error": "No such background process."}
        pid = proc.popen.pid
    import shutil
    if not shutil.which("lsof"):
        return {"ok": True, "id": proc_id, "ports": [], "note": "lsof is not installed."}
    try:
        listed = subprocess.run(
            ["lsof", "-nP", "-p", str(pid), "-iTCP", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}
    found: list[int] = []
    for line in (listed.stdout or "").splitlines()[1:]:
        for token in line.split():
            if ":" in token:
                maybe = token.rsplit(":", 1)[-1]
                if maybe.isdigit():
                    port = int(maybe)
                    if port not in found:
                        found.append(port)
    return {"ok": True, "id": proc_id, "ports": found}


def stop_session(session_id: str) -> None:
    with _lock:
        procs = list(_procs.pop(session_id, {}).values())
    for proc in procs:
        if proc.popen.poll() is None:
            try:
                proc.popen.terminate()
            except OSError:
                pass
