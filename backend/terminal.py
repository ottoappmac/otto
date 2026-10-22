"""Pseudo-terminals for a Code session.

One mechanism serves two jobs: an interactive shell the user types into, and
the long-running commands the agent starts (a dev server, a watcher). Both run
on a PTY, so output keeps its colours, a full-screen program works, and the
user and the agent read the same scrollback.

Every terminal runs under the same Seatbelt profile and scrubbed environment
as ``execute``. A read-only folder gets no write path. A session keeps a
handful of terminals.
"""

from __future__ import annotations

import fcntl
import logging
import os
import pty
import re
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

MAX_PER_SESSION = 6
SCROLLBACK_BYTES = 400_000
START_MARK = "\x1b]777;otto-start\x07"
DONE_MARK = "\x1b]777;otto-done;"
_DONE_RE = re.compile(r"\x1b\]777;otto-done;(-?\d+)\x07")
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")

_lock = threading.RLock()
_terminals: dict[str, dict[str, "Terminal"]] = {}


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text).replace("\r\n", "\n").replace("\r", "\n")


@dataclass
class Terminal:
    id: str
    session_id: str
    kind: str  # "shell" | "process"
    title: str
    cwd: str
    command: str | None
    popen: subprocess.Popen
    master_fd: int
    created: float = field(default_factory=time.time)
    scrollback: bytearray = field(default_factory=bytearray)
    # Total bytes ever written, so a reader can ask for "everything after N".
    total: int = 0
    listeners: list[Callable[[bytes], None]] = field(default_factory=list)
    exit_listeners: list[Callable[[int | None], None]] = field(default_factory=list)
    _closing: bool = False

    @property
    def alive(self) -> bool:
        return self.popen.poll() is None

    @property
    def exit_code(self) -> int | None:
        return self.popen.returncode

    def info(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "command": self.command,
            "cwd": self.cwd,
            "running": self.alive,
            "exit_code": self.exit_code,
            "pid": self.popen.pid,
            "created": self.created,
        }

    def _feed(self, data: bytes) -> None:
        with _lock:
            self.scrollback.extend(data)
            self.total += len(data)
            overflow = len(self.scrollback) - SCROLLBACK_BYTES
            if overflow > 0:
                del self.scrollback[:overflow]
            # Listeners run under the lock so attach() can take a snapshot and
            # subscribe without missing or repeating a chunk. They only queue.
            for listener in list(self.listeners):
                try:
                    listener(data)
                except Exception:  # noqa: BLE001
                    logger.debug("terminal listener failed", exc_info=True)

    def attach(self, listener: Callable[[bytes], None]) -> bytes:
        """Subscribe to new output. Returns the scrollback so far."""
        with _lock:
            self.listeners.append(listener)
            return bytes(self.scrollback)

    def detach(self, listener: Callable[[bytes], None]) -> None:
        with _lock:
            if listener in self.listeners:
                self.listeners.remove(listener)

    def snapshot(self) -> tuple[bytes, int]:
        """The retained scrollback and the total byte count it ends at."""
        with _lock:
            return bytes(self.scrollback), self.total

    def text_since(self, offset: int) -> tuple[str, int]:
        """Plain text written after byte *offset* (clamped to what is retained)."""
        with _lock:
            start = max(0, offset - (self.total - len(self.scrollback)))
            chunk = bytes(self.scrollback[start:])
            end = self.total
        return chunk.decode("utf-8", errors="replace"), end

    def write(self, data: bytes) -> None:
        if not self.alive:
            return
        try:
            os.write(self.master_fd, data)
        except OSError:
            pass

    def resize(self, rows: int, cols: int) -> None:
        if rows < 1 or cols < 1:
            return
        try:
            fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
            if self.alive:
                os.killpg(self.popen.pid, signal.SIGWINCH)
        except (OSError, ProcessLookupError):
            pass

    def interrupt(self) -> None:
        self.write(b"\x03")

    def kill(self) -> None:
        self._closing = True
        if self.alive:
            try:
                os.killpg(self.popen.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    self.popen.terminate()
                except OSError:
                    pass
            threading.Timer(2.0, self._force_kill).start()

    def _force_kill(self) -> None:
        if self.alive:
            try:
                os.killpg(self.popen.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass


def _set_size(fd: int, rows: int, cols: int) -> None:
    try:
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    except OSError:
        pass


def _user_shell() -> str:
    candidate = os.environ.get("SHELL") or "/bin/zsh"
    return candidate if Path(candidate).exists() else "/bin/sh"


def _wrap_sandbox(
    argv: list[str],
    *,
    cwd: Path,
    network: bool,
    network_hosts: list[str],
    write_paths: list[Path],
    read_paths: list[Path],
) -> tuple[list[str] | None, str | None, Path | None]:
    """``(argv, error, profile_dir)``. The caller deletes ``profile_dir`` after the spawn."""
    if sys.platform != "darwin":
        return None, "The sandbox is on, and this system has no Seatbelt. The terminal was not started.", None
    from backend.exec_sandbox import _sandbox_exec, render_profile

    binary = _sandbox_exec()
    if not binary:
        return None, "sandbox-exec was not found. The terminal was not started.", None
    folder = Path(tempfile.mkdtemp(prefix="otto-term-"))
    profile = folder / "terminal.sb"
    try:
        profile.write_text(
            render_profile(
                write_paths=write_paths,
                read_paths=read_paths,
                network=network,
                network_hosts=network_hosts,
            ),
            encoding="utf-8",
        )
        profile.chmod(0o600)
    except OSError as exc:
        shutil.rmtree(folder, ignore_errors=True)
        return None, f"Could not write the sandbox profile ({exc}). The terminal was not started.", None
    return [binary, "-f", str(profile), *argv], None, folder


def create(
    session_id: str,
    cwd: Path,
    *,
    kind: str = "shell",
    command: str | None = None,
    sandbox: bool,
    network: bool,
    network_hosts: list[str] | None = None,
    write_paths: list[Path] | None = None,
    read_paths: list[Path] | None = None,
    env: dict[str, str] | None = None,
    rows: int = 24,
    cols: int = 100,
) -> dict:
    """Start a terminal. Returns ``{ok, terminal}`` or ``{ok: False, error}``."""
    if kind == "process" and not (command and command.strip()):
        return {"ok": False, "error": "Command must be a non-empty string."}
    with _lock:
        live = [t for t in _terminals.get(session_id, {}).values() if t.alive]
        if len(live) >= MAX_PER_SESSION:
            return {"ok": False, "error": f"This session already has {MAX_PER_SESSION} running terminals."}

    from backend.exec_sandbox import passthrough_names, scrub_env

    environment = scrub_env(os.environ, extra=env or {}, passthrough=passthrough_names())
    environment.setdefault("TERM", "xterm-256color")
    environment["TERM"] = "xterm-256color"
    environment["COLORTERM"] = "truecolor"
    # The sandbox forbids writing the history file; skip the noise.
    environment["HISTFILE"] = "/dev/null"
    environment["SHELL_SESSIONS_DISABLE"] = "1"

    if kind == "process":
        argv = ["/bin/sh", "-c", command or ""]
        title = (command or "").strip().splitlines()[0][:60]
    else:
        argv = [_user_shell(), "-i"]
        title = "Shell"

    profile_dir: Path | None = None
    if sandbox:
        wrapped, error, profile_dir = _wrap_sandbox(
            argv,
            cwd=cwd,
            network=network,
            network_hosts=list(network_hosts or []),
            write_paths=list(write_paths or []),
            read_paths=list(read_paths or []),
        )
        if wrapped is None:
            return {"ok": False, "error": error}
        argv = wrapped

    master_fd, slave_fd = pty.openpty()
    _set_size(slave_fd, rows, cols)
    try:
        popen = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=environment,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            start_new_session=True,
            close_fds=True,
            preexec_fn=lambda: fcntl.ioctl(0, termios.TIOCSCTTY, 0),
        )
    except OSError as exc:
        os.close(master_fd)
        os.close(slave_fd)
        if profile_dir:
            shutil.rmtree(profile_dir, ignore_errors=True)
        return {"ok": False, "error": str(exc)}
    finally:
        try:
            os.close(slave_fd)
        except OSError:
            pass

    term = Terminal(
        id=uuid.uuid4().hex[:8],
        session_id=session_id,
        kind=kind,
        title=title,
        cwd=str(cwd),
        command=command,
        popen=popen,
        master_fd=master_fd,
    )
    with _lock:
        _terminals.setdefault(session_id, {})[term.id] = term
    threading.Thread(target=_reader, args=(term, profile_dir), daemon=True).start()
    return {"ok": True, "terminal": term}


def _reader(term: Terminal, profile_dir: Path | None) -> None:
    try:
        while True:
            try:
                data = os.read(term.master_fd, 65536)
            except OSError:
                break
            if not data:
                break
            term._feed(data)
    finally:
        try:
            term.popen.wait(timeout=5)
        except subprocess.TimeoutExpired:
            term._force_kill()
        try:
            os.close(term.master_fd)
        except OSError:
            pass
        if profile_dir:
            shutil.rmtree(profile_dir, ignore_errors=True)
        with _lock:
            listeners = list(term.exit_listeners)
        for listener in listeners:
            try:
                listener(term.exit_code)
            except Exception:  # noqa: BLE001
                logger.debug("terminal exit listener failed", exc_info=True)


def get(session_id: str, terminal_id: str) -> Terminal | None:
    with _lock:
        return _terminals.get(session_id, {}).get(terminal_id)


def list_terminals(session_id: str) -> list[Terminal]:
    with _lock:
        return sorted(_terminals.get(session_id, {}).values(), key=lambda t: t.created)


def close(session_id: str, terminal_id: str) -> bool:
    """Stop a terminal and forget it."""
    with _lock:
        term = _terminals.get(session_id, {}).pop(terminal_id, None)
    if term is None:
        return False
    term.kill()
    return True


def close_session(session_id: str) -> None:
    with _lock:
        terms = list(_terminals.pop(session_id, {}).values())
    for term in terms:
        term.kill()


def listening_ports(term: Terminal) -> list[int]:
    """TCP ports the terminal's process tree listens on, via ``lsof``."""
    if not shutil.which("lsof") or not term.alive:
        return []
    pids = {term.popen.pid}
    try:
        tree = subprocess.run(
            ["pgrep", "-g", str(term.popen.pid)], capture_output=True, text=True, timeout=3, check=False,
        )
        pids.update(int(p) for p in tree.stdout.split() if p.isdigit())
    except (OSError, subprocess.TimeoutExpired):
        pass
    found: list[int] = []
    try:
        listed = subprocess.run(
            ["lsof", "-a", "-nP", "-p", ",".join(str(p) for p in sorted(pids)), "-iTCP", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    for line in (listed.stdout or "").splitlines()[1:]:
        for token in line.split():
            if ":" in token:
                maybe = token.rsplit(":", 1)[-1]
                if maybe.isdigit() and int(maybe) not in found:
                    found.append(int(maybe))
    return found


def run_in_terminal(term: Terminal, command: str, *, timeout: float = 120.0) -> dict:
    """Type *command* into a shell terminal and wait for it to finish.

    Silent OSC markers bracket the command's output and carry its exit status
    back, so the user sees a normal terminal and the agent still knows where
    the output starts and when the command ended.
    """
    if not term.alive:
        return {"ok": False, "error": "That terminal has exited."}
    if term.kind != "shell":
        return {"ok": False, "error": "That terminal runs one process. Use process_output to read it."}
    _, start = term.snapshot()
    body = command.rstrip("\n")
    line = f"printf '\\033]777;otto-start\\007'; {{ {body}\n}}; printf '\\033]777;otto-done;%d\\007' $?\n"
    term.write(line.encode("utf-8"))
    deadline = time.time() + timeout
    seen = ""
    end = start
    while time.time() < deadline:
        seen, end = term.text_since(start)
        match = _DONE_RE.search(seen)
        if match:
            return {"ok": True, "output": _between(seen[: match.start()]), "exit_code": int(match.group(1)), "offset": end}
        if not term.alive:
            break
        time.sleep(0.1)
    return {"ok": True, "output": _between(seen), "exit_code": None, "timed_out": term.alive, "offset": end}


def _between(raw: str) -> str:
    """Output after the start marker, with escape codes removed."""
    at = raw.find(START_MARK)
    if at >= 0:
        raw = raw[at + len(START_MARK):]
    return strip_ansi(raw).strip("\n")


def close_all() -> None:
    """Kill every terminal. Called when the backend exits."""
    with _lock:
        sessions = list(_terminals)
    for session_id in sessions:
        close_session(session_id)
