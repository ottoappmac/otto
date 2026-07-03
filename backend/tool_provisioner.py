"""Homebrew-free installation primitives shared across tool provisioners.

Otto installs a handful of external CLIs at runtime (Node for the
Playwright MCP, ``uv`` for MCP sandboxes, oMLX, exo prereqs).  Each of
those historically went through ``brew install``, which is a hard
dependency on Homebrew — a third-party package manager that is *not*
part of macOS and cannot be installed silently (it needs an admin
password and the Xcode Command Line Tools).

This module provides the building blocks for a brew-free default path:

* :func:`tools_dir` — a user-writable directory under Otto's app-data
  folder where portable binaries are unpacked (no sudo required).
* :class:`ToolJob` — a pollable background-job record (mirrors
  :class:`backend.omlx_provisioner.OmlxJob`) so install progress can be
  surfaced over REST.
* :func:`run_streaming` — stream a subprocess's combined output into a
  job log.
* :func:`download_file` / :func:`extract_tarball` / :func:`verify_sha256`
  — fetch and unpack official prebuilt artifacts.
* :func:`ainstall_uv` — install ``uv`` via its official standalone
  installer when Homebrew is absent.

The individual tool provisioners (``node_provisioner``,
``omlx_provisioner``) decide *which* path to take: they prefer
``brew install`` when :func:`backend.omlx_provisioner.has_homebrew`
returns true (fast, clean) and fall back to these primitives otherwise.
"""

from __future__ import annotations

import asyncio
import hashlib
import platform
import shutil
import stat
import tarfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

from backend.config import get_app_data_dir


def tools_dir() -> Path:
    """Return the user-writable directory for portable tool installs.

    Lives under Otto's app-data dir (``~/Library/Application Support/Otto``
    on macOS) so installs need no admin privileges and survive app
    upgrades.  Created on first access.
    """
    d = get_app_data_dir() / "tools"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Job tracking (mirrors backend.omlx_provisioner.OmlxJob)
# ---------------------------------------------------------------------------


@dataclass
class ToolJob:
    """Background install task with a bounded tail of log lines."""

    id: str
    kind: str  # e.g. "install-node" | "install-uv"
    status: str = "pending"  # pending | running | done | error
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    error: Optional[str] = None
    log_lines: list[str] = field(default_factory=list)
    result: Optional[dict] = None

    _LOG_TAIL = 600

    def append(self, line: str) -> None:
        line = line.rstrip("\n")
        if not line:
            return
        self.log_lines.append(line)
        if len(self.log_lines) > self._LOG_TAIL:
            self.log_lines = self.log_lines[-self._LOG_TAIL:]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "log_lines": list(self.log_lines),
            "result": self.result,
        }


_jobs: dict[str, ToolJob] = {}
_JOB_RETENTION_S = 60 * 60


def new_job(kind: str) -> ToolJob:
    cutoff = time.time() - _JOB_RETENTION_S
    for jid in list(_jobs.keys()):
        j = _jobs[jid]
        if j.finished_at and j.finished_at < cutoff:
            _jobs.pop(jid, None)
    job = ToolJob(id=uuid.uuid4().hex, kind=kind)
    _jobs[job.id] = job
    return job


def get_job(job_id: str) -> Optional[ToolJob]:
    return _jobs.get(job_id)


def list_jobs() -> list[ToolJob]:
    return sorted(_jobs.values(), key=lambda j: j.started_at, reverse=True)


# ---------------------------------------------------------------------------
# Subprocess streaming helper
# ---------------------------------------------------------------------------


async def run_streaming(
    cmd: list[str],
    *,
    job: ToolJob,
    timeout: float = 600.0,
    env: Optional[dict[str, str]] = None,
) -> int:
    """Run ``cmd`` and stream stdout/stderr lines into ``job.append``.

    Returns the process exit code. Raises :class:`asyncio.TimeoutError`
    if the process doesn't finish within ``timeout`` seconds (it's
    killed before the exception propagates).
    """
    job.append(f"$ {' '.join(cmd)}")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    assert proc.stdout is not None

    async def _drain() -> None:
        try:
            async for raw in proc.stdout:  # type: ignore[union-attr]
                job.append(raw.decode("utf-8", errors="replace").rstrip("\n"))
        except Exception as exc:  # noqa: BLE001
            job.append(f"[stream error: {exc}]")

    drain_task = asyncio.create_task(_drain())

    try:
        rc = await asyncio.wait_for(proc.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        job.append(f"[timeout after {timeout:.0f}s — terminating]")
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
        await drain_task
        raise

    await drain_task
    return rc


# ---------------------------------------------------------------------------
# Download / extract / verify
# ---------------------------------------------------------------------------


async def download_file(
    url: str,
    dest: Path,
    *,
    job: Optional[ToolJob] = None,
    timeout: float = 300.0,
) -> Path:
    """Stream ``url`` to ``dest`` (creating parent dirs). Returns ``dest``.

    Logs coarse progress into ``job`` when supplied. Raises on any
    non-2xx response.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if job is not None:
        job.append(f"Downloading {url}")

    written = 0
    next_log = 10 * 1024 * 1024  # log every ~10 MB
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with dest.open("wb") as fh:
                async for chunk in resp.aiter_bytes(1 << 20):
                    fh.write(chunk)
                    written += len(chunk)
                    if job is not None and written >= next_log:
                        job.append(f"  …{written // (1024 * 1024)} MB")
                        next_log += 10 * 1024 * 1024

    if job is not None:
        job.append(f"Downloaded {written // (1024 * 1024)} MB -> {dest}")
    return dest


def verify_sha256(file: Path, expected: str) -> None:
    """Raise :class:`ValueError` if ``file``'s SHA-256 != ``expected``."""
    h = hashlib.sha256()
    with file.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    actual = h.hexdigest()
    if actual.lower() != expected.strip().lower():
        raise ValueError(
            f"SHA-256 mismatch for {file.name}: expected {expected}, got {actual}"
        )


def extract_tarball(archive: Path, dest: Path, *, strip_top: bool = True) -> Path:
    """Extract ``archive`` into ``dest``.

    When ``strip_top`` is true (the default), the single top-level
    directory that prebuilt tarballs wrap their contents in (e.g.
    ``node-v22.14.0-darwin-arm64/``) is stripped so files land directly
    under ``dest``.  Returns ``dest``.
    """
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        members = tf.getmembers()
        if strip_top and members:
            top = members[0].name.split("/")[0] + "/"
            for m in members:
                if m.name.startswith(top):
                    m.name = m.name[len(top):]
                elif m.name == top.rstrip("/"):
                    m.name = "."
            members = [m for m in members if m.name not in ("", ".")]
        for m in members:
            tf.extract(m, dest)
    return dest


# ---------------------------------------------------------------------------
# uv installer (brew-free)
# ---------------------------------------------------------------------------


def _local_bin_dir() -> Path:
    """``~/.local/bin`` — where the official uv installer drops binaries.

    Matches the resolution order in :func:`backend.mcp_builder._find_uv`
    (and the registry copy) so a portable install is discoverable, and is
    placed on ``PATH`` by :func:`backend.server._ensure_sidecar_paths`.
    """
    return Path.home() / ".local" / "bin"


def _uv_arch() -> str:
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "aarch64"
    return "x86_64"


def _uv_tarball_name() -> str:
    return f"uv-{_uv_arch()}-apple-darwin.tar.gz"


def _uv_tarball_url() -> str:
    # ``latest`` mirrors the behaviour of the official install.sh script.
    return f"https://github.com/astral-sh/uv/releases/latest/download/{_uv_tarball_name()}"


async def _install_uv_portable(job: ToolJob) -> None:
    """Download the official prebuilt ``uv`` binary into ``~/.local/bin``.

    A brew-free, ``curl|sh``-free path: fetch the GitHub release tarball
    directly via :func:`download_file` (httpx) and copy the ``uv`` /
    ``uvx`` binaries out of it.  More robust inside the frozen app than
    shelling out to ``curl … | sh`` (which depends on an external
    network helper and a writable shell environment).
    """
    if platform.system() != "Darwin":
        raise RuntimeError(
            "Portable uv install is only implemented for macOS. "
            "Install uv from https://astral.sh/uv."
        )

    archive = tools_dir() / _uv_tarball_name()
    await download_file(_uv_tarball_url(), archive, job=job)

    extract_root = tools_dir() / "uv-extract"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    job.append(f"Extracting {archive.name}…")
    extract_tarball(archive, extract_root, strip_top=True)
    archive.unlink(missing_ok=True)

    bin_dir = _local_bin_dir()
    bin_dir.mkdir(parents=True, exist_ok=True)
    installed = False
    for tool in ("uv", "uvx"):
        src = extract_root / tool
        if not src.exists():
            continue
        dst = bin_dir / tool
        shutil.copy2(src, dst)
        dst.chmod(dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        installed = True
    shutil.rmtree(extract_root, ignore_errors=True)

    if not installed or not (bin_dir / "uv").exists():
        raise RuntimeError("uv binary missing after extraction")
    job.append(f"uv installed at {bin_dir / 'uv'}")


async def ainstall_uv(*, job: Optional[ToolJob] = None) -> ToolJob:
    """Install ``uv`` without requiring Homebrew.

    Path selection:

    1. Homebrew present -> ``brew install uv`` (fast path); on failure
       fall through.
    2. Official ``curl -LsSf https://astral.sh/uv/install.sh | sh``
       installer (drops a static binary into ``~/.local/bin``).
    3. Portable GitHub-release binary downloaded via httpx into
       ``~/.local/bin`` — used when the shell installer is unavailable
       or fails (e.g. no ``curl``/``bash`` on PATH inside the frozen
       app, or a sandbox that blocks the piped script).

    Caller polls :func:`get_job` for status.
    """
    # Imported lazily to avoid a circular import at module load time.
    from backend.omlx_provisioner import _brew_bin, has_homebrew

    job = job or new_job("install-uv")
    job.status = "running"

    async def _go() -> None:
        try:
            if has_homebrew():
                rc = await run_streaming(
                    [_brew_bin(), "install", "uv"], job=job, timeout=600.0
                )
                if rc == 0 and shutil.which("uv"):
                    job.status = "done"
                    job.finished_at = time.time()
                    return
                job.append("brew install uv failed — falling back to official installer")

            if shutil.which("curl") and shutil.which("bash"):
                try:
                    rc = await run_streaming(
                        ["bash", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"],
                        job=job,
                        timeout=600.0,
                    )
                    if rc == 0 and (_local_bin_dir() / "uv").exists():
                        job.status = "done"
                        job.finished_at = time.time()
                        return
                    job.append(
                        f"curl|sh uv installer did not yield a usable uv "
                        f"(exit {rc}) — falling back to portable binary"
                    )
                except Exception as exc:  # noqa: BLE001
                    job.append(f"[curl|sh uv installer error: {exc} — falling back]")

            await _install_uv_portable(job)
            job.result = {"bin_dir": str(_local_bin_dir())}
            job.status = "done"
            job.finished_at = time.time()
        except Exception as exc:  # noqa: BLE001
            job.error = str(exc)
            job.status = "error"
            job.finished_at = time.time()

    asyncio.create_task(_go())
    return job
