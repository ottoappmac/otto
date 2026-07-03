"""Install Node.js without requiring Homebrew.

The Playwright MCP that powers Otto's browser automation is launched via
``npx`` (see :func:`backend.mcp_manager._build_playwright_command`), which
needs a Node.js runtime on ``PATH``.  Node is *not* bundled in the
Otto.app, and Homebrew — the usual ``brew install node`` route — is a
third-party package manager that isn't present on every Mac and can't be
installed silently.

This module provides a brew-free install path: it downloads Node's
official prebuilt tarball into a user-writable directory under Otto's
app-data folder (no sudo, no Xcode Command Line Tools, no Homebrew) and
puts its ``bin/`` on ``PATH`` via
:func:`backend.server._ensure_sidecar_paths`.  When Homebrew *is*
present we still prefer ``brew install node`` as a faster, cleaner path.

The install runs as a pollable background :class:`ToolJob` so the setup
UI can show progress, mirroring the oMLX install flow.
"""

from __future__ import annotations

import asyncio
import platform
import shutil
import time
from pathlib import Path
from typing import Optional

from backend.tool_provisioner import (
    ToolJob,
    download_file,
    extract_tarball,
    new_job,
    run_streaming,
    tools_dir,
    verify_sha256,
)

# Pinned like the Playwright MCP version (mcp_manager pins
# ``@playwright/mcp@0.0.70``) so behaviour is reproducible across hosts.
NODE_VERSION = "v22.14.0"
_DIST_BASE = "https://nodejs.org/dist"


def node_home() -> Path:
    return tools_dir() / "node"


def node_bin_dir() -> Path:
    return node_home() / "bin"


def node_is_present() -> bool:
    """Whether a usable ``node`` + ``npx`` pair is available to the backend.

    Checks both ``node`` and ``npx`` on the system PATH, then the portable
    copy this module manages under app-data.  We require *both* because
    some packages (e.g. the ``playwright`` pip package) ship an internal
    ``node`` binary without a matching ``npx``, and the Playwright MCP
    launcher needs ``npx`` specifically.
    """
    system_node = shutil.which("node")
    system_npx = shutil.which("npx")
    if system_node and system_npx:
        # Sanity-check they live in the same directory — an internal
        # Playwright node won't have npx next to it, but this catches
        # any PATH where npx comes from a different install than node.
        if Path(system_node).parent == Path(system_npx).parent:
            return True
    candidate = node_bin_dir() / "node"
    return candidate.exists() and (node_bin_dir() / "npx").exists()


def _node_arch() -> str:
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "arm64"
    return "x64"


def _tarball_name() -> str:
    return f"node-{NODE_VERSION}-darwin-{_node_arch()}.tar.gz"


def _tarball_url() -> str:
    return f"{_DIST_BASE}/{NODE_VERSION}/{_tarball_name()}"


def _shasums_url() -> str:
    return f"{_DIST_BASE}/{NODE_VERSION}/SHASUMS256.txt"


async def _fetch_expected_sha(job: ToolJob) -> Optional[str]:
    """Return the published SHA-256 for our tarball, or ``None`` on failure.

    ``SHASUMS256.txt`` is a list of ``<sha>  <filename>`` lines; we pick
    the line for :func:`_tarball_name`.
    """
    sums = tools_dir() / "SHASUMS256.txt"
    try:
        await download_file(_shasums_url(), sums, job=job, timeout=60.0)
        name = _tarball_name()
        for line in sums.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == name:
                return parts[0]
    except Exception as exc:  # noqa: BLE001
        job.append(f"[sha lookup failed: {exc}]")
    finally:
        sums.unlink(missing_ok=True)
    return None


async def _install_portable(job: ToolJob) -> None:
    """Download + extract the official Node tarball into ``node_home()``."""
    if platform.system() != "Darwin":
        raise RuntimeError(
            "Portable Node install is only implemented for macOS. "
            "Install Node.js manually from https://nodejs.org/."
        )

    archive = tools_dir() / _tarball_name()
    expected_sha = await _fetch_expected_sha(job)

    await download_file(_tarball_url(), archive, job=job)

    if expected_sha:
        job.append("Verifying SHA-256…")
        verify_sha256(archive, expected_sha)
    else:
        job.append("[warning: could not verify checksum — proceeding]")

    home = node_home()
    if home.exists():
        shutil.rmtree(home)
    job.append(f"Extracting to {home}…")
    extract_tarball(archive, home, strip_top=True)
    archive.unlink(missing_ok=True)

    if not (node_bin_dir() / "node").exists():
        raise RuntimeError("node binary missing after extraction")
    job.append(f"Node installed at {node_bin_dir()}")


async def ainstall_node(*, job: Optional[ToolJob] = None, force: bool = False) -> ToolJob:
    """Install Node.js in a background job. Returns the job immediately.

    Path selection:

    1. Already present (system PATH or app-data) -> no-op, ``done``.
       Pass ``force=True`` to skip the presence check and (re-)install.
    2. Homebrew present -> ``brew install node`` (fast path); on failure
       fall through to the portable tarball.
    3. Otherwise -> download Node's official prebuilt tarball into
       app-data (no admin required).

    Caller polls :func:`backend.tool_provisioner.get_job` for status.
    """
    from backend.omlx_provisioner import _brew_bin, has_homebrew

    job = job or new_job("install-node")
    job.status = "running"

    async def _go() -> None:
        try:
            if not force and node_is_present():
                job.append("node already available — nothing to do")
                job.result = {"bin_dir": str(node_bin_dir())}
                job.status = "done"
                job.finished_at = time.time()
                return

            if has_homebrew():
                rc = await run_streaming(
                    [_brew_bin(), "install", "node"], job=job, timeout=600.0
                )
                if rc == 0 and shutil.which("node"):
                    job.append("Installed Node via Homebrew")
                    job.status = "done"
                    job.finished_at = time.time()
                    return
                job.append(
                    "brew install node did not yield a usable node — "
                    "falling back to portable tarball"
                )

            await _install_portable(job)
            job.result = {"bin_dir": str(node_bin_dir())}
            job.status = "done"
            job.finished_at = time.time()
        except Exception as exc:  # noqa: BLE001
            job.error = str(exc)
            job.status = "error"
            job.finished_at = time.time()

    asyncio.create_task(_go())
    return job


def status_snapshot() -> dict:
    """Detection snapshot consumed by ``GET /api/node/status``."""
    return {
        "present": node_is_present(),
        "system_node": shutil.which("node"),
        "app_data_bin": str(node_bin_dir()),
        "app_data_node_exists": (node_bin_dir() / "node").exists(),
        "pinned_version": NODE_VERSION,
    }
