"""Detect, install, and run the local ``oMLX`` inference server.

oMLX (`<https://github.com/jundot/omlx>`_) is an external macOS process —
this module is the analogue of :mod:`backend.exo_provisioner` for it.
We deliberately do *not* bundle oMLX in the Otto.app: it ships its own
``.dmg`` and Homebrew formula, and embedding a Python-based server inside
our PyInstaller bundle would double both the download size and the
notarisation matrix.  Instead, this module:

* **Detects** existing oMLX installs (``omlx`` on ``$PATH`` or the
  ``/Applications/oMLX.app`` bundle, plus a live HTTP probe of the
  configured port).
* **Installs** via Homebrew when the user opts in:

    .. code-block:: bash

        brew tap jundot/omlx https://github.com/jundot/omlx
        brew install omlx

* **Manages lifecycle** preferring ``brew services start omlx`` (a
  ``launchd`` job that keep-alives across reboots) and falling back to
  spawning ``omlx serve --port <port>`` directly when Homebrew isn't
  managing the service.
* Provides a **status snapshot** the UI / agent can poll: detection
  flags, brew service state, and HTTP reachability of ``/v1/models``.

The provisioner never *requires* Homebrew — manual installs (drag the
``.dmg`` into ``Applications``, or build from source) are detected and
honoured.  The "install for me" button is purely an opt-in convenience.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import secrets
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx

from backend.config import OmlxConfig

# Path to oMLX's own settings file. We need to read/write it directly
# (not just via the admin API) because the admin API requires us to know
# the current API key, but we manage the lifecycle of that key ourselves.
_OMLX_SETTINGS_PATH = Path.home() / ".omlx" / "settings.json"

# oMLX writes its own server-side log here (distinct from Otto's
# backend.log). This is the authoritative record of *why* a generation
# failed: when inference fails after the response headers are sent, oMLX
# returns HTTP 200 and then drops the connection, so Otto's HTTP client
# only sees an opaque "peer closed connection / incomplete chunked read".
# The real cause (Metal stream error, OOM, etc.) lands here instead.
_OMLX_SERVER_LOG_PATH = Path.home() / ".omlx" / "logs" / "server.log"

logger = logging.getLogger(__name__)


def read_recent_omlx_error(
    *, within_seconds: float = 60.0, max_scan_bytes: int = 65536,
) -> Optional[str]:
    """Return oMLX's most recent server-side ERROR line, if it is recent.

    Reads the tail of ``~/.omlx/logs/server.log`` (cheap, bounded to
    *max_scan_bytes*) and returns the message of the last ``ERROR`` line
    whose embedded timestamp is within *within_seconds* of now. Returns
    ``None`` when the log is missing, has no recent error, or can't be
    parsed — callers must treat the result as best-effort.
    """
    try:
        size = _OMLX_SERVER_LOG_PATH.stat().st_size
        with _OMLX_SERVER_LOG_PATH.open("rb") as fh:
            if size > max_scan_bytes:
                fh.seek(-max_scan_bytes, os.SEEK_END)
            tail = fh.read().decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return None

    now = time.time()
    for line in reversed(tail.splitlines()):
        if " - ERROR - " not in line:
            continue
        # oMLX log line: "2026-06-08 09:52:28,508 - omlx.scheduler - ERROR - [-] - <msg>"
        ts_raw = line[:23]
        try:
            import datetime as _dt

            ts = _dt.datetime.strptime(ts_raw, "%Y-%m-%d %H:%M:%S,%f").timestamp()
            if now - ts > within_seconds:
                return None  # newest error is stale → not relevant to this request
        except (ValueError, OverflowError):
            pass  # unparseable timestamp → fall through and return the message
        # Return the human message after the final " - " separator.
        return line.rsplit(" - ", 1)[-1].strip()
    return None


def diagnose_omlx_stream_drop() -> tuple[str, str]:
    """Return an accurate ``(user_message, error_code)`` for an oMLX disconnect.

    Otto only observes "peer closed connection / incomplete chunked read"
    when oMLX fails *after* sending response headers. Consult oMLX's own
    server log to recover the real cause instead of guessing, which keeps
    the user from chasing the wrong remedy (e.g. lowering the context
    window when the actual fault is a Metal-stream bug).
    """
    err = read_recent_omlx_error()

    if err and "no stream(gpu" in err.lower():
        # "There is no Stream(gpu, N) in current thread" is a Metal-stream
        # thread-isolation error that has appeared in two distinct forms:
        #
        # • oMLX ≤ 0.3.x (PR #891 era): VLM prefill ran on a worker thread
        #   without an active Metal stream. Fixed in oMLX 0.3.x.
        #
        # • oMLX 0.4.0–0.4.1 regression (issue #1685): the per-engine
        #   executor split (#1248) caused Qwen3-VL's RoPE `inv_freq` (stored
        #   as a lazy mx.array on a plain Python class, not an nn.Module) to
        #   remain pinned to the construction thread's stream. Fixed on the
        #   mlx-vlm side by calling mx.eval(inv_freq) at construction;
        #   not yet merged into a stable oMLX release as of 0.4.1.
        #
        # In both cases `brew upgrade omlx` and keeping mlx-vlm up-to-date
        # is the primary remedy.
        return (
            "The local oMLX server crashed mid-generation. Its log shows "
            "\"There is no Stream(gpu, …) in current thread\" — a known "
            "Metal-thread isolation error that affects vision-language (VL) "
            "models. This has appeared across several oMLX versions. "
            "To fix: run `brew upgrade omlx` to get the latest stable build. "
            "If the error persists on oMLX 0.4.x with a Qwen3-VL model, "
            "the root cause is a lazy mx.array in mlx-vlm's "
            "Qwen3VLRotaryEmbedding not being materialised before the engine "
            "thread runs (issue #1685). Switching to a text-only model "
            "sidesteps the issue until a fix lands in mlx-vlm.",
            "llm_omlx_stream_bug",
        )

    if err and any(k in err.lower() for k in ("out of memory", "insufficient memory", "metal error")):
        return (
            "The local oMLX server ran out of memory mid-generation "
            f"(oMLX reported: {err}). Try a smaller / more heavily quantized "
            "model, reduce the context window in the oMLX cache settings, or "
            "close other GPU-heavy apps, then retry.",
            "llm_omlx_oom",
        )

    if err:
        return (
            "The local oMLX server dropped the connection mid-generation. "
            f"oMLX's log reports: {err}. Restart oMLX from Settings → LLM → "
            "oMLX and retry; if it persists, check the full oMLX server log "
            "(~/.omlx/logs/server.log).",
            "llm_connection",
        )

    # No recent server-side error found — fall back to a neutral message
    # rather than asserting a specific (possibly wrong) cause.
    return (
        "The local oMLX server dropped the streaming connection "
        "mid-response. Restart oMLX from Settings → LLM → oMLX and retry. "
        "If it keeps happening, check the oMLX server log "
        "(~/.omlx/logs/server.log) for the underlying error.",
        "llm_connection",
    )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


# Common locations the ``omlx`` CLI can land in.  PATH is checked first
# (covers Homebrew on Apple Silicon + Intel + custom shells), with these
# absolute paths as a fallback for non-interactive environments where the
# backend's ``$PATH`` doesn't include Homebrew's bin dir (e.g. when Otto
# is launched directly by Finder / launchd).
#
# The omlx formula is a Python-based Homebrew formula that installs
# everything into a libexec virtualenv — the real binary lives at
# /opt/homebrew/opt/omlx/libexec/bin/omlx rather than the conventional
# /opt/homebrew/bin/omlx shim. Both paths are listed here.
_CANDIDATE_BIN_PATHS = (
    "/opt/homebrew/bin/omlx",                       # standard Homebrew shim
    "/usr/local/bin/omlx",                          # Intel Homebrew / manual symlink
    "/opt/homebrew/opt/omlx/libexec/bin/omlx",      # Python venv inside libexec (actual binary)
    "/opt/homebrew/opt/omlx/bin/omlx",              # conventional opt/bin (future-proofing)
    "/usr/local/opt/omlx/libexec/bin/omlx",         # Intel Homebrew libexec
    "/usr/local/opt/omlx/bin/omlx",                 # Intel Homebrew bin
)

_APP_BUNDLE_PATH = Path("/Applications/oMLX.app")

# The oMLX GUI app bundles a fully self-contained CLI launcher under
# ``Contents/MacOS/``: ``omlx-cli`` is a shell wrapper that points
# ``PYTHONHOME`` at the bundled CPython and runs ``python -m omlx.cli``.
# Newer releases (>= 0.4.x) ship this even in the ``.dmg`` GUI build, so the
# app bundle alone is enough to manage the server — no Homebrew required.
_APP_BUNDLE_CLI_NAMES = ("omlx-cli", "omlx")


def _app_bundle_cli() -> Optional[Path]:
    """Return the CLI launcher embedded in ``/Applications/oMLX.app`` if any."""
    macos_dir = _APP_BUNDLE_PATH / "Contents" / "MacOS"
    for name in _APP_BUNDLE_CLI_NAMES:
        cand = macos_dir / name
        if cand.is_file():
            return cand
    return None


def find_cli(explicit: str = "") -> Optional[Path]:
    """Locate the ``omlx`` CLI binary.

    ``explicit`` (from :class:`OmlxConfig.cli_path`) wins when set and
    points at a real file. Otherwise we try the system PATH first and
    fall back to known Homebrew locations.

    Returns ``None`` when no binary is found.
    """
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file():
            return p

    found = shutil.which("omlx")
    if found:
        return Path(found)

    for cand in _CANDIDATE_BIN_PATHS:
        p = Path(cand)
        if p.is_file():
            return p

    # The GUI app bundle ships a working CLI launcher (omlx-cli) — prefer it
    # over the Homebrew cellar scan so a brew-free .dmg install is usable.
    bundle_cli = _app_bundle_cli()
    if bundle_cli is not None:
        return bundle_cli

    # Last-resort: scan the Homebrew cellar for any installed version of omlx.
    # The formula puts the binary in libexec/bin/ (Python venv) rather than
    # the conventional bin/ directory, so a glob is the safest way to find it
    # regardless of which patch version is currently installed.
    for cellar_root in ("/opt/homebrew/Cellar/omlx", "/usr/local/Cellar/omlx"):
        cellar = Path(cellar_root)
        if cellar.is_dir():
            for candidate in sorted(cellar.glob("*/libexec/bin/omlx"), reverse=True):
                if candidate.is_file():
                    return candidate
            for candidate in sorted(cellar.glob("*/bin/omlx"), reverse=True):
                if candidate.is_file():
                    return candidate

    return None


def find_app_bundle() -> Optional[Path]:
    """Return the ``oMLX.app`` path when the GUI app is installed.

    Recent oMLX builds embed a self-contained CLI launcher inside the
    bundle (``Contents/MacOS/omlx-cli``), so the app's presence is enough
    for Otto to manage the server — see :func:`_app_bundle_cli` and
    :func:`find_cli`.
    """
    return _APP_BUNDLE_PATH if _APP_BUNDLE_PATH.exists() else None


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def has_homebrew() -> bool:
    """Whether ``brew`` is on PATH (or in the standard Apple Silicon
    location) and at least executes ``--version`` cleanly."""
    brew = _which("brew") or "/opt/homebrew/bin/brew"
    if not Path(brew).exists():
        return False
    try:
        proc = subprocess.run(
            [brew, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return proc.returncode == 0
    except Exception:  # noqa: BLE001
        return False


@dataclass
class Detection:
    cli_path: Optional[str]
    app_bundle: Optional[str]
    homebrew: bool
    brew_service_state: Optional[str]  # started | stopped | none | error | None when brew absent
    cli_version: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "cli_path": self.cli_path,
            "app_bundle": self.app_bundle,
            "homebrew": self.homebrew,
            "brew_service_state": self.brew_service_state,
            "cli_version": self.cli_version,
            "installed": bool(self.cli_path or self.app_bundle),
        }


def detect(cfg: OmlxConfig) -> Detection:
    """Snapshot the current install state without making any changes."""
    cli = find_cli(cfg.cli_path)
    bundle = find_app_bundle()
    brew = has_homebrew()

    cli_version: Optional[str] = None
    if cli is not None:
        try:
            proc = subprocess.run(
                [str(cli), "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if proc.returncode == 0:
                cli_version = (proc.stdout or proc.stderr or "").strip().splitlines()[0:1]
                cli_version = cli_version[0] if cli_version else None
        except Exception:  # noqa: BLE001
            cli_version = None

    svc_state: Optional[str] = None
    if brew:
        svc_state = _brew_service_state(cfg.brew_formula)

    return Detection(
        cli_path=str(cli) if cli else None,
        app_bundle=str(bundle) if bundle else None,
        homebrew=brew,
        brew_service_state=svc_state,
        cli_version=cli_version,
    )


# ---------------------------------------------------------------------------
# Brew helpers
# ---------------------------------------------------------------------------


def _brew_bin() -> str:
    return _which("brew") or "/opt/homebrew/bin/brew"


def _brew_service_state(formula: str) -> Optional[str]:
    """Return ``brew services info`` state for ``formula``.

    Returns ``"started"``, ``"stopped"``, ``"none"`` (formula installed
    but no service block), ``"error"``, or ``None`` when brew is missing.
    """
    if not has_homebrew():
        return None
    try:
        proc = subprocess.run(
            [_brew_bin(), "services", "info", formula, "--json"],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except Exception:  # noqa: BLE001
        return "error"

    if proc.returncode != 0:
        # ``brew services info <formula>`` returns non-zero when the
        # formula isn't installed (either as a known formula or via tap).
        # Distinguish "expected absent" from real failures by checking
        # stderr for the familiar phrasing.  Both pre-tap ("No available
        # formula") and post-tap-but-unbuilt ("not installed") collapse
        # to ``"none"`` — the UI treats them the same way.
        msg = (proc.stderr or "").lower()
        if "no available formula" in msg or "not installed" in msg:
            return "none"
        return "error"

    try:
        data = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return "error"

    if isinstance(data, list) and data:
        entry = data[0]
        status = (entry.get("status") or "").lower()
        if status in ("started", "running"):
            return "started"
        if status in ("stopped", "none"):
            return "stopped"
        return status or "none"

    return "none"


# ---------------------------------------------------------------------------
# Job tracking (mirrors exo_provisioner.ExoJob, simpler)
# ---------------------------------------------------------------------------


@dataclass
class OmlxJob:
    """Background install / start / stop task with a tail of log lines."""

    id: str
    kind: str  # "install" | "upgrade" | "start" | "stop" | "uninstall"
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


_jobs: dict[str, OmlxJob] = {}
_JOB_RETENTION_S = 60 * 60


def _new_job(kind: str) -> OmlxJob:
    cutoff = time.time() - _JOB_RETENTION_S
    for jid in list(_jobs.keys()):
        j = _jobs[jid]
        if j.finished_at and j.finished_at < cutoff:
            _jobs.pop(jid, None)
    job = OmlxJob(id=uuid.uuid4().hex, kind=kind)
    _jobs[job.id] = job
    return job


def get_job(job_id: str) -> Optional[OmlxJob]:
    return _jobs.get(job_id)


def list_jobs() -> list[OmlxJob]:
    return sorted(_jobs.values(), key=lambda j: j.started_at, reverse=True)


# ---------------------------------------------------------------------------
# Subprocess streaming helper
# ---------------------------------------------------------------------------


async def _run_streaming(
    cmd: list[str],
    *,
    job: OmlxJob,
    timeout: float = 600.0,
) -> int:
    """Run ``cmd`` and stream stdout/stderr lines into ``job.append``.

    Returns the process exit code. Raises ``TimeoutError`` if the
    process doesn't finish within ``timeout`` seconds (it's killed
    before the exception propagates).
    """
    job.append(f"$ {' '.join(cmd)}")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout is not None

    async def _drain() -> None:
        try:
            async for raw in proc.stdout:  # type: ignore[union-attr]
                line = raw.decode("utf-8", errors="replace").rstrip("\n")
                job.append(line)
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
# Install / uninstall (Homebrew-driven)
# ---------------------------------------------------------------------------


async def _install_via_brew(job: OmlxJob, cfg: OmlxConfig) -> None:
    """Install oMLX through Homebrew (fast path). Raises on failure."""
    brew = _brew_bin()

    tap_args = [brew, "tap", cfg.brew_tap]
    if cfg.brew_tap_url:
        tap_args.append(cfg.brew_tap_url)
    rc = await _run_streaming(tap_args, job=job, timeout=180.0)
    # ``brew tap`` returns 0 on a fresh tap *and* on "already
    # tapped" — non-zero is a real failure.
    if rc != 0:
        raise RuntimeError(f"brew tap failed (exit {rc})")

    install_args = [brew, "install", cfg.brew_formula]
    rc = await _run_streaming(install_args, job=job, timeout=600.0)
    if rc != 0:
        raise RuntimeError(
            f"brew install {cfg.brew_formula} failed (exit {rc}). "
            "If this is a dylib relink error, the latest tap "
            "usually fixes it — try ``brew untap --force "
            f"{cfg.brew_tap} && brew tap {cfg.brew_tap} "
            f"{cfg.brew_tap_url} && brew reinstall "
            f"{cfg.brew_formula}``."
        )

    det = detect(cfg)
    if det.cli_path is None:
        # brew install exits 0 when the formula is already installed
        # but not linked ("it's just not linked").  Attempt an
        # automatic ``brew link --overwrite`` to fix that before
        # surfacing an error to the user.
        not_linked = any(
            "just not linked" in line or "already installed" in line
            for line in job.log_lines
        )
        if not_linked:
            job.append(
                "Formula is installed but not linked — running "
                f"brew link --overwrite {cfg.brew_formula} …"
            )
            link_rc = await _run_streaming(
                [brew, "link", "--overwrite", cfg.brew_formula],
                job=job,
                timeout=30.0,
            )
            if link_rc == 0:
                det = detect(cfg)

    if det.cli_path is None:
        raise RuntimeError(
            "brew install reported success but the omlx CLI is "
            "still not on PATH. This usually means /opt/homebrew/bin "
            "isn't in $PATH for the backend process. Restart Otto "
            "or set OMLX_CLI_PATH explicitly."
        )

    job.result = det.to_dict()


async def aget_latest_release_assets() -> tuple[str | None, list[dict]]:
    """Return ``(tag, assets)`` for the latest oMLX GitHub release.

    ``assets`` is the raw GitHub asset list (each item has ``name`` and
    ``browser_download_url``).  Returns ``(None, [])`` on any failure.
    """
    def _fetch() -> tuple[str | None, list[dict]]:
        try:
            import urllib.request
            import json as _json

            url = "https://api.github.com/repos/jundot/omlx/releases/latest"
            req = urllib.request.Request(
                url,
                headers={"Accept": "application/vnd.github+json", "User-Agent": "Otto/1.0"},
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = _json.loads(resp.read())
            return data.get("tag_name"), list(data.get("assets") or [])
        except Exception as exc:  # noqa: BLE001
            logger.debug("aget_latest_release_assets: fetch failed: %s", exc)
            return None, []

    return await asyncio.to_thread(_fetch)


def _pick_release_asset(assets: list[dict]) -> Optional[dict]:
    """Choose the best macOS asset for a brew-free install.

    Preference: an arch-matched archive carrying the CLI (``.tar.gz`` /
    ``.tgz`` / ``.zip``) over a ``.dmg`` GUI bundle (which ships no CLI).
    """
    machine = platform.machine().lower()
    arch_aliases = ("arm64", "aarch64") if machine in ("arm64", "aarch64") else ("x64", "x86_64", "amd64", "intel")

    def matches_arch(name: str) -> bool:
        low = name.lower()
        return any(a in low for a in arch_aliases)

    archives = [a for a in assets if a.get("name", "").lower().endswith((".tar.gz", ".tgz", ".zip"))]
    dmgs = [a for a in assets if a.get("name", "").lower().endswith(".dmg")]

    for group in (archives, dmgs):
        arch_matched = [a for a in group if matches_arch(a.get("name", ""))]
        if arch_matched:
            return arch_matched[0]
        if group:
            return group[0]
    return None


async def _install_via_release(job: OmlxJob, cfg: OmlxConfig) -> None:
    """Install oMLX from its official GitHub release (Homebrew-free).

    Downloads the best macOS asset into Otto's app-data ``tools`` dir.
    For a ``.dmg`` it mounts the image, copies ``oMLX.app`` into
    ``/Applications``, and detaches.  For an archive it extracts and
    symlinks the ``omlx`` CLI into ``~/.local/bin`` (already on the
    backend's augmented PATH).  Raises on failure.
    """
    from backend.tool_provisioner import (
        download_file,
        extract_tarball,
        tools_dir,
    )

    tag, assets = await aget_latest_release_assets()
    asset = _pick_release_asset(assets)
    if asset is None:
        raise RuntimeError(
            "No suitable oMLX release asset found for this Mac. Download it "
            "manually from https://github.com/jundot/omlx/releases."
        )

    name = asset["name"]
    url = asset["browser_download_url"]
    job.append(f"Installing oMLX {tag or ''} from release asset {name}")

    dest_dir = tools_dir() / "omlx"
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir / name
    await download_file(url, archive, job=job)

    low = name.lower()
    try:
        if low.endswith(".dmg"):
            await _install_dmg(job, archive)
        elif low.endswith((".tar.gz", ".tgz")):
            extract_tarball(archive, dest_dir, strip_top=True)
            _link_omlx_cli(job, dest_dir)
        elif low.endswith(".zip"):
            import zipfile
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(dest_dir)
            _link_omlx_cli(job, dest_dir)
    finally:
        # Always remove the downloaded installer — it can be ~700 MB and is
        # never needed after extraction/mount. This used to run only on the
        # success path, so a mid-install failure (or a later retry) orphaned the
        # archive, silently filling small disks (e.g. clean-room test VMs).
        archive.unlink(missing_ok=True)
        # The .dmg path copies oMLX.app into /Applications and keeps nothing in
        # dest_dir, so drop the now-empty staging dir as well. (Archive installs
        # keep dest_dir — the extracted CLI lives there and is symlinked.)
        if low.endswith(".dmg"):
            shutil.rmtree(dest_dir, ignore_errors=True)

    det = detect(cfg)
    if det.cli_path is None and det.app_bundle is None:
        raise RuntimeError(
            "oMLX release was downloaded but neither the CLI nor the app "
            "bundle could be detected afterward. Install it manually from "
            "https://github.com/jundot/omlx/releases."
        )
    if det.cli_path is None:
        job.append(
            "Installed the oMLX app bundle, but it does not ship the CLI "
            "Otto needs to manage the server. You may need the CLI build."
        )
    job.result = det.to_dict()


async def _install_dmg(job: OmlxJob, dmg: Path) -> None:
    """Mount ``dmg``, copy ``oMLX.app`` into /Applications, detach.

    Uses a per-job mountpoint and force-detaches any stale mount left over
    from a previous/concurrent attempt so a busy mountpoint can't make
    ``hdiutil attach`` fail. ``-noverify`` skips the (slow, multi-hundred-MB)
    checksum pass which otherwise risks blowing the timeout in a VM.
    """
    # Unique mountpoint per job so concurrent/retried installs never collide
    # on a shared ``mnt`` dir (the prior cause of "hdiutil attach failed").
    mount_point = dmg.parent / f"mnt-{job.id}"
    # Force-detach any stale mount and clear the dir before attaching.
    await _run_streaming(
        ["hdiutil", "detach", "-force", str(mount_point)], job=job, timeout=30.0
    )
    if mount_point.exists():
        shutil.rmtree(mount_point, ignore_errors=True)
    mount_point.mkdir(parents=True, exist_ok=True)

    rc = await _run_streaming(
        [
            "hdiutil", "attach", str(dmg),
            "-nobrowse", "-noautoopen", "-noverify",
            "-mountpoint", str(mount_point),
        ],
        job=job,
        timeout=300.0,
    )
    if rc != 0:
        raise RuntimeError("hdiutil attach failed for the oMLX disk image")
    try:
        app_src = next(mount_point.glob("*.app"), None)
        if app_src is None:
            raise RuntimeError("No .app found inside the oMLX disk image")
        app_dst = Path("/Applications") / app_src.name
        if app_dst.exists():
            shutil.rmtree(app_dst)
        job.append(f"Copying {app_src.name} -> {app_dst}")
        shutil.copytree(app_src, app_dst, symlinks=True)
    finally:
        await _run_streaming(
            ["hdiutil", "detach", "-force", str(mount_point)], job=job, timeout=60.0
        )
        shutil.rmtree(mount_point, ignore_errors=True)

    # The .dmg ships the CLI *inside* the bundle; expose it on PATH so
    # ``shutil.which("omlx")`` and PATH-based callers find it too.
    _link_bundled_cli(job)


def _link_bundled_cli(job: OmlxJob) -> None:
    """Symlink the app bundle's embedded CLI launcher into ~/.local/bin."""
    cli = _app_bundle_cli()
    if cli is None:
        return
    try:
        local_bin = Path.home() / ".local" / "bin"
        local_bin.mkdir(parents=True, exist_ok=True)
        dst = local_bin / "omlx"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(cli)
        job.append(f"Linked bundled oMLX CLI ({cli.name}) -> {dst}")
    except Exception as exc:  # noqa: BLE001
        job.append(f"[warning: could not link bundled oMLX CLI: {exc}]")


def _link_omlx_cli(job: OmlxJob, root: Path) -> None:
    """Symlink an extracted ``omlx`` binary into ~/.local/bin."""
    cli = next((p for p in root.rglob("omlx") if p.is_file()), None)
    if cli is None:
        job.append("[warning: no omlx CLI binary found in the extracted archive]")
        return
    cli.chmod(0o755)
    local_bin = Path.home() / ".local" / "bin"
    local_bin.mkdir(parents=True, exist_ok=True)
    dst = local_bin / "omlx"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(cli)
    job.append(f"Linked omlx CLI -> {dst}")


async def ainstall(cfg: OmlxConfig) -> OmlxJob:
    """Install oMLX in a background job.

    Prefers ``brew tap`` + ``brew install`` when Homebrew is present
    (fast, clean, and gives ``brew services`` auto-restart).  When
    Homebrew is absent, falls back to downloading the official GitHub
    release asset (no admin / no Homebrew required).

    Caller polls :func:`get_job` for status.
    """
    # Single-flight: if an install is already in progress, return it instead
    # of starting a second job. Concurrent installs raced on a shared
    # download dir + mountpoint and caused "hdiutil attach failed".
    for existing in _jobs.values():
        if existing.kind == "install" and existing.status == "running":
            return existing

    job = _new_job("install")
    job.status = "running"

    async def _go() -> None:
        try:
            if has_homebrew():
                await _install_via_brew(job, cfg)
            else:
                job.append(
                    "Homebrew not found — installing oMLX from its official "
                    "GitHub release instead."
                )
                await _install_via_release(job, cfg)

            job.status = "done"
            job.finished_at = time.time()
        except Exception as exc:  # noqa: BLE001
            job.error = str(exc)
            job.status = "error"
            job.finished_at = time.time()
            logger.warning("oMLX install job failed: %s", exc)

    asyncio.create_task(_go())
    return job


async def aupgrade(cfg: OmlxConfig) -> OmlxJob:
    """Run ``brew upgrade omlx`` in a background job.

    Stops the brew service first (if running) so the running binary is
    not replaced mid-request.  After the upgrade completes the service
    is restarted so the new version takes over immediately.

    Caller polls :func:`get_job` for status.
    """
    job = _new_job("upgrade")
    job.status = "running"

    async def _go() -> None:
        try:
            if not has_homebrew():
                raise RuntimeError(
                    "Homebrew is not installed — cannot upgrade via brew."
                )

            brew = _brew_bin()

            # Stop the service so we don't replace a running binary.
            svc_state = _brew_service_state(cfg.brew_formula)
            was_running = svc_state == "started"
            if was_running:
                job.append("Stopping oMLX service before upgrade…")
                await _run_streaming(
                    [brew, "services", "stop", cfg.brew_formula],
                    job=job, timeout=30.0,
                )

            job.append(f"Running: brew upgrade {cfg.brew_formula}")
            rc = await _run_streaming(
                [brew, "upgrade", cfg.brew_formula],
                job=job, timeout=600.0,
            )
            if rc != 0:
                raise RuntimeError(
                    f"brew upgrade {cfg.brew_formula} failed (exit {rc}). "
                    "Check the log above for details."
                )

            # Restart if it was running before.
            if was_running:
                job.append("Restarting oMLX service with new version…")
                await _run_streaming(
                    [brew, "services", "start", cfg.brew_formula],
                    job=job, timeout=30.0,
                )

            # Invalidate the cached latest-version so the next check re-fetches.
            global _latest_version_cache
            _latest_version_cache = None

            det = detect(cfg)
            job.result = {**det.to_dict(), "upgraded": True}
            job.status = "done"
            job.finished_at = time.time()
        except Exception as exc:  # noqa: BLE001
            job.error = str(exc)
            job.status = "error"
            job.finished_at = time.time()
            logger.warning("oMLX upgrade job failed: %s", exc)

    asyncio.create_task(_go())
    return job


# Latest-release cache: (version_string, fetched_at_timestamp)
_latest_version_cache: tuple[str, float] | None = None
_LATEST_VERSION_TTL = 3600.0  # 1 hour


async def aget_latest_release_version() -> str | None:
    """Return the latest stable oMLX release tag from GitHub.

    Caches the result for one hour so repeated Settings page renders
    don't hammer the GitHub API.  Returns ``None`` on any network or
    parse failure.
    """
    global _latest_version_cache

    now = asyncio.get_event_loop().time()
    if _latest_version_cache is not None:
        cached_ver, fetched_at = _latest_version_cache
        if now - fetched_at < _LATEST_VERSION_TTL:
            return cached_ver

    def _fetch() -> str | None:
        try:
            import urllib.request
            import json as _json

            url = "https://api.github.com/repos/jundot/omlx/releases/latest"
            req = urllib.request.Request(
                url,
                headers={"Accept": "application/vnd.github+json", "User-Agent": "Otto/1.0"},
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = _json.loads(resp.read())
            tag: str = data.get("tag_name", "")
            # Strip leading "v" — stored as "0.4.2" not "v0.4.2"
            return tag.lstrip("v") if tag else None
        except Exception as exc:  # noqa: BLE001
            logger.debug("aget_latest_release_version: GitHub fetch failed: %s", exc)
            return None

    result = await asyncio.to_thread(_fetch)
    if result is not None:
        _latest_version_cache = (result, now)
    return result


async def auninstall(cfg: OmlxConfig) -> OmlxJob:
    """Run ``brew uninstall`` in a background job.

    Stops the brew service first if running so we don't leave an
    orphaned launchd job pointing at a removed bottle.
    """
    job = _new_job("uninstall")
    job.status = "running"

    async def _go() -> None:
        try:
            if not has_homebrew():
                raise RuntimeError("Homebrew is not installed.")

            brew = _brew_bin()

            # Best-effort stop — ignore failure (service may not be set up)
            try:
                await _run_streaming(
                    [brew, "services", "stop", cfg.brew_formula],
                    job=job,
                    timeout=30.0,
                )
            except Exception as exc:  # noqa: BLE001
                job.append(f"[brew services stop failed (ignored): {exc}]")

            rc = await _run_streaming(
                [brew, "uninstall", cfg.brew_formula],
                job=job,
                timeout=180.0,
            )
            if rc != 0:
                raise RuntimeError(f"brew uninstall failed (exit {rc})")

            job.result = detect(cfg).to_dict()
            job.status = "done"
            job.finished_at = time.time()
        except Exception as exc:  # noqa: BLE001
            job.error = str(exc)
            job.status = "error"
            job.finished_at = time.time()

    asyncio.create_task(_go())
    return job


# ---------------------------------------------------------------------------
# Lifecycle (start / stop / status)
# ---------------------------------------------------------------------------


async def afetch_status(cfg: OmlxConfig) -> dict:
    """Snapshot the running state of the local oMLX server.

    Always returns a dict (even on error) so the UI / agent can render
    a stable shape.

    ``reachable`` — whether ``/v1/models`` returned 200.
    ``models``    — ALL ids registered with oMLX (used by
                    :func:`_resolve_omlx_model_id` for ID matching).
    ``loaded_models`` — ids that are actually resident in GPU RAM, from
                        ``/admin/api/models`` (``loaded: true`` entries).
                        Falls back to ``models`` if the admin endpoint is
                        unavailable (older oMLX builds without admin API).
    """
    base = cfg.effective_base_url
    url = f"{base.rstrip('/')}/v1/models"

    out: dict[str, Any] = {
        "base_url": base,
        "reachable": False,
        "models": [],
        "loaded_models": [],
        "error": None,
    }
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.get(url)
            if r.status_code == 200:
                out["reachable"] = True
                payload = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                # OpenAI shape: {"data": [{"id": "..."}, ...]}
                all_models = []
                if isinstance(payload, dict):
                    data = payload.get("data") or []
                    for m in data:
                        if isinstance(m, dict) and m.get("id"):
                            all_models.append({"id": m["id"]})
                out["models"] = all_models

                # oMLX /v1/models lists *all registered* models regardless of
                # whether they are resident in GPU RAM. Use the admin API to
                # find what is actually loaded.
                try:
                    ar = await client.get(
                        f"{base.rstrip('/')}/admin/api/models",
                        timeout=3.0,
                    )
                    if ar.status_code == 200:
                        adata = ar.json()
                        if isinstance(adata, dict) and isinstance(adata.get("models"), list):
                            loaded = [
                                {"id": m["id"]}
                                for m in adata["models"]
                                if isinstance(m, dict) and m.get("id") and m.get("loaded")
                            ]
                            out["loaded_models"] = loaded
                        else:
                            out["loaded_models"] = list(all_models)
                    else:
                        out["loaded_models"] = list(all_models)
                except Exception:  # noqa: BLE001
                    out["loaded_models"] = list(all_models)
            else:
                out["error"] = f"HTTP {r.status_code}"
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)

    return out


async def astart(cfg: OmlxConfig, *, model_id: str = "") -> OmlxJob:
    """Start the local oMLX server.

    Once running, models are loaded dynamically via the HTTP admin/data
    APIs (see :func:`aload_model`) — we do **not** spawn the server with
    ``--model`` because that requires a restart for every model swap and
    blocks port binding behind weight loading. The optional ``model_id``
    parameter is accepted for backward compatibility but ignored.

    Strategy:

    1. If the server is already reachable → fast no-op.
    2. If brew is present and the service block is installed → prefer
       ``brew services start <formula>`` (launchd auto-restart on crash).
    3. Otherwise spawn ``omlx serve --port <port>`` as a detached child.

    After the server is up, we run :func:`aprovision_omlx_server` to
    set the admin API key, point ``model_dirs`` at the HF cache, and
    enable ``skip_api_key_verification`` so subsequent ``/v1`` calls
    don't need auth.
    """
    job = _new_job("start")
    job.status = "running"

    async def _go() -> None:
        try:
            status = await afetch_status(cfg)
            if status["reachable"]:
                job.append(
                    f"oMLX already reachable at {cfg.effective_base_url}; "
                    "checking provisioning…"
                )
                await aprovision_omlx_server(cfg, job=job)
                job.result = {"already_running": True, "status": status}
                job.status = "done"
                job.finished_at = time.time()
                return

            cli = find_cli(cfg.cli_path)
            if cli is None:
                raise RuntimeError(
                    "omlx CLI not found. Install via Homebrew "
                    f"(`brew install {cfg.brew_formula}`) or download the "
                    "macOS app from https://github.com/jundot/omlx/releases."
                )

            spawned_via: str = "spawn"
            if has_homebrew() and _brew_service_state(cfg.brew_formula) in (
                "stopped", "none",
            ):
                rc = await _run_streaming(
                    [_brew_bin(), "services", "start", cfg.brew_formula],
                    job=job,
                    timeout=30.0,
                )
                if rc != 0:
                    job.append(
                        "[brew services start failed; falling back to "
                        "direct ``omlx serve``]"
                    )
                else:
                    started = await _wait_until_reachable(cfg, timeout=30.0)
                    if started:
                        spawned_via = "brew_services"
                        await aprovision_omlx_server(cfg, job=job)
                        job.result = {"strategy": "brew_services", **started}
                        job.status = "done"
                        job.finished_at = time.time()
                        return

            log_path = _spawn_log_path()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [str(cli), "serve", "--port", str(cfg.api_port)]
            job.append(f"Spawning {' '.join(cmd)} (log: {log_path})")
            with log_path.open("ab", buffering=0) as fh:
                fh.write(
                    f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} === "
                    "starting omlx serve\n".encode()
                )
                proc = subprocess.Popen(  # noqa: S603 — we control argv, no shell
                    cmd,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    cwd=str(Path.home()),
                )

            _record_spawned_pid(proc.pid)
            job.append(f"Spawned pid={proc.pid}; waiting for /v1/models …")

            ok = await _wait_until_reachable(cfg, timeout=30.0)
            if not ok:
                raise RuntimeError(
                    "omlx serve was spawned but /v1/models did not become "
                    "reachable within 30s. Check the log via "
                    "/api/omlx/log."
                )
            await aprovision_omlx_server(cfg, job=job)
            job.result = {"strategy": spawned_via, "pid": proc.pid, **ok}
            job.status = "done"
            job.finished_at = time.time()
        except Exception as exc:  # noqa: BLE001
            job.error = str(exc)
            job.status = "error"
            job.finished_at = time.time()

    asyncio.create_task(_go())
    return job


async def astop(cfg: OmlxConfig) -> OmlxJob:
    """Stop the local oMLX server.

    Tries ``brew services stop`` first (if the service is running under
    launchd), then falls back to killing the pid we spawned directly.
    """
    job = _new_job("stop")
    job.status = "running"

    async def _go() -> None:
        try:
            stopped_via: Optional[str] = None

            if has_homebrew() and _brew_service_state(cfg.brew_formula) == "started":
                rc = await _run_streaming(
                    [_brew_bin(), "services", "stop", cfg.brew_formula],
                    job=job,
                    timeout=30.0,
                )
                if rc == 0:
                    stopped_via = "brew_services"

            spawned = _read_spawned_pid()
            if spawned and stopped_via is None:
                try:
                    os.kill(spawned, 15)
                    job.append(f"Sent SIGTERM to spawned pid {spawned}")
                    stopped_via = "spawn_kill"
                except ProcessLookupError:
                    job.append(f"Spawned pid {spawned} no longer exists")
                except PermissionError as exc:
                    job.append(f"[permission error killing pid {spawned}: {exc}]")

            if stopped_via is None:
                # Server might still be reachable if managed by the GUI app.
                status = await afetch_status(cfg)
                if status["reachable"]:
                    raise RuntimeError(
                        "oMLX is reachable but Otto isn't managing the "
                        "process. It may be running under the macOS GUI "
                        "app — quit it from the menu bar."
                    )
                job.append("Nothing to stop — oMLX wasn't running.")

            _clear_spawned_pid()
            job.result = {"stopped_via": stopped_via}
            job.status = "done"
            job.finished_at = time.time()
        except Exception as exc:  # noqa: BLE001
            job.error = str(exc)
            job.status = "error"
            job.finished_at = time.time()

    asyncio.create_task(_go())
    return job


async def aunload_model(cfg: OmlxConfig, model_id: str) -> OmlxJob:
    """Unload a model from the running oMLX server via HTTP.

    Calls ``DELETE /v1/models/<resolved_id>`` on the live server so the
    weights are evicted from GPU RAM.  The model remains on disk and can
    be reloaded at any time via :func:`aload_model`.

    Returns an :class:`OmlxJob` the caller can poll.
    """
    job = _new_job("unload")
    job.status = "running"

    async def _go() -> None:
        try:
            status = await afetch_status(cfg)
            if not status.get("reachable"):
                job.append("oMLX server is not reachable — nothing to unload.")
                job.result = {"skipped": True, "reason": "server_unreachable"}
                job.status = "done"
                job.finished_at = time.time()
                return

            resolved = await _resolve_omlx_model_id(cfg, model_id)
            if resolved is None:
                job.append(f"Model '{model_id}' not found in oMLX registry — nothing to unload.")
                job.result = {"skipped": True, "reason": "model_not_found"}
                job.status = "done"
                job.finished_at = time.time()
                return

            base = cfg.effective_base_url.rstrip("/")
            url = f"{base}/v1/models/{resolved}/unload"
            job.append(f"POST {url}")

            skipped = False
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.post(url)
                if r.status_code in (200, 204):
                    job.append(f"oMLX unload response: HTTP {r.status_code}")
                elif r.status_code == 404:
                    # Model not found at that path — treat as already unloaded.
                    job.append(f"Model '{resolved}' not found on oMLX server (already unloaded?).")
                    skipped = True
                elif r.status_code == 400:
                    # oMLX returns 400 when the model is registered but not
                    # currently loaded in GPU RAM — treat as a no-op.
                    body = r.text[:300]
                    if "not loaded" in body.lower():
                        job.append(
                            f"Model '{resolved}' is registered but not loaded in GPU RAM — nothing to unload."
                        )
                        skipped = True
                    else:
                        raise RuntimeError(
                            f"oMLX unload failed: HTTP {r.status_code} — {body}"
                        )
                else:
                    raise RuntimeError(
                        f"oMLX unload failed: HTTP {r.status_code} — "
                        f"{r.text[:300]}"
                    )

            job.result = {
                "strategy": "http_delete",
                "requested_id": model_id,
                "resolved_id": resolved,
                "skipped": skipped,
            }
            job.status = "done"
            job.finished_at = time.time()
        except Exception as exc:  # noqa: BLE001
            job.error = str(exc)
            job.status = "error"
            job.finished_at = time.time()
            logger.warning("oMLX unload job failed: %s", exc)

    asyncio.create_task(_go())
    return job


async def aload_model(cfg: OmlxConfig, model_id: str) -> OmlxJob:
    """Dynamically load a model into the running oMLX server via HTTP.

    Calls ``POST /v1/models/<resolved_id>/load`` on the live server. No
    server restart, no CLI invocation — typically completes in a few
    seconds for a cached model.

    The caller passes Otto's stored model name (often a HuggingFace repo
    id like ``mlx-community/Qwen3.5-9B-4bit``).  We resolve it to oMLX's
    short id (``Qwen3.5-9B-4bit``) before issuing the load.
    """
    job = _new_job("load")
    job.status = "running"

    async def _go() -> None:
        try:
            status = await afetch_status(cfg)
            if not status.get("reachable"):
                raise RuntimeError(
                    "oMLX server is not reachable — start it first via "
                    "the oMLX setup screen or Otto session start."
                )

            resolved = await _resolve_omlx_model_id(cfg, model_id)
            if resolved is None:
                # The model isn't in oMLX's registry. Try a one-shot
                # provisioning step (point model_dirs at the HF cache and
                # reload) so previously-downloaded HF models become
                # discoverable.
                job.append(
                    f"Model '{model_id}' not in oMLX registry — "
                    "provisioning HF cache and rescanning…"
                )
                await aprovision_omlx_server(cfg, job=job)
                # Give oMLX up to ~10 s to finish its async rescan before
                # giving up — the reload endpoint may return before the
                # directory walk completes.
                for _attempt in range(5):
                    resolved = await _resolve_omlx_model_id(cfg, model_id)
                    if resolved is not None:
                        break
                    await asyncio.sleep(2.0)
                if resolved is None:
                    raise RuntimeError(_diagnose_unregistered_model(model_id))

            # Already loaded?
            loaded_ids = [m["id"] for m in (status.get("models") or [])]
            if resolved in loaded_ids:
                job.append(f"Model '{resolved}' is already loaded.")
                job.result = {"already_loaded": True, "model_id": resolved}
                job.status = "done"
                job.finished_at = time.time()
                return

            base = cfg.effective_base_url.rstrip("/")
            url = f"{base}/v1/models/{resolved}/load"
            job.append(f"POST {url}")

            # Loading large models can take 30-90s on first load. Use a
            # generous timeout that still surfaces hangs reasonably fast.
            async with httpx.AsyncClient(timeout=600.0) as client:
                r = await client.post(url)
                if r.status_code != 200:
                    raise RuntimeError(
                        f"oMLX load failed: HTTP {r.status_code} — "
                        f"{r.text[:300]}"
                    )
                payload = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                job.append(f"oMLX response: {payload}")

            confirmed = await _wait_until_model_loaded(
                cfg, resolved, timeout=30.0,
            )
            if not confirmed:
                raise RuntimeError(
                    f"oMLX reported load success but '{resolved}' is not "
                    "in /v1/models. Try the oMLX setup screen for details."
                )
            job.result = {
                "strategy": "http_load",
                "requested_id": model_id,
                "resolved_id": resolved,
                **confirmed,
            }
            job.status = "done"
            job.finished_at = time.time()
        except Exception as exc:  # noqa: BLE001
            job.error = str(exc)
            job.status = "error"
            job.finished_at = time.time()

    asyncio.create_task(_go())
    return job


# ---------------------------------------------------------------------------
# HTTP-based provisioning + model id resolution
# ---------------------------------------------------------------------------


def _read_omlx_settings() -> dict:
    """Read oMLX's settings.json safely (returns {} if missing or invalid)."""
    if not _OMLX_SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(_OMLX_SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


async def _admin_login(
    cfg: OmlxConfig, api_key: str, *, client: httpx.AsyncClient,
) -> Optional[str]:
    """Log in to oMLX admin API; return the session cookie value or None."""
    base = cfg.effective_base_url.rstrip("/")
    try:
        r = await client.post(
            f"{base}/admin/api/login",
            json={"api_key": api_key, "remember": False},
        )
    except httpx.HTTPError:
        return None
    if r.status_code != 200:
        return None
    cookie = r.cookies.get("omlx_admin_session")
    return cookie


async def aprovision_omlx_server(
    cfg: OmlxConfig, *, job: Optional[OmlxJob] = None,
) -> None:
    """Make sure the running oMLX server is set up the way Otto needs it.

    Idempotent. Performs three steps:

    1. Ensure an admin API key is configured. If oMLX has none, generate
       one and call ``POST /admin/api/setup-api-key``. The key is also
       persisted into Otto's :class:`OmlxConfig.admin_api_key` field
       (caller is responsible for saving the config back).
    2. Set ``skip_api_key_verification`` to true so Otto's chat client
       can hit ``/v1/chat/completions`` without auth headers.
    3. Configure ``model_dirs`` to include ``~/.cache/huggingface/hub``
       so HuggingFace-cached MLX models are auto-discovered, then call
       ``POST /admin/api/reload`` to rediscover.

    Logs progress into *job* when supplied so the user sees what's
    happening in the setup screen.
    """
    def _log(msg: str) -> None:
        if job is not None:
            job.append(msg)
        logger.info("oMLX provision: %s", msg)

    settings = _read_omlx_settings()
    auth = settings.get("auth", {}) if isinstance(settings, dict) else {}
    server_api_key = (auth.get("api_key") or "").strip()
    server_api_key_set = bool(server_api_key)

    base = cfg.effective_base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: ensure admin api key.
        if not server_api_key_set:
            new_key = cfg.admin_api_key.strip() or _generate_admin_key()
            _log(f"Setting up oMLX admin API key (length={len(new_key)})")
            r = await client.post(
                f"{base}/admin/api/setup-api-key",
                json={"api_key": new_key, "api_key_confirm": new_key},
            )
            if r.status_code != 200:
                _log(f"setup-api-key returned HTTP {r.status_code}: {r.text[:200]}")
                # Continue regardless — the server may have been provisioned
                # via another channel (CLI, GUI app) and we'll try to login
                # with cfg.admin_api_key below.
            else:
                # Persist back into Otto's config so subsequent runs reuse it.
                cfg.admin_api_key = new_key
                _persist_admin_key_to_config(new_key)
        elif not cfg.admin_api_key.strip():
            # The server already has an admin key (e.g. set up by a previous
            # Otto run whose config was later reset, or via the oMLX CLI/GUI),
            # but Otto's own copy is missing. oMLX stores the key in plaintext
            # in settings.json, so adopt it instead of giving up — otherwise we
            # can never authenticate to /admin/api/* (cache stats, reload, etc.)
            # even though the server is fully reachable.
            _log("Adopting existing oMLX admin API key from settings.json")
            cfg.admin_api_key = server_api_key
            _persist_admin_key_to_config(server_api_key)

        api_key = (cfg.admin_api_key or "").strip()
        if not api_key:
            # No way to authenticate; skip the optional global-settings
            # tweaks. /v1/ may still work if skip_api_key_verification was
            # already true (default fresh install).
            _log("No admin API key available — skipping admin provisioning")
            return

        cookie = await _admin_login(cfg, api_key, client=client)
        if cookie is None:
            _log(
                "admin login failed — is the saved API key out of sync "
                "with oMLX's settings.json? Skipping admin provisioning."
            )
            return

        cookies = {"omlx_admin_session": cookie}

        # Step 2: skip_api_key_verification = true (only if not already).
        if not auth.get("skip_api_key_verification", False):
            _log("Enabling skip_api_key_verification on /v1/")
            r = await client.post(
                f"{base}/admin/api/global-settings",
                json={"skip_api_key_verification": True},
                cookies=cookies,
            )
            if r.status_code != 200:
                _log(
                    f"Failed to set skip_api_key_verification "
                    f"(HTTP {r.status_code}): {r.text[:200]}"
                )

        # Step 3: model_dirs → reload.
        # Resolve configured dirs (expand ~ and filter to dirs that exist).
        configured_dirs = [
            str(Path(d).expanduser())
            for d in (cfg.model_dirs or ["~/.cache/huggingface/hub"])
        ]
        existing_dirs = [d for d in configured_dirs if Path(d).is_dir()]
        if existing_dirs:
            current_dirs = (settings.get("model", {}) or {}).get("model_dirs", []) or []
            if set(existing_dirs) != set(current_dirs):
                _log(f"Setting model_dirs to {existing_dirs} and rescanning")
                r = await client.post(
                    f"{base}/admin/api/global-settings",
                    json={"model_dirs": existing_dirs},
                    cookies=cookies,
                )
                if r.status_code != 200:
                    _log(
                        f"Failed to update model_dirs "
                        f"(HTTP {r.status_code}): {r.text[:200]}"
                    )
            # Always trigger a reload so newly-downloaded models that
            # arrived after the last scan are discovered, even when
            # model_dirs was already correctly configured.
            _log("Triggering oMLX rescan to discover any newly downloaded models")
            r = await client.post(
                f"{base}/admin/api/reload", cookies=cookies,
            )
            if r.status_code == 200:
                _log(f"Reload: {r.json().get('message','ok')}")
            else:
                _log(f"Reload returned HTTP {r.status_code}")
        else:
            _log(
                f"None of the configured model_dirs exist "
                f"({configured_dirs}) — skipping model_dirs setup"
            )

        # Step 4: raise context window.
        # oMLX ships with sampling_max_context_window = 32768; conversations
        # quickly exceed that.  Otto sets it to cfg.max_context_window
        # (default 131072) to avoid "prompt too long" 400 errors.
        current_ctx = (settings.get("sampling", {}) or {}).get(
            "max_context_window", 32768
        )
        desired_ctx = int(cfg.max_context_window)
        if current_ctx < desired_ctx:
            _log(f"Raising max_context_window {current_ctx} → {desired_ctx}")
            r = await client.post(
                f"{base}/admin/api/global-settings",
                json={"sampling_max_context_window": desired_ctx},
                cookies=cookies,
            )
            if r.status_code != 200:
                _log(
                    f"Failed to set sampling_max_context_window "
                    f"(HTTP {r.status_code}): {r.text[:200]}"
                )


def _generate_admin_key() -> str:
    """Generate a fresh admin API key (32 url-safe chars)."""
    return f"otto-{secrets.token_urlsafe(24)}"


def _persist_admin_key_to_config(api_key: str) -> None:
    """Save the freshly generated admin key into Otto's config.json so it
    survives Otto restarts. Best-effort — failures are logged but not raised.
    """
    try:
        from backend.config import AppConfig
        cfg = AppConfig.load()
        if cfg.omlx.admin_api_key != api_key:
            cfg.omlx.admin_api_key = api_key
            cfg.save()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to persist oMLX admin key: %s", exc)


def adopt_existing_admin_key(cfg: OmlxConfig) -> bool:
    """Adopt oMLX's existing admin key when Otto's own copy is missing.

    oMLX stores its admin API key in plaintext in ``~/.omlx/settings.json``.
    When the server was started outside Otto's own :func:`astart` path — a
    prior Otto run, ``brew services``, or the macOS GUI app — Otto's copy of
    the key (:attr:`OmlxConfig.admin_api_key`) can be empty even though the
    server is fully provisioned and reachable. Without the key, every
    ``/admin/api/*`` call (cache settings, context window, cache clear) fails
    with a 400 ("No oMLX admin API key configured"). This copies the server's
    existing key into Otto's in-memory config and persists it, so those
    operations self-heal without a Stop→Start cycle.

    Idempotent and cheap (a local file read). Returns ``True`` when a key was
    adopted, ``False`` when Otto already had one or oMLX has none configured.
    """
    if (cfg.admin_api_key or "").strip():
        return False
    settings = _read_omlx_settings()
    auth = settings.get("auth", {}) if isinstance(settings, dict) else {}
    server_api_key = (auth.get("api_key") or "").strip()
    if not server_api_key:
        return False
    cfg.admin_api_key = server_api_key
    _persist_admin_key_to_config(server_api_key)
    logger.info("Adopted existing oMLX admin API key from settings.json")
    return True


def _find_hf_cache_snapshot(repo_id: str) -> Optional[Path]:
    """Locate the cached snapshot dir for a HuggingFace repo id, if present.

    Mirrors the ``models--org--name`` layout HuggingFace uses under the hub
    cache. Returns the newest snapshot directory or ``None`` when the repo
    isn't downloaded.
    """
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    folder = "models--" + repo_id.replace("/", "--")
    snaps_root = hub / folder / "snapshots"
    if not snaps_root.is_dir():
        return None
    snaps = [p for p in snaps_root.iterdir() if p.is_dir()]
    if not snaps:
        return None
    return max(snaps, key=lambda p: p.stat().st_mtime)


def _read_model_type(snapshot: Path) -> Optional[str]:
    """Read ``model_type`` from a model snapshot's config.json (or None)."""
    cfg_path = snapshot / "config.json"
    if not cfg_path.is_file():
        return None
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    mt = data.get("model_type")
    return str(mt) if mt else None


def _diagnose_unregistered_model(model_id: str) -> str:
    """Build an accurate error for a model oMLX won't expose in /v1/models.

    By the time this is called, Otto has already pointed oMLX's
    ``model_dirs`` at the HF cache and triggered a rescan, so the two
    realistic causes are:

    * Not downloaded → tell the user to download it.
    * Downloaded but oMLX's scan didn't register it → most often an
      incomplete download (missing config.json / safetensors) or, less
      commonly, an architecture this oMLX build can't load. We surface
      the model_type as a hint and point at oMLX's own log rather than
      asserting the architecture is unsupported (oMLX adds support over
      time, so a hardcoded list would go stale and mislead).
    """
    snapshot = _find_hf_cache_snapshot(model_id)
    if snapshot is None:
        return (
            f"Model '{model_id}' isn't downloaded yet (no snapshot found in "
            "the HuggingFace cache). Download it first from the oMLX setup "
            "screen's Discover/Custom tab, then try again."
        )

    has_weights = any(snapshot.glob("*.safetensors"))
    has_config = (snapshot / "config.json").is_file()
    model_type = _read_model_type(snapshot)
    type_hint = f" (model_type='{model_type}')" if model_type else ""

    if not has_config or not has_weights:
        missing = []
        if not has_config:
            missing.append("config.json")
        if not has_weights:
            missing.append("*.safetensors weights")
        return (
            f"Model '{model_id}' is in the HuggingFace cache but the download "
            f"looks incomplete — missing {', '.join(missing)}. Re-download it "
            "from the oMLX setup screen, then try again."
        )

    return (
        f"Model '{model_id}'{type_hint} is downloaded but oMLX did not "
        "register it even after a rescan. This usually means oMLX's scanner "
        "rejected the model — check oMLX's log "
        "($(brew --prefix)/var/log/omlx.log) for a 'Failed to discover "
        "model' line, and make sure your oMLX version supports this "
        "architecture (`brew upgrade omlx`)."
    )


async def _resolve_omlx_model_id(
    cfg: OmlxConfig, requested: str,
) -> Optional[str]:
    """Map Otto's stored model name to oMLX's actual model id.

    Otto stores HuggingFace repo ids (``mlx-community/Foo-4bit``) while
    oMLX exposes models using one of several id formats depending on how
    the model was discovered:

    * The short directory name only (``Foo-4bit``) — when the model dir
      sits directly inside ``model_dirs``.
    * The ``org--name`` double-dash form (``mlx-community--Foo-4bit``)
      — when oMLX scans the HuggingFace hub cache and finds a dir like
      ``models--mlx-community--Foo-4bit``, it strips the ``models--``
      prefix but preserves the ``org--name`` structure.

    Resolution order (first match wins):

    1. Exact match against ``GET /v1/models``.
    2. Basename after the last ``/`` (drops org prefix).
    3. Double-dash form: ``org/name`` → ``org--name``.
    4. Case-insensitive comparison of all three forms.

    Returns the resolved oMLX model id, or ``None`` if no candidate was
    found (caller should run :func:`aprovision_omlx_server` and retry).
    """
    status = await afetch_status(cfg)
    ids = [m["id"] for m in (status.get("models") or [])]
    if not ids:
        return None

    # Form 1: exact (e.g. already using oMLX id directly)
    if requested in ids:
        return requested

    # Form 2: basename (drops org prefix)
    short = requested.rsplit("/", 1)[-1]
    if short in ids:
        return short

    # Form 3: double-dash org--name (oMLX's HF hub cache id format)
    dashed = requested.replace("/", "--")
    if dashed in ids:
        return dashed

    # Case-insensitive fallback for all three forms
    lowered = {i.lower(): i for i in ids}
    for cand in (requested, short, dashed):
        if cand.lower() in lowered:
            return lowered[cand.lower()]

    return None


async def _wait_until_reachable(cfg: OmlxConfig, *, timeout: float) -> Optional[dict]:
    """Poll ``/v1/models`` until reachable or ``timeout`` elapses.

    Returns the status dict on success, ``None`` on timeout.
    """
    deadline = time.time() + timeout
    last: Optional[dict] = None
    while time.time() < deadline:
        last = await afetch_status(cfg)
        if last["reachable"]:
            return last
        await asyncio.sleep(0.5)
    return None


async def _wait_until_model_loaded(
    cfg: OmlxConfig, model_id: str, *, timeout: float,
) -> Optional[dict]:
    """Poll ``/v1/models`` until *model_id* appears in the loaded list.

    Stricter than :func:`_wait_until_reachable`: requires the requested
    model to actually be present, not just any model. Returns the status
    dict on success, ``None`` on timeout.
    """
    deadline = time.time() + timeout
    last: Optional[dict] = None
    while time.time() < deadline:
        last = await afetch_status(cfg)
        if last.get("reachable"):
            ids = [m["id"] for m in (last.get("models") or [])]
            if model_id in ids:
                return last
        await asyncio.sleep(0.5)
    return None


# ---------------------------------------------------------------------------
# Spawned-pid + log file (only used in the direct-spawn fallback path)
# ---------------------------------------------------------------------------


def _state_dir() -> Path:
    from backend.config import get_app_data_dir
    d = get_app_data_dir() / "omlx"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pid_path() -> Path:
    return _state_dir() / "spawn.pid"


def _spawn_log_path() -> Path:
    return _state_dir() / "omlx.log"


def _record_spawned_pid(pid: int) -> None:
    try:
        _pid_path().write_text(str(pid), encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to record omlx spawn pid: %s", exc)


def _read_spawned_pid() -> Optional[int]:
    p = _pid_path()
    if not p.exists():
        return None
    try:
        return int(p.read_text(encoding="utf-8").strip() or "0") or None
    except (OSError, ValueError):
        return None


def _clear_spawned_pid() -> None:
    try:
        _pid_path().unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("Failed to clear omlx spawn pid: %s", exc)


def tail_log(*, max_lines: int = 200) -> list[str]:
    """Tail of our spawn-fallback log file.

    When oMLX is running via ``brew services``, the canonical log is at
    ``$(brew --prefix)/var/log/omlx.log`` instead — surface that path
    in the snapshot so the UI can fetch it via a separate endpoint if
    needed.
    """
    p = _spawn_log_path()
    if not p.exists():
        return []
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines = text.splitlines()
    return lines[-max_lines:]


def info_snapshot(cfg: OmlxConfig) -> dict:
    """Combined snapshot used by both the REST API and the agent tools."""
    det = detect(cfg)
    out = {
        "config": cfg.model_dump(),
        "detection": det.to_dict(),
        "spawn_pid": _read_spawned_pid(),
        "spawn_log_path": str(_spawn_log_path()),
    }
    return out


# ---------------------------------------------------------------------------
# Cache settings (turbo mode / SSD cold tier)
# ---------------------------------------------------------------------------

_CACHE_DEFAULTS: dict[str, Any] = {
    "cache_enabled": True,
    "hot_cache_only": False,   # False = SSD tier is on
    "hot_cache_max_size": "0",
    "ssd_cache_dir": "",
    "ssd_cache_max_size": "auto",
    "initial_cache_blocks": 256,
    "max_context_window": 131072,
    # Continuous-batching concurrency cap: how many requests oMLX decodes
    # in parallel.  Higher values raise throughput at the cost of memory.
    "max_concurrent_requests": 8,
}


async def aget_cache_settings(cfg: OmlxConfig) -> dict:
    """Return oMLX's current cache/sampling settings.

    Reads the live ``~/.omlx/settings.json``; returns sensible defaults
    when the file is absent or the server hasn't been provisioned yet.
    """
    raw = _read_omlx_settings()
    cache = raw.get("cache", {}) or {}
    sampling = raw.get("sampling", {}) or {}
    scheduler = raw.get("scheduler", {}) or {}
    return {
        "cache_enabled":       bool(cache.get("enabled",         _CACHE_DEFAULTS["cache_enabled"])),
        "hot_cache_only":      bool(cache.get("hot_cache_only",  _CACHE_DEFAULTS["hot_cache_only"])),
        "hot_cache_max_size":  str(cache.get("hot_cache_max_size", _CACHE_DEFAULTS["hot_cache_max_size"])),
        "ssd_cache_dir":       str(cache.get("ssd_cache_dir") or ""),
        "ssd_cache_max_size":  str(cache.get("ssd_cache_max_size", _CACHE_DEFAULTS["ssd_cache_max_size"])),
        "initial_cache_blocks": int(cache.get("initial_cache_blocks", _CACHE_DEFAULTS["initial_cache_blocks"])),
        "max_context_window":  int(sampling.get("max_context_window", _CACHE_DEFAULTS["max_context_window"])),
        "max_concurrent_requests": int(scheduler.get("max_concurrent_requests", _CACHE_DEFAULTS["max_concurrent_requests"])),
    }


async def aset_cache_settings(cfg: OmlxConfig, patch: dict) -> dict:
    """Update oMLX's cache settings via the admin API.

    *patch* may contain any subset of the keys returned by
    :func:`aget_cache_settings`.  Only the supplied keys are changed;
    omitted keys are left untouched.

    Returns the resulting settings dict on success, raises on failure.
    """
    # Translate our field names to GlobalSettingsRequest keys.
    field_map = {
        "cache_enabled":        "cache_enabled",
        "hot_cache_only":       "hot_cache_only",
        "hot_cache_max_size":   "hot_cache_max_size",
        "ssd_cache_dir":        "ssd_cache_dir",
        "ssd_cache_max_size":   "ssd_cache_max_size",
        "initial_cache_blocks": "initial_cache_blocks",
        # sampling fields
        "max_context_window":   "sampling_max_context_window",
        # scheduler fields (continuous-batching concurrency)
        "max_concurrent_requests": "max_concurrent_requests",
    }
    payload = {field_map[k]: v for k, v in patch.items() if k in field_map}
    if not payload:
        return await aget_cache_settings(cfg)

    # Self-heal: if Otto's key is blank but the running server already has
    # one (started outside Otto's astart path), adopt it before giving up.
    adopt_existing_admin_key(cfg)
    api_key = (cfg.admin_api_key or "").strip()
    if not api_key:
        raise RuntimeError(
            "No oMLX admin API key configured. Start the server once via "
            "the oMLX setup screen so Otto can provision it."
        )

    base = cfg.effective_base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=10.0) as client:
        cookie = await _admin_login(cfg, api_key, client=client)
        if cookie is None:
            raise RuntimeError(
                "oMLX admin login failed. The server may not be running, "
                "or the API key in Otto's config is out of sync."
            )
        r = await client.post(
            f"{base}/admin/api/global-settings",
            json=payload,
            cookies={"omlx_admin_session": cookie},
        )
        if r.status_code != 200:
            raise RuntimeError(
                f"oMLX global-settings update failed (HTTP {r.status_code}): "
                f"{r.text[:300]}"
            )

    return await aget_cache_settings(cfg)


# ---------------------------------------------------------------------------
# Live cache statistics + clear
# ---------------------------------------------------------------------------


async def aget_cache_stats(cfg: OmlxConfig) -> dict:
    """Return live cache performance stats from the running oMLX server.

    Combines the admin ``/admin/api/stats`` response with the on-disk
    size of ``~/.omlx/cache/`` so the UI can show both efficiency numbers
    and storage footprint in one call.  Returns a dict of zeros when the
    server is unreachable.
    """
    base = cfg.effective_base_url.rstrip("/")
    adopt_existing_admin_key(cfg)
    api_key = (cfg.admin_api_key or "").strip()

    stats: dict = {}
    if api_key:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                cookie = await _admin_login(cfg, api_key, client=client)
                if cookie:
                    r = await client.get(
                        f"{base}/admin/api/stats",
                        cookies={"omlx_admin_session": cookie},
                    )
                    if r.status_code == 200:
                        stats = r.json()
        except Exception:  # noqa: BLE001
            pass

    cache_dir = Path.home() / ".omlx" / "cache"
    disk_bytes = 0
    if cache_dir.is_dir():
        try:
            disk_bytes = sum(
                f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()
            )
        except OSError:
            pass

    return {
        "reachable":            bool(stats),
        "cache_efficiency_pct": round(float(stats.get("cache_efficiency", 0.0)), 1),
        "total_requests":       int(stats.get("total_requests", 0)),
        "total_tokens_served":  int(stats.get("total_tokens_served", 0)),
        "total_cached_tokens":  int(stats.get("total_cached_tokens", 0)),
        "total_prompt_tokens":  int(stats.get("total_prompt_tokens", 0)),
        "avg_prefill_tps":      round(float(stats.get("avg_prefill_tps", 0.0))),
        "avg_generation_tps":   round(float(stats.get("avg_generation_tps", 0.0)), 1),
        "uptime_seconds":       int(stats.get("uptime_seconds", 0)),
        "disk_bytes":           disk_bytes,
        "disk_gb":              round(disk_bytes / (1024 ** 3), 2),
        "cache_dir":            str(cache_dir),
    }


async def _aclear_cache_endpoint(cfg: OmlxConfig, endpoint: str) -> dict:
    adopt_existing_admin_key(cfg)
    api_key = (cfg.admin_api_key or "").strip()
    if not api_key:
        raise RuntimeError("No oMLX admin API key configured.")
    base = cfg.effective_base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=10.0) as client:
        cookie = await _admin_login(cfg, api_key, client=client)
        if cookie is None:
            raise RuntimeError("oMLX admin login failed.")
        r = await client.post(
            f"{base}{endpoint}",
            cookies={"omlx_admin_session": cookie},
        )
        if r.status_code != 200:
            raise RuntimeError(
                f"oMLX {endpoint} failed (HTTP {r.status_code}): {r.text[:200]}"
            )
        return r.json() if r.content else {"ok": True}


async def aclear_omlx_hot_cache(cfg: OmlxConfig) -> dict:
    """Clear the in-memory (hot) KV page cache on the running oMLX server."""
    return await _aclear_cache_endpoint(cfg, "/admin/api/hot-cache/clear")


async def aclear_omlx_ssd_cache(cfg: OmlxConfig) -> dict:
    """Clear the SSD cold-tier KV cache on the running oMLX server."""
    return await _aclear_cache_endpoint(cfg, "/admin/api/ssd-cache/clear")
