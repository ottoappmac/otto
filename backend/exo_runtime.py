"""Prebuilt exo runtime: download + verify + launch (no per-machine build).

The legacy path provisions exo on the user's machine by cloning the repo
and running ``uv sync`` (which compiles a Rust/pyo3 extension under a
nightly toolchain) plus an ``npm`` dashboard build — 2-5 minutes of
fragile, toolchain-dependent work that requires git, uv, node, and rustup
to all be present and healthy.

This module replaces that with an **on-demand download of a prebuilt,
notarized runtime** produced once in CI (see ``scripts/build_exo.py``).
When the user enables the local cluster we:

1. resolve which artifact to fetch (pinned exo ref + host arch) from a
   small JSON manifest,
2. download the tarball over HTTPS,
3. verify its SHA-256 (and, on macOS, rely on the notarization stapled at
   build time),
4. strip the ``com.apple.quarantine`` xattr defensively,
5. unpack it atomically into the per-user app-data dir, and
6. launch exo directly from the bundled venv — no git/uv/npm/rustup.

The runtime is **Apple-Silicon only**. On any other platform (or when no
artifact is published for the pinned ref) callers should fall back to the
legacy source-build path in :mod:`backend.exo_cli`.

Configuration (all optional; sensible defaults baked in):

    EXO_PREBUILT_MANIFEST_URL  JSON manifest listing artifacts by ref+arch
    EXO_PREBUILT_URL           direct artifact URL (skips the manifest)
    EXO_PREBUILT_SHA256        expected SHA-256 for EXO_PREBUILT_URL
    EXO_REF                    pinned exo ref used to key the manifest
    EXO_RUNTIME_DIR            override the extracted-runtime location
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

from backend import exo_cli

# A new exo runtime artifact + manifest bump upgrades users without a full
# Otto release. Keep this in sync with ``ExoConfig.repo_ref`` /
# ``exo_cli.DEFAULT_EXO_REF``.
DEFAULT_EXO_REF = exo_cli.DEFAULT_EXO_REF

# Published by ``.github/workflows/build-app.yml``. Overridable via env so
# forks / staging channels can point elsewhere without a code change.
DEFAULT_PREBUILT_MANIFEST_URL = (
    "https://github.com/ottoappmac/otto/releases/latest/download/"
    "exo-runtime-manifest.json"
)

ProgressCb = Callable[[str], None]


def _noop(_msg: str) -> None:  # pragma: no cover - trivial
    pass


# ── Platform / arch ──────────────────────────────────────────────────────

def host_arch_tag() -> str:
    """Return the artifact arch tag for this host, e.g. ``aarch64-apple-darwin``."""
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin":
        arch = "aarch64" if machine in ("arm64", "aarch64") else "x86_64"
        return f"{arch}-apple-darwin"
    if system == "Windows":
        arch = "aarch64" if machine in ("arm64", "aarch64") else "x86_64"
        return f"{arch}-pc-windows-msvc"
    arch = "x86_64" if machine in ("x86_64", "amd64") else machine
    return f"{arch}-unknown-linux-gnu"


def is_supported_host() -> bool:
    """Prebuilt runtime is published for Apple Silicon only."""
    return platform.system() == "Darwin" and platform.machine().lower() in (
        "arm64",
        "aarch64",
    )


# ── Paths ─────────────────────────────────────────────────────────────────

def runtime_dir() -> Path:
    """Where the extracted prebuilt runtime lives."""
    override = os.environ.get("EXO_RUNTIME_DIR")
    if override:
        return Path(override).expanduser()
    return exo_cli.exo_root() / "runtime"


def _state_file() -> Path:
    return runtime_dir() / ".prebuilt_state.json"


def _venv_bin() -> Path:
    return runtime_dir() / ".venv" / "bin"


def exo_executable() -> Path:
    """Path to the ``exo`` console script inside the bundled venv."""
    return _venv_bin() / "exo"


def runtime_python() -> Path:
    return _venv_bin() / "python"


def _site_packages(base: Optional[Path] = None) -> Optional[Path]:
    """Locate the venv's ``site-packages`` dir under ``base`` (or the runtime)."""
    root = base if base is not None else runtime_dir()
    libs = sorted((root / ".venv" / "lib").glob("python*/site-packages"))
    return libs[0] if libs else None


def _find_exo_pkg_parent(base: Path) -> Optional[Path]:
    """Locate the directory that *contains* the packed ``exo`` package.

    The prebuilt artifact ships exo's full source tree (editable install), so
    even when the ``.pth`` is broken the package itself is present — usually at
    ``<base>/src/exo``. Returns the dir to put on ``sys.path`` (the parent of
    ``exo/``), or ``None`` if no packed source is found.
    """
    # Fast path: the handful of layouts exo actually uses.
    for rel in ("src", "python/src", "python", "."):
        cand = base / rel
        if (cand / "exo" / "__init__.py").is_file():
            return cand
    # Fallback: scan, skipping the venv (whose site-packages we are fixing).
    best: Optional[Path] = None
    for init in base.glob("**/exo/__init__.py"):
        if ".venv" in init.parts:
            continue
        parent = init.parent.parent
        if best is None or len(parent.parts) < len(best.parts):
            best = parent
    return best


def _repair_editable_pth(base: Path) -> bool:
    """Fix a broken editable ``exo.pth`` in-place when the source is packed.

    Some artifacts ship ``exo.pth`` with an ABSOLUTE build-host path (the
    build-time relativization step didn't run / was skipped by ``--pack-only``).
    That path doesn't exist on the user's machine, so ``import exo`` fails even
    though the source travels inside the tarball. Rewrite the ``.pth`` to a path
    relative to site-packages so the runtime works wherever it's unpacked.

    Returns ``True`` when a repair was written. No-op when exo already resolves.
    """
    if _exo_importable(base):
        return False
    sp = _site_packages(base)
    if sp is None:
        return False
    pkg_parent = _find_exo_pkg_parent(base)
    if pkg_parent is None:
        return False
    rel = os.path.relpath(pkg_parent, sp)
    try:
        (sp / "exo.pth").write_text(rel + "\n")
    except OSError:
        return False
    return True


def _exo_importable(base: Optional[Path] = None) -> bool:
    """Whether the ``exo`` package will actually import from this runtime.

    exo is installed as an *editable* package: there is no real
    ``site-packages/exo/`` — instead a ``.pth`` points at the source tree
    packed inside the artifact (relativized at build time). A runtime whose
    ``.pth`` target is missing (a bad/old artifact, or one whose paths were
    never relativized) still has a ``bin/exo`` console script but dies with
    ``ModuleNotFoundError: No module named 'exo'`` at launch. So checking the
    console script alone is not enough — we must confirm the package resolves.

    Filesystem-only (no subprocess) so it is cheap enough for status polls.
    """
    sp = _site_packages(base)
    if sp is None:
        return False
    # Real (non-editable) install.
    if (sp / "exo" / "__init__.py").is_file():
        return True
    # Editable install — resolve each .pth entry relative to site-packages and
    # see if any of them contains an importable ``exo`` package.
    for pth in sp.glob("*.pth"):
        try:
            lines = pth.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            s = line.strip()
            if not s or s.startswith("import ") or s.startswith("#"):
                continue
            target = Path(s) if os.path.isabs(s) else Path(os.path.normpath(str(sp / s)))
            if (target / "exo" / "__init__.py").is_file():
                return True
    return False


def _interpreter_launches(base: Optional[Path] = None) -> bool:
    """Whether the venv's own ``python`` can actually execute on this host.

    Filesystem checks (:func:`_exo_importable`, the console-script
    existence check) can't catch a broken *interpreter*: a bad artifact
    whose ``bin/python`` was accidentally linked against a build-host-only
    path (e.g. a Homebrew ``Python.framework``, see ``scripts/build_exo.py``
    ``uv_sync``/``verify_relocatable``) still has all the right files on
    disk, but dies instantly with a dyld ``Library not loaded`` error the
    moment it's invoked — which previously only surfaced as a cryptic crash
    in the exo daemon log after "Up" was clicked. Running a trivial
    ``--version`` here lets a stale/bad install be detected up front and
    re-downloaded automatically instead of repeatedly failing to launch.
    """
    py = runtime_python() if base is None else (base / ".venv" / "bin" / "python")
    if not py.exists():
        return False
    try:
        subprocess.run(
            [str(py), "--version"],
            check=True, capture_output=True, timeout=10,
        )
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        return False
    return True


# ── Install state ──────────────────────────────────────────────────────────

@dataclass
class RuntimeState:
    exo_ref: str = ""
    sha256: str = ""
    arch: str = ""
    source_url: str = ""
    installed_at: str = ""


def load_runtime_state() -> RuntimeState:
    p = _state_file()
    if not p.exists():
        return RuntimeState()
    try:
        data = json.loads(p.read_text())
        known = {f for f in RuntimeState.__dataclass_fields__}
        return RuntimeState(**{k: v for k, v in data.items() if k in known})
    except Exception:
        return RuntimeState()


def _save_runtime_state(state: RuntimeState) -> None:
    p = _state_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(state), indent=2))


def is_installed(exo_ref: str = DEFAULT_EXO_REF) -> bool:
    """True when a *usable* prebuilt runtime for ``exo_ref`` is on disk.

    "Usable" means the ``exo`` console script exists, the ``exo`` package
    actually resolves (see :func:`_exo_importable`), **and** the bundled
    interpreter itself can execute (see :func:`_interpreter_launches`). A
    half-installed, stale, or non-relocatable runtime reports ``False``
    here so :func:`backend.exo_provisioner` re-downloads it on the next
    ``up`` instead of repeatedly launching a broken daemon.
    """
    if not exo_executable().exists():
        return False
    if not _exo_importable():
        return False
    if not _interpreter_launches():
        return False
    state = load_runtime_state()
    # Empty ref in state means a legacy/unknown install — treat as usable
    # only when the caller didn't pin a specific ref.
    if state.exo_ref and exo_ref and state.exo_ref != exo_ref:
        return False
    return True


# ── Manifest resolution ──────────────────────────────────────────────────

def _http_get(url: str, *, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


def resolve_artifact(
    exo_ref: str = DEFAULT_EXO_REF,
    prebuilt_url: str = "",
) -> tuple[str, str]:
    """Resolve ``(url, sha256)`` for ``exo_ref`` on this host.

    Resolution order:
    1. Explicit ``EXO_PREBUILT_URL`` env var (+ optional ``EXO_PREBUILT_SHA256``).
    2. ``prebuilt_url`` config value (from Settings → Cluster → Advanced):
       - ends in ``.json`` → treated as a manifest URL (same format as the
         default GitHub manifest).
       - otherwise → treated as a direct artifact URL (SHA-256 check skipped).
    3. JSON manifest at ``EXO_PREBUILT_MANIFEST_URL`` (env) or the built-in
       ``DEFAULT_PREBUILT_MANIFEST_URL`` (GitHub releases).

    Raises ``RuntimeError`` when no artifact can be resolved so the caller
    can fall back to the source-build path.
    """
    direct = os.environ.get("EXO_PREBUILT_URL")
    if direct:
        return direct, os.environ.get("EXO_PREBUILT_SHA256", "")

    # Config-level override from Settings → Cluster → Advanced.
    cfg_url = prebuilt_url.strip()
    if cfg_url:
        if cfg_url.endswith(".json"):
            # Manifest URL — fall through to manifest parsing below.
            manifest_url = cfg_url
        else:
            # Direct artifact URL (tarball or file:// path).
            return cfg_url, ""

    if not cfg_url:
        manifest_url = os.environ.get(
            "EXO_PREBUILT_MANIFEST_URL", DEFAULT_PREBUILT_MANIFEST_URL
        )
    arch = host_arch_tag()
    try:
        raw = _http_get(manifest_url)
        manifest = json.loads(raw.decode("utf-8"))
    except (urllib.error.URLError, ValueError, OSError) as exc:
        raise RuntimeError(
            f"no prebuilt exo runtime manifest available yet "
            f"(fetching {manifest_url} → {exc}). "
            "Switch to source mode in Settings → Cluster → Setup, "
            "or set EXO_PREBUILT_URL to a local file:// path for testing."
        ) from exc

    # Manifest shape:
    # { "artifacts": [ {"exo_ref","arch","url","sha256"}, ... ] }
    for entry in manifest.get("artifacts", []):
        if entry.get("arch") == arch and entry.get("exo_ref") == exo_ref:
            url = entry.get("url")
            if url:
                return url, entry.get("sha256", "")

    raise RuntimeError(
        f"no prebuilt exo runtime for ref={exo_ref!r} arch={arch!r} "
        f"in manifest {manifest_url}. "
        "Switch to source mode in Settings → Cluster → Setup."
    )


# ── Download + verify + extract ────────────────────────────────────────────

def _download(url: str, dest: Path, progress: ProgressCb) -> None:
    progress(f"Downloading exo runtime: {url}")
    last = [0.0]

    def _hook(block: int, block_size: int, total: int) -> None:
        now = time.monotonic()
        if now - last[0] < 1.0:
            return
        last[0] = now
        got = block * block_size
        if total > 0:
            pct = min(100, int(got * 100 / total))
            progress(f"  {pct}%  ({got // (1024 * 1024)}/{total // (1024 * 1024)} MB)")
        else:
            progress(f"  {got // (1024 * 1024)} MB")

    urllib.request.urlretrieve(url, dest, reporthook=_hook)  # noqa: S310


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _strip_quarantine(root: Path) -> None:
    """Best-effort removal of the macOS quarantine xattr.

    Files written by our own backend process are typically not quarantined,
    but we strip defensively so Gatekeeper never blocks the bundled python /
    dylibs / Rust extension at launch.
    """
    if platform.system() != "Darwin":
        return
    try:
        subprocess.run(
            ["xattr", "-dr", "com.apple.quarantine", str(root)],
            check=False,
            capture_output=True,
        )
    except Exception:
        pass


def _extract_tarball(archive: Path, dest: Path, progress: ProgressCb) -> None:
    """Extract ``archive`` so its contents land directly in ``dest``.

    The artifact is packed with a single top-level directory; we strip it so
    ``dest/.venv`` resolves regardless of the archive's root folder name.
    """
    progress("Unpacking runtime…")
    with tarfile.open(archive) as tf:
        members = tf.getmembers()
        if not members:
            raise RuntimeError("exo runtime archive is empty")
        top = members[0].name.split("/")[0] + "/"
        stripped = []
        for m in members:
            if m.name == top.rstrip("/"):
                continue
            if m.name.startswith(top):
                m.name = m.name[len(top):]
                if m.name:
                    stripped.append(m)
        tf.extractall(dest, members=stripped)  # noqa: S202 - trusted, sha-verified


def install(
    *,
    exo_ref: str = DEFAULT_EXO_REF,
    prebuilt_url: str = "",
    progress: Optional[ProgressCb] = None,
    force: bool = False,
) -> RuntimeState:
    """Download, verify, and install the prebuilt exo runtime.

    Idempotent: returns immediately when an install for ``exo_ref`` already
    exists unless ``force`` is set. Raises ``RuntimeError`` on any failure so
    the caller can surface it / fall back to source mode.

    ``prebuilt_url`` is forwarded to :func:`resolve_artifact`; see that
    function's docstring for the full resolution order.
    """
    progress = progress or _noop

    if not is_supported_host():
        raise RuntimeError(
            "prebuilt exo runtime is only published for Apple Silicon (arm64 macOS)"
        )

    if not force and is_installed(exo_ref):
        progress("exo runtime already installed — skipping download")
        return load_runtime_state()

    url, expected_sha = resolve_artifact(exo_ref, prebuilt_url=prebuilt_url)
    rt = runtime_dir()
    rt.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="exo-rt-") as tmp:
        archive = Path(tmp) / "exo-runtime.tar.gz"
        _download(url, archive, progress)

        actual_sha = _sha256(archive)
        if expected_sha and actual_sha != expected_sha:
            raise RuntimeError(
                "exo runtime checksum mismatch: "
                f"expected {expected_sha}, got {actual_sha}"
            )
        progress(f"Verified SHA-256 {actual_sha[:16]}…")

        # Atomic swap: extract into a staging dir next to the target, then
        # replace. Avoids a half-written runtime if extraction is interrupted.
        staging = Path(tmp) / "staged"
        staging.mkdir()
        _extract_tarball(archive, staging, progress)

        if not (staging / ".venv" / "bin" / "exo").exists():
            raise RuntimeError(
                "extracted runtime is missing .venv/bin/exo — bad artifact"
            )

        # The console script is not enough: exo is installed editable, so the
        # packed source tree + relativized .pth must actually resolve. If the
        # artifact shipped a broken (absolute build-host) .pth, repair it
        # in-place against the packed source before validating.
        if _repair_editable_pth(staging):
            progress("Repaired editable exo.pth to point at the packed source")
        if not _exo_importable(staging):
            raise RuntimeError(
                "extracted runtime has no importable 'exo' package (no real "
                "package and no .pth resolving to one) — the prebuilt artifact "
                "looks broken. Try a different Runtime URL, or switch to "
                "source mode in Settings -> Cluster -> Advanced."
            )

        if rt.exists():
            shutil.rmtree(rt)
        shutil.move(str(staging), str(rt))

    _strip_quarantine(rt)

    state = RuntimeState(
        exo_ref=exo_ref,
        sha256=actual_sha,
        arch=host_arch_tag(),
        source_url=url,
        installed_at=_now_iso(),
    )
    _save_runtime_state(state)
    progress(f"exo runtime installed at {rt}")
    return state


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


# ── Launch ──────────────────────────────────────────────────────────────────

def launch_command(
    *,
    api_port: int,
    libp2p_port: int = 0,
    force_master: bool = False,
) -> list[str]:
    """Build the argv to launch exo from the prebuilt runtime.

    Uses the venv's ``exo`` console script directly — no ``uv``. When the
    venv was packed with ``uv venv --relocatable`` its shebang resolves the
    interpreter relative to the script, so this works from any install path.

    ``force_master`` is set for the single-Mac default so the node serves
    immediately as master without waiting on libp2p peer discovery (which is
    what makes a single machine usable without the Local Network grant).
    """
    cmd = [str(exo_executable()), "--api-port", str(api_port)]
    if libp2p_port:
        cmd += ["--libp2p-port", str(libp2p_port)]
    if force_master:
        cmd += ["--force-master"]
    return cmd
