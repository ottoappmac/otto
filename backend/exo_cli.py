#!/usr/bin/env python3
"""exo cluster lifecycle CLI for the Otto desktop app.

Stdlib-only single-file module. Designed to run identically on the master
machine and (after ``scp``) on every secondary node. Every persistent path
mirrors what Otto's backend expects so this module is also imported by
``backend.exo_provisioner`` (the FastAPI / deep-agent integration layer).

This file MUST NOT introduce any non-stdlib imports — it is shipped as a
single file to remote hosts via ``scp`` and executed with the bare system
``python3``. ``scripts/exo_cli.py`` is a thin user-facing shim that
delegates here.

Subcommands
-----------

``info``        Print resolved paths, env, and detected prereqs.
``doctor``      Check prereqs and print a diagnosis (no installs).
``status``      Hit the local exo HTTP API and report cluster state.
``provision``   Idempotent install: clone, ``uv sync``, build dashboard.
``start``       Launch the local exo daemon (background, pidfile-tracked).
``stop``        Stop the local exo daemon.
``up``          ``provision`` then ``start``.
``down``        ``stop``.
``smoke``       Battery of API checks against a running cluster.

Every command accepts ``--remote ALIAS`` to run the *same* command on a
remote host via SSH. The script ``scp``s itself to ``~/.exo_cli.py`` on
the remote, then invokes ``python3 ~/.exo_cli.py <subcommand>`` there
and streams output back.

Configuration
-------------

All defaults can be overridden via env vars or CLI flags::

    EXO_REPO_URL       https://github.com/exo-explore/exo.git
    EXO_REF            v1.0.71            (git tag/branch/sha)
    EXO_REPO_DIR       <app_data>/exo/repo  (where to install)
    EXO_BASE_URL       http://localhost:52415
    EXO_API_PORT       52415              (HTTP API; --api-port passed to exo)
    EXO_LIBP2P_PORT    0                  (libp2p TCP; 0 = OS-assigned)
    EXO_REMOTE_SSH     (no default; set to your ~/.ssh/config alias)
    EXO_NO_TERMINAL_WRAP  unset/0/false  on macOS, route ``start`` through
                       Terminal.app so the daemon inherits the Local
                       Network Privacy grant needed for libp2p mDNS
                       discovery; set to 1 to disable (e.g. when the
                       parent process — like Otto.app — already
                       holds the grant).
    OTTO_APP_DATA_DIR  override the per-user app data directory

Compatibility with the Otto backend
-------------------------------------

The functions below (``provision_exo``, ``fetch_cluster_status``,
``start_local``, ``stop_local``, ``run_remote``) are pure and side-effect
free except where explicit. ``backend/exo_provisioner.py`` imports these
directly to expose REST endpoints and LangChain tools — no
re-implementation needed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


# ════════════════════════════════════════════════════════════════════════
#  Constants and configuration
# ════════════════════════════════════════════════════════════════════════

DEFAULT_EXO_REPO_URL = "https://github.com/exo-explore/exo.git"
DEFAULT_EXO_REF = "v1.0.71"
DEFAULT_EXO_PORT = 52415
DEFAULT_LIBP2P_PORT = 0


def _env_api_port() -> int:
    raw = os.environ.get("EXO_API_PORT") or os.environ.get("EXO_NODE_PORT")
    return int(raw) if raw else DEFAULT_EXO_PORT


def _env_libp2p_port() -> int:
    raw = os.environ.get("EXO_LIBP2P_PORT")
    return int(raw) if raw else DEFAULT_LIBP2P_PORT


REMOTE_SCRIPT_PATH = "~/.exo_cli.py"

PROGRESS_PREFIX = "[exo-cli]"


# ════════════════════════════════════════════════════════════════════════
#  Path resolution (kept identical to backend/config.py:get_app_data_dir)
# ════════════════════════════════════════════════════════════════════════

def get_app_data_dir() -> Path:
    """Return Otto's per-user app data dir.

    Matches ``backend.config.get_app_data_dir`` exactly, plus an
    explicit ``OTTO_APP_DATA_DIR`` override so this script can be
    pointed at a custom location when run on a remote machine.
    """
    override = os.environ.get("OTTO_APP_DATA_DIR")
    if override:
        return Path(override).expanduser()

    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Otto"
    if system == "Windows":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "Otto"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "Otto"


def exo_root() -> Path:
    return get_app_data_dir() / "exo"


def exo_repo_dir() -> Path:
    override = os.environ.get("EXO_REPO_DIR")
    return Path(override).expanduser() if override else exo_root() / "repo"


def state_file() -> Path:
    return exo_root() / ".install_state.json"


def pid_file() -> Path:
    return exo_root() / "exo.pid"


def log_file() -> Path:
    return exo_root() / "exo.log"


# ════════════════════════════════════════════════════════════════════════
#  Provisioning state
# ════════════════════════════════════════════════════════════════════════

@dataclass
class ProvisionState:
    exo_ref: str = ""
    git_commit: str = ""
    deps_installed_for_commit: str = ""
    dashboard_built_for_commit: str = ""
    last_success_at: str = ""
    last_error: str = ""
    exo_repo_dir: str = ""
    # Cached human-readable warning when EXO's pinned MLX version differs
    # from the MLX installed in Otto's main process.  Empty string means
    # "preflight passed" or "preflight not yet run".  Surfaced via
    # ``provision_status`` JSON and the EXO setup UI so the user can
    # decide whether to ``--force-mismatch`` or pin EXO to a matching tag.
    mlx_version_warning: str = ""
    mlx_version_pinned: str = ""    # what EXO's pyproject pins
    mlx_version_bundled: str = ""   # what Otto's main runtime imports


def load_state() -> ProvisionState:
    p = state_file()
    if not p.exists():
        return ProvisionState()
    try:
        data = json.loads(p.read_text())
        # tolerate extra fields from future versions
        known = {f for f in ProvisionState.__dataclass_fields__}
        return ProvisionState(**{k: v for k, v in data.items() if k in known})
    except Exception:
        return ProvisionState()


def save_state(state: ProvisionState) -> None:
    p = state_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(state), indent=2))


# ════════════════════════════════════════════════════════════════════════
#  Logging / progress
# ════════════════════════════════════════════════════════════════════════

def progress(msg: str) -> None:
    print(f"{PROGRESS_PREFIX} {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"{PROGRESS_PREFIX} WARN: {msg}", file=sys.stderr, flush=True)


def fail(msg: str, code: int = 1) -> None:
    print(f"{PROGRESS_PREFIX} ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


# ════════════════════════════════════════════════════════════════════════
#  Subprocess helpers
# ════════════════════════════════════════════════════════════════════════

def _augment_env(extra_path_dirs: Iterable[str] = ()) -> dict[str, str]:
    env = os.environ.copy()
    extras = [str(Path(p).expanduser()) for p in extra_path_dirs]
    if extras:
        env["PATH"] = os.pathsep.join(extras + [env.get("PATH", "")])
    return env


def run(
    *args: str,
    cwd: Optional[Path] = None,
    env: Optional[dict[str, str]] = None,
    check: bool = True,
    capture: bool = False,
    label: str = "",
) -> tuple[int, str, str]:
    """Run a command; optionally capture, optionally stream to stdout."""
    label = label or args[0]
    if not capture:
        progress(f"$ {' '.join(shlex.quote(a) for a in args)}")
    proc = subprocess.run(
        list(args),
        cwd=str(cwd) if cwd else None,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if check and proc.returncode != 0:
        out = (proc.stdout or "") + (proc.stderr or "")
        fail(f"{label} exited {proc.returncode}: {out[-2000:].strip()}")
    return proc.returncode, (proc.stdout or ""), (proc.stderr or "")


def run_streaming(
    *args: str,
    cwd: Optional[Path] = None,
    env: Optional[dict[str, str]] = None,
    check: bool = True,
    label: str = "",
) -> int:
    """Run a command, streaming combined output to our stdout in real time."""
    label = label or args[0]
    progress(f"$ {' '.join(shlex.quote(a) for a in args)}")
    proc = subprocess.Popen(
        list(args),
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
    finally:
        rc = proc.wait()
    if check and rc != 0:
        fail(f"{label} exited {rc}")
    return rc


def which(binary: str) -> Optional[str]:
    return shutil.which(binary)


# ════════════════════════════════════════════════════════════════════════
#  HTTP helpers (stdlib only, used for /state polling)
# ════════════════════════════════════════════════════════════════════════

def http_get_json(url: str, timeout: float = 5.0) -> Any:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post_json(url: str, payload: dict, timeout: float = 30.0) -> Any:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_alive(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return False


# ════════════════════════════════════════════════════════════════════════
#  Prereq checks
# ════════════════════════════════════════════════════════════════════════

def _detect_brew_prefix() -> Optional[str]:
    for cand in ("/opt/homebrew/bin", "/usr/local/bin"):
        if Path(cand, "brew").exists():
            return cand
    return None


def _augmented_install_env() -> dict[str, str]:
    """PATH including likely toolchain locations after fresh installs."""
    extras: list[str] = []
    bp = _detect_brew_prefix()
    if bp:
        extras.append(bp)
    extras += [
        str(Path.home() / ".cargo" / "bin"),
        str(Path.home() / ".local" / "bin"),
    ]
    env = _augment_env(extras)
    # ``uv`` ships its own bundled rustls/webpki trust store and ignores
    # the OS keychain by default.  Behind a TLS-intercepting proxy
    # (Zscaler and friends), every ``uv sync`` / ``uv run`` then dies with
    # ``invalid peer certificate: UnknownIssuer`` because the proxy's root
    # CA only lives in the system trust store.  ``UV_NATIVE_TLS=1`` tells
    # uv to use the platform certificate store (macOS Keychain / Linux CA
    # bundle), which already trusts such corporate roots.  Harmless on
    # machines without a proxy.  ``setdefault`` honours an explicit
    # user-supplied value.
    env.setdefault("UV_NATIVE_TLS", "1")
    return env


@dataclass
class Prereqs:
    brew: Optional[str] = None
    uv: Optional[str] = None
    node: Optional[str] = None
    npm: Optional[str] = None
    git: Optional[str] = None
    rustup: Optional[str] = None
    cargo: Optional[str] = None
    rust_nightly: bool = False
    platform: str = ""

    def missing(self) -> list[str]:
        out = []
        if not self.git:
            out.append("git")
        if not self.uv:
            out.append("uv")
        if not self.node:
            out.append("node")
        if not self.npm:
            out.append("npm")
        if not self.rustup:
            out.append("rustup")
        if not self.cargo:
            out.append("cargo")
        if not self.rust_nightly:
            out.append("rust-nightly")
        # Homebrew is no longer required: on macOS we install uv via the
        # official script and Node via the official tarball when brew is
        # absent, so it is not reported as a missing prereq.
        return out


def _which_with_path(binary: str, path: str) -> Optional[str]:
    return shutil.which(binary, path=path)


def macos_clt_present() -> bool:
    """Whether the Xcode Command Line Tools are actually installed.

    ``/usr/bin/git`` (and ``clang``, ``python3``, …) exist as *stubs* on a
    vanilla macOS install even when the CLT are absent — invoking them just
    triggers the "No developer tools were found" installer and exits
    non-zero. ``shutil.which`` therefore can't tell us whether git/clang are
    usable. ``xcode-select -p`` is the reliable probe: it only succeeds when
    a developer dir is actually selected.
    """
    if platform.system() != "Darwin":
        return True
    try:
        rc, out, _ = run("xcode-select", "-p", capture=True, check=False)
    except Exception:  # noqa: BLE001
        return False
    return rc == 0 and bool(out.strip()) and Path(out.strip()).exists()


def detect_prereqs() -> Prereqs:
    env = _augmented_install_env()

    def w(b: str) -> Optional[str]:
        return _which_with_path(b, env["PATH"])

    p = Prereqs(
        brew=w("brew"),
        uv=w("uv"),
        node=w("node"),
        npm=w("npm"),
        git=w("git"),
        rustup=w("rustup"),
        cargo=w("cargo"),
        platform=platform.system(),
    )
    if p.rustup:
        rc, out, _ = run(p.rustup, "toolchain", "list", capture=True, check=False)
        p.rust_nightly = rc == 0 and "nightly" in out
    return p


# Pinned to match backend.node_provisioner.NODE_VERSION.
_NODE_VERSION = "v22.14.0"


def _install_node_portable_macos(env: dict[str, str]) -> None:
    """Download + install Node's official macOS tarball into ~/.local.

    Homebrew-free: unpacks the prebuilt distribution under
    ``~/.local/node`` and symlinks ``node``/``npm``/``npx`` into
    ``~/.local/bin`` (already on the augmented PATH).  No sudo required.
    """
    import tarfile
    import tempfile

    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x64"
    name = f"node-{_NODE_VERSION}-darwin-{arch}"
    url = f"https://nodejs.org/dist/{_NODE_VERSION}/{name}.tar.gz"

    node_root = Path.home() / ".local" / "node"
    local_bin = Path.home() / ".local" / "bin"
    local_bin.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / f"{name}.tar.gz"
        progress(f"Downloading {url}")
        urllib.request.urlretrieve(url, archive)  # noqa: S310

        if node_root.exists():
            shutil.rmtree(node_root)
        node_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tf:
            members = tf.getmembers()
            top = members[0].name.split("/")[0] + "/"
            for m in members:
                if m.name.startswith(top):
                    m.name = m.name[len(top):]
            tf.extractall(node_root, members=[m for m in members if m.name])

    for tool in ("node", "npm", "npx"):
        src = node_root / "bin" / tool
        dst = local_bin / tool
        if src.exists():
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src)

    if not shutil.which("node", path=env["PATH"]):
        # ``~/.local/bin`` is on the augmented PATH; if node still isn't
        # found the extraction failed.
        raise RuntimeError("Node install completed but `node` is not on PATH")


def install_prereqs(*, auto: bool) -> Prereqs:
    """Install missing prereqs. ``auto=False`` raises with manual instructions."""
    p = detect_prereqs()
    missing = p.missing()
    if not missing:
        progress("All prereqs already present.")
        return p

    if not auto:
        fail(
            "Missing prereqs: "
            + ", ".join(missing)
            + ". Re-run with --auto-prereqs, or install them manually:\n"
            + "  macOS (no Homebrew needed):\n"
            + "    curl -LsSf https://astral.sh/uv/install.sh | sh   # uv\n"
            + "    download Node from https://nodejs.org/ (or `brew install node`)\n"
            + "    curl https://sh.rustup.rs -sSf | sh && rustup toolchain install nightly\n"
            + "  Linux:  install uv (https://astral.sh/uv) + node + git via your package manager,\n"
            + "          then curl https://sh.rustup.rs -sSf | sh && rustup toolchain install nightly"
        )

    env = _augmented_install_env()

    if p.platform == "Darwin":
        # Xcode Command Line Tools are a hard requirement on macOS that no
        # brew-free fallback can paper over: `uv sync` fetches git+-sourced
        # deps (uv shells out to git) and exo builds native/Rust extensions
        # (exo-pyo3-bindings, mlx) — both need a real git + clang/linker. The
        # /usr/bin/git stub passes `shutil.which` but fails at runtime, so we
        # probe `xcode-select -p` explicitly and fail fast with guidance
        # instead of letting it blow up deep inside `uv sync`.
        if not macos_clt_present():
            progress("Triggering Xcode Command Line Tools install…")
            run("xcode-select", "--install", check=False, label="xcode-select")
            fail(
                "Xcode Command Line Tools are required to provision exo "
                "(uv fetches git-based dependencies and exo compiles native "
                "Rust/C extensions). The installer was launched — complete it "
                "(or run `xcode-select --install`), then re-run exo provisioning."
            )

        # Prefer Homebrew when present (fast, clean), but never require it —
        # uv installs via its official script and Node via the official
        # prebuilt tarball, both into the user's home dir (no sudo).
        if not shutil.which("uv", path=env["PATH"]):
            if p.brew:
                progress("Installing uv via Homebrew…")
                run_streaming("brew", "install", "uv", env=env, label="brew/uv")
            else:
                progress("Installing uv via official installer…")
                run_streaming(
                    "bash", "-c",
                    "curl -LsSf https://astral.sh/uv/install.sh | sh",
                    env=env, label="uv-install",
                )
            env = _augmented_install_env()

        if not shutil.which("node", path=env["PATH"]):
            if p.brew:
                progress("Installing Node via Homebrew…")
                run_streaming("brew", "install", "node", env=env, label="brew/node")
            else:
                progress("Installing Node via official prebuilt tarball…")
                _install_node_portable_macos(env)
            env = _augmented_install_env()

        if not shutil.which("git", path=env["PATH"]):
            if p.brew:
                progress("Installing git via Homebrew…")
                run_streaming("brew", "install", "git", env=env, label="brew/git")
            else:
                progress("Triggering Xcode Command Line Tools install for git…")
                run("xcode-select", "--install", check=False, label="xcode-select")
                fail(
                    "git is required. The Xcode Command Line Tools installer "
                    "was launched — complete it, then re-run."
                )
            env = _augmented_install_env()
    elif p.platform == "Linux":
        for bin_name in ("git", "node", "npm"):
            if not shutil.which(bin_name, path=env["PATH"]):
                fail(
                    f"`{bin_name}` is required on the secondary. Install it via "
                    "your distro's package manager (apt/dnf/pacman) and re-run."
                )
        if not shutil.which("uv", path=env["PATH"]):
            progress("Installing uv via official installer…")
            run_streaming(
                "bash", "-c",
                "curl -LsSf https://astral.sh/uv/install.sh | sh",
                env=env, label="uv-install",
            )
            env = _augmented_install_env()
    else:
        fail(f"Unsupported platform: {p.platform}")

    if not shutil.which("rustup", path=env["PATH"]):
        progress("Installing rustup…")
        run_streaming(
            "bash", "-c",
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
            "| sh -s -- -y --default-toolchain none",
            env=env, label="rustup-init",
        )
        env = _augmented_install_env()

    progress("Ensuring Rust nightly toolchain…")
    run_streaming("rustup", "toolchain", "install", "nightly", env=env, label="rustup")

    p = detect_prereqs()
    if p.missing():
        fail("Prereq installation completed but the following are still missing: "
             + ", ".join(p.missing()))
    return p


# ════════════════════════════════════════════════════════════════════════
#  Provisioning steps
# ════════════════════════════════════════════════════════════════════════

def _git_resolve_commit(repo: Path, ref: str) -> str:
    env = _augmented_install_env()
    rc, out, _ = run(
        "git", "rev-parse", "--verify", f"{ref}^{{commit}}",
        cwd=repo, env=env, capture=True, label="git/rev-parse",
    )
    return out.strip()


def _download_repo_tarball(repo_url: str, ref: str, repo_dir: Path) -> None:
    """Fetch a repo at ``ref`` as a source tarball — no ``git`` required.

    Fallback for environments where ``git clone`` fails because the Xcode
    Command Line Tools aren't installed (``/usr/bin/git`` is a stub that
    exits non-zero).  Uses GitHub's ``/archive/<ref>.tar.gz`` endpoint,
    which urllib follows through the codeload redirect.
    """
    import tarfile
    import tempfile

    base = repo_url[:-4] if repo_url.endswith(".git") else repo_url
    archive_url = f"{base.rstrip('/')}/archive/{ref}.tar.gz"

    progress(f"Downloading source tarball {archive_url}…")
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "src.tar.gz"
        urllib.request.urlretrieve(archive_url, archive)  # noqa: S310
        repo_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tf:
            members = tf.getmembers()
            top = members[0].name.split("/")[0] + "/"
            for m in members:
                if m.name.startswith(top):
                    m.name = m.name[len(top):]
            tf.extractall(repo_dir, members=[m for m in members if m.name])

    if not any(repo_dir.iterdir()):
        fail(f"source tarball for {ref} extracted no files")


def ensure_repo(repo_dir: Path, ref: str, repo_url: str) -> str:
    env = _augmented_install_env()
    src_marker = repo_dir / ".exo_src_ref"

    if not (repo_dir / ".git").exists():
        # A previous tarball checkout at this exact ref — reuse it so we
        # don't re-download and wipe the existing ``.venv`` on every run.
        if src_marker.exists() and src_marker.read_text(encoding="utf-8").strip() == ref:
            progress(f"exo source present (tarball @ {ref}) — skipping clone")
            return ref

        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        if repo_dir.exists() and any(repo_dir.iterdir()):
            warn(f"{repo_dir} exists but is not a git repo; removing.")
            shutil.rmtree(repo_dir)
        progress(f"Cloning {repo_url} ({ref}) into {repo_dir}…")
        # --depth 1 + --branch lets us pull a tag or branch without full history.
        # ``check=False`` so we can fall back to a git-free tarball download when
        # git is unusable (e.g. Xcode Command Line Tools not installed).
        rc = run_streaming(
            "git", "clone", "--depth", "1", "--branch", ref,
            repo_url, str(repo_dir),
            env=env, label="git/clone", check=False,
        )
        if rc != 0:
            warn(
                f"git clone failed (exit {rc}) — often the Xcode Command Line "
                "Tools aren't installed (/usr/bin/git is a stub that exits "
                "non-zero). Falling back to a git-free source tarball download."
            )
            _download_repo_tarball(repo_url, ref, repo_dir)
            src_marker.write_text(ref, encoding="utf-8")
            return ref
    else:
        progress(f"Updating exo repo at {repo_dir}…")
        run("git", "fetch", "--tags", "--depth", "1", "origin", ref,
            cwd=repo_dir, env=env, label="git/fetch")
        run("git", "checkout", "-f", ref, cwd=repo_dir, env=env, label="git/checkout")
    return _git_resolve_commit(repo_dir, ref)


def ensure_uv_sync(repo_dir: Path) -> None:
    env = _augmented_install_env()
    progress("Installing exo Python deps via `uv sync` (~2-5min on first run)…")
    run_streaming("uv", "sync", cwd=repo_dir, env=env, label="uv/sync")


# ── MLX version preflight ────────────────────────────────────────────────
#
# EXO ships its own ``.venv`` and its own pinned MLX, completely
# independent of the MLX bundled in Otto's main process.  That isolation
# is great until you load the same model in both engines: Otto on
# ``mlx==0.31.x`` and EXO on ``mlx==0.18.x`` can disagree about chat
# templates, KV-cache layout, or quantisation metadata even though the
# weights are byte-identical on disk.  The preflight reads EXO's
# ``pyproject.toml`` after we've cloned at the requested ref and compares
# the pinned version to whatever ``import mlx`` resolves to in the live
# Otto process.  We only ever WARN — blocking would prevent legitimate
# advanced workflows where the user knows they want a specific EXO ref —
# and store the message on ProvisionState so the UI / CLI can echo it.

_MLX_PIN_RE = re.compile(
    r'(?im)^\s*"?(mlx(?:-lm)?)"?\s*[=~!<>]+\s*"?([0-9][0-9A-Za-z.\-+]*)',
)


def _read_exo_pinned_mlx(repo_dir: Path) -> dict[str, str]:
    """Best-effort scan of EXO's pyproject for ``mlx`` / ``mlx-lm`` pins.

    Returns a dict like ``{"mlx": "0.18.0", "mlx-lm": "0.18.0"}`` (only
    keys we found).  Empty dict means the file is missing, malformed, or
    just doesn't pin MLX explicitly — both treated the same downstream
    (the warning is suppressed because we have nothing to compare).
    """
    found: dict[str, str] = {}
    for fname in ("pyproject.toml", "requirements.txt", "setup.py"):
        path = repo_dir / fname
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in _MLX_PIN_RE.finditer(text):
            pkg = match.group(1).lower()
            ver = match.group(2)
            found.setdefault(pkg, ver)
    return found


def _bundled_mlx_versions() -> dict[str, str]:
    """Return ``{"mlx": ..., "mlx-lm": ...}`` for the host process.

    Uses ``importlib.metadata`` so we don't have to import the modules
    themselves (which would force MLX initialisation just to read a
    version string).  Missing keys mean the package isn't installed in
    Otto's main runtime, which is fine — the preflight just falls
    silent.
    """
    import importlib.metadata as md
    out: dict[str, str] = {}
    for pkg in ("mlx", "mlx-lm"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            continue
        except Exception:
            continue
    return out


def _major_minor(ver: str) -> tuple[int, int] | None:
    """Best-effort extract ``(major, minor)`` from a PEP-440-ish version.

    Returns ``None`` for anything we can't parse — callers downgrade the
    severity to a soft note in that case rather than blocking.
    """
    parts = re.findall(r"\d+", ver or "")
    if not parts:
        return None
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return (major, minor)
    except ValueError:
        return None


def mlx_version_preflight(repo_dir: Path) -> tuple[str, str, str, str]:
    """Compare EXO's pinned MLX version to Otto's bundled MLX version.

    Returns ``(severity, bundled, pinned, message)`` where ``severity``
    is one of ``""`` (match / nothing to compare), ``"minor"``
    (different patch / build but same major.minor) or ``"major"``
    (different major.minor — the case most likely to cause subtle
    runtime divergence).
    """
    pinned = _read_exo_pinned_mlx(repo_dir)
    bundled = _bundled_mlx_versions()

    # Prefer mlx-lm when both are present — it's the higher-level wheel
    # and pulls a transitive ``mlx`` of its own, so a mlx-lm match
    # implicitly satisfies the mlx pin too.
    for key in ("mlx-lm", "mlx"):
        if key in pinned and key in bundled:
            p = pinned[key]
            b = bundled[key]
            if p == b:
                return ("", b, p, f"OK: {key} matches ({b}).")
            mm_p = _major_minor(p)
            mm_b = _major_minor(b)
            if mm_p and mm_b and mm_p != mm_b:
                return (
                    "major",
                    b,
                    p,
                    f"EXO pins {key}=={p} but Otto's main process is using {key}=={b}. "
                    "Major/minor mismatch — chat templates, KV-cache layout or "
                    "quantisation metadata may diverge between engines. Pass "
                    "--force-mismatch to provision anyway, or pin EXO_REF to a "
                    "tag that matches your local MLX.",
                )
            return (
                "minor",
                b,
                p,
                f"EXO pins {key}=={p}, Otto bundles {key}=={b}. Same major/minor "
                "line but different patch — usually safe.",
            )

    # Nothing comparable; treat as a no-op.  Do not invent a warning.
    return ("", bundled.get("mlx", ""), pinned.get("mlx", ""), "")


def ensure_dashboard(repo_dir: Path) -> None:
    env = _augmented_install_env()
    dash = repo_dir / "dashboard"
    if not dash.exists():
        warn(f"{dash} not present in repo; skipping dashboard build.")
        return
    progress("Installing dashboard npm deps…")
    run_streaming("npm", "ci", cwd=dash, env=env, label="npm/ci")
    progress("Building dashboard…")
    run_streaming("npm", "run", "build", cwd=dash, env=env, label="npm/build")


def smoke_test_exo_cli(repo_dir: Path) -> None:
    """Verify ``uv run exo --help`` exits 0 inside the cloned repo."""
    env = _augmented_install_env()
    progress("Smoke testing `uv run exo --help` (verifies build)…")
    run_streaming("uv", "run", "--project", str(repo_dir), "exo", "--help",
                  cwd=repo_dir, env=env, label="exo/--help")


def provision_exo(
    *,
    exo_ref: str = DEFAULT_EXO_REF,
    repo_url: str = DEFAULT_EXO_REPO_URL,
    repo_dir: Optional[Path] = None,
    auto_prereqs: bool = True,
    force: bool = False,
    force_mismatch: bool = False,
) -> ProvisionState:
    """Top-level idempotent bootstrap.

    Re-entrant: when called twice with the same ``exo_ref`` and the
    install state matches, all heavy steps are skipped within ~50ms.

    ``force_mismatch`` controls behaviour when the MLX version preflight
    detects a major/minor mismatch between EXO's pinned MLX and Otto's
    bundled MLX.  Default behaviour is to log a loud warning and
    continue — the trap is silent runtime divergence, not provisioning
    failure, and we want the user to be able to make an informed
    choice rather than be blocked.  Pass ``force_mismatch=True`` to
    explicitly suppress the warning when you know what you're doing.
    """
    repo_dir = repo_dir or exo_repo_dir()
    state = load_state()

    install_prereqs(auto=auto_prereqs)

    try:
        commit = ensure_repo(repo_dir, exo_ref, repo_url)

        # MLX-version preflight — runs after the repo is at the
        # requested ref (so we can read its pyproject) but BEFORE
        # ``uv sync`` (so the user can abort and re-run with a
        # different EXO_REF without burning 5 minutes on installs).
        severity, bundled, pinned, message = mlx_version_preflight(repo_dir)
        state.mlx_version_bundled = bundled
        state.mlx_version_pinned = pinned
        # Only surface the warning banner for major/minor mismatches.
        # Patch-level differences (severity == "minor") are "usually safe" and
        # are already logged to the terminal — no need to alarm the user in UI.
        state.mlx_version_warning = message if severity == "major" else ""
        if severity == "major" and not force_mismatch:
            warn(message)
            warn(
                "Continuing anyway — EXO will be installed in its own .venv "
                "so the mismatch only matters if you actually load the same "
                "model in both engines. Pass --force-mismatch to silence "
                "this warning."
            )
        elif severity == "major":
            progress(f"MLX preflight: forced past mismatch ({pinned} vs {bundled}).")
        elif severity == "minor":
            progress(f"MLX preflight: minor diff ({pinned} vs {bundled}) — proceeding.")

        if force or state.deps_installed_for_commit != commit \
                or not (repo_dir / ".venv").exists():
            ensure_uv_sync(repo_dir)
            state.deps_installed_for_commit = commit

        dashboard_built = (repo_dir / "dashboard" / "build").exists()
        if force or state.dashboard_built_for_commit != commit or not dashboard_built:
            ensure_dashboard(repo_dir)
            state.dashboard_built_for_commit = commit

        smoke_test_exo_cli(repo_dir)

        state.exo_ref = exo_ref
        state.git_commit = commit
        state.exo_repo_dir = str(repo_dir)
        state.last_success_at = datetime.now(timezone.utc).isoformat()
        state.last_error = ""
        save_state(state)
        progress(f"exo provisioned at {repo_dir} (commit {commit[:8]})")
        return state

    except SystemExit:
        # `fail()` already printed the error; record it.
        state.last_error = f"provision failed at {datetime.now(timezone.utc).isoformat()}"
        save_state(state)
        raise


# ════════════════════════════════════════════════════════════════════════
#  Daemon lifecycle (start / stop)
# ════════════════════════════════════════════════════════════════════════

def _read_pid() -> Optional[int]:
    p = pid_file()
    if not p.exists():
        return None
    try:
        return int(p.read_text().strip())
    except ValueError:
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False
    return True


def _port_listening(port: int) -> bool:
    return http_alive(f"http://127.0.0.1:{port}/node_id", timeout=1.0)


def _is_macos() -> bool:
    return platform.system() == "Darwin"


def _pid_listening_on(port: int) -> Optional[int]:
    """Return the PID of the process listening on TCP ``port`` (macOS/Linux)."""
    try:
        out = subprocess.check_output(
            ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    for line in out.split():
        line = line.strip()
        if line.isdigit():
            return int(line)
    return None


def is_running(port: int = DEFAULT_EXO_PORT) -> bool:
    pid = _read_pid()
    if pid and _pid_alive(pid):
        return True
    return _port_listening(port)


def start_local(
    *,
    repo_dir: Optional[Path] = None,
    api_port: int = DEFAULT_EXO_PORT,
    libp2p_port: int = DEFAULT_LIBP2P_PORT,
    extra_args: Optional[list[str]] = None,
    wait_seconds: float = 60.0,
    cmd_override: Optional[list[str]] = None,
    cwd_override: Optional[Path] = None,
) -> int:
    """Launch exo in the background. Returns the pid.

    No-ops (returns existing pid) if exo is already running on ``api_port``.

    Prebuilt mode
    -------------
    When ``cmd_override`` is supplied (by :mod:`backend.exo_provisioner` in
    ``prebuilt`` mode) we launch that argv directly instead of ``uv run``,
    skip the source-checkout ``.venv`` requirement, and always take the
    direct ``Popen`` path — the prebuilt runtime runs as a child of the
    signed Otto.app and inherits its Local Network grant, so the Terminal.app
    workaround is neither needed nor wanted.

    macOS Local Network Privacy / TCC
    ---------------------------------

    exo's libp2p discovery uses its own raw UDP multicast socket on port
    5353 (it does *not* use macOS ``mDNSResponder``). On macOS,
    multicast / "Local Network" access is gated per *responsible
    application*: Terminal.app holds the grant after the user clicks
    "Allow" once, but ``sshd`` and IDE-helper-spawned processes do not.
    A daemon launched via ``subprocess.Popen(start_new_session=True)``
    from such a parent silently has its mDNS multicast packets dropped
    by the kernel — the cluster never forms even though every TCP
    listener and every other ``exo`` subsystem looks healthy.

    On macOS we therefore route the launch through
    ``osascript -e 'tell app "Terminal" to do script ...'`` so the daemon
    runs as a child of Terminal.app and inherits its Local Network
    grant. Closing the original shell does not kill it (Terminal.app
    keeps it). Set ``EXO_NO_TERMINAL_WRAP=1`` to bypass this and use
    the direct ``Popen`` path (e.g. when running under Otto.app,
    which will hold its own Local Network grant once shipped).
    """
    prebuilt = cmd_override is not None

    if not prebuilt:
        repo_dir = repo_dir or Path(load_state().exo_repo_dir or exo_repo_dir())
        if not (repo_dir / ".venv").exists():
            fail(f"exo is not provisioned at {repo_dir}. Run `provision` first.")
    else:
        repo_dir = cwd_override or repo_dir or exo_root()

    if is_running(api_port):
        pid = _read_pid() or _pid_listening_on(api_port)
        progress(f"exo already running on :{api_port} (pid={pid}).")
        return pid or 0

    exo_root().mkdir(parents=True, exist_ok=True)

    env = _augmented_install_env()
    if prebuilt:
        cmd = list(cmd_override)
        if extra_args:
            cmd.extend(extra_args)
    else:
        cmd = [
            "uv", "run", "--project", str(repo_dir), "exo",
            "--api-port", str(api_port),
        ]
        if libp2p_port:
            cmd.extend(["--libp2p-port", str(libp2p_port)])
        if extra_args:
            cmd.extend(extra_args)

    # Prebuilt runtime runs under signed Otto.app (holds the Local Network
    # grant) so it never needs the Terminal.app workaround.
    use_terminal = (
        not prebuilt
        and _is_macos()
        and not os.environ.get("EXO_NO_TERMINAL_WRAP")
    )
    if use_terminal:
        return _start_local_via_terminal(
            repo_dir=repo_dir,
            api_port=api_port,
            cmd=cmd,
            env=env,
            wait_seconds=wait_seconds,
        )
    return _start_local_direct(
        repo_dir=repo_dir,
        api_port=api_port,
        cmd=cmd,
        env=env,
        wait_seconds=wait_seconds,
    )


def _start_local_direct(
    *,
    repo_dir: Path,
    api_port: int,
    cmd: list[str],
    env: dict[str, str],
    wait_seconds: float,
) -> int:
    """Launch ``exo`` directly via ``Popen`` (Linux, or opt-out on macOS)."""
    log_fp = open(log_file(), "ab", buffering=0)
    log_fp.write(
        f"\n=== exo start at {datetime.now(timezone.utc).isoformat()} ===\n".encode()
    )
    progress(f"$ {' '.join(shlex.quote(a) for a in cmd)}  &")
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_dir),
        env=env,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    pid_file().write_text(str(proc.pid))
    progress(f"exo started (pid={proc.pid}); logs: {log_file()}")

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            fail(f"exo exited early (rc={proc.returncode}); see {log_file()}")
        if _port_listening(api_port):
            progress(f"exo HTTP API ready on :{api_port}.")
            return proc.pid
        time.sleep(1.0)
    warn(f"exo started (pid={proc.pid}) but HTTP API not ready after {wait_seconds:.0f}s "
         f"— check {log_file()}.")
    return proc.pid


def _start_local_via_terminal(
    *,
    repo_dir: Path,
    api_port: int,
    cmd: list[str],
    env: dict[str, str],
    wait_seconds: float,
) -> int:
    """Launch ``exo`` as a child of Terminal.app (macOS-only).

    See ``start_local`` docstring for why this is necessary on macOS.
    """
    wrapper = exo_root() / "run_exo.sh"
    log_path = log_file()

    inherited = {"PATH", "HOME", "USER", "LANG", "LC_ALL", "TMPDIR"}
    env_lines: list[str] = []
    for k, v in env.items():
        if k in inherited or k.startswith("EXO_") or k.startswith("OTTO_"):
            env_lines.append(f"export {k}={shlex.quote(v)}")
    env_block = "\n".join(env_lines)
    cmd_str = " ".join(shlex.quote(a) for a in cmd)
    quoted_cwd = shlex.quote(str(repo_dir))
    quoted_log = shlex.quote(str(log_path))

    wrapper.write_text(
        "#!/bin/bash\n"
        "set -u\n"
        f"{env_block}\n"
        f"cd {quoted_cwd}\n"
        f"echo \"=== exo start at $(date -u +%FT%TZ) (pid=$$) ===\" "
        f">> {quoted_log}\n"
        f"exec {cmd_str} 2>&1 | tee -a {quoted_log}\n"
    )
    wrapper.chmod(0o755)

    shell_cmd = shlex.quote(str(wrapper))
    apple_script = (
        f'tell application "Terminal" to do script {json.dumps(shell_cmd)}'
    )
    progress("Launching exo via Terminal.app (macOS Local Network grant).")
    progress(f"$ {cmd_str}  (wrapped via Terminal.app)")
    rc = subprocess.run(
        ["osascript", "-e", apple_script],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if rc.returncode != 0:
        stderr = rc.stderr.decode().strip() if rc.stderr else ""
        fail(
            f"osascript launch failed (rc={rc.returncode}): {stderr}\n"
            f"Set EXO_NO_TERMINAL_WRAP=1 to fall back to direct Popen "
            f"(may break macOS mDNS discovery)."
        )

    deadline = time.time() + wait_seconds
    pid: Optional[int] = None
    last_pid_write = 0.0
    while time.time() < deadline:
        pid = _pid_listening_on(api_port)
        if pid and time.time() - last_pid_write > 1.0:
            pid_file().write_text(str(pid))
            last_pid_write = time.time()
        if pid and _port_listening(api_port):
            progress(f"exo started (pid={pid}); logs: {log_path}")
            progress(f"exo HTTP API ready on :{api_port}.")
            return pid
        time.sleep(1.0)

    if pid:
        pid_file().write_text(str(pid))
        warn(f"exo started (pid={pid}) but HTTP API not ready after "
             f"{wait_seconds:.0f}s — check {log_path}.")
        return pid
    fail(
        f"exo did not start within {wait_seconds:.0f}s — check {log_path} "
        f"and the Terminal.app window that just opened."
    )


def stop_local(*, api_port: int = DEFAULT_EXO_PORT, sigterm_grace: float = 8.0) -> bool:
    """Stop the local exo daemon. Returns True if a process was stopped.

    Falls back to ``lsof`` if the pidfile is missing or stale — useful
    after Terminal-wrapped launches where the pidfile is written
    asynchronously and may lag behind the actual listener.
    """
    pid = _read_pid()
    if not pid or not _pid_alive(pid):
        pid = _pid_listening_on(api_port)
    if not pid:
        if _port_listening(api_port):
            warn(f"Port :{api_port} is in use but no pid found; "
                 f"investigate with `lsof -i :{api_port}`.")
            return False
        progress("exo is not running.")
        return False

    progress(f"Stopping exo (pid={pid})…")
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + sigterm_grace
    while time.time() < deadline and _pid_alive(pid):
        time.sleep(0.2)
    if _pid_alive(pid):
        warn(f"pid {pid} did not exit on SIGTERM, sending SIGKILL.")
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    try:
        pid_file().unlink()
    except FileNotFoundError:
        pass
    progress("exo stopped.")
    return True


# ════════════════════════════════════════════════════════════════════════
#  Cluster status (parses /state)
# ════════════════════════════════════════════════════════════════════════

@dataclass
class NodeInfo:
    node_id: str
    chip: Optional[str] = None
    friendly_name: Optional[str] = None
    memory_total_gb: Optional[float] = None
    memory_free_gb: Optional[float] = None


@dataclass
class ClusterStatus:
    reachable: bool
    base_url: str
    master_node_id: Optional[str] = None
    nodes: list[NodeInfo] = field(default_factory=list)
    instances: list[str] = field(default_factory=list)
    runners: list[str] = field(default_factory=list)
    loaded_models: list[str] = field(default_factory=list)
    rdma_connections: int = 0
    raw: Optional[dict] = None

    @property
    def peer_count(self) -> int:
        return len(self.nodes)


def _strip_v1(base_url: str) -> str:
    return base_url.rstrip("/").removesuffix("/v1")


def fetch_cluster_status(base_url: str, *, timeout: float = 4.0) -> ClusterStatus:
    base = _strip_v1(base_url)
    status = ClusterStatus(reachable=False, base_url=base)
    try:
        node_id = http_get_json(f"{base}/node_id", timeout=timeout)
        state = http_get_json(f"{base}/state", timeout=timeout)
    except Exception as exc:
        status.raw = {"error": str(exc)}
        return status

    status.reachable = True
    status.master_node_id = (
        node_id.get("node_id") if isinstance(node_id, dict) else str(node_id)
    )
    topo = state.get("topology") or {}
    identities = state.get("nodeIdentities") or {}
    memory = state.get("nodeMemory") or {}

    def _to_gb(v: Any) -> Optional[float]:
        if isinstance(v, (int, float)) and v > 0:
            return round(float(v) / 1e9, 1)
        return None

    def _mem_bytes(mem_block: Any, *legacy_keys: str) -> Optional[int]:
        """Pull a byte count out of a node memory block.

        Recent exo builds (Otto-bundled) wrap each value as
        ``{"inBytes": N}`` and use ``ramTotal`` / ``ramAvailable`` keys.
        Older builds returned a flat integer keyed as ``total`` /
        ``free`` — try both shapes so this CLI keeps working across
        upgrades.
        """
        if not isinstance(mem_block, dict):
            return None
        for k in ("ramTotal",) + legacy_keys[:1]:
            v = mem_block.get(k)
            if isinstance(v, dict):
                inner = v.get("inBytes")
                if isinstance(inner, (int, float)) and inner > 0:
                    return int(inner)
            elif isinstance(v, (int, float)) and v > 0:
                return int(v)
        return None

    for nid in (topo.get("nodes") or []):
        ident = identities.get(nid) or {}
        mem = memory.get(nid) or {}
        total_bytes = _mem_bytes(mem, "total")
        # ``ramAvailable`` is what we want for the "free" column on
        # newer builds; ``free`` is the legacy fallback.
        free_v = mem.get("ramAvailable")
        if isinstance(free_v, dict):
            free_bytes = free_v.get("inBytes")
        else:
            free_bytes = mem.get("free")
        free_bytes = (
            int(free_bytes)
            if isinstance(free_bytes, (int, float)) and free_bytes > 0
            else None
        )
        status.nodes.append(NodeInfo(
            node_id=nid,
            chip=ident.get("chipId"),
            friendly_name=ident.get("friendlyName"),
            memory_total_gb=_to_gb(total_bytes),
            memory_free_gb=_to_gb(free_bytes),
        ))

    rdma = 0
    for src, sinks in (topo.get("connections") or {}).items():
        for sink, edges in (sinks or {}).items():
            for edge in edges or []:
                if isinstance(edge, dict) and edge.get("sourceRdmaIface"):
                    rdma += 1
    status.rdma_connections = rdma

    instances = state.get("instances") or {}
    status.instances = sorted(instances.keys())

    def _inst_mid(inst: Any) -> Optional[str]:
        """Walk the (variant-wrapped) instance dict for its model id."""
        if not isinstance(inst, dict):
            return None
        direct = inst.get("modelId") or inst.get("model_id") or inst.get("model")
        if isinstance(direct, str) and direct:
            return direct
        for v in inst.values():
            if not isinstance(v, dict):
                continue
            sa = v.get("shardAssignments")
            if isinstance(sa, dict):
                mid = sa.get("modelId") or sa.get("model_id")
                if isinstance(mid, str) and mid:
                    return mid
            mid = v.get("modelId") or v.get("model_id") or v.get("model")
            if isinstance(mid, str) and mid:
                return mid
        return None

    status.loaded_models = sorted(
        {
            mid
            for inst in instances.values()
            if (mid := _inst_mid(inst)) is not None
        }
    )
    status.runners = sorted((state.get("runners") or {}).keys())
    status.raw = state
    return status


def format_status(s: ClusterStatus) -> str:
    if not s.reachable:
        err = ""
        if isinstance(s.raw, dict) and s.raw.get("error"):
            err = f" ({s.raw['error']})"
        return f"exo NOT reachable at {s.base_url}{err}"

    lines = [
        f"exo reachable at {s.base_url}",
        f"  master_node_id : {s.master_node_id}",
        f"  nodes          : {len(s.nodes)}",
    ]
    for n in s.nodes:
        chip = n.chip or "?"
        name = n.friendly_name or "?"
        mem = (
            f"{n.memory_free_gb}/{n.memory_total_gb} GB free"
            if n.memory_total_gb else "memory:?"
        )
        lines.append(f"    - {n.node_id[:14]}…  {name}  ({chip}, {mem})")
    lines.append(f"  rdma edges     : {s.rdma_connections}")
    lines.append(f"  instances      : {len(s.instances)}")
    lines.append(f"  loaded models  : {', '.join(s.loaded_models) or '(none)'}")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
#  Smoke test
# ════════════════════════════════════════════════════════════════════════

def smoke_test(base_url: str) -> int:
    base = _strip_v1(base_url)
    fails = 0
    print()
    progress("Running smoke tests…")

    def check(name: str, fn: Callable[[], Any]) -> None:
        nonlocal fails
        try:
            res = fn()
            print(f"  PASS  {name}: {res}")
        except Exception as exc:
            fails += 1
            print(f"  FAIL  {name}: {exc}")

    def _check_node_id() -> str:
        nid = http_get_json(f"{base}/node_id")
        if isinstance(nid, dict):
            nid = nid.get("node_id") or nid.get("id") or ""
        nid = str(nid)
        return (nid[:16] + "…") if len(nid) > 16 else (nid or "(empty)")

    check("GET /node_id", _check_node_id)
    check("GET /state", lambda: f"{len((http_get_json(f'{base}/state').get('topology') or {}).get('nodes') or [])} node(s)")
    check("GET /v1/models", lambda: f"{len(http_get_json(f'{base}/v1/models').get('data') or [])} models")

    try:
        models = (http_get_json(f"{base}/v1/models?status=downloaded").get("data") or [])
        if models:
            check("GET /v1/models?status=downloaded",
                  lambda: f"first id={models[0].get('id')}")
        else:
            print("  SKIP  /v1/chat/completions: no downloaded models in cluster")
    except Exception as exc:
        fails += 1
        print(f"  FAIL  GET /v1/models?status=downloaded: {exc}")
        models = []

    if models:
        first_model = models[0].get("id")
        # exo also exposes loaded model instances; pick any model with a
        # loaded instance so the chat call doesn't trigger a full download.
        loaded_ids = []
        try:
            instances = (http_get_json(f"{base}/state").get("instances") or {})
            loaded_ids = [
                inst.get("modelId") or inst.get("model_id") or inst.get("model")
                for inst in instances.values()
                if isinstance(inst, dict)
            ]
            loaded_ids = [m for m in loaded_ids if m]
        except Exception:
            pass

        target = next((m for m in loaded_ids if m), first_model)
        if loaded_ids:
            check(
                f"POST /v1/chat/completions ({target})",
                lambda: (
                    http_post_json(
                        f"{base}/v1/chat/completions",
                        {
                            "model": target,
                            "messages": [{"role": "user", "content": "ping"}],
                            "max_tokens": 5,
                            "stream": False,
                        },
                        timeout=120.0,
                    ).get("choices", [{}])[0].get("message", {}).get("content", "")[:40]
                    or "(empty)"
                ),
            )
        else:
            print(f"  SKIP  /v1/chat/completions: no loaded model instances "
                  f"(first available model: {first_model})")

    print()
    if fails:
        progress(f"smoke: {fails} check(s) failed")
        return 1
    progress("smoke: all checks passed")
    return 0


# ════════════════════════════════════════════════════════════════════════
#  Remote (SSH) orchestration
# ════════════════════════════════════════════════════════════════════════

def run_remote(
    ssh_alias: str,
    subcommand: list[str],
    *,
    forward_env: Iterable[str] = (
        "EXO_REF", "EXO_REPO_URL", "EXO_REPO_DIR",
        "EXO_BASE_URL", "EXO_API_PORT", "EXO_LIBP2P_PORT",
        "EXO_NODE_PORT", "EXO_NO_TERMINAL_WRAP",
        "OTTO_APP_DATA_DIR",
    ),
) -> int:
    """SCP this module to the remote, then run ``python3 ~/.exo_cli.py …``.

    Streams stdout+stderr line by line so the master shows live progress.
    The shipped file is the implementation module (this file), not whatever
    user-facing shim launched the call — that way a one-liner shim at
    ``scripts/exo_cli.py`` still scps the full self-contained code.
    """
    me = Path(__file__).resolve()
    progress(f"Copying {me.name} to {ssh_alias}:{REMOTE_SCRIPT_PATH}…")
    rc = run_streaming(
        "scp",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        "-q", str(me), f"{ssh_alias}:{REMOTE_SCRIPT_PATH}",
        check=False, label="scp/script",
    )
    if rc != 0:
        fail(f"Could not scp script to {ssh_alias} (rc={rc})")

    env_pairs = [
        f"{k}={shlex.quote(os.environ[k])}"
        for k in forward_env
        if os.environ.get(k)
    ]
    env_prefix = " ".join(env_pairs)
    remote_cmd = (
        (env_prefix + " ") if env_prefix else ""
    ) + f"python3 {REMOTE_SCRIPT_PATH} " + " ".join(shlex.quote(a) for a in subcommand)

    progress(f"Running on {ssh_alias}: {remote_cmd}")
    return run_streaming(
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=accept-new",
        ssh_alias,
        remote_cmd,
        check=False, label=f"ssh/{ssh_alias}",
    )


# ════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════

def _add_common_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--remote", default=os.environ.get("EXO_REMOTE_SSH"),
                   help="Run this command on a remote host via SSH "
                        "(value is your ~/.ssh/config alias). "
                        "Defaults to $EXO_REMOTE_SSH.")
    p.add_argument("--app-data-dir", default=os.environ.get("OTTO_APP_DATA_DIR"),
                   help="Override the per-user app data dir "
                        "(also overridable via $OTTO_APP_DATA_DIR).")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exo_cli.py",
        description="Provision, supervise, and inspect an exo cluster for the Otto app.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_info = sub.add_parser("info", help="Print resolved paths and detected prereqs.")
    _add_common_flags(p_info)

    p_doctor = sub.add_parser("doctor", help="Check prereqs (no installs).")
    _add_common_flags(p_doctor)

    p_status = sub.add_parser("status", help="Hit the local exo HTTP API and print state.")
    _add_common_flags(p_status)
    p_status.add_argument("--base-url", default=os.environ.get("EXO_BASE_URL", f"http://localhost:{DEFAULT_EXO_PORT}"))
    p_status.add_argument("--json", action="store_true", help="Emit JSON instead of human format.")

    p_prov = sub.add_parser("provision", help="Idempotent install: clone + uv sync + dashboard.")
    _add_common_flags(p_prov)
    p_prov.add_argument("--ref", default=os.environ.get("EXO_REF", DEFAULT_EXO_REF))
    p_prov.add_argument("--repo-url", default=os.environ.get("EXO_REPO_URL", DEFAULT_EXO_REPO_URL))
    p_prov.add_argument("--force", action="store_true",
                        help="Re-run uv sync + dashboard even if state matches.")
    p_prov.add_argument("--no-auto-prereqs", action="store_true",
                        help="Fail on missing prereqs instead of installing them.")
    p_prov.add_argument("--force-mismatch", action="store_true",
                        help="Suppress the MLX-version preflight warning when EXO's "
                             "pinned MLX differs from Otto's bundled MLX.")

    p_start = sub.add_parser("start", help="Launch local exo daemon (background).")
    _add_common_flags(p_start)
    p_start.add_argument("--api-port", type=int, default=_env_api_port(),
                         help="HTTP API port (default 52415, env: EXO_API_PORT).")
    p_start.add_argument("--libp2p-port", type=int, default=_env_libp2p_port(),
                         help="Fixed libp2p TCP port (0 = OS-assigned, env: EXO_LIBP2P_PORT).")
    p_start.add_argument("--wait", type=float, default=60.0,
                         help="Seconds to wait for the HTTP API to come up.")
    p_start.add_argument("extra_args", nargs="*", help="Extra args to forward to `exo`.")

    p_stop = sub.add_parser("stop", help="Stop the local exo daemon.")
    _add_common_flags(p_stop)
    p_stop.add_argument("--api-port", type=int, default=_env_api_port())

    p_up = sub.add_parser("up", help="provision + start.")
    _add_common_flags(p_up)
    p_up.add_argument("--ref", default=os.environ.get("EXO_REF", DEFAULT_EXO_REF))
    p_up.add_argument("--repo-url", default=os.environ.get("EXO_REPO_URL", DEFAULT_EXO_REPO_URL))
    p_up.add_argument("--force", action="store_true")
    p_up.add_argument("--no-auto-prereqs", action="store_true")
    p_up.add_argument("--force-mismatch", action="store_true",
                      help="Suppress the MLX-version preflight warning when EXO's "
                           "pinned MLX differs from Otto's bundled MLX.")
    p_up.add_argument("--api-port", type=int, default=_env_api_port())
    p_up.add_argument("--libp2p-port", type=int, default=_env_libp2p_port())
    p_up.add_argument("--wait", type=float, default=60.0)

    p_down = sub.add_parser("down", help="stop.")
    _add_common_flags(p_down)
    p_down.add_argument("--api-port", type=int, default=_env_api_port())

    p_smoke = sub.add_parser("smoke", help="Battery of API checks against a running cluster.")
    _add_common_flags(p_smoke)
    p_smoke.add_argument("--base-url", default=os.environ.get("EXO_BASE_URL", f"http://localhost:{DEFAULT_EXO_PORT}"))

    return parser


def _apply_app_data_override(args: argparse.Namespace) -> None:
    if getattr(args, "app_data_dir", None):
        os.environ["OTTO_APP_DATA_DIR"] = args.app_data_dir


def cmd_info(args: argparse.Namespace) -> int:
    state = load_state()
    p = detect_prereqs()
    print(json.dumps({
        "platform": platform.system(),
        "python": sys.version.split()[0],
        "app_data_dir": str(get_app_data_dir()),
        "exo_root": str(exo_root()),
        "exo_repo_dir": str(exo_repo_dir()),
        "state_file": str(state_file()),
        "pid_file": str(pid_file()),
        "log_file": str(log_file()),
        "state": asdict(state),
        "prereqs": asdict(p),
        "running": is_running(),
    }, indent=2))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    p = detect_prereqs()
    print(f"Platform        : {p.platform}")
    print(f"  brew          : {p.brew or '(missing)'}")
    print(f"  uv            : {p.uv or '(missing)'}")
    print(f"  node          : {p.node or '(missing)'}")
    print(f"  npm           : {p.npm or '(missing)'}")
    print(f"  git           : {p.git or '(missing)'}")
    print(f"  rustup        : {p.rustup or '(missing)'}")
    print(f"  cargo         : {p.cargo or '(missing)'}")
    print(f"  rust-nightly  : {'yes' if p.rust_nightly else 'no'}")
    missing = p.missing()
    if missing:
        print(f"\nMissing: {', '.join(missing)}")
        print("Run `provision` to install (or pass --no-auto-prereqs to fail loudly).")
        return 1
    print("\nAll prereqs present.")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    s = fetch_cluster_status(args.base_url)
    if args.json:
        out = {
            "reachable": s.reachable,
            "base_url": s.base_url,
            "master_node_id": s.master_node_id,
            "peer_count": s.peer_count,
            "rdma_connections": s.rdma_connections,
            "loaded_models": s.loaded_models,
            "instances": s.instances,
            "runners": s.runners,
            "nodes": [asdict(n) for n in s.nodes],
        }
        print(json.dumps(out, indent=2))
    else:
        print(format_status(s))
    return 0 if s.reachable else 2


def cmd_provision(args: argparse.Namespace) -> int:
    provision_exo(
        exo_ref=args.ref,
        repo_url=args.repo_url,
        force=args.force,
        auto_prereqs=not args.no_auto_prereqs,
        force_mismatch=getattr(args, "force_mismatch", False),
    )
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    start_local(
        api_port=args.api_port,
        libp2p_port=args.libp2p_port,
        extra_args=list(args.extra_args or []),
        wait_seconds=args.wait,
    )
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    stop_local(api_port=args.api_port)
    return 0


def cmd_up(args: argparse.Namespace) -> int:
    provision_exo(
        exo_ref=args.ref,
        repo_url=args.repo_url,
        force=args.force,
        auto_prereqs=not args.no_auto_prereqs,
        force_mismatch=getattr(args, "force_mismatch", False),
    )
    start_local(
        api_port=args.api_port,
        libp2p_port=args.libp2p_port,
        wait_seconds=args.wait,
    )
    return 0


def cmd_down(args: argparse.Namespace) -> int:
    stop_local(api_port=args.api_port)
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    return smoke_test(args.base_url)


COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "info": cmd_info,
    "doctor": cmd_doctor,
    "status": cmd_status,
    "provision": cmd_provision,
    "start": cmd_start,
    "stop": cmd_stop,
    "up": cmd_up,
    "down": cmd_down,
    "smoke": cmd_smoke,
}


def main(argv: Optional[list[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = _build_parser()
    args = parser.parse_args(argv)
    _apply_app_data_override(args)

    # `--remote ALIAS` short-circuits: re-invoke ourselves on the secondary.
    remote = getattr(args, "remote", None)
    if remote:
        # Strip --remote from the forwarded argv so the remote runs locally.
        forwarded: list[str] = []
        skip_next = False
        for a in argv:
            if skip_next:
                skip_next = False
                continue
            if a == "--remote":
                skip_next = True
                continue
            if a.startswith("--remote="):
                continue
            forwarded.append(a)
        return run_remote(remote, forwarded)

    handler = COMMANDS[args.cmd]
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
