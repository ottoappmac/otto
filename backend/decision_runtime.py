"""Lifecycle for the local OpenJev sidecar.

OpenJev is installed into its own ``uv`` venv under the app-data dir
because its MLX pins conflict with Otto's. The process is started lazily
on the first ``jev`` call, warmed with one tiny read, and stopped
after ``idle_stop_minutes`` so the resident set is given back.

The server is started with ``HF_HUB_OFFLINE=1`` and a local snapshot
path. A startup that phones the Hub would break the privacy lock.
"""

from __future__ import annotations

import logging
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from backend.config import AppConfig, DecisionConfig, get_app_data_dir
from backend.decision_catalog import get_model, score_model
from backend.decision_client import is_loopback_url
from backend.setup_capabilities import probe_capabilities

logger = logging.getLogger(__name__)

_LOCAL_KEY_ACCOUNT = "openjev_local_api_key"
_REMOTE_KEY_ACCOUNT = "openjev_api_key"
_GIT_URL = "https://github.com/razorback16/openjev.git"

_lock = threading.Lock()
_proc: subprocess.Popen[bytes] | None = None
_last_used = 0.0
_watcher_started = False
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def runtime_dir() -> Path:
    return get_app_data_dir() / "openjev"


def bundled_runtime_dir() -> Path | None:
    """OpenJev tree copied into the packaged .app, when this process is that app.

    Layout is ``Contents/Resources/backend/<exe>`` beside
    ``Contents/Resources/openjev/venv``. Logs and the API key stay under
    :func:`runtime_dir`, which is writable. The bundle is not.
    """
    exe = Path(sys.executable).resolve()
    candidate = exe.parent.parent / "openjev"
    if (candidate / "venv" / "bin" / "python").is_file():
        return candidate
    return None


def venv_dir() -> Path:
    bundled = bundled_runtime_dir()
    if bundled is not None:
        return bundled / "venv"
    return runtime_dir() / ".venv"


def venv_python() -> Path:
    return venv_dir() / "bin" / "python"


def log_path() -> Path:
    return runtime_dir() / "server.log"


def installed_ref_path() -> Path:
    return runtime_dir() / "INSTALLED_REF"


def hf_repo_dir(repo_id: str) -> Path:
    slug = "models--" + repo_id.replace("/", "--")
    return Path.home() / ".cache" / "huggingface" / "hub" / slug


def snapshot_dir(repo_id: str) -> Path | None:
    """Local HF snapshot for ``repo_id``, or None when it is not complete.

    Complete means weights, ``config.json``, and ``chat_template.jinja``.
    """
    root = hf_repo_dir(repo_id)
    candidates: list[Path] = []
    ref = root / "refs" / "main"
    if ref.is_file():
        rev = ref.read_text(encoding="utf-8").strip()
        if rev:
            candidates.append(root / "snapshots" / rev)
    snaps = root / "snapshots"
    if snaps.is_dir():
        candidates.extend(p for p in snaps.iterdir() if p.is_dir())
    seen: set[Path] = set()
    for snap in candidates:
        if snap in seen:
            continue
        seen.add(snap)
        # DiffusionGemma keeps its chat template in chat_template.jinja.
        # Weights alone still 500 on the first /v1/systemone call.
        if (
            (snap / "config.json").is_file()
            and (snap / "chat_template.jinja").is_file()
            and any(snap.glob("*.safetensors"))
        ):
            return snap
    return None


def model_cached(repo_id: str) -> bool:
    return snapshot_dir(repo_id) is not None


def _fit(cfg: DecisionConfig, hub_cache: str) -> str:
    row = get_model(cfg.model_repo)
    if row is None:
        return "unknown"
    caps = probe_capabilities(hub_cache)
    return str(score_model(
        row,
        ram_gb=float(caps.get("ram_gb") or 0),
        wired_limit_gb=float(caps.get("wired_limit_gb") or 0),
        free_disk_gb=float(caps.get("free_disk_gb") or 0),
    )["fits"])


def local_base_url(cfg: DecisionConfig) -> str:
    return f"http://127.0.0.1:{int(cfg.port)}"


def effective_base_url(cfg: DecisionConfig) -> str:
    override = (cfg.endpoint_override or "").strip()
    if override:
        return override.rstrip("/")
    return local_base_url(cfg)


def uses_local_sidecar(cfg: DecisionConfig) -> bool:
    override = (cfg.endpoint_override or "").strip()
    return not override or is_loopback_url(override)


def _vault_get(account: str) -> str | None:
    try:
        from backend.credential_vault import app_vault
        value = app_vault.get(account)
        return value if value else None
    except Exception:
        logger.debug("decision vault read failed for %s", account, exc_info=True)
        return None


def _vault_set(account: str, value: str) -> None:
    from backend.credential_vault import app_vault
    app_vault.set(account, value)


def _key_file(name: str) -> Path:
    return runtime_dir() / name


def _read_key_file(path: Path) -> str | None:
    try:
        stored = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return stored or None


def _write_key_file(path: Path, key: str) -> None:
    runtime_dir().mkdir(parents=True, exist_ok=True)
    path.write_text(key + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def local_api_key() -> str:
    """Bearer token the sidecar requires. Generated once.

    Stored beside the sidecar, not in the app keychain bundle. A Settings
    save replaces that bundle and would otherwise mint a new key while the
    already-running server kept the old one (HTTP 401).
    """
    path = _key_file("local_api_key")
    existing = _read_key_file(path)
    if existing:
        return existing
    migrated = _vault_get(_LOCAL_KEY_ACCOUNT)
    key = migrated or ("ojv_" + secrets.token_urlsafe(24))
    _write_key_file(path, key)
    return key


def remote_api_key() -> str | None:
    """Key for a custom endpoint. File first, then the vault for older installs."""
    stored = _read_key_file(_key_file("remote_api_key"))
    if stored:
        return stored
    return _vault_get(_REMOTE_KEY_ACCOUNT)


def set_remote_api_key(value: str) -> None:
    _write_key_file(_key_file("remote_api_key"), value.strip())


def remote_api_key_set() -> bool:
    return bool(remote_api_key())


def build_process_env(
    cfg: DecisionConfig,
    snapshot: Path,
    *,
    api_key: str | None = None,
) -> dict[str, str]:
    """Env for ``python -m openjev``. Host is always loopback; Hub is offline."""
    env = os.environ.copy()
    env.update({
        "OPENJEV_BACKEND": "mlx",
        "OPENJEV_HOST": "127.0.0.1",
        "OPENJEV_PORT": str(int(cfg.port)),
        "OPENJEV_MLX_MODEL": str(snapshot),
        "OPENJEV_MLX_CACHE_LIMIT_GB": str(cfg.mlx_cache_limit_gb),
        "OPENJEV_API_KEY": api_key if api_key is not None else local_api_key(),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    })
    return env


def installed() -> bool:
    return venv_python().is_file() and bool(installed_ref())


def installed_ref() -> str:
    bundled = bundled_runtime_dir()
    path = (bundled / "INSTALLED_REF") if bundled is not None else installed_ref_path()
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


def runtime_current(cfg: DecisionConfig | None = None) -> bool:
    """True when the sidecar venv exists and matches the pinned commit."""
    cfg = cfg or DecisionConfig()
    return installed() and installed_ref() == (cfg.runtime_ref or "").strip()


def _job(kind: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "kind": kind,
            "status": "running",
            "message": "",
            "error": None,
        }
    return job_id


def _finish_job(job_id: str, *, ok: bool, message: str) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job["status"] = "done" if ok else "error"
        job["message"] = message
        job["error"] = None if ok else message


def get_job(job_id: str) -> dict[str, Any] | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def _port_open(port: int) -> bool:
    import socket
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=0.3):
            return True
    except OSError:
        return False


def running(cfg: DecisionConfig | None = None) -> bool:
    cfg = cfg or AppConfig.load().decision
    with _lock:
        proc = _proc
    if proc is not None and proc.poll() is None:
        return True
    return _port_open(cfg.port)


def status_snapshot(cfg: DecisionConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AppConfig.load().decision
    hub = str(Path.home() / ".cache" / "huggingface" / "hub")
    with _lock:
        proc = _proc
        pid = proc.pid if proc is not None and proc.poll() is None else None
    return {
        "installed": installed(),
        "installed_ref": installed_ref(),
        "running": running(cfg),
        "pid": pid,
        "port": int(cfg.port),
        "model_repo": cfg.model_repo,
        "model_cached": model_cached(cfg.model_repo),
        "fit": _fit(cfg, hub),
        "endpoint": effective_base_url(cfg),
        "local": uses_local_sidecar(cfg),
        "remote_key_set": remote_api_key_set(),
    }


def tail_log(max_lines: int = 200) -> str:
    path = log_path()
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max(1, max_lines):])


def install_async(cfg: DecisionConfig | None = None) -> str:
    cfg = cfg or AppConfig.load().decision
    job_id = _job("install")

    def _run() -> None:
        try:
            message = install_sync(cfg)
        except Exception as exc:
            logger.exception("openjev install failed")
            _finish_job(job_id, ok=False, message=str(exc))
            return
        _finish_job(job_id, ok=True, message=message)

    threading.Thread(target=_run, name="openjev-install", daemon=True).start()
    return job_id


def install_sync(cfg: DecisionConfig) -> str:
    from backend.mcp_builder import _find_uv, _run_uv
    import asyncio

    uv = _find_uv()
    if not uv:
        raise RuntimeError(
            "uv is not installed, so the Jev runtime cannot be created. "
            "Install uv and try again from Settings → Jev."
        )
    runtime_dir().mkdir(parents=True, exist_ok=True)
    # Always the writable app-data venv. The copy inside a signed .app
    # is produced by scripts/build_openjev.py and must not be modified.
    managed = runtime_dir() / ".venv"
    py = managed / "bin" / "python"
    if not py.is_file():
        code, out, err = asyncio.run(_run_uv(
            uv, "venv", "--python", "3.11", str(managed),
        ))
        if code != 0:
            raise RuntimeError(err.strip() or out.strip() or "uv venv failed")
    ref = (cfg.runtime_ref or "").strip()
    spec = f"openjev[mlx] @ git+{_GIT_URL}@{ref}"
    code, out, err = asyncio.run(_run_uv(
        uv, "pip", "install", "--python", str(py), spec,
    ))
    if code != 0:
        raise RuntimeError(err.strip() or out.strip() or "uv pip install failed")
    installed_ref_path().write_text(ref + "\n", encoding="utf-8")
    return f"Installed openjev {ref[:12]}"


def _terminate(proc: subprocess.Popen[bytes]) -> None:
    """Stop a sidecar process group. No-op if it has already exited."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            proc.kill()
        proc.wait(timeout=5)


def _listener_pids(port: int) -> list[int]:
    """PIDs listening on ``port``. Used to stop a sidecar this process did not spawn."""
    try:
        out = subprocess.check_output(
            ["lsof", "-nP", f"-iTCP:{int(port)}", "-sTCP:LISTEN", "-t"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return []
    pids: list[int] = []
    for line in out.split():
        try:
            pids.append(int(line))
        except ValueError:
            continue
    return pids


def stop_sync() -> str:
    """Stop the sidecar, including one left behind by a backend reload."""
    global _proc
    with _lock:
        proc = _proc
        _proc = None
    if proc is not None and proc.poll() is None:
        _terminate(proc)
        return "Decision server stopped."
    try:
        port = int(AppConfig.load().decision.port)
    except Exception:
        port = 52416
    pids = _listener_pids(port)
    if not pids:
        return "Decision server is not running."
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline and _port_open(port):
        time.sleep(0.2)
    return "Decision server stopped."


def stop_async() -> str:
    job_id = _job("stop")

    def _run() -> None:
        try:
            message = stop_sync()
        except Exception as exc:
            _finish_job(job_id, ok=False, message=str(exc))
            return
        _finish_job(job_id, ok=True, message=message)

    threading.Thread(target=_run, name="openjev-stop", daemon=True).start()
    return job_id


def _mark_used() -> None:
    global _last_used
    _last_used = time.monotonic()


def _ensure_watcher() -> None:
    global _watcher_started
    with _lock:
        if _watcher_started:
            return
        _watcher_started = True
    threading.Thread(target=_idle_loop, name="openjev-idle", daemon=True).start()


def _idle_loop() -> None:
    while True:
        time.sleep(30)
        try:
            cfg = AppConfig.load().decision
        except Exception:
            continue
        minutes = max(1, int(cfg.idle_stop_minutes or 10))
        with _lock:
            proc = _proc
            last = _last_used
        if proc is None or proc.poll() is not None:
            continue
        if last and time.monotonic() - last > minutes * 60:
            logger.info("decision sidecar idle for %s min; stopping", minutes)
            try:
                stop_sync()
            except Exception:
                logger.debug("decision idle stop failed", exc_info=True)


def _wait_until_ready(base_url: str, api_key: str, timeout: float = 120.0) -> None:
    deadline = time.monotonic() + timeout
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    last_error = "timed out waiting for the Jev server"
    while time.monotonic() < deadline:
        try:
            response = httpx.get(base_url.rstrip("/") + "/v1/models", headers=headers, timeout=2.0)
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}"
        except httpx.HTTPError as exc:
            last_error = str(exc)
        time.sleep(0.4)
    raise RuntimeError(last_error)


def _warmup(base_url: str, api_key: str) -> None:
    from backend.decision_client import ask
    ask(
        "ping",
        "Is this a warmup ping?",
        base_url=base_url,
        api_key=api_key,
        timeout=60,
    )


def start_sync(cfg: DecisionConfig | None = None) -> str:
    """Start the sidecar if needed and run one warmup read. Blocks until ready."""
    global _proc
    cfg = cfg or AppConfig.load().decision
    if not uses_local_sidecar(cfg):
        return "Using a custom endpoint; the local server was not started."
    fit = _fit(cfg, str(Path.home() / ".cache" / "huggingface" / "hub"))
    if fit == "over":
        raise RuntimeError(
            f"{cfg.model_repo} does not fit beside a chat model on this Mac. "
            "Pick a smaller Jev model in Settings → Jev."
        )
    snap = snapshot_dir(cfg.model_repo)
    if snap is None:
        raise RuntimeError(
            f"{cfg.model_repo} is not downloaded. Open Settings → Jev and download it."
        )
    if not installed() or installed_ref() != (cfg.runtime_ref or "").strip():
        raise RuntimeError(
            "The Jev runtime is not installed. Open Settings → Jev and install it."
        )
    base = local_base_url(cfg)
    key = local_api_key()
    with _lock:
        proc = _proc
        if proc is not None and proc.poll() is None:
            already = True
        elif _port_open(cfg.port):
            already = True
        else:
            already = False
            runtime_dir().mkdir(parents=True, exist_ok=True)
            log = open(log_path(), "ab")
            proc = subprocess.Popen(
                [str(venv_python()), "-m", "openjev"],
                cwd=str(runtime_dir()),
                env=build_process_env(cfg, snap, api_key=key),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _proc = proc
    _ensure_watcher()
    _mark_used()
    if not already:
        try:
            _wait_until_ready(base, key)
            _warmup(base, key)
        except Exception:
            stop_sync()
            raise
    return f"Decision server ready at {base}"


def start_async(cfg: DecisionConfig | None = None) -> str:
    cfg = cfg or AppConfig.load().decision
    job_id = _job("start")

    def _run() -> None:
        try:
            message = start_sync(cfg)
        except Exception as exc:
            logger.exception("openjev start failed")
            _finish_job(job_id, ok=False, message=str(exc))
            return
        _finish_job(job_id, ok=True, message=message)

    threading.Thread(target=_run, name="openjev-start", daemon=True).start()
    return job_id


def ensure_running(cfg: DecisionConfig | None = None) -> str:
    """Called by the ``jev`` tool. Starts the sidecar on first use."""
    cfg = cfg or AppConfig.load().decision
    if not uses_local_sidecar(cfg):
        return effective_base_url(cfg)
    if running(cfg):
        _mark_used()
        _ensure_watcher()
        return local_base_url(cfg)
    start_sync(cfg)
    return local_base_url(cfg)


def note_used() -> None:
    _mark_used()
