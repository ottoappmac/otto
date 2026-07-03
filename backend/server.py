"""FastAPI backend for the Deep Agent desktop app."""

from __future__ import annotations

import asyncio
import concurrent.futures
import faulthandler
import logging
import os
import shutil
import signal
import time
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from typing import AsyncGenerator

faulthandler.enable()
if hasattr(signal, "SIGUSR1"):
    faulthandler.register(signal.SIGUSR1)

# Use the operating system's trust store for TLS verification.  On machines
# behind a corporate TLS-inspection proxy (Zscaler, Netskope, etc.) the
# intercepting root CA lives in the OS keychain but NOT in certifi's bundle,
# so certifi-based verification fails with CERTIFICATE_VERIFY_FAILED while
# curl (which uses the system trust) succeeds.  ``truststore`` patches the
# stdlib ``ssl`` module to consult the OS trust store, which fixes outbound
# calls to e.g. huggingface.co (the MLX catalog discovery).  Must run before
# any SSLContext is created.
try:
    import truststore

    truststore.inject_into_ssl()
except Exception:  # noqa: BLE001 — never let TLS setup crash startup
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import AppConfig, get_app_data_dir
from backend.agent_library import seed_defaults
from backend.state import mcp_mgr, session_mgr

from backend.routes.settings import router as settings_router
from backend.routes.mcp import router as mcp_router
from backend.routes.mcp_config import router as mcp_config_router
from backend.routes.agents import router as agents_router
from backend.routes.sessions import router as sessions_router, ws_router
from backend.routes.schedules import router as schedules_router
from backend.routes.triggers import router as triggers_router
from backend.routes.hooks import router as hooks_router
from backend.routes.openclaw_hooks import router as openclaw_hooks_router
from backend.routes.memory import router as memory_router
from backend.routes.embeddings import router as embeddings_router
from backend.routes.mlx import router as mlx_router
from backend.routes.exo import router as exo_router
from backend.routes.node import router as node_router
from backend.routes.omlx import router as omlx_router
from backend.routes.privacy import router as privacy_router
from backend.routes.vault import router as vault_router
from backend.routes.activity import router as activity_router
from backend.routes.setup import router as setup_router
from backend.routes.ambient import router as ambient_router
from backend.routes.voice import router as voice_router, ws_router as voice_ws_router
from backend.routes.runs import router as runs_router
from backend.activity_tracker import tracker as activity_tracker
from backend.scheduler import init_scheduler, shutdown_scheduler
from backend.trigger_manager import init_trigger_manager
import backend.summarization_guard  # noqa: F401 — installs orphan-ToolMessage guard

logger = logging.getLogger(__name__)

_LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
_LOG_BACKUP_COUNT = 5


def _attach_file_handler() -> None:
    """Add a rotating file handler to the root logger so all backend logs
    are persisted under the platform app-data directory."""
    try:
        log_dir = get_app_data_dir() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            log_dir / "backend.log",
            maxBytes=_LOG_MAX_BYTES,
            backupCount=_LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        ))
        root = logging.getLogger()
        if root.level > logging.INFO:
            root.setLevel(logging.INFO)
        root.addHandler(handler)
    except OSError:
        logger.warning("Could not create log file — file logging disabled")


def _suppress_mcp_cleanup_noise(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    """Silence noisy MCP cleanup tracebacks from the asyncio event loop handler."""
    exc = context.get("exception")
    if exc and isinstance(exc, (RuntimeError, BaseExceptionGroup)):
        msg = str(exc)
        if "cancel scope" in msg or "exit cancel scope" in msg:
            return
        if isinstance(exc, BaseExceptionGroup):
            return
    loop.default_exception_handler(context)


class _MCPNoiseFilter(logging.Filter):
    """Suppress 'Task exception was never retrieved' log records that come from
    MCP client cleanup.  These are emitted by ``asyncio.Task.__del__`` and bypass
    the event loop exception handler, so a logging filter is the only way to
    silence them."""

    _NOISE_FRAGMENTS = (
        "Task exception was never retrieved",
        "cancel scope",
        "GeneratorExit",
        "streamablehttp_client",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if any(frag in msg for frag in self._NOISE_FRAGMENTS):
            return False
        return True


def _uv_is_present() -> bool:
    """Whether a usable ``uv`` binary is discoverable.

    Mirrors the resolution order in :func:`backend.mcp_builder._find_uv`
    (PATH first, then the official-installer + Homebrew locations) so the
    presence check and the consumers agree.
    """
    import shutil
    from pathlib import Path

    if shutil.which("uv"):
        return True
    for cand in (
        Path.home() / ".local" / "bin" / "uv",
        Path("/opt/homebrew/bin/uv"),
        Path("/usr/local/bin/uv"),
    ):
        if cand.exists():
            return True
    return False


async def _ensure_uv_present() -> None:
    """Install ``uv`` silently at startup if it is not already available.

    ``uv`` provisions the per-MCP virtualenvs used by built-in and
    agent-generated MCP servers.  On a fresh VM without Homebrew it is
    absent, so install it via :func:`backend.tool_provisioner.ainstall_uv`
    (brew fast path, else the official ``astral.sh`` installer into
    ``~/.local/bin``).  Best-effort and time-boxed — never raises.
    """
    if _uv_is_present():
        logger.info("Auto-provision: uv already present — nothing to do")
        return

    logger.info("Auto-provision: uv not found — installing in background…")
    try:
        from backend.tool_provisioner import ainstall_uv
        job = await ainstall_uv()
    except Exception as exc:
        logger.warning("Auto-provision: could not start uv install: %s", exc)
        return

    deadline = asyncio.get_event_loop().time() + 600
    while job.status == "running":
        if asyncio.get_event_loop().time() >= deadline:
            logger.warning("Auto-provision: uv install timed out after 10 min")
            return
        await asyncio.sleep(5)

    if job.status == "done":
        logger.info("Auto-provision: uv installed")
    else:
        logger.warning(
            "Auto-provision: uv install ended with status=%s error=%s",
            job.status, job.error,
        )


async def _startup_mcp(cfg: AppConfig) -> None:
    """Start and connect all enabled MCP servers in the background.

    Runs after the lifespan yields so it never blocks the health endpoint.
    Servers with auto_start are started first (sequentially, so process ports
    don't race), then all connections are established in parallel.

    Provisions per-MCP venvs for source-bundled built-in MCPs (e.g.
    ``edgar-sec``) up-front, since the stdio MCP connection in
    :mod:`backend.mcp_manager` spawns the venv interpreter directly —
    we need it on disk before the first connection attempt.
    """
    from backend.utils import platform_label
    _current_os = platform_label()

    # uv is required to provision the per-MCP venvs below (built-in MCPs
    # like edgar-sec, plus any agent-generated/sandboxed MCP).  On a fresh
    # brew-free VM it won't be present, so install it first — mirrors the
    # Node auto-provision.  Best-effort: a failure here just means the
    # built-in MCP venvs report an error the user can act on.
    await _ensure_uv_present()

    # Mirror built-in MCP source files (server.py, requirements.txt) into
    # app-data *before* provisioning their venvs.  Although config-load also
    # syncs, that can race the venv build on a cold start (observed:
    # ensure_builtin_mcp_venvs ran before requirements.txt was copied,
    # failing with "requirements.txt missing").  Syncing here in the same
    # task guarantees the source files exist first.
    try:
        from backend.builtin_mcps import sync_builtin_mcp_files
        await asyncio.to_thread(sync_builtin_mcp_files)
    except Exception:
        logger.exception("Failed to sync built-in MCP source files before venv provisioning")

    try:
        from backend.builtin_mcps import ensure_builtin_mcp_venvs
        statuses = await ensure_builtin_mcp_venvs()
        for sid, status in statuses.items():
            if status.startswith("error:"):
                logger.warning("Built-in MCP %s venv: %s", sid, status)
            else:
                logger.info("Built-in MCP %s venv: %s", sid, status)
    except Exception:
        logger.exception("Failed to provision built-in MCP venvs")

    servers_to_connect = []
    for srv in cfg.mcp_servers:
        if not srv.enabled:
            continue
        if srv.requires_os and srv.requires_os != _current_os:
            logger.info("Skipping %s (requires %s, running %s)", srv.name, srv.requires_os, _current_os)
            continue
        if srv.auto_start:
            try:
                await mcp_mgr.ensure_process(srv)
                logger.info("Auto-started process: %s", srv.name)
            except Exception as exc:
                logger.warning("Failed to auto-start %s: %s", srv.name, exc)
        servers_to_connect.append(srv)

    async def _connect_one(srv: object) -> None:
        try:
            conn = await mcp_mgr.connect(srv, skip_process_start=True)  # type: ignore[arg-type]
            if conn.connected:
                logger.info("Auto-connected %s — %d tools", srv.name, len(conn.tools))  # type: ignore[attr-defined]
            else:
                logger.warning("Auto-connect %s failed: %s", srv.name, conn.error)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning("Failed to auto-connect %s: %s", srv.name, exc)  # type: ignore[attr-defined]

    await asyncio.gather(*(_connect_one(srv) for srv in servers_to_connect))
    logger.info("MCP startup complete — %d server(s) processed", len(servers_to_connect))


_CLEANUP_INTERVAL_SECS = 60


async def _periodic_cleanup() -> None:
    """Evict idle sessions and prune orphaned message queues periodically."""
    from backend.state import context_queues, message_queues, running_tasks

    while True:
        await asyncio.sleep(_CLEANUP_INTERVAL_SECS)
        try:
            busy = {sid for sid, t in running_tasks.items() if not t.done()}
            evicted = await session_mgr.evict_idle(busy)
            if evicted:
                logger.info("Periodic cleanup: evicted %d idle session(s)", evicted)

            stale_queues: list[str] = []
            for sid in set(message_queues) | set(context_queues):
                task = running_tasks.get(sid)
                has_running_task = task is not None and not task.done()
                has_active_session = session_mgr.get_session(sid) is not None
                if not has_running_task and not has_active_session:
                    stale_queues.append(sid)
            for sid in stale_queues:
                message_queues.pop(sid, None)
                context_queues.pop(sid, None)
            if stale_queues:
                logger.info("Periodic cleanup: removed %d stale queue(s)", len(stale_queues))
        except Exception:
            logger.debug("Periodic cleanup error", exc_info=True)


def _warm_up_graph_imports() -> None:
    """Pre-import all modules used by _build_graph so that the first session
    creation doesn't pay the frozen-archive import cost on the event loop.

    Runs in a thread pool at startup so these synchronous imports never block
    the asyncio event loop, even in the PyInstaller production binary where
    each import reads .pyc files from sys._MEIPASS.
    """
    from backend.agent_library import get_agent, get_agent_system_prompt  # noqa: F401
    from backend.mcp_manager import MCPManager  # noqa: F401
    from backend.agent_management_tools import build_management_tools  # noqa: F401
    from backend.ask_user_tools import build_ask_user_tools  # noqa: F401
    from backend.file_tools import build_file_tools  # noqa: F401
    from backend.memory_extraction import MemoryExtractionMiddleware  # noqa: F401
    from backend.schedule_tools import build_schedule_tools  # noqa: F401
    from backend.trigger_tools import build_trigger_tools  # noqa: F401
    from deep_agent.model_factory import create_llm  # noqa: F401
    from deep_agent.prompt import build_orchestrator_prompt  # noqa: F401
    from deepagents import create_deep_agent  # noqa: F401
    logger.info("Graph import warm-up complete")


_AMBIENT_JOB_ID = "ambient:periodic_sweep"


def _init_ambient(cfg: AppConfig) -> None:
    """Register the periodic ambient sweep job if enabled.

    Re-reads config each time so toggling the setting doesn't require a
    backend restart — the scheduler simply starts/stops the job.
    """
    from backend.scheduler import get_scheduler  # type: ignore[import]

    sched = get_scheduler()
    # Remove any stale job first so config changes (interval, enabled) take
    # effect immediately without restarting the server.
    try:
        sched.remove_job(_AMBIENT_JOB_ID)
    except Exception:
        pass

    if not cfg.ambient.enabled:
        return

    from apscheduler.triggers.interval import IntervalTrigger

    async def _sweep() -> None:
        from backend.ambient_agent import run_sweep
        await run_sweep(is_manual=False)

    sched.add_job(
        _sweep,
        trigger=IntervalTrigger(minutes=cfg.ambient.interval_mins),
        id=_AMBIENT_JOB_ID,
        replace_existing=True,
        misfire_grace_time=60,
        max_instances=1,
    )
    logger.info(
        "[ambient] periodic sweep scheduled every %d min", cfg.ambient.interval_mins
    )


def _resolve_npx() -> str | None:
    """Return a usable ``npx`` path, including Otto's portable Node.

    The portable Node installed by :mod:`backend.node_provisioner` may
    not be on ``PATH`` yet when this runs, so fall back to its ``bin/``.
    """
    npx = shutil.which("npx")
    if npx:
        return npx
    from backend.node_provisioner import node_bin_dir

    cand = node_bin_dir() / "npx"
    return str(cand) if cand.exists() else None


async def _ensure_playwright_browser() -> None:
    """Install the Chromium build the Playwright MCP needs (sudo-free).

    Node alone isn't enough for browser automation — the Playwright MCP
    drives a browser binary that is *not* present on a clean install/VM
    (and the "chrome" channel needs admin rights).  This downloads the
    Chromium build that matches the pinned ``@playwright/mcp`` version
    (see :data:`backend.mcp_manager.PLAYWRIGHT_VERSION`) into the user
    browser cache, which is exactly what the MCP launches now that it
    defaults to ``--browser chromium``.

    Idempotent: a version-stamped marker skips the work on later boots,
    and re-runs automatically when the pinned version is bumped.
    """
    from backend.config import get_app_data_dir
    from backend.mcp_manager import PLAYWRIGHT_VERSION
    from backend.node_provisioner import node_bin_dir
    from backend.tool_provisioner import new_job, run_streaming

    npx = _resolve_npx()
    if not npx:
        logger.warning("Auto-provision: npx not found — cannot install Playwright browser")
        return

    marker = get_app_data_dir() / ".playwright-browser-installed"
    try:
        if marker.exists() and marker.read_text(encoding="utf-8").strip() == PLAYWRIGHT_VERSION:
            logger.info("Auto-provision: Playwright Chromium already installed")
            return
    except OSError:
        pass

    logger.info(
        "Auto-provision: installing Playwright Chromium (playwright@%s)…",
        PLAYWRIGHT_VERSION,
    )
    env = dict(os.environ)
    env["PATH"] = str(node_bin_dir()) + os.pathsep + env.get("PATH", "")

    job = new_job("install-playwright-browser")
    try:
        rc = await run_streaming(
            [npx, "--yes", f"playwright@{PLAYWRIGHT_VERSION}", "install", "chromium"],
            job=job, timeout=600.0, env=env,
        )
    except Exception as exc:
        logger.warning("Auto-provision: Playwright browser install errored: %s", exc)
        return

    if rc == 0:
        try:
            marker.write_text(PLAYWRIGHT_VERSION, encoding="utf-8")
        except OSError:
            pass
        logger.info("Auto-provision: Playwright Chromium installed")
    else:
        logger.warning(
            "Auto-provision: Playwright browser install exited %d: %s",
            rc, " | ".join(job.log_lines[-3:]),
        )


async def _reconnect_playwright(cfg: AppConfig) -> None:
    """(Re-)start and connect the Playwright MCP after Node became available."""
    for srv in cfg.mcp_servers:
        if not srv.enabled or srv.id != "playwright-mcp":
            continue
        try:
            if srv.auto_start:
                await mcp_mgr.ensure_process(srv)
            conn = await mcp_mgr.connect(srv, skip_process_start=not srv.auto_start)
            if conn.connected:
                logger.info(
                    "Auto-provision: Playwright MCP connected — %d tools",
                    len(conn.tools),
                )
            else:
                logger.warning(
                    "Auto-provision: Playwright MCP still failed after Node install: %s",
                    conn.error,
                )
        except Exception as exc:
            logger.warning("Auto-provision: error (re-)connecting Playwright MCP: %s", exc)


async def _auto_provision_node() -> None:
    """Silently provision Node.js + the Playwright browser at startup so
    browser automation works on a clean install/VM with no manual steps.

    Runs as a background task (never blocks the health endpoint or setup):

    1. If Node (``node``+``npx``) is missing, download the official tarball
       (no Homebrew, no sudo) and wait for it (up to 10 minutes).
    2. With Node available, install the matching Playwright Chromium build.
    3. If Node was *just* installed, (re-)connect the Playwright MCP so its
       tools appear without a restart.

    Browser/MCP work is skipped when the ``playwright-mcp`` server is disabled.
    """
    from backend.node_provisioner import ainstall_node, node_is_present

    cfg = await AppConfig.aload()
    pw_enabled = any(
        srv.enabled and srv.id == "playwright-mcp" for srv in cfg.mcp_servers
    )

    newly_installed = False
    if not node_is_present():
        logger.info("Auto-provision: Node not found — installing in background…")
        try:
            job = await ainstall_node()
        except Exception as exc:
            logger.warning("Auto-provision: could not start Node install: %s", exc)
            return

        deadline = asyncio.get_event_loop().time() + 600
        while job.status == "running":
            if asyncio.get_event_loop().time() >= deadline:
                logger.warning("Auto-provision: Node install timed out after 10 min")
                return
            await asyncio.sleep(5)

        if job.status != "done":
            logger.warning(
                "Auto-provision: Node install ended with status=%s error=%s",
                job.status, job.error,
            )
            return
        newly_installed = True
        logger.info("Auto-provision: Node installed")
    else:
        logger.info("Auto-provision: Node already present")

    if not pw_enabled:
        return

    await _ensure_playwright_browser()

    if newly_installed:
        logger.info("Auto-provision: (re-)connecting Playwright MCP…")
        await _reconnect_playwright(cfg)


async def _loop_health_monitor() -> None:
    """Background task that detects event loop blocking.

    Schedules a callback every 500ms.  If the actual delay exceeds 1s the
    event loop was blocked — log a warning with the duration so we can
    track down the culprit.
    """
    threshold = 1.0
    while True:
        t0 = time.monotonic()
        await asyncio.sleep(0.5)
        elapsed = time.monotonic() - t0
        if elapsed > threshold:
            logger.warning(
                "Event loop was blocked for %.2fs (expected ~0.5s)",
                elapsed,
            )


_SERVER_BOOT_TIME: float = 0.0


def get_boot_time() -> float:
    """Return the Unix timestamp when this server process started."""
    return _SERVER_BOOT_TIME


def _repair_stale_running_sessions() -> None:
    """Reset sessions left with status 'running' on disk after a crash or restart.

    Mirrors the orphan-repair logic used by the scheduler and trigger manager
    for schedule/trigger runs.  Must be called synchronously at startup before
    any agent tasks are created so there is no race with legitimate running tasks.
    """
    import json as _json
    from datetime import datetime, timezone as _tz
    from backend.session_manager import _sessions_dir

    now = datetime.now(_tz.utc).isoformat()
    count = 0
    sessions_dir = _sessions_dir()
    for p in sessions_dir.glob("*.json"):
        if p.name.endswith((".messages.json", ".eval.json")):
            continue
        try:
            data = _json.loads(p.read_text(encoding="utf-8"))
            if data.get("status") == "running":
                data["status"] = "error"
                data["error"] = "Interrupted — server restarted while running"
                data["finished_at"] = now
                p.write_text(_json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
                count += 1
        except Exception:
            logger.debug("Skipping corrupt session meta during repair: %s", p.name, exc_info=True)
    if count:
        logger.info("Startup: reset %d stale running session(s) to error", count)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _SERVER_BOOT_TIME  # noqa: PLW0603
    _SERVER_BOOT_TIME = time.time()

    _attach_file_handler()

    from backend.exo_setup import install_secret_scrub_filter
    install_secret_scrub_filter()

    loop = asyncio.get_running_loop()
    loop.set_exception_handler(_suppress_mcp_cleanup_noise)

    # Use a large default executor so LLM streaming chunks, file I/O, and
    # DNS resolution never starve each other.  ChatBedrockConverse._astream
    # uses run_in_executor(None, next, ...) for EVERY streaming chunk, so
    # concurrent LLM streams can exhaust a small pool quickly.
    default_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=40, thread_name_prefix="asyncio-default",
    )
    loop.set_default_executor(default_executor)

    _debug = os.environ.get("CURA_DEBUG", "").lower() in ("1", "true", "yes")
    if _debug:
        loop.set_debug(True)
        loop.slow_callback_duration = 0.2
        logging.getLogger("asyncio").setLevel(logging.WARNING)

    logging.getLogger("asyncio").addFilter(_MCPNoiseFilter())

    from tools.research._loaders import warm_up as _warm_up_loaders
    _warm_up_loaders()
    asyncio.create_task(asyncio.to_thread(_warm_up_graph_imports))

    cfg = await AppConfig.aload()
    cfg.apply_to_environ()
    seed_defaults()
    logger.info(
        "Backend starting — config loaded (executor: 40 threads, debug=%s)",
        _debug,
    )

    asyncio.create_task(_startup_mcp(cfg))
    asyncio.create_task(_auto_provision_node())
    cleanup_task = asyncio.create_task(_periodic_cleanup())
    health_task = asyncio.create_task(_loop_health_monitor())
    init_scheduler()
    init_trigger_manager()
    _repair_stale_running_sessions()
    _init_ambient(cfg)

    # Activity timeline: opt-in metadata-only tracker.  The tracker
    # itself reads the config on every tick, so this start() is a no-op
    # when ``activity.enabled`` is false; users can flip it on without
    # restarting the backend.
    try:
        await activity_tracker.start()
    except Exception:
        logger.exception("Activity tracker failed to start")

    from backend.exo_provisioner import auto_start_if_enabled as _exo_auto_start
    asyncio.create_task(_exo_auto_start(cfg))

    from backend.openclaw_watcher import oc_watcher
    if cfg.openclaw.enabled and cfg.openclaw.watcher_enabled:
        await oc_watcher.start(poll_interval=cfg.openclaw.watcher_poll_interval)

    yield
    await oc_watcher.stop()
    shutdown_scheduler()
    health_task.cancel()
    cleanup_task.cancel()
    try:
        await activity_tracker.stop()
    except Exception:
        logger.debug("Activity tracker stop failed", exc_info=True)
    await session_mgr.close_all()
    await mcp_mgr.disconnect_all()
    # Release mic + speakers so macOS clears the mic indicator on shutdown/reload
    try:
        from backend.voice.voice_manager import get_manager as _get_voice_mgr
        await _get_voice_mgr().stop()
    except Exception:
        logger.debug("Voice manager stop on shutdown failed", exc_info=True)
    default_executor.shutdown(wait=False)
    logger.info("Backend shut down")


app = FastAPI(
    title="Otto",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:1420",   # Tauri dev
        "http://localhost:5173",   # Vite dev
        "https://tauri.localhost", # Tauri production (Windows WebView2)
        "http://tauri.localhost",  # Tauri production (Windows WebView2 http scheme)
        "tauri://localhost",       # Tauri production (macOS/Linux)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(settings_router)
app.include_router(mcp_router)
app.include_router(mcp_config_router)
app.include_router(agents_router)
app.include_router(sessions_router)
app.include_router(ws_router)
app.include_router(schedules_router)
app.include_router(triggers_router)
app.include_router(hooks_router)
app.include_router(openclaw_hooks_router)
app.include_router(memory_router)
app.include_router(embeddings_router)
app.include_router(mlx_router)
app.include_router(exo_router)
app.include_router(omlx_router)
app.include_router(node_router)
app.include_router(privacy_router)
app.include_router(vault_router)
app.include_router(activity_router)
app.include_router(setup_router)
app.include_router(ambient_router)
app.include_router(voice_router)
app.include_router(voice_ws_router)
app.include_router(runs_router)


# =========================================================================
# Standalone entry point (PyInstaller sidecar)
# =========================================================================

def _ensure_sidecar_paths() -> None:
    """When running as a frozen PyInstaller binary, add the bundled ``src/``
    directory to ``sys.path`` so that ``deep_agent``, ``tools``, etc. resolve,
    and ensure common Node.js locations are on PATH for npx discovery."""
    import os
    import sys
    from pathlib import Path

    if not getattr(sys, "frozen", False):
        return

    bundle_dir = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    src_dir = str(bundle_dir / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    current_path = os.environ.get("PATH", "")
    sep = ";" if sys.platform == "win32" else ":"

    if sys.platform == "win32":
        extra_paths = [
            r"C:\Program Files\nodejs",
            r"C:\Program Files (x86)\nodejs",
            str(Path.home() / "AppData" / "Roaming" / "npm"),
            str(Path.home() / "AppData" / "Roaming" / "nvm"),
            str(Path.home() / "AppData" / "Local" / "fnm_multishells"),
        ]
        nvm_root = Path.home() / "AppData" / "Roaming" / "nvm"
        if nvm_root.is_dir():
            for ver in sorted(nvm_root.iterdir(), reverse=True):
                if (ver / "node.exe").exists():
                    extra_paths.insert(0, str(ver))
                    break
    else:
        extra_paths = [
            "/usr/local/bin",
            "/opt/homebrew/bin",
            str(Path.home() / ".local" / "share" / "fnm" / "aliases" / "default" / "bin"),
            # Brew-free tool installs managed by Otto (node_provisioner /
            # tool_provisioner) and the uv official installer.
            str(Path.home() / "Library" / "Application Support" / "Otto" / "tools" / "node" / "bin"),
            str(Path.home() / ".local" / "bin"),
        ]
        nvm_base = Path.home() / ".nvm" / "versions" / "node"
        if nvm_base.is_dir():
            for ver in sorted(nvm_base.iterdir(), reverse=True):
                bin_dir = str(ver / "bin")
                extra_paths.insert(0, bin_dir)
                break

    for p in extra_paths:
        if p not in current_path:
            current_path = f"{p}{sep}{current_path}"
    os.environ["PATH"] = current_path


def _run_mcp_server(server_name: str, remaining_args: list[str]) -> None:
    """Run a bundled MCP server in-process (used by frozen binary subcommand)."""
    import sys

    sys.argv = [server_name] + remaining_args

    if server_name == "agent-eval-service":
        from tools.evaluation.mcp_server import main
    elif server_name == "claude-eval-hook":
        from tools.transcripts.claude_mcp_server import main
    elif server_name == "openclaw-eval-hook":
        from tools.transcripts.openclaw_mcp_server import main
    else:
        raise SystemExit(f"Unknown MCP server: {server_name}")

    main()


if __name__ == "__main__":
    import argparse
    import atexit
    import multiprocessing
    import sys
    from typing import Any

    multiprocessing.freeze_support()
    _ensure_sidecar_paths()

    if len(sys.argv) >= 3 and sys.argv[1] == "--mcp-server":
        _run_mcp_server(sys.argv[2], sys.argv[3:])
        raise SystemExit(0)

    try:
        import uvloop
        uvloop.install()
        _loop_impl = "uvloop"
    except ImportError:
        _loop_impl = "asyncio (SelectorEventLoop) — uvloop not available"
    logger.info("Event loop implementation: %s", _loop_impl)

    def _force_kill_managed_processes() -> None:
        """Last-resort cleanup: kill any managed process groups on exit."""
        for proc in mcp_mgr._processes.values():
            if proc._proc is not None and proc._proc.returncode is None:
                try:
                    pgid = os.getpgid(proc._proc.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass

    atexit.register(_force_kill_managed_processes)

    def _sigterm_handler(signum: int, frame: Any) -> None:
        _force_kill_managed_processes()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    _DEFAULT_HOST = "127.0.0.1"
    _DEFAULT_PORT = 18081

    parser = argparse.ArgumentParser(description="Deep Agent backend server")
    parser.add_argument("--host", default=_DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT)
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
