"""Shared fixtures for live LLM integration tests.

All tests in this package require a configured LLM provider (set via the
normal environment variables or app config) and make real network / on-device
calls.  They are excluded from the default pytest run via the ``live`` marker;
run them explicitly with::

    pytest -m live
    pytest -m "live and mlx_qwen"
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import AsyncGenerator, Any

import pytest
import pytest_asyncio

# Ensure both src/ and the repo root are on sys.path so imports work the same
# as in the running application.

# ---------------------------------------------------------------------------
# Bootstrap: stamp config.json API keys into os.environ so the LLM clients
# (which read from env vars, not from the config object directly) can
# authenticate in the test process.  The running app does this at startup via
# AppConfig.to_env_dict(); tests never call that path, so we do it here once.
# ---------------------------------------------------------------------------


def _apply_config_env() -> None:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from backend.config import AppConfig

        cfg = AppConfig.load()
        for key, val in cfg.to_env_dict().items():
            if val and not os.environ.get(key):
                os.environ[key] = val
    except Exception:
        pass  # best-effort; missing keys will still cause individual skips


_apply_config_env()
_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root / "src"))
sys.path.insert(0, str(_repo_root))

# Apply the summarization_guard monkey-patch early, exactly as server.py does
# at startup.  Without it, SummarizationMiddleware can leave orphaned
# ToolMessages that trigger Anthropic API errors in tests.
import backend.summarization_guard  # noqa: F401


# ---------------------------------------------------------------------------
# App-data isolation
# ---------------------------------------------------------------------------


@pytest.fixture()
def live_app_dir(tmp_path, monkeypatch):
    """Redirect all data-dir lookups to a per-test temp directory.

    Patches every module that imports ``get_app_data_dir`` directly so
    sessions, checkpoints, and audit logs don't touch the real app data.
    """
    _data = tmp_path / "data"
    _data.mkdir()

    # Copy the real config.json (provider settings + API keys) into the temp
    # data dir BEFORE we redirect get_app_data_dir.  Otherwise AppConfig.load()
    # returns a default config with empty keys, and create_session() ->
    # apply_to_environ() would then *clear* ANTHROPIC_API_KEY from os.environ
    # right before the graph builds its ChatAnthropic client.  Session,
    # checkpoint, and audit files are stored separately, so isolation is kept.
    import shutil

    try:
        from backend.config import get_app_data_dir as _real_data_dir

        _real_cfg = _real_data_dir() / "config.json"
        if _real_cfg.exists():
            shutil.copy2(_real_cfg, _data / "config.json")
    except Exception:
        pass

    _stub = lambda: _data  # noqa: E731

    for target in (
        "backend.config.get_app_data_dir",
        "backend.session_manager.get_app_data_dir",
        "backend.privacy_lock.get_app_data_dir",
        "backend.memory_relevance._memory_dir",
        "backend.memory_extraction",  # patched via module attribute below
    ):
        try:
            monkeypatch.setattr(target, _stub, raising=False)
        except Exception:
            pass

    # memory_relevance._memory_dir is a function, not a simple attribute —
    # patch the private helper so relevance logs also land in tmp.
    try:
        import backend.memory_relevance as _mr
        monkeypatch.setattr(_mr, "_memory_dir", _stub)
    except Exception:
        pass

    return _data


# ---------------------------------------------------------------------------
# SessionManager fixture
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture()
async def session_manager(live_app_dir):
    """A fresh SessionManager backed by the temp data directory."""
    from backend.session_manager import SessionManager

    mgr = SessionManager()
    yield mgr
    await mgr.close_all()


async def _probe_all_llms(cfg) -> None:
    """Skip the calling test if any required LLM (main or orchestrator) is unreachable.

    Probes both the main chat provider and the orchestrator provider (when it
    differs from the main one) by issuing a real ``ainvoke("ping")`` call.
    Calls ``pytest.skip()`` on the first failure so that tests which create
    sessions manually get the same guard as the ``live_session`` fixture.
    """
    from deep_agent.model_factory import create_llm
    from langchain_core.messages import HumanMessage

    # --- probe main provider ---
    try:
        probe = create_llm(cfg.llm.provider)
        await probe.ainvoke([HumanMessage(content="ping")])
    except Exception as exc:
        pytest.skip(
            f"LLM provider '{cfg.llm.provider}' not reachable in test env "
            f"(set the matching API-key env var to run live tests): {exc}"
        )

    # --- probe orchestrator provider if it differs from the main one ---
    # This mirrors _build_orch_llm_sync so that tests fail-fast when the
    # orchestrator (e.g. Anthropic 'frontier') is not configured, rather than
    # reaching the stream before hitting the auth error.
    orch = cfg.orchestrator
    po = (getattr(orch, "provider_override", None) or "").strip().lower()
    fam = (getattr(orch, "llm_family", "follow_main") or "follow_main").strip().lower()

    orch_provider: str | None = None
    if po:
        orch_provider = po
    elif fam == "frontier":
        orch_provider = "anthropic"
    elif fam == "openai":
        orch_provider = "openai"
    elif fam == "exo":
        orch_provider = "exo"
    # 'follow_main', 'mlx', 'inherit' → shares the main probe above

    if orch_provider and orch_provider != cfg.llm.provider:
        try:
            orch_probe = create_llm(orch_provider)
            await orch_probe.ainvoke([HumanMessage(content="ping")])
        except Exception as exc:
            pytest.skip(
                f"Orchestrator LLM provider '{orch_provider}' not reachable in test env "
                f"(set the matching API-key env var to run live tests): {exc}"
            )


@pytest_asyncio.fixture()
async def live_session(session_manager):
    """A created Session using the env-configured provider.

    Skips if the configured provider cannot be reached (e.g. no API key set in
    the test-process environment, even if the app reads credentials from a
    keychain at runtime).
    """
    from backend.config import AppConfig

    cfg = AppConfig.load()

    # Pre-flight: confirm both the main and orchestrator LLMs are callable.
    # The session graph can build successfully while an API key is absent from
    # the test-process env (the running app uses keychain; tests use env vars).
    await _probe_all_llms(cfg)

    try:
        session = await session_manager.create_session(cfg)
    except Exception as exc:
        pytest.skip(f"Could not create session (provider not configured?): {exc}")

    yield session
    try:
        await session_manager.close_session(session.id)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helper: collect all stream events
# ---------------------------------------------------------------------------


async def collect_stream(agen: AsyncGenerator[dict, None]) -> list[dict]:
    """Drain an async-generator event stream into a list."""
    events: list[dict] = []
    async for ev in agen:
        events.append(ev)
    return events


async def run_session(mgr: Any, session_id: str, prompt: str) -> list[dict]:
    """Drive stream_message to completion; return the full event list."""
    return await collect_stream(mgr.stream_message(session_id, prompt))
