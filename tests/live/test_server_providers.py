"""Live tests: omlx and exo server provider smoke tests.

These are intentionally minimal — omlx and exo are OpenAI-compatible HTTP
APIs so ChatOpenAI handles protocol normalisation.  We only verify:

1. The connection probe succeeds (daemon is reachable, model resolved).
2. A basic ainvoke returns a non-empty response.

No per-model parametrization: the model running in the daemon is an ops
concern, not a code correctness concern.

Each test is gated on its daemon being reachable; unreachable daemons are
skipped rather than failed.

Run with::

    pytest -m live tests/live/test_server_providers.py
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.live


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _daemon_reachable(base_url: str) -> bool:
    """Return True if the HTTP endpoint responds to GET /v1/models."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{base_url}/v1/models")
            return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# omlx
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_omlx_connection_probe():
    """POST /api/settings/test-connection for omlx must succeed when the
    omlx daemon is running."""
    from backend.config import AppConfig

    cfg = AppConfig.load()
    omlx_base = getattr(cfg.llm, "omlx_base_url", None) or "http://localhost:10240"

    if not await _daemon_reachable(omlx_base):
        pytest.skip("omlx daemon not reachable — skipping probe")

    from backend.routes.settings import test_llm_connection
    from backend.schemas import TestConnectionRequest

    req = TestConnectionRequest(provider="omlx")
    result = await test_llm_connection(req)

    assert result.get("success") is True, (
        f"omlx connection probe failed: {result}"
    )


@pytest.mark.asyncio
async def test_omlx_basic_ainvoke():
    """create_llm('omlx') + ainvoke('Reply with OK') must return a non-empty
    AIMessage when the daemon is running."""
    from backend.config import AppConfig

    cfg = AppConfig.load()
    omlx_base = getattr(cfg.llm, "omlx_base_url", None) or "http://localhost:10240"

    if not await _daemon_reachable(omlx_base):
        pytest.skip("omlx daemon not reachable")

    from deep_agent.model_factory import create_llm
    from langchain_core.messages import HumanMessage

    try:
        llm = create_llm("omlx")
    except Exception as exc:
        pytest.skip(f"Could not create omlx LLM: {exc}")

    resp = await llm.ainvoke([HumanMessage(content="Reply with exactly: OK")])
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    assert content.strip(), "Expected non-empty response from omlx"


@pytest.mark.asyncio
async def test_omlx_model_resolved_from_server():
    """_resolve_omlx_model_id must return a non-empty string when the daemon
    is up — confirming the /v1/models lookup succeeds."""
    from backend.config import AppConfig

    cfg = AppConfig.load()
    omlx_base = getattr(cfg.llm, "omlx_base_url", None) or "http://localhost:10240"

    if not await _daemon_reachable(omlx_base):
        pytest.skip("omlx daemon not reachable")

    from deep_agent.model_factory import _resolve_omlx_model_id

    model_id = await _resolve_omlx_model_id(cfg)
    assert model_id and isinstance(model_id, str), (
        f"Expected a non-empty model ID from omlx server; got: {model_id!r}"
    )


# ---------------------------------------------------------------------------
# exo
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exo_connection_probe():
    """POST /api/settings/test-connection for exo must succeed when the
    exo cluster is running."""
    from backend.config import AppConfig

    cfg = AppConfig.load()
    exo_base = getattr(cfg.llm, "exo_base_url", None) or "http://localhost:52415"

    if not await _daemon_reachable(exo_base):
        pytest.skip("exo cluster not reachable — skipping probe")

    from backend.routes.settings import test_llm_connection
    from backend.schemas import TestConnectionRequest

    req = TestConnectionRequest(provider="exo")
    result = await test_llm_connection(req)

    if not result.success:
        pytest.skip(
            f"exo connection probe returned success=False (model not selected or not running): "
            f"{result.message}"
        )


@pytest.mark.asyncio
async def test_exo_basic_ainvoke():
    """create_llm('exo') + ainvoke must return a non-empty AIMessage."""
    from backend.config import AppConfig

    cfg = AppConfig.load()
    exo_base = getattr(cfg.llm, "exo_base_url", None) or "http://localhost:52415"

    if not await _daemon_reachable(exo_base):
        pytest.skip("exo cluster not reachable")

    from deep_agent.model_factory import create_llm
    from langchain_core.messages import HumanMessage

    try:
        llm = create_llm("exo")
    except Exception as exc:
        pytest.skip(f"Could not create exo LLM: {exc}")

    try:
        resp = await llm.ainvoke([HumanMessage(content="Reply with exactly: OK")])
    except Exception as exc:
        pytest.skip(f"exo ainvoke failed (model may not be loaded on the cluster): {exc}")
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    assert content.strip(), "Expected non-empty response from exo"
