"""Settings routes for the local OpenJev decision sidecar.

Weight downloads go through ``POST /api/mlx/download``. This router covers
runtime install, start, stop, model selection, a connectivity test, and
the API key for an optional remote endpoint.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.config import AppConfig
from backend.decision_catalog import get_model, scored_catalog
from backend.privacy_lock import PrivacyLockActive, enforce_decision_endpoint

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/decision", tags=["decision"])


def _hub() -> str:
    return str(__import__("pathlib").Path.home() / ".cache" / "huggingface" / "hub")


class SelectBody(BaseModel):
    model_repo: str = Field(..., min_length=1)


class TestBody(BaseModel):
    situation: str = ""
    question: str = "Is this a test?"
    options: list[str] | None = None


class CredentialBody(BaseModel):
    api_key: str = ""


@router.get("")
async def decision_snapshot() -> dict[str, Any]:
    from backend.decision_runtime import status_snapshot

    cfg = await AppConfig.aload()
    return {
        "config": cfg.decision.model_dump(),
        "status": status_snapshot(cfg.decision),
        "models": scored_catalog(_hub()),
    }


@router.get("/status")
async def decision_status() -> dict[str, Any]:
    from backend.decision_runtime import status_snapshot

    cfg = await AppConfig.aload()
    return status_snapshot(cfg.decision)


@router.get("/models")
async def decision_models() -> dict[str, Any]:
    return {"models": scored_catalog(_hub())}


@router.get("/log")
async def decision_log(lines: int = 200) -> dict[str, str]:
    from backend.decision_runtime import tail_log

    return {"log": tail_log(max_lines=lines)}


@router.get("/jobs/{job_id}")
async def decision_job(job_id: str) -> JSONResponse:
    from backend.decision_runtime import get_job

    job = get_job(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"error": "unknown job"})
    return JSONResponse(job)


@router.post("/install")
async def decision_install() -> dict[str, str]:
    from backend.decision_runtime import install_async

    cfg = await AppConfig.aload()
    return {"job_id": install_async(cfg.decision), "status": "started"}


@router.post("/start")
async def decision_start() -> JSONResponse:
    from backend.decision_runtime import start_async, uses_local_sidecar

    cfg = await AppConfig.aload()
    try:
        if not uses_local_sidecar(cfg.decision):
            enforce_decision_endpoint(cfg.decision.endpoint_override, cfg)
    except PrivacyLockActive as exc:
        return JSONResponse(status_code=403, content={"error": str(exc)})
    return JSONResponse({"job_id": start_async(cfg.decision), "status": "started"})


@router.post("/stop")
async def decision_stop() -> dict[str, str]:
    from backend.decision_runtime import stop_async

    return {"job_id": stop_async(), "status": "started"}


@router.post("/select")
async def decision_select(body: SelectBody) -> JSONResponse:
    from backend.decision_runtime import running, stop_sync

    repo = body.model_repo.strip()
    if get_model(repo) is None:
        return JSONResponse(
            status_code=400,
            content={"error": f"{repo} is not in the decision catalog"},
        )
    cfg = await AppConfig.aload()
    changed = cfg.decision.model_repo != repo
    cfg.decision.model_repo = repo
    await cfg.asave()
    if changed and running(cfg.decision):
        stop_sync()
    return JSONResponse({"model_repo": repo, "restarted": changed})


@router.post("/credential")
async def decision_credential(body: CredentialBody) -> JSONResponse:
    from backend.credential_vault import CredentialVaultError
    from backend.decision_runtime import set_remote_api_key

    key = body.api_key.strip()
    if not key:
        return JSONResponse(status_code=400, content={"error": "api_key is required"})
    try:
        set_remote_api_key(key)
    except CredentialVaultError as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
    return JSONResponse({"set": True})


@router.post("/test")
async def decision_test(body: TestBody) -> JSONResponse:
    from backend.decision_client import DecisionClientError, ask, format_answer, source_label
    from backend.decision_runtime import (
        effective_base_url,
        ensure_running,
        local_api_key,
        remote_api_key,
        uses_local_sidecar,
    )

    cfg = await AppConfig.aload()
    decision = cfg.decision
    base = effective_base_url(decision)
    try:
        enforce_decision_endpoint("" if uses_local_sidecar(decision) else base, cfg)
    except PrivacyLockActive as exc:
        return JSONResponse(status_code=403, content={"error": str(exc)})
    try:
        if uses_local_sidecar(decision):
            base = ensure_running(decision)
            api_key = local_api_key()
        else:
            api_key = remote_api_key() or ""
            if not api_key:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Set an API key for the custom endpoint first."},
                )
        result = ask(
            body.situation or "Settings test.",
            body.question,
            body.options,
            base_url=base,
            api_key=api_key,
        )
    except DecisionClientError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception as exc:
        logger.info("decision test failed: %s", exc)
        return JSONResponse(status_code=503, content={"error": str(exc)})
    return JSONResponse({
        "text": format_answer(
            result,
            confident_threshold=float(decision.confident_threshold),
            unsure_threshold=float(decision.unsure_threshold),
            unattended=False,
            source=source_label(base),
        ),
    })
