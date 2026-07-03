"""REST API for the brew-free Node.js provisioner.

Wraps :mod:`backend.node_provisioner`.  The setup flow uses these
endpoints to install Node (required by the Playwright MCP) without
Homebrew, then polls the job for progress — mirroring the oMLX install
endpoints in :mod:`backend.routes.omlx`.

Endpoints
---------

``GET    /api/node/status``        detection snapshot (present? where?)
``POST   /api/node/install``       kick off a background install job
``GET    /api/node/jobs``          list recent install jobs
``GET    /api/node/jobs/{id}``     poll one job
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from backend import node_provisioner as nodep
from backend import tool_provisioner as tp

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/node", tags=["node"])


@router.get("/status")
async def node_status() -> dict:
    return nodep.status_snapshot()


@router.post("/install")
async def node_install(force: bool = False) -> dict:
    """Kick off a Node.js install job.

    Pass ``?force=true`` to skip the presence check and re-install even
    when a ``node``+``npx`` pair is already detected (useful for testing
    or repairing a broken install).
    """
    job = await nodep.ainstall_node(force=force)
    return {"job_id": job.id, **job.to_dict()}


@router.get("/jobs")
async def list_node_jobs() -> dict:
    return {"jobs": [j.to_dict() for j in tp.list_jobs()]}


@router.get("/jobs/{job_id}")
async def get_node_job(job_id: str) -> dict:
    job = tp.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job.to_dict()
