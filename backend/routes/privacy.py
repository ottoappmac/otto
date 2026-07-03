"""REST endpoints for the verifiable privacy lock.

These routes back Settings → Privacy & Security in the Tauri UI.  They
are intentionally small: get/update the config, engage/disengage the
lock, render the pf template, read the audit log.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Body, HTTPException

from backend import privacy_lock
from backend.config import AppConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/privacy", tags=["privacy"])


# Bounded so a hostile actor can't fill the JSON response with thousands
# of unrelated audit lines.
_MAX_TAIL = 500


def _summary(cfg: AppConfig) -> dict[str, Any]:
    p = cfg.privacy
    return {
        "engaged": p.enabled,
        "engaged_at": p.engaged_at,
        "audit_token": p.audit_token,
        "allow_loopback": p.allow_loopback,
        "allow_mdns": p.allow_mdns,
        "allowed_hosts": list(p.allowed_hosts),
        "local_only_providers": list(p.local_only_providers),
        "pf_anchor": p.pf_anchor,
    }


@router.get("")
async def get_privacy() -> dict[str, Any]:
    cfg = await AppConfig.aload()
    return _summary(cfg)


@router.put("")
async def update_privacy(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    """Patch the privacy config.

    Only the user-tunable fields are accepted; ``enabled``, ``engaged_at``
    and ``audit_token`` are written exclusively through ``/engage`` and
    ``/disengage`` so the audit log stays accurate.
    """
    cfg = await AppConfig.aload()
    p = cfg.privacy

    if "allow_loopback" in payload:
        p.allow_loopback = bool(payload["allow_loopback"])
    if "allow_mdns" in payload:
        p.allow_mdns = bool(payload["allow_mdns"])
    if "allowed_hosts" in payload:
        hosts = payload["allowed_hosts"]
        if not isinstance(hosts, list):
            raise HTTPException(status_code=400, detail="allowed_hosts must be a list")
        p.allowed_hosts = [str(h) for h in hosts if str(h).strip()]
    if "local_only_providers" in payload:
        provs = payload["local_only_providers"]
        if not isinstance(provs, list):
            raise HTTPException(status_code=400, detail="local_only_providers must be a list")
        p.local_only_providers = [str(s).strip().lower() for s in provs if str(s).strip()]
    if "pf_anchor" in payload:
        anchor = str(payload["pf_anchor"]).strip()
        if not anchor:
            raise HTTPException(status_code=400, detail="pf_anchor must be non-empty")
        p.pf_anchor = anchor

    await cfg.asave()
    return _summary(cfg)


@router.post("/engage")
async def engage_route() -> dict[str, Any]:
    cfg = await AppConfig.aload()
    result = privacy_lock.engage(cfg)
    await cfg.asave()
    cfg.apply_to_environ()
    return result | _summary(cfg)


@router.post("/disengage")
async def disengage_route() -> dict[str, Any]:
    cfg = await AppConfig.aload()
    result = privacy_lock.disengage(cfg)
    await cfg.asave()
    cfg.apply_to_environ()
    return result | _summary(cfg)


@router.get("/status")
async def status_route() -> dict[str, Any]:
    """Engagement state + live pf anchor inspection."""
    cfg = await AppConfig.aload()
    return {
        **_summary(cfg),
        "pf": privacy_lock.pf_status(cfg),
    }


@router.get("/pf-template")
async def pf_template_route() -> dict[str, Any]:
    """Render the pf.conf snippet plus the install command.

    Useful for the UI to surface a "copy this into Terminal" block.
    """
    cfg = await AppConfig.aload()
    return {
        "anchor": cfg.privacy.pf_anchor,
        "install_command": privacy_lock.pf_install_command(cfg),
        "pf_template": privacy_lock.render_pf_template(cfg),
    }


@router.get("/audit")
async def audit_route(limit: int = 100) -> dict[str, Any]:
    if limit < 1 or limit > _MAX_TAIL:
        raise HTTPException(
            status_code=400,
            detail=f"limit must be in [1, {_MAX_TAIL}]",
        )
    return {"events": privacy_lock.tail_audit(limit)}


@router.post("/check-provider")
async def check_provider_route(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    """Dry-run check that *provider* would be allowed under the current lock.

    Useful for the Settings UI: when the user toggles the LLM provider
    while engaged, the panel can preflight the choice and warn before
    a session-start failure.
    """
    provider = str(payload.get("provider") or "").strip().lower()
    if not provider:
        raise HTTPException(status_code=400, detail="provider is required")
    cfg = await AppConfig.aload()
    try:
        privacy_lock.enforce_provider_allowed(provider, cfg)
    except privacy_lock.PrivacyLockActive as exc:
        return {"allowed": False, "reason": str(exc)}
    return {"allowed": True}
