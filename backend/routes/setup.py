"""First-run Setup Wizard lifecycle endpoints.

Endpoints
---------

* ``GET  /api/setup/state``        — current wizard state (current step, completed flags).
* ``POST /api/setup/step``         — mark a step as completed and persist a current-step cursor.
* ``POST /api/setup/complete``     — flip ``setup.completed`` and stop showing the wizard.
* ``POST /api/setup/skip``         — flip ``setup.dismissed`` and stop showing the wizard.
* ``GET  /api/setup/permissions/accessibility`` — macOS Accessibility permission probe used by the Activity screen.

Each handler is intentionally tiny — they only mutate :class:`SetupState`
fields on the persisted :class:`AppConfig`.  All real configuration
(model provider, memory, activity) is saved through the existing
``PUT /api/settings`` route as the user progresses through the wizard,
so resuming a half-finished setup re-reads the same source of truth as
the SettingsPage.
"""

from __future__ import annotations

import logging
import platform
from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel

from backend.config import AppConfig
from backend.setup_chat import SETUP_MODEL_ID, generate_reply

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/setup", tags=["setup"])


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@router.get("/state")
async def setup_state():
    cfg = await AppConfig.aload()
    return {
        "completed": cfg.setup.completed,
        "dismissed": cfg.setup.dismissed,
        "current_step": cfg.setup.current_step,
        "completed_steps": list(cfg.setup.completed_steps),
        "first_run": cfg.is_first_run(),
    }


class StepUpdate(BaseModel):
    step: str
    completed: bool = True


@router.post("/step")
async def mark_step(req: StepUpdate):
    """Persist that the user reached / left a step.

    Idempotent — re-marking the same step is a no-op.  Saves the config
    so a force-quit mid-wizard resumes on the right screen.
    """
    cfg = await AppConfig.aload()
    cfg.setup.current_step = req.step
    if req.completed and req.step not in cfg.setup.completed_steps:
        cfg.setup.completed_steps.append(req.step)
    await cfg.asave()
    return {
        "current_step": cfg.setup.current_step,
        "completed_steps": list(cfg.setup.completed_steps),
    }


# ---------------------------------------------------------------------------
# Terminal actions
# ---------------------------------------------------------------------------


@router.post("/complete")
async def complete_setup():
    cfg = await AppConfig.aload()
    cfg.setup.completed = True
    cfg.setup.dismissed = False
    cfg.setup.current_step = "done"
    await cfg.asave()
    return {"ok": True}


@router.post("/skip")
async def skip_setup():
    cfg = await AppConfig.aload()
    cfg.setup.dismissed = True
    cfg.setup.completed = False
    await cfg.asave()
    return {"ok": True}


@router.post("/reset")
async def reset_setup():
    """Re-open the wizard.  Used by Settings → "Re-run setup wizard"."""
    cfg = await AppConfig.aload()
    cfg.setup.completed = False
    cfg.setup.dismissed = False
    cfg.setup.current_step = "welcome"
    cfg.setup.completed_steps = []
    await cfg.asave()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Permission probes
# ---------------------------------------------------------------------------


@router.get("/permissions/accessibility")
async def accessibility_permission():
    """Probe macOS Accessibility permission for the calling process.

    Used by the Activity wizard step to (a) show whether the user has
    already granted it (skip the system prompt), and (b) re-check after
    the user clicks "Open System Settings…" without forcing a relaunch.

    Returns ``{platform, supported, granted, can_prompt}`` with sensible
    defaults on non-macOS.
    """
    if platform.system() != "Darwin":
        return {
            "platform": platform.system(),
            "supported": False,
            "granted": False,
            "can_prompt": False,
        }

    try:
        from ApplicationServices import (  # type: ignore[import]
            AXIsProcessTrustedWithOptions,
            kAXTrustedCheckOptionPrompt,
        )

        # Probe-only call (no prompt) — passing False as the value.
        options = {kAXTrustedCheckOptionPrompt: False}
        granted = bool(AXIsProcessTrustedWithOptions(options))
        return {
            "platform": "Darwin",
            "supported": True,
            "granted": granted,
            "can_prompt": True,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Accessibility probe failed: %s", exc)
        return {
            "platform": "Darwin",
            "supported": False,
            "granted": False,
            "can_prompt": False,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Setup chat — conversational Q&A reply generation
# ---------------------------------------------------------------------------


class SetupChatRequest(BaseModel):
    """One turn of the first-run setup conversation.

    The frontend owns the step flow and config mutations; this endpoint
    only generates the conversational reply text.
    """

    step: str
    """Current wizard step (``provider``, ``cloud_sub``, ``cloud_key``,
    ``local_model``, ``memory``, ``activity``, ``done``)."""

    user_message: str
    """Raw text the user typed."""

    extracted: str | None = None
    """Structured answer the frontend extracted (e.g. ``"anthropic"``).
    ``None`` signals an ambiguous answer — the reply will ask for
    clarification and ``needs_clarification`` will be ``True``."""

    context: dict[str, Any] = {}
    """Machine context — at minimum ``{"chip": "...", "ram_gb": 16}``."""

    model_ready: bool = False
    """Whether the Qwen3-1.7B-4bit model is confirmed in the HF cache.
    When ``False`` the backend skips LLM inference and returns a
    template string immediately."""


class SetupChatResponse(BaseModel):
    reply: str
    needs_clarification: bool
    setup_model_id: str


@router.post("/chat", response_model=SetupChatResponse)
async def setup_chat(req: SetupChatRequest) -> SetupChatResponse:
    """Generate a natural-language reply for the current setup step.

    Config mutations (provider, API keys, model selection, memory,
    activity) are the caller's responsibility — handled by the frontend
    via ``PUT /api/settings`` and the existing MLX / oMLX routes.
    """
    cfg = await AppConfig.aload()

    reply = await generate_reply(
        step=req.step,
        user_message=req.user_message,
        extracted=req.extracted,
        context=req.context,
        model_ready=req.model_ready,
        model_id=cfg.setup_chat.setup_model_id,
        max_tokens=cfg.setup_chat.setup_max_tokens,
        temp=cfg.setup_chat.setup_temp,
    )

    return SetupChatResponse(
        reply=reply,
        needs_clarification=(req.extracted is None),
        setup_model_id=SETUP_MODEL_ID,
    )


class PermissionPromptRequest(BaseModel):
    kind: Literal["accessibility"] = "accessibility"


@router.post("/permissions/prompt")
async def prompt_permission(req: PermissionPromptRequest):
    """Open the macOS system-prompt for Accessibility.

    The prompt only fires the first time per app bundle; afterwards macOS
    silently returns the cached answer.  When that happens the UI should
    fall back to ``Open System Settings…`` — handled client-side.
    """
    if req.kind != "accessibility":
        return {"ok": False, "error": "unsupported permission kind"}
    if platform.system() != "Darwin":
        return {"ok": False, "error": "non-macOS"}
    try:
        from ApplicationServices import (  # type: ignore[import]
            AXIsProcessTrustedWithOptions,
            kAXTrustedCheckOptionPrompt,
        )

        options = {kAXTrustedCheckOptionPrompt: True}
        granted = bool(AXIsProcessTrustedWithOptions(options))
        return {"ok": True, "granted": granted}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Accessibility prompt failed: %s", exc)
        return {"ok": False, "error": str(exc)}
