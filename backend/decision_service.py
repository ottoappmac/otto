"""One Jev question against whichever endpoint Settings selected.

Shared by the ``jev`` tool and the safety guardrail so both resolve the
endpoint, honour the privacy lock, and recover from a stale sidecar key the
same way. Every failure is raised as :class:`DecisionUnavailable` with a
message that is safe to show to the agent or the user.

The modules this one depends on are imported at call time so a caller (or a
test) that replaces them sees the replacement.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.config import AppConfig
    from backend.decision_client import DecisionAnswer

logger = logging.getLogger(__name__)


class DecisionUnavailable(Exception):
    """Jev could not give an answer. ``str(exc)`` says why."""


def ask_once(
    cfg: "AppConfig",
    situation: str,
    question: str,
    options: Any = None,
    scale: Any = None,
    *,
    timeout: float | None = None,
) -> "DecisionAnswer":
    """Ask one question. Starts the local sidecar when it is not running.

    Does not check ``cfg.decision.enabled``: that flag gates the ``jev`` tool,
    and the safety guardrail has its own. Does not count against the tool's
    per-turn call cap.
    """
    import httpx

    from backend import decision_client
    from backend.decision_client import DecisionClientError
    from backend.decision_runtime import (
        effective_base_url,
        ensure_running,
        local_api_key,
        note_used,
        remote_api_key,
        stop_sync,
        uses_local_sidecar,
    )
    from backend.privacy_lock import PrivacyLockActive, enforce_decision_endpoint

    decision = cfg.decision
    local = uses_local_sidecar(decision)
    base = effective_base_url(decision)
    try:
        enforce_decision_endpoint("" if local else base, cfg)
    except PrivacyLockActive as exc:
        raise DecisionUnavailable(str(exc)) from exc

    if local:
        try:
            base = ensure_running(decision)
        except Exception as exc:
            logger.info("jev: sidecar not ready: %s", exc)
            raise DecisionUnavailable(str(exc)) from exc
        api_key = local_api_key()
    else:
        api_key = remote_api_key() or ""
        if not api_key:
            raise DecisionUnavailable(
                "A custom Jev endpoint is set but no API key is stored."
            )

    extra = {} if timeout is None else {"timeout": timeout}

    def _call(url: str, key: str) -> "DecisionAnswer":
        return decision_client.ask(
            situation,
            question,
            options,
            scale,
            base_url=url,
            api_key=key,
            **extra,
        )

    try:
        try:
            result = _call(base, api_key)
        except httpx.HTTPStatusError as exc:
            # A Settings save used to rotate the key out from under a
            # sidecar that was already up. Restart once and retry.
            if exc.response.status_code == 401 and local:
                logger.info("jev: sidecar rejected the API key; restarting")
                stop_sync()
                base = ensure_running(decision)
                api_key = local_api_key()
                result = _call(base, api_key)
            else:
                raise
    except DecisionClientError as exc:
        raise DecisionUnavailable(str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        logger.info("jev: HTTP %s", code)
        if code == 401:
            raise DecisionUnavailable(
                "The Jev server rejected the API key (HTTP 401)."
            ) from exc
        raise DecisionUnavailable(f"The Jev server returned HTTP {code}.") from exc
    except httpx.TimeoutException as exc:
        raise DecisionUnavailable("The Jev server timed out.") from exc
    except Exception as exc:
        logger.info("jev: request failed: %s", exc)
        raise DecisionUnavailable(
            "The Jev server did not answer (connection error)."
        ) from exc
    note_used()
    return result
