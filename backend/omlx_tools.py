"""``oMLX`` server tools for the deep agent.

Companion of :mod:`backend.exo_tools` — diagnostic and lifecycle tools
the orchestrator can invoke when the oMLX server is the active local
provider.

Deliberately excluded from agent access:

* ``install`` / ``uninstall`` — multi-minute Homebrew jobs whose
  failure modes are best surfaced in the UI install screen, not as
  agent self-modifications.
* Config mutations (port, brew tap, model_name) — managed via the
  oMLX page in the app and ``PUT /api/settings``.

If oMLX is disabled or its server isn't reachable,
:mod:`backend.session_manager` won't attach these tools.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from langchain_core.tools import tool

from backend.config import AppConfig

logger = logging.getLogger(__name__)


def _format_status(s: dict[str, Any]) -> str:
    if not s.get("reachable"):
        err = s.get("error")
        return (
            f"oMLX NOT reachable at {s.get('base_url')}"
            + (f" ({err})" if err else "")
            + ". Ask the user to start the server from the oMLX page in the app, "
            "or call omlx_start to bring it up."
        )
    models = s.get("models") or []
    if models:
        ids = ", ".join(str(m.get("id")) for m in models if m.get("id"))
    else:
        ids = "(none loaded)"
    return (
        f"oMLX reachable at {s.get('base_url')}\n"
        f"  models: {ids}"
    )


def build_omlx_tools() -> list:
    """Diagnostic and lifecycle tools for the oMLX local server."""

    @tool
    def omlx_status() -> str:
        """Show whether the local oMLX inference server is reachable and
        which models it is currently serving. No side effects.

        If the server is not reachable, call ``omlx_start`` to bring it up
        or ask the user to start it from the oMLX page in the app."""
        from backend import omlx_provisioner as op

        cfg = AppConfig.load().omlx
        loop = asyncio.new_event_loop()
        try:
            data = loop.run_until_complete(op.afetch_status(cfg))
        finally:
            loop.close()
        return _format_status(data)

    @tool
    def omlx_info() -> str:
        """Print detection state (CLI path, app bundle, Homebrew /
        services status) and the active oMLX configuration. Useful for
        diagnosing setup issues. No side effects."""
        from backend import omlx_provisioner as op

        cfg = AppConfig.load().omlx
        return json.dumps(op.info_snapshot(cfg), indent=2)

    @tool
    def omlx_tail_log(lines: int = 80) -> str:
        """Return the last ``lines`` lines of the oMLX spawn log, when Otto
        spawned the server directly (not under ``brew services``).

        When the user runs ``brew services start omlx`` themselves, the
        canonical log lives at ``$(brew --prefix)/var/log/omlx.log`` —
        the user can tail it manually if our log is empty.
        """
        from backend import omlx_provisioner as op

        n = max(1, min(2000, int(lines)))
        return "\n".join(op.tail_log(max_lines=n)) or "(spawn log empty)"

    @tool
    def omlx_start() -> str:
        """Start the local oMLX inference server if it isn't already running.

        Use this when ``omlx_status`` reports the server is unreachable and
        you need it to serve inference requests.  Idempotent: if the server
        is already up, this reports the existing state and exits cleanly.

        Does NOT install oMLX: if the CLI / app isn't found, direct the user
        to the oMLX page in the app to run the install step."""
        from backend import omlx_provisioner as op

        cfg = AppConfig.load().omlx
        loop = asyncio.new_event_loop()
        try:
            job = loop.run_until_complete(op.astart(cfg))
        finally:
            loop.close()

        return f"omlx start job kicked off (id={job.id}). Poll /api/omlx/jobs/{job.id} for progress."

    @tool
    def omlx_stop() -> str:
        """Stop the local oMLX server.

        Only call this when the user explicitly asks to shut it down.
        Stopping oMLX affects every active session that uses it as the
        inference provider — do not call autonomously as a clean-up step.
        """
        from backend import omlx_provisioner as op

        cfg = AppConfig.load().omlx
        loop = asyncio.new_event_loop()
        try:
            job = loop.run_until_complete(op.astop(cfg))
        finally:
            loop.close()

        return f"omlx stop job kicked off (id={job.id})."

    return [
        omlx_status,
        omlx_info,
        omlx_tail_log,
        omlx_start,
        omlx_stop,
    ]
