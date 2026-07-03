"""``exo`` cluster tools for the deep agent.

Diagnostic (read-only) tools let the agent inspect the cluster without
side effects.  Lifecycle tools (start / stop) allow the agent to bring
the local daemon up when needed or shut it down at the user's request.

Deliberately excluded from agent access:
* ``provision`` — involves repo clones, venv builds, and SSH; failure
  modes are hard to recover from mid-run.
* Remote up/down — SSH access to other machines is a human action.
* Config mutations (ports, remotes, model settings) — managed via
  ``ExoPage`` UI and ``PUT /api/settings``.

If the cluster is disabled or not running, ``backend.session_manager``
won't attach these tools — see the gating logic there.
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
            f"exo NOT reachable at {s.get('base_url')}"
            + (f" ({err})" if err else "")
            + ". Ask the user to start the cluster from the exo page in the app."
        )
    lines = [
        f"exo reachable at {s.get('base_url')}",
        f"  master_node_id : {s.get('master_node_id')}",
        f"  nodes          : {s.get('peer_count')}",
    ]
    for n in s.get("nodes") or []:
        chip = n.get("chip") or "?"
        name = n.get("friendly_name") or "?"
        if n.get("memory_total_gb"):
            mem = f"{n.get('memory_free_gb')}/{n.get('memory_total_gb')} GB free"
        else:
            mem = "memory:?"
        nid = (n.get("node_id") or "")[:14]
        lines.append(f"    - {nid}…  {name}  ({chip}, {mem})")
    lines.append(f"  rdma edges     : {s.get('rdma_connections')}")
    lines.append(f"  loaded models  : {', '.join(s.get('loaded_models') or []) or '(none)'}")
    return "\n".join(lines)


def build_exo_tools() -> list:
    """Return exo diagnostic and lifecycle tools for the deep agent.

    ``backend.session_manager`` only calls this when exo is enabled *and*
    the local daemon is already running, so the agent always starts with
    a reachable cluster.
    """

    @tool
    def exo_status() -> str:
        """Show live status of the local ``exo`` distributed inference
        cluster: which nodes have joined, what models are loaded, and
        whether the HTTP API is reachable. No side effects.

        If the cluster is not reachable, call ``exo_start`` to bring it
        up, or ask the user to start it from the exo page in the app."""
        from backend import exo_provisioner as ep

        cfg = AppConfig.load().exo
        loop = asyncio.new_event_loop()
        try:
            data = loop.run_until_complete(ep.afetch_status(cfg))
        finally:
            loop.close()
        return _format_status(data)

    @tool
    def exo_info() -> str:
        """Print resolved exo paths, prereqs, install state, and the
        active configuration. Useful for diagnosing setup issues. No
        side effects."""
        from backend import exo_provisioner as ep

        cfg = AppConfig.load().exo
        snap = ep.info_snapshot(cfg)
        return json.dumps(snap, indent=2)

    @tool
    def exo_tail_log(lines: int = 80) -> str:
        """Return the last ``lines`` lines of the local ``exo.log`` file.

        Use to diagnose why an inference call against the cluster
        misbehaved or why discovery is unstable. Defaults to 80 lines
        (max 2000). Read-only."""
        from backend import exo_provisioner as ep

        n = max(1, min(2000, int(lines)))
        return "\n".join(ep.tail_log(max_lines=n)) or "(log empty)"

    @tool
    def list_exo_remotes() -> str:
        """List configured remote (secondary) nodes in the cluster.

        Read-only. Adding, removing, starting, or stopping remotes is a
        human action — direct the user to the exo page in the app."""
        cfg = AppConfig.load().exo
        if not cfg.remotes:
            return (
                "No exo remotes configured. The user can add one from the "
                "exo page in the app."
            )
        out = []
        for r in cfg.remotes:
            label = f"{r.label} — " if r.label else ""
            disabled = "" if r.enabled else " [disabled]"
            out.append(f"- {label}{r.ssh_alias}{disabled}")
        return "\n".join(out)

    @tool
    def exo_start() -> str:
        """Start the local ``exo`` cluster daemon if it is not already running.

        Use this when ``exo_status`` reports the cluster is unreachable and
        you need it to serve inference requests. Safe to call if the daemon
        is already running — it will report the current pid and confirm the
        cluster is up.

        Does NOT re-provision: if the cluster has never been set up, direct
        the user to the exo page in the app to run the initial provision step.
        """
        from backend import exo_provisioner as ep
        from backend.exo_provisioner import ExoCliError

        cfg = AppConfig.load().exo
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(ep.astart_local(cfg))
        except ExoCliError as exc:
            return (
                f"Failed to start exo daemon: {exc}. "
                "If the cluster has not been provisioned yet, ask the user to "
                "run the initial setup from the exo page in the app."
            )
        finally:
            loop.close()

        if result.get("running"):
            pid = result.get("pid") or "unknown"
            return f"exo daemon started (pid {pid}). Cluster is reachable."
        return (
            "exo start command ran but the daemon does not appear to be running. "
            "Check the log with exo_tail_log for details."
        )

    @tool
    def exo_stop() -> str:
        """Stop the local ``exo`` cluster daemon.

        Only call this when the user explicitly asks to shut down the cluster.
        Stopping exo affects ALL active sessions that use it as their inference
        provider — do not call autonomously as a clean-up step.
        """
        from backend import exo_provisioner as ep

        cfg = AppConfig.load().exo
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(ep.astop_local(cfg))
        finally:
            loop.close()

        if result.get("stopped"):
            return "exo daemon stopped successfully."
        still_running = result.get("running")
        if still_running:
            return (
                "Stop command ran but the daemon appears to still be running. "
                "Check the log with exo_tail_log for details."
            )
        return "exo daemon was not running (nothing to stop)."

    @tool
    def exo_load_model(
        model_id: str,
        min_nodes: int = 1,
        timeout_seconds: int = 1800,
    ) -> str:
        """Load a model into exo cluster memory so it can serve inference.

        REQUIRED for EXO: unlike MLX/Anthropic, exo does not auto-load on first
        request — the model must be explicitly placed onto cluster nodes via
        POST /place_instance before any chat completion can route to it.

        Use this tool whenever:
          - The user asks to switch to an exo model that is downloaded but not
            loaded (``list_exo_models`` shows ``[downloaded]`` without ``[loaded]``).
          - You just called ``switch_model_provider(provider='exo', ...)`` for
            a model that isn't already loaded.
          - The user wants to "warm up" a different model before switching.

        This call BLOCKS until the model is loaded and ready (or timeout).
        Frontier-class models can take several minutes; default timeout is 30
        minutes which matches the UI's preload behaviour.

        Args:
            model_id: The exo catalogue model id (e.g.
                "mlx-community/Qwen3.6-35B-A3B-4bit"). Use ``list_exo_models``
                to see what's available.
            min_nodes: Force pipeline-parallel placement across at least this
                many nodes. Default 1 = let the scheduler pick the cheapest
                single-node placement.
            timeout_seconds: Maximum seconds to wait for placement and load.
                Default 1800 (30 min). Decrease only if you want to bail
                quickly when the cluster is busy.
        """
        from backend import exo_provisioner as ep

        cfg = AppConfig.load().exo

        # Skip if already loaded — preload defaults to replace_existing=True
        # which would tear down a working instance just to recreate it.
        try:
            loop = asyncio.new_event_loop()
            try:
                listing = loop.run_until_complete(ep.alist_models(cfg))
            finally:
                loop.close()
            if any(
                m.get("id") == model_id and m.get("loaded")
                for m in (listing.get("models") or [])
            ):
                return (
                    f"Model '{model_id}' is already loaded in exo cluster memory. "
                    "Nothing to do."
                )
        except Exception:
            pass

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                ep.apreload_model(
                    cfg,
                    model_id,
                    timeout=float(timeout_seconds),
                    min_nodes=int(min_nodes),
                )
            )
        finally:
            loop.close()

        if result.get("ok"):
            elapsed = result.get("elapsed_seconds", 0)
            replaced = result.get("replaced", 0)
            replaced_note = f" (replaced {replaced} existing instance)" if replaced else ""
            return (
                f"Model '{model_id}' loaded into exo cluster memory in {elapsed}s"
                f"{replaced_note}. Ready for inference."
            )
        err = result.get("error", "unknown error")
        detail = result.get("detail", "")
        detail_part = f"\nDetail: {detail}" if detail else ""
        return (
            f"Failed to load '{model_id}' into exo cluster: {err}{detail_part}\n"
            "Check exo_tail_log for cluster-side errors. Common causes: model "
            "not yet downloaded (see list_exo_models), cluster offline, or "
            "insufficient memory across nodes (try increasing min_nodes)."
        )

    return [
        exo_status,
        exo_info,
        exo_tail_log,
        list_exo_remotes,
        exo_start,
        exo_stop,
        exo_load_model,
    ]
