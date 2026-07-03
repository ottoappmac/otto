"""Filesystem watcher for OpenClaw session directories.

Periodically scans ``~/.openclaw/agents/*/sessions/`` for:

- **New** ``.jsonl`` files (new sessions)
- **Size changes** on existing ``.jsonl`` files (activity on resumed sessions)

Pushes synthetic events into the
:mod:`~tools.transcripts.openclaw_hook_event_buffer` so the OpenClaw
MCP server's ``wait_for_activity`` tool can wake instantly.

When auto-monitor is enabled, the watcher also creates an eval-agent
session for each new OpenClaw session — mirroring the Claude Code hook
auto-monitor behaviour in :mod:`backend.routes.hooks`.

Supports both **local** and **SSH** access modes by reusing the
:class:`~tools.transcripts.parsers.openclaw.OpenClawParser` I/O layer.

The watcher is started/stopped from the backend lifespan and gated by
``Settings → OpenClaw → Session Watcher``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class _SessionSnapshot:
    """Tracks the last-known size of a session file."""

    file_path: str
    size_bytes: int


class OpenClawWatcher:
    """Background task that detects new and changed OpenClaw sessions."""

    def __init__(self) -> None:
        # agent_id -> {session_id -> snapshot}
        self._sessions: dict[str, dict[str, _SessionSnapshot]] = {}
        self._poll_interval: int = 10
        self._task: asyncio.Task[None] | None = None
        self._auto_monitor_sessions: dict[str, str] = {}
        self._auto_monitor_lock = asyncio.Lock()
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, poll_interval: int = 10) -> None:
        """Begin the background scan loop."""
        if self._task is not None and not self._task.done():
            return
        self._poll_interval = max(5, poll_interval)
        self._task = asyncio.create_task(
            self._poll_loop(), name="openclaw-watcher",
        )
        self._started = True
        logger.info(
            "OpenClaw watcher started (poll every %ds)", self._poll_interval,
        )

    async def stop(self) -> None:
        """Cancel the background scan loop."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._started = False
        logger.info("OpenClaw watcher stopped")

    @property
    def running(self) -> bool:
        return self._started and self._task is not None and not self._task.done()

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config() -> dict[str, Any]:
        """Load OpenClaw config; returns empty dict when disabled."""
        try:
            from backend.config import AppConfig
            cfg = AppConfig.load()
            oc = cfg.openclaw
            if not oc.enabled or not oc.watcher_enabled:
                return {}
            return {
                "mode": oc.mode,
                "state_dir": oc.state_dir,
                "ssh_host": oc.ssh_host,
                "ssh_user": oc.ssh_user,
                "ssh_key_path": oc.ssh_key_path,
                "ssh_port": oc.ssh_port,
                "auto_monitor_enabled": oc.auto_monitor_enabled,
                "max_auto_sessions": oc.max_auto_sessions,
                "auto_monitor_agent": oc.auto_monitor_agent,
                "watcher_poll_interval": oc.watcher_poll_interval,
            }
        except Exception:
            return {}

    @staticmethod
    def _get_parser() -> Any:
        """Return a lazily-created parser (reuses the MCP server's singleton)."""
        from tools.transcripts.openclaw_mcp_server import _get_parser
        return _get_parser()

    # ------------------------------------------------------------------
    # Scan loop
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        await self._scan(seed=True)
        total = sum(len(s) for s in self._sessions.values())
        logger.info(
            "OpenClaw watcher seed scan complete — tracking %d agent(s), "
            "%d session(s)",
            len(self._sessions), total,
        )

        while True:
            await asyncio.sleep(self._poll_interval)

            cfg = await asyncio.to_thread(self._load_config)
            if not cfg:
                continue

            new_interval = cfg.get("watcher_poll_interval", self._poll_interval)
            if new_interval != self._poll_interval:
                self._poll_interval = max(5, new_interval)

            try:
                await self._scan(seed=False, cfg=cfg)
            except Exception:
                logger.warning("OpenClaw watcher scan failed", exc_info=True)

    async def _scan(
        self,
        seed: bool = False,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        """Walk the agents directory, detect new sessions and size changes."""
        try:
            parser = await asyncio.to_thread(self._get_parser)
        except Exception:
            return

        projects = await asyncio.to_thread(parser.list_projects)

        for project in projects:
            agent_id = project.workspace
            sessions_path = project.path
            session_files = await asyncio.to_thread(
                parser.io.list_files, sessions_path, ".jsonl",
            )

            known = self._sessions.get(agent_id, {})

            for file_path in session_files:
                session_id = file_path.rsplit("/", 1)[-1].replace(".jsonl", "")
                try:
                    size = await asyncio.to_thread(
                        parser.io.file_size, file_path,
                    )
                except Exception:
                    continue

                prev = known.get(session_id)

                if seed:
                    known[session_id] = _SessionSnapshot(
                        file_path=file_path, size_bytes=size,
                    )
                    continue

                if prev is None:
                    # Brand new session file
                    logger.info(
                        "OpenClaw watcher: new session %s (agent=%s)",
                        session_id[:12], agent_id,
                    )
                    known[session_id] = _SessionSnapshot(
                        file_path=file_path, size_bytes=size,
                    )
                    self._push_event(
                        session_id, agent_id, file_path, "SessionNew",
                    )
                    if cfg and cfg.get("auto_monitor_enabled"):
                        asyncio.create_task(
                            self._auto_start_eval_session(
                                session_id, file_path, cfg,
                            ),
                            name=f"oc-auto-monitor-{session_id[:12]}",
                        )

                elif size > prev.size_bytes:
                    # Existing session grew — activity detected
                    logger.info(
                        "OpenClaw watcher: activity on %s (+%d bytes)",
                        session_id[:12], size - prev.size_bytes,
                    )
                    prev.size_bytes = size
                    self._push_event(
                        session_id, agent_id, file_path, "SessionActivity",
                    )
                    if (
                        cfg
                        and cfg.get("auto_monitor_enabled")
                        and session_id not in self._auto_monitor_sessions
                    ):
                        asyncio.create_task(
                            self._auto_start_eval_session(
                                session_id, file_path, cfg,
                                event_type="SessionActivity",
                            ),
                            name=f"oc-auto-monitor-{session_id[:12]}",
                        )

            self._sessions[agent_id] = known

    # ------------------------------------------------------------------
    # Event push
    # ------------------------------------------------------------------

    @staticmethod
    def _push_event(
        session_id: str,
        agent_id: str,
        file_path: str,
        event_name: str = "SessionNew",
    ) -> None:
        """Push a synthetic event into the buffer."""
        from tools.transcripts.openclaw_hook_event_buffer import oc_hook_buffer

        oc_hook_buffer.push(session_id, {
            "hook_event_name": event_name,
            "session_id": session_id,
            "agent_id": agent_id,
            "transcript_path": file_path,
            "timestamp": time.time(),
        })

    # ------------------------------------------------------------------
    # Auto-monitor (mirrors backend/routes/hooks.py logic)
    # ------------------------------------------------------------------

    async def _auto_start_eval_session(
        self,
        oc_session_id: str,
        transcript_path: str,
        cfg: dict[str, Any],
        event_type: str = "SessionNew",
    ) -> None:
        """Create an eval-agent session for a new OpenClaw session."""
        from backend.config import AppConfig
        from backend.routes.sessions import _LazyPersistingSubagentQueue
        from backend.state import running_tasks, session_mgr
        from backend.streaming_subagent import reset_subagent_queue, set_subagent_queue

        async with self._auto_monitor_lock:
            if oc_session_id in self._auto_monitor_sessions:
                return

            max_sessions = cfg.get("max_auto_sessions", 3)
            active = self._count_active_auto_sessions()
            if active >= max_sessions:
                logger.info(
                    "OpenClaw auto-monitor: cap reached (%d/%d) — skipping %s",
                    active, max_sessions, oc_session_id[:12],
                )
                return

            try:
                app_cfg = AppConfig.load()
                trigger = (
                    "oc-watcher-new" if event_type == "SessionNew"
                    else "oc-watcher-activity"
                )
                agent_name = cfg.get("auto_monitor_agent") or "openclaw-session-eval-agent"
                session = await session_mgr.create_session(
                    config=app_cfg,
                    agent_name=agent_name,
                    trigger_source=trigger,
                )
            except Exception:
                logger.error(
                    "OpenClaw auto-monitor: failed to create session for %s",
                    oc_session_id[:12],
                    exc_info=True,
                )
                return

            self._auto_monitor_sessions[oc_session_id] = session.id
            logger.info(
                "OpenClaw auto-monitor: created eval session %s for OC session %s "
                "(transcript: %s)",
                session.id[:12], oc_session_id[:12], transcript_path,
            )

        prompt = (
            f"A new OpenClaw session has been detected by the session watcher. "
            f"Evaluate this session in real-time.\n\n"
            f"Session path: {transcript_path}\n\n"
            f"Monitor the session, wait for activity, then evaluate the quality "
            f"and efficiency of the agent's work. Report your findings when the "
            f"session ends or after significant milestones."
        )

        lazy_queue = _LazyPersistingSubagentQueue(session.id)
        token = set_subagent_queue(lazy_queue, session_id=session.id)

        async def _run() -> None:
            try:
                async for resp in session_mgr.stream_message(session.id, prompt):
                    await lazy_queue.put(resp)
            except asyncio.CancelledError:
                logger.info(
                    "OpenClaw auto-monitor session %s cancelled",
                    session.id[:12],
                )
            except Exception as exc:
                logger.error(
                    "OpenClaw auto-monitor session %s failed: %s",
                    session.id[:12], exc,
                    exc_info=True,
                )
            finally:
                reset_subagent_queue(token, session_id=session.id)
                running_tasks.pop(session.id, None)

        task = asyncio.create_task(
            _run(), name=f"oc-auto-eval-{session.id[:12]}",
        )
        running_tasks[session.id] = task

    def _count_active_auto_sessions(self) -> int:
        """Prune finished sessions and return the active count."""
        from backend.state import running_tasks

        stale = [
            oc_id for oc_id, cura_id in self._auto_monitor_sessions.items()
            if cura_id not in running_tasks or running_tasks[cura_id].done()
        ]
        for oc_id in stale:
            self._auto_monitor_sessions.pop(oc_id, None)

        return len(self._auto_monitor_sessions)

    # ------------------------------------------------------------------
    # Status (used by the /hooks/openclaw/status endpoint)
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Return watcher status for the status endpoint."""
        from tools.transcripts.openclaw_hook_event_buffer import oc_hook_buffer

        return {
            "running": self.running,
            "poll_interval": self._poll_interval,
            "tracked_agents": len(self._sessions),
            "tracked_sessions": sum(
                len(s) for s in self._sessions.values()
            ),
            "buffered_events": oc_hook_buffer.active_session_count(),
            "auto_monitor": {
                "active": dict(self._auto_monitor_sessions),
                "count": len(self._auto_monitor_sessions),
            },
        }


# Module-level singleton
oc_watcher = OpenClawWatcher()
