"""Background memory extraction middleware.

Runs a lightweight LLM call after the agent loop to extract durable
learnings from the conversation and persist them to AGENTS.md files
-- without requiring the main agent to remember to do it itself.

Adapted from Claude Code's ``extractMemories`` background agent
pattern.

Usage::

    from backend.memory_extraction import MemoryExtractionMiddleware

    mw = MemoryExtractionMiddleware(
        model=llm,
        memory_path="/path/to/AGENTS.md",
        extract_every_n_turns=3,
    )
    graph = create_deep_agent(..., middleware=[mw])
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
)
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

_background_tasks: set[asyncio.Task[None]] = set()


def _task_done(task: asyncio.Task[None]) -> None:
    """Remove completed tasks and log any unhandled exceptions."""
    _background_tasks.discard(task)
    if not task.cancelled() and task.exception():
        logger.warning(
            "Memory extraction background task failed",
            exc_info=task.exception(),
        )


# ------------------------------------------------------------------
# Extraction prompt
# ------------------------------------------------------------------

MEMORY_EXTRACTION_PROMPT = """\
You are the memory extraction agent. Analyze the recent \
conversation messages above and determine if any durable \
learnings should be saved.

## What to extract

- **User preferences**: role, communication style, \
preferred languages/frameworks
- **Feedback**: corrections the user made, approaches \
they confirmed or rejected, and *why*
- **Project context**: ongoing goals, constraints, \
deadlines, team coordination details not derivable \
from code alone
- **External references**: links, tool IDs, Slack \
channels, dashboards the user mentioned for future use

## What NOT to extract

- Code patterns, architecture, or file structure -- \
these are derivable from the codebase itself
- Ephemeral task details, one-off questions, or \
transient state
- Anything already present in the existing memory below
- API keys, passwords, tokens, or any credentials

## Existing memory

<existing_memory>
{existing_memory}
</existing_memory>

## Instructions

If there are durable learnings worth saving, output \
them in the exact format below. If there is nothing \
new worth saving, respond with exactly: NO_UPDATES

Format for updates (append-friendly markdown):

```
## Learned Preferences
- [preference or pattern]

## User Feedback
- [correction or confirmation, with why]

## Project Context
- [relevant context]

## References
- [external resource pointer]
```

Only include sections that have new content. Be \
concise -- each bullet should be one or two sentences. \
Do not duplicate information already in existing \
memory."""


# ------------------------------------------------------------------
# Middleware
# ------------------------------------------------------------------


class MemoryExtractionMiddleware(AgentMiddleware):
    """Extract and persist learnings after each agent turn.

    Uses ``aafter_agent`` to run a lightweight LLM call that
    analyzes recent messages and writes durable learnings to an
    AGENTS.md file.

    The extraction is throttled to run only every
    *extract_every_n_turns* agent turns to balance reliability
    with cost. If the agent already wrote to the memory file
    during its turn, extraction is skipped (mutual exclusion
    with agent-driven memory updates).

    Args:
        model: Chat model for the extraction call.
        memory_path: Filesystem path to the AGENTS.md to
            update. Must be absolute (not a virtual path).
        extract_every_n_turns: Run extraction every N turns.
    """

    def __init__(
        self,
        *,
        model: BaseChatModel,
        memory_path: str | Path,
        extract_every_n_turns: int = 3,
    ) -> None:
        self._model = model
        self._memory_path = Path(memory_path)
        self._extract_every_n = max(1, extract_every_n_turns)
        self._turn_count = 0

    # -- helpers ------------------------------------------------

    def _read_existing_memory(self) -> str:
        """Read current AGENTS.md content from disk."""
        try:
            if self._memory_path.exists():
                return self._memory_path.read_text(
                    encoding="utf-8",
                )
        except Exception:
            logger.debug(
                "Could not read memory file %s",
                self._memory_path,
                exc_info=True,
            )
        return "(No existing memory)"

    def _write_memory_update(
        self, existing: str, new_content: str,
    ) -> None:
        """Append extracted learnings to the file."""
        try:
            self._memory_path.parent.mkdir(
                parents=True, exist_ok=True,
            )
            sep = "\n\n" if existing.strip() else ""
            updated = (
                existing.rstrip()
                + sep
                + new_content.strip()
                + "\n"
            )
            self._memory_path.write_text(
                updated, encoding="utf-8",
            )
            logger.info(
                "Memory extraction: updated %s",
                self._memory_path,
            )
        except Exception:
            logger.warning(
                "Memory extraction: failed to write %s",
                self._memory_path,
                exc_info=True,
            )

    def _has_memory_writes_in_messages(
        self, messages: list,
    ) -> bool:
        """Check if the agent already wrote to memory.

        Looks for edit_file / write_file tool calls targeting
        the memory path in recent AI messages.
        """
        memory_name = self._memory_path.name
        for msg in reversed(messages[-10:]):
            if not isinstance(msg, AIMessage):
                continue
            for tc in getattr(msg, "tool_calls", []) or []:
                name = tc.get("name", "")
                if name not in ("edit_file", "write_file"):
                    continue
                args = tc.get("args", {})
                fpath = (
                    args.get("file_path", "")
                    or args.get("path", "")
                )
                if memory_name in str(fpath):
                    return True
        return False

    def _get_recent_conversation_text(
        self, messages: list, max_messages: int = 20,
    ) -> str:
        """Format recent messages for the extraction prompt."""
        recent = messages[-max_messages:]
        parts: list[str] = []
        for msg in recent:
            if isinstance(msg, HumanMessage):
                content = (
                    msg.content
                    if isinstance(msg.content, str)
                    else str(msg.content)
                )
                parts.append(f"User: {content[:2000]}")
            elif isinstance(msg, AIMessage):
                content = msg.content
                if isinstance(content, list):
                    content = " ".join(
                        p.get("text", "")
                        for p in content
                        if isinstance(p, dict)
                        and p.get("type") == "text"
                    )
                if content and str(content).strip():
                    parts.append(
                        f"Assistant: {str(content)[:2000]}"
                    )
        return "\n\n".join(parts)

    # -- extraction core ----------------------------------------

    async def _run_extraction(self, messages: list) -> None:
        """Run the extraction LLM call and persist results.

        File I/O is offloaded to a thread to avoid blocking the
        event loop (this method runs as a fire-and-forget task).
        """
        existing = await asyncio.to_thread(
            self._read_existing_memory,
        )

        conversation = self._get_recent_conversation_text(
            messages,
        )
        if not conversation.strip():
            return

        prompt = MEMORY_EXTRACTION_PROMPT.format(
            existing_memory=existing,
        )

        try:
            response = await self._model.ainvoke([
                SystemMessage(content=prompt),
                HumanMessage(
                    content=(
                        "Recent conversation:\n\n"
                        + conversation
                    ),
                ),
            ])

            result = response.content
            if isinstance(result, list):
                result = " ".join(
                    p.get("text", "")
                    for p in result
                    if isinstance(p, dict)
                    and p.get("type") == "text"
                )

            result = str(result).strip()

            if not result or "NO_UPDATES" in result:
                logger.debug(
                    "Memory extraction: no updates needed",
                )
                return

            await asyncio.to_thread(
                self._write_memory_update, existing, result,
            )

        except Exception:
            logger.warning(
                "Memory extraction: LLM call failed",
                exc_info=True,
            )

    # -- middleware hooks ----------------------------------------

    async def aafter_agent(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Run memory extraction after the agent turn."""
        self._turn_count += 1

        if self._turn_count % self._extract_every_n != 0:
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        if self._has_memory_writes_in_messages(messages):
            logger.debug(
                "Memory extraction: skipped -- "
                "agent already wrote to memory",
            )
            return None

        task = asyncio.create_task(
            self._run_extraction(messages),
            name="memory-extraction",
        )
        _background_tasks.add(task)
        task.add_done_callback(_task_done)
        return None

    def after_agent(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Sync fallback -- schedules extraction async."""
        self._turn_count += 1

        if self._turn_count % self._extract_every_n != 0:
            return None

        messages = state.get("messages", [])
        if not messages:
            return None

        if self._has_memory_writes_in_messages(messages):
            logger.debug(
                "Memory extraction: skipped -- "
                "agent already wrote to memory",
            )
            return None

        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(
                self._run_extraction(messages),
                name="memory-extraction",
            )
            _background_tasks.add(task)
            task.add_done_callback(_task_done)
        except RuntimeError:
            pass
        return None
