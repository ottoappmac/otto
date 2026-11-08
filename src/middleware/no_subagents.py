"""Hide the ``task`` (subagent) tool from an agent.

``create_deep_agent`` always adds ``SubAgentMiddleware``, which binds a
``task`` tool and appends its instructions to the system prompt, even when no
subagents are passed. Some agents must work alone: the SWE-bench agent's
subagents get neither its prompt nor its environment rules, and in benchmark
runs they spent many minutes searching the wrong folders.

This middleware must sit after ``SubAgentMiddleware`` (any position in the
``middleware=`` list passed to ``create_deep_agent`` is). It drops the ``task``
tool and the instruction block from each model request, and refuses a
``task`` call the model makes anyway.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage, ToolMessage

TASK_TOOL = "task"
_TASK_PROMPT_HEADING = "## `task` (subagent spawner)"
_REFUSAL = (
    "The task tool is not available to this agent. Do the work yourself "
    "with the file and shell tools."
)


def _tool_name(tool: Any) -> str | None:
    if isinstance(tool, dict):
        return tool.get("name")
    return getattr(tool, "name", None)


def _strip_task_prompt(message: SystemMessage | None) -> SystemMessage | None:
    if message is None:
        return None
    blocks = list(message.content_blocks)
    kept = [
        b for b in blocks
        if not (b.get("type") == "text"
                and str(b.get("text", "")).lstrip().startswith(_TASK_PROMPT_HEADING))
    ]
    if len(kept) == len(blocks):
        return message
    return SystemMessage(content_blocks=kept)


class NoSubagentsMiddleware(AgentMiddleware):
    def _filter(self, request: ModelRequest) -> ModelRequest:
        tools = [t for t in request.tools if _tool_name(t) != TASK_TOOL]
        return request.override(
            tools=tools, system_message=_strip_task_prompt(request.system_message),
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._filter(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._filter(request))

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        if request.tool_call.get("name") == TASK_TOOL:
            return ToolMessage(_REFUSAL, tool_call_id=request.tool_call.get("id") or "",
                               name=TASK_TOOL, status="error")
        return handler(request)

    async def awrap_tool_call(self, request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
        if request.tool_call.get("name") == TASK_TOOL:
            return ToolMessage(_REFUSAL, tool_call_id=request.tool_call.get("id") or "",
                               name=TASK_TOOL, status="error")
        return await handler(request)
