"""Last-resort context-window budget enforcement for small-context models.

Small on-device models (those with a 2 K–8 K token hard limit) cannot
accommodate Otto's default agent middleware stack.  The deepagents
framework hardcodes :class:`TodoListMiddleware`,
:class:`FilesystemMiddleware`, and :class:`SubAgentMiddleware`, each of
which injects a multi-hundred-token system-prompt block of its own.
Combined with Otto's lite orchestrator prompt, the deepagents
``BASE_AGENT_PROMPT``, and the ReAct tool descriptions, the assembled
system prompt routinely exceeds 3 000 tokens — *before any user message
is added* — which on a 4 K-token model leaves no room for the response.

This middleware runs **innermost** (last in the middleware list) so it
sees the fully-assembled :class:`ModelRequest` after every other
middleware has had a chance to modify it.  When the estimated input
token count exceeds the configured budget it:

1. Drops conversation messages oldest-first until the budget is met.
2. If still over, truncates the *tail* of the system message (the head
   typically carries the agent identity and the load-bearing rules; tail
   content is usually middleware-injected boilerplate that the agent can
   live without for one turn).
3. If still over, shrinks OpenAI tool schemas (name + one-line description)
   and then drops non-core tools.  oMLX counts the ``tools`` array in the
   prompt; a distilled 8B student overflows at 40 k before any user text.

The middleware is a no-op for models whose
``profile["max_input_tokens"]`` is generous; gating happens at
construction time, so it is safe to include unconditionally.

Token estimation is intentionally cheap and conservative: roughly 3
characters per token for English/Spanish/German (Apple's own published
figure is 3–4).  Erring low means we sometimes trim more than strictly
necessary — which is exactly the right failure mode when the
alternative is a hard ``exceededContextWindowSize`` from Apple.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

logger = logging.getLogger(__name__)

_KEEP_TOOL_NAMES = {
    "task",
    "write_todos",
    "write_file",
    "read_file",
    "ls",
    "edit_file",
    "execute",
    "ask_user",
}


def _tool_name(tool: Any) -> str:
    if isinstance(tool, dict):
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        return str((fn or {}).get("name") or tool.get("name") or "")
    return str(getattr(tool, "name", "") or "")


def _tool_description(tool: Any) -> str:
    if isinstance(tool, dict):
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        return str((fn or {}).get("description") or tool.get("description") or "")
    return str(getattr(tool, "description", "") or "")


def _tool_schema_text(tool: Any) -> str:
    if isinstance(tool, dict):
        return json.dumps(tool, default=str)
    desc = _tool_description(tool)
    schema = ""
    args = getattr(tool, "args_schema", None)
    if args is not None:
        try:
            if hasattr(args, "model_json_schema"):
                schema = json.dumps(args.model_json_schema(), default=str)
            elif hasattr(args, "schema"):
                schema = json.dumps(args.schema(), default=str)
            else:
                schema = str(args)
        except Exception:
            schema = str(args)
    return f"{_tool_name(tool)}\n{desc}\n{schema}"


def _compact_tool_schema(tool: Any) -> dict[str, Any]:
    desc = _tool_description(tool).split(".")[0].strip()
    if len(desc) > 160:
        desc = desc[:157] + "…"
    return {
        "type": "function",
        "function": {
            "name": _tool_name(tool),
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            },
        },
    }


class SmallContextTruncationMiddleware(AgentMiddleware):
    """Clip the assembled request to fit a small context window.

    Place this middleware **last** in the ``middleware`` list passed to
    ``create_agent`` / ``create_deep_agent`` so it observes the fully
    composed system prompt and message history right before the model
    call.

    Args:
        max_input_tokens: Hard ceiling on input tokens (system + all
            messages).  Typically computed as
            ``context_window - max_output_tokens`` for the target model.
        safety_margin_tokens: Subtracted from ``max_input_tokens`` to leave
            headroom for tokenizer mismatch.  Defaults to 256.
        chars_per_token: Estimation ratio.  Defaults to 3.0 (conservative
            for English; Apple cites 3–4 chars per token).
        min_messages_kept: Always keep at least this many of the most
            recent messages, even if the budget would otherwise force
            dropping them.  Defaults to 1 so the user's current request
            is never dropped.
    """

    def __init__(
        self,
        *,
        max_input_tokens: int,
        safety_margin_tokens: int = 256,
        chars_per_token: float = 3.0,
        min_messages_kept: int = 1,
    ) -> None:
        if max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive")
        self._budget = max(1, max_input_tokens - max(0, safety_margin_tokens))
        self._cpt = max(1.0, float(chars_per_token))
        self._min_kept = max(1, int(min_messages_kept))

    # ── AgentMiddleware hooks ──────────────────────────────────────────────

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._fit(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._fit(request))

    # ── Estimation + fitting ──────────────────────────────────────────────

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return int(len(text) / self._cpt) + 1

    def _message_text(self, msg: Any) -> str:
        content = getattr(msg, "content", "") or ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
            return "\n".join(p for p in parts if p)
        return str(content)

    def _system_text(self, system_message: Any) -> str:
        if system_message is None:
            return ""
        return self._message_text(system_message)

    def _fit(self, request: ModelRequest) -> ModelRequest:
        """Return a request whose assembled prompt fits in the budget."""
        sys_text = self._system_text(request.system_message)
        msgs = list(request.messages)
        tools = list(request.tools or [])

        sys_tokens = self._estimate_tokens(sys_text)
        msg_tokens = [self._estimate_tokens(self._message_text(m)) for m in msgs]
        tool_tokens = sum(self._estimate_tokens(_tool_schema_text(t)) for t in tools)
        total = sys_tokens + sum(msg_tokens) + tool_tokens

        if total <= self._budget:
            return request

        original_total = total
        dropped_messages = 0
        tools_compacted = False

        # Step 0: shrink the OpenAI tools array first.  oMLX counts it in
        # the prompt; a distilled 8B student overflows here on turn one.
        if total > self._budget and tools:
            compacted = [_compact_tool_schema(t) for t in tools if _tool_name(t)]
            new_tool_tokens = sum(
                self._estimate_tokens(_tool_schema_text(t)) for t in compacted
            )
            if new_tool_tokens < tool_tokens:
                tools = compacted
                total = total - tool_tokens + new_tool_tokens
                tool_tokens = new_tool_tokens
                tools_compacted = True
            keep: list[Any] = []
            rest: list[Any] = []
            for t in tools:
                (keep if _tool_name(t) in _KEEP_TOOL_NAMES else rest).append(t)
            while total > self._budget and rest:
                dropped = rest.pop()
                dropped_tok = self._estimate_tokens(_tool_schema_text(dropped))
                tool_tokens -= dropped_tok
                total -= dropped_tok
                tools_compacted = True
            tools = keep + rest

        # Step 1: drop oldest messages (preserving at least the last
        # ``_min_kept`` so the user's current turn always reaches the model).
        while total > self._budget and len(msgs) > self._min_kept:
            removed_tokens = msg_tokens.pop(0)
            msgs.pop(0)
            total -= removed_tokens
            dropped_messages += 1

        # Step 1b: purge any orphaned ToolMessages that now lead the list.
        #
        # When an AIMessage with tool_calls is dropped above, the ToolMessages
        # that follow it reference tool_use_ids that no longer exist in the
        # remaining history.  Anthropic's API rejects such requests with:
        #   "unexpected `tool_use_id` found in `tool_result` blocks"
        # We collect all tool_call_ids still present in remaining AIMessages
        # and strip any leading ToolMessage whose id is not among them.
        valid_tc_ids: set[str] = set()
        for m in msgs:
            if isinstance(m, AIMessage):
                for tc in (getattr(m, "tool_calls", None) or []):
                    tc_id = tc.get("id") or ""
                    if tc_id:
                        valid_tc_ids.add(tc_id)

        while msgs and isinstance(msgs[0], ToolMessage):
            tc_id = getattr(msgs[0], "tool_call_id", None) or ""
            if tc_id not in valid_tc_ids:
                orphan_tokens = msg_tokens.pop(0)
                msgs.pop(0)
                total -= orphan_tokens
                dropped_messages += 1
            else:
                break

        # Step 2: if still over budget, clip the *tail* of the system message.
        # Head usually carries identity + critical rules; tail is typically
        # middleware-injected boilerplate (todos / filesystem / subagents).
        sys_truncated = False
        new_system_message = request.system_message
        if total > self._budget and sys_tokens > 0:
            allowed_sys_tokens = max(0, self._budget - sum(msg_tokens) - tool_tokens)
            allowed_chars = int(allowed_sys_tokens * self._cpt)
            if allowed_chars < len(sys_text):
                clipped = sys_text[: max(0, allowed_chars - 64)].rstrip()
                clipped += (
                    "\n\n[Note: middleware prompt sections trimmed to fit the "
                    "model's context window.  Keep responses concise.]"
                )
                new_system_message = SystemMessage(content=clipped)
                sys_truncated = True
                total = (
                    self._estimate_tokens(clipped) + sum(msg_tokens) + tool_tokens
                )

        if dropped_messages or sys_truncated or tools_compacted:
            logger.warning(
                "SmallContextTruncationMiddleware: trimmed request to fit budget "
                "(budget=%d tok, before=%d tok, after=%d tok, dropped_messages=%d, "
                "system_truncated=%s, tools=%d)",
                self._budget,
                original_total,
                total,
                dropped_messages,
                sys_truncated,
                len(tools),
            )

        if (
            new_system_message is request.system_message
            and not dropped_messages
            and not tools_compacted
        ):
            return request
        return request.override(
            system_message=new_system_message, messages=msgs, tools=tools,
        )
