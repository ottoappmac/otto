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
3. Logs a structured warning so the operator can spot the situation in
   ``backend.log`` and either tune the prompt or use a bigger model.

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

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

logger = logging.getLogger(__name__)


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

        sys_tokens = self._estimate_tokens(sys_text)
        msg_tokens = [self._estimate_tokens(self._message_text(m)) for m in msgs]
        total = sys_tokens + sum(msg_tokens)

        if total <= self._budget:
            return request

        original_total = total
        dropped_messages = 0

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
            allowed_sys_tokens = max(0, self._budget - sum(msg_tokens))
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
                    self._estimate_tokens(clipped) + sum(msg_tokens)
                )

        if dropped_messages or sys_truncated:
            logger.warning(
                "SmallContextTruncationMiddleware: trimmed request to fit budget "
                "(budget=%d tok, before=%d tok, after=%d tok, dropped_messages=%d, "
                "system_truncated=%s)",
                self._budget,
                original_total,
                total,
                dropped_messages,
                sys_truncated,
            )

        if new_system_message is request.system_message and not dropped_messages:
            return request
        return request.override(system_message=new_system_message, messages=msgs)
