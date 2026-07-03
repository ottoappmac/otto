"""Middleware enabling ReAct-style tool calling for text-only MLX models.

MLX models (``ChatMLXText``) produce plain text output and do not implement
``model.bind_tools()``, which is required by the deepagents framework.

This middleware bridges that gap transparently:

1. **Strips native tool bindings** — passes ``tools=[]`` to the underlying
   ``create_agent`` machinery so ``model.bind_tools()`` is never called.
2. **Injects ReAct tool descriptions** — appends a structured tool-use section
   to the system message so the model knows how to invoke tools.
3. **Rewrites message history** — converts prior ``AIMessage(tool_calls)`` /
   ``ToolMessage`` pairs back into ``Thought / Action / Observation`` text so
   the MLX model sees a coherent scratchpad rather than structured objects.
4. **Parses text output** — detects ``Action: ```json{...}``` `` blocks in the
   model's response and synthesises ``AIMessage.tool_calls`` that the framework
   routes to its ``ToolNode``.
5. **Exits cleanly** — when no action block is found (the model writes
   ``Final Answer:``) the response is returned unchanged, allowing the agent
   loop to terminate normally.

Usage::

    from deepagents import create_deep_agent
    from chat_models.mlx import ChatMLXText
    from middleware.react_middleware import MLXReActMiddleware

    llm = ChatMLXText(model_path="mlx-community/Qwen3-8B-4bit", max_tokens=2048)
    agent = create_deep_agent(
        model=llm,
        tools=[search_tool, calculator_tool],
        middleware=[MLXReActMiddleware()],
    )

    result = await agent.ainvoke({"messages": [{"role": "user", "content": "..."}]})
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, ClassVar

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import BaseTool

from middleware._react_core import (
    TOOL_SECTION_TEMPLATE,
    content_to_text,
    make_tool_call_id,
    normalise_args,
    parse_action,
    prioritise_planning_tools,
    reformat_tool_history,
    render_tools,
    render_tools_compact,
    split_final_answer,
)


class MLXReActMiddleware(AgentMiddleware):
    """Makes text-only models (e.g. ``ChatMLXText``) compatible with deepagents.

    The deepagents ``create_deep_agent`` helper requires ``model.bind_tools()``,
    which raises ``NotImplementedError`` on plain text models such as MLX-backed
    ``ChatMLXText``.  This middleware intercepts the model call pipeline to
    provide a text-based ReAct shim, making those models behave as if they
    support native tool calling.

    **Placement in the middleware stack**

    Insert this middleware **first** (outermost) in the ``middleware`` list so
    it wraps all other middleware::

        agent = create_deep_agent(
            model=mlx_llm,
            tools=[...],
            middleware=[MLXReActMiddleware(), ...other_middleware...],
        )

    **Customising the tool-use prompt**

    Subclasses can override ``TOOL_SECTION_TEMPLATE`` to change the ReAct
    instructions injected into the system message::

        class MyMLXMiddleware(MLXReActMiddleware):
            TOOL_SECTION_TEMPLATE = "... custom instructions {tool_descriptions} ..."

    The template must contain exactly one ``{tool_descriptions}`` placeholder.

    **Compact tool rendering for small context windows**

    For on-device models with a small context window, set
    ``compact_tools=True`` to render each tool as a single
    ``- name: description`` line instead of the full argument schema.
    Use ``max_tools`` to cap the number of tools included::

        middleware=[MLXReActMiddleware(compact_tools=True, max_tools=20)]
    """

    TOOL_SECTION_TEMPLATE: ClassVar[str] = TOOL_SECTION_TEMPLATE

    def __init__(self, *, compact_tools: bool = False, max_tools: int | None = None) -> None:
        """Initialise the middleware.

        Args:
            compact_tools: When ``True``, render each tool as a single
                ``- name: brief_description`` line.  Strongly recommended for
                models with context windows below ~8 000 tokens.
            max_tools: Hard cap on the number of tools injected into the system
                prompt.  Tools beyond this limit are silently dropped.  ``None``
                means no cap (all tools are included, subject to ``compact_tools``
                rendering).
        """
        self._compact_tools = compact_tools
        self._max_tools = max_tools

    # ── AgentMiddleware hooks ───────────────────────────────────────────────

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        if not self._needs_react_shim(request.model):
            return handler(request)
        modified = self._prepare_request(request)
        response = handler(modified)
        return self._process_response(response, request.tools)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        if not self._needs_react_shim(request.model):
            return await handler(request)
        modified = self._prepare_request(request)
        response = await handler(modified)
        return self._process_response(response, request.tools)

    # ── Private helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _needs_react_shim(model: Any) -> bool:
        """Return ``True`` if *model* lacks native tool-calling support.

        Resolution order:

        1. If the model exposes ``supports_native_tools()`` (our MLX wrappers
           do), trust its self-report — this correctly handles ``ChatMLXText``
           instances loaded with a non-tool-aware chat template, where
           ``bind_tools`` is overridden but raises at call time.
        2. Otherwise fall back to the legacy check: did the model override
           the ``BaseChatModel.bind_tools`` stub?

        This allows the middleware to be included unconditionally in the
        middleware stack — it becomes a no-op for models that already
        support tool calling natively.
        """
        from langchain_core.language_models.chat_models import BaseChatModel

        check = getattr(model, "supports_native_tools", None)
        if callable(check):
            try:
                return not bool(check())
            except Exception:
                pass

        bind_tools = getattr(type(model), "bind_tools", None)
        if bind_tools is None:
            return True
        return bind_tools is BaseChatModel.bind_tools

    def _prepare_request(self, request: ModelRequest) -> ModelRequest:
        """Return a modified ``ModelRequest`` suitable for a text-only model.

        - Builds a ReAct tool-description section and appends it to the system
          message (creating one if none exists).
        - Rewrites any tool-call/tool-result message pairs in the history into
          ``Thought / Action / Observation`` format.
        - Sets ``tools=[]`` so the ``create_agent`` machinery skips
          ``model.bind_tools()`` entirely.

        Args:
            request: The original ``ModelRequest`` from the agent framework.

        Returns:
            A new ``ModelRequest`` with the modifications applied.
        """
        tools = [t for t in request.tools if isinstance(t, BaseTool)]
        # Float write_todos to the front so a ``max_tools`` cap never drops the
        # planning tool the ReAct prompt tells the model to call first.
        if self._max_tools is not None:
            tools = prioritise_planning_tools(tools)
        if self._compact_tools:
            tool_descriptions = render_tools_compact(tools, max_tools=self._max_tools)
        else:
            tool_descriptions = render_tools(
                tools if self._max_tools is None else tools[: self._max_tools]
            )
        tool_section = self.TOOL_SECTION_TEMPLATE.format(
            tool_descriptions=tool_descriptions
        )

        # Flatten the system message to plain text (MLX models don't handle
        # list-of-blocks content) and append the tool-use instructions.
        if request.system_message:
            existing_text = content_to_text(request.system_message.content)
            augmented_system = SystemMessage(content=existing_text + tool_section)
        else:
            augmented_system = SystemMessage(content=tool_section.lstrip())

        return request.override(
            system_message=augmented_system,
            messages=reformat_tool_history(list(request.messages)),
            tools=[],  # prevents model.bind_tools() call in the framework
        )

    def _process_response(
        self,
        response: ModelResponse,
        original_tools: list[BaseTool | dict[str, Any]],
    ) -> ModelResponse:
        """Parse the model's text response and synthesise ``tool_calls`` if needed.

        If the response contains a ReAct ``Action:`` block the returned
        ``AIMessage`` has a ``tool_calls`` list populated so the framework's
        ``ToolNode`` picks it up and executes the tool.

        If no action block is found (the model wrote ``Final Answer:`` or plain
        prose) the response is returned unchanged and the agent loop exits.

        Args:
            response: The ``ModelResponse`` returned by the text model.
            original_tools: The tools from the *original* request (before they
                were stripped in ``_prepare_request``), used to normalise args.

        Returns:
            Either the original ``ModelResponse`` or a modified one containing
            a synthesised ``AIMessage`` with ``tool_calls``.
        """
        if not response.result:
            return response

        last_msg = response.result[-1]
        if not isinstance(last_msg, AIMessage):
            return response

        raw_text = content_to_text(last_msg.content)
        parsed = parse_action(raw_text)

        if parsed is None:
            # Final-answer turn: clean the text (strip <think>, stop tokens,
            # ReAct scaffolding) and stash the Thought separately so the UI
            # can render them as collapsible reasoning + plain answer.
            answer, thought = split_final_answer(raw_text)
            cleaned_msg = AIMessage(
                content=answer,
                additional_kwargs={
                    **(last_msg.additional_kwargs or {}),
                    **({"thought": thought} if thought else {}),
                },
                response_metadata=last_msg.response_metadata or {},
            )
            return ModelResponse(
                result=[*response.result[:-1], cleaned_msg],
                structured_response=response.structured_response,
            )

        tool_name, action_input = parsed
        tools_by_name = {
            t.name: t for t in original_tools if isinstance(t, BaseTool)
        }
        args = normalise_args(action_input, tools_by_name.get(tool_name))

        synthesised_msg = AIMessage(
            content=raw_text,
            tool_calls=[{
                "name": tool_name,
                "args": args,
                "id": make_tool_call_id(),
                "type": "tool_call",
            }],
            # Preserve the underlying model's response metadata (MLX
            # throughput stats: prompt/generation TPS, KV cache hits, peak
            # memory) so tool-calling turns still contribute to the
            # session-level token-stats panel.
            response_metadata=last_msg.response_metadata or {},
        )

        return ModelResponse(
            result=[*response.result[:-1], synthesised_msg],
            structured_response=response.structured_response,
        )
