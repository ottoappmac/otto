"""LangGraph-compatible ReAct wrapper for text-only MLX models.

Adds ``bind_tools()`` support to ``MLXVLChatModel`` (or any ``BaseChatModel``
that lacks native tool calling) so it can be used directly with LangGraph's
``create_react_agent`` without any separate middleware infrastructure.

How it works
------------
1. ``bind_tools(tools)`` returns a copy of the wrapper with the tool list stored.
2. ``_generate`` / ``_agenerate`` injects a ReAct tool-description section into
   the system message and rewrites any ``AIMessage(tool_calls)`` / ``ToolMessage``
   history to ``Thought / Action / Observation`` text before calling the inner
   model.
3. The inner model's text output is parsed for an ``Action:`` block.  If one is
   found, a synthetic ``AIMessage`` with ``tool_calls`` is returned so LangGraph's
   ``ToolNode`` can execute it.  If not, the response is returned as-is and the
   agent loop exits normally.

force_action mode
-----------------
Reasoning models such as DeepSeek-R1 tend to simulate the entire task inside
their ``<think>`` block and then emit a fabricated "Final Answer" without ever
calling a single tool.  Setting ``force_action=True`` appends a hard reminder
to the *last* human message on turns where no tool observations have been
received yet, forcing the model to emit an ``Action:`` block before doing
anything else.

Usage::

    from chat_models.mlx.chat_vlm import MLXVLChatModel
    from middleware.react_wrapper import MLXReActWrapper
    from langgraph.prebuilt import create_react_agent

    # Standard instruction-following model (default):
    inner = MLXVLChatModel(model_path="mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
    llm = MLXReActWrapper(inner)

    # Reasoning model (DeepSeek-R1, Qwen3-thinking, etc.) — needs force_action:
    from chat_models.mlx.chat_mlx_text import ChatMLXText
    inner = ChatMLXText(model_path="mlx-community/DeepSeek-R1-Distill-Qwen-14B-8bit")
    llm = MLXReActWrapper(inner, force_action=True)

    agent = create_react_agent(model=llm, tools=MACOS_TOOLS, prompt=system_prompt)
"""

from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict

from middleware._react_core import (
    TOOL_SECTION_TEMPLATE,
    content_to_text,
    extract_thought,
    make_tool_call_id,
    normalise_args,
    parse_action,
    reformat_tool_history,
    render_tools,
    split_final_answer,
)


_FORCE_ACTION_REMINDER = (
    "\n\n---\n"
    "REMINDER: You have received NO tool observations yet. "
    "You MUST call a tool RIGHT NOW.\n"
    "Output ONLY a Thought: line followed by an Action: JSON block, then STOP.\n"
    "For a multi-step task make that first Action a `write_todos` call listing the "
    "steps; otherwise call the tool that takes the first real step.\n"
    "Do NOT write a prose plan and do NOT write a Final Answer — you have not done "
    "anything yet."
)

_STOP_OBSERVING_NUDGE = (
    "\n\n---\n"
    "STOP calling get_screen_controls / screenshot / list_apps — you already "
    "have the controls. LOOK at the Observations above: they contain the "
    "control indices you need.\n"
    "Your NEXT action MUST be one of: press_control, type_into_control, "
    "get_control_value, hotkey, click, type_text, open_app, activate_app.\n"
    "Output a Thought: line then an Action: JSON block that ACTS on the UI."
)

_OBSERVATION_TOOLS = frozenset({
    "get_screen_controls", "list_apps",
})


class MLXReActWrapper(BaseChatModel):
    """Wraps any text-only ``BaseChatModel`` to add LangGraph ``bind_tools()`` support.

    Passes the tool-use shim logic from ``_react_core`` directly into the
    model's generate step, making it fully compatible with LangGraph's
    ``create_react_agent`` without any middleware framework.

    Args:
        inner:        The underlying text-generation model.
        force_action: When ``True``, append a hard reminder to the last human
                      message on turns where no tool observations exist yet.
                      Required for reasoning models (DeepSeek-R1, Qwen3-thinking)
                      that otherwise simulate the whole task and emit a fabricated
                      Final Answer without calling any tools.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    inner: Any
    force_action: bool = False
    _bound_tools: list[BaseTool] = []
    # Populated by ``bind_tools`` when the inner model declares native tool-call
    # support — we then bypass the ReAct text shim and forward the OpenAI-format
    # tool list straight to the inner model's ``_generate``.
    _native_openai_tools: list[dict] | None = None

    def __init__(self, inner: BaseChatModel, force_action: bool = False, **kwargs):
        super().__init__(inner=inner, force_action=force_action, **kwargs)

    # ── LangGraph compatibility ───────────────────────────────────────────────

    def _inner_supports_native_tools(self) -> bool:
        """Return True when the wrapped model exposes ``supports_native_tools()``
        and reports True.

        Used to decide whether ``bind_tools`` and ``_generate`` should bypass
        the ReAct text shim entirely and let the inner model emit structured
        ``tool_calls`` natively.
        """
        check = getattr(self.inner, "supports_native_tools", None)
        if not callable(check):
            return False
        try:
            return bool(check())
        except Exception:
            return False

    def bind_tools(self, tools: list, **kwargs) -> "MLXReActWrapper":
        """Return a copy of this wrapper with *tools* registered.

        Called by ``create_react_agent`` before the agent graph is built.
        The original instance is left unmodified so the same ``MLXReActWrapper``
        can be reused with different tool sets.

        When the wrapped model supports native tool calling (Qwen, Llama 3.1+,
        Mistral-Nemo, etc.) the OpenAI-format tool list is also stored so the
        ReAct text shim is bypassed at generate time.
        """
        new = MLXReActWrapper(inner=self.inner, force_action=self.force_action)
        new._bound_tools = [t for t in tools if isinstance(t, BaseTool)]
        if self._inner_supports_native_tools() and new._bound_tools:
            new._native_openai_tools = [
                convert_to_openai_tool(t) for t in new._bound_tools
            ]
        return new

    # ── Prompt + history preparation ─────────────────────────────────────────

    def _prepare_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Inject tool descriptions and rewrite history to ReAct text format."""
        tools = self._bound_tools
        if not tools:
            return list(messages)

        tool_section = TOOL_SECTION_TEMPLATE.format(
            tool_descriptions=render_tools(tools)
        )

        # Append tool section to the existing system message (or create one).
        prepared: List[BaseMessage] = []
        system_injected = False
        for msg in messages:
            if isinstance(msg, SystemMessage) and not system_injected:
                existing = content_to_text(msg.content)
                prepared.append(SystemMessage(content=existing + tool_section))
                system_injected = True
            else:
                prepared.append(msg)
        if not system_injected:
            prepared.insert(0, SystemMessage(content=tool_section.lstrip()))

        # Rewrite AIMessage(tool_calls) / ToolMessage pairs to Thought/Action/Observation.
        prepared = reformat_tool_history(prepared)

        if self.force_action:
            has_observations = any(
                isinstance(m, ToolMessage)
                or (isinstance(m, HumanMessage) and content_to_text(m.content).startswith("Observation:"))
                for m in prepared
            )
            if not has_observations:
                # First turn — no observations yet. Force an Action: block.
                for i in range(len(prepared) - 1, -1, -1):
                    if isinstance(prepared[i], HumanMessage):
                        original = content_to_text(prepared[i].content)
                        prepared[i] = HumanMessage(content=original + _FORCE_ACTION_REMINDER)
                        break
                else:
                    prepared.append(HumanMessage(content=_FORCE_ACTION_REMINDER.lstrip()))

        # Detect repeated observation-only tool calls (the model keeps reading
        # controls but never acts). Count how many of the last N AI turns called
        # only observation tools.
        recent_observation_only = 0
        for msg in reversed(prepared):
            text = content_to_text(msg.content) if isinstance(msg, (AIMessage, HumanMessage)) else ""
            if isinstance(msg, AIMessage) and "Action:" in text:
                import re as _re
                action_m = _re.search(r'"action"\s*:\s*"([^"]+)"', text)
                if action_m and action_m.group(1) in _OBSERVATION_TOOLS:
                    recent_observation_only += 1
                else:
                    break
            elif isinstance(msg, HumanMessage) and text.startswith("Observation:"):
                continue
            elif isinstance(msg, SystemMessage):
                continue
            else:
                break

        if recent_observation_only >= 2:
            prepared.append(HumanMessage(content=_STOP_OBSERVING_NUDGE.lstrip()))

        return prepared

    # ── Response parsing ─────────────────────────────────────────────────────

    def _build_result(self, text: str, response_metadata: dict | None = None) -> ChatResult:
        """Return a ChatResult, synthesising tool_calls if an Action block is found.

        ``response_metadata`` is forwarded from the inner model's ``AIMessage``
        so that MLX perf stats (TPS, cache hit ratio, peak memory, etc.) survive
        the wrapper and appear in LangSmith's Metadata panel.

        On the final (no-action) turn the raw text is split into a clean
        ``Final Answer`` body and a separate ``Thought`` (stashed in
        ``additional_kwargs``) so the frontend can render them independently
        and the conversation history stops carrying ReAct scaffolding forward.
        """
        meta = response_metadata or {}
        parsed = parse_action(text)
        if parsed is None:
            answer, thought = split_final_answer(text)
            additional: dict[str, Any] = {"thought": thought} if thought else {}
            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content=answer,
                response_metadata=meta,
                additional_kwargs=additional,
            ))])

        tool_name, action_input = parsed
        tools_by_name = {t.name: t for t in self._bound_tools}
        args = normalise_args(action_input, tools_by_name.get(tool_name))

        thought = extract_thought(text)
        msg = AIMessage(
            content=text,
            response_metadata=meta,
            tool_calls=[{
                "name": tool_name,
                "args": args,
                "id": make_tool_call_id(),
                "type": "tool_call",
            }],
            additional_kwargs={"thought": thought} if thought else {},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    # ── BaseChatModel interface ───────────────────────────────────────────────

    def _native_generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> ChatResult:
        """Delegate straight to the inner model with native tool calling.

        Skips the ReAct text shim entirely: no system-prompt injection, no
        ``Thought / Action / Observation`` history rewriting, no Action-block
        parsing.  The inner model's ``_generate`` already returns an
        ``AIMessage`` with structured ``tool_calls`` when the model emits
        family-specific tool-call markers.
        """
        return self.inner._generate(
            messages, stop=stop, tools=self._native_openai_tools, **kwargs
        )

    async def _native_agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> ChatResult:
        return await self.inner._agenerate(
            messages, stop=stop, tools=self._native_openai_tools, **kwargs
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        if self._native_openai_tools is not None:
            return self._native_generate(messages, stop=stop, **kwargs)
        prepared = self._prepare_messages(messages)
        result = self.inner._generate(prepared, stop=stop, **kwargs)
        inner_msg = result.generations[-1].message
        text = content_to_text(inner_msg.content)
        return self._build_result(text, response_metadata=inner_msg.response_metadata or {})

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        if self._native_openai_tools is not None:
            return await self._native_agenerate(messages, stop=stop, **kwargs)
        prepared = self._prepare_messages(messages)
        result = await self.inner._agenerate(prepared, stop=stop, **kwargs)
        inner_msg = result.generations[-1].message
        text = content_to_text(inner_msg.content)
        return self._build_result(text, response_metadata=inner_msg.response_metadata or {})

    @property
    def _llm_type(self) -> str:
        return "mlx-react-wrapper"
