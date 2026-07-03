"""Computer Voyager agent — desktop automation via OS-native tools.

Uses LangChain ``create_agent`` with an ``MLXReActWrapper``-compatible LLM
and a pluggable ``ComputerNavigator`` (macOS today, Windows/Linux in future).

Modeled after ``WebVoyagerGraph`` but delegates tool selection to the
LangGraph ReAct loop instead of manually parsing ``Action:`` lines.

Usage::

    from langchain_anthropic import ChatAnthropic
    from agents.computer_voyager import ComputerVoyagerGraph
    from tools.navigation.computer import MacOSNavigator, MacOSToolkit
    from utilities.environment import Environment

    toolkit = MacOSToolkit(
        ax_ipc_timeout=Environment.get_ax_ipc_timeout(),
        scan_max_depth=Environment.get_scan_depth(),
        scan_max_elements=Environment.get_scan_max_elements(),
        scan_max_workers=Environment.get_scan_max_workers(),
    )
    llm = ChatAnthropic(model="claude-sonnet-4-5", max_tokens=4096)
    agent = ComputerVoyagerGraph(llm=llm, navigator=MacOSNavigator(toolkit=toolkit))
    result = await agent.arun("Open Calculator, compute 12+34, read the result.")
    print(result)

    # Streaming
    async for event in agent.stream("Open TextEdit and type hello"):
        print(event)
"""

from __future__ import annotations

import logging
import re
from collections import deque

logger = logging.getLogger(__name__)
from typing import Any, AsyncGenerator, List, Literal, Optional, Union

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from typing_extensions import TypedDict

from callbacks.agent_callback import AgentCallback, CallbackMixin
from middleware._react_core import content_to_text, extract_thought
from tools.navigation.computer.base import ComputerNavigator


# ── Result types ──────────────────────────────────────────────────────────────

class RunResult(TypedDict):
    answer: str
    steps: int
    thoughts: List[str]
    observations: List[str]


class AgentEvent(TypedDict):
    type: Literal["agent"]
    thought: str


class ToolsEvent(TypedDict):
    type: Literal["tools"]
    tool_name: str
    observation: str
    step: int


StreamEvent = Union[AgentEvent, ToolsEvent]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract_final_answer(content: Any) -> str:
    if isinstance(content, list):
        text = "\n".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    else:
        text = str(content)
    m = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()


def _get_thought(msg: AIMessage) -> str:
    """Extract the thought from an AIMessage, trying multiple sources."""
    thought = getattr(msg, "additional_kwargs", {}).get("thought")
    if thought:
        return thought
    text = content_to_text(msg.content) if msg.content else ""
    return extract_thought(text) or ""


# ── Vision detection ──────────────────────────────────────────────────────────

def _openai_base_url(model: Any) -> str:
    """Return the configured base URL of an OpenAI-compatible client, or ""."""
    for attr in ("openai_api_base", "base_url"):
        val = getattr(model, attr, None)
        if val:
            return str(val)
    client = getattr(model, "root_client", None) or getattr(model, "client", None)
    base = getattr(client, "base_url", None)
    return str(base) if base else ""


def _is_local_url(url: str) -> bool:
    u = url.lower()
    return any(h in u for h in ("localhost", "127.0.0.1", "0.0.0.0", "::1"))


def _is_vision_model(llm: BaseChatModel) -> bool:
    """Return ``True`` if *llm* can interpret image inputs.

    Detection order:
    1. Unwrap ``MLXReActWrapper`` — peek at the inner model it wraps.
    2. Explicit ``_text_only`` flag on the (unwrapped) model.
    3. Class-name heuristic — catches ``MLXVLChatModel`` and any future
       VLM wrappers without importing concrete classes.
    4. OpenAI-compatible clients pointing at a LOCAL server (oMLX / exo) may
       serve a text-only model, so inspect the served model id honestly via
       :func:`supports_vision` instead of blanket-trusting the class.
    5. Remaining API-backed models (cloud OpenAI, Anthropic, Bedrock, Cohere)
       handle images natively, so they default to ``True``.
    """
    inner = getattr(llm, "inner", llm)
    if getattr(inner, "_text_only", False):
        return False
    cls_name = type(inner).__name__.lower()
    if "vlm" in cls_name or "vl" in cls_name or "vision" in cls_name:
        return True
    if "mlx" in cls_name or "react" in cls_name:
        return False
    base_url = _openai_base_url(inner)
    if base_url and _is_local_url(base_url):
        from deep_agent.model_factory import supports_vision
        model_id = getattr(inner, "model_name", None) or getattr(inner, "model", None) or ""
        return supports_vision("omlx", str(model_id))
    return True


# ── Screen-control pruning middleware ─────────────────────────────────────────

_SCREEN_CONTROL_TOOLS = frozenset({
    "get_screen_controls",
    "launch_app",
    "capture_app_screenshot",
    "read_screen",
})

_SUPERSEDED_PLACEHOLDER = "[screen controls superseded by newer read]"

_DEFAULT_MAX_MESSAGES = 20


class ScreenControlPruningMiddleware(AgentMiddleware):
    """Prune the LLM context to keep it fast and focused.

    Two pruning strategies applied in order:

    1. **Screen-control dedup** — Tools like ``get_screen_controls`` and
       ``launch_app`` return large control lists that become stale after
       each action.  All but the latest such ``ToolMessage`` are replaced
       with a short placeholder.

    2. **Max-messages window** — Only the last *max_messages* messages
       (after screen-control pruning) are sent to the LLM.  This caps
       context growth in long sessions.

    The full message history is preserved in the graph state — only the
    view sent to the LLM is trimmed.

    Args:
        max_messages: Maximum number of messages to keep.  Sourced from
            ``Environment.get_computer_voyager_max_messages()`` via
            ``ComputerVoyagerGraph`` / ``DeepAgent``.
    """

    name: str = "screen_control_pruning"

    def __init__(self, max_messages: int = _DEFAULT_MAX_MESSAGES) -> None:
        self.max_messages = max_messages

    async def awrap_model_call(self, request, handler):
        pruned = _prune(request.messages, self.max_messages)
        return await handler(request.override(messages=pruned))

    def wrap_model_call(self, request, handler):
        pruned = _prune(request.messages, self.max_messages)
        return handler(request.override(messages=pruned))


def _prune_screen_controls(messages: list) -> list:
    """Replace all but the newest screen-control ToolMessage with a placeholder."""
    latest_idx: int | None = None
    copied = False
    for i in reversed(range(len(messages))):
        msg = messages[i]
        if (
            isinstance(msg, ToolMessage)
            and getattr(msg, "name", "") in _SCREEN_CONTROL_TOOLS
            and msg.content != _SUPERSEDED_PLACEHOLDER
        ):
            if latest_idx is None:
                latest_idx = i
            else:
                if not copied:
                    messages = list(messages)
                    copied = True
                messages[i] = ToolMessage(
                    content=_SUPERSEDED_PLACEHOLDER,
                    name=msg.name,
                    tool_call_id=msg.tool_call_id,
                )
    return messages


def _trim_to_window(messages: list, max_messages: int) -> list:
    """Keep only the last *max_messages* messages, preserving SystemMessages at the front."""
    if len(messages) <= max_messages:
        return messages
    prefix = [m for m in messages if isinstance(m, SystemMessage) or isinstance(m, HumanMessage)]
    remaining = [m for m in messages if not isinstance(m, SystemMessage) and not isinstance(m, HumanMessage)]
    budget = max(0, max_messages - len(prefix))
    return prefix + remaining[-budget:]


def _prune(messages: list, max_messages: int) -> list:
    """Apply screen-control dedup then trim to window size."""
    messages = _prune_screen_controls(messages)
    messages = _trim_to_window(messages, max_messages)
    return messages


# ── ComputerVoyagerGraph ──────────────────────────────────────────────────────

class ComputerVoyagerGraph(CallbackMixin):
    """Desktop automation agent backed by a ``ComputerNavigator``.

    The agent uses LangGraph's ``create_agent`` with the navigator's tools
    and system instructions.  Any ``BaseChatModel`` works — Anthropic, Bedrock,
    Cohere, or a local MLX model wrapped in ``MLXReActWrapper``.

    Vision-capable models (VLMs) automatically gain access to the
    ``capture_app_screenshot`` tool.  Detection is handled by
    :func:`_is_vision_model` which unwraps ``MLXReActWrapper`` to
    inspect the inner model.

    Args:
        llm:       The chat model (must support ``bind_tools`` or be wrapped).
        navigator: A ``ComputerNavigator`` instance (e.g. ``MacOSNavigator``).
        max_steps: Default maximum agent steps per run.
        max_messages: Context-window cap for the pruning middleware.
            ``None`` means use the default (``_DEFAULT_MAX_MESSAGES``).
        max_repeat: Force-stop after this many consecutive identical tool
            calls (same name + same args). ``0`` disables detection.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        navigator: ComputerNavigator,
        max_steps: int = 50,
        max_messages: int | None = None,
        max_repeat: int = 3,
        callback: Optional[AgentCallback] = None,
    ) -> None:
        self.llm = llm
        self.navigator = navigator
        self.max_steps = max_steps
        self.max_repeat = max_repeat
        self._callback = callback

        vision = _is_vision_model(llm)
        tools = navigator.get_tools(vision=vision)
        system_prompt = navigator.get_system_instructions(vision=vision)

        inner = getattr(llm, "inner", llm)
        logger.info(
            "ComputerVoyagerGraph: model=%s, inner=%s, vision=%s, tools=%s",
            type(llm).__name__,
            type(inner).__name__,
            vision,
            [t.name for t in tools],
        )

        effective_max = max_messages if max_messages is not None else _DEFAULT_MAX_MESSAGES
        self._graph = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            middleware=[ScreenControlPruningMiddleware(max_messages=effective_max)],
        )

    # ── Loop detection ─────────────────────────────────────────────────────

    def _is_loop(self, msg: AIMessage) -> bool:
        """Return True if the model is stuck in a tool-call loop.

        Two patterns trip this:

        * **Consecutive identical** — the same ``(name, args)`` repeated
          ``max_repeat`` times in a row (the classic stuck-click loop).
        * **Low-diversity alternation** — over the last ``2 * max_repeat``
          calls the model only ever used one or two distinct ``(name, args)``
          keys, e.g. alternating ``A, B, A, B, …``.  Consecutive-identical
          detection misses this, but it is just as much a non-progressing
          loop (the pattern behind the carpet ``doc_research`` failure).
        """
        if not self.max_repeat or not getattr(msg, "tool_calls", None):
            return False
        for tc in msg.tool_calls:
            key = (tc.get("name", ""), str(tc.get("args", {})))
            if key == self._last_tool_key:
                self._repeat_count += 1
            else:
                self._last_tool_key = key
                self._repeat_count = 1
            self._recent_keys.append(key)
            if self._repeat_count >= self.max_repeat:
                return True
            # Alternation / low-diversity: a full window made up of at most
            # two distinct keys means no real progress is being made.
            if (
                len(self._recent_keys) >= self._recent_keys.maxlen
                and len(set(self._recent_keys)) <= 2
            ):
                return True
        return False

    # ── Public interface ──────────────────────────────────────────────────────

    async def arun(
        self,
        task: str,
        max_steps: int | None = None,
    ) -> RunResult:
        """Run the agent to completion and return a ``RunResult``.

        Args:
            task:      Natural-language instruction (e.g. "Open Calculator...").
            max_steps: Override the default max steps for this run.
        """
        limit = max_steps or self.max_steps
        thoughts: List[str] = []
        steps = 0
        output = ""
        self._last_tool_key: tuple[str, str] | None = None
        self._repeat_count: int = 0
        self._recent_keys: deque = deque(maxlen=max(2, self.max_repeat * 2))

        await self._emit_info(f"Starting desktop automation: {task}", type="status")

        async for chunk in self._graph.astream(
            {"messages": [HumanMessage(content=task)]},
            config={"recursion_limit": limit * 2},
        ):
            model_chunk = chunk.get("model") or chunk.get("agent")
            if model_chunk:
                for msg in model_chunk["messages"]:
                    if isinstance(msg, AIMessage):
                        thought = _get_thought(msg)
                        if thought:
                            thoughts.append(thought)
                            await self._emit_info(thought[:300], type="thought")
                        if msg.content:
                            output = _extract_final_answer(msg.content)
                        if self._is_loop(msg):
                            await self._emit_warning(
                                f"Loop detected: same tool call repeated "
                                f"{self.max_repeat} times — stopping.",
                                type="status",
                            )
                            return RunResult(
                                answer=output or "Stopped: repeated action loop detected.",
                                steps=steps,
                                thoughts=thoughts,
                                observations=[],
                            )

            if "tools" in chunk:
                for msg in chunk["tools"]["messages"]:
                    steps += 1
                    tool_name = getattr(msg, "name", "tool")
                    content = str(getattr(msg, "content", ""))
                    await self._emit_info(
                        f"{tool_name} → {content[:300]}", type="tool"
                    )
                    if steps >= limit:
                        await self._emit_warning(
                            f"Max steps ({limit}) reached — stopping.",
                            type="status",
                        )
                        return RunResult(
                            answer=output,
                            steps=steps,
                            thoughts=thoughts,
                            observations=[],
                        )

        await self._emit_info(
            f"Desktop automation complete — {steps} steps, answer: {output[:200]}",
            type="status",
        )
        return RunResult(
            answer=output,
            steps=steps,
            thoughts=thoughts,
            observations=[],
        )

    async def stream(
        self,
        task: str,
        max_steps: int | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Async generator that yields one ``StreamEvent`` per graph node firing.

        Yields ``AgentEvent`` after each LLM turn and ``ToolsEvent`` after
        each tool execution.

        Usage::

            async for event in agent.stream("Open TextEdit and type hello"):
                if event["type"] == "agent":
                    print(event["thought"])
                elif event["type"] == "tools":
                    print(f"[{event['step']}] {event['tool_name']} → {event['observation']}")
        """
        limit = max_steps or self.max_steps
        steps = 0
        self._last_tool_key = None
        self._repeat_count = 0
        self._recent_keys = deque(maxlen=max(2, self.max_repeat * 2))

        await self._emit_info(f"Starting desktop automation: {task}", type="status")

        async for chunk in self._graph.astream(
            {"messages": [HumanMessage(content=task)]},
            config={"recursion_limit": limit * 2},
        ):
            model_chunk = chunk.get("model") or chunk.get("agent")
            if model_chunk:
                for msg in model_chunk["messages"]:
                    if isinstance(msg, AIMessage):
                        thought = _get_thought(msg)
                        if thought:
                            await self._emit_info(thought[:300], type="thought")
                        if self._is_loop(msg):
                            await self._emit_warning(
                                f"Loop detected: same tool call repeated "
                                f"{self.max_repeat} times — stopping.",
                                type="status",
                            )
                            return
                        yield AgentEvent(
                            type="agent",
                            thought=thought,
                        )

            if "tools" in chunk:
                for msg in chunk["tools"]["messages"]:
                    steps += 1
                    tool_name = getattr(msg, "name", "tool")
                    content = str(getattr(msg, "content", ""))
                    await self._emit_info(
                        f"{tool_name} → {content[:200]}", type="tool"
                    )
                    yield ToolsEvent(
                        type="tools",
                        tool_name=tool_name,
                        observation=content,
                        step=steps,
                    )
                    if steps >= limit:
                        await self._emit_warning(
                            f"Max steps ({limit}) reached — stopping.",
                            type="status",
                        )
                        return
