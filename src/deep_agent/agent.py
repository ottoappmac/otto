"""DeepAgent — high-level orchestrator with enum-driven configuration.

Usage::

    from deep_agent import DeepAgent, ToolOption, SubAgentOption

    agent = DeepAgent(
        tools=[ToolOption.DOC_READER, ToolOption.WEB_RESEARCHER],
        subagents=[SubAgentOption.WEB_VOYAGER],
    )

    # Streaming
    async for chunk in agent.astream("Compare Paris and Tokyo"):
        for msg in chunk:
            print(msg)

    # Single invocation
    result = await agent.arun("What is the speed of light?")
"""

from __future__ import annotations

import asyncio

from typing import Any, AsyncGenerator, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from deepagents import create_deep_agent
from callbacks.agent_callback import AgentCallback, CallbackMixin
from deep_agent.model_factory import create_deep_agent_llm, create_llm, create_mlx_vlm
from deep_agent.options import SubAgentOption, ToolOption
from deep_agent.prompt import build_orchestrator_prompt
from deep_agent.subagent_factory import create_subagents
from deep_agent.tool_factory import ToolSet, create_tools
from utilities.environment import Environment
from utilities.logger import get_logger, init_logger

logger = get_logger()


def _to_backend_path(absolute_path: str, backend: Any) -> str:
    """Convert *absolute_path* to a virtual path when *backend* uses
    ``virtual_mode``.  Returns the original path otherwise.

    ``FilesystemBackend(virtual_mode=True)`` treats every path as
    relative to its ``cwd``, so absolute filesystem paths must be
    expressed as ``"/" + relpath`` to be resolved correctly by
    ``_resolve_path``.
    """
    from pathlib import Path

    if backend is None or not getattr(backend, "virtual_mode", False):
        return absolute_path
    try:
        rel = Path(absolute_path).resolve().relative_to(backend.cwd)
        return "/" + rel.as_posix()
    except ValueError:
        return absolute_path


class DeepAgent(CallbackMixin):
    """Orchestrator that coordinates tools and subagents.

    Args:
        tools:      Which direct tools to enable (default: all).
        subagents:  Which subagents to enable (default: all).
        memory:     Paths to memory / AGENTS.md files.
        skills_dir: Base directory containing skill sub-folders (each with a
                    ``SKILL.md``).  All skills found in the directory are loaded.
        backend:    A ``BackendProtocol`` instance (e.g. ``FilesystemBackend``).
        callback:   Optional :class:`AgentCallback` for lifecycle events.
        browser_headless: Run browser-based tools headless (default ``False``).
    """

    _ALL_TOOLS = list(ToolOption)
    _ALL_SUBAGENTS = list(SubAgentOption)

    # Subagents that always need a VLM when provider=mlx
    _MLX_VLM_SUBAGENTS: frozenset[SubAgentOption] = frozenset({
        SubAgentOption.WEB_VOYAGER,
    })

    def __init__(
        self,
        tools: Optional[List[ToolOption]] = None,
        subagents: Optional[List[SubAgentOption]] = None,
        memory: Optional[List[str]] = None,
        skills_dir: Optional[str] = None,
        backend: Any = None,
        callback: Optional[AgentCallback] = None,
        browser_headless: bool = False,
    ) -> None:
        self._callback = callback

        Environment.load()
        init_logger()
        self._provider = Environment.get_llm_provider()
        await_info = self._sync_emit_info

        await_info(f"LLM_PROVIDER: {self._provider}", type="status")

        # ── Models ────────────────────────────────────────────────────────
        self._llm = create_llm(self._provider)

        # Optional dedicated orchestrator model (DEEP_AGENT_LLM_PROVIDER)
        self._orchestrator_llm = create_deep_agent_llm(self._provider) or self._llm

        sa_opts = subagents if subagents is not None else []
        _needs_mlx_vlm = self._provider == "mlx" and self._mlx_vlm_required(sa_opts)
        _vlm_model_id = Environment.get_hf_vlm_model_id()
        if _needs_mlx_vlm and not _vlm_model_id:
            logger.warning(
                "Subagents %s require a VLM but HF_VLM_MODEL_ID is not set — "
                "falling back to the text LLM. Set HF_VLM_MODEL_ID in .env "
                "to enable vision capabilities.",
                [s.value for s in sa_opts if s in self._MLX_VLM_SUBAGENTS],
            )
            _needs_mlx_vlm = False
        self._vlm = create_mlx_vlm(self._provider, self._llm) if _needs_mlx_vlm else self._llm
        await_info(f"LLM: {type(self._llm).__name__}", type="status")
        if self._orchestrator_llm is not self._llm:
            _da_provider = Environment.get_deep_agent_llm_provider() or self._provider
            _inner = getattr(self._orchestrator_llm, "inner", self._orchestrator_llm)
            _model_label = getattr(_inner, "model_path", None) or getattr(_inner, "model", None) or type(_inner).__name__
            await_info(f"Orchestrator LLM: {_da_provider} / {_model_label}", type="status")
        if self._vlm is not self._llm:
            await_info(f"VLM: {type(self._vlm).__name__}", type="status")

        # ── Tools ─────────────────────────────────────────────────────────
        # Tools use the LLM for plain text tasks (sub-query generation,
        # summarisation).  Unwrap MLXReActWrapper and create a cache-free
        # copy so the agent's ReAct prompt cache doesn't bleed into tools.
        import copy
        _llm_for_tools = getattr(self._orchestrator_llm, "inner", self._orchestrator_llm)
        if getattr(_llm_for_tools, "_prompt_cache", None) is not None:
            _llm_for_tools = copy.copy(_llm_for_tools)
            _llm_for_tools._prompt_cache = None
            _llm_for_tools._last_prompt_tokens = None
            _llm_for_tools.enable_prompt_cache = False
            _llm_for_tools.enable_system_prompt_cache = False
        tool_opts = tools if tools is not None else self._ALL_TOOLS
        self._tool_set: ToolSet = create_tools(
            tool_opts, _llm_for_tools,
            browser_headless=browser_headless,
            provider=self._provider,
        )
        await_info(
            f"Direct tools: {[t.name for t in self._tool_set.deep_agent_tools]}",
            type="status",
        )

        # ── Subagents ─────────────────────────────────────────────────────
        cv_model_type = (
            Environment.get_computer_voyager_mlx_model_type()
            if self._provider == "mlx"
            else None
        )
        self._subagents = create_subagents(
            sa_opts,
            llm=self._llm,
            vlm=self._vlm if self._vlm is not self._llm else None,
            provider=self._provider,
            browser_headless=browser_headless,
            cv_model_type=cv_model_type,
            computer_voyager_max_messages=Environment.get_computer_voyager_max_messages(),
            callback=callback,
        )
        await_info(
            f"Subagents: {[s['name'] for s in self._subagents]}",
            type="status",
        )

        # ── Prompt ────────────────────────────────────────────────────────
        # Lite mode is selected by Environment based on LOCAL_PROMPT_MODE
        # (default ``auto``: lite when the orchestrator runs on an OSS-local
        # provider — mlx or exo).  See :mod:`deep_agent.prompt` for what
        # lite mode keeps vs. drops.
        _use_lite = Environment.use_lite_orchestrator_prompt()
        self._system_prompt = build_orchestrator_prompt(
            self._tool_set.deep_agent_tools, self._subagents,
            lite=_use_lite,
        )
        await_info(
            f"Orchestrator prompt: {'lite' if _use_lite else 'full'} "
            f"(~{len(self._system_prompt) // 4} tok)",
            type="status",
        )

        # ── Skills ────────────────────────────────────────────────────────
        resolved_skills: list[str] | None = None
        if skills_dir is not None:
            resolved_skills = [_to_backend_path(skills_dir, backend)]
            for sp in resolved_skills:
                await_info(f"Skills source: {sp}", type="status")

        # ── Middleware ────────────────────────────────────────────────────
        middleware: list = []
        if self._provider == "mlx":
            if ToolOption.PLAYWRIGHT_MCP in tool_opts:
                from middleware.playwright_pruning import PlaywrightSnapshotPruningMiddleware

                middleware.append(PlaywrightSnapshotPruningMiddleware(
                    max_messages=Environment.get_playwright_max_messages(),
                ))

            from middleware.react_middleware import MLXReActMiddleware

            middleware.append(MLXReActMiddleware())

        # OpenAI-compatible servers only render image content in *user*
        # messages — relocate tool-result images so vision models see them.
        # No-op for providers that render tool-result images natively.
        from middleware.tool_image_relocation import maybe_for_model

        image_relocation = maybe_for_model(self._llm)
        if image_relocation is not None:
            middleware.append(image_relocation)

        # ── Assemble ──────────────────────────────────────────────────────
        from langgraph.checkpoint.memory import MemorySaver

        resolved_memory = (
            [_to_backend_path(m, backend) for m in memory]
            if memory is not None
            else None
        )

        self._graph = create_deep_agent(
            model=self._orchestrator_llm,
            system_prompt=self._system_prompt,
            tools=self._tool_set.deep_agent_tools,
            subagents=self._subagents,
            middleware=middleware,
            memory=resolved_memory,
            skills=resolved_skills,
            backend=backend,
            checkpointer=MemorySaver(),
        )
        await_info("DeepAgent ready", type="status")

    # ── Model helpers ────────────────────────────────────────────────────

    def _mlx_vlm_required(self, sa_opts: List[SubAgentOption]) -> bool:
        """Return ``True`` if any requested subagent needs the VLM under MLX.

        Unconditional VLM subagents are declared in :attr:`_MLX_VLM_SUBAGENTS`.
        Subagents with opt-in VLM behaviour (driven by environment config) are
        checked inline here.
        """
        requested = frozenset(sa_opts)
        if requested & self._MLX_VLM_SUBAGENTS:
            return True
        if (
            SubAgentOption.COMPUTER_VOYAGER in requested
            and Environment.get_computer_voyager_mlx_model_type() == "vlm"
        ):
            return True
        return False

    # ── Sync emit helper (for use inside __init__) ────────────────────────

    def _sync_emit_info(self, message: str, type: str = "") -> None:
        """Synchronous log + callback for use during ``__init__``."""
        logger.info(f"{type}: {message}")
        if self._callback:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._callback.on_info(message, type=type))
            except RuntimeError:
                asyncio.run(self._callback.on_info(message, type=type))

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def llm(self) -> BaseChatModel:
        return self._llm

    @property
    def vlm(self) -> BaseChatModel:
        return self._vlm

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def graph(self):
        return self._graph

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def subagents(self) -> list:
        return list(self._subagents)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Release resources (MCP connections, etc.)."""
        if self._tool_set:
            await self._tool_set.close()

    # ── Public interface ──────────────────────────────────────────────────

    async def arun(
        self,
        query: str,
        thread_id: str = "default",
    ) -> str:
        """Run the agent to completion and return the final answer."""
        await self._emit_info(f"Running: {query[:120]}", type="status")
        result = await self._graph.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": thread_id}},
        )
        final = next(
            (
                m
                for m in reversed(result.get("messages", []))
                if isinstance(m, AIMessage)
            ),
            None,
        )
        answer = ""
        if final:
            content = final.content
            if isinstance(content, list):
                answer = " ".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            else:
                answer = str(content)
        await self._emit_info(
            f"Complete: {answer[:200]}", type="status",
        )
        return answer

    async def astream(
        self,
        query: str,
        thread_id: str = "default",
        callbacks: list | None = None,
        run_id: Optional[str] = None,
    ) -> AsyncGenerator[list, None]:
        """Yield lists of new messages as the agent streams.

        Each yielded list contains the messages added since the last yield.

        Args:
            callbacks: Optional list of LangChain callbacks (e.g. ``CacheStatsCallback``)
                       forwarded to every model call within the run.
            run_id: Optional LangSmith run ID for tracing.
        """
        await self._emit_info(f"Streaming: {query[:120]}", type="status")
        printed = 0
        run_config: dict = {"configurable": {"thread_id": thread_id}}
        if callbacks:
            run_config["callbacks"] = callbacks
        if run_id is not None:
            run_config["run_id"] = run_id
        async for chunk in self._graph.astream(
            {"messages": [HumanMessage(content=query)]},
            config=run_config,
            stream_mode="values",
        ):
            if "messages" in chunk:
                new_msgs = chunk["messages"][printed:]
                printed = len(chunk["messages"])
                if new_msgs:
                    yield new_msgs


# ── Utility ──────────────────────────────────────────────────────────────────

def print_message(msg: Any) -> None:
    """Log a LangChain message at the appropriate level."""
    if isinstance(msg, HumanMessage):
        logger.info(f"[You] {msg.content}")

    elif isinstance(msg, AIMessage):
        content = msg.content
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        if content and content.strip():
            logger.info(f"[Agent] {content.strip()}")
        if msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name", "")
                args = tc.get("args", {})
                if name == "task":
                    logger.debug(
                        f"  → delegating to {args.get('subagent_type')!r}: "
                        f"{str(args.get('description', ''))[:80]}"
                    )
                else:
                    logger.debug(f"  → tool {name!r}: {str(args)[:80]}")

    elif isinstance(msg, ToolMessage):
        preview = str(msg.content)[:120].replace("\n", " ")
        logger.debug(f"  ← [{getattr(msg, 'name', 'tool')}] {preview}")
