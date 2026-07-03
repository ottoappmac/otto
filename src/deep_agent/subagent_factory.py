"""SubAgent wiring based on :class:`SubAgentOption` selection."""

from __future__ import annotations

import logging
import re
from typing import Any, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from deep_agent.options import SubAgentOption

logger = logging.getLogger(__name__)

# Type alias matching deepagents.CompiledSubAgent
CompiledSubAgent = dict[str, Any]


# ── Shared helpers ────────────────────────────────────────────────────────────

def _last_human_content(state: dict) -> str:
    messages = state.get("messages", [])
    return next(
        (m.content for m in reversed(messages) if getattr(m, "type", "") == "human"),
        messages[-1].content if messages else "",
    )


# ── Web voyager ──────────────────────────────────────────────────────────────

def _build_web_voyager(
    vlm: BaseChatModel | None,
    headless: bool = False,
    callback: Any = None,
) -> Optional[CompiledSubAgent]:
    if vlm is None:
        logger.warning("web-voyager skipped: no VLM available")
        return None
    try:
        from agents.web_voyager import WebVoyagerGraph
    except Exception as exc:
        logger.warning("web-voyager unavailable: %s", exc)
        return None

    agent = WebVoyagerGraph(llm=vlm, headless=headless, callback=callback)

    async def _run(state: dict) -> dict:
        from utilities.environment import Environment
        messages = state.get("messages", [])
        question = _last_human_content(state)
        url_match = re.search(r"https?://\S+", question)
        start_url = (
            url_match.group(0).rstrip(".,)")
            if url_match
            else "https://www.google.com"
        )
        run_result = await agent.arun(
            question=question,
            start_url=start_url,
            max_steps=Environment.get_recursion_limit(),
        )

        parts: list[str] = []
        for i, (thought, step_actions) in enumerate(
            zip(
                run_result.get("thoughts") or [],
                run_result.get("actions") or [],
            ),
            1,
        ):
            observations = run_result.get("observations") or []
            obs = observations[i - 1] if i <= len(observations) else ""
            step_line = f"Step {i}: {thought}"
            if step_actions:
                step_line += " | Actions: " + ", ".join(
                    a.get("action", "") for a in step_actions
                )
            if obs:
                step_line += f"\n  → {obs}"
            parts.append(step_line)
        parts.append(run_result.get("answer") or "(no answer returned)")
        if run_result.get("logs"):
            parts.append(
                "**Notes:**\n" + "\n".join(f"- {n}" for n in run_result["logs"])
            )
        response = "\n\n".join(parts)
        return {**state, "messages": [*messages, AIMessage(content=response)]}

    return {
        "name": "web-voyager",
        "description": (
            "A browser automation agent using Playwright. "
            "Use when you need to navigate a live website, interact with page elements, "
            "fill forms, or extract content that requires JavaScript rendering. "
            "Include the target URL in your task description."
        ),
        "runnable": RunnableLambda(_run),
    }


# ── Computer voyager ─────────────────────────────────────────────────────────

def _build_computer_voyager(
    llm: BaseChatModel,
    vlm: BaseChatModel | None,
    cv_model_type: str | None,
    provider: str,
    callback: Any = None,
    max_messages: int = 20,
) -> Optional[CompiledSubAgent]:
    try:
        from agents.computer_voyager import ComputerVoyagerGraph
        from tools.navigation.computer import MacOSNavigator, MacOSToolkit
        from utilities.environment import Environment
    except Exception as exc:
        logger.warning("computer-voyager unavailable: %s", exc)
        return None

    # Resolve the model the computer-voyager will use
    if provider == "mlx" and cv_model_type == "vlm":
        if vlm is None:
            logger.warning("computer-voyager: cv_model_type=vlm but no VLM loaded; falling back to LLM")
            cv_model = llm
        else:
            from middleware.react_wrapper import MLXReActWrapper
            cv_model = MLXReActWrapper(vlm)
    else:
        cv_model = llm

    cv_inner = getattr(cv_model, "inner", cv_model)
    logger.info(
        "computer-voyager model: %s (inner: %s, cv_model_type: %s)",
        type(cv_model).__name__,
        type(cv_inner).__name__,
        cv_model_type or "llm",
    )

    toolkit = MacOSToolkit(
        ax_ipc_timeout=Environment.get_ax_ipc_timeout(),
        scan_max_depth=Environment.get_scan_depth(),
        scan_max_elements=Environment.get_scan_max_elements(),
        scan_max_workers=Environment.get_scan_max_workers(),
    )
    navigator = MacOSNavigator(toolkit=toolkit)
    agent = ComputerVoyagerGraph(
        llm=cv_model,
        navigator=navigator,
        max_messages=max_messages,
        max_repeat=Environment.get_computer_voyager_max_repeat(),
        callback=callback,
    )

    async def _run(state: dict) -> dict:
        from utilities.environment import Environment
        messages = state.get("messages", [])
        task = _last_human_content(state)
        run_result = await agent.arun(task=task, max_steps=Environment.get_recursion_limit())
        parts: list[str] = []
        for i, thought in enumerate(run_result.get("thoughts") or [], 1):
            parts.append(f"Step {i}: {thought}")
        parts.append(run_result.get("answer") or "(no answer returned)")
        response = "\n\n".join(parts)
        return {**state, "messages": [*messages, AIMessage(content=response)]}

    return {
        "name": "computer-voyager",
        "description": (
            "A macOS desktop automation agent using the Accessibility API and pyautogui. "
            "Use when you need to interact with native macOS applications — open apps, "
            "click buttons, type into fields, read UI values, or perform keyboard "
            "shortcuts. Describe the task in natural language."
        ),
        "runnable": RunnableLambda(_run),
    }


# ── Public factory ────────────────────────────────────────────────────────────

def create_subagents(
    options: list[SubAgentOption],
    *,
    llm: BaseChatModel,
    vlm: BaseChatModel | None = None,
    provider: str,
    browser_headless: bool = False,
    cv_model_type: str | None = None,
    computer_voyager_max_messages: int = 20,
    callback: Any = None,
) -> List[CompiledSubAgent]:
    """Build and return only the requested subagents."""
    result: List[CompiledSubAgent] = []
    builders = {
        SubAgentOption.WEB_VOYAGER: lambda: _build_web_voyager(
            vlm, headless=browser_headless, callback=callback,
        ),
        SubAgentOption.COMPUTER_VOYAGER: lambda: _build_computer_voyager(
            llm, vlm, cv_model_type, provider,
            callback=callback, max_messages=computer_voyager_max_messages,
        ),
    }
    for opt in options:
        builder = builders.get(opt)
        if builder is None:
            logger.warning("Unknown subagent option: %s", opt)
            continue
        sub = builder()
        if sub is not None:
            result.append(sub)
    return result
