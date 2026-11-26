"""Keep otto's main agent the same across the deepagents 0.5 -> 0.7 upgrade.

deepagents 0.7 stopped adding two things otto's prompts and guards rely on:

* the base agent prompt (``BASE_AGENT_PROMPT``) appended after otto's own
  system prompt, and
* ``TodoListMiddleware``, which provides ``write_todos`` and the ``todos``
  state key that run-depth, the ReAct shim and the MacArena guard read.

:func:`with_legacy_defaults` puts both back for a ``create_deep_agent`` call.
"""

from __future__ import annotations

from typing import Any


def _legacy_base_prompt() -> str:
    import deepagents.graph as graph

    # 0.7 keeps the 0.5 text under a private name; reading the public name
    # only emits a deprecation warning.
    return getattr(graph, "_LEGACY_BASE_AGENT_PROMPT", "") or ""


def with_legacy_defaults(system_prompt: str | None, middleware: list[Any]) -> tuple[str | None, list[Any]]:
    """Return ``(system_prompt, middleware)`` with the 0.5 defaults restored."""
    from langchain.agents.middleware import TodoListMiddleware

    base = _legacy_base_prompt()
    if base and system_prompt is not None and base not in system_prompt:
        system_prompt = f"{system_prompt}\n\n{base}"
    elif base and system_prompt is None:
        system_prompt = base

    if not any(isinstance(m, TodoListMiddleware) for m in middleware):
        middleware = [TodoListMiddleware(), *middleware]
    return system_prompt, middleware
