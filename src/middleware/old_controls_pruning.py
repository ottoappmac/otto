"""Middleware that shortens old desktop-tool results in the replayed history.

**Why this exists**

The macOS desktop agent's acting tools return the screen's controls with their result ("Controls now ...",
"Dialog now ..."), often several thousand characters each, and reads such as ``get_screen_controls`` and
``read_screen`` return more. Over a long run these pile up: one 100-step benchmark run died at step 98 when the
local model server (oMLX) refused the prompt for lack of memory ("Prefill context too large"), long before
summarization (a context-length trigger) would have run. Only the latest screen matters for the next step.

**What it does**

Before each model call, every tool result except the newest :data:`KEEP_RECENT` is shortened: the controls list
an action appended is replaced with a one-line note, and a result still longer than :data:`MAX_OLD_CHARS` keeps
its head and a note saying it was cut. The tool-call / tool-result pairing and every message's id are unchanged.

The transform runs on a throwaway copy of the request (``request.override``) and is recomputed from the saved
conversation on every call, so it is idempotent and never changes what is persisted.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import ToolMessage

KEEP_RECENT = 6
"""Newest tool results kept as they are."""

MAX_OLD_CHARS = 1500
"""Longest an older tool result's text may stay after its controls list is removed."""

# Where an action's result starts the controls list read after it (scripts/macarena/guest, the app's observe-after).
MARKERS = ("\n\nControls now (", "\n\nDialog now (")
NOTE_CONTROLS = "\n\n(the controls read after this action were removed from the history; the latest result has them)"
NOTE_CUT = "\n… (older tool result cut to save memory; call the tool again if you need it)"

# Agents whose runs are long and whose tool results carry screens of controls.
AGENTS = frozenset({"macos-desktop-agent", "macarena-desktop-agent"})


def maybe_for_agent(agent_name: str | None) -> "OldControlsPruningMiddleware | None":
    """The middleware for agents that need it (see :data:`AGENTS`), else ``None``."""
    if (agent_name or "") not in AGENTS:
        return None
    return OldControlsPruningMiddleware()


def shorten(text: str, max_chars: int = MAX_OLD_CHARS) -> str:
    """``text`` without an appended controls list, then cut to ``max_chars`` (with a note) if still longer."""
    cut = min((i for i in (text.find(m) for m in MARKERS) if i >= 0), default=-1)
    out = text[:cut] + NOTE_CONTROLS if cut >= 0 else text
    if len(out) > max_chars + len(NOTE_CONTROLS):
        out = out[:max_chars] + NOTE_CUT
    return out


class OldControlsPruningMiddleware(AgentMiddleware):
    """Shorten every tool result but the newest ``keep_recent``."""

    def __init__(self, keep_recent: int = KEEP_RECENT, max_chars: int = MAX_OLD_CHARS) -> None:
        super().__init__()
        self._keep = max(1, int(keep_recent))
        self._max = max(200, int(max_chars))

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._prune_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._prune_request(request))

    def _prune_request(self, request: ModelRequest) -> ModelRequest:
        pruned = self.prune(list(request.messages))
        return request if pruned is None else request.override(messages=pruned)

    def prune(self, messages: list[Any]) -> list[Any] | None:
        """A new message list with the older tool results shortened, or ``None`` if nothing changed."""
        tool_idx = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
        old = set(tool_idx[:-self._keep]) if len(tool_idx) > self._keep else set()
        if not old:
            return None
        changed = False
        out: list[Any] = []
        for i, m in enumerate(messages):
            if i in old:
                new = self._shorten_content(m.content)
                if new is not None:
                    m = m.model_copy(update={"content": new})
                    changed = True
            out.append(m)
        return out if changed else None

    def _shorten_content(self, content: Any) -> Any:
        if isinstance(content, str):
            short = shorten(content, self._max)
            return short if short != content else None
        if isinstance(content, list):
            blocks, changed = [], False
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str):
                    short = shorten(b["text"], self._max)
                    if short != b["text"]:
                        b, changed = {**b, "text": short}, True
                blocks.append(b)
            return blocks if changed else None
        return None
