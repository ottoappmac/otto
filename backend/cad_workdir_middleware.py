"""Confine the CAD tools to the session's mapped folder.

The cad-tools MCP is one long-lived process shared by every session, so it cannot know which
folder the current session owns. This middleware does: before a cad-tools call runs it

* fills ``workdir`` with the session's mapped folder when the agent passes ``.`` or nothing, and
* refuses any ``workdir`` outside that folder.

The root is the session's mapped project folder, or its own files folder when none is mapped
(where the agent already writes by default).
"""

from __future__ import annotations

import os

import logging
from pathlib import Path
from typing import Any, Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

logger = logging.getLogger(__name__)

CAD_TOOLS = frozenset({
    "cad_run", "cad_render", "cad_compare", "cad_export", "step_inspect", "answer_checkpoint", "netlist_validate",
    "budget_status", "assembly_status", "cad_autoplace",
})


# Tools an agent may not use. BenchCAD runs have no business with a shell: it only found ways to
# waste time (decoding PNGs by hand, searching the disk) and could read the answer files.
NO_TOOLS_FOR_AGENT = {"benchcad-agent": frozenset({"execute"})}


def blocked_tools(agent: str) -> frozenset:
    """Tools the agent may not call. BENCHCAD_NO_SUBAGENTS=1 (an ablation run) also takes `task` from
    benchcad-agent, so it works with no subagent at all, the built-in general-purpose one included."""
    out = NO_TOOLS_FOR_AGENT.get(agent, frozenset())
    if agent == "benchcad-agent" and os.environ.get("BENCHCAD_NO_SUBAGENTS", "").strip() in ("1", "true", "yes"):
        out = out | {"task"}
    return out


_WRITES: dict[str, set] = {}    # hashes of files written so far, per session
_VIEWS: dict[str, int] = {}     # view_image calls so far, per session


def view_image_cap() -> int:
    try:
        return int(os.environ.get("BENCHCAD_MAX_VIEW_IMAGES", "25"))
    except ValueError:
        return 25


def _blocked(request: ToolCallRequest, text: str) -> ToolMessage:
    return ToolMessage(content=text, name=request.tool_call.get("name") or "tool",
                       tool_call_id=request.tool_call.get("id") or "", status="error")


def resolve_workdir(root: Path, asked: Any) -> Path | str:
    """The absolute folder to use, or an error message."""
    raw = str(asked or "").strip()
    p = Path(raw).expanduser() if raw and raw != "." else root
    if not p.is_absolute():
        p = root / p
    p = p.resolve()
    if p != root and root not in p.parents and not p.exists():
        # An invented absolute path such as /output: the agent's file tools treat it as
        # relative to the session folder, so do the same here instead of refusing.
        p = (root / str(p).lstrip("/")).resolve()
    if p != root and root not in p.parents:
        return f"workdir {p} is outside this session's folder {root}. Use '.' or a folder inside it."
    return p


class CadWorkdirMiddleware(AgentMiddleware):
    def _root(self) -> Path | str:
        from backend.action_policy.runtime import current_scope
        scope = current_scope()
        if scope is None:
            return "no session scope; the CAD tools need a session with a mapped folder."
        from backend.session_manager import _read_workspace_meta, _session_files_dir
        ws = _read_workspace_meta(scope.session_id)
        if ws is not None and ws.host_path:
            return Path(ws.host_path).expanduser().resolve()
        files = _session_files_dir(scope.session_id)
        files.mkdir(parents=True, exist_ok=True)
        return files.resolve()

    def _prepare(self, request: ToolCallRequest) -> ToolCallRequest | ToolMessage:
        name = request.tool_call.get("name")
        from backend.action_policy.runtime import current_scope
        scope = current_scope()
        if scope is not None and name in blocked_tools(scope.agent_name or ""):
            if name == "task":
                return _blocked(request, "Blocked: subagents are off for this run. Do the work yourself.")
            return _blocked(request, f"Blocked: `{name}` is off for this agent. Use `cad_run` to run code, "
                                     "`view_image` to look at images, `read_file`/`write_file`/`edit_file` for files.")
        if name == "view_image" and scope is not None and (scope.agent_name or "") == "benchcad-agent":
            cap = view_image_cap()
            seen = _VIEWS.get(scope.session_id, 0)
            if seen >= cap:
                return _blocked(request, f"Blocked: you have looked at {seen} images, the limit for this run is {cap}. "
                                         "Use `cad_compare` (it draws the difference for you) and `assembly_status` "
                                         "to decide what to do next.")
            _VIEWS[scope.session_id] = seen + 1
        if name == "write_file" and scope is not None and (scope.agent_name or "") == "benchcad-agent":
            import hashlib
            text = str((request.tool_call.get("args") or {}).get("content") or "")
            if len(text) > 200:     # a small model can loop, writing the same program under new names
                digest = hashlib.sha1(text.encode()).hexdigest()
                seen = _WRITES.setdefault(scope.session_id, set())
                if digest in seen:
                    return _blocked(request, "Blocked: this is exactly a file you already wrote. Do not write it again "
                                             "under another name: run it with `cad_run`, or change something first "
                                             "(use `edit_file` for small changes).")
                seen.add(digest)
        if name not in CAD_TOOLS:
            return request
        root = self._root()
        if isinstance(root, str):
            return _blocked(request, f"Blocked: {root}")
        args = dict(request.tool_call.get("args") or {})
        wd = resolve_workdir(root, args.get("workdir"))
        if isinstance(wd, str):
            return _blocked(request, f"Blocked: {wd}")
        args["workdir"] = str(wd)
        return request.override(tool_call={**request.tool_call, "args": args})

    def wrap_tool_call(self, request: ToolCallRequest,
                       handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]]):
        prepared = self._prepare(request)
        return prepared if isinstance(prepared, ToolMessage) else handler(prepared)

    async def awrap_tool_call(self, request: ToolCallRequest,
                              handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]]):
        prepared = self._prepare(request)
        return prepared if isinstance(prepared, ToolMessage) else await handler(prepared)
