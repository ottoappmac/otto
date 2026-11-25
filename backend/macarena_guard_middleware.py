"""Keep the MacArena benchmark agents to the tools that act on the benchmark VM.

Otto gives every agent file and shell tools from its filesystem backend, which the per-agent tool allowlist
does not remove. On a benchmark agent those tools would act on the benchmark host, where MacArena's task files
and evaluator scripts live, so the agent could read the answer. This middleware refuses them. It is an
allowlist, so a tool added later is refused until listed here.

* ``macarena-agent`` (standard track): only the ``vm_*`` tools.
* ``macarena-desktop-agent`` (Otto-native track): Otto's macOS desktop and AppleScript tools plus ``vm_done`` /
  ``vm_fail``, and a desktop or AppleScript tool only when it is the remote variant (``backend/macarena_native.py``)
  that runs in the VM. Anything else would act on the host's own desktop and apps, so it is refused, not run.

Every other agent is untouched.

``macarena-desktop-agent`` also keeps ``write_todos`` (Otto's planning list): it acts on nothing.

Refusing a call is the backstop. The model is also never offered the tools it may not use: on each model request
this middleware drops them, and the system-prompt sections that describe them, so the agent does not spend turns
on calls that can only be refused (a benchmark run showed ``execute``, ``ls``, ``write_file`` and ``glob`` tried
and refused dozens of times).

Two more things this middleware does to the model request, for the benchmark agents only:

* It drops the sections the app's session builder appends for a chat session on this computer (``Session
  Context`` with this host's session folder, ``/output/`` and subagent file rules, scheduling offers, mapped
  projects, "when the user says a fix did not work"). None of them applies in the VM, and the host path misled
  the agent: runs showed it typing this host's home folder into the VM, where the user is someone else.
* Once ``vm_done`` / ``vm_fail`` has been accepted, the model is offered no tools, so it gives its final answer
  instead of spending turns on calls the bridge can only refuse ("the run is over").
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from backend import macarena_native

AGENT = "macarena-agent"
NATIVE_AGENT = macarena_native.AGENT

END_TOOLS = frozenset({"vm_done", "vm_fail"})

VM_TOOLS = frozenset({
    "vm_screenshot", "vm_accessibility_tree", "vm_click", "vm_double_click", "vm_right_click",
    "vm_move_to", "vm_drag", "vm_scroll", "vm_type", "vm_press", "vm_hotkey", "vm_wait",
    "vm_done", "vm_fail",
})


# The native agent keeps Otto's planning tool: it only updates the agent's own todo list (no file, shell or
# desktop effect, on the host or the VM), so it is like thinking out loud and uses no benchmark step.
PLANNING_TOOLS = frozenset({"write_todos"})

# System-prompt sections deepagents adds for its tools (each is one text block), by the tool they describe.
_PROMPT_HEADINGS = {
    "write_todos": ("## `write_todos`",),
    "task": ("## `task`",),
    "files": ("## Following Conventions", "## Filesystem Tools", "## Execute Tool"),
}


# Sections of the app's session prompt (``backend/session_manager.py``, ``_build_graph``) that describe this host
# or a chat with a person. A section runs from its ``## `` heading to the next ``#``/``##`` heading.
HOST_SECTIONS = (
    "## Session Context",
    "## Ambient Scheduling",
    "## Parallel Subagents & Output Files",
    "## When the user says a fix did not work",
    "## Mapped project",
)

# What the finish tools answer when the bridge accepted them (backend/builtin_mcps/macarena_vm/server.py).
FINISHED_REPLIES = ("Marked done.", "Marked as infeasible.")


def drop_sections(text: str, headings: tuple[str, ...] = HOST_SECTIONS) -> str:
    """``text`` without the markdown sections that start with one of ``headings`` (heading line included, up to
    the next level-1 or level-2 heading)."""
    out: list[str] = []
    skipping = False
    for line in text.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("## ") or stripped.startswith("# "):
            skipping = stripped.startswith(headings)
        if not skipping:
            out.append(line)
    return "\n".join(out)


def run_finished(messages: Any) -> bool:
    """True once a finish tool (``vm_done`` / ``vm_fail``) has been accepted in this conversation."""
    for m in messages or ():
        if getattr(m, "type", None) != "tool" or getattr(m, "name", None) not in macarena_native.FINISH_TOOLS:
            continue
        content = m.content if isinstance(m.content, str) else " ".join(
            str(b.get("text", "")) if isinstance(b, dict) else str(b) for b in (m.content or []))
        if content.lstrip().startswith(FINISHED_REPLIES):
            return True
    return False


def allowed_tools(agent: str | None) -> frozenset[str] | None:
    """The tool names ``agent`` is offered, or None when this middleware leaves its tools alone."""
    if agent == AGENT:
        return VM_TOOLS
    if agent == NATIVE_AGENT:
        return macarena_native.AGENT_TOOLS | PLANNING_TOOLS
    return None


def _tool_name(tool: Any) -> str | None:
    return tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", None)


def strip_hidden_prompt(message: Any, allowed: frozenset[str] = frozenset()) -> Any:
    """``message`` without the text blocks that describe tools not in ``allowed`` (files, shell, subagents,
    and planning unless ``write_todos`` is allowed), and without the app's host-specific sections
    (``HOST_SECTIONS``)."""
    if message is None:
        return None
    headings = tuple(h for key, hs in _PROMPT_HEADINGS.items() if key not in allowed for h in hs)
    from langchain_core.messages import SystemMessage
    blocks = list(message.content_blocks)
    kept, changed = [], False
    for b in blocks:
        if b.get("type") != "text":
            kept.append(b)
            continue
        text = str(b.get("text", ""))
        if text.lstrip().startswith(headings):
            changed = True
            continue
        trimmed = drop_sections(text)
        if trimmed != text:
            changed = True
            b = {**b, "text": trimmed}
        kept.append(b)
    return SystemMessage(content_blocks=kept) if changed else message


def refusal(agent: str | None, tool: str | None, tool_obj: Any = None) -> str | None:
    """The error text when ``agent`` may not call ``tool``, else None."""
    if agent == AGENT:
        if tool in VM_TOOLS:
            return None
        return (f"Blocked: `{tool}` is not available. You can only use the vm_* tools, which act on the VM's "
                "screen, mouse and keyboard.")
    if agent == NATIVE_AGENT:
        if tool in macarena_native.FINISH_TOOLS or tool in PLANNING_TOOLS:
            return None
        if tool in macarena_native.MEMORY_TOOL_NAMES:
            meta = getattr(tool_obj, "metadata", None) or {}
            if meta.get("procedure_memory") is True:
                return None
            return f"Blocked: `{tool}` is not the procedure memory tool. Nothing was run."
        if tool in macarena_native.REMOTE_TOOL_NAMES:
            meta = getattr(tool_obj, "metadata", None) or {}
            if macarena_native.ACTIVE and meta.get("macarena_remote") is True:
                return None
            return (f"Blocked: `{tool}` is not connected to the benchmark machine, and it must never act on "
                    "this computer. Nothing was run.")
        return (f"Blocked: `{tool}` is not available here; nothing was run. For a command on the test Mac use "
                "`run_osascript` with `do shell script`; otherwise use the macOS desktop tools, then `vm_done`.")
    return None


# ── Checks on the native agent's own way of working (benchmark only) ─────────────────────────────────────────
# Both answer in this middleware, before the bridge sees the call, so they use no benchmark step, and each fires at
# most once per task: the agent decides, and the same call made again runs.

# A shell command (inside ``do shell script``) that makes or changes files, settings or software, as opposed to one
# that only looks. Runs made the task's result this way instead of in the app the task names (a hand-written
# .webloc, an image re-encoded with Pillow, gimprc edited by hand, ``pip install`` and the user's code rewritten).
_SHELL_WRITES = (
    (r"(?<![0-9&>])>{1,2}(?!\s*/dev/null)(?!&)", "a redirect into a file"),
    (r"\btee\b", "tee"),
    (r"\b(?:cp|mv|ditto|ln|mkdir|touch|unzip)\s", "a file command"),
    (r"\bsed\s+(?:-[a-zA-Z]*\s+)*-i", "sed -i"),
    (r"\b(?:pip3?|brew|npm|conda|gem)\s+install\b", "an install"),
    (r"\bpython[0-9.]*\s+(?:-c\b|-\s|<<|\S+\.py\b)", "a Python script"),
    (r"\bdefaults\s+write\b", "defaults write"),
    (r"\bplutil\s+-(?:replace|insert|remove|convert)\b", "plutil"),
    (r"\bsips\b|\bmagick\b|\bffmpeg\b", "a media converter"),
    (r"--convert-to\b|\bgimp[\w-]*\s+(?:.*\s)?-[bi]\b", "an app run from the command line"),
)
SHELL_WRITE_NOTE = "[check: shell change]"
DONE_CHECK_NOTE = "[check: before vm_done]"
# Skip the vm_done check when this few steps are left, so a real finish is never pushed past the budget.
DONE_CHECK_MIN_STEPS = 4


def shell_write(tool: str | None, args: dict | None) -> str | None:
    """What kind of change a ``run_osascript`` call makes with the shell, or None when it only looks (or is not a
    shell call)."""
    import re
    if tool != "run_osascript":
        return None
    script = str((args or {}).get("script") or "")
    command = str((args or {}).get("shell_command") or "")
    if command.strip():
        script = f"do shell script {command}"        # the same command, given unquoted
    if "do shell script" not in script:
        return None
    for pattern, what in _SHELL_WRITES:
        if re.search(pattern, script):
            return what
    return None


def _texts(messages: Any) -> list[tuple[str, str]]:
    out = []
    for m in messages or ():
        content = getattr(m, "content", "")
        text = content if isinstance(content, str) else " ".join(
            str(b.get("text", "")) if isinstance(b, dict) else str(b) for b in (content or []))
        out.append((str(getattr(m, "type", "")), text))
    return out


def already_checked(messages: Any, note: str) -> bool:
    return any(kind == "tool" and note in text for kind, text in _texts(messages))


def steps_left(messages: Any) -> int | None:
    """Steps left as the last budget note said (``[benchmark] steps used N, steps left M``); None when unknown or
    there is no step limit."""
    import re
    for kind, text in reversed(_texts(messages)):
        if kind != "tool":
            continue
        m = re.search(r"\[benchmark\] steps used \d+, steps left (\d+)", text)
        if m:
            return int(m.group(1))
        if "[benchmark] steps used" in text:
            return None
    return None


def has_budget_room(messages: Any) -> bool:
    """Did the last budget note leave at least ``DONE_CHECK_MIN_STEPS`` steps (or say there is no limit)?"""
    for kind, text in reversed(_texts(messages)):
        if kind == "tool" and "[benchmark] steps used" in text:
            left = steps_left([SimpleNamespace(type="tool", content=text)])
            return left is None or left >= DONE_CHECK_MIN_STEPS
    return False


def shell_write_check(what: str) -> str:
    return (f"{SHELL_WRITE_NOTE} Not run yet: this command makes or changes something with the shell ({what}). "
            "The Mac's final state is judged as if a person had used the app the task names, and a file or setting "
            "made around the app is often not what that app would make. If the task names an app, do it in that "
            "app's window or dialogs. If the shell really is the way here (a terminal or file task), send the same "
            "call again and it will run. If the app cannot do what is asked at all, that may be a task that cannot "
            "be done: then call vm_fail with the reason. This check used no step and will not be asked again.")


def done_check() -> str:
    return (f"{DONE_CHECK_NOTE} Not recorded yet; one check before the run ends (it used no step and is asked once). "
            "Call vm_done again only if all of these hold: (1) after your last change you read the end state back "
            "from the app itself (the value, the open tab, the saved file), not from what a tool said it typed; "
            "(2) the app the task names made the result, with no file made by a shell script instead, no weaker "
            "version of what was asked, and no changes to the task's own input files beyond what it asked; (3) you "
            "added no names, titles or extra items the task did not ask for. If something is not done, finish it. "
            "If the request cannot be done as asked (the app has no such feature, or it contradicts itself), call "
            "vm_fail with the reason.")


def native_check(name: str | None, args: dict | None, messages: Any) -> str | None:
    """The once-per-task check text for this native-agent call, or None to let it run."""
    if name == "vm_done":
        # Only with room to spare, as the last budget note says. With no note (the bridge could not be reached) the
        # call goes straight on, so the bridge's own answer (a budget refusal, say) is never pre-empted.
        if not has_budget_room(messages):
            return None
        return None if already_checked(messages, DONE_CHECK_NOTE) else done_check()
    what = shell_write(name, args)
    if what and not already_checked(messages, SHELL_WRITE_NOTE):
        return shell_write_check(what)
    return None


def turn_id(request: Any) -> str | None:
    """Id of the model response that issued this tool call: the bridge's ``--step-unit turns`` counts one step per
    distinct id, so the parallel calls of one response share a step. None when it cannot be told (the bridge then
    counts the call as an action)."""
    call_id = (getattr(request, "tool_call", None) or {}).get("id")
    messages = (getattr(request, "state", None) or {}).get("messages") or []
    for m in reversed(messages):
        calls = getattr(m, "tool_calls", None) or []
        if calls and any(c.get("id") == call_id for c in calls):
            return str(getattr(m, "id", None) or calls[0].get("id") or "") or None
    return None


class MacarenaGuardMiddleware(AgentMiddleware):
    def __init__(self, agent_name: str | None = None):
        super().__init__()
        self.agent_name = agent_name

    def _offer(self, request: ModelRequest) -> ModelRequest:
        allowed = allowed_tools(self.agent_name)
        if allowed is None:
            return request
        tools = [] if run_finished(request.messages) else [t for t in request.tools if _tool_name(t) in allowed]
        return request.override(tools=tools, system_message=strip_hidden_prompt(request.system_message, allowed))

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]):
        if self._is_native():
            macarena_native.tick("model")
        return handler(self._offer(request))

    async def awrap_model_call(self, request: ModelRequest,
                               handler: Callable[[ModelRequest], Awaitable[ModelResponse]]):
        if self._is_native():
            await macarena_native.atick("model")
        return await handler(self._offer(request))

    def _check(self, request: ToolCallRequest) -> ToolMessage | None:
        from backend.action_policy.runtime import current_scope
        scope = current_scope()
        name = request.tool_call.get("name")
        msg = refusal(getattr(scope, "agent_name", None) if scope is not None else None, name,
                      getattr(request, "tool", None))
        if msg is None:
            return None
        return ToolMessage(content=msg, name=name or "tool",
                           tool_call_id=request.tool_call.get("id") or "", status="error")

    def _is_native(self) -> bool:
        """Is the agent running the Otto-native benchmark agent? Every other agent skips the benchmark bookkeeping."""
        if self.agent_name == NATIVE_AGENT:
            return True
        from backend.action_policy.runtime import current_scope
        scope = current_scope()
        return (getattr(scope, "agent_name", None) if scope is not None else None) == NATIVE_AGENT

    @staticmethod
    def _end_refused(request: ToolCallRequest, res: dict | None) -> ToolMessage | None:
        """The bridge refused vm_done / vm_fail (or a memory lookup) past the budget: answer it here, do not run it."""
        name = request.tool_call.get("name")
        if (name not in END_TOOLS and name not in macarena_native.MEMORY_TOOL_NAMES) or not res or res.get("ok", True):
            return None
        return ToolMessage(content=str(res.get("error") or "no turns left"), name=name,
                           tool_call_id=request.tool_call.get("id") or "", status="error")

    def _native_check(self, request: ToolCallRequest) -> ToolMessage | None:
        """A once-per-task check (``native_check``), answered here so the bridge counts no step for it."""
        call = request.tool_call or {}
        messages = (getattr(request, "state", None) or {}).get("messages") or []
        text = native_check(call.get("name"), call.get("args") or {}, messages)
        if text is None:
            return None
        return ToolMessage(content=text, name=call.get("name") or "tool", tool_call_id=call.get("id") or "",
                           status="error")

    def wrap_tool_call(self, request: ToolCallRequest,
                       handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]]):
        if self._is_native():
            if (checked := self._native_check(request)) is not None:
                return checked
            turn = turn_id(request)
            macarena_native.TURN.set(turn)
            refused = self._end_refused(request, macarena_native.tick("tool", request.tool_call.get("name"), turn))
            if refused is not None:
                return refused
        return self._check(request) or handler(request)

    async def awrap_tool_call(self, request: ToolCallRequest,
                              handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]]):
        if self._is_native():
            if (checked := self._native_check(request)) is not None:
                return checked
            turn = turn_id(request)
            macarena_native.TURN.set(turn)
            res = await macarena_native.atick("tool", request.tool_call.get("name"), turn)
            refused = self._end_refused(request, res)
            if refused is not None:
                return refused
        return self._check(request) or await handler(request)
